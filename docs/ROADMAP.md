# Roadmap: from mini-llm to an inference engine, and an agent platform on top

Written 2026-11-04. The project's goal changed on 2026-10-28: no longer "make this model produce good
language", but **learn inference serving by building it**: a vLLM / llama.cpp-style engine that can
serve any open model, then an agentic platform on top of it. The end state: good enough to contribute
to vLLM (or SGLang, llama.cpp, TGI) and to design serving systems, not just use them.

The training code (`src/mini_llm/`) stays as it is: the reference implementation, the source of the
golden outputs, and the place the engine started from.

## Principles (carried over from the training work)

1. **Measure, don't assume.** Every step ends with a number (tokens/s, time to first token, memory,
   cache hit rate) and an explanation of it, not just "it works".
2. **A reference before a fast version.** Each component gets an obviously correct version first; the
   fast one must match it. Golden outputs (`engine.golden`) and parity tests are the safety net.
3. **One change at a time,** each with its own test, commit and benchmark.
4. **You write the core.** Scheduler, KV cache, attention, sampler, kernels: those are the learning.
   Claude writes scaffolding (interfaces, tests, oracles, benchmarks, infrastructure) and reviews.

## Architecture

```
            agents, apps, the control page
                       │  OpenAI-compatible API (chat, tools, streaming)
   ┌───────────────────▼────────────────────┐
   │ gateway        routing, sessions, quotas│   ← Rust (phase E/A6)
   ├─────────────────────────────────────────┤
   │ engine         the step loop            │
   │   scheduler    who runs this step       │
   │   kv_cache     blocks, prefix cache     │   ← you, in Python, then Rust (phase E)
   │   sampler      decoding, constraints    │
   │   tokenizer    + incremental detokenize │
   │   attention    reference → paged → kernels  ← Triton, then CUDA C++
   ├─────────────────────────────────────────┤
   │ models         architectures only       │   forward(input_ids, positions, kv_cache) -> logits
   │   mini_llm (ours), llama (Llama/Qwen/Mistral), …│
   └─────────────────────────────────────────┘
```

**The contract:** a model is a stateless function of `(input_ids, positions, kv_cache)`. It calls the
engine's attention layer and knows nothing about batching, paging, sampling or requests. The engine
knows nothing about any architecture's layers. That's what lets you swap an attention backend or a
scheduler without touching a model, and add a model without touching the engine.

## How to work on a step

- Code lives in `src/engine/`. Each step has a test file (`tests/test_engine_stepN.py`) written ahead of
  time against the golden outputs.
- `engine.PROGRESS` (in `src/engine/__init__.py`) is the last finished step. Later steps' tests are
  skipped in normal runs; work on one with
  `uv run pytest tests/test_engine_step1.py --engine-todo`, then bump `PROGRESS` when it passes, so it
  becomes a regression test for everything after.
- The golden outputs (`tests/golden/engine_golden.json`) were recorded on the CPU in float32 from
  mini_llm's own generation, before any decoupling: greedy and seeded sampling for the 39M RoPE and 95M
  fused models, generation past the 1024-token window, and logits. `tests/test_engine_golden.py` checks
  mini_llm still produces them. Re-record (`uv run python -m engine.golden record`) only deliberately,
  and say why in the commit.
- When a step's tests pass, ask for a review before moving on.

## Part 1: decouple the model (build the engine from what exists)

`src/engine/models/mini_llm.py` is a history-preserving copy of `mini_llm/model_fused.py` (`git log
--follow` reaches back through model.py). Today it does the engine's job in several places; each step
moves one of them out.

| Step | What moves out of the model | Done when | Tests |
|---|---|---|---|
| **1. Stateless model** | The KV cache (`k_cache`/`v_cache` in each attention module) into an engine-owned `KVCache`; positions become an input instead of `cache_len()` / `pos % block_size` | Same logits, greedy tokens and past-window tokens as the golden outputs; one model serves two caches in alternation | `test_engine_step1.py` (written) |
| **2. The engine generates** | Sampling, stopping and the rolling-window policy (`next_token_logits`, `_generate`, `report.generate_until_eos`) into `Sampler` and `Engine.generate` | Greedy, seeded sampling and past-window outputs match golden token for token | `test_engine_step2.py` (written) |
| **3. Static batching** | The single shared `pos`: per-sequence positions and lengths, a random generator per request | A batch of 8-token and 1000-token prompts gives each prompt's solo output | `test_engine_step3.py` (written) |
| **4. A model interface and loader** | `build_model(ModelConfig)` and our checkpoint format: a model registry, an adapter for mini_llm checkpoints, the GPT-2 tokenizer as the model's tokenizer | `Engine.from_pretrained(path)` for both golden models | written when you start it |
| **5. An engine loop and a server** | `/api/generate`'s one-at-a-time lock: a request queue, a step loop, streaming; the control page's Generate tab talks to the engine | Concurrent requests stream correctly; an OpenAI-compatible `/v1/completions` | written when you start it |

Expect 4–6 weeks at ~10 hours a week. Steps 1–3 teach the core data flow of every serving engine;
after step 5 you have a small, correct, slow engine, which is exactly what the next part speeds up.

## Part 2: a real engine

Each phase: concepts → your implementation → correctness against the previous version → a benchmark
with a number you can explain.

**Phase A: serve any open model (3–4 weeks).** Pairs with CS336 assignment 1.
Write the Llama family yourself: RMSNorm, RoPE with its scaling variants, grouped-query attention,
SwiGLU, tied and untied embeddings. A safetensors loader, Hugging Face config parsing, HF tokenizers
and chat templates. A **parity oracle** against Hugging Face `transformers` (fp32 logits within a
tolerance, greedy tokens identical). Dev models that fit the Mac: SmolLM2-135M/360M, Qwen2.5-0.5B,
Llama-3.2-1B. *Done when:* three architectures pass parity and stream through your server.

**Phase B: continuous batching and the API (3–4 weeks).** The heart of a serving engine.
Iteration-level scheduling (Orca), a per-step token budget, chunked prefill (Sarathi-Serve),
preemption, request cancellation; OpenAI-compatible chat and completions with streaming (SSE). A load
generator with Poisson arrivals and recorded request traces, replayable through any version of the
scheduler. *Done when:* you have throughput-vs-concurrency curves and p50/p99 time to first token and
time per output token that you can explain.

**Phase C: memory (3–4 weeks).** PagedAttention: a block allocator, block tables, attention over pages
(PyTorch gather first, a kernel later). Prefix caching: blocks shared by content hash, then a radix
tree (SGLang); copy-on-write for parallel sampling. *Done when:* N× more concurrent requests in the
same memory, and measured prefix-cache hit rates on a multi-turn trace.

**Phase D: kernels and performance, on Modal GPUs (4–6 weeks).** Pairs with CS336 assignment 2.
Triton kernels: RMSNorm, fused RoPE, paged decode attention, FlashAttention forward. CUDA graphs for
decode, `torch.compile`. Quantization: weight-only int8/int4 (AWQ/GPTQ-style), an FP8 KV cache. Then
CUDA C++ for one kernel you've already written in Triton. *Done when:* within a stated margin of vLLM on
the same model and GPU at batch 1 and batch 32, with a roofline explanation of each gap.

**Phase E: the Rust core (3–4 weeks, after C).** Profile first: show the per-step CPU overhead
(scheduling, block tables, input preparation, detokenization). Port the scheduler, block manager and
radix-tree prefix cache to Rust via PyO3/maturin, verified against the Python versions by replaying the
same request traces (same schedules, same block tables). *Done when:* the same outputs and the per-step
CPU time saved, measured. Kernels stay Triton/CUDA: Rust runs the GPU from the CPU, not on it.

**Phase F: advanced (pick 2–3).** Speculative decoding (a draft model, then n-gram, Medusa, EAGLE);
structured output (grammar-constrained decoding); many LoRA adapters at once (Punica, S-LoRA); tensor
parallelism across 2 GPUs; mixture-of-experts routing; prefill and decode on separate workers
(DistServe, Splitwise).

**Phase G: contribute (start reading in phase B).** Read vLLM's engine next to yours (scheduler, KV
cache manager, model runner); reproduce a real issue; good-first issues; benchmark contributions.
llama.cpp/ggml is a good second target and fits the M4 Mac (Metal).

## Part 3: an agentic platform on the engine

An agent is a loop: prompt → model → (tool call → tool result →)* → answer. That workload stresses a
serving engine differently from chat: many turns that resend a growing shared prefix, tool calls that
must be valid JSON, fan-out to parallel branches, interactive latency mixed with long background
jobs, and per-agent cost and quota accounting. So the platform's milestones are mostly
**requirements on the engine**, plus a runtime on top.

The models: our own 95M model can't drive agents. Agents run on open instruct models with tool-calling
support (small Qwen and Llama instruct models on the Mac, 7B+ on Modal GPUs), served by your engine
once phase A is done. Until then, the runtime can be built against any OpenAI-compatible server (a
stock vLLM or llama.cpp server) behind the same API, so the two tracks don't block each other; swap in
your engine when it's ready, and the differences become test cases.

| Milestone | What | Requires | Done when |
|---|---|---|---|
| **A1. Chat and tool calling** | Chat templates per model; tool definitions in the prompt; parsing tool calls from the output, including streamed partial calls; OpenAI `tools` / `tool_choice` | Phase A (templates), step 5 / phase B (streaming API) | An instruct model calls a tool and gets its result, through your API |
| **A2. Valid structured output** | JSON-schema / grammar-constrained decoding: a token mask from a state machine at each step, so tool arguments always parse | Phase B sampler hooks; F (structured output) | 100% of tool calls parse on a function-calling test set, with the cost per token measured |
| **A3. Agent runtime** | The loop itself: tool registry with schemas, sandboxed tools (code execution in Modal sandboxes), timeouts, retries, budgets (tokens, dollars, wall time), human approval for risky tools, session state | A1–A2 | A multi-step task (search → read → compute → answer) finishes within its budget, and its trace can be replayed |
| **A4. Sessions and prefix reuse** | Agents resend the whole history each turn: prefix caching, session affinity, keeping (or offloading to CPU) a session's KV cache between turns, the cost of a turn made proportional to what's new | Phase C (prefix cache) | On a recorded agent trace: cache hit rate, time to first token per turn, vs. no caching |
| **A5. Agent-shaped scheduling** | Priorities (interactive before background), fair share across agents, fan-out (parallel branches sharing a prefix, copy-on-write) and cancelling the losers, speculative decoding for latency | Phases B, C, F | p99 latency for interactive steps holds while background agents saturate the GPU |
| **A6. Multi-tenant platform** | The Rust gateway: auth, quotas, rate limits, routing by model and by cached prefix across engine replicas, per-agent LoRA adapters, cost per agent and per task (from our cost tracking) | Phases E, F (LoRA) | Two replicas behind the gateway, prefix-aware routing measurably raising hit rates |
| **A7. Observability and evals** | Traces of every step (prompt, tool call, tokens, cache hits, latency, cost) on the control page; an agent eval suite: function-calling accuracy, multi-step task success, success vs. cost vs. latency; replay of recorded traces for regressions | From A3 onward, growing with each milestone | Every platform change reports its effect on success rate, cost and latency |

Order: A1 and A3 can start right after engine step 5 (against a reference server if your engine lacks
a feature); A2 with phase B; A4 after phase C; A5–A6 after phases E–F. A7 grows alongside everything.

## What carries over from the training project

| Existing piece | New role |
|---|---|
| `model.py` / `model_fused.py`, the KV cache, rolling RoPE, the angle table | The first model plugin and the source of the golden outputs |
| The KV benchmark and the decode-shape diagnostic | The start of the benchmark harness (the MPS fixed-shape finding is the CUDA-graphs idea) |
| Modal launching, cost tracking, the H100 default, `modal_prepare.py` | GPU benchmark and kernel jobs, with each experiment's cost visible |
| Sweeps | Sweep concurrency, batch size, block size, chunk size |
| Run reports | Benchmark reports to compare against vLLM, shareable as md/pdf |
| Generation evals (degenerate, rep4, samples) | Catch sampler and cache regressions, which change outputs before they crash anything |
| The control page | A live engine dashboard (queue, KV-cache use, throughput, latency), then agent traces |

## Hardware and cost

- **Mac mini (M4, 16 GB, MPS):** parts 1 and phases A–C with sub-1B models. Memory is tight (the dataset
  build was killed, the test suite slows under swap): keep dev models small, close what you can.
- **Modal:** Triton and CUDA need NVIDIA GPUs. An L4 (~$0.80/h) is enough for kernel development; keep the
  H100 (~$4/h) for final benchmarks. `modal shell` or a sandbox makes kernel iteration practical.
- Every Modal run already shows its cost on the control page.

## Reading, by phase

- **Overview:** "Efficiently Scaling Transformer Inference" (Pope et al.); Horace He, "Making Deep Learning Go Brrrr".
- **B:** Orca; Sarathi-Serve. **C:** PagedAttention (the vLLM paper); SGLang (RadixAttention).
- **D:** FlashAttention 1/2/3, Flash-Decoding; AWQ, GPTQ, SmoothQuant.
- **F:** speculative decoding (Leviathan et al.; Chen et al.), Medusa, EAGLE; DistServe, Splitwise; Punica, S-LoRA.
- **Agents (A1–A5):** your target models' tool-calling formats; constrained-decoding libraries' designs
  (e.g. Outlines, XGrammar); SGLang's programming model for multi-call workloads.
- **Code to read end to end:** nano-vllm, gpt-fast, mini-sglang; then vLLM itself.
  These projects move fast: check each one's current state when you reach its phase.

## Timeline (at ~10 hours a week)

| | Weeks |
|---|---|
| Part 1, steps 1–5 | 4–6 |
| Phases A–D | ~4–5 months |
| Phase E (Rust core) | 3–4 weeks |
| Agent track A1–A3 | in parallel from step 5 |
| A4–A7, phase F, contributions | ongoing |

Phases B–D are where you become "dangerous"; contributing to vLLM can realistically start during D.

## Open decisions

- Hours per week (sizes the milestones).
- Whether the engine eventually moves to its own repository (it can be split out with its history).
- Which agent eval tasks matter most to you (coding, research, tool use), to pick A7's suite.
