# Roadmap: from mini-llm to an inference engine, and an agent platform on top

Written 2026-11-04. The project's goal changed on 2026-10-28: no longer "make this model produce good
language", but **learn inference serving by building it**: a vLLM / llama.cpp-style engine that can
serve any open model, then an agentic platform on top of it. The end state: good enough to contribute
to vLLM (or SGLang, llama.cpp, TGI) and to design serving systems, not just use them.

The training code (`src/mini_llm/`) is the reference implementation, the source of the golden outputs,
and the place the engine started from. Improving the model itself is optional side work (Part 4).

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
| **2b. Logits processors** | Nothing moves: new sampler features. Repetition, frequency and presence penalties, no-repeat n-gram, min-p, all off by default (details below) | Defaults still match golden; each processor matches hand-computed logits; a decoding sweep on the 95M model with a chosen default | written when you start it |
| **3. Static batching** | The single shared `pos`: per-sequence positions and lengths, a random generator per request | A batch of 8-token and 1000-token prompts gives each prompt's solo output | `test_engine_step3.py` (written) |
| **4. A model interface and loader** | `build_model(ModelConfig)` and our checkpoint format: a model registry, an adapter for mini_llm checkpoints, the GPT-2 tokenizer as the model's tokenizer | `Engine.from_pretrained(path)` for both golden models | written when you start it |
| **5. An engine loop and a server** | `/api/generate`'s one-at-a-time lock: a request queue, a step loop, streaming; the control page's Generate tab talks to the engine | Concurrent requests stream correctly; an OpenAI-compatible `/v1/completions` | written when you start it |

Expect 4–6 weeks at ~10 hours a week. Steps 1–3 teach the core data flow of every serving engine;
after step 5 you have a small, correct, slow engine, which is exactly what the next part speeds up.

### Step 2b: logits processors

A logits processor changes the next-token scores before sampling, using the tokens the sequence has
already seen. Every serving engine has them (vLLM's `SamplingParams`, llama.cpp's sampler chain, Hugging
Face's `LogitsProcessor`), and they are the cheapest fix for the 95M model's looping: 33/100 frozen
samples degenerate today (25–43% CI), mostly in lists, instructions and definitions.

**What to build**, as new `SamplingParams` fields, each off by default:

| Processor | What it does | Usual range |
|---|---|---|
| `repetition_penalty` (CTRL; HF and vLLM name) | For every token already in the sequence: positive logits divided by the penalty, negative ones multiplied | 1.1–1.3 |
| `frequency_penalty`, `presence_penalty` (OpenAI) | logit −= α·count + β·[count > 0]: the more often a token has appeared, the harder it is pushed down | α 0.2–0.5, β 0–0.5 |
| `no_repeat_ngram_size` | Bans any token that would complete an n-gram already in the sequence: exact loops become impossible | 3–4 |
| `min_p` | Keeps tokens with probability ≥ min_p × p_max: trims the junk tail without top-k's fixed cutoff | 0.05–0.1 |

**Design questions to settle (the real learning):**
- **Per-sequence state.** The penalties need each sequence's token counts. Keep a `[batch, vocab]`
  count tensor updated each step, or rebuild it from token ids with a bincount (vLLM keeps output token
  ids per request and builds the penalty tensor per batch). This shapes step 3 (batching) and A2
  (structured output), which hooks into the same place.
- **Which tokens count.** Prompt and output, or output only (OpenAI's penalties count output only; HF's
  `repetition_penalty` counts both)? Should punctuation and very common tokens be exempt? Lists need
  `- ` and digits to repeat, and biographies repeat names.
- **Order.** Penalties → temperature → top-k / top-p / min-p → sample. Write the order down; changing it
  changes outputs.
- **Vectorized over the batch,** each request with its own settings, with no Python loop over the vocab.

**Tests (Claude writes):** with every processor off, step 2's golden outputs still match token for
token; each processor matches a hand-computed result on small made-up logits; no-repeat n-gram never
emits a repeated n-gram over long greedy runs.

**The sweep (Claude writes the harness, runs locally, no Modal cost):** the frozen 100-sample eval on
the 95M model over about 8–12 settings (each processor alone, then the best combinations), at roughly
3–5 minutes per setting. Report degenerate, rep4, distinct-n and topic held, plus two guards. Penalties
lower the repetition metrics almost by definition, so better metrics alone don't show better text:
- the perplexity of the generated text under a stronger model (e.g. GPT-2 medium, or the 95M model
  itself at T=1);
- reading 10–20 samples per setting, especially the list prompts.

*Expected:* exact loops close to 0 with no-repeat n-gram, and degenerate maybe 5–15/100 with a mild
penalty. That's a guess to check, not a promise. Coherence and staying on topic come from the weights
and shouldn't change much. *Done when:* a recommended default for interactive generation (the frozen
eval stays at T=0.7, top-k 40, no penalties, for comparability with every earlier model), recorded with
its sweep in `evals/`.

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
| **A2. Valid structured output** | JSON-schema / grammar-constrained decoding: a token mask from a state machine at each step, so tool arguments always parse | Step 2b's per-sequence processor state, phase B sampler hooks; F (structured output) | 100% of tool calls parse on a function-calling test set, with the cost per token measured |
| **A3. Agent runtime** | The loop itself: tool registry with schemas, sandboxed tools (code execution in Modal sandboxes), timeouts, retries, budgets (tokens, dollars, wall time), human approval for risky tools, session state | A1–A2 | A multi-step task (search → read → compute → answer) finishes within its budget, and its trace can be replayed |
| **A4. Sessions and prefix reuse** | Agents resend the whole history each turn: prefix caching, session affinity, keeping (or offloading to CPU) a session's KV cache between turns, the cost of a turn made proportional to what's new | Phase C (prefix cache) | On a recorded agent trace: cache hit rate, time to first token per turn, vs. no caching |
| **A5. Agent-shaped scheduling** | Priorities (interactive before background), fair share across agents, fan-out (parallel branches sharing a prefix, copy-on-write) and cancelling the losers, speculative decoding for latency | Phases B, C, F | p99 latency for interactive steps holds while background agents saturate the GPU |
| **A6. Multi-tenant platform** | The Rust gateway: auth, quotas, rate limits, routing by model and by cached prefix across engine replicas, per-agent LoRA adapters, cost per agent and per task (from our cost tracking) | Phases E, F (LoRA) | Two replicas behind the gateway, prefix-aware routing measurably raising hit rates |
| **A7. Observability and evals** | Traces of every step (prompt, tool call, tokens, cache hits, latency, cost) on the control page; an agent eval suite: function-calling accuracy, multi-step task success, success vs. cost vs. latency; replay of recorded traces for regressions | From A3 onward, growing with each milestone | Every platform change reports its effect on success rate, cost and latency |

Order: A1 and A3 can start right after engine step 5 (against a reference server if your engine lacks
a feature); A2 with phase B; A4 after phase C; A5–A6 after phases E–F. A7 grows alongside everything.

## Part 4: a better mini-llm (optional, alongside the engine)

The engine doesn't need a better mini-llm. This part is for when you want the model's own text to
improve. Change the golden models only deliberately: a new checkpoint is a new model to serve, not a
replacement for `tests/golden/engine_golden.json`.

**Where it stands (2026-11-07, `evals/summary.md`).** Best: 95M (d768-L8, T1024, RoPE, fused attention),
160K steps on data640k, full_val 3.3310; 33/100 degenerate samples, topic held 46%. It was trained on
1.31B tokens: about 14 tokens per parameter (Chinchilla-optimal is about 20), and about 2 epochs of
data640k's ~632M training tokens.

**What the evals say:**
- **Model size is the only thing that has clearly improved generation.** 16M → 39M → 95M took
  degenerate from 46–59/100 to 33/100 and topic held from about 30% to 46%.
- **More steps lowered loss but didn't help generation.** The 95M +30K continuation: 3.3239, 36/100
  degenerate. The same happened with the 39M 80K model.
- **The data mix didn't clearly help.** 70/30 FineWeb / FineWeb-Edu: 28/100 degenerate, within the CI.

### M1. Better decoding (= engine step 2b)

No training. Done by step 2b's sweep above. It fixes looping, not knowledge or coherence.

### M2. Training throughput (a few days, small Modal cost)

The 95M run reached only 61K tokens/s on an H100: batch 8 × 1024 tokens (8K tokens per step), fp32
(`--bf16` exists but wasn't passed), no `torch.compile`, no gradient accumulation, no gradient
clipping. An H100 is badly underused at that size. In order, each measured on a short proxy run
(tokens/s, peak memory, loss at a fixed token count, against the current 95M loss curve):
1. `--bf16` (already implemented; just turn it on).
2. Bigger batches: gradient accumulation and/or a larger micro-batch, toward about 0.5M tokens per
   step (GPT-2 / nanoGPT scale), with the learning rate re-swept for the new batch size.
3. `torch.compile` on the training model.
4. Gradient clipping at 1.0 (standard at this scale, and protection against bigger-batch, higher-LR
   instability).
5. Optional: a fused AdamW (`fused=True`), TF32 matmuls for anything left in fp32.

*Done when:* the 95M config trains at a measured multiple of 61K tokens/s (3–5× is plausible) and
matches the old run's loss at equal tokens. This is also a preview of phase D: the same profiling and
roofline reasoning, on training instead of inference.

### M3. One scale step (1–2 weeks of runs, the main cost)

A GPT-2-small-sized model (about 125–150M parameters, e.g. d768-L12) on 5–10B tokens of FineWeb /
FineWeb-Edu. That's the size and data where models typically write several coherent paragraphs and
loop much less.
- **Data first.** 5–10B tokens is 8–16× data640k. Build it on Modal (`modal_prepare.py`,
  `prepare-data --mix`), not on the Mac: the 16 GB machine already ran out of memory building smaller
  sets. Keep `val.pt` as data20k's, byte for byte, so `full_val_loss` stays comparable.
- **Proxy runs first,** as before: a short LR sweep at the new size and batch, then one full run.
- **Cost.** At today's 61K tokens/s, 10B tokens is about 45 H100-hours (~$180). After M2, maybe 10–15
  hours (~$40–60). Check the estimate on the control page before launching.
- *Done when:* the full eval suite on the new model, against the 95M row, with M1's decoding default
  reported alongside the frozen settings. A win counts only beyond the noise (degenerate CI, topic
  held), not on loss alone.

### M4. Instruction tuning (later, only if you want a chat model)

Supervised fine-tuning on a small instruction dataset, with a chat template. It teaches format (answer
the question, stop at the end), not knowledge, and a 100–150M model still won't drive agents (Part 3
uses open instruct models for that). Worth doing mainly to have our own chat model to test engine
features on: chat templates (A1), stop sequences, streaming.

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
| Part 1, steps 1–5 (and 2b) | 4–6, plus about 1 for 2b |
| Phases A–D | ~4–5 months |
| Phase E (Rust core) | 3–4 weeks |
| Agent track A1–A3 | in parallel from step 5 |
| A4–A7, phase F, contributions | ongoing |
| Part 4 (mini-llm): M2–M3 | optional, about 2–3 weeks of mostly unattended runs, alongside the engine |

Phases B–D are where you become "dangerous"; contributing to vLLM can realistically start during D.

## Open decisions

- Hours per week (sizes the milestones).
- Whether the engine eventually moves to its own repository (it can be split out with its history).
- Which agent eval tasks matter most to you (coding, research, tool use), to pick A7's suite.
- Whether to spend on Part 4's M2–M3 (about $40–180 of H100 time, depending on M2's speedup).
