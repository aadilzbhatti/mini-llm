# Handoff: wiki-llm (mini-llm), Mac mini — 2026-10-04

## Start here

**What this project is.** mini-llm is a small, hand-written decoder-only Transformer (from the original wiki-llm project) trained on Wikipedia-style text. Around it is a setup for running experiments cheaply and repeatably: training locally on MPS or on Modal GPUs, a phone control page, automatic import and evaluation of every finished run, and one shared validation set so every number is comparable. A separate autolab daemon proposes and runs its own experiments on its own branch.

**What we're trying to do.**
1. **Find what actually makes a small model better, one variable at a time.** Every experiment changes a single thing (batch, data size, context length, width, steps, LR, and next position encoding or attention implementation) against a fixed baseline, at a fixed token budget where possible, scored on the same val set (`full_val_loss`).
2. **Judge models on generation, not just loss.** Lower loss has stopped meaning better text among the 39M models (the 80K model is best on loss but no better at generation). Every model gets the same eval suite: loss, how loss improves with more context, retrieval, and 100 frozen generation samples. A change counts as a win only if it shows up there, beyond the noise.
3. **Keep it cheap and reproducible.** Modal GPUs are billed, so anything that can run here (evals, samples, benchmarks) runs here. Prompts, seeds and decoding settings are frozen so results stay comparable over time. Record results in the repo (README, baselines.md, evals/) as they come in.
4. **Near-term:** make inference fast enough to iterate on (fused attention), then test whether RoPE helps the model use context, especially past its training window (evals/kv_reference.md is the baseline to beat).

**How to read the rest of this file.**
- **Where things are / How the system works** is reference: paths, services, commands. Skim it now and come back when you need a command. When it disagrees with the repo, trust the repo (and fix this file).
- **Results so far** is the current state of knowledge. Read it before proposing an experiment, so you don't repeat one or contradict a settled decision. Each result names its checkpoint or eval file; open those for details.
- **Open next steps** is the work queue, roughly in priority order. Confirm with the user before starting a training run (it costs money) or anything large.
- **Gotchas** are mistakes that already cost time. Read them all before touching launchd, the GPU, or plists.

**First moves in a new session:** check that the six launchd agents are running (`launchctl list | grep -E 'mini-llm|autolab'`), that `main` is clean and up to date, and that no GPU job is running (`ps aux | grep mini-llm`). Then ask the user which next step to take.

## Where things are
- This machine (Mac mini, M4 16 GB, user aadil) runs everything. The MacBook Air's services were stopped and disabled on 2026-10-04.
- `~/dev/wiki-llm` = branch `main` (github.com/aadilzbhatti/mini-llm). Work and push directly on main.
- `~/dev/wiki-llm-autolab` = branch `autolab`, a separate clone. Autolab's own state (autolab/data, autolab/runs, autolab/state) is gitignored. Its tracked notebook files (NOTEBOOK.md, notebook.jsonl, research/cards.jsonl, …) have uncommitted changes on purpose: the daemon writes them. Don't revert them.
- launchd agents (~/Library/LaunchAgents): com.aadil.mini-llm-{control,modal-mirror,runner,tensorboard}, com.aadil.autolab-{daemon,dashboard}. All six were running on 2026-10-04.
  Restart one: `launchctl kickstart -k gui/$(id -u)/com.aadil.<name>`. After editing a plist: `launchctl bootout` then `launchctl bootstrap`.
- Logs: ~/dev/wiki-llm/runner/*.log (control.log, modal-mirror.log, runner log), ~/dev/wiki-llm-autolab/autolab/state/*.log.
- Pages via tailscale serve: https://aadils-mac-mini.taile67486.ts.net (`/` control on :8765, `/autolab` on :8766, `:8443` TensorBoard on :6006).
- Untracked state lives only here: data/ (data20k…data640k), checkpoints/*.pt, runs/ (status files and imported Modal runs), queue/. Modal credentials are in ~/.modal.toml.
- GitHub works: `gh` is logged in as aadilzbhatti (keyring token, `repo` scope), and pushes from both clones authenticate. Autolab pushes `autolab-accepted` from a working copy it creates at accept time; that branch doesn't exist locally in the autolab clone, so a manual push of it fails with "src refspec does not match". That error is expected.

## How the system works
- Training runs on Modal (2×L4). Configs are in configs/modal/*.json. Launch through the control API (POST /api/jobs with target "modal", or the page). Do NOT launch with `modal run --detach` from a shell you might kill: killing that client killed a run once. (README "Current best" still shows that command; prefer the API.)
- The mirror (modal-mirror agent) syncs Modal runs into runs/, imports finished checkpoints, and auto-evaluates them on this GPU. It retries Modal connection errors instead of crashing. Runs trained without --save aren't imported.
- Sample reports are never generated on Modal (they'd hold billed GPUs). The eval writes them here.
- One evaluation task per model: `uv run mini-llm-eval checkpoints/<m>.pt` runs quality, context curve, context benefit, retrieval and 100 generation samples, then refreshes evals/summary.md, evals/samples.md and evals/inference.md (the benchmark, skipped if another GPU job is running). Also `--all` and `--only samples`.
  - Generation samples: 20 frozen prompts × 5 seeds, T=0.7, top-k 40, 256 tokens. Prompts are append-only (seeds are per index). Metrics: rep4, distinct-2/4, loop rate/onset, topic held/span.
  - What each eval means: evals/GUIDE.md (linked from the Best tab).
- Decoding sweep: `uv run mini-llm-samples checkpoints/<ckpt>.pt --out evals/sweep_<name>.md` (temperature × top-k plus greedy, 20 prompts once each). Without `--out` it overwrites evals/sweep.md, so always name the output.
- KV cache: per-head and preallocated. No config flag: generate() and report.generate_until_eos always use it, and model(x, use_cache=True) opts a single forward in. A plain forward is stateless, so loss evals and training are unaffected.
  - evals/kv_reference.md is the fixed "absolute PE + KV cache" baseline at T=128–1024. Regenerate with `uv run mini-llm-bench --kv-reference`. Its "past the window" row (0.9–1.0×) is what RoPE should improve.
- Tests: `uv run pytest -q` (~140 tests, ~6 min). Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

## Results so far (same val set, 327.68M tokens unless noted; full_val)
- 16M d256-L4: T128 4.2601, T256 4.1774, T512 4.1550, T1024 4.1141.
- 39M d512-L4 T1024 (LR sweep at 10K steps picked 6e-4): data160k 40K 3.8919; data320k 40K 3.9376; data320k 80K (655M tokens) 3.7476 (best).
- Generation (100 samples/model, loop rate at T=0.7 k=40): 16M T1024 26%, 39M data160k 17%, 39M data320k 40K 17%, 39M 80K 26%. Width clearly helps generation; doubling data at a fixed budget made no difference.
- Decoding sweep (done 2026-10-04, commit aa3e5db; evals/sweep_39m_data320k_80k.md, evals/sweep_39m_data160k_40k.md): the over-sharpening hypothesis is NOT supported. The 80K and data160k 40K models behave almost the same at every setting (loops out of 20 — greedy 19 vs 19; T=0.6 k=40 6 vs 8; T=0.7 k=40 1 vs 3; T=0.8 k=40 0 vs 1). The 26% vs 17% gap is within noise (95% CIs overlap). Decision: keep the eval frozen at T=0.7 k=40 for comparability; use T=0.8 k=40 for interactive generation. Among the 39M models, loss gains aren't showing up as better generation.
- The README "Current best" section is stale: it still describes the 16M T1024 model (4.1141). It should be rewritten around the 39M 80K model (3.7476). Not done yet.

## Open next steps
1. Fused attention for the KV cache (one QKV projection + SDPA). It was deferred as a larger change. A prototype ran ~3.5× faster with identical outputs.
2. RoPE experiment: identical config except position scheme. Compare against kv_reference and the evals.
3. data640k was added by the user (see data/data640k/MANIFEST.md); not yet used for training.
4. Rewrite README "Current best" for the 39M 80K model.
5. Autolab's research step logged `Can't reach the API server (ENOTFOUND)` around the machine migration. Check that later cycles in autolab/state/research.log succeed.

## Gotchas
- launchd agents doing GPU/CPU work must be `ProcessType Standard`: `Background` throttled MPS ~7×. Edit plists with a text edit, never PlistBuddy (it rewrites the file and drops comments).
- Don't run two GPU-heavy jobs at once on this 16 GB machine: evals batch by tokens (8,192/forward) to stay in memory, and concurrent jobs thrash. Check `ps` for mini-llm-eval/-samples/-bench before starting one.
- Open phone pages reload themselves when index.html changes (X-UI-Version header).
- The server logs the reason for every rejected request as `[api] …` in control.log.
- `ls` in the user's shell is aliased (eza); use `command ls` in scripts.
