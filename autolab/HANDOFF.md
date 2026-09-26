# Autolab handoff (for Claude Code on the Mac)

Read this first, then `autolab/BRIEF.md` (the spec) and `autolab/REPO_NOTES.md`
(discovery results). This file holds the decisions and standing rules that
override the brief where they conflict, plus implementation guidance per
milestone. Keep it up to date: when you make a design decision, add it to
the "Decision log" at the bottom.

## Where things stand

- Repo: `~/dev/wiki-llm-autolab`, branch `autolab`, cloned from
  `~/dev/wiki-llm` (branch `minimal`, commit d4257f0). `origin` is the GitHub
  repo, but its push URL is deliberately `DISABLED-autolab-never-pushes`.
- Milestone 1 (discovery) is done: `autolab/REPO_NOTES.md`.
- Milestones 2–6 are not started. No code in `src/autolab/` yet.
- Milestone 1 was done from a Linux VM that can't run MPS or long processes,
  so no tests have been run in this clone yet. Start with `uv sync` and
  `uv run pytest` to get a baseline of the existing suite.

## Decisions (override the brief where they conflict)

1. **Frozen val.** Copy `~/dev/wiki-llm/data/data20k/val.pt` to
   `autolab/data/val_frozen.pt`. It holds 918,728 tokens in 937 docs and
   overlaps no train file. Store its sha256 in the autolab config and check it
   before every run. Always pass it explicitly as `--val-tokens`.
   Never use `data/val.pt` (byte-identical to `data/train.pt`) or
   `data/data10k/val.pt` (90% of its docs are in data20k train).
2. **Starting train set.** Copy `~/dev/wiki-llm/data/data20k/train.pt` to
   `autolab/data/datasets/data20k/train.pt` (20.5M tokens, 20k docs).
3. **build_dataset** runs `mini-llm-prepare-data` into a scratch dir under
   `autolab/data/datasets/<id>/` with `--val-examples 0`, and deletes the
   `val.pt` it writes. It then checks the new train file against the frozen
   val by doc hash (reuse `mini_llm.token_overlap`) and rejects it on any
   overlap. It records tokens, docs, sha256, CLI args and disk size.
   Dataset builds need network access to Hugging Face. That's allowed: the
   no-network rule covers training and gates only.
   data20k was built by hand and its seed isn't recorded (the CLI default is 0).
   Before trusting "bigger = superset", build a tiny set (`--num-examples 200 --seed 0`)
   and check its docs are all in data20k train. Record the result either way.
4. **Everything runs natively on this Mac (MPS)** via `uv run` in this repo.
   `.venv`, `autolab/data/`, `autolab/runs/`, `autolab/state/` and checkpoints are
   gitignored. `autolab/notebook.jsonl`, `autolab/NOTEBOOK.md` and the config are committed.

## Standing rules

- **Never modify `~/dev/wiki-llm`.** Reading and copying from it is fine. Never
  run uv, git or anything that writes inside it. Never push.
- **Shared GPU.** The owner's queue runner trains on this same Mac. Before
  starting any training run (including smoke runs and gates that train),
  check `~/dev/wiki-llm/runs/*.live.json`. If any file's `updated` is less than
  2 minutes old and it doesn't have `"finished": true`, wait (poll every
  60 s) and log that you're waiting. CPU-only unit tests may run anytime.
  Implement this check in `src/autolab/` so the orchestrator obeys it too.
- **Trainer invocation.** Run `mini-llm-train` as a subprocess with
  `HF_HUB_OFFLINE=1`, `MINI_LLM_RUN_ID=<run_id>`, and cwd set to
  `autolab/runs/<run_id>/`, because train.py writes `plots/`, `checkpoints/`
  and `runs/` relative to cwd. Never pass `--baseline`. Pass `--no-tensorboard`
  never: TensorBoard events are the report source.
- **Wall-clock caps and early stops** go through train.py's existing control
  inbox. Append `{"id": "...", "type": "stop"}` as one newline-terminated
  line to `<cwd>/runs/<run_id>.commands.jsonl`, and keep `--control-poll` small
  (e.g. 25). The run still does its final full eval. The heartbeat is
  `<cwd>/runs/<run_id>.live.json`.
- **Changes to existing code** must be minimal and marked `# AUTOLAB: <why>`.
  New code goes in `src/autolab/` and tests in `tests/autolab/`. Add dependencies with
  `uv add` (e.g. optuna, jsonschema); never hand-edit the lockfile.
- **Commit** to `autolab` at sensible points. End each commit message with:
  `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Stop after each milestone** and report what you built, test results
  (counts, anything skipped), and anything surprising. Wait for "go" before the
  next milestone.
- **Long commands.** Your Bash tool has a time limit. Run anything longer
  than a few minutes in the background (`nohup ... &` with a log file,
  wrapped in `caffeinate -i` so the Mac doesn't sleep) and poll it.

## Facts you'll need (from REPO_NOTES)

- Train CLI defaults: block 64, n_embd 128, n_head 4, n_layer 4, dropout 0,
  bs 4, lr 1e-3, min_lr 2e-6, warmup 500, wd 0, seed 42, eval every 100 steps
  over 20 fixed batches (`--eval-seed 1234`), full val every 5000 steps and at the end.
- Budget in train.py is steps only: tokens = steps × batch × block.
- TB scalars: `train/batch_loss`, `train/lr`, `eval/train_loss`,
  `eval/val_loss`, `eval/full_val_loss`, `control/lr_scale`.
- MPS throughput (incl. eval overhead): emb128/L4/blk64/bs4 about 14 it/s,
  about 3.5k tok/s. emb256/L4/blk64 about 8 it/s. So a 10-min trial is about 2M tokens.
- Params: 7.28M at emb128/L4, but 6.43M of that is the tied 50,257×128
  embedding. Report tokens/param against both total and non-embedding params.
- Loss is computed inside `ModelCustomTransformer.forward` (model.py). It's
  protected.
- AdamW is built with one line in `train.py:main`; the LR schedule is
  `train.py:lr_at_step`.
- The existing causal test (`tests/test_model.py::test_causal_isolation`)
  uses only one sentence pair.

## Milestone guidance

### M2: reports + diagnosis
- `src/autolab/report.py`: build `report.json` from the TB event files
  (`tensorboard.backend.event_processing.event_accumulator`) plus the run's
  launch args and timing. Fields are in BRIEF §2. Use `eval/val_loss` and
  `eval/train_loss` as the curves, and final `eval/full_val_loss` as the headline
  number. Tokens/sec = tokens seen / training wall time. Subsample curves to 200 points or fewer.
  Spike count: residuals vs a rolling median of `train/batch_loss`, above 3
  robust std (MAD).
- Grad norm: one marked change in train.py that computes the total grad norm after
  `backward()` (no clipping) and logs TB `train/grad_norm` at the log interval.
- `src/autolab/diagnose.py`: pure function, thresholds in
  `src/autolab/thresholds.toml` (or one dataclass). Every label carries
  confidence plus the numeric evidence. `history` holds past reports and
  notebook entries (needed for "after LR tuned" and "data increase didn't help").
- Tests: synthetic curves for each of the 6 labels, plus report-building on a
  fabricated TB event file.
- Real check: one run of about 5 minutes at the default small config on data20k +
  frozen val → report.json + diagnosis. Show both in the milestone report.

### M3: orchestrator + config/data actions
- Serializable request/response dataclasses for every action (BRIEF §11).
  JSON round-trip tests.
- **Budget:** `{tokens, wall_clock_s}`. Convert tokens to `--steps`, and enforce
  wall-clock with the stop command. Record whether the run hit the token or time limit.
  Default: about 10 min per full trial and about 2–3 min per screening run. Make both configurable.
- **Default model:** pick the smallest sensible one and justify it in the
  decision log. The brief's 10-min default at emb128/L4 gives about 2M tokens,
  about 0.3 tok/param, and about 0.1 epochs of data20k. So almost every run will read
  `still_improving`, and `data_limited` can't fire. Consider a
  `train_subset_tokens` option (a prefix at a doc boundary) so the planner can
  create regimes where data limits are observable. Document whatever you choose.
- **lr_range_test** without editing the schedule: run at constant LR
  (`--min-lr == --lr`, `--warmup-steps 0`) with a high base LR (e.g. 1e-2), and
  step `lr_scale` up exponentially through the control inbox (allowed range
  1e-4 to 10, so 1e-6 to 1e-1 effective). Read loss vs LR from `train/batch_loss`
  and `train/lr`. Return LR at min smoothed loss and at divergence (loss above 4× min, or NaN).
- **ablation:** `half_data` = first half of the champion's train tokens,
  cut at a doc boundary. `wider_model` = next n_embd step up (e.g. 128→192),
  same data and budget.
- **hparam_search:** Optuna TPE + MedianPruner, storage in SQLite under
  `autolab/state/`. Report intermediate `eval/val_loss` by polling TB or the
  live heartbeat, and stop pruned trials with the stop command. LR bounds come
  from lr_range_test. Batch size varies at a fixed token budget.
- **Noise:** run the champion on 3 seeds at session start and after any
  dataset change. Store mean/std per (champion, dataset).
- **Acceptance** (BRIEF §7): screening first. A candidate goes to full budget
  only if its screen is within `screen_margin` (configurable) of the champion's
  screen. Accept only if the mean over at least 2 seeds beats the champion by
  more than 2× the noise std. After adding data, re-baseline the champion first.
- **Orchestrator:** `autolab start|resume|status|stop` (a console script in
  pyproject). State lives in `autolab/state/state.json`, written atomically after
  every step. `stop` writes a flag file checked between actions (and
  stops the active run). Caps from BRIEF §10. Checkpoints: keep the
  champion's and the latest 3; delete the rest.
- **Notebook:** `autolab/notebook.jsonl` + regenerated `autolab/NOTEBOOK.md`.
  Accepted changes are committed on `autolab` with the hypothesis and result in the
  message.
- **E2E test** (CPU, a few minutes at most): tiny model (n_embd 32, n_layer 1,
  block 32, bs 4) on about 200k tokens sliced from data20k, tiny budgets. Run
  start → a few RulePlanner actions → stop → resume → finish, and check
  state/notebook consistency. Force CPU in tests (e.g. env var read by a
  small marked hook in `device.py`, or monkeypatch).

### M4: LLMPlanner
- Check `claude --help` for current flags. Expected shape:
  `claude -p "<prompt>" --output-format json` with a JSON-schema option if
  one exists, no tool access for planning (disallow all tools), `--max-turns`
  small, and a subprocess timeout.
- Input: champion summary, latest report summary (not raw curves), diagnosis,
  last N notebook entries, allowed actions with arg schemas, remaining
  budget/caps. Output must validate against a jsonschema:
  `{action, args, hypothesis, expected_effect}`. Also validate args per action.
- Fall back to RulePlanner on timeout, non-zero exit, bad JSON, schema
  failure or a disallowed action. Log the fallback reason in the notebook.
- Tests: mocked subprocess for each failure mode, plus one real call (mark it
  so it can be skipped offline). Log each call's prompt/response under `autolab/state/llm/`.

### M5: code edits
- Add `# AUTOLAB-EDITABLE-BEGIN <name>` / `# AUTOLAB-EDITABLE-END <name>`
  markers around: `Head` + `MultiHeadAttention` (attention), `FeedForward`
  (MLP), `Block` (norm placement/residual wiring), positional embedding
  creation and its use at the top of `forward` (the region must end before
  the logits/loss code), `init_weights` methods, and in train.py a new
  `build_optimizer(model, args)` (move the AdamW line into it, marked) and
  `lr_at_step`. Commit this refactor on its own, confirm the existing suite still
  passes, and confirm a fixed-seed short run gives identical losses before and after.
- Protected paths (never editable): data.py, prepare_dataset.py,
  token_overlap.py, the eval functions and loop in train.py, the loss lines in
  model.py, `src/autolab/`, `tests/`, `autolab/data/`.
- **Worktrees:** `git worktree add ../wiki-llm-autolab-wt/cand-<n> -b
  autolab/cand-<n> <champion commit>`. To run gates against the worktree's code
  without a new venv, use this repo's `.venv` python with
  `PYTHONPATH=<worktree>/src` (it takes precedence over the editable install).
  Verify that with a check (e.g. print `mini_llm.__file__`).
- **Editing:** `claude -p` in the worktree with Read/Edit tools only
  (no Bash, no web), a small max-turns, and a prompt containing the
  hypothesis, the allowed regions and "minimal diff".
- **Scope check:** diff the candidate against the champion. Every changed line
  must fall strictly inside a BEGIN/END pair that exists unchanged in both,
  marker lines must be unchanged, and no files outside model.py/train.py may change.
- **Gates** in order (BRIEF §6): pytest (from the worktree, excluding nothing),
  shape test, then the new protected causal-leak test
  (`tests/autolab/test_causal_leak.py`). That test: eval mode, dropout 0,
  several random sequences; for each of several t, randomize all tokens > t and
  assert logits[:, :t+1] match (atol 1e-5). Then param/throughput checks
  (default +10% params unless the planner declared a capacity change) and the smoke run
  (a few hundred steps, no NaN, throughput at least 0.7× the champion's).
- **Test with deliberately bad edits** as hand-written patches, not via
  Claude, so the tests are deterministic: (a) a causal leak (drop the tril mask),
  (b) a shape bug, (c) an edit outside the allowed regions (e.g. in the loss line),
  (d) a param blow-up. Each must be rejected at the right gate with a clear
  reason. Also run one good patch (e.g. RMSNorm swap) end-to-end.
- Remove finished worktrees; keep their branches.

### M6: short real session
- Before starting, make sure the owner's runner is idle (see the shared-GPU rule).
- Pick the smallest sensible model and trial length so an hour holds about 8–12
  trials after the 3-seed noise baseline. Set caps to match (max wall-clock
  60 min, max runs, max consecutive rejections).
- Launch in the background with `caffeinate -i nohup uv run autolab start ... &`,
  and poll `autolab status` every few minutes. Don't babysit with long sleeps
  inside a single tool call.
- Final report in `autolab/SESSION_1.md` (and in chat):
  - what the agent tried, in order, with hypotheses;
  - what it accepted/rejected and why, with numbers vs noise;
  - how often the LLM planner fell back;
  - wall-clock breakdown (training vs overhead vs waiting on the GPU);
  - failure modes observed;
  - concrete recommendations for the next session.

## Decision log
- 2026-09-26: frozen val = data20k/val.pt copy; train start = data20k/train.pt;
  real runs native on macOS; autolab yields the GPU to the owner's queue runner.
