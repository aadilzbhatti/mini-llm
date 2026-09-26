# Autolab: autonomous experimentation agent for wiki-llm — build brief

You are building an automated experimentation loop for wiki-llm, a hand-written decoder-only Transformer trained on Wikipedia text, run locally on a MacBook Air (MPS when available). The loop runs training, diagnoses the result, picks the next action (tune hyperparameters, add data, run an ablation, or edit model code), and repeats. It is a separate, fully automated track. The owner keeps working on the main code independently, so never touch the original checkout.

Work milestone by milestone (section 12). After each milestone, stop and report: what you built, test results, and anything that surprised you.

## 0. Setup

- Clone the existing repo into a new directory: `git clone ~/dev/wiki-llm ~/dev/wiki-llm-autolab`. If `~/dev/wiki-llm` has a GitHub remote, set that as `origin` in the new clone; otherwise leave it pointing at the local repo.
- Create and work on branch `autolab`. Never push anywhere, and never modify `~/dev/wiki-llm`.
- Tooling: uv, pyproject.toml, src/ layout, pytest. No Poetry, Conda, setup.py or requirements.txt.
- Put all new code in `src/autolab/` with tests in `tests/autolab/`. Keep changes to existing training code minimal and clearly marked.

## 1. Discovery (do this first, change nothing)

Read the repo and write `autolab/REPO_NOTES.md` covering: the training entrypoint and how it's configured; the dataset build CLI and its size parameters; where train and val data live and how val is built; what metrics are logged and where (stdout, JSONL, TensorBoard, plots); how checkpoints are saved; which files define attention, MLP, norm, positional encoding, init, optimizer and LR schedule; the existing tests; and typical run time for a small config on this machine.

Stop and flag it if the validation set is rebuilt whenever the dataset is rebuilt, or if val overlaps train. A fixed, held-out val set is a hard requirement (see section 7).

## 2. Run reports

Every run produces `runs/<run_id>/report.json`. Reuse existing logging where possible: if TensorBoard event files exist, parse them rather than building a parallel logger. Only add a small JSONL metrics writer to the training script if nothing usable exists.

Report fields:
- Identity: run_id, git commit, config (full) and config hash, seed, dataset id, start/end time.
- Scale: parameter count, dataset size in tokens, tokens seen, epochs over data, tokens per parameter.
- Curves: train and val loss series (subsampled to at most about 200 points each).
- Summary: final smoothed train and val loss (EMA), gap (val − train), gap trend, slope of each curve over the last 20% of training (loss per log-step), best val loss and its step.
- Health: NaN/inf seen, count of loss spikes (e.g. > 3 robust std above the local trend), grad-norm stats if available.
- Performance: tokens/sec, wall time, device.
- Budget: the budget type and value the run was given, and whether it hit it.

## 3. Diagnosis

Implement `diagnose(report, history) -> Diagnosis`, a pure rule-based function. Output: one or more labels, each with a confidence and the numeric evidence behind it. Put all thresholds in one config file.

Labels:
- `still_improving`: val slope clearly negative at the end. Suggest running longer or leaving as is.
- `data_limited`: train keeps falling while val flattens or rises, with a growing gap; supported by epochs > about 1–2 or low tokens per parameter (use about 20 as a rough reference). Suggest `build_dataset` (bigger) or more regularization.
- `optimization_limited`: train and val flatten together with a small, stable gap. Spikes or instability point to LR too high; a slow smooth crawl points to LR too low or a poor schedule. Suggest hyperparameter changes or `lr_range_test`.
- `capacity_limited`: train loss itself stalls with a small gap after LR has been reasonably tuned, and a previous data increase didn't help. Suggest architecture changes or a larger model.
- `unstable`: NaNs, divergence or frequent spikes.
- `inconclusive`: evidence conflicts or the differences are within noise. Suggest an ablation.

Unit-test it with synthetic curves for each case.

## 4. Actions

Each action is a Python function the orchestrator calls, and each returns a structured result.

- `run(config, budget, seed)`: launch training, produce a report.
- `build_dataset(size)`: call the existing dataset CLI to produce a larger (or smaller) training set. Never rebuild val. Record the dataset id and size. Enforce a disk cap.
- `lr_range_test(config)`: one short run sweeping LR upward exponentially; return the LR at minimum loss and at divergence, and use them to set LR search bounds.
- `ablation(kind)`: `half_data` (same config on half the training data) and `wider_model` (a modestly larger model on the same data). These separate data-limited from capacity-limited when the diagnosis is inconclusive.
- `hparam_search(space, n_trials)`: Optuna with TPE and median pruning for numeric knobs (max LR, warmup, min LR ratio, weight decay, batch size, dropout). The planner chooses the space; use `lr_range_test` output for LR bounds.
- `code_edit(hypothesis)`: see section 6.

## 5. Orchestrator and planner

`autolab` CLI with `start`, `resume`, `status` and `stop`. The loop:

1. Read the champion (the current best accepted config + code commit) and the notebook.
2. Run or reuse the latest report and call `diagnose`.
3. Ask the planner for the next action and its hypothesis.
4. Execute the action, pass it through the gates (sections 6 and 7), and decide accept or reject.
5. Write a notebook entry and repeat.

Planner interface with two implementations:
- `RulePlanner`: deterministic mapping from diagnosis to action. Used for tests and as a fallback.
- `LLMPlanner`: calls Claude Code headless (`claude -p ... ` with JSON output; check `claude --help` for current flags). Give it the champion summary, latest report summary, diagnosis, the last N notebook entries and the list of allowed actions, and require a JSON reply: `{action, args, hypothesis, expected_effect}`. Validate the JSON against a schema, and fall back to `RulePlanner` if it's invalid.

Persist all state to disk so a session can be stopped and resumed at any point.

## 6. Model code edits

- Each code candidate is a git worktree on branch `autolab/cand-<n>` created from the champion commit.
- Invoke Claude Code headless in that worktree with the hypothesis and the allowed scope, asking for a minimal diff.
- Allowed scope: only regions between `# AUTOLAB-EDITABLE-BEGIN` and `# AUTOLAB-EDITABLE-END` markers. Add these markers around attention, MLP, normalization, positional encoding, init, and the optimizer/LR schedule. Protected, never editable: data loading, tokenization, loss computation, evaluation code, the val set, `src/autolab/`, and tests. After each edit, check the diff and reject the candidate if anything outside the allowed regions changed.

Gates, in order; the first failure rejects the candidate:
1. Existing pytest suite passes.
2. Forward/backward shape test on a tiny batch.
3. Causal-leak test (add it to the protected tests): for a random input, change tokens at positions > t and assert the logits at positions ≤ t are unchanged within tolerance. This is critical: an edit that leaks future tokens produces a dramatic fake drop in loss.
4. Parameter count and tokens/sec recorded. Reject if parameters grow more than a configurable amount (default 10%) unless the planner explicitly proposed a capacity change, in which case compare against a baseline at the same parameter count.
5. Smoke run: a few hundred steps with no NaNs, no divergence, and throughput no worse than a configurable fraction (default 0.7) of the champion's.

Only candidates that pass all gates get a full evaluation.

## 7. Fair comparison and acceptance

- Every compared run uses the same budget. Default: a fixed token count with a wall-clock cap (MPS timing is noisy and a MacBook Air throttles thermally). Make it configurable; target about 10 minutes per trial by default.
- The val set is fixed for the whole session and never changes when data is added.
- Estimate noise by running the champion on 3 seeds at the start and whenever the dataset changes.
- Two-stage evaluation: a short screening run first; only candidates within reach of the champion get the full budget.
- Accept a change only if mean val loss over at least 2 seeds beats the champion by more than 2× the seed noise std. When data is added, re-baseline the champion on the new data before comparing.

## 8. Lab notebook

Append to `autolab/notebook.jsonl` for every action: timestamp, diagnosis summary, action and args, hypothesis, expected effect, run ids, outcome, accepted or rejected, and why. Also maintain a readable `autolab/NOTEBOOK.md`. Each accepted change is a commit on `autolab` whose message contains the hypothesis and result.

## 9. Git flow

- `autolab`: the champion branch; only accepted changes land here.
- `autolab/cand-<n>`: one per code candidate; rejected ones stay for the record.
- Clean up worktrees for finished candidates; keep their branches.

## 10. Limits and safety

- Configurable caps: max runs per session, max total wall-clock, max disk for datasets and checkpoints, max consecutive rejected candidates before stopping to report.
- Keep only the champion's and the latest few checkpoints.
- No network access from training or gates. The only external call is the planner/code-edit call to Claude.
- Never modify `~/dev/wiki-llm`, never push, never edit protected paths.

## 11. Integration notes

The owner plans a separate service for controlling training remotely (phone control, run requests, reports, plots) that reuses existing tools and doesn't duplicate TensorBoard. Keep autolab's actions and reports as plain, serializable request/response objects so autolab can later become one more client of that service. Don't build the service here.

## 12. Milestones

1. **Discovery:** `REPO_NOTES.md`. Stop and report.
2. **Reports + diagnosis:** report generation for real runs; `diagnose` with synthetic-curve unit tests; generate a report for one short real run.
3. **Orchestrator with config and data actions:** `run`, `build_dataset`, `lr_range_test`, `ablation`, `hparam_search`, `RulePlanner`, noise estimation, acceptance rule, notebook, resume. End-to-end test with a tiny model and a tiny budget.
4. **LLMPlanner:** headless Claude integration with schema validation and fallback.
5. **Code edits:** editable markers, worktree flow, all gates including the causal-leak test, parameter/throughput checks. Test the gates with deliberately bad edits (a causal leak, a shape bug, an edit outside the allowed regions) and confirm each one is rejected.
6. **Short real session:** run for about an hour with the smallest sensible model, then report what the agent tried, what it accepted, and any failure modes you noticed.
