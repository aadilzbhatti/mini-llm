# Autolab handoff (for Claude Code on the Mac)

Read this first, then `autolab/BRIEF.md` (the spec) and `autolab/REPO_NOTES.md`
(discovery results). This file holds the decisions and standing rules that
override the brief where they conflict, plus implementation guidance per
milestone. Keep it up to date: when you make a design decision, add it to
the "Decision log" at the bottom.

## Where things stand

- Repo: `~/dev/wiki-llm-autolab`, branch `autolab`, cloned from
  `~/dev/wiki-llm` (branch `minimal`, commit d4257f0). `origin` is the GitHub
  repo (github.com/aadilzbhatti/mini-llm). Pushing the `autolab` branch there is
  allowed (see standing rules).
- Milestone 1 (discovery) is done: `autolab/REPO_NOTES.md`.
- Setup on the Mac is done: `uv sync`; baseline suite 47 passed; data copied
  into `autolab/data/` (gitignored); frozen val sha256 in `autolab/config.toml`.
- Milestone 2 (reports + diagnosis) code and tests are done: `src/autolab/`
  {config, gpu, trainer, report, diagnose}.py + `thresholds.toml`; tests in
  `tests/autolab/`. Run the suite with `AUTOLAB_FORCE_CPU=1 uv run pytest`
  while the owner's runner is training.
- 2026-09-27: merged the owner's `modal-ddp` branch (DDP training, Modal
  launcher, volume mirror) into `autolab`, and added a Modal backend for
  autolab trials (`src/autolab/modal_backend.py`). The local Mac GPU waiter was
  stopped; the M2 real check runs on Modal instead.
- 2026-09-27: autolab runs without a chat session. `autolab daemon` (launchd
  `com.aadil.autolab-daemon`) collects Modal results and live progress every
  60 s. `autolab dashboard` (launchd `com.aadil.autolab-dashboard`, port 8766)
  serves the dashboard, mounted on the tailnet at `/autolab` next to the
  owner's control page. Plists are in `autolab/launchd/`. The M5 controller loop
  will run inside the daemon.
- M3 (evaluator + cascade) is built and running live: `src/autolab/{program,evaluate}.py`,
  the `EVOLVE-BLOCK` markers (commit 7a7e0e7, the base of every program), and the protected
  `tests/autolab/test_causal_leak.py`. The daemon advances programs every cycle, and the
  dashboard has a Programs tab. `autolab evolve propose --diff FILE` adds a candidate by hand.
  M4 (LLM proposer) will call the same `propose()`.
- Milestones 4–6 are not started. M3–M6 were re-planned on 2026-09-26 around
  AlphaEvolve (see "Design pivot" under Milestone guidance), which overrides
  BRIEF §4–§5 and parts of §6–§9.

## Decisions (override the brief where they conflict)

1. **Frozen val.** Copy `~/dev/wiki-llm/data/data20k/val.pt` to
   `autolab/data/val_frozen.pt`. It holds 918,728 tokens in 937 docs and
   overlaps no train file. Store its sha256 in the autolab config and check it
   before every run. Always pass it explicitly as `--val-tokens`.
   Never use `data/val.pt` (byte-identical to `data/train.pt`) or
   `data/data10k/val.pt` (90% of its docs are in data20k train).
2. **Starting train set.** Copy `~/dev/wiki-llm/data/data20k/train.pt` to
   `autolab/data/datasets/data20k/train.pt` (20.5M tokens, 20k docs).
3. **build_dataset** (between sessions only since the design pivot) runs `mini-llm-prepare-data` into a scratch dir under
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
  run uv, git or anything that writes inside it. The owner works there and his
  launchd runner executes that checkout.
- **Pushing:** the owner allows pushing the `autolab` branch to `origin`. The
  first time, use `git push -u origin autolab`, and push after each commit that
  lands on `autolab`. Never force-push. Never push any other branch (`main`,
  `minimal`, etc.). `autolab/cand-*` branches stay local unless the owner asks.
  Never commit data, checkpoints or run outputs (`*.pt` is already
  gitignored; keep it that way).
- **Within this clone** you're free to modify whatever the milestones need,
  subject to the protected-path rules the gates enforce on *candidate* edits
  (BRIEF §6). Those rules govern what the agent may change when proposing model
  edits, not your own build work. Still keep changes to existing training code
  small and marked, so the owner can follow them.
- **Modal (default backend since 2026-09-27).** Trials run on Modal via
  `autolab.modal_backend` (app `autolab-train`, volumes `autolab-runs` and
  `autolab-data`). Never write to the owner's `wiki-llm-runs` or `wiki-llm-data`
  volumes: his `mini-llm-modal-mirror` imports finished runs from
  `wiki-llm-runs` into his baselines. Every submit is priced against
  `[modal] max_usd` in `autolab/config.toml` (spent + pending estimates). Raise it
  only with the owner's say-so. Redeploy (`python -m autolab.modal_backend deploy`)
  after changing `src/autolab/` or dependencies. Candidate `mini_llm` code is sent
  with each call, so it needs no redeploy.
- **Shared GPU (local runs only).** The owner's queue runner trains on this same Mac. Before
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

## Running autolab (no chat session needed)

- `uv run autolab modal submit-batch autolab/experiments/<batch>.json` queues trials.
  The daemon collects them, and the dashboard shows them.
- `uv run autolab modal status` prints the queue table and spend.
- Dashboard: https://aadils-macbook-air.taile67486.ts.net/autolab (tailnet only),
  or http://127.0.0.1:8766 on the Mac. It reads `autolab/state/`, `autolab/runs/`,
  `autolab/experiments/`, HANDOFF's decision log and both TOML configs, and never writes.
- Services: `launchctl list | grep autolab`. Logs are in `autolab/state/{daemon,dashboard}.log`.
- An experiment batch file carries `id`, `title`, `purpose`, base
  model/optim/eval, per-job `variant`/`budget`, and an `analysis` kind. Record the
  why of every batch in `purpose`: it is what the dashboard shows.

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

### Design pivot (2026-09-26): AlphaEvolve-style evolution

The owner wants autolab to reproduce AlphaEvolve (Novikov et al., 2025,
"AlphaEvolve: A coding agent for scientific and algorithmic discovery") at
laptop scale. This replaces BRIEF §4–§5 (fixed action menu + planner) and
reshapes §6–§9. BRIEF §2–§3 (reports, diagnosis), §6's gates, §7's fairness
rules and §10's caps still apply.

The paper's loop (its Fig. 2) is:

    parent, inspirations = database.sample()
    prompt = prompt_sampler.build(parent, inspirations)
    diff   = llm.generate(prompt)           # SEARCH/REPLACE blocks
    child  = apply_diff(parent, diff)
    results = evaluator.execute(child)     # evaluation cascade -> dict of scores
    database.add(child, results)

Mapping to autolab:

| AlphaEvolve | autolab |
| --- | --- |
| `# EVOLVE-BLOCK-START/END` markers | the same markers in model.py / train.py (they replace the planned `AUTOLAB-EDITABLE` markers) |
| program | base commit + the contents of every EVOLVE block + a hyperparameter patch (JSON) |
| `evaluate()` → dict of scalars | the cascade below, returning `neg_val_loss`, `tokens_per_sec`, `params`, …; the M2 report.json is the "rendered evaluation result" |
| evaluation cascade | static checks → CPU gates → smoke → screen → full → extra seeds |
| program database (MAP-Elites + islands) | a small version: SQLite under `autolab/state/` |
| prompt sampler (+ stochastic formatting, explicit context) | parent + inspirations + their reports/diagnoses + recent failures |
| LLM ensemble (Flash + Pro) | `claude -p` with a configurable model mix |
| async pipeline, many evaluators | Modal: spawn up to `max_containers` trials at once. The controller keeps that many children in flight and adds each to the database as it returns |

What changes at our scale (updated 2026-09-27 for Modal: trials now run in
parallel on Modal GPUs, up to `max_containers`, priced against `max_usd`. That makes
the paper's "evaluators pool" real, and the main limit is now dollars, not
one Mac GPU):
- **Evaluations are the scarce resource**: tens per night, not thousands.
  LLM latency is irrelevant next to a 3–10 min training run. So the model mix
  should lean toward the strongest model (the paper used the fast model for
  volume; our volume is capped by the GPU). Spend effort on gates that reject
  bad children before they touch the GPU.
- **Score noise is comparable to real improvements.** Programs are ranked by
  mean over evaluated seeds, and "best" claims need extra seeds (BRIEF §7).
- **Data is part of the evaluator, not the program.** A session fixes the
  train set, frozen val, token budget and `block_size` (loss at different
  context lengths isn't comparable). `build_dataset`, Optuna `hparam_search`,
  `lr_range_test` and the ablations are retired from the loop.
  Hyperparameters are evolved by the LLM through the hparam patch instead
  (the paper's Fig. 3 evolves a `sweep()` the same way). Changing data is a
  between-sessions decision.
- **The M2 diagnosis becomes prompt context**, not an action selector. As in
  the paper, the LLM decides what to change; it sees the parent's report
  summary and diagnosis labels with evidence.

### M3: evaluator + cascade (the `h` function)
Build this first; everything else hill-climbs on it.
- **Markers.** Add `# EVOLVE-BLOCK-START <name>` / `# EVOLVE-BLOCK-END <name>`
  around: `Head` + `MultiHeadAttention` (attention), `FeedForward` (mlp),
  `Block` (block wiring/norms), positional embedding creation and its use at the top
  of `forward` (ending before the logits/loss lines), the `init_weights` methods, and
  in train.py a new `build_optimizer(model, args)` (move the AdamW line into it)
  and `lr_at_step`. Commit this refactor on its own. Confirm the existing suite
  passes and that a fixed-seed short CPU run gives identical losses before and after.
- **Protected** (never inside a block): data.py, prepare_dataset.py,
  token_overlap.py, the eval functions and loop in train.py, the loss lines in
  model.py, `src/autolab/`, `tests/`, `autolab/data/`.
- **Program representation.** `Program{id, parent_id, base_commit, blocks:
  {name: text}, hparams: {..}, rationale, created_by (llm model | mutation),
  scores, stage_reached, reports: [run_id...]}`. Store block texts, not branches.
  Materialize a program by writing its blocks into a fresh worktree at
  `base_commit` (`git worktree add --detach ../wiki-llm-autolab-wt/<id>`), run with
  this repo's `.venv` python and `PYTHONPATH=<worktree>/src`. Verify that with a
  check (print `mini_llm.__file__`). Remove the worktree after evaluation.
- **Hparam patch.** Allowed keys and ranges in a config file: lr, min_lr,
  warmup_steps, weight_decay, dropout, batch_size, n_embd, n_head, n_layer.
  Fixed by the session: block_size, token budget, eval settings, data, val.
- **Diff application.** Parse `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE`
  blocks. Each SEARCH must match exactly once, inside one EVOLVE block of the
  parent. After applying, re-check scope: block markers unchanged, no text outside
  blocks changed, no files outside model.py/train.py touched. Reject with a
  specific reason otherwise.
- **Cascade**, stopping at the first failure. Every stage's outcome and reason is
  recorded on the program.
  0. static: diff applies; scope check; `python -c "import mini_llm.model, mini_llm.train"`.
  1. CPU gates (`AUTOLAB_FORCE_CPU=1`): pytest from the worktree (full suite),
     shape test, then the protected causal-leak test
     (`tests/autolab/test_causal_leak.py`: eval mode, dropout 0, several random
     sequences; for several t randomize all tokens > t and assert logits[:, :t+1]
     match, atol 1e-5). Must also pass on the program's own hparams (n_embd etc.).
  2. params/throughput: record params. Params above a configurable cap
     (default 1.5× the initial program) are rejected. The wall-clock cap
     handles speed.
  3. smoke (GPU): a few hundred steps, no NaN/divergence, throughput ≥ 0.5× the
     initial program (configurable).
  4. screen: `screen_tokens` (default about 2 min of the initial program's
     throughput), 1 seed.
  5. full: `full_tokens` (default about 8–10 min), only if the screen is within
     `screen_margin` of the best screen in the database (or of the parent's).
  6. confirm: extra seeds (to ≥ 2, 3 for a new best) only for programs whose full
     score would make them the best.
- **Budget fairness.** Stages 4–5 use a fixed token budget with a wall-clock cap
  of `1.25 ×` the initial program's time for that budget. A slower child that hits
  the cap is scored on what it trained (fewer tokens, final full eval still
  runs), so speed is paid for in loss. Record which limit was hit.
- **Scores** (maximize, as in the paper): `neg_full_val_loss` (primary, mean over
  evaluated seeds, with `n_seeds` and std), `tokens_per_sec`, `neg_params`,
  and the screen score. The report.json + diagnosis are stored alongside.
- **Noise.** At session start, run the initial program at the screen and full
  budgets on 3 seeds. Store mean/std per (base program, dataset, budget). A
  "new best" needs its mean over ≥2 seeds to beat the incumbent by > 2× std
  (BRIEF §7). Until then it's recorded as a "contender".
- **Screen vs full ranking check** (the M3 real-data deliverable). Before
  trusting screens, evaluate about 4 hand-written variants (e.g. LR ×0.3/×3,
  batch size 16, RMSNorm) at both budgets and report the rank agreement. If the
  screen ranks poorly, lengthen it or drop the stage. Record the result in the
  decision log.
- **Default model and budgets.** Pick the smallest sensible model and budgets
  so a night holds ≳ 40 screens. Justify in the decision log (throughput
  numbers: see "Facts").
- **Bad-patch tests** (hand-written SEARCH/REPLACE, deterministic): (a) causal
  leak (drop the tril mask) → stage 1 causal test; (b) shape bug → stage 1;
  (c) edit outside a block (the loss line) → stage 0 scope; (d) param blow-up →
  stage 2; (e) SEARCH text not found / ambiguous → stage 0. One good patch
  (RMSNorm swap) must pass stages 0–2 on CPU.

### M4: program database + prompt sampler + LLM proposer
- **Database** (SQLite in `autolab/state/`): programs, scores, lineage, failures.
  Scaled-down MAP-Elites + islands:
  - Islands: `n_islands` (default 2) evolve independently. Every
    `migrate_every` accepted children, copy each island's best to the others.
  - MAP-Elites grid per island over two descriptors, bucketed: params and
    tokens/sec. Each cell keeps its best program by the primary score.
  - `sample()`: parent = the island's best with prob `p_exploit` (0.5), else a
    uniformly random occupied cell's elite. Inspirations = top-k (2) by score,
    plus 1 random elite from another cell, excluding the parent.
  - Failed children are stored with their failure stage and reason. They
    never become parents, but the prompt shows the recent ones.
- **Prompt sampler** (template files under `src/autolab/prompts/`):
  - Explicit context: task statement (decoder-only LM on Wikipedia text,
    minimize full val loss at a fixed token budget on an M-series MacBook Air
    with MPS), the fixed session settings, the hparam keys/ranges, the rules for
    SEARCH/REPLACE and EVOLVE blocks, and "keep throughput".
  - Prior programs: each inspiration's changed blocks and hparams (as a diff vs
    the initial program to save context), with scores.
  - Current program: full EVOLVE blocks + hparams + scores + a report summary
    (final losses, gap, slopes, tokens/param, spikes, grad-norm stats) +
    diagnosis labels with evidence.
  - Recent failures: the last N children's rationale + failure reason (e.g.
    "causal-leak test failed", "OOM", "diverged at step 900").
  - Stochastic formatting: a few template alternatives for the task
    instruction (e.g. "propose a new idea…" / "make a targeted fix to the
    diagnosed problem…" / "simplify…"), picked by configured probabilities.
  - Meta-prompt evolution: out of scope for now. Log it as a follow-up.
- **LLM call.** `claude -p` with `--output-format json`, `--json-schema` for a
  reply `{rationale, diffs: [{search, replace}], hparams: {…}}`, all tools disabled
  (`--tools ""` or `--disallowed-tools`; check `claude --help`), `--model` from
  a weighted mix in config (e.g. 80% the strongest model, 20% a faster one;
  record which model made each child), and a subprocess timeout. Log every
  prompt/response under `autolab/state/llm/`.
  Diffs are applied by autolab (M3), never by Claude editing files. This
  replaces the old M5 plan of Claude using Read/Edit tools in a worktree.
- **Fallback mutation operator** (no LLM): perturb 1–2 hparams within ranges
  (log-scale for lr/wd). Used on LLM failure (timeout, non-zero exit, bad JSON,
  schema failure, a diff that doesn't apply), in tests, and as a "no LLM" baseline.
  Record the fallback reason.
- Tests: database sampling/migration/elite replacement (deterministic with a
  seed), prompt rendering snapshot, mocked subprocess for each LLM failure mode,
  and one real `claude -p` call (skippable offline).

### M5: controller loop, persistence, notebook
- `autolab start|resume|status|stop` (console script). State in
  `autolab/state/state.json` + the SQLite db, written atomically after every
  step. `stop` writes a flag file checked between stages (and sends a stop to
  the active run).
- **Pipeline:** one GPU worker runs stages 3–6 serially. A producer keeps up to
  `prefetch` (default 2) children that have passed stages 0–2 ready, so LLM
  calls and CPU gates overlap GPU training. Threads or asyncio, either is fine.
  Obey the shared-GPU rule before every GPU stage (`autolab.gpu.wait_for_gpu`)
  and log time spent waiting.
- **Caps** (BRIEF §10): max wall-clock, max GPU evaluations, max LLM calls,
  max consecutive children failing stage ≤ 2 (a sign the prompt is broken), disk.
  Checkpoints: don't `--save` except for the current best.
- **Notebook**: `autolab/notebook.jsonl` gets one entry per child: parent,
  inspirations, model, rationale, diffs summary, hparams, stage reached,
  scores, failure reason. Keep `action` (= `evolve` | `mutate` | `baseline`),
  `accepted` and `improved` for diagnose(). Regenerate `autolab/NOTEBOOK.md` with
  the best-so-far curve vs evaluations and the lineage of the best program.
- **Git:** when a new best is confirmed over noise, write its blocks onto
  `autolab` and commit with its rationale, scores vs the incumbent, and lineage
  in the message, then push. Individual children are not branches.
- **E2E test** (CPU, a few minutes at most): tiny model (n_embd 32, n_layer 1,
  block 32, bs 4) on ~200k tokens sliced from data20k, tiny budgets, a mocked LLM
  that returns a fixed sequence of good/bad diffs. Run start → a few children →
  stop → resume → finish. Check that the db, state and notebook agree and at
  least one child beats the initial program.

### M6: real session
- Make sure the owner's runner is idle (shared-GPU rule). The owner's queue
  can be busy for many hours, so prefer an overnight window.
- First a ~1 h pilot (noise baseline + a handful of children) to catch prompt
  and gate problems, then a longer run with caps set to fit the window.
- Launch in the background with `caffeinate -i nohup uv run autolab start ... &`
  and poll `autolab status` every few minutes. Don't babysit with long sleeps.
- Final report in `autolab/SESSION_1.md` (and in chat):
  - best-so-far score vs number of GPU evaluations, against the initial
    program's seed noise;
  - the best program's lineage with each step's rationale and diff;
  - cascade funnel: how many children reached/failed each stage and why;
  - LLM stats: calls, fallbacks and reasons, per-model success rate;
  - wall-clock breakdown (training vs gates vs LLM vs waiting on the GPU);
  - failure modes and recommendations.
- Follow-up (not in M6 unless time allows): the paper's "no evolution"
  ablation (always sample the initial program as parent) at equal GPU budget.

## Decision log
- 2026-09-26: frozen val = data20k/val.pt copy; train start = data20k/train.pt;
  real runs native on macOS; autolab yields the GPU to the owner's queue runner.
- 2026-09-26: owner allows pushing the `autolab` branch to origin (no force-push,
  no other branches). This overrides BRIEF §0/§10 "never push".
- 2026-09-26 (M2): frozen val sha256 (file bytes) `28b1041a…569ee` lives in
  `autolab/config.toml`, checked by `autolab.config.check_frozen_val` before
  every run. The token-buffer hash `08f241fd…` from REPO_NOTES is recorded next to it.
- 2026-09-26 (M2): report slopes are loss per 1k steps plus the fitted change
  across the tail window (absolute and relative), not "per log-step", so
  they don't depend on the eval interval. diagnose() thresholds use the relative change.
- 2026-09-26 (M2): non-embedding params = total − (vocab + block_size) × n_embd
  (tied token table + learned position table). The lm_head bias counts as non-embedding.
  Params, device and dataset size are parsed from the trainer's stdout, so they stay
  right for edited candidate code.
- 2026-09-26 (M2): tokens/sec and train_wall_s come from TB wall times of the
  first and last `train/batch_loss` (the loop incl. periodic evals, excluding
  startup and the final full eval). `performance.wall_s` is the whole process.
- 2026-09-26 (M2): `config_hash` covers model+optim+eval+steps+dataset_id and
  excludes the seed, so seeds of one config share a hash (for noise grouping).
- 2026-09-26 (M2): `train/grad_norm` is computed only on log-interval steps
  (no sync on other steps). The last step is logged for loss but not grad norm
  unless it falls on the interval.
- 2026-09-26 (M2): pulled the M3 CPU hook forward: `AUTOLAB_FORCE_CPU=1` in
  `device.py` (marked). tests/autolab sets it automatically. The existing
  `test_train_control` trains via `select_device()`, so run the full suite with the env var
  while the GPU is shared.
- 2026-09-26 (M2): GPU counts as free only after 2 consecutive idle checks
  60 s apart, so autolab doesn't slip into the gap between two queued owner jobs.
- 2026-09-26 (M2): autolab runs pass `--full-eval-interval 0` (only the final
  full eval) and `--control-poll 25`. Wall-clock stop has a 15-min kill grace
  for the final eval.
- 2026-09-26 (M2): diagnose() reads only `action`, `accepted` and `improved`
  from notebook entries. M3's notebook schema must keep those three fields.
  "LR tuned" = an lr_range_test/hparam_search entry, or ≥3 distinct LRs for
  the same model shape in past reports. "Data increase didn't help" = an
  explicit build_dataset entry with improved/accepted false, or two reports with
  the same config on different dataset sizes where the bigger one isn't better by
  more than noise_mult × noise_std.
- 2026-09-26 (M2): caveat. With cosine-to-min_lr, the last 20% of a run is
  low-LR annealing, so flat tails are partly the schedule. diagnose() notes it
  when final LR < 5% of peak. M3 should keep this in mind when choosing budgets.
- 2026-09-26: design pivot. The owner wants an AlphaEvolve-style loop (program
  database + LLM-proposed SEARCH/REPLACE diffs in EVOLVE blocks + an evaluation
  cascade) instead of the brief's fixed action menu and planner. M3–M6 are
  rewritten. The M2 report and diagnosis are kept as the evaluator's output and
  as prompt context.
- 2026-09-27: Modal backend design.
  - One GPU per trial, running `python -m mini_llm.train` directly (no
    torchrun), so the control inbox and the wall-clock stop keep working.
  - Every call ships the trial's full `src/mini_llm/*.py` as an overlay on
    PYTHONPATH, so programs never need an image rebuild.
  - The container checks the val/train sha256 recorded on the Mac, trains with
    `block_network=True` (the tokenizer is baked into the image), builds
    report.json + diagnosis in-container, copies the run dir (minus *.pt) to
    `autolab-runs`, and returns report/diagnosis/log.
  - Local state lives in `autolab/state/modal_calls.json`. Cost = wall_s ×
    GPU $/s from config, a lower bound (CPU/memory charges aren't counted).
  - Throughput and wall-clock caps are only comparable on the same GPU type,
    so a session fixes one GPU type. The noise baseline must be measured on
    that GPU type, not on the Mac.
- 2026-09-27 (M3 prep results, `autolab/experiments/m3_prep.json`, 15 trials, $1.15 on L4):
  Seed noise of the base program (emb256/L4/blk128, bs64, lr 1.2e-3): σ = 0.027 at
  the 5.2M-token screen (mean 6.536) and σ = 0.043 at the 21M-token full budget
  (mean 5.365). Screens rank the 5 variants almost like full runs (Spearman ρ = 0.9;
  only lr 3.6e-3 and lr 3.6e-4 swap places), but they exaggerate gains from faster
  early progress. bs16 is −0.34 at the screen (12.7σ) and only −0.046 at full (1.1σ,
  not significant); warmup 500 is −0.15 at the screen and −0.012 at full. So screens
  are good for rejecting, not for accepting: keep the cascade's rule (screen within
  margin → full runs decide). Screens also flatter high LR: lr 3.6e-3 is +0.16 at the screen
  and +1.09 at full. Calibrate before M3: spike_rate_max = 0.02 flags healthy bs64
  runs (2–3% spikes, grad-norm max/median ≈ 3) as `unstable`, while the diverging lr 3.6e-3 run
  (max/median ≈ 10) isn't flagged. Tune spike_rate_max / grad_norm_max_over_median
  on these runs.
- 2026-09-27 (M3): diagnosis recalibrated on the 15 prep runs. Spike/grad-norm stats skip
  the initial descent (the steep drop from ~10.8 read as spikes in every healthy run).
  The "final smoothed" losses use the tail fit's end value (the EMA lagged ~0.1 on screens).
  `unstable` now needs >= 5 spikes above 3%, grad-norm max/median > 3, or p95/median
  > 1.5. That flags the diverging lr 3.6e-3 run and no healthy one.
- 2026-09-27 (M3): no git worktrees. A program is block texts + hparams in
  `autolab/state/evolve/programs/<id>.json`, materialized from the base commit into
  `autolab/state/evolve/work/<id>/src` when needed (deleted when done). Tests always come
  from this checkout, since they're protected. The JSON store is simple and inspectable;
  M4 can index it or move it to SQLite for sampling.
- 2026-09-27 (M3): regions are attention, mlp, block, model_init, forward_body (up to ln_f;
  the logits/loss lines stay outside), lr_schedule, optimizer. The optimizer region also holds a
  new no-op `before_optimizer_step` hook (called between backward() and step()). Without it,
  gradient clipping, one of the most common useful edits, would be impossible.
- 2026-09-27 (M3): static rules beyond the scope check. Block code may not name
  `targets` (labels are in scope in forward(), so a block could leak them into the logits), file/OS/
  network/pickle modules, exec/eval/compile/__import__, or torch.load/save. The protected
  test also checks that logits don't depend on the targets passed.
- 2026-09-27 (M3): the CPU gate runs shape → causal/label-leak (at the program's own size) →
  full protected suite, as three pytest runs so reject reasons are specific. Every test is
  seeded via `-p autolab.pytest_seed`. The owner's `test_overfits_one_batch` fails ~1 in
  3 on the unmodified code without seeding (unseeded init + tokens). Worth fixing on the main
  track. The full suite costs ~90 s of Mac CPU per candidate.
- 2026-09-27 (M3): the smoke stage is folded into the screen. On Modal a 640-step screen
  (~$0.03) costs barely more than a separate smoke run's container overhead. The screen
  applies the smoke checks (NaN, failed run, throughput >= 0.5x initial) first.
- 2026-09-27 (M3): acceptance. The screen passes if <= incumbent screen mean + 0.10 (screens
  overstate gains, so they only reject). A full run below the incumbent's mean triggers 2
  more seeds. Accept iff the mean over >= 2 seeds < incumbent mean − 2σ_full (= 5.280 for
  p0), else "contender". Wall caps = 1.25x p0's mean wall time: screen 184 s, full 667 s.
  p0 = the 6 base runs of m3_prep (run on pre-marker code 2c351a5, behavior-identical).
