# REPO_NOTES — discovery for autolab (milestone 1)

Written 2026-09-26 from a read-only pass over `~/dev/wiki-llm` (branch `minimal`,
HEAD `d4257f0`) and the clone at `~/dev/wiki-llm-autolab` (branch `autolab`).
Nothing in `~/dev/wiki-llm` was modified.

## 0. Setup state

- Clone: `~/dev/wiki-llm-autolab`, branch `autolab` off `minimal@d4257f0`.
  `origin` = `https://github.com/aadilzbhatti/mini-llm` (the original's remote);
  the push URL is set to `DISABLED-autolab-never-pushes` so a stray `git push` fails.
- The original checkout has **uncommitted** work (server/phone page, runner, a
  `build_parser()` refactor in `train.py` and `prepare_dataset.py`). None of it is
  in the clone. Autolab doesn't need it; if later code wants `build_parser()`,
  it has to be re-derived, not copied.
- Data, checkpoints and `runs/` are gitignored, so the clone has **no data**.
  Autolab must copy the frozen val file (and a train file) in; see §3.

## 1. Training entrypoint and configuration

- Entry: `mini-llm-train` → `src/mini_llm/train.py:main(argv)`. Pure argparse CLI,
  no config file. `main()` accepts an argv list, so autolab can call it
  in-process or as a subprocess.
- Model flags: `--block-size 64 --n-embd 128 --n-head 4 --n-layer 4 --dropout 0.0`
  (the CLI defaults; `ModelConfig` dataclass defaults differ: `n_layer=2`).
- Optim flags: `--batch-size 4 --steps 100 --lr 1e-3 --min-lr 2e-6
  --warmup-steps 500 --weight-decay 0.0 --seed 42`, plus `--resume`, `--restart-lr`.
- Budget is **steps only**. There is no token or wall-clock budget; tokens =
  steps × batch × block. Stopping on wall-clock is possible through the live
  control inbox (`stop` command, see §4) without editing train.py.
- Seeding: `--seed` drives init + training batch order (dedicated Generator).
  `--eval-seed` (default 1234) independently fixes the eval batches. Good for
  multi-seed noise estimates: only `--seed` should vary.
- Device: `device.select_device()` = MPS → CUDA → CPU.
- Paths are **cwd-relative**: `plots/`, `checkpoints/`, `runs/` (+ `runs/tb`),
  and `--baseline` rewrites `baselines.md/json` in cwd. Autolab must run the
  trainer with cwd set to a per-run directory (or never pass `--baseline`/`--plot-loss`).
- Tokenizer: `AutoTokenizer.from_pretrained("gpt2")` on every start. Needs the HF
  cache; set `HF_HUB_OFFLINE=1` in the trainer env to honour "no network in training".
- The queue runner (`runner/run_queue.py`, launchd) runs jobs with
  `uv run --project /Users/aadil/dev/wiki-llm` — it always executes the
  **original** checkout's code, so autolab cannot reuse it for candidate code.

## 2. Dataset build CLI

- `mini-llm-prepare-data` → `src/mini_llm/prepare_dataset.py`. Streams
  `HuggingFaceTB/smollm-corpus` / `fineweb-edu-dedup`, seeded shuffle
  (`--seed 0`, buffer 10k), GPT-2 tokenizes, EOS-separated, writes
  `<out-dir>/train.pt` and `<out-dir>/val.pt` (1-D int64 tensors).
- Size knobs: `--num-examples` (train docs, default 2000), `--val-examples`
  (default 200), `--val-pool-fraction` (0.1, must never change).
- Split rule: a doc goes to val iff sha256(text) < 0.1 → content-addressed, so
  growing `--num-examples` never moves a doc across sides. Needs network (HF).
- Roughly 1,030 tokens/doc: 10k docs ≈ 10.3M tokens (82 MB), 20k ≈ 20.5M (164 MB).
- `token_overlap.py` measures/removes doc overlap between token files by
  sha256 of token ids (no tokenizer needed).

## 3. Train/val data — FLAGS (hard requirement from the brief)

Measured by hashing every EOS-delimited document in each file:

| file | tokens | docs | overlap with a train set |
| --- | --- | --- | --- |
| `data/train.pt` | 2.0M | 2000 | — |
| `data/val.pt` | 2.0M | 2000 | **100% — byte-identical to `data/train.pt`** |
| `data/data10k/val.pt` | 0.97M | 1000 | 0% vs data10k/train, **90.1% vs data20k/train** (older positional split; owner already documented this in token_overlap.py) |
| `data/data20k/val.pt` | 0.92M | 937 | **0% vs every train file** |

1. **`data/val.pt` == `data/train.pt`.** Same SHA-256. The queue runner fills in
   `tokens=data/train.pt` and `val-tokens=data/val.pt` as defaults, so any job
   that doesn't override both reports train loss as val loss. Worth fixing on
   the main track too.
2. **Val is rebuilt whenever the dataset is rebuilt.** `prepare()` always
   writes `val.pt` next to `train.pt`. It's deterministic for a fixed
   seed/val-examples, but the data20k val has 937 docs (it was cleaned with
   `token_overlap`), so a fresh build would *not* reproduce it.
3. data10k/train is not a subset of data20k/train (978 docs differ), so
   "data10k → data20k" was not a pure data increase.

Decision for autolab:
- Frozen val = copy of `data/data20k/val.pt` → `autolab/data/val_frozen.pt`,
  recorded with its sha256 (`08f241fde5bc…` prefix on the int64 buffer). It's
  never rebuilt. Every run passes it explicitly as `--val-tokens`.
- `build_dataset(size)` calls the CLI into a scratch dir, **discards** the
  `val.pt` it writes, and then runs a doc-hash overlap check of the new train
  file against the frozen val (must be 0), rejecting the dataset otherwise.
- Starting train set: `data/data20k/train.pt` (20.5M tokens). Autolab can build
  smaller/larger sets from there using the same seed 0.

## 4. Metrics and where they go

- **stdout** (the runner tees it to `runs/<id>.log`): `step N | loss x | lr y`
  every `--log-interval` (10); `eval_train_loss`/`eval_val_loss` every
  `--eval-interval` (100); `full_val_loss` every `--full-eval-interval` (5000)
  and at the end; `Model: N parameters`; a dataset-stats table.
- **TensorBoard** `runs/tb/<run_id>/` (via `control.RunControl`): scalars
  `train/batch_loss`, `train/lr`, `eval/train_loss`, `eval/val_loss`,
  `eval/full_val_loss`, `control/lr_scale`; text `config`, events, sample
  report; `add_hparams` at the end. **This is the source autolab reports
  should parse.** `tensorboard` is already a dependency.
- **Live control files**: `runs/<id>.live.json` heartbeat (step, steps/sec,
  ETA, latest losses), `runs/<id>.commands.jsonl` inbox (`stop`, `pause`,
  `set lr_scale`, `eval_now`, `checkpoint`), `runs/<id>.events.jsonl` outbox.
  Autolab can enforce a wall-clock cap by appending `{"type":"stop"}`.
- **Runner records**: `runs/<id>.status.json` + `runs/index.jsonl` (parsed
  final metrics, forecast). Only written by the queue runner.
- eval_train/eval_val = mean over 20 fixed batches (5,120 tokens at bs4 × blk64)
  in eval mode. Fixed batches → low eval noise across steps, but it's a small
  sample of val; full_val_loss (all ~0.92M val tokens) is the one to accept on.
- **Not logged**: grad norm, tokens/sec (only steps/sec in live.json),
  NaN/inf flags. There's no grad clipping. Grad-norm stats will need a small
  marked addition to train.py (TB scalar `train/grad_norm`).
- Plots (`--plot-loss`) are PNGs in `plots/`; autolab doesn't need them.

## 5. Checkpoints

- Only with `--save`; `checkpoints/<save-name or generated>.pt`, containing
  config, model + optimizer state, batch RNG state, step, and the
  train/val/full_val/lr histories. Mid-run via the `checkpoint` command.
- `--resume` checks the model config matches and continues the batch RNG.
- Sizes: ~30 MB (emb128), ~65 MB (emb256) each with optimizer state.

## 6. Where the model pieces live (for AUTOLAB-EDITABLE markers)

| piece | location |
| --- | --- |
| attention | `model.py`: `Head` (per-head q/k/v, tril causal mask, softmax, dropout), `MultiHeadAttention` (concat + proj) |
| MLP | `model.py`: `FeedForward` (4× GELU) |
| norm | `model.py`: `Block.ln1/ln2` (pre-norm), `ModelCustomTransformer.ln_f` — all `nn.LayerNorm`, inline |
| positional encoding | `model.py`: learned `position_embedding_table` in `ModelCustomTransformer.__init__` and the first lines of `forward` |
| init | `model.py`: `init_weights` on Head, MHA and the top model (xavier) |
| weight tying | `model.py`: `lm_head.weight = token_embedding_table.weight` (lm_head keeps a bias) |
| **loss** | `model.py`: end of `ModelCustomTransformer.forward` (`F.cross_entropy`) — protected; the markers in `forward` must stop before it |
| optimizer | `train.py:main`: one `AdamW(model.parameters(), lr, weight_decay)` line (weight decay applies to all params, incl. norms/biases/embeddings) |
| LR schedule | `train.py:lr_at_step` (linear warmup → cosine to min_lr) |

To keep the train.py diff minimal, the plan is to move the AdamW line into a
`build_optimizer(model, args)` function inside a marked region, and mark `lr_at_step`.

Side notes: `Head.forward` stashes `attention_values` and `FeedForward`
stashes `gelu_activation` on every training step (a bit of extra memory,
no correctness issue). Heads are computed in a Python loop rather than
batched, which is slow on MPS.

## 7. Existing tests (`tests/`, pytest)

- `test_model.py`: forward/backward, generate, fixed-batch reproducibility,
  output shapes, target shift, **`test_causal_isolation`** (one pair of real
  sentences, compares logits before the first differing token), overfit one
  batch, weight tying (2 tests). Needs the GPT-2 tokenizer (HF cache).
- `test_train_control.py`: runs `train.main` end-to-end with live commands
  (lr_scale, checkpoint, stop, pause/resume).
- `test_control.py`, `test_baselines.py`, `test_prepare_dataset.py`,
  `test_server.py`, `test_runner.py`.
- Autolab will add a stronger random-input causal-leak test (vary *all*
  positions > t, check every t) as a protected test.

## 8. Run time on this Mac (MPS, from runs/index.jsonl, wall time incl. evals)

| config | steps | wall | it/s | tokens/s |
| --- | --- | --- | --- | --- |
| emb128 L4 blk64 bs4 (7.28M params) | 40k–160k | 0.8–3.3 h | 13–15 | ~3,400–3,800 |
| emb256 L4 blk64 bs4 (16.1M) | 80k–160k | 2.5–6.3 h | 7–9 | ~1,800–2,300 |
| emb256 L4 blk128 bs4 (16.1M) | 160k | 10 h | 4.5 | ~2,300 |

A 10-minute trial at the default small config is ~8–9k steps, **~2M tokens**.

## 9. Things that shape autolab's design

1. **Where autolab runs.** The shell I have on the Mac is a Linux VM
   (4 CPU, 3 GB RAM, no MPS). Processes there don't outlive a 3-minute tool
   call, and a Linux torch install is ~2.5 GB of CUDA wheels. So the real loop
   has to run **natively on macOS** (a launchd agent like the existing
   runner, or `uv run autolab start` in a terminal). I can build it and run the
   unit tests + tiny CPU end-to-end tests from the VM, but the owner has to start
   the real session on the Mac.
2. **Shared GPU.** The owner's queue runner is training on the same Mac right
   now (data20k emb256 blk128, 160k steps, ~1.7 h left). Two jobs on MPS at once
   would wreck throughput and timing comparisons. Autolab should wait while
   `runs/*.live.json` in the original repo shows an active run (read-only check),
   and only start trials when the GPU is free.
3. **Scale regime.** At ~2M tokens per trial, the default model sees ~0.3
   tokens/param and ~0.1 epochs of data20k. Nearly every trial will be
   `still_improving`. `data_limited` can't trigger unless autolab trains on a
   much smaller subset. `capacity_limited` is unlikely too. Also 6.4M of the
   7.28M params are the tied 50,257×128 embedding, so tokens/param should be
   reported against **non-embedding** params as well.
4. **Batch size 4** under-uses MPS. Bigger batches at a fixed token budget are
   the cheapest likely win, and `hparam_search` covers them.
5. **Noise.** Seed-to-seed noise hasn't been measured anywhere (all baselines
   are seed 42). Autolab's 3-seed champion runs will be the first estimate.
