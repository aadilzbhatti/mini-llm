# mini-llm

A minimal local environment for debugging a hand-written decoder-only
Transformer: tiny model, tiny text, one training loop, nothing else.

The model in `src/mini_llm/model.py` is copied from the original `wiki-llm`
project and is **unchanged**. See `BOOTSTRAP_NOTES.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv sync            # creates .venv and installs torch + transformers + pytest
uv run pytest      # runs the test suite
```

No manual virtualenv activation: `uv run` handles it.

## Usage

```bash
uv run mini-llm-train                      # 100 steps on data/tiny.txt
uv run mini-llm-train --fixed-batch --steps 500
uv run mini-llm-train --steps 500 --save model.pt --sample-tokens 40
uv run mini-llm-generate --checkpoint model.pt --prompt "The lighthouse"
```

Both commands print the selected device at startup (MPS on this Mac, else
CUDA, else CPU).

Useful flags: `--text`, `--block-size`, `--n-embd`, `--n-head`, `--n-layer`,
`--dropout`, `--batch-size`, `--steps`, `--lr`, `--weight-decay`, `--seed`,
`--fixed-batch`, `--log-interval`, `--sample-tokens`, `--save`.

## Defaults

Deliberately tiny, for fast correctness experiments rather than quality:
block size 64, embedding 128, 4 heads, 2 layers, dropout 0.0, batch 4,
AdamW at lr 1e-3, weight decay 0.0. Model defaults live in
`src/mini_llm/config.py`.

## Layout

```
pyproject.toml          uv / hatchling project config, entry points
data/tiny.txt           a few paragraphs of plain text
src/mini_llm/
  model.py              your custom Transformer, copied as-is
  config.py             ModelConfig dataclass + build_model()
  data.py               text -> tokens -> (x, y) batches
  device.py             select_device(): MPS -> CUDA -> CPU
  train.py              forward -> loss -> zero_grad -> backward -> step
  distributed.py        torchrun/DDP plumbing (no-op without torchrun)
  remote/modal_train.py launch train.py on Modal GPUs (see below)
  generate.py           encode -> model.generate() -> decode
tests/test_model.py     smoke tests + skipped placeholders for your tests
tests/test_ddp.py       single-process unchanged; 2-rank gloo DDP via torchrun
configs/modal/          example configs for remote runs
scripts/fetch_modal_run.sh  pull a Modal run to ./runs/<run_id>
BOOTSTRAP_NOTES.md      what came from where, and what looks suspicious
```

## The training loop

The whole thing, in `train.py`:

```python
_, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

No scheduler, no AMP, no gradient accumulation, no clipping, no eval loop,
no checkpoint resume. Add back what you want, when you want it.

## Data

`data.py` turns one text file into a single 1-D tensor of GPT-2 token ids
and cuts random crops out of it:

```python
x = tokens[i     : i + block_size]
y = tokens[i + 1 : i + block_size + 1]
```

`fixed_batch(tokens, batch_size, block_size, seed=0)` returns the same batch
every call, which is what `--fixed-batch` trains on. The first time you run
anything that uses the tokenizer, `transformers` downloads the GPT-2
tokenizer files (a few MB) and caches them.

## Multi-GPU runs on Modal (DDP)

`train.py` also runs under `torchrun` as DistributedDataParallel: one process
per GPU, each on its own contiguous shard of `--tokens`, gradients averaged
across processes every step. `--batch-size` is the **global** batch (each
of N ranks takes `batch_size / N`), so the same flags mean the same
optimization on 1 or N devices. Eval uses the same fixed batches as a local
run, split across ranks, so eval losses are comparable with your Mac runs.
Add `--bf16` for bf16 autocast (CUDA only; ignored locally). Without
`torchrun`, nothing changes: `uv run mini-llm-train` is the same
single-device loop. Details are in `src/mini_llm/distributed.py` and the
DDP comments in `train.py`.

### One-time setup

```bash
uv sync --group modal                 # modal is an optional dependency group
uv run --group modal modal setup      # browser login; writes ~/.modal.toml
# Upload datasets once. The wiki-llm-data volume mirrors the local data/ tree:
uv run --group modal modal volume put wiki-llm-data data/data10k /data10k
```

### Launch

```bash
uv run --group modal modal run --detach src/mini_llm/remote/modal_train.py \
    --config configs/modal/smoke.json --gpus H100:2
```

- `--config`: a JSON of `mini-llm-train` flags, in the queue runner's job
  format (`{"name": ..., "args": {...}}`, so `runner/*.json` jobs work as-is)
  or a flat `{"flag": value}` dict. `--save` and `--plot-loss` are on by
  default. Relative `data/...` paths resolve into the `wiki-llm-data` volume.
- `--args "--steps 2000 --bf16"`: extra raw flags appended to the config.
- `--gpus`: any Modal GPU spec. The count after `:` becomes
  `torchrun --nproc_per_node`, e.g. `H100:2` (default), `H100:8`, `A100-80GB:4`,
  or `L4` (single GPU, still through torchrun). `--gpus cpu` runs 2 gloo ranks
  on CPU, which is the cheapest end-to-end check of the whole path.
- `--name`: a suffix for the run id. `--timeout-hours`: defaults to 24 (Modal's maximum).
- `--detach` keeps the job running if your laptop sleeps or disconnects.
  Watch it at modal.com/apps, or stream logs with
  `uv run --group modal modal app logs -f <app id>` (`modal app list` shows the `ap-...` id).
- `uv run modal run ...` without `--group modal` also works, as long as your
  last sync included the group (a plain `uv sync` removes it again).

### Outputs

Each run writes to the `wiki-llm-runs` volume under
`<UTC timestamp>-<short git sha>[-dirty][-name]/`:

```
run.json        resolved config + argv, git sha/branch, torch/CUDA/NCCL versions,
                GPU names, start/finish time, exit code
git.diff        uncommitted changes that were part of the run (if the tree was dirty)
train.log       full stdout of rank 0
checkpoints/    ckpt_*.pt: plain state_dict (no "module." prefix), loads on the Mac
plots/          loss_*.png
runs/tb/<id>/   TensorBoard events
```

Fetch a run (it's safe to repeat on a live run, since outputs are committed every 30 s):

```bash
scripts/fetch_modal_run.sh                              # list runs
scripts/fetch_modal_run.sh <run_id>                     # -> ./runs/<run_id>, then import
scripts/fetch_modal_run.sh <run_id> --repo ~/dev/wiki-llm   # import into another checkout
scripts/fetch_modal_run.sh <run_id> --no-import         # fetch only
uv run tensorboard --logdir runs/<run_id>/runs/tb
```

A finished run is then imported like a local `--baseline` run
(`mini-llm-import-run`): checkpoint and sample report go to
`checkpoints/modal_<config name>_seed<N>.pt|.md`, the plot to `plots/`, and a
row to `baselines.md`, ranked with local runs on `full_val_loss`. The run id,
GPUs and git sha are kept in `baselines.json` only. Re-importing replaces the
row instead of adding a second one. Unfinished runs are fetched but not imported.

### Tracking Modal runs in the control page

`mini-llm-modal-mirror` polls the `wiki-llm-runs` volume every 30 s and writes
each Modal run into `runs/` in the same files the queue runner writes
(`<id>.status.json`, `<id>.live.json`, `<id>.log`). The phone page then shows
Modal runs in Live and History, with a `modal · <gpus>` badge, step/ETA,
losses, log and full_val curve. Finished runs that had `--val-tokens` are
imported automatically (checkpoint, plot, sample report, baselines row), so
the page's Plot and Samples buttons work for them too. A run with no
heartbeat for 15 min (timeout, disabled workspace) shows as `interrupted`.
Modal runs are read-only on the page: live control is off for multi-GPU runs.

```bash
uv run --group modal mini-llm-modal-mirror --repo ~/dev/wiki-llm            # foreground
cp runner/com.aadil.mini-llm-modal-mirror.plist ~/Library/LaunchAgents/     # or as a service
launchctl load ~/Library/LaunchAgents/com.aadil.mini-llm-modal-mirror.plist
```

A checkpoint resumes locally as usual (`--resume runs/<run_id>/checkpoints/...`),
or remotely with `--args "--resume /runs/<run_id>/checkpoints/<file>.pt"`.

### Cost

Modal bills per second while the container runs, including image build and
startup, on top of a CPU/memory charge. List prices from
[modal.com/pricing](https://modal.com/pricing) as of 2026-09-26, per GPU:

| GPU        | $/sec     | ≈ $/hr | `H100:2`-style multiples |
|------------|-----------|--------|--------------------------|
| H100       | 0.001097  | 3.95   | ×2 = 7.90/hr, ×4 = 15.80/hr, ×8 = 31.59/hr |
| A100 80GB  | 0.000694  | 2.50   | ×4 = 9.99/hr |
| A100 40GB  | 0.000583  | 2.10   | |
| L40S       | 0.000542  | 1.95   | |
| L4         | 0.000222  | 0.80   | |
| T4         | 0.000164  | 0.59   | |

CPU is $0.0000131/core/sec (≈ $0.05/core/hr) and memory $0.00000222/GiB/sec.
The Starter plan includes $30/month of free compute. Rules of thumb:

- The first launch builds the image (the CUDA torch wheels are several GB).
  That's minutes of CPU time, cached afterwards; code edits only re-upload `src/`.
- The models here are small, and a 2-GPU H100 run is mostly communication
  and Python overhead unless the per-GPU batch is large. Check throughput
  (`steps_per_sec` in `runs/<id>/runs/<id>.live.json`) on a short run before
  paying for 8 GPUs; a single L4 or A100 may be the better deal.
- A 24 h timeout on `H100:8` is a ~$760 ceiling. Set `--timeout-hours` to
  what you expect the run to need.
