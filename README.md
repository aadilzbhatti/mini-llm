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
  generate.py           encode -> model.generate() -> decode
tests/test_model.py     smoke tests + skipped placeholders for your tests
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
