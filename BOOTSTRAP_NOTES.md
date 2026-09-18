# Bootstrap notes

Source: `~/Documents/Workspace/ml-projects/wiki-llm` at commit `3ba9e94`
("Add pytorch transformer"), branch `main`, working tree clean apart from
untracked `.DS_Store` files and `packages.txt`. Nothing in the old repo was
modified or deleted. This is a fresh repo; the old history stayed behind.

## File mapping

| Old | New | What happened |
| --- | --- | --- |
| `src/text_prediction/model.py` | `src/mini_llm/model.py` | Copied. Two mechanical edits only (below). |
| `src/text_prediction/tokenized_dataset.py` | `src/mini_llm/data.py` | The `x`/`y` slicing convention was carried over; the `Dataset` class itself was not. |
| `src/text_prediction/data_pipeline.py` | `src/mini_llm/data.py` | Only the GPT-2 tokenizer choice survives. |
| `src/text_prediction/text_completer.py` | `src/mini_llm/generate.py` | `get_text_completions` became the function `generate_text`. |
| `src/text_prediction/trainer.py` | `src/mini_llm/train.py` | Only the inner ~5 lines of the loop and the device line survive; the 457-line `Trainer` did not. |
| `src/text_prediction/trainer.py` (device line) | `src/mini_llm/device.py` | Extracted, with CUDA/MPS order swapped (below). |
| — | `src/mini_llm/config.py` | New. A dataclass holding the six model hyperparameters, replacing the `hyperparams` dict. |
| — | `data/tiny.txt` | New. A few paragraphs of plain text, written for this project. |
| — | `tests/test_model.py` | New. Smoke tests and skipped placeholders. |

## What was copied unchanged

`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `ModelCustomTransformer`
and `generate()` — attention math, the `tril` causal mask, LayerNorm
placement, the Xavier initialization, the dropout placement, the residual
structure, the debug attribute stores (`attention_values`,
`gelu_activation`, `tok_embedding_values`, `pos_embedding_values`) and the
shape assertions are all byte-for-byte the originals.

## Mechanical changes (the only edits to model.py)

1. `forward()` lost its `writer=None, step=None` parameters, and the loop
   that normalized each head's attention matrix and called
   `writer.add_image(...)` on every forward pass was deleted. TensorBoard is
   gone from the project, and that loop iterated all modules on every call.
   A comment marks where it was.
2. A module docstring was added at the top recording provenance.

Nothing else in that file was touched.

## Adapted, not copied verbatim

- **Batching.** The original sampled one random crop per article inside
  `TokenizedDataset.__getitem__`, then padded and collated a batch with a
  combined padding + causal mask. `make_batch` samples `batch_size` random
  crops from one continuous token stream instead. No padding, so no padding
  mask. The `x` / `y` offsets are identical.
- **Tokenizer.** Still GPT-2 via `AutoTokenizer`, but without the
  `<ARTICLE_START>` / `<ARTICLE_END>` special tokens, so `vocab_size` is
  `len(tokenizer)` (50257) rather than `gpt2.vocab_size + 2` (50259).
  `sanitize_text` was dropped with the Wikipedia pipeline.
- **Device.** The original wrote
  `cuda if cuda.is_available() else mps if mps.is_available() else cpu`.
  `select_device()` tries MPS first, as you asked.
- **Decoding.** `TextCompleter` decoded with `skip_special_tokens=True`;
  with no special tokens registered, `generate_text` omits that argument.

## Deleted (not carried over)

`updated_model.py` (`nn.TransformerDecoder`), `trainer.py` (DDP, NCCL,
`DistributedSampler`, AMP/`GradScaler`, linear warmup scheduler, gradient
accumulation, gradient clipping, checkpoint save/resume by hyperparameter
hash, early stopping, TensorBoard writer and histograms, `estimate_loss`),
`data_pipeline.py` and `streaming_data_pipeline.py` (HF `datasets`,
Wikipedia, disk cache, augmentation, `custom_collate`),
`hyperparam_tuning.py` (Optuna), `live_plotter.py`, `utils.py`
(`RankFilter`, `sanitize_text`), `engine.py` (the 3-mode CLI), `tasks.py`
(Invoke), `scripts/` (`run_tensorboard.py`, `install_requirements.sh`),
`notebooks/`, `setup.py`, `poetry.lock`, `requirements.txt`,
`packages.txt`, the checked-in `src/text_prediction.egg-info/`, and the
`models/wiki-llm` submodule with its 3.7 GB checkpoint and its
`best_plot.png`.

Dropped config keys that existed only to serve the above: `max_len`,
`grad_accum_steps`, `grad_norm_clip_value`, `checkpoint_dir`, `num_samples`,
`max_epochs`, `max_iters`, `eval_iters`, `eval_interval`,
`early_stopping_patience`, `early_stopping_min_delta`, `num_gpus`,
`distributed`, `save_checkpoints`, `enable_tqdm`, `n_trials`,
`regenerate_dataset`.

## Things that look odd, left alone for you

In `model.py`:

1. **LayerNorm is applied twice per attention sublayer.** `Block.forward`
   does `x + self.sa(self.ln1(x), mask)`, and every `Head.forward` then does
   `x = self.ln(x)` again on that same input — with a separate `LayerNorm`
   per head. So `ln1` feeds four more LayerNorms in a 4-head model.
2. **`attention_mask` is threaded everywhere and never used.** `Block`,
   `MultiHeadAttention` and `Head` all accept `mask` and pass it along, but
   the two lines in `Head.forward` that would apply it are commented out.
   Masking is causal-only, from the `tril` buffer. With no padding in the
   new data path nothing is silently attended to here, but the parameter is
   a dead end as written.
3. **Head initialization is overwritten.** `Head.init_weights` sets
   `key`/`query`/`value` with `xavier_normal_`, then
   `ModelCustomTransformer.init_weights` loops over `self.modules()` and
   re-initializes *every* `nn.Linear` with `xavier_uniform_`. The
   `xavier_normal_` calls have no effect on the final weights.
4. **`self.blocks` is an `nn.Sequential` that can't be called.** Blocks take
   `(x, mask)`, so `forward` iterates it manually. It works; it's just not
   what `Sequential` is for.
5. **Side-effect stores grow references.** `Head.attention_values` (training
   only) and `FeedForward.gelu_activation` (every forward, including under
   `torch.no_grad()`) hold detached tensors until the next forward pass.
   They were feeding TensorBoard histograms that no longer exist. Harmless
   at this size, and useful for debugging, so they stayed.
6. **`forward` moves its inputs to the weights' device** before using them,
   which will hide device mismatches rather than raise.
7. **`self.step = 0`** is set and never used.
8. **`generate()` doesn't set eval mode itself.** `generate_text` calls
   `model.eval()` before delegating, and does not switch back to
   `model.train()` afterwards — worth knowing if you generate mid-training.
   With the default `dropout=0.0` it makes no difference.

In the old `trainer.py`, so not carried over but worth recording:

9. **Gradient clipping ran after the optimizer step.**
   `optimizer.step()` came first, then `clip_grad_norm_`, then
   `scheduler.step()` and `zero_grad()`. The clip never affected an update;
   it only produced the number that was logged as `grad_norm`.
10. **`TokenizedDataset.__len__` returned `len(self._data) - block_size`** —
    the number of *articles* minus a *token* count, which is not the number
    of samples.
11. **`get_batch` did `next(iter(self.train_dataloader))`**, rebuilding the
    iterator on every call, so it always returned the dataloader's first
    batch (shuffled fresh each time, so it wasn't stuck on one batch, but it
    never walked an epoch either).

## Verification performed

Run in a Linux sandbox, CPU only:

- `uv sync` resolved and installed cleanly: torch 2.14.0, transformers
  5.17.0, pytest 9.1.1 on Python 3.13.
- `uv run pytest` → 4 passed, 4 skipped (the skipped ones are your
  placeholders).
- `mini-llm-train --fixed-batch --steps 5 --save ... --sample-tokens 8`:
  model instantiated (472,064 parameters), forward, cross-entropy, backward,
  AdamW step, loss fell 5.75 → 3.16 over 5 steps, checkpoint written,
  generation ran.
- `mini-llm-generate --checkpoint ...` loaded the checkpoint and generated.

Two things that sandbox could **not** check, both for you to confirm on the
Mac: the GPT-2 tokenizer download (huggingface.co is blocked from there, so
the runs above used a stub tokenizer), and MPS selection (no Apple hardware
there — the runs printed `Using device: cpu`).
