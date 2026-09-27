# The task

Minimize the **full validation loss** (mean cross-entropy in nats over all {val_tokens:,} tokens of
a fixed, held-out Wikipedia-style validation set; GPT-2 tokenizer, vocab 50,257) of a small
decoder-only Transformer trained from scratch on {dataset_tokens:,} training tokens (`data20k`).

Fixed by the evaluator (you cannot change these):
- Training budget: **{full_tokens:,} tokens** (≈{epochs:.1f} epochs of the training set), context
  length (block_size) {block_size}. The number of steps is the token budget ÷ (batch_size × {block_size}).
- One NVIDIA {gpu} GPU, fp32, PyTorch; wall-clock cap {full_cap_min:.0f} min per full run
  ({wall_cap_mult}× the current program's time) — a slower program trains on fewer tokens.
- Evaluation code, data loading, tokenization and the loss computation.

# How a proposal is evaluated (a cascade; the first failure rejects it)

1. Static: every SEARCH must match exactly once inside an EVOLVE block; nothing outside the
   blocks may change; block code may not reference `targets`, files/OS/network, `eval`/`exec`,
   or `torch.load`/`torch.save`; hyperparameters must be in range.
2. CPU tests at your program's size: forward/backward shapes, a causal test (logits at position t
   must not change when tokens after t change; logits must not depend on targets), and the full
   test suite.
3. Parameter count ≤ {param_cap:,} ({param_cap_mult}× the initial program).
4. Screen: {screen_tokens:,} tokens, one seed. Rejected if NaN, slower than {throughput_floor}× the initial
   throughput, or if its val loss is more than {screen_margin} above the incumbent's screen
   ({inc_screen:.4f}). Short screens overstate gains from faster early progress, so they only reject.
5. Full run: {full_tokens:,} tokens. If it beats the incumbent by ≥ {confirm_sigma}σ, two more seeds.
6. **Accepted** as the new incumbent only if the mean over ≥ 2 seeds is below
   **{bar:.4f}** (incumbent {inc_full:.4f} − {accept_sigma} × seed-noise σ {sigma:.4f}).
   Differences smaller than ~{sigma:.3f} are noise.

# What you can change

1. Code inside the EVOLVE blocks shown below (attention, MLP, block wiring/normalization,
   model __init__/init_weights, the top of forward() up to the final LayerNorm, the LR schedule,
   the optimizer and a hook that runs between backward() and optimizer.step(), e.g. for
   gradient clipping). Keep the function signatures the surrounding code calls. The training
   loop sets every param group's "lr" to lr_at_step(...) × lr_scale each step.
2. Hyperparameters (`hparams`), within these ranges:
{hparam_ranges}

Diffs use exact text from the current program: `search` must be copied verbatim (including
indentation) from a block below and be unique; `replace` is the new text. Several small diffs are
fine; keep each one minimal. You may return no diffs (hyperparameters only) or no hparams (code
only), but not neither.
