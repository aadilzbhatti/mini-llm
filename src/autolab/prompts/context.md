# The task

Improve a small decoder-only Transformer trained from scratch on {dataset_tokens:,} training tokens
(`{dataset_id}`) along **four dimensions at once**. Nothing is judged on loss alone: the system keeps a
**Pareto frontier**, every program that no other program beats on all four:

1. **Quality**: full validation loss (mean cross-entropy in nats over all {val_tokens:,} tokens of a fixed,
   held-out validation set; GPT-2 tokenizer, vocab 50,257). Lower is better. Also reported: the loss on
   positions < 128 (comparable across context lengths) and on positions ≥ 128.
2. **Context capability**: a copy-at-distance probe. A random 16-token span, then d tokens of real text,
   then the span again; accuracy of continuing the repeat for d in 16…992. Distances beyond the context
   length score 0. `long_range_score` (mean over distances) is higher-is-better; `effective_context` is the
   largest d still ≥ 50% accurate.
3. **Training compute**: wall-clock time to train the fixed token budget on the GPU (and peak memory).
   Lower is better.
4. **Inference cost**: batch-1 decode latency per token at the program's context (the model re-runs the
   full context each step; there is no KV cache), plus prefill latency, parameters and memory. Lower is better.

The text is FineWeb-Edu (HuggingFaceTB/smollm-corpus, `fineweb-edu-dedup`): English web pages
filtered for educational value (explainers, articles, how-tos, blog posts), one document per page,
EOS-separated. Train and validation are disjoint by document hash.

Fixed by the evaluator (you cannot change these):
- Training budget: **{full_tokens:,} tokens** (≈{epochs:.1f} epochs of the training set), always
  {tokens_per_step:,} tokens per optimizer step: batch_size × block_size must equal {tokens_per_step:,}
  (128×64, 256×32, 512×16, 1024×8). Context length (block_size) is yours to choose among 128/256/512/1024.
- One NVIDIA {gpu} GPU, fp32, PyTorch; a generous wall-clock safety cap ({full_cap_min:.0f} min).
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
   (Longer contexts and regularizers can screen worse and still win at full budget.)
5. Full run: {full_tokens:,} tokens. If it beats the incumbent by ≥ {confirm_sigma}σ, two more seeds.
6. The **quality champion** (incumbent) changes only if the mean over ≥ 2 seeds is below
   **{bar:.4f}** (incumbent {inc_full:.4f} − {accept_sigma} × seed-noise σ {sigma:.4f}); loss
   differences smaller than ~{sigma:.3f} are noise. Separately, any fully evaluated program that no
   other program beats on all four dimensions joins the **Pareto frontier**: a program with loss within
   noise of the champion but, say, 2× the usable context or half the decode latency is a win too.

# What you can change

1. Code inside the EVOLVE blocks shown below (attention, MLP, block wiring/normalization,
   model __init__/init_weights, the top of forward() up to the final LayerNorm, the LR schedule,
   the optimizer and a hook that runs between backward() and optimizer.step(), e.g. for
   gradient clipping). Keep the function signatures the surrounding code calls. The training
   loop sets every param group's "lr" to lr_at_step(...) × lr_scale each step.
2. Hyperparameters (`hparams`), within these ranges (block_size and batch_size move together):
{hparam_ranges}

Diffs use exact text from the current program: `search` must be copied verbatim (including
indentation) from a block below and be unique; `replace` is the new text. Several small diffs are
fine; keep each one minimal. You may return no diffs (hyperparameters only) or no hparams (code
only), but not neither.
