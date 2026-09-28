# Audit the current best program and find upgrades

## The system you serve

A small decoder-only Transformer is trained from scratch and scored on full validation loss (nats, lower
is better) at a fixed budget. Fixed by the evaluator (do not propose changing these):
- Data: {dataset_id}, {dataset_tokens:,} training tokens of FineWeb-Edu (English educational web pages),
  GPT-2 tokenizer (vocab 50,257), a frozen held-out validation set.
- Budget: {full_tokens:,} training tokens (≈{epochs:.1f} epochs), context length {block_size}.
- Hardware: one NVIDIA {gpu}, fp32, PyTorch {torch_version}. Wall-clock cap ≈ {cap_min:.0f} min per run.
- Parameter cap: {param_cap:,} (the program has {params:,}).

What can change: the code inside the program's EVOLVE blocks (attention, MLP, block wiring and norms,
model init / embeddings, the top of forward(), the LR schedule, the optimizer, and a hook between
backward() and optimizer.step()), and these hyperparameters: {hparam_keys}.

## Why you're being called

{trigger}

## The current best program ({program}) — full val loss {full_mean:.4f} over {n_seeds} seeds

Hyperparameters: {hparams}

Training report and diagnosis:
{report}

Its code (every EVOLVE block):
{blocks}

## Already tried in this line of work (don't re-propose these)

{tried}

## Existing technique cards (don't duplicate these)

{cards}

## Your task

1. Go through the program component by component (attention, MLP, normalization, residual wiring,
   positional encoding, initialization, optimizer, LR schedule, gradient handling, regularization, and
   hyperparameter scaling). For each, ask: is this how strong small-LM training recipes of 2023-2026 do
   it (e.g. nanoGPT/modded-nanoGPT speedruns, Pythia/OLMo/SmolLM/TinyLlama-style recipes, recent
   optimizer and normalization papers)? What is outdated, missing, or mis-set *for this scale and budget*?
2. Search the web to check current practice and evidence. Aim for about {max_searches} searches.
3. Return up to {max_cards} cards for the most promising upgrades, best first. Each must be implementable
   inside the EVOLVE blocks and/or the hyperparameters above, run in fp32 on one {gpu} with PyTorch
   {torch_version}, and plausibly lower validation loss at this scale. Favor well-evidenced, low-risk
   upgrades over exotic ideas, but include one bolder idea if the evidence is good.
