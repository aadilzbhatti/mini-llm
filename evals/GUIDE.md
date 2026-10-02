# How to read the evals

Every trained model gets one evaluation on the Mac's GPU, started automatically when a run finishes (or by hand with `uv run mini-llm-eval checkpoints/<model>.pt`). It produces:

- **a model report** (`evals/<model>.md`) with everything below for that model;
- **the summary table** (`evals/summary.md`): one row per model, plus every model paired against a reference;
- **samples side by side** (`evals/samples.md`): every model's generations on identical prompts and random draws;
- **the inference benchmark** (`evals/inference.md`): every model timed together in one session.

Models are labelled by what they are, e.g. `data320k · d512-L4 · 38.9M · T1024 · 80K steps`: training data, width and depth, parameter count, context length, and training steps (×8,192 tokens per step for the current recipe).

All evals use the same 937-document validation set, which no model trains on. Everything is seeded, so every model sees exactly the same windows, targets, trials and random draws, and two models can be compared item by item ("paired"). A paired difference of more than about 2 standard errors is real; one inside 1 SE is noise.

## Quality

### val@ctx (`full_val@T`)

**What:** average next-token loss (cross-entropy, in nats; lower is better) over the whole validation set, cut into windows of the model's own context length T.

**Why:** it's the single number training optimises, and the one in the Best table.

**How to read it:** e^loss is the perplexity. 4.04 means the model is, on average, as unsure as if it were choosing between about 57 equally likely tokens. **Caution:** targets in a T-token window sit at positions 0…T-1, so a model with a longer context is scored partly on easier, later positions. A lower val@ctx for a longer-context model mixes "better model" with "saw more history". The context curve below separates the two.

### By position

The same loss split by position in the window (0–15, 16–63, …). Early positions have little history and are always worse. It shows where in the window a model gains or loses.

## Context

### Context curve, L(c)

**What:** the same 8,000 target tokens, each predicted from exactly *c* preceding tokens, for c = 16, 32, … up to the model's context. The targets are identical for every model and every c; only the amount of history changes.

**Why:** this is the fair way to compare models with different context lengths, and to see how much each extra stretch of history is worth.

**How to read it:** compare models **at the same c**. If model A beats B at every c, it's a better predictor, not just a longer one. The **gain** rows (16→32, …, 512→1024) are how much each doubling of history helps, with a paired SE: a gain several SEs above 0 means the model really uses that extra context.

### Context benefit, cb@W

**What:** take a W-token window of a document and score the second half twice: once after the document's real first half, once after the first half of a different document. The benefit is the loss difference in nats.

**Why:** it measures whether the model uses the *meaning* of earlier text, not just local grammar. A model that only tracks the last few words would score near 0.

**How to read it:** bigger is better. Compare cb@W only between models whose context is at least W; cb@128 exists for every model.

## Long-range retrieval, ret@d

**What:** plant "The secret word is X." (X is one of 10 single-token words), add *d* tokens of real filler text, then ask "The secret word is" and check whether the model ranks X highest among the 10. 400 trials per distance; chance is 10%.

**Why:** it's the cleanest test of whether a model can find and copy a specific fact from far back, independent of general text quality.

**How to read it:** accuracy with a 95% interval. Past the model's context (shown as "key in context: no") the word has been cropped out, so accuracy must sit at chance; if it doesn't, something is wrong with the eval. Paired comparisons show "only this model right / only the reference right" counts over identical trials.

## Generation samples

**What:** free-running text. 20 frozen prompts that begin the kind of web pages the model was trained on (a definition, a biography, a list, a recipe, …), 5 draws each, so **100 generations per model**. Each is 256 new tokens at temperature 0.7, top-k 40, stopping at end-of-document. Draw *j* of prompt *i* uses the same random seed for every model, so models are compared on identical draws and nothing is cherry-picked.

**Why:** every eval above is *teacher-forced*: each prediction sees the real preceding text. Generation feeds the model its own output back hundreds of times, so small errors compound, and a model can improve on every other eval and still write degenerate text. These are base models, not instruction-tuned, so the question is whether the continuation reads like a coherent document, not whether its facts are right. One sample shows what kinds of failure occur; a hundred show whether a model actually got better.

**Per-sample numbers** (each model's table aggregates its 100):

- **rep4**: share of 4-token phrases that repeat an earlier one. 0 means no repetition; above ~0.5 the text is mostly recycling itself. Reported as the **median** (a typical sample) and **p90** (the worst 10%). Lower is better, up to a point: very high temperatures also get low rep4 by producing word salad, so read it with the text.
- **distinct-2 / distinct-4**: unique 2- and 4-token phrases as a share of all of them. 1.0 means nothing repeats. It's the complement of rep4 and is less dominated by one long loop. Higher is better, with the same caveat.
- **loops**: samples that *end* stuck in an exact cycle (the same 1–64-token span repeated to the end, at least 32 tokens), shown with a 95% interval. "…see it as a whole, see it as a whole, …" is a loop. Fewer is better. With 100 samples, two models whose intervals don't overlap really differ.
- **first loop at**: the median token where loops start, among the samples that loop. Later is better: the model writes longer before collapsing.
- **topic held**: share of the prompt's content words (e.g. "einstein", "german", "physicist") still used in the second half of the continuation. Higher is better.
- **topic span**: the median token position of the last mention of any of those words, i.e. how long the model stays on the subject before drifting for good. 256 means it was still on topic at the end; 0 means it never mentioned it. Higher is better.
- **EOS**: samples where the model ended the document before 256 tokens. Neither good nor bad by itself.

**What progress looks like:** degenerate text (frequent loops starting early, high rep4, short topic span) → coherent but wrong (few or late loops, low rep4, long topic span, wrong facts) → coherent and plausible. The **rep4 by prompt** table shows where a model is on that path for each kind of text, since averages hide prompts that improve while others get worse. The **reading set** in evals/samples.md (draw 1 of every prompt, every model side by side) is the fixed subset to read by eye; a model's Samples page has all 100.

## Cost

### Inference benchmark

**What:** every evaluated model loaded into one process and timed in interleaved rounds (order rotated), warmed up, with synchronised timing. Reports the median and p10–p90.

- **prefill@ctx**: one forward pass over a full context window (ms).
- **prefill@128**: one pass over 128 tokens. This is identical work for every model of the same width, so it's a **noise check**: if these differ a lot between same-width models, something else was using the GPU and the session is unreliable.
- **decode tok/s**: generating 64 tokens from a nearly full context. These models have no KV cache, so every new token re-runs the whole window; this is the slowest, steady-state case.

**Why:** a better model that's 3× slower may not be worth it; this keeps cost on the same page as quality.

### Training throughput

`train tok/s` and `train peak GB` are recorded by the training run itself (on its own hardware, e.g. 2×L4 on Modal), so compare them only between runs on the same hardware.

## Paired comparisons (summary table, bottom)

Every model minus a reference model, on identical targets, windows and trials: ΔL(c) and Δcb with paired standard errors, and retrieval as discordant-trial counts. Pairing removes the noise of "which targets happened to be hard", so a difference of 0.02 nats can be decisive here when the unpaired numbers' error bars overlap.
