# Evals: modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42

- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,105,297
- step: 15000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.4679**
- by_position@128: pos 0-15: 4.9445, pos 16-63: 4.4731, pos 64-127: 4.3448

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.6352 | 0.0371 |
| 32 | 4.4804 | 0.0365 |
| 64 | 4.3677 | 0.0363 |
| 128 | 4.3159 | 0.0363 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1549 | 0.0088 |
| 32->64 | +0.1127 | 0.0080 |
| 64->128 | +0.0517 | 0.0066 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.4619 ± 0.0075 nats** (same-doc prefix 4.2959, other-doc prefix 4.7577)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 87.0% | 83%–90% | +3.33 | yes |
| 32 | 76.8% | 72%–81% | +2.85 | yes |
| 64 | 53.0% | 48%–58% | +1.98 | yes |
| 96 | 41.8% | 37%–47% | +1.43 | yes |
| 128 | 7.2% | 5%–10% | -0.04 | no |
| 160 | 11.5% | 9%–15% | +0.08 | no |
| 192 | 7.0% | 5%–10% | -0.04 | no |
| 224 | 8.2% | 6%–11% | -0.03 | no |
| 256 | 9.0% | 7%–12% | -0.05 | no |
| 320 | 9.5% | 7%–13% | -0.00 | no |
| 384 | 10.0% | 7%–13% | -0.01 | no |
| 448 | 8.0% | 6%–11% | -0.01 | no |
| 496 | 11.0% | 8%–14% | +0.10 | no |
| 640 | 11.2% | 9%–15% | +0.08 | no |
| 768 | 9.2% | 7%–12% | -0.03 | no |
| 896 | 9.2% | 7%–12% | -0.05 | no |
| 992 | 9.0% | 7%–12% | +0.09 | no |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 12.88 ms
- decode: 56.9 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.088 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
