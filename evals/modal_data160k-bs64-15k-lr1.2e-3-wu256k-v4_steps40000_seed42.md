# Evals: data160k · d256-L4 · 16.1M · T128 · 40K steps

- checkpoint: modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,105,297
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.2601**
- by_position@128: pos 0-15: 4.7788, pos 16-63: 4.2646, pos 64-127: 4.1271

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.4500 | 0.0366 |
| 32 | 4.2746 | 0.0361 |
| 64 | 4.1530 | 0.0358 |
| 128 | 4.0944 | 0.0357 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1754 | 0.0097 |
| 32->64 | +0.1215 | 0.0086 |
| 64->128 | +0.0587 | 0.0063 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5074 ± 0.0081 nats** (same-doc prefix 4.0767, other-doc prefix 4.5841)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 95.8% | 93%–97% | +4.94 | yes |
| 32 | 78.8% | 74%–82% | +3.64 | yes |
| 64 | 39.0% | 34%–44% | +1.83 | yes |
| 96 | 21.5% | 18%–26% | +0.69 | yes |
| 128 | 8.2% | 6%–11% | -0.06 | no |
| 160 | 11.5% | 9%–15% | +0.10 | no |
| 192 | 9.5% | 7%–13% | +0.01 | no |
| 224 | 7.2% | 5%–10% | -0.09 | no |
| 256 | 10.8% | 8%–14% | -0.05 | no |
| 320 | 8.2% | 6%–11% | -0.09 | no |
| 384 | 7.5% | 5%–10% | -0.09 | no |
| 448 | 9.0% | 7%–12% | +0.03 | no |
| 496 | 10.0% | 7%–13% | +0.11 | no |
| 640 | 10.8% | 8%–14% | +0.04 | no |
| 768 | 7.5% | 5%–10% | -0.11 | no |
| 896 | 9.0% | 7%–12% | -0.11 | no |
| 992 | 7.2% | 5%–10% | +0.08 | no |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 11.49 ms
- decode: 63.9 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.088 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.283 / 0.617 | 5/100 (2%–11%) | token 216 | 0.475 / 0.683 | 28% | 187 tokens | 4/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.17 | 0/5 | 0 |
| biography | 0.32 | 1/5 | 236 |
| science_explainer | 0.31 | 1/5 | 174 |
| instructional | 0.29 | 0/5 | 92 |
| bullet_list | 0.62 | 0/5 | 0 |
| numbered_list | 0.55 | 1/5 | 249 |
| enumeration | 0.71 | 1/5 | 0 |
| long_dependency | 0.32 | 1/5 | 137 |
| attribution | 0.18 | 0/5 | 203 |
| numeric_units | 0.29 | 0/5 | 255 |
| agreement_gap | 0.17 | 0/5 | 246 |
| history | 0.15 | 0/5 | 179 |
| anatomy | 0.52 | 0/5 | 82 |
| geography | 0.27 | 0/5 | 253 |
| math_definition | 0.45 | 0/5 | 210 |
| environment | 0.14 | 0/5 | 7 |
| recipe | 0.22 | 0/5 | 193 |
| literature | 0.16 | 0/5 | 89 |
| technology | 0.13 | 0/5 | 160 |
| economics | 0.25 | 0/5 | 241 |

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
