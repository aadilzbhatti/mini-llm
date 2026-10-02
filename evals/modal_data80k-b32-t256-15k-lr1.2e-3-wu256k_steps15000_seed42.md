# Evals: data80k · d256-L4 · 16.1M · T256 · 15K steps

- checkpoint: modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42
- config: {'vocab_size': 50257, 'block_size': 256, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,138,065
- step: 15000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.5967**
- by_position@128: pos 0-15: 5.1385, pos 16-63: 4.5992, pos 64-127: 4.4595
- full_val@256: **4.4990**
- by_position@256: pos 0-15: 5.1321, pos 16-63: 4.6061, pos 64-127: 4.4560, pos 128-255: 4.4013

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.7630 | 0.0371 |
| 32 | 4.6007 | 0.0366 |
| 64 | 4.4899 | 0.0365 |
| 128 | 4.4122 | 0.0366 |
| 256 | 4.3807 | 0.0365 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1623 | 0.0091 |
| 32->64 | +0.1108 | 0.0083 |
| 64->128 | +0.0777 | 0.0069 |
| 128->256 | +0.0314 | 0.0059 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.4611 ± 0.0075 nats** (same-doc prefix 4.4129, other-doc prefix 4.8740)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.3602 ± 0.0057 nats** (same-doc prefix 4.3427, other-doc prefix 4.7029)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 87.0% | 83%–90% | +3.50 | yes |
| 32 | 77.2% | 73%–81% | +3.04 | yes |
| 64 | 60.0% | 55%–65% | +2.19 | yes |
| 96 | 48.5% | 44%–53% | +1.62 | yes |
| 128 | 27.3% | 23%–32% | +1.04 | yes |
| 160 | 31.2% | 27%–36% | +1.11 | yes |
| 192 | 23.8% | 20%–28% | +0.80 | yes |
| 224 | 20.8% | 17%–25% | +0.75 | yes |
| 256 | 11.5% | 9%–15% | +0.07 | no |
| 320 | 10.5% | 8%–14% | -0.02 | no |
| 384 | 9.2% | 7%–12% | -0.07 | no |
| 448 | 11.8% | 9%–15% | +0.04 | no |
| 496 | 10.0% | 7%–13% | +0.12 | no |
| 640 | 7.8% | 6%–11% | -0.02 | no |
| 768 | 8.5% | 6%–12% | -0.05 | no |
| 896 | 9.2% | 7%–12% | -0.10 | no |
| 992 | 12.0% | 9%–16% | +0.06 | no |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 16.28 ms
- decode: 49.9 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.104 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.271 / 0.570 | 10/100 (6%–17%) | token 181 | 0.472 / 0.690 | 23% | 142 tokens | 4/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.29 | 1/5 | 0 |
| biography | 0.20 | 0/5 | 73 |
| science_explainer | 0.32 | 1/5 | 32 |
| instructional | 0.41 | 1/5 | 253 |
| bullet_list | 0.82 | 2/5 | 0 |
| numbered_list | 0.56 | 1/5 | 253 |
| enumeration | 0.32 | 1/5 | 0 |
| long_dependency | 0.35 | 0/5 | 240 |
| attribution | 0.10 | 0/5 | 200 |
| numeric_units | 0.40 | 0/5 | 255 |
| agreement_gap | 0.15 | 0/5 | 227 |
| history | 0.22 | 0/5 | 4 |
| anatomy | 0.27 | 0/5 | 49 |
| geography | 0.25 | 1/5 | 101 |
| math_definition | 0.36 | 1/5 | 218 |
| environment | 0.12 | 0/5 | 256 |
| recipe | 0.23 | 1/5 | 56 |
| literature | 0.24 | 0/5 | 0 |
| technology | 0.12 | 0/5 | 156 |
| economics | 0.13 | 0/5 | 158 |

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
