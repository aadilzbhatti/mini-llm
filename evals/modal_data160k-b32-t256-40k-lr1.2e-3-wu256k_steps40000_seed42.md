# Evals: data160k · d256-L4 · 16.1M · T256 · 40K steps

- checkpoint: modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 256, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,138,065
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.2809**
- by_position@128: pos 0-15: 4.8422, pos 16-63: 4.2859, pos 64-127: 4.1369
- full_val@256: **4.1774**
- by_position@256: pos 0-15: 4.8387, pos 16-63: 4.2906, pos 64-127: 4.1340, pos 128-255: 4.0739

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.4876 | 0.0367 |
| 32 | 4.3009 | 0.0360 |
| 64 | 4.1626 | 0.0358 |
| 128 | 4.0806 | 0.0357 |
| 256 | 4.0417 | 0.0357 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1867 | 0.0099 |
| 32->64 | +0.1382 | 0.0091 |
| 64->128 | +0.0820 | 0.0071 |
| 128->256 | +0.0389 | 0.0062 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5291 ± 0.0083 nats** (same-doc prefix 4.0856, other-doc prefix 4.6147)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4018 ± 0.0061 nats** (same-doc prefix 4.0126, other-doc prefix 4.4144)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 95.5% | 93%–97% | +4.60 | yes |
| 32 | 82.8% | 79%–86% | +3.65 | yes |
| 64 | 68.0% | 63%–72% | +2.90 | yes |
| 96 | 56.0% | 51%–61% | +2.18 | yes |
| 128 | 40.0% | 35%–45% | +1.51 | yes |
| 160 | 40.2% | 36%–45% | +1.51 | yes |
| 192 | 31.5% | 27%–36% | +1.21 | yes |
| 224 | 24.2% | 20%–29% | +0.92 | yes |
| 256 | 11.8% | 9%–15% | +0.06 | no |
| 320 | 10.0% | 7%–13% | -0.00 | no |
| 384 | 9.2% | 7%–12% | -0.03 | no |
| 448 | 10.0% | 7%–13% | -0.03 | no |
| 496 | 11.2% | 9%–15% | +0.07 | no |
| 640 | 10.8% | 8%–14% | +0.03 | no |
| 768 | 8.2% | 6%–11% | -0.02 | no |
| 896 | 10.5% | 8%–14% | -0.05 | no |
| 992 | 11.5% | 9%–15% | +0.12 | no |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 15.81 ms
- decode: 55.6 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.104 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.324 / 0.651 | 7/100 (3%–14%) | token 180 | 0.452 / 0.644 | 30% | 232 tokens | 5/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.37 | 1/5 | 0 |
| biography | 0.14 | 0/5 | 250 |
| science_explainer | 0.43 | 0/5 | 111 |
| instructional | 0.44 | 0/5 | 247 |
| bullet_list | 0.92 | 1/5 | 0 |
| numbered_list | 0.57 | 1/5 | 23 |
| enumeration | 0.39 | 0/5 | 0 |
| long_dependency | 0.49 | 1/5 | 252 |
| attribution | 0.25 | 0/5 | 128 |
| numeric_units | 0.28 | 1/5 | 253 |
| agreement_gap | 0.27 | 0/5 | 252 |
| history | 0.31 | 0/5 | 251 |
| anatomy | 0.26 | 0/5 | 231 |
| geography | 0.47 | 0/5 | 254 |
| math_definition | 0.41 | 1/5 | 133 |
| environment | 0.10 | 0/5 | 160 |
| recipe | 0.27 | 1/5 | 253 |
| literature | 0.17 | 0/5 | 244 |
| technology | 0.12 | 0/5 | 180 |
| economics | 0.35 | 0/5 | 246 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 4386.6
- train_sec: 3864.4
- eval_sec: 522.3
- train_tokens_per_sec: 84795.4
- peak_mem_gb: 3.662
- peak_mem_kind: cuda max_memory_allocated
