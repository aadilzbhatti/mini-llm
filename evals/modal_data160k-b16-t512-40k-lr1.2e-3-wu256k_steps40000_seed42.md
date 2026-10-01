# Evals: data160k · d256-L4 · 16.2M · T512 · 40K steps

- checkpoint: modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 512, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,203,601
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.3381**
- by_position@128: pos 0-15: 4.9075, pos 16-63: 4.3475, pos 64-127: 4.1888
- full_val@256: **4.2286**
- by_position@256: pos 0-15: 4.9038, pos 16-63: 4.3544, pos 64-127: 4.1863, pos 128-255: 4.1181
- full_val@512: **4.1550**
- by_position@512: pos 0-15: 4.8916, pos 16-63: 4.3686, pos 64-127: 4.1827, pos 128-255: 4.1255, pos 256-511: 4.0767

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.5308 | 0.0365 |
| 32 | 4.3546 | 0.0360 |
| 64 | 4.2206 | 0.0358 |
| 128 | 4.1351 | 0.0359 |
| 256 | 4.0788 | 0.0358 |
| 512 | 4.0626 | 0.0358 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1762 | 0.0097 |
| 32->64 | +0.1340 | 0.0089 |
| 64->128 | +0.0856 | 0.0074 |
| 128->256 | +0.0563 | 0.0063 |
| 256->512 | +0.0162 | 0.0048 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5134 ± 0.0085 nats** (same-doc prefix 4.1464, other-doc prefix 4.6598)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4046 ± 0.0062 nats** (same-doc prefix 4.0576, other-doc prefix 4.4621)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.2784 ± 0.0049 nats** (same-doc prefix 4.0210, other-doc prefix 4.2994)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 95.8% | 93%–97% | +5.38 | yes |
| 32 | 91.2% | 88%–94% | +4.85 | yes |
| 64 | 85.5% | 82%–89% | +4.31 | yes |
| 96 | 79.2% | 75%–83% | +3.64 | yes |
| 128 | 64.0% | 59%–69% | +2.99 | yes |
| 160 | 67.5% | 63%–72% | +3.11 | yes |
| 192 | 57.5% | 53%–62% | +2.60 | yes |
| 224 | 57.5% | 53%–62% | +2.49 | yes |
| 256 | 45.8% | 41%–51% | +1.98 | yes |
| 320 | 44.0% | 39%–49% | +1.75 | yes |
| 384 | 43.0% | 38%–48% | +1.71 | yes |
| 448 | 32.8% | 28%–37% | +1.34 | yes |
| 496 | 37.2% | 33%–42% | +1.45 | yes |
| 640 | 9.2% | 7%–12% | +0.04 | no |
| 768 | 9.2% | 7%–12% | -0.06 | no |
| 896 | 9.5% | 7%–13% | -0.02 | no |
| 992 | 9.5% | 7%–13% | +0.03 | no |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 25.96 ms
- decode: 41.3 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.145 GB

## Generation samples

10 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 | looping | topic held | stopped at EOS |
|---|---|---|---|
| 0.498 | 3/50 | 26% | 4/50 |

| prompt | rep4 | looping |
|---|---|---|
| definition | 0.46 | 0/5 |
| biography | 0.16 | 0/5 |
| science_explainer | 0.63 | 0/5 |
| instructional | 0.33 | 0/5 |
| bullet_list | 0.82 | 2/5 |
| numbered_list | 0.65 | 0/5 |
| enumeration | 0.55 | 0/5 |
| long_dependency | 0.22 | 0/5 |
| attribution | 0.51 | 1/5 |
| numeric_units | 0.64 | 0/5 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 5352.6
- train_sec: 4834.1
- eval_sec: 518.4
- train_tokens_per_sec: 67784.6
- peak_mem_gb: 3.741
- peak_mem_kind: cuda max_memory_allocated
