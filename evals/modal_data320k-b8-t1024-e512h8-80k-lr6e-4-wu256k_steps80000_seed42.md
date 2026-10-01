# Evals: data320k · d512-L4 · 38.9M · T1024 · 80K steps

- checkpoint: modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 512, 'n_head': 8, 'n_layer': 4, 'dropout': 0.0}
- params: 38,910,545
- step: 80000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.0205**
- by_position@128: pos 0-15: 4.7256, pos 16-63: 4.0256, pos 64-127: 3.8403
- full_val@256: **3.8899**
- by_position@256: pos 0-15: 4.7225, pos 16-63: 4.0303, pos 64-127: 3.8403, pos 128-255: 3.7579
- full_val@512: **3.8018**
- by_position@512: pos 0-15: 4.7072, pos 16-63: 4.0427, pos 64-127: 3.8357, pos 128-255: 3.7601, pos 256-511: 3.7123
- full_val@1024: **3.7476**
- by_position@1024: pos 0-15: 4.7108, pos 16-63: 4.0478, pos 64-127: 3.8047, pos 128-255: 3.7508, pos 256-511: 3.7283, pos 512-1023: 3.6910

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.2671 | 0.0361 |
| 32 | 4.0514 | 0.0355 |
| 64 | 3.8868 | 0.0350 |
| 128 | 3.7960 | 0.0350 |
| 256 | 3.7321 | 0.0349 |
| 512 | 3.7002 | 0.0348 |
| 1024 | 3.6854 | 0.0347 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2157 | 0.0107 |
| 32->64 | +0.1647 | 0.0095 |
| 64->128 | +0.0908 | 0.0076 |
| 128->256 | +0.0639 | 0.0065 |
| 256->512 | +0.0319 | 0.0048 |
| 512->1024 | +0.0148 | 0.0043 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.6036 ± 0.0092 nats** (same-doc prefix 3.7932, other-doc prefix 4.3968)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4649 ± 0.0068 nats** (same-doc prefix 3.6951, other-doc prefix 4.1600)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3163 ± 0.0053 nats** (same-doc prefix 3.6566, other-doc prefix 3.9729)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2302 ± 0.0063 nats** (same-doc prefix 3.6409, other-doc prefix 3.8711)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 94.2% | 92%–96% | +5.19 | yes |
| 32 | 89.5% | 86%–92% | +4.60 | yes |
| 64 | 84.2% | 80%–87% | +4.33 | yes |
| 96 | 82.5% | 78%–86% | +3.85 | yes |
| 128 | 74.8% | 70%–79% | +3.58 | yes |
| 160 | 73.0% | 68%–77% | +3.60 | yes |
| 192 | 62.7% | 58%–67% | +3.15 | yes |
| 224 | 58.2% | 53%–63% | +2.93 | yes |
| 256 | 55.0% | 50%–60% | +2.80 | yes |
| 320 | 52.5% | 48%–57% | +2.63 | yes |
| 384 | 41.2% | 37%–46% | +2.16 | yes |
| 448 | 40.8% | 36%–46% | +2.08 | yes |
| 496 | 40.8% | 36%–46% | +1.96 | yes |
| 640 | 26.0% | 22%–31% | +1.40 | yes |
| 768 | 21.8% | 18%–26% | +1.10 | yes |
| 896 | 19.5% | 16%–24% | +0.94 | yes |
| 992 | 20.5% | 17%–25% | +0.89 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 120.45 ms
- decode: 6.2 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.399 GB

## Generation samples

10 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 | looping | topic held | stopped at EOS |
|---|---|---|---|
| 0.550 | 17/50 | 32% | 3/50 |

| prompt | rep4 | looping |
|---|---|---|
| definition | 0.67 | 4/5 |
| biography | 0.29 | 0/5 |
| science_explainer | 0.72 | 3/5 |
| instructional | 0.60 | 2/5 |
| bullet_list | 0.58 | 2/5 |
| numbered_list | 0.70 | 2/5 |
| enumeration | 0.76 | 2/5 |
| long_dependency | 0.33 | 0/5 |
| attribution | 0.27 | 0/5 |
| numeric_units | 0.59 | 2/5 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 80000
- tokens: 655360000
- wall_sec: 18516.5
- train_sec: 18294.0
- eval_sec: 222.4
- train_tokens_per_sec: 35823.7
- peak_mem_gb: 4.854
- peak_mem_kind: cuda max_memory_allocated
