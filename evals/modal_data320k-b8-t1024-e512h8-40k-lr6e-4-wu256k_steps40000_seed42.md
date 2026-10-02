# Evals: data320k · d512-L4 · 38.9M · T1024 · 40K steps

- checkpoint: modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 512, 'n_head': 8, 'n_layer': 4, 'dropout': 0.0}
- params: 38,910,545
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.1958**
- by_position@128: pos 0-15: 4.8421, pos 16-63: 4.2037, pos 64-127: 4.0284
- full_val@256: **4.0733**
- by_position@256: pos 0-15: 4.8351, pos 16-63: 4.2099, pos 64-127: 4.0272, pos 128-255: 3.9499
- full_val@512: **3.9900**
- by_position@512: pos 0-15: 4.8271, pos 16-63: 4.2245, pos 64-127: 4.0240, pos 128-255: 3.9564, pos 256-511: 3.9021
- full_val@1024: **3.9376**
- by_position@1024: pos 0-15: 4.8370, pos 16-63: 4.2296, pos 64-127: 3.9938, pos 128-255: 3.9425, pos 256-511: 3.9161, pos 512-1023: 3.8846

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.4398 | 0.0364 |
| 32 | 4.2424 | 0.0357 |
| 64 | 4.0773 | 0.0353 |
| 128 | 3.9897 | 0.0353 |
| 256 | 3.9216 | 0.0352 |
| 512 | 3.8913 | 0.0352 |
| 1024 | 3.8792 | 0.0352 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1974 | 0.0103 |
| 32->64 | +0.1651 | 0.0095 |
| 64->128 | +0.0876 | 0.0075 |
| 128->256 | +0.0681 | 0.0065 |
| 256->512 | +0.0303 | 0.0045 |
| 512->1024 | +0.0122 | 0.0043 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5547 ± 0.0087 nats** (same-doc prefix 3.9809, other-doc prefix 4.5357)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4324 ± 0.0066 nats** (same-doc prefix 3.8878, other-doc prefix 4.3202)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3052 ± 0.0052 nats** (same-doc prefix 3.8459, other-doc prefix 4.1511)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2277 ± 0.0062 nats** (same-doc prefix 3.8352, other-doc prefix 4.0630)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 88.8% | 85%–91% | +4.33 | yes |
| 32 | 74.5% | 70%–79% | +3.41 | yes |
| 64 | 60.0% | 55%–65% | +2.73 | yes |
| 96 | 49.5% | 45%–54% | +2.04 | yes |
| 128 | 33.8% | 29%–39% | +1.64 | yes |
| 160 | 38.5% | 34%–43% | +1.69 | yes |
| 192 | 29.0% | 25%–34% | +1.43 | yes |
| 224 | 23.5% | 20%–28% | +1.09 | yes |
| 256 | 24.0% | 20%–28% | +1.01 | yes |
| 320 | 17.0% | 14%–21% | +0.80 | yes |
| 384 | 19.0% | 15%–23% | +0.84 | yes |
| 448 | 14.2% | 11%–18% | +0.72 | yes |
| 496 | 19.2% | 16%–23% | +0.75 | yes |
| 640 | 11.5% | 9%–15% | +0.47 | yes |
| 768 | 13.5% | 10%–17% | +0.41 | yes |
| 896 | 14.0% | 11%–18% | +0.42 | yes |
| 992 | 11.2% | 9%–15% | +0.44 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 135.76 ms
- decode: 6.4 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.399 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.423 / 0.780 | 17/100 (11%–26%) | token 183 | 0.387 / 0.553 | 33% | 237 tokens | 9/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.36 | 1/5 | 0 |
| biography | 0.24 | 0/5 | 254 |
| science_explainer | 0.42 | 1/5 | 245 |
| instructional | 0.55 | 1/5 | 246 |
| bullet_list | 0.96 | 4/5 | 0 |
| numbered_list | 0.64 | 1/5 | 209 |
| enumeration | 0.73 | 2/5 | 0 |
| long_dependency | 0.41 | 1/5 | 253 |
| attribution | 0.28 | 1/5 | 236 |
| numeric_units | 0.47 | 0/5 | 248 |
| agreement_gap | 0.42 | 0/5 | 252 |
| history | 0.14 | 1/5 | 249 |
| anatomy | 0.45 | 0/5 | 241 |
| geography | 0.21 | 0/5 | 237 |
| math_definition | 0.53 | 1/5 | 252 |
| environment | 0.23 | 0/5 | 237 |
| recipe | 0.50 | 1/5 | 157 |
| literature | 0.37 | 0/5 | 175 |
| technology | 0.41 | 1/5 | 47 |
| economics | 0.47 | 1/5 | 16 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 10666.0
- train_sec: 9565.5
- eval_sec: 1100.5
- train_tokens_per_sec: 34256.4
- peak_mem_gb: 4.854
- peak_mem_kind: cuda max_memory_allocated
