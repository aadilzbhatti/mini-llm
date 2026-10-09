# Evals: data640k · d768-L8 · 95.3M · T1024 · 160K steps

- checkpoint: modal_data640k-fw70edu30-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 8, 'dropout': 0.0, 'use_rope_embeddings': True, 'fused_attention': True}
- params: 95,333,713
- step: 160000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **3.6940**
- by_position@128: pos 0-15: 4.4680, pos 16-63: 3.6972, pos 64-127: 3.4981
- full_val@256: **3.5540**
- by_position@256: pos 0-15: 4.4709, pos 16-63: 3.7025, pos 64-127: 3.4980, pos 128-255: 3.4117
- full_val@512: **3.4602**
- by_position@512: pos 0-15: 4.4568, pos 16-63: 3.7068, pos 64-127: 3.4908, pos 128-255: 3.4130, pos 256-511: 3.3676
- full_val@1024: **3.4038**
- by_position@1024: pos 0-15: 4.4570, pos 16-63: 3.7045, pos 64-127: 3.4634, pos 128-255: 3.4030, pos 256-511: 3.3845, pos 512-1023: 3.3451

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 3.9734 | 0.0353 |
| 32 | 3.7201 | 0.0344 |
| 64 | 3.5445 | 0.0339 |
| 128 | 3.4388 | 0.0337 |
| 256 | 3.3831 | 0.0336 |
| 512 | 3.3491 | 0.0335 |
| 1024 | 3.3347 | 0.0334 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2533 | 0.0113 |
| 32->64 | +0.1755 | 0.0099 |
| 64->128 | +0.1058 | 0.0074 |
| 128->256 | +0.0557 | 0.0062 |
| 256->512 | +0.0340 | 0.0041 |
| 512->1024 | +0.0144 | 0.0027 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.6775 ± 0.0097 nats** (same-doc prefix 3.4561, other-doc prefix 4.1336)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.5109 ± 0.0068 nats** (same-doc prefix 3.3440, other-doc prefix 3.8549)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3531 ± 0.0056 nats** (same-doc prefix 3.3010, other-doc prefix 3.6542)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2436 ± 0.0063 nats** (same-doc prefix 3.2858, other-doc prefix 3.5294)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 100.0% | 99%–100% | +7.29 | yes |
| 32 | 99.0% | 97%–100% | +8.14 | yes |
| 64 | 99.0% | 97%–100% | +7.78 | yes |
| 96 | 99.0% | 97%–100% | +8.00 | yes |
| 128 | 96.8% | 95%–98% | +7.59 | yes |
| 160 | 97.8% | 96%–99% | +7.57 | yes |
| 192 | 98.5% | 97%–99% | +7.63 | yes |
| 224 | 97.0% | 95%–98% | +7.01 | yes |
| 256 | 95.5% | 93%–97% | +7.02 | yes |
| 320 | 91.2% | 88%–94% | +6.27 | yes |
| 384 | 88.0% | 84%–91% | +5.92 | yes |
| 448 | 88.0% | 84%–91% | +5.45 | yes |
| 496 | 70.2% | 66%–75% | +3.87 | yes |
| 640 | 44.8% | 40%–50% | +2.05 | yes |
| 768 | 28.0% | 24%–33% | +1.30 | yes |
| 896 | 23.2% | 19%–28% | +0.86 | yes |
| 992 | 17.0% | 14%–21% | +0.71 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 96.28 ms
- decode: 19.0 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.526 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.342 / 0.690 | 11/100 (6%–19%) | token 195 | 0.443 / 0.619 | 36% | 236 tokens | 6/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.49 | 2/5 | 248 |
| biography | 0.27 | 0/5 | 186 |
| science_explainer | 0.35 | 0/5 | 70 |
| instructional | 0.41 | 0/5 | 78 |
| bullet_list | 0.55 | 1/5 | 118 |
| numbered_list | 0.55 | 0/5 | 253 |
| enumeration | 0.73 | 2/5 | 0 |
| long_dependency | 0.25 | 2/5 | 239 |
| attribution | 0.15 | 0/5 | 0 |
| numeric_units | 0.29 | 0/5 | 248 |
| agreement_gap | 0.23 | 0/5 | 207 |
| history | 0.32 | 0/5 | 253 |
| anatomy | 0.31 | 1/5 | 255 |
| geography | 0.48 | 0/5 | 255 |
| math_definition | 0.49 | 1/5 | 252 |
| environment | 0.37 | 1/5 | 221 |
| recipe | 0.30 | 1/5 | 195 |
| literature | 0.22 | 0/5 | 231 |
| technology | 0.27 | 0/5 | 105 |
| economics | 0.26 | 0/5 | 247 |

## Training systems (recorded by the run)

- device: NVIDIA H100 80GB HBM3
- world_size: 1
- steps: 160000
- tokens: 1310720000
- wall_sec: 21617.8
- train_sec: 21320.0
- eval_sec: 297.8
- train_tokens_per_sec: 61478.3
- peak_mem_gb: 11.058
- peak_mem_kind: cuda max_memory_allocated
