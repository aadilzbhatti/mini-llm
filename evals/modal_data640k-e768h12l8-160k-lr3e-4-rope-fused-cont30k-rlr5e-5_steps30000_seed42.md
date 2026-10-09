# Evals: data640k · d768-L8 · 95.3M · T1024 · 190K steps

- checkpoint: modal_data640k-e768h12l8-160k-lr3e-4-rope-fused-cont30k-rlr5e-5_steps30000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 8, 'dropout': 0.0, 'use_rope_embeddings': True, 'fused_attention': True}
- params: 95,333,713
- step: 190000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **3.6055**
- by_position@128: pos 0-15: 4.3726, pos 16-63: 3.6068, pos 64-127: 3.4128
- full_val@256: **3.4692**
- by_position@256: pos 0-15: 4.3757, pos 16-63: 3.6112, pos 64-127: 3.4147, pos 128-255: 3.3298
- full_val@512: **3.3784**
- by_position@512: pos 0-15: 4.3576, pos 16-63: 3.6220, pos 64-127: 3.4109, pos 128-255: 3.3314, pos 256-511: 3.2870
- full_val@1024: **3.3239**
- by_position@1024: pos 0-15: 4.3598, pos 16-63: 3.6245, pos 64-127: 3.3803, pos 128-255: 3.3211, pos 256-511: 3.3050, pos 512-1023: 3.2666

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 3.8739 | 0.0354 |
| 32 | 3.6183 | 0.0344 |
| 64 | 3.4377 | 0.0339 |
| 128 | 3.3356 | 0.0335 |
| 256 | 3.2775 | 0.0334 |
| 512 | 3.2527 | 0.0334 |
| 1024 | 3.2410 | 0.0333 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2556 | 0.0113 |
| 32->64 | +0.1806 | 0.0096 |
| 64->128 | +0.1020 | 0.0074 |
| 128->256 | +0.0581 | 0.0060 |
| 256->512 | +0.0249 | 0.0041 |
| 512->1024 | +0.0117 | 0.0026 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.7004 ± 0.0098 nats** (same-doc prefix 3.3598, other-doc prefix 4.0602)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.5168 ± 0.0068 nats** (same-doc prefix 3.2624, other-doc prefix 3.7792)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3535 ± 0.0056 nats** (same-doc prefix 3.2293, other-doc prefix 3.5827)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2453 ± 0.0061 nats** (same-doc prefix 3.2199, other-doc prefix 3.4652)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 99.8% | 99%–100% | +7.75 | yes |
| 32 | 99.8% | 99%–100% | +8.80 | yes |
| 64 | 99.5% | 98%–100% | +8.89 | yes |
| 96 | 99.5% | 98%–100% | +9.28 | yes |
| 128 | 99.0% | 97%–100% | +9.09 | yes |
| 160 | 98.8% | 97%–99% | +8.66 | yes |
| 192 | 96.5% | 94%–98% | +8.38 | yes |
| 224 | 94.0% | 91%–96% | +6.94 | yes |
| 256 | 95.8% | 93%–97% | +8.15 | yes |
| 320 | 80.5% | 76%–84% | +5.22 | yes |
| 384 | 84.2% | 80%–87% | +5.30 | yes |
| 448 | 71.8% | 67%–76% | +4.19 | yes |
| 496 | 68.5% | 64%–73% | +3.57 | yes |
| 640 | 41.8% | 37%–47% | +2.03 | yes |
| 768 | 31.5% | 27%–36% | +1.44 | yes |
| 896 | 22.5% | 19%–27% | +0.92 | yes |
| 992 | 18.5% | 15%–23% | +0.77 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 96.24 ms
- decode: 19.1 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.526 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.356 / 0.672 | 14/100 (8%–22%) | token 156 | 0.469 / 0.627 | 43% | 233 tokens | 10/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.32 | 0/5 | 243 |
| biography | 0.27 | 2/5 | 248 |
| science_explainer | 0.41 | 1/5 | 251 |
| instructional | 0.49 | 3/5 | 0 |
| bullet_list | 0.72 | 1/5 | 0 |
| numbered_list | 0.53 | 1/5 | 248 |
| enumeration | 0.56 | 0/5 | 0 |
| long_dependency | 0.38 | 1/5 | 248 |
| attribution | 0.08 | 0/5 | 144 |
| numeric_units | 0.15 | 0/5 | 228 |
| agreement_gap | 0.60 | 1/5 | 253 |
| history | 0.29 | 0/5 | 247 |
| anatomy | 0.34 | 0/5 | 201 |
| geography | 0.36 | 0/5 | 253 |
| math_definition | 0.58 | 1/5 | 130 |
| environment | 0.14 | 0/5 | 216 |
| recipe | 0.25 | 0/5 | 235 |
| literature | 0.09 | 0/5 | 234 |
| technology | 0.12 | 1/5 | 249 |
| economics | 0.33 | 2/5 | 159 |

## Training systems (recorded by the run)

- device: NVIDIA H100 80GB HBM3
- world_size: 1
- steps: 30000
- tokens: 245760000
- wall_sec: 4143.9
- train_sec: 4028.7
- eval_sec: 115.2
- train_tokens_per_sec: 61002.4
- peak_mem_gb: 11.06
- peak_mem_kind: cuda max_memory_allocated
