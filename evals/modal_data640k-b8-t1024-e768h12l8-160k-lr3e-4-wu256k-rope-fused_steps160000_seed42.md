# Evals: data640k · d768-L8 · 95.3M · T1024 · 160K steps

- checkpoint: modal_data640k-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 8, 'dropout': 0.0, 'use_rope_embeddings': True, 'fused_attention': True}
- params: 95,333,713
- step: 160000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **3.6121**
- by_position@128: pos 0-15: 4.3760, pos 16-63: 3.6136, pos 64-127: 3.4201
- full_val@256: **3.4760**
- by_position@256: pos 0-15: 4.3783, pos 16-63: 3.6184, pos 64-127: 3.4215, pos 128-255: 3.3371
- full_val@512: **3.3855**
- by_position@512: pos 0-15: 4.3601, pos 16-63: 3.6298, pos 64-127: 3.4178, pos 128-255: 3.3386, pos 256-511: 3.2941
- full_val@1024: **3.3310**
- by_position@1024: pos 0-15: 4.3624, pos 16-63: 3.6321, pos 64-127: 3.3871, pos 128-255: 3.3281, pos 256-511: 3.3121, pos 512-1023: 3.2736

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 3.8828 | 0.0354 |
| 32 | 3.6280 | 0.0345 |
| 64 | 3.4474 | 0.0339 |
| 128 | 3.3452 | 0.0336 |
| 256 | 3.2868 | 0.0335 |
| 512 | 3.2617 | 0.0335 |
| 1024 | 3.2497 | 0.0334 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2548 | 0.0113 |
| 32->64 | +0.1806 | 0.0096 |
| 64->128 | +0.1022 | 0.0075 |
| 128->256 | +0.0584 | 0.0060 |
| 256->512 | +0.0251 | 0.0041 |
| 512->1024 | +0.0120 | 0.0026 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.6987 ± 0.0097 nats** (same-doc prefix 3.3673, other-doc prefix 4.0660)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.5158 ± 0.0069 nats** (same-doc prefix 3.2696, other-doc prefix 3.7854)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3532 ± 0.0056 nats** (same-doc prefix 3.2362, other-doc prefix 3.5895)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2448 ± 0.0061 nats** (same-doc prefix 3.2277, other-doc prefix 3.4725)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 100.0% | 99%–100% | +8.15 | yes |
| 32 | 100.0% | 99%–100% | +9.38 | yes |
| 64 | 99.8% | 99%–100% | +9.44 | yes |
| 96 | 99.8% | 99%–100% | +10.12 | yes |
| 128 | 99.8% | 99%–100% | +9.96 | yes |
| 160 | 99.2% | 98%–100% | +9.78 | yes |
| 192 | 98.0% | 96%–99% | +9.31 | yes |
| 224 | 96.5% | 94%–98% | +7.82 | yes |
| 256 | 98.5% | 97%–99% | +9.22 | yes |
| 320 | 85.0% | 81%–88% | +5.90 | yes |
| 384 | 91.2% | 88%–94% | +6.03 | yes |
| 448 | 79.5% | 75%–83% | +4.69 | yes |
| 496 | 73.0% | 68%–77% | +3.96 | yes |
| 640 | 44.0% | 39%–49% | +2.20 | yes |
| 768 | 34.5% | 30%–39% | +1.50 | yes |
| 896 | 24.2% | 20%–29% | +0.94 | yes |
| 992 | 18.2% | 15%–22% | +0.79 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 96.56 ms
- decode: 19.0 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.526 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.257 / 0.661 | 16/100 (10%–24%) | token 191 | 0.494 / 0.669 | 46% | 241 tokens | 14/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.64 | 2/5 | 252 |
| biography | 0.15 | 0/5 | 254 |
| science_explainer | 0.39 | 0/5 | 224 |
| instructional | 0.52 | 2/5 | 97 |
| bullet_list | 0.78 | 1/5 | 0 |
| numbered_list | 0.54 | 0/5 | 0 |
| enumeration | 0.38 | 2/5 | 0 |
| long_dependency | 0.22 | 1/5 | 208 |
| attribution | 0.09 | 0/5 | 182 |
| numeric_units | 0.25 | 0/5 | 254 |
| agreement_gap | 0.21 | 2/5 | 253 |
| history | 0.27 | 1/5 | 247 |
| anatomy | 0.17 | 1/5 | 244 |
| geography | 0.35 | 0/5 | 255 |
| math_definition | 0.38 | 1/5 | 246 |
| environment | 0.10 | 0/5 | 242 |
| recipe | 0.22 | 0/5 | 250 |
| literature | 0.07 | 0/5 | 161 |
| technology | 0.11 | 1/5 | 256 |
| economics | 0.55 | 2/5 | 236 |

## Training systems (recorded by the run)

- device: NVIDIA H100 80GB HBM3
- world_size: 1
- steps: 160000
- tokens: 1310720000
- wall_sec: 21634.7
- train_sec: 21339.5
- eval_sec: 295.2
- train_tokens_per_sec: 61422.2
- peak_mem_gb: 11.058
- peak_mem_kind: cuda max_memory_allocated
