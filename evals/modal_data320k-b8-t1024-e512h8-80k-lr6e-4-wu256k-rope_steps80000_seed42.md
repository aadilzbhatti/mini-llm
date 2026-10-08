# Evals: data320k · d512-L4 · 38.4M · T1024 · 80K steps

- checkpoint: modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 512, 'n_head': 8, 'n_layer': 4, 'dropout': 0.0, 'use_rope_embeddings': True}
- params: 38,386,257
- step: 80000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **3.9263**
- by_position@128: pos 0-15: 4.6262, pos 16-63: 3.9329, pos 64-127: 3.7463
- full_val@256: **3.7979**
- by_position@256: pos 0-15: 4.6232, pos 16-63: 3.9387, pos 64-127: 3.7448, pos 128-255: 3.6685
- full_val@512: **3.7127**
- by_position@512: pos 0-15: 4.6114, pos 16-63: 3.9502, pos 64-127: 3.7372, pos 128-255: 3.6698, pos 256-511: 3.6272
- full_val@1024: **3.6622**
- by_position@1024: pos 0-15: 4.6116, pos 16-63: 3.9562, pos 64-127: 3.7133, pos 128-255: 3.6601, pos 256-511: 3.6437, pos 512-1023: 3.6083

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.1873 | 0.0360 |
| 32 | 3.9620 | 0.0354 |
| 64 | 3.7963 | 0.0349 |
| 128 | 3.6990 | 0.0347 |
| 256 | 3.6398 | 0.0346 |
| 512 | 3.6128 | 0.0346 |
| 1024 | 3.6010 | 0.0344 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2254 | 0.0107 |
| 32->64 | +0.1657 | 0.0094 |
| 64->128 | +0.0972 | 0.0073 |
| 128->256 | +0.0592 | 0.0062 |
| 256->512 | +0.0270 | 0.0042 |
| 512->1024 | +0.0118 | 0.0027 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.6133 ± 0.0092 nats** (same-doc prefix 3.6992, other-doc prefix 4.3125)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4655 ± 0.0066 nats** (same-doc prefix 3.6064, other-doc prefix 4.0719)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3233 ± 0.0053 nats** (same-doc prefix 3.5676, other-doc prefix 3.8909)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2213 ± 0.0057 nats** (same-doc prefix 3.5627, other-doc prefix 3.7839)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 100.0% | 99%–100% | +6.39 | yes |
| 32 | 99.2% | 98%–100% | +5.79 | yes |
| 64 | 98.8% | 97%–99% | +5.14 | yes |
| 96 | 95.2% | 93%–97% | +4.34 | yes |
| 128 | 92.5% | 90%–95% | +3.95 | yes |
| 160 | 90.2% | 87%–93% | +3.83 | yes |
| 192 | 83.8% | 80%–87% | +3.52 | yes |
| 224 | 78.8% | 74%–82% | +3.12 | yes |
| 256 | 69.0% | 64%–73% | +2.78 | yes |
| 320 | 50.5% | 46%–55% | +2.07 | yes |
| 384 | 51.5% | 47%–56% | +2.13 | yes |
| 448 | 51.5% | 47%–56% | +2.13 | yes |
| 496 | 49.2% | 44%–54% | +1.87 | yes |
| 640 | 30.5% | 26%–35% | +1.42 | yes |
| 768 | 16.8% | 13%–21% | +0.47 | yes |
| 896 | 11.5% | 9%–15% | +0.14 | yes |
| 992 | 12.8% | 10%–16% | +0.23 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 54.04 ms
- decode: 35.8 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.398 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.411 / 0.752 | 19/100 (12%–28%) | token 200 | 0.412 / 0.573 | 37% | 246 tokens | 6/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.33 | 1/5 | 247 |
| biography | 0.41 | 1/5 | 248 |
| science_explainer | 0.26 | 0/5 | 109 |
| instructional | 0.41 | 0/5 | 45 |
| bullet_list | 0.66 | 2/5 | 0 |
| numbered_list | 0.56 | 1/5 | 246 |
| enumeration | 0.57 | 2/5 | 0 |
| long_dependency | 0.37 | 1/5 | 255 |
| attribution | 0.40 | 0/5 | 50 |
| numeric_units | 0.41 | 1/5 | 251 |
| agreement_gap | 0.68 | 3/5 | 244 |
| history | 0.27 | 0/5 | 254 |
| anatomy | 0.32 | 1/5 | 255 |
| geography | 0.51 | 0/5 | 254 |
| math_definition | 0.74 | 3/5 | 254 |
| environment | 0.20 | 1/5 | 253 |
| recipe | 0.38 | 0/5 | 251 |
| literature | 0.22 | 0/5 | 234 |
| technology | 0.14 | 1/5 | 187 |
| economics | 0.66 | 1/5 | 250 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 80000
- tokens: 655360000
- wall_sec: 18419.5
- train_sec: 18190.6
- eval_sec: 229.0
- train_tokens_per_sec: 36027.5
- peak_mem_gb: 4.851
- peak_mem_kind: cuda max_memory_allocated
