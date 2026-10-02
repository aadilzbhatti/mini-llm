# Evals: data160k · d512-L4 · 38.9M · T1024 · 40K steps

- checkpoint: modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 512, 'n_head': 8, 'n_layer': 4, 'dropout': 0.0}
- params: 38,910,545
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.1461**
- by_position@128: pos 0-15: 4.7770, pos 16-63: 4.1546, pos 64-127: 3.9821
- full_val@256: **4.0251**
- by_position@256: pos 0-15: 4.7720, pos 16-63: 4.1586, pos 64-127: 3.9803, pos 128-255: 3.9041
- full_val@512: **3.9427**
- by_position@512: pos 0-15: 4.7585, pos 16-63: 4.1734, pos 64-127: 3.9719, pos 128-255: 3.9087, pos 256-511: 3.8582
- full_val@1024: **3.8919**
- by_position@1024: pos 0-15: 4.7626, pos 16-63: 4.1824, pos 64-127: 3.9395, pos 128-255: 3.8936, pos 256-511: 3.8737, pos 512-1023: 3.8402

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.3775 | 0.0367 |
| 32 | 4.1722 | 0.0360 |
| 64 | 4.0143 | 0.0356 |
| 128 | 3.9207 | 0.0356 |
| 256 | 3.8580 | 0.0355 |
| 512 | 3.8260 | 0.0355 |
| 1024 | 3.8151 | 0.0354 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.2053 | 0.0107 |
| 32->64 | +0.1579 | 0.0094 |
| 64->128 | +0.0936 | 0.0073 |
| 128->256 | +0.0627 | 0.0065 |
| 256->512 | +0.0320 | 0.0046 |
| 512->1024 | +0.0109 | 0.0040 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5660 ± 0.0086 nats** (same-doc prefix 3.9305, other-doc prefix 4.4965)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4391 ± 0.0066 nats** (same-doc prefix 3.8371, other-doc prefix 4.2762)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.3027 ± 0.0052 nats** (same-doc prefix 3.8006, other-doc prefix 4.1033)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2248 ± 0.0061 nats** (same-doc prefix 3.7919, other-doc prefix 4.0167)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 98.5% | 97%–99% | +5.20 | yes |
| 32 | 92.8% | 90%–95% | +4.48 | yes |
| 64 | 82.8% | 79%–86% | +3.63 | yes |
| 96 | 73.5% | 69%–78% | +2.81 | yes |
| 128 | 57.0% | 52%–62% | +2.27 | yes |
| 160 | 56.2% | 51%–61% | +2.17 | yes |
| 192 | 52.5% | 48%–57% | +2.03 | yes |
| 224 | 46.2% | 41%–51% | +1.72 | yes |
| 256 | 41.2% | 37%–46% | +1.59 | yes |
| 320 | 33.2% | 29%–38% | +1.33 | yes |
| 384 | 30.5% | 26%–35% | +1.17 | yes |
| 448 | 27.5% | 23%–32% | +1.14 | yes |
| 496 | 29.8% | 25%–34% | +1.17 | yes |
| 640 | 26.2% | 22%–31% | +0.95 | yes |
| 768 | 22.8% | 19%–27% | +0.70 | yes |
| 896 | 22.5% | 19%–27% | +0.64 | yes |
| 992 | 22.0% | 18%–26% | +0.74 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 120.43 ms
- decode: 6.8 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.399 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.436 / 0.748 | 17/100 (11%–26%) | token 180 | 0.402 / 0.566 | 30% | 180 tokens | 13/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.42 | 1/5 | 0 |
| biography | 0.13 | 1/5 | 90 |
| science_explainer | 0.44 | 0/5 | 146 |
| instructional | 0.68 | 0/5 | 244 |
| bullet_list | 0.71 | 2/5 | 0 |
| numbered_list | 0.68 | 1/5 | 253 |
| enumeration | 0.28 | 2/5 | 0 |
| long_dependency | 0.36 | 0/5 | 244 |
| attribution | 0.22 | 0/5 | 224 |
| numeric_units | 0.52 | 0/5 | 58 |
| agreement_gap | 0.49 | 1/5 | 253 |
| history | 0.31 | 0/5 | 252 |
| anatomy | 0.19 | 1/5 | 239 |
| geography | 0.38 | 1/5 | 192 |
| math_definition | 0.53 | 3/5 | 240 |
| environment | 0.51 | 1/5 | 22 |
| recipe | 0.18 | 1/5 | 105 |
| literature | 0.53 | 1/5 | 68 |
| technology | 0.17 | 0/5 | 151 |
| economics | 0.47 | 1/5 | 0 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 10270.6
- train_sec: 9159.0
- eval_sec: 1111.6
- train_tokens_per_sec: 35776.8
- peak_mem_gb: 4.854
- peak_mem_kind: cuda max_memory_allocated
