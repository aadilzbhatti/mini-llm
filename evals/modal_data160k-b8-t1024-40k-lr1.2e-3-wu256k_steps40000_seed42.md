# Evals: data160k · d256-L4 · 16.3M · T1024 · 40K steps

- checkpoint: modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42
- config: {'vocab_size': 50257, 'block_size': 1024, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,334,673
- step: 40000
- val: data/data20k/val.pt

## Quality

full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each target had. Use the context curve below to compare context lengths.

- full_val@128: **4.3599**
- by_position@128: pos 0-15: 4.9292, pos 16-63: 4.3749, pos 64-127: 4.2064
- full_val@256: **4.2445**
- by_position@256: pos 0-15: 4.9220, pos 16-63: 4.3795, pos 64-127: 4.2058, pos 128-255: 4.1286
- full_val@512: **4.1645**
- by_position@512: pos 0-15: 4.9084, pos 16-63: 4.3915, pos 64-127: 4.2007, pos 128-255: 4.1343, pos 256-511: 4.0816
- full_val@1024: **4.1141**
- by_position@1024: pos 0-15: 4.9177, pos 16-63: 4.3984, pos 64-127: 4.1691, pos 128-255: 4.1188, pos 256-511: 4.0950, pos 512-1023: 4.0638

## Context curve (fixed targets)

The same 8,000 target tokens, each predicted from exactly c preceding tokens (identical targets for every model). Gain = paired loss reduction from doubling c.

| history c | loss | ± SE |
|---|---|---|
| 16 | 4.5763 | 0.0366 |
| 32 | 4.3916 | 0.0362 |
| 64 | 4.2426 | 0.0359 |
| 128 | 4.1566 | 0.0361 |
| 256 | 4.0917 | 0.0360 |
| 512 | 4.0605 | 0.0361 |
| 1024 | 4.0407 | 0.0360 |

| doubling | gain (nats) | ± SE |
|---|---|---|
| 16->32 | +0.1847 | 0.0099 |
| 32->64 | +0.1490 | 0.0094 |
| 64->128 | +0.0860 | 0.0070 |
| 128->256 | +0.0649 | 0.0068 |
| 256->512 | +0.0312 | 0.0050 |
| 512->1024 | +0.0198 | 0.0046 |

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@W only across models with context ≥ W.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5230 ± 0.0086 nats** (same-doc prefix 4.1576, other-doc prefix 4.6806)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4112 ± 0.0069 nats** (same-doc prefix 4.0667, other-doc prefix 4.4779)
- **cb@512** (window 512, prefix 256, seed 0, 534 windows): benefit **0.2807 ± 0.0053 nats** (same-doc prefix 4.0235, other-doc prefix 4.3042)
- **cb@1024** (window 1024, prefix 512, seed 0, 242 windows): benefit **0.2077 ± 0.0065 nats** (same-doc prefix 4.0207, other-doc prefix 4.2284)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 98.2% | 96%–99% | +5.91 | yes |
| 32 | 96.5% | 94%–98% | +5.23 | yes |
| 64 | 88.5% | 85%–91% | +4.63 | yes |
| 96 | 86.5% | 83%–90% | +3.99 | yes |
| 128 | 76.2% | 72%–80% | +3.49 | yes |
| 160 | 80.2% | 76%–84% | +3.40 | yes |
| 192 | 73.5% | 69%–78% | +3.23 | yes |
| 224 | 70.0% | 65%–74% | +2.92 | yes |
| 256 | 63.2% | 58%–68% | +2.67 | yes |
| 320 | 54.8% | 50%–60% | +2.12 | yes |
| 384 | 49.2% | 44%–54% | +1.86 | yes |
| 448 | 44.8% | 40%–50% | +1.67 | yes |
| 496 | 48.2% | 43%–53% | +1.82 | yes |
| 640 | 29.0% | 25%–34% | +0.98 | yes |
| 768 | 29.8% | 25%–34% | +1.02 | yes |
| 896 | 32.8% | 28%–37% | +1.09 | yes |
| 992 | 34.5% | 30%–39% | +1.28 | yes |

## Inference (this eval run; see evals/inference.md for the controlled comparison)

- device: mps
- prefill, full context: 53.65 ms
- decode: 21.2 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.244 GB

## Generation samples

20 prompts x 5 draws, 256 new tokens, T=0.7, top-k 40, stop at EOS, seeds 20260929 + 1000*prompt + draw. The samples themselves: this run's Samples button, or evals/samples.md next to every other model.

| rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | topic span | EOS |
|---|---|---|---|---|---|---|
| 0.538 / 0.856 | 26/100 (18%–35%) | token 191 | 0.310 / 0.464 | 25% | 119 tokens | 3/100 |

| prompt | rep4 median | loops | topic span (median) |
|---|---|---|---|
| definition | 0.53 | 3/5 | 0 |
| biography | 0.15 | 1/5 | 255 |
| science_explainer | 0.44 | 1/5 | 31 |
| instructional | 0.66 | 1/5 | 62 |
| bullet_list | 0.91 | 2/5 | 0 |
| numbered_list | 0.67 | 1/5 | 58 |
| enumeration | 0.66 | 2/5 | 0 |
| long_dependency | 0.63 | 1/5 | 252 |
| attribution | 0.48 | 2/5 | 77 |
| numeric_units | 0.74 | 1/5 | 251 |
| agreement_gap | 0.37 | 0/5 | 109 |
| history | 0.41 | 0/5 | 255 |
| anatomy | 0.68 | 0/5 | 254 |
| geography | 0.41 | 0/5 | 15 |
| math_definition | 0.50 | 1/5 | 40 |
| environment | 0.53 | 3/5 | 200 |
| recipe | 0.58 | 1/5 | 213 |
| literature | 0.47 | 1/5 | 241 |
| technology | 0.45 | 2/5 | 128 |
| economics | 0.64 | 3/5 | 253 |

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 6857.8
- train_sec: 6290.1
- eval_sec: 567.7
- train_tokens_per_sec: 52094.4
- peak_mem_gb: 3.927
- peak_mem_kind: cuda max_memory_allocated
