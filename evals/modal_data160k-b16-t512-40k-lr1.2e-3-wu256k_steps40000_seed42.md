# Evals: modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42

- config: {'vocab_size': 50257, 'block_size': 512, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,203,601
- step: 40000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.3381**
- by_position@128: pos 0-15: 4.9075, pos 16-63: 4.3475, pos 64-127: 4.1888
- full_val@256: **4.2286**
- by_position@256: pos 0-15: 4.9038, pos 16-63: 4.3544, pos 64-127: 4.1863, pos 128-255: 4.1181
- full_val@512: **4.1550**
- by_position@512: pos 0-15: 4.8916, pos 16-63: 4.3686, pos 64-127: 4.1827, pos 128-255: 4.1255, pos 256-511: 4.0767

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@128 across all models; cb@256 only across models with context ≥ 256.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5134 ± 0.0085 nats** (same-doc prefix 4.1464, other-doc prefix 4.6598)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4046 ± 0.0062 nats** (same-doc prefix 4.0576, other-doc prefix 4.4621)

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

## Inference

- device: mps
- prefill, full context: 19.29 ms
- decode: 54.8 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.145 GB

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
