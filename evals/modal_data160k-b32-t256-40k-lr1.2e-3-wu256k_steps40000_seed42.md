# Evals: modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42

- config: {'vocab_size': 50257, 'block_size': 256, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,138,065
- step: 40000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.2809**
- by_position@128: pos 0-15: 4.8422, pos 16-63: 4.2859, pos 64-127: 4.1369
- full_val@256: **4.1774**
- by_position@256: pos 0-15: 4.8387, pos 16-63: 4.2906, pos 64-127: 4.1340, pos 128-255: 4.0739

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@128 across all models; cb@256 only across models with context ≥ 256.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5291 ± 0.0083 nats** (same-doc prefix 4.0856, other-doc prefix 4.6147)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.4018 ± 0.0061 nats** (same-doc prefix 4.0126, other-doc prefix 4.4144)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 95.5% | 93%–97% | +4.60 | yes |
| 32 | 82.8% | 79%–86% | +3.65 | yes |
| 64 | 68.0% | 63%–72% | +2.90 | yes |
| 96 | 56.0% | 51%–61% | +2.18 | yes |
| 128 | 40.0% | 35%–45% | +1.51 | yes |
| 160 | 40.2% | 36%–45% | +1.51 | yes |
| 192 | 31.5% | 27%–36% | +1.21 | yes |
| 224 | 24.2% | 20%–29% | +0.92 | yes |
| 256 | 11.8% | 9%–15% | +0.06 | no |
| 320 | 10.0% | 7%–13% | -0.00 | no |
| 384 | 9.2% | 7%–12% | -0.03 | no |
| 448 | 10.0% | 7%–13% | -0.03 | no |
| 496 | 11.2% | 9%–15% | +0.07 | no |

## Inference

- device: mps
- prefill, full context: 13.42 ms
- decode: 64.9 tokens/s (medians of 21 prefills / 7 decode runs of 64 tokens; no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.104 GB

## Training systems (recorded by the run)

- device: NVIDIA L4
- world_size: 2
- steps: 40000
- tokens: 327680000
- wall_sec: 4386.6
- train_sec: 3864.4
- eval_sec: 522.3
- train_tokens_per_sec: 84795.4
- peak_mem_gb: 3.662
- peak_mem_kind: cuda max_memory_allocated
