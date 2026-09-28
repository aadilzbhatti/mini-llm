# Evals: modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42

- config: {'vocab_size': 50257, 'block_size': 256, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,138,065
- step: 15000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.5967**
- by_position@128: pos 0-15: 5.1385, pos 16-63: 4.5992, pos 64-127: 4.4595
- full_val@256: **4.4990**
- by_position@256: pos 0-15: 5.1321, pos 16-63: 4.6061, pos 64-127: 4.4560, pos 128-255: 4.4013

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@128 across all models; cb@256 only across models with context ≥ 256.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.4611 ± 0.0075 nats** (same-doc prefix 4.4129, other-doc prefix 4.8740)
- **cb@256** (window 256, prefix 128, seed 0, 762 windows): benefit **0.3602 ± 0.0057 nats** (same-doc prefix 4.3427, other-doc prefix 4.7029)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 87.0% | 83%–90% | +3.50 | yes |
| 32 | 77.2% | 73%–81% | +3.04 | yes |
| 64 | 60.0% | 55%–65% | +2.19 | yes |
| 96 | 48.5% | 44%–53% | +1.62 | yes |
| 128 | 27.3% | 23%–32% | +1.04 | yes |
| 160 | 31.2% | 27%–36% | +1.11 | yes |
| 192 | 23.8% | 20%–28% | +0.80 | yes |
| 224 | 20.8% | 17%–25% | +0.75 | yes |
| 256 | 11.5% | 9%–15% | +0.07 | no |
| 320 | 10.5% | 8%–14% | -0.02 | no |
| 384 | 9.2% | 7%–12% | -0.07 | no |
| 448 | 11.8% | 9%–15% | +0.04 | no |
| 496 | 10.0% | 7%–13% | +0.12 | no |

## Inference

- device: mps
- prefill, full context: 17.82 ms
- decode: 30.2 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.117 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
