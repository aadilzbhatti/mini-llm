# Evals: modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42

- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,105,297
- step: 40000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.2601**
- by_position@128: pos 0-15: 4.7788, pos 16-63: 4.2646, pos 64-127: 4.1271

## Context benefit

Loss on the second half of each val window, given the first half from the same document vs from a different one (one window per eligible document, identical windows for every model). Compare cb@128 across all models; cb@256 only across models with context ≥ 256.

- **cb@128** (window 128, prefix 64, seed 0, 903 windows): benefit **0.5074 ± 0.0081 nats** (same-doc prefix 4.0767, other-doc prefix 4.5841)

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 400 trials per distance, identical trials for every model.

| distance | accuracy | 95% CI | log-prob margin | key in context |
|---|---|---|---|---|
| 16 | 95.8% | 93%–97% | +4.94 | yes |
| 32 | 78.8% | 74%–82% | +3.64 | yes |
| 64 | 39.0% | 34%–44% | +1.83 | yes |
| 96 | 21.5% | 18%–26% | +0.69 | yes |
| 128 | 8.2% | 6%–11% | -0.06 | no |
| 160 | 11.5% | 9%–15% | +0.10 | no |
| 192 | 9.5% | 7%–13% | +0.01 | no |
| 224 | 7.2% | 5%–10% | -0.09 | no |
| 256 | 10.8% | 8%–14% | -0.05 | no |
| 320 | 8.2% | 6%–11% | -0.09 | no |
| 384 | 7.5% | 5%–10% | -0.09 | no |
| 448 | 9.0% | 7%–12% | +0.03 | no |
| 496 | 10.0% | 7%–13% | +0.11 | no |

## Inference

- device: mps
- prefill, full context: 10.93 ms
- decode: 77.4 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.088 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
