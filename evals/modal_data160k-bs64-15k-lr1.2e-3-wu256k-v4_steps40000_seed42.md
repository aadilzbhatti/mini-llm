# Evals: modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42

- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,105,297
- step: 40000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.2601**
- by_position@128: pos 0-15: 4.7788, pos 16-63: 4.2646, pos 64-127: 4.1271

## Context benefit

Loss on the second half of 400 val windows, given a 64-token prefix from the same document vs from a different document.

- same-document prefix: 4.1319
- other-document prefix: 4.6275
- **benefit: 0.4956 nats**

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 60 trials per distance.

| distance | accuracy | log-prob margin | key in context |
|---|---|---|---|
| 16 | 95% | +4.62 | yes |
| 32 | 82% | +3.83 | yes |
| 64 | 47% | +2.17 | yes |
| 96 | 30% | +1.33 | yes |
| 128 | 12% | -0.14 | no |
| 160 | 13% | +0.25 | no |
| 192 | 7% | -0.08 | no |
| 224 | 15% | +0.32 | no |

## Inference

- device: mps
- prefill, full context: 11.05 ms
- decode: 59.6 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.088 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
