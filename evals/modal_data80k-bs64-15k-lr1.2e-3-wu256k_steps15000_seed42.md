# Evals: modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42

- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- params: 16,105,297
- step: 15000
- val: data/data20k/val.pt

## Quality

- full_val@128: **4.4679**
- by_position@128: pos 0-15: 4.9445, pos 16-63: 4.4731, pos 64-127: 4.3448

## Context benefit

Loss on the second half of 400 val windows, given a 64-token prefix from the same document vs from a different document.

- same-document prefix: 4.3569
- other-document prefix: 4.8059
- **benefit: 0.4490 nats**

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 60 trials per distance.

| distance | accuracy | log-prob margin | key in context |
|---|---|---|---|
| 16 | 90% | +3.47 | yes |
| 32 | 82% | +2.90 | yes |
| 64 | 60% | +2.31 | yes |
| 96 | 45% | +1.80 | yes |
| 128 | 13% | -0.08 | yes |
| 160 | 12% | +0.21 | no |
| 192 | 3% | -0.20 | no |
| 224 | 18% | +0.25 | no |

## Inference

- device: mps
- prefill, full context: 18.38 ms
- decode: 38.7 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.088 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
