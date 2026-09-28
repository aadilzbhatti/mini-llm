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

Loss on the second half of 400 val windows, given a 64-token prefix from the same document vs from a different document.

- same-document prefix: 4.4739
- other-document prefix: 4.9149
- **benefit: 0.4410 nats**

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 60 trials per distance.

| distance | accuracy | log-prob margin | key in context |
|---|---|---|---|
| 16 | 87% | +3.48 | yes |
| 32 | 80% | +3.21 | yes |
| 64 | 62% | +2.39 | yes |
| 96 | 55% | +1.86 | yes |
| 128 | 22% | +0.79 | yes |
| 160 | 40% | +1.54 | yes |
| 192 | 25% | +1.04 | yes |
| 224 | 27% | +1.01 | yes |

## Inference

- device: mps
- prefill, full context: 26.02 ms
- decode: 30.6 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.117 GB

## Training systems (recorded by the run)

- not recorded (run predates mini_llm.systems)
