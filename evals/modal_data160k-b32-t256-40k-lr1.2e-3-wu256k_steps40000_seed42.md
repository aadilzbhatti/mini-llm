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

Loss on the second half of 400 val windows, given a 64-token prefix from the same document vs from a different document.

- same-document prefix: 4.1427
- other-document prefix: 4.6574
- **benefit: 0.5147 nats**

## Long-range retrieval

Forced choice among 10 single-token words (chance 10%), 60 trials per distance.

| distance | accuracy | log-prob margin | key in context |
|---|---|---|---|
| 16 | 93% | +4.66 | yes |
| 32 | 88% | +3.69 | yes |
| 64 | 73% | +3.20 | yes |
| 96 | 55% | +2.48 | yes |
| 128 | 35% | +1.42 | yes |
| 160 | 53% | +2.01 | yes |
| 192 | 25% | +0.91 | yes |
| 224 | 30% | +1.14 | yes |

## Inference

- device: mps
- prefill, full context: 14.86 ms
- decode: 43.0 tokens/s (no KV cache: each decode step re-runs the (cropped) window)
- memory: 0.117 GB

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
