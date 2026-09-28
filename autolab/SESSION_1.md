# Autolab session report

Generated 2026-09-28T03:21+00:00 by `autolab session-report` (the daemon refreshes it hourly). Active session: `s2+data40k`.

## Headline

- Best: **s2+data40k/p6**, full val loss **4.4983** over 3 seeds on `data40k` at 81,920,000 tokens.
- Start at the same budget: `s2-82m` p0 4.7522 (σ 0.0328, `data20k`). Owner's best single run for reference: 4.7118 (seed 42, data20k).
- Improvement: **0.2538** nats (7.7σ of the starting seed noise).

## How we got here

### Session `s1-21m` — data `data20k`, full budget 20,971,520 tokens, seed noise σ 0.0426

- 2026-09-27T07:59 incumbent **p0** 5.3651

### Session `s2-82m` — data `data20k`, full budget 81,920,000 tokens, seed noise σ 0.0328

- 2026-09-27T09:30 incumbent **p0** 4.7522
- 2026-09-27T21:06 incumbent **p6** 4.6636 — This is a hyperparameter-only change to p2: peak lr goes from 1.2e-3 to 2e-3, and warmup goes from 100 to 300 steps. Mechanism: the evaluator fixes the batch at
- data check `data40k` on p0: helped: 4.6433 vs 4.7522 (-3.3σ; bar −0.0656)

### Session `s2+data40k` — data `data40k`, full budget 81,920,000 tokens, seed noise σ 0.0306

- 2026-09-27T20:35 incumbent **p0** 4.6433
- 2026-09-27T22:20 incumbent **p1** 4.5793 — Ported from s2-82m/p6 (4.6636 there): This is a hyperparameter-only change to p2: peak lr goes from 1.2e-3 to 2e-3, and warmup goes from 100 to 300 steps. Mecha
- 2026-09-28T01:28 incumbent **p6** 4.4983 — The 'current program' shown (p0) is missing two changes already validated in the lineage: (1) using fused scaled_dot_product_attention for the causal self-atten
- data check `data80k` on p0: did not help: 4.6488 vs 4.6433 (+0.2σ; bar −0.0613)

## Best so far after each full-budget evaluation (81,920,000 tokens)

| # | when | session/program | full val (1st seed) | best so far |
| --- | --- | --- | --- | --- |
| 1 | 2026-09-27T10:11 | s2-82m/p2 (data20k) | 4.7198 | 4.7198 |
| 2 | 2026-09-27T19:12 | s2-82m/p3 (data20k) | 4.9193 | 4.7198 |
| 3 | 2026-09-27T20:05 | s2-82m/p4 (data20k) | 4.8255 | 4.7198 |
| 4 | 2026-09-27T20:09 | s2-82m/p1 (data20k) | 4.7054 | 4.7054 |
| 5 | 2026-09-27T20:17 | s2-82m/p5 (data20k) | 4.7568 | 4.7054 |
| 6 | 2026-09-27T20:19 | s2-82m/p6 (data20k) | 4.6479 | 4.6479 |
| 7 | 2026-09-27T21:06 | s2-82m/p7 (data20k) | 4.6737 | 4.6479 |
| 8 | 2026-09-27T21:47 | s2+data40k/p1 (data40k) | 4.5765 | 4.5765 |
| 9 | 2026-09-28T00:47 | s2+data40k/p2 (data40k) | 4.5395 | 4.5395 |
| 10 | 2026-09-28T00:50 | s2+data40k/p4 (data40k) | 4.5063 | 4.5063 |
| 11 | 2026-09-28T00:51 | s2+data40k/p5 (data40k) | 4.6362 | 4.5063 |
| 12 | 2026-09-28T00:54 | s2+data40k/p6 (data40k) | 4.4999 | 4.4999 |

## Lineage of the best program

- **s2+data40k/p0** (human) — 4.6433 over 3 seeds; initial
- **s2+data40k/p6** (claude-sonnet-5) — 4.4983 over 3 seeds; n_layer 4→6, lr 0.0012→0.002, warmup_steps 100→300, weight_decay 0.0→0.1; code: attention, optimizer
  - The 'current program' shown (p0) is missing two changes already validated in the lineage: (1) using fused scaled_dot_product_attention for the causal self-attention forward pass during training (numerically identical, just faster/more stable, letting the larger model train within the time cap), and (2) the larger-capacity/higher-LR config (n_layer 6, lr 2e-3, warmup 300) that took p0's 4.6433 down

## Cascade funnel and rejections

- `s1-21m`: 8 children — rejected 7, contender 1 · rejected at static 4, cpu 2, params 1
  - p2 (human) @ cpu: causal-leak: 3 failed: tests/autolab/test_causal_leak.py::test_no_future_leak[0], tests/autolab/test_causal_leak.py::test_no_future_leak[1], tests/autolab/test_causal_leak.py::test_no_future_leak[2]
  - p3 (human) @ cpu: shape: 1 failed: tests/autolab/test_causal_leak.py::test_shapes_and_backward
  - p4 (human) @ static: scope: diff 1: SEARCH text in mini_llm/model.py is not entirely inside one EVOLVE block
  - p5 (human) @ params: 30,814,033 parameters > cap 24,157,946 (1.5x initial)
  - p6 (human) @ static: scope: diff 1: SEARCH text not found in model.py/train.py
  - p7 (human) @ static: mini_llm/model.py:forward_body: uses forbidden name 'targets' (line 17); mini_llm/model.py:forward_body: uses forbidden name 'targets' (line 18)
  - p8 (human) @ static: hparam lr=0.5 outside [1e-05, 0.01]
- `s2-82m`: 7 children — evaluated 4, contender 2, accepted 1
- `s2+data40k`: 6 children — accepted 2, contender 2, rejected 1, evaluated 1 · rejected at cpu 1
  - p3 (claude-sonnet-5) @ cpu: shape: 1 failed: tests/autolab/test_causal_leak.py::test_shapes_and_backward

## Proposer (Claude) statistics

| proposer | proposed | trained | accepted | contender | evaluated | rejected |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | 7 | 7 | 1 | 3 | 3 | 0 |
| claude-sonnet-5 | 4 | 3 | 1 | 1 | 1 | 1 |
| human | 8 | 1 | 0 | 1 | 0 | 7 |
| mutation | 1 | 1 | 0 | 0 | 1 | 0 |
| port | 1 | 1 | 1 | 0 | 0 | 0 |

- Claude calls: 12 (11 ok), $1.15. Failures: exit 1 ×1.
- Fallback mutations (proposer failed): 1.

## Spend

| stage | runs | GPU-hours | $ |
| --- | --- | --- | --- |
| confirmations | 16 | 7.88 | 6.30 |
| full runs | 13 | 6.60 | 5.27 |
| baselines & prep experiments | 25 | 6.23 | 4.98 |
| data checks | 6 | 2.97 | 2.38 |
| screens | 13 | 0.93 | 0.74 |
| data-switch screens | 3 | 0.20 | 0.16 |
| **Modal total** | 76 | 24.81 | **19.83** |
| Claude | 12 calls | | 1.15 |

Modal $ = wall time × GPU list price (L4); CPU/memory charges excluded, so a lower bound.

