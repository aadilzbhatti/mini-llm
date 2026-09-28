# Autolab session report

Generated 2026-09-28T07:42+00:00 by `autolab session-report` (the daemon refreshes it hourly). Active session: `s2+data40k@122M`.

## Headline

- Best: **s2+data40k/p10**, full val loss **4.3754** over 3 seeds on `data40k` at 81,920,000 tokens.
- Start at the same budget: `s2-82m` p0 4.7522 (σ 0.0328, `data20k`). Owner's best single run for reference: 4.7118 (seed 42, data20k).
- Improvement: **0.3767** nats (11.5σ of the starting seed noise).

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
- 2026-09-28T06:16 incumbent **p10** 4.3754 — Three linked changes, all aimed at letting the model use a faster learning rate safely. (1) Fix the embedding init scale. xavier_uniform on the 50257×256 token 
- data check `data80k` on p0: did not help: 4.6488 vs 4.6433 (+0.2σ; bar −0.0613)
- data check `data40k` on p6: helped: 4.3942 vs 4.4983 (-3.4σ; bar −0.0613)

### Session `s2+data40k@122M` — data `data40k`, full budget 122,880,000 tokens, seed noise σ 0.0069

- 2026-09-28T06:04 incumbent **p0** 4.3942

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
| 13 | 2026-09-28T05:35 | s2+data40k/p9 (data40k) | 4.4883 | 4.4883 |
| 14 | 2026-09-28T05:39 | s2+data40k/p7 (data40k) | 4.4939 | 4.4883 |
| 15 | 2026-09-28T05:41 | s2+data40k/p10 (data40k) | 4.3738 | 4.3738 |
| 16 | 2026-09-28T06:02 | s2+data40k/p11 (data40k) | 4.7166 | 4.3738 |
| 17 | 2026-09-28T06:02 | s2+data40k/p14 (data40k) | 4.7065 | 4.3738 |
| 18 | 2026-09-28T06:09 | s2+data40k/p16 (data40k) | 4.4706 | 4.3738 |
| 19 | 2026-09-28T06:29 | s2+data40k/p17 (data40k) | 4.4344 | 4.3738 |
| 20 | 2026-09-28T06:36 | s2+data40k/p18 (data40k) | 4.4803 | 4.3738 |

## Lineage of the best program

- **s2+data40k/p0** (human) — 4.6433 over 3 seeds; initial
- **s2+data40k/p6** (claude-sonnet-5) — 4.4983 over 3 seeds; n_layer 4→6, lr 0.0012→0.002, warmup_steps 100→300, weight_decay 0.0→0.1; code: attention, optimizer
  - The 'current program' shown (p0) is missing two changes already validated in the lineage: (1) using fused scaled_dot_product_attention for the causal self-attention forward pass during training (numerically identical, just faster/more stable, letting the larger model train within the time cap), and (2) the larger-capacity/higher-LR config (n_layer 6, lr 2e-3, warmup 300) that took p0's 4.6433 down
- **s2+data40k/p10** (claude-opus-5-5) — 4.3754 over 3 seeds; lr 0.002→0.003, warmup_steps 300→400; code: model_init, optimizer
  - Three linked changes, all aimed at letting the model use a faster learning rate safely. (1) Fix the embedding init scale. xavier_uniform on the 50257×256 token table gives std ≈ 0.0063. On the 128×256 position table it gives std ≈ 0.07, so at init the position signal is about 10× larger than the token signal in the residual stream. Rare tokens get few updates, so their rows stay near that tiny sca

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
- `s2+data40k`: 18 children — evaluated 7, rejected 5, accepted 3, contender 3 · rejected at cpu 4, screen 1
  - p3 (claude-sonnet-5) @ cpu: shape: 1 failed: tests/autolab/test_causal_leak.py::test_shapes_and_backward
  - p8 (claude-opus-5-5) @ cpu: shape: 1 failed: tests/autolab/test_causal_leak.py::test_shapes_and_backward
  - p12 (claude-opus-5-5) @ screen: screen 6.3576 > incumbent 5.7621 + margin 0.25
  - p13 (claude-sonnet-5) @ cpu: shape: 1 failed: tests/autolab/test_causal_leak.py::test_shapes_and_backward
  - p15 (claude-opus-5-5) @ cpu: tests: 1 failed: tests/test_model.py::test_overfits_one_batch
- `s2+data40k@122M`: 3 children — running 2, evaluated 1

## Proposer (Claude) statistics

| proposer | proposed | trained | accepted | contender | evaluated | rejected |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | 16 | 14 | 2 | 3 | 8 | 3 |
| claude-sonnet-5 | 7 | 5 | 1 | 2 | 2 | 2 |
| human | 8 | 1 | 0 | 1 | 0 | 7 |
| mutation | 1 | 1 | 0 | 0 | 1 | 0 |
| port | 4 | 4 | 1 | 0 | 1 | 0 |

- Claude calls: 25 (23 ok), $2.88. Failures: exit 1 ×1, rate_limited: {"is_error":true,"duration_api_ms":63194,"num_ ×1.
- Fallback mutations (proposer failed): 1.

## Spend

| stage | runs | GPU-hours | $ |
| --- | --- | --- | --- |
| full runs | 24 | 13.49 | 10.78 |
| confirmations | 24 | 10.03 | 8.01 |
| baselines & prep experiments | 25 | 6.23 | 4.98 |
| data checks | 9 | 5.40 | 4.32 |
| screens | 28 | 2.08 | 1.66 |
| data-switch screens | 3 | 0.20 | 0.16 |
| **Modal total** | 113 | 37.43 | **29.92** |
| Claude | 25 calls | | 2.88 |

Modal $ = wall time × GPU list price (L4); CPU/memory charges excluded, so a lower bound.

## Pauses

- 2026-09-28T05:11: 4 proposals in a row rejected before training (tests: 1 failed: tests/autolab/test_modal_backend.py::test_cost_cap_refuses) (until 2026-09-28T11:11)
- 2026-09-28T06:03: Claude usage limit: claude exited 1: {"is_error":true,"duration_api_ms":63194,"num_turns":6,"stop_reason":"tool_use","session_id":"87264ec6-6dc5-4f49-a4c4-3bd504dd1bb7","total_cost_usd":0.178043,"usag (until 2026-09-28T07:02)

