# KV cache reference: absolute position embeddings

A fixed record of what the KV cache alone buys with this model's learned absolute position embeddings, to compare later changes against (e.g. RoPE). Regenerate with `uv run mini-llm-bench --kv-reference`.

- device: mps (Aadils-Mac-mini.local, torch 2.14.0)
- at: 2026-10-08T16:49:25Z
- protocol: 15 interleaved rounds (order rotated); prefill = one forward over c tokens filling the cache (last-position logits); decode@c = 64 greedy tokens from a (c - 64)-token prompt, without / with the KV cache; past the window = 64 tokens from a full window (absolute PE: the cache is refilled every step; RoPE: the cache rolls); fp32; median (p10-p90)

**Past the window**: once the context fills `block_size`, every token's absolute position shifts by one each step, so with absolute PE the cached keys and values go stale and the cache is refilled from scratch every step. RoPE encodes position through rotations such that Q·K attention depends on relative positional offsets, so retained cached K/V do not need to be re-positioned when the window rolls.

## modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 (absolute PE + KV cache)

d512 · 4 layers · 8 heads · block_size 1024 · cache 16 KiB per token, 16.0 MiB allocated (the full window, reused)

| context | prefill ms | decode tok/s, no cache | decode tok/s, KV cache | speedup | cache in use (MiB) |
|---|---|---|---|---|---|
| 128 | 9.3 (8.0–33.6) | 136.6 (129.8–137.5) | 179.3 (167.8–181.9) | 1.3× | 2.0 |
| 256 | 8.5 (8.4–8.7) | 87.9 (87.7–88.9) | 176.4 (171.8–180.0) | 2.0× | 4.0 |
| 512 | 13.6 (13.5–13.8) | 47.4 (46.6–47.4) | 163.9 (148.4–169.3) | 3.5× | 8.0 |
| 1024 | 29.1 (28.8–29.2) | 20.4 (20.3–20.5) | 157.8 (151.7–160.9) | 7.7× | 16.0 |
| past the window | – | 21.2 (21.0–21.2) | 37.8 (37.5–38.0) | 1.8× | 16.0 |

## modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42 (RoPE + rolling KV cache)

d512 · 4 layers · 8 heads · block_size 1024 · cache 16 KiB per token, 16.0 MiB allocated (the full window, reused)

| context | prefill ms | decode tok/s, no cache | decode tok/s, KV cache | speedup | cache in use (MiB) |
|---|---|---|---|---|---|
| 128 | 26.2 (9.2–37.4) | 118.9 (117.0–124.7) | 148.6 (143.5–151.3) | 1.2× | 2.0 |
| 256 | 9.6 (9.2–11.0) | 82.4 (82.1–82.6) | 147.4 (141.8–150.9) | 1.8× | 4.0 |
| 512 | 14.9 (14.8–15.0) | 45.0 (43.0–45.0) | 139.2 (137.7–141.0) | 3.1× | 8.0 |
| 1024 | 31.0 (30.5–31.8) | 19.8 (19.7–19.8) | 134.4 (131.3–136.8) | 6.8× | 16.0 |
| past the window | – | 20.4 (20.3–20.5) | 184.2 (181.2–189.8) | 9.0× | 16.0 |

