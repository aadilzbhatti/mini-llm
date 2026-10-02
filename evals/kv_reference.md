# KV cache reference: absolute position embeddings

A fixed record of what the KV cache alone buys with this model's learned absolute position embeddings, to compare later changes against (e.g. RoPE). Regenerate with `uv run mini-llm-bench --kv-reference`.

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-10-02T17:22:55Z
- protocol: absolute position embeddings; 15 interleaved rounds (order rotated); prefill = one forward over c tokens filling the cache (last-position logits); decode@c = 64 greedy tokens from a (c - 64)-token prompt, without / with the KV cache; past the window = 64 tokens from a full window (the cache is refilled every step); fp32; median (p10-p90)

**Past the window**: once the context fills `block_size`, every token's absolute position shifts by one each step, so the cached keys and values go stale and the cache is refilled from scratch every step. That row is what relative positions (RoPE) would let the cache avoid.

## modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42

d512 · 4 layers · 8 heads · block_size 1024 · cache 16 KiB per token, 16.0 MiB allocated (the full window, reused)

| context | prefill ms | decode tok/s, no cache | decode tok/s, KV cache | speedup | cache in use (MiB) |
|---|---|---|---|---|---|
| 128 | 31.8 (29.0–35.2) | 90.7 (84.0–105.2) | 84.1 (78.3–100.5) | 0.9× | 2.0 |
| 256 | 27.1 (25.0–28.0) | 82.2 (76.4–94.7) | 86.0 (80.2–102.9) | 1.0× | 4.0 |
| 512 | 35.1 (29.0–40.6) | 60.4 (58.4–62.5) | 91.0 (82.9–107.8) | 1.5× | 8.0 |
| 1024 | 54.1 (45.4–57.8) | 19.9 (17.7–22.2) | 89.3 (82.5–103.4) | 4.5× | 16.0 |
| past the window | – | 23.3 (21.4–25.9) | 21.8 (19.5–24.7) | 0.9× | 16.0 |

## modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42

d256 · 4 layers · 4 heads · block_size 1024 · cache 8 KiB per token, 8.0 MiB allocated (the full window, reused)

| context | prefill ms | decode tok/s, no cache | decode tok/s, KV cache | speedup | cache in use (MiB) |
|---|---|---|---|---|---|
| 128 | 27.9 (26.7–34.5) | 129.9 (121.2–148.0) | 120.4 (107.5–133.9) | 0.9× | 1.0 |
| 256 | 14.1 (13.7–14.7) | 114.6 (98.9–125.8) | 120.3 (113.5–135.1) | 1.0× | 2.0 |
| 512 | 21.1 (20.5–21.6) | 118.8 (109.9–127.9) | 128.3 (119.9–135.2) | 1.1× | 4.0 |
| 1024 | 32.3 (28.4–33.6) | 54.1 (53.3–54.7) | 131.9 (117.1–139.9) | 2.4× | 8.0 |
| past the window | – | 63.9 (62.3–65.6) | 60.8 (58.0–62.4) | 1.0× | 8.0 |

