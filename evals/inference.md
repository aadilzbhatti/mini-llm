# Inference benchmark

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-09-29T07:38:58Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, no KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s (uncached, full context) |
|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 16.75 (15.21–64.42) | 15.55 (13.98–16.98) | 48.58 (44.14–53.29) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 13.80 (12.58–37.78) | 13.34 (12.90–14.43) | 56.22 (49.97–57.11) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 15.90 (15.01–21.62) | 14.02 (12.56–41.66) | 45.27 (43.64–50.47) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 17.29 (14.92–23.33) | 14.18 (12.33–19.95) | 45.30 (38.70–50.09) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 21.25 (20.67–27.30) | 13.29 (12.51–18.19) | 35.10 (34.33–36.34) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 40.99 (40.16–46.27) | 14.55 (13.26–51.78) | 17.41 (17.07–18.23) |
