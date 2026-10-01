# Inference benchmark

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-10-01T09:06:06Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, no KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s (uncached, full context) |
|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 39.16 (33.19–62.97) | 46.65 (44.86–49.35) | 13.88 (13.73–14.14) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 46.05 (40.89–135.00) | 47.00 (40.73–48.86) | 14.01 (13.55–15.71) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 54.79 (48.97–82.28) | 47.73 (40.22–75.45) | 12.91 (12.74–13.34) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 44.68 (24.85–66.68) | 46.55 (33.61–71.05) | 13.22 (13.04–14.67) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 63.43 (50.17–182.08) | 46.74 (31.22–69.46) | 11.14 (10.93–12.67) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 75.47 (63.35–139.63) | 47.49 (38.01–89.97) | 9.40 (9.14–10.05) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 122.46 (118.83–222.01) | 87.85 (72.07–94.86) | 6.22 (6.16–6.26) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 123.65 (119.82–231.68) | 91.14 (74.09–98.01) | 6.16 (5.59–6.21) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 124.26 (120.21–177.05) | 91.10 (76.46–124.75) | 6.22 (4.50–6.27) |
