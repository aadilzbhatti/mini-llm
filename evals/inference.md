# Inference benchmark

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-10-01T23:36:46Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, no KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s (uncached, full context) |
|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 15.31 (14.47–21.95) | 15.44 (14.84–16.41) | 53.05 (51.15–55.85) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 13.99 (13.44–18.18) | 14.45 (13.41–15.58) | 56.47 (53.85–57.08) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 16.49 (15.82–23.13) | 13.56 (13.02–18.46) | 49.33 (40.10–50.45) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 19.28 (18.14–28.04) | 14.82 (14.54–20.08) | 45.39 (39.67–45.99) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 22.70 (21.94–53.17) | 13.38 (12.73–17.92) | 35.82 (34.69–36.37) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 43.17 (41.78–50.88) | 14.54 (13.18–19.64) | 16.62 (15.53–16.83) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 106.20 (85.22–132.08) | 33.65 (28.01–39.91) | 7.91 (7.28–8.13) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 91.80 (87.84–111.38) | 32.02 (29.41–40.36) | 8.09 (7.62–8.35) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 106.85 (89.67–123.82) | 33.68 (29.36–39.31) | 8.05 (7.49–8.27) |
