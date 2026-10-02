# Inference benchmark

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-10-02T17:16:48Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, without and with the KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s, no cache | decode tok/s, KV cache |
|---|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 10.71 (7.67–15.22) | 7.88 (7.59–8.12) | 120.36 (109.45–123.44) | 126.35 (103.36–128.20) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 9.84 (7.78–15.47) | 8.04 (7.72–11.41) | 116.38 (106.06–120.68) | 126.30 (115.52–127.54) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 10.67 (10.25–22.44) | 8.39 (7.97–12.18) | 122.41 (109.27–122.80) | 121.70 (115.28–130.50) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 10.65 (10.26–22.72) | 8.30 (7.88–12.29) | 112.96 (112.14–114.45) | 119.80 (111.82–121.68) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 14.01 (13.64–46.30) | 7.24 (6.98–11.81) | 67.70 (66.61–67.97) | 132.43 (119.03–135.58) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 32.25 (31.89–51.34) | 7.35 (7.22–11.89) | 23.36 (22.95–24.24) | 132.42 (105.82–141.83) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 74.23 (62.42–84.31) | 15.10 (14.59–28.11) | 10.37 (10.13–10.98) | 98.22 (97.92–100.20) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 63.02 (60.50–81.99) | 16.97 (13.06–25.98) | 11.02 (10.44–11.15) | 96.88 (94.09–97.86) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 73.08 (63.58–84.41) | 15.34 (14.11–27.70) | 10.54 (10.41–11.08) | 96.81 (89.20–97.85) |
