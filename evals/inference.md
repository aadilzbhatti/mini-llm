# Inference benchmark

- device: mps (Aadils-MacBook-Air.local, torch 2.14.0)
- at: 2026-10-02T04:57:27Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, no KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s (uncached, full context) |
|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 13.49 (13.14–19.95) | 13.42 (12.95–14.81) | 59.33 (56.42–60.49) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 12.90 (12.36–17.84) | 12.63 (12.01–13.01) | 60.66 (60.35–61.26) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 14.92 (14.55–19.10) | 12.50 (12.25–17.26) | 54.42 (53.12–55.23) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 16.83 (16.41–22.42) | 13.71 (13.03–18.36) | 50.20 (48.96–50.58) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 20.75 (20.47–45.95) | 12.25 (11.87–16.81) | 39.13 (38.26–39.69) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 39.74 (39.21–44.44) | 12.54 (12.04–17.19) | 19.51 (18.76–19.75) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 92.59 (75.83–111.89) | 28.93 (25.51–33.00) | 9.16 (8.73–9.33) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 81.15 (78.84–102.97) | 26.69 (25.51–31.26) | 9.45 (9.12–9.55) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 94.21 (79.73–100.47) | 28.61 (25.95–33.61) | 9.25 (8.88–9.65) |
