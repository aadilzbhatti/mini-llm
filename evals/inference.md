# Inference benchmark

- device: mps (Aadils-Mac-mini.local, torch 2.14.0)
- at: 2026-10-07T23:57:16Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, without and with the KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s, no cache | decode tok/s, KV cache |
|---|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 5.57 (5.40–15.99) | 5.52 (5.39–5.57) | 231.78 (227.54–236.09) | 252.02 (250.74–255.46) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 5.47 (5.38–7.34) | 5.44 (5.37–5.68) | 235.86 (232.26–238.91) | 250.96 (247.58–252.77) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 7.06 (6.94–9.43) | 5.56 (5.39–7.34) | 171.54 (168.32–174.53) | 249.70 (233.31–258.08) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 7.15 (6.96–26.50) | 5.58 (5.45–7.74) | 175.81 (170.18–176.80) | 246.22 (230.11–254.48) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 11.79 (11.25–13.92) | 5.78 (5.66–7.69) | 94.26 (93.66–94.51) | 229.71 (220.46–237.83) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 24.65 (24.45–27.11) | 6.95 (6.81–10.17) | 40.13 (38.58–40.20) | 211.51 (206.10–220.99) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 47.78 (47.40–56.39) | 13.63 (13.27–24.91) | 20.44 (19.82–20.57) | 158.49 (156.81–162.05) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 47.78 (47.53–51.13) | 13.60 (13.01–23.53) | 20.43 (19.78–20.61) | 159.05 (149.77–160.78) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42 | 1024 | 50.33 (50.01–61.77) | 15.97 (14.34–26.47) | 19.58 (19.07–19.64) | 129.08 (124.86–129.61) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 47.68 (47.41–49.75) | 13.50 (12.74–23.64) | 20.26 (19.88–20.64) | 159.54 (156.10–160.56) |
