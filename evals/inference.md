# Inference benchmark

- device: mps (Aadils-Mac-mini.local, torch 2.14.0)
- at: 2026-10-09T03:14:25Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, without and with the KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s, no cache | decode tok/s, KV cache |
|---|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 5.54 (5.42–7.76) | 5.45 (5.37–5.62) | 236.96 (235.33–238.79) | 252.88 (242.06–258.14) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 5.41 (5.33–7.38) | 5.36 (5.29–5.57) | 238.04 (201.14–239.13) | 252.72 (243.27–259.59) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 7.06 (6.93–8.90) | 5.53 (5.43–7.20) | 175.65 (174.89–176.67) | 252.24 (242.15–254.80) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 11.46 (9.03–26.96) | 9.05 (6.49–16.66) | 163.86 (154.30–171.71) | 250.20 (248.95–252.82) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 12.14 (11.21–33.99) | 5.85 (5.64–8.23) | 92.16 (90.57–93.70) | 224.14 (220.76–231.15) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 24.69 (24.38–43.18) | 6.80 (6.42–9.67) | 38.84 (0.66–39.80) | 211.41 (203.66–224.23) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 48.18 (47.88–66.86) | 13.59 (12.13–24.71) | 19.38 (16.70–20.19) | 158.55 (156.05–162.33) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 48.23 (47.83–53.09) | 14.83 (13.29–25.22) | 19.46 (19.27–20.11) | 158.29 (156.60–161.60) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42 | 1024 | 50.12 (49.72–52.17) | 15.56 (12.70–25.52) | 18.82 (18.69–19.68) | 134.94 (131.48–137.68) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 48.40 (48.02–50.34) | 13.87 (13.21–25.68) | 19.36 (19.29–20.22) | 159.03 (154.93–160.54) |
| modal_data640k-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42 | 1024 | 94.12 (91.16–96.64) | 17.50 (16.75–17.78) | 10.74 (10.72–11.26) | 172.59 (171.90–175.26) |
| modal_data640k-e768h12l8-160k-lr3e-4-rope-fused-cont30k-rlr5e-5_steps30000_seed42 | 1024 | 96.70 (93.15–101.82) | 17.56 (16.80–17.79) | 10.73 (10.70–10.94) | 172.06 (171.54–174.15) |
