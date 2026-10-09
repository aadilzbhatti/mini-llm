# Inference benchmark

- device: mps (Aadils-Mac-mini.local, torch 2.14.0)
- at: 2026-10-09T14:22:46Z
- protocol: 7 interleaved rounds (order rotated), 5 prefills per model per round, decode 64 greedy tokens from a (block_size - 64)-token prompt, without and with the KV cache, synchronized timing, warmed up

Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the comparison is clean.

| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s, no cache | decode tok/s, KV cache |
|---|---|---|---|---|---|
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 5.52 (5.29–17.22) | 5.38 (5.25–5.61) | 237.59 (228.34–238.63) | 253.99 (247.25–257.10) |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 5.38 (5.30–7.45) | 5.36 (5.22–5.45) | 238.00 (231.32–238.56) | 253.22 (247.03–254.35) |
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 7.05 (6.91–9.11) | 5.48 (5.31–7.34) | 165.68 (163.28–178.85) | 250.51 (246.25–255.69) |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 8.94 (7.01–120.38) | 7.14 (5.36–11.81) | 173.12 (166.16–179.15) | 249.83 (249.09–251.94) |
| modal_data160k-b16-t512-40k-lr1.2e-3-wu256k_steps40000_seed42 | 512 | 11.61 (11.20–13.95) | 5.74 (5.56–7.63) | 91.93 (90.95–94.32) | 226.50 (224.14–232.98) |
| modal_data160k-b8-t1024-40k-lr1.2e-3-wu256k_steps40000_seed42 | 1024 | 24.73 (24.56–50.10) | 6.76 (6.20–10.23) | 39.65 (-0.06–40.14) | 213.51 (207.44–231.38) |
| modal_data160k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 48.35 (47.89–70.27) | 12.49 (9.31–38.39) | 19.80 (19.42–20.40) | 159.03 (157.27–162.13) |
| modal_data320k-b8-t1024-e512h8-40k-lr6e-4-wu256k_steps40000_seed42 | 1024 | 48.41 (47.94–55.44) | 13.72 (12.20–26.00) | 19.63 (19.05–20.32) | 158.03 (153.83–161.02) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42 | 1024 | 50.13 (49.83–52.49) | 13.42 (10.65–44.99) | 19.08 (18.76–19.74) | 133.90 (82.68–135.53) |
| modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k_steps80000_seed42 | 1024 | 48.34 (47.97–50.51) | 12.05 (11.36–40.62) | 19.82 (19.46–20.32) | 156.41 (153.46–160.77) |
| modal_data640k-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42 | 1024 | 93.71 (91.48–96.61) | 17.85 (17.15–24.32) | 10.75 (10.72–11.00) | 170.37 (140.16–171.16) |
| modal_data640k-e768h12l8-160k-lr3e-4-rope-fused-cont30k-rlr5e-5_steps30000_seed42 | 1024 | 96.58 (92.70–101.03) | 17.77 (17.26–18.10) | 10.75 (10.71–10.81) | 172.53 (171.08–173.44) |
| modal_data640k-fw70edu30-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42 | 1024 | 96.12 (91.73–271.06) | 17.85 (17.20–18.27) | 10.74 (10.71–10.80) | 171.08 (136.55–173.69) |
