# Evals summary

One row per checkpoint. cb = context benefit in nats (± standard error), same windows for every model; cb@256 only for context ≥ 256. ret = forced-choice retrieval accuracy at that key-to-question distance (chance 10%), same trials for every model. Inference timings are from one device and noisy (±25%).

| model | ctx | val@128 | val@256 | val@ctx | cb@128 | cb@256 | ret@16 | ret@32 | ret@64 | ret@96 | ret@128 | ret@160 | ret@192 | ret@224 | ret@256 | ret@320 | ret@384 | ret@448 | ret@496 | train tok/s | train peak GB | prefill ms | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 4.2809 | 4.1774 | 4.1774 | 0.5291 ± 0.0083 | 0.4018 ± 0.0061 | 96% | 83% | 68% | 56% | 40% | 40% | 32% | 24% | 12% | 10% | 9% | 10% | 11% | 84,795 | 3.66 | 25.77 | 16.7 |
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 4.2601 | – | 4.2601 | 0.5074 ± 0.0081 | – | 96% | 79% | 39% | 22% | 8% | 12% | 10% | 7% | 11% | 8% | 8% | 9% | 10% | – | – | 10.93 | 77.4 |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 4.5967 | 4.4990 | 4.4990 | 0.4611 ± 0.0075 | 0.3602 ± 0.0057 | 87% | 77% | 60% | 48% | 27% | 31% | 24% | 21% | 12% | 10% | 9% | 12% | 10% | – | – | 17.82 | 30.2 |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 4.4679 | – | 4.4679 | 0.4619 ± 0.0075 | – | 87% | 77% | 53% | 42% | 7% | 12% | 7% | 8% | 9% | 10% | 10% | 8% | 11% | – | – | 20.07 | 42.0 |

## Paired against modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42

Differences on the identical windows / trials. cb: other − reference, ± paired standard error. ret: trials only this model got right / only the reference got right (a large imbalance is a real difference; roughly equal counts are noise).

| model | Δcb@128 | Δcb@256 | ret@16 | ret@32 | ret@64 | ret@96 | ret@128 | ret@160 | ret@192 | ret@224 | ret@256 | ret@320 | ret@384 | ret@448 | ret@496 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | +0.0217 ± 0.0031 | – | 11/12 | 49/33 | 128/12 | 145/7 | 130/3 | 122/7 | 93/5 | 78/10 | 25/21 | 20/13 | 20/13 | 21/17 | 22/17 |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | -0.0463 ± 0.0039 | – | 10/45 | 43/49 | 103/19 | 123/15 | 88/12 | 87/8 | 71/14 | 63/9 | 25/22 | 27/18 | 21/14 | 33/22 | 21/21 |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | -0.0455 ± 0.0033 | – | 7/42 | 35/43 | 86/30 | 101/20 | 18/22 | 20/20 | 14/24 | 20/16 | 17/24 | 19/14 | 31/21 | 19/23 | 27/23 |
