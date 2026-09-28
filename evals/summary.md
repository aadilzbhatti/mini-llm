# Evals summary

Quality, context use, long-range retrieval and cost, one row per checkpoint. Retrieval is forced-choice accuracy (chance 10%) at each key-to-question distance.

| model | ctx | params | val@128 | val@ctx | ctx benefit | ret@16 | ret@32 | ret@64 | ret@96 | ret@128 | ret@160 | ret@192 | ret@224 | train tok/s | train peak GB | prefill ms | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_data160k-b32-t256-40k-lr1.2e-3-wu256k_steps40000_seed42 | 256 | 16.1M | 4.2809 | 4.1774 | 0.515 | 93% | 88% | 73% | 55% | 35% | 53% | 25% | 30% | 84,795 | 3.66 | 14.86 | 43.0 |
| modal_data160k-bs64-15k-lr1.2e-3-wu256k-v4_steps40000_seed42 | 128 | 16.1M | 4.2601 | 4.2601 | 0.496 | 95% | 82% | 47% | 30% | 12% | 13% | 7% | 15% | – | – | 11.05 | 59.6 |
| modal_data80k-b32-t256-15k-lr1.2e-3-wu256k_steps15000_seed42 | 256 | 16.1M | 4.5967 | 4.4990 | 0.441 | 87% | 80% | 62% | 55% | 22% | 40% | 25% | 27% | – | – | 26.02 | 30.6 |
| modal_data80k-bs64-15k-lr1.2e-3-wu256k_steps15000_seed42 | 128 | 16.1M | 4.4679 | 4.4679 | 0.449 | 90% | 82% | 60% | 45% | 13% | 12% | 3% | 18% | – | – | 18.38 | 38.7 |
