# data160k

Built 2026-09-28. Train is new; **val is data20k's, byte for byte**, the same
measuring stick as data20k, data40k and data80k.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 159,204,972 | 160,000 | `c6d7fe4a8f0cabd74b6a9b8f632c586fc0733986a238e003eff6a8c1a77e0a18` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

How it was made:

```bash
uv run mini-llm-prepare-data --num-examples 160000 --val-examples 1 --seed 0 --out-dir <tmp>
# keep <tmp>/train.pt; discard its 1-doc val
cp -p data/data20k/val.pt data/data160k/val.pt
```

Verified:
- data80k/train.pt is an exact prefix of data160k/train.pt (one seed-0 scan
  order: 20k ⊂ 40k ⊂ 80k ⊂ 160k).
- val.pt is identical to data20k/val.pt.
- 0 of the 937 val docs appear in train.pt
  (`python -m mini_llm.token_overlap data/data160k/val.pt --exclude data/data160k/train.pt`).

At the 122.88M-token budget (B64 × T128 × 15K) this is ~0.77 passes. Batches
are random crops drawn with replacement, so that still means only ~54% of
train tokens are ever seen, and ~30% of presentations are repeats.
