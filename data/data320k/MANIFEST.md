# data320k

Built 2026-09-30. Train is new; **val is data20k's, byte for byte**, the same
measuring stick as data20k, data40k, data80k and data160k.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 316,231,039 | 320,000 | `513239cc0b5b862af4cb53b0157573f9e3e9a6d08dab7ddde62a95869ec51b64` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

How it was made:

```bash
uv run mini-llm-prepare-data --num-examples 320000 --val-examples 1 --seed 0 --out-dir <tmp>
# keep <tmp>/train.pt; discard its 1-doc val
cp -p data/data20k/val.pt data/data320k/val.pt
```

Verified:
- data160k/train.pt is an exact prefix of data320k/train.pt (one seed-0 scan
  order: 20k ⊂ 40k ⊂ 80k ⊂ 160k ⊂ 320k).
- val.pt is identical to data20k/val.pt.
- 0 of the 937 val docs appear in train.pt
  (`python -m mini_llm.token_overlap data/data320k/val.pt --exclude data/data320k/train.pt`).

At the 327.68M-token budget (B8 × T1024 × 40K) this is ~1.04 passes, against
~2.06 over data160k at the same budget.
