# data640k

Built 2026-10-02. Train is new; **val is data20k's, byte for byte**, the same
measuring stick as data20k, data40k, data80k, data160k and data320k.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 632,350,143 | 640,000 | `6cdba9bb31652e5e70cfcc141f2a2060fd39297e1547c68f8ace8583116fa12d` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

How it was made:

```bash
uv run mini-llm-prepare-data --num-examples 640000 --val-examples 1 --seed 0 --out-dir <tmp>
# keep <tmp>/train.pt; discard its 1-doc val
cp -p data/data20k/val.pt data/data640k/val.pt
```

This is the first dataset that needed the streaming `prepare_dataset` (rows
tokenized during the scan into int32 chunks). The old version held every token
as a Python int, ~20+ GB at this size. Output is unchanged: a 2,000-doc run of
the new code is an exact prefix of data320k/train.pt, and train.pt here was
built with the same scan, split and tokenization.

Verified:
- data320k/train.pt is an exact prefix of data640k/train.pt (one seed-0 scan
  order: 20k ⊂ 40k ⊂ 80k ⊂ 160k ⊂ 320k ⊂ 640k).
- val.pt is identical to data20k/val.pt.
- 0 of the 937 val docs appear in train.pt
  (`python -m mini_llm.token_overlap data/data640k/val.pt --exclude data/data640k/train.pt`).

train.pt is 5.06 GB on disk (int64). Training memory-maps it by default, so it
needn't fit in RAM; Modal runs read it fully (`--tokens-in-ram`). Upload with
`uv run --group modal modal volume put wiki-llm-data data/data640k /data640k`.

At the 327.68M-token budget (B8 × T1024 × 40K) this is ~0.52 passes, against
~1.04 over data320k at the same budget.
