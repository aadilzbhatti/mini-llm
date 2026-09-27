# data40k

Built 2026-09-27. Train is new; **val is data20k's, byte for byte**, so runs on
data20k and data40k are scored on the identical measuring stick.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 40,504,110 | 40,000 | `199330d5d7b57b9a13b3083ebb6d97d1ff3621f25818f71ddbacf921ae65f1e2` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

How it was made:

```bash
uv run mini-llm-prepare-data --num-examples 40000 --val-examples 1 --seed 0 --out-dir <tmp>
# keep <tmp>/train.pt; discard its 1-doc val
cp -p data/data20k/val.pt data/data40k/val.pt
```

Defaults: HuggingFaceTB/smollm-corpus, fineweb-edu-dedup, split train,
val-pool-fraction 0.1, GPT-2 tokenizer.

Verified:
- data20k/train.pt is an exact prefix of data40k/train.pt (same seed-0 scan
  order; the first 20,000 train docs are identical).
- data40k/val.pt is identical to data20k/val.pt (same sha256).
- 0 of the 937 val docs appear in data40k/train.pt
  (`python -m mini_llm.token_overlap data/data40k/val.pt --exclude data/data40k/train.pt`).
