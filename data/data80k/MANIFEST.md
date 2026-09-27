# data80k

Copied 2026-09-27 from the autolab clone
(`~/dev/wiki-llm-autolab/autolab/data/datasets/data80k/train.pt`, built there
2026-09-27 18:48 UTC as `prepare-data --num-examples 80000 --val-examples 0
--seed 0`). **Val is data20k's, byte for byte**, the same measuring stick as
data20k and data40k.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 80,491,699 | 80,000 | `ee52172304ff5700c192c05017183f24c2aac4f7b1aa72559223b2aaf61b82d9` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

Verified here:
- train.pt's sha256 matches autolab's dataset.json.
- data20k/train.pt and data40k/train.pt are exact prefixes of data80k/train.pt
  (one seed-0 scan order: 20k ⊂ 40k ⊂ 80k).
- val.pt is identical to data20k/val.pt.
- 0 of the 937 val docs appear in train.pt
  (`python -m mini_llm.token_overlap data/data80k/val.pt --exclude data/data80k/train.pt`).

At the 122.88M-token budget (B64 × 15K) this is ~1.53 passes.
