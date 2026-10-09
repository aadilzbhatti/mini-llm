# data640k-fw70edu30

Built 2026-10-09 on Modal (`src/mini_llm/remote/modal_prepare.py`, commit after 8171e40). A
breadth experiment: the same token budget as data640k, but 70% general FineWeb and 30% FineWeb-Edu
**by tokens**, instead of 100% FineWeb-Edu. **val.pt is data20k's, byte for byte**, the same
measuring stick as every dataset since data20k; val_mix.pt is this mixture's own val.

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 633,453,440 | 830,000 | `06888cafcf8bba64afd1208b30a6ec281e76f99da3bc8e96900b4f6337692fe1` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |
| val_mix.pt | 919,373 | 1,220 | `6f6fa8c6c0d508a3b034e4c77e29170713f3d64285d74264e69d667a46c2762b` |

| source | train tokens | share | train docs | val_mix tokens | val_mix docs |
|---|---|---|---|---|---|
| HuggingFaceFW/fineweb : sample-10BT | 443,066,574 | 69.94% | 639,000 | 643,688 | 943 |
| HuggingFaceTB/smollm-corpus : fineweb-edu-dedup | 190,386,866 | 30.06% | 191,000 | 275,685 | 277 |

How it was made:

```bash
uv run --group modal modal run --detach src/mini_llm/remote/modal_prepare.py \
    --name data640k-fw70edu30 \
    --mix HuggingFaceFW/fineweb:sample-10BT=0.7,HuggingFaceTB/smollm-corpus:fineweb-edu-dedup=0.3 \
    --num-tokens 632350143 --val-num-tokens 918728
```

- Token quotas: 632,350,143 (data640k's train size) split 70/30; each source overshoots by under one
  1,000-doc tokenized batch (+0.17% in total). General FineWeb documents average ~690 tokens, Edu ~990,
  so document shares (77/23) differ from token shares.
- Same seed (0), shuffle buffer and content-hash train/val rule as every other dataset, so the Edu part
  is the first 191,000 train documents of the seed-0 Edu scan: a prefix of data640k/train.pt.
- Sources interleave in ~1,000-document runs, so any prefix of train.pt holds the mixture.
- Built on Modal because streaming general FineWeb spiked the Mac's build past 6 GB and macOS killed it.

Verified (in the build, which refuses to save otherwise; see mix.json):
- 0 of the 937 val.pt docs and 0 of the 1,220 val_mix.pt docs appear in train.pt (exact
  document hashes, mini_llm.token_overlap).
- val.pt == data/data20k/val.pt (cmp).

At B8 x T1024 x 160K steps (1.31B token presentations) this is ~2.07 passes, the same as data640k.
