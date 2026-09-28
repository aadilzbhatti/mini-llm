# data20k

Provenance reconstructed 2026-09-27 (it was not recorded when built).

| file | tokens | docs | sha256 |
|---|---|---|---|
| train.pt | 20,543,855 | 20,000 | `5d4ba5216aec847e49a3a835d08035ae6914881943af8289d7c66f7bc1c26f26` |
| val.pt | 918,728 | 937 | `28b1041acaa527ac6d7e0a464b07115e766c5bf76252ecbc56a8519de02569ee` |

- **train**: `mini-llm-prepare-data --num-examples 20000 --seed 0` (defaults
  otherwise). Confirmed: a seed-0 rebuild reproduces its first document, and
  it is an exact prefix of data40k/train.pt.
- **val**: made with the content-hash splitter, then cleaned with
  `mini_llm.token_overlap ... --exclude data/data10k/train.pt data/data10k/val.pt`
  (data10k used an older positional split and ~90% of its val docs are in
  data20k's train). 63 of 1,000 docs removed, leaving 937. That cleaning step
  means this val set can't be regenerated from prepare-data flags alone:
  copy the file instead (as data40k does).
- 0 val docs appear in train.
