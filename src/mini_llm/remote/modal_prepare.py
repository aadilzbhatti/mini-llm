"""Build a mixed token dataset on Modal, straight into the wiki-llm-data volume.

    uv run --group modal modal run --detach src/mini_llm/remote/modal_prepare.py \\
        --name data640k-fw70edu30 \\
        --mix HuggingFaceFW/fineweb:sample-10BT=0.7,HuggingFaceTB/smollm-corpus:fineweb-edu-dedup=0.3 \\
        --num-tokens 632350143 --val-num-tokens 918728

Why not on the Mac: streaming general FineWeb reads whole Parquet row groups at a time, and the
build's memory spikes past 6 GB within a minute, which macOS kills on the 16 GB Mac mini (twice, at
~100K documents). A CPU container with room to spare costs cents and sits next to the Hub.

Writes /data/<name>/ on the volume, in the layout every training run expects:
  train.pt     the mixture (prepare_dataset.prepare_mix)
  val.pt       the FIXED yardstick: data20k's val, byte for byte, as every dataset since data20k
  val_mix.pt   the mixture's own val (same token proportions), for in-distribution loss
  mix.json     per-source documents, tokens and shares, plus the overlap check below
and refuses to finish if either val file shares a document with train.pt.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

from mini_llm.remote.modal_train import DATA_MOUNT, data_volume, image

app = modal.App("wiki-llm-prepare", image=image)
YARDSTICK = "data20k/val.pt"


@app.function(volumes={DATA_MOUNT: data_volume}, cpu=8, memory=32768, timeout=4 * 3600)
def build(name: str, mix: list[str], num_tokens: int, val_num_tokens: int, seed: int = 0) -> dict:
    from mini_llm.prepare_dataset import MixSource, prepare_mix
    from mini_llm.token_overlap import file_hashes

    out = Path(DATA_MOUNT) / name
    if out.exists() and (out / "train.pt").exists():
        raise FileExistsError(f"{out} already has a train.pt; pick a new --name or delete it first")
    work = Path("/tmp") / name  # build on local disk, then copy: the volume is network-backed
    _, _, stats = prepare_mix(
        [MixSource.parse(spec) for spec in mix],
        num_tokens=num_tokens,
        val_num_tokens=val_num_tokens,
        seed=seed,
        out_dir=work,
    )
    (work / "val.pt").rename(work / "val_mix.pt")
    shutil.copy(Path(DATA_MOUNT) / YARDSTICK, work / "val.pt")

    train = file_hashes(work / "train.pt")
    stats["overlap_with_train"] = {
        f: {"docs": len(h := file_hashes(work / f)), "in_train": len(h & train)} for f in ("val.pt", "val_mix.pt")
    }
    stats["yardstick"] = f"val.pt is {YARDSTICK} byte for byte"
    (work / "mix.json").write_text(json.dumps(stats, indent=2) + "\n")
    leaked = {f: o for f, o in stats["overlap_with_train"].items() if o["in_train"]}
    if leaked:
        raise RuntimeError(f"val documents found in train.pt, not saving: {leaked}")

    out.mkdir(parents=True, exist_ok=True)
    for f in ("val.pt", "val_mix.pt", "mix.json", "train.pt"):
        shutil.copy(work / f, out / f)
    data_volume.commit()
    return stats


@app.local_entrypoint()
def main(name: str, mix: str, num_tokens: int, val_num_tokens: int, seed: int = 0):
    """--mix takes the sources comma-separated: A=0.7,B=0.3 (modal run passes one value per flag)."""
    stats = build.remote(name, mix.split(","), num_tokens, val_num_tokens, seed)
    print(json.dumps(stats, indent=2))
