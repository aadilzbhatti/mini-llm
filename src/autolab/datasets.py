"""build_dataset: grow the training set with the owner's `mini-llm-prepare-data` (HANDOFF decision 3).

    uv run autolab data build --num-examples 80000          # -> autolab/data/datasets/data80k/
    uv run autolab data slice data80k --docs 40000          # doc-prefix -> data40k (no download)
    uv run autolab data check --num-examples 200            # "bigger = superset" check

A build:
1. runs prepare_dataset into a scratch dir with --val-examples 0 and deletes any val.pt it
   writes (the frozen val is never rebuilt);
2. rejects the new train file if any document (sha256 of its token ids, mini_llm.token_overlap)
   is also in the frozen val;
3. records how many of the reference set's (data20k) documents it contains: prepare_dataset
   streams with a seeded shuffle and appends train rows in order, so a bigger build with the
   same seed should contain a smaller one as a prefix. That's checked, not assumed;
4. writes dataset.json (tokens, docs, sha256, bytes, CLI args) and enforces [datasets] max_total_gb.

Dataset builds need network (Hugging Face streaming). That's allowed: the no-network rule covers
training and gates only. Upload to Modal afterwards with `autolab modal upload-data`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch

from autolab.config import REPO_ROOT, load_config, sha256_file
from mini_llm.token_overlap import EOS_TOKEN_ID, doc_hash, split_documents


def data_cfg() -> dict:
    return tomllib.loads((REPO_ROOT / "autolab" / "config.toml").read_text())["datasets"]


def doc_hashes(path: Path) -> list[str]:
    return [doc_hash(d) for d in split_documents(torch.load(path))]


def dataset_id(num_docs: int) -> str:
    return f"data{num_docs // 1000}k" if num_docs % 1000 == 0 else f"data{num_docs}"


def _dir_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9 if path.exists() else 0.0


def describe(train: Path, args: dict, reference: Path | None, log=print) -> dict:
    """Validate a train file against the frozen val and the reference set; return its record."""
    cfg = load_config()
    hashes = doc_hashes(train)
    val = set(doc_hashes(cfg.frozen_val))
    overlap = sum(h in val for h in hashes)
    rec = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": args,
           "tokens": int(torch.load(train).numel()), "docs": len(hashes), "sha256": sha256_file(train),
           "bytes": train.stat().st_size, "overlap_with_frozen_val_docs": overlap}
    if reference is not None and reference.exists():
        ref = doc_hashes(reference)
        mine = set(hashes)
        rec["reference"] = {"id": reference.parent.name, "docs": len(ref),
                            "contained": sum(h in mine for h in ref),
                            "is_prefix": hashes[: len(ref)] == ref}
        log(f"contains {rec['reference']['contained']:,}/{len(ref):,} {reference.parent.name} docs "
            f"(prefix: {rec['reference']['is_prefix']})")
    return rec


def build(num_examples: int, seed: int | None = None, log=print) -> dict:
    cfg, dcfg = load_config(), data_cfg()
    seed = dcfg["seed"] if seed is None else seed
    ds_id = dataset_id(num_examples)
    dest = cfg.datasets_dir / ds_id
    if (dest / "train.pt").exists():
        raise FileExistsError(f"{ds_id} already exists")
    est_gb = num_examples * 8.3e-6  # ~1,030 tokens/doc x 8 bytes
    if _dir_gb(cfg.datasets_dir) + est_gb > dcfg["max_total_gb"]:
        raise RuntimeError(f"disk cap: datasets use {_dir_gb(cfg.datasets_dir):.2f} GB + ~{est_gb:.2f} GB "
                           f"> {dcfg['max_total_gb']} GB")
    scratch = cfg.datasets_dir / f".build-{ds_id}"
    shutil.rmtree(scratch, ignore_errors=True)
    args = {"num_examples": num_examples, "val_examples": 0, "seed": seed}
    cmd = [sys.executable, "-m", "mini_llm.prepare_dataset", "--num-examples", str(num_examples),
           "--val-examples", "0", "--seed", str(seed), "--out-dir", str(scratch)]
    log(f"building {ds_id}: {' '.join(cmd[2:])}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    (scratch / "val.pt").unlink(missing_ok=True)
    rec = {"id": ds_id, **describe(scratch / "train.pt", args, cfg.datasets_dir / dcfg["reference"] / "train.pt", log)}
    if rec["overlap_with_frozen_val_docs"]:
        shutil.rmtree(scratch)
        raise RuntimeError(f"{ds_id} rejected: {rec['overlap_with_frozen_val_docs']} docs overlap the frozen val")
    scratch.rename(dest)
    (dest / "dataset.json").write_text(json.dumps(rec, indent=2))
    log(f"{ds_id}: {rec['tokens']:,} tokens, {rec['docs']:,} docs, 0 overlap with frozen val")
    return rec


def slice_prefix(source_id: str, num_docs: int, log=print) -> dict:
    """The first `num_docs` documents of an existing set (what a smaller build with the same seed yields)."""
    cfg = load_config()
    src = cfg.datasets_dir / source_id / "train.pt"
    tokens = torch.load(src)
    cuts = (tokens == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(cuts) < num_docs:
        raise ValueError(f"{source_id} has only {len(cuts)} docs")
    ds_id = dataset_id(num_docs)
    dest = cfg.datasets_dir / ds_id
    if (dest / "train.pt").exists():
        raise FileExistsError(f"{ds_id} already exists")
    dest.mkdir(parents=True)
    torch.save(tokens[: int(cuts[num_docs - 1]) + 1].clone(), dest / "train.pt")
    args = {"prefix_of": source_id, "docs": num_docs}
    rec = {"id": ds_id, **describe(dest / "train.pt", args, cfg.datasets_dir / data_cfg()["reference"] / "train.pt", log)}
    if rec["overlap_with_frozen_val_docs"]:
        shutil.rmtree(dest)
        raise RuntimeError(f"{ds_id} overlaps the frozen val")
    (dest / "dataset.json").write_text(json.dumps(rec, indent=2))
    log(f"{ds_id}: {rec['tokens']:,} tokens, {rec['docs']:,} docs (prefix of {source_id})")
    return rec
