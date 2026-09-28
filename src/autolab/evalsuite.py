"""Autolab's adapter over the owner's evals (`mini_llm.evals`, `mini_llm.systems`).

The owner builds the measurements in the main project; autolab doesn't duplicate them. This module runs
`mini_llm.evals.evaluate_checkpoint` on a trial's checkpoint, in the Modal container with the candidate's
model code, and maps the result onto autolab's four frontier dimensions. The only addition is a longer
distance sweep for the owner's `retrieval()` (its default stops at 224, but autolab's candidates may use
up to 1024 tokens of context), passed through that function's own `distances` parameter.

    python -m autolab.evalsuite <run_dir> --val <val.pt>     (PYTHONPATH = the program's src)

Writes <run_dir>/eval.json: the owner's full result under "owner_evals", plus autolab's summary:
  quality      full val at a fixed 128 window (comparable across contexts) and at the program's own window
  context      retrieval accuracy by distance -> long_range_score = mean over distances of accuracy
               above chance, normalized to [0, 1]; effective_context = largest distance with accuracy >= 0.5;
               context_benefit_nats (the owner's real-prefix vs other-document-prefix gap)
  inference    prefill ms (full window), decode tokens/sec (-> ms/token), memory, params
  training     the checkpoint's `systems` (train tokens/s with eval time excluded, peak memory, train time)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# 16..224 are the owner's DISTANCES; beyond them, up to autolab's longest allowed context (1024)
EXTENDED_DISTANCES = (16, 32, 64, 96, 128, 160, 192, 224, 320, 448, 640, 896)


def summarize(owner: dict) -> dict:
    cfg = owner["config"]
    T = cfg["block_size"]
    q = owner["quality"]
    ret = owner["retrieval"]
    chance = ret["chance"]
    accs = {int(d): v["accuracy"] for d, v in ret["by_distance"].items()}
    lr = sum(max(0.0, (a - chance) / (1 - chance)) for a in accs.values()) / max(1, len(accs))
    usable = [d for d, a in accs.items() if a >= 0.5]
    inf = owner["inference"]
    sysm = owner.get("training_systems") or {}
    params = owner["params"]
    non_emb = params - (cfg["vocab_size"] + T) * cfg["n_embd"]
    return {
        "context": T,
        "quality": {"full_val_at_128": q.get("full_val@128"), f"full_val_at_{T}": q.get(f"full_val@{T}"),
                    "short_context_loss": q.get("full_val@128")},
        "context_capability": {"accuracy_by_distance": {str(d): a for d, a in sorted(accs.items())},
                               "chance": chance, "long_range_score": lr,
                               "effective_context": max(usable) if usable else 0,
                               "context_benefit_nats": owner["context_benefit"]["benefit_nats"]},
        "inference": {"params": params, "non_embedding_params": non_emb,
                      "fwd_flops_per_token": 2 * non_emb + 2 * cfg["n_layer"] * T * cfg["n_embd"]
                      + 2 * cfg["n_embd"] * cfg["vocab_size"],
                      "prefill_ms": inf["prefill_ms_full_context"], "decode_tokens_per_s": inf["decode_tokens_per_sec"],
                      "decode_ms_per_token": 1000.0 / inf["decode_tokens_per_sec"] if inf["decode_tokens_per_sec"] else None,
                      "peak_inference_mem_bytes": (inf.get("memory_gb") or 0) * 2**30 or None},
        "training": {"train_tokens_per_sec": sysm.get("train_tokens_per_sec"), "train_sec": sysm.get("train_sec"),
                     "peak_train_mem_bytes": (sysm.get("peak_mem_gb") or 0) * 2**30 or None},
    }


def run(run_dir: Path, val_path: Path) -> dict:
    from mini_llm import evals  # the owner's implementation, from the program's code tree
    from mini_llm.data import get_tokenizer, load_tokens
    from mini_llm.device import select_device

    t0 = time.time()
    ckpt = run_dir / "checkpoints" / "model.pt"
    device = select_device()
    owner = evals.evaluate_checkpoint(ckpt, val_path, device)
    model, cfg, _ = evals.load_model(ckpt, device)
    longer = [d for d in EXTENDED_DISTANCES if d > max(evals.DISTANCES)]
    if longer:  # same function, more distances
        extra = evals.retrieval(model, get_tokenizer(), load_tokens(val_path), cfg.block_size, device, distances=longer)
        owner["retrieval"]["by_distance"].update(extra["by_distance"])
    out = {**summarize(owner), "owner_evals": owner, "eval_s": round(time.time() - t0, 1), "device": str(device)}
    (run_dir / "eval.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--val", type=Path, required=True)
    a = p.parse_args(argv)
    out = run(a.run_dir, a.val)
    print(json.dumps({"context": out["context"], "eval_s": out["eval_s"],
                      "long_range_score": out["context_capability"]["long_range_score"]}))


if __name__ == "__main__":
    main()
