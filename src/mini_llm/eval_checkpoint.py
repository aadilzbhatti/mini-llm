"""Score saved checkpoints against arbitrary token files, after the fact.

Training only ever evaluates against the val file it was handed. When the
yardstick changes -- as it did when `data/data10k` turned out to be a
pre-hash positional split, 90% of whose val documents sit in
`data/data20k/train.pt` -- every past number needs re-measuring on the new
set before any of them can be compared to a new run.

Every checkpoint carries `full_val_history`, so the last logged value is
the run's own number on its own val file. This re-computes it and refuses
to report anything until that reproduces: if the same weights and the same
tokens don't return the same loss, the eval path here differs from the one
that produced the table, and every cross-eval in the output would be
quietly wrong. Pass `--reproduce` naming the original val file to arm that
check.

    python -m mini_llm.eval_checkpoint checkpoints/*.pt \
        --val-tokens data/val_clean.pt \
        --reproduce data/data10k/val.pt
"""

import argparse
from pathlib import Path

import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import load_tokens
from mini_llm.device import select_device
from mini_llm.train import evaluate_full

REPRODUCE_TOLERANCE = 1e-3


def load_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate saved checkpoints on one or more token files.")
    p.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files to score.")
    p.add_argument("--val-tokens", nargs="+", required=True, help="Token files to score against.")
    p.add_argument(
        "--reproduce",
        default=None,
        help="The val file each run originally used. Its recomputed loss is checked "
        "against the checkpoint's own last logged full_val; a mismatch is fatal.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Eval batch size. evaluate_full weights "
                   "by window count, so this changes speed, not the result.")
    p.add_argument("--device", default=None, help="Override device (default: MPS, then CUDA, then CPU).")
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else select_device()
    print(f"device: {device}\n")

    paths = list(dict.fromkeys(([args.reproduce] if args.reproduce else []) + args.val_tokens))
    tokens = {q: load_tokens(q) for q in paths}
    for q, t in tokens.items():
        print(f"{q}: {t.numel():,} tokens")
    print()

    rows: list[dict[str, object]] = []
    for ckpt_path in args.checkpoints:
        model, cfg, ckpt = load_model(ckpt_path, device)
        n_params = sum(t.numel() for t in dict.fromkeys(model.parameters()))
        name = Path(ckpt_path).stem
        logged = ckpt["full_val_history"][-1][1] if ckpt.get("full_val_history") else None

        losses: dict[str, float] = {}
        for q in paths:
            losses[q] = evaluate_full(model, tokens[q], args.batch_size, cfg.block_size, device)

        if args.reproduce is not None and logged is not None:
            delta = abs(losses[args.reproduce] - logged)
            mark = "ok" if delta < REPRODUCE_TOLERANCE else "MISMATCH"
            print(f"{name}: logged {logged:.4f} vs recomputed {losses[args.reproduce]:.4f}  [{mark}]")
            if delta >= REPRODUCE_TOLERANCE:
                raise SystemExit(
                    f"{name}: recomputed {args.reproduce} loss {losses[args.reproduce]:.4f} does not "
                    f"reproduce the logged {logged:.4f} (delta {delta:.4f}). The eval path here differs "
                    "from the one that produced the results table -- fix that before trusting any "
                    "number in this run."
                )
        else:
            print(f"{name}: no reproduction check ({'no --reproduce' if args.reproduce is None else 'no logged full_val'})")

        rows.append({"run": name, "params": n_params, "block": cfg.block_size,
                     "step": ckpt.get("step"), **{q: losses[q] for q in paths}})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print()
    headers = ["run", "params", "block", "step"] + paths
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in sorted(rows, key=lambda r: r[args.val_tokens[0]]):
        cells = [str(r["run"]), f"{r['params']:,}", str(r["block"]), f"{r['step']:,}"]
        cells += [f"{r[q]:.4f}" for q in paths]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
