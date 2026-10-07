"""Bring a fetched remote (Modal) run into this repo the way a local run lands.

    uv run mini-llm-import-run runs/<run_id>                 # into this repo
    uv run mini-llm-import-run runs/<run_id> --repo ~/dev/wiki-llm

A local `mini-llm-train --save --plot-loss --sample-report --baseline` leaves
a checkpoint + sample report in checkpoints/, a plot in plots/ and a row in
baselines.md. A remote run leaves the same files, but inside its own run
directory (see mini_llm.remote.modal_train). This copies them into place
and upserts the baselines row with train.py's exact columns, so remote and
local runs sit in one table, ranked on the same full_val_loss.

The row's hyperparams come from re-parsing the run's own argv with train.py's
parser (so defaults resolve exactly as they did in the run) and its losses
from the checkpoint's histories. Remote-only details (run id, GPUs, git sha)
go into baselines.json only; the markdown table keeps its columns.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from mini_llm.baselines import update_baselines
from mini_llm.config import ModelConfig, build_model
from mini_llm.train import build_parser


def final_checkpoint(run_dir: Path) -> Path:
    """The end-of-training checkpoint (mid-run ones are named *.step<N>.pt)."""
    ckpts = [p for p in (run_dir / "checkpoints").glob("*.pt") if ".step" not in p.name]
    if len(ckpts) != 1:
        raise SystemExit(f"expected exactly one final checkpoint in {run_dir}/checkpoints, found {ckpts}")
    return ckpts[0]


def default_stem(record: dict) -> str:
    """Checkpoint/baselines name for an imported run: modal_<config name>_steps<N>_seed<N>.

    The step count is part of the name because config names aren't unique
    across budgets (the same "bs64-lr1.2e-3" config ran at 1,250 and 10,000
    steps), and a shared name would make the second import overwrite the first.
    """
    args = build_parser().parse_args(record["resolved_argv"])
    name = (record.get("config") or {}).get("name") or record["run_id"]
    return f"modal_{name}_steps{args.steps}_seed{args.seed}"


def import_run(run_dir: Path, repo: Path, name: str | None = None) -> dict[str, object]:
    record = json.loads((run_dir / "run.json").read_text())
    if record.get("returncode") != 0:
        raise SystemExit(f"{run_dir.name} did not finish (returncode={record.get('returncode')}); nothing to import")

    args = build_parser().parse_args(record["resolved_argv"])
    src_ckpt = final_checkpoint(run_dir)
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    # parameters() dedups the tied embedding/lm_head weight; the state_dict would count it twice.
    n_params = sum(p.numel() for p in build_model(cfg).parameters())

    stem = name or default_stem(record)
    ckpt_dest = repo / "checkpoints" / f"{stem}.pt"
    ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, ckpt_dest)
    report = src_ckpt.with_suffix(".md")
    if report.exists():
        shutil.copy2(report, ckpt_dest.with_suffix(".md"))

    plot_dest = None
    plots = sorted((run_dir / "plots").glob("*.png"))
    if plots:
        plot_dest = repo / "plots" / plots[-1].name
        plot_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plots[-1], plot_dest)

    def last(key: str) -> float | None:
        history = ckpt.get(key) or []
        return history[-1][1] if history else None

    row: dict[str, object] = {
        "run": ckpt_dest.name,
        "date": record["finished_at"][:16].replace("T", " ") + " UTC",
        "steps": ckpt["step"],
        "params": n_params,
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "block_size": cfg.block_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "eval_train_loss": last("train_history"),
        "eval_val_loss": last("val_history"),
        "full_val_loss": last("full_val_history"),
        "checkpoint": str(ckpt_dest.relative_to(repo)),
        "plot": str(plot_dest.relative_to(repo)) if plot_dest else None,
        # JSON-only: where and how it ran.
        "remote_run_id": record["run_id"],
        "gpus": record.get("gpus"),
        "world_size": record.get("nproc"),
        "git_sha": record.get("git_sha"),
        "duration_sec": record.get("duration_sec"),
        # Training systems metrics recorded by the run itself (mini_llm.systems).
        **{k: (ckpt.get("systems") or {}).get(k) for k in ("train_tokens_per_sec", "peak_mem_gb", "wall_sec")},
    }
    update_baselines(row, repo / "baselines.md")
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Import a fetched remote run into checkpoints/, plots/ and baselines.md.")
    p.add_argument("run_dirs", nargs="+", type=Path, help="Fetched run directories, e.g. runs/<run_id>.")
    p.add_argument("--repo", type=Path, default=Path("."), help="Repo to import into (default: current directory).")
    p.add_argument("--name", default=None, help="Checkpoint stem / baselines run name (single run only).")
    args = p.parse_args(argv)
    if args.name and len(args.run_dirs) > 1:
        raise SystemExit("--name only makes sense for a single run")
    for run_dir in args.run_dirs:
        row = import_run(run_dir, args.repo.expanduser().resolve(), args.name)
        print(f"imported {run_dir.name} -> {row['checkpoint']} (full_val_loss {row['full_val_loss']})")


if __name__ == "__main__":
    main()
