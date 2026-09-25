#!/usr/bin/env python3
"""One-off: backfill baselines.md from runs/*.log for jobs that finished
before train.py grew --baseline.

Parses each completed run's OWN printed Config: / Optimization: dicts
(not the job's args delta, which only records overrides and would silently
mis-attribute early runs to whatever train.py's defaults happen to be
*today* if a default has since changed) plus its Saved to / Saved loss
plot to lines, and upserts one row per run_id via the same
update_baselines() a live --baseline run uses.

Usage: uv run python runner/backfill_baselines.py
"""

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from mini_llm.baselines import update_baselines

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"

CONFIG_RE = re.compile(r"^Config: (\{.*\})$", re.M)
OPTIM_RE = re.compile(r"^Optimization: (\{.*\})$", re.M)
PARAMS_RE = re.compile(r"^Model: ([0-9,]+) parameters", re.M)
SAVED_CKPT_RE = re.compile(r"^Saved to (\S+)$", re.M)
SAVED_PLOT_RE = re.compile(r"^Saved loss plot to (\S+)$", re.M)
EVAL_TRAIN_RE = re.compile(r"eval_train_loss ([0-9.]+)")
EVAL_VAL_RE = re.compile(r"eval_val_loss ([0-9.]+)")
FULL_VAL_RE = re.compile(r"full_val_loss ([0-9.]+)")


def _last_float(pattern: re.Pattern, text: str) -> float | None:
    found = pattern.findall(text)
    return float(found[-1]) if found else None


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def backfill_one(status_path: Path) -> str | None:
    status = json.loads(status_path.read_text())
    if status.get("status") != "completed":
        return f"skip {status_path.name}: status={status.get('status')!r}"

    log_path = REPO / status["log"]
    text = log_path.read_text(errors="replace")

    config_match = CONFIG_RE.search(text)
    optim_match = OPTIM_RE.search(text)
    if not config_match or not optim_match:
        return f"skip {status_path.name}: no Config:/Optimization: line in log"
    cfg = ast.literal_eval(config_match.group(1))
    optim = ast.literal_eval(optim_match.group(1))

    params_match = PARAMS_RE.search(text)
    params = int(params_match.group(1).replace(",", "")) if params_match else None

    ckpt_matches = SAVED_CKPT_RE.findall(text)
    plot_matches = SAVED_PLOT_RE.findall(text)

    args = status.get("args", {})
    run_name = args.get("save-name") or args.get("plot-name") or status.get("name") or status["run_id"]
    # save-name is a bare filename (train.py writes it under checkpoints/ as-is),
    # so a job that passed e.g. "foo.pt" leaves a redundant extension on the
    # display name -- the checkpoint column already has the real path.
    if run_name.endswith(".pt"):
        run_name = run_name[: -len(".pt")]

    row = {
        "run": run_name,
        "date": _fmt_date(status.get("finished") or status.get("started")),
        "steps": optim.get("total_steps"),
        "params": params,
        "n_layer": cfg.get("n_layer"),
        "n_embd": cfg.get("n_embd"),
        "n_head": cfg.get("n_head"),
        "block_size": cfg.get("block_size"),
        "batch_size": optim.get("batch_size"),
        "lr": optim.get("lr"),
        "min_lr": optim.get("min_lr"),
        "warmup_steps": optim.get("warmup_steps"),
        "seed": optim.get("seed"),
        "eval_train_loss": _last_float(EVAL_TRAIN_RE, text),
        "eval_val_loss": _last_float(EVAL_VAL_RE, text),
        "full_val_loss": _last_float(FULL_VAL_RE, text),
        "checkpoint": ckpt_matches[-1] if ckpt_matches else None,
        "plot": plot_matches[-1] if plot_matches else None,
    }
    update_baselines(row, md_path=REPO / "baselines.md")
    return f"OK   {status_path.name} -> run={run_name!r}"


def main() -> None:
    for status_path in sorted(RUNS_DIR.glob("*.status.json")):
        print(backfill_one(status_path))


if __name__ == "__main__":
    main()
