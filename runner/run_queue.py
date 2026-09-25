#!/usr/bin/env python3
"""Watch queue/ for job files and run them, one at a time.

A job is a small JSON file dropped into queue/. This watcher validates it
against an allowlist of mini-llm-train flags, runs the training command,
streams output to runs/<job>.log, and writes runs/<job>.status.json with
the parsed final metrics.

Deliberately paranoid about input: a job file can only ever turn into a
`uv run mini-llm-train` invocation with numeric flags and repo-relative
paths. No shell, no arbitrary commands, no writes outside the repo. The
worst a malformed or hostile job file can do is train a silly model.

Usage (normally via launchd, see README.md):
    python3 runner/run_queue.py [--repo PATH] [--once] [--poll SECONDS]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- what a job is allowed to ask for ---------------------------------------
#
# name -> (kind, minimum, maximum). Anything not listed here is rejected.
# The bounds aren't security boundaries (the allowlist is); they're there to
# catch typos before you spend an hour training emb=99999.

INT_FLAGS: dict[str, tuple[int, int]] = {
    "block-size": (1, 2048),
    "n-embd": (1, 2048),
    "n-head": (1, 64),
    "n-layer": (1, 64),
    "batch-size": (1, 512),
    "steps": (1, 1_000_000),
    "warmup-steps": (0, 100_000),
    "seed": (0, 2**31 - 1),
    "log-interval": (1, 100_000),
    "eval-interval": (1, 1_000_000),
    "eval-batches": (1, 10_000),
    "eval-seed": (0, 2**31 - 1),
    "full-eval-interval": (0, 1_000_000),
    "sample-tokens": (0, 10_000),
}

FLOAT_FLAGS: dict[str, tuple[float, float]] = {
    "dropout": (0.0, 1.0),
    "lr": (1e-8, 10.0),
    "min-lr": (1e-9, 10.0),
    "restart-lr": (1e-9, 10.0),
    "weight-decay": (0.0, 10.0),
}

# Paths, constrained to relative locations inside the repo.
PATH_FLAGS = {"tokens", "val-tokens", "text", "resume"}

# Filenames only (no directory part) -- train.py puts these in plots/ and
# checkpoints/ itself. plot-suffix is a fragment appended to the generated
# plot name, so it gets the same character rules.
NAME_FLAGS = {"plot-name", "save-name", "plot-suffix"}

BOOL_FLAGS = {"fixed-batch", "plot-loss", "save", "sample-report"}

DEFAULT_ARGS: dict[str, object] = {
    "tokens": "data/train.pt",
    "val-tokens": "data/val.pt",
    "plot-loss": True,
}

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class JobError(ValueError):
    """A job file that we refuse to run, with a reason worth reporting."""


def _check_relative_path(value: object, repo: Path, flag: str) -> str:
    if not isinstance(value, str) or not value:
        raise JobError(f"--{flag} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise JobError(f"--{flag} must be a relative path inside the repo, got {value!r}")
    resolved = (repo / path).resolve()
    if not str(resolved).startswith(str(repo.resolve())):
        raise JobError(f"--{flag} escapes the repo: {value!r}")
    if not resolved.exists():
        raise JobError(f"--{flag} points at a file that does not exist: {value}")
    return value


def build_command(args: dict, repo: Path, uv: str) -> list[str]:
    """Turn a validated job dict into an argv list. Raises JobError."""
    merged: dict[str, object] = dict(DEFAULT_ARGS)
    merged.update(args)

    cmd = [uv, "run", "--project", str(repo), "mini-llm-train"]

    for flag, value in sorted(merged.items()):
        if flag in INT_FLAGS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise JobError(f"--{flag} must be an integer, got {value!r}")
            low, high = INT_FLAGS[flag]
            if not low <= value <= high:
                raise JobError(f"--{flag}={value} outside allowed range [{low}, {high}]")
            cmd += [f"--{flag}", str(value)]

        elif flag in FLOAT_FLAGS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JobError(f"--{flag} must be a number, got {value!r}")
            low, high = FLOAT_FLAGS[flag]
            if not low <= float(value) <= high:
                raise JobError(f"--{flag}={value} outside allowed range [{low}, {high}]")
            cmd += [f"--{flag}", repr(float(value))]

        elif flag in PATH_FLAGS:
            cmd += [f"--{flag}", _check_relative_path(value, repo, flag)]

        elif flag in NAME_FLAGS:
            if not isinstance(value, str) or not SAFE_NAME.match(value):
                raise JobError(
                    f"--{flag} must be plain [A-Za-z0-9._-] text, got {value!r}"
                )
            cmd += [f"--{flag}", value]

        elif flag in BOOL_FLAGS:
            if not isinstance(value, bool):
                raise JobError(f"--{flag} must be true or false, got {value!r}")
            if value:
                cmd.append(f"--{flag}")

        else:
            raise JobError(f"unknown option {flag!r} (not in the allowlist)")

    return cmd


def parse_job(path: Path, repo: Path, uv: str) -> tuple[str, list[str], dict]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise JobError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise JobError("job must be a JSON object")

    name = raw.get("name") or path.stem
    if not isinstance(name, str) or not JOB_ID.match(name):
        raise JobError(f"invalid job name {name!r} (letters, digits, . _ - only)")

    args = raw.get("args", {k: v for k, v in raw.items() if k != "name"})
    if not isinstance(args, dict):
        raise JobError("'args' must be a JSON object")

    return name, build_command(args, repo, uv), args


# --- metrics --------------------------------------------------------------

METRIC_PATTERNS = {
    "eval_train_loss": re.compile(r"eval_train_loss ([0-9.]+)"),
    "eval_val_loss": re.compile(r"eval_val_loss ([0-9.]+)"),
    "full_val_loss": re.compile(r"full_val_loss ([0-9.]+)"),
}
PLOT_PATTERN = re.compile(r"Saved loss plot to (\S+)")
PARAMS_PATTERN = re.compile(r"^Model: ([0-9,]+) parameters", re.M)


def summarize(log_path: Path) -> dict:
    """Pull the numbers worth reporting out of a finished run's log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}

    out: dict[str, object] = {}
    for key, pattern in METRIC_PATTERNS.items():
        found = pattern.findall(text)
        if found:
            out[key] = float(found[-1])

    full_vals = re.findall(r"^step +(\d+) \| full_val_loss ([0-9.]+)", text, re.M)
    if full_vals:
        out["full_val_curve"] = [[int(s), float(v)] for s, v in full_vals]

    plot = PLOT_PATTERN.findall(text)
    if plot:
        out["plot"] = plot[-1]

    params = PARAMS_PATTERN.search(text)
    if params:
        out["params"] = int(params.group(1).replace(",", ""))

    steps = re.findall(r"^step +(\d+) \| loss", text, re.M)
    if steps:
        out["last_step"] = int(steps[-1])

    return out


# --- the loop -------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def settled(path: Path, wait: float = 1.0) -> bool:
    """True if the file has stopped changing -- avoids reading a partial write."""
    try:
        first = path.stat().st_size
        time.sleep(wait)
        return path.stat().st_size == first
    except OSError:
        return False


def run_job(job_path: Path, repo: Path, uv: str) -> None:
    queue = job_path.parent
    runs = repo / "runs"
    runs.mkdir(exist_ok=True)
    (queue / "done").mkdir(exist_ok=True)
    (queue / "failed").mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        name, cmd, args = parse_job(job_path, repo, uv)
    except JobError as exc:
        log(f"REJECTED {job_path.name}: {exc}")
        dest = queue / "failed" / f"{stamp}-{job_path.name}"
        shutil.move(str(job_path), dest)
        (runs / f"{stamp}-{job_path.stem}.status.json").write_text(
            json.dumps({"status": "rejected", "error": str(exc), "at": now()}, indent=2)
        )
        return

    run_id = f"{stamp}-{name}"
    log_path = runs / f"{run_id}.log"
    status_path = runs / f"{run_id}.status.json"

    status = {
        "run_id": run_id,
        "name": name,
        "status": "running",
        "args": args,
        "cmd": cmd,
        "log": str(log_path.relative_to(repo)),
        "started": now(),
    }
    status_path.write_text(json.dumps(status, indent=2))
    log(f"RUNNING {run_id}: {' '.join(cmd)}")

    env = dict(os.environ, PYTHONUNBUFFERED="1")  # so the log streams live
    started = time.time()
    try:
        with log_path.open("w") as sink:
            proc = subprocess.run(
                cmd, cwd=repo, stdout=sink, stderr=subprocess.STDOUT, env=env, check=False
            )
        returncode = proc.returncode
    except Exception as exc:  # noqa: BLE001 - want the message in the status file
        returncode = -1
        with log_path.open("a") as sink:
            sink.write(f"\nrunner error: {exc}\n")

    status.update(
        {
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "finished": now(),
            "duration_sec": round(time.time() - started, 1),
            "metrics": summarize(log_path),
        }
    )
    status_path.write_text(json.dumps(status, indent=2))

    with (runs / "index.jsonl").open("a") as index:
        index.write(json.dumps({k: status[k] for k in
                                ("run_id", "status", "started", "finished", "args", "metrics")
                                if k in status}) + "\n")

    shutil.move(str(job_path), str(queue / "done" / f"{stamp}-{job_path.name}"))
    log(f"{status['status'].upper()} {run_id} in {status['duration_sec']}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--poll", type=float, default=5.0, help="Seconds between queue checks.")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    parser.add_argument("--uv", default=shutil.which("uv") or os.path.expanduser("~/.local/bin/uv"))
    opts = parser.parse_args()

    repo = Path(opts.repo).resolve()
    queue = repo / "queue"
    queue.mkdir(exist_ok=True)

    if not Path(opts.uv).exists():
        log(f"FATAL: uv not found at {opts.uv} (pass --uv /path/to/uv)")
        return 1

    log(f"watching {queue} (repo={repo}, uv={opts.uv})")

    while True:
        jobs = sorted(p for p in queue.glob("*.json") if p.is_file())
        for job in jobs:
            if settled(job):
                run_job(job, repo, opts.uv)
        if opts.once:
            return 0
        time.sleep(opts.poll)


if __name__ == "__main__":
    sys.exit(main())
