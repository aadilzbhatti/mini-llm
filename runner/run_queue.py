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

Two job kinds: "train" (the default) runs mini-llm-train, and
"prepare-data" runs mini-llm-prepare-data to build a new token set under
data/. Both go through the same allowlist treatment.

Each training process gets MINI_LLM_RUN_ID in its environment, so its
TensorBoard logs and live-control files (runs/<run_id>.commands.jsonl etc.,
see mini_llm.control) share the run id used here. If NTFY_TOPIC is set, job
start/finish/failure is pushed to https://ntfy.sh/<topic> (or NTFY_SERVER).

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
    "stop-after": (1, 1_000_000),
    "warmup-steps": (0, 100_000),
    "warmup-tokens": (0, 10_000_000_000),
    "seed": (0, 2**31 - 1),
    "log-interval": (1, 100_000),
    "eval-interval": (1, 1_000_000),
    "eval-batches": (1, 10_000),
    "eval-seed": (0, 2**31 - 1),
    "full-eval-interval": (0, 1_000_000),
    "sample-tokens": (0, 10_000),
    "sample-report-tokens": (1, 10_000),
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

BOOL_FLAGS = {"fixed-batch", "plot-loss", "save", "sample-report", "baseline", "fused-attention"}

DEFAULT_ARGS: dict[str, object] = {
    "tokens": "data/train.pt",
    "val-tokens": "data/val.pt",
    "plot-loss": True,
    "baseline": True,
}

# --- prepare-data jobs ------------------------------------------------------

PREP_INT_FLAGS: dict[str, tuple[int, int]] = {
    "num-examples": (1, 5_000_000),
    "val-examples": (1, 1_000_000),
    "seed": (0, 2**31 - 1),
}
PREP_FLOAT_FLAGS: dict[str, tuple[float, float]] = {
    "val-pool-fraction": (0.001, 0.9),
}
# HF ids and names: plain identifier characters (plus / for "org/name").
PREP_STR_FLAGS = {"dataset", "config", "split", "text-field", "tokenizer"}
HF_NAME = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")

JOB_KINDS = {"train", "prepare-data"}

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class JobError(ValueError):
    """A job file that we refuse to run, with a reason worth reporting."""


def _check_relative_path(value: object, repo: Path, flag: str, pending: frozenset[str] = frozenset()) -> str:
    if not isinstance(value, str) or not value:
        raise JobError(f"--{flag} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise JobError(f"--{flag} must be a relative path inside the repo, got {value!r}")
    resolved = (repo / path).resolve()
    if not str(resolved).startswith(str(repo.resolve())):
        raise JobError(f"--{flag} escapes the repo: {value!r}")
    if not resolved.exists() and value not in pending:
        raise JobError(f"--{flag} points at a file that does not exist: {value}")
    return value


def build_command(args: dict, repo: Path, uv: str, pending: frozenset[str] = frozenset()) -> list[str]:
    """Turn a validated job dict into an argv list. Raises JobError.

    `pending` holds repo-relative paths that don't exist yet but will by the
    time this job runs -- the checkpoint a queued or running job is going to
    save -- so a continuation can be queued behind the run it continues. The
    runner itself always validates with an empty set, so at run time the file
    really has to be there.
    """
    merged: dict[str, object] = dict(DEFAULT_ARGS)
    merged.update(args)

    # Cross-field rules train.py would otherwise only enforce after startup.
    if "restart-lr" in merged and "resume" not in merged:
        raise JobError("restart-lr only applies to a continuation; set resume too")
    if merged.get("tokens") and merged.get("tokens") == merged.get("val-tokens"):
        raise JobError("tokens and val-tokens are the same file: that trains on the validation set")
    if "warmup-steps" in merged and "warmup-tokens" in merged:
        raise JobError("set warmup-steps or warmup-tokens, not both (warmup-tokens is converted to steps)")
    n_embd, n_head = merged.get("n-embd", 128), merged.get("n-head", 4)
    if isinstance(n_embd, int) and isinstance(n_head, int) and n_head > 0 and n_embd % n_head:
        raise JobError(f"n-embd ({n_embd}) must be divisible by n-head ({n_head})")

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
            cmd += [f"--{flag}", _check_relative_path(value, repo, flag, pending)]

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


def build_prepare_command(args: dict, repo: Path, uv: str) -> list[str]:
    """argv for a prepare-data job. out-dir is required and must live under data/."""
    cmd = [uv, "run", "--project", str(repo), "mini-llm-prepare-data"]
    if "out-dir" not in args:
        raise JobError("prepare-data jobs need an out-dir, e.g. \"data/data20k\"")
    for flag, value in sorted(args.items()):
        if flag in PREP_INT_FLAGS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise JobError(f"--{flag} must be an integer, got {value!r}")
            low, high = PREP_INT_FLAGS[flag]
            if not low <= value <= high:
                raise JobError(f"--{flag}={value} outside allowed range [{low}, {high}]")
            cmd += [f"--{flag}", str(value)]
        elif flag in PREP_FLOAT_FLAGS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JobError(f"--{flag} must be a number, got {value!r}")
            low, high = PREP_FLOAT_FLAGS[flag]
            if not low <= float(value) <= high:
                raise JobError(f"--{flag}={value} outside allowed range [{low}, {high}]")
            cmd += [f"--{flag}", repr(float(value))]
        elif flag in PREP_STR_FLAGS:
            if not isinstance(value, str) or not HF_NAME.match(value) or ".." in value:
                raise JobError(f"--{flag} must be a plain dataset/tokenizer name, got {value!r}")
            cmd += [f"--{flag}", value]
        elif flag == "out-dir":
            if not isinstance(value, str) or not value:
                raise JobError("--out-dir must be a non-empty string")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data" \
                    or len(path.parts) < 2 or not all(SAFE_NAME.match(part) for part in path.parts):
                raise JobError(f"--out-dir must be a new folder under data/, got {value!r}")
            cmd += ["--out-dir", value]
        else:
            raise JobError(f"unknown option {flag!r} for a prepare-data job (not in the allowlist)")
    return cmd


def validate_job(raw: object, repo: Path, uv: str, default_name: str = "job",
                 pending: frozenset[str] = frozenset()) -> tuple[str, str, list[str], dict]:
    """Validate a job object -> (name, kind, argv, args). Raises JobError.

    Shared with the control API, which validates before it ever writes a job
    file, so a bad request is rejected at submit time rather than in queue/.
    """
    if not isinstance(raw, dict):
        raise JobError("job must be a JSON object")

    name = raw.get("name") or default_name
    if not isinstance(name, str) or not JOB_ID.match(name):
        raise JobError(f"invalid job name {name!r} (letters, digits, . _ - only)")

    kind = raw.get("kind", "train")
    if kind not in JOB_KINDS:
        raise JobError(f"unknown job kind {kind!r}; expected one of {sorted(JOB_KINDS)}")

    args = raw.get("args", {k: v for k, v in raw.items() if k not in ("name", "kind")})
    if not isinstance(args, dict):
        raise JobError("'args' must be a JSON object")

    if kind == "prepare-data":
        return name, kind, build_prepare_command(args, repo, uv), args
    return name, kind, build_command(args, repo, uv, pending), args


def parse_job(path: Path, repo: Path, uv: str) -> tuple[str, list[str], dict]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise JobError(f"not valid JSON: {exc}") from exc
    name, _kind, cmd, args = validate_job(raw, repo, uv, default_name=path.stem)
    return name, cmd, args


def job_kind(path: Path) -> str:
    try:
        raw = json.loads(path.read_text())
        return raw.get("kind", "train") if isinstance(raw, dict) else "train"
    except (OSError, json.JSONDecodeError):
        return "train"


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


# --- forecasting -----------------------------------------------------------
#
# Every job records what the closed-form models expected before it ran, and on
# completion its own error against that. Over time runs/ becomes a calibration
# record rather than just a log -- and a model that drifts is visible instead
# of quietly wrong. Loaded by path so this stays stdlib-only and hermetic:
# importing mini_llm as a package would pull in torch.

def _load_predict(repo: Path):
    import importlib.util
    path = repo / "src" / "mini_llm" / "predict.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_predict", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def forecast_for(repo: Path, args: dict):
    """Best-effort: a broken forecast must never stop a job from running."""
    mod = _load_predict(repo)
    if mod is None:
        return None
    try:
        runs = mod.load_runs(repo / "runs")
        if not runs:
            return None
        g = lambda k, d: args.get(k, d)
        cfg = dict(n_embd=int(g("n-embd", 128)), n_head=int(g("n-head", 4)),
                   n_layer=int(g("n-layer", 4)), block_size=int(g("block-size", 64)),
                   batch_size=int(g("batch-size", 4)), steps=int(args["steps"]),
                   lr=float(g("lr", 1e-3)), min_lr=float(g("min-lr", 2e-6)),
                   warmup_steps=(-(-int(args["warmup-tokens"]) // (int(g("batch-size", 4)) * int(g("block-size", 64))))
                                 if "warmup-tokens" in args else int(g("warmup-steps", 500))))
        return mod.forecast(cfg, runs)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def forecast_error(pred: dict, status: dict) -> dict:
    """Signed error of the forecast, once the truth is in."""
    if not pred or "error" in pred:
        return {}
    out = {}
    met = status.get("metrics") or {}
    actual_sec = status.get("duration_sec")
    if pred.get("time_sec") and actual_sec:
        out["time_pct"] = round(100 * (pred["time_sec"] - actual_sec) / actual_sec, 1)
        out["time_actual_sec"] = actual_sec
    if pred.get("loss") is not None and met.get("full_val_loss") is not None:
        out["loss_nats"] = round(pred["loss"] - met["full_val_loss"], 4)
        out["loss_actual"] = met["full_val_loss"]
    if pred.get("params_est") and met.get("params"):
        out["params_pct"] = round(100 * (pred["params_est"] - met["params"]) / met["params"], 2)
    return out


# --- notifications ---------------------------------------------------------

def notify(title: str, message: str, tags: str = "", priority: str = "default") -> None:
    """Best-effort push via ntfy (https://ntfy.sh). No-op unless NTFY_TOPIC is set."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    import urllib.request

    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(
        f"{server}/{topic}",
        data=message.encode(),
        headers={"Title": title, "Tags": tags, "Priority": priority},
        method="POST",
    )
    click = os.environ.get("NTFY_CLICK_URL")
    if click:
        req.add_header("Click", click)
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:  # noqa: BLE001 - never let a push failure touch a job
        log(f"  ntfy push failed: {exc}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def mark_interrupted(repo: Path) -> None:
    """At startup, any status still 'running' belongs to a process that's gone
    (the Mac slept, rebooted, or the watcher was restarted). Say so, instead of
    leaving it looking live forever."""
    for status_path in (repo / "runs").glob("*.status.json"):
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "running":
            if status.get("remote"):
                continue  # mirrored from Modal (mini_llm.remote.modal_mirror), not ours to judge
            pid = status.get("runner_pid")
            if pid and pid != os.getpid() and _alive(pid):
                continue  # another watcher (e.g. a manual --once) owns it
            status["status"] = "interrupted"
            status["finished"] = now()
            status_path.write_text(json.dumps(status, indent=2))
            log(f"marked {status_path.name} interrupted (was running when the watcher started)")


def with_caffeinate(cmd: list[str]) -> list[str]:
    """On macOS, hold an idle-sleep assertion for the life of the job.

    Doesn't stop lid-close sleep (nothing short of clamshell mode does), but
    does stop the Mac idling to sleep mid-run while it's on power.
    """
    if sys.platform == "darwin" and Path("/usr/bin/caffeinate").exists():
        return ["/usr/bin/caffeinate", "-i", *cmd]
    return cmd


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
        notify(f"rejected: {job_path.name}", str(exc), tags="warning")
        dest = queue / "failed" / f"{stamp}-{job_path.name}"
        shutil.move(str(job_path), dest)
        (runs / f"{stamp}-{job_path.stem}.status.json").write_text(
            json.dumps({"status": "rejected", "error": str(exc), "at": now()}, indent=2)
        )
        return

    run_id = f"{stamp}-{name}"
    log_path = runs / f"{run_id}.log"
    status_path = runs / f"{run_id}.status.json"
    kind = job_kind(job_path)

    pred = forecast_for(repo, args) if kind == "train" else None
    status = {
        "run_id": run_id,
        "kind": kind,
        "forecast": pred,
        "name": name,
        "status": "running",
        "args": args,
        "cmd": cmd,
        "log": str(log_path.relative_to(repo)),
        "started": now(),
        "job_file": job_path.name,
        "runner_pid": os.getpid(),
    }
    status_path.write_text(json.dumps(status, indent=2))
    log(f"RUNNING {run_id}: {' '.join(cmd)}")
    if pred and "error" not in pred:
        loss_txt = f"{pred['loss']:.4f}" if pred.get("loss") is not None else "n/a (no family data)"
        log(f"  forecast: {pred['time_hours']:.1f}h, full_val {loss_txt}")

    notify(f"started: {name}", f"{kind} job {run_id}", tags="arrow_forward")

    # PYTHONUNBUFFERED so the log streams live; MINI_LLM_RUN_ID so the trainer's
    # TensorBoard dir and control files use this same id.
    env = dict(os.environ, PYTHONUNBUFFERED="1", MINI_LLM_RUN_ID=run_id)
    started = time.time()
    try:
        with log_path.open("w") as sink:
            proc = subprocess.run(
                with_caffeinate(cmd), cwd=repo, stdout=sink, stderr=subprocess.STDOUT, env=env,
                check=False,
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
    status["forecast_error"] = forecast_error(pred, status)
    status_path.write_text(json.dumps(status, indent=2))
    if status["forecast_error"]:
        log(f"  forecast error: {status['forecast_error']}")

    with (runs / "index.jsonl").open("a") as index:
        index.write(json.dumps({k: status[k] for k in
                                ("run_id", "kind", "status", "started", "finished", "args", "metrics",
                                 "forecast", "forecast_error")
                                if k in status}) + "\n")

    shutil.move(str(job_path), str(queue / "done" / f"{stamp}-{job_path.name}"))
    log(f"{status['status'].upper()} {run_id} in {status['duration_sec']}s")

    met = status["metrics"]
    hours = status["duration_sec"] / 3600
    if status["status"] == "completed":
        body = f"{hours:.1f}h"
        if met.get("full_val_loss") is not None:
            body += f", full_val_loss {met['full_val_loss']:.4f}"
        notify(f"done: {name}", body, tags="white_check_mark")
    else:
        notify(f"FAILED: {name}", f"exit {returncode} after {hours:.1f}h -- see runs/{run_id}.log",
               tags="x", priority="high")


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
    mark_interrupted(repo)

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
