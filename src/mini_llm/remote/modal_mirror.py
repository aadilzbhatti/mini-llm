"""Mirror Modal runs into runs/ so the control web app tracks them like local runs.

    uv run --group modal mini-llm-modal-mirror --repo ~/dev/wiki-llm

The web app (mini_llm.server) knows nothing about Modal and doesn't need to:
it reads runs/<id>.status.json, runs/<id>.live.json and runs/<id>.log, which
the queue runner writes for local jobs. Every --interval seconds this reads
each run in the wiki-llm-runs volume (run.json, the trainer's heartbeat, and
train.log, all committed by modal_train.py every 30 s) and writes those same
three files for it. So a Modal run shows up in Live while it trains and in
History afterwards, with its log, metrics and full_val curve.

When a run finishes successfully and was a real experiment (it had
--val-tokens), it is also imported (mini_llm.import_run): checkpoint, plot
and sample report into checkpoints/ and plots/, a row into baselines.md. That
is what makes the page's Plot and Samples buttons work for it.

Status mapping: run.json's returncode -> completed / failed. No returncode yet
-> running, unless the heartbeat is more than --stale-after seconds old, in
which case the container is gone (timeout, preemption, a disabled workspace)
and the run shows as interrupted.

Live control (pause / stop / LR) is off under DDP, so Modal runs are
read-only here; the status carries "remote" so the page hides those buttons.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from mini_llm.import_run import default_stem, import_run
from mini_llm.remote import costs

RUNS_VOLUME = "wiki-llm-runs"
FINAL = {"completed", "failed"}


class VolumeLike(Protocol):
    """The two modal.Volume calls this needs, so tests can pass a fake."""

    def listdir(self, path: str, recursive: bool = False) -> list[Any]: ...
    def read_file(self, path: str) -> Any: ...


def _iso_z(ts: str | None) -> str | None:
    """'2026-09-26T20:15:56.97+00:00' -> '2026-09-26T20:15:56Z', the runner's format."""
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(ts: str | None) -> float | None:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None


def _read(vol: VolumeLike, path: str) -> bytes:
    return b"".join(vol.read_file(path))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_runner(repo: Path):
    """runner/run_queue.py, for summarize(): Modal runs get their metrics
    pulled from the log exactly the way local runs do."""
    spec = importlib.util.spec_from_file_location("_run_queue", repo / "runner" / "run_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mirror_run(
    vol: VolumeLike, run_id: str, repo: Path, runner: Any, stale_after: float = 900, now: float | None = None
) -> dict | None:
    """Bring one run's status/live/log files up to date. Returns the status written."""
    now = time.time() if now is None else now
    runs = repo / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    status_path = runs / f"{run_id}.status.json"
    try:
        prev = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        prev = {}
    if prev.get("status") in FINAL and prev.get("remote", {}).get("synced"):
        return prev  # finished and fully mirrored (and imported, if it was going to be)

    files = {e.path.removeprefix(f"{run_id}/"): e for e in vol.listdir(run_id, recursive=True)}
    if "run.json" not in files:
        return None  # container still starting
    record = json.loads(_read(vol, f"{run_id}/run.json"))

    live: dict = {}
    live_rel = f"runs/{run_id}.live.json"
    if live_rel in files:
        raw = _read(vol, f"{run_id}/{live_rel}")
        _write_atomic(runs / f"{run_id}.live.json", raw)
        live = json.loads(raw)

    log_path = runs / f"{run_id}.log"
    log_sig = None
    if "train.log" in files:
        entry = files["train.log"]
        log_sig = [entry.size, entry.mtime]
        if log_sig != prev.get("remote", {}).get("log_sig") or not log_path.exists():
            _write_atomic(log_path, _read(vol, f"{run_id}/train.log"))

    returncode = record.get("returncode")
    error = None
    if returncode is not None:
        state = "completed" if returncode == 0 else "failed"
        if returncode != 0:
            error = f"torchrun exited with {returncode}"
    else:
        beat = _epoch(live.get("updated")) or _epoch(record.get("started_at")) or now
        state = "running" if now - beat < stale_after else "interrupted"
        if state == "interrupted":
            error = f"no heartbeat from Modal for {int((now - beat) // 60)} min (container gone?)"

    config = record.get("config") or {}
    args = dict(config.get("args", {k: v for k, v in config.items() if k != "name"}))
    stem = default_stem(record)
    # Where the import puts the sample report, so the page's Samples button finds it.
    args["save-name"] = f"{stem}.pt"
    started = _iso_z(record.get("started_at"))
    argv = record.get("resolved_argv", [])
    # Only a saved run has a checkpoint to import (LR proxies run without --save).
    importable = returncode == 0 and "--val-tokens" in argv and "--save" in argv

    status = {
        "run_id": run_id,
        "kind": "train",
        "name": config.get("name") or run_id,
        "status": state,
        "args": args,
        "cmd": record.get("command"),
        "log": str(log_path.relative_to(repo)),
        "started": started,
        "finished": _iso_z(record.get("finished_at")),
        "duration_sec": (
            record.get("duration_sec")
            if returncode is not None
            else round(now - (_epoch(record.get("started_at")) or now), 1)
        ),
        "returncode": returncode,
        "metrics": runner.summarize(log_path) if log_path.exists() else {},
        "error": error,
        "job_file": None,  # not from queue/; stops the runner matching it to a queued job
        "remote": {
            "provider": "modal",
            "gpus": record.get("gpus"),
            "git_sha": record.get("git_sha"),
            "app_id": record.get("modal_app_id") or prev.get("remote", {}).get("app_id"),
            "log_sig": log_sig,
            "imported": prev.get("remote", {}).get("imported", False),
        },
    }
    if prev.get("remote", {}).get("cost"):
        status["remote"]["cost"] = prev["remote"]["cost"]  # written by costs.update_costs

    if importable and not status["remote"]["imported"]:
        local = runs / run_id  # same layout scripts/fetch_modal_run.sh produces
        for rel, entry in files.items():
            if entry.type == 1 and (rel == "run.json" or rel.startswith(("checkpoints/", "plots/"))):
                dest = local / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(dest, _read(vol, f"{run_id}/{rel}"))
        import_run(local, repo)
        status["remote"]["imported"] = True
    status["remote"]["synced"] = state in FINAL and (status["remote"]["imported"] or not importable)

    _write_atomic(status_path, json.dumps(status, indent=2).encode())
    return status


def mirror_all(vol: VolumeLike, repo: Path, runner: Any, stale_after: float = 900) -> list[dict]:
    out = []
    for entry in vol.listdir("/"):
        if entry.type != 2:  # directories only: one per run
            continue
        try:
            status = mirror_run(vol, entry.path, repo, runner, stale_after)
        except (
            Exception,
            SystemExit,
        ) as exc:  # noqa: BLE001 - one bad run must not stop the others (import_run exits on errors)
            print(f"[mirror] {entry.path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        if status:
            out.append(status)
    return out


def refresh_costs(repo: Path, interval: float, last_cost: float, last_rates: float) -> tuple[float, float]:
    """Billing for runs whose cost can still change every `interval` s; list prices daily.
    A billing failure is logged and retried next interval: it never stops mirroring."""
    now = time.time()
    if now - last_rates >= 24 * 3600:
        try:
            costs.write_rates(repo, costs.fetch_rates())
            last_rates = now
        except Exception as exc:  # noqa: BLE001
            print(f"[cost] rates failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
    if now - last_cost >= interval:
        last_cost = now
        try:
            costs.update_costs(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[cost] billing report failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
    return last_cost, last_rates


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Mirror Modal runs into runs/ for the control web app.")
    p.add_argument("--repo", type=Path, default=Path("."), help="Checkout whose runs/ the web app reads.")
    p.add_argument("--interval", type=float, default=30.0, help="Seconds between polls.")
    p.add_argument(
        "--stale-after",
        type=float,
        default=900.0,
        help="Seconds without a heartbeat before a run shows as interrupted.",
    )
    p.add_argument("--once", action="store_true", help="Sync once and exit.")
    p.add_argument("--no-auto-eval", action="store_true", help="Don't evaluate finished runs (see mini_llm.auto_eval).")
    p.add_argument(
        "--cost-interval",
        type=float,
        default=300.0,
        help="Seconds between billing checks for running and recently finished runs (mini_llm.remote.costs).",
    )
    args = p.parse_args(argv)

    import modal

    repo = args.repo.expanduser().resolve()
    vol = modal.Volume.from_name(RUNS_VOLUME)
    runner = load_runner(repo)
    seen: dict[str, str] = {}
    # Evaluates finished runs (Modal and local) on this machine's GPU, one at a time.
    evaluator = None
    if not (args.once or args.no_auto_eval):
        from mini_llm.auto_eval import AutoEvaluator

        evaluator = AutoEvaluator(repo)
        print(f"[auto-eval] on: evaluating runs that complete after {evaluator.since}", flush=True)
    failures = 0
    last_cost = last_rates = 0.0
    while True:
        try:
            for st in mirror_all(vol, repo, runner, args.stale_after):
                if seen.get(st["run_id"]) != st["status"]:
                    print(f"[mirror] {st['run_id']}: {st['status']}", flush=True)
                    seen[st["run_id"]] = st["status"]
            failures = 0
            last_cost, last_rates = refresh_costs(repo, args.cost_interval, last_cost, last_rates)
        except Exception as exc:  # noqa: BLE001 - e.g. Modal unreachable: retry, don't exit
            # Exiting would also kill an eval the auto-eval worker has running, and launchd's
            # restart re-queues it. Local runs don't need Modal, so keep scanning them below.
            if args.once:
                raise
            failures += 1
            print(f"[mirror] poll failed ({type(exc).__name__}: {exc}); retry #{failures}", file=sys.stderr, flush=True)
        if evaluator is not None:
            for run_id in evaluator.scan():
                print(f"[auto-eval] queued {run_id}", flush=True)
        if args.once:
            return
        time.sleep(min(args.interval * 2 ** min(failures, 4), 600))


if __name__ == "__main__":
    main()
