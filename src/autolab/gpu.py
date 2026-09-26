"""Yield the GPU to the owner's queue runner.

The owner's runner trains on this same Mac. Any `<owner_runs_dir>/*.live.json`
heartbeat updated within `stale_after_s` and not marked `"finished": true` means
the GPU is busy. Reads only; never writes to the owner's checkout.
"""

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


def active_owner_runs(owner_runs_dir: Path, stale_after_s: float = 120, now: float | None = None) -> list[dict]:
    """Heartbeats that look like a live training run."""
    now = time.time() if now is None else now
    active = []
    for path in sorted(Path(owner_runs_dir).glob("*.live.json")):
        try:
            live = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # mid-write or unreadable: the next poll sees it
        if live.get("finished") is True:
            continue
        try:
            updated = datetime.fromisoformat(str(live["updated"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = now - updated.timestamp()
        if age < stale_after_s:
            active.append({"file": path.name, "run_id": live.get("run_id"), "step": live.get("step"),
                           "total_steps": live.get("total_steps"), "age_s": round(age, 1)})
    return active


def wait_for_gpu(
    owner_runs_dir: Path,
    stale_after_s: float = 120,
    poll_s: float = 60,
    idle_checks_required: int = 2,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    max_wait_s: float | None = None,
) -> float:
    """Block until the owner's runner has been idle for `idle_checks_required` polls.

    Returns seconds spent waiting. Raises TimeoutError past `max_wait_s`.
    """
    waited = 0.0
    idle_streak = 0
    while True:
        active = active_owner_runs(owner_runs_dir, stale_after_s)
        if active:
            idle_streak = 0
            desc = ", ".join(f"{a['run_id']} step {a['step']}/{a['total_steps']}" for a in active)
            log(f"GPU busy (owner run: {desc}); waiting {poll_s:.0f}s")
        else:
            idle_streak += 1
            if idle_streak >= idle_checks_required:
                return waited
        if max_wait_s is not None and waited >= max_wait_s:
            raise TimeoutError(f"GPU still busy after {waited:.0f}s")
        sleep(poll_s)
        waited += poll_s
