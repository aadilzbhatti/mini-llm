"""What Modal runs cost: billed (from Modal) and estimated (from elapsed time).

Two halves, so the web app never has to talk to Modal:

  * the mirror (mini_llm.remote.modal_mirror) calls fetch_rates() about once a
    day and writes runs/modal-rates.json, and every few minutes calls
    fetch_billed() for runs whose cost is still changing and stores the result
    in runs/<id>.status.json under remote.cost;
  * the server calls run_cost() on each status it serves, which turns those
    numbers plus the clock into what the page shows.

Modal's billing report is per App and hourly, and it lags by minutes to an hour,
so "spent so far" is always an estimate: elapsed hours x an hourly rate. The rate
is the run's own billed rate once Modal has reported a full hour (it includes the
CPU and memory a GPU-only price misses, ~11% on 2xL4), else the list price of its
GPUs. The billed total is shown next to it as the authoritative, if late, number.
Runs launched before app ids were recorded (no remote.app_id) only get estimates.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RATES_FILE = "modal-rates.json"
# Modal list prices ($/GPU-hour, `modal billing rates`, 2026-10-07): used until the mirror has
# written runs/modal-rates.json, and for any GPU that file doesn't name.
FALLBACK_GPU_HOURLY = {
    "t4": 0.59,
    "l4": 0.80,
    "a10g": 1.10,
    "l40s": 1.95,
    "a100_40gb": 2.10,
    "a100_80gb": 2.50,
    "rtx6000": 3.03,
    "h100": 3.95,
    "h200": 4.54,
    "b200": 6.25,
    "b300": 7.10,
}
CPU_RANKS = 4.0  # modal_train runs `--gpus cpu` on 4 cores
# Modal's spellings -> the rate keys (`gpu_hour_cost_<key>`).
ALIASES = {"a10": "a10g", "a100": "a100_40gb", "a100_40g": "a100_40gb", "a100_80g": "a100_80gb", "h100!": "h100"}
SETTLE_SEC = 2 * 3600  # after this long past the finish, Modal's report has the whole run


def gpu_key(spec: str) -> str:
    key = spec.strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def list_hourly(gpus: str | None, rates: dict[str, float] | None = None) -> float | None:
    """List price per hour of a `--gpus` value: "L4:2" -> 1.60. None if unknown."""
    if not gpus:
        return None
    rates = rates or {}
    if gpus == "cpu":
        cpu = rates.get("cpu_hour_cost")
        return round(cpu * CPU_RANKS, 4) if cpu else None
    spec, _, count = gpus.partition(":")
    key = gpu_key(spec)
    price = rates.get(f"gpu_hour_cost_{key}", FALLBACK_GPU_HOURLY.get(key))
    return None if price is None else round(price * int(count or 1), 4)


def _epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def run_cost(status: dict, live: dict | None, rates: dict[str, float] | None, now: float) -> dict | None:
    """The cost block the page shows for one Modal run, or None for local runs.

    usd: spent so far (running) or in total (finished); always set when an hourly rate is known.
    source: "billed" when usd is Modal's settled figure, else "estimate".
    billed_usd / billed_updated: Modal's report as of the mirror's last check, if any.
    hourly_usd / rate_source: the rate behind the estimate ("billed" or "list").
    projected_usd: running runs only: usd + the trainer's ETA at that rate.
    """
    remote = status.get("remote") or {}
    if remote.get("provider") != "modal":
        return None
    billed = remote.get("cost") or {}
    listed = list_hourly(remote.get("gpus"), rates)
    hourly = billed.get("hourly_usd") or listed
    out: dict[str, Any] = {
        "hourly_usd": hourly,
        "rate_source": "billed" if billed.get("hourly_usd") else "list",
        "list_hourly_usd": listed,
        "billed_usd": billed.get("billed_usd"),
        "billed_updated": billed.get("updated"),
        "usd": None,
        "source": "estimate",
        "projected_usd": None,
    }
    if remote.get("phase") in ("launching", "launch-failed"):
        out["usd"] = 0.0 if remote.get("phase") == "launching" else None
        return out
    if status.get("status") != "running" and billed.get("final") and billed.get("billed_usd") is not None:
        out.update(usd=billed["billed_usd"], source="billed")
        return out
    if hourly is None:
        return out
    started = _epoch(status.get("started"))
    if status.get("status") == "running":
        elapsed = max(0.0, now - started) if started else float(status.get("duration_sec") or 0)
    elif status.get("status") == "interrupted":
        # The mirror keeps counting duration_sec after the container is gone; it
        # stopped costing at its last heartbeat.
        beat = _epoch((live or {}).get("updated"))
        if beat is None or started is None:
            return out
        elapsed = max(0.0, beat - started)
    else:
        elapsed = float(status.get("duration_sec") or 0)
    out["usd"] = round(elapsed / 3600 * hourly, 2)
    if status.get("status") == "running" and live and live.get("eta_sec") is not None and not live.get("paused"):
        out["projected_usd"] = round(out["usd"] + float(live["eta_sec"]) / 3600 * hourly, 2)
    return out


# --- the mirror's side: everything below talks to Modal ------------------------


def load_rates(runs_dir: Path) -> dict[str, float]:
    try:
        return json.loads((runs_dir / RATES_FILE).read_text()).get("rates", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def fetch_rates(workspace=None) -> dict[str, float]:
    """Modal's current $/hour rates for GPUs, CPU cores and memory."""
    if workspace is None:
        import modal

        workspace = modal.Workspace.from_context()
    rates = workspace.billing.rates()
    return {k: float(v) for k, v in rates.items() if k.startswith(("gpu_hour_cost_", "cpu_hour_cost", "memory"))}


def fetch_billed(apps: dict[str, float], workspace=None, now: float | None = None) -> dict[str, dict]:
    """Billed cost so far per Modal app id, from one hourly report.

    apps: app id -> the run's start (epoch seconds), which bounds the report.
    Returns app id -> {"billed_usd", "hourly_usd"}. hourly_usd is the mean of the app's
    full hours (every reported hour but its first and last, which are partial), or None
    before Modal has reported one. Apps with nothing billed yet are left out.
    """
    if not apps:
        return {}
    if workspace is None:
        import modal

        workspace = modal.Workspace.from_context()
    now_dt = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc)
    start = datetime.fromtimestamp(min(apps.values()), timezone.utc) - timedelta(hours=1)
    # The report drops a partial final interval, so end at the next hour to include this one.
    end = now_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    hours: dict[str, list[tuple[datetime, float]]] = {}
    for item in workspace.billing.report(start=start, end=end, resolution="h"):
        if item.object_id in apps:
            hours.setdefault(item.object_id, []).append((item.interval_start, float(item.cost)))
    out = {}
    for app_id, rows in hours.items():
        rows.sort()
        full = [cost for _, cost in rows[1:-1]]
        out[app_id] = {
            "billed_usd": round(sum(cost for _, cost in rows), 4),
            "hourly_usd": round(sum(full) / len(full), 4) if full else None,
        }
    return out


def update_costs(repo: Path, workspace=None, now: float | None = None) -> list[str]:
    """Refresh remote.cost in every Modal run's status whose bill can still change.

    That's runs that are running, or finished less than SETTLE_SEC ago or not yet
    marked final. Returns the run ids it updated.
    """
    now = time.time() if now is None else now
    runs_dir = repo / "runs"
    todo: dict[str, tuple[Path, dict]] = {}
    apps: dict[str, float] = {}
    for path in runs_dir.glob("*.status.json"):
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        remote = status.get("remote") or {}
        app_id = remote.get("app_id")
        if remote.get("provider") != "modal" or not app_id or (remote.get("cost") or {}).get("final"):
            continue
        started = _epoch(status.get("started"))
        if started is None:
            continue
        todo[app_id] = (path, status)
        apps[app_id] = started
    if not apps:
        return []
    billed = fetch_billed(apps, workspace, now)
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = []
    for app_id, (path, status) in todo.items():
        # Re-read: the mirror may have rewritten the status since the scan above.
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        finished = _epoch(status.get("finished"))
        settled = status.get("status") != "running" and (finished is None or now - finished > SETTLE_SEC)
        got = billed.get(app_id)
        if got is None and not settled:
            continue
        prev = status["remote"].get("cost") or {}
        status["remote"]["cost"] = {
            "billed_usd": (got or {}).get("billed_usd", prev.get("billed_usd")),
            "hourly_usd": (got or {}).get("hourly_usd") or prev.get("hourly_usd"),
            "updated": stamp,
            "final": settled,
        }
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(path)
        updated.append(status.get("run_id") or path.name)
    return updated


def write_rates(repo: Path, rates: dict[str, float], now: float | None = None) -> None:
    path = repo / "runs" / RATES_FILE
    stamp = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({"updated": stamp, "rates": rates}, indent=2))
    tmp.replace(path)
