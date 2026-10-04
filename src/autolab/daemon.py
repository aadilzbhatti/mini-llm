"""The always-on side of autolab: runs under launchd, independent of any chat session.

    uv run autolab daemon            # what com.aadil.autolab-daemon runs

Every `interval` seconds:
  1. collect finished Modal trials into autolab/runs/<run_id>/ (report, diagnosis, log);
  2. copy live progress (heartbeat + log tail) of running trials from the runs volume;
  3. advance every unfinished program through the evaluation cascade (autolab.evaluate),
     judge finished data checks, and run the controller (autolab.controller: proposals within
     the daily budget, pauses, the data policy, the notebook, accepted-program commits);
  4. write a heartbeat to autolab/state/daemon.json, which the dashboard shows.

While held (`autolab pause` / the dashboard's Pause button) a cycle does only 1-2 and the heartbeat.
A failed cycle is logged and retried next cycle. It never takes the daemon down.
The M5 controller loop will run from here too, so the whole system keeps going
with nobody at the keyboard.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from autolab.config import REPO_ROOT

STATE = REPO_ROOT / "autolab" / "state"
_last_report = 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_heartbeat(beat: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / ".daemon.json.tmp"
    tmp.write_text(json.dumps(beat, indent=2))
    tmp.replace(STATE / "daemon.json")


def cycle(log=print) -> dict:
    from autolab import modal_backend as mb

    from autolab.activity import set_activity

    t0 = time.time()
    set_activity("collect", "collecting finished Modal runs and live progress")
    finished = mb.collect(log=log)
    live = mb.fetch_live()
    from autolab.controller import load_control

    if (held := load_control().get("hold")):  # owner's pause: collect only; nothing advances, submits or proposes
        pending = sum(c["state"] == "pending" for c in mb.load_calls().values())
        set_activity("idle", f"paused by {held.get('by', 'owner')} since {held.get('at', '?')} "
                             f"({pending} run{'s' if pending != 1 else ''} still training on Modal)")
        return {"finished_this_cycle": finished, "live_files": live, "programs_advanced": [], "held": held,
                "pending": pending, "cycle_s": round(time.time() - t0, 2)}
    from autolab.evaluate import advance_everything

    # evaluation cascade + data-check verdicts, in every session (an old one may still have work)
    advanced = advance_everything(log=log)
    from autolab.controller import step

    control = step(log=log)  # propose within budget, data policy, notebook, accepted commits
    global _last_report
    if time.time() - _last_report > 3600:  # the M6 session report, hourly
        from autolab.session_report import build

        build()
        _last_report = time.time()
    calls = mb.load_calls()
    pending = sum(c["state"] == "pending" for c in calls.values())
    set_activity("idle", f"waiting for the next cycle ({pending} run{'s' if pending != 1 else ''} training on Modal)")
    return {
        "finished_this_cycle": finished,
        "live_files": live,
        "programs_advanced": advanced,
        "controller": control,
        "pending": sum(c["state"] == "pending" for c in calls.values()),
        "cycle_s": round(time.time() - t0, 2),
    }


def run(interval: float = 60.0, once: bool = False) -> None:
    started = now_iso()
    last_ok, last_error, cycles = None, None, 0

    def log(msg: str) -> None:
        print(f"{now_iso()} {msg}", flush=True)

    log(f"autolab daemon started (pid {os.getpid()}, every {interval:.0f}s)")
    while True:
        cycles += 1
        result: dict = {}
        # heartbeat at the start too: a cycle with Claude calls + CPU checks can take minutes
        write_heartbeat({"pid": os.getpid(), "started": started, "updated": now_iso(), "interval_s": interval,
                         "cycles": cycles, "last_ok": last_ok, "last_error": last_error, "in_cycle": True})
        try:
            result = cycle(log)
            last_ok = now_iso()
        except Exception as exc:  # noqa: BLE001 - keep running; surface it on the dashboard
            last_error = {"at": now_iso(), "error": f"{type(exc).__name__}: {exc}",
                          "trace": traceback.format_exc()[-4000:]}
            log(f"cycle failed: {last_error['error']}")
        write_heartbeat({"pid": os.getpid(), "started": started, "updated": now_iso(), "interval_s": interval,
                         "cycles": cycles, "last_ok": last_ok, "last_error": last_error, **result})
        if once:
            return
        time.sleep(interval)
