"""One process runs every lab: each turn gives each lab one cycle (a stage, a proposal or a wait), so GPU stages of
different labs never overlap and benchmarks stay clean. Heartbeat: autolab/state/labs/daemon.json.

    uv run labs daemon        # what com.aadil.autolab-labs runs
"""

import json
import os
import time
import traceback

from labs import core


def all_labs() -> list:
    from labs.decode.lab import LAB as decode
    from labs.infer.lab import LAB as infer

    return [infer, decode]


def turn(log=print) -> dict:
    out = {}
    for lab in all_labs():
        if not core.labs_cfg().get(lab.name, {}).get("enabled", True):
            out[lab.name] = {"waiting": "disabled in labs.toml"}
            continue
        try:
            out[lab.name] = lab.cycle(log=log)
        except Exception as exc:  # noqa: BLE001 - one lab's failure never stops the others
            out[lab.name] = {"error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-3000:]}
            log(f"{lab.name}: cycle failed: {type(exc).__name__}: {exc}")
    return out


def run(interval: float = 20.0) -> None:
    def log(msg: str) -> None:
        print(f"{core.iso()} {msg}", flush=True)

    log(f"labs daemon started (pid {os.getpid()})")
    while True:
        t0 = time.time()
        res = turn(log)
        core.LABS_STATE.mkdir(parents=True, exist_ok=True)
        (core.LABS_STATE / "daemon.json").write_text(json.dumps(
            {"pid": os.getpid(), "updated": core.iso(), "labs": res, "turn_s": round(time.time() - t0, 1)}, indent=2))
        busy = any(r.get("advanced") or r.get("proposed") or r.get("initialized") for r in res.values())
        time.sleep(1 if busy else interval)
