"""One-variable sweeps: one base training job, one flag varied over a few values.

    POST /api/sweeps   {"name": "e768l8-lr", "target": "modal", "gpus": "H100",
                        "vary": {"lr": [3e-4, 4e-4, 5e-4]}, "stop_after": 10000,
                        "args": {...the full job, e.g. --steps 160000...}}

The control server (mini_llm.server) expands it with expand(), validates every
child with the runner's rules before launching any, launches them (Modal: one
after another from a single process, so they never race to upload data; local:
into queue/, which runs one at a time), and records the sweep in
runs/sweeps/<id>.json. summarize() turns that record plus the children's
status files into what the page shows: full_val at the steps every child has
reached, the leader, and each child's gap to it.

Deliberately one flag per sweep (the project changes one variable at a time),
and stop_after so proxies share the real schedule: several LRs over the first
N steps of the same long cosine, not each on its own short one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

MAX_VALUES = 8
SWEEP_KEYS = ("vary", "stop_after")


class SweepError(ValueError):
    pass


@dataclass
class Sweep:
    name: str
    flag: str
    values: list[float | int]
    stop_after: int | None
    children: list[tuple[float | int, dict]] = field(default_factory=list)  # (value, child job body)


def value_tag(v: float | int) -> str:
    """3e-4 -> "3e-4", 0.0005 -> "5e-4", 2.5e-4 -> "2.5e-4", 0.1 -> "0.1", 256000 -> "256000":
    short and safe in a run name."""
    if isinstance(v, int):
        return str(v)
    mantissa, exp = f"{v:e}".split("e")
    e = int(exp)
    return f"{float(mantissa):g}e{e}" if e < -2 else f"{v:g}"


def expand(body: object, int_flags: dict, float_flags: dict) -> Sweep:
    """Validate the sweep's own fields and build one child job body per value.
    The children still go through the runner's validation (the server does that)."""
    if not isinstance(body, dict):
        raise SweepError("sweep must be a JSON object")
    if body.get("kind", "train") != "train":
        raise SweepError("only training jobs can be swept")
    name = body.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", name):
        raise SweepError("sweep needs a name (letters, digits, . _ - only)")
    vary = body.get("vary")
    if not isinstance(vary, dict) or len(vary) != 1:
        raise SweepError('vary must name exactly one flag, e.g. {"lr": [3e-4, 4e-4, 5e-4]}')
    [(flag, values)] = vary.items()
    if flag not in int_flags and flag not in float_flags:
        raise SweepError(f"can't sweep {flag!r}: only numeric training flags")
    if flag in ("steps", "stop-after"):
        raise SweepError(f"sweep {flag!r} by launching separate runs; a sweep compares runs over the same steps")
    if not isinstance(values, list) or not 2 <= len(values) <= MAX_VALUES:
        raise SweepError(f"vary needs 2-{MAX_VALUES} values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise SweepError("sweep values must be numbers")
    if flag in int_flags and any(float(v) != int(v) for v in values):
        raise SweepError(f"{flag} takes whole numbers")
    values = [int(v) for v in values] if flag in int_flags else [float(v) for v in values]
    if len({value_tag(v) for v in values}) != len(values):
        raise SweepError("sweep values must be distinct")
    args = body.get("args")
    if not isinstance(args, dict):
        raise SweepError("args must be the base job's flags")
    stop_after = body.get("stop_after")
    if stop_after is not None:
        if isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 1:
            raise SweepError("stop_after must be a positive whole number of steps")
        if stop_after > int(args.get("steps", 0) or 0):
            raise SweepError("stop_after must be within --steps (it stops early on the full schedule)")
    base = {k: v for k, v in body.items() if k not in SWEEP_KEYS and k not in ("name", "args")}
    sweep = Sweep(name=name, flag=flag, values=values, stop_after=stop_after)
    for v in values:
        child_name = f"{name}-{flag}{value_tag(v)}"
        child_args = {**args, flag: v}
        if stop_after is not None:
            child_args["stop-after"] = stop_after
        child_args["plot-suffix"] = f"{child_name}-{base.get('target', 'local')}"  # one plot per child
        sweep.children.append((v, {**base, "name": child_name, "kind": "train", "args": child_args}))
    return sweep


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def summarize(record: dict, statuses: dict[str, dict | None], costs: dict[str, dict | None]) -> dict:
    """The page's view of a sweep.

    record: runs/sweeps/<id>.json. statuses / costs: each child's status (with "live" if running) and
    cost block, keyed by the child's key in record["children"] (its run id, or its queue file locally).
    """
    children = []
    for c in record["children"]:
        st = statuses.get(c["key"]) or {}
        live = st.get("live") or {}
        curve = [(int(s), float(v)) for s, v in (st.get("metrics") or {}).get("full_val_curve") or []]
        cost = costs.get(c["key"]) or {}
        children.append(
            {
                "value": c["value"],
                "tag": value_tag(c["value"]),
                "name": c["name"],
                "run_id": st.get("run_id") or c.get("run_id"),
                "status": st.get("status") or "queued",
                "step": live.get("step") or (st.get("metrics") or {}).get("last_step"),
                "total_steps": live.get("total_steps") or record.get("stop_after"),
                "eval_val_loss": live.get("eval_val_loss") or (st.get("metrics") or {}).get("eval_val_loss"),
                "full_val_curve": curve,
                "diverged": any(not _finite(v) for _, v in curve),
                "cost_usd": cost.get("usd"),
                "cost_source": cost.get("source"),
            }
        )
    # Steps where every child has a full_val (step 0 says nothing about the variable, so it's left out).
    common = None
    for ch in children:
        steps = {s for s, _ in ch["full_val_curve"] if s > 0}
        common = steps if common is None else common & steps
    rows = []
    for step in sorted(common or ()):
        losses = {value_tag(ch["value"]): dict(ch["full_val_curve"])[step] for ch in children}
        finite = {k: v for k, v in losses.items() if _finite(v)}
        leader = min(finite, key=finite.get) if finite else None
        rows.append(
            {
                "step": step,
                "full_val": losses,
                "leader": leader,
                "gap": {k: (v - finite[leader]) if leader and _finite(v) else None for k, v in losses.items()},
            }
        )
    spent = [ch["cost_usd"] for ch in children if ch["cost_usd"] is not None]
    return {
        **{k: record[k] for k in ("id", "name", "flag", "values", "stop_after", "target", "created")},
        "children": children,
        "table": rows,
        "leader": rows[-1]["leader"] if rows else None,
        "cost_usd": round(sum(spent), 2) if spent else None,
        "done": all(ch["status"] in ("completed", "failed", "interrupted") for ch in children),
    }
