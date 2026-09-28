"""The Pareto frontier over four dimensions (M8; owner: "optimize for all dimensions, not just loss").

Objectives, from a program's scores:
    quality          full_mean (lower)                      tie if within `quality_tol` (default 1 σ of seed noise)
    context          metrics.long_range_score (higher)      tie if within `rel_tol` (3%) or 0.01 absolute
    training cost    metrics.train_wall_s (lower)           full-budget training time on the session's GPU
    inference cost   metrics.decode_ms_per_token (lower)    batch-1 decode latency at the program's context

A program is on the frontier if no other program is at least as good on every objective and strictly
better on one (with the ties above, so noise can't put a program on or off the frontier). Only programs
that finished the full budget with eval-suite metrics take part. The lowest-loss program stays the
"quality champion" (the session incumbent) that the data and compute policies use.
"""

from __future__ import annotations

OBJECTIVES = (("quality", "full_mean", -1), ("context", "long_range_score", +1),
              ("train_cost", "train_wall_s", -1), ("inference_cost", "decode_ms_per_token", -1))


def vector(p) -> dict | None:
    m = (p.scores or {}).get("metrics") or {}
    fm = (p.scores or {}).get("full_mean")
    if fm is None or not all(k in m for k in ("long_range_score", "train_wall_s", "decode_ms_per_token")):
        return None
    return {"quality": fm, "context": m["long_range_score"], "train_cost": m["train_wall_s"],
            "inference_cost": m["decode_ms_per_token"]}


def _better(a: float, b: float, sign: int, tol: float) -> int:
    """+1 if a is better than b beyond tol, -1 if worse beyond tol, 0 if tied."""
    d = (a - b) * sign
    tol = tol * (1 + 1e-9) + 1e-12  # a difference of exactly `tol` is a tie (0.19 - 0.20 isn't exactly -0.01)
    return 1 if d > tol else (-1 if d < -tol else 0)


def compare(va: dict, vb: dict, quality_tol: float, rel_tol: float = 0.03) -> tuple[int, int]:
    """(# objectives where a is better, # where a is worse)."""
    better = worse = 0
    for name, _, sign in OBJECTIVES:
        if name == "quality":
            tol = quality_tol
        else:
            tol = max(rel_tol * max(abs(va[name]), abs(vb[name])), 0.01 if name == "context" else 0.0)
        c = _better(va[name], vb[name], sign, tol)
        better += c > 0
        worse += c < 0
    return better, worse


def frontier(programs, quality_tol: float, rel_tol: float = 0.03) -> list:
    """Programs not dominated by any other (ties don't dominate)."""
    cands = [(p, v) for p in programs if (v := vector(p)) is not None and p.status in (
        "accepted", "contender", "evaluated")]
    out = []
    for p, v in cands:
        dominated = False
        for q, w in cands:
            if q is p:
                continue
            b, wo = compare(w, v, quality_tol, rel_tol)
            if b > 0 and wo == 0:
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


def table(programs, quality_tol: float) -> list[dict]:
    """Rows for prompts and the dashboard: every program with metrics, frontier flag, best-on marks."""
    front = {p.id for p in frontier(programs, quality_tol)}
    rows = []
    for p in programs:
        v = vector(p)
        if v is None:
            continue
        m = p.scores["metrics"]
        rows.append({"id": p.id, "frontier": p.id in front, "full_mean": v["quality"], "n_seeds": p.scores.get("n_seeds"),
                     "long_range_score": v["context"], "effective_context": m.get("effective_context"),
                     "context": m.get("context"), "train_wall_s": v["train_cost"],
                     "train_tokens_per_sec": m.get("train_tokens_per_sec"), "decode_ms_per_token": v["inference_cost"],
                     "prefill_ms": m.get("prefill_ms"), "params": m.get("params"),
                     "peak_train_mem_bytes": m.get("peak_train_mem_bytes"),
                     "peak_inference_mem_bytes": m.get("peak_inference_mem_bytes"),
                     "short_context_loss": m.get("short_context_loss")})
    return sorted(rows, key=lambda r: (not r["frontier"], r["full_mean"]))
