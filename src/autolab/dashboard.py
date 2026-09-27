"""Autolab dashboard: a read-only web view of trials, experiments, spend and decisions.

    uv run autolab dashboard --port 8766      # launchd: com.aadil.autolab-dashboard

Served on the tailnet at https://<mac>.ts.net/autolab via `tailscale serve
--set-path /autolab`, beside the owner's control page at /. The page uses only
relative URLs, so it works under that prefix and at the root alike.

Everything comes from files the daemon and backends write:
- autolab/state/modal_calls.json   every submitted trial (state, GPU, $)
- autolab/state/daemon.json        daemon heartbeat
- autolab/runs/<id>/               launch.json, report.json, diagnosis.json, train.log, live.json
- autolab/experiments/*.json       experiment batches (purpose, base config, jobs, analysis kind)
- autolab/HANDOFF.md               decision log and status
- autolab/config.toml, src/autolab/thresholds.toml
It never calls Modal and never writes.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from autolab.config import REPO_ROOT

AUTOLAB = REPO_ROOT / "autolab"
STATIC = Path(__file__).with_name("static")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="autolab dashboard", docs_url=None, redoc_url=None)


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _age_s(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(datetime.now(timezone.utc).timestamp() - t.timestamp(), 1)


def git_info() -> dict:
    def git(*a: str) -> str:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip()

    return {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "--short", "HEAD"),
            "subject": git("log", "-1", "--format=%s"), "date": git("log", "-1", "--format=%cI"),
            "dirty": bool(git("status", "--porcelain")),
            "log": [line.split(" ", 1) for line in git("log", "-15", "--format=%h %s").splitlines()]}


def load_experiments() -> list[dict]:
    out = []
    for p in sorted((AUTOLAB / "experiments").glob("*.json")):
        b = _json(p)
        if b:
            b.setdefault("id", p.stem)
            out.append(b)
    return out


def _flat(d: dict | None) -> dict:
    d = d or {}
    return {**d.get("model", {}), **d.get("optim", {}), **d.get("eval", {})}


def run_rows(calls: dict, experiments: list[dict]) -> list[dict]:
    membership = {}
    for e in experiments:
        base = _flat(e)
        for j in e.get("jobs", []):
            membership[j["run_id"]] = (e, j, base)
    rows = []
    for run_id, c in calls.items():
        d = AUTOLAB / "runs" / run_id
        report = _json(d / "report.json") or {}
        diag = _json(d / "diagnosis.json") or {}
        live = _json(d / "live.json") if c["state"] == "pending" else None
        req = c.get("request", {})
        cfg = {**req.get("model", {}), **req.get("optim", {})}
        e, j, base = membership.get(run_id, (None, None, None))
        diff = ({k: v for k, v in _flat(j).items()} if j else {})
        s, sc, perf = report.get("summary", {}), report.get("scale", {}), report.get("performance", {})
        top = max(diag.get("labels", []), key=lambda lab: lab["confidence"], default=None)
        rows.append({
            "run_id": run_id,
            "experiment": e["id"] if e else "adhoc",
            "variant": (j or {}).get("variant") or "-",
            "budget_name": (j or {}).get("budget") or "-",
            "state": c["state"],
            "error": c.get("error"),
            "gpu": c.get("gpu"),
            "seed": req.get("seed"),
            "submitted_at": c.get("submitted_at"),
            "tokens_budget": req.get("budget", {}).get("tokens"),
            "wall_cap_s": req.get("budget", {}).get("wall_clock_s"),
            "config": cfg,
            "diff": diff,
            "usd": c.get("usd"),
            "usd_estimate": c.get("usd_estimate"),
            "budget_hit": c.get("budget_hit"),
            "full_val": s.get("final_full_val_loss"),
            "best_val": s.get("best_val_loss"),
            "train_ema": s.get("final_train_loss_ema"),
            "gap": s.get("gap"),
            "val_rel_tail": (s.get("val_slope_tail") or {}).get("rel_change"),
            "tokens_seen": sc.get("tokens_seen"),
            "epochs": sc.get("epochs"),
            "params": sc.get("params"),
            "tok_per_param": sc.get("tokens_per_param"),
            "tokens_per_sec": perf.get("tokens_per_sec"),
            "train_wall_s": perf.get("train_wall_s"),
            "wall_s": perf.get("wall_s"),
            "ended_at": report.get("identity", {}).get("end_time"),
            "started_at": report.get("identity", {}).get("start_time"),
            "spikes": (report.get("health", {}).get("spikes") or {}).get("count"),
            "nan": report.get("health", {}).get("nan_or_inf"),
            "diagnosis": top["name"] if top else None,
            "diagnosis_conf": top["confidence"] if top else None,
            "live": ({"step": live.get("step"), "total_steps": live.get("total_steps"),
                      "steps_per_sec": live.get("steps_per_sec"), "eta_sec": live.get("eta_sec"),
                      "eval_val_loss": live.get("eval_val_loss"), "updated": live.get("updated"),
                      "age_s": _age_s(live.get("updated"))} if live else None),
        })
    return rows


def _rank(values: dict[str, float]) -> dict[str, int]:
    return {k: i + 1 for i, (k, _) in enumerate(sorted(values.items(), key=lambda kv: kv[1]))}


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 3:
        return None
    ra, rb = _rank({k: a[k] for k in keys}), _rank({k: b[k] for k in keys})
    return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in keys) / (n * (n * n - 1))


def analyze(exp: dict, rows: list[dict]) -> dict:
    mine = [r for r in rows if r["experiment"] == exp["id"]]
    groups: dict[tuple[str, str], list[float]] = {}
    for r in mine:
        if r["full_val"] is not None:
            groups.setdefault((r["variant"], r["budget_name"]), []).append(r["full_val"])
    table = [{"variant": v, "budget": b, "n": len(xs), "mean": mean(xs),
              "std": stdev(xs) if len(xs) > 1 else None, "values": xs} for (v, b), xs in sorted(groups.items())]
    out = {"groups": table, "done": sum(r["state"] != "pending" for r in mine), "total": len(exp.get("jobs", [])),
           "usd": round(sum(r["usd"] or 0 for r in mine), 3)}
    a = exp.get("analysis", {})
    if a.get("kind") == "noise_and_ranking":
        base = a.get("baseline_variant", "base")
        noise = {g["budget"]: {"n": g["n"], "mean": g["mean"], "std": g["std"]} for g in table if g["variant"] == base}
        screen = {g["variant"]: g["mean"] for g in table if g["budget"] == a.get("screen", "screen")}
        full = {g["variant"]: g["mean"] for g in table if g["budget"] == a.get("full", "full")}
        rs, rf = _rank(screen), _rank(full)
        ranking = [{"variant": v, "screen": screen.get(v), "full": full.get(v), "screen_rank": rs.get(v),
                    "full_rank": rf.get(v),
                    "screen_delta_vs_base": (screen[v] - screen[base]) if v in screen and base in screen else None,
                    "full_delta_vs_base": (full[v] - full[base]) if v in full and base in full else None}
                   for v in sorted(set(screen) | set(full), key=lambda v: (full.get(v, math.inf), v))]
        out.update(noise=noise, ranking=ranking, spearman=spearman(screen, full))
    return out


def decisions() -> list[dict]:
    try:
        text = (AUTOLAB / "HANDOFF.md").read_text()
    except OSError:
        return []
    section = text.split("## Decision log", 1)[-1]
    items, cur = [], None
    for line in section.splitlines():
        m = re.match(r"^- (\d{4}-\d{2}-\d{2})(?: \(([^)]+)\))?:\s*(.*)", line)
        if m:
            cur = {"date": m.group(1), "tag": m.group(2) or "", "text": m.group(3)}
            items.append(cur)
        elif cur is not None and line.strip():
            cur["text"] += " " + line.strip()
    return items[::-1]


def status_bullets() -> list[str]:
    try:
        text = (AUTOLAB / "HANDOFF.md").read_text()
    except OSError:
        return []
    section = text.split("## Where things stand", 1)[-1].split("\n## ", 1)[0]
    bullets, cur = [], None
    for line in section.splitlines():
        if line.startswith("- "):
            cur = line[2:].strip()
            bullets.append(cur)
        elif cur is not None and line.strip():
            bullets[-1] += " " + line.strip()
    return bullets


def settings() -> dict:
    out = {}
    for name, path in (("config", AUTOLAB / "config.toml"),
                       ("thresholds", REPO_ROOT / "src" / "autolab" / "thresholds.toml")):
        try:
            out[name] = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            out[name] = {"error": str(exc)}
    return out


def overview() -> dict:
    calls = _json(AUTOLAB / "state" / "modal_calls.json", {}) or {}
    exps = load_experiments()
    rows = run_rows(calls, exps)
    cfg = settings().get("config", {})
    cap = cfg.get("modal", {}).get("max_usd")
    spent = sum(r["usd"] or 0 for r in rows if r["state"] != "pending")
    pending = sum(r["usd_estimate"] or 0 for r in rows if r["state"] == "pending")
    timeline, cum = [], 0.0
    for r in sorted((r for r in rows if r["state"] != "pending" and r["ended_at"]), key=lambda r: r["ended_at"]):
        cum += r["usd"] or 0
        timeline.append([r["ended_at"], round(cum, 4), r["run_id"]])
    finished = [r for r in rows if r["full_val"] is not None]
    best = min(finished, key=lambda r: r["full_val"], default=None)
    daemon = _json(AUTOLAB / "state" / "daemon.json")
    if daemon:
        daemon["age_s"] = _age_s(daemon.get("updated"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "daemon": daemon,
        "git": git_info(),
        "status": status_bullets(),
        "spend": {"spent": round(spent, 4), "pending_estimate": round(pending, 4), "cap": cap,
                  "timeline": timeline,
                  "by_gpu": {g: round(sum(r["usd"] or 0 for r in rows if r["gpu"] == g and r["state"] != "pending"), 4)
                             for g in sorted({r["gpu"] for r in rows if r["gpu"]})},
                  "gpu_hours": round(sum((r["wall_s"] or 0) for r in rows) / 3600, 3)},
        "counts": {s: sum(r["state"] == s for r in rows) for s in ("pending", "finished", "failed")},
        "best": ({k: best[k] for k in ("run_id", "full_val", "experiment", "variant", "budget_name")} if best else None),
        "runs": rows,
        "experiments": [{k: e.get(k) for k in ("id", "title", "purpose", "gpu", "dataset_id", "model", "optim", "eval")}
                        | {"jobs": len(e.get("jobs", [])), "analysis": analyze(e, rows)} for e in exps],
        "decisions": decisions(),
        "settings": settings(),
    }


@app.get("/api/overview")
def api_overview() -> JSONResponse:
    return JSONResponse(overview())


@app.get("/api/run/{run_id}")
def api_run(run_id: str) -> JSONResponse:
    if not SAFE_ID.match(run_id):
        raise HTTPException(400, "bad run id")
    d = AUTOLAB / "runs" / run_id
    call = (_json(AUTOLAB / "state" / "modal_calls.json", {}) or {}).get(run_id)
    if call is None and not d.exists():
        raise HTTPException(404, "unknown run")
    log = ""
    for name in ("train.log", "train.tail.log"):
        try:
            log = (d / name).read_text(errors="replace")[-12_000:]
            break
        except OSError:
            continue
    return JSONResponse({"run_id": run_id, "call": call, "launch": _json(d / "launch.json"),
                         "report": _json(d / "report.json"), "diagnosis": _json(d / "diagnosis.json"),
                         "live": _json(d / "live.json"), "log_tail": log})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html", headers={"Cache-Control": "no-store"})
