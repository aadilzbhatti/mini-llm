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

from fastapi import FastAPI, HTTPException, Request
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


def _evolve_label(run_id: str) -> str | None:
    """ev-<session>-<program>-<stage>-s<seed> -> "evolve <session>"."""
    m = re.match(r"^ev-(.+?)-(p\d+|data-.+)$", run_id)
    return f"evolve {m.group(1)}" if m else None


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
            "experiment": e["id"] if e else (_evolve_label(run_id) or "adhoc"),
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


STAGE_ORDER = ["static", "cpu", "params", "screen", "full", "confirm", "done"]


def evolve_root() -> Path:
    base = AUTOLAB / "state" / "evolve"
    try:
        return base / (base / "ACTIVE").read_text().strip()
    except OSError:
        return base / "default"


def evolve_view() -> dict | None:
    root = evolve_root()
    session = _json(root / "session.json")
    if not session:
        return None
    progs = [_json(p) for p in sorted((root / "programs").glob("*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)]
    progs = [p for p in progs if p]
    inc = next((p for p in progs if p["id"] == session["incumbent"]), None)
    sigma = max(session["noise"]["full"]["std"], float(settings().get("config", {}).get("evolve", {}).get("noise_floor", 0.0)))
    inc_mean = (inc or {}).get("scores", {}).get("full_mean")
    rows = []
    for p in progs:
        sc = p.get("scores", {})
        passed = [st["stage"] for st in p.get("stages", []) if st.get("ok")]
        rows.append({
            "id": p["id"], "parent": p.get("parent_id"), "by": p.get("created_by"), "created": p.get("created_at"),
            "rationale": p.get("rationale", ""), "stage": p.get("stage"), "status": p.get("status"), "reason": p.get("reason", ""),
            "screen": sc.get("screen_loss"), "full_mean": sc.get("full_mean"), "n_seeds": sc.get("n_seeds", 0),
            "delta_sigma": ((sc["full_mean"] - inc_mean) / sigma) if sc.get("full_mean") is not None and inc_mean and sigma else None,
            "params": sc.get("params"), "tokens_per_sec": sc.get("tokens_per_sec"),
            "furthest": p.get("stage") if p.get("status") not in ("evaluated", "contender", "accepted") else "done",
            "passed": passed, "hparams": p.get("hparams"), "runs": p.get("runs", {}),
        })
    def furthest(r: dict) -> int:
        """Index of the last stage a program got to (its current one, or 'done')."""
        if r["status"] in ("evaluated", "contender", "accepted"):
            return STAGE_ORDER.index("confirm") if r["status"] != "evaluated" else STAGE_ORDER.index("full")
        return STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 0

    children = [r for r in rows if r["id"] != "p0"]
    funnel = [{"stage": st, "reached": sum(furthest(r) >= i for r in children),
               "rejected_here": sum(r["status"] == "rejected" and r["stage"] == st for r in children)}
              for i, st in enumerate(STAGE_ORDER[:-1])]
    sessions = sorted(q.name for q in (AUTOLAB / "state" / "evolve").iterdir() if (q / "session.json").exists())
    return {"session": session, "sessions": sessions, "incumbent": inc and inc["id"], "bar": (inc_mean - 2 * sigma) if inc_mean else None,
            "programs": rows, "funnel": funnel,
            "counts": {k: sum(r["status"] == k for r in rows if r["id"] != "p0")
                       for k in ("queued", "running", "blocked", "rejected", "evaluated", "contender", "accepted")}}


def incumbent_view() -> dict | None:
    """The active evolve session's accepted best: the number autolab optimizes and reports."""
    root = evolve_root()
    session = _json(root / "session.json")
    if not session:
        return None
    p = _json(root / "programs" / f"{session['incumbent']}.json") or {}
    sc = p.get("scores", {})
    return {"session": session.get("name"), "program": session["incumbent"], "full_mean": sc.get("full_mean"),
            "n_seeds": sc.get("n_seeds"), "dataset": session.get("dataset_id"),
            "tokens": session["budgets"]["full_tokens"], "sigma": session["noise"]["full"]["std"]}


def controller_view(calls: dict) -> dict:
    ctl = _json(AUTOLAB / "state" / "controller.json", {}) or {}
    cfg = settings().get("config", {}).get("controller", {})
    since = datetime.now(timezone.utc).timestamp() - 86400

    def recent(ts):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() >= since
        except (AttributeError, ValueError):
            return False

    llm = []
    try:
        llm = [json.loads(x) for x in (AUTOLAB / "state" / "llm_spend.jsonl").read_text().splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        pass
    spend = (sum(c.get("usd") or 0 for c in calls.values() if c["state"] != "pending" and recent(c.get("submitted_at")))
             + sum(c.get("usd_estimate") or 0 for c in calls.values() if c["state"] == "pending")
             + sum(e.get("usd") or 0 for e in llm if recent(e.get("at"))))
    ov = ctl.get("daily_usd_override") or {}
    daily = cfg.get("daily_usd")
    if ov.get("until") and _age_s(ov["until"]) is not None and _age_s(ov["until"]) < 0:
        daily = ov.get("usd")
    modal_cap = settings().get("config", {}).get("modal", {}).get("max_usd")
    modal_spent = sum(c.get("usd") or 0 for c in calls.values() if c["state"] != "pending")
    modal_pending = sum(c.get("usd_estimate") or 0 for c in calls.values() if c["state"] == "pending")
    return {"enabled": ctl.get("enabled", False), "paused_until": ctl.get("paused_until"),
            "base_daily_usd": cfg.get("daily_usd"), "modal_cap": modal_cap,
            "modal_spent": round(modal_spent, 2), "modal_pending": round(modal_pending, 2),
            "override": ov if daily == ov.get("usd") else None,
            "pause_reason": ctl.get("pause_reason"), "data_flow": ctl.get("data_flow", {}),
            "ladder": ctl.get("ladder", {}),
            "spend_24h": round(spend, 3), "daily_usd": daily, "max_in_flight": cfg.get("max_in_flight")}


def notebook_view(limit: int = 200) -> list[dict]:
    try:
        lines = (AUTOLAB / "notebook.jsonl").read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[::-1]


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
    best = min(finished, key=lambda r: r["full_val"], default=None)  # lowest single run, any regime
    llm_calls = []
    try:
        llm_calls = [json.loads(line) for line in (AUTOLAB / "state" / "llm_spend.jsonl").read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError):
        pass
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
        "llm": {"calls": len(llm_calls), "failed": sum(not c.get("ok") for c in llm_calls),
                "usd": round(sum(c.get("usd") or 0 for c in llm_calls), 4)},
        "best": ({**{k: best[k] for k in ("run_id", "full_val", "experiment", "variant", "budget_name", "tokens_seen")},
                  "dataset": (calls.get(best["run_id"], {}).get("request") or {}).get("dataset_id")} if best else None),
        "incumbent": incumbent_view(),
        "runs": rows,
        "experiments": [{k: e.get(k) for k in ("id", "title", "purpose", "gpu", "dataset_id", "model", "optim", "eval")}
                        | {"jobs": len(e.get("jobs", [])), "analysis": analyze(e, rows)} for e in exps],
        "decisions": decisions(),
        "settings": settings(),
        "evolve": evolve_view(),
        "controller": controller_view(calls),
        "notebook": notebook_view(),
    }


PIPE_COLUMNS = ["proposing", "cpu", "screen", "full", "confirm", "finished"]


def _run_progress(run_id: str, calls: dict) -> dict:
    c = calls.get(run_id, {})
    d = AUTOLAB / "runs" / run_id
    rep = _json(d / "report.json")
    if rep:
        return {"run_id": run_id, "state": c.get("state", "finished"), "full_val": rep["summary"].get("final_full_val_loss")}
    live = _json(d / "live.json")
    if live:
        return {"run_id": run_id, "state": "training", "step": live.get("step"), "total": live.get("total_steps"),
                "eval_val": live.get("eval_val_loss"), "eta": live.get("eta_sec"), "age_s": _age_s(live.get("updated"))}
    return {"run_id": run_id, "state": c.get("state", "queued") if c else "queued",
            "error": (c.get("error") or "")[:200] if c.get("state") == "failed" else None}


def _session_info(root: Path) -> dict | None:
    s = _json(root / "session.json")
    if not s:
        return None
    p0 = _json(root / "programs" / "p0.json") or {}
    return {"name": s.get("name"), "created": s.get("created"), "dataset": s.get("dataset_id"),
            "tokens": s["budgets"]["full_tokens"], "why": p0.get("rationale", ""), "incumbent": s.get("incumbent")}


def live_view() -> dict:
    """Everything the Live tab shows: now, pipeline, proposals, feed."""
    calls = _json(AUTOLAB / "state" / "modal_calls.json", {}) or {}
    act = _json(AUTOLAB / "state" / "activity.json", {}) or {}
    cur = act.get("current")
    if cur:
        cur["age_s"] = _age_s(cur.get("at"))
    base = AUTOLAB / "state" / "evolve"
    cards = []
    for sdir in sorted(p for p in base.iterdir() if (p / "session.json").exists()) if base.exists() else []:
        sess = _json(sdir / "session.json") or {}
        inc = sess.get("incumbent")
        for f in (sdir / "programs").glob("*.json"):
            p = _json(f)
            if not p or p.get("parent_id") is None:
                continue
            done = p["status"] in ("rejected", "evaluated", "contender", "accepted")
            if done:
                last = max((st.get("at", "") for st in p.get("stages", [])), default=p.get("created_at", ""))
                age = _age_s(last)
                if age is None or age > 6 * 3600:
                    continue  # only recent finishes on the board
                col = "finished"
            else:
                col = {"static": "cpu", "params": "cpu"}.get(p["stage"], p["stage"])
            runs = [_run_progress(r, calls) for r in p.get("runs", {}).get(p["stage"], [])] if not done else []
            sc = p.get("scores", {})
            cards.append({"session": sess.get("name"), "id": p["id"], "parent": p.get("parent_id"),
                          "by": p.get("created_by"), "status": p["status"], "stage": p["stage"], "column": col,
                          "rationale": " ".join((p.get("rationale") or "").split())[:280], "reason": p.get("reason", "")[:240],
                          "screen": sc.get("screen_loss"), "full_mean": sc.get("full_mean"), "n_seeds": sc.get("n_seeds"),
                          "incumbent": p["id"] == inc, "runs": runs, "created": p.get("created_at"),
                          "instruction": (p.get("meta") or {}).get("instruction"),
                          "cost_usd": (p.get("meta") or {}).get("cost_usd")})
    if cur and cur.get("kind") == "propose" and (cur.get("age_s") or 1e9) < 900:
        cards.append({"session": None, "id": "…", "column": "proposing", "status": "running", "stage": "proposing",
                      "by": cur.get("model"), "parent": cur.get("parent"), "rationale": cur.get("text"), "runs": []})
    active = evolve_root()
    llm = []
    for f in sorted((AUTOLAB / "state" / "llm").glob("*/*.json"), key=lambda f: f.name, reverse=True)[:12]:
        r = _json(f) or {}
        reply = r.get("reply") or {}
        tag = r.get("tag", "")
        llm.append({"at": r.get("at"), "model": r.get("model") or r.get("model_requested"), "cost_usd": r.get("cost_usd"),
                    "duration_s": r.get("duration_s"), "error": r.get("error"), "session": f.parent.name,
                    "program": "p" + tag.rsplit("-", 1)[-1] if tag.rsplit("-", 1)[-1].isdigit() else None,
                    "rationale": " ".join((reply.get("rationale") or "").split())[:400],
                    "expected": reply.get("expected_effect"), "hparams": reply.get("hparams"),
                    "n_diffs": len(reply.get("diffs") or [])})
    try:
        log_tail = (AUTOLAB / "state" / "daemon.log").read_text(errors="replace").splitlines()[-40:][::-1]
    except OSError:
        log_tail = []
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "current": cur,
            "history": act.get("history", [])[:40], "controller": controller_view(calls),
            "daemon": _json(AUTOLAB / "state" / "daemon.json"), "columns": PIPE_COLUMNS, "cards": cards,
            "blocked": ((_json(AUTOLAB / "state" / "daemon.json") or {}).get("controller") or {}).get("blocked"),
            "llm": llm, "log": log_tail, "active_session": active.name,
            "session_info": _session_info(active),
            "pending_runs": sum(c["state"] == "pending" for c in calls.values())}


BUDGET_BOUNDS = {"daily_usd": (0.0, 200.0), "max_usd": (0.0, 1000.0), "max_in_flight": (1, 12)}
CONFIG_KEYS = {"daily_usd": "controller", "max_in_flight": "controller", "max_usd": "modal"}


def set_config_value(path: Path, section: str, key: str, value) -> None:
    """Replace `key = ...` inside [section] of a TOML file, keeping comments; validate the result."""
    lines = path.read_text().splitlines(keepends=True)
    current, done = None, False
    for i, line in enumerate(lines):
        m = re.match(r"^\[([^\]]+)\]\s*$", line.strip())
        if m:
            current = m.group(1)
            continue
        if current == section and re.match(rf"^{key}\s*=", line):
            comment = line.split("#", 1)[1] if "#" in line else ""
            head = f"{key} = {value}"
            lines[i] = (head.ljust(max(len(head) + 1, 34)) + ("# " + comment.strip() if comment else "")).rstrip() + "\n"
            done = True
            break
    if not done:
        raise ValueError(f"[{section}] {key} not found in {path.name}")
    text = "".join(lines)
    got = tomllib.loads(text)[section][key]
    if got != value:
        raise ValueError(f"write check failed for {key}")
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _bounded(key: str, value):
    lo, hi = BUDGET_BOUNDS[key]
    v = int(value) if isinstance(lo, int) else float(value)
    if not lo <= v <= hi:
        raise HTTPException(400, f"{key} must be between {lo} and {hi}")
    return v


@app.post("/api/budget")
async def api_budget(request: Request) -> JSONResponse:
    """Budget controls from the page. Tailnet-only; the custom header blocks cross-site form posts."""
    if request.headers.get("x-autolab") != "1":
        raise HTTPException(403, "missing X-Autolab header")
    from datetime import timedelta

    from autolab import controller as ctl

    body = await request.json()
    action = body.get("action")
    if action == "override":
        daily = _bounded("daily_usd", body.get("daily_usd"))
        hours = float(body.get("hours", 12))
        if not 0.5 <= hours <= 72:
            raise HTTPException(400, "hours must be between 0.5 and 72")
        c = ctl.load_control()
        until = ctl.now() + timedelta(hours=hours)
        c["daily_usd_override"] = {"usd": daily, "until": ctl.iso(until), "set": ctl.iso(ctl.now()), "by": "dashboard"}
        ctl.save_control(c)
        ctl.note("budget_override", daily_usd=daily, until=ctl.iso(until), by="dashboard")
        return JSONResponse({"ok": True, "message": f"daily budget ${daily:g} until {ctl.iso(until)}"})
    if action == "resume":
        c = ctl.load_control()
        c.update(enabled=True, paused_until=None, pause_reason=None)
        ctl.save_control(c)
        ctl.note("resumed", by="dashboard")
        return JSONResponse({"ok": True, "message": "resumed: proposals continue on the next cycle"})
    if action == "clear_override":
        c = ctl.load_control()
        c.pop("daily_usd_override", None)
        ctl.save_control(c)
        ctl.note("budget_override_cleared", by="dashboard")
        return JSONResponse({"ok": True, "message": "one-time raise cleared"})
    if action == "permanent":
        changed = {}
        for key in ("daily_usd", "max_usd", "max_in_flight"):
            if body.get(key) not in (None, ""):
                value = _bounded(key, body[key])
                set_config_value(AUTOLAB / "config.toml", CONFIG_KEYS[key], key, value)
                changed[key] = value
        if not changed:
            raise HTTPException(400, "nothing to change")
        ctl.note("settings_changed", by="dashboard", **changed)
        return JSONResponse({"ok": True, "message": "saved to autolab/config.toml: " +
                             ", ".join(f"{k} = {v}" for k, v in changed.items())})
    raise HTTPException(400, "unknown action")


@app.get("/api/live")
def api_live() -> JSONResponse:
    return JSONResponse(live_view())


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


@app.get("/api/program/{pid}")
def api_program(pid: str) -> JSONResponse:
    import difflib

    if not SAFE_ID.match(pid):
        raise HTTPException(400, "bad program id")
    root = evolve_root() / "programs"
    p = _json(root / f"{pid}.json")
    if p is None:
        raise HTTPException(404, "unknown program")
    parent = _json(root / f"{p['parent_id']}.json") if p.get("parent_id") else None
    diff = []
    if parent:
        for key in sorted(set(p["blocks"]) | set(parent["blocks"])):
            a, b = parent["blocks"].get(key, ""), p["blocks"].get(key, "")
            if a != b:
                diff += list(difflib.unified_diff(a.splitlines(), b.splitlines(), f"{parent['id']}/{key}",
                                                  f"{p['id']}/{key}", lineterm="", n=2))
    hp_diff = {k: [parent["hparams"].get(k), v] for k, v in p["hparams"].items()
               if parent and parent["hparams"].get(k) != v}
    return JSONResponse({"program": {k: v for k, v in p.items() if k != "blocks"}, "diff": "\n".join(diff),
                         "hparams_diff": hp_diff})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html", headers={"Cache-Control": "no-store"})
