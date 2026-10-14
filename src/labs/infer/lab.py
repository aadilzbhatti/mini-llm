"""labs.infer: evolve the inference engine of a fixed checkpoint for speed on this Mac, quality held.

A copy of autolab's loop cut down to a cheap, local, noisy evaluator (copied, not abstracted: the shared core gets
extracted once a second lab exists). One program = one whole engine.py (see seed_engine.py for the API).

Cascade per program, one stage per daemon cycle, every GPU stage a subprocess of labs.infer.harness:
  static   parses, defines Engine, no forbidden names (autolab.program.static_violations), size cap
  quality  teacher-forced on fixed val windows vs the fp32 reference: delta loss <= max_delta_loss, top-1 agreement
           >= min_top1, generate() consistent with the engine's own logits, peak memory <= max_mem_mult x p0's
  bench    vs the incumbent, interleaved and paired (bench_rounds rounds, order alternating); promoted to confirm
           when the mean paired log speedup is > 0
  confirm  bench_rounds more rounds on a new seed; accepted iff over all rounds the mean log speedup >= min_gain
           and its lower bound (mean - accept_se x SE over round pairs) > 0; else a contender (faster, unproven)
           Either bench stage ends a program whose worst scenario falls below 1 - max_scenario_regression of the
           incumbent's (a geometric mean must not trade one scenario away for another).
Speeds are geometric means over the scenarios (harness.SCENARIOS + PREFILL); x_p0 chains each accepted ratio.

Proposals: Claude (autolab.llm.call) with the parent's code, scores, the best others and recent failures, within a
rolling-24h budget shared by all labs (autolab/state/labs/llm_spend.jsonl). The owner's pause (state/labs/
control.json "hold") and the owner's training runs on this GPU (autolab.gpu) both stop the loop.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autolab.config import REPO_ROOT

LABS_STATE = REPO_ROOT / "autolab" / "state" / "labs"
STATE = LABS_STATE / "infer"
DATA = REPO_ROOT / "autolab" / "data" / "labs" / "infer"
CONFIG = REPO_ROOT / "autolab" / "labs.toml"
SEED = Path(__file__).with_name("seed_engine.py")
DONE = ("rejected", "evaluated", "contender", "accepted")
OWNER_RUNS = Path.home() / "dev" / "wiki-llm" / "runs"


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime | None = None) -> str:
    return (t or now()).isoformat(timespec="seconds")


def cfg() -> dict:
    c = tomllib.loads(CONFIG.read_text())
    return {**c.get("labs", {}), **c.get("infer", {}), "llm": c.get("llm", {})}


@dataclass
class Program:
    id: str
    parent_id: str | None
    code: str
    rationale: str = ""
    expected: str = ""
    created_by: str = "human"
    created_at: str = field(default_factory=iso)
    stage: str = "static"  # static | quality | bench | confirm | done
    status: str = "queued"  # queued | rejected | evaluated | contender | accepted
    reason: str = ""
    stages: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def save(p: Program) -> None:
    (STATE / "programs").mkdir(parents=True, exist_ok=True)
    tmp = STATE / "programs" / f".{p.id}.json.tmp"
    tmp.write_text(json.dumps(asdict(p), indent=2))
    tmp.replace(STATE / "programs" / f"{p.id}.json")


def programs() -> dict[str, Program]:
    d = STATE / "programs"
    out = {f.stem: Program(**json.loads(f.read_text())) for f in d.glob("p*.json")} if d.exists() else {}
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def load_lab() -> dict:
    try:
        return json.loads((STATE / "lab.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_lab(lab: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "lab.json").write_text(json.dumps(lab, indent=2))


def note(event: str, **fields) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "notebook.jsonl", "a") as fh:
        fh.write(json.dumps({"at": iso(), "event": event, **fields}) + "\n")


def control() -> dict:
    try:
        return json.loads((LABS_STATE / "control.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def set_hold(on: bool, by: str) -> None:
    c = control()
    if on and not c.get("hold"):
        c["hold"] = {"at": iso(), "by": by}
    elif not on:
        c.pop("hold", None)
    LABS_STATE.mkdir(parents=True, exist_ok=True)
    (LABS_STATE / "control.json").write_text(json.dumps(c, indent=2))


# --- setup ------------------------------------------------------------------------------------------------


def paths(c: dict) -> tuple[Path, Path]:
    return DATA / c["checkpoint"], DATA / c["val"]


def harness(mode: str, out: Path, c: dict, timeout: float, **args) -> dict:
    ckpt, val = paths(c)
    cmd = [sys.executable, "-m", "labs.infer.harness", mode, "--ckpt", str(ckpt), "--val", str(val), "--out", str(out)]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    out.unlink(missing_ok=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
                           env={**os.environ, "PYTHONHASHSEED": "0"})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout:.0f}s"}
    if not out.exists():
        return {"ok": False, "error": f"harness exited {r.returncode}: {(r.stderr or r.stdout)[-400:]}"}
    return json.loads(out.read_text())


def init(log=print) -> None:
    """Reference numbers for the checkpoint, and p0 (the seed engine) measured as the first incumbent."""
    c = cfg()
    STATE.mkdir(parents=True, exist_ok=True)
    if not (STATE / "ref.json").exists():
        ref = harness("reference", STATE / "ref.json", c, 900)
        if not ref.get("ok"):
            raise RuntimeError(f"reference failed: {ref.get('error')}")
    if "p0" in programs():
        return
    p = Program("p0", None, SEED.read_text(), rationale="Seed: the reference model's inference path, unoptimized.")
    save(p)
    advance(p, log=log)  # static + quality
    advance(p, log=log)
    q = p.scores.get("quality") or {}
    b = harness("bench", STATE / "work" / "p0-bench.json", c, 1800, engine=STATE / "work" / "p0.py",
                rounds=c["bench_rounds"], seed=1) if (STATE / "work" / "p0.py").exists() else {}
    if p.status == "rejected" or not b.get("ok"):
        raise RuntimeError(f"p0 failed: {p.reason or b.get('error')}")
    p.stage, p.status, p.reason = "done", "accepted", "seed"
    p.scores.update(speed=b["speed"]["cand"], x_p0=1.0, peak_bytes=max(q.get("peak_bytes", 0), b["peak_bytes"]))
    save(p)
    save_lab({"incumbent": "p0", "history": [{"at": iso(), "program": "p0", "x_p0": 1.0}], "p0_peak": p.scores["peak_bytes"]})
    note("initialized", incumbent="p0", speed=p.scores["speed"])
    log(f"infer: p0 measured {p.scores['speed']}")


# --- cascade ----------------------------------------------------------------------------------------------


def record(p: Program, stage: str, ok: bool, detail: str, **extra) -> None:
    p.stages.append({"stage": stage, "ok": ok, "at": iso(), "detail": detail, **extra})


def reject(p: Program, stage: str, why: str) -> None:
    record(p, stage, False, why)
    p.stage, p.status, p.reason = "done", "rejected", f"{stage}: {why}"


def static_check(code: str, c: dict) -> str | None:
    import ast

    from autolab.program import static_violations

    if len(code) > c.get("max_code_bytes", 40_000):
        return f"engine.py is {len(code)} bytes (cap {c.get('max_code_bytes', 40_000)})"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    if not any(isinstance(n, ast.ClassDef) and n.name == "Engine" for n in tree.body):
        return "no top-level class Engine"
    bad = static_violations({"engine.py": code})
    return "; ".join(bad)[:400] if bad else None


def pair_stats(rounds: list[float]) -> tuple[float, float, int]:
    """Mean paired log speedup and its standard error over (even, odd) round pairs, which cancel the order effect."""
    pairs = [(rounds[i] + rounds[i + 1]) / 2 for i in range(0, len(rounds) - 1, 2)]
    m = statistics.fmean(pairs)
    se = statistics.stdev(pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else float("inf")
    return m, se, len(pairs)


def advance(p: Program, log=print) -> Program:
    """Run the program's next stage."""
    c, lab = cfg(), load_lab()
    work = STATE / "work"
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"{p.id}.py"
    if p.stage == "static":
        why = static_check(p.code, c)
        if why:
            reject(p, "static", why)
        else:
            src.write_text(p.code)
            record(p, "static", True, "ok")
            p.stage = "quality"
    elif p.stage == "quality":
        src.write_text(p.code)
        q = harness("quality", work / f"{p.id}-quality.json", c, c.get("quality_timeout_s", 900), engine=src,
                    ref=STATE / "ref.json")
        p.scores["quality"] = {k: v for k, v in q.items() if k != "trace"}
        p0_peak = lab.get("p0_peak")
        if not q.get("ok"):
            reject(p, "quality", f"crashed: {q.get('error')}")
        elif not q["finite"]:
            reject(p, "quality", "non-finite logits")
        elif q["generate_error"]:
            reject(p, "quality", q["generate_error"])
        elif q["delta_loss"] > c["max_delta_loss"]:
            reject(p, "quality", f"loss {q['mean_loss']:.5f} is {q['delta_loss']:+.5f} vs the reference (max +{c['max_delta_loss']})")
        elif q["top1_agreement"] < c["min_top1"]:
            reject(p, "quality", f"top-1 agreement {q['top1_agreement']:.4f} < {c['min_top1']}")
        elif p0_peak and q["peak_bytes"] > c["max_mem_mult"] * p0_peak:
            reject(p, "quality", f"peak memory {q['peak_bytes'] / 2**20:.0f} MiB > {c['max_mem_mult']}x p0's")
        else:
            record(p, "quality", True, f"delta loss {q['delta_loss']:+.5f}, top-1 {q['top1_agreement']:.4f}, "
                   f"peak {q['peak_bytes'] / 2**20:.0f} MiB")
            p.stage = "bench" if p.id != "p0" else "done"
    elif p.stage in ("bench", "confirm"):
        inc = lab["incumbent"]
        (work / f"{inc}.py").write_text(programs()[inc].code)
        src.write_text(p.code)
        seed = random.SystemRandom().randrange(1, 2**31)
        b = harness("bench", work / f"{p.id}-{p.stage}.json", c, c.get("bench_timeout_s", 1800), engine=src,
                    against=work / f"{inc}.py", rounds=c["bench_rounds"], seed=seed)
        if not b.get("ok"):
            reject(p, p.stage, f"crashed: {b.get('error')}")
        elif b["errors"]:
            reject(p, p.stage, f"timed generate() output failed the check: {b['errors'][0]}")
        else:
            runs = p.scores.setdefault("bench_rounds", {}).setdefault(inc, [])
            runs += b["round_log_ratio"]
            per = p.scores.setdefault("scenario_rounds", {}).setdefault(inc, {})
            for s, v in b["scenario_round_log_ratio"].items():
                per.setdefault(s, []).extend(v)
            worst_s, worst = min(((s, statistics.fmean(v)) for s, v in per.items()), key=lambda x: x[1])
            p.scores.setdefault("bench_speed", []).append(b["speed"]["cand"])
            p.scores["peak_bytes"] = max(p.scores.get("peak_bytes", 0), b["peak_bytes"])
            m, se, n = pair_stats(runs)
            p.scores.update(vs=inc, log_speedup=m, se=se, speedup=math.exp(m),
                            x_p0=programs()[inc].scores.get("x_p0", 1.0) * math.exp(m),
                            per_scenario={s: math.exp(statistics.fmean(v)) for s, v in per.items()})
            detail = f"{math.exp(m):.3f}x {inc} (±{se:.3f} log, {n} round pairs); " + ", ".join(
                f"{s} {math.exp(statistics.fmean(v)):.2f}x" for s, v in per.items())
            floor = math.log(1 - c["max_scenario_regression"])
            if worst < floor:  # a geometric mean must not buy one scenario's speed with another's
                record(p, p.stage, False, detail)
                p.stage, p.status, p.reason = "done", "evaluated", \
                    f"{worst_s} regressed to {math.exp(worst):.2f}x (floor {1 - c['max_scenario_regression']:.2f}x): {detail}"
            elif p.stage == "bench":
                record(p, "bench", m > 0, detail)
                if m > 0:
                    p.stage = "confirm"
                else:
                    p.stage, p.status, p.reason = "done", "evaluated", f"not faster: {detail}"
            else:
                ok = m >= c["min_gain"] and m - c["accept_se"] * se > 0
                record(p, "confirm", ok, detail)
                if ok:
                    p.stage, p.status, p.reason = "done", "accepted", detail
                    p.scores["speed"] = b["speed"]["cand"]
                    lab["incumbent"] = p.id
                    lab.setdefault("history", []).append({"at": iso(), "program": p.id, "x_p0": p.scores["x_p0"]})
                    save_lab(lab)
                    note("accepted", program=p.id, x_p0=p.scores["x_p0"], detail=detail)
                    log(f"infer: {p.id} ACCEPTED ({detail}; {p.scores['x_p0']:.2f}x p0)")
                else:
                    p.stage, p.status, p.reason = "done", "contender" if m > 0 else "evaluated", detail
    save(p)
    if p.status in DONE:
        note("program_done", program=p.id, status=p.status, reason=p.reason[:300])
        log(f"infer: {p.id} {p.status}: {p.reason[:200]}")
    return p


# --- proposals ---------------------------------------------------------------------------------------------

SYSTEM = """You optimize the inference engine of a small GPT-style language model for speed on one machine, \
keeping its outputs faithful. You get the current engine.py and its measurements; you return one focused change \
as SEARCH/REPLACE edits (or a whole new file) plus a short rationale. Be concrete about the mechanism you expect \
to save time, and keep the change small enough that a failure teaches something."""

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "what changes and why it should be faster (2-5 sentences)"},
        "expected": {"type": "string", "description": "which scenarios should speed up, roughly how much"},
        "diffs": {"type": "array", "items": {"type": "object", "properties": {
            "search": {"type": "string"}, "replace": {"type": "string"}}, "required": ["search", "replace"]}},
        "code": {"type": "string", "description": "a whole new engine.py instead of diffs (for rewrites only)"},
    },
    "required": ["rationale", "expected"],
}

INSTRUCTIONS = {
    "bandwidth": "Decode at batch 1 is bound by reading the weights for every token. Find a way to move fewer bytes.",
    "overhead": "Each decode step launches many small kernels and runs Python per layer. Cut per-token overhead.",
    "prefill": "Look at the prefill path (1024 tokens in one call) and the attention/RoPE implementation.",
    "open": "Pick the most promising unexplored idea given what has been tried.",
    "refine": "Take the parent's own idea further, or fix what kept its last attempt from paying off.",
}


def describe(p: Program) -> str:
    s = p.scores
    bits = [f"{p.id} ({p.status}{', ' + format(s['x_p0'], '.2f') + 'x p0' if s.get('x_p0') else ''})"]
    if s.get("per_scenario"):
        bits.append("vs " + s.get("vs", "?") + ": " + ", ".join(f"{k} {v:.2f}x" for k, v in s["per_scenario"].items()))
    q = s.get("quality") or {}
    if q.get("delta_loss") is not None:
        bits.append(f"delta loss {q['delta_loss']:+.5f}, top-1 {q['top1_agreement']:.4f}")
    return "; ".join(bits)


def build_prompt(parent: Program, progs: dict[str, Program], instruction: str, c: dict) -> str:
    import platform

    import torch

    lab = load_lab()
    inc = progs[lab["incumbent"]]
    c = {**c, "model": json.loads((STATE / "ref.json").read_text())["config"]}
    best = sorted((q for q in progs.values() if q.scores.get("x_p0") and q.id != parent.id and q.status != "rejected"),
                  key=lambda q: -q.scores["x_p0"])[:4]
    recent = [q for q in progs.values() if q.status in DONE and q.parent_id][-12:]
    speeds = inc.scores.get("speed") or {}
    return f"""## Task
Make engine.py faster on this machine without changing what the model computes (beyond the precision allowed below).

Machine: {platform.machine()} Apple M4 (10-core GPU, 16 GB unified memory, ~120 GB/s), PyTorch {torch.__version__} on MPS.
Model: decoder-only transformer, n_layer {c['model']['n_layer']}, n_embd {c['model']['n_embd']}, n_head {c['model']['n_head']}, \
vocab 50257, context 1024, RoPE, pre-LayerNorm, GELU MLP (4x), tied embedding/lm_head (lm_head also has a bias); \
~95M params, fp32 checkpoint (~380 MB).

## The contract (the harness is fixed; you only change engine.py)
`Engine(weights, config, device)`: weights is a dict of fp32 CPU tensors (the checkpoint's state dict), config a dict.
`prefill(ids)` -> (1, vocab) logits of the last position; ids (1, T) int64 on the device, 1 <= T <= 1024; resets state.
`step(token)` -> (1, vocab) logits after appending token (1, 1). Past 1024 tokens the context rolls (RoPE ring cache).
`generate(ids, n)` -> (1, n) int64 greedy tokens continuing ids. Each returned token must be (within {0.05} logits) the \
argmax of what prefill/step would give, which the harness checks.
Batch size is always 1. No file, OS, network or subprocess access (static check); torch and math are fine.

## How it is judged
Quality gate, teacher-forced through prefill/step on held-out text vs the fp32 reference: mean loss may rise by at \
most {c['max_delta_loss']} nats, top-1 agreement >= {c['min_top1']}; peak memory <= {c['max_mem_mult']}x the seed's. \
Lower precision (fp16/bf16, also for the cache) is allowed if it passes this gate.
Speed: end-to-end generate() tokens/s at decode_128 (128-token prompt, 128 new), decode_768 (768, 128), decode_roll \
(1024-token prompt, 64 new, past the window), plus prefill_1024 (one 1024-token prefill). Fitness is the geometric \
mean of the four, measured interleaved against the incumbent; accepted at >= {math.exp(c['min_gain']):.2f}x and \
clear of the noise (~1% per round pair), and no single scenario may fall below \
{1 - c['max_scenario_regression']:.2f}x the incumbent's.

## Incumbent {inc.id}: {inc.scores.get('x_p0', 1):.2f}x the seed
Absolute speeds: {', '.join(f'{k} {v:.1f}/s' for k, v in speeds.items())} (prefill: calls/s)

## Other strong programs
{chr(10).join('- ' + describe(q) + ': ' + ' '.join(q.rationale.split())[:300] for q in best) or '(none yet)'}

## Recent attempts (learn from the failures; don't repeat them)
{chr(10).join('- ' + describe(q) + ': ' + ' '.join(q.rationale.split())[:200] + ' -> ' + q.reason[:200] for q in recent) or '(none yet)'}

## Parent {parent.id} ({describe(parent)})
```python
{parent.code}
```

## This time
{INSTRUCTIONS[instruction]}
Return SEARCH/REPLACE edits against the parent (each SEARCH must match exactly once), or `code` for a full rewrite."""


def apply_edits(code: str, diffs: list[dict]) -> str:
    for n, d in enumerate(diffs, 1):
        hits = code.count(d["search"])
        if not d["search"].strip() or hits != 1:
            raise ValueError(f"edit {n}: SEARCH matches {hits} places" if d["search"].strip() else f"edit {n}: empty SEARCH")
        code = code.replace(d["search"], d["replace"], 1)
    return code


def spend_24h() -> float:
    try:
        rows = [json.loads(x) for x in (LABS_STATE / "llm_spend.jsonl").read_text().splitlines() if x.strip()]
    except OSError:
        return 0.0
    since = now() - timedelta(hours=24)
    return sum(r.get("usd") or 0 for r in rows if datetime.fromisoformat(r["at"]) >= since)


def pick_parent(progs: dict[str, Program], rng: random.Random) -> Program:
    lab = load_lab()
    pool = sorted((p for p in progs.values() if p.status in ("accepted", "contender") and p.scores.get("x_p0")),
                  key=lambda p: -p.scores["x_p0"])[:5]
    return progs[lab["incumbent"]] if rng.random() < 0.6 or not pool else rng.choice(pool)


def propose(log=print, call=None, rng: random.Random | None = None) -> Program | None:
    from autolab.llm import LLMError, RateLimited
    from autolab.llm import call as llm_call

    c, rng = cfg(), rng or random.Random()
    progs = programs()
    parent = pick_parent(progs, rng)
    instruction = rng.choices(list(INSTRUCTIONS), weights=[c.get("instruction_weights", {}).get(k, 1.0) for k in INSTRUCTIONS])[0]
    pid = f"p{max(int(k[1:]) for k in progs) + 1}"
    try:
        res = (call or llm_call)(build_prompt(parent, progs, instruction, c), SYSTEM, SCHEMA, c["llm"]["model"],
                                 c["llm"], LABS_STATE / "llm" / "infer", f"infer-{pid}")
    except RateLimited as exc:
        lab = load_lab()
        lab["paused_until"] = iso(now() + timedelta(hours=1))
        save_lab(lab)
        log(f"infer: rate limited, pausing an hour ({exc})")
        return None
    except LLMError as exc:
        note("propose_failed", error=str(exc)[:300])
        log(f"infer: proposal failed: {exc}")
        return None
    r = res["reply"]
    p = Program(pid, parent.id, parent.code, rationale=r["rationale"], expected=r.get("expected", ""),
                created_by=res["model"])
    p.scores["instruction"] = instruction
    try:
        if r.get("code"):
            p.code = r["code"]
        elif r.get("diffs"):
            p.code = apply_edits(parent.code, r["diffs"])
        else:
            raise ValueError("neither diffs nor code")
    except ValueError as exc:
        reject(p, "static", f"edits don't apply: {exc}")
    save(p)
    note("proposed", program=pid, parent=parent.id, instruction=instruction, cost_usd=res["cost_usd"],
         rationale=r["rationale"][:300])
    log(f"infer: proposed {pid} from {parent.id} ({instruction}, ${res['cost_usd']:.2f}): {r['rationale'][:120]}")
    return p


# --- the loop -----------------------------------------------------------------------------------------------


def blocked() -> str | None:
    """Why the lab may not use the GPU or spend right now, or None."""
    from autolab.gpu import active_owner_runs

    if (h := control().get("hold")):
        return f"paused by {h.get('by')} since {h.get('at')}"
    if (u := load_lab().get("paused_until")) and datetime.fromisoformat(u) > now():
        return f"rate limited until {u}"
    if OWNER_RUNS.exists() and active_owner_runs(OWNER_RUNS):
        return "the owner is training on this GPU"
    return None


def cycle(log=print) -> dict:
    if (why := blocked()):
        return {"waiting": why}
    if "p0" not in programs() or load_lab().get("incumbent") is None:
        init(log=log)
        return {"initialized": True}
    pending = [p for p in programs().values() if p.status not in DONE]
    if pending:
        p = advance(pending[0], log=log)
        return {"advanced": p.id, "stage": p.stage, "status": p.status}
    c = cfg()
    if spend_24h() + c.get("est_call_usd", 0.3) > c["daily_usd"]:
        return {"waiting": f"daily budget: ${spend_24h():.2f} of ${c['daily_usd']} in 24 h"}
    p = propose(log=log)
    return {"proposed": p.id if p else None}


def run(interval: float = 20.0) -> None:
    def log(msg: str) -> None:
        print(f"{iso()} {msg}", flush=True)

    log(f"labs.infer daemon started (pid {os.getpid()})")
    while True:
        t0 = time.time()
        try:
            res = cycle(log)
            err = None
        except Exception as exc:  # noqa: BLE001 - keep going; surface it in the heartbeat
            import traceback

            res, err = {}, {"at": iso(), "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
            log(f"cycle failed: {err['error']}")
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "daemon.json").write_text(json.dumps({"pid": os.getpid(), "updated": iso(), "last": res,
                                                       "last_error": err, "cycle_s": round(time.time() - t0, 1)}, indent=2))
        busy = bool(res.get("advanced") or res.get("proposed") or res.get("initialized"))
        time.sleep(1 if busy else interval)
