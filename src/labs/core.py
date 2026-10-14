"""The loop every lab shares, extracted from labs.infer once labs.decode gave it a second shape.

A lab evolves one whole Python file (its program) that defines one class (`class_name`) against a fixed, protected
harness. What the base class owns, identical for every lab:

  programs   state/labs/<lab>/programs/p<N>.json: code, rationale, stage, status, stage log, scores
  cascade    advance() runs one stage per call; "static" (parses, defines the class, no forbidden names, size cap) is
             shared, the rest is the lab's run_stage(). Statuses: queued -> rejected | evaluated | contender | accepted
  incumbent  lab.json: the current best and its history; accept() moves it
  proposals  Claude (autolab.llm.call) with the lab's task text, the incumbent, strong others, recent attempts and
             the parent's code; SEARCH/REPLACE edits or a whole file. Parent: the incumbent 60%, else a top-5 program
  budget     one rolling-24h Claude budget for all labs ([labs] daily_usd; state/labs/llm_spend.jsonl)
  pause      state/labs/control.json "hold" (dashboard Labs tab, `labs pause`), a Claude rate limit, or the owner's
             training run on this GPU (autolab.gpu) stops the loop
  harness    every measurement is `python -m <lab harness> <mode> ...` in a subprocess with a timeout; its JSON is
             the result, and a crash is the candidate's result, not the daemon's
What a lab supplies: seed file, task text for the prompt, instructions, init() (reference numbers + p0),
run_stage() (its cascade after static), fitness() for parent selection, describe() for the prompt and dashboard.
labs.daemon runs every lab's cycle() in one process, so GPU stages of different labs never overlap.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autolab.config import REPO_ROOT

LABS_STATE = REPO_ROOT / "autolab" / "state" / "labs"
LABS_DATA = REPO_ROOT / "autolab" / "data" / "labs"
CONFIG = REPO_ROOT / "autolab" / "labs.toml"
OWNER_RUNS = Path.home() / "dev" / "wiki-llm" / "runs"
DONE = ("rejected", "evaluated", "contender", "accepted")


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime | None = None) -> str:
    return (t or now()).isoformat(timespec="seconds")


@dataclass
class Program:
    id: str
    parent_id: str | None
    code: str
    rationale: str = ""
    expected: str = ""
    created_by: str = "human"
    created_at: str = field(default_factory=iso)
    stage: str = "static"
    status: str = "queued"  # queued | rejected | evaluated | contender | accepted
    reason: str = ""
    stages: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


# --- labs-wide state: pause, budget -----------------------------------------------------------------------------


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


def spend_24h() -> float:
    try:
        rows = [json.loads(x) for x in (LABS_STATE / "llm_spend.jsonl").read_text().splitlines() if x.strip()]
    except OSError:
        return 0.0
    since = now() - timedelta(hours=24)
    return sum(r.get("usd") or 0 for r in rows if datetime.fromisoformat(r["at"]) >= since)


def labs_cfg() -> dict:
    return tomllib.loads(CONFIG.read_text())


# --- shared pieces of every cascade --------------------------------------------------------------------------


def pair_stats(rounds: list[float]) -> tuple[float, float, int]:
    """Mean and standard error over (even, odd) pairs of paired rounds, which cancels an order effect when the
    order alternates; also right for unordered paired rounds."""
    pairs = [(rounds[i] + rounds[i + 1]) / 2 for i in range(0, len(rounds) - 1, 2)]
    m = statistics.fmean(pairs)
    se = statistics.stdev(pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else float("inf")
    return m, se, len(pairs)


def static_check(code: str, class_name: str, max_bytes: int) -> str | None:
    from autolab.program import static_violations

    if len(code) > max_bytes:
        return f"the program is {len(code)} bytes (cap {max_bytes})"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    if not any(isinstance(n, ast.ClassDef) and n.name == class_name for n in tree.body):
        return f"no top-level class {class_name}"
    bad = static_violations({"program": code})
    return "; ".join(bad)[:400] if bad else None


def apply_edits(code: str, diffs: list[dict]) -> str:
    for n, d in enumerate(diffs, 1):
        hits = code.count(d["search"])
        if not d["search"].strip() or hits != 1:
            raise ValueError(f"edit {n}: SEARCH matches {hits} places" if d["search"].strip() else f"edit {n}: empty SEARCH")
        code = code.replace(d["search"], d["replace"], 1)
    return code


SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "what changes and the mechanism by which it should help (2-5 sentences)"},
        "expected": {"type": "string", "description": "what should improve, roughly how much"},
        "diffs": {"type": "array", "items": {"type": "object", "properties": {
            "search": {"type": "string"}, "replace": {"type": "string"}}, "required": ["search", "replace"]}},
        "code": {"type": "string", "description": "a whole new file instead of diffs (for rewrites only)"},
    },
    "required": ["rationale", "expected"],
}


class Lab:
    name = ""
    title = ""
    seed_file: Path
    class_name = ""
    file_name = ""  # what the prompt calls the program, e.g. engine.py
    harness_module = ""
    system = ""
    instructions: dict[str, str] = {}

    # --- storage ---------------------------------------------------------------------------------------------

    @property
    def state(self) -> Path:
        return LABS_STATE / self.name

    @property
    def data(self) -> Path:
        return LABS_DATA / self.name

    def cfg(self) -> dict:
        c = labs_cfg()
        return {**c.get("labs", {}), **c.get(self.name, {}), "llm": c.get("llm", {})}

    def save(self, p: Program) -> None:
        d = self.state / "programs"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{p.id}.json.tmp"
        tmp.write_text(json.dumps(asdict(p), indent=2))
        tmp.replace(d / f"{p.id}.json")

    def programs(self) -> dict[str, Program]:
        d = self.state / "programs"
        out = {f.stem: Program(**json.loads(f.read_text())) for f in d.glob("p*.json")} if d.exists() else {}
        return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))

    def load_lab(self) -> dict:
        try:
            return json.loads((self.state / "lab.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def save_lab(self, lab: dict) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "lab.json").write_text(json.dumps(lab, indent=2))

    def note(self, event: str, **fields) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        with open(self.state / "notebook.jsonl", "a") as fh:
            fh.write(json.dumps({"at": iso(), "event": event, **fields}) + "\n")

    def work(self) -> Path:
        d = self.state / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- measurement ------------------------------------------------------------------------------------------

    def harness(self, mode: str, out: Path, timeout: float, **args) -> dict:
        cmd = [sys.executable, "-m", self.harness_module, mode, "--out", str(out)]
        for k, v in args.items():
            cmd += [f"--{k}", str(v)]
        out.unlink(missing_ok=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
                               env={**os.environ, "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout:.0f}s"}
        if not out.exists():
            return {"ok": False, "error": f"harness exited {r.returncode}: {(r.stderr or r.stdout)[-400:]}"}
        return json.loads(out.read_text())

    # --- the cascade --------------------------------------------------------------------------------------------

    def record(self, p: Program, stage: str, ok: bool, detail: str, **extra) -> None:
        p.stages.append({"stage": stage, "ok": ok, "at": iso(), "detail": detail, **extra})

    def reject(self, p: Program, stage: str, why: str) -> None:
        self.record(p, stage, False, why)
        p.stage, p.status, p.reason = "done", "rejected", f"{stage}: {why}"

    def finish(self, p: Program, status: str, reason: str) -> None:
        p.stage, p.status, p.reason = "done", status, reason

    def accept(self, p: Program, detail: str, log=print) -> None:
        lab = self.load_lab()
        self.finish(p, "accepted", detail)
        lab["incumbent"] = p.id
        lab.setdefault("history", []).append({"at": iso(), "program": p.id, "fitness": self.fitness(p)})
        self.save_lab(lab)
        self.note("accepted", program=p.id, detail=detail)
        log(f"{self.name}: {p.id} ACCEPTED ({detail})")

    def advance(self, p: Program, log=print) -> Program:
        """Run the program's next stage (static here; the rest is the lab's)."""
        if p.stage == "static":
            why = static_check(p.code, self.class_name, self.cfg().get("max_code_bytes", 40_000))
            if why:
                self.reject(p, "static", why)
            else:
                self.record(p, "static", True, "ok")
                p.stage = self.first_stage
        else:
            (self.work() / f"{p.id}.py").write_text(p.code)
            self.run_stage(p, log)
        self.save(p)
        if p.status in DONE:
            self.note("program_done", program=p.id, status=p.status, reason=p.reason[:300])
            log(f"{self.name}: {p.id} {p.status}: {p.reason[:200]}")
        return p

    # --- what a lab implements --------------------------------------------------------------------------------

    first_stage = ""

    def init(self, log=print) -> None:
        raise NotImplementedError

    def run_stage(self, p: Program, log=print) -> None:
        raise NotImplementedError

    def fitness(self, p: Program) -> float | None:
        """Higher is better; None if unmeasured."""
        raise NotImplementedError

    def describe(self, p: Program) -> str:
        raise NotImplementedError

    def task(self, c: dict) -> str:
        """The prompt's description of the problem, the contract and how candidates are judged."""
        raise NotImplementedError

    def incumbent_summary(self, p: Program) -> str:
        return self.describe(p)

    # --- proposals ------------------------------------------------------------------------------------------------

    def build_prompt(self, parent: Program, progs: dict[str, Program], instruction: str, c: dict) -> str:
        inc = progs[self.load_lab()["incumbent"]]
        best = sorted((q for q in progs.values() if self.fitness(q) is not None and q.id != parent.id
                       and q.status in ("accepted", "contender")), key=lambda q: -self.fitness(q))[:4]
        recent = [q for q in progs.values() if q.status in DONE and q.parent_id][-12:]
        line = lambda q, n: f"- {self.describe(q)}: {' '.join(q.rationale.split())[:n]}"  # noqa: E731
        return f"""{self.task(c)}

## Incumbent {inc.id}
{self.incumbent_summary(inc)}

## Other strong programs
{chr(10).join(line(q, 300) for q in best) or '(none yet)'}

## Recent attempts (learn from the failures; don't repeat them)
{chr(10).join(line(q, 200) + ' -> ' + q.reason[:200] for q in recent) or '(none yet)'}

## Parent {parent.id} ({self.describe(parent)})
```python
{parent.code}
```

## This time
{self.instructions[instruction]}
Return SEARCH/REPLACE edits against the parent (each SEARCH must match exactly once), or `code` for a full rewrite."""

    def pick_parent(self, progs: dict[str, Program], rng: random.Random) -> Program:
        pool = sorted((p for p in progs.values() if p.status in ("accepted", "contender") and self.fitness(p) is not None),
                      key=lambda p: -self.fitness(p))[:5]
        return progs[self.load_lab()["incumbent"]] if rng.random() < 0.6 or not pool else rng.choice(pool)

    def propose(self, log=print, call=None, rng: random.Random | None = None) -> Program | None:
        from autolab import llm

        c, rng = self.cfg(), rng or random.Random()
        progs = self.programs()
        parent = self.pick_parent(progs, rng)
        names = list(self.instructions)
        instruction = rng.choices(names, weights=[c.get("instruction_weights", {}).get(k, 1.0) for k in names])[0]
        pid = f"p{max(int(k[1:]) for k in progs) + 1}"
        try:
            res = (call or llm.call)(self.build_prompt(parent, progs, instruction, c), self.system, SCHEMA,
                                     c["llm"]["model"], c["llm"], LABS_STATE / "llm" / self.name, f"{self.name}-{pid}")
        except llm.RateLimited as exc:
            lab = self.load_lab()
            lab["paused_until"] = iso(now() + timedelta(hours=1))
            self.save_lab(lab)
            log(f"{self.name}: rate limited, pausing an hour ({exc})")
            return None
        except llm.LLMError as exc:
            self.note("propose_failed", error=str(exc)[:300])
            log(f"{self.name}: proposal failed: {exc}")
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
            self.reject(p, "static", f"edits don't apply: {exc}")
        self.save(p)
        self.note("proposed", program=pid, parent=parent.id, instruction=instruction, cost_usd=res["cost_usd"],
                  rationale=r["rationale"][:300])
        log(f"{self.name}: proposed {pid} from {parent.id} ({instruction}, ${res['cost_usd']:.2f}): {r['rationale'][:120]}")
        return p

    # --- the loop ---------------------------------------------------------------------------------------------------

    def blocked(self) -> str | None:
        from autolab.gpu import active_owner_runs

        if (h := control().get("hold")):
            return f"paused by {h.get('by')} since {h.get('at')}"
        if (u := self.load_lab().get("paused_until")) and datetime.fromisoformat(u) > now():
            return f"rate limited until {u}"
        if OWNER_RUNS.exists() and active_owner_runs(OWNER_RUNS):
            return "the owner is training on this GPU"
        return None

    def cycle(self, log=print) -> dict:
        if (why := self.blocked()):
            return {"waiting": why}
        if self.load_lab().get("incumbent") is None:
            self.init(log=log)
            return {"initialized": True}
        pending = [p for p in self.programs().values() if p.status not in DONE]
        if pending:
            p = self.advance(pending[0], log=log)
            return {"advanced": p.id, "stage": p.stage, "status": p.status}
        c = self.cfg()
        if spend_24h() + c.get("est_call_usd", 0.3) > c["daily_usd"]:
            return {"waiting": f"daily budget: ${spend_24h():.2f} of ${c['daily_usd']} in 24 h"}
        p = self.propose(log=log)
        return {"proposed": p.id if p else None}

    # --- dashboard ----------------------------------------------------------------------------------------------------

    def view(self) -> dict:
        """What the dashboard's Labs tab shows for this lab."""
        progs = self.programs()
        st = self.load_lab()
        nb = []
        try:
            nb = [json.loads(x) for x in (self.state / "notebook.jsonl").read_text().splitlines()[-60:] if x.strip()][::-1]
        except OSError:
            pass
        rows = [{"id": p.id, "parent": p.parent_id, "status": p.status, "stage": p.stage, "by": p.created_by,
                 "created": p.created_at, "rationale": p.rationale, "reason": p.reason, "fitness": self.fitness(p),
                 "summary": self.row_summary(p), "instruction": p.scores.get("instruction")}
                for p in reversed(progs.values())]
        return {"name": self.name, "title": self.title, "incumbent": st.get("incumbent"), "history": st.get("history", []),
                "blocked": self.blocked() if progs else None, "rows": rows, "notebook": nb, "tiles": self.tiles(progs, st),
                "about": self.about()}

    def row_summary(self, p: Program) -> str:
        return ""

    def tiles(self, progs: dict[str, Program], st: dict) -> list[dict]:
        return []

    def about(self) -> str:
        return ""
