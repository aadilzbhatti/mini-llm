"""labs.pack: circle packing (26 circles in the unit square, maximize the sum of radii), the calibration run.

The benchmark AlphaEvolve, OpenEvolve and ShinkaEvolve all report, so labs.core's loop can be compared with theirs
on score per evaluated program. Published references (sum of radii, n = 26):
  2.634   best known before AlphaEvolve          2.63586  AlphaEvolve (2025)
  2.635983 ShinkaEvolve (2025)                   2.63598  optimize_anything, 63 evaluations; OpenEvolve 2.6307 with
                                                          over 3x that budget, same proposer (arXiv 2605.19633)
Deterministic and CPU-only: one stage after static, `run`: the program runs once under a deadline in a subprocess
(labs.pack.harness), the harness checks the packing itself, and the score is the verified sum of radii. Accepted
when it beats the incumbent by > min_gain; a valid packing within contender_gap of the incumbent stays a parent.
A finite run: proposals stop after max_programs programs.
"""

from __future__ import annotations

from pathlib import Path

from labs.core import Lab, Program, iso

REFERENCES = {"best known before AlphaEvolve": 2.634, "AlphaEvolve": 2.63586, "ShinkaEvolve": 2.635983,
              "optimize_anything (63 evals)": 2.63598, "OpenEvolve (matched, ~200 evals)": 2.6307}


class PackLab(Lab):
    name = "pack"
    title = "Circle packing (calibration)"
    seed_file = Path(__file__).with_name("seed_packer.py")
    class_name = "Packer"
    file_name = "pack.py"
    harness_module = "labs.pack.harness"
    first_stage = "run"
    system = """You write and improve a program that packs circles into a unit square. You get the current \
pack.py and its score; you return one focused change as SEARCH/REPLACE edits (or a whole new file) plus a short \
rationale. Be concrete about why the change should raise the score, and keep runs well inside the time limit."""
    instructions = {
        "construct": "Improve the initial arrangement: a better geometric pattern or a smarter way to seed the search.",
        "optimize": "Improve the numerical optimization of centers and radii (constraints, solver, restarts, polishing).",
        "search": "Explore more of the space within the time limit: restarts, perturbations, annealing, or combining packings.",
        "open": "Pick the most promising unexplored idea given what has been tried.",
        "refine": "Take the parent's own idea further, or fix what kept its last attempt from paying off.",
    }

    def run_program(self, p: Program) -> dict:
        c = self.cfg()
        return self.harness("run", self.work() / f"{p.id}-run.json", c["deadline_s"] + 60, program=self.work() / f"{p.id}.py",
                            n=c["n"], seed=0, deadline=c["deadline_s"])

    def init(self, log=print) -> None:
        p = self.programs().get("p0") or Program("p0", None, self.seed_file.read_text(),
                                                 rationale="Seed: a plain grid, each radius as large as its neighbours allow.")
        while p.status == "queued" and p.stage in ("static", "run"):
            p = self.advance(p, log=log)
        if "score" not in p.scores:
            raise RuntimeError(f"p0 failed: {p.reason}")
        p.status, p.stage, p.reason = "accepted", "done", "seed"
        self.save(p)
        self.save_lab({"incumbent": "p0", "history": [{"at": iso(), "program": "p0", "fitness": p.scores["score"]}]})
        self.note("initialized", incumbent="p0", score=p.scores["score"])
        log(f"pack: p0 scores {p.scores['score']:.6f}")

    def run_stage(self, p: Program, log=print) -> None:
        c = self.cfg()
        r = self.run_program(p)
        if not r.get("ok"):
            return self.reject(p, "run", r.get("error") or "failed")
        p.scores.update(score=r["score"], wall_s=r["wall_s"], circles=r["circles"],
                        evaluation=len(self.programs()) - 1)
        if p.id == "p0":
            self.record(p, "run", True, f"score {r['score']:.6f} in {r['wall_s']:.0f} s")
            p.stage = "done"
            return
        inc = self.programs()[self.load_lab()["incumbent"]]
        best = inc.scores["score"]
        detail = f"score {r['score']:.6f} vs {inc.id} {best:.6f} ({r['score'] - best:+.6f}) in {r['wall_s']:.0f} s"
        ok = r["score"] > best + c["min_gain"]
        self.record(p, "run", ok, detail)
        if ok:
            self.accept(p, detail, log)
        else:
            self.finish(p, "contender" if r["score"] >= best - c["contender_gap"] else "evaluated", detail)

    def exhausted(self) -> str | None:
        n, cap = len(self.programs()) - 1, self.cfg()["max_programs"]
        return f"calibration run done: {n} of {cap} programs" if n >= cap else None

    def fitness(self, p: Program) -> float | None:
        return p.scores.get("score")

    def describe(self, p: Program) -> str:
        s = p.scores.get("score")
        return f"{p.id} ({p.status}{f', score {s:.6f}' if s is not None else ''})"

    def task(self, c: dict) -> str:
        return f"""## Task
Place {c['n']} non-overlapping circles inside the unit square [0, 1] x [0, 1] to maximize the sum of their radii.

## The contract (the harness is fixed; you only change pack.py)
`Packer().run(n, rng, deadline_s)` -> a list of n (x, y, r) floats; called once in a fresh process with n = {c['n']}, \
rng a seeded random.Random, deadline_s = {c['deadline_s']} seconds (the process is killed {60} s after that; return \
the best valid packing found before the deadline). The harness checks the packing itself: each circle inside the \
square and no two overlapping, within a tolerance of 1e-9; any violation scores nothing. Score = sum of radii.
numpy, scipy, math and random are available; no file, OS, network or subprocess access (static check)."""

    def incumbent_summary(self, p: Program) -> str:
        return f"score {p.scores.get('score', 0):.6f} (sum of radii), {p.scores.get('wall_s', 0):.0f} s"

    def row_summary(self, p: Program) -> str:
        s = p.scores.get("score")
        return f"score {s:.6f} · {p.scores.get('wall_s', 0):.0f} s" if s is not None else ""

    def tiles(self, progs: dict, st: dict) -> list[dict]:
        inc = progs.get(st.get("incumbent") or "")
        if not inc:
            return []
        n = len(progs) - 1
        return [{"label": "Best sum of radii", "value": f"{inc.scores['score']:.6f}", "sub": f"{inc.id}, evaluation {inc.scores.get('evaluation', 0)}"},
                {"label": "Programs evaluated", "value": str(n), "sub": f"of {self.cfg()['max_programs']}"},
                {"label": "AlphaEvolve", "value": f"{REFERENCES['AlphaEvolve']:.5f}", "sub": "ShinkaEvolve 2.635983"},
                {"label": "OpenEvolve (matched)", "value": f"{REFERENCES['OpenEvolve (matched, ~200 evals)']:.4f}",
                 "sub": "optimize_anything 2.63598 @ 63"}]

    def about(self) -> str:
        return ("Calibration: the circle-packing benchmark (26 circles, maximize the sum of radii) that AlphaEvolve, "
                "OpenEvolve and ShinkaEvolve report, to compare this loop with theirs on score per evaluated program. "
                "Deterministic and CPU-only; the harness verifies each packing itself.")


LAB = PackLab()
