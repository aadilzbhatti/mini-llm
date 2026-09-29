"""The "no evolution" ablation (AlphaEvolve §3.4), at equal candidate count: does the program database help?

    uv run autolab ablation start [--source SESSION] [--candidates N]
    uv run autolab ablation status

`start` opens a control session `<source>~noevo`: the same p0, data, token budget, noise baseline and cascade as
the source session, but every child is proposed from p0 with no feedback from evaluations: no prior programs,
no "what has been tried", no frontier, no rejections, no technique cards, and the fixed instruction weights.
It is the paper's "repeatedly feed the initial program to the LLM" baseline. The daemon's controller proposes
for it (up to [ablation] max_in_flight at once, within the same daily budget) until it has as many candidates
as the source session had when it started (or --candidates). The control session never becomes the active
one, nothing is ported out of it, its acceptances are not committed, and it is left out of the regime chain
and the bandits.

`compare` puts the arms side by side on the first-seed full-budget loss (the only number both arms have for
every candidate that reached the full stage): the best after each number of candidates, and how many were
accepted. Ports (re-evaluations of other sessions' programs) don't count as candidates.
"""

from __future__ import annotations

from autolab import evaluate as ev
from autolab.program import Program

ARM = "no_evolution"
SUFFIX = "~noevo"


def is_ablation(session: dict | None) -> bool:
    return bool(session and session.get("ablation"))


def candidates(progs: dict[str, Program]) -> list[Program]:
    """Proposed children in creation order (ports excluded)."""
    kids = [p for p in progs.values() if p.parent_id is not None and not p.created_by.startswith("port:")]
    return sorted(kids, key=lambda p: int(p.id[1:]))


def start(source: str | None = None, n: int | None = None, root=None, runs_dir=None) -> dict:
    root = root or ev.STATE_ROOT
    source = source or ev.active_session_name(root)
    src_paths = ev.Paths(root / source)
    src = ev.load_session(src_paths)
    if src is None:
        raise ValueError(f"no session {source}")
    if is_ablation(src):
        raise ValueError(f"{source} is itself an ablation arm")
    p0 = ev.programs(src_paths)["p0"]
    target = n if n is not None else len(candidates(ev.programs(src_paths)))
    if target < 1:
        raise ValueError(f"{source} has no candidates to match yet; pass --candidates")
    name = source + SUFFIX
    paths = ev.Paths(root / name)
    kw = {"runs_dir": runs_dir} if runs_dir else {}
    ev.init_session(src["base_commit"], p0.hparams, {k: list(p0.runs[k]) for k in ("screen", "full")}, paths=paths,
                    budgets=src["budgets"], name=name, blocks=p0.blocks, dataset_id=src.get("dataset_id"),
                    rationale=f"{source}/p0, no-evolution control arm", **kw)
    session = ev.load_session(paths)
    session["ablation"] = {"arm": ARM, "of": source, "target": target, "started": ev.now_iso()}
    ev.save_session(session, paths)
    return session


def active_arms(root=None) -> list[ev.Paths]:
    """Ablation sessions still short of their target."""
    root = root or ev.STATE_ROOT
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        s = ev.load_session(ev.Paths(d)) if (d / "session.json").exists() else None
        if is_ablation(s) and len(candidates(ev.programs(ev.Paths(d)))) < s["ablation"]["target"]:
            out.append(ev.Paths(d))
    return out


def _curve(progs: dict[str, Program]) -> list[dict]:
    best, out = None, []
    for i, p in enumerate(candidates(progs), 1):
        first = (p.scores.get("full_losses") or [None])[0]
        if first is not None and p.status in ("evaluated", "contender", "accepted"):
            best = first if best is None else min(best, first)
        out.append({"n": i, "program": p.id, "status": p.status, "first_seed_full": first, "best": best})
    return out


def compare(name: str, root=None) -> dict:
    """Side by side at equal candidate counts. `name`: the ablation session (or its source)."""
    root = root or ev.STATE_ROOT
    name = name if name.endswith(SUFFIX) else name + SUFFIX
    abl = ev.load_session(ev.Paths(root / name))
    if not is_ablation(abl):
        raise ValueError(f"{name} is not an ablation session")
    source = abl["ablation"]["of"]
    arms = {"evolution": ev.programs(ev.Paths(root / source)), ARM: ev.programs(ev.Paths(root / name))}
    curves = {k: _curve(v) for k, v in arms.items()}
    done = {k: [c for c in curves[k] if c["status"] in ev.DONE] for k in arms}
    k = min(len(done["evolution"]), len(done[ARM]))
    at_k = {a: {"best_first_seed_full": (curves[a][k - 1]["best"] if k else None),
                "accepted": sum(c["status"] == "accepted" for c in curves[a][:k]),
                "reached_full": sum(c["first_seed_full"] is not None for c in curves[a][:k]),
                "rejected_before_training": sum(p.status == "rejected" and p.stage in ("static", "cpu", "params")
                                                for p in candidates(arms[a])[:k])} for a in arms}
    return {"source": source, "ablation": name, "target": abl["ablation"]["target"], "compared_at": k,
            "p0_full_mean": arms[ARM]["p0"].scores.get("full_mean"), "sigma": ev.sigma(abl), "arms": at_k,
            "curves": curves, "finished": {a: len(done[a]) for a in arms}}


def render(cmp: dict) -> str:
    def f(x):
        return "n/a" if x is None else f"{x:.4f}"

    lines = [f"No-evolution ablation: {cmp['ablation']} vs {cmp['source']} (target {cmp['target']} candidates; "
             f"finished: evolution {cmp['finished']['evolution']}, no evolution {cmp['finished'][ARM]})",
             f"p0 full mean {f(cmp['p0_full_mean'])}, σ {cmp['sigma']:.4f}. Compared at the first "
             f"{cmp['compared_at']} candidates of each arm (first-seed full-budget loss):"]
    for arm, a in cmp["arms"].items():
        lines.append(f"- {arm}: best {f(a['best_first_seed_full'])}, accepted {a['accepted']}, reached the full "
                     f"stage {a['reached_full']}, rejected before training {a['rejected_before_training']}")
    ev_best, ab_best = (cmp["arms"][a]["best_first_seed_full"] for a in ("evolution", ARM))
    if ev_best is not None and ab_best is not None:
        lines.append(f"Difference (no evolution − evolution): {ab_best - ev_best:+.4f} "
                     f"({(ab_best - ev_best) / cmp['sigma']:+.1f}σ; positive = the database helped)")
    return "\n".join(lines)
