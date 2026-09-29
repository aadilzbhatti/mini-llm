"""Regime-scoped memory: what the proposer is told has been tried, and for how long.

A session is a regime (dataset, token budget, eval settings). A new one starts at each data switch or
compute-ladder step, with the previous session as its `origin`, so the sessions form a chain and the
**regime distance** of a program is the number of switches between its session and the current one.
A rejection is evidence about an idea only in the regime where it was measured, so how long it keeps
suppressing the idea depends on what failed and how far away that regime is, not on elapsed time
(owner, 2026-09-29):

| kind        | what happened                                          | distance 0            | 1                     | >= 2        |
| ----------- | ------------------------------------------------------ | --------------------- | --------------------- | ----------- |
| `bug`       | rejected before training (scope, static, shape/CPU      | shown as a failed     | hidden                | hidden      |
|             | tests, parameter cap) or the run crashed               | attempt: idea untested|                       |             |
| `screen`    | rejected at the screen (weak evidence; screens are     | don't repeat          | hidden                | hidden      |
|             | biased against regularization and longer context)      |                       |                       |             |
| `worse`     | full run more than near_miss_sigma x σ above the       | don't repeat          | weak, tagged with the | hidden      |
|             | incumbent of the time (or failed there: NaN, too slow) |                       | regime it was tested  |             |
| `near_miss` | contender, or full run within near_miss_sigma x σ      | inconclusive, may     | worth retesting (and  | same as 1   |
|             |                                                        | be refined            | a targeted retest)    |             |
| `accepted`  | became the incumbent                                   | shown                 | hidden (it is ported) | hidden      |

Only programs whose session is on the current session's origin chain are remembered; everything stays
in the program database and the lineage tree. The rendered list is capped at [evolve.database] history
entries (current regime first, newest first), the only count-based limit.

Research cards use the same distance: an outcome counts 0.5^distance in the card bandit
([evolve.memory] card_decay), so a technique that failed in a small regime gets another look later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autolab.program import Program

EARLY = ("static", "cpu", "params")


@dataclass
class Memory:
    session: str      # session name the program belongs to
    distance: int     # regime changes between that session and the current one
    program: Program
    kind: str         # bug | screen | worse | near_miss | accepted | pending
    z: float | None   # (full mean - incumbent mean then) / σ, for full-run outcomes
    regime: str       # short description of that session's regime, for tags

    @property
    def key(self) -> str:
        return f"{self.session}/{self.program.id}"


def memory_cfg(cfg: dict) -> dict:
    return {"history": cfg.get("database", {}).get("history", 60), "near_miss_sigma": 2.0, "retest_per_regime": 3,
            "card_decay": 0.5, **cfg.get("memory", {})}


# --- sessions and the regime chain -------------------------------------------------------------------


def _load(root: Path, name: str) -> tuple[dict, dict[str, Program]] | None:
    from autolab import evaluate as ev

    paths = ev.Paths(root / name)
    session = ev.load_session(paths)
    return (session, ev.programs(paths)) if session else None


def chain(session: dict, root: Path) -> list[tuple[dict, dict[str, Program] | None]]:
    """[(session, programs)] from `session` (distance 0, programs not loaded) back along its origins."""
    out: list[tuple[dict, dict[str, Program] | None]] = [(session, None)]
    seen = {session.get("name")}
    origin = session.get("origin") or {}
    while origin.get("session") and origin["session"] not in seen:
        loaded = _load(root, origin["session"])
        if loaded is None:
            break
        seen.add(origin["session"])
        out.append(loaded)
        origin = loaded[0].get("origin") or {}
    return out


def distances(session: dict, root: Path) -> dict[str, int]:
    """Session name -> regime distance, for the sessions on the origin chain."""
    return {s.get("name"): d for d, (s, _) in enumerate(chain(session, root))}


def regime_label(session: dict) -> str:
    tokens = (session.get("budgets") or {}).get("full_tokens") or 0
    return f"{session.get('name')}: {session.get('dataset_id') or 'data20k'}, {tokens / 1e6:.0f}M tokens"


# --- classifying outcomes --------------------------------------------------------------------------


def incumbent_mean_at(session: dict, progs: dict[str, Program], when: str) -> float | None:
    """The incumbent's full mean when a program was created (what its outcome was judged against)."""
    ref = progs["p0"].scores.get("full_mean") if "p0" in progs else None
    for h in session.get("incumbent_history") or []:
        if h.get("at", "") <= when and h.get("full_mean") is not None:
            ref = h["full_mean"]
    return ref


def is_bug(p: Program) -> bool:
    """The implementation failed, so the idea itself was never measured."""
    return p.status == "rejected" and (p.stage in EARLY or p.reason.startswith(("run failed", "smoke: run failed")))


def classify(p: Program, session: dict, progs: dict[str, Program], sigma: float,
             near_miss_sigma: float) -> tuple[str, float | None]:
    if p.status == "accepted":
        return "accepted", None
    if p.status not in ("rejected", "evaluated", "contender"):
        return "pending", None
    if is_bug(p):
        return "bug", None
    if p.status == "rejected" and p.stage == "screen":
        return "screen", None
    m = p.scores.get("full_mean")
    ref = incumbent_mean_at(session, progs, p.created_at)
    z = (m - ref) / sigma if m is not None and ref is not None and sigma > 0 else None
    if p.status == "contender" or (z is not None and z <= near_miss_sigma):
        return "near_miss", z
    return "worse", z


def remembered(session: dict, progs: dict[str, Program], cfg: dict, root: Path) -> list[Memory]:
    """Every program the proposer should hear about, per the table in the module docstring: the current
    regime first (newest first), then older regimes, capped at `history` entries."""
    from autolab.evaluate import sigma

    mcfg = memory_cfg(cfg)
    out: list[Memory] = []
    for d, (sess, sprogs) in enumerate(chain(session, root)):
        sprogs = progs if d == 0 else sprogs
        sig = sigma(sess, cfg)
        label = regime_label(sess)
        for p in sorted((q for q in sprogs.values() if q.parent_id is not None),
                        key=lambda q: (q.created_at, int(q.id[1:]) if q.id[1:].isdigit() else 0), reverse=True):
            kind, z = classify(p, sess, sprogs, sig, mcfg["near_miss_sigma"])
            if d == 0 or kind == "near_miss" or (d == 1 and kind == "worse"):
                out.append(Memory(sess.get("name"), d, p, kind, z, label))
    return out[: int(mcfg["history"])]


def _z(m: Memory) -> str:
    return "" if m.z is None else f", {m.z:+.1f}σ vs the incumbent then"


def tag(m: Memory) -> str:
    """What the entry means for the proposer."""
    if m.distance == 0:
        return {"bug": "attempt failed (bug): the idea is untested; a correct implementation may be retried",
                "screen": "rejected at the screen: don't repeat in this regime (weak evidence)",
                "worse": f"worse at full budget{_z(m)}: don't repeat in this regime",
                "near_miss": f"near miss{_z(m)}: inconclusive, may be refined",
                "accepted": "accepted",
                "pending": f"in evaluation (stage {m.program.stage})"}[m.kind]
    where = f"in {m.regime}, {m.distance} regime change{'s' if m.distance > 1 else ''} ago"
    if m.kind == "near_miss":
        return f"near miss {where}{_z(m)}: worth retesting here"
    return f"worse {where}{_z(m)}: weak evidence here"


def render(mem: list[Memory], outcome, width: int = 180) -> str:
    """One line per remembered program. `outcome` renders a program's result (autolab.prompt.outcome)."""
    lines = []
    for m in mem:
        p = m.program
        name = p.id if m.distance == 0 else m.key
        rationale = " ".join(p.rationale.split())[:width]
        lines.append(f"- {name} (parent {p.parent_id}, by {p.created_by}): {rationale} → {outcome(p)} "
                     f"[{tag(m)}]")
    return "\n".join(lines)


HEADER = ("# What has been tried (this regime first, latest first)\n\n"
          "A rejection is evidence only in the regime (data, token budget) where it was measured. Don't repeat "
          "ideas tagged \"don't repeat in this regime\". Entries from earlier regimes are weak evidence: an idea "
          "that was worse there may be worth another look if the regime change plausibly helps it. A failed "
          "attempt (bug) says nothing about the idea. Near misses are inconclusive.")


# --- targeted retests of near misses after a regime change ----------------------------------------------


def retest_candidates(session: dict, cfg: dict, root: Path, done: set[str], k: int | None = None) -> list[Memory]:
    """Near misses from earlier regimes to re-apply to the new champion: nearest regime first, then the
    closest to winning. Skips programs already retested or ported (an accepted program is ported anyway)."""
    mcfg = memory_cfg(cfg)
    k = int(mcfg["retest_per_regime"]) if k is None else k
    cands = [m for m in remembered(session, {}, {**cfg, "memory": {**mcfg, "history": 10**6}}, root)
             if m.distance >= 1 and m.kind == "near_miss" and m.key not in done]
    cands.sort(key=lambda m: (m.distance, m.program.status != "contender", m.z if m.z is not None else 0.0))
    return cands[:k]


def load_source(key: str, root: Path) -> tuple[dict, Program, Program | None] | None:
    """(session, program, its parent) for 'session/pid'."""
    name, _, pid = key.rpartition("/")
    loaded = _load(root, name)
    if loaded is None or pid not in loaded[1]:
        return None
    session, progs = loaded
    p = progs[pid]
    return session, p, progs.get(p.parent_id or "")


# --- card bandit decay ------------------------------------------------------------------------------------


def session_weights(active: str | None, root: Path, decay: float = 0.5) -> dict[str, float]:
    """Session name -> weight of its outcomes in the card bandit: decay^distance on the active chain,
    and decay^(chain length) for sessions off the chain (older lines of work)."""
    from autolab import evaluate as ev

    if not active:
        return {}
    session = ev.load_session(ev.Paths(root / active))
    if not session:
        return {}
    dist = distances(session, root)
    far = decay ** len(dist)
    names = [d.name for d in root.iterdir() if (d / "session.json").exists()] if root.exists() else []
    return {n: (decay ** dist[n] if n in dist else far) for n in names}

