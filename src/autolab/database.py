"""Program database: which programs to show the LLM (AlphaEvolve §2.5, scaled down).

The paper combines MAP-Elites with an island model. Here:
- Elites are programs that finished the full-budget stage (status evaluated / contender /
  accepted), plus the initial program. Fitness = −(mean full val loss).
- Each elite sits in a MAP-Elites cell by two descriptors, parameter count and training
  throughput (bucketed by [evolve.database] bins). A cell keeps only its fittest program.
- Programs belong to an island (a child inherits the island it was generated for; p0 is in
  every island). Islands evolve separately. Every `migrate_every` finished children, each
  island's best is copied into the others (recorded in session["database"]["migrants"]).
- `sample(island)`: the parent is the island's best with probability p_exploit, else the
  elite of a uniformly random occupied cell. Inspirations are the top_inspirations fittest
  elites overall plus diverse_inspirations random elites from other cells (parent excluded).

Everything is a pure function of the program list, the session and an RNG, so it's testable
and restart-safe.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass

from autolab.program import Program

ELITE_STATUSES = {"evaluated", "contender", "accepted"}


@dataclass(frozen=True)
class DBConfig:
    n_islands: int = 2
    p_exploit: float = 0.5
    top_inspirations: int = 2
    diverse_inspirations: int = 1
    migrate_every: int = 6
    params_bins: tuple = (12e6, 16e6, 20e6)
    tokens_per_sec_bins: tuple = (30e3, 38e3, 45e3)
    decode_ms_bins: tuple = (5.0, 10.0, 20.0)   # M8 descriptor: batch-1 decode latency
    p_frontier: float = 0.3                     # M8: chance the parent is a random Pareto-frontier program
    quality_tol: float = 0.02
    recent_failures: int = 5
    history: int = 30

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DBConfig":
        d = dict(cfg.get("database", {}))
        for k in ("params_bins", "tokens_per_sec_bins"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def fitness(p: Program) -> float | None:
    if p.status not in ELITE_STATUSES or p.scores.get("full_mean") is None:
        return None
    return -p.scores["full_mean"]


def cell(p: Program, cfg: DBConfig) -> tuple:
    """MAP-Elites descriptors. With eval-suite metrics (M8): context length x decode-cost bucket, so a
    long-context or cheap-to-serve program keeps its cell even when it isn't the lowest-loss one.
    Otherwise (older programs): parameter count x training throughput."""
    m = p.scores.get("metrics") or {}
    if "context" in m and "decode_ms_per_token" in m:
        return ("ctx", int(m["context"]), bisect.bisect(cfg.decode_ms_bins, m["decode_ms_per_token"]))
    params = p.scores.get("params") or 0
    tps = p.scores.get("tokens_per_sec") or 0
    return bisect.bisect(cfg.params_bins, params), bisect.bisect(cfg.tokens_per_sec_bins, tps)


def island_members(progs: dict[str, Program], island: int, session: dict) -> list[Program]:
    migrants = set(session.get("database", {}).get("migrants", {}).get(str(island), []))
    return [p for p in progs.values() if p.island is None or p.island == island or p.id in migrants]


def elites(members: list[Program], cfg: DBConfig) -> dict[tuple[int, int], Program]:
    grid: dict[tuple[int, int], Program] = {}
    for p in members:
        f = fitness(p)
        if f is None:
            continue
        c = cell(p, cfg)
        if c not in grid or f > fitness(grid[c]):
            grid[c] = p
    return grid


def next_island(progs: dict[str, Program], cfg: DBConfig) -> int:
    """Round-robin over islands by number of generated children."""
    return sum(p.parent_id is not None for p in progs.values()) % cfg.n_islands


def sample(progs: dict[str, Program], session: dict, island: int, cfg: DBConfig,
           rng: random.Random) -> tuple[Program, list[Program]]:
    grid = elites(island_members(progs, island, session), cfg)
    from autolab.pareto import frontier

    front = [p for p in frontier(list(progs.values()), cfg.quality_tol)]
    if not grid:  # nothing evaluated yet: start from the incumbent
        parent = progs[session["incumbent"]]
    elif front and rng.random() < cfg.p_frontier:  # M8: build on any frontier point, not only the loss leader
        parent = rng.choice(sorted(front, key=lambda p: p.id))
    elif rng.random() < cfg.p_exploit:
        parent = max(grid.values(), key=fitness)
    else:
        parent = grid[rng.choice(sorted(grid))]
    all_elites = [p for p in elites(list(progs.values()), cfg).values() if p.id != parent.id]
    # the global grid keeps one per cell; the top list should see every finished program
    finished = sorted((p for p in progs.values() if fitness(p) is not None and p.id != parent.id),
                      key=fitness, reverse=True)
    top = finished[: cfg.top_inspirations]
    rest = [p for p in all_elites if p not in top and cell(p, cfg) != cell(parent, cfg)]
    diverse = rng.sample(rest, min(cfg.diverse_inspirations, len(rest)))
    return parent, top + diverse


def maybe_migrate(progs: dict[str, Program], session: dict, cfg: DBConfig) -> bool:
    """Copy each island's best into the other islands every `migrate_every` finished children.

    Mutates `session["database"]`; returns True if a migration happened (caller saves the session).
    """
    db = session.setdefault("database", {"migrated_at": 0, "migrants": {}})
    done = sum(p.parent_id is not None and p.status in ELITE_STATUSES | {"rejected"} for p in progs.values())
    if done - db["migrated_at"] < cfg.migrate_every:
        return False
    bests = {}
    for i in range(cfg.n_islands):
        grid = elites([p for p in progs.values() if p.island == i], cfg)
        if grid:
            bests[i] = max(grid.values(), key=fitness).id
    for i in range(cfg.n_islands):
        mig = set(db["migrants"].get(str(i), []))
        mig |= {pid for j, pid in bests.items() if j != i}
        db["migrants"][str(i)] = sorted(mig)
    db["migrated_at"] = done
    return True
