"""Research (M7): a web-enabled agent that audits winning programs and writes technique cards.

Where this sits in the AlphaEvolve design: the paper feeds "relevant literature" to the LLM as explicit
prompt context (human-supplied) and co-evolves prompt context in its own database. Here:
- the research agent (claude -p with WebSearch/WebFetch only) produces the literature, as structured
  **technique cards** with sources, focused on upgrading the *current winner* ("what is it missing or doing
  in an outdated way?");
- cards are explicit context in the proposer's prompt, sampled by a bandit over how the programs that used
  them fared (the analogue of the paper's meta-prompt evolution);
- every new card also gets one directed child ("apply card c<n> to the incumbent") so it is tested promptly.

Cards live in autolab/research/cards.jsonl (committed, for the owner to review; no approval step). Research
spend is capped at [research] budget_share of the daily budget.

    uv run autolab research run [--trigger TEXT]    # one audit now (the controller runs it on its own)
    uv run autolab research cards                    # list cards with their stats
"""

from __future__ import annotations

import json
import math
import random
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from autolab import evaluate as ev
from autolab.config import REPO_ROOT
from autolab.program import Program

CARDS = REPO_ROOT / "autolab" / "research" / "cards.jsonl"
PROMPTS = Path(__file__).with_name("prompts")
CATEGORIES = ["attention", "mlp", "normalization", "residual", "positional", "initialization", "optimizer",
              "lr_schedule", "regularization", "gradient", "hyperparameters", "efficiency", "other"]

CARD = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "category", "component", "current", "proposal", "mechanism", "evidence",
                 "expected_effect", "applicability", "risks", "implementation"],
    "properties": {
        "name": {"type": "string", "maxLength": 120},
        "category": {"type": "string", "enum": CATEGORIES},
        "component": {"type": "string", "maxLength": 80,
                      "description": "EVOLVE block(s) or hyperparameter(s) it touches, e.g. 'optimizer', 'attention, hparams'"},
        "current": {"type": "string", "maxLength": 400, "description": "What the current program does now"},
        "proposal": {"type": "string", "maxLength": 600, "description": "What to change it to"},
        "mechanism": {"type": "string", "maxLength": 600},
        "evidence": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False, "required": ["source", "url", "finding"],
            "properties": {"source": {"type": "string", "maxLength": 200}, "url": {"type": "string", "maxLength": 300},
                           "year": {"type": "integer"}, "finding": {"type": "string", "maxLength": 400}}}},
        "expected_effect": {"type": "string", "maxLength": 300},
        "applicability": {"type": "string", "maxLength": 400, "description": "Why it should work at THIS scale/budget"},
        "risks": {"type": "string", "maxLength": 300},
        "implementation": {"type": "string", "maxLength": 1200, "description": "Sketch of the code/hparam change"},
    },
}
REPLY = {"type": "object", "additionalProperties": False, "required": ["summary", "cards"],
         "properties": {"summary": {"type": "string", "maxLength": 1500},
                        "cards": {"type": "array", "maxItems": 8, "items": CARD}}}


def research_cfg() -> dict:
    return tomllib.loads((REPO_ROOT / "autolab" / "config.toml").read_text()).get("research", {})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- card store ---------------------------------------------------------------------------------------


def load_cards(path: Path | None = None) -> list[dict]:
    path = path or CARDS
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _key(card: dict) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in card["name"]).split())


def add_cards(new: list[dict], meta: dict, path: Path | None = None) -> list[dict]:
    """Append new cards (skipping near-duplicates by normalized name), numbering them c1, c2, ..."""
    path = path or CARDS
    cards = load_cards(path)
    seen = {_key(c) for c in cards}
    added = []
    for c in new:
        if _key(c) in seen:
            continue
        seen.add(_key(c))
        card = {"id": f"c{len(cards) + len(added) + 1}", **c, "created": now_iso(), **meta, "status": "active"}
        added.append(card)
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            for card in added:
                fh.write(json.dumps(card) + "\n")
    return added


# --- card statistics and sampling (the "meta-prompt evolution" analogue) -------------------------------

REWARD = {"accepted": 1.0, "contender": 0.6, "evaluated": 0.3, "rejected": 0.0}


def all_programs() -> list[tuple[str, Program]]:
    out = []
    for paths in ev.all_session_paths():
        s = ev.load_session(paths)
        for p in ev.programs(paths).values():
            out.append((s.get("name") if s else paths.root.name, p))
    return out


def card_stats(cards: list[dict], progs: list[tuple[str, Program]] | None = None) -> dict[str, dict]:
    progs = progs if progs is not None else all_programs()
    stats = {c["id"]: {"used": 0, "finished": 0, "reward": 0.0, "outcomes": {}, "programs": []} for c in cards}
    for sess, p in progs:
        for cid in (p.meta or {}).get("technique_ids", []) or []:
            st = stats.get(cid)
            if st is None:
                continue
            st["used"] += 1
            st["programs"].append(f"{sess}/{p.id}")
            if p.status in REWARD:
                st["finished"] += 1
                st["reward"] += REWARD[p.status]
                st["outcomes"][p.status] = st["outcomes"].get(p.status, 0) + 1
    for st in stats.values():
        st["mean_reward"] = st["reward"] / st["finished"] if st["finished"] else None
    return stats


def pick_cards(k: int, rng: random.Random, cards: list[dict] | None = None, stats: dict | None = None,
               c_explore: float = 1.0) -> list[dict]:
    """UCB1 over finished outcomes: untried cards first, then mean reward + exploration bonus."""
    cards = [c for c in (cards if cards is not None else load_cards()) if c.get("status", "active") == "active"]
    if not cards:
        return []
    stats = stats if stats is not None else card_stats(cards)
    total = sum(s["finished"] for s in stats.values()) + 1

    def score(c):
        s = stats.get(c["id"], {"finished": 0, "reward": 0.0})
        if s["finished"] == 0:
            return 10.0 + rng.random()  # untried: explore first, random order among them
        return s["reward"] / s["finished"] + c_explore * math.sqrt(2 * math.log(total) / s["finished"])

    return sorted(cards, key=score, reverse=True)[:k]


def render_cards(cards: list[dict], stats: dict | None = None) -> str:
    """Cards as prompt context: claims to evaluate, not instructions."""
    out = []
    for c in cards:
        st = (stats or {}).get(c["id"])
        record = ""
        if st and st["finished"]:
            record = " Record so far: " + ", ".join(f"{k} {v}" for k, v in st["outcomes"].items()) + "."
        src = "; ".join(f"{e['source']} ({e.get('year', '?')}): {e['finding']}" for e in c["evidence"])
        out.append(f"### {c['id']}: {c['name']} [{c['category']} · {c['component']}]\n"
                   f"- Now: {c['current']}\n- Proposal: {c['proposal']}\n- Mechanism: {c['mechanism']}\n"
                   f"- Evidence: {src}\n- Applicability here: {c['applicability']}\n- Risks: {c['risks']}\n"
                   f"- Implementation sketch: {c['implementation']}{record}")
    return "\n\n".join(out)


# --- running the research agent ----------------------------------------------------------------------


def research_prompt(trigger: str, session: dict, progs: dict[str, Program], runs_dir: Path, cfg: dict,
                    rcfg: dict) -> str:
    import torch

    from autolab.prompt import _read_json, hparam_ranges, outcome, report_summary, session_history

    inc = progs[session["incumbent"]]
    p0 = progs["p0"]
    rep0 = _read_json(runs_dir / p0.runs["full"][0] / "report.json") or {}
    sc0 = rep0.get("scale", {})
    tried = [q for q in progs.values() if q.parent_id]
    blocks = "\n\n".join(f"### `{k}`\n```python\n{v}\n```" for k, v in inc.blocks.items())
    existing = load_cards()
    return (PROMPTS / "research.md").read_text().format(
        dataset_id=session.get("dataset_id"), dataset_tokens=sc0.get("dataset_tokens") or 0,
        full_tokens=session["budgets"]["full_tokens"],
        epochs=session["budgets"]["full_tokens"] / max(1, sc0.get("dataset_tokens") or 1),
        block_size=cfg["block_size"], gpu=cfg["gpu"], torch_version=torch.__version__.split("+")[0],
        cap_min=session["wall_caps"]["full"] / 60, param_cap=int(cfg["param_cap_mult"] * session["initial_params"]),
        params=inc.scores.get("params") or session["initial_params"],
        hparam_keys=", ".join(cfg["hparams"]), trigger=trigger, program=inc.id,
        full_mean=inc.scores.get("full_mean") or float("nan"), n_seeds=inc.scores.get("n_seeds") or 0,
        hparams=json.dumps(inc.hparams), report=report_summary(inc, runs_dir, session_history(progs, session, runs_dir)),
        blocks=blocks,
        tried="\n".join(f"- {q.id}: {' '.join(q.rationale.split())[:220]} → {outcome(q)}" for q in tried[-40:]) or "(nothing yet)",
        cards="\n".join(f"- {c['id']}: {c['name']} ({c['category']})" for c in existing) or "(none yet)",
        max_searches=rcfg.get("max_searches", 8), max_cards=rcfg.get("max_cards", 5),
    ) + "\n\nHyperparameter ranges:\n" + hparam_ranges(cfg["hparams"])


def run(trigger: str, paths: ev.Paths | None = None, caller=None, log=print) -> dict:
    """One research run: audit the incumbent, store new cards. Returns {"added": [...], "summary", "cost_usd"}."""
    from autolab.activity import set_activity
    from autolab.generate import llm_cfg
    from autolab.llm import call

    import fcntl

    lock_path = REPO_ROOT / "autolab" / "state" / "research.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:  # one research run at a time, however it was started
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another research run is in progress") from None
    paths = paths or ev.Paths()
    session = ev.load_session(paths)
    progs = ev.programs(paths)
    cfg, rcfg, lcfg = ev.evolve_cfg(), research_cfg(), llm_cfg()
    runs_dir = REPO_ROOT / "autolab" / "runs"
    prompt = research_prompt(trigger, session, progs, runs_dir, cfg, rcfg)
    tag = f"research-{session.get('name')}-{session['incumbent']}"
    set_activity("research", f"researching upgrades for {session['incumbent']} ({trigger[:80]})")
    res = (caller or call)(prompt, (PROMPTS / "research_system.md").read_text(), REPLY, rcfg.get("model", "opus"), lcfg,
                           paths.root.parent.parent / "llm" / "research", tag,
                           tools=["WebSearch", "WebFetch"], timeout_s=rcfg.get("timeout_s", 1500))
    added = add_cards(res["reply"]["cards"], {"session": session.get("name"), "for_program": session["incumbent"],
                                              "trigger": trigger, "research_log": res["log"]})
    log(f"research: {len(added)} new card(s) for {session['incumbent']} (${res['cost_usd']:.2f}): "
        + ", ".join(f"{c['id']} {c['name']}" for c in added))
    return {"added": [c["id"] for c in added], "summary": res["reply"]["summary"], "cost_usd": res["cost_usd"],
            "model": res["model"]}
