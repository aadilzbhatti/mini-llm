"""Make new children: sample the database, prompt Claude, apply its diffs; fall back to mutation.

    uv run autolab evolve generate -n 3

One child per call of `generate_one`:
1. pick the island (round-robin) and sample a parent + inspirations (autolab.database);
2. build the prompt (autolab.prompt) and pick a model from [llm] models by weight;
3. call Claude (autolab.llm). The reply's diffs and hparam patch go through the same `propose()`
   as hand-written patches, so the scope check applies identically;
4. on any proposer failure (timeout, non-zero exit, bad JSON, schema failure, no actual change,
   or diffs that don't apply) fall back to `mutate()`: perturb 1–2 hyperparameters within range.
   The fallback reason is kept on the child (meta.fallback_reason). A child whose diffs don't apply
   is still saved as a rejected program, because its failure shows up in later prompts.

The daemon then carries every queued child through the evaluation cascade.
"""

from __future__ import annotations

import math
import random
import tomllib

from autolab import evaluate as ev
from autolab.config import REPO_ROOT
from autolab.database import DBConfig, maybe_migrate, next_island, sample
from autolab.llm import LLMError, RateLimited, call
from autolab.program import Program, save, validate_hparams
from autolab.prompt import PROMPTS, REPLY_SCHEMA, build_prompt

INT_KEYS = {"warmup_steps", "batch_size", "n_embd", "n_head", "n_layer"}


def llm_cfg() -> dict:
    return tomllib.loads((REPO_ROOT / "autolab" / "config.toml").read_text())["llm"]


def mutate(parent: Program, spec: dict, rng: random.Random) -> tuple[dict, str]:
    """Perturb 1-2 optimization hyperparameters of the parent, staying inside `spec`."""
    hp = parent.hparams
    # batch_size only moves together with block_size (tokens per update is fixed), see below
    keys = [k for k in ("lr", "warmup_steps", "weight_decay", "min_lr", "dropout", "block_size") if k in spec]
    patch: dict = {}
    for k in rng.sample(keys, rng.choice([1, 2])):
        r, v = spec[k], hp.get(k)
        if k == "block_size":
            opts = sorted(r["choices"])
            cur = v or 128
            i = opts.index(cur) if cur in opts else 0
            new = opts[max(0, min(len(opts) - 1, i + rng.choice([-1, 1])))]
            if new != cur:  # keep tokens per update: batch x context constant
                patch["batch_size"] = int(hp["batch_size"] * cur // new)
        elif "choices" in r:
            opts = sorted(r["choices"])
            i = opts.index(v) if v in opts else 0
            new = opts[max(0, min(len(opts) - 1, i + rng.choice([-1, 1])))]
        elif k == "lr":
            new = v * math.exp(rng.gauss(0, 0.35))
        elif k == "warmup_steps":
            new = int(round(max(v, 8) * rng.choice([0.25, 0.5, 2.0, 4.0])))
        elif k == "weight_decay":
            new = 10 ** rng.uniform(-2.5, -0.7) if not v else v * math.exp(rng.gauss(0, 0.6))
        elif k == "min_lr":
            new = (v or 1e-6) * 10 ** rng.uniform(-1, 1)
        else:  # dropout
            new = (v or 0.0) + rng.choice([-0.05, 0.05, 0.1])
        if "min" in r:
            new = min(max(new, r["min"]), r["max"])
        if k in INT_KEYS:
            new = int(round(new))
        elif isinstance(new, float):
            new = float(f"{new:.3g}")
        if new != v:
            patch[k] = new
    if "min_lr" in patch or "lr" in patch:  # keep min_lr <= lr
        lr = patch.get("lr", hp.get("lr"))
        if patch.get("min_lr", hp.get("min_lr", 0)) > lr:
            patch["min_lr"] = float(f"{lr / 100:.3g}")
    if not patch:
        return mutate(parent, spec, rng)
    return patch, "mutation: " + ", ".join(f"{k} {hp.get(k)} → {v}" for k, v in patch.items())


def _clean_hparams(raw: dict, parent: Program) -> dict:
    out = {}
    for k, v in (raw or {}).items():
        v = int(round(v)) if k in INT_KEYS else float(v)
        if parent.hparams.get(k) != v:
            out[k] = v
    return out


def generate_one(paths: ev.Paths | None = None, rng: random.Random | None = None, log=print,
                 caller=call, model: str | None = None, runs_dir=None, card: dict | None = None,
                 cards: list[dict] | None = None) -> Program:
    """One child. `card` = directed: apply that technique card to the incumbent (no sampling)."""
    paths = paths or ev.Paths()
    rng = rng or random.Random()
    cfg, lcfg = ev.evolve_cfg(), llm_cfg()
    session = ev.load_session(paths)
    if session is None:
        raise RuntimeError("no evolve session")
    progs = ev.programs(paths)
    dbc = DBConfig.from_cfg(cfg)
    if maybe_migrate(progs, session, dbc):
        ev.save_session(session, paths)
        log(f"migrated island bests: {session['database']['migrants']}")
    island = next_island(progs, dbc)
    parent, insp = sample(progs, session, island, dbc, rng)
    if card is not None:  # directed: the card is an upgrade proposed for the incumbent
        parent = progs[session["incumbent"]]
    runs_dir = runs_dir or REPO_ROOT / "autolab" / "runs"
    prompt, pmeta = build_prompt(parent, insp, progs, session, cfg, lcfg, runs_dir, rng, card=card, cards=cards)
    models = lcfg.get("models", {"opus": 1.0})
    model = model or rng.choices(list(models), weights=list(models.values()))[0]
    meta = {**pmeta, "island": island, "model_requested": model}
    tag = f"{session.get('name', 'session')}-{session['next_id']}"

    fallback = None
    from autolab.activity import set_activity

    set_activity("propose", f"asking Claude ({model}) for a child of {parent.id} — instruction: {pmeta['instruction']}",
                 parent=parent.id, model=model)
    try:
        res = caller(prompt, (PROMPTS / "system.md").read_text(), REPLY_SCHEMA, model, lcfg,
                     paths.root.parent.parent / "llm" / session.get("name", "session"), tag)
        reply = res["reply"]
        known = set(pmeta.get("cards_shown", []))
        used = [c for c in reply.get("technique_ids", []) if c in known]
        if card is not None and card["id"] not in used:
            used.insert(0, card["id"])  # a directed child always counts for its card
        meta.update(model=res["model"], cost_usd=res["cost_usd"], llm_s=res["duration_s"], log=res["log"],
                    expected_effect=reply.get("expected_effect", ""), technique_ids=used)
        hp = _clean_hparams(reply.get("hparams"), parent)
        diffs = [d for d in reply.get("diffs", []) if d["search"] != d["replace"]]
        problems = validate_hparams({**parent.hparams, **hp}, cfg["hparams"]) if hp else []
        if not diffs and not hp:
            raise LLMError("reply changes nothing")
        if problems:
            raise LLMError("hparams out of range: " + "; ".join(problems))
        child = ev.propose(parent.id, diffs, hp, reply["rationale"], created_by=res["model"], paths=paths)
        child.island, child.meta = island, meta
        save(child, paths.programs)
        if child.status == "rejected" and child.stage == "static" and child.reason.startswith("scope:"):
            fallback = f"diffs don't apply ({child.reason[:200]})"
            log(f"{child.id}: LLM diffs rejected: {child.reason[:200]}")
        else:
            log(f"{child.id}: from {res['model']} on parent {parent.id} (island {island}, ${res['cost_usd']:.3f}): "
                f"{reply['rationale'][:120]}")
            return child
    except RateLimited:
        raise  # a usage limit: stop proposing until it resets rather than filling the queue with mutations
    except LLMError as exc:
        fallback = str(exc)
        log(f"LLM proposal failed ({fallback}); falling back to mutation")

    patch, why = mutate(parent, cfg["hparams"], rng)
    child = ev.propose(parent.id, [], patch, f"{why} (LLM fallback: {fallback[:200]})", created_by="mutation",
                       paths=paths)
    child.island, child.meta = island, {**meta, "fallback_reason": fallback}
    save(child, paths.programs)
    log(f"{child.id}: mutation on parent {parent.id}: {why}")
    return child
