"""Prompt sampler (AlphaEvolve §2.2): turn a parent + inspirations into one LLM prompt.

Sections, in order:
1. explicit context: the task, what the evaluator fixes, how the cascade judges a proposal,
   what may change (prompts/context.md, filled from the session and config);
2. prior programs: each inspiration's changes vs the initial program (unified diff of its
   blocks + hparam changes), its scores and rationale;
3. the current program: every EVOLVE block verbatim, its hparams and scores, a compact
   training report and the rule-based diagnosis with evidence ("rendered evaluation results");
4. what has been tried: one line per recent program, and the latest rejections with reasons;
5. the task instruction, drawn from prompts/instructions.toml (stochastic formatting).
"""

from __future__ import annotations

import difflib
import json
import random
import tomllib
from pathlib import Path

from autolab.program import Program

PROMPTS = Path(__file__).with_name("prompts")

REPLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale", "expected_effect", "diffs", "hparams"],
    "properties": {
        "rationale": {"type": "string", "minLength": 10, "maxLength": 2000,
                      "description": "What you change and the mechanism by which it should lower val loss."},
        "expected_effect": {"type": "string", "maxLength": 500,
                            "description": "Predicted effect on full val loss / throughput, with rough size."},
        "diffs": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False, "required": ["search", "replace"],
            "properties": {"search": {"type": "string", "minLength": 1}, "replace": {"type": "string"}}}},
        "hparams": {"type": "object", "description": "Only the hyperparameters to change (may be empty).",
                    "additionalProperties": False, "properties": {
                        k: {"type": "number"} for k in ("lr", "min_lr", "warmup_steps", "weight_decay", "dropout",
                                                        "batch_size", "n_embd", "n_head", "n_layer")}},
    },
}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def report_summary(p: Program, runs_dir: Path, history=None) -> str:
    """Compact rendering of the program's first full-budget run (else its screen).

    The diagnosis is recomputed here with the session's history (other programs' reports, the
    data-check notebook, seed noise): the one stored with the run was made in the Modal
    container without history, so e.g. capacity_limited ("a data increase didn't help") can't
    appear in it.
    """
    rid = (p.runs.get("full") or p.runs.get("screen") or [None])[0]
    rep = _read_json(runs_dir / rid / "report.json") if rid else None
    if not rep:
        return "(no training report yet)"
    s, sc, h, perf = rep["summary"], rep["scale"], rep["health"], rep["performance"]
    gn = h.get("grad_norm") or {}
    ts, vs, gt = s.get("train_slope_tail") or {}, s.get("val_slope_tail") or {}, s.get("gap_trend") or {}

    def f(x, d=4):
        return "n/a" if x is None else f"{x:.{d}f}"

    lines = [
        f"- run {rid}: {sc.get('tokens_seen', 0):,} tokens, {sc.get('epochs') or 0:.2f} epochs, "
        f"{sc.get('params', 0):,} params ({sc.get('non_embedding_params', 0):,} non-embedding), "
        f"{sc.get('tokens_per_param') or 0:.1f} tokens/param",
        f"- final full val loss {f(s.get('final_full_val_loss'))}; smoothed final train {f(s.get('final_train_loss_smooth'))}, "
        f"val {f(s.get('final_val_loss_smooth'))} (fixed eval batches), gap val−train {f(s.get('gap'))}",
        f"- last 20% of training: train changed {f(ts.get('rel_change'), 4)} (relative), val {f(vs.get('rel_change'), 4)}, "
        f"gap changed {f(gt.get('change'))}",
        f"- LR peak {s.get('lr_peak')} → final {s.get('lr_final')}; grad-norm median {gn.get('median')}, "
        f"p95 {gn.get('p95')}, max/median {gn.get('max_over_median')}; loss spikes {(h.get('spikes') or {}).get('count')}",
        f"- throughput {perf.get('tokens_per_sec') or 0:,.0f} tokens/s, train wall {perf.get('train_wall_s') or 0:.0f} s",
    ]
    diag = _read_json(runs_dir / rid / "diagnosis.json")
    if history is not None:
        from autolab.diagnose import diagnose

        try:
            diag = diagnose(rep, history).to_dict()
        except (KeyError, TypeError):
            pass  # fall back to the stored diagnosis
    if diag:
        for lab in diag["labels"]:
            ev = ", ".join(f"{k}={v}" for k, v in lab["evidence"].items())
            lines.append(f"- diagnosis **{lab['name']}** (confidence {lab['confidence']}): {ev}. Suggests: {lab['suggestion']}")
        lines += [f"- note: {n}" for n in diag.get("notes", [])]
    return "\n".join(lines)


def session_history(progs: dict[str, Program], session: dict, runs_dir: Path):
    """History for diagnose(): full-run reports of this session's programs, the session's
    notebook (data checks), and the full-budget seed noise."""
    from autolab.diagnose import History

    reports = []
    for q in progs.values():
        rid = (q.runs.get("full") or [None])[0]
        rep = _read_json(runs_dir / rid / "report.json") if rid else None
        if rep:
            reports.append(rep)
    return History(reports=reports, notebook=session.get("notebook", []), noise_std=session["noise"]["full"]["std"])


def block_diff(a: dict[str, str], b: dict[str, str]) -> str:
    out = []
    for key in sorted(set(a) | set(b)):
        if a.get(key, "") != b.get(key, ""):
            out += difflib.unified_diff(a.get(key, "").splitlines(), b.get(key, "").splitlines(),
                                        f"initial/{key}", f"this/{key}", lineterm="", n=2)
    return "\n".join(out)


def hparam_changes(base: dict, hp: dict) -> str:
    ch = [f"{k}: {base.get(k)} → {v}" for k, v in hp.items() if base.get(k) != v]
    return ", ".join(ch) if ch else "none"


def score_line(p: Program) -> str:
    sc = p.scores
    parts = []
    if sc.get("full_mean") is not None:
        parts.append(f"full val loss {sc['full_mean']:.4f} over {sc.get('n_seeds', 1)} seed(s)")
    if sc.get("screen_loss") is not None:
        parts.append(f"screen {sc['screen_loss']:.4f}")
    if sc.get("params"):
        parts.append(f"{sc['params']:,} params")
    if sc.get("tokens_per_sec"):
        parts.append(f"{sc['tokens_per_sec']:,.0f} tok/s")
    return "; ".join(parts) or "not scored"


def outcome(p: Program) -> str:
    if p.status == "rejected":
        return f"rejected at {p.stage}: {p.reason[:160]}"
    if p.status in ("evaluated", "contender", "accepted"):
        return f"{p.status}: {score_line(p)}"
    return f"{p.status} (in evaluation, stage {p.stage})"


def hparam_ranges(spec: dict) -> str:
    rows = []
    for k, r in spec.items():
        rows.append(f"  - `{k}`: " + (f"one of {r['choices']}" if "choices" in r else
                                       f"{r['min']} … {r['max']}" + (" (integer)" if r.get("int") else "")))
    return "\n".join(rows)


def pick_instruction(weights: dict[str, float], rng: random.Random) -> tuple[str, str]:
    texts = tomllib.loads((PROMPTS / "instructions.toml").read_text())
    keys = [k for k in weights if k in texts and weights[k] > 0]
    key = rng.choices(keys, weights=[weights[k] for k in keys])[0]
    return key, texts[key]


def build_prompt(parent: Program, inspirations: list[Program], progs: dict[str, Program], session: dict,
                 cfg: dict, llm_cfg: dict, runs_dir: Path, rng: random.Random) -> tuple[str, dict]:
    """Return (prompt text, metadata about how it was built)."""
    p0 = progs["p0"]
    inc = progs[session["incumbent"]]
    sigma = session["noise"]["full"]["std"]
    budgets = session["budgets"]
    p0_rep = _read_json(runs_dir / p0.runs["full"][0] / "report.json") or {}
    sc0 = p0_rep.get("scale", {})
    context = (PROMPTS / "context.md").read_text().format(
        val_tokens=sc0.get("val_tokens") or 0, dataset_tokens=sc0.get("dataset_tokens") or 0,
        dataset_id=session.get("dataset_id") or cfg.get("dataset_id", "data20k"),
        full_tokens=budgets["full_tokens"], screen_tokens=budgets["screen_tokens"],
        epochs=budgets["full_tokens"] / (sc0.get("dataset_tokens") or 1), block_size=cfg["block_size"],
        gpu=cfg["gpu"], full_cap_min=session["wall_caps"]["full"] / 60, wall_cap_mult=cfg["wall_cap_mult"],
        param_cap=int(cfg["param_cap_mult"] * session["initial_params"]), param_cap_mult=cfg["param_cap_mult"],
        throughput_floor=cfg["throughput_floor"], screen_margin=cfg["screen_margin"],
        inc_screen=inc.scores.get("screen_loss") or float("nan"), confirm_sigma=cfg.get("confirm_trigger_sigma", 0),
        bar=inc.scores["full_mean"] - cfg["accept_sigma"] * sigma, inc_full=inc.scores["full_mean"],
        accept_sigma=cfg["accept_sigma"], sigma=sigma, hparam_ranges=hparam_ranges(cfg["hparams"]),
    )
    parts = [context]

    if inspirations:
        parts.append("# Prior programs\n\nPreviously evaluated programs, as changes relative to the initial program:")
        for q in inspirations:
            d = block_diff(p0.blocks, q.blocks)
            parts.append(f"## Program {q.id} — {score_line(q)}\n\nRationale: {q.rationale or '(initial program)'}\n\n"
                         f"Hyperparameter changes vs initial: {hparam_changes(p0.hparams, q.hparams)}\n\n"
                         + (f"```diff\n{d}\n```" if d else "No code changes vs the initial program."))

    blocks = "\n\n".join(f"### `{key.split(':')[0]}` — block `{key.split(':')[1]}`\n```python\n{text}\n```"
                         for key, text in parent.blocks.items())
    parts.append(
        f"# Current program ({parent.id}) — the one to modify\n\n{score_line(parent)}. "
        f"Rationale when it was made: {parent.rationale or '(initial program)'}\n\n"
        f"Hyperparameters: {json.dumps(parent.hparams)}\n\n## Training report and diagnosis\n\n"
        f"{report_summary(parent, runs_dir, session_history(progs, session, runs_dir))}\n\n## EVOLVE blocks\n\n{blocks}")

    others = [q for q in progs.values() if q.parent_id is not None]
    history = others[-cfg["database"].get("history", 30):]
    if history:
        parts.append("# What has been tried (latest first)\n\n" + "\n".join(
            f"- {q.id} (parent {q.parent_id}, by {q.created_by}): {q.rationale[:180]} → {outcome(q)}"
            for q in reversed(history)))
    fails = [q for q in others if q.status == "rejected"][-cfg["database"].get("recent_failures", 5):]
    if fails:
        parts.append("# Recent rejections and why (avoid these mistakes)\n\n" + "\n".join(
            f"- {q.id}: {q.rationale[:200]}\n  → rejected at **{q.stage}**: {q.reason[:400]}" for q in reversed(fails)))

    key, instruction = pick_instruction(llm_cfg.get("instructions", {"open": 1.0}), rng)
    parts.append(f"# Task\n\n{instruction}\n\nState the mechanism in `rationale`, predict the effect in "
                 f"`expected_effect`, and give the change as `diffs` (exact SEARCH text from the current "
                 f"program's blocks) and/or `hparams`.")
    return "\n\n".join(parts), {"instruction": key, "parent": parent.id, "inspirations": [q.id for q in inspirations]}
