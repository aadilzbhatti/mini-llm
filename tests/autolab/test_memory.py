"""Regime-scoped memory (autolab.memory): what a rejection suppresses, for how many regime changes,
near-miss retests, and the card bandit's per-regime decay."""

import json
import random
import subprocess

import pytest

from autolab import controller as ctl
from autolab import evaluate as ev
from autolab import memory, research
from autolab.config import REPO_ROOT
from autolab.generate import generate_one, llm_cfg
from autolab.program import Program, save
from autolab.prompt import build_prompt, outcome

from test_generate import CARD, GOOD, HP, report

SIGMA = 0.02  # the three base runs' std (0.01) is below [evolve] noise_floor


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Sessions a → b → c (each started from the previous one), p0 full mean 4.72 in each."""
    runs = tmp_path / "runs"
    for i, (screen, full) in enumerate([(5.95, 4.72), (5.96, 4.71), (5.94, 4.73)], 1):
        for kind, loss in (("screen", screen), ("full", full)):
            (runs / f"{kind}{i}").mkdir(parents=True)
            (runs / f"{kind}{i}" / "report.json").write_text(json.dumps(report(loss)))
    root = tmp_path / "state" / "evolve"
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    origin = None
    for name, tokens in (("a", 20_971_520), ("b", 81_920_000), ("c", 122_880_000)):
        ev.init_session(head, HP, {"screen": ["screen1", "screen2", "screen3"], "full": ["full1", "full2", "full3"]},
                        paths=ev.Paths(root / name), runs_dir=runs, name=name, dataset_id="data40k", origin=origin,
                        budgets={"screen_tokens": 10_485_760, "full_tokens": tokens, "eval": {"eval_interval": 100}})
        origin = {"session": name, "program": "p0", "via": "ladder", "detail": "test"}
    return {"root": root, "runs": runs, "tmp": tmp_path}


def add(chain, session, pid, status, stage="done", full=None, reason="", at="2026-09-29T00:00:00+00:00", **kw):
    p = Program(id=pid, parent_id="p0", base_commit="x", blocks=kw.pop("blocks", {}), hparams=kw.pop("hparams", HP),
                status=status, stage=stage, reason=reason, created_at=at, rationale=f"idea {session}/{pid}",
                scores={"full_mean": full} if full is not None else {}, **kw)
    save(p, ev.Paths(chain["root"] / session).programs)
    return p


def fill(chain, session):
    """One program of every kind, in the order the table in autolab.memory lists them."""
    add(chain, session, "p1", "rejected", "cpu", reason="shape: 1 failed")
    add(chain, session, "p2", "rejected", "screen", reason="screen 6.3 > incumbent 5.95 + margin 0.25")
    add(chain, session, "p3", "evaluated", full=4.72 + 4 * SIGMA)   # +4σ: clearly worse
    add(chain, session, "p4", "evaluated", full=4.72 + 1 * SIGMA)   # +1σ: near miss
    add(chain, session, "p5", "contender", full=4.70)
    add(chain, session, "p6", "rejected", "full", reason="run failed: container OOM")
    add(chain, session, "p7", "rejected", "full", reason="NaN/inf in training")


def kinds(mem):
    return {m.key: (m.distance, m.kind) for m in mem}


def load(chain, name):
    paths = ev.Paths(chain["root"] / name)
    return ev.load_session(paths), ev.programs(paths)


def test_classify(chain):
    fill(chain, "c")
    session, progs = load(chain, "c")
    got = {pid: memory.classify(progs[pid], session, progs, SIGMA, 2.0)[0] for pid in progs if pid != "p0"}
    assert got == {"p1": "bug", "p2": "screen", "p3": "worse", "p4": "near_miss", "p5": "near_miss", "p6": "bug",
                   "p7": "worse"}
    assert memory.classify(progs["p4"], session, progs, SIGMA, 2.0)[1] == pytest.approx(1.0)


def test_near_miss_is_judged_against_the_incumbent_of_its_time(chain):
    session, _ = load(chain, "c")
    session["incumbent_history"] = [{"at": "2026-09-29T01:00:00+00:00", "program": "p9", "full_mean": 4.60}]
    ev.save_session(session, ev.Paths(chain["root"] / "c"))
    before = add(chain, "c", "p1", "evaluated", full=4.75, at="2026-09-29T00:30:00+00:00")  # vs 4.72: +1.5σ
    after = add(chain, "c", "p2", "evaluated", full=4.75, at="2026-09-29T02:00:00+00:00")   # vs 4.60: +7.5σ
    session, progs = load(chain, "c")
    assert memory.classify(before, session, progs, SIGMA, 2.0)[0] == "near_miss"
    assert memory.classify(after, session, progs, SIGMA, 2.0)[0] == "worse"


def test_what_each_regime_distance_remembers(chain):
    for s in ("a", "b", "c"):
        fill(chain, s)
    add(chain, "b", "p8", "accepted", full=4.60)
    session, progs = load(chain, "c")
    got = kinds(memory.remembered(session, progs, ev.evolve_cfg(), chain["root"]))
    # this regime: everything, bugs included (tagged as untested)
    assert {k for k in got if k.startswith("c/")} == {f"c/p{i}" for i in range(1, 8)}
    # one change ago: full-run failures (weak) and near misses; not bugs, screens, or the (ported) acceptance
    assert {k: v for k, v in got.items() if k.startswith("b/")} == {
        "b/p3": (1, "worse"), "b/p4": (1, "near_miss"), "b/p5": (1, "near_miss"), "b/p7": (1, "worse")}
    # two changes ago: only near misses
    assert {k: v for k, v in got.items() if k.startswith("a/")} == {"a/p4": (2, "near_miss"), "a/p5": (2, "near_miss")}


def test_off_chain_sessions_are_not_remembered_and_history_is_capped(chain):
    fill(chain, "a")
    session, progs = load(chain, "a")
    session["origin"] = None
    assert all(m.session == "a" for m in memory.remembered(session, progs, ev.evolve_cfg(), chain["root"]))
    session, progs = load(chain, "c")
    fill(chain, "b")
    cfg = {**ev.evolve_cfg(), "database": {"history": 3}}
    mem = memory.remembered(session, progs, cfg, chain["root"])
    assert len(mem) == 3


def test_tags_tell_the_proposer_what_the_entry_means(chain):
    fill(chain, "b")
    fill(chain, "c")
    session, progs = load(chain, "c")
    text = memory.render(memory.remembered(session, progs, ev.evolve_cfg(), chain["root"]), outcome)
    line = {ln.split(" ")[1]: ln for ln in text.splitlines()}
    assert "attempt failed (bug)" in line["p1"] and "idea is untested" in line["p1"]
    assert "don't repeat in this regime (weak evidence)" in line["p2"]
    assert "worse at full budget, +4.0σ" in line["p3"] and "don't repeat" in line["p3"]
    assert "near miss, +1.0σ" in line["p4"]
    assert "in b: data40k, 82M tokens, 1 regime change ago" in line["b/p3"] and "weak evidence here" in line["b/p3"]
    assert "worth retesting here" in line["b/p4"]


def test_prompt_uses_the_regime_memory(chain):
    fill(chain, "b")
    fill(chain, "c")
    session, progs = load(chain, "c")
    text, meta = build_prompt(progs["p0"], [], progs, session, ev.evolve_cfg(), llm_cfg(), chain["runs"],
                              random.Random(0), cards=[], state_root=chain["root"])
    assert "# What has been tried (this regime first" in text and "- b/p4 (parent p0" in text
    assert "rejected at **cpu** (bug): shape" in text  # recent rejections flag bugs too
    assert meta["retest_of"] is None


def test_retest_candidates(chain):
    for s in ("a", "b"):
        fill(chain, s)
    session, _ = load(chain, "c")
    picks = [m.key for m in memory.retest_candidates(session, ev.evolve_cfg(), chain["root"], set())]
    assert picks == ["b/p5", "b/p4", "a/p5"]  # nearest regime first, contenders first, then closest to winning
    picks = [m.key for m in memory.retest_candidates(session, ev.evolve_cfg(), chain["root"], {"b/p5"}, k=2)]
    assert picks == ["b/p4", "a/p5"]


def test_retest_child_reapplies_the_change_to_the_incumbent(chain):
    session, progs = load(chain, "b")
    blocks = dict(progs["p0"].blocks)
    key = next(k for k in blocks if k.endswith(":attention"))
    blocks[key] = blocks[key] + "\n        # a marker line"
    add(chain, "b", "p4", "evaluated", full=4.74, blocks=blocks, hparams={**HP, "warmup_steps": 300})
    seen = {}

    def caller(prompt, system, schema, model, cfg, log_dir, tag, **kw):
        seen["prompt"] = prompt
        return {"reply": {**GOOD, "hparams": {"warmup_steps": 300}, "technique_ids": []}, "cost_usd": 0.1,
                "model": "claude-opus-5-5", "duration_s": 1.0, "log": "x"}

    c = generate_one(paths=ev.Paths(chain["root"] / "c"), rng=random.Random(0), log=lambda m: None, caller=caller,
                     model="opus", runs_dir=chain["runs"], cards=[], retest="b/p4")
    p = seen["prompt"]
    assert "Retest a near miss from an earlier regime. Program **b/p4** (b: data40k, 82M tokens)" in p
    assert "warmup_steps: 100 → 300" in p and "+        # a marker line" in p
    assert c.parent_id == "p0" and c.meta["retest_of"] == "b/p4" and c.meta["instruction"] == "retest"


def test_controller_queues_retests_once_per_regime_change(chain, monkeypatch):
    monkeypatch.setattr(ev, "STATE_ROOT", chain["root"])
    notes = []
    monkeypatch.setattr(ctl, "note", lambda event, **f: notes.append((event, f)))
    fill(chain, "b")
    state = {}
    session, _ = load(chain, "c")
    assert ctl.retest_step(state, session, log=lambda m: None) == ["b/p5", "b/p4"]
    assert state["retest"]["pending"] == ["b/p5", "b/p4"] and notes[0][0] == "retest_queued"
    assert ctl.retest_step(state, session, log=lambda m: None) == []  # once per session
    first, _ = load(chain, "a")
    assert ctl.retest_step(state, first, log=lambda m: None) == []  # no origin: no regime change to react to


def test_card_record_halves_per_regime_change(chain):
    card = research.add_cards([CARD], {"session": "c"})[0]
    progs = [(s, Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams=HP, status="accepted",
                         meta={"technique_ids": [card["id"]]})) for s in ("a", "b", "c")]
    weights = memory.session_weights("c", chain["root"], 0.5)
    assert weights == {"a": 0.25, "b": 0.5, "c": 1.0}
    st = research.card_stats([card], progs, weights)[card["id"]]
    assert st["finished"] == pytest.approx(1.75) and st["outcomes"] == {"accepted": 3}
    assert st["outcomes_here"] == {"accepted": 1}
    text = research.render_cards([card], {card["id"]: st})
    assert "Record in this regime: accepted 1. All regimes: accepted 3" in text
