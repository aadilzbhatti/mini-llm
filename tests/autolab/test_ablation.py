"""The no-evolution ablation (autolab.ablation): a control arm with the same p0 and cascade, no feedback."""

import random
from datetime import datetime, timezone

import pytest

from autolab import ablation, memory
from autolab import controller as ctl
from autolab import evaluate as ev
from autolab.generate import generate_one

from test_generate import GOOD
from test_memory import add, chain, fill  # noqa: F401 (fixture)

T = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def arm(chain, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ev, "STATE_ROOT", chain["root"])
    ev.set_active_session("c", chain["root"])
    fill(chain, "c")                                      # 7 candidates
    add(chain, "c", "p8", "accepted", full=4.60, created_by="port:b/p3")  # a port: not a candidate
    s = ablation.start("c", root=chain["root"], runs_dir=chain["runs"])
    return {**chain, "session": s, "paths": ev.Paths(chain["root"] / "c~noevo")}


def test_start_copies_the_source_regime(arm):
    s, src = arm["session"], ev.load_session(ev.Paths(arm["root"] / "c"))
    assert s["name"] == "c~noevo" and s["ablation"]["target"] == 7 and s["ablation"]["of"] == "c"
    assert s["budgets"] == src["budgets"] and s["noise"]["full"] == {k: v for k, v in src["noise"]["full"].items()
                                                                     if k != "programs"}
    assert ev.programs(arm["paths"])["p0"].blocks == ev.programs(ev.Paths(arm["root"] / "c"))["p0"].blocks
    assert s.get("origin") is None  # off the regime chain
    assert "c~noevo" not in memory.session_weights("c", arm["root"])  # and out of the bandits
    with pytest.raises(FileExistsError):
        ablation.start("c", root=arm["root"], runs_dir=arm["runs"])


def test_control_arm_prompt_has_no_feedback(arm):
    add(arm, "c~noevo", "p1", "accepted", full=4.50)  # even a winner in the arm is never shown or used
    seen = {}

    def caller(prompt, system, schema, model, cfg, log_dir, tag, **kw):
        seen["prompt"] = prompt
        return {"reply": {**GOOD, "hparams": {"lr": 2e-3}, "technique_ids": []}, "cost_usd": 0.1,
                "model": "claude-opus-5-5", "duration_s": 1.0, "log": "x"}

    c = generate_one(paths=arm["paths"], rng=random.Random(0), log=lambda m: None, caller=caller, model="opus",
                     runs_dir=arm["runs"])
    p = seen["prompt"]
    assert c.parent_id == "p0" and c.meta["inspirations"] == [] and c.meta["cards_shown"] == []
    for absent in ("# Prior programs", "# What has been tried", "Pareto frontier so far", "# Recent rejections",
                   "# Relevant techniques", "idea c~noevo/p1"):
        assert absent not in p, absent
    assert "# Current program (p0)" in p and "# Task" in p


def test_controller_fills_the_arm_and_reports_once(arm, monkeypatch):
    notes = []
    monkeypatch.setattr(ctl, "note", lambda event, **f: notes.append((event, f)))
    monkeypatch.setattr(ctl, "affordable", lambda *a: True)

    def gen(log, paths=None, **kw):
        return ev.propose("p0", [], {"lr": 1e-3}, "x", created_by="fake", paths=paths)

    monkeypatch.setattr(ctl, "_generate", gen)
    cfg = {"ablation_max_in_flight": 2}
    assert ctl.ablation_step({}, cfg, T, log=lambda m: None) == 2  # capped by in-flight
    assert ctl.ablation_step({}, cfg, T, log=lambda m: None) == 0
    # finish them all, as the cascade would: the arm's best is 4.70 vs the evolution arm's 4.70 (c/p5)
    progs = ev.programs(arm["paths"])
    for pid in ("p1", "p2"):
        add(arm, "c~noevo", pid, "evaluated", full=4.80, full_losses=[4.80])
    for i in range(3, 8):
        add(arm, "c~noevo", f"p{i}", "rejected", "screen", reason="screen")
    assert len(ablation.candidates(ev.programs(arm["paths"]))) == 7 and progs
    assert ctl.ablation_step({}, cfg, T, log=lambda m: None) == 0
    done = [f for e, f in notes if e == "ablation_finished"]
    assert len(done) == 1 and done[0]["compared_at"] == 7
    assert done[0]["arms"]["no_evolution"]["best_first_seed_full"] == 4.80
    ctl.ablation_step({}, cfg, T, log=lambda m: None)
    assert len([e for e, _ in notes if e == "ablation_finished"]) == 1  # once


def test_compare_at_equal_candidate_counts(arm):
    for pid, full in (("p1", 4.75), ("p2", 4.60)):
        add(arm, "c~noevo", pid, "evaluated", full=full, full_losses=[full])
    cmp = ablation.compare("c", arm["root"])
    assert cmp["compared_at"] == 2  # the arm has 2 finished; the evolution arm's first 2 are a bug and a screen
    assert cmp["arms"]["evolution"]["best_first_seed_full"] is None
    assert cmp["arms"]["no_evolution"]["best_first_seed_full"] == 4.60
    assert "No-evolution ablation: c~noevo vs c" in ablation.render(cmp)


def test_ports_skip_the_arm(arm):
    add(arm, "c~noevo", "p1", "accepted", full=4.50)
    session = ev.load_session(ev.Paths(arm["root"] / "c"))
    state = {}
    ported = ctl.port_from_old_sessions(state, session, log=lambda m: None)
    assert not any("noevo" in k for k in state.get("ported", [])) and ported is not None
