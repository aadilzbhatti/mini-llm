"""labs.pack: the harness's own check of a packing, and the lab's accept / contender / finite-run logic."""

import math

import pytest
from labs_testkit import run_to_done

from labs import core
from labs.pack.harness import check
from labs.pack.lab import PackLab

GOOD = PackLab.seed_file.read_text()


def test_check_verifies_the_packing_not_the_claim():
    one = [(0.5, 0.5, 0.5)]
    assert check(one, 1) == (0.5, None)
    assert "leaves the unit square" in check([(0.5, 0.5, 0.6)], 1)[1]
    assert "overlap" in check([(0.25, 0.5, 0.25), (0.5, 0.5, 0.25)], 2)[1]
    assert "expected 2" in check(one, 2)[1]
    assert "not finite" in check([(0.5, 0.5, math.nan)], 1)[1]
    assert "not a list" in check([(0.5, 0.5)], 1)[1]
    s, err = check([(0.25, 0.5, 0.25), (0.75, 0.5, 0.25 + 5e-10)], 2)  # within the tolerance
    assert err is None and s == pytest.approx(0.5)


@pytest.fixture
def lab(world):
    lab = PackLab()
    lab.init(log=lambda m: None)  # the real seed, through the real harness
    return lab


def test_seed_runs_and_scores(lab):
    p0 = lab.programs()["p0"]
    assert p0.status == "accepted" and 1.4 < p0.scores["score"] < 1.6 and len(p0.scores["circles"]) == 26


def test_better_is_accepted_close_is_a_contender_worse_is_evaluated(lab, monkeypatch):
    base = lab.programs()["p0"].scores["score"]
    for pid, score, want in (("p1", base + 0.1, "accepted"), ("p2", base + 0.095, "contender"), ("p3", 1.0, "evaluated")):
        monkeypatch.setattr(lab, "run_program", lambda p, s=score: {"ok": True, "score": s, "wall_s": 1.0, "circles": []})
        assert run_to_done(lab, core.Program(pid, "p0", GOOD)).status == want
    assert lab.load_lab()["incumbent"] == "p1"
    assert lab.exhausted() == "calibration run done: 3 of 3 programs"
    assert lab.cycle(log=lambda m: None) == {"waiting": "calibration run done: 3 of 3 programs"}


def test_invalid_packing_is_rejected(lab):
    bad = GOOD.replace("return [(x, y, r) for (x, y), r in zip(centers, radii)]",
                       "return [(x, y, r * 1.5) for (x, y), r in zip(centers, radii)]")
    p = run_to_done(lab, core.Program("p1", "p0", bad))
    assert p.status == "rejected" and ("overlap" in p.reason or "leaves" in p.reason)
