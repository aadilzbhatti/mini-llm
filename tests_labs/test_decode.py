"""labs.decode: the paired verdict and its guards (stubbed harness), and the real harness on a tiny CPU model."""


import pytest
from labs_testkit import run_to_done, seed
from labs_infer_tiny import run, tiny  # noqa: F401 - the tiny checkpoint fixture

from labs import core
from labs.decode.lab import DecodeLab

GOOD = DecodeLab.seed_file.read_text()


def sample(degenerate=False, tokens=256, judge=2.0, human=3.3, topic=0.2):
    return {"degenerate": degenerate, "near_loop": degenerate, "tokens": tokens, "judge_nll": judge, "human_nll": human,
            "topic": topic, "rep4": 0.6 if degenerate else 0.1}


def stub(lab, monkeypatch, cand, inc):
    def h(mode, out, timeout, **args):
        if mode == "owner":
            return {"ok": True, "degenerate": 20, "n": 100, "samples": [sample()] * 100}
        if "against" not in args:
            return {"ok": True, "samplers": {"cand": {"samples": [sample()] * 16, "sampler_ms_per_token": 0.5}}}
        return {"ok": True, "samplers": {"cand": {"samples": cand, "sampler_ms_per_token": 0.5},
                                         "inc": {"samples": inc, "sampler_ms_per_token": 0.5}}}
    monkeypatch.setattr(lab, "h", h)


@pytest.fixture
def lab(world):
    lab = DecodeLab()
    seed(lab, holdout={"degenerate": 36, "n": 100})
    return lab


def mixed(n_degen, n=4, **kw):
    return [sample(degenerate=i < n_degen, **kw) for i in range(n)]


def test_fewer_degenerations_with_guards_held_is_accepted_and_runs_the_holdout(lab, monkeypatch):
    stub(lab, monkeypatch, cand=mixed(0), inc=mixed(3))
    p = run_to_done(lab, core.Program("p1", "p0", GOOD + "\n# better\n"))
    assert p.status == "accepted", p.reason
    assert [s["stage"] for s in p.stages] == ["static", "smoke", "screen", "full"]
    assert p.scores["n"] == 8 and p.scores["drop"] == 0.75 and p.scores["holdout"]["degenerate"] == 20
    assert lab.load_lab()["incumbent"] == "p1"


def test_noise_is_not_plausible_text(lab, monkeypatch):
    stub(lab, monkeypatch, cand=mixed(0, judge=4.5), inc=mixed(3))
    p = run_to_done(lab, core.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and "less plausible than real text" in p.reason


def test_stopping_early_counts_as_degenerate(lab, monkeypatch):
    stub(lab, monkeypatch, cand=mixed(0, tokens=10), inc=mixed(3))
    p = run_to_done(lab, core.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and "degenerates more" in p.reason


def test_topic_drift_and_no_gain(lab, monkeypatch):
    stub(lab, monkeypatch, cand=mixed(0, topic=0.05), inc=mixed(3))
    assert "drifts off topic" in run_to_done(lab, core.Program("p1", "p0", GOOD)).reason
    stub(lab, monkeypatch, cand=mixed(2), inc=mixed(2))
    p = run_to_done(lab, core.Program("p2", "p0", GOOD))
    assert p.status == "evaluated" and p.scores["drop"] == 0


def test_harness_pairs_draws_and_scores(tiny):  # noqa: F811
    common = ["--gen", str(tiny / "c.pt"), "--judge", str(tiny / "c.pt"), "--val", str(tiny / "v.pt")]
    r = run(tiny, "labs.decode.harness", "val", *common, "--sampler", str(DecodeLab.seed_file), "--against",
            str(DecodeLab.seed_file), "--n", "6", "--seed", "5")
    assert r["ok"], r.get("trace")
    cand, inc = r["samplers"]["cand"]["samples"], r["samplers"]["inc"]["samples"]
    assert len(cand) == 6 and [s["rep4"] for s in cand] == [s["rep4"] for s in inc]  # same draws, same samples
    assert all(s["judge_nll"] is not None and s["human_nll"] is not None for s in cand)
    bad = tiny / "bad.py"
    bad.write_text(GOOD.replace("return int(torch.multinomial(probs, 1, generator=generator))", "return -1"))
    r = run(tiny, "labs.decode.harness", "val", *common, "--sampler", str(bad), "--n", "2", "--seed", "5")
    assert not r["ok"] and "not a token id" in r["error"]
