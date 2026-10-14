"""labs.core: the loop every lab shares, exercised through labs.infer with a stubbed harness and LLM."""

import json
import random

import pytest
from labs_testkit import run_to_done, seed, write_json

from labs import core
from labs.infer.lab import InferLab


def test_pair_stats_cancel_the_order_effect():
    m, se, n = core.pair_stats([0.03, -0.03, 0.031, -0.029])
    assert n == 2 and abs(m - 0.0005) < 1e-9 and se < 0.001


def test_apply_edits_needs_one_exact_match():
    assert core.apply_edits("a b c", [{"search": "b", "replace": "B"}]) == "a B c"
    with pytest.raises(ValueError, match="2 places"):
        core.apply_edits("b b", [{"search": "b", "replace": "x"}])
    with pytest.raises(ValueError, match="0 places"):
        core.apply_edits("a", [{"search": "z", "replace": "x"}])


def test_static_check():
    good = InferLab.seed_file.read_text()
    assert core.static_check(good, "Engine", 40_000) is None
    assert "Engine" in core.static_check("x = 1\n", "Engine", 40_000)
    assert "open" in core.static_check(good + "\nopen('/etc/passwd')\n", "Engine", 40_000)
    assert "syntax" in core.static_check("class Engine(:\n", "Engine", 40_000)
    assert "bytes" in core.static_check(good, "Engine", 10)


def stub_infer(lab, monkeypatch):
    def h(mode, out, timeout, **args):
        if mode == "quality":
            return {"ok": True, "finite": True, "generate_error": None, "delta_loss": 0.0, "top1_agreement": 1.0,
                    "peak_bytes": 1000, "mean_loss": 3.7}
        per = {"decode_128": [0.1, 0.1, 0.1, 0.1], "prefill_1024": [0.06] * 4}
        return {"ok": True, "errors": [], "peak_bytes": 1000, "speed": {"cand": {"decode_128": 1.0}},
                "scenario_round_log_ratio": per, "round_log_ratio": [0.08, 0.079, 0.081, 0.08]}
    monkeypatch.setattr(lab, "h", h)


def test_propose_cycle_and_budget(world, monkeypatch):
    lab = InferLab()
    seed(lab, x_p0=1.0, speed={"decode_128": 200.0})
    write_json(lab.state / "ref.json", {"config": {"n_layer": 8, "n_embd": 768, "n_head": 12}})
    stub_infer(lab, monkeypatch)
    prompts = []

    def fake_call(prompt, system, schema, model, llm_cfg, log_dir, tag):
        prompts.append(prompt)
        core.LABS_STATE.mkdir(parents=True, exist_ok=True)
        with open(core.LABS_STATE / "llm_spend.jsonl", "a") as fh:
            fh.write(json.dumps({"at": core.iso(), "usd": 0.4}) + "\n")
        return {"reply": {"rationale": "fuse things", "expected": "2%",
                          "diffs": [{"search": "class Engine:", "replace": "class Engine:  # edited"}]},
                "cost_usd": 0.4, "model": "claude-test"}

    monkeypatch.setattr("autolab.llm.call", fake_call)
    p = lab.propose(log=lambda m: None, rng=random.Random(0))
    assert p.id == "p1" and "# edited" in p.code and p.status == "queued" and "## Parent p0" in prompts[0]
    assert "## Task" in prompts[0] and "engine.py faster" in prompts[0]
    p = run_to_done(lab, p)
    assert p.status == "accepted" and lab.load_lab()["incumbent"] == "p1"
    assert lab.cycle(log=lambda m: None) == {"proposed": "p2"}  # $0.4 + 0.3 <= $1
    while lab.programs()["p2"].status not in core.DONE:
        assert lab.cycle(log=lambda m: None)["advanced"] == "p2"
    assert "daily budget" in lab.cycle(log=lambda m: None)["waiting"]  # $0.8 + 0.3 > $1


def test_bad_edits_are_rejected_at_static(world, monkeypatch):
    lab = InferLab()
    seed(lab, x_p0=1.0)
    write_json(lab.state / "ref.json", {"config": {"n_layer": 8, "n_embd": 768, "n_head": 12}})
    monkeypatch.setattr("autolab.llm.call", lambda *a, **k: {
        "reply": {"rationale": "r", "expected": "e", "diffs": [{"search": "nope", "replace": "x"}]},
        "cost_usd": 0.1, "model": "m"})
    p = lab.propose(log=lambda m: None, rng=random.Random(0))
    assert p.status == "rejected" and "SEARCH matches 0 places" in p.reason


def test_hold_and_owner_runs_stop_the_loop(world):
    lab = InferLab()
    seed(lab, x_p0=1.0)
    core.set_hold(True, "test")
    assert lab.cycle(log=lambda m: None)["waiting"].startswith("paused by test")
    core.set_hold(False, "test")
    assert lab.blocked() is None
    write_json(core.OWNER_RUNS / "r.live.json", {"updated": core.iso(), "finished": False})
    assert lab.blocked() == "the owner is training on this GPU"


def test_daemon_turn_isolates_failures(world, monkeypatch):
    from labs import daemon
    from labs.decode.lab import DecodeLab

    class Boom(DecodeLab):
        def cycle(self, log=print):
            raise RuntimeError("boom")

    class Fine(InferLab):
        def cycle(self, log=print):
            return {"waiting": "x"}

    monkeypatch.setattr(daemon, "all_labs", lambda: [Fine(), Boom()])
    out = daemon.turn(log=lambda m: None)
    assert out["infer"] == {"waiting": "x"} and out["decode"]["error"] == "RuntimeError: boom"


def test_dashboard_labs_api(world):
    from fastapi.testclient import TestClient

    from autolab.dashboard import app

    lab = InferLab()
    seed(lab, x_p0=1.0, speed={"decode_128": 200.0})
    client = TestClient(app)
    r = client.get("/api/labs").json()
    infer = next(x for x in r["labs"] if x["name"] == "infer")
    assert infer["incumbent"] == "p0" and [x["id"] for x in infer["rows"]] == ["p0"] and r["hold"] is None
    assert infer["tiles"][0]["value"] == "p0"
    d = client.get("/api/labs/infer/program/p0").json()
    assert d["diff"].startswith('"""p0 of labs.infer') and d["file_name"] == "engine.py"
    assert client.get("/api/labs/infer/program/p9").status_code == 404
    assert client.get("/api/labs/nope/program/p0").status_code == 404
    assert client.post("/api/labs", json={"action": "pause"}).status_code == 403
    assert client.post("/api/labs", json={"action": "pause"}, headers={"X-Autolab": "1"}).status_code == 200
    assert client.get("/api/labs").json()["hold"]["by"] == "dashboard"
    client.post("/api/labs", json={"action": "unpause"}, headers={"X-Autolab": "1"})
    assert client.get("/api/labs").json()["hold"] is None
