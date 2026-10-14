"""labs.infer: the cascade's decisions (stubbed harness and LLM) and the real harness on a tiny CPU checkpoint."""

import json
import math
import random
import subprocess
import sys

import pytest
import torch

from labs.infer import lab

GOOD = (lab.SEED.read_text())


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LABS_STATE", tmp_path / "labs")
    monkeypatch.setattr(lab, "STATE", tmp_path / "labs" / "infer")
    monkeypatch.setattr(lab, "OWNER_RUNS", tmp_path / "owner_runs")
    cfg = tmp_path / "labs.toml"
    cfg.write_text("""[labs]\ndaily_usd = 1.0\nest_call_usd = 0.3\n[llm]\nmodel = "opus"\n[infer]
checkpoint = "c.pt"\nval = "v.pt"\nmax_delta_loss = 0.005\nmin_top1 = 0.97\nmax_mem_mult = 1.5\nbench_rounds = 4
min_gain = 0.02\naccept_se = 2.0\nmax_scenario_regression = 0.10\n""")
    monkeypatch.setattr(lab, "CONFIG", cfg)
    lab.STATE.mkdir(parents=True)
    (lab.STATE / "ref.json").write_text(json.dumps({"config": {"n_layer": 8, "n_embd": 768, "n_head": 12}}))
    p0 = lab.Program("p0", None, GOOD, stage="done", status="accepted", scores={"x_p0": 1.0, "speed": {"decode_128": 200}})
    lab.save(p0)
    lab.save_lab({"incumbent": "p0", "history": [], "p0_peak": 1000})
    return tmp_path


def fake_harness(quality=None, bench=None):
    def h(mode, out, c, timeout, **args):
        if mode == "quality":
            return {"ok": True, "finite": True, "generate_error": None, "delta_loss": 0.0, "top1_agreement": 1.0,
                    "peak_bytes": 1000, "mean_loss": 3.7, **(quality or {})}
        ratios = bench or {"decode_128": 0.10, "prefill_1024": 0.06}
        rounds = 4
        per = {s: [v + (0.004 if r % 2 else -0.004) for r in range(rounds)] for s, v in ratios.items()}
        return {"ok": True, "errors": [], "peak_bytes": 1000, "speed": {"cand": {s: 1.0 for s in ratios}},
                "log_ratio": dict(ratios), "scenario_round_log_ratio": per,
                "round_log_ratio": [sum(per[s][r] for s in per) / len(per) for r in range(rounds)]}
    return h


def run_to_done(p):
    for _ in range(6):
        if p.status in lab.DONE:
            return p
        p = lab.advance(p, log=lambda m: None)
    return p


def test_pair_stats_cancel_the_order_effect():
    m, se, n = lab.pair_stats([0.03, -0.03, 0.031, -0.029])
    assert n == 2 and abs(m - 0.0005) < 1e-9 and se < 0.001


def test_apply_edits_needs_one_exact_match():
    assert lab.apply_edits("a b c", [{"search": "b", "replace": "B"}]) == "a B c"
    with pytest.raises(ValueError, match="2 places"):
        lab.apply_edits("b b", [{"search": "b", "replace": "x"}])
    with pytest.raises(ValueError, match="0 places"):
        lab.apply_edits("a", [{"search": "z", "replace": "x"}])


def test_static_check(world):
    c = lab.cfg()
    assert lab.static_check(GOOD, c) is None
    assert "Engine" in lab.static_check("x = 1\n", c)
    assert "open" in lab.static_check(GOOD + "\nopen('/etc/passwd')\n", c)
    assert "syntax" in lab.static_check("class Engine(:\n", c)


def test_faster_program_is_accepted_and_becomes_incumbent(world, monkeypatch):
    monkeypatch.setattr(lab, "harness", fake_harness())
    p = run_to_done(lab.Program("p1", "p0", GOOD + "\n# faster\n"))
    assert p.status == "accepted", p.reason
    assert lab.load_lab()["incumbent"] == "p1" and math.isclose(p.scores["x_p0"], math.exp(0.08), rel_tol=1e-6)
    assert [s["stage"] for s in p.stages] == ["static", "quality", "bench", "confirm"]


def test_quality_gate_rejects(world, monkeypatch):
    monkeypatch.setattr(lab, "harness", fake_harness(quality={"delta_loss": 0.02}))
    p = run_to_done(lab.Program("p1", "p0", GOOD))
    assert p.status == "rejected" and p.reason.startswith("quality: loss")
    monkeypatch.setattr(lab, "harness", fake_harness(quality={"generate_error": "generate token 3 = 7 is not the argmax"}))
    assert run_to_done(lab.Program("p2", "p0", GOOD)).reason.startswith("quality: generate token 3")


def test_one_scenario_cannot_pay_for_another(world, monkeypatch):
    monkeypatch.setattr(lab, "harness", fake_harness(bench={"decode_128": 0.6, "prefill_1024": -0.3}))
    p = run_to_done(lab.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and p.reason.startswith("prefill_1024 regressed")
    assert lab.load_lab()["incumbent"] == "p0"


def test_slower_program_stops_after_bench(world, monkeypatch):
    monkeypatch.setattr(lab, "harness", fake_harness(bench={"decode_128": -0.02, "prefill_1024": -0.01}))
    p = run_to_done(lab.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and [s["stage"] for s in p.stages][-1] == "bench"


def test_propose_applies_edits_and_budget_stops_it(world, monkeypatch):
    calls = []

    def fake_call(prompt, system, schema, model, llm_cfg, log_dir, tag):
        calls.append(prompt)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(lab.LABS_STATE / "llm_spend.jsonl", "a") as fh:
            fh.write(json.dumps({"at": lab.iso(), "usd": 0.4}) + "\n")
        return {"reply": {"rationale": "fuse things", "expected": "2%",
                          "diffs": [{"search": "class Engine:", "replace": "class Engine:  # edited"}]},
                "cost_usd": 0.4, "model": "claude-test"}

    monkeypatch.setattr("autolab.llm.call", fake_call)
    p = lab.propose(log=lambda m: None, rng=random.Random(0))
    assert p.id == "p1" and "# edited" in p.code and p.status == "queued" and "## Parent p0" in calls[0]
    monkeypatch.setattr(lab, "harness", fake_harness())
    lab.save(run_to_done(p))
    assert lab.cycle(log=lambda m: None) == {"proposed": "p2"}  # $0.4 + 0.3 <= $1
    assert lab.cycle(log=lambda m: None)["advanced"] == "p2"
    while lab.programs()["p2"].status not in lab.DONE:
        lab.cycle(log=lambda m: None)
    assert "daily budget" in lab.cycle(log=lambda m: None)["waiting"]  # $0.8 + 0.3 > $1


def test_hold_stops_the_loop(world):
    lab.set_hold(True, "test")
    assert lab.cycle(log=lambda m: None) == {"waiting": f"paused by test since {lab.control()['hold']['at']}"}
    lab.set_hold(False, "test")
    assert lab.blocked() is None


# --- the real harness, on a tiny random checkpoint on the CPU --------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from labs.infer.reference import ModelConfig, ModelCustomTransformer

    d = tmp_path_factory.mktemp("tiny")
    cfg = ModelConfig(vocab_size=50257, block_size=1024, n_embd=32, n_head=2, n_layer=2, use_rope_embeddings=True,
                      fused_attention=True)
    torch.manual_seed(0)
    m = ModelCustomTransformer(cfg)
    torch.save({"config": cfg.to_dict(), "model_state_dict": m.state_dict()}, d / "c.pt")
    torch.save(torch.randint(0, 50257, (20_000,)), d / "v.pt")
    return d


def harness(tiny, *args):
    out = tiny / f"{args[0]}-{abs(hash(args))}.json"
    subprocess.run([sys.executable, "-m", "labs.infer.harness", *args, "--ckpt", str(tiny / "c.pt"), "--val",
                    str(tiny / "v.pt"), "--out", str(out)], check=True, env={"LABS_DEVICE": "cpu", "PATH": ""})
    return json.loads(out.read_text())


def test_harness_seed_matches_reference_and_catches_a_lying_generate(tiny):
    ref = harness(tiny, "reference")
    assert ref["ok"], ref.get("trace")
    (tiny / "ref.json").write_text(json.dumps(ref))
    q = harness(tiny, "quality", "--ref", str(tiny / "ref.json"), "--engine", str(lab.SEED))
    assert q["ok"] and q["delta_loss"] == 0 and q["top1_agreement"] == 1.0 and q["generate_error"] is None, q
    liar = tiny / "liar.py"
    liar.write_text(GOOD.replace("        return torch.cat(out, dim=1)", "        return torch.zeros(1, n, dtype=torch.long, device=ids.device) + 7"))
    q = harness(tiny, "quality", "--ref", str(tiny / "ref.json"), "--engine", str(liar))
    assert q["ok"] and "not the argmax" in q["generate_error"]
    b = harness(tiny, "bench", "--engine", str(liar), "--against", str(lab.SEED), "--rounds", "2", "--seed", "3")
    assert b["ok"] and b["errors"] and len(b["round_log_ratio"]) == 2


def test_dashboard_labs_api(world, monkeypatch):
    from fastapi.testclient import TestClient

    from autolab.dashboard import app

    client = TestClient(app)
    r = client.get("/api/labs").json()
    assert r["infer"]["incumbent"] == "p0" and [x["id"] for x in r["infer"]["rows"]] == ["p0"] and r["hold"] is None
    assert client.get("/api/labs/infer/program/p0").json()["diff"].startswith(GOOD[:20])  # no parent: the whole file
    assert client.get("/api/labs/infer/program/p9").status_code == 404
    assert client.post("/api/labs", json={"action": "pause"}).status_code == 403
    assert client.post("/api/labs", json={"action": "pause"}, headers={"X-Autolab": "1"}).status_code == 200
    assert client.get("/api/labs").json()["hold"]["by"] == "dashboard"
    client.post("/api/labs", json={"action": "unpause"}, headers={"X-Autolab": "1"})
    assert client.get("/api/labs").json()["hold"] is None
