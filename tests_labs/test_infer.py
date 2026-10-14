"""labs.infer: its cascade decisions (stubbed harness) and the real harness on a tiny CPU checkpoint."""

import json
import math

import pytest
from labs_testkit import run_to_done, seed

from labs import core
from labs.infer.lab import InferLab

GOOD = InferLab.seed_file.read_text()


def stub(lab, monkeypatch, quality=None, ratios=None):
    def h(mode, out, timeout, **args):
        if mode == "quality":
            return {"ok": True, "finite": True, "generate_error": None, "delta_loss": 0.0, "top1_agreement": 1.0,
                    "peak_bytes": 1000, "mean_loss": 3.7, **(quality or {})}
        r = ratios or {"decode_128": 0.10, "prefill_1024": 0.06}
        per = {s: [v + (0.004 if i % 2 else -0.004) for i in range(4)] for s, v in r.items()}
        return {"ok": True, "errors": [], "peak_bytes": 1000, "speed": {"cand": {s: 1.0 for s in r}},
                "scenario_round_log_ratio": per, "round_log_ratio": [sum(per[s][i] for s in per) / len(per) for i in range(4)]}
    monkeypatch.setattr(lab, "h", h)


@pytest.fixture
def lab(world):
    lab = InferLab()
    seed(lab, x_p0=1.0, speed={"decode_128": 200.0})
    return lab


def test_faster_program_is_accepted(lab, monkeypatch):
    stub(lab, monkeypatch)
    p = run_to_done(lab, core.Program("p1", "p0", GOOD + "\n# faster\n"))
    assert p.status == "accepted", p.reason
    assert lab.load_lab()["incumbent"] == "p1" and math.isclose(p.scores["x_p0"], math.exp(0.08), rel_tol=1e-6)
    assert [s["stage"] for s in p.stages] == ["static", "quality", "bench", "confirm"]


def test_quality_gate_rejects(lab, monkeypatch):
    stub(lab, monkeypatch, quality={"delta_loss": 0.02})
    p = run_to_done(lab, core.Program("p1", "p0", GOOD))
    assert p.status == "rejected" and p.reason.startswith("quality: loss")
    stub(lab, monkeypatch, quality={"generate_error": "generate token 3 = 7 is not the argmax"})
    assert run_to_done(lab, core.Program("p2", "p0", GOOD)).reason.startswith("quality: generate token 3")


def test_one_scenario_cannot_pay_for_another(lab, monkeypatch):
    stub(lab, monkeypatch, ratios={"decode_128": 0.6, "prefill_1024": -0.3})
    p = run_to_done(lab, core.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and p.reason.startswith("prefill_1024 regressed")
    assert lab.load_lab()["incumbent"] == "p0"


def test_slower_program_stops_after_bench(lab, monkeypatch):
    stub(lab, monkeypatch, ratios={"decode_128": -0.02, "prefill_1024": -0.01})
    p = run_to_done(lab, core.Program("p1", "p0", GOOD))
    assert p.status == "evaluated" and [s["stage"] for s in p.stages][-1] == "bench"


from labs_infer_tiny import run, tiny  # noqa: E402,F401 - the tiny checkpoint fixture


def test_harness_seed_matches_reference_and_catches_a_lying_generate(tiny):  # noqa: F811
    common = ["--ckpt", str(tiny / "c.pt"), "--val", str(tiny / "v.pt")]
    ref = run(tiny, "labs.infer.harness", "reference", *common)
    assert ref["ok"], ref.get("trace")
    (tiny / "ref.json").write_text(json.dumps(ref))
    q = run(tiny, "labs.infer.harness", "quality", *common, "--ref", str(tiny / "ref.json"), "--engine", str(InferLab.seed_file))
    assert q["ok"] and q["delta_loss"] == 0 and q["top1_agreement"] == 1.0 and q["generate_error"] is None, q
    liar = tiny / "liar.py"
    liar.write_text(GOOD.replace("        return torch.cat(out, dim=1)",
                                 "        return torch.zeros(1, n, dtype=torch.long, device=ids.device) + 7"))
    q = run(tiny, "labs.infer.harness", "quality", *common, "--ref", str(tiny / "ref.json"), "--engine", str(liar))
    assert q["ok"] and "not the argmax" in q["generate_error"]
    b = run(tiny, "labs.infer.harness", "bench", *common, "--engine", str(liar), "--against", str(InferLab.seed_file),
            "--rounds", "2", "--seed", "3")
    assert b["ok"] and b["errors"] and len(b["round_log_ratio"]) == 2
