"""The evaluation cascade: hand-written bad patches must be rejected at the right gate.

The CPU stages run for real (subprocess pytest against the materialized program);
GPU stages are driven with a fake submit and fabricated reports. Skipped inside a
cascade's own test run (AUTOLAB_IN_CASCADE), which would otherwise recurse.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from autolab import evaluate as ev
from autolab.config import REPO_ROOT
from autolab.program import Program, parse_diff

pytestmark = pytest.mark.skipif(os.environ.get("AUTOLAB_IN_CASCADE") == "1", reason="inside a cascade run")

PATCHES = REPO_ROOT / "autolab" / "experiments" / "patches"
HP = {"n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.0, "batch_size": 64, "lr": 1.2e-3,
      "min_lr": 2e-6, "warmup_steps": 100, "weight_decay": 0.0}


INITIAL_PARAMS = 16_105_297  # emb256/L4/blk128, vocab 50257 (asserted below)


def fake_report(loss, tps=40_000.0, nan=False, params=INITIAL_PARAMS, wall=150.0):
    return {"summary": {"final_full_val_loss": loss}, "health": {"nan_or_inf": nan},
            "performance": {"tokens_per_sec": tps, "wall_s": wall}, "scale": {"params": params}}


@pytest.fixture
def lab(tmp_path):
    paths = ev.Paths(tmp_path / "evolve")
    runs = tmp_path / "runs"
    for i, (screen, full) in enumerate([(6.52, 5.39), (6.56, 5.39), (6.51, 5.32)], 1):
        for kind, loss in (("screen", screen), ("full", full)):
            (runs / f"{kind}{i}").mkdir(parents=True)
            (runs / f"{kind}{i}" / "report.json").write_text(json.dumps(fake_report(loss)))
    cfg = ev.evolve_cfg()
    cfg["cpu_tests"] = ["tests/autolab/test_causal_leak.py", "tests/test_model.py"]
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    session = ev.init_session(head, HP, {"screen": ["screen1", "screen2", "screen3"], "full": ["full1", "full2", "full3"]},
                              cfg=cfg, paths=paths, runs_dir=runs)
    submitted = []

    def submit(req, gpu, src):
        assert (src / "mini_llm" / "model.py").exists()
        submitted.append(req)

    def step(p, calls=None):
        return ev.advance(p, ev.load_session(paths), cfg, calls or {}, paths=paths, runs_dir=runs,
                          submit=submit, log=lambda m: None)

    return {"paths": paths, "runs": runs, "cfg": cfg, "session": session, "submitted": submitted, "step": step}


def propose(lab, patch, hparams=None):
    return ev.propose("p0", parse_diff((PATCHES / patch).read_text()), hparams, rationale=patch, paths=lab["paths"])


def test_session_from_baseline_runs(lab):
    s = lab["session"]
    assert s["incumbent"] == "p0" and s["noise"]["full"]["n"] == 3
    assert s["noise"]["full"]["std"] == pytest.approx(0.0404, abs=1e-3)
    assert s["wall_caps"] == {"screen": 187.5, "full": 187.5}
    from mini_llm.config import ModelConfig, build_model

    m = build_model(ModelConfig(vocab_size=50257, block_size=128, n_embd=256, n_head=4, n_layer=4))
    assert sum(x.numel() for x in m.parameters()) == INITIAL_PARAMS
    p0 = ev.programs(lab["paths"])["p0"]
    assert p0.scores["screen_loss"] == pytest.approx((6.52 + 6.56 + 6.51) / 3)
    assert p0.status == "accepted" and p0.scores["full_mean"] == pytest.approx((5.39 + 5.39 + 5.32) / 3)


@pytest.mark.parametrize("patch,stage,needle", [
    ("bad_outside_region.txt", "static", "not entirely inside one EVOLVE block"),
    ("bad_search_not_found.txt", "static", "SEARCH text not found"),
    ("bad_label_cheat.txt", "static", "forbidden name 'targets'"),
])
def test_static_rejections(lab, patch, stage, needle):
    p = propose(lab, patch)
    if p.status != "rejected":  # static rules run in advance(), scope at propose()
        p = lab["step"](p)
    assert (p.status, p.stage) == ("rejected", stage) and needle in p.reason, p.reason
    assert not lab["submitted"]


@pytest.mark.parametrize("patch,stage,needle", [
    ("bad_causal_leak.txt", "cpu", "causal-leak:"),
    ("bad_shape.txt", "cpu", "shape:"),
    ("bad_param_blowup.txt", "params", "parameters > cap"),
])
def test_runtime_rejections(lab, patch, stage, needle):
    p = lab["step"](propose(lab, patch))
    assert (p.status, p.stage) == ("rejected", stage) and needle in p.reason, p.reason
    assert not lab["submitted"]
    assert not lab["paths"].work(p.id).exists()  # cleaned up


def test_hparams_out_of_range(lab):
    p = lab["step"](ev.propose("p0", [], {"lr": 0.5}, paths=lab["paths"]))
    assert (p.status, p.stage) == ("rejected", "static") and "lr=0.5 outside" in p.reason


def test_good_patch_passes_cpu_gates_and_submits_screen(lab):
    p = lab["step"](propose(lab, "good_rmsnorm.txt"))
    assert (p.stage, p.status) == ("screen", "running"), p.reason
    assert [s["stage"] for s in p.stages] == ["static", "cpu", "params", ]
    assert all(s["ok"] for s in p.stages)
    req = lab["submitted"][0]
    assert req.run_id == f"ev-{p.id}-screen-s1" and req.budget.wall_clock_s == 187.5
    assert req.model["block_size"] == 128 and req.optim["lr"] == 1.2e-3


# --- GPU stages with fabricated results ----------------------------------------------------


def gpu_program(lab):
    p = ev.propose("p0", [], {"lr": 1.5e-3}, paths=lab["paths"])
    p.stage, p.status = "screen", "queued"  # skip the (already tested) CPU stages
    return p


def finish(lab, p, stage, losses, **kw):
    calls = {}
    for rid, loss in zip(p.runs[stage], losses):
        (lab["runs"] / rid).mkdir(parents=True, exist_ok=True)
        (lab["runs"] / rid / "report.json").write_text(json.dumps(fake_report(loss, **kw)))
        calls[rid] = {"state": "finished"}
    return lab["step"](p, calls)


def test_screen_margin_rejects(lab):
    p = lab["step"](gpu_program(lab))
    p = finish(lab, p, "screen", [6.70])  # incumbent screen mean 6.53 + 0.10 margin
    assert (p.status, p.stage) == ("rejected", "screen") and "margin" in p.reason


def test_smoke_rejects_nan_and_slow(lab):
    p = finish(lab, lab["step"](gpu_program(lab)), "screen", [6.5], nan=True)
    assert p.status == "rejected" and "smoke: NaN" in p.reason
    p = finish(lab, lab["step"](gpu_program(lab)), "screen", [6.5], tps=10_000.0)
    assert p.status == "rejected" and "throughput" in p.reason


def test_full_not_better_is_evaluated(lab):
    p = finish(lab, lab["step"](gpu_program(lab)), "screen", [6.55])
    assert (p.stage, p.status) == ("full", "running")
    p = finish(lab, p, "full", [5.40])
    assert p.status == "evaluated" and p.scores["full_mean"] == 5.40


def test_confirm_accepts_only_beyond_noise(lab):
    p = finish(lab, lab["step"](gpu_program(lab)), "screen", [6.40])
    p = finish(lab, p, "full", [5.30])
    assert (p.stage, p.status) == ("confirm", "running") and len(p.runs["confirm"]) == 2
    p = finish(lab, p, "confirm", [5.31, 5.33])  # mean 5.313 > bar 5.367 - 2*0.040 = 5.286
    assert p.status == "contender" and ev.load_session(lab["paths"])["incumbent"] == "p0"

    q = finish(lab, lab["step"](gpu_program(lab)), "screen", [6.30])
    q = finish(lab, q, "full", [5.20])
    q = finish(lab, q, "confirm", [5.22, 5.21])
    assert q.status == "accepted" and ev.load_session(lab["paths"])["incumbent"] == q.id


def test_failed_run_rejects_and_cost_cap_blocks(lab):
    p = lab["step"](gpu_program(lab))
    p = lab["step"](p, {p.runs["screen"][0]: {"state": "failed", "error": "RuntimeError: boom"}})
    assert p.status == "rejected" and "boom" in p.reason

    def capped(req, gpu, src):
        raise RuntimeError("cost cap: spent $25")

    q = gpu_program(lab)
    q = ev.advance(q, ev.load_session(lab["paths"]), lab["cfg"], {}, paths=lab["paths"], runs_dir=lab["runs"],
                   submit=capped, log=lambda m: None)
    assert q.status == "blocked" and "cost cap" in q.reason
    q = lab["step"](q)  # cap lifted: retried
    assert q.status == "running"
