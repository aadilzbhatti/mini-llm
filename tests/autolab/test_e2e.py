"""End to end on CPU (HANDOFF M5): start -> proposals -> cascade -> stop -> resume -> finish.

Real controller, real cascade (static/CPU gates with a protected test, params), real training
of a tiny model on slices of data20k and the frozen val. Only the two external services are
replaced: Claude (a fake proposer through the real generate_one) and Modal (a local submit
that trains synchronously on CPU and records the call as finished).
"""

import json
import os
import random
import subprocess

import pytest
import torch

from autolab import config as acfg
from autolab import controller as ctl
from autolab import evaluate as ev
from autolab import modal_backend as mb
from autolab.config import REPO_ROOT, sha256_file
from autolab.diagnose import History, diagnose
from autolab.generate import generate_one
from autolab.report import build_report, write_report
from autolab.trainer import execute

pytestmark = pytest.mark.skipif(os.environ.get("AUTOLAB_IN_CASCADE") == "1", reason="inside a cascade run")

HP = {"n_embd": 32, "n_head": 2, "n_layer": 1, "dropout": 0.0, "batch_size": 4, "lr": 3e-3,
      "min_lr": 1e-4, "warmup_steps": 2, "weight_decay": 0.0}


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    real = acfg.load_config()
    data = tmp_path / "datasets" / "tiny"
    data.mkdir(parents=True)
    torch.save(torch.load(real.datasets_dir / "data20k" / "train.pt")[:60_000].clone(), data / "train.pt")
    val = tmp_path / "val.pt"
    torch.save(torch.load(real.frozen_val)[:3_000].clone(), val)

    class Cfg:
        datasets_dir = tmp_path / "datasets"
        frozen_val = val
        frozen_val_sha256 = sha256_file(val)
    monkeypatch.setattr(acfg, "load_config", lambda: Cfg)

    cfg = ev.evolve_cfg()
    cfg.update(block_size=32, dataset_id="tiny", gpu="L4", screen_margin=10.0, cpu_tests=[
        "tests/autolab/test_causal_leak.py"], cpu_test_deselect=[], confirm_seeds=[2])
    cfg["hparams"] = {**cfg["hparams"], "n_embd": {"choices": [32, 48]}, "n_head": {"choices": [2]},
                      "batch_size": {"choices": [4, 8]}}
    monkeypatch.setattr(ev, "evolve_cfg", lambda: cfg)

    root = tmp_path / "state" / "evolve"
    runs = tmp_path / "runs"
    monkeypatch.setattr(ev, "STATE_ROOT", root)
    for name, value in (("STATE", tmp_path / "state"), ("CONTROL", tmp_path / "state" / "controller.json"),
                        ("NOTEBOOK", tmp_path / "notebook.jsonl"), ("NOTEBOOK_MD", tmp_path / "NOTEBOOK.md"),
                        ("RUNS", runs)):
        monkeypatch.setattr(ctl, name, value)
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 2, "daily_usd": 100.0})
    committed = []
    monkeypatch.setattr(ctl, "commit_accepted", lambda p, s, log=print: committed.append(p.id) or "test")
    calls = {}
    monkeypatch.setattr(mb, "load_calls", lambda: calls)
    monkeypatch.setattr(mb, "price_per_s", lambda gpu: 0.000222)

    def local_submit(req, gpu, src):
        """Modal stand-in: train now, on CPU, with the program's code on PYTHONPATH."""
        run_dir = runs / req.run_id
        run_dir.mkdir(parents=True)
        execute(req, run_dir, data / "train.pt", val, {"backend": "local-test"},
                env_extra={"PYTHONPATH": str(src), "AUTOLAB_FORCE_CPU": "1"})
        report = build_report(run_dir)
        write_report(report, run_dir)
        (run_dir / "diagnosis.json").write_text(json.dumps(diagnose(report, History()).to_dict()))
        calls[req.run_id] = {"state": "finished", "usd": 0.01, "usd_estimate": 0.01, "gpu": gpu,
                             "submitted_at": ctl.iso(ctl.now())}

    # p0's baseline: 2 seeds per budget, trained for real
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    from autolab.program import Program, base_sources, extract_blocks, materialize
    from autolab.trainer import Budget, TrainRequest

    src = materialize(base_sources(REPO_ROOT, head), tmp_path / "p0code")
    budgets = {"screen_tokens": 2_048, "full_tokens": 4_096, "eval": {"eval_interval": 4, "eval_batches": 2,
               "eval_seed": 1234, "log_interval": 2, "full_eval_interval": 0, "control_poll": 5}}
    base = {"screen": [], "full": []}
    for kind in ("screen", "full"):
        for seed in (1, 2):
            req = TrainRequest(run_id=f"base-{kind}-s{seed}", dataset_id="tiny", train_tokens=str(data / "train.pt"),
                               budget=Budget(tokens=budgets[f"{kind}_tokens"], wall_clock_s=600), seed=seed,
                               model={"block_size": 32, **{k: HP[k] for k in ev.MODEL_KEYS}},
                               optim={k: HP[k] for k in ev.OPTIM_KEYS}, eval=dict(budgets["eval"]))
            local_submit(req, "L4", src)
            base[kind].append(req.run_id)
    paths = ev.Paths(root / "e2e")
    ev.init_session(head, HP, base, cfg=cfg, paths=paths, runs_dir=runs, name="e2e", budgets=budgets, dataset_id="tiny")
    ev.set_active_session("e2e", root)

    rng = random.Random(0)

    def fake_claude(prompt, system, schema, model, lcfg, log_dir, tag):
        assert "# Current program" in prompt
        lr = round(rng.choice([2e-3, 4e-3, 5e-3]), 6)
        return {"reply": {"rationale": f"try lr {lr}", "expected_effect": "small", "diffs": [], "hparams": {"lr": lr}},
                "cost_usd": 0.0, "model": "fake-claude", "duration_s": 0.0, "log": "none"}

    def generate(log):
        return generate_one(paths=ev.Paths(), rng=rng, log=log, caller=fake_claude, model="opus", runs_dir=runs)

    def cycle():
        """One daemon cycle: cascade pass (local submit) + controller step."""
        for p in ev.programs(ev.Paths()).values():
            if p.status not in ev.DONE:
                ev.advance(p, ev.load_session(ev.Paths()), cfg, calls, paths=ev.Paths(), runs_dir=runs,
                           submit=local_submit, log=lambda m: None)
        return ctl.step(log=lambda m: None, generate=generate)

    return {"cycle": cycle, "calls": calls, "runs": runs, "tmp": tmp_path, "committed": committed}


def test_start_stop_resume_finish(e2e):
    from autolab.cli import control_main

    class A:
        cancel_running = False

    def cli(cmd):
        a = A()
        a.cmd = cmd
        control_main(a)

    cli("start")
    s = e2e["cycle"]()
    assert s["enabled"] and s["proposed"] == 2
    e2e["cycle"]()  # both children go static -> cpu -> params -> screen -> full (-> confirm), trained on CPU
    cli("stop")
    for _ in range(4):
        s = e2e["cycle"]()
    assert s["enabled"] is False and "proposed" not in s  # stopped: nothing new
    progs = ev.programs()
    assert len(progs) == 3 and all(p.status in ev.DONE for p in progs.values()), \
        {p.id: (p.stage, p.status, p.reason) for p in progs.values()}
    cli("resume")
    e2e["cycle"]()
    for _ in range(4):
        e2e["cycle"]()
    cli("stop")
    for _ in range(3):  # let anything proposed on the last cycle finish
        e2e["cycle"]()
    progs = ev.programs()
    children = [p for p in progs.values() if p.parent_id]
    assert len(children) > 2 and all(p.status in ev.DONE for p in children), \
        {p.id: (p.stage, p.status) for p in children}

    # consistency: every child's runs exist, were recorded as calls, and have reports
    for p in children:
        for stage, ids in p.runs.items():
            for rid in ids:
                assert e2e["calls"][rid]["state"] == "finished"
                assert (e2e["runs"] / rid / "report.json").exists()
        assert p.stages[0]["stage"] == "static"
        # a reply that repeats the parent's lr "changes nothing" and falls back to mutation
        assert p.created_by == "fake-claude" or (p.created_by == "mutation" and p.meta["fallback_reason"])
    notebook = [json.loads(x) for x in (e2e["tmp"] / "notebook.jsonl").read_text().splitlines()]
    events = [e["event"] for e in notebook]
    assert events.count("proposed") == len(children) == events.count("program_done")
    accepted = [p.id for p in children if p.status == "accepted"]
    assert e2e["committed"] == accepted  # every acceptance committed exactly once
    assert ev.load_session()["incumbent"] == (accepted[-1] if accepted else "p0")
    assert events[0] == "started" and "stopped" in events and "resumed" in events
    assert {e["program"] for e in notebook if e["event"] == "program_done"} == {p.id for p in children}
    md = (e2e["tmp"] / "NOTEBOOK.md").read_text()
    assert all(f"| {p.id} |" in md for p in children)
