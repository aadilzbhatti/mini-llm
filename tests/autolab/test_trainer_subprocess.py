"""run_training end to end: a tiny model in a CPU subprocess, then report + diagnosis."""

import json
from dataclasses import replace

import torch

from autolab.config import load_config, sha256_file
from autolab.diagnose import History, diagnose
from autolab.report import build_report, write_report
from autolab.trainer import Budget, TrainRequest, run_training

TINY = {"block_size": 8, "n_embd": 16, "n_head": 2, "n_layer": 1, "dropout": 0.0}
EVAL = {"eval_interval": 10, "eval_batches": 2, "eval_seed": 1234, "log_interval": 5,
        "full_eval_interval": 0, "control_poll": 5}


def setup(tmp_path):
    g = torch.Generator().manual_seed(0)
    torch.save(torch.randint(0, 50257, (20_000,), generator=g), tmp_path / "train.pt")
    torch.save(torch.randint(0, 50257, (2_000,), generator=g), tmp_path / "val.pt")
    return replace(load_config(), frozen_val=tmp_path / "val.pt",
                   frozen_val_sha256=sha256_file(tmp_path / "val.pt"), runs_dir=tmp_path / "runs")


def request(tmp_path, run_id, budget):
    return TrainRequest(run_id=run_id, dataset_id="tiny", train_tokens=str(tmp_path / "train.pt"),
                        budget=budget, model=dict(TINY), eval=dict(EVAL),
                        optim={"batch_size": 4, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 0,
                               "weight_decay": 0.0})


def test_token_budget_run(tmp_path):
    cfg = setup(tmp_path)
    run_dir = run_training(request(tmp_path, "tok", Budget(tokens=40 * 32)), cfg, wait_gpu=False,
                           env_extra={"AUTOLAB_FORCE_CPU": "1"}, log=lambda m: None)
    launch = json.loads((run_dir / "launch.json").read_text())
    assert launch["status"] == "finished", (run_dir / "train.log").read_text()[-2000:]
    assert launch["budget_hit"] == "tokens" and launch["steps"] == 40

    report = build_report(run_dir)
    write_report(report, run_dir)
    assert report["performance"]["device"] == "cpu"
    assert report["scale"]["steps_done"] == 40 and report["scale"]["tokens_seen"] == 40 * 32
    assert report["identity"]["dataset_id"] == "tiny" and report["identity"]["git_commit"]
    assert report["summary"]["final_full_val_loss"] is not None
    assert report["health"]["grad_norm"]["n"] >= 7
    assert diagnose(report, History()).labels  # always at least one label


def test_wall_clock_cap_stops_run(tmp_path):
    cfg = setup(tmp_path)
    run_dir = run_training(request(tmp_path, "wall", Budget(tokens=10_000_000 * 32, wall_clock_s=8)), cfg,
                           wait_gpu=False, env_extra={"AUTOLAB_FORCE_CPU": "1"}, log=lambda m: None,
                           poll_s=0.2)
    launch = json.loads((run_dir / "launch.json").read_text())
    assert launch["status"] == "finished" and launch["budget_hit"] == "wall_clock"
    report = build_report(run_dir)
    assert report["budget"]["hit"] == "wall_clock"
    assert 0 < report["scale"]["steps_done"] < 10_000_000
    assert report["summary"]["final_full_val_loss"] is not None  # final eval still ran after the stop
