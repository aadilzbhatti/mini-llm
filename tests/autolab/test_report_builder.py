"""report.json from a fabricated TB event file, and from a real tiny CPU training run."""

import json
import math

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

import mini_llm.train as train
from autolab.report import ReportParams, build_report, config_hash, linfit, spike_stats, subsample, write_report

RUN_ID = "fab1"

LOG = """Using device: mps
Config: {'vocab_size': 50257, 'block_size': 64, 'n_embd': 128, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
Model: 7,280,209 parameters
Train tokens:      20,543,855
Validation tokens:    918,728
"""


def launch(seed=42, hit="tokens"):
    return {
        "run_id": RUN_ID,
        "request": {"run_id": RUN_ID, "dataset_id": "data20k", "train_tokens": "x.pt", "seed": seed,
                    "budget": {"tokens": 7_680_000, "wall_clock_s": 600},
                    "model": {"block_size": 64, "n_embd": 128, "n_head": 4, "n_layer": 4, "dropout": 0.0},
                    "optim": {"batch_size": 4, "lr": 1e-3, "min_lr": 2e-6, "warmup_steps": 500,
                              "weight_decay": 0.0},
                    "eval": {"eval_interval": 100}, "code_src": None},
        "steps": 30000, "tokens_per_step": 256, "git": {"commit": "abc123", "dirty": False},
        "started_at": "2026-09-26T10:00:00+00:00", "ended_at": "2026-09-26T10:10:00+00:00",
        "wall_s": 600.0, "gpu_wait_s": 0.0, "budget_hit": hit, "status": "finished",
    }


def fabricate(run_dir, last_step=29_999, nan_at=None):
    tb = run_dir / "runs" / "tb" / RUN_ID
    w = SummaryWriter(log_dir=str(tb))
    steps = list(range(0, last_step, 10)) + [last_step]
    for s in steps:
        value = 4 + 2 * math.exp(-s / 5000) + (0.8 if s == 15_000 else 0.0)
        if nan_at is not None and s == nan_at:
            value = float("nan")
        w.add_scalar("train/batch_loss", value, s, walltime=1000 + s * 0.02)
        w.add_scalar("train/lr", 1e-3 * (1 - s / 30_000), s, walltime=1000 + s * 0.02)
        w.add_scalar("train/grad_norm", 2.0 + (10.0 if s == 15_000 else 0.0), s, walltime=1000 + s * 0.02)
    for s in list(range(0, last_step, 100)) + [last_step]:
        w.add_scalar("eval/train_loss", 4 + 2 * math.exp(-s / 5000), s)
        w.add_scalar("eval/val_loss", 4.1 + 2 * math.exp(-s / 5000), s)
    w.add_scalar("eval/full_val_loss", 4.12, last_step)
    w.close()
    (run_dir / "train.log").write_text(LOG)


def test_report_from_fabricated_events(tmp_path):
    fabricate(tmp_path)
    (tmp_path / "launch.json").write_text(json.dumps(launch()))
    r = build_report(tmp_path)

    ident, scale, summ = r["identity"], r["scale"], r["summary"]
    assert ident["run_id"] == RUN_ID and ident["git_commit"] == "abc123" and ident["seed"] == 42
    assert ident["config"]["vocab_size"] == 50257 and ident["config"]["n_embd"] == 128
    assert scale["params"] == 7_280_209
    assert scale["embedding_params"] == (50257 + 64) * 128
    assert scale["non_embedding_params"] == 7_280_209 - (50257 + 64) * 128
    assert scale["tokens_seen"] == 30_000 * 256 and scale["steps_done"] == 30_000
    assert scale["epochs"] == pytest.approx(7_680_000 / 20_543_855, rel=1e-5)
    assert scale["tokens_per_param"] == pytest.approx(7_680_000 / 7_280_209, rel=1e-5)

    assert len(r["curves"]["train_loss"]) == 200 and len(r["curves"]["val_loss"]) == 200
    assert r["curves"]["val_loss"][0][0] == 0 and r["curves"]["val_loss"][-1][0] == 29_999
    assert summ["final_full_val_loss"] == pytest.approx(4.12)
    assert summ["gap"] == pytest.approx(0.1, abs=1e-4)
    assert summ["best_val_step"] == 29_999
    assert summ["val_slope_tail"]["slope_per_1k_steps"] < 0
    assert summ["gap_trend"]["change"] == pytest.approx(0.0, abs=1e-5)
    assert summ["lr_peak"] == pytest.approx(1e-3)

    h = r["health"]
    assert h["nan_or_inf"] is False
    assert h["spikes"]["count"] >= 1 and 15_000 in h["spikes"]["steps"]
    assert h["grad_norm"]["median"] == pytest.approx(2.0) and h["grad_norm"]["max"] == pytest.approx(12.0)

    perf = r["performance"]
    assert perf["train_wall_s"] == pytest.approx(29_999 * 0.02, rel=1e-3)
    assert perf["tokens_per_sec"] == pytest.approx(256 / 0.02, rel=1e-3)
    assert perf["device"] == "mps"
    assert r["budget"] == {"type": "tokens", "tokens": 7_680_000, "wall_clock_s": 600, "hit": "tokens"}

    path = write_report(r, tmp_path)
    assert json.loads(path.read_text())["identity"]["run_id"] == RUN_ID


def test_nan_is_flagged_and_serializes(tmp_path):
    fabricate(tmp_path, last_step=2_000, nan_at=1_000)
    (tmp_path / "launch.json").write_text(json.dumps(launch()))
    r = build_report(tmp_path)
    assert r["health"]["nan_or_inf"] is True
    assert r["health"]["non_finite_counts"] == {"train/batch_loss": 1}
    json.dumps(r, allow_nan=False)  # NaN never leaks into report.json


def test_config_hash_ignores_seed_only():
    base = {"n_embd": 128, "lr": 1e-3, "seed": 1}
    assert config_hash(base) == config_hash({**base, "seed": 2})
    assert config_hash(base) != config_hash({**base, "lr": 3e-4})


def test_helpers():
    series = [(i, float(i)) for i in range(1000)]
    sub = subsample(series, 200)
    assert len(sub) == 200 and sub[0] == (0, 0.0) and sub[-1] == (999, 999.0)
    fit = linfit([(0, 1.0), (1000, 0.5), (2000, 0.0)])
    assert fit["slope_per_1k_steps"] == pytest.approx(-0.5) and fit["change"] == pytest.approx(-1.0)
    flat = [(i, 5.0) for i in range(100)]
    assert spike_stats(flat, 21, 3.0)["count"] == 0


# --- a real (tiny) training run through mini_llm.train, forced onto CPU ----------------


class StubTokenizer:
    def __len__(self):
        return 64


def test_report_from_real_tiny_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("MINI_LLM_RUN_ID", RUN_ID)
    g = torch.Generator().manual_seed(0)
    torch.save(torch.randint(0, 64, (4000,), generator=g), tmp_path / "train.pt")
    torch.save(torch.randint(0, 64, (600,), generator=g), tmp_path / "val.pt")

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train.main(["--tokens", "train.pt", "--val-tokens", "val.pt", "--block-size", "8", "--n-embd", "16",
                    "--n-head", "2", "--n-layer", "1", "--eval-interval", "10", "--eval-batches", "2",
                    "--full-eval-interval", "0", "--log-interval", "5", "--warmup-steps", "0", "--steps", "60"])
    (tmp_path / "train.log").write_text(buf.getvalue())
    lj = launch()
    lj["request"]["model"] = {"block_size": 8, "n_embd": 16, "n_head": 2, "n_layer": 1, "dropout": 0.0}
    lj["tokens_per_step"], lj["steps"] = 4 * 8, 60
    (tmp_path / "launch.json").write_text(json.dumps(lj))

    r = build_report(tmp_path, ReportParams())
    assert r["performance"]["device"] == "cpu"
    assert r["scale"]["tokens_seen"] == 60 * 32 and r["scale"]["dataset_tokens"] == 4000
    assert r["scale"]["params"] > 0 and r["scale"]["embedding_params"] == (64 + 8) * 16
    assert r["health"]["grad_norm"]["n"] == 12  # steps 0,5,...,55 (the last step 59 isn't a log step)
    assert r["health"]["grad_norm"]["median"] > 0
    assert len(r["curves"]["val_loss"]) == 7   # steps 0,10,...,50 and the final 59
    assert r["summary"]["final_full_val_loss"] is not None
