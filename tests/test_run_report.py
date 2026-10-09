"""Shareable run reports (mini_llm.run_report): one md/html/pdf per evaluated run, served by the control API."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import mini_llm.run_report as rr
from mini_llm.samples import label
from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent
STEM = "modal_data640k-fw70edu30-b8-t1024-e768h12l8-160k_steps160000_seed42"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8ffff3f0005fe02fe"
    "a7d6a4d50000000049454e44ae426082"
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "runner").mkdir()
    shutil.copy(REPO / "runner" / "run_queue.py", tmp_path / "runner" / "run_queue.py")
    for d in ("runs", "queue", "evals", "plots", "checkpoints", "data/data640k-fw70edu30"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "data/data640k-fw70edu30/MANIFEST.md").write_text(
        "# data640k-fw70edu30\n\n| train.pt | 633,453,440 |\n"
    )
    (tmp_path / "data/data640k-fw70edu30/mix.json").write_text(json.dumps({"train_tokens": 633453440}))
    (tmp_path / "plots/loss.png").write_bytes(PNG)
    (tmp_path / "baselines.json").write_text(
        json.dumps([{"checkpoint": f"checkpoints/{STEM}.pt", "plot": "plots/loss.png"}])
    )
    status = {
        "run_id": "r1",
        "name": "mix-160k",
        "status": "completed",
        "started": "2026-10-09T07:32:53Z",
        "finished": "2026-10-09T13:33:34Z",
        "duration_sec": 21640.2,
        "args": {
            "tokens": "data/data640k-fw70edu30/train.pt",
            "val-tokens": "data/x/val.pt",
            "batch-size": 8,
            "lr": 3e-4,
            "min-lr": 2e-6,
            "seed": 42,
        },
        "metrics": {
            "full_val_curve": [[0, 10.8], [5000, 4.52], [160000, 3.4038]],
            "eval_train_loss": 3.4147,
            "eval_val_loss": 3.4139,
        },
        "remote": {
            "provider": "modal",
            "gpus": "H100",
            "git_sha": "a1d6343abc",
            "cost": {"billed_usd": 23.25, "final": True},
        },
        "eval": {"state": "done", "report": STEM},
    }
    (tmp_path / "runs/r1.status.json").write_text(json.dumps(status))
    (tmp_path / "runs/r1.log").write_text(
        "step     0 | eval_train_loss 10.8275 | eval_val_loss 10.8264\nstep  2000 | eval_train_loss 5.3389 | eval_val_loss 5.2654\n"
    )
    draw = {
        "text": " in 1789 the people rose up.",
        "tokens": 9,
        "eos": True,
        "rep4": 0.1,
        "looped": False,
        "topic": 1.0,
    }
    ev = {
        "checkpoint": str(tmp_path / f"checkpoints/{STEM}.pt"),
        "config": {
            "vocab_size": 50257,
            "block_size": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 8,
            "dropout": 0.0,
            "use_rope_embeddings": True,
            "fused_attention": True,
        },
        "params": 95333713,
        "step": 160000,
        "val_tokens": "data/data20k/val.pt",
        "quality": {"full_val@1024": 3.4038, "by_position@1024": {"0-15": 4.36, "512-1023": 3.27}},
        "context_curve": {
            "protocol": "8000 fixed targets",
            "by_context": {"16": {"loss": 3.97}, "1024": {"loss": 3.3347}},
        },
        "context_benefit": {
            "cb@1024": {
                "loss_real_prefix": 3.2,
                "loss_other_doc_prefix": 3.5,
                "benefit_nats": 0.2436,
                "benefit_se": 0.0063,
                "windows": 242,
            }
        },
        "retrieval": {
            "candidates": 10,
            "chance": 0.1,
            "trials_per_distance": 400,
            "by_distance": {"256": {"accuracy": 0.96}, "496": {"accuracy": 0.70}},
        },
        "training_systems": {
            "device": "NVIDIA H100",
            "world_size": 1,
            "tokens": 1310720000,
            "train_tokens_per_sec": 61478,
            "peak_mem_gb": 11.058,
        },
        "samples": {
            "protocol": "20 prompts x 5 draws",
            "summary": {"n": 100, "rep4": 0.342, "looped": 11, "loop_ci95": [0.063, 0.186], "topic": 0.36},
            "prompts": [{"label": "history", "prompt": "The French Revolution began in 1789, when", "draws": [draw]}],
        },
    }
    (tmp_path / f"evals/{STEM}.json").write_text(json.dumps(ev))
    return tmp_path


def test_report_has_the_numbers_curves_samples_and_glossary(repo):
    out = rr.write(repo, "r1", pdf=False)
    md = out["md"].read_text()
    for needle in (
        "3.4038",  # headline full_val
        "11/100",  # loops
        "36%",  # topic held
        "95,333,713 params",
        "RoPE · fused (SDPA)",
        "633,453,440 train tokens",
        "~2.07 passes",
        "$23.25 (billed)",
        "5.2654",  # quick eval curve from the log
        "**this run** · data640k-fw70edu30",  # comparison table, dataset named in full
        "The French Revolution began in 1789, when in 1789 the people rose up.",  # samples, prompt + text
        "## Glossary",
        "# data640k-fw70edu30",  # dataset manifest
    ):
        assert needle in md, needle
    html = out["html"].read_text()
    assert "data:image/png;base64," in html and "<table>" in html  # self-contained, plot embedded


def test_no_chrome_means_no_pdf_with_a_reason(repo, monkeypatch):
    monkeypatch.setattr(rr, "CHROME", "/nonexistent/chrome")
    out = rr.write(repo, "r1")
    assert "Chrome not found" in out["pdf"] and out["md"].exists()


def test_runs_without_finished_evals_are_skipped(repo):
    st = json.loads((repo / "runs/r1.status.json").read_text())
    st["eval"]["state"] = "running"
    (repo / "runs/r1.status.json").write_text(json.dumps(st))
    with pytest.raises(ValueError, match="evals not done"):
        rr.write(repo, "r1", pdf=False)


def test_server_lists_and_serves_the_report(repo):
    c = TestClient(create_app(repo=repo, uv="uv"))
    assert c.get("/api/runs").json()[0]["report_files"] == []
    rr.write(repo, "r1", pdf=False)
    assert c.get("/api/runs").json()[0]["report_files"] == ["md", "html"]
    r = c.get("/api/runs/r1/report_file?fmt=md")
    assert r.status_code == 200 and "Training run report" in r.text and STEM in r.headers["content-disposition"]
    assert c.get("/api/runs/r1/report_file?fmt=pdf").status_code == 404


@pytest.mark.parametrize(
    "stem, data",
    [
        ("modal_data640k-fw70edu30-b8-t1024-e768h12l8-160k", "data640k-fw70edu30"),
        ("modal_data640k-b8-t1024-e768h12l8-160k", "data640k"),
        ("modal_data640k-e768h12l8-160k-lr3e-4-rope-fused-cont30k", "data640k"),
        ("data160k_b8_t1024", "data160k"),
    ],
)
def test_label_names_the_whole_dataset(stem, data):
    r = {
        "checkpoint": f"{stem}.pt",
        "config": {"n_embd": 768, "n_layer": 8, "block_size": 1024},
        "params": 95e6,
        "step": 1000,
    }
    assert label(r).split(" · ")[0] == data
