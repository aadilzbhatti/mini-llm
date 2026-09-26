"""Importing a fetched remote run lands it like a local one: files in place,
one baselines row with train.py's columns, idempotent on re-import."""

import json
import os

import pytest
import torch

import mini_llm.train as train
from mini_llm.import_run import import_run


class StubTokenizer:
    def __len__(self):
        return 64


ARGV = ["--tokens", "/data/train.pt", "--val-tokens", "/data/val.pt", "--block-size", "8", "--n-embd", "16",
        "--n-head", "2", "--n-layer", "1", "--batch-size", "4", "--steps", "20", "--lr", "2e-3",
        "--warmup-steps", "0", "--eval-interval", "10", "--eval-batches", "2", "--full-eval-interval", "0",
        "--save", "--plot-loss", "--no-tensorboard"]


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run directory shaped like one fetched from Modal."""
    run = tmp_path / "runs" / "20260101-000000-abc1234-tiny"
    run.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    data = tmp_path / "data"
    data.mkdir()
    torch.save(torch.randint(0, 64, (4000,), generator=g), data / "train.pt")
    torch.save(torch.randint(0, 64, (600,), generator=g), data / "val.pt")
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("MINI_LLM_RUN_ID", run.name)
    monkeypatch.chdir(run)  # remote runs write relative to their run dir
    local_argv = [str(data / a[len("/data/"):]) if a.startswith("/data/") else a for a in ARGV]
    train.main(local_argv)
    (run / "run.json").write_text(json.dumps({
        "run_id": run.name, "config": {"name": "tiny"}, "gpus": "L4:2", "nproc": 2, "git_sha": "abc1234",
        "resolved_argv": ARGV, "returncode": 0, "finished_at": "2026-01-01T00:05:00+00:00", "duration_sec": 300.0,
    }))
    monkeypatch.chdir(tmp_path)
    return run


def test_import_lands_files_and_one_row(run_dir, tmp_path):
    repo = tmp_path / "repo"
    row = import_run(run_dir, repo)

    assert (repo / "checkpoints" / "modal_tiny_seed42.pt").exists()
    assert row["plot"] and (repo / row["plot"]).exists()
    assert row["batch_size"] == 4 and row["lr"] == 2e-3 and row["min_lr"] == 2e-6  # parser default resolved
    assert row["steps"] == 20 and row["full_val_loss"] is not None
    assert row["params"] == sum(p.numel() for p in train.build_model(
        train.ModelConfig(vocab_size=64, block_size=8, n_embd=16, n_head=2, n_layer=1)).parameters())

    import_run(run_dir, repo)  # re-import replaces, never duplicates
    rows = json.loads((repo / "baselines.json").read_text())
    assert [r["run"] for r in rows] == ["modal_tiny_seed42.pt"]
    assert rows[0]["gpus"] == "L4:2"                                    # JSON-only detail kept
    assert "modal_tiny_seed42.pt" in (repo / "baselines.md").read_text()
    assert "L4:2" not in (repo / "baselines.md").read_text()           # table columns unchanged


def test_unfinished_run_is_refused(run_dir, tmp_path):
    record = json.loads((run_dir / "run.json").read_text())
    del record["returncode"]
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="did not finish"):
        import_run(run_dir, tmp_path / "repo")
    assert not os.path.exists(tmp_path / "repo" / "baselines.md")
