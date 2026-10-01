"""The Modal wrapper's local helpers (no Modal account or network needed).

Skipped unless the optional group is installed: uv run --group modal pytest
"""

import json
from pathlib import Path

import pytest

modal_train = pytest.importorskip("mini_llm.remote.modal_train")


def test_config_to_argv_accepts_runner_jobs():
    job = json.loads(Path("runner/example-job.json").read_text())
    argv = modal_train.config_to_argv(job)
    assert argv[:2] == ["--save", "--plot-loss"]  # remote defaults
    assert argv[argv.index("--n-embd") + 1] == "192"
    assert modal_train.config_to_argv({"save": False, "bf16": True, "name": "x"}) == ["--plot-loss", "--bf16"]


def test_parse_gpus():
    assert modal_train.parse_gpus("H100:4") == ("H100:4", 4)
    assert modal_train.parse_gpus("A100-80GB") == ("A100-80GB", 1)
    assert modal_train.parse_gpus("cpu") == (None, 2)


def test_resolve_args_maps_data_paths(tmp_path, monkeypatch):
    (tmp_path / "vol" / "data10k").mkdir(parents=True)
    (tmp_path / "vol" / "data10k" / "train.pt").touch()
    monkeypatch.setattr(modal_train, "DATA_MOUNT", str(tmp_path / "vol"))
    monkeypatch.setattr(modal_train, "BUNDLED_DATA", str(tmp_path / "bundled"))

    out = modal_train.resolve_args(["--tokens", "data/data10k/train.pt", "--resume=/runs/x/c.pt", "--steps", "5"])
    assert out == ["--tokens", str(tmp_path / "vol/data10k/train.pt"), "--resume=/runs/x/c.pt", "--steps", "5"]
    with pytest.raises(FileNotFoundError, match="modal volume put"):
        modal_train.resolve_args(["--val-tokens", "data/missing.pt"])


def test_sample_report_is_never_run_on_modal():
    argv = modal_train.config_to_argv({"steps": 10, "sample-report": True, "sample-report-tokens": 64})
    assert argv == ["--save", "--plot-loss", "--steps", "10"]
