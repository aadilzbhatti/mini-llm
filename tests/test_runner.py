"""Job validation in the queue runner, including prepare-data jobs."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rq():
    spec = importlib.util.spec_from_file_location("_rq", REPO / "runner" / "run_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_is_default_kind(rq, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.pt").write_bytes(b"")
    (tmp_path / "data" / "val.pt").write_bytes(b"")
    name, kind, cmd, _ = rq.validate_job({"name": "a", "args": {"steps": 10}}, tmp_path, "uv")
    assert kind == "train" and "mini-llm-train" in cmd and "--steps" in cmd


def test_prepare_data_job(rq, tmp_path):
    _, kind, cmd, _ = rq.validate_job(
        {"name": "d", "kind": "prepare-data",
         "args": {"num-examples": 20000, "out-dir": "data/data20k", "dataset": "HuggingFaceTB/smollm-corpus"}},
        tmp_path, "uv",
    )
    assert kind == "prepare-data" and "mini-llm-prepare-data" in cmd
    assert cmd[cmd.index("--out-dir") + 1] == "data/data20k"


@pytest.mark.parametrize("out_dir", ["/tmp/x", "data/../x", "elsewhere/x", "data", "data/a b"])
def test_prepare_data_out_dir_must_be_under_data(rq, tmp_path, out_dir):
    with pytest.raises(rq.JobError):
        rq.validate_job({"kind": "prepare-data", "args": {"out-dir": out_dir}}, tmp_path, "uv")


def test_prepare_data_rejects_train_flags_and_unknown_kind(rq, tmp_path):
    with pytest.raises(rq.JobError):
        rq.validate_job({"kind": "prepare-data", "args": {"out-dir": "data/x", "lr": 0.1}}, tmp_path, "uv")
    with pytest.raises(rq.JobError):
        rq.validate_job({"kind": "shell", "args": {}}, tmp_path, "uv")


def test_mark_interrupted_skips_live_owner(rq, tmp_path):
    import json, os
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "dead.status.json").write_text(json.dumps({"status": "running", "runner_pid": 999999}))
    (runs / "live.status.json").write_text(json.dumps({"status": "running", "runner_pid": os.getppid()}))
    rq.mark_interrupted(tmp_path)
    assert json.loads((runs / "dead.status.json").read_text())["status"] == "interrupted"
    assert json.loads((runs / "live.status.json").read_text())["status"] == "running"
