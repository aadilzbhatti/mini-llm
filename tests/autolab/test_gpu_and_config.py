"""Shared-GPU wait, frozen-val check, and trainer argv (no training)."""

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autolab.config import FrozenValError, check_frozen_val, load_config, sha256_file
from autolab.gpu import active_owner_runs, wait_for_gpu
from autolab.trainer import Budget, TrainRequest, trainer_argv


def live(path: Path, age_s: float, finished: bool = False, run_id: str = "r"):
    updated = datetime.fromtimestamp(time.time() - age_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"run_id": run_id, "updated": updated, "step": 10, "total_steps": 100}
    if finished:
        body["finished"] = True
    path.write_text(json.dumps(body))


def test_active_owner_runs(tmp_path):
    live(tmp_path / "a.live.json", 5, run_id="active")
    live(tmp_path / "b.live.json", 5, finished=True, run_id="done")
    live(tmp_path / "c.live.json", 600, run_id="stale")
    (tmp_path / "d.live.json").write_text("{not json")
    assert [r["run_id"] for r in active_owner_runs(tmp_path, 120)] == ["active"]


def test_wait_for_gpu_waits_then_needs_two_idle_checks(tmp_path):
    live(tmp_path / "a.live.json", 5)
    logs, sleeps = [], []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:  # the owner's run finishes during the second wait
            live(tmp_path / "a.live.json", 5, finished=True)

    waited = wait_for_gpu(tmp_path, 120, 60, 2, log=logs.append, sleep=fake_sleep)
    assert waited == 180 and len(sleeps) == 3  # busy, busy, idle (1 of 2), idle (2 of 2)
    assert sum("GPU busy" in m for m in logs) == 2


def test_wait_for_gpu_timeout(tmp_path):
    live(tmp_path / "a.live.json", 5)
    with pytest.raises(TimeoutError):
        wait_for_gpu(tmp_path, 120, 60, 1, log=lambda m: None, sleep=lambda s: None, max_wait_s=120)


def test_config_points_at_frozen_val():
    cfg = load_config()
    assert cfg.frozen_val.name == "val_frozen.pt"
    assert len(cfg.frozen_val_sha256) == 64
    if cfg.frozen_val.exists():  # data is gitignored; present on the Mac
        assert check_frozen_val(cfg) == cfg.frozen_val


def test_frozen_val_mismatch_refuses(tmp_path):
    fake = tmp_path / "val.pt"
    fake.write_bytes(b"not the val")
    cfg = replace(load_config(), frozen_val=fake, frozen_val_sha256="0" * 64)
    with pytest.raises(FrozenValError, match="mismatch"):
        check_frozen_val(cfg)
    assert check_frozen_val(replace(cfg, frozen_val_sha256=sha256_file(fake))) == fake
    with pytest.raises(FrozenValError, match="missing"):
        check_frozen_val(replace(cfg, frozen_val=tmp_path / "nope.pt"))


def test_trainer_argv_and_request_roundtrip(tmp_path):
    req = TrainRequest(run_id="r1", dataset_id="data20k", train_tokens=str(tmp_path / "t.pt"),
                       budget=Budget(tokens=1_000_000, wall_clock_s=420), seed=7)
    assert req.steps() == 3907  # ceil(1e6 / (4 * 64))
    argv = trainer_argv(req, tmp_path / "val.pt")
    assert argv[1:3] == ["-m", "mini_llm.train"]
    flags = dict(zip(argv[3::2], argv[4::2]))
    assert flags["--steps"] == "3907" and flags["--seed"] == "7"
    assert flags["--val-tokens"] == str((tmp_path / "val.pt").resolve())
    assert "--baseline" not in argv and "--no-tensorboard" not in argv
    assert TrainRequest.from_dict(json.loads(json.dumps(req.to_dict()))) == req
