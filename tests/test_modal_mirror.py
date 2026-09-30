"""Modal runs mirrored into runs/ look like local runs to the web app and runner."""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import mini_llm.train as train
from mini_llm.remote.modal_mirror import load_runner, mirror_all, mirror_run
from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent


class FakeVolume:
    """modal.Volume's listdir/read_file over a local directory."""

    def __init__(self, root: Path):
        self.root = root
        self.reads: list[str] = []

    def listdir(self, path, recursive=False):
        base = self.root / path.strip("/")
        entries = base.rglob("*") if recursive else base.iterdir()
        return [SimpleNamespace(path=str(p.relative_to(self.root)), type=1 if p.is_file() else 2,
                                size=p.stat().st_size if p.is_file() else 0, mtime=int(p.stat().st_mtime))
                for p in entries]

    def read_file(self, path):
        self.reads.append(path)
        yield (self.root / path).read_bytes()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "repo" / "runner").mkdir(parents=True)
    shutil.copy(REPO / "runner" / "run_queue.py", tmp_path / "repo" / "runner" / "run_queue.py")
    (tmp_path / "repo" / "queue").mkdir()
    return tmp_path / "repo"


def remote_run(vol_root: Path, run_id: str, *, finished: bool, heartbeat: float | None = None, argv=None) -> Path:
    """A run directory as modal_train.py leaves it on the volume."""
    run = vol_root / run_id
    (run / "runs").mkdir(parents=True)
    record = {"run_id": run_id, "config": {"name": "tiny", "args": {"steps": 20}}, "gpus": "L4:2", "nproc": 2,
              "git_sha": "abc1234", "resolved_argv": argv or ["--steps", "20"], "command": "torchrun ...",
              "started_at": iso(time.time() - 60)}
    if finished:
        record.update(returncode=0, finished_at=iso(time.time()), duration_sec=60.0)
    (run / "run.json").write_text(json.dumps(record))
    (run / "train.log").write_text("Model: 1,234 parameters\nstep     0 | loss 4.0000 | lr 1.00e-03\n"
                                   "step    10 | loss 3.5000 | lr 1.00e-03\n")
    if heartbeat is not None:
        (run / "runs" / f"{run_id}.live.json").write_text(
            json.dumps({"run_id": run_id, "step": 10, "total_steps": 20, "updated": iso(heartbeat)}))
    return run


def test_running_run_appears_live_and_is_read_only(tmp_path, repo):
    vol = FakeVolume(tmp_path / "vol")
    remote_run(vol.root, "r-live", finished=False, heartbeat=time.time())
    [st] = mirror_all(vol, repo, load_runner(repo))

    assert st["status"] == "running" and st["remote"]["gpus"] == "L4:2"
    assert (repo / "runs" / "r-live.live.json").exists()
    assert st["metrics"]["last_step"] == 10 and st["metrics"]["params"] == 1234  # runner.summarize

    c = TestClient(create_app(repo=repo, uv="uv"))
    [row] = c.get("/api/runs").json()
    assert row["status"] == "running" and row["live"]["step"] == 10 and row["remote"]["provider"] == "modal"
    assert "step    10" in c.get("/api/runs/r-live/log").text
    resp = c.post("/api/runs/r-live/commands", json={"type": "stop"})
    assert resp.status_code == 409 and "Modal" in resp.json()["detail"]

    load_runner(repo).mark_interrupted(repo)  # a runner restart must leave it alone
    assert json.loads((repo / "runs" / "r-live.status.json").read_text())["status"] == "running"


def test_silent_run_shows_interrupted(tmp_path, repo):
    vol = FakeVolume(tmp_path / "vol")
    remote_run(vol.root, "r-dead", finished=False, heartbeat=time.time() - 3600)
    st = mirror_run(vol, "r-dead", repo, load_runner(repo), stale_after=900)
    assert st["status"] == "interrupted" and "no heartbeat" in st["error"]


def test_finished_run_is_imported_once(tmp_path, repo, monkeypatch):
    vol = FakeVolume(tmp_path / "vol")
    argv = ["--tokens", "/data/train.pt", "--val-tokens", "/data/val.pt", "--block-size", "8", "--n-embd", "16",
            "--n-head", "2", "--n-layer", "1", "--batch-size", "4", "--steps", "20", "--warmup-steps", "0",
            "--eval-interval", "10", "--eval-batches", "2", "--full-eval-interval", "0", "--save",
            "--plot-loss", "--no-tensorboard"]
    run = remote_run(vol.root, "r-done", finished=True, argv=argv)

    # Real outputs in the run dir, as the container would have written them.
    data = tmp_path / "data"
    data.mkdir()
    g = torch.Generator().manual_seed(0)
    torch.save(torch.randint(0, 64, (4000,), generator=g), data / "train.pt")
    torch.save(torch.randint(0, 64, (600,), generator=g), data / "val.pt")
    monkeypatch.setattr(train, "get_tokenizer", lambda: type("T", (), {"__len__": lambda self: 64})())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.chdir(run)
    train.main([str(data / a[6:]) if a.startswith("/data/") else a for a in argv])
    monkeypatch.chdir(tmp_path)
    [plot] = (run / "plots").glob("*.png")  # train.log in a real run carries this line
    with (run / "train.log").open("a") as log:
        log.write(f"Saved loss plot to plots/{plot.name}\n")

    st = mirror_run(vol, "r-done", repo, load_runner(repo))
    assert st["status"] == "completed" and st["remote"]["imported"] and st["remote"]["synced"]
    rows = json.loads((repo / "baselines.json").read_text())
    assert [r["run"] for r in rows] == ["modal_tiny_steps20_seed42.pt"]
    assert (repo / st["metrics"]["plot"]).exists()                      # the page's Plot button path

    vol.reads.clear()
    mirror_run(vol, "r-done", repo, load_runner(repo))                  # finished + synced: no more reads
    assert vol.reads == []


def test_finished_run_without_save_is_synced_not_imported(tmp_path, repo):
    # LR proxies train with a val set but no --save: nothing to import, and trying used to
    # SystemExit the whole mirror on every poll.
    vol = FakeVolume(tmp_path / "vol")
    remote_run(vol.root, "r-proxy", finished=True, argv=["--val-tokens", "/data/val.pt", "--steps", "20"])
    [st] = mirror_all(vol, repo, load_runner(repo))
    assert st["status"] == "completed" and not st["remote"]["imported"] and st["remote"]["synced"]
