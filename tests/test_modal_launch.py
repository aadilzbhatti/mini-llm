"""Starting Modal runs from the web page: server validation + background launch."""

import json
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import mini_llm.remote.launch as launch_mod
import mini_llm.server as server
from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent
JOB = {"name": "web-bs64", "kind": "train", "args": {"batch-size": 64, "steps": 100, "tokens": "data/d1/train.pt"}}


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "runner").mkdir()
    shutil.copy(REPO / "runner" / "run_queue.py", tmp_path / "runner" / "run_queue.py")
    for rel in ("data/d1/train.pt", "data/train.pt", "data/val.pt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    (tmp_path / "runs").mkdir()
    (tmp_path / "queue").mkdir()
    return tmp_path


@pytest.fixture
def spawned(monkeypatch):
    """Capture the launcher spawn; let everything else (git, via make_run_id) run for real."""
    calls, real_popen = [], server.subprocess.Popen

    def popen(cmd, *args, **kwargs):
        if "mini_llm.remote.launch" in cmd:
            calls.append(cmd)
            return None
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    return calls


def post(repo, body, dry_run=False):
    return TestClient(create_app(repo=repo, uv="uv")).post(f"/api/jobs{'?dry_run=true' if dry_run else ''}", json=body)


def test_dry_run_previews_without_launching(repo, spawned):
    r = post(repo, {**JOB, "target": "modal", "gpus": "L4:2"}, dry_run=True)
    assert r.status_code == 200, r.text
    assert r.json()["nproc"] == 2 and r.json()["run_id"].endswith("-web-bs64")
    assert "--baseline" not in r.json()["argv"] and "--plot-loss" in r.json()["argv"]  # the argv Modal runs
    assert spawned == [] and not list((repo / "runs").glob("*.status.json"))


@pytest.mark.parametrize("extra, message", [
    ({"gpus": "L4;rm -rf"}, "gpus must look like"),
    ({"gpus": "L4:3"}, "divide evenly across 3"),
    ({"timeout_hours": 48}, "between 0.1 and 24"),
    ({"kind": "prepare-data", "args": {"out-dir": "data/x"}}, "only train jobs"),
])
def test_rejects_bad_modal_jobs(repo, spawned, extra, message):
    r = post(repo, {**JOB, "target": "modal", **extra})
    assert r.status_code == 422 and message in r.text
    assert spawned == []


def test_launch_shows_run_at_once_and_skips_queue(repo, spawned):
    r = post(repo, {**JOB, "target": "modal", "gpus": "H100:4", "timeout_hours": 2})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]

    status = json.loads((repo / "runs" / f"{run_id}.status.json").read_text())
    assert status["status"] == "running" and status["remote"] == {"provider": "modal", "gpus": "H100:4", "phase": "launching"}
    job = json.loads((repo / "runs" / f"{run_id}.modal-job.json").read_text())
    assert job["args"]["plot-loss"] is True and "baseline" not in job["args"]   # runner defaults, minus baseline
    assert job["timeout_hours"] == 2.0
    assert not list((repo / "queue").glob("*.json"))                            # never queued locally
    [cmd] = spawned
    assert cmd[1:3] == ["-m", "mini_llm.remote.launch"]
    # ...and it's on the page, as a live run.
    runs = TestClient(create_app(repo=repo, uv="uv")).get("/api/runs").json()
    assert [(x["run_id"], x["status"]) for x in runs] == [(run_id, "running")]


def fake_uv(tmp_path, exit_code, output):
    script = tmp_path / "uv"
    script.write_text(f"#!/bin/sh\necho '{output}'\nexit {exit_code}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def job_file(repo, run_id="20260101-000000-abc1234-web"):
    path = repo / "runs" / f"{run_id}.modal-job.json"
    path.write_text(json.dumps({"run_id": run_id, "name": "web", "args": {"steps": 10}, "gpus": "L4:2",
                                "timeout_hours": 1.0}))
    return path


def test_failed_launch_is_reported(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(launch_mod, "upload_missing_data", lambda repo, args: [])
    path = job_file(repo)
    rc = launch_mod.launch(path, repo, uv=fake_uv(tmp_path, 1, "Error: App create rate limit exceeded."))
    status = json.loads((repo / "runs" / "20260101-000000-abc1234-web.status.json").read_text())
    assert rc == 1 and status["status"] == "failed" and "rate limit" in status["error"]
    assert "rate limit" in (repo / "runs" / "20260101-000000-abc1234-web.log").read_text()  # Log button


def test_successful_launch_leaves_run_to_the_mirror(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(launch_mod, "upload_missing_data", lambda repo, args: [])
    rc = launch_mod.launch(job_file(repo), repo, uv=fake_uv(tmp_path, 0, "spawned: fc-123"))
    status = json.loads((repo / "runs" / "20260101-000000-abc1234-web.status.json").read_text())
    assert rc == 0 and status["status"] == "running" and status["remote"]["phase"] == "launching"


class FakeDataVolume:
    def __init__(self, present):
        self.present, self.uploaded = set(present), []

    def listdir(self, path):
        if path not in self.present:
            raise FileNotFoundError(path)
        return [path]

    @contextmanager
    def batch_upload(self):
        yield self

    def put_file(self, local, remote):
        self.uploaded.append((Path(local).name, remote))


def test_uploads_only_missing_data(repo):
    vol = FakeDataVolume(present={"/d1/train.pt"})
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "c.pt").write_bytes(b"")
    args = {"tokens": "data/d1/train.pt", "val-tokens": "data/val.pt", "resume": "checkpoints/c.pt", "steps": 5}
    assert sorted(launch_mod.upload_missing_data(repo, args, volume=vol)) == ["/checkpoints/c.pt", "/val.pt"]
    assert sorted(vol.uploaded) == [("c.pt", "/checkpoints/c.pt"), ("val.pt", "/val.pt")]
