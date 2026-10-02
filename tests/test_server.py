"""Control API over a fake repo: submit, cancel, command a run, read results."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "runner").mkdir()
    shutil.copy(REPO / "runner" / "run_queue.py", tmp_path / "runner" / "run_queue.py")
    (tmp_path / "data" / "d1").mkdir(parents=True)
    (tmp_path / "data" / "d1" / "train.pt").write_bytes(b"")
    (tmp_path / "data" / "d1" / "val.pt").write_bytes(b"")
    (tmp_path / "data" / "train.pt").write_bytes(b"")  # the runner's default token paths
    (tmp_path / "data" / "val.pt").write_bytes(b"")
    (tmp_path / "runs").mkdir()
    (tmp_path / "queue").mkdir()
    return tmp_path


def client(repo, token=None):
    return TestClient(create_app(repo=repo, token=token, uv="uv"))


def running(repo, run_id="r1", **extra):
    (repo / "runs" / f"{run_id}.status.json").write_text(
        json.dumps({"run_id": run_id, "name": run_id, "status": "running", "kind": "train", **extra}))


def test_token_required_when_set(repo):
    c = client(repo, token="s3cret")
    assert c.get("/api/runs").status_code == 401
    assert c.get("/api/runs", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_meta_lists_knobs_and_datasets(repo):
    m = client(repo).get("/api/meta").json()
    assert "lr_scale" in m["knobs"] and m["datasets"] == ["data", "data/d1"]


def test_submit_valid_job_lands_in_queue(repo):
    r = client(repo).post("/api/jobs", json={"name": "a", "args": {"steps": 10, "tokens": "data/d1/train.pt"}})
    assert r.status_code == 200, r.text
    files = list((repo / "queue").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["args"]["steps"] == 10


def test_submit_invalid_job_never_touches_queue(repo):
    r = client(repo).post("/api/jobs", json={"name": "a", "args": {"steps": 10, "rm": "-rf"}})
    assert r.status_code == 422
    assert list((repo / "queue").iterdir()) == []


def test_cancel_moves_job_aside(repo):
    c = client(repo)
    f = c.post("/api/jobs", json={"name": "a", "args": {"steps": 10}}).json()["file"]
    assert c.delete(f"/api/queue/{f}").status_code == 200
    assert (repo / "queue" / "cancelled" / f).exists()
    assert c.delete("/api/queue/..%2Fpyproject.json").status_code in (400, 404)


def test_command_only_to_running_run(repo):
    c = client(repo)
    running(repo)
    r = c.post("/api/runs/r1/commands", json={"type": "set", "knob": "lr_scale", "value": 0.5})
    assert r.status_code == 200
    line = (repo / "runs" / "r1.commands.jsonl").read_text().strip()
    assert json.loads(line)["value"] == 0.5
    assert c.post("/api/runs/r1/commands", json={"type": "set", "knob": "lr", "value": 1}).status_code == 422
    (repo / "runs" / "done.status.json").write_text(json.dumps({"run_id": "done", "status": "completed"}))
    assert c.post("/api/runs/done/commands", json={"type": "stop"}).status_code == 409


def test_runs_list_includes_live(repo):
    running(repo)
    (repo / "runs" / "r1.live.json").write_text(json.dumps({"step": 5, "total_steps": 10}))
    rows = client(repo).get("/api/runs").json()
    assert rows[0]["live"]["step"] == 5


def test_report_found_by_save_name(repo):
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "m.md").write_text("# samples")
    (repo / "runs" / "r2.status.json").write_text(json.dumps({"run_id": "r2", "status": "completed", "args": {"save-name": "m.pt"}}))
    assert client(repo).get("/api/runs/r2/report").text == "# samples"


def test_index_page_served(repo):
    r = client(repo).get("/")
    assert r.status_code == 200 and "mini-llm" in r.text


def test_plot_served_from_plots_dir_only(repo):
    (repo / "plots").mkdir()
    (repo / "plots" / "p.png").write_bytes(b"\x89PNG fake")
    (repo / "secret.png").write_bytes(b"nope")
    (repo / "runs" / "r3.status.json").write_text(json.dumps({"run_id": "r3", "status": "completed", "metrics": {"plot": "plots/p.png"}}))
    (repo / "runs" / "r4.status.json").write_text(json.dumps({"run_id": "r4", "status": "completed", "metrics": {"plot": "plots/../secret.png"}}))
    c = client(repo)
    r = c.get("/api/runs/r3/plot")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert c.get("/api/runs/r4/plot").status_code == 404


def test_dry_run_validates_without_queueing(repo):
    c = client(repo)
    r = c.post("/api/jobs?dry_run=true", json={"name": "a", "args": {"steps": 10}})
    assert r.status_code == 200 and r.json()["ok"]
    assert list((repo / "queue").glob("*.json")) == []
    bad = c.post("/api/jobs?dry_run=true", json={"name": "a", "args": {"n-embd": 100, "n-head": 3}})
    assert bad.status_code == 422 and "divisible" in bad.json()["detail"]
    assert c.post("/api/jobs?dry_run=true", json={"name": "a", "args": {"restart-lr": 1e-4}}).status_code == 422


def test_continuation_of_running_run_validates_against_pending_checkpoint(repo):
    c = client(repo)
    (repo / "queue" / "j.json").write_text(json.dumps({"name": "big", "args": {"steps": 160000, "save": True, "save-name": "big_160k_x.pt", "n-embd": 256}}))
    running(repo, "r9", name="big", job_file="j.json", args={"steps": 160000, "save": True, "save-name": "big_160k_x.pt", "n-embd": 256})
    q = c.get("/api/queue").json()
    assert q[0]["running"] is True
    assert c.delete("/api/queue/j.json").status_code == 409          # can't cancel what's running
    job = c.get("/api/runs/r9/continuation").json()
    assert job["args"]["resume"] == "checkpoints/big_160k_x.pt"
    assert job["args"]["n-embd"] == 256 and job["args"]["save-name"] == "big_200k_x_resume.pt"
    # checkpoint doesn't exist yet, but it's pending, so the continuation queues
    r = c.post("/api/jobs", json=job)
    assert r.status_code == 200, r.text
    assert c.get("/api/queue/j.json/continuation").json()["args"]["resume"] == "checkpoints/big_160k_x.pt"


def test_resume_of_nonexistent_non_pending_checkpoint_rejected(repo):
    r = client(repo).post("/api/jobs", json={"name": "a", "args": {"resume": "checkpoints/nope.pt", "steps": 5}})
    assert r.status_code == 422


def test_meta_form_schema(repo):
    form = client(repo).get("/api/meta").json()["form"]
    fields = {f["flag"]: f for f in form["train"]}
    assert fields["tokens"]["type"] == "path" and "data/d1/train.pt" in fields["tokens"]["choices"]
    assert fields["lr"]["min"] > 0 and fields["save"]["type"] == "bool"
    assert {f["flag"] for f in form["prepare-data"]} >= {"out-dir", "dataset", "num-examples"}


def test_api_responses_carry_the_page_version(repo):  # noqa: F811
    from fastapi.testclient import TestClient
    from mini_llm.server import STATIC, create_app
    c = TestClient(create_app(repo=repo, uv="uv"))
    assert c.get("/api/meta").headers["x-ui-version"] == str(int((STATIC / "index.html").stat().st_mtime))
    assert c.get("/").headers["cache-control"] == "no-cache"
