"""Sweeps: one base job, one flag varied -> validated children, launched in order, summarized for the page."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import mini_llm.remote.launch as launch_mod
import mini_llm.server as server
from mini_llm import sweeps
from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent
BASE = {
    "name": "e768-lr",
    "kind": "train",
    "vary": {"lr": [3e-4, 4e-4, 5e-4]},
    "stop_after": 100,
    "args": {"steps": 1000, "batch-size": 8, "tokens": "data/d1/train.pt"},
}


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
    calls, real_popen = [], server.subprocess.Popen

    def popen(cmd, *args, **kwargs):
        if "mini_llm.remote.launch" in cmd:
            calls.append(cmd)
            return None
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    return calls


def client(repo):
    return TestClient(create_app(repo=repo, uv="uv"))


@pytest.mark.parametrize(
    "v, tag", [(3e-4, "3e-4"), (5e-4, "5e-4"), (2.5e-4, "2.5e-4"), (1.2e-3, "1.2e-3"), (0.1, "0.1"), (256000, "256000")]
)
def test_value_tag(v, tag):
    assert sweeps.value_tag(v) == tag


@pytest.mark.parametrize(
    "change, message",
    [
        ({"vary": {"lr": [3e-4], "seed": [1, 2]}}, "exactly one flag"),
        ({"vary": {"lr": [3e-4]}}, "2-8 values"),
        ({"vary": {"lr": ["big", 1e-4]}}, "numbers"),
        ({"vary": {"lr": [3e-4, 0.0003]}}, "distinct"),
        ({"vary": {"steps": [10, 20]}}, "separate runs"),
        ({"vary": {"tokens": [1, 2]}}, "numeric"),
        ({"vary": {"seed": [1, 2.5]}}, "whole numbers"),
        ({"stop_after": 5000}, "within --steps"),
        ({"name": "bad name"}, "name"),
    ],
)
def test_bad_sweeps_rejected_and_nothing_launched(repo, spawned, change, message):
    r = client(repo).post("/api/sweeps", json={**BASE, **change})
    assert r.status_code == 422 and message in r.json()["detail"]
    assert spawned == [] and not (repo / "runs" / "sweeps").exists() and not list((repo / "queue").glob("*.json"))


def test_a_child_failing_runner_validation_rejects_the_whole_sweep(repo, spawned):
    r = client(repo).post("/api/sweeps", json={**BASE, "target": "modal", "vary": {"lr": [3e-4, 50.0]}})
    assert r.status_code == 422 and "e768-lr-lr50" in r.json()["detail"]
    assert spawned == [] and not list((repo / "runs").glob("*.status.json"))


def test_modal_sweep_launches_children_in_order_from_one_process(repo, spawned):
    r = client(repo).post("/api/sweeps", json={**BASE, "target": "modal", "gpus": "H100"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert [c["name"] for c in out["children"]] == ["e768-lr-lr3e-4", "e768-lr-lr4e-4", "e768-lr-lr5e-4"]

    [cmd] = spawned  # ONE launcher, all three job files in sweep order
    job_files = [a for a in cmd if a.endswith(".modal-job.json")]
    jobs = [json.loads(Path(f).read_text()) for f in job_files]
    assert [j["args"]["lr"] for j in jobs] == [3e-4, 4e-4, 5e-4]
    assert all(j["args"]["stop-after"] == 100 and j["args"]["steps"] == 1000 for j in jobs)  # the real schedule
    assert all(j["gpus"] == "H100" for j in jobs)
    for j in jobs:  # every child visible at once
        assert json.loads((repo / "runs" / f"{j['run_id']}.status.json").read_text())["remote"]["phase"] == "launching"

    record = json.loads((repo / "runs" / "sweeps" / f"{out['id']}.json").read_text())
    assert record["flag"] == "lr" and [c["run_id"] for c in record["children"]] == [j["run_id"] for j in jobs]
    view = client(repo).get(f"/api/sweeps/{out['id']}").json()
    assert [c["status"] for c in view["children"]] == ["running"] * 3 and view["table"] == []


def test_dry_run_writes_nothing(repo, spawned):
    r = client(repo).post("/api/sweeps?dry_run=true", json={**BASE, "target": "modal"})
    assert r.status_code == 200 and len(r.json()["children"]) == 3 and r.json()["children"][0]["hourly_usd"]
    assert spawned == [] and not (repo / "runs" / "sweeps").exists()


def test_local_sweep_queues_children_and_finds_their_runs(repo, spawned):
    out = client(repo).post("/api/sweeps", json=BASE).json()
    queued = sorted(p.name for p in (repo / "queue").glob("*.json"))
    assert len(queued) == 3 and spawned == []
    # the runner later starts one and records which queue file it came from
    first = json.loads((repo / "runs" / "sweeps" / f"{out['id']}.json").read_text())["children"][0]
    (repo / "runs" / "r1.status.json").write_text(
        json.dumps({"run_id": "r1", "status": "running", "job_file": first["key"], "metrics": {}})
    )
    view = client(repo).get(f"/api/sweeps/{out['id']}").json()
    assert [c["status"] for c in view["children"]] == ["running", "queued", "queued"]
    assert view["children"][0]["run_id"] == "r1"


def test_summary_compares_at_steps_every_child_reached():
    record = {
        "id": "s",
        "name": "s",
        "flag": "lr",
        "values": [2e-4, 3e-4, 4e-4],
        "stop_after": 10000,
        "target": "modal",
        "created": "x",
        "children": [{"value": v, "name": f"s-lr{v}", "key": k} for v, k in ((2e-4, "a"), (3e-4, "b"), (4e-4, "c"))],
    }

    def st(curve, status="completed"):
        return {"status": status, "metrics": {"full_val_curve": curve, "last_step": curve[-1][0]}}

    statuses = {
        "a": st([[0, 10.8], [2500, 5.03], [5000, 4.48], [7500, 4.25]]),
        "b": st([[0, 10.8], [2500, 4.93], [5000, 4.42], [7500, 4.21]]),
        "c": st([[0, 10.8], [2500, 4.91], [5000, 4.43]], status="running"),  # not at 7500 yet
    }
    view = sweeps.summarize(record, statuses, {"a": {"usd": 1.5}, "b": {"usd": 1.6}, "c": None})
    assert [r["step"] for r in view["table"]] == [2500, 5000]  # 7500 isn't common yet; step 0 is skipped
    assert view["table"][0]["leader"] == "4e-4" and view["table"][1]["leader"] == "3e-4"
    assert view["table"][1]["gap"]["2e-4"] == pytest.approx(0.06)
    assert view["leader"] == "3e-4" and view["cost_usd"] == 3.1 and not view["done"]


def test_summary_flags_divergence():
    record = {
        "id": "s",
        "name": "s",
        "flag": "lr",
        "values": [1e-3, 1e-2],
        "stop_after": None,
        "target": "local",
        "created": "x",
        "children": [{"value": 1e-3, "name": "a", "key": "a"}, {"value": 1e-2, "name": "b", "key": "b"}],
    }
    statuses = {
        "a": {"status": "completed", "metrics": {"full_val_curve": [[100, 5.0]]}},
        "b": {"status": "completed", "metrics": {"full_val_curve": [[100, float("nan")]]}},
    }
    view = sweeps.summarize(record, statuses, {})
    assert view["children"][1]["diverged"] and view["table"][0]["leader"] == "1e-3"


def test_launcher_runs_several_job_files_in_order(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(launch_mod, "launch", lambda f, repo, uv: order.append(f.name) or (1 if f.name == "b" else 0))
    with pytest.raises(SystemExit) as exc:
        launch_mod.main(["a", "b", "c", "--repo", str(tmp_path)])
    assert order == ["a", "b", "c"] and exc.value.code == 1  # a failure doesn't stop the rest, but is reported
