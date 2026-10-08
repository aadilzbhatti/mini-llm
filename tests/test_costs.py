"""Modal run costs: list prices, the server's estimate, billed totals from the mirror."""

import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mini_llm.remote import costs
from mini_llm.remote.modal_mirror import load_runner, mirror_run
from mini_llm.server import create_app

REPO = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 7, 21, 30, tzinfo=timezone.utc).timestamp()


def z(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeWorkspace:
    """modal.Workspace's billing.rates() / billing.report() with canned data."""

    def __init__(self, rows=(), rates=None):
        self.rows, self.reports = list(rows), []
        self.billing = SimpleNamespace(rates=lambda: rates or {}, report=self.report)

    def report(self, *, start, end, resolution):
        self.reports.append((start, end, resolution))
        return [r for r in self.rows if start <= r.interval_start < end]


def row(app_id, hour, cost):
    start = datetime(2026, 10, 7, hour, tzinfo=timezone.utc)
    return SimpleNamespace(object_id=app_id, interval_start=start, cost=cost)


def modal_status(run_id="r1", state="running", started=NOW - 3 * 3600, **remote):
    return {
        "run_id": run_id,
        "name": run_id,
        "kind": "train",
        "status": state,
        "started": z(started),
        "remote": {"provider": "modal", "gpus": "L4:2", **remote},
    }


@pytest.mark.parametrize(
    "gpus, expected",
    [("L4:2", 1.60), ("H100", 3.95), ("A100-80GB:4", 10.0), ("a100", 2.10), ("H100!:2", 7.90), ("nope:2", None)],
)
def test_list_hourly_from_fallback_prices(gpus, expected):
    assert costs.list_hourly(gpus) == expected


def test_list_hourly_prefers_fetched_rates_and_prices_cpu_runs():
    rates = {"gpu_hour_cost_l4": 1.0, "cpu_hour_cost": 0.05}
    assert costs.list_hourly("L4:2", rates) == 2.0
    assert costs.list_hourly("cpu", rates) == 0.2  # 4 cores


def test_running_cost_is_elapsed_times_list_price_until_billed():
    c = costs.run_cost(modal_status(), {"eta_sec": 3600}, {}, NOW)
    assert c["usd"] == 4.80 and c["rate_source"] == "list" and c["source"] == "estimate"
    assert c["projected_usd"] == 6.40 and c["billed_usd"] is None


def test_running_cost_uses_the_billed_hourly_rate_once_known():
    st = modal_status(cost={"billed_usd": 4.6, "hourly_usd": 1.78, "updated": z(NOW), "final": False})
    c = costs.run_cost(st, {"eta_sec": 1800, "paused": False}, {}, NOW)
    assert c["usd"] == 5.34 and c["rate_source"] == "billed" and c["billed_usd"] == 4.6
    assert c["projected_usd"] == 6.23


def test_finished_cost_is_billed_once_final_else_duration_estimate():
    st = modal_status(state="completed", cost={"billed_usd": 9.75, "hourly_usd": 1.78, "final": True})
    st["duration_sec"] = 19728.6
    assert costs.run_cost(st, None, {}, NOW)["usd"] == 9.75
    assert costs.run_cost(st, None, {}, NOW)["source"] == "billed"
    st["remote"].pop("cost")
    c = costs.run_cost(st, None, {}, NOW)
    assert c["usd"] == 8.77 and c["source"] == "estimate"  # 5.48 h x $1.60


def test_interrupted_cost_stops_at_the_last_heartbeat():
    st = modal_status(state="interrupted", started=NOW - 10 * 3600)
    st["duration_sec"] = 10 * 3600  # the mirror's count, still running on
    assert costs.run_cost(st, {"updated": z(NOW - 9 * 3600)}, {}, NOW)["usd"] == 1.60
    assert costs.run_cost(st, None, {}, NOW)["usd"] is None


def test_local_runs_have_no_cost():
    assert costs.run_cost({"status": "running", "started": z(NOW)}, None, {}, NOW) is None


def test_fetch_billed_sums_hours_and_averages_full_ones():
    ws = FakeWorkspace([row("ap-a", 18, 1.11), row("ap-a", 19, 1.77), row("ap-a", 20, 1.79), row("ap-a", 21, 0.65)])
    ws.rows.append(row("ap-other", 19, 9.0))
    got = costs.fetch_billed({"ap-a": NOW - 3.2 * 3600}, ws, NOW)
    assert got == {"ap-a": {"billed_usd": 5.32, "hourly_usd": 1.78}}
    start, end, resolution = ws.reports[0]
    # From an hour before the start through the end of the current (partial) hour.
    assert start <= datetime(2026, 10, 7, 18, tzinfo=timezone.utc) and end == datetime(
        2026, 10, 7, 22, tzinfo=timezone.utc
    )
    assert resolution == "h"


def test_fetch_billed_has_no_hourly_rate_before_a_full_hour():
    ws = FakeWorkspace([row("ap-a", 21, 0.4)])
    assert costs.fetch_billed({"ap-a": NOW - 600}, ws, NOW) == {"ap-a": {"billed_usd": 0.4, "hourly_usd": None}}


def test_update_costs_writes_billed_and_marks_settled_runs_final(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "live.status.json").write_text(json.dumps(modal_status("live", app_id="ap-a")))
    done = modal_status("done", state="completed", started=NOW - 6 * 3600, app_id="ap-b")
    done["finished"] = z(NOW - 3 * 3600)
    (runs / "done.status.json").write_text(json.dumps(done))
    (runs / "old.status.json").write_text(json.dumps(modal_status("old")))  # no app id: skipped
    ws = FakeWorkspace([row("ap-a", 19, 1.0), row("ap-a", 20, 1.78), row("ap-a", 21, 0.5), row("ap-b", 16, 2.0)])

    assert sorted(costs.update_costs(tmp_path, ws, NOW)) == ["done", "live"]
    live = json.loads((runs / "live.status.json").read_text())["remote"]["cost"]
    assert live == {"billed_usd": 3.28, "hourly_usd": 1.78, "updated": z(NOW), "final": False}
    assert json.loads((runs / "done.status.json").read_text())["remote"]["cost"]["final"] is True
    assert "cost" not in json.loads((runs / "old.status.json").read_text())["remote"]

    assert costs.update_costs(tmp_path, ws, NOW) == ["live"]  # the settled one is never asked about again


def test_final_runs_are_not_queried_again(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    st = modal_status("done", state="completed", app_id="ap-b", cost={"billed_usd": 2.0, "final": True})
    (runs / "done.status.json").write_text(json.dumps(st))
    ws = FakeWorkspace()
    assert costs.update_costs(tmp_path, ws, NOW) == [] and ws.reports == []


def test_rates_round_trip_through_runs_dir(tmp_path):
    (tmp_path / "runs").mkdir()
    ws = FakeWorkspace(rates={"gpu_hour_cost_l4": 0.9, "cpu_hour_cost": 0.05, "egress_gib_cost": 0.04})
    costs.write_rates(tmp_path, costs.fetch_rates(ws), NOW)
    assert costs.load_rates(tmp_path / "runs") == {"gpu_hour_cost_l4": 0.9, "cpu_hour_cost": 0.05}
    assert costs.load_rates(tmp_path / "missing") == {}


# --- through the mirror and the server -----------------------------------------


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "repo" / "runner").mkdir(parents=True)
    shutil.copy(REPO / "runner" / "run_queue.py", tmp_path / "repo" / "runner" / "run_queue.py")
    (tmp_path / "repo" / "queue").mkdir()
    return tmp_path / "repo"


class FakeVolume:
    def __init__(self, root: Path):
        self.root = root

    def listdir(self, path, recursive=False):
        base = self.root / path.strip("/")
        return [
            SimpleNamespace(
                path=str(p.relative_to(self.root)),
                type=1 if p.is_file() else 2,
                size=p.stat().st_size if p.is_file() else 0,
                mtime=int(p.stat().st_mtime),
            )
            for p in (base.rglob("*") if recursive else base.iterdir())
        ]

    def read_file(self, path):
        yield (self.root / path).read_bytes()


def test_mirror_keeps_app_id_and_cost_and_server_serves_it(tmp_path, repo):
    run = tmp_path / "vol" / "r1"
    (run / "runs").mkdir(parents=True)
    started = time.time() - 2 * 3600
    record = {
        "run_id": "r1",
        "config": {"name": "r1", "args": {}},
        "gpus": "L4:2",
        "modal_app_id": "ap-xyz",
        "resolved_argv": [],
        "command": "torchrun ...",
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
    }
    (run / "run.json").write_text(json.dumps(record))
    (run / "train.log").write_text("step     0 | loss 4.0000 | lr 1.00e-03\n")
    (run / "runs" / "r1.live.json").write_text(
        json.dumps({"step": 10, "total_steps": 20, "eta_sec": 3600, "updated": z(time.time())})
    )
    vol, runner = FakeVolume(tmp_path / "vol"), load_runner(repo)

    st = mirror_run(vol, "r1", repo, runner)
    assert st["remote"]["app_id"] == "ap-xyz"
    hour0 = datetime.fromtimestamp(started, timezone.utc).replace(minute=0, second=0, microsecond=0)
    ws = FakeWorkspace(  # billed hours covering the run so far, relative to now (not a fixed date)
        [SimpleNamespace(object_id="ap-xyz", interval_start=hour0 + timedelta(hours=h), cost=1.78) for h in range(4)]
    )
    costs.update_costs(repo, ws)
    st = mirror_run(vol, "r1", repo, runner)  # the next poll rewrites the status
    assert st["remote"]["cost"]["hourly_usd"] == 1.78

    c = TestClient(create_app(repo=repo, uv="uv"))
    [listed] = c.get("/api/runs").json()
    assert listed["cost"]["rate_source"] == "billed" and abs(listed["cost"]["usd"] - 3.56) < 0.05
    assert listed["cost"]["projected_usd"] == pytest.approx(listed["cost"]["usd"] + 1.78, abs=0.01)
    assert c.get("/api/runs/r1").json()["cost"]["hourly_usd"] == 1.78


def test_modal_launch_preview_shows_the_hourly_price(repo, monkeypatch):
    (repo / "data").mkdir()
    for name in ("train.pt", "val.pt"):
        (repo / "data" / name).write_bytes(b"")
    c = TestClient(create_app(repo=repo, uv="uv"))
    resp = c.post(
        "/api/jobs?dry_run=true",
        json={"name": "t", "kind": "train", "target": "modal", "gpus": "H100:2", "args": {"batch-size": 8}},
    )
    if resp.status_code == 503:
        pytest.skip("modal package not installed")
    assert resp.status_code == 200, resp.text
    assert resp.json()["hourly_usd"] == 7.90
