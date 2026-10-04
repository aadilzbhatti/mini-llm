"""Dashboard API over a fabricated autolab/ tree."""

import json

import pytest
from fastapi.testclient import TestClient

import autolab.dashboard as dash


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "autolab"
    (root / "state").mkdir(parents=True)
    (root / "experiments").mkdir()
    runs = {}
    exp = {"id": "e1", "title": "t", "purpose": "p", "gpu": "L4", "dataset_id": "d",
           "model": {"n_embd": 8}, "optim": {"lr": 1e-3}, "eval": {},
           "analysis": {"kind": "noise_and_ranking", "baseline_variant": "base"}, "jobs": []}
    vals = {("base", "screen"): [6.50, 6.52, 6.54], ("base", "full"): [5.00, 5.02, 5.04],
            ("a", "screen"): [6.40], ("a", "full"): [4.90], ("b", "screen"): [6.60], ("b", "full"): [5.10],
            ("c", "screen"): [6.70], ("c", "full"): [5.20]}
    i = 0
    for (variant, budget), xs in vals.items():
        for x in xs:
            i += 1
            rid = f"r{i}"
            exp["jobs"].append({"run_id": rid, "variant": variant, "budget": budget})
            runs[rid] = {"call_id": "fc", "state": "finished", "gpu": "L4", "usd": 0.1, "usd_estimate": 0.2,
                         "submitted_at": "2026-09-27T00:00:00+00:00", "request": {"seed": i, "budget": {}}}
            d = root / "runs" / rid
            d.mkdir(parents=True)
            (d / "report.json").write_text(json.dumps({
                "summary": {"final_full_val_loss": x}, "scale": {}, "performance": {"wall_s": 60},
                "identity": {"end_time": f"2026-09-27T00:{i:02d}:00+00:00"}, "health": {}}))
    runs["live1"] = {"call_id": "fc", "state": "pending", "gpu": "L4", "usd_estimate": 0.5, "request": {}}
    (root / "runs" / "live1").mkdir()
    (root / "runs" / "live1" / "live.json").write_text(json.dumps({"step": 5, "total_steps": 10,
                                                                    "updated": "2026-09-27T00:00:00Z"}))
    (root / "state" / "modal_calls.json").write_text(json.dumps(runs))
    (root / "experiments" / "e1.json").write_text(json.dumps(exp))
    (root / "HANDOFF.md").write_text("## Where things stand\n- a `x`\n  more\n## Decision log\n- 2026-09-26 (M2): one\n  two\n")
    (root / "config.toml").write_text("[modal]\nmax_usd = 5.0\n")
    monkeypatch.setattr(dash, "AUTOLAB", root)
    return TestClient(dash.app)


def test_overview(client):
    d = client.get("/api/overview").json()
    assert d["counts"] == {"pending": 1, "finished": 12, "failed": 0}
    assert d["spend"]["spent"] == pytest.approx(1.2) and d["spend"]["pending_estimate"] == pytest.approx(0.5)
    assert d["spend"]["cap"] == 5.0 and len(d["spend"]["timeline"]) == 12
    assert d["best"]["full_val"] == 4.90
    a = d["experiments"][0]["analysis"]
    assert a["noise"]["full"]["n"] == 3 and a["noise"]["full"]["std"] == pytest.approx(0.02)
    assert a["spearman"] == pytest.approx(1.0)  # screens rank exactly like full runs
    assert [r["variant"] for r in a["ranking"]][:2] == ["a", "base"]
    live = next(r for r in d["runs"] if r["run_id"] == "live1")
    assert live["live"]["step"] == 5
    assert d["decisions"][0]["text"] == "one two" and d["status"] == ["a `x` more"]


def test_run_detail_and_safety(client):
    assert client.get("/api/run/r1").json()["report"]["summary"]["final_full_val_loss"] == 6.50
    assert client.get("/api/run/nope").status_code == 404
    assert client.get("/api/run/..%2F..%2Fetc").status_code in (400, 404)
    assert "autolab" in client.get("/").text


def test_spearman():
    assert dash.spearman({"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1}) == pytest.approx(-1.0)
    assert dash.spearman({"a": 1, "b": 2}, {"a": 1, "b": 2}) is None


def test_page_script_parses(tmp_path):
    """A syntax error blanks the whole page; catch it here (skipped without node)."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    html = (dash.STATIC / "dashboard.html").read_text()
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", html, re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_budget_controls(client, tmp_path, monkeypatch):
    from autolab import controller as ctl

    monkeypatch.setattr(ctl, "CONTROL", tmp_path / "controller.json")
    monkeypatch.setattr(ctl, "NOTEBOOK", tmp_path / "notebook.jsonl")
    cfg = dash.AUTOLAB / "config.toml"
    cfg.write_text("[modal]\nmax_usd = 50.0              # cap\n\n[controller]\ndaily_usd = 10.0   # per day\nmax_in_flight = 6\n")
    h = {"X-Autolab": "1"}
    assert client.post("/api/budget", json={"action": "override", "daily_usd": 30, "hours": 12}).status_code == 403
    r = client.post("/api/budget", json={"action": "override", "daily_usd": 30, "hours": 12}, headers=h)
    assert r.status_code == 200 and ctl.load_control()["daily_usd_override"]["usd"] == 30
    assert client.post("/api/budget", json={"action": "override", "daily_usd": 5000, "hours": 1}, headers=h).status_code == 400
    assert client.post("/api/budget", json={"action": "clear_override"}, headers=h).status_code == 200
    assert "daily_usd_override" not in ctl.load_control()
    r = client.post("/api/budget", json={"action": "permanent", "daily_usd": 15, "max_usd": 80}, headers=h)
    assert r.status_code == 200, r.text
    import tomllib

    got = tomllib.loads(cfg.read_text())
    assert got["controller"]["daily_usd"] == 15.0 and got["modal"]["max_usd"] == 80.0 and got["controller"]["max_in_flight"] == 6
    assert "# cap" in cfg.read_text() and "# per day" in cfg.read_text()  # comments kept
    events = [json.loads(x)["event"] for x in (tmp_path / "notebook.jsonl").read_text().splitlines()]
    assert events == ["budget_override", "budget_override_cleared", "settings_changed"]

    # pause / unpause: a hold the daemon obeys, idempotent, noted once each
    assert client.post("/api/budget", json={"action": "pause"}).status_code == 403
    assert client.post("/api/budget", json={"action": "pause"}, headers=h).status_code == 200
    at = ctl.load_control()["hold"]["at"]
    assert client.post("/api/budget", json={"action": "pause"}, headers=h).status_code == 200
    assert ctl.load_control()["hold"] == {"at": at, "by": "dashboard", "reason": ""}
    assert client.post("/api/budget", json={"action": "unpause"}, headers=h).status_code == 200
    assert "hold" not in ctl.load_control()
    events = [json.loads(x)["event"] for x in (tmp_path / "notebook.jsonl").read_text().splitlines()]
    assert events[-2:] == ["held", "released"]

    # ceilings: the token cap and the parameter cap can be raised from the page, within bounds
    cfg.write_text("[evolve]\nparam_cap_mult = 2.0   # cap\n\n[controller]\nmax_full_tokens = 184320000\n")
    r = client.post("/api/budget", json={"action": "permanent", "max_full_tokens": 276480000, "param_cap_mult": 2.5},
                    headers=h)
    assert r.status_code == 200, r.text
    got = tomllib.loads(cfg.read_text())
    assert got["controller"]["max_full_tokens"] == 276480000 and got["evolve"]["param_cap_mult"] == 2.5
    assert client.post("/api/budget", json={"action": "permanent", "param_cap_mult": 9}, headers=h).status_code == 400
