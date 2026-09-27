"""M5 controller: pacing, budget, pauses, the data policy and the notebook (Modal and Claude faked)."""

import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from autolab import controller as ctl
from autolab import evaluate as ev
from autolab import modal_backend as mb
from autolab.config import REPO_ROOT
from autolab.llm import RateLimited
from autolab.program import Program, save

HP = {"n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.0, "batch_size": 64, "lr": 1.2e-3,
      "min_lr": 2e-6, "warmup_steps": 100, "weight_decay": 0.0}
T = datetime(2026, 9, 27, 20, 0, tzinfo=timezone.utc)


def rep(loss):
    return {"summary": {"final_full_val_loss": loss}, "health": {"nan_or_inf": False},
            "performance": {"tokens_per_sec": 45_000.0, "wall_s": 1800.0}, "scale": {"params": 16_105_297}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for i, (screen, full) in enumerate([(5.94, 4.76), (5.96, 4.78), (5.87, 4.72)], 1):
        for kind, loss in (("screen", screen), ("full", full)):
            (runs / f"{kind}{i}").mkdir(parents=True)
            (runs / f"{kind}{i}" / "report.json").write_text(json.dumps(rep(loss)))
            (runs / f"{kind}{i}" / "diagnosis.json").write_text(json.dumps({"labels": [
                {"name": "data_limited", "confidence": 0.85, "evidence": {"epochs": 3.99}, "suggestion": "data"}]}))
    root = tmp_path / "state" / "evolve"
    monkeypatch.setattr(ev, "STATE_ROOT", root)
    monkeypatch.setattr(ctl, "STATE", tmp_path / "state")
    monkeypatch.setattr(ctl, "RUNS", runs)
    monkeypatch.setattr(ctl, "CONTROL", tmp_path / "state" / "controller.json")
    monkeypatch.setattr(ctl, "NOTEBOOK", tmp_path / "notebook.jsonl")
    monkeypatch.setattr(ctl, "NOTEBOOK_MD", tmp_path / "NOTEBOOK.md")
    monkeypatch.setattr(ctl, "note", lambda event, **f: _note(tmp_path / "notebook.jsonl", event, **f))
    monkeypatch.setattr(ctl, "read_notebook", lambda: _read(tmp_path / "notebook.jsonl"))
    monkeypatch.setattr(ctl, "render_notebook_md", lambda e, s, p, out=tmp_path / "NOTEBOOK.md":
                        ctl.__dict__["_render"](e, s, p, out))
    calls = {}
    monkeypatch.setattr(mb, "load_calls", lambda: calls)
    monkeypatch.setattr(mb, "price_per_s", lambda gpu: 0.000222)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ev.init_session(head, HP, {"screen": ["screen1", "screen2", "screen3"], "full": ["full1", "full2", "full3"]},
                    paths=ev.Paths(root / "s2"), runs_dir=runs, name="s2",
                    budgets={"screen_tokens": 10_485_760, "full_tokens": 81_920_000, "eval": {}})
    ev.set_active_session("s2", root)
    monkeypatch.setattr(ev, "active_session_name", lambda root_=root: (root / "ACTIVE").read_text().strip())
    return {"tmp": tmp_path, "runs": runs, "calls": calls, "root": root}


ctl._render = ctl.render_notebook_md  # keep the real renderer reachable after monkeypatching


def _note(path, event, **fields):
    e = {"at": T.isoformat(), "event": event, **fields}
    with open(path, "a") as fh:
        fh.write(json.dumps(e) + "\n")
    return e


def _read(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def enable(**extra):
    ctl.save_control({"enabled": True, **extra})


class Gen:
    """Fake proposer: makes queued hparam children through the real propose()."""

    def __init__(self, error=None):
        self.n, self.error = 0, error

    def __call__(self, log):
        if self.error:
            raise self.error
        self.n += 1
        return ev.propose("p0", [], {"lr": 1e-3 + self.n * 1e-4}, f"idea {self.n}", created_by="fake")


def test_parse_reset_and_spend_window():
    t = datetime(2026, 9, 27, 19, 0, tzinfo=timezone.utc)
    assert ctl.parse_reset("limit · resets Sep 30 at 6pm (America/Chicago)", t) == datetime(2026, 9, 30, 23, 5, tzinfo=timezone.utc)
    assert ctl.parse_reset("rate limited", t) == t + timedelta(hours=1)
    calls = {"a": {"state": "finished", "usd": 1.0, "submitted_at": (t - timedelta(hours=2)).isoformat()},
             "b": {"state": "finished", "usd": 5.0, "submitted_at": (t - timedelta(hours=30)).isoformat()},
             "c": {"state": "pending", "usd_estimate": 0.8, "submitted_at": t.isoformat()}}
    llm = [{"at": (t - timedelta(hours=1)).isoformat(), "usd": 0.2}, {"at": (t - timedelta(days=2)).isoformat(), "usd": 9}]
    assert ctl.spend_24h(calls, llm, t) == pytest.approx({"modal_done": 1.0, "modal_pending": 0.8, "llm": 0.2, "total": 2.0})


def test_disabled_does_nothing(world):
    gen = Gen()
    s = ctl.step(log=lambda m: None, generate=gen, t=T)
    assert s["enabled"] is False and gen.n == 0


def test_proposes_up_to_in_flight(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 3, "daily_usd": 100.0})
    enable(data_flow={"session": "s2", "state": "not_helped"})
    gen = Gen()
    s = ctl.step(log=lambda m: None, generate=gen, t=T)
    assert s["proposed"] == 3 and gen.n == 3
    assert ctl.step(log=lambda m: None, generate=gen, t=T)["proposed"] == 0  # 3 still in flight
    events = [e["event"] for e in _read(world["tmp"] / "notebook.jsonl")]
    assert events.count("proposed") == 3
    assert "idea 1" in (world["tmp"] / "NOTEBOOK.md").read_text()


def test_daily_budget_stops_proposals(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 4, "daily_usd": 10.0})
    world["calls"]["x"] = {"state": "finished", "usd": 9.8, "submitted_at": T.isoformat()}
    enable(data_flow={"session": "s2", "state": "not_helped"})
    s = ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert s["proposed"] == 0 and "24h spend $9.80" in s["budget"]


def test_rate_limit_pauses_until_reset(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 100.0})
    enable(data_flow={"session": "s2", "state": "not_helped"})
    ctl.step(log=lambda m: None, generate=Gen(RateLimited("HTTP 429: limit · resets Sep 30 at 6pm (America/Chicago)")), t=T)
    c = ctl.load_control()
    assert c["paused_until"].startswith("2026-09-30T23:05")
    gen = Gen()
    assert "paused_until" in ctl.step(log=lambda m: None, generate=gen, t=T + timedelta(hours=1)) and gen.n == 0
    assert ctl.step(log=lambda m: None, generate=gen, t=datetime(2026, 10, 1, tzinfo=timezone.utc))["proposed"] > 0


def test_consecutive_early_rejects_pause(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 100.0, "max_consecutive_early_rejects": 3})
    paths = ev.Paths()
    for i in range(1, 4):
        save(Program(id=f"p{i}", parent_id="p0", base_commit="x", blocks={}, hparams=HP, stage="cpu",
                     status="rejected", reason="causal-leak: ..."), paths.programs)
    enable(data_flow={"session": "s2", "state": "not_helped"})
    s = ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert "paused_until" in s and "in a row rejected before training" in ctl.load_control()["pause_reason"]
    assert [e["event"] for e in _read(world["tmp"] / "notebook.jsonl")].count("program_done") == 3


def test_accepted_program_is_recorded_and_committed_once(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 0.0})
    committed = []
    monkeypatch.setattr(ctl, "commit_accepted", lambda p, s, log=print: committed.append(p.id) or "abc123")
    save(Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams=HP, stage="done", status="accepted",
                 scores={"full_mean": 4.6, "n_seeds": 3}, reason="beat the bar"), ev.Paths().programs)
    enable(data_flow={"session": "s2", "state": "not_helped"})
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert committed == ["p1"]
    assert ev.programs()["p1"].meta["commit"] == "abc123"


def test_data_policy_end_to_end(world, monkeypatch):
    """data_limited incumbent -> build -> upload -> data check -> helped -> screens -> new session."""
    from autolab import config as acfg

    datasets = world["tmp"] / "datasets"
    (datasets / "data20k").mkdir(parents=True)
    (datasets / "data20k" / "dataset.json").write_text(json.dumps({"docs": 20000}))

    class Cfg:
        datasets_dir = datasets
    monkeypatch.setattr(acfg, "load_config", lambda: Cfg)
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 0.0, "data_trigger_confidence": 0.75})
    spawned, alive = [], {"v": True}
    monkeypatch.setattr(ctl, "_spawn", lambda args, logname: spawned.append(args) or 4242)
    monkeypatch.setattr(ctl, "_alive", lambda pid: alive["v"])
    started = []

    def fake_check(pid, dataset, paths=None):
        s = ev.load_session(paths)
        chk = {"id": f"{dataset}-{pid}", "program": pid, "dataset": dataset, "runs": ["d1", "d2", "d3"],
               "baseline_mean": 4.752, "status": "running"}
        s.setdefault("data_checks", []).append(chk)
        ev.save_session(s, paths)
        started.append(dataset)
        return chk
    monkeypatch.setattr(ev, "start_data_check", fake_check)
    screens = []
    monkeypatch.setattr(ctl, "submit_screens", lambda flow, s, p, paths, log: (
        flow.update(state="baselining", screen_runs=["b1", "b2", "b3"]), screens.append(1)))
    enable()

    def step():
        ctl.step(log=lambda m: None, generate=Gen(), t=T)
        return ctl.load_control()["data_flow"]

    assert step()["state"] == "building" and spawned[-1] == ["data", "build", "--num-examples", "40000"]
    alive["v"] = False
    (datasets / "data40k").mkdir()
    (datasets / "data40k" / "dataset.json").write_text(json.dumps({"docs": 40000}))
    (datasets / "data40k" / "train.pt").write_bytes(b"x")
    assert step()["state"] == "uploading" and spawned[-1] == ["modal", "upload-data"]
    assert step()["state"] == "checking" and started == ["data40k"]
    s = ev.load_session()  # the daemon's judge says it helped
    s["data_checks"][0].update(status="done", helped=True, verdict="helped: 4.60 vs 4.752 (-4.6σ)")
    ev.save_session(s)
    assert step()["state"] == "baselining" and screens
    for rid, loss in [("d1", 4.60), ("d2", 4.62), ("d3", 4.58), ("b1", 5.8), ("b2", 5.82), ("b3", 5.79)]:
        (world["runs"] / rid).mkdir()
        (world["runs"] / rid / "report.json").write_text(json.dumps(rep(loss)))
        world["calls"][rid] = {"state": "finished"}
    monkeypatch.setattr(ev, "_report", lambda rid, runs_dir: json.loads((world["runs"] / rid / "report.json").read_text()))
    flow = step()
    assert flow["state"] == "switched" and flow["new_session"] == "s2+data40k"
    new = ev.load_session()
    assert new["name"] == "s2+data40k" and new["dataset_id"] == "data40k"
    assert new["noise"]["full"]["mean"] == pytest.approx(4.60)
    events = [e["event"] for e in _read(world["tmp"] / "notebook.jsonl")]
    assert events[:4] == ["data_limited", "data_check_started", "data_check_done", "session_switched"]


def test_data_policy_prefers_slicing_a_bigger_set(world, monkeypatch):
    from autolab import config as acfg

    datasets = world["tmp"] / "datasets"
    for name, docs in (("data20k", 20000), ("data80k", 80000)):
        (datasets / name).mkdir(parents=True)
        (datasets / name / "dataset.json").write_text(json.dumps({"docs": docs}))

    class Cfg:
        datasets_dir = datasets
    monkeypatch.setattr(acfg, "load_config", lambda: Cfg)
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 0.0})
    spawned = []
    monkeypatch.setattr(ctl, "_spawn", lambda args, logname: spawned.append(args) or 1)
    enable()
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert spawned == [["data", "slice", "data80k", "--docs", "40000"]]


def test_commit_accepted_in_a_throwaway_clone(tmp_path):
    """Commits the accepted program's code into its own worktree/branch (on a clone, never the real repo)."""
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(REPO_ROOT), str(clone)], check=True)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    from autolab.program import base_sources, extract_blocks

    blocks = extract_blocks(base_sources(clone, head))
    blocks["mini_llm/train.py:optimizer"] = blocks["mini_llm/train.py:optimizer"].replace(
        "No-op by default.\"\"\"", "No-op by default.\"\"\"\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
    p = Program(id="p7", parent_id="p3", base_commit=head, blocks=blocks, hparams=HP, status="accepted",
                scores={"full_mean": 4.61, "n_seeds": 3}, rationale="clip grads", reason="beat the bar")
    wt = tmp_path / "wt" / "accepted"
    sha = ctl.commit_accepted(p, {"name": "s2", "base_commit": head}, repo=clone, worktree=wt, log=lambda m: None)
    assert sha
    log = subprocess.run(["git", "-C", str(wt), "log", "-1", "--format=%s%n%b"], capture_output=True, text=True).stdout
    assert "accept s2/p7" in log and "Hypothesis: clip grads" in log
    assert "clip_grad_norm_" in (wt / "src" / "mini_llm" / "train.py").read_text()
    assert json.loads((wt / "autolab" / "accepted" / "s2-p7.json").read_text())["scores"]["full_mean"] == 4.61
