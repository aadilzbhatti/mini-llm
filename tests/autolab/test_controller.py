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
    monkeypatch.setattr(ctl, "measure_step", lambda *a, **k: None)  # tested on its own below
    monkeypatch.setattr(mb, "price_per_s", lambda gpu: 0.000222)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ev.init_session(head, HP, {"screen": ["screen1", "screen2", "screen3"], "full": ["full1", "full2", "full3"]},
                    paths=ev.Paths(root / "s2"), runs_dir=runs, name="s2",
                    budgets={"screen_tokens": 10_485_760, "full_tokens": 81_920_000, "eval": {}})
    ev.set_active_session("s2", root)
    monkeypatch.setattr(ev, "active_session_name", lambda root_=root: (root / "ACTIVE").read_text().strip())
    return {"tmp": tmp_path, "runs": runs, "calls": calls, "root": root}


ctl._render = ctl.render_notebook_md  # keep the real renderer reachable after monkeypatching
ctl._real_measure_step = ctl.measure_step


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
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 100.0, "max_in_flight": 0,
                                                        "data_trigger_confidence": 0.75})
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
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 100.0, "max_in_flight": 0})
    spawned = []
    monkeypatch.setattr(ctl, "_spawn", lambda args, logname: spawned.append(args) or 1)
    enable()
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert spawned == [["data", "slice", "data80k", "--docs", "40000"]]


@pytest.mark.real_commit
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


def test_old_session_programs_still_advance_and_accepted_ones_are_ported(world, monkeypatch):
    """A data switch mid-evaluation: the old session's programs are still judged, and an
    acceptance there is re-evaluated in the new session."""
    root = world["root"]
    old = ev.Paths(root / "s2")
    head = ev.load_session(old)["base_commit"]
    ev.init_session(head, HP, {"screen": ["screen1", "screen2"], "full": ["full1", "full2"]},
                    paths=ev.Paths(root / "s3"), runs_dir=world["runs"], name="s3",
                    budgets={"screen_tokens": 1, "full_tokens": 2, "eval": {}}, dataset_id="data40k")
    ev.set_active_session("s3", root)
    assert [p.root.name for p in ev.all_session_paths()] == ["s2", "s3"]  # active last
    save(Program(id="p1", parent_id="p0", base_commit=head, blocks={}, hparams={**HP, "lr": 2e-3}, stage="done",
                 status="accepted", scores={"full_mean": 4.66, "n_seeds": 3}, rationale="higher lr"), old.programs)
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 0.0})
    enable(data_flow={"session": "s3", "state": "not_helped"})
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    ctl.step(log=lambda m: None, generate=Gen(), t=T)  # ported once only
    new = ev.programs(ev.Paths(root / "s3"))
    ported = [p for p in new.values() if p.created_by.startswith("port:")]
    assert len(ported) == 1 and ported[0].created_by == "port:s2/p1" and ported[0].hparams["lr"] == 2e-3
    assert ported[0].status == "queued" and ported[0].parent_id == "p0"


def test_data_policy_waits_for_budget(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 1.0, "data_trigger_confidence": 0.75})
    spawned = []
    monkeypatch.setattr(ctl, "_spawn", lambda args, logname: spawned.append(args) or 1)
    enable()
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    flow = ctl.load_control()["data_flow"]
    assert flow["state"] == "idle" and flow["waiting_for_budget"] and not spawned


def test_session_report(world, monkeypatch, tmp_path):
    from autolab import session_report

    monkeypatch.setattr(session_report, "REPO_ROOT", tmp_path)  # notebook / llm spend / runs lookups
    paths = ev.Paths()
    save(Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams={**HP, "lr": 2e-3}, stage="done",
                 status="accepted", scores={"full_mean": 4.60, "n_seeds": 3}, rationale="higher\nlr",
                 stages=[{"stage": "full", "ok": True, "at": "2026-09-27T20:00:00", "loss": 4.61}]), paths.programs)
    s = ev.load_session(paths)
    s["incumbent"] = "p1"
    s["incumbent_history"].append({"at": "2026-09-27T21:00:00", "program": "p1", "full_mean": 4.60})
    ev.save_session(s, paths)
    text = session_report.build(out=tmp_path / "S.md")
    assert "Best: **s2/p1**, full val loss **4.6000**" in text
    assert "Start at the same budget: `s2` p0 4.7533" in text
    assert "| 1 | 2026-09-27T20:00 | s2/p1" in text and "higher lr" in text


def test_one_time_budget_override(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 2, "daily_usd": 10.0})
    world["calls"]["x"] = {"state": "finished", "usd": 9.8, "submitted_at": T.isoformat()}
    enable(data_flow={"session": "s2", "state": "not_helped"},
           daily_usd_override={"usd": 20.0, "until": (T + timedelta(hours=24)).isoformat()})
    assert ctl.step(log=lambda m: None, generate=Gen(), t=T)["proposed"] == 2
    ctl.save_control({**ctl.load_control(), "daily_usd_override": {"usd": 20.0, "until": (T - timedelta(hours=1)).isoformat()}})
    assert ctl.daily_budget(ctl.load_control(), {"daily_usd": 10.0}, T) == 10.0  # expired


def test_compute_ladder(world, monkeypatch):
    """Stalled search -> incumbent at 1.5x tokens (3 seeds, scaled caps) -> helped -> screens -> new session."""
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"daily_usd": 100.0, "max_in_flight": 0, "ladder_patience": 2,
                                                        "ladder_factor": 1.5, "max_full_tokens": 122_880_000})
    monkeypatch.setattr(ev, "evolve_cfg", lambda: {**ev.tomllib.loads((REPO_ROOT / "autolab" / "config.toml").read_text())["evolve"]})
    from autolab import config as acfg

    datasets = world["tmp"] / "datasets"
    (datasets / "data20k").mkdir(parents=True)
    (datasets / "data20k" / "train.pt").write_bytes(b"x")

    class Cfg:
        datasets_dir = datasets
    monkeypatch.setattr(acfg, "load_config", lambda: Cfg)
    paths = ev.Paths()
    s = ev.load_session(paths)
    s["dataset_id"] = "data20k"
    ev.save_session(s, paths)
    enable(data_flow={"session": "s2", "state": "not_helped"})
    submitted = []
    import autolab.modal_backend as mbm
    monkeypatch.setattr(mbm, "submit", lambda req, gpu, src_root=None: submitted.append(req))

    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert ctl.load_control()["ladder"]["state"] == "idle"  # nothing stalled yet
    for i in (1, 2):
        save(Program(id=f"p{i}", parent_id="p0", base_commit="x", blocks={}, hparams=HP, stage="done",
                     status="evaluated", scores={"full_mean": 4.8}, created_at="2099-01-01T00:00:00+00:00"), paths.programs)
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    lad = ctl.load_control()["ladder"]
    assert lad["state"] == "checking" and lad["tokens"] == 15_000 * 8192
    assert [r.budget.tokens for r in submitted] == [122_880_000] * 3
    cap = ev.load_session(paths)["wall_caps"]["full"]
    assert submitted[0].budget.wall_clock_s == round(cap * 1.5)

    s = ev.load_session(paths)
    s["data_checks"][0].update(status="done", helped=True, verdict="helped: 4.60 vs 4.753")
    ev.save_session(s, paths)
    monkeypatch.setattr(ctl, "submit_screens", lambda flow, s, p, paths, log: flow.update(
        state="baselining", screen_runs=["b1", "b2", "b3"]))
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert ctl.load_control()["ladder"]["state"] == "baselining"
    for rid, loss in [(submitted[0].run_id, 4.60), (submitted[1].run_id, 4.62), (submitted[2].run_id, 4.61),
                      ("b1", 5.8), ("b2", 5.82), ("b3", 5.79)]:
        (world["runs"] / rid).mkdir(exist_ok=True)
        (world["runs"] / rid / "report.json").write_text(json.dumps(rep(loss)))
        world["calls"][rid] = {"state": "finished"}
    monkeypatch.setattr(ev, "_report", lambda rid, runs_dir: json.loads((world["runs"] / rid / "report.json").read_text()))
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    new = ev.load_session()
    assert new["name"] == "s2@122M" and new["budgets"]["full_tokens"] == 122_880_000
    assert new["noise"]["full"]["mean"] == pytest.approx(4.61)
    # at the owner's max now: no further rung
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert ctl.load_control()["ladder"].get("state") == "idle"


def test_blocked_status_says_how_to_unblock(world, monkeypatch):
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 4, "daily_usd": 10.0})
    world["calls"]["old"] = {"state": "finished", "usd": 6.0, "submitted_at": (T - timedelta(hours=20)).isoformat()}
    world["calls"]["new"] = {"state": "finished", "usd": 3.9, "submitted_at": (T - timedelta(hours=1)).isoformat()}
    enable(data_flow={"session": "s2", "state": "not_helped"})
    s = ctl.step(log=lambda m: None, generate=Gen(), t=T)
    b = s["blocked"]
    assert b["kind"] == "daily" and b["spend_24h"] == 9.9 and b["unblock_at_limit"] > 9.9 + b["need"]
    assert b["frees_at"] == (T - timedelta(hours=20) + timedelta(hours=24)).isoformat(timespec="seconds")


def test_sigma_floor():
    s = {"noise": {"full": {"std": 0.0069}}}
    assert ev.sigma(s, {"noise_floor": 0.02}) == 0.02 and ev.sigma(s, {"noise_floor": 0.0}) == 0.0069


def test_research_audits_new_winner_within_its_share(world, monkeypatch):
    from autolab import research

    monkeypatch.setattr(research, "research_cfg", lambda: {"enabled": True, "budget_share": 0.3, "est_usd_per_run": 1.5,
                                                           "stall_patience": 6})
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 0, "daily_usd": 10.0})
    spawned = []
    monkeypatch.setattr(ctl, "_spawn", lambda args, logname: spawned.append(args) or 999)
    enable(data_flow={"session": "s2", "state": "not_helped"})
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert spawned and spawned[0][:2] == ["research", "run"] and ctl.load_control()["research"]["state"] == "running"
    # finished: its cards become directed proposals, first in line
    monkeypatch.setattr(ctl, "_alive", lambda pid: False)
    (world["tmp"] / "state" / "research_result.json").write_text(json.dumps({"added": ["c1", "c2"], "cost_usd": 1.2}))
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    r = ctl.load_control()["research"]
    assert r["state"] == "idle" and r["pending_directed"] == ["c1", "c2"]
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert len(spawned) == 1  # same incumbent, already audited: no second run
    # over its share: waits
    monkeypatch.setattr(ctl, "_llm_spend", lambda: [{"at": T.isoformat(), "usd": 2.5, "tag": "research-s2-p0"}])
    s = ev.load_session()
    s["incumbent"] = "p0"
    c = ctl.load_control()
    c["research"]["audited"] = []
    ctl.save_control(c)
    ctl.step(log=lambda m: None, generate=Gen(), t=T)
    assert "research budget" in ctl.load_control()["research"]["waiting"] and len(spawned) == 1



def test_measure_step_backfills_incumbent_metrics(world, monkeypatch):
    """The quality champion without four-dimensional metrics gets one measured run (budget permitting)."""
    import importlib

    real = importlib.reload(ctl).measure_step if False else ctl.__dict__.get("_real_measure_step")
    submitted = []

    def fake_start(pid, paths=None):
        submitted.append(pid)
        s = ev.load_session(paths)
        s.setdefault("measures", {})[pid] = f"ev-s2-{pid}-measure-s1"
        ev.save_session(s, paths)
        return f"ev-s2-{pid}-measure-s1"

    monkeypatch.setattr(ev, "start_measure", fake_start)
    monkeypatch.setattr(ctl, "controller_cfg", lambda: {"max_in_flight": 0, "daily_usd": 100.0})
    enable(data_flow={"session": "s2", "state": "not_helped"})
    ctl._real_measure_step(ctl.load_control(), ev.load_session(), ev.programs(), {}, {"daily_usd": 100.0}, T,
                           ev.Paths(), lambda m: None)
    assert submitted == ["p0"]
    ctl._real_measure_step(ctl.load_control(), ev.load_session(), ev.programs(), {}, {"daily_usd": 100.0}, T,
                           ev.Paths(), lambda m: None)
    assert submitted == ["p0"]  # already measuring: not again


def test_noise_is_pooled_over_multi_seed_programs(world):
    paths = ev.Paths(world["root"] / "s2")
    session = ev.load_session(paths)
    base = session["noise"]["full"]["std"]
    p = Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams=HP, status="contender", stage="done",
                scores={"full_losses": [4.70, 4.70, 4.70]})
    q = Program(id="p2", parent_id="p0", base_commit="x", blocks={}, hparams=HP, status="evaluated", stage="done",
                scores={"full_losses": [4.80]})  # one seed: no information about noise
    assert ev.update_noise_pool(session, {"p0": ev.programs(paths)["p0"], "p1": p, "p2": q})
    assert session["noise"]["full"]["programs"] == {"p1": [4.70, 4.70, 4.70]}
    std, df = ev.pooled_full_std(session)
    assert df == 4 and std == pytest.approx(base / 2 ** 0.5)  # a zero-spread program halves the variance
    assert not ev.update_noise_pool(session, {"p1": p})
    assert ev.sigma(session, {"noise_floor": 0.0}) == pytest.approx(std)


@pytest.mark.parametrize("flow_key,step", [("data_flow", "data_step"), ("ladder", "ladder_step")])
def test_policies_rearm_for_a_new_incumbent(world, monkeypatch, flow_key, step):
    paths = ev.Paths(world["root"] / "s2")
    session, progs = ev.load_session(paths), ev.programs(paths)
    state = {flow_key: {"session": "s2", "state": "not_helped", "program": "p0"}}
    monkeypatch.setattr(ctl, "budget_ok", lambda *a: False)  # stop right after re-arming
    cfg = {"ladder_patience": 99, "max_full_tokens": 10**12}
    getattr(ctl, step)(state, session, progs, {}, cfg, paths, lambda m: None)
    assert state[flow_key]["state"] == "not_helped"  # same incumbent: stays done
    session["incumbent"] = "p9"
    progs["p9"] = progs["p0"]
    getattr(ctl, step)(state, session, progs, {}, cfg, paths, lambda m: None)
    assert state[flow_key]["state"] == "idle" and "program" not in state[flow_key]
    assert any(e["event"].endswith("_rearmed") for e in _read(world["tmp"] / "notebook.jsonl"))
