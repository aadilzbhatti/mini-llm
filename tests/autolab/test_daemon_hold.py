"""The owner's pause: a held daemon cycle collects finished Modal runs and does nothing else."""

from autolab import controller as ctl
from autolab import daemon
from autolab import evaluate as ev
from autolab import modal_backend as mb


def test_held_cycle_only_collects(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "CONTROL", tmp_path / "controller.json")
    monkeypatch.setattr(ctl, "NOTEBOOK", tmp_path / "notebook.jsonl")
    monkeypatch.setattr(mb, "collect", lambda log=print: ["run-a"])
    monkeypatch.setattr(mb, "fetch_live", lambda: 0)
    monkeypatch.setattr(mb, "load_calls", lambda: {"run-b": {"state": "pending"}, "run-a": {"state": "finished"}})
    monkeypatch.setattr("autolab.activity.set_activity", lambda *a, **k: None)

    def forbidden(*a, **k):
        raise AssertionError("a held daemon must not advance, judge or propose")

    monkeypatch.setattr(ev, "advance_everything", forbidden)
    monkeypatch.setattr(ctl, "step", forbidden)
    ctl.hold("test")
    out = daemon.cycle(log=lambda m: None)
    assert out["finished_this_cycle"] == ["run-a"] and out["pending"] == 1 and out["held"]["by"] == "test"
    assert ctl.release("test")["by"] == "test" and "hold" not in ctl.load_control()
    assert ctl.release("test") is None  # unpausing twice is harmless
