"""Local side of the Modal backend with modal calls faked (no network, no spend)."""

import json

import pytest

modal = pytest.importorskip("modal")

import autolab.modal_backend as mb  # noqa: E402
from autolab.trainer import Budget, TrainRequest  # noqa: E402


class FakeCall:
    results: dict = {}

    def __init__(self, object_id):
        self.object_id = object_id

    @classmethod
    def from_id(cls, object_id):
        return cls(object_id)

    def get(self, timeout=None):
        r = FakeCall.results.get(self.object_id)
        if r is None:
            raise TimeoutError
        if isinstance(r, Exception):
            raise r
        return r


class FakeFn:
    n = 0

    def with_options(self, **kw):
        self.opts = kw
        return self

    def spawn(self, request, meta, code):
        FakeFn.n += 1
        FakeFn.last = (request, meta, code)
        return FakeCall(f"fc-{FakeFn.n}")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "_state_path", lambda: tmp_path / "state" / "calls.json")
    monkeypatch.setattr(mb.modal.Function, "from_name", staticmethod(lambda app, name: FakeFn()))
    monkeypatch.setattr(mb.modal, "FunctionCall", FakeCall)
    monkeypatch.setitem(mb._CFG, "max_usd", 1.0)
    FakeCall.results = {}
    return tmp_path


def req(run_id, wall=420.0, tmp_path=None):
    from autolab.config import load_config

    cfg = load_config()
    train = cfg.datasets_dir / "data20k" / "train.pt"
    if not train.exists() or not cfg.frozen_val.exists():
        pytest.skip("autolab data not present")
    return TrainRequest(run_id=run_id, dataset_id="data20k", train_tokens=str(train),
                        budget=Budget(tokens=1_000_000, wall_clock_s=wall))


def test_submit_records_call_and_sends_code(fake):
    c = mb.submit(req("a"), gpu="L4")
    assert c["state"] == "pending" and c["call_id"] == f"fc-{FakeFn.n}"
    assert c["usd_estimate"] == pytest.approx((420 + 180) * 0.000222, rel=1e-3)
    request, meta, code = FakeFn.last
    assert request["run_id"] == "a" and len(meta["val_sha256"]) == 64 and len(meta["train_sha256"]) == 64
    assert "mini_llm/train.py" in code and "mini_llm/model.py" in code
    with pytest.raises(FileExistsError):
        mb.submit(req("a"), gpu="L4")


def test_cost_cap_refuses(fake):
    mb.submit(req("a", wall=3000), gpu="L4")  # ~$0.71 committed
    with pytest.raises(RuntimeError, match="cost cap"):
        mb.submit(req("b", wall=3000), gpu="L4")


def test_collect_writes_results_and_failures(fake):
    mb.submit(req("ok"), gpu="L4")
    mb.submit(req("bad"), gpu="L4")
    calls = mb.load_calls()
    FakeCall.results = {
        calls["ok"]["call_id"]: {"launch": {"status": "finished", "wall_s": 400.0, "budget_hit": "tokens"},
                                 "train_log": "log", "report": {"summary": {"final_full_val_loss": 6.1}},
                                 "diagnosis": {"primary": "still_improving"}},
        calls["bad"]["call_id"]: RuntimeError("sha256 mismatch"),
    }
    done = mb.collect(runs_dir=fake / "runs", log=lambda m: None)
    assert sorted(done) == ["bad", "ok"]
    calls = mb.load_calls()
    assert calls["ok"]["state"] == "finished" and calls["ok"]["full_val_loss"] == 6.1
    assert calls["ok"]["usd"] == pytest.approx(400 * 0.000222, rel=1e-3)
    assert calls["bad"]["state"] == "failed" and "sha256" in calls["bad"]["error"]
    assert json.loads((fake / "runs" / "ok" / "report.json").read_text())["summary"]["final_full_val_loss"] == 6.1
    assert mb.collect(runs_dir=fake / "runs", log=lambda m: None) == []  # nothing new
