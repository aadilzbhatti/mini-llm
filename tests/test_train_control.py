"""End to end: a real (tiny) training run steered through the control files."""

import json
import threading
import time

import pytest
import torch

import mini_llm.train as train
from mini_llm.control import append_command, read_jsonl


class StubTokenizer:
    """No network: a 64-token vocab is all the loop needs."""

    def __len__(self):
        return 64


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setenv("MINI_LLM_RUN_ID", "t1")
    g = torch.Generator().manual_seed(0)
    torch.save(torch.randint(0, 64, (4000,), generator=g), tmp_path / "train.pt")
    torch.save(torch.randint(0, 64, (600,), generator=g), tmp_path / "val.pt")
    (tmp_path / "runs").mkdir()
    return tmp_path


ARGS = [
    "--tokens",
    "train.pt",
    "--val-tokens",
    "val.pt",
    "--block-size",
    "8",
    "--n-embd",
    "16",
    "--n-head",
    "2",
    "--n-layer",
    "1",
    "--eval-interval",
    "10",
    "--eval-batches",
    "2",
    "--full-eval-interval",
    "0",
    "--log-interval",
    "5",
    "--control-poll",
    "1",
    "--warmup-steps",
    "0",
]


def test_lr_scale_checkpoint_and_stop(workdir):
    append_command(workdir / "runs", "t1", {"type": "set", "knob": "lr_scale", "value": 0.5})
    append_command(workdir / "runs", "t1", {"type": "checkpoint"})

    # Queue the stop so it lands a few steps in: poll() reads the inbox every step.
    def delayed_stop():
        time.sleep(0.5)
        append_command(workdir / "runs", "t1", {"type": "stop"})

    t = threading.Thread(target=delayed_stop)
    t.start()
    train.main(ARGS + ["--steps", "100000", "--lr", "1e-3", "--min-lr", "1e-3", "--save", "--save-name", "m.pt"])
    t.join()

    events = read_jsonl(workdir / "runs" / "t1.events.jsonl")
    assert [e["type"] for e in events] == ["set", "checkpoint", "stop"]
    assert events[0]["step"] == 0

    ckpt = torch.load(workdir / "checkpoints" / "m.pt", map_location="cpu", weights_only=False)
    stopped = events[-1]["step"]
    assert ckpt["step"] == stopped + 1 < 100000  # named/recorded by the training it got
    assert all(abs(lr - 5e-4) < 1e-12 for _, lr in ckpt["lr_history"])  # constant lr, halved
    assert (workdir / "checkpoints" / "m.step1.pt").exists()  # mid-run checkpoint
    assert ckpt["val_history"][-1][0] == stopped  # final eval ran at the stop

    live = json.loads((workdir / "runs" / "t1.live.json").read_text())
    assert live["finished"] is True
    assert list((workdir / "runs" / "tb" / "t1").glob("events.out.tfevents.*"))


def test_pause_then_resume(workdir):
    append_command(workdir / "runs", "t1", {"type": "pause"})

    def resume_later():
        time.sleep(3)
        append_command(workdir / "runs", "t1", {"type": "resume"})

    t = threading.Thread(target=resume_later)
    t.start()
    started = time.time()
    train.main(ARGS + ["--steps", "20", "--no-tensorboard"])
    t.join()
    assert time.time() - started >= 3
    assert [e["type"] for e in read_jsonl(workdir / "runs" / "t1.events.jsonl")] == ["pause", "resume"]


def test_stop_after_ends_early_on_the_long_schedule(workdir, capsys):
    """--stop-after N: N steps of a --steps-long cosine (an LR proxy on the real schedule), not a short cosine."""
    train.main(
        ARGS
        + ["--steps", "1000", "--stop-after", "30", "--lr", "1e-3", "--min-lr", "1e-5", "--save", "--save-name", "p.pt"]
    )
    assert "stopped early by --stop-after" in capsys.readouterr().out

    ckpt = torch.load(workdir / "checkpoints" / "p.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 30 and ckpt["val_history"][-1][0] == 29  # recorded as 30 steps, final eval ran
    lrs = dict(ckpt["lr_history"])
    for step in (0, 15, 29):  # still the 1000-step cosine: barely decayed by step 29
        assert abs(lrs[step] - train.lr_at_step(step, 1000, 1e-3, 1e-5, 0)) < 1e-12
    assert lrs[29] > 0.99e-3
    live = json.loads((workdir / "runs" / "t1.live.json").read_text())
    assert live["total_steps"] == 30  # the page's ETA runs to the stop, not to step 1000
