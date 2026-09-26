"""The live-control protocol: validation, inbox/outbox, idempotency, heartbeat."""

import json

import pytest

from mini_llm.control import (
    CommandError,
    ControlState,
    RunControl,
    append_command,
    read_jsonl,
    run_paths,
    validate_command,
)


def make(tmp_path, **kw):
    return RunControl(run_id="r1", runs_dir=tmp_path, state=ControlState(), poll_every=1, **kw)


def test_validate_rejects_unknown_type_and_knob():
    with pytest.raises(CommandError):
        validate_command({"type": "rm -rf"})
    with pytest.raises(CommandError):
        validate_command({"type": "set", "knob": "n_layer", "value": 8})


def test_validate_range_and_int_knobs():
    with pytest.raises(CommandError):
        validate_command({"type": "set", "knob": "lr_scale", "value": 50})
    with pytest.raises(CommandError):
        validate_command({"type": "set", "knob": "eval_interval", "value": 2.5})
    with pytest.raises(CommandError):
        validate_command({"type": "set", "knob": "lr_scale", "value": True})
    cmd = validate_command({"type": "set", "knob": "eval_interval", "value": 200.0})
    assert cmd["value"] == 200 and isinstance(cmd["value"], int)


def test_commands_apply_and_are_logged_with_step(tmp_path):
    rc = make(tmp_path)
    append_command(tmp_path, "r1", {"type": "set", "knob": "lr_scale", "value": 0.5})
    append_command(tmp_path, "r1", {"type": "checkpoint"})
    events = rc.poll(step=42)
    assert rc.state.lr_scale == 0.5 and rc.state.checkpoint_now
    assert [e["step"] for e in events] == [42, 42]
    assert all(e["ok"] for e in read_jsonl(run_paths(tmp_path, "r1")["events"]))


def test_duplicate_id_is_applied_once(tmp_path):
    rc = make(tmp_path)
    append_command(tmp_path, "r1", {"id": "abc", "type": "set", "knob": "lr_scale", "value": 0.5})
    rc.poll(1)
    rc.state.lr_scale = 1.0  # pretend something else reset it
    append_command(tmp_path, "r1", {"id": "abc", "type": "set", "knob": "lr_scale", "value": 0.5})
    assert rc.poll(2) == []
    assert rc.state.lr_scale == 1.0


def test_restart_does_not_replay_applied_commands(tmp_path):
    rc = make(tmp_path)
    append_command(tmp_path, "r1", {"id": "x", "type": "stop"})
    rc.poll(1)
    fresh = make(tmp_path)
    fresh.poll(1)
    assert fresh.state.stop is False


def test_partial_line_waits_for_newline(tmp_path):
    rc = make(tmp_path)
    path = run_paths(tmp_path, "r1")["commands"]
    path.write_text('{"type": "pause"')
    assert rc.poll(1) == []
    with path.open("a") as fh:
        fh.write("}\n")
    rc.poll(2)
    assert rc.state.paused


def test_bad_line_is_rejected_not_fatal(tmp_path):
    rc = make(tmp_path)
    path = run_paths(tmp_path, "r1")["commands"]
    path.write_text('not json\n{"type": "set", "knob": "lr_scale", "value": 99}\n{"type": "eval_now"}\n')
    events = rc.poll(3)
    assert [e["ok"] for e in events] == [False, False, True]
    assert rc.state.eval_now


def test_poll_respects_interval(tmp_path):
    rc = RunControl(run_id="r1", runs_dir=tmp_path, state=ControlState(), poll_every=10)
    append_command(tmp_path, "r1", {"type": "pause"})
    assert rc.poll(3) == []
    assert rc.poll(10)


def test_heartbeat_writes_live_json(tmp_path):
    rc = make(tmp_path)
    rc.heartbeat(10, 100, extra={"lr": 1e-3})
    rc.heartbeat(11, 100)  # throttled: within heartbeat_every of the last write
    live = json.loads(run_paths(tmp_path, "r1")["live"].read_text())
    assert live["step"] == 10 and live["total_steps"] == 100 and live["lr"] == 1e-3
