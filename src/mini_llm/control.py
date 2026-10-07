"""Live control of a running training job, over plain files.

The trainer never opens a socket. Everything that talks to a live run does
so through three files in runs/, keyed by the run id:

    runs/<run_id>.commands.jsonl   inbox: one JSON command per line (append-only)
    runs/<run_id>.events.jsonl     outbox: every command applied or rejected, by step
    runs/<run_id>.live.json        heartbeat: step, lr, latest losses, rate, ETA

Anyone who can append a line to the inbox can steer the run -- the control
API (mini_llm.server), a shell one-liner, or another researcher's tooling.
The trainer drains the inbox only at step boundaries (every
`poll_every` steps), so a command never lands mid-backward, and every
change is written to the outbox with the exact step it took effect. A run's
effective config is therefore `launch args + events`, which is what makes a
run that was steered by hand still reproducible.

This module is stdlib-only at import time (tensorboard is imported lazily),
so the control API can import the command schema without pulling in torch.

Command shapes (the "protocol"):

    {"id": "...", "type": "set", "knob": "lr_scale", "value": 0.5}
    {"id": "...", "type": "pause"} / {"type": "resume"}
    {"id": "...", "type": "eval_now"}        sampled + full val eval at next boundary
    {"id": "...", "type": "checkpoint"}      save a mid-run checkpoint
    {"id": "...", "type": "stop"}            finish early, but still plot/save/report

`id` is an idempotency key: a command whose id was already applied is ignored,
so a client retrying over a flaky connection can't apply `lr_scale` twice.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# knob -> (type, min, max). Only these can change mid-run. Everything else
# (model shape, data, seed) is fixed at launch; changing it would make the run
# a different experiment, which is what a new job is for.
KNOBS: dict[str, tuple[type, float, float]] = {
    # Multiplies the scheduled LR. 1.0 = the schedule as launched. The
    # schedule keeps its shape; this scales it, so warmup/cosine still apply.
    "lr_scale": (float, 1e-4, 10.0),
    "log_interval": (int, 1, 100_000),
    "eval_interval": (int, 1, 1_000_000),
    # 0 disables periodic full evals (the end-of-training one still runs).
    "full_eval_interval": (int, 0, 1_000_000),
}

COMMAND_TYPES = {"set", "pause", "resume", "eval_now", "checkpoint", "stop"}


class CommandError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_command(raw: Any) -> dict:
    """Normalize a command dict, or raise CommandError. Shared by trainer and API."""
    if not isinstance(raw, dict):
        raise CommandError("command must be a JSON object")
    ctype = raw.get("type")
    if ctype not in COMMAND_TYPES:
        raise CommandError(f"unknown command type {ctype!r}; expected one of {sorted(COMMAND_TYPES)}")
    cmd: dict[str, Any] = {"id": str(raw.get("id") or uuid.uuid4().hex), "type": ctype}
    if ctype == "set":
        knob = raw.get("knob")
        if knob not in KNOBS:
            raise CommandError(f"unknown knob {knob!r}; expected one of {sorted(KNOBS)}")
        kind, low, high = KNOBS[knob]
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CommandError(f"{knob} must be a number, got {value!r}")
        if kind is int and float(value) != int(value):
            raise CommandError(f"{knob} must be an integer, got {value!r}")
        value = kind(value)
        if not low <= value <= high:
            raise CommandError(f"{knob}={value} outside allowed range [{low}, {high}]")
        cmd["knob"], cmd["value"] = knob, value
    if raw.get("note"):
        cmd["note"] = str(raw["note"])[:500]
    return cmd


def run_paths(runs_dir: Path, run_id: str) -> dict[str, Path]:
    return {
        "commands": runs_dir / f"{run_id}.commands.jsonl",
        "events": runs_dir / f"{run_id}.events.jsonl",
        "live": runs_dir / f"{run_id}.live.json",
    }


def append_command(runs_dir: Path, run_id: str, raw: dict) -> dict:
    """Validate and append a command to a run's inbox. Returns the stored command.

    A single os.write of one line with O_APPEND, so concurrent writers never
    interleave partial lines, and the reader only consumes newline-terminated
    lines.
    """
    cmd = validate_command(raw)
    cmd["sent"] = now_iso()
    path = run_paths(runs_dir, run_id)["commands"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(cmd) + "\n").encode())
    finally:
        os.close(fd)
    return cmd


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


@dataclass
class ControlState:
    """What the training loop reads each step. Mutated only by RunControl.poll."""

    lr_scale: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    full_eval_interval: int = 5000
    eval_now: bool = False
    checkpoint_now: bool = False
    stop: bool = False
    paused: bool = False


@dataclass
class RunControl:
    """The trainer's side of the protocol: inbox reader, outbox writer, heartbeat.

    Also owns the TensorBoard writer, so every applied command shows up in
    TensorBoard as text at the step it took effect, and `lr_scale` as a scalar.
    """

    run_id: str
    runs_dir: Path
    state: ControlState
    poll_every: int = 25
    tensorboard_dir: Path | None = None
    _offset: int = 0
    _seen: set[str] = field(default_factory=set)
    _tb: Any = None
    _last_beat: tuple[float, int] | None = None
    _rate: float | None = None
    _last_write: float = 0.0
    heartbeat_every: float = 2.0  # seconds; small runs step far faster than anyone polls

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths = run_paths(self.runs_dir, self.run_id)
        # A process restarted under the same run id must not re-apply commands
        # an earlier attempt already applied: seed the seen-set from the outbox.
        for ev in read_jsonl(self.paths["events"]):
            if ev.get("id"):
                self._seen.add(ev["id"])
        if self.tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.tensorboard_dir))
            except Exception as exc:  # noqa: BLE001 - TB is a nicety, never a blocker
                print(f"TensorBoard disabled: {type(exc).__name__}: {exc}")

    # --- TensorBoard -------------------------------------------------------

    @property
    def tb(self):
        return self._tb

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

    def text(self, tag: str, text: str, step: int) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, text, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()

    # --- inbox / outbox ----------------------------------------------------

    def _read_new(self) -> list[dict]:
        path = self.paths["commands"]
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return []
        if size <= self._offset:
            return []
        with path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read(size - self._offset)
        # Only consume complete lines; a partial trailing line waits for next poll.
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        out = []
        for line in chunk[: end + 1].splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"type": None, "_raw": line.decode(errors="replace")})
        return out

    def _event(self, step: int, **fields: Any) -> dict:
        ev = {"step": step, "at": now_iso(), **fields}
        with self.paths["events"].open("a") as fh:
            fh.write(json.dumps(ev) + "\n")
        return ev

    def _apply(self, cmd: dict, step: int) -> str:
        s = self.state
        t = cmd["type"]
        if t == "set":
            old = getattr(s, cmd["knob"])
            setattr(s, cmd["knob"], cmd["value"])
            if cmd["knob"] == "lr_scale":
                self.scalar("control/lr_scale", s.lr_scale, step)
            return f"{cmd['knob']}: {old} -> {cmd['value']}"
        if t == "pause":
            s.paused = True
            return "paused"
        if t == "resume":
            s.paused = False
            return "resumed"
        if t == "eval_now":
            s.eval_now = True
            return "eval scheduled"
        if t == "checkpoint":
            s.checkpoint_now = True
            return "checkpoint scheduled"
        if t == "stop":
            s.stop = True
            return "stopping after this step"
        raise CommandError(f"unhandled command type {t!r}")

    def poll(self, step: int, force: bool = False) -> list[dict]:
        """Drain the inbox if it's time. Returns the events written."""
        if not force and step % self.poll_every != 0:
            return []
        events = []
        for raw in self._read_new():
            cid = str(raw.get("id") or "")
            if cid and cid in self._seen:
                continue
            try:
                cmd = validate_command(raw)
                result = self._apply(cmd, step)
                ev = self._event(
                    step,
                    id=cmd["id"],
                    type=cmd["type"],
                    ok=True,
                    result=result,
                    **({"knob": cmd["knob"], "value": cmd["value"]} if cmd["type"] == "set" else {}),
                )
                print(f"step {step:5d} | control: {result}", flush=True)
                self.text("control/events", f"step {step}: {result}", step)
            except CommandError as exc:
                ev = self._event(step, id=cid or None, type=raw.get("type"), ok=False, error=str(exc))
                print(f"step {step:5d} | control: rejected {raw!r}: {exc}", flush=True)
            if ev.get("id"):
                self._seen.add(ev["id"])
            events.append(ev)
        return events

    def wait_while_paused(self, step: int, sleep: float = 2.0) -> None:
        """Block at a step boundary until resumed (or stopped). Heartbeats meanwhile."""
        while self.state.paused and not self.state.stop:
            self.heartbeat(step, extra={"paused": True}, force=True)
            time.sleep(sleep)
            self.poll(step, force=True)
        self._last_beat = None  # don't average the pause into steps/sec

    # --- heartbeat ---------------------------------------------------------

    def heartbeat(
        self, step: int, total_steps: int | None = None, extra: dict | None = None, force: bool = False
    ) -> None:
        t = time.time()
        if not force and t - self._last_write < self.heartbeat_every:
            return
        self._last_write = t
        if self._last_beat is not None and step > self._last_beat[1]:
            dt = t - self._last_beat[0]
            inst = (step - self._last_beat[1]) / dt if dt > 0 else None
            if inst:
                self._rate = inst if self._rate is None else 0.8 * self._rate + 0.2 * inst
        if not (extra or {}).get("paused"):
            self._last_beat = (t, step)
        live = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "updated": now_iso(),
            "step": step,
            "total_steps": total_steps,
            "steps_per_sec": round(self._rate, 3) if self._rate else None,
            "eta_sec": (round((total_steps - step) / self._rate) if self._rate and total_steps else None),
            "paused": self.state.paused,
            "lr_scale": self.state.lr_scale,
            "log_interval": self.state.log_interval,
            "eval_interval": self.state.eval_interval,
            "full_eval_interval": self.state.full_eval_interval,
            **(extra or {}),
        }
        if total_steps is None:
            try:
                prev = json.loads(self.paths["live"].read_text())
                live["total_steps"] = prev.get("total_steps")
            except (OSError, json.JSONDecodeError):
                pass
        tmp = self.paths["live"].with_suffix(".json.tmp")
        tmp.write_text(json.dumps(live, indent=2))
        tmp.replace(self.paths["live"])
