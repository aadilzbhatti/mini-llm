"""Headless Claude as the proposer: one `claude -p` call per child, no tools, schema-checked JSON.

    claude -p --output-format json --json-schema <schema> --tools "" --model <m>
           --system-prompt <...> --no-session-persistence --strict-mcp-config   (prompt on stdin)

Runs in an empty temp directory, so no project CLAUDE.md or files are visible, with a
subprocess timeout. The CLI returns `structured_output` validated against the schema; it is
validated again here with jsonschema. Every call is logged (prompt, reply, cost, duration) to
autolab/state/llm/<session>/, and its cost is appended to autolab/state/llm_spend.jsonl
(both located relative to the log directory the caller passes).
Any failure raises LLMError with a short reason, which the caller logs before falling back.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from autolab.config import REPO_ROOT

STATE = REPO_ROOT / "autolab" / "state"


class LLMError(RuntimeError):
    """The proposer call failed; the message is the fallback reason."""


class RateLimited(LLMError):
    """Usage limit hit (HTTP 429 / "limit" in the message). Callers should pause proposing, not mutate."""


def claude_bin(llm_cfg: dict) -> str:
    path = Path(os.path.expanduser(llm_cfg.get("claude_bin", "claude")))
    if path.exists():
        return str(path)
    found = shutil.which("claude")
    if not found:
        raise LLMError("claude CLI not found")
    return found


def call(prompt: str, system: str, schema: dict, model: str, llm_cfg: dict, log_dir: Path, tag: str,
         runner=subprocess.run) -> dict:
    """Return {"reply": <validated object>, "cost_usd", "model", "duration_s", "log"}. Raises LLMError."""
    cmd = [claude_bin(llm_cfg), "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
           "--tools", "", "--model", model, "--system-prompt", system,
           "--no-session-persistence", "--strict-mcp-config"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{tag}.json"
    record: dict = {"tag": tag, "model_requested": model, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "system": system, "prompt": prompt}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as cwd:
            r = runner(cmd, input=prompt, capture_output=True, text=True, cwd=cwd,
                       timeout=llm_cfg.get("timeout_s", 600))
    except subprocess.TimeoutExpired:
        _finish(record, log_path, t0, error="timeout")
        raise LLMError(f"timeout after {llm_cfg.get('timeout_s', 600)}s") from None
    record.update(returncode=r.returncode, stdout=r.stdout[-200_000:], stderr=r.stderr[-20_000:])
    if r.returncode != 0:
        # The CLI still prints its JSON result on failure; its `result` is the readable message.
        try:
            info = json.loads(r.stdout)
            msg, status = str(info.get("result") or ""), info.get("api_error_status")
        except json.JSONDecodeError:
            msg, status = "", None
        msg = msg or ((r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"])[0]
        limited = status == 429 or "limit" in msg.lower()
        _finish(record, log_path, t0, error=("rate_limited: " if limited else f"exit {r.returncode}: ") + msg[:200])
        raise (RateLimited if limited else LLMError)(f"claude exited {r.returncode}"
                                                     f"{f' (HTTP {status})' if status else ''}: {msg[:300]}")
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        _finish(record, log_path, t0, error="bad json")
        raise LLMError("claude output is not JSON") from None
    cost = float(out.get("total_cost_usd") or 0.0)
    used = next(iter(out.get("modelUsage") or {}), model)
    record.update(cost_usd=cost, model=used)
    if out.get("is_error"):
        _finish(record, log_path, t0, error="is_error", cost=cost, model=used)
        raise LLMError(f"claude reported an error: {str(out.get('result'))[:300]}")
    reply = out.get("structured_output")
    if reply is None:
        try:
            reply = json.loads(out.get("result") or "")
        except json.JSONDecodeError:
            _finish(record, log_path, t0, error="no structured output", cost=cost, model=used)
            raise LLMError("no structured output in the reply") from None
    try:
        jsonschema.validate(reply, schema)
    except jsonschema.ValidationError as exc:
        _finish(record, log_path, t0, error="schema", cost=cost, model=used, reply=reply)
        raise LLMError(f"reply fails the schema: {exc.message[:300]}") from None
    _finish(record, log_path, t0, cost=cost, model=used, reply=reply)
    return {"reply": reply, "cost_usd": cost, "model": used, "duration_s": round(time.time() - t0, 1),
            "log": str(log_path.relative_to(REPO_ROOT)) if log_path.is_relative_to(REPO_ROOT) else str(log_path)}


def _finish(record: dict, log_path: Path, t0: float, error: str | None = None, cost: float = 0.0,
            model: str | None = None, reply: dict | None = None) -> None:
    record.update(duration_s=round(time.time() - t0, 1), error=error, reply=reply)
    log_path.write_text(json.dumps(record, indent=2))
    spend = log_path.parent.parent.parent / "llm_spend.jsonl"  # <state>/llm/<session>/<call>.json
    spend.parent.mkdir(parents=True, exist_ok=True)
    with open(spend, "a") as fh:
        fh.write(json.dumps({"at": record["at"], "tag": record["tag"], "model": model or record["model_requested"],
                             "usd": round(cost, 5), "ok": error is None, "error": error}) + "\n")
