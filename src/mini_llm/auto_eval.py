"""Evaluate every finished run automatically, like its loss plot.

Runs inside the mirror service (mini_llm.remote.modal_mirror), the one
always-on Python process with the project environment. Each scan looks at
runs/*.status.json for runs that

  - completed after auto-eval was switched on (runs/auto_eval.json "since"),
    so turning it on doesn't re-evaluate the whole history,
  - trained with a val set (a real experiment, same rule as the import), and
  - have their checkpoint in checkpoints/: a Modal run once it's imported, a
    local queue run as soon as the runner has saved it,

and queues `mini-llm-eval <checkpoint>` for them, then (GPU permitting) the
inference benchmark and the samples report (mini_llm.samples). A single worker thread runs
them one at a time on the local GPU (serialised so two evals never share it)
and regenerates evals/summary.md. Progress is written into the run's status
file ("eval": {"state": "queued" | "running" | "done" | "failed", ...}) so the
page can show it, and a failed eval keeps its log (runs/<id>.eval.log) and
isn't retried. Inference timings taken while a local training job holds the
GPU are flagged in the report.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = "auto_eval.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def checkpoint_of(status: dict, repo: Path) -> Path | None:
    """The run's final checkpoint, if it's in checkpoints/."""
    if status.get("remote"):  # Modal: the mirror's import names it after args["save-name"]
        name = (status.get("args") or {}).get("save-name")
        path = repo / "checkpoints" / name if name else None
        return path if path and path.is_file() else None
    log = repo / (status.get("log") or f"runs/{status.get('run_id')}.log")
    try:
        saved = re.findall(r"^Saved to (\S+\.pt)$", log.read_text(errors="replace"), re.M)
    except OSError:
        return None
    path = repo / saved[-1] if saved else None
    return path if path and path.is_file() else None


def _pgrep(pattern: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except OSError:
        return False


def training_running() -> bool:
    return _pgrep("mini-llm-train")


def gpu_busy() -> bool:
    """Training, or another eval/benchmark (e.g. a manual backfill) outside this worker."""
    return _pgrep("mini-llm-train|mini-llm-eval|mini_llm[.]evals|mini-llm-bench|mini_llm[.]bench")


class AutoEvaluator:
    def __init__(self, repo: Path, python: str = sys.executable):
        self.repo, self.python = Path(repo), python
        self.runs = self.repo / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        state_path = self.runs / STATE_FILE
        state = _read(state_path)
        if "since" not in state:
            state = {"since": _now()}
            _write(state_path, state)
        self.since = state["since"]
        self.jobs: queue.Queue = queue.Queue()
        self.queued: set[str] = set()
        self.thread = threading.Thread(target=self._work, daemon=True, name="auto-eval")
        self.thread.start()

    def _set(self, status_path: Path, **fields) -> None:
        st = _read(status_path)
        st["eval"] = {**(st.get("eval") or {}), **fields}
        _write(status_path, st)

    def scan(self) -> list[str]:
        """Queue every eligible, not-yet-evaluated run. Returns the run ids queued now."""
        new = []
        for path in sorted(self.runs.glob("*.status.json")):
            st = _read(path)
            run_id = st.get("run_id") or path.name.removesuffix(".status.json")
            if run_id in self.queued or st.get("status") != "completed" or st.get("kind", "train") != "train":
                continue
            if (st.get("finished") or "") < self.since or not (st.get("args") or {}).get("val-tokens"):
                continue
            if (st.get("eval") or {}).get("state") in ("done", "failed"):
                continue
            ckpt = checkpoint_of(st, self.repo)
            if ckpt is None:
                continue  # not saved, or a Modal run not imported yet: try again next scan
            self.queued.add(run_id)
            self._set(path, state="queued", report=ckpt.stem, at=_now())
            self.jobs.put((run_id, path, ckpt))
            new.append(run_id)
        return new

    def _work(self) -> None:
        while True:
            run_id, status_path, ckpt = self.jobs.get()
            log = self.runs / f"{run_id}.eval.log"
            contended = training_running()
            self._set(status_path, state="running", started=_now(), gpu_shared_with_training=contended)
            print(f"[auto-eval] {run_id}: evaluating {ckpt.name}"
                  f"{' (a local training job is running: timings will be skewed)' if contended else ''}", flush=True)
            with log.open("w") as fh:
                cmd = [self.python, "-m", "mini_llm.evals", str(ckpt)] + (["--gpu-shared"] if contended else [])
                rc = subprocess.run(cmd, cwd=self.repo, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc == 0 and not gpu_busy():
                # Re-run the controlled inference benchmark across every evaluated model, so the
                # summary's cost columns come from one session (skipped if the GPU is shared).
                # Then the fixed-prompt samples of the best model per context (evals/samples.md),
                # which a new best model changes.
                with log.open("a") as fh:
                    subprocess.run([self.python, "-m", "mini_llm.bench"], cwd=self.repo, stdout=fh, stderr=subprocess.STDOUT)
                    subprocess.run([self.python, "-m", "mini_llm.samples"], cwd=self.repo, stdout=fh, stderr=subprocess.STDOUT)
            self._set(status_path, state="done" if rc == 0 else "failed", finished=_now(),
                      **({} if rc == 0 else {"error": f"mini-llm-eval exited {rc}; see runs/{run_id}.eval.log"}))
            print(f"[auto-eval] {run_id}: {'done' if rc == 0 else f'FAILED ({rc})'}", flush=True)
            self.jobs.task_done()
