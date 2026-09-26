"""Control API: submit jobs, steer live runs, read results -- from a phone.

A thin HTTP layer over the files the queue runner and trainer already use.
It holds no state of its own and never runs training itself:

    POST /api/jobs                  -> validated, then written into queue/
    POST /api/runs/{id}/commands    -> appended to runs/{id}.commands.jsonl
    GET  /api/...                   <- read from runs/, queue/, checkpoints/, plots/

So if this process dies, nothing is lost and training carries on; restart it
and it sees exactly what's on disk. Plots are TensorBoard's job, not this
one's: the page just links to it (MINI_LLM_TENSORBOARD_URL).

Auth: if MINI_LLM_TOKEN is set, every /api call needs
`Authorization: Bearer <token>` (the page asks for it once and remembers it).
Meant to sit behind Tailscale either way -- see runner/README.md.

    uv run mini-llm-control --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from mini_llm.control import KNOBS, COMMAND_TYPES, CommandError, append_command, read_jsonl, run_paths

RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
STATIC = Path(__file__).parent / "static"


def _load_runner(repo: Path):
    """runner/run_queue.py is deliberately stdlib-only and not a package; load it by path
    so the API validates jobs with exactly the rules the runner will apply."""
    path = repo / "runner" / "run_queue.py"
    spec = importlib.util.spec_from_file_location("_run_queue", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def create_app(repo: Path | str | None = None, token: str | None = None,
               tensorboard_url: str | None = None, uv: str | None = None) -> FastAPI:
    repo = Path(repo or os.environ.get("MINI_LLM_REPO") or Path.cwd()).resolve()
    token = token if token is not None else os.environ.get("MINI_LLM_TOKEN") or None
    tensorboard_url = tensorboard_url or os.environ.get("MINI_LLM_TENSORBOARD_URL") or ""
    uv = uv or shutil.which("uv") or "uv"
    runs_dir, queue_dir = repo / "runs", repo / "queue"
    runner = _load_runner(repo)

    app = FastAPI(title="mini-llm control", version="1")

    def auth(request: Request) -> None:
        if token and request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    def check_id(run_id: str) -> str:
        if not RUN_ID.match(run_id):
            raise HTTPException(400, "bad run id")
        return run_id

    def status_of(run_id: str) -> dict:
        status = _read_json(runs_dir / f"{run_id}.status.json")
        if status is None:
            raise HTTPException(404, f"no run {run_id}")
        live = _read_json(run_paths(runs_dir, run_id)["live"])
        if live:
            status["live"] = live
        return status

    # --- pages ---------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # --- meta ----------------------------------------------------------------

    @app.get("/api/meta", dependencies=[Depends(auth)])
    def meta() -> dict:
        """Everything a client needs to build forms: allowed flags, knobs, command types."""
        return {
            "protocol": 1,
            "tensorboard_url": tensorboard_url,
            "job_kinds": {
                "train": {
                    "int": runner.INT_FLAGS, "float": runner.FLOAT_FLAGS,
                    "path": sorted(runner.PATH_FLAGS), "name": sorted(runner.NAME_FLAGS),
                    "bool": sorted(runner.BOOL_FLAGS), "defaults": runner.DEFAULT_ARGS,
                },
                "prepare-data": {
                    "int": runner.PREP_INT_FLAGS, "float": runner.PREP_FLOAT_FLAGS,
                    "str": sorted(runner.PREP_STR_FLAGS), "path": ["out-dir"],
                },
            },
            "knobs": {k: {"type": t.__name__, "min": lo, "max": hi} for k, (t, lo, hi) in KNOBS.items()},
            "commands": sorted(COMMAND_TYPES),
            "datasets": sorted(
                str(p.parent.relative_to(repo)) for p in (repo / "data").rglob("train.pt")
            ) if (repo / "data").exists() else [],
        }

    # --- runs ----------------------------------------------------------------

    @app.get("/api/runs", dependencies=[Depends(auth)])
    def list_runs(limit: int = 50) -> list[dict]:
        """Newest first. Status files are the source of truth (index.jsonl only has finished runs)."""
        out = []
        for path in sorted(runs_dir.glob("*.status.json"), reverse=True)[: max(1, min(limit, 500))]:
            status = _read_json(path) or {}
            run_id = status.get("run_id") or path.name.removesuffix(".status.json")
            row = {k: status.get(k) for k in
                   ("run_id", "name", "kind", "status", "started", "finished", "duration_sec", "error")}
            row["run_id"] = run_id
            met = status.get("metrics") or {}
            row["full_val_loss"] = met.get("full_val_loss")
            row["eval_val_loss"] = met.get("eval_val_loss")
            if status.get("status") == "running":
                row["live"] = _read_json(run_paths(runs_dir, run_id)["live"])
            out.append(row)
        return out

    @app.get("/api/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str) -> dict:
        return status_of(check_id(run_id))

    @app.get("/api/runs/{run_id}/log", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def get_log(run_id: str, tail: int = 200) -> str:
        path = runs_dir / f"{check_id(run_id)}.log"
        if not path.exists():
            raise HTTPException(404, "no log")
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-max(1, min(tail, 5000)):])

    @app.get("/api/runs/{run_id}/events", dependencies=[Depends(auth)])
    def get_events(run_id: str) -> dict:
        paths = run_paths(runs_dir, check_id(run_id))
        return {"sent": read_jsonl(paths["commands"]), "applied": read_jsonl(paths["events"])}

    @app.get("/api/runs/{run_id}/report", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def get_report(run_id: str) -> str:
        """The fixed-prompt sample report, found via the run's save-name."""
        status = status_of(check_id(run_id))
        save_name = (status.get("args") or {}).get("save-name")
        candidates = []
        if save_name:
            candidates.append(repo / "checkpoints" / f"{Path(save_name).stem}.md")
        # Fall back to the path the trainer printed.
        log_path = runs_dir / f"{run_id}.log"
        if log_path.exists():
            found = re.findall(r"Saved sample report to (\S+)", log_path.read_text(errors="replace"))
            candidates += [repo / f for f in found[-1:]]
        for path in candidates:
            if path.exists() and path.resolve().is_relative_to(repo):
                return path.read_text(errors="replace")
        raise HTTPException(404, "no sample report for this run (was sample-report set?)")

    @app.get("/api/runs/{run_id}/plot", dependencies=[Depends(auth)])
    def get_plot(run_id: str) -> FileResponse:
        """The run's matplotlib loss plot (the PNG train.py saves with --plot-loss)."""
        status = status_of(check_id(run_id))
        candidates = []
        plot = (status.get("metrics") or {}).get("plot")
        if plot:
            candidates.append(repo / plot)
        log_path = runs_dir / f"{run_id}.log"
        if log_path.exists():
            found = re.findall(r"Saved loss plot to (\S+)", log_path.read_text(errors="replace"))
            candidates += [repo / f for f in found[-1:]]
        plots_dir = (repo / "plots").resolve()
        for path in candidates:
            path = path.resolve()
            if path.suffix == ".png" and path.is_relative_to(plots_dir) and path.exists():
                return FileResponse(path, media_type="image/png")
        raise HTTPException(404, "no loss plot for this run (was plot-loss set, and has it finished?)")

    @app.post("/api/runs/{run_id}/commands", dependencies=[Depends(auth)])
    def send_command(run_id: str, body: dict) -> dict:
        status = status_of(check_id(run_id))
        if status.get("status") != "running":
            raise HTTPException(409, f"run is {status.get('status')}, not running")
        if status.get("kind", "train") != "train":
            raise HTTPException(409, "only training runs take commands")
        try:
            return append_command(runs_dir, run_id, body)
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc

    # --- queue ---------------------------------------------------------------

    @app.get("/api/queue", dependencies=[Depends(auth)])
    def list_queue() -> list[dict]:
        out = []
        for path in sorted(p for p in queue_dir.glob("*.json") if p.is_file()):
            out.append({"file": path.name, "job": _read_json(path)})
        return out

    @app.post("/api/jobs", dependencies=[Depends(auth)])
    def submit_job(body: dict) -> dict:
        """Validate with the runner's own rules, then drop the job into queue/.

        Written to a dotfile first and renamed, so the runner (which globs
        *.json) never sees a half-written job.
        """
        try:
            name, kind, cmd, args = runner.validate_job(body, repo, uv)
        except runner.JobError as exc:
            raise HTTPException(422, str(exc)) from exc
        queue_dir.mkdir(exist_ok=True)
        fname = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}.json"
        tmp = queue_dir / f".{fname}.tmp"
        tmp.write_text(json.dumps({"name": name, "kind": kind, "args": args}, indent=2))
        tmp.replace(queue_dir / fname)
        forecast = runner.forecast_for(repo, args) if kind == "train" else None
        return {"file": fname, "name": name, "kind": kind, "argv": cmd[4:], "forecast": forecast}

    @app.delete("/api/queue/{file}", dependencies=[Depends(auth)])
    def cancel_job(file: str) -> dict:
        if not RUN_ID.match(file) or not file.endswith(".json"):
            raise HTTPException(400, "bad job file name")
        path = queue_dir / file
        if not path.exists():
            raise HTTPException(404, "not queued (already started or finished?)")
        dest = queue_dir / "cancelled"
        dest.mkdir(exist_ok=True)
        shutil.move(str(path), dest / file)
        return {"cancelled": file}

    # --- results -------------------------------------------------------------

    @app.get("/api/baselines", dependencies=[Depends(auth)])
    def baselines() -> Any:
        return _read_json(repo / "baselines.json") or []

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="mini-llm control API")
    p.add_argument("--repo", default=os.environ.get("MINI_LLM_REPO", "."))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    uvicorn.run(create_app(repo=args.repo), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
