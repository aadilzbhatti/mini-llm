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
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from mini_llm.control import KNOBS, COMMAND_TYPES, CommandError, append_command, read_jsonl, run_paths
from mini_llm import sweeps
from mini_llm.remote import costs

RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
# Modal GPU spec as modal_train.py takes it: "L4:2", "A100-80GB:4", "H100", or "cpu".
MODAL_GPUS = re.compile(r"^(cpu|[A-Za-z0-9-]{2,20}(:[1-8])?)$")
MODAL_KEYS = ("target", "gpus", "timeout_hours")
CKPT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}\.pt$")
EVAL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
# Decoding knobs the page offers and /api/generate accepts: (min, max, default).
# temperature 0 = greedy; top_k 0 and top_p 1 = off. Ranges stop where they stop
# being useful for a ~16M-param model: above T=1 or k=100 it samples its garbage tail.
SAMPLING = {
    "temperature": (0.0, 1.5, 0.7),
    "top_k": (0, 200, 40),
    "top_p": (0.05, 1.0, 1.0),
    "max_new_tokens": (1, 1024, 256),
}
EOS_TOKEN_ID = 50256
STATIC = Path(__file__).parent / "static"


def _load_runner(repo: Path):
    """runner/run_queue.py is deliberately stdlib-only and not a package; load it by path
    so the API validates jobs with exactly the rules the runner will apply."""
    path = repo / "runner" / "run_queue.py"
    spec = importlib.util.spec_from_file_location("_run_queue", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Which flags the phone form shows up front; everything else sits under "More options".
COMMON_FLAGS = {
    "train": [
        "tokens",
        "val-tokens",
        "resume",
        "restart-lr",
        "steps",
        "n-embd",
        "n-head",
        "n-layer",
        "block-size",
        "lr",
        "min-lr",
        "warmup-steps",
        "full-eval-interval",
        "save",
        "save-name",
        "sample-report",
        "plot-suffix",
    ],
    "prepare-data": ["out-dir", "num-examples", "val-examples", "dataset", "config"],
}
# Free-text fields get suggestions, not a closed list (any HF id is allowed).
EXAMPLES = {
    "dataset": ["HuggingFaceTB/smollm-corpus", "HuggingFaceFW/fineweb-edu", "wikimedia/wikipedia"],
    "config": ["fineweb-edu-dedup", "cosmopedia-v2", "20231101.en"],
    "split": ["train"],
    "text-field": ["text"],
    "tokenizer": ["gpt2"],
    "out-dir": ["data/data50k"],
}


def _parser_defaults(module: str) -> dict[str, dict]:
    """flag -> {default, help} straight from the script's argparse definition,
    so the form never drifts from what the script actually does."""
    try:
        mod = importlib.import_module(f"mini_llm.{module}")
        parser = mod.build_parser()
    except Exception:  # noqa: BLE001 - form still works, just without defaults/help
        return {}
    out = {}
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                # first sentence only; don't cut at "e.g." / "i.e."
                help_text = re.split(r"(?<!e\.g)(?<!i\.e)\.\s", action.help or "", maxsplit=1)[0].strip()
                out[opt[2:]] = {"default": action.default, "help": help_text}
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def create_app(
    repo: Path | str | None = None, token: str | None = None, tensorboard_url: str | None = None, uv: str | None = None
) -> FastAPI:
    repo = Path(repo or os.environ.get("MINI_LLM_REPO") or Path.cwd()).resolve()
    token = token if token is not None else os.environ.get("MINI_LLM_TOKEN") or None
    tensorboard_url = tensorboard_url or os.environ.get("MINI_LLM_TENSORBOARD_URL") or ""
    uv = uv or shutil.which("uv") or "uv"
    runs_dir, queue_dir = repo / "runs", repo / "queue"
    runner = _load_runner(repo)
    parser_info = {"train": _parser_defaults("train"), "prepare-data": _parser_defaults("prepare_dataset")}

    def rel(p: Path) -> str:
        return str(p.relative_to(repo))

    def running_statuses() -> list[dict]:
        out = []
        for path in runs_dir.glob("*.status.json"):
            st = _read_json(path) or {}
            if st.get("status") == "running":
                out.append(st)
        return out

    def queued_jobs() -> list[tuple[Path, dict]]:
        return [(p, _read_json(p) or {}) for p in sorted(q for q in queue_dir.glob("*.json") if q.is_file())]

    def job_is_running(path: Path, job: dict, running: list[dict]) -> bool:
        # The runner leaves a job's file in queue/ while it runs. Newer statuses
        # name the file; older ones are matched by their args.
        for st in running:
            if st.get("job_file") == path.name:
                return True
            if "job_file" not in st and st.get("args") == job.get(
                "args", {k: v for k, v in job.items() if k not in ("name", "kind")}
            ):
                return True
        return False

    def pending_outputs() -> frozenset[str]:
        """Checkpoints that queued or running jobs will save: valid resume targets
        for a continuation queued behind them."""
        out = set()
        arg_sets = [job.get("args") or {} for _, job in queued_jobs()] + [
            st.get("args") or {} for st in running_statuses()
        ]
        for args in arg_sets:
            if args.get("save") and args.get("save-name"):
                out.add(f"checkpoints/{args['save-name']}")
        return frozenset(out)

    def path_choices(flag: str, pending: frozenset[str]) -> list[str]:
        data = repo / "data"
        if flag in ("tokens", "val-tokens"):
            files = sorted(data.rglob("*.pt")) if data.exists() else []
            want = "val" if flag == "val-tokens" else "train"
            files.sort(key=lambda f: (want not in f.name, str(f)))
            return [rel(f) for f in files]
        if flag == "text":
            return [rel(f) for f in sorted(data.rglob("*.txt"))] if data.exists() else []
        if flag == "resume":
            ckpts = repo / "checkpoints"
            have = (
                [rel(f) for f in sorted(ckpts.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)]
                if ckpts.exists()
                else []
            )
            return sorted(pending - set(have)) + have
        return []

    def job_schema() -> dict:
        pending = pending_outputs()
        kinds: dict[str, list[dict]] = {}
        for kind, groups in {
            "train": [
                ("int", runner.INT_FLAGS),
                ("float", runner.FLOAT_FLAGS),
                ("path", {f: None for f in runner.PATH_FLAGS}),
                ("name", {f: None for f in runner.NAME_FLAGS}),
                ("bool", {f: None for f in runner.BOOL_FLAGS}),
            ],
            "prepare-data": [
                ("int", runner.PREP_INT_FLAGS),
                ("float", runner.PREP_FLOAT_FLAGS),
                ("str", {f: None for f in runner.PREP_STR_FLAGS}),
                ("outdir", {"out-dir": None}),
            ],
        }.items():
            fields = []
            info = parser_info.get(kind, {})
            defaults = runner.DEFAULT_ARGS if kind == "train" else {}
            for ftype, flags in groups:
                for flag, bounds in flags.items():
                    field = {
                        "flag": flag,
                        "type": ftype,
                        "help": info.get(flag, {}).get("help", ""),
                        "default": defaults.get(flag, info.get(flag, {}).get("default")),
                        "common": flag in COMMON_FLAGS[kind],
                    }
                    if bounds:
                        field["min"], field["max"] = bounds
                    if ftype == "path":
                        field["choices"] = path_choices(flag, pending)
                        field["pending"] = sorted(pending) if flag == "resume" else []
                    if flag in EXAMPLES:
                        field["examples"] = EXAMPLES[flag]
                    if ftype == "outdir":
                        field["required"] = True
                    fields.append(field)
            order = COMMON_FLAGS[kind]
            fields.sort(key=lambda f: (not f["common"], order.index(f["flag"]) if f["flag"] in order else 0, f["flag"]))
            kinds[kind] = fields
        return kinds

    app = FastAPI(title="mini-llm control", version="1")

    from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    # uvicorn's access log has the status code but not why: log the reason for every rejected
    # request, so an error seen on the phone can be found here.
    @app.exception_handler(StarletteHTTPException)
    async def log_http_error(request: Request, exc: StarletteHTTPException):
        if 400 <= exc.status_code < 500 and exc.status_code != 404:
            print(
                f"[api] {request.method} {request.url.path} -> {exc.status_code}: {exc.detail}",
                file=sys.stderr,
                flush=True,
            )
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def log_validation_error(request: Request, exc: RequestValidationError):
        print(
            f"[api] {request.method} {request.url.path} -> 422 (request body): {exc.errors()}",
            file=sys.stderr,
            flush=True,
        )
        return await request_validation_exception_handler(request, exc)

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
        cost = costs.run_cost(status, live, costs.load_rates(runs_dir), time.time())
        if cost:
            status["cost"] = cost
        return status

    # --- pages ---------------------------------------------------------------

    def ui_version() -> str:
        return str(int((STATIC / "index.html").stat().st_mtime))

    # A phone can keep the page open for days while the server and API change under it. Every
    # API response carries the page's version; the page reloads itself when it changes.
    @app.middleware("http")
    async def stamp_ui_version(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["X-UI-Version"] = ui_version()
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    # --- meta ----------------------------------------------------------------

    @app.get("/api/meta", dependencies=[Depends(auth)])
    def meta() -> dict:
        """Everything a client needs to build forms: allowed flags, knobs, command types."""
        return {
            "protocol": 1,
            "tensorboard_url": tensorboard_url,
            "job_kinds": {
                "train": {
                    "int": runner.INT_FLAGS,
                    "float": runner.FLOAT_FLAGS,
                    "path": sorted(runner.PATH_FLAGS),
                    "name": sorted(runner.NAME_FLAGS),
                    "bool": sorted(runner.BOOL_FLAGS),
                    "defaults": runner.DEFAULT_ARGS,
                },
                "prepare-data": {
                    "int": runner.PREP_INT_FLAGS,
                    "float": runner.PREP_FLOAT_FLAGS,
                    "str": sorted(runner.PREP_STR_FLAGS),
                    "path": ["out-dir"],
                },
            },
            "knobs": {k: {"type": t.__name__, "min": lo, "max": hi} for k, (t, lo, hi) in KNOBS.items()},
            "commands": sorted(COMMAND_TYPES),
            "sampling": {k: {"min": lo, "max": hi, "default": d} for k, (lo, hi, d) in SAMPLING.items()},
            "form": job_schema(),
            "datasets": (
                sorted(str(p.parent.relative_to(repo)) for p in (repo / "data").rglob("train.pt"))
                if (repo / "data").exists()
                else []
            ),
        }

    # --- runs ----------------------------------------------------------------

    @app.get("/api/runs", dependencies=[Depends(auth)])
    def list_runs(limit: int = 50) -> list[dict]:
        """Newest first. Status files are the source of truth (index.jsonl only has finished runs)."""
        out = []
        rates, now = costs.load_rates(runs_dir), time.time()
        for path in sorted(runs_dir.glob("*.status.json"), reverse=True)[: max(1, min(limit, 500))]:
            status = _read_json(path) or {}
            run_id = status.get("run_id") or path.name.removesuffix(".status.json")
            row = {
                k: status.get(k)
                for k in (
                    "run_id",
                    "name",
                    "kind",
                    "status",
                    "started",
                    "finished",
                    "duration_sec",
                    "error",
                    "remote",
                    "eval",
                )
            }
            row["run_id"] = run_id
            met = status.get("metrics") or {}
            row["full_val_loss"] = met.get("full_val_loss")
            row["eval_val_loss"] = met.get("eval_val_loss")
            if status.get("status") == "running":
                row["live"] = _read_json(run_paths(runs_dir, run_id)["live"])
            live = row.get("live")
            if status.get("status") == "interrupted":  # its cost ends at the last heartbeat
                live = _read_json(run_paths(runs_dir, run_id)["live"])
            row["cost"] = costs.run_cost(status, live, rates, now)
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
        return "\n".join(lines[-max(1, min(tail, 5000)) :])

    @app.get("/api/runs/{run_id}/events", dependencies=[Depends(auth)])
    def get_events(run_id: str) -> dict:
        paths = run_paths(runs_dir, check_id(run_id))
        return {"sent": read_jsonl(paths["commands"]), "applied": read_jsonl(paths["events"])}

    @app.get("/api/runs/{run_id}/report", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def get_report(run_id: str) -> str:
        """The run's generation samples: from its eval (every evaluated model), else the old per-run report."""
        status = status_of(check_id(run_id))
        save_name = (status.get("args") or {}).get("save-name")
        if save_name:
            ev = _read_json(repo / "evals" / f"{Path(save_name).stem}.json")
            if isinstance(ev, dict) and ev.get("samples"):
                from mini_llm.samples import render_model

                return render_model(ev)
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

    def make_continuation(args: dict, name: str, ckpt: str, done: int | None, steps: int) -> dict:
        keep = [
            "n-embd",
            "n-head",
            "n-layer",
            "block-size",
            "dropout",
            "batch-size",
            "tokens",
            "val-tokens",
            "weight-decay",
            "seed",
            "eval-interval",
            "eval-batches",
            "eval-seed",
            "full-eval-interval",
            "min-lr",
        ]
        new = {k: args[k] for k in keep if k in args}
        lr = float(args.get("restart-lr") or args.get("lr") or 1e-3)
        stem = Path(ckpt).stem
        renamed = re.sub(r"_(\d+)k_", f"_{(done + steps) // 1000}k_", stem, count=1) if done else stem
        save_name = (renamed if renamed != stem else f"{stem}_cont") + "_resume.pt"
        new.update(
            {
                "resume": ckpt,
                "steps": steps,
                # its own cosine from a tenth of the previous peak: the anneal recipe used so far
                "restart-lr": float(f"{lr / 10:.3g}"),
                "save": True,
                "save-name": save_name,
                "sample-report": True,
                "plot-suffix": (str(args.get("plot-suffix") or "run") + "-resume")[:60],
            }
        )
        where = f"at step {done:,}, {steps:,} more to {done + steps:,}" if done else f"for {steps:,} more steps"
        return {"name": (name + "-resume")[:80], "kind": "train", "args": new, "note": f"resumes {ckpt} {where}"}

    def checkpoint_of(args: dict, run_id: str | None = None) -> str | None:
        if args.get("save") and args.get("save-name"):
            return f"checkpoints/{args['save-name']}"
        if run_id:
            log_path = runs_dir / f"{run_id}.log"
            found = (
                re.findall(r"^Saved to (\S+)$", log_path.read_text(errors="replace"), re.M) if log_path.exists() else []
            )
            return found[-1] if found else None
        return None

    NO_CKPT = "this job doesn't save a checkpoint (save is off, or no save-name), so there's nothing to resume"

    @app.get("/api/runs/{run_id}/continuation", dependencies=[Depends(auth)])
    def run_continuation(run_id: str, steps: int = 40000) -> dict:
        """A ready-to-edit job resuming this run's checkpoint. Works for a run that's
        still going: its checkpoint counts as pending, so the job validates now."""
        status = status_of(check_id(run_id))
        args = status.get("args") or {}
        if status.get("kind", "train") != "train":
            raise HTTPException(409, "only training runs can be continued")
        ckpt = checkpoint_of(args, run_id)
        if not ckpt:
            raise HTTPException(409, NO_CKPT)
        last = (status.get("metrics") or {}).get("last_step")
        done = last + 1 if isinstance(last, int) else (None if "resume" in args else int(args.get("steps", 0)))
        return make_continuation(args, str(status.get("name") or run_id), ckpt, done, steps)

    @app.get("/api/queue/{file}/continuation", dependencies=[Depends(auth)])
    def queued_continuation(file: str, steps: int = 40000) -> dict:
        """Same, for a job that hasn't started yet."""
        if not RUN_ID.match(file) or not (queue_dir / file).exists():
            raise HTTPException(404, "no such queued job")
        job = _read_json(queue_dir / file) or {}
        if job.get("kind", "train") != "train":
            raise HTTPException(409, "only training jobs can be continued")
        args = job.get("args") or {}
        ckpt = checkpoint_of(args)
        if not ckpt:
            raise HTTPException(409, NO_CKPT)
        done = None if "resume" in args else int(args.get("steps", 0))
        return make_continuation(args, str(job.get("name") or Path(file).stem), ckpt, done, steps)

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
        if status.get("remote"):
            # Mirrored from Modal: nothing there reads a commands file, and live
            # control is off under DDP anyway. Say so rather than queue it silently.
            raise HTTPException(409, "live control isn't available for Modal runs")
        try:
            return append_command(runs_dir, run_id, body)
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc

    # --- queue ---------------------------------------------------------------

    @app.get("/api/queue", dependencies=[Depends(auth)])
    def list_queue() -> list[dict]:
        running = running_statuses()
        return [
            {"file": path.name, "job": job, "running": job_is_running(path, job, running)}
            for path, job in queued_jobs()
        ]

    @app.post("/api/jobs", dependencies=[Depends(auth)])
    def submit_job(body: dict, dry_run: bool = False) -> dict:
        """Validate with the runner's own rules, then drop the job into queue/.

        Written to a dotfile first and renamed, so the runner (which globs
        *.json) never sees a half-written job.
        """
        target = body.get("target", "local") if isinstance(body, dict) else "local"
        if target not in ("local", "modal"):
            raise HTTPException(422, f"target must be 'local' or 'modal', got {target!r}")
        job = {k: v for k, v in body.items() if k not in MODAL_KEYS} if isinstance(body, dict) else body
        try:
            name, kind, cmd, args = runner.validate_job(job, repo, uv, pending=pending_outputs())
        except runner.JobError as exc:
            raise HTTPException(422, str(exc)) from exc
        if target == "modal":
            preview, job = plan_modal(name, kind, args, body)
            if dry_run:
                return {"ok": True, **preview}
            start_modal([job])
            return preview
        forecast = runner.forecast_for(repo, args) if kind == "train" else None
        if dry_run:
            return {"ok": True, "name": name, "kind": kind, "argv": cmd[4:], "forecast": forecast}
        fname = enqueue(name, kind, args)
        return {"file": fname, "name": name, "kind": kind, "argv": cmd[4:], "forecast": forecast}

    def enqueue(name: str, kind: str, args: dict) -> str:
        queue_dir.mkdir(exist_ok=True)
        fname = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}.json"
        tmp = queue_dir / f".{fname}.tmp"
        tmp.write_text(json.dumps({"name": name, "kind": kind, "args": args}, indent=2))
        tmp.replace(queue_dir / fname)
        return fname

    def plan_modal(name: str, kind: str, args: dict, body: dict) -> tuple[dict, dict]:
        """Check a validated job against Modal's rules -> (preview for the page, job for start_modal).

        Modal jobs skip queue/ entirely: the queue runs one local job at a
        time, and a remote run has no reason to wait behind a long Mac run.
        The run shows up in runs/ at once ("launching"), and the mirror
        (mini_llm.remote.modal_mirror) takes it from there.
        """
        if kind != "train":
            raise HTTPException(422, "only train jobs can run on Modal")
        # 1xH100: the cheapest per run at this model size, and ~3x faster than 2xL4 (measured 2026-10-07).
        gpus = str(body.get("gpus") or "H100")
        if not MODAL_GPUS.match(gpus):
            raise HTTPException(422, f"gpus must look like L4:2, A100-80GB:4, H100 or cpu; got {gpus!r}")
        nproc = 2 if gpus == "cpu" else int(gpus.partition(":")[2] or 1)
        try:
            timeout_hours = float(body.get("timeout_hours", 6))
        except (TypeError, ValueError):
            raise HTTPException(422, "timeout_hours must be a number") from None
        if not 0.1 <= timeout_hours <= 24:
            raise HTTPException(422, "timeout_hours must be between 0.1 and 24 (Modal's maximum)")
        # The runner's defaults (data paths, plot-loss) apply on Modal too. --baseline
        # is dropped: the mirror's import writes the baselines row for Modal runs.
        effective = {k: v for k, v in {**runner.DEFAULT_ARGS, **args}.items() if k != "baseline"}
        batch = int(effective.get("batch-size", 4))
        if batch % nproc:
            raise HTTPException(
                422, f"batch-size {batch} is the global batch; it must divide evenly across {nproc} GPUs"
            )
        if importlib.util.find_spec("modal") is None:
            raise HTTPException(503, "the modal package isn't installed here: run `uv sync --group modal`")
        from mini_llm.remote.modal_train import config_to_argv, make_run_id

        run_id = make_run_id(name)
        preview = {
            "target": "modal",
            "name": name,
            "kind": kind,
            "gpus": gpus,
            "nproc": nproc,
            "timeout_hours": timeout_hours,
            "run_id": run_id,
            "hourly_usd": costs.list_hourly(gpus, costs.load_rates(runs_dir)),
            # What torchrun will actually run on Modal (not the local queue's argv).
            "argv": ["mini-llm-train", *config_to_argv({"args": effective})],
        }
        job = {"run_id": run_id, "name": name, "args": effective, "gpus": gpus, "timeout_hours": timeout_hours}
        return preview, job

    def start_modal(jobs: list[dict]) -> None:
        """Write the jobs and hand them to ONE background mini_llm.remote.launch, which launches them in
        order (a sweep's runs then never race each other to upload missing data)."""
        from mini_llm.remote.launch import launching_status, write_status

        runs_dir.mkdir(exist_ok=True)
        job_files = []
        for job in jobs:
            job_file = runs_dir / f"{job['run_id']}.modal-job.json"
            job_file.write_text(json.dumps(job, indent=2))
            write_status(repo, job["run_id"], launching_status(job["run_id"], job))  # visible at once
            job_files.append(str(job_file))
        subprocess.Popen(
            [sys.executable, "-m", "mini_llm.remote.launch", *job_files, "--repo", str(repo), "--uv", uv],
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # survives a server restart mid-launch
        )

    # --- sweeps: one base job, one flag varied (mini_llm.sweeps) ---------------

    sweeps_dir = runs_dir / "sweeps"

    @app.post("/api/sweeps", dependencies=[Depends(auth)])
    def submit_sweep(body: dict, dry_run: bool = False) -> dict:
        """Validate every child before launching any; launch them in order (Modal) or queue them (local)."""
        try:
            sweep = sweeps.expand(body, runner.INT_FLAGS, runner.FLOAT_FLAGS)
        except sweeps.SweepError as exc:
            raise HTTPException(422, str(exc)) from exc
        target = body.get("target", "local")
        if target not in ("local", "modal"):
            raise HTTPException(422, f"target must be 'local' or 'modal', got {target!r}")
        planned = []  # (value, name, kind, args, preview, modal job or None)
        pending = pending_outputs()
        for value, child in sweep.children:
            job = {k: v for k, v in child.items() if k not in MODAL_KEYS}
            try:
                name, kind, cmd, args = runner.validate_job(job, repo, uv, pending=pending)
            except runner.JobError as exc:
                raise HTTPException(422, f"{child['name']}: {exc}") from exc
            if target == "modal":
                preview, modal_job = plan_modal(name, kind, args, child)
            else:
                preview, modal_job = {"name": name, "argv": cmd[4:]}, None
            planned.append((value, name, kind, args, preview, modal_job))
        sweep_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{sweep.name}"
        out = {
            "id": sweep_id,
            "name": sweep.name,
            "flag": sweep.flag,
            "values": sweep.values,
            "stop_after": sweep.stop_after,
            "target": target,
            "children": [p[4] for p in planned],
        }
        if dry_run:
            return {"ok": True, **out}
        children = []
        if target == "modal":
            start_modal([p[5] for p in planned])
            children = [{"value": p[0], "name": p[1], "key": p[5]["run_id"], "run_id": p[5]["run_id"]} for p in planned]
        else:
            for value, name, kind, args, _, _ in planned:
                children.append({"value": value, "name": name, "key": enqueue(name, kind, args)})
        from mini_llm.remote.launch import now_z

        record = {**{k: v for k, v in out.items() if k != "children"}, "created": now_z(), "children": children}
        sweeps_dir.mkdir(parents=True, exist_ok=True)
        tmp = sweeps_dir / f".{sweep_id}.json.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(sweeps_dir / f"{sweep_id}.json")
        return out

    def sweep_view(record: dict) -> dict:
        # Local children are keyed by their queue file until the runner starts them; find their run by it.
        by_job_file = {}
        if any("run_id" not in c for c in record["children"]):
            for path in runs_dir.glob("*.status.json"):
                st = _read_json(path) or {}
                if st.get("job_file"):
                    by_job_file[st["job_file"]] = st.get("run_id") or path.name.removesuffix(".status.json")
        statuses, cost_blocks = {}, {}
        rates, now = costs.load_rates(runs_dir), time.time()
        for c in record["children"]:
            run_id = c.get("run_id") or by_job_file.get(c["key"])
            st = _read_json(runs_dir / f"{run_id}.status.json") if run_id else None
            if st is not None and st.get("status") == "running":
                st["live"] = _read_json(run_paths(runs_dir, run_id)["live"])
            statuses[c["key"]] = st
            cost_blocks[c["key"]] = costs.run_cost(st, st.get("live"), rates, now) if st else None
        return sweeps.summarize(record, statuses, cost_blocks)

    @app.get("/api/sweeps", dependencies=[Depends(auth)])
    def list_sweeps(limit: int = 20) -> list[dict]:
        paths = sorted(sweeps_dir.glob("*.json"), reverse=True)[: max(1, min(limit, 200))]
        return [sweep_view(r) for r in (_read_json(p) for p in paths) if r]

    @app.get("/api/sweeps/{sweep_id}", dependencies=[Depends(auth)])
    def get_sweep(sweep_id: str) -> dict:
        record = _read_json(sweeps_dir / f"{check_id(sweep_id)}.json")
        if record is None:
            raise HTTPException(404, f"no sweep {sweep_id}")
        return sweep_view(record)

    # --- checkpoints, inference, evals -------------------------------------
    #
    # The server otherwise never touches torch; these import it lazily. Inference
    # runs on the CPU by default so it never competes with a training job for
    # the Mac's GPU. One lock serialises generation, so a seeded request gets
    # the same random stream every time (torch's RNG is process-global).

    ckpt_dir = repo / "checkpoints"
    meta_cache: dict[tuple[str, float], dict] = {}
    model_cache: OrderedDict = OrderedDict()
    gen_lock = threading.Lock()
    tokenizer_box: list = []

    def ckpt_path(name: str) -> Path:
        if not isinstance(name, str) or not CKPT_NAME.match(name) or not (ckpt_dir / name).is_file():
            raise HTTPException(404, f"no checkpoint {name!r} in checkpoints/")
        return ckpt_dir / name

    def ckpt_meta(path: Path) -> dict:
        key = (path.name, path.stat().st_mtime)
        if key not in meta_cache:
            import torch

            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            fv = ck.get("full_val_history") or []
            meta_cache[key] = {
                "config": ck.get("config"),
                "step": ck.get("step"),
                "full_val_loss": fv[-1][1] if fv else None,
                "systems": ck.get("systems"),
            }
        return meta_cache[key]

    def load_for_inference(path: Path, device: str):
        import torch
        from mini_llm.config import ModelConfig, build_model

        key = (path.name, path.stat().st_mtime, device)
        if key in model_cache:
            model_cache.move_to_end(key)
            return model_cache[key]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(ck["config"])
        model = build_model(cfg)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        model_cache[key] = (model, cfg)
        while len(model_cache) > 2:  # a 16M-param model is ~65 MB; keep the last two
            model_cache.popitem(last=False)
        return model, cfg

    @app.get("/api/checkpoints", dependencies=[Depends(auth)])
    def list_checkpoints() -> list[dict]:
        out = []
        for path in sorted(ckpt_dir.glob("*.pt"), key=lambda q: q.stat().st_mtime, reverse=True):
            if not CKPT_NAME.match(path.name):
                continue
            row = {
                "name": path.name,
                "size_mb": round(path.stat().st_size / 2**20, 1),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"),
            }
            try:
                row.update(ckpt_meta(path))
            except Exception as exc:  # noqa: BLE001 - one unreadable file shouldn't hide the rest
                row["error"] = f"{type(exc).__name__}: {exc}"
            out.append(row)
        return out

    def inference_request(body: dict):
        """Validate the fields /api/generate and /api/next_token share -> (path, prompt, device, tokenizer).

        The model itself is loaded by the caller *inside* gen_lock: moving a model onto MPS while another
        request runs on it trips a Metal assertion that aborts the whole server.
        """
        import torch
        from mini_llm.data import get_tokenizer

        path = ckpt_path(body.get("checkpoint"))
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str) or len(prompt) > 20_000:
            raise HTTPException(422, "prompt must be a string of at most 20,000 characters")
        device = body.get("device", "cpu")
        if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
            raise HTTPException(422, "device must be cpu (default) or mps (if available)")
        if not tokenizer_box:
            tokenizer_box.append(get_tokenizer())
        return path, prompt, device, tokenizer_box[0]

    def prompt_ids(prompt: str, tokenizer):
        import torch
        from mini_llm.data import encode

        return encode(prompt, tokenizer) if prompt else torch.tensor([EOS_TOKEN_ID])

    @app.post("/api/generate", dependencies=[Depends(auth)])
    def generate(body: dict) -> dict:
        import torch
        from mini_llm.data import decode
        from mini_llm.report import generate_until_eos

        knobs = {}
        for k, (lo, hi, default) in SAMPLING.items():
            v = body.get(k, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = None
            if isinstance(lo, int) and v is not None and v.is_integer():
                v = int(v)
            if v is None or (isinstance(lo, int) and not isinstance(v, int)) or not lo <= v <= hi:
                raise HTTPException(
                    422, f"{k} must be {'an integer' if isinstance(lo, int) else 'a number'} from {lo} to {hi}"
                )
            knobs[k] = v
        try:
            seed = None if body.get("seed") in (None, "") else int(body["seed"])
        except (TypeError, ValueError):
            raise HTTPException(422, "seed must be an integer") from None
        path, prompt, device, tokenizer = inference_request(body)
        temperature, top_k, top_p = knobs["temperature"], knobs["top_k"], knobs["top_p"]
        greedy = temperature == 0
        kwargs = (
            {"greedy": True}
            if greedy
            else {"temperature": temperature, "top_k": top_k or None, "top_p": top_p if top_p < 1 else None}
        )
        with gen_lock:
            model, cfg = load_for_inference(path, device)
            idx = prompt_ids(prompt, tokenizer).unsqueeze(0).to(device)
            if seed is not None:
                torch.manual_seed(seed)
            t = time.perf_counter()
            out, hit_eos = generate_until_eos(
                model,
                idx,
                knobs["max_new_tokens"],
                cfg.block_size,
                eos_token_id=EOS_TOKEN_ID if body.get("stop_at_eos", True) else None,
                **kwargs,
            )
            seconds = time.perf_counter() - t
        n_prompt, n_new = idx.size(1), out.size(1) - idx.size(1)
        return {
            "checkpoint": path.name,
            "device": device,
            "greedy": greedy,
            "temperature": temperature,
            "top_k": None if greedy else top_k or None,
            "top_p": None if greedy or top_p >= 1 else top_p,
            "seed": seed,
            "prompt": prompt,
            "completion": decode(out[0, n_prompt:], tokenizer),
            "prompt_tokens": n_prompt,
            "new_tokens": n_new,
            "hit_eos": hit_eos,
            "block_size": cfg.block_size,
            "prompt_truncated": n_prompt > cfg.block_size,
            "seconds": round(seconds, 3),
            "tokens_per_sec": round(n_new / seconds, 1) if seconds > 0 else None,
        }

    @app.post("/api/next_token", dependencies=[Depends(auth)])
    def next_token(body: dict) -> dict:
        """The model's raw next-token logits after the prompt, for the page to preview decoding settings.

        All logits go back (base64 float32, ~200 KB) so the page can apply any
        temperature / top-k / top-p exactly, on every keystroke, without another
        forward pass; `top` names the 200 highest, the only ones any setting can
        make visible.
        """
        import base64
        import torch

        path, prompt, device, tokenizer = inference_request(body)
        ids = prompt_ids(prompt, tokenizer)
        with gen_lock, torch.no_grad():
            model, cfg = load_for_inference(path, device)
            logits, _ = model(ids[-cfg.block_size :].unsqueeze(0).to(device))
        z = logits[0, -1].float().cpu()
        top = torch.topk(z, 200)
        return {
            "checkpoint": path.name,
            "prompt_tokens": ids.numel(),
            "block_size": cfg.block_size,
            "logits": base64.b64encode(z.numpy().astype("<f4").tobytes()).decode(),
            "top": [{"id": i, "text": tokenizer.decode([i])} for i in top.indices.tolist()],
        }

    @app.get("/api/evals", dependencies=[Depends(auth)])
    def list_evals() -> dict:
        """Summary, guide, cross-model reports, and one row per model report (for sorting on the page)."""
        d = repo / "evals"
        text = lambda n: (d / f"{n}.md").read_text() if (d / f"{n}.md").exists() else None
        shared = [n for n in ("samples", "inference", "kv_reference") if (d / f"{n}.md").exists()]
        rows = []
        for md in (sorted(d.glob("*.md")) if d.exists() else []):
            if md.stem in ("summary", "GUIDE", *shared) or md.stem.startswith("sweep"):
                continue
            ev = _read_json(md.with_suffix(".json"))
            row = {
                "name": md.stem,
                "label": md.stem,
                "val": None,
                "ctx": None,
                "params": None,
                "date": datetime.fromtimestamp(
                    (repo / "checkpoints" / f"{md.stem}.pt").stat().st_mtime
                    if (repo / "checkpoints" / f"{md.stem}.pt").exists()
                    else md.stat().st_mtime
                ).isoformat(timespec="minutes"),
            }
            if isinstance(ev, dict) and "config" in ev:
                from mini_llm.samples import label

                T = ev["config"]["block_size"]
                row.update(
                    label=label(ev), ctx=T, params=ev.get("params"), val=(ev.get("quality") or {}).get(f"full_val@{T}")
                )
            rows.append(row)
        return {"summary": text("summary"), "guide": text("GUIDE"), "shared": shared, "reports": rows}

    @app.get("/api/evals/{name}", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def get_eval(name: str) -> str:
        path = repo / "evals" / f"{name}.md"
        if not EVAL_NAME.match(name) or not path.is_file():
            raise HTTPException(404, f"no eval report {name!r}")
        return path.read_text()

    @app.delete("/api/queue/{file}", dependencies=[Depends(auth)])
    def cancel_job(file: str) -> dict:
        if not RUN_ID.match(file) or not file.endswith(".json"):
            raise HTTPException(400, "bad job file name")
        path = queue_dir / file
        if not path.exists():
            raise HTTPException(404, "not queued (already started or finished?)")
        if job_is_running(path, _read_json(path) or {}, running_statuses()):
            raise HTTPException(409, "that job is running now -- use Stop on the Live tab instead")
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
