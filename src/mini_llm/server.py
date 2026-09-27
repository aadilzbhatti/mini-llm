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


# Which flags the phone form shows up front; everything else sits under "More options".
COMMON_FLAGS = {
    "train": ["tokens", "val-tokens", "resume", "restart-lr", "steps", "n-embd", "n-head", "n-layer",
              "block-size", "lr", "min-lr", "warmup-steps", "full-eval-interval", "save", "save-name",
              "sample-report", "plot-suffix"],
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


def create_app(repo: Path | str | None = None, token: str | None = None,
               tensorboard_url: str | None = None, uv: str | None = None) -> FastAPI:
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
            if "job_file" not in st and st.get("args") == job.get("args", {k: v for k, v in job.items() if k not in ("name", "kind")}):
                return True
        return False

    def pending_outputs() -> frozenset[str]:
        """Checkpoints that queued or running jobs will save: valid resume targets
        for a continuation queued behind them."""
        out = set()
        arg_sets = [job.get("args") or {} for _, job in queued_jobs()] + [st.get("args") or {} for st in running_statuses()]
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
            have = [rel(f) for f in sorted(ckpts.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)] if ckpts.exists() else []
            return sorted(pending - set(have)) + have
        return []

    def job_schema() -> dict:
        pending = pending_outputs()
        kinds: dict[str, list[dict]] = {}
        for kind, groups in {
            "train": [("int", runner.INT_FLAGS), ("float", runner.FLOAT_FLAGS),
                      ("path", {f: None for f in runner.PATH_FLAGS}), ("name", {f: None for f in runner.NAME_FLAGS}),
                      ("bool", {f: None for f in runner.BOOL_FLAGS})],
            "prepare-data": [("int", runner.PREP_INT_FLAGS), ("float", runner.PREP_FLOAT_FLAGS),
                             ("str", {f: None for f in runner.PREP_STR_FLAGS}), ("outdir", {"out-dir": None})],
        }.items():
            fields = []
            info = parser_info.get(kind, {})
            defaults = runner.DEFAULT_ARGS if kind == "train" else {}
            for ftype, flags in groups:
                for flag, bounds in flags.items():
                    field = {"flag": flag, "type": ftype, "help": info.get(flag, {}).get("help", ""),
                             "default": defaults.get(flag, info.get(flag, {}).get("default")),
                             "common": flag in COMMON_FLAGS[kind]}
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
            "form": job_schema(),
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
                   ("run_id", "name", "kind", "status", "started", "finished", "duration_sec", "error", "remote")}
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

    def make_continuation(args: dict, name: str, ckpt: str, done: int | None, steps: int) -> dict:
        keep = ["n-embd", "n-head", "n-layer", "block-size", "dropout", "batch-size", "tokens", "val-tokens",
                "weight-decay", "seed", "eval-interval", "eval-batches", "eval-seed", "full-eval-interval", "min-lr"]
        new = {k: args[k] for k in keep if k in args}
        lr = float(args.get("restart-lr") or args.get("lr") or 1e-3)
        stem = Path(ckpt).stem
        renamed = re.sub(r"_(\d+)k_", f"_{(done + steps) // 1000}k_", stem, count=1) if done else stem
        save_name = (renamed if renamed != stem else f"{stem}_cont") + "_resume.pt"
        new.update({
            "resume": ckpt, "steps": steps,
            # its own cosine from a tenth of the previous peak: the anneal recipe used so far
            "restart-lr": float(f"{lr / 10:.3g}"),
            "save": True, "save-name": save_name, "sample-report": True,
            "plot-suffix": (str(args.get("plot-suffix") or "run") + "-resume")[:60],
        })
        where = f"at step {done:,}, {steps:,} more to {done + steps:,}" if done else f"for {steps:,} more steps"
        return {"name": (name + "-resume")[:80], "kind": "train", "args": new, "note": f"resumes {ckpt} {where}"}

    def checkpoint_of(args: dict, run_id: str | None = None) -> str | None:
        if args.get("save") and args.get("save-name"):
            return f"checkpoints/{args['save-name']}"
        if run_id:
            log_path = runs_dir / f"{run_id}.log"
            found = re.findall(r"^Saved to (\S+)$", log_path.read_text(errors="replace"), re.M) if log_path.exists() else []
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
        return [{"file": path.name, "job": job, "running": job_is_running(path, job, running)}
                for path, job in queued_jobs()]

    @app.post("/api/jobs", dependencies=[Depends(auth)])
    def submit_job(body: dict, dry_run: bool = False) -> dict:
        """Validate with the runner's own rules, then drop the job into queue/.

        Written to a dotfile first and renamed, so the runner (which globs
        *.json) never sees a half-written job.
        """
        try:
            name, kind, cmd, args = runner.validate_job(body, repo, uv, pending=pending_outputs())
        except runner.JobError as exc:
            raise HTTPException(422, str(exc)) from exc
        forecast = runner.forecast_for(repo, args) if kind == "train" else None
        if dry_run:
            return {"ok": True, "name": name, "kind": kind, "argv": cmd[4:], "forecast": forecast}
        queue_dir.mkdir(exist_ok=True)
        fname = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}.json"
        tmp = queue_dir / f".{fname}.tmp"
        tmp.write_text(json.dumps({"name": name, "kind": kind, "args": args}, indent=2))
        tmp.replace(queue_dir / fname)
        return {"file": fname, "name": name, "kind": kind, "argv": cmd[4:], "forecast": forecast}

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
