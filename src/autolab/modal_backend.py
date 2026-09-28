"""Run autolab trials on Modal: one GPU per trial, many trials in parallel.

    uv run --group modal python -m autolab.modal_backend deploy        # after changing autolab/ or deps
    uv run --group modal python -m autolab.modal_backend upload-data   # once: frozen val + train sets
    uv run --group modal python -m autolab.modal_backend submit --run-id x --tokens 1000000 --wall-clock 420
    uv run --group modal python -m autolab.modal_backend collect       # fetch finished trials
    uv run --group modal python -m autolab.modal_backend status

Remote side (`run_trial`): writes the trial's `mini_llm` source (sent with every
call, so candidate programs need no image rebuild) into /tmp/code/src, checks the
frozen val and train sha256 against what the Mac recorded, runs
`python -m mini_llm.train` (single GPU, no torchrun, so the control inbox and the
wall-clock stop work) via autolab.trainer.execute, builds report.json and the
diagnosis, copies the run dir to the `autolab-runs` volume and returns the results.

Local side: `submit` spawns calls on the deployed app and records them in
autolab/state/modal_calls.json; `collect` writes finished results into
autolab/runs/<run_id>/ (launch.json, report.json, diagnosis.json, train.log).
No network in training (BRIEF §10) is enforced by HF_HUB_OFFLINE=1 with the tokenizer baked into
the image. Modal's block_network can't be used: it also blocks the call's result upload.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import modal

IN_CONTAINER = not modal.is_local()
REPO = Path("/") if IN_CONTAINER else Path(__file__).resolve().parents[2]
RUNS_MOUNT, DATA_MOUNT = "/runs", "/data"


def _modal_cfg() -> dict:
    return tomllib.loads((REPO / "autolab" / "config.toml").read_text())["modal"] if not IN_CONTAINER else {}


_CFG = _modal_cfg()


def max_usd() -> float:
    """The Modal spend cap, re-read on every use so a change (e.g. from the dashboard) applies without a restart."""
    try:
        return float(_modal_cfg().get("max_usd", _CFG.get("max_usd", 25.0)))
    except (OSError, ValueError):
        return float(_CFG.get("max_usd", 25.0))
APP_NAME = _CFG.get("app_name", "autolab-train")
runs_volume = modal.Volume.from_name(_CFG.get("runs_volume", "autolab-runs"), create_if_missing=True)
data_volume = modal.Volume.from_name(_CFG.get("data_volume", "autolab-data"), create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(REPO), extra_options="--no-dev")
    .env({"HF_HOME": "/opt/hf"})
    .run_commands("python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')\"")
    # autolab needs thresholds.toml, which the default NON_PYTHON_FILES filter would drop.
    .add_local_python_source("autolab", "mini_llm", ignore=["**/__pycache__/**"])
)
app = modal.App(APP_NAME, image=image)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LIVE_SYNC_S = 30


def _sync_live(run_dir: Path, run_id: str, stop) -> None:
    """Push the trainer's heartbeat and log tail to the runs volume while it trains,
    so the Mac-side daemon (and the dashboard) can show progress."""
    dest = Path(RUNS_MOUNT) / run_id
    while not stop.wait(LIVE_SYNC_S):
        try:
            dest.mkdir(parents=True, exist_ok=True)
            live = run_dir / "runs" / f"{run_id}.live.json"
            if live.exists():
                shutil.copyfile(live, dest / "live.json")
            log = run_dir / "train.log"
            if log.exists():
                (dest / "train.tail.log").write_text(log.read_text(errors="replace")[-20_000:])
            runs_volume.commit()
        except Exception as exc:  # noqa: BLE001 - progress is best-effort
            print(f"live sync failed: {exc}", flush=True)


@app.function(
    volumes={RUNS_MOUNT: runs_volume, DATA_MOUNT: data_volume},
    gpu=_CFG.get("default_gpu", "L4"),
    timeout=4 * 3600,
    max_containers=int(_CFG.get("max_containers", 8)),
)
def run_trial(request: dict, meta: dict, code: dict[str, str]) -> dict:
    from autolab.diagnose import History, diagnose
    from autolab.report import build_report, write_report
    from autolab.trainer import TrainRequest, execute

    req = TrainRequest.from_dict(request)
    val_path = Path(DATA_MOUNT) / "val_frozen.pt"
    train_path = Path(DATA_MOUNT) / "datasets" / req.dataset_id / "train.pt"
    for path, want in ((val_path, meta["val_sha256"]), (train_path, meta["train_sha256"])):
        got = _sha256(path)
        if got != want:
            raise RuntimeError(f"{path}: sha256 {got} != expected {want}; refusing to train")

    src = Path("/tmp/code/src")
    shutil.rmtree(src, ignore_errors=True)
    for rel, text in code.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)

    run_dir = Path("/tmp/run") / req.run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    stop = threading.Event()
    syncer = threading.Thread(target=_sync_live, args=(run_dir, req.run_id, stop), daemon=True)
    syncer.start()
    try:
        launch = execute(
            req, run_dir, train_path, val_path,
            {**meta, "backend": "modal", "modal_task_id": os.environ.get("MODAL_TASK_ID")},
            env_extra={"PYTHONPATH": str(src), "PYTHONUNBUFFERED": "1"},
        )
    finally:
        stop.set()
        syncer.join()
    if req.suite and launch.get("status") == "finished":  # M8: quality / context / inference metrics
        import subprocess as sp

        r = sp.run([sys.executable, "-m", "autolab.evalsuite", str(run_dir), "--val", str(val_path)],
                   env={**os.environ, "PYTHONPATH": str(src), "PYTHONUNBUFFERED": "1"},
                   capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            (run_dir / "eval_error.txt").write_text((r.stdout + r.stderr)[-8000:])
    out: dict = {"launch": launch, "train_log": (run_dir / "train.log").read_text(errors="replace")[-400_000:]}
    try:
        report = build_report(run_dir)
        write_report(report, run_dir)
        diag = diagnose(report, History()).to_dict()
        (run_dir / "diagnosis.json").write_text(json.dumps(diag, indent=2))
        out.update(report=report, diagnosis=diag)
    except Exception as exc:  # noqa: BLE001 - a failed run still returns its log
        out["report_error"] = f"{type(exc).__name__}: {exc}"
    dest = Path(RUNS_MOUNT) / req.run_id
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(run_dir, dest, ignore=shutil.ignore_patterns("*.pt"))
    runs_volume.commit()
    return out


# --- local side -------------------------------------------------------------------


def _state_path() -> Path:
    return REPO / "autolab" / "state" / "modal_calls.json"


@contextmanager
def calls_lock():
    """Serialize read-modify-write of modal_calls.json across processes (CLI, daemon)."""
    import fcntl

    path = _state_path().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_calls() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_calls(calls: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(calls, indent=2))
    tmp.replace(p)


def price_per_s(gpu: str) -> float:
    prices = _CFG.get("gpu_usd_per_s", {})
    return float(prices.get(gpu.split(":")[0], max(prices.values(), default=0.0)))


def spend(calls: dict) -> tuple[float, float]:
    """(actual $ of finished calls, estimated $ of pending calls at their wall-clock caps)."""
    done = sum(c.get("usd", 0.0) for c in calls.values() if c["state"] != "pending")
    pending = sum(c.get("usd_estimate", 0.0) for c in calls.values() if c["state"] == "pending")
    return done, pending


def code_snapshot(src_root: Path | None = None) -> dict[str, str]:
    """Every .py file of mini_llm, relative to src/: the program the trial runs."""
    src_root = src_root or REPO / "src"
    return {str(p.relative_to(src_root)): p.read_text()
            for p in sorted((src_root / "mini_llm").rglob("*.py")) if "__pycache__" not in p.parts}


def submit(req, gpu: str | None = None, src_root: Path | None = None, startup_s: float = 180) -> dict:
    with calls_lock():
        return _submit(req, gpu, src_root, startup_s)


def _submit(req, gpu, src_root, startup_s) -> dict:
    from autolab.config import check_frozen_val, load_config, sha256_file
    from autolab.trainer import git_state

    cfg = load_config()
    check_frozen_val(cfg)
    calls = load_calls()
    if req.run_id in calls:
        raise FileExistsError(f"run {req.run_id} already submitted ({calls[req.run_id]['state']})")
    if req.budget.wall_clock_s is None:
        raise ValueError("Modal trials need a wall-clock cap (it bounds the cost estimate)")
    gpu = gpu or _CFG["default_gpu"]
    estimate = (req.budget.wall_clock_s + startup_s) * price_per_s(gpu)
    done, pending = spend(calls)
    cap = max_usd()
    if done + pending + estimate > cap:
        raise RuntimeError(f"cost cap: spent ${done:.2f} + pending ${pending:.2f} + this ${estimate:.2f} "
                           f"> max_usd ${cap}")
    meta = {
        "val_sha256": cfg.frozen_val_sha256,
        "train_sha256": sha256_file(Path(req.train_tokens)),
        "git": git_state(REPO if src_root is None else src_root.parent),
        "gpu": gpu,
        "gpu_wait_s": 0.0,
    }
    fn = modal.Function.from_name(APP_NAME, "run_trial").with_options(gpu=gpu)
    call = fn.spawn(req.to_dict(), meta, code_snapshot(src_root))
    calls[req.run_id] = {"call_id": call.object_id, "state": "pending", "gpu": gpu,
                         "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "usd_estimate": round(estimate, 4), "request": req.to_dict()}
    save_calls(calls)
    return calls[req.run_id]


def collect(runs_dir: Path | None = None, log=print) -> list[str]:
    """Write every finished call's results under autolab/runs/<run_id>/. Returns newly finished ids."""
    with calls_lock():
        return _collect(runs_dir, log)


def _collect(runs_dir: Path | None, log) -> list[str]:
    runs_dir = runs_dir or REPO / "autolab" / "runs"
    calls = load_calls()
    finished = []
    for run_id, c in calls.items():
        if c["state"] != "pending":
            continue
        try:
            if not c.get("call_id"):  # restored entry without a call id: read the result from the runs volume
                out = from_volume(run_id)
                if out is None or out["launch"].get("status") not in ("finished", "failed"):
                    continue
            else:
                out = modal.FunctionCall.from_id(c["call_id"]).get(timeout=0)
        except TimeoutError:
            continue
        except Exception as exc:  # noqa: BLE001 - the remote raised; record it
            out = from_volume(run_id)  # the run dir may have been saved before the failure
            if out is None:
                c.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:2000])
                c["usd"] = c["usd_estimate"]  # unknown runtime; count the cap
                finished.append(run_id)
                log(f"{run_id}: FAILED {c['error'][:200]}")
                continue
            c["call_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        d = runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "launch.json").write_text(json.dumps(out["launch"], indent=2))
        (d / "train.log").write_text(out["train_log"])
        for key in ("report", "diagnosis"):
            if key in out:
                (d / f"{key}.json").write_text(json.dumps(out[key], indent=2))
        wall = out["launch"].get("wall_s") or 0.0
        c.update(state=out["launch"]["status"], usd=round(wall * price_per_s(c["gpu"]), 4),
                 budget_hit=out["launch"].get("budget_hit"), report_error=out.get("report_error"),
                 full_val_loss=(out.get("report") or {}).get("summary", {}).get("final_full_val_loss"),
                 primary=(out.get("diagnosis") or {}).get("primary"))
        finished.append(run_id)
        log(f"{run_id}: {c['state']} full_val={c['full_val_loss']} diag={c['primary']} ${c['usd']:.3f}")
    save_calls(calls)
    return finished


def from_volume(run_id: str, volume=None) -> dict | None:
    """Rebuild a call's result from the run dir run_trial saved on the runs volume."""
    volume = volume or runs_volume
    out: dict = {}
    for name, key in (("launch.json", "launch"), ("report.json", "report"), ("diagnosis.json", "diagnosis"),
                      ("train.log", "train_log"), ("eval.json", "eval")):
        try:
            raw = b"".join(volume.read_file(f"/{run_id}/{name}"))
        except Exception:  # noqa: BLE001 - missing file
            continue
        out[key] = raw.decode(errors="replace") if key == "train_log" else json.loads(raw)
    if "launch" not in out:
        return None
    out.setdefault("train_log", "")
    return out


def fetch_live(runs_dir: Path | None = None, volume=None) -> int:
    """Copy each pending run's live.json / log tail from the volume to autolab/runs/<id>/."""
    volume = volume or runs_volume
    runs_dir = runs_dir or REPO / "autolab" / "runs"
    n = 0
    for run_id, c in load_calls().items():
        if c["state"] != "pending":
            continue
        for name in ("live.json", "train.tail.log"):
            try:
                raw = b"".join(volume.read_file(f"/{run_id}/{name}"))
            except Exception:  # noqa: BLE001 - not started yet
                continue
            d = runs_dir / run_id
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{name}.tmp"
            tmp.write_bytes(raw)
            tmp.replace(d / name)
            n += 1
    return n


def status_lines() -> list[str]:
    calls = load_calls()
    done, pending = spend(calls)
    lines = [f"{'run_id':40} {'state':9} {'gpu':6} {'full_val':>9} {'hit':10} {'diag':22} $"]
    for run_id, c in calls.items():
        fv = c.get("full_val_loss")
        lines.append(f"{run_id:40} {c['state']:9} {c['gpu']:6} {fv if fv is None else round(fv, 4)!s:>9} "
                     f"{c.get('budget_hit') or '':10} {c.get('primary') or '':22} "
                     f"{c.get('usd', c.get('usd_estimate', 0)):.3f}")
    lines.append(f"spent ${done:.2f} (lower bound), pending up to ${pending:.2f}, cap ${max_usd()}")
    return lines


def upload_data(log=print) -> None:
    from autolab.config import load_config

    cfg = load_config()
    wanted = {"/val_frozen.pt": cfg.frozen_val}
    for train in sorted(cfg.datasets_dir.glob("*/train.pt")):
        wanted[f"/datasets/{train.parent.name}/train.pt"] = train
    missing = []
    for remote, local in wanted.items():
        try:
            data_volume.listdir(remote)
            log(f"already on volume: {remote}")
        except Exception:  # noqa: BLE001 - not found
            missing.append((remote, local))
    if missing:
        with data_volume.batch_upload() as batch:
            for remote, local in missing:
                batch.put_file(str(local), remote)
                log(f"uploaded {local} -> {remote}")


def main(argv: list[str] | None = None) -> None:
    import subprocess

    from autolab.config import load_config
    from autolab.trainer import Budget, TrainRequest

    p = argparse.ArgumentParser(description="Autolab trials on Modal.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy")
    sub.add_parser("upload-data")
    sub.add_parser("collect")
    sub.add_parser("status")
    b = sub.add_parser("submit-batch", help="Submit every job in a JSON file (see autolab/experiments/).")
    b.add_argument("file", type=Path)
    s = sub.add_parser("submit")
    s.add_argument("--run-id", required=True)
    s.add_argument("--dataset-id", default="data20k")
    s.add_argument("--tokens", type=int, required=True)
    s.add_argument("--wall-clock", type=float, required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--gpu", default=None)
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a model/optim/eval field, e.g. --set lr=3e-3 --set n_embd=256")
    args = p.parse_args(argv)

    if args.cmd == "deploy":
        raise SystemExit(subprocess.call([sys.executable, "-m", "modal", "deploy", __file__], cwd=REPO))
    if args.cmd == "upload-data":
        return upload_data()
    if args.cmd == "collect":
        collect()
        return print("\n".join(status_lines()))
    if args.cmd == "status":
        return print("\n".join(status_lines()))

    cfg = load_config()
    if args.cmd == "submit-batch":
        batch = json.loads(args.file.read_text())
        calls = load_calls()
        for job in batch["jobs"]:
            if job["run_id"] in calls:
                print(f"skip {job['run_id']}: already submitted")
                continue
            r = TrainRequest(run_id=job["run_id"], dataset_id=job.get("dataset_id", batch.get("dataset_id", "data20k")),
                             train_tokens=str(cfg.datasets_dir / job.get("dataset_id", batch.get("dataset_id", "data20k"))
                                              / "train.pt"),
                             budget=Budget(tokens=job["tokens"], wall_clock_s=job["wall_clock_s"]),
                             seed=job.get("seed", 42))
            for section, values in (("model", {**batch.get("model", {}), **job.get("model", {})}),
                                    ("optim", {**batch.get("optim", {}), **job.get("optim", {})}),
                                    ("eval", {**batch.get("eval", {}), **job.get("eval", {})})):
                getattr(r, section).update(values)
            c = submit(r, job.get("gpu", batch.get("gpu")))
            print(f"submitted {r.run_id}: {c['call_id']} on {c['gpu']}, {r.steps()} steps, est ${c['usd_estimate']:.3f}")
        return
    req = TrainRequest(run_id=args.run_id, dataset_id=args.dataset_id,
                       train_tokens=str(cfg.datasets_dir / args.dataset_id / "train.pt"),
                       budget=Budget(tokens=args.tokens, wall_clock_s=args.wall_clock), seed=args.seed)
    for item in args.set:
        key, _, raw = item.partition("=")
        value = json.loads(raw) if raw not in ("", None) else None
        section = next((sec for sec in (req.model, req.optim, req.eval) if key in sec), None)
        if section is None:
            raise SystemExit(f"unknown field {key!r}")
        section[key] = value
    c = submit(req, args.gpu)
    print(f"submitted {args.run_id}: call {c['call_id']} on {c['gpu']}, est ${c['usd_estimate']:.3f}")


if __name__ == "__main__":
    main()
