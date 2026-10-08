"""Run mini_llm.train as a one-off multi-GPU (DDP) job on Modal.

    uv run --group modal modal run --detach src/mini_llm/remote/modal_train.py \\
        --config configs/modal/smoke.json --gpus H100:2

What happens:
  * Locally (this file's local entrypoint): read the config, make a run id
    (UTC timestamp + short git sha [+ name]), record the git state, and call
    the remote function with the requested GPU spec.
  * On Modal: one container with N GPUs runs
        torchrun --nproc_per_node=N -m mini_llm.train <flags>
    with its working directory set to /runs/<run_id>, so everything train.py
    writes (checkpoints/, plots/, runs/tb/, train.log) lands in the
    `wiki-llm-runs` volume under that run id, next to run.json (resolved
    config, git sha, torch/CUDA versions, GPU names).

Config: a JSON file of train.py flags, in the queue runner's job format
({"name": ..., "args": {"n-embd": 256, "save": true, ...}}, so any
runner/*.json job works) or just the flat {"flag": value} dict. `true`
passes a bare flag and `false` omits it. --args "--steps 50 --bf16" appends
raw flags on top (argparse lets the last one win).

Data: relative --tokens/--val-tokens/--text/--resume paths are looked up
in the `wiki-llm-data` volume, which mirrors the local data/ tree, so
data/data10k/train.pt resolves to /data/data10k/train.pt. Upload once with
    uv run --group modal modal volume put wiki-llm-data data/data10k /data10k
data/tiny.txt is baked into the image, so the smoke config needs no upload.
Absolute paths (e.g. /runs/<earlier run>/checkpoints/x.pt for --resume) pass
through unchanged.

Fetch outputs:  scripts/fetch_modal_run.sh <run_id>   (-> ./runs/<run_id>)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

RUNS_VOLUME, RUNS_MOUNT = "wiki-llm-runs", "/runs"
DATA_VOLUME, DATA_MOUNT = "wiki-llm-data", "/data"
BUNDLED_DATA = "/opt/mini-llm/data"  # data/tiny.txt, baked into the image
PATH_FLAGS = {"--tokens", "--val-tokens", "--text", "--resume"}
# A remote run is useless without its outputs. Tokens are read into RAM up
# front: the data volume is network-backed, so memory-mapping it would turn
# every cold crop into a remote read, and containers have RAM to spare.
DEFAULT_ARGS = {"save": True, "plot-loss": True, "tokens-in-ram": True}
# Not run on Modal: generating the sample report there holds the GPUs (billed) for
# minutes of uncached inference. The flag stays in the config as a request, and the
# Mac's auto-eval writes the report after import (mini_llm.auto_eval).
LOCAL_ONLY_FLAGS = {"sample-report", "sample-report-tokens"}
# Push partial outputs to the volume this often, so a live run can be fetched and
# mini_llm.remote.modal_mirror can show its progress in the control web app.
COMMIT_EVERY_SEC = 30

runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

# This module is imported again inside the container, where it lives at
# /root/modal_train.py and there is no repo to point at.
REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")

image = (
    modal.Image.debian_slim(python_version="3.13")
    # Third-party deps exactly as locked in uv.lock (uv sync --frozen), minus
    # the dev group. The Linux torch wheel in the lock is the CUDA build.
    .uv_sync(str(REPO), extra_options="--no-dev")
    # Bake the GPT-2 tokenizer into the image, so N ranks don't race to
    # download it into the same cache at startup.
    .env({"HF_HOME": "/opt/hf"})
    .run_commands("python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')\"")
    .add_local_file(REPO / "data" / "tiny.txt", f"{BUNDLED_DATA}/tiny.txt")
    # The project's own code, straight from the working tree. Added last, so
    # an edit re-uploads only this layer and doesn't rebuild the deps.
    .add_local_python_source("mini_llm")
)

app = modal.App("wiki-llm-train", image=image)


# --- remote side ---------------------------------------------------------------


def resolve_data_path(value: str) -> str:
    """Map a repo-relative data path onto the data volume (or the bundled tiny.txt)."""
    if value.startswith("/"):
        return value
    rel = value.removeprefix("data/")
    for root in (DATA_MOUNT, BUNDLED_DATA):
        if (Path(root) / rel).exists():
            return str(Path(root) / rel)
    raise FileNotFoundError(
        f"{value!r} is not in the {DATA_VOLUME} volume. Upload it first:\n"
        f"  uv run --group modal modal volume put {DATA_VOLUME} {value} /{rel}"
    )


def resolve_args(argv: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        flag = argv[i]
        if "=" in flag and flag.split("=", 1)[0] in PATH_FLAGS:
            name, value = flag.split("=", 1)
            out.append(f"{name}={resolve_data_path(value)}")
        elif flag in PATH_FLAGS and i + 1 < len(argv):
            out += [flag, resolve_data_path(argv[i + 1])]
            i += 1
        else:
            out.append(flag)
        i += 1
    # No data flag at all would make train.py fall back to a hard-coded
    # snippet (data/tiny.txt isn't in the run dir). Point it at the real file.
    if not any(a.split("=", 1)[0] in {"--tokens", "--text"} for a in out):
        out += ["--text", f"{BUNDLED_DATA}/tiny.txt"]
    return out


def environment_info() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        info["nvidia_smi"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = None
    return info


@app.function(volumes={RUNS_MOUNT: runs_volume, DATA_MOUNT: data_volume}, timeout=24 * 3600)
def train_remote(run_id: str, train_argv: list[str], nproc: int, meta: dict) -> dict:
    import mini_llm

    run_dir = Path(RUNS_MOUNT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = resolve_args(train_argv)
    # One container = one node, so rendezvous over loopback. (Not --standalone,
    # which resolves the hostname and can hang where that doesn't resolve.)
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--master-addr=127.0.0.1",
        "--master-port=29500",
        "-m",
        "mini_llm.train",
        *argv,
    ]

    record = {
        **meta,
        "run_id": run_id,
        "nproc": nproc,
        "resolved_argv": argv,
        "command": shlex.join(cmd),
        "environment": environment_info(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta.get("git_diff"):
        # Uncommitted changes were part of what ran; keep them as a patch.
        (run_dir / "git.diff").write_text(record.pop("git_diff"))
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    runs_volume.commit()
    print(json.dumps({k: record[k] for k in ("run_id", "command", "environment")}, indent=2))

    env = {
        **os.environ,
        "MINI_LLM_RUN_ID": run_id,  # train.py names its TensorBoard dir etc. after it
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",  # tokenizer is baked into the image; don't have N ranks hit the Hub
        # The run dir is the cwd, so point Python at wherever Modal put the package.
        "PYTHONPATH": str(Path(mini_llm.__file__).resolve().parents[1]),
    }

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_SEC):
            runs_volume.commit()

    committer = threading.Thread(target=commit_periodically, daemon=True)
    committer.start()
    start = time.time()
    with open(run_dir / "train.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:  # tee: Modal's log stream and train.log
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        returncode = proc.wait()
    stop.set()
    committer.join()

    record.update(
        returncode=returncode,
        finished_at=datetime.now(timezone.utc).isoformat(),
        duration_sec=round(time.time() - start, 1),
    )
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    runs_volume.commit()
    if returncode != 0:
        raise RuntimeError(f"torchrun exited with {returncode}; see /runs/{run_id}/train.log")
    return {"run_id": run_id, "duration_sec": record["duration_sec"]}


# --- local side ----------------------------------------------------------------


def config_to_argv(config: dict) -> list[str]:
    argv: list[str] = []
    for flag, value in {**DEFAULT_ARGS, **config.get("args", config)}.items():
        if flag == "name" or flag in LOCAL_ONLY_FLAGS:
            continue
        if value is True:
            argv.append(f"--{flag}")
        elif value is not False and value is not None:
            argv += [f"--{flag}", str(value)]
    return argv


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def parse_gpus(spec: str) -> tuple[str | None, int]:
    """'H100:4' -> ('H100:4', 4); 'A100-80GB' -> (..., 1); 'cpu' -> (None, 2) for a gloo smoke test."""
    if spec.lower() == "cpu":
        return None, 2
    _, _, count = spec.partition(":")
    return spec, int(count or 1)


def git_state() -> tuple[str, bool]:
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))


def make_run_id(name: str = "") -> str:
    """<UTC timestamp>-<short sha>[-dirty][-name]. Shared with mini_llm.remote.launch,
    which needs the id before the launch so the web page can show the run at once."""
    sha, dirty = git_state()
    return "-".join(
        filter(
            None, [datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), sha[:7] + ("-dirty" if dirty else ""), name]
        )
    )


@app.local_entrypoint()
def main(
    config: str = "",
    gpus: str = "H100:2",
    args: str = "",
    name: str = "",
    timeout_hours: float = 24.0,
    run_id: str = "",
    wait: bool = True,
):
    """--run-id: use this id instead of making one. --no-wait: submit and exit
    without streaming the run (with --detach the run carries on in Modal)."""
    cfg = json.loads(Path(config).read_text()) if config else {}
    train_argv = config_to_argv(cfg) + shlex.split(args)
    gpu, nproc = parse_gpus(gpus)

    sha, dirty = git_state()
    name = name or cfg.get("name", "")
    run_id = run_id or make_run_id(name)
    meta = {
        "config_file": config or None,
        "config": cfg,
        "extra_args": args,
        "train_argv": train_argv,
        "gpus": gpus,
        "git_sha": sha,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": dirty,
        "git_diff": git("diff", "HEAD") if dirty else "",
        # Modal bills per App; the mirror looks the run's cost up by this id (mini_llm.remote.costs).
        "modal_app_id": app.app_id,
    }

    options: dict = {"timeout": int(timeout_hours * 3600)}
    if gpu is None:
        options["cpu"] = 4.0  # two gloo ranks on CPU
    else:
        options["gpu"] = gpu
    print(f"run id:  {run_id}")
    print(f"gpus:    {gpus} -> torchrun --nproc_per_node={nproc}")
    print(f"argv:    {shlex.join(train_argv)}")
    print(f"fetch:   scripts/fetch_modal_run.sh {run_id}")
    fn = train_remote.with_options(**options)
    if not wait:
        call = fn.spawn(run_id, train_argv, nproc, meta)
        print(f"spawned: {call.object_id}")
        return
    result = fn.remote(run_id, train_argv, nproc, meta)
    print(f"done:    {result}")
