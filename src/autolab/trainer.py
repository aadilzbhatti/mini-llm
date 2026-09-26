"""Launch one training run of `mini_llm.train` as a subprocess and report on it.

Layout of a run directory (`autolab/runs/<run_id>/`, also the trainer's cwd,
since train.py writes plots/, checkpoints/ and runs/ relative to cwd):

    launch.json                     request, argv, git commit, timing, budget outcome
    train.log                       trainer stdout + stderr
    runs/<run_id>.{live.json,commands.jsonl,events.jsonl}
    runs/tb/<run_id>/events.out.tfevents.*
    report.json                     built by autolab.report after the run

Budget: `tokens` becomes `--steps` (rounded up to whole batches); `wall_clock_s`
is enforced by appending a `stop` command to the trainer's control inbox, so
the run still does its final full eval. A hard kill only happens if the
trainer ignores the stop for `kill_grace_s`.

    uv run python -m autolab.trainer --run-id m2-check --tokens 1000000 --wall-clock 420
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mini_llm.control import append_command, read_jsonl

from autolab.config import AutolabConfig, check_frozen_val, load_config, sha256_file
from autolab.gpu import wait_for_gpu

# Trainer CLI defaults (REPO_NOTES §1), spelled out so every report carries the full config.
DEFAULT_MODEL = {"block_size": 64, "n_embd": 128, "n_head": 4, "n_layer": 4, "dropout": 0.0}
DEFAULT_OPTIM = {"batch_size": 4, "lr": 1e-3, "min_lr": 2e-6, "warmup_steps": 500, "weight_decay": 0.0}
DEFAULT_EVAL = {"eval_interval": 100, "eval_batches": 20, "eval_seed": 1234, "log_interval": 10,
                "full_eval_interval": 0, "control_poll": 25}


@dataclass
class Budget:
    tokens: int
    wall_clock_s: float | None = None


@dataclass
class TrainRequest:
    run_id: str
    dataset_id: str
    train_tokens: str
    budget: Budget
    seed: int = 42
    model: dict = field(default_factory=lambda: dict(DEFAULT_MODEL))
    optim: dict = field(default_factory=lambda: dict(DEFAULT_OPTIM))
    eval: dict = field(default_factory=lambda: dict(DEFAULT_EVAL))
    # Source tree to import mini_llm from (a candidate worktree's src/); None = this repo.
    code_src: str | None = None

    def steps(self) -> int:
        return max(1, math.ceil(self.budget.tokens / (self.optim["batch_size"] * self.model["block_size"])))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainRequest":
        return cls(**{**d, "budget": Budget(**d["budget"])})


def trainer_argv(req: TrainRequest, val_path: Path) -> list[str]:
    flags = {**req.model, **req.optim, **req.eval, "seed": req.seed, "steps": req.steps(),
             "tokens": str(Path(req.train_tokens).resolve()), "val_tokens": str(val_path.resolve())}
    argv = [sys.executable, "-m", "mini_llm.train"]
    for key, value in flags.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return argv  # never --baseline, never --no-tensorboard (HANDOFF)


def git_state(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def run_training(
    req: TrainRequest,
    cfg: AutolabConfig | None = None,
    wait_gpu: bool = True,
    env_extra: dict[str, str] | None = None,
    kill_grace_s: float = 900,
    poll_s: float = 1.0,
    log=print,
) -> Path:
    """Run the trainer to completion (or its wall-clock cap). Returns the run dir."""
    cfg = cfg or load_config()
    val_path = check_frozen_val(cfg)
    run_dir = cfg.runs_dir / req.run_id
    if (run_dir / "launch.json").exists():
        raise FileExistsError(f"run {req.run_id} already exists at {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_wait_s = 0.0
    if wait_gpu:
        gpu_wait_s = wait_for_gpu(cfg.owner_runs_dir, cfg.gpu_stale_after_s, cfg.gpu_poll_s,
                                  cfg.gpu_idle_checks_required, log=log)

    argv = trainer_argv(req, val_path)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "MINI_LLM_RUN_ID": req.run_id, **(env_extra or {})}
    code_repo = cfg.repo_root
    if req.code_src:
        env["PYTHONPATH"] = str(Path(req.code_src).resolve())
        code_repo = Path(req.code_src).resolve().parent
    launch = {
        "run_id": req.run_id,
        "request": req.to_dict(),
        "argv": argv,
        "steps": req.steps(),
        "tokens_per_step": req.optim["batch_size"] * req.model["block_size"],
        "val_path": str(val_path),
        "val_sha256": cfg.frozen_val_sha256,
        "train_sha256": sha256_file(Path(req.train_tokens)),
        "git": git_state(code_repo),
        "gpu_wait_s": gpu_wait_s,
        "started_at": now_iso(),
        "status": "running",
    }
    _write_json(run_dir / "launch.json", launch)

    t0 = time.time()
    stop_sent_at: float | None = None
    killed = False
    with open(run_dir / "train.log", "w") as logf:
        proc = subprocess.Popen(argv, cwd=run_dir, env=env, stdout=logf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(poll_s)
            elapsed = time.time() - t0
            cap = req.budget.wall_clock_s
            if cap is not None and stop_sent_at is None and elapsed >= cap:
                append_command(run_dir / "runs", req.run_id,
                               {"id": "autolab-wallclock", "type": "stop", "note": f"wall-clock cap {cap}s"})
                stop_sent_at = elapsed
                log(f"{req.run_id}: wall-clock cap {cap}s reached, stop sent")
            if stop_sent_at is not None and elapsed - stop_sent_at > kill_grace_s:
                proc.kill()
                killed = True
    wall_s = time.time() - t0

    stop_applied = any(
        e.get("type") == "stop" and e.get("id") == "autolab-wallclock"
        for e in read_jsonl(run_dir / "runs" / f"{req.run_id}.events.jsonl")
    )
    if proc.returncode != 0 or killed:
        hit = None
    elif stop_applied:
        hit = "wall_clock"
    else:
        hit = "tokens"
    launch.update({
        "ended_at": now_iso(),
        "wall_s": round(wall_s, 2),
        "returncode": proc.returncode,
        "killed": killed,
        "stop_sent_at_s": None if stop_sent_at is None else round(stop_sent_at, 2),
        "budget_hit": hit,
        "status": "finished" if proc.returncode == 0 and not killed else "failed",
    })
    _write_json(run_dir / "launch.json", launch)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    from autolab.diagnose import History, diagnose
    from autolab.report import build_report, write_report

    p = argparse.ArgumentParser(description="Run one autolab training run and write report.json.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset-id", default="data20k")
    p.add_argument("--tokens", type=int, required=True, help="Token budget (converted to --steps).")
    p.add_argument("--wall-clock", type=float, default=None, help="Wall-clock cap in seconds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu-wait", action="store_true", help="Skip the owner-runner check (tests only).")
    args = p.parse_args(argv)

    cfg = load_config()
    req = TrainRequest(
        run_id=args.run_id,
        dataset_id=args.dataset_id,
        train_tokens=str(cfg.datasets_dir / args.dataset_id / "train.pt"),
        budget=Budget(tokens=args.tokens, wall_clock_s=args.wall_clock),
        seed=args.seed,
    )
    run_dir = run_training(req, cfg, wait_gpu=not args.no_gpu_wait)
    report = build_report(run_dir)
    write_report(report, run_dir)
    diag = diagnose(report, History())
    (run_dir / "diagnosis.json").write_text(json.dumps(diag.to_dict(), indent=2))
    print(json.dumps(diag.to_dict(), indent=2))


if __name__ == "__main__":
    main()
