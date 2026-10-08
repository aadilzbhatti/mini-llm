"""Launch a web-submitted training job on Modal, in the background.

The control server (mini_llm.server) validates a job with the queue
runner's rules, writes it to runs/<run_id>.modal-job.json, and starts

    python -m mini_llm.remote.launch runs/<run_id>.modal-job.json --repo <repo>

as a detached process, so the request returns at once. This then:

  1. writes runs/<run_id>.status.json as a running Modal run in its
     "launching" phase, so the page shows it immediately;
  2. uploads any repo-relative data files (--tokens, --val-tokens, --text,
     --resume) that the wiki-llm-data volume doesn't have yet;
  3. runs `modal run --detach modal_train.py --run-id <id> --no-wait`, which
     builds/updates the image, submits the run and exits (the run carries on
     in Modal);
  4. on failure, rewrites the status as failed with the error.

After a successful launch nothing here touches the run again: the mirror
(mini_llm.remote.modal_mirror) takes over as soon as the run writes its
run.json, exactly as for a run launched from the command line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_VOLUME = "wiki-llm-data"
PATH_FLAGS = ("tokens", "val-tokens", "text", "resume")
MODAL_TRAIN = Path("src/mini_llm/remote/modal_train.py")


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(repo: Path, run_id: str, status: dict) -> None:
    path = repo / "runs" / f"{run_id}.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)


def launching_status(run_id: str, job: dict) -> dict:
    return {
        "run_id": run_id,
        "kind": "train",
        "name": job["name"],
        "status": "running",
        "args": job["args"],
        "log": f"runs/{run_id}.log",  # launch output until the mirror replaces it with train.log
        "started": now_z(),
        "job_file": None,
        "remote": {"provider": "modal", "gpus": job["gpus"], "phase": "launching"},
    }


def _on_volume(volume, path: str, attempts: int = 4) -> bool:
    """Whether `path` exists on the volume. Only "not found" means missing: any other error
    (Modal rate-limiting several launches at once, a dropped connection) is retried and then
    raised, because treating it as missing re-uploads a file that is there, which fails."""
    for attempt in range(attempts):
        try:
            volume.listdir(path)
            return True
        except Exception as exc:  # noqa: BLE001 - classified below
            if type(exc).__name__ in ("NotFoundError", "FileNotFoundError"):
                return False
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    return False


def upload_missing_data(repo: Path, args: dict, volume=None) -> list[str]:
    """Put repo-relative data files on the data volume where modal_train
    looks for them (data/x/y.pt -> /x/y.pt, other/z.pt -> /other/z.pt)."""
    wanted = {}
    for flag in PATH_FLAGS:
        value = args.get(flag)
        if isinstance(value, str) and not value.startswith("/"):
            wanted["/" + value.removeprefix("data/")] = repo / value
    if not wanted:
        return []
    if volume is None:
        import modal

        volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    missing = [(remote, local) for remote, local in wanted.items() if not _on_volume(volume, remote)]
    if missing:
        with volume.batch_upload() as batch:
            for remote, local in missing:
                batch.put_file(str(local), remote)
    return [remote for remote, _ in missing]


def launch(job_file: Path, repo: Path, uv: str = "uv") -> int:
    job = json.loads(job_file.read_text())
    run_id = job["run_id"]
    status = launching_status(run_id, job)
    write_status(repo, run_id, status)
    log_path = repo / "runs" / f"{run_id}.log"
    config_path = job_file.with_name(f"{run_id}.modal-config.json")
    config_path.write_text(json.dumps({"name": job["name"], "args": job["args"]}, indent=2))

    with log_path.open("w") as log:
        try:
            uploaded = upload_missing_data(repo, job["args"])
            if uploaded:
                log.write(f"uploaded to {DATA_VOLUME}: {', '.join(uploaded)}\n")
                log.flush()
            proc = subprocess.run(
                [
                    uv,
                    "run",
                    "--project",
                    str(repo),
                    "--group",
                    "modal",
                    "modal",
                    "run",
                    "--detach",
                    str(repo / MODAL_TRAIN),
                    "--config",
                    str(config_path),
                    "--gpus",
                    job["gpus"],
                    "--timeout-hours",
                    str(job["timeout_hours"]),
                    "--run-id",
                    run_id,
                    "--no-wait",
                ],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = proc.returncode
        except Exception as exc:  # noqa: BLE001 - want the message on the page
            log.write(f"\nlaunch error: {type(exc).__name__}: {exc}\n")
            returncode = -1

    if returncode != 0:
        # The mirror only takes over once the run writes run.json on the volume,
        # which a failed launch never does -- so this is the run's final word.
        lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
        errors = [line for line in lines if "error" in line.lower()] or lines
        status.update(
            status="failed",
            finished=now_z(),
            returncode=returncode,
            error=f"Modal launch failed: {errors[-1] if errors else f'exit {returncode}'}"[:500],
        )
        status["remote"]["phase"] = "launch-failed"
        write_status(repo, run_id, status)
    return returncode


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Launch a web-submitted job on Modal (used by mini_llm.server).")
    p.add_argument("job_file", type=Path)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--uv", default="uv")
    args = p.parse_args(argv)
    raise SystemExit(launch(args.job_file, args.repo.expanduser().resolve(), args.uv))


if __name__ == "__main__":
    main()
