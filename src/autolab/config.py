"""Load autolab/config.toml and check the frozen val file."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "autolab" / "config.toml"


@dataclass(frozen=True)
class AutolabConfig:
    repo_root: Path
    frozen_val: Path
    frozen_val_sha256: str
    datasets_dir: Path
    runs_dir: Path
    owner_runs_dir: Path
    gpu_stale_after_s: float
    gpu_poll_s: float
    gpu_idle_checks_required: int


def load_config(path: Path = CONFIG_PATH, repo_root: Path = REPO_ROOT) -> AutolabConfig:
    raw = tomllib.loads(Path(path).read_text())
    return AutolabConfig(
        repo_root=repo_root,
        frozen_val=repo_root / raw["data"]["frozen_val"],
        frozen_val_sha256=raw["data"]["frozen_val_sha256"],
        datasets_dir=repo_root / raw["data"]["datasets_dir"],
        runs_dir=repo_root / raw["runs"]["runs_dir"],
        owner_runs_dir=Path(raw["gpu"]["owner_runs_dir"]).expanduser(),
        gpu_stale_after_s=float(raw["gpu"]["stale_after_s"]),
        gpu_poll_s=float(raw["gpu"]["poll_s"]),
        gpu_idle_checks_required=int(raw["gpu"]["idle_checks_required"]),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenValError(RuntimeError):
    pass


def check_frozen_val(cfg: AutolabConfig) -> Path:
    """Refuse to run if the frozen val file is missing or its bytes changed."""
    if not cfg.frozen_val.exists():
        raise FrozenValError(f"frozen val missing: {cfg.frozen_val}")
    got = sha256_file(cfg.frozen_val)
    if got != cfg.frozen_val_sha256:
        raise FrozenValError(
            f"frozen val sha256 mismatch for {cfg.frozen_val}: expected {cfg.frozen_val_sha256}, got {got}"
        )
    return cfg.frozen_val
