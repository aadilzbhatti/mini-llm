import json

import pytest

from autolab.config import REPO_ROOT

STATE = REPO_ROOT / "autolab" / "state"


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Autolab tests never touch the shared GPU (see HANDOFF, shared-GPU rule)."""
    monkeypatch.setenv("AUTOLAB_FORCE_CPU", "1")


def _git(*args):
    import subprocess

    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout


def _live_switches():
    try:
        active = (STATE / "evolve" / "ACTIVE").read_text()
    except OSError:
        active = None
    try:
        enabled = json.loads((STATE / "controller.json").read_text()).get("enabled")
    except (OSError, json.JSONDecodeError):
        enabled = None
    return active, enabled, _git("branch", "--list"), _git("worktree", "list", "--porcelain")


@pytest.fixture(autouse=True)
def _no_real_commits(request, monkeypatch):
    """controller.commit_accepted writes a git branch/worktree. Tests get a recorder unless they
    are marked real_commit (and then must pass an explicit throwaway repo)."""
    if request.node.get_closest_marker("real_commit"):
        return
    from autolab import controller

    monkeypatch.setattr(controller, "commit_accepted", lambda p, s, log=print, **kw: "stub")


@pytest.fixture(autouse=True)
def _live_state_untouched():
    """Tests must never flip the live daemon's switches (active session, controller on/off) or
    touch the real repo's branches/worktrees (commit_accepted once did).

    Twice a default argument bound at import time sent a test's writes to the real state;
    this turns any repeat into a test failure instead of a surprise for the daemon.
    """
    before = _live_switches()
    yield
    assert _live_switches() == before, f"test changed live autolab state: {before} -> {_live_switches()}"


def pytest_configure(config):
    config.addinivalue_line("markers", "real_commit: uses controller.commit_accepted on a throwaway repo")
