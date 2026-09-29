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
def _isolated_research_cards(tmp_path, monkeypatch):
    """Tests never read or write the real technique-card store."""
    from autolab import research

    monkeypatch.setattr(research, "CARDS", tmp_path / "cards.jsonl")
    # research runs a real web-enabled Claude agent; off unless a test turns it on explicitly
    real = research.research_cfg
    monkeypatch.setattr(research, "research_cfg", lambda: {**real(), "enabled": False})


@pytest.fixture(autouse=True)
def _no_real_modal(tmp_path, monkeypatch):
    """No test touches real Modal or the real call list. Once, a test whose load_calls was faked (but not the
    file it saves to) overwrote autolab/state/modal_calls.json and submitted a real training run."""
    import modal

    from autolab import modal_backend as mb

    monkeypatch.setattr(mb, "_state_path", lambda: tmp_path / "modal_calls.json")

    def refuse(*a, **k):
        raise AssertionError("test tried to reach real Modal (Function.from_name / FunctionCall.from_id)")

    monkeypatch.setattr(modal.Function, "from_name", staticmethod(refuse))
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(refuse))


@pytest.fixture(autouse=True)
def _no_real_subprocess_spawns(monkeypatch):
    """controller._spawn starts detached real work (research runs, dataset builds, uploads). A test that
    reaches it without replacing it would launch that for real (once: 7 real research runs, ~$4)."""
    from autolab import controller

    def refuse(args, logname):
        raise AssertionError(f"test tried to spawn a real subprocess: autolab {' '.join(args)}")

    monkeypatch.setattr(controller, "_spawn", refuse)

    def refuse_generate(log, card=None, retest=None):
        raise AssertionError("test reached controller._generate (a real Claude call); pass generate= or patch it")

    # directed children (research cards, near-miss retests) go through _generate even when a test passes generate=
    monkeypatch.setattr(controller, "_generate", refuse_generate)


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
