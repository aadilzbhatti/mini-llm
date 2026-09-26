import pytest


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Autolab tests never touch the shared GPU (see HANDOFF, shared-GPU rule)."""
    monkeypatch.setenv("AUTOLAB_FORCE_CPU", "1")
