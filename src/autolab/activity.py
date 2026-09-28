"""What autolab is doing right now, for the dashboard's Live tab.

`set_activity(kind, text)` records the current step (autolab/state/activity.json) and appends it to a
short history. The daemon, the cascade, the proposer and the controller call it as they work, so the
page can answer "what is autolab doing this minute?" without reading logs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from autolab.config import REPO_ROOT

PATH = REPO_ROOT / "autolab" / "state" / "activity.json"
HISTORY = 60


def set_activity(kind: str, text: str, path: Path | None = None, **extra) -> None:
    if os.environ.get("AUTOLAB_IN_CASCADE") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        return  # tests and candidate test runs never touch the live page
    path = path or PATH
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        state = {"history": []}
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "text": text, **extra}
    state["current"] = entry
    state["history"] = ([entry] + state.get("history", []))[:HISTORY]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)
