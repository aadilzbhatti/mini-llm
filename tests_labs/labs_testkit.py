"""Helpers shared by the labs tests (a unique module name: tests/autolab has its own conftest)."""

import json

from labs import core


def seed(lab, **scores):
    """p0 as the accepted incumbent."""
    p0 = core.Program("p0", None, lab.seed_file.read_text(), stage="done", status="accepted", scores=scores)
    lab.save(p0)
    lab.save_lab({"incumbent": "p0", "history": [], "p0_peak": 1000})
    return p0


def run_to_done(lab, p):
    for _ in range(8):
        if p.status in core.DONE:
            return p
        p = lab.advance(p, log=lambda m: None)
    return p


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
