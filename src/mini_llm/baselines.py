"""Maintain baselines.md: one row per run, upserted and kept sorted.

baselines.json next to it is the source of truth -- a plain list of row
dicts. baselines.md is fully regenerated from it on every update, so this
never parses its own previous markdown output back in (fragile) the way an
in-place table edit would.

A "run" is identified by its `run` name (the checkpoint/plot name a train.py
call resolved to). Calling update_baselines() again with the same name
replaces that row in place rather than duplicating it -- e.g. a --resume
continuation naturally supersedes its own earlier row.

Rows are sorted by the best loss available across the whole table, cheapest
metric first: full_val_loss (exhaustive, most trustworthy) if ANY row has
it, else eval_val_loss, else eval_train_loss. Mixing rows that only have
eval_train_loss with rows that have a real full_val_loss and comparing them
on the same column would rank an unvalidated run against a validated one,
so the column choice is table-wide, not decided per row.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MD_PATH = Path("baselines.md")

COLUMNS = [
    "run",
    "date",
    "steps",
    "params",
    "n_layer",
    "n_embd",
    "n_head",
    "block_size",
    "batch_size",
    "lr",
    "min_lr",
    "warmup_steps",
    "seed",
    "eval_train_loss",
    "eval_val_loss",
    "full_val_loss",
    "checkpoint",
    "plot",
]

_LOSS_COLUMNS_BY_PREFERENCE = ("full_val_loss", "eval_val_loss", "eval_train_loss")


def _json_path(md_path: Path) -> Path:
    return md_path.with_suffix(".json")


def _load_rows(md_path: Path) -> list[dict[str, object]]:
    json_path = _json_path(md_path)
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _table_sort_column(rows: list[dict[str, object]]) -> str | None:
    """The best loss column to sort by, given what the whole table has.

    Picks the most trustworthy metric that ANY row actually has, so every
    row is ranked on the same column rather than each on its own best
    available one.
    """
    for col in _LOSS_COLUMNS_BY_PREFERENCE:
        if any(row.get(col) is not None for row in rows):
            return col
    return None


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sort_col = _table_sort_column(rows)
    if sort_col is None:
        return rows

    def key(row: dict[str, object]) -> tuple[int, float]:
        value = row.get(sort_col)
        return (0, float(value)) if value is not None else (1, float("inf"))

    return sorted(rows, key=key)


def _fmt_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_md(rows: list[dict[str, object]], sort_col: str | None) -> str:
    if sort_col:
        description = (
            f"One row per run, upserted by name. Sorted by `{sort_col}` ascending "
            "(full_val_loss > eval_val_loss > eval_train_loss, whichever the table has)."
        )
    else:
        description = "One row per run, upserted by name. No loss columns populated yet, so unsorted."
    lines = [
        "# Baselines",
        "",
        description,
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def update_baselines(row: dict[str, object], md_path: Path | str = DEFAULT_MD_PATH) -> Path:
    """Insert or replace `row` (keyed by row["run"]) and rewrite the table, sorted.

    Returns the path to the written .md file.
    """
    md_path = Path(md_path)
    rows = [r for r in _load_rows(md_path) if r.get("run") != row.get("run")]
    rows.append(row)
    rows = _sort_rows(rows)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    _json_path(md_path).write_text(json.dumps(rows, indent=2) + "\n")
    md_path.write_text(_render_md(rows, _table_sort_column(rows)))
    return md_path
