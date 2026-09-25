"""Upsert-and-sort behavior for baselines.md."""

import json

from mini_llm.baselines import update_baselines


def test_insert_creates_table_with_one_row(tmp_path):
    path = update_baselines({"run": "a", "eval_val_loss": 3.0}, md_path=tmp_path / "baselines.md")
    text = path.read_text()
    assert "| a |" in text or "| a " in text
    rows = json.loads((tmp_path / "baselines.json").read_text())
    assert [r["run"] for r in rows] == ["a"]


def test_second_insert_appends_and_sorts_by_best_loss(tmp_path):
    md_path = tmp_path / "baselines.md"
    update_baselines({"run": "worse", "eval_val_loss": 5.0}, md_path=md_path)
    update_baselines({"run": "better", "eval_val_loss": 2.0}, md_path=md_path)

    rows = json.loads((tmp_path / "baselines.json").read_text())
    assert [r["run"] for r in rows] == ["better", "worse"]


def test_same_run_name_replaces_in_place_not_duplicated(tmp_path):
    md_path = tmp_path / "baselines.md"
    update_baselines({"run": "a", "eval_val_loss": 5.0}, md_path=md_path)
    update_baselines({"run": "a", "eval_val_loss": 1.0}, md_path=md_path)

    rows = json.loads((tmp_path / "baselines.json").read_text())
    assert len(rows) == 1
    assert rows[0]["eval_val_loss"] == 1.0


def test_full_val_loss_preferred_over_eval_val_loss_when_any_row_has_it(tmp_path):
    md_path = tmp_path / "baselines.md"
    # "a" only has eval_val_loss (worse-looking number), "b" has full_val_loss
    # (better-looking number). Once ANY row has full_val_loss, the whole
    # table must sort on that column, not let "a" win on a cheaper metric.
    update_baselines({"run": "a", "eval_val_loss": 1.0}, md_path=md_path)
    update_baselines({"run": "b", "eval_val_loss": 5.0, "full_val_loss": 2.0}, md_path=md_path)

    rows = json.loads((tmp_path / "baselines.json").read_text())
    # "a" has no full_val_loss, so it sorts after "b" despite the smaller eval_val_loss.
    assert [r["run"] for r in rows] == ["b", "a"]


def test_missing_loss_values_sort_last(tmp_path):
    md_path = tmp_path / "baselines.md"
    update_baselines({"run": "no_loss"}, md_path=md_path)
    update_baselines({"run": "has_loss", "eval_val_loss": 3.0}, md_path=md_path)

    rows = json.loads((tmp_path / "baselines.json").read_text())
    assert [r["run"] for r in rows] == ["has_loss", "no_loss"]
