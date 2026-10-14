
import pytest

from labs import core

CONFIG = """[labs]\ndaily_usd = 1.0\nest_call_usd = 0.3\n[llm]\nmodel = "opus"
[infer]\ncheckpoint = "c.pt"\nval = "v.pt"\nmax_delta_loss = 0.005\nmin_top1 = 0.97\nmax_mem_mult = 1.5\nbench_rounds = 4
min_gain = 0.02\naccept_se = 2.0\nmax_scenario_regression = 0.10
[decode]\ngenerator = "g.pt"\njudge = "j.pt"\nval = "v.pt"\nmin_tokens = 64\nscreen_n = 4\nfull_n = 4\nmin_gain = 0.02
accept_z = 2.0\nmax_judge_excess = 0.0\nmax_topic_drop = 0.03\nmax_sampler_ms = 2.0
[pack]\nn = 26\ndeadline_s = 5\nmin_gain = 1e-7\ncontender_gap = 0.01\nmax_programs = 3
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Labs state, data and config in tmp; never the live ones."""
    monkeypatch.setattr(core, "LABS_STATE", tmp_path / "labs")
    monkeypatch.setattr(core, "LABS_DATA", tmp_path / "data")
    monkeypatch.setattr(core, "OWNER_RUNS", tmp_path / "owner_runs")
    cfg = tmp_path / "labs.toml"
    cfg.write_text(CONFIG)
    monkeypatch.setattr(core, "CONFIG", cfg)
    return tmp_path
