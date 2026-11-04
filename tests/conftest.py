"""Shared test setup: engine step gating and the engine's golden-output fixtures.

Tests marked @pytest.mark.engine_step(n) are for roadmap step n of the serving engine. They run once
engine.PROGRESS >= n, and are skipped before that so the suite stays green while a step is in
progress; --engine-todo runs them anyway (that's how you work on a step).
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--engine-todo",
        action="store_true",
        help="Also run engine-step tests beyond engine.PROGRESS (the step you're working on).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "engine_step(n): a test for serving-engine roadmap step n")


def pytest_collection_modifyitems(config, items):
    import engine

    if config.getoption("--engine-todo"):
        return
    for item in items:
        marker = item.get_closest_marker("engine_step")
        if marker and marker.args[0] > engine.PROGRESS:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"engine step {marker.args[0]} not done (engine.PROGRESS = {engine.PROGRESS}); "
                    "run with --engine-todo to work on it"
                )
            )


# --- golden outputs (engine.golden) -------------------------------------------------


@pytest.fixture(scope="session")
def golden():
    from engine import golden

    return golden.load()


def _golden_models():
    from engine import golden

    data = golden.load()
    return [n for n, m in data["models"].items() if (golden.ROOT / m["checkpoint"]).exists()] or [
        pytest.param("none", marks=pytest.mark.skip(reason="no golden checkpoint on disk"))
    ]


def pytest_generate_tests(metafunc):
    if "golden_name" in metafunc.fixturenames:
        metafunc.parametrize("golden_name", _golden_models(), scope="module")


_ENGINE_MODELS: dict = {}


@pytest.fixture(scope="module")
def engine_model(golden, golden_name):
    """(model, config, golden record) for one golden checkpoint, loaded into engine/models/mini_llm.py.
    Per-head (model.py) checkpoints load too: the fused attention stacks their weights on load.
    (Step 4 replaces this hand-loading with the engine's own loader.)"""
    import torch

    from engine import golden as g
    from engine.models.mini_llm import ModelCustomTransformer
    from mini_llm.config import ModelConfig

    if golden_name not in _ENGINE_MODELS:
        rec = golden["models"][golden_name]
        ckpt = torch.load(g.ROOT / rec["checkpoint"], map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict({**ckpt["config"], "fused_attention": True})
        model = ModelCustomTransformer(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        _ENGINE_MODELS[golden_name] = (model.eval(), cfg, rec)
    return _ENGINE_MODELS[golden_name]
