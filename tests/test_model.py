"""Smoke tests for the bootstrap, plus placeholders for you to fill in.

The skipped tests below are the four things you said you want to check
yourself. They are intentionally empty: the point of this project is that
you write them.
"""

import pytest
import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import fixed_batch, make_batch

VOCAB_SIZE = 64
BLOCK_SIZE = 8


@pytest.fixture
def cfg() -> ModelConfig:
    """Very small model so tests run in milliseconds on CPU."""
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    return build_model(cfg)


@pytest.fixture
def tokens() -> torch.Tensor:
    """A deterministic 1-D stream of ids, standing in for tokenized text."""
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (256,), generator=g)


# --- smoke: proves the bootstrap itself works -------------------------------


def test_imports():
    import mini_llm  # noqa: F401
    import mini_llm.data  # noqa: F401
    import mini_llm.device  # noqa: F401
    import mini_llm.generate  # noqa: F401
    import mini_llm.train  # noqa: F401


def test_forward_and_backward_run(model, tokens):
    x, y = make_batch(tokens, batch_size=2, block_size=BLOCK_SIZE)
    logits, loss = model(x, y)
    assert logits.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_generate_runs(model):
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3, block_size=BLOCK_SIZE)
    assert out.shape == (1, 4)


def test_fixed_batch_is_reproducible(tokens):
    a_x, a_y = fixed_batch(tokens, batch_size=2, block_size=BLOCK_SIZE, seed=0)
    b_x, b_y = fixed_batch(tokens, batch_size=2, block_size=BLOCK_SIZE, seed=0)
    assert torch.equal(a_x, b_x) and torch.equal(a_y, b_y)


# --- yours to write ---------------------------------------------------------


@pytest.mark.skip(reason="TODO: assert logits are (B, T, vocab_size) and loss is a scalar")
def test_output_shapes():
    ...


@pytest.mark.skip(reason="TODO: assert y is x shifted by one position")
def test_target_shifting():
    ...


@pytest.mark.skip(reason="TODO: assert position t's logits don't change when tokens > t change")
def test_causal_isolation():
    ...


@pytest.mark.skip(reason="TODO: assert loss on one fixed batch drops toward zero")
def test_overfits_one_batch():
    ...
