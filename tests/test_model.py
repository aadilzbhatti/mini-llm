"""Smoke tests for the bootstrap, plus placeholders for you to fill in.

The skipped tests below are the four things you said you want to check
yourself. They are intentionally empty: the point of this project is that
you write them.
"""

import pytest
import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.model import ModelCustomTransformer
from mini_llm.data import fixed_batch, make_batch, get_tokenizer, encode

VOCAB_SIZE = 64
BLOCK_SIZE = 8
LARGE_VOCAB_SIZE = len(get_tokenizer())


@pytest.fixture
def cfg() -> ModelConfig:
    tokenizer = get_tokenizer()
    """Very small model so tests run in milliseconds on CPU."""
    return ModelConfig(
        vocab_size=len(tokenizer),
        block_size=BLOCK_SIZE,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg: ModelConfig) -> ModelCustomTransformer:
    return build_model(cfg)


@pytest.fixture
def tokens() -> torch.Tensor:
    """A deterministic 1-D stream of ids, standing in for tokenized text."""
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (256,), generator=g)


def test_forward_and_backward_run(model: ModelCustomTransformer, tokens: torch.Tensor):
    x, y = make_batch(tokens, batch_size=2, block_size=BLOCK_SIZE)
    logits, loss = model(x, y)
    assert logits.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_generate_runs(model: ModelCustomTransformer):
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3, block_size=BLOCK_SIZE)
    assert out.shape == (1, 4)


def test_fixed_batch_is_reproducible(tokens: torch.Tensor):
    a_x, a_y = fixed_batch(tokens, batch_size=2, block_size=BLOCK_SIZE, seed=0)
    b_x, b_y = fixed_batch(tokens, batch_size=2, block_size=BLOCK_SIZE, seed=0)
    assert torch.equal(a_x, b_x) and torch.equal(a_y, b_y)


def test_output_shapes(model: ModelCustomTransformer):
    """Check that the model output shapes are as expected."""
    tokens = torch.randint(0, LARGE_VOCAB_SIZE, (256,))
    x, y = make_batch(tokens, batch_size=4, block_size=BLOCK_SIZE)
    logits, loss = model(x, y)
    assert logits.shape == (32, LARGE_VOCAB_SIZE), f"Expected logits shape {(4, BLOCK_SIZE, LARGE_VOCAB_SIZE)}, but got {logits.shape}"
    assert loss.shape == (), f"Expected loss to be a scalar tensor, but got shape {loss.shape}"


def test_target_shifting():
    """Check that the targets are correctly shifted by one position."""
    tokens = torch.randint(0, VOCAB_SIZE, (256,))
    x, y = make_batch(tokens, batch_size=4, block_size=BLOCK_SIZE)
    # The target y should be the input x shifted by one position
    assert torch.equal(y[:, :-1], x[:, 1:]), "Targets are not correctly shifted by one position."


def test_causal_isolation(model: ModelCustomTransformer):
    seq1 = "The lighthouse keeper watched the ships"
    seq2 = "The lighthouse keeper watched elephants dance"
    tokenizer = get_tokenizer()
    tokens_1 = encode(seq1, tokenizer)
    tokens_2 = encode(seq2, tokenizer)
    x1 = tokens_1[:BLOCK_SIZE].unsqueeze(0)
    x2 = tokens_2[:BLOCK_SIZE].unsqueeze(0)

    model.eval()
    logits_a, _ = model(x1)
    logits_b, _ = model(x2)
    # t is the position that the sequences match through
    diff_positions = (x1 != x2).nonzero()
    t = diff_positions[0, 1].item()
    assert torch.allclose(logits_a[:, :t, :], logits_b[:, :t, :], atol=1e-5, rtol=1e-5), f"Logits differ before position {t}"


def test_overfits_one_batch(model: ModelCustomTransformer):
    """Check that the model can overfit a single batch of data."""
    tokens = torch.randint(0, VOCAB_SIZE, (256,))
    x, y = fixed_batch(tokens, batch_size=4, block_size=BLOCK_SIZE, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(400):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    # After training, check that the loss is very low
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    assert loss.item() < 1e-2, f"Loss did not decrease enough: {loss.item()}"
