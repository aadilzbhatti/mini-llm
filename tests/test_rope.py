"""RoPE tests: Head.rotate against its mathematical definition, plus cache equivalence."""

import pytest
import torch
import torch.nn.functional as F

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import get_tokenizer
from mini_llm.model import Head, ModelCustomTransformer

BLOCK_SIZE = 32
N_EMBD = 32
N_HEAD = 2
HEAD_SIZE = N_EMBD // N_HEAD


@pytest.fixture
def model() -> ModelCustomTransformer:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=len(get_tokenizer()),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=2,
        dropout=0.0,
        use_rope_embeddings=True,
    )
    return build_model(cfg).eval()


@pytest.fixture
def head(model: ModelCustomTransformer) -> Head:
    return model.blocks[0].sa.heads[0]


def rotate_at(head: Head, x: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotate a (head_size,) vector as if it sat at position `pos`."""
    return head.rotate(x.view(1, 1, -1), pos).view(-1)


def rotation_matrix(head: Head, pos: int) -> torch.Tensor:
    """The full (d, d) block-diagonal rotation: one 2x2 block per (even, odd) pair."""
    d = head.head_size
    R = torch.zeros(d, d)
    for i in range(d // 2):
        angle = torch.tensor(pos * 10000.0 ** (-2 * i / d))  # straight from the definition, not the head's speeds
        c, s = torch.cos(angle), torch.sin(angle)
        R[2 * i, 2 * i], R[2 * i, 2 * i + 1] = c, -s
        R[2 * i + 1, 2 * i], R[2 * i + 1, 2 * i + 1] = s, c
    return R


def test_speeds_match_formula(head: Head):
    """The per-pair frequencies are 10000^(-2i/d); every angle p * speed is built from these."""
    d = HEAD_SIZE
    expected = torch.tensor([10000.0 ** (-2 * i / d) for i in range(d // 2)])
    assert torch.allclose(head.speeds, expected, rtol=1e-5)


def test_rope_is_enabled(model: ModelCustomTransformer):
    for block in model.blocks:
        for h in block.sa.heads:
            assert h.use_rope_embeddings
    assert not hasattr(model, "position_embedding_table")


def test_forward_rotates_q_and_k(head: Head):
    """Training mode, no dropout: the head's logged attention must equal the hand-computed
    softmax(mask(rot(q) rot(k)^T * scale)). Rotating only k (or only q) gives different weights."""
    head.train()
    T = 6
    x = torch.randn(1, T, N_EMBD)
    head(x)
    q, k = head.query(x), head.key(x)
    q, k = head.rotate(q, 0), head.rotate(k, 0)
    wei = (q @ k.transpose(-2, -1)) * head.scale
    wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    expected = F.softmax(wei, dim=-1)
    assert torch.allclose(head.attention_values, expected, atol=1e-5)

    # sanity: the check is sensitive to the bug it targets
    k_only = (head.query(x) @ k.transpose(-2, -1)) * head.scale
    k_only = F.softmax(k_only.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf")), dim=-1)
    assert not torch.allclose(head.attention_values, k_only, atol=1e-5)


def test_rotation_preserves_norm(head: Head):
    x = torch.randn(4, BLOCK_SIZE, HEAD_SIZE)
    rotated = head.rotate(x, 0)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_position_zero_is_identity(head: Head):
    x = torch.randn(3, 1, HEAD_SIZE)
    assert torch.allclose(head.rotate(x, 0), x)


def test_relative_invariance(head: Head):
    q, k = torch.randn(HEAD_SIZE), torch.randn(HEAD_SIZE)
    m, n = 7, 3
    base = rotate_at(head, q, m) @ rotate_at(head, k, n)
    for shift in (1, 5, 20):
        shifted = rotate_at(head, q, m + shift) @ rotate_at(head, k, n + shift)
        assert torch.allclose(base, shifted, atol=1e-4), f"shift {shift}"


def test_matches_slow_reference(head: Head):
    x = torch.randn(HEAD_SIZE)
    for pos in (0, 1, 9, BLOCK_SIZE - 1):
        slow = rotation_matrix(head, pos) @ x
        assert torch.allclose(rotate_at(head, x, pos), slow, atol=1e-5), f"pos {pos}"


def test_batched_rotate_matches_per_position(head: Head):
    """Rotating a (B, T, d) block at `start` rotates row t by position start + t."""
    x = torch.randn(2, 5, HEAD_SIZE)
    start = 4
    out = head.rotate(x, start)
    for t in range(5):
        for b in range(2):
            assert torch.allclose(out[b, t], rotation_matrix(head, start + t) @ x[b, t], atol=1e-5)


@pytest.mark.parametrize("prefill", [1, 5])
def test_cache_matches_full_forward(model: ModelCustomTransformer, prefill: int):
    """Last-token logits: one full pass vs. a prefill followed by one-token cached steps.
    Catches position-offset bugs, since a wrong `start` rotates new tokens to the wrong angle."""
    T = 12
    idx = torch.randint(0, model.token_embedding_table.num_embeddings, (1, T))

    with torch.no_grad():
        full, _ = model(idx)

        steps = [model(idx[:, :prefill], use_cache=True)[0]]
        for t in range(prefill, T):
            steps.append(model(idx[:, t:t + 1], use_cache=True)[0])
        assert model.cache_len() == T

    cached = torch.cat(steps, dim=1)  # (1, T, vocab): prefill logits, then one row per step
    for t in range(T):
        assert torch.allclose(full[:, t], cached[:, t], atol=1e-4), f"diverged at position {t}"


@pytest.fixture
def one_layer_model() -> ModelCustomTransformer:
    """A rolling cache only reproduces a windowed recompute exactly with a single layer. Past the first
    layer, a cached token's hidden state was computed while it could still see tokens that have since
    left the window, so a deeper model drifts slightly from the recompute by design."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=len(get_tokenizer()),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=1,
        dropout=0.0,
        use_rope_embeddings=True,
    )
    return build_model(cfg).eval()


def test_wrapped_cache_matches_windowed_full_forward(one_layer_model: ModelCustomTransformer):
    """One-token cached steps well past block_size, so the ring buffer wraps. Keys sit in the cache
    rotated at their absolute positions; the reference re-rotates the last block_size tokens from 0.
    They agree only because RoPE depends on relative offsets, so this catches a wrong rotation start
    or a wrapped slot that attends to the wrong keys."""
    model = one_layer_model
    T = BLOCK_SIZE + 12
    idx = torch.randint(0, model.token_embedding_table.num_embeddings, (1, T))
    prefill = 5

    with torch.no_grad():
        model.clear_cache()
        model(idx[:, :prefill], use_cache=True)
        for t in range(prefill, T):
            step, _ = model(idx[:, t : t + 1], use_cache=True)
            window = idx[:, max(0, t + 1 - BLOCK_SIZE) : t + 1]
            full, _ = model(window)
            assert torch.allclose(step[:, -1], full[:, -1], atol=1e-4), f"diverged at position {t}"
    assert model.cache_len() == T  # pos keeps growing past block_size
    model.clear_cache()


def test_generate_through_window_overflow_matches_uncached(one_layer_model: ModelCustomTransformer):
    """With RoPE, generate keeps decoding through the wrapped cache once the window is full. Greedy
    output past block_size must match re-running the cropped window every step with no cache."""
    model = one_layer_model
    prompt = torch.randint(0, model.token_embedding_table.num_embeddings, (1, 6))
    new_tokens = 2 * BLOCK_SIZE

    expected = prompt
    with torch.no_grad():
        for _ in range(new_tokens):
            logits, _ = model(expected[:, -BLOCK_SIZE:], last_only=True)
            expected = torch.cat((expected, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        got = model.generate(prompt, max_new_tokens=new_tokens, block_size=BLOCK_SIZE, greedy=True)

    assert torch.equal(got, expected)


def test_deep_model_wrapped_cache_drift_is_bounded(model: ModelCustomTransformer):
    """With more than one layer the rolling cache is not exact (see one_layer_model): cached hidden
    states were computed with tokens that have since left the window. This only guards against the
    drift becoming gross (NaNs, a blown-up cache), not against small bugs. A wrong rotation start
    gives about the same KL as the legitimate drift here, so the one-layer tests are what catch those."""
    T = BLOCK_SIZE + 40
    idx = torch.randint(0, model.token_embedding_table.num_embeddings, (1, T))

    worst = 0.0
    with torch.no_grad():
        model.clear_cache()
        model(idx[:, :5], use_cache=True)
        for t in range(5, T):
            step, _ = model(idx[:, t : t + 1], use_cache=True)
            if t < BLOCK_SIZE:
                continue  # the window has not dropped anything yet
            full, _ = model(idx[:, t + 1 - BLOCK_SIZE : t + 1])
            kl = F.kl_div(
                F.log_softmax(step[:, -1], dim=-1), F.log_softmax(full[:, -1], dim=-1), log_target=True, reduction="batchmean"
            )
            assert torch.isfinite(kl), f"non-finite drift at position {t}"
            worst = max(worst, kl.item())
    model.clear_cache()
    assert worst < 0.1, f"wrapped-cache drift too large: KL {worst:.3f}"  # random 2-layer weights measure ~0.02


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_generate_until_eos_matches_generate_past_window(model: ModelCustomTransformer, greedy: bool):
    """Both generation paths share one cache/window policy, so on a deep RoPE model (where the rolling
    cache is approximate) they must still agree token for token past block_size."""
    from mini_llm.report import generate_until_eos

    prompt = torch.randint(0, model.token_embedding_table.num_embeddings, (1, 6))
    new_tokens = 2 * BLOCK_SIZE
    torch.manual_seed(1)
    expected = model.generate(prompt, max_new_tokens=new_tokens, block_size=BLOCK_SIZE, greedy=greedy)
    torch.manual_seed(1)
    got, hit_eos = generate_until_eos(
        model, prompt, new_tokens, BLOCK_SIZE, greedy=greedy, eos_token_id=None
    )
    assert not hit_eos
    assert torch.equal(got, expected)
