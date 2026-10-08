"""model_fused.py (fused q/k/v + scaled_dot_product_attention) computes what model.py computes.

Every check loads model.py's weights into the fused model (its per-head query/key/value weights
stack into qkv.weight) and compares: logits, gradients, cached decoding past the window, generation.
"""

import math

import pytest
import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.model import ModelCustomTransformer as PerHead
from mini_llm.model_fused import ModelCustomTransformer as Fused

VOCAB, BLOCK = 97, 32


def pair(rope: bool, n_layer: int = 2, dropout: float = 0.0):
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=VOCAB,
        block_size=BLOCK,
        n_embd=48,
        n_head=4,
        n_layer=n_layer,
        dropout=dropout,
        use_rope_embeddings=rope,
    )
    ref = build_model(cfg)
    fused = build_model(ModelConfig(**{**cfg.to_dict(), "fused_attention": True}))
    fused.load_state_dict(ref.state_dict())  # per-head checkpoint -> fused
    return ref.eval(), fused.eval()


@pytest.mark.parametrize("rope", [True, False], ids=["rope", "absolute"])
def test_build_model_picks_the_fused_module_and_loads_per_head_weights(rope):
    ref, fused = pair(rope)
    assert isinstance(ref, PerHead) and isinstance(fused, Fused)
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in fused.parameters())
    sa = ref.blocks[0].sa
    stacked = torch.cat([getattr(h, n).weight for n in ("query", "key", "value") for h in sa.heads])
    assert torch.equal(fused.blocks[0].sa.qkv.weight, stacked)


@pytest.mark.parametrize("rope", [True, False], ids=["rope", "absolute"])
def test_same_logits_and_loss(rope):
    ref, fused = pair(rope)
    x = torch.randint(0, VOCAB, (3, BLOCK))
    y = torch.randint(0, VOCAB, (3, BLOCK))
    with torch.no_grad():
        (a, la), (b, lb) = ref(x, y), fused(x, y)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(la, lb, atol=1e-6)


def test_same_gradients():
    """Training equivalence, not just inference: d loss / d weights match (qkv vs the stacked per-head grads)."""
    ref, fused = pair(rope=True)
    ref.train(), fused.train()
    x, y = torch.randint(0, VOCAB, (2, BLOCK)), torch.randint(0, VOCAB, (2, BLOCK))
    ref(x, y)[1].backward()
    fused(x, y)[1].backward()
    for rb, fb in zip(ref.blocks, fused.blocks):
        stacked = torch.cat([getattr(h, n).weight.grad for n in ("query", "key", "value") for h in rb.sa.heads])
        assert torch.allclose(fb.sa.qkv.weight.grad, stacked, atol=1e-6)
        assert torch.allclose(fb.sa.proj.weight.grad, rb.sa.proj.weight.grad, atol=1e-6)
    assert torch.allclose(fused.token_embedding_table.weight.grad, ref.token_embedding_table.weight.grad, atol=1e-6)


@pytest.mark.parametrize("rope", [True, False], ids=["rope", "absolute"])
@pytest.mark.parametrize("prefill", [1, 7])
def test_cached_steps_match_full_forward(rope, prefill):
    _, fused = pair(rope)
    T = 20
    idx = torch.randint(0, VOCAB, (2, T))
    with torch.no_grad():
        full, _ = fused(idx)
        fused.clear_cache()
        steps = [fused(idx[:, :prefill], use_cache=True)[0]]
        steps += [fused(idx[:, t : t + 1], use_cache=True)[0] for t in range(prefill, T)]
    assert fused.cache_len() == T
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_prefill_on_top_of_a_cache_uses_the_offset_mask():
    """A multi-token write after the cache already holds tokens: the 1 < T < T_total mask path."""
    _, fused = pair(rope=True)
    idx = torch.randint(0, VOCAB, (1, 12))
    with torch.no_grad():
        full, _ = fused(idx)
        fused.clear_cache()
        fused(idx[:, :5], use_cache=True)
        second, _ = fused(idx[:, 5:12], use_cache=True)
    assert torch.allclose(full[:, 5:12], second, atol=1e-4)


@pytest.mark.parametrize("rope", [True, False], ids=["rope", "absolute"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_generation_matches_per_head_past_the_window(rope, greedy):
    """Through the rolling cache (RoPE) or the window refill (absolute) at 2x block_size: token for token."""
    ref, fused = pair(rope)
    prompt = torch.randint(0, VOCAB, (1, 6))
    torch.manual_seed(1)
    a = ref.generate(prompt, 2 * BLOCK, BLOCK, greedy=greedy)
    torch.manual_seed(1)
    b = fused.generate(prompt, 2 * BLOCK, BLOCK, greedy=greedy)
    assert torch.equal(a, b)


def test_checkpoint_round_trip_rebuilds_the_fused_model():
    _, fused = pair(rope=True)
    cfg = ModelConfig.from_dict({"vocab_size": VOCAB, **{k: v for k, v in fused_cfg_dict(fused).items()}})
    again = build_model(cfg)
    again.load_state_dict(fused.state_dict())
    assert isinstance(again, Fused) and cfg.fused_attention
    old = ModelConfig.from_dict({"vocab_size": VOCAB, "block_size": BLOCK, "n_embd": 48, "n_head": 4, "n_layer": 2})
    assert not old.fused_attention and isinstance(build_model(old), PerHead)  # old checkpoints: per-head


def fused_cfg_dict(model) -> dict:
    sa = model.blocks[0].sa
    return {
        "block_size": sa.block_size,
        "n_embd": sa.proj.out_features,
        "n_head": sa.n_head,
        "n_layer": len(model.blocks),
        "use_rope_embeddings": sa.use_rope_embeddings,
        "fused_attention": True,
    }


def test_init_has_per_head_xavier_normal_scale():
    """Fresh fused weights: each head's q/k/v slice is xavier_normal over (head_size, n_embd), as model.py."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK, n_embd=768, n_head=12, n_layer=1, fused_attention=True)
    w = build_model(cfg).blocks[0].sa.qkv.weight.detach()
    expected = math.sqrt(2.0 / (64 + 768))
    assert abs(w.std().item() - expected) / expected < 0.02


def test_dropout_trains():
    _, fused = pair(rope=True, dropout=0.1)
    fused.train()
    x = torch.randint(0, VOCAB, (2, BLOCK))
    _, loss = fused(x, x)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("rope", [True, False], ids=["rope", "absolute"])
def test_attention_is_causal(rope):
    """Changing token t must not change any output before t (no peeking at future tokens), in training mode
    (the path training uses) as well as eval."""
    _, fused = pair(rope)
    for mode in (fused.train, fused.eval):
        mode()
        x = torch.randint(0, VOCAB, (1, BLOCK))
        y = x.clone()
        t = BLOCK // 2
        y[0, t] = (x[0, t] + 1) % VOCAB
        with torch.no_grad():
            a, _ = fused(x)
            b, _ = fused(y)
        assert torch.equal(a[:, :t], b[:, :t]), "an output before the changed token moved: attention sees the future"
        assert not torch.allclose(a[:, t:], b[:, t:])  # and the change does reach t and later
