"""Engine step 1: a stateless model and an engine-owned KV cache.

The model (engine/models/mini_llm.py) becomes forward(input_ids, positions, kv_cache) -> logits: it
reads and writes the cache it's handed, takes positions as an input instead of counting them, and keeps
nothing between calls. Checked against the golden outputs recorded from mini_llm (engine.golden),
including decoding past the 1024-token window through a cache that wraps.

    uv run pytest tests/test_engine_step1.py --engine-todo
"""

import pytest
import torch

from engine import KVCache
from engine import golden as g

pytestmark = pytest.mark.engine_step(1)


def new_cache(cfg, batch_size=1):
    return KVCache(
        num_layers=cfg.n_layer,
        batch_size=batch_size,
        num_kv_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        max_len=cfg.block_size,
    )


def forward(model, ids, positions, cache=None):
    out = model(ids, positions=positions, kv_cache=cache)
    return out[0] if isinstance(out, tuple) else out  # logits (B, T, vocab)


@torch.no_grad()
def greedy(model, cfg, prompt_ids, n, cache=None):
    """Prefill the prompt, then one token per step at the next position: the decode loop step 2 will own."""
    cache = cache or new_cache(cfg)
    ids = torch.tensor([prompt_ids])
    logits = forward(model, ids, torch.arange(len(prompt_ids)).unsqueeze(0), cache)
    out = []
    for i in range(n):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        pos = torch.tensor([[len(prompt_ids) + i]])
        logits = forward(model, torch.tensor([[nxt]]), pos, cache)
    return out


@torch.no_grad()
def test_uncached_forward_matches_the_golden_logits(engine_model, golden):
    model, cfg, rec = engine_model
    T = cfg.block_size
    logits = forward(model, torch.load(g.VAL)[:T].unsqueeze(0), torch.arange(T).unsqueeze(0))
    for pos, row in rec["logits"]["positions"].items():
        v, i = torch.topk(logits[0, int(pos)], len(row["top_ids"]))
        assert i.tolist() == row["top_ids"], f"position {pos}"
        assert torch.allclose(v, torch.tensor(row["top_logits"]), atol=golden["logit_atol"]), f"position {pos}"


@torch.no_grad()
def test_prefill_then_decode_through_the_cache_matches_a_full_forward(engine_model):
    model, cfg, _ = engine_model
    ids = torch.load(g.VAL)[:200].unsqueeze(0)
    full = forward(model, ids, torch.arange(200).unsqueeze(0))
    cache = new_cache(cfg)
    rows = [forward(model, ids[:, :150], torch.arange(150).unsqueeze(0), cache)]
    rows += [forward(model, ids[:, t : t + 1], torch.tensor([[t]]), cache) for t in range(150, 200)]
    assert torch.allclose(torch.cat(rows, dim=1), full, atol=1e-4)


def test_greedy_decoding_reproduces_the_golden_tokens(engine_model):
    model, cfg, rec = engine_model
    for case in rec["greedy"]:
        assert greedy(model, cfg, case["prompt_ids"], len(case["output_ids"])) == case["output_ids"], case["prompt"]


def test_decoding_past_the_window_rolls_the_cache(engine_model):
    """1000-token prompt + 80 tokens: positions run past the 1024-slot cache, which must wrap (RoPE)."""
    model, cfg, rec = engine_model
    case = rec["past_window"]
    assert greedy(model, cfg, case["prompt_ids"], len(case["output_ids"])) == case["output_ids"]


def test_the_model_keeps_no_state_between_calls(engine_model):
    """Two caches, two prompts, decoded in alternation through ONE model: each must come out as if alone."""
    model, cfg, rec = engine_model
    a, b = rec["greedy"][0], rec["greedy"][1]
    ca, cb = new_cache(cfg), new_cache(cfg)
    got_a, got_b = [], []
    with torch.no_grad():
        la = forward(model, torch.tensor([a["prompt_ids"]]), torch.arange(len(a["prompt_ids"])).unsqueeze(0), ca)
        lb = forward(model, torch.tensor([b["prompt_ids"]]), torch.arange(len(b["prompt_ids"])).unsqueeze(0), cb)
        for i in range(12):
            got_a.append(int(la[0, -1].argmax()))
            got_b.append(int(lb[0, -1].argmax()))
            la = forward(model, torch.tensor([[got_a[-1]]]), torch.tensor([[len(a["prompt_ids"]) + i]]), ca)
            lb = forward(model, torch.tensor([[got_b[-1]]]), torch.tensor([[len(b["prompt_ids"]) + i]]), cb)
    assert got_a == a["output_ids"][:12] and got_b == b["output_ids"][:12]
    for name, m in model.named_modules():
        for attr in ("k_cache", "v_cache", "pos"):
            assert not isinstance(getattr(m, attr, None), (torch.Tensor, int)), f"{name or 'model'}.{attr} is state"


def test_reset_reuses_the_cache(engine_model):
    model, cfg, rec = engine_model
    case = rec["greedy"][2]
    cache = new_cache(cfg)
    first = greedy(model, cfg, case["prompt_ids"], 16, cache)
    cache.reset()
    assert greedy(model, cfg, case["prompt_ids"], 16, cache) == first == case["output_ids"][:16]
