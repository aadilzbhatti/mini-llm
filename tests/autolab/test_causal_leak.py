"""PROTECTED (never editable by candidates): no future-token or label leakage, and sane shapes.

Runs against whichever `mini_llm` is importable, so the evaluation cascade points
PYTHONPATH at a candidate program and runs this file. AUTOLAB_TEST_MODEL (JSON with
vocab_size/block_size/n_embd/n_head/n_layer) sets the model size to the candidate's
own hyperparameters; the default is a small config.

A model that sees tokens > t when predicting t+1 gets a dramatic, fake drop in loss,
so this is the most important gate. Dropout is off (eval mode) so any difference
in the logits is information flow, not noise.
"""

import json
import os

import pytest
import torch

from mini_llm.config import ModelConfig, build_model

DEFAULT = {"vocab_size": 97, "block_size": 32, "n_embd": 32, "n_head": 4, "n_layer": 2}


def model_cfg() -> dict:
    return {**DEFAULT, **json.loads(os.environ.get("AUTOLAB_TEST_MODEL", "{}")), "dropout": 0.0}


def build(seed: int):
    torch.manual_seed(seed)
    m = build_model(ModelConfig(**model_cfg()))
    m.eval()
    return m


def logits_of(model, idx, targets=None):
    logits, _ = model(idx, targets)
    return logits.view(idx.shape[0], idx.shape[1], -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
@torch.no_grad()
def test_no_future_leak(seed):
    cfg = model_cfg()
    model = build(seed)
    g = torch.Generator().manual_seed(100 + seed)
    B, T, V = 2, cfg["block_size"], cfg["vocab_size"]
    idx = torch.randint(0, V, (B, T), generator=g)
    base = logits_of(model, idx)
    for t in sorted({0, 1, T // 4, T // 2, T - 2}):
        changed = idx.clone()
        changed[:, t + 1 :] = torch.randint(0, V, (B, T - t - 1), generator=g)
        out = logits_of(model, changed)
        diff = (out[:, : t + 1] - base[:, : t + 1]).abs().max().item()
        assert diff <= 1e-5, f"logits at positions <= {t} changed by {diff:.3g} when tokens > {t} changed"


@torch.no_grad()
def test_logits_ignore_targets():
    cfg = model_cfg()
    model = build(3)
    g = torch.Generator().manual_seed(7)
    B, T, V = 2, cfg["block_size"], cfg["vocab_size"]
    idx = torch.randint(0, V, (B, T), generator=g)
    a = logits_of(model, idx, torch.randint(0, V, (B, T), generator=g))
    b = logits_of(model, idx, torch.randint(0, V, (B, T), generator=g))
    c = logits_of(model, idx)
    assert (a - b).abs().max().item() <= 1e-6, "logits depend on the targets passed to forward()"
    assert (a - c).abs().max().item() <= 1e-6, "logits differ with and without targets"


def test_shapes_and_backward():
    cfg = model_cfg()
    torch.manual_seed(0)
    model = build_model(ModelConfig(**{**cfg, "dropout": 0.1}))
    model.train()
    B, T, V = 3, cfg["block_size"], cfg["vocab_size"]
    idx = torch.randint(0, V, (B, T))
    logits, loss = model(idx, torch.randint(0, V, (B, T)))
    assert logits.shape == (B * T, V)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), "missing or non-finite grads"
    logits, loss = model(idx[:, : T // 2])  # shorter than block_size must work too
    assert logits.shape == (B, T // 2, V) and loss is None
