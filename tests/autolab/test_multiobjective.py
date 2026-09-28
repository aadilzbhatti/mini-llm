"""M8: the four-dimensional evaluation, the Pareto frontier, and evolvable context length."""

import json
import random

import pytest
import torch

from autolab import evaluate as ev
from autolab import evalsuite
from autolab.generate import mutate
from autolab.pareto import compare, frontier, table
from autolab.program import Program

HP = {"n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.0, "batch_size": 64, "lr": 1.2e-3,
      "min_lr": 2e-6, "warmup_steps": 100, "weight_decay": 0.0}


def prog(pid, loss, lr_score, train_s, decode_ms, status="evaluated", ctx=128):
    return Program(id=pid, parent_id="p0", base_commit="x", blocks={}, hparams=HP, status=status,
                   scores={"full_mean": loss, "n_seeds": 3, "metrics": {
                       "long_range_score": lr_score, "train_wall_s": train_s, "decode_ms_per_token": decode_ms,
                       "context": ctx, "effective_context": 64, "params": 16e6}})


def test_frontier_keeps_tradeoffs_and_drops_dominated():
    a = prog("a", 4.40, 0.20, 3000, 6.0)             # best loss
    b = prog("b", 4.45, 0.55, 3600, 9.0, ctx=512)    # worse loss, much better context
    c = prog("c", 4.41, 0.19, 2000, 4.0)             # loss within noise of a, much cheaper
    d = prog("d", 4.50, 0.18, 3500, 7.0)             # worse than a everywhere
    e = prog("e", 4.39, 0.20, 3010, 6.1)             # a's twin within tolerances: tie, not domination
    front = {p.id for p in frontier([a, b, c, d, e], quality_tol=0.02)}
    assert front == {"b", "c"}  # c ties a/e on quality and context but is much cheaper: it dominates both
    assert "d" not in front and "b" in front and "c" in front
    rows = table([a, b, c, d], 0.02)
    assert rows[0]["frontier"] and not next(r for r in rows if r["id"] == "d")["frontier"]


def test_noise_ties_on_quality():
    x, y = prog("x", 4.400, 0.2, 3000, 6.0).scores, prog("y", 4.415, 0.2, 3000, 6.0).scores
    va = {"quality": 4.400, "context": 0.2, "train_cost": 3000, "inference_cost": 6.0}
    vb = {"quality": 4.415, "context": 0.2, "train_cost": 3000, "inference_cost": 6.0}
    assert compare(va, vb, quality_tol=0.02) == (0, 0)  # 0.015 < 1σ: a tie, nobody dominates
    assert compare(va, vb, quality_tol=0.01) == (1, 0)


def test_context_mutation_keeps_tokens_per_update():
    spec = ev.evolve_cfg()["hparams"]
    parent = Program(id="p0", parent_id=None, base_commit="x", blocks={}, hparams=HP)
    seen = set()
    for s in range(300):
        patch, _ = mutate(parent, spec, random.Random(s))
        hp = {**HP, **patch}
        assert hp["batch_size"] * hp.get("block_size", 128) == 8192, patch
        seen.add(hp.get("block_size", 128))
    assert 256 in seen


def test_static_rejects_context_that_breaks_tokens_per_update(tmp_path):
    paths = ev.Paths(tmp_path / "ev")
    p = Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams={**HP, "block_size": 256})  # bs 64
    import subprocess

    from autolab.config import REPO_ROOT
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    p.base_commit = head
    from autolab.program import base_sources, extract_blocks

    p.blocks = extract_blocks(base_sources(REPO_ROOT, head))
    assert ev.stage_static(p, ev.evolve_cfg(), paths, REPO_ROOT) is None
    assert "must equal 8192 tokens per update" in p.reason


def test_eval_suite_adapter_over_owner_evals(tmp_path):
    """autolab.evalsuite runs the owner's mini_llm.evals and maps it onto the four dimensions."""
    from mini_llm.config import ModelConfig, build_model

    (tmp_path / "checkpoints").mkdir()
    cfg = ModelConfig(vocab_size=50257, block_size=128, n_embd=16, n_head=2, n_layer=1)
    m = build_model(cfg)
    torch.save({"config": cfg.to_dict(), "model_state_dict": m.state_dict(), "step": 1,
                "systems": {"train_tokens_per_sec": 1234.0, "peak_mem_gb": 0.5, "train_sec": 10.0}},
               tmp_path / "checkpoints" / "model.pt")
    val = torch.load(ev.REPO_ROOT / "autolab" / "data" / "val_frozen.pt")[:60_000].clone()
    torch.save(val, tmp_path / "val.pt")
    out = evalsuite.run(tmp_path, tmp_path / "val.pt")
    cc = out["context_capability"]
    assert set(cc["accuracy_by_distance"]) >= {"16", "224", "896"}  # the owner's distances + the longer sweep
    assert 0.0 <= cc["long_range_score"] <= 1.0 and cc["chance"] == 0.1
    assert out["quality"]["full_val_at_128"] > 9  # untrained: ~ln(50257)
    assert out["inference"]["decode_ms_per_token"] > 0 and out["inference"]["prefill_ms"] > 0
    assert out["training"]["train_tokens_per_sec"] == 1234.0
    assert "retrieval" in out["owner_evals"] and json.loads((tmp_path / "eval.json").read_text())["context"] == 128


def test_run_metrics_from_report():
    rep = {"eval": {"context": 256, "context_capability": {"long_range_score": 0.3, "effective_context": 128},
                    "quality": {"short_context_loss": 4.5}, "inference": {"decode_ms_per_token": 7.0, "params": 16e6},
                    "training": {"train_tokens_per_sec": 40000.0, "peak_train_mem_bytes": 3e9}},
           "performance": {"train_wall_s": 2000.0}}
    m = ev.run_metrics(rep)
    assert m["long_range_score"] == 0.3 and m["context"] == 256 and m["train_wall_s"] == 2000.0
    assert ev.run_metrics({"performance": {}}) is None
