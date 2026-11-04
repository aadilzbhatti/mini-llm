"""The golden outputs still come out of mini_llm's own code (the reference the engine must match).

If this fails, mini_llm's generation changed (or the environment did): re-record deliberately with
`uv run python -m engine.golden record` and say why in the commit, never just to make a test pass.
"""

import pytest
import torch

from engine import golden as g


@pytest.fixture(scope="module")
def reference(golden, golden_name):
    rec = golden["models"][golden_name]
    model, cfg = g.load_reference(g.ROOT / rec["checkpoint"])
    return model, cfg, rec


def _gen(model, cfg, case, greedy=True, **kw):
    from mini_llm.report import generate_until_eos

    ids = case["prompt_ids"]
    if "seed" in case:
        torch.manual_seed(case["seed"])
    out, hit_eos = generate_until_eos(
        model,
        torch.tensor([ids]),
        len(case["output_ids"]) + (case["finish_reason"] == "eos"),
        cfg.block_size,
        greedy=greedy,
        **kw,
    )
    return out[0, len(ids) :].tolist(), "eos" if hit_eos else "length"


@torch.no_grad()
def test_reference_still_produces_the_golden_outputs(reference, golden):
    model, cfg, rec = reference
    for case in rec["greedy"] + [rec["past_window"]]:
        assert _gen(model, cfg, case) == (case["output_ids"], case["finish_reason"])
    sp = golden["sampled_params"]
    for case in rec["sampled"]:
        got = _gen(model, cfg, case, greedy=False, temperature=sp["temperature"], top_k=sp["top_k"])
        assert got == (case["output_ids"], case["finish_reason"]), f"seed {case['seed']}"
    val = torch.load(g.VAL)[: cfg.block_size].unsqueeze(0)
    logits, _ = model(val)
    for pos, row in rec["logits"]["positions"].items():
        v, i = torch.topk(logits[0, int(pos)], len(row["top_ids"]))
        assert i.tolist() == row["top_ids"]
        assert torch.allclose(v, torch.tensor(row["top_logits"]), atol=golden["logit_atol"])
