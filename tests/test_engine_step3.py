"""Engine step 3: static batching. Prompts of different lengths in one generate() call.

Per-sequence positions and cache lengths, padding (or no padding) handled so that no sequence sees
another's tokens, sequences finishing at different times, and one random generator per request
(a seeded torch.Generator draws exactly what torch.manual_seed would, so the golden sampled tokens
still apply). Every output must equal that prompt generated alone.

    uv run pytest tests/test_engine_step3.py --engine-todo
"""

import pytest

from engine import Engine, SamplingParams

pytestmark = pytest.mark.engine_step(3)


@pytest.fixture(scope="module")
def eng(engine_model):
    model, cfg, _ = engine_model
    return Engine(model, block_size=cfg.block_size, eos_token_id=50256)


def test_a_batch_of_different_lengths_matches_each_prompt_alone(eng, engine_model):
    _, _, rec = engine_model
    cases = rec["greedy"] + [rec["past_window"]]  # 8-10 tokens and 1000 tokens in one batch
    outs = eng.generate(
        [c["prompt_ids"] for c in cases], [SamplingParams(max_tokens=len(c["output_ids"])) for c in cases]
    )
    for c, out in zip(cases, outs):
        assert out.token_ids == c["output_ids"]


def test_sampled_requests_with_their_own_seeds_in_one_batch(eng, engine_model, golden):
    _, _, rec = engine_model
    sp = golden["sampled_params"]
    params = [
        SamplingParams(max_tokens=sp["max_tokens"], temperature=sp["temperature"], top_k=sp["top_k"], seed=c["seed"])
        for c in rec["sampled"]
    ]
    outs = eng.generate([c["prompt_ids"] for c in rec["sampled"]], params)
    for c, out in zip(rec["sampled"], outs):
        assert out.token_ids == c["output_ids"], f"seed {c['seed']}"


def test_requests_finish_independently(eng, engine_model):
    _, _, rec = engine_model
    a, b = rec["greedy"][0], rec["greedy"][1]
    short, long_ = eng.generate(
        [a["prompt_ids"], b["prompt_ids"]], [SamplingParams(max_tokens=3), SamplingParams(max_tokens=30)]
    )
    assert short.token_ids == a["output_ids"][:3] and short.finish_reason == "length"
    assert long_.token_ids == b["output_ids"][:30]
