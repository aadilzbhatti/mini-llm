"""Engine step 2: the engine owns generation (sampler, stopping, window policy), one prompt at a time.

Engine.generate replaces mini_llm's model.generate / report.generate_until_eos and must reproduce the
golden outputs token for token: greedy, seeded sampling (the procedure is in engine/sampling.py), and
generation past the window.

    uv run pytest tests/test_engine_step2.py --engine-todo
"""

import pytest

from engine import Engine, SamplingParams

pytestmark = pytest.mark.engine_step(2)
EOS = 50256


@pytest.fixture(scope="module")
def eng(engine_model):
    from mini_llm.data import get_tokenizer

    model, cfg, _ = engine_model
    return Engine(model, block_size=cfg.block_size, tokenizer=get_tokenizer(), eos_token_id=EOS)


def test_greedy_from_token_ids(eng, engine_model):
    _, _, rec = engine_model
    for case in rec["greedy"]:
        [out] = eng.generate([case["prompt_ids"]], SamplingParams(max_tokens=len(case["output_ids"])))
        assert out.token_ids == case["output_ids"] and out.finish_reason == case["finish_reason"]
        assert out.prompt_token_ids == case["prompt_ids"]


def test_text_in_text_out(eng, engine_model):
    from mini_llm.data import decode, get_tokenizer

    import torch

    _, _, rec = engine_model
    case = rec["greedy"][0]
    [out] = eng.generate([case["prompt"]], SamplingParams(max_tokens=len(case["output_ids"])))
    assert out.token_ids == case["output_ids"]
    assert out.text == decode(torch.tensor(case["output_ids"]), get_tokenizer())


def test_seeded_sampling_reproduces_the_golden_tokens(eng, engine_model, golden):
    _, _, rec = engine_model
    sp = golden["sampled_params"]
    for case in rec["sampled"]:
        params = SamplingParams(
            max_tokens=sp["max_tokens"], temperature=sp["temperature"], top_k=sp["top_k"], seed=case["seed"]
        )
        [out] = eng.generate([case["prompt_ids"]], params)
        assert out.token_ids == case["output_ids"], f"{case['prompt']!r} seed {case['seed']}"
        [again] = eng.generate([case["prompt_ids"]], params)
        assert again.token_ids == out.token_ids  # a seed means the same tokens every time


def test_generation_past_the_window(eng, engine_model):
    _, _, rec = engine_model
    case = rec["past_window"]
    [out] = eng.generate([case["prompt_ids"]], SamplingParams(max_tokens=len(case["output_ids"])))
    assert out.token_ids == case["output_ids"]


def test_max_tokens_and_eos(eng, engine_model):
    _, _, rec = engine_model
    case = rec["greedy"][1]
    [out] = eng.generate([case["prompt_ids"]], SamplingParams(max_tokens=5))
    assert out.token_ids == case["output_ids"][:5] and out.finish_reason == "length"
    assert all(t != EOS for t in out.token_ids)
