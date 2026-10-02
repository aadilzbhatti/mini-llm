"""mini_llm.evals on a tiny random model: shapes, flags, and the cropping guarantee."""

import torch

from mini_llm import evals
from mini_llm.config import ModelConfig, build_model


def tiny(tmp_path, block_size=128):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50257, block_size=block_size, n_embd=16, n_head=2, n_layer=1)
    path = tmp_path / f"t{block_size}.pt"
    torch.save({"config": cfg.to_dict(), "model_state_dict": build_model(cfg).state_dict(), "step": 1,
                "systems": {"train_tokens_per_sec": 123.0, "peak_mem_gb": 0.5}}, path)
    g = torch.Generator().manual_seed(1)
    docs = [torch.cat([torch.randint(0, 50000, (int(n),), generator=g), torch.tensor([evals.EOS])])
            for n in torch.randint(150, 400, (60,), generator=g)]
    val = tmp_path / "val.pt"
    torch.save(torch.cat(docs), val)
    return path, val


def test_full_report_and_summary(tmp_path):
    ck, val = tiny(tmp_path)
    r = evals.evaluate_checkpoint(ck, val, device="cpu")
    by = r["retrieval"]["by_distance"]
    assert set(by) == {str(d) for d in evals.DISTANCES}
    assert by["96"]["key_in_context"] and not by["128"]["key_in_context"]  # lead-in cropped at 128
    assert len(by["16"]["hits"]) == evals.RETRIEVAL_TRIALS and by["16"]["ci95"][0] <= by["16"]["accuracy"] <= by["16"]["ci95"][1]
    cb = r["context_benefit"]["cb@128"]
    assert cb["protocol"] == "window 128, prefix 64, seed 0" and cb["windows"] == len(cb["per_window_benefit"]) > 0
    assert "cb@256" not in r["context_benefit"] and "full_val@128" in r["quality"]
    cc = r["context_curve"]
    assert list(cc["by_context"]) == ["16", "32", "64", "128"] and list(cc["gain"]) == ["16->32", "32->64", "64->128"]
    assert all(len(v) == evals.CURVE_TARGETS for v in cc["per_target"].values())
    assert r["training_systems"]["train_tokens_per_sec"] == 123.0
    from mini_llm.samples import DRAWS, PROMPT_LABELS, label, render_comparison, render_model
    sm = r["samples"]
    assert [p["label"] for p in sm["prompts"]] == list(PROMPT_LABELS) and all(len(p["draws"]) == DRAWS for p in sm["prompts"])
    assert sm["summary"]["n"] == len(PROMPT_LABELS) * DRAWS and {"rep4", "looped", "topic"} <= set(sm["prompts"][0]["draws"][0])
    md, table = evals.render_markdown(r), evals.summary_table([r, {**r, "checkpoint": "other.pt"}], reference="t128")
    assert "## Long-range retrieval" in md and "## Generation samples" in md and f"| {label(r)} | 128 |" in table
    assert f"## Paired against {label(r)}" in table and "+0.0000 ± 0.0000" in table  # identical model: zero paired difference
    assert render_model(r).startswith(f"# Samples: {label(r)}") and "| M1 | " in render_comparison([r])
    assert "## Context curve (fixed targets)" in md and "L(c=128)" in table


def test_key_outside_context_cannot_be_retrieved(tmp_path):
    """Past block_size the key is cropped away, so the model's choice can't depend on it."""
    ck, val = tiny(tmp_path)
    model, cfg, _ = evals.load_model(ck, torch.device("cpu"))
    from mini_llm.data import get_tokenizer
    r = evals.retrieval(model, get_tokenizer(), torch.load(val), cfg.block_size, "cpu", distances=(256,), trials=40)
    # With the key gone, the prediction is identical whichever word was planted: accuracy is ~chance.
    assert r["by_distance"]["256"]["accuracy"] <= 0.35


def test_bench_runs_models_in_one_session(tmp_path):
    from mini_llm import bench
    a, _ = tiny(tmp_path, 128)
    b, _ = tiny(tmp_path, 256)
    res = bench.benchmark([str(a), str(b)], device="cpu", rounds=2, prefills_per_round=2)
    assert set(res["models"]) == {"t128", "t256"}
    m = res["models"]["t256"]
    assert m["block_size"] == 256 and m["decode_tok_s"]["n"] == 2 and m["prefill_full_ms"]["n"] == 4
    assert "| t128 | 128 |" in bench.render(res)


def test_sample_metrics():
    from mini_llm.samples import loop_info, rep4, topic_retention
    assert loop_info(list(range(40)))["looped"] is False
    looped = list(range(10)) + [7, 8, 9] * 12  # a 3-token cycle from token 7 to the end
    assert loop_info(looped) == {"looped": True, "period": 3, "onset": 7}
    assert rep4([1, 2, 3, 4] * 5) > 0.7 and rep4(list(range(20))) == 0
    p = "Albert Einstein was a German-born theoretical physicist who"
    assert topic_retention(p, "Einstein studied physics in Germany") == 3 / 6  # einst, germa, physi of 6 stems
    assert topic_retention(p, "the cat sat on the mat") == 0


def test_more_sample_metrics_and_summary():
    from mini_llm.data import get_tokenizer
    from mini_llm.samples import distinct, summarize, topic_span
    assert distinct([1, 2, 1, 2, 1, 2], 2) == 2 / 5 and distinct(list(range(10)), 4) == 1.0
    tok = get_tokenizer()
    ids = tok.encode(" Einstein was born in Ulm. Later he moved away. The weather was nice and the food was good.")
    span = topic_span("Albert Einstein was a German-born theoretical physicist who", ids, tok)
    assert span == 3  # " Einstein was born": "born" (from "German-born") is the last mention, token 3
    assert topic_span("Photosynthesis is a process that", ids, tok) == 0
    s = summarize([{"rep4": r, "distinct2": 0.9, "distinct4": 0.95, "looped": r > 0.5, "loop_onset": 100 if r > 0.5 else None,
                    "topic": 0.5, "topic_span": 120, "eos": False, "tokens": 256} for r in (0.1, 0.2, 0.3, 0.6, 0.9)])
    assert s["rep4"] == 0.3 and s["looped"] == 2 and s["loop_onset_median"] == 100 and s["topic_span_median"] == 120
    assert s["loop_ci95"][0] < 0.4 < s["loop_ci95"][1]


def test_generate_samples_reuses_existing_draws(monkeypatch):
    from mini_llm import samples
    from mini_llm.data import get_tokenizer
    prev = {"temperature": samples.TEMPERATURE, "top_k": samples.TOP_K, "new_tokens": samples.NEW_TOKENS,
            "prompts": [{"prompt": p, "draws": [{"text": " the same text again", "tokens": 4, "eos": False}] * samples.DRAWS}
                        for _, p in samples.GEN_PROMPTS[:-1]]}  # all but the last prompt already generated
    calls = []
    monkeypatch.setattr(samples, "complete", lambda *a, **k: calls.append(a[4]) or {
        "text": " new", "tokens": 1, "eos": True, **samples.score(a[4], " new", a[1])})
    out = samples.generate_samples(None, get_tokenizer(), 128, "cpu", previous=prev)
    assert calls == [samples.GEN_PROMPTS[-1][1]] * samples.DRAWS  # only the missing prompt is generated
    assert out["summary"]["n"] == len(samples.GEN_PROMPTS) * samples.DRAWS
    assert out["prompts"][0]["draws"][0]["text"] == " the same text again" and "distinct2" in out["prompts"][0]["draws"][0]
