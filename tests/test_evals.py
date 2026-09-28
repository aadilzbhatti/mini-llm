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
    assert r["training_systems"]["train_tokens_per_sec"] == 123.0
    md, table = evals.render_markdown(r), evals.summary_table([r, {**r, "checkpoint": "other.pt"}], reference="t128")
    assert "## Long-range retrieval" in md and "| t128 | 128 |" in table
    assert "## Paired against t128" in table and "+0.0000 ± 0.0000" in table  # identical model: zero paired difference


def test_key_outside_context_cannot_be_retrieved(tmp_path):
    """Past block_size the key is cropped away, so the model's choice can't depend on it."""
    ck, val = tiny(tmp_path)
    model, cfg, _ = evals.load_model(ck, torch.device("cpu"))
    from mini_llm.data import get_tokenizer
    r = evals.retrieval(model, get_tokenizer(), torch.load(val), cfg.block_size, "cpu", distances=(256,), trials=40)
    # With the key gone, the prediction is identical whichever word was planted: accuracy is ~chance.
    assert r["by_distance"]["256"]["accuracy"] <= 0.35
