"""mini_llm.evals on a tiny random model: shapes, flags, and the cropping guarantee."""

import torch

from mini_llm import evals
from mini_llm.config import ModelConfig, build_model


def tiny(tmp_path, block_size=32):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50257, block_size=block_size, n_embd=16, n_head=2, n_layer=1)
    path = tmp_path / f"t{block_size}.pt"
    torch.save({"config": cfg.to_dict(), "model_state_dict": build_model(cfg).state_dict(), "step": 1,
                "systems": {"train_tokens_per_sec": 123.0, "peak_mem_gb": 0.5}}, path)
    g = torch.Generator().manual_seed(1)
    docs = [torch.cat([torch.randint(0, 50000, (int(n),), generator=g), torch.tensor([evals.EOS])])
            for n in torch.randint(80, 200, (60,), generator=g)]
    val = tmp_path / "val.pt"
    torch.save(torch.cat(docs), val)
    return path, val


def test_full_report_and_summary(tmp_path):
    ck, val = tiny(tmp_path)
    r = evals.evaluate_checkpoint(ck, val, device="cpu")
    by = r["retrieval"]["by_distance"]
    assert set(by) == {"16", "32", "64", "96", "128", "160", "192", "224"}
    assert by["16"]["key_in_context"] and not by["64"]["key_in_context"]  # block_size 32
    assert r["context_benefit"]["windows"] > 0 and "full_val@128" not in r["quality"]  # window > block_size
    assert r["training_systems"]["train_tokens_per_sec"] == 123.0
    md, table = evals.render_markdown(r), evals.summary_table([r])
    assert "## Long-range retrieval" in md and "| t32 | 32 |" in table


def test_key_outside_context_cannot_be_retrieved(tmp_path):
    """Past block_size the key is cropped away, so the model's choice can't depend on it."""
    ck, val = tiny(tmp_path)
    model, cfg, _ = evals.load_model(ck, torch.device("cpu"))
    from mini_llm.data import get_tokenizer
    r = evals.retrieval(model, get_tokenizer(), torch.load(val), cfg.block_size, "cpu", distances=(64,), trials=40)
    # With the key gone, the prediction is identical whichever word was planted: accuracy is ~chance.
    assert r["by_distance"]["64"]["accuracy"] <= 0.35
