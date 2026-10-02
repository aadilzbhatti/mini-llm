"""KV cache vs. no cache on the best baseline checkpoint.

The best baseline is the baselines.json row with the lowest full_val_loss
whose checkpoint is on disk. The test is skipped when there is none (CI, a
fresh clone), since checkpoints are not committed.

Runs on CPU in float32 so the cached and uncached paths are numerically
close enough that greedy decoding picks the same tokens and seeded
sampling draws the same ones.
"""

import json
import time
from pathlib import Path

import pytest
import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import encode, get_tokenizer
from mini_llm.model import ModelCustomTransformer

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "The lighthouse keeper watched the ships",
    "In 1905, Albert Einstein published",
    "The city of Paris is",
    "Photosynthesis is the process by which",
    "The war ended when",
]


def best_baseline_checkpoint() -> Path | None:
    rows = json.loads((ROOT / "baselines.json").read_text())
    rows = [r for r in rows if r.get("checkpoint") and (ROOT / r["checkpoint"]).exists()]
    if not rows:
        return None
    best = min(rows, key=lambda r: r.get("full_val_loss") or r.get("eval_val_loss") or float("inf"))
    return ROOT / best["checkpoint"]


CHECKPOINT = best_baseline_checkpoint()
pytestmark = pytest.mark.skipif(CHECKPOINT is None, reason="no baseline checkpoint on disk")


@pytest.fixture(scope="module")
def models() -> tuple[ModelCustomTransformer, ModelCustomTransformer]:
    """(uncached, cached) copies of the best baseline, same weights."""
    assert CHECKPOINT is not None
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    out = []
    for use_cache in (False, True):
        cfg = ModelConfig(**ckpt["config"])
        cfg.use_cache = use_cache
        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        out.append(model)
    return out[0], out[1]


def time_generate(model: ModelCustomTransformer, idx: torch.Tensor, max_new_tokens: int, block_size: int) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=max_new_tokens, block_size=block_size, greedy=False)
    return out, time.perf_counter() - t0


@torch.no_grad()
@pytest.mark.parametrize("prompt", PROMPTS)
def test_cached_step_logits_match_full_forward(models, prompt: str):
    """Prefill the prompt, then feed one token through the cache: its logits
    must match running the whole sequence through the uncached model."""
    plain, cached = models
    tokenizer = get_tokenizer()
    idx = encode(prompt, tokenizer).unsqueeze(0)
    next_tok = torch.tensor([[tokenizer.encode(" the")[0]]])

    cached.clear_cache()
    cached(idx)  # prefill
    step_logits, _ = cached(next_tok)
    cached.clear_cache()

    full_logits, _ = plain(torch.cat([idx, next_tok], dim=1))
    assert torch.allclose(step_logits[:, -1], full_logits[:, -1], atol=1e-4, rtol=1e-4)


@torch.no_grad()
@pytest.mark.parametrize("block_size", [1024, 32], ids=["full-window", "window-overflow"])
def test_cached_greedy_generation_matches_uncached(models, block_size: int):
    """Greedy outputs must be identical with and without the cache. The
    block_size=32 case generates past the window, exercising the refill path."""
    plain, cached = models
    tokenizer = get_tokenizer()
    max_new_tokens = 48

    total_plain = total_cached = 0.0
    for prompt in PROMPTS:
        idx = encode(prompt, tokenizer).unsqueeze(0)
        out_plain, t_plain = time_generate(plain, idx, max_new_tokens, block_size)
        out_cached, t_cached = time_generate(cached, idx, max_new_tokens, block_size)
        total_plain += t_plain
        total_cached += t_cached

        print(f"\n[{prompt!r}] no cache {t_plain*1000:.0f} ms, cache {t_cached*1000:.0f} ms")
        print(f"  -> {tokenizer.decode(out_cached[0])!r}")
        assert torch.equal(out_plain, out_cached), (
            f"cached output diverged for {prompt!r}:\n"
            f"  plain:  {tokenizer.decode(out_plain[0])!r}\n"
            f"  cached: {tokenizer.decode(out_cached[0])!r}"
        )

    print(f"\nblock_size={block_size}: no cache {total_plain:.2f} s, cache {total_cached:.2f} s, "
          f"speedup {total_plain/total_cached:.2f}x")


@torch.no_grad()
@pytest.mark.parametrize("block_size", [1024, 32], ids=["full-window", "window-overflow"])
def test_generate_until_eos_cached_matches_uncached(models, block_size: int):
    """The server decodes through report.generate_until_eos, not model.generate;
    it must give identical greedy output with the cache on, and leave the
    shared model with an empty cache for the next request."""
    from mini_llm.report import generate_until_eos

    plain, cached = models
    tokenizer = get_tokenizer()
    for prompt in PROMPTS:
        idx = encode(prompt, tokenizer).unsqueeze(0)
        out_plain, eos_plain = generate_until_eos(plain, idx, 48, block_size, greedy=True)
        out_cached, eos_cached = generate_until_eos(cached, idx, 48, block_size, greedy=True)
        assert torch.equal(out_plain, out_cached) and eos_plain == eos_cached, prompt
        assert cached.cache_len() == 0


SEEDS = [0, 1, 2]
# The server's default (T=0.7, top-k 40) plus plain and nucleus sampling.
SAMPLING_SETTINGS = {
    "server-default": {"temperature": 0.7, "top_k": 40},
    "plain": {"temperature": 1.0},
    "nucleus": {"temperature": 0.9, "top_p": 0.9},
}


@torch.no_grad()
@pytest.mark.parametrize("block_size", [1024, 32], ids=["full-window", "window-overflow"])
@pytest.mark.parametrize("setting", SAMPLING_SETTINGS, ids=list(SAMPLING_SETTINGS))
def test_sampled_generate_until_eos_matches_uncached_per_seed(models, setting: str, block_size: int):
    """With the same seed, sampled decoding must give the same tokens and the
    same EOS stop with and without the cache. Sampling draws one multinomial
    per step, so this only holds if every step's probabilities match."""
    from mini_llm.report import generate_until_eos

    plain, cached = models
    tokenizer = get_tokenizer()
    kwargs = SAMPLING_SETTINGS[setting]
    for prompt in PROMPTS:
        idx = encode(prompt, tokenizer).unsqueeze(0)
        for seed in SEEDS:
            torch.manual_seed(seed)
            out_plain, eos_plain = generate_until_eos(plain, idx, 32, block_size, **kwargs)
            torch.manual_seed(seed)
            out_cached, eos_cached = generate_until_eos(cached, idx, 32, block_size, **kwargs)
            assert torch.equal(out_plain, out_cached) and eos_plain == eos_cached, (
                f"{setting} seed={seed} diverged for {prompt!r}:\n"
                f"  plain:  {tokenizer.decode(out_plain[0])!r}\n"
                f"  cached: {tokenizer.decode(out_cached[0])!r}"
            )


@torch.no_grad()
@pytest.mark.parametrize("block_size", [1024, 32], ids=["full-window", "window-overflow"])
def test_sampled_model_generate_matches_uncached_per_seed(models, block_size: int):
    """Same check for model.generate's own multinomial sampling (the CLI path)."""
    plain, cached = models
    tokenizer = get_tokenizer()
    for prompt in PROMPTS:
        idx = encode(prompt, tokenizer).unsqueeze(0)
        for seed in SEEDS:
            torch.manual_seed(seed)
            out_plain = plain.generate(idx, max_new_tokens=32, block_size=block_size)
            torch.manual_seed(seed)
            out_cached = cached.generate(idx, max_new_tokens=32, block_size=block_size)
            assert torch.equal(out_plain, out_cached), f"seed={seed} diverged for {prompt!r}"
