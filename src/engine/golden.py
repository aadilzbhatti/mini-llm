"""Golden outputs: what the training code's generation produces today, frozen as the engine's target.

    uv run python -m engine.golden record     # -> tests/golden/engine_golden.json (CPU, float32)

Recorded from mini_llm's own models and generation (mini_llm.report.generate_until_eos), before any
decoupling, so they stay a valid reference however far the engine's code diverges. The engine is done
with a step when it reproduces these: the same greedy tokens (exactly), the same seeded sampled
tokens (exactly), and the same logits (within LOGIT_ATOL), including generation past the window.

CPU and float32 so the reference is deterministic and portable (MPS and CUDA can break near-ties
differently). Checkpoints are local files (untracked); tests skip a model whose checkpoint is absent.

Sampling procedure (what the engine's sampler must do to match the sampled cases exactly):
  torch.manual_seed(seed) once per prompt, then each step: logits / temperature; keep the top_k
  logits (set the rest to -inf); softmax; torch.multinomial(probs, 1). Greedy = argmax. Stop before
  appending EOS (id 50256). One prompt at a time (batch 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "engine_golden.json"
VAL = ROOT / "data" / "data20k" / "val.pt"  # the fixed validation stream: deterministic long prompts
LOGIT_ATOL = 1e-3

CHECKPOINTS = {
    # model.py (per-head attention), RoPE, d512-L4, 38.4M
    "rope-39m": "checkpoints/modal_data320k-b8-t1024-e512h8-80k-lr6e-4-wu256k-rope_steps80000_seed42.pt",
    # model_fused.py, RoPE, d768-L8-H12, 95.3M
    "rope-fused-95m": "checkpoints/modal_data640k-b8-t1024-e768h12l8-160k-lr3e-4-wu256k-rope-fused_steps160000_seed42.pt",
}
TEXT_PROMPTS = [
    "The French Revolution began in 1789, when",
    "In mathematics, a prime number is",
    "To make bread at home, you will need",
]
GREEDY_TOKENS = 48
PAST_WINDOW_PROMPT = 1000  # tokens of the val stream; + PAST_WINDOW_TOKENS crosses the 1024 window
PAST_WINDOW_TOKENS = 80
SAMPLED = {"temperature": 0.7, "top_k": 40, "max_tokens": 48, "seeds": [0, 1]}
LOGIT_POSITIONS = [0, 255, 511, 1023]  # rows of one uncached 1024-token forward
TOP = 10


def load_reference(path: Path):
    """The training code's model, exactly as mini_llm builds it."""
    from mini_llm.config import ModelConfig, build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def record_model(path: Path) -> dict:
    from mini_llm.data import encode, get_tokenizer
    from mini_llm.report import generate_until_eos

    model, cfg = load_reference(path)
    tok = get_tokenizer()
    T = cfg.block_size
    val = torch.load(VAL)

    def run(ids: list[int], max_new: int, greedy: bool, **kw) -> dict:
        out, hit_eos = generate_until_eos(model, torch.tensor([ids]), max_new, T, greedy=greedy, **kw)
        new = out[0, len(ids) :].tolist()
        return {"prompt_ids": ids, "output_ids": new, "finish_reason": "eos" if hit_eos else "length"}

    greedy = [{"prompt": p, **run(encode(p, tok).tolist(), GREEDY_TOKENS, True)} for p in TEXT_PROMPTS]
    past = {
        "note": f"{PAST_WINDOW_PROMPT}-token prompt from the val stream + {PAST_WINDOW_TOKENS} new tokens: "
        f"generation crosses the {T}-token window, through the rolling RoPE cache",
        **run(val[:PAST_WINDOW_PROMPT].tolist(), PAST_WINDOW_TOKENS, True),
    }
    sampled = []
    for p in TEXT_PROMPTS[:2]:
        for seed in SAMPLED["seeds"]:
            torch.manual_seed(seed)
            r = run(
                encode(p, tok).tolist(),
                SAMPLED["max_tokens"],
                False,
                temperature=SAMPLED["temperature"],
                top_k=SAMPLED["top_k"],
            )
            sampled.append({"prompt": p, "seed": seed, **r})

    x = val[:T].unsqueeze(0)
    logits, _ = model(x)  # stateless, uncached forward
    rows = {}
    for pos in LOGIT_POSITIONS:
        row = logits[0, pos]
        v, i = torch.topk(row, TOP)
        rows[str(pos)] = {
            "top_ids": i.tolist(),
            "top_logits": [round(float(a), 5) for a in v],
            "mean": round(float(row.mean()), 6),
            "std": round(float(row.std()), 6),
        }
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(path),
        "config": cfg.to_dict(),
        "greedy": greedy,
        "past_window": past,
        "sampled": sampled,
        "logits": {"input": f"val stream tokens [0, {T})", "positions": rows},
    }


def record(names: list[str] | None = None) -> dict:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "dtype": "float32",
        "torch": torch.__version__,
        "protocol": __doc__.split("Sampling procedure", 1)[1].strip(),
        "sampled_params": {k: v for k, v in SAMPLED.items() if k != "seeds"},
        "logit_atol": LOGIT_ATOL,
        "models": {},
    }
    for name, rel in CHECKPOINTS.items():
        if names and name not in names:
            continue
        path = ROOT / rel
        if not path.exists():
            print(f"{name}: {rel} not found, skipped")
            continue
        t = time.time()
        out["models"][name] = record_model(path)
        print(f"{name}: recorded in {time.time() - t:.0f}s")
    return out


def load() -> dict:
    return json.loads(GOLDEN.read_text())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Record the engine's golden outputs from mini_llm's current code.")
    p.add_argument("command", choices=["record"])
    p.add_argument("--model", action="append", choices=list(CHECKPOINTS), help="Only these (default: all).")
    args = p.parse_args(argv)
    data = record(args.model)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
