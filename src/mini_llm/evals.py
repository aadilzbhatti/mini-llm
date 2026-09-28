"""Post-hoc evals: quality, context use, long-range retrieval, inference cost.

Validation loss alone can't separate "a better model" from "a model that got
more context at eval time", and says nothing about cost. This scores a saved
checkpoint along several axes so models can be compared on a Pareto table:

  quality     full_val at a fixed 128-token window (same for every model) and
              at the model's own block_size; mean loss by position in window.
  context     context_benefit: loss on the second half of each val window when
              the first half is the document's real preceding text vs text
              from a different document. The gap (nats) is how much the model
              uses the *meaning* of what came before, not just local syntax.
  retrieval   "The secret word is X." + real val text as filler + "The secret
              word is" -> forced choice of X among K single-token candidates
              (chance = 1/K), swept over key-to-question distance. Past the
              model's block_size the key is cropped out, exactly as in
              generation, so accuracy should fall to chance there.
  inference   prefill latency for a full-context prompt, decode tokens/sec
              (this model has no KV cache, so each new token re-runs the
              cropped window), peak memory.

    uv run mini-llm-eval checkpoints/a.pt checkpoints/b.pt      # -> evals/*.json|md, evals/summary.md

Everything is seeded; run all models on the same device before comparing the
inference numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import get_tokenizer, load_tokens
from mini_llm.device import select_device

EVALS_DIR = Path("evals")
DEFAULT_VAL = Path("data/data20k/val.pt")  # the shared 937-doc val set (see data/*/MANIFEST.md)
EOS = 50256
DISTANCES = (16, 32, 64, 96, 128, 160, 192, 224)
# Single GPT-2 tokens (with leading space), deliberately unrelated to each other.
CANDIDATES = (" apple", " river", " castle", " tiger", " violin", " copper", " garden", " rocket", " marble", " desert")
KEY_PREFIX = " The secret word is"


def load_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def position_losses(model, val: torch.Tensor, T: int, device, batch: int = 32) -> torch.Tensor:
    """Mean loss at each position 0..T-1 over every non-overlapping T-window."""
    n = (val.numel() - 1) // T
    tot, cnt = torch.zeros(T, dtype=torch.float64), 0
    for s in range(0, n, batch):
        offs = range(s, min(s + batch, n))
        x = torch.stack([val[o * T:o * T + T] for o in offs]).to(device)
        y = torch.stack([val[o * T + 1:o * T + T + 1] for o in offs]).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
        tot += loss.view(len(offs), T).sum(0).cpu().double()
        cnt += len(offs)
    return tot / cnt


def _documents(val: torch.Tensor) -> list[torch.Tensor]:
    ends = (val == EOS).nonzero().flatten().tolist()
    docs, start = [], 0
    for e in ends:
        docs.append(val[start:e])
        start = e + 1
    return docs


@torch.no_grad()
def context_benefit(model, val: torch.Tensor, T: int, device, max_windows: int = 400, seed: int = 0) -> dict:
    """Loss on window[T/2:] given its real first half vs a first half from another document."""
    half = T // 2
    docs = [d for d in _documents(val) if d.numel() >= T + 1]
    g = torch.Generator().manual_seed(seed)
    real_x, swap_x, ys = [], [], []
    for i in torch.randperm(len(docs), generator=g).tolist()[:max_windows]:
        d = docs[i]
        o = int(torch.randint(0, d.numel() - T, (1,), generator=g))
        w = d[o:o + T + 1]
        other = docs[(i + 1 + int(torch.randint(0, len(docs) - 1, (1,), generator=g))) % len(docs)]
        oo = int(torch.randint(0, other.numel() - half, (1,), generator=g))
        real_x.append(w[:T])
        swap_x.append(torch.cat([other[oo:oo + half], w[half:T]]))
        ys.append(w[1:T + 1])

    def second_half_loss(xs):
        losses = []
        for s in range(0, len(xs), 32):
            x = torch.stack(xs[s:s + 32]).to(device)
            y = torch.stack(ys[s:s + 32]).to(device)
            logits, _ = model(x)
            lp = F.cross_entropy(logits[:, half:].reshape(-1, logits.size(-1)).float(),
                                 y[:, half:].reshape(-1), reduction="none")
            losses.append(lp.cpu())
        return torch.cat(losses).mean().item()

    real, swapped = second_half_loss(real_x), second_half_loss(swap_x)
    return {"windows": len(real_x), "prefix_tokens": half, "loss_real_prefix": real,
            "loss_other_doc_prefix": swapped, "benefit_nats": swapped - real}


@torch.no_grad()
def retrieval(model, tokenizer, val: torch.Tensor, block_size: int, device,
              distances=DISTANCES, trials: int = 60, seed: int = 0) -> dict:
    """Forced-choice recall of a planted word vs key-to-question distance."""
    cand_ids = [tokenizer.encode(c) for c in CANDIDATES]
    assert all(len(c) == 1 for c in cand_ids), "every candidate must be a single token"
    cand = torch.tensor([c[0] for c in cand_ids])
    prefix = tokenizer.encode(KEY_PREFIX)
    stream = val[val != EOS]  # filler: real val text, document separators removed
    g = torch.Generator().manual_seed(seed)
    out = {}
    for dist in distances:
        n_fill = dist - len(prefix) - 2  # key word + "." + filler + query prefix == dist tokens back
        if n_fill < 1:
            continue
        correct, margins = 0, []
        for _ in range(trials):
            answer = int(torch.randint(0, len(cand), (1,), generator=g))
            o = int(torch.randint(0, stream.numel() - n_fill, (1,), generator=g))
            seq = prefix + [int(cand[answer]), 13] + stream[o:o + n_fill].tolist() + prefix  # 13 = "."
            x = torch.tensor(seq[-block_size:], device=device).unsqueeze(0)  # cropped like generation
            logits, _ = model(x)
            scores = logits[0, -1, cand.to(device)].float().log_softmax(-1).cpu()
            correct += int(scores.argmax()) == answer
            margins.append((scores[answer] - scores.mean()).item())
        # The key is only usable if its lead-in (" The secret word is") fits too: at
        # dist == block_size the key token is the window's first token and the
        # prefix that makes it retrievable is cropped away.
        out[str(dist)] = {"accuracy": correct / trials, "mean_logprob_margin": sum(margins) / trials,
                          "key_in_context": dist + len(prefix) <= block_size}
    return {"candidates": len(cand), "chance": 1 / len(cand), "trials_per_distance": trials, "by_distance": out}


@torch.no_grad()
def inference_cost(model, block_size: int, device, decode_tokens: int = 128, repeats: int = 5) -> dict:
    x = torch.randint(0, 50000, (1, block_size), device=device)
    for _ in range(2):  # warmup
        model(x)
    _sync(device)
    t = time.perf_counter()
    for _ in range(repeats):
        model(x)
    _sync(device)
    prefill_ms = (time.perf_counter() - t) / repeats * 1000
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    idx = x[:, : block_size // 2]
    _sync(device)
    t = time.perf_counter()
    for _ in range(decode_tokens):
        logits, _ = model(idx[:, -block_size:])
        idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)
    _sync(device)
    decode = decode_tokens / (time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else (
        torch.mps.current_allocated_memory() / 2**30 if device.type == "mps" else None)
    return {"device": str(device), "prefill_ms_full_context": round(prefill_ms, 2),
            "decode_tokens_per_sec": round(decode, 1), "memory_gb": round(mem, 3) if mem is not None else None,
            "note": "no KV cache: each decode step re-runs the (cropped) window"}


def evaluate_checkpoint(path: str | Path, val_path: str | Path = DEFAULT_VAL, device=None) -> dict:
    device = torch.device(device) if device else select_device()
    model, cfg, ckpt = load_model(path, device)
    tokenizer = get_tokenizer()
    val = load_tokens(val_path)
    T = cfg.block_size
    quality = {}
    for window in sorted({128, T}):
        if window <= T:
            p = position_losses(model, val, window, device)
            quality[f"full_val@{window}"] = p.mean().item()
            quality[f"by_position@{window}"] = {
                f"{a}-{b - 1}": p[a:b].mean().item() for a, b in
                [(0, 16), (16, 64), (64, window // 2 if window > 128 else 128)] + ([(128, window)] if window > 128 else [])
                if b > a}
    return {
        "checkpoint": str(path),
        "config": cfg.to_dict(),
        "params": sum(t.numel() for t in dict.fromkeys(model.parameters())),
        "step": ckpt.get("step"),
        "val_tokens": str(val_path),
        "quality": quality,
        "context_benefit": context_benefit(model, val, min(T, 128), device),
        "retrieval": retrieval(model, tokenizer, val, T, device),
        "inference": inference_cost(model, T, device),
        "training_systems": ckpt.get("systems"),
    }


def render_markdown(r: dict) -> str:
    q, rt, inf, ts = r["quality"], r["retrieval"], r["inference"], r.get("training_systems") or {}
    lines = [f"# Evals: {Path(r['checkpoint']).stem}", "",
             f"- config: {r['config']}", f"- params: {r['params']:,}", f"- step: {r['step']}",
             f"- val: {r['val_tokens']}", "", "## Quality", ""]
    for k, v in q.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: " + ", ".join(f"pos {p}: {x:.4f}" for p, x in v.items()))
        else:
            lines.append(f"- {k}: **{v:.4f}**")
    cb = r["context_benefit"]
    lines += ["", "## Context benefit", "",
              f"Loss on the second half of {cb['windows']} val windows, given a {cb['prefix_tokens']}-token prefix "
              "from the same document vs from a different document.", "",
              f"- same-document prefix: {cb['loss_real_prefix']:.4f}",
              f"- other-document prefix: {cb['loss_other_doc_prefix']:.4f}",
              f"- **benefit: {cb['benefit_nats']:.4f} nats**", "",
              "## Long-range retrieval", "",
              f"Forced choice among {rt['candidates']} single-token words (chance {rt['chance']:.0%}), "
              f"{rt['trials_per_distance']} trials per distance.", "",
              "| distance | accuracy | log-prob margin | key in context |", "|---|---|---|---|"]
    for d, v in rt["by_distance"].items():
        lines.append(f"| {d} | {v['accuracy']:.0%} | {v['mean_logprob_margin']:+.2f} | {'yes' if v['key_in_context'] else 'no'} |")
    lines += ["", "## Inference", "", f"- device: {inf['device']}",
              f"- prefill, full context: {inf['prefill_ms_full_context']} ms",
              f"- decode: {inf['decode_tokens_per_sec']} tokens/s ({inf['note']})",
              f"- memory: {inf['memory_gb']} GB", "", "## Training systems (recorded by the run)", ""]
    lines += [f"- {k}: {v}" for k, v in ts.items()] if ts else ["- not recorded (run predates mini_llm.systems)"]
    return "\n".join(lines) + "\n"


def summary_table(results: list[dict]) -> str:
    dists = [d for d in (str(x) for x in DISTANCES)]
    head = ["model", "ctx", "params", "val@128", "val@ctx", "ctx benefit"] + [f"ret@{d}" for d in dists] + \
           ["train tok/s", "train peak GB", "prefill ms", "decode tok/s"]
    lines = ["# Evals summary", "", "Quality, context use, long-range retrieval and cost, one row per checkpoint. "
             "Retrieval is forced-choice accuracy (chance 10%) at each key-to-question distance.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        T = r["config"]["block_size"]; q = r["quality"]; ts = r.get("training_systems") or {}
        by = r["retrieval"]["by_distance"]
        cells = [Path(r["checkpoint"]).stem, str(T), f"{r['params'] / 1e6:.1f}M",
                 f"{q.get('full_val@128', float('nan')):.4f}", f"{q.get(f'full_val@{T}', float('nan')):.4f}",
                 f"{r['context_benefit']['benefit_nats']:.3f}"]
        cells += [f"{by[d]['accuracy']:.0%}" if d in by else "" for d in dists]
        cells += [f"{ts['train_tokens_per_sec']:,.0f}" if ts.get("train_tokens_per_sec") else "–",
                  f"{ts['peak_mem_gb']:.2f}" if ts.get("peak_mem_gb") else "–",
                  f"{r['inference']['prefill_ms_full_context']}", f"{r['inference']['decode_tokens_per_sec']}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quality, context, retrieval and inference evals for checkpoints.")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--val-tokens", default=str(DEFAULT_VAL))
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(EVALS_DIR))
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoints:
        r = evaluate_checkpoint(ck, args.val_tokens, args.device)
        stem = Path(ck).stem
        (out / f"{stem}.json").write_text(json.dumps(r, indent=2))
        (out / f"{stem}.md").write_text(render_markdown(r))
        print(f"{stem}: val@128 {r['quality'].get('full_val@128', math.nan):.4f}  "
              f"benefit {r['context_benefit']['benefit_nats']:.3f}  -> {out / (stem + '.md')}", flush=True)
    results = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    (out / "summary.md").write_text(summary_table(results))
    print(f"summary of {len(results)} checkpoints -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
