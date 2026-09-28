"""Multi-objective evaluation suite (M8): run on a trained checkpoint, with the candidate's own code.

    python -m autolab.evalsuite <run_dir> --val <val.pt>      (PYTHONPATH = the program's src)

Runs in the Modal container right after training, so model code, GPU and data match the trial. Writes
<run_dir>/eval.json with four groups of metrics:

- quality: full-val loss by position within the window (16 buckets), the loss on positions < 128 (short-
  context quality, comparable across context lengths) and on positions >= 128 (what longer context buys);
- context: copy-at-distance. A random 16-token span, then d tokens of real val text, then the span again;
  top-1 accuracy of continuing the repeat (tokens 2..16, teacher-forced) for each d in a FIXED set of
  distances. Distances beyond the model's context count as 0, so `long_range_score` (mean over the set)
  rewards context the model can actually use; `effective_context` = the largest d with accuracy >= 0.5.
  (A question-answering probe like "what was Alice's number?" needs instruction following a 16M-parameter
  base model doesn't have; copying a repeated span is what induction-style context use looks like at this scale.)
- inference: parameters, analytic forward FLOPs/token, prefill latency at batch 1 over a full window,
  decode tokens/sec at batch 1 (the model's own generate(), which re-runs the full context per token:
  there is no KV cache, so decode cost grows with context), and peak inference memory;
- training compute is taken from the run itself (throughput, wall time, peak memory) by the report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DISTANCES = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 992]
SPAN = 16
SHORT = 128


def load_model(run_dir: Path, device):
    from mini_llm.config import ModelConfig, build_model

    ckpt = torch.load(run_dir / "checkpoints" / "model.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def position_losses(model, val: torch.Tensor, T: int, device, max_windows: int = 2048, batch: int = 16) -> dict:
    n = min((val.numel() - 1) // T, max_windows)
    total = torch.zeros(T, dtype=torch.float64)
    for start in range(0, n, batch):
        idx = [i * T for i in range(start, min(start + batch, n))]
        x = torch.stack([val[o:o + T] for o in idx]).to(device)
        y = torch.stack([val[o + 1:o + T + 1] for o in idx]).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
        total += loss.view(len(idx), T).sum(0).double().cpu()
    per_pos = (total / n).tolist()
    k = max(1, T // 16)
    buckets = [[i * k, statistics.fmean(per_pos[i * k:(i + 1) * k])] for i in range(T // k)]
    out = {"windows": n, "by_position": buckets, "short_context_loss": statistics.fmean(per_pos[:min(SHORT, T)])}
    if T > SHORT:
        out["long_context_loss"] = statistics.fmean(per_pos[SHORT:])
    return out


@torch.no_grad()
def copy_at_distance(model, val: torch.Tensor, T: int, device, n: int = 64, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    acc = {}
    for d in DISTANCES:
        need = 2 * SPAN + d
        if need > T:
            acc[d] = 0.0  # beyond the model's context: it cannot use information that far back
            continue
        span_pos = torch.randint(0, val.numel() - SPAN, (n,), generator=g)
        fill_pos = torch.randint(0, val.numel() - d, (n,), generator=g)
        # a random span of real (but unrelated) tokens, so the repeat can't be predicted from local context
        spans = torch.stack([val[p:p + SPAN] for p in span_pos])
        fillers = torch.stack([val[p:p + d] for p in fill_pos])
        x = torch.cat([spans, fillers, spans], dim=1).to(device)
        logits, _ = model(x)
        start = SPAN + d  # position of the repeat's first token
        pred = logits[:, start:start + SPAN - 1].argmax(-1)  # predicts repeat tokens 2..SPAN
        target = x[:, start + 1:start + SPAN]
        acc[d] = (pred == target).float().mean().item()
    usable = [d for d, a in acc.items() if a >= 0.5]
    return {"accuracy_by_distance": {str(d): a for d, a in acc.items()},
            "long_range_score": statistics.fmean(acc.values()),
            "effective_context": max(usable) if usable else 0}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def inference_cost(model, cfg, device, decode_tokens: int = 64) -> dict:
    params = sum(p.numel() for p in model.parameters())
    emb = (cfg.vocab_size + cfg.block_size) * cfg.n_embd
    non_emb = params - emb
    T = cfg.block_size
    fwd_flops = 2 * non_emb + 2 * cfg.n_layer * T * cfg.n_embd + 2 * cfg.n_embd * cfg.vocab_size
    x = torch.randint(0, cfg.vocab_size, (1, T), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model(x)
    _sync(device)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append(time.perf_counter() - t0)
    prompt = torch.randint(0, cfg.vocab_size, (1, T // 2), device=device)
    model.generate(prompt, 4, T, greedy=True)  # warmup
    _sync(device)
    t0 = time.perf_counter()
    model.generate(prompt, decode_tokens, T, greedy=True)
    _sync(device)
    dt = time.perf_counter() - t0
    out = {"params": params, "non_embedding_params": non_emb, "fwd_flops_per_token": fwd_flops,
           "train_flops_per_token": 3 * fwd_flops, "prefill_ms": 1000 * statistics.median(times),
           "prefill_tokens_per_s": T / statistics.median(times), "decode_tokens_per_s": decode_tokens / dt,
           "decode_ms_per_token": 1000 * dt / decode_tokens, "context": T}
    if device.type == "cuda":
        out["peak_inference_mem_bytes"] = torch.cuda.max_memory_allocated()
    return out


def run(run_dir: Path, val_path: Path) -> dict:
    from mini_llm.device import select_device

    device = select_device()
    model, cfg = load_model(run_dir, device)
    val = torch.load(val_path).long()
    T = cfg.block_size
    t0 = time.time()
    out = {"context": T, "quality": position_losses(model, val, T, device),
           "context_capability": copy_at_distance(model, val, T, device),
           "inference": inference_cost(model, cfg, device), "device": str(device)}
    out["eval_s"] = round(time.time() - t0, 1)
    (run_dir / "eval.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--val", type=Path, required=True)
    a = p.parse_args(argv)
    out = run(a.run_dir, a.val)
    print(json.dumps({k: out[k] for k in ("context", "eval_s")}))


if __name__ == "__main__":
    main()
