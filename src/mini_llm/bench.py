"""Controlled inference benchmark: every model in one process, same conditions.

Timing one model per eval run is unreliable on a shared machine (a single
reading once came out 25x off), so this compares checkpoints the way a
benchmark should:

  - all models loaded into one process on one device, each warmed up (MPS/CUDA
    compile kernels per shape on first use, so warmup covers every shape the
    timed runs use);
  - interleaved rounds, rotating the model order each round, so a drift in
    system load hits every model equally instead of whichever ran last;
  - synchronize around every timed region (GPU work is asynchronous);
  - report median and p10-p90 over all rounds.

Measured per model:
  prefill_full_ms   one forward pass over a full block_size window
  prefill_128_ms    one forward pass over 128 tokens: identical work for every
                    model here, so it doubles as a noise check
  decode_tok_s      64 new tokens from a prompt of block_size - 64 tokens,
                    greedy. This model has no KV cache, so every step re-runs
                    the whole (cropped) window: the steady state of uncached
                    decoding at a full context.

    uv run mini-llm-bench checkpoints/a.pt checkpoints/b.pt   # -> evals/inference.json|md
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mini_llm.device import select_device
from mini_llm.evals import EVALS_DIR, _sync, eval_results, load_model

DECODE_TOKENS = 64


def _stats(xs: list[float]) -> dict:
    q = statistics.quantiles(xs, n=10) if len(xs) >= 2 else [xs[0]] * 9
    return {"median": round(statistics.median(xs), 3), "p10": round(q[0], 3), "p90": round(q[-1], 3), "n": len(xs)}


@torch.no_grad()
def _prefill_ms(model, x, device) -> float:
    _sync(device)
    t = time.perf_counter()
    model(x)
    _sync(device)
    return (time.perf_counter() - t) * 1000


@torch.no_grad()
def _decode_tok_s(model, prompt, block_size: int, device, n: int = DECODE_TOKENS) -> float:
    idx = prompt
    _sync(device)
    t = time.perf_counter()
    for _ in range(n):
        logits, _ = model(idx[:, -block_size:])
        idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)
    _sync(device)
    return n / (time.perf_counter() - t)


@torch.no_grad()
def _decode_cached_tok_s(model, prompt, block_size: int, device, n: int = DECODE_TOKENS) -> float:
    """Same work as _decode_tok_s, generated the way the evals and the server do: with the KV cache."""
    from mini_llm.report import generate_until_eos
    model.set_use_cache(True)
    try:
        _sync(device)
        t = time.perf_counter()
        generate_until_eos(model, prompt, n, block_size, greedy=True, eos_token_id=None)
        _sync(device)
        return n / (time.perf_counter() - t)
    finally:
        model.set_use_cache(False)


def benchmark(checkpoints: list[str], device=None, rounds: int = 7, prefills_per_round: int = 5) -> dict:
    device = torch.device(device) if device else select_device()
    models = {}
    for ck in checkpoints:
        model, cfg, _ = load_model(ck, device)
        T = cfg.block_size
        g = torch.Generator().manual_seed(0)
        full = torch.randint(0, 50000, (1, T), generator=g).to(device)
        models[Path(ck).stem] = {"model": model, "T": T, "full": full, "short": full[:, :128],
                                 "prompt": full[:, : T - DECODE_TOKENS]}
    for m in models.values():  # warmup: every shape the timed runs will use
        for _ in range(3):
            m["model"](m["full"]); m["model"](m["short"])
        _decode_tok_s(m["model"], m["prompt"], m["T"], device)
        _decode_cached_tok_s(m["model"], m["prompt"], m["T"], device)
    samples = {k: {"prefill_full_ms": [], "prefill_128_ms": [], "decode_tok_s": [], "decode_cached_tok_s": []} for k in models}
    names = list(models)
    for r in range(rounds):
        for k in names[r % len(names):] + names[: r % len(names)]:  # rotate the order each round
            m, sm = models[k], samples[k]
            sm["prefill_full_ms"] += [_prefill_ms(m["model"], m["full"], device) for _ in range(prefills_per_round)]
            sm["prefill_128_ms"] += [_prefill_ms(m["model"], m["short"], device) for _ in range(prefills_per_round)]
            sm["decode_tok_s"].append(_decode_tok_s(m["model"], m["prompt"], m["T"], device))
            sm["decode_cached_tok_s"].append(_decode_cached_tok_s(m["model"], m["prompt"], m["T"], device))
    return {
        "device": str(device), "host": platform.node(), "torch": torch.__version__,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": f"{rounds} interleaved rounds (order rotated), {prefills_per_round} prefills per model per round, "
                    f"decode {DECODE_TOKENS} greedy tokens from a (block_size - {DECODE_TOKENS})-token prompt, "
                    "without and with the KV cache, synchronized timing, warmed up",
        "models": {k: {"block_size": models[k]["T"], **{m: _stats(v) for m, v in samples[k].items()}} for k in names},
    }


def render(b: dict) -> str:
    lines = ["# Inference benchmark", "", f"- device: {b['device']} ({b['host']}, torch {b['torch']})",
             f"- at: {b['at']}", f"- protocol: {b['protocol']}", "",
             "Median (p10–p90). prefill@128 is the same work for every model, so similar values there mean the "
             "comparison is clean.", "",
             "| model | ctx | prefill@ctx ms | prefill@128 ms | decode tok/s, no cache | decode tok/s, KV cache |",
             "|---|---|---|---|---|---|"]
    f = lambda s: f"{s['median']:.2f} ({s['p10']:.2f}–{s['p90']:.2f})"
    for k, m in sorted(b["models"].items(), key=lambda kv: kv[1]["block_size"]):
        cached = f(m["decode_cached_tok_s"]) if "decode_cached_tok_s" in m else "–"
        lines.append(f"| {k} | {m['block_size']} | {f(m['prefill_full_ms'])} | {f(m['prefill_128_ms'])} | "
                     f"{f(m['decode_tok_s'])} | {cached} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark inference of several checkpoints under identical conditions.")
    p.add_argument("checkpoints", nargs="*", help="Default: every checkpoint that has an eval report in evals/.")
    p.add_argument("--device", default=None)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--out-dir", default=str(EVALS_DIR))
    p.add_argument("--reference", default="data160k-bs64-15k-lr1.2e-3-wu256k-v4")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    cks = args.checkpoints or [r["checkpoint"] for r in eval_results(out)]
    b = benchmark(cks, args.device, args.rounds)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference.json").write_text(json.dumps(b, indent=2))
    (out / "inference.md").write_text(render(b))
    print(render(b))
    # Refresh the eval summary so its inference columns use these numbers.
    from mini_llm.evals import load_bench, summary_table
    results = eval_results(out)
    if results:
        (out / "summary.md").write_text(summary_table(results, args.reference, load_bench(out)))


if __name__ == "__main__":
    main()
