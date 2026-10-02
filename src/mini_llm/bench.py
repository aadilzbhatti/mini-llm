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


KV_CONTEXTS = (128, 256, 512, 1024)


def _cache_bytes(model) -> tuple[int, int]:
    """(bytes per cached token, bytes allocated) for the model's KV cache."""
    heads = [h for b in model.blocks for h in b.sa.heads]
    per_token = sum(2 * h.key.out_features for h in heads) * model.token_embedding_table.weight.element_size()
    alloc = sum(t.numel() * t.element_size() for h in heads for t in (h.k_cache, h.v_cache) if t is not None)
    return per_token, alloc


@torch.no_grad()
def kv_reference(checkpoints: list[str], device=None, contexts=KV_CONTEXTS, rounds: int = 7) -> dict:
    """Absolute position embeddings + KV cache: prefill, decode with and without the cache, cache memory,
    at each context length -- the fixed reference for what caching alone bought (e.g. before RoPE).

    decode@c: 64 greedy tokens from a (c - 64)-token prompt, so the context grows to exactly c and stays
    inside the window. "past the window": 64 tokens from a full window, where absolute positions force
    the cache to be refilled every step.
    """
    from mini_llm.report import generate_until_eos
    device = torch.device(device) if device else select_device()
    models = {}
    for ck in checkpoints:
        model, cfg, _ = load_model(ck, device)
        g = torch.Generator().manual_seed(0)
        models[Path(ck).stem] = {"model": model, "cfg": cfg,
                                 "x": torch.randint(0, 50000, (1, cfg.block_size), generator=g).to(device)}

    def prefill_ms(m, c):
        m["model"].set_use_cache(True)
        _sync(device); t = time.perf_counter()
        m["model"](m["x"][:, :c], last_only=True)
        _sync(device); ms = (time.perf_counter() - t) * 1000
        m["model"].set_use_cache(False)
        return ms

    def decode(m, prompt_len, cached):
        model, T = m["model"], m["cfg"].block_size
        model.set_use_cache(cached)
        _sync(device); t = time.perf_counter()
        generate_until_eos(model, m["x"][:, :prompt_len], DECODE_TOKENS, T, greedy=True, eos_token_id=None)
        _sync(device); tok_s = DECODE_TOKENS / (time.perf_counter() - t)
        model.set_use_cache(False)
        return tok_s

    cases = {k: [c for c in contexts if c <= m["cfg"].block_size] for k, m in models.items()}
    for k, m in models.items():  # warmup every shape
        for c in cases[k]:
            prefill_ms(m, c); decode(m, c - DECODE_TOKENS, True); decode(m, c - DECODE_TOKENS, False)
        decode(m, m["cfg"].block_size, True)
    raw = {k: {str(c): {"prefill_ms": [], "decode_tok_s": [], "decode_cached_tok_s": []} for c in cases[k]} | {
        "past_window": {"decode_tok_s": [], "decode_cached_tok_s": []}} for k in models}
    names = list(models)
    for r in range(rounds):
        for k in names[r % len(names):] + names[: r % len(names)]:
            m = models[k]
            for c in cases[k]:
                raw[k][str(c)]["prefill_ms"].append(prefill_ms(m, c))
                raw[k][str(c)]["decode_tok_s"].append(decode(m, c - DECODE_TOKENS, False))
                raw[k][str(c)]["decode_cached_tok_s"].append(decode(m, c - DECODE_TOKENS, True))
            raw[k]["past_window"]["decode_tok_s"].append(decode(m, m["cfg"].block_size, False))
            raw[k]["past_window"]["decode_cached_tok_s"].append(decode(m, m["cfg"].block_size, True))
    out = {}
    for k, m in models.items():
        per_token, alloc = _cache_bytes(m["model"])
        cfg = m["cfg"]
        out[k] = {"n_embd": cfg.n_embd, "n_layer": cfg.n_layer, "n_head": cfg.n_head, "block_size": cfg.block_size,
                  "cache_bytes_per_token": per_token, "cache_bytes_allocated": alloc,
                  "by_context": {c: {**{s: _stats(v) for s, v in d.items()},
                                     **({"cache_bytes_used": per_token * int(c)} if c != "past_window" else {})}
                                 for c, d in raw[k].items()}}
    return {"device": str(device), "host": platform.node(), "torch": torch.__version__,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "protocol": f"absolute position embeddings; {rounds} interleaved rounds (order rotated); prefill = one "
                        f"forward over c tokens filling the cache (last-position logits); decode@c = {DECODE_TOKENS} "
                        f"greedy tokens from a (c - {DECODE_TOKENS})-token prompt, without / with the KV cache; "
                        f"past the window = {DECODE_TOKENS} tokens from a full window (the cache is refilled every step); "
                        "fp32; median (p10-p90)",
            "models": out}


def render_kv_reference(b: dict) -> str:
    f = lambda s: f"{s['median']:.1f} ({s['p10']:.1f}–{s['p90']:.1f})"
    mb = lambda n: f"{n / 2**20:.1f}"
    lines = ["# KV cache reference: absolute position embeddings", "",
             "A fixed record of what the KV cache alone buys with this model's learned absolute position "
             "embeddings, to compare later changes against (e.g. RoPE). Regenerate with "
             "`uv run mini-llm-bench --kv-reference`.", "",
             f"- device: {b['device']} ({b['host']}, torch {b['torch']})", f"- at: {b['at']}", f"- protocol: {b['protocol']}", "",
             "**Past the window**: once the context fills `block_size`, every token's absolute position shifts by one "
             "each step, so the cached keys and values go stale and the cache is refilled from scratch every step. "
             "That row is what relative positions (RoPE) would let the cache avoid.", ""]
    for k, m in b["models"].items():
        lines += [f"## {k}", "",
                  f"d{m['n_embd']} · {m['n_layer']} layers · {m['n_head']} heads · block_size {m['block_size']} · "
                  f"cache {m['cache_bytes_per_token'] / 1024:.0f} KiB per token, "
                  f"{mb(m['cache_bytes_allocated'])} MiB allocated (the full window, reused)", "",
                  "| context | prefill ms | decode tok/s, no cache | decode tok/s, KV cache | speedup | cache in use (MiB) |",
                  "|---|---|---|---|---|---|"]
        for c, d in m["by_context"].items():
            sp = d["decode_cached_tok_s"]["median"] / d["decode_tok_s"]["median"]
            name = "past the window" if c == "past_window" else c
            lines.append(f"| {name} | {f(d['prefill_ms']) if 'prefill_ms' in d else '–'} | {f(d['decode_tok_s'])} | "
                         f"{f(d['decode_cached_tok_s'])} | {sp:.1f}× | "
                         f"{mb(d['cache_bytes_used']) if 'cache_bytes_used' in d else mb(m['cache_bytes_allocated'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


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
    p.add_argument("--kv-reference", action="store_true",
                   help="Write evals/kv_reference.md instead: prefill / decode / cache memory at T=128..1024 "
                        "(default models: the best block_size-1024 checkpoint of each width and depth).")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    if args.kv_reference:
        cks = args.checkpoints
        if not cks:
            best = {}
            for r in eval_results(out):
                c = r["config"]
                if c["block_size"] == max(KV_CONTEXTS):
                    key, val = (c["n_embd"], c["n_layer"]), r["quality"].get(f"full_val@{c['block_size']}", 9e9)
                    if key not in best or val < best[key][0]:
                        best[key] = (val, r["checkpoint"])
            cks = [ck for _, ck in sorted(best.values(), key=lambda v: v[1])]
        b = kv_reference(cks, args.device, rounds=args.rounds)
        (out / "kv_reference.json").write_text(json.dumps(b, indent=2))
        (out / "kv_reference.md").write_text(render_kv_reference(b))
        print(render_kv_reference(b))
        return
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
