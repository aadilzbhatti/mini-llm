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
              Fixed protocols so numbers are comparable across models:
              cb@128 (128-token windows, 64-token prefix) for every model;
              cb@256 (256 / 128) only for models with block_size >= 256.
              One window per eligible val document, same windows for every
              model (seeded), with a standard error; per-window values are
              kept so two models can be compared paired.
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
DISTANCES = (16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 496)
RETRIEVAL_TRIALS = 400  # per distance; 60 showed the effect, 400 gives ~±5% CIs for comparing models
POSITION_BUCKETS = ((0, 16), (16, 64), (64, 128), (128, 256), (256, 512))
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
def context_benefit(model, val: torch.Tensor, T: int, device, max_windows: int | None = None, seed: int = 0) -> dict:
    """Loss on window[T/2:] given its real first half vs a first half from another document."""
    half = T // 2
    docs = [d for d in _documents(val) if d.numel() >= T + 1]
    g = torch.Generator().manual_seed(seed)
    real_x, swap_x, ys = [], [], []
    for i in torch.randperm(len(docs), generator=g).tolist()[:max_windows or len(docs)]:
        d = docs[i]
        o = int(torch.randint(0, d.numel() - T, (1,), generator=g))
        w = d[o:o + T + 1]
        other = docs[(i + 1 + int(torch.randint(0, len(docs) - 1, (1,), generator=g))) % len(docs)]
        oo = int(torch.randint(0, other.numel() - half, (1,), generator=g))
        real_x.append(w[:T])
        swap_x.append(torch.cat([other[oo:oo + half], w[half:T]]))
        ys.append(w[1:T + 1])

    def second_half_loss(xs):  # -> per-window mean loss over the second half
        losses = []
        for s in range(0, len(xs), 32):
            x = torch.stack(xs[s:s + 32]).to(device)
            y = torch.stack(ys[s:s + 32]).to(device)
            logits, _ = model(x)
            lp = F.cross_entropy(logits[:, half:].reshape(-1, logits.size(-1)).float(),
                                 y[:, half:].reshape(-1), reduction="none")
            losses.append(lp.view(x.size(0), -1).mean(1).cpu())
        return torch.cat(losses)

    real, swapped = second_half_loss(real_x), second_half_loss(swap_x)
    diff = swapped - real
    return {"protocol": f"window {T}, prefix {half}, seed {seed}", "windows": len(real_x), "prefix_tokens": half,
            "loss_real_prefix": real.mean().item(), "loss_other_doc_prefix": swapped.mean().item(),
            "benefit_nats": diff.mean().item(), "benefit_se": (diff.std() / len(diff) ** 0.5).item(),
            "per_window_benefit": [round(v, 5) for v in diff.tolist()]}


@torch.no_grad()
def retrieval(model, tokenizer, val: torch.Tensor, block_size: int, device,
              distances=DISTANCES, trials: int = RETRIEVAL_TRIALS, seed: int = 0) -> dict:
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
        hits, margins = [], []
        for _ in range(trials):
            answer = int(torch.randint(0, len(cand), (1,), generator=g))
            o = int(torch.randint(0, stream.numel() - n_fill, (1,), generator=g))
            seq = prefix + [int(cand[answer]), 13] + stream[o:o + n_fill].tolist() + prefix  # 13 = "."
            x = torch.tensor(seq[-block_size:], device=device).unsqueeze(0)  # cropped like generation
            logits, _ = model(x)
            scores = logits[0, -1, cand.to(device)].float().log_softmax(-1).cpu()
            hits.append(int(int(scores.argmax()) == answer))
            margins.append((scores[answer] - scores.mean()).item())
        p = sum(hits) / trials
        # Wilson 95% interval: well behaved near 0 and 1, unlike p +/- 1.96 SE.
        z = 1.96; c = (p + z * z / (2 * trials)) / (1 + z * z / trials)
        h = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
        # The key is only usable if its lead-in (" The secret word is") fits too: at
        # dist == block_size the key token is the window's first token and the
        # prefix that makes it retrievable is cropped away.
        out[str(dist)] = {"accuracy": p, "ci95": [round(c - h, 4), round(c + h, 4)],
                          "mean_logprob_margin": sum(margins) / trials,
                          "key_in_context": dist + len(prefix) <= block_size,
                          "hits": "".join(map(str, hits))}  # per trial; same trials for every model (seeded)
    return {"candidates": len(cand), "chance": 1 / len(cand), "trials_per_distance": trials, "by_distance": out}


@torch.no_grad()
def inference_cost(model, block_size: int, device, decode_tokens: int = 64, runs: int = 7) -> dict:
    """Medians over repeated runs: single timings on a shared machine vary by 2x or more."""
    import statistics
    x = torch.randint(0, 50000, (1, block_size), device=device)
    for _ in range(3):  # warmup
        model(x)
    _sync(device)
    prefill = []
    for _ in range(runs * 3):
        t = time.perf_counter(); model(x); _sync(device)
        prefill.append((time.perf_counter() - t) * 1000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decode = []
    for _ in range(runs):
        idx = x[:, : block_size // 2]
        _sync(device); t = time.perf_counter()
        for _ in range(decode_tokens):
            logits, _ = model(idx[:, -block_size:])
            idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)
        _sync(device)
        decode.append(decode_tokens / (time.perf_counter() - t))
    mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else (
        torch.mps.current_allocated_memory() / 2**30 if device.type == "mps" else None)
    return {"device": str(device), "prefill_ms_full_context": round(statistics.median(prefill), 2),
            "prefill_ms_iqr": [round(statistics.quantiles(prefill, n=4)[0], 2), round(statistics.quantiles(prefill, n=4)[2], 2)],
            "decode_tokens_per_sec": round(statistics.median(decode), 1),
            "decode_runs": [round(d, 1) for d in decode],
            "memory_gb": round(mem, 3) if mem is not None else None,
            "note": f"medians of {runs * 3} prefills / {runs} decode runs of {decode_tokens} tokens; no KV cache: "
                    "each decode step re-runs the (cropped) window"}


def evaluate_checkpoint(path: str | Path, val_path: str | Path = DEFAULT_VAL, device=None) -> dict:
    device = torch.device(device) if device else select_device()
    model, cfg, ckpt = load_model(path, device)
    tokenizer = get_tokenizer()
    val = load_tokens(val_path)
    T = cfg.block_size
    quality = {}
    for window in sorted({w for w in (128, 256, T) if w <= T}):
        p = position_losses(model, val, window, device)
        quality[f"full_val@{window}"] = p.mean().item()
        quality[f"by_position@{window}"] = {f"{a}-{b - 1}": p[a:b].mean().item()
                                             for a, b in POSITION_BUCKETS if b <= window}
    # cb@128 for every model (a model with a shorter context gets cb@<its context>, labelled as such).
    cb = {f"cb@{min(T, 128)}": context_benefit(model, val, min(T, 128), device)}
    if T >= 256:
        cb["cb@256"] = context_benefit(model, val, 256, device)
    return {
        "checkpoint": str(path),
        "config": cfg.to_dict(),
        "params": sum(t.numel() for t in dict.fromkeys(model.parameters())),
        "step": ckpt.get("step"),
        "val_tokens": str(val_path),
        "quality": quality,
        "context_benefit": cb,
        "retrieval": retrieval(model, tokenizer, val, T, device),
        "inference": inference_cost(model, T, device),
        "training_systems": ckpt.get("systems"),
    }


def _cb_line(cb: dict) -> str:
    return f"{cb['benefit_nats']:.4f} ± {cb['benefit_se']:.4f}"


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
    lines += ["", "## Context benefit", "",
              "Loss on the second half of each val window, given the first half from the same document vs from "
              "a different one (one window per eligible document, identical windows for every model). "
              "Compare cb@128 across all models; cb@256 only across models with context ≥ 256.", ""]
    for name, cb in r["context_benefit"].items():
        lines += [f"- **{name}** ({cb['protocol']}, {cb['windows']} windows): benefit **{_cb_line(cb)} nats** "
                  f"(same-doc prefix {cb['loss_real_prefix']:.4f}, other-doc prefix {cb['loss_other_doc_prefix']:.4f})"]
    lines += ["", "## Long-range retrieval", "",
              f"Forced choice among {rt['candidates']} single-token words (chance {rt['chance']:.0%}), "
              f"{rt['trials_per_distance']} trials per distance, identical trials for every model.", "",
              "| distance | accuracy | 95% CI | log-prob margin | key in context |", "|---|---|---|---|---|"]
    for d, v in rt["by_distance"].items():
        lo, hi = v.get("ci95", [None, None])
        ci = f"{lo:.0%}–{hi:.0%}" if lo is not None else "–"
        lines.append(f"| {d} | {v['accuracy']:.1%} | {ci} | {v['mean_logprob_margin']:+.2f} | {'yes' if v['key_in_context'] else 'no'} |")
    lines += ["", "## Inference", "", f"- device: {inf['device']}",
              f"- prefill, full context: {inf['prefill_ms_full_context']} ms",
              f"- decode: {inf['decode_tokens_per_sec']} tokens/s ({inf['note']})",
              f"- memory: {inf['memory_gb']} GB", "", "## Training systems (recorded by the run)", ""]
    lines += [f"- {k}: {v}" for k, v in ts.items()] if ts else ["- not recorded (run predates mini_llm.systems)"]
    return "\n".join(lines) + "\n"


def paired(ref: dict, other: dict) -> dict:
    """Paired differences other - ref on identical windows / trials."""
    out = {}
    for name in ("cb@128", "cb@256"):
        a, b = ref["context_benefit"].get(name), other["context_benefit"].get(name)
        if a and b and a.get("per_window_benefit") and len(a["per_window_benefit"]) == len(b["per_window_benefit"]):
            d = torch.tensor(b["per_window_benefit"]) - torch.tensor(a["per_window_benefit"])
            out[name] = (d.mean().item(), (d.std() / len(d) ** 0.5).item())
    for dist, bv in other["retrieval"]["by_distance"].items():
        av = ref["retrieval"]["by_distance"].get(dist)
        if av and av.get("hits") and bv.get("hits") and len(av["hits"]) == len(bv["hits"]):
            only_b = sum(x == "1" and y == "0" for x, y in zip(bv["hits"], av["hits"]))
            only_a = sum(x == "0" and y == "1" for x, y in zip(bv["hits"], av["hits"]))
            out[f"ret@{dist}"] = (only_b, only_a)  # discordant pairs (McNemar)
    return out


def summary_table(results: list[dict], reference: str | None = None) -> str:
    dists = [str(x) for x in DISTANCES]
    name = lambda r: Path(r["checkpoint"]).stem
    head = ["model", "ctx", "val@128", "val@256", "val@ctx", "cb@128", "cb@256"] + [f"ret@{d}" for d in dists] + \
           ["train tok/s", "train peak GB", "prefill ms", "decode tok/s"]
    lines = ["# Evals summary", "",
             "One row per checkpoint. cb = context benefit in nats (± standard error), same windows for every model; "
             "cb@256 only for context ≥ 256. ret = forced-choice retrieval accuracy at that key-to-question distance "
             "(chance 10%), same trials for every model. Inference timings are medians of repeated runs on one device; compare only rows measured in the same session.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        T = r["config"]["block_size"]; q = r["quality"]; ts = r.get("training_systems") or {}
        by = r["retrieval"]["by_distance"]; cb = r["context_benefit"]
        if "cb@128" not in cb:  # report from before the fixed protocols
            cb = {}
        f = lambda k: f"{q[k]:.4f}" if k in q else "–"
        cells = [name(r), str(T), f("full_val@128"), f("full_val@256"), f(f"full_val@{T}"),
                 _cb_line(cb["cb@128"]) if "cb@128" in cb else "–", _cb_line(cb["cb@256"]) if "cb@256" in cb else "–"]
        cells += [f"{by[d]['accuracy']:.0%}" if d in by else "" for d in dists]
        cells += [f"{ts['train_tokens_per_sec']:,.0f}" if ts.get("train_tokens_per_sec") else "–",
                  f"{ts['peak_mem_gb']:.2f}" if ts.get("peak_mem_gb") else "–",
                  f"{r['inference']['prefill_ms_full_context']}", f"{r['inference']['decode_tokens_per_sec']}"]
        lines.append("| " + " | ".join(cells) + " |")
    ref = next((r for r in results if reference and reference in name(r)), None)
    if ref is not None:
        lines += ["", f"## Paired against {name(ref)}", "",
                  "Differences on the identical windows / trials. cb: other − reference, ± paired standard error. "
                  "ret: trials only this model got right / only the reference got right (a large imbalance is a real "
                  "difference; roughly equal counts are noise).", "",
                  "| model | Δcb@128 | Δcb@256 | " + " | ".join(f"ret@{d}" for d in dists) + " |",
                  "|" + "---|" * (3 + len(dists))]
        for r in results:
            if r is ref:
                continue
            pr = paired(ref, r)
            cell = lambda k: f"{pr[k][0]:+.4f} ± {pr[k][1]:.4f}" if k in pr else "–"
            lines.append(f"| {name(r)} | {cell('cb@128')} | {cell('cb@256')} | " +
                         " | ".join(f"{pr[f'ret@{d}'][0]}/{pr[f'ret@{d}'][1]}" if f"ret@{d}" in pr else "–" for d in dists) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quality, context, retrieval and inference evals for checkpoints.")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--val-tokens", default=str(DEFAULT_VAL))
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(EVALS_DIR))
    p.add_argument("--reference", default="data160k-bs64-15k-lr1.2e-3-wu256k-v4",
                   help="Substring of the checkpoint to pair every other model against in the summary.")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoints:
        r = evaluate_checkpoint(ck, args.val_tokens, args.device)
        stem = Path(ck).stem
        (out / f"{stem}.json").write_text(json.dumps(r, indent=2))
        (out / f"{stem}.md").write_text(render_markdown(r))
        print(f"{stem}: val@128 {r['quality'].get('full_val@128', math.nan):.4f}  "
              f"{' '.join(f'{k} {_cb_line(v)}' for k, v in r['context_benefit'].items())}  -> {out / (stem + '.md')}", flush=True)
    results = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    (out / "summary.md").write_text(summary_table(results, args.reference))
    print(f"summary of {len(results)} checkpoints -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
