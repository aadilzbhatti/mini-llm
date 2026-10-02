"""The evaluation of a trained model: one task, run on the Mac for every model.

    uv run mini-llm-eval checkpoints/a.pt     # evaluate a, then refresh every shared report
    uv run mini-llm-eval --all                # re-evaluate every model that has a report
    uv run mini-llm-eval --all --only samples # just (re)generate samples, keep the rest

Per model -> evals/<model>.json|md: everything below, plus generation samples
(mini_llm.samples). Then, across all evaluated models: the controlled inference
benchmark (evals/inference.md), the summary table (evals/summary.md) and the
side-by-side samples (evals/samples.md). What each eval measures and how to
read it: evals/GUIDE.md.

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
  curve       fixed-target context curve: the same 8,000 target tokens (all at
              least 1,024 tokens into the val stream, identical for every
              model), each predicted from exactly c preceding tokens for c in
              16..1024 (c <= block_size). Only the available history changes,
              so L(c) is a genuine context-scaling curve, unlike full_val@T,
              whose targets sit at positions 0..T-1 and so change with T.
              Reports the paired gain of each doubling of c.
  retrieval   "The secret word is X." + real val text as filler + "The secret
              word is" -> forced choice of X among K single-token candidates
              (chance = 1/K), swept over key-to-question distance. Past the
              model's block_size the key is cropped out, exactly as in
              generation, so accuracy should fall to chance there.
  inference   prefill latency for a full-context prompt, decode tokens/sec
              (this model has no KV cache, so each new token re-runs the
              cropped window), peak memory.

  samples     20 frozen prompts x 5 draws = 100 free-running generations at
              T=0.7 / top-k 40, identical seeds for every model, scored for
              repetition, diversity, exact loops and topic retention
              (mini_llm.samples).

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
DISTANCES = (16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 496, 640, 768, 896, 992)
RETRIEVAL_TRIALS = 400  # per distance; 60 showed the effect, 400 gives ~±5% CIs for comparing models
POSITION_BUCKETS = ((0, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 1024))
EVAL_WINDOWS = (128, 256, 512, 1024)  # full_val and context-benefit windows, each run when <= block_size
CONTEXTS = (16, 32, 64, 128, 256, 512, 1024)  # history lengths for the fixed-target curve
CURVE_TARGETS = 8000
# Tokens per forward pass. Batches are sized by tokens, not sequences: 32 x 1024-token windows
# would need ~13 GB just for the fp32 logits and their log-softmax (vocab 50257).
BATCH_TOKENS = 8192
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


def _rows(T: int) -> int:
    return max(1, BATCH_TOKENS // T)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def position_losses(model, val: torch.Tensor, T: int, device) -> torch.Tensor:
    """Mean loss at each position 0..T-1 over every non-overlapping T-window."""
    n, batch = (val.numel() - 1) // T, _rows(T)
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
        losses, batch = [], _rows(T)
        for s in range(0, len(xs), batch):
            x = torch.stack(xs[s:s + batch]).to(device)
            y = torch.stack(ys[s:s + batch]).to(device)
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
def context_curve(model, val: torch.Tensor, block_size: int, device, n_targets: int = CURVE_TARGETS,
                  seed: int = 0) -> dict:
    """loss(y_p | x_{p-c}..x_{p-1}) for fixed targets p and each history length c <= block_size.

    History is the raw val stream (it can cross a document boundary, as in
    training). Targets are drawn once from positions >= max(CONTEXTS), so every
    c and every model predicts exactly the same tokens.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(max(CONTEXTS), val.numel(), (n_targets,), generator=g)
    per: dict[int, torch.Tensor] = {}
    for c in (c for c in CONTEXTS if c <= block_size):
        losses, batch = [], _rows(c)
        for s in range(0, n_targets, batch):
            p = pos[s:s + batch].tolist()
            x = torch.stack([val[q - c:q] for q in p]).to(device)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits[:, -1].float(), val[p].to(device), reduction="none").cpu())
        per[c] = torch.cat(losses)
    se = lambda t: (t.std() / len(t) ** 0.5).item()
    cs = sorted(per)
    return {
        "targets": n_targets, "seed": seed, "protocol": f"{n_targets} fixed targets at stream positions >= {max(CONTEXTS)}",
        "by_context": {str(c): {"loss": per[c].mean().item(), "se": se(per[c])} for c in cs},
        "gain": {f"{a}->{b}": {"nats": (per[a] - per[b]).mean().item(), "se": se(per[a] - per[b])}
                 for a, b in zip(cs, cs[1:])},
        "per_target": {str(c): [round(v, 4) for v in per[c].tolist()] for c in cs},  # for paired model comparisons
    }


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


def evaluate_checkpoint(path: str | Path, val_path: str | Path = DEFAULT_VAL, device=None, gpu_shared: bool = False,
                        only: set[str] | None = None, previous: dict | None = None) -> dict:
    """All evals for one checkpoint. `only={"samples"}` recomputes just those parts on top of `previous`."""
    from mini_llm.samples import generate_samples

    device = torch.device(device) if device else select_device()
    model, cfg, ckpt = load_model(path, device)
    tokenizer = get_tokenizer()
    if only:
        r = dict(previous or {})
        if "samples" in only:  # reuses draws already in the report, generates the rest
            r["samples"] = generate_samples(model, tokenizer, cfg.block_size, device, previous=r.get("samples"))
        return r
    val = load_tokens(val_path)
    T = cfg.block_size
    quality = {}
    for window in sorted({w for w in (*EVAL_WINDOWS, T) if w <= T}):
        p = position_losses(model, val, window, device)
        quality[f"full_val@{window}"] = p.mean().item()
        quality[f"by_position@{window}"] = {f"{a}-{b - 1}": p[a:b].mean().item()
                                             for a, b in POSITION_BUCKETS if b <= window}
    # cb@128 for every model (a model with a shorter context gets cb@<its context>, labelled as such),
    # then cb@256 / 512 / 1024 for models whose context reaches that far.
    cb = {f"cb@{min(T, 128)}": context_benefit(model, val, min(T, 128), device)}
    for w in EVAL_WINDOWS[1:]:
        if T >= w:
            cb[f"cb@{w}"] = context_benefit(model, val, w, device)
    return {
        "checkpoint": str(path),
        "config": cfg.to_dict(),
        "params": sum(t.numel() for t in dict.fromkeys(model.parameters())),
        "step": ckpt.get("step"),
        "val_tokens": str(val_path),
        "quality": quality,
        "context_benefit": cb,
        "context_curve": context_curve(model, val, T, device),
        "retrieval": retrieval(model, tokenizer, val, T, device),
        "inference": {**(inf := inference_cost(model, T, device)),
                      **({"note": inf["note"] + " -- MEASURED WHILE A TRAINING JOB SHARED THE GPU: timings skewed"}
                         if gpu_shared else {})},
        "training_systems": ckpt.get("systems"),
        "samples": generate_samples(model, tokenizer, T, device),
    }


def _cb_line(cb: dict) -> str:
    return f"{cb['benefit_nats']:.4f} ± {cb['benefit_se']:.4f}"


def render_markdown(r: dict) -> str:
    q, rt, inf, ts = r["quality"], r["retrieval"], r["inference"], r.get("training_systems") or {}
    from mini_llm.samples import label
    lines = [f"# Evals: {label(r)}", "", f"- checkpoint: {Path(r['checkpoint']).stem}",
             f"- config: {r['config']}", f"- params: {r['params']:,}", f"- step: {r['step']}",
             f"- val: {r['val_tokens']}", "", "## Quality", "",
             "full_val@W averages over targets at positions 0..W-1, so it mixes in how much history each "
             "target had. Use the context curve below to compare context lengths.", ""]
    for k, v in q.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: " + ", ".join(f"pos {p}: {x:.4f}" for p, x in v.items()))
        else:
            lines.append(f"- {k}: **{v:.4f}**")
    cc = r.get("context_curve")
    if cc:
        lines += ["", "## Context curve (fixed targets)", "",
                  f"The same {cc['targets']:,} target tokens, each predicted from exactly c preceding tokens "
                  "(identical targets for every model). Gain = paired loss reduction from doubling c.", "",
                  "| history c | loss | ± SE |", "|---|---|---|"]
        lines += [f"| {c} | {v['loss']:.4f} | {v['se']:.4f} |" for c, v in cc["by_context"].items()]
        lines += ["", "| doubling | gain (nats) | ± SE |", "|---|---|---|"]
        lines += [f"| {k} | {v['nats']:+.4f} | {v['se']:.4f} |" for k, v in cc["gain"].items()]
    lines += ["", "## Context benefit", "",
              "Loss on the second half of each val window, given the first half from the same document vs from "
              "a different one (one window per eligible document, identical windows for every model). "
              "Compare cb@W only across models with context ≥ W.", ""]
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
    lines += ["", "## Inference (this eval run; see evals/inference.md for the controlled comparison)", "",
              f"- device: {inf['device']}", f"- prefill, full context: {inf['prefill_ms_full_context']} ms",
              f"- decode: {inf['decode_tokens_per_sec']} tokens/s ({inf['note']})",
              f"- memory: {inf['memory_gb']} GB", ""]
    sm = r.get("samples")
    if sm:
        from mini_llm.samples import _head, _quantile, _summary_cells
        lines += ["## Generation samples", "", f"{sm['protocol']}. The samples themselves: this run's Samples "
                  "button, or evals/samples.md next to every other model.", "",
                  *_head(), f"| {_summary_cells(sm['summary'])} |", "",
                  "| prompt | rep4 median | loops | topic span (median) |", "|---|---|---|---|"]
        for p in sm["prompts"]:
            ds = p["draws"]
            span = _quantile([x["topic_span"] for x in ds if "topic_span" in x], 0.5)
            lines.append(f"| {p['label']} | {_quantile([x['rep4'] for x in ds], 0.5):.2f} | "
                         f"{sum(bool(x.get('looped')) for x in ds)}/{len(ds)} | {'–' if span is None else f'{span:.0f}'} |")
        lines.append("")
    lines += ["## Training systems (recorded by the run)", ""]
    lines += [f"- {k}: {v}" for k, v in ts.items()] if ts else ["- not recorded (run predates mini_llm.systems)"]
    return "\n".join(lines) + "\n"


def paired(ref: dict, other: dict) -> dict:
    """Paired differences other - ref on identical windows / trials / targets."""
    out = {}
    for name, b in other["context_benefit"].items():
        a = ref["context_benefit"].get(name)
        if a and a.get("per_window_benefit") and len(a["per_window_benefit"]) == len(b.get("per_window_benefit", [])):
            d = torch.tensor(b["per_window_benefit"]) - torch.tensor(a["per_window_benefit"])
            out[name] = (d.mean().item(), (d.std() / len(d) ** 0.5).item())
    ca, cb_ = (ref.get("context_curve") or {}).get("per_target", {}), (other.get("context_curve") or {}).get("per_target", {})
    for c in cb_:
        if c in ca and len(ca[c]) == len(cb_[c]):
            d = torch.tensor(cb_[c]) - torch.tensor(ca[c])
            out[f"L@{c}"] = (d.mean().item(), (d.std() / len(d) ** 0.5).item())
    for dist, bv in other["retrieval"]["by_distance"].items():
        av = ref["retrieval"]["by_distance"].get(dist)
        if av and av.get("hits") and bv.get("hits") and len(av["hits"]) == len(bv["hits"]):
            only_b = sum(x == "1" and y == "0" for x, y in zip(bv["hits"], av["hits"]))
            only_a = sum(x == "0" and y == "1" for x, y in zip(bv["hits"], av["hits"]))
            out[f"ret@{dist}"] = (only_b, only_a)  # discordant pairs (McNemar)
    return out


def eval_results(out_dir: Path) -> list[dict]:
    """Per-checkpoint eval reports in out_dir (skips inference.json, samples.json)."""
    rs = [json.loads(f.read_text()) for f in sorted(out_dir.glob("*.json"))]
    return [r for r in rs if isinstance(r, dict) and "quality" in r]


def load_bench(out_dir: Path) -> dict:
    try:
        return json.loads((out_dir / "inference.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def summary_table(results: list[dict], reference: str | None = None, bench: dict | None = None) -> str:
    dists = [str(x) for x in DISTANCES]
    ctxs = [str(c) for c in CONTEXTS]
    cbs = [f"cb@{w}" for w in EVAL_WINDOWS]
    from mini_llm.samples import label
    name = lambda r: Path(r["checkpoint"]).stem
    bench = (bench or {}).get("models", {})
    head = ["model", "ctx", "val@ctx", "rep4 median", "loops", "topic held"] + [f"L(c={c})" for c in ctxs] + cbs + [f"ret@{d}" for d in dists] + \
           ["train tok/s", "train peak GB", "prefill@ctx ms", "decode tok/s", "decode tok/s (KV cache)"]
    lines = ["# Evals summary", "",
             "One row per checkpoint. **L(c)**: loss on the same 8,000 target tokens given exactly c tokens of "
             "history (compare models at equal c; this is the fair context comparison, not val@ctx). "
             "**cb@W**: context benefit in nats ± SE on identical windows, only for models with context ≥ W. "
             "**ret@d**: forced-choice retrieval at distance d (chance 10%), identical trials. "
             "**rep4 median / loops / topic held**: generation samples (see evals/samples.md, evals/GUIDE.md). "
             "Inference columns come from `mini-llm-bench` (all models in one session, interleaved rounds; "
             "median, p10–p90) when available.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        T = r["config"]["block_size"]; q = r["quality"]; ts = r.get("training_systems") or {}
        by = r["retrieval"]["by_distance"]; cb = r["context_benefit"]
        curve = (r.get("context_curve") or {}).get("by_context", {})
        bm = bench.get(name(r))
        ss = (r.get("samples") or {}).get("summary")
        cells = [label(r), str(T), f"{q[f'full_val@{T}']:.4f}" if f"full_val@{T}" in q else "–"]
        cells += ([f"{ss['rep4']:.3f}", f"{ss['looped']}/{ss['n']}", "–" if ss.get("topic") is None else f"{ss['topic']:.0%}"]
                  if ss else ["–", "–", "–"])
        cells += [f"{curve[c]['loss']:.4f}" if c in curve else "–" for c in ctxs]
        cells += [_cb_line(cb[k]) if k in cb and "benefit_se" in cb[k] else "–" for k in cbs]
        cells += [f"{by[d]['accuracy']:.0%}" if d in by else "" for d in dists]
        cells += [f"{ts['train_tokens_per_sec']:,.0f}" if ts.get("train_tokens_per_sec") else "–",
                  f"{ts['peak_mem_gb']:.2f}" if ts.get("peak_mem_gb") else "–"]
        if bm:
            pf, dc, dk = bm["prefill_full_ms"], bm["decode_tok_s"], bm.get("decode_cached_tok_s")
            cells += [f"{pf['median']:.1f} ({pf['p10']:.1f}–{pf['p90']:.1f})", f"{dc['median']:.1f} ({dc['p10']:.1f}–{dc['p90']:.1f})",
                      f"{dk['median']:.1f} ({dk['p10']:.1f}–{dk['p90']:.1f})" if dk else "–"]
        else:
            cells += [f"{r['inference']['prefill_ms_full_context']}*", f"{r['inference']['decode_tokens_per_sec']}*", "–"]
        lines.append("| " + " | ".join(cells) + " |")
    if any(not bench.get(name(r)) for r in results):
        lines += ["", "\\* from the model's own eval run, not the controlled benchmark: don't compare across rows."]
    ref = next((r for r in results if reference and reference in name(r)), None)
    if ref is not None:
        lines += ["", f"## Paired against {label(ref)}", "",
                  "Differences on identical targets / windows / trials, other − reference ± paired SE. "
                  "L(c): negative = this model predicts the same tokens better from the same history. "
                  "ret: trials only this model got right / only the reference got right.", "",
                  "| model | " + " | ".join(f"ΔL(c={c})" for c in ctxs) + " | " + " | ".join(f"Δ{k}" for k in cbs) +
                  " | " + " | ".join(f"ret@{d}" for d in dists) + " |",
                  "|" + "---|" * (1 + len(ctxs) + len(cbs) + len(dists))]
        for r in results:
            if r is ref:
                continue
            pr = paired(ref, r)
            cell = lambda k: f"{pr[k][0]:+.4f} ± {pr[k][1]:.4f}" if k in pr else "–"
            lines.append(f"| {label(r)} | " + " | ".join(cell(f"L@{c}") for c in ctxs) + " | " +
                         " | ".join(cell(k) for k in cbs) + " | " +
                         " | ".join(f"{pr[f'ret@{d}'][0]}/{pr[f'ret@{d}'][1]}" if f"ret@{d}" in pr else "–" for d in dists) + " |")
    return "\n".join(lines) + "\n"


def _other_gpu_jobs() -> bool:
    """Is training, or another eval/benchmark (not this process or its launchers), running?"""
    import os
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", "mini-llm-train|mini-llm-eval|mini_llm[.]evals|mini-llm-bench|mini_llm[.]bench"],
                             capture_output=True, text=True).stdout.split()
    except OSError:
        return False
    mine, pid = set(), os.getpid()
    while pid > 1 and pid not in mine:  # this process and its ancestors (uv run, caffeinate, ...)
        mine.add(pid)
        try:
            pid = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() or 1)
        except (OSError, ValueError):
            break
    others = [int(x) for x in out if int(x) not in mine]
    try:  # launchers like caffeinate aren't always ancestors, but share our process group
        group = os.getpgrp()
        others = [x for x in others if os.getpgid(x) != group]
    except OSError:
        pass
    return bool(others)


def refresh(out: Path, reference: str | None, bench: bool = True, device=None) -> None:
    """Rebuild every cross-model report from the per-model evals: benchmark, summary, samples."""
    from mini_llm.samples import render_comparison
    results = eval_results(out)
    if bench and results:
        if _other_gpu_jobs():
            print("[eval] another GPU job is running: keeping the previous inference benchmark", flush=True)
        else:
            from mini_llm.bench import benchmark, render
            b = benchmark([r["checkpoint"] for r in results], device)
            (out / "inference.json").write_text(json.dumps(b, indent=2))
            (out / "inference.md").write_text(render(b))
            print("[eval] refreshed inference.md", flush=True)
    (out / "summary.md").write_text(summary_table(results, reference, load_bench(out)))
    (out / "samples.md").write_text(render_comparison(results))
    print(f"[eval] refreshed summary.md, samples.md for {len(results)} models", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate checkpoints (quality, context, retrieval, samples), "
                                            "then refresh the cross-model reports.")
    p.add_argument("checkpoints", nargs="*")
    p.add_argument("--all", action="store_true", help="Every checkpoint that already has an eval report.")
    p.add_argument("--only", choices=["samples"], action="append",
                   help="Recompute just these parts, keeping the rest of each existing report.")
    p.add_argument("--no-bench", action="store_true", help="Skip the inference benchmark when refreshing.")
    p.add_argument("--val-tokens", default=str(DEFAULT_VAL))
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(EVALS_DIR))
    p.add_argument("--reference", default="data160k-bs64-15k-lr1.2e-3-wu256k-v4",
                   help="Substring of the checkpoint to pair every other model against in the summary.")
    p.add_argument("--gpu-shared", action="store_true",
                   help="Mark inference timings as skewed (another job was using the GPU).")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cks = list(args.checkpoints) + ([r["checkpoint"] for r in eval_results(out)] if args.all else [])
    for ck in dict.fromkeys(cks):
        stem = Path(ck).stem
        prev_path = out / f"{stem}.json"
        previous = json.loads(prev_path.read_text()) if args.only and prev_path.exists() else None
        if args.only and previous is None:
            print(f"{stem}: no existing report, evaluating everything", flush=True)
        r = evaluate_checkpoint(ck, args.val_tokens, args.device, args.gpu_shared,
                                only=set(args.only) if previous else None, previous=previous)
        prev_path.write_text(json.dumps(r, indent=2))
        (out / f"{stem}.md").write_text(render_markdown(r))
        ss = (r.get("samples") or {}).get("summary") or {}
        print(f"{stem}: val@128 {r['quality'].get('full_val@128', math.nan):.4f}  "
              f"{' '.join(f'{k} {_cb_line(v)}' for k, v in r['context_benefit'].items())}  "
              f"rep4 {ss.get('rep4')} looping {ss.get('looped')}/{ss.get('n')}  -> {out / (stem + '.md')}", flush=True)
    refresh(out, args.reference, bench=not args.no_bench, device=args.device)


if __name__ == "__main__":
    main()
