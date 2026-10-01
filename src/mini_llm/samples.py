"""Generation samples: part of every model's eval, plus the side-by-side comparison report.

Loss, the context curve and retrieval are teacher-forced: every prediction sees
the real preceding tokens. Generation feeds the model its own output back,
hundreds of times, so small errors compound -- a model can improve on every
other eval and still write degenerate text. This is the eval for that.

Per model (stored in its eval JSON under "samples", generated once):
  10 document-like prompts x 5 draws, 256 new tokens, temperature 0.7,
  top-k 40, stop at EOS. Draw j of prompt i uses the same seed for every
  model, so models are compared on identical random draws and nothing is
  cherry-picked. Prompts are beginnings of the web text this base model was
  trained on (report.PROMPTS), not questions: it is not instruction-tuned.

Per sample, three numbers that separate "degenerate" from "coherent but wrong":
  rep4     fraction of 4-grams that repeat an earlier 4-gram (0 = none). High
           means looping or near-looping.
  loop     whether the text ENDS in an exact cycle (the same span of 1-64
           tokens repeated to the end, >= 32 tokens), and the token where the
           cycle starts. "see it as a whole, see it as a whole, ..." is a loop.
  topic    share of the prompt's content words (e.g. "einstein", "physicist")
           still used in the second half of the continuation: is the model
           still writing about the subject it was given?

evals/samples.md lays every evaluated model side by side with full labels
(data / width-depth / params / context / steps).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from mini_llm.data import decode, encode
from mini_llm.report import EOS_TOKEN_ID, PROMPTS, generate_until_eos

PROMPT_LABELS = ("definition", "biography", "science_explainer", "instructional", "bullet_list",
                 "numbered_list", "enumeration", "long_dependency", "attribution", "numeric_units")
DRAWS = 5
NEW_TOKENS = 256
TEMPERATURE, TOP_K = 0.7, 40
BASE_SEED = 20260929
SWEEP: tuple[tuple[float, int | None], ...] = (  # (temperature, top_k); 0 = greedy. Manual: --sweep
    (0.0, None), (0.5, 20), (0.6, 20), (0.6, 40), (0.7, 20), (0.7, 40), (0.7, 50), (0.8, 40), (0.8, 50), (0.9, 50),
)
_STOP = {"that", "this", "with", "from", "have", "were", "will", "there", "their", "which", "about", "these",
         "those", "into", "than", "then", "them", "they", "what", "when", "where", "while", "also", "been",
         "being", "some", "such", "several", "follow", "steps", "main", "types", "according", "although", "process"}


def seed_for(prompt_index: int, draw: int) -> int:
    """Distinct per (prompt, draw) -- one shared seed would give every prompt the same uniform draws."""
    return BASE_SEED + 1000 * prompt_index + draw


def rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def loop_info(ids: list[int], max_period: int = 64, min_len: int = 32) -> dict:
    """Does the sequence end in an exact cycle? -> {looped, period, onset} (earliest-starting cycle)."""
    best = None
    for p in range(1, min(max_period, len(ids) // 2) + 1):
        run = 0
        for i in range(len(ids) - 1, p - 1, -1):
            if ids[i] != ids[i - p]:
                break
            run += 1
        span = run + p  # tokens covered by the cycle, counting its first occurrence
        if span >= max(2 * p, min_len) and (best is None or len(ids) - span < best["onset"]):
            best = {"looped": True, "period": p, "onset": len(ids) - span}
    return best or {"looped": False, "period": None, "onset": None}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def topic_retention(prompt: str, second_half: str) -> float | None:
    """Share of the prompt's content words (matched on a 5-letter stem) used again in the second half."""
    want = {w[:5] for w in content_words(prompt)}
    if not want:
        return None
    have = {w[:5] for w in content_words(second_half)}
    return len(want & have) / len(want)


def complete(model, tokenizer, block_size: int, device, prompt: str, seed: int,
             temperature: float = TEMPERATURE, top_k: int | None = TOP_K, new_tokens: int = NEW_TOKENS) -> dict:
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    kw = {"greedy": True} if temperature == 0 else {"temperature": temperature, "top_k": top_k}
    out, hit_eos = generate_until_eos(model, idx, new_tokens, block_size, eos_token_id=EOS_TOKEN_ID, **kw)
    ids = out[0, idx.size(1):].tolist()
    loop = loop_info(ids)
    return {"text": decode(out[0, idx.size(1):], tokenizer), "tokens": len(ids), "eos": hit_eos,
            "rep4": round(rep4(ids), 3), "looped": loop["looped"], "loop_onset": loop["onset"],
            "loop_period": loop["period"],
            "topic": topic_retention(prompt, decode(torch.tensor(ids[len(ids) // 2:]), tokenizer) if ids else "")}


@torch.no_grad()
def generate_samples(model, tokenizer, block_size: int, device, draws: int = DRAWS) -> dict:
    """The samples section of a model's eval."""
    by_label = dict(PROMPTS)
    prompts = [{"label": lab, "prompt": by_label[lab],
                "draws": [complete(model, tokenizer, block_size, device, by_label[lab], seed_for(i, j))
                          for j in range(draws)]}
               for i, lab in enumerate(PROMPT_LABELS)]
    return {"protocol": f"{len(prompts)} prompts x {draws} draws, {NEW_TOKENS} new tokens, T={TEMPERATURE}, "
                        f"top-k {TOP_K}, stop at EOS, seeds {BASE_SEED} + 1000*prompt + draw",
            "temperature": TEMPERATURE, "top_k": TOP_K, "new_tokens": NEW_TOKENS, "draws": draws,
            "summary": summarize([s for p in prompts for s in p["draws"]]), "prompts": prompts}


def summarize(ss: list[dict]) -> dict:
    topics = [s["topic"] for s in ss if s.get("topic") is not None]
    return {"n": len(ss), "rep4": round(sum(s["rep4"] for s in ss) / len(ss), 3),
            "looped": sum(bool(s.get("looped")) for s in ss), "eos": sum(s["eos"] for s in ss),
            "topic": round(sum(topics) / len(topics), 3) if topics else None,
            "tokens": round(sum(s["tokens"] for s in ss) / len(ss), 1)}


# --- labels and rendering --------------------------------------------------------------

def label(r: dict) -> str:
    """data320k · d512-L4 · 38.9M · T1024 · 80K steps -- what a model IS, not just its context."""
    stem, cfg = Path(r["checkpoint"]).stem, r["config"]
    data = re.search(r"data\d+k", stem)
    steps = r.get("step")
    return " · ".join([data.group(0) if data else "data?", f"d{cfg['n_embd']}-L{cfg['n_layer']}",
                       f"{r['params'] / 1e6:.1f}M", f"T{cfg['block_size']}",
                       f"{steps / 1000:g}K steps" if steps else "steps?"])


def _summary_cells(s: dict) -> str:
    topic = "–" if s["topic"] is None else f"{s['topic']:.0%}"
    return f"{s['rep4']:.3f} | {s['looped']}/{s['n']} | {topic} | {s['eos']}/{s['n']}"


def _sample_block(head: str, prompt: str, s: dict) -> list[str]:
    loop = f" · loops from token {s['loop_onset']} (period {s['loop_period']})" if s.get("looped") else ""
    topic = "" if s.get("topic") is None else f" · topic {s['topic']:.0%}"
    return [f"{head} · {s['tokens']} tokens{' · EOS' if s['eos'] else ''} · rep4 {s['rep4']}{loop}{topic}", "",
            "```", prompt + s["text"], "```", ""]


def render_model(r: dict) -> str:
    """One model's samples (the run's Samples button, and the samples part of its eval report)."""
    sm = r["samples"]
    lines = [f"# Samples: {label(r)}", "", f"- checkpoint: {Path(r['checkpoint']).stem}", f"- {sm['protocol']}", "",
             "| rep4 | looping | topic held | stopped at EOS |", "|---|---|---|---|",
             f"| {_summary_cells(sm['summary'])} |", "",
             "rep4: repeated 4-grams (lower is better). looping: samples that end in an exact cycle. "
             "topic held: share of the prompt's content words still used in the second half. "
             "See evals/GUIDE.md.", ""]
    for p in sm["prompts"]:
        lines += [f"## {p['label']}", "", f"prompt: {p['prompt']!r}", ""]
        for j, s in enumerate(p["draws"]):
            lines += _sample_block(f"**draw {j + 1}**", p["prompt"], s)
    return "\n".join(lines) + "\n"


def render_comparison(results: list[dict]) -> str:
    """evals/samples.md: every evaluated model with samples, side by side on identical draws."""
    rs = [r for r in results if r.get("samples")]
    if not rs:
        return "# Samples\n\nNo model has samples yet: run `mini-llm-eval <checkpoint>`.\n"
    rs.sort(key=lambda r: (r["config"]["block_size"], -r["quality"].get(f"full_val@{r['config']['block_size']}", 0)))
    keys = {id(r): f"M{i + 1}" for i, r in enumerate(rs)}
    sm0 = rs[0]["samples"]
    lines = ["# Samples", "",
             f"Every evaluated model on the same {len(sm0['prompts'])} prompts × {sm0['draws']} draws, "
             f"{sm0['new_tokens']} new tokens, T={sm0['temperature']}, top-k {sm0['top_k']}. Draw j of prompt i uses "
             "the same seed for every model. Base LMs, not instruction-tuned: judge whether the text stays a "
             "coherent document, not whether its facts are right. What each column means: evals/GUIDE.md.", "",
             "## Models", "",
             "| | model | val@ctx | rep4 | looping | topic held | EOS |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        T = r["config"]["block_size"]
        lines.append(f"| {keys[id(r)]} | {label(r)} | {r['quality'].get(f'full_val@{T}', float('nan')):.4f} | "
                     f"{_summary_cells(r['samples']['summary'])} |")
    labels = [p["label"] for p in sm0["prompts"]]
    lines += ["", "## rep4 by prompt", "", "Mean over draws; (n) = draws that end in an exact loop.", "",
              "| prompt | " + " | ".join(keys[id(r)] for r in rs) + " |", "|" + "---|" * (len(rs) + 1)]
    for i, lab in enumerate(labels):
        cells = []
        for r in rs:
            ds = r["samples"]["prompts"][i]["draws"]
            n = sum(bool(s.get("looped")) for s in ds)
            cells.append(f"{sum(s['rep4'] for s in ds) / len(ds):.2f}{f' ({n})' if n else ''}")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    for i, lab in enumerate(labels):
        p0 = sm0["prompts"][i]
        lines += [f"## {lab}", "", f"prompt: {p0['prompt']!r}", ""]
        for j in range(sm0["draws"]):
            lines += [f"### {lab} · draw {j + 1}", ""]
            for r in rs:
                ps = r["samples"]["prompts"]
                if i < len(ps) and j < len(ps[i]["draws"]):
                    lines += _sample_block(f"**{keys[id(r)]} · {label(r)}**", p0["prompt"], ps[i]["draws"][j])
    return "\n".join(lines) + "\n"


# --- manual decoding sweep on one model --------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Decoding sweep (temperature x top-k, plus greedy) on one checkpoint -> evals/sweep.md.

    Model comparisons don't need this: they're part of every eval (mini-llm-eval).
    """
    from mini_llm.data import get_tokenizer
    from mini_llm.device import select_device
    from mini_llm.evals import EVALS_DIR, load_model

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=EVALS_DIR / "sweep.md")
    args = p.parse_args(argv)
    device = torch.device(args.device) if args.device else select_device()
    model, cfg, _ = load_model(args.checkpoint, device)
    tok, by_label = get_tokenizer(), dict(PROMPTS)
    settings = {("greedy" if t == 0 else f"T={t}, k={k}"): (t, k) for t, k in SWEEP}
    lines = [f"# Decoding sweep: {Path(args.checkpoint).stem}", "",
             "Each prompt once (draw 1's seed), every setting. Same model throughout: only decoding varies.", ""]
    rows = {key: [] for key in settings}
    blocks = []
    for i, lab in enumerate(PROMPT_LABELS):
        blocks += [f"## {lab}", ""]
        for key, (t, k) in settings.items():
            s = complete(model, tok, cfg.block_size, device, by_label[lab], seed_for(i, 0), t, k)
            rows[key].append(s)
            blocks += _sample_block(f"**{key}**", by_label[lab], s)
        print(f"[sweep] {lab}", flush=True)
    lines += ["| setting | rep4 | looping | topic held | EOS |", "|---|---|---|---|---|"]
    lines += [f"| {key} | {_summary_cells(summarize(ss))} |" for key, ss in rows.items()]
    args.out.write_text("\n".join(lines + [""] + blocks) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
