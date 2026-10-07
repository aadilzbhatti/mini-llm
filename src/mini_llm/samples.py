"""Generation samples: part of every model's eval, plus the side-by-side comparison report.

Loss, the context curve and retrieval are teacher-forced: every prediction sees
the real preceding tokens. Generation feeds the model its own output back,
hundreds of times, so small errors compound -- a model can improve on every
other eval and still write degenerate text. This is the eval for that.

Per model (stored in its eval JSON under "samples"):
  20 frozen document-like prompts x 5 draws = 100 generations, 256 new tokens,
  temperature 0.7, top-k 40, stop at EOS. Draw j of prompt i uses the same
  seed for every model, so models are compared on identical random draws and
  nothing is cherry-picked. Prompts are beginnings of the educational web
  text this base model was trained on, not questions: it is not
  instruction-tuned. One sample shows what kinds of failure occur; a hundred
  show whether a model actually got better.

Per sample, scored from its text (so samples are scored identically however
they were produced):
  rep4         fraction of 4-grams that repeat an earlier 4-gram (0 = none)
  distinct-n   unique n-grams / all n-grams, n = 2 and 4 (1 = no repeats)
  loop         whether the text ENDS in an exact cycle (the same span of
               1-64 tokens repeated to the end, >= 32 tokens) and the token
               where the cycle starts
  topic        share of the prompt's content words (e.g. "einstein",
               "physicist") still used in the second half
  topic span   token position of the last mention of any of them: how far the
               model stays on the subject before drifting (0 = never mentioned)

Per model: medians and p90s of these, loop rate with a 95% interval, and the
median token at which loops start.

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

# Frozen: never reorder or edit (seeds are per index). Append only, with a new protocol version.
_BY_LABEL = dict(PROMPTS)
GEN_PROMPTS: tuple[tuple[str, str], ...] = tuple(
    (lab, _BY_LABEL[lab])
    for lab in (
        "definition",
        "biography",
        "science_explainer",
        "instructional",
        "bullet_list",
        "numbered_list",
        "enumeration",
        "long_dependency",
        "attribution",
        "numeric_units",
        "agreement_gap",
    )
) + (
    ("history", "The French Revolution began in 1789, when"),
    ("anatomy", "The human heart is a muscular organ that"),
    ("geography", "The Amazon River flows through"),
    ("math_definition", "In mathematics, a prime number is"),
    ("environment", "Climate change refers to long-term shifts in"),
    ("recipe", "To make bread at home, you will need"),
    ("literature", "William Shakespeare wrote many plays, including"),
    ("technology", "The internet began as a research project in"),
    ("economics", "Inflation occurs when"),
)
PROMPT_LABELS = tuple(lab for lab, _ in GEN_PROMPTS)
DRAWS = 5
NEW_TOKENS = 256
TEMPERATURE, TOP_K = 0.7, 40
BASE_SEED = 20260929
SWEEP: tuple[tuple[float, int | None], ...] = (  # (temperature, top_k); 0 = greedy. Manual: --sweep
    (0.0, None),
    (0.5, 20),
    (0.6, 20),
    (0.6, 40),
    (0.7, 20),
    (0.7, 40),
    (0.7, 50),
    (0.8, 40),
    (0.8, 50),
    (0.9, 50),
)
_STOP = {
    "that",
    "this",
    "with",
    "from",
    "have",
    "were",
    "will",
    "there",
    "their",
    "which",
    "about",
    "these",
    "those",
    "into",
    "than",
    "then",
    "them",
    "they",
    "what",
    "when",
    "where",
    "while",
    "also",
    "been",
    "being",
    "some",
    "such",
    "several",
    "follow",
    "steps",
    "main",
    "types",
    "according",
    "although",
    "process",
}


def seed_for(prompt_index: int, draw: int) -> int:
    """Distinct per (prompt, draw) -- one shared seed would give every prompt the same uniform draws."""
    return BASE_SEED + 1000 * prompt_index + draw


def rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def distinct(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 if not grams else len(set(grams)) / len(grams)


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


def topic_span(prompt: str, ids: list[int], tokenizer) -> int:
    """Token position just past the last mention of any prompt content word (0 if never mentioned)."""
    want = {w[:5] for w in content_words(prompt)}
    text, owner = "", []
    for i, t in enumerate(ids):  # char -> token map from per-token pieces
        piece = tokenizer.decode([t])
        text += piece
        owner += [i] * len(piece)
    last = 0
    for m in re.finditer(r"[A-Za-z]{4,}", text):
        w = m.group(0).lower()
        if w not in _STOP and w[:5] in want:
            last = owner[m.start()] + 1
    return last


def score(prompt: str, text: str, tokenizer) -> dict:
    """Every per-sample metric, from the text alone."""
    ids = encode(text, tokenizer).tolist() if text else []
    loop = loop_info(ids)
    return {
        "rep4": round(rep4(ids), 3),
        "distinct2": round(distinct(ids, 2), 3),
        "distinct4": round(distinct(ids, 4), 3),
        "looped": loop["looped"],
        "loop_onset": loop["onset"],
        "loop_period": loop["period"],
        "topic": topic_retention(prompt, decode(torch.tensor(ids[len(ids) // 2 :]), tokenizer) if ids else ""),
        "topic_span": topic_span(prompt, ids, tokenizer),
    }


def complete(
    model,
    tokenizer,
    block_size: int,
    device,
    prompt: str,
    seed: int,
    temperature: float = TEMPERATURE,
    top_k: int | None = TOP_K,
    new_tokens: int = NEW_TOKENS,
) -> dict:
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    kw = {"greedy": True} if temperature == 0 else {"temperature": temperature, "top_k": top_k}
    out, hit_eos = generate_until_eos(model, idx, new_tokens, block_size, eos_token_id=EOS_TOKEN_ID, **kw)
    text = decode(out[0, idx.size(1) :], tokenizer)
    return {"text": text, "tokens": out.size(1) - idx.size(1), "eos": hit_eos, **score(prompt, text, tokenizer)}


@torch.no_grad()
def generate_samples(
    model, tokenizer, block_size: int, device, draws: int = DRAWS, previous: dict | None = None
) -> dict:
    """The samples section of a model's eval. Draws already in `previous` (same prompt, seed and
    decoding) are reused, so growing the prompt set only generates what's new; every draw is re-scored."""
    return _generate_samples(model, tokenizer, block_size, device, draws, previous)


def _generate_samples(model, tokenizer, block_size: int, device, draws: int, previous: dict | None) -> dict:
    old = {}
    if previous and (previous.get("temperature"), previous.get("top_k"), previous.get("new_tokens")) == (
        TEMPERATURE,
        TOP_K,
        NEW_TOKENS,
    ):
        old = {(p["prompt"], j): d for p in previous.get("prompts", []) for j, d in enumerate(p["draws"])}
    prompts = []
    for i, (lab, prompt) in enumerate(GEN_PROMPTS):
        ds = []
        for j in range(draws):
            d = old.get((prompt, j))
            ds.append(
                {"text": d["text"], "tokens": d["tokens"], "eos": d["eos"], **score(prompt, d["text"], tokenizer)}
                if d
                else complete(model, tokenizer, block_size, device, prompt, seed_for(i, j))
            )
        prompts.append({"label": lab, "prompt": prompt, "draws": ds})
    return {
        "protocol": f"{len(prompts)} prompts x {draws} draws, {NEW_TOKENS} new tokens, T={TEMPERATURE}, "
        f"top-k {TOP_K}, stop at EOS, seeds {BASE_SEED} + 1000*prompt + draw",
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "new_tokens": NEW_TOKENS,
        "draws": draws,
        "summary": summarize([d for p in prompts for d in p["draws"]]),
        "prompts": prompts,
    }


def _quantile(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarize(ss: list[dict]) -> dict:
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    r4 = [s["rep4"] for s in ss]
    topics = [s["topic"] for s in ss if s.get("topic") is not None]
    onsets = [s["loop_onset"] for s in ss if s.get("looped")]
    looped = len(onsets)
    return {
        "n": len(ss),
        "rep4": round(_quantile(r4, 0.5), 3),
        "rep4_p90": round(_quantile(r4, 0.9), 3),
        "rep4_mean": mean(r4),
        "distinct2": mean([s["distinct2"] for s in ss if "distinct2" in s]),
        "distinct4": mean([s["distinct4"] for s in ss if "distinct4" in s]),
        "looped": looped,
        "loop_rate": round(looped / len(ss), 3),
        "loop_ci95": _wilson(looped, len(ss)),
        "loop_onset_median": _quantile(onsets, 0.5),
        "topic": mean(topics),
        "topic_span_median": _quantile([s["topic_span"] for s in ss if "topic_span" in s], 0.5),
        "eos": sum(s["eos"] for s in ss),
        "tokens": mean([s["tokens"] for s in ss]),
    }


# --- labels and rendering --------------------------------------------------------------


def label(r: dict) -> str:
    """data320k · d512-L4 · 38.9M · T1024 · 80K steps -- what a model IS, not just its context."""
    stem, cfg = Path(r["checkpoint"]).stem, r["config"]
    data = re.search(r"data\d+k", stem)
    steps = r.get("step")
    return " · ".join(
        [
            data.group(0) if data else "data?",
            f"d{cfg['n_embd']}-L{cfg['n_layer']}",
            f"{r['params'] / 1e6:.1f}M",
            f"T{cfg['block_size']}",
            f"{steps / 1000:g}K steps" if steps else "steps?",
        ]
    )


SUMMARY_HEAD = "rep4 median / p90 | loops (95% CI) | first loop at | distinct-2 / -4 | topic held | " "topic span | EOS"


def _summary_cells(s: dict) -> str:
    """One row of SUMMARY_HEAD. Old summaries (pre-100-sample) lack some fields: shown as –."""
    f = lambda v, fmt: "–" if v is None else format(v, fmt)
    lo, hi = s.get("loop_ci95") or (None, None)
    loops = f"{s['looped']}/{s['n']}" + (f" ({lo:.0%}–{hi:.0%})" if lo is not None else "")
    onset = s.get("loop_onset_median")
    return " | ".join(
        [
            f"{f(s.get('rep4'), '.3f')} / {f(s.get('rep4_p90'), '.3f')}",
            loops,
            "–" if onset is None else f"token {onset:.0f}",
            f"{f(s.get('distinct2'), '.3f')} / {f(s.get('distinct4'), '.3f')}",
            f(s.get("topic"), ".0%"),
            "–" if s.get("topic_span_median") is None else f"{s['topic_span_median']:.0f} tokens",
            f"{s['eos']}/{s['n']}",
        ]
    )


def _head(*first: str) -> list[str]:
    cols = [*first, *SUMMARY_HEAD.split(" | ")]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def _sample_block(head: str, prompt: str, s: dict) -> list[str]:
    loop = f" · loops from token {s['loop_onset']} (period {s['loop_period']})" if s.get("looped") else ""
    topic = (
        "" if s.get("topic") is None else f" · topic {s['topic']:.0%}, last mention at token {s.get('topic_span', '?')}"
    )
    return [
        f"{head} · {s['tokens']} tokens{' · EOS' if s['eos'] else ''} · rep4 {s['rep4']}{loop}{topic}",
        "",
        "```",
        prompt + s["text"],
        "```",
        "",
    ]


def render_model(r: dict) -> str:
    """One model's samples (the run's Samples button, and the samples part of its eval report)."""
    sm = r["samples"]
    lines = [
        f"# Samples: {label(r)}",
        "",
        f"- checkpoint: {Path(r['checkpoint']).stem}",
        f"- {sm['protocol']}",
        "",
        *_head(),
        f"| {_summary_cells(sm['summary'])} |",
        "",
        "What each column means: evals/GUIDE.md (How to read the evals).",
        "",
    ]
    for p in sm["prompts"]:
        lines += [f"## {p['label']}", "", f"prompt: {p['prompt']!r}", ""]
        for j, s in enumerate(p["draws"]):
            lines += _sample_block(f"**draw {j + 1}**", p["prompt"], s)
    return "\n".join(lines) + "\n"


def render_comparison(results: list[dict]) -> str:
    """evals/samples.md: every model's sample statistics, and a fixed reading set side by side."""
    rs = [r for r in results if r.get("samples")]
    if not rs:
        return "# Samples\n\nNo model has samples yet: run `mini-llm-eval <checkpoint>`.\n"
    rs.sort(key=lambda r: (r["config"]["block_size"], -r["quality"].get(f"full_val@{r['config']['block_size']}", 0)))
    keys = {id(r): f"M{i + 1}" for i, r in enumerate(rs)}
    sm0 = max((r["samples"] for r in rs), key=lambda sm: len(sm["prompts"]))
    lines = [
        "# Samples",
        "",
        f"Every evaluated model on the same {len(sm0['prompts'])} prompts × {sm0['draws']} draws, "
        f"{sm0['new_tokens']} new tokens, T={sm0['temperature']}, top-k {sm0['top_k']}. Draw j of prompt i uses "
        "the same seed for every model. Base LMs, not instruction-tuned: judge whether the text stays a "
        "coherent document, not whether its facts are right. What each column means: evals/GUIDE.md.",
        "",
        "## Models",
        "",
        *_head("", "model", "val@ctx"),
    ]
    for r in rs:
        T = r["config"]["block_size"]
        lines.append(
            f"| {keys[id(r)]} | {label(r)} | {r['quality'].get(f'full_val@{T}', float('nan')):.4f} | "
            f"{_summary_cells(r['samples']['summary'])} |"
        )
    labels = [p["label"] for p in sm0["prompts"]]
    by = lambda r: {p["label"]: p for p in r["samples"]["prompts"]}
    lines += [
        "",
        "## rep4 by prompt",
        "",
        "Median over draws; (n) = draws that end in an exact loop.",
        "",
        "| prompt | " + " | ".join(keys[id(r)] for r in rs) + " |",
        "|" + "---|" * (len(rs) + 1),
    ]
    for lab in labels:
        cells = []
        for r in rs:
            p = by(r).get(lab)
            if not p:
                cells.append("–")
                continue
            n = sum(bool(s.get("looped")) for s in p["draws"])
            cells.append(f"{_quantile([s['rep4'] for s in p['draws']], 0.5):.2f}{f' ({n})' if n else ''}")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Reading set",
        "",
        "A fixed subset to read: draw 1 of every prompt, every model. All "
        f"{sm0['draws']} draws of a model: its run's Samples button.",
        "",
    ]
    for lab in labels:
        p0 = next(p for p in sm0["prompts"] if p["label"] == lab)
        lines += [f"### {lab}", "", f"prompt: {p0['prompt']!r}", ""]
        for r in rs:
            p = by(r).get(lab)
            if p and p["draws"]:
                lines += _sample_block(f"**{keys[id(r)]} · {label(r)}**", p0["prompt"], p["draws"][0])
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
    tok, by_label = get_tokenizer(), dict(GEN_PROMPTS)
    settings = {("greedy" if t == 0 else f"T={t}, k={k}"): (t, k) for t, k in SWEEP}
    lines = [
        f"# Decoding sweep: {Path(args.checkpoint).stem}",
        "",
        "Each prompt once (draw 1's seed), every setting. Same model throughout: only decoding varies.",
        "",
    ]
    rows = {key: [] for key in settings}
    blocks = []
    for i, lab in enumerate(PROMPT_LABELS):
        blocks += [f"## {lab}", ""]
        for key, (t, k) in settings.items():
            s = complete(model, tok, cfg.block_size, device, by_label[lab], seed_for(i, 0), t, k)
            rows[key].append(s)
            blocks += _sample_block(f"**{key}**", by_label[lab], s)
        print(f"[sweep] {lab}", flush=True)
    lines += _head("setting")
    lines += [f"| {key} | {_summary_cells(summarize(ss))} |" for key, ss in rows.items()]
    args.out.write_text("\n".join(lines + [""] + blocks) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
