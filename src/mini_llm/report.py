"""Fixed-prompt sample report: what the model actually writes, run to run.

A loss number says how surprised the model is; it does not say whether the
model has learned the shape of a sentence in its training distribution. This module generates
from a fixed battery of prompts under fixed decoding settings so that two
checkpoints can be compared by reading them side by side.

Everything that could drift is pinned: the prompts, the token budget, the
seed, and the order the samples are drawn in. The only thing that varies
between two reports is the model.

One section: top-k sampling (temperature 0.8, k=50). Plain sampling
(temperature 1) and greedy/argmax used to follow it; they were dropped because
neither is how anyone would decode from this model -- plain sampling reads
the garbage tail, greedy loops -- so they cost two thirds of the report's time
for text nobody reads. The section seeds its stream once (seed+1, as when it
shared the report with the others, so its text is unchanged) and draws
sequentially, so each prompt consumes a different slice of the random stream.
Seeding per prompt instead would hand every prompt the same uniform draws,
which on a weak model produces near-identical text and looks like a model
pathology rather than the sampling artefact it is. New prompts are only ever
appended, so earlier prompts keep their exact draws.

Cross-model comparisons at the recommended 0.7/40 live in evals/samples.md
(mini_llm.samples); this per-run report keeps 0.8/50 so it stays comparable
with every earlier run's report.

Reports are reproducible on the same device. MPS and CPU do not produce
identical streams from the same seed, so a report regenerated on CPU from a
checkpoint will read differently from the one written at the end of an MPS
training run. Compare like with like.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mini_llm.data import decode, encode

REPORT_SEED = 1234
# Token budget defaults to 2x block_size, so a report always runs exactly one
# full context past the point where the prompt has scrolled out -- half the
# generation conditioned on the prompt, half on the model's own output. At
# block_size 64 this is 128, identical to the previous fixed default, so
# existing reports stay comparable; at 128 it becomes 256 automatically.
REPORT_TOKENS_PER_BLOCK = 2
REPORT_SAMPLES_PER_PROMPT = 2
REPORT_TEMPERATURE = 0.8
REPORT_TOP_K = 50
# GPT-2's <|endoftext|>. Hardcoded because a tokenizer loaded from a bare
# local directory can report eos_token_id = None; prepare_dataset.py writes
# this same id between documents, so it is what the model was trained on.
EOS_TOKEN_ID = 50256

_RETRIEVAL_KEY = "Alice's secret number is 3817."
_RETRIEVAL_QUERY = " Later that day, someone asked Alice what her secret number was. She said it was"
_RETRIEVAL_SHORT = " She wrote it on a small card and put the card in her pocket."
_RETRIEVAL_FILLER_1 = (
    " The library was quiet in the afternoon, and sunlight came through the tall windows onto the long wooden tables."
    " Students read about the water cycle, which describes how water evaporates from oceans and lakes, forms clouds,"
    " and returns to the ground as rain or snow. Plants take up some of this water through their roots, and the rest"
    " flows through rivers back to the sea, where the cycle begins again."
)
_RETRIEVAL_FILLER_2 = (
    " In another part of the room, a teacher explained how bridges are designed. Engineers must consider the weight of"
    " traffic, the strength of the materials, and the effect of wind and temperature. Steel expands when it is warm and"
    " contracts when it is cold, so bridges include small gaps called expansion joints. Without them, the structure could"
    " crack or bend over many years of use."
)

# (label, prompt). Chosen against what is ACTUALLY in the corpus, not what the
# repo name suggests. prepare_dataset.py pulls HuggingFaceTB/smollm-corpus,
# config fineweb-edu-dedup -- filtered educational WEB text, not Wikipedia.
# Measured over 6.9M decoded characters of data10k/train.pt (per 100k chars):
#
#   "is a"/"was a" definitions   31.1      bullet "\n- "            57.5
#   semicolon in prose           24.4      colon list intro ":"     14.7
#   numbered list "\n1. "        12.3      date "in 1919"            9.9
#   quoted span                   8.5      units "12 km"             6.8
#   "according to"                3.6      "known as"                3.1
#   quoted speech + said          0.7      def/function/class        0.1
#   "== Section ==" headings      0.0      "\n\n" paragraph breaks    0.0
#   "[1]" ref markers             0.0      "(born"                   0.0
#
# So: no wiki markup, no paragraph breaks, effectively no code. Lists and
# definitional frames are everywhere. The battery is built accordingly; the
# one code prompt is kept deliberately as an out-of-distribution control.
PROMPTS: list[tuple[str, str]] = [
    # --- core frames, very high frequency in this corpus ---
    ("definition", "Photosynthesis is a process that"),
    ("biography", "Albert Einstein was a German-born theoretical physicist who"),
    ("science_explainer", "Oxygen is a chemical element with"),
    ("instructional", "In this lesson, students will learn how to"),

    # --- structure: lists are the most distinctive formatting the model sees ---
    ("bullet_list", "There are several benefits to regular exercise:\n- "),
    ("numbered_list", "To solve a quadratic equation, follow these steps:\n1."),
    ("enumeration", "There are three main types of"),

    # --- long-range dependency: can the continuation resolve a distant head? ---
    ("long_dependency", "Although the treaty was signed in 1919, it"),
    (
        "agreement_gap",
        "The students who had spent the entire semester preparing for the "
        "final examination in organic chemistry",
    ),

    # --- attribution, quotation, dialogue (present but comparatively rare) ---
    ("attribution", "According to a study published in"),
    ("dialogue", '"I do not think that is correct," she said, "because'),

    # --- facts and numbers ---
    ("factual", "The capital of France is"),
    ("numeric_units", "The mountain rises to a height of"),

    # --- out-of-distribution control: ~0.1 code occurrences per 100k chars, so
    #     this should fail. It is here to show WHETHER it fails, and how. ---
    ("code_ood", "def fibonacci(n):"),

    # --- long-range retrieval: the answer (3817) is only in the first sentence.
    #     Distances from the key to the end of the prompt are 32 / 97 / 171 GPT-2
    #     tokens, so a 128-context model can still see it in the first two and has
    #     lost it in the third, while a 256-context model sees all three. Appended
    #     last so every prompt above keeps its exact sampled draws. ---
    ("retrieval_32", _RETRIEVAL_KEY + _RETRIEVAL_SHORT + _RETRIEVAL_QUERY),
    ("retrieval_97", _RETRIEVAL_KEY + _RETRIEVAL_FILLER_1 + _RETRIEVAL_QUERY),
    ("retrieval_171", _RETRIEVAL_KEY + _RETRIEVAL_FILLER_1 + _RETRIEVAL_FILLER_2 + _RETRIEVAL_QUERY),
]


def context_note(n_prompt: int, block_size: int, max_new_tokens: int) -> str:
    """How long the prompt survives in context.

    generate() crops to the last block_size tokens each step, so after k
    generated tokens the visible window is the last block_size of (prompt + k).
    The prompt therefore starts scrolling out at k = block_size - n_prompt + 1
    and is entirely gone at k = block_size -- independent of prompt length.
    Past that point the model is continuing its own output with no sight of
    what it was asked, which is the first thing to check before reading a
    late-generation collapse as a modelling failure.
    """
    if n_prompt >= block_size:
        return f"{n_prompt} tokens, longer than block_size={block_size}: truncated from the first step"
    first_evicted = block_size - n_prompt + 1
    if max_new_tokens < first_evicted:
        return f"{n_prompt} tokens, fully in context throughout"
    if max_new_tokens < block_size:
        return f"{n_prompt} tokens, starts scrolling out at generated token {first_evicted}"
    return (
        f"{n_prompt} tokens, starts scrolling out at generated token {first_evicted}, "
        f"fully gone by {block_size}"
    )


@torch.no_grad()
def generate_until_eos(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    greedy: bool = False,
    temperature: float | None = None,
    top_k: int | None = None,
    eos_token_id: int | None = EOS_TOKEN_ID,
    top_p: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Generate, stopping as soon as EOS is sampled. Returns (idx, hit_eos).

    Implemented here, NOT in the model: ModelCustomTransformer.generate() is
    the preserved original and runs a fixed token budget with no stop
    condition. The context cropping and the greedy/multinomial arithmetic
    mirror it exactly, so a generation that never emits EOS is identical to
    what generate() would have produced.

    Why stopping matters: EOS is a document boundary. The corpus is an
    EOS-separated stream of documents, so once the model emits it, it has
    said "this document is finished" -- and every token after it is the
    model starting a NEW document from nothing. Continuing past it and then
    judging whether the model held the prompt's subject measures the wrong
    thing entirely. The EOS token itself is not appended to the output.

    With a KV-cached model (use_cache=True) only the newest token is fed once
    the cache holds the context; when the cache fills the window it is
    cleared and the cropped window re-run, so the output is unchanged.
    """
    was_training = model.training
    model.eval()
    use_cache = getattr(model, "use_cache", False)
    if use_cache:
        model.clear_cache()
    try:
        for _ in range(max_new_tokens):
            if use_cache and 0 < model.cache_len() < block_size:
                idx_cond = idx[:, -1:]
            else:
                if use_cache:
                    model.clear_cache()
                idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond, last_only=True)
            logits = logits[:, -1, :]
            if temperature is not None:
                logits = logits / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if top_p is not None and top_p < 1:
                probs = nucleus(probs, top_p)
            nxt = torch.argmax(probs, dim=-1, keepdim=True) if greedy else torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and int(nxt.item()) == eos_token_id:
                return idx, True
            idx = torch.cat((idx, nxt), dim=1)
        return idx, False
    finally:
        if use_cache:
            model.clear_cache()
        model.train(was_training)


def nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of most-likely tokens whose mass reaches top_p (always >= 1 token), renormalised."""
    sorted_p, order = probs.sort(dim=-1, descending=True)
    drop = sorted_p.cumsum(-1) - sorted_p >= top_p  # mass *before* a token already reaches top_p
    sorted_p = sorted_p.masked_fill(drop, 0.0)
    kept = torch.zeros_like(probs).scatter(-1, order, sorted_p)
    return kept / kept.sum(-1, keepdim=True)


def generate_sample(
    model, tokenizer, prompt: str, max_new_tokens: int, block_size: int, device, **kwargs
) -> tuple[str, int, bool]:
    """-> (text, tokens generated, stopped at EOS)"""
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    n_prompt = idx.size(1)
    out, hit_eos = generate_until_eos(model, idx, max_new_tokens, block_size, **kwargs)
    return decode(out[0], tokenizer), out.size(1) - n_prompt, hit_eos


def _footer(n: int, hit_eos: bool, budget: int) -> str:
    if hit_eos:
        return f"[stopped at EOS after {n} of {budget} tokens -- the model ended the document]"
    return f"[{n} tokens, no EOS]"


def sample_report(
    model,
    tokenizer,
    block_size: int,
    device: torch.device | str,
    meta: dict[str, object] | None = None,
    max_new_tokens: int | None = None,
    seed: int = REPORT_SEED,
    samples_per_prompt: int = REPORT_SAMPLES_PER_PROMPT,
    temperature: float = REPORT_TEMPERATURE,
    top_k: int = REPORT_TOP_K,
) -> str:
    """Build the report text. Leaves the model in whatever mode it was in."""
    if max_new_tokens is None:
        max_new_tokens = REPORT_TOKENS_PER_BLOCK * block_size
    n_tokens = {label: encode(prompt, tokenizer).numel() for label, prompt in PROMPTS}

    lines: list[str] = ["# Sample report", ""]
    for key, value in (meta or {}).items():
        lines.append(f"- {key}: {value}")
    lines += [
        f"- max_new_tokens: {max_new_tokens}",
        f"- seed: {seed}",
        f"- block_size: {block_size}",
        f"- device: {device}",
        "",
        f"Generation stops when EOS ({EOS_TOKEN_ID}) is sampled: EOS is a document "
        f"boundary, so text past it would be the model starting a new document.",
        "",
        f"Context note: the window holds {block_size} tokens, so with "
        f"{max_new_tokens} new tokens every prompt has left the window by generated "
        f"token {block_size}; everything after that continues the model's own output only.",
        "",
    ]

    # Its own stream (seed + 1), exactly as when the report had other sections,
    # so its text for a given checkpoint is unchanged.
    lines += [f"## Sampled, temperature {temperature}, top-k {top_k}", ""]
    torch.manual_seed(seed + 1)  # its own stream, independent of the section above
    for label, prompt in PROMPTS:
        lines += [
            f"### {label}",
            "",
            f"prompt: {prompt!r}  [{context_note(n_tokens[label], block_size, max_new_tokens)}]",
            "",
        ]
        for i in range(samples_per_prompt):
            text, n, eos = generate_sample(
                model, tokenizer, prompt, max_new_tokens, block_size, device,
                temperature=temperature, top_k=top_k,
            )
            lines += [f"draw {i + 1}:", "", "```", text, "```", _footer(n, eos, max_new_tokens), ""]

    return "\n".join(lines)


def report_path_for(checkpoint_path: str | Path) -> Path:
    """Report sits next to the checkpoint, same stem: foo.pt -> foo.md"""
    return Path(checkpoint_path).with_suffix(".md")


def write_sample_report(
    model,
    tokenizer,
    block_size: int,
    device: torch.device | str,
    checkpoint_path: str | Path,
    meta: dict[str, object] | None = None,
    **kwargs,
) -> Path:
    path = report_path_for(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_report(model, tokenizer, block_size, device, meta=meta, **kwargs))
    return path


def main(argv: list[str] | None = None) -> None:
    """Regenerate a report from a checkpoint, without retraining.

        python -m mini_llm.report --checkpoint checkpoints/foo.pt
    """
    import argparse

    from mini_llm.config import ModelConfig, build_model
    from mini_llm.data import get_tokenizer
    from mini_llm.device import select_device

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Default: 2 x block_size.")
    p.add_argument("--seed", type=int, default=REPORT_SEED)
    p.add_argument("--device", default=None, help="Override the auto-selected device.")
    args = p.parse_args(argv)

    device = args.device or select_device()
    tokenizer = get_tokenizer()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    meta = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "params": f"{sum(t.numel() for t in dict.fromkeys(model.parameters())):,}",
        "config": cfg.to_dict(),
    }
    path = write_sample_report(
        model, tokenizer, cfg.block_size, device, args.checkpoint,
        meta=meta, max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    print(f"Wrote sample report to {path}")


if __name__ == "__main__":
    main()
