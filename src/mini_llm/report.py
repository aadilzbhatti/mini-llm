"""Fixed-prompt sample report: what the model actually writes, run to run.

A loss number says how surprised the model is; it does not say whether the
model has learned the shape of a sentence in its training distribution. This module generates
from a fixed battery of prompts under fixed decoding settings so that two
checkpoints can be compared by reading them side by side.

Everything that could drift is pinned: the prompts, the token budget, the
seed, and the order the samples are drawn in. The only thing that varies
between two reports is the model.

Greedy decoding comes first and uses no randomness at all -- it is the
model's single most-likely continuation, and the step at which it collapses
into a repetition loop is a cheap progress metric. The sampled section then
seeds once and draws sequentially, so each prompt consumes a different slice
of the random stream. Seeding per prompt instead would hand every prompt the
same uniform draws, which on a weak model produces near-identical text and
looks like a model pathology rather than the sampling artefact it is.

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
from mini_llm.generate import generate_text

REPORT_SEED = 1234
REPORT_MAX_NEW_TOKENS = 128
REPORT_SAMPLES_PER_PROMPT = 2
REPORT_TEMPERATURE = 0.8
REPORT_TOP_K = 50

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
def generate_temp_topk(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Temperature + top-k sampling.

    Implemented here, NOT in the model: ModelCustomTransformer.generate() is
    the preserved original and only does argmax or plain multinomial. This
    mirrors its context cropping exactly (last block_size tokens, logits from
    the final position) and differs only in how the next token is picked --
    divide by temperature, keep the top_k logits, renormalise, sample.
    """
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
    finally:
        model.train(was_training)


def generate_text_temp_topk(
    model, tokenizer, prompt: str, max_new_tokens: int, block_size: int, device,
    temperature: float, top_k: int,
) -> str:
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    out = generate_temp_topk(model, idx, max_new_tokens, block_size, temperature, top_k)
    return decode(out[0], tokenizer)


def sample_report(
    model,
    tokenizer,
    block_size: int,
    device: torch.device | str,
    meta: dict[str, object] | None = None,
    max_new_tokens: int = REPORT_MAX_NEW_TOKENS,
    seed: int = REPORT_SEED,
    samples_per_prompt: int = REPORT_SAMPLES_PER_PROMPT,
    temperature: float = REPORT_TEMPERATURE,
    top_k: int = REPORT_TOP_K,
) -> str:
    """Build the report text. Leaves the model in whatever mode it was in."""
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
        f"Context note: the window holds {block_size} tokens, so with "
        f"{max_new_tokens} new tokens every prompt has left the window by generated "
        f"token {block_size}; everything after that continues the model's own output only.",
        "",
        "## Greedy (deterministic)",
        "",
    ]

    for label, prompt in PROMPTS:
        text = generate_text(model, tokenizer, prompt, max_new_tokens, block_size, device, greedy=True)
        lines += [f"### {label}", "", f"prompt: {prompt!r}  [{context_note(n_tokens[label], block_size, max_new_tokens)}]", "", "```", text, "```", ""]

    lines += ["## Sampled", ""]
    torch.manual_seed(seed)  # once, then draw sequentially -- see module docstring
    for label, prompt in PROMPTS:
        lines += [f"### {label}", "", f"prompt: {prompt!r}  [{context_note(n_tokens[label], block_size, max_new_tokens)}]", ""]
        for i in range(samples_per_prompt):
            text = generate_text(model, tokenizer, prompt, max_new_tokens, block_size, device, greedy=False)
            lines += [f"draw {i + 1}:", "", "```", text, "```", ""]

    # Third regime, added after the first two so their RNG streams are
    # untouched -- greedy and plain sampling are longitudinal benchmarks and
    # must keep producing identical text for a given checkpoint.
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
            text = generate_text_temp_topk(
                model, tokenizer, prompt, max_new_tokens, block_size, device, temperature, top_k
            )
            lines += [f"draw {i + 1}:", "", "```", text, "```", ""]

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
    p.add_argument("--max-new-tokens", type=int, default=REPORT_MAX_NEW_TOKENS)
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
