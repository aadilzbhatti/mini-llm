"""Samples report: the best model at each context length, same prompts, same seeds, same decoding.

Loss, context curves and retrieval are teacher-forced: every prediction sees
the real preceding tokens. Generation feeds the model its own samples back,
hundreds of times, so small errors compound -- a model can improve on every
eval and still write mediocre paragraphs. This report is for reading that.

Two sections, written to evals/samples.md (+ samples.json):

1. Comparison. For every context length the best evaluated checkpoint
   (lowest val@ctx in evals/*.json), 10 document-like prompts x 3 draws,
   256 new tokens, temperature 0.7, top-k 40. Nothing varies but the model:
   draw j of prompt i uses the same seed for every model, so no sample is
   cherry-picked and the models even share their random numbers.
2. Sweep. The best model overall, each prompt once (draw 1's seed), across a
   grid of temperature x top-k settings plus greedy, to show where decoding
   turns from repetitive into incoherent -- one model, so the settings are
   the only thing that varies.

Prompts are beginnings of the kind of text this base model was trained on
(fineweb-edu web pages; see report.PROMPTS), not questions or instructions:
it is not instruction-tuned. Generation stops at EOS, the corpus' document
boundary. rep4 is the fraction of 4-grams in a completion that repeat an
earlier 4-gram (0 = no repetition; greedy loops push it towards 1).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mini_llm.data import decode, encode, get_tokenizer
from mini_llm.device import select_device
from mini_llm.evals import load_model
from mini_llm.report import EOS_TOKEN_ID, PROMPTS, generate_until_eos

PROMPT_LABELS = ("definition", "biography", "science_explainer", "instructional", "bullet_list",
                 "numbered_list", "enumeration", "long_dependency", "attribution", "numeric_units")
DRAWS = 3
NEW_TOKENS = 256
TEMPERATURE, TOP_K = 0.7, 40
SWEEP: tuple[tuple[float, int | None], ...] = (  # (temperature, top_k); temperature 0 = greedy
    (0.0, None), (0.5, 20), (0.6, 20), (0.6, 40), (0.7, 20), (0.7, 40), (0.7, 50), (0.8, 40), (0.8, 50), (0.9, 50),
)
BASE_SEED = 20260929


def seed_for(prompt_index: int, draw: int) -> int:
    """Distinct per (prompt, draw) -- one shared seed would give every prompt the same uniform draws."""
    return BASE_SEED + 1000 * prompt_index + draw


def rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def best_per_context(eval_dir: Path) -> dict[int, dict]:
    """block_size -> the evaluated checkpoint with the lowest val@block_size."""
    best: dict[int, dict] = {}
    for f in sorted(eval_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if "quality" not in r:
            continue  # inference.json, samples.json
        T = r["config"]["block_size"]
        val = r["quality"].get(f"full_val@{T}")
        if val is not None and Path(r["checkpoint"]).is_file() and (T not in best or val < best[T]["val"]):
            best[T] = {"checkpoint": r["checkpoint"], "val": val, "name": Path(r["checkpoint"]).stem}
    return dict(sorted(best.items()))


def complete(model, tokenizer, block_size: int, device, prompt: str, seed: int,
             temperature: float, top_k: int | None, new_tokens: int = NEW_TOKENS) -> dict:
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    kw = {"greedy": True} if temperature == 0 else {"temperature": temperature, "top_k": top_k}
    out, hit_eos = generate_until_eos(model, idx, new_tokens, block_size, eos_token_id=EOS_TOKEN_ID, **kw)
    ids = out[0, idx.size(1):].tolist()
    return {"text": decode(out[0, idx.size(1):], tokenizer), "tokens": len(ids), "eos": hit_eos, "rep4": round(rep4(ids), 3)}


def _setting(t: float, k: int | None) -> str:
    return "greedy" if t == 0 else f"T={t}, k={k}"


def _ctx(name: str, T: int) -> str:
    return f"T{T} ({name})"


def render(res: dict) -> str:
    models, prompts = res["models"], res["prompts"]
    L = ["# Samples", "",
         f"- at: {res['at']} · device: {res['device']} · {res['new_tokens']} new tokens, stop at EOS",
         f"- comparison: best checkpoint per context, {len(prompts)} prompts × {res['draws']} draws, "
         f"T={res['temperature']}, top-k {res['top_k']}; draw j of prompt i uses the same seed for every model",
         f"- sweep: {res['sweep']['model']} (best val overall), each prompt with draw 1's seed",
         "- Base LM, not instruction-tuned: judge whether a continuation is a plausible web page, not whether it answers.",
         "- rep4: fraction of 4-grams that repeat an earlier one (lower = less looping).", "",
         "## Models", "", "| context | checkpoint | val@ctx |", "|---|---|---|"]
    L += [f"| {m['block_size']} | {m['name']} | {m['val']:.4f} |" for m in models]
    L += ["", "Mean over all comparison samples:", "", "| context | tokens | stopped at EOS | rep4 |", "|---|---|---|---|"]
    for m in models:
        ss = [s for p in prompts for s in p["by_model"][m["name"]]]
        L.append(f"| {m['block_size']} | {sum(s['tokens'] for s in ss) / len(ss):.0f} | "
                 f"{sum(s['eos'] for s in ss)}/{len(ss)} | {sum(s['rep4'] for s in ss) / len(ss):.3f} |")
    L += ["", f"## Comparison (T={res['temperature']}, top-k {res['top_k']})", ""]
    for p in prompts:
        L += [f"### {p['label']}", "", f"prompt: {p['prompt']!r}", ""]
        for j in range(res["draws"]):
            L += [f"#### {p['label']} · draw {j + 1} (seed {seed_for(p['index'], j)})", ""]
            for m in models:
                s = p["by_model"][m["name"]][j]
                L += [f"**T{m['block_size']}** · {s['tokens']} tokens{' · EOS' if s['eos'] else ''} · rep4 {s['rep4']}", "",
                      "```", p["prompt"] + s["text"], "```", ""]
    sw = res["sweep"]
    L += [f"## Sweep on {sw['model']}", "", "Mean over prompts:", "",
          "| setting | tokens | stopped at EOS | rep4 |", "|---|---|---|---|"]
    for key in sw["settings"]:
        ss = [p["by_setting"][key] for p in sw["prompts"]]
        L.append(f"| {key} | {sum(s['tokens'] for s in ss) / len(ss):.0f} | {sum(s['eos'] for s in ss)}/{len(ss)} | "
                 f"{sum(s['rep4'] for s in ss) / len(ss):.3f} |")
    L.append("")
    for p in sw["prompts"]:
        L += [f"### sweep · {p['label']}", "", f"prompt: {p['prompt']!r}", ""]
        for key in sw["settings"]:
            s = p["by_setting"][key]
            L += [f"**{key}** · {s['tokens']} tokens{' · EOS' if s['eos'] else ''} · rep4 {s['rep4']}", "",
                  "```", p["prompt"] + s["text"], "```", ""]
    return "\n".join(L)


def run(eval_dir: Path, device=None, new_tokens: int = NEW_TOKENS, draws: int = DRAWS, log=print) -> dict:
    device = torch.device(device) if device else select_device()
    tokenizer = get_tokenizer()
    by_label = dict(PROMPTS)
    prompts = [{"index": i, "label": lab, "prompt": by_label[lab], "by_model": {}} for i, lab in enumerate(PROMPT_LABELS)]
    best = best_per_context(eval_dir)
    if not best:
        raise SystemExit(f"no evaluated checkpoints in {eval_dir}")
    models = [{"block_size": T, **b} for T, b in best.items()]
    for m in models:
        model, cfg, _ = load_model(m["checkpoint"], device)
        log(f"[samples] T{cfg.block_size}: {m['name']}")
        for p in prompts:
            p["by_model"][m["name"]] = [complete(model, tokenizer, cfg.block_size, device, p["prompt"],
                                                 seed_for(p["index"], j), TEMPERATURE, TOP_K, new_tokens)
                                        for j in range(draws)]
        del model
    top = min(models, key=lambda m: m["val"])
    model, cfg, _ = load_model(top["checkpoint"], device)
    log(f"[samples] sweep on {top['name']}")
    settings = {_setting(t, k): (t, k) for t, k in SWEEP}
    sweep = {"model": top["name"], "settings": list(settings), "prompts": [
        {"label": p["label"], "prompt": p["prompt"], "by_setting": {
            key: complete(model, tokenizer, cfg.block_size, device, p["prompt"], seed_for(p["index"], 0), t, k, new_tokens)
            for key, (t, k) in settings.items()}}
        for p in prompts]}
    return {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "device": str(device),
            "new_tokens": new_tokens, "draws": draws, "temperature": TEMPERATURE, "top_k": TOP_K,
            "models": models, "prompts": prompts, "sweep": sweep}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fixed-prompt, fixed-seed samples from the best model at each context length.")
    p.add_argument("--eval-dir", type=Path, default=Path("evals"))
    p.add_argument("--device", default=None)
    p.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    p.add_argument("--draws", type=int, default=DRAWS)
    args = p.parse_args(argv)
    res = run(args.eval_dir, args.device, args.new_tokens, args.draws, log=lambda s: print(s, flush=True))
    (args.eval_dir / "samples.json").write_text(json.dumps(res, indent=2))
    (args.eval_dir / "samples.md").write_text(render(res))
    print(f"-> {args.eval_dir / 'samples.md'}")


if __name__ == "__main__":
    main()
