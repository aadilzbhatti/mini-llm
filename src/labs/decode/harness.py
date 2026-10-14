"""The protected measuring side of labs.decode. Runs in a subprocess; candidates never see it.

    python -m labs.decode.harness val   --gen G --judge J --val V --sampler cand.py [--against inc.py] --n N --seed S --out o.json
    python -m labs.decode.harness owner --gen G --judge J --val V --sampler cand.py --out o.json

val: N documents of the frozen val tokens (fresh per seed), each prompted with its first PROMPT_TOKENS tokens; every
  sampler continues every prompt for up to NEW_TOKENS tokens (stop at EOS), batched through the fp32 generator model.
  Row r of every sampler gets a CPU generator seeded (seed, r): common random numbers, so candidate and incumbent are
  compared on identical draws. Each sample is scored with the owner's metrics (labs.decode.scoring), plus its mean
  per-token loss under an independent judge model (a different 95M checkpoint), next to the judge's loss on the
  document's real continuation of the same length: generated text must stay as plausible as real text.
owner: the owner's 20 prompts x 5 draws (seeds BASE_SEED + 1000 * prompt + draw), the holdout; no human reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from labs.decode import scoring
from labs.infer.harness import device, load_checkpoint, sync

PROMPT_TOKENS = 32
NEW_TOKENS = scoring.NEW_TOKENS
EOS = 50256
BATCH = 50


def load_sampler(path: str):
    spec = importlib.util.spec_from_file_location(f"samp_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Sampler


def load_model(path: str, dev):
    from labs.infer.reference import ModelConfig, ModelCustomTransformer

    weights, config = load_checkpoint(path)
    m = ModelCustomTransformer(ModelConfig.from_dict(config)).to(dev)
    m.load_state_dict(weights)
    return m.eval()


def documents(val: torch.Tensor, n: int, seed: int) -> list[torch.Tensor]:
    """n val documents with at least PROMPT_TOKENS + NEW_TOKENS tokens, chosen by seed."""
    ends = (val == EOS).nonzero().flatten().tolist()
    starts = [0] + [e + 1 for e in ends]
    docs = [val[a:b] for a, b in zip(starts, ends + [val.numel()]) if b - a >= PROMPT_TOKENS + NEW_TOKENS]
    return random.Random(seed).sample(docs, min(n, len(docs)))


@torch.no_grad()
def generate(model, Sampler, prompts: list[list[int]], seeds: list[int], dev) -> tuple[list[list[int]], float]:
    """Continue equal-length prompts as one batch; -> (generated ids per row, sampler seconds)."""
    B = len(prompts)
    samplers = [Sampler(model.lm_head.out_features, EOS, list(p)) for p in prompts]
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    out: list[list[int]] = [[] for _ in range(B)]
    alive = [True] * B
    model.clear_cache()
    logits = model(torch.tensor(prompts, device=dev), last_only=True, use_cache=True)[0][:, -1, :]
    t_samp = 0.0
    for _ in range(NEW_TOKENS):
        cpu = logits.float().cpu()
        nxt = []
        t = time.perf_counter()
        for r in range(B):
            if not alive[r]:
                nxt.append(EOS)
                continue
            tok = samplers[r].next_token(cpu[r].clone(), list(out[r]), gens[r])
            if not isinstance(tok, int) or not 0 <= tok < cpu.size(1):
                raise ValueError(f"next_token returned {tok!r}, not a token id")
            if tok == EOS:
                alive[r] = False
            else:
                out[r].append(tok)
            nxt.append(tok)
        t_samp += time.perf_counter() - t
        if not any(alive):
            break
        logits = model(torch.tensor(nxt, device=dev)[:, None], last_only=True, use_cache=True)[0][:, -1, :]
    model.clear_cache()
    return out, t_samp


@torch.no_grad()
def judge_nll(judge, rows: list[tuple[list[int], list[int]]], dev) -> list[float | None]:
    """Mean per-token loss of each continuation given its prompt, under the judge (None if empty)."""
    res = []
    for prompt, cont in rows:
        if not cont:
            res.append(None)
            continue
        x = torch.tensor([prompt + cont], device=dev)
        logits = judge(x[:, :-1])[0][0]
        lp = F.cross_entropy(logits[len(prompt) - 1 :].float(), x[0, len(prompt) :], reduction="mean")
        res.append(float(lp))
    return res


def run_val(a) -> dict:
    from mini_llm.data import get_tokenizer

    dev, tok = device(), get_tokenizer()
    gen, judge = load_model(a.gen, dev), load_model(a.judge, dev)
    docs = documents(torch.load(a.val, map_location="cpu"), a.n, a.seed)
    prompts = [d[:PROMPT_TOKENS].tolist() for d in docs]
    real = [d[PROMPT_TOKENS : PROMPT_TOKENS + NEW_TOKENS].tolist() for d in docs]
    seeds = [a.seed * 100_003 + r for r in range(len(docs))]
    samplers = {"cand": load_sampler(a.sampler)} | ({"inc": load_sampler(a.against)} if a.against else {})
    out = {"n": len(docs), "seed": a.seed, "samplers": {}}
    for name, S in samplers.items():
        ids, t_samp, t0 = [], 0.0, time.perf_counter()
        for b in range(0, len(docs), BATCH):
            o, ts = generate(gen, S, prompts[b : b + BATCH], seeds[b : b + BATCH], dev)
            ids += o
            t_samp += ts
        sync(dev)
        wall = time.perf_counter() - t0
        nll = judge_nll(judge, list(zip(prompts, ids)), dev)
        human = judge_nll(judge, [(p, r[: max(len(g), 1)]) for p, r, g in zip(prompts, real, ids)], dev)
        samples = []
        for p, g, jl, hl in zip(prompts, ids, nll, human):
            sc = scoring.score_ids(tok.decode(p), g, tok)
            sc.pop("text")
            samples.append({**sc, "tokens": len(g), "judge_nll": jl, "human_nll": hl})
        n_tok = sum(len(g) for g in ids)
        out["samplers"][name] = {"samples": samples, "wall_s": round(wall, 1),
                                 "sampler_ms_per_token": round(1000 * t_samp / max(n_tok, 1), 3)}
    return out


def run_owner(a) -> dict:
    from mini_llm.data import get_tokenizer

    dev, tok = device(), get_tokenizer()
    gen, judge = load_model(a.gen, dev), load_model(a.judge, dev)
    S = load_sampler(a.sampler)
    samples = []
    for i, (label, prompt) in enumerate(scoring.GEN_PROMPTS):
        p = tok.encode(prompt)
        o, _ = generate(gen, S, [p] * scoring.DRAWS, [scoring.seed_for(i, j) for j in range(scoring.DRAWS)], dev)
        nll = judge_nll(judge, [(p, g) for g in o], dev)
        for j, (g, jl) in enumerate(zip(o, nll)):
            sc = scoring.score_ids(prompt, g, tok)
            samples.append({"label": label, "draw": j, **sc, "tokens": len(g), "judge_nll": jl})
    return {"samples": samples, "degenerate": sum(s["degenerate"] for s in samples), "n": len(samples)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["val", "owner"])
    for k in ("gen", "judge", "val", "sampler", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--against")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    try:
        res = {"val": run_val, "owner": run_owner}[a.mode](a)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the candidate's failure is a result
        import traceback

        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-3000:]}
    Path(a.out).write_text(json.dumps(res))
    sys.exit(0)


if __name__ == "__main__":
    main()
