"""labs.decode: evolve the sampler of a fixed checkpoint so its text degenerates less, without gaming the metric.

One program = one whole sampler.py (API in seed_sampler.py), measured by labs.decode.harness: the owner's 95M model
continues 32-token prompts from val documents for 256 tokens, and every sample is scored with the owner's metrics
(labs.decode.scoring). "Degenerate" is the owner's headline (ends in an exact loop, or rep4 >= 0.5); a sample that
stops at EOS within min_tokens counts as degenerate too, so stopping early can't hide repetition.

The loop is labs.core's; the cascade after the shared static check:
  smoke   16 prompts, the candidate alone: it runs, and the sampler costs <= max_sampler_ms per token
  screen  screen_n fresh prompts, candidate and incumbent on identical draws (common random numbers); on to full
          unless the candidate degenerates more often or a guard fails
  full    full_n more fresh prompts, paired as above; pooled with the screen: accepted iff the paired drop in the
          degenerate rate is >= min_gain and >= accept_z standard errors, and every guard holds:
            plausibility  mean loss of the generated text under an independent judge model (another 95M checkpoint)
                          <= the judge's loss on the documents' real continuations + max_judge_excess:
                          text may get more varied, never less plausible than real text
            topic         mean topic retention >= the incumbent's - max_topic_drop
            speed         sampler <= max_sampler_ms per token
          else a contender if it degenerated less, otherwise evaluated
  holdout (reported, never selects) the owner's 20 prompts x 5 draws for p0 and every accepted program
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

from labs.core import Lab, Program, iso


def degen(s: dict, min_tokens: int) -> bool:
    return bool(s["degenerate"] or s["tokens"] < min_tokens)


class DecodeLab(Lab):
    name = "decode"
    title = "Decoding vs degeneration"
    seed_file = Path(__file__).with_name("seed_sampler.py")
    class_name = "Sampler"
    file_name = "sampler.py"
    harness_module = "labs.decode.harness"
    first_stage = "smoke"
    system = """You design the decoding strategy (the sampler) of a small base language model so that its \
generations stop degenerating into loops and repetition, while staying as plausible and on-topic as real text. You \
get the current sampler.py and its measurements; you return one focused change as SEARCH/REPLACE edits (or a whole \
new file) plus a short rationale. Be concrete about the mechanism, and keep the change small enough that a failure \
teaches something."""
    instructions = {
        "repetition": "Repetition grows from the model's own recent output. Use the generated history to steer away from it.",
        "distribution": "Change how probability mass is truncated or reshaped (temperature, top-k/top-p, typical, min-p, entropy-adaptive...).",
        "adaptive": "Make the strategy react to the state of the generation (confidence, entropy, recent repetition).",
        "open": "Pick the most promising unexplored idea given what has been tried.",
        "refine": "Take the parent's own idea further, or fix what kept its last attempt from paying off.",
    }

    def h(self, mode: str, out: Path, timeout: float, **args) -> dict:
        c = self.cfg()
        return self.harness(mode, out, timeout, gen=self.data.parent / "infer" / c["generator"],
                            judge=self.data / c["judge"], val=self.data.parent / "infer" / c["val"], **args)

    # --- setup ------------------------------------------------------------------------------------------------------

    def init(self, log=print) -> None:
        """p0 (the owner's protocol) is the first incumbent; its holdout is the baseline."""
        p = self.programs().get("p0") or Program("p0", None, self.seed_file.read_text(),
                                                 rationale="Seed: the owner's protocol, temperature 0.7, top-k 40.")
        while p.stage == "static" or p.stage == "smoke":
            p = self.advance(p, log=log)
        if p.status == "rejected":
            raise RuntimeError(f"p0 failed: {p.reason}")
        p.status, p.stage, p.reason = "accepted", "done", "seed"
        self.holdout(p)
        self.save(p)
        self.save_lab({"incumbent": "p0", "history": [{"at": iso(), "program": "p0", "fitness": self.fitness(p)}]})
        self.note("initialized", incumbent="p0", holdout=p.scores.get("holdout"))
        log(f"decode: p0 holdout {p.scores.get('holdout')}")

    def holdout(self, p: Program) -> None:
        o = self.h("owner", self.work() / f"{p.id}-holdout.json", 1800, sampler=self.work() / f"{p.id}.py")
        if o.get("ok"):
            s = o["samples"]
            p.scores["holdout"] = {"degenerate": o["degenerate"], "n": o["n"],
                                   "near_loop": sum(x["near_loop"] for x in s),
                                   "judge_nll": statistics.fmean(x["judge_nll"] for x in s if x["judge_nll"] is not None),
                                   "topic": statistics.fmean(x["topic"] for x in s if x["topic"] is not None)}
        else:
            p.scores["holdout"] = {"error": o.get("error")}

    # --- the cascade after static -----------------------------------------------------------------------------------

    def run_stage(self, p: Program, log=print) -> None:
        c = self.cfg()
        if p.stage == "smoke":
            r = self.h("val", self.work() / f"{p.id}-smoke.json", 900, sampler=self.work() / f"{p.id}.py", n=16, seed=7)
            if not r.get("ok"):
                return self.reject(p, "smoke", f"crashed: {r.get('error')}")
            ms = r["samplers"]["cand"]["sampler_ms_per_token"]
            if ms > c["max_sampler_ms"]:
                return self.reject(p, "smoke", f"sampler takes {ms:.2f} ms per token (max {c['max_sampler_ms']})")
            self.record(p, "smoke", True, f"runs; {ms:.2f} ms per token")
            p.stage = "screen" if p.id != "p0" else "done"
            return
        n = c["screen_n"] if p.stage == "screen" else c["full_n"]
        lab = self.load_lab()
        inc = lab["incumbent"]
        (self.work() / f"{inc}.py").write_text(self.programs()[inc].code)
        r = self.h("val", self.work() / f"{p.id}-{p.stage}.json", c.get("val_timeout_s", 2400),
                   sampler=self.work() / f"{p.id}.py", against=self.work() / f"{inc}.py", n=n,
                   seed=random.SystemRandom().randrange(1, 2**31))
        if not r.get("ok"):
            return self.reject(p, p.stage, f"crashed: {r.get('error')}")
        pool = p.scores.setdefault("pairs", {}).setdefault(inc, {"cand": [], "inc": []})
        for k in ("cand", "inc"):
            pool[k] += [{f: s[f] for f in ("degenerate", "near_loop", "tokens", "judge_nll", "human_nll", "topic", "rep4")}
                        for s in r["samplers"][k]["samples"]]
        p.scores["sampler_ms"] = r["samplers"]["cand"]["sampler_ms_per_token"]
        v = self.verdict(pool, c, p.scores["sampler_ms"])
        p.scores.update(vs=inc, **{k: v[k] for k in ("rate", "inc_rate", "drop", "se", "z", "judge", "human", "topic",
                                                       "inc_topic", "n")})
        detail = (f"degenerate {v['rate']:.1%} vs {inc} {v['inc_rate']:.1%} (drop {v['drop']:+.1%}, z {v['z']:.1f}, "
                  f"n {v['n']}); judge loss {v['judge']:.2f} vs real text {v['human']:.2f}; topic {v['topic']:.3f} "
                  f"vs {v['inc_topic']:.3f}; {p.scores['sampler_ms']:.2f} ms/token")
        if v["guard"]:
            self.record(p, p.stage, False, detail)
            return self.finish(p, "evaluated" if v["drop"] > 0 else "rejected", f"{v['guard']}: {detail}")
        if p.stage == "screen":
            self.record(p, "screen", v["drop"] >= 0, detail)
            if v["drop"] >= 0:
                p.stage = "full"
            else:
                self.finish(p, "evaluated", f"degenerates more: {detail}")
            return
        ok = v["drop"] >= c["min_gain"] and v["z"] >= c["accept_z"]
        self.record(p, "full", ok, detail)
        if ok:
            self.holdout(p)
            ho = p.scores.get("holdout") or {}
            self.accept(p, detail + (f"; holdout {ho['degenerate']}/{ho['n']} degenerate" if "n" in ho else ""), log)
        else:
            self.finish(p, "contender" if v["drop"] > 0 else "evaluated", detail)

    def verdict(self, pool: dict, c: dict, ms: float) -> dict:
        mt = c["min_tokens"]
        cd = [degen(s, mt) for s in pool["cand"]]
        idg = [degen(s, mt) for s in pool["inc"]]
        diffs = [int(b) - int(a) for a, b in zip(cd, idg)]  # +1 where only the incumbent degenerated
        n = len(diffs)
        drop = statistics.fmean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 and len(set(diffs)) > 1 else float("inf")
        mean = lambda xs: statistics.fmean(xs) if xs else float("nan")  # noqa: E731
        judge = mean([s["judge_nll"] for s in pool["cand"] if s["judge_nll"] is not None])
        human = mean([s["human_nll"] for s in pool["cand"] if s["human_nll"] is not None])
        topic = mean([s["topic"] for s in pool["cand"] if s["topic"] is not None])
        inc_topic = mean([s["topic"] for s in pool["inc"] if s["topic"] is not None])
        guard = None
        if judge > human + c["max_judge_excess"]:
            guard = f"less plausible than real text (judge loss {judge:.2f} > {human:.2f} + {c['max_judge_excess']})"
        elif topic < inc_topic - c["max_topic_drop"]:
            guard = f"drifts off topic ({topic:.3f} < {inc_topic:.3f} - {c['max_topic_drop']})"
        elif ms > c["max_sampler_ms"]:
            guard = f"too slow ({ms:.2f} ms per token)"
        return {"rate": mean(cd), "inc_rate": mean(idg), "drop": drop, "se": se, "z": drop / se if se else 0.0,
                "judge": judge, "human": human, "topic": topic, "inc_topic": inc_topic, "n": n, "guard": guard}

    # --- prompt and views -------------------------------------------------------------------------------------------

    def fitness(self, p: Program) -> float | None:
        """Minus the degenerate rate on its own evaluation (p0: its smoke + holdout aren't paired, use the holdout)."""
        if p.scores.get("rate") is not None:
            return -p.scores["rate"]
        ho = p.scores.get("holdout") or {}
        return -ho["degenerate"] / ho["n"] if ho.get("n") else None

    def describe(self, p: Program) -> str:
        s = p.scores
        bits = [f"{p.id} ({p.status})"]
        if s.get("rate") is not None:
            bits.append(f"degenerate {s['rate']:.1%} vs {s['vs']} {s['inc_rate']:.1%}, judge loss {s['judge']:.2f} "
                        f"(real text {s['human']:.2f}), topic {s['topic']:.3f}")
        ho = s.get("holdout") or {}
        if ho.get("n"):
            bits.append(f"holdout {ho['degenerate']}/{ho['n']} degenerate")
        return "; ".join(bits)

    def task(self, c: dict) -> str:
        return f"""## Task
Change sampler.py so a small base language model's generations degenerate (loop, repeat) less often, while the text \
stays as plausible and on-topic as real text.

Model: the owner's 95M GPT-style base model (8 layers, d768, RoPE, context 1024, GPT-2 tokenizer, vocab 50257, EOS \
50256), trained on educational web text; not instruction-tuned. Prompts are the first 32 tokens of real documents; \
each sample is up to 256 new tokens and stops when the sampler returns EOS.

## The contract (the harness is fixed; you only change sampler.py)
A fresh `Sampler(vocab_size, eos_id, prompt_ids)` per sample; `next_token(logits, generated, generator) -> int` once \
per token: logits is the model's raw next-token logits, a (vocab,) float32 CPU tensor; generated is the list of \
tokens sampled so far (prompt excluded); generator is a seeded CPU torch.Generator: use it for all randomness. You \
cannot call the model. No file, OS, network or subprocess access (static check); torch and math are fine.

## How it is judged (paired: candidate and incumbent on the same prompts and random draws)
Degenerate = the sample ends in an exact loop (a span of 1-64 tokens repeating to the end, >= 32 tokens), or its \
rep4 (share of 4-grams repeating an earlier one) is >= 0.5, or it stops before {c['min_tokens']} tokens. Accepted when \
the degenerate rate drops by >= {c['min_gain']:.0%} with z >= {c['accept_z']} over {c['screen_n']} + {c['full_n']} \
prompts, and all guards hold:
- plausibility: the generated text's mean per-token loss under an independent judge model must not exceed the judge's \
loss on the documents' real continuations (+{c['max_judge_excess']}). Today's samples are far more predictable than \
real text (judge loss ~1.4 vs ~3.3 nats), so there is room for variety, but not for noise.
- topic: share of the prompt's content words still used in the second half >= the incumbent's - {c['max_topic_drop']}.
- speed: <= {c['max_sampler_ms']} ms per token of sampler time."""

    def row_summary(self, p: Program) -> str:
        s, bits = p.scores, []
        if s.get("rate") is not None:
            bits.append(f"degenerate {s['rate']:.1%} vs {s['vs']} {s['inc_rate']:.1%} (z {s['z']:.1f}, n {s['n']})")
            bits.append(f"judge {s['judge']:.2f} / real {s['human']:.2f} · topic {s['topic']:.3f}")
        ho = s.get("holdout") or {}
        if ho.get("n"):
            bits.append(f"holdout {ho['degenerate']}/{ho['n']}")
        return " | ".join(bits)

    def tiles(self, progs: dict, st: dict) -> list[dict]:
        inc, p0 = progs.get(st.get("incumbent") or ""), progs.get("p0")
        if not inc:
            return []
        ho, ho0 = inc.scores.get("holdout") or {}, (p0.scores.get("holdout") or {}) if p0 else {}
        out = [{"label": "Incumbent", "value": inc.id, "sub": inc.rationale[:60]}]
        if ho.get("n"):
            out.append({"label": "holdout degenerate", "value": f"{ho['degenerate']}/{ho['n']}",
                        "sub": f"seed {ho0['degenerate']}/{ho0['n']}" if ho0.get("n") else ""})
            out.append({"label": "holdout judge loss", "value": f"{ho['judge_nll']:.2f}",
                        "sub": f"seed {ho0['judge_nll']:.2f}" if ho0.get("n") else ""})
            out.append({"label": "holdout topic", "value": f"{ho['topic']:.3f}",
                        "sub": f"seed {ho0['topic']:.3f}" if ho0.get("n") else ""})
        return out

    def about(self) -> str:
        return ("Evolves the sampler of the same 95M model to cut degenerate generations (your headline metric), "
                "judged paired against the incumbent on fresh val-document prompts with guards against gaming it: an "
                "independent model must find the text no less plausible than real text, and topic retention must hold. "
                "Your 20×5 sample protocol is the holdout.")


LAB = DecodeLab()
