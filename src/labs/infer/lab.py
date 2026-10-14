"""labs.infer: evolve the inference engine of a fixed checkpoint for speed on this Mac, quality held.

One program = one whole engine.py (API in seed_engine.py), measured by labs.infer.harness against a vendored fp32
reference (reference.py). The loop is labs.core's; this file is the cascade after the shared static check:

  quality  teacher-forced on fixed val windows vs the fp32 reference: delta loss <= max_delta_loss, top-1 agreement
           >= min_top1, generate() consistent with the engine's own logits, peak memory <= max_mem_mult x p0's
  bench    vs the incumbent, interleaved and paired (bench_rounds rounds, order alternating); on to confirm when the
           mean paired log speedup is > 0
  confirm  bench_rounds more rounds on a new seed; accepted iff over all rounds the mean log speedup >= min_gain and
           mean - accept_se x SE (over round pairs) > 0; else a contender (faster, unproven)
  Either bench stage ends a program whose worst scenario falls below 1 - max_scenario_regression of the incumbent's
  (a geometric mean must not trade one scenario away for another).
Speeds are geometric means over the scenarios (harness.SCENARIOS + PREFILL); x_p0 chains each accepted ratio.
"""

from __future__ import annotations

import json
import math
import platform
import random
import statistics
from pathlib import Path

from labs.core import Lab, Program, iso, pair_stats


class InferLab(Lab):
    name = "infer"
    title = "Inference speed"
    seed_file = Path(__file__).with_name("seed_engine.py")
    class_name = "Engine"
    file_name = "engine.py"
    harness_module = "labs.infer.harness"
    first_stage = "quality"
    system = """You optimize the inference engine of a small GPT-style language model for speed on one machine, \
keeping its outputs faithful. You get the current engine.py and its measurements; you return one focused change \
as SEARCH/REPLACE edits (or a whole new file) plus a short rationale. Be concrete about the mechanism you expect \
to save time, and keep the change small enough that a failure teaches something."""
    instructions = {
        "bandwidth": "Decode at batch 1 is bound by reading the weights for every token. Find a way to move fewer bytes.",
        "overhead": "Each decode step launches many small kernels and runs Python per layer. Cut per-token overhead.",
        "prefill": "Look at the prefill path (1024 tokens in one call) and the attention/RoPE implementation.",
        "open": "Pick the most promising unexplored idea given what has been tried.",
        "refine": "Take the parent's own idea further, or fix what kept its last attempt from paying off.",
    }

    def h(self, mode: str, out: Path, timeout: float, **args) -> dict:
        c = self.cfg()
        return self.harness(mode, out, timeout, ckpt=self.data / c["checkpoint"], val=self.data / c["val"], **args)

    # --- setup ---------------------------------------------------------------------------------------------------

    def init(self, log=print) -> None:
        """Reference numbers for the checkpoint, and p0 (the seed engine) measured as the first incumbent."""
        c = self.cfg()
        if not (self.state / "ref.json").exists():
            self.state.mkdir(parents=True, exist_ok=True)
            ref = self.h("reference", self.state / "ref.json", 900)
            if not ref.get("ok"):
                raise RuntimeError(f"reference failed: {ref.get('error')}")
        p = self.programs().get("p0") or Program("p0", None, self.seed_file.read_text(),
                                                 rationale="Seed: the reference model's inference path, unoptimized.")
        while p.stage != "done":
            p = self.advance(p, log=log)  # static + quality (p0 stops after quality)
        if p.status == "rejected":
            raise RuntimeError(f"p0 failed: {p.reason}")
        b = self.h("bench", self.work() / "p0-bench.json", 1800, engine=self.work() / "p0.py",
                   rounds=c["bench_rounds"], seed=1)
        if not b.get("ok"):
            raise RuntimeError(f"p0 bench failed: {b.get('error')}")
        q = p.scores.get("quality") or {}
        p.status, p.reason = "accepted", "seed"
        p.scores.update(speed=b["speed"]["cand"], x_p0=1.0, peak_bytes=max(q.get("peak_bytes", 0), b["peak_bytes"]))
        self.save(p)
        self.save_lab({"incumbent": "p0", "history": [{"at": iso(), "program": "p0", "fitness": 1.0}],
                       "p0_peak": p.scores["peak_bytes"]})
        self.note("initialized", incumbent="p0", speed=p.scores["speed"])
        log(f"infer: p0 measured {p.scores['speed']}")

    # --- the cascade after static ------------------------------------------------------------------------------------

    def run_stage(self, p: Program, log=print) -> None:
        if p.stage == "quality":
            self.quality(p)
        elif p.stage in ("bench", "confirm"):
            self.bench(p, log)

    def quality(self, p: Program) -> None:
        c, lab = self.cfg(), self.load_lab()
        q = self.h("quality", self.work() / f"{p.id}-quality.json", c.get("quality_timeout_s", 900),
                   engine=self.work() / f"{p.id}.py", ref=self.state / "ref.json")
        p.scores["quality"] = {k: v for k, v in q.items() if k != "trace"}
        p0_peak = lab.get("p0_peak")
        if not q.get("ok"):
            self.reject(p, "quality", f"crashed: {q.get('error')}")
        elif not q["finite"]:
            self.reject(p, "quality", "non-finite logits")
        elif q["generate_error"]:
            self.reject(p, "quality", q["generate_error"])
        elif q["delta_loss"] > c["max_delta_loss"]:
            self.reject(p, "quality", f"loss {q['mean_loss']:.5f} is {q['delta_loss']:+.5f} vs the reference "
                                      f"(max +{c['max_delta_loss']})")
        elif q["top1_agreement"] < c["min_top1"]:
            self.reject(p, "quality", f"top-1 agreement {q['top1_agreement']:.4f} < {c['min_top1']}")
        elif p0_peak and q["peak_bytes"] > c["max_mem_mult"] * p0_peak:
            self.reject(p, "quality", f"peak memory {q['peak_bytes'] / 2**20:.0f} MiB > {c['max_mem_mult']}x p0's")
        else:
            self.record(p, "quality", True, f"delta loss {q['delta_loss']:+.5f}, top-1 {q['top1_agreement']:.4f}, "
                        f"peak {q['peak_bytes'] / 2**20:.0f} MiB")
            p.stage = "bench" if p.id != "p0" else "done"

    def bench(self, p: Program, log) -> None:
        c, lab = self.cfg(), self.load_lab()
        progs = self.programs()
        inc = lab["incumbent"]
        (self.work() / f"{inc}.py").write_text(progs[inc].code)
        b = self.h("bench", self.work() / f"{p.id}-{p.stage}.json", c.get("bench_timeout_s", 1800),
                   engine=self.work() / f"{p.id}.py", against=self.work() / f"{inc}.py", rounds=c["bench_rounds"],
                   seed=random.SystemRandom().randrange(1, 2**31))
        if not b.get("ok"):
            return self.reject(p, p.stage, f"crashed: {b.get('error')}")
        if b["errors"]:
            return self.reject(p, p.stage, f"timed generate() output failed the check: {b['errors'][0]}")
        runs = p.scores.setdefault("bench_rounds", {}).setdefault(inc, [])
        runs += b["round_log_ratio"]
        per = p.scores.setdefault("scenario_rounds", {}).setdefault(inc, {})
        for s, v in b["scenario_round_log_ratio"].items():
            per.setdefault(s, []).extend(v)
        worst_s, worst = min(((s, statistics.fmean(v)) for s, v in per.items()), key=lambda x: x[1])
        p.scores.setdefault("bench_speed", []).append(b["speed"]["cand"])
        p.scores["peak_bytes"] = max(p.scores.get("peak_bytes", 0), b["peak_bytes"])
        m, se, n = pair_stats(runs)
        p.scores.update(vs=inc, log_speedup=m, se=se, speedup=math.exp(m),
                        x_p0=progs[inc].scores.get("x_p0", 1.0) * math.exp(m),
                        per_scenario={s: math.exp(statistics.fmean(v)) for s, v in per.items()})
        detail = f"{math.exp(m):.3f}x {inc} (±{se:.3f} log, {n} round pairs); " + ", ".join(
            f"{s} {math.exp(statistics.fmean(v)):.2f}x" for s, v in per.items())
        if worst < math.log(1 - c["max_scenario_regression"]):
            self.record(p, p.stage, False, detail)
            self.finish(p, "evaluated", f"{worst_s} regressed to {math.exp(worst):.2f}x "
                                        f"(floor {1 - c['max_scenario_regression']:.2f}x): {detail}")
        elif p.stage == "bench":
            self.record(p, "bench", m > 0, detail)
            if m > 0:
                p.stage = "confirm"
            else:
                self.finish(p, "evaluated", f"not faster: {detail}")
        else:
            ok = m >= c["min_gain"] and m - c["accept_se"] * se > 0
            self.record(p, "confirm", ok, detail)
            if ok:
                p.scores["speed"] = b["speed"]["cand"]
                self.accept(p, f"{detail}; {p.scores['x_p0']:.2f}x p0", log)
            else:
                self.finish(p, "contender" if m > 0 else "evaluated", detail)

    # --- prompt and views ------------------------------------------------------------------------------------------

    def fitness(self, p: Program) -> float | None:
        return p.scores.get("x_p0")

    def describe(self, p: Program) -> str:
        s = p.scores
        bits = [f"{p.id} ({p.status}{', ' + format(s['x_p0'], '.2f') + 'x p0' if s.get('x_p0') else ''})"]
        if s.get("per_scenario"):
            bits.append("vs " + s.get("vs", "?") + ": " + ", ".join(f"{k} {v:.2f}x" for k, v in s["per_scenario"].items()))
        q = s.get("quality") or {}
        if q.get("delta_loss") is not None:
            bits.append(f"delta loss {q['delta_loss']:+.5f}, top-1 {q['top1_agreement']:.4f}")
        return "; ".join(bits)

    def incumbent_summary(self, p: Program) -> str:
        speeds = p.scores.get("speed") or {}
        return (f"{p.scores.get('x_p0', 1):.2f}x the seed. Absolute speeds: "
                + ", ".join(f"{k} {v:.1f}/s" for k, v in speeds.items()) + " (prefill: calls/s)")

    def task(self, c: dict) -> str:
        import torch

        m = json.loads((self.state / "ref.json").read_text())["config"]
        return f"""## Task
Make engine.py faster on this machine without changing what the model computes (beyond the precision allowed below).

Machine: {platform.machine()} Apple M4 (10-core GPU, 16 GB unified memory, ~120 GB/s), PyTorch {torch.__version__} on MPS.
Model: decoder-only transformer, n_layer {m['n_layer']}, n_embd {m['n_embd']}, n_head {m['n_head']}, vocab 50257, \
context 1024, RoPE, pre-LayerNorm, GELU MLP (4x), tied embedding/lm_head (lm_head also has a bias); ~95M params, fp32 \
checkpoint (~380 MB).

## The contract (the harness is fixed; you only change engine.py)
`Engine(weights, config, device)`: weights is a dict of fp32 CPU tensors (the checkpoint's state dict), config a dict.
`prefill(ids)` -> (1, vocab) logits of the last position; ids (1, T) int64 on the device, 1 <= T <= 1024; resets state.
`step(token)` -> (1, vocab) logits after appending token (1, 1). Past 1024 tokens the context rolls (RoPE ring cache).
`generate(ids, n)` -> (1, n) int64 greedy tokens continuing ids. Each returned token must be (within 0.05 logits) the \
argmax of what prefill/step would give, which the harness checks.
Batch size is always 1. No file, OS, network or subprocess access (static check); torch and math are fine.

## How it is judged
Quality gate, teacher-forced through prefill/step on held-out text vs the fp32 reference: mean loss may rise by at \
most {c['max_delta_loss']} nats, top-1 agreement >= {c['min_top1']}; peak memory <= {c['max_mem_mult']}x the seed's. \
Lower precision (fp16/bf16, also for the cache) is allowed if it passes this gate.
Speed: end-to-end generate() tokens/s at decode_128 (128-token prompt, 128 new), decode_768 (768, 128), decode_roll \
(1024-token prompt, 64 new, past the window), plus prefill_1024 (one 1024-token prefill). Fitness is the geometric \
mean of the four, measured interleaved against the incumbent; accepted at >= {math.exp(c['min_gain']):.2f}x and \
clear of the noise (~1% per round pair), and no single scenario may fall below \
{1 - c['max_scenario_regression']:.2f}x the incumbent's."""

    def row_summary(self, p: Program) -> str:
        s, q = p.scores, p.scores.get("quality") or {}
        bits = []
        if s.get("speedup") and s.get("vs"):
            bits.append(f"{s['speedup']:.3f}x {s['vs']}")
        if s.get("per_scenario"):
            bits.append(" · ".join(f"{k.replace('decode_', 'd').replace('prefill_', 'p')} {v:.2f}"
                                   for k, v in s["per_scenario"].items()))
        if q.get("delta_loss") is not None:
            bits.append(f"Δloss {q['delta_loss']:+.4f} · top-1 {q['top1_agreement']:.3f}")
        return " | ".join(bits)

    def tiles(self, progs: dict, st: dict) -> list[dict]:
        inc, p0 = progs.get(st.get("incumbent") or ""), progs.get("p0")
        if not inc:
            return []
        out = [{"label": "Incumbent", "value": inc.id, "sub": f"{inc.scores.get('x_p0', 1):.2f}× the seed engine"}]
        fmt = lambda v: f"{v:.0f}/s" if v >= 100 else f"{1000 / v:.1f} ms"  # noqa: E731
        for k, v in (inc.scores.get("speed") or {}).items():
            seed = (p0.scores.get("speed") or {}).get(k) if p0 else None
            out.append({"label": k.replace("_", " "), "value": fmt(v), "sub": f"seed {fmt(seed)}" if seed else ""})
        return out

    def about(self) -> str:
        return (f"Evolves the inference engine (engine.py) of a fixed checkpoint ({self.cfg().get('checkpoint')}) for "
                "speed: a static check and a quality gate (teacher-forced loss vs the fp32 reference) screen each "
                "proposal, then it is benchmarked interleaved against the incumbent and accepted when it is clearly "
                "faster with no scenario regressing.")


LAB = InferLab()
