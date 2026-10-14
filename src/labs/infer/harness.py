"""The protected measuring side of labs.infer. Runs in a subprocess; candidates never see it.

    python -m labs.infer.harness reference --ckpt C --val V --out ref.json
    python -m labs.infer.harness quality   --ckpt C --val V --ref ref.json --engine cand.py --out q.json
    python -m labs.infer.harness bench     --ckpt C --val V --engine cand.py --against inc.py --rounds R --seed S --out b.json

reference: the fp32 reference model (labs/infer/reference.py) teacher-forced through its KV cache on fixed windows
  of the frozen val tokens: per-prediction loss and argmax, recorded once per checkpoint.
quality: the candidate on the same windows through prefill/step: mean loss vs the reference's, top-1 agreement,
  the worst per-prediction loss difference; generate() checked against the engine's own step() logits (a
  generate that skips work can't pass); peak device memory; wall time of each phase.
bench: candidate and incumbent in one process, warmed up, then R interleaved rounds (order alternating) over fresh
  random val windows each round, one paired candidate/incumbent ratio per scenario per round. Every round also
  re-checks one of the candidate's timed generate() outputs against its own step() logits, untimed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# (prompt tokens, new tokens) per decode scenario; the last one runs past the 1024 window (rolling cache)
SCENARIOS = {"decode_128": (128, 128), "decode_768": (768, 128), "decode_roll": (1024, 64)}
PREFILL = {"prefill_1024": 1024}
# teacher-forced quality windows: (prefill length, single-token steps), 4 inside the window + 1 rolling past it
QUALITY = [(256, 256)] * 4 + [(1024, 128)]
QUALITY_SEED = 1234
GEN_CHECK = (128, 64)  # generate() consistency: prompt length, new tokens
GEN_TOL = 0.05  # a generated token's logit may trail the max by this much (ties under other numerics)


def device() -> torch.device:
    if os.environ.get("LABS_DEVICE"):  # tests run on the CPU
        return torch.device(os.environ["LABS_DEVICE"])
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def peak_bytes(dev: torch.device) -> int:
    return int(torch.mps.driver_allocated_memory()) if dev.type == "mps" else 0


def load_checkpoint(path: str) -> tuple[dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    weights = {k: v.float() for k, v in ck["model_state_dict"].items() if torch.is_tensor(v)}
    return weights, dict(ck["config"])


def load_engine_class(path: str):
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Engine


def windows(val: torch.Tensor, lengths: list[int], rng: random.Random) -> list[torch.Tensor]:
    return [val[(o := rng.randrange(0, val.numel() - n - 1)) : o + n + 1].long() for n in lengths]


class ReferenceEngine:
    """The vendored model behind the Engine API (next_token_logits semantics at the model's own block_size)."""

    def __init__(self, weights: dict, config: dict, dev: torch.device):
        from labs.infer.reference import ModelConfig, ModelCustomTransformer

        cfg = ModelConfig.from_dict(config)
        self.m = ModelCustomTransformer(cfg).to(dev)
        self.m.load_state_dict(weights)
        self.m.eval()
        self.T = cfg.block_size

    @torch.no_grad()
    def prefill(self, ids):
        self.m.clear_cache()
        return self.m(ids, last_only=True, use_cache=True)[0][:, -1, :]

    @torch.no_grad()
    def step(self, token):
        return self.m(token, last_only=True, use_cache=True)[0][:, -1, :]


@torch.no_grad()
def teacher_forced(engine, seqs: list[torch.Tensor], dev) -> tuple[list[float], list[int]]:
    """Per-prediction loss and argmax over every QUALITY window: prefill, then one step per token."""
    losses, argmaxes = [], []
    for seq, (p, s) in zip(seqs, QUALITY):
        seq = seq.to(dev)
        logits = engine.prefill(seq[None, :p])
        for i in range(s + 1):
            lf = logits.float()
            losses.append(F.cross_entropy(lf, seq[p + i : p + i + 1]).item())
            argmaxes.append(int(lf.argmax(-1).item()))
            if i < s:
                logits = engine.step(seq[None, p + i : p + i + 1])
    return losses, argmaxes


def quality_windows(val: torch.Tensor) -> list[torch.Tensor]:
    return windows(val, [p + s for p, s in QUALITY], random.Random(QUALITY_SEED))


@torch.no_grad()
def check_generate(engine, prompt: torch.Tensor, n: int, dev) -> str | None:
    """None if each token generate() returned is (near) the argmax of the engine's own step() logits."""
    out = engine.generate(prompt, n)
    if not (torch.is_tensor(out) and tuple(out.shape) == (1, n)):
        return f"generate returned {type(out).__name__} {tuple(getattr(out, 'shape', ()))}, expected (1, {n})"
    return check_generate_output(engine, prompt, out, dev)


def cmd_reference(a) -> dict:
    dev = device()
    weights, config = load_checkpoint(a.ckpt)
    val = torch.load(a.val, map_location="cpu")
    eng = ReferenceEngine(weights, config, dev)
    t = time.perf_counter()
    losses, argmaxes = teacher_forced(eng, quality_windows(val), dev)
    return {"config": config, "losses": losses, "argmaxes": argmaxes, "mean_loss": statistics.fmean(losses),
            "wall_s": round(time.perf_counter() - t, 2), "torch": torch.__version__, "device": str(dev)}


def cmd_quality(a) -> dict:
    dev = device()
    weights, config = load_checkpoint(a.ckpt)
    val = torch.load(a.val, map_location="cpu")
    ref = json.loads(Path(a.ref).read_text())
    t0 = time.perf_counter()
    eng = load_engine_class(a.engine)(weights, config, dev)
    sync(dev)
    init_s = time.perf_counter() - t0
    t = time.perf_counter()
    losses, argmaxes = teacher_forced(eng, quality_windows(val), dev)
    tf_s = time.perf_counter() - t
    rng = random.Random(QUALITY_SEED + 1)
    gen_err = None
    for w in windows(val, [GEN_CHECK[0]] * 2, rng):
        gen_err = gen_err or check_generate(eng, w[None, : GEN_CHECK[0]].to(dev), GEN_CHECK[1], dev)
    finite = all(math.isfinite(x) for x in losses)
    diffs = [abs(x - y) for x, y in zip(losses, ref["losses"])]
    return {
        "mean_loss": statistics.fmean(losses) if finite else None,
        "ref_mean_loss": ref["mean_loss"],
        "delta_loss": (statistics.fmean(losses) - ref["mean_loss"]) if finite else None,
        "top1_agreement": sum(x == y for x, y in zip(argmaxes, ref["argmaxes"])) / len(argmaxes),
        "max_abs_loss_diff": max(diffs) if finite else None,
        "finite": finite,
        "generate_error": gen_err,
        "peak_bytes": peak_bytes(dev),
        "init_s": round(init_s, 2),
        "teacher_forced_s": round(tf_s, 2),
    }


def _time(dev, fn) -> float:
    sync(dev)
    t = time.perf_counter()
    fn()
    sync(dev)
    return time.perf_counter() - t


def cmd_bench(a) -> dict:
    dev = device()
    weights, config = load_checkpoint(a.ckpt)
    val = torch.load(a.val, map_location="cpu")
    engines = {"cand": load_engine_class(a.engine)(weights, config, dev)}
    if a.against:
        engines["inc"] = load_engine_class(a.against)(weights, config, dev)
    warm = random.Random(a.seed ^ 0x5EED)
    for e in engines.values():  # every shape the timed runs use, twice (lazy compilation, allocator)
        for _ in range(2):
            for name, (p, n) in SCENARIOS.items():
                e.generate(windows(val, [p], warm)[0][None, :p].to(dev), n)
            for name, p in PREFILL.items():
                e.prefill(windows(val, [p], warm)[0][None, :p].to(dev))
    sync(dev)
    rng = random.Random(a.seed)
    names = list(engines)
    raw = {k: {s: [] for s in [*SCENARIOS, *PREFILL]} for k in names}
    errors = []
    for r in range(a.rounds):
        order = names if r % 2 == 0 else names[::-1]
        for s, (p, n) in SCENARIOS.items():
            prompt = windows(val, [p], rng)[0][None, :p].to(dev)
            outs = {}
            for k in order:
                dt = _time(dev, lambda k=k: outs.__setitem__(k, engines[k].generate(prompt, n)))
                raw[k][s].append(n / dt)
            if s == list(SCENARIOS)[r % len(SCENARIOS)]:  # spot-check one timed output per round, untimed
                err = check_generate_output(engines["cand"], prompt, outs["cand"], dev)
                if err:
                    errors.append(f"round {r} {s}: {err}")
        for s, p in PREFILL.items():
            prompt = windows(val, [p], rng)[0][None, :p].to(dev)
            for k in order:
                raw[k][s].append(1.0 / _time(dev, lambda k=k: engines[k].prefill(prompt)))
    out = {"rounds": a.rounds, "seed": a.seed, "errors": errors, "peak_bytes": peak_bytes(dev),
           "speed": {k: {s: statistics.median(v) for s, v in d.items()} for k, d in raw.items()}, "raw": raw}
    if "inc" in engines:
        # paired log ratios: per scenario per round, and the per-round geometric mean over scenarios
        per = {s: [math.log(c / i) for c, i in zip(raw["cand"][s], raw["inc"][s])] for s in raw["cand"]}
        rounds = [statistics.fmean(per[s][r] for s in per) for r in range(a.rounds)]
        out.update(log_ratio={s: statistics.median(v) for s, v in per.items()}, round_log_ratio=rounds,
                   scenario_round_log_ratio=per)
    return out


@torch.no_grad()
def check_generate_output(engine, prompt, out, dev) -> str | None:
    n = out.shape[1] if torch.is_tensor(out) and out.dim() == 2 else -1
    if n <= 0:
        return f"generate returned {type(out).__name__}"
    out = out.to(dev).long()
    logits = engine.prefill(prompt)
    for i in range(n):
        lf = logits.float()[0]
        tok = int(out[0, i])
        if not 0 <= tok < lf.numel() or lf.max().item() - lf[tok].item() > GEN_TOL:
            return f"token {i} = {tok} is not the argmax of the engine's own logits"
        if i + 1 < n:
            logits = engine.step(out[:, i : i + 1])
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["reference", "quality", "bench"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--ref")
    p.add_argument("--engine")
    p.add_argument("--against")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    try:
        res = {"reference": cmd_reference, "quality": cmd_quality, "bench": cmd_bench}[a.mode](a)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the candidate's failure is a result, not a harness crash
        import traceback

        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-3000:]}
    Path(a.out).write_text(json.dumps(res))
    sys.exit(0)


if __name__ == "__main__":
    main()
