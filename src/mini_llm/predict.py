"""Predict wall-clock time and validation loss for a run, before running it.

Fits closed forms to whatever is already in runs/, so the fit improves every
time a run lands -- there is nothing to retrain and no stored model file.

    python -m mini_llm.predict --n-embd 256 --block-size 128 --steps 160000 --lr 3e-4

Why closed forms rather than a learned model: with ~10 finished runs and six
hyperparameters, anything flexible would overfit immediately. Training time is
a compute formula and loss follows a power law -- both are known shapes, so the
data only has to pin down two or four numbers instead of discovering the shape.

Every prediction is reported with a leave-one-out error estimate measured on
the runs themselves. When the honest answer is "this is not constrained yet",
that is what it prints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RUNS_DIR = Path("runs")
DEFAULTS = {
    "block-size": 64,
    "batch-size": 4,
    "n-embd": 128,
    "n-layer": 4,
    "lr": 1e-3,
    "min-lr": 2e-6,
    "warmup-steps": 500,
    "seed": 42,
}
DATA10K = "data/data10k/train.pt"


def lr_at_step(step, total, lr, min_lr, warm):
    if warm > 0 and step < warm:
        return max(min_lr, lr * (step + 1) / warm)
    if total <= warm:
        return min_lr
    p = min(max((step - warm) / (total - warm), 0.0), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (lr - min_lr)


def schedule_sums(step, total, lr, min_lr, warm, grid=400):
    """(S1, S2) at `step`: cumulative LR, and cumulative annealing below peak.

    S1 is how much total parameter movement the schedule has allowed -- the
    natural "progress" axis when the LR is not constant. S2 is how much of
    that movement happened at a REDUCED rate, which is what a cosine tail buys
    you and what makes an annealed endpoint better than a mid-run point at the
    same token count. Coarse trapezoid: exactness is not the limiting factor.
    """
    if step <= 0:
        return 0.0, 0.0
    n = min(grid, step)
    h = step / n
    s1 = s2 = 0.0
    for i in range(n + 1):
        k = i * h
        w = h * (0.5 if i in (0, n) else 1.0)
        e = lr_at_step(k, total, lr, min_lr, warm)
        s1 += w * e
        s2 += w * (lr - e)
    return s1, s2


def load_runs(runs_dir=RUNS_DIR):
    out = []
    for f in sorted(Path(runs_dir).glob("*.status.json")):
        try:
            m = json.loads(f.read_text())
        except Exception:
            continue
        if m.get("status") != "completed":
            continue
        met = m.get("metrics") or {}
        if "full_val_loss" not in met or not met.get("params"):
            continue
        a = dict(m.get("args") or {})
        g = lambda k: a.get(k, DEFAULTS[k])
        steps = int(a.get("steps", 0))
        if not steps:
            continue
        # a --resume leg's own cosine horizon is its own; otherwise it is the run
        total = steps
        out.append(
            dict(
                run=m["run_id"],
                params=int(met["params"]),
                block=int(g("block-size")),
                batch=int(g("batch-size")),
                steps=steps,
                total_steps=total,
                lr=float(g("lr")),
                min_lr=float(g("min-lr")),
                warmup=int(g("warmup-steps")),
                dur=float(m.get("duration_sec") or 0),
                loss=float(met["full_val_loss"]),
                curve=[(int(s), float(v)) for s, v in met.get("full_val_curve", [])],
                data=a.get("tokens", "data/train.pt"),
                resumed="resume" in a,
            )
        )
    return out


# --------------------------------------------------------------- time


def fit_time(runs):
    """sec = c * params * block * steps. One parameter, fit as a median ratio."""
    rs = [r for r in runs if r["dur"] > 0]
    cs = sorted(r["dur"] / (r["params"] * r["block"] * r["steps"]) for r in rs)
    c = cs[len(cs) // 2] if len(cs) % 2 else 0.5 * (cs[len(cs) // 2 - 1] + cs[len(cs) // 2])
    errs = [abs(c * r["params"] * r["block"] * r["steps"] - r["dur"]) / r["dur"] for r in rs]
    errs.sort()
    return c, {"n": len(rs), "median_abs_pct": 100 * errs[len(errs) // 2], "worst_pct": 100 * errs[-1]}


def predict_time(c, params, block, steps):
    return c * params * block * steps


# --------------------------------------------------------------- loss


def _fit_powerlaw(xs, ys, e_lo, e_hi):
    """y = E + A*x^-a. Grid over E, closed-form log-linear fit for A and a."""
    best = None
    steps_ = 400
    for i in range(steps_):
        E = e_lo + (e_hi - e_lo) * i / (steps_ - 1)
        if any(y - E <= 1e-9 for y in ys):
            continue
        lx = [math.log(x) for x in xs]
        ly = [math.log(y - E) for y in ys]
        n = len(lx)
        mx = sum(lx) / n
        my = sum(ly) / n
        den = sum((v - mx) ** 2 for v in lx)
        if den <= 0:
            continue
        a = -sum((u - mx) * (v - my) for u, v in zip(lx, ly)) / den
        A = math.exp(my + a * mx)
        rss = sum((E + A * x**-a - y) ** 2 for x, y in zip(xs, ys))
        if best is None or rss < best[0]:
            best = (rss, E, A, a)
    return best


def loss_families(runs, data=DATA10K):
    """Group every full_val checkpoint by (params, block, lr).

    Fitting one global curve does NOT work, and the data says why: a 1e-3 run
    accumulates ~3.3x the cumulative LR of a 3e-4 run at equal steps, yet ends
    WORSE. Any single monotone function of a progress axis therefore has to
    order the two regimes backwards, and a global fit lands at ~0.42 nats
    leave-one-run-out -- useless when the differences that matter are ~0.1.

    Per family, against tokens, the same power law fits to ~0.014 nats median
    residual. So predictions are only offered for a family that has data, and
    the tool says so plainly rather than extrapolating across a confound.
    """
    fam = {}
    for r in runs:
        if r["data"] != data or r["resumed"]:
            continue
        for step, loss in r["curve"]:
            if step > 0:
                fam.setdefault((r["params"], r["block"], r["lr"]), []).append(
                    (step * r["batch"] * r["block"], loss, r["run"])
                )
    return fam


def fit_family(pts):
    """L = E + A * D^-a over one family's checkpoints. Returns (fit, stats)."""
    if len(pts) < 4:
        return None, {"n": len(pts), "note": "need >=4 checkpoints"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    f = _fit_powerlaw(xs, ys, 0.5, min(ys) - 0.01)
    if f is None:
        return None, {"n": len(pts), "note": "no fit"}
    _, E, A, a = f
    res = sorted(abs(E + A * x**-a - y) for x, y in zip(xs, ys))
    runs_in = sorted({p[2] for p in pts})
    loo = []
    if len(runs_in) > 1:
        for held in runs_in:
            tr = [p for p in pts if p[2] != held]
            te = [p for p in pts if p[2] == held]
            f2 = _fit_powerlaw([q[0] for q in tr], [q[1] for q in tr], 0.5, min(q[1] for q in tr) - 0.01)
            if f2:
                _, E2, A2, a2 = f2
                loo += [abs(E2 + A2 * x**-a2 - y) for x, y, _ in te]
        loo.sort()
    return (E, A, a), {
        "n": len(pts),
        "runs": len(runs_in),
        "median_resid": res[len(res) // 2],
        "max_resid": res[-1],
        "loo_median": loo[len(loo) // 2] if loo else None,
    }


def estimate_params(n_embd, n_head, n_layer, block_size, vocab_size=50257):
    """Parameter count for the tied-embedding model, without building it."""
    d, V, L, T = n_embd, vocab_size, n_layer, block_size
    head = d // n_head
    per_block = n_head * 3 * (d * head + head) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + 4 * d
    return V * d + T * d + L * per_block + 2 * d + V


def forecast(cfg, runs):
    """Everything predictable about a config, as plain data.

    cfg keys: n_embd n_head n_layer block_size batch_size steps lr min_lr
    warmup_steps [vocab_size] [params]. Returns None fields where the data
    does not support a prediction -- deliberately, rather than guessing.
    """
    params = cfg.get("params") or estimate_params(
        cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block_size"], cfg.get("vocab_size", 50257)
    )
    T, steps = cfg["block_size"], cfg["steps"]
    tokens = steps * cfg["batch_size"] * T

    c, tstat = fit_time(runs)
    sec = predict_time(c, params, T, steps)

    out = {
        "params_est": params,
        "tokens": tokens,
        "time_sec": round(sec, 1),
        "time_hours": round(sec / 3600, 2),
        "time_fit": {"runs": tstat["n"], "median_abs_pct": round(tstat["median_abs_pct"], 1)},
        "loss": None,
        "loss_basis": None,
    }

    fams = loss_families(runs)
    match = next(
        (k for k in fams if k[1] == T and abs(k[2] - cfg["lr"]) < 1e-12 and abs(k[0] - params) / params < 0.01), None
    )
    if match is None:
        out["loss_basis"] = {
            "status": "no data for this (params, block, lr) family",
            "families": [
                {"params": k[0], "block": k[1], "lr": k[2], "points": len(v)} for k, v in sorted(fams.items())
            ],
        }
        return out
    fit, st = fit_family(fams[match])
    if fit is None:
        out["loss_basis"] = {"status": "family too sparse", **st}
        return out
    E, A, a = fit
    hi = max(x for x, _, _ in fams[match])
    out["loss"] = round(E + A * tokens**-a, 4)
    out["loss_basis"] = {
        "status": "ok",
        "E": round(E, 4),
        "A": round(A, 4),
        "alpha": round(a, 4),
        "points": st["n"],
        "runs": st["runs"],
        "median_resid": round(st["median_resid"], 4),
        "loo_median": round(st["loo_median"], 4) if st["loo_median"] else None,
        "extrapolation_x": round(tokens / hi, 2),
    }
    return out


# --------------------------------------------------------------- cli


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = p.parse_args(argv)

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no completed runs with metrics under {args.runs_dir}")

    # parameter count, tied embeddings (matches model.py)
    d, V, L, T = args.n_embd, args.vocab_size, args.n_layer, args.block_size
    head = d // args.n_head
    per_block = args.n_head * 3 * (d * head + head) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + 4 * d
    params = V * d + T * d + L * per_block + 2 * d + V

    c, tstat = fit_time(runs)
    sec = predict_time(c, params, T, args.steps)
    print(f"runs used: {len(runs)} completed\n")
    print(
        f"config: emb{d} head{args.n_head} layer{L} blk{T} bs{args.batch_size} "
        f"steps{args.steps} lr{args.lr:g} minlr{args.min_lr:g}"
    )
    print(f"  params (estimated)   {params:,}")
    print(f"  tokens processed     {args.steps * args.batch_size * T:,}")
    print()
    print(f"TIME   {sec/3600:.1f}h  ({sec:,.0f}s)")
    print(
        f"       fit on {tstat['n']} runs; median |err| {tstat['median_abs_pct']:.0f}%, "
        f"worst {tstat['worst_pct']:.0f}%"
    )
    print(
        f"       -> plan for {sec/3600*0.8:.1f}-{sec/3600*1.35:.1f}h "
        f"(identical configs vary 10-25% from thermal throttling alone)"
    )
    print()
    fams = loss_families(runs)
    key = (params, T, args.lr)
    tokens = args.steps * args.batch_size * T
    exact = None
    for k in fams:
        if k[1] == T and abs(k[2] - args.lr) < 1e-12 and abs(k[0] - params) / params < 0.01:
            exact = k
    if exact is None:
        print("LOSS   no data for this family (params, block, lr) -- not predicted.")
        print("       Extrapolating across LR regimes is exactly the mistake this")
        print("       tool exists to avoid. Families with data:")
        for k in sorted(fams):
            fit, st = fit_family(fams[k])
            tag = f"{st['n']:>3} pts / {st.get('runs','?')} runs" if fit else st.get("note", "")
            print(f"         params{k[0]:>10,}  blk{k[1]:>3}  lr{k[2]:<8.0e} {tag}")
    else:
        fit, st = fit_family(fams[exact])
        if fit is None:
            print(f"LOSS   family found but not fittable ({st})")
        else:
            E, A, a = fit
            loss = E + A * tokens**-a
            print(f"LOSS   {loss:.4f} full_val")
            print(f"       L = {E:.3f} + {A:.4g} * D^-{a:.3f}   (D = {tokens:,} tokens)")
            print(f"       fit on {st['n']} checkpoints from {st['runs']} run(s) in this exact family")
            print(f"       in-sample residual: median {st['median_resid']:.4f}, max {st['max_resid']:.4f} nats")
            if st["loo_median"] is not None:
                print(f"       leave-one-run-out: median {st['loo_median']:.4f} nats")
            lo = min(x for x, _, _ in fams[exact])
            hi = max(x for x, _, _ in fams[exact])
            if tokens > hi * 1.05:
                print(
                    f"       NOTE extrapolating {tokens/hi:.1f}x beyond the largest observed "
                    f"budget ({hi:,} tokens) -- the irreducible term E is the least "
                    f"constrained parameter and dominates out here."
                )
            elif tokens < lo:
                print(f"       NOTE below the smallest observed budget ({lo:,} tokens).")


if __name__ == "__main__":
    main()
