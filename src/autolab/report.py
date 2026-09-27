"""Build `report.json` for one run from its TensorBoard events, launch.json and log.

Sources, all inside the run dir (see autolab.trainer):
- TensorBoard scalars under runs/tb/<run_id>/: `train/batch_loss`, `train/lr`,
  `train/grad_norm`, `eval/train_loss`, `eval/val_loss`, `eval/full_val_loss`.
- launch.json: request (config, seed, dataset, budget), git commit, timing, budget outcome.
- train.log: device, parameter count, full model config and dataset size, as
  printed by the trainer. Parsing the log (rather than rebuilding the model)
  keeps parameter counts right for edited candidate code too.

Curves are `eval/train_loss` and `eval/val_loss` (fixed eval batches); the
headline number is the final `eval/full_val_loss` over the whole frozen val.
Non-finite values are written as null.

    uv run python -m autolab.report autolab/runs/<run_id>
"""

import ast
import hashlib
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from statistics import median

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.toml")
REPORT_VERSION = 1

Series = list[tuple[int, float]]


@dataclass(frozen=True)
class ReportParams:
    max_curve_points: int = 200
    ema_alpha: float = 0.3
    tail_fraction: float = 0.2
    spike_window: int = 21
    spike_mad_k: float = 3.0
    early_skip_fraction: float = 0.05

    @classmethod
    def load(cls, path: Path = THRESHOLDS_PATH) -> "ReportParams":
        return cls(**tomllib.loads(Path(path).read_text())["report"])


# --- small numeric helpers (pure) ------------------------------------------


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def clean(x: float | None, digits: int = 6) -> float | None:
    """JSON-safe float: non-finite -> None."""
    return round(float(x), digits) if finite(x) else None


def subsample(series: Series, max_points: int) -> Series:
    """Evenly spaced points, always keeping the first and last."""
    n = len(series)
    if n <= max_points:
        return list(series)
    idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return [series[i] for i in idx]


def ema(values: list[float], alpha: float) -> float | None:
    out = None
    for v in values:
        if not finite(v):
            continue
        out = v if out is None else alpha * v + (1 - alpha) * out
    return out


def tail(series: Series, fraction: float) -> Series:
    pts = [(s, v) for s, v in series if finite(v)]
    if not pts:
        return []
    k = max(4, math.ceil(len(pts) * fraction))  # diagnose() needs >= min_tail_points (4)
    return pts[-k:]


def linfit(series: Series) -> dict:
    """OLS of value on step. Slope is per 1k steps; drop = fitted change across the window."""
    n = len(series)
    out = {"n": n, "end_value": None, "slope_per_1k_steps": None, "t": None, "change": None, "rel_change": None}
    if n < 3:
        return out
    xs = [s for s, _ in series]
    ys = [v for _, v in series]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return out
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    ssr = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(ssr / (n - 2) / sxx) if ssr > 0 else 0.0
    t = slope / se if se > 0 else (math.copysign(1e6, slope) if slope != 0 else 0.0)
    change = slope * (xs[-1] - xs[0])
    out.update(
        end_value=my + slope * (xs[-1] - mx),
        slope_per_1k_steps=slope * 1000,
        t=max(-1e6, min(1e6, t)),
        change=change,
        rel_change=change / abs(my) if my else None,
    )
    return out


def rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]


def spike_stats(batch_loss: Series, window: int, k: float) -> dict:
    """Upward spikes: residual vs rolling median above k robust std (1.4826 * MAD)."""
    vals = [v for _, v in batch_loss if finite(v)]
    if len(vals) < max(window, 5):
        return {"count": 0, "rate": 0.0, "n": len(vals), "robust_std": None, "steps": []}
    resid = [v - m for v, m in zip(vals, rolling_median(vals, window))]
    med = median(resid)
    # Floor so a noise-free stretch (MAD = 0) still flags real jumps instead of everything or nothing.
    robust_std = max(1.4826 * median(abs(r - med) for r in resid), 1e-6 * median(abs(v) for v in vals))
    finite_steps = [s for s, v in batch_loss if finite(v)]
    spikes = [s for s, r in zip(finite_steps, resid) if r - med > k * robust_std]
    return {"count": len(spikes), "rate": len(spikes) / len(vals), "n": len(vals),
            "robust_std": robust_std, "steps": spikes[:50]}


def summarize_curves(train: Series, val: Series, params: ReportParams) -> dict:
    """Summary block from the eval curves (pure; unit-tested on synthetic curves)."""
    train_ema = ema([v for _, v in train], params.ema_alpha)
    val_ema = ema([v for _, v in val], params.ema_alpha)
    val_by_step = dict(val)
    gap_series = [(s, val_by_step[s] - v) for s, v in train if s in val_by_step
                  and finite(v) and finite(val_by_step[s])]
    finite_val = [(s, v) for s, v in val if finite(v)]
    best_step, best_val = min(finite_val, key=lambda p: p[1]) if finite_val else (None, None)
    train_fit = linfit(tail(train, params.tail_fraction))
    val_fit = linfit(tail(val, params.tail_fraction))
    gap_fit = linfit(tail(gap_series, params.tail_fraction))
    # "Smoothed final" = the tail line fit evaluated at the last step. Unlike an EMA it
    # doesn't lag a still-falling curve (on 13-point screens the EMA sat ~0.1 above the curve).
    train_end = train_fit["end_value"] if finite(train_fit["end_value"]) else train_ema
    val_end = val_fit["end_value"] if finite(val_fit["end_value"]) else val_ema
    return {
        "final_train_loss_smooth": clean(train_end),
        "final_val_loss_smooth": clean(val_end),
        "final_train_loss_ema": clean(train_ema),
        "final_val_loss_ema": clean(val_ema),
        "final_train_loss": clean(train[-1][1]) if train else None,
        "final_val_loss": clean(val[-1][1]) if val else None,
        "gap": clean(val_end - train_end) if finite(train_end) and finite(val_end) else None,
        "gap_trend": {k: clean(v) if isinstance(v, float) else v for k, v in gap_fit.items()},
        "train_slope_tail": {k: clean(v) if isinstance(v, float) else v for k, v in train_fit.items()},
        "val_slope_tail": {k: clean(v) if isinstance(v, float) else v for k, v in val_fit.items()},
        "best_val_loss": clean(best_val),
        "best_val_step": best_step,
        "tail_fraction": params.tail_fraction,
    }


def grad_norm_stats(grad_norm: Series) -> dict | None:
    vals = sorted(v for _, v in grad_norm if finite(v))
    if not vals:
        return None
    med = median(vals)
    return {
        "n": len(vals),
        "median": clean(med),
        "mean": clean(sum(vals) / len(vals)),
        "p95": clean(vals[min(len(vals) - 1, int(0.95 * len(vals)))]),
        "max": clean(vals[-1]),
        "last": clean(grad_norm[-1][1]),
        "max_over_median": clean(vals[-1] / med) if med > 0 else None,
        "non_finite": sum(1 for _, v in grad_norm if not finite(v)),
    }


# --- sources ----------------------------------------------------------------


def load_scalars(tb_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    """All scalars in a TB log dir as {tag: [(step, value, wall_time), ...]}, sorted by step.

    A step logged twice (e.g. a resumed run) keeps the latest write.
    """
    from tensorboard.backend.event_processing import event_accumulator as ea

    acc = ea.EventAccumulator(str(tb_dir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        by_step: dict[int, tuple[int, float, float]] = {}
        for e in acc.Scalars(tag):
            by_step[e.step] = (e.step, float(e.value), e.wall_time)
        out[tag] = sorted(by_step.values())
    return out


_INT = r"([\d,]+)"


def parse_train_log(text: str) -> dict:
    """Device, params, model config and dataset sizes as printed by mini_llm.train."""
    out: dict = {}
    if m := re.search(r"^Using device: (\S+)", text, re.M):
        out["device"] = m.group(1)
    if m := re.search(rf"^Model: {_INT} parameters", text, re.M):
        out["params"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"^Config: (\{.*\})$", text, re.M):
        try:
            out["model_config"] = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            pass
    if m := re.search(rf"^Train tokens:\s+{_INT}", text, re.M):
        out["train_tokens"] = int(m.group(1).replace(",", ""))
    if m := re.search(rf"^Validation tokens:\s+{_INT}", text, re.M):
        out["val_tokens"] = int(m.group(1).replace(",", ""))
    out["traceback"] = "Traceback (most recent call last)" in text
    return out


def config_hash(config: dict) -> str:
    """Hash of the run config without seed, so seeds of one config share it."""
    body = {k: v for k, v in config.items() if k != "seed"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _series(scalars: dict, tag: str) -> Series:
    return [(s, v) for s, v, _ in scalars.get(tag, [])]


def build_report(run_dir: Path, params: ReportParams | None = None) -> dict:
    run_dir = Path(run_dir)
    params = params or ReportParams.load()
    launch = json.loads((run_dir / "launch.json").read_text()) if (run_dir / "launch.json").exists() else {}
    run_id = launch.get("run_id") or run_dir.name
    log = parse_train_log((run_dir / "train.log").read_text(errors="replace")) \
        if (run_dir / "train.log").exists() else {}
    scalars = load_scalars(run_dir / "runs" / "tb" / run_id)

    request = launch.get("request", {})
    model_cfg = log.get("model_config") or request.get("model", {})
    config = {**request.get("model", {}), **request.get("optim", {}), **request.get("eval", {}),
              "steps": launch.get("steps"), "seed": request.get("seed")}
    if log.get("model_config"):
        config["vocab_size"] = log["model_config"].get("vocab_size")

    batch_loss, lr = _series(scalars, "train/batch_loss"), _series(scalars, "train/lr")
    train, val = _series(scalars, "eval/train_loss"), _series(scalars, "eval/val_loss")
    full_val = _series(scalars, "eval/full_val_loss")
    grad_norm = _series(scalars, "train/grad_norm")

    all_steps = [s for series in (batch_loss, train, val) for s, _ in series]
    last_step = max(all_steps) if all_steps else None
    # Spikes and grad-norm stats skip the initial descent (warmup, or the first
    # early_skip_fraction of the run if longer): there the loss falls from ~10.8 so
    # steeply that every point sits far above a rolling median.
    skip_until = max(params.early_skip_fraction * (last_step or 0), config.get("warmup_steps") or 0)
    tokens_per_step = launch.get("tokens_per_step") or (
        (config.get("batch_size") or 0) * (model_cfg.get("block_size") or 0) or None)
    tokens_seen = (last_step + 1) * tokens_per_step if last_step is not None and tokens_per_step else None

    n_params = log.get("params")
    embedding_params = None
    if n_params and model_cfg.get("vocab_size") and model_cfg.get("n_embd"):
        # token table (tied with lm_head) + learned position table
        embedding_params = (model_cfg["vocab_size"] + model_cfg.get("block_size", 0)) * model_cfg["n_embd"]
    non_emb = n_params - embedding_params if n_params and embedding_params else None
    dataset_tokens = log.get("train_tokens")

    # Training wall time: first to last logged batch loss (the loop, incl. periodic evals).
    bl_raw = scalars.get("train/batch_loss", [])
    train_wall_s = bl_raw[-1][2] - bl_raw[0][2] if len(bl_raw) >= 2 else None
    tok_per_s = ((bl_raw[-1][0] - bl_raw[0][0]) * tokens_per_step / train_wall_s
                 if train_wall_s and tokens_per_step else None)

    non_finite = {tag: sum(1 for _, v, _ in rows if not math.isfinite(v)) for tag, rows in scalars.items()}
    budget = request.get("budget", {})

    return {
        "report_version": REPORT_VERSION,
        "identity": {
            "run_id": run_id,
            "git_commit": launch.get("git", {}).get("commit"),
            "git_dirty": launch.get("git", {}).get("dirty"),
            "config": config,
            "config_hash": config_hash({**config, "dataset_id": request.get("dataset_id")}),
            "seed": request.get("seed"),
            "dataset_id": request.get("dataset_id"),
            "train_sha256": launch.get("train_sha256"),
            "val_sha256": launch.get("val_sha256"),
            "start_time": launch.get("started_at"),
            "end_time": launch.get("ended_at"),
            "status": launch.get("status"),
        },
        "scale": {
            "params": n_params,
            "embedding_params": embedding_params,
            "non_embedding_params": non_emb,
            "dataset_tokens": dataset_tokens,
            "val_tokens": log.get("val_tokens"),
            "tokens_seen": tokens_seen,
            "steps_done": last_step + 1 if last_step is not None else None,
            "epochs": clean(tokens_seen / dataset_tokens) if tokens_seen and dataset_tokens else None,
            "tokens_per_param": clean(tokens_seen / n_params) if tokens_seen and n_params else None,
            "tokens_per_non_embedding_param": clean(tokens_seen / non_emb) if tokens_seen and non_emb else None,
        },
        "curves": {
            "train_loss": [[s, clean(v)] for s, v in subsample(train, params.max_curve_points)],
            "val_loss": [[s, clean(v)] for s, v in subsample(val, params.max_curve_points)],
            "full_val_loss": [[s, clean(v)] for s, v in full_val],
        },
        "summary": {
            **summarize_curves(train, val, params),
            "final_full_val_loss": clean(full_val[-1][1]) if full_val else None,
            "lr_peak": clean(max((v for _, v in lr if finite(v)), default=None)),
            "lr_final": clean(lr[-1][1]) if lr else None,
        },
        "health": {
            "nan_or_inf": any(non_finite.values()) or bool(log.get("traceback")),
            "non_finite_counts": {k: v for k, v in non_finite.items() if v},
            "trainer_traceback": bool(log.get("traceback")),
            "spikes": {**spike_stats([(st, v) for st, v in batch_loss if st > skip_until],
                                     params.spike_window, params.spike_mad_k), "skipped_until_step": skip_until},
            "grad_norm": grad_norm_stats([(st, v) for st, v in grad_norm if st > skip_until]),
        },
        "performance": {
            "tokens_per_sec": clean(tok_per_s, 1),
            "train_wall_s": clean(train_wall_s, 2),
            "wall_s": launch.get("wall_s"),
            "gpu_wait_s": launch.get("gpu_wait_s"),
            "device": log.get("device"),
        },
        "budget": {
            "type": "tokens",
            "tokens": budget.get("tokens"),
            "wall_clock_s": budget.get("wall_clock_s"),
            "hit": launch.get("budget_hit"),
        },
    }


def write_report(report: dict, run_dir: Path) -> Path:
    path = Path(run_dir) / "report.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m autolab.report <run_dir>")
    report = build_report(Path(argv[0]))
    print(write_report(report, Path(argv[0])))


if __name__ == "__main__":
    main()
