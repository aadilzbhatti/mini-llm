"""Rule-based diagnosis of one run report: `diagnose(report, history) -> Diagnosis`.

Pure: no I/O besides loading thresholds (pass `Thresholds` explicitly to avoid
even that). Every label carries a confidence in [0, 1], the numeric evidence
it was derived from, and a suggested next step.

Trend of a curve over the report's tail window (last `tail_fraction` of eval points):
- falling: fitted relative drop >= falling_rel_drop and slope t <= -t_min
- crawl:   significant (t <= -t_min) but smaller drop, above flat_rel_drop
- flat:    |relative change| < flat_rel_drop, or |t| < t_min
- rising:  relative rise >= rising_rel and t >= t_min

`History` carries what the report alone can't show:
- `reports`: past report dicts (same shape as autolab.report output);
- `notebook`: past notebook entries. diagnose() reads only `action` (str),
  `accepted` (bool | None) and `improved` (bool | None) from each, so the
  notebook schema can grow freely around them;
- `noise_std`: seed-to-seed std of final val loss for the champion, if known.
"""

import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.toml")

LABELS = ("still_improving", "data_limited", "optimization_limited", "capacity_limited",
          "unstable", "inconclusive")


@dataclass(frozen=True)
class Thresholds:
    min_tail_points: int = 4
    t_min: float = 3.0
    falling_rel_drop: float = 0.005
    flat_rel_drop: float = 0.002
    rising_rel: float = 0.002
    gap_small: float = 0.05
    gap_growth: float = 0.005
    gap_stable_max: float = 0.01
    epochs_support: float = 1.0
    epochs_overfit: float = 2.0
    data_gap_large: float = 0.15
    tokens_per_param_ref: float = 20.0
    spike_rate_max: float = 0.03
    spike_min_count: int = 5
    divergence_rel: float = 0.03
    grad_norm_max_over_median: float = 3.0
    grad_norm_p95_over_median: float = 1.5
    noise_mult: float = 1.0
    lr_tuning_actions: tuple[str, ...] = ("lr_range_test", "hparam_search")
    data_actions: tuple[str, ...] = ("build_dataset",)

    @classmethod
    def load(cls, path: Path = THRESHOLDS_PATH) -> "Thresholds":
        raw = tomllib.loads(Path(path).read_text())["diagnose"]
        known = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items() if k in known})


@dataclass
class History:
    reports: list[dict] = field(default_factory=list)
    notebook: list[dict] = field(default_factory=list)
    noise_std: float | None = None


@dataclass
class Label:
    name: str
    confidence: float
    evidence: dict
    suggestion: str


@dataclass
class Diagnosis:
    labels: list[Label]
    notes: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return max(self.labels, key=lambda lab: lab.confidence).name

    def names(self) -> list[str]:
        return [lab.name for lab in self.labels]

    def get(self, name: str) -> Label | None:
        return next((lab for lab in self.labels if lab.name == name), None)

    def to_dict(self) -> dict:
        return {"primary": self.primary, "labels": [asdict(lab) for lab in self.labels], "notes": self.notes}

    def summary(self) -> str:
        return ", ".join(f"{lab.name}({lab.confidence:.2f})" for lab in self.labels)


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x, digits)


def trend(fit: dict, th: Thresholds) -> str:
    rel, t = fit.get("rel_change"), fit.get("t")
    if rel is None or t is None:
        return "unknown"
    if rel >= th.rising_rel and t >= th.t_min:
        return "rising"
    if abs(rel) < th.flat_rel_drop or abs(t) < th.t_min:
        return "flat"
    if rel <= -th.falling_rel_drop:
        return "falling"
    if rel < 0:
        return "crawl"
    return "flat"  # small significant rise below rising_rel


def _model_key(report: dict) -> tuple:
    cfg = report.get("identity", {}).get("config", {})
    return tuple(cfg.get(k) for k in ("block_size", "n_embd", "n_head", "n_layer"))


def _config_sans(report: dict, *drop: str) -> dict:
    cfg = dict(report.get("identity", {}).get("config", {}))
    for k in drop:
        cfg.pop(k, None)
    return cfg


def lr_tuned(report: dict, history: History, th: Thresholds) -> tuple[bool, dict]:
    actions = [e.get("action") for e in history.notebook if e.get("action") in th.lr_tuning_actions]
    same_model_lrs = {
        r.get("identity", {}).get("config", {}).get("lr")
        for r in history.reports + [report]
        if _model_key(r) == _model_key(report)
    } - {None}
    return bool(actions) or len(same_model_lrs) >= 3, {
        "lr_tuning_actions": len(actions), "distinct_lrs_same_model": len(same_model_lrs)}


def data_increase_failed(report: dict, history: History, th: Thresholds) -> tuple[bool, dict]:
    """A past data increase that didn't beat the smaller-data run by more than noise."""
    explicit = [e for e in history.notebook if e.get("action") in th.data_actions
                and (e.get("improved") is False or e.get("accepted") is False)]
    margin = th.noise_mult * (history.noise_std or 0.0)
    pairs = []
    runs = history.reports + [report]
    for small in runs:
        for big in runs:
            ts, tb = small.get("scale", {}).get("dataset_tokens"), big.get("scale", {}).get("dataset_tokens")
            if not ts or not tb or tb <= ts:
                continue
            if _config_sans(small, "seed") != _config_sans(big, "seed"):
                continue
            vs = small.get("summary", {}).get("final_full_val_loss") or small.get("summary", {}).get("final_val_loss_ema")
            vb = big.get("summary", {}).get("final_full_val_loss") or big.get("summary", {}).get("final_val_loss_ema")
            if vs is not None and vb is not None:
                pairs.append({"small_tokens": ts, "big_tokens": tb, "small_val": vs, "big_val": vb,
                              "helped": vs - vb > margin})
    failed = bool(explicit) or any(not p["helped"] for p in pairs)
    return failed, {"explicit_entries": len(explicit), "report_pairs": pairs[:5], "noise_margin": margin}


def diagnose(report: dict, history: History | None = None, th: Thresholds | None = None) -> Diagnosis:
    history = history or History()
    th = th or Thresholds.load()
    s, health, scale = report["summary"], report["health"], report.get("scale", {})
    tf, vf, gf = s["train_slope_tail"], s["val_slope_tail"], s["gap_trend"]
    labels: list[Label] = []
    notes: list[str] = []

    # --- unstable: checked first; NaN makes every other number meaningless -----------
    if health.get("nan_or_inf"):
        return Diagnosis([Label("unstable", 1.0,
                                {"nan_or_inf": True, "non_finite_counts": health.get("non_finite_counts"),
                                 "trainer_traceback": health.get("trainer_traceback")},
                                "Lower the LR (or add warmup) and rerun; check the trainer log.")])

    if min(tf.get("n") or 0, vf.get("n") or 0) < th.min_tail_points:
        return Diagnosis([Label("inconclusive", 0.9,
                                {"train_tail_points": tf.get("n"), "val_tail_points": vf.get("n"),
                                 "min_tail_points": th.min_tail_points},
                                "Too few eval points to judge; run longer or evaluate more often.")])

    t_train, t_val = trend(tf, th), trend(vf, th)
    gap = s.get("gap")
    gap_change, gap_t = gf.get("change"), gf.get("t")
    gap_growing = gap_change is not None and gap_t is not None and gap_change >= th.gap_growth and gap_t >= th.t_min
    gap_stable = gap_change is not None and abs(gap_change) < th.gap_stable_max
    spikes = health.get("spikes") or {}
    spike_rate = spikes.get("rate") or 0.0
    gn = health.get("grad_norm") or {}
    gn_ratio = gn.get("max_over_median")
    gn_p95_ratio = gn["p95"] / gn["median"] if gn.get("p95") and gn.get("median") else None
    best = s.get("best_val_loss")
    final_val = s.get("final_val_loss_smooth", s.get("final_val_loss_ema"))
    lr_peak, lr_final = s.get("lr_peak"), s.get("lr_final")
    lr_ratio = lr_final / lr_peak if lr_peak and lr_final is not None else None

    base_ev = {
        "train_trend": t_train, "val_trend": t_val,
        "train_rel_change_tail": _r(tf.get("rel_change")), "train_t": _r(tf.get("t"), 1),
        "val_rel_change_tail": _r(vf.get("rel_change")), "val_t": _r(vf.get("t"), 1),
        "gap": _r(gap), "gap_change_tail": _r(gap_change), "gap_t": _r(gap_t, 1),
    }

    # --- unstable ------------------------------------------------------------------
    unstable_ev = {}
    diverged = best is not None and final_val is not None and final_val > best * (1 + th.divergence_rel)
    if diverged and t_train == "rising":
        unstable_ev.update(diverged=True, best_val=best, final_val_smooth=final_val, train_trend=t_train)
    spiky = spike_rate > th.spike_rate_max and (spikes.get("count") or 0) >= th.spike_min_count
    if spiky:
        unstable_ev.update(spike_rate=_r(spike_rate), spike_count=spikes.get("count"),
                           spike_rate_max=th.spike_rate_max)
    if gn_ratio is not None and gn_ratio > th.grad_norm_max_over_median:
        unstable_ev.update(grad_norm_max_over_median=gn_ratio, grad_norm_max=gn.get("max"))
    if gn_p95_ratio is not None and gn_p95_ratio > th.grad_norm_p95_over_median:
        unstable_ev.update(grad_norm_p95_over_median=_r(gn_p95_ratio, 2))
    if unstable_ev:
        conf = 0.9 if unstable_ev.get("diverged") else 0.5
        if spiky:
            conf = max(conf, min(0.9, 0.55 + 0.1 * (spike_rate / th.spike_rate_max - 1)))
        if "grad_norm_p95_over_median" in unstable_ev:  # sustained, not one bad batch
            conf = max(conf, 0.75)
        if "grad_norm_max_over_median" in unstable_ev and len(unstable_ev) > 1:
            conf = min(0.95, conf + 0.1)
        labels.append(Label("unstable", round(conf, 2), {**unstable_ev, "lr_peak": lr_peak},
                            "Lower the peak LR or lengthen warmup; lr_range_test to find the divergence point."))
        if unstable_ev.get("diverged"):
            return Diagnosis(labels, notes)

    if lr_ratio is not None and lr_ratio < 0.05 and t_val in ("flat", "crawl"):
        notes.append("LR annealed to <5% of peak by the end, so flat tail curves are partly the schedule")

    # --- still_improving -------------------------------------------------------------
    val_drop = -(vf.get("change") or 0.0)
    noise_floor = th.noise_mult * history.noise_std if history.noise_std else None
    if t_val == "falling":
        if noise_floor is not None and val_drop < noise_floor:
            notes.append(f"val still falling but the tail drop {val_drop:.4f} is under seed noise "
                         f"{noise_floor:.4f}; treated as a crawl")
            t_val = "crawl"
        else:
            strength = min(1.0, abs(vf["rel_change"]) / (3 * th.falling_rel_drop))
            labels.append(Label("still_improving", round(0.6 + 0.3 * strength, 2),
                                {**base_ev, "val_drop_tail": _r(val_drop), "noise_floor": _r(noise_floor),
                                 "lr_final_over_peak": _r(lr_ratio)},
                                "Run longer (bigger token budget) or leave as is."))

    # --- data_limited ------------------------------------------------------------------
    epochs = scale.get("epochs")
    params = scale.get("params")
    ds_tokens = scale.get("dataset_tokens")
    ds_tok_per_param = ds_tokens / params if ds_tokens and params else None
    multi_epoch = epochs is not None and epochs >= th.epochs_support
    # Classic: val flat/rising while train falls. Under an annealing schedule val can still
    # crawl down while the model overfits, so "train falling faster than val, gap growing,
    # after >= 1 epoch" counts too (the 82M-token runs: 4 epochs, gap 0.40, +0.01 over the tail).
    train_outpaces_val = (tf.get("rel_change") is not None and vf.get("rel_change") is not None
                          and tf["rel_change"] < vf["rel_change"])
    classic = t_train in ("falling", "crawl") and t_val in ("flat", "rising") and gap_growing
    annealed = (t_train in ("falling", "crawl") and t_val == "crawl" and gap_growing
                and train_outpaces_val and multi_epoch)
    # Longer runs keep val falling while they overfit (s2+data40k@122M/p21: val -1.2% over the tail, but 3 epochs,
    # gap 0.28 growing at t ~ 12, train falling faster): still_improving AND data_limited, so the data policy can
    # test more data instead of waiting for val to flatten (owner, 2026-09-30).
    improving_but_overfitting = (t_train == "falling" and t_val == "falling" and gap_growing and train_outpaces_val
                                 and epochs is not None and epochs >= th.epochs_overfit)
    if classic or annealed or improving_but_overfitting:
        supported = multi_epoch or (ds_tok_per_param is not None and ds_tok_per_param < th.tokens_per_param_ref)
        conf = (0.5 + (0.25 if supported else 0.0) + (0.1 if t_val == "rising" else 0.0)
                + (0.1 if gap is not None and gap >= th.data_gap_large else 0.0))
        labels.append(Label("data_limited", round(conf, 2),
                            {**base_ev, "epochs": epochs, "dataset_tokens_per_param": _r(ds_tok_per_param, 2),
                             "tokens_per_param": scale.get("tokens_per_param"), "supported": supported,
                             "pattern": "val flat/rising" if classic else "train outpaces a crawling val" if annealed
                             else "overfitting while val still falls"},
                            "build_dataset (bigger), or more regularization (dropout / weight decay)."))

    # --- optimization_limited / capacity_limited ---------------------------------------
    if (t_train in ("flat", "crawl") and t_val in ("flat", "crawl") and gap is not None
            and abs(gap) < th.gap_small and gap_stable):
        instability = spike_rate > th.spike_rate_max / 2 or (
            gn_p95_ratio is not None and gn_p95_ratio > 1 + (th.grad_norm_p95_over_median - 1) / 2)
        if instability:
            hint, suggestion = "lr_too_high", "Lower the peak LR (hparam_search) or run lr_range_test."
        elif t_train == "crawl" or t_val == "crawl":
            hint, suggestion = "lr_too_low_or_schedule", "Raise the LR or change the schedule; lr_range_test for bounds."
        else:
            hint, suggestion = "plateau", "hparam_search over LR / warmup / min-LR ratio / batch size."
        opt_ev = {**base_ev, "hint": hint, "spike_rate": _r(spike_rate),
                  "grad_norm_max_over_median": gn_ratio, "lr_final_over_peak": _r(lr_ratio)}

        tuned, tuned_ev = lr_tuned(report, history, th)
        data_failed, data_ev = data_increase_failed(report, history, th)
        prereqs = int(tuned) + int(data_failed)
        if t_train == "flat" and prereqs == 2:
            labels.append(Label("capacity_limited", 0.7, {**base_ev, **tuned_ev, **data_ev},
                                "Architecture change (code_edit) or a larger model (wider_model ablation)."))
            labels.append(Label("optimization_limited", 0.3, opt_ev, suggestion))
        else:
            labels.append(Label("optimization_limited", 0.65 if prereqs == 0 else 0.55, opt_ev, suggestion))
            if t_train == "flat" and prereqs == 1:
                labels.append(Label("capacity_limited", 0.35, {**base_ev, **tuned_ev, **data_ev,
                                    "missing": "data increase" if tuned else "LR tuning"},
                                    "Establish the missing prerequisite before concluding capacity."))

    # --- inconclusive ---------------------------------------------------------------------
    trend_labels = [lab for lab in labels if lab.name != "unstable"]
    if not trend_labels or max(lab.confidence for lab in trend_labels) < 0.5:
        labels.append(Label("inconclusive", 0.6 if not trend_labels else 0.4, base_ev,
                            "Run an ablation (half_data / wider_model) to separate data from capacity."))
    return Diagnosis(labels, notes)
