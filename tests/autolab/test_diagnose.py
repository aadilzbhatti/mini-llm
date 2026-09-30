"""diagnose() on synthetic curves, one or more cases per label."""

import math
import random

from autolab.diagnose import History, Thresholds, diagnose, trend
from autolab.report import ReportParams, spike_stats, summarize_curves

TH = Thresholds()
PARAMS = ReportParams()
STEPS = range(0, 5001, 100)


def curve(fn, noise=0.002, seed=0):
    rng = random.Random(seed)
    return [(s, fn(s) + rng.gauss(0, noise)) for s in STEPS]


def batch_losses(n=500, spikes=0, base=4.5, noise=0.05, seed=1):
    rng = random.Random(seed)
    pts = [(i * 10, base + rng.gauss(0, noise)) for i in range(n)]
    for j in range(spikes):
        i = 10 + j * (n - 20) // max(spikes, 1)
        pts[i] = (pts[i][0], pts[i][1] + 1.5)
    return pts


def make_report(train_fn, val_fn, *, spikes=0, nan=False, epochs=0.1, lr=1e-3, lr_final=5e-4,
                dataset_tokens=20_543_855, gap_noise=0.002, grad_norm=None):
    train, val = curve(train_fn, gap_noise, 0), curve(val_fn, gap_noise, 1)
    summary = summarize_curves(train, val, PARAMS)
    summary.update(final_full_val_loss=summary["final_val_loss_ema"], lr_peak=lr, lr_final=lr_final)
    return {
        "identity": {"config": {"block_size": 64, "n_embd": 128, "n_head": 4, "n_layer": 4,
                                "batch_size": 4, "lr": lr, "steps": 5000, "seed": 42}},
        "scale": {"params": 7_280_000, "dataset_tokens": dataset_tokens, "epochs": epochs,
                  "tokens_per_param": 0.2},
        "summary": summary,
        "health": {"nan_or_inf": nan, "non_finite_counts": {"train/batch_loss": 3} if nan else {},
                   "spikes": spike_stats(batch_losses(spikes=spikes), PARAMS.spike_window, PARAMS.spike_mad_k),
                   "grad_norm": grad_norm},
    }


def plateau(s):
    return 4 + 2 * math.exp(-s / 500)


# --- still_improving -----------------------------------------------------------------


def test_still_improving():
    r = make_report(lambda s: 4 + 3 * math.exp(-s / 2500), lambda s: 4.05 + 3 * math.exp(-s / 2500))
    d = diagnose(r, History(), TH)
    assert d.primary == "still_improving"
    lab = d.get("still_improving")
    assert lab.confidence >= 0.8
    assert lab.evidence["val_trend"] == "falling" and lab.evidence["val_rel_change_tail"] < -0.01


def test_still_improving_below_seed_noise_is_not_a_trend():
    r = make_report(lambda s: 4 + 3 * math.exp(-s / 2500), lambda s: 4.05 + 3 * math.exp(-s / 2500))
    d = diagnose(r, History(noise_std=0.5), TH)
    assert "still_improving" not in d.names()
    assert any("seed noise" in n for n in d.notes)


# --- data_limited ------------------------------------------------------------------------


def overfit_train(s):
    return 4 + 2 * math.exp(-s / 800) - 0.00008 * s


def overfit_val(s):
    return 4.1 + 2 * math.exp(-s / 800) + 0.00004 * max(0, s - 2000)


def test_data_limited_supported_by_epochs():
    d = diagnose(make_report(overfit_train, overfit_val, epochs=2.5), History(), TH)
    assert d.primary == "data_limited"
    lab = d.get("data_limited")
    assert lab.evidence["supported"] is True and lab.confidence >= 0.8
    assert lab.evidence["gap_change_tail"] > TH.gap_growth


def test_data_limited_unsupported_has_lower_confidence():
    d = diagnose(make_report(overfit_train, overfit_val, epochs=0.1, dataset_tokens=500_000_000), History(), TH)
    assert d.get("data_limited").evidence["supported"] is False
    assert d.get("data_limited").confidence < 0.8


# --- optimization_limited ------------------------------------------------------------------


def test_optimization_limited_plateau():
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02), History(), TH)
    assert d.primary == "optimization_limited"
    assert d.get("optimization_limited").evidence["hint"] == "plateau"
    assert "capacity_limited" not in d.names()


def test_optimization_limited_spiky_points_to_lr_too_high():
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02, spikes=8), History(), TH)
    lab = d.get("optimization_limited")
    assert lab.evidence["hint"] == "lr_too_high"
    assert "unstable" not in d.names()  # 1.6% spikes: below the unstable threshold


def test_optimization_limited_slow_crawl_points_to_lr_too_low():
    def crawl(s):
        return 5 - 0.000015 * s  # ~0.3% over the 1000-step tail window, smooth

    d = diagnose(make_report(crawl, lambda s: crawl(s) + 0.02, gap_noise=0.0001), History(), TH)
    lab = d.get("optimization_limited")
    assert lab.evidence["train_trend"] == "crawl"
    assert lab.evidence["hint"] == "lr_too_low_or_schedule"


def test_annealed_schedule_is_noted():
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02, lr_final=2e-6), History(), TH)
    assert any("annealed" in n for n in d.notes)


# --- capacity_limited -------------------------------------------------------------------------


def test_capacity_limited_after_lr_tuned_and_data_didnt_help():
    history = History(notebook=[{"action": "lr_range_test", "accepted": None},
                                {"action": "build_dataset", "accepted": False, "improved": False}])
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02), history, TH)
    assert d.primary == "capacity_limited"
    assert d.get("capacity_limited").evidence["lr_tuning_actions"] == 1


def test_capacity_limited_derived_from_past_reports():
    current = make_report(plateau, lambda s: plateau(s) + 0.02)
    smaller = make_report(plateau, lambda s: plateau(s) + 0.02, dataset_tokens=10_000_000)
    lrs = [make_report(plateau, plateau, lr=lr) for lr in (3e-4, 3e-3)]
    d = diagnose(current, History(reports=[smaller, *lrs], noise_std=0.01), TH)
    assert d.primary == "capacity_limited"
    ev = d.get("capacity_limited").evidence
    assert ev["distinct_lrs_same_model"] == 3 and ev["report_pairs"][0]["helped"] is False


def test_capacity_needs_both_prerequisites():
    history = History(notebook=[{"action": "hparam_search"}])
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02), history, TH)
    assert d.primary == "optimization_limited"
    assert d.get("capacity_limited").evidence["missing"] == "data increase"


# --- unstable ---------------------------------------------------------------------------------


def test_unstable_nan():
    d = diagnose(make_report(plateau, plateau, nan=True), History(), TH)
    assert d.names() == ["unstable"] and d.get("unstable").confidence == 1.0


def test_unstable_divergence():
    def diverging(s):
        return 4 + 2 * math.exp(-s / 500) + 0.0003 * max(0, s - 3000)

    d = diagnose(make_report(diverging, lambda s: diverging(s) + 0.05), History(), TH)
    assert d.primary == "unstable" and d.get("unstable").evidence["diverged"] is True


def test_unstable_frequent_spikes_and_grad_norm():
    gn = {"max": 900.0, "median": 3.0, "max_over_median": 300.0}
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.02, spikes=25, grad_norm=gn), History(), TH)
    lab = d.get("unstable")
    assert lab is not None and lab.evidence["spike_count"] >= 25
    assert lab.evidence["grad_norm_max_over_median"] == 300.0


# --- inconclusive -----------------------------------------------------------------------------


def test_inconclusive_flat_with_large_stable_gap():
    d = diagnose(make_report(plateau, lambda s: plateau(s) + 0.3), History(), TH)
    assert d.primary == "inconclusive"


def test_inconclusive_too_few_points():
    train = [(0, 6.0), (100, 5.0)]
    r = make_report(plateau, plateau)
    r["summary"] = {**r["summary"], **summarize_curves(train, train, PARAMS)}
    assert diagnose(r, History(), TH).names() == ["inconclusive"]


# --- helpers ----------------------------------------------------------------------------------


def test_trend_classes():
    assert trend({"rel_change": -0.02, "t": -10}, TH) == "falling"
    assert trend({"rel_change": -0.003, "t": -10}, TH) == "crawl"
    assert trend({"rel_change": -0.02, "t": -1}, TH) == "flat"
    assert trend({"rel_change": 0.01, "t": 8}, TH) == "rising"
    assert trend({"rel_change": None, "t": None}, TH) == "unknown"


def test_thresholds_file_loads_every_field():
    loaded = Thresholds.load()
    assert loaded == Thresholds()  # toml and dataclass defaults agree
    assert ReportParams.load() == ReportParams()


def test_diagnosis_serializes():
    import json

    d = diagnose(make_report(overfit_train, overfit_val, epochs=2.5), History(), TH)
    out = json.loads(json.dumps(d.to_dict(), allow_nan=False))
    assert out["primary"] == "data_limited" and out["labels"][0]["evidence"]


def test_data_limited_under_annealing_val_still_crawls():
    """82M-token runs: 4 epochs, val creeps down (cosine to ~0), train falls faster, gap large and growing."""
    def train(s):  # tail: -0.66% (as in m4p-base-full-s1)
        return 4.40 + 2 * math.exp(-s / 400) - 0.000029 * s

    def val(s):    # tail: -0.44%, still crawling down
        return 4.90 + 2 * math.exp(-s / 400) - 0.0000213 * s

    d = diagnose(make_report(train, val, epochs=3.99, gap_noise=0.0005), History(), TH)
    lab = d.get("data_limited")
    assert lab is not None, d.summary()
    assert lab.evidence["pattern"] == "train outpaces a crawling val" and lab.confidence >= 0.85
    assert d.primary == "data_limited"


def test_data_limited_while_val_still_falls_after_several_epochs():
    """Long runs keep val falling while they overfit: train falls faster and the gap grows (p21 at 122M: 3 epochs,
    gap 0.28 growing). That is still_improving AND data_limited; before 2 epochs it is only still_improving."""
    train = lambda s: 3.8 + 3.0 * math.exp(-s / 2500)  # noqa: E731
    val = lambda s: 4.0 + 2.2 * math.exp(-s / 2500)  # noqa: E731
    d = diagnose(make_report(train, val, epochs=3.0, gap_noise=0.0005), History(), TH)
    assert d.get("still_improving") is not None
    lab = d.get("data_limited")
    assert lab is not None and lab.confidence >= 0.75 and lab.evidence["pattern"] == "overfitting while val still falls"
    early = diagnose(make_report(train, val, epochs=1.5, gap_noise=0.0005), History(), TH)
    assert early.get("data_limited") is None
