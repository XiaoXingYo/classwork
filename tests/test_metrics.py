import numpy as np

from detectors import select_threshold
from evaluation import binary_metrics


def test_binary_metrics_known_case():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2])
    m = binary_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 2, 0)
    assert abs(m["precision"] - 0.5) < 1e-12
    assert abs(m["recall"] - 1.0) < 1e-12
    assert abs(m["fpr"] - 1 / 3) < 1e-12


def test_threshold_respects_calibration_fpr():
    rng = np.random.default_rng(0)
    s_cal = rng.normal(size=200)
    s_val = np.concatenate([rng.normal(size=40), rng.normal(loc=6, size=10)])
    y_val = np.array([0] * 40 + [1] * 10)
    tau, info = select_threshold(s_cal, s_val, y_val, max_fpr=0.10)
    assert info["cal_fpr"] <= 0.10 + 1e-12
    assert np.isfinite(tau)


def test_threshold_prefers_recall_within_safety_budget():
    """The usable budget is max_fpr * 0.8; inside it recall comes first."""
    from detectors import threshold_candidates

    rng = np.random.default_rng(1)
    s_cal = rng.normal(size=200)
    s_val = np.concatenate([rng.normal(size=40), rng.normal(loc=4, size=10)])
    y_val = np.array([0] * 40 + [1] * 10)
    tau, info = select_threshold(s_cal, s_val, y_val, max_fpr=0.10)
    assert info["cal_fpr"] <= 0.08 + 1e-12
    best_recall = max(
        binary_metrics(y_val, s_val, t)["recall"]
        for t in threshold_candidates(s_cal)
        if np.mean(s_cal >= t) <= 0.08 + 1e-12
    )
    assert info["recall"] >= best_recall - 1e-12
