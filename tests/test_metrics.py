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
