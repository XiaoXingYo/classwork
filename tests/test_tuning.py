import numpy as np

from detectors import cv_score, cv_select, default_grid, default_params, tune_detector


def _folds(rng, n_folds=3, n_normal=60, n_anom=6, dim=4, center=0.0):
    folds = []
    for _ in range(n_folds):
        train = rng.normal(size=(n_normal, dim))
        cal = rng.normal(size=(n_normal // 3, dim))
        hold_normal = rng.normal(size=(n_normal // 3, dim))
        # Shifted points stand in for the synthetic anomalies of a fold.
        hold_anom = rng.normal(size=(n_anom, dim)) + center
        folds.append({
            "X_train": train, "X_cal": cal,
            "X_hold": np.vstack([hold_normal, hold_anom]),
            "y_hold": np.r_[np.zeros(len(hold_normal), dtype=int), np.ones(n_anom, dtype=int)],
        })
    return folds


def test_ocsvm_grid_covers_sklearn_default():
    grid = default_grid("housing", "One-Class SVM")
    nus = {g["nu"] for g in grid}
    assert 0.5 in nus and 0.7 in nus
    assert default_params("One-Class SVM") in grid


def test_every_default_setting_is_inside_its_grid():
    for name in ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM"]:
        assert default_params(name) in default_grid("housing", name), name


def test_cv_select_ranks_by_cross_validated_f1():
    rng = np.random.default_rng(3)
    folds = _folds(rng)
    best, worst, rows = cv_select("LOF", folds, "housing", 42,
                                 grid=[{"n_neighbors": 3}, {"n_neighbors": 40}])
    assert len(rows) == 2
    assert best["cv_f1_mean"] >= worst["cv_f1_mean"]
    assert best is not worst


def test_cv_score_penalises_a_blind_detector():
    """A detector that cannot separate the shifted points must score lower."""
    rng = np.random.default_rng(4)
    separable = _folds(rng, center=8.0)
    overlap = _folds(rng, center=0.0)
    good = cv_score("Mahalanobis", {"support_fraction": 1.0}, separable, 42)["cv_f1_mean"]
    bad = cv_score("Mahalanobis", {"support_fraction": 1.0}, overlap, 42)["cv_f1_mean"]
    assert np.isfinite(good) and np.isfinite(bad)
    assert good > bad


def test_tune_detector_uses_cv_not_the_validation_split():
    rng = np.random.default_rng(5)
    folds = _folds(rng, center=8.0)
    X_fit = rng.normal(size=(90, 4))
    X_cal = rng.normal(size=(30, 4))
    # Validation anomalies sit far away, so any setting looks good there.
    X_val = np.vstack([rng.normal(size=(20, 4)), rng.normal(size=(6, 4)) + 8.0])
    y_val = np.r_[np.zeros(20, dtype=int), np.ones(6, dtype=int)]
    grid = [{"n_neighbors": 5}, {"n_neighbors": 45}]
    tuned = tune_detector("LOF", X_fit, X_cal, X_val, y_val, "housing", 42,
                          grid=grid, folds=folds)
    best, _, _ = cv_select("LOF", folds, "housing", 42, grid=grid)
    assert tuned.params == best["params"]
    assert tuned.n_candidates == 2
    assert tuned.cv_f1 == best["cv_f1_mean"]
    assert tuned.cv_f1_range >= 0.0
    assert 0.0 <= tuned.threshold <= np.inf
