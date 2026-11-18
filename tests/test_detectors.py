import numpy as np

from detectors import (IForestDetector, MahalanobisDetector, build_detector)


def test_detectors_score_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    for name, params in [("Z-Score", {}), ("Mahalanobis", {"support_fraction": 0.75}),
                         ("LOF", {"n_neighbors": 10}),
                         ("Isolation Forest", {"n_estimators": 50, "max_samples": 1.0, "max_features": 1.0}),
                         ("One-Class SVM", {"nu": 0.05, "gamma": "scale"})]:
        m = build_detector(name, params, 42).fit(X)
        s = m.score(X[:15])
        assert s.shape == (15,)
        assert np.isfinite(s).all()


def test_iforest_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    a = IForestDetector(50, 1.0, 1.0, 42).fit(X).score(X)
    b = IForestDetector(50, 1.0, 1.0, 42).fit(X).score(X)
    assert np.allclose(a, b)


def test_mahalanobis_singular_fallback():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4))
    X = np.column_stack([X, X[:, 0], X[:, 1]])  # rank-deficient
    m = MahalanobisDetector(0.75, 42).fit(X)
    s = m.score(X)
    assert np.isfinite(s).all()
