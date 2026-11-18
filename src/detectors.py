"""Unsupervised anomaly detectors with a shared tuning protocol."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from evaluation import binary_metrics, score_metrics


class BaseDetector:
    name = "base"

    def fit(self, X: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZScoreDetector(BaseDetector):
    name = "Z-Score"

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        self.mu = np.mean(X, axis=0)
        self.sd = np.std(X, axis=0, ddof=0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.max(np.abs((X - self.mu) / self.sd), axis=1)


class MahalanobisDetector(BaseDetector):
    name = "Mahalanobis"

    def __init__(self, support_fraction: float = 0.75, random_state: int = 42):
        self.support_fraction = float(support_fraction)
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        self.estimator = None
        self._mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov) + np.eye(X.shape[1]) * 1e-6
        self._prec = np.linalg.pinv(cov)
        try:
            if self.support_fraction >= 0.999:
                self.estimator = EmpiricalCovariance().fit(X)
            else:
                self.estimator = MinCovDet(
                    support_fraction=self.support_fraction,
                    random_state=self.random_state,
                ).fit(X)
        except Exception:
            self.estimator = None
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is not None:
            try:
                d = self.estimator.mahalanobis(X)
                return np.sqrt(np.maximum(d, 0.0))
            except Exception:
                pass
        delta = X - self._mu
        d = np.einsum("ij,jk,ik->i", delta, self._prec, delta)
        return np.sqrt(np.maximum(d, 0.0))


class LOFDetector(BaseDetector):
    name = "LOF"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = int(n_neighbors)

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, contamination="auto"
        )
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class IForestDetector(BaseDetector):
    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 200, max_samples=1.0,
                 max_features=1.0, random_state: int = 42):
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "IForestDetector":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class OCSVMDetector(BaseDetector):
    name = "One-Class SVM"

    def __init__(self, nu: float = 0.05, gamma="scale"):
        self.nu = float(nu)
        self.gamma = gamma

    def fit(self, X: np.ndarray) -> "OCSVMDetector":
        self.model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


def build_detector(name: str, params: Dict, seed: int) -> BaseDetector:
    if name == "Z-Score":
        return ZScoreDetector()
    if name == "Mahalanobis":
        return MahalanobisDetector(support_fraction=params.get("support_fraction", 0.75),
                                   random_state=seed)
    if name == "LOF":
        return LOFDetector(n_neighbors=params.get("n_neighbors", 20))
    if name == "Isolation Forest":
        return IForestDetector(
            n_estimators=params.get("n_estimators", 200),
            max_samples=params.get("max_samples", 1.0),
            max_features=params.get("max_features", 1.0),
            random_state=seed,
        )
    if name == "One-Class SVM":
        return OCSVMDetector(nu=params.get("nu", 0.05), gamma=params.get("gamma", "scale"))
    raise ValueError(name)


def default_grid(dataset: str, name: str) -> List[Dict]:
    if name == "Z-Score":
        return [{}]
    if name == "Mahalanobis":
        return [{"support_fraction": v} for v in (0.5, 0.6, 0.75, 0.9, 1.0)]
    if name == "LOF":
        ks = (5, 10, 15, 20, 30, 50) if dataset == "housing" else (3, 5, 8, 10, 15, 20)
        return [{"n_neighbors": k} for k in ks]
    if name == "Isolation Forest":
        if dataset == "iris":
            samples = (0.5, 0.75, 1.0)
            features = (1.0,)
        else:
            samples = (0.5, 0.75, 1.0)
            features = (0.5, 1.0)
        return [{"n_estimators": n, "max_samples": s, "max_features": f}
                for n in (100, 200, 300) for s in samples for f in features]
    if name == "One-Class SVM":
        nus = (0.01, 0.05, 0.1, 0.2, 0.3)
        gammas = ("scale", 0.01, 0.05, 0.1, 0.5)
        return [{"nu": n, "gamma": g} for n in nus for g in gammas]
    raise ValueError(name)


def threshold_candidates(s_cal: np.ndarray) -> np.ndarray:
    s = np.asarray(s_cal, dtype=float)
    qs = np.linspace(0.80, 0.995, 40)
    taus = np.unique(np.quantile(s, qs))
    taus = taus[np.isfinite(taus)]
    if len(taus) == 0:
        return np.array([0.0])
    return taus


def select_threshold(s_cal: np.ndarray, s_val: np.ndarray,
                     y_val: np.ndarray, max_fpr: float = 0.10) -> Tuple[float, Dict]:
    """Choose a threshold on normal calibration scores, constrained by FPR.

    Validation anomalies are synthetic and independent from the test injection;
    the test labels are never used here.
    """
    best = None
    for tau in threshold_candidates(s_cal):
        cal_fpr = float(np.mean(s_cal >= tau))
        if cal_fpr > max_fpr + 1e-12:
            continue
        m = binary_metrics(y_val, s_val, tau)
        key = (m["f1"], m["recall"], -cal_fpr, m["precision"])
        if best is None or key > best[0]:
            best = (key, float(tau), m, cal_fpr)
    if best is None:
        tau = float(np.quantile(s_cal, 0.90))
        m = binary_metrics(y_val, s_val, tau)
        best = ((m["f1"], m["recall"], -1.0, m["precision"]), tau, m,
                float(np.mean(s_cal >= tau)))
    _, tau, m, cal_fpr = best
    return tau, {"cal_fpr": cal_fpr, **m}


@dataclass
class TunedDetector:
    name: str
    params: Dict
    threshold: float
    model: BaseDetector
    val_metrics: Dict[str, float]
    val_pr_auc: float
    fit_seconds: float
    score_seconds: float


def tune_detector(name: str, X_fit: np.ndarray, X_cal: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  dataset: str, seed: int,
                  grid: Optional[Sequence[Dict]] = None) -> TunedDetector:
    grid = list(grid) if grid is not None else default_grid(dataset, name)
    best: Optional[TunedDetector] = None
    best_key = None
    for params in grid:
        t0 = time.perf_counter()
        try:
            model = build_detector(name, params, seed).fit(X_fit)
            s_cal = model.score(X_cal)
            s_val = model.score(X_val)
        except Exception:
            continue
        fit_seconds = time.perf_counter() - t0
        tau, sel = select_threshold(s_cal, s_val, y_val)
        vm = binary_metrics(y_val, s_val, tau)
        sm = score_metrics(y_val, s_val)
        key = (vm["f1"], sm.get("pr_auc", 0.0), vm["recall"], -sel["cal_fpr"])
        cand = TunedDetector(
            name=name, params=dict(params), threshold=float(tau), model=model,
            val_metrics={**vm, "cal_fpr": sel["cal_fpr"]},
            val_pr_auc=float(sm.get("pr_auc", 0.0)),
            fit_seconds=fit_seconds, score_seconds=0.0,
        )
        if best_key is None or key > best_key:
            best_key, best = key, cand
    if best is None:
        raise RuntimeError(f"all parameter settings failed for {name}")
    return best
