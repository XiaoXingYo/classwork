"""Unsupervised anomaly detectors with a shared tuning protocol."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
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
        nus = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
        gammas = ("scale", 0.01, 0.05, 0.1, 0.5)
        return [{"nu": n, "gamma": g} for n in nus for g in gammas]
    raise ValueError(name)


# Documented scikit-learn defaults, used to quantify what tuning actually buys.
# Mahalanobis uses EmpiricalCovariance (support_fraction=None, i.e. the full fit
# set) by default; IsolationForest's max_samples="auto" also equals the full fit
# sample for both datasets here, so both map onto the grid points used below.
SKLEARN_DEFAULTS: Dict[str, Dict] = {
    "Z-Score": {},
    "Mahalanobis": {"support_fraction": 1.0},
    "LOF": {"n_neighbors": 20},
    "Isolation Forest": {"n_estimators": 100, "max_samples": 1.0, "max_features": 1.0},
    "One-Class SVM": {"nu": 0.5, "gamma": "scale"},
}


def default_params(name: str) -> Dict:
    return dict(SKLEARN_DEFAULTS[name])


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
    cv_f1: float = float("nan")
    cv_f1_std: float = 0.0
    cv_f1_worst: float = float("nan")
    cv_f1_range: float = 0.0
    worst_params: Dict = field(default_factory=dict)
    n_candidates: int = 1


def evaluate_params(name: str, params: Dict, X_fit: np.ndarray, X_cal: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, seed: int) -> TunedDetector:
    """Fit one parameter setting and calibrate its threshold on the val anomalies."""
    t0 = time.perf_counter()
    model = build_detector(name, params, seed).fit(X_fit)
    fit_seconds = time.perf_counter() - t0
    s_cal = model.score(X_cal)
    s_val = model.score(X_val)
    tau, sel = select_threshold(s_cal, s_val, y_val)
    vm = binary_metrics(y_val, s_val, tau)
    sm = score_metrics(y_val, s_val)
    return TunedDetector(
        name=name, params=dict(params), threshold=float(tau), model=model,
        val_metrics={**vm, "cal_fpr": sel["cal_fpr"]},
        val_pr_auc=float(sm.get("pr_auc", 0.0)),
        fit_seconds=fit_seconds, score_seconds=0.0,
    )


def cv_score(name: str, params: Dict, folds: Sequence[Dict], seed: int,
             max_fpr: float = 0.10) -> Dict:
    """Cross-validated F1 / PR-AUC of a single parameter setting."""
    f1s: List[float] = []
    prs: List[float] = []
    for fold in folds:
        try:
            model = build_detector(name, params, seed).fit(fold["X_train"])
            s_cal = model.score(fold["X_cal"])
            s_hold = model.score(fold["X_hold"])
        except Exception:
            return {"cv_f1_mean": float("nan"), "cv_f1_std": float("nan"),
                    "cv_f1_min": float("nan"), "cv_pr_auc_mean": float("nan")}
        tau, _ = select_threshold(s_cal, s_hold, fold["y_hold"], max_fpr=max_fpr)
        f1s.append(binary_metrics(fold["y_hold"], s_hold, tau)["f1"])
        pr = score_metrics(fold["y_hold"], s_hold)["pr_auc"]
        if np.isfinite(pr):
            prs.append(pr)
    return {
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s, ddof=0)),
        "cv_f1_min": float(np.min(f1s)),
        "cv_pr_auc_mean": float(np.mean(prs)) if prs else 0.0,
    }


def cv_select(name: str, folds: Sequence[Dict], dataset: str, seed: int,
              grid: Optional[Sequence[Dict]] = None, max_fpr: float = 0.10
              ) -> Tuple[Dict, Dict, List[Dict]]:
    """Rank the grid by k-fold cross-validation on the normal training rows.

    Every fold fits on 2/3 of the normal rows, calibrates the threshold on a
    held-out slice of that normal part, then scores the remaining normal rows
    together with a fold-specific synthetic injection. Folds are built from the
    training split only, so neither the validation nor the test labels are used
    to choose a hyper-parameter.

    Returns (best_row, worst_row, all_rows).
    """
    grid = list(grid) if grid is not None else default_grid(dataset, name)
    rows: List[Dict] = []
    for params in grid:
        r = cv_score(name, params, folds, seed, max_fpr)
        if not np.isfinite(r["cv_f1_mean"]):
            continue
        rows.append({"params": dict(params), **r})
    if not rows:
        raise RuntimeError(f"cross-validation failed for {name}")
    ordered = sorted(rows, key=lambda r: (r["cv_f1_mean"], r["cv_pr_auc_mean"],
                                          -r["cv_f1_std"]), reverse=True)
    return ordered[0], ordered[-1], rows


def _select_on_validation(name: str, grid: Sequence[Dict], X_fit, X_cal, X_val,
                          y_val, seed: int) -> Dict:
    """Fallback protocol: pick the setting that wins the single validation split."""
    best_key, best = None, None
    for params in grid:
        try:
            t = evaluate_params(name, params, X_fit, X_cal, X_val, y_val, seed)
        except Exception:
            continue
        key = (t.val_metrics["f1"], t.val_pr_auc, t.val_metrics["recall"],
               -t.val_metrics["cal_fpr"])
        if best_key is None or key > best_key:
            best_key, best = key, dict(params)
    if best is None:
        raise RuntimeError(f"all parameter settings failed for {name}")
    return best


def tune_detector(name: str, X_fit: np.ndarray, X_cal: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  dataset: str, seed: int,
                  grid: Optional[Sequence[Dict]] = None,
                  folds: Optional[Sequence[Dict]] = None) -> TunedDetector:
    """Select hyper-parameters (CV when folds are given), then calibrate a threshold."""
    grid = list(grid) if grid is not None else default_grid(dataset, name)
    if folds:
        best, worst, rows = cv_select(name, folds, dataset, seed, grid)
        tuned = evaluate_params(name, best["params"], X_fit, X_cal, X_val, y_val, seed)
        tuned.cv_f1 = best["cv_f1_mean"]
        tuned.cv_f1_std = best["cv_f1_std"]
        tuned.cv_f1_worst = worst["cv_f1_mean"]
        tuned.cv_f1_range = best["cv_f1_mean"] - worst["cv_f1_mean"]
        tuned.worst_params = dict(worst["params"])
        tuned.n_candidates = len(rows)
    else:
        params = _select_on_validation(name, grid, X_fit, X_cal, X_val, y_val, seed)
        tuned = evaluate_params(name, params, X_fit, X_cal, X_val, y_val, seed)
        tuned.cv_f1 = tuned.val_metrics["f1"]
        tuned.cv_f1_range = float("nan")
        tuned.n_candidates = len(grid)
    return tuned
