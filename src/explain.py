"""Model-agnostic-ish feature contribution and error-case analysis."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config import OUT_DIR, TABLE_DIR
from detectors import IForestDetector, MahalanobisDetector, build_detector, tune_detector
from evaluation import binary_metrics
from experiment import prepare_seed
from features import add_features, feature_columns, fit_scaler, select_features, transform

ART = OUT_DIR / "artifacts"


def _fit_best(dataset: str, seed: int = 42):
    p = prepare_seed(dataset, seed, "engineered")
    X = p["X"]
    mb = tune_detector("Mahalanobis", X["fit"], X["cal"], X["val"], p["y_val"], dataset, seed)
    rf = tune_detector("Isolation Forest", X["fit"], X["cal"], X["val"], p["y_val"], dataset, seed)
    return p, mb, rf


def mahalanobis_contributions(model: MahalanobisDetector, X: np.ndarray,
                              names: List[str]) -> pd.DataFrame:
    mu = model.estimator.location_
    prec = model.estimator.precision_
    delta = X - mu
    contrib = delta * (delta @ prec)
    contrib = np.maximum(contrib, 0.0)
    denom = contrib.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    share = contrib / denom
    return pd.DataFrame({
        "feature": names,
        "mean_contribution": contrib.mean(axis=0),
        "mean_share": share.mean(axis=0),
    }).sort_values("mean_contribution", ascending=False).reset_index(drop=True)


def iforest_contributions(model: IForestDetector, X: np.ndarray, X_fit: np.ndarray,
                          names: List[str]) -> pd.DataFrame:
    baseline = model.score(X)
    med = np.median(X_fit, axis=0)
    rows = []
    for j, name in enumerate(names):
        Xp = X.copy()
        Xp[:, j] = med[j]
        rows.append({"feature": name, "mean_contribution": float(np.mean(baseline - model.score(Xp)))})
    out = pd.DataFrame(rows)
    out["mean_share"] = out["mean_contribution"].clip(lower=0) / max(out["mean_contribution"].clip(lower=0).sum(), 1e-12)
    return out.sort_values("mean_contribution", ascending=False).reset_index(drop=True)


def error_cases(p, model, threshold: float, dataset: str = "housing") -> pd.DataFrame:
    X_test = p["X"]["test"]
    s = model.score(X_test)
    pred = (s >= threshold).astype(int)
    y = p["y_test"]
    raw = p["test_injection"].frame.reset_index(drop=True)
    types = p["type_test"]
    kind = np.where(y == 1, types, "normal")
    rows = []
    for i in range(len(y)):
        if pred[i] != y[i]:
            rows.append({
                "row_position": i,
                "original_index": int(p["test_injection"].manifest.set_index("row_position").loc[i, "original_index"])
                if i in set(p["test_injection"].manifest["row_position"]) else -1,
                "label": int(y[i]), "prediction": int(pred[i]),
                "error_type": "漏报(FN)" if y[i] == 1 else "误报(FP)",
                "injection_type": kind[i], "score": float(s[i]), "threshold": float(threshold),
                **{c: float(raw.at[i, c]) for c in ["CRIM", "ZN", "INDUS", "RM", "AGE", "DIS", "TAX", "PTRATIO", "LSTAT", "MEDV"]
                   if c in raw.columns},
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["margin"] = (df["score"] - threshold).abs()
        df = df.sort_values(["error_type", "margin"], ascending=[True, False])
    return df


def main() -> None:
    p, mb, rf = _fit_best("housing", 42)
    X_fit, X_test = p["X"]["fit"], p["X"]["test"]
    names = p["selected"]
    y_pred_mb = (mb.model.score(X_test) >= mb.threshold).astype(int)
    idx = np.where(y_pred_mb == 1)[0]
    mb_contrib = mahalanobis_contributions(mb.model, X_test[idx], names)
    rf_contrib = iforest_contributions(rf.model, X_test[idx], X_fit, names)
    mb_contrib["method"] = "Mahalanobis"
    rf_contrib["method"] = "Isolation Forest"
    contrib = pd.concat([mb_contrib, rf_contrib], ignore_index=True)
    contrib.to_csv(TABLE_DIR / "table_feature_contrib.csv", index=False, encoding="utf-8-sig")

    err = error_cases(p, mb.model, mb.threshold, "housing")
    err.to_csv(TABLE_DIR / "table_error_cases.csv", index=False, encoding="utf-8-sig")

    with open(ART / "explain.json", "w", encoding="utf-8") as f:
        json.dump({
            "mahalanobis": mb_contrib.head(10).to_dict("records"),
            "iforest": rf_contrib.head(10).to_dict("records"),
            "n_detected": int(len(idx)),
            "threshold": float(mb.threshold),
            "errors": err.head(10).to_dict("records"),
        }, f, ensure_ascii=False, indent=2, default=str)
    print("feature contribution:")
    print(contrib.groupby("method").head(6).to_string(index=False))
    print("\nerror cases:", len(err))
    print(err.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
