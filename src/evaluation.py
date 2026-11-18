"""Thresholded and threshold-free evaluation metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "accuracy": (tp + tn) / max(len(y_true), 1),
    }


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def per_type_recall(manifest: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["injection_type", "n", "recall"])
    rows = []
    for kind, grp in manifest.groupby("injection_type"):
        pos = grp["row_position"].astype(int).to_numpy()
        rows.append({"injection_type": kind, "n": len(pos), "recall": float(np.mean(y_pred[pos]))})
    order = {"point": 0, "multi": 1, "extreme": 2}
    return pd.DataFrame(rows).sort_values("injection_type", key=lambda s: s.map(order)).reset_index(drop=True)


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean/std for every numeric key across seeds."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    out: Dict[str, Dict[str, float]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = {
                "mean": float(df[c].mean()),
                "std": float(df[c].std(ddof=1)) if len(df) > 1 else 0.0,
            }
    return out


def fmt_mean_std(agg: Dict[str, Dict[str, float]], key: str, pct: bool = False) -> str:
    if key not in agg:
        return "-"
    m, s = agg[key]["mean"], agg[key]["std"]
    if pct:
        return f"{m * 100:.2f}±{s * 100:.2f}%"
    return f"{m:.4f}±{s:.4f}"
