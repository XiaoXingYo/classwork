"""Deterministic feature construction, selection and scaling."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import HOUSING_FEATURES, IRIS_FEATURES


def add_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if dataset == "iris":
        out["Petal_Area"] = out["Petal_Length"] * out["Petal_Width"]
        out["Sepal_Area"] = out["Sepal_Length"] * out["Sepal_Width"]
        out["Petal_Aspect"] = out["Petal_Length"] / out["Petal_Width"].replace(0, np.nan)
        out["Sepal_Aspect"] = out["Sepal_Length"] / out["Sepal_Width"].replace(0, np.nan)
        out["Petal_Sepal_Ratio"] = out["Petal_Length"] / out["Sepal_Length"].replace(0, np.nan)
    elif dataset == "housing":
        out["CRIM_log"] = np.log1p(out["CRIM"].clip(lower=0))
        out["LSTAT_log"] = np.log1p(out["LSTAT"].clip(lower=0))
        out["TAX_log"] = np.log1p(out["TAX"].clip(lower=0))
        out["RM2"] = out["RM"] ** 2
        out["LSTAT_RM"] = out["LSTAT"] / out["RM"].replace(0, np.nan)
        out["NOX_DIS"] = out["NOX"] * out["DIS"]
        out["TAX_RAD"] = out["TAX"] / (out["RAD"] + 1.0)
        out["AGE_LSTAT"] = out["AGE"] * out["LSTAT"]
        out["CRIM_LSTAT"] = out["CRIM"] * out["LSTAT"]
        out["PTRATIO_LSTAT"] = out["PTRATIO"] * out["LSTAT"]
    else:
        raise ValueError(dataset)
    return out.replace([np.inf, -np.inf], np.nan).fillna(out.median(numeric_only=True))


def feature_columns(dataset: str, mode: str = "engineered") -> List[str]:
    base = IRIS_FEATURES if dataset == "iris" else HOUSING_FEATURES
    if mode == "base":
        return list(base)
    engineered = {
        "iris": ["Petal_Area", "Sepal_Area", "Petal_Aspect", "Sepal_Aspect", "Petal_Sepal_Ratio"],
        "housing": ["CRIM_log", "LSTAT_log", "TAX_log", "RM2", "LSTAT_RM",
                    "NOX_DIS", "TAX_RAD", "AGE_LSTAT", "CRIM_LSTAT", "PTRATIO_LSTAT"],
    }[dataset]
    return list(base) + engineered


def variance_filter(X: pd.DataFrame, threshold: float = 1e-10) -> List[str]:
    keep = []
    for c in X.columns:
        v = X[c].var(ddof=0)
        if np.isfinite(v) and v > threshold:
            keep.append(c)
    return keep


def correlation_filter(X: pd.DataFrame, columns: Iterable[str], threshold: float = 0.95) -> List[str]:
    cols = list(columns)
    if len(cols) <= 1:
        return cols
    corr = X[cols].corr().abs()
    keep: List[str] = []
    for c in cols:
        if not keep:
            keep.append(c)
            continue
        if float(corr.loc[c, keep].max()) <= threshold:
            keep.append(c)
    return keep


def select_features(X_fit: pd.DataFrame, columns: Iterable[str], corr_threshold: float = 0.95) -> List[str]:
    cols = variance_filter(X_fit[list(columns)])
    return correlation_filter(X_fit, cols, threshold=corr_threshold)


def fit_scaler(X: pd.DataFrame, method: str = "robust"):
    scaler = RobustScaler() if method == "robust" else StandardScaler()
    scaler.fit(X)
    return scaler


def transform(X: pd.DataFrame, scaler, columns: List[str]) -> np.ndarray:
    arr = scaler.transform(X[columns])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
