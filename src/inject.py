"""Controlled anomaly injection with an auditable ground-truth manifest."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class InjectionResult:
    frame: pd.DataFrame
    labels: np.ndarray
    manifest: pd.DataFrame
    type_by_row: Dict[int, str]


def reference_stats(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        stats[c] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "q01": float(s.quantile(0.01)),
            "q99": float(s.quantile(0.99)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return stats


def _continuous_columns(df: pd.DataFrame, columns: Iterable[str], min_unique: int = 2) -> List[str]:
    out = []
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) >= min_unique and np.isfinite(s.std(ddof=0)) and s.std(ddof=0) > 0:
            out.append(c)
    return out


def inject_anomalies(
    df: pd.DataFrame,
    n_anomalies: int,
    seed: int,
    candidate_columns: Sequence[str],
    stats: Dict[str, Dict[str, float]],
    type_probs: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    tag: str = "test",
) -> InjectionResult:
    """Replace n randomly selected rows with one of three documented anomaly types.

    Row positions and original/injected values are recorded so that every metric can
    be traced back to a concrete manipulation.
    """
    if n_anomalies <= 0:
        empty = pd.DataFrame(columns=["row_position", "original_index", "injection_type",
                                      "n_features", "features", "original_values", "injected_values"])
        return InjectionResult(df.copy(), np.zeros(len(df), dtype=int), empty, {})

    rng = np.random.default_rng(seed)
    cols = _continuous_columns(df, candidate_columns)
    if not cols:
        raise ValueError("no continuous columns available for injection")
    n_anomalies = min(n_anomalies, len(df))
    selected = rng.choice(len(df), size=n_anomalies, replace=False)
    out = df.copy()
    labels = np.zeros(len(df), dtype=int)
    records = []
    type_by_row: Dict[int, str] = {}

    for pos in selected:
        kind = str(rng.choice(["point", "multi", "extreme"], p=list(type_probs)))
        chosen: List[str] = []
        originals: List[float] = []
        injected: List[float] = []
        for _attempt in range(20):
            m = len(cols)
            if kind == "point" or m == 1:
                k = 1
            elif kind == "multi":
                low, high = min(3, m), min(5, m)
                k = int(rng.integers(low, high + 1))
            else:
                low, high = min(2, m), min(4, m)
                k = int(rng.integers(low, high + 1))
            chosen = rng.choice(cols, size=k, replace=False).tolist()
            originals, injected = [], []
            for c in chosen:
                mu, sd = stats[c]["mean"], stats[c]["std"]
                orig = float(out.at[pos, c])
                if kind == "point":
                    direction = float(rng.choice([-1.0, 1.0]))
                    new = mu + direction * float(rng.uniform(4.0, 6.0)) * sd
                elif kind == "multi":
                    direction = float(rng.choice([-1.0, 1.0]))
                    new = mu + direction * float(rng.uniform(2.0, 4.0)) * sd
                else:
                    if rng.random() < 0.5:
                        new = stats[c]["q01"] - float(rng.uniform(0.0, 0.5)) * sd
                    else:
                        new = stats[c]["q99"] + float(rng.uniform(0.0, 0.5)) * sd
                originals.append(orig)
                injected.append(float(new))
            if any(abs(a - b) > 1e-9 for a, b in zip(originals, injected)):
                break
        for c, v in zip(chosen, injected):
            out.at[pos, c] = v
        labels[pos] = 1
        type_by_row[int(pos)] = kind
        records.append({
            "row_position": int(pos),
            "original_index": int(df.index[pos]),
            "injection_type": kind,
            "n_features": len(chosen),
            "features": "|".join(chosen),
            "original_values": "|".join(f"{v:.6g}" for v in originals),
            "injected_values": "|".join(f"{v:.6g}" for v in injected),
            "seed": seed,
            "tag": tag,
        })
    return InjectionResult(out, labels, pd.DataFrame(records), type_by_row)


def injection_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["injection_type", "count", "share"])
    counts = manifest["injection_type"].value_counts().rename_axis("injection_type").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts
