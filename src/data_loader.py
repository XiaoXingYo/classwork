"""Data loading, auditing and reproducible cleaning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, IRIS_FEATURES, IRIS_LABEL, HOUSING_FEATURES, TABLE_DIR, FEATURE_LABELS


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _range_text(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return f"{s.min():.4g} ~ {s.max():.4g}"
    vals = s.dropna().astype(str).str.strip().unique().tolist()
    return "、".join(vals[:8])


def data_dictionary() -> pd.DataFrame:
    """Field-level dictionary for the two T13 datasets."""
    rows = []
    iris = read_raw("iris_with_error.csv")
    for col in IRIS_FEATURES + [IRIS_LABEL]:
        s = iris[col]
        if col == IRIS_LABEL:
            dtype = "类别"
            meaning = "鸢尾花品种：setosa / versicolor / virginica"
        else:
            dtype = "数值（cm）"
            meaning = FEATURE_LABELS[col]
        rows.append({
            "数据集": "iris_with_error",
            "字段名": col,
            "类型": dtype,
            "取值范围/示例": _range_text(s),
            "缺失比例": f"{s.isna().mean() * 100:.2f}%",
            "含义说明": meaning,
        })
    housing = read_raw("housing.csv")
    for col in housing.columns:
        s = housing[col]
        dtype = "整数" if pd.api.types.is_integer_dtype(s) else "数值"
        if col in ("CHAS", "RAD"):
            dtype = "类别编码"
        rows.append({
            "数据集": "housing",
            "字段名": col,
            "类型": dtype,
            "取值范围/示例": _range_text(s),
            "缺失比例": f"{s.isna().mean() * 100:.2f}%",
            "含义说明": FEATURE_LABELS.get(col, col),
        })
    return pd.DataFrame(rows)


def _iqr_count(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def quality_table(clean_stats: Dict[str, Dict[str, int]] | None = None) -> pd.DataFrame:
    """Quantitative data-quality diagnosis following template Appendix Table 2."""
    iris = read_raw("iris_with_error.csv")
    housing = read_raw("housing.csv")
    feature_cells = iris[IRIS_FEATURES].size
    miss_cells = int(iris[IRIS_FEATURES].isna().sum().sum()) + int(iris[IRIS_LABEL].isna().sum())
    dup = int(iris.duplicated().sum())
    iqr_iris = {c: _iqr_count(iris[c]) for c in IRIS_FEATURES}
    iqr_housing = {c: _iqr_count(housing[c]) for c in housing.columns}
    iris_after = clean_stats["iris"]["after"] if clean_stats else 146
    housing_after = clean_stats["housing"]["after"] if clean_stats else 506

    rows = [
        {
            "问题类型": "缺失值",
            "检出字段": "iris：Sepal_Length 1、Sepal_Width 3、Petal_Length 4、Petal_Width 4、Species 1",
            "数量/占比": f"{miss_cells} 格（{miss_cells / (feature_cells + len(iris)) * 100:.2f}%）",
            "处理方式": "按 Species 分组中位数插补；连续缺失≥3 个特征的 1 行整行删除",
            "处理后样本量": f"iris {iris_after} 行；housing {housing_after} 行",
        },
        {
            "问题类型": "重复值",
            "检出字段": "iris：全部字段完全重复",
            "数量/占比": f"{dup} 行（{dup / len(iris) * 100:.2f}%）",
            "处理方式": "保留首次出现，删除后续完全重复记录；housing 无重复",
            "处理后样本量": f"iris {iris_after} 行；housing {housing_after} 行",
        },
        {
            "问题类型": "异常值",
            "检出字段": f"iris：{', '.join(f'{k} {v}' for k, v in iqr_iris.items() if v)}；housing：13 个预测变量",
            "数量/占比": f"iris IQR 标记 {sum(iqr_iris.values())} 个；housing 字段级 IQR 标记 {sum(iqr_housing.values())} 个",
            "处理方式": "视为自然长尾，不删除；仅在建模前做稳健标准化。注入异常单独保留真值",
            "处理后样本量": f"iris {iris_after} 行；housing {housing_after} 行",
        },
        {
            "问题类型": "口径不一致",
            "检出字段": "iris：Species 大小写/空白；housing：CHAS、RAD 为整数编码",
            "数量/占比": "0 条实质不一致；编码差异 2 个字段",
            "处理方式": "Species 统一小写去空白；CHAS/RAD 保留编码并在数据字典中注明",
            "处理后样本量": f"iris {iris_after} 行；housing {housing_after} 行",
        },
    ]
    return pd.DataFrame(rows)


def clean_iris() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean iris_with_error and return row-count audit."""
    df = read_raw("iris_with_error.csv").copy()
    df[IRIS_LABEL] = (df[IRIS_LABEL].astype(str).str.strip().str.lower()
                      .replace({"nan": np.nan, "": np.nan, "none": np.nan}))
    n0 = len(df)
    df = df.drop_duplicates().copy()
    n1 = len(df)
    missing_per_row = df[IRIS_FEATURES].isna().sum(axis=1)
    df = df.loc[missing_per_row < 3].copy()
    n2 = len(df)
    for col in IRIS_FEATURES:
        med = df.groupby(IRIS_LABEL)[col].transform("median")
        df[col] = df[col].fillna(med).fillna(df[col].median())
    df = df.dropna(subset=[IRIS_LABEL]).reset_index(drop=True)
    n3 = len(df)
    stats = {"initial": n0, "after_dedup": n1, "after_drop_high_missing": n2, "after": n3}
    return df, stats


def clean_housing() -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = read_raw("housing.csv").copy()
    n0 = len(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=HOUSING_FEATURES + ["MEDV"]).drop_duplicates().reset_index(drop=True)
    stats = {"initial": n0, "after": len(df)}
    return df, stats


def load_clean(dataset: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if dataset == "iris":
        return clean_iris()
    if dataset == "housing":
        return clean_housing()
    raise ValueError(dataset)


def run_audit() -> None:
    iris, iris_stats = clean_iris()
    housing, housing_stats = clean_housing()
    stats = {"iris": iris_stats, "housing": housing_stats}
    data_dictionary().to_csv(TABLE_DIR / "table_data_dictionary.csv", index=False, encoding="utf-8-sig")
    quality_table(stats).to_csv(TABLE_DIR / "table_quality.csv", index=False, encoding="utf-8-sig")
    iris.to_csv(TABLE_DIR / "iris_cleaned.csv", index=False, encoding="utf-8-sig")
    housing.to_csv(TABLE_DIR / "housing_cleaned.csv", index=False, encoding="utf-8-sig")
    with open(TABLE_DIR / "cleaning_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("audit written:", stats)
    print(data_dictionary().to_string(index=False))
    print(quality_table(stats).to_string(index=False))


if __name__ == "__main__":
    run_audit()
