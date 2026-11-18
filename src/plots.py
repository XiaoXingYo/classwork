"""All report figures (300 dpi PNG)."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Font 'default'.*")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)

from config import FIG_DIR, OUT_DIR, TABLE_DIR
from data_loader import load_clean, read_raw
from config import IRIS_FEATURES

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

ART = OUT_DIR / "artifacts"
METHODS = ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM", "Rank-Ensemble"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def save(fig, name: str):
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_route():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.2, 2.6, "数据审计\n缺失/重复/长尾"), (2.05, 2.6, "清洗与划分\nfit/calib/val/test"),
        (3.9, 2.6, "受控异常注入\n点/协同/极端"), (5.75, 2.6, "特征工程\n派生+筛选+缩放"),
        (7.6, 2.6, "五类检测器\n统计/密度/树/边界"),
        (2.05, 0.6, "阈值选择\nFPR≤10%"), (3.9, 0.6, "测试评估\nPR-AUC/Recall/FPR"),
        (5.75, 0.6, "参数敏感性\n热图"), (7.6, 0.6, "特征贡献\n与误差剖析"),
    ]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.55, 0.85, boxstyle="round,pad=0.05",
                                    fc="#EAF2F8", ec="#2C3E50", lw=1.2))
        ax.text(x + 0.775, y + 0.425, text, ha="center", va="center", fontsize=9)
    arrows = [((1.75, 3.02), (2.05, 3.02)), ((3.6, 3.02), (3.9, 3.02)),
              ((5.45, 3.02), (5.75, 3.02)), ((7.35, 3.02), (7.6, 3.02)),
              ((8.35, 2.6), (8.35, 1.45)), ((7.6, 1.02), (5.75, 1.02)),
              ((5.45, 1.02), (3.9, 1.02)), ((3.6, 1.02), (2.82, 1.45))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color="#2C3E50", lw=1.2))
    ax.text(5, 3.85, "T13 多元异常检测：技术路线", ha="center", fontsize=14, weight="bold")
    save(fig, "fig01_technical_route.png")


def fig_eda():
    housing, _ = load_clean("housing")
    iris, _ = load_clean("iris")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(housing["MEDV"], bins=24, color="#4C72B0", edgecolor="white", alpha=0.9)
    axes[0].set_title("housing：自住房屋中位价分布")
    axes[0].set_xlabel("MEDV（千美元）"); axes[0].set_ylabel("频数")
    for sp, color in zip(["setosa", "versicolor", "virginica"], ["#55A868", "#DD8452", "#C44E52"]):
        sub = iris[iris["Species"] == sp]
        axes[1].scatter(sub["Petal_Length"], sub["Petal_Width"], s=22, alpha=0.8, label=sp, color=color)
    axes[1].set_title("iris：花瓣长度—宽度")
    axes[1].set_xlabel("Petal_Length（cm）"); axes[1].set_ylabel("Petal_Width（cm）")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig02_eda_distributions.png")


def fig_corr():
    housing, _ = load_clean("housing")
    iris, _ = load_clean("iris")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    sns.heatmap(housing.drop(columns=["MEDV"]).corr(), cmap="RdBu_r", center=0,
                ax=axes[0], cbar_kws={"label": "Pearson r"}, square=True,
                xticklabels=True, yticklabels=True,
                annot_kws={"fontsize": 7})
    axes[0].tick_params(axis="both", labelsize=7)
    axes[0].set_title("housing：13 个预测变量相关性")
    sns.heatmap(iris[IRIS_FEATURES].corr(), cmap="RdBu_r", center=0, annot=True,
                fmt=".2f", ax=axes[1], cbar_kws={"label": "Pearson r"}, square=True,
                annot_kws={"fontsize": 8})
    axes[1].tick_params(axis="both", labelsize=8)
    axes[1].set_title("iris：4 个形态特征相关性")
    fig.tight_layout()
    save(fig, "fig03_correlation.png")


def fig_pca():
    z = np.load(ART / "housing_seed42_scores.npz", allow_pickle=True)
    X, y, types = z["X_test"], z["y_test"], z["type_test"]
    pcs = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(pcs[y == 0, 0], pcs[y == 0, 1], s=22, c="#B0B0B0", alpha=0.65, label="正常")
    for t, color, marker in [("point", "#C44E52", "o"), ("multi", "#DD8452", "^"), ("extreme", "#8172B3", "s")]:
        m = (y == 1) & (types == t)
        ax.scatter(pcs[m, 0], pcs[m, 1], s=55, c=color, marker=marker, edgecolor="k",
                   linewidth=0.4, label={"point": "点异常", "multi": "协同异常", "extreme": "极端组合"}[t])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("housing 测试集 PCA：注入异常与正常样本")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig04_pca_anomalies.png")


def fig_quality():
    iris = read_raw("iris_with_error.csv")
    housing = read_raw("housing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    miss = iris[IRIS_FEATURES + ["Species"]].isna().sum()
    axes[0].bar(miss.index, miss.values, color="#4C72B0")
    axes[0].set_title("iris_with_error：各字段缺失格数")
    axes[0].set_ylabel("缺失格数"); axes[0].tick_params(axis="x", rotation=20)
    labels = ["iris 缺失格", "iris 重复行", "housing 缺失格", "housing 重复行"]
    vals = [int(iris.isna().sum().sum()), int(iris.duplicated().sum()),
            int(housing.isna().sum().sum()), int(housing.duplicated().sum())]
    axes[1].bar(labels, vals, color=["#DD8452", "#C44E52", "#55A868", "#8172B3"])
    axes[1].set_title("数据质量问题计数"); axes[1].set_ylabel("数量")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "fig05_quality.png")


def _load_scores(dataset="housing", seed=42):
    z = np.load(ART / f"{dataset}_seed{seed}_scores.npz", allow_pickle=True)
    scores = {m: z[f"score_{m.replace(' ', '_').replace('-', '_')}"] for m in METHODS}
    return z, scores


def fig_score_dist():
    z, scores = _load_scores()
    y = z["y_test"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (m, s), color in zip(axes.ravel(), scores.items(), COLORS):
        bins = np.linspace(np.min(s), np.max(s), 24)
        ax.hist(s[y == 0], bins=bins, alpha=0.65, label="正常", color="#9E9E9E", density=True)
        ax.hist(s[y == 1], bins=bins, alpha=0.75, label="异常", color=color, density=True)
        ax.set_title(m); ax.set_xlabel("异常得分"); ax.set_ylabel("密度")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("housing：正常与异常样本的异常得分分布（seed=42）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig06_score_distributions.png")


def fig_pr_roc():
    z, scores = _load_scores()
    y = z["y_test"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for (m, s), color in zip(scores.items(), COLORS):
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(r, p, label=f"{m} (AP={ap:.3f})", color=color, lw=1.8)
    ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
    ax.set_title("housing：Precision-Recall 曲线（seed=42）")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); save(fig, "fig07_pr_curves.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for (m, s), color in zip(scores.items(), COLORS):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{m} (AUC={auc(fpr, tpr):.3f})", color=color, lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("假阳性率 FPR"); ax.set_ylabel("真阳性率 TPR")
    ax.set_title("housing：ROC 曲线（seed=42）")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); save(fig, "fig08_roc_curves.png")


def fig_recall_fpr():
    m = json.load(open(OUT_DIR / "metrics.json", encoding="utf-8"))
    agg = m["housing"]["aggregate"]
    methods = [k for k in METHODS if k in agg]
    recall = [agg[k]["recall"]["mean"] for k in methods]
    recall_s = [agg[k]["recall"]["std"] for k in methods]
    fpr = [agg[k]["fpr"]["mean"] for k in methods]
    fpr_s = [agg[k]["fpr"]["std"] for k in methods]
    x = np.arange(len(methods)); w = 0.36
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(x - w / 2, recall, w, yerr=recall_s, capsize=3, color="#4C72B0", label="Recall")
    ax1.axhline(0.85, color="#C44E52", ls="--", lw=1.2, label="Recall 目标 0.85")
    ax1.set_ylabel("召回率 Recall"); ax1.set_ylim(0, 1.15)
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, fpr, w, yerr=fpr_s, capsize=3, color="#DD8452", label="FPR")
    ax2.axhline(0.10, color="#8172B3", ls=":", lw=1.3, label="FPR 目标 0.10")
    ax2.set_ylabel("假阳性率 FPR"); ax2.set_ylim(0, 0.18)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=15)
    ax1.set_title("housing：5 个随机种子的 Recall 与 FPR（均值±标准差）")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.tight_layout(); save(fig, "fig09_recall_fpr.png")


def fig_sensitivity():
    long = pd.read_csv(TABLE_DIR / "table_sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # LOF: k x quantile
    lof = long[long.method == "LOF"].pivot_table(index="param_value", columns="threshold_quantile", values="f1")
    lof.index = pd.to_numeric(lof.index, errors="coerce")
    lof = lof.sort_index()
    sns.heatmap(lof, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[0, 0])
    axes[0, 0].set_title("LOF：n_neighbors × 阈值分位（F1）")
    axes[0, 0].set_xlabel("阈值分位"); axes[0, 0].set_ylabel("n_neighbors")
    # IForest
    rf = long[long.method == "Isolation Forest"].pivot_table(index="param_value", columns="threshold_quantile", values="f1")
    sns.heatmap(rf, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("Isolation Forest：max_samples × 阈值分位（F1）")
    axes[0, 1].set_xlabel("阈值分位"); axes[0, 1].set_ylabel("max_samples")
    # OCSVM: parse nu/gamma
    oc = long[long.method == "One-Class SVM"].copy()
    oc[["nu", "gamma"]] = oc["param_value"].str.extract(r"nu=([^,]+),gamma=(.+)")
    oc_p = oc.pivot_table(index="nu", columns="gamma", values="f1")
    sns.heatmap(oc_p, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[1, 0])
    axes[1, 0].set_title("One-Class SVM：nu × gamma（阈值分位=0.95，F1）")
    axes[1, 0].set_xlabel("gamma"); axes[1, 0].set_ylabel("nu")
    # Mahalanobis
    mb = long[long.method == "Mahalanobis"].pivot_table(index="param_value", columns="threshold_quantile", values="f1")
    sns.heatmap(mb, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[1, 1])
    axes[1, 1].set_title("Mahalanobis：support_fraction × 阈值分位（F1）")
    axes[1, 1].set_xlabel("阈值分位"); axes[1, 1].set_ylabel("support_fraction")
    fig.tight_layout(); save(fig, "fig10_sensitivity.png")


def fig_contrib():
    df = pd.read_csv(TABLE_DIR / "table_feature_contrib.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, method in zip(axes, ["Mahalanobis", "Isolation Forest"]):
        sub = df[df.method == method].sort_values("mean_share", ascending=True).tail(8)
        ax.barh(sub["feature"], sub["mean_share"] * 100, color="#4C72B0" if method == "Mahalanobis" else "#DD8452")
        ax.set_title(f"{method}：特征贡献占比")
        ax.set_xlabel("平均贡献占比（%）")
    fig.tight_layout(); save(fig, "fig11_feature_contrib.png")


def fig_iris():
    m = json.load(open(OUT_DIR / "metrics.json", encoding="utf-8"))
    agg = m["iris"]["aggregate"]
    methods = [k for k in METHODS if k in agg]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title, target in zip(
        axes,
        ["recall", "fpr", "pr_auc"],
        ["iris：Recall（10 个种子）", "iris：FPR（10 个种子）", "iris：PR-AUC（10 个种子）"],
        [0.85, 0.10, None],
    ):
        vals = [agg[k][key]["mean"] for k in methods]
        errs = [agg[k][key]["std"] for k in methods]
        ax.bar(np.arange(len(methods)), vals, yerr=errs, capsize=3,
               color={"recall": "#4C72B0", "fpr": "#DD8452", "pr_auc": "#55A868"}[key])
        if target is not None:
            ax.axhline(target, color="#C44E52", ls="--", lw=1.1)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylim(0, 0.30 if key == "fpr" else 1.12)
        ax.set_title(title)
    fig.tight_layout(); save(fig, "fig12_iris.png")


def fig_runtime():
    m = json.load(open(OUT_DIR / "metrics.json", encoding="utf-8"))
    agg = m["housing"]["aggregate"]
    methods = [k for k in METHODS if k in agg]
    vals = [agg[k]["fit_seconds"]["mean"] for k in methods]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(methods, vals, color="#8172B3")
    ax.set_xscale("log")
    ax.set_xlabel("训练耗时（秒，对数坐标）")
    ax.set_title("housing：单次训练耗时对比")
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v:.3f}s", va="center", fontsize=8)
    fig.tight_layout(); save(fig, "fig13_runtime.png")


def fig_per_type():
    df = pd.read_csv(TABLE_DIR / "table_per_type_housing.csv")
    types = ["点异常", "协同异常", "极端组合"]
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, t in enumerate(types):
        ax.bar(x + (i - 1) * w, pd.to_numeric(df[t], errors="coerce"), w, label=t)
    ax.set_xticks(x); ax.set_xticklabels(df["模型"], rotation=15)
    ax.set_ylabel("召回率"); ax.set_ylim(0, 1.05)
    ax.set_title("housing：分异常类型的召回率（seed 聚合均值）")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); save(fig, "fig14_per_type_recall.png")


def main():
    fig_route(); fig_eda(); fig_corr(); fig_pca(); fig_quality()
    fig_score_dist(); fig_pr_roc(); fig_recall_fpr(); fig_sensitivity()
    fig_contrib(); fig_iris(); fig_runtime(); fig_per_type()
    print("all figures done")


if __name__ == "__main__":
    main()
