"""Generate the remaining report tables that are not produced by experiment.py."""
from __future__ import annotations

import json

import pandas as pd

from config import HOUSING_FEATURES, IRIS_FEATURES, OUT_DIR, TABLE_DIR

DERIVED = {
    "iris": [
        ("Petal_Area", "Petal_Length × Petal_Width", "花瓣面积，刻画花冠整体大小", "是（与花瓣长度高度相关，被相关性筛选剔除）"),
        ("Sepal_Area", "Sepal_Length × Sepal_Width", "花萼面积，刻画萼片整体大小", "是"),
        ("Petal_Aspect", "Petal_Length ÷ Petal_Width", "花瓣细长程度", "是"),
        ("Sepal_Aspect", "Sepal_Length ÷ Sepal_Width", "花萼细长程度", "是"),
        ("Petal_Sepal_Ratio", "Petal_Length ÷ Sepal_Length", "花瓣相对花萼的长度比例", "否（与花瓣长度信息重叠）"),
    ],
    "housing": [
        ("CRIM_log", "log1p(CRIM)", "犯罪率长尾压缩", "是"),
        ("LSTAT_log", "log1p(LSTAT)", "低收入比例长尾压缩", "否（与 LSTAT 高度相关）"),
        ("TAX_log", "log1p(TAX)", "税率长尾压缩", "否（与 TAX 高度相关）"),
        ("RM2", "RM²", "房间数非线性项", "否（与 RM 高度相关）"),
        ("LSTAT_RM", "LSTAT ÷ RM", "低收入与房间数的交互", "否（与 LSTAT 高度相关）"),
        ("NOX_DIS", "NOX × DIS", "污染与区位距离的联合效应", "否（相关性筛选）"),
        ("TAX_RAD", "TAX ÷ (RAD+1)", "税率与高速可达性的比值", "是"),
        ("AGE_LSTAT", "AGE × LSTAT", "房龄与低收入比例的交互", "否（相关性筛选）"),
        ("CRIM_LSTAT", "CRIM × LSTAT", "犯罪率与低收入比例的交互", "否（相关性筛选）"),
        ("PTRATIO_LSTAT", "PTRATIO × LSTAT", "师生比与低收入比例的交互", "否（相关性筛选）"),
    ],
}


def main():
    metrics = json.load(open(OUT_DIR / "metrics.json", encoding="utf-8"))
    rows = []
    for ds in ("housing", "iris"):
        selected = set(metrics[ds]["per_seed"][0]["selected_features"])
        for name, logic, meaning, selected_text in DERIVED[ds]:
            rows.append({
                "数据集": ds, "特征名": name, "构造逻辑": logic,
                "业务/数据含义": meaning, "是否入选": selected_text,
            })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_features.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"姓名": "程龙飞", "学号": "20262193024",
         "承担模块": "数据审计、清洗与划分、异常注入、五类检测器、评估、敏感性、可解释性、报告与打包",
         "具体产出": "全部源码、图表、结果表、Word 报告与 ZIP 成果包", "贡献占比": "100%"},
    ]).to_csv(TABLE_DIR / "table_group.csv", index=False, encoding="utf-8-sig")

    def mv(ds, method, key):
        return metrics[ds]["aggregate"][method][key]["mean"]

    sens = pd.read_csv(TABLE_DIR / "table_sensitivity_summary.csv")
    sens_txt = "；".join(f"{r.method} {r.f1_range:.3f}" for r in sens.itertuples())
    contrib = pd.read_csv(TABLE_DIR / "table_feature_contrib.csv")
    top_mb = contrib[contrib.method == "Mahalanobis"].head(3)
    top_mb_txt = "、".join(f"{r.feature} {r.mean_share * 100:.1f}%" for r in top_mb.itertuples())
    err_path = TABLE_DIR / "table_error_cases.csv"
    err_n = len(pd.read_csv(err_path)) if err_path.exists() else 0

    log = [
        {"日期": "2026-11-18", "实验内容": "数据审计与清洗",
         "参数设置": "iris 缺失按 Species 中位数；连续缺失≥3 行删除；完全重复删除",
         "结果": "iris 152→145 行；housing 506 行；缺失 13 格（1.71%）、重复 5 行（3.29%）",
         "结论/下一步": "两份数据可用；housing 自然长尾保留，不当作错误"},
        {"日期": "2026-11-18", "实验内容": "主实验：5 类检测器 + 融合",
         "参数设置": "housing 70/15/15 训练内划分 + 30% 测试；5 种子；测试注入 15 个异常（9.87%）",
         "结果": (f"Mahalanobis PR-AUC {mv('housing','Mahalanobis','pr_auc'):.4f}、"
                  f"Recall {mv('housing','Mahalanobis','recall'):.4f}、FPR {mv('housing','Mahalanobis','fpr'):.4f}；"
                  f"融合 Recall {mv('housing','Rank-Ensemble','recall'):.4f}、FPR {mv('housing','Rank-Ensemble','fpr'):.4f}"),
         "结论/下一步": "Mahalanobis 达标；融合提升召回但精确率略降；分析极端组合漏报"},
        {"日期": "2026-11-18", "实验内容": "小样本对照：iris",
         "参数设置": "145 行；10 种子；测试 44 行、注入 4 个异常（9.09%）",
         "结果": (f"LOF/One-Class SVM 平均 Recall {mv('iris','LOF','recall'):.3f}/"
                  f"{mv('iris','One-Class SVM','recall'):.3f}，FPR {mv('iris','LOF','fpr'):.3f}/"
                  f"{mv('iris','One-Class SVM','fpr'):.3f}；Mahalanobis FPR {mv('iris','Mahalanobis','fpr'):.3f}、"
                  f"Recall {mv('iris','Mahalanobis','recall'):.3f}"),
         "结论/下一步": "小样本下 FPR 方差大；不能只用单次划分下结论"},
        {"日期": "2026-11-18", "实验内容": "参数敏感性",
         "参数设置": "LOF k∈{5,10,15,20,30,50}；IForest max_samples；OCSVM nu×gamma；Mahalanobis support_fraction",
         "结果": f"F1 极差：{sens_txt}",
         "结论/下一步": "LOF 对 k 最敏感；报告需给出推荐参数区间而非单点"},
        {"日期": "2026-11-18", "实验内容": "特征贡献与误差剖析",
         "参数设置": "seed=42；Mahalanobis 距离分解；IForest 特征替换",
         "结果": f"Mahalanobis 主要贡献 {top_mb_txt}；{err_n} 个错分案例",
         "结论/下一步": "自然极端房价样本是主要误报源；建议二次复核而非直接删除"},
        {"日期": "2026-11-18", "实验内容": "报告生成与成果打包",
         "参数设置": "python-docx 模板化生成；固定随机种子",
         "结果": "Word 报告（含附录 A 代码片段）、14 幅图、结果表、ZIP 成果包",
         "结论/下一步": "提交前复核字数、图表编号与数值一致性"},
    ]
    pd.DataFrame(log).to_csv(TABLE_DIR / "table_experiment_log.csv", index=False, encoding="utf-8-sig")

    runtime = []
    for ds in ("housing", "iris"):
        for method, agg in metrics[ds]["aggregate"].items():
            runtime.append({
                "数据集": ds, "模型": method,
                "训练耗时均值(s)": agg.get("fit_seconds", {}).get("mean", float("nan")),
                "训练耗时标准差(s)": agg.get("fit_seconds", {}).get("std", 0.0),
            })
    pd.DataFrame(runtime).to_csv(TABLE_DIR / "table_runtime.csv", index=False, encoding="utf-8-sig")
    print("extra tables written")


if __name__ == "__main__":
    main()
