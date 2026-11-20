# T13 多元异常检测算法对比 —— 实训成果包

华北理工大学理学院《大数据技术与工程》研究生实训（2026—2027 秋季学期）
组长：程龙飞（20262193024，单人组）

## 1. 项目简介

本项目完成 T13 选题的完整链路：数据审计 → 清洗与划分 → 受控异常注入 → 特征工程 → 五类检测器 → 阈值校准 → 测试评估 → 参数敏感性与调参对照 → 轴对齐消融 → 特征贡献与误差剖析 → Word 报告。

- 主数据：`data/data/housing.csv`（506×14），按 T13 要求注入带真值的点异常、多特征协同异常与极端组合异常。
- 小样本对照：`data/data/iris_with_error.csv`（152×5，含 13 个缺失格与 5 条重复行），清洗后 145 行。
- 检测器：Z-Score（稳健中位数/MAD）、马氏距离（MCD / Ledoit–Wolf）、LOF、Isolation Forest、One-Class SVM，另加按 val PR-AUC 选取前三模型的 z-score 排名融合。
- 评价：PR-AUC 为主，Recall、FPR、Precision、F1、ROC-AUC、训练耗时与分类型召回为辅。
- 协议：fit / calib / val / test 四划分；超参数在训练集内部用 3 折交叉验证选取，阈值只在 calib 正常样本与独立种子的 val 合成异常上确定（calib FPR≤8% 安全裕度内最大化 val 召回），测试集只做最终评估；housing 与 iris 各 10 个随机种子。
- 附加实验：①默认参数 / 交叉验证最优 / 交叉验证最差 三档对照；②Isolation Forest 的 PCA 白化轴对齐消融；③两种载体（Isolation Forest、马氏距离）的特征工程增益对照。

## 2. 目录结构

```
T13_程龙飞_实训成果/
├─ data/data/                 # 原始数据（来自课程 data.zip）
├─ src/
│  ├─ config.py               # 路径、种子、字段与特征标签
│  ├─ data_loader.py          # 读取、审计、清洗
│  ├─ features.py             # 派生特征、筛选、缩放
│  ├─ inject.py               # 三类异常注入与注入清单
│  ├─ detectors.py            # 五类检测器、交叉验证调参与阈值协议
│  ├─ evaluation.py           # 指标与聚合
│  ├─ experiment.py           # 主实验、特征增益、敏感性、调参对照
│  ├─ explain.py              # 特征贡献与错分案例
│  ├─ make_tables.py          # 报告附表与实验记录
│  ├─ plots.py                # 15 幅 300 dpi 图
│  ├─ report_builder.py       # 按模板生成 Word
│  ├─ make_git_stats.py       # 生成报告 9.3 节的提交统计
│  ├─ export_pdf.ps1          # 用 Word 导出 report/ 下的 PDF
│  └─ run_all.py              # 一键复现
├─ tests/                     # pytest 单元测试（16 项）
├─ outputs/
│  ├─ tables/                 # 结果 CSV
│  ├─ figures/                # 报告图
│  ├─ logs/                   # 运行日志、环境信息
│  ├─ artifacts/              # 注入清单、得分数组
│  ├─ metrics.json            # 全部种子明细
│  └─ metrics_summary.json    # 聚合均值/标准差
├─ report/13_程龙飞_实训报告.docx
├─ requirements.txt
└─ README.md
```

## 3. 复现

```powershell
cd "T13_程龙飞_实训成果"
python -m pip install -r requirements.txt
python -m pytest -q
python src/run_all.py
```

`run_all.py` 会依次执行数据审计、主实验、附表、特征贡献、绘图与报告生成。由于主实验含 3 折交叉验证调参与三组附加实验，完整运行约 12–15 分钟（CPU：AMD Ryzen 9 8945HX，16 核 32 线程）。

报告由 `python-docx` 生成；`report/` 目录同时提供 Word 与由 Word 导出的 PDF（33 页）。

> 报告使用 `python-docx` 生成，PDF 由 Word 导出，两者内容一致。
>
> 排版约定：正文宋体小四、首行缩进 2 字符、固定行距 20 磅；全部数据表为三线表；正文中的文献引用为 Word 交叉引用域（REF），可随参考文献编号自动更新（选中全文按 F9 刷新）。

## 4. 主要结果（housing，10 个随机种子均值）

| 模型 | PR-AUC | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| Z-Score（稳健中位数/MAD） | 0.3492 | 0.3000 | 0.0664 | 0.3040 |
| **Mahalanobis** | **0.8866** | **0.9667** | **0.0511** | **0.7991** |
| LOF | 0.7488 | 0.8067 | 0.0686 | 0.6716 |
| Isolation Forest | 0.4434 | 0.4133 | 0.0737 | 0.3801 |
| One-Class SVM | 0.7301 | 0.8000 | 0.0613 | 0.6723 |
| Rank-Ensemble（val PR-AUC 前三） | 0.8588 | 0.9333 | 0.0569 | 0.7635 |

- 达标线（Recall≥0.85 且 FPR≤10%）：马氏距离在均值意义上满足；融合模型召回更高（0.9333），但精确率降至 0.6504。
- 分类型召回：马氏距离对点/协同/极端组合分别为 0.966/1.000/0.830；融合为 0.949/0.988/0.802。
- 轴对齐消融（表 6-5）：同一批特征经 PCA 白化后，Isolation Forest 的 PR-AUC 由 0.4434 升到 0.7866、F1 由 0.3801 升到 0.7109，证实其失利源于切分方向与相关偏移方向错配。
- 调参必要性（默认参数 → 交叉验证最优，测试 F1）：One-Class SVM 0.4118 → 0.6723；马氏距离 0.7915 → 0.7991（默认已接近最优）；LOF 0.5652 → 0.6716；Isolation Forest 0.3909 → 0.3801（负收益）。参数取到网格最差端时，马氏距离 F1 仅 0.4350。
- 参数敏感性：LOF 的 F1 极差 0.566（最敏感），马氏距离 0.455，One-Class SVM 0.344，Isolation Forest 0.210，Z-Score 0.127。
- 特征贡献（马氏距离）：NOX 18.0%、TAX 13.2%、RM 12.6%、DIS 8.5%、INDUS 7.0%。
- 特征工程（seed=42，表 5-2）：派生特征对马氏距离基本中性（PR-AUC +0.0055，Recall 0.9333→1.0000），对 Isolation Forest 为负（PR-AUC 0.6239→0.5907）。
- 小样本 iris：LOF/OCSVM 平均召回 0.9250/1.0000，但 FPR 分别 12.50%/13.75%，方差很大；说明小样本不能用单次划分下结论。
- 稳健 Z-Score 的失效机制（附表 8）：CRIM 的稳健尺度仅 0.1123、测试集最大 |z| 达 235.5，得分几乎由 1–2 个重尾字段主导。

## 5. 数据与合规

- `housing.csv` 源自 Boston Housing 公开教学数据；`iris_with_error.csv` 源自经典 Iris 数据。两者不含个人身份信息，仅用于课程实训。
- 原始数据未被手工修改；所有清洗、注入、划分、缩放均由代码完成。
- 注入异常带完整清单（行号、类型、原值、注入值、种子），可逐条追溯。

## 6. 说明

- 报告中的数值全部来自 `outputs/metrics.json` 与 `outputs/tables/*.csv`，由 `report_builder.py` 回填，未手工填写。
- 单次运行的具体数值会随 scikit-learn 版本与随机种子略有差异；本成果包锁定 `requirements.txt` 中的版本。
- 详细实验记录见 `outputs/logs/run_main.log`、`outputs/logs/pytest.log` 与报告附表 6。
- `outputs/tables/table_if_pca*.csv` 为 PCA 白化消融的原始记录与汇总，`table_zscore_profile.csv` 为稳健 Z-Score 的逐字段尺度诊断（对应附表 8）。
