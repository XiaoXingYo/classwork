# T13 多元异常检测算法对比 —— 实训成果包

华北理工大学理学院《大数据技术与工程》研究生实训（2026—2027 秋季学期）
组长：程龙飞（20262193024，单人组）

## 1. 项目简介

本项目完成 T13 选题的完整链路：数据审计 → 清洗与划分 → 受控异常注入 → 特征工程 → 五类检测器 → 阈值校准 → 测试评估 → 参数敏感性 → 特征贡献与误差剖析 → Word 报告。

- 主数据：`data/data/housing.csv`（506×14），按 T13 要求注入带真值的点异常、多特征协同异常与极端组合异常。
- 小样本对照：`data/data/iris_with_error.csv`（152×5，含 13 个缺失格与 5 条重复行），清洗后 145 行。
- 检测器：Z-Score、马氏距离（MCD）、LOF、Isolation Forest、One-Class SVM，另加五模型 z-score 排名融合。
- 评价：PR-AUC 为主，Recall、FPR、Precision、F1、ROC-AUC、训练耗时与分类型召回为辅。
- 防泄漏：fit / calib / val / test 四划分；阈值与超参数只在 calib 正常样本与独立种子的 val 合成异常上选择，测试集只做最终评估。

## 2. 目录结构

```
13_程龙飞_实训成果/
├─ data/data/                 # 原始数据（来自课程 data.zip）
├─ src/
│  ├─ config.py               # 路径、种子、字段与特征标签
│  ├─ data_loader.py          # 读取、审计、清洗
│  ├─ features.py             # 派生特征、筛选、缩放
│  ├─ inject.py               # 三类异常注入与注入清单
│  ├─ detectors.py            # 五类检测器与统一调参/阈值协议
│  ├─ evaluation.py           # 指标与聚合
│  ├─ experiment.py           # 主实验、特征增益、敏感性
│  ├─ explain.py              # 特征贡献与错分案例
│  ├─ make_tables.py          # 报告附表与实验记录
│  ├─ plots.py                # 14 幅 300 dpi 图
│  ├─ report_builder.py       # 按模板生成 Word
│  ├─ make_git_stats.py       # 生成报告 9.3 节的提交统计
│  └─ run_all.py              # 一键复现
├─ tests/                     # pytest 单元测试（10 项）
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
cd "D:\life\研\项目\课程作业\13_程龙飞_实训成果"
python -m pip install -r requirements.txt
python -m pytest -q
python src/run_all.py
```

`run_all.py` 会依次执行数据审计、主实验、附表、特征贡献、绘图与报告生成。完整运行约数分钟（CPU：AMD Ryzen 9 8945HX，16 核 32 线程）。

> 报告使用 `python-docx` 生成；如需 PDF，可在 Word 中“另存为 PDF”。按本次交付约定，成果包不附 PDF。

## 4. 主要结果（housing，5 个随机种子均值）

| 模型 | PR-AUC | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| Z-Score | 0.5851 | 0.6667 | 0.0905 | 0.5323 |
| **Mahalanobis** | **0.9026** | **0.8933** | **0.0204** | **0.8563** |
| LOF | 0.7440 | 0.6800 | 0.0307 | 0.6861 |
| Isolation Forest | 0.5089 | 0.5067 | 0.0759 | 0.4538 |
| One-Class SVM | 0.7752 | 0.8533 | 0.0526 | 0.7451 |
| Rank-Ensemble | 0.8590 | 0.9200 | 0.0394 | 0.8048 |

- 达标线（Recall≥0.85 且 FPR≤10%）：马氏距离在均值意义上满足；融合模型召回更高（0.9200），但精确率降至 0.7237。
- 分类型召回：马氏距离对点/协同/极端组合分别为 0.88/1.00/0.63；融合为 0.92/0.95/0.92。
- 参数敏感性：LOF 的 F1 极差 0.566（最敏感），马氏距离 0.455，Isolation Forest 0.210，One-Class SVM 0.209，Z-Score 0.100。
- 特征贡献（马氏距离）：NOX 21.3%、TAX 13.7%、RM 12.8%、DIS 9.0%、CRIM 7.5%。
- 小样本 iris：LOF/OCSVM 平均召回 0.95，但 FPR 分别 12.50%/9.50%，方差很大；说明小样本不能用单次划分下结论。

## 5. 数据与合规

- `housing.csv` 源自 Boston Housing 公开教学数据；`iris_with_error.csv` 源自经典 Iris 数据。两者不含个人身份信息，仅用于课程实训。
- 原始数据未被手工修改；所有清洗、注入、划分、缩放均由代码完成。
- 注入异常带完整清单（行号、类型、原值、注入值、种子），可逐条追溯。

## 6. 说明

- 报告中的数值全部来自 `outputs/metrics.json` 与 `outputs/tables/*.csv`，由 `report_builder.py` 回填，未手工填写。
- 单次运行的具体数值会随 scikit-learn 版本与随机种子略有差异；本成果包锁定 `requirements.txt` 中的版本。
- 详细实验记录见 `outputs/logs/run_main.log`、`outputs/logs/pytest.log` 与报告附表 6。
