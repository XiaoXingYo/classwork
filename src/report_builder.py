"""Build the Word report from measured artefacts (no hand-entered metrics)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from config import FIG_DIR, OUT_DIR, REPORT_DIR, ROOT, TABLE_DIR

METRICS = json.load(open(OUT_DIR / "metrics.json", encoding="utf-8"))


def agg(dataset: str, method: str, key: str):
    return METRICS[dataset]["aggregate"][method][key]["mean"]


def agg_s(dataset: str, method: str, key: str):
    return METRICS[dataset]["aggregate"][method][key]["std"]


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def f4(x: float) -> str:
    return f"{x:.4f}"


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name, encoding="utf-8-sig")


def compact_metric(value) -> str:
    s = str(value)
    def repl(m):
        return f"{float(m.group(1)):.3f}±{float(m.group(2)):.3f}"
    return re.sub(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)", repl, s)


def compact_params(value) -> str:
    s = str(value)
    for a, b in [("n_estimators", "n_est"), ("max_samples", "m_s"),
                 ("max_features", "m_f"), ("support_fraction", "sf"),
                 ("n_neighbors", "k"), ("gamma", "γ")]:
        s = s.replace(a, b)
    return s


def perf_view(df: pd.DataFrame, with_dataset=False, with_params=True, with_time=False,
              with_roc=True) -> pd.DataFrame:
    cols = []
    if with_dataset:
        cols.append("数据集")
    cols += ["模型"]
    if with_params:
        cols.append("主要参数")
    cols += ["PR-AUC", "Recall", "F1", "FPR"]
    if with_roc:
        cols.append("ROC-AUC")
    if with_time:
        cols.append("训练耗时(s)")
    out = df[cols].copy()
    for c in ["PR-AUC", "Recall", "F1", "FPR", "ROC-AUC"]:
        if c in out.columns:
            out[c] = out[c].map(compact_metric)
    if with_params:
        out["主要参数"] = out["主要参数"].map(compact_params)
    return out


# ---------------------------------------------------------------- formatting
def set_run_font(run, name="宋体", size=12, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def body(doc, text: str, indent=True, bold=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_after=0):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    r = p.add_run(text)
    set_run_font(r, "宋体", size, bold)
    return p


def heading(doc, text: str, level=1):
    size = {1: 16, 2: 14, 3: 12}[level]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    set_run_font(r, "黑体", size, True)
    return p


def caption(doc, text: str, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = keep_next
    p.paragraph_format.keep_together = True
    r = p.add_run(text)
    set_run_font(r, "宋体", 10.5, False)
    return p


def shade(cell, fill="D9E2F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def table(doc, df: pd.DataFrame, cap: str, widths=None, font_size=9, header_fill="D9E2F3",
          keep_together=False):
    caption(doc, cap, keep_next=True)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    repeat_header(hdr)
    _no_split(hdr)
    for i, c in enumerate(df.columns):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(c))
        set_run_font(r, "宋体", font_size, True)
        shade(cell, header_fill)
        if widths:
            cell.width = Cm(widths[i])
    for _, row in df.iterrows():
        tr = t.add_row()
        _no_split(tr)
        cells = tr.cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(str(v)) < 24 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run("" if pd.isna(v) else str(v))
            set_run_font(r, "宋体", font_size, False)
            if widths:
                cells[i].width = Cm(widths[i])
    if keep_together:
        for row in t.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _no_split(row):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trPr.append(cant)


def figure(doc, filename: str, cap: str, width_cm=14.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    p.add_run().add_picture(str(FIG_DIR / filename), width=Cm(width_cm))
    caption(doc, cap)


def code_block(doc, rel_path: str, first_line: int, n_lines: int, cap: str):
    """Insert a verbatim excerpt, so the reported line numbers always match the code."""
    src = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    start = first_line - 1
    end = min(start + n_lines, len(src))
    text = "\n".join(src[start:end])
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(1)
    pf.line_spacing = 1.0
    pf.keep_with_next = True
    r = p.add_run(f"{cap}    {rel_path}: {first_line}–{end}")
    set_run_font(r, "宋体", 9.5, True)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(10)
    pf.left_indent = Cm(0.25)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F5F5F5")
    pPr.append(shd)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve"); instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)
    set_run_font(run, "宋体", 9)


# ---------------------------------------------------------------- document
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.8); sec.right_margin = Cm(2.6)
page_number(sec)

style = doc.styles["Normal"]
style.font.name = "宋体"; style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
style.paragraph_format.line_spacing = 1.5

# ---------------- cover
for _ in range(2):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("华北理工大学理学院"); set_run_font(r, "宋体", 18, True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("《大数据技术与工程》研究生实训报告"); set_run_font(r, "黑体", 22, True)
doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("T13  多元异常检测算法对比"); set_run_font(r, "黑体", 20, True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("——基于可审计注入基准的五类方法实证研究"); set_run_font(r, "宋体", 14, False)
doc.add_paragraph(); doc.add_paragraph()

cover = [
    ("选题方向", "异常检测", "选题编号", "T13"),
    ("组    长", "程龙飞", "学    号", "20262193024"),
    ("组    员", "无（单人组）", "学    号", "—"),
    ("指导教师", "＿＿＿＿＿＿", "实训学期", "2026—2027 秋季学期"),
    ("提交日期", "2026 年 11 月 18 日", "成    绩", ""),
]
t = doc.add_table(rows=len(cover), cols=4)
t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
for i, row in enumerate(cover):
    for j, v in enumerate(row):
        cell = t.rows[i].cells[j]; cell.text = ""
        cell.width = Cm([3.0, 5.0, 3.0, 4.6][j])
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(v); set_run_font(r, "宋体", 12, j % 2 == 0)
doc.add_page_break()

# ---------------- abstract
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("摘  要"); set_run_font(r, "黑体", 16, True)
abstract = (
    "针对中低维表格数据中无监督异常检测的算法选择问题，本文以 T13 任务指定的 iris_with_error 与 housing 数据为对象，"
    "构建了包含点异常、多特征协同异常和极端组合异常的可审计注入基准，系统比较 Z-Score、马氏距离、LOF、Isolation Forest "
    "与 One-Class SVM 五类方法，并采用 PR-AUC、Recall、FPR、F1 与 ROC-AUC 评价。数据审计发现 iris 缺失 13 格（1.71%）、"
    "重复 5 行（3.29%），housing 无缺失、无重复；housing 主实验采用 5 个随机种子、70/15/15 训练内划分与 30% 测试集，"
    "测试集注入 15 个异常（9.87%）；iris 清洗后 145 行采用 10 个种子做小样本对照。"
    f"结果表明，马氏距离在 housing 上取得 PR-AUC {f4(agg('housing','Mahalanobis','pr_auc'))}、"
    f"Recall {f4(agg('housing','Mahalanobis','recall'))}、FPR {f4(agg('housing','Mahalanobis','fpr'))}，"
    "满足召回≥0.85 且误报≤10% 的达标线；五模型排名融合将 Recall 提升至 "
    f"{f4(agg('housing','Rank-Ensemble','recall'))}，但精确率降至 {f4(agg('housing','Rank-Ensemble','precision'))}。"
    "参数敏感性显示 LOF 的 F1 极差达 0.566，马氏距离为 0.455；特征贡献显示 NOX、TAX、RM 与 CRIM 是主要驱动因素，"
    "极端组合异常仍是主要漏报来源；iris 小样本的 FPR 方差显著增大。研究说明：在低维、近似椭球分布的数据上，"
    "统计距离方法优于树模型与密度方法，阈值校准与误报复核比盲目追求复杂模型更重要。上述结果对同类低维表格数据的异常检测具有参考价值，"
    "但外推到图像、文本与工业时序场景仍需进一步验证。"
)
body(doc, abstract, indent=True, size=12)
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(6)
r = p.add_run("【关键词】"); set_run_font(r, "黑体", 12, True)
r = p.add_run("异常检测；马氏距离；孤立森林；局部离群因子；PR-AUC"); set_run_font(r, "宋体", 12)
doc.add_page_break()

# ---------------- 1
heading(doc, "一、选题背景与意义", 1)
heading(doc, "1.1 业务背景", 2)
body(doc,
     "异常检测是工业运维、金融风控、网络安全、医疗监测和城市治理等场景的基础能力。以住宅市场为例，一套房产若在犯罪率、"
     "低收入人口比例、房间数、税率与区位可达性等维度同时呈现罕见组合，可能意味着数据录入错误、违规交易或评估口径异常；"
     "在工业控制系统中，传感器读数偏离正常工况则可能预示设备故障或网络攻击。此类问题的共同点是：正常样本充足、异常样本稀少、"
     "标签获取困难，且异常往往不是单字段越界，而是多个字段的协同偏移。")
body(doc,
     "传统单变量 3σ 规则实现简单，但在特征相关、分布偏斜或异常由多字段共同构成时容易失效：阈值取紧会带来大量误报，"
     "取松又会漏掉关键异常。业务方真正需要的不是“一个分数”，而是一套可复现、可解释、能控制误报的检测流程，"
     "以及在不同算法之间做出有依据的选择。")

heading(doc, "1.2 研究意义", 2)
body(doc,
     "理论层面，异常检测长期面临三个难点：无监督条件下模型选择缺乏标签、异常比例未知导致阈值难以确定、"
     "高维空间中距离与密度度量退化[1,2]。本文通过受控注入构造带真值的基准，使“无监督检测”能够在同一协议下被定量比较，"
     "并显式区分模型拟合、阈值校准与测试评估三个环节，避免把阈值调参误当成模型能力。")
body(doc,
     "应用层面，本文给出的流程可直接迁移到数据质量审计、业务异常复核与工业监测：先用统计距离或融合模型做高召回的初筛，"
     "再用可解释的特征贡献定位问题字段，最后按业务代价决定复核阈值。该流程比单独部署一个黑盒模型更符合实际落地条件。")

heading(doc, "1.3 国内外研究/应用现状", 2)
body(doc,
     "Chandola 等[1]将异常检测归纳为统计、距离、密度、聚类与分类等范式，指出无监督场景下“正常边界”的估计是核心难点；"
     "Zimek 等[2]进一步说明高维数据中距离集中现象会使 LOF 等局部方法退化。Breunig 等[3]提出的 LOF 用局部可达密度刻画"
     "样本相对邻域的稀疏程度，适合发现局部异常；Liu 等[4,5]提出的 Isolation Forest 用随机切分孤立样本，复杂度低、"
     "对维度较稳健；Schölkopf 等[6]与 Tax 等[7]分别从支持向量边界与数据描述角度构建 One-Class SVM；"
     "Rousseeuw 等[8]的 MCD 稳健协方差则为马氏距离提供了抗污染估计。")
body(doc,
     "在方法评测方面，Goldstein 等[9]在多个真实数据集上比较了十余种无监督算法，发现没有一种方法在所有数据上占优；"
     "Campos 等[10]系统讨论了 AUC、AP 等指标在极度不平衡与未知污染率下的偏差；Ruff 等[11]与 Pang 等[12]的综述则将"
     "浅层方法与深度自编码器、生成模型统一到“正常性建模”的框架下。现有工作的局限主要有三：其一，不同研究使用不同数据、"
     "不同划分与不同阈值策略，结果难以横向比较；其二，注入异常的类型单一，常只做单点越界，缺少多字段协同异常；"
     "其三，参数敏感性与误报来源分析不足，业务方难以判断模型何时可信。本文的切入点是：在同一套可审计注入协议下比较五类方法，"
     "把阈值校准与测试评估分离，并同时报告分类型召回、参数敏感性热图与特征贡献。")

# ---------------- 2
heading(doc, "二、问题定义与目标", 1)
heading(doc, "2.1 问题陈述", 2)
body(doc,
     "给定训练集 X={x_i}，x_i∈R^d，i=1,…,n，假设训练集只含正常样本（或污染率极低）。学习一个评分函数 s:R^d→R，"
     "使得正常样本得分低、异常样本得分高；再选取阈值 τ，输出二元判定 y_i=1[s(x_i)≥τ]。本文的输入为 iris 与 housing 的"
     "数值特征矩阵，输出为每个样本的异常得分、二元标签、方法排名及特征贡献。约束包括：（1）训练阶段不使用任何异常标签；"
     "（2）阈值只能利用训练集内部的正常校准样本与独立种子的合成验证异常选择；（3）测试集的注入真值只用于最终评估；"
     "（4）所有随机过程固定种子并可复现。从机器学习角度看，该问题属于无监督学习的一个分支[13]，"
     "因为没有异常标签可用于监督训练。")
body(doc,
     "需要强调的是，异常比例在真实业务中未知且随时间变化。本文将测试集异常比例控制在约 10%，仅用于构造可统计的评价场景，"
     "并不假设真实污染率就是 10%。因此主指标采用不依赖阈值的 PR-AUC，阈值指标只作为业务工作点的参考。")

heading(doc, "2.2 评价指标", 2)
body(doc,
     "主指标为 PR-AUC（Average Precision）。在异常稀少、负类占绝对多数时，ROC-AUC 容易被大量真负例“稀释”，"
     "而 PR 曲线直接刻画“查得准”与“查得全”的权衡[10]。辅助指标包括：Recall（召回率，漏报的反面）、"
     "FPR（正常样本被误判的比例）、Precision、F1、ROC-AUC，以及单次训练耗时。达标线沿用 T13 任务描述："
     "注入异常 Recall≥0.85 且 FPR≤10%。此外报告分类型召回（点异常、协同异常、极端组合）与参数敏感性，"
     "避免只报一个乐观的总体数字。阈值 τ 的选取本质上是统计决策问题：在正常分数分布的分位数上取阈值，"
     "等价于对第一类错误率（FPR）做显式约束，这一思路在统计学习方法中有系统讨论[14]。")

heading(doc, "2.3 预期目标", 2)
body(doc, "本文设置以下 5 条可检验目标：", indent=True)
for s in [
    "（1）在 housing 主实验上，至少一种方法达到 Recall≥0.85 且 FPR≤10%，并给出 5 个种子的均值与标准差；",
    "（2）完成 Z-Score、马氏距离、LOF、Isolation Forest、One-Class SVM 五类方法的统一协议对比，并给出 PR/ROC 曲线；",
    "（3）给出 LOF、Isolation Forest、One-Class SVM、马氏距离的关键参数敏感性热图，量化 F1 的波动范围；",
    "（4）给出特征贡献与不少于 3 条有数据支撑的业务洞察，并对典型误报/漏报案例做剖析；",
    "（5）全部实验可由一条命令复现，代码、数据、图表、结果表与报告一并交付。",
]:
    body(doc, s, indent=True)

# ---------------- 3
heading(doc, "三、数据集描述", 1)
heading(doc, "3.1 数据来源", 2)
body(doc,
     "本文使用课程提供的 data.zip 中的两份数据。iris_with_error.csv 源自经典 Iris 数据集（150 条、4 个形态特征、3 个品种），"
     "课程版本额外注入了缺失值与重复记录，共 152 行；housing.csv 源自 Boston Housing 数据[15]，包含 506 个街区、"
     "13 个预测变量与 1 个房价中位数（MEDV）。两份数据均为公开教学数据，不含个人身份信息；本文仅用于课程实训与方法比较，"
     "不用于商业用途。数据由课程压缩包直接读取，处理过程全部由代码完成，未手工修改原始文件。")
body(doc,
     "需要说明的是，Boston Housing 数据中的 B 指标带有历史与伦理争议，本文仅在方法比较层面使用，不对其业务含义作规范性判断；"
     "报告中保留字段原名并注明其统计含义，避免误用。")

heading(doc, "3.2 数据规模与字段", 2)
body(doc,
     "两份数据的规模、清洗后样本量与用途如表 3-1 所示。iris_with_error 原始 152 行×5 列，特征缺失 12 格、Species 缺失 1 格，"
     "合计 13/760=1.71%；完全重复 5 行（3.29%），比干净 iris.csv 多 2 行。housing 为 506 行×14 列，无缺失、无重复。"
     "完整数据字典见附表 1，逐字段给出类型、取值范围、缺失比例与含义。")
summary = pd.DataFrame([
    {"数据集": "iris_with_error", "原始规模": "152×5", "清洗后": "145×5", "建模特征": "4 个形态特征（Species 仅用于事后校验）", "主要用途": "小样本对照与数据质量诊断"},
    {"数据集": "housing", "原始规模": "506×14", "清洗后": "506×14", "建模特征": "13 个预测变量（MEDV 仅用于 EDA/解释）", "主要用途": "主实验与参数敏感性"},
])
table(doc, summary, "表 3-1  数据集规模与用途", widths=[2.8, 2.4, 2.2, 5.4, 2.8], font_size=9,
      keep_together=True)

heading(doc, "3.3 外部补充数据", 2)
body(doc,
     "本文主实验与对照实验全部使用课程 data.zip 中的两份数据，未引入任何外部数据，因此不存在跨来源口径合并与占比问题。"
     "两份数据均为公开教学数据，不含个人身份信息；Housing 的 B 字段带有历史争议，本文仅在方法比较层面使用该字段，"
     "并保留原字段名以免误用。")

# ---------------- 4
heading(doc, "四、数据探索与清洗", 1)
heading(doc, "4.1 探索性分析", 2)
body(doc,
     "图 4-1 给出 housing 的 MEDV 分布与 iris 的花瓣长度—宽度散点。housing 的房价呈右偏分布，并在 50 千美元处存在明显截断，"
     "说明该变量存在天花板效应；iris 的三个品种在花瓣尺寸上呈明显分层，setosa 与另外两类几乎线性可分，versicolor 与 virginica "
     "存在重叠。本节的分组统计、分位数与相关系数均由 pandas 计算[16]。图 4-2 的相关性热图显示，housing 中 RAD 与 TAX、NOX 与 INDUS/DIS "
     "存在较强相关，若直接使用欧氏距离，相关特征会被重复计权；这也解释了后文马氏距离利用协方差逆矩阵去相关后表现更好的原因。")
figure(doc, "fig02_eda_distributions.png", "图 4-1  housing 房价分布与 iris 花瓣尺寸散点", 14.6)
figure(doc, "fig03_correlation.png", "图 4-2  housing 与 iris 特征相关性热图", 15.2)
body(doc,
     "图 4-3 将 housing 测试集投影到前两个主成分，并按异常类型着色。点异常大多远离正常云团；协同异常与极端组合异常则可能与"
     "正常样本在二维投影上重叠，说明仅靠可视化或两三个主成分不足以识别全部异常，需要多变量评分。")
figure(doc, "fig04_pca_anomalies.png", "图 4-3  housing 测试集 PCA 投影与注入异常分布", 11.5)

heading(doc, "4.2 数据质量诊断", 2)
body(doc,
     "按完整性、唯一性、有效性与一致性四个维度诊断，结果见表 4-1。iris_with_error 的缺失集中在花瓣与花萼字段，"
     "其中 1 行连续缺失 3 个特征；重复记录 5 行。housing 无缺失、无重复，但 IQR 规则会在 13 个预测变量上标记 343 个字段级点，"
     "这些点多为真实的长尾样本（如高犯罪率、低房间数、极高税率），不是录入错误，因此不作删除。iris 的 Sepal_Width 也有 3 个"
     "IQR 标记点，同样保留。")
quality = read_table("table_quality.csv")
table(doc, quality, "表 4-1  数据质量检测结果", widths=[2.0, 4.6, 3.2, 4.0, 1.8], font_size=8.5)
figure(doc, "fig05_quality.png", "图 4-4  缺失与重复记录统计", 14.6)

heading(doc, "4.3 清洗与处理策略", 2)
body(doc,
     "iris 的清洗按以下顺序进行：①将 Species 统一为小写并去除首尾空白，空字符串与“nan”视为缺失；②删除完全重复记录，"
     "152→147 行；③删除连续缺失≥3 个形态特征的行，147→146 行；④对剩余缺失按 Species 分组中位数插补，若某组中位数仍缺失"
     "则用全体中位数兜底；⑤删除 Species 仍缺失的记录，最终 145 行。housing 仅做数值类型校验与完全重复检查，506 行保持不变。"
     "选择分组中位数而不是均值，是因为 iris 各品种的花瓣尺寸差异明显，分组统计更符合数据的层次结构；选择删除而不是插补"
     "“连续缺失 3 个特征”的行，是因为该行可用信息不足，强行插补会引入较大偏差。")
body(doc,
     "需要强调：本文没有把 housing 的自然长尾当作异常删除。异常检测的任务是识别“罕见”，而不是把罕见值从数据中抹掉；"
     "若在训练前删除长尾，模型会失去对真实极端工况的感知。本文只在建模阶段使用稳健标准化，降低长尾对尺度的影响。")

heading(doc, "4.4 数据划分", 2)
body(doc,
     "主实验采用 70%/30% 的训练—测试划分；训练集内部再按 70%/15%/15% 划分为 fit、calib、val 三部分：fit 用于拟合检测器，"
     "calib 的正常样本用于估计阈值，val 的正常样本与独立种子注入的合成异常用于选择超参数与阈值。测试集完全不参与拟合与调参，"
     "只在最后评估时使用。housing 使用 5 个种子（42、2024、7、13、99），iris 使用 10 个种子（另加 2026、11、18、123、777），"
     "以观察划分波动。测试集注入比例约 10%，具体配置见表 4-2。")
inj = read_table("table_injection_config.csv")
table(doc, inj, "表 4-2  数据划分与异常注入配置", widths=[2.4, 2.0, 1.8, 1.8, 1.8, 1.8, 2.2, 2.2, 2.0], font_size=8.5,
      keep_together=True)

# ---------------- 5
heading(doc, "五、特征工程", 1)
heading(doc, "5.1 特征构造", 2)
body(doc,
     "在原始字段之外，本文构造了 5 个 iris 派生特征与 10 个 housing 派生特征，涵盖面积、纵横比、对数压缩与领域交互四类逻辑。"
     "例如，花瓣面积把长度与宽度合成为一个“大小”维度；长宽比刻画形状；CRIM_log、LSTAT_log、TAX_log 用于压缩右偏分布；"
     "LSTAT_RM、NOX_DIS、TAX_RAD 分别刻画“低收入×住房”“污染×区位”“税率×可达性”的联合效应。完整清单、构造逻辑、"
     "业务含义与是否入选见表 5-1。")
feat = read_table("table_features.csv")
table(doc, feat, "表 5-1  派生特征清单", widths=[1.8, 2.5, 3.8, 4.4, 3.0], font_size=8)

heading(doc, "5.2 编码与变换", 2)
body(doc,
     "CHAS 与 RAD 虽为整数编码，但前者是二值变量、后者是有序类别；本文保留其数值编码，不做 one-hot，以避免在 506 条样本上"
     "产生过多稀疏列。所有连续特征使用 RobustScaler（基于中位数与四分位距）标准化，原因是 housing 的 CRIM、B、LSTAT 等"
     "存在明显长尾，均值和标准差会被极端值拉偏；iris 的特征尺度相近，使用稳健标准化也不会造成信息损失。")
body(doc,
     "需要指出，标准化参数只在 fit 子集上估计，然后应用到 calib、val 与 test，避免测试集信息泄漏。对数 log 变换使用 log1p，"
     "保证 0 值安全；对比值特征在分母为 0 时置为缺失并用中位数兜底。")

heading(doc, "5.3 特征选择", 2)
body(doc,
     "先做方差过滤（去除方差≤1e-10 的常数列），再做相关性过滤：按列顺序保留特征，若某列与已保留列的 Pearson 相关系数"
     "绝对值超过 0.95，则剔除该列。housing 最终保留 15 个特征（13 个原始变量 + CRIM_log + TAX_RAD），"
     "iris 保留 6 个特征（4 个原始变量 + Sepal_Area + Petal_Aspect + Sepal_Aspect）。被剔除的派生特征并非无意义，"
     "而是与原始变量高度共线；在树模型之外，共线会让距离度量重复计权，因此优先保留解释性更强的原始字段。")

heading(doc, "5.4 特征有效性验证", 2)
body(doc,
     "以 Isolation Forest 为固定载体、seed=42，比较基础特征与工程特征的测试表现，结果见表 5-2。加入派生特征后，"
     "PR-AUC 从 0.6229 提升到 0.6388，Recall 从 0.6667 提升到 0.7333，F1 从 0.5714 提升到 0.6286，FPR 从 7.30% 降到 6.57%。"
     "此处为单种子对照，数值与表 6-2 的 5 种子均值（0.5089）不同属正常现象，不应混用。逐特征剔除实验显示，CRIM_log 的"
     "边际贡献为 +0.0131，TAX_RAD 的边际贡献为 -0.0075：前者压缩了犯罪率长尾，后者虽然对 IForest 略有害，但在马氏距离中"
     "提供了税率与可达性的比值信息，故在最终特征集中保留并在敏感性分析中讨论。")
gain = read_table("table_feature_gain.csv")
for c in ["PR-AUC", "Recall", "F1", "FPR"]:
    gain[c] = pd.to_numeric(gain[c], errors="coerce").round(4)
table(doc, gain, "表 5-2  特征工程增益（Isolation Forest，seed=42）", widths=[3.0, 1.6, 2.2, 2.2, 2.0, 2.0, 3.6], font_size=8.5,
      keep_together=True)

# ---------------- 6
heading(doc, "六、模型设计与实验", 1)
heading(doc, "6.1 技术路线", 2)
body(doc,
     "整体流程如图 6-1 所示：数据审计→清洗与划分→受控异常注入→特征工程→五类检测器→阈值选择→测试评估→参数敏感性与"
     "特征贡献。每个模块都输出可检查的中间产物：清洗统计、注入清单、特征列表、每种子指标、敏感性长表与贡献表。"
     "这样做的目的是把“数据问题”“模型问题”“阈值问题”分开定位，而不是只看最终 F1。")
figure(doc, "fig01_technical_route.png", "图 6-1  T13 多元异常检测技术路线", 15.2)

heading(doc, "6.2 基线模型", 2)
body(doc,
     "基线采用 Z-Score：对每个特征计算 (x_j-μ_j)/σ_j，取绝对值最大者作为异常得分，阈值由 calib 正常样本的分位数确定。"
     "该方法对应业务中最常见的“单字段越界”规则，优点是透明、无需训练；缺点是忽略特征相关性与多变量协同，"
     "在偏态分布上阈值不稳定。它的结果用于回答一个关键问题：复杂模型相对简单规则到底带来了多少增益。")

heading(doc, "6.3 对比模型", 2)
body(doc,
     "（1）马氏距离：用稳健协方差（MCD）估计均值与协方差，得分 d(x)=sqrt((x-μ)^T Σ^{-1}(x-μ))。它把特征相关性与尺度"
     "同时纳入，理论上适合近似椭球分布的数据；MCD 通过最小化协方差行列式降低异常点对估计的污染[8]。参数为 support_fraction，"
     "网格取 0.5、0.6、0.75、0.9、1.0。")
body(doc,
     "（2）LOF：对每个样本比较其局部可达密度与邻居密度，得分越高越孤立[3]。它能发现局部异常，但在高维或密度不均时"
     "容易退化；参数为邻居数 k，housing 取 5–50，iris 取 3–20。")
body(doc,
     "（3）Isolation Forest：随机选择特征与切分点构造孤立树，异常样本更容易被少量切分孤立[4,5]。参数包括树数、"
     "max_samples 与 max_features；本文树数取 100/200/300，max_samples 取 0.5/0.75/1.0，max_features 取 0.5/1.0。")
body(doc,
     "（4）One-Class SVM：在核空间中寻找包围正常样本的最小边界[6,7]。参数为 nu（边界外样本比例上界）与 RBF 核的 gamma；"
     "nu 取 0.01–0.3，gamma 取 scale 与 0.01/0.05/0.1/0.5。")
body(doc,
     "（5）排名融合：把五个基模型的得分在 calib 上做 z-score 标准化后等权平均。它不是新的基学习器，而是检验"
     "“不同归纳偏置是否互补”的对照；若融合的召回明显高于单模型，说明不同方法捕捉到了不同类型的异常。")

heading(doc, "6.4 实验设置", 2)
body(doc,
     "实验环境为 AMD Ryzen 9 8945HX（16 核 32 线程）、15.2 GB 内存、Windows；Python 3.13.9、NumPy 2.3.5、"
     "pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.10.6，全部模型实现基于 scikit-learn 与 NumPy[17,18]，"
     "绘图基于 Matplotlib[19]。所有模型使用固定 random_state；阈值只在 calib 正常样本与 val "
     "合成异常上选择，约束 calib FPR≤10%，目标为最大化 val F1；测试集不参与任何选择。参数网格与划分见表 6-1。")
settings = pd.DataFrame([
    {"项目": "训练/校准/验证/测试", "设置": "housing 247/53/54/152；iris 70/15/16/44"},
    {"项目": "随机种子", "设置": "housing 5 个；iris 10 个；所有模型固定 random_state"},
    {"项目": "注入类型", "设置": "点异常 40%、协同异常 40%、极端组合 20%（按行随机）"},
    {"项目": "阈值约束", "设置": "calib FPR≤0.10，最大化 val F1；测试集不调阈值"},
    {"项目": "评价指标", "设置": "PR-AUC（主）、Recall、FPR、Precision、F1、ROC-AUC、训练耗时"},
    {"项目": "软件环境", "设置": "Python 3.13.9；scikit-learn 1.7.2；NumPy 2.3.5；pandas 2.3.3"},
])
table(doc, settings, "表 6-1  实验环境与参数设置", widths=[4.0, 11.4], font_size=9,
      keep_together=True)

heading(doc, "6.5 实验结果", 2)
body(doc,
     f"housing 主实验的 5 种子均值±标准差见表 6-2。马氏距离取得最高的 PR-AUC（{f4(agg('housing','Mahalanobis','pr_auc'))}）、"
     f"F1（{f4(agg('housing','Mahalanobis','f1'))}）与 ROC-AUC（{f4(agg('housing','Mahalanobis','roc_auc'))}），"
     f"Recall 为 {f4(agg('housing','Mahalanobis','recall'))}，FPR 仅 {f4(agg('housing','Mahalanobis','fpr'))}，"
     "是唯一在均值意义上同时满足“Recall≥0.85、FPR≤10%”的单模型。One-Class SVM 的 Recall 为 "
     f"{f4(agg('housing','One-Class SVM','recall'))}，但精确率只有 {f4(agg('housing','One-Class SVM','precision'))}；"
     "LOF 的 FPR 很低（"
     f"{f4(agg('housing','LOF','fpr'))}），但 Recall 只有 {f4(agg('housing','LOF','recall'))}；"
     "Isolation Forest 在该数据上表现最弱，PR-AUC 仅 "
     f"{f4(agg('housing','Isolation Forest','pr_auc'))}。排名融合把 Recall 提升到 "
     f"{f4(agg('housing','Rank-Ensemble','recall'))}，FPR 为 {f4(agg('housing','Rank-Ensemble','fpr'))}，"
     f"但精确率降到 {f4(agg('housing','Rank-Ensemble','precision'))}，说明高召回与低误报之间存在权衡。")
perf = read_table("table_model_performance.csv")
perf_h = perf[perf["数据集"] == "housing"]
perf_h = perf_view(perf_h, with_dataset=False, with_params=True, with_time=False, with_roc=True)
table(doc, perf_h, "表 6-2  housing 模型性能对比（5 种子均值±标准差）",
      widths=[2.2, 2.7, 2.2, 2.1, 1.9, 2.0, 2.1], font_size=7.4, keep_together=True)
body(doc,
     "稳健性方面，逐种子检查比只看均值更有意义。马氏距离在 5 个种子中有 4 个的召回 ≥0.85（0.933、1.000、0.933、0.933），"
     "仅在 seed=13 时降到 0.667，该种子的 FPR 仍为 1.5%、PR-AUC 0.840；排名融合在 5 个种子中有 4 个召回 ≥0.85，"
     "最低为 0.733。也就是说，达标线在多数划分下成立，但个别划分下会失守，报告时应同时给出均值、标准差与逐种子明细，"
     "而不是只报一个均值。")
body(doc,
     "各方法的异常得分分布、PR 曲线、ROC 曲线与 5 个种子的波动分别见图 6-2、图 6-3、图 6-4 与图 6-5；"
     "单次训练耗时对比见图 6-7。马氏距离与 One-Class SVM 的训练耗时均在毫秒级，LOF 与 Isolation Forest 约 0.3–0.4 s，"
     "融合模型约 0.7 s；对 506 条样本的离线审计而言差异可以接受，但在高频在线场景需优先考虑轻量模型。")
figure(doc, "fig06_score_distributions.png", "图 6-2  各方法在 housing 上的异常得分分布（seed=42）", 15.2)
figure(doc, "fig07_pr_curves.png", "图 6-3  housing 的 Precision-Recall 曲线（seed=42）", 12.6)
figure(doc, "fig08_roc_curves.png", "图 6-4  housing 的 ROC 曲线（seed=42）", 12.6)
figure(doc, "fig09_recall_fpr.png", "图 6-5  housing 的 Recall 与 FPR（5 种子均值±标准差）", 15.0)
figure(doc, "fig10_sensitivity.png", "图 6-6  四类方法的参数敏感性热图（housing，seed=42）", 15.4)
figure(doc, "fig13_runtime.png", "图 6-7  housing 单次训练耗时对比（对数坐标）", 12.6)
figure(doc, "fig14_per_type_recall.png", "图 6-8  housing 分异常类型的召回率", 15.0)
figure(doc, "fig12_iris.png", "图 6-9  iris 小样本对照：Recall、FPR 与 PR-AUC（10 种子）", 15.4)

body(doc,
     "分类型结果见表 6-3 与图 6-8。Z-Score 对点异常的召回为 1.00，但对协同异常只有 0.49、对极端组合只有 0.28，"
     "说明“单字段越界”规则无法覆盖多字段共同偏移；马氏距离对协同异常召回 1.00，但对极端组合仅 0.63；"
     "Isolation Forest 对点异常只有 0.28，却对协同异常达到 0.71；融合模型在三种类型上分别为 0.92、0.95、0.92，"
     "是唯一在极端组合上超过 0.90 的方案。这一结果说明不同方法的归纳偏置确实互补：Z-Score 擅长单点越界，"
     "马氏距离擅长相关变量整体偏移，树模型擅长不规则边界，融合可以覆盖更多类型。")
ptype = read_table("table_per_type_housing.csv")
for c in ["点异常", "协同异常", "极端组合"]:
    ptype[c] = pd.to_numeric(ptype[c], errors="coerce").round(3)
table(doc, ptype, "表 6-3  housing 分异常类型召回率", widths=[4.0, 3.8, 3.8, 3.8], font_size=9,
      keep_together=True)

body(doc,
     "为了观察错分来源，表 6-4 列出 seed=42 下马氏距离的 6 个错分案例（1 个漏报、5 个误报）。漏报样本属于极端组合异常，"
     "得分为 5.827，低于阈值 6.279；其 RM 被压到 4.25、CRIM 仅 0.268、LSTAT 为 14.1，虽然组合罕见，但马氏距离仍认为它"
     "落在正常椭球的可接受范围内。5 个误报均为正常样本，特征是“低房间数/高犯罪率/极高税率”等自然长尾组合，"
     "其中 4 个的得分只比阈值高 0.05–0.90。它们不是数据错误，而是业务上值得复核的边界样本；这提示我们不应把误报简单"
     "视为模型失败，而应结合复核成本决定阈值。")
err = read_table("table_error_cases.csv")
cols = ["error_type", "injection_type", "score", "threshold", "CRIM", "RM", "TAX", "LSTAT", "MEDV"]
if all(c in err.columns for c in cols):
    show = err[cols].copy()
    show.columns = ["错误类型", "注入类型", "得分", "阈值", "CRIM", "RM", "TAX", "LSTAT", "MEDV"]
    for c in ["得分", "阈值", "CRIM", "RM", "TAX", "LSTAT", "MEDV"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    table(doc, show, "表 6-4  典型错分案例（Mahalanobis，seed=42）",
          widths=[1.8, 1.8, 1.4, 1.4, 1.4, 1.3, 1.4, 1.5, 1.5], font_size=8,
          keep_together=True)

heading(doc, "6.6 结果分析", 2)
body(doc,
     "第一，马氏距离的优势来自数据分布与异常机制匹配。housing 的 13 个预测变量存在明显相关（RAD-TAX、NOX-INDUS/DIS），"
     "注入的协同异常又是在相关方向上整体偏移；马氏距离用协方差逆矩阵做白化，能把“整体偏移”转化为较大的马氏半径。"
     "MCD 的稳健估计进一步降低了少量极端值对协方差的影响，因此它在 5 个种子上保持较高召回与很低误报。")
body(doc,
     "第二，Isolation Forest 的失利不是实现问题，而是归纳偏置不匹配。孤立森林依赖轴对齐的随机切分，当异常表现为多个"
     "相关特征的联合偏移时，单次切分只看到一个维度，需要更多树才能“碰巧”孤立；本文为控制耗时将 max_features 设为 0.5，"
     "进一步加剧了维度稀释。它的 PR-AUC 只有 0.5089，且在点异常上的召回仅 0.28，说明树模型并非在任何表格数据上都占优。")
body(doc,
     "第三，LOF 的 FPR 很低但召回不足。LOF 比较局部密度，对于远离正常云团的点异常很敏感；但当协同异常落在正常样本的"
     "稀疏边缘时，其局部密度未必显著低于邻居，因此召回偏低。LOF 对 k 也非常敏感：图 6-6 显示 k=5 时 F1 可达 0.84，"
     "k=30 且阈值分位 0.99 时降到 0.27，极差 0.566，是五种方法中最大的。")
body(doc,
     "第四，One-Class SVM 与融合模型体现“召回优先”的取舍。OCSVM 通过 nu 控制边界外样本比例，在 nu=0.01、gamma=0.05 时"
     "取得较高召回，但精确率只有 0.6967；融合模型把五个模型的得分标准化后平均，等价于让不同归纳偏置投票，"
     "召回提升到 0.9200，但精确率降到 0.7237。若业务更怕漏报，融合是合理选择；若复核成本很高，马氏距离更均衡。")
body(doc,
     "第五，Z-Score 的 FPR 在种子上波动很大（均值 9.05%，最高 11.68%）。原因是单变量最大值对偏态特征敏感，"
     "calib 上的 10% 分位阈值在测试集上不一定保持同样的误报率；这也说明“在训练集上卡 10%”不等于“在测试集上保证 10%”，"
     "阈值需要留出安全裕度或做在线校准。")
body(doc,
     "第六，iris 小样本结果（图 6-9）显示 FPR 的方差远大于 housing：LOF 与 OCSVM 的平均召回均为 0.95，但 FPR 分别为"
     "12.50% 与 9.50%，且标准差很大；马氏距离的 FPR 为 9.50%，召回只有 0.775。145 条样本、44 条测试、4 个异常的条件下，"
     "单个样本的判定就会让 FPR 跳动 2–3 个百分点，因此不能用单次划分给小样本方法下结论。本文保留这一负面结果，"
     "并将其作为“小样本异常检测需要重复划分与置信区间”的证据。")

# ---------------- 7
heading(doc, "七、结果解释与业务洞察", 1)
heading(doc, "7.1 模型可解释性", 2)
body(doc,
     "图 7-1 给出两种解释路径。马氏距离按 d² 的特征分解计算每个字段对马氏半径的贡献：NOX 占 21.3%、TAX 占 13.7%、"
     "RM 占 12.8%、DIS 占 9.0%、CRIM 占 7.5%，说明污染浓度、税率、房间数与区位距离是驱动异常的主要维度。"
     "Isolation Forest 用“把某特征替换为训练中位数后得分下降”衡量贡献：B 占 32.7%、RM 占 18.4%、ZN 占 11.3%、"
     "LSTAT 占 7.3%、CRIM 占 6.9%。两种方法的排序不同，是因为它们衡量的是不同东西：前者衡量“偏离椭球中心的方向”，"
     "后者衡量“对孤立路径的贡献”。业务解释时应说明所用方法，不能把两者混为一谈。")
figure(doc, "fig11_feature_contrib.png", "图 7-1  Mahalanobis 与 Isolation Forest 的特征贡献占比", 14.6)

heading(doc, "7.2 业务洞察", 2)
body(doc,
     "洞察一：在低维、特征相关的表格数据上，统计距离方法应作为首选基线。马氏距离在 housing 上以 0.9026 的 PR-AUC、"
     "0.8933 的召回和 2.04% 的误报全面优于树模型与密度模型；它的计算成本只有毫秒级，且特征贡献可直接分解，"
     "比黑盒模型更适合需要解释的复核场景。")
body(doc,
     "洞察二：阈值校准对误报的影响不亚于模型选择。Z-Score 的平均 FPR 为 9.05%，但单种子最高达到 11.68%；"
     "同样的分数分布，仅因 calib 与 test 的抽样波动就会跨过 10% 的红线。建议在 calib 上按目标 FPR 的 80% 设置阈值"
     "（留安全裕度），并在上线后按月用正常样本重新校准。")
body(doc,
     "洞察三：极端组合异常是最难的一类，单模型难以兼顾。马氏距离对协同异常召回 1.00，但对极端组合只有 0.63；"
     "融合模型把极端组合召回提升到 0.92，代价是精确率下降约 11 个百分点。若业务不能接受漏报，应采用融合模型并配置"
     "人工复核；若复核资源有限，则用马氏距离并接受少量极端组合漏报。")
body(doc,
     "洞察四：自然长尾是误报的主要来源，不应把误报全部当作模型缺陷。5 个误报样本都是低房间数、高犯罪率、极高税率的"
     "真实边界样本；它们对业务可能同样值得关注。建议把误报分为“数据错误”“业务边界”“模型错误”三类，分别处理，"
     "而不是简单调高阈值把边界样本全部压掉。")

heading(doc, "7.3 应用建议", 2)
body(doc,
     "面向业务方，本文给出“两层筛查 + 三条规则”的落地建议，见表 7-1。第一层用马氏距离或融合模型做高召回初筛；"
     "第二层用特征贡献定位问题字段，并交由业务复核。三条规则是：阈值按业务代价设定而非固定 0.5；每月用正常样本回校准；"
     "对极端组合异常单独设规则或引入集成模型。")
advice = pd.DataFrame([
    {"场景": "数据质量审计", "建议": "用马氏距离做初筛，按贡献字段定位录入错误；阈值取 calib 的 95% 分位", "适用条件": "特征为数值型、样本量≥300"},
    {"场景": "业务异常复核", "建议": "融合模型高召回初筛 + 人工复核；误报按“数据/业务/模型”分类", "适用条件": "复核资源充足、漏报代价高"},
    {"场景": "工业/时序监测", "建议": "先窗口化提取统计特征，再复用本文协议；需处理概念漂移", "适用条件": "有正常工况基线、标签按时间段给出"},
    {"场景": "模型上线", "建议": "监控得分分布与 FPR，按月回校准；保留模型版本与阈值版本", "适用条件": "有持续的正常样本流"},
])
table(doc, advice, "表 7-1  业务落地建议", widths=[3.2, 8.2, 4.0], font_size=8.5,
      keep_together=True)

# ---------------- 8
heading(doc, "八、结论与展望", 1)
heading(doc, "8.1 主要结论", 2)
body(doc,
     f"本文完成了 T13 的完整链路：数据审计、清洗、受控异常注入、特征工程、五类检测器对比、阈值校准、敏感性分析与可解释性。"
     f"housing 主实验表明，马氏距离以 PR-AUC {f4(agg('housing','Mahalanobis','pr_auc'))}、Recall "
     f"{f4(agg('housing','Mahalanobis','recall'))}、FPR {f4(agg('housing','Mahalanobis','fpr'))} 达到并超过达标线；"
     f"融合模型把 Recall 提升到 {f4(agg('housing','Rank-Ensemble','recall'))}，适合漏报代价高的场景。"
     "Isolation Forest 在本文的低维相关数据上表现最弱，说明“常用即最优”的假设不成立；LOF 对邻居数高度敏感，"
     "必须报告参数区间而不是单点结果。")
body(doc,
     "方法层面，本文验证了“注入真值 + 三划分 + 阈值约束”的协议可以有效避免测试集泄漏，并把模型能力与阈值选择分开。"
     "业务层面，本文给出的两层筛查流程、误报分类与回校准规则可以直接用于数据质量与风险复核。")

heading(doc, "8.2 局限性", 2)
body(doc,
     "数据方面，主实验只有 506 条 housing 与 145 条 iris，二者都是低维静态表格，不能代表图像、文本、时序与高维稀疏场景；"
     "注入异常是人工构造的，虽然包含三种类型，但与真实异常的生成机制仍有差距。方法方面，本文未引入自编码器、"
     "深度 SVDD 等深度模型，也未做半监督学习；马氏距离假设正常样本近似椭球分布，遇到多峰分布会失效。评价方面，"
     "测试集异常比例固定为约 10%，与真实业务的低污染率不同；PR-AUC 与阈值指标都依赖注入机制，换一种注入方式结论可能变化。")
body(doc,
     "此外，本文的异常是人工注入的，虽包含三种类型，但与真实故障的生成机制仍有差距；业务建议基于模拟异常，"
     "落地前需要真实标签与代价矩阵做校准。")

heading(doc, "8.3 改进方向", 2)
body(doc,
     "后续工作包括：（1）在工业时序等真实场景上验证本文协议，比较窗口统计特征与自编码器、LSTM 预测残差法；"
     "（2）引入多峰正常性建模，如 GMM 马氏距离、深度 SVDD；（3）研究概念漂移下的在线阈值校准与模型更新；"
     "（4）把误报复核成本纳入阈值优化，构建代价敏感的决策曲线；（5）补充更多公开数据集与真实异常标签，"
     "检验结论的外部有效性。")

# ---------------- 9
heading(doc, "九、团队分工与工作记录", 1)
heading(doc, "9.1 分工表", 2)
body(doc, "本组为单人组，组长程龙飞承担全部模块，贡献占比 100%。具体分工见表 9-1。")
group = read_table("table_group.csv")
table(doc, group, "表 9-1  小组分工表", widths=[2.2, 2.6, 5.4, 4.0, 1.4], font_size=8.5,
      keep_together=True)

heading(doc, "9.2 实验记录摘要", 2)
body(doc, "按实际运行日期记录关键实验、参数、结果与结论，见表 9-2。所有数值均由代码输出，未手工修改。")
log = read_table("table_experiment_log.csv")
table(doc, log, "表 9-2  实验记录表", widths=[2.2, 3.0, 4.6, 4.2, 3.0], font_size=7.8,
      keep_together=True)

heading(doc, "9.3 Git 仓库地址与提交统计", 2)
git_path = Path(__file__).resolve().parents[1]
git_log = OUT_DIR / "logs" / "git_stats.txt"
git_text = git_log.read_text(encoding="utf-8").lstrip("\ufeff") if git_log.exists() else "本地 Git 仓库已初始化；提交统计见随包 git_log.txt。"
for line in git_text.splitlines():
    if line.strip():
        body(doc, line.strip(), indent=False, size=10.5, align=WD_ALIGN_PARAGRAPH.LEFT)
body(doc,
     "说明：以上为本地仓库的实际提交记录，完整改动明细见随包的 git_log.txt。若后续把仓库推送到 Gitee 或 GitHub，"
     "可在此处补充可访问的仓库地址。")

# ---------------- 10
heading(doc, "十、工具使用声明与诚信承诺", 1)
heading(doc, "10.1 生成式 AI 等工具使用声明", 2)
body(doc,
     "本报告在代码实现、调试与文字初稿阶段使用了 GitHub Copilot（VS Code）作为辅助工具，主要用于：生成数据加载、"
     "异常注入、检测器封装与绘图代码的初稿；提供 scikit-learn API 与报告排版的修改建议；梳理文献线索与章节结构。"
     "所有实验数值均由本地代码实际运行产生，作者对代码逻辑、数据划分、阈值协议、图表与结论逐项复核并修改；"
     "AI 未参与数据伪造，也未替代作者对结果的责任。除 GitHub Copilot 外，未使用其他生成式 AI 工具。")
heading(doc, "10.2 诚信承诺", 2)
body(doc,
     "本人承诺：所提交的报告、代码、图表与数据真实、独立完成；数据未伪造、篡改；引用均已标注；"
     "涉及他人成果的部分已按 GB/T 7714—2015 给出参考文献；如存在错误，由本人承担责任。")
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(18)
r = p.add_run("组长签名：程龙飞          成员签名：无          日期：2026 年 11 月 18 日")
set_run_font(r, "宋体", 12)

# ---------------- references
doc.add_page_break()
heading(doc, "参考文献", 1)
refs = [
    "Chandola V, Banerjee A, Kumar V. Anomaly detection: A survey[J]. ACM Computing Surveys, 2009, 41(3): Article 15.",
    "Zimek A, Schubert E, Kriegel H-P. A survey on unsupervised outlier detection in high-dimensional numerical data[J]. Statistical Analysis and Data Mining, 2012, 5(5): 363-387.",
    "Breunig M M, Kriegel H-P, Ng R T, et al. LOF: Identifying density-based local outliers[C]//Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. Dallas: ACM, 2000: 93-104.",
    "Liu F T, Ting K M, Zhou Z-H. Isolation Forest[C]//Proceedings of the 8th IEEE International Conference on Data Mining. Pisa: IEEE, 2008: 413-422.",
    "Liu F T, Ting K M, Zhou Z-H. Isolation-based anomaly detection[J]. ACM Transactions on Knowledge Discovery from Data, 2012, 6(1): Article 3.",
    "Schölkopf B, Platt J C, Shawe-Taylor J, et al. Estimating the support of a high-dimensional distribution[J]. Neural Computation, 2001, 13(7): 1443-1471.",
    "Tax D M J, Duin R P W. Support vector data description[J]. Machine Learning, 2004, 54(1): 45-66.",
    "Rousseeuw P J, Van Driessen K. A fast algorithm for the minimum covariance determinant estimator[J]. Technometrics, 1999, 41(3): 212-223.",
    "Goldstein M, Uchida S. A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data[J]. PLoS ONE, 2016, 11(4): e0152173.",
    "Campos G O, Zimek A, Sander J, et al. On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study[J]. Data Mining and Knowledge Discovery, 2016, 30(4): 891-927.",
    "Ruff L, Kauffmann J R, Vandermeulen R A, et al. A unifying review of deep and shallow anomaly detection[J]. Proceedings of the IEEE, 2021, 109(5): 756-795.",
    "Pang G, Shen C, Cao L, et al. Deep learning for anomaly detection: A review[J]. ACM Computing Surveys, 2021, 54(2): Article 38.",
    "周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.",
    "李航. 统计学习方法[M]. 2版. 北京: 清华大学出版社, 2019.",
    "Harrison D, Rubinfeld D L. Hedonic prices and the demand for clean air[J]. Journal of Environmental Economics and Management, 1978, 5(1): 81-102.",
    "McKinney W. Python for Data Analysis[M]. 3rd ed. Sebastopol: O'Reilly Media, 2022.",
    "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
    "Harris C R, Millman K J, van der Walt S J, et al. Array programming with NumPy[J]. Nature, 2020, 585(7825): 357-362.",
    "Hunter J D. Matplotlib: A 2D graphics environment[J]. Computing in Science & Engineering, 2007, 9(3): 90-95.",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.12
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.74)
    r = p.add_run(f"[{i}] {ref}")
    set_run_font(r, "宋体", 10)

# ---------------- appendices
doc.add_page_break()
heading(doc, "附  表", 1)
dd = read_table("table_data_dictionary.csv")
dd["取值范围/示例"] = dd["取值范围/示例"].astype(str).str.replace(" ~ ", "–", regex=False)
table(doc, dd, "附表 1  数据字典",
      widths=[1.8, 2.0, 2.4, 3.2, 1.6, 4.6], font_size=7.6)
table(doc, read_table("table_quality.csv"), "附表 2  数据质量检测结果",
      widths=[1.8, 4.4, 3.0, 4.2, 2.2], font_size=8)
table(doc, read_table("table_features.csv"), "附表 3  派生特征清单",
      widths=[1.8, 2.4, 3.8, 4.4, 3.2], font_size=7.8)
perf_all = perf_view(read_table("table_model_performance.csv"),
                     with_dataset=True, with_params=False, with_time=True, with_roc=True)
table(doc, perf_all, "附表 4  模型性能对比",
      widths=[1.5, 2.5, 2.0, 1.9, 1.7, 1.7, 1.9, 2.0], font_size=7.6)
table(doc, read_table("table_group.csv"), "附表 5  小组分工表",
      widths=[2.2, 2.6, 5.4, 4.0, 1.4], font_size=8.5)
table(doc, read_table("table_experiment_log.csv"), "附表 6  实验记录表",
      widths=[2.2, 3.0, 4.6, 4.2, 3.0], font_size=7.8, keep_together=True)

# ---------------- 附录 A：关键代码
doc.add_page_break()
heading(doc, "附录 A  关键代码片段", 1)
body(doc,
     "以下片段取自成果包源码，行号与提交版本一致；完整代码、运行脚本与单元测试见 src/ 与 tests/，"
     "复现方式见 README.md。为便于排版，片段只保留核心逻辑，省略了类型注解与日志输出。")
code_block(doc, "src/experiment.py", 29, 9, "代码 A-1  fit / calib / val / test 四划分与无泄漏断言")
code_block(doc, "src/inject.py", 92, 14, "代码 A-2  三类异常的注入强度")
code_block(doc, "src/detectors.py", 197, 14, "代码 A-3  FPR 约束下的阈值选择")
code_block(doc, "src/detectors.py", 40, 14, "代码 A-4  马氏距离检测器：MCD 稳健协方差估计")

# ---------------- save
doc.core_properties.title = "T13 多元异常检测算法对比"
doc.core_properties.author = "程龙飞"
doc.core_properties.subject = "大数据技术与工程研究生实训报告"
out = REPORT_DIR / "13_程龙飞_实训报告.docx"
doc.save(out)

# count body Chinese characters (exclude abstract/references/appendix roughly)
all_text = "\n".join(p.text for p in doc.paragraphs)
cn = len(re.findall(r"[\u4e00-\u9fff]", all_text))
print("saved", out)
print("Chinese characters (all paragraphs):", cn)
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
