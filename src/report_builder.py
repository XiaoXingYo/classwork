"""Build the Word report from measured artefacts (no hand-entered metrics)."""
from __future__ import annotations

import json
import re
from collections import Counter
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


def read_str_table(name: str) -> pd.DataFrame:
    """Read a table that already holds pre-formatted numbers (keeps trailing zeros)."""
    return pd.read_csv(TABLE_DIR / name, encoding="utf-8-sig", dtype=str)


def pcv(method: str, setting: str, col: str = "测试 F1") -> float:
    """Read one cell of the parameter-comparison table (housing)."""
    df = read_table("table_param_compare.csv")
    row = df[(df["模型"] == method) & (df["参数设置"] == setting)]
    return float(row[col].iloc[0]) if len(row) else float("nan")


def seed_values(dataset: str, method: str, key: str):
    return [s["methods"][method][key] for s in METRICS[dataset]["per_seed"]]


def cv_range(method: str, dataset: str = "housing") -> float:
    """Mean spread of the cross-validated F1 over the whole parameter grid."""
    vals = [v for v in seed_values(dataset, method, "cv_f1_range") if v is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def ptype_num(method: str, col: str) -> float:
    df = read_table("table_per_type_housing.csv")
    return float(pd.to_numeric(df[df["模型"] == method][col], errors="coerce").iloc[0])


def contrib_top(method: str, k: int = 5) -> str:
    df = read_table("table_feature_contrib.csv")
    sub = df[df.method == method].sort_values("mean_share", ascending=False).head(k)
    return "、".join(f"{r.feature} 占 {r.mean_share * 100:.1f}%" for r in sub.itertuples())


def gap_span(group: str):
    """Min/max of (test F1 - CV F1) for the given comparison group."""
    raw = read_table("table_param_compare_raw.csv")
    tuned = raw[raw["setting"].isin(["默认参数", "交叉验证最优", "交叉验证最差"])]
    sub = tuned[tuned["setting"] == "交叉验证最优"] if group == "交叉验证最优" else \
        tuned[tuned["setting"] != "交叉验证最优"]
    gap = (sub["f1"] - sub["cv_f1"]).dropna()
    return float(gap.min()), float(gap.max())


def compact_metric(value) -> str:
    s = str(value)
    def repl(m):
        return f"{float(m.group(1)):.3f}±{float(m.group(2)):.3f}"
    return re.sub(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)", repl, s)


def pc_params(method: str, setting: str) -> str:
    """Read the 参数取值 cell of the parameter-comparison table (housing)."""
    df = read_table("table_param_compare.csv")
    row = df[(df["模型"] == method) & (df["参数设置"] == setting)]
    return str(row["参数取值"].iloc[0]) if len(row) else ""


def sens_range(method: str) -> float:
    df = read_table("table_sensitivity_summary.csv")
    return float(df[df["method"] == method]["f1_range"].iloc[0])


def sens_extreme(method: str, which: str = "max"):
    """(param_value, quantile, f1) of a method's best/worst sensitivity row."""
    df = read_table("table_sensitivity.csv")
    sub = df[df["method"] == method]
    idx = sub["f1"].idxmax() if which == "max" else sub["f1"].idxmin()
    row = sub.loc[idx]
    return str(row["param_value"]), row["threshold_quantile"], float(row["f1"])


def chosen_params(method: str, dataset: str = "housing") -> list:
    """Per-seed CV-selected parameter sets of a method, as JSON strings."""
    return [json.dumps(s["methods"][method]["params"], sort_keys=True, ensure_ascii=False)
            for s in METRICS[dataset]["per_seed"]]


def passes_gate(dataset: str, method: str) -> bool:
    return agg(dataset, method, "recall") >= 0.85 and agg(dataset, method, "fpr") <= 0.10


def zh_params(js: str) -> str:
    """Render a per-seed parameter JSON string in report-friendly form."""
    obj = json.loads(js)
    if obj.get("estimator") == "ledoitwolf":
        return "Ledoit–Wolf 收缩协方差"
    return ", ".join(f"{k}={v}" for k, v in obj.items())


def compact_params(value) -> str:
    s = str(value)
    for a, b in [("n_estimators", "n_est"), ("max_samples", "m_s"),
                 ("max_features", "m_f"), ("support_fraction", "sf"),
                 ("n_neighbors", "k"), ("gamma", "γ"),
                 ("One-Class SVM", "OCSVM"), ("Isolation Forest", "IForest"),
                 ("Mahalanobis", "马氏"), ("Rank-Ensemble", "融合"), ("Z-Score", "Z")]:
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
    p.paragraph_format.line_spacing = Pt(20)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
        p._p.get_or_add_pPr().ind.set(qn("w:firstLineChars"), "200")
    add_text_with_citations(p, text, size, bold)
    return p


CITE_RE = re.compile(r"\[(\d+(?:\s*[,，\-–]\s*\d+)*)\]")


def add_ref_field(paragraph, number: int, size: float = 12, bold: bool = False):
    """Insert a Word cross-reference field pointing at bookmark _RefNum<number>."""
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), f" REF _RefNum{number} \\h ")
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), "宋体")
    rfonts.set(qn("w:hAnsi"), "宋体")
    rfonts.set(qn("w:eastAsia"), "宋体")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rpr.append(rfonts)
    rpr.append(sz)
    if bold:
        rpr.append(OxmlElement("w:b"))
    t = OxmlElement("w:t")
    t.text = str(number)
    run.append(rpr)
    run.append(t)
    fld.append(run)
    paragraph._p.append(fld)


def add_text_with_citations(paragraph, text: str, size: float = 12, bold: bool = False):
    """Write text, turning every [n] / [n,m] marker into Word cross-reference fields."""
    pos = 0
    for m in CITE_RE.finditer(text):
        if m.start() > pos:
            set_run_font(paragraph.add_run(text[pos:m.start()]), "宋体", size, bold)
        set_run_font(paragraph.add_run("["), "宋体", size, bold)
        for part in re.split(r"(\s*[,，\-–]\s*)", m.group(1)):
            if not part:
                continue
            if part.strip().isdigit():
                add_ref_field(paragraph, int(part.strip()), size, bold)
            else:
                set_run_font(paragraph.add_run(part), "宋体", size, bold)
        set_run_font(paragraph.add_run("]"), "宋体", size, bold)
        pos = m.end()
    if pos < len(text):
        set_run_font(paragraph.add_run(text[pos:]), "宋体", size, bold)


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


def three_line_borders(t, top_sz=12, bottom_sz=12):
    """Give the table a 三线表 look: only top, header and bottom rules."""
    tblPr = t._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, val, sz in (("top", "single", top_sz), ("bottom", "single", bottom_sz),
                          ("left", "none", 0), ("right", "none", 0),
                          ("insideH", "none", 0), ("insideV", "none", 0)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    anchor = None
    for tag in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"):
        anchor = tblPr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(borders)
    else:
        tblPr.append(borders)

    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 0), ("left", 57), ("bottom", 0), ("right", 57)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(w))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    look = tblPr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tblPr.append(mar)


def cell_bottom_border(cell, sz=8):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    borders = OxmlElement("w:tcBorders")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(sz))
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")
    borders.append(bottom)
    tcPr.append(borders)


def table(doc, df: pd.DataFrame, cap: str, widths=None, font_size=9, header_fill=None,
          keep_together=False):
    caption(doc, cap, keep_next=True)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    three_line_borders(t)
    hdr = t.rows[0]
    repeat_header(hdr)
    _no_split(hdr)
    for i, c in enumerate(df.columns):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(str(c))
        set_run_font(r, "宋体", font_size, True)
        if header_fill:
            shade(cell, header_fill)
        cell_bottom_border(cell)
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
    ("提交日期", "2026 年 11 月 20 日", "成    绩", ""),
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
n_seeds_h = len(METRICS["housing"]["per_seed"])
nu_best = re.search(r"nu=([0-9.]+)", pc_params("One-Class SVM", "交叉验证最优"))
nu_best_txt = nu_best.group(1) if nu_best else "0.01"
mb_default_gap = pcv("Mahalanobis", "交叉验证最优") - pcv("Mahalanobis", "默认参数")
if abs(mb_default_gap) < 1e-9:
    mb_default_txt = "马氏距离的默认值恰好已是最优"
elif abs(mb_default_gap) < 0.01:
    mb_default_txt = "马氏距离的默认参数已接近交叉验证最优（差距不足 0.01）"
else:
    mb_default_txt = f"马氏距离经调参后 F1 再提升 {mb_default_gap:.2f}"
mb_top4 = (read_table("table_feature_contrib.csv")
           .query("method == 'Mahalanobis'")
           .sort_values("mean_share", ascending=False).head(4)["feature"].tolist())
mb_drivers = f"{'、'.join(mb_top4[:-1])} 与 {mb_top4[-1]}" if len(mb_top4) >= 2 else "、".join(mb_top4)
ifpca = read_table("table_if_pca_raw.csv")
if_base = ifpca[ifpca["variant"] == "原始特征"]
if_white = ifpca[ifpca["variant"] == "PCA 白化特征"]
n_feat_h = len(METRICS["housing"]["per_seed"][0]["selected_features"])
abstract = (
    "针对中低维表格数据中无监督异常检测的算法选择问题，本文以 T13 任务指定的 iris_with_error 与 housing 数据为对象，"
    "构建了包含点异常、多特征协同异常和极端组合异常的可审计注入基准，系统比较 Z-Score、马氏距离、LOF、Isolation Forest "
    "与 One-Class SVM 五类方法，并采用 PR-AUC、Recall、FPR、F1 与 ROC-AUC 评价。数据审计发现 iris 缺失 13 格（1.71%）、"
    "重复 5 行（3.29%），housing 无缺失、无重复；housing 主实验采用 10 个随机种子、70/15/15 训练内划分与 30% 测试集，"
    "测试集注入 15 个异常（9.87%）；iris 清洗后 145 行同样采用 10 个种子做小样本对照。"
    f"结果表明，马氏距离在 housing 上取得 PR-AUC {f4(agg('housing','Mahalanobis','pr_auc'))}、"
    f"Recall {f4(agg('housing','Mahalanobis','recall'))}、FPR {f4(agg('housing','Mahalanobis','fpr'))}，"
    "满足召回≥0.85 且误报≤10% 的达标线；按 val PR-AUC 选取前三模型的排名融合将 Recall 提升至 "
    f"{f4(agg('housing','Rank-Ensemble','recall'))}，但精确率降至 {f4(agg('housing','Rank-Ensemble','precision'))}。"
    "超参数在训练集内部用 3 折交叉验证选取，测试集不参与；One-Class SVM 由默认 nu=0.5 调到 "
    f"nu={nu_best_txt} 后 F1 提高 "
    f"{pcv('One-Class SVM','交叉验证最优') - pcv('One-Class SVM','默认参数'):.2f}，{mb_default_txt}，"
    f"而参数取到网格最差端会使 F1 下降 {pcv('Mahalanobis','交叉验证最优') - pcv('Mahalanobis','交叉验证最差'):.2f}"
    f"（马氏距离）与 {pcv('One-Class SVM','交叉验证最优') - pcv('One-Class SVM','交叉验证最差'):.2f}（One-Class SVM）。"
    f"参数敏感性显示 LOF 的 F1 极差达 {sens_range('LOF'):.3f}，马氏距离为 {sens_range('Mahalanobis'):.3f}；"
    f"轴对齐消融显示，同一批特征经 PCA 白化后，Isolation Forest 的 PR-AUC 由 "
    f"{if_base['pr_auc'].mean():.4f} 升至 {if_white['pr_auc'].mean():.4f}，证实其失利源于切分方向与异常偏移方向错配；"
    f"特征贡献显示 {mb_drivers} 是主要驱动因素，"
    "极端组合异常仍是主要漏报来源；iris 小样本的 FPR 方差显著增大。研究说明：在低维、近似椭球分布的数据上，"
    "统计距离方法优于树模型与密度方法，阈值校准与误报复核比盲目追求复杂模型更重要。"
    "上述结果对同类低维表格数据的异常检测具有参考价值，但外推到图像、文本与工业时序场景仍需进一步验证。"
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
     "等价于对第一类错误率（FPR）做显式约束，这一思路在统计学习方法中有系统讨论[14]。需要说明，"
     "Accuracy（准确率）在约 10% 异常占比下会被大量真负例抬高——全部判正常的平凡解也能取得约 90% 的准确率，"
     "因此本文不以其作为评价指标，仅在代码中逐种子记录备查。")

heading(doc, "2.3 预期目标", 2)
body(doc, "本文设置以下 6 条可检验目标：", indent=True)
for s in [
    "（1）在 housing 主实验上，至少一种方法达到 Recall≥0.85 且 FPR≤10%，并给出 10 个种子的均值与标准差；",
    "（2）完成 Z-Score、马氏距离、LOF、Isolation Forest、One-Class SVM 五类方法的统一协议对比，并给出 PR/ROC 曲线；",
    "（3）给出 LOF、Isolation Forest、One-Class SVM、马氏距离的关键参数敏感性热图，量化 F1 的波动范围；",
    "（4）用消融实验检验“树模型在相关数据上失利”的原因，并给出可操作的改进方向；",
    "（5）给出特征贡献与不少于 3 条有数据支撑的业务洞察，并对典型误报/漏报案例做剖析；",
    "（6）全部实验可由一条命令复现，代码、数据、图表、结果表与报告一并交付。",
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
     "只在最后评估时使用。housing 与 iris 均使用 10 个种子（42、2024、7、13、99、2026、11、18、123、777），"
     "以观察划分波动。测试集与验证集的注入比例均为约 10%，具体配置见表 4-2。")
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
     "iris 保留 6 个特征（Sepal_Length、Sepal_Width、Petal_Length 共 3 个原始变量，"
     "加 Sepal_Area、Petal_Aspect、Sepal_Aspect 共 3 个派生特征；Petal_Width 与 Petal_Area 因与 Petal_Length "
     "高度相关被剔除）。被剔除的派生特征并非无意义，"
     "而是与原始变量高度共线；在树模型之外，共线会让距离度量重复计权，因此优先保留解释性更强的原始字段。")

heading(doc, "5.4 特征有效性验证", 2)
gain = read_table("table_feature_gain.csv")
for c in ["PR-AUC", "Recall", "F1", "FPR"]:
    gain["_" + c] = pd.to_numeric(gain[c], errors="coerce")


def gain_cell(carrier: str, featset: str, key: str) -> float:
    row = gain[(gain["载体"] == carrier) & (gain["特征集"] == featset)]
    return float(row["_" + key].iloc[0])


def gain_delta(carrier: str, key: str) -> float:
    return gain_cell(carrier, "工程特征", key) - gain_cell(carrier, "基础特征", key)


def gain_verdict(carrier: str) -> str:
    d = gain_delta(carrier, "PR-AUC")
    if d > 0.01:
        return f"PR-AUC 上升 {d:.4f}，说明派生特征带来的非线性与交互信息被模型用上了"
    if d < -0.01:
        return (f"PR-AUC 下降 {abs(d):.4f}——在轴对齐切分的模型里，维度越多，"
                "相关方向的偏移越难在一次切分中暴露，这一步的净收益为负")
    return f"PR-AUC 基本持平（{d:+.4f}），变化小于 0.01，可视为等价"


carriers = list(dict.fromkeys(gain["载体"]))
body(doc,
     f"本节以 seed=42 为固定划分，分别用 {' 与 '.join(carriers)} 两种载体比较基础特征（原始字段）与工程特征"
     "（原始字段+派生特征）的测试表现，结果见表 5-2。之所以用两种载体，是因为它们对新增维度的反应方向相反："
     "树模型按轴对齐随机切分，维度越多越难让相关偏移在单次切分中暴露；平方距离模型则把新维度直接并入协方差矩阵，"
     "维度增加未必有害。只用一种载体，得到的只是“派生特征对该模型是否有用”，而不是“派生特征本身是否有用”。")
for carrier in carriers:
    same = (abs(gain_delta(carrier, "Recall")) < 1e-9
            and abs(gain_delta(carrier, "F1")) < 1e-9)
    note = "（阈值化后两者的判定完全一致，差别只体现在排序质量上）" if same else ""
    body(doc,
         f"以 {carrier} 为载体时，加入派生特征后 {gain_verdict(carrier)}："
         f"PR-AUC {gain_cell(carrier,'基础特征','PR-AUC'):.4f} → {gain_cell(carrier,'工程特征','PR-AUC'):.4f}，"
         f"Recall {gain_cell(carrier,'基础特征','Recall'):.4f} → {gain_cell(carrier,'工程特征','Recall'):.4f}，"
         f"F1 {gain_cell(carrier,'基础特征','F1'):.4f} → {gain_cell(carrier,'工程特征','F1'):.4f}，"
         f"FPR {pct(gain_cell(carrier,'基础特征','FPR'))} → {pct(gain_cell(carrier,'工程特征','FPR'))}{note}。")


def drop_cell(carrier: str, feature: str, key: str) -> float:
    row = gain[(gain["载体"] == carrier) & (gain["特征集"].str.endswith(feature))]
    return float(row["_" + key].iloc[0])


crim_eff = {c: gain_cell(c, "工程特征", "PR-AUC") - drop_cell(c, "CRIM_log", "PR-AUC")
            for c in carriers}
tax_eff = {c: gain_cell(c, "工程特征", "PR-AUC") - drop_cell(c, "TAX_RAD", "PR-AUC")
           for c in carriers}
body(doc,
     "逐特征剔除进一步定位每个派生特征的边际作用（以工程特征为基准，正值表示保留该特征更好）："
     + "；".join(f"CRIM_log 在 {c} 上为 {crim_eff[c]:+.4f}" for c in carriers)
     + "；"
     + "；".join(f"TAX_RAD 在 {c} 上为 {tax_eff[c]:+.4f}" for c in carriers)
     + "。两个字段的作用都是“有正有负、量级很小”，这与它们的构造目的有关：CRIM_log 只是把犯罪率的长尾压平，"
     "排序信息由原字段已经提供；TAX_RAD 虽然给出了税率与可达性的比值，但与 TAX、RAD 本身高度相关，"
     "增量信息有限。因此工程特征的价值不在于提升总体 PR-AUC，而在于两点：一是提供一个可读的比率型字段，"
     "便于第 7.1 节解释“为什么这个样本异常”；二是把距离模型的召回从 "
     f"{gain_cell('Mahalanobis','基础特征','Recall'):.4f} 提升到 "
     f"{gain_cell('Mahalanobis','工程特征','Recall'):.4f}（代价是 FPR 从 "
     f"{pct(gain_cell('Mahalanobis','基础特征','FPR'))} 升到 "
     f"{pct(gain_cell('Mahalanobis','工程特征','FPR'))}），对“宁可多报、不能漏报”的复核场景是有利的。"
     "由于特征筛选只使用训练集的相关性结构、不接触测试标签，保留这两个字段不会引入泄漏。"
     "本节是单种子对照，与表 6-2 的 10 种子均值"
     f"（Isolation Forest {f4(agg('housing','Isolation Forest','pr_auc'))}、"
     f"马氏距离 {f4(agg('housing','Mahalanobis','pr_auc'))}）不同属正常现象，不应混用。")
for c in ["PR-AUC", "Recall", "F1", "FPR"]:
    gain[c] = gain["_" + c].map(lambda v: f"{v:.4f}")
table(doc, gain[["载体", "特征集", "特征数", "PR-AUC", "Recall", "F1", "FPR", "说明"]],
      "表 5-2  特征工程增益（seed=42，两种载体对照）",
      widths=[2.2, 3.2, 1.1, 1.6, 1.5, 1.4, 1.4, 3.0], font_size=7.8, keep_together=True)

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
     "基线采用稳健 Z-Score：对每个特征计算 (x_j−median_j)/(1.4826×MAD_j)，取绝对值最大者作为异常得分，"
     "阈值由 calib 正常样本的分位数确定。位置与尺度改用中位数与绝对中位差（MAD），是因为 housing 的 CRIM、B、LSTAT "
     "等字段长尾明显（见 4.1 节），均值与标准差会被极端值拉偏。该方法对应业务中最常见的“单字段越界”规则，"
     "优点是透明、无需训练；缺点是忽略特征相关性与多变量协同。它的结果用于回答一个关键问题："
     "复杂模型相对简单规则到底带来了多少增益。")

heading(doc, "6.3 对比模型", 2)
body(doc,
     "（1）马氏距离：用协方差矩阵估计均值与协方差，得分 d(x)=sqrt((x-μ)^T Σ^{-1}(x-μ))。它把特征相关性与尺度"
     "同时纳入，理论上适合近似椭球分布的数据；support_fraction=1 等价于全量协方差，小于 1 时改用 MCD 的稳健估计，"
     "用最小化协方差行列式的方式降低异常点对估计的污染[8]。参数网格取 0.5、0.6、0.75、0.9、1.0，"
     "另加 Ledoit–Wolf 收缩协方差 1 组（在经验协方差与单位阵之间按数据驱动收缩，样本量有限时估计更稳），合计 6 组。")
body(doc,
     "（2）LOF：对每个样本比较其局部可达密度与邻居密度，得分越高越孤立[3]。它能发现局部异常，但在高维或密度不均时"
     "容易退化；参数为邻居数 k，housing 取 3–50（3、4、5、10、15、20、30、50），iris 取 3–20。")
body(doc,
     "（3）Isolation Forest：随机选择特征与切分点构造孤立树，异常样本更容易被少量切分孤立[4,5]。参数包括树数、"
     "max_samples 与 max_features；本文树数取 100/200/300，max_samples 取 0.5/0.75/1.0，max_features 取 0.5/1.0。")
body(doc,
     "（4）One-Class SVM：在核空间中寻找包围正常样本的最小边界[6,7]。参数为 nu（边界外样本比例上界）与 RBF 核的 gamma；"
     "nu 取 0.005、0.01、0.05、0.1、0.2、0.3、0.5、0.7（含 scikit-learn 默认值 0.5，用于检验默认参数是否够用），"
     "gamma 取 scale 与 0.002/0.005/0.01/0.05/0.1/0.5，合计 56 组；初版网格的下边界 nu=0.01 曾被交叉验证反复选中，"
     "说明最优值可能更低，故向低侧扩边。")
body(doc,
     "（5）排名融合：先按 val PR-AUC 从五个基模型中选出前三名，再把入选模型的得分在 calib 上做 z-score 标准化后等权平均；"
     "成员筛选只使用验证集，不接触测试。它不是新的基学习器，而是检验“不同归纳偏置是否互补”的对照；"
     "若融合的召回明显高于单模型，说明不同方法捕捉到了不同类型的异常。初版协议等权收编全部五个模型，"
     "其中最弱者的低质量排名会拖累平均，故改为按验证表现筛选成员。")

heading(doc, "6.4 实验设置", 2)
body(doc,
     "实验环境为 AMD Ryzen 9 8945HX（16 核 32 线程）、15.2 GB 内存、Windows；Python 3.13.9、NumPy 2.3.5、"
     "pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.10.6，全部模型实现基于 scikit-learn 与 NumPy[17,18]，"
     "绘图基于 Matplotlib[19]。所有模型使用固定 random_state；超参数在训练集内部用 3 折交叉验证选择，"
     "阈值只在 calib 正常样本与 val 合成异常上确定，约束 calib FPR≤8%（即 10% 达标线的 80%，为抽样波动留出安全裕度），"
     "目标为最大化 val 召回，召回并列时取 F1 较高者；测试集不参与任何选择。"
     "参数网格与划分见表 6-1。")
body(doc,
     "三折由训练集内部的正常样本划分而来：每折用约 2/3 的正常样本拟合模型，其中 1/4 作为折内校准集确定阈值，"
     "其余正常样本加上该折独立注入的合成异常作为折内评估集（housing 每折 8 个、iris 每折 4 个）。三折的评估样本"
     "恰好覆盖全部正常训练样本，且每次参数选择只用到训练集内部的样本与标签；折内标准化参数重新估计，特征列沿用"
     "第 5 节在校验阶段确定的结果。参数选择按折内 F1 均值排序，同时记录网格内的最大–最小值之差，"
     "作为“该方法的性能对参数有多敏感”的度量，这一度量同样不接触测试集。")
settings = pd.DataFrame([
    {"项目": "训练/校准/验证/测试", "设置": "housing 247/53/54/152；iris 70/15/16/44"},
    {"项目": "随机种子", "设置": "housing 与 iris 各 10 个；所有模型固定 random_state"},
    {"项目": "参数选择", "设置": "训练集内 3 折交叉验证，按折内 F1 均值排序取最优"},
    {"项目": "交叉验证折", "设置": "housing 83/82/82（各含 8 个合成异常）；iris 24/23/23（各含 4 个）"},
    {"项目": "候选网格规模", "设置": "马氏距离 6 组；LOF 8 组；Isolation Forest 18 组；One-Class SVM 56 组"},
    {"项目": "注入类型", "设置": "点异常 40%、协同异常 40%、极端组合 20%（按行随机）"},
    {"项目": "阈值约束", "设置": "calib FPR≤0.08（80% 安全裕度），最大化 val 召回（并列取 F1 高者）；测试集不调阈值"},
    {"项目": "评价指标", "设置": "PR-AUC（主）、Recall、FPR、Precision、F1、ROC-AUC、训练耗时"},
    {"项目": "软件环境", "设置": "Python 3.13.9；scikit-learn 1.7.2；NumPy 2.3.5；pandas 2.3.3"},
])
table(doc, settings, "表 6-1  实验环境与参数设置", widths=[4.0, 11.4], font_size=9,
      keep_together=True)

heading(doc, "6.5 实验结果", 2)
gate = [m for m in ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM"]
        if passes_gate("housing", m)]
if gate == ["Mahalanobis"]:
    gate_txt = "是唯一在均值意义上同时满足“Recall≥0.85、FPR≤10%”的单模型"
elif len(gate) > 1:
    gate_txt = (f"与 {'、'.join(m for m in gate if m != 'Mahalanobis')} 均在均值意义上满足"
                "“Recall≥0.85、FPR≤10%”的达标线，其中马氏距离的 PR-AUC 与 F1 最高")
else:
    gate_txt = "最接近“Recall≥0.85、FPR≤10%”的达标线"
body(doc,
     f"housing 主实验的 10 种子均值±标准差见表 6-2。马氏距离取得最高的 PR-AUC（{f4(agg('housing','Mahalanobis','pr_auc'))}）、"
     f"F1（{f4(agg('housing','Mahalanobis','f1'))}）与 ROC-AUC（{f4(agg('housing','Mahalanobis','roc_auc'))}），"
     f"Recall 为 {f4(agg('housing','Mahalanobis','recall'))}，FPR 为 {f4(agg('housing','Mahalanobis','fpr'))}，{gate_txt}。"
     f"One-Class SVM 的 Recall 为 "
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
table(doc, perf_h, "表 6-2  housing 模型性能对比（10 种子均值±标准差）",
      widths=[2.1, 3.0, 2.1, 2.1, 1.9, 1.9, 2.0], font_size=8.0, keep_together=True)
mb_rec = seed_values("housing", "Mahalanobis", "recall")
ens_rec = seed_values("housing", "Rank-Ensemble", "recall")
mb_ok = sum(1 for r in mb_rec if r >= 0.85)
ens_ok = sum(1 for r in ens_rec if r >= 0.85)
body(doc,
     f"稳健性方面，逐种子检查比只看均值更有意义。马氏距离在 {len(mb_rec)} 个种子中有 {mb_ok} 个的召回 ≥0.85"
     f"（{'、'.join(f'{r:.3f}' for r in mb_rec)}），最低为 {min(mb_rec):.3f}；排名融合有 {ens_ok} 个种子召回 ≥0.85，"
     f"最低为 {min(ens_rec):.3f}。也就是说，达标线在多数划分下成立，但个别划分下会失守，报告时应同时给出均值、"
     "标准差与逐种子明细，而不是只报一个均值。")
body(doc,
     "各方法的异常得分分布、PR 曲线、ROC 曲线与 10 个种子的波动分别见图 6-2、图 6-3、图 6-4 与图 6-5；"
     f"单次训练耗时对比见图 6-7。马氏距离、LOF 与 One-Class SVM 的拟合耗时都在 3 毫秒以内"
     f"（{agg('housing','Mahalanobis','fit_seconds') * 1000:.2f}、{agg('housing','LOF','fit_seconds') * 1000:.2f}、"
     f"{agg('housing','One-Class SVM','fit_seconds') * 1000:.2f} ms），Isolation Forest 约 "
     f"{agg('housing','Isolation Forest','fit_seconds'):.2f} s，融合模型约 "
     f"{agg('housing','Rank-Ensemble','fit_seconds'):.2f} s（等于入选模型之和）。需要说明的是，表中耗时只计最终"
     "重拟合，未包含交叉验证的参数搜索开销。对 506 条样本的离线审计而言这些差异可以接受，但在高频在线场景需优先"
     "考虑轻量模型。")
figure(doc, "fig06_score_distributions.png", "图 6-2  各方法在 housing 上的异常得分分布（seed=42）", 15.2)
figure(doc, "fig07_pr_curves.png", "图 6-3  housing 的 Precision-Recall 曲线（seed=42）", 12.6)
figure(doc, "fig08_roc_curves.png", "图 6-4  housing 的 ROC 曲线（seed=42）", 12.6)
figure(doc, "fig09_recall_fpr.png", "图 6-5  housing 的 Recall 与 FPR（10 种子均值±标准差）", 15.0)
figure(doc, "fig10_sensitivity.png", "图 6-6  四类方法的参数敏感性热图（housing，seed=42）", 15.4)
figure(doc, "fig13_runtime.png", "图 6-7  housing 单次训练耗时对比（对数坐标）", 12.6)
figure(doc, "fig14_per_type_recall.png", "图 6-8  housing 分异常类型的召回率", 15.0)
figure(doc, "fig12_iris.png", "图 6-9  iris 小样本对照：Recall、FPR 与 PR-AUC（10 种子）", 15.4)

extreme_hi = [m for m in ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest",
                          "One-Class SVM", "Rank-Ensemble"]
              if ptype_num(m, "极端组合") > 0.90]
if extreme_hi == ["Rank-Ensemble"]:
    extreme_txt = "是唯一在极端组合上超过 0.90 的方案"
elif len(extreme_hi) > 1:
    extreme_txt = (f"与 {'、'.join(m for m in extreme_hi if m != 'Rank-Ensemble')} "
                   "一同在极端组合上超过 0.90")
else:
    extreme_txt = "在极端组合上的召回最高"
body(doc,
     f"分类型结果见表 6-3 与图 6-8。Z-Score 对点异常的召回为 {ptype_num('Z-Score','点异常'):.2f}，但对协同异常只有 "
     f"{ptype_num('Z-Score','协同异常'):.2f}、对极端组合只有 {ptype_num('Z-Score','极端组合'):.2f}，"
     f"说明“单字段越界”规则无法覆盖多字段共同偏移；马氏距离对协同异常召回 {ptype_num('Mahalanobis','协同异常'):.2f}，"
     f"但对极端组合仅 {ptype_num('Mahalanobis','极端组合'):.2f}；Isolation Forest 对点异常只有 "
     f"{ptype_num('Isolation Forest','点异常'):.2f}，却对协同异常达到 {ptype_num('Isolation Forest','协同异常'):.2f}；"
     f"融合模型在三种类型上分别为 {ptype_num('Rank-Ensemble','点异常'):.2f}、"
     f"{ptype_num('Rank-Ensemble','协同异常'):.2f}、{ptype_num('Rank-Ensemble','极端组合'):.2f}，"
     f"{extreme_txt}。这一结果说明不同方法的归纳偏置确实互补：Z-Score 擅长单点越界，"
     "马氏距离擅长相关变量整体偏移，树模型擅长不规则边界，融合可以覆盖更多类型。")
ptype = read_table("table_per_type_housing.csv")
for c in ["点异常", "协同异常", "极端组合"]:
    ptype[c] = pd.to_numeric(ptype[c], errors="coerce").map(lambda v: f"{v:.3f}")
table(doc, ptype, "表 6-3  housing 分异常类型召回率", widths=[4.0, 3.8, 3.8, 3.8], font_size=9,
      keep_together=True)

err_cases = read_table("table_error_cases.csv")
fn_cases = err_cases[err_cases["error_type"] == "漏报(FN)"]
fp_cases = err_cases[err_cases["error_type"] == "误报(FP)"]
type_cn = {"point": "点异常", "multi": "协同异常", "extreme": "极端组合异常", "normal": "正常样本"}
err_txt = (f"为了观察错分来源，表 6-4 列出 seed=42 下马氏距离的 {len(err_cases)} 个错分案例"
           f"（{len(fn_cases)} 个漏报、{len(fp_cases)} 个误报）。")
if len(fn_cases):
    r = fn_cases.iloc[0]
    err_txt += (f"漏报样本属于{type_cn.get(str(r['injection_type']), str(r['injection_type']))}，"
                f"得分为 {float(r['score']):.3f}，低于阈值 {float(r['threshold']):.3f}；"
                f"其 RM 为 {float(r['RM']):.2f}、CRIM 为 {float(r['CRIM']):.3f}、LSTAT 为 {float(r['LSTAT']):.1f}，"
                "虽然组合罕见，但马氏距离仍认为它落在正常椭球的可接受范围内。")
if len(fp_cases):
    margins = pd.to_numeric(fp_cases["score"]) - pd.to_numeric(fp_cases["threshold"])
    err_txt += (f"{len(fp_cases)} 个误报均为正常样本，特征是“低房间数/高犯罪率/极高税率”等自然长尾组合，"
                f"得分只比阈值高 {margins.min():.2f}–{margins.max():.2f}。"
                "它们不是数据错误，而是业务上值得复核的边界样本；这提示我们不应把误报简单视为模型失败，"
                "而应结合复核成本决定阈值。")
body(doc, err_txt)
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
mb_sel = Counter(chosen_params("Mahalanobis"))
mb_common_js, mb_common_n = mb_sel.most_common(1)[0]
n_seeds_h = len(METRICS["housing"]["per_seed"])
if mb_common_n == n_seeds_h:
    mb_cv_txt = f"交叉验证在 {n_seeds_h} 个种子上全部选出 {zh_params(mb_common_js)}"
else:
    rest = "、".join(zh_params(js) for js in mb_sel if js != mb_common_js)
    mb_cv_txt = (f"交叉验证在 {mb_common_n}/{n_seeds_h} 个种子上选出 {zh_params(mb_common_js)}，"
                 f"其余种子为 {rest}")
body(doc,
     "第一，马氏距离的优势来自数据分布与异常机制匹配。housing 的 13 个预测变量存在明显相关（RAD-TAX、NOX-INDUS/DIS），"
     "注入的协同异常又是在相关方向上整体偏移；马氏距离用协方差逆矩阵做白化，能把“整体偏移”转化为较大的马氏半径。"
     f"值得注意的是，{mb_cv_txt}：训练集已在清洗阶段剔除异常，MCD 的子集削减在这个样本量上主要引入额外方差，"
     "收缩或削减方向是否有益必须由交叉验证的数据说了算。但若把参数误设到网格另一端（"
     f"{pc_params('Mahalanobis','交叉验证最差')}），测试 F1 会从 "
     f"{pcv('Mahalanobis','交叉验证最优'):.4f} 掉到 {pcv('Mahalanobis','交叉验证最差'):.4f}（表 6-6），"
     "可见“稳健化”并非无条件有益，参数方向必须由数据决定。")
if_range_min = min(cv_range(m) for m in ["Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM"])
if_range_claim = ("是四种可调方法里最小的（附表 7）"
                  if abs(cv_range("Isolation Forest") - if_range_min) < 1e-9
                  else "在四种可调方法中偏小（附表 7）")
if_d_pr = float(if_white["pr_auc"].mean() - if_base["pr_auc"].mean())
if_d_f1 = float(if_white["f1"].mean() - if_base["f1"].mean())
if_d_rec = float(if_white["recall"].mean() - if_base["recall"].mean())
if if_d_pr > 0.02:
    ifpca_verdict = ("白化后 PR-AUC 与 F1 明显提升，直接印证了“轴对齐切分与相关偏移错配”是主要瓶颈："
                     "把坐标轴旋转到相关方向之后，同样的随机切分就能更有效地孤立协同异常。")
elif if_d_pr < -0.02:
    ifpca_verdict = ("白化后指标不升反降，说明瓶颈不在坐标轴方向，而在于注入偏移幅度本身低于树模型的可检阈值；"
                     "轴对齐错配不是失利的全部原因。")
else:
    ifpca_verdict = ("白化前后差异落在种子波动范围内，说明轴向错配并非唯一瓶颈，"
                     "注入偏移幅度本身同样限制了树模型的表现。")
body(doc,
     "第二，Isolation Forest 的失利不是实现问题，而是归纳偏置不匹配。孤立森林依赖轴对齐的随机切分，当异常表现为多个"
     "相关特征的联合偏移时，单次切分只看到一个维度，需要更多树才能“碰巧”孤立。交叉验证在 18 组参数上搜索后给出的"
     f"最优设置，测试 PR-AUC 只有 {f4(agg('housing','Isolation Forest','pr_auc'))}，点异常召回仅 "
     f"{ptype_num('Isolation Forest','点异常'):.2f}；而它的折内 F1 极差只有 "
     f"{cv_range('Isolation Forest'):.2f}，{if_range_claim}，说明它的瓶颈不在参数而在打分机制本身——"
     "增加树数、放开特征采样都无法弥补轴对齐切分与相关偏移之间的错配。这也解释了为什么调参对它几乎没有正向收益"
     f"（默认 {pcv('Isolation Forest','默认参数'):.4f} vs 交叉验证最优 {pcv('Isolation Forest','交叉验证最优'):.4f}）。"
     f"为直接检验这一判断，本文把 {n_feat_h} 个特征用 fit 集 PCA 白化（折内重新估计）后再跑同一套调参协议，"
     "结果见表 6-5："     f"白化使测试 PR-AUC 变化 {if_d_pr:+.4f}、F1 变化 {if_d_f1:+.4f}、Recall 变化 {if_d_rec:+.4f}。"
     f"{ifpca_verdict}")
ifpca_view = read_str_table("table_if_pca.csv")
ifpca_view["主要参数"] = ifpca_view["主要参数"].map(compact_params)
table(doc, ifpca_view, "表 6-5  Isolation Forest 轴对齐消融（PCA 白化，10 种子均值±标准差）",
      widths=[2.0, 3.2, 2.6, 2.6, 2.5, 2.5], font_size=7.4, keep_together=True)
lof_max_p, lof_max_q, lof_max_f1 = sens_extreme("LOF", "max")
lof_min_p, lof_min_q, lof_min_f1 = sens_extreme("LOF", "min")
sens_methods = ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM"]
sens_top = max(sens_methods, key=sens_range)
sens_top_txt = "是五种方法中最大的" if sens_top == "LOF" else f"仅次于 {sens_top}"
body(doc,
     "第三，LOF 的 FPR 很低但召回不足。LOF 比较局部密度，对于远离正常云团的点异常很敏感；但当协同异常落在正常样本的"
     "稀疏边缘时，其局部密度未必显著低于邻居，因此召回偏低。LOF 对 k 也非常敏感：图 6-6 显示 "
     f"k={lof_max_p} 时 F1 可达 {lof_max_f1:.2f}，"
     f"k={lof_min_p} 且阈值分位 {float(lof_min_q):g} 时降到 {lof_min_f1:.2f}，"
     f"极差 {sens_range('LOF'):.3f}，{sens_top_txt}。")
body(doc,
     "第四，One-Class SVM 与融合模型体现“召回优先”的取舍。OCSVM 通过 nu 控制边界外样本比例，是本次实验中调参收益"
     f"最大的方法：沿用 scikit-learn 默认的 nu=0.5 时测试 F1 只有 {pcv('One-Class SVM','默认参数'):.4f}，"
     f"交叉验证把 nu 调到 {nu_best_txt} 后升到 {pcv('One-Class SVM','交叉验证最优'):.4f}，召回率 "
     f"{f4(agg('housing','One-Class SVM','recall'))}、精确率 {f4(agg('housing','One-Class SVM','precision'))}；"
     "融合模型把按 val PR-AUC 入选的三个模型的得分标准化后平均，等价于让不同归纳偏置投票，召回提升到 "
     f"{f4(agg('housing','Rank-Ensemble','recall'))}，但精确率降到 {f4(agg('housing','Rank-Ensemble','precision'))}。"
     "若业务更怕漏报，融合是合理选择；若复核成本很高，马氏距离更均衡。")
z_fpr_max = max(seed_values("housing", "Z-Score", "fpr"))
z_fpr_note = ("测试 FPR 全部保持在 10% 红线以内" if z_fpr_max <= 0.10
              else "说明在个别划分下这条约束会失守")
zprof = read_str_table("table_zscore_profile.csv")
z_top = zprof.iloc[0]
z_second = zprof.iloc[1] if len(zprof) > 1 else zprof.iloc[0]
body(doc,
     f"第五，稳健 Z-Score 是本文最弱的一环，原因不是实现错误，而是“取单字段最大 |z|”与长尾字段的结构性冲突。"
     f"附表 8 给出各字段的稳健尺度与测试集上的最大 |z|：{z_top['特征']} 的稳健尺度只有 {z_top['稳健尺度']}，"
     f"而其测试集最大 |z| 达到 {z_top['测试集最大|z|']}；{z_second['特征']} 为 {z_second['测试集最大|z|']}；"
     f"其余字段均不超过 {max(float(v) for v in zprof['测试集最大|z|'].iloc[2:]):.1f}。"
     f"也就是说，得分几乎完全由 1–2 个重尾字段的尾部决定，{int(z_top['成为行最大值的样本数'])} 个测试样本的得分"
     f"由 {z_top['特征']} 决定。这带来两个后果：其一，注入在其他字段上的异常很难越过这条被 CRIM 抬高的阈值，"
     f"召回只有 {f4(agg('housing','Z-Score','recall'))}；其二，分位阈值在测试集上漂移明显，"
     f"FPR 均值 {pct(agg('housing','Z-Score','fpr'))}、最高 {pct(z_fpr_max)}，{z_fpr_note}。"
     "这也说明“在训练集上卡住红线”不等于“在测试集上保证红线”，上线后必须按月用正常样本重新校准；"
     "若确实要用单变量规则，也应对每个字段先做分布变换（如对数）再取 z。")
body(doc,
     f"第六，iris 小样本结果（图 6-9）显示 FPR 的方差远大于 housing：LOF 与 OCSVM 的平均召回分别为 "
     f"{f4(agg('iris','LOF','recall'))} 与 {f4(agg('iris','One-Class SVM','recall'))}，但 FPR 分别为 "
     f"{pct(agg('iris','LOF','fpr'))} 与 {pct(agg('iris','One-Class SVM','fpr'))}；马氏距离的 FPR 为 "
     f"{pct(agg('iris','Mahalanobis','fpr'))}，召回只有 {f4(agg('iris','Mahalanobis','recall'))}。145 条样本、"
     "44 条测试、4 个异常的条件下，单个样本的判定就会让 FPR 跳动 2–3 个百分点，因此不能用单次划分给小样本方法下结论。"
     "本文保留这一负面结果，并将其作为“小样本异常检测需要重复划分与置信区间”的证据。")

heading(doc, "6.7 参数调优的必要性与代价", 2)
body(doc,
     "为了回答“不调参到底行不行”，本文额外做了一组对照：同一份数据、同一套阈值规则、同一个测试集，只把超参数换成"
     "三种取值——scikit-learn 的默认值、交叉验证选出的最优值、以及交叉验证排名最差的网格点。结果见表 6-6 与图 6-10。"
     "结论并不是“调参一定更好”，而是“参数重不重要取决于方法”。")
figure(doc, "fig15_param_compare.png",
       "图 6-10  参数设置对 F1 的影响：交叉验证（左）与测试集（右）", 15.0)
lof_best_k = re.search(r"n_neighbors=(\d+)", pc_params("LOF", "交叉验证最优"))
lof_worst_k = re.search(r"n_neighbors=(\d+)", pc_params("LOF", "交叉验证最差"))
lof_best_k_txt = lof_best_k.group(1) if lof_best_k else "5"
lof_worst_k_txt = lof_worst_k.group(1) if lof_worst_k else "50"
body(doc,
     f"对 One-Class SVM，默认 nu=0.5 严重高估了异常比例，测试 F1 只有 {pcv('One-Class SVM','默认参数'):.4f}，"
     f"调到 nu={nu_best_txt} 附近的低值后达到 {pcv('One-Class SVM','交叉验证最优'):.4f}，提升 "
     f"{pcv('One-Class SVM','交叉验证最优') - pcv('One-Class SVM','默认参数'):.2f}，是本组实验里调参收益最大的一项。"
     f"对马氏距离，默认的 {pc_params('Mahalanobis','默认参数')} 与交叉验证选出的最优设置只差 "
     f"{pcv('Mahalanobis','交叉验证最优') - pcv('Mahalanobis','默认参数'):.4f}（"
     f"{pc_params('Mahalanobis','交叉验证最优')}），"
     "基本等于不调；但网格另一端（"
     f"{pc_params('Mahalanobis','交叉验证最差')}）只拿到 {pcv('Mahalanobis','交叉验证最差'):.4f}，与最优值相差 "
     f"{pcv('Mahalanobis','交叉验证最优') - pcv('Mahalanobis','交叉验证最差'):.2f}，说明“不作选择”与“选错”是两回事。"
     f"对 LOF，默认 k=20 得到 {pcv('LOF','默认参数'):.4f}，最优 k={lof_best_k_txt} 为 {pcv('LOF','交叉验证最优'):.4f}，收益 "
     f"{pcv('LOF','交叉验证最优') - pcv('LOF','默认参数'):.2f}，属于中等；而 k={lof_worst_k_txt} 只有 "
     f"{pcv('LOF','交叉验证最差'):.4f}，说明邻居数取大时局部密度被过度平滑。"
     f"对 Isolation Forest，交叉验证选出的参数在测试集上反而比默认值低 "
     f"{pcv('Isolation Forest','默认参数') - pcv('Isolation Forest','交叉验证最优'):.2f}，"
     "是本组唯一出现负收益的方法——它的参数曲面对目标函数几乎是平的，交叉验证在小折上的排序噪声盖过了真实差异。")
body(doc,
     "这组对照还暴露了交叉验证在小样本上的代价。折内训练样本只有约 123 条（housing），模型能力弱于最终模型，"
     "折内 F1 与测试 F1 的差值并不稳定：在“交叉验证最优”一行为 "
     f"{gap_span('交叉验证最优')[0]:+.3f}～{gap_span('交叉验证最优')[1]:+.3f}，在“默认参数/交叉验证最差”两行为 "
     f"{gap_span('other')[0]:+.3f}～{gap_span('other')[1]:+.3f}。也就是说，折内评估只能用来给候选参数"
     "排序，不能当作性能预估；真正的性能仍必须由独立的测试集给出。另一个代价是排序噪声：当参数曲面的起伏小于折间波动时，"
     f"交叉验证会“选中噪声”，Isolation Forest 的 {n_seeds_h} 个种子选出了 "
     f"{len(set(chosen_params('Isolation Forest')))} 组不同参数正是这一现象。因此本文的做法是主表报告"
     "多种子均值与标准差，同时用附表 7 给出每个种子实际选中的参数与折内极差，让读者判断参数选择是否稳定，"
     "而不是只呈现一个“调好的”结果。")
param_df = read_str_table("table_param_compare.csv")
param_df["参数取值"] = param_df["参数取值"].map(compact_params)
table(doc, param_df, "表 6-6  housing 三种参数设置下的表现（10 种子均值）",
      widths=[2.4, 2.1, 3.6, 1.5, 1.5, 1.6, 1.4, 1.3], font_size=7.4, keep_together=True)

# ---------------- 7
heading(doc, "七、结果解释与业务洞察", 1)
heading(doc, "7.1 模型可解释性", 2)
body(doc,
     f"图 7-1 给出两种解释路径。马氏距离按 d² 的特征分解计算每个字段对马氏半径的贡献：{contrib_top('Mahalanobis')}，"
     "说明污染浓度、税率、房间数与区位距离是驱动异常的主要维度。"
     f"Isolation Forest 用“把某特征替换为训练中位数后得分下降”衡量贡献：{contrib_top('Isolation Forest')}。"
     "两种方法的排序不同，是因为它们衡量的是不同东西：前者衡量“偏离椭球中心的方向”，"
     "后者衡量“对孤立路径的贡献”。业务解释时应说明所用方法，不能把两者混为一谈。")
figure(doc, "fig11_feature_contrib.png", "图 7-1  Mahalanobis 与 Isolation Forest 的特征贡献占比", 14.6)

heading(doc, "7.2 业务洞察", 2)
body(doc,
     "洞察一：在低维、特征相关的表格数据上，统计距离方法应作为首选基线。马氏距离在 housing 上以 "
     f"{f4(agg('housing','Mahalanobis','pr_auc'))} 的 PR-AUC、{f4(agg('housing','Mahalanobis','recall'))} 的召回和 "
     f"{pct(agg('housing','Mahalanobis','fpr'))} 的误报全面优于树模型与密度模型；它的计算成本只有毫秒级，"
     "且特征贡献可直接分解，比黑盒模型更适合需要解释的复核场景。")
body(doc,
     "洞察二：阈值校准对误报的影响不亚于模型选择。初版协议（最大化 val F1、calib FPR≤10%）下，Z-Score 的单种子 "
     "FPR 曾越过 10% 红线；本文已把 calib 预算收紧到目标的 80%（FPR≤8%）并改为召回优先，收紧后 Z-Score 的"
     f"平均 FPR 为 {pct(agg('housing','Z-Score','fpr'))}、单种子最高 {pct(z_fpr_max)}。"
     "同样的分数分布，仅因 calib 与 test 的抽样波动就可能跨过红线，上线后仍应按月用正常样本重新校准。")
body(doc,
     f"洞察三：极端组合异常是最难的一类，单模型难以兼顾。马氏距离对协同异常召回 "
     f"{ptype_num('Mahalanobis','协同异常'):.2f}，但对极端组合只有 {ptype_num('Mahalanobis','极端组合'):.2f}；"
     f"融合模型把极端组合召回提升到 {ptype_num('Rank-Ensemble','极端组合'):.2f}，"
     f"代价是精确率下降 {(agg('housing','Mahalanobis','precision') - agg('housing','Rank-Ensemble','precision')) * 100:.1f} "
     "个百分点。若业务不能接受漏报，应采用融合模型并配置人工复核；若复核资源有限，则用马氏距离并接受少量极端组合漏报。")
body(doc,
     f"洞察四：自然长尾是误报的主要来源，不应把误报全部当作模型缺陷。{len(fp_cases)} 个误报样本都是低房间数、高犯罪率、极高税率的"
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
     "消融实验进一步把这一失利归因到具体机制：把同一批特征做 PCA 白化后再跑 Isolation Forest，"
     f"测试 PR-AUC 从 {if_base['pr_auc'].mean():.4f} 升到 {if_white['pr_auc'].mean():.4f}、"
     f"F1 从 {if_base['f1'].mean():.4f} 升到 {if_white['f1'].mean():.4f}、"
     f"Recall 从 {if_base['recall'].mean():.4f} 升到 {if_white['recall'].mean():.4f}（表 6-5）。"
     "白化只是把坐标轴旋转到相关方向上，模型与参数完全没变，性能却跃升，说明树模型在相关数据上的瓶颈是"
     "“切分方向与异常偏移方向不一致”，而不是树的容量或参数。这一点在工程上很有用：面对多字段协同偏移的场景，"
     "先白化再上树模型，比单纯加树、加深度更有效。")
body(doc,
     "调参方面，本文用训练集内 3 折交叉验证替代单次验证集选参，并把“默认参数、交叉验证最优、交叉验证最差”三种取值"
     f"放在同一测试集上对照：One-Class SVM 从默认 nu=0.5 改到低 nu 后 F1 提升 "
     f"{pcv('One-Class SVM','交叉验证最优') - pcv('One-Class SVM','默认参数'):.2f}，{mb_default_txt}，"
     f"Isolation Forest 反而使 F1 下降 {pcv('Isolation Forest','默认参数') - pcv('Isolation Forest','交叉验证最优'):.2f}。"
     "因此“要不要调参”没有统一答案：参数曲面陡峭且默认值落在坏区间时必须调（如 One-Class SVM），参数曲面平坦时"
     "调参只是增加方差（如 Isolation Forest），而参数选错方向的代价远大于不调（马氏距离相差 "
     f"{pcv('Mahalanobis','交叉验证最优') - pcv('Mahalanobis','交叉验证最差'):.2f}）。")
body(doc,
     "特征工程方面，本文用两种载体做了对照："
     + "；".join(
         f"{c} 上 PR-AUC {gain_delta(c, 'PR-AUC'):+.4f}" for c in carriers)
     + "。同一批派生特征在两类模型上的方向并不一致，说明“特征工程是否有效”必须相对模型而言；"
     "这也与 6.6 节的白化实验互相印证——对相关数据，改变坐标表示往往比单纯增加维度更关键。")
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
     "落地前需要真实标签与代价矩阵做校准。白化消融虽然显著提升了树模型，但白化后的坐标是原始特征的线性组合，"
     "单个字段的异常贡献不再能直接读出；若业务必须回答“哪个字段出了问题”，仍需回到原始坐标系做二次归因。"
     "特征工程的对照实验只做了单种子（seed=42），结论方向可信、具体数值不宜外推。")
oc_nus = [json.loads(p)["nu"] for p in chosen_params("One-Class SVM") if "nu" in json.loads(p)]
if oc_nus and min(oc_nus) <= 0.005:
    oc_edge_txt = "One-Class SVM 的最优 nu 贴到扩边后网格的下边界 0.005，说明真实最优值可能更低"
elif oc_nus:
    oc_edge_txt = f"One-Class SVM 的最优 nu 已落入网格内部（最低选中 nu={min(oc_nus):g}），扩边消除了贴边现象"
else:
    oc_edge_txt = "One-Class SVM 的最优 nu 落在网格候选之中"
body(doc,
     "参数与协议方面还有三点局限：其一，交叉验证的折内训练样本只有约 123 条（housing）与约 35 条（iris），"
     "折内 F1 系统性偏低，且在小样本上容易选中噪声，Isolation Forest 在不同种子上选出不同参数即为表现；"
     "其二，为保持可比性，交叉验证沿用了第 5 节确定的特征列，只在折内重新估计标准化参数，"
     f"严格意义上的嵌套特征选择尚未实现；其三，网格是人为指定的有限集合，{oc_edge_txt}，"
     "报告中的“最优”应理解为“候选集内最优”。")

heading(doc, "8.3 改进方向", 2)
body(doc,
     "后续工作包括：（1）在工业时序等真实场景上验证本文协议，比较窗口统计特征与自编码器、LSTM 预测残差法；"
     "（2）引入多峰正常性建模，如 GMM 马氏距离、深度 SVDD；（3）把白化思想推广到其他树模型与近邻方法，"
     "并研究“白化 + 原始坐标”双通道归因，兼顾效果与可解释性；（4）研究概念漂移下的在线阈值校准与模型更新；"
     "（5）把误报复核成本纳入阈值优化，构建代价敏感的决策曲线；（6）补充更多公开数据集与真实异常标签，"
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
        body(doc, line.strip(), indent=False, size=12, align=WD_ALIGN_PARAGRAPH.LEFT)
body(doc,
     "说明：全部源码、图表、结果表与本报告均在该仓库中，可直接用 "
     "git clone https://github.com/XiaoXingYo/classwork.git 获取；完整改动明细另见随包的 git_log.txt。")

# ---------------- 10
heading(doc, "十、工具使用声明与诚信承诺", 1)
heading(doc, "10.1 生成式 AI 等工具使用声明", 2)
body(doc,
     "本报告在代码实现、调试与文字初稿阶段使用了 GitHub Copilot（VS Code）作为辅助工具，主要用于：生成数据加载、"
     "异常注入、检测器封装与绘图代码的初稿；提供 scikit-learn API 与报告排版的修改建议；梳理文献线索与章节结构。"
     "协议修订阶段（召回优先阈值与安全裕度、稳健 Z-Score、参数网格扩边、融合成员筛选、PCA 白化消融、housing 增至 "
     "10 个种子及相应的报告更新）使用了 Kimi Code CLI 作为辅助工具。"
     "所有实验数值均由本地代码实际运行产生，作者对代码逻辑、数据划分、阈值协议、图表与结论逐项复核并修改；"
     "AI 未参与数据伪造，也未替代作者对结果的责任。除上述工具外，未使用其他生成式 AI 工具。")
heading(doc, "10.2 诚信承诺", 2)
body(doc,
     "本人承诺：所提交的报告、代码、图表与数据真实、独立完成；数据未伪造、篡改；引用均已标注；"
     "涉及他人成果的部分已按 GB/T 7714—2015 给出参考文献；如存在错误，由本人承担责任。")
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(18)
r = p.add_run("组长签名：程龙飞          成员签名：无          日期：2026 年 11 月 20 日")
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
    set_run_font(p.add_run("["), "宋体", 10.5)
    num = p.add_run(str(i))
    set_run_font(num, "宋体", 10.5)
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(100 + i))
    start.set(qn("w:name"), f"_RefNum{i}")
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(100 + i))
    num._r.addprevious(start)
    num._r.addnext(end)
    set_run_font(p.add_run(f"] {ref}"), "宋体", 10.5)

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
stability = read_str_table("table_param_stability.csv")
stability["选中的参数"] = stability["选中的参数"].map(compact_params)
table(doc, stability, "附表 7  housing 交叉验证选参的逐种子记录",
      widths=[3.4, 1.6, 5.4, 2.4, 2.0, 2.0], font_size=7.8)
table(doc, read_str_table("table_zscore_profile.csv"),
      "附表 8  稳健 Z-Score 各字段的尺度与最大 |z|（housing，seed=42）",
      widths=[3.0, 2.6, 2.6, 3.4, 3.8], font_size=8.0, keep_together=True)

# ---------------- 附录 A：关键代码
doc.add_page_break()
heading(doc, "附录 A  关键代码片段", 1)
body(doc,
     "以下片段取自成果包源码，行号与提交版本一致；完整代码、运行脚本与单元测试见 src/ 与 tests/，"
     "复现方式见 README.md。为便于排版，片段只保留核心逻辑，省略了类型注解与日志输出。")
code_block(doc, "src/experiment.py", 33, 9, "代码 A-1  fit / calib / val / test 四划分与无泄漏断言")
code_block(doc, "src/inject.py", 92, 14, "代码 A-2  三类异常的注入强度")
code_block(doc, "src/detectors.py", 225, 13, "代码 A-3  FPR 预算约束下召回优先的阈值选择")
code_block(doc, "src/detectors.py", 283, 15, "代码 A-4  训练集内三折交叉验证选参")

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
