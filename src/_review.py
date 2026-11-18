"""One-off report audit (teacher-facing checklist)."""
import glob
import re

import docx

d = docx.Document(glob.glob("report/*.docx")[0])
paras = [p.text.strip() for p in d.paragraphs]
refs = [p for p in paras if re.match(r"^\[\d+\]", p)]
body = "\n".join(p for p in paras if not re.match(r"^\[\d+\]", p))

cited = set()
for group in re.findall(r"\[([\d,\-\s]+)\]", body):
    for part in group.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cited.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            cited.add(int(part))

caps = [p for p in paras if re.match(r"^(图|表|附表|代码) ?[0-9A]", p)]
figs = [c for c in caps if c.startswith("图 ")]
tbls = [c for c in caps if c.startswith("表 ")]
appx = [c for c in caps if c.startswith("附表")]

abstract = next(p for p in paras if len(p) > 200 and "异常检测" in p and "T13" in p)
kw = next(p for p in paras if p.startswith("【关键词】"))

print("页内段落:", len(paras), "表格:", len(d.tables))
print("图:", len(figs), "表:", len(tbls), "附表:", len(appx))
print("参考文献:", len(refs), "| 未被引用:", [i for i in range(1, len(refs) + 1) if i not in cited])
print("引用但无条目:", [i for i in sorted(cited) if i > len(refs)])
print("摘要汉字:", len(re.findall(r"[\u4e00-\u9fff]", abstract)))
print("关键词:", kw)
body_chars = len(re.findall(r"[\u4e00-\u9fff]", body))
print("全文汉字(不含文献):", body_chars)
for probe in ["SWAT", "SWaT", "待填", "尚未", "未开展", "高维表格"]:
    hit = [p[:40] for p in paras if probe in p]
    print(f"{probe}: {len(hit)}", hit[:2])
print("\n目录结构:")
for p in paras:
    if re.match(r"^(一|二|三|四|五|六|七|八|九|十)、|^附|^参考文献", p):
        print(" ", p)
