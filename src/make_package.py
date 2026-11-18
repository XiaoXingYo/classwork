"""Package the deliverable folder as T13_<name>_实训成果.zip."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT
TOP = SRC_DIR.name if SRC_DIR.name.startswith("T13_") else SRC_DIR.name.replace("13_", "T13_", 1)
OUT = SRC_DIR.parent / f"{TOP}.zip"
EXCLUDE = {".git", "__pycache__", ".pytest_cache", "preview"}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    n = 0
    for p in SRC_DIR.rglob("*"):
        if p.is_dir() or any(part in EXCLUDE for part in p.parts) or p.name == "_preview.pdf":
            continue
        if p.name.startswith("~$"):
            continue
        rel = str(p.relative_to(SRC_DIR)).replace("\\", "/")
        z.write(p, arcname=f"{TOP}/{rel}")
        n += 1

print("zip:", OUT, "files:", n, "MB:", round(OUT.stat().st_size / 1024 / 1024, 2))
with zipfile.ZipFile(OUT) as z:
    print("testzip:", z.testzip())
    ns = z.namelist()
    print("top-level:", sorted({x.split("/")[0] for x in ns}))
    print("docx:", [x for x in ns if x.endswith(".docx")])
    print("figures:", sum(1 for x in ns if "/figures/" in x),
          "tables:", sum(1 for x in ns if "/tables/" in x))
