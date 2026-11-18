"""Regenerate the Git statistics block that is embedded in section 9.3.

Hashes are omitted on purpose: the last commit is amended after the report is
rebuilt, which would invalidate any hash printed inside the document.
"""
import subprocess
from pathlib import Path

from config import LOG_DIR, ROOT


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


count = int(git("rev-list", "--count", "HEAD"))
log = git("--no-pager", "log", "--pretty=format:%ad  %s", "--date=short")

text = "\n".join([
    f"Git 仓库：{ROOT}（本地仓库，未配置远程地址）",
    f"提交总数：{count} 次（含本报告所在提交）",
    "提交记录（日期  说明）：",
    log,
])
(LOG_DIR / "git_stats.txt").write_text(text + "\n", encoding="utf-8")
print(text)
