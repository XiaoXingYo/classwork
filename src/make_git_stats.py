"""Regenerate the Git statistics block that is embedded in section 9.3.

Hashes are omitted on purpose: the report is committed after it is rebuilt, so
any hash printed inside the document would be stale.

Pass ``--next "<commit message>"`` when the report being built is about to be
committed with that message; the block then already counts and lists that
commit, so the numbers stay exact after the commit is made (and the push stays
a fast-forward).
"""
import subprocess
import sys
from datetime import date

from config import LOG_DIR, ROOT

GITHUB_URL = "https://github.com/XiaoXingYo/classwork"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


next_msg = None
if "--next" in sys.argv:
    idx = sys.argv.index("--next")
    if idx + 1 < len(sys.argv):
        next_msg = sys.argv[idx + 1]

count = int(git("rev-list", "--count", "HEAD"))
log = git("--no-pager", "log", "--pretty=format:%ad  %s", "--date=short")
branch = git("rev-parse", "--abbrev-ref", "HEAD") or "main"
if next_msg:
    count += 1
    log = f"{date.today().isoformat()}  {next_msg}\n{log}"

text = "\n".join([
    f"Git 仓库：{GITHUB_URL}（GitHub 公开仓库，默认分支 {branch}）",
    "本地目录：T13_程龙飞_实训成果/（与远程仓库同一份提交历史）",
    f"提交总数：{count} 次（含本报告所在提交）",
    "提交记录（日期  说明）：",
    log,
])
(LOG_DIR / "git_stats.txt").write_text(text + "\n", encoding="utf-8")
print(text)
