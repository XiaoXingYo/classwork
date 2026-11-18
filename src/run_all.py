"""One-command reproduction entry point."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, "-X", "utf8", script, *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SRC, check=True)


def main() -> None:
    run("data_loader.py")
    run("experiment.py")
    run("make_tables.py")
    run("explain.py")
    run("plots.py")
    run("report_builder.py")
    print("\nAll steps finished. See outputs/ and report/.")


if __name__ == "__main__":
    main()
