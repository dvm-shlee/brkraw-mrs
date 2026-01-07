#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "src" / "brkraw_mrs" / "__init__.py"


def get_version() -> str:
    text = INIT_PATH.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise SystemExit("Failed to detect __version__ in src/brkraw_mrs/__init__.py")
    return match.group(1)


def run_git(args: list[str]) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"git {' '.join(args)} failed: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag and push the current version.")
    parser.add_argument("--remote", default="origin", help="Remote to push tag to.")
    args = parser.parse_args()

    version = get_version()
    run_git(["tag", version])
    run_git(["push", args.remote, version])
    print(f"Pushed tag {version} to {args.remote}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
