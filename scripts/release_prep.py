#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "src" / "brkraw_mrs" / "__init__.py"
README_PATH = REPO_ROOT / "README.md"
RELEASE_NOTES_PATH = REPO_ROOT / "RELEASE_NOTES.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def update_init_version(version: str) -> None:
    text = INIT_PATH.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"^__version__\s*=\s*['\"][^'\"]+['\"]\s*$",
        f"__version__ = '{version}'",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Failed to update __version__ in __init__.py")
    INIT_PATH.write_text(new_text, encoding="utf-8")


def update_readme_version(version: str) -> None:
    if not README_PATH.exists():
        return
    text = README_PATH.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"brkraw-mrs\s+\(v[0-9A-Za-z.\-]+\)",
        f"brkraw-mrs (v{version})",
        text,
        count=1,
    )
    if count == 0:
        return
    README_PATH.write_text(new_text, encoding="utf-8")


def update_pyproject_version(version: str) -> bool:
    if not PYPROJECT_PATH.exists():
        return False
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"^version\s*=\s*['\"][^'\"]+['\"]\s*$",
        f'version = "{version}"',
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        return False
    if count != 1:
        raise RuntimeError("Failed to update version in pyproject.toml")
    PYPROJECT_PATH.write_text(new_text, encoding="utf-8")
    return True


def update_pyproject_classifiers(status_label: str) -> bool:
    if not PYPROJECT_PATH.exists():
        return False
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    status_value = f"Development Status :: {status_label}"
    new_text, count = re.subn(
        r"^\s*[\"']Development Status :: [^\"']+[\"'],?\s*$",
        f"    '{status_value}',",
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        return False
    if count != 1:
        raise RuntimeError("Failed to update Development Status classifier in pyproject.toml")
    PYPROJECT_PATH.write_text(new_text, encoding="utf-8")
    return True


def determine_status(version: str) -> tuple[str, str, bool]:
    v = version.lower()
    if re.search(r"a\d+", v):
        return ("3 - Alpha", "alpha", False)
    if re.search(r"b\d+", v):
        return ("4 - Beta", "beta", False)
    if re.search(r"rc\d+", v):
        return ("4 - Beta", "release candidate", False)
    return ("5 - Production/Stable", "stable", True)


def generate_release_notes(version: str) -> None:
    last_tag = None
    tag_result = run_git(["describe", "--tags", "--abbrev=0"])
    if tag_result.returncode == 0:
        last_tag = tag_result.stdout.strip()

    log_range = f"{last_tag}..HEAD" if last_tag else "HEAD"
    log_result = run_git(
        ["log", log_range, "--no-merges", "--pretty=format:- %s (%h)"]
    )
    changes = log_result.stdout.strip()
    if not changes:
        changes = "- (no changes found)"

    date_str = dt.date.today().isoformat()
    header = f"# Release v{version}\n\n"
    meta = f"Date: {date_str}\n"
    scope = f"Changes since {last_tag}\n\n" if last_tag else "Changes\n\n"
    RELEASE_NOTES_PATH.write_text(header + meta + scope + changes + "\n", encoding="utf-8")


def fetch_tags(remote: str) -> None:
    fetch_result = run_git(["fetch", "--tags", remote])
    if fetch_result.returncode != 0:
        stderr = fetch_result.stderr.strip()
        print(f"Warning: failed to fetch tags from {remote}: {stderr}")
        if remote != "origin":
            fallback = run_git(["fetch", "--tags", "origin"])
            if fallback.returncode != 0:
                fallback_err = fallback.stderr.strip()
                print(f"Warning: failed to fetch tags from origin: {fallback_err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare release: bump version and generate RELEASE_NOTES.md"
    )
    parser.add_argument("--version", required=True, help="Release version (PEP 440)")
    parser.add_argument(
        "--fetch-tags",
        action="store_true",
        help="Fetch tags from remote before generating notes",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote name for fetching tags (default: origin)",
    )
    args = parser.parse_args()

    if args.fetch_tags:
        fetch_tags(args.remote)

    status_classifier, status_label, _is_stable = determine_status(args.version)
    update_init_version(args.version)
    update_readme_version(args.version)
    update_pyproject_version(args.version)
    update_pyproject_classifiers(status_classifier)
    generate_release_notes(args.version)

    print(f"Updated {INIT_PATH}")
    if README_PATH.exists():
        print(f"Checked {README_PATH}")
    print(f"Updated {PYPROJECT_PATH}")
    print(f"Generated {RELEASE_NOTES_PATH}")


if __name__ == "__main__":
    main()
