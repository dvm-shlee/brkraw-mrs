from __future__ import annotations

import argparse
from pathlib import Path
from importlib import resources
from typing import Optional, List

import yaml

from brkraw.apps import addon as addon_app


def install_specs_and_rules(root: Optional[str] = None):
    spec_path = resources.files("brkraw_mrs.specs").joinpath("metadata_press_svs.yaml")
    rules_path = resources.files("brkraw_mrs.rules").joinpath("press_svs.yaml")

    installed = []
    with resources.as_file(spec_path) as spec_file:
        spec_data = yaml.safe_load(Path(spec_file).read_text(encoding="utf-8"))
        installed += addon_app.add_spec_data(
            spec_data,
            filename="metadata_press_svs.yaml",
            source_path=Path(spec_file),
            root=root,
        )
    with resources.as_file(rules_path) as rules_file:
        rules_data = yaml.safe_load(Path(rules_file).read_text(encoding="utf-8"))
        installed += addon_app.add_rule_data(
            rules_data,
            filename="press_svs.yaml",
            source_path=Path(rules_file),
            root=root,
        )
    return installed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brkraw-mrs-install",
        description="Install brkraw-mrs rules and metadata spec into BrkRaw config.",
    )
    parser.add_argument(
        "--root",
        help="BrkRaw config root (defaults to BRKRAW_CONFIG_HOME or ~/.brkraw)",
        default=None,
    )
    args = parser.parse_args(argv)
    install_specs_and_rules(root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
