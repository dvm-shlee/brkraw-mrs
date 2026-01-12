from __future__ import annotations

from typing import Any, Optional

import tkinter as tk
from tkinter import ttk

from brkraw.core import config as brkraw_config
from brkraw.specs.rules import load_rules, select_rule_use

from .ui import MRSPanel


class MRSViewerHook:
    name = "brkraw-mrs"
    tab_title = "MRS"
    priority = 10

    def __init__(self) -> None:
        self._panel: Optional[MRSPanel] = None

    def build_tab(self, parent: tk.Misc, app: Any) -> Optional[tk.Widget]:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        panel = MRSPanel(frame, app=app)
        panel.grid(row=0, column=0, sticky="nsew")
        self._panel = panel
        panel.refresh_from_viewer()
        return frame

    def can_handle(self, scan: Any) -> bool:
        try:
            base = brkraw_config.resolve_root()
            rules = load_rules(root=base, validate=False)
            use = select_rule_use(
                scan,
                rules.get("converter_hook", []),
                base=base,
                resolve_paths=False,
            )
            return use == "mrs"
        except Exception:
            return False

    def on_dataset_loaded(self, app: Any) -> None:
        if self._panel is not None:
            self._panel.refresh_from_viewer()

    def on_scan_selected(self, app: Any) -> None:
        if self._panel is not None:
            self._panel.refresh_from_viewer()
