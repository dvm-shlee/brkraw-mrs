"""Transform helpers for brkraw-mrs metadata mapping."""

from __future__ import annotations

import os
from typing import Any, Optional
from importlib import metadata


def strip_bruker_string(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text.strip()


def ensure_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    try:
        import numpy as np
        if isinstance(value, np.ndarray) and value.size:
            return value.flat[0]
    except Exception:
        pass
    return value


def first_float(value: Any) -> Optional[float]:
    val = _first(value)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def to_float_list(value: Any) -> Optional[list]:
    values = ensure_list(value)
    if values is None:
        return None
    out = []
    for item in values:
        try:
            out.append(float(item))
        except Exception:
            return None
    return out


def ms_to_s(value: Any) -> Optional[float]:
    val = first_float(value)
    if val is None:
        return None
    return val * 1e-3


def dwell_time_seconds(*, sw_hz: Any = None, dwell_us: Any = None) -> Optional[float]:
    sw = first_float(sw_hz)
    if sw:
        return 1.0 / sw
    dwell = first_float(dwell_us)
    if dwell is None:
        return None
    return float(dwell) * 1e-6 * 2.0


def value_only(*, value: Any) -> Any:
    return value


def plugin_version(*, value: Any) -> str:
    _ = value
    try:
        return metadata.version("brkraw-mrs")
    except Exception:
        return "unknown"


def source_dataset_env(*, value: Any) -> Optional[str]:
    _ = value
    return os.environ.get("BRKRAW_SOURCE_DATASET")


def scan_id_to_string(*, scan_id: Any) -> Optional[str]:
    if scan_id is None:
        return None
    return str(scan_id)
