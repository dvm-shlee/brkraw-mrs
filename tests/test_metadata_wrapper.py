"""The metadata cache wrapper must call the original get_metadata with only
the arguments it accepts: brkraw 0.6.0 removed ``context_map`` from
``get_metadata`` (0.5.x still has it). Fake scans stand in for both, so these
tests do not depend on the installed brkraw version."""

import pytest

from brkraw_mrs import hook


class ScanNew:
    """brkraw 0.6.0: get_metadata(reco_id, spec, return_spec)."""

    def __init__(self):
        self.calls = []

    def get_metadata(self, reco_id=None, spec=None, return_spec=False):
        self.calls.append({"reco_id": reco_id, "spec": spec, "return_spec": return_spec})
        return {"from": "original", "reco_id": reco_id}


class ScanOld:
    """brkraw 0.5.x: get_metadata(reco_id, spec, context_map, return_spec)."""

    def __init__(self):
        self.calls = []

    def get_metadata(self, reco_id=None, spec=None, context_map=None, return_spec=False):
        self.calls.append({"reco_id": reco_id, "spec": spec, "context_map": context_map,
                           "return_spec": return_spec})
        return {"from": "original", "reco_id": reco_id}


def _wrapped(scan):
    hook._cache_metadata(scan, {"from": "cache"}, 1)
    return scan


def test_cache_hit_returns_cached_metadata():
    scan = _wrapped(ScanNew())
    assert scan.get_metadata(reco_id=1) == {"from": "cache"}
    assert scan.calls == []


def test_cache_miss_calls_new_style_original_without_context_map():
    scan = _wrapped(ScanNew())
    assert scan.get_metadata(reco_id=2) == {"from": "original", "reco_id": 2}
    assert scan.calls == [{"reco_id": 2, "spec": None, "return_spec": False}]


def test_spec_bypasses_cache_on_new_style():
    scan = _wrapped(ScanNew())
    scan.get_metadata(reco_id=1, spec={"__meta__": {}})
    assert scan.calls[0]["spec"] == {"__meta__": {}}


def test_return_spec_bypasses_cache_on_new_style():
    scan = _wrapped(ScanNew())
    scan.get_metadata(reco_id=1, return_spec=True)
    assert scan.calls[0]["return_spec"] is True


def test_old_style_original_still_gets_context_map():
    scan = _wrapped(ScanOld())
    scan.get_metadata(reco_id=2, context_map="map.yaml")
    assert scan.calls == [{"reco_id": 2, "spec": None, "context_map": "map.yaml", "return_spec": False}]


def test_context_map_on_new_style_is_a_clear_error():
    scan = _wrapped(ScanNew())
    with pytest.raises(TypeError, match="context_map"):
        scan.get_metadata(reco_id=2, context_map="map.yaml")
    assert scan.calls == []


def test_wrapper_is_installed_once():
    scan = _wrapped(ScanNew())
    first = scan.get_metadata
    hook._cache_metadata(scan, {"from": "cache2"}, 1)
    assert scan.get_metadata == first
    assert scan.get_metadata(reco_id=1) == {"from": "cache2"}
