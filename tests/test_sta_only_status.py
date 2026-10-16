from types import SimpleNamespace

import pytest
from dreamplace.Placer import PlacementEngine


def _sta_engine(monkeypatch, *, wns):
    class Inner:
        initialization_profile = {}

        def __init__(self, params, placedb, data):
            pass

        def run_sta_only(self, params, placedb):
            return float("inf"), float("inf"), {
                "timing_stage_summary": {
                    "sta": {"wns": wns, "tns": -2.0, "timing_objective": 3.0}
                }
            }

    monkeypatch.setattr("dreamplace.Placer.NonLinearPlace.NonLinearPlace", Inner)
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace()
    engine.placedb = object()
    engine.congestion = 0.0
    engine.density = 0.0
    engine.setup_placedb = lambda: None
    return engine


def test_sta_only_accepts_finite_timing_without_placement_hpwl(monkeypatch):
    result = _sta_engine(monkeypatch, wns=-1.0).run_sta_only()
    assert result["status"] == "ok"
    assert result["hpwl"] == float("inf")


def test_sta_only_rejects_nonfinite_timing(monkeypatch):
    with pytest.raises(RuntimeError, match="non-finite timing"):
        _sta_engine(monkeypatch, wns=float("nan")).run_sta_only()
