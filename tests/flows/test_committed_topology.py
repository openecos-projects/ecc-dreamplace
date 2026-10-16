"""Geometry metrics survive the diagnostic STA-only return convention."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from dreamplace.ops.placeio_common import physical_mutation
from dreamplace.ops.routability import gpugr_context
from dreamplace.Placer import PlacementEngine

from dreamplace.flows import committed_topology


def test_failed_gp_does_not_enter_s5b1_native_finalization(monkeypatch):
    engine = object.__new__(PlacementEngine)
    engine.params = SimpleNamespace(flow_kind="placement", timing_opt_enabled=1)
    engine.setup_placedb = Mock()
    engine.place = Mock()
    failure = {"status": "failed", "stop_reason": "nonfinite_metric", "entered_legalization": False}
    engine.placer = SimpleNamespace(last_placement_debug_summary=failure)
    finalize = Mock(side_effect=AssertionError("native commit must not follow a failed GP"))
    monkeypatch.setattr(committed_topology, "finalize_inflation_s5b1", finalize)
    assert engine.run() == failure
    finalize.assert_not_called()


def test_s5b1_terminal_keeps_legal_geometry_metrics_after_sta_only(tmp_path, monkeypatch):
    class LegalPlacer:
        def __init__(self, params, placedb, timer):
            self.pos = [torch.zeros(2)]
            self.op_collections = SimpleNamespace(legality_check_op=lambda pos: True)
            self.last_placement_debug_summary = {"status": "ok"}

        def __call__(self, params, placedb):
            return 3.25, 4.5, {}

        def run_sta_only(self, params, placedb):
            return float("inf"), float("inf"), {
                "timing_stage_summary": {"sta": {"wns": -10., "tns": -100.}}
            }

    request = SimpleNamespace(actions=({},), to_summary=lambda: {"actions": 1})
    owner = SimpleNamespace(
        terminal_request=request, terminal_commit_started=False,
        terminal_summary={"gp_area": {"movable": 2.}},
        _sync_native_sizing=lambda: {"ok": True},
        ops=SimpleNamespace(steiner_topo_op=SimpleNamespace(
            frozen_nets=SimpleNamespace(segment_state=object())
        )), data=SimpleNamespace(), lane=SimpleNamespace(),
    )
    backend = SimpleNamespace(
        commit_buffers=lambda *args, **kwargs: {"status": "accepted", "accepted_action_count": 1},
        refresh=lambda **kwargs: {"status": "ok"},
    )
    engine = SimpleNamespace(
        params=SimpleNamespace(
            result_dir=str(tmp_path), design_name=lambda: "gcd", enable_fillers=0,
        ),
        placedb=SimpleNamespace(total_movable_node_area=2., backend_caps={}),
        placer=SimpleNamespace(inflation_s5b1=owner, pos=[torch.zeros(2)]),
        _physical_mutation_backend=lambda: backend,
        write_verilog=Mock(), _write_json_artifact=Mock(),
    )
    monkeypatch.setattr(committed_topology.NonLinearPlace, "NonLinearPlace", LegalPlacer)
    monkeypatch.setattr(physical_mutation, "write_native_def", Mock())
    monkeypatch.setattr(gpugr_context, "write_back_movable_lpos", Mock())
    result = committed_topology.finalize_inflation_s5b1(engine)
    assert {key: result[key] for key in ("status", "rsmt", "hpwl")} == {
        "status": "ok", "rsmt": 3.25, "hpwl": 4.5,
    }
    assert result["inflation_s5b1_terminal"]["final_timing"] == {"wns": -10., "tns": -100.}
