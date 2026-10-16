import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamplace.macroPlaceDB import MacroPlaceDB
from dreamplace.Placer import PlacementEngine
from dreamplace.ops.buffer_insertion.buffering_lane import BufferCommitRequest
from dreamplace.ops.placeio_common.backend_contract import ieda_backend_caps


class _PendingCommitRequest:
    def __init__(self, reason="joint_buffer_final_commit"):
        self.version = 1
        self.action_digest = "request-digest"
        self.actions = ({"candidate_id": 7},)
        self.reason = reason

    def to_summary(self):
        return {
            "request_version": self.version,
            "action_digest": self.action_digest,
            "action_count": len(self.actions),
            "reason": self.reason,
        }


class _FakeMutationBackend:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def commit_buffers(self, request, *, output_dir):
        self.calls.append((request, output_dir))
        return dict(self.result)


def _pending_commit_engine(result, *, reason="joint_buffer_final_commit"):
    request = _PendingCommitRequest(reason=reason)
    mutation_backend = _FakeMutationBackend(result)
    lane = SimpleNamespace(config=SimpleNamespace(output_dir="unit_output"))
    inner = SimpleNamespace(
        last_buffer_commit_request=request,
        joint_coordinator=SimpleNamespace(buffering_lane=lane),
        last_joint_flow_artifact_path=None,
        _mutation_backend=mutation_backend,
    )
    inner.recorded = []
    inner.record_outer_buffer_commit_result = lambda payload: inner.recorded.append(
        dict(payload)
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(result_dir="", base_design_name="unit_design")
    engine.placedb = object()
    engine.placer = inner
    engine._physical_mutation_backend = lambda: engine.placer._mutation_backend
    engine.metrics = {
        "post_commit_control_result": {
            "status": "pending_buffer_commit_request",
            "commit_request": request.to_summary(),
        }
    }
    return engine, mutation_backend, inner, request


def _refresh_backend(placedb):
    return SimpleNamespace(
        refresh=lambda *, refresh_mode, rebuild_mode: (
            placedb.refresh_from_openroad_bridge(
                refresh_mode=refresh_mode,
                rebuild_mode=rebuild_mode,
            )
        )
    )


@pytest.mark.parametrize(
    ("with_sizing", "with_buffer", "expected_order"),
    (
        (True, False, ["coordinate_sync", "apply_sizing"]),
        (False, True, ["coordinate_sync", "insert_buffers"]),
        (
            True,
            True,
            ["coordinate_sync", "apply_sizing", "insert_buffers"],
        ),
    ),
)
def test_segment_joint_executor_orders_sizing_and_buffer_actions_once(
    monkeypatch,
    with_sizing,
    with_buffer,
    expected_order,
):
    calls = []

    monkeypatch.setattr(
        "dreamplace.Placer.placeio_openroad.PlaceIOFunction.apply",
        staticmethod(lambda bridge, x, y: calls.append("coordinate_sync")),
    )
    monkeypatch.setattr(
        "dreamplace.Placer.placeio_openroad.PlaceIOFunction.apply_sizing",
        staticmethod(
            lambda bridge, cell_ids, master_names: (
                calls.append("apply_sizing")
                or {
                    "applied": 1,
                    "missing_masters": 0,
                    "invalid_cell_ids": 0,
                }
            )
        ),
    )

    class MutationBackend:
        def commit_buffers(self, request, *, output_dir):
            calls.append("insert_buffers")
            return {
                "status": "accepted",
                "accepted_action_count": len(request.actions),
                "records": [],
            }

    actions = ({"action_id": 1},) if with_buffer else ()
    sizing_actions = (
        ({"instance_id": 0, "area_delta_internal": 1.5},)
        if with_sizing
        else ()
    )
    request = BufferCommitRequest(
        version=1,
        mode="segment",
        actions=actions,
        action_digest="buffer-digest",
        iteration=20,
        reason="joint_buffer_final_commit",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path="",
        refresh_mode="topo",
        rebuild_mode="topo",
        projection_metadata={},
        runtime_refresh_summary={},
        sizing_actions=sizing_actions,
        sizing_action_digest="sizing-digest",
        sizing_cell_ids=(torch.tensor([1]) if with_sizing else None),
        placement_node_x=torch.tensor([10.0]),
        placement_node_y=torch.tensor([20.0]),
        placement_snapshot_digest="placement-digest",
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.placedb = SimpleNamespace(
        dtype=np.float32,
        openroad_bridge=object(),
        flat_libcell_names=[b"A", b"B"],
    )

    result = engine._execute_segment_joint_physical_actions(
        request,
        mutation_backend=MutationBackend(),
        output_dir="unused",
    )

    assert calls == expected_order
    assert [row["stage"] for row in result["physical_action_trace"]] == expected_order
    assert result["sizing_action_count"] == int(with_sizing)
    assert result["buffer_action_count"] == int(with_buffer)
    assert result["sizing_area_delta_internal"] == (1.5 if with_sizing else 0.0)


def test_segment_joint_buffer_preflight_uses_fresh_def_only_openroad(
    monkeypatch,
    tmp_path,
):
    class SourceBridge:
        def write_def(self, path):
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n")

    class TrialBridge:
        def __init__(self):
            self.commands = []

        def eval_tcl_string(self, command):
            self.commands.append(command)

    trial_bridge = TrialBridge()
    replay_params_seen = []
    trial_calls = []

    def fake_read(params):
        replay_params_seen.append(params)
        return trial_bridge

    def fake_commit(actions, **kwargs):
        trial_calls.append((tuple(actions), dict(kwargs)))
        return {
            "status": "accepted",
            "accepted_action_count": 1,
            "failed_action_count": 0,
            "qor_status": "pass",
            "qor_pass": True,
            "before_wns": -10.0,
            "before_tns": -100.0,
            "after_wns": -9.0,
            "after_tns": -80.0,
            "delta_wns": 1.0,
            "delta_tns": 20.0,
            "artifact_path": str(tmp_path / "trial.json"),
        }

    monkeypatch.setattr(
        "dreamplace.Placer.placeio_openroad.PlaceIOFunction.read",
        staticmethod(fake_read),
    )
    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.coordinate_backend.commit_coordinate_buffer_actions",
        fake_commit,
    )

    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_quality_profile="segment_count_direct_joint_v1",
        design_inputs={
            "def": "source.def",
            "verilog": "source.v",
            "sdc": "source.sdc",
        },
    )
    engine.placedb = SimpleNamespace(openroad_bridge=SourceBridge())
    request = SimpleNamespace(actions=({"action_id": 7},))

    result = engine._run_segment_joint_buffer_preflight(
        request,
        output_dir=str(tmp_path),
        source_baseline={"status": "ok", "wns": -10.0, "tns": -100.0},
    )

    assert result["status"] == "pass"
    assert result["delta_tns"] == 20.0
    assert result["source_baseline_match"] is True
    assert result["source_baseline_gap"] == {"wns": 0.0, "tns": 0.0}
    assert result["pre_buffer_def_sha256"]
    assert replay_params_seen[0].design_inputs["def"] == result["pre_buffer_def_path"]
    assert "verilog" not in replay_params_seen[0].design_inputs
    assert replay_params_seen[0].verilog_input == ""
    assert trial_bridge.commands == ["estimate_parasitics -placement"]
    assert trial_calls[0][0] == request.actions
    assert trial_calls[0][1]["openroad_bridge"] is trial_bridge
    assert trial_calls[0][1]["realization_config"] == {
        "disposable_design": True
    }
    assert (tmp_path / "unit_design_segment_joint_buffer_preflight.json").exists()


def test_segment_joint_executor_preserves_sizing_state_when_preflight_rejects(
    monkeypatch,
    tmp_path,
):
    calls = []

    class SourceBridge:
        def write_def(self, path):
            calls.append(("write_def", path))
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n")

    class MutationBackend:
        def commit_buffers(self, request, *, output_dir):
            raise AssertionError("rejected preflight must not mutate the source design")

    monkeypatch.setattr(
        "dreamplace.Placer.placeio_openroad.PlaceIOFunction.apply",
        staticmethod(lambda bridge, x, y: calls.append(("coordinate_sync", len(x)))),
    )
    request = BufferCommitRequest(
        version=1,
        mode="segment",
        actions=({"action_id": 1},),
        action_digest="buffer-digest",
        iteration=20,
        reason="joint_buffer_final_commit",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path=str(tmp_path / "preserved.def"),
        refresh_mode="topo",
        rebuild_mode="topo",
        projection_metadata={},
        runtime_refresh_summary={},
        placement_node_x=torch.tensor([10.0]),
        placement_node_y=torch.tensor([20.0]),
        placement_snapshot_digest="placement-digest",
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_quality_profile="segment_count_direct_joint_v1",
    )
    engine.placedb = SimpleNamespace(
        dtype=np.float32,
        openroad_bridge=SourceBridge(),
        flat_libcell_names=[b"A"],
    )
    engine._query_segment_joint_pre_buffer_opensta = lambda: {
        "status": "ok",
        "wns": -10.0,
        "tns": -100.0,
    }
    engine._run_segment_joint_buffer_preflight = (
        lambda request, output_dir, source_baseline: {
        "status": "rejected",
        "trial_qor_status": "fail_tns_nonpositive",
        "before_wns": -10.0,
        "before_tns": -100.0,
        "after_wns": -11.0,
        "after_tns": -110.0,
        "delta_wns": -1.0,
        "delta_tns": -10.0,
        }
    )

    result = engine._execute_segment_joint_physical_actions(
        request,
        mutation_backend=MutationBackend(),
        output_dir=str(tmp_path),
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "segment_joint_buffer_preflight_rejected"
    assert result["accepted_action_count"] == 0
    assert result["preflight_rejected"] is True
    assert result["before_tns"] == -100.0
    assert result["physical_action_trace"][-1]["status"] == "preflight_rejected"
    assert result["physical_action_trace"][-1]["preflight_delta_tns"] == -10.0
    assert (tmp_path / "preserved.def").exists()


def test_segment_joint_preflight_rejection_skips_terminal_legalization():
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        flow_kind="joint",
        joint_quality_profile="segment_count_direct_joint_v1",
    )
    control_result = {
        "status": "completed",
        "commit": {
            "preflight_rejected": True,
            "qor_status": "fail_wns_regression",
            "action_count": 9,
            "accepted_action_count": 0,
            "preflight": {
                "trial_qor_status": "fail_wns_regression",
                "pre_buffer_def_path": "/tmp/pre_buffer.def",
            },
        },
    }

    result = engine._continue_after_joint_buffer_commit(control_result)

    assert result["status"] == "ok"
    assert result["qor_status"] == "fail_wns_regression"
    assert result["continuation_policy"] == (
        "preflight_rejected_preserve_prebuffer_state"
    )
    assert result["accepted_action_count"] == 0
    assert result["pre_buffer_def_path"] == "/tmp/pre_buffer.def"


def test_segment_joint_final_def_verifies_every_requested_sizing_master(tmp_path):
    final_def = tmp_path / "final.def"
    final_def.write_text(
        "VERSION 5.8 ;\n"
        "COMPONENTS 2 ;\n"
        "- inst0 CELL_X2 + PLACED ( 0 0 ) N ;\n"
        "- inst1 CELL_X1 + PLACED ( 10 0 ) N ;\n"
        "END COMPONENTS\n"
        "END DESIGN\n",
        encoding="utf-8",
    )
    request = BufferCommitRequest(
        version=1,
        mode="segment",
        actions=(),
        action_digest="buffer-digest",
        iteration=20,
        reason="joint_buffer_final_commit",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path="",
        refresh_mode="topo",
        rebuild_mode="topo",
        projection_metadata={},
        runtime_refresh_summary={},
        sizing_actions=(
            {
                "instance_id": 0,
                "after_master": "CELL_X2",
                "area_delta_internal": 1.0,
            },
        ),
        sizing_action_digest="sizing-digest",
        sizing_cell_ids=torch.tensor([1, 0]),
        placement_node_x=torch.tensor([0.0, 10.0]),
        placement_node_y=torch.tensor([0.0, 0.0]),
        placement_snapshot_digest="placement-digest",
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.placedb = SimpleNamespace(node_names=[b"inst0", b"inst1"])

    result = engine._verify_segment_joint_final_sizing_masters(final_def, request)

    assert result["status"] == "ok"
    assert result["expected_count"] == 1
    assert result["matched_count"] == 1
    assert result["request_action_digest"] == "sizing-digest"


def test_segment_joint_final_def_rejects_wrong_sizing_master(tmp_path):
    final_def = tmp_path / "final.def"
    final_def.write_text(
        "COMPONENTS 1 ;\n"
        "- inst0 CELL_X1 + PLACED ( 0 0 ) N ;\n"
        "END COMPONENTS\n",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        sizing_actions=({"instance_id": 0, "after_master": "CELL_X2"},),
        sizing_action_digest="sizing-digest",
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.placedb = SimpleNamespace(node_names=[b"inst0"])

    with pytest.raises(RuntimeError, match="do not match request"):
        engine._verify_segment_joint_final_sizing_masters(final_def, request)


def test_placement_engine_executes_pending_request_once_and_records_accepted_result():
    engine, lane, inner, request = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1}
    )

    result = engine._execute_pending_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert len(lane.calls) == 1
    assert lane.calls[0][0] is request
    assert result["status"] == "safe_stopped_after_accepted_buffer_commit"
    assert result["topology_mutated"]
    assert result["continuation_policy"] == "rebuild_placer_outer_loop"
    assert result["commit_request"]["action_digest"] == "request-digest"
    assert inner.recorded[0]["commit"]["status"] == "accepted"


def test_placement_engine_executes_standalone_buffering_request_once():
    engine, mutation_backend, inner, request = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1}
    )
    engine.params.flow_kind = "buffering"
    inner.joint_coordinator = None
    inner.last_buffering_lane = None

    result = engine._execute_pending_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert len(mutation_backend.calls) == 1
    assert mutation_backend.calls[0][0] is request
    assert result["status"] == "safe_stopped_after_accepted_buffer_commit"
    assert result["topology_mutated"]


def test_placement_engine_executes_request_without_result_recorder():
    engine, lane, inner, _ = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1}
    )
    inner.record_outer_buffer_commit_result = None

    result = engine._execute_pending_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert len(lane.calls) == 1
    assert result["status"] == "safe_stopped_after_accepted_buffer_commit"
    assert result["topology_mutated"]


def test_placement_engine_fails_stale_request_without_retrying_executor():
    engine, lane, inner, _ = _pending_commit_engine(
        {"status": "failed", "reason": "backend_failure"}
    )

    result = engine._execute_pending_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert len(lane.calls) == 1
    assert result["status"] == "failed"
    assert not result["topology_mutated"]
    assert result["continuation_policy"] == "terminate_outer_flow"
    assert inner.recorded[0]["commit"]["reason"] == "backend_failure"


def test_placement_engine_skips_when_inner_window_has_no_request():
    engine, lane, _, _ = _pending_commit_engine({"status": "accepted"})
    engine.placer.last_buffer_commit_request = None

    result = engine._execute_pending_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert result["status"] == "failed"
    assert result["reason"] == "missing_buffer_commit_request"
    assert lane.calls == []


def test_placement_engine_run_executes_pending_joint_request_before_rebuild():
    engine, lane, _, _ = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1}
    )
    engine.params.flow_kind = "joint"
    engine.setup_placedb = lambda: None
    engine.place = lambda: None
    engine.rsmt = float("nan")
    engine.hpwl = float("nan")
    engine.congestion = float("inf")
    engine.density = float("inf")
    rebuild_controls = []
    engine._continue_after_joint_buffer_commit = lambda control: (
        rebuild_controls.append(dict(control))
        or {"status": "ok", "topology_generation": 1}
    )

    result = engine.run()

    assert result["status"] == "ok"
    assert result["joint_post_rebuild"]["topology_generation"] == 1
    assert len(lane.calls) == 1
    assert rebuild_controls[0]["topology_mutated"]


def test_standalone_buffering_refreshes_once_without_another_optimization_window(monkeypatch):
    engine, lane, _, _ = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1}
    )
    engine.params.flow_kind = "buffering"
    engine.setup_placedb = lambda: None
    engine.place = lambda: None
    engine.rsmt = engine.hpwl = engine.congestion = engine.density = float("inf")
    refresh_calls = []
    refreshed = {"status": "ok", "topology_generation": 1}

    def refresh(engine_arg, control):
        refresh_calls.append(control)
        assert engine_arg is engine
        return refreshed

    monkeypatch.setattr("dreamplace.flows.committed_topology.refresh_committed_topology", refresh)
    engine.placer.last_buffering_lane_summary = {
        "status": "completed",
        "mode": "segment",
        "commit": {"status": "accepted"},
    }

    result = engine.run()

    assert result["status"] == "ok"
    assert result["joint_post_rebuild"] == refreshed
    assert len(refresh_calls) == 1
    assert len(lane.calls) == 1


def test_placement_engine_run_stops_after_executor_failure():
    engine, lane, _, _ = _pending_commit_engine(
        {"status": "failed", "reason": "backend_failure"}
    )
    engine.params.flow_kind = "joint"
    engine.setup_placedb = lambda: None
    engine.place = lambda: None
    engine._continue_after_joint_buffer_commit = lambda _: pytest.fail(
        "failed request must not rebuild"
    )

    result = engine.run()

    assert result["status"] == "failed"
    assert result["stop_reason"] == "buffer_commit_execution_failed"
    assert len(lane.calls) == 1


def test_placement_engine_run_stops_before_refresh_for_ieda_buffer_commit():
    engine, _, _, _ = _pending_commit_engine({"status": "accepted"})
    engine.params.flow_kind = "joint"
    engine.placedb = SimpleNamespace(backend_caps=ieda_backend_caps().to_dict())
    engine._physical_mutation_backend = PlacementEngine._physical_mutation_backend.__get__(
        engine,
        PlacementEngine,
    )
    engine.setup_placedb = lambda: None
    engine.place = lambda: None
    engine._continue_after_joint_buffer_commit = lambda _: pytest.fail(
        "unsupported iEDA buffer mutation must not refresh or rebuild"
    )

    result = engine.run()

    assert result["status"] == "failed"
    assert result["stop_reason"] == "buffer_commit_execution_failed"
    assert result["joint_buffer_commit"]["reason"] == (
        "unsupported_backend_capability"
    )
    assert not result["joint_buffer_commit"]["topology_mutated"]
    assert result["joint_buffer_commit"]["continuation_policy"] == (
        "terminate_outer_flow"
    )


def test_placement_engine_restarts_a_fresh_window_after_periodic_commit():
    engine, first_lane, first_inner, _ = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1},
        reason="joint_buffer_periodic_commit",
    )
    _, second_lane, second_inner, _ = _pending_commit_engine(
        {"status": "accepted", "accepted_action_count": 1},
        reason="joint_buffer_final_commit",
    )
    engine.params.flow_kind = "joint"
    engine.params.joint_buffer_outer_iterations = 2
    engine.setup_placedb = lambda: None
    engine.rsmt = float("nan")
    engine.hpwl = float("nan")
    engine.congestion = float("inf")
    engine.density = float("inf")
    windows = [first_inner, second_inner]
    place_reuse_flags = []

    def fake_place(*, reuse_existing=False):
        place_reuse_flags.append(bool(reuse_existing))
        engine.placer = windows.pop(0)
        engine.metrics = {
            "post_commit_control_result": {
                "status": "pending_buffer_commit_request",
                "commit_request": engine.placer.last_buffer_commit_request.to_summary(),
            }
        }

    topology_generations = []

    def fake_continue(control):
        topology_generations.append(control["commit_request"]["reason"])
        return {"status": "ok", "topology_generation": len(topology_generations)}

    engine.place = fake_place
    engine._continue_after_joint_buffer_commit = fake_continue

    result = engine.run()

    assert result["status"] == "ok"
    assert place_reuse_flags == [False, True]
    assert topology_generations == [
        "joint_buffer_periodic_commit",
        "joint_buffer_final_commit",
    ]
    assert len(first_lane.calls) == 1
    assert len(second_lane.calls) == 1
    assert [window["boundary"] for window in result["outer_windows"]] == [
        "periodic",
        "final",
    ]


class FakePyDB:
    def __init__(self, nodes=2, pins=3, nets=1):
        self.num_nodes = nodes
        self.num_pins = pins
        self.num_nets = nets
        self.pin_names = [f"p{i}" for i in range(pins)]
        self.net_names = [f"n{i}" for i in range(nets)]


class FakeBridge:
    def __init__(self):
        self.pydb = FakePyDB(nodes=4, pins=5, nets=2)
        self.tcl_calls = []

    def sync_from_openroad(self):
        return self.pydb

    def eval_tcl_string(self, command):
        self.tcl_calls.append(command)
        return ""


def test_macroplacedb_refresh_requires_openroad_bridge():
    placedb = MacroPlaceDB(SimpleNamespace())

    with pytest.raises(RuntimeError, match="no openroad_bridge"):
        placedb.refresh_from_openroad_bridge()


def test_macroplacedb_refresh_rebuilds_from_synced_pydb(monkeypatch):
    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.openroad_bridge = FakeBridge()
    placedb.params = SimpleNamespace(
        shift_factor=[0.0, 0.0],
        scale_factor=0.0,
        macro_halo_x=0.0,
        macro_halo_y=0.0,
        macro_pin_halo_x=0.0,
        macro_pin_halo_y=0.0,
        cell_padding_x=0.0,
        bndry_padding_x=0.0,
        bndry_padding_y=0.0,
        target_density=0.7,
        route_info_input=None,
        num_bins_x=16,
        num_bins_y=16,
        macro_place_flag=False,
    )
    placedb.num_physical_nodes = 1
    placedb.pin2net_map = [0]
    placedb.net2pin_map = [[0]]
    calls = []

    def fake_initialize_from_rawdb(pydb, params):
        calls.append(("initialize_from_rawdb", pydb.num_nodes))
        placedb.num_physical_nodes = pydb.num_nodes
        placedb.pin2net_map = list(range(pydb.num_pins))
        placedb.net2pin_map = [list(range(pydb.num_pins)) for _ in range(pydb.num_nets)]

    def fake_initialize(params):
        calls.append(("initialize", placedb.num_physical_nodes))
        placedb.num_filler_nodes = 0

    monkeypatch.setattr(placedb, "initialize_from_rawdb", fake_initialize_from_rawdb)
    monkeypatch.setattr(placedb, "initialize", fake_initialize)

    summary = placedb.refresh_from_openroad_bridge(
        refresh_mode="topo",
        rebuild_mode="topo",
    )

    assert calls == [("initialize_from_rawdb", 4), ("initialize", 4)]
    assert summary["requested_refresh_mode"] == "topo"
    assert summary["effective_refresh_mode"] == "topo"
    assert summary["rebuild"]["requested_rebuild_mode"] == "topo"
    assert summary["rebuild"]["post_counts"] == {"nodes": 4, "pins": 5, "nets": 2}
    assert placedb.topology_generation == 1


def test_macroplacedb_rejects_unsupported_rebuild_mode():
    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.params = SimpleNamespace()

    with pytest.raises(RuntimeError, match="unsupported macroPlaceDB rebuild_mode"):
        placedb.rebuild_from_pydb(FakePyDB(), rebuild_mode="affected_net")


def test_macroplacedb_apply_skips_sizing_writeback_for_place_only(monkeypatch):
    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.num_physical_nodes = 1
    placedb.num_terminals = 0
    placedb.num_terminal_NIs = 0
    placedb.node_x = torch.zeros(2).numpy()
    placedb.node_y = torch.zeros(2).numpy()
    sizing_calls = []
    placement_calls = []

    monkeypatch.setattr(placedb, "write_sizing_back", lambda: sizing_calls.append("sizing"))
    monkeypatch.setattr(
        placedb,
        "write_placement_back",
        lambda node_x, node_y, **kwargs: placement_calls.append((node_x.copy(), node_y.copy(), kwargs)),
    )

    params = SimpleNamespace(
        scale_factor=1.0,
        shift_factor=[0.0, 0.0],
        placement_sizing_mode="place_only",
    )
    placedb.apply(params, torch.tensor([3.0]).numpy(), torch.tensor([4.0]).numpy())

    assert sizing_calls == []
    assert len(placement_calls) == 1
    assert placement_calls[0][0][0] == 3.0
    assert placement_calls[0][1][0] == 4.0
    assert placement_calls[0][2] == {"refresh_parasitics": True}


def test_openroad_write_placement_back_refreshes_parasitics_after_sync(monkeypatch):
    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.params = SimpleNamespace(place_io_engine="openroad", macro_only=False)
    placedb.openroad_bridge = FakeBridge()
    calls = []

    def fake_apply(raw_db, node_x, node_y):
        calls.append(("apply", raw_db, tuple(node_x), tuple(node_y)))

    def fake_refresh_from_openroad_bridge():
        calls.append(("refresh",))

    monkeypatch.setattr(placeio_openroad.PlaceIOFunction, "apply", fake_apply)
    monkeypatch.setattr(
        placedb,
        "refresh_from_openroad_bridge",
        fake_refresh_from_openroad_bridge,
    )

    placedb.write_placement_back([1.0, 2.0], [3.0, 4.0])

    assert calls == [
        ("apply", placedb.openroad_bridge, (1.0, 2.0), (3.0, 4.0)),
        ("refresh",),
    ]
    assert placedb.openroad_bridge.tcl_calls == ["estimate_parasitics -placement"]


def test_openroad_size_only_apply_skips_placement_parasitic_refresh(monkeypatch):
    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.params = SimpleNamespace(place_io_engine="openroad", macro_only=False)
    placedb.num_physical_nodes = 1
    placedb.num_terminals = 0
    placedb.num_terminal_NIs = 0
    placedb.node_x = torch.tensor([1.0]).numpy()
    placedb.node_y = torch.tensor([2.0]).numpy()
    placedb.openroad_bridge = FakeBridge()

    monkeypatch.setattr(
        placedb,
        "write_sizing_back",
        lambda: None,
    )
    monkeypatch.setattr(
        placedb,
        "refresh_from_openroad_bridge",
        lambda: None,
    )
    monkeypatch.setattr(
        placeio_openroad.PlaceIOFunction,
        "apply",
        lambda raw_db, node_x, node_y: None,
    )

    params = SimpleNamespace(
        scale_factor=1.0,
        shift_factor=(0.0, 0.0),
        placement_sizing_mode="size_only",
    )
    placedb.apply(params, torch.tensor([3.0]).numpy(), torch.tensor([4.0]).numpy())

    assert placedb.openroad_bridge.tcl_calls == []


def test_macroplacedb_apply_keeps_sizing_writeback_for_size_only(monkeypatch):
    placedb = MacroPlaceDB(SimpleNamespace())
    placedb.num_physical_nodes = 1
    placedb.num_terminals = 0
    placedb.num_terminal_NIs = 0
    placedb.node_x = torch.zeros(2).numpy()
    placedb.node_y = torch.zeros(2).numpy()
    sizing_calls = []

    monkeypatch.setattr(placedb, "write_sizing_back", lambda: sizing_calls.append("sizing"))
    writeback_kwargs = []
    monkeypatch.setattr(
        placedb,
        "write_placement_back",
        lambda node_x, node_y, **kwargs: writeback_kwargs.append(kwargs),
    )

    params = SimpleNamespace(
        scale_factor=1.0,
        shift_factor=[0.0, 0.0],
        placement_sizing_mode="size_only",
    )
    placedb.apply(params, torch.tensor([3.0]).numpy(), torch.tensor([4.0]).numpy())

    assert sizing_calls == ["sizing"]
    assert writeback_kwargs == [{"refresh_parasitics": False}]


def test_placement_engine_rebuilds_fresh_buffering_lane_state_and_parameters(
    monkeypatch, tmp_path
):
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_post_commit_continue=True,
        joint_post_commit_openroad_eco="none",
        plot_flag=False,
        enable_relaxed_buffer_timing=True,
        buffering_segment_count_tns_gradient=1,
    )
    post_commit_pysta_path = tmp_path / "committed_topology_pysta.json"
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = params
    engine.metrics = {
        "post_commit_control_result": {
            "status": "safe_stopped_after_accepted_buffer_commit",
            "topology_mutated": True,
            "refresh_required": True,
            "refresh_mode": "topo",
            "rebuild_mode": "topo",
            "continuation_policy": "rebuild_placer_outer_loop",
            "accepted_action_count": 1,
            "commit": {
                "status": "accepted",
                "accepted_action_count": 1,
                "before_wns": -10.0,
                "before_tns": -20.0,
                "after_wns": -9.0,
                "after_tns": -15.0,
                "delta_wns": 1.0,
                "delta_tns": 5.0,
            },
        }
    }

    class FakePlaceDB:
        def __init__(self):
            self.topology_generation = 0
            self.openroad_bridge = SimpleNamespace(eval_tcl_string=lambda _: "")

        def refresh_from_openroad_bridge(self, refresh_mode="topo", rebuild_mode="topo"):
            self.topology_generation += 1
            return {
                "status": "ok",
                "requested_refresh_mode": refresh_mode,
                "effective_refresh_mode": refresh_mode,
                "rebuild": {
                    "status": "ok",
                    "requested_rebuild_mode": rebuild_mode,
                    "effective_rebuild_mode": rebuild_mode,
                    "topology_generation": self.topology_generation,
                    "pre_counts": {"nodes": 2, "pins": 3, "nets": 1},
                    "post_counts": {"nodes": 3, "pins": 5, "nets": 2},
                },
            }

    constructed = []

    class FakeNonLinearPlace:
        def __init__(self, params_arg, placedb_arg, timer):
            del params_arg, timer
            state = SimpleNamespace(
                z_param=torch.nn.Parameter(
                    torch.tensor([float(placedb_arg.topology_generation)])
                ),
                bsu_index_param=torch.nn.Parameter(torch.tensor([1.0])),
            )
            self.buffer_optimization_state = state
            self.buffer_params = torch.nn.ParameterList(
                [state.z_param, state.bsu_index_param]
            )
            self.last_buffering_lane = SimpleNamespace(
                buffer_optimization_state=state,
                config=SimpleNamespace(
                    post_commit_pysta_json=str(post_commit_pysta_path)
                ),
                optimizer_parameters=lambda: list(self.buffer_params.parameters()),
            )
            self.last_joint_flow_artifact_path = None
            constructed.append(self)

        def run_sta_only(self, params_arg, placedb_arg):
            assert params_arg.enable_relaxed_buffer_timing is False
            assert params_arg.buffering_segment_count_tns_gradient == 0
            return 1.0, 2.0, {"wns": -3.0, "tns": -4.0}

    engine.placedb = FakePlaceDB()
    old_placer = FakeNonLinearPlace(params, engine.placedb, None)
    engine.placer = old_placer
    engine._physical_mutation_backend = lambda: _refresh_backend(engine.placedb)
    monkeypatch.setattr(
        "dreamplace.Placer.NonLinearPlace.NonLinearPlace",
        FakeNonLinearPlace,
    )

    result = engine._continue_after_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    fresh_placer = engine.placer
    old_state = old_placer.buffer_optimization_state
    fresh_state = fresh_placer.buffer_optimization_state
    assert result["status"] == "ok"
    assert len(constructed) == 2
    assert fresh_placer is constructed[1]
    assert fresh_placer.last_buffering_lane is not old_placer.last_buffering_lane
    assert fresh_state is not old_state
    assert fresh_state.z_param is not old_state.z_param
    assert fresh_state.bsu_index_param is not old_state.bsu_index_param
    assert list(fresh_placer.buffer_params.parameters()) == [
        fresh_state.z_param,
        fresh_state.bsu_index_param,
    ]
    assert all(
        fresh_param is not old_param
        for fresh_param, old_param in zip(
            fresh_placer.buffer_params.parameters(),
            old_placer.buffer_params.parameters(),
        )
    )
    assert result["post_commit_pysta_json_path"] == str(post_commit_pysta_path)
    post_commit_pysta = json.loads(post_commit_pysta_path.read_text(encoding="utf-8"))
    assert post_commit_pysta["status"] == "ok"
    assert post_commit_pysta["timing_mode"] == "committed_topology_plain_pysta"
    assert post_commit_pysta["wns"] == -3.0
    assert post_commit_pysta["tns"] == -4.0


def test_placement_engine_runs_optional_openroad_resynth_before_refresh(monkeypatch, tmp_path):
    commit_artifact_path = tmp_path / "buffering_coordinate_commit_results.json"
    commit_artifact_path.write_text(
        json.dumps(
            {
                "artifact": "buffering_coordinate_commit_results",
                "results": [
                    {
                        "accepted": True,
                        "inserted_buffer_name": "buffer_coord_buf_0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_post_commit_continue=True,
        joint_post_commit_openroad_eco="resynth_once",
        plot_flag=False,
        enable_relaxed_buffer_timing=True,
        buffering_segment_count_tns_gradient=1,
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = params
    engine.rsmt = float("nan")
    engine.hpwl = float("nan")
    engine.congestion = float("inf")
    engine.density = float("inf")
    engine.metrics = {
        "post_commit_control_result": {
            "status": "safe_stopped_after_accepted_buffer_commit",
            "topology_mutated": True,
            "refresh_required": True,
            "refresh_mode": "topo",
            "rebuild_mode": "topo",
            "continuation_policy": "rebuild_placer_outer_loop",
            "accepted_action_count": 1,
            "commit": {
                "status": "accepted",
                "accepted_action_count": 1,
                "artifact_path": str(commit_artifact_path),
            },
        }
    }
    joint_flow_artifact_path = tmp_path / "unit_design_joint_flow_summary.json"
    joint_flow_artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "summary": {
                    "buffer_commit_trace": [
                        {
                            "commit": {
                                "status": "accepted",
                                "accepted_action_count": 1,
                            }
                        }
                    ],
                    "post_commit_control_result": {
                        "status": "safe_stopped_after_accepted_buffer_commit"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    tcl_calls = []
    event_order = []

    class FakeBridge:
        def eval_tcl_string(self, cmd):
            event_order.append(("tcl", cmd))
            tcl_calls.append(cmd)
            return f"output:{cmd}"

    class FakePlaceDB:
        topology_generation = 0

        def __init__(self):
            self.openroad_bridge = FakeBridge()

        def refresh_from_openroad_bridge(self, refresh_mode="topo", rebuild_mode="topo"):
            event_order.append(("refresh", refresh_mode, rebuild_mode))
            self.topology_generation += 1
            return {
                "status": "ok",
                "requested_refresh_mode": refresh_mode,
                "effective_refresh_mode": refresh_mode,
                "rebuild": {
                    "status": "ok",
                    "requested_rebuild_mode": rebuild_mode,
                    "effective_rebuild_mode": rebuild_mode,
                    "topology_generation": self.topology_generation,
                    "pre_counts": {"nodes": 2, "pins": 3, "nets": 1},
                    "post_counts": {"nodes": 3, "pins": 5, "nets": 2},
                },
            }

    constructed = []

    class FakeNonLinearPlace:
        def __init__(self, params_arg, placedb_arg, timer):
            constructed.append(self)

        def run_sta_only(self, params_arg, placedb_arg):
            return 1.0, 2.0, {"wns": -3.0, "tns": -4.0}

    old_placer = SimpleNamespace(
        last_joint_flow_artifact_path=str(joint_flow_artifact_path)
    )
    engine.placer = old_placer
    engine.placedb = FakePlaceDB()
    engine._physical_mutation_backend = lambda: _refresh_backend(engine.placedb)
    monkeypatch.setattr(
        "dreamplace.Placer.NonLinearPlace.NonLinearPlace",
        FakeNonLinearPlace,
    )

    result = engine._continue_after_joint_buffer_commit(
        engine._post_commit_control_result()
    )

    assert result["status"] == "ok"
    assert result["openroad_eco_status"] == "ok"
    assert result["openroad_eco_mode"] == "resynth_once"
    assert event_order[-1][0] != "tcl"
    refresh_index = next(i for i, event in enumerate(event_order) if event[0] == "refresh")
    assert all(event[0] == "tcl" for event in event_order[:refresh_index])
    assert "update_timing" in tcl_calls[0]
    assert any(call == "resynth" for call in tcl_calls)
    assert tcl_calls[-1] == "report_tns -max"
    eco_summary_path = tmp_path / "unit_design_joint_post_commit_openroad_eco_summary.json"
    assert eco_summary_path.exists()
    eco_summary = json.loads(eco_summary_path.read_text(encoding="utf-8"))
    assert eco_summary["artifact_scope"] == "joint_post_commit_openroad_eco"
    assert eco_summary["status"] == "ok"
    assert eco_summary["mode"] == "resynth_once"
    assert [entry["label"] for entry in eco_summary["commands"]] == [
        "before_update_timing",
        "before_report_checks_json",
        "before_report_worst_slack",
        "before_report_tns",
        "resynth",
        "after_update_timing",
        "after_report_checks_json",
        "after_report_worst_slack",
        "after_report_tns",
    ]
    refresh_summary = json.loads(
        (tmp_path / "unit_design_buffer_commit_pydb_refresh_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert refresh_summary["openroad_eco_summary_path"] == str(eco_summary_path)
    joint_flow_artifact = json.loads(joint_flow_artifact_path.read_text(encoding="utf-8"))
    assert joint_flow_artifact["summary"][
        "joint_post_commit_openroad_eco_summary_path"
    ] == str(eco_summary_path)
    commit_artifact = json.loads(commit_artifact_path.read_text(encoding="utf-8"))
    assert commit_artifact["joint_post_commit_openroad_eco_summary_path"] == str(
        eco_summary_path
    )


def test_joint_staged_smoke_runs_loop_and_final_eco(monkeypatch, tmp_path):
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_quality_profile="staged_smoke_v1",
        joint_staged_smoke_outer_iterations=1,
        placement_sizing_mode="joint",
        continuous_size_dynamics_mode="discrete_gradient_topk",
        diff_timing_driven_placement=1,
        differentiable_timing_obj=1,
        with_sta=1,
        legalize_flag=1,
        detailed_place_flag=1,
        random_center_init_flag=1,
        global_place_stages=[{"iteration": 2, "optimizer": "adam"}],
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = params
    engine.rsmt = float("inf")
    engine.hpwl = float("inf")
    engine.congestion = float("inf")
    engine.density = float("inf")
    engine.metrics = None
    engine.placer = None
    engine.placedb = SimpleNamespace(
        num_movable_nodes=1,
        num_nodes=2,
        openroad_bridge=SimpleNamespace(),
    )
    applied_positions = []
    tcl_calls = []
    write_back_calls = []
    bridge_write_def_calls = []
    stage_modes = []
    setup_calls = []
    refresh_calls = []

    def fake_bridge_write_def(def_file):
        bridge_write_def_calls.append(def_file)
        with open(def_file, "w", encoding="utf-8") as fp:
            fp.write("VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n")

    engine.placedb.openroad_bridge.write_def = fake_bridge_write_def

    def fake_setup_placedb():
        setup_calls.append("setup")

    def fake_apply(params_arg, node_x, node_y):
        applied_positions.append(
            {
                "flow_kind": params_arg.flow_kind,
                "mode": params_arg.placement_sizing_mode,
                "x": float(node_x[0]),
                "y": float(node_y[0]),
            }
        )

    def fake_place():
        stage_modes.append(
            {
                "flow_kind": engine.params.flow_kind,
                "mode": engine.params.placement_sizing_mode,
                "legalize_flag": getattr(engine.params, "legalize_flag", None),
                "detailed_place_flag": getattr(engine.params, "detailed_place_flag", None),
                "iteration": engine.params.global_place_stages[0].get("iteration"),
                "optimizer": engine.params.global_place_stages[0].get("optimizer"),
                "timing_placement_carrier": getattr(
                    engine.params,
                    "timing_placement_carrier",
                    None,
                ),
                "timing_topology_enable_overflow_threshold": getattr(
                    engine.params,
                    "timing_topology_enable_overflow_threshold",
                    None,
                ),
                "discrete_gradient_topk_ranking_mode": getattr(
                    engine.params,
                    "discrete_gradient_topk_ranking_mode",
                    None,
                ),
                "discrete_gradient_topk_up_percent": getattr(
                    engine.params,
                    "discrete_gradient_topk_up_percent",
                    None,
                ),
                "discrete_gradient_topk_down_percent": getattr(
                    engine.params,
                    "discrete_gradient_topk_down_percent",
                    None,
                ),
                "early_stop_restore_best": getattr(
                    engine.params,
                    "early_stop_restore_best",
                    None,
                ),
                "buffering_commit_enabled": getattr(
                    engine.params,
                    "buffering_commit_enabled",
                    None,
                ),
                "enable_fillers": getattr(engine.params, "enable_fillers", None),
            }
        )
        offset = float(len(stage_modes))
        engine.placer = SimpleNamespace(
            pos=[torch.tensor([offset, 99.0, offset + 10.0, 199.0])]
        )
        engine.rsmt = offset
        engine.hpwl = offset + 0.5
        engine.metrics = {
            "objective": [offset],
            "overflow": [0.2],
            "density": [0.7],
            "wns": -offset,
            "tns": -10.0 * offset,
        }

    def fake_openroad_tcl(command):
        tcl_calls.append(command)
        if command == "worst_slack -max":
            return "-0.111"
        if command == "total_negative_slack -max":
            return "-44.4"
        if command == "report_worst_slack -max":
            return "-0.123"
        if command == "report_tns -max":
            return "-45.6"
        return f"ok:{command}"

    def fake_write_back(def_file):
        write_back_calls.append(def_file)
        with open(def_file, "w", encoding="utf-8") as fp:
            fp.write("VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n")

    engine.setup_placedb = fake_setup_placedb
    engine.placedb.apply = fake_apply
    def fake_refresh_from_openroad_bridge(refresh_mode="topo", rebuild_mode="topo"):
        refresh_calls.append((refresh_mode, rebuild_mode))
        return {
            "status": "ok",
            "requested_refresh_mode": refresh_mode,
            "effective_refresh_mode": refresh_mode,
            "rebuild": {
                "status": "ok",
                "requested_rebuild_mode": rebuild_mode,
                "effective_rebuild_mode": rebuild_mode,
            },
        }

    engine.placedb.refresh_from_openroad_bridge = fake_refresh_from_openroad_bridge
    engine._openroad_tcl = fake_openroad_tcl
    engine.write_back = fake_write_back
    engine.place = fake_place

    result = engine.run()

    assert setup_calls == ["setup"]
    assert [stage["mode"] for stage in stage_modes] == [
        "place_only",
        "size_only",
        "place_only",
        "place_only",
    ]
    assert [stage["flow_kind"] for stage in stage_modes] == [
        "placement",
        "sizing",
        "placement",
        "placement",
    ]
    assert stage_modes[1]["legalize_flag"] == 0
    assert stage_modes[1]["detailed_place_flag"] == 0
    assert [stage["optimizer"] for stage in stage_modes] == [
        "nesterov",
        "adam",
        "nesterov",
        "nesterov",
    ]
    assert [stage["iteration"] for stage in stage_modes] == [
        3000,
        50,
        3000,
        3000,
    ]
    assert stage_modes[0]["timing_placement_carrier"] == "gradient_net_weight"
    assert stage_modes[0]["timing_topology_enable_overflow_threshold"] == 0.35
    assert stage_modes[0]["enable_fillers"] == 0
    assert stage_modes[2]["enable_fillers"] == 0
    assert stage_modes[3]["enable_fillers"] == 0
    assert stage_modes[1]["discrete_gradient_topk_ranking_mode"] == "real_size_taylor"
    assert stage_modes[1]["discrete_gradient_topk_up_percent"] == 30.0
    assert stage_modes[1]["discrete_gradient_topk_down_percent"] == 0.0
    assert stage_modes[1]["early_stop_restore_best"] is True
    assert params.flow_kind == "joint"
    assert params.placement_sizing_mode == "joint"
    assert params.legalize_flag == 1
    assert params.detailed_place_flag == 1
    assert params.global_place_stages[0]["optimizer"] == "adam"
    assert params.global_place_stages[0]["iteration"] == 2
    assert len(applied_positions) == 4
    assert refresh_calls == [("topo", "topo")]
    assert "repair_design" in tcl_calls
    assert "repair_timing -setup" in tcl_calls
    repair_design_index = tcl_calls.index("repair_design")
    repair_timing_indices = [
        index for index, command in enumerate(tcl_calls)
        if command == "repair_timing -setup"
    ]
    assert len(repair_timing_indices) == 2
    assert repair_timing_indices[0] < repair_design_index
    assert repair_design_index < repair_timing_indices[-1]
    summary = result["joint_staged_smoke"]
    assert summary["stage_count"] == 5
    assert [stage["stage_name"] for stage in summary["stages"]] == [
        "initial_tdp_place",
        "outer_000_fixed_position_sizing",
        "outer_000_after_sizing_tdp_return",
        "outer_000_buffering",
        "outer_000_tdp_return",
    ]
    assert [stage["stage_role"] for stage in summary["stages"]] == [
        "timing_driven_placement",
        "fixed_position_sizing",
        "timing_driven_placement_after_sizing",
        "openroad_repair_timing_buffering",
        "timing_driven_placement_after_buffering",
    ]
    buffering_stage = summary["stages"][3]
    assert buffering_stage["topology_refreshed"] is True
    assert buffering_stage["refresh_mode"] == "topo"
    assert buffering_stage["rebuild_mode"] == "topo"
    assert any(
        command["command"] == "repair_timing -setup"
        for command in buffering_stage["openroad_commands"]
    )
    assert summary["finalization"]["status"] == "ok"
    assert summary["finalization"]["timing_summary"]["pre_eco"] == {
        "wns": -0.123,
        "tns": -45.6,
    }
    assert summary["finalization"]["timing_summary"]["post_repair_timing"] == {
        "wns": -0.111,
        "tns": -44.4,
    }
    assert summary["finalization"]["direct_metrics"][
        "post_repair_timing_wns"
    ]["command"] == "worst_slack -max"
    assert summary["finalization"]["direct_metrics"][
        "post_repair_timing_tns"
    ]["command"] == "total_negative_slack -max"
    final_def = summary["finalization"]["final_def"]
    assert final_def["path"] == str(tmp_path / "unit_design_staged_smoke_v1_final.def")
    assert final_def["writer"] == "openroad_bridge.write_def"
    assert final_def["sha256"] == hashlib.sha256(
        b"VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n"
    ).hexdigest()
    assert write_back_calls == []
    assert bridge_write_def_calls == [final_def["path"]]
    summary_path = tmp_path / "unit_design_joint_staged_smoke_summary.json"
    assert summary_path.exists()
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["finalization"]["final_def"]["path"] == final_def["path"]


def test_segment_joint_terminal_commit_legalizes_refreshes_and_verifies_opensta(
    monkeypatch,
    tmp_path,
):
    commit_artifact_path = tmp_path / "buffering_coordinate_commit_results.json"
    commit_artifact_path.write_text(
        json.dumps(
            {
                "artifact": "buffering_coordinate_commit_results",
                "results": [
                    {
                        "accepted": True,
                        "action_id": 7,
                        "segment_id": 11,
                        "inserted_buffer_name": "buffer_coord_buf_7",
                        "candidate_location_x_dbu": 1000.0,
                        "candidate_location_y_dbu": 2000.0,
                        "candidate_location_x_dbu_analytical": 1000.25,
                        "candidate_location_y_dbu_analytical": 1999.75,
                        "coordinate_source": "final_live_geometry",
                        "placement_snapshot_identity": "sha256:final",
                        "topology_generation": 4,
                        "frozen_topology_generation": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.def_paths = []
            self.command_output = ""

        def eval_tcl_string(self, command):
            self.commands.append(command)
            if command.startswith("set __autodmp_command_code [catch {"):
                self.command_output = ""
                return "0"
            if command == "set __autodmp_command_output":
                return self.command_output
            return "ok"

        def query_diff_guided_batch_timing_metrics(self):
            return {
                "status": "ok",
                "wns": -8.0,
                "tns": -70.0,
                "timing_refresh": {
                    "status": "ok",
                    "method": "sta::dbSta::updateTiming(false)",
                    "elapsed_ms": 1.0,
                },
            }

        def write_def(self, path):
            self.def_paths.append(path)
            with open(path, "w", encoding="utf-8") as fp:
                fp.write("VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n")

    class FakePlaceDB:
        def __init__(self):
            self.openroad_bridge = FakeBridge()
            self.total_movable_node_area = 10_000_000.0
            self.node_names = [b"base"]
            self.node_x = torch.tensor([0.0]).numpy()
            self.node_y = torch.tensor([0.0]).numpy()
            self.dbu = 1000.0
            self.topology_generation = 0
            self.refresh_calls = []

        def unscale_pl_positions(self, node_x, node_y):
            return node_x, node_y

        def refresh_from_openroad_bridge(self, refresh_mode="topo", rebuild_mode="topo"):
            self.refresh_calls.append((refresh_mode, rebuild_mode))
            self.node_names = np.asarray([b"base", b"buffer_coord_buf_7"])
            self.node_x = torch.tensor([0.0, 1002.0]).numpy()
            self.node_y = torch.tensor([0.0, 1998.0]).numpy()
            self.total_movable_node_area = 12_000_000.0
            self.topology_generation = 1
            return {
                "status": "ok",
                "requested_refresh_mode": refresh_mode,
                "effective_refresh_mode": refresh_mode,
                "rebuild": {
                    "status": "ok",
                    "requested_rebuild_mode": rebuild_mode,
                    "effective_rebuild_mode": rebuild_mode,
                    "topology_generation": 1,
                },
            }

    replay_params_seen = []

    class FakeReplayBridge:
        @staticmethod
        def query_diff_guided_batch_timing_metrics():
            return {
                "status": "ok",
                "wns": -7.75,
                "tns": -69.5,
                "timing_refresh": {
                    "status": "ok",
                    "method": "sta::dbSta::updateTiming(false)",
                },
            }

    def fake_replay_read(replay_params):
        replay_params_seen.append(replay_params)
        return FakeReplayBridge()

    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    monkeypatch.setattr(
        placeio_openroad.PlaceIOFunction,
        "read",
        staticmethod(fake_replay_read),
    )

    class FakeMutationBackend:
        def __init__(self, placedb):
            self.placedb = placedb

        def refresh(self, *, refresh_mode, rebuild_mode):
            return self.placedb.refresh_from_openroad_bridge(
                refresh_mode=refresh_mode,
                rebuild_mode=rebuild_mode,
            )

    params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind="joint",
        joint_quality_profile="segment_count_direct_joint_v1",
        joint_post_commit_continue=False,
        scale_factor=1.0,
        design_inputs={
            "tech_lef": ["tech.lef"],
            "lef": ["cells.lef"],
            "lib": ["cells.lib"],
            "def": "source.def",
            "verilog": "source.v",
            "sdc": "source.sdc",
            "rc_tcl": "setRC.tcl",
        },
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = params
    engine.placedb = FakePlaceDB()
    engine.last_physical_mutation_backend = FakeMutationBackend(engine.placedb)
    engine.metrics = {"overflow": [0.12], "density": [0.93]}
    control_result = {
        "status": "safe_stopped_after_accepted_buffer_commit",
        "topology_mutated": True,
        "refresh_required": True,
        "refresh_mode": "topo",
        "rebuild_mode": "topo",
        "continuation_policy": "rebuild_placer_outer_loop",
            "commit_request": {
                "action_count": 1,
                "sizing_action_count": 0,
                "projection_metadata": {
                "projection_provenance": {
                    "coordinate_source": "final_live_geometry",
                    "placement_snapshot_identity": "sha256:final",
                    "topology_generation": 4,
                    "frozen_topology_generation": 4,
                }
            }
        },
        "commit": {
            "status": "accepted",
            "accepted_action_count": 1,
            "before_wns": -10.0,
                "before_tns": -100.0,
                "artifact_path": str(commit_artifact_path),
                "physical_action_trace": [
                    {"stage": "coordinate_sync", "status": "ok"},
                    {"stage": "insert_buffers", "status": "accepted"},
                ],
            },
    }

    result = engine._continue_after_joint_buffer_commit(control_result)

    assert result["status"] == "ok"
    assert result["continuation_policy"] == "terminal_verification_only"
    assert result["wns"] == -8.0
    assert result["tns"] == -70.0
    assert result["delta_wns"] == 2.0
    assert result["delta_tns"] == 30.0
    assert result["added_area_um2"] == 2.0
    legalize_command = engine._full_core_detailed_placement_command()
    assert engine.placedb.openroad_bridge.commands[::2] == [
        "set __autodmp_command_code [catch {" + legalize_command + "} "
        "__autodmp_command_output]; set __autodmp_command_code",
        "set __autodmp_command_code [catch {check_placement -verbose} "
        "__autodmp_command_output]; set __autodmp_command_code",
        "set __autodmp_command_code [catch {estimate_parasitics -placement} "
        "__autodmp_command_output]; set __autodmp_command_code",
    ]
    assert engine.placedb.openroad_bridge.commands[1::2] == [
        "set __autodmp_command_output",
    ] * 3
    assert engine.placedb.refresh_calls == [("topo", "topo")]
    summary = json.loads(
        (tmp_path / "unit_design_segment_joint_committed_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["qor_status"] == "pass"
    assert summary["detailed_placement_search_window"] == "full_core"
    assert summary["opensta_update_status"] == "ok"
    assert summary["opensta_update_method"] == "sta::dbSta::updateTiming(false)"
    assert summary["commands"][-1]["label"] == "timing_refresh"
    assert summary["commands"][-1]["status"] == "ok"
    assert summary["def_only_opensta_verification_status"] == "ok"
    assert summary["def_only_opensta_after"] == {"wns": -7.75, "tns": -69.5}
    assert summary["def_only_vs_in_process_gap"] == {
        "wns": 0.25,
        "tns": 0.5,
    }
    replay_path = tmp_path / "unit_design_segment_joint_def_only_opensta_replay.json"
    assert replay_path.exists()
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["load_contract"]["design_carrier"] == "final_def"
    assert replay["load_contract"]["fresh_openroad_bridge"] is True
    assert replay["load_contract"]["verilog_read"] is False
    assert replay["load_contract"]["source_def_path"] == "source.def"
    assert len(replay_params_seen) == 1
    assert replay_params_seen[0].design_inputs["def"] == str(
        tmp_path / "unit_design_segment_joint_final.def"
    )
    assert "verilog" not in replay_params_seen[0].design_inputs
    assert replay_params_seen[0].verilog_input == ""
    assert summary["missing_legalized_coordinate_count"] == 0
    coordinate = summary["buffer_coordinates"][0]
    assert coordinate["analytical_x_dbu"] == 1000.25
    assert coordinate["legalized_x_dbu"] == 1002.0
    assert coordinate["displacement_x_dbu"] == 1.75
    assert coordinate["displacement_y_dbu"] == -1.75
    assert (tmp_path / "unit_design_segment_joint_final.def").exists()
    updated_commit = json.loads(commit_artifact_path.read_text(encoding="utf-8"))
    assert updated_commit["terminal_delta_tns"] == 30.0
    assert updated_commit["terminal_def_only_opensta_after"] == {
        "wns": -7.75,
        "tns": -69.5,
    }


def test_segment_joint_terminal_commit_rejects_check_placement_error(tmp_path):
    class FailingPlacementBridge:
        def __init__(self):
            self.commands = []
            self.command_output = ""

        def eval_tcl_string(self, command):
            self.commands.append(command)
            if command.startswith("set __autodmp_command_code [catch {"):
                if "check_placement -verbose" in command:
                    self.command_output = "DPL-0033"
                    return "1"
                self.command_output = ""
                return "0"
            if command == "set __autodmp_command_output":
                return self.command_output
            raise AssertionError(f"unexpected command: {command}")

    bridge = FailingPlacementBridge()
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
    )
    engine.placedb = SimpleNamespace(openroad_bridge=bridge)
    control_result = {
        "commit_request": {
            "projection_metadata": {
                "projection_provenance": {
                    "coordinate_source": "final_live_geometry",
                }
            }
        }
    }

    with pytest.raises(RuntimeError, match="check_placement.*DPL-0033"):
        engine._finalize_segment_joint_committed_state(control_result)

    artifact = json.loads(
        (tmp_path / "unit_design_segment_joint_committed_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["status"] == "failed"
    assert artifact["legalization_status"] == "failed"
    assert artifact["parasitic_refresh_status"] == "not_run"
    assert artifact["commands"][-1]["label"] == "check_placement"
    assert artifact["commands"][-1]["status"] == "failed"
    assert artifact["commands"][-1]["tcl_return_code"] == 1
    assert artifact["commands"][-1]["output"] == "DPL-0033"
    assert not any("estimate_parasitics" in command for command in bridge.commands)


def test_segment_joint_def_only_opensta_replay_rejects_failed_metrics(
    monkeypatch,
    tmp_path,
):
    final_def = tmp_path / "unit_design_segment_joint_final.def"
    final_def.write_text(
        "VERSION 5.8 ;\nDESIGN unit_design ;\nEND DESIGN\n",
        encoding="utf-8",
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        design_inputs={
            "tech_lef": ["tech.lef"],
            "lef": ["cells.lef"],
            "lib": ["cells.lib"],
            "def": "source.def",
            "sdc": "source.sdc",
            "rc_tcl": "setRC.tcl",
        },
    )

    class FailedReplayBridge:
        @staticmethod
        def query_diff_guided_batch_timing_metrics():
            return {"status": "failed", "error": "replay failed"}

    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    monkeypatch.setattr(
        placeio_openroad.PlaceIOFunction,
        "read",
        staticmethod(lambda _params: FailedReplayBridge()),
    )

    with pytest.raises(RuntimeError, match="fresh DEF-only OpenSTA replay failed"):
        engine._run_segment_joint_def_only_opensta_replay(
            str(final_def),
            in_process_metrics={"wns": -8.0, "tns": -70.0},
        )
