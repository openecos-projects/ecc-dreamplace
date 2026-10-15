from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from dreamplace.ops.buffer_insertion.buffering_config import (
    build_buffering_config_from_params,
)
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    build_discrete_candidate_state,
)
from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state
from dreamplace.ops.placeio_common.backend_contract import openroad_backend_caps
from dreamplace.ops.placeio_common.physical_mutation import (
    physical_mutation_backend_for,
)


def _config(mode="candidate"):
    return build_buffering_config_from_params(
        SimpleNamespace(buffering_mode=mode),
        flow_kind="buffering",
    )


def _candidate_state():
    return build_buffer_optimization_state(
        [
            {
                "candidate_id": 0,
                "tree_node_id": 10,
                "net_id": 3,
                "x_dbu": 100,
                "y_dbu": 200,
                "buffer_main_type_index": 1,
            }
        ],
        buffer_main_type_index=1,
        legal_buffer_count=4,
    )


def _discrete_candidate_state():
    return build_discrete_candidate_state(
        [
            {
                "candidate_id": 0,
                "candidate_node_id": 10,
                "net_id": 3,
                "x_dbu": 100,
                "y_dbu": 200,
                "buffer_main_type_index": 1,
            },
            {
                "candidate_id": 1,
                "candidate_node_id": 11,
                "net_id": 4,
                "x_dbu": 300,
                "y_dbu": 400,
                "buffer_main_type_index": 1,
            },
        ],
        buffer_main_type_index=1,
        legal_buffer_count=4,
        fixed_bsu_index=2,
    )


def _commit_action(action_id=0):
    return {
        "action_id": int(action_id),
        "action_kind": "buffer_insert",
        "net_name": "n1",
        "buffer_main_type_index": 1,
        "bsu": 0,
        "buffer_master_id": 10,
        "buffer_master_name": "BUF_X1",
        "candidate_location_x_dbu": 100.0,
        "candidate_location_y_dbu": 200.0,
        "driver_pin_name": "drv/Y",
        "load_pin_name": "sink/A",
        "downstream_pin_names": ["sink/A"],
    }


def _segment_net():
    return {
        "net_id": 10,
        "net_name": "n10",
        "driver_pin_id": 0,
        "coordinates": {0: (0, 0), 1: (100, 0)},
        "rc_tree": {
            "root_node_id": 0,
            "children_by_node": {0: [1], 1: []},
            "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}},
            "node_cap": {0: 0.0, 1: 1.0},
            "sink_nodes": [1],
        },
    }


def _branch_segment_net():
    return {
        "net_id": 11,
        "net_name": "n11",
        "driver_pin_id": 0,
        "driver_pin_name": "drv/Y",
        "pin_name_by_id": {
            0: "drv/Y",
            1: "sink_weak/A",
            2: "sink_strong/A",
        },
        "sink_slack_by_pin": {
            1: -1.0,
            2: -10.0,
        },
        "npath_by_pin": {
            1: 1,
            2: 10,
        },
        "coordinates": {0: (0, 0), 1: (100, 0), 2: (200, 0)},
        "rc_tree": {
            "root_node_id": 0,
            "children_by_node": {0: [1, 2], 1: [], 2: []},
            "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}, (0, 2): {"r": 1.0, "c": 0.0}},
            "node_cap": {0: 0.0, 1: 1.0, 2: 1.0},
            "sink_nodes": [1, 2],
        },
    }


def _segment_state(initial_z=1.25, fixed_bsu_index=None):
    return build_segment_count_state(
        [_segment_net()],
        buffer_main_type_index=7,
        legal_buffer_count=4,
        max_repeater_count=3,
        initial_z=initial_z,
        initial_bsu_index=(
            1.0 if fixed_bsu_index is None else float(fixed_bsu_index)
        ),
        fixed_bsu_index=fixed_bsu_index,
    )


def _multi_segment_state():
    state = _segment_state(initial_z=0.1)
    with torch.no_grad():
        state.z_param.copy_(torch.tensor([0.49]))
        state.bsu_index_param.copy_(torch.tensor([1.51]))
    return state


def _segment_payload():
    return {
        "metadata": {"state_kind": "segment_count"},
        "nets": [_segment_net()],
        "per_size_input_cap": torch.tensor([[0.3, 0.5, 0.7, 0.9]]),
        "per_size_delay": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "per_size_output_slew": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    }


def _branch_segment_payload_and_state():
    net = _branch_segment_net()
    state = build_segment_count_state(
        [net],
        buffer_main_type_index=7,
        legal_buffer_count=4,
        max_repeater_count=3,
        initial_z=1.0,
        initial_bsu_index=1.0,
    )
    payload = {
        "metadata": {"state_kind": "segment_count"},
        "nets": [net],
        "per_size_input_cap": torch.tensor([[0.3, 0.5, 0.7, 0.9], [0.3, 0.5, 0.7, 0.9]]),
        "per_size_delay": torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
        "per_size_output_slew": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
    }
    return payload, state


def test_lane_constructs_from_buffering_config_without_design_inputs():
    lane = BufferingOptimizationLane(_config())

    assert lane.summarize()["status"] == "initialized"
    assert lane.optimizer_parameters() == []


def test_lane_module_keeps_physical_backend_out_of_runtime_refresh():
    source = (
        Path(__file__).resolve().parents[3]
        / "dreamplace/ops/buffer_insertion/buffering_lane.py"
    ).read_text(encoding="utf-8")

    refresh_body = source.split("def refresh_runtime_state", 1)[1].split(
        "def build_commit_request",
        1,
    )[0]
    assert "commit_coordinate_buffer_actions" not in refresh_body
    assert "run_coordinate_buffer_insert" not in refresh_body
    assert "def execute_commit_request" not in source
    assert "def commit_if_enabled" not in source
    assert "def commit_final_projection_if_enabled" not in source


def test_lane_attaches_buffer_state_and_payload_to_data_collections():
    state = _candidate_state()
    payload = {"metadata": {"source": "unit"}}
    data = SimpleNamespace()
    lane = BufferingOptimizationLane(
        _config(),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )

    lane.attach_state(data)

    assert data.buffer_optimization_state is state
    assert data.buffer_relaxed_timing_payload is payload
    assert lane.summarize()["state_attached"]


def test_lane_optimizer_parameters_returns_candidate_trainable_tensors():
    state = _candidate_state()
    lane = BufferingOptimizationLane(_config(), buffer_optimization_state=state)

    params = lane.optimizer_parameters()

    assert params == [state.bu_logits, state.bsu_index_param]


def test_lane_segment_mode_exposes_z_and_bsu_params():
    state = _segment_state()
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )

    params = lane.optimizer_parameters()

    assert params == [state.z_param, state.bsu_index_param]
    assert lane.summarize()["state_kind"] == "segment_count"


def test_lane_freezes_segment_bsu_when_profile_selects_one_fixed_size():
    state = _segment_state(fixed_bsu_index=2)
    config = build_buffering_config_from_params(
        SimpleNamespace(buffering_mode="segment", buffering_fixed_bsu_index=2),
        flow_kind="buffering",
    )
    lane = BufferingOptimizationLane(
        config,
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )

    assert lane.optimizer_parameters() == [state.z_param]


def test_lane_registers_segment_state_trainables_on_model():
    state = _segment_state()
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )
    model = SimpleNamespace()

    def register_buffer_optimization_state(registered_state):
        model.registered_state = registered_state

    model.register_buffer_optimization_state = register_buffer_optimization_state

    lane.register_model_parameters(model)

    assert model.registered_state is state


def test_lane_after_step_projects_segment_params_back_to_legal_domain():
    state = _segment_state(initial_z=1.0)
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )
    with torch.no_grad():
        state.z_param.fill_(-2.0)
        state.bsu_index_param.fill_(99.0)
    metrics = {}

    lane.after_step(0, torch.tensor(0.0), metrics)

    assert torch.allclose(state.z_param, torch.zeros_like(state.z_param))
    assert torch.allclose(
        state.bsu_index_param,
        torch.full_like(state.bsu_index_param, float(state.legal_buffer_count - 1)),
    )
    assert metrics["projected_after_step"]
    assert metrics["z_param_min"] == 0.0
    assert metrics["z_param_max"] == 0.0
    assert metrics["bsu_param_min"] == float(state.legal_buffer_count - 1)
    assert metrics["bsu_param_max"] == float(state.legal_buffer_count - 1)


def test_lane_periodic_integer_projection_rounds_z_without_bsu_by_default():
    state = _multi_segment_state()
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )

    result = lane.project_segment_count_optimizer_state_to_integer(
        project_bsu=False,
        reset_optimizer_state=False,
        iteration=4,
    )

    assert result["status"] == "ok"
    assert result["type"] == "periodic_integer_projection"
    assert result["iteration"] == 4
    assert result["z_changed_count"] == 1
    assert result["z_nonzero_count_before"] == 1
    assert result["z_nonzero_count_after"] == 0
    assert result["z_ge_0p5_count_before"] == 0
    assert result["z_ge_0p5_count_after"] == 0
    assert torch.allclose(state.z_param, torch.zeros_like(state.z_param))
    assert torch.allclose(state.bsu_index_param, torch.tensor([1.51]))
    summary = lane.summarize()
    assert summary["periodic_integer_projection_count"] == 1
    assert summary["periodic_integer_projection_trace"][0]["project_bsu"] is False


def test_lane_periodic_integer_projection_can_project_bsu_and_reset_adam_state():
    state = _multi_segment_state()
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=_segment_payload(),
    )
    optimizer = torch.optim.Adam([state.z_param, state.bsu_index_param], lr=0.1)
    loss = state.z_param.sum() + state.bsu_index_param.sum()
    loss.backward()
    optimizer.step()
    assert torch.any(optimizer.state[state.z_param]["exp_avg"] != 0)
    assert torch.any(optimizer.state[state.bsu_index_param]["exp_avg"] != 0)
    with torch.no_grad():
        state.z_param.copy_(torch.tensor([1.51]))
        state.bsu_index_param.copy_(torch.tensor([2.51]))

    result = lane.project_segment_count_optimizer_state_to_integer(
        project_bsu=True,
        reset_optimizer_state=True,
        optimizer=optimizer,
        iteration=9,
    )

    assert torch.allclose(state.z_param, torch.tensor([2.0]))
    assert torch.allclose(state.bsu_index_param, torch.tensor([3.0]))
    assert result["z_changed_count"] == 1
    assert result["bsu_changed_count"] == 1
    assert result["optimizer_state_reset"]["status"] == "ok"
    assert result["optimizer_state_reset"]["reset_param_count"] == 2
    assert torch.all(optimizer.state[state.z_param]["exp_avg"] == 0)
    assert torch.all(optimizer.state[state.z_param]["exp_avg_sq"] == 0)
    assert torch.all(optimizer.state[state.bsu_index_param]["exp_avg"] == 0)
    assert torch.all(optimizer.state[state.bsu_index_param]["exp_avg_sq"] == 0)


def test_lane_after_step_projects_candidate_bsu_but_keeps_bu_logits_free():
    state = _candidate_state()
    lane = BufferingOptimizationLane(_config("candidate"), buffer_optimization_state=state)
    with torch.no_grad():
        state.bu_logits.fill_(42.0)
        state.bsu_index_param.fill_(-5.0)
    metrics = {}

    lane.after_step(0, torch.tensor(0.0), metrics)

    assert torch.allclose(state.bu_logits, torch.full_like(state.bu_logits, 42.0))
    assert torch.allclose(state.bsu_index_param, torch.zeros_like(state.bsu_index_param))
    assert metrics["projected_after_step"]
    assert metrics["bsu_param_min"] == 0.0
    assert metrics["bsu_param_max"] == 0.0


def test_lane_attaches_segment_count_state_fields_to_data_collections():
    state = _segment_state()
    payload = _segment_payload()
    data = SimpleNamespace()
    lane = BufferingOptimizationLane(
        _config("segment"),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )

    lane.attach_state(data)

    assert data.buffer_optimization_state is state
    assert data.buffer_segment_count_state is state
    assert data.buffer_segment_count_nets == payload["nets"]
    assert data.buffer_segment_count_per_size_input_cap is payload["per_size_input_cap"]
    assert data.buffer_segment_count_per_size_delay is payload["per_size_delay"]
    assert data.buffer_segment_count_per_size_output_slew is payload["per_size_output_slew"]
    assert data.buffer_relaxed_timing_payload is payload
    assert lane.summarize()["state_kind"] == "segment_count"


def test_lane_records_placeobj_timing_objective_boundary_on_prepare():
    params = SimpleNamespace()
    lane = BufferingOptimizationLane(
        _config(),
        buffer_optimization_state=_candidate_state(),
        buffer_relaxed_timing_payload={"metadata": {}},
    )

    lane.prepare(params=params)
    summary = lane.summarize()

    assert summary["objective_boundary"] == "PlaceObj.timing_obj"
    assert summary["timing_integration_mode"] == "dynamic_net_provider"
    assert params.enable_relaxed_buffer_timing
    assert params.relaxed_buffer_timing_integration_mode == "dynamic_net_provider"


def test_lane_prepare_builds_state_from_model_attached_buffering_inputs():
    nets = [
        {
            "net_id": 10,
            "driver_pin_id": 0,
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}},
                "node_cap": {0: 0.0, 1: 1.0},
                "sink_nodes": [1],
            },
        }
    ]
    candidates = [
        {
            "candidate_id": 100,
            "net_id": 10,
            "candidate_node_id": 1,
            "x_dbu": 100,
            "y_dbu": 0,
            "buffer_main_type_index": 7,
        }
    ]
    data = SimpleNamespace(
        pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        buffering_nets=nets,
        buffering_candidate_rows=candidates,
        buffering_buffer_library={
            0: {"input_cap": 0.4, "delay": 2.0, "output_slew": 0.3},
            1: {"input_cap": 0.8, "delay": 3.0, "output_slew": 0.2},
        },
    )
    model = SimpleNamespace(data_collections=data)
    lane = BufferingOptimizationLane(_config())

    lane.prepare(model=model, params=SimpleNamespace())
    lane.attach_state(data)

    assert lane.buffer_optimization_state is not None
    assert lane.buffer_relaxed_timing_payload is not None
    assert data.buffer_optimization_state is lane.buffer_optimization_state
    assert data.buffer_relaxed_timing_payload is lane.buffer_relaxed_timing_payload
    assert lane.summarize()["state_build_status"] == "ok"


def test_lane_prepare_runtime_profile_nests_segment_state_builder_profile():
    nets = [_segment_net()]
    buffer_library = {
        0: {"input_cap": 0.4, "delay": 2.0, "output_slew": 0.3},
        1: {"input_cap": 0.8, "delay": 3.0, "output_slew": 0.2},
    }
    data = SimpleNamespace(
        pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        buffering_nets=nets,
        buffering_buffer_library=buffer_library,
        buffer_main_type_index=7,
    )
    params = SimpleNamespace(
        buffering_mode="segment",
        buffering_max_repeaters_per_segment=3,
        buffering_segment_count_z_init=0.0,
        buffering_initial_bsu_index=1.0,
        buffering_runtime_profile=True,
    )
    lane = BufferingOptimizationLane(_config("segment"))

    lane.prepare(model=SimpleNamespace(data_collections=data), params=params)

    profile = lane.summarize()["lane_prepare_runtime_profile"]
    state_profile = profile["state_build_profile"]
    assert profile["total_ms"] >= profile["accounted_ms"]
    assert profile["unattributed_ms"] >= 0.0
    assert profile["state_build_ms"] >= 0.0
    assert profile["segment_capacity_build_ms"] == 0.0
    assert state_profile["segment_state_build_ms"] >= 0.0
    assert state_profile["segment_rc_tree_edge_count"] == 1


def test_lane_prepare_without_relaxed_state_does_not_enable_dynamic_provider():
    params = SimpleNamespace()
    lane = BufferingOptimizationLane(_config())

    lane.prepare(params=params)

    assert not getattr(params, "enable_relaxed_buffer_timing", False)
    assert (
        params.relaxed_buffer_timing_integration_mode == "dynamic_net_provider"
    )


def test_lane_project_and_refresh_use_structured_result():
    state = _candidate_state()
    with torch.no_grad():
        state.bu_logits.fill_(3.0)
    data = SimpleNamespace(buffer_relaxed_timing_payload={"metadata": {}})
    lane = BufferingOptimizationLane(_config(), buffer_optimization_state=state)
    lane.attach_state(data)

    projection = lane.project(top_k=1)
    refresh = lane.refresh_runtime_state(projection)

    assert projection.projected_buffer_count == 1
    assert projection.affected_nets == (3,)
    assert refresh["status"] == "ok"
    assert data.last_buffering_projection_result is projection


def test_lane_segment_projection_honors_max_selected_actions():
    payload, state = _branch_segment_payload_and_state()
    data = SimpleNamespace(buffer_relaxed_timing_payload=payload)
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_max_selected_actions=1,
            ),
            flow_kind="buffering",
        ),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    lane.attach_state(data)

    projection = lane.project()

    assert projection.projected_buffer_count == 1
    assert projection.metadata["projected_candidate_count_before_topk"] == 2
    assert projection.metadata["top_k"] == 1
    assert projection.selected_actions[0]["load_pin_name"] == "sink_strong/A"


def test_lane_segment_projection_overlays_current_pin_slack_before_ranking():
    payload, state = _branch_segment_payload_and_state()
    net = dict(payload["nets"][0])
    net["sink_slack_by_pin"] = {1: 0.0, 2: 0.0}
    payload = dict(payload)
    payload["nets"] = [net]
    with torch.no_grad():
        state.z_param.copy_(torch.tensor([1.8, 0.6]))
    data = SimpleNamespace(
        buffer_relaxed_timing_payload=payload,
        pin_slack=torch.tensor([0.0, -1.0, -10.0]),
    )
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_max_selected_actions=1,
            ),
            flow_kind="buffering",
        ),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    lane.attach_state(data)

    projection = lane.project()

    assert projection.projected_buffer_count == 1
    assert projection.selected_actions[0]["load_pin_name"] == "sink_strong/A"
    assert projection.selected_actions[0]["downstream_sink_criticality_sum"] == 100.0
    assert projection.metadata["criticality_overlay_status"] == "ok"
    assert projection.metadata["criticality_overlay_source"] == "current_pin_slack"
    assert projection.metadata["criticality_overlay_updated_sink_count"] == 2
    assert projection.metadata["selected_candidate_diagnostics"][0]["load_pin_name"] == "sink_strong/A"
    assert lane.summarize()["projection"]["criticality_overlay_status"] == "ok"


def test_lane_segment_projection_default_round_threshold_keeps_small_z_uncommitted():
    payload, state = _branch_segment_payload_and_state()
    with torch.no_grad():
        state.z_param.fill_(0.18)
    data = SimpleNamespace(buffer_relaxed_timing_payload=payload)
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="segment"),
            flow_kind="buffering",
        ),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    lane.attach_state(data)

    projection = lane.project()

    assert projection.projected_buffer_count == 0
    assert projection.metadata["min_z_to_insert"] == 0.5
    assert projection.metadata["z_value_distribution"]["ge_min_insert_count"] == 0
    assert projection.metadata["z_value_distribution"]["ge_0p5_count"] == 0


def test_lane_segment_projection_can_use_profile_min_z_threshold_for_natural_commit():
    payload, state = _branch_segment_payload_and_state()
    with torch.no_grad():
        state.z_param.fill_(0.18)
    data = SimpleNamespace(buffer_relaxed_timing_payload=payload)
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_segment_projection_min_z_to_insert=0.15,
            ),
            flow_kind="buffering",
        ),
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    lane.attach_state(data)

    projection = lane.project()
    summary = lane.summarize()

    assert projection.projected_buffer_count == 2
    assert projection.metadata["min_z_to_insert"] == 0.15
    assert projection.metadata["z_value_distribution"]["ge_min_insert_count"] == 2
    assert projection.metadata["z_value_distribution"]["ge_0p5_count"] == 0
    assert {action["projected_repeater_count"] for action in projection.selected_actions} == {1}
    assert summary["segment_projection_min_z_to_insert"] == 0.15
    assert summary["z_value_distribution"]["max"] == 0.18000000715255737


def test_lane_objective_uses_placeobj_timing_obj_with_current_pos():
    class FakeModel:
        def __init__(self):
            self.data_collections = SimpleNamespace(pos=[torch.nn.Parameter(torch.tensor(1.5))])
            self.calls = []

        def timing_obj(self, pos):
            self.calls.append(pos)
            tns = pos * -2.0
            return torch.tensor(-0.1), tns, torch.tensor(0.0), torch.tensor(0.0)

    model = FakeModel()
    lane = BufferingOptimizationLane(_config())

    loss, metrics = lane.objective(model)

    assert model.calls == [model.data_collections.pos[0]]
    assert loss.item() == torch.tensor(3.0).item()
    assert metrics["objective_boundary"] == "PlaceObj.timing_obj"
    assert metrics["tns"].item() == torch.tensor(-3.0).item()


def test_lane_default_hooks_are_noop():
    lane = BufferingOptimizationLane(_config())

    assert lane.should_project(0)
    assert lane.before_step(0, torch.tensor(0.0), {}) is None
    assert lane.after_step(0, torch.tensor(0.0), {}) is None


def test_lane_build_commit_request_is_backend_pure_and_action_stable():
    class ForbiddenBridge:
        def __getattr__(self, name):
            raise AssertionError(f"request construction must not access {name}")

    action = _commit_action()
    projection = SimpleNamespace(
        mode="candidate",
        selected_actions=(action,),
        metadata={"projection_source": "unit"},
    )
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="candidate", buffering_commit_enabled=1),
            flow_kind="buffering",
        ),
        placedb=SimpleNamespace(openroad_bridge=ForbiddenBridge()),
    )

    request = lane.build_commit_request(
        projection,
        iteration=7,
        reason="unit",
        runtime_refresh={"status": "ok"},
        effective_config_reference="effective_params.json",
    )
    equivalent_request = lane.build_commit_request(
        projection,
        iteration=7,
        reason="unit",
        runtime_refresh={"status": "ok"},
    )
    action["net_name"] = "mutated_after_request"

    assert request.action_digest == equivalent_request.action_digest
    assert request.actions[0]["net_name"] == "n1"
    assert request.iteration == 7
    assert request.runtime_refresh_summary == {"status": "ok"}
    assert request.to_summary()["effective_config_reference"] == "effective_params.json"
    assert lane.last_commit_request is not None


def test_lane_build_commit_request_rejects_tensor_payloads():
    lane = BufferingOptimizationLane(_config())
    projection = SimpleNamespace(
        mode="candidate",
        selected_actions=({"action_id": 0, "tensor": torch.tensor([1.0])},),
        metadata={},
    )

    with pytest.raises(ValueError, match="JSON-safe"):
        lane.build_commit_request(projection)


def test_physical_mutation_skips_empty_enabled_request_without_backend_mutation():
    class ForbiddenBridge:
        def __getattr__(self, name):
            raise AssertionError(f"empty request must not access {name}")

    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="segment", buffering_commit_enabled=1),
            flow_kind="buffering",
        ),
    )
    request = lane.build_commit_request(
        SimpleNamespace(mode="segment", selected_actions=(), metadata={})
    )

    result = physical_mutation_backend_for(
        SimpleNamespace(
            backend_caps=openroad_backend_caps().to_dict(),
            openroad_bridge=ForbiddenBridge(),
        )
    ).commit_buffers(
        request,
        output_dir="unused",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_projected_buffer_actions"
    assert request.is_noop


def test_combined_request_is_not_noop_when_only_sizing_actions_exist():
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="segment", buffering_commit_enabled=1),
            flow_kind="joint",
        ),
    )
    request = lane.build_commit_request(
        SimpleNamespace(mode="segment", selected_actions=(), metadata={}),
        sizing_actions=(
            {
                "instance_id": 4,
                "before_cell_id": 2,
                "after_cell_id": 3,
            },
        ),
        sizing_cell_ids=torch.tensor([0, 1, 2, 3, 3]),
        placement_node_x=torch.tensor([10.0, 20.0]),
        placement_node_y=torch.tensor([30.0, 40.0]),
    )

    summary = request.to_summary()
    assert not request.is_noop
    assert summary["action_count"] == 0
    assert summary["sizing_action_count"] == 1
    assert summary["placement_coordinate_count"] == 2
    assert summary["placement_snapshot_digest"]


def test_physical_mutation_rejects_actions_mutated_after_request_construction():
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="candidate", buffering_commit_enabled=1),
            flow_kind="buffering",
        ),
    )
    request = lane.build_commit_request(
        SimpleNamespace(
            mode="candidate",
            selected_actions=(_commit_action(),),
            metadata={},
        )
    )
    request.actions[0]["net_name"] = "tampered"

    placedb = SimpleNamespace(
        backend_caps=openroad_backend_caps().to_dict(),
        openroad_bridge=None,
    )
    with pytest.raises(ValueError, match="do not match action digest"):
        physical_mutation_backend_for(placedb).commit_buffers(
            request,
            output_dir="unused",
        )


def test_physical_mutation_uses_coordinate_backend_once_when_enabled():
    class FakeBridge:
        def __init__(self):
            self.calls = []
            self.metrics_calls = 0
            self.tcl_commands = []

        def query_diff_guided_batch_timing_metrics(self):
            self.metrics_calls += 1
            if self.metrics_calls == 1:
                return {"status": "ok", "wns": -1.0, "tns": -10.0}
            return {"status": "ok", "wns": -0.9, "tns": -8.0}

        def run_coordinate_buffer_insert(self, action, config):
            self.calls.append((dict(action), dict(config)))
            return {
                "status": "ok",
                "inserted_buffer_count_delta": 1,
            }

        def eval_tcl_string(self, command):
            self.tcl_commands.append(command)
            return ""

    bridge = FakeBridge()
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="candidate", buffering_commit_enabled=1),
            flow_kind="buffering",
        ),
        placedb=SimpleNamespace(
            backend_caps=openroad_backend_caps().to_dict(),
            openroad_bridge=bridge,
        ),
    )
    projection = SimpleNamespace(selected_actions=(_commit_action(),))

    request = lane.build_commit_request(projection)
    summary = physical_mutation_backend_for(lane.placedb).commit_buffers(
        request,
        output_dir="unused",
    )

    assert summary["status"] == "accepted"
    assert summary["commit_enabled"]
    assert summary["action_count"] == 1
    assert summary["attempted_action_count"] == 1
    assert summary["accepted_action_count"] == 1
    assert summary["before_tns"] == -10.0
    assert summary["after_tns"] == -8.0
    assert summary["delta_tns"] == 2.0
    assert request.to_summary()["action_count"] == 1
    assert len(bridge.calls) == 1
    assert bridge.calls[0][0]["net_name"] == "n1"
    assert bridge.calls[0][1]["defer_timing_update"]
    assert bridge.metrics_calls == 2
    assert bridge.tcl_commands == ["estimate_parasitics -placement"]


def test_physical_mutation_blocks_incomplete_coordinate_payload_before_backend():
    class FakeBridge:
        def run_coordinate_buffer_insert(self, action, config):
            raise AssertionError("backend should not be called for blocked payload")

    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="candidate", buffering_commit_enabled=1),
            flow_kind="buffering",
        ),
        placedb=SimpleNamespace(
            backend_caps=openroad_backend_caps().to_dict(),
            openroad_bridge=FakeBridge(),
        ),
    )
    action = _commit_action()
    action.pop("buffer_master_name")
    projection = SimpleNamespace(selected_actions=(action,))

    summary = physical_mutation_backend_for(lane.placedb).commit_buffers(
        lane.build_commit_request(projection),
        output_dir="unused",
    )

    assert summary["status"] == "blocked"
    assert summary["reason"] == "blocked_missing_action_payload"
    assert summary["attempted_action_count"] == 0
    assert summary["blocked_action_count"] == 1


def test_physical_mutation_preserves_optional_committed_def_export(tmp_path):
    class FakeBridge:
        def __init__(self):
            self.def_paths = []

        def query_diff_guided_batch_timing_metrics(self):
            return {"status": "ok", "wns": -1.0, "tns": -10.0}

        def run_coordinate_buffer_insert(self, action, config):
            return {"status": "ok", "inserted_buffer_count_delta": 1}

        def eval_tcl_string(self, command):
            return ""

        def write_def(self, path):
            self.def_paths.append(path)

    committed_def = tmp_path / "committed.def"
    bridge = FakeBridge()
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="candidate",
                buffering_commit_enabled=1,
                buffering_committed_def_path=str(committed_def),
            ),
            flow_kind="buffering",
        ),
        placedb=SimpleNamespace(
            backend_caps=openroad_backend_caps().to_dict(),
            openroad_bridge=bridge,
        ),
    )
    request = lane.build_commit_request(
        SimpleNamespace(mode="candidate", selected_actions=(_commit_action(),), metadata={})
    )

    result = physical_mutation_backend_for(lane.placedb).commit_buffers(
        request,
        output_dir=str(tmp_path),
    )

    assert result["status"] == "accepted"
    assert result["committed_def_path"] == str(committed_def)
    assert bridge.def_paths == [str(committed_def)]


def test_nonlinear_place_exposes_neutral_buffering_lane_helpers():
    source = (
        Path(__file__).resolve().parents[3] / "dreamplace/NonLinearPlace.py"
    ).read_text(encoding="utf-8")

    assert "def _buffering_flow_enabled" in source
    assert "def _build_buffering_lane" in source
    assert "def _maybe_run_buffering_inner_loop" in source
    assert "def _maybe_run_buffering_lane" not in source
    assert "reason=\"buffering_inner_loop_completed\"" in source
    assert "BufferingOptimizationLane" in source


def test_nonlinear_place_buffering_flow_runs_inner_loop_without_physical_commit():
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    state = _candidate_state()
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[state.bu_logits]))
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_continuous_steps=1,
        buffering_continuous_lr=0.01,
    )

    def timing_obj(pos):
        return torch.tensor(-0.1), -(pos * pos).sum(), torch.tensor(0.0), torch.tensor(0.0)

    model.timing_obj = timing_obj
    placer.buffer_optimization_state = state
    placer.buffer_relaxed_timing_payload = {"metadata": {}}

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert summary["mode"] == "candidate"
    assert summary["inner_loop"]["iterations"] == 1
    assert summary["commit"]["status"] == "disabled"
    assert summary["commit"]["reason"] == "commit_not_requested"
    assert model.data_collections.last_buffering_projection_result is not None
    assert placer.last_buffering_lane_summary == summary


def test_nonlinear_place_candidate_discrete_dispatches_before_optimizer(tmp_path):
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    state = _discrete_candidate_state()
    pos = torch.nn.Parameter(torch.tensor([0.0]))
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[pos]))
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_candidate",
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_candidate_strategy="discrete_net_gradient",
        buffering_fixed_bsu_index=2,
        buffering_continuous_steps=1,
        buffering_commit_enabled=0,
    )
    placer.buffer_optimization_state = state
    placer.buffer_relaxed_timing_payload = {
        "metadata": {
            "state_kind": "candidate",
            "candidate_strategy": "discrete_net_gradient",
        }
    }

    def timing_obj(_pos):
        attached = model.data_collections.buffer_optimization_state
        tns = attached.activation_param.sum()
        return torch.tensor(-0.1), tns, torch.tensor(0.0), torch.tensor(0.0)

    model.timing_obj = timing_obj

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert summary["mode"] == "candidate"
    assert summary["strategy"] == "discrete_net_gradient"
    assert summary["inner_loop"]["iterations"] == 1
    assert summary["inner_loop"]["terminal_metrics"][
        "total_virtual_buffer_count"
    ] == 1
    assert summary["commit"]["status"] == "disabled"
    assert state.activation_param.grad is None
    assert state.activation_param.sum().item() == 1.0


def test_nonlinear_place_buffering_flow_runs_segment_inner_loop_without_physical_commit():
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    state = _segment_state(initial_z=1.25)
    payload = _segment_payload()
    pos = torch.nn.Parameter(torch.tensor([0.0]))
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[pos]))
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="segment",
        buffering_continuous_steps=1,
        buffering_continuous_lr=0.01,
    )
    placer.buffer_optimization_state = state
    placer.buffer_relaxed_timing_payload = payload

    def timing_obj(_pos):
        data_state = model.data_collections.buffer_segment_count_state
        tns = -(
            data_state.z_param.square().sum()
            + 0.5 * data_state.bsu_index_param.square().sum()
        )
        return torch.tensor(-0.1), tns, torch.tensor(0.0), torch.tensor(0.0)

    model.timing_obj = timing_obj

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert summary["mode"] == "segment"
    assert summary["state_kind"] == "segment_count"
    assert summary["inner_loop"]["iterations"] == 1
    assert summary["commit"]["status"] == "disabled"
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert summary["inner_loop"]["metrics_trace"][0]["grad_z_norm"] > 0.0
    assert summary["inner_loop"]["metrics_trace"][0]["grad_bsu_norm"] > 0.0
    assert model.data_collections.last_buffering_projection_result is not None


def test_nonlinear_place_buffering_commit_returns_request_before_engine_execution(
    monkeypatch, tmp_path
):
    from dreamplace.NonLinearPlace import NonLinearPlace
    from dreamplace.ops.buffer_insertion import buffering_inner_loop
    from dreamplace.ops.buffer_insertion.buffering_lane import BufferCommitRequest

    action = _commit_action()
    projection = SimpleNamespace(
        mode="candidate",
        selected_actions=(action,),
        metadata={},
        projected_buffer_count=1,
    )
    request = BufferCommitRequest(
        version=1,
        mode="candidate",
        actions=(action,),
        action_digest="unit-request-digest",
        iteration=1,
        reason="buffering_inner_loop_final_commit",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path="",
        refresh_mode="topo",
        rebuild_mode="topo",
        projection_metadata={},
        runtime_refresh_summary={"status": "ok"},
    )

    class FakeLane:
        def __init__(self):
            self.config = SimpleNamespace(continuous_lr=0.01, continuous_steps=1)
            self.parameter = torch.nn.Parameter(torch.tensor([0.0]))
            self.prepare_calls = 0
            self.attach_calls = 0
            self.last_runtime_refresh_summary = {"status": "ok"}
            self.request_calls = []

        def prepare(self, model=None, params=None):
            del model, params
            self.prepare_calls += 1
            return self

        def attach_state(self, data_collections):
            del data_collections
            self.attach_calls += 1
            return self

        def optimizer_parameters(self):
            return [self.parameter]

        def build_commit_request(self, projection_result, **kwargs):
            self.request_calls.append((projection_result, kwargs))
            return request

        def summarize(self):
            return {"mode": "candidate", "state_kind": "candidate"}

    lane = FakeLane()
    loop_summary = SimpleNamespace(
        iterations=1,
        projection_trace=[],
        periodic_integer_projection_trace=[],
        final_projection_wall_ms=0.0,
        metrics_trace=[],
        final_projection=projection,
    )
    monkeypatch.setattr(
        buffering_inner_loop,
        "run_buffering_inner_loop",
        lambda **_: loop_summary,
    )

    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer._build_buffering_lane = lambda _: lane
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[lane.parameter]))
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_design",
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_continuous_steps=1,
        buffering_continuous_lr=0.01,
    )

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert lane.prepare_calls == 1
    assert lane.attach_calls == 1
    assert lane.request_calls == [
        (
            projection,
            {
                "iteration": 1,
                "reason": "buffering_inner_loop_final_commit",
                "runtime_refresh": {"status": "ok"},
            },
        )
    ]
    assert placer.last_buffer_commit_request is request
    assert placer.last_post_commit_control_result["status"] == (
        "pending_buffer_commit_request"
    )
    assert summary["commit"]["status"] == "pending_buffer_commit_request"
    assert summary["commit_request"] == request.to_summary()

    placer.params = params
    control_result = placer.record_outer_buffer_commit_result(
        {
            "status": "completed",
            "commit_request": request.to_summary(),
            "commit": {"status": "accepted", "accepted_action_count": 1},
        }
    )

    assert control_result["status"] == "safe_stopped_after_accepted_buffer_commit"
    assert placer.last_buffering_lane_summary["commit"] == {
        "status": "accepted",
        "accepted_action_count": 1,
    }
    artifact = json.loads(
        (tmp_path / "unit_design_buffering_inner_loop_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["summary"]["commit"]["status"] == "accepted"


def test_nonlinear_place_buffering_flow_builds_state_from_model_inputs():
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    data = SimpleNamespace(
        pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        buffering_nets=[
            {
                "net_id": 10,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: []},
                    "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}},
                    "node_cap": {0: 0.0, 1: 1.0},
                    "sink_nodes": [1],
                },
            }
        ],
        buffering_candidate_rows=[
            {
                "candidate_id": 100,
                "net_id": 10,
                "candidate_node_id": 1,
                "x_dbu": 100,
                "y_dbu": 0,
                "buffer_main_type_index": 7,
            }
        ],
        buffering_buffer_library={
            0: {"input_cap": 0.4, "delay": 2.0, "output_slew": 0.3},
            1: {"input_cap": 0.8, "delay": 3.0, "output_slew": 0.2},
        },
    )
    model = SimpleNamespace(data_collections=data)
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_continuous_steps=1,
        buffering_continuous_lr=0.01,
    )

    def timing_obj(pos):
        state = model.data_collections.buffer_optimization_state
        return (
            torch.tensor(-0.1),
            -state.relaxed_bu().sum(),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    model.timing_obj = timing_obj

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert summary["state_build_status"] == "ok"
    assert summary["candidate_count"] == 1
    assert data.buffer_optimization_state.bu_logits.grad is not None
    assert data.buffer_relaxed_timing_payload["metadata"]["state_source"] == (
        "data_collections_buffering_inputs"
    )


def test_nonlinear_place_buffering_flow_refreshes_timing_topology_before_objective():
    from dreamplace.NonLinearPlace import NonLinearPlace

    class FakePlacer(NonLinearPlace):
        def __init__(self):
            pass

        def _refresh_live_timing_topology(self, pos):
            self.refresh_calls.append(pos)
            self._live_timing_topology_initialized = True

    placer = FakePlacer()
    placer.refresh_calls = []
    placer._live_timing_topology_initialized = False
    placer.op_collections = SimpleNamespace(pin_pos_op=object(), steiner_topo_op=object())
    state = _candidate_state()
    pos = torch.nn.Parameter(torch.tensor([0.0]))
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[pos]))
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_continuous_steps=1,
        buffering_continuous_lr=0.01,
    )
    placer.buffer_optimization_state = state
    placer.buffer_relaxed_timing_payload = {"metadata": {}}

    def timing_obj(_pos):
        assert placer._live_timing_topology_initialized
        return (
            torch.tensor(-0.1),
            -state.relaxed_bu().sum(),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    model.timing_obj = timing_obj

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert placer.refresh_calls == [pos]


def test_nonlinear_place_buffering_flow_refreshes_timing_topology_before_state_build():
    from dreamplace.NonLinearPlace import NonLinearPlace

    class FakePlacer(NonLinearPlace):
        def __init__(self):
            pass

        def _refresh_live_timing_topology(self, pos):
            self.refresh_calls.append(pos)
            self._live_timing_topology_initialized = True
            self.data_collections.buffering_nets = [_segment_net()]
            self.data_collections.buffering_buffer_library = {
                0: {"input_cap": 0.4, "delay": 2.0, "output_slew": 0.3},
                1: {"input_cap": 0.8, "delay": 3.0, "output_slew": 0.2},
            }
            self.data_collections.buffer_main_type_index = 7

    placer = FakePlacer()
    placer.refresh_calls = []
    placer._live_timing_topology_initialized = False
    placer.op_collections = SimpleNamespace(pin_pos_op=object(), steiner_topo_op=object())
    pos = torch.nn.Parameter(torch.tensor([0.0]))
    data = SimpleNamespace(pos=[pos])
    model = SimpleNamespace(data_collections=data)
    placer.data_collections = data
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="segment",
        buffering_continuous_steps=0,
        buffering_continuous_lr=0.01,
        buffering_max_repeaters_per_segment=2,
        buffering_segment_count_z_init=0.0,
        buffering_initial_bsu_index=1.0,
    )

    def timing_obj(_pos):
        return (
            torch.tensor(-0.1),
            -data.buffer_optimization_state.z_value().sum(),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    model.timing_obj = timing_obj

    summary = placer._maybe_run_buffering_inner_loop(params, model)

    assert summary["status"] == "completed"
    assert summary["state_build_status"] == "ok"
    assert summary["state_kind"] == "segment_count"
    assert placer.refresh_calls == [pos]


def test_nonlinear_place_writes_buffering_inner_loop_summary_artifact(tmp_path):
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_design",
        flow_kind="buffering",
    )
    summary = {
        "status": "completed",
        "mode": "candidate",
        "candidate_count": 3,
        "inner_loop": {"iterations": 1},
    }

    artifact = placer._write_buffering_inner_loop_artifact(params, summary)

    path = tmp_path / "unit_design_buffering_inner_loop_summary.json"
    assert artifact["summary_path"] == str(path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_version"] == 1
    assert payload["metadata"]["flow_kind"] == "buffering"
    assert payload["summary"] == summary


def test_nonlinear_place_omits_rebuildable_arrays_from_buffering_artifact(tmp_path):
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_design",
        flow_kind="buffering",
    )
    summary = {
        "status": "completed",
        "candidate_count": 3,
        "candidate_ids": [0, 1, 2],
        "synthetic_segment_splits": [{"candidate_id": 0}],
        "state_build_metadata": {
            "candidate_count": 3,
            "candidate_ids": [0, 1, 2],
            "segment_state_builder_backend_used": "native",
        },
    }

    placer._write_buffering_inner_loop_artifact(params, summary)

    payload = json.loads(
        (tmp_path / "unit_design_buffering_inner_loop_summary.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_summary = payload["summary"]
    assert "candidate_ids" not in artifact_summary
    assert "synthetic_segment_splits" not in artifact_summary
    assert "candidate_ids" not in artifact_summary["state_build_metadata"]
    assert artifact_summary["state_build_metadata"][
        "segment_state_builder_backend_used"
    ] == "native"
    assert artifact_summary["artifact_omitted_fields"] == {
        "candidate_ids": {"entry_count": 3},
        "synthetic_segment_splits": {"entry_count": 1},
    }


def test_nonlinear_place_detects_request_that_requires_outer_commit():
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.last_buffer_commit_request = SimpleNamespace(
        commit_enabled=True,
        is_noop=False,
    )

    assert placer._joint_buffer_request_requires_outer_commit(
        {"status": "request_ready"}
    )
    placer.last_buffer_commit_request = SimpleNamespace(
        commit_enabled=False,
        is_noop=False,
    )
    assert not placer._joint_buffer_request_requires_outer_commit(
        {"status": "request_ready"}
    )
    placer.last_buffer_commit_request = SimpleNamespace(
        commit_enabled=True,
        is_noop=True,
    )
    assert not placer._joint_buffer_request_requires_outer_commit(
        {"status": "request_ready"}
    )
    assert not placer._joint_buffer_request_requires_outer_commit(None)


def test_nonlinear_place_safe_stops_before_outer_joint_buffer_commit(tmp_path):
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_design",
        flow_kind="joint",
    )
    placer.last_joint_flow_artifact = {
        "summary": {
            "buffer_commit_trace": [
                {
                    "status": "request_ready",
                    "projected_buffer_count": 1,
                }
            ],
            "buffer_commit_trace_count": 1,
        }
    }

    rsmt, hpwl, processed = placer._safe_stop_after_joint_buffer_request(
        params,
        iteration=3,
        place_stage_metrics=[],
        processed_metrics={"objective": [1.0]},
        commit_result={
            "status": "request_ready",
            "projected_buffer_count": 1,
            "commit_request": {"refresh_mode": "topo", "rebuild_mode": "topo"},
        },
    )

    assert rsmt != rsmt
    assert hpwl != hpwl
    assert processed["joint_buffer_commit_safe_stop"] == {
        "status": "safe_stopped_before_buffer_commit",
        "projected_buffer_count": 1,
    }
    path = tmp_path / "unit_design_joint_flow_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["post_commit_control_policy"] == {
        "status": "safe_stopped_before_buffer_commit",
        "reason": "physical_buffer_commit_owned_by_placement_engine",
        "projected_buffer_count": 1,
    }


def test_nonlinear_place_records_outer_commit_request_provenance(tmp_path):
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit_design",
        flow_kind="joint",
        buffering_commit_enabled=1,
    )
    placer.params = params
    placer.joint_coordinator = SimpleNamespace(summarize=lambda: {})
    placer.joint_buffer_commit_trace = [
        {
            "status": "request_ready",
            "commit_request": {"action_digest": "request-digest"},
        }
    ]

    result = placer.record_outer_buffer_commit_result(
        {
            "status": "completed",
            "commit_request": {
                "action_digest": "request-digest",
                "reason": "joint_buffer_periodic_commit",
            },
            "commit": {"status": "accepted", "accepted_action_count": 1},
        }
    )

    assert result["commit_request"] == {
        "action_digest": "request-digest",
        "reason": "joint_buffer_periodic_commit",
    }
    path = tmp_path / "unit_design_joint_flow_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["post_commit_control_result"]["commit_request"][
        "reason"
    ] == "joint_buffer_periodic_commit"


def test_nonlinear_place_non_buffering_flow_skips_lane():
    from dreamplace.NonLinearPlace import NonLinearPlace

    placer = NonLinearPlace.__new__(NonLinearPlace)
    model = SimpleNamespace(data_collections=SimpleNamespace())
    params = SimpleNamespace(flow_kind="sizing")

    assert placer._maybe_run_buffering_inner_loop(params, model) is None
    assert not hasattr(placer, "last_buffering_lane_summary")
