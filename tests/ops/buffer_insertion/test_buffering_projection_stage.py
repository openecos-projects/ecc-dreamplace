import pytest
import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    build_discrete_candidate_state,
)
from dreamplace.ops.buffer_insertion.projection import (
    empty_buffering_projection,
    project_candidate_buffer_state,
    project_buffering_lane_state,
)
from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state


def _state():
    return build_buffer_optimization_state(
        [
            {
                "candidate_id": 0,
                "tree_node_id": 10,
                "net_id": 1,
                "x_dbu": 100,
                "y_dbu": 200,
                "buffer_main_type_index": 1,
            },
            {
                "candidate_id": 1,
                "tree_node_id": 11,
                "net_id": 2,
                "x_dbu": 300,
                "y_dbu": 400,
                "buffer_main_type_index": 1,
            },
        ],
        buffer_main_type_index=1,
        legal_buffer_count=4,
    )


def test_empty_projection_has_structured_fields():
    result = empty_buffering_projection(mode="segment", reason="unit")

    assert result.mode == "segment"
    assert result.selected_actions == ()
    assert result.projected_buffer_count == 0
    assert result.affected_nets == ()
    assert result.validation["status"] == "empty"


def test_candidate_projection_returns_count_and_affected_nets():
    state = _state()
    with torch.no_grad():
        state.bu_logits.copy_(torch.tensor([-3.0, 3.0]))

    result = project_candidate_buffer_state(state, top_k=1)

    assert result.mode == "candidate"
    assert result.projected_buffer_count == 1
    assert result.affected_nets == (2,)
    assert result.validation["status"] == "ok"
    assert result.selected_actions[0]["candidate_id"] == 1


def test_discrete_candidate_projection_exactly_materializes_active_ids():
    candidates = [dict(candidate) for candidate in _state().candidate_records]
    state = build_discrete_candidate_state(
        candidates,
        buffer_main_type_index=1,
        legal_buffer_count=4,
        fixed_bsu_index=2,
    )
    with torch.no_grad():
        state.activation_param.copy_(torch.tensor([1.0, 0.0]))

    result = project_candidate_buffer_state(state)

    assert result.projected_buffer_count == 1
    assert result.selected_actions[0]["candidate_id"] == 0
    assert result.selected_actions[0]["bsu"] == 2
    assert result.validation["exact_active_candidate_match"]
    assert result.validation["active_candidate_ids"] == (0,)
    assert result.metadata["projection_source"] == "discrete_candidate_state"


def test_discrete_candidate_projection_rejects_topk_filtering():
    candidates = [dict(candidate) for candidate in _state().candidate_records]
    state = build_discrete_candidate_state(
        candidates,
        buffer_main_type_index=1,
        legal_buffer_count=4,
        fixed_bsu_index=2,
    )

    with pytest.raises(ValueError, match="does not allow filtering"):
        project_candidate_buffer_state(state, top_k=1)


def _segment_nets():
    return [
        {
            "net_id": 10,
            "net_name": "n10",
            "driver_pin_id": 0,
            "driver_pin_name": "drv/Y",
            "pin_name_by_id": {
                0: "drv/Y",
                1: "a/A",
                2: "b/A",
                3: "c/A",
                4: "d/A",
            },
            "coordinates": {
                0: (0, 0),
                1: (100, 0),
                2: (200, 0),
                3: (300, 0),
                4: (400, 0),
            },
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: [2], 2: [3], 3: [4], 4: []},
                "edge_rc": {
                    (0, 1): {"r": 1.0, "c": 0.0},
                    (1, 2): {"r": 1.0, "c": 0.0},
                    (2, 3): {"r": 1.0, "c": 0.0},
                    (3, 4): {"r": 1.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
                "sink_nodes": [1, 2, 3, 4],
            },
        }
    ]


def test_segment_count_projection_uses_round_clip_candidates():
    nets = _segment_nets()
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=8,
        max_repeater_count=4,
        initial_z=0.0,
        initial_bsu_index=1.0,
    )
    with torch.no_grad():
        state.z_param.copy_(torch.tensor([-1.0, 0.49, 2.6, 99.0]))
        state.bsu_index_param.fill_(1.0)
    lane = type(
        "Lane",
        (),
        {
            "config": type("Config", (), {"mode": "segment"})(),
            "buffer_optimization_state": state,
            "buffer_relaxed_timing_payload": {
                "metadata": {"state_kind": "segment_count"},
                "nets": nets,
            },
            "data_collections": None,
        },
    )()

    result = project_buffering_lane_state(lane)

    counts_by_segment = {}
    for action in result.selected_actions:
        counts_by_segment.setdefault(int(action["segment_id"]), int(action["projected_repeater_count"]))
    assert result.mode == "segment"
    assert result.validation["status"] == "ok"
    assert result.metadata["projection_source"] == "segment_count_state"
    assert result.metadata["action_model"] == "batch_projected_candidates"
    assert result.projected_buffer_count == 7
    assert counts_by_segment == {2: 3, 3: 4}


def test_segment_count_projection_materializes_coordinate_actions_with_legal_table():
    nets = _segment_nets()
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=8,
        max_repeater_count=4,
        initial_z=1.0,
        initial_bsu_index=2.0,
    )
    lane = type(
        "Lane",
        (),
        {
            "config": type("Config", (), {"mode": "segment"})(),
            "buffer_optimization_state": state,
            "buffer_relaxed_timing_payload": {
                "metadata": {"state_kind": "segment_count"},
                "nets": nets,
                "buffer_legal_table": {
                    "legal_cell_ids": list(range(10, 18)),
                    "legal_master_names": [f"BUF_X{i}" for i in range(8)],
                    "legal_size_values": [float(i + 1) for i in range(8)],
                },
            },
            "data_collections": None,
        },
    )()

    result = project_buffering_lane_state(lane)

    assert result.metadata["action_model"] == "coordinate_commit_actions"
    assert result.projected_buffer_count == 4
    assert result.affected_nets == (10,)
    action = result.selected_actions[0]
    assert action["action_kind"] == "buffer_insert"
    assert action["net_name"] == "n10"
    assert action["buffer_master_id"] == 12
    assert action["buffer_master_name"] == "BUF_X2"
    assert action["target_size_idx"] == 2
    assert action["candidate_location_x_dbu"] == 50.0
    assert action["candidate_location_y_dbu"] == 0.0
    assert "projection_wall_ms" in result.metadata
    assert "action_materialize_wall_ms" in result.metadata
