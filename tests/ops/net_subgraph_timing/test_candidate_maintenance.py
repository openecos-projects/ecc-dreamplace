from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
    CandidateGeometryRefreshError,
    active_candidate_net_ids,
    apply_candidate_geometry_rows,
    collect_candidate_dirty_nets,
    compute_candidate_dirty_nets,
    refresh_candidate_geometry_rows,
)
from dreamplace.ops.buffer_insertion.candidate_geometry import (
    refresh_candidate_geometry,
)


def test_dirty_net_collection_unions_instances_and_accepted_nets():
    result = compute_candidate_dirty_nets(
        changed_instance_ids=[1, 1, 2],
        moved_instance_ids=[3],
        accepted_net_ids=[5, 5],
        candidate_net_ids=[2, 4, 5, 6],
        flat_node2pin_map=[0, 1, 2, 3, 4],
        flat_node2pin_start_map=[0, 0, 2, 3, 4],
        pin2net_map=[2, 6, 4, 5, 3],
    )

    assert result["dirty_net_ids"].tolist() == [2, 4, 5, 6]
    assert result["counts"] == {
        "changed_instance_count": 2,
        "moved_instance_count": 1,
        "source_instance_count": 3,
        "incident_pin_count": 4,
        "unconnected_pin_count": 0,
        "accepted_net_count": 1,
        "accepted_candidate_net_count": 1,
        "dirty_net_count": 4,
    }
    assert result["metadata"]["backend"] == "native_cpp"
    assert result["metadata"]["unrestricted_candidate_domain"] is False


def test_dirty_net_collection_accepts_an_empty_change_set():
    result = compute_candidate_dirty_nets(
        changed_instance_ids=[],
        moved_instance_ids=[],
        accepted_net_ids=[],
        candidate_net_ids=[1, 2],
        flat_node2pin_map=[0, 1],
        flat_node2pin_start_map=[0, 1, 2],
        pin2net_map=[1, 2],
    )

    assert result["dirty_net_ids"].numel() == 0
    assert result["counts"]["dirty_net_count"] == 0


def test_dirty_net_collection_ignores_unconnected_pins():
    result = compute_candidate_dirty_nets(
        changed_instance_ids=[1],
        moved_instance_ids=[],
        accepted_net_ids=[],
        candidate_net_ids=[2],
        flat_node2pin_map=[0, -1],
        flat_node2pin_start_map=[0, 0, 2],
        pin2net_map=[2, -1],
    )

    assert result["dirty_net_ids"].tolist() == [2]
    assert result["counts"]["incident_pin_count"] == 1
    assert result["counts"]["unconnected_pin_count"] == 1


def test_dirty_net_collection_accepts_numpy_csr_inputs():
    result = compute_candidate_dirty_nets(
        changed_instance_ids=[1],
        moved_instance_ids=[],
        accepted_net_ids=[],
        candidate_net_ids=torch.tensor([2, 4]),
        flat_node2pin_map=np.array([0, 2, 1, 3], dtype=np.int32),
        flat_node2pin_start_map=np.array([0, 0, 2, 4], dtype=np.int32),
        pin2net_map=np.array([2, -1, 4, 4], dtype=np.int32),
    )

    assert result["dirty_net_ids"].tolist() == [2, 4]


def test_active_candidate_nets_are_unique_and_ignore_zero_activations():
    state = type(
        "State",
        (),
        {
            "candidate_net_id": torch.tensor([5, 5, 8, 9]),
            "activation_param": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
        },
    )()

    assert active_candidate_net_ids(state) == [5, 9]


def test_dirty_net_collection_uses_conservative_all_nets_without_csr():
    state = type(
        "State",
        (),
        {
            "candidate_net_id": torch.tensor([4, 4, 7]),
            "activation_param": torch.zeros(3, 2),
        },
    )()

    dirty, summary = collect_candidate_dirty_nets(
        state=state,
        lane=SimpleNamespace(buffer_relaxed_timing_payload={}),
        placedb=SimpleNamespace(),
        data=SimpleNamespace(),
        changed_instance_ids=[1],
        accepted_net_ids=[],
    )

    assert dirty == [4, 7]
    assert summary["status"] == "conservative_all"
    assert summary["dirty_net_ids"] == [4, 7]


def test_dirty_net_collection_skips_native_mapping_without_changes():
    state = type(
        "State",
        (),
        {
            "candidate_net_id": torch.tensor([1]),
            "activation_param": torch.zeros(1, 2),
        },
    )()

    dirty, summary = collect_candidate_dirty_nets(
        state=state,
        lane=SimpleNamespace(),
        placedb=SimpleNamespace(),
        data=SimpleNamespace(),
    )

    assert dirty == []
    assert summary["reason"] == "no_pending_changes"


def test_native_candidate_geometry_refreshes_only_dirty_rows():
    payload = {
        "candidate_net_id": torch.tensor([1, 2, 2]),
        "candidate_parent_node_id": torch.tensor([0, 2, 2]),
        "candidate_child_node_id": torch.tensor([1, 3, 3]),
        "candidate_tree_node_id": torch.tensor([-1, -1, -1]),
        "candidate_is_segment": torch.tensor([1, 1, 1]),
        "candidate_split_ratio": torch.tensor(
            [0.25, 0.5, 0.75],
            dtype=torch.float64,
        ),
    }

    result = refresh_candidate_geometry_rows(
        payload=payload,
        dirty_net_ids=[2],
        node_x_dbu=torch.tensor([0.0, 100.0, 0.0, 200.0]),
        node_y_dbu=torch.tensor([0.0, 0.0, 20.0, 20.0]),
        all_candidates=False,
    )

    assert result["candidate_rows"].tolist() == [1, 2]
    assert result["net_ids"].tolist() == [2, 2]
    assert result["x_dbu"].tolist() == [100.0, 150.0]
    assert result["y_dbu"].tolist() == [20.0, 20.0]
    assert result["segment_length_dbu"].tolist() == [200.0, 200.0]
    assert result["upstream_length_dbu"].tolist() == [100.0, 150.0]
    assert result["downstream_length_dbu"].tolist() == [100.0, 50.0]


def test_native_candidate_geometry_reports_invalid_dirty_rows():
    payload = {
        "candidate_net_id": torch.tensor([1]),
        "candidate_parent_node_id": torch.tensor([0]),
        "candidate_child_node_id": torch.tensor([99]),
        "candidate_tree_node_id": torch.tensor([-1]),
        "candidate_is_segment": torch.tensor([1]),
        "candidate_split_ratio": torch.tensor([0.5], dtype=torch.float64),
    }

    with pytest.raises(
        CandidateGeometryRefreshError,
        match="invalid dirty rows",
    ):
        refresh_candidate_geometry_rows(
            payload=payload,
            dirty_net_ids=[1],
            node_x_dbu=torch.tensor([0.0, 100.0]),
            node_y_dbu=torch.zeros(2),
        )


def test_apply_native_geometry_updates_only_returned_records():
    records = (
        {"net_id": 1, "x_dbu": 0, "y_dbu": 0},
        {
            "net_id": 2,
            "parent_node_id": 2,
            "child_node_ids": [3],
            "x_dbu": 0,
            "y_dbu": 0,
        },
    )
    result = {
        "candidate_rows": torch.tensor([1]),
        "x_dbu": torch.tensor([100.25], dtype=torch.float64),
        "y_dbu": torch.tensor([20.6], dtype=torch.float64),
        "segment_length_dbu": torch.tensor([10.0], dtype=torch.float64),
        "upstream_length_dbu": torch.tensor([5.0], dtype=torch.float64),
        "downstream_length_dbu": torch.tensor([5.0], dtype=torch.float64),
    }

    updated = apply_candidate_geometry_rows(records, result, dbu=1000.0)

    assert updated[0] is records[0]
    assert updated[1]["x_dbu"] == 100
    assert updated[1]["y_dbu"] == 21
    assert updated[1]["x_um"] == 0.10025
    assert updated[1]["segment_length_dbu"] == 10
    assert updated[1]["candidate_geometry_status"] == "refreshed_native_dirty_net"


def test_native_candidate_geometry_matches_python_full_refresh():
    records = (
        {
            "net_id": 1,
            "candidate_id": 0,
            "parent_node_id": 0,
            "child_node_ids": [1],
            "segment_split_ratio": 0.25,
        },
        {
            "net_id": 2,
            "candidate_id": 1,
            "tree_node_id": 2,
        },
    )
    payload = {
        "candidate_net_id": torch.tensor([1, 2]),
        "candidate_parent_node_id": torch.tensor([0, -1]),
        "candidate_child_node_id": torch.tensor([1, -1]),
        "candidate_tree_node_id": torch.tensor([-1, 2]),
        "candidate_is_segment": torch.tensor([1, 0]),
        "candidate_split_ratio": torch.tensor(
            [0.25, 0.0],
            dtype=torch.float64,
        ),
    }
    topology = {
        "dbu": 1000.0,
        "node_x_dbu": torch.tensor([0.0, 400.0, 200.0]),
        "node_y_dbu": torch.tensor([0.0, 0.0, 60.0]),
    }

    result = refresh_candidate_geometry_rows(
        payload=payload,
        dirty_net_ids=[1, 2],
        node_x_dbu=topology["node_x_dbu"],
        node_y_dbu=topology["node_y_dbu"],
    )
    native = apply_candidate_geometry_rows(records, result, dbu=1000.0)
    reference = refresh_candidate_geometry(records, topology).candidate_records
    fields = (
        "x_dbu",
        "y_dbu",
        "candidate_location_x_dbu",
        "candidate_location_y_dbu",
        "upstream_length_dbu",
        "downstream_length_dbu",
    )

    for native_record, reference_record in zip(native, reference, strict=True):
        assert {
            name: native_record.get(name)
            for name in fields
        } == {
            name: reference_record.get(name)
            for name in fields
        }
    assert native[0]["segment_length_dbu"] == reference[0]["segment_length_dbu"]
