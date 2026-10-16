from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.flows.joint_physical_actions import (
    commit_joint_physical_actions,
    final_physical_context,
)
from dreamplace.ops.buffer_insertion.buffering_lane import BufferCommitRequest, _action_digest


def test_joint_context_compares_live_cell_ids_to_native_baseline():
    placedb = SimpleNamespace(
        num_movable_nodes=1,
        num_nodes=2,
        inst_cell_id=np.array([1, -1]),  # Already changed by the live optimizer.
        pydb=SimpleNamespace(
            inst_main_id=[0, -1],
            main_id_2_cell_id_start=[0, 2],
            inst_libcell_offset=[0, -1],
        ),
        flat_libcell_names=[b"CELL_X1", b"CELL_X2"],
        node_names=[b"inst0", b"port"],
        flat_libcell_width=[1, 2],
        flat_libcell_height=[2, 2],
    )
    placer = SimpleNamespace(
        placedb=placedb,
        params=SimpleNamespace(scale_factor=0.5, shift_factor=[10, 20]),
        data_collections=SimpleNamespace(inst_cell_id=torch.tensor([1, -1])),
    )
    result = final_physical_context(placer, torch.tensor([1.26, 0.0, 2.26, 0.0]), iteration=6)
    assert result["sizing_actions"] == [
        {
            "instance_id": 0,
            "instance_name": "inst0",
            "before_cell_id": 0,
            "after_cell_id": 1,
            "before_master": "CELL_X1",
            "after_master": "CELL_X2",
            "before_area_internal": 2.0,
            "after_area_internal": 4.0,
            "area_delta_internal": 2.0,
        }
    ]
    assert result["placement_node_x"].tolist() == [13.0]
    assert result["placement_node_y"].tolist() == [25.0]


@pytest.mark.parametrize("backend", ["ecc", "openroad"])
def test_joint_boundary_writes_live_state_before_buffer_transaction(backend):
    calls = []
    placedb = SimpleNamespace()

    def apply(params, node_x, node_y):
        calls.append(
            (
                "placement_and_sizing",
                placedb.inst_cell_id.tolist(),
                node_x.tolist(),
                node_y.tolist(),
            )
        )
        placedb.last_sizing_writeback_summary = (
            {"ok": True, "accepted_count": 1, "requested_count": 1, "rejected_count": 0}
            if backend == "ecc"
            else {"applied": 1}
        )

    placedb.apply = apply
    engine = SimpleNamespace(
        placedb=placedb,
        params=SimpleNamespace(scale_factor=0.5, shift_factor=[10, 20]),
    )
    actions = ({"action_id": 0},)
    sizing = ({"instance_id": 0, "after_cell_id": 1},)
    request = BufferCommitRequest(
        version=1,
        mode="segment",
        actions=actions,
        action_digest=_action_digest(actions),
        iteration=60,
        reason="joint_buffer_periodic_commit",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path="",
        refresh_mode="full_rebuild",
        rebuild_mode="full_rebuild",
        projection_metadata={},
        runtime_refresh_summary={},
        sizing_actions=sizing,
        sizing_action_digest=_action_digest(sizing),
        sizing_cell_ids=torch.tensor([1, -1]),
        placement_node_x=torch.tensor([12.0]),
        placement_node_y=torch.tensor([24.0]),
        placement_snapshot_digest="snapshot",
    )

    def commit_buffers(actual, *, output_dir):
        assert actual is request
        calls.append(("buffer_transaction", output_dir))
        return {"status": "accepted", "accepted_action_count": 1}

    result = commit_joint_physical_actions(
        engine,
        request,
        mutation_backend=SimpleNamespace(commit_buffers=commit_buffers),
        output_dir="output",
    )
    assert calls == [
        ("placement_and_sizing", [1, -1], [1.0], [2.0]),
        ("buffer_transaction", "output"),
    ]
    assert {
        key: result[key]
        for key in (
            "status",
            "accepted_action_count",
            "sizing_action_count",
            "placement_coordinate_count",
            "atomic_across_sizing_and_buffer",
        )
    } == {
        "status": "accepted",
        "accepted_action_count": 1,
        "sizing_action_count": 1,
        "placement_coordinate_count": 1,
        "atomic_across_sizing_and_buffer": False,
    }
