"""A coincident driver/load still permits physical load isolation by a buffer."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp


def test_coincident_wire_supports_a_buffer_chain_without_creating_wire_rc():
    state = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        2,
        {
            "edges": [{"parent": 0, "child": 1, "r": 0.0, "c": 0.0}],
            "node_loads": {1: [1]},
            "coordinates": {0: [10.0, 20.0], 1: [10.0, 20.0]},
        },
    )
    parent = 0
    for action_id in range(2):
        resolved = dict(
            rebuffer_cpp.resolve_overlay_split(
                state,
                {
                    "parent_node_id": parent,
                    "child_node_id": 1,
                    "downstream_pin_ids": [1],
                    "x_dbu": 10.0,
                    "y_dbu": 20.0,
                },
            )
        )
        assert (resolved["parent_node_id"], resolved["child_node_id"]) == (parent, 1)
        assert 0.0 < resolved["split_ratio"] < 1.0
        result = rebuffer_cpp.apply_incremental_buffer_split(
            state, {**resolved, "action_id": action_id, "buffer_master_name": "BUF"}
        )
        parent = result["output_node"]

    view = rebuffer_cpp.incremental_topology_view(state)
    assert len(view["buffers"]) == 2
    assert sum(edge["r"] for edge in view["edges"]) == 0.0
    assert sum(edge["c"] for edge in view["edges"]) == 0.0
    assert all(point == [10.0, 20.0] for point in view["coordinates"].values())


@pytest.mark.parametrize(
    "override",
    [{"downstream_pin_ids": [2]}, {"x_dbu": 30.0}, {"y_dbu": 40.0}],
)
def test_coincident_wire_preserves_load_partition_and_location_checks(override):
    state = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        2,
        {
            "edges": [{"parent": 0, "child": 1, "r": 0.0, "c": 0.0}],
            "node_loads": {1: [1]},
            "coordinates": {0: [10.0, 20.0], 1: [10.0, 20.0]},
        },
    )
    with pytest.raises(ValueError, match="no active wire segment"):
        rebuffer_cpp.resolve_overlay_split(
            state, {"downstream_pin_ids": [1], "x_dbu": 10.0, "y_dbu": 20.0, **override}
        )
