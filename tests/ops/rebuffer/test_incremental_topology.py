import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp


@pytest.fixture
def branch_state():
    return rebuffer_cpp.create_incremental_topology(
        7,
        0,
        10,
        {
            "edges": [
                {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
                {"parent": 1, "child": 2, "r": 5.0, "c": 1.0},
                {"parent": 1, "child": 3, "r": 6.0, "c": 1.0},
            ],
            "node_loads": {2: [20], 3: [30]},
        },
    )


def _split(action_id, parent, child, loads):
    return {
        "action_id": action_id,
        "parent_node_id": parent,
        "child_node_id": child,
        "split_ratio": 0.4,
        "downstream_pin_ids": loads,
    }


def test_import_existing_steiner_arrays_keeps_geometry_rc_and_pin_identity():
    state = rebuffer_cpp.incremental_topology_from_steiner(
        {
            "steiner_node_base": 2,
            "net_flat_topo_sort_start": [0, 2],
            "net_flat_topo_sort": [0, 1],
            "pin_fa": [0, 0],
            "flat_pin_to_start": [0, 1, 1],
            "flat_pin_to": [1],
            "node_x": [10.0, 110.0],
            "node_y": [0.0, 0.0],
            "edge_resistance": [0.0, 10.0],
            "node_capacitance": [0.0, 0.0],
        },
        0,
        0,
        5.0,
        1.0,
    )
    expected = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        2,
        {
            "edges": [{"parent": 0, "child": 1, "r": 10.0, "c": 2.0}],
            "node_loads": {1: [1]},
            "coordinates": {0: [10.0, 0.0], 1: [110.0, 0.0]},
        },
    )
    assert rebuffer_cpp.incremental_topology_view(state) == (
        rebuffer_cpp.incremental_topology_view(expected)
    )


def test_incremental_topology_partitions_only_selected_branch(branch_state):
    result = rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 1, 2, [20]))
    view = rebuffer_cpp.incremental_topology_view(branch_state)

    assert result["epoch"] == 1
    assert view["node_loads"] == {2: [20], 3: [30]}
    assert view["partitions"] == {
        -1: {
            "driver_node_id": 0,
            "node_ids": [0, 1, 3, result["input_node"]],
            "load_pin_ids": [result["input_node"], 30],
            "edges": [
                {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
                {"parent": 1, "child": 3, "r": 6.0, "c": 1.0},
                {"parent": 1, "child": result["input_node"], "r": 2.0, "c": 0.4},
            ],
        },
        101: {
            "driver_node_id": result["output_node"],
            "node_ids": [2, result["output_node"]],
            "load_pin_ids": [20],
            "edges": [
                {"parent": result["output_node"], "child": 2, "r": 3.0, "c": 0.6},
            ],
        },
    }
    assert rebuffer_cpp.validate_incremental_topology(branch_state)["valid"]


def test_incremental_topology_keeps_buffer_master_metadata(branch_state):
    action = _split(101, 1, 2, [20])
    action["buffer_master_name"] = "BUF_X4"
    rebuffer_cpp.apply_incremental_buffer_split(branch_state, action)

    view = rebuffer_cpp.incremental_topology_view(branch_state)
    assert view["buffers"] == [
        {
            "action_id": 101,
            "buffer_master_name": "BUF_X4",
            "input_node": view["buffers"][0]["input_node"],
            "output_node": view["buffers"][0]["output_node"],
            "original_parent": 1,
            "original_child": 2,
            "split_ratio": 0.4,
        }
    ]


def test_remap_incremental_topology_shifts_non_pin_namespace(branch_state):
    action = _split(101, 1, 2, [20])
    action["buffer_master_name"] = "BUF_X4"
    rebuffer_cpp.apply_incremental_buffer_split(branch_state, action)

    original = rebuffer_cpp.incremental_topology_view(branch_state)
    remapped = rebuffer_cpp.remap_incremental_topology(branch_state, 2, 4)
    current = rebuffer_cpp.incremental_topology_view(remapped)

    assert current["net_id"] == original["net_id"]
    assert current["driver_pin_id"] == original["driver_pin_id"]
    assert current["node_loads"] == {
        int(node) + 4 if int(node) >= 2 else int(node): pins
        for node, pins in original["node_loads"].items()
    }
    assert current["coordinates"] == {
        int(node) + 4 if int(node) >= 2 else int(node): point
        for node, point in original["coordinates"].items()
    }
    assert current["edges"] == [
        {
            **edge,
            "parent": int(edge["parent"]) + (4 if int(edge["parent"]) >= 2 else 0),
            "child": int(edge["child"]) + (4 if int(edge["child"]) >= 2 else 0),
        }
        for edge in original["edges"]
    ]
    original_buffer = original["buffers"][0]
    remapped_buffer = current["buffers"][0]
    assert remapped_buffer["input_node"] == original_buffer["input_node"] + 4
    assert remapped_buffer["output_node"] == original_buffer["output_node"] + 4
    assert remapped_buffer["original_parent"] == original_buffer["original_parent"]
    assert remapped_buffer["original_child"] == original_buffer["original_child"] + 4


def test_remap_incremental_topology_preserves_source_identity_with_explicit_mapping(branch_state):
    original = rebuffer_cpp.incremental_topology_view(branch_state)
    remapped = rebuffer_cpp.remap_incremental_topology(
        branch_state,
        2,
        4,
        19,
        -1,
        {2: 50, 3: 51},
    )
    current = rebuffer_cpp.incremental_topology_view(remapped)

    assert current["net_id"] == 19
    assert current["source_net_id"] == 7
    assert current["source_node_by_node"] == {0: 0, 1: 1, 50: 2, 51: 3}
    assert current["edges"] == [
        {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
        {"parent": 1, "child": 50, "r": 5.0, "c": 1.0},
        {"parent": 1, "child": 51, "r": 6.0, "c": 1.0},
    ]
    assert current["node_loads"] == {50: [20], 51: [30]}
    assert original["source_node_by_node"] == {0: 0, 1: 1, 2: 2, 3: 3}


def test_incremental_topology_supports_same_net_chain_and_reverse_remove(branch_state):
    first = rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 0, 1, [20, 30]))
    second = rebuffer_cpp.apply_incremental_buffer_split(
        branch_state, _split(102, first["output_node"], 1, [20, 30])
    )

    assert second["epoch"] == 2
    rebuffer_cpp.remove_incremental_buffer(branch_state, 102)
    buffers = rebuffer_cpp.incremental_topology_view(branch_state)["buffers"]
    assert [buffer["action_id"] for buffer in buffers] == [101]
    rebuffer_cpp.remove_incremental_buffer(branch_state, 101)
    view = rebuffer_cpp.incremental_topology_view(branch_state)
    assert view["edges"] == [
        {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
        {"parent": 1, "child": 2, "r": 5.0, "c": 1.0},
        {"parent": 1, "child": 3, "r": 6.0, "c": 1.0},
    ]
    assert view["node_loads"] == {2: [20], 3: [30]}


def test_incremental_topology_checkpoint_restores_partition(branch_state):
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(branch_state)
    rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 0, 1, [20, 30]))

    rebuffer_cpp.restore_incremental_topology(branch_state, checkpoint)
    view = rebuffer_cpp.incremental_topology_view(branch_state)
    assert view["epoch"] == 0
    assert view["buffers"] == []
    assert view["node_loads"] == {2: [20], 3: [30]}


def test_incremental_topology_checkpoint_redoes_after_last_buffer_was_removed(
    branch_state,
):
    result = rebuffer_cpp.apply_incremental_buffer_split(
        branch_state, _split(101, 0, 1, [20, 30])
    )
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(branch_state)
    expected = rebuffer_cpp.incremental_topology_view(branch_state)

    rebuffer_cpp.remove_incremental_buffer(branch_state, 101)
    rebuffer_cpp.restore_incremental_topology(branch_state, checkpoint)

    restored = rebuffer_cpp.incremental_topology_view(branch_state)
    assert restored == expected
    assert restored["buffers"][0]["input_node"] == result["input_node"]


def test_incremental_topology_checkpoint_rebuilds_after_history_diverges(
    branch_state,
):
    rebuffer_cpp.apply_incremental_buffer_split(
        branch_state, _split(101, 0, 1, [20, 30])
    )
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(branch_state)
    rebuffer_cpp.remove_incremental_buffer(branch_state, 101)
    rebuffer_cpp.apply_incremental_buffer_split(
        branch_state, _split(202, 0, 1, [20, 30])
    )

    rebuffer_cpp.restore_incremental_topology(branch_state, checkpoint)
    buffers = rebuffer_cpp.incremental_topology_view(branch_state)["buffers"]
    assert [buffer["action_id"] for buffer in buffers] == [101]


def test_incremental_topology_rejects_stale_edge_and_cross_branch_load(branch_state):
    rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 0, 1, [20, 30]))
    with pytest.raises(ValueError, match="active_edge_not_found"):
        rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(102, 0, 1, [30]))
    with pytest.raises(ValueError, match="outside_edge_subtree"):
        rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(103, 1, 2, [30]))


@pytest.mark.parametrize("loads", [[], [20, 20], [20]])
def test_rejected_partition_preserves_entire_native_state(branch_state, loads):
    before = rebuffer_cpp.incremental_topology_view(branch_state)
    with pytest.raises(ValueError, match="partition"):
        rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 0, 1, loads))
    assert rebuffer_cpp.incremental_topology_view(branch_state) == before


def test_rc_conservation_and_cell_edge_cannot_be_split(branch_state):
    before = rebuffer_cpp.incremental_topology_view(branch_state)
    first = rebuffer_cpp.apply_incremental_buffer_split(branch_state, _split(101, 0, 1, [20, 30]))
    view = rebuffer_cpp.incremental_topology_view(branch_state)
    for quantity in ("r", "c"):
        assert sum(edge[quantity] for edge in view["edges"]) == pytest.approx(
            sum(edge[quantity] for edge in before["edges"])
        )
    with pytest.raises(ValueError, match="buffer_cell_edge"):
        rebuffer_cpp.apply_incremental_buffer_split(
            branch_state, _split(102, first["input_node"], first["output_node"], [20, 30])
        )
    assert rebuffer_cpp.incremental_topology_view(branch_state) == view


def test_coordinates_follow_exact_ratio_and_restore():
    state = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        2,
        {
            "edges": [{"parent": 0, "child": 1, "r": 10.0, "c": 2.0}],
            "node_loads": {1: [1]},
            "coordinates": {0: [0, 0], 1: [100, 0]},
        },
    )
    before = rebuffer_cpp.incremental_topology_view(state)
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(state)
    action = {**_split(0, 0, 1, [1]), "x_dbu": 60, "y_dbu": 0}
    with pytest.raises(ValueError, match="location"):
        rebuffer_cpp.apply_incremental_buffer_split(state, action)
    assert rebuffer_cpp.incremental_topology_view(state) == before
    action["x_dbu"] = 40
    result = rebuffer_cpp.apply_incremental_buffer_split(state, action)
    view = rebuffer_cpp.incremental_topology_view(state)
    assert view["coordinates"][result["input_node"]] == [40.0, 0.0]
    assert view["coordinates"][result["output_node"]] == [40.0, 0.0]
    rebuffer_cpp.restore_incremental_topology(state, checkpoint)
    assert rebuffer_cpp.incremental_topology_view(state) == before
