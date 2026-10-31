from dreamplace.ops.buffer_insertion.candidate_sites import downstream_loads


def test_downstream_load_is_child_subtree_plus_wire_below_the_split():
    # Driver 0 -> Steiner 5 -> sinks 1 (2 fF) and 2 (3 fF); wire c in pF.
    rc_tree = {
        "children_by_node": {0: [5], 5: [1, 2], 1: [], 2: []},
        "node_cap": {0: 0.0, 5: 0.0, 1: 0.002, 2: 0.003},
        "edge_rc": {(0, 5): {"c": 0.004}, (5, 1): {"c": 0.001}, (5, 2): {"c": 0.002}},
    }
    candidates = [
        {"parent_node_id": 0, "child_node_ids": [5], "segment_split_ratio": 0.5},
        {"parent_node_id": 5, "child_node_ids": [2], "segment_split_ratio": 0.0},
    ]

    loads = downstream_loads(candidates, rc_tree)

    assert loads == [
        0.002 + 0.003 + 0.001 + 0.002 + 0.004 * 0.5,
        0.003 + 0.002,
    ]
