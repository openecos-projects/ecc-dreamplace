"""Carrier vertices and per-net mutable node ids are separate identity domains."""

from types import SimpleNamespace

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp


@pytest.mark.parametrize("foreign_source_ids", [False, True])
def test_descendant_load_uses_carrier_identity_before_an_aliasing_mutable_node(foreign_source_ids):
    base = SimpleNamespace(
        pin_names=["drv:Y", "buf:A", "buf:Y", "load:A", "branch:A"],
        net_names=["n0", "n1"],
        net2driver_pin_map=[0, 2],
        pin2node_map=[0, 1, 1, 2, 3],
        pin2net_map=[0, 0, 1, 1, 0],
        flat_libcell_names=["BUF"],
    )
    upstream = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        7,
        {
            "edges": [
                {"parent": 0, "child": 6, "r": 1.0, "c": 0.01},
                {"parent": 6, "child": 1, "r": 1.0, "c": 0.01},
                {"parent": 0, "child": 3, "r": 1.0, "c": 0.01},
                {"parent": 3, "child": 4, "r": 1.0, "c": 0.01},
            ],
            "node_loads": {1: [1], 4: [4]},
            "coordinates": {
                0: [0.0, 0.0],
                6: [50.0, 0.0],
                1: [100.0, 0.0],
                3: [0.0, 50.0],
                4: [0.0, 100.0],
            },
        },
    )
    downstream = rebuffer_cpp.create_incremental_topology(
        1,
        2,
        5,
        {
            "edges": [{"parent": 2, "child": 3, "r": 1.0, "c": 0.01}],
            "node_loads": {3: [3]},
            "coordinates": {2: [100.0, 0.0], 3: [200.0, 0.0]},
        },
    )
    # Carrier vertex 3 is the downstream terminal. Mutable node 3 also
    # exists upstream, where it is an unrelated internal wire vertex.
    payload = {
        "original_net_by_net": [0, 1],
        "state_node_by_vertex": [0, 1, 2, 3, 4, -1, 6],
        "state_net_by_vertex": [0, 0, 1, 1, 0, -1, 0],
        "state_provenance": [
            {"partition_id": p, "vertex_id": v, "state_net_id": n, "state_node_id": s}
            for p, v, n, s in [(0, 0, 0, 0), (0, 1, 0, 1), (0, 6, 0, 6), (1, 3, 1, 3)]
        ],
        "physical": {
            "pin2node_map": base.pin2node_map,
            "pin2net_map": base.pin2net_map,
            "net2driver_pin_map": base.net2driver_pin_map,
        },
    }
    action = {
        "net_id": 0,
        "driver_pin": 0,
        "buffer_cell_index": 0,
        "parent_node_id": 0,
        "child_node_id": 6,
        "source_parent_node_id": 0,
        "source_child_node_id": 6,
        "downstream_pins": [3],
        "x_dbu": 25.0,
        "y_dbu": 0.0,
    }
    if foreign_source_ids:
        # The active TP tree and the exported carrier can number Steiner
        # vertices differently. These source hints alias a different net.
        payload["state_node_by_vertex"] += [-1, 2, 3]
        payload["state_net_by_vertex"] += [-1, 1, 1]
        payload["state_provenance"] += [
            {"partition_id": 1, "vertex_id": v, "state_net_id": 1, "state_node_id": n}
            for v, n in [(8, 2), (9, 3)]
        ]
        action.update(
            parent_node_id=8,
            child_node_id=9,
            source_parent_node_id=8,
            source_child_node_id=9,
        )
    result = rebuffer_cpp.project_overlay_actions(base, payload, [upstream, downstream], [action])[
        0
    ]
    expected = {
        "net_id": 0,
        "downstream_pins": [1],
        "downstream_pin_names": ["buf/A"],
        "parent_node_id": 0,
        "child_node_id": 6,
        "projection_cross_partition_boundary": True,
    }
    assert {key: result[key] for key in expected} == expected
