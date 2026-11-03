from types import SimpleNamespace

import torch

from dreamplace.ops.placeio_ecc.persistent_timing import (
    _expanded,
    _persistent_steiner_payload,
)
from dreamplace.ops.rebuffer import rebuffer_cpp


def test_expanded_timing_lanes_pad_new_provenance_vertices():
    result = _expanded(torch.arange(5.0), 2, 6, 1)
    assert result.tolist() == [0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0]


def test_compact_payload_separates_equal_raw_steiner_ids_by_source_net():
    state = rebuffer_cpp.create_incremental_topology(
        1,
        1,
        5,
        {
            "edges": [{"parent": 1, "child": 4, "r": 1.0, "c": 1.0}],
            "node_loads": {4: [4]},
            "coordinates": {1: (10.0, 0.0), 4: (20.0, 0.0)},
        },
    )
    rebuffer_cpp.apply_incremental_buffer_split(
        state,
        {
            "action_id": 0,
            "parent_node_id": 1,
            "child_node_id": 4,
            "split_ratio": 0.5,
            "downstream_pin_ids": [4],
            "x_dbu": 15.0,
            "y_dbu": 0.0,
            "buffer_master_name": "BUF",
        },
    )
    owner = SimpleNamespace(
        new_x=torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0]),
        new_y=torch.zeros(7),
        steiner=SimpleNamespace(
            net_flat_topo_sort_start=[0, 2, 4],
            net_flat_topo_sort=[0, 4, 1, 4],
            pin_fa=[-1, -1, 0, -1, 0],
        ),
        rc=SimpleNamespace(flat_pin_to_res=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        r_unit=1.0,
        c_unit=1.0,
        payload={},
    )
    base_pydb = SimpleNamespace(pin_names=["p0", "p1"], net_names=["n0", "n1"])
    patch = {
        "buffer_count": 1,
        "slot_vertex_count": 7,
        "rc": {
            "topo_start": [0, 2, 4],
            "topo": [1, 2, 3, 6],
            "parent": [-1, -1, 1, 3, -1, -1, 3],
            "edge_resistance": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "edge_capacitance": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    }

    payload = _persistent_steiner_payload(
        owner,
        base_pydb,
        {},
        patch,
        [{"x_dbu": 15.0, "y_dbu": 0.0}],
        state,
    )

    provenance = {
        (row["source_net_id"], row["source_node_id"]): row["vertex_id"]
        for row in payload["vertex_provenance"]
    }
    assert provenance[(0, 4)] != provenance[(1, 4)]
    assert payload["pin_fa"][provenance[(0, 4)]] == 0
    assert payload["pin_fa"][provenance[(1, 4)]] == 3
    assert len(payload["pin_fa"]) == len(payload["node_x"])
