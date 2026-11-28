from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("dreamplace.ops.rebuffer.rebuffer_cpp")

from dreamplace.ops.rebuffer import (  # noqa: E402
    build_rebuffer_design_from_pydb,
    build_rebuffer_forest,
    build_rebuffer_forest_from_pydb,
)


def _pydb():
    return SimpleNamespace(
        dbu=1000,
        flat_net2pin_map=[2, 0, 1],
        flat_net2pin_start_map=[0, 3],
        net2driver_pin_map=[0],
        pin2node_map=[0, 1, 2],
        pin2net_map=[0, 0, 0],
        pin_offset_x=[0, 0, 0],
        pin_offset_y=[0, 0, 0],
        node_x=[0, 100, 200],
        node_y=[0, 100, 0],
        node_size_x=[10, 10, 10],
        node_size_y=[10, 10, 10],
        node_is_buffer=[False, False, False],
        node_is_fixed=[False, False, False],
        inst_main_id=[0, 0, 0],
        inst_libcell_offset=[0, 0, 0],
        main_id_2_cell_id_start=[0, 1],
        cell_id_2_libpin_id_start=[0, 2],
        pin_2_libpin_offset=[1, 0, 1],
        pin_direct=[-1, 1, 1],
        flat_lib_pin_cap=[0.0, 0.42],
        flat_lib_pin_rcap=[0.0, 0.42],
        flat_lib_pin_cap_limit=[0.0, 0.0],
        flat_lib_pin_slew_limit=[0.0, 0.0],
        flat_lib_pin_drive_resistance=[0.0, 0.0],
        flat_libcell_info=[["BUF_X1", 0, 1.0, 0.0]],
        flat_libcell_names=["BUF_X1"],
        net_names=["n0"],
        pin_names=["u0/Y", "u1/A", "u2/A"],
        node_names=["u0", "u1", "u2"],
        clock_pins=[],
        clock_net_names=[],
        rows=[(0, 0, 10_000, 1_000)],
        site_width=500,
        row_height=1_000,
        num_movable_nodes=3,
        buffer_main_type_index=0,
        buffer_main_type_candidate_indices=[0],
    )


def _topology():
    return {
        "status": "ok",
        "topology_source": "steiner",
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 4],
        "net_flat_topo_sort": [2, 0, 1, 5],
        "pin_fa": [2, 0, -1, -1, -1, 2],
        "steiner_node_base": 5,
    }


def test_native_forest_matches_topology_counts():
    design = build_rebuffer_design_from_pydb(_pydb())
    forest = build_rebuffer_forest(design, _topology())

    assert forest["status"] == "ok"
    assert forest["num_nets"] == 1
    assert forest["num_nodes"] == 4
    assert forest["num_pin_nodes"] == 3
    assert forest["num_steiner_nodes"] == 1
    assert forest["num_edges"] == 3
    assert forest["max_depth"] == 2
    assert forest["max_fanout"] == 2
    assert forest["net_summaries"] == [
        {
            "net_id": 0,
            "driver_pin": 0,
            "root_node": 2,
            "node_count": 4,
            "pin_node_count": 3,
            "steiner_node_count": 1,
            "edge_count": 3,
            "max_depth": 2,
            "max_fanout": 2,
            "driver_in_topology": True,
        }
    ]


def test_parent_outside_net_slice_fails_closed():
    design = build_rebuffer_design_from_pydb(_pydb())
    topology = _topology()
    topology["pin_fa"] = [5, 0, -1, -1, -1, 2]

    forest = build_rebuffer_forest(design, topology)

    assert forest["status"] == "unsupported"
    assert forest["unsupported_reasons"] == [
        "parent_not_before_child_in_topology"
    ]


def test_forest_from_pydb_carries_design_and_topology_summaries():
    forest = build_rebuffer_forest_from_pydb(
        _pydb(),
        topology=_topology(),
    )

    assert forest["status"] == "ok"
    assert forest["design_summary"]["num_nets"] == 1
    assert forest["topology_summary"] == {
        "status": "ok",
        "net_count": 1,
        "node_count": 6,
        "steiner_node_base": 5,
        "skipped_reasons": {},
    }
    np.testing.assert_array_equal(forest["net_node_ids"], [2, 0, 1, 5])
