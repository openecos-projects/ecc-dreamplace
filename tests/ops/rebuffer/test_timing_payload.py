"""Metadata gathers preserve path arc identities and external output loads."""

from types import SimpleNamespace

import numpy as np
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import current_tree_topology_payload
from dreamplace.ops.rebuffer.timing_model_payload import _pin_pair_arc_mapping


def test_pin_pair_mapping_keeps_parallel_arcs_and_excludes_net_predecessors():
    db = SimpleNamespace(
        pin_pred_start=[0, 0, 5, 6],
        pin_pred_pin=[0, 0, 0, 2, 0, 1],
        pin_pred_arc_id=[4, 2, 4, 7, -1, -1],
    )

    actual = _pin_pair_arc_mapping(db)

    assert [value.tolist() for value in actual] == [[[0, 1], [2, 1]], [0, 2, 3], [2, 4, 7]]
    assert all(value.dtype.is_floating_point is False for value in actual)
    assert all(value.is_contiguous() for value in actual)


def test_current_tree_topology_reuses_tp_csr_ids_and_coordinates():
    model = SimpleNamespace(
        steiner=SimpleNamespace(
            net_flat_topo_sort_start=[0, 3, 5],
            net_flat_topo_sort=[0, 5, 1, 2, 3],
            pin_fa=[-1, 5, -1, 2, -1, 0],
        ),
        new_x=[11, 22, 33, 44, 55, 66],
        new_y=[12, 23, 34, 45, 56, 67],
    )
    result = current_tree_topology_payload(SimpleNamespace(pin_names=["p"] * 4), model, [1, 0])
    actual = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }

    assert actual == {
        "status": "ok",
        "topology_source": "active_timing_propagation_steiner",
        "steiner_node_base": 4,
        "net_ids": [1, 0],
        "net_flat_topo_sort_start": [0, 2, 5],
        "net_flat_topo_sort": [2, 3, 0, 5, 1],
        "pin_fa": [-1, 5, -1, 2, -1, 0],
        "node_x": {2: 33.0, 3: 44.0, 0: 11.0, 5: 66.0, 1: 22.0},
        "node_y": {2: 34.0, 3: 45.0, 0: 12.0, 5: 67.0, 1: 23.0},
    }


def test_pin_capacitance_resolves_current_master_and_transition_specific_output_loads():
    db = SimpleNamespace(
        pin2node_map=[0, 0, 1, 1, 2],
        inst_main_id=[0, 0, -1],
        inst_libcell_offset=[1, 0, 0],
        main_id_2_cell_id_start=[0, 2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        pin_2_libpin_offset=[0, 1, 0, 1, -1],
        end_points=[4, 3],
        flat_lib_pin_cap=[0.1, 0.0, 0.2, 0.0],
        flat_lib_pin_rcap=[0.11, 0.0, 0.21, 0.0],
        flat_lib_pin_fcap=[0.12, 0.0, 0.22, 0.0],
        flat_lib_pin_cap_limit=[0.0, 1.0, 0.0, 2.0],
        flat_lib_pin_slew_limit=[10.0, 20.0, 30.0, 40.0],
        outcaps=[0.3, 99.0],
        out_rcaps=[0.31, 99.0],
        out_fcaps=[0.32, 99.0],
    )

    result = rebuffer_cpp.timing_pin_capacitance(db)

    expected = {
        "caps": [0.2, 0.0, 0.1, 0.0, 0.3],
        "rcaps": [0.21, 0.0, 0.11, 0.0, 0.31],
        "fcaps": [0.22, 0.0, 0.12, 0.0, 0.32],
        "max_cap": [0.0, 2.0, 0.0, 1.0, 0.0],
        "max_slew": [30.0, 40.0, 10.0, 20.0, 0.0],
    }
    assert result.keys() == expected.keys()
    for key, values in expected.items():
        np.testing.assert_array_equal(result[key], values)
        assert result[key].strides == (8,)
