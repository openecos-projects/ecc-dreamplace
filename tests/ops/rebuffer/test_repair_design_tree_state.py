"""Electrical state returned upstream after a native repair_design insertion."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _characterization, _design, _liberty, _steiner, _timing


def _fork():
    # One upstream wire ends at a fork with two colocated loads. A repeater
    # isolating the larger branch should make that upstream wire feasible.
    design = _design()
    design.update(
        num_pins=3,
        num_nodes=3,
        flat_net2pin_map=[0, 1, 2],
        flat_net2pin_start_map=[0, 3],
        pin2node_map=[0, 1, 2],
        pin2net_map=[0, 0, 0],
        pin_offset_x=[0.0] * 3,
        pin_offset_y=[0.0] * 3,
        node_x=[0.0, 1000.0, 1000.0],
        node_y=[0.0] * 3,
        inst_main_id=[0, 1, 1],
        inst_libcell_offset=[0, 0, 0],
        pin_2_libpin_offset=[0, 0, 0],
    )
    timing = _timing()
    timing.update(
        pin_slew_rise=[0.01] * 3,
        pin_slew_fall=[0.01] * 3,
        pin_load=[0.012189, 0.0, 0.0],
        pin_cap=[0.0, 0.007, 0.005],
        pin_max_slew=[0.0] * 3,
        pin_max_cap=[0.01, 0.0, 0.0],
    )
    steiner = {
        "steiner_node_base": 3,
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 4],
        "net_flat_topo_sort": [0, 3, 1, 2],
        "pin_fa": [-1, 3, 3, 0],
        "flat_pin_to_start": [0, 1, 1, 1, 3],
        "flat_pin_to": [3, 1, 2],
        "node_x": [0.0, 1000.0, 1000.0, 1000.0],
        "node_y": [0.0] * 4,
        "edge_resistance": [0.0, 0.0, 0.0, 0.914],
        "node_capacitance": [0.0] * 4,
    }
    liberty = _liberty()
    liberty["cell_is_buffer"][0] = False
    liberty["cell_max_slew"][1] = 1000.0
    return design, timing, steiner, liberty


@pytest.mark.parametrize("violation", ["cap", "slew"])
def test_junction_returns_repeater_input_state_to_upstream_wire(violation):
    design, timing, steiner, liberty = _fork()
    if violation == "slew":
        timing["pin_max_cap"][0] = liberty["cell_max_cap"][0] = 0.1
        timing["pin_max_slew"] = [0.0, 0.002, 50.0]

    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {"resize_drivers": False, "repair_net_tree": True},
    )

    assert result["status"] == "ok"
    assert [
        (action["reason"], action["downstream_pins"], action["x_dbu"])
        for action in result["actions"]
    ] == [(f"junction_{violation}", [1], 1000)]
    assert result["actions"][0]["downstream_load"] == pytest.approx(0.007)
    assert result["inserted_count"] == 1


def test_tree_after_resize_matches_tree_started_from_the_new_driver():
    design, timing, steiner, liberty = _design(), _timing(), _steiner(), _liberty()
    timing["pin_load"][0] = 0.005189
    timing["pin_cap"][1] = 0.005
    timing["pin_max_slew"][1] = 0.005
    steiner["node_x"][1] = design["node_x"][1] = 1000.0
    # Cell 0's output slew implies R=1, cell 1's implies R=0.01. The
    # stronger cell satisfies the load's slew budget without another buffer.
    liberty["arc_values"][8:16] = [0.1] * 8
    liberty["arc_values"][24:32] = [0.001] * 8
    characterization = _characterization()
    characterization["filtered_buffers"][0]["drive_resistance"] = 0.01
    config = {"repair_net_tree": True, "r_unit": 0.0}

    resized = rebuffer_cpp.plan_repair_design(
        design, timing, steiner, liberty, characterization, config
    )
    design["inst_libcell_offset"][0] = 1
    timing["pin_slew_rise"][0] = timing["pin_slew_fall"][0] = 0.001
    from_new_driver = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {**config, "resize_drivers": False},
    )

    assert resized["status"] == from_new_driver["status"] == "ok"
    assert resized["resizes"] == [{"node_id": 0, "target_cell_index": 1}]
    assert resized["actions"] == from_new_driver["actions"] == []
