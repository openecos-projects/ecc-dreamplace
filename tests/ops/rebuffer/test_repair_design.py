from dreamplace.ops.rebuffer import rebuffer_cpp


def _design():
    return {
        "dbu": 1000,
        "num_nets": 1,
        "num_pins": 2,
        "num_nodes": 2,
        "flat_net2pin_map": [0, 1],
        "flat_net2pin_start_map": [0, 2],
        "net2driver_pin_map": [0],
        "pin2node_map": [0, 1],
        "pin2net_map": [0, 0],
        "pin_offset_x": [0.0, 0.0],
        "pin_offset_y": [0.0, 0.0],
        "node_x": [0.0, 100.0],
        "node_y": [0.0, 0.0],
        "flat_lib_pin_cap_limit": [0.01, 0.1, 0.0],
        "flat_lib_pin_slew_limit": [0.1, 0.1, 0.0],
        "flat_lib_pin_drive_resistance": [1000.0, 500.0, 0.0],
        "inst_main_id": [0, 1],
        "inst_libcell_offset": [0, 0],
        "main_id_2_cell_id_start": [0, 2, 3],
        "cell_id_2_libpin_id_start": [0, 1, 2, 3],
        "pin_2_libpin_offset": [0, 0],
    }


def _timing():
    return {
        "pin_slew_rise": [0.2, 0.2],
        "pin_slew_fall": [0.2, 0.2],
        "pin_load": [0.05, 0.001],
        "pin_cap": [0.001, 0.001],
        "pin_max_slew": [0.1, 0.0],
        "pin_max_cap": [0.01, 0.0],
        "clock_pins": [],
    }


def _steiner():
    return {
        "steiner_node_base": 2,
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 2],
        "net_flat_topo_sort": [0, 1],
        "pin_fa": [-1, 0],
        "flat_pin_to_start": [0, 1, 2],
        "flat_pin_to": [1],
        "node_x": [0.0, 100.0],
        "node_y": [0.0, 0.0],
        "edge_resistance": [0.0, 91.4],
        "node_capacitance": [0.0, 0.0],
    }


def _liberty():
    return {
        "dbu": 1000,
        "names": ["DRV", "BUF", "SINK"],
        "cell_main_id": [0, 0, 1],
        "cell_size": [1, 2, 1],
        "cell_vt": [0, 0, 0],
        "cell_area": [1.0, 2.0, 1.0],
        "cell_leakage": [0.0, 0.0, 0.0],
        "cell_input_cap": [0.001, 0.002, 0.001],
        "cell_max_cap": [0.01, 0.1, 0.0],
        "cell_max_slew": [0.1, 0.1, 0.0],
        "cell_is_buffer": [True, True, False],
        "cell_arc_start": [0, 1, 2, 3],
        "arc_sense": [1, 1, 1],
        "arc_trigger": [0, 0, 0],
        "arc_dim": [2, 2, 2, 2, 2, 2],
        "arc_trans_start": [0, 2, 4, 6],
        "arc_trans_axis": [0.01, 0.1, 0.01, 0.1, 0.01, 0.1],
        "arc_cap_start": [0, 2, 4, 6],
        "arc_cap_axis": [0.001, 0.1, 0.001, 0.1, 0.001, 0.1],
        "arc_value_start": [0, 16, 32, 48],
        "arc_values": [0.01] * 48,
    }


def _characterization():
    buffer = {
        "cell_id": 1,
        "name": "BUF",
        "main_id": 0,
        "area": 2.0,
        "leakage": 0.0,
        "input_cap": 0.002,
        "max_cap": 0.1,
        "max_slew": 0.1,
        "drive_resistance": 500.0,
        "target_load": 0.01,
        "margined_max_cap": 0.08,
    }
    return {
        "filtered_buffers": [buffer],
        "fast_buffers": [buffer],
        "target_slew_rise": 0.05,
        "target_slew_fall": 0.05,
        "buffer_lowest_drive": "BUF",
    }


def test_plan_repair_design_resizes_driver_before_inserting_buffer():
    result = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        _liberty(),
        _characterization(),
        {"resize_drivers": True, "insert_buffers": True},
    )

    assert result == {
        "status": "ok",
        "actions": [],
        "resizes": [{"node_id": 0, "target_cell_index": 1}],
        "driver_slew_violations": 1,
        "driver_cap_violations": 1,
        "load_slew_violations": 0,
        "load_cap_violations": 0,
        "repaired_nets": 1,
        "inserted_count": 0,
        "resized_count": 1,
    }


def test_plan_repair_design_skips_marginal_driver_resize():
    timing = _timing()
    timing["pin_slew_rise"] = [0.102, 0.2]
    timing["pin_slew_fall"] = [0.102, 0.2]

    result = rebuffer_cpp.plan_repair_design(
        _design(),
        timing,
        _steiner(),
        _liberty(),
        _characterization(),
        {"resize_drivers": True, "insert_buffers": True},
    )

    assert result == {
        "status": "ok",
        "actions": [
            {
                "net_id": 0,
                "driver_pin": 0,
                "downstream_pins": [1],
                "buffer_cell_index": 1,
                "segment_tree_depth": -1,
                "x_dbu": 50,
                "y_dbu": 0,
                "downstream_load": 0.01045,
                "target_load": 0.2,
                "reason": "load_slew",
            }
        ],
        "resizes": [],
        "driver_slew_violations": 1,
        "driver_cap_violations": 1,
        "load_slew_violations": 0,
        "load_cap_violations": 0,
        "repaired_nets": 1,
        "inserted_count": 1,
        "resized_count": 0,
    }


def test_plan_repair_design_inserts_buffer_when_resize_is_disabled():
    result = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        _liberty(),
        _characterization(),
        {"resize_drivers": False, "insert_buffers": True},
    )

    assert result == {
        "status": "ok",
        "actions": [
            {
                "net_id": 0,
                "driver_pin": 0,
                "downstream_pins": [1],
                "buffer_cell_index": 1,
                "segment_tree_depth": -1,
                "x_dbu": 50,
                "y_dbu": 0,
                "downstream_load": 0.01045,
                "target_load": 0.2,
                "reason": "load_slew",
            }
        ],
        "resizes": [],
        "driver_slew_violations": 1,
        "driver_cap_violations": 1,
        "load_slew_violations": 0,
        "load_cap_violations": 0,
        "repaired_nets": 1,
        "inserted_count": 1,
        "resized_count": 0,
    }


def test_repair_design_uses_driver_pin_coordinate_for_wire_split():
    design = _design()
    # Deliberately separate the owning cell center from the driver pin.  The
    # active Steiner edge is rooted at pin 0, so the repair location must be
    # the midpoint of pin 0 and pin 1, not the midpoint of the cell center
    # and pin 1.
    design["node_x"][0] = 1000.0
    design["node_y"][0] = 2000.0

    result = rebuffer_cpp.plan_repair_design(
        design,
        _timing(),
        _steiner(),
        _liberty(),
        _characterization(),
        {"resize_drivers": False, "insert_buffers": True},
    )

    assert result["actions"]
    assert result["actions"][0]["x_dbu"] == 50
    assert result["actions"][0]["y_dbu"] == 0


def test_plan_repair_design_tree_path_inserts_repeaters_until_cap_met():
    design = _design()
    design["node_x"] = [0.0, 100000.0]
    steiner = _steiner()
    steiner["node_x"] = [0.0, 100000.0]
    steiner["edge_resistance"] = [0.0, 91.4]
    liberty = _liberty()
    # Keep the repeater input slew limit out of the way so the loop is driven
    # by the 100 um wire's capacitance alone.
    liberty["cell_max_slew"] = [0.1, 1000.0, 0.0]

    result = rebuffer_cpp.plan_repair_design(
        design,
        _timing(),
        steiner,
        liberty,
        _characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "repair_net_tree": True,
        },
    )

    assert result == {
        "status": "ok",
        "actions": [
            {
                "net_id": 0,
                "driver_pin": 0,
                "downstream_pins": [1],
                "buffer_cell_index": 1,
                "segment_tree_depth": 1,
                "x_dbu": 54762,
                "y_dbu": 0,
                "downstream_load": 0.009549982,
                "target_load": 0.2,
                "reason": "wire_cap",
            },
            {
                "net_id": 0,
                "driver_pin": 0,
                "downstream_pins": [1],
                "buffer_cell_index": 1,
                "segment_tree_depth": 0,
                "x_dbu": 14550,
                "y_dbu": 0,
                "downstream_load": 0.009600068,
                "target_load": 0.2,
                "reason": "wire_cap",
            },
        ],
        "resizes": [],
        "driver_slew_violations": 1,
        "driver_cap_violations": 1,
        "load_slew_violations": 0,
        "load_cap_violations": 0,
        "repaired_nets": 1,
        "inserted_count": 2,
        "resized_count": 0,
    }
