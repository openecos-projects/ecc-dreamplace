from dreamplace.ops.rebuffer import rebuffer_cpp


def _dp_inputs():
    design = {
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
        "node_x": [0.0, 1000.0],
        "node_y": [0.0, 0.0],
        "flat_lib_pin_cap_limit": [0.01, 0.1, 0.0],
        "flat_lib_pin_slew_limit": [1000.0, 1000.0, 0.0],
        "flat_lib_pin_drive_resistance": [1000.0, 500.0, 0.0],
        "inst_main_id": [0, 1],
        "inst_libcell_offset": [0, 0],
        "main_id_2_cell_id_start": [0, 2, 3],
        "cell_id_2_libpin_id_start": [0, 1, 2, 3],
        "pin_2_libpin_offset": [0, 0],
    }
    steiner = {
        "steiner_node_base": 2,
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 2],
        "net_flat_topo_sort": [0, 1],
        "pin_fa": [-1, 0],
        "flat_pin_to_start": [0, 1, 2],
        "flat_pin_to": [1],
        "node_x": [0.0, 1000.0],
        "node_y": [0.0, 0.0],
        "edge_resistance": [0.0, 0.914],
        "node_capacitance": [0.0, 0.0],
    }
    timing = {
        "pin_slew_rise": [0.2, 0.2],
        "pin_slew_fall": [0.2, 0.2],
        "pin_load": [0.050189, 0.05],
        "pin_cap": [0.001, 0.05],
        "pin_max_slew": [1000.0, 0.0],
        "pin_max_cap": [0.01, 0.0],
        "pin_slack": [0.0, -0.2],
        "clock_pins": [],
    }
    liberty = {
        "dbu": 1000,
        "names": ["DRV", "BUF"],
        "cell_main_id": [0, 1],
        "cell_size": [1, 1],
        "cell_vt": [0, 0],
        "cell_area": [1.0, 1.0],
        "cell_leakage": [0.0, 0.0],
        "cell_input_cap": [0.001, 0.002],
        "cell_max_cap": [0.01, 0.1],
        "cell_max_slew": [1000.0, 1000.0],
        "cell_is_buffer": [False, True],
        "cell_arc_start": [0, 1, 2],
        "arc_sense": [1, 1],
        "arc_trigger": [0, 0],
        "arc_dim": [2, 2, 2, 2],
        "arc_trans_start": [0, 2, 4],
        "arc_trans_axis": [0.01, 0.1, 0.01, 0.1],
        "arc_cap_start": [0, 2, 4],
        "arc_cap_axis": [0.001, 0.1, 0.001, 0.1],
        "arc_value_start": [0, 16, 32],
        "arc_values": [
            # DRV: delay grows with the load capacitance.
            0.005, 0.05, 0.005, 0.05,
            0.005, 0.05, 0.005, 0.05,
            0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01,
            # BUF: flat 0.005 delay, so isolating the load always wins.
            0.005, 0.005, 0.005, 0.005,
            0.005, 0.005, 0.005, 0.005,
            0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01,
        ],
    }
    buffer = {
        "cell_id": 1,
        "name": "BUF",
        "main_id": 1,
        "area": 1.0,
        "leakage": 0.0,
        "input_cap": 0.002,
        "max_cap": 0.1,
        "max_slew": 1000.0,
        "drive_resistance": 500.0,
        "target_load": 0.01,
        "margined_max_cap": 0.08,
    }
    characterization = {
        "filtered_buffers": [buffer],
        "fast_buffers": [buffer],
        "target_slew_rise": 0.05,
        "target_slew_fall": 0.05,
        "buffer_lowest_drive": "BUF",
    }
    return design, timing, steiner, liberty, characterization


def test_rank_path_driver_pins_uses_native_load_delay_order():
    design, timing, steiner, liberty, _characterization = _dp_inputs()
    del steiner

    ranked = rebuffer_cpp.rank_path_driver_pins(
        design,
        timing,
        liberty,
        [1, 0],
    )

    assert ranked == [0]


def test_select_buffer_set_filtered_falls_back_to_filtered_buffers():
    from dreamplace.ops.rebuffer.rebuffer_timing import _select_buffer_set

    characterization = {
        "status": "ok",
        "filtered_buffers": [{"cell_id": 0}],
        "fast_buffers": [{"cell_id": 1}],
    }

    selected = _select_buffer_set(
        characterization, {"buffer_set": "filtered"}
    )

    assert selected["fast_buffers"] == []
    assert selected["filtered_buffers"] == [{"cell_id": 0}]
    assert characterization["fast_buffers"] == [{"cell_id": 1}]


