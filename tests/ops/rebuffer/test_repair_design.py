from copy import deepcopy

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
        # The production profile plans in reverse topological order and
        # refuses to guess without the native pin rank.
        "pin_rank": [0, 0],
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


def _post_resize_liberty():
    liberty = deepcopy(_liberty())
    liberty["cell_is_buffer"][0] = False
    liberty["names"].append("BUF_SMALL")
    liberty["cell_main_id"].append(0)
    liberty["cell_size"].append(1)
    liberty["cell_vt"].append(0)
    liberty["cell_area"].append(0.5)
    liberty["cell_leakage"].append(0.0)
    liberty["cell_input_cap"].append(0.001)
    liberty["cell_max_cap"].append(0.02)
    liberty["cell_max_slew"].append(0.1)
    liberty["cell_is_buffer"].append(True)
    liberty["cell_arc_start"] = [0, 1, 2, 3, 4]
    liberty["arc_sense"].append(1)
    liberty["arc_trigger"].append(0)
    liberty["arc_dim"].extend([2, 2])
    liberty["arc_trans_start"] = [0, 2, 4, 6, 8]
    liberty["arc_trans_axis"].extend([0.01, 0.1])
    liberty["arc_cap_start"] = [0, 2, 4, 6, 8]
    liberty["arc_cap_axis"].extend([0.001, 0.1])
    liberty["arc_value_start"] = [0, 16, 32, 48, 64]
    liberty["arc_values"].extend([0.005] * 16)
    return liberty


def _post_resize_characterization():
    characterization = deepcopy(_characterization())
    small = deepcopy(characterization["filtered_buffers"][0])
    small.update(
        {
            "cell_id": 3,
            "name": "BUF_SMALL",
            "area": 0.5,
            "input_cap": 0.001,
            "max_cap": 0.02,
            "max_slew": 0.1,
        }
    )
    characterization["filtered_buffers"] = [
        characterization["filtered_buffers"][0],
        small,
    ]
    characterization["fast_buffers"] = list(characterization["filtered_buffers"])
    return characterization


def _two_net_design():
    design = _design()
    design.update(
        {
            "num_nets": 2,
            "num_pins": 4,
            "num_nodes": 4,
            "flat_net2pin_map": [0, 1, 2, 3],
            "flat_net2pin_start_map": [0, 2, 4],
            "net2driver_pin_map": [0, 2],
            "pin2node_map": [0, 1, 2, 3],
            "pin2net_map": [0, 0, 1, 1],
            "node_x": [0.0, 100.0, 0.0, 100.0],
            "node_y": [0.0, 0.0, 100.0, 100.0],
            "inst_main_id": [0, 1, 0, 1],
            "inst_libcell_offset": [0, 0, 0, 0],
            "pin_2_libpin_offset": [0, 0, 0, 0],
        }
    )
    return design


def _two_net_timing():
    timing = _timing()
    timing.update(
        {
            "pin_slew_rise": [0.102, 0.2, 0.102, 0.2],
            "pin_slew_fall": [0.102, 0.2, 0.102, 0.2],
            "pin_load": [0.05, 0.001, 0.05, 0.001],
            "pin_cap": [0.001, 0.001, 0.001, 0.001],
            "pin_max_slew": [0.1, 0.0, 0.1, 0.0],
            "pin_max_cap": [0.01, 0.0, 0.01, 0.0],
        }
    )
    return timing


def _two_net_steiner():
    steiner = _steiner()
    steiner.update(
        {
            "steiner_node_base": 4,
            "net_ids": [0, 1],
            "net_flat_topo_sort_start": [0, 2, 4],
            "net_flat_topo_sort": [0, 1, 2, 3],
            "pin_fa": [-1, 0, -1, 2],
            "flat_pin_to_start": [0, 1, 2, 3, 4],
            "flat_pin_to": [1, 0, 3, 2],
            "node_x": [0.0, 100.0, 0.0, 100.0],
            "node_y": [0.0, 0.0, 100.0, 100.0],
            "edge_resistance": [0.0, 91.4, 0.0, 91.4],
            "node_capacitance": [0.0, 0.0, 0.0, 0.0],
        }
    )
    return steiner


def _resize_chain_design():
    design = _design()
    design.update(
        {
            "num_nets": 2,
            "num_pins": 5,
            "num_nodes": 3,
            "flat_net2pin_map": [0, 2, 3, 4],
            "flat_net2pin_start_map": [0, 2, 4],
            "net2driver_pin_map": [0, 3],
            "pin2node_map": [0, 0, 1, 1, 2],
            "pin2net_map": [0, -1, 0, 1, 1],
            "node_x": [0.0, 100.0, 200.0],
            "node_y": [0.0, 0.0, 0.0],
            "flat_lib_pin_cap": [0.001, 0.001, 0.002, 0.002, 0.001],
            "inst_main_id": [0, 0, 1],
            "inst_libcell_offset": [0, 0, 0],
            "main_id_2_cell_id_start": [0, 2, 3],
            "cell_id_2_libpin_id_start": [0, 2, 4, 5],
            "pin_2_libpin_offset": [1, 0, 0, 1, 0],
        }
    )
    return design


def _resize_chain_timing():
    return {
        "pin_slew_rise": [0.01, 0.01, 0.2, 0.2, 0.2],
        "pin_slew_fall": [0.01, 0.01, 0.2, 0.2, 0.2],
        "pin_load": [0.0095, 0.0, 0.001, 0.005, 0.001],
        "pin_cap": [0.001, 0.001, 0.001, 0.001, 0.001],
        "pin_max_slew": [0.1, 0.0, 0.0, 0.1, 0.0],
        "pin_max_cap": [0.01, 0.0, 0.0, 0.01, 0.0],
        "pin_rank": [0, 0, 0, 10, 0],
        "clock_pins": [],
    }


def _resize_chain_liberty():
    liberty = _liberty()
    values = [0.01] * 48
    # The small driver has an input-slew-dependent transition: a quiet
    # upstream input is legal, while a 0.2 input slew remains violating.  The
    # larger cell keeps the original fast transition used by the resize test.
    values[8:12] = [0.01, 0.01, 0.2, 0.2]
    values[12:16] = [0.01, 0.01, 0.2, 0.2]
    liberty["arc_values"] = values
    return liberty


def _annotation_chain_timing():
    timing = _resize_chain_timing()
    timing["pin_slew_rise"][2] = 0.2
    timing["pin_slew_fall"][2] = 0.2
    timing["pin_max_slew"][2] = 0.05
    return timing


def _resize_chain_steiner():
    return {
        "steiner_node_base": 5,
        "net_ids": [0, 1],
        "net_flat_topo_sort_start": [0, 2, 4],
        "net_flat_topo_sort": [0, 2, 3, 4],
        "pin_fa": [-1, -1, 0, -1, 3],
        "flat_pin_to_start": [0, 1, 2, 3, 4, 5],
        "flat_pin_to": [2, 0, 4, 3, 0],
        "node_x": [0.0, 0.0, 100.0, 200.0, 300.0],
        "node_y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "edge_resistance": [0.0, 0.0, 91.4, 0.0, 91.4],
        "node_capacitance": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_repair_design_reverse_topological_order_is_native_and_opt_in():
    design = _two_net_design()
    timing = _two_net_timing()
    steiner = _two_net_steiner()
    base_config = {"resize_drivers": False, "insert_buffers": True}

    default_result = rebuffer_cpp.plan_repair_design(
        design, timing, steiner, _liberty(), _characterization(), base_config
    )
    assert [action["net_id"] for action in default_result["actions"]] == [0, 1]

    timing["pin_rank"] = [0, 0, 10, 0]
    reverse_result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        _liberty(),
        _characterization(),
        {**base_config, "repair_design_reverse_topological": True},
    )
    assert [action["net_id"] for action in reverse_result["actions"]] == [1, 0]


def test_placement_profile_enables_openroad_repair_design_order_by_default():
    """The production placement profile runs the OpenROAD order, not the legacy one."""
    from dreamplace.ops.rebuffer.repair_design import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["repair_design_reverse_topological"] is True
    assert DEFAULT_CONFIG["repair_design_local_update"] is True
    assert DEFAULT_CONFIG["repair_design_load_slew_annotation"] is True
    assert DEFAULT_CONFIG["repair_design_resize_to_target_slew"] is False
    # repair_design has no global action budget; the tree DP keeps a
    # per-net safety bound under a separate name.
    assert "max_actions" not in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["tree_max_buffers"] == 1024


def test_repair_design_reverse_topological_mode_fails_closed_without_pin_rank():
    result = rebuffer_cpp.plan_repair_design(
        _two_net_design(),
        _two_net_timing(),
        _two_net_steiner(),
        _liberty(),
        _characterization(),
        {"repair_design_reverse_topological": True},
    )

    assert result == {
        "status": "unsupported",
        "unsupported_reasons": ["missing_repair_design_pin_rank"],
    }


def test_repair_design_local_update_propagates_resize_cap_to_upstream_net():
    design = _resize_chain_design()
    timing = _resize_chain_timing()
    steiner = _resize_chain_steiner()
    liberty = _resize_chain_liberty()
    characterization = _characterization()

    without_local_update = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {
            "repair_design_reverse_topological": True,
            "resize_drivers": True,
            "insert_buffers": True,
        },
    )
    assert without_local_update["resizes"] == [
        {"node_id": 1, "target_cell_index": 1}
    ]
    assert without_local_update["actions"] == []

    with_local_update = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {
            "repair_design_reverse_topological": True,
            "repair_design_local_update": True,
            "resize_drivers": True,
            "insert_buffers": True,
        },
    )
    assert with_local_update["resizes"] == [
        {"node_id": 1, "target_cell_index": 1}
    ]
    assert [action["net_id"] for action in with_local_update["actions"]] == [0]


def test_repair_design_local_update_requires_reverse_topological_order():
    result = rebuffer_cpp.plan_repair_design(
        _resize_chain_design(),
        _resize_chain_timing(),
        _resize_chain_steiner(),
        _liberty(),
        _characterization(),
        {"repair_design_local_update": True},
    )

    assert result == {
        "status": "unsupported",
        "unsupported_reasons": [
            "repair_design_local_update_requires_reverse_topological"
        ],
    }


def test_repair_design_load_slew_annotation_suppresses_downstream_resize():
    design = _resize_chain_design()
    timing = _annotation_chain_timing()
    steiner = _resize_chain_steiner()
    liberty = _resize_chain_liberty()
    characterization = _characterization()

    without_annotation = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {
            "repair_design_reverse_topological": True,
            "resize_drivers": True,
            "insert_buffers": True,
        },
    )
    assert without_annotation["resizes"] == [
        {"node_id": 1, "target_cell_index": 1}
    ]

    with_annotation = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {
            "repair_design_reverse_topological": True,
            "repair_design_load_slew_annotation": True,
            "resize_drivers": True,
            "insert_buffers": True,
        },
    )
    assert with_annotation["resizes"] == []
    assert with_annotation["actions"] == []


def test_repair_design_load_slew_annotation_requires_reverse_topological_order():
    result = rebuffer_cpp.plan_repair_design(
        _resize_chain_design(),
        _annotation_chain_timing(),
        _resize_chain_steiner(),
        _resize_chain_liberty(),
        _characterization(),
        {"repair_design_load_slew_annotation": True},
    )

    assert result == {
        "status": "unsupported",
        "unsupported_reasons": [
            "repair_design_load_slew_annotation_requires_reverse_topological"
        ],
    }


def test_repair_design_openroad_resize_rule_considers_smallest_legal_equivalent_cell():
    legacy = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        _liberty(),
        _characterization(),
        {
            "resize_drivers": True,
            "insert_buffers": False,
        },
    )
    assert legacy["resizes"] == [{"node_id": 0, "target_cell_index": 1}]

    openroad_rule = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        _liberty(),
        _characterization(),
        {
            "resize_drivers": True,
            "insert_buffers": False,
            "repair_design_openroad_driver_resize": True,
        },
    )
    assert openroad_rule["resizes"] == []


def test_repair_design_resize_to_target_slew_normalizes_every_driver():
    """OpenROAD placement repair normalizes drive strength before violations."""
    liberty = deepcopy(_liberty())
    # DRV target load is 0.005 pF, BUF target load is 0.05 pF.  The driver
    # starts as DRV with a 0.05 pF load, so the native selector must swap it
    # to BUF even though no limit is violated after the swap.
    liberty["arc_values"] = [0.01] * 48
    liberty["arc_values"][8:12] = [0.01, 1.0, 0.01, 1.0]
    liberty["arc_values"][12:16] = [0.01, 1.0, 0.01, 1.0]
    liberty["arc_values"][24:28] = [0.001, 0.1, 0.001, 0.1]
    liberty["arc_values"][28:32] = [0.001, 0.1, 0.001, 0.1]

    result = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        liberty,
        _characterization(),
        {
            "resize_drivers": True,
            "insert_buffers": False,
            "repair_design_resize_to_target_slew": True,
        },
    )

    assert result["status"] == "ok"
    assert result["resizes"] == [{"node_id": 0, "target_cell_index": 1}]
    assert result["actions"] == []


def _fanout_design():
    design = _design()
    design.update(
        {
            "num_nets": 1,
            "num_pins": 6,
            "num_nodes": 6,
            "flat_net2pin_map": [0, 1, 2, 3, 4, 5],
            "flat_net2pin_start_map": [0, 6],
            "net2driver_pin_map": [0],
            "pin2node_map": [0, 1, 2, 3, 4, 5],
            "pin2net_map": [0, 0, 0, 0, 0, 0],
            "node_x": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
            "node_y": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
            "inst_main_id": [0, 1, 1, 1, 1, 1],
            "inst_libcell_offset": [0, 0, 0, 0, 0, 0],
            "pin_2_libpin_offset": [0, 0, 0, 0, 0, 0],
        }
    )
    return design


def _fanout_timing():
    return {
        "pin_slew_rise": [0.01] * 6,
        "pin_slew_fall": [0.01] * 6,
        "pin_load": [0.005, 0.001, 0.001, 0.001, 0.001, 0.001],
        "pin_cap": [0.001] * 6,
        "pin_max_slew": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "pin_max_cap": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "clock_pins": [],
    }


def _fanout_steiner():
    return {
        "steiner_node_base": 6,
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 6],
        "net_flat_topo_sort": [0, 1, 2, 3, 4, 5],
        "pin_fa": [-1, 0, 0, 0, 0, 0],
        "flat_pin_to_start": [0, 1, 2, 3, 4, 5, 6],
        "flat_pin_to": [1, 2, 3, 4, 5, 0],
        "node_x": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "node_y": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
        "edge_resistance": [0.0, 91.4, 91.4, 91.4, 91.4, 91.4],
        "node_capacitance": [0.0] * 6,
    }


def test_repair_design_fanout_region_repair_is_opt_in_and_partitions_loads():
    design = _fanout_design()
    timing = _fanout_timing()
    steiner = _fanout_steiner()
    legacy = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        _liberty(),
        _characterization(),
        {"resize_drivers": False, "insert_buffers": True, "max_fanout": 2},
    )
    assert legacy["actions"] == []

    fanout = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        _liberty(),
        _characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "max_fanout": 2,
            "repair_design_fanout_repair": True,
        },
    )
    assert fanout["actions"]
    assert all(action["reason"] == "fanout" for action in fanout["actions"])
    assert all(len(action["downstream_pins"]) <= 2 for action in fanout["actions"])
    assert sorted(
        pin
        for action in fanout["actions"]
        for pin in action["downstream_pins"]
    ) == [1, 2, 3, 4, 5]


def test_repair_design_fanout_repair_skips_stale_tree_repair_on_same_net():
    design = _fanout_design()
    timing = _fanout_timing()
    timing["pin_slew_rise"] = [0.2] * 6
    timing["pin_slew_fall"] = [0.2] * 6
    timing["pin_load"] = [0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
    timing["pin_cap"] = [0.01] * 6
    timing["pin_max_slew"] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    timing["pin_max_cap"] = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
    steiner = _fanout_steiner()

    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        _liberty(),
        _characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "max_fanout": 2,
            "repair_design_fanout_repair": True,
            "repair_net_tree": True,
        },
    )
    assert result["status"] == "ok"
    assert result["actions"]
    assert all(action["reason"] == "fanout" for action in result["actions"])
    assert sorted(
        pin
        for action in result["actions"]
        for pin in action["downstream_pins"]
    ) == [1, 2, 3, 4, 5]


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


def test_repair_design_openroad_repeater_resize_uses_post_split_load():
    design = _design()
    design["node_x"] = [0.0, 200000.0]
    steiner = _steiner()
    steiner["node_x"] = [0.0, 200000.0]

    legacy = rebuffer_cpp.plan_repair_design(
        design,
        _timing(),
        steiner,
        _post_resize_liberty(),
        _post_resize_characterization(),
        {"resize_drivers": False, "insert_buffers": True, "repair_net_tree": True},
    )
    resized = rebuffer_cpp.plan_repair_design(
        design,
        _timing(),
        steiner,
        _post_resize_liberty(),
        _post_resize_characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "repair_net_tree": True,
            "repair_design_openroad_repeater_resize": True,
        },
    )

    assert legacy["actions"]
    assert resized["actions"]
    assert legacy["actions"][0]["buffer_cell_index"] == 1
    assert resized["actions"][0]["buffer_cell_index"] == 3
    assert resized["actions"][0]["downstream_load"] < 0.02


def test_repair_design_openroad_repeater_resize_requires_tree_path():
    result = rebuffer_cpp.plan_repair_design(
        _design(),
        _timing(),
        _steiner(),
        _post_resize_liberty(),
        _post_resize_characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "repair_design_openroad_repeater_resize": True,
        },
    )

    assert result == {
        "status": "unsupported",
        "unsupported_reasons": [
            "repair_design_openroad_repeater_resize_requires_repair_net_tree"
        ],
    }


def test_repair_design_max_wire_length_is_native_and_opt_in():
    design = _design()
    design["node_x"] = [0.0, 100000.0]
    design["node_y"] = [0.0, 0.0]
    steiner = _steiner()
    steiner["node_x"] = [0.0, 100000.0]
    steiner["node_y"] = [0.0, 0.0]
    timing = _timing()
    timing["pin_slew_rise"] = [0.01, 0.01]
    timing["pin_slew_fall"] = [0.01, 0.01]
    timing["pin_load"] = [0.001, 0.001]
    timing["pin_max_slew"] = [1.0, 0.0]
    timing["pin_max_cap"] = [1.0, 0.0]

    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        _liberty(),
        _characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "repair_design_max_wire_length_um": 20.0,
        },
    )

    assert result["status"] == "ok"
    assert result["actions"]
    assert all(action["reason"] == "wire_length" for action in result["actions"])
    assert len(result["actions"]) >= 2


def test_repair_design_skips_external_pad_net():
    design = _design()
    design["num_base_nodes"] = 2
    design["num_terminals"] = 1
    design["num_terminal_NIs"] = 1
    design["pin2node_map"] = [1, 0]
    design["node_is_fixed"] = [1, 0]
    timing = _timing()
    timing["pin_max_slew"] = [0.0, 0.05]

    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        _steiner(),
        _liberty(),
        _characterization(),
        {
            "resize_drivers": False,
            "insert_buffers": True,
            "repair_net_tree": True,
        },
    )

    assert result["status"] == "ok"
    assert result["actions"] == []
    assert result["resizes"] == []

