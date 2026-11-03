
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_rebuffer_timing import _dp_inputs


def test_driver_rank_validates_pin_ids_against_pin_count():
    design, timing, _steiner, liberty, _characterization = _dp_inputs()
    design["net2driver_pin_map"] = [1]
    design["inst_main_id"] = [0, 0]

    assert rebuffer_cpp.rank_path_driver_pins(design, timing, liberty, [0, 1]) == [1]


def _path_load_inputs():
    design, timing, _steiner, _liberty, _characterization = _dp_inputs()
    design.update(
        {
            "num_nets": 3,
            "num_pins": 3,
            "num_nodes": 3,
            "flat_net2pin_map": [0, 1, 1, 2, 2],
            "flat_net2pin_start_map": [0, 2, 4, 5],
            "net2driver_pin_map": [0, 1, 2],
            "pin2node_map": [0, 1, 2],
            "pin2net_map": [0, 1, 2],
            "inst_main_id": [0, 1, 2],
            "inst_libcell_offset": [0, 0, 0],
            "main_id_2_cell_id_start": [0, 1, 2, 3],
            "cell_id_2_libpin_id_start": [0, 1, 2, 3],
            "pin_2_libpin_offset": [0, 0, 0],
        }
    )
    timing.update(
        {
            "pin_slew_rise": [0.2, 0.2, 0.2],
            "pin_slew_fall": [0.2, 0.2, 0.2],
            "pin_load": [0.0, 0.05, 0.05],
            "pin_cap": [0.0, 0.0, 0.0],
            "path_arc_delays": [12.0, 7.0],
            "path_arc_local_ids": [0, 0],
            "path_transitions": [0, 0, 0],
        }
    )

    def arc_values(zero_load, full_load):
        return [zero_load, full_load, zero_load, full_load] * 4

    liberty = {
        "dbu": 1000,
        "names": ["SRC", "DRV1", "DRV2"],
        "cell_main_id": [0, 1, 2],
        "cell_size": [1, 1, 1],
        "cell_vt": [0, 0, 0],
        "cell_area": [1.0, 1.0, 1.0],
        "cell_leakage": [0.0, 0.0, 0.0],
        "cell_input_cap": [0.0, 0.0, 0.0],
        "cell_max_cap": [1.0, 1.0, 1.0],
        "cell_max_slew": [1000.0, 1000.0, 1000.0],
        "cell_is_buffer": [False, False, False],
        "cell_arc_start": [0, 1, 2, 3],
        "arc_source": ["A", "A", "A"],
        "arc_sink": ["Y", "Y", "Y"],
        "arc_local_id": [0, 0, 0],
        "arc_sense": [1, 1, 1],
        "arc_trigger": [0, 0, 0],
        "arc_dim": [2, 2] * 3,
        "arc_trans_start": [0, 2, 4, 6],
        "arc_trans_axis": [0.01, 0.1] * 3,
        "arc_cap_start": [0, 2, 4, 6],
        "arc_cap_axis": [0.0, 0.1] * 3,
        "arc_value_start": [0, 16, 32, 48],
        "arc_values": [0.0] * 16
        + arc_values(10.0, 14.0)
        + arc_values(1.0, 13.0),
    }

    return design, timing, liberty


def test_driver_rank_prefers_path_load_dependent_delay_over_total_cell_delay():
    design, timing, liberty = _path_load_inputs()
    assert rebuffer_cpp.rank_path_driver_pins(
        design, timing, liberty, [0, 1, 2]
    ) == [2, 1]


def test_openroad_intrinsic_uses_zero_slew_not_current_input_slew():
    design, timing, liberty = _path_load_inputs()
    timing.update(pin_slew_rise=[100.0] * 3, pin_slew_fall=[100.0] * 3)
    liberty["arc_trans_axis"] = [0.0, 100.0] * 3
    liberty["arc_values"] = [0.0] * 16 + [0.0, 4.0, 10.0, 14.0] * 4 + [1.0, 13.0, 1.0, 13.0] * 4
    # Actual path delays are 12 and 7 ps. At current slew, intrinsic terms
    # are 10 and 1: scores 2 and 6. OpenSTA evaluates intrinsic at (0, 0),
    # so scores are instead 12 and 6 and the first driver must lead.
    assert rebuffer_cpp.rank_path_driver_pins(design, timing, liberty, [0, 1, 2]) == [2, 1]
    assert rebuffer_cpp.rank_path_driver_pins(
        design, timing, liberty, [0, 1, 2], openroad_intrinsic=True
    ) == [1, 2]




def test_driver_rank_includes_the_launching_register_output():
    # Path starts at FF:Q (pin 0).  The register's CK input (pin 2) sits on
    # its own clock net and is not on the data path; the Q driver must still
    # be ranked because its fanout load delays every downstream stage.
    design, timing, _steiner, liberty, _characterization = _dp_inputs()
    design.update(
        num_nets=2,
        num_pins=3,
        flat_net2pin_map=[0, 1, 2],
        flat_net2pin_start_map=[0, 2, 3],
        net2driver_pin_map=[0, 2],
        pin2node_map=[0, 1, 0],
        pin2net_map=[0, 0, 1],
        pin_offset_x=[0.0] * 3,
        pin_offset_y=[0.0] * 3,
        pin_2_libpin_offset=[0, 0, 0],
    )
    design["names"] = {"pin": ["ff/Q", "load/A", "ff/CK"]}
    timing.update(
        pin_slew_rise=[0.2, 0.2, 0.05],
        pin_slew_fall=[0.2, 0.2, 0.05],
        pin_load=[0.050189, 0.05, 0.0],
        pin_cap=[0.001, 0.05, 0.001],
        pin_max_slew=[1000.0, 0.0, 0.0],
        pin_max_cap=[0.01, 0.0, 0.0],
        pin_slack=[0.0, -0.2, 0.0],
    )
    liberty = dict(liberty)
    liberty.update(arc_source=["CK", "A"], arc_sink=["Q", "Y"])

    assert rebuffer_cpp.rank_path_driver_pins(design, timing, liberty, [0, 1]) == [0]
