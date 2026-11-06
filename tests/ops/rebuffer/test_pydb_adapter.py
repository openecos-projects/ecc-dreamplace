from types import SimpleNamespace

import numpy as np
from dreamplace.ops.rebuffer import build_rebuffer_design_from_pydb


def _fake_pydb():
    return SimpleNamespace(
        dbu=1000,
        flat_net2pin_map=[2, 0, 1, 3, 4],
        flat_net2pin_start_map=[0, 3, 5],
        net2driver_pin_map=[0, 3],
        pin2node_map=[0, 1, 2, 3, 4],
        pin2net_map=[0, 0, 0, 1, 1],
        pin_offset_x=[100, 200, 300, 0, 0],
        pin_offset_y=[10, 20, 30, 0, 0],
        node_x=[10_000, 30_000, 50_000, 70_000, 90_000],
        node_y=[1_000, 1_000, 1_000, 2_000, 2_000],
        node_size_x=[1_000, 1_000, 1_000, 2_000, 2_000],
        node_size_y=[1_000, 1_000, 1_000, 1_000, 1_000],
        node_is_buffer=[False, False, False, False, False],
        node_is_fixed=[False, False, False, False, False],
        inst_main_id=[0, 1, 1, 0, 1],
        inst_libcell_offset=[0, 0, 0, 0, 0],
        main_id_2_cell_id_start=[0, 1, 2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        pin_2_libpin_offset=[1, 0, 1, 1, 0],
        pin_direct=[-1, 1, 1, -1, 1],
        flat_lib_pin_cap=[0.0, 0.42, 0.0, 0.84],
        flat_lib_pin_rcap=[0.0, 0.42, 0.0, 0.84],
        flat_lib_pin_cap_limit=[0.0, 0.0, 0.0, 0.0],
        flat_lib_pin_slew_limit=[0.0, 0.0, 0.0, 0.0],
        flat_lib_pin_drive_resistance=[0.0, 0.0, 0.0, 0.0],
        flat_libcell_info=[
            ["BUF_X1", 0, 1.0, 0.0],
            ["INV_X1", 1, 1.0, 0.0],
        ],
        flat_libcell_names=["BUF_X1", "INV_X1"],
        net_names=["net0", "net1"],
        pin_names=["U0/Y", "U1/A", "U2/A", "U3/Y", "U4/A"],
        node_names=["U0", "U1", "U2", "U3", "U4"],
        clock_pins=[0],
        clock_net_names=["clk"],
        rows=[(0, 0, 100_000, 1_000)],
        site_width=500,
        row_height=1_000,
        num_movable_nodes=5,
        buffer_main_type_index=0,
        buffer_main_type_candidate_indices=[0],
    )


def _summary_without_reasons(design):
    summary = dict(design["summary"])
    summary.pop("unsupported_reasons", None)
    return summary


def test_builds_native_ready_design_view_from_pydb():
    design = build_rebuffer_design_from_pydb(_fake_pydb())

    assert design["status"] == "ok"
    assert design["source"] == "pydb"
    assert design["dbu"] == 1000
    assert design["num_nets"] == 2
    assert design["num_pins"] == 5
    assert design["num_nodes"] == 5
    assert design["num_libcells"] == 2
    assert design["num_lib_pins"] == 4
    assert design["summary"] == {
        "status": "ok",
        "source": "pydb",
        "dbu": 1000,
        "num_nets": 2,
        "num_pins": 5,
        "num_nodes": 5,
        "num_base_nodes": 5,
        "num_terminals": 0,
        "num_terminal_NIs": 0,
        "num_libcells": 2,
        "num_lib_pins": 4,
        "has_buffer_metadata": True,
        "buffer_metadata_source": "node_is_buffer",
        "has_fixed_metadata": True,
        "has_pin_limits": True,
        "has_drive_resistance": True,
        "has_pin_direction": True,
        "has_libcell_info": True,
        "pin_cap_source": "rcap",
        "requires_legalization": True,
        "unsupported_reasons": [],
    }
    assert design["flat_net2pin_map"].dtype == np.int32
    assert design["flat_net2pin_map"].flags.c_contiguous
    assert design["node_x"].dtype == np.float64
    assert design["node_x"].flags.c_contiguous
    np.testing.assert_array_equal(design["flat_net2pin_map"], [2, 0, 1, 3, 4])
    np.testing.assert_array_equal(design["flat_net2pin_start_map"], [0, 3, 5])
    np.testing.assert_array_equal(design["pin2node_map"], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(design["node_is_buffer"], [False] * 5)
    np.testing.assert_array_equal(design["node_is_fixed"], [False] * 5)
    assert design["names"]["libcell"] == ("BUF_X1", "INV_X1")
    np.testing.assert_array_equal(
        design["flat_libcell_info"],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    assert design["buffer_main_type_candidate_indices"] == (0,)


def test_missing_buffer_metadata_is_reported_but_not_invented():
    pydb = _fake_pydb()
    del pydb.node_is_buffer
    del pydb.buffer_main_type_candidate_indices

    design = build_rebuffer_design_from_pydb(pydb)
    assert design["status"] == "ok"
    assert design["node_is_buffer"] is None
    assert design["summary"]["has_buffer_metadata"] is False

    required = build_rebuffer_design_from_pydb(
        pydb,
        require_buffer_metadata=True,
    )
    assert required["status"] == "unsupported"
    assert required["summary"]["unsupported_reasons"] == ["missing_node_is_buffer"]
    assert required["node_is_buffer"] is None


def test_buffer_metadata_can_be_derived_from_candidate_families():
    pydb = _fake_pydb()
    del pydb.node_is_buffer

    design = build_rebuffer_design_from_pydb(pydb)

    assert design["status"] == "ok"
    assert design["summary"]["buffer_metadata_source"] == ("buffer_main_type_candidate_indices")
    np.testing.assert_array_equal(design["node_is_buffer"], [True, False, False, True, False])


def test_invalid_csr_span_is_rejected_before_native_planning():
    pydb = _fake_pydb()
    pydb.flat_net2pin_start_map = [0, 3, 4]

    design = build_rebuffer_design_from_pydb(pydb)

    assert design["status"] == "unsupported"
    assert "flat_net2pin_start_map_does_not_cover_elements" in design["unsupported_reasons"]
    assert design["summary"]["status"] == "unsupported"


def test_rcap_is_preferred_and_missing_cap_limits_are_explicit():
    pydb = _fake_pydb()
    pydb.flat_lib_pin_cap_limit = []
    pydb.flat_lib_pin_slew_limit = []

    design = build_rebuffer_design_from_pydb(pydb)

    assert design["status"] == "ok"
    assert design["summary"]["pin_cap_source"] == "rcap"
    assert design["summary"]["has_pin_limits"] is False
    np.testing.assert_array_equal(design["flat_lib_pin_rcap"], [0.0, 0.42, 0.0, 0.84])


def test_missing_libcell_rows_block_native_buffer_library():
    pydb = _fake_pydb()
    pydb.flat_libcell_info = []

    design = build_rebuffer_design_from_pydb(pydb)

    assert design["status"] == "unsupported"
    assert design["summary"]["has_libcell_info"] is False
    assert design["summary"]["unsupported_reasons"] == ["missing_flat_libcell_info"]
