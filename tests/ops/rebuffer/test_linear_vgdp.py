"""Linear VGDP is independent of NLDM tables once gates are characterized."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("dreamplace.ops.rebuffer.rebuffer_cpp")

from dreamplace.ops.rebuffer import rebuffer_cpp  # noqa: E402
from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb  # noqa: E402
from test_repair_design import (  # noqa: E402
    _characterization,
    _design,
    _liberty,
    _steiner,
    _timing,
)
from test_tree_dp import (  # noqa: E402
    _pydb,
    _serial_buffer_pydb,
    _serial_buffer_topology,
    _single_edge_topology,
    _topology,
)


def _inputs():
    design = _design()
    design["node_is_buffer"] = [False, False]
    design["flat_lib_pin_drive_resistance"] = [0.0, 0.0, 0.0]
    topology = _steiner()
    topology["node_x"] = [0.0, 100_000.0]
    design["node_x"] = topology["node_x"]
    timing = _timing()
    timing["pin_slack"] = [0.0, -100.0]
    liberty = _liberty()
    liberty["cell_intrinsic_delay"] = [0.0, 0.01, 0.0]
    liberty["cell_drive_resistance"] = [0.0, 500.0, 0.0]
    return design, topology, timing, liberty, _characterization()


def _plan(inputs):
    design, topology, timing, liberty, characterization = inputs
    return rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        liberty,
        characterization,
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
        rebuffer_cpp.import_current_driver_tree(design, topology, 0),
    )


def test_linear_buffer_charges_intrinsic_output_cap_and_both_wire_segments():
    inputs = _inputs()
    inputs[3]["cell_output_cap"] = [0.0, 0.003, 0.0]
    result = _plan(inputs)

    # 100 um of wire: R=91.4, C=0.0189. One midpoint buffer with
    # Cin=0.002, Cout=0.003, d0=0.01, Rb=500, driving a 0.001 load.
    np.testing.assert_allclose(
        [(c["buffer_count"], c["cap"], c["slack"]) for c in result["candidates"]],
        [(0, 0.0199, -100.95513), (1, 0.01145, -107.303965), (1, 0.002, -112.41513)],
    )


def test_fixed_linear_vgdp_does_not_consume_nldm_or_sink_slew():
    inputs = _inputs()
    expected = _plan(inputs)
    changed = deepcopy(inputs)
    changed[3]["arc_values"] = [float(index * index + 100) for index in range(48)]
    assert _plan(changed) == expected
    changed[2]["pin_slew_rise"] = [50.0, 70.0]
    changed[2]["pin_slew_fall"] = [80.0, 90.0]
    actual = _plan(changed)

    # Wire diagnostic slews reflect observations; timing score/frontier/actions
    # are functions only of the fixed linear model and load budgets.
    def decisions(result):
        return [
            {key: c[key] for key in ("cap", "slack", "actions", "buffer_count")}
            for c in result["candidates"]
        ]

    assert decisions(actual) == decisions(expected)


def test_cq_dominance_removes_timing_inferior_higher_cap_option():
    inputs = _inputs()
    inputs[4]["fast_buffers"][0]["input_cap"] = 0.02
    result = _plan(inputs)

    np.testing.assert_allclose(
        [(c["buffer_count"], c["cap"], c["slack"]) for c in result["candidates"]],
        [(0, 0.0199, -100.95513)],
    )


def test_root_linear_load_relief_changes_candidate_ranking():
    inputs = _inputs()
    assert _plan(inputs)["candidates"][0]["buffer_count"] == 0
    inputs[0]["flat_lib_pin_drive_resistance"][0] = 1000.0
    result = _plan(inputs)

    assert result["candidates"][0]["buffer_count"] == 1
    np.testing.assert_allclose(
        [(c["buffer_count"], c["slack"]) for c in result["candidates"]],
        [(1, -63.91513), (1, -68.253965), (0, -71.85513)],
    )


@pytest.mark.parametrize("exported_resistance", [None, 0.0])
def test_missing_driver_resistance_uses_characterized_linear_cell(exported_resistance):
    inputs = _inputs()
    if exported_resistance is None:
        del inputs[0]["flat_lib_pin_drive_resistance"]
    else:
        inputs[0]["flat_lib_pin_drive_resistance"][0] = exported_resistance
    inputs[3]["cell_drive_resistance"][0] = 1000.0
    result = _plan(inputs)

    assert result["driver_resistance"] == 1000.0
    action = result["candidates"][0]["actions"][0]
    assert (action["x_dbu"], action["y_dbu"]) == (0, 0)


def test_load_annotation_preserves_worst_transition_and_root_arrival_basis():
    inputs = _inputs()
    timing = inputs[2]
    timing.update(
        {
            "pin_arrival_rise": [100.0, 110.0],
            "pin_arrival_fall": [200.0, 230.0],
            "pin_required_rise": [float("inf"), 150.0],
            "pin_required_fall": [float("inf"), 210.0],
        }
    )
    result = _plan(inputs)
    unbuffered = next(c for c in result["candidates"] if c["buffer_count"] == 0)

    assert unbuffered["slack"] == pytest.approx(10.0 - 0.95513)
    timing["pin_arrival_fall"][1] = 240.0
    assert _plan(inputs)["candidates"][0]["slack"] == pytest.approx(unbuffered["slack"])


def test_partial_timing_snapshots_are_rejected_at_native_boundary():
    inputs = _inputs()
    inputs[2]["pin_arrival_rise"] = [1.0, 2.0]
    result = _plan(inputs)

    assert result == {
        "status": "unsupported",
        "unsupported_reasons": ["tree_timing_snapshot_size_mismatch"],
    }


def test_two_branch_cq_frontier_matches_all_nine_physical_combinations():
    design = build_rebuffer_design_from_pydb(_pydb(), require_buffer_metadata=True)
    topology = _topology()
    topology["node_x"] = {0: 0, 1: 100_000, 2: 200_000, 3: 0, 4: 0}
    topology["node_y"] = {pin: 0 for pin in range(5)}
    timing = _timing()
    timing["pin_cap"] = [0.0, 0.001, 0.002, 0.0, 0.0]
    timing["pin_slack"] = [0.0, -100.0, -90.0, 0.0, 0.0]
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
        rebuffer_cpp.import_current_driver_tree(design, topology, 0),
    )

    # Each branch has pass, midpoint and parent-site options. Their nine
    # combinations leave these five points on the exact (C,Q) frontier.
    np.testing.assert_allclose(
        [(c["buffer_count"], c["cap"], c["slack"]) for c in result["candidates"]],
        [
            (0, 0.0597, -100.95513),
            (1, 0.0408, -102.55306),
            (2, 0.03235, -105.803965),
            (2, 0.0229, -110.91513),
            (2, 0.004, -113.73052),
        ],
    )


def test_existing_buffer_delay_uses_downstream_load_before_replacing_it_with_cin():
    design = build_rebuffer_design_from_pydb(_serial_buffer_pydb(), require_buffer_metadata=True)
    topology = _serial_buffer_topology()
    timing = _timing()
    timing["pin_cap"] = [0.001] * 7
    timing["pin_slack"] = [1000.0] * 7
    timing["pin_slack"][5] = -100.0
    characterization = _characterization()
    characterization["fast_buffers"] = []
    liberty = _liberty()
    liberty["cell_intrinsic_delay"] = [0.0, 0.01, 0.0]
    liberty["cell_drive_resistance"] = [0.0, 500.0, 0.0]
    tree = rebuffer_cpp.import_current_driver_tree(design, topology, 0)

    def score():
        result = rebuffer_cpp.plan_current_driver_tree(
            design,
            timing,
            liberty,
            characterization,
            {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
            tree,
        )
        return result["candidates"][0]["slack"]

    before = score()
    timing["pin_cap"][5] += 0.01
    # Only the last buffer sees the added load; upstream sees its fixed Cin.
    # The last wire also charges Rwire * delta(C) = .914 * .01.
    assert score() - before == pytest.approx(-5.0 - 0.00914)


def test_joint_tree_vgdp_matches_exhaustive_single_edge_pareto_reference():
    pydb = _pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(design, _single_edge_topology(), 1)
    timing = _timing()
    timing["pin_slew_rise"] = [0.2] * 5
    timing["pin_slew_fall"] = [0.2] * 5
    timing["pin_load"] = [0.05, 0.001, 0.001, 0.05, 0.001]
    timing["pin_cap"] = [0.001, 0.001, 0.001, 0.001, 0.001]
    timing["pin_max_slew"] = [0.1] * 5
    timing["pin_max_cap"] = [0.01] * 5
    timing["pin_slack"] = [0.0, 0.0, 0.0, 0.0, -100.0]
    config = {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4}
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        config,
        tree,
    )

    load_cap = timing["pin_cap"][4]
    load_slack = timing["pin_slack"][4]
    length_um = 100.0
    wire_cap = length_um * config["c_unit"]
    wire_res = length_um * config["r_unit"]
    pass_through = {
        "buffer_count": 0,
        "cap": load_cap + wire_cap,
        "slack": load_slack - wire_res * (load_cap + wire_cap / 2.0),
    }
    half_cap = wire_cap / 2.0
    half_res = wire_res / 2.0
    buffer = _characterization()["fast_buffers"][0]
    buffered = {
        "buffer_count": 1,
        "cap": buffer["input_cap"] + half_cap,
        "slack": load_slack
        - half_res * (load_cap + buffer["input_cap"] + half_cap)
        - (0.01 + buffer["drive_resistance"] * (load_cap + half_cap)),
    }
    buffered_at_parent = {
        "buffer_count": 1,
        "cap": buffer["input_cap"],
        "slack": pass_through["slack"] - 0.01 - buffer["drive_resistance"] * (load_cap + wire_cap),
    }

    observed = [
        {
            "buffer_count": candidate["buffer_count"],
            "cap": candidate["cap"],
            "slack": candidate["slack"],
        }
        for candidate in result["candidates"]
    ]
    assert observed == pytest.approx([pass_through, buffered, buffered_at_parent])
    assert result["candidates"][1]["actions"][0]["parent_node_id"] == 0
    assert result["candidates"][1]["actions"][0]["child_node_id"] == 1
    assert result["candidates"][1]["actions"][0]["source_parent_node_id"] == 3
    assert result["candidates"][1]["actions"][0]["source_child_node_id"] == 4
    assert result["candidates"][1]["actions"][0]["segment_tree_depth"] == 0
    assert all(
        "slew_rise" in candidate and "slew_fall" in candidate for candidate in result["candidates"]
    )
    assert all("slew_rise" in item and "slew_fall" in item for item in result["trace"])
