"""Repair decisions must preserve loads and combine junction constraints."""

from copy import deepcopy

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _characterization
from test_repair_design_tree_state import _fork


@pytest.mark.parametrize("violation", ["cap", "slew"])
def test_load_on_internal_tree_node_matches_colocated_leaf(violation):
    design, timing, steiner, liberty = _fork()
    if violation == "slew":
        timing["pin_max_cap"][0] = liberty["cell_max_cap"][0] = 0.1
        timing["pin_max_slew"] = [0.0, 0.002, 50.0]
    config = {"resize_drivers": False, "repair_net_tree": True}
    as_leaf = rebuffer_cpp.plan_repair_design(
        design, timing, steiner, liberty, _characterization(), config
    )

    # Move load 1 onto the same geometrical path to load 2. Its capacitance
    # and slew limit still belong to the net even though it has a child.
    internal = deepcopy(steiner)
    internal.update(
        net_flat_topo_sort_start=[0, 3],
        net_flat_topo_sort=[0, 1, 2],
        pin_fa=[-1, 0, 1, -1],
        flat_pin_to_start=[0, 1, 2, 2, 2],
        flat_pin_to=[1, 2],
    )
    as_internal = rebuffer_cpp.plan_repair_design(
        design, timing, internal, liberty, _characterization(), config
    )

    assert as_leaf["inserted_count"] == 1
    assert as_internal == as_leaf


def test_junction_isolates_different_branches_for_cap_and_slew():
    design, timing, steiner, liberty = _fork()
    # Each branch alone meets the slew limit. Combining them violates both
    # constraints: the larger cap is on load 1, the tighter slew on load 2.
    timing["pin_max_slew"] = [0.0, 50.0, 0.002]
    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {
            "resize_drivers": False,
            "repair_net_tree": True,
            "r_unit": 0.0,
            "c_unit": 0.0,
        },
    )

    assert result["status"] == "ok"
    assert [
        (a["reason"], a["downstream_pins"], a["downstream_load"]) for a in result["actions"]
    ] == [("junction_cap", [1], 0.007), ("junction_cap", [2], 0.005)]


def test_junction_length_checks_both_branches_and_uses_right_tie_break():
    design, timing, steiner, liberty = _fork()
    timing["pin_max_cap"][0] = liberty["cell_max_cap"][0] = 0.1
    # Both 0.4 um branches are individually legal, but their 0.8 um span
    # exceeds the junction budget. The driver is colocated with the fork.
    design["node_x"] = [1000.0, 1400.0, 600.0]
    steiner["node_x"] = [1000.0, 1400.0, 600.0, 1000.0]
    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {
            "resize_drivers": False,
            "repair_net_tree": True,
            "repair_design_max_wire_length_um": 0.6,
            "r_unit": 0.0,
            "c_unit": 0.0,
        },
    )

    assert result["status"] == "ok"
    assert [(a["reason"], a["downstream_pins"], a["x_dbu"]) for a in result["actions"]] == [
        ("junction_length", [2], 1000)
    ]


def test_junction_equal_cap_uses_openroad_right_branch_tie_break():
    design, timing, steiner, liberty = _fork()
    timing["pin_cap"] = [0.0, 0.006, 0.006]
    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {"resize_drivers": False, "repair_net_tree": True},
    )

    assert result["status"] == "ok"
    assert [(a["reason"], a["downstream_pins"]) for a in result["actions"]] == [
        ("junction_cap", [2])
    ]


def test_junction_equal_slew_slack_uses_openroad_right_branch_tie_break():
    design, timing, steiner, liberty = _fork()
    design["node_x"][0] = steiner["node_x"][0] = 1000.0
    timing["pin_cap"] = [0.0, 0.006, 0.006]
    timing["pin_max_cap"][0] = liberty["cell_max_cap"][0] = 0.1
    timing["pin_max_slew"] = [0.0, 0.002, 0.002]
    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {
            "resize_drivers": False,
            "repair_net_tree": True,
            "r_unit": 0.0,
            "c_unit": 0.0,
        },
    )

    assert result["status"] == "ok"
    assert [(a["reason"], a["downstream_pins"]) for a in result["actions"]] == [
        ("junction_slew", [2])
    ]


def test_junction_isolates_different_branches_for_cap_and_length():
    design, timing, steiner, liberty = _fork()
    design["node_x"] = [1000.0, 1200.0, 600.0]
    steiner["node_x"] = [1000.0, 1200.0, 600.0, 1000.0]
    result = rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        _characterization(),
        {
            "resize_drivers": False,
            "repair_net_tree": True,
            "repair_design_max_wire_length_um": 0.5,
            "r_unit": 0.0,
            "c_unit": 0.0,
        },
    )

    assert result["status"] == "ok"
    assert [(a["reason"], a["downstream_pins"]) for a in result["actions"]] == [
        ("junction_length", [1]),
        ("junction_length", [2]),
    ]
