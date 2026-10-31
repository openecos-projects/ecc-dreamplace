"""Native tree search: physical chains, active ERC, rounds and recovery."""

from copy import deepcopy

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp as native
from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb
from test_linear_vgdp import _inputs
from test_repair_design import _characterization, _liberty, _timing
from test_tree_dp import _pydb, _topology


def _long_wire():
    design, topology, timing, liberty, characterization = _inputs()
    topology["node_x"] = [0.0, 1_000_000.0]
    design["node_x"] = topology["node_x"]
    design["flat_lib_pin_drive_resistance"][0] = 1000.0
    timing.update(
        pin_cap=[0.0, 0.001], pin_load=[0.19, 0.001],
        pin_slew_rise=[10.0, 10.0], pin_slew_fall=[10.0, 10.0],
        pin_max_slew=[50.0, 50.0], pin_max_cap=[0.03, 0.0],
    )
    liberty.update(
        cell_drive_resistance=[1000.0, 500.0, 0.0],
        cell_max_cap=[0.03, 0.03, 0.0], cell_max_slew=[50.0, 50.0, 0.0],
    )
    characterization["fast_buffers"][0].update(
        max_cap=0.03, max_slew=50.0, margined_max_cap=0.03,
    )
    tree = native.import_current_driver_tree(design, topology, 0)
    return design, timing, liberty, characterization, tree


def _plan(inputs, config=None, *, optimize=True):
    design, timing, liberty, characterization, tree = inputs
    planner = native.optimize_current_driver_tree if optimize else native.plan_current_driver_tree
    return planner(design, timing, liberty, characterization, config or {}, tree)


def test_long_wire_chain_materializes_and_obeys_cap_slew_and_action_budget():
    inputs = _long_wire()
    result = _plan(inputs)
    assert result["status"] == "ok"
    candidate = result["candidates"][0]
    assert candidate["buffer_count"] > 1
    assert result["long_wire_edge_count"] > 0
    assert result["wire_site_count"] > candidate["buffer_count"]
    actions = candidate["actions"]
    assert {(a["source_parent_node_id"], a["source_child_node_id"]) for a in actions} == {(0, 1)}
    assert [a["x_dbu"] for a in actions] == sorted((a["x_dbu"] for a in actions), reverse=True)
    tree = native.apply_current_driver_tree_buffer_actions(inputs[-1], inputs[2], actions)
    exported = native.export_current_driver_tree(tree)
    assert exported["buffer_count"] == candidate["buffer_count"]
    assert exported["edge_count"] == exported["node_count"] - 1
    assert _plan(inputs, {"insert_buffers": False})["status"] == "unsupported"
    assert _plan(inputs, {"max_actions": 1})["status"] == "unsupported"
    assert _plan(inputs, {"insert_buffers": False, "tree_enforce_electrical": False})["status"] == "ok"
    rescored = native.plan_current_driver_tree(
        *inputs[:4], {"insert_buffers": False}, tree,
    )
    assert rescored["status"] == "ok"
    assert rescored["candidates"][0]["slack"] == pytest.approx(candidate["slack"])


def test_chain_materialization_rejects_reversed_source_edge_order():
    inputs = _long_wire()
    candidate = _plan(inputs)["candidates"][0]
    before = native.export_current_driver_tree(inputs[-1])
    with pytest.raises(ValueError, match="chain_order_invalid"):
        native.apply_current_driver_tree_buffer_actions(inputs[-1], inputs[2], candidate["actions"][::-1])
    assert native.export_current_driver_tree(inputs[-1]) == before


@pytest.mark.parametrize("config", [{"cap_margin": 20.0}, {"slew_margin": 50.0}])
def test_electrical_margins_change_the_actual_wire_search(config):
    inputs = _long_wire()
    baseline = _plan(inputs)
    margined = _plan(inputs, config)
    assert margined["status"] == "ok"
    assert margined["wire_site_count"] > baseline["wire_site_count"]
    assert margined["candidates"][0]["actions"] != baseline["candidates"][0]["actions"]


def _branches():
    design = build_rebuffer_design_from_pydb(_pydb(), require_buffer_metadata=True)
    topology = _topology()
    topology["node_x"] = {0: 0, 1: 100_000, 2: 100_000, 3: 0, 4: 0}
    topology["node_y"] = {pin: 0 for pin in range(5)}
    timing = _timing()
    timing.update(pin_cap=[0.0, 0.001, 0.001, 0.0, 0.0], pin_slack=[0.0, -100.0, -10.0, 0.0, 0.0])
    liberty = _liberty()
    liberty.update(cell_drive_resistance=[0.0, 10.0, 0.0], cell_intrinsic_delay=[0.0, 0.01, 0.0])
    characterization = _characterization()
    characterization["fast_buffers"][0]["drive_resistance"] = 10.0
    tree = native.import_current_driver_tree(design, topology, 0)
    return design, timing, liberty, characterization, tree


def test_multiple_rounds_consume_changed_trees_and_improve_the_fixed_model():
    inputs = _branches()
    config = {"tree_enforce_electrical": False, "tree_recovery_rounds": 0}
    first = _plan(inputs, {**config, "tree_timing_rounds": 1})
    result = _plan(inputs, config)
    assert result["timing_round_count"] == 3
    assert result["candidates"][0]["slack"] > first["candidates"][0]["slack"]
    digests = result["round_source_tree_digests"]
    assert len(set(digests)) == 3
    assert digests[0] == native.export_current_driver_tree(inputs[-1])["tree_digest"]
    assert result["round_source_buffer_counts"][0] == 0
    assert result["round_source_buffer_counts"][1] > 0


def test_recovery_removes_noncritical_buffers_and_preserves_predicted_timing():
    inputs = _branches()
    config = {"tree_enforce_electrical": False}
    unrecovered = _plan(inputs, {**config, "tree_recovery_rounds": 0})["candidates"][0]
    result = _plan(inputs, config)
    recovered = result["candidates"][0]
    assert result["recovery_attempt_count"] > 0
    assert result["recovered_buffer_count"] > 0
    assert recovered["slack"] >= unrecovered["slack"] - 1e-9
    assert recovered["area"] < unrecovered["area"]
    assert {a["child_node_id"] for a in recovered["actions"]} == {1}
    materialized = native.apply_current_driver_tree_buffer_actions(inputs[-1], inputs[2], recovered["actions"])
    assert native.export_current_driver_tree(materialized)["buffer_count"] == recovered["buffer_count"]


def test_complete_tree_search_is_independent_of_nldm_tables():
    inputs = _long_wire()
    expected = _plan(inputs)
    changed = deepcopy(inputs[:4])
    changed[2]["arc_values"] = [1000.0 + i * i for i in range(48)]
    assert _plan((*changed, inputs[-1])) == expected


def test_fanout_constraint_is_executed_at_the_root():
    inputs = _branches()
    inputs[1]["pin_max_slew"] = [0.0] * 5
    inputs[1]["pin_max_cap"] = [0.0] * 5
    # No insertion: two leaves cannot satisfy a one-load driver constraint.
    assert _plan(inputs, {"insert_buffers": False, "max_fanout": 1})["status"] == "unsupported"
    assert _plan(inputs, {"insert_buffers": False, "max_fanout": 2})["status"] == "ok"
