"""OpenROAD split-wire break-even max_wire_length characterization and plumbing."""

from copy import deepcopy

import pytest
from dreamplace.ops.rebuffer import (
    characterize_buffers_from_pydb,
    find_max_wire_length,
    rebuffer_cpp,
    repair_design,
)
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_repair_design import (
    _post_resize_characterization,
    _post_resize_liberty,
)
from tests.ops.rebuffer.test_repair_design_target_math import _table, _tables
from tests.ops.rebuffer.test_tree_search import _long_wire


def _bisect_split_wire_root(zero_length_delay, drive_resistance, r_unit, c_unit):
    """Exact OpenROAD 1% bisection lower bound for (r*c/4)*L^2 - D(0) == 0."""
    low = 0.0
    high = drive_resistance / r_unit
    coeff = 0.25 * r_unit * c_unit
    diff_high = coeff * high * high - zero_length_delay
    for _ in range(100):
        if abs(low - high) <= max(low, high) * 0.01:
            break
        if diff_high < 0.0:
            low, high = high, 2.0 * high
            diff_high = coeff * high * high - zero_length_delay
        else:
            mid = 0.5 * (low + high)
            diff_mid = coeff * mid * mid - zero_length_delay
            if diff_mid < 0.0:
                low = mid
            else:
                high, diff_high = mid, diff_mid
    return low


def test_find_max_wire_length_matches_analytic_split_wire_bisection():
    liberty = _post_resize_liberty()
    liberty["cell_input_cap"] = [0.001, 0.01, 0.001, 0.005]
    liberty["cell_output_cap"] = [0.0, 0.02, 0.0, 0.005]
    # Linear delay D_gate(s, C) = d0 + k_s*s + R_drv*C cancels all O(L) terms in
    # D(L) - 2*D(L/2) = (r*c/4)*L^2 - D(0), giving exact roots 600 um and 500 um.
    buf_large = [
        _table(15.0, 0.1, 500.0),
        _table(15.0, 0.1, 500.0),
        _table(0.0, 0.0, 500.0),
        _table(0.0, 0.0, 500.0),
    ]
    buf_small = [
        _table(16.0, 0.1, 300.0),
        _table(16.0, 0.1, 300.0),
        _table(0.0, 0.0, 300.0),
        _table(0.0, 0.0, 300.0),
    ]
    _tables(liberty, [[buf_large], [buf_large], [buf_large], [buf_small]])
    characterization = _post_resize_characterization()
    characterization["target_slew_rise"] = 60.0
    characterization["target_slew_fall"] = 60.0
    characterization["filtered_buffers"][0].update(
        input_cap=0.01, output_cap=0.02, drive_resistance=500.0
    )
    characterization["filtered_buffers"][1].update(
        input_cap=0.005, output_cap=0.005, drive_resistance=300.0
    )

    r_unit, c_unit = 1.0, 4.0e-4
    expected_large = _bisect_split_wire_root(36.0, 500.0, r_unit, c_unit)
    expected_small = _bisect_split_wire_root(25.0, 300.0, r_unit, c_unit)
    assert expected_small < expected_large
    assert find_max_wire_length(
        liberty, characterization, r_unit=r_unit, c_unit=c_unit
    ) == expected_small

    only_large = deepcopy(characterization)
    only_large["filtered_buffers"] = [only_large["filtered_buffers"][0]]
    assert find_max_wire_length(
        liberty, only_large, r_unit=r_unit, c_unit=c_unit
    ) == expected_large

    assert find_max_wire_length(liberty, characterization, r_unit=0.0, c_unit=c_unit) == 0.0
    assert find_max_wire_length(liberty, characterization, r_unit=r_unit, c_unit=0.0) == 0.0

    default_char = rebuffer_cpp.characterize_buffers(liberty)
    rc_char = rebuffer_cpp.characterize_buffers(liberty, r_unit=r_unit, c_unit=c_unit)
    assert default_char["max_wire_length_um"] == 0.0
    assert rc_char["max_wire_length_um"] == pytest.approx(
        find_max_wire_length(liberty, rc_char, r_unit=r_unit, c_unit=c_unit)
    )
    assert rc_char["max_wire_length_um"] > 0.0


def test_repair_timing_context_and_tree_wire_step_enforce_auto_max_wire_length():
    design, timing, liberty, characterization, tree = _long_wire()
    timing.update(pin_max_cap=[100.0, 0.0], pin_max_slew=[1.0e6, 1.0e6])
    liberty.update(cell_max_cap=[100.0, 100.0, 0.0], cell_max_slew=[1.0e6, 1.0e6, 0.0])
    characterization["fast_buffers"][0].update(
        max_cap=100.0, max_slew=1.0e6, margined_max_cap=100.0
    )

    without_limit = rebuffer_cpp.optimize_current_driver_tree(
        design, timing, liberty, characterization, {}, tree
    )
    assert without_limit["long_wire_edge_count"] == 0

    with_auto = deepcopy(characterization)
    with_auto["max_wire_length_um"] = 300.0
    stepped = rebuffer_cpp.optimize_current_driver_tree(
        design, timing, liberty, with_auto, {}, tree
    )
    assert stepped["status"] == "ok"
    assert stepped["long_wire_edge_count"] == 1
    assert stepped["wire_site_count"] > without_limit["wire_site_count"]

    explicit_override = rebuffer_cpp.optimize_current_driver_tree(
        design, timing, liberty, with_auto, {"tree_max_wire_length_um": 2000.0}, tree
    )
    assert explicit_override["long_wire_edge_count"] == 0

    electrical_off = rebuffer_cpp.optimize_current_driver_tree(
        design, timing, liberty, with_auto, {"tree_enforce_electrical": False}, tree
    )
    assert electrical_off["long_wire_edge_count"] == 0

    pydb = _base()
    model = RebufferTimingModel(pydb, r_unit=1.2, c_unit=2.5e-4, use_pin_offset=False)
    context = RebufferPlanContext(pydb, model)
    expected = find_max_wire_length(
        context.liberty, context.characterization, r_unit=1.2, c_unit=2.5e-4
    )
    assert expected > 0.0
    assert context.characterization["max_wire_length_um"] == pytest.approx(expected)
    assert (
        characterize_buffers_from_pydb(pydb, r_unit=1.2, c_unit=2.5e-4)["max_wire_length_um"]
        == pytest.approx(expected)
    )
    payload = context.native_current_tree_payload(1)
    assert payload["characterization"]["max_wire_length_um"] == pytest.approx(expected)


def test_repair_design_preserves_zero_default_and_supports_opt_in_auto_max_wire_length():
    pydb = _base()
    # Stretch n1 (u:Y -> OUT) to 1200 um while keeping cap/slew DRV-clean.
    pydb.node_x = [10_000.0, 0.0, 1_210_000.0]
    pydb.flat_lib_pin_cap_limit = [0.0, 100.0, 0.0, 100.0]
    pydb.flat_lib_pin_slew_limit = [1.0e6] * 4
    model = RebufferTimingModel(pydb, use_pin_offset=False)

    base_cfg = {"resize_drivers": False}
    default_plan = repair_design.plan_repair_design_from_pydb(pydb, model, config=base_cfg)
    assert default_plan["status"] == "ok"
    assert default_plan["config"]["repair_design_max_wire_length_um"] == 0.0
    assert default_plan["inserted_count"] == 0

    auto_by_value = repair_design.plan_repair_design_from_pydb(
        pydb, model, config={**base_cfg, "repair_design_max_wire_length_um": "auto"}
    )
    auto_by_alias = repair_design.plan_repair_design_from_pydb(
        pydb, model, config={**base_cfg, "max_wire_length": "auto"}
    )
    auto_by_flag = repair_design.plan_repair_design_from_pydb(
        pydb, model, config={**base_cfg, "repair_design_auto_max_wire_length": True}
    )

    resolved_um = auto_by_value["config"]["repair_design_max_wire_length_um"]
    assert 0.0 < resolved_um < 1200.0
    assert auto_by_alias["config"]["repair_design_max_wire_length_um"] == pytest.approx(
        resolved_um
    )
    assert auto_by_flag["config"]["repair_design_max_wire_length_um"] == pytest.approx(
        resolved_um
    )
    assert auto_by_value["inserted_count"] > 0
    assert {row["reason"] for row in auto_by_value["actions"]} == {"wire_length"}
    assert auto_by_alias["actions"] == auto_by_value["actions"]
    assert auto_by_flag["actions"] == auto_by_value["actions"]

    native_auto = rebuffer_cpp.plan_repair_design(
        repair_design.build_rebuffer_design_from_pydb(pydb),
        repair_design.timing_payload(
            pydb, model, max_transition_load=True, include_pin_rank=True
        ),
        repair_design.steiner_payload(pydb, model, r_unit=model.r_unit),
        repair_design.build_liberty_view_from_pydb(pydb),
        characterize_buffers_from_pydb(pydb),
        {
            **repair_design.DEFAULT_CONFIG,
            **base_cfg,
            "r_unit": model.r_unit,
            "c_unit": model.c_unit,
            "repair_design_target_cell_selection": False,
            "repair_design_max_wire_length_um": "auto",
        },
    )
    assert repair_design.map_buffer_actions(native_auto, pydb)[0] == auto_by_value["actions"]

    explicit_plan = repair_design.plan_repair_design_from_pydb(
        pydb,
        model,
        config={
            **base_cfg,
            "repair_design_max_wire_length_um": 700.0,
            "repair_design_auto_max_wire_length": True,
        },
    )
    assert explicit_plan["config"]["repair_design_max_wire_length_um"] == 700.0
    assert 0 < explicit_plan["inserted_count"] < auto_by_value["inserted_count"]

    with pytest.raises(ValueError, match="unsupported repair_design max wire length"):
        repair_design.plan_repair_design_from_pydb(
            pydb, model, config={**base_cfg, "repair_design_max_wire_length_um": "bogus"}
        )
    native_invalid = rebuffer_cpp.plan_repair_design(
        repair_design.build_rebuffer_design_from_pydb(pydb),
        repair_design.timing_payload(
            pydb, model, max_transition_load=True, include_pin_rank=True
        ),
        repair_design.steiner_payload(pydb, model, r_unit=model.r_unit),
        repair_design.build_liberty_view_from_pydb(pydb),
        characterize_buffers_from_pydb(pydb),
        {**repair_design.DEFAULT_CONFIG, "repair_design_max_wire_length_um": "bogus"},
    )
    assert native_invalid["status"] == "unsupported"
    assert native_invalid["unsupported_reasons"] == [
        "invalid_repair_design_max_wire_length"
    ]
