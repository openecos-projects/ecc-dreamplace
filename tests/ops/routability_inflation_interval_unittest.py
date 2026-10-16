"""Inflation spacing counts completed GP steps across geometry/stage restarts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dreamplace.ops.routability.routability_controller import RoutabilityController


def make_params(enhanced, overrides):
    schema = json.loads((Path(__file__).parents[2] / "dreamplace/params.json").read_text())
    fields = ("inflation_min_interval", "node_area_adjust_overflow", "max_num_area_adjust")
    values = {key: schema[key]["default"] for key in fields}
    values.update(routability_opt_flag=1, enhanced_inflation_flag=int(enhanced))
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize(
    "overrides,interval", [({"inflation_min_interval": 15}, 15), ({"inflation_min_interval": 3}, 3)]
)
def test_interval_blocks_both_triggers_until_completed_gp_boundary(enhanced, overrides, interval):
    params = make_params(enhanced, overrides)
    controller = RoutabilityController(params)
    enabled = (enhanced, not enhanced)

    def triggers(rounds, overflow=0.1):
        return controller.inflation_triggers(params, rounds, params.max_num_area_adjust, overflow)

    # The first eligible round is immediate. Only actual area changes are recorded.
    assert triggers(0) == enabled
    controller.before_area_adjust(model=None, position=None)
    assert triggers(0) == enabled
    controller.record_inflation()
    assert triggers(1) == (False, False)

    for step in range(interval - 1):
        controller.after_iteration(model=None, iteration=step, metrics=None)
    # Neither geometry/optimizer recovery nor a new stage resets the interval.
    controller.after_geometry_change(model=None, position=None)
    controller.before_stage(model=None, stage_idx=1)
    assert triggers(1) == (False, False)
    controller.after_iteration(model=None, iteration=1000, metrics=None)
    assert triggers(1) == enabled
    # Spacing adds a condition; it does not override overflow or round limits.
    assert triggers(1, overflow=0.2) == (False, False)
    assert triggers(params.max_num_area_adjust) == (False, False)

    controller.record_inflation()
    assert triggers(2) == (False, False)
    # Advancing a displayed iteration without completing GP does not count.
    controller.before_iteration(model=None, iteration=2000, position=None)
    assert triggers(2) == (False, False)
    for step in range(interval):
        controller.after_iteration(model=None, iteration=2000 + step, metrics=None)
    assert triggers(2) == enabled


@pytest.mark.parametrize("enhanced", [False, True])
def test_zero_interval_allows_the_existing_trigger(enhanced):
    params = make_params(enhanced, {})
    controller = RoutabilityController(params)
    controller.record_inflation()
    assert controller.inflation_triggers(params, 1, params.max_num_area_adjust, 0.1) == (
        enhanced,
        not enhanced,
    )
