"""OpenROAD checks both transitions and builds loads from max port C."""

import numpy as np
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _characterization, _design, _liberty, _steiner, _timing


def test_native_transition_load_payload_repairs_a_fall_only_cap_violation():
    timing = _timing()
    timing.update(
        pin_load=[0.009, 0.001],
        pin_cap=[0.0, 0.009],
        pin_slew_rise=[0.0, 0.0],
        pin_slew_fall=[0.0, 0.0],
    )
    config = {"resize_drivers": False}
    args = (_design(), timing, _steiner(), _liberty(), _characterization(), config)
    legacy = rebuffer_cpp.plan_repair_design(*args)
    assert legacy["actions"] == []

    normalized = rebuffer_cpp.repair_design_electrical_payload(
        np.array(timing["pin_load"]),
        np.array([0.012, 0.002]),
        np.array(timing["pin_cap"]),
        np.array([0.0, 0.012]),
    )
    assert {key: value.tolist() for key, value in normalized.items()} == {
        "pin_load": [0.012, 0.002],
        "pin_cap": [0.0, 0.012],
    }
    actual = rebuffer_cpp.plan_repair_design(
        _design(), {**timing, **normalized}, _steiner(), _liberty(), _characterization(), config
    )
    assert actual["status"] == "ok"
    assert len(actual["actions"]) == 1


def test_native_transition_payload_handles_opposite_transition_winners():
    result = rebuffer_cpp.repair_design_electrical_payload(
        np.array([0.03, 0.01]),
        np.array([0.02, 0.04]),
        np.array([0.009, 0.011]),
        np.array([0.012, 0.010]),
    )
    assert {key: value.tolist() for key, value in result.items()} == {
        "pin_load": [0.03, 0.04],
        "pin_cap": [0.012, 0.011],
    }
