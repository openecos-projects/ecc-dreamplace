"""Native acceptance checks per-pin observed TP electrical changes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp as native


def _inputs():
    db = SimpleNamespace(
        pin_names=["driver/Y", "a/A", "b/A"],
        net_names=["n"],
        net2driver_pin_map=[0],
        flat_net2pin_start_map=[0, 3],
        flat_net2pin_map=[0, 1, 2],
        clock_pins=[],
        clock_net_names=[],
    )
    timing = {
        "pin_slew_rise": [10.0, 25.0, 5.0],
        "pin_slew_fall": [15.0, 24.0, 8.0],
        "pin_load_rise": [0.04, 0.0, 0.0],
        "pin_load_fall": [0.06, 0.0, 0.0],
        "pin_max_slew": [20.0, 20.0, 0.0],
        "pin_max_cap": [0.05, 0.0, 0.0],
    }
    return db, timing


def test_native_snapshot_checks_both_transitions_driver_cap_and_configured_fanout():
    db, timing = _inputs()
    result = native.electrical_snapshot(db, timing, max_fanout=1)
    assert result["violations_by_pin"] == {
        "driver:Y": {"slew_ps": 0.0, "cap_pf": pytest.approx(0.01), "fanout": 1.0},
        "a:A": {"slew_ps": 5.0, "cap_pf": 0.0, "fanout": 0.0},
    }
    assert (
        result["slew_ps_checked_count"],
        result["cap_pf_checked_count"],
        result["fanout_checked_count"],
    ) == (2, 1, 1)
    assert (
        result["slew_ps_violation_count"],
        result["cap_pf_violation_count"],
        result["fanout_violation_count"],
    ) == (1, 1, 1)


def test_missing_limits_are_unavailable_and_ideal_clock_net_is_excluded():
    db, timing = _inputs()
    timing["pin_max_slew"] = [0.0] * 3
    timing["pin_max_cap"] = [0.0] * 3
    result = native.electrical_snapshot(db, timing, max_fanout=0)
    assert result["violations_by_pin"] == {}
    db.clock_net_names = ["n"]
    db, timing = db, _inputs()[1]
    assert native.electrical_snapshot(db, timing, max_fanout=1)["violations_by_pin"] == {}


def test_clock_net_driver_is_excluded_by_clock_pin_membership():
    """A pre-CTS clock buffer carries the whole clock load; it is not a violation."""
    db, timing = _inputs()
    # The pydb clock-net name list may not cover every routed clock segment, so
    # the rule must also key off the clock pins themselves.
    db.clock_net_names = []
    db.clock_pins = [1]
    timing["pin_load_rise"] = [62.5, 0.0, 0.0]
    timing["pin_load_fall"] = [62.5, 0.0, 0.0]
    result = native.electrical_snapshot(db, timing, max_fanout=32)
    assert result["violations_by_pin"] == {}
    assert result["cap_pf_violation_count"] == 0
    assert result["fanout_violation_count"] == 0


def test_equal_violation_count_cannot_hide_a_new_violating_pin():
    db, timing = _inputs()
    before = {"electrical": native.electrical_snapshot(db, timing)}
    changed = deepcopy(timing)
    changed["pin_max_slew"][2] = 20.0
    changed["pin_slew_rise"][1:] = [15.0, 25.0]
    changed["pin_slew_fall"][1:] = [15.0, 24.0]
    after = {"electrical": native.electrical_snapshot(db, changed)}
    assert (
        before["electrical"]["slew_ps_violation_count"]
        == after["electrical"]["slew_ps_violation_count"]
    )
    result = native.compare_electrical_observations(before, after)
    assert result == {
        "checked": True,
        "accepted": False,
        "worsened_pins": [
            {"pin_name": "b:A", "check": "slew_ps", "before_excess": 0.0, "after_excess": 5.0}
        ],
    }


def test_existing_violation_can_improve_and_numeric_pin_reordering_is_safe():
    db, timing = _inputs()
    before = {"electrical": native.electrical_snapshot(db, timing)}
    changed = deepcopy(timing)
    changed["pin_slew_rise"][1] = 23.0
    changed["pin_slew_fall"][1] = 23.0
    order = [2, 0, 1]
    db.pin_names = [db.pin_names[i] for i in order]
    db.net2driver_pin_map = [1]
    db.flat_net2pin_map = [1, 2, 0]
    changed = {key: [values[i] for i in order] for key, values in changed.items()}
    after = {"electrical": native.electrical_snapshot(db, changed)}
    assert native.compare_electrical_observations(before, after)["accepted"] is True


@pytest.mark.parametrize("field", ["pin_slew_fall", "pin_load_fall"])
def test_nonfinite_checked_transition_is_not_hidden_by_other_transition(field):
    db, timing = _inputs()
    timing[field][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        native.electrical_snapshot(db, timing)


def test_native_queue_rejects_timing_gain_that_worsens_existing_pin_erc():
    db, timing = _inputs()
    baseline = {
        "status": "ok",
        "wns_ps": -10.0,
        "tns_ps": -20.0,
        "electrical": native.electrical_snapshot(db, timing),
    }
    changed = deepcopy(timing)
    changed["pin_slew_rise"][1] = 30.0
    candidate = {
        "status": "ok",
        "wns_ps": -9.0,
        "tns_ps": -19.0,
        "electrical": native.electrical_snapshot(db, changed),
    }
    state = dict(baseline)
    result = native.run_repair_timing_queue_operator(
        lambda: {"items": [{"endpoint_pin": 1, "slack_ps": -5.0, "net_ids": [0]}]},
        lambda _request: {"actions": [{"net_id": 0}]},
        lambda _label: 0,
        lambda _actions: state.update(candidate),
        lambda: dict(state),
        lambda _token: (state.clear(), state.update(baseline)),
        {"max_passes": 1},
    )
    assert result["accepted_actions"] == []
    assert state == result["final"] == baseline
    assert result["passes"][0]["frontier_trials"][0]["electrical_check"]["worsened_pins"] == [
        {"pin_name": "a:A", "check": "slew_ps", "before_excess": 5.0, "after_excess": 10.0},
    ]
