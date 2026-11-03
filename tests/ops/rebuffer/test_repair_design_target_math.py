"""Analytic LUTs isolate target-cell math from candidate eligibility and RC."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.rebuffer.test_repair_design import (
    _characterization,
    _post_resize_characterization,
    _post_resize_liberty,
)
from tests.ops.rebuffer.test_repair_design_tree_state import _fork


def _table(offset, slew_coefficient, cap_coefficient):
    return [
        offset + slew_coefficient * slew + cap_coefficient * cap
        for slew in (0.0, 1000.0)
        for cap in (0.0, 1.0)
    ]


def _tables(liberty, cell_arcs):
    arcs = [arc for cell in cell_arcs for arc in cell]
    starts = [0]
    for cell in cell_arcs:
        starts.append(starts[-1] + len(cell))
    liberty.update(
        cell_arc_start=starts,
        arc_sense=[1] * len(arcs),
        arc_trigger=[0] * len(arcs),
        arc_dim=[2, 2] * len(arcs),
        arc_trans_start=list(range(0, 2 * len(arcs) + 1, 2)),
        arc_trans_axis=[0.0, 1000.0] * len(arcs),
        arc_cap_start=list(range(0, 2 * len(arcs) + 1, 2)),
        arc_cap_axis=[0.0, 1.0] * len(arcs),
        arc_value_start=list(range(0, 16 * len(arcs) + 1, 16)),
        arc_values=[value for arc in arcs for table in arc for value in table],
    )


def _plan(design, timing, steiner, liberty, characterization, *, enabled):
    timing = {
        **timing,
        "pin_max_cap": [1.0e6, 0.0, 0.0],
        "pin_max_slew": [1.0e6, 0.0, 0.0],
    }
    return rebuffer_cpp.plan_repair_design(
        design,
        timing,
        steiner,
        liberty,
        characterization,
        {
            "resize_drivers": False,
            "repair_design_fanout_repair": True,
            "max_fanout": 1,
            "repair_design_target_cell_math": enabled,
        },
    )


def test_asymmetric_multi_arc_target_load_averages_each_transition_root():
    design, timing, steiner, liberty = _fork()
    delay = _table(1.0, 0.0, 1.0)
    first = [delay, delay, _table(0.0, 0.25, 500.0), _table(20.0, 0.25, 1000.0)]
    second = [delay, delay, _table(0.0, 0.5, 1000.0), _table(10.0, 0.5, 500.0)]
    _tables(liberty, [[first], [first, second], [first]])
    characterization = _characterization()
    characterization.update(target_slew_rise=80.0, target_slew_fall=40.0)

    legacy = _plan(design, timing, steiner, liberty, characterization, enabled=False)
    actual = _plan(design, timing, steiner, liberty, characterization, enabled=True)

    assert legacy["status"] == actual["status"] == "ok"
    assert len(legacy["actions"]) == len(actual["actions"]) == 2
    # The four analytic roots are 0.04, 0.06, 0.06 and 0.02 pF.
    assert [row["target_load"] for row in actual["actions"]] == pytest.approx(
        [0.045, 0.045], rel=0.01
    )
    assert [row["target_load"] for row in legacy["actions"]] == pytest.approx([0.04, 0.04])


def test_repair_and_characterization_share_target_load_math():
    design, timing, steiner, liberty = _fork()
    delay = _table(1.0, 0.1, 1.0)
    arc = [delay, delay, _table(0.0, 0.25, 500.0), _table(0.0, 0.5, 1000.0)]
    _tables(liberty, [[arc], [arc], [arc]])
    characterization = rebuffer_cpp.characterize_buffers(liberty, canonical_target_load=True)
    assert characterization["status"] == "ok"

    result = _plan(design, timing, steiner, liberty, characterization, enabled=True)

    assert result["status"] == "ok"
    target = characterization["filtered_buffers"][0]["target_load"]
    assert target > 0.0
    # The action serializer rounds electrical fields to twelve decimals.
    assert [row["target_load"] for row in result["actions"]] == pytest.approx(
        [target, target], rel=0.0, abs=5.0e-13
    )


def test_canonical_target_load_uses_one_pf_initial_upper_bound():
    design, timing, steiner, liberty = _fork()
    liberty["cell_is_buffer"] = [False, True, False]
    delay = _table(1.0, 0.0, 0.0)
    transition = _table(0.0, 0.0, 1.0 / 0.37)
    _tables(liberty, [[[delay, delay, transition, transition]]] * 3)
    characterization = _characterization()
    characterization.update(target_slew_rise=1.0, target_slew_fall=1.0)

    result = _plan(design, timing, steiner, liberty, characterization, enabled=True)

    assert result["status"] == "ok"
    # The root is 0.37 pF. Starting at 1 pF, the first bracket narrower than
    # 1% is [0.369140625, 0.37109375]; OpenROAD returns its lower bound.
    assert [row["target_load"] for row in result["actions"]] == [0.369140625] * 2


@pytest.mark.parametrize("input_slew", [1.0, 200.0])
def test_delay_scoring_uses_target_input_slew(input_slew):
    design, timing, steiner, _ = _fork()
    liberty = _post_resize_liberty()
    liberty["cell_max_slew"] = [1000.0] * 4
    timing["pin_slew_rise"] = timing["pin_slew_fall"] = [input_slew] * 3
    rise, fall = _table(40.0, 0.0, 1000.0), _table(0.0, 0.0, 1000.0)
    slow_at_target = [_table(1.0, 0.5, 0.0)] * 2 + [fall, rise]
    fast_at_target = [_table(20.0, 0.1, 0.0)] * 2 + [fall, rise]
    _tables(liberty, [[slow_at_target], [slow_at_target], [slow_at_target], [fast_at_target]])
    characterization = _post_resize_characterization()
    characterization.update(target_slew_rise=80.0, target_slew_fall=40.0)

    legacy = _plan(design, timing, steiner, liberty, characterization, enabled=False)
    actual = _plan(design, timing, steiner, liberty, characterization, enabled=True)

    assert legacy["status"] == actual["status"] == "ok"
    assert [row["buffer_cell_index"] for row in actual["actions"]] == [3, 3]
    assert [row["buffer_cell_index"] for row in legacy["actions"]] == (
        [1, 1] if input_slew == 1.0 else [3, 3]
    )
