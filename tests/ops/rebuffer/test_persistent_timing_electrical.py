"""Graph acceptance observes the same physical ERC and endpoints as fresh TP."""

import pytest
from dreamplace.ops.placeio_ecc.electrical_observation import electrical_observation
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.timing_model_payload import timing_pin_slack
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _reference, _split
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _assert_observation(graph, owner, *, fanout=1, cap_margin=0.0, slew_margin=0.0):
    actual = graph.evaluation_observation(fanout, cap_margin, slew_margin)
    expected = electrical_observation(
        owner, max_fanout=fanout, cap_margin=cap_margin, slew_margin=slew_margin
    )
    assert actual["electrical"] == expected
    slack = timing_pin_slack(owner.model)
    assert actual["endpoint_slacks_by_name"] == {
        owner.pydb.pin_names[int(pin)]: float(slack[int(pin)]) for pin in owner.model.end_points
    }
    return actual


def _constrained_base():
    base = _base()
    base.outcaps = [0.03]
    base.out_rcaps = [0.02]
    base.out_fcaps = [0.04]
    base.flat_lib_pin_cap_limit = [0.0, 0.04, 0.0, 0.02]
    base.flat_lib_pin_slew_limit = [15.0, 16.0, 17.0, 18.0]
    return base


@pytest.mark.parametrize("margins", [(0.0, 0.0), (5.0, 10.0)])
def test_native_electrical_uses_both_transitions_and_current_limits(margins):
    owner, graph = _graph(_constrained_base())
    graph.propagate()

    result = _assert_observation(graph, owner, cap_margin=margins[0], slew_margin=margins[1])

    assert result["electrical"]["cap_pf_violation_count"] > 0
    assert result["electrical"]["slew_ps_violation_count"] > 0


def test_split_remove_holes_and_restore_share_fresh_tp_observation():
    owner, graph = _graph(_constrained_base())
    graph.propagate()
    before = _assert_observation(graph, owner)
    root = graph.checkpoint()
    first, _ = _split(graph, 1, 3, 0, master="BUF2")
    second, _ = _split(graph, 1, 3, 1, master="BUF1")
    graph.propagate_incremental()
    _assert_observation(graph, _reference(owner, graph, [first, second]))

    graph.remove_buffer(1, 0)
    graph.propagate_incremental()
    _assert_observation(graph, _reference(owner, graph, [second]))
    graph.restore(root)
    assert graph.evaluation_observation(1, 0.0, 0.0) == before


def test_native_electrical_excludes_clock_net_and_clock_pins():
    base = _constrained_base()
    base.clock_net_names = ["n0"]
    owner, graph = _graph(base)
    graph.propagate()

    result = _assert_observation(graph, owner)

    assert "u:A" not in result["electrical"]["violations_by_pin"]


def test_stale_native_observation_rejects_until_propagated():
    _, graph = _graph(_base())
    graph.propagate()
    graph.resize_cell("u", "BUF2")

    with pytest.raises(ValueError, match="current native propagation"):
        graph.evaluation_observation()

    graph.propagate_incremental()
    assert graph.evaluation_observation()["electrical"]["pin_count"] == 4


@pytest.mark.parametrize("margins", [(-1.0, 0.0), (0.0, 100.0)])
def test_native_electrical_rejects_invalid_margins(margins):
    _, graph = _graph(_base())
    graph.propagate()

    with pytest.raises(ValueError, match="margins"):
        graph.evaluation_observation(32, *margins)


def test_uninitialized_native_electrical_rejects():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())

    with pytest.raises(ValueError, match="initialized native timing"):
        graph.evaluation_observation()
