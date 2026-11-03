"""Actual path delay and zero-slew intrinsic must drive opt-in ranking."""

from types import SimpleNamespace

import pytest
from dreamplace.flows.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_native_timing_campaign import _action
from tests.ops.placeio_ecc.test_timing_carrier import _base, _reference_after


def _session(*, fall_noise=False):
    base = _reference_after(_base())
    base.node_master_names = ["BUF1", "BUF2", "", ""]
    base.node_size_x = [1000.0, 2000.0, 0.0, 0.0]
    base.node_size_y = [1000.0, 1000.0, 0.0, 0.0]
    base.pin_direct = ["R0"] * len(base.pin_names)
    # u's delay is 0.2 * input slew + 100 * output C; b's delay is a
    # constant 50 ps. Total delay ranks b first; OpenROAD's intrinsic at
    # (slew=0, C=0) removes b's 50 ps and ranks u first.
    for prefix in ("r_delay", "f_delay"):
        setattr(base, prefix + "_flat_luts_values", [[0.0, 100.0, 20.0, 120.0], [50.0] * 4])
    if fall_noise:
        base.r_delay_flat_luts_values[1] = [0.0] * 4
        base.f_delay_flat_luts_values[1] = [1000.0] * 4
        base.endpoints_fRAT = [2000.0]
    model = RebufferTimingModel(
        base,
        use_pin_offset=False,
        use_native_timing_graph=True,
        use_native_timing_evaluation=True,
    )
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=model)
    session.begin()
    session.evaluate()
    return model, session, session.overlay.native_timing_bridge


def _drivers(snapshot):
    return [row["driver_pin_id"] for row in rebuffer_cpp.select_timing_queue_items(snapshot, {})]


@pytest.mark.parametrize("fall_noise", [False, True])
def test_native_path_context_ranks_actual_load_delay_instead_of_total_delay(fall_noise):
    model, session, bridge = _session(fall_noise=fall_noise)
    original = list(range(len(model.pydb.net_names)))
    assert _drivers(bridge.graph.timing_queue_state(original)) == [4, 2]
    selected = bridge.graph.timing_queue_state(original, use_path_context=True)
    assert _drivers(selected) == [2, 4]
    assert selected["graph_queue_statistics"]["path_context_count"] == 1
    assert bridge.statistics()["python_materialization_count"] == 0


def test_native_planning_and_carrier_queue_use_the_same_opt_in_path_ranking():
    model, session, bridge = _session()
    config = resolve_repair_timing_config(
        {
            "use_native_timing_graph": True,
            "use_native_timing_queue": True,
            "use_native_timing_evaluation": True,
            "use_native_timing_planning": True,
            "use_native_timing_path_context": True,
        }
    )
    context = RebufferPlanContext(model.pydb, model, config=config)
    bridge.initialize_planning(context)
    snapshot = bridge.planning_snapshot(session.overlay, config).queue_state()
    projected = bridge.queue_state(session.overlay, model.pydb, use_path_context=True)
    assert _drivers(snapshot) == _drivers(projected) == [2, 4]


def test_path_context_keeps_local_arc_identity_after_insert_and_restore():
    model, session, bridge = _session()
    original = list(range(len(model.pydb.net_names)))
    root = session.checkpoint()["checkpoint_id"]
    session.apply([_action(session.overlay)])
    session.evaluate()
    assert _drivers(bridge.graph.timing_queue_state(original, use_path_context=True)) == [2, 4, 7]
    session.restore(root)
    session.evaluate()
    assert _drivers(bridge.graph.timing_queue_state(original, use_path_context=True)) == [2, 4]
