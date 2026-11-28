"""Overlay/session mutations use a single native topology/RC/timing journal."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay
from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _compare
from tests.ops.rebuffer.test_persistent_timing_resize import _same_state


def _session():
    base = _base()
    model = RebufferTimingModel(base, use_pin_offset=False, use_native_timing_graph=True)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=model)
    session.begin()
    return model, session, session.overlay.native_timing_bridge


def _action(overlay, net=1, master="BUF2"):
    name = overlay.base_pydb.net_names[net]
    state = overlay.native_topologies.get(name)
    if state is None:
        state = overlay.timing_model.create_net_overlay(name)
    view = rebuffer_cpp.incremental_topology_view(state)
    sink = (1, 3)[net]
    edge = next(edge for edge in view["edges"] if edge["child"] == sink)
    start, end = view["coordinates"][edge["parent"]], view["coordinates"][sink]
    return {
        "action_kind": "buffer_insert",
        "net_name": name,
        "driver_pin_name": overlay.base_pydb.pin_names[(0, 2)[net]],
        "downstream_pin_names": [overlay.base_pydb.pin_names[sink]],
        "buffer_master_name": master,
        "candidate_location_x_dbu": round((start[0] + end[0]) / 2),
        "candidate_location_y_dbu": round((start[1] + end[1]) / 2),
    }


def _snapshot(overlay, bridge):
    bridge.evaluate(overlay)
    return {
        "state": bridge.graph.state_view(),
        "metadata": bridge.graph.mutation_view(),
        "actions": deepcopy(overlay.actions),
        "ledger": bridge.campaign.summary()["action_ledger"],
        "topologies": dict(overlay.native_topologies),
        "allocation": overlay._next_buffer_id,
    }


def _same(overlay, bridge, before):
    _same_state(bridge.graph.state_view(), before["state"], bitwise=True)
    _same_state(bridge.graph.mutation_view(), before["metadata"], bitwise=True)
    assert overlay.actions == before["actions"]
    assert bridge.campaign.summary()["action_ledger"] == before["ledger"]
    assert overlay.native_topologies == before["topologies"]
    assert overlay._next_buffer_id == before["allocation"]
    assert overlay.native_campaign_session is bridge.campaign
    assert bridge.graph.mutation_index_oracle()["status"] == "ok"


def test_empty_root_checkpoint_keeps_owner_and_detaches_new_net_bindings():
    model, session, bridge = _session()
    overlay = session.overlay
    root = session.checkpoint()["checkpoint_id"]
    before = _snapshot(overlay, bridge)
    checkpoint = overlay._checkpoints[root]
    assert checkpoint["native_checkpoints"] == {}
    assert checkpoint["native_campaign_checkpoint"] is None
    assert checkpoint["persistent_graph_checkpoint"] is None
    assert checkpoint["native_timing_checkpoint"] is not None

    session.apply([_action(overlay)])
    session.evaluate()
    session.restore(root)

    _same(overlay, bridge, before)
    assert model._persistent_graph_bridge is bridge
    session.apply([_action(overlay, master="BUF1")])
    session.evaluate()
    _compare(bridge.graph, model, overlay.actions)


def test_multi_net_session_apply_restore_and_projection_match_fresh_tp():
    model, session, bridge = _session()
    overlay = session.overlay
    session.apply([_action(overlay, net=0)])
    session.evaluate()
    before = _snapshot(overlay, bridge)
    middle = session.checkpoint()["checkpoint_id"]
    session.apply([_action(overlay)])
    observation = session.evaluate()
    assert observation["native_timing_graph"]["mode"] == "events"
    _compare(bridge.graph, model, overlay.actions)

    expected = rebuffer_cpp.finalize_overlay_actions(
        model.pydb, list(overlay.native_topologies.values()), list(overlay.actions)
    )
    assert session.export_actions() == expected
    assert overlay.projection_audit["projection_source"] == "native_timing_graph"
    session.restore(middle)
    _same(overlay, bridge, before)


def test_apply_batch_failure_restores_graph_actions_ledger_and_allocation():
    _, session, bridge = _session()
    overlay = session.overlay
    before = _snapshot(overlay, bridge)

    with pytest.raises(ValueError, match="absent from Liberty"):
        session.apply([_action(overlay, net=0), _action(overlay, master="MISSING_MASTER")])

    _same(overlay, bridge, before)


def test_successful_and_failed_tree_replacement_share_native_rollback():
    model, session, bridge = _session()
    overlay = session.overlay
    session.apply([_action(overlay)])
    session.apply([_action(overlay, master="BUF1")])
    session.evaluate()
    before = _snapshot(overlay, bridge)
    checkpoint = session.checkpoint()["checkpoint_id"]

    with pytest.raises(ValueError, match="absent from Liberty"):
        session.replace_session_tree(
            "n1", [_action(overlay), _action(overlay, master="MISSING_MASTER")]
        )
    _same(overlay, bridge, before)

    result = session.replace_session_tree("n1", [_action(overlay)])
    assert result["replacement_mode"] == "native_campaign"
    assert len(overlay.actions) == 1
    session.evaluate()
    _compare(bridge.graph, model, overlay.actions)
    session.restore(checkpoint)
    _same(overlay, bridge, before)


def test_remove_ancestor_through_overlay_then_restore_preserves_descendant():
    model, session, bridge = _session()
    overlay = session.overlay
    for master in ("BUF1", "BUF2", "BUF1"):
        session.apply([_action(overlay, master=master)])
    session.evaluate()
    before = _snapshot(overlay, bridge)
    checkpoint = session.checkpoint()["checkpoint_id"]

    session.apply([{"action_kind": "buffer_remove", "session_buffer_id": 0}])
    session.evaluate()
    assert [row["session_buffer_id"] for row in overlay.actions] == [1, 2]
    _compare(bridge.graph, model, overlay.actions)
    session.restore(checkpoint)
    _same(overlay, bridge, before)


def test_discarded_overlay_checkpoint_rejects_before_python_state_changes():
    _, session, bridge = _session()
    overlay = session.overlay
    root = session.checkpoint()["checkpoint_id"]
    session.apply([_action(overlay)])
    session.evaluate()
    discarded = session.checkpoint()["checkpoint_id"]
    old_epoch = bridge.graph.epoch
    session.restore(root)
    session.apply([_action(overlay, master="BUF1")])
    session.evaluate()
    assert bridge.graph.epoch == old_epoch
    before = _snapshot(overlay, bridge)

    with pytest.raises(ValueError, match="stale journal history"):
        session.restore(discarded)

    _same(overlay, bridge, before)


def test_checkpoint_before_evaluate_restores_stale_mark_then_propagates():
    model, session, bridge = _session()
    overlay = session.overlay
    session.apply([_action(overlay)])
    checkpoint = session.checkpoint()["checkpoint_id"]
    session.evaluate()
    session.apply([_action(overlay, net=0)])
    session.evaluate()

    session.restore(checkpoint)

    assert len(overlay.actions) == 1
    _compare(bridge.graph, model, overlay.actions)


def test_repeated_evaluation_and_statistics_do_not_append_timing_undo():
    _, session, bridge = _session()
    session.apply([_action(session.overlay)])
    session.evaluate()
    before = bridge.graph.summary()
    count = bridge.propagation_count
    first = bridge.statistics()
    assert bridge.statistics() == first

    for _ in range(3):
        session.evaluate()

    assert bridge.graph.summary() == before
    assert bridge.propagation_count == count


def test_rollback_and_rebind_use_each_overlays_own_graph():
    model, session, bridge = _session()
    session.apply([_action(session.overlay)])
    session.evaluate()
    other = BufferOverlay(model.pydb, timing_model=model)
    other_bridge = other.native_timing_bridge
    assert other_bridge is not bridge
    assert other_bridge.graph.summary()["active_pin_count"] == len(model.pydb.pin_names)
    model.evaluate(model.pydb, other)

    session.rollback()

    assert bridge.graph.summary()["net_count"] == 0
    assert other_bridge.graph.summary()["net_count"] == 0
    assert other.native_campaign_session is other_bridge.campaign
    other.apply([_action(other)])
    model.evaluate(model.pydb, other)
    assert other_bridge.graph.summary()["active_pin_count"] == len(model.pydb.pin_names) + 2
    assert bridge.graph.summary()["active_pin_count"] == len(model.pydb.pin_names)


def test_native_owner_rejects_base_removal_without_mutating_either_state():
    _, session, bridge = _session()
    before = _snapshot(session.overlay, bridge)
    with pytest.raises(ValueError, match="base buffer removal"):
        session.remove_base_buffers(["u"])
    _same(session.overlay, bridge, before)
