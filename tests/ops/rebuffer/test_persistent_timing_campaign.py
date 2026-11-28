"""One native campaign checkpoint owns topology, RC, timing and action state."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.liberty import build_liberty_view_from_pydb
from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _compare
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _campaign():
    owner, graph = _graph(_base(), attach=False)
    campaign = rebuffer_cpp.create_repair_timing_campaign_session(graph)
    states = {}
    for net, name in enumerate(owner.pydb.net_names):
        states[net] = owner.create_net_overlay(name)
        campaign.attach_topology(net, states[net])
    graph.propagate()
    design = build_rebuffer_design_from_pydb(owner.pydb)
    graph.initialize_queue(
        build_liberty_view_from_pydb(owner.pydb), design["node_is_buffer"].tolist()
    )
    return owner, graph, campaign, states


def _action(graph, net, child, action_id, master="BUF1"):
    view = graph.topology_view(net)
    edge = next(row for row in view["edges"] if row["child"] == child)
    start, end = view["coordinates"][edge["parent"]], view["coordinates"][child]
    return {
        "action_id": action_id,
        "session_buffer_id": action_id,
        "buffer_master_name": master,
        "downstream_pin_ids": [child],
        "x_dbu": (start[0] + end[0]) / 2,
        "y_dbu": (start[1] + end[1]) / 2,
    }


def _snapshot(graph, campaign):
    return {
        "state": graph.state_view(),
        "metadata": graph.mutation_view(),
        "paths": graph.setup_path_details()["paths"],
        "queue": rebuffer_cpp.select_timing_queue_items(graph.timing_queue_state([0, 1]), {}),
        "ledger": campaign.summary()["action_ledger"],
    }


def _assert_snapshot(graph, campaign, before):
    _same_state(graph.state_view(), before["state"], bitwise=True)
    _same_state(graph.mutation_view(), before["metadata"], bitwise=True)
    assert graph.setup_path_details()["paths"] == before["paths"]
    assert (
        rebuffer_cpp.select_timing_queue_items(graph.timing_queue_state([0, 1]), {})
        == before["queue"]
    )
    assert campaign.summary()["action_ledger"] == before["ledger"]
    assert graph.mutation_index_oracle()["status"] == "ok"


def test_campaign_apply_matches_direct_graph_geometry_rc_timing_and_fresh_tp():
    owner, graph, campaign, states = _campaign()
    _, direct, _, direct_states = _campaign()
    actions = []
    for net, child, action_id in ((0, 1, 7), (1, 3, 3), (1, 3, 9)):
        action = _action(graph, net, child, action_id)
        result = campaign.apply_candidate(net, action)
        resolved = rebuffer_cpp.resolve_overlay_split(direct_states[net], action)
        direct.apply_buffer_split(net, resolved)
        actions.append(action)
        graph.propagate_incremental()
        direct.propagate_incremental()
        _same_state(graph.state_view(), direct.state_view(), bitwise=True)
        _same_state(graph.mutation_view(), direct.mutation_view(), bitwise=True)
        for key in ("parent_node_id", "child_node_id", "split_ratio", "x_dbu", "y_dbu"):
            assert result[key] == resolved[key]
        assert states[net].epoch == direct_states[net].epoch
    _compare(graph, owner, actions)


def test_checkpoint_restores_multi_net_state_and_keeps_lifecycle_history():
    _, graph, campaign, _ = _campaign()
    before = _snapshot(graph, campaign)
    checkpoint = campaign.checkpoint()
    campaign.apply_candidate(0, _action(graph, 0, 1, 7))
    campaign.apply_candidate(1, _action(graph, 1, 3, 3))
    graph.propagate_incremental()
    campaign.record_candidate(1, {"status": "ok", "wns_ps": -2.0}, "observed")
    count = len(campaign.summary()["ledger_history"])

    campaign.restore(checkpoint)

    _assert_snapshot(graph, campaign, before)
    assert len(campaign.summary()["ledger_history"]) == count


def test_failed_replacement_restores_old_tree_rc_timing_and_ledger_atomically():
    _, graph, campaign, _ = _campaign()
    for action_id in (7, 3):
        campaign.apply_candidate(1, _action(graph, 1, 3, action_id))
    graph.propagate_incremental()
    before = _snapshot(graph, campaign)
    history = campaign.summary()["ledger_history"]
    actions = [_action(graph, 1, 3, 11), _action(graph, 1, 3, 12, "MISSING_MASTER")]

    with pytest.raises(ValueError):
        campaign.replace_candidate_tree(1, actions)

    _assert_snapshot(graph, campaign, before)
    assert campaign.summary()["ledger_history"] == history


def test_successful_replacement_matches_fresh_tp_and_original_checkpoint():
    owner, graph, campaign, _ = _campaign()
    checkpoint = campaign.checkpoint()
    before = _snapshot(graph, campaign)
    campaign.apply_candidate(1, _action(graph, 1, 3, 7))
    graph.propagate_incremental()
    new_action = _action(graph, 1, 3, 11, "BUF2")

    result = campaign.replace_candidate_tree(1, [new_action])
    graph.propagate_incremental()

    assert result["removed_action_ids"] == [7]
    assert campaign.summary()["action_ledger"] == {7: "replaced", 11: "active"}
    _compare(graph, owner, [new_action])
    campaign.restore(checkpoint)
    _assert_snapshot(graph, campaign, before)


def test_general_removal_then_replacement_and_restore_keep_ancestor_history():
    owner, graph, campaign, _ = _campaign()
    actions = []
    for action_id in (7, 3, 9):
        action = _action(graph, 1, 3, action_id)
        actions.append(action)
        campaign.apply_candidate(1, action)
    campaign.remove_candidate(1, 7)
    graph.propagate_incremental()
    _compare(graph, owner, actions[1:])
    checkpoint = campaign.checkpoint()
    before = _snapshot(graph, campaign)

    new_action = _action(graph, 1, 3, 11)
    result = campaign.replace_candidate_tree(1, [new_action])
    graph.propagate_incremental()

    assert result["removed_action_ids"] == [9, 3]
    _compare(graph, owner, [new_action])
    campaign.restore(checkpoint)
    _assert_snapshot(graph, campaign, before)


def test_single_late_liberty_failure_does_not_change_campaign_or_graph():
    _, graph, campaign, _ = _campaign()
    before = _snapshot(graph, campaign)

    with pytest.raises(ValueError):
        campaign.apply_candidate(1, _action(graph, 1, 3, 7, "MISSING_MASTER"))

    _assert_snapshot(graph, campaign, before)


def test_checkpoint_restores_existing_cell_resize_and_upstream_rc():
    _, graph, campaign, _ = _campaign()
    checkpoint = campaign.checkpoint()
    before = _snapshot(graph, campaign)
    graph.resize_cell("u", "BUF2")
    graph.propagate_incremental()
    assert graph.state_view()["pin_net_cap_rise"][0] > before["state"]["pin_net_cap_rise"][0]

    campaign.restore(checkpoint)

    _assert_snapshot(graph, campaign, before)


def test_foreign_campaign_checkpoint_and_changed_owner_mode_are_rejected():
    _, graph, campaign, _ = _campaign()
    before = _snapshot(graph, campaign)
    other = rebuffer_cpp.create_repair_timing_campaign_session(graph)
    legacy = rebuffer_cpp.create_repair_timing_campaign_session()

    with pytest.raises(ValueError, match="another timing campaign"):
        campaign.restore(other.checkpoint())
    with pytest.raises(ValueError, match="timing owner"):
        campaign.restore(legacy.checkpoint())
    with pytest.raises(ValueError, match="timing owner"):
        legacy.restore(campaign.checkpoint())

    _assert_snapshot(graph, campaign, before)


def test_discarded_branch_checkpoint_cannot_restore_a_same_epoch_fork():
    _, graph, campaign, _ = _campaign()
    root = campaign.checkpoint()
    campaign.apply_candidate(1, _action(graph, 1, 3, 7))
    graph.propagate_incremental()
    discarded = campaign.checkpoint()
    old_epoch = graph.summary()["epoch"]
    campaign.restore(root)
    campaign.apply_candidate(1, _action(graph, 1, 3, 9, "BUF2"))
    graph.propagate_incremental()
    assert graph.summary()["epoch"] == old_epoch
    before = _snapshot(graph, campaign)

    with pytest.raises(ValueError, match="stale journal history"):
        campaign.restore(discarded)

    _assert_snapshot(graph, campaign, before)


def test_committed_graph_invalidates_campaign_checkpoint_before_any_restore():
    _, graph, campaign, _ = _campaign()
    checkpoint = campaign.checkpoint()
    graph.commit(graph.checkpoint())
    before = _snapshot(graph, campaign)
    with pytest.raises(ValueError, match="stale journal history"):
        campaign.restore(checkpoint)
    _assert_snapshot(graph, campaign, before)


def test_native_campaign_requires_a_full_timing_mutation_owner():
    owner, _ = _graph(_base(), attach=False)
    empty = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    with pytest.raises(ValueError, match="initialized graph state"):
        rebuffer_cpp.create_repair_timing_campaign_session(empty)


def test_campaign_keeps_timing_owner_alive_and_rejects_topology_replacement():
    owner, graph, campaign, _ = _campaign()
    before = _snapshot(graph, campaign)
    with pytest.raises(ValueError, match="already attached"):
        campaign.attach_topology(1, owner.create_net_overlay("n1"))
    _assert_snapshot(graph, campaign, before)
    del graph

    campaign.apply_candidate(1, _action(campaign.timing_graph, 1, 3, 7))
    campaign.timing_graph.propagate_incremental()

    assert campaign.summary()["action_ledger"] == {7: "active"}
    assert campaign.timing_graph.mutation_index_oracle()["status"] == "ok"
