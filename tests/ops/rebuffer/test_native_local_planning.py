"""Selected-tree views preserve canonical tree, VGDP and projection identities."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.rebuffer.test_native_timing_planning import _planning
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_resize import _chain


def _config(config):
    return {
        **config,
        "native_endpoint_pin": 3,
        "native_tree_per_driver": True,
        "native_tree_branch_queue": False,
        "native_tree_round_robin": False,
        "use_native_timing_path_context": True,
    }


def _compare(graph, actions, config, endpoint=3):
    config = {**_config(config), "native_endpoint_pin": endpoint}
    full = graph.planning_snapshot(actions, config)
    local = graph.local_planning_snapshot(actions, config)
    rows = rebuffer_cpp.select_timing_queue_items(full.queue_state(), config)
    actual = rebuffer_cpp.select_timing_queue_items(local.queue_state(), config)
    assert actual == rows
    for row in rows:
        root = row["root_net_id"]
        assert rebuffer_cpp.export_current_driver_tree(local.current_tree(root)) == (
            rebuffer_cpp.export_current_driver_tree(full.current_tree(root))
        )
        request = {**row, "net_id": root, "candidate_variant": 0}
        expected = full.plan(request, config)
        result = local.plan(request, config)
        assert result == expected
        assert local.source_nets(result["actions"]) == full.source_nets(expected["actions"])
        assert local.project_actions(result["actions"]) == full.project_actions(expected["actions"])
    return local, rows


def test_local_views_match_full_before_and_after_nested_mutations():
    _, graph, _, config = _planning()
    root = graph.checkpoint()
    _compare(graph, [], config)
    first = _split(graph, 1, 3, 5, master="BUF2")[0]
    graph.propagate_incremental()
    _compare(graph, [first], config)
    graph.resize_buffer(1, {"action_id": 5, "buffer_master_name": "BUF1"})
    first = {**first, "buffer_master_name": "BUF1"}
    graph.propagate_incremental()
    _compare(graph, [first], config)
    graph.resize_cell("u", "BUF2")
    graph.propagate_incremental()
    _compare(graph, [first], config)
    second = _split(graph, 1, 3, 2, master="BUF1")[0]
    graph.propagate_incremental()
    _compare(graph, [first, second], config)
    graph.remove_buffer(1, 5)
    graph.propagate_incremental()
    _compare(graph, [second], config)
    graph.commit(graph.checkpoint())
    graph.compact()
    _compare(graph, [second], config)
    # A committed/compacted graph has a different journal generation.
    with pytest.raises(ValueError):
        graph.restore(root)


def test_local_views_preserve_original_buffer_boundaries_and_capture_only_selected_owners():
    _, graph, _, config = _planning(base=_chain())
    local, rows = _compare(graph, [], config, endpoint=5)
    assert rows
    single = graph.local_planning_snapshot([], {**_config(config), "native_endpoint_pin": 5})
    root = max(row["root_net_id"] for row in rows)
    row = next(row for row in rows if row["root_net_id"] == root)
    single.plan({**row, "net_id": root}, config)
    assert 0 < single.summary()["local_filled_pin_count"] < single.summary()["pin_count"]


def test_local_views_keep_old_captured_trees_and_reject_stale_planning():
    _, graph, _, config = _planning()
    root = graph.checkpoint()
    local, rows = _compare(graph, [], config)
    request = {**rows[0], "net_id": rows[0]["root_net_id"]}
    before = local.plan(request, config)
    old_tree = rebuffer_cpp.export_current_driver_tree(local.current_tree(request["net_id"]))
    _split(graph, 1, 3, 0)
    graph.propagate_incremental()
    assert (
        rebuffer_cpp.export_current_driver_tree(local.current_tree(request["net_id"])) == old_tree
    )
    with pytest.raises(ValueError, match="stale"):
        local.plan(request, config)
    graph.restore(root)
    assert local.plan(request, config) == before


def test_local_views_match_cross_net_buffer_chains():
    _, graph, _, config = _planning(base=_chain())
    first = _split(graph, 0, 1, 9, master="BUF2")[0]
    second = _split(graph, 2, 5, 3, master="BUF1")[0]
    graph.propagate_incremental()
    _compare(graph, [first, second], config, endpoint=5)


def test_local_projection_rejects_sources_and_loads_outside_the_captured_domain():
    _, graph, _, config = _planning()
    config = _config(config)
    local = graph.local_planning_snapshot([], config)
    rows = rebuffer_cpp.select_timing_queue_items(local.queue_state(), config)
    actions = []
    for row in rows:
        actions = local.plan({**row, "net_id": row["root_net_id"]}, config)["actions"]
        if actions:
            break
    assert actions
    with pytest.raises(ValueError, match="partition membership"):
        local.project_actions([{**actions[0], "source_parent_node_id": -1}])
    with pytest.raises(ValueError, match="tree membership"):
        local.project_actions([{**actions[0], "downstream_pins": [0]}])
