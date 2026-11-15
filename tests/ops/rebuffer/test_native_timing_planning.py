"""Native DP snapshots match the existing carrier consumer and own their views."""

from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.flows.baselines.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.current_driver_tree import map_current_driver_tree_actions
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _planning(base=None, config=None, *, attach=True):
    owner, graph = _graph(_base() if base is None else base, attach=attach)
    resolved = resolve_repair_timing_config(config)
    context = RebufferPlanContext(owner.pydb, owner, config=resolved)
    graph.initialize_queue(context.liberty, context.design["node_is_buffer"].tolist())
    graph.initialize_planning(
        context.design,
        context.characterization,
        steiner_payload(owner.pydb, owner),
        context.config,
        owner.r_unit,
        owner.c_unit,
    )
    graph.propagate()
    return owner, graph, context, resolved


def _reference(owner, graph, actions, config):
    exported = graph.export_carrier(
        steiner_payload(owner.pydb, owner), actions, owner.r_unit, owner.c_unit
    )
    carrier = CommittedTimingPyDB(owner.pydb, SimpleNamespace(**exported["physical"]))
    fresh = RebufferTimingModel(carrier, use_pin_offset=False, steiner_view=exported["steiner"])
    return RebufferPlanContext(carrier, fresh, config=config), exported


def _compare(owner, graph, actions, config):
    snapshot = graph.planning_snapshot(actions, config)
    reference, exported = _reference(owner, graph, actions, config)
    expected = reference.native_queue_views()["timing"]
    actual = snapshot.timing_view()
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=0, err_msg=key)
    rows = rebuffer_cpp.select_timing_queue_items(snapshot.queue_state(), config)
    for row in rows:
        root = int(row["root_net_id"])
        payload = reference.native_current_tree_payload(root)
        native_tree = snapshot.current_tree(root)
        assert rebuffer_cpp.export_current_driver_tree(native_tree) == (
            rebuffer_cpp.export_current_driver_tree(payload["current_tree"]())
        )
    return snapshot, reference, exported, rows


def test_native_gradient_scores_use_driver_names_and_reject_a_stale_graph_epoch():
    owner, graph, _, config = _planning()
    scores = {str(name): float(i + 1) for i, name in enumerate(owner.pydb.pin_names)}
    gradient = {
        **config, "gradient_ordering": True, "gradient_window_tiebreak": True,
        "gradient_slack_window_ps": 5.0,
        "gradient_score_epoch": int(graph.epoch), "gradient_score_by_driver_name": scores,
    }
    snapshot = graph.planning_snapshot([], gradient)
    queue = snapshot.queue_state()
    rows = rebuffer_cpp.select_timing_queue_items(queue, gradient)
    assert rows
    assert queue["gradient_scored_item_count"] == len(rows)
    assert [row["gradient_score"] for row in rows] == [
        scores[str(owner.pydb.pin_names[row["driver_pin_id"]])] for row in rows
    ]
    action, _ = _split(graph, 1, 3, 0, master="BUF2")
    graph.propagate_incremental()
    with pytest.raises(ValueError, match="current timing graph epoch"):
        graph.planning_snapshot([action], gradient)
    gradient["gradient_score_epoch"] = int(graph.epoch)
    current = graph.planning_snapshot([action], gradient).queue_state()
    assert current["gradient_score_epoch"] == graph.epoch


def test_native_gradient_cannot_select_global_gradient_first():
    _, graph, _, config = _planning()
    with pytest.raises(ValueError, match="primary slack window"):
        graph.planning_snapshot([], {**config, "gradient_ordering": True})


@pytest.mark.parametrize("per_driver", [False, True])
def test_native_snapshot_matches_current_tree_and_timing_before_and_after_split(per_driver):
    owner, graph, _, config = _planning(config={"native_tree_per_driver": per_driver})
    _compare(owner, graph, [], config)
    action, _ = _split(graph, 1, 3, 0, master="BUF2")
    graph.propagate_incremental()

    snapshot, _, _, _ = _compare(owner, graph, [action], config)

    assert snapshot.summary()["pin_count"] == 6
    assert graph.state_view_statistics()["exports"] == 0


def test_native_snapshot_compacts_retired_slots_and_keeps_old_views():
    owner, graph, _, config = _planning()
    actions = [_split(graph, 1, 3, i, master="BUF2")[0] for i in range(3)]
    graph.propagate_incremental()
    old, _, _, _ = _compare(owner, graph, actions, config)
    old_timing = old.timing_view()
    old_queue = rebuffer_cpp.select_timing_queue_items(old.queue_state(), config)
    checkpoint = graph.checkpoint()
    graph.remove_buffer(1, 1)
    graph.propagate_incremental()

    current, _, _, _ = _compare(owner, graph, [actions[0], actions[2]], config)

    assert current.summary()["pin_count"] == 8
    assert old.timing_view() == old_timing
    assert rebuffer_cpp.select_timing_queue_items(old.queue_state(), config) == old_queue
    graph.restore(checkpoint)
    assert old.source_nets([]) == []


def test_native_planning_and_projection_reuse_imported_physical_metadata():
    owner, graph, _, config = _planning()
    original = graph.planning_snapshot([], config)
    queue = original.queue_state()
    requests = [
        {**row, "net_id": int(row["root_net_id"]), "candidate_variant": 0}
        for row in rebuffer_cpp.select_timing_queue_items(queue, config)
    ]
    expected = [original.plan(request, config) for request in requests]
    projected = [original.project_actions(result["actions"]) for result in expected]
    assert any(projected)

    # A campaign owns these inputs after initialization. Neither a new queue
    # snapshot nor action mapping may re-import the original Python carrier.
    for field in (
        "pin_names",
        "node_names",
        "node_master_names",
        "net_names",
        "pin2node_map",
        "pin2net_map",
        "net2driver_pin_map",
        "pin_offset_x",
        "pin_offset_y",
        "pin_direct",
        "flat_libcell_names",
        "cell_id_2_arc_id_start",
        "flat_libarc_info",
        "node_x",
        "node_y",
        "node_size_x",
        "node_size_y",
        "flat_libcell_width",
        "flat_libcell_height",
        "num_terminals",
        "num_terminal_NIs",
    ):
        delattr(owner.pydb, field)

    current = graph.planning_snapshot([], config)
    observed = [current.plan(request, config) for request in requests]
    assert observed == expected
    assert [current.project_actions(result["actions"]) for result in observed] == projected


def test_native_planner_and_action_projection_match_existing_cpp_vgdp():
    owner, graph, _, config = _planning()
    snapshot, reference, exported, rows = _compare(owner, graph, [], config)
    assert rows
    projected_count = 0
    for item in rows:
        root = int(item["root_net_id"])
        request = {**item, "net_id": root, "candidate_variant": 0}
        actual = snapshot.plan(request, config)
        payload = reference.native_current_tree_payload(root)
        result = rebuffer_cpp.optimize_current_driver_tree(
            payload["design"],
            payload["timing"],
            payload["liberty"],
            payload["characterization"],
            payload["config"],
            payload["current_tree"](),
        )
        assert result["status"] == "ok"
        candidates = [row for row in result["candidates"] if row["actions"]]
        assert actual["actions"] == (candidates[0]["actions"] if candidates else [])
        if actual["actions"]:
            projected_count += len(actual["actions"])
            mapped = map_current_driver_tree_actions(actual["actions"], reference.pydb)
            expected = rebuffer_cpp.project_overlay_actions(
                owner.pydb,
                exported,
                [owner.create_net_overlay(name) for name in owner.pydb.net_names],
                mapped,
            )
            assert snapshot.project_actions(actual["actions"]) == expected
    assert projected_count > 0


def test_native_planning_rejects_stale_and_discarded_branch_before_projection():
    _, graph, _, config = _planning()
    root = graph.checkpoint()
    _split(graph, 1, 3, 0)
    graph.propagate_incremental()
    # The action metadata must refer to this exact active topology.
    action = {"session_buffer_id": 0, "buffer_master_name": "BUF1"}
    snapshot = graph.planning_snapshot([action], config)
    graph.resize_cell("u", "BUF2")
    graph.propagate_incremental()
    with pytest.raises(ValueError, match="stale"):
        snapshot.project_actions([])
    graph.restore(root)
    _split(graph, 1, 3, 0, master="BUF2")
    graph.propagate_incremental()
    with pytest.raises(ValueError, match="stale journal history"):
        snapshot.source_nets([])


def test_planning_inputs_are_immutable():
    owner, graph, context, _ = _planning()
    with pytest.raises(ValueError, match="immutable"):
        graph.initialize_planning(
            context.design,
            context.characterization,
            steiner_payload(owner.pydb, owner),
            context.config,
            owner.r_unit,
            owner.c_unit,
        )


def test_native_planning_snapshot_requires_initialized_inputs():
    _, graph = _graph(_base())
    with pytest.raises(ValueError, match="not initialized"):
        graph.planning_snapshot([], {})


@pytest.mark.parametrize("field", ["pin_names", "net_ids"])
def test_planning_metadata_mismatch_rejects_before_publishing_inputs(field):
    owner, graph = _graph(_base())
    context = RebufferPlanContext(owner.pydb, owner)
    graph.initialize_queue(context.liberty, context.design["node_is_buffer"].tolist())
    design = dict(context.design)
    steiner = steiner_payload(owner.pydb, owner)
    if field == "pin_names":
        design["names"] = {**design["names"], "pin": tuple(reversed(design["names"]["pin"]))}
    else:
        steiner["net_ids"] = list(reversed(steiner["net_ids"]))

    with pytest.raises(ValueError, match="planning"):
        graph.initialize_planning(
            design, context.characterization, steiner, context.config, owner.r_unit, owner.c_unit
        )

    graph.initialize_planning(
        context.design,
        context.characterization,
        steiner_payload(owner.pydb, owner),
        context.config,
        owner.r_unit,
        owner.c_unit,
    )
    graph.propagate()
    assert graph.planning_snapshot([], {}).summary()["pin_count"] == 4


def test_native_snapshot_preserves_undriven_net_with_steiner_topology():
    base = _base()
    base.num_nodes = 5
    base.num_terminal_NIs = 4
    base.node_names.extend(["PAD", "PORT_OUT"])
    base.node_master_names.extend(["", ""])
    base.node_x.extend([20000.0, 20000.0])
    base.node_y.extend([0.0, 1000.0])
    base.node_size_x.extend([0.0, 0.0])
    base.node_size_y.extend([0.0, 0.0])
    base.inst_main_id.extend([-1, -1])
    base.inst_libcell_offset.extend([0, 0])
    base.inst_flat_arcs_start.extend([1, 1])
    base.pin_names.extend(["PORT_OUT", "PAD:PAD"])
    base.pin2node_map.extend([4, 3])
    base.pin_to_inst_id.extend([4, 3])
    base.pin_to_node_id.extend([4, 3])
    base.pin_offset_x.extend([0.0, 0.0])
    base.pin_offset_y.extend([0.0, 0.0])
    base.pin_direct.extend(["R0", "R0"])
    base.pin_2_libpin_offset.extend([-1, -1])
    base.pin2net_map.extend([2, 2])
    base.net_names.append("undriven_pad_net")
    base.flat_net2pin_map.extend([4, 5])
    base.flat_net2pin_start_map.append(6)
    base.net2driver_pin_map.append(-1)
    base.net_flat_arcs_start.append(2)
    base.flat_pin_to_graph_start.extend([3, 3])
    base.flat_pin_to_graph_start_reverse.extend([3, 3])
    base.pin_pred_start.extend([3, 3])
    base.pin_succ_start.extend([3, 3])

    owner, graph, _, config = _planning(base=base, attach=False)
    snapshot, _, _, _ = _compare(owner, graph, [], config)
    assert snapshot.summary()["net_count"] == 3
    graph.attach_net_topology_state(1, graph.create_net_topology(1))
    action, _ = _split(graph, 1, 3, 0, master="BUF2")
    graph.propagate_incremental()
    after_split, _, _, _ = _compare(owner, graph, [action], config)
    assert after_split.summary()["net_count"] == 4
