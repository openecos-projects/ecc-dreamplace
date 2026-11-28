"""Non-LIFO buffer removal must preserve descendants, projection and exact undo."""

import numpy as np
import pytest
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _compare, _split
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


@pytest.mark.parametrize("removed", [0, 1])
def test_remove_ancestor_or_middle_keeps_current_descendants_and_commit_projection(removed):
    owner, graph = _graph(_base())
    root = graph.checkpoint()
    original, original_meta = graph.state_view(), graph.mutation_view()
    actions = [
        _split(graph, 1, 3, action_id, master)[0]
        for action_id, master in ((20, "BUF1"), (3, "BUF2"), (11, "BUF1"))
    ]
    _compare(graph, owner, actions)
    checkpoint = graph.checkpoint()
    before, before_meta = graph.state_view(), graph.mutation_view()

    graph.remove_buffer(1, actions[removed]["action_id"])
    survivors = [action for index, action in enumerate(actions) if index != removed]
    _compare(graph, owner, survivors)

    projected = graph.finalize_actions(survivors)
    assert [
        (
            action["session_buffer_id"],
            action["parent_action_id"],
            action["segment_tree_depth"],
            action["downstream_pin_names"],
        )
        for action in projected
    ] == [
        (survivors[0]["action_id"], -1, 0, ["OUT"]),
        (survivors[1]["action_id"], survivors[0]["action_id"], 1, ["OUT"]),
    ]
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), before_meta, bitwise=True)
    graph.restore(root)
    _same_state(graph.state_view(), original, bitwise=True)
    _same_state(graph.mutation_view(), original_meta, bitwise=True)


def test_remove_middle_compact_resize_insert_and_restore_keep_full_tp_oracle():
    owner, graph = _graph(_base())
    actions = [_split(graph, 1, 3, action_id)[0] for action_id in (10, 11, 12)]
    graph.remove_buffer(1, 11)
    actions.pop(1)
    _compare(graph, owner, actions)
    graph.commit(graph.checkpoint())

    maps = graph.compact()

    assert maps["changed"]
    assert graph.summary()["pin_count"] == graph.summary()["active_pin_count"]
    _compare(graph, owner, actions)
    checkpoint = graph.checkpoint()
    before, before_meta = graph.state_view(), graph.mutation_view()
    graph.resize_buffer(1, {"action_id": 12, "buffer_master_name": "BUF2"})
    resized = [actions[0], dict(actions[1], buffer_master_name="BUF2")]
    _compare(graph, owner, resized)
    newest, _ = _split(graph, 1, 3, 14)
    _compare(graph, owner, resized + [newest])
    graph.remove_buffer(1, 10)
    _compare(graph, owner, [resized[1], newest])
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), before_meta, bitwise=True)


def test_checkpoint_after_middle_removal_restores_that_branch_exactly():
    owner, graph = _graph(_base())
    actions = [_split(graph, 1, 3, action_id)[0] for action_id in (7, 2, 8)]
    graph.remove_buffer(1, 2)
    actions.pop(1)
    _compare(graph, owner, actions)
    checkpoint = graph.checkpoint()
    before, before_meta = graph.state_view(), graph.mutation_view()

    graph.remove_buffer(1, 7)
    _compare(graph, owner, [actions[1]])
    _split(graph, 0, 1, 15)
    graph.restore(checkpoint)

    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), before_meta, bitwise=True)
    repeated = _compare(graph, owner, actions)
    assert repeated["changed_pins"] == []
    np.testing.assert_array_equal(graph.mutation_view()["pin_active"], before_meta["pin_active"])
