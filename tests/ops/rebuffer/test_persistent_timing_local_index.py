"""Local graph edits retain canonical ranks, ordered operations and exact restore."""

import numpy as np
import pytest
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base, _reference_after
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_events import _assert_event
from tests.ops.rebuffer.test_persistent_timing_propagation import _checks_base
from tests.ops.rebuffer.test_persistent_timing_resize import _chain, _graph, _same_state


def test_insert_and_restore_do_not_build_a_full_index_or_schedule():
    _, graph = _graph(_base())
    graph.propagate()
    graph.commit(graph.checkpoint())
    checkpoint = graph.checkpoint()
    original = graph.mutation_view()
    before = graph.mutation_statistics()

    _split(graph, 1, 3, 7)

    after = graph.mutation_statistics()
    assert after["index_builds"] == before["index_builds"]
    assert after["schedule_builds"] == before["schedule_builds"]
    assert after["local_updates"] == before["local_updates"] + 1
    assert after["rank_visits"] > before["rank_visits"]
    assert graph.mutation_index_oracle()["status"] == "ok"
    _assert_event(graph, None)
    before = graph.mutation_statistics()
    graph.restore(checkpoint)
    after = graph.mutation_statistics()

    assert after["index_builds"] == before["index_builds"]
    assert after["schedule_builds"] == before["schedule_builds"]
    _same_state(graph.mutation_view(), original, bitwise=True)
    assert graph.mutation_index_oracle()["status"] == "ok"


def test_net_slots_survive_retarget_retire_and_committed_compact():
    _, graph = _graph(_base())
    original = graph.mutation_view()
    _split(graph, 1, 3, 7)
    inserted = graph.mutation_view()
    original_slot = int(np.flatnonzero(original["net_arcs"][:, 1] == 3)[0])

    assert inserted["net_arcs"][original_slot, 1] == 3
    assert inserted["net_arcs"][original_slot, 0] != original["net_arcs"][original_slot, 0]
    graph.remove_buffer(1, 7)
    retired = graph.mutation_view()
    np.testing.assert_array_equal(
        retired["net_arcs"][: len(original["net_arcs"])], original["net_arcs"]
    )
    assert len(retired["net_arcs"]) > len(original["net_arcs"])
    assert not all(retired["net_arc_active"])
    assert graph.mutation_index_oracle()["status"] == "ok"
    graph.propagate_incremental()
    graph.commit(graph.checkpoint())
    mapping = graph.compact()["net_arc_old_to_new"]

    np.testing.assert_array_equal(mapping < 0, np.logical_not(retired["net_arc_active"]))
    assert graph.mutation_index_oracle()["status"] == "ok"
    np.testing.assert_array_equal(graph.mutation_view()["net_arcs"], original["net_arcs"])


def test_middle_removal_decreases_downstream_rank_and_restore_recovers_it():
    _, graph = _graph(_base())
    graph.propagate()
    original_rank = graph.mutation_view()["pin_rank"][3]
    for action in (5, 1, 9):
        _split(graph, 1, 3, action)
        assert graph.mutation_index_oracle()["status"] == "ok"
    checkpoint = graph.checkpoint()
    before = graph.mutation_view()

    graph.remove_buffer(1, 1)

    assert original_rank < graph.mutation_view()["pin_rank"][3] < before["pin_rank"][3]
    assert graph.mutation_index_oracle()["status"] == "ok"
    _assert_event(graph, None)
    graph.restore(checkpoint)
    _same_state(graph.mutation_view(), before, bitwise=True)
    assert graph.mutation_index_oracle()["status"] == "ok"


def test_existing_cell_resize_after_buffering_keeps_the_local_graph_schedule():
    _, graph = _graph(_base())
    _split(graph, 1, 3, 7)
    graph.propagate_incremental()
    before = graph.mutation_statistics()

    graph.resize_cell("u", "BUF2")

    after = graph.mutation_statistics()
    assert after["index_builds"] == before["index_builds"]
    assert after["schedule_builds"] == before["schedule_builds"]
    assert graph.mutation_index_oracle()["status"] == "ok"
    _assert_event(graph, None)


def test_clock_local_namespace_setup_and_recovery_survive_local_data_net_edits():
    _, graph = _graph(_checks_base())
    graph.propagate()
    _split(graph, 1, 3, 7, "BUF2")

    assert graph.mutation_index_oracle()["status"] == "ok"
    _assert_event(graph, None)
    # The manually indexed clock fixture carries an explicit clock prefix;
    # retain it in the independent two-buffer Python TP reference as well.
    reference = _reference_after(_checks_base())
    reference.node_x[1] = 55000.0
    reference.flat_inst_arcs_by_level = [
        [0, 2, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 1, 0, 0, 0],
        [3, 4, 1, 1, 1, 0, 0, 1],
    ]
    reference.flat_inst_arcs_by_level_start = [0, 1, 2, 3]
    reference.endpoints_constraint_arcs[:, 1] = 5
    reference.endpoints_timing_check_arcs[:, 1] = 5
    expected = persistent_timing_state_payload(RebufferTimingModel(reference, use_pin_offset=False))
    observed = graph.state_view()
    for field, values in expected.items():
        actual = (
            observed[field]
            if field.startswith("cell_arc_")
            else observed[field][[0, 1, 2, 4, 5, 3]]
        )
        np.testing.assert_allclose(actual, values, rtol=0, atol=1e-6, err_msg=field)


def test_late_duplicate_pin_failure_restores_index_rows_and_identity_maps():
    base = _base()
    base.flat_libarc_info[1][0] = "Y"
    _, graph = _graph(base)
    graph.propagate()
    graph.commit(graph.checkpoint())
    before = graph.summary()
    state, metadata = graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError, match="duplicate timing pin"):
        _split(graph, 1, 3, 7, "BUF2")

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)
    assert graph.mutation_index_oracle()["status"] == "ok"
    _split(graph, 1, 3, 7)
    assert graph.mutation_index_oracle()["status"] == "ok"
    _assert_event(graph, None)


def test_mixed_local_graph_mutations_restore_and_compact_match_full_graph():
    _, graph = _graph(_chain())
    graph.propagate()
    metadata = graph.mutation_view()
    sinks = {
        net: next(pin for pin in metadata["net_pins"][net] if pin != metadata["net_drivers"][net])
        for net in (0, 1)
    }
    actions = []
    for step in range(100):
        checkpoint = graph.checkpoint()
        before, state = graph.mutation_view(), graph.state_view()
        previous = [dict(action) for action in actions]
        if len(actions) < 3 or step % 4 == 0:
            net = step % 2
            # Pick an original terminal even after an ancestor split.
            _split(graph, net, sinks[net], 100 + step)
            actions.append({"net": net, "id": 100 + step, "master": "BUF1"})
        elif step % 4 == 2:
            action = actions[step % len(actions)]
            action["master"] = "BUF2" if action["master"] == "BUF1" else "BUF1"
            graph.resize_buffer(
                action["net"], {"action_id": action["id"], "buffer_master_name": action["master"]}
            )
        else:
            action = actions.pop(0 if step % 4 == 1 else -1)
            graph.remove_buffer(action["net"], action["id"])
        assert graph.mutation_index_oracle()["status"] == "ok"
        _assert_event(graph, None)
        if step % 7 == 0:
            graph.restore(checkpoint)
            actions = previous
            _same_state(graph.mutation_view(), before, bitwise=True)
            _same_state(graph.state_view(), state, bitwise=True)
            assert graph.mutation_index_oracle()["status"] == "ok"
        if step % 15 == 0:
            graph.commit(graph.checkpoint())
            graph.compact()
            assert graph.mutation_index_oracle()["status"] == "ok"
