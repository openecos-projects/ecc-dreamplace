"""Native compaction preserves live values, original ids and subsequent mutation."""

import numpy as np
import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _compare, _split
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _seal(graph):
    graph.commit(graph.checkpoint())


def _live_gap():
    owner, graph = _graph(_base())
    first, _ = _split(graph, 0, 1, 10)
    second, _ = _split(graph, 0, 1, 11)
    third, _ = _split(graph, 1, 3, 12, "BUF2")
    graph.remove_buffer(0, second["action_id"])
    _compare(graph, owner, [first, third])
    return owner, graph, [first, third]


def _assert_gathered(graph, previous, maps):
    current = graph.state_view()
    for name, values in previous.items():
        if name in ("status", "epoch"):
            continue
        mapping = maps["cell_arc_old_to_new" if name.startswith("cell_arc_") else "pin_old_to_new"]
        np.testing.assert_array_equal(
            current[name].view(np.uint64), values[mapping >= 0].view(np.uint64), err_msg=name
        )


def test_compact_maps_live_slots_and_preserves_original_constraints_and_raw_topology():
    owner, graph, actions = _live_gap()
    previous = graph.state_view()
    topologies = [graph.topology_view(net) for net in (0, 1)]
    _seal(graph)
    stale = graph.checkpoint()

    result = graph.compact()

    assert result["changed"]
    assert {
        name: result[name].tolist()
        for name in ("pin_old_to_new", "cell_arc_old_to_new", "node_old_to_new", "net_old_to_new")
    } == {
        "pin_old_to_new": [0, 1, 2, 3, 4, 5, -1, -1, 6, 7],
        "cell_arc_old_to_new": [0, 1, -1, 2],
        "node_old_to_new": [0, 1, 2, 3, -1, 4],
        "net_old_to_new": [0, 1, 2, -1, 3],
    }
    assert graph.summary()["pin_count"] == graph.summary()["active_pin_count"] == 8
    assert graph.summary()["cell_arc_count"] == graph.summary()["active_cell_arc_count"] == 3
    assert [graph.topology_view(net) for net in (0, 1)] == topologies
    _assert_gathered(graph, previous, result)
    repeated = _compare(graph, owner, actions)
    assert repeated["changed_pins"] == repeated["changed_endpoints"] == []
    with pytest.raises(ValueError, match="stale journal history"):
        graph.restore(stale)
    action, _ = _split(graph, 0, 1, 13, "BUF2")
    _compare(graph, owner, actions + [action])


def test_compact_only_dead_arc_slots_and_keep_library_metadata_for_future_resize():
    base = _base()
    base.cell_id_2_arc_id_start = [0, 1, 3]
    base.flat_libarc_info.append(["A", "Y", 1, 1, 1, 0])
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        for suffix in ("values", "trans_table", "cap_table", "dim"):
            values = getattr(base, f"{prefix}_flat_luts_{suffix}")
            values.append(list(values[-1]))
    owner, graph = _graph(base)
    action, _ = _split(graph, 1, 3, 10)
    graph.resize_buffer(1, {"action_id": 10, "buffer_master_name": "BUF2"})
    graph.resize_buffer(1, {"action_id": 10, "buffer_master_name": "BUF1"})
    _compare(graph, owner, [action])
    previous = graph.state_view()
    _seal(graph)

    result = graph.compact()

    assert result["changed"]
    np.testing.assert_array_equal(result["cell_arc_old_to_new"], [0, -1, -1, -1, 1])
    np.testing.assert_array_equal(result["pin_old_to_new"], np.arange(6))
    _assert_gathered(graph, previous, result)
    _compare(graph, owner, [action])
    graph.resize_buffer(1, {"action_id": 10, "buffer_master_name": "BUF2"})
    _compare(graph, owner, [dict(action, buffer_master_name="BUF2")])


def test_compact_requires_a_committed_boundary_and_failure_is_atomic():
    _, graph, _ = _live_gap()
    previous, metadata, summary = graph.state_view(), graph.mutation_view(), graph.summary()

    with pytest.raises(ValueError, match="committed journal boundary"):
        graph.compact()

    assert graph.summary() == summary
    _same_state(graph.state_view(), previous, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_generic_scratch_state_is_not_silently_remapped_as_pin_state():
    _, graph, _ = _live_gap()
    graph.set_state({"diagnostic": np.array([1.0, 2.0])})
    _seal(graph)
    previous, metadata, summary = graph.state_view(), graph.mutation_view(), graph.summary()

    with pytest.raises(ValueError, match="generic scratch fields"):
        graph.compact()

    assert graph.summary() == summary
    _same_state(graph.state_view(), previous, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_noop_compact_has_identity_maps_and_keeps_current_checkpoints_valid():
    _, graph = _graph(_base())
    _seal(graph)
    checkpoint = graph.checkpoint()
    previous, summary = graph.state_view(), graph.summary()

    result = graph.compact()

    assert not result["changed"]
    assert graph.summary() == summary
    np.testing.assert_array_equal(result["pin_old_to_new"], np.arange(4))
    np.testing.assert_array_equal(result["cell_arc_old_to_new"], [0])
    graph.set_state({"pin_rAAT": np.ones(4)})
    graph.restore(checkpoint)
    _same_state(graph.state_view(), previous, bitwise=True)


def test_compact_preserves_nan_payloads_infinities_and_signed_zero_bits():
    _, graph, _ = _live_gap()
    previous = graph.state_view()
    bits = previous["pin_rAAT"].view(np.uint64).copy()
    bits[[0, 4, 8, 9]] = [
        0x7FF8000000000012,
        0xFFF0000000000000,
        0x7FF0000000000000,
        0x8000000000000000,
    ]
    graph.set_state({"pin_rAAT": bits.view(np.float64)})
    previous = graph.state_view()
    _seal(graph)

    maps = graph.compact()

    _assert_gathered(graph, previous, maps)


def test_compact_requires_native_state_owner():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())

    with pytest.raises(ValueError, match="initialized native mutation"):
        graph.compact()


def test_mutation_restore_and_compact_cycles_keep_full_tp_parity():
    owner, graph = _graph(_base())
    actions = []
    for attempt in range(40):
        net, child = (0, 1) if attempt % 2 else (1, 3)
        current = [index for index, action in enumerate(actions) if action["source_net"] == net]
        if current and attempt % 3 == 0:
            index = current[-1]
            graph.remove_buffer(net, actions[index]["action_id"])
            actions.pop(index)
        else:
            action, _ = _split(graph, net, child, 100 + attempt, "BUF2" if attempt % 3 else "BUF1")
            actions.append(dict(action, source_net=net))
        _compare(graph, owner, actions)
        if attempt % 5 == 0:
            _seal(graph)
            previous = graph.state_view()
            maps = graph.compact()
            _assert_gathered(graph, previous, maps)
            _compare(graph, owner, actions)
