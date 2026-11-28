"""Source RC preserves electrical results independently of global node labels."""

import numpy as np
import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_events import _assert_event
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _source_graph(steiner):
    _, graph = _graph(_base(), attach=False)
    graph.attach_net_topology(
        0,
        0,
        steiner + 1,
        {
            "edges": [
                {"parent": 0, "child": steiner, "r": 4.57, "c": 0.000945},
                {"parent": steiner, "child": 1, "r": 4.57, "c": 0.000945},
            ],
            "node_loads": {1: [1]},
            "coordinates": {0: [0.0, 0.0], steiner: [5000.0, 0.0], 1: [10000.0, 0.0]},
        },
    )
    graph.propagate()
    graph.commit(graph.checkpoint())
    return graph


def test_shared_rc_evaluator_matches_hand_computed_branch_in_all_three_cap_modes():
    # Pi half-caps give nodal wire caps [5, 2, 3]. Independent leaf branches
    # have delay R*C and impulse (R*C)^2; root load includes both branches.
    rc = {
        "affected_nodes": [0, 1, 2],
        "parent": [0, 0, 0],
        "child_start": [0, 2, 2, 2],
        "child": [1, 2],
        "topo_start": [0, 3],
        "topo": [0, 1, 2],
        "edge_resistance": [0.0, 2.0, 3.0],
        "edge_capacitance": [0.0, 4.0, 6.0],
    }

    result = rebuffer_cpp.evaluate_incremental_rc(
        rc,
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 4.0]),
        np.array([0.0, 3.0, 6.0]),
        {"slots": []},
        3,
    )

    np.testing.assert_array_equal(result.pop("affected_nodes"), [0, 1, 2])
    assert result == {
        "loads_generic": [13.0, 3.0, 5.0],
        "delays_generic": [0.0, 6.0, 15.0],
        "impulses_generic": [0.0, 36.0, 225.0],
        "loads_rise": [16.0, 4.0, 7.0],
        "delays_rise": [0.0, 8.0, 21.0],
        "impulses_rise": [0.0, 64.0, 441.0],
        "loads_fall": [19.0, 5.0, 9.0],
        "delays_fall": [0.0, 10.0, 27.0],
        "impulses_fall": [0.0, 100.0, 729.0],
    }


def test_sparse_steiner_ids_keep_rc_work_local_and_all_state_bits_identical():
    regular, sparse = _source_graph(4), _source_graph(1_000_000_000)
    roots = [graph.checkpoint() for graph in (regular, sparse)]
    original = regular.state_view()
    for step in range(8):
        for graph in (regular, sparse):
            before = graph.mutation_statistics()
            if step < 3:
                _split(graph, 0, 1, 10 + step)
            elif step == 3:
                graph.resize_buffer(0, {"action_id": 11, "buffer_master_name": "BUF2"})
            elif step == 4:
                graph.remove_buffer(0, 11)
            elif step == 5:
                graph.resize_cell("u", "BUF2")
            elif step == 6:
                graph.remove_buffer(0, 10)
            else:
                graph.remove_buffer(0, 12)
            buffers = len(graph.topology_view(0)["buffers"])
            after = graph.mutation_statistics()
            assert after["rc_builds"] - before["rc_builds"] == 1
            assert after["rc_nodes"] - before["rc_nodes"] == 3 + 2 * buffers
            assert after["rc_pin_reads"] - before["rc_pin_reads"] == 2 * (2 + 2 * buffers)
            _assert_event(graph, None)
            assert graph.mutation_index_oracle()["status"] == "ok"
        _same_state(sparse.state_view(), regular.state_view(), bitwise=True)

    for graph, root in zip((regular, sparse), roots, strict=True):
        graph.restore(root)
        _same_state(graph.state_view(), original, bitwise=True)


def test_physical_source_mismatch_rolls_back_resize_values_and_event_baseline():
    _, graph = _graph(_base(), attach=False)
    graph.attach_net_topology(
        0,
        0,
        4,
        {
            "edges": [{"parent": 0, "child": 3, "r": 1.0, "c": 0.001}],
            "node_loads": {3: [3]},
        },
    )
    graph.propagate()
    graph.commit(graph.checkpoint())
    state, metadata, summary = graph.state_view(), graph.mutation_view(), graph.summary()

    with pytest.raises(ValueError, match="physical pins do not match source net"):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == summary
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)
    assert graph.propagate_incremental()["mode"] == "events"
