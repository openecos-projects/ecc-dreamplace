"""Read-only native state snapshots survive writes, rollback and owner deletion."""

import gc

import numpy as np
import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _arrays(view):
    return {name: value for name, value in view.items() if isinstance(value, np.ndarray)}


def _snapshot(graph):
    view = graph.state_view(copy=False)
    bits = {name: value.view(np.uint64).copy() for name, value in _arrays(view).items()}
    return view, bits


def _unchanged(snapshot):
    view, bits = snapshot
    assert set(_arrays(view)) == set(bits)
    for name, expected in bits.items():
        assert not view[name].flags.writeable
        np.testing.assert_array_equal(view[name].view(np.uint64), expected)


def _committed_graph():
    _, graph = _graph(_base())
    graph.propagate()
    graph.commit(graph.checkpoint())
    return graph


def test_zero_copy_views_share_native_storage_and_default_copies_stay_independent():
    graph = _committed_graph()
    first, second = graph.state_view(copy=False), graph.state_view(copy=False)
    copied = graph.state_view()

    for name, value in _arrays(first).items():
        assert np.shares_memory(value, second[name])
        assert not np.shares_memory(value, copied[name])
        assert not value.flags.owndata
        np.testing.assert_array_equal(value.view(np.uint64), copied[name].view(np.uint64))
    before = first["pin_rAAT"].copy()
    copied["pin_rAAT"][0] += 100.0
    np.testing.assert_array_equal(first["pin_rAAT"], before)
    with pytest.raises(ValueError, match="read-only"):
        first["pin_rAAT"][0] = 10.0
    with pytest.raises(ValueError):
        first["pin_rAAT"].setflags(write=True)


@pytest.mark.parametrize("operation", ["patch", "full_seed", "reinitialize"])
def test_retained_views_detach_before_sparse_and_full_state_writes(operation):
    graph = _committed_graph()
    previous = _snapshot(graph)
    before = graph.state_view_statistics()
    if operation == "patch":
        replacement = graph.state_view()["pin_rAAT"] + 7.0
        graph.set_state({"pin_rAAT": replacement})
    elif operation == "full_seed":
        graph.propagate()
    else:
        graph.initialize_timing_state(_arrays(graph.state_view()))

    _unchanged(previous)
    current = graph.state_view(copy=False)
    assert all(
        not np.shares_memory(current[name], values) for name, values in _arrays(previous[0]).items()
    )
    after = graph.state_view_statistics()
    assert after["detached_fields"] - before["detached_fields"] == len(previous[1])
    assert after["detached_values"] - before["detached_values"] == sum(
        len(values) for values in previous[1].values()
    )


def test_snapshots_keep_all_bits_across_growth_event_resize_removal_restore_and_compact():
    graph = _committed_graph()
    root = graph.checkpoint()
    snapshots = [_snapshot(graph)]
    _split(graph, 1, 3, 7)
    snapshots.append(_snapshot(graph))
    graph.propagate_incremental()
    snapshots.append(_snapshot(graph))
    graph.resize_cell("u", "BUF2")
    graph.propagate_incremental()
    snapshots.append(_snapshot(graph))
    _split(graph, 1, 3, 8, "BUF2")
    graph.propagate_incremental()
    snapshots.append(_snapshot(graph))
    graph.remove_buffer(1, 7)
    graph.propagate_incremental()
    snapshots.append(_snapshot(graph))
    graph.restore(root)
    graph.commit(graph.checkpoint())
    _split(graph, 1, 3, 9)
    graph.propagate_incremental()
    graph.remove_buffer(1, 9)
    graph.propagate_incremental()
    graph.commit(graph.checkpoint())
    snapshots.append(_snapshot(graph))

    assert graph.compact()["changed"]
    assert len(graph.state_view()["pin_rAAT"]) < len(snapshots[-1][0]["pin_rAAT"])
    del graph
    gc.collect()
    for snapshot in snapshots:
        _unchanged(snapshot)


@pytest.mark.parametrize("detached", [False, True])
def test_a_derived_numpy_view_outlives_the_graph_and_its_original_array(detached):
    graph = _committed_graph()
    array = graph.state_view(copy=False)["pin_rAAT"]
    derived = array[::-1]
    bits = derived.view(np.uint64).copy()
    del array
    if detached:
        graph.set_state({"pin_rAAT": graph.state_view()["pin_rAAT"] + 3.0})
        assert graph.state_view_statistics()["detached_fields"] == 1
    del graph
    gc.collect()

    np.testing.assert_array_equal(derived.view(np.uint64), bits)
    assert not derived.flags.writeable


def test_empty_generic_arrays_and_nonfinite_bits_survive_replacement_erase_and_commit():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())
    root = graph.checkpoint()
    graph.set_state({"scratch": np.empty(0, dtype=np.float64)})
    empty = _snapshot(graph)
    bits = np.array([0x7FF8000000000123, 0x7FF0000000000000, 0x8000000000000000], dtype=np.uint64)
    graph.set_state({"scratch": bits.view(np.float64)})
    nonfinite = _snapshot(graph)
    graph.set_state({"scratch": np.array([1.0, 2.0])})
    replaced = _snapshot(graph)

    graph.restore(root)
    assert "scratch" not in graph.state_view()
    graph.commit(graph.checkpoint())
    for snapshot in [empty, nonfinite, replaced]:
        _unchanged(snapshot)
    assert empty[0]["scratch"].base is not None
    with pytest.raises(ValueError):
        empty[0]["scratch"].setflags(write=True)


def test_releasing_views_before_mutation_keeps_native_state_on_the_inplace_path():
    graph = _committed_graph()
    for step in range(20):
        view = graph.state_view(copy=False)
        np.testing.assert_array_equal(view["pin_rAAT"], graph.state_view()["pin_rAAT"])
        del view
        graph.resize_cell("u", "BUF2" if step % 2 == 0 else "BUF1")
        graph.propagate_incremental()

    statistics = graph.state_view_statistics()
    assert statistics["exports"] > 0
    assert statistics["current_view_fields"] == 0
    assert statistics["detached_fields"] == statistics["detached_values"] == 0


def test_invalid_input_does_not_detach_a_live_state_view():
    graph = _committed_graph()
    previous = _snapshot(graph)
    before = graph.state_view_statistics()

    with pytest.raises(ValueError, match="float64"):
        graph.set_state({"pin_rAAT": np.zeros(4, dtype=np.float32)})

    _unchanged(previous)
    assert graph.state_view_statistics() == before
    current = graph.state_view(copy=False)
    assert np.shares_memory(current["pin_rAAT"], previous[0]["pin_rAAT"])
