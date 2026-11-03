"""Sparse state undo retains exact bits, growth boundaries and event baselines."""

import numpy as np
import pytest
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_events import _assert_event
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _committed_graph():
    owner, graph = _graph(_base())
    graph.propagate()
    graph.commit(graph.checkpoint())
    return owner, graph


def _changed_values(before, after, *, common_prefix=False):
    count = 0
    for field, values in before.items():
        if not isinstance(values, np.ndarray):
            continue
        current = after[field][: len(values)] if common_prefix else after[field]
        count += np.count_nonzero(values.view(np.uint64) != current.view(np.uint64))
    return count


def test_noop_event_journals_no_state_values_and_restores_its_baseline_mark():
    owner, graph = _committed_graph()
    checkpoint = graph.checkpoint()
    before = graph.state_view()
    for _ in range(3):
        result = _assert_event(graph, owner)
        assert result["changed_pins"] == result["changed_endpoints"] == []

    summary = graph.summary()
    assert summary["journal_full_state_values"] == 0
    assert summary["journal_sparse_state_values"] == 0
    assert summary["journal_state_value_bytes"] == 0
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    assert graph.summary()["journal_size"] == 0
    assert graph.propagate_incremental()["mode"] == "events"


@pytest.mark.parametrize("bits", [0x7FF8000000000123, 0x7FF0000000000000, 0x8000000000000000])
def test_one_slot_patch_and_nested_restore_preserve_nonfinite_and_signed_zero_bits(bits):
    _, graph = _committed_graph()
    root = graph.checkpoint()
    original = graph.state_view()
    values = original["pin_net_delay_rise"].copy()
    values.view(np.uint64)[0] = bits
    graph.set_state({"pin_net_delay_rise": values})
    first = graph.checkpoint()

    assert graph.summary()["journal_full_state_values"] == 0
    assert graph.summary()["journal_sparse_state_values"] == 1
    assert graph.state_view()["pin_net_delay_rise"].view(np.uint64)[0] == bits
    values[0] = 123.0
    graph.set_state({"pin_net_delay_rise": values})
    assert graph.summary()["journal_sparse_state_values"] == 2

    graph.restore(first)
    assert graph.state_view()["pin_net_delay_rise"].view(np.uint64)[0] == bits
    graph.restore(root)
    _same_state(graph.state_view(), original, bitwise=True)
    assert graph.summary()["journal_state_value_bytes"] == 0


def test_resize_and_event_store_exactly_the_changed_values_without_full_field_undo():
    _, graph = _committed_graph()
    checkpoint = graph.checkpoint()
    original = graph.state_view()
    graph.resize_cell("u", "BUF2")
    after_resize = graph.state_view()
    summary = graph.summary()

    assert summary["journal_full_state_values"] == 0
    assert summary["journal_sparse_state_values"] == _changed_values(original, after_resize)
    graph.propagate_incremental()
    after_event = graph.state_view()
    final = graph.summary()
    assert final["journal_full_state_values"] == 0
    assert final["journal_sparse_state_values"] - summary["journal_sparse_state_values"] == (
        _changed_values(after_resize, after_event)
    )
    _assert_event(graph, None)
    graph.restore(checkpoint)
    _same_state(graph.state_view(), original, bitwise=True)
    assert graph.summary()["journal_state_value_bytes"] == 0


def test_buffer_growth_saves_existing_changes_and_restores_appended_slots_by_length():
    _, graph = _committed_graph()
    root = graph.checkpoint()
    original = graph.state_view()
    _split(graph, 1, 3, 7)
    grown = graph.state_view()
    summary = graph.summary()

    assert len(grown["pin_rAAT"]) == len(original["pin_rAAT"]) + 2
    assert summary["journal_full_state_values"] == 0
    assert summary["journal_sparse_state_values"] == _changed_values(
        original, grown, common_prefix=True
    )
    graph.propagate_incremental()
    _assert_event(graph, None)
    graph.restore(root)
    _same_state(graph.state_view(), original, bitwise=True)
    assert graph.summary()["journal_state_value_bytes"] == 0


def test_sparse_rc_and_generic_scratch_share_atomic_validation_and_nested_restore():
    _, graph = _committed_graph()
    root = graph.checkpoint()
    original, summary = graph.state_view(), graph.summary()
    changed = original["pin_net_delay_rise"] + 1.0
    with pytest.raises(ValueError, match="float64"):
        graph.set_state({"pin_net_delay_rise": changed, "pin_rRAT": np.zeros(4, np.float32)})
    assert graph.summary() == summary
    _same_state(graph.state_view(), original, bitwise=True)

    graph.set_state({"pin_net_delay_rise": changed, "net_0_scratch": np.array([5.0, 6.0])})
    assert graph.propagate_incremental()["mode"] == "full_seed"
    graph.restore(root)
    _same_state(graph.state_view(), original, bitwise=True)
    assert "net_0_scratch" not in graph.state_view()
    assert graph.summary()["journal_state_value_bytes"] == 0


def test_late_resize_rc_failure_restores_local_metadata_and_event_baseline():
    _, graph = _graph(_base(), attach=False)
    graph.propagate()
    graph.commit(graph.checkpoint())
    state, metadata, summary = graph.state_view(), graph.mutation_view(), graph.summary()

    with pytest.raises(ValueError, match="not attached"):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == summary
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)
    assert graph.propagate_incremental()["mode"] == "events"
