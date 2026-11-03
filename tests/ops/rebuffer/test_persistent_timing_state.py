"""Full Python TP state import, ownership, and transactional validation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base


def _assert_state(actual, expected):
    assert set(actual) - {"status", "epoch"} == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)


@pytest.fixture
def timing_state():
    base = _base()
    owner = RebufferTimingModel(base, use_pin_offset=False)
    return base, owner, persistent_timing_state_payload(owner)


def test_full_state_import_copies_python_tp_and_matches_each_source(timing_state):
    base, owner, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    schema = graph.timing_state_schema()

    result = graph.initialize_timing_state(payload)

    assert result["timing_state_initialized"]
    _assert_state(graph.state_view(), payload)
    for name, spec in schema["fields"].items():
        source = getattr(owner.model, spec["source"]).detach().numpy()
        if name.startswith("pin_net_"):
            source = source[: len(base.pin_names)]
        np.testing.assert_array_equal(payload[name], source)

    original = deepcopy(payload)
    payload["pin_rAAT"][:] = -123.0
    _assert_state(graph.state_view(), original)
    view = graph.state_view()
    view["pin_fRAT"][:] = -456.0
    _assert_state(graph.state_view(), original)


def test_full_state_import_uses_physical_prefix_of_steiner_rc(timing_state):
    base, owner, payload = timing_state
    owner.model.pin_net_cap_rise_live = torch.cat(
        (owner.model.pin_net_cap_rise_live, torch.tensor([999.0], dtype=torch.float64))
    )

    current = persistent_timing_state_payload(owner)

    assert len(current["pin_net_cap_rise"]) == len(base.pin_names)
    for name in payload:
        np.testing.assert_array_equal(current[name], payload[name])


def test_import_reinitialize_and_nested_restore_are_exact(timing_state):
    base, _, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    empty = graph.checkpoint()
    graph.initialize_timing_state(payload)
    first = graph.checkpoint()
    replacement = {name: values + 0.5 for name, values in payload.items()}
    graph.initialize_timing_state(replacement)
    second = graph.checkpoint()
    graph.set_state({"pin_rAAT": payload["pin_rAAT"] + 2.0})

    graph.restore(second)
    _assert_state(graph.state_view(), replacement)
    graph.restore(first)
    _assert_state(graph.state_view(), payload)
    assert graph.summary()["timing_state_initialized"]
    graph.restore(empty)
    assert not graph.summary()["timing_state_initialized"]
    assert graph.state_view() == {"status": "ok", "epoch": 0}


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unknown",
        "pin_length",
        "arc_length",
        "float32",
        "int64",
        "shape",
        "stride",
        "list",
    ],
)
def test_invalid_import_is_atomic_and_does_not_change_journal(timing_state, failure):
    base, _, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.initialize_timing_state(payload)
    before = graph.summary()
    invalid = {name: values + 7.0 for name, values in payload.items()}
    if failure == "missing":
        del invalid["pin_fRAT"]
    elif failure == "unknown":
        invalid["pin_typo"] = np.zeros(len(base.pin_names))
    elif failure == "pin_length":
        invalid["pin_fRAT"] = np.zeros(len(base.pin_names) + 1)
    elif failure == "arc_length":
        invalid["cell_arc_ff_delays"] = np.zeros(len(base.pin_names))
    elif failure == "float32":
        invalid["pin_fRAT"] = invalid["pin_fRAT"].astype(np.float32)
    elif failure == "int64":
        invalid["pin_fRAT"] = np.arange(len(base.pin_names), dtype=np.int64)
    elif failure == "shape":
        invalid["pin_fRAT"] = invalid["pin_fRAT"].reshape(-1, 1)
    elif failure == "stride":
        invalid["pin_fRAT"] = np.zeros(2 * len(base.pin_names))[::2]
    else:
        invalid["pin_fRAT"] = invalid["pin_fRAT"].tolist()

    with pytest.raises(ValueError):
        graph.initialize_timing_state(invalid)

    assert graph.summary() == before
    _assert_state(graph.state_view(), payload)


def test_partial_canonical_write_validates_whole_batch_before_mutation(timing_state):
    base, _, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.initialize_timing_state(payload)
    before = graph.summary()

    with pytest.raises(ValueError, match="cell_arc_count"):
        graph.set_state(
            {
                "pin_rAAT": payload["pin_rAAT"] + 1.0,
                "cell_arc_rr_delays": np.zeros(len(base.pin_names)),
            }
        )

    assert graph.summary() == before
    _assert_state(graph.state_view(), payload)


def test_sentinel_bits_and_signed_zero_survive_import_and_restore(timing_state):
    base, _, payload = timing_state
    bits = np.array(
        [0x7FF8000000000012, 0xFFF0000000000000, 0x7FF0000000000000, 0x8000000000000000],
        dtype=np.uint64,
    )
    payload["pin_rAAT"] = bits.view(np.float64)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.initialize_timing_state(payload)
    checkpoint = graph.checkpoint()
    graph.set_state({"pin_rAAT": np.zeros(len(base.pin_names))})

    graph.restore(checkpoint)

    np.testing.assert_array_equal(graph.state_view()["pin_rAAT"].view(np.uint64), bits)


def test_import_accepts_empty_cell_arc_domain(timing_state):
    base, _, payload = timing_state
    no_cells = SimpleNamespace(**vars(base))
    no_cells.flat_inst_arcs_by_level = []
    no_cells.flat_inst_arcs_by_level_start = [0, 0]
    graph = rebuffer_cpp.create_persistent_timing_graph(no_cells)
    for name in payload:
        if name.startswith("cell_arc_"):
            payload[name] = np.empty(0, dtype=np.float64)

    graph.initialize_timing_state(payload)

    _assert_state(graph.state_view(), payload)


def test_payload_rejects_unavailable_python_tp_state(timing_state):
    _, owner, _ = timing_state
    owner.model.pin_net_impulse_fall_live = None

    with pytest.raises(ValueError, match="pin_net_impulse_fall_live"):
        persistent_timing_state_payload(owner)


def test_payload_rejects_a_stale_carrier(timing_state):
    base, owner, _ = timing_state
    owner.pydb = SimpleNamespace(pin_names=base.pin_names[:-1])

    with pytest.raises(ValueError, match="current matching carrier"):
        persistent_timing_state_payload(owner)


def test_full_state_import_retains_clock_arc_slots():
    base = _base()
    base.clock_pins = [0]
    base.FF_ids = [2]
    base.clk_pin_rtran = [10.0]
    base.clk_pin_ftran = [11.0]
    base.flat_inst_arcs_by_level = [
        [0, 2, 0, 0, 1, 0, 0, 0],
        [1, 2, 0, 0, 1, 0, 0, 0],
    ]
    base.flat_inst_arcs_by_level_start = [0, 1, 2]
    owner = RebufferTimingModel(base, use_pin_offset=False)
    payload = persistent_timing_state_payload(owner)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)

    graph.initialize_timing_state(payload)

    assert graph.summary()["clock_end"] == 1
    for name in ("cell_arc_rr_delays", "cell_arc_ff_delays"):
        assert len(payload[name]) == 2
        np.testing.assert_array_equal(graph.state_view()[name], getattr(owner.model, name).numpy())


def test_checkpoint_rejects_another_owner_before_mutation(timing_state):
    base, _, payload = timing_state
    first = rebuffer_cpp.create_persistent_timing_graph(base)
    second = rebuffer_cpp.create_persistent_timing_graph(base)
    first.initialize_timing_state(payload)
    second.initialize_timing_state(payload)
    checkpoint = first.checkpoint()
    before = second.summary()

    with pytest.raises(ValueError, match="another graph"):
        second.restore(checkpoint)
    with pytest.raises(ValueError, match="another graph"):
        second.commit(checkpoint)

    assert second.summary() == before
    _assert_state(second.state_view(), payload)


def test_checkpoint_rejects_discarded_branch_with_same_journal_length(timing_state):
    base, _, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    root = graph.checkpoint()
    graph.initialize_timing_state(payload)
    discarded = graph.checkpoint()
    graph.restore(root)
    replacement = {name: values + 1.0 for name, values in payload.items()}
    graph.initialize_timing_state(replacement)
    before = graph.summary()

    with pytest.raises(ValueError, match="stale journal history"):
        graph.restore(discarded)

    assert graph.summary() == before
    _assert_state(graph.state_view(), replacement)


def test_commit_invalidates_old_offsets_and_keeps_imported_state(timing_state):
    base, _, payload = timing_state
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    old_empty = graph.checkpoint()
    graph.initialize_timing_state(payload)
    current = graph.checkpoint()

    graph.commit(current)

    assert graph.summary()["journal_size"] == 0
    assert graph.summary()["timing_state_initialized"]
    _assert_state(graph.state_view(), payload)
    with pytest.raises(ValueError, match="stale journal history"):
        graph.restore(old_empty)
    fresh = graph.checkpoint()
    graph.set_state({"pin_fRAT": payload["pin_fRAT"] + 1.0})
    graph.restore(fresh)
    _assert_state(graph.state_view(), payload)
