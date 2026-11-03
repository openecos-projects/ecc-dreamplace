"""Native resize must update Liberty, RC and timing together and restore exactly."""

from copy import deepcopy

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.propagation_payload import persistent_propagation_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base, _reference_after


def _chain():
    base = _reference_after(_base())
    base.node_master_names = ["BUF1", "BUF2", "", ""]
    base.node_size_x, base.node_size_y = [1000.0, 2000.0, 0.0, 0.0], [1000.0] * 2 + [0.0] * 2
    base.pin_direct = ["R0"] * 6
    return base


def _graph(base, *, attach=True):
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    graph.initialize_propagation(persistent_propagation_payload(owner))
    graph.initialize_mutations()
    if attach:
        tree = steiner_payload(base, owner)
        for net, driver in enumerate(base.net2driver_pin_map):
            state = rebuffer_cpp.incremental_topology_from_steiner(
                tree, net, driver, owner.r_unit, owner.c_unit
            )
            graph.attach_net_topology_state(net, state)
    return owner, graph


def _reference(owner, masters):
    base = owner.pydb
    physical = deepcopy(base)
    for name, master in masters.items():
        physical.node_master_names[physical.node_names.index(name)] = master
    carrier = CommittedTimingPyDB(base, physical)
    tree = {
        name: getattr(owner.steiner, name).numpy()
        for name in (
            "net_flat_topo_sort",
            "net_flat_topo_sort_start",
            "pin_fa",
            "flat_pin_to_start",
            "flat_pin_to",
            "flat_pin_from",
        )
    }
    tree.update(node_x=owner.new_x.numpy(), node_y=owner.new_y.numpy())
    return RebufferTimingModel(carrier, use_pin_offset=False, steiner_view=tree)


def _same_state(actual, expected, *, bitwise=False):
    for name, value in expected.items():
        if name in ("status", "epoch"):
            continue
        if isinstance(value, np.ndarray):
            if bitwise:
                np.testing.assert_array_equal(actual[name].view(np.uint64), value.view(np.uint64))
            else:
                np.testing.assert_allclose(actual[name], value, rtol=0, atol=1e-6, err_msg=name)
        else:
            assert actual[name] == value


def _compare(graph, reference):
    result = graph.propagate()
    _same_state(graph.state_view(), persistent_timing_state_payload(reference))
    assert result["metrics"] == pytest.approx(reference.baseline.as_dict(), abs=1e-6, rel=0)
    view = graph.mutation_view()
    assert view["index_matches"]
    np.testing.assert_array_equal(view["cell_arcs"], reference.model.flat_inst_arcs_by_level)
    np.testing.assert_array_equal(view["pin_caps"], reference.lib_pin_caps)
    np.testing.assert_array_equal(view["pin_rcaps"], reference.lib_pin_rcaps)
    np.testing.assert_array_equal(view["pin_fcaps"], reference.lib_pin_fcaps)
    np.testing.assert_array_equal(view["pin_max_slew"], reference.pin_max_slew)
    np.testing.assert_array_equal(view["pin_max_cap"], reference.pin_max_cap)
    return result


def test_resize_updates_upstream_cap_and_all_timing_fields_against_fresh_tp():
    owner, graph = _graph(_base())
    before = persistent_timing_state_payload(owner)

    resized = graph.resize_cell("u", "BUF2")
    reference = _reference(owner, {"u": "BUF2"})

    assert resized["affected_nets"] == [0]
    assert resized["updated_arc_count"] == 1
    assert graph.state_view()["pin_net_cap_rise"][0] > before["pin_net_cap_rise"][0]
    _compare(graph, reference)
    _same_state(persistent_timing_state_payload(owner), before, bitwise=True)


def test_two_resizes_nested_restore_and_commit_keep_metadata_and_state_exact():
    owner, graph = _graph(_chain())
    root = graph.checkpoint()
    original_state, original_meta = graph.state_view(), graph.mutation_view()
    graph.resize_cell("u", "BUF2")
    _compare(graph, _reference(owner, {"u": "BUF2"}))
    middle = graph.checkpoint()
    middle_state, middle_meta = graph.state_view(), graph.mutation_view()
    graph.resize_cell("b", "BUF1")
    _compare(graph, _reference(owner, {"u": "BUF2", "b": "BUF1"}))

    graph.restore(middle)

    _same_state(graph.state_view(), middle_state, bitwise=True)
    _same_state(graph.mutation_view(), middle_meta, bitwise=True)
    graph.restore(root)
    _same_state(graph.state_view(), original_state, bitwise=True)
    _same_state(graph.mutation_view(), original_meta, bitwise=True)
    graph.resize_cell("u", "BUF2")
    _compare(graph, _reference(owner, {"u": "BUF2"}))
    graph.commit(graph.checkpoint())
    assert graph.summary()["journal_size"] == 0
    checkpoint = graph.checkpoint()
    graph.resize_cell("u", "BUF1")
    _compare(graph, owner)
    graph.restore(checkpoint)
    _same_state(graph.state_view(), middle_state, bitwise=True)
    _same_state(graph.mutation_view(), middle_meta, bitwise=True)


def test_resize_has_a_private_index_and_does_not_change_another_owner():
    base = _base()
    owner, first = _graph(base)
    other = rebuffer_cpp.create_persistent_timing_graph(base)
    other.initialize_timing_state(persistent_timing_state_payload(owner))
    other.initialize_propagation(persistent_propagation_payload(owner))

    first.resize_cell("u", "BUF2")
    _compare(first, _reference(owner, {"u": "BUF2"}))
    result = other.propagate()

    _same_state(other.state_view(), persistent_timing_state_payload(owner))
    assert result["metrics"] == pytest.approx(owner.baseline.as_dict(), abs=1e-6, rel=0)


@pytest.mark.parametrize(
    "instance,master", [("unknown", "BUF2"), ("u", "unknown"), ("OUT", "BUF2")]
)
def test_invalid_resize_keeps_journal_metadata_and_state(instance, master):
    _, graph = _graph(_base())
    before, state, metadata = graph.summary(), graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError):
        graph.resize_cell(instance, master)

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_missing_input_net_rc_restores_resize_metadata_and_state():
    _, graph = _graph(_base(), attach=False)
    before, state, metadata = graph.summary(), graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError, match="not attached"):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_same_master_resize_is_a_noop():
    _, graph = _graph(_base())
    before = graph.summary()

    result = graph.resize_cell("u", "BUF1")

    assert not result["changed"]
    assert graph.summary() == before


@pytest.mark.parametrize("mode", ["reordered", "repeated"])
def test_resize_uses_canonical_port_signature_and_occurrence_mapping(mode):
    base = _base()
    if mode == "reordered":
        base.cell_id_2_arc_id_start = [0, 1, 3]
        base.flat_libarc_info = [
            ["A", "Y", 0, 0, 1, 0],
            ["A", "Y", 1, 0, -1, 0],
            ["A", "Y", 1, 1, 1, 0],
        ]
        expected = 2
    else:
        base.cell_id_2_arc_id_start = [0, 2, 5]
        base.flat_libarc_info = [
            ["A", "Y", 0, 0, 1, 0],
            ["A", "Y", 0, 1, 1, 0],
            ["A", "Y", 1, 0, -1, 0],
            ["A", "Y", 1, 1, 1, 0],
            ["A", "Y", 1, 2, 1, 0],
        ]
        base.flat_inst_arcs_by_level[0][3] = base.flat_inst_arcs_by_level[0][6] = 1
        expected = 4
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        for suffix in ("values", "trans_table", "cap_table", "dim"):
            values = getattr(base, f"{prefix}_flat_luts_{suffix}")
            values.extend([deepcopy(values[-1]) for _ in range(len(base.flat_libarc_info) - 2)])
    owner, graph = _graph(base)

    graph.resize_cell("u", "BUF2")

    assert graph.mutation_view()["cell_arcs"][0, 3] == expected
    _compare(graph, _reference(owner, {"u": "BUF2"}))


@pytest.mark.parametrize("failure", ["family", "signature", "pin_schema", "lut"])
def test_incompatible_master_rejects_atomically(failure):
    base = _base()
    if failure == "family":
        base.flat_libcell_info[1][1] = 1
    elif failure == "signature":
        base.flat_libarc_info[1][4] = -1
    elif failure == "pin_schema":
        base.flat_lib_pin_names[2] = "B"
    else:
        base.cell_id_2_arc_id_start[-1] = 3
        base.flat_libarc_info.append(["A", "Y", 1, 1, 1, 0])
        base.flat_libarc_info[1][4] = -1
    _, graph = _graph(base)
    before, state, metadata = graph.summary(), graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


@pytest.mark.parametrize("kind", ["wrong_net", "missing_pin", "wrong_driver"])
def test_rc_tree_must_cover_the_current_net_before_any_write(kind):
    _, graph = _graph(_base(), attach=False)
    driver, sink = (1, 0) if kind == "wrong_driver" else (0, 3 if kind == "wrong_net" else 4)
    graph.attach_net_topology(
        0,
        driver,
        5,
        {
            "edges": [{"parent": driver, "child": sink, "r": 1.0, "c": 0.1}],
            "node_loads": {sink: [sink]},
        },
    )
    before, state, metadata = graph.summary(), graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError, match="does not match|do not match"):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_clock_cell_resize_is_rejected_before_changing_timing_constraints():
    from tests.ops.rebuffer.test_persistent_timing_propagation import _checks_base

    _, graph = _graph(_checks_base())
    before, state, metadata = graph.summary(), graph.state_view(), graph.mutation_view()

    with pytest.raises(ValueError, match="sequential"):
        graph.resize_cell("u", "BUF2")

    assert graph.summary() == before
    _same_state(graph.state_view(), state, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_one_hundred_resize_and_restore_operations_match_full_tp():
    owner, graph = _graph(_chain())
    original_state, original_metadata = graph.state_view(), graph.mutation_view()
    root = graph.checkpoint()
    masters = {}
    for attempt in range(100):
        instance = "u" if attempt % 3 else "b"
        original = "BUF1" if instance == "u" else "BUF2"
        opposite = "BUF2" if instance == "u" else "BUF1"
        target = opposite if masters.get(instance, original) == original else original
        graph.resize_cell(instance, target)
        masters[instance] = target
        _compare(graph, _reference(owner, masters))
        if attempt % 7 == 0:
            graph.restore(root)
            _same_state(graph.state_view(), original_state, bitwise=True)
            _same_state(graph.mutation_view(), original_metadata, bitwise=True)
            masters.clear()
