"""Native buffer topology must drive graph-owned RC/timing and exact rollback."""

from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _split(graph, net, child, action_id, master="BUF1", *, load=None):
    view = graph.topology_view(net)
    edge = next(edge for edge in view["edges"] if edge["child"] == child)
    action = {
        "action_id": action_id,
        "session_buffer_id": action_id,
        "parent_node_id": edge["parent"],
        "child_node_id": child,
        "downstream_pin_ids": [child if load is None else load],
        "split_ratio": 0.5,
        "buffer_master_name": master,
    }
    result = graph.apply_buffer_split(net, action)
    return action, result


def _reference(owner, graph, actions, masters=None):
    exported = graph.export_carrier(
        steiner_payload(owner.pydb, owner), actions, owner.r_unit, owner.c_unit
    )
    physical = SimpleNamespace(**exported["physical"])
    for name, master in (masters or {}).items():
        physical.node_master_names[physical.node_names.index(name)] = master
    carrier = CommittedTimingPyDB(owner.pydb, physical)
    return RebufferTimingModel(carrier, use_pin_offset=False, steiner_view=exported["steiner"])


def _arc_key(row, names, node_names, clocks, is_clock):
    source = int(clocks[int(row[0])]) if is_clock else int(row[0])
    return (
        names[source],
        names[int(row[1])],
        int(row[3]),
        int(row[4]),
        int(row[5]),
        node_names[int(row[7])],
        bool(is_clock),
    )


def _compare(graph, owner, actions, masters=None):
    result = graph.propagate()
    reference = _reference(owner, graph, actions, masters)
    expected = persistent_timing_state_payload(reference)
    metadata = graph.mutation_view()
    assert metadata["index_matches"]
    active_pins = np.flatnonzero(metadata["pin_active"])
    names = metadata["pin_names"]
    lookup = {name: index for index, name in enumerate(reference.pydb.pin_names)}
    assert {names[pin] for pin in active_pins} == set(lookup)
    pin_map = [lookup[names[pin]] for pin in active_pins]
    active_arcs = np.flatnonzero(metadata["cell_active"])
    clock_end = graph.summary()["clock_end"]
    reference_clock_end = int(reference.model.flat_inst_arcs_by_level_start[1])
    arcs = reference.model.flat_inst_arcs_by_level.numpy()
    arc_lookup = {
        _arc_key(
            row,
            reference.pydb.pin_names,
            reference.pydb.node_names,
            reference.pydb.clock_pins,
            index < reference_clock_end,
        ): index
        for index, row in enumerate(arcs)
    }
    assert len(arc_lookup) == len(arcs)
    arc_map = [
        arc_lookup[
            _arc_key(
                metadata["cell_arcs"][slot],
                names,
                metadata["node_names"],
                owner.pydb.clock_pins,
                slot < clock_end,
            )
        ]
        for slot in active_arcs
    ]
    actual = graph.state_view()
    for name, expected_values in expected.items():
        selected, mapping = (
            (active_arcs, arc_map) if name.startswith("cell_arc_") else (active_pins, pin_map)
        )
        np.testing.assert_allclose(
            actual[name][selected], expected_values[mapping], atol=1e-6, rtol=0, err_msg=name
        )
    assert result["metrics"] == pytest.approx(reference.baseline.as_dict(), abs=1e-6, rel=0)
    np.testing.assert_allclose(
        result["endpoint_slacks"], reference.model.last_endpoint_slack_tensor, atol=1e-6, rtol=0
    )
    assert graph.summary()["active_pin_count"] == len(reference.pydb.pin_names)
    assert graph.summary()["active_cell_arc_count"] == len(arcs)
    for field, source in (
        ("pin_caps", "lib_pin_caps"),
        ("pin_rcaps", "lib_pin_rcaps"),
        ("pin_fcaps", "lib_pin_fcaps"),
        ("pin_max_slew", "pin_max_slew"),
        ("pin_max_cap", "pin_max_cap"),
    ):
        np.testing.assert_array_equal(
            metadata[field][active_pins], getattr(reference, source)[pin_map]
        )
    return result


@pytest.mark.parametrize("master", ["BUF1", "BUF2"])
def test_insert_resizes_canonical_state_and_recomputes_from_original_constraints(master):
    owner, graph = _graph(_base())
    checkpoint = graph.checkpoint()
    before, metadata = graph.state_view(), graph.mutation_view()

    action, insertion = _split(graph, 1, 3, 101, master)

    assert insertion["timing_pin_ids"] == [4, 5]
    assert insertion["timing_arc_ids"] == [1]
    assert graph.timing_state_schema()["pin_count"] == 6
    assert graph.timing_state_schema()["cell_arc_count"] == 2
    _compare(graph, owner, [action])
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_same_net_chain_remove_keeps_stable_slots_and_nested_restore_is_exact():
    owner, graph = _graph(_base())
    root = graph.checkpoint()
    baseline, baseline_meta = graph.state_view(), graph.mutation_view()
    actions, slots = [], []
    middle = middle_state = middle_meta = None
    for index in range(4):
        action, result = _split(graph, 1, 3, index + 10, "BUF1" if index % 2 else "BUF2")
        actions.append(action)
        slots.append(result["timing_pin_ids"])
        _compare(graph, owner, actions)
        if index == 1:
            middle, middle_state, middle_meta = (
                graph.checkpoint(),
                graph.state_view(),
                graph.mutation_view(),
            )
    allocated = graph.summary()["pin_count"]

    graph.remove_buffer(1, 13)
    _compare(graph, owner, actions[:-1])

    assert graph.summary()["pin_count"] == allocated
    assert graph.summary()["active_pin_count"] == allocated - 2
    assert not np.any(np.asarray(graph.mutation_view()["pin_active"])[slots[-1]])
    for field in ("pin_net_cap_rise", "pin_net_delay_rise", "pin_net_impulse_rise"):
        np.testing.assert_array_equal(graph.state_view()[field][slots[-1]], [0.0, 0.0])
    graph.restore(middle)
    _same_state(graph.state_view(), middle_state, bitwise=True)
    _same_state(graph.mutation_view(), middle_meta, bitwise=True)
    _compare(graph, owner, actions[:2])
    graph.restore(root)
    _same_state(graph.state_view(), baseline, bitwise=True)
    _same_state(graph.mutation_view(), baseline_meta, bitwise=True)


def test_cross_net_local_node_collisions_and_out_of_order_actions_match_fresh_tp():
    owner, graph = _graph(_base())
    first, _ = _split(graph, 0, 1, 8)
    second, second_result = _split(graph, 1, 3, 3, "BUF2")
    third, third_result = _split(graph, 0, 1, 1, "BUF2")

    assert second_result["input_node"] == third_result["input_node"]
    assert second_result["timing_pin_ids"] != third_result["timing_pin_ids"]
    _compare(graph, owner, [first, second, third])
    graph.remove_buffer(0, 1)
    _compare(graph, owner, [first, second])


def test_buffer_resize_updates_current_cap_and_lut_and_restores_exactly():
    owner, graph = _graph(_base())
    action, _ = _split(graph, 1, 3, 20)
    _compare(graph, owner, [action])
    checkpoint = graph.checkpoint()
    before, metadata = graph.state_view(), graph.mutation_view()

    graph.resize_buffer(1, {"action_id": 20, "buffer_master_name": "BUF2"})
    changed = dict(action, buffer_master_name="BUF2")
    _compare(graph, owner, [changed])

    assert graph.state_view()["pin_net_cap_rise"][2] > before["pin_net_cap_rise"][2]
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_existing_cell_resize_then_insert_and_remove_preserves_current_master():
    owner, graph = _graph(_base())
    graph.resize_cell("u", "BUF2")
    action, _ = _split(graph, 1, 3, 12)
    _compare(graph, owner, [action], {"u": "BUF2"})
    graph.remove_buffer(1, 12)
    _compare(graph, owner, [], {"u": "BUF2"})


def test_invalid_master_and_duplicate_identity_fail_without_any_partial_state():
    owner, graph = _graph(_base())
    action, _ = _split(graph, 1, 3, 10)
    _compare(graph, owner, [action])
    before, metadata, topology = graph.state_view(), graph.mutation_view(), graph.topology_view(1)
    summary = graph.summary()

    with pytest.raises(ValueError, match="absent from Liberty"):
        _split(graph, 1, 3, 11, "UNKNOWN")
    with pytest.raises(ValueError, match="another source net"):
        _split(graph, 0, 1, 10)

    assert graph.summary() == summary
    assert graph.topology_view(1) == topology
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_late_rc_failure_restores_appended_records_index_and_topology():
    _, graph = _graph(_base(), attach=False)
    graph.attach_net_topology(
        1,
        2,
        5,
        {
            "edges": [{"parent": 2, "child": 4, "r": 10.0, "c": 0.01}],
            "node_loads": {4: [3]},
        },
    )
    before, metadata, topology = graph.state_view(), graph.mutation_view(), graph.topology_view(1)
    summary = graph.summary()

    with pytest.raises(ValueError, match="explicit physical pin vertices"):
        _split(graph, 1, 4, 10, load=3)

    assert graph.summary() == summary
    assert graph.topology_view(1) == topology
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_removed_buffer_can_be_reapplied_to_the_same_stable_slots_then_committed():
    owner, graph = _graph(_base())
    action, first = _split(graph, 1, 3, 6)
    _compare(graph, owner, [action])
    graph.remove_buffer(1, 6)
    _compare(graph, owner, [])

    action, second = _split(graph, 1, 3, 6)

    assert second["timing_pin_ids"] == first["timing_pin_ids"]
    assert second["timing_arc_ids"] == first["timing_arc_ids"]
    _compare(graph, owner, [action])
    graph.commit(graph.checkpoint())
    assert graph.summary()["journal_size"] == 0


def test_existing_cell_resize_after_input_net_split_uses_source_net_rc_and_current_schedule():
    owner, graph = _graph(_base())
    action, _ = _split(graph, 0, 1, 10)
    _compare(graph, owner, [action])
    checkpoint = graph.checkpoint()
    before, metadata = graph.state_view(), graph.mutation_view()

    graph.resize_cell("u", "BUF2")

    _compare(graph, owner, [action], {"u": "BUF2"})
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)
    _compare(graph, owner, [action])


def test_resize_cell_routes_inserted_buffer_to_its_topology_owner():
    owner, graph = _graph(_base())
    action, _ = _split(graph, 1, 3, 7)

    graph.resize_cell("__dreamplace_rebuffer_7", "BUF2")

    assert graph.topology_view(1)["buffers"][0]["buffer_master_name"] == "BUF2"
    _compare(graph, owner, [dict(action, buffer_master_name="BUF2")])


def test_buffer_arc_count_change_retains_dead_slots_and_matches_full_tp():
    base = _base()
    base.cell_id_2_arc_id_start = [0, 1, 3]
    base.flat_libarc_info.append(["A", "Y", 1, 1, 1, 0])
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        for suffix in ("values", "trans_table", "cap_table", "dim"):
            values = getattr(base, f"{prefix}_flat_luts_{suffix}")
            values.append(list(values[-1]))
    owner, graph = _graph(base)
    action, _ = _split(graph, 1, 3, 10)
    _compare(graph, owner, [action])

    graph.resize_buffer(1, {"action_id": 10, "buffer_master_name": "BUF2"})
    _compare(graph, owner, [dict(action, buffer_master_name="BUF2")])

    assert graph.summary()["cell_arc_count"] == 4
    assert graph.summary()["active_cell_arc_count"] == 3
    graph.resize_buffer(1, {"action_id": 10, "buffer_master_name": "BUF1"})
    _compare(graph, owner, [action])
    assert graph.summary()["cell_arc_count"] == 5
    assert graph.summary()["active_cell_arc_count"] == 2


def test_clock_net_insertion_is_rejected_before_topology_or_timing_changes():
    base = _base()
    base.clock_net_names = ["n1"]
    _, graph = _graph(base)
    before, metadata, topology = graph.state_view(), graph.mutation_view(), graph.topology_view(1)

    with pytest.raises(ValueError, match="signal source net"):
        _split(graph, 1, 3, 10)

    assert graph.topology_view(1) == topology
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_external_topology_mutation_cannot_silently_reuse_stale_timing_records():
    owner, graph = _graph(_base(), attach=False)
    state = rebuffer_cpp.incremental_topology_from_steiner(
        steiner_payload(owner.pydb, owner), 1, 2, owner.r_unit, owner.c_unit
    )
    graph.attach_net_topology_state(1, state)
    action, _ = _split(graph, 1, 3, 10)
    _compare(graph, owner, [action])
    rebuffer_cpp.remove_incremental_buffer(state, 10)

    with pytest.raises(ValueError, match="not synchronized|outside graph"):
        graph.propagate()
    before, metadata, topology = graph.state_view(), graph.mutation_view(), graph.topology_view(1)
    with pytest.raises(ValueError, match="not synchronized|outside graph"):
        _split(graph, 1, 3, 11)
    assert graph.topology_view(1) == topology
    _same_state(graph.state_view(), before, bitwise=True)
    _same_state(graph.mutation_view(), metadata, bitwise=True)


def test_one_hundred_mixed_buffer_mutations_and_restores_match_full_tp():
    owner, graph = _graph(_base())
    actions = []
    for attempt in range(100):
        checkpoint = graph.checkpoint()
        saved, saved_meta = graph.state_view(), graph.mutation_view()
        previous = [dict(action) for action in actions]
        net, child = (0, 1) if attempt % 2 else (1, 3)
        current = [index for index, action in enumerate(actions) if action["source_net"] == net]
        if current and attempt % 5 == 0:
            index = current[-1]
            master = "BUF2" if actions[index]["buffer_master_name"] == "BUF1" else "BUF1"
            graph.resize_buffer(
                net, {"action_id": actions[index]["action_id"], "buffer_master_name": master}
            )
            actions[index] = dict(actions[index], buffer_master_name=master)
        elif current and (len(current) >= 3 or attempt % 3 == 0):
            index = current[-1]
            graph.remove_buffer(net, actions[index]["action_id"])
            actions.pop(index)
        else:
            action, _ = _split(graph, net, child, attempt + 100, "BUF2" if attempt % 3 else "BUF1")
            actions.append(dict(action, source_net=net))
        _compare(graph, owner, actions)
        if attempt % 7 == 0:
            graph.restore(checkpoint)
            _same_state(graph.state_view(), saved, bitwise=True)
            _same_state(graph.mutation_view(), saved_meta, bitwise=True)
            actions = previous
    graph.commit(graph.checkpoint())
    _compare(graph, owner, actions)
