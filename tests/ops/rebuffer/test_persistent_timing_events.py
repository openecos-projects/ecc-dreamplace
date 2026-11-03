"""Native event worklists agree with full sweeps and the Python TP oracle."""

import numpy as np
import pytest
import torch
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.propagation_payload import persistent_propagation_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base, _reference_after
from tests.ops.rebuffer.test_persistent_timing_buffers import (
    _compare as _buffer_oracle,
)
from tests.ops.rebuffer.test_persistent_timing_buffers import (
    _reference,
    _split,
)
from tests.ops.rebuffer.test_persistent_timing_propagation import _checks_base
from tests.ops.rebuffer.test_persistent_timing_resize import _graph, _same_state


def _assert_event(graph, reference):
    before = graph.state_view()
    result = graph.propagate_incremental()
    assert result["mode"] == "events"
    observed = graph.state_view()
    checkpoint = graph.checkpoint()
    full = graph.propagate()
    _same_state(observed, graph.state_view())
    np.testing.assert_allclose(
        result["endpoint_slacks"], full["endpoint_slacks"], rtol=0, atol=1e-8
    )
    for key, value in full["metrics"].items():
        np.testing.assert_allclose(result["metrics"][key], value, rtol=0, atol=1e-8)
    graph.restore(checkpoint)
    if reference is not None:
        expected = persistent_timing_state_payload(reference)
        # Carrier slots are compact; callers with appended/dead slots use
        # the independent name-aware buffer oracle below instead.
        _same_state(observed, expected)
        np.testing.assert_allclose(
            result["endpoint_slacks"],
            reference.model.last_endpoint_slack_tensor,
            rtol=0,
            atol=1e-6,
        )
    changed = np.zeros(len(observed["pin_rAAT"]), dtype=bool)
    for name in ("pin_rAAT", "pin_fAAT", "pin_rtran", "pin_ftran", "pin_rRAT", "pin_fRAT"):
        changed |= observed[name].view(np.uint64) != before[name].view(np.uint64)
    assert result["changed_pins"] == np.flatnonzero(changed).tolist()
    return result


def _owner_graph(owner):
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    graph.initialize_propagation(persistent_propagation_payload(owner))
    return graph


def _rc_changed(graph, owner, pin):
    owner.loads["rise"][pin] += 0.01
    owner.delays["rise"][pin] += 7
    owner.impulses["rise"][pin] += 49
    owner._baseline = owner._forward()
    values = persistent_timing_state_payload(owner)
    graph.set_state({key: value for key, value in values.items() if key.startswith("pin_net_")})


def test_first_event_call_establishes_native_baseline_then_noop_visits_no_rows():
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = _owner_graph(owner)
    graph.set_state({"pin_rAAT": np.full(4, 987654.0)})

    first = graph.propagate_incremental()

    assert first["mode"] == "full_seed"
    _same_state(graph.state_view(), persistent_timing_state_payload(owner))
    repeated = _assert_event(graph, owner)
    assert repeated["changed_pins"] == repeated["changed_endpoints"] == []
    assert {
        key: repeated["coverage"][key]
        for key in (
            "forward_pins",
            "reverse_pins",
            "cell_forward_rows",
            "cell_reverse_rows",
            "net_forward_rows",
            "net_reverse_rows",
            "clk2q_rows",
            "setup_rows",
            "recovery_rows",
        )
    } == dict.fromkeys(
        (
            "forward_pins",
            "reverse_pins",
            "cell_forward_rows",
            "cell_reverse_rows",
            "net_forward_rows",
            "net_reverse_rows",
            "clk2q_rows",
            "setup_rows",
            "recovery_rows",
        ),
        0,
    )


@pytest.mark.parametrize("sense", [-1, 0, 1])
def test_changed_rc_recomputes_all_fanin_and_full_rat_then_stops(sense):
    base = _reference_after(_base())
    for row in base.flat_inst_arcs_by_level:
        row[4] = sense
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph = _owner_graph(owner)
    graph.propagate()
    checkpoint = graph.checkpoint()
    before = graph.state_view()

    _rc_changed(graph, owner, 3)
    result = _assert_event(graph, owner)

    assert result["coverage"]["forward_pins"] > 0
    assert result["coverage"]["reverse_pins"] > 0
    assert result["changed_endpoints"] == [5]
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    assert graph.propagate_incremental()["mode"] == "events"


@pytest.mark.parametrize("live_reference", [False, True])
@pytest.mark.parametrize("scope", ["setup_only", "setup_plus_recovery"])
def test_event_clock_setup_recovery_and_qualification_use_shared_math(live_reference, scope):
    owner = RebufferTimingModel(_checks_base(), use_pin_offset=False)
    owner.model.recovery_check_uses_live_slew = live_reference
    owner.model.timing_metric_scope = scope
    owner.model.backend_endpoint_rSlew = torch.tensor([5.0], dtype=torch.float64)
    owner.model.backend_endpoint_fSlew = torch.tensor([6.0], dtype=torch.float64)
    owner._baseline = owner._forward()
    graph = _owner_graph(owner)
    graph.propagate()

    _rc_changed(graph, owner, 2)
    result = _assert_event(graph, owner)

    assert result["coverage"]["clk2q_rows"] == 1
    assert result["coverage"]["setup_rows"] == 2
    assert result["coverage"]["recovery_rows"] == 2


def test_metadata_deadline_recovery_events_match_full_state_and_python_without_native_slews():
    base = _checks_base()
    base.recovery_rat_is_capture_deadline = True
    owner = RebufferTimingModel(base, use_pin_offset=False)
    owner.model.timing_metric_scope = "setup_plus_recovery"
    owner._baseline = owner._forward()
    graph = _owner_graph(owner)
    graph.propagate()
    checkpoint = graph.checkpoint()
    before = graph.state_view()
    _rc_changed(graph, owner, 2)
    result = _assert_event(graph, owner)
    assert result["coverage"]["recovery_rows"] == 2
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)


def test_reconvergence_recomputes_every_incoming_edge_after_old_max_decreases():
    base = _base()
    base.flat_inst_arcs_by_level = [
        [0, 2, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 1, 0, 0, 0],
        [1, 2, 1, 1, -1, 0, 0, 0],
    ]
    base.flat_inst_arcs_by_level_start = [0, 0, 3]
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph = _owner_graph(owner)
    graph.propagate()
    _rc_changed(graph, owner, 1)
    _assert_event(graph, owner)
    owner.delays["rise"][1] -= 7
    owner._baseline = owner._forward()
    graph.set_state(
        {"pin_net_delay_rise": persistent_timing_state_payload(owner)["pin_net_delay_rise"]}
    )

    result = _assert_event(graph, owner)

    assert result["coverage"]["cell_forward_rows"] == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0])
def test_nonfinite_rc_and_signed_zero_do_not_prune_changed_events(value):
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = _owner_graph(owner)
    graph.propagate()
    owner.impulses["rise"][1] = value
    owner._baseline = owner._forward()
    graph.set_state(
        {"pin_net_impulse_rise": persistent_timing_state_payload(owner)["pin_net_impulse_rise"]}
    )

    _assert_event(graph, owner)


def test_insert_resize_middle_remove_compact_and_rollback_use_event_state():
    owner, graph = _graph(_base())
    graph.propagate()
    actions = []
    for action_id in (20, 3, 11):
        action, _ = _split(graph, 1, 3, action_id)
        actions.append(action)
        _assert_event(graph, None)
    graph.resize_buffer(1, {"action_id": 3, "buffer_master_name": "BUF2"})
    actions[1] = dict(actions[1], buffer_master_name="BUF2")
    _assert_event(graph, None)
    graph.remove_buffer(1, 3)
    actions.pop(1)
    _assert_event(graph, None)
    graph.commit(graph.checkpoint())
    graph.compact()
    _assert_event(graph, None)
    checkpoint = graph.checkpoint()
    before = graph.state_view()
    graph.remove_buffer(1, 20)
    _assert_event(graph, None)
    graph.restore(checkpoint)
    _same_state(graph.state_view(), before, bitwise=True)
    _assert_event(graph, None)
    # Compare active pin values by stable identity against a fresh TP owner.
    reference = _reference(owner, graph, actions)
    names = list(graph.mutation_view()["pin_names"])
    mapping = [names.index(name) for name in reference.pydb.pin_names]
    expected = persistent_timing_state_payload(reference)
    for name, values in expected.items():
        if not name.startswith("cell_arc_"):
            np.testing.assert_allclose(graph.state_view()[name][mapping], values, rtol=0, atol=1e-6)


def test_committing_pending_rc_requires_full_seed_instead_of_losing_event_seeds():
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = _owner_graph(owner)
    graph.propagate()
    _rc_changed(graph, owner, 1)
    graph.commit(graph.checkpoint())

    result = graph.propagate_incremental()

    assert result["mode"] == "full_seed"
    _same_state(graph.state_view(), persistent_timing_state_payload(owner))


def test_one_hundred_mixed_mutations_preserve_event_state_across_restore_and_compact():
    owner, graph = _graph(_base())
    graph.propagate()
    actions = []
    for step in range(100):
        checkpoint = graph.checkpoint()
        previous_actions = [dict(action) for action in actions]
        if len(actions) < 3 or step % 4 == 0:
            net, sink = (0, 1) if step % 2 else (1, 3)
            action, _ = _split(graph, net, sink, 100 + step)
            actions.append(dict(action, source_net=net))
        elif step % 4 == 2:
            selected = actions[step % len(actions)]
            master = "BUF2" if selected["buffer_master_name"] == "BUF1" else "BUF1"
            graph.resize_buffer(
                selected["source_net"],
                {
                    "action_id": selected["action_id"],
                    "buffer_master_name": master,
                },
            )
            selected["buffer_master_name"] = master
        else:
            selected = actions.pop(0 if step % 4 == 1 else -1)
            graph.remove_buffer(selected["source_net"], selected["action_id"])
        _assert_event(graph, None)
        if step % 7 == 0:
            graph.restore(checkpoint)
            actions = previous_actions
            _assert_event(graph, None)
        if step % 10 == 0:
            evaluated = graph.checkpoint()
            _buffer_oracle(graph, owner, actions)
            graph.restore(evaluated)
        if step % 15 == 0:
            graph.commit(graph.checkpoint())
            graph.compact()
            _assert_event(graph, None)
