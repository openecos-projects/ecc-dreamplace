"""Affected propagation must agree with complete Python TimingPropagation."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.placeio_ecc.incremental_timing import IncrementalTimingState
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_overlay_carrier import _export, _insert, _model, _state
from test_timing_carrier import _base

STATE_FIELDS = (
    "pin_rAAT",
    "pin_fAAT",
    "pin_rRAT",
    "pin_fRAT",
    "pin_rtran",
    "pin_ftran",
    "cell_arc_rr_delays",
    "cell_arc_fr_delays",
    "cell_arc_rf_delays",
    "cell_arc_ff_delays",
)


def _same(actual, expected):
    assert actual.baseline.as_dict() == pytest.approx(expected.baseline.as_dict(), abs=1e-8, rel=0)
    for field in STATE_FIELDS:
        torch.testing.assert_close(
            getattr(actual.model, field), getattr(expected.model, field), rtol=0, atol=1e-8
        )


def _full_reference_for_session(session, timing_model, runtime):
    """Build the independent full-forward oracle after the native probe."""
    from dreamplace.ops.rebuffer.payloads import steiner_payload

    if "steiner" in runtime.payload:
        return _model(runtime.model.pydb, steiner_view=runtime.payload["steiner"])

    states = [
        state
        for _, state in sorted(session.overlay.native_topologies.items())
    ]
    payload = rebuffer_cpp.export_overlay_carrier(
        session.base_pydb,
        steiner_payload(session.base_pydb, timing_model),
        states,
        list(session.overlay.actions),
    )
    carrier = CommittedTimingPyDB(
        session.base_pydb,
        SimpleNamespace(**payload["physical"]),
    )
    return _model(carrier, steiner_view=payload["steiner"])


@pytest.mark.parametrize("side_is_endpoint", [True, False])
def test_native_schedule_includes_unchanged_fanin_and_reverse_side_branch(side_is_endpoint):
    db = SimpleNamespace(
        pin_names=["a", "b", "y", "out", "side"],
        flat_inst_arcs_by_level=[[0, 2, 0, 0, 1, 0, 0, 0], [1, 2, 0, 1, 1, 0, 1, 0]],
        flat_inst_arcs_by_level_start=[0, 0, 2],
        net_flat_arcs=[[2, 3], [1, 4]],
        clock_pins=[],
        start_points=[0, 1],
        end_points=[3, 4] if side_is_endpoint else [3],
        endpoints_constraint_arcs=[],
        endpoints_timing_check_arcs=[],
    )
    before = np.zeros((6, 5))
    after = before.copy()
    after[0, 2] = 0.01

    schedule = rebuffer_cpp.affected_timing_schedule(db, db, before, after)

    assert schedule["forward_pins"].tolist() == [2, 3]
    assert schedule["forward_cells"].tolist() == [0, 1]
    assert schedule["backward_pins"].tolist() == [0, 1, 2, 3]
    assert schedule["backward_nets"].tolist() == ([0, 1] if side_is_endpoint else [0])


def test_timing_graph_index_handle_reuses_structure_for_new_seed():
    db = SimpleNamespace(
        pin_names=["a", "b", "y", "out", "side"],
        flat_inst_arcs_by_level=[[0, 2, 0, 0, 1, 0, 0, 0], [1, 2, 0, 1, 1, 0, 1, 0]],
        flat_inst_arcs_by_level_start=[0, 0, 2],
        net_flat_arcs=[[2, 3], [1, 4]],
        clock_pins=[],
        start_points=[0, 1],
        end_points=[3, 4],
        endpoints_constraint_arcs=[],
        endpoints_timing_check_arcs=[],
    )
    previous_rc = np.zeros((6, 5))
    current_rc = previous_rc.copy()
    current_rc[0, 2] = 0.01

    first = rebuffer_cpp.affected_timing_schedule(
        db, db, previous_rc, current_rc, [2]
    )
    second = rebuffer_cpp.affected_timing_schedule(
        db, db, previous_rc, current_rc, [1]
    )

    fresh = SimpleNamespace(**{name: value for name, value in db.__dict__.items()
                               if name != "timing_graph_index"})
    reference = rebuffer_cpp.affected_timing_schedule(
        fresh, fresh, previous_rc, current_rc, [1]
    )
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(second[key], value)


def test_timing_graph_index_structural_cache_reuses_distinct_carriers():
    def carrier():
        return SimpleNamespace(
            pin_names=["a", "b", "y", "out", "side"],
            flat_inst_arcs_by_level=[
                [0, 2, 0, 0, 1, 0, 0, 0],
                [1, 2, 0, 1, 1, 0, 1, 0],
            ],
            flat_inst_arcs_by_level_start=[0, 0, 2],
            net_flat_arcs=[[2, 3], [1, 4]],
            clock_pins=[],
            start_points=[0, 1],
            end_points=[3, 4],
            endpoints_constraint_arcs=[],
            endpoints_timing_check_arcs=[],
        )

    previous = carrier()
    current = carrier()
    previous_rc = np.zeros((6, 5))
    current_rc = previous_rc.copy()
    current_rc[0, 2] = 0.01

    schedule = rebuffer_cpp.affected_timing_schedule(
        previous, current, previous_rc, current_rc, [2]
    )

    assert schedule["timing_graph_cache_hit_current"] is True


def test_split_remove_and_restore_match_full_state_without_mutating_previous():
    baseline = _model(_base())
    saved = {field: getattr(baseline.model, field).clone() for field in STATE_FIELDS}
    state = _state(baseline)
    first = _insert(state, ratio=4 / 9)
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(state)
    payload, carrier = _export(baseline, state, [first])
    current = _model(
        carrier, steiner_view=payload["steiner"], incremental_state=IncrementalTimingState(baseline)
    )
    assert current.model.last_incremental_profile["schedule_cache_hit"] is False
    _same(current, _model(carrier, steiner_view=payload["steiner"]))
    for field, value in saved.items():
        assert torch.equal(getattr(baseline.model, field), value)
    assert current.model.arcs_info is baseline.model.arcs_info

    buffer = rebuffer_cpp.incremental_topology_view(state)["buffers"][0]
    second = _insert(state, action_id=1, parent=buffer["output_node"], master="BUF1")
    payload2, carrier2 = _export(baseline, state, [first, second])
    after = _model(
        carrier2,
        steiner_view=payload2["steiner"],
        incremental_state=IncrementalTimingState(current),
    )
    _same(after, _model(carrier2, steiner_view=payload2["steiner"]))
    rebuffer_cpp.restore_incremental_topology(state, checkpoint)
    payload3, carrier3 = _export(baseline, state, [first])
    restored = _model(
        carrier3, steiner_view=payload3["steiner"], incremental_state=IncrementalTimingState(after)
    )
    _same(restored, current)


def _join_base():
    base = _base()
    base.num_nodes, base.num_terminal_NIs = 5, 3
    base.node_names = ["u", "a", "IN", "IN2", "OUT"]
    base.node_master_names = ["BUF1", "AND1", "", "", ""]
    base.node_x = [10000.0, 100000.0, 0.0, 0.0, 150000.0]
    base.node_y = [0.0, 0.0, 0.0, 10000.0, 0.0]
    base.node_size_x = base.node_size_y = [1000.0, 1000.0, 0.0, 0.0, 0.0]
    base.pin_names = ["IN", "u:A", "u:Y", "a:A", "IN2", "a:B", "a:Y", "OUT"]
    base.pin2node_map = [2, 0, 0, 1, 3, 1, 1, 4]
    base.pin_offset_x = base.pin_offset_y = [0.0] * 8
    base.pin_direct = ["R0"] * 8
    base.net_names = ["n0", "n1", "n2", "n3"]
    base.pin2net_map = [0, 0, 1, 1, 2, 2, 3, 3]
    base.flat_net2pin_map = list(range(8))
    base.flat_net2pin_start_map = [0, 2, 4, 6, 8]
    base.net2driver_pin_map = [0, 2, 4, 6]
    base.start_points = [0, 4]
    base.end_points = [7]
    base.inrdelays, base.infdelays = [2.0, 100.0], [3.0, 120.0]
    base.inrtrans, base.inftrans = [15.0, 25.0], [20.0, 30.0]
    base.flat_libcell_names.append("AND1")
    base.flat_libcell_info.append(["AND1", 1, 1, 0])
    base.flat_libcell_width.append(1000.0)
    base.flat_libcell_height.append(1000.0)
    base.flat_libcell_leakage.append(0.0)
    base.main_id_2_cell_id_start = [0, 2, 3]
    base.inst_main_id, base.inst_libcell_offset = [0, 1, -1, -1, -1], [0] * 5
    base.cell_id_2_arc_id_start = [0, 1, 2, 4]
    base.cell_id_2_libpin_id_start = [0, 2, 4, 7]
    base.pin_2_libpin_offset = [-1, 0, 1, 0, -1, 1, 2, -1]
    base.flat_lib_pin_names += ["A", "B", "Y"]
    for field in ("flat_lib_pin_cap", "flat_lib_pin_rcap", "flat_lib_pin_fcap"):
        getattr(base, field).extend([0.01, 0.01, 0.0])
    base.flat_lib_pin_cap_limit += [0.0, 0.0, 1.0]
    base.flat_lib_pin_slew_limit += [100.0] * 3
    base.flat_libarc_info += [["A", "Y", 2, 0, 1, 0], ["B", "Y", 2, 1, 1, 0]]
    base.flat_inst_arcs_by_level += [[3, 6, 2, 2, 1, 0, 0, 1], [5, 6, 2, 3, 1, 0, 1, 1]]
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        for suffix in ("values", "trans_table", "cap_table", "dim"):
            values = getattr(base, f"{prefix}_flat_luts_{suffix}")
            values.extend([list(values[0]), list(values[1])])
    return CommittedTimingPyDB(base, base)


def test_changed_branch_reduces_with_unchanged_fanin_and_preserves_other_branch_state():
    baseline = _model(_join_base())
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    payload, carrier = _export(baseline, state, [action])
    current = _model(
        carrier, steiner_view=payload["steiner"], incremental_state=IncrementalTimingState(baseline)
    )
    _same(current, _model(carrier, steiner_view=payload["steiner"]))
    for pin in (0, 1, 4, 5):
        assert torch.equal(current.model.pin_rAAT[pin], baseline.model.pin_rAAT[pin])
    assert current.model.last_incremental_profile["forward_pins"] < current.model.num_pins


def test_event_forward_propagation_is_opt_in_and_matches_full_timing(monkeypatch):
    monkeypatch.setenv("REBUFFER_EVENT_PROPAGATION", "1")
    baseline = _model(_join_base())
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    payload, carrier = _export(baseline, state, [action])
    current = _model(
        carrier,
        steiner_view=payload["steiner"],
        incremental_state=IncrementalTimingState(baseline),
    )
    _same(current, _model(carrier, steiner_view=payload["steiner"]))
    assert current.model.last_incremental_profile["event_forward"] is True
    assert current.model.last_incremental_profile["forward_pins"] <= current.model.num_pins


def test_event_native_recovery_and_clk2q_are_active_and_match_full_timing(monkeypatch):
    """Exercise native recovery/clk->Q through the incremental carrier path."""

    for name in (
        "REBUFFER_EVENT_PROPAGATION",
        "REBUFFER_EVENT_NATIVE_KERNELS",
        "REBUFFER_EVENT_NATIVE_RAT",
        "REBUFFER_EVENT_NATIVE_CLK2Q",
        "REBUFFER_EVENT_NATIVE_RAT_ORACLE",
    ):
        monkeypatch.setenv(name, "1")

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
    base.endpoints_timing_check_arcs = np.asarray(
        [[0, 3, 0, 0, 1, 1, 3, 0]], dtype=np.int64
    )
    base.endpoints_timing_check_max_valid = np.ones((1, 2), dtype=bool)
    base.endpoints_timing_check_max_reason = np.zeros((1, 2), dtype=np.int32)

    baseline = _model(base)
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    payload, carrier = _export(baseline, state, [action])
    current = _model(
        carrier,
        steiner_view=payload["steiner"],
        incremental_state=IncrementalTimingState(baseline),
    )
    reference = _model(carrier, steiner_view=payload["steiner"])

    _same(current, reference)
    profile = current.model.last_incremental_profile
    assert profile["native_recovery_levels"] > 0
    assert profile["native_clk2q_levels"] > 0


def test_one_hundred_synthetic_split_and_restore_actions_match_full_timing():
    from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession

    base = _join_base()
    model = _model(base)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=model)
    session.begin()
    rng = random.Random(8137)
    for attempt in range(100):
        checkpoint = session.checkpoint()["checkpoint_id"]
        previous = {name: getattr(model.model, name).clone() for name in STATE_FIELDS}
        net, driver, load = (1, 2, 3) if attempt % 2 else (2, 4, 5)
        source, sink = base.pin2node_map[driver], base.pin2node_map[load]
        ratio = rng.uniform(0.1, 0.9)
        action = {
            "action_kind": "buffer_insert",
            "net_name": base.net_names[net],
            "driver_pin_name": base.pin_names[driver].replace(":", "/"),
            "downstream_pin_names": [base.pin_names[load].replace(":", "/")],
            "buffer_master_name": "BUF1" if attempt % 3 else "BUF2",
            "candidate_location_x_dbu": round(
                base.node_x[source] + ratio * (base.node_x[sink] - base.node_x[source])
            ),
            "candidate_location_y_dbu": round(
                base.node_y[source] + ratio * (base.node_y[sink] - base.node_y[source])
            ),
        }
        session.apply([action])
        session.evaluate()
        runtime = model.current_graph()
        _same(runtime.model, _full_reference_for_session(session, model, runtime))
        if attempt % 3 == 0:
            session.restore(checkpoint)
            for name, values in previous.items():
                torch.testing.assert_close(getattr(model.model, name), values, atol=1e-8, rtol=0)


def test_multi_slot_rc_patch_matches_full_carrier_for_eight_buffer_chain(monkeypatch):
    from dreamplace.ops.placeio_ecc.persistent_timing import _patch_rc

    monkeypatch.setenv("REBUFFER_PATCH_EVENT_SCHEDULE", "1")
    base = _base()
    model = _model(base)
    state = _state(model)
    actions = []
    for action_id in range(8):
        view = rebuffer_cpp.incremental_topology_view(state)
        child = 3
        parent = next(
            edge["parent"]
            for edge in view["edges"]
            if edge["child"] == child
        )
        source = view["coordinates"][parent]
        target = view["coordinates"][child]
        ratio = 0.5
        point = [a + ratio * (b - a) for a, b in zip(source, target, strict=True)]
        master = "BUF1" if action_id % 2 else "BUF2"
        rebuffer_cpp.apply_incremental_buffer_split(
            state,
            {
                "action_id": action_id,
                "parent_node_id": parent,
                "child_node_id": child,
                "split_ratio": ratio,
                "downstream_pin_ids": [3],
                "x_dbu": point[0],
                "y_dbu": point[1],
                "buffer_master_name": master,
            },
        )
        actions.append(
            {
                "session_buffer_id": action_id,
                "buffer_master_name": master,
                "candidate_location_x_dbu": round(point[0]),
                "candidate_location_y_dbu": round(point[1]),
                "split_ratio": ratio,
                "net_name": "n1",
                "downstream_pin_names": ["OUT"],
            }
        )

    patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        state,
        {**actions[-1], "downstream_pin_ids": [3]},
        int(model.new_x.numel()),
        0,
    )
    assert patch["buffer_count"] == 8
    assert len(patch["slot_pin_ids"]) == 16
    assert patch["graph"]["num_pins"] == len(base.pin_names) + 16
    schedule = patch["schedule"]
    for key in (
        "pin_rank",
        "forward_seed_pins",
        "reverse_seed_pins",
        "cell_out_start",
        "cell_out_ids",
        "cell_in_start",
        "cell_in_ids",
        "net_out_start",
        "net_out_ids",
        "net_in_start",
        "net_in_ids",
    ):
        assert key in schedule
    assert len(schedule["pin_rank"]) == patch["graph"]["num_pins"]

    rc = _patch_rc(model, patch, actions[-1])
    payload, carrier = _export(model, state, actions)
    full = _model(carrier, steiner_view=payload["steiner"])
    for mode in ("rise", "fall"):
        for field in ("loads", "delays", "impulses"):
            torch.testing.assert_close(
                rc[field][mode],
                getattr(full, field)[mode],
                rtol=0,
                atol=1e-8,
            )


def test_incremental_patch_schedule_cache_reuses_exact_topology(monkeypatch):
    monkeypatch.setenv("REBUFFER_PATCH_SCHEDULE_CACHE", "1")
    base = _base()
    model = _model(base)
    state = _state(model)
    action = _insert(state, ratio=4 / 9)
    action_payload = {**action, "downstream_pin_ids": [3]}

    first = rebuffer_cpp.build_incremental_timing_patch(
        base, state, action_payload, int(model.new_x.numel()), 0
    )
    second = rebuffer_cpp.build_incremental_timing_patch(
        base, state, action_payload, int(model.new_x.numel()), 0
    )

    assert first["schedule"]["schedule_cache_hit"] is False
    assert second["schedule"]["schedule_cache_hit"] is True
    assert second["patch_base_cache_hit"] is True
    assert set(first["schedule"]) == set(second["schedule"])
    for key, value in first["schedule"].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second["schedule"][key])

    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(state)
    buffer = rebuffer_cpp.incremental_topology_view(state)["buffers"][0]
    next_action = _insert(
        state,
        action_id=1,
        parent=buffer["output_node"],
        child=3,
        master="BUF2",
    )
    next_patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        state,
        {**next_action, "downstream_pin_ids": [3]},
        int(model.new_x.numel()),
        0,
    )
    assert next_patch["schedule"]["schedule_cache_hit"] is False

    rebuffer_cpp.restore_incremental_topology(state, checkpoint)
    restored = rebuffer_cpp.build_incremental_timing_patch(
        base, state, action_payload, int(model.new_x.numel()), 0
    )
    assert restored["schedule"]["schedule_cache_hit"] is True

    buffer = rebuffer_cpp.incremental_topology_view(state)["buffers"][0]
    _insert(
        state,
        action_id=1,
        parent=buffer["output_node"],
        child=3,
        master="BUF2",
    )
    rebuffer_cpp.remove_incremental_buffer(state, 1)
    removed = rebuffer_cpp.build_incremental_timing_patch(
        base, state, action_payload, int(model.new_x.numel()), 0
    )
    assert removed["schedule"]["schedule_cache_hit"] is False
