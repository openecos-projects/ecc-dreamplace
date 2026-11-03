"""Numerical gates for native current partitions and reused Steiner vertices."""

from types import SimpleNamespace

import pytest
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo
from test_timing_carrier import _base, _reference_after


def _model(pydb, **kwargs):
    return RebufferTimingModel(pydb, use_pin_offset=False, **kwargs)


def _state(model):
    return rebuffer_cpp.incremental_topology_from_steiner(
        steiner_payload(model.pydb, model),
        1,
        2,
        model.r_unit,
        model.c_unit,
    )


def _insert(state, *, action_id=0, parent=None, child=3, ratio=0.5, master="BUF2"):
    view = rebuffer_cpp.incremental_topology_view(state)
    if parent is None:
        parent = next(edge["parent"] for edge in view["edges"] if edge["child"] == child)
    source, target = view["coordinates"][parent], view["coordinates"][child]
    point = [a + ratio * (b - a) for a, b in zip(source, target, strict=True)]
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
    return {
        "session_buffer_id": action_id,
        "buffer_master_name": master,
        "candidate_location_x_dbu": round(point[0]),
        "candidate_location_y_dbu": round(point[1]),
    }


def _export(base_model, state, actions):
    payload = rebuffer_cpp.export_overlay_carrier(
        base_model.pydb,
        steiner_payload(base_model.pydb, base_model),
        [state],
        actions,
    )
    physical = SimpleNamespace(**payload["physical"])
    carrier = CommittedTimingPyDB(base_model.pydb, physical)
    return payload, carrier


def test_force_full_overlay_runtime_materializes_empty_native_carrier():
    base = _base()
    timing = _model(base)
    from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

    overlay = BufferOverlay(base, timing_model=timing)
    timing.bind_overlay(overlay)

    assert timing.current_graph() is None
    runtime = timing.force_full_overlay_runtime()

    assert runtime is not None
    assert runtime.model._arcs_info is timing._arcs_info
    assert runtime.payload["original_net_by_net"] == [0, 1]
    assert timing.current_graph() is runtime


def test_persistent_graph_bridge_owns_commit_projection(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

    class Bridge:
        def __init__(self):
            self.calls = []

        def finalize_actions(self, actions, *, overlay):
            self.calls.append((list(actions), overlay))
            return [{"projected": True, **dict(actions[0])}]

    timing = SimpleNamespace(
        use_persistent_graph=True,
        _persistent_graph_bridge=Bridge(),
        create_net_overlay=lambda _name: object(),
        _net_record=lambda _net_id: {},
    )
    monkeypatch.setenv("REBUFFER_PERSISTENT_GRAPH_FINALIZE", "1")
    base = _base()
    overlay = BufferOverlay(base, timing_model=timing)
    overlay.actions = [
        {
            "action_kind": "buffer_insert",
            "session_buffer_id": 0,
            "net_name": "n1",
            "buffer_master_name": "BUF2",
            "candidate_location_x_dbu": 50000,
            "candidate_location_y_dbu": 0,
            "downstream_pin_names": ["OUT"],
        }
    ]
    state = _state(_model(base))
    overlay.native_topologies["n1"] = state

    monkeypatch.setattr(
        rebuffer_cpp,
        "incremental_topology_view",
        lambda _state: {"buffers": [{"action_id": 0}]},
    )
    projected = overlay.commit_actions()

    assert projected[0]["projected"] is True
    assert timing._persistent_graph_bridge.calls[0][1] is overlay
    assert overlay.projection_audit["projection_source"] == "persistent_graph"


def test_persistent_graph_projection_falls_back_to_canonical_finalizer(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

    class Bridge:
        def finalize_actions(self, actions, *, overlay):
            raise ValueError("stale graph state")

    timing = SimpleNamespace(
        use_persistent_graph=True,
        _persistent_graph_bridge=Bridge(),
        create_net_overlay=lambda _name: object(),
        _net_record=lambda _net_id: {},
    )
    monkeypatch.setenv("REBUFFER_PERSISTENT_GRAPH_FINALIZE", "1")
    base = _base()
    overlay = BufferOverlay(base, timing_model=timing)
    overlay.actions = [
        {
            "action_kind": "buffer_insert",
            "session_buffer_id": 0,
            "net_name": "n1",
            "buffer_master_name": "BUF2",
            "candidate_location_x_dbu": 50000,
            "candidate_location_y_dbu": 0,
            "downstream_pin_names": ["OUT"],
        }
    ]
    overlay.native_topologies["n1"] = _state(_model(base))
    monkeypatch.setattr(
        rebuffer_cpp,
        "incremental_topology_view",
        lambda _state: {"buffers": [{"action_id": 0}]},
    )
    monkeypatch.setattr(
        rebuffer_cpp,
        "finalize_overlay_actions",
        lambda _base, _states, actions: [{"canonical": True, **dict(actions[0])}],
    )

    projected = overlay.commit_actions()

    assert projected[0]["canonical"] is True
    assert overlay.projection_audit["projection_source"] == "native_overlay"
    assert "stale graph state" in overlay.projection_audit["persistent_graph_fallback"]


def test_persistent_graph_keeps_canonical_projection_by_default(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

    class Bridge:
        def finalize_actions(self, actions, *, overlay):
            raise AssertionError("diagnostic graph finalizer must stay disabled")

    timing = SimpleNamespace(
        use_persistent_graph=True,
        _persistent_graph_bridge=Bridge(),
        create_net_overlay=lambda _name: object(),
        _net_record=lambda _net_id: {},
    )
    base = _base()
    overlay = BufferOverlay(base, timing_model=timing)
    overlay.actions = [
        {
            "action_kind": "buffer_insert",
            "session_buffer_id": 0,
            "net_name": "n1",
            "buffer_master_name": "BUF2",
            "candidate_location_x_dbu": 50000,
            "candidate_location_y_dbu": 0,
            "downstream_pin_names": ["OUT"],
        }
    ]
    overlay.native_topologies["n1"] = _state(_model(base))
    monkeypatch.setattr(
        rebuffer_cpp,
        "incremental_topology_view",
        lambda _state: {"buffers": [{"action_id": 0}]},
    )
    monkeypatch.setattr(
        rebuffer_cpp,
        "finalize_overlay_actions",
        lambda _base, _states, actions: [{"canonical": True, **dict(actions[0])}],
    )

    projected = overlay.commit_actions()

    assert projected[0]["canonical"] is True
    assert overlay.projection_audit["projection_source"] == "native_overlay"


def test_persistent_graph_bridge_tracks_overlay_checkpoint_restore():
    from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

    base = _base()
    timing = _model(base, use_persistent_graph=True)
    overlay = BufferOverlay(base, timing_model=timing)
    timing.bind_overlay(overlay)

    first = {
        "action_kind": "buffer_insert",
        "net_name": "n1",
        "driver_pin_name": "u/Y",
        "downstream_pin_names": ["OUT"],
        "buffer_master_name": "BUF2",
        "candidate_location_x_dbu": 50000,
        "candidate_location_y_dbu": 0,
        "split_ratio": 0.5,
    }
    overlay.apply([first])
    timing._native_setup_path_cache = {"stale": True}
    timing._native_setup_path_cache_epoch = 0
    timing.evaluate(base, overlay)
    assert timing._native_setup_path_cache is None
    checkpoint = overlay.checkpoint("persistent-graph-prefix")
    assert checkpoint["checkpoint_id"] >= 0
    bridge_checkpoint = timing._persistent_graph_bridge.checkpoint()
    assert bridge_checkpoint["mode"] == "bindings_only"
    assert timing._persistent_graph_bridge.graph.summary()["net_count"] == 1
    assert len(timing._persistent_graph_bridge.graph.topology_view(1)["buffers"]) == 1

    second = {
        **first,
        "candidate_location_x_dbu": 75000,
        "session_buffer_id": 1,
    }
    overlay.apply([second])
    timing.evaluate(base, overlay)
    assert len(timing._persistent_graph_bridge.graph.topology_view(1)["buffers"]) == 2

    overlay.restore(checkpoint["checkpoint_id"])

    assert len(overlay.actions) == 1
    assert len(timing._persistent_graph_bridge.graph.topology_view(1)["buffers"]) == 1
    assert timing._persistent_graph_bridge.graph.summary()["net_count"] == 1
    assert timing.evaluate(base, overlay)["topology_source"] == "native_current_partitions_cached_snapshot"
    assert timing.evaluate(base, overlay)["topology_source"] == "native_current_partitions_cached_snapshot"

    overlay.rollback()
    assert timing._persistent_graph_bridge.graph.summary()["net_count"] == 0


def test_current_partition_timing_matches_independent_graph_without_flute_rebuild(monkeypatch):
    base = _base()
    baseline = _model(base)
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    reference = _model(_reference_after(base))
    payload, carrier = _export(baseline, state, [action])

    def forbidden(*args, **kwargs):
        pytest.fail("current overlay rebuilt FLUTE instead of using native partitions")

    monkeypatch.setattr(SteinerTopo, "rebuild_tree", forbidden)
    current = _model(carrier, steiner_view=payload["steiner"])

    assert current.baseline.as_dict() == pytest.approx(
        reference.baseline.as_dict(), abs=1e-8, rel=0
    )
    assert current.baseline.wns_ps != pytest.approx(baseline.baseline.wns_ps)
    assert current.pydb.pin_names[:4] == base.pin_names
    assert payload["original_net_by_net"] == [0, 1, 1]
    assert len(payload["steiner"]["net_flat_topo_sort_start"]) == 4
    assert carrier.pin2net_map[3] == 2


def test_exported_current_partition_includes_vertex_domain_rc():
    baseline = _model(_base())
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    payload, _ = _export(baseline, state, [action])
    steiner = payload["steiner"]

    assert len(steiner["edge_resistance"]) == len(steiner["pin_fa"])
    assert len(steiner["node_capacitance"]) == len(steiner["pin_fa"])
    assert all(value >= 0.0 for value in steiner["edge_resistance"])
    assert all(value >= 0.0 for value in steiner["node_capacitance"])

    children = steiner["flat_pin_to"]
    expected_wire_cap = sum(
        steiner["edge_resistance"][child] / baseline.r_unit * baseline.c_unit
        for child in children
    )
    assert sum(steiner["node_capacitance"]) == pytest.approx(
        expected_wire_cap, abs=1.0e-12, rel=0
    )


def test_second_split_remove_and_checkpoint_restore_use_current_partitions():
    baseline = _model(_base())
    state = _state(baseline)
    first = _insert(state, ratio=4 / 9)
    checkpoint = rebuffer_cpp.checkpoint_incremental_topology(state)
    payload, carrier = _export(baseline, state, [first])
    before = _model(carrier, steiner_view=payload["steiner"])
    buffer = rebuffer_cpp.incremental_topology_view(state)["buffers"][0]
    second = _insert(state, action_id=1, parent=buffer["output_node"], master="BUF1")
    payload2, carrier2 = _export(baseline, state, [first, second])
    current = _model(carrier2, steiner_view=payload2["steiner"])
    fresh = _model(carrier2)

    assert current.baseline.as_dict() == pytest.approx(fresh.baseline.as_dict(), abs=1e-8, rel=0)
    assert len(carrier2.flat_inst_arcs_by_level) == 3
    assert carrier2.pin2net_map[3] == 3
    assert current.baseline.wns_ps != pytest.approx(before.baseline.wns_ps)

    rebuffer_cpp.remove_incremental_buffer(state, 1)
    removed_payload, removed = _export(baseline, state, [first])
    assert removed_payload == payload
    assert _model(
        removed, steiner_view=removed_payload["steiner"]
    ).baseline.as_dict() == pytest.approx(
        before.baseline.as_dict(),
        abs=1e-8,
        rel=0,
    )
    _insert(state, action_id=1, parent=buffer["output_node"], master="BUF1")
    rebuffer_cpp.restore_incremental_topology(state, checkpoint)
    restored_payload, restored = _export(baseline, state, [first])
    assert restored_payload == payload
    assert _model(
        restored, steiner_view=restored_payload["steiner"]
    ).baseline.as_dict() == pytest.approx(
        before.baseline.as_dict(),
        abs=1e-8,
        rel=0,
    )


def test_parent_delta_patch_matches_full_carrier_for_cross_net_action(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession

    base = _base()
    timing = _model(base)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=timing)
    session.begin()
    first = {
        "action_kind": "buffer_insert",
        "net_name": "n1",
        "driver_pin_name": "u/Y",
        "downstream_pin_names": ["OUT"],
        "buffer_master_name": "BUF2",
        "candidate_location_x_dbu": 50000,
        "candidate_location_y_dbu": 0,
        "split_ratio": 0.5,
    }
    session.apply([first])
    parent = session.evaluate()
    checkpoint = session.checkpoint("cross_net_parent")
    second = {
        "action_kind": "buffer_insert",
        "net_name": "n0",
        "driver_pin_name": "IN",
        "downstream_pin_names": ["u:A"],
        "buffer_master_name": "BUF2",
        "candidate_location_x_dbu": 5000,
        "candidate_location_y_dbu": 0,
        "split_ratio": 0.5,
    }
    session.apply([second])
    monkeypatch.setattr(
        rebuffer_cpp,
        "export_overlay_carrier",
        lambda *args, **kwargs: pytest.fail(
            "cross-net parent delta rebuilt the full overlay carrier"
        ),
    )
    delta = session.evaluate()
    monkeypatch.undo()

    states = [
        session.overlay.native_topologies[name]
        for name in sorted(session.overlay.native_topologies)
    ]
    reference_base = _model(base)
    payload = rebuffer_cpp.export_overlay_carrier(
        base,
        steiner_payload(base, reference_base),
        states,
        list(session.overlay.actions),
    )
    reference = _model(
        CommittedTimingPyDB(base, SimpleNamespace(**payload["physical"])),
        steiner_view=payload["steiner"],
    ).baseline.as_dict()

    assert delta["topology_source"] == "native_session_parent_delta_patch"
    assert delta["wns_ps"] == pytest.approx(reference["wns_ps"], abs=1e-8, rel=0)
    assert delta["tns_ps"] == pytest.approx(reference["tns_ps"], abs=1e-8, rel=0)
    assert delta["endpoint_count"] == reference["endpoint_count"]
    runtime = timing.current_graph()
    assert runtime is not None
    assert "net_flat_topo_sort" in runtime.payload["steiner"]
    assert runtime.payload["original_net_by_net"] == [0, 1, 1, 0]

    def forbidden_full_materialization():
        pytest.fail("cross-net parent delta lost the native current-tree payload")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(timing, "force_full_overlay_runtime", forbidden_full_materialization)
    try:
        from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext

        context = RebufferPlanContext(base, timing_model=timing)
        views = context.native_queue_views()
        tree_payload = context.native_current_tree_payload(0)
    finally:
        monkeypatch.undo()
    assert views["status"] == "ok"
    assert views["original_net_by_net"] == [0, 1, 1, 0]
    assert tree_payload["status"] == "ok"
    assert tree_payload["topology_source"] == "active_timing_propagation_patch_steiner"
    tree = rebuffer_cpp.export_current_driver_tree(tree_payload["current_tree"]())
    assert tree["driver_pin_id"] == int(timing.active_pydb.net2driver_pin_map[0])
    assert timing.current_graph() is runtime

    session.restore(checkpoint["checkpoint_id"])
    restored = session.evaluate()
    assert restored["wns_ps"] == pytest.approx(parent["wns_ps"], abs=1e-8, rel=0)
    assert restored["tns_ps"] == pytest.approx(parent["tns_ps"], abs=1e-8, rel=0)


def test_compact_parent_delta_can_be_reused_for_second_cross_net_action(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
    from test_incremental_timing import _join_base

    base = _join_base()
    timing = _model(base)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=timing)
    session.begin()

    def action(net_name, driver, load, x, y=0):
        return {
            "action_kind": "buffer_insert",
            "net_name": net_name,
            "driver_pin_name": driver,
            "downstream_pin_names": [load],
            "buffer_master_name": "BUF1",
            "candidate_location_x_dbu": x,
            "candidate_location_y_dbu": y,
            "split_ratio": 0.5,
        }

    session.apply([action("n1", "u/Y", "a:A", 55000)])
    session.evaluate()
    session.apply([action("n0", "IN", "u:A", 5000)])
    parent = session.evaluate()
    parent_runtime = timing.current_graph()
    assert parent["topology_source"] == "native_session_parent_delta_patch"
    assert parent_runtime is not None
    assert parent_runtime.model.pydb._compact_timing_graph is True
    assert callable(parent_runtime.model._pin_id)
    assert parent_runtime.model.new_x.numel() > 0
    assert hasattr(parent_runtime.model, "steiner")
    assert hasattr(parent_runtime.model, "rc")

    monkeypatch.setattr(
        rebuffer_cpp,
        "export_overlay_carrier",
        lambda *args, **kwargs: pytest.fail(
            "second compact parent delta rebuilt the full overlay carrier"
        ),
    )
    session.apply([action("n2", "IN2", "a:B", 50000, 5000)])
    delta = session.evaluate()
    monkeypatch.undo()

    states = [
        session.overlay.native_topologies[name]
        for name in sorted(session.overlay.native_topologies)
    ]
    reference_base = _model(base)
    payload = rebuffer_cpp.export_overlay_carrier(
        base,
        steiner_payload(base, reference_base),
        states,
        list(session.overlay.actions),
    )
    reference = _model(
        CommittedTimingPyDB(base, SimpleNamespace(**payload["physical"])),
        steiner_view=payload["steiner"],
    ).baseline.as_dict()

    assert delta["topology_source"] == "native_session_parent_delta_patch"
    assert delta["wns_ps"] == pytest.approx(reference["wns_ps"], abs=1e-8, rel=0)
    assert delta["tns_ps"] == pytest.approx(reference["tns_ps"], abs=1e-8, rel=0)
    assert delta["endpoint_count"] == reference["endpoint_count"]
    runtime = timing.current_graph()
    assert runtime is not None
    assert runtime.model.pydb._compact_timing_graph is True
    assert len(runtime.model.pydb.pin_names) == len(set(runtime.model.pydb.pin_names))


def test_cross_net_strip_uses_incremental_carrier_removal(monkeypatch):
    from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession

    base = _base()
    timing = _model(base)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=timing)
    session.begin()
    session.apply(
        [
            {
                "action_kind": "buffer_insert",
                "net_name": "n1",
                "driver_pin_name": "u/Y",
                "downstream_pin_names": ["OUT"],
                "buffer_master_name": "BUF2",
                "candidate_location_x_dbu": 50000,
                "candidate_location_y_dbu": 0,
                "split_ratio": 0.5,
            },
            {
                "action_kind": "buffer_insert",
                "net_name": "n0",
                "driver_pin_name": "IN",
                "downstream_pin_names": ["u:A"],
                "buffer_master_name": "BUF2",
                "candidate_location_x_dbu": 5000,
                "candidate_location_y_dbu": 0,
                "split_ratio": 0.5,
            },
        ]
    )
    session.evaluate()

    calls = {"incremental": 0, "full": 0}
    incremental = rebuffer_cpp.rebuild_timing_carrier_incremental
    full = rebuffer_cpp.rebuild_timing_carrier

    def observe_incremental(*args, **kwargs):
        calls["incremental"] += 1
        return incremental(*args, **kwargs)

    def observe_full(*args, **kwargs):
        calls["full"] += 1
        return full(*args, **kwargs)

    monkeypatch.setattr(
        rebuffer_cpp, "rebuild_timing_carrier_incremental", observe_incremental
    )
    monkeypatch.setattr(rebuffer_cpp, "rebuild_timing_carrier", observe_full)

    # Model a current-tree replacement: the accepted parent still contains
    # both nets, while the replacement candidate first removes n0's old
    # buffer and then inserts a different one. The final overlay therefore
    # has fewer synthetic timing pins than its predecessor and a new key,
    # so the carrier path cannot be satisfied by the cached parent snapshot.
    session.strip_net("n0")
    session.apply(
        [
            {
                "action_kind": "buffer_insert",
                "net_name": "n0",
                "driver_pin_name": "IN",
                "downstream_pin_names": ["u:A"],
                "buffer_master_name": "BUF2",
                "candidate_location_x_dbu": 2500,
                "candidate_location_y_dbu": 0,
                "split_ratio": 0.25,
            }
        ]
    )
    stripped = session.evaluate()

    from test_incremental_timing import _full_reference_for_session

    reference = _full_reference_for_session(session, timing, timing.current_graph())
    assert stripped["wns_ps"] == pytest.approx(reference.baseline.wns_ps, abs=1e-8, rel=0)
    assert stripped["tns_ps"] == pytest.approx(reference.baseline.tns_ps, abs=1e-8, rel=0)
    assert calls == {"incremental": 1, "full": 0}
    assert stripped["topology_source"] == "native_current_partitions"


def test_branch_split_keeps_sibling_in_original_electrical_partition():
    import numpy as np

    base = _base()
    base.num_nodes, base.num_terminal_NIs = 4, 3
    base.node_names.append("OUT2")
    base.node_master_names.append("")
    base.node_x.append(10000.0)
    base.node_y.append(20000.0)
    base.node_size_x.append(0.0)
    base.node_size_y.append(0.0)
    base.pin_names.append("OUT2")
    base.pin2node_map.append(3)
    base.pin2net_map.append(1)
    base.pin_offset_x.append(0.0)
    base.pin_offset_y.append(0.0)
    base.pin_direct.append("R0")
    base.flat_net2pin_map.append(4)
    base.flat_net2pin_start_map[-1] += 1
    base.end_points = base.endpoint_pin_ids = [3, 4]
    base.endpoints_rRAT = base.endpoints_fRAT = [0.0, 0.0]
    base.outcaps = [0.0, 0.0]
    base.endpoints_max_valid = np.ones((2, 2), dtype=bool)
    base.endpoints_max_reason = np.zeros((2, 2), dtype=np.int32)
    # The base physical graph is reconstructed separately from the explicit
    # star Steiner fixture so graph compilation and RC construction agree.
    rebuilt = CommittedTimingPyDB(base, base)
    star = {
        "net_flat_topo_sort": [0, 1, 2, 3, 4],
        "net_flat_topo_sort_start": [0, 2, 5],
        "pin_fa": [-1, 0, -1, 2, 2],
        "flat_pin_to_start": [0, 1, 1, 3, 3, 3],
        "flat_pin_to": [1, 3, 4],
        "flat_pin_from": [0, 2, 2],
        "node_x": [0.0, 10000.0, 10000.0, 100000.0, 10000.0],
        "node_y": [0.0, 0.0, 0.0, 0.0, 20000.0],
    }
    baseline = _model(rebuilt, steiner_view=star)
    state = _state(baseline)
    action = _insert(state, ratio=4 / 9)
    payload, carrier = _export(baseline, state, [action])
    current = _model(carrier, steiner_view=payload["steiner"])
    fresh = _model(carrier)

    assert carrier.pin2net_map[3:5] == [2, 1]
    assert current.baseline.as_dict() == pytest.approx(fresh.baseline.as_dict(), abs=1e-8, rel=0)


def test_net_local_synthetic_ids_do_not_alias_another_nets_physical_pins():
    base = _base()
    base.num_nodes = 5
    base.num_terminal_NIs = 4
    base.node_names.extend(["AUX_IN", "AUX_OUT"])
    base.node_master_names.extend(["", ""])
    base.node_x.extend([200000.0, 300000.0])
    base.node_y.extend([10000.0, 10000.0])
    base.node_size_x.extend([0.0, 0.0])
    base.node_size_y.extend([0.0, 0.0])
    base.pin_names.extend(["AUX_IN", "AUX_OUT"])
    base.pin2node_map.extend([3, 4])
    base.pin2net_map.extend([2, 2])
    base.pin_offset_x.extend([0.0, 0.0])
    base.pin_offset_y.extend([0.0, 0.0])
    base.pin_direct.extend(["R0", "R0"])
    base.net_names.append("aux")
    base.net2driver_pin_map.append(4)
    base.flat_net2pin_map.extend([4, 5])
    base.flat_net2pin_start_map.append(6)
    base.start_points = [0, 4]
    base.inrdelays = base.infdelays = [2.0, 3.0]
    base.inrtrans = base.inftrans = [15.0, 20.0]
    model = _model(CommittedTimingPyDB(base, base))
    state = _state(model)
    action = _insert(state, ratio=4 / 9)
    payload, carrier = _export(model, state, [action])

    assert rebuffer_cpp.incremental_topology_view(state)["buffers"][0]["input_node"] == 4
    assert carrier.pin_names[:6] == base.pin_names
    assert carrier.pin2net_map[4:] == [2, 2, 1, 3]
    assert payload["state_node_by_vertex"][4] == payload["state_node_by_vertex"][6] == 4
    assert (payload["state_net_by_vertex"][4], payload["state_net_by_vertex"][6]) == (2, 1)
    assert _model(carrier, steiner_view=payload["steiner"]).baseline.as_dict() == pytest.approx(
        _model(carrier).baseline.as_dict(),
        abs=1e-8,
        rel=0,
    )


def test_projection_uses_current_tree_source_identity_before_candidate_graph_ids():
    """Map a complete-tree edge through its physical carrier provenance.

    ``parent_node_id``/``child_node_id`` in a tree candidate are graph-local
    ids.  The source ids identify the active carrier edge.  They are
    deliberately chosen so that treating graph ids as carrier vertices would
    select two different original nets and fail closed.
    """

    base = _base()
    baseline = _model(base)
    states = []
    actions = []
    for net_id, driver_pin, child_pin, action_id in (
        (0, 0, 1, 0),
        (1, 2, 3, 1),
    ):
        state = rebuffer_cpp.incremental_topology_from_steiner(
            steiner_payload(base, baseline),
            net_id,
            driver_pin,
            baseline.r_unit,
            baseline.c_unit,
        )
        view = rebuffer_cpp.incremental_topology_view(state)
        edge = next(item for item in view["edges"] if item["child"] == child_pin)
        source = view["coordinates"][edge["parent"]]
        target = view["coordinates"][edge["child"]]
        point = [
            source[axis] + 0.5 * (target[axis] - source[axis])
            for axis in (0, 1)
        ]
        rebuffer_cpp.apply_incremental_buffer_split(
            state,
            {
                "action_id": action_id,
                "parent_node_id": edge["parent"],
                "child_node_id": child_pin,
                "split_ratio": 0.5,
                "downstream_pin_ids": [child_pin],
                "x_dbu": point[0],
                "y_dbu": point[1],
                "buffer_master_name": "BUF2",
            },
        )
        states.append(state)
        actions.append(
            {
                "session_buffer_id": action_id,
                "buffer_master_name": "BUF2",
                "candidate_location_x_dbu": point[0],
                "candidate_location_y_dbu": point[1],
            }
        )

    payload = rebuffer_cpp.export_overlay_carrier(
        base,
        steiner_payload(base, baseline),
        states,
        actions,
    )

    # Partition 2 is the synthetic output net of state 0.  Graph ids 1 -> 2
    # belong to different physical states, while source 5 -> 1 is the active
    # state-0 output edge after the existing buffer.
    raw_action = {
        "net_id": 2,
        "driver_pin": 0,
        "downstream_pins": [1],
        "buffer_cell_index": 1,
        "parent_node_id": 1,
        "child_node_id": 2,
        "source_parent_node_id": 5,
        "source_child_node_id": 1,
        "x_dbu": 7500.0,
        "y_dbu": 0.0,
    }

    projected = dict(
        rebuffer_cpp.project_overlay_actions(
            base,
            payload,
            states,
            [raw_action],
        )[0]
    )

    assert projected["planning_partition_id"] == 2
    assert projected["planning_original_net_id"] == 0
    assert projected["resolved_state_net_id"] == 0
    assert projected["resolution_source"] == "native_source_provenance"
    assert projected["source_parent_node_id"] == 5
    assert projected["source_child_node_id"] == 1
    assert projected["resolved_parent_node_id"] == 3
    assert projected["resolved_child_node_id"] == 1
    assert projected["parent_node_id"] == 3
    assert projected["child_node_id"] == 1
    assert projected["net_id"] == 0
    assert projected["net_name"] == "n0"

    resolved = dict(projected)
    resolved.update(
        {
            "action_id": 99,
            "buffer_master_name": "BUF2",
            "downstream_pin_ids": projected["downstream_pins"],
        }
    )
    native = dict(rebuffer_cpp.resolve_overlay_split(states[0], resolved))
    assert native["parent_node_id"] == 3
    assert native["child_node_id"] == 1
    assert native["split_ratio"] == pytest.approx(0.5)


def test_projection_expands_source_child_to_active_branch_closure():
    """A source-child hint must select the active incoming branch edge.

    The current-tree candidate can name one descendant load while its source
    child is an active branch head.  Projecting that one load onto a later
    leaf edge changes the topology.  The native projector must use the full
    terminal closure of the source-child edge instead.
    """

    base = _base()
    state = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        10,
        {
            "edges": [
                {"parent": 0, "child": 1, "r": 1.0, "c": 0.1},
                {"parent": 1, "child": 2, "r": 1.0, "c": 0.1},
                {"parent": 2, "child": 3, "r": 1.0, "c": 0.1},
                {"parent": 2, "child": 4, "r": 1.0, "c": 0.1},
            ],
            "node_loads": {3: [3], 4: [1]},
            "coordinates": {
                0: [0.0, 0.0],
                1: [10.0, 0.0],
                2: [20.0, 0.0],
                3: [30.0, 0.0],
                4: [30.0, 10.0],
            },
        },
    )
    payload = {
        "original_net_by_net": [0],
        "state_node_by_vertex": list(range(10)),
        "state_net_by_vertex": [0] * 10,
    }
    payload["state_node_by_vertex"][8] = -1
    payload["state_net_by_vertex"][8] = -1
    raw_action = {
        "net_id": 0,
        "driver_pin": 0,
        "downstream_pins": [3],
        "buffer_cell_index": 1,
        "parent_node_id": 8,
        "child_node_id": 2,
        "source_parent_node_id": 8,
        "source_child_node_id": 2,
        "x_dbu": 15.0,
        "y_dbu": 0.0,
    }

    projected = dict(
        rebuffer_cpp.project_overlay_actions(
            base,
            payload,
            [state],
            [raw_action],
        )[0]
    )

    assert projected["downstream_pins"] == [1, 3]
    assert projected["source_closure_expanded"] is True
    assert projected["resolved_parent_node_id"] == 1
    assert projected["resolved_child_node_id"] == 2
    assert projected["parent_node_id"] == 1
    assert projected["child_node_id"] == 2


def test_projection_fallback_finds_unique_vertex_owner_in_other_active_state():
    base = _base()
    baseline = _model(base)
    steiner = steiner_payload(base, baseline)
    state0 = rebuffer_cpp.incremental_topology_from_steiner(
        steiner, 0, 0, baseline.r_unit, baseline.c_unit
    )
    state1 = rebuffer_cpp.incremental_topology_from_steiner(
        steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )

    payload = {
        "original_net_by_net": [0, 0],
        "state_node_by_vertex": [-1] * 10,
        "state_net_by_vertex": [-1] * 10,
    }
    raw_action = {
        "net_id": 1,
        "driver_pin": 2,
        "downstream_pins": [3],
        "buffer_cell_index": 1,
        "parent_node_id": 8,
        "child_node_id": 3,
        "source_parent_node_id": 2,
        "source_child_node_id": 3,
        "x_dbu": 55000.0,
        "y_dbu": 0.0,
    }

    projected = dict(
        rebuffer_cpp.project_overlay_actions(
            base,
            payload,
            [state0, state1],
            [raw_action],
        )[0]
    )

    assert projected["planning_partition_id"] == 1
    assert projected["planning_original_net_id"] == 0
    assert projected["resolved_state_net_id"] == 1
    assert projected["parent_node_id"] == 2
    assert projected["child_node_id"] == 3
    assert projected["downstream_pins"] == [3]
