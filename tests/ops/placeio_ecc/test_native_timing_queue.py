"""Project native stable slots onto the current placement carrier queue."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.flows.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.persistent_timing import _compact_current_pydb
from dreamplace.ops.placeio_ecc.repair_timing import plan_timing_repair
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_native_timing_campaign import _action, _session
from tests.ops.rebuffer.test_persistent_timing_queue import _reference_queue


def _export(model, overlay, bridge):
    payload = bridge.graph.export_carrier(
        steiner_payload(model.pydb, model), list(overlay.actions), model.r_unit, model.c_unit
    )
    physical = SimpleNamespace(**payload["physical"])
    carrier = CommittedTimingPyDB(model.pydb, physical)
    reference = RebufferTimingModel(
        carrier,
        use_pin_offset=False,
        r_unit=model.r_unit,
        c_unit=model.c_unit,
        steiner_view=payload["steiner"],
    )
    return physical, reference, payload["original_net_by_net"]


def _assert_queue(bridge, overlay, reference, original_map):
    actual = bridge.queue_state(overlay, reference.pydb)
    expected, _ = _reference_queue(reference, original_map)
    configurations = [
        {},
        {"native_tree_round_robin": True, "native_tree_round_robin_endpoint_window": 1},
    ]
    for config in configurations:
        left = rebuffer_cpp.select_timing_queue_items(actual, config)
        right = rebuffer_cpp.select_timing_queue_items(expected, config)
        assert [{k: v for k, v in row.items() if k != "slack_ps"} for row in left] == [
            {k: v for k, v in row.items() if k != "slack_ps"} for row in right
        ]
        np.testing.assert_allclose(
            [row["slack_ps"] for row in left],
            [row["slack_ps"] for row in right],
            rtol=0,
            atol=1e-6,
        )
    assert actual["snapshot_statistics"] == expected["snapshot_statistics"]
    return actual


def test_native_queue_projects_removed_stable_slots_and_keeps_prior_snapshot():
    model, session, bridge = _session()
    overlay = session.overlay
    for master in ("BUF1", "BUF2", "BUF1"):
        session.apply([_action(overlay, master=master)])
    bridge.evaluate(overlay)
    _, reference, original_map = _export(model, overlay, bridge)
    old = _assert_queue(bridge, overlay, reference, original_map)
    old_records = rebuffer_cpp.select_timing_queue_items(old, {})
    checkpoint = session.checkpoint()["checkpoint_id"]
    session.apply([{"action_kind": "buffer_remove", "session_buffer_id": 1}])
    bridge.evaluate(overlay)
    _, reference, original_map = _export(model, overlay, bridge)

    current = _assert_queue(bridge, overlay, reference, original_map)

    assert current["graph_queue_statistics"]["remapped_pin_count"] > 0
    assert current["graph_queue_statistics"]["remapped_net_count"] > 0
    assert rebuffer_cpp.select_timing_queue_items(old, {}) == old_records
    session.restore(checkpoint)
    _, reference, original_map = _export(model, overlay, bridge)
    _assert_queue(bridge, overlay, reference, original_map)


@pytest.mark.parametrize("net", [0, 1])
def test_original_and_session_drivers_project_to_fresh_carrier(net):
    model, session, bridge = _session()
    session.apply([_action(session.overlay, net=net)])
    bridge.evaluate(session.overlay)
    _, reference, original_map = _export(model, session.overlay, bridge)
    _assert_queue(bridge, session.overlay, reference, original_map)


def test_queue_uses_explicit_action_pin_ids_for_compact_slot_names():
    model, session, bridge = _session()
    overlay = session.overlay
    for master in ("BUF1", "BUF2"):
        session.apply([_action(overlay, master=master)])
    bridge.evaluate(overlay)
    exported = bridge.export(overlay)
    patch = rebuffer_cpp.build_incremental_timing_patch(
        model.pydb,
        overlay.native_topologies["n1"],
        overlay._native_action(overlay.actions[-1]),
        int(model.new_x.numel()),
        0,
    )
    compact = _compact_current_pydb(model.pydb, patch["graph"], list(overlay.actions), 0)
    reference = RebufferTimingModel(compact, use_pin_offset=False, steiner_view=exported["steiner"])

    assert compact.session_buffer_pin_ids == {0: (4, 5), 1: (6, 7)}
    _assert_queue(bridge, overlay, reference, exported["original_net_by_net"])

    compact.session_buffer_pin_ids[1] = compact.session_buffer_pin_ids[0]
    with pytest.raises(ValueError, match="overlap"):
        bridge.queue_state(overlay, compact)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("pin_names", "duplicate pin names"),
        ("net_names", "duplicate net names"),
        ("pin2net_map", "pin net partitions"),
        ("net2driver_pin_map", "net drivers"),
        ("node_master_names", "instance masters"),
    ],
)
def test_queue_projection_rejects_incompatible_carrier_without_graph_changes(field, message):
    model, session, bridge = _session()
    session.apply([_action(session.overlay)])
    bridge.evaluate(session.overlay)
    _, reference, _ = _export(model, session.overlay, bridge)
    target = SimpleNamespace(
        **{
            name: deepcopy(getattr(reference.pydb, name))
            for name in (
                "pin_names",
                "net_names",
                "pin2net_map",
                "net2driver_pin_map",
                "node_names",
                "node_master_names",
                "pin2node_map",
            )
        }
    )
    values = list(getattr(target, field))
    if field in {"pin_names", "net_names"}:
        values[-1] = values[0]
    elif field == "node_master_names":
        values[0] = "MISSING_MASTER"
    else:
        values[0] = 1
    setattr(target, field, values)
    before = bridge.graph.summary()

    with pytest.raises(ValueError, match=message):
        bridge.graph.timing_queue_state_for_carrier([0, 1], target)

    assert bridge.graph.summary() == before


def test_repair_queue_consumes_native_state_without_python_path_records(monkeypatch):
    model, session, bridge = _session()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native queue must read graph-owned paths and timing")

    monkeypatch.setattr(model, "native_setup_path_details", forbidden)
    monkeypatch.setattr(rebuffer_cpp, "build_native_timing_queue_snapshot", forbidden)
    result = plan_timing_repair(
        session,
        model,
        max_passes=1,
        config=resolve_repair_timing_config(
            {
                "use_native_timing_graph": True,
                "use_native_timing_queue": True,
                "use_native_timing_evaluation": False,
                "use_native_timing_planning": False,
            }
        ),
    )

    assert result["status"] == "ok"
    assert result["local_snapshot_statistics"]["native_graph_queue_count"] == 1
    assert bridge.queue_snapshot_count == 1
