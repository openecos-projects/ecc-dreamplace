"""Strict queue transport must keep names, timing and carrier IDs consistent."""

from types import SimpleNamespace

import torch
from dreamplace.ops.placeio_ecc import electrical_observation
from dreamplace.ops.placeio_ecc import repair_timing as bridge
from dreamplace.ops.rebuffer import rebuffer_cpp


def test_strict_snapshot_reads_materialized_carrier_and_transports_named_slacks(monkeypatch):
    model = SimpleNamespace()

    def set_carrier(names, slack, endpoints):
        model.active_pydb = SimpleNamespace(pin_names=names)
        model.model = SimpleNamespace(end_points=torch.tensor(endpoints))
        model.slack = slack

    set_carrier(["other:D", "target:D"], [-10.0, -5.0], [0, 1])
    model.native_setup_path_details = lambda: {
        int(pin): {"path": [int(pin)]} for pin in model.model.end_points
    }
    model.native_setup_path_statistics = lambda: {}
    model.carrier_rebuild_statistics = lambda: {}
    captured = {}

    class Context:
        status = "ok"

        def native_queue_views(self):
            set_carrier(["new:A", "target:D", "other:D"], [0.0, -5.0, -10.0], [1, 2])
            return {
                "status": "ok",
                "design": {},
                "timing": {},
                "liberty": {},
                "original_net_by_net": [],
            }

        def native_path_context(self, path):
            return {"observed_path": path}

    def build_snapshot(_design, _timing, _liberty, records, _net_map):
        captured["records"] = records
        return {"status": "ok", "items": []}

    def run_snapshot(snapshot, _checkpoint, _apply, evaluate, _restore, _config):
        captured["before"] = evaluate()
        snapshot()
        set_carrier(["other:D", "new:A", "target:D"], [-10.0, 0.0, -4.0], [0, 2])
        captured["after"] = evaluate()
        return {"status": "ok"}

    monkeypatch.setattr(bridge, "_slack_vector", lambda active: active.slack)
    monkeypatch.setattr(
        electrical_observation, "electrical_observation", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(rebuffer_cpp, "build_native_timing_queue_snapshot", build_snapshot)
    monkeypatch.setattr(rebuffer_cpp, "select_timing_queue_items", lambda snapshot, config: [])
    monkeypatch.setattr(rebuffer_cpp, "run_repair_timing_native_snapshot_operator", run_snapshot)
    session = SimpleNamespace(
        state="open",
        evaluate=lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
        export_actions=lambda: [],
    )

    bridge.plan_timing_repair(session, model, max_passes=1, config={}, context=Context())

    assert captured["before"]["endpoint_slacks_by_name"] == {"other:D": -10.0, "target:D": -5.0}
    assert captured["after"]["endpoint_slacks_by_name"] == {"other:D": -10.0, "target:D": -4.0}
    assert captured["records"] == [
        {
            "endpoint_pin": 1,
            "endpoint_pin_name": "target:D",
            "slack_ps": -5.0,
            "path_pins": [1],
            "path_timing": {"observed_path": [1]},
        },
        {
            "endpoint_pin": 2,
            "endpoint_pin_name": "other:D",
            "slack_ps": -10.0,
            "path_pins": [2],
            "path_timing": {"observed_path": [2]},
        },
    ]
