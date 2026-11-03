"""Strict queue transport must keep names, timing and carrier IDs consistent."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.placeio_ecc import electrical_observation
from dreamplace.ops.placeio_ecc import repair_timing as bridge
from dreamplace.ops.rebuffer import rebuffer_cpp


def test_incremental_delta_snapshot_hint_avoids_full_timing_arrays():
    previous = {
        "pin_names": ("a", "b", "c"),
        "pin2net": np.asarray([0, 0, 1], dtype=np.int64),
        "net_drivers": np.asarray([0, 2], dtype=np.int64),
        "arrays": {"pin_rAAT_live_snapshot": np.zeros(3)},
        "net_pins": {0: (0, 1), 1: (2,)},
        "delta_pins": None,
    }
    current = {
        "pin_names": previous["pin_names"],
        "pin2net": previous["pin2net"].copy(),
        "net_drivers": previous["net_drivers"].copy(),
        "arrays": None,
        "net_pins": None,
        "delta_pins": frozenset({1}),
    }

    assert bridge._changed_snapshot_pins(previous, current) == {1}


@pytest.mark.parametrize("delta_hint", [None, frozenset()])
def test_driver_change_invalidates_its_net_pins(delta_hint):
    previous = {
        "pin_names": ("p0", "p1", "p2", "p3", "p4"),
        "pin2net": np.asarray([0, 0, 1, 1, 1]),
        "net_drivers": np.asarray([0, 2]),
        "net_pins": {0: (0, 1), 1: (2, 3, 4)},
        "arrays": {"pin_rAAT_live_snapshot": np.zeros(5)},
        "delta_pins": None,
    }
    current = {**previous, "net_drivers": np.asarray([0, 3]), "delta_pins": delta_hint}

    assert bridge._changed_snapshot_pins(previous, current) == {2, 3, 4}


def test_timing_snapshot_state_prefers_incremental_delta_hint():
    timing_model = SimpleNamespace(
        model=SimpleNamespace(last_incremental_changed_pins=torch.tensor([1])),
        active_pydb=SimpleNamespace(
            pin_names=["a", "b"],
            pin2net_map=[0, 1],
            net2driver_pin_map=[0, 1],
        ),
    )

    state = bridge._timing_snapshot_state(timing_model)

    assert state["arrays"] is None
    assert state["net_pins"] is None
    assert state["delta_pins"] == frozenset({1})


def test_snapshot_oracle_checks_item_identity_and_window_order():
    items = [
        {
            "endpoint_pin": 7,
            "endpoint_pin_name": "u:D",
            "slack_ps": -3.5,
            "path_pins": [1, 4, 7],
            "root_net_id": 2,
            "driver_pin_id": 1,
            "net_ids": [2],
            "tree_key": "2:1",
        },
    ]
    actual = {"items": items}
    reference = {"items": [dict(items[0])]}

    def select(snapshot, _config):
        return snapshot["items"]

    bridge._assert_queue_snapshot_oracle(actual, reference, select, {})

    reference["items"][0]["root_net_id"] = 3
    with pytest.raises(RuntimeError, match="root_net_id"):
        bridge._assert_queue_snapshot_oracle(actual, reference, select, {})


def test_gradient_scores_reduce_pin_gradient_to_net_and_driver():
    from dreamplace.ops.placeio_ecc.gradient_order import scores_from_position_gradient

    pydb = SimpleNamespace(
        pin2net_map=np.asarray([0, 0, 1], dtype=np.int64),
        net2driver_pin_map=np.asarray([0, 2], dtype=np.int64),
        net_names=["n0", "n1"],
    )
    scores = scores_from_position_gradient(
        torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 5.0]), pydb
    )

    assert scores["net_scores"] == {0: 4.0, 1: 5.0}
    assert scores["driver_scores"] == {0: 4.0, 2: 5.0}


def test_gradient_ranking_oracle_compares_native_order_and_score_order():
    from dreamplace.ops.placeio_ecc.gradient_order import compare_driver_rankings

    result = compare_driver_rankings(
        [2, 1, 0],
        {2: 1.0, 1: 3.0, 0: 2.0},
    )

    assert result["status"] == "ok"
    assert result["scored_order"] == [1, 0, 2]
    assert result["native_top_driver"] == 2
    assert result["score_top_driver"] == 1
    assert result["top1_match"] is False
    assert result["pairwise_comparisons"] == 3
    assert result["pairwise_agreements"] == 1
    assert result["pairwise_agreement"] == pytest.approx(1.0 / 3.0)


def test_gradient_path_oracle_calls_native_driver_policy():
    from dreamplace.ops.placeio_ecc.gradient_order import (
        compare_path_driver_ranking,
    )
    from test_repair_design import _design, _liberty, _timing

    context = SimpleNamespace(
        native_queue_views=lambda: {
            "status": "ok",
            "design": _design(),
            "timing": _timing(),
            "liberty": _liberty(),
        },
        native_path_context=lambda _path, path_details=None: {},
        timing_gradient_scores=lambda: {
            "status": "ok",
            "driver_scores": {0: 1.0},
        },
    )

    result = compare_path_driver_ranking(context, [1, 0])

    assert result["status"] == "ok"
    assert result["native_order"] == [0]
    assert result["scored_order"] == [0]
    assert result["top1_match"] is True


def test_native_queue_gradient_ordering_is_opt_in_and_native():
    snapshot = {
        "items": [
            {
                "endpoint_pin": 1,
                "endpoint_pin_name": "slow:D",
                "slack_ps": -20.0,
                "path_pins": [0, 1],
                "root_net_id": 3,
                "driver_pin_id": 0,
                "net_ids": [3],
                "tree_key": "3:0",
                "gradient_score": 1.0,
            },
            {
                "endpoint_pin": 2,
                "endpoint_pin_name": "critical:D",
                "slack_ps": -1.0,
                "path_pins": [4, 2],
                "root_net_id": 7,
                "driver_pin_id": 4,
                "net_ids": [7],
                "tree_key": "7:4",
                "gradient_score": 9.0,
            },
        ]
    }
    default_items = rebuffer_cpp.select_timing_queue_items(
        snapshot, {"native_tree_round_robin": False}
    )
    gradient_items = rebuffer_cpp.select_timing_queue_items(
        snapshot,
        {"native_tree_round_robin": False, "gradient_ordering": True},
    )
    constrained_items = rebuffer_cpp.select_timing_queue_items(
        snapshot,
        {
            "native_tree_round_robin": True,
            "native_tree_round_robin_endpoint_window": 1,
            "gradient_ordering": True,
            "gradient_window_tiebreak": True,
        },
    )
    assert [item["root_net_id"] for item in default_items] == [3, 7]
    assert [item["root_net_id"] for item in gradient_items] == [7, 3]
    assert [item["root_net_id"] for item in constrained_items] == [3]


def test_gradient_ordering_preserves_native_driver_order_within_endpoint():
    snapshot = {
        "items": [
            {
                "endpoint_pin": 1,
                "slack_ps": -20.0,
                "root_net_id": 3,
                "driver_pin_id": 0,
                "net_ids": [3],
                "gradient_score": 1.0,
            },
            {
                "endpoint_pin": 1,
                "slack_ps": -20.0,
                "root_net_id": 7,
                "driver_pin_id": 4,
                "net_ids": [7],
                "gradient_score": 9.0,
            },
            {
                "endpoint_pin": 2,
                "slack_ps": -1.0,
                "root_net_id": 11,
                "driver_pin_id": 8,
                "net_ids": [11],
                "gradient_score": 10.0,
            },
        ]
    }

    for config in (
        {"gradient_ordering": True},
        {
            "native_tree_round_robin": True,
            "native_tree_round_robin_endpoint_window": 2,
            "gradient_ordering": True,
            "gradient_window_tiebreak": True,
        },
    ):
        selected = rebuffer_cpp.select_timing_queue_items(snapshot, config)
        assert [item["root_net_id"] for item in selected] == [11, 3, 7]


def test_gradient_score_map_is_transported_to_native_queue_records(monkeypatch):
    model = SimpleNamespace(
        active_pydb=SimpleNamespace(
            pin_names=["u:D"],
            pin2net_map=[0],
            net2driver_pin_map=[0],
        ),
        model=SimpleNamespace(end_points=torch.tensor([0])),
    )
    model.native_setup_path_details = lambda: {0: {"path": [0]}}
    model.native_setup_path_statistics = lambda: {}
    model.carrier_rebuild_statistics = lambda: {}
    captured = {}

    class Context:
        status = "ok"

        def native_queue_views(self):
            return {
                "status": "ok",
                "design": {},
                "timing": {},
                "liberty": {},
                "original_net_by_net": [],
            }

        def native_path_context(self, path, path_details=None):
            return {"path": list(path)}

        def timing_gradient_scores(self):
            return {"status": "ok", "loss": 1.0, "net_scores": {0: 7.5}}

    def build_snapshot(_design, _timing, _liberty, records, _net_map):
        captured["records"] = [dict(record) for record in records]
        return {"status": "ok", "items": []}

    def run_snapshot(snapshot, _checkpoint, _apply, _evaluate, _restore, _config):
        snapshot()
        return {"status": "ok"}

    monkeypatch.setattr(bridge, "_slack_vector", lambda _active: np.asarray([-2.0]))
    monkeypatch.setattr(
        electrical_observation, "electrical_observation", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(rebuffer_cpp, "build_native_timing_queue_snapshot", build_snapshot)
    monkeypatch.setattr(rebuffer_cpp, "select_timing_queue_items", lambda *_args: [])
    monkeypatch.setattr(
        rebuffer_cpp, "run_repair_timing_native_snapshot_operator", run_snapshot
    )
    session = SimpleNamespace(
        state="open",
        evaluate=lambda: {"status": "ok", "wns_ps": -2.0, "tns_ps": -2.0},
        export_actions=lambda: [],
    )

    result = bridge.plan_timing_repair(
        session,
        model,
        max_passes=1,
        config={"repair_gradient_ordering": True},
        context=Context(),
    )

    assert captured["records"][0]["gradient_score_by_net"] == {0: 7.5}
    assert result["local_snapshot_statistics"]["gradient_score_status"] == "ok"


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


@pytest.mark.parametrize("context_reuse", [False, True])
def test_local_snapshot_reuses_unchanged_endpoint_record(monkeypatch, context_reuse):
    model = SimpleNamespace()
    model.active_pydb = SimpleNamespace(
        pin_names=["first:D", "second:D"],
        pin2net_map=[0, 1],
        net2driver_pin_map=[0, 1],
        flat_net2pin_map=[0, 1],
        flat_net2pin_start_map=[0, 1, 2],
    )
    model.model = SimpleNamespace(end_points=torch.tensor([0, 1]))
    model.slack = [-10.0, -5.0]
    requested = []

    def path_details(endpoint_pins=None):
        requested.append(None if endpoint_pins is None else tuple(sorted(endpoint_pins)))
        endpoints = model.model.end_points.tolist() if endpoint_pins is None else endpoint_pins
        return {int(pin): {"path": [int(pin)]} for pin in endpoints}

    model.native_setup_path_details = path_details
    model.native_setup_path_statistics = lambda: {}
    model.carrier_rebuild_statistics = lambda: {}
    captured = {"records": [], "build_calls": 0, "update_calls": 0}

    class Context:
        status = "ok"

        def native_queue_views(self):
            return {
                "status": "ok",
                "design": {},
                "timing": {},
                "liberty": {},
                "original_net_by_net": [],
            }

        def native_path_context(self, path, path_details=None):
            return {"observed_path": path}

    def build_snapshot(_design, _timing, _liberty, records, _net_map):
        captured["build_calls"] += 1
        captured["records"].append([dict(record) for record in records])
        return {"status": "ok", "items": []}

    def update_snapshot(
        _design,
        _timing,
        _liberty,
        _previous,
        changed_records,
        _active_endpoints,
        _changed_endpoints,
        _net_map,
    ):
        captured["update_calls"] += 1
        captured["changed_records"] = [dict(record) for record in changed_records]
        return {
            "status": "ok",
            "items": [],
            "snapshot_statistics": {"item_count": 0},
        }

    def run_snapshot(snapshot, _checkpoint, _apply, evaluate, _restore, _config):
        snapshot()
        model.slack[1] = -4.0
        snapshot()
        if context_reuse:
            snapshot()
        return {"status": "ok"}

    monkeypatch.setattr(bridge, "_slack_vector", lambda active: active.slack)
    monkeypatch.setattr(
        electrical_observation, "electrical_observation", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(rebuffer_cpp, "build_native_timing_queue_snapshot", build_snapshot)
    monkeypatch.setattr(
        rebuffer_cpp, "update_native_timing_queue_snapshot", update_snapshot
    )
    monkeypatch.setattr(rebuffer_cpp, "select_timing_queue_items", lambda snapshot, config: [])
    monkeypatch.setattr(rebuffer_cpp, "run_repair_timing_native_snapshot_operator", run_snapshot)
    session = SimpleNamespace(
        state="open",
        evaluate=lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -15.0},
        export_actions=lambda: [],
    )

    monkeypatch.setenv("REBUFFER_LOCAL_SNAPSHOT", "1")
    monkeypatch.setenv("REBUFFER_SNAPSHOT_ORACLE", "1")
    monkeypatch.setenv("REBUFFER_NATIVE_LOCAL_SNAPSHOT_UPDATE", "1")
    if context_reuse:
        monkeypatch.setenv("REBUFFER_NATIVE_QUEUE_CONTEXT_REUSE", "1")
    result = bridge.plan_timing_repair(
        session, model, max_passes=1, config={}, context=Context()
    )

    assert requested == [(0, 1), (1,)]
    assert captured["records"][0][0]["slack_ps"] == -10.0
    expected_records = [
        {
            "endpoint_pin": 1,
            "endpoint_pin_name": "second:D",
            "slack_ps": -4.0,
            "path_pins": [1],
            "path_timing": {"observed_path": [1]},
        }
    ]
    if context_reuse:
        expected_records.insert(0, captured["records"][0][0])
        assert result["local_snapshot_statistics"]["reused_record_counts"] == [0, 1, 2]
    assert captured["changed_records"] == expected_records
    assert result["local_snapshot_statistics"]["oracle_check_count"] == (3 if context_reuse else 2)

    captured["build_calls"] = 0
    captured["update_calls"] = 0
    model.slack[1] = -5.0
    monkeypatch.delenv("REBUFFER_NATIVE_LOCAL_SNAPSHOT_UPDATE")
    monkeypatch.delenv("REBUFFER_NATIVE_QUEUE_CONTEXT_REUSE", raising=False)
    result = bridge.plan_timing_repair(
        session, model, max_passes=1, config={}, context=Context()
    )

    assert captured["update_calls"] == 0
    assert captured["build_calls"] == (6 if context_reuse else 4)
    assert result["local_snapshot_statistics"]["oracle_check_count"] == (3 if context_reuse else 2)
