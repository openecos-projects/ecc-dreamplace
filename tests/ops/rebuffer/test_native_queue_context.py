"""Native endpoint identity, window coverage and refresh invariants."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _characterization, _liberty, _timing
from test_tree_dp import _unbuffered_tree


def _item(endpoint, root, slack):
    return {
        "endpoint_pin": endpoint,
        "slack_ps": slack,
        "root_net_id": root,
        "net_ids": [root],
    }


def test_endpoint_window_covers_all_drivers_and_matches_native_requests():
    items = [
        _item(30, 5, -2.0),
        _item(10, 0, -10.0),
        _item(10, 1, -10.0),
        _item(20, 3, -5.0),
        _item(10, 2, -10.0),
        _item(20, 4, -5.0),
    ]
    snapshot = {"status": "ok", "items": items}
    config = {
        "max_passes": 20,
        "native_tree_queue": True,
        "native_tree_round_robin": True,
        "native_tree_round_robin_endpoint_window": 2,
    }
    selected = rebuffer_cpp.select_timing_queue_items(snapshot, config)
    requests = []

    def provider(request):
        requests.append(request)
        return {"status": "no_actions"}

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        lambda: snapshot,
        provider,
        lambda _label: None,
        lambda _actions: None,
        lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
        lambda _token: None,
        config,
    )

    assert selected == [items[index] for index in [1, 2, 4, 3, 5]]
    assert [request["root_net_id"] for request in requests] == [0, 1, 2, 3, 4]
    assert result["queue_coverage"]["visited_endpoint_pins"] == {10, 20}
    assert result["stop_reason"] == "queue_exhausted"
    assert rebuffer_cpp.select_timing_queue_items(snapshot, {}) == [
        items[index] for index in [1, 2, 4, 3, 5, 0]
    ]


def test_retired_endpoints_release_their_window_slot():
    # Endpoint 10 spends its one-attempt budget; the window must then admit
    # endpoint 30 instead of stopping with the window still full.
    items = [_item(10, 0, -10.0), _item(20, 1, -5.0), _item(30, 2, -2.0)]
    snapshot = {"status": "ok", "items": items}
    config = {
        "max_passes": 20,
        "native_tree_queue": True,
        "native_tree_round_robin": True,
        "native_tree_round_robin_endpoint_window": 2,
        "native_tree_max_attempts_per_endpoint": 1,
    }
    state = {"wns": -10.0, "tns": -20.0}
    requests = []

    def provider(request):
        requests.append(request["endpoint_pin"])
        return {"status": "ok", "actions": [{"net_id": request["net_id"]}]}

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def apply(_actions):
        # Only endpoint 20 improves.
        if requests[-1] == 20:
            state["tns"] += 1.0
        else:
            state["wns"] -= 1.0

    def restore(_token):
        state["wns"] = -10.0

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        lambda: snapshot, provider, lambda _label: 0, apply, evaluate, restore, config
    )

    assert requests == [10, 20, 30]
    assert result["queue_coverage"]["visited_endpoint_pins"] == {10, 20, 30}
    # Published before the last epoch: 10 and 20 had spent their budget.
    assert config["retired_endpoints"] == ["10", "20"]
    assert rebuffer_cpp.select_timing_queue_items(snapshot, config) == [items[2]]


def _strict_endpoint_run(after_slacks):
    design, _tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing["pin_slack"] = [0.0, -0.2, -0.1, 0.0, 0.0]
    payload = {
        "status": "ok",
        "net_id": 0,
        "current_tree": native_tree,
        "design": design,
        "timing": timing,
        "liberty": _liberty(),
        "characterization": _characterization(),
        "config": {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 0.01},
    }
    state = {"applied": False, "restores": 0}

    def snapshot():
        item = {**_item(1, 0, -5.0), "endpoint_pin_name": "target:D"}
        return {
            "status": "ok",
            "items": [item],
            "planner_payloads": [
                {"endpoint_pin": 1, "root_net_id": 0, "net_id": 0, "payload": payload}
            ],
        }

    def apply(actions):
        assert actions
        state["applied"] = True

    def evaluate():
        return {
            "status": "ok",
            "wns_ps": -10.0,
            "tns_ps": -20.0,
            # A stale/global scalar must not override the named target.
            "endpoint_slack_ps": -10.0,
            "endpoint_slacks_by_name": (
                after_slacks if state["applied"] else {"other:D": -10.0, "target:D": -5.0}
            ),
        }

    def restore(_token):
        state["applied"] = False
        state["restores"] += 1

    result = rebuffer_cpp.run_repair_timing_native_snapshot_operator(
        snapshot,
        lambda _label: None,
        apply,
        evaluate,
        restore,
        {"max_passes": 1, "acceptance": "endpoint_journal", "allow_decreasing_slack": True},
    )
    return result, state


def test_strict_snapshot_accepts_target_gain_when_global_metrics_are_constant():
    result, state = _strict_endpoint_run({"other:D": -10.0, "target:D": -4.0})

    assert state == {"applied": True, "restores": 0}
    assert (result["planner_payload_hits"], result["planner_payload_misses"]) == (1, 0)
    row = result["passes"][0]
    assert (row["status"], row["endpoint_pin_name"], row["endpoint_slack_after_ps"]) == (
        "accepted",
        "target:D",
        -4.0,
    )
    assert result["final"]["endpoint_slack_ps"] == -4.0


@pytest.mark.parametrize(
    ("after_slacks", "reason"),
    [
        ({"other:D": -10.0}, "target endpoint missing"),
        ({"target:D": float("nan")}, "target endpoint slack is not finite"),
    ],
)
def test_unavailable_target_restores_candidate_instead_of_using_wns(after_slacks, reason):
    result, state = _strict_endpoint_run(after_slacks)

    assert state == {"applied": False, "restores": 1}
    assert result["accepted_count"] == 0
    assert result["passes"][0]["status"] == "evaluate_failed"
    assert reason in result["passes"][0]["reason"]


def test_accepted_mutation_refreshes_endpoint_ids_paths_and_slacks():
    state = {"step": 0}
    snapshots, requests = [], []

    def snapshot():
        step = state["step"]
        snapshots.append(step)
        return {
            "status": "ok",
            "items": [
                {
                    **_item(1 + 6 * step, 10 + step, -5.0 + step),
                    "endpoint_pin_name": "target:D",
                    "path_pins": [2 + step, 1 + 6 * step],
                }
            ],
        }

    def provider(request):
        requests.append(dict(request))
        return {"status": "ok", "actions": [{"step": state["step"]}]}

    def apply(_actions):
        state["step"] += 1

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        lambda _label: state["step"],
        apply,
        lambda: {
            "status": "ok",
            "wns_ps": -10.0,
            "tns_ps": -20.0,
            "endpoint_slacks_by_name": {"target:D": -5.0 + state["step"]},
        },
        lambda step: state.update(step=step),
        {"max_passes": 2, "acceptance": "endpoint_journal", "allow_decreasing_slack": True},
    )

    assert snapshots == [0, 1]
    assert [
        (r["endpoint_pin"], r["endpoint_pin_name"], r["endpoint_slack_ps"], r["path_pins"])
        for r in requests
    ] == [(1, "target:D", -5.0, [2, 1]), (7, "target:D", -4.0, [3, 7])]
    assert [row["endpoint_slack_after_ps"] for row in result["passes"]] == [-4.0, -3.0]
    assert result["accepted_count"] == 2

