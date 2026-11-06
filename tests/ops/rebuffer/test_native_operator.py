"""Regression tests for the C++ repair_timing campaign operator."""

from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _characterization, _liberty, _timing
from test_tree_dp import (
    _serial_buffer_pydb,
    _serial_buffer_topology,
    _unbuffered_tree,
)
from test_tree_search import _branches


def _run_tree_snapshot(snapshot, payload, checkpoint, apply, evaluate, restore, config):
    """Serve one current-tree payload per queue request through the strict operator."""

    def with_payloads():
        values = dict(snapshot())
        rows = []
        for item in values["items"]:
            for focus in item.get("focus_net_ids") or [item["root_net_id"]]:
                request = {"endpoint_pin": item["endpoint_pin"], "focus_net_id": focus}
                rows.append(
                    {
                        "endpoint_pin": item["endpoint_pin"],
                        "root_net_id": item["root_net_id"],
                        "net_id": focus,
                        "payload": payload(request),
                    }
                )
        values["planner_payloads"] = rows
        return values

    return rebuffer_cpp.run_repair_timing_native_snapshot_operator(
        with_payloads, checkpoint, apply, evaluate, restore, config
    )


def test_native_queue_sorts_endpoint_slack_and_walks_driver_net_queue():
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []
    requests = []

    def timing_snapshot():
        # Deliberately return the endpoint records out of priority order.
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 20,
                    "slack_ps": -2.0,
                    "path_pins": [20, 19],
                    "net_ids": [200],
                },
                {
                    "endpoint_pin": 10,
                    "slack_ps": -5.0,
                    "path_pins": [10, 9],
                    "net_ids": [100],
                },
            ],
        }

    def candidate_provider(request):
        requests.append((request["endpoint_pin"], request["net_id"]))
        if request["net_id"] == 100:
            return {"status": "no_actions", "net_id": 100}
        return {
            "status": "ok",
            "net_id": 200,
            "candidate_key": "net200",
            "actions": [{"action_id": 200}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {
            "status": "ok",
            "wns_ps": state["wns"],
            "tns_ps": state["tns"],
        }

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        timing_snapshot,
        candidate_provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {"max_passes": 2},
    )

    assert requests == [(10, 100), (20, 200)]
    assert result["core"] == "cpp"
    assert result["accepted_count"] == 1
    assert result["queue_coverage"]["visited_endpoint_count"] == 2
    assert state["actions"] == [{"action_id": 200}]


def test_native_queue_scopes_duplicate_candidates_to_endpoint_journal():
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "path_pins": [7, 6],
                    "net_ids": [40],
                },
                {
                    "endpoint_pin": 8,
                    "slack_ps": -3.0,
                    "root_net_id": 40,
                    "path_pins": [8, 6],
                    "net_ids": [40],
                },
            ],
        }

    def provider(request):
        return {
            "status": "ok",
            "net_id": 40,
            # The planner's key digests the action bundle, so different
            # endpoints produce different bundles and keys here.
            "candidate_key": f"bundle_{request['endpoint_pin']}",
            "actions": [{"action_id": request["endpoint_pin"]}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        action_id = actions[0]["action_id"]
        if action_id == 7:
            state["wns"] -= 1.0
            state["tns"] -= 1.0
        else:
            state["wns"] += 1.0
            state["tns"] += 1.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {"max_passes": 2},
    )

    assert [row["status"] for row in result["passes"]] == [
        "rejected",
        "accepted",
    ]
    assert result["accepted_count"] == 1
    assert state["actions"] == [{"action_id": 8}]


def test_native_queue_keeps_openroad_decreasing_slack_journal_until_improvement():
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []
    calls = {"count": 0}

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "path_pins": [7, 6],
                    "net_ids": [40],
                }
            ],
        }

    def provider(_request):
        calls["count"] += 1
        return {
            "status": "ok",
            "net_id": 40,
            "candidate_key": f"journal_{calls['count']}",
            "actions": [{"action_id": calls["count"]}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        action_id = actions[0]["action_id"]
        if action_id < 3:
            state["wns"] -= 1.0
            state["tns"] -= 1.0
        else:
            state["wns"] += 3.0
            state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {
            "max_passes": 3,
            "openroad_journal_keep_nonimproving": True,
            "allow_decreasing_slack": True,
            "decreasing_slack_max_passes": 2,
            "native_openroad_decreasing_slack_journal": True,
        },
    )

    assert [row["status"] for row in result["passes"]] == [
        "journal_kept",
        "journal_kept",
        "accepted",
    ]
    assert result["accepted_count"] == 3
    assert state["actions"] == [
        {"action_id": 1},
        {"action_id": 2},
        {"action_id": 3},
    ]
    assert state["wns"] == -9.0


def test_native_tree_queue_requests_one_root_candidate_and_keeps_branch_coverage():
    requests = []
    checkpoints = []
    state = {"wns": -10.0, "tns": -20.0, "actions": []}

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "driver_pin_id": 41,
                    "tree_key": "40:41",
                    "path_pins": [7, 6, 5],
                    "net_ids": [40, 42, 43],
                }
            ],
        }

    def provider(request):
        requests.append((request["root_net_id"], request["net_id"]))
        return {
            "status": "ok",
            "net_id": request["net_id"],
            "candidate_key": "tree40",
            "actions": [{"action_id": 40}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {"max_passes": 1, "native_tree_queue": True},
    )

    assert requests == [(40, 40)]
    assert result["accepted_count"] == 1
    assert result["queue_coverage"]["visited_branch_count"] == 3
    assert result["queue_coverage"]["visited_tree_count"] == 1
    assert state["actions"] == [{"action_id": 40}]


def test_native_tree_branch_queue_visits_each_focus_partition():
    requests = []
    checkpoints = []
    state = {"wns": -10.0, "tns": -20.0, "actions": []}

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "driver_pin_id": 41,
                    "tree_key": "40:41",
                    "path_pins": [7, 6, 5],
                    "net_ids": [40],
                    "focus_net_ids": [40, 42, 43],
                }
            ],
        }

    def provider(request):
        requests.append((request["root_net_id"], request["focus_net_id"]))
        if request["focus_net_id"] != 43:
            return {"status": "no_actions", "net_id": request["net_id"]}
        return {
            "status": "ok",
            "net_id": request["net_id"],
            "candidate_key": "focus43",
            "actions": [{"action_id": 43}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {
            "max_passes": 3,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_branch_candidate_variants": 1,
        },
    )

    assert requests == [(40, 40), (40, 42), (40, 43)]
    assert result["accepted_count"] == 1
    assert result["queue_coverage"]["visited_focus_net_ids"] == {40, 42, 43}
    assert state["actions"] == [{"action_id": 43}]


def test_native_tree_round_robin_interleaves_endpoints_before_tree_variants():
    requests = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "driver_pin_id": 41,
                    "tree_key": "40:41",
                    "path_pins": [7, 6, 5],
                    "net_ids": [40],
                    "focus_net_ids": [40, 42],
                },
                {
                    "endpoint_pin": 8,
                    "slack_ps": -3.0,
                    "root_net_id": 50,
                    "driver_pin_id": 51,
                    "tree_key": "50:51",
                    "path_pins": [8, 6, 5],
                    "net_ids": [50],
                    "focus_net_ids": [50, 52],
                },
            ],
        }

    def provider(request):
        requests.append((request["endpoint_pin"], request["focus_net_id"]))
        return {"status": "no_actions", "net_id": request["net_id"]}

    def evaluate():
        return {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0}

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        lambda _label: {"checkpoint_id": 0},
        lambda _actions: None,
        evaluate,
        lambda _checkpoint: None,
        {
            "max_passes": 4,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_candidate_variants": 1,
            "native_tree_branch_candidate_variants": 1,
            "native_tree_round_robin": True,
        },
    )

    assert requests == [(7, 40), (8, 50), (7, 42), (8, 52)]
    assert result["accepted_count"] == 0
    assert result["queue_coverage"]["visited_endpoint_count"] == 2


def test_native_tree_round_robin_exposes_root_variants_before_branches():
    requests = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "driver_pin_id": 41,
                    "tree_key": "40:41",
                    "path_pins": [7, 6, 5],
                    "net_ids": [40],
                    "focus_net_ids": [40, 42],
                }
            ],
        }

    def provider(request):
        requests.append((request["focus_net_id"], request["candidate_variant"]))
        return {"status": "no_actions", "net_id": request["net_id"]}

    def evaluate():
        return {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0}

    rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        lambda _label: {"checkpoint_id": 0},
        lambda _actions: None,
        evaluate,
        lambda _checkpoint: None,
        {
            "max_passes": 3,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_candidate_variants": 2,
            "native_tree_branch_candidate_variants": 1,
            "native_tree_round_robin": True,
            "native_tree_round_robin_endpoint_window": 1,
        },
    )

    assert requests == [(40, 0), (40, 1), (42, 0)]


def test_native_tree_branch_candidate_cannot_regress_global_tns():
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 7,
                    "slack_ps": -4.0,
                    "root_net_id": 40,
                    "driver_pin_id": 41,
                    "path_pins": [7, 6, 5],
                    "net_ids": [40],
                    "focus_net_ids": [42],
                }
            ],
        }

    def provider(_request):
        return {
            "status": "ok",
            "candidate_key": "tns_regression",
            "actions": [{"action_id": 42}],
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["tns"] -= 1.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = rebuffer_cpp.run_repair_timing_queue_operator(
        snapshot,
        provider,
        checkpoint,
        apply,
        evaluate,
        restore,
        {
            "max_passes": 1,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
        },
    )

    assert result["accepted_count"] == 0
    assert result["passes"][0]["status"] == "rejected"
    assert state["actions"] == []
    assert state["tns"] == -20.0


def test_native_tree_branch_planner_preserves_complete_cpp_vgdp_bundle():
    pydb = _serial_buffer_pydb()
    from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb

    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    native_tree = rebuffer_cpp.import_current_driver_tree(
        design, _serial_buffer_topology(), root_net_id=0
    )
    timing = _timing()
    timing.update(
        {
            "pin_slew_rise": [0.2] * 7,
            "pin_slew_fall": [0.2] * 7,
            "pin_load": [0.05] * 7,
            "pin_cap": [0.001] * 7,
            "pin_max_slew": [1000.0] * 7,
            "pin_max_cap": [1000.0] * 7,
            "pin_slack": [0.0, -0.2, -0.1, 0.0, -0.3, -0.4, -0.5],
        }
    )
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -0.2,
                    "root_net_id": 0,
                    "driver_pin_id": 0,
                    "path_pins": [0, 1],
                    "net_ids": [0],
                    "focus_net_ids": [2],
                }
            ],
        }

    def payload(request):
        return {
            "status": "ok",
            "net_id": 0,
            "net_name": "n0",
            "focus_net_id": request["focus_net_id"],
            "current_tree": native_tree,
            "design": design,
            "timing": timing,
            "liberty": _liberty(),
            "characterization": _characterization(),
            # Make the net-2 wire resistive enough for insertion to improve Q.
            "config": {"tree_enforce_electrical": False, "r_unit": 5000.0, "c_unit": 0.01},
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        # Focus selects a candidate containing net 2. Its upstream partitions
        # must survive so the physical action bundle matches the DP solution.
        assert {action["net_id"] for action in actions} == {0, 1, 2}
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(token):
        _label, wns, tns, actions = checkpoints[int(token)]
        state["wns"], state["tns"], state["actions"] = wns, tns, list(actions)

    result = _run_tree_snapshot(
        snapshot,
        payload,
        checkpoint,
        apply,
        evaluate,
        restore,
        {
            "max_passes": 8,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_candidate_variants": 8,
        },
    )

    assert result["status"] == "ok"
    assert result["accepted_count"] >= 1
    assert state["actions"]
    assert {action["net_id"] for action in state["actions"]} == {0, 1, 2}


def test_native_tree_planner_applies_openroad_timing_fanout_eligibility():
    design, timing, liberty, characterization, tree = _branches()
    result = _run_tree_snapshot(
        lambda: {
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -0.2,
                    "root_net_id": 0,
                    "driver_pin_id": 0,
                    "path_pins": [0, 1],
                    "net_ids": [0],
                }
            ]
        },
        lambda _request: {
            "status": "ok",
            "current_tree": tree,
            "design": design,
            "timing": timing,
            "liberty": liberty,
            "characterization": characterization,
            "config": {"tree_enforce_electrical": False},
        },
        lambda _label: 0,
        lambda _actions: None,
        lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
        lambda _token: None,
        {
            "max_passes": 1,
            "native_tree_queue": True,
            "enforce_timing_buffer_move_fanout": True,
            "timing_max_fanout": 2,
        },
    )
    assert result["passes"][0]["status"] == "unsupported"
    assert result["passes"][0]["reason"] == "timing_buffer_move_fanout"


def test_native_tree_planner_skips_openroad_top_level_output_port_nets():
    design, timing, liberty, characterization, tree = _branches()
    # Synthetic PyPlaceDB layout: the final node slot is the top-level output
    # terminal, and the current net's driver is that terminal pin.
    design["num_terminal_NIs"] = design["num_nodes"]
    result = _run_tree_snapshot(
        lambda: {
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -0.2,
                    "root_net_id": 0,
                    "driver_pin_id": 0,
                    "path_pins": [0, 1],
                    "net_ids": [0],
                }
            ]
        },
        lambda _request: {
            "status": "ok",
            "current_tree": tree,
            "design": design,
            "timing": timing,
            "liberty": liberty,
            "characterization": characterization,
            "config": {"tree_enforce_electrical": False},
        },
        lambda _label: 0,
        lambda _actions: None,
        lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
        lambda _token: None,
        {
            "max_passes": 1,
            "native_tree_queue": True,
            "native_skip_top_level_output_ports": True,
        },
    )
    assert result["passes"][0]["status"] == "unsupported"
    assert result["passes"][0]["reason"] == "timing_buffer_move_top_level_output_port"


def test_native_tree_branch_planner_falls_back_when_focus_has_no_action():
    design, _tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing.update(
        {
            "pin_slew_rise": [0.2] * 5,
            "pin_slew_fall": [0.2] * 5,
            "pin_load": [0.05] * 5,
            "pin_cap": [0.001] * 5,
            "pin_max_slew": [1000.0] * 5,
            "pin_max_cap": [1000.0] * 5,
            "pin_slack": [0.0, -0.2, -0.1, 0.0, 0.0],
        }
    )
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []
    requests = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -0.2,
                    "root_net_id": 0,
                    "driver_pin_id": 0,
                    "tree_key": "0:0",
                    "path_pins": [0, 1],
                    "net_ids": [0],
                    # There is no action on this synthetic focus partition.
                    "focus_net_ids": [99],
                }
            ],
        }

    def payload(request):
        requests.append(request["focus_net_id"])
        return {
            "status": "ok",
            "net_id": 0,
            "net_name": "n0",
            "focus_net_id": request["focus_net_id"],
            "current_tree": native_tree,
            "design": design,
            "timing": timing,
            "liberty": _liberty(),
            "characterization": _characterization(),
            "config": {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 0.01},
        }

    def checkpoint(label):
        token = len(checkpoints)
        checkpoints.append((label, state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(checkpoint_id):
        _label, wns, tns, actions = checkpoints[int(checkpoint_id)]
        state["wns"] = wns
        state["tns"] = tns
        state["actions"] = list(actions)

    result = _run_tree_snapshot(
        snapshot,
        payload,
        checkpoint,
        apply,
        evaluate,
        restore,
        {
            "max_passes": 1,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_branch_candidate_variants": 1,
        },
    )

    assert requests == [99]
    assert result["accepted_count"] == 1
    assert state["actions"]
    assert all(action["net_id"] == 0 for action in state["actions"])


def test_native_queue_plans_current_tree_vgdp_in_cpp_without_candidate_provider():
    design, _tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing.update(
        {
            "pin_slew_rise": [0.2] * 5,
            "pin_slew_fall": [0.2] * 5,
            "pin_load": [0.05] * 5,
            "pin_cap": [0.001] * 5,
            "pin_max_slew": [1000.0] * 5,
            "pin_max_cap": [1000.0] * 5,
            "pin_slack": [0.0, -0.2, -0.1, 0.0, 0.0],
        }
    )
    state = {"wns": -10.0, "tns": -20.0, "actions": []}
    checkpoints = []

    def snapshot():
        return {
            "status": "ok",
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -0.2,
                    "root_net_id": 0,
                    "driver_pin_id": 0,
                    "path_pins": [0, 1],
                    "net_ids": [0],
                }
            ],
        }

    def payload(_request):
        return {
            "status": "ok",
            "net_id": 0,
            "net_name": "n0",
            "current_tree": native_tree,
            "design": design,
            "timing": timing,
            "liberty": _liberty(),
            "characterization": _characterization(),
            "config": {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 0.01},
        }

    def checkpoint(_label):
        token = len(checkpoints)
        checkpoints.append((state["wns"], state["tns"], list(state["actions"])))
        return {"checkpoint_id": token}

    def apply(actions):
        state["wns"] += 2.0
        state["tns"] += 3.0
        state["actions"].extend(actions)

    def evaluate():
        return {"status": "ok", "wns_ps": state["wns"], "tns_ps": state["tns"]}

    def restore(token):
        wns, tns, actions = checkpoints[int(token)]
        state["wns"], state["tns"], state["actions"] = wns, tns, list(actions)

    result = _run_tree_snapshot(
        snapshot,
        payload,
        checkpoint,
        apply,
        evaluate,
        restore,
        {"max_passes": 1, "native_tree_queue": True},
    )

    assert result["status"] == "ok"
    assert result["native_candidate_planner"] is True
    assert result["accepted_count"] == 1
    assert state["actions"]
    assert "source_parent_node_id" in state["actions"][0]
