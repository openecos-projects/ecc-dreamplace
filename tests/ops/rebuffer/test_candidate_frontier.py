"""TP chooses complete native tree alternatives from a shared checkpoint."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp as native


def _run(observations, *, top_k, config=None):
    baseline = {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0, "endpoint_slack_ps": -5.0}
    state = dict(baseline)
    applied = []
    checkpoints = []
    restores = []

    def checkpoint(_label):
        checkpoints.append(dict(state))
        return {"checkpoint_id": len(checkpoints) - 1}

    def apply(actions):
        # A trial must see the source state, never another alternative's TP.
        assert state == baseline
        rank = actions[0]["rank"]
        applied.append(rank)
        outcome = observations[rank]
        if isinstance(outcome, Exception):
            state.update(wns_ps=-100.0)
            raise outcome
        state.update(outcome)

    def restore(token):
        restores.append(token)
        state.clear()
        state.update(checkpoints[token])

    bundles = [[{"net_id": 0, "rank": rank}] for rank in range(len(observations))]
    result = native.run_repair_timing_queue_operator(
        lambda: {
            "status": "ok",
            "items": [{"endpoint_pin": 1, "slack_ps": -5.0, "root_net_id": 0, "net_ids": [0]}],
        },
        lambda _request: {"status": "ok", "actions": bundles[0], "candidate_frontier": bundles},
        checkpoint,
        apply,
        lambda: dict(state),
        restore,
        {"max_passes": 1, "python_timing_candidate_top_k": top_k, **(config or {})},
    )
    return result, state, applied, restores


def test_tp_selects_later_acceptable_candidate_and_reapplies_it_after_rejected_trial():
    result, state, applied, restores = _run(
        [
            {"wns_ps": -9.0, "tns_ps": -19.0},
            {"wns_ps": -8.0, "tns_ps": -18.0},
            {"wns_ps": -7.0, "tns_ps": -25.0},
        ],
        top_k=3,
    )
    assert result["accepted_actions"] == [{"net_id": 0, "rank": 1}]
    assert (applied, restores) == ([0, 1, 2, 1], [0, 0, 0])
    assert (
        state
        == result["final"]
        == {
            "status": "ok",
            "wns_ps": -8.0,
            "tns_ps": -18.0,
            "endpoint_slack_ps": -5.0,
        }
    )
    assert result["evaluation_calls"] == 5  # baseline, three trials, selected reapply
    assert result["passes"][0]["selected_frontier_rank"] == 1
    assert [trial["acceptable"] for trial in result["passes"][0]["frontier_trials"]] == [
        True,
        True,
        False,
    ]


def test_failed_partial_trial_restores_before_later_candidate():
    result, state, applied, restores = _run(
        [
            ValueError("fixture partial mutation"),
            {"wns_ps": -8.0, "tns_ps": -18.0},
        ],
        top_k=2,
    )
    assert (applied, restores) == ([0, 1], [0])
    assert result["accepted_actions"] == [{"net_id": 0, "rank": 1}]
    assert result["passes"][0]["frontier_trials"][0]["status"] == "evaluate_failed"
    assert state["wns_ps"] == -8.0


@pytest.mark.parametrize("top_k", [1, 2])
def test_all_rejected_frontier_restores_accepted_source_state(top_k):
    result, state, applied, restores = _run(
        [
            {"wns_ps": -11.0, "tns_ps": -19.0},
            {"wns_ps": -12.0, "tns_ps": -18.0},
        ],
        top_k=top_k,
    )
    assert result["accepted_actions"] == []
    assert state == result["baseline"] == result["final"]
    assert result["passes"][0]["status"] == "rejected"
    assert len(result["passes"][0]["frontier_trials"]) == top_k
    assert applied[0] == 0
    assert restores


@pytest.mark.parametrize(
    "outcome",
    [
        {"wns_ps": float("nan"), "tns_ps": -19.0},
        {"status": "unsupported", "wns_ps": -9.0, "tns_ps": -19.0},
    ],
)
def test_invalid_tp_observation_is_restored_before_selecting_valid_candidate(outcome):
    result, state, applied, restores = _run(
        [
            outcome,
            {"wns_ps": -8.0, "tns_ps": -18.0},
        ],
        top_k=2,
    )
    assert (applied, restores) == ([0, 1], [0])
    assert result["accepted_actions"] == [{"net_id": 0, "rank": 1}]
    assert result["evaluation_calls"] == 3
    assert state["wns_ps"] == -8.0


def test_failure_after_valid_trial_restores_and_reapplies_selected_bundle():
    result, state, applied, restores = _run(
        [
            {"wns_ps": -8.0, "tns_ps": -18.0},
            ValueError("partial mutation"),
        ],
        top_k=2,
    )
    assert (applied, restores) == ([0, 1, 0], [0, 0])
    assert result["accepted_actions"] == [{"net_id": 0, "rank": 0}]
    assert state["wns_ps"] == -8.0


def test_all_invalid_trials_preserve_tp_call_count_and_failure_ledger():
    result, state, applied, restores = _run(
        [
            {"wns_ps": float("nan"), "tns_ps": -19.0},
            {"wns_ps": -9.0, "tns_ps": float("inf")},
        ],
        top_k=2,
    )
    assert result["evaluation_calls"] == 3
    assert result["accepted_actions"] == []
    assert state == result["baseline"] == result["final"]
    assert (applied, restores) == ([0, 1], [0, 0])
    assert [trial["status"] for trial in result["passes"][0]["frontier_trials"]] == [
        "evaluate_failed",
        "evaluate_failed",
    ]


def test_restore_failure_aborts_before_another_alternative_is_applied():
    applied = []

    def restore(_token):
        raise RuntimeError("fixture restore failed")

    with pytest.raises(RuntimeError, match="fixture restore failed"):
        native.run_repair_timing_queue_operator(
            lambda: {"items": [{"endpoint_pin": 1, "slack_ps": -5.0, "net_ids": [0]}]},
            lambda _request: {
                "actions": [{"rank": 0}],
                "candidate_frontier": [[{"rank": 0}], [{"rank": 1}]],
            },
            lambda _label: 0,
            lambda actions: applied.append(actions[0]["rank"]),
            lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
            restore,
            {"max_passes": 1, "python_timing_candidate_top_k": 2},
        )
    assert applied == [0]


@pytest.mark.parametrize("round_robin", [False, True])
def test_variant_chunks_do_not_overlap_frontier_slots(round_robin):
    requested = []

    def provider(request):
        requested.append((request["focus_net_id"], request["candidate_variant"]))
        return {"status": "no_actions"}

    result = native.run_repair_timing_queue_operator(
        lambda: {
            "items": [
                {
                    "endpoint_pin": 1,
                    "slack_ps": -5.0,
                    "root_net_id": 0,
                    "net_ids": [0],
                    "focus_net_ids": [0, 2],
                }
            ]
        },
        provider,
        lambda _label: None,
        lambda _actions: None,
        lambda: {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0},
        lambda _token: None,
        {
            "max_passes": 20,
            "native_tree_queue": True,
            "native_tree_branch_queue": True,
            "native_tree_round_robin": round_robin,
            "native_tree_candidate_variants": 8,
            "native_tree_branch_candidate_variants": 4,
            "python_timing_candidate_top_k": 3,
        },
    )
    assert sorted(requested) == [(0, 0), (0, 3), (0, 6), (2, 0), (2, 3)]
    assert result["attempt_count"] == 5


@pytest.mark.parametrize(
    ("budget", "applied_ranks", "evaluations"),
    [
        (1, [], 1),
        (2, [0], 2),
        (3, [0], 2),
        (4, [0, 1, 0], 4),
    ],
)
def test_tp_budget_reserves_selected_bundle_reapplication(budget, applied_ranks, evaluations):
    result, state, applied, _restores = _run(
        [
            {"wns_ps": -8.0, "tns_ps": -18.0},
            {"wns_ps": -9.0, "tns_ps": -19.0},
            {"wns_ps": -7.0, "tns_ps": -17.0},
        ],
        top_k=3,
        config={"max_tp_evaluations": budget},
    )
    assert applied == applied_ranks
    assert result["evaluation_calls"] == evaluations <= budget
    assert state == result["final"]
    if applied:
        assert result["accepted_actions"] == [{"net_id": 0, "rank": 0}]
    else:
        assert result["stop_reason"] == "max_tp_evaluations"
