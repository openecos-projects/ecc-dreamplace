"""LEGACY endpoint/pass behavior is independent of the candidate generator."""

from copy import deepcopy

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp


class Campaign:
    def __init__(self, proposals, **config):
        self.config = {"max_passes": 20, "legacy_max_passes_per_endpoint": 2, **config}
        self.slacks = [-10.0, -8.0, -6.0]
        self.names = ["a:D", "b:D", "c:D"]
        self.pins = [3, 5, 7]
        self.actions = []
        self.proposals = iter(proposals)
        self.checkpoints = []
        self.requests = []
        self.snapshots = []
        self.restores = 0
        self.electrical = {}

    def endpoints(self):
        return rebuffer_cpp.TimingEndpointState(self.pins, self.names, self.slacks)

    def snapshot(self):
        pin = self.config["native_endpoint_pin"]
        index = self.pins.index(pin)
        self.snapshots.append((pin, tuple(self.slacks)))
        return {
            "items": [
                {
                    "endpoint_pin": pin,
                    "endpoint_pin_name": self.names[index],
                    "slack_ps": self.slacks[index],
                    "path_pins": [len(self.actions), pin],
                    "net_ids": [net],
                    "root_net_id": net,
                    "driver_pin_id": driver,
                }
                for net, driver in ((0, 2), (1, 4))
            ]
        }

    def provider(self, request):
        self.requests.append(dict(request))
        proposal = next(self.proposals, None)
        if proposal is None:
            return {"status": "no_actions"}
        return {"status": "ok", "actions": [{"after": proposal, "id": len(self.requests)}]}

    def checkpoint(self, _label):
        self.checkpoints.append(deepcopy((self.slacks, self.actions, self.electrical)))
        return {"checkpoint_id": len(self.checkpoints) - 1}

    def apply(self, actions):
        self.actions.extend(deepcopy(actions))
        self.slacks = list(actions[-1]["after"])

    def evaluate(self):
        return {
            "status": "ok",
            "wns_ps": min(self.slacks),
            "tns_ps": sum(min(0, slack) for slack in self.slacks),
            "endpoint_slacks_by_name": dict(zip(self.names, self.slacks, strict=True)),
            "electrical": {"violations_by_pin": deepcopy(self.electrical)},
        }

    def restore(self, token):
        self.slacks, self.actions, self.electrical = deepcopy(self.checkpoints[token])
        self.restores += 1

    def run(self):
        return rebuffer_cpp.run_repair_timing_legacy_queue_operator(
            self.endpoints,
            self.snapshot,
            self.provider,
            self.checkpoint,
            self.apply,
            self.evaluate,
            self.restore,
            self.config,
        )


def test_wns_reselects_each_pass_then_visits_the_initial_endpoint_order():
    campaign = Campaign([None, [-7, -8, -6], [-7, -5, -6]])

    result = campaign.run()

    assert [(row["phase"], row["endpoint_pin_name"], row["pass"]) for row in result["passes"]] == [
        ("wns", "a:D", 1),
        ("wns", "b:D", 2),
        ("endpoint", "b:D", 1),
        ("endpoint", "c:D", 1),
    ]
    assert [request["driver_pin_id"] for request in campaign.requests[:3]] == [2, 4, 2]
    assert result["passes"][0]["drivers"][0]["status"] == "no_actions"
    assert result["passes"][1]["drivers"][0]["path_pins"] == [1, 5]
    assert (result["attempt_count"], result["pass_count"], result["accepted_count"]) == (7, 4, 2)
    assert result["final"] == {
        **campaign.evaluate(),
        "endpoint_pin_name": "b:D",
        "endpoint_slack_ps": -5.0,
    }
    assert "native_endpoint_pin" not in campaign.config


def test_same_wns_endpoint_gain_is_kept_only_in_the_endpoint_sweep():
    campaign = Campaign([[-10, -5, -6], [-10, -5, -6]], legacy_max_passes_per_endpoint=1)

    result = campaign.run()

    assert [(row["phase"], row["status"]) for row in result["passes"]] == [
        ("wns", "provisional"),
        ("wns", "restored_best"),
        ("endpoint", "accepted"),
        ("endpoint", "no_change"),
    ]
    assert campaign.slacks == [-10, -5, -6]
    assert [action["id"] for action in result["accepted_actions"]] == [2]
    assert campaign.actions == result["accepted_actions"]


def test_legacy_wns_improvement_uses_the_reference_objective_when_tns_worsens():
    campaign = Campaign([[-9, -9, -9]], legacy_max_passes_per_endpoint=1)

    result = campaign.run()

    assert result["passes"][0]["status"] == "accepted"
    assert (result["final"]["wns_ps"], result["final"]["tns_ps"]) == (-9.0, -27.0)
    assert campaign.actions == result["accepted_actions"]


@pytest.mark.parametrize("limit", [0, 1])
def test_nonimproving_passes_restore_the_best_prefix(limit):
    campaign = Campaign(
        [[-11, -8, -6], [-12, -8, -6]],
        legacy_max_passes_per_endpoint=4,
        decreasing_slack_max_passes=limit,
    )
    # After the configured provisional prefix, subsequent drivers have no move.
    campaign.proposals = iter([[-11 - step, -8, -6] for step in range(limit + 1)])

    result = campaign.run()

    assert campaign.slacks == [-10, -8, -6]
    assert campaign.actions == result["accepted_actions"] == []
    assert sum(row["status"] == "provisional" for row in result["passes"]) == limit + 1
    assert sum(row["status"] == "restored_best" for row in result["passes"]) == 1


@pytest.mark.parametrize("budget", [{"max_passes": 1}, {"max_tp_evaluations": 2}])
def test_budget_stop_restores_an_unimproved_candidate(budget):
    campaign = Campaign([[-11, -8, -6]], **budget)

    result = campaign.run()

    assert campaign.slacks == [-10, -8, -6]
    assert campaign.actions == result["accepted_actions"] == []
    assert result["stop_reason"] == (
        "max_attempts" if "max_passes" in budget else "max_tp_evaluations"
    )
    assert (result["attempt_count"], result["evaluation_calls"]) == (1, 2)


def test_endpoint_fraction_and_pass_limit_are_separate_from_attempts():
    campaign = Campaign([], repair_tns_end_percent=0.0, legacy_max_passes_per_endpoint=1)

    result = campaign.run()

    assert [(row["phase"], row["endpoint_pin_name"]) for row in result["passes"]] == [
        ("wns", "a:D")
    ]
    assert (result["attempt_count"], result["pass_count"]) == (2, 1)


def test_fatal_snapshot_error_restores_the_campaign_root_and_query_config():
    campaign = Campaign([[-7, -8, -6]], native_endpoint_pin=123)
    original = campaign.snapshot

    def invalid_snapshot():
        snapshot = original()
        if campaign.actions:
            snapshot["items"][0]["endpoint_pin"] = 99
        return snapshot

    campaign.snapshot = invalid_snapshot
    with pytest.raises(ValueError, match="unrequested endpoint"):
        campaign.run()

    assert campaign.slacks == [-10, -8, -6]
    assert campaign.actions == []
    assert campaign.config["native_endpoint_pin"] == 123


def test_electrical_regression_is_rejected_even_if_wns_improves():
    campaign = Campaign([[-7, -8, -6]], legacy_max_passes_per_endpoint=1)
    original = campaign.apply

    def worsen(actions):
        original(actions)
        campaign.electrical = {"u:Y": {"slew_ps": 20.0, "cap_pf": 0.0, "fanout": 0.0}}

    campaign.apply = worsen
    result = campaign.run()

    assert result["passes"][0]["drivers"][0]["status"] == "electrical_rejected"
    assert campaign.slacks == [-10, -8, -6]
    assert campaign.actions == result["accepted_actions"] == []


def test_clean_design_does_not_request_any_path():
    campaign = Campaign([])
    campaign.slacks = [1, 2, 3]

    result = campaign.run()

    assert campaign.snapshots == campaign.requests == []
    assert (result["attempt_count"], result["pass_count"], result["accepted_count"]) == (0, 0, 0)
