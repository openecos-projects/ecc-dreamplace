"""The native acceptance boundary protects accumulated global TP QoR."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp as native


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ({"wns_ps": -11.0, "tns_ps": -19.0}, "rejected"),
        ({"wns_ps": -9.0, "tns_ps": -21.0}, "rejected"),
        ({"wns_ps": -10.0, "tns_ps": -20.0}, "accepted"),
        ({"wns_ps": -9.0, "tns_ps": -19.0}, "accepted"),
        ({"wns_ps": float("nan"), "tns_ps": -19.0}, "evaluate_failed"),
    ],
)
def test_endpoint_gain_cannot_bypass_global_envelope_and_rejection_restores(after, expected):
    baseline = {"status": "ok", "wns_ps": -10.0, "tns_ps": -20.0, "endpoint_slack_ps": -5.0}
    current = dict(baseline)
    checkpoints = []
    restores = []

    def checkpoint(_label):
        checkpoints.append(dict(current))
        return {"checkpoint_id": len(checkpoints) - 1}

    def apply(_actions):
        current.update(after, endpoint_slack_ps=-4.0)

    def restore(token):
        current.clear()
        current.update(checkpoints[token])
        restores.append(token)

    result = native.run_repair_timing_queue_operator(
        lambda: {"status": "ok", "items": [{"endpoint_pin": 1, "slack_ps": -5.0, "root_net_id": 0, "net_ids": [0]}]},
        lambda _request: {"status": "ok", "actions": [{"net_id": 0}]},
        checkpoint, apply, lambda: dict(current), restore,
        {"max_passes": 1, "acceptance": "endpoint_journal", "allow_decreasing_slack": True},
    )
    assert result["passes"][0]["status"] == expected
    if expected != "accepted":
        assert current == result["final"] == baseline
        assert restores == [0]
        assert result["accepted_actions"] == []
    else:
        assert result["accepted_actions"] == [{"net_id": 0}]
        assert restores == []
