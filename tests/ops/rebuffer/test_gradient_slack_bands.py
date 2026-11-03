"""Native slack bands retain the primary order outside a numerical neighborhood."""

from dreamplace.ops.rebuffer import rebuffer_cpp


def test_gradient_cannot_jump_a_slack_band_or_reorder_one_endpoints_drivers():
    records = [
        (1, -100.0, 0, 1.0),
        (1, -100.0, 1, 100.0),
        (2, -98.0, 2, 1000.0),
        (3, -80.0, 3, 1000000.0),
    ]
    snapshot = {
        "items": [
            {
                "endpoint_pin": endpoint,
                "slack_ps": slack,
                "root_net_id": net,
                "driver_pin_id": net,
                "net_ids": [net],
                "path_pins": [net, endpoint],
                "gradient_score": score,
                "tree_key": str(net),
            }
            for endpoint, slack, net, score in records
        ]
    }
    config = {
        "gradient_ordering": True,
        "gradient_window_tiebreak": True,
        "native_tree_round_robin": True,
        "native_tree_round_robin_endpoint_window": 3,
        "gradient_slack_window_ps": 5.0,
    }
    rows = rebuffer_cpp.select_timing_queue_items(snapshot, config)
    assert [row["root_net_id"] for row in rows] == [2, 0, 1, 3]
    config["gradient_slack_window_ps"] = 0.0
    rows = rebuffer_cpp.select_timing_queue_items(snapshot, config)
    assert [row["root_net_id"] for row in rows] == [0, 1, 2, 3]
