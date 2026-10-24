import math

import pytest
from dreamplace.flows.diff_sizing_spec import (
    DiffSizingFlowSpec,
    canonical_diff_sizing_flows,
    parse_diff_sizing_flows,
)


def test_parse_diff_sizing_flows_preserves_order_and_action_masks():
    flows = parse_diff_sizing_flows(
        [
            ["up", 10, [500, 5, 1, 1], 1],
            ["down", 1, [100, 2, 3, 4], 0.1],
            ["quad", 3, [10, 1, 1, 1], 5],
        ]
    )

    assert [spec.action for spec in flows] == ["up", "down", "quad"]
    assert [spec.action_mask for spec in flows] == [
        (True, False, False, True),
        (False, True, True, False),
        (True, True, True, True),
    ]
    assert flows[1].coefficients.to_dict() == {
        "wns": 100.0,
        "tns": 2.0,
        "slew": 3.0,
        "cap": 4.0,
    }
    assert flows[1].selection_percent == 0.1
    assert canonical_diff_sizing_flows(
        [["up", 10, [500, 5, 1, 1], 1]]
    ) == [
        [
            "up",
            10,
            {"wns": 500.0, "tns": 5.0, "slew": 1.0, "cap": 1.0},
            1.0,
        ]
    ]


def test_parse_diff_sizing_flows_none_is_legacy_mode():
    assert parse_diff_sizing_flows(None) is None
    assert canonical_diff_sizing_flows(None) is None


@pytest.mark.parametrize(
    "value",
    [
        [],
        [["up", 1, [500, 5, 1, 1]]],
        [["sideways", 1, [500, 5, 1, 1], 1]],
        [["up", 0, [500, 5, 1, 1], 1]],
        [["up", True, [500, 5, 1, 1], 1]],
        [["up", 1, [500, 5, 1, 1], 0]],
        [["up", 1, [500, 5, 1, 1], 100.1]],
        [["up", 1, [500, 5, 1, 1], math.nan]],
        [["up", 1, [500, 5, 1, 1], True]],
        [["up", 1, [500, 5, 1], 1]],
    ],
)
def test_parse_diff_sizing_flows_rejects_invalid_phases(value):
    with pytest.raises(ValueError):
        parse_diff_sizing_flows(value)


def test_diff_sizing_flow_spec_rejects_unknown_action():
    with pytest.raises(ValueError, match="unsupported diff_sizing action"):
        DiffSizingFlowSpec(
            action="left",
            rounds=1,
            coefficients=parse_diff_sizing_flows(
                [["quad", 1, [1, 1, 1, 1], 1]]
            )[0].coefficients,
            selection_percent=1.0,
        )
