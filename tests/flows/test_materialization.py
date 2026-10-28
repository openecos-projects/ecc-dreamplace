"""Physical area and buffer-count contracts at the terminal commit boundary."""

import pytest
from dreamplace.flows.committed_topology import (
    MaterializationAreaPolicy,
    validate_materialized_area,
)


@pytest.mark.parametrize(
    "policy,area,post_nodes,error",
    [
        (MaterializationAreaPolicy.EXACT, 100.0, 12, None),
        (MaterializationAreaPolicy.EXACT, 110.0, 12, "changed physical movable area"),
        (MaterializationAreaPolicy.CAPACITY, 110.0, 12, None),
        (MaterializationAreaPolicy.CAPACITY, 110.0, 11, "node count"),
        (MaterializationAreaPolicy.CAPACITY, 201.0, 12, "exceeds placement capacity"),
        (MaterializationAreaPolicy.CAPACITY, float("nan"), 12, "non-finite"),
    ],
)
def test_materialization_validates_selected_accounting_contract(policy, area, post_nodes, error):
    summary = {
        "gp_area": {"movable": 100.0, "capacity": 200.0},
        "refresh": {"rebuild": {"pre_counts": {"nodes": 10}, "post_counts": {"nodes": post_nodes}}},
    }
    if error:
        with pytest.raises(RuntimeError, match=error):
            validate_materialized_area(area, summary, 2, policy)
    else:
        validate_materialized_area(area, summary, 2, policy)
        assert summary["materialized_area_consistency"] == policy.value
