import pytest
import torch

quad_gradient_topk_cpp = pytest.importorskip(
    "dreamplace.ops.discrete_gradient_topk.quad_gradient_topk_cpp"
)
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    _select_direction,
    _select_quad_actions,
    _select_shared_budget,
    _select_vt,
)


def test_native_quad_selection_matches_tensor_reference():
    torch.manual_seed(17)
    scores = torch.randn(101, 4, dtype=torch.float64)
    valid = torch.rand(101, 4) > 0.2
    sizeable = torch.rand(101) > 0.1
    blocked = torch.rand(101) > 0.9

    for shared_budget_percent in (None, 1.0, 17.5, 100.0):
        selected, budget, baseline, backend = _select_quad_actions(
            scores,
            valid,
            sizeable,
            shared_budget_percent=shared_budget_percent,
            blocked_instances=blocked,
            size_up_percent=13.0,
            size_down_percent=7.0,
            vt_percent=11.0,
        )
        assert backend == "cpp"
        if shared_budget_percent is None:
            expected = (
                _select_direction(scores, valid, 0, 13.0)
                | _select_direction(scores, valid, 1, 7.0)
                | _select_vt(scores, valid, 11.0)
            )
        else:
            expected_baseline, _ = _select_shared_budget(
                scores, valid, sizeable, shared_budget_percent
            )
            expected_valid = valid & ~blocked.unsqueeze(1)
            expected, expected_budget = _select_shared_budget(
                scores,
                expected_valid,
                sizeable,
                shared_budget_percent,
            )
            assert budget == expected_budget
            torch.testing.assert_close(baseline, expected_baseline)
        torch.testing.assert_close(selected, expected)


def test_native_module_is_importable():
    assert quad_gradient_topk_cpp is not None
