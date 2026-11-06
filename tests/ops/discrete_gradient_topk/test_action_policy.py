"""Action policy reaches both selectors even when a forbidden move scores best."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.discrete_timing_common import timing_sizing_action_mask
from dreamplace.ops.discrete_gradient_topk import (
    apply_quad_gradient_from_action_scores,
    apply_quad_gradient_from_data_collections,
    build_quad_candidate_table,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["lut_action", "off"])
@pytest.mark.parametrize(
    "policy, explicit_mask, expected_actions, expected_cells",
    [
        (None, None, ["size_up", "vt_speedup"], [7, 5]),
        ("up", None, ["size_up", "vt_speedup"], [7, 5]),
        ("down", None, ["vt_slowdown", "size_down"], [3, 1]),
        ("quad", None, ["vt_slowdown", "size_down"], [3, 1]),
        ("up", (True, True, True, True), ["vt_slowdown", "size_down"], [3, 1]),
    ],
)
def test_sizing_action_policy_filters_before_shared_budget_ranking(
    device, backend, policy, explicit_mask, expected_actions, expected_cells
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    info = torch.tensor(
        [[3 * (size - 1) + vt, 0, size, vt] for size in (1, 2, 3) for vt in (0, 1, 2)],
        dtype=torch.float64,
        device=device,
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([4, 4], device=device),
        inst_is_sizeable=torch.ones(2, dtype=torch.bool, device=device),
        flat_libcell_info=info,
        flat_libcell_leakage=info[:, 2] * (info[:, 3] + 1),
        inst_size_lower=torch.ones(2, dtype=torch.float64, device=device),
        inst_size_upper=torch.full((2,), 3.0, dtype=torch.float64, device=device),
        inst_vt_mask=torch.ones((2, 3), dtype=torch.bool, device=device),
    )
    size = torch.full((2,), 2.0, dtype=torch.float64, device=device, requires_grad=True)
    vt = torch.tensor(
        [[-4.0, 4.0, -4.0]] * 2, dtype=torch.float64, device=device, requires_grad=True
    )
    candidates = build_quad_candidate_table(data.flat_libcell_info, data.flat_libcell_leakage)
    original_targets = candidates.target_cell_ids.clone()
    params = SimpleNamespace() if policy is None else SimpleNamespace(timing_sizing_action=policy)
    candidates_masked = candidates.with_action_mask(
        timing_sizing_action_mask(params, explicit_mask)
    )
    arguments = dict(
        data_collections=data,
        size_logits=size,
        vt_logits=vt,
        candidate_table=candidates_masked,
        shared_budget_percent=100.0,
        size_parameterization="real_size",
    )
    if backend == "lut_action":
        scores = torch.tensor(
            [[-2.0, 2.0, -20.0, 1.0], [8.0, -8.0, 1.0, -3.0]],
            dtype=torch.float64,
            device=device,
        )
        _, _, summary = apply_quad_gradient_from_action_scores(
            action_scores=scores,
            valid_instances=torch.ones_like(scores, dtype=torch.bool),
            **arguments,
        )
    else:
        vt_cost = torch.tensor(
            [[-20.0, 0.0, 1.0], [1.0, 0.0, -3.0]], dtype=torch.float64, device=device
        )
        loss = -2 * size[0] + 8 * size[1] + (torch.softmax(vt, dim=1) * vt_cost).sum()
        size_grad, vt_grad = torch.autograd.grad(loss, (size, vt))
        _, _, summary = apply_quad_gradient_from_data_collections(
            size_grad=size_grad,
            vt_grad=vt_grad,
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=0.0,
            **arguments,
        )
    assert (summary["applied_actions"], summary["applied_cell_ids"]) == (
        expected_actions,
        expected_cells,
    )
    assert summary["shared_budget_cells"] == 2
    torch.testing.assert_close(candidates.target_cell_ids, original_targets, rtol=0, atol=0)
