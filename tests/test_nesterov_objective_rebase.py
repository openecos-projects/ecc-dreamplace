import pytest
import torch

from dreamplace.NesterovAcceleratedGradientOptimizer import (
    NesterovAcceleratedGradientOptimizer,
)


def _quadratic(target):
    target = torch.as_tensor(target, dtype=torch.float32)

    def objective_and_gradient(position):
        objective = torch.sum((position - target.to(position)) ** 2)
        gradient = torch.autograd.grad(objective, position)[0]
        return objective, gradient

    return objective_and_gradient


@pytest.mark.parametrize("use_bb", [False, True])
def test_rebase_objective_state_reanchors_nesterov_without_moving_position(use_bb):
    position = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = NesterovAcceleratedGradientOptimizer(
        [position],
        lr=0.1,
        obj_and_grad_fn=_quadratic([0.0, 0.0]),
        constraint_fn=lambda value: value,
        use_bb=use_bb,
    )
    position.grad = torch.ones_like(position)
    optimizer.step()
    group = optimizer.param_groups[0]
    eval_count = int(group["obj_eval_count"])
    position_before = position.detach().clone()
    assert group["u_k"]
    assert group["v_k"]
    assert group["v_k_1"]

    summary = optimizer.rebase_objective_state(reason="fixture_objective_change")

    assert torch.equal(position.detach(), position_before)
    assert summary["reason"] == "fixture_objective_change"
    assert summary["parameter_count"] == 1
    assert summary["obj_eval_count"] == eval_count
    assert summary["parameter_gradient_reset"] == "zero_sentinel"
    assert summary["reanchored_fields"] == ["u_k", "v_k"]
    assert torch.equal(position.grad, torch.zeros_like(position))
    assert group["v_k"][0] is position
    assert torch.equal(group["u_k"][0], position_before)
    for key in (
        "g_k",
        "obj_k",
        "a_k",
        "alpha_k",
        "v_k_1",
        "g_k_1",
        "obj_k_1",
    ):
        assert group[key] == []
    assert group["v_kp1"] == [None]

    optimizer.obj_and_grad_fn = _quadratic([1.0, 1.0])
    optimizer.step()

    assert group["u_k"]
    assert group["v_k"][0] is position
    assert group["v_k_1"]
    assert torch.isfinite(position).all()
    assert not torch.equal(position.detach(), position_before)
