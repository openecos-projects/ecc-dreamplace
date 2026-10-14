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


def test_refresh_changed_objective_preserves_momentum_and_uses_current_gradient():
    position = torch.nn.Parameter(torch.tensor([4.]))
    optimizer = NesterovAcceleratedGradientOptimizer(
        [position], lr=.1, obj_and_grad_fn=_quadratic([0.]),
        constraint_fn=lambda value: value, use_bb=False,
    )
    position.grad = torch.zeros_like(position)
    optimizer.step()
    group = optimizer.param_groups[0]
    momentum = {key: [value.detach().clone() for value in group[key]] for key in ("u_k", "a_k", "alpha_k")}
    stale_gradient = group["g_k"][0].clone()
    optimizer.obj_and_grad_fn = _quadratic([10.])
    expected_objective, expected_gradient = optimizer.obj_and_grad_fn(position)

    summary = optimizer.refresh_objective_state(reason="changed_target")

    assert summary["relative_gradient_changes"][0] > 0
    assert not torch.equal(stale_gradient, group["g_k"][0])
    torch.testing.assert_close(group["g_k"][0], expected_gradient)
    torch.testing.assert_close(group["obj_k"][0], expected_objective)
    for key, values in momentum.items():
        for actual, expected in zip(group[key], values, strict=True):
            torch.testing.assert_close(actual, expected)
    before = position.detach().clone()
    optimizer.step()
    assert position.item() > before.item()


@pytest.mark.parametrize("use_bb", [False, True])
@pytest.mark.parametrize("slope", [0.0, 0.05, 1.0])
def test_zero_curvature_keeps_steps_finite(use_bb, slope):
    position = torch.nn.Parameter(torch.tensor([2.0]))

    def objective_and_gradient(probe):
        objective = slope * probe.sum()
        return objective, torch.autograd.grad(objective, probe)[0]

    optimizer = NesterovAcceleratedGradientOptimizer(
        [position],
        lr=0.1,
        obj_and_grad_fn=objective_and_gradient,
        constraint_fn=lambda value: value,
        use_bb=use_bb,
    )
    position.grad = torch.zeros_like(position)
    initial = position.detach().clone()

    for _ in range(3):
        optimizer.step()
        assert torch.isfinite(position).all()
        assert torch.isfinite(optimizer.param_groups[0]["alpha_k"][0]).all()

    if slope == 0:
        torch.testing.assert_close(position.detach(), initial)
    else:
        assert position.item() < initial.item()


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
