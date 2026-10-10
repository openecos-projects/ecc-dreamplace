"""Degenerate curvature must keep the placement learning rate finite."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.PlaceObj import PlaceObj


@pytest.mark.parametrize("slope", [0.0, 0.05, 1.0])
def test_backtracking_constant_gradient_keeps_probes_leaf_and_live_position_unchanged(slope):
    position = torch.nn.Parameter(torch.tensor([1.0]))
    initial = position.detach().clone()

    def objective_and_gradient(probe):
        assert probe.is_leaf
        if probe.grad is not None:
            probe.grad.zero_()
        objective = slope * probe.sum()
        objective.backward()
        return objective, probe.grad.clone()

    model = SimpleNamespace(obj_and_grad_fn=objective_and_gradient)
    learning_rate = PlaceObj.estimate_initial_learning_rate(model, position, 0.01)

    assert torch.isfinite(learning_rate)
    assert learning_rate > 0
    torch.testing.assert_close(position.detach(), initial)
