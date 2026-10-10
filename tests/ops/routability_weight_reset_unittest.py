"""New density/geometry epochs must calibrate route weight without old momentum."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.routability.l_shape_gradient import apply_l_shape_gradient
from dreamplace.PlaceObj import PlaceObj


def make_model(route_gradient):
    model = object.__new__(PlaceObj)
    torch.nn.Module.__init__(model)
    model.params = SimpleNamespace()
    model._l_shape_outer_iteration = 0
    model.l_shape_routability_weight_init = 0.0
    model.l_shape_routability_weight = torch.tensor([0.01])
    model._l_shape_weight_initialized = True
    model.l_shape_grad_target_ratio = 0.1
    model.l_shape_weight_momentum = 0.1
    model.l_shape_weight_min = 1e-12
    model.l_shape_weight_max = 1.0
    model.l_shape_filler_reverse_force = 0.0
    model.l_shape_filler_pseudo_wire_ratio = 0.0
    model.density_weight = torch.tensor([0.2])
    model.update_mask = model.fix_nodes_mask = None
    model.l_shape_routability_op = SimpleNamespace(
        gradient_mode="translation", density_op=SimpleNamespace(fast_mode=False, energy_valid=True)
    )
    model.l_shape_routability_obj = lambda pos, **kwargs: (pos * route_gradient).sum()
    model.op_collections = SimpleNamespace(
        precondition_op=SimpleNamespace(apply_components=lambda gradients, *args: gradients)
    )
    model._apply_gradient_masks_only = lambda gradient: None
    return model


def evaluate(model, base_gradient):
    pos = torch.ones(2, requires_grad=True)
    base = (pos * base_gradient).sum()
    base.backward()
    apply_l_shape_gradient(model, pos, base)
    return pos.grad


def test_reset_calibrates_first_new_gradient_then_resumes_smoothing():
    model = make_model(torch.tensor([0.0, 2.0]))
    model.reset_l_shape_weight_state()
    torch.testing.assert_close(evaluate(model, torch.tensor([3.0, 4.0])), torch.tensor([3.0, 4.5]))
    assert (model.l_shape_last_weight, model.l_shape_last_target_weight) == pytest.approx(
        (0.25, 0.25)
    )
    assert model._l_shape_weight_initialized
    torch.testing.assert_close(evaluate(model, torch.tensor([6.0, 8.0])), torch.tensor([6.0, 8.55]))
    assert (model.l_shape_last_weight, model.l_shape_last_target_weight) == pytest.approx(
        (0.275, 0.5)
    )


def test_zero_route_gradient_defers_calibration_until_valid_gradient():
    route = torch.zeros(2)
    model = make_model(route)
    model.reset_l_shape_weight_state()
    torch.testing.assert_close(evaluate(model, torch.tensor([3.0, 4.0])), torch.tensor([3.0, 4.0]))
    assert not model._l_shape_weight_initialized
    route[1] = 2.0
    torch.testing.assert_close(evaluate(model, torch.tensor([3.0, 4.0])), torch.tensor([3.0, 4.5]))
    assert model._l_shape_weight_initialized
