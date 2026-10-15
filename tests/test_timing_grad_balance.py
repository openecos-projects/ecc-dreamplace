from types import SimpleNamespace

import pytest
import torch

from dreamplace.PlaceObj import PlaceObj


def _model(**kwargs):
    model = PlaceObj.__new__(PlaceObj)
    defaults = {
        "timing_objective_lane": "timing_only",
        "differentiable_timing_obj": 1,
        "critical_endpoint_pruning_mode": "off",
        "production_fast_loop": True,
        "timing_slew_weight": 1.0,
        "timing_cap_weight": 1.0,
        "timing_leakage_weight": 0.0,
    }
    defaults.update(kwargs)
    model.params = SimpleNamespace(**defaults)
    model.op_collections = SimpleNamespace(timing_propagation_op=None)
    model.timing_wns_coeff = 0.01
    model.timing_tns_coeff = 0.0001
    model.timing_slew_weight = 1.0
    model.timing_cap_weight = 1.0
    model.timing_leakage_weight = 0.0
    model.timing_grad_balance_weight = 1.0
    model.timing_grad_balance_summary = {}
    return model


def test_timing_grad_balance_uses_inverse_gradient_norm_ratio():
    result = PlaceObj._timing_grad_balance_from_norms(
        wirelength_grad_norm=20.0,
        timing_grad_norm=100.0,
        target_ratio=0.1,
    )

    assert result["status"] == "initialized"
    assert result["weight_raw"] == pytest.approx(0.02)
    assert result["weight_applied"] == pytest.approx(0.02)
    assert result["weighted_timing_to_wirelength_grad_ratio"] == pytest.approx(0.1)


def test_timing_grad_balance_zero_target_preserves_fixed_weight():
    result = PlaceObj._timing_grad_balance_from_norms(
        wirelength_grad_norm=20.0,
        timing_grad_norm=100.0,
        target_ratio=0.0,
    )

    assert result["status"] == "disabled"
    assert result["initialized"]
    assert result["weight_applied"] == 1.0


def test_timing_grad_balance_defers_zero_timing_gradient():
    result = PlaceObj._timing_grad_balance_from_norms(
        wirelength_grad_norm=20.0,
        timing_grad_norm=0.0,
        target_ratio=0.1,
    )

    assert result["status"] == "deferred_zero_timing_grad"
    assert not result["initialized"]
    assert result["weight_applied"] == 1.0


def test_timing_outer_weight_scales_composite_wns_tns_term():
    model = _model()
    model.timing_grad_balance_weight = 2.0

    loss = model._timing_loss(
        torch.tensor(-10.0),
        torch.tensor(-100.0),
        None,
        None,
    )

    assert loss.item() == pytest.approx(0.22)
    assert model.last_timing_objective_terms["raw_terms"]["timing"] == pytest.approx(
        0.11
    )
    assert model.last_timing_objective_terms["weights"]["timing"] == 2.0
    assert model.last_timing_objective_terms["weighted_terms"][
        "timing"
    ] == pytest.approx(0.22)


def test_terminal_coefficients_follow_live_model_after_balance_initialization():
    model = _model()
    state = {
        "initialized": True,
        "weight_applied": 7.0,
        "timing_wns_coeff": model.timing_wns_coeff,
        "timing_tns_coeff": model.timing_tns_coeff,
    }
    model.apply_timing_grad_balance_state(state)
    model.timing_wns_coeff = 2.5
    model.timing_tns_coeff = 0.025
    loss = model._timing_loss(torch.tensor(-10.0), torch.tensor(-100.0), None, None)

    assert loss.item() == pytest.approx(192.5)
    assert model.last_timing_objective_terms["coefficients"] == {
        "wns": 2.5, "tns": 0.025,
    }
    assert model.last_timing_objective_terms["grad_balance"] == state


def test_timing_grad_balance_initializes_from_real_autograd_gradients():
    model = _model()
    model.timing_wns_coeff = 1.0
    model.timing_tns_coeff = 0.0
    model.timing_grad_balance_target_ratio = 0.1
    model.timing_grad_balance_summary = {
        "status": "pending",
        "initialized": False,
    }
    model.op_collections.wirelength_op = lambda pos: (2.0 * pos).sum()
    model.timing_obj = lambda pos: (
        -(5.0 * pos).sum(),
        pos.sum() * 0.0,
        None,
        None,
    )
    pos = torch.tensor([1.0, 2.0], requires_grad=True)

    result = model.initialize_timing_grad_balance(
        pos,
        iteration=20,
        overflow=0.19,
    )

    assert result["wirelength_grad_norm"] == pytest.approx(4.0)
    assert result["timing_grad_norm"] == pytest.approx(10.0)
    assert result["weight_applied"] == pytest.approx(0.04)
    assert model.timing_grad_balance_weight == pytest.approx(0.04)
    assert pos.grad is None


def test_size_only_density_weight_update_does_not_drift_timing_coefficients():
    model = PlaceObj.__new__(PlaceObj)
    model.params = SimpleNamespace(placement_sizing_mode="size_only")
    model.density_weight = torch.tensor([1.0], dtype=torch.float32)
    model.scale_factor = 1.0
    model.quad_penalty = False
    model.placedb = SimpleNamespace(regions=[])
    model.timing_wns_coeff = 0.01
    model.timing_tns_coeff = 0.0001

    update = model.build_update_density_weight(
        SimpleNamespace(
            RePlAce_ref_hpwl=350000.0,
            RePlAce_LOWER_PCOF=0.95,
            RePlAce_UPPER_PCOF=1.05,
            scale_factor=1.0,
        ),
        SimpleNamespace(regions=[]),
        algo="hpwl",
    )
    update(
        SimpleNamespace(hpwl=torch.tensor(101.0)),
        SimpleNamespace(hpwl=torch.tensor(100.0)),
        1,
    )

    assert model.timing_wns_coeff == pytest.approx(0.01)
    assert model.timing_tns_coeff == pytest.approx(0.0001)
