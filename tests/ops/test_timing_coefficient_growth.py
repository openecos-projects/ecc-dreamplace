"""Configured timing coefficients follow the real density-update schedule."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.PlaceObj import PlaceObj


@pytest.fixture
def density_update_model():
    metadata = json.loads(
        (Path(__file__).resolve().parents[2] / "dreamplace/params.json").read_text()
    )
    params = SimpleNamespace(**{key: entry["default"] for key, entry in metadata.items()})
    params.scale_factor = 1.0
    model = PlaceObj.__new__(PlaceObj)
    torch.nn.Module.__init__(model)
    model.params = params
    model.placedb = SimpleNamespace(regions=[object()])
    model.quad_penalty = True
    model.density_weight = torch.tensor([1.0])
    model.density_quad_coeff = 2000.0
    model.density_weight_grad_precond = torch.ones(1)
    model.density_weight_u = torch.ones(1)
    model.density_weight_step_size = 0.001
    model.density_weight_step_size_inc_low = 1.01
    model.density_weight_step_size_inc_high = 1.05
    model.update_mask = None
    model.timing_wns_coeff = 0.02
    model.timing_tns_coeff = 0.0002
    model.timing_slew_weight = 3.0
    model.timing_cap_weight = 4.0
    model.timing_grad_balance_weight = 7.0
    return model


@pytest.mark.parametrize("algo", ["hpwl", "overflow"])
@pytest.mark.parametrize("mode", ["place_only", "size_only"])
@pytest.mark.parametrize("factor", [None, 1.0, 1.02, 0.99])
def test_density_updates_apply_configured_multiplier(algo, mode, factor, density_update_model):
    model = density_update_model
    model.params.placement_sizing_mode = mode
    if factor is not None:
        model.params.timing_coeff_growth_factor = factor
    factor = model.params.timing_coeff_growth_factor
    update = model.build_update_density_weight(model.params, model.placedb, algo=algo)
    metric = SimpleNamespace(
        hpwl=torch.tensor(101.0),
        density=torch.tensor([0.5]),
        overflow=torch.tensor([0.2]),
    )
    previous = SimpleNamespace(hpwl=torch.tensor(100.0))
    for iteration in range(1, 4):
        update(metric, previous, iteration)
        multiplier = 1.0 if mode == "size_only" else factor**iteration
        assert (model.timing_wns_coeff, model.timing_tns_coeff) == pytest.approx(
            (0.02 * multiplier, 0.0002 * multiplier)
        )
        assert (
            model.timing_slew_weight,
            model.timing_cap_weight,
            model.timing_grad_balance_weight,
        ) == (3.0, 4.0, 7.0)
    assert model.density_weight.item() != 1.0


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_growth_factor_is_rejected(factor, density_update_model):
    model = density_update_model
    model.params.timing_coeff_growth_factor = factor
    with pytest.raises(ValueError, match="timing_coeff_growth_factor"):
        model.build_update_density_weight(model.params, model.placedb, algo="hpwl")
