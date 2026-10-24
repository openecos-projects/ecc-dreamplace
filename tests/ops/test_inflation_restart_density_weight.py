import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.Params import Params
from dreamplace.PlaceObj import PlaceObj

WIRELENGTH_COEFF = torch.tensor([1.0, -2.0, 3.0])
DENSITY_COEFF = torch.tensor([0.5, 1.5, 0.5])
EXPECTED_GRAD_NORM_RATIO = 6.0 / 2.5


def _make_model():
    model = PlaceObj.__new__(PlaceObj)
    torch.nn.Module.__init__(model)
    model.data_collections = SimpleNamespace(
        pos=[torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))]
    )
    model.op_collections = SimpleNamespace(
        wirelength_op=lambda pos: (pos * WIRELENGTH_COEFF).sum()
    )
    model.placement_density = lambda pos: (pos * DENSITY_COEFF).sum()
    model.placedb = SimpleNamespace(regions=[])
    model.density_quad_coeff = 2000.0
    model._timing_geometry_cache = None
    return model


def _make_params(density_weight=8.0e-5, restart_scale=0.001):
    return SimpleNamespace(
        density_weight=density_weight,
        inflation_density_weight_scale=restart_scale,
    )


def test_initial_density_weight_keeps_configured_scale():
    model = _make_model()
    params = _make_params(density_weight=8.0e-5, restart_scale=0.25)

    weight = model.initialize_density_weight(params, model.placedb)

    torch.testing.assert_close(
        weight,
        torch.tensor([8.0e-5 * EXPECTED_GRAD_NORM_RATIO]),
    )


def test_restart_uses_xplace_norm_ratio_without_params_density_weight():
    model = _make_model()
    model.initialize_density_weight(_make_params(), model.placedb)

    params = _make_params(density_weight=7.0e-3, restart_scale=1.0)
    weight = model.initialize_restart_density_weight(params, model.placedb)

    torch.testing.assert_close(
        weight,
        torch.tensor([EXPECTED_GRAD_NORM_RATIO]),
    )
    assert model._density_weight_grad_norm_ratio == pytest.approx(
        EXPECTED_GRAD_NORM_RATIO
    )


def test_restart_scale_reproduces_legacy_post_inflation_factor():
    model = _make_model()
    params = _make_params(restart_scale=0.1)

    weight = model.initialize_restart_density_weight(params, model.placedb)

    torch.testing.assert_close(
        weight,
        torch.tensor([0.1 * EXPECTED_GRAD_NORM_RATIO]),
    )


def test_fence_region_restart_scales_region_weights_linearly():
    def make_fence_model():
        model = _make_model()
        model.placedb = SimpleNamespace(regions=[object()])
        model.op_collections.fence_region_density_ops = [
            lambda pos: (pos * DENSITY_COEFF).sum(),
            lambda pos: (pos * DENSITY_COEFF * 2.0).sum(),
        ]
        return model

    direct = make_fence_model().initialize_restart_density_weight(
        _make_params(restart_scale=1.0),
        None,
    )
    legacy = make_fence_model().initialize_restart_density_weight(
        _make_params(density_weight=7.0e-3, restart_scale=0.1),
        None,
    )

    torch.testing.assert_close(legacy * 10.0, direct)


def test_inflation_density_weight_scale_schema_and_normalization():
    metadata = json.loads(
        (Path(__file__).resolve().parents[2] / "dreamplace/params.json").read_text()
    )
    assert metadata["inflation_density_weight_scale"]["default"] == 0.001

    params = Params()
    params.fromJson({})
    assert params.inflation_density_weight_scale == pytest.approx(0.001)

    params.fromJson({"inflation_density_weight_scale": "0.1"})
    assert params.inflation_density_weight_scale == pytest.approx(0.1)

    params.fromCmdLine(["--inflation_density_weight_scale=0.25"])
    assert params.inflation_density_weight_scale == pytest.approx(0.25)

    for invalid in (0.0, -0.1, float("nan"), float("inf"), None, "bad"):
        with pytest.raises(ValueError, match="inflation_density_weight_scale"):
            Params().fromJson({"inflation_density_weight_scale": invalid})
