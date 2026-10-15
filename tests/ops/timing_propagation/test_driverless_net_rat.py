"""Nets without timing arcs have no driver to receive a required arrival time."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


@pytest.mark.parametrize("drivers", [[0, -1], [-1, -1]])
def test_net_rat_ignores_driverless_nets_and_preserves_gradients(drivers):
    model = SimpleNamespace(
        pin_net=torch.tensor([0, 0, 1, 1]),
        net2driver_pin_map=torch.tensor(drivers),
        timing_aggregation_mode="hard",
        timing_aggregation_tau_ps=1.0,
    )
    rise = torch.tensor([100.0, 10.0, 2.0, 1.0], requires_grad=True)
    fall = torch.tensor([100.0, 20.0, 3.0, 2.0], requires_grad=True)
    delays = torch.tensor([0.0, 1.0, 1.0, 1.0], requires_grad=True)
    actual = TimingPropagation.calculate_net_rat_level(
        model, torch.tensor([1, 2, 3]), rise, fall, delays, delays
    )
    if drivers[0] == 0:
        expected = (
            torch.stack((rise[1] - delays[1], *rise[1:])),
            torch.stack((fall[1] - delays[1], *fall[1:])),
        )
    else:
        expected = (rise, fall)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(
        sum(x.sum() for x in actual), (rise, fall, delays), retain_graph=True, allow_unused=True
    )
    expected_grad = torch.autograd.grad(
        sum(x.sum() for x in expected), (rise, fall, delays), allow_unused=True
    )
    # Empty scatter still connects an empty delay selection to the graph.
    actual_grad = tuple(
        torch.zeros_like(x) if g is None else g
        for x, g in zip((rise, fall, delays), actual_grad, strict=True)
    )
    expected_grad = tuple(
        torch.zeros_like(x) if g is None else g
        for x, g in zip((rise, fall, delays), expected_grad, strict=True)
    )
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("drivers", [[0, -1], [-1, -1], [0, -2], [0, 4]])
def test_driver_map_validation_accepts_only_missing_driver_sentinel(drivers):
    model = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(model)
    model.num_pins = 4
    model.net2driver_pin_map = torch.tensor(drivers)
    model.pin_net = torch.tensor([0, 0, 1, 1])
    for name in (
        "start_points",
        "end_points",
        "clock_pins",
        "FF_ids",
        "flat_pin_to_graph",
        "flat_pin_to_graph_reverse",
        "endpoints_constraint_arcs",
        "flat_inst_arcs_by_level",
    ):
        setattr(model, name, None)
    if min(drivers) < -1 or max(drivers) >= 4:
        with pytest.raises(RuntimeError, match="net2driver_pin_map"):
            model._validate_runtime_indices()
    else:
        model._validate_runtime_indices()
