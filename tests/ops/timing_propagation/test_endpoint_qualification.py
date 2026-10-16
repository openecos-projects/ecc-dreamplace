from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.endpoint_qualification import (
    endpoint_metrics,
    qualify_pin_slacks,
)
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def test_invalid_edges_and_endpoints_do_not_affect_metrics_or_gradients():
    rise = torch.tensor([-2.0, -1000.0], dtype=torch.float64, requires_grad=True)
    fall = torch.tensor([-100.0, -2000.0], dtype=torch.float64, requires_grad=True)
    timing = SimpleNamespace(
        endpoint_max_valid_by_pin=torch.tensor([[True, False], [False, False]])
    )
    rslack, fslack = qualify_pin_slacks(timing, torch.tensor([0, 1]), rise, fall)
    wns, tns, worst = endpoint_metrics(torch.minimum(rslack, fslack), rise.new_zeros(()))
    torch.testing.assert_close(
        torch.stack([wns, tns, worst]), torch.full((3,), -2.0, dtype=rise.dtype)
    )
    (wns + tns).backward()
    torch.testing.assert_close(rise.grad, torch.tensor([2.0, 0.0], dtype=rise.dtype))
    torch.testing.assert_close(fall.grad, torch.zeros_like(fall))


@pytest.mark.parametrize("count", [0, 2])
def test_unconstrained_timing_has_finite_zero_objective_and_gradient(count):
    load = torch.tensor([3.0], requires_grad=True)
    slack = torch.full((count,), torch.inf)
    metrics = endpoint_metrics(slack, load.sum() * 0)
    torch.testing.assert_close(torch.stack(metrics), torch.zeros(3))
    sum(metrics).backward()
    torch.testing.assert_close(load.grad, torch.zeros_like(load))


def test_invalid_setup_check_cannot_pollute_rat_or_snapshot():
    timing = object.__new__(TimingPropagation)
    timing.timing_aggregation_mode = "hard"
    timing.timing_aggregation_tau_ps = 2.0
    timing.endpoints_constraint_arcs = torch.tensor([[0, 2, 0, 0, 1], [1, 2, 0, 1, 1]])
    timing.endpoints_constraint_max_valid = torch.tensor([[True, False], [False, False]])
    timing._critical_path_snapshot_requested = True
    timing.r_setup_entry = lambda cells, clocks, data, arcs: data + torch.where(
        arcs == 0, 10.0, 1000.0
    )
    timing.f_setup_entry = timing.r_setup_entry
    slew = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    rat = torch.tensor([100.0, 100.0, 100.0])
    rise, fall = timing.calculate_setup_rat(rat, rat, torch.zeros(2), torch.zeros(2), slew, slew)
    torch.testing.assert_close(rise, torch.tensor([100.0, 100.0, 89.0]))
    torch.testing.assert_close(fall, rat)
    rise.sum().backward()
    torch.testing.assert_close(slew.grad, torch.tensor([0.0, 0.0, -1.0]))
    assert timing._critical_path_constraint_state["test_ids"].tolist() == [0]
    assert timing._critical_path_constraint_state["endpoint_pins"].tolist() == [2]
    assert torch.isposinf(timing._critical_path_constraint_state["fall_rat"]).all()
