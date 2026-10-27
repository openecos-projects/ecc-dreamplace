"""Analytic min-side circuits, independently specified from the LUT kernels."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.placeio_ecc.hold_schema import HoldContext, HoldLaunch, HoldTimingData
from dreamplace.ops.timing_propagation.early_propagation import EarlyPropagation, _cell_lanes


@pytest.fixture
def circuit():
    # CK -> Q -> wire -> BUF -> wire -> D. Rise/fall C2Q are 10/12 ps.
    # The BUF adds 20 ps; the live hold check is 40 ps plus 0.1*data_slew.
    timing = SimpleNamespace(
        num_pins=5,
        device=torch.device("cpu"),
        timing_aggregation_mode="hard",
        timing_aggregation_tau_ps=2.0,
        flat_inst_arcs_by_level=torch.tensor([[0, 1, 0, 0, 0, 1], [2, 3, 0, 1, 1, 0]]),
        flat_inst_arcs_by_level_start=torch.tensor([0, 1, 2]),
        net_flat_arcs=torch.tensor([[1, 2], [3, 4]]),
        net_flat_arcs_start=torch.tensor([0, 1, 2]),
        pin_net=torch.tensor([0, 0, 0, 1, 1]),
        start_points=torch.tensor([1]),
        clock_pins=torch.tensor([0]),
        end_points=torch.tensor([4]),
        endpoints_timing_check_arcs=torch.tensor([[0, 4, 0, 2, 1, 1, 2, 0]]),
        arcs_info=SimpleNamespace(
            r_check_luts=SimpleNamespace(valid_arc_mask=torch.ones(3, dtype=torch.bool)),
            f_check_luts=SimpleNamespace(valid_arc_mask=torch.ones(3, dtype=torch.bool)),
        ),
    )
    timing.r_delay_entry = lambda cell, slew, cap, arc, node, **kw: 10 + arc * 10 + cap * 2
    timing.f_delay_entry = lambda cell, slew, cap, arc, node, **kw: 12 + arc * 8 + cap * 3
    timing.r_tran_entry = lambda cell, slew, cap, arc, node, **kw: 2 + slew * 0.5 + cap
    timing.f_tran_entry = lambda cell, slew, cap, arc, node, **kw: 3 + slew * 0.5 + cap
    timing.lut_entry_vectorized = lambda cell, clock, data, arc, lut: 40 + data * 0.1
    context = HoldContext(
        start_pin=1,
        native_start_pin=1,
        launch_pin=0,
        endpoint_pin=4,
        capture_pin=0,
        launch_clock="clk",
        capture_clock="clk",
        path_tag="clk",
        source_type=2,
        check_type=2,
        data_transition=1,
        launch_transition=1,
        capture_transition=1,
        capture_pin_transition=1,
        capture_edge_index=-1,
        selected_check_arc_idx=-1,
        arrival_ps=30,
        required_ps=40,
        data_slew_ps=2,
        normal_required_ps=0,
        launch_time_ps=0,
        launch_clock_arrival_ps=0,
        launch_clock_slew_ps=2,
        capture_edge_ps=0,
        capture_arrival_ps=0,
        capture_slew_ps=2,
        check_time_ps=40,
        uncertainty_ps=0,
        cppr_ps=0,
    )
    launch = HoldLaunch(1, 0, "clk", 2, 1, 1, 0, 2, 0, 2, "clk")
    data = HoldTimingData(
        np.array([[True, True]]),
        np.array([[0, 0]]),
        np.array([[True, True]]),
        np.array([[0, 0]]),
        np.array([[True, True]]),
        np.array([[10.0, 12.0]]),
        np.array([[2.0, 3.0]]),
        (context, replace(context, data_transition=-1)),
        (launch,),
        (np.zeros(1), np.zeros(1)),
        np.array([0, 0, 2]),
        (np.zeros(1), np.zeros(1)),
        np.zeros(3, dtype=bool),
    )
    return timing, data


def evaluate(circuit, caps=None, *, mode="hard"):
    timing, data = circuit
    timing.timing_aggregation_mode = mode
    zero = torch.zeros(5, dtype=torch.float64)
    caps = zero if caps is None else caps
    return EarlyPropagation(data).forward(timing, (zero, zero), (zero, zero), (caps, caps))


def test_live_clock_to_q_delay_and_live_check_have_analytic_slack(circuit):
    result = evaluate(circuit)
    torch.testing.assert_close(
        result.context_slack, torch.tensor([-10.35, -8.5], dtype=torch.float64)
    )
    assert tuple(float(v) for v in (result.hold_wns, result.hold_tns)) == pytest.approx(
        (-10.35, -10.35)
    )
    assert int(result.hold_violations) == 1
    assert int(result.removal_violations) == 0


def test_removal_uses_clock_then_data_axes_and_stays_outside_hold_objective(circuit):
    timing, data = circuit
    timing.endpoints_timing_check_arcs[:, 6] = 4
    timing.lut_entry_vectorized = (
        lambda cell, related, constrained, arc, lut: related * 10 + constrained
    )
    data = replace(data, contexts=tuple(replace(c, check_type=4) for c in data.contexts))
    result = evaluate((timing, data))
    torch.testing.assert_close(result.context_slack, torch.tensor([6.5, 7.0], dtype=torch.float64))
    assert float(result.removal_worst) == 6.5
    assert sum(result.deficits(100.0)) == 0


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_hold_deficit_gradient_and_finite_difference_agree(circuit, mode):
    cap = torch.full((5,), 0.2, dtype=torch.float64, requires_grad=True)
    result = evaluate(circuit, cap, mode=mode)
    loss = sum(result.deficits(0))
    (gradient,) = torch.autograd.grad(loss, cap)
    eps = 1e-5
    for pin in (1, 3):
        plus, minus = cap.detach().clone(), cap.detach().clone()
        plus[pin] += eps
        minus[pin] -= eps
        difference = (
            sum(evaluate(circuit, plus, mode=mode).deficits(0))
            - sum(evaluate(circuit, minus, mode=mode).deficits(0))
        ) / (2 * eps)
        torch.testing.assert_close(gradient[pin], difference)
        assert gradient[pin] < 0  # More path delay reduces a hold deficit.
    assert torch.isfinite(gradient).all()


def test_second_forward_recomputes_q_instead_of_retaining_old_bounds(circuit):
    timing, data = circuit
    op = EarlyPropagation(data)
    zero = torch.zeros(5, dtype=torch.float64)
    high = torch.ones_like(zero)
    first = op.forward(timing, (zero, zero), (zero, zero), (high, high))
    second = op.forward(timing, (zero, zero), (zero, zero), (zero, zero))
    assert second.hold_wns < first.hold_wns
    torch.testing.assert_close(second.context_slack, evaluate(circuit).context_slack)


def test_setup_traversal_pruning_keeps_hold_only_path_and_gradient(circuit):
    from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation

    timing, _ = circuit
    timing._traversal_pruning_enabled = lambda: True
    timing.last_traversal_pruning_stats = {"applied": True}
    timing.last_traversal_pruning_result = SimpleNamespace(
        kept_flat_arc_indices=torch.tensor([0]),
        kept_level_offsets=torch.tensor([0, 1, 1]),
    )
    late_arcs, _, _, applied = TimingPropagation._build_traversal_pruning_working_view(
        timing, timing.device,
    )
    assert applied and late_arcs.shape[0] == 1  # BUF is outside the late view.
    caps = torch.full((5,), 0.2, dtype=torch.float64, requires_grad=True)
    result = evaluate(circuit, caps)
    deficit = sum(result.deficits(0))
    assert deficit > 0
    gradient, = torch.autograd.grad(deficit, caps)
    assert gradient[3] < 0  # The excluded BUF still contributes to hold repair.
    assert torch.isfinite(gradient).all()


def test_arrival_and_required_are_paired_before_endpoint_reduction(circuit):
    timing, data = circuit
    a = replace(
        data.contexts[0],
        check_type=0,
        arrival_ps=10,
        required_ps=5,
        check_time_ps=0,
    )
    b = replace(a, launch_clock="other", arrival_ps=20, required_ps=30)
    data = replace(
        data,
        contexts=(a, b),
        launches=(data.launches[0], replace(data.launches[0], launch_clock="other")),
    )
    op = EarlyPropagation(data)
    reference = torch.zeros(5, dtype=torch.float64)
    domains = {a.launch_key: (reference + 10, reference), b.launch_key: (reference + 20, reference)}
    result = op._evaluate(timing, domains, (reference, reference), reference)
    assert float(result.hold_wns) == -10  # The unpaired extrema would give -20.


def test_non_unate_early_slew_is_independent_of_arrival_winner(circuit):
    timing, _ = circuit
    arcs = torch.tensor([[0, 1, 0, 0, 0, 0]])
    slew = (torch.full((5,), 8.0), torch.full((5,), 2.0))
    caps = (torch.zeros(5), torch.zeros(5))
    lanes = _cell_lanes(timing, arcs, slew, caps)
    rising = [lane for lane in lanes if lane.output_edge == 0]
    assert [float(lane.slew) for lane in rising] == [6.0, 3.0]


def test_margin_zero_does_not_reward_positive_hold_slack(circuit):
    cap = torch.full((5,), 10.0, dtype=torch.float64, requires_grad=True)
    result = evaluate(circuit, cap)
    loss = sum(result.deficits(0))
    assert float(loss.detach()) == 0
    (gradient,) = torch.autograd.grad(loss, cap)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
