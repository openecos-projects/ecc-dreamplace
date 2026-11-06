"""Hand-derived hold derivatives for the shared four-action beta graph."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.flows.hold_objective import HoldObjective
from dreamplace.flows.hold_rc import HoldRC
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import ACTION_NAMES
from dreamplace.ops.placeio_ecc.hold_schema import HoldContext, HoldLaunch, HoldTimingData
from dreamplace.ops.timing_propagation.discrete_actions import DiscreteActionContext
from dreamplace.ops.timing_propagation.early_propagation import EarlyPropagation


@pytest.fixture
def action_circuit():
    # FF: CK(0)->Q(1), Q->BUF/A(2), BUF/Y(3)->FF/D(4).
    # a = cap(BUF/A), b = cap(FF/D), CK slew = 2 ps.
    # dQ=10+2a, sQ=3+a; dBUF=20+1.5*sQ+2b;
    # sBUF=2+.5*sQ+b; hold_check=40+.1*sBUF+.2*2.
    # Thus rise slack = -6.25 + 3.45a + 1.9b = -5.37 ps.
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    ff_delay, ff_slew, ff_check = (
        tensor([-2, 3, -1, 2]),
        tensor([-0.2, 0.3, -0.1, 0.4]),
        tensor([-0.5, 0.7, -0.2, 0.6]),
    )
    buf_delay, buf_slew = tensor([-3, 4, -2, 3]), tensor([-0.3, 0.4, -0.2, 0.5])
    delta_a, delta_b = tensor([0.1, -0.07, 0.04, -0.09]), tensor([0.03, -0.02, 0.01, -0.03])
    delay_bias, slew_bias, slew_slope, check_bias = (
        torch.zeros(15, dtype=torch.float64) for _ in range(4)
    )
    delay_bias[0], delay_bias[1], slew_bias[:2] = 10, 20, 2
    delay_bias[3:7], delay_bias[7:11] = 10 + ff_delay, 20 + buf_delay
    slew_bias[3:7], slew_bias[7:11] = 2 + ff_slew, 2 + buf_slew
    slew_slope[1], slew_slope[7:11] = 1.5, 1.5
    check_bias[2], check_bias[11:15] = 40, 40 + ff_check
    timing = SimpleNamespace(
        num_pins=5,
        device=torch.device("cpu"),
        timing_aggregation_mode="hard",
        timing_aggregation_tau_ps=2.0,
        flat_inst_arcs_by_level=torch.tensor([[0, 1, 0, 0, 0, 1], [2, 3, 5, 1, 1, 0]]),
        flat_inst_arcs_by_level_start=torch.tensor([0, 1, 2]),
        net_flat_arcs=torch.tensor([[1, 2], [3, 4]]),
        net_flat_arcs_start=torch.tensor([0, 1, 2, 2]),
        pin_net=torch.tensor([2, 0, 0, 1, 1]),
        start_points=torch.tensor([1]),
        clock_pins=torch.tensor([0]),
        end_points=torch.tensor([4]),
        endpoints_timing_check_arcs=torch.tensor([[0, 4, 0, 2, 1, 1, 2, 0]]),
        arcs_info=SimpleNamespace(
            **{
                f"{edge}_check_luts": SimpleNamespace(
                    valid_arc_mask=torch.ones(15, dtype=torch.bool)
                )
                for edge in ("r", "f")
            }
        ),
    )
    timing.r_delay_entry = (
        lambda cell, slew, cap, arc, node, **kw: delay_bias[arc] + slew_slope[arc] * slew + 2 * cap
    )
    timing.f_delay_entry = lambda *args, **kw: timing.r_delay_entry(*args, **kw) + 1
    timing.r_tran_entry = lambda cell, slew, cap, arc, node, **kw: slew_bias[arc] + 0.5 * slew + cap
    timing.f_tran_entry = lambda *args, **kw: timing.r_tran_entry(*args, **kw) + 0.5
    timing.lut_entry_vectorized = (
        lambda cell, clock, data, arc, lut: check_bias[arc] + 0.2 * clock + 0.1 * data
    )
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
        arrival_ps=30,
        required_ps=40,
        data_slew_ps=2,
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
    cap_table = np.zeros(25)
    for cell, value in enumerate([0.1, *(0.1 + delta_b).tolist()]):
        cap_table[cell * 3 + 1] = value
    for cell, value in enumerate([0.2, *(0.2 + delta_a).tolist()]):
        cap_table[15 + cell * 2] = value
    data = HoldTimingData(
        np.ones((1, 2), bool),
        np.zeros((1, 2), int),
        np.ones((1, 2), bool),
        np.zeros((1, 2), int),
        np.ones((1, 2), bool),
        np.zeros((1, 2)),
        np.array([[3.2, 3.66]]),
        (context, replace(context, data_transition=-1)),
        (HoldLaunch(1, 0, "clk", 2, 1, 1, 0, 2, 0, 2, "clk"),),
        (cap_table, cap_table * 0.8),
        np.array([2 if i in (2, 11, 12, 13, 14) else 0 for i in range(15)]),
        (np.zeros(1), np.zeros(1)),
        np.zeros(15, bool),
    )
    probe = DiscreteActionContext(
        beta=torch.zeros((2, 4), dtype=torch.float64, requires_grad=True),
        valid_instances=torch.ones((2, 4), dtype=torch.bool),
        target_cell_ids=torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]]),
        target_arc_ids=torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
        target_senses=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]),
        valid_arcs=torch.ones((2, 4), dtype=torch.bool),
        inst_ids=torch.tensor([0, 1]),
        action_names=ACTION_NAMES,
        topology_key=(),
        check_target_arc_ids=torch.tensor([[11, 12, 13, 14]]),
        check_inst_ids=torch.tensor([0]),
        evaluate_targets=True,
    )
    timing._active_discrete_action_context = probe
    model = SimpleNamespace(
        _active_discrete_action_context=probe,
        params=SimpleNamespace(timing_action_pin_cap_gradient=1),
        _current_timing_surrogate_mode=lambda: "lut_only",
        pin_2_libpin_ids=lambda offset, data: (
            torch.ones(5, dtype=torch.bool),
            torch.tensor([0, 2, 15, 16, 1]),
        ),
        data_collections=SimpleNamespace(
            inst_libcell_offset=torch.zeros(2, dtype=torch.long),
            pin2node_map=torch.tensor([0, 0, 1, 1, 0]),
            pin_2_libpin_offset=torch.tensor([0, 2, 0, 1, 1]),
            end_points=torch.tensor([4]),
            cell_id_2_libpin_id_start=torch.tensor([0, 3, 6, 9, 12, 15, 17, 19, 21, 23, 25]),
            flat_lib_pin_rcap=torch.tensor(cap_table * 5),
            flat_lib_pin_fcap=torch.tensor(cap_table * 7),
        ),
    )
    expected = -2.1 * torch.stack(
        (
            ff_delay + 1.45 * ff_slew - ff_check + 1.9 * delta_b,
            buf_delay - 0.1 * buf_slew + 3.45 * delta_a,
        )
    )
    return timing, model, data, expected


def evaluate(circuit, beta, mode="hard"):
    timing, model, data, _ = circuit
    model._active_discrete_action_context.beta = beta
    timing.timing_aggregation_mode = mode
    zeros = torch.zeros(5, dtype=torch.float64)
    _, rise, fall = HoldRC(data).pin_caps(model, zeros)
    loads = tuple(torch.stack((zeros[0], cap[2], cap[2], cap[4], cap[4])) for cap in (rise, fall))
    result = EarlyPropagation(data).forward(timing, (zeros, zeros), (zeros, zeros), loads)
    loss = HoldObjective(2, 0.1, 0).loss(SimpleNamespace(hold=result))
    return result, loss


def test_all_four_min_directions_match_hand_chain_in_one_backward(action_circuit):
    beta = action_circuit[0]._active_discrete_action_context.beta
    result, loss = evaluate(action_circuit, beta)
    torch.testing.assert_close(result.endpoint_slack, torch.tensor([-5.37], dtype=torch.float64))
    (gradient,) = torch.autograd.grad(loss, beta)
    torch.testing.assert_close(gradient, action_circuit[3], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_four_direction_hold_chain_matches_finite_difference(action_circuit, mode):
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(evaluate(action_circuit, beta, mode)[1], beta)
    difference = torch.empty_like(beta)
    for node in range(2):
        for action in range(4):
            plus, minus = beta.detach().clone(), beta.detach().clone()
            plus[node, action], minus[node, action] = 1e-5, -1e-5
            difference[node, action] = (
                evaluate(action_circuit, plus, mode)[1] - evaluate(action_circuit, minus, mode)[1]
            ) / 2e-5
    torch.testing.assert_close(gradient, difference, rtol=2e-5, atol=2e-5)


def test_invalid_master_direction_has_no_hold_probe(action_circuit):
    context = action_circuit[0]._active_discrete_action_context
    context.valid_instances[:, 1] = False
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(evaluate(action_circuit, beta)[1], beta)
    expected = action_circuit[3].clone()
    expected[:, 1] = 0
    torch.testing.assert_close(gradient, expected)


def test_missing_target_hold_check_invalidates_the_instance_action(action_circuit):
    timing = action_circuit[0]
    timing.arcs_info.r_check_luts.valid_arc_mask[11] = False
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(evaluate(action_circuit, beta)[1], beta)
    context = timing._active_discrete_action_context
    assert not context.valid_instances[0, 0]
    assert context.action_scores(gradient)[0, 0] == 0


def test_report_only_hold_builds_no_action_backward_graph(action_circuit):
    context = action_circuit[0]._active_discrete_action_context
    context.evaluate_targets = False
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        result, loss = evaluate(action_circuit, beta)
    assert not result.context_slack.requires_grad and not loss.requires_grad
    assert beta.grad is None


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_non_unate_master_interpolation_precedes_early_path_reduction(mode):
    from dreamplace.ops.timing_propagation.early_actions import attach_cell_probes
    from dreamplace.ops.timing_propagation.early_propagation import _apply_cells, _cell_lanes

    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    timing = SimpleNamespace(
        timing_aggregation_mode=mode,
        timing_aggregation_tau_ps=2.0,
        _active_discrete_action_context=SimpleNamespace(
            beta=beta,
            evaluate_targets=True,
            target_arc_ids=torch.ones((1, 4), dtype=torch.long),
            target_senses=torch.zeros((1, 4), dtype=torch.long),
            valid_arcs=torch.ones((1, 4), dtype=torch.bool),
            valid_instances=torch.ones((1, 4), dtype=torch.bool),
            inst_ids=torch.zeros(1, dtype=torch.long),
        ),
    )
    timing.r_delay_entry = lambda cell, slew, cap, arc, node, **kw: torch.where(
        arc == 0, 20 + 10 * slew, 35 - 20 * slew
    )
    timing.f_delay_entry = timing.r_delay_entry
    timing.r_tran_entry = lambda cell, slew, cap, arc, node, **kw: torch.ones_like(slew)
    timing.f_tran_entry = timing.r_tran_entry
    arcs = torch.tensor([[0, 1, 0, 0, 0, 0]])
    arrivals = tuple(torch.tensor([0.0, torch.inf], dtype=torch.float64) for _ in range(2))
    slews = tuple(torch.tensor([v, torch.inf], dtype=torch.float64) for v in (0.1, 0.9))
    caps = tuple(torch.zeros(2, dtype=torch.float64) for _ in range(2))
    lanes = _cell_lanes(timing, arcs, slews, caps)
    lanes = attach_cell_probes(timing, arcs, lanes, slews, caps, torch.tensor([0]))
    value = _apply_cells(timing, lanes, arrivals, slews)[0][0][1]
    (gradient,) = torch.autograd.grad(value, beta)

    def reference(step):
        # Current rise branch wins at 21 ps; target fall branch wins at 17 ps.
        # Each branch interpolates separately, so the local hard derivative
        # is +12 ps, not the -4 ps difference between the two minima.
        branches = torch.tensor([21 + 12 * step, 29 - 12 * step], dtype=torch.float64)
        return branches.min() if mode == "hard" else -2 * torch.logsumexp(-branches / 2, 0)

    difference = (reference(1e-5) - reference(-1e-5)) / 2e-5
    torch.testing.assert_close(gradient, difference.expand_as(gradient), atol=1e-8, rtol=0)
