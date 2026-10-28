"""Independent scalar oracle for cap -> native RC -> LUT -> endpoint timing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.placeio_ecc.sparse_timing_schema import SparseTimingView
from dreamplace.ops.timing_propagation.action_cell_aat import apply_action_cell_aat
from dreamplace.ops.timing_propagation.discrete_actions import DiscreteActionContext
from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule
from sparse_fixtures import add_transition_masks

DTYPE = torch.float64


def test_branched_native_rc_cap_jacobian_matches_finite_difference():
    from dreamplace.ops.rc_timing.rc_timing import evaluate_rc_tree

    cap = torch.linspace(0.01, 0.02, 6, dtype=DTYPE, requires_grad=True)
    resistance = torch.tensor([0.0, 100.0, 200.0, 300.0, 0.0, 400.0], dtype=DTYPE)
    parent = torch.tensor([-1, 0, 1, 0, -1, 4], dtype=torch.int32)
    child_start = torch.tensor([0, 2, 3, 3, 3, 4, 4], dtype=torch.int32)
    children = torch.tensor([1, 3, 2, 5], dtype=torch.int32)
    topo = torch.arange(6, dtype=torch.int32)
    topo_start = torch.tensor([0, 4, 6], dtype=torch.int32)

    def forward(c):
        load, delay, _, _, impulse = evaluate_rc_tree(
            c, resistance, parent, child_start, children, topo, topo_start
        )
        return torch.cat((load, delay, impulse))

    assert torch.autograd.gradcheck(forward, (cap,), atol=1e-6, rtol=1e-6)


def _circuit(sense):
    # Three cells, each with a current master and four independent neighbours.
    coeff = torch.tensor(
        [
            [2.0, 0.7, 1.3],
            [1.4, 0.5, 1.1],
            [3.1, 0.9, 1.8],
            [2.7, 0.8, 1.6],
            [1.8, 0.6, 1.2],
        ],
        dtype=DTYPE,
    )
    delay = coeff.repeat(3, 1)
    slew = delay * torch.tensor([0.2, 0.6, 0.4], dtype=DTYPE)
    pin_cap = (
        torch.tensor(
            [
                [0.10, 0.02],
                [0.16, 0.03],
                [0.06, 0.01],
                [0.08, 0.025],
                [0.12, 0.015],
            ],
            dtype=DTYPE,
        )
        .repeat(3, 1)
        .flatten()
    )
    targets = torch.arange(15).reshape(3, 5)[:, 1:].contiguous()
    current = torch.tensor([0, 5, 10])
    base = torch.tensor(
        [[0, 1, 0, 0, sense, 0, 0, 0], [2, 3, 1, 5, -1, 0, 0, 0], [4, 5, 2, 10, 1, 0, 0, 0]]
    )
    tables = []
    for coefficients in (delay, delay * 1.1, slew, slew * 1.2):
        values = torch.stack(
            [
                coefficients[:, 0],
                coefficients[:, 0] + coefficients[:, 2],
                coefficients[:, 0] + 10 * coefficients[:, 1],
                coefficients[:, 0] + 10 * coefficients[:, 1] + coefficients[:, 2],
            ],
            dim=1,
        ).contiguous()
        tables.extend(
            (
                values,
                torch.tensor([[0.0, 10.0]] * 15, dtype=DTYPE),
                torch.tensor([[0.0, 1.0]] * 15, dtype=DTYPE),
                torch.full((15,), 2),
                torch.full((15,), 2),
            )
        )
    arrays = {
        "pin_state_start": np.arange(7),
        "state_pin": np.arange(6),
        "state_domain": np.zeros(6, dtype=np.int64),
        "state_tag": np.zeros(6, dtype=np.int64),
        "state_source_type": np.ones(6, dtype=np.int64),
        "state_level": np.arange(6),
        "state_slew_group": np.arange(6),
        "state_valid": np.ones((6, 2), dtype=np.uint8),
        "state_seed": np.array([[1, 1]] + [[0, 0]] * 5, dtype=np.uint8),
        "state_reference": np.array([[0.0, 0.0, 0.3, 0.5]] + [[-2e8, -2e8, 0.0, 0.0]] * 5),
        "net_edges": np.array([[0, 1, 2], [1, 3, 4]]),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]] * 2),
        "cell_edges": np.array([[0, 0, 1], [1, 2, 3], [2, 4, 5]]),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[3, 0, -1, 0, -1, 0, 1], [5, 0, -1, 0, -1, 0, 1]]),
        "check_values": np.zeros((2, 4)),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "fd", "fd")
    add_transition_masks(arrays, base)
    schedule = SparseTimingSchedule(view, base.numpy(), device="cpu", dtype=DTYPE)
    return schedule, base, tables, delay, slew, pin_cap, current, targets


def _reference(beta, sense, delay, slew, cap, current, targets):
    # This oracle uses neither custom backward nor the production action op.
    weights = torch.cat((1 - beta.sum(1, keepdim=True), beta), dim=1)
    d = (delay.reshape(3, 5, 3) * weights[..., None]).sum(1)
    s = (slew.reshape(3, 5, 3) * weights[..., None]).sum(1)
    c = (cap.reshape(3, 5, 2) * weights[..., None]).sum(1)
    arrival = torch.zeros(2, dtype=DTYPE)
    transition = torch.tensor([0.3, 0.5], dtype=DTYPE)
    endpoints = []
    for i, unate in enumerate((sense, -1, 1)):
        # Zero wire C, unit wire R: load=own output C + next input C.
        load = c[i, 1] + (c[i + 1, 0] if i < 2 else 0)
        incoming = transition if unate == 1 else transition.flip(0)
        arrival = (arrival if unate == 1 else arrival.flip(0)) + (
            d[i, 0] + d[i, 1] * incoming + d[i, 2] * load
        ) * torch.tensor([1.0, 1.1], dtype=DTYPE)
        transition = (s[i, 0] + s[i, 1] * incoming + s[i, 2] * load) * torch.tensor(
            [1.0, 1.2], dtype=DTYPE
        )
        if i > 0:
            endpoints.append(-arrival[0])
        if i < 2:
            wire_delay = c[i + 1, 0]
            arrival = arrival + wire_delay
            transition = torch.sqrt(transition.square() + wire_delay.square())
    slack = torch.stack(endpoints)
    return torch.stack((slack.clamp(max=0).sum(), slack.min()))


@pytest.mark.parametrize("sense", [1, -1])
def test_four_directions_match_independent_forward_and_native_rc(sense):
    # Run with the same patched native environment as BM64; no mock RC.
    from dreamplace.ops.rc_timing.rc_timing import RCTiming
    from dreamplace.PlaceObj import PlaceObj

    schedule, base, tables, delay, slew, cap, current, targets = _circuit(sense)
    beta = torch.zeros((3, 4), dtype=DTYPE, requires_grad=True)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.ones((3, 4), dtype=torch.bool),
        target_cell_ids=targets,
        target_arc_ids=targets,
        target_senses=base[:, 4, None].expand(3, 4),
        valid_arcs=torch.ones((3, 4), dtype=torch.bool),
        inst_ids=torch.arange(3),
        action_names=("size_up", "size_down", "vt_slowdown", "vt_speedup"),
        topology_key=(),
    )
    data = SimpleNamespace(
        pin2node_map=torch.arange(3).repeat_interleave(2),
        pin_2_libpin_offset=torch.tensor([0, 1] * 3),
        cell_id_2_libpin_id_start=torch.arange(0, 31, 2),
        flat_lib_pin_cap=cap,
        flat_lib_pin_rcap=cap,
        flat_lib_pin_fcap=cap,
    )
    owner = SimpleNamespace(
        params=SimpleNamespace(timing_action_pin_cap_gradient=1),
        _active_discrete_action_context=context,
        _action_target_pin_ids=lambda *args: PlaceObj._action_target_pin_ids(owner, *args),
        _invalidate_action_instances=lambda *args: PlaceObj._invalidate_action_instances(
            owner, *args
        ),
    )
    initial_cap = cap.reshape(15, 2)[current].flatten()
    pin_caps = PlaceObj._apply_discrete_action_pin_caps(
        owner, initial_cap, initial_cap, initial_cap, data
    )
    # A placement-only detach optimization must not cut the action cap edge.
    assert PlaceObj._timing_rc_cap_input_tensor(owner, pin_caps[0]) is pin_caps[0]
    rc = RCTiming(r_unit=1.0, c_unit=0.0, dbu=1.0)
    _, loads, delays, _, _, impulses = rc(
        torch.arange(6, dtype=DTYPE),
        torch.zeros(6, dtype=DTYPE),
        torch.arange(6, dtype=torch.int32),
        torch.tensor([0, 1, 3, 5, 6], dtype=torch.int32),
        torch.tensor([-1, -1, 1, -1, 3, -1], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
        torch.tensor([1, 3], dtype=torch.int32),
        *pin_caps,
    )

    def cell_updates(level, lanes, caps):
        rows = level.base_cells
        return apply_action_cell_aat(
            level.cell,
            (*lanes, *caps),
            tables,
            smooth=False,
            alpha=1.0,
            target_arc_ids=targets[rows],
            target_senses=context.target_senses[rows],
            valid_arcs=context.valid_arcs[rows],
            inst_ids=rows,
            beta=beta,
            context=context,
            global_rows=rows,
            defer_target_queries=True,
        )

    lanes = schedule.propagate(
        tables,
        tuple(loads[k] for k in ("rise", "fall")),
        tuple(delays[k] for k in ("rise", "fall")),
        tuple(impulses[k] for k in ("rise", "fall")),
        cell_updates=cell_updates,
    )
    result = schedule.reduce_endpoints(lanes, torch.zeros(2, dtype=DTYPE))
    value = torch.stack((result.tns, result.wns))

    def reference(b):
        return _reference(b, sense, delay, slew, cap, current, targets)

    torch.testing.assert_close(value, reference(beta.detach()), atol=1e-12, rtol=0.0)
    for metric in range(2):
        gradient = torch.autograd.grad(value[metric], beta, retain_graph=True)[0]
        finite_difference = torch.empty_like(beta)
        for i in range(3):
            for a in range(4):
                h = torch.zeros_like(beta)
                h[i, a] = 1e-6
                finite_difference[i, a] = (reference(h)[metric] - reference(-h)[metric]) / 2e-6
        torch.testing.assert_close(gradient, finite_difference, atol=1e-8, rtol=1e-7)
    assert not torch.allclose(gradient[:, 0], -gradient[:, 1])
