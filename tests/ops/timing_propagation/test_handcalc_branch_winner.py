"""Hand-calculated two-branch winner crossing with four action directions.

The circuit is intentionally linear inside every LUT:

    PI_A --D--> S.input A \
                           S --T--> E1, E2
    PI_B -------> S.input B /

S is the sized instance.  Its two input arcs have current delays 15 ps and
40 ps, so the B branch wins by 15 ps at beta=0.  S's output slew is the max
of the two arc slews; T's delay is 5 + 10*input_slew.  The two endpoints have
required times 40 and 35 ps, so both are negative throughout this test.

At beta=0, with tau=2 ps,

    w = exp((25-40)/2) / (1 + exp((25-40)/2))
    dTNS/dbeta = -2 * w * (target_delay - 15)

The four target delays are 5, 45, 10 and 35 ps.  The columns are therefore
independent one-sided derivatives, not +/- pairs.  size_down also changes
the target arc slew from .2 to .8; at beta=1 it wins both the arrival LSE and
the output-slew max, so its complete TNS change is much larger than the
beta=0 linear prediction.  The test checks the hand values, one backward,
and that finite-action deviation.
"""

import math

import numpy as np
import torch
from dreamplace.ops.placeio_ecc.sparse_timing_schema import SparseTimingView
from dreamplace.ops.timing_propagation.action_cell_aat import apply_action_cell_aat
from dreamplace.ops.timing_propagation.discrete_actions import DiscreteActionContext
from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule
from sparse_fixtures import add_transition_masks

DTYPE = torch.float64
NP_DTYPE = np.float64
TAU = 2.0
ACTIONS = ("size_up", "size_down", "vt_speedup", "vt_slowdown")
TARGET_DELAYS = (5.0, 45.0, 10.0, 35.0)


def _table_rows(delays, slews):
    """Build the 20 LUT tensors, with two values per row for interpolation."""
    assert len(delays) == len(slews) == 8

    def group(values, trans, caps, trans_dims, cap_dims):
        return (
            torch.tensor(values, dtype=DTYPE),
            torch.tensor(trans, dtype=DTYPE),
            torch.tensor(caps, dtype=DTYPE),
            torch.tensor(trans_dims, dtype=torch.long),
            torch.tensor(cap_dims, dtype=torch.long),
        )

    # T is the only row whose delay depends on input slew.  All other rows
    # use the first value and the trans_dims=1/cap_dims=1 interpolation.
    delay_values = []
    delay_trans = []
    delay_caps = []
    delay_tdims = []
    delay_cdims = []
    for row, value in enumerate(delays):
        if row == 3:
            delay_values.append([value, value + 10.0])
            delay_trans.append([0.0, 1.0])
            delay_tdims.append(2)
        else:
            delay_values.append([value, 0.0])
            delay_trans.append([0.0, 0.0])
            delay_tdims.append(1)
        delay_caps.append([0.0])
        delay_cdims.append(1)
    slew_values = [[value, 0.0] for value in slews]
    slew_trans = [[0.0, 0.0] for _ in slews]
    slew_caps = [[0.0] for _ in slews]
    slew_dims = [1] * len(slews)
    delay = group(delay_values, delay_trans, delay_caps, delay_tdims, delay_cdims)
    slew = group(slew_values, slew_trans, slew_caps, slew_dims, slew_dims)
    return [tensor for table in (delay, delay, slew, slew) for tensor in table]


def _schedule():
    # Pins: 0 PI_A, 1 PI_B, 2 D out, 3 S in A, 4 S in B,
    # 5 S out, 6 T in, 7 T out, 8 E1, 9 E2.
    arrays = {
        "pin_state_start": np.arange(11, dtype=np.int64),
        "state_pin": np.arange(10, dtype=np.int64),
        "state_domain": np.zeros(10, dtype=np.int64),
        "state_tag": np.zeros(10, dtype=np.int64),
        "state_source_type": np.ones(10, dtype=np.int64),
        "state_level": np.array([0, 0, 1, 2, 2, 3, 4, 5, 6, 6], dtype=np.int64),
        "state_slew_group": np.arange(10, dtype=np.int64),
        "state_valid": np.ones((10, 2), dtype=np.uint8),
        "state_seed": np.array([[1, 1], [1, 1]] + [[0, 0]] * 8, dtype=np.uint8),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 8,
            dtype=NP_DTYPE,
        ),
        "net_edges": np.array(
            [[1, 2, 3], [2, 1, 4], [3, 5, 6], [4, 7, 8], [5, 7, 9]],
            dtype=np.int64,
        ),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]] * 5, dtype=NP_DTYPE),
        "cell_edges": np.array([[0, 0, 2], [1, 3, 5], [2, 4, 5], [3, 6, 7]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array(
            [[8, 0, -1, 0, -1, 0, 1], [9, 0, -1, 0, -1, 0, 1]],
            dtype=np.int64,
        ),
        "check_values": np.array(
            [[40.0, 0.0, 0.0, 0.0], [35.0, 0.0, 0.0, 0.0]],
            dtype=NP_DTYPE,
        ),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "branch-winner", "branch-winner")
    # Flat base arcs: D, S.A, S.B, T.  Each instance owns one or two arcs.
    base = np.array(
        [
            [0, 2, 0, 0, 1, 0, 0, 0],
            [3, 5, 1, 1, 1, 0, 0, 0],
            [4, 5, 1, 2, 1, 0, 0, 0],
            [6, 7, 2, 3, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    add_transition_masks(arrays, base)
    schedule = SparseTimingSchedule(
        view,
        base,
        device="cpu",
        dtype=DTYPE,
        aggregation_mode="smooth",
        aggregation_tau_ps=TAU,
    )
    return schedule, torch.tensor(base, dtype=torch.long)


def _tables():
    # Current rows: D=10, S.A=15/slew .2, S.B=40/slew .4, T=slew-coupled.
    # Target rows: size_up=5/.15, size_down=45/.8, vt_up=10/.25,
    # vt_down=35/.6.
    return _table_rows(
        [10.0, 15.0, 40.0, 5.0, 5.0, 45.0, 10.0, 35.0],
        [0.1, 0.2, 0.4, 0.1, 0.15, 0.8, 0.25, 0.6],
    )


def _context(beta):
    # Instance 0 is D (unsized), 1 is S (the sized cell), 2 is T.
    target_rows = torch.tensor(
        [
            [0, 0, 0, 0],
            [4, 5, 6, 7],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    valid_arcs = torch.tensor(
        [
            [False, False, False, False],
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor([[False] * 4, [True] * 4, [False] * 4]),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=target_rows,
        target_senses=torch.ones((4, 4), dtype=torch.long),
        valid_arcs=valid_arcs,
        inst_ids=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        action_names=ACTIONS,
        topology_key=(),
    )
    context.validate(device=torch.device("cpu"), dtype=DTYPE)
    return context


def _forward(schedule, base, tables, beta_values, *, evaluate_targets=False):
    beta = torch.tensor(beta_values, dtype=DTYPE, requires_grad=True)
    context = _context(beta)
    context.evaluate_targets = evaluate_targets
    caps = torch.zeros(10, dtype=DTYPE)
    delays = torch.zeros(10, dtype=DTYPE)
    impulses = torch.zeros(10, dtype=DTYPE)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            smooth=True,
            alpha=1.0 / TAU,
            target_arc_ids=context.target_arc_ids.index_select(0, rows),
            target_senses=context.target_senses.index_select(0, rows),
            valid_arcs=context.valid_arcs.index_select(0, rows),
            inst_ids=context.inst_ids.index_select(0, rows),
            beta=beta,
            context=context,
            global_rows=rows,
            defer_target_queries=True,
        )

    lanes = schedule.propagate(
        tables,
        (caps, caps.clone()),
        (delays, delays.clone()),
        (impulses.clone(), impulses.clone()),
        smooth=True,
        tau_ps=TAU,
        cell_updates=cell_updates,
    )
    result = schedule.reduce_endpoints(lanes, torch.zeros(2, dtype=DTYPE))
    return beta, lanes, result


def test_four_directional_gradients_match_hand_lse_and_crossing():
    schedule, base = _schedule()
    tables = _tables()
    beta, lanes, result = _forward(schedule, base, tables, [[0.0] * 4, [0.0] * 4, [0.0] * 4])

    # Beta=0 hand values: S output = LSE(25, 40), slew=max(.2,.4),
    # T delay=9, endpoint arrival=49.0011; the two required times make
    # TNS=-23.0022.
    weight = math.exp(-7.5) / (1.0 + math.exp(-7.5))
    s_out = 40.0 + TAU * math.log1p(math.exp(-7.5))
    t_delay = 5.0 + 10.0 * 0.4
    torch.testing.assert_close(lanes[0][5], torch.tensor(s_out, dtype=DTYPE), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(lanes[2][5], torch.tensor(0.4, dtype=DTYPE), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        lanes[0][8],
        torch.tensor(s_out + t_delay, dtype=DTYPE),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.tns,
        torch.tensor(-(2.0 * (s_out + t_delay) - 75.0), dtype=DTYPE),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(lanes[0][5].detach()) < 40.01  # the B branch wins at beta=0

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [
                -2.0 * weight * (TARGET_DELAYS[0] - 15.0),
                -2.0 * weight * (TARGET_DELAYS[1] - 15.0),
                -2.0 * weight * (TARGET_DELAYS[2] - 15.0),
                -2.0 * weight * (TARGET_DELAYS[3] - 15.0),
            ],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    gradient = torch.autograd.grad(result.tns, beta)[0]
    torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=0.0)

    # The negative direction is a separate target row, not the negation of
    # the positive one.
    assert expected[1, 1] < 0.0
    assert not torch.allclose(gradient[1, 1], -gradient[1, 0], atol=1e-8, rtol=0.0)


def test_full_size_down_crosses_winner_and_exceeds_local_prediction():
    schedule, base = _schedule()
    tables = _tables()
    beta0, _lanes0, result0 = _forward(schedule, base, tables, [[0.0] * 4, [0.0] * 4, [0.0] * 4])
    gradient = torch.autograd.grad(result0.tns, beta0)[0]

    beta1, lanes1, result1 = _forward(
        schedule,
        base,
        tables,
        [[0.0] * 4, [0.0, 1.0, 0.0, 0.0], [0.0] * 4],
        evaluate_targets=True,
    )
    actual = float((result1.tns - result0.tns).detach())
    local = float(gradient[1, 1].detach())

    # Full size_down: A candidate 55, B candidate 40, so LSE is ~55.0011;
    # target slew .8 wins, T delay 13; both endpoints worsen by 38 ps.
    torch.testing.assert_close(
        lanes1[0][5],
        torch.tensor(55.0 + TAU * math.log1p(math.exp(-7.5)), dtype=DTYPE),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(lanes1[2][5], torch.tensor(0.8, dtype=DTYPE), atol=1e-12, rtol=0.0)
    assert float(lanes1[0][5].detach()) > 55.0  # A/size_down now wins
    torch.testing.assert_close(
        torch.tensor(actual, dtype=DTYPE),
        torch.tensor(-38.0, dtype=DTYPE),
        atol=1e-9,
        rtol=0.0,
    )
    # The local derivative is only a fraction of a picosecond, whereas the
    # complete action is -38 ps.  This is the finite-step failure mode.
    assert local < 0.0
    assert abs(actual - local) > 30.0
    assert abs(actual - local) > 1000.0 * abs(local)
