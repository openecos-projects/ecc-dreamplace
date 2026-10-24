"""Hand-calculated PI -> driver -> sized cell -> endpoint chain.

The oracle is first-order at beta=0, with slew held fixed:

    delay(D) = 10 + 1000 * load
    load     = cap(S)
    delay(S) = 20, size_up -> 15, size_down -> 28
    cap(S)   = 0.002, size_up -> 0.005, size_down -> 0.001

    size_up:   load +3 ps, master -5 ps, arrival -2 ps, TNS improvement +2 ps
    size_down: load -1 ps, master +8 ps, arrival +7 ps, TNS improvement -7 ps

size_down is not -size_up. These two numbers separate a broken load edge (+5),
a broken arrival pass-through (0), a doubled delay edge, and a sign error.
"""

import numpy as np
import torch

from dreamplace.ops.placeio_ecc.sparse_timing_schema import SparseTimingView
from dreamplace.ops.timing_propagation.discrete_actions import DiscreteActionContext
from dreamplace.ops.timing_propagation.action_cell_aat import apply_action_cell_aat
from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule


def _lut(values, trans, caps, trans_dims, cap_dims):
    return (
        torch.tensor(values, dtype=torch.float64),
        torch.tensor(trans, dtype=torch.float64),
        torch.tensor(caps, dtype=torch.float64),
        torch.tensor(trans_dims, dtype=torch.long),
        torch.tensor(cap_dims, dtype=torch.long),
    )


def _tables():
    # Rows: D, S current, S size_up, S size_down. Cap axis [0, 0.01] makes
    # D delay exactly 10 + 1000*cap inside the bracket. S delays are flat.
    delay = _lut(
        [[10.0, 20.0], [20.0, 20.0], [15.0, 15.0], [28.0, 28.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0, 0.01]] * 4,
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    )
    slew = _lut(
        [[0.1], [0.1], [0.1], [0.1]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    return [tensor for group in (delay, delay, slew, slew) for tensor in group]


def _schedule():
    # states/pins: 0 PI, 1 driver output, 2 sized-cell input, 3 endpoint
    arrays = {
        "pin_state_start": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "state_pin": np.array([0, 1, 2, 3], dtype=np.int64),
        "state_domain": np.zeros(4, dtype=np.int64),
        "state_tag": np.zeros(4, dtype=np.int64),
        "state_source_type": np.ones(4, dtype=np.int64),
        "state_level": np.array([0, 1, 2, 3], dtype=np.int64),
        "state_slew_group": np.array([0, 1, 2, 3], dtype=np.int64),
        "state_valid": np.ones((4, 2), dtype=np.uint8),
        "state_seed": np.array([[1, 1], [0, 0], [0, 0], [0, 0]], dtype=np.uint8),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 3,
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2]], dtype=np.int64),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        "cell_edges": np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[3, 0, -1, 0, -1, 0, 1]], dtype=np.int64),
        "check_values": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "handcalc", "handcalc")
    base = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 1, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    return SparseTimingSchedule(view, base, device="cpu", dtype=torch.float64), torch.tensor(base)


def _context(beta):
    # Pin 2 is the sized cell input. Its cap delta lands on the driver pin.
    rise = torch.zeros((4, 4), dtype=torch.float64)
    rise[2, 0] = 0.003
    rise[2, 1] = -0.001
    fall = torch.zeros_like(rise)
    pin_net = torch.tensor([1, 0, 0, 2], dtype=torch.long)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [[True, False, False, False], [True, True, False, False]]
        ),
        target_cell_ids=torch.zeros((2, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[0, 0, 0, 0], [2, 3, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.ones((2, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [[True, False, False, False], [True, True, False, False]]
        ),
        inst_ids=torch.tensor([0, 1], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=fall,
        drive_load_node=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        drive_load_pin_net=pin_net,
        drive_load_net_count=3,
        drive_load_driver_mask=torch.tensor([False, True, False, False]),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    return context


def test_handcalc_chain_matches_one_sided_load_and_master_deltas():
    schedule, base = _schedule()
    tables = _tables()
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    context = _context(beta)
    base_caps = torch.tensor([0.0, 0.002, 0.0, 0.0], dtype=torch.float64)
    cap_rise, cap_fall = context.apply_drive_load(base_caps, base_caps.clone())
    zeros = torch.zeros(4, dtype=torch.float64)
    schedule.discrete_action_context = context

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (cap_rise, cap_fall),
        (zeros, zeros),
        (zeros, zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    # beta=0 arrival: driver 12 ps, endpoint 32 ps.
    torch.testing.assert_close(lanes[0][1], torch.tensor(12.0, dtype=torch.float64))
    torch.testing.assert_close(lanes[0][3], torch.tensor(32.0, dtype=torch.float64))
    result = schedule.reduce_endpoints(lanes, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(result.tns, torch.tensor(-32.0, dtype=torch.float64))
    assert torch.isfinite(result.tns)

    score = -result.tns
    grad = torch.autograd.grad(score, beta)[0]
    predicted = -grad
    # Columns are size_up, size_down, vt_speedup, vt_slowdown.
    # Driver row is a no-op master change and must stay 0 while its load
    # Jacobian still serves the sized cell.
    torch.testing.assert_close(
        predicted,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, -7.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        atol=1e-6,
        rtol=0.0,
    )


def _slew_tables():
    # Row 0: driver. delay = 10+1000*cap, slew = 0.1+100*cap.
    # Row 1: sized cell. delay = 20+10*slew.
    # Row 2: sized size_up. delay = 15.
    # Row 3: side fanout. delay = 10*slew.
    # Row 4: driver slew-only alternate. same delay, slew = 0.2.
    delay = _lut(
        [
            [10.0, 20.0],
            [20.0, 30.0],
            [15.0, 15.0],
            [0.0, 10.0],
            [10.0, 20.0],
        ],
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.01],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.01],
        ],
        [1, 2, 1, 2, 1],
        [2, 1, 1, 1, 2],
    )
    slew = _lut(
        [
            [0.1, 1.1],
            [0.1, 0.1],
            [0.1, 0.1],
            [0.1, 0.1],
            [0.2, 0.2],
        ],
        [[0.0, 0.0]] * 5,
        [
            [0.0, 0.01],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1],
    )
    return [tensor for group in (delay, delay, slew, slew) for tensor in group]


def _slew_schedule():
    # 0 PI, 1 driver out, 2 sized in, 3 sized out, 4 fanout in, 5 fanout out.
    arrays = {
        "pin_state_start": np.arange(7, dtype=np.int64),
        "state_pin": np.arange(6, dtype=np.int64),
        "state_domain": np.zeros(6, dtype=np.int64),
        "state_tag": np.zeros(6, dtype=np.int64),
        "state_source_type": np.ones(6, dtype=np.int64),
        "state_level": np.array([0, 1, 2, 3, 2, 3], dtype=np.int64),
        "state_slew_group": np.arange(6, dtype=np.int64),
        "state_valid": np.ones((6, 2), dtype=np.uint8),
        "state_seed": np.array(
            [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.uint8
        ),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 5,
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2], [1, 1, 4]], dtype=np.int64),
        "net_threshold_adjustments": np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64
        ),
        "cell_edges": np.array(
            [[0, 0, 1], [1, 2, 3], [2, 4, 5]], dtype=np.int64
        ),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array(
            [[3, 0, -1, 0, -1, 0, 1], [5, 0, -1, 0, -1, 0, 1]],
            dtype=np.int64,
        ),
        "check_values": np.zeros((2, 4), dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "slew", "slew")
    base = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 1, 1, 0, 0, 0],
            [4, 5, 0, 3, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    return SparseTimingSchedule(view, base, device="cpu", dtype=torch.float64), torch.tensor(base)


def test_replaced_cell_keeps_base_point_slew_jacobian():
    schedule, base = _slew_schedule()
    tables = _slew_tables()
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((6, 4), dtype=torch.float64)
    rise[2, 0] = 0.003
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[0, 4, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.ones((3, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        drive_load_pin_net=torch.tensor([1, 0, 0, 2, 0, 3], dtype=torch.long),
        drive_load_net_count=4,
        drive_load_driver_mask=torch.tensor(
            [False, True, False, False, False, False]
        ),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    base_caps = torch.tensor(
        [0.0, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    cap_rise, cap_fall = context.apply_drive_load(base_caps, base_caps.clone())
    zeros = torch.zeros(6, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (cap_rise, cap_fall),
        (zeros, zeros),
        (zeros, zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    # slew = 0.3. sized delay = 23. fanout delay = 3. driver delay = 12.
    torch.testing.assert_close(lanes[2][1], torch.tensor(0.3, dtype=torch.float64))
    torch.testing.assert_close(lanes[0][3], torch.tensor(35.0, dtype=torch.float64))
    torch.testing.assert_close(lanes[0][5], torch.tensor(15.0, dtype=torch.float64))
    result = schedule.reduce_endpoints(lanes, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(result.tns, torch.tensor(-50.0, dtype=torch.float64))

    predicted = -torch.autograd.grad(-result.tns, beta)[0]
    # size_up of S: two driver delays +6, S master -8, two slew penalties +6.
    # S's current-LUT Jacobian is evaluated at beta=0. Net TNS change is -4.
    # Driver's own slew step (0.3 -> 0.2) still moves both receivers: +2.
    torch.testing.assert_close(
        predicted,
        torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [-4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        atol=1e-5,
        rtol=0.0,
    )


def test_elmore_cap_is_not_added_to_driver_load_again():
    """One pin-cap change moves both net delay and driver load, once each.

    A 1000 ohm wire and a 1000 ps/pF cell give the same +3 ps from a
    +0.003 pF size_up. The master change is -5 ps. Arrival therefore moves
    by +1 ps and TNS by -1 ps. Adding the cap a second time would make the
    cell term +6 ps and the TNS change -4 ps.
    """

    schedule, base = _schedule()
    tables = _tables()
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    context = _context(beta)
    context.rc_includes_action_cap = True
    schedule.discrete_action_context = context
    cap_delta = 0.003 * beta[1, 0]
    cap = torch.tensor([0.0, 0.002, 0.0, 0.0], dtype=torch.float64)
    cap = cap.clone()
    cap[1] = cap[1] + cap_delta
    unchanged, _ = context.maybe_apply_drive_load(cap, cap.clone())
    torch.testing.assert_close(unchanged, cap)
    delay = torch.zeros(4, dtype=torch.float64)
    delay[2] = 1000.0 * (0.002 + cap_delta)
    zeros = torch.zeros(4, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (cap, cap.clone()),
        (delay, delay.clone()),
        (zeros, zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    # beta=0: driver cell 12 ps + wire 2 ps + sized-cell delay 20 ps.
    torch.testing.assert_close(lanes[0][3], torch.tensor(34.0, dtype=torch.float64))
    predicted = -torch.autograd.grad(-schedule.reduce_endpoints(
        lanes, torch.zeros(1, dtype=torch.float64)
    ).tns, beta)[0]
    torch.testing.assert_close(
        predicted[1, 0],
        torch.tensor(-1.0, dtype=torch.float64),
        atol=1e-5,
        rtol=0.0,
    )


def _predict(schedule, tables, beta, context, caps, delays, impulses, base):
    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        caps,
        delays,
        impulses,
        smooth=False,
        cell_updates=cell_updates,
    )
    result = schedule.reduce_endpoints(lanes, torch.zeros(2, dtype=torch.float64))
    predicted = -torch.autograd.grad(-result.tns, beta)[0]
    return lanes, predicted


def test_own_impulse_edge_reaches_current_lut():
    """A negative-unate cell keeps its base-point LUT impulse sensitivity.

    Fall impulse is 0.01 + 0.02 * beta at the sized pin. At beta=0 the fall
    slew is sqrt(0.3^2 + 0.01). The current rise delay is 20 + 10 * slew and
    the size_up master is the flat 15, so the directional TNS change is

        5 + 10*sqrt(0.1) - 10*0.02/(2*sqrt(0.1)) = 7.8460498942 ps.

    This is the derivative of (1-beta)*D0(slew(beta))+beta*D1(slew(beta)).
    """

    schedule, base = _slew_schedule()
    base = base.clone()
    base[1, 4] = -1
    tables = _slew_tables()
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((6, 4), dtype=torch.float64)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.tensor(
            [[1, 1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.long,
        ),
        valid_arcs=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        drive_load_pin_net=torch.tensor([1, 0, 0, 2, 0, 3], dtype=torch.long),
        drive_load_net_count=4,
        drive_load_driver_mask=torch.tensor(
            [False, True, False, False, False, False]
        ),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    caps = torch.tensor([0.0, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    zeros = torch.zeros(6, dtype=torch.float64)
    fall_impulse = zeros.clone()
    fall_impulse[2] = 0.01 + 0.02 * beta[1, 0]
    lanes, predicted = _predict(
        schedule,
        tables,
        beta,
        context,
        (caps, caps.clone()),
        (zeros, zeros.clone()),
        (zeros.clone(), fall_impulse),
        base,
    )
    # 12 ps driver + 20 ps intercept + 10*sqrt(0.1) from the fall slew.
    torch.testing.assert_close(
        lanes[0][3],
        torch.tensor(32.0 + 10.0 * (0.1 ** 0.5), dtype=torch.float64),
    )
    torch.testing.assert_close(
        predicted[1, 0],
        torch.tensor(7.8460498941519835, dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )


def test_fanout_keeps_impulse_and_driver_delay():
    """Both sinks retain impulse, driver slew and upstream delay derivatives.

    size_up moves the sized pin by +0.003 pF and both sink impulses by +0.02.
    size_down moves the pin by -0.001 pF and both impulses by -0.04, with no
    master change. The two columns are not negatives of each other.
    """

    schedule, base = _slew_schedule()
    tables = _slew_tables()
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((6, 4), dtype=torch.float64)
    rise[2, 0] = 0.003
    rise[2, 1] = -0.001
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.ones((3, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        drive_load_pin_net=torch.tensor([1, 0, 0, 2, 0, 3], dtype=torch.long),
        drive_load_net_count=4,
        drive_load_driver_mask=torch.tensor(
            [False, True, False, False, False, False]
        ),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    caps = torch.tensor([0.0, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    cap_rise, cap_fall = context.apply_drive_load(caps, caps.clone())
    zeros = torch.zeros(6, dtype=torch.float64)
    shared = 0.01 + 0.02 * beta[1, 0] - 0.04 * beta[1, 1]
    impulse = zeros.clone()
    impulse[2] = shared
    impulse[4] = shared
    _lanes, predicted = _predict(
        schedule,
        tables,
        beta,
        context,
        (cap_rise, cap_fall),
        (zeros, zeros.clone()),
        (impulse, zeros.clone()),
        base,
    )

    slew = 0.1 ** 0.5
    gain_driver = 0.3 / slew
    gain_impulse = 1.0 / (2.0 * slew)

    def column(cap_delta, impulse_delta, master):
        driver_delay = 1000.0 * cap_delta
        driver_slew = 100.0 * cap_delta
        sink = gain_driver * driver_slew + gain_impulse * impulse_delta
        fanout = driver_delay + 10.0 * sink
        sized = driver_delay + master + 10.0 * sink
        return -(sized + fanout)

    master_up = 15.0 - (20.0 + 10.0 * slew)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [
                column(0.003, 0.02, master_up),
                column(-0.001, -0.04, 0.0),
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(predicted, expected, atol=1e-8, rtol=0.0)
    # The size_down impulse slope is not the negation of size_up.
    assert predicted[1, 1].abs() > 1.0
    assert not torch.allclose(
        predicted[1, 1], -predicted[1, 0], atol=1e-3, rtol=0.0
    )


def test_impulse_cap_chain_matches_direct_beta_derivative():
    """An explicit cap -> impulse edge reaches the cell LUT through slew."""

    schedule, base = _slew_schedule()
    base = base.clone()
    base[1, 4] = -1
    tables = _slew_tables()
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((6, 4), dtype=torch.float64)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.tensor(
            [[1, 1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.long,
        ),
        valid_arcs=torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        drive_load_pin_net=torch.tensor([1, 0, 0, 2, 0, 3], dtype=torch.long),
        drive_load_net_count=4,
        drive_load_driver_mask=torch.tensor(
            [False, True, False, False, False, False]
        ),
    )
    node_cap = torch.zeros(6, dtype=torch.float64)
    node_cap[2] = 0.003 * beta[1, 0]
    jac = torch.zeros(6, dtype=torch.float64)
    jac[2] = 0.02 / 0.003
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    caps = torch.tensor([0.0, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    zeros = torch.zeros(6, dtype=torch.float64)
    fall_impulse = zeros.clone()
    fall_impulse[2] = 0.01 + jac[2] * node_cap[2]
    _lanes, predicted = _predict(
        schedule,
        tables,
        beta,
        context,
        (caps, caps.clone()),
        (zeros, zeros.clone()),
        (zeros.clone(), fall_impulse),
        base,
    )
    torch.testing.assert_close(
        predicted[1, 0],
        torch.tensor(7.8460498941519835, dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )



def _output_load_schedule():
    # 0 PI, 1 driver out, 2 sized in, 3 sized out, 4 endpoint.
    arrays = {
        "pin_state_start": np.arange(6, dtype=np.int64),
        "state_pin": np.arange(5, dtype=np.int64),
        "state_domain": np.zeros(5, dtype=np.int64),
        "state_tag": np.zeros(5, dtype=np.int64),
        "state_source_type": np.ones(5, dtype=np.int64),
        "state_level": np.arange(5, dtype=np.int64),
        "state_slew_group": np.arange(5, dtype=np.int64),
        "state_valid": np.ones((5, 2), dtype=np.uint8),
        "state_seed": np.array(
            [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.uint8
        ),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 4,
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int64),
        "net_threshold_adjustments": np.ones((2, 4), dtype=np.float64)
        * np.array([0.0, 0.0, 1.0, 1.0]),
        "cell_edges": np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[4, 0, -1, 0, -1, 0, 1]], dtype=np.int64),
        "check_values": np.zeros((1, 4), dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "load", "load")
    base = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 1, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    return SparseTimingSchedule(view, base, device="cpu", dtype=torch.float64), torch.tensor(base)


def _output_load_tables():
    # Driver delay is flat 10 ps. Sized-cell delay is 20 + 1000*load.
    # size_up is flat 15 ps and size_down is flat 28 ps, both at the current load.
    delay = _lut(
        [[10.0, 10.0], [20.0, 30.0], [15.0, 15.0], [28.0, 28.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0, 0.01], [0.0, 0.01], [0.0, 0.01], [0.0, 0.01]],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    )
    slew = _lut(
        [[0.1], [0.1], [0.1], [0.1]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    return [tensor for group in (delay, delay, slew, slew) for tensor in group]


def test_own_output_load_reaches_current_lut_and_net():
    """Both the current LUT and output net retain the output-cap derivative.

    At beta=0 the sized-cell load is 0.002 pF, so its delay is 22 ps. The
    output wire is 500 ohm and adds 1 ps. Driver delay is 10 ps.

        size_up:   master 15-22 = -7 ps, wire +2 ps, own LUT load +4 ps
        size_down: master 28-22 = +6 ps, wire -0.5 ps, own LUT load -1 ps

    The base-point chain rule includes all three terms. TNS therefore moves
    by +1 ps and -4.5 ps.
    """

    schedule, base = _output_load_schedule()
    tables = _output_load_tables()
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [[False, False, False, False], [True, True, False, False]]
        ),
        target_cell_ids=torch.zeros((2, 4), dtype=torch.long),
        target_arc_ids=torch.tensor([[0, 0, 0, 0], [2, 3, 0, 0]], dtype=torch.long),
        target_senses=torch.ones((2, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [[False, False, False, False], [True, True, False, False]]
        ),
        inst_ids=torch.tensor([0, 1], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=torch.zeros((5, 4), dtype=torch.float64),
        drive_load_fcap=torch.zeros((5, 4), dtype=torch.float64),
        drive_load_node=torch.zeros(5, dtype=torch.long),
        drive_load_pin_net=torch.zeros(5, dtype=torch.long),
        drive_load_net_count=1,
        drive_load_driver_mask=torch.zeros(5, dtype=torch.bool),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    cap_change = 0.004 * beta[1, 0] - 0.001 * beta[1, 1]
    load = torch.zeros(5, dtype=torch.float64)
    load[3] = 0.002 + cap_change
    fall = torch.full((5,), 0.002, dtype=torch.float64)
    delay = torch.zeros(5, dtype=torch.float64)
    delay[4] = 500.0 * (0.002 + cap_change)
    zeros = torch.zeros(5, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (load, fall),
        (delay, zeros.clone()),
        (zeros.clone(), zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    torch.testing.assert_close(lanes[0][4], torch.tensor(33.0, dtype=torch.float64))
    predicted = -torch.autograd.grad(
        -schedule.reduce_endpoints(lanes, torch.zeros(1, dtype=torch.float64)).tns,
        beta,
    )[0]
    torch.testing.assert_close(
        predicted,
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, -4.5, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        atol=1e-6,
        rtol=0.0,
    )



def test_physical_pin_ids_do_not_follow_state_ids():
    """State index and physical pin id are different in the real timing graph.

    The electrical values live at physical pins 5..8. The +2/-7 ps hand
    result has to survive that indirection; indexing the load Jacobian by
    the state slot would drop the upstream load term and return +5/-8.
    """

    arrays = {
        "pin_state_start": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "state_pin": np.array([5, 6, 7, 8], dtype=np.int64),
        "state_domain": np.zeros(4, dtype=np.int64),
        "state_tag": np.zeros(4, dtype=np.int64),
        "state_source_type": np.ones(4, dtype=np.int64),
        "state_level": np.array([0, 1, 2, 3], dtype=np.int64),
        "state_slew_group": np.array([0, 1, 2, 3], dtype=np.int64),
        "state_valid": np.ones((4, 2), dtype=np.uint8),
        "state_seed": np.array([[1, 1], [0, 0], [0, 0], [0, 0]], dtype=np.uint8),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 3,
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2]], dtype=np.int64),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        "cell_edges": np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[3, 0, -1, 0, -1, 0, 1]], dtype=np.int64),
        "check_values": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "offset", "offset")
    base = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    schedule = SparseTimingSchedule(view, base.numpy(), device="cpu", dtype=torch.float64)
    tables = _tables()
    beta = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((9, 4), dtype=torch.float64)
    rise[7, 0] = 0.003
    rise[7, 1] = -0.001
    pin_net = torch.zeros(9, dtype=torch.long)
    pin_net[5] = 1
    pin_net[6] = 0
    pin_net[7] = 0
    pin_net[8] = 2
    node = torch.zeros(9, dtype=torch.long)
    node[7] = 1
    node[8] = 1
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [[True, False, False, False], [True, True, False, False]]
        ),
        target_cell_ids=torch.zeros((2, 4), dtype=torch.long),
        target_arc_ids=torch.tensor([[0, 0, 0, 0], [2, 3, 0, 0]], dtype=torch.long),
        target_senses=torch.ones((2, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [[True, False, False, False], [True, True, False, False]]
        ),
        inst_ids=torch.tensor([0, 1], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=node,
        drive_load_pin_net=pin_net,
        drive_load_net_count=3,
        drive_load_driver_mask=torch.tensor(
            [False, False, False, False, False, False, True, False, False]
        ),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    base_caps = torch.zeros(9, dtype=torch.float64)
    base_caps[6] = 0.002
    cap_rise, cap_fall = context.apply_drive_load(base_caps, base_caps.clone())
    zeros = torch.zeros(9, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (cap_rise, cap_fall),
        (zeros, zeros.clone()),
        (zeros.clone(), zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    torch.testing.assert_close(lanes[0][3], torch.tensor(32.0, dtype=torch.float64))
    predicted = -torch.autograd.grad(
        -schedule.reduce_endpoints(lanes, torch.zeros(1, dtype=torch.float64)).tns,
        beta,
    )[0]
    torch.testing.assert_close(
        predicted,
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [2.0, -7.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        atol=1e-6,
        rtol=0.0,
    )



def test_two_input_cell_sums_both_driver_loads_and_both_arcs():
    """A two-input cell is one instance with two arcs, not one pin.

    size_up raises pin A by 0.003 pF and pin B by 0.001 pF. Each driver
    delay moves by 1000 ps/pF, and each arc master moves from 20 ps to 15 ps.
    The two endpoints give

        dTNS/dbeta = -((3 - 5) + (1 - 5)) = +6 ps.

    Counting only one input or one arc cannot produce +6.
    """

    arrays = {
        "pin_state_start": np.arange(9, dtype=np.int64),
        "state_pin": np.arange(8, dtype=np.int64),
        "state_domain": np.zeros(8, dtype=np.int64),
        "state_tag": np.zeros(8, dtype=np.int64),
        "state_source_type": np.ones(8, dtype=np.int64),
        "state_level": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "state_slew_group": np.arange(8, dtype=np.int64),
        "state_valid": np.ones((8, 2), dtype=np.uint8),
        "state_seed": np.array(
            [[1, 1], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
            dtype=np.uint8,
        ),
        "state_reference": np.array(
            [
                [0.0, 0.0, 0.1, 0.1],
                [-2e8, -2e8, 0.0, 0.0],
                [-2e8, -2e8, 0.0, 0.0],
                [-2e8, -2e8, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.1],
                [-2e8, -2e8, 0.0, 0.0],
                [-2e8, -2e8, 0.0, 0.0],
                [-2e8, -2e8, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2], [1, 5, 6]], dtype=np.int64),
        "net_threshold_adjustments": np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64
        ),
        "cell_edges": np.array(
            [[0, 0, 1], [1, 4, 5], [2, 2, 3], [3, 6, 7]], dtype=np.int64
        ),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array(
            [[3, 0, -1, 0, -1, 0, 1], [7, 0, -1, 0, -1, 0, 1]],
            dtype=np.int64,
        ),
        "check_values": np.zeros((2, 4), dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "xor", "xor")
    base = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [4, 5, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 2, 1, 0, 0, 0],
            [6, 7, 0, 2, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    schedule = SparseTimingSchedule(
        view, base.numpy(), device="cpu", dtype=torch.float64
    )
    delay = _lut(
        [[10.0, 20.0], [10.0, 20.0], [20.0, 20.0], [15.0, 15.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0, 0.01], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]],
        [1, 1, 1, 1],
        [2, 2, 1, 1],
    )
    slew = _lut(
        [[0.1], [0.1], [0.1], [0.1]],
        [[0.0], [0.0], [0.0], [0.0]],
        [[0.0], [0.0], [0.0], [0.0]],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    tables = [tensor for group in (delay, delay, slew, slew) for tensor in group]
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    rise = torch.zeros((8, 4), dtype=torch.float64)
    rise[2, 0] = 0.003
    rise[6, 0] = 0.001
    pin_net = torch.tensor([1, 0, 0, 2, 3, 1, 1, 4], dtype=torch.long)
    node = torch.tensor([0, 0, 2, 2, 1, 1, 2, 2], dtype=torch.long)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 0, 0, 0],
                [3, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        target_senses=torch.ones((4, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=rise,
        drive_load_fcap=torch.zeros_like(rise),
        drive_load_node=node,
        drive_load_pin_net=pin_net,
        drive_load_net_count=5,
        drive_load_driver_mask=torch.tensor(
            [False, True, False, False, False, True, False, False]
        ),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    caps = torch.zeros(8, dtype=torch.float64)
    caps[1] = 0.002
    caps[5] = 0.002
    cap_rise, cap_fall = context.apply_drive_load(caps, caps.clone())
    zeros = torch.zeros(8, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (cap_rise, cap_fall),
        (zeros, zeros.clone()),
        (zeros.clone(), zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    torch.testing.assert_close(lanes[0][3], torch.tensor(32.0, dtype=torch.float64))
    torch.testing.assert_close(lanes[0][7], torch.tensor(32.0, dtype=torch.float64))
    predicted = -torch.autograd.grad(
        -schedule.reduce_endpoints(lanes, torch.zeros(2, dtype=torch.float64)).tns,
        beta,
    )[0]
    torch.testing.assert_close(
        predicted[2, 0],
        torch.tensor(6.0, dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )


def test_slew_step_propagates_through_multiple_cells():
    """The timing chain carries slew changes through every downstream cell.

    A size_up changes only A's output slew, 0.4 -> 0.2. B's delay ignores
    slew and its slew LUT is the identity. C's delay is 10*slew. The
    immediate receiver contributes no delay change. Its slew identity still
    moves C by -2 ps and TNS by +2 ps, both for an infinitesimal step and a
    complete move in this linear circuit.
    """

    delay = _lut(
        [
            [10.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
            [0.0, 10.0],
        ],
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        [[0.0], [0.0], [0.0], [0.0]],
        [2, 2, 2, 2],
        [1, 1, 1, 1],
    )
    slew = _lut(
        [
            [0.4, 0.4],
            [0.2, 0.2],
            [0.0, 1.0],
            [0.1, 0.1],
        ],
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        [[0.0], [0.0], [0.0], [0.0]],
        [2, 2, 2, 2],
        [1, 1, 1, 1],
    )
    tables = [tensor for group in (delay, delay, slew, slew) for tensor in group]
    arrays = {
        "pin_state_start": np.arange(7, dtype=np.int64),
        "state_pin": np.arange(6, dtype=np.int64),
        "state_domain": np.zeros(6, dtype=np.int64),
        "state_tag": np.zeros(6, dtype=np.int64),
        "state_source_type": np.ones(6, dtype=np.int64),
        "state_level": np.arange(6, dtype=np.int64),
        "state_slew_group": np.arange(6, dtype=np.int64),
        "state_valid": np.ones((6, 2), dtype=np.uint8),
        "state_seed": np.array(
            [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.uint8
        ),
        "state_reference": np.array(
            [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 5,
            dtype=np.float64,
        ),
        "net_edges": np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int64),
        "net_threshold_adjustments": np.ones((2, 4), dtype=np.float64),
        "cell_edges": np.array([[0, 0, 1], [1, 2, 3], [2, 4, 5]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[5, 0, -1, 0, -1, 0, 1]], dtype=np.int64),
        "check_values": np.zeros((1, 4), dtype=np.float64),
    }
    # Threshold scale is 1 on slew, 0 on the arrival-slew cross term.
    arrays["net_threshold_adjustments"][:, 0] = 0.0
    arrays["net_threshold_adjustments"][:, 1] = 0.0
    view = SparseTimingView(arrays, ("sys",), ("sys",), (), "slewregen", "slewregen")
    base = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [2, 3, 0, 2, 1, 0, 0, 0],
            [4, 5, 0, 3, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    schedule = SparseTimingSchedule(
        view, base, device="cpu", dtype=torch.float64
    )
    base_t = torch.tensor(base)
    beta = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    context = DiscreteActionContext(
        beta=beta,
        valid_instances=torch.tensor(
            [
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        target_cell_ids=torch.zeros((3, 4), dtype=torch.long),
        target_arc_ids=torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        target_senses=torch.ones((3, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        inst_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        action_names=("size_up", "size_down", "vt_speedup", "vt_slowdown"),
        topology_key=(),
        drive_load_rcap=torch.zeros((6, 4), dtype=torch.float64),
        drive_load_fcap=torch.zeros((6, 4), dtype=torch.float64),
        drive_load_node=torch.zeros(6, dtype=torch.long),
        drive_load_pin_net=torch.zeros(6, dtype=torch.long),
        drive_load_net_count=1,
        drive_load_driver_mask=torch.zeros(6, dtype=torch.bool),
    )
    context.validate(device=torch.device("cpu"), dtype=torch.float64)
    schedule.discrete_action_context = context
    zeros = torch.zeros(6, dtype=torch.float64)

    def cell_updates(level, lanes, local_caps):
        rows = level.base_cells
        arcs = torch.cat((level.cell[:, :2], base_t[rows, 2:]), dim=1)
        return apply_action_cell_aat(
            arcs,
            (*lanes, *local_caps),
            tables,
            False,
            1.0,
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
        (),
        (zeros, zeros.clone()),
        (zeros, zeros.clone()),
        (zeros, zeros.clone()),
        smooth=False,
        cell_updates=cell_updates,
    )
    result = schedule.reduce_endpoints(lanes, torch.zeros(1, dtype=torch.float64))
    predicted = -torch.autograd.grad(-result.tns, beta)[0]
    torch.testing.assert_close(
        predicted[0, 0],
        torch.tensor(2.0, dtype=torch.float64),
        atol=1e-5,
        rtol=0.0,
    )
