import pytest
import torch
from dreamplace.ops.timing_propagation import lut_entry_2d_op, timing_propagation_cpp


def _tables(dtype):
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    trans = torch.tensor([[0.1, 1.0]], dtype=dtype)
    caps = torch.tensor([[0.1, 1.0]], dtype=dtype)
    dims = torch.tensor([2], dtype=torch.long)
    return [value for _ in range(4) for value in (values, trans, caps, dims, dims)]


def _amax(base, index, values):
    return torch.scatter_reduce(
        base.clone(), 0, index, values, reduce="amax", include_self=True
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("smooth_pairwise", (False, True))
def test_cell_aat_hard_level_matches_existing_native_rows(dtype, smooth_pairwise):
    arcs = torch.tensor([[0, 1, 0, 0, 1], [0, 2, 0, 0, -1]], dtype=torch.long)
    pin_rAAT = torch.tensor([0.0, 10.0, -2.0], dtype=dtype)
    pin_fAAT = torch.tensor([0.0, 5.0, 3.0], dtype=dtype)
    pin_rtran = torch.tensor([0.2, 0.3, 0.4], dtype=dtype)
    pin_ftran = torch.tensor([0.25, 0.35, 0.45], dtype=dtype)
    cap_rise = torch.tensor([0.2, 0.4, 0.5], dtype=dtype)
    cap_fall = torch.tensor([0.3, 0.5, 0.6], dtype=dtype)
    tables = _tables(dtype)

    reference_updates, _ = lut_entry_2d_op._select_module(pin_rAAT).cell_aat_forward(
        arcs,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        cap_rise,
        cap_fall,
        tables,
        smooth_pairwise,
        20.0,
    )
    reference = tuple(
        _amax(base, arcs[:, 1], reference_updates[:, lane])
        for lane, base in enumerate((pin_rAAT, pin_fAAT, pin_rtran, pin_ftran))
    )
    actual = timing_propagation_cpp.cell_aat_hard_level(
        arcs,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        cap_rise,
        cap_fall,
        tables,
        smooth_pairwise,
        20.0,
    )

    for actual_lane, reference_lane in zip(actual[:4], reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=1e-6, atol=1e-6)
    for actual_lane, reference_lane in zip(
        actual[4:], (reference_updates[:, lane] for lane in range(4, 8)), strict=True
    ):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=1e-6, atol=1e-6)


def test_net_aat_hard_level_matches_existing_native_rows():
    dtype = torch.float64
    arcs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    pin_rAAT = torch.tensor([0.0, 10.0, -2.0], dtype=dtype)
    pin_fAAT = torch.tensor([0.0, 5.0, 3.0], dtype=dtype)
    pin_rtran = torch.tensor([0.2, 0.3, 0.4], dtype=dtype)
    pin_ftran = torch.tensor([0.25, 0.35, 0.45], dtype=dtype)
    delay_rise = torch.tensor([0.0, 0.7, 0.8], dtype=dtype)
    delay_fall = torch.tensor([0.0, 0.9, 1.0], dtype=dtype)
    impulse_rise = torch.tensor([0.0, 0.03, 0.04], dtype=dtype)
    impulse_fall = torch.tensor([0.0, 0.05, 0.06], dtype=dtype)

    reference_updates, _ = lut_entry_2d_op._select_module(pin_rAAT).net_aat_forward(
        arcs,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        delay_rise,
        delay_fall,
        impulse_rise,
        impulse_fall,
        1e-8,
    )
    reference = tuple(
        _amax(base, arcs[:, 1], reference_updates[:, lane])
        for lane, base in enumerate((pin_rAAT, pin_fAAT, pin_rtran, pin_ftran))
    )
    actual = timing_propagation_cpp.net_aat_hard_level(
        arcs,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        delay_rise,
        delay_fall,
        impulse_rise,
        impulse_fall,
        1e-8,
    )

    for actual_lane, reference_lane in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=0, atol=0)


def test_cell_rat_hard_level_matches_unate_and_non_unate_min_reduction():
    dtype = torch.float64
    arcs = torch.tensor(
        [[0, 2, 0, 0, 1], [1, 2, 0, 0, -1], [2, 3, 0, 0, 0]],
        dtype=torch.long,
    )
    pin_rRAT = torch.tensor([100.0, 90.0, 80.0, 70.0], dtype=dtype)
    pin_fRAT = torch.tensor([110.0, 95.0, 85.0, 75.0], dtype=dtype)
    delays = tuple(
        torch.tensor(values, dtype=dtype)
        for values in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0])
    )
    sense = arcs[:, 4]
    outputs = arcs[:, 1]
    rise_updates = torch.where(
        sense == 1,
        pin_rRAT[outputs] - delays[0],
        torch.where(
            sense == -1,
            pin_fRAT[outputs] - delays[2],
            torch.minimum(pin_rRAT[outputs] - delays[0], pin_fRAT[outputs] - delays[2]),
        ),
    )
    fall_updates = torch.where(
        sense == 1,
        pin_fRAT[outputs] - delays[3],
        torch.where(
            sense == -1,
            pin_rRAT[outputs] - delays[1],
            torch.minimum(pin_rRAT[outputs] - delays[1], pin_fRAT[outputs] - delays[3]),
        ),
    )
    reference = (
        torch.scatter_reduce(
            pin_rRAT.clone(), 0, arcs[:, 0], rise_updates, reduce="amin", include_self=True
        ),
        torch.scatter_reduce(
            pin_fRAT.clone(), 0, arcs[:, 0], fall_updates, reduce="amin", include_self=True
        ),
    )
    actual = timing_propagation_cpp.cell_rat_hard_level(arcs, pin_rRAT, pin_fRAT, *delays)
    for actual_lane, reference_lane in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=0, atol=0)


def test_net_rat_hard_level_matches_sink_to_driver_min_reduction():
    dtype = torch.float64
    arcs = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.long)
    pin_rRAT = torch.tensor([100.0, 90.0, 80.0, 70.0], dtype=dtype)
    pin_fRAT = torch.tensor([110.0, 95.0, 85.0, 75.0], dtype=dtype)
    delay_rise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    delay_fall = torch.tensor([0.0, 4.0, 5.0, 6.0], dtype=dtype)
    reference = (
        torch.scatter_reduce(
            pin_rRAT.clone(),
            0,
            arcs[:, 0],
            pin_rRAT[arcs[:, 1]] - delay_rise[arcs[:, 1]],
            reduce="amin",
            include_self=True,
        ),
        torch.scatter_reduce(
            pin_fRAT.clone(),
            0,
            arcs[:, 0],
            pin_fRAT[arcs[:, 1]] - delay_fall[arcs[:, 1]],
            reduce="amin",
            include_self=True,
        ),
    )
    actual = timing_propagation_cpp.net_rat_hard_level(
        arcs, pin_rRAT, pin_fRAT, delay_rise, delay_fall
    )
    for actual_lane, reference_lane in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=0, atol=0)


def test_setup_rat_hard_level_matches_sense_selection_and_validity():
    dtype = torch.float64
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    trans = torch.tensor([[0.1, 1.0]], dtype=dtype)
    caps = torch.tensor([[0.1, 1.0]], dtype=dtype)
    dims = torch.tensor([2], dtype=torch.long)
    tables = [value for _ in range(2) for value in (values, trans, caps, dims, dims)]
    arcs = torch.tensor(
        [[0, 2, 0, 0, 1], [1, 2, 0, 0, -1], [0, 3, 0, 0, 0]],
        dtype=torch.long,
    )
    pin_rRAT = torch.tensor([100.0, 90.0, 80.0, 70.0], dtype=dtype)
    pin_fRAT = torch.tensor([110.0, 95.0, 85.0, 75.0], dtype=dtype)
    clk_rtran = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=dtype)
    clk_ftran = torch.tensor([0.25, 0.35, 0.45, 0.55], dtype=dtype)
    data_rtran = torch.tensor([0.3, 0.4, 0.5, 0.6], dtype=dtype)
    data_ftran = torch.tensor([0.35, 0.45, 0.55, 0.65], dtype=dtype)
    valid = torch.tensor([[True, True], [True, False], [True, True]])

    def lut_value(t, c):
        t = (t - 0.1) / 0.9
        c = (c - 0.1) / 0.9
        return 1.0 * (1.0 - t) * (1.0 - c) + 2.0 * (1.0 - t) * c + 3.0 * t * (1.0 - c) + 4.0 * t * c

    rise_rr = torch.stack(
        [
            lut_value(clk_rtran[0], data_rtran[2]),
            lut_value(clk_ftran[1], data_rtran[2]),
            lut_value(clk_rtran[0], data_rtran[3]),
        ]
    )
    fall_ff = torch.stack(
        [
            lut_value(clk_ftran[0], data_ftran[2]),
            lut_value(clk_rtran[1], data_ftran[2]),
            lut_value(clk_ftran[0], data_ftran[3]),
        ]
    )
    rise_fr = torch.stack(
        [
            lut_value(clk_ftran[0], data_rtran[2]),
            lut_value(clk_ftran[1], data_rtran[2]),
            lut_value(clk_ftran[0], data_rtran[3]),
        ]
    )
    fall_rf = torch.stack(
        [
            lut_value(clk_rtran[0], data_ftran[2]),
            lut_value(clk_rtran[1], data_ftran[2]),
            lut_value(clk_rtran[0], data_ftran[3]),
        ]
    )
    sense = arcs[:, 4]
    rise_setup = torch.where(
        sense == 1,
        rise_rr,
        torch.where(
            sense == -1,
            rise_fr,
            torch.logsumexp(torch.stack((rise_rr, rise_fr)) * 2.0, dim=0) / 2.0,
        ),
    )
    fall_setup = torch.where(
        sense == 1,
        fall_ff,
        torch.where(
            sense == -1,
            fall_rf,
            torch.logsumexp(torch.stack((fall_ff, fall_rf)) * 2.0, dim=0) / 2.0,
        ),
    )
    rise_updates = torch.where(valid[:, 0], pin_rRAT[arcs[:, 1]] - rise_setup, torch.inf)
    fall_updates = torch.where(valid[:, 1], pin_fRAT[arcs[:, 1]] - fall_setup, torch.inf)
    reference = (
        torch.scatter_reduce(
            pin_rRAT.clone(), 0, arcs[:, 1], rise_updates, reduce="amin", include_self=True
        ),
        torch.scatter_reduce(
            pin_fRAT.clone(), 0, arcs[:, 1], fall_updates, reduce="amin", include_self=True
        ),
    )
    actual = timing_propagation_cpp.setup_rat_hard_level(
        arcs,
        pin_rRAT,
        pin_fRAT,
        clk_rtran,
        clk_ftran,
        data_rtran,
        data_ftran,
        tables,
        valid,
        smooth_pairwise=True,
        alpha=2.0,
    )
    for actual_lane, reference_lane in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=1e-12, atol=1e-12)


def test_recovery_rat_hard_level_matches_capture_minus_live_and_validity():
    dtype = torch.float64
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    trans = torch.tensor([[0.1, 1.0]], dtype=dtype)
    caps = torch.tensor([[0.1, 1.0]], dtype=dtype)
    dims = torch.tensor([2], dtype=torch.long)
    tables = [value for _ in range(2) for value in (values, trans, caps, dims, dims)]
    arcs = torch.tensor(
        [[0, 2, 0, 0, 1, 1, 3, 0], [1, 2, 0, 0, 1, -1, 3, 0]],
        dtype=torch.long,
    )
    pin_rRAT = torch.tensor([100.0, 90.0, 80.0], dtype=dtype)
    pin_fRAT = torch.tensor([110.0, 95.0, 85.0], dtype=dtype)
    pin_rtran = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)
    pin_ftran = torch.tensor([0.25, 0.35, 0.55], dtype=dtype)
    clock_rtran = torch.tensor([0.2, 0.3, 0.4], dtype=dtype)
    clock_ftran = torch.tensor([0.25, 0.35, 0.45], dtype=dtype)
    check_data_rtran = torch.tensor([0.45, 0.5], dtype=dtype)
    check_data_ftran = torch.tensor([0.5, 0.55], dtype=dtype)
    valid = torch.tensor([[True, True], [True, False]])

    def lut_value(t, c):
        t = (t - 0.1) / 0.9
        c = (c - 0.1) / 0.9
        return 1.0 * (1.0 - t) * (1.0 - c) + 2.0 * (1.0 - t) * c + 3.0 * t * (1.0 - c) + 4.0 * t * c

    reference_r = torch.tensor(
        [
            lut_value(clock_rtran[0], check_data_rtran[0]),
            lut_value(clock_rtran[1], check_data_rtran[1]),
        ],
        dtype=dtype,
    )
    live_r = torch.tensor(
        [
            lut_value(clock_rtran[0], pin_rtran[2]),
            lut_value(clock_rtran[1], pin_rtran[2]),
        ],
        dtype=dtype,
    )
    reference_f = torch.tensor(
        [
            lut_value(clock_ftran[0], check_data_ftran[0]),
            lut_value(clock_ftran[1], check_data_ftran[1]),
        ],
        dtype=dtype,
    )
    live_f = torch.tensor(
        [
            lut_value(clock_ftran[0], pin_ftran[2]),
            lut_value(clock_ftran[1], pin_ftran[2]),
        ],
        dtype=dtype,
    )
    endpoint_pins = arcs[:, 1]
    rise_candidates = torch.where(
        valid[:, 0], pin_rRAT[endpoint_pins] + reference_r - live_r, torch.inf
    )
    fall_candidates = torch.where(
        valid[:, 1], pin_fRAT[endpoint_pins] + reference_f - live_f, torch.inf
    )
    reference = (
        torch.scatter_reduce(
            pin_rRAT.clone(), 0, endpoint_pins, rise_candidates, reduce="amin", include_self=False
        ),
        torch.scatter_reduce(
            pin_fRAT.clone(), 0, endpoint_pins, fall_candidates, reduce="amin", include_self=False
        ),
    )
    actual = timing_propagation_cpp.recovery_rat_hard_level(
        arcs,
        pin_rRAT,
        pin_fRAT,
        pin_rtran,
        pin_ftran,
        clock_rtran,
        clock_ftran,
        check_data_rtran,
        check_data_ftran,
        tables,
        valid,
    )
    for actual_lane, reference_lane in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=1e-12, atol=1e-12)


def test_clk2q_hard_level_matches_edge_and_sense_selection():
    dtype = torch.float64
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    trans = torch.tensor([[0.1, 1.0]], dtype=dtype)
    caps = torch.tensor([[0.1, 1.0]], dtype=dtype)
    dims = torch.tensor([2], dtype=torch.long)
    tables = [value for _ in range(4) for value in (values, trans, caps, dims, dims)]
    arcs = torch.tensor(
        [[0, 1, 0, 0, 1, 1], [1, 2, 0, 0, -1, 0], [0, 2, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    pin_rAAT = torch.tensor([0.0, -100.0, -100.0], dtype=dtype)
    pin_fAAT = torch.tensor([0.0, -100.0, -100.0], dtype=dtype)
    pin_rtran = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    pin_ftran = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    cap_rise = torch.tensor([0.2, 0.4, 0.6], dtype=dtype)
    cap_fall = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
    clock_rtran = torch.tensor([0.2, 0.35], dtype=dtype)
    clock_ftran = torch.tensor([0.25, 0.4], dtype=dtype)

    def lut_value(t, c):
        t = (t - 0.1) / 0.9
        c = (c - 0.1) / 0.9
        return 1.0 * (1.0 - t) * (1.0 - c) + 2.0 * (1.0 - t) * c + 3.0 * t * (1.0 - c) + 4.0 * t * c

    def smooth(left, right):
        return torch.logsumexp(torch.stack((left, right)) * 2.0, dim=0) / 2.0

    # First positive-unate rising edge.
    r1 = lut_value(clock_rtran[0], cap_rise[1])
    t1 = lut_value(clock_rtran[0], cap_rise[1])
    # Second negative-unate both-edge arc.
    r2 = lut_value(clock_ftran[1], cap_rise[2])
    f2 = lut_value(clock_rtran[1], cap_fall[2])
    tr2 = lut_value(clock_ftran[1], cap_rise[2])
    tf2 = lut_value(clock_rtran[1], cap_fall[2])
    # Third non-unate both-edge arc, sharing output pin 2.
    rr3 = lut_value(clock_rtran[0], cap_rise[2])
    fr3 = lut_value(clock_ftran[0], cap_rise[2])
    rf3 = lut_value(clock_rtran[0], cap_fall[2])
    ff3 = lut_value(clock_ftran[0], cap_fall[2])
    trr3 = lut_value(clock_rtran[0], cap_rise[2])
    tfr3 = lut_value(clock_ftran[0], cap_rise[2])
    trf3 = lut_value(clock_rtran[0], cap_fall[2])
    tff3 = lut_value(clock_ftran[0], cap_fall[2])
    expected_r = torch.tensor([0.0, r1, max(r2, smooth(rr3, fr3)),], dtype=dtype)
    expected_f = torch.tensor([0.0, -torch.inf, max(f2, smooth(rf3, ff3))], dtype=dtype)
    expected_rs = torch.tensor([0.0, t1, max(tr2, smooth(trr3, tfr3))], dtype=dtype)
    expected_fs = torch.tensor([0.0, 0.0, max(tf2, smooth(trf3, tff3))], dtype=dtype)
    actual = timing_propagation_cpp.clk2q_hard_level(
        arcs,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        cap_rise,
        cap_fall,
        clock_rtran,
        clock_ftran,
        tables,
        smooth_pairwise=True,
        alpha=2.0,
    )
    for actual_lane, reference_lane in zip(
        actual, (expected_r, expected_f, expected_rs, expected_fs), strict=True
    ):
        torch.testing.assert_close(actual_lane, reference_lane, rtol=1e-12, atol=1e-12)
