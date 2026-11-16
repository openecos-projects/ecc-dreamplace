"""Unit and integration tests for AutoDMP R0_reference Star-Pin-Pair TDP in ecc-dreamplace."""

from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch

from dreamplace.PlaceObj import PreconditionOp
from dreamplace.NesterovAcceleratedGradientOptimizer import (
    NesterovAcceleratedGradientOptimizer,
)
from dreamplace.flows.gp.continuation import MidplaceSchedule
from dreamplace.flows.gp.star_tdp_runtime import StarTdpRuntime
from dreamplace.flows.gp.timing_carriers import (
    diff_tdp_controls_direct_timing_objective,
    diff_tdp_enabled,
    diff_tdp_uses_direct_loss,
    diff_tdp_uses_star_pin_pair,
    should_use_live_timing_loss,
)
from dreamplace.flows.config.specs import configure_timing_placement_carrier
from dreamplace.ops.placeio_ecc.sparse_timing_schema import SparseTimingView
from dreamplace.ops.rc_timing.rc_timing import RCTiming
from dreamplace.ops.star_pin_pair import (
    InputAccumulator,
    SlopeCalibration,
    StarPinPair,
    clamp_load_length_weight,
    clock_rowsum_hessian,
    launch_capture_clock_pair_adjoint,
    rowsum_hessian,
    transport_momentum,
)
from dreamplace.ops.timing_propagation.at_merge_temperature import AtMergeTemperature
from dreamplace.ops.timing_propagation.lcb_ff_timing import (
    apply_lcb_ff_to_compact_clk_pins,
    build_lcb_ff_clock_star_domain,
    compute_lcb_ff_clock_state,
)
from dreamplace.ops.timing_propagation.timing_propagation import (
    LUTS_INFO,
    TimingPropagation,
)


def _finite_diff_grad(fn, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    grad = torch.zeros_like(x, dtype=torch.float64)
    base = x.detach().clone().double()
    for idx in range(base.numel()):
        xp = base.clone()
        xm = base.clone()
        xp[idx] += eps
        xm[idx] -= eps
        fp = float(fn(xp.to(dtype=x.dtype)).detach().double().item())
        fm = float(fn(xm.to(dtype=x.dtype)).detach().double().item())
        grad[idx] = (fp - fm) / (2.0 * eps)
    return grad


def test_star_pin_pair_handles_arcless_clock_nets_and_matches_fd_gradient():
    # Net 0: 2-pin data net (0 -> 1)
    # Net 1: 3-pin clock/arcless net (2, 3, 4) with driver_pin_by_net = -1 and empty arc slice
    # Net 2: 3-pin data net (5 -> 6, 7)
    drivers = torch.tensor([0, -1, 5], dtype=torch.int32)
    flat_net_arcs = torch.tensor([[0, 1], [5, 6], [5, 7]], dtype=torch.long)
    flat_net_arcs_start = torch.tensor([0, 1, 1, 3], dtype=torch.long)

    star = StarPinPair(
        net2driver_pin_map=drivers,
        timing_net_flat_arcs=flat_net_arcs,
        timing_net_flat_arcs_start=flat_net_arcs_start,
        num_pins=8,
        r_unit=0.25,
        c_unit=0.15,
        scale_factor=1.0,
        dbu=100.0,
    )
    assert star.pair_count == 3
    assert star.sink_pin_ids.tolist() == [1, 6, 7]

    pin_pos = torch.tensor(
        [
            10.0, 110.0, 0.0, 20.0, 40.0, 50.0, 180.0, 60.0,
            20.0, 90.0, 0.0, 10.0, 30.0, 40.0, 40.0, 190.0,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    caps = {
        k: torch.tensor([0.0, 1.2, 0.5, 0.5, 0.5, 0.0, 0.8, 1.5], dtype=torch.float64)
        for k in ("generic", "rise", "fall")
    }
    raw_k = torch.tensor([2.5, 1.0, 3.2], dtype=torch.float32)
    raw_b = torch.tensor([-0.9, 0.4, 0.7], dtype=torch.float32)
    pub_b, curv_b = clamp_load_length_weight(raw_b, mode="clamp")
    assert pub_b.tolist() == pytest.approx([0.0, 0.4, 0.7])
    assert curv_b.tolist() == pytest.approx([-0.9, 0.4, 0.7])

    cand = star.prepare_snapshot(
        raw_weight=raw_k,
        length_um=star.length_um(pin_pos.detach()),
        load_length_weight=pub_b,
        curvature_load_length_weight=curv_b,
    )
    star.publish_snapshot(cand)

    for distance_model in ("l1", "wa"):
        p = pin_pos.detach().clone().requires_grad_(True)
        gamma = 25.0 if distance_model == "wa" else None
        val = star.structured_delay_objective(
            p, base_caps=caps, distance_model=distance_model, smoothing_gamma=gamma
        )
        analytic = torch.autograd.grad(val, p)[0].double()
        numeric = _finite_diff_grad(
            lambda q: star.structured_delay_objective(
                q, base_caps=caps, distance_model=distance_model, smoothing_gamma=gamma
            ),
            p,
            eps=1e-4,
        )
        torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_input_accumulator_and_slope_calibration():
    acc = InputAccumulator(gain=0.2, half_life=0.0, mode="accumulate")
    g1 = {
        "delay": {"rise": torch.tensor([1.0, 2.0]), "fall": torch.tensor([0.0, 4.0])},
        "cap": {"rise": torch.tensor([0.5, 0.0]), "fall": torch.tensor([0.5, 1.0])},
    }
    c1 = acc.prepare(g1, identity=("db", 2, 1), pin_count=2, iteration=100)
    acc.commit(c1)
    torch.testing.assert_close(
        c1["values"]["delay"]["rise"], torch.tensor([0.2, 0.4])
    )
    c2 = acc.prepare(g1, identity=("db", 2, 1), pin_count=2, iteration=115)
    acc.commit(c2)
    torch.testing.assert_close(
        c2["values"]["delay"]["rise"], torch.tensor([0.4, 0.8])
    )

    # 2-pin net (net 0: pins 0 -> 1) + arcless clock net (net 1: pins 2, 3) + 3-pin colinear net (net 2: 4 -> 5 -> 6)
    flat_netpin = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)
    netpin_start = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
    pin2net = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
    drivers = torch.tensor([0, -1, 4], dtype=torch.int32)
    flat_net_arcs = torch.tensor([[0, 1], [4, 5], [4, 6]], dtype=torch.long)
    flat_net_arcs_start = torch.tensor([0, 1, 1, 3], dtype=torch.long)

    star = StarPinPair(
        net2driver_pin_map=drivers,
        timing_net_flat_arcs=flat_net_arcs,
        timing_net_flat_arcs_start=flat_net_arcs_start,
        num_pins=7,
        r_unit=0.2,
        c_unit=0.1,
        scale_factor=1.0,
        dbu=1.0,
    )
    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    placedb = SimpleNamespace(rawdb=None)
    dc = SimpleNamespace(
        flat_net2pin_map=flat_netpin,
        flat_net2pin_start_map=netpin_start,
        pin2net_map=pin2net,
        clock_net_indices=[],
    )
    params = SimpleNamespace(ignore_net_degree=100, deterministic_flag=True)
    cal = SlopeCalibration(
        placedb=placedb,
        params=params,
        data_collections=dc,
        star=star,
        elmore_delay_op=elmore_op,
    )
    # Pin positions: net 0 has (0,0)->(10,0); net 2 has (0,0)->(10,0)->(20,0) sharing trunk [0,10]!
    pin_pos = torch.tensor(
        [0.0, 10.0, 0.0, 5.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    caps = torch.tensor([0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0], dtype=torch.float32)
    raw_k = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
    rescaled, meta = cal.rescale_weights(
        pin_pos=pin_pos, base_caps_generic=caps, raw_weight=raw_k, iteration=50
    )
    # Net 0 is a 2-pin net: Star Elmore == Steiner Elmore, so s_0 == 1.0 -> K[0] unchanged!
    assert float(rescaled[0].item()) == pytest.approx(2.0, rel=1e-5)
    # Net 2 shares trunk (0->10), so Steiner teacher delay slope > Star independent branch slope (s_2 > 1.0)!
    assert float(rescaled[1].item()) > 1.05
    assert meta["nets_with_k"] == 2


def _build_launch_capture_sparse_timing(
    num_pins: int = 6,
    clock_pins=(4, 5),
    inst_arcs_rows=((4, 0, 0, 0, 0, 1, 0, 0),),
    pin_net=(0, 0, 1, 1, 2, 2),
):
    # Pins:
    # 0: launch FF Q (clock pin 4, clock row 0)
    # 1: capture FF D (endpoint, clock pin 5, clock row 1)
    # 2, 3: unused dummy pins; 4: launch FF CK; 5: capture FF CK
    pin_state_start = [0, 1] + [2] * (num_pins - 1)
    arrays = {
        "pin_state_start": np.array(pin_state_start, dtype=np.int64),
        "state_pin": np.array([0, 1], dtype=np.int64),
        "state_domain": np.array([0, 0], dtype=np.int64),
        "state_tag": np.array([0, 0], dtype=np.int64),
        "state_launch_tag": np.array([0, 0], dtype=np.int64),
        "state_source_type": np.array([2, 2], dtype=np.int64),
        "state_role": np.array([0, 0], dtype=np.int64),
        "state_level": np.array([1, 2], dtype=np.int64),
        "state_slew_group": np.array([0, 1], dtype=np.int64),
        "state_valid": np.array([[1, 1], [1, 1]], dtype=np.uint8),
        "state_seed": np.array([[1, 1], [0, 0]], dtype=np.uint8),
        # Base clk2q = [50, 48] ps at launch clock AT = 0
        "state_reference": np.array(
            [[50.0, 48.0, 5.0, 5.0], [-2e8, -2e8, 0.0, 0.0]], dtype=np.float64
        ),
        # Net 0: pin 0 -> pin 1
        "net_edges": np.array([[0, 0, 1]], dtype=np.int64),
        "net_transition_mask": np.array([9], dtype=np.int64),
        "cell_transition_mask": np.empty(0, dtype=np.int64),
        "slew_only_cell_transition_mask": np.empty(0, dtype=np.int64),
        "context_state": np.empty(0, dtype=np.int64),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        "cell_edges": np.empty((0, 3), dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        # Setup check on state 1 (pin 1), check_arc=0 -> capture clock pin 5 (row 1)
        "check_index": np.array(
            [[1, 0, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]], dtype=np.int64
        ),
        # Period = 60 ps, setup = 15 ps -> base RAT = 45 ps (< 50 ps -> violated!)
        "check_values": np.array(
            [[60.0, -15.0, 0.0, 45.0], [60.0, -15.0, 0.0, 45.0]], dtype=np.float64
        ),
    }
    view = SparseTimingView(arrays, ("clk",), ("clk",), (), "lc", "lc")
    lut_val = torch.tensor([[10.0, 12.0, 14.0, 16.0]], dtype=torch.float64)
    lut_ax = torch.tensor([[0.0, 100.0]], dtype=torch.float64)
    lut_sz = torch.tensor([[2, 2]], dtype=torch.long)
    lut = LUTS_INFO(lut_val, lut_ax, lut_ax, lut_sz)
    arcs_info = SimpleNamespace(
        r_delay_luts=lut,
        f_delay_luts=lut,
        r_trans_luts=lut,
        f_trans_luts=lut,
        r_check_luts=lut,
        f_check_luts=lut,
    )
    ids = lambda v: torch.tensor(v, dtype=torch.long)
    empty = torch.empty(0, dtype=torch.float64)
    inst_arcs = ids(list(inst_arcs_rows))
    # endpoints_timing_check_arcs row 0: capture FF CK(5) -> D(1), lut=0, rising-edge(5=1)
    check_arcs = ids([[5, 1, 0, 0, 0, 1, 0, 0]])
    n_clk = len(clock_pins)
    return TimingPropagation(
        inrdelays=empty,
        infdelays=empty,
        inrtrans=empty,
        inftrans=empty,
        outcaps=empty,
        pin_net=ids(list(pin_net)),
        start_points=ids([]),
        end_points=ids([1]),
        clock_pins=ids(list(clock_pins)),
        FF_ids=ids([]),
        clk_pin_rtran=torch.full((n_clk,), 5.0, dtype=torch.float64),
        clk_pin_ftran=torch.full((n_clk,), 5.0, dtype=torch.float64),
        net_flat_arcs_start=ids([0, 1, 1, 1]),
        net_flat_arcs=ids([[0, 1]]),
        net2driver_pin_map=ids([0, -1, -1]),
        arcs_info=arcs_info,
        inst_flat_arcs_start=ids([0, len(inst_arcs_rows)]),
        inst_flat_arcs=inst_arcs,
        endpoints_constraint_arcs=ids([[5, 1, 60000, 0, 0, 1, 0, 0]]),
        endpoints_timing_check_arcs=check_arcs,
        flat_inst_arcs_by_level=ids([]).reshape(0, 8),
        flat_inst_arcs_by_level_start=ids([0, 0]),
        endpoints_rRAT=torch.tensor([45.0], dtype=torch.float64),
        endpoints_fRAT=torch.tensor([45.0], dtype=torch.float64),
        flat_pin_to_graph=ids([]),
        flat_pin_to_graph_start=ids([0] * (num_pins + 1)),
        flat_pin_to_graph_reverse=ids([]),
        flat_pin_to_graph_start_reverse=ids([0] * (num_pins + 1)),
        timing_metric_scope="setup_only",
        timing_aggregation_mode="smooth",
        timing_aggregation_tau_ps=2.0,
        sparse_timing_view=view,
        timing_propagation_device="cpu",
        production_fast_loop=True,
    )


def test_sparse_timing_launch_capture_clk_pin_aat_vjp_matches_fd():
    timing = _build_launch_capture_sparse_timing()
    delays = {
        "rise": torch.tensor([0.0, 8.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "fall": torch.tensor([0.0, 7.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    }
    zeros = {
        "rise": torch.zeros(6, dtype=torch.float64),
        "fall": torch.zeros(6, dtype=torch.float64),
    }
    # Shape is [2, num_clk]: row 0 = rise [launch, capture], row 1 = fall [launch, capture]
    clk_aat = torch.tensor(
        [[12.0, 4.0], [11.0, 3.0]], dtype=torch.float64, requires_grad=True
    )
    clk_tran = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float64)
    mask = torch.tensor([True, True])

    def loss_fn(aat_flat: torch.Tensor) -> torch.Tensor:
        aat_2d = aat_flat.reshape(2, 2)
        _wns, tns, _ws, _ts = timing(
            delays,
            zeros,
            zeros,
            surrogate_mode="lut_only",
            clk_pin_aat=aat_2d,
            clk_pin_tran=clk_tran,
            clk_pin_tran_mask=mask,
        )
        return -tns

    loss = loss_fn(clk_aat.reshape(-1))
    analytic = torch.autograd.grad(loss, clk_aat)[0].reshape(-1)
    numeric = _finite_diff_grad(loss_fn, clk_aat.detach().reshape(-1), eps=1e-4)
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-5)

    analytic_2d = analytic.reshape(2, 2)
    # Launch rising clock row (analytic_2d[0, 0]) increases AT -> worsens TNS -> positive d(-tns)/d(aat_launch_r)
    assert float(analytic_2d[0, 0].item()) > 0.5
    # Capture rising clock row (analytic_2d[0, 1]) increases RAT -> improves TNS -> negative d(-tns)/d(aat_capture_r)
    assert float(analytic_2d[0, 1].item()) < -0.5

    # launch_capture_clock_pair_adjoint retains positive launch pull and clips beneficial capture to 0
    weights, receipt = launch_capture_clock_pair_adjoint(
        clock_gradients={
            "clk_pin_r_aat": analytic_2d[0],
            "clk_pin_f_aat": analytic_2d[1],
        },
        pair_sink_pin_ids=torch.tensor([1, 2], dtype=torch.long),
        clock_row_to_compact_pin_ids=torch.tensor([1, 2], dtype=torch.long),
    )
    assert float(weights[0].item()) > 0.5
    assert float(weights[1].item()) == 0.0
    assert receipt["negative_pair_count"] == 1


def test_lcb_ff_domain_distinguishes_io_root_clock_from_movable_leaf_icg_and_runtime():
    # Pins:
    # 0: launch FF Q; 1: capture FF D; 2,3: dummy; 4: launch FF CK; 5: capture FF CK;
    # 6: IO clock port -> pin 7 (root FF CK on net 1); 8: ICG.ECK -> FF CK pins 4, 5 on net 2; 9: ICG.CK on net 1
    timing = _build_launch_capture_sparse_timing(
        num_pins=10,
        clock_pins=(4, 5, 7),
        inst_arcs_rows=(
            (4, 0, 0, 0, 0, 1, 0, 0),
            (9, 8, 0, 0, 1, 0, 0, 0),
        ),
        pin_net=(0, 0, 0, 0, 2, 2, 1, 1, 2, 1),
    )

    flat_net2pin = torch.tensor([0, 1, 6, 7, 9, 8, 4, 5], dtype=torch.int32)
    flat_net2pin_start = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    pin2net = torch.tensor([0, 0, -1, -1, 2, 2, 1, 1, 2, 1], dtype=torch.int32)
    # Nodes: 0=launch FF, 1=capture FF, 2=root FF, 3=ICG (all movable < 4), 4=IO clk port (terminal NI >= 4)
    pin2node = torch.tensor([0, 1, 0, 1, 0, 1, 4, 2, 3, 3], dtype=torch.int32)
    net2driver = torch.tensor([0, -1, -1], dtype=torch.int32)
    pin_direct = [
        "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "OUTPUT", "INPUT"
    ]
    dc = SimpleNamespace(
        flat_net2pin_map=flat_net2pin,
        flat_net2pin_start_map=flat_net2pin_start,
        pin2net_map=pin2net,
        pin2node_map=pin2node,
        net2driver_pin_map=net2driver,
        net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1, 1, 1], dtype=torch.long),
        start_points=torch.tensor([6], dtype=torch.long),
        inst_cell_id=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor([0, 1], dtype=torch.long),
        inst_libcell_offset=torch.tensor([0, 0, 0, 0], dtype=torch.long),
    )
    placedb = SimpleNamespace(
        num_movable_nodes=4,
        num_terminals=0,
        num_terminal_NIs=1,
        num_physical_nodes=4,
        num_pins=10,
        dbu=1.0,
        # Simulate ecc_py storing instance orientation strings in pin_direct!
        pin_direct=["R0"] * 10,
    )
    domain = build_lcb_ff_clock_star_domain(
        clock_pins_native=timing.clock_pins,
        pin2net_map=dc.pin2net_map,
        pin2node_map=dc.pin2node_map,
        pin_direct=pin_direct,
        flat_net2pin_map=dc.flat_net2pin_map,
        flat_net2pin_start_map=dc.flat_net2pin_start_map,
        placedb=placedb,
        params=SimpleNamespace(ignore_net_degree=100),
    )
    assert len(domain.plan.leaf_net_ids) == 1
    assert len(domain.plan.root_net_ids) == 1
    assert domain.plan.leaf_pair_count() == 2  # ICG.ECK(8) -> FF0.CK(4), FF1.CK(5)

    compact_pos = torch.tensor([0.0, 20.0, 50.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    compact_caps = {
        k: torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        for k in ("generic", "rise", "fall")
    }
    state = compute_lcb_ff_clock_state(
        plan=domain.plan,
        pin_pos=compact_pos,
        base_caps=compact_caps,
        r_unit=0.2,
        c_unit=0.1,
        coordinate_to_um=1.0,
        timing_op=timing,
        data_collections=dc,
    )
    receipt = apply_lcb_ff_to_compact_clk_pins(timing, state)
    assert receipt["leaf_pair_count"] == 2
    assert receipt["root_sink_count"] == 1
    # Row 0 (FF0 @ 20um) < Row 1 (FF1 @ 50um), Row 2 (root FF) == 0.0!
    r_aat = state["late_rise_arrival_ps"]
    assert float(r_aat[1].item()) > float(r_aat[0].item()) > 0.0
    assert float(r_aat[2].item()) == 0.0

    # Also verify StarTdpRuntime synthesizes pin_direct when placedb.pin_direct == ["R0"]*10!
    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    pin_pos_full = torch.tensor(
        [10.0, 90.0, 0.0, 0.0, 20.0, 50.0, 0.0, 15.0, 0.0, 0.0] + [0.0] * 10,
        dtype=torch.float32,
    )
    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor(
            [True, False, False, False, False, False, False, False, True, False]
        ),
        pin_caps_op=lambda _off, _dc: (
            torch.tensor([0.0, 1.2, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8], dtype=torch.float32),
            torch.tensor([0.0, 1.2, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8], dtype=torch.float32),
            torch.tensor([0.0, 1.2, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8], dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _resolve_timing_surrogate_mode=lambda: "lut_only",
    )
    op_cols = SimpleNamespace(
        timing_propagation_op=timing,
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: torch.cat([p[pin2node.long()], p[5 + pin2node.long()]]),
        wirelength_op=lambda p: (p[1] - p[0]).abs() + (p[6] - p[5]).abs(),
    )
    rt_params = SimpleNamespace(
        scale_factor=1.0,
        ignore_net_degree=100,
        deterministic_flag=True,
        star_rc_calibration_enabled=0,
        star_slope_calibration_enabled=0,
        star_lcb_ff_clock_enabled=1,
        star_kappa=0.05,
    )
    rt = StarTdpRuntime(
        model=model,
        params=rt_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    assert rt.clock_star is not None
    assert rt.clock_star.pair_count == 2
    pos_nodes = torch.tensor(
        [10.0, 90.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )
    summary = rt.refresh_snapshot(pos_nodes, iteration=120, overflow=0.35)
    assert summary["kappa_matched"] is True
    assert summary["lambda_star"] > 0.0
    assert summary["k_l1"] > 0.0
    assert summary["clock_k_l1"] > 0.0


def test_precondition_2n_lshape_components_and_nesterov_transport():
    # Minimal 2-movable-node placement fixture with active Star curvature + Nesterov step
    placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=2,
        num_filler_nodes=0,
        num_nodes=2,
        num_pins=2,
        regions=[],
        dbu=100.0,
    )
    pos = torch.nn.Parameter(torch.tensor([10.0, 90.0, 20.0, 40.0], dtype=torch.float32))
    pos.grad = torch.zeros_like(pos)

    pin2node = torch.tensor([0, 1], dtype=torch.long)
    drivers = torch.tensor([0], dtype=torch.int32)

    star = StarPinPair(
        net2driver_pin_map=drivers,
        timing_net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        timing_net_flat_arcs_start=torch.tensor([0, 1], dtype=torch.long),
        num_pins=2,
        r_unit=0.3,
        c_unit=0.2,
        scale_factor=1.0,
        dbu=100.0,
    )
    cand = star.prepare_snapshot(
        raw_weight=torch.tensor([5.0], dtype=torch.float32),
        length_um=star.length_um(pos.detach()),
        load_length_weight=torch.tensor([0.5], dtype=torch.float32),
        curvature_load_length_weight=torch.tensor([0.5], dtype=torch.float32),
    )
    star.publish_snapshot(cand)

    dc = SimpleNamespace(
        pos=[pos],
        node_areas=torch.tensor([4.0, 4.0], dtype=torch.float32),
        net_weights=torch.tensor([1.0], dtype=torch.float32),
        num_pins_in_nodes=torch.tensor([1.0, 1.0], dtype=torch.float32),
        pin2node_map=pin2node,
    )
    op_collections = SimpleNamespace(
        pin_pos_op=lambda p: p,
        pws_op=lambda w: torch.ones(2, dtype=torch.float32),
        star_pin_pair_op=star,
        clock_star_pin_pair_op=None,
    )

    class _DummyRuntime:
        curvature_precondition_enabled = True
        is_objective_active = True
        refresh_nesterov_driver = True
        momentum_transport_envelope = 1.05
        segment_timing_coefficient = 2.0
        gen = 1

        def __init__(self):
            self.placedb = placedb
            self.data_collections = dc
            self.op_collections = op_collections
            self.transport_state = {"last": None, "carried": None}
            self.step_context = None

        @property
        def driver_signature(self):
            return (self.gen, True, 2.0)

        def _star_pin_pair_op(self):
            return star

        def _segment_timing_star_load_curvature(self):
            return True

        def _segment_timing_star_pair_objective(self):
            return "elmore"

        def _segment_timing_cut_effective_smoothing_gamma(self):
            return 10.0

        def _driver_sink_star_base_cap_map(self):
            c = torch.tensor([0.0, 1.2], dtype=torch.float32)
            return {"generic": c, "rise": c, "fall": c}

        def effective_curvature_2n(self, *, fallback_pos, fallback_alpha=1.0):
            p = self.step_context["position"] if self.step_context else fallback_pos
            a = self.step_context["alpha"] if self.step_context else fallback_alpha
            h, _ = rowsum_hessian(self, p, elmore_axes=2, full_slope=True)
            return h * float(a)

        def record_precondition_diagonal(self, precond_2n):
            self.transport_state["last"] = precond_2n.reshape(-1).detach().clone()

        def begin_nesterov_step(self, position, alpha_start):
            self.step_context = {"position": position.detach().clone(), "alpha": float(alpha_start)}

        def end_nesterov_step(self, u_next, v_next):
            self.step_context = None
            last = self.transport_state.get("last")
            carried = self.transport_state.get("carried")
            if last is not None and carried is not None and u_next is not None:
                transport_momentum(u_next, v_next, carried, last, 1.05)
            self.transport_state["carried"] = last

    rt = _DummyRuntime()
    precond = PreconditionOp(
        placedb,
        dc,
        op_collections,
        precond_pin_count_flag=True,
        use_net_weights=False,
        star_runtime_getter=lambda: rt,
    )
    density_w = torch.tensor([1.0], dtype=torch.float32)
    # Verify L-shape apply_components works cleanly with [2, N] Star curvature!
    g_base = torch.tensor([10.0, -10.0, 8.0, -8.0], dtype=torch.float32)
    g_lshape = torch.tensor([2.0, -2.0, 4.0, -4.0], dtype=torch.float32)
    c_base, c_lshape = precond.apply_components([g_base, g_lshape], density_w)
    assert c_base.shape == (4,) and c_lshape.shape == (4,)
    # Because dx=80 != dy=20, X and Y WA curvatures differ, producing anisotropic [2, N] scaling!
    assert rt.transport_state["last"] is not None
    assert rt.transport_state["last"].shape == (4,)
    assert not torch.allclose(rt.transport_state["last"][:2], rt.transport_state["last"][2:])

    class _Owner:
        star_tdp_runtime = rt

        def obj_and_grad_fn(self, x):
            p = x.detach().clone().requires_grad_(True)
            caps = rt._driver_sink_star_base_cap_map()
            loss = float(rt.gen) * star.structured_delay_objective(
                p, base_caps=caps, distance_model="wa", smoothing_gamma=10.0
            )
            g = torch.autograd.grad(loss, p)[0]
            precond(g, density_w)
            return loss.detach(), g.detach()

    owner = _Owner()
    opt = NesterovAcceleratedGradientOptimizer(
        [pos],
        lr=0.5,
        obj_and_grad_fn=owner.obj_and_grad_fn,
        constraint_fn=lambda _v: None,
        use_bb=False,
    )
    opt.step()
    sig_before = opt.param_groups[0]["star_driver_signature"][0]
    assert sig_before == (1, True, 2.0)

    # Advance snapshot generation: next step must refresh g_k at v_k and update signature!
    rt.gen = 2
    evals_before = opt.param_groups[0]["obj_eval_count"]
    opt.step()
    assert opt.param_groups[0]["star_driver_signature"][0] == (2, True, 2.0)
    assert opt.param_groups[0]["obj_eval_count"] >= evals_before + 2

    # Rebase clears carried diagonal and signature cleanly (S5B1 milestone contract).
    opt.rebase_objective_state(reason="s5b1_milestone")
    assert rt.transport_state["carried"] is None
    assert opt.param_groups[0]["star_driver_signature"] == []


def test_carrier_policy_and_midplace_schedule_preserve_star_isolation():
    params = SimpleNamespace(
        timing_placement_carrier="star_pin_pair",
        _timing_placement_carrier_explicit=True,
        global_place_flag=1,
        placement_sizing_mode="place_only",
        star_kappa_recalibrate_on_restart=0,
    )
    configure_timing_placement_carrier(params)
    assert params.with_sta == 1
    assert params.diff_timing_driven_placement == 1
    assert params.differentiable_timing_obj == 1
    assert params.enable_net_weighting == 0
    assert diff_tdp_enabled(params) is True
    assert diff_tdp_uses_star_pin_pair(params) is True
    assert diff_tdp_uses_direct_loss(params) is False
    assert diff_tdp_controls_direct_timing_objective(params) is False
    assert should_use_live_timing_loss(params, {"timing_objective_active": True}) is False

    # MidplaceSchedule must NOT turn on model.use_timing_obj when carrier == "star_pin_pair"!
    sched = MidplaceSchedule(
        use_timing_obj=False,
        diff_tdp_step_status={"timing_objective_active": True},
        star_tdp_state={"lambda_star": 12.5, "kappa_matched": True},
    )
    restored = {}

    class _FakeStarRt:
        def restore_midplace_state(self, state, *, recalibrate_kappa):
            restored["state"] = state
            restored["recalibrate"] = recalibrate_kappa

    placer = SimpleNamespace()
    model = SimpleNamespace(use_timing_obj=False, gamma=torch.tensor(5.0), star_tdp_runtime=_FakeStarRt())
    sched.restore_model_state(placer, model, params)
    assert model.use_timing_obj is False
    assert restored["state"]["lambda_star"] == 12.5
    assert restored["recalibrate"] is False

    # Control check: direct_loss STILL restores model.use_timing_obj = True!
    direct_params = SimpleNamespace(
        timing_placement_carrier="direct_loss",
        placement_sizing_mode="place_only",
    )
    direct_model = SimpleNamespace(use_timing_obj=False, gamma=torch.tensor(5.0))
    sched.restore_model_state(placer, direct_model, direct_params)
    assert direct_model.use_timing_obj is True


def test_apply_flow_defaults_allows_explicit_star_pin_pair_with_all_on_s5b1():
    from dreamplace.Params import Params
    from dreamplace.flows.config.resolver import apply_flow_defaults

    cfg = {
        "flow_kind": "placement",
        "routability_opt_flag": 1,
        "l_shape_routability_flag": 1,
        "l_shape_use_ggr_topology": 0,
        "adjust_gpugr_area_flag": 1,
        "timing_opt_enabled": 1,
        "buffering_mode": "candidate",
        "buffering_candidate_strategy": "discrete_net_gradient",
        "with_sta": 1,
        "diff_timing_driven_placement": 1,
        "differentiable_timing_obj": 1,
        "timing_placement_carrier": "star_pin_pair",
    }
    params = Params()
    params.fromJson(cfg)
    for key in cfg:
        setattr(params, f"_{key}_explicit", True)
    apply_flow_defaults(params)
    assert params.timing_placement_carrier == "star_pin_pair"
    assert params.star_load_geometry == "wa_net"
    assert params.star_recal_m_eff_scale_fix == 1
    assert params.star_restart_history_mode == "selective_sigmoid"
    assert params.macro_exclusion_field_flag == 1
    assert params.macro_exclusion_force_mode == "taylor_kkt"
    assert params.star_kappa_recalibrate_on_restart == 1
    assert math.isclose(params.star_post_restart_max_kappa, 0.20, rel_tol=1e-6)
    assert params.star_endpoint_slack_tau_mode == "post_restart_drift"
    assert math.isclose(params.star_endpoint_slack_tau_scale, 1.0, rel_tol=1e-6)
    assert params.routability_opt_flag == 1
    assert params.l_shape_routability_flag == 1
    assert params.adjust_gpugr_area_flag == 1
    assert params.timing_opt_enabled == 1
    assert params.buffering_mode == "candidate"


def test_input_accumulator_max_carry_ratio_bounds_cold_start_spike():
    acc = InputAccumulator(
        gain=0.2, half_life=150.0, mode="ema", max_carry_ratio=4.0
    )
    identity = ("test", 3, 1)

    def _fresh(scale: float):
        vec = torch.tensor([scale, 0.5 * scale, 0.0], dtype=torch.float32)
        return {
            "delay": {"rise": vec.clone(), "fall": vec.clone()},
            "cap": {"rise": vec.clone(), "fall": vec.clone()},
            "impulse": {"rise": torch.zeros_like(vec), "fall": torch.zeros_like(vec)},
        }

    # Step 1: cold-start spike of 1000.0 -> stored = 0.2 * 1000 = 200.0
    cand1 = acc.prepare(_fresh(1000.0), identity=identity, pin_count=3, iteration=10)
    acc.commit(cand1)
    assert math.isclose(float(cand1["values"]["delay"]["rise"][0].item()), 200.0, rel_tol=1e-6)

    # Step 2: fresh drops 100x to 10.0 -> updated_fresh = 2.0; carried is capped at 4.0 * 2.0 = 8.0 (total 10.0, not ~189!)
    cand2 = acc.prepare(_fresh(10.0), identity=identity, pin_count=3, iteration=25)
    acc.commit(cand2)
    assert math.isclose(float(cand2["values"]["delay"]["rise"][0].item()), 10.0, rel_tol=1e-5)


def test_star_tdp_runtime_live_kappa_ceiling_and_gate_reentry_reset():
    placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=2,
        num_filler_nodes=0,
        num_nodes=2,
        num_pins=2,
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT"],
    )
    pin2node = torch.tensor([0, 1], dtype=torch.long)
    dc = SimpleNamespace(
        pin2net_map=torch.tensor([0, 0], dtype=torch.long),
        pin2node_map=pin2node,
        net2driver_pin_map=torch.tensor([0], dtype=torch.int32),
        net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
        inst_libcell_offset=torch.tensor([0, 0], dtype=torch.long),
    )

    state = {"scale": 1.0, "slack_bias": -10.0}

    def _fake_timing_op(delays, impulses, loads, **_kwargs):
        d = delays["rise"][1] + delays["fall"][1]
        tns = state["slack_bias"] - state["scale"] * d
        wns = tns
        return wns, tns, wns, tns

    timing_op = SimpleNamespace(
        __call__=_fake_timing_op,
        clock_pins=torch.empty(0, dtype=torch.long),
        num_pins=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    class _CallableTiming:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 2
        device = torch.device("cpu")
        dtype = torch.float32

        def __call__(self, delays, impulses, loads, **kwargs):
            return _fake_timing_op(delays, impulses, loads, **kwargs)

    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False]),
        pin_caps_op=lambda _off, _dc: (
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _current_timing_surrogate_mode=lambda: "lut_only",
    )
    op_cols = SimpleNamespace(
        timing_propagation_op=_CallableTiming(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: (p[1] - p[0]).abs() + (p[3] - p[2]).abs(),
    )
    rt_params = SimpleNamespace(
        scale_factor=1.0,
        ignore_net_degree=100,
        deterministic_flag=True,
        star_rc_calibration_enabled=1,
        star_slope_calibration_enabled=0,
        star_lcb_ff_clock_enabled=0,
        star_kappa=0.05,
        star_max_kappa=0.25,
        star_kappa_recalibrate_on_restart=1,
    )
    rt = StarTdpRuntime(
        model=model,
        params=rt_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    pos = torch.tensor([10.0, 90.0, 20.0, 40.0], dtype=torch.float32)

    # Initial low-sensitivity calibration -> real_k == 0.05
    s1 = rt.refresh_snapshot(pos, iteration=10, overflow=0.34)
    assert math.isclose(s1["realized_kappa"], 0.05, rel_tol=1e-4)
    lam1 = s1["lambda_star"]

    # Post-S5B1 100x surge in path multiplicity -> lambda_star MUST clamp so realized_kappa <= 0.25!
    state["scale"] = 100.0
    s2 = rt.refresh_snapshot(pos, iteration=25, overflow=0.28)
    assert s2["realized_kappa"] <= 0.25 + 1e-5
    assert math.isclose(s2["realized_kappa"], 0.25, rel_tol=1e-4)
    assert s2["lambda_star"] < lam1 * 0.2

    # Gate deactivation (overflow > 0.35 recovery) must clear RC anchor & history for clean re-entry!
    assert rt.data_rc_cal_state is not None
    rt.set_gate_active(False)
    assert rt.data_rc_cal_state is None
    assert rt.rc_next_overflow is None
    assert rt.kappa_matched is False


def test_star_tdp_runtime_default_restart_preserves_lambda_and_filters_high_degree_nets():
    # Net 0: degree 2 (kept); Net 1: degree 4 >= ignore_net_degree=4 (filtered).
    placedb = SimpleNamespace(
        num_movable_nodes=6,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=6,
        num_filler_nodes=0,
        num_nodes=6,
        num_pins=6,
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"],
    )
    dc = SimpleNamespace(
        pin2net_map=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long),
        pin2node_map=torch.arange(6, dtype=torch.long),
        net2driver_pin_map=torch.tensor([0, 2], dtype=torch.int32),
        net_flat_arcs=torch.tensor(
            [[0, 1], [2, 3], [2, 4], [2, 5]], dtype=torch.long
        ),
        net_flat_arcs_start=torch.tensor([0, 1, 4], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2, 6], dtype=torch.int32),
        inst_libcell_offset=torch.zeros(6, dtype=torch.long),
    )

    state = {"scale": 10.0, "tns_base": -200000.0}

    class _CallableTiming:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 6
        device = torch.device("cpu")
        dtype = torch.float32

        def __call__(self, delays, impulses, loads, **kwargs):
            d = delays["rise"][1] + delays["fall"][1]
            tns = state["tns_base"] - state["scale"] * d
            return tns, tns, tns, tns

    # Simulate S5B1 leaving _active_timing_surrogate_mode="surrogate_only".
    seen_surrogate_modes = []
    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False, True, False, False, False]),
        _active_timing_surrogate_mode="surrogate_only",
        _relaxed_buffer_timing_enabled=lambda: False,
    )

    def _pin_caps_op(_off, _dc):
        seen_surrogate_modes.append(model._active_timing_surrogate_mode)
        c = (
            torch.ones(6, dtype=torch.float32)
            if model._active_timing_surrogate_mode == "lut_only"
            else torch.full((6,), 99.0, dtype=torch.float32)
        )
        return c, c, c

    model.pin_caps_op = _pin_caps_op
    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    op_cols = SimpleNamespace(
        timing_propagation_op=_CallableTiming(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: (p[1] - p[0]).abs() + (p[7] - p[6]).abs(),
    )
    rt_params = SimpleNamespace(
        scale_factor=1.0,
        ignore_net_degree=4,
        deterministic_flag=True,
        star_rc_calibration_enabled=1,
        star_slope_calibration_enabled=1,
        star_lcb_ff_clock_enabled=0,
        star_kappa=0.05,
        star_max_kappa=0.18,
    )
    rt = StarTdpRuntime(
        model=model,
        params=rt_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    # Net 1 (degree 4 >= 4) must be filtered out; only Net 0's single pair remains!
    assert rt.star.pair_count == 1
    assert int(rt.star.driver_pin_ids[0].item()) == 0
    assert int(rt.star.sink_pin_ids[0].item()) == 1

    pos = torch.tensor(
        [10.0, 90.0, 0.0, 20.0, 30.0, 40.0, 20.0, 40.0, 0.0, 10.0, 20.0, 30.0],
        dtype=torch.float32,
    )
    s1 = rt.refresh_snapshot(pos, iteration=10, overflow=0.30)
    assert seen_surrogate_modes[-1] == "lut_only"
    assert s1["teacher_resampled"] is True
    assert rt.slope_calibrator is not None
    lam1 = s1["lambda_star"]
    base1 = rt.base_lambda_star
    assert math.isclose(s1["realized_kappa"], 0.05, rel_tol=1e-4)

    # Pure AutoDMP contract: default restart (recalibrate=False) keeps base_lambda_star
    # strictly fixed after first calibration, while resetting InputAccumulator,
    # AtMergeTemperature, and resampling FLUTE teacher RC on restart + xplace_overflow milestones.
    model._active_timing_surrogate_mode = "surrogate_only"
    rt.reset_for_restart()
    assert rt.kappa_matched is True
    assert math.isclose(rt.base_lambda_star, base1, rel_tol=1e-6)
    state["scale"] = 0.5
    s2 = rt.refresh_snapshot(pos, iteration=25, overflow=0.29)
    assert seen_surrogate_modes[-1] == "lut_only"
    assert s2["teacher_resampled"] is True
    assert math.isclose(rt.base_lambda_star, base1, rel_tol=1e-6)
    assert math.isclose(s2["lambda_star"], lam1, rel_tol=1e-6)

    # Above next 0.05 overflow milestone (0.29 -> next <= 0.25): teacher is NOT resampled.
    state["tns_base"] = -5000.0
    s_above_milestone = rt.refresh_snapshot(pos, iteration=30, overflow=0.27)
    assert s_above_milestone["teacher_resampled"] is False
    assert math.isclose(rt.base_lambda_star, base1, rel_tol=1e-6)
    assert math.isclose(s_above_milestone["lambda_star"], lam1, rel_tol=1e-6)

    # Crossing <= 0.25 overflow milestone triggers FLUTE teacher RC resample while lambda_star stays fixed.
    state["tns_base"] = -200000.0
    s_at_milestone = rt.refresh_snapshot(pos, iteration=35, overflow=0.24)
    assert s_at_milestone["teacher_resampled"] is True
    assert math.isclose(rt.base_lambda_star, base1, rel_tol=1e-6)
    assert math.isclose(s_at_milestone["lambda_star"], lam1, rel_tol=1e-6)


def test_sparse_smooth_extremum_softplus_tns_and_drift_at_merge_temperature():
    # Two launch seeds (state 0 on pin 0, state 1 on pin 2) merge at state 2 (pin 1)
    # via two competing net edges at level 2.
    num_pins = 6
    pin_state_start = [0, 1, 2, 3, 3, 3, 3]
    arrays = {
        "pin_state_start": np.array(pin_state_start, dtype=np.int64),
        "state_pin": np.array([0, 1, 2], dtype=np.int64),
        "state_domain": np.array([0, 0, 0], dtype=np.int64),
        "state_tag": np.array([0, 0, 0], dtype=np.int64),
        "state_launch_tag": np.array([0, 0, 0], dtype=np.int64),
        "state_source_type": np.array([2, 2, 2], dtype=np.int64),
        "state_role": np.array([0, 0, 0], dtype=np.int64),
        "state_level": np.array([1, 2, 1], dtype=np.int64),
        "state_slew_group": np.array([0, 1, 2], dtype=np.int64),
        "state_valid": np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8),
        "state_seed": np.array([[1, 1], [0, 0], [1, 1]], dtype=np.uint8),
        "state_reference": np.array(
            [
                [50.0, 48.0, 5.0, 5.0],
                [-2e8, -2e8, 0.0, 0.0],
                [50.0, 48.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        ),
        # Two competing fan-in edges merging onto state 1 (pin 1):
        # edge 0 from state 0 (pin 0 -> pin 1), edge 1 from state 2 (pin 2 -> pin 1).
        # Wait: net delay indexes sink pin `level.pins[dst]`, so to give independent delays
        # per branch we use two cell arcs driving state 1 from state 0 and state 2, OR
        # two intermediate states (pins 3, 4) merging at state 1 via cell arcs!
        "net_edges": np.array([[0, 0, 3], [1, 2, 4]], dtype=np.int64),
        "net_transition_mask": np.array([9, 9], dtype=np.int64),
        "cell_transition_mask": np.array([5, 5], dtype=np.int64),
        "slew_only_cell_transition_mask": np.empty(0, dtype=np.int64),
        "context_state": np.empty(0, dtype=np.int64),
        "net_threshold_adjustments": np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64
        ),
        "cell_edges": np.array([[0, 3, 1], [1, 4, 1]], dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array(
            [[1, 0, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]], dtype=np.int64
        ),
        "check_values": np.array(
            [[70.0, -1.0, 0.0, 55.0], [70.0, -1.0, 0.0, 55.0]], dtype=np.float64
        ),
    }
    # Extend state arrays to 5 states (0: pin0 seed, 1: pin1 merge endpoint, 2: pin2 seed, 3: pin3 branchA, 4: pin4 branchB)
    arrays["pin_state_start"] = np.array([0, 1, 2, 3, 4, 5, 5], dtype=np.int64)
    arrays["state_pin"] = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    arrays["state_domain"] = np.zeros(5, dtype=np.int64)
    arrays["state_tag"] = np.zeros(5, dtype=np.int64)
    arrays["state_launch_tag"] = np.zeros(5, dtype=np.int64)
    arrays["state_source_type"] = np.full(5, 2, dtype=np.int64)
    arrays["state_role"] = np.zeros(5, dtype=np.int64)
    arrays["state_level"] = np.array([1, 3, 1, 2, 2], dtype=np.int64)
    arrays["state_slew_group"] = np.arange(5, dtype=np.int64)
    arrays["state_valid"] = np.ones((5, 2), dtype=np.uint8)
    arrays["state_seed"] = np.array(
        [[1, 1], [0, 0], [1, 1], [0, 0], [0, 0]], dtype=np.uint8
    )
    arrays["state_reference"] = np.array(
        [
            [50.0, 48.0, 5.0, 5.0],
            [-2e8, -2e8, 0.0, 0.0],
            [50.0, 48.0, 5.0, 5.0],
            [-2e8, -2e8, 0.0, 0.0],
            [-2e8, -2e8, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    arrays["cell_transition_mask"] = np.array([15, 15], dtype=np.int64)

    view = SparseTimingView(arrays, ("clk",), ("clk",), (), "merge", "merge")
    lut_val = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float64)
    lut_ax = torch.tensor([[0.0, 100.0]], dtype=torch.float64)
    lut_sz = torch.tensor([[2, 2]], dtype=torch.long)
    lut = LUTS_INFO(lut_val, lut_ax, lut_ax, lut_sz)
    arcs_info = SimpleNamespace(
        r_delay_luts=lut,
        f_delay_luts=lut,
        r_trans_luts=lut,
        f_trans_luts=lut,
        r_check_luts=lut,
        f_check_luts=lut,
    )
    ids = lambda v: torch.tensor(v, dtype=torch.long)
    empty = torch.empty(0, dtype=torch.float64)
    # Two positive-unate combinational cell arcs (pin 3 -> pin 1, pin 4 -> pin 1)
    level_arcs = ids([[3, 1, 0, 0, 1, 0, 0, 0], [4, 1, 0, 0, 1, 0, 0, 0]])
    timing = TimingPropagation(
        inrdelays=empty,
        infdelays=empty,
        inrtrans=empty,
        inftrans=empty,
        outcaps=empty,
        pin_net=ids([0, 2, 1, 0, 1, 3]),
        start_points=ids([]),
        end_points=ids([1]),
        clock_pins=ids([5]),
        FF_ids=ids([]),
        clk_pin_rtran=torch.tensor([5.0], dtype=torch.float64),
        clk_pin_ftran=torch.tensor([5.0], dtype=torch.float64),
        net_flat_arcs_start=ids([0, 1, 2, 2, 2]),
        net_flat_arcs=ids([[0, 3], [2, 4]]),
        net2driver_pin_map=ids([0, 2, -1, -1]),
        arcs_info=arcs_info,
        inst_flat_arcs_start=ids([0, 2]),
        inst_flat_arcs=level_arcs,
        endpoints_constraint_arcs=ids([[5, 1, 70000, 0, 0, 1, 0, 0]]),
        endpoints_timing_check_arcs=ids([[5, 1, 0, 0, 0, 1, 0, 0]]),
        flat_inst_arcs_by_level=level_arcs,
        flat_inst_arcs_by_level_start=ids([0, 0, 2]),
        endpoints_rRAT=torch.tensor([60.0], dtype=torch.float64),
        endpoints_fRAT=torch.tensor([60.0], dtype=torch.float64),
        flat_pin_to_graph=ids([]),
        flat_pin_to_graph_start=ids([0] * (num_pins + 1)),
        flat_pin_to_graph_reverse=ids([]),
        flat_pin_to_graph_start_reverse=ids([0] * (num_pins + 1)),
        timing_metric_scope="setup_only",
        timing_aggregation_mode="smooth",
        timing_aggregation_tau_ps=2.0,
        sparse_timing_view=view,
        timing_propagation_device="cpu",
        production_fast_loop=True,
    )

    zeros = {
        "rise": torch.zeros(num_pins, dtype=torch.float64),
        "fall": torch.zeros(num_pins, dtype=torch.float64),
    }

    # 1. Verify smooth_extremum=True at tau=1.0 matches finite differences on last_soft_tns_loss.
    d_rise = torch.tensor(
        [0.0, 0.0, 0.0, 6.0, 5.2, 0.0], dtype=torch.float64, requires_grad=True
    )
    d_fall = torch.tensor([0.0, 0.0, 0.0, 4.0, 3.5, 0.0], dtype=torch.float64)

    def _eval_soft_tns(dr):
        timing(
            {"rise": dr, "fall": d_fall},
            zeros,
            zeros,
            surrogate_mode="lut_only",
            smooth_extremum=True,
            extremum_tau_ps=1.0,
            at_merge_temperature=None,
        )
        return timing.last_soft_tns_loss

    soft_loss = _eval_soft_tns(d_rise)
    analytic = torch.autograd.grad(soft_loss, d_rise)[0]
    numeric = _finite_diff_grad(_eval_soft_tns, d_rise, eps=1e-5)
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-5)

    # 2. Verify AtMergeTemperature drift calibration across 3 publications:
    # Pub 1: gap = 6.0 - 5.2 = 0.8 ps -> tau_used = 1.0, tau_next = 1.0 (no previous).
    # Pub 2: gap = 12.0 - 5.2 = 6.8 ps -> drift = |6.8 - 0.8| = 6.0 ps -> sigma = 1.4826 * 6.0 = 8.8956 -> tau_next = 8.8956 / 1.70 = 5.2327 ps!
    # Pub 3: forward AT stays identical to fixed tau=1.0, while VJP splits across branches 3 & 4 using tau=5.2327 ps!
    at_temp = AtMergeTemperature(1.0)
    at_temp.begin(1)
    dr1 = torch.tensor([0.0, 0.0, 0.0, 6.0, 5.2, 0.0], dtype=torch.float64, requires_grad=True)
    timing(
        {"rise": dr1, "fall": d_fall},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
        at_merge_temperature=at_temp,
    )
    _ = torch.autograd.grad(timing.last_soft_tns_loss, dr1)[0]
    rec1 = at_temp.finish()
    assert math.isclose(rec1["tau_used_ps"], 1.0, rel_tol=1e-6)
    assert math.isclose(rec1["tau_next_ps"], 1.0, rel_tol=1e-6)

    at_temp.begin(2)
    dr2 = torch.tensor([0.0, 0.0, 0.0, 12.0, 5.2, 0.0], dtype=torch.float64, requires_grad=True)
    timing(
        {"rise": dr2, "fall": d_fall},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
        at_merge_temperature=at_temp,
    )
    _ = torch.autograd.grad(timing.last_soft_tns_loss, dr2)[0]
    rec2 = at_temp.finish()
    expected_tau3 = 1.4826 * 6.0 / 1.70
    assert math.isclose(rec2["tau_next_ps"], expected_tau3, rel_tol=1e-4)

    # Pub 3: compare VJP at drifted tau (~5.23 ps) vs fixed tau=1.0 on identical delays (gap = 2.0 ps).
    at_temp.begin(3)
    dr3 = torch.tensor([0.0, 0.0, 0.0, 8.0, 6.0, 0.0], dtype=torch.float64, requires_grad=True)
    wns_drift, tns_drift, _, _ = timing(
        {"rise": dr3, "fall": d_fall},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
        at_merge_temperature=at_temp,
    )
    soft_drift = float(timing.last_soft_tns_loss.detach().item())
    g_drift = torch.autograd.grad(timing.last_soft_tns_loss, dr3)[0]
    rec3 = at_temp.finish()
    assert math.isclose(rec3["tau_used_ps"], expected_tau3, rel_tol=1e-4)

    dr3_fixed = dr3.detach().clone().requires_grad_(True)
    wns_fixed, tns_fixed, _, _ = timing(
        {"rise": dr3_fixed, "fall": d_fall},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
        at_merge_temperature=None,
    )
    soft_fixed = float(timing.last_soft_tns_loss.detach().item())
    g_fixed = torch.autograd.grad(timing.last_soft_tns_loss, dr3_fixed)[0]

    # Forward values MUST be 100% bit-identical (drift tau affects ONLY VJP)!
    assert math.isclose(soft_drift, soft_fixed, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(tns_drift.item()), float(tns_fixed.item()), rel_tol=0.0, abs_tol=1e-12)
    # Runner-up branch (pin 4, 2.0 ps behind pin 3) gets much higher share under tau=5.23 ps than tau=1.0 ps,
    # while total adjoint sum (g[3] + g[4]) stays conserved!
    assert math.isclose(
        float((g_drift[3] + g_drift[4]).item()),
        float((g_fixed[3] + g_fixed[4]).item()),
        rel_tol=1e-6,
    )
    assert float(g_drift[4].item()) > float(g_fixed[4].item()) * 2.5


def test_slack_retention_beta_and_drift_tau_math():
    from dreamplace.ops.star_pin_pair import (
        compute_slack_retention_beta,
        estimate_slack_drift_tau,
        soft_slack_severity,
    )

    s_before = {
        "rise": torch.tensor([-50.0, -40.0, -20.0, -40.0, 2.0e8], dtype=torch.float32),
        "fall": torch.tensor([100.0, -10.0, -10.0, 2.0e8, 2.0e8], dtype=torch.float32),
    }
    s_after = {
        "rise": torch.tensor([200.0, -40.0, -60.0, 5.0, 2.0e8], dtype=torch.float32),
        "fall": torch.tensor([120.0, 80.0, -10.0, 2.0e8, 2.0e8], dtype=torch.float32),
    }
    beta = compute_slack_retention_beta(s_before, s_after, tau_ps=20.0)
    br = beta["rise"]
    assert float(br[0].item()) < 1e-4  # deeply cleared -> ~0
    assert math.isclose(float(br[1].item()), 1.0, rel_tol=1e-5)  # unchanged -> 1
    assert math.isclose(float(br[2].item()), 1.0, rel_tol=1e-5)  # worsened -> clipped at 1
    expected_guard = float(
        soft_slack_severity(torch.tensor([5.0]), tau_ps=20.0).item()
        / soft_slack_severity(torch.tensor([-40.0]), tau_ps=20.0).item()
    )
    assert 0.15 < float(br[3].item()) < 0.40
    assert math.isclose(float(br[3].item()), expected_guard, rel_tol=1e-5)
    assert float(br[4].item()) == 0.0  # unconstrained sentinel -> 0

    # Drift tau estimation matches 1.4826 * weighted_median(|ds|) / 1.70.
    sp0 = {"rise": torch.tensor([-30.0, -20.0, -10.0], dtype=torch.float32), "fall": torch.full((3,), 2.0e8)}
    sp1 = {"rise": torch.tensor([-18.0, 0.0, -40.0], dtype=torch.float32), "fall": torch.full((3,), 2.0e8)}
    pw = {"rise": torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32), "fall": torch.zeros(3)}
    # |ds| = [12, 20, 30] with weights [1, 2, 1] -> weighted median = 20.0!
    tau_est, sigma_est = estimate_slack_drift_tau(sp0, sp1, pin_weights=pw, base_tau_ps=1.0)
    assert math.isclose(sigma_est, 1.4826 * 20.0, rel_tol=1e-5)
    assert math.isclose(tau_est, 1.4826 * 20.0 / 1.70, rel_tol=1e-5)

    # Transactional carry_beta in InputAccumulator.
    acc = InputAccumulator(gain=0.2, mode="accumulate")
    z = torch.zeros(3, dtype=torch.float32)
    f1 = {
        "delay": {"rise": torch.tensor([10.0, 20.0, 30.0]), "fall": z.clone()},
        "cap": {"rise": torch.tensor([5.0, 10.0, 15.0]), "fall": z.clone()},
        "impulse": {"rise": z.clone(), "fall": z.clone()},
    }
    c1 = acc.prepare(f1, identity=("a", 3, 1), pin_count=3, iteration=1)
    acc.commit(c1)
    snap_before = acc.values["delay"]["rise"].clone()

    f2 = {
        "delay": {"rise": z.clone(), "fall": z.clone()},
        "cap": {"rise": z.clone(), "fall": z.clone()},
        "impulse": {"rise": z.clone(), "fall": z.clone()},
    }
    cb = {"rise": torch.tensor([0.0, 0.25, 1.0]), "fall": torch.ones(3)}
    c2 = acc.prepare(
        f2,
        identity=("b", 3, 2),
        pin_count=3,
        iteration=2,
        carry_beta=cb,
        allow_identity_change=True,
    )
    # Transactional: acc.values untouched before commit!
    assert torch.allclose(acc.values["delay"]["rise"], snap_before)
    acc.commit(c2)
    assert torch.allclose(
        acc.values["delay"]["rise"],
        torch.tensor([0.0, 1.0, 6.0], dtype=torch.float32),
    )


def test_star_tdp_runtime_slack_beta_and_restart_kappa_guards():
    placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=2,
        num_filler_nodes=0,
        num_nodes=2,
        num_pins=2,
        pin_names=[b"u0/Z", b"u1/A"],
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT"],
    )
    dc = SimpleNamespace(
        pin2net_map=torch.tensor([0, 0], dtype=torch.long),
        pin2node_map=torch.tensor([0, 1], dtype=torch.long),
        net2driver_pin_map=torch.tensor([0], dtype=torch.int32),
        net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
        inst_libcell_offset=torch.tensor([0, 0], dtype=torch.long),
    )

    sim = {"slack_pin1": -40.0, "grad_scale": 10.0}

    class _MockTiming:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 2
        device = torch.device("cpu")
        dtype = torch.float32
        pin_rslack_live_snapshot = None
        pin_fslack_live_snapshot = None
        last_soft_tns_loss = None

        def __call__(self, delays, impulses, loads, **_kwargs):
            s1 = float(sim["slack_pin1"])
            self.pin_rslack_live_snapshot = torch.tensor([2.0e8, s1], dtype=torch.float32)
            self.pin_fslack_live_snapshot = torch.tensor([2.0e8, 2.0e8], dtype=torch.float32)
            d = delays["rise"][1]
            if s1 < 0.0:
                tns = torch.tensor(s1, dtype=torch.float32) - float(sim["grad_scale"]) * d
                self.last_soft_tns_loss = -tns
            else:
                tns = torch.tensor(0.0, dtype=torch.float32)
                self.last_soft_tns_loss = 0.0 * d
            return tns, tns, tns, tns

    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False]),
        pin_caps_op=lambda _off, _dc: (
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _current_timing_surrogate_mode=lambda: "lut_only",
    )
    op_cols = SimpleNamespace(
        timing_propagation_op=_MockTiming(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: (p[1] - p[0]).abs() + (p[3] - p[2]).abs(),
    )
    rt_params = SimpleNamespace(
        scale_factor=1.0,
        ignore_net_degree=100,
        deterministic_flag=True,
        star_rc_calibration_enabled=1,
        star_slope_calibration_enabled=0,
        star_lcb_ff_clock_enabled=0,
        star_kappa=0.05,
        star_max_kappa=0.0,
        star_post_restart_max_kappa=0.20,
        star_kappa_recalibrate_on_restart=1,
        star_restart_history_mode="slack_beta",
        star_restart_beta_tau_mode="drift",
    )
    rt = StarTdpRuntime(
        model=model,
        params=rt_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    pos = torch.tensor([10.0, 90.0, 20.0, 40.0], dtype=torch.float32)

    # Pre-restart GP snapshots: measure pin-slack drift (-40 -> -20 ps => |ds|=20 => tau ~ 17.44 ps),
    # while pre-restart max_kappa=0.0 leaves realized_kappa uncapped (pure AutoDMP!).
    s1 = rt.refresh_snapshot(pos, iteration=10, overflow=0.34)
    assert math.isclose(s1["realized_kappa"], 0.05, rel_tol=1e-4)
    pre_lam = s1["lambda_star"]

    sim["slack_pin1"] = -20.0
    sim["grad_scale"] = 50.0
    s2 = rt.refresh_snapshot(pos, iteration=25, overflow=0.25)
    assert len(rt.slack_drift_tau_history) == 1
    assert s2["realized_kappa"] > 0.25  # pre-restart has NO ceiling (max_kappa=0.0)
    k_before_sizing = s2["k_l1"]

    # S5B1 sizing clears violations to +5.0 ps (tns == 0.0) + restart!
    rt.reset_for_restart(recalibrate_kappa=True)
    assert rt.kappa_matched is False
    sim["slack_pin1"] = 5.0
    sim["grad_scale"] = 0.0
    s3 = rt.refresh_snapshot(pos, iteration=0, overflow=0.22)
    # 1) slack_beta retains a calibrated fraction of W_before (NOT wiped to 0)!
    assert s3["beta_carry_ratio"] is not None
    assert 0.15 < s3["beta_carry_ratio"] < 0.55
    assert 0.15 * k_before_sizing < s3["k_l1"] < 0.55 * k_before_sizing
    # 2) Guard 1: because live_tns == 0.0, kappa recalibration is deferred (does NOT blow up lambda_star,
    #    and holds fallback base_lambda_star capped by post_restart_max_kappa=0.20)!
    assert s3["kappa_matched"] is False
    assert s3["lambda_star"] > 0.0
    assert s3["lambda_star"] <= pre_lam + 1e-6
    assert s3["realized_kappa"] <= 0.20 + 1e-5

    # Next snapshot: violations dip to -12.0 ps -> recalibrates base_lambda_star cleanly,
    # preserving carried slack_beta stiffness bounded by m_eff * requested_kappa and post_restart_max_kappa!
    sim["slack_pin1"] = -12.0
    sim["grad_scale"] = 5.0
    s4 = rt.refresh_snapshot(pos, iteration=15, overflow=0.18)
    assert s4["kappa_matched"] is True
    assert 0.05 < s4["realized_kappa"] <= 0.20 + 1e-5
    expected_target = min(
        0.05 + s3["beta_carry_ratio"] * s2["realized_kappa"],
        0.05 * rt._pending_recal_m_eff,
        0.20,
    )
    assert math.isclose(s4["realized_kappa"], expected_target, rel_tol=1e-4)

    # Guard 2: 20x separation surge post-restart -> post_restart_max_kappa=0.20 caps realized_kappa!
    sim["grad_scale"] = 100.0
    s5 = rt.refresh_snapshot(pos, iteration=30, overflow=0.14)
    assert s5["realized_kappa"] <= 0.20 + 1e-5
    assert math.isclose(s5["realized_kappa"], 0.20, rel_tol=1e-4)

    # MidplaceRepair export -> rebuild with appended buffer pins -> restore!
    exported = rt.export_midplace_state()
    placedb2 = SimpleNamespace(
        num_movable_nodes=3,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=3,
        num_filler_nodes=0,
        num_nodes=3,
        num_pins=4,
        pin_names=[b"u0/Z", b"u1/A", b"buf0/A", b"buf0/Z"],
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT", "INPUT", "OUTPUT"],
    )
    dc2 = SimpleNamespace(
        pin2net_map=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        pin2node_map=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        net2driver_pin_map=torch.tensor([0, 3], dtype=torch.int32),
        net_flat_arcs=torch.tensor([[0, 2], [3, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1, 2], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 2, 3, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2, 4], dtype=torch.int32),
        inst_libcell_offset=torch.tensor([0, 0, 0], dtype=torch.long),
    )

    class _MockTiming4:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 4
        device = torch.device("cpu")
        dtype = torch.float32
        pin_rslack_live_snapshot = None
        pin_fslack_live_snapshot = None
        last_soft_tns_loss = None

        def __call__(self, delays, impulses, loads, **_kwargs):
            self.pin_rslack_live_snapshot = torch.tensor(
                [2.0e8, 4.0, 4.0, 2.0e8], dtype=torch.float32
            )
            self.pin_fslack_live_snapshot = torch.full((4,), 2.0e8, dtype=torch.float32)
            z = torch.tensor(0.0, dtype=torch.float32)
            self.last_soft_tns_loss = 0.0 * delays["rise"].sum()
            return z, z, z, z

    model2 = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False, False, True]),
        pin_caps_op=lambda _off, _dc: (
            torch.ones(4, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _current_timing_surrogate_mode=lambda: "lut_only",
    )
    op_cols2 = SimpleNamespace(
        timing_propagation_op=_MockTiming4(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: p.abs().sum(),
    )
    rt2 = StarTdpRuntime(
        model=model2,
        params=rt_params,
        placedb=placedb2,
        data_collections=dc2,
        op_collections=op_cols2,
    )
    rt2.restore_midplace_state(exported, recalibrate_kappa=True)
    pos2 = torch.tensor([10.0, 90.0, 50.0, 50.0, 20.0, 40.0, 30.0, 30.0], dtype=torch.float32)
    s_mid = rt2.refresh_snapshot(pos2, iteration=0, overflow=0.20)
    # Existing sink pin u1/A retains slack_beta guardband while newly inserted buf0/A starts at 0!
    assert float(rt2.input_accumulator.values["delay"]["rise"][1].item()) > 0.0
    assert float(rt2.input_accumulator.values["delay"]["rise"][2].item()) == 0.0
    assert s_mid["k_l1"] > 0.0


def test_sparse_endpoint_zero_tau_ps_softens_near_zero_margin_endpoints():
    arrays = {
        "pin_state_start": np.array([0, 1, 2, 2], dtype=np.int64),
        "state_pin": np.array([0, 1], dtype=np.int64),
        "state_domain": np.array([0, 0], dtype=np.int64),
        "state_tag": np.array([0, 0], dtype=np.int64),
        "state_launch_tag": np.array([0, 0], dtype=np.int64),
        "state_source_type": np.array([2, 2], dtype=np.int64),
        "state_role": np.array([0, 0], dtype=np.int64),
        "state_level": np.array([1, 2], dtype=np.int64),
        "state_slew_group": np.array([0, 1], dtype=np.int64),
        "state_valid": np.array([[1, 0], [1, 0]], dtype=np.uint8),
        "state_seed": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        "state_reference": np.array([[50.0, -2e8, 5.0, 0.0], [-2e8, -2e8, 0.0, 0.0]], dtype=np.float64),
        "net_edges": np.array([[0, 0, 1]], dtype=np.int64),
        "net_transition_mask": np.array([1], dtype=np.int64),
        "cell_transition_mask": np.empty(0, dtype=np.int64),
        "slew_only_cell_transition_mask": np.empty(0, dtype=np.int64),
        "context_state": np.empty(0, dtype=np.int64),
        "net_threshold_adjustments": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        "cell_edges": np.empty((0, 3), dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array([[1, 0, -1, 0, 0, 1, 1]], dtype=np.int64),
        "check_values": np.array([[60.0, -1.0, 0.0, 2e8]], dtype=np.float64),
    }
    view = SparseTimingView(arrays, ("clk",), ("clk",), (), "ep", "ep")
    lut_val = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float64)
    lut_ax = torch.tensor([[0.0, 100.0]], dtype=torch.float64)
    lut_sz = torch.tensor([[2, 2]], dtype=torch.long)
    lut = LUTS_INFO(lut_val, lut_ax, lut_ax, lut_sz)
    arcs_info = SimpleNamespace(
        r_delay_luts=lut,
        f_delay_luts=lut,
        r_trans_luts=lut,
        f_trans_luts=lut,
        r_check_luts=lut,
        f_check_luts=lut,
    )
    ids = lambda v: torch.tensor(v, dtype=torch.long)
    empty = torch.empty(0, dtype=torch.float64)
    timing = TimingPropagation(
        inrdelays=empty,
        infdelays=empty,
        inrtrans=empty,
        inftrans=empty,
        outcaps=empty,
        pin_net=ids([0, 0, 1]),
        start_points=ids([]),
        end_points=ids([1]),
        clock_pins=ids([2]),
        FF_ids=ids([]),
        clk_pin_rtran=torch.tensor([5.0], dtype=torch.float64),
        clk_pin_ftran=torch.tensor([5.0], dtype=torch.float64),
        net_flat_arcs_start=ids([0, 1, 1]),
        net_flat_arcs=ids([[0, 1]]),
        net2driver_pin_map=ids([0, -1]),
        arcs_info=arcs_info,
        inst_flat_arcs_start=ids([0]),
        inst_flat_arcs=torch.empty((0, 8), dtype=torch.long),
        endpoints_constraint_arcs=ids([[2, 1, 60000, 0, 0, 1, 0, 0]]),
        endpoints_timing_check_arcs=ids([[2, 1, 0, 0, 0, 1, 0, 0]]),
        flat_inst_arcs_by_level=torch.empty((0, 8), dtype=torch.long),
        flat_inst_arcs_by_level_start=ids([0, 0]),
        endpoints_rRAT=torch.tensor([60.0], dtype=torch.float64),
        endpoints_fRAT=torch.tensor([2e8], dtype=torch.float64),
        flat_pin_to_graph=ids([]),
        flat_pin_to_graph_start=ids([0, 0, 0, 0]),
        flat_pin_to_graph_reverse=ids([]),
        flat_pin_to_graph_start_reverse=ids([0, 0, 0, 0]),
        timing_metric_scope="setup_only",
        sparse_timing_view=view,
        timing_propagation_device="cpu",
        production_fast_loop=True,
    )
    zeros = {"rise": torch.zeros(3, dtype=torch.float64), "fall": torch.zeros(3, dtype=torch.float64)}
    # Seed=50, delay=6 -> AAT=56, RAT=60 -> endpoint slack = +4.0 ps (positive near-zero margin!).
    dr_base = torch.tensor([0.0, 6.0, 0.0], dtype=torch.float64, requires_grad=True)
    timing._request_pin_slack_snapshot = True
    timing(
        {"rise": dr_base, "fall": zeros["fall"]},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
    )
    timing._request_pin_slack_snapshot = False
    # Verify _request_pin_slack_snapshot propagated RAT backwards onto driver pin 0 even in production_fast_loop!
    assert math.isclose(float(timing.pin_rslack_live_snapshot[0].item()), 4.0, rel_tol=1e-5)
    assert math.isclose(float(timing.pin_rslack_live_snapshot[1].item()), 4.0, rel_tol=1e-5)
    g_base = float(torch.autograd.grad(timing.last_soft_tns_loss, dr_base)[0][1].item())

    dr_drift = torch.tensor([0.0, 6.0, 0.0], dtype=torch.float64, requires_grad=True)
    timing(
        {"rise": dr_drift, "fall": zeros["fall"]},
        zeros,
        zeros,
        surrogate_mode="lut_only",
        smooth_extremum=True,
        extremum_tau_ps=1.0,
        endpoint_zero_tau_ps=20.0,
    )
    g_drift = float(torch.autograd.grad(timing.last_soft_tns_loss, dr_drift)[0][1].item())
    assert math.isclose(g_base, float(torch.sigmoid(torch.tensor(-4.0)).item()), rel_tol=1e-5)
    assert math.isclose(g_drift, float(torch.sigmoid(torch.tensor(-4.0 / 20.0)).item()), rel_tol=1e-5)
    assert g_drift > 20.0 * g_base


def test_row_gap_blockages_extend_strip_and_insert():
    from dreamplace.row_gap_blockage import row_gap_boxes, row_gap_blockages

    # Layout [0, 100] x [0, 20], row_height=10, site_width=1.
    # Row 0 spans [0, 100]; Row 1 spans [20, 80] leaving two gaps (with 0.25 site/row margin).
    rows = [(0.0, 0.0, 100.0, 10.0), (20.0, 10.0, 80.0, 20.0)]
    boxes = row_gap_boxes(rows, 0.0, 0.0, 100.0, 20.0, 1.0, 10.0)
    assert boxes.tolist() == [[-0.25, 12.5, 19.75, 17.5], [80.25, 12.5, 100.25, 17.5]]

    placedb = SimpleNamespace(
        rows=rows,
        xl=0.0,
        yl=0.0,
        xh=100.0,
        yh=20.0,
        site_width=1.0,
        row_height=10.0,
        num_movable_nodes=2,
        num_terminals=1,
        num_terminal_NIs=0,
        num_filler_nodes=1,
        num_nodes=4,
    )
    dc = SimpleNamespace(
        node_size_x=torch.tensor([4.0, 6.0, 10.0, 2.0], dtype=torch.float32),
        node_size_y=torch.tensor([10.0, 10.0, 20.0, 10.0], dtype=torch.float32),
        num_pins_in_nodes=torch.tensor([3.0, 2.0, 0.0, 0.0], dtype=torch.float32),
        node2fence_region_map=torch.tensor([4, 4, 4, 4], dtype=torch.int32),
    )
    gaps = row_gap_blockages(placedb, dc)
    assert gaps is not None
    assert gaps.count == 2
    assert gaps.num_terminals == 3
    assert gaps.node_size_x.tolist() == [4.0, 6.0, 10.0, 20.0, 20.0, 2.0]

    pos = torch.tensor([5.0, 30.0, 50.0, 90.0, 0.0, 10.0, 0.0, 0.0], dtype=torch.float32)
    ext = gaps.extend(pos)
    assert ext.numel() == 12
    # Fixed gap x/y coordinates inserted at indices 3 and 4:
    assert ext[3:5].tolist() == [-0.25, 80.25]
    assert ext[6 + 3 : 6 + 5].tolist() == [12.5, 12.5]
    assert torch.equal(gaps.strip(ext), pos)


def test_macro_exclusion_field_gradient_hessian_kappa_evac_and_sizing_sync():
    from dreamplace.ops.macro_exclusion_field import (
        MacroExclusionField,
        extract_macro_and_gap_boxes,
    )

    # 3 movable cells (0, 1, 2), 2 fixed terminals:
    # - node 3: 1-row fixed standard cell (height=10) -> must NOT be haloed
    # - node 4: 4-row fixed macro (height=40) at [40, 20, 80, 60] -> haloed
    # Plus a short row gap on row 0 at [0, 0, 20, 10].
    rows = [(20.0, 0.0, 120.0, 10.0)] + [
        (0.0, float(r * 10), 120.0, float((r + 1) * 10)) for r in range(1, 10)
    ]
    placedb = SimpleNamespace(
        num_nodes=5,
        num_movable_nodes=3,
        num_terminals=2,
        num_terminal_NIs=0,
        num_filler_nodes=0,
        xl=0.0,
        yl=0.0,
        xh=120.0,
        yh=100.0,
        row_height=10.0,
        site_width=1.0,
        dbu_per_micron=1.0,
        rows=rows,
        node_names=[b"u_crit", b"u_safe", b"u_far", b"u_fixed_1row", b"u_macro"],
        node_size_x=torch.tensor([4.0, 4.0, 4.0, 4.0, 40.0], dtype=torch.float64),
        node_size_y=torch.tensor([10.0, 10.0, 10.0, 10.0, 40.0], dtype=torch.float64),
        pin_direct=[b"OUTPUT", b"INPUT", b"OUTPUT", b"INPUT"],
    )
    pos = torch.tensor(
        [
            # x: cell 0 & 1 inside macro [40..80], cell 2 far away at x=105
            52.0, 52.0, 105.0, 95.0, 40.0,
            # y: cell 0 & 1 inside macro [20..60], cell 2 far away at y=85
            33.0, 33.0, 85.0, 80.0, 20.0,
        ],
        dtype=torch.float64,
    )
    boxes, box_stats = extract_macro_and_gap_boxes(placedb, pos)
    assert box_stats["macro_base_instances"] == 1
    assert box_stats["row_gap_boxes"] == 1
    assert boxes.shape == (2, 4)

    dc = SimpleNamespace(
        node_size_x=placedb.node_size_x.clone(),
        node_size_y=placedb.node_size_y.clone(),
        pin_offset_x=torch.zeros(4, dtype=torch.float64),
        pin_offset_y=torch.zeros(4, dtype=torch.float64),
        # Net 0: pin 0 (cell 0, critical) -> pin 1 (fixed node 3);
        # Net 1: pin 2 (cell 1, +5000ps safe) -> pin 3 (cell 2, +5000ps far).
        pin2node_map=torch.tensor([0, 3, 1, 2], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2, 4], dtype=torch.int32),
        net_mask_ignore_large_degrees=torch.tensor([1, 1], dtype=torch.uint8),
        num_pins_in_nodes=torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0], dtype=torch.float64),
    )

    field = MacroExclusionField(
        placedb,
        dc,
        pos,
        ov_start=0.30,
        ov_full=0.10,
        warmup_iters=0,
        ramp_iters=10,
        slack_gate_mode="kappa_evac",
    )
    assert field.has_obstacles is True

    # Pin 0 (on cell 0) is deeply critical (-250 ps); Pins 2,3 (on cell 1 & 2) have +5000 ps safe slack.
    fake_timing_op = SimpleNamespace(
        timing_state_generation=1,
        pin_rslack_live_snapshot=torch.tensor([-250.0, -250.0, 5000.0, 5000.0], dtype=torch.float64),
        pin_fslack_live_snapshot=torch.tensor([-250.0, -250.0, 5000.0, 5000.0], dtype=torch.float64),
    )
    fake_star = SimpleNamespace(
        snapshot_generation=1,
        coordinate_to_um=1.0,
        r_unit=0.2,
        c_unit=0.1,
        driver_pin_ids=torch.tensor([0, 2], dtype=torch.long),
        sink_pin_ids=torch.tensor([1, 3], dtype=torch.long),
        pair_net_ids=torch.tensor([0, 1], dtype=torch.long),
        num_nets=2,
        frozen_weight=torch.tensor([2.0, 0.0], dtype=torch.float32),
        frozen_load_length_weight=torch.tensor([0.5, 0.0], dtype=torch.float32),
        geometry=lambda _p: {"length_um": torch.tensor([10.0, 50.0], dtype=torch.float32)},
    )
    owner = SimpleNamespace(
        data_collections=dc,
        op_collections=SimpleNamespace(
            timing_propagation_op=fake_timing_op,
            pin_pos_op=lambda p: p[:8] if p.numel() >= 8 else torch.zeros(8, dtype=p.dtype),
        ),
        _star_pin_pair_op=lambda: fake_star,
        _driver_sink_star_base_cap_map=lambda: {"generic": torch.ones(4, dtype=torch.float32)},
        segment_timing_coefficient=1.0,
        refresh_count=1,
        _committed_pin_slack=None,
    )
    field.refresh_criticality_pulls(owner, pos)
    # Critical cell 0 receives strictly positive evacuation pull, while +5000ps safe cell 1 & 2 are gated to 0!
    w_all = field.movable_base_pull.detach().cpu()
    assert float(w_all[0].item()) > 1.0
    assert float(w_all[1].item()) == 0.0
    assert float(w_all[2].item()) == 0.0

    # Central finite-difference verification of exact cubic B-spline objective gradient:
    p_var = pos.clone().detach().requires_grad_(True)
    obj = field.objective(owner, p_var, overflow=0.05)
    assert float(obj.item()) > 0.0
    (grad_analytic,) = torch.autograd.grad(obj, p_var)
    eps = 1e-5
    for idx in (0, 5):  # cell 0 x and y
        p_plus = pos.clone()
        p_minus = pos.clone()
        p_plus[idx] += eps
        p_minus[idx] -= eps
        fd = float(
            (
                field.objective(owner, p_plus, overflow=0.05)
                - field.objective(owner, p_minus, overflow=0.05)
            ).item()
        ) / (2.0 * eps)
        assert math.isclose(float(grad_analytic[idx].item()), fd, rel_tol=1e-4, abs_tol=1e-5)
    # Safe cell 1 inside macro and far cell 2 outside support both receive zero macro gradient:
    assert float(grad_analytic[1].abs().item()) == 0.0
    assert float(grad_analytic[2].abs().item()) == 0.0

    # Secant MM diagonal Hessian (2, num_nodes) is strictly positive on active interior cell 0 and zero on cell 1 & 2:
    h_diag = field.diagonal_hessian(owner, pos, overflow=0.05)
    assert h_diag.shape == (2, 5)
    assert float(h_diag[0, 0].item()) > 0.0
    assert float(h_diag[1, 0].item()) > 0.0
    assert float(h_diag[0, 1].item()) == 0.0
    assert float(h_diag[0, 2].item()) == 0.0
    assert torch.all(h_diag >= 0.0)

    # Sizing geometry sync updates movable half-widths:
    dc.node_size_x = torch.tensor([12.0, 4.0, 4.0, 4.0, 40.0], dtype=torch.float64)
    field.sync_movable_geometry(dc)
    assert math.isclose(float(field.movable_half_w[0].item()), 6.0, rel_tol=1e-6)


def test_d_metric_legalize_anchoring_and_carried_buffer_inheritance():
    from dreamplace.ops.greedy_legalize.greedy_legalize import GreedyLegalize
    from dreamplace.ops.abacus_legalize.abacus_legalize import AbacusLegalize
    from dreamplace.NonLinearPlace import NonLinearPlace

    # 3 movable cells of width 10 stacked at the exact same x=40 on row [0, 100] x [0, 10]:
    # cell 0 is critical (weight=32.0), cells 1 & 2 are non-critical (weight=1.0).
    node_size_x = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    node_size_y = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    pin_weights = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    flat_region_boxes = torch.empty(0, dtype=torch.float32)
    flat_region_boxes_start = torch.tensor([0], dtype=torch.int32)
    node2fence = torch.tensor([99, 99, 99], dtype=torch.int32)
    init_pos = torch.tensor([40.0, 40.0, 40.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    kw = dict(
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        flat_region_boxes=flat_region_boxes,
        flat_region_boxes_start=flat_region_boxes_start,
        node2fence_region_map=node2fence,
        xl=0.0,
        yl=0.0,
        xh=100.0,
        yh=10.0,
        site_width=1.0,
        row_height=10.0,
        num_bins_x=1,
        num_bins_y=1,
        num_movable_nodes=3,
        num_terminal_NIs=0,
        num_filler_nodes=0,
    )
    g_base = GreedyLegalize(node_weights=pin_weights, use_node_weights=False, anchor_boost=0.0, **kw)
    a_base = AbacusLegalize(node_weights=pin_weights, use_node_weights=False, **kw)
    p_base = a_base(init_pos, g_base(init_pos, init_pos.clone()))

    # Verify use_node_weights=False is invariant to arbitrary node_weights tensor:
    d_weights_1d = torch.tensor([32.0, 1.0, 1.0], dtype=torch.float32)
    d_weights_2d = torch.cat([d_weights_1d, d_weights_1d], dim=0)
    g_off = GreedyLegalize(node_weights=d_weights_2d, use_node_weights=False, anchor_boost=0.0, **kw)
    a_off = AbacusLegalize(node_weights=d_weights_1d, use_node_weights=False, **kw)
    p_off = a_off(init_pos, g_off(init_pos, init_pos.clone()))
    assert torch.equal(p_base, p_off)

    # Now enable D-metric weights: cell 0 (stiffness=32) must stay anchored much closer to x=40!
    g_on = GreedyLegalize(node_weights=d_weights_2d, use_node_weights=True, anchor_boost=0.5, **kw)
    a_on = AbacusLegalize(node_weights=d_weights_1d, use_node_weights=True, **kw)
    p_on = a_on(init_pos, g_on(init_pos, init_pos.clone()))
    # Check non-overlapping legal x coordinates in [0, 90]:
    xs = sorted(p_on[:3].tolist())
    assert xs[0] >= 0.0 and xs[2] + 10.0 <= 100.0
    assert xs[1] >= xs[0] + 10.0 - 1e-5 and xs[2] >= xs[1] + 10.0 - 1e-5
    disp0_base = abs(float(p_base[0].item()) - 40.0)
    disp0_on = abs(float(p_on[0].item()) - 40.0)
    assert disp0_on < disp0_base

    # Verify NonLinearPlace export + carried terminal legalization 2-hop buffer inheritance:
    gp_placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_nodes=2,
        node_names=[b"u_drv", b"u_sink"],
    )
    gp_params = SimpleNamespace(
        d_metric_legalize_flag=1,
        d_metric_legalize_max_weight=64.0,
        d_metric_legalize_anchor_boost=0.5,
    )
    dummy_gp = SimpleNamespace(
        _routability_model=SimpleNamespace(
            op_collections=SimpleNamespace(
                precondition_op=SimpleNamespace(
                    _last_d_metric_base=torch.tensor([2.0, 4.0], dtype=torch.float32),
                    _last_d_metric_timing_x=torch.tensor([40.0, 4.0], dtype=torch.float32),
                    _last_d_metric_timing_y=torch.tensor([10.0, 4.0], dtype=torch.float32),
                )
            )
        ),
        _decode_node_name=NonLinearPlace._decode_node_name,
    )
    carried = NonLinearPlace.export_d_metric_movable_map(dummy_gp, gp_params, gp_placedb)
    assert carried is not None
    assert math.isclose(carried["node_map"]["u_drv"], 20.0, rel_tol=1e-5)
    assert "u_sink" not in carried["node_map"]

    # Terminal rebuild inserted new buffer u_buf_0 on net 0 driven by u_drv:
    term_placedb = SimpleNamespace(
        num_movable_nodes=3,
        num_nodes=4,  # +1 fixed node
        node_names=[b"u_drv", b"u_sink", b"u_buf_0", b"FIXED_IO"],
        _carried_d_metric_movable_map=carried,
    )
    dummy_term = SimpleNamespace(
        _routability_model=None,
        _decode_node_name=NonLinearPlace._decode_node_name,
        data_collections=SimpleNamespace(
            d_metric_legalize_weights=None,
            # pin 0 on u_drv (net 0), pin 1 on u_buf_0 (net 0), pin 2 on u_buf_0 (net 1), pin 3 on u_sink (net 1)
            pin2node_map=torch.tensor([0, 2, 2, 1], dtype=torch.long),
            pin2net_map=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ),
    )
    NonLinearPlace._prepare_d_metric_legalize_weights(dummy_term, gp_params, term_placedb)
    ex, ey = dummy_term.data_collections.d_metric_legalize_weights
    assert torch.equal(ex, ey)
    # u_drv=20.0, u_sink=1.0, newly materialized u_buf_0 inherited 20.0 from net 0, FIXED_IO=1.0!
    assert math.isclose(float(ex[0].item()), 20.0, rel_tol=1e-5)
    assert math.isclose(float(ex[1].item()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(ex[2].item()), 20.0, rel_tol=1e-5)
    assert math.isclose(float(ex[3].item()), 1.0, rel_tol=1e-5)


def test_delay_sigmoid_clears_cap_and_fixes_m_eff_scale_and_consecutive_restarts():
    from dreamplace.ops.star_pin_pair.slack_beta import (
        compute_delay_sigmoid_retention_beta,
    )

    s_after = {
        "rise": torch.tensor([-300.0, 0.0, 120.0, 2.0e8], dtype=torch.float32),
        "fall": torch.tensor([-30.0, 30.0, 300.0, float("inf")], dtype=torch.float32),
    }
    beta = compute_delay_sigmoid_retention_beta(s_after, tau_ps=30.0)
    assert set(beta.keys()) == {"delay", "cap"}
    assert torch.count_nonzero(beta["cap"]["rise"]) == 0
    assert torch.count_nonzero(beta["cap"]["fall"]) == 0
    assert math.isclose(float(beta["delay"]["rise"][0].item()), 1.0 / (1.0 + math.exp(-10.0)), rel_tol=1e-5)
    assert math.isclose(float(beta["delay"]["rise"][1].item()), 0.5, rel_tol=1e-5)
    assert math.isclose(float(beta["delay"]["rise"][2].item()), 1.0 / (1.0 + math.exp(4.0)), rel_tol=1e-5)
    assert float(beta["delay"]["rise"][3].item()) == 0.0
    assert float(beta["delay"]["fall"][3].item()) == 0.0

    # End-to-end runtime check: delay_sigmoid clears cap history, retains delay history,
    # computes m_eff relative to gain*fresh (not unscaled fresh), and protects
    # _pre_restart_realized_kappa across deferred/consecutive restarts.
    placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=2,
        num_filler_nodes=0,
        num_nodes=2,
        num_pins=2,
        pin_names=[b"u0/Z", b"u1/A"],
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT"],
    )
    offsets = torch.tensor([0, 0], dtype=torch.long)
    dc = SimpleNamespace(
        pin2net_map=torch.tensor([0, 0], dtype=torch.long),
        pin2node_map=torch.tensor([0, 1], dtype=torch.long),
        net2driver_pin_map=torch.tensor([0], dtype=torch.int32),
        net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
        inst_libcell_offset=offsets,
    )
    sim = {"slack": -60.0, "delay_grad": 1.0, "cap_grad": 2.0}

    class _MockTimingCapDelay:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 2
        device = torch.device("cpu")
        dtype = torch.float32
        pin_rslack_live_snapshot = None
        pin_fslack_live_snapshot = None
        last_soft_tns_loss = None

        def __call__(self, delays, impulses, loads, **_kwargs):
            s = float(sim["slack"])
            self.pin_rslack_live_snapshot = torch.tensor([s, s], dtype=torch.float32)
            self.pin_fslack_live_snapshot = torch.full((2,), 2.0e8, dtype=torch.float32)
            tns_v = torch.tensor(min(0.0, s), dtype=torch.float32)
            self.last_soft_tns_loss = (
                float(sim["delay_grad"]) * delays["rise"][1]
                + float(sim["cap_grad"]) * loads["rise"][0]
            )
            return tns_v, tns_v, tns_v, tns_v

    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False]),
        pin_caps_op=lambda _off, _dc: (
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _current_timing_surrogate_mode=lambda: "lut_only",
    )
    elmore_op = SimpleNamespace(r_unit=0.1, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    op_cols = SimpleNamespace(
        timing_propagation_op=_MockTimingCapDelay(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: p.abs().sum(),
    )
    params = SimpleNamespace(
        scale_factor=1.0,
        ignore_net_degree=100,
        star_kappa=0.05,
        star_max_kappa=0.0,
        star_post_restart_max_kappa=0.20,
        star_kappa_recalibrate_on_restart=1,
        star_restart_history_mode="delay_sigmoid",
        star_restart_beta_tau_mode="fixed",
        star_restart_beta_fixed_tau_ps=30.0,
        star_rc_calibration_enabled=0,
        star_slope_calibration_enabled=0,
        star_lcb_ff_clock_enabled=0,
    )
    rt = StarTdpRuntime(
        model=model,
        params=params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    pos = torch.tensor([0.0, 50.0, 0.0, 50.0], dtype=torch.float32)
    # 3 refreshes at identical unit gradient -> accumulator holds 3 * 0.2 = 0.6 on both delay and cap
    rt.refresh_snapshot(pos, iteration=10, overflow=0.30)
    rt.refresh_snapshot(pos, iteration=15, overflow=0.28)
    s_pre = rt.refresh_snapshot(pos, iteration=20, overflow=0.26)
    assert math.isclose(s_pre["realized_kappa"], 0.15, rel_tol=1e-4)

    # Inline S5B1 sizing sub-round 1 clears slack to +5 ps (deferred recalibration because live_tns == 0)
    offsets[0] = 1
    sim["slack"] = 5.0
    sim["delay_grad"] = 0.0
    sim["cap_grad"] = 0.0
    s_def = rt.refresh_snapshot(pos, iteration=25, overflow=0.25)
    assert s_def["kappa_matched"] is False
    # cap history MUST be completely cleared, while delay history is retained by sigmoid(-5/30)
    assert float(rt.input_accumulator.values["cap"]["rise"].abs().sum().item()) == 0.0
    assert float(rt.input_accumulator.values["delay"]["rise"][1].item()) > 0.0

    # Inline S5B1 sizing sub-round 2 BEFORE kappa matched -> must NOT overwrite _pre_restart_realized_kappa (0.15)!
    offsets[1] = 1
    sim["slack"] = -30.0
    sim["delay_grad"] = 1.0
    sim["cap_grad"] = 2.0
    s_recal = rt.refresh_snapshot(pos, iteration=30, overflow=0.24)
    assert math.isclose(rt._pre_restart_realized_kappa, 0.15, rel_tol=1e-4)
    assert s_recal["kappa_matched"] is True
    # Because cap history was cleared while delay history survived, m_eff is strictly > 1.0 (not crushed by 5x bug)
    # and strictly < delay-only multiple (does not over-inflate fresh B+):
    assert 1.05 < rt._pending_recal_m_eff < 2.2

    # Also verify inline master_changed in "reset" mode truly clears InputAccumulator:
    params_reset = SimpleNamespace(**{**params.__dict__, "star_restart_history_mode": "reset"})
    rt_reset = StarTdpRuntime(
        model=model,
        params=params_reset,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    rt_reset.refresh_snapshot(pos, iteration=10, overflow=0.30)
    rt_reset.refresh_snapshot(pos, iteration=15, overflow=0.28)
    assert float(rt_reset.input_accumulator.values["delay"]["rise"][1].item()) > 0.35
    offsets[0] = 2  # inline S5B1 sizing!
    s_rst = rt_reset.refresh_snapshot(pos, iteration=20, overflow=0.26)
    assert math.isclose(float(rt_reset.input_accumulator.values["delay"]["rise"][1].item()), 0.2, rel_tol=1e-5)
    assert math.isclose(s_rst["realized_kappa"], 0.05, rel_tol=1e-4)

    # Verify selective_sigmoid: resizing sink node 1 keeps untouched driver pin 0 cap history
    # gated by sigmoid(-slack/tau), whereas resizing driver node 0 zeroes driver pin 0 cap history!
    params_sel = SimpleNamespace(**{**params.__dict__, "star_restart_history_mode": "selective_sigmoid"})
    rt_sel = StarTdpRuntime(
        model=model,
        params=params_sel,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    offsets[0] = 0
    offsets[1] = 0
    sim["slack"] = -30.0
    sim["delay_grad"] = 1.0
    sim["cap_grad"] = 2.0
    rt_sel.refresh_snapshot(pos, iteration=10, overflow=0.30)
    rt_sel.refresh_snapshot(pos, iteration=15, overflow=0.28)
    cap_before = float(rt_sel.input_accumulator.values["cap"]["rise"][0].item())
    assert math.isclose(cap_before, 0.8, rel_tol=1e-5)

    # Sink node 1 resized (driver node 0 untouched, net degree/pins untouched), slack = 0 ps -> gate = 0.5
    offsets[1] = 3
    sim["slack"] = 0.0
    sim["delay_grad"] = 0.0
    sim["cap_grad"] = 0.0
    rt_sel.refresh_snapshot(pos, iteration=20, overflow=0.26)
    # Untouched driver pin 0 retains 0.5 * 0.8 = 0.4 cap history!
    assert math.isclose(float(rt_sel.input_accumulator.values["cap"]["rise"][0].item()), 0.4, rel_tol=1e-4)

    # Now resize driver node 0 -> driver pin 0 cap history MUST be zeroed!
    offsets[0] = 4
    rt_sel.refresh_snapshot(pos, iteration=25, overflow=0.25)
    assert float(rt_sel.input_accumulator.values["cap"]["rise"][0].item()) == 0.0


def test_wa_net_two_pin_exact_identity_and_multi_fanout_parasitic_elimination():
    """Verify 2-pin exact identity with star WA, multi-fanout interior sink decoupling, and large-DBU stability."""
    # Net 0: 2-pin net (driver 0 -> sink 1)
    # Net 1: 5-pin net (driver 2 -> sinks 3, 4, 5, 6), where sink 4 sits strictly inside bbox
    net2driver = torch.tensor([0, 2], dtype=torch.long)
    arcs = torch.tensor(
        [[0, 1], [2, 3], [2, 4], [2, 5], [2, 6]], dtype=torch.long
    )
    arcs_start = torch.tensor([0, 1, 5], dtype=torch.long)
    base_caps = {
        m: torch.full((7,), 0.002, dtype=torch.float64)
        for m in ("generic", "rise", "fall")
    }
    star_legacy = StarPinPair(
        net2driver_pin_map=net2driver,
        timing_net_flat_arcs=arcs,
        timing_net_flat_arcs_start=arcs_start,
        num_pins=7,
        r_unit=0.5,
        c_unit=0.0002,
        scale_factor=1.0,
        dbu=2000.0,
        load_geometry="star",
    )
    star_wa = StarPinPair(
        net2driver_pin_map=net2driver,
        timing_net_flat_arcs=arcs,
        timing_net_flat_arcs_start=arcs_start,
        num_pins=7,
        r_unit=0.5,
        c_unit=0.0002,
        scale_factor=1.0,
        dbu=2000.0,
        load_geometry="wa_net",
    )

    # 1. Exact 2-pin identity (only Net 0 active in cap_gradients)
    cap_grad_2pin = {
        "rise": torch.tensor([15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "fall": torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    }
    b_leg_2 = star_legacy.load_length_adjoint(cap_grad_2pin)
    b_wa_2 = star_wa.load_length_adjoint(cap_grad_2pin)
    assert torch.allclose(b_leg_2, b_wa_2, rtol=1e-12, atol=1e-12)

    zero_k = torch.zeros(5, dtype=torch.float64)
    pin_pos = torch.tensor(
        [
            1000.0, 4200.0, 0.0, -20000.0, 100.0, 20000.0, 10000.0,  # x
            -500.0, 1800.0, 0.0, -20000.0, -200.0, 20000.0, -10000.0,  # y
        ],
        dtype=torch.float64,
    )
    gamma = 400.0
    len_um = star_legacy.length_um(pin_pos)
    star_legacy.publish_snapshot(
        star_legacy.prepare_snapshot(
            length_um=len_um, raw_weight=zero_k, load_length_weight=b_leg_2
        )
    )
    star_wa.publish_snapshot(
        star_wa.prepare_snapshot(
            length_um=len_um, raw_weight=zero_k, load_length_weight=b_wa_2
        )
    )

    p1 = pin_pos.clone().requires_grad_(True)
    p2 = pin_pos.clone().requires_grad_(True)
    obj1 = star_legacy.structured_delay_objective(
        p1, base_caps=base_caps, distance_model="wa", smoothing_gamma=gamma
    )
    obj2 = star_wa.structured_delay_objective(
        p2, base_caps=base_caps, distance_model="wa", smoothing_gamma=gamma
    )
    assert torch.allclose(obj1, obj2, rtol=1e-12, atol=1e-12)
    g1 = torch.autograd.grad(obj1, p1)[0]
    g2 = torch.autograd.grad(obj2, p2)[0]
    assert torch.allclose(g1, g2, rtol=1e-12, atol=1e-12)

    pin_map = torch.arange(7, dtype=torch.long)
    tmpl = torch.zeros(7, dtype=torch.float64)
    dx1, dy1 = star_legacy.timing_wa_curvature_diagonal(
        pin_pos,
        base_caps=base_caps,
        smoothing_gamma=gamma,
        pin2node_map=pin_map,
        num_movable_nodes=7,
        num_nodes=7,
        output_template=tmpl,
    )
    dx2, dy2 = star_wa.timing_wa_curvature_diagonal(
        pin_pos,
        base_caps=base_caps,
        smoothing_gamma=gamma,
        pin2node_map=pin_map,
        num_movable_nodes=7,
        num_nodes=7,
        output_template=tmpl,
    )
    # Verify exact match at coincident 2-pin coordinates (z = 0 -> 2/gamma)
    coincident_pos = pin_pos.clone()
    coincident_pos[1] = coincident_pos[0]
    coincident_pos[7 + 1] = coincident_pos[7 + 0]
    cdx1, cdy1 = star_legacy.timing_wa_curvature_diagonal(
        coincident_pos,
        base_caps=base_caps,
        smoothing_gamma=gamma,
        pin2node_map=pin_map,
        num_movable_nodes=7,
        num_nodes=7,
        output_template=tmpl,
    )
    cdx2, cdy2 = star_wa.timing_wa_curvature_diagonal(
        coincident_pos,
        base_caps=base_caps,
        smoothing_gamma=gamma,
        pin2node_map=pin_map,
        num_movable_nodes=7,
        num_nodes=7,
        output_template=tmpl,
    )
    assert torch.allclose(cdx1, cdx2, rtol=1e-12, atol=1e-12)
    assert torch.allclose(cdy1, cdy2, rtol=1e-12, atol=1e-12)
    assert dx2.min().item() >= 0.0 and dy2.min().item() >= 0.0

    # 2. Multi-fanout net (Net 1: fanout 4, interior non-critical sink pin 4)
    # Suppose FLUTE Steiner WL on Net 1 is 1.25x HPWL (HPWL = (40000 + 40000) / 2000 = 40.0 um -> teacher = 50.0 um)
    _, s_loads, s_delays, _, _, s_impulses = star_wa.rc_forward(len_um, base_caps)
    from dreamplace.ops.star_pin_pair import (
        StarRCCalibrationState,
        pair_wirelength_by_net,
        rowsum_hessian,
    )
    star_wl = pair_wirelength_by_net(len_um, star_wa.pair_net_ids, 2)
    teacher_wl = torch.tensor([star_wl[0].item(), 50.0], dtype=torch.float64)
    cal = StarRCCalibrationState.from_reference(
        pair_identity_hash=star_wa.pair_identity_hash,
        pair_topology_generation=star_wa.pair_topology_generation,
        source_snapshot_generation=star_wa.snapshot_generation,
        teacher_topology_generation=1,
        position=pin_pos,
        cap_digest="cap_test",
        r_unit=0.5,
        c_unit=0.0002,
        coordinate_to_um=2000.0,
        pair_net_ids=star_wa.pair_net_ids,
        driver_pin_ids=star_wa.driver_pin_ids,
        sink_pin_ids=star_wa.sink_pin_ids,
        base_caps=base_caps,
        star_wirelength_by_net=star_wl,
        teacher_wirelength_by_net=teacher_wl,
        star_loads=s_loads,
        teacher_loads=s_loads,
        star_delays=s_delays,
        teacher_delays=s_delays,
        teacher_impulses=s_impulses,
    )
    assert math.isclose(float(cal.anchor_steiner_to_hpwl_ratio[0].item()), 1.0, rel_tol=1e-9)
    assert math.isclose(float(cal.anchor_steiner_to_hpwl_ratio[1].item()), 1.25, rel_tol=1e-9)

    cap_grad_mf = {
        "rise": torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "fall": torch.zeros(7, dtype=torch.float64),
    }
    b_leg_mf = star_legacy.load_length_adjoint(cap_grad_mf, cal)
    b_wa_mf = star_wa.load_length_adjoint(cap_grad_mf, cal)
    # By Hanan's RSMT strip theorem, sum of B over Net 1 pairs in wa_net equals c_unit * 1.0 * H_drv = 0.0002 * 100 = 0.020 (NOT 4x!)
    assert math.isclose(float(b_wa_mf[1:].sum().item()), 0.020, rel_tol=1e-10)

    star_legacy.publish_snapshot(
        star_legacy.prepare_snapshot(
            length_um=len_um, raw_weight=zero_k, load_length_weight=b_leg_mf
        )
    )
    star_wa.publish_snapshot(
        star_wa.prepare_snapshot(
            length_um=len_um, raw_weight=zero_k, load_length_weight=b_wa_mf
        )
    )
    pm1 = pin_pos.clone().requires_grad_(True)
    pm2 = pin_pos.clone().requires_grad_(True)
    gm1 = torch.autograd.grad(
        star_legacy.structured_delay_objective(
            pm1, base_caps=base_caps, distance_model="l1", smoothing_gamma=gamma
        ),
        pm1,
    )[0]
    gm2 = torch.autograd.grad(
        star_wa.structured_delay_objective(
            pm2, base_caps=base_caps, distance_model="l1", smoothing_gamma=gamma
        ),
        pm2,
    )[0]
    # Interior sink pin 4 is at (100, -200) inside [-20000, 20000]^2 with gamma=400:
    # legacy star pulls pin 4 strongly toward driver (0, 0), whereas wa_net force on pin 4 is ~0!
    assert float(gm1[4].abs().item()) > 1e-6
    assert float(gm2[4].abs().item()) < 1e-18
    assert float(gm2[7 + 4].abs().item()) < 1e-18

    # 3. Large-DBU float32 stability (2e7 DBU offset, ~4 DBU span -> zero exp overflow & zero WA cancellation)
    shift = 2.0e7
    pin_pos_f32 = (pin_pos * 1e-3 + shift).to(dtype=torch.float32)
    pin_pos_ref = pin_pos_f32.double() - shift
    wa_shifted = star_wa.net_wa_wirelength_um(pin_pos_f32, smoothing_gamma=2.0)
    wa_ref = star_wa.net_wa_wirelength_um(pin_pos_ref, smoothing_gamma=2.0)
    assert torch.isfinite(wa_shifted).all()
    assert torch.allclose(wa_shifted.double(), wa_ref, rtol=1e-5, atol=1e-6)

    # 4. Verify rowsum_hessian with wa_net and distance_model="l1" (owner gamma fallback)
    owner = SimpleNamespace(
        _star_pin_pair_op=lambda: star_wa,
        data_collections=SimpleNamespace(pin2node_map=pin_map),
        placedb=SimpleNamespace(num_movable_nodes=7, num_nodes=7),
        segment_timing_coefficient=2.5,
        _segment_timing_cut_effective_smoothing_gamma=lambda: None,
        gamma=torch.tensor([gamma], dtype=torch.float64),
        op_collections=SimpleNamespace(pin_pos_op=lambda p: p),
        _driver_sink_star_base_cap_map=lambda: base_caps,
        _segment_timing_star_load_curvature=lambda: True,
        _segment_timing_star_pair_objective=lambda: "elmore",
    )
    rh, rg = rowsum_hessian(owner, pin_pos, elmore_axes=2, full_slope=True)
    assert math.isclose(rg, gamma, rel_tol=1e-12)
    assert rh.shape == (2, 7) and torch.isfinite(rh).all() and float(rh.min().item()) >= 0.0
    # Interior non-critical sink pin 4 (K=0) must also receive ~0 rowsum curvature under wa_net!
    assert float(rh[:, 4].abs().max().item()) < 1e-16


def test_wa_net_curvature_diagonal_strict_psd_majorization():
    """Verify diag(bound) - H_exact and diag(bound) + H_exact are PSD across multi-fanout, fixed, and multi-pin-same-cell nets."""
    torch.manual_seed(20261114)
    # 3 nets:
    # Net 0: driver 0 -> sinks 1, 2, 3, 4 (pins 1 & 2 share movable cell 1; pin 4 on fixed cell 5)
    # Net 1: driver 5 -> sinks 6, 7 (all 3 pins on movable cell 2 -> single-cell net, exact H=0!)
    # Net 2: driver 8 -> sinks 9, 10 (mixed movable cells 0, 3, 4)
    net2driver = torch.tensor([0, 5, 8], dtype=torch.long)
    arcs = torch.tensor(
        [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [5, 6], [5, 7],
            [8, 9], [8, 10],
        ],
        dtype=torch.long,
    )
    arcs_start = torch.tensor([0, 4, 6, 8], dtype=torch.long)
    # 5 movable cells (0..4) + 1 fixed cell (5) = 6 nodes total
    pin2node = torch.tensor([0, 1, 1, 3, 5, 2, 2, 2, 4, 0, 3], dtype=torch.long)
    pin_offsets_x = torch.randn(11, dtype=torch.float64) * 15.0
    pin_offsets_y = torch.randn(11, dtype=torch.float64) * 15.0
    fixed_xy = torch.tensor([50.0, -30.0], dtype=torch.float64)
    base_caps = {
        m: torch.rand(11, dtype=torch.float64) * 0.005 + 0.001
        for m in ("generic", "rise", "fall")
    }

    star = StarPinPair(
        net2driver_pin_map=net2driver,
        timing_net_flat_arcs=arcs,
        timing_net_flat_arcs_start=arcs_start,
        num_pins=11,
        r_unit=0.8,
        c_unit=0.0003,
        scale_factor=1.0,
        dbu=100.0,
        load_geometry="wa_net",
    )

    def _pin_pos_from_mov(mov_xy: torch.Tensor) -> torch.Tensor:
        mx = mov_xy[:5]
        my = mov_xy[5:]
        nx = torch.cat((mx, fixed_xy[:1]))
        ny = torch.cat((my, fixed_xy[1:]))
        px = nx[pin2node] + pin_offsets_x
        py = ny[pin2node] + pin_offsets_y
        return torch.cat((px, py))

    for trial in range(6):
        mov_xy = (torch.randn(10, dtype=torch.float64) * 40.0).requires_grad_(True)
        gamma = 8.0 + 7.0 * trial
        pin_pos = _pin_pos_from_mov(mov_xy)
        len_um = star.length_um(pin_pos.detach())
        k_weight = torch.rand(8, dtype=torch.float64) * 2.0
        # Test both positive and signed load weights
        b_weight = (
            torch.randn(8, dtype=torch.float64) * 0.5
            if trial % 2 == 1
            else torch.rand(8, dtype=torch.float64) * 0.5
        )
        star.publish_snapshot(
            star.prepare_snapshot(
                length_um=len_um,
                raw_weight=k_weight,
                load_length_weight=b_weight,
                curvature_load_length_weight=b_weight,
            )
        )

        # 1. Pure load term exact Hessian vs net_wa_load_curvature_diagonal
        def _load_only_fn(mxy: torch.Tensor) -> torch.Tensor:
            pp = _pin_pos_from_mov(mxy)
            net_b = star._aggregate_pair_load_to_active_nets(b_weight)
            return torch.sum(star.net_wa_wirelength_um(pp, gamma) * net_b)

        h_load = torch.autograd.functional.hessian(_load_only_fn, mov_xy.detach())
        tmpl = torch.zeros(6, dtype=torch.float64)
        bd_x, bd_y = star.net_wa_load_curvature_diagonal(
            pin_pos.detach(),
            load=b_weight,
            smoothing_gamma=gamma,
            pin2node_map=pin2node,
            num_movable_nodes=5,
            num_nodes=6,
            output_template=tmpl,
        )
        # Single-cell Net 1 sits exclusively on cell 2 -> cell 2 must receive zero curvature!
        assert float(bd_x[2].item()) == 0.0 and float(bd_y[2].item()) == 0.0
        assert float(bd_x[5].item()) == 0.0 and float(bd_y[5].item()) == 0.0
        diag_load = torch.diag(torch.cat((bd_x[:5], bd_y[:5])))
        eig_minus = torch.linalg.eigvalsh(diag_load - h_load).min().item()
        eig_plus = torch.linalg.eigvalsh(diag_load + h_load).min().item()
        assert eig_minus >= -1e-10, f"PSD violation (D - H): {eig_minus}"
        assert eig_plus >= -1e-10, f"PSD violation (D + H): {eig_plus}"

        # 2. Combined Elmore WA + wa_net load exact Hessian vs timing_wa_curvature_diagonal
        def _full_wa_fn(mxy: torch.Tensor) -> torch.Tensor:
            pp = _pin_pos_from_mov(mxy)
            return star.structured_delay_objective(
                pp, base_caps=base_caps, distance_model="wa", smoothing_gamma=gamma
            )

        h_full = torch.autograd.functional.hessian(_full_wa_fn, mov_xy.detach())
        fdx, fdy = star.timing_wa_curvature_diagonal(
            pin_pos.detach(),
            base_caps=base_caps,
            smoothing_gamma=gamma,
            pin2node_map=pin2node,
            num_movable_nodes=5,
            num_nodes=6,
            output_template=tmpl,
        )
        diag_full = torch.diag(torch.cat((fdx[:5], fdy[:5])))
        assert torch.linalg.eigvalsh(diag_full - h_full).min().item() >= -1e-10
        assert torch.linalg.eigvalsh(diag_full + h_full).min().item() >= -1e-10


def test_macro_exclusion_taylor_kkt_and_tent_bell_modes():
    """Verify K5a taylor_kkt interior-equilibrium push + legal-row zero false repulsion, and K5b tent_bell ridge shift."""
    import numpy as np
    from dreamplace.ops.macro_exclusion_field import (
        MacroExclusionField,
        build_macro_potential_grid,
    )
    from dreamplace.ops.macro_exclusion_field.macro_exclusion_field import (
        _eval_cubic_bspline_2d,
    )

    rows = [(0.0, float(r * 10), 2000.0, float((r + 1) * 10)) for r in range(144)]
    sx = torch.tensor([10.0, 10.0, 10.0, 600.0], dtype=torch.float32)
    sy = torch.tensor([10.0, 10.0, 10.0, 600.0], dtype=torch.float32)
    # Macro [500, 1100] x [400, 1000]:
    # c0: deep inside macro (x_center=950, gx=0 interior equilibrium)
    # c1: legal Row 1 outside right wall (x_center=1105, d_in==0, gx=-3 away from macro)
    # c2: shallow inside macro (x_center=1090, strong natural outward pull gx=-50)
    pos = torch.tensor(
        [945.0, 1100.0, 1085.0, 500.0, 695.0, 695.0, 695.0, 400.0],
        dtype=torch.float32,
    )
    placedb = SimpleNamespace(
        num_nodes=4,
        num_movable_nodes=3,
        num_terminals=1,
        row_height=10.0,
        site_width=1.0,
        xl=0.0,
        yl=0.0,
        xh=2000.0,
        yh=1440.0,
        rows=rows,
        node_size_x=sx,
        node_size_y=sy,
        node_names=[b"c0_eq", b"c1_row1", b"c2_exit", b"u_macro"],
    )
    dc = SimpleNamespace(
        node_size_x=sx,
        node_size_y=sy,
        pin2node_map=torch.tensor([0, 1, 2, 3, 3, 3], dtype=torch.long),
        sum_pin_weights_in_nodes=torch.full((4,), 4.0, dtype=torch.float32),
    )
    field_k4a = MacroExclusionField(placedb, dc, pos, force_mode="normal_proj")
    field_k5a = MacroExclusionField(placedb, dc, pos)
    assert field_k5a.force_mode == "taylor_kkt"

    fake_tp = SimpleNamespace(
        timing_state_generation=1,
        pin_rslack_live_snapshot=torch.full((6,), -40.0, dtype=torch.float32),
        pin_fslack_live_snapshot=torch.full((6,), -40.0, dtype=torch.float32),
    )
    fake_star = SimpleNamespace(
        snapshot_generation=1,
        num_nets=3,
        pair_net_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        coordinate_to_um=1.0,
        r_unit=0.5,
        c_unit=0.00016,
        driver_pin_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        sink_pin_ids=torch.tensor([3, 4, 5], dtype=torch.long),
        frozen_weight=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
        frozen_load_length_weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        geometry=lambda _p: {
            "raw_dx": torch.tensor([450.0, -295.0, -410.0], dtype=torch.float32),
            "raw_dy": torch.zeros(3, dtype=torch.float32),
            "length_um": torch.tensor([450.0, 295.0, 410.0], dtype=torch.float32),
        },
    )
    owner = SimpleNamespace(
        _star_pin_pair_op=lambda: fake_star,
        _driver_sink_star_base_cap_map=lambda: {"generic": torch.full((6,), 0.004, dtype=torch.float32)},
        segment_timing_coefficient=1.0,
        mock_pull_grad_fn=lambda _p: (
            torch.tensor([0.0, -3.0, -50.0], dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        ),
        mock_hessian_and_gamma_fn=lambda _p: (
            torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32),
            torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32),
            20.0,
        ),
        data_collections=dc,
        op_collections=SimpleNamespace(
            pin_pos_op=lambda p: p,
            timing_propagation_op=fake_tp,
        ),
    )
    field_k4a.refresh_criticality_pulls(owner, pos)
    field_k5a.refresh_criticality_pulls(owner, pos)
    # K4a stalls at interior equilibrium (0.0), whereas K5a pushes strongly via H_n * d_in (capped cleanly at F_cap):
    assert float(field_k4a.movable_base_pull[0].item()) == 0.0
    assert float(field_k5a.movable_base_pull[0].item()) > 4.0
    # Legal Row 1 cell and self-exiting cell both receive identically 0.0 false repulsion:
    assert float(field_k5a.movable_base_pull[1].item()) == 0.0
    assert float(field_k5a.movable_base_pull[2].item()) == 0.0

    # Opt-in tent_bell reflects die-edge wall and shifts macro ridge toward cramped 6-row alley:
    boxes = np.array([[0.0, 200.0, 200.0, 600.0], [212.0, 200.0, 412.0, 600.0]], dtype=np.float64)
    g_rim, m_rim = build_macro_potential_grid(
        boxes, 0.0, 0.0, 1200.0, 800.0, 2.0, torch.device("cpu"), profile_mode="tent_rim"
    )
    g_bell, m_bell = build_macro_potential_grid(
        boxes, 0.0, 0.0, 1200.0, 800.0, 2.0, torch.device("cpu"), profile_mode="tent_bell"
    )
    _, gx_rim, _ = _eval_cubic_bspline_2d(
        g_rim, torch.tensor([30.0]), torch.tensor([400.0]), m_rim["g_xl"], m_rim["g_yl"], m_rim["h_g"]
    )
    _, gx_bell, _ = _eval_cubic_bspline_2d(
        g_bell, torch.tensor([30.0]), torch.tensor([400.0]), m_bell["g_xl"], m_bell["g_yl"], m_bell["h_g"]
    )
    assert float(gx_rim[0].item()) > 0.5
    assert float(gx_bell[0].item()) < -0.2
    xs_m2 = torch.linspace(220.0, 404.0, 185)
    v_m2, _, _ = _eval_cubic_bspline_2d(
        g_bell, xs_m2, torch.full_like(xs_m2, 400.0), m_bell["g_xl"], m_bell["g_yl"], m_bell["h_g"]
    )
    assert float(xs_m2[int(torch.argmax(v_m2))].item()) < 305.0


def test_star_tdp_runtime_metric_consistent_momentum_transport_default():
    placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        num_physical_nodes=2,
        num_filler_nodes=0,
        num_nodes=2,
        num_pins=2,
        regions=[],
        dbu=1.0,
        pin_direct=["OUTPUT", "INPUT"],
    )
    dc = SimpleNamespace(
        pin2net_map=torch.tensor([0, 0], dtype=torch.long),
        pin2node_map=torch.tensor([0, 1], dtype=torch.long),
        net2driver_pin_map=torch.tensor([0], dtype=torch.int32),
        net_flat_arcs=torch.tensor([[0, 1]], dtype=torch.long),
        net_flat_arcs_start=torch.tensor([0, 1], dtype=torch.long),
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
        inst_libcell_offset=torch.tensor([0, 0], dtype=torch.long),
    )

    class _CallableTiming:
        clock_pins = torch.empty(0, dtype=torch.long)
        num_pins = 2
        device = torch.device("cpu")
        dtype = torch.float32

        def __call__(self, delays, impulses, loads, **kwargs):
            d = delays["rise"][1] + delays["fall"][1]
            tns = -10.0 - d
            return tns, tns, tns, tns

    elmore_op = RCTiming(r_unit=0.2, c_unit=0.1, scale_factor=1.0, dbu=1.0)
    model = SimpleNamespace(
        gamma=torch.tensor(5.0),
        output_pin_mask=torch.tensor([True, False]),
        pin_caps_op=lambda _off, _dc: (
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        ),
        _relaxed_buffer_timing_enabled=lambda: False,
        _current_timing_surrogate_mode=lambda: "lut_only",
    )
    op_cols = SimpleNamespace(
        timing_propagation_op=_CallableTiming(),
        elmore_delay_op=elmore_op,
        pin_pos_op=lambda p: p,
        wirelength_op=lambda p: (p[1] - p[0]).abs() + (p[3] - p[2]).abs(),
    )
    rt_params = SimpleNamespace(
        timing_placement_carrier="star_pin_pair",
        _timing_placement_carrier_explicit=True,
        global_place_flag=1,
        placement_sizing_mode="place_only",
        macro_exclusion_field_flag=0,
        scale_factor=1.0,
        ignore_net_degree=100,
        deterministic_flag=True,
        star_rc_calibration_enabled=1,
        star_slope_calibration_enabled=0,
        star_lcb_ff_clock_enabled=0,
        star_kappa=0.05,
    )
    configure_timing_placement_carrier(rt_params)
    rt = StarTdpRuntime(
        model=model,
        params=rt_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    assert rt.momentum_transport_timing is True
    assert rt.momentum_transport_envelope == 0.0

    pos = torch.tensor([10.0, 90.0, 20.0, 40.0], dtype=torch.float32)
    # Step 499 (pre-activation): records (base, 0, alpha=2.0)
    rt.begin_nesterov_step(pos, 2.0)
    base_499 = torch.tensor([10.0, 10.0], dtype=torch.float32)
    rt.record_precondition_base_only(base_499)
    u = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    v = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    rt.end_nesterov_step(u, v)
    assert torch.allclose(u, torch.zeros(4))

    # Step 500 (activation): gate opens, D_base jumps 2x on node 1 (H=0), while node 0 gets H_x = 15 (alpha_next=0.5 -> curv=7.5)
    rt.set_gate_active(True)
    rt.refresh_snapshot(pos, iteration=500, overflow=0.30)
    rt.begin_nesterov_step(pos, 0.5)
    base_500 = torch.tensor([10.0, 20.0], dtype=torch.float32)
    curv_500 = torch.tensor([[7.5, 0.0], [0.0, 0.0]], dtype=torch.float32)
    precond_500 = (base_500.unsqueeze(0) + curv_500).clamp(min=1.0)
    rt.record_precondition_diagonal(precond_500, base_1n=base_500, curv_2n=curv_500)
    u = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    v = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    rt.end_nesterov_step(u, v)

    # Node 0 x: used = 10 + 0 = 10, hyp = 10 + (2.0 / 0.5) * 7.5 = 40 -> factor = sqrt(10 / 40) = 0.5 -> u[0] = 5.0
    assert math.isclose(float(u[0].item()), 5.0, rel_tol=1e-5)
    # Node 1 x & both y coordinates have H=0 -> factor == 1.0 identically despite D_base doubling!
    assert torch.allclose(u[1:], torch.zeros(3))

    # Legacy opt-in envelope override still works cleanly
    env_params = SimpleNamespace(**vars(rt_params))
    delattr(env_params, "star_momentum_transport_timing")
    env_params.star_momentum_transport_envelope = 1.05
    configure_timing_placement_carrier(env_params)
    rt_env = StarTdpRuntime(
        model=model,
        params=env_params,
        placedb=placedb,
        data_collections=dc,
        op_collections=op_cols,
    )
    assert rt_env.momentum_transport_envelope == 1.05
    assert rt_env.momentum_transport_timing is False






