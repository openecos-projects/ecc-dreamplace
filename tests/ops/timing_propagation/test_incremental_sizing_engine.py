"""Unit and parity tests for the C++ IncrementalSizingEngine."""

import numpy as np
import pytest
import torch  # noqa: F401 -- loads libtorch symbols before timing_propagation_cpp
from dreamplace.ops.timing_propagation import timing_propagation_cpp
from sparse_fixtures import add_transition_masks


def _lut_bundle(values, trans, caps, trans_dims, cap_dims):
    return [
        np.asarray(values, dtype=np.float64),
        np.asarray(trans, dtype=np.float64),
        np.asarray(caps, dtype=np.float64),
        np.asarray(trans_dims, dtype=np.int64),
        np.asarray(cap_dims, dtype=np.int64),
    ]


def _build_coupling_fixture(*, sparse_mode=True, aggregation_mode="hard"):
    """Build 1-driver -> 2-parallel-sink circuit with a driver load knee at 0.006 pF.

    Cells:
      0: DRV_X1  (2 pins: A=in, Y=out; 1 arc row 0)
         delay(load) = 10 + 500*load for load in [0, 0.006]
                     = 13 + 3000*(load - 0.006) for load in [0.006, 0.012]
      1: SINK_X1 (2 pins: A=in cap 0.002, Y=out; 1 arc row 1, delay = 20 ps, width = 1)
      2: SINK_UP1(2 pins: A=in cap 0.004, Y=out; 1 arc row 2, delay = 14 ps, width = 4)
      3: SINK_UP2(2 pins: A=in cap 0.004, Y=out; 1 arc row 3, delay = 15 ps, width = 2)
      4: SINK_LVT(2 pins: A=in cap 0.002, Y=out; 1 arc row 4, delay = 17 ps, width = 1)

    Pins:
      0: PI (net 0 driver)
      1: DRV.A (inst 0, pin offset 0, net 0 sink)
      2: DRV.Y (inst 0, pin offset 1, net 1 driver)
      3: S1.A  (inst 1, pin offset 0, net 1 sink)
      4: S1.Y  (inst 1, pin offset 1, net 2 driver, endpoint 0, req = 25 ps)
      5: S2.A  (inst 2, pin offset 0, net 1 sink)
      6: S2.Y  (inst 2, pin offset 1, net 3 driver, endpoint 1, req = 25 ps)
    """
    delay_lut = _lut_bundle(
        values=[
            [10.0, 13.0, 31.0],  # row 0: DRV_X1 with knee at 0.006 pF
            [20.0, 20.0, 20.0],  # row 1: SINK_X1
            [14.0, 14.0, 14.0],  # row 2: SINK_UP1 (-6 ps, +0.002 pF)
            [15.0, 15.0, 15.0],  # row 3: SINK_UP2 (-5 ps, +0.002 pF)
            [17.0, 17.0, 17.0],  # row 4: SINK_LVT (-3 ps, +0.000 pF, zero area delta)
        ],
        trans=[[0.0, 1.0, 2.0]] * 5,
        caps=[[0.0, 0.006, 0.012]] * 5,
        trans_dims=[1] * 5,
        cap_dims=[3] * 5,
    )
    slew_lut = _lut_bundle(
        values=[[0.2], [0.3], [0.25], [0.25], [0.25]],
        trans=[[0.0]] * 5,
        caps=[[0.0]] * 5,
        trans_dims=[1] * 5,
        cap_dims=[1] * 5,
    )
    check_lut = _lut_bundle(
        values=[[1.0]],
        trans=[[0.0]],
        caps=[[0.0]],
        trans_dims=[1],
        cap_dims=[1],
    )
    luts = [delay_lut, delay_lut, slew_lut, slew_lut, check_lut, check_lut]

    # 5 cells, 2 pins each (offset 0 = A, offset 1 = Y).
    lib_caps = np.array(
        [0.001, 0.0, 0.002, 0.0, 0.004, 0.0, 0.004, 0.0, 0.002, 0.0],
        dtype=np.float64,
    )
    lib_slew_lim = np.full(10, 10.0, dtype=np.float64)
    lib_cap_lim = np.full(10, 0.05, dtype=np.float64)

    inst_cell_id = np.array([0, 1, 1], dtype=np.int32)
    pin2node = np.array([-1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 1, 3], dtype=np.int32)
    pin_offset = np.array([-1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

    pin_cap_base = np.array([0.0, 0.001, 0.0, 0.002, 0.0, 0.002, 0.0], dtype=np.float64)
    pin_slew_lim = np.full(7, 10.0, dtype=np.float64)
    pin_cap_lim = np.full(7, 0.05, dtype=np.float64)

    cell_arcs = np.array(
        [
            [1, 2, 0, 0, 1, 0, 0, 0],
            [3, 4, 1, 1, 1, 0, 0, 0],
            [5, 6, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    flat_libarc_info = np.array(
        [[0, 1, c, 0, 1, 0] for c in range(5)],
        dtype=np.int64,
    )

    # Zero-wire-RC star forest for 4 nets over 7 vertices (pin == vertex).
    # Net 0: 0 -> 1; Net 1: 2 -> {3, 5}; Net 2: 4; Net 3: 6.
    rc_forest = {
        "wire_cap": np.zeros(7, dtype=np.float64),
        "incoming_resistance": np.zeros(7, dtype=np.float64),
        "parent": np.array([-1, 0, -1, 2, -1, 2, -1], dtype=np.int32),
        "child_start": np.array([0, 1, 1, 3, 3, 3, 3, 3], dtype=np.int32),
        "children": np.array([1, 3, 5], dtype=np.int32),
        "topo": np.array([0, 1, 2, 3, 5, 4, 6], dtype=np.int32),
        "topo_start": np.array([0, 2, 5, 6, 7], dtype=np.int32),
        "tree_by_net": np.array([0, 1, 2, 3], dtype=np.int32),
        "pin_to_vertex": np.arange(7, dtype=np.int32),
    }

    config = {
        "num_pins": 7,
        "num_nets": 4,
        "num_instances": 3,
        "num_cells": 5,
        "aggregation_mode": aggregation_mode,
        "aggregation_tau_ps": 2.0,
        "pairwise_smooth": False,
        "pairwise_alpha": 20.0,
        "slew_epsilon": 1e-8,
        "pin2node_map": pin2node,
        "pin2net_map": pin2net,
        "pin_2_libpin_offset": pin_offset,
        "inst_cell_id": inst_cell_id,
        "cell_id_2_arc_id_start": np.arange(6, dtype=np.int64),
        "cell_id_2_libpin_id_start": np.arange(0, 12, 2, dtype=np.int64),
        "flat_libarc_info": flat_libarc_info,
        "flat_lib_pin_cap": lib_caps,
        "flat_lib_pin_rcap": lib_caps,
        "flat_lib_pin_fcap": lib_caps,
        "flat_lib_pin_slew_limit": lib_slew_lim,
        "flat_lib_pin_cap_limit": lib_cap_lim,
        "pin_cap_base": pin_cap_base,
        "pin_rcap_base": pin_cap_base,
        "pin_fcap_base": pin_cap_base,
        "pin_slew_limit": pin_slew_lim,
        "pin_cap_limit": pin_cap_lim,
        "luts": luts,
        "cell_arcs": cell_arcs,
        "cell_level_start": np.array([0, 0, 1, 3], dtype=np.int64),
        "setup_arcs": np.zeros((0, 8), dtype=np.int64),
        "check_arcs": np.zeros((0, 8), dtype=np.int64),
        "rc_forest": rc_forest,
        "sparse_mode": sparse_mode,
    }

    if sparse_mode:
        arrays = {
            "state_pin": np.arange(7, dtype=np.int32),
            "state_slew_group": np.arange(7, dtype=np.int32),
            "state_level": np.array([0, 1, 2, 3, 4, 3, 4], dtype=np.int32),
            "state_valid": np.ones((7, 2), dtype=bool),
            "state_seed": np.array(
                [[True, True]] + [[False, False]] * 6, dtype=bool
            ),
            "state_reference": np.array(
                [[0.0, 0.0, 0.1, 0.1]] + [[-2e8, -2e8, 0.0, 0.0]] * 6,
                dtype=np.float64,
            ),
            "net_edges": np.array([[0, 0, 1], [1, 2, 3], [1, 2, 5]], dtype=np.int64),
            "net_threshold_adjustments": np.tile(
                np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64), (3, 1)
            ),
            "cell_edges": np.array([[0, 1, 2], [1, 3, 4], [2, 5, 6]], dtype=np.int64),
            "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
            "check_index": np.array(
                [
                    [4, 0, -1, 0, -1, 0, 1],
                    [6, 0, -1, 0, -1, 0, 1],
                ],
                dtype=np.int64,
            ),
            "check_values": np.array(
                [[25.0, 0.0, 0.0, 0.0], [25.0, 0.0, 0.0, 0.0]], dtype=np.float64
            ),
        }
        add_transition_masks(arrays, cell_arcs)
        config["sparse"] = {
            **arrays,
            "legal_checks": np.array([1, 1], dtype=np.uint8),
        }
    else:
        config.update(
            {
                "start_pins": np.array([0], dtype=np.int64),
                "start_rAAT": np.array([0.0], dtype=np.float64),
                "start_fAAT": np.array([0.0], dtype=np.float64),
                "start_rtran": np.array([0.1], dtype=np.float64),
                "start_ftran": np.array([0.1], dtype=np.float64),
                "clock_pins": np.zeros(0, dtype=np.int64),
                "clock_rtran": np.zeros(0, dtype=np.float64),
                "clock_ftran": np.zeros(0, dtype=np.float64),
                "net_arcs": np.array([[0, 1], [2, 3], [2, 5]], dtype=np.int64),
                "endpoint_pins": np.array([4, 6], dtype=np.int64),
                "endpoint_rRAT": np.array([25.0, 25.0], dtype=np.float64),
                "endpoint_fRAT": np.array([25.0, 25.0], dtype=np.float64),
                "endpoint_valid": np.tile(
                    np.array([[True, False]], dtype=bool), (7, 1)
                ),
                "setup_valid": None,
                "check_valid": None,
                "metric_scope": "setup_plus_recovery",
                "recovery_uses_live_slew": True,
                "recovery_rat_is_capture_deadline": False,
                "backend_rSlew": None,
                "backend_fSlew": None,
            }
        )

    policy = {
        "wns_coeff": 0.01,
        "tns_coeff": 1.0,
        "timing_weight": 1.0,
        "slew_weight": 1.0,
        "cap_weight": 1.0,
        "leakage_weight": 0.0,
        "loss_tolerance": 0.0,
        "slew_pin_mask": np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.uint8),
        "cap_pin_mask": np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
        "cell_width": np.array([1.0, 1.0, 4.0, 2.0, 1.0], dtype=np.float64),
        "cell_height": np.ones(5, dtype=np.float64),
        "cell_leakage": np.zeros(5, dtype=np.float64),
        "padding_x": 0.0,
        "num_movable_nodes": 3,
        "virtual_area": 0.0,
        "capacity": 20.0,
        "inflated_mask": np.zeros(3, dtype=np.uint8),
        "inflated_width": np.zeros(3, dtype=np.float64),
        "inflated_height": np.zeros(3, dtype=np.float64),
    }
    return config, policy


@pytest.mark.parametrize("sparse_mode", [True, False])
def test_sequential_coupling_accepts_top_candidate_and_rejects_knee_overshoot(sparse_mode):
    """Both S1->UP1 and S2->UP2 improve TNS individually, but together cross the driver knee.

    Base (cells [0, 1, 1], load = 0.004):
      delay(DRV) = 12, arr(S1) = 32 (slack -7), arr(S2) = 32 (slack -7)
      WNS = -7, TNS = -14, loss = 14.07
    After #1 (inst 1 -> cell 2, load = 0.006):
      delay(DRV) = 13, arr(S1) = 27 (slack -2), arr(S2) = 33 (slack -8)
      WNS = -8, TNS = -10, loss = 10.08 -> ACCEPTED
    Candidate #2 on updated state (inst 2 -> cell 3, load = 0.008 > 0.006 knee):
      delay(DRV) = 19, arr(S1) = 33 (slack -8), arr(S2) = 34 (slack -9)
      WNS = -9, TNS = -17, loss = 17.09 > 10.08 -> REJECTED and rolled back!
    """
    config, policy = _build_coupling_fixture(sparse_mode=sparse_mode)
    engine = timing_propagation_cpp.IncrementalSizingEngine(config)

    assert engine.current_metrics() == {
        "wns_ps": pytest.approx(-7.0),
        "tns_ps": pytest.approx(-14.0),
        "endpoint_count": 2,
    }

    report = dict(engine.evaluate_ranked_candidates([1, 2], [2, 3], policy))
    assert report["status"] == "ok"
    assert report["accepted_instance_ids"] == [1]
    assert report["accepted_cell_ids"] == [2]
    assert report["evaluated_count"] == 2
    assert report["accepted_count"] == 1
    assert report["rejected_loss_count"] == 1
    assert report["rejected_area_count"] == 0
    assert report["initial_loss"] == pytest.approx(14.07)
    assert report["final_loss"] == pytest.approx(10.08)
    assert report["final_wns_ps"] == pytest.approx(-8.0)
    assert report["final_tns_ps"] == pytest.approx(-10.0)

    # Verify rollback left zero residual state drift by building a fresh engine on [0, 2, 1].
    fresh_cfg, _ = _build_coupling_fixture(sparse_mode=sparse_mode)
    fresh_cfg["inst_cell_id"] = np.array([0, 2, 1], dtype=np.int32)
    fresh_cfg["cell_arcs"][1, 2] = 2
    fresh_cfg["cell_arcs"][1, 3] = 2
    fresh_cfg["pin_cap_base"][3] = 0.004
    fresh_cfg["pin_rcap_base"][3] = 0.004
    fresh_cfg["pin_fcap_base"][3] = 0.004
    fresh_engine = timing_propagation_cpp.IncrementalSizingEngine(fresh_cfg)
    assert engine.current_metrics() == fresh_engine.current_metrics()


def test_capacity_gate_rejects_oversize_and_accepts_subsequent_lvt_swap():
    """Capacity = 4.0 blocks inst 1 -> UP1 while accepting inst 2 -> LVT."""
    config, policy = _build_coupling_fixture(sparse_mode=True)
    policy["capacity"] = 4.0
    # Inflate inst 0 to width 2 so base movable area is 2 + 1 + 1 = 4.0 == capacity.
    policy["inflated_mask"] = np.array([1, 0, 0], dtype=np.uint8)
    policy["inflated_width"] = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    policy["inflated_height"] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    engine = timing_propagation_cpp.IncrementalSizingEngine(config)
    report = dict(engine.evaluate_ranked_candidates([1, 2], [2, 4], policy))

    assert report["accepted_instance_ids"] == [2]
    assert report["accepted_cell_ids"] == [4]
    assert report["rejected_area_count"] == 1
    assert report["evaluated_count"] == 1
    assert report["accepted_count"] == 1
    assert report["final_area"] == pytest.approx(4.0)
    # LVT swap has zero cap delta so delay(DRV) stays 12; arr(S2) drops 32 -> 29 (slack -7 -> -4).
    assert report["final_wns_ps"] == pytest.approx(-7.0)
    assert report["final_tns_ps"] == pytest.approx(-11.0)
    assert report["final_loss"] == pytest.approx(11.07)


def test_sync_state_matches_fresh_engine_under_smooth_aggregation_and_drv():
    """Verify sync_state + DRV slew/cap units (cap violation pF * 1000 -> fF) match fresh build."""
    config, policy = _build_coupling_fixture(
        sparse_mode=True, aggregation_mode="smooth"
    )
    # Tight cap limit 0.005 pF on DRV.Y (pin 2) triggers 0.001 pF = 1.0 fF violation.
    config["pin_cap_limit"][2] = 0.005
    config["flat_lib_pin_cap_limit"][1] = 0.005
    policy["cap_weight"] = 2.0

    engine = timing_propagation_cpp.IncrementalSizingEngine(config)
    engine.sync_state(np.array([0, 2, 1], dtype=np.int64))

    fresh_cfg, _ = _build_coupling_fixture(
        sparse_mode=True, aggregation_mode="smooth"
    )
    fresh_cfg["inst_cell_id"] = np.array([0, 2, 1], dtype=np.int32)
    fresh_cfg["cell_arcs"][1, 2] = 2
    fresh_cfg["cell_arcs"][1, 3] = 2
    fresh_cfg["pin_cap_base"][3] = 0.004
    fresh_cfg["pin_rcap_base"][3] = 0.004
    fresh_cfg["pin_fcap_base"][3] = 0.004
    fresh_cfg["pin_cap_limit"][2] = 0.005
    fresh_cfg["flat_lib_pin_cap_limit"][1] = 0.005
    fresh_engine = timing_propagation_cpp.IncrementalSizingEngine(fresh_cfg)

    assert engine.current_metrics() == pytest.approx(fresh_engine.current_metrics())

    # Evaluating a no-op candidate list reports exact initial/final cap_violation_pf and loss.
    rep_synced = dict(engine.evaluate_ranked_candidates([], [], policy))
    rep_fresh = dict(fresh_engine.evaluate_ranked_candidates([], [], policy))
    assert rep_synced["initial_cap_violation_pf"] == pytest.approx(0.001)
    assert rep_synced["initial_loss"] == pytest.approx(rep_fresh["initial_loss"])
    expected_loss = (
        -(0.01 * rep_synced["initial_wns_ps"] + 1.0 * rep_synced["initial_tns_ps"])
        + 0.001 * 1000.0 * 2.0
    )
    assert rep_synced["initial_loss"] == pytest.approx(expected_loss)
