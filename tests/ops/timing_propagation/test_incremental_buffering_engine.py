"""Unit and ground-truth parity tests for C++ IncrementalSizingEngine buffer insertion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.flows.discrete.buffering_filter import IncrementalBufferingGate
from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
    TypedDiscreteCandidateTransition,
)
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


def _make_library():
    """Return 4-cell library: DRV_X1 (0), SINK_X1 (1), BUF_FAST (2), BUF_SLOW_WIDE (3).

    Delay(slew, load) is linear in load and input slew so analytical & fresh-engine
    checks are exact across arbitrary wire RC splits:
      0 DRV_X1:        delay = 10 + 2.0*slew + 2000*load, out_slew = 0.2 + 500*load, c=0.002, w=2
      1 SINK_X1:       delay = 15 + 1.0*slew +    0*load, out_slew = 0.2,            c=0.008, w=2
      2 BUF_FAST:      delay =  4 + 0.5*slew +  400*load, out_slew = 0.1 + 100*load, c=0.002, w=1
      3 BUF_SLOW_WIDE: delay = 40 + 1.0*slew +  800*load, out_slew = 0.4 + 200*load, c=0.006, w=6
    """
    trans_grid = [0.0, 10.0]
    cap_grid = [0.0, 0.1]

    def linear_table(base, k_slew, k_cap):
        return [
            base + k_slew * s + k_cap * c
            for s in trans_grid
            for c in cap_grid
        ]

    delay_lut = _lut_bundle(
        values=[
            linear_table(10.0, 2.0, 2000.0),
            linear_table(15.0, 1.0, 0.0),
            linear_table(4.0, 0.5, 400.0),
            linear_table(40.0, 1.0, 800.0),
        ],
        trans=[trans_grid] * 4,
        caps=[cap_grid] * 4,
        trans_dims=[2] * 4,
        cap_dims=[2] * 4,
    )
    slew_lut = _lut_bundle(
        values=[
            linear_table(0.2, 0.0, 500.0),
            linear_table(0.2, 0.0, 0.0),
            linear_table(0.1, 0.0, 100.0),
            linear_table(0.4, 0.0, 200.0),
        ],
        trans=[trans_grid] * 4,
        caps=[cap_grid] * 4,
        trans_dims=[2] * 4,
        cap_dims=[2] * 4,
    )
    check_lut = _lut_bundle([[0.0]], [[0.0]], [[0.0]], [1], [1])
    luts = [delay_lut, delay_lut, slew_lut, slew_lut, check_lut, check_lut]

    lib_caps = np.array(
        [0.002, 0.0, 0.008, 0.0, 0.002, 0.0, 0.006, 0.0],
        dtype=np.float64,
    )
    return {
        "luts": luts,
        "lib_caps": lib_caps,
        "lib_slew_lim": np.full(8, 5.0, dtype=np.float64),
        "lib_cap_lim": np.array(
            [1.0, 0.015, 1.0, 0.05, 1.0, 0.05, 1.0, 0.05], dtype=np.float64
        ),
        "cell_width": np.array([2.0, 2.0, 1.0, 6.0], dtype=np.float64),
        "flat_libarc_info": np.array(
            [[0, 1, c, 0, 1, 0] for c in range(4)], dtype=np.int64
        ),
        "lib_arc_pin_offsets": np.array([[0, 1]] * 4, dtype=np.int32),
    }


def _pack_rc_forest(num_pins, total_vertices, net_trees):
    """Build exact lumped-pi rc_forest arrays from per-net directed edge lists.

    Each entry in ``net_trees`` is ``(root_v, [(parent_v, child_v, r_kohm, c_pf), ...])``.
    """
    parent = np.full(total_vertices, -1, dtype=np.int32)
    inc_r = np.zeros(total_vertices, dtype=np.float64)
    inc_c = np.zeros(total_vertices, dtype=np.float64)
    wire_cap = np.zeros(total_vertices, dtype=np.float64)
    adj = [[] for _ in range(total_vertices)]
    topo = []
    topo_start = [0]

    for root_v, edges in net_trees:
        for u, v, r_val, c_val in edges:
            parent[v] = u
            inc_r[v] = r_val
            inc_c[v] = c_val
            wire_cap[u] += 0.5 * c_val
            wire_cap[v] += 0.5 * c_val
            adj[u].append(v)
        # BFS topological order from root_v
        queue = [root_v]
        idx = 0
        while idx < len(queue):
            cur = queue[idx]
            idx += 1
            for ch in adj[cur]:
                queue.append(ch)
        topo.extend(queue)
        topo_start.append(len(topo))

    child_start = np.zeros(total_vertices + 1, dtype=np.int32)
    children = []
    for v in range(total_vertices):
        child_start[v] = len(children)
        children.extend(adj[v])
    child_start[total_vertices] = len(children)

    return {
        "wire_cap": wire_cap,
        "incoming_wire_cap": inc_c,
        "incoming_resistance": inc_r,
        "parent": parent,
        "child_start": child_start,
        "children": np.asarray(children, dtype=np.int32),
        "topo": np.asarray(topo, dtype=np.int32),
        "topo_start": np.asarray(topo_start, dtype=np.int32),
        "tree_by_net": np.arange(len(net_trees), dtype=np.int32),
        "pin_to_vertex": np.arange(num_pins, dtype=np.int32),
    }


def _build_chain_netlist(
    *,
    inst_cells,
    net1_chain_cells=(),
    net1_split_ratios=(),
    total_r=2.0,
    total_c=0.020,
    sparse_mode=True,
):
    """Build PI -> DRV -> (0..K buffers along net 1) -> SINK -> endpoint.

    In sparse_mode=True, each pin carries TWO clock-domain states (fast clk0 at 0 ps
    and slow clk1 at +5 ps) to exercise multi-state sparse edge duplication and topological
    level maintenance across inserted buffers.
    """
    lib = _make_library()
    k_buf = len(net1_chain_cells)
    assert len(net1_split_ratios) == k_buf

    # Base instances: 0 = DRV, 1 = SINK; followed by k_buf buffers (2 .. 1+k_buf).
    all_cells = list(inst_cells) + list(net1_chain_cells)
    num_inst = len(all_cells)
    # Base pins:
    # 0: PI (net 0 driver)
    # 1: DRV.A (net 0 sink)
    # 2: DRV.Y (net 1 driver)
    # 3: SINK.A (last segment sink)
    # 4: SINK.Y (net 2 driver, endpoint)
    # Each buffer j adds pins (5 + 2*j = BUF_j.A, 6 + 2*j = BUF_j.Y) and net (3 + j).
    num_pins = 5 + 2 * k_buf
    num_nets = 3 + k_buf

    pin2node = [-1, 0, 0, 1, 1]
    pin_offset = [-1, 0, 1, 0, 1]
    for j in range(k_buf):
        pin2node.extend([2 + j, 2 + j])
        pin_offset.extend([0, 1])

    # Ordered segments along original edge (2 -> 3) at sorted positions ratios_sorted:
    pts = [0.0] + list(net1_split_ratios) + [1.0]
    seg_fracs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]

    # Net connectivity:
    # Net 0: 0 -> 1
    # Net 2: 4 (endpoint)
    # Chain along original net 1:
    #   if k_buf == 0: net 1 is 2 -> 3
    #   else: net 1 is 2 -> BUF_0.A (pin 5);
    #         for j in 0..k_buf-2: net (3+j) is BUF_j.Y (6+2*j) -> BUF_{j+1}.A (7+2*j);
    #         net (3+k_buf-1) is BUF_{k_buf-1}.Y (6+2*(k_buf-1)) -> 3.
    pin2net = [0, 0, 1, 1 if k_buf == 0 else (2 + k_buf), 2]
    for j in range(k_buf):
        in_net = 1 if j == 0 else (2 + j)
        out_net = 3 + j
        pin2net.extend([in_net, out_net])

    net_driver_pin = [0, 2, 4] + [6 + 2 * j for j in range(k_buf)]
    net_trees = [(0, [(0, 1, 0.0, 0.0)])]
    if k_buf == 0:
        net_trees.append((2, [(2, 3, total_r, total_c)]))
    else:
        net_trees.append((2, [(2, 5, total_r * seg_fracs[0], total_c * seg_fracs[0])]))
    net_trees.append((4, []))
    for j in range(k_buf):
        drv_p = 6 + 2 * j
        snk_p = 3 if j == k_buf - 1 else (7 + 2 * j)
        frac = seg_fracs[j + 1]
        net_trees.append((drv_p, [(drv_p, snk_p, total_r * frac, total_c * frac)]))

    rc_forest = _pack_rc_forest(num_pins, num_pins, net_trees)

    pin_cap_base = np.zeros(num_pins, dtype=np.float64)
    pin_slew_lim = np.full(num_pins, 5.0, dtype=np.float64)
    pin_cap_lim = np.full(num_pins, 1.0, dtype=np.float64)
    for p in range(num_pins):
        nid = pin2node[p]
        if nid >= 0:
            cid = all_cells[nid]
            lp = cid * 2 + pin_offset[p]
            pin_cap_base[p] = lib["lib_caps"][lp]
            pin_slew_lim[p] = lib["lib_slew_lim"][lp]
            pin_cap_lim[p] = lib["lib_cap_lim"][lp]

    # Topological cell arcs: DRV (level 1), BUF_0..BUF_{k-1} (levels 2..k+1), SINK (level k+2)
    cell_arc_rows = [[1, 2, all_cells[0], all_cells[0], 1, 0, 0, 0]]
    for j in range(k_buf):
        c_id = all_cells[2 + j]
        cell_arc_rows.append([5 + 2 * j, 6 + 2 * j, c_id, c_id, 1, 0, 0, 0])
    cell_arc_rows.append([3, 4, all_cells[1], all_cells[1], 1, 0, 0, 0])
    cell_arcs = np.asarray(cell_arc_rows, dtype=np.int64)
    cell_level_start = np.array([0, 0] + list(range(1, num_inst + 1)), dtype=np.int64)

    net_pairs = [(0, 1)]
    if k_buf == 0:
        net_pairs.append((2, 3))
    else:
        net_pairs.append((2, 5))
        for j in range(k_buf):
            net_pairs.append((6 + 2 * j, 3 if j == k_buf - 1 else (7 + 2 * j)))

    config = {
        "num_pins": num_pins,
        "num_nets": num_nets,
        "num_instances": num_inst,
        "num_cells": 4,
        "aggregation_mode": "hard",
        "aggregation_tau_ps": 2.0,
        "pairwise_smooth": False,
        "pairwise_alpha": 20.0,
        "slew_epsilon": 1e-8,
        "pin2node_map": np.asarray(pin2node, dtype=np.int32),
        "pin2net_map": np.asarray(pin2net, dtype=np.int32),
        "pin_2_libpin_offset": np.asarray(pin_offset, dtype=np.int32),
        "net_driver_pin": np.asarray(net_driver_pin, dtype=np.int32),
        "lib_arc_pin_offsets": lib["lib_arc_pin_offsets"],
        "inst_cell_id": np.asarray(all_cells, dtype=np.int32),
        "cell_id_2_arc_id_start": np.arange(5, dtype=np.int64),
        "cell_id_2_libpin_id_start": np.arange(0, 10, 2, dtype=np.int64),
        "flat_libarc_info": lib["flat_libarc_info"],
        "flat_lib_pin_cap": lib["lib_caps"],
        "flat_lib_pin_rcap": lib["lib_caps"],
        "flat_lib_pin_fcap": lib["lib_caps"],
        "flat_lib_pin_slew_limit": lib["lib_slew_lim"],
        "flat_lib_pin_cap_limit": lib["lib_cap_lim"],
        "pin_cap_base": pin_cap_base,
        "pin_rcap_base": pin_cap_base,
        "pin_fcap_base": pin_cap_base,
        "pin_slew_limit": pin_slew_lim,
        "pin_cap_limit": pin_cap_lim,
        "luts": lib["luts"],
        "cell_arcs": cell_arcs,
        "cell_level_start": cell_level_start,
        "setup_arcs": np.zeros((0, 8), dtype=np.int64),
        "check_arcs": np.zeros((0, 8), dtype=np.int64),
        "rc_forest": rc_forest,
        "sparse_mode": sparse_mode,
    }

    if sparse_mode:
        # 2 states per pin: state 2*p = clk0 (seed 0 ps), state 2*p+1 = clk1 (seed 5 ps).
        pin_levels = {0: 0, 1: 1, 2: 2}
        for j in range(k_buf):
            pin_levels[5 + 2 * j] = 3 + 2 * j
            pin_levels[6 + 2 * j] = 4 + 2 * j
        pin_levels[3] = 3 + 2 * k_buf
        pin_levels[4] = 4 + 2 * k_buf

        num_states = 2 * num_pins
        state_pin = np.repeat(np.arange(num_pins, dtype=np.int32), 2)
        state_level = np.array(
            [pin_levels[p] for p in state_pin], dtype=np.int32
        )
        state_seed = np.zeros((num_states, 2), dtype=bool)
        state_seed[0] = [True, True]
        state_seed[1] = [True, True]
        state_ref = np.tile(
            np.array([[-2e8, -2e8, 0.0, 0.0]], dtype=np.float64), (num_states, 1)
        )
        state_ref[0] = [0.0, 0.0, 0.1, 0.1]
        state_ref[1] = [5.0, 5.0, 0.1, 0.1]

        net_edges = []
        for u, v in net_pairs:
            net_id = pin2net[u]
            net_edges.append([net_id, 2 * u, 2 * v])
            net_edges.append([net_id, 2 * u + 1, 2 * v + 1])
        cell_edges = []
        for arc_idx, row in enumerate(cell_arc_rows):
            u, v = int(row[0]), int(row[1])
            cell_edges.append([arc_idx, 2 * u, 2 * v])
            cell_edges.append([arc_idx, 2 * u + 1, 2 * v + 1])

        arrays = {
            "state_pin": state_pin,
            "state_slew_group": state_pin.copy(),
            "state_level": state_level,
            "state_source_type": np.ones(num_states, dtype=np.int32),
            "state_role": np.zeros(num_states, dtype=np.int32),
            "state_valid": np.ones((num_states, 2), dtype=bool),
            "state_seed": state_seed,
            "state_reference": state_ref,
            "net_edges": np.asarray(net_edges, dtype=np.int64),
            "net_threshold_adjustments": np.tile(
                np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
                (len(net_edges), 1),
            ),
            "cell_edges": np.asarray(cell_edges, dtype=np.int64),
            "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
            "check_index": np.array(
                [
                    [2 * 4, 0, -1, 0, -1, 0, 1],
                    [2 * 4 + 1, 0, -1, 0, -1, 0, 1],
                ],
                dtype=np.int64,
            ),
            "check_values": np.array(
                [[45.0, 0.0, 0.0, 0.0], [52.0, 0.0, 0.0, 0.0]], dtype=np.float64
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
                "net_arcs": np.asarray(net_pairs, dtype=np.int64),
                "endpoint_pins": np.array([4], dtype=np.int64),
                "endpoint_rRAT": np.array([45.0], dtype=np.float64),
                "endpoint_fRAT": np.array([45.0], dtype=np.float64),
                "endpoint_valid": np.tile(
                    np.array([[True, False]], dtype=bool), (num_pins, 1)
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

    slew_mask = np.ones(num_pins, dtype=np.uint8)
    slew_mask[0] = 0
    cap_mask = np.zeros(num_pins, dtype=np.uint8)
    cap_mask[2] = 1
    cap_mask[4] = 1
    for j in range(k_buf):
        cap_mask[6 + 2 * j] = 1

    policy = {
        "wns_coeff": 0.5,
        "tns_coeff": 1.0,
        "timing_weight": 1.0,
        "slew_weight": 2.0,
        "cap_weight": 1.0,
        "leakage_weight": 0.0,
        "loss_tolerance": 0.0,
        "slew_pin_mask": slew_mask,
        "cap_pin_mask": cap_mask,
        "cell_width": lib["cell_width"],
        "cell_height": np.ones(4, dtype=np.float64),
        "cell_leakage": np.zeros(4, dtype=np.float64),
        "padding_x": 0.0,
        "num_movable_nodes": num_inst,
        "virtual_area": 0.0,
        "capacity": 20.0,
        "inflated_mask": np.zeros(num_inst, dtype=np.uint8),
        "inflated_width": np.zeros(num_inst, dtype=np.float64),
        "inflated_height": np.zeros(num_inst, dtype=np.float64),
    }
    return config, policy


@pytest.mark.parametrize("sparse_mode", [True, False])
def test_single_edge_buffer_matches_fresh_rebuild_exactly(sparse_mode):
    """Midpoint BUF_FAST splits wire RC + shields DRV cap violation, matching fresh rebuild."""
    base_cfg, base_policy = _build_chain_netlist(
        inst_cells=[0, 1], sparse_mode=sparse_mode
    )
    engine = timing_propagation_cpp.IncrementalSizingEngine(base_cfg)

    spec = {
        "net_id": 1,
        "parent_vertex": 2,
        "child_vertices": [3],
        "split_ratio": 0.5,
        "target_cell_id": 2,
    }
    report = dict(engine.evaluate_ranked_buffer_candidates([spec], base_policy))
    assert report["status"] == "ok"
    assert report["accepted_indices"] == [0]
    assert report["accepted_count"] == 1
    assert report["final_loss"] < report["initial_loss"]
    # Before buffering, DRV load = 0.020 + 0.008 = 0.028 > 0.015 limit (0.013 pF violation).
    # After midpoint BUF_FAST, DRV load = 0.010 + 0.002 = 0.012 <= 0.015, BUF load = 0.018 <= 0.05.
    assert report["initial_cap_violation_pf"] == pytest.approx(0.013)
    assert report["final_cap_violation_pf"] == pytest.approx(0.0)
    assert report["final_area"] == pytest.approx(5.0)

    ref_cfg, ref_policy = _build_chain_netlist(
        inst_cells=[0, 1],
        net1_chain_cells=[2],
        net1_split_ratios=[0.5],
        sparse_mode=sparse_mode,
    )
    ref_engine = timing_propagation_cpp.IncrementalSizingEngine(ref_cfg)
    ref_noop = dict(ref_engine.evaluate_ranked_buffer_candidates([], ref_policy))

    assert engine.current_metrics() == pytest.approx(ref_engine.current_metrics())
    for field in (
        "final_wns_ps",
        "final_tns_ps",
        "final_slew_violation_ps",
        "final_cap_violation_pf",
        "final_loss",
        "final_area",
    ):
        assert report[field] == pytest.approx(ref_noop[field.replace("final_", "initial_")])


@pytest.mark.parametrize("sparse_mode", [True, False])
def test_chained_same_edge_buffers_are_order_invariant_and_match_fresh_rebuild(sparse_mode):
    """Inserting buffers at r=0.25 and r=0.75 in either order matches fresh 2-buffer rebuild."""
    # Use high wire RC + cap penalty so both buffers strictly reduce loss in either order.
    cfg_fwd, pol_fwd = _build_chain_netlist(
        inst_cells=[0, 1], total_r=150.0, total_c=0.060, sparse_mode=sparse_mode
    )
    cfg_rev, pol_rev = _build_chain_netlist(
        inst_cells=[0, 1], total_r=150.0, total_c=0.060, sparse_mode=sparse_mode
    )
    pol_fwd["cap_weight"] = 10.0
    pol_rev["cap_weight"] = 10.0
    eng_fwd = timing_propagation_cpp.IncrementalSizingEngine(cfg_fwd)
    eng_rev = timing_propagation_cpp.IncrementalSizingEngine(cfg_rev)

    spec_near = {
        "net_id": 1,
        "parent_vertex": 2,
        "child_vertices": [3],
        "split_ratio": 0.25,
        "target_cell_id": 2,
    }
    spec_far = {
        "net_id": 1,
        "parent_vertex": 2,
        "child_vertices": [3],
        "split_ratio": 0.75,
        "target_cell_id": 2,
    }

    rep_fwd = dict(
        eng_fwd.evaluate_ranked_buffer_candidates([spec_near, spec_far], pol_fwd)
    )
    rep_rev = dict(
        eng_rev.evaluate_ranked_buffer_candidates([spec_far, spec_near], pol_rev)
    )
    assert rep_fwd["accepted_indices"] == [0, 1]
    assert rep_rev["accepted_indices"] == [0, 1]

    ref_cfg, ref_pol = _build_chain_netlist(
        inst_cells=[0, 1],
        net1_chain_cells=[2, 2],
        net1_split_ratios=[0.25, 0.75],
        total_r=150.0,
        total_c=0.060,
        sparse_mode=sparse_mode,
    )
    ref_pol["cap_weight"] = 10.0
    ref_eng = timing_propagation_cpp.IncrementalSizingEngine(ref_cfg)
    ref_noop = dict(ref_eng.evaluate_ranked_buffer_candidates([], ref_pol))

    assert eng_fwd.current_metrics() == pytest.approx(ref_eng.current_metrics())
    assert eng_rev.current_metrics() == pytest.approx(ref_eng.current_metrics())
    for field in (
        "final_wns_ps",
        "final_tns_ps",
        "final_slew_violation_ps",
        "final_cap_violation_pf",
        "final_loss",
        "final_area",
    ):
        expected = ref_noop[field.replace("final_", "initial_")]
        assert rep_fwd[field] == pytest.approx(expected)
        assert rep_rev[field] == pytest.approx(expected)


def test_tree_node_side_branch_shielding_and_rejection_rollback():
    """Tree-node buffer (split_ratio < 0) shields critical branch; rejected ones roll back."""
    lib = _make_library()
    # Net 1 has Steiner vertex 7:
    #   DRV.Y (pin 2) --(r=1.0, c=0.004)--> Steiner 7
    #   Steiner 7 --(r=0.5, c=0.002)--> SINK_CRIT.A (pin 3, req=35 ps at pin 4)
    #   Steiner 7 --(r=0.5, c=0.020)--> SINK_SIDE.A (pin 5, slack req=200 ps at pin 6)
    num_pins = 7
    total_vertices = 8  # vertex 7 is Steiner node on net 1
    rc_forest = _pack_rc_forest(
        num_pins,
        total_vertices,
        [
            (0, [(0, 1, 0.0, 0.0)]),
            (
                2,
                [
                    (2, 7, 1.0, 0.004),
                    (7, 3, 0.5, 0.002),
                    (7, 5, 0.5, 0.020),
                ],
            ),
            (4, []),
            (6, []),
        ],
    )
    inst_cells = np.array([0, 1, 1], dtype=np.int32)
    pin2node = np.array([-1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 1, 3], dtype=np.int32)
    pin_offset = np.array([-1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    pin_cap_base = np.array(
        [0.0, 0.002, 0.0, 0.008, 0.0, 0.008, 0.0], dtype=np.float64
    )
    cell_arcs = np.array(
        [
            [1, 2, 0, 0, 1, 0, 0, 0],
            [3, 4, 1, 1, 1, 0, 0, 0],
            [5, 6, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    arrays = {
        "state_pin": np.arange(num_pins, dtype=np.int32),
        "state_slew_group": np.arange(num_pins, dtype=np.int32),
        "state_level": np.array([0, 1, 2, 3, 4, 3, 4], dtype=np.int32),
        "state_source_type": np.ones(num_pins, dtype=np.int32),
        "state_role": np.zeros(num_pins, dtype=np.int32),
        "state_valid": np.ones((num_pins, 2), dtype=bool),
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
            [[35.0, 0.0, 0.0, 0.0], [200.0, 0.0, 0.0, 0.0]], dtype=np.float64
        ),
    }
    add_transition_masks(arrays, cell_arcs)

    config = {
        "num_pins": num_pins,
        "num_nets": 4,
        "num_instances": 3,
        "num_cells": 4,
        "aggregation_mode": "hard",
        "aggregation_tau_ps": 2.0,
        "pairwise_smooth": False,
        "pairwise_alpha": 20.0,
        "slew_epsilon": 1e-8,
        "pin2node_map": pin2node,
        "pin2net_map": pin2net,
        "pin_2_libpin_offset": pin_offset,
        "net_driver_pin": np.array([0, 2, 4, 6], dtype=np.int32),
        "lib_arc_pin_offsets": lib["lib_arc_pin_offsets"],
        "inst_cell_id": inst_cells,
        "cell_id_2_arc_id_start": np.arange(5, dtype=np.int64),
        "cell_id_2_libpin_id_start": np.arange(0, 10, 2, dtype=np.int64),
        "flat_libarc_info": lib["flat_libarc_info"],
        "flat_lib_pin_cap": lib["lib_caps"],
        "flat_lib_pin_rcap": lib["lib_caps"],
        "flat_lib_pin_fcap": lib["lib_caps"],
        "flat_lib_pin_slew_limit": lib["lib_slew_lim"],
        "flat_lib_pin_cap_limit": lib["lib_cap_lim"],
        "pin_cap_base": pin_cap_base,
        "pin_rcap_base": pin_cap_base,
        "pin_fcap_base": pin_cap_base,
        "pin_slew_limit": np.full(num_pins, 5.0, dtype=np.float64),
        "pin_cap_limit": np.array(
            [1.0, 1.0, 0.020, 1.0, 0.05, 1.0, 0.05], dtype=np.float64
        ),
        "luts": lib["luts"],
        "cell_arcs": cell_arcs,
        "cell_level_start": np.array([0, 0, 1, 3], dtype=np.int64),
        "setup_arcs": np.zeros((0, 8), dtype=np.int64),
        "check_arcs": np.zeros((0, 8), dtype=np.int64),
        "rc_forest": rc_forest,
        "sparse_mode": True,
        "sparse": {
            **arrays,
            "legal_checks": np.array([1, 1], dtype=np.uint8),
        },
    }
    # Base area = 2 + 2 + 2 = 6.0; capacity = 10.0 blocks BUF_SLOW_WIDE (+6 -> 12 > 10)
    # while allowing BUF_FAST (+1 -> 7 <= 10).
    policy = {
        "wns_coeff": 0.5,
        "tns_coeff": 1.0,
        "timing_weight": 1.0,
        "slew_weight": 1.0,
        "cap_weight": 1.0,
        "leakage_weight": 0.0,
        "loss_tolerance": 0.0,
        "slew_pin_mask": np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.uint8),
        "cap_pin_mask": np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
        "cell_width": lib["cell_width"],
        "cell_height": np.ones(4, dtype=np.float64),
        "cell_leakage": np.zeros(4, dtype=np.float64),
        "padding_x": 0.0,
        "num_movable_nodes": 3,
        "virtual_area": 0.0,
        "capacity": 10.0,
        "inflated_mask": np.zeros(3, dtype=np.uint8),
        "inflated_width": np.zeros(3, dtype=np.float64),
        "inflated_height": np.zeros(3, dtype=np.float64),
    }

    engine = timing_propagation_cpp.IncrementalSizingEngine(config)
    base_metrics = dict(engine.current_metrics())

    # Candidate 0: oversize BUF_SLOW_WIDE (+6 area -> 12 > 10) -> area rejected.
    # Candidate 1: BUF_FAST on zero-RC net 0 (0 -> 1), purely adding buffer delay
    #              on critical path with zero load shielding -> loss worsens -> rolled back!
    # Candidate 2: tree-node BUF_FAST at Steiner 7 shielding side branch [5] -> accepted!
    candidates = [
        {
            "net_id": 1,
            "parent_vertex": 7,
            "child_vertices": [5],
            "split_ratio": -1.0,
            "target_cell_id": 3,
        },
        {
            "net_id": 0,
            "parent_vertex": 0,
            "child_vertices": [1],
            "split_ratio": 0.5,
            "target_cell_id": 2,
        },
        {
            "net_id": 1,
            "parent_vertex": 7,
            "child_vertices": [5],
            "split_ratio": -1.0,
            "target_cell_id": 2,
        },
    ]

    # First verify Candidate 1 alone rolls back to exact base metrics with zero drift.
    rep_reject = dict(engine.evaluate_ranked_buffer_candidates([candidates[1]], policy))
    assert rep_reject["accepted_indices"] == []
    assert rep_reject["rejected_loss_count"] == 1
    assert engine.current_metrics() == pytest.approx(base_metrics)

    # Now evaluate all 3 in sequence.
    report = dict(engine.evaluate_ranked_buffer_candidates(candidates, policy))
    assert report["accepted_indices"] == [2]
    assert report["rejected_area_count"] == 1
    assert report["rejected_loss_count"] == 1
    assert report["accepted_count"] == 1
    assert report["final_loss"] < report["initial_loss"]
    assert report["final_wns_ps"] > base_metrics["wns_ps"]
    assert report["final_cap_violation_pf"] == pytest.approx(0.0)


def test_incremental_buffering_gate_filters_typed_and_masks_rejected_actions():
    """IncrementalBufferingGate rejects loss-worsening typed actions and masks them out."""
    base_cfg, base_policy = _build_chain_netlist(inst_cells=[0, 1], sparse_mode=True)
    cpp_engine = timing_propagation_cpp.IncrementalSizingEngine(base_cfg)

    owner = SimpleNamespace(
        params=SimpleNamespace(
            timing_buffering_incremental_commit=1,
            critical_endpoint_pruning_mode="off",
        ),
        placedb=SimpleNamespace(gr_sizing=None),
        data=SimpleNamespace(cell_id_2_arc_id_start=torch.arange(5)),
        _incremental_sizing_cache="cached",
    )
    model = SimpleNamespace(
        hold_objective=None,
        op_collections=SimpleNamespace(
            timing_propagation_op=SimpleNamespace(
                arcs_info=object(),
                num_pins=5,
                net2driver_pin_map=torch.zeros(3, dtype=torch.int32),
            )
        ),
    )
    gate = IncrementalBufferingGate(owner, model, torch.zeros(4), None, 20.0)
    gate._ensure_engine = lambda: (cpp_engine, False, 0.25)
    gate._policy = lambda: dict(base_policy)

    state = SimpleNamespace(
        is_typed_candidate=True,
        activation_param=torch.zeros((2, 2), dtype=torch.float32),
        drive_type_cell_ids=torch.tensor([3, 2], dtype=torch.long),  # 0=BUF_SLOW_WIDE, 1=BUF_FAST
        drive_type_allowed=torch.ones((2, 2), dtype=torch.bool),
        candidate_net_id=torch.tensor([0, 1], dtype=torch.long),
        candidate_records=[
            {
                "parent_node_id": 0,
                "child_node_ids": [1],
                "segment_split_ratio": 0.5,
            },
            {
                "parent_node_id": 2,
                "child_node_ids": [3],
                "segment_split_ratio": 0.5,
            },
        ],
    )

    # Round 1: scheduler proposes BUF_SLOW_WIDE (dtype 0 -> cell 3) on row 0 (net 0) -> rejected!
    bad_transition = TypedDiscreteCandidateTransition(
        next_activations=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        selected_candidate_indices=torch.tensor([0], dtype=torch.long),
        selected_drive_type_ids=torch.tensor([0], dtype=torch.long),
        selected_net_indices=torch.tensor([0], dtype=torch.long),
        selected_candidate_ids=torch.tensor([10], dtype=torch.long),
        selected_raw_grads=torch.tensor([-1.0]),
        selected_predicted_improvements=torch.tensor([1.0]),
        affected_net_count=2,
        eligible_net_count=2,
        prefix_size=1,
        positive_prefix_count=1,
    )
    filtered_bad, rep_bad = gate.filter_candidate_transition(state, bad_transition)
    assert filtered_bad.accepted_count == 0
    assert rep_bad["rejected_loss_count"] == 1
    allowed_after_reject = gate.allowed_typed_actions(state)
    assert allowed_after_reject.tolist() == [[False, True], [True, True]]

    # Tier 2 on same gradient: scheduler proposes BUF_FAST (dtype 1 -> cell 2) on row 1 -> accepted!
    good_transition = TypedDiscreteCandidateTransition(
        next_activations=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        selected_candidate_indices=torch.tensor([1], dtype=torch.long),
        selected_drive_type_ids=torch.tensor([1], dtype=torch.long),
        selected_net_indices=torch.tensor([1], dtype=torch.long),
        selected_candidate_ids=torch.tensor([11], dtype=torch.long),
        selected_raw_grads=torch.tensor([-0.8]),
        selected_predicted_improvements=torch.tensor([0.8]),
        affected_net_count=2,
        eligible_net_count=2,
        prefix_size=1,
        positive_prefix_count=1,
    )
    filtered_good, rep_good = gate.filter_candidate_transition(state, good_transition)
    assert filtered_good.accepted_count == 1
    assert filtered_good.next_activations.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert rep_good["accepted_count"] == 1
    assert owner._incremental_sizing_cache is None


def test_incremental_buffering_gate_checks_dynamic_virtual_count_after_static_cache():
    """Calling check_available() first must not hide later preexisting_virtual_count."""
    from dreamplace.Params import Params

    params = Params()
    owner = SimpleNamespace(
        params=params,
        placedb=SimpleNamespace(gr_sizing=None),
    )
    model = SimpleNamespace(
        hold_objective=None,
        op_collections=SimpleNamespace(
            timing_propagation_op=SimpleNamespace(arcs_info=object())
        ),
    )
    gate = IncrementalBufferingGate(owner, model, torch.zeros(2), None, 20.0)
    assert gate.check_available() is None
    assert (
        gate.check_available(preexisting_virtual_count=2)
        == "preexisting_virtual_buffers"
    )

    params.timing_buffering_incremental_commit = "off"
    disabled_gate = IncrementalBufferingGate(
        SimpleNamespace(params=params, placedb=owner.placedb),
        model,
        torch.zeros(2),
        None,
        20.0,
    )
    assert disabled_gate.check_available() == "disabled"


def test_cpp_refresh_dirty_subnet_steiner_trees_and_multi_depth_session():
    """Verify C++ 2n sub-net FLUTE refresh + IncrementalCandidateSession multi-depth export."""
    from dreamplace.ops.buffer_insertion.incremental_candidate_refresh import (
        IncrementalCandidateSession,
    )
    from dreamplace.ops.buffer_insertion.optimization_state import (
        build_typed_discrete_candidate_state,
    )
    from dreamplace.ops.net_subgraph_timing.candidate_live_geometry import (
        CandidateLiveGeometry,
    )
    from dreamplace.ops.steiner_topo.steiner_topo import (
        _FLUTE_POST_FILE,
        _FLUTE_POWV_FILE,
    )

    cfg, policy = _build_chain_netlist(
        inst_cells=[0, 1],
        total_r=600.0,
        total_c=0.240,
        sparse_mode=True,
    )
    policy["cap_weight"] = 10.0
    # Populate coordinates + wire RC unit + FLUTE LUTs on the 5-pin netlist (pins 0..4, 3 nets).
    cfg["rc_forest"]["vertex_x_dbu"] = np.array([0.0, 1000.0, 1000.0, 601000.0, 601000.0])
    cfg["rc_forest"]["vertex_y_dbu"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cfg["rc_forest"]["dbu"] = 1000.0
    cfg["rc_forest"]["r_unit"] = 1.0
    cfg["rc_forest"]["c_unit"] = 0.0004
    cfg["rc_forest"]["flute_powv_file"] = str(_FLUTE_POWV_FILE)
    cfg["rc_forest"]["flute_post_file"] = str(_FLUTE_POST_FILE)

    engine = timing_propagation_cpp.IncrementalSizingEngine(cfg)
    r0 = dict(
        engine.evaluate_ranked_buffer_candidates(
            [
                {
                    "net_id": 1,
                    "parent_vertex": 2,
                    "child_vertices": [3],
                    "split_ratio": 0.5,
                    "target_cell_id": 2,
                    "x_dbu": 301000.0,
                    "y_dbu": 0.0,
                }
            ],
            policy,
        )
    )
    assert r0["accepted_indices"] == [0]
    assert len(r0["dirty_nets"]) == 2
    assert r0["dirty_nets"] == sorted([1, r0["accepted_actions"][0]["new_net"]])

    # Build matching initial Python state & payload for both nets (0: pins 0->1, 1: pins 2->3).
    from dreamplace.ops.buffer_insertion.drive_type_table import DriveTypeEntry

    table = SimpleNamespace(
        entries=(
            DriveTypeEntry(
                drive_type_id=0,
                master_name="BUF_FAST",
                drive_strength=1.0,
                cell_id=2,
                bsu_index=0,
            ),
        ),
        to_metadata=lambda: {
            "masters": [
                {
                    "drive_type_id": 0,
                    "bsu_index": 0,
                    "cell_id": 2,
                    "master_name": "BUF_FAST",
                }
            ]
        },
    )
    rec0 = {
        "candidate_id": 1,
        "net_id": 0,
        "net_name": "net0",
        "driver_pin_id": 0,
        "driver_pin_name": "pi",
        "candidate_node_id": 2,
        "parent_node_id": 0,
        "child_node_ids": [1],
        "segment_split_ratio": 0.5,
        "buffer_main_type_index": 0,
        "x_dbu": 500.0,
        "y_dbu": 0.0,
        "candidate_location_x": 0.5,
        "candidate_location_y": 0.0,
        "downstream_pin_ids": (1,),
        "downstream_pin_names": ("drv/A",),
    }
    rec1 = {
        "candidate_id": 2,
        "net_id": 1,
        "net_name": "net1",
        "driver_pin_id": 2,
        "driver_pin_name": "drv/Y",
        "candidate_node_id": 5,
        "parent_node_id": 2,
        "child_node_ids": [3],
        "segment_split_ratio": 0.5,
        "buffer_main_type_index": 0,
        "x_dbu": 76000.0,
        "y_dbu": 0.0,
        "candidate_location_x": 76.0,
        "candidate_location_y": 0.0,
        "downstream_pin_ids": (3,),
        "downstream_pin_names": ("sink/A",),
    }
    state = build_typed_discrete_candidate_state(
        [rec0, rec1],
        buffer_main_type_index=0,
        legal_buffer_count=2,
        drive_type_table=table,
        dtype=torch.float64,
    )
    state.activation_param.data[1, 0] = 1.0
    # Nodes 0..2 belong to net 0; nodes 3..5 belong to net 1.
    payload = {
        "net_flat_topo_sort": torch.tensor([0, 2, 1, 3, 5, 4], dtype=torch.long),
        "net_flat_topo_sort_start": torch.tensor([0, 3, 6], dtype=torch.long),
        "pin_fa": torch.tensor([-1, 2, 0, -1, 5, 3], dtype=torch.long),
        "flat_pin_to_start": torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.long),
        "flat_pin_to": torch.tensor([2, 1, 5, 4], dtype=torch.long),
        "edge_resistance": torch.tensor(
            [0.0, 0.5, 0.5, 0.0, 75.0, 75.0], dtype=torch.float64
        ),
        "edge_capacitance": torch.tensor(
            [0.0, 0.0002, 0.0002, 0.0, 0.03, 0.03], dtype=torch.float64
        ),
        "node_capacitance": torch.tensor(
            [0.0, 0.002, 0.0, 0.0, 0.008, 0.0], dtype=torch.float64
        ),
        "candidate_node_id": torch.tensor([2, 5], dtype=torch.long),
        "candidate_net_id": torch.tensor([0, 1], dtype=torch.long),
        "sink_node_id": torch.tensor([1, 4], dtype=torch.long),
        "sink_pin_id": torch.tensor([1, 3], dtype=torch.long),
        "sink_net_index": torch.tensor([0, 1], dtype=torch.long),
        "per_size_input_cap": torch.tensor([[0.002, 0.002]] * 2, dtype=torch.float64),
        "metadata": {
            "net_ids": [0, 1],
            "compact_node_to_original": [0, 1, -1, 2, 3, -1],
        },
    }

    class _FakeGate:
        def refresh_dirty_subnets(self, dirty):
            return list(engine.refresh_dirty_subnet_steiner_trees(dirty))

        def remap_rejection_masks(self, **_kwargs):
            return None

    class _FakeLane:
        def __init__(self):
            self.buffer_optimization_state = state
            self.buffer_relaxed_timing_payload = payload

        def _current_state(self):
            return self.buffer_optimization_state

        def attach_state(self, data):
            data.buffer_optimization_state = self.buffer_optimization_state
            data.buffer_relaxed_timing_payload = self.buffer_relaxed_timing_payload

        def register_model_parameters(self, _model):
            return self

    lane = _FakeLane()
    placedb = SimpleNamespace(
        num_nodes=5,
        num_pins=5,
        pin_names=["pi", "drv/A", "drv/Y", "sink/A", "sink/Y"],
    )
    data = SimpleNamespace(dbu=1000)
    model = SimpleNamespace(placedb=placedb, data_collections=data)
    params = SimpleNamespace(
        dbu=1000,
        buffering_max_candidates_per_segment=3,
        buffering_include_tree_node_candidates=0,
        buffering_candidate_policy="rebuffer_sites",
        buffering_candidate_drive_type_mode="single_fixed",
    )
    session = IncrementalCandidateSession(
        lane=lane, model=model, params=params, incremental_gate=_FakeGate()
    )

    info0 = dict(r0["accepted_actions"][0])
    info0.update({"candidate_id": 2, "orig_net_id": 1, "drive_type_id": 0, "row_idx": 1})
    rep0 = session.refresh_round(
        accepted_action_infos=[info0],
        dirty_nets=r0["dirty_nets"],
    )
    assert rep0["status"] == "ok"
    assert rep0["dirty_subnet_count"] == 2
    assert rep0["dirty_orig_net_count"] == 1
    # Untouched net 0 row preserved; net 1 now has sub-net candidates with cpp_parent_vertex.
    new_state = lane.buffer_optimization_state
    new_payload = lane.buffer_relaxed_timing_payload
    assert bool(new_payload["static_rc_edge_mask"].any().item())
    locked_net1 = [
        rec
        for row, rec in enumerate(new_state.candidate_records)
        if int(rec["net_id"]) == 1 and float(new_state.activation_param[row, 0].item()) == 1.0
    ]
    assert len(locked_net1) == 1
    assert locked_net1[0]["segment_tree_depth"] == 0
    assert locked_net1[0]["downstream_pin_names"] == ("sink/A",)

    # Verify CandidateLiveGeometry accepts refreshed mixed static/live masks cleanly!
    geom = CandidateLiveGeometry(new_payload, new_state)
    geom.bind(
        new_x=torch.tensor([0.0, 1.0, 1.0, 601.0, 601.0], dtype=torch.float64),
        new_y=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        r_unit=1.0,
        c_unit=0.0004,
        scale_factor=0.001,
        dbu=1000.0,
        pin_capacitance_by_sense={
            "base": torch.tensor([0.0, 0.002, 0.0, 0.008, 0.0], dtype=torch.float64)
        },
    )
    assert geom.rc["edge_resistance"].shape[0] == new_payload["edge_resistance"].shape[0]

    # Round 1: insert a 2nd buffer DOWNSTREAM inside net 1's new_net -> verify depth 0 < depth 1!
    downstream_cand = max(
        (
            rec
            for rec in new_state.candidate_records
            if rec.get("cpp_subnet_refreshed")
            and int(rec["cpp_net_id"]) == int(r0["accepted_actions"][0]["new_net"])
        ),
        key=lambda rec: float(rec.get("segment_split_ratio", 0.0)),
    )
    r1 = dict(
        engine.evaluate_ranked_buffer_candidates(
            [
                {
                    "net_id": downstream_cand["cpp_net_id"],
                    "parent_vertex": downstream_cand["cpp_parent_vertex"],
                    "child_vertices": list(downstream_cand["cpp_child_vertices"]),
                    "split_ratio": downstream_cand["segment_split_ratio"],
                    "target_cell_id": 2,
                    "x_dbu": downstream_cand["x_dbu"],
                    "y_dbu": downstream_cand["y_dbu"],
                }
            ],
            policy,
        )
    )
    assert r1["accepted_indices"] == [0]
    info1 = dict(r1["accepted_actions"][0])
    info1.update(
        {
            "candidate_id": downstream_cand["candidate_id"],
            "orig_net_id": 1,
            "drive_type_id": 0,
        }
    )
    session.refresh_round(accepted_action_infos=[info1], dirty_nets=r1["dirty_nets"])
    final_state = lane.buffer_optimization_state
    locked_all = sorted(
        [
            rec
            for row, rec in enumerate(final_state.candidate_records)
            if int(rec["net_id"]) == 1 and float(final_state.activation_param[row, 0].item()) == 1.0
        ],
        key=lambda rec: rec["segment_tree_depth"],
    )
    assert [rec["segment_tree_depth"] for rec in locked_all] == [0, 1]
    assert locked_all[0]["downstream_pin_names"] == ("sink/A",)
    assert locked_all[1]["downstream_pin_names"] == ("sink/A",)

