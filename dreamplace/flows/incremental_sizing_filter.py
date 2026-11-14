"""Sequential C++ incremental-STA gate for gradient-ranked discrete sizing."""

import math
from time import perf_counter

import numpy as np
import torch

from dreamplace.flows.discrete_timing_common import parse_enabled_flag
from dreamplace.ops.discrete_gradient_topk.discrete_gradient_topk import sizes_to_logits
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    _target_vt_logits,
)
from dreamplace.ops.net_subgraph_timing.segment_count_live_geometry import (
    build_live_edge_geometry,
)


def _cpu_np(tensor, dtype):
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor, dtype=dtype, order="C")
    return (
        tensor.detach()
        .to(device="cpu")
        .numpy()
        .astype(dtype, copy=False)
        .__array__()
    )


def _contiguous_np(value, dtype):
    return np.ascontiguousarray(_cpu_np(value, dtype), dtype=dtype)


def common_incremental_sta_skip_reason(params, model) -> str | None:
    """Check runtime prerequisites shared by incremental sizing and buffering gates."""
    hold_obj = getattr(model, "hold_objective", None)
    if hold_obj is not None and (
        float(getattr(hold_obj, "wns", 0.0) or 0.0) > 0.0
        or float(getattr(hold_obj, "tns", 0.0) or 0.0) > 0.0
    ):
        return "hold_objective_active"
    if str(getattr(params, "critical_endpoint_pruning_mode", "off") or "off") != "off":
        return "critical_endpoint_pruning_active"
    op_collections = getattr(model, "op_collections", None)
    timing_op = (
        None
        if op_collections is None
        else getattr(op_collections, "timing_propagation_op", None)
    )
    if timing_op is None or getattr(timing_op, "arcs_info", None) is None:
        return "timing_propagation_unavailable"
    return None


def _skip_reason(owner, model):
    params = owner.params
    if not parse_enabled_flag(params, "timing_sizing_incremental_commit", default=True):
        return "disabled"
    relaxed_fn = getattr(model, "_relaxed_buffer_timing_enabled", None)
    if callable(relaxed_fn) and bool(relaxed_fn()):
        return "relaxed_buffer_timing_active"
    return common_incremental_sta_skip_reason(params, model)


def _build_rc_forest_payload(owner, model, pos, num_pins, num_nets):
    placedb = owner.placedb
    data = owner.data
    gr_session = getattr(placedb, "gr_sizing", None)
    if gr_session is not None:
        snap = gr_session.op.snapshot
        ident = snap.identity
        key = (
            "gr",
            ident.input_sha256,
            ident.geometry_sha256,
            int(ident.generation),
            int(num_pins),
            int(num_nets),
        )
        net_ids = _contiguous_np(snap.net_ids, np.int32)
        tree_by_net = np.full(num_nets, -1, dtype=np.int32)
        valid_nets = (net_ids >= 0) & (net_ids < num_nets)
        tree_by_net[net_ids[valid_nets]] = np.arange(len(net_ids), dtype=np.int32)[
            valid_nets
        ]
        wire_cap = _contiguous_np(snap.wire_cap, np.float64)
        incoming_res = _contiguous_np(snap.incoming_resistance, np.float64)
        payload = {
            "wire_cap": wire_cap,
            "incoming_resistance": incoming_res,
            "parent": _contiguous_np(snap.parent, np.int32),
            "child_start": _contiguous_np(snap.child_start, np.int32),
            "children": _contiguous_np(snap.child_vertex, np.int32),
            "topo": _contiguous_np(snap.topo_order, np.int32),
            "topo_start": _contiguous_np(snap.net_topo_start, np.int32),
            "tree_by_net": tree_by_net,
            "pin_to_vertex": _contiguous_np(snap.pin_to_vertex, np.int32),
        }
        return key, payload, wire_cap, incoming_res

    rc_op = model.op_collections.elmore_delay_op
    topo_op = getattr(model.op_collections, "steiner_topo_op", None)
    topo_gen = int(getattr(topo_op, "topology_generation", 0) or 0)
    with torch.no_grad():
        _pin_pos, new_x, new_y = model._timing_geometry_for_pos(pos.detach())
        geom = build_live_edge_geometry(
            new_x,
            new_y,
            data.flat_pin_from,
            data.flat_pin_to,
            r_unit=rc_op.r_unit,
            c_unit=rc_op.c_unit,
            scale_factor=rc_op.scale_factor,
            dbu=rc_op.dbu,
        )
        edge_cap = geom["edge_capacitance"].to(dtype=torch.float64)
        net_caps = torch.zeros(
            new_x.numel(), dtype=torch.float64, device=edge_cap.device
        )
        from_idx = data.flat_pin_from.to(device=edge_cap.device, dtype=torch.long)
        to_idx = data.flat_pin_to.to(device=edge_cap.device, dtype=torch.long)
        net_caps.scatter_add_(0, from_idx, edge_cap * 0.5)
        net_caps.scatter_add_(0, to_idx, edge_cap * 0.5)
        edge_res = geom["edge_resistance"].to(dtype=torch.float64)
        flat_res = torch.zeros(
            new_x.numel(), dtype=torch.float64, device=edge_res.device
        )
        flat_res.scatter_(0, to_idx, edge_res)
        flat_edge_cap = torch.zeros(
            new_x.numel(), dtype=torch.float64, device=edge_cap.device
        )
        flat_edge_cap.scatter_(0, to_idx, edge_cap)

    wire_cap = _contiguous_np(net_caps, np.float64)
    incoming_res = _contiguous_np(flat_res, np.float64)
    incoming_wire_cap = _contiguous_np(flat_edge_cap, np.float64)
    scale = float(getattr(owner.params, "scale_factor", 1.0) or 1.0)
    shift = getattr(owner.params, "shift_factor", (0.0, 0.0)) or (0.0, 0.0)
    vertex_x_dbu = _contiguous_np(
        (new_x.detach().double() / scale + float(shift[0])).round(), np.float64
    )
    vertex_y_dbu = _contiguous_np(
        (new_y.detach().double() / scale + float(shift[1])).round(), np.float64
    )
    from dreamplace.ops.steiner_topo.steiner_topo import (
        _FLUTE_POST_FILE,
        _FLUTE_POWV_FILE,
    )

    pin_fa_sig = (int(data.pin_fa.data_ptr()), int(data.pin_fa._version))
    key = (
        "placement",
        topo_gen,
        pin_fa_sig,
        int(new_x.numel()),
        int(num_pins),
        int(num_nets),
    )
    payload = {
        "wire_cap": wire_cap,
        "incoming_resistance": incoming_res,
        "incoming_wire_cap": incoming_wire_cap,
        "parent": _contiguous_np(data.pin_fa, np.int32),
        "child_start": _contiguous_np(data.flat_pin_to_start, np.int32),
        "children": _contiguous_np(data.flat_pin_to, np.int32),
        "topo": _contiguous_np(data.net_flat_topo_sort, np.int32),
        "topo_start": _contiguous_np(data.net_flat_topo_sort_start, np.int32),
        "tree_by_net": np.arange(num_nets, dtype=np.int32),
        "pin_to_vertex": np.arange(num_pins, dtype=np.int32),
        "vertex_x_dbu": vertex_x_dbu,
        "vertex_y_dbu": vertex_y_dbu,
        "dbu": float(rc_op.dbu),
        "r_unit": float(rc_op.r_unit),
        "c_unit": float(rc_op.c_unit),
        "flute_powv_file": str(_FLUTE_POWV_FILE),
        "flute_post_file": str(_FLUTE_POST_FILE),
    }
    return key, payload, wire_cap, incoming_res


def _decode_lib_name(val) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _compute_lib_arc_pin_offsets(placedb, data):
    arc_names = getattr(placedb, "flat_libarc_names", None)
    pin_names = getattr(placedb, "flat_lib_pin_names", None)
    if arc_names is None or pin_names is None:
        return None
    arc_names_np = np.asarray(arc_names)
    pin_names_np = np.asarray(pin_names)
    if arc_names_np.ndim != 2 or arc_names_np.shape[1] < 2:
        return None
    cache_key = (int(arc_names_np.shape[0]), int(pin_names_np.shape[0]))
    cached = getattr(placedb, "_cached_lib_arc_pin_offsets", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    arc_start = _contiguous_np(data.cell_id_2_arc_id_start, np.int64)
    pin_start = _contiguous_np(data.cell_id_2_libpin_id_start, np.int64)
    num_cells = len(arc_start) - 1
    offsets = np.full((int(arc_names_np.shape[0]), 2), -1, dtype=np.int64)
    for cell in range(num_cells):
        a0, a1 = int(arc_start[cell]), int(arc_start[cell + 1])
        if a0 >= a1:
            continue
        p0, p1 = int(pin_start[cell]), int(pin_start[cell + 1])
        pin_map = {
            _decode_lib_name(pin_names_np[p]): p - p0 for p in range(p0, p1)
        }
        for a in range(a0, a1):
            offsets[a, 0] = pin_map.get(_decode_lib_name(arc_names_np[a, 0]), -1)
            offsets[a, 1] = pin_map.get(_decode_lib_name(arc_names_np[a, 1]), -1)
    placedb._cached_lib_arc_pin_offsets = (cache_key, offsets)
    return offsets


def _build_engine_config(owner, model, rc_payload):
    from dreamplace.ops.timing_propagation.recovery_checks import (
        TIMING_CHECK_CLASS_RECOVERY,
    )
    from dreamplace.ops.timing_propagation.timing_propagation import (
        SLEW_SQRT_EPS,
        SMOOTH_MAX_ALPHA,
        SMOOTH_MAX_ENABLED,
        _force_hard_pairwise_max_enabled,
    )

    data = owner.data
    timing = model.op_collections.timing_propagation_op
    num_pins = int(timing.num_pins)
    num_nets = int(timing.net2driver_pin_map.numel())
    num_instances = int(data.inst_cell_id.numel())
    num_cells = int(data.cell_id_2_arc_id_start.numel()) - 1

    with torch.no_grad():
        (
            pin2libpin_flat_ids,
            inst_pins_mask,
            pin_cap_base,
            pin_rcap_base,
            pin_fcap_base,
        ) = model._pin_caps_from_offsets(data.inst_libcell_offset, data)
        end_points = getattr(data, "end_points", None)
        outcaps = getattr(data, "outcaps", None)
        if end_points is not None and outcaps is not None and end_points.numel():
            ep = end_points.to(device=pin_cap_base.device, dtype=torch.long)
            oc = outcaps.to(device=pin_cap_base.device, dtype=pin_cap_base.dtype)
            pin_cap_base = pin_cap_base.clone()
            pin_rcap_base = pin_rcap_base.clone()
            pin_fcap_base = pin_fcap_base.clone()
            pin_cap_base[ep] += oc
            pin_rcap_base[ep] += oc
            pin_fcap_base[ep] += oc
        model.pin2libpin_flat_ids = pin2libpin_flat_ids
        model.inst_pins_mask = inst_pins_mask
        slew_limits = data.flat_lib_pin_slew_limit[
            pin2libpin_flat_ids.to(device=data.flat_lib_pin_slew_limit.device)
        ]
        cap_limits = data.flat_lib_pin_cap_limit[
            pin2libpin_flat_ids.to(device=data.flat_lib_pin_cap_limit.device)
        ]

    luts = []
    for name in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        table = getattr(timing.arcs_info, f"{name}_luts")
        luts.append(
            [
                _contiguous_np(table.flat_luts_values, np.float64),
                _contiguous_np(table.flat_luts_trans_table, np.float64),
                _contiguous_np(table.flat_luts_cap_table, np.float64),
                _contiguous_np(table.trans_dims_actual, np.int64),
                _contiguous_np(table.cap_dims_actual, np.int64),
            ]
        )

    setup_arcs = getattr(timing, "endpoints_constraint_arcs", None)
    check_arcs = getattr(timing, "endpoints_timing_check_arcs", None)
    empty_arcs = np.zeros((0, 8), dtype=np.int64)

    sparse_view = getattr(timing, "sparse_timing_view", None)
    sparse_mode = sparse_view is not None
    config = {
        "num_pins": num_pins,
        "num_nets": num_nets,
        "num_instances": num_instances,
        "num_cells": num_cells,
        "aggregation_mode": str(getattr(timing, "timing_aggregation_mode", "hard") or "hard"),
        "aggregation_tau_ps": float(getattr(timing, "timing_aggregation_tau_ps", 2.0) or 2.0),
        "pairwise_smooth": bool(SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled()),
        "pairwise_alpha": float(SMOOTH_MAX_ALPHA),
        "slew_epsilon": float(SLEW_SQRT_EPS),
        "pin2node_map": _contiguous_np(data.pin2node_map, np.int32),
        "pin2net_map": _contiguous_np(
            getattr(data, "pin2net_map", timing.pin_net), np.int32
        ),
        "pin_2_libpin_offset": _contiguous_np(data.pin_2_libpin_offset, np.int32),
        "inst_cell_id": _contiguous_np(data.inst_cell_id, np.int32),
        "cell_id_2_arc_id_start": _contiguous_np(data.cell_id_2_arc_id_start, np.int64),
        "cell_id_2_libpin_id_start": _contiguous_np(
            data.cell_id_2_libpin_id_start, np.int64
        ),
        "flat_libarc_info": _contiguous_np(data.flat_libarc_info, np.int64),
        "flat_libarc_check_types": (
            None
            if getattr(data, "flat_libarc_check_types", None) is None
            else _contiguous_np(data.flat_libarc_check_types, np.int64)
        ),
        "lib_arc_pin_offsets": _compute_lib_arc_pin_offsets(owner.placedb, data),
        "net_driver_pin": _contiguous_np(timing.net2driver_pin_map, np.int32),
        "flat_lib_pin_cap": _contiguous_np(data.flat_lib_pin_cap, np.float64),
        "flat_lib_pin_rcap": _contiguous_np(data.flat_lib_pin_rcap, np.float64),
        "flat_lib_pin_fcap": _contiguous_np(data.flat_lib_pin_fcap, np.float64),
        "flat_lib_pin_slew_limit": _contiguous_np(
            data.flat_lib_pin_slew_limit, np.float64
        ),
        "flat_lib_pin_cap_limit": _contiguous_np(
            data.flat_lib_pin_cap_limit, np.float64
        ),
        "pin_cap_base": _contiguous_np(pin_cap_base, np.float64),
        "pin_rcap_base": _contiguous_np(pin_rcap_base, np.float64),
        "pin_fcap_base": _contiguous_np(pin_fcap_base, np.float64),
        "pin_slew_limit": _contiguous_np(slew_limits, np.float64),
        "pin_cap_limit": _contiguous_np(cap_limits, np.float64),
        "luts": luts,
        "cell_arcs": _contiguous_np(timing.flat_inst_arcs_by_level, np.int64),
        "cell_level_start": _contiguous_np(
            timing.flat_inst_arcs_by_level_start, np.int64
        ),
        "setup_arcs": (
            empty_arcs
            if setup_arcs is None or setup_arcs.numel() == 0
            else _contiguous_np(setup_arcs, np.int64)
        ),
        "check_arcs": (
            empty_arcs
            if check_arcs is None or check_arcs.numel() == 0
            else _contiguous_np(check_arcs, np.int64)
        ),
        "rc_forest": rc_payload,
        "sparse_mode": sparse_mode,
    }

    if sparse_mode:
        arrays = sparse_view.arrays
        checks = arrays["check_index"]
        legal = checks[:, 6].astype(bool).copy()
        if str(getattr(timing, "timing_metric_scope", "setup_plus_recovery")) == "setup_only":
            check_np = config["check_arcs"]
            dynamic = checks[:, 2] >= 0
            if len(check_np):
                legal[dynamic] &= (
                    check_np[checks[dynamic, 2], 6] != TIMING_CHECK_CLASS_RECOVERY
                )
        config["sparse"] = {
            "state_pin": _contiguous_np(arrays["state_pin"], np.int32),
            "state_slew_group": _contiguous_np(arrays["state_slew_group"], np.int32),
            "state_level": _contiguous_np(arrays["state_level"], np.int32),
            "state_valid": np.ascontiguousarray(arrays["state_valid"], dtype=bool),
            "state_seed": np.ascontiguousarray(arrays["state_seed"], dtype=bool),
            "state_reference": _contiguous_np(arrays["state_reference"], np.float64),
            "state_source_type": (
                None
                if "state_source_type" not in arrays
                else _contiguous_np(arrays["state_source_type"], np.int32)
            ),
            "state_role": (
                None
                if "state_role" not in arrays
                else _contiguous_np(arrays["state_role"], np.int32)
            ),
            "cell_edges": _contiguous_np(arrays["cell_edges"], np.int64),
            "cell_transition_mask": _contiguous_np(
                arrays["cell_transition_mask"], np.int64
            ),
            "slew_only_cell_edges": _contiguous_np(
                arrays["slew_only_cell_edges"], np.int64
            ),
            "slew_only_cell_transition_mask": _contiguous_np(
                arrays["slew_only_cell_transition_mask"], np.int64
            ),
            "net_edges": _contiguous_np(arrays["net_edges"], np.int64),
            "net_transition_mask": _contiguous_np(
                arrays["net_transition_mask"], np.int64
            ),
            "net_threshold_adjustments": _contiguous_np(
                arrays["net_threshold_adjustments"], np.float64
            ),
            "check_index": _contiguous_np(checks, np.int64),
            "check_values": _contiguous_np(arrays["check_values"], np.float64),
            "legal_checks": np.ascontiguousarray(legal, dtype=np.uint8),
        }
    else:
        net_arcs = getattr(timing, "net_flat_arcs", None)
        config.update(
            {
                "start_pins": _contiguous_np(timing.start_points, np.int64),
                "start_rAAT": _contiguous_np(timing.inrdelays, np.float64),
                "start_fAAT": _contiguous_np(timing.infdelays, np.float64),
                "start_rtran": _contiguous_np(timing.inrtrans, np.float64),
                "start_ftran": _contiguous_np(timing.inftrans, np.float64),
                "clock_pins": _contiguous_np(timing.clock_pins, np.int64),
                "clock_rtran": _contiguous_np(timing.clk_pin_rtran, np.float64),
                "clock_ftran": _contiguous_np(timing.clk_pin_ftran, np.float64),
                "net_arcs": (
                    np.zeros((0, 2), dtype=np.int64)
                    if net_arcs is None or net_arcs.numel() == 0
                    else _contiguous_np(net_arcs, np.int64)
                ),
                "endpoint_pins": _contiguous_np(timing.end_points, np.int64),
                "endpoint_rRAT": _contiguous_np(timing.endpoints_rRAT, np.float64),
                "endpoint_fRAT": _contiguous_np(timing.endpoints_fRAT, np.float64),
                "endpoint_valid": (
                    None
                    if getattr(timing, "endpoint_max_valid_by_pin", None) is None
                    else _contiguous_np(timing.endpoint_max_valid_by_pin, bool)
                ),
                "setup_valid": (
                    None
                    if getattr(timing, "endpoints_constraint_max_valid", None) is None
                    else _contiguous_np(timing.endpoints_constraint_max_valid, bool)
                ),
                "check_valid": (
                    None
                    if getattr(timing, "endpoints_timing_check_max_valid", None) is None
                    else _contiguous_np(timing.endpoints_timing_check_max_valid, bool)
                ),
                "metric_scope": str(
                    getattr(timing, "timing_metric_scope", "setup_plus_recovery")
                ),
                "recovery_uses_live_slew": bool(
                    getattr(timing, "recovery_check_uses_live_slew", True)
                ),
                "recovery_rat_is_capture_deadline": bool(
                    getattr(timing, "recovery_rat_is_capture_deadline", False)
                ),
                "backend_rSlew": (
                    None
                    if getattr(timing, "backend_endpoint_rSlew", None) is None
                    else _contiguous_np(timing.backend_endpoint_rSlew, np.float64)
                ),
                "backend_fSlew": (
                    None
                    if getattr(timing, "backend_endpoint_fSlew", None) is None
                    else _contiguous_np(timing.backend_endpoint_fSlew, np.float64)
                ),
            }
        )
    return config


def _phase_policy(owner, model, geometry, capacity, num_pins, num_cells):
    data = owner.data
    params = owner.params
    placedb = owner.placedb
    enabled_terms = set(model._timing_objective_lane_enabled_terms())
    balance_weight = float(getattr(model, "timing_grad_balance_weight", 1.0))

    def _scaled(name, raw_weight):
        if name != "timing" and name not in enabled_terms:
            return 0.0
        scale = balance_weight if model._timing_grad_balance_scales_term(name) else 1.0
        return float(raw_weight) * scale

    timing_weight = _scaled("timing", 1.0)
    slew_weight = _scaled(
        "slew",
        getattr(model, "timing_slew_weight", getattr(params, "timing_slew_weight", 1.0)),
    )
    cap_weight = _scaled(
        "cap",
        getattr(model, "timing_cap_weight", getattr(params, "timing_cap_weight", 1.0)),
    )
    leakage_weight = _scaled(
        "leakage",
        getattr(
            model,
            "timing_leakage_weight",
            getattr(params, "timing_leakage_weight", 0.0),
        ),
    )

    slew_mask = getattr(model, "inst_pins_mask", None)
    cap_mask = getattr(model, "output_pin_mask", None)
    num_movable = int(getattr(placedb, "num_movable_nodes", 0) or 0)
    if geometry is not None and hasattr(geometry, "inflated"):
        inflated_mask = _contiguous_np(geometry.inflated[:num_movable], np.uint8)
        inflated_width = _contiguous_np(geometry.width[:num_movable], np.float64)
        inflated_height = _contiguous_np(geometry.height[:num_movable], np.float64)
    else:
        inflated_mask = np.zeros(num_movable, dtype=np.uint8)
        inflated_width = np.zeros(num_movable, dtype=np.float64)
        inflated_height = np.zeros(num_movable, dtype=np.float64)

    leakage = getattr(data, "flat_libcell_leakage", None)
    return {
        "wns_coeff": float(model.timing_wns_coeff),
        "tns_coeff": float(model.timing_tns_coeff),
        "timing_weight": timing_weight,
        "slew_weight": slew_weight,
        "cap_weight": cap_weight,
        "leakage_weight": leakage_weight,
        "loss_tolerance": 0.0,
        "slew_pin_mask": (
            np.ones(num_pins, dtype=np.uint8)
            if slew_mask is None
            else _contiguous_np(slew_mask, np.uint8)
        ),
        "cap_pin_mask": (
            np.zeros(num_pins, dtype=np.uint8)
            if cap_mask is None
            else _contiguous_np(cap_mask, np.uint8)
        ),
        "cell_width": _contiguous_np(data.flat_libcell_width, np.float64),
        "cell_height": _contiguous_np(data.flat_libcell_height, np.float64),
        "cell_leakage": (
            np.zeros(num_cells, dtype=np.float64)
            if leakage is None
            else _contiguous_np(leakage, np.float64)
        ),
        "padding_x": float(getattr(params, "cell_padding_x", 0.0) or 0.0),
        "num_movable_nodes": num_movable,
        "virtual_area": float(owner.virtual_area(model)),
        "capacity": float(capacity),
        "inflated_mask": inflated_mask,
        "inflated_width": inflated_width,
        "inflated_height": inflated_height,
    }


def _rebuild_filtered_outputs(
    data,
    base_size,
    base_vt,
    summary,
    accepted_instance_ids,
    accepted_cell_ids,
    eval_report,
    *,
    size_parameterization="real_size",
    eps=1.0e-6,
):
    accepted_map = {
        int(inst): int(cell)
        for inst, cell in zip(accepted_instance_ids, accepted_cell_ids, strict=True)
    }
    selected_scores = list(summary.get("selected_scores") or ())
    kept_items = [
        item for item in selected_scores if int(item["instance_id"]) in accepted_map
    ]
    # Canonical instance-id ascending order matches quad_gradient_topk.py.
    kept_items.sort(key=lambda item: int(item["instance_id"]))

    applied_instance_ids = [int(item["instance_id"]) for item in kept_items]
    applied_cell_ids = [int(item["target_cell_id"]) for item in kept_items]
    applied_sizes = [float(item["target_size"]) for item in kept_items]
    applied_vts = [int(item["target_vt"]) for item in kept_items]
    applied_actions = [str(item["action"]) for item in kept_items]
    applied_deltas = [
        1 if action in ("size_up", "vt_speedup") else -1
        for action in applied_actions
    ]
    action_counts = {
        action: sum(a == action for a in applied_actions) for action in ACTION_NAMES
    }

    next_size = base_size.detach().clone()
    next_vt = base_vt.detach().clone()
    if applied_instance_ids:
        rows = torch.tensor(
            applied_instance_ids, dtype=torch.long, device=next_size.device
        )
        sizes_tensor = torch.tensor(
            applied_sizes, dtype=next_size.dtype, device=next_size.device
        )
        if size_parameterization == "real_size":
            next_size[rows] = sizes_tensor
        else:
            lower = data.inst_size_lower[
                rows.to(device=data.inst_size_lower.device)
            ].to(device=next_size.device, dtype=next_size.dtype)
            upper = data.inst_size_upper[
                rows.to(device=data.inst_size_upper.device)
            ].to(device=next_size.device, dtype=next_size.dtype)
            next_size[rows] = sizes_to_logits(sizes_tensor, lower, upper, eps)
        vt_rows = rows.to(device=next_vt.device)
        vts_tensor = torch.tensor(
            applied_vts, dtype=torch.long, device=next_vt.device
        )
        next_vt[vt_rows] = _target_vt_logits(
            vts_tensor,
            int(next_vt.shape[1]),
            dtype=next_vt.dtype,
            device=next_vt.device,
            eps=eps,
        )

    updated = dict(summary)
    updated.update(
        {
            "proposed_action_counts": dict(summary.get("action_counts") or {}),
            "num_proposed_instances": int(summary.get("num_changed_instances", 0) or 0),
            "num_up_applied": action_counts["size_up"],
            "num_down_applied": action_counts["size_down"],
            "num_vt_applied": action_counts["vt_slowdown"] + action_counts["vt_speedup"],
            "num_vt_slowdown_applied": action_counts["vt_slowdown"],
            "num_vt_speedup_applied": action_counts["vt_speedup"],
            "action_counts": dict(action_counts),
            "num_changed_instances": len(applied_instance_ids),
            "applied_instance_ids": applied_instance_ids,
            "applied_cell_ids": applied_cell_ids,
            "applied_sizes": applied_sizes,
            "applied_vts": applied_vts,
            "applied_actions": applied_actions,
            "applied_legal_index_deltas": applied_deltas,
            "selected_scores": kept_items,
            "incremental_commit": eval_report,
        }
    )
    if "step" in updated and isinstance(updated["step"], dict):
        step = dict(updated["step"])
        step["direct_step_num_instances"] = len(applied_instance_ids)
        step["mean_abs_legal_index_delta"] = 1.0 if applied_instance_ids else 0.0
        step["max_abs_legal_index_delta"] = 1 if applied_instance_ids else 0
        updated["step"] = step
    return next_size, next_vt, updated


def filter_sizing_candidates_incrementally(
    owner,
    model,
    pos,
    geometry,
    capacity,
    base_size,
    base_vt,
    next_size,
    next_vt,
    summary,
    *,
    size_parameterization="real_size",
):
    """Evaluate gradient-ranked sizing candidates one-by-one via C++ incremental STA."""
    reason = _skip_reason(owner, model)
    if reason is not None:
        if reason != "disabled":
            summary = dict(summary)
            summary["incremental_commit"] = {"status": "skipped", "reason": reason}
        return next_size, next_vt, summary

    selected_scores = list(summary.get("selected_scores") or ())
    if not selected_scores:
        summary = dict(summary)
        summary["incremental_commit"] = {
            "status": "ok",
            "input_count": 0,
            "evaluated_count": 0,
            "accepted_count": 0,
            "rejected_area_count": 0,
            "rejected_loss_count": 0,
            "elapsed_ms": 0.0,
        }
        return next_size, next_vt, summary

    from dreamplace.ops.timing_propagation import timing_propagation_cpp

    started = perf_counter()
    timing = model.op_collections.timing_propagation_op
    num_pins = int(timing.num_pins)
    num_nets = int(timing.net2driver_pin_map.numel())
    num_cells = int(owner.data.cell_id_2_arc_id_start.numel()) - 1

    rc_key, rc_payload, wire_cap, incoming_res = _build_rc_forest_payload(
        owner, model, pos, num_pins, num_nets
    )
    engine_key = (
        rc_key,
        str(getattr(timing, "timing_aggregation_mode", "hard") or "hard"),
        float(getattr(timing, "timing_aggregation_tau_ps", 2.0) or 2.0),
        str(getattr(timing, "timing_metric_scope", "setup_plus_recovery")),
        int(getattr(timing, "critical_path_topology_epoch", 0) or 0),
    )
    cache = getattr(owner, "_incremental_sizing_cache", None)
    cache_hit = cache is not None and cache.get("key") == engine_key
    build_ms = 0.0
    sync_ms = 0.0
    if not cache_hit:
        build_started = perf_counter()
        engine_cfg = _build_engine_config(owner, model, rc_payload)
        engine = timing_propagation_cpp.IncrementalSizingEngine(engine_cfg)
        build_ms = (perf_counter() - build_started) * 1000.0
        owner._incremental_sizing_cache = {"key": engine_key, "engine": engine}
    else:
        sync_started = perf_counter()
        engine = cache["engine"]
        inst_cells = _contiguous_np(owner.data.inst_cell_id, np.int64)
        if rc_key[0] == "gr":
            engine.sync_state(inst_cells)
        else:
            engine.sync_state(inst_cells, wire_cap, incoming_res)
        sync_ms = (perf_counter() - sync_started) * 1000.0

    # Order strictly by gradient score ascending (strongest predicted gain first).
    ranked = sorted(
        selected_scores,
        key=lambda item: (
            float(item.get("predicted_delta_obj", math.inf)),
            int(item["instance_id"]),
        ),
    )
    ranked_inst_ids = [int(item["instance_id"]) for item in ranked]
    ranked_cell_ids = [int(item["target_cell_id"]) for item in ranked]

    policy = _phase_policy(owner, model, geometry, capacity, num_pins, num_cells)
    eval_result = dict(
        engine.evaluate_ranked_candidates(ranked_inst_ids, ranked_cell_ids, policy)
    )
    eval_result["cache_hit"] = bool(cache_hit)
    eval_result["engine_build_ms"] = float(build_ms)
    eval_result["engine_sync_ms"] = float(sync_ms)
    eval_result["total_filter_ms"] = float((perf_counter() - started) * 1000.0)

    return _rebuild_filtered_outputs(
        owner.data,
        base_size,
        base_vt,
        summary,
        eval_result["accepted_instance_ids"],
        eval_result["accepted_cell_ids"],
        eval_result,
        size_parameterization=size_parameterization,
    )


class IncrementalSizingGate:
    """Per-window incremental STA filter with unchanged-state candidate advancement."""

    def __init__(self, owner, model, geometry, capacity):
        self.owner = owner
        self.model = model
        self.geometry = geometry
        self.capacity = capacity
        self._rejected_actions = None
        self._rejected_instances = None

    def mask_valid_actions(self, selection_valid):
        if self._rejected_actions is None:
            return selection_valid
        return selection_valid & ~self._rejected_actions.to(
            device=selection_valid.device
        )

    def mask_blocked_instances(self, blocked):
        if self._rejected_instances is None:
            return blocked
        return blocked | self._rejected_instances.to(device=blocked.device)

    def filter_round(
        self,
        pos,
        base_size,
        base_vt,
        next_size,
        next_vt,
        summary,
        *,
        size_parameterization="real_size",
    ):
        proposed = list(summary.get("selected_scores") or ())
        filtered_size, filtered_vt, updated = filter_sizing_candidates_incrementally(
            self.owner,
            self.model,
            pos,
            self.geometry,
            self.capacity,
            base_size,
            base_vt,
            next_size,
            next_vt,
            summary,
            size_parameterization=size_parameterization,
        )
        report = updated.get("incremental_commit")
        if not isinstance(report, dict) or report.get("status") != "ok":
            return filtered_size, filtered_vt, updated
        if int(updated.get("num_changed_instances", 0) or 0) > 0:
            self._rejected_actions = None
            self._rejected_instances = None
        elif proposed:
            num_inst = int(self.owner.data.inst_cell_id.numel())
            if self._rejected_actions is None:
                self._rejected_actions = torch.zeros(
                    (num_inst, len(ACTION_NAMES)), dtype=torch.bool
                )
            if self._rejected_instances is None:
                self._rejected_instances = torch.zeros(num_inst, dtype=torch.bool)
            for item in proposed:
                inst = int(item.get("instance_id", -1))
                action = str(item.get("action", ""))
                if 0 <= inst < num_inst:
                    self._rejected_instances[inst] = True
                    if action in ACTION_NAMES:
                        self._rejected_actions[inst, ACTION_NAMES.index(action)] = True
        return filtered_size, filtered_vt, updated


__all__ = ["IncrementalSizingGate", "filter_sizing_candidates_incrementally"]
