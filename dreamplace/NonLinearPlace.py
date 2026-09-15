##
# @file   NonLinearPlace.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Nonlinear placement engine to be called with parameters and placement database
#

import os
import sys
import time
import pickle
from typing import Any
import numpy as np
import logging
import torch
import torch.nn.functional as F
import gzip
import copy
import matplotlib.pyplot as plt
import inspect

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.BasicPlace as BasicPlace
import dreamplace.PlaceObj as PlaceObj
import dreamplace.NesterovAcceleratedGradientOptimizer as NesterovAcceleratedGradientOptimizer
import dreamplace.EvalMetrics as EvalMetrics
import pdb
import dreamplace.ops.fence_region.fence_region as fence_region
import math

from dreamplace.ops.routability.egr_resample import (
    create_supply_map_from_placedb,
    create_supply_and_demand_maps_from_egr,
    create_directional_supply_and_demand_maps_from_egr,
)
from dreamplace.ops.routability.same_net_topo_scoring import (
    build_same_net_topology_cache,
)
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose, profile_scope
from dreamplace.ops.routability.l_shape_electric_potential import (
    compute_fixed_macro_overlap_stats,
    compute_movable_displacement_stats,
)
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
    use_ggr_l_shape_topology,
    validate_ggr_l_shape_topology_params,
)
from dreamplace.ops.routability.leiden_clustering import (
    build_active_leiden_clusters,
    plot_modularity_clusters,
)
from dreamplace.ops.routability import enhanced_inflation_controller
from dreamplace.ops.irt_egr.egr_padding import apply_egr_padding, restore_egr_padding
from dreamplace import inflation_legalization
from dreamplace.post_legalization_adaptive_padding import (
    allocate_padding_sites,
    build_smoothed_overflow_map,
    compute_cell_box_overlap_stats,
    score_cells_from_overflow,
)


def _log_inflation_macro_overlap(
    placedb,
    pos,
    data_collections,
    round_idx,
    stage,
    reference_pos=None,
):
    try:
        overlap = compute_fixed_macro_overlap_stats(
            pos,
            data_collections.node_size_x,
            data_collections.node_size_y,
            placedb,
        )
        displacement = (
            compute_movable_displacement_stats(reference_pos, pos, placedb)
            if reference_pos is not None
            else {}
        )
        logging.info(
            "Inflation legalization telemetry: round=%d stage=%s "
            "macro_overlap_cells=%d macro_overlap_pairs=%d "
            "macro_overlap_area=%.6E macro_overlap_area_ratio=%.6E "
            "moved=%d displacement_max=%.6E displacement_mean=%.6E",
            int(round_idx),
            stage,
            int(overlap.get("fixed_macro_overlap_cell_count", 0)),
            int(overlap.get("fixed_macro_overlap_pair_count", 0)),
            float(overlap.get("fixed_macro_overlap_area", 0.0)),
            float(overlap.get("fixed_macro_overlap_area_ratio", 0.0)),
            int(displacement.get("movable_displacement_moved_count", 0)),
            float(displacement.get("movable_displacement_max", 0.0)),
            float(displacement.get("movable_displacement_mean", 0.0)),
        )
    except Exception as error:
        logging.warning(
            "Failed to compute inflation legalization telemetry for round %d "
            "at stage %s: %s",
            int(round_idx),
            stage,
            error,
        )


def _snapshot_l_shape_forward_source():
    from dreamplace.ops.routability.l_shape_electric_potential import (
        SegmentElectricPotentialFunction,
    )

    return (
        SegmentElectricPotentialFunction.last_rho_map,
        SegmentElectricPotentialFunction.last_rho_map_h,
        SegmentElectricPotentialFunction.last_rho_map_v,
    )


def _restore_l_shape_forward_source(snapshot):
    from dreamplace.ops.routability.l_shape_electric_potential import (
        SegmentElectricPotentialFunction,
    )

    (
        SegmentElectricPotentialFunction.last_rho_map,
        SegmentElectricPotentialFunction.last_rho_map_h,
        SegmentElectricPotentialFunction.last_rho_map_v,
    ) = snapshot


def _extract_autodmp_movable_lpos(pos, params, placedb):
    if pos.is_cuda:
        pos_cpu = pos.detach().cpu().numpy().copy()
    else:
        pos_cpu = pos.detach().numpy().copy()

    node_x = pos_cpu[: placedb.num_movable_nodes]
    node_y = pos_cpu[
        placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
    ]
    if params.cell_padding_x >= 0:
        node_x += params.cell_padding_x

    unscale_factor = 1.0 / params.scale_factor
    node_x = node_x * unscale_factor + params.shift_factor[0]
    node_y = node_y * unscale_factor + params.shift_factor[1]
    return node_x, node_y


def _write_back_autodmp_pos_to_ieda(pos, params, placedb):
    node_x, node_y = _extract_autodmp_movable_lpos(pos, params, placedb)
    placedb.write_placement_back(node_x, node_y)


def _build_gpugr_parser_cache_inputs(pos, params, placedb):
    node_x, node_y = _extract_autodmp_movable_lpos(pos, params, placedb)
    def std_round(values):
        values = np.asarray(values)
        return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))

    node_lpos = np.stack([std_round(node_x), std_round(node_y)], axis=1).astype(np.float32, copy=False)
    node_names = _get_gpugr_parser_cache_node_names(placedb)
    return node_lpos, node_names


def _get_gpugr_parser_cache_node_names(placedb):
    placedb_node_names = getattr(placedb, "node_names", [])
    if len(placedb_node_names) < placedb.num_movable_nodes:
        raise ValueError(
            "gpugr parser cache requires placedb.node_names for all movable nodes; "
            f"got {len(placedb_node_names)} names for {placedb.num_movable_nodes} movable nodes"
        )
    cache_key = (
        id(placedb_node_names),
        int(placedb.num_movable_nodes),
        int(len(placedb_node_names)),
    )
    cache = getattr(placedb, "_gpugr_parser_cache_node_names_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["names"]
    node_names = tuple(
        _normalize_placedb_name(name)
        for name in placedb_node_names[: placedb.num_movable_nodes]
    )
    setattr(
        placedb,
        "_gpugr_parser_cache_node_names_cache",
        {
            "key": cache_key,
            "names": node_names,
        },
    )
    return node_names


def _get_cached_gpugr_operator(params, placedb):
    from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR

    gpugr_op = getattr(placedb, "_autodmp_gpugr_op", None)
    if gpugr_op is None:
        gpugr_op = XplaceGPUGR(params, placedb)
        setattr(placedb, "_autodmp_gpugr_op", gpugr_op)
    return gpugr_op


def _compute_gpugr_route_grid_like_xplace(params, placedb):
    place_num_bins_y = int(getattr(placedb, "num_bins_y", getattr(params, "num_bins_y", 512)))
    if place_num_bins_y <= 0:
        place_num_bins_y = 512

    die_w = float(placedb.xh - placedb.xl)
    die_h = float(placedb.yh - placedb.yl)
    if die_w <= 0 or die_h <= 0:
        logging.warning(
            "Invalid die size for gpugr grid computation (die_w=%g, die_h=%g). Fallback to square %dx%d grid.",
            die_w,
            die_h,
            place_num_bins_y,
            place_num_bins_y,
        )
        return place_num_bins_y, place_num_bins_y

    route_size = min(512, place_num_bins_y)
    die_ratio = die_w / die_h
    route_xsize = route_size if die_ratio <= 1.0 else int(round(route_size * die_ratio))
    route_ysize = route_size if die_ratio >= 1.0 else int(round(route_size / die_ratio))
    route_xsize = max(1, route_xsize)
    route_ysize = max(1, route_ysize)

    logging.info(
        "Compute gpugr grid with Xplace rule: place_bins_y=%d die_ratio=%.4f -> route_xsize=%d route_ysize=%d",
        place_num_bins_y,
        die_ratio,
        route_xsize,
        route_ysize,
    )
    return route_xsize, route_ysize


def _normalize_placedb_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    if hasattr(name, "decode"):
        try:
            return name.decode("utf-8")
        except Exception:
            pass
    return str(name)


def _build_gpugr_topology_net_name_to_id(placedb):
    net_names = getattr(placedb, "net_names", [])
    cache_key = (id(net_names), int(len(net_names)))
    cache = getattr(placedb, "_gpugr_topology_net_name_to_id_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["mapping"]
    mapping = {}
    for net_id, net_name in enumerate(net_names):
        mapping[_normalize_placedb_name(net_name)] = int(net_id)
    setattr(
        placedb,
        "_gpugr_topology_net_name_to_id_cache",
        {
            "key": cache_key,
            "mapping": mapping,
        },
    )
    return mapping


def _build_gpugr_topology_pin_name_to_id(placedb):
    pin_names = getattr(placedb, "pin_names", [])
    cache_key = (id(pin_names), int(len(pin_names)))
    cache = getattr(placedb, "_gpugr_topology_pin_name_to_id_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]

    mapping = {}
    for pin_id, pin_name in enumerate(pin_names):
        if isinstance(pin_name, bytes):
            pin_name = pin_name.decode("utf-8")
        mapping[str(pin_name)] = int(pin_id)
    setattr(
        placedb,
        "_gpugr_topology_pin_name_to_id_cache",
        (cache_key, mapping),
    )
    return mapping


def _as_cached_gpugr_int64_array(placedb, attr_name, cache_name):
    values = getattr(placedb, attr_name, None)
    if values is None:
        return np.asarray([], dtype=np.int64)

    array = np.asarray(values)
    cache_key = (
        id(values),
        tuple(array.shape),
        str(array.dtype),
        tuple(array.strides),
        int(array.size),
    )
    cache = getattr(placedb, cache_name, None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["array"]

    int64_array = np.asarray(array, dtype=np.int64)
    setattr(
        placedb,
        cache_name,
        {
            "key": cache_key,
            "array": int64_array,
        },
    )
    return int64_array


def _build_gpugr_topology_flat_net2pin_inputs(placedb):
    return (
        _as_cached_gpugr_int64_array(
            placedb,
            "flat_net2pin_map",
            "_gpugr_topology_flat_net2pin_map_int64_cache",
        ),
        _as_cached_gpugr_int64_array(
            placedb,
            "flat_net2pin_start_map",
            "_gpugr_topology_flat_net2pin_start_map_int64_cache",
        ),
    )


def _build_gpugr_topology_pack_geometry(placedb, route_xsize, route_ysize):
    xl = float(getattr(placedb, "routing_grid_xl", placedb.xl))
    yl = float(getattr(placedb, "routing_grid_yl", placedb.yl))
    xh = float(getattr(placedb, "routing_grid_xh", placedb.xh))
    yh = float(getattr(placedb, "routing_grid_yh", placedb.yh))
    return (
        xl,
        yl,
        (xh - xl) / float(max(int(route_xsize), 1)),
        (yh - yl) / float(max(int(route_ysize), 1)),
    )


def _resample_xy_map(map_xy, target_x, target_y):
    if map_xy.shape == (target_x, target_y):
        return map_xy.contiguous()
    image_yx = map_xy.t().unsqueeze(0).unsqueeze(0)
    image_yx = F.interpolate(
        image_yx,
        size=(target_y, target_x),
        mode="bilinear",
        align_corners=False,
    )
    return image_yx.squeeze(0).squeeze(0).t().contiguous()


def _build_l_shape_supply_original_maps_from_placedb(
    placedb, params, num_bins_x, num_bins_y, device, dtype
):
    supply_original, supply_original_h, supply_original_v = create_supply_map_from_placedb(
        placedb,
        params,
        num_bins_x,
        num_bins_y,
        device=device,
        dtype=dtype,
        as_area=False,
        return_directional=True,
    )
    return {
        "supply_original": supply_original,
        "supply_original_h": supply_original_h,
        "supply_original_v": supply_original_v,
    }


def _split_gpugr_maps_by_direction(placedb, capacity_map, demand_map):
    """
    Aggregate per-layer gpugr maps into horizontal / vertical 2D maps.
    """
    if capacity_map.dim() != 3 or demand_map.dim() != 3:
        raise ValueError("Expected gpugr capacity/demand maps with shape [layers, x, y]")

    raw_h_caps = getattr(placedb, "unit_horizontal_capacities", None)
    raw_v_caps = getattr(placedb, "unit_vertical_capacities", None)
    if raw_h_caps is None or raw_v_caps is None:
        total_cap = capacity_map.sum(dim=0)
        total_dmd = demand_map.sum(dim=0)
        return total_cap, total_cap, total_dmd, total_dmd

    num_layers = min(capacity_map.size(0), len(raw_h_caps), len(raw_v_caps))
    if num_layers <= 0:
        total_cap = capacity_map.sum(dim=0)
        total_dmd = demand_map.sum(dim=0)
        return total_cap, total_cap, total_dmd, total_dmd

    capacity_map = capacity_map[:num_layers]
    demand_map = demand_map[:num_layers]

    h_caps = torch.as_tensor(raw_h_caps[:num_layers], device=capacity_map.device, dtype=capacity_map.dtype)
    v_caps = torch.as_tensor(raw_v_caps[:num_layers], device=capacity_map.device, dtype=capacity_map.dtype)
    h_mask = h_caps > v_caps
    v_mask = v_caps > h_caps

    if not h_mask.any() or not v_mask.any():
        indices = torch.arange(num_layers, device=capacity_map.device)
        first_is_h = bool((h_caps[0] >= v_caps[0]).item())
        h_mask = (indices % 2 == 0) if first_is_h else (indices % 2 == 1)
        v_mask = ~h_mask

    supply_h_xy = capacity_map[h_mask].sum(dim=0) if h_mask.any() else torch.zeros_like(capacity_map[0])
    supply_v_xy = capacity_map[v_mask].sum(dim=0) if v_mask.any() else torch.zeros_like(capacity_map[0])
    demand_h_xy = demand_map[h_mask].sum(dim=0) if h_mask.any() else torch.zeros_like(demand_map[0])
    demand_v_xy = demand_map[v_mask].sum(dim=0) if v_mask.any() else torch.zeros_like(demand_map[0])
    return supply_h_xy, supply_v_xy, demand_h_xy, demand_v_xy


def _prepare_modularity_maps_from_gpugr(placedb, gpugr_congestion_map_op, pos, eps=1e-6):
    result = getattr(gpugr_congestion_map_op, "last_result", None)
    if not isinstance(result, dict):
        raise RuntimeError("modularity_inflation_flag=1 requires gpugr last_result to be populated")
    maps = result.get("maps")
    if not isinstance(maps, dict):
        raise RuntimeError("modularity_inflation_flag=1 requires gpugr last_result['maps']")
    params = getattr(gpugr_congestion_map_op, "params", None)

    target_x = int(getattr(placedb, "num_routing_grids_x", 0))
    target_y = int(getattr(placedb, "num_routing_grids_y", 0))
    if target_x <= 0 or target_y <= 0:
        raise RuntimeError("Invalid routing grid for modularity gpugr maps: %dx%d" % (target_x, target_y))

    skip_m1_route = bool(getattr(params, "gpugr_area_adjust_skip_m1_route", 1))

    def _reduce_map(name, required=True, fallback=None, start_layer=0):
        raw_map = maps.get(name)
        if raw_map is None:
            if required:
                raise RuntimeError("modularity_inflation_flag=1 requires gpugr map '%s'" % name)
            return fallback
        raw_map = raw_map.detach().to(device=pos.device, dtype=pos.dtype)
        if raw_map.dim() == 3:
            effective_start = int(start_layer) if raw_map.size(0) > int(start_layer) else 0
            if effective_start != int(start_layer):
                logging.warning(
                    "modularity_inflation_flag requested start_layer=%d for gpugr map '%s' but only %d layers are available; fall back to layer 0",
                    int(start_layer),
                    name,
                    int(raw_map.size(0)),
                )
            xy_map = raw_map[effective_start:].sum(dim=0)
        elif raw_map.dim() == 2:
            xy_map = raw_map
        else:
            raise RuntimeError("Unsupported gpugr map '%s' dim=%d" % (name, raw_map.dim()))
        return _resample_xy_map(xy_map, target_x, target_y)

    start_layer = 1 if skip_m1_route else 0
    capacity_xy = _reduce_map("capacity_map", start_layer=start_layer)
    wire_demand_xy = _reduce_map("wire_demand_map", start_layer=start_layer)
    via_demand_xy = _reduce_map("via_demand_map", start_layer=start_layer)
    total_demand_xy = wire_demand_xy + via_demand_xy
    fix_usage_xy = _reduce_map(
        "fix_usage_map",
        required=False,
        fallback=torch.zeros_like(capacity_xy),
        start_layer=start_layer,
    )
    mov_usage_xy = _reduce_map(
        "mov_usage_map",
        required=False,
        fallback=total_demand_xy.clamp(min=0.0),
        start_layer=start_layer,
    )
    solver_capacity_xy = capacity_xy.clamp(min=float(eps))
    solver_demand_xy = total_demand_xy.clamp(min=0.0)
    gpugr_overflow_xy = _reduce_map(
        "cg_map_union_overflow",
        required=False,
        fallback=torch.zeros_like(solver_capacity_xy),
    ).clamp(min=0.0)
    solver_ratio_overflow_xy = torch.clamp(
        solver_demand_xy / solver_capacity_xy.clamp(min=float(eps)) - 1.0,
        min=0.0,
    )
    solver_diff_overflow_xy = solver_demand_xy - solver_capacity_xy

    if l_shape_log_verbose(params) >= 2:
        logging.info(
            "Prepared modularity gpugr maps: skip_m1=%d solver_demand max=%.4f capacity max=%.4f ratio_ovfl max=%.4f diff_ovfl max=%.4f gpugr_ovfl max=%.4f local max=%.4f global max=%.4f",
            int(skip_m1_route),
            float(solver_demand_xy.max().item()) if solver_demand_xy.numel() else 0.0,
            float(solver_capacity_xy.max().item()) if solver_capacity_xy.numel() else 0.0,
            float(solver_ratio_overflow_xy.max().item()) if solver_ratio_overflow_xy.numel() else 0.0,
            float(solver_diff_overflow_xy.max().item()) if solver_diff_overflow_xy.numel() else 0.0,
            float(gpugr_overflow_xy.max().item()) if gpugr_overflow_xy.numel() else 0.0,
            float(mov_usage_xy.max().item()) if mov_usage_xy.numel() else 0.0,
            float(fix_usage_xy.max().item()) if fix_usage_xy.numel() else 0.0,
        )
    return {
        "solver_demand_map": solver_demand_xy,
        "solver_capacity_map": solver_capacity_xy,
        "local_demand_map": mov_usage_xy,
        "global_demand_map": fix_usage_xy,
        "capacity_map": capacity_xy,
        "total_demand_map": total_demand_xy,
    }


def _is_modularity_inflation_enabled(params):
    return bool(getattr(params, "modularity_inflation_flag", False))


def _resolve_route_map_source(params):
    """Resolve the active route map without constructing legacy operators."""
    if getattr(params, "adjust_gpugr_area_flag", False):
        return "gpugr"
    if getattr(params, "adjust_nctugr_area_flag", False):
        logging.info(
            "adjust_nctugr_area_flag is a legacy compatibility key; "
            "using ECC/iRT EGR route map for area adjustment "
            "(NCTUgr is not invoked)"
        )
        return "irt_egr"
    return "rudy"


def _ensure_modularity_inflation_contract(params, route_map_source=None):
    if not _is_modularity_inflation_enabled(params):
        return
    if not getattr(params, "routability_opt_flag", False):
        raise RuntimeError(
            "modularity_inflation_flag=1 requires routability_opt_flag=1"
        )
    if not getattr(params, "modularity_require_gpugr_flag", 1):
        return
    if not getattr(params, "adjust_gpugr_area_flag", False):
        raise RuntimeError(
            "modularity_inflation_flag=1 requires adjust_gpugr_area_flag=1"
        )
    if getattr(params, "adjust_nctugr_area_flag", False):
        raise RuntimeError(
            "modularity_inflation_flag=1 does not support adjust_nctugr_area_flag=1"
        )
    if getattr(params, "adjust_rudy_area_flag", False):
        raise RuntimeError(
            "modularity_inflation_flag=1 does not support adjust_rudy_area_flag=1"
        )
    if route_map_source is not None and route_map_source != "gpugr":
        raise RuntimeError(
            "modularity_inflation_flag=1 requires gpugr route source, got %s"
            % route_map_source
        )


def _ensure_modularity_active_clusters(params, placedb, model, pos, num_area_adjust):
    if not _is_modularity_inflation_enabled(params):
        return
    if int(num_area_adjust) != 0:
        return
    if getattr(placedb, "modularity_active_clustering_result", None) is not None:
        return

    active_result = build_active_leiden_clusters(placedb, params, pos)
    placedb.modularity_active_clustering_result = active_result
    if hasattr(model, "data_collections") and model.data_collections is not None:
        model.data_collections.refresh_modularity_clusters_from_placedb(placedb)
    logging.info(
        "Prepared modularity active clusters at first inflation trigger: levels=%d counts=%s",
        len(active_result.cluster_ids_by_level),
        active_result.num_clusters_by_level,
    )
    if getattr(params, "modularity_plot_flag", False):
        saved_paths = plot_modularity_clusters(
            placedb=placedb,
            params=params,
            pos=pos,
            clustering_result=active_result,
            source="active",
            round_idx=int(num_area_adjust),
        )
        logging.info(
            "Saved modularity debug plots: %s",
            saved_paths,
        )
        if getattr(params, "modularity_plot_exit_flag", False):
            logging.info(
                "modularity_plot_exit_flag=1, exit(0) after saving modularity debug plots"
            )
            raise SystemExit(0)


def _sync_gpugr_route_grid_to_autodmp(params, placedb, model=None):
    route_xsize, route_ysize = _compute_gpugr_route_grid_like_xplace(params, placedb)
    old_route_xsize = getattr(placedb, "num_routing_grids_x", None)
    old_route_ysize = getattr(placedb, "num_routing_grids_y", None)
    grid_changed = old_route_xsize != route_xsize or old_route_ysize != route_ysize

    params.route_num_bins_x = route_xsize
    params.route_num_bins_y = route_ysize
    placedb.num_routing_grids_x = route_xsize
    placedb.num_routing_grids_y = route_ysize

    logging.info(
        "Sync AutoDMP routing grid to gpugr grid: route_num_bins (%s, %s) -> (%d, %d)",
        str(old_route_xsize),
        str(old_route_ysize),
        route_xsize,
        route_ysize,
    )

    if grid_changed and model is not None and hasattr(model, "op_collections") and hasattr(model, "data_collections"):
        model.op_collections.pin_utilization_map_op = model.build_pin_utilization_map(
            params,
            placedb,
            model.data_collections,
        )
        model.op_collections.adjust_node_area_op = model.build_adjust_node_area(
            params,
            placedb,
            model.data_collections,
        )
        logging.info("Rebuilt pin_utilization_map_op and adjust_node_area_op for the gpugr routing grid.")
    elif not grid_changed:
        logging.info("AutoDMP routing grid already matches the gpugr grid. Skip rebuilding related ops.")

    return route_xsize, route_ysize


def _fallback_l_shape_wire_width(placedb, route_xsize=None, route_ysize=None, fallback_wire_width=None):
    if fallback_wire_width is not None and float(fallback_wire_width) > 0:
        return float(fallback_wire_width), "external_fallback"

    if route_xsize is not None and route_ysize is not None:
        gcell_wire_width = min(
            float(placedb.xh - placedb.xl) / max(int(route_xsize), 1),
            float(placedb.yh - placedb.yl) / max(int(route_ysize), 1),
        )
        return gcell_wire_width, f"gcell_size({int(route_xsize)}x{int(route_ysize)})"

    route_x = getattr(placedb, "num_routing_grids_x", None)
    route_y = getattr(placedb, "num_routing_grids_y", None)
    if route_x is not None and route_y is not None:
        gcell_wire_width = min(
            float(placedb.xh - placedb.xl) / max(int(route_x), 1),
            float(placedb.yh - placedb.yl) / max(int(route_y), 1),
        )
        return gcell_wire_width, f"routing_grid({int(route_x)}x{int(route_y)})"

    return 0.0, "none"


def _resolve_l_shape_wire_width(placedb, route_xsize=None, route_ysize=None, fallback_wire_width=None):
    raw_widths = getattr(placedb, "min_wire_widths", None)
    widths = np.array(raw_widths, dtype=np.float32) if raw_widths is not None and len(raw_widths) > 0 else np.array([], dtype=np.float32)
    raw_h_caps = getattr(placedb, "unit_horizontal_capacities", None)
    raw_v_caps = getattr(placedb, "unit_vertical_capacities", None)
    h_caps = np.array(raw_h_caps, dtype=np.float32) if raw_h_caps is not None else np.array([], dtype=np.float32)
    v_caps = np.array(raw_v_caps, dtype=np.float32) if raw_v_caps is not None else np.array([], dtype=np.float32)

    def _finalize(value, source):
        wire_width = float(value)
        logging.info(
            "Resolved L-shape wire width: source=%s value=%.4f",
            source,
            wire_width,
        )
        return wire_width

    positive_width_mask = widths > 0
    positive_widths = widths[positive_width_mask]
    num_layers = widths.size
    if num_layers > 0 and h_caps.size >= num_layers and v_caps.size >= num_layers:
        layer_caps = h_caps[:num_layers] + v_caps[:num_layers]
        layer_mask = positive_width_mask & (layer_caps > 0)
        if np.any(layer_mask):
            return _finalize(
                np.average(widths[:num_layers][layer_mask], weights=layer_caps[layer_mask]),
                "capacity_weighted(width)",
            )
    elif positive_widths.size > 0:
        logging.warning(
            "Cannot compute capacity_weighted(width) for L-shape wire width because routing-layer capacities are unavailable. "
            "Fall back to min_width."
        )
        return _finalize(positive_widths.min(), "min_wire_width(no_capacity)")

    fallback_value, fallback_source = _fallback_l_shape_wire_width(
        placedb,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        fallback_wire_width=fallback_wire_width,
    )
    if fallback_value > 0:
        logging.warning(
            "Fallback L-shape wire width: source=%s value=%.4f",
            fallback_source,
            fallback_value,
        )
        return float(fallback_value)

    logging.warning(
        "Failed to resolve L-shape wire width. Fall back to 0.0, which may collapse segment thickness."
    )
    return 0.0


def _prepare_l_shape_inputs_from_gpugr(params, placedb, pos, model=None):
    with profile_scope(params, "gpugr_prepare.sync_route_grid", tensor=pos):
        route_xsize, route_ysize = _sync_gpugr_route_grid_to_autodmp(
            params,
            placedb,
            model=model,
        )
    l_shape_num_bins_x = int(getattr(params, "num_bins_x", placedb.num_bins_x))
    l_shape_num_bins_y = int(getattr(params, "num_bins_y", placedb.num_bins_y))
    topology_xl, topology_yl, topology_bin_size_x, topology_bin_size_y = _build_gpugr_topology_pack_geometry(
        placedb,
        route_xsize,
        route_ysize,
    )
    need_l_shape_topology_pack = use_ggr_l_shape_topology(params)
    need_same_net_topology_pack = bool(getattr(params, "soft_l_assignment", False))
    need_route_entries = (
        bool(getattr(params, "l_direction_use_gpugr", False))
        and not _should_skip_resolver_l_direction_for_soft(params)
        and not need_l_shape_topology_pack
    )
    need_route_entries = need_route_entries or bool(getattr(params, "gpugr_l_direction_save_artifacts", 0))
    need_route_entries = need_route_entries or bool(getattr(params, "l_shape_plot_flag", 0))

    parser_cache_enable = bool(getattr(params, "gpugr_parser_cache_enable", True))
    parser_cache_node_lpos = None
    parser_cache_node_names = None
    gpugr_op = _get_cached_gpugr_operator(params, placedb)
    if parser_cache_enable:
        with profile_scope(params, "gpugr_prepare.parser_cache_inputs", tensor=pos):
            parser_cache_node_lpos, parser_cache_node_names = _build_gpugr_parser_cache_inputs(
                pos,
                params,
                placedb,
            )
    parser_cache_hit_for_writeback = False
    if parser_cache_enable and parser_cache_node_names is not None and parser_cache_node_lpos is not None:
        with profile_scope(params, "gpugr_prepare.parser_cache_hit_check"):
            parser_cache_hit_for_writeback = gpugr_op.parser_cache_would_hit(
                design_name=params.design_name(),
                node_names=parser_cache_node_names,
                node_count=len(parser_cache_node_names),
            )
    with profile_scope(
        params,
        "gpugr_prepare.write_back_pos",
        tensor=pos,
        skipped=1 if parser_cache_hit_for_writeback else 0,
    ):
        if not parser_cache_hit_for_writeback:
            _write_back_autodmp_pos_to_ieda(pos, params, placedb)

    parser_cache_fallback_before_export = None
    if parser_cache_hit_for_writeback:
        def parser_cache_fallback_before_export():
            with profile_scope(params, "gpugr_prepare.parser_cache_fallback_write_back_pos", tensor=pos):
                _write_back_autodmp_pos_to_ieda(pos, params, placedb)

    with profile_scope(params, "gpugr_prepare.topology_net_name_to_id", tensor=pos):
        topology_net_name_to_id = _build_gpugr_topology_net_name_to_id(placedb)
    with profile_scope(params, "gpugr_prepare.topology_pin_name_to_id", tensor=pos):
        topology_pin_name_to_id = _build_gpugr_topology_pin_name_to_id(placedb)
    with profile_scope(params, "gpugr_prepare.topology_flat_net2pin_inputs", tensor=pos):
        topology_flat_net2pin_map, topology_flat_net2pin_start_map = (
            _build_gpugr_topology_flat_net2pin_inputs(placedb)
        )
    with profile_scope(
        params,
        "gpugr_prepare.run_gpugr",
        tensor=pos,
        route_grid=f"{route_xsize}x{route_ysize}",
    ):
        result = gpugr_op.run_gpugr(
            out_dir=os.path.join(params.result_dir, "gpugr_l_shape"),
            design_name=params.design_name(),
            gpu=getattr(params, "gpu_id", 0),
            threads=params.num_threads,
            route_xsize=route_xsize,
            route_ysize=route_ysize,
            rrr_iters=int(getattr(params, "gpugr_l_direction_rrr_iters", 0)),
            skip_m1_route=True,
            # Keep the DEF for every L-shape GPUGR evaluation so the selected
            # intermediate placement can be replayed by Innovus after the run.
            keep_temp_def=True,
            save_artifacts=bool(getattr(params, "gpugr_l_direction_save_artifacts", 0)),
            include_route_entries=need_route_entries,
            include_topology_pack=need_same_net_topology_pack,
            include_l_shape_topology_pack=need_l_shape_topology_pack,
            topology_net_name_to_id=topology_net_name_to_id,
            topology_pin_name_to_id=topology_pin_name_to_id,
            topology_flat_net2pin_map=topology_flat_net2pin_map,
            topology_flat_net2pin_start_map=topology_flat_net2pin_start_map,
            topology_num_pins=len(getattr(placedb, "pin_names", [])),
            topology_num_nets=int(getattr(placedb, "num_nets", 0)),
            topology_max_gap=1,
            topology_xl=topology_xl,
            topology_yl=topology_yl,
            topology_bin_size_x=topology_bin_size_x,
            topology_bin_size_y=topology_bin_size_y,
            parser_cache_enable=parser_cache_enable,
            parser_cache_node_lpos=parser_cache_node_lpos,
            parser_cache_node_names=parser_cache_node_names,
            parser_cache_fallback_before_export=parser_cache_fallback_before_export,
            profile_enabled=bool(getattr(params, "l_shape_profile_flag", False)),
            profile_prefix="gpugr_prepare.run_gpugr",
            backend=getattr(params, "gpugr_backend", "auto"),
        )

    maps = result["maps"]
    metrics = result["metrics"]
    route_entries = result.get("route_entries", [])
    gpugr_topology_stats = result.get("same_net_topology_stats", {})
    total_entries = int(
        gpugr_topology_stats.get(
            "total_entry_count",
            sum(len(net.get("entries", [])) for net in route_entries),
        )
    )
    route_nets = int(gpugr_topology_stats.get("num_route_entry_nets", len(route_entries)))
    with profile_scope(
        params,
        "gpugr_prepare.same_net_topology_cache",
        tensor=pos,
        route_nets=route_nets,
        route_entries=total_entries,
        skipped=0 if need_same_net_topology_pack else 1,
    ):
        if need_same_net_topology_pack:
            same_net_topo_cache, same_net_topo_stats = build_same_net_topology_cache(
                route_entries,
                placedb,
                profile_enabled=params,
                prebuilt_cache=result.get("same_net_topology_cache", {}),
                prebuilt_stats=gpugr_topology_stats,
            )
        else:
            same_net_topo_cache, same_net_topo_stats = None, {}

    with profile_scope(params, "gpugr_prepare.map_to_device", tensor=pos):
        capacity_map = maps["capacity_map"].detach().to(device=pos.device, dtype=pos.dtype)
        raw_wire_layer_map = maps["raw_wire_demand_map"].detach().to(device=pos.device, dtype=pos.dtype)
        fix_usage_layer_map = maps["fix_usage_map"].detach().to(device=pos.device, dtype=pos.dtype)
        mov_usage_layer_map = maps["mov_usage_map"].detach().to(device=pos.device, dtype=pos.dtype)
        total_demand_map = (maps["wire_demand_map"] + maps["via_demand_map"]).detach().to(
            device=pos.device, dtype=pos.dtype
        )
    supply_xy = capacity_map.sum(dim=0)
    raw_wire_xy = raw_wire_layer_map.sum(dim=0)
    fix_usage_xy = fix_usage_layer_map.sum(dim=0)
    mov_usage_xy = mov_usage_layer_map.sum(dim=0)
    demand_xy = total_demand_map.sum(dim=0)
    _, _, raw_wire_h_xy, raw_wire_v_xy = _split_gpugr_maps_by_direction(
        placedb,
        capacity_map,
        raw_wire_layer_map,
    )
    supply_h_xy, supply_v_xy, demand_h_xy, demand_v_xy = _split_gpugr_maps_by_direction(
        placedb,
        capacity_map,
        total_demand_map,
    )
    _, _, fix_usage_h_xy, fix_usage_v_xy = _split_gpugr_maps_by_direction(
        placedb,
        capacity_map,
        fix_usage_layer_map,
    )
    _, _, mov_usage_h_xy, mov_usage_v_xy = _split_gpugr_maps_by_direction(
        placedb,
        capacity_map,
        mov_usage_layer_map,
    )
    with profile_scope(
        params,
        "gpugr_prepare.split_resample_maps",
        tensor=pos,
        l_shape_bins=f"{l_shape_num_bins_x}x{l_shape_num_bins_y}",
    ):
        supply_map = _resample_xy_map(supply_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        raw_wire_demand_map = _resample_xy_map(raw_wire_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        fix_usage_map = _resample_xy_map(fix_usage_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        mov_usage_map = _resample_xy_map(mov_usage_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        demand_map = _resample_xy_map(demand_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        supply_map_h = _resample_xy_map(supply_h_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        supply_map_v = _resample_xy_map(supply_v_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        raw_wire_demand_map_h = _resample_xy_map(raw_wire_h_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        raw_wire_demand_map_v = _resample_xy_map(raw_wire_v_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        fix_usage_map_h = _resample_xy_map(fix_usage_h_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        fix_usage_map_v = _resample_xy_map(fix_usage_v_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        mov_usage_map_h = _resample_xy_map(mov_usage_h_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        mov_usage_map_v = _resample_xy_map(mov_usage_v_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        demand_map_h = _resample_xy_map(demand_h_xy, l_shape_num_bins_x, l_shape_num_bins_y)
        demand_map_v = _resample_xy_map(demand_v_xy, l_shape_num_bins_x, l_shape_num_bins_y)
    supply_original_maps = {
        "supply_original": supply_map.clone(),
        "supply_original_h": supply_map_h.clone(),
        "supply_original_v": supply_map_v.clone(),
    }
    wire_width = _resolve_l_shape_wire_width(
        placedb,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
    )

    if l_shape_log_verbose(params) >= 1:
        logging.info(
            "Prepared gpugr L-shape inputs: route_grid=%dx%d l_shape_bins=%dx%d nets=%d entries=%d "
            "supply[min=%.3f max=%.3f mean=%.3f] demand[min=%.3f max=%.3f mean=%.3f] "
            "CHmax/top1/bin=%.1f%%/%.1f%%/%.2f%% CVmax/top1/bin=%.1f%%/%.1f%%/%.2f%% "
            "#OvflNets=%d EstShorts=%.0f",
            route_xsize,
            route_ysize,
            l_shape_num_bins_x,
            l_shape_num_bins_y,
            route_nets,
            total_entries,
            supply_map.min().item(),
            supply_map.max().item(),
            supply_map.mean().item(),
            demand_map.min().item(),
            demand_map.max().item(),
            demand_map.mean().item(),
            metrics["cg_map_h_raw_max"] * 100.0,
            metrics["cg_map_h_raw_top1pct_mean"] * 100.0,
            metrics["cg_map_h_raw_overflow_bin_ratio"] * 100.0,
            metrics["cg_map_v_raw_max"] * 100.0,
            metrics["cg_map_v_raw_top1pct_mean"] * 100.0,
            metrics["cg_map_v_raw_overflow_bin_ratio"] * 100.0,
            metrics["num_overflow_nets"],
            metrics["gr_est_shorts"],
        )
    if l_shape_log_verbose(params) >= 2:
        logging.info(
            "Use gpugr capacity_map as L-shape supply_original: total_diff_vs_supply=%.3e h_diff=%.3e v_diff=%.3e",
            float((supply_original_maps["supply_original"] - supply_map).abs().sum().item()),
            float((supply_original_maps["supply_original_h"] - supply_map_h).abs().sum().item()),
            float((supply_original_maps["supply_original_v"] - supply_map_v).abs().sum().item()),
        )
    return {
        "supply_map": supply_map,
        "demand_map": demand_map,
        "raw_wire_demand_map": raw_wire_demand_map,
        "supply_map_h": supply_map_h,
        "supply_map_v": supply_map_v,
        "demand_map_h": demand_map_h,
        "demand_map_v": demand_map_v,
        "raw_wire_demand_map_h": raw_wire_demand_map_h,
        "raw_wire_demand_map_v": raw_wire_demand_map_v,
        "fix_usage_map": fix_usage_map,
        "fix_usage_map_h": fix_usage_map_h,
        "fix_usage_map_v": fix_usage_map_v,
        "mov_usage_map": mov_usage_map,
        "mov_usage_map_h": mov_usage_map_h,
        "mov_usage_map_v": mov_usage_map_v,
        "wire_width": wire_width,
        "route_entries": route_entries,
        "metrics": metrics,
        "route_xsize": route_xsize,
        "route_ysize": route_ysize,
        "num_bins_x": l_shape_num_bins_x,
        "num_bins_y": l_shape_num_bins_y,
        "same_net_topo_cache": same_net_topo_cache,
        "same_net_topo_stats": same_net_topo_stats,
        "l_shape_topology_pack": result.get("l_shape_topology_pack"),
        **supply_original_maps,
    }


def _run_gpugr_final_eval(params, placedb, pos):
    if not bool(getattr(params, "gpugr_final_eval_flag", 0)):
        return None

    _write_back_autodmp_pos_to_ieda(pos, params, placedb)

    override_xsize = int(getattr(params, "gpugr_final_eval_route_xsize", 0))
    override_ysize = int(getattr(params, "gpugr_final_eval_route_ysize", 0))
    if override_xsize > 0 and override_ysize > 0:
        route_xsize = override_xsize
        route_ysize = override_ysize
        logging.info(
            "Run final gpugr eval with user grid override: route_xsize=%d route_ysize=%d",
            route_xsize,
            route_ysize,
        )
    else:
        route_xsize, route_ysize = _compute_gpugr_route_grid_like_xplace(params, placedb)

    out_dir = os.path.join(params.result_dir, "gpugr_final_eval")
    logging.info(
        "Run final gpugr eval after placement: route_grid=%dx%d rrr_iters=%d skip_m1_route=%s",
        route_xsize,
        route_ysize,
        int(getattr(params, "gpugr_final_eval_rrr_iters", 1)),
        str(bool(getattr(params, "gpugr_final_eval_skip_m1_route", 1))),
    )
    gpugr_op = _get_cached_gpugr_operator(params, placedb)
    result = gpugr_op.run_gpugr(
        out_dir=out_dir,
        design_name=params.design_name(),
        gpu=getattr(params, "gpu_id", 0),
        threads=params.num_threads,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        rrr_iters=int(getattr(params, "gpugr_final_eval_rrr_iters", 1)),
        skip_m1_route=bool(getattr(params, "gpugr_final_eval_skip_m1_route", 1)),
        verbose_parser_log=bool(getattr(params, "gpugr_final_eval_verbose_parser_log", 0)),
        cpp_log_level=int(getattr(params, "gpugr_final_eval_cpp_log_level", 2)),
        keep_temp_def=bool(getattr(params, "gpugr_final_eval_keep_temp_def", 0)),
        save_artifacts=bool(getattr(params, "gpugr_final_eval_save_artifacts", 1)),
        include_route_entries=False,
        backend=getattr(params, "gpugr_backend", "auto"),
    )
    metrics = result["metrics"]
    logging.info(
        "Final gpugr eval finished. #OvflNets=%d GR_WL=%.0f GR_Vias=%.0f EstShorts=%.0f elapsed=%.3fs "
        "CHmax/mean/top1/bin=%.1f%%/%.1f%%/%.1f%%/%.2f%% "
        "CVmax/mean/top1/bin=%.1f%%/%.1f%%/%.1f%%/%.2f%%",
        metrics["num_overflow_nets"],
        metrics["gr_wirelength"],
        metrics["gr_num_vias"],
        metrics["gr_est_shorts"],
        metrics["elapsed_sec"],
        metrics["cg_map_h_raw_max"] * 100.0,
        metrics["cg_map_h_raw_mean"] * 100.0,
        metrics["cg_map_h_raw_top1pct_mean"] * 100.0,
        metrics["cg_map_h_raw_overflow_bin_ratio"] * 100.0,
        metrics["cg_map_v_raw_max"] * 100.0,
        metrics["cg_map_v_raw_mean"] * 100.0,
        metrics["cg_map_v_raw_top1pct_mean"] * 100.0,
        metrics["cg_map_v_raw_overflow_bin_ratio"] * 100.0,
    )
    return result


def _run_post_legalization_adaptive_padding(
    params,
    placedb,
    pos,
    model,
):
    if not bool(
        getattr(params, "post_legalization_adaptive_padding_flag", 0)
    ):
        return pos, None
    if not bool(getattr(params, "legalize_flag", 0)):
        raise ValueError(
            "post_legalization_adaptive_padding_flag requires legalize_flag=1"
        )
    if len(placedb.regions) > 0:
        raise ValueError(
            "post-legalization adaptive padding does not support fence regions"
        )

    _write_back_autodmp_pos_to_ieda(pos, params, placedb)
    route_xsize, route_ysize = _compute_gpugr_route_grid_like_xplace(
        params, placedb
    )
    rrr_iters = int(getattr(params, "post_legalization_padding_rrr_iters", 0))
    out_dir = os.path.join(
        params.result_dir, "gpugr_post_legalization_padding"
    )
    logging.info(
        "Run gpugr for post-legalization adaptive padding: "
        "route_grid=%dx%d rrr_iters=%d",
        route_xsize,
        route_ysize,
        rrr_iters,
    )
    gpugr_op = _get_cached_gpugr_operator(placedb)
    result = gpugr_op.run_gpugr(
        out_dir=out_dir,
        design_name=params.design_name(),
        gpu=getattr(params, "gpu_id", 0),
        threads=params.num_threads,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        rrr_iters=rrr_iters,
        skip_m1_route=bool(
            getattr(params, "post_legalization_padding_skip_m1_route", 1)
        ),
        verbose_parser_log=False,
        cpp_log_level=int(getattr(params, "gpugr_final_eval_cpp_log_level", 2)),
        keep_temp_def=False,
        save_artifacts=bool(
            getattr(params, "post_legalization_padding_save_artifacts", 0)
        ),
        include_route_entries=False,
        backend=getattr(params, "gpugr_backend", "auto"),
    )
    maps = result["maps"]
    metrics = result["metrics"]
    overflow_xy = build_smoothed_overflow_map(
        maps["cg_map_h_overflow"],
        maps["cg_map_v_overflow"],
        smooth_kernel=int(
            getattr(params, "post_legalization_padding_smooth_kernel", 3)
        ),
    )
    scores = score_cells_from_overflow(
        pos=pos,
        node_size_x=model.data_collections.node_size_x,
        node_size_y=model.data_collections.node_size_y,
        num_nodes=placedb.num_nodes,
        num_movable_nodes=placedb.num_movable_nodes,
        overflow_xy=overflow_xy,
        grid_xl=placedb.xl,
        grid_yl=placedb.yl,
        grid_xh=placedb.xh,
        grid_yh=placedb.yh,
    )
    eligible_mask = np.ones(placedb.num_movable_nodes, dtype=bool)
    movable_macro_mask = getattr(
        model.data_collections, "movable_macro_mask", None
    )
    if movable_macro_mask is not None:
        eligible_mask &= ~movable_macro_mask.detach().cpu().numpy().astype(bool)
    plan = allocate_padding_sites(
        scores=scores,
        pos=pos,
        node_size_x=model.data_collections.node_size_x,
        node_size_y=model.data_collections.node_size_y,
        num_nodes=placedb.num_nodes,
        num_movable_nodes=placedb.num_movable_nodes,
        num_physical_nodes=int(
            getattr(
                placedb,
                "num_physical_nodes",
                placedb.num_nodes - placedb.num_filler_nodes,
            )
        ),
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        site_width=placedb.site_width,
        row_height=placedb.row_height,
        hot_cell_ratio=float(
            getattr(params, "post_legalization_padding_hot_cell_ratio", 0.2)
        ),
        row_free_ratio=float(
            getattr(params, "post_legalization_padding_row_free_ratio", 0.5)
        ),
        max_padding_sites=int(
            getattr(params, "post_legalization_padding_max_sites", 1)
        ),
        eligible_mask=eligible_mask,
    )
    logging.info(
        "Post-legalization adaptive padding plan: "
        "eligible=%d positive_score=%d requested_hot=%d allocated=%d "
        "added_sites=%d free_sites=%d budget_sites=%d "
        "max_score=%.6g min_allocated_score=%.6g "
        "gpugr_overflow_nets=%d gpugr_est_shorts=%.6g",
        plan["eligible_count"],
        plan["positive_score_count"],
        plan["requested_hot_count"],
        plan["allocated_count"],
        plan["total_added_sites"],
        plan["total_free_sites"],
        plan["total_budget_sites"],
        plan["max_score"],
        plan["min_allocated_score"],
        metrics["num_overflow_nets"],
        metrics["gr_est_shorts"],
    )

    rail_boxes = model.data_collections.m2_pg_rail_density_boxes
    rail_before = compute_cell_box_overlap_stats(
        pos,
        model.data_collections.node_size_x,
        model.data_collections.node_size_y,
        rail_boxes,
        placedb.num_nodes,
        placedb.num_movable_nodes,
    )
    hpwl_before = float(model.op_collections.hpwl_op(pos).item())
    padded_result, legalize_stats = model.run_adaptive_padding_legalization(
        placedb=placedb,
        pos=pos,
        padding_sites=plan["padding_sites"],
        scores=plan["scores"],
        max_retries=int(
            getattr(params, "post_legalization_padding_max_retries", 4)
        ),
    )
    hpwl_after = float(model.op_collections.hpwl_op(padded_result).item())
    rail_after = compute_cell_box_overlap_stats(
        padded_result,
        model.data_collections.node_size_x,
        model.data_collections.node_size_y,
        rail_boxes,
        placedb.num_nodes,
        placedb.num_movable_nodes,
    )
    logging.info(
        "Post-legalization adaptive padding result: "
        "requested=%d used=%d attempts=%d rollback=%d "
        "padded_legal=%d physical_legal=%d greedy_fallback=%d "
        "moved=%d displacement_total=%.6g displacement_max=%.6g "
        "HPWL=%.6g->%.6g delta=%.6g "
        "M2_overlap_count=%d->%d M2_overlap_area=%.6g->%.6g",
        legalize_stats["requested_count"],
        legalize_stats["used_count"],
        legalize_stats["attempts"],
        int(legalize_stats["rollback"]),
        int(legalize_stats["padded_legal"]),
        int(legalize_stats["physical_legal"]),
        int(legalize_stats["greedy_fallback"]),
        legalize_stats["moved_count"],
        legalize_stats["total_displacement"],
        legalize_stats["max_displacement"],
        hpwl_before,
        hpwl_after,
        hpwl_after - hpwl_before,
        rail_before["overlap_count"],
        rail_after["overlap_count"],
        rail_before["overlap_area"],
        rail_after["overlap_area"],
    )
    telemetry = {
        "allocated_count": plan["allocated_count"],
        "total_added_sites": plan["total_added_sites"],
        "used_count": legalize_stats["used_count"],
        "attempts": legalize_stats["attempts"],
        "rollback": int(legalize_stats["rollback"]),
        "moved_count": legalize_stats["moved_count"],
        "max_displacement": legalize_stats["max_displacement"],
        "hpwl_delta": hpwl_after - hpwl_before,
        "m2_overlap_count_before": rail_before["overlap_count"],
        "m2_overlap_count_after": rail_after["overlap_count"],
        "m2_overlap_area_before": rail_before["overlap_area"],
        "m2_overlap_area_after": rail_after["overlap_area"],
        "gpugr_overflow_nets_before": metrics["num_overflow_nets"],
        "gpugr_est_shorts_before": metrics["gr_est_shorts"],
    }
    return padded_result, telemetry


def _prepare_l_shape_inputs_from_egr(params, placedb, pos, model):
    model.op_collections.irt_egr_congestion_map_op(
        pos, stage="egr2D", resolve_congestion="low"
    )
    l_shape_num_bins_x = int(getattr(params, "num_bins_x", placedb.num_bins_x))
    l_shape_num_bins_y = int(getattr(params, "num_bins_y", placedb.num_bins_y))
    egr_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(params.result_dir))),
        "iEDA/data/rt/rt_temp_directory/early_router/",
    )
    supply_map, demand_map, wire_width = create_supply_and_demand_maps_from_egr(
        egr_dir=egr_dir,
        placedb=placedb,
        params=params,
        num_bins_x=l_shape_num_bins_x,
        num_bins_y=l_shape_num_bins_y,
        layer='planar',
        normalize_supply=False,
        normalize_demand=False,
        return_wire_width=True,
    )
    supply_map_h, supply_map_v, demand_map_h, demand_map_v, _ = create_directional_supply_and_demand_maps_from_egr(
        egr_dir=egr_dir,
        placedb=placedb,
        params=params,
        num_bins_x=l_shape_num_bins_x,
        num_bins_y=l_shape_num_bins_y,
        device=pos.device,
        dtype=pos.dtype,
        normalize_supply=False,
        normalize_demand=False,
        return_wire_width=True,
    )
    wire_width = _resolve_l_shape_wire_width(
        placedb,
        fallback_wire_width=wire_width,
    )
    supply_original_maps = _build_l_shape_supply_original_maps_from_placedb(
        placedb,
        params,
        l_shape_num_bins_x,
        l_shape_num_bins_y,
        device=pos.device,
        dtype=pos.dtype,
    )
    return {
        "supply_map": supply_map,
        "demand_map": demand_map,
        "supply_map_h": supply_map_h,
        "supply_map_v": supply_map_v,
        "demand_map_h": demand_map_h,
        "demand_map_v": demand_map_v,
        "wire_width": wire_width,
        "num_bins_x": l_shape_num_bins_x,
        "num_bins_y": l_shape_num_bins_y,
        **supply_original_maps,
    }


def _run_gpugr_before_first_area_adjust_and_exit(params, placedb, pos, num_area_adjust, model=None):
    if not getattr(params, "gpugr_first_inflation_exit", False):
        return
    if num_area_adjust != 0:
        return

    logging.info(
        "Run gpugr operator before the first AutoDMP area-adjust round and exit after it finishes."
    )
    _write_back_autodmp_pos_to_ieda(pos, params, placedb)
    route_xsize, route_ysize = _sync_gpugr_route_grid_to_autodmp(params, placedb, model=model)

    out_dir = os.path.join(params.result_dir, "gpugr_first_inflation")
    gpugr_op = _get_cached_gpugr_operator(params, placedb)
    result = gpugr_op.run_gpugr(
        out_dir=out_dir,
        design_name=params.design_name(),
        gpu=getattr(params, "gpu_id", 0),
        threads=params.num_threads,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        rrr_iters=getattr(params, "gpugr_first_inflation_rrr_iters", 0),
        skip_m1_route=True,
        verbose_parser_log=bool(getattr(params, "gpugr_first_inflation_verbose_parser_log", 0)),
        cpp_log_level=int(getattr(params, "gpugr_first_inflation_cpp_log_level", 2)),
        keep_temp_def=True,
        save_artifacts=True,
        backend=getattr(params, "gpugr_backend", "auto"),
    )
    metrics = result["metrics"]
    logging.info(
        "gpugr finished before first area-adjust. #OvflNets=%d GR_WL=%.0f GR_Vias=%.0f EstShorts=%.0f"
        % (
            metrics["num_overflow_nets"],
            metrics["gr_wirelength"],
            metrics["gr_num_vias"],
            metrics["gr_est_shorts"],
        )
    )
    raise SystemExit(0)


def _get_default_egr_guide_path(params):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(params.result_dir))),
        "iEDA/data/rt/rt_temp_directory/early_router/route_planar.guide",
    )


def _should_skip_resolver_l_direction_for_soft(params):
    return bool(getattr(params, "soft_l_assignment", False)) and not bool(
        getattr(params, "soft_l_use_resolver_prior", True)
    )


def _resolve_l_directions_for_l_shape(
    params,
    placedb,
    pos,
    steiner_topo_op,
    gpugr_route_entries=None,
    gpugr_metrics=None,
    gpugr_route_grid=None,
):
    if steiner_topo_op.l_direction_resolver is None:
        steiner_topo_op.init_l_direction_resolver(placedb, params)

    if getattr(params, "l_direction_use_gpugr", False):
        if gpugr_route_entries is None:
            gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(params, placedb, pos)
            route_entries = gpugr_inputs["route_entries"]
            metrics = gpugr_inputs["metrics"]
            route_xsize = gpugr_inputs["route_xsize"]
            route_ysize = gpugr_inputs["route_ysize"]
        else:
            route_entries = gpugr_route_entries
            metrics = gpugr_metrics or {"num_overflow_nets": -1, "gr_est_shorts": -1.0}
            if gpugr_route_grid is None:
                route_xsize, route_ysize = _compute_gpugr_route_grid_like_xplace(params, placedb)
            else:
                route_xsize, route_ysize = gpugr_route_grid
        total_entries = sum(len(net.get("entries", [])) for net in route_entries)
        logging.info(
            "Resolve L directions from gpugr route entries: grid=%dx%d nets=%d entries=%d #OvflNets=%d EstShorts=%.0f",
            route_xsize,
            route_ysize,
            len(route_entries),
            total_entries,
            metrics["num_overflow_nets"],
            metrics["gr_est_shorts"],
        )
        with profile_scope(
            params,
            "l_direction.resolve_from_gpugr",
            tensor=pos,
            route_nets=len(route_entries),
            route_entries=total_entries,
        ):
            return steiner_topo_op.resolve_l_directions_from_gpugr(route_entries)

    egr_guide_path = getattr(params, "egr_guide_path", _get_default_egr_guide_path(params))
    logging.info("Resolve L directions from EGR guide: %s", egr_guide_path)
    with profile_scope(params, "l_direction.resolve_from_egr", tensor=pos):
        return steiner_topo_op.resolve_l_directions_from_egr(egr_guide_path)


def _load_l_shape_topology_pack_from_gpugr(
    params,
    pos,
    pin_pos_op,
    steiner_topo_op,
    data_collections,
    l_shape_inputs,
    profile_name,
    iteration,
):
    topology_pack = None if l_shape_inputs is None else l_shape_inputs.get("l_shape_topology_pack")
    if topology_pack is None:
        raise RuntimeError(
            "l_shape_use_ggr_topology requires GGR l_shape_topology_pack output"
        )
    with profile_scope(params, profile_name, tensor=pos, iteration=iteration):
        with torch.no_grad():
            pin_pos = pin_pos_op(pos)
            if pin_pos.is_cuda:
                pin_pos = pin_pos.cpu()
            data_collections.net_flat_topo_sort, data_collections.net_flat_topo_sort_start, \
                data_collections.pin_fa, data_collections.flat_pin_to, data_collections.flat_pin_to_start, \
                data_collections.flat_pin_from = steiner_topo_op.load_ggr_topology_pack(
                    topology_pack,
                    pin_pos,
                )
    if steiner_topo_op.edge_l_directions is None:
        raise RuntimeError(
            "l_shape_use_ggr_topology requires edge_l_directions in GGR topology pack"
        )
    return steiner_topo_op.edge_l_directions


class NonLinearPlace(BasicPlace.BasicPlace):
    """
    @brief Nonlinear placement engine.
    It takes parameters and placement database and runs placement flow.
    """

    def plot_steiner_and_guide(self, guide_path, steiner_topo_op, flat_pin_from, flat_pin_to, output_path, params, placedb, use_l_direction=True):
        """
        绘制Steiner树与EGR Route Guide的对比图
        
        Args:
            use_l_direction: 是否使用解析的L方向信息绘制（默认True）
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
        from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo

        logging.info(f"Plotting Steiner tree and Route Guide to {output_path}")
        
        # 1. Read Route Guide
        guide_wires = []
        guide_length = 0.0
        if os.path.exists(guide_path):
            with open(guide_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0] == 'wire':
                        try:
                            # wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer
                            x1, y1, x2, y2 = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                            guide_wires.append([(x1, y1), (x2, y2)])
                            guide_length += abs(x1 - x2) + abs(y1 - y2)
                        except (ValueError, IndexError): pass
        else:
            logging.warning(f"Route guide file not found: {guide_path}")

        # 2. Get Steiner Segments
        newx = steiner_topo_op.newx.detach().cpu().numpy()
        newy = steiner_topo_op.newy.detach().cpu().numpy()
        
        # Unscale coordinates
        if hasattr(params, 'scale_factor') and hasattr(params, 'shift_factor'):
            unscale_factor = 1.0 / params.scale_factor
            newx = newx * unscale_factor + params.shift_factor[0]
            newy = newy * unscale_factor + params.shift_factor[1]

        newx /= placedb.dbu
        newy /= placedb.dbu
        
        from_idx_raw = flat_pin_from.detach().cpu().numpy()
        to_idx_raw = flat_pin_to.detach().cpu().numpy()
        
        # 获取L方向信息
        has_l_directions = use_l_direction and steiner_topo_op.edge_l_directions is not None
        if has_l_directions:
            l_directions = steiner_topo_op.edge_l_directions.cpu().numpy()
            logging.info("Using resolved L directions for plotting")
        else:
            l_directions = None
            logging.info("Using default L direction (H_FIRST) for plotting")
        
        steiner_segments = []
        steiner_length = 0.0
        
        # 统计L方向使用情况
        l_stats = {'upper_L': 0, 'lower_L': 0, 'straight': 0, 'default': 0}

        for edge_idx in range(len(from_idx_raw)):
            idx1 = from_idx_raw[edge_idx]
            idx2 = to_idx_raw[edge_idx]
            
            if idx1 == -1 or idx2 == -1:
                continue
            if idx1 >= len(newx) or idx2 >= len(newx):
                continue
                
            p1 = (newx[idx1], newy[idx1])
            p2 = (newx[idx2], newy[idx2])
            
            # Check if horizontal or vertical (straight line)
            if abs(p1[0] - p2[0]) < 1e-5 or abs(p1[1] - p2[1]) < 1e-5:
                steiner_segments.append([p1, p2])
                l_stats['straight'] += 1
            else:
                # 需要画L形
                if has_l_directions and edge_idx < len(l_directions):
                    l_dir = l_directions[edge_idx]
                else:
                    l_dir = SteinerTopo.H_FIRST  # 默认使用上L
                    l_stats['default'] += 1
                
                if l_dir == SteinerTopo.H_FIRST:
                    # 上L: 先水平后垂直, 拐点在 (x2, y1)
                    p_mid = (p2[0], p1[1])
                    l_stats['upper_L'] += 1
                elif l_dir == SteinerTopo.V_FIRST:
                    # 下L: 先垂直后水平, 拐点在 (x1, y2)
                    p_mid = (p1[0], p2[1])
                    l_stats['lower_L'] += 1
                else:
                    # UNKNOWN或其他，默认使用上L
                    p_mid = (p2[0], p1[1])
                    l_stats['default'] += 1
                
                steiner_segments.append([p1, p_mid])
                steiner_segments.append([p_mid, p2])

            steiner_length += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        logging.info(f"L direction stats: upper_L={l_stats['upper_L']}, lower_L={l_stats['lower_L']}, "
                    f"straight={l_stats['straight']}, default={l_stats['default']}")

        # 3. Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if guide_wires:
            lc_guide = LineCollection(guide_wires, colors='blue', linewidths=0.5, label=f'Route Guide (L={guide_length:.2f})', alpha=0.5)
            ax.add_collection(lc_guide)
            
        if steiner_segments:
            lc_steiner = LineCollection(steiner_segments, colors='red', linewidths=0.5, label=f'Steiner Tree (L={steiner_length:.2f})', alpha=0.5)
            ax.add_collection(lc_steiner)
            
        ax.autoscale()
        ax.set_aspect('equal')
        plt.legend(loc='upper right')
        
        l_info = "with EGR L-direction" if has_l_directions else "default L-direction"
        plt.title(f"Route Guide vs Steiner Tree ({l_info})\nGuide Length: {guide_length:.2f}, Steiner Length: {steiner_length:.2f}")
        plt.xlabel("X (microns)")
        plt.ylabel("Y (microns)")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"Plot saved. Guide Length: {guide_length}, Steiner Length: {steiner_length}")

    def plot_egr_steiner_and_guide(self, guide_path, steiner_topo_op, pin_pos, output_path, params, placedb):
        """
        绘制方案二：使用EGR构建的Steiner树与EGR Route Guide的对比图
        
        Args:
            guide_path: EGR route guide文件路径
            steiner_topo_op: SteinerTopo操作对象（包含EGR构建结果）
            pin_pos: pin坐标tensor
            output_path: 输出图片路径
            params: 参数对象
            placedb: placement database
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
        from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo

        logging.info(f"Plotting EGR-built Steiner tree and Route Guide to {output_path}")
        
        # 检查是否有EGR构建的边列表
        if not hasattr(steiner_topo_op, 'egr_flat_pin_from') or steiner_topo_op.egr_flat_pin_from is None:
            logging.error("No EGR-built edges found. Call rebuild_tree_from_egr first.")
            return
        
        # 1. Read Route Guide
        guide_wires = []
        guide_length = 0.0
        if os.path.exists(guide_path):
            with open(guide_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0] == 'wire':
                        try:
                            x1, y1, x2, y2 = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                            guide_wires.append([(x1, y1), (x2, y2)])
                            guide_length += abs(x1 - x2) + abs(y1 - y2)
                        except (ValueError, IndexError): pass
        else:
            logging.warning(f"Route guide file not found: {guide_path}")

        # 2. 计算顶点坐标（使用EGR的relate关系）
        # 通过forward计算newx, newy
        newx, newy = steiner_topo_op(pin_pos)
        newx = newx.detach().cpu().numpy()
        newy = newy.detach().cpu().numpy()
        
        # Unscale coordinates
        if hasattr(params, 'scale_factor') and hasattr(params, 'shift_factor'):
            unscale_factor = 1.0 / params.scale_factor
            newx = newx * unscale_factor + params.shift_factor[0]
            newy = newy * unscale_factor + params.shift_factor[1]

        newx /= placedb.dbu
        newy /= placedb.dbu
        
        # 3. 获取EGR构建的边列表
        from_idx_raw = steiner_topo_op.egr_flat_pin_from.cpu().numpy()
        to_idx_raw = steiner_topo_op.egr_flat_pin_to.cpu().numpy()
        l_directions = steiner_topo_op.egr_edge_l_directions.cpu().numpy()
        
        logging.info(f"EGR edges: {len(from_idx_raw)}, L directions: {len(l_directions)}")
        
        steiner_segments = []
        steiner_length = 0.0
        
        # 统计L方向使用情况
        l_stats = {'upper_L': 0, 'lower_L': 0, 'straight': 0, 'unknown': 0}

        for edge_idx in range(len(from_idx_raw)):
            idx1 = from_idx_raw[edge_idx]
            idx2 = to_idx_raw[edge_idx]
            
            if idx1 < 0 or idx2 < 0:
                continue
            if idx1 >= len(newx) or idx2 >= len(newx):
                logging.warning(f"Edge {edge_idx}: index out of range ({idx1}, {idx2}) >= {len(newx)}")
                continue
                
            p1 = (newx[idx1], newy[idx1])
            p2 = (newx[idx2], newy[idx2])
            
            l_dir = l_directions[edge_idx] if edge_idx < len(l_directions) else -1
            
            # Check if horizontal or vertical (straight line)
            if abs(p1[0] - p2[0]) < 1e-5 or abs(p1[1] - p2[1]) < 1e-5:
                steiner_segments.append([p1, p2])
                l_stats['straight'] += 1
            else:
                # 需要画L形
                if l_dir == SteinerTopo.H_FIRST:
                    # 上L: 先水平后垂直, 拐点在 (x2, y1)
                    p_mid = (p2[0], p1[1])
                    l_stats['upper_L'] += 1
                elif l_dir == SteinerTopo.V_FIRST:
                    # 下L: 先垂直后水平, 拐点在 (x1, y2)
                    p_mid = (p1[0], p2[1])
                    l_stats['lower_L'] += 1
                else:
                    # UNKNOWN或其他，默认使用上L
                    p_mid = (p2[0], p1[1])
                    l_stats['unknown'] += 1
                
                steiner_segments.append([p1, p_mid])
                steiner_segments.append([p_mid, p2])

            steiner_length += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        logging.info(f"EGR Steiner L direction stats: upper_L={l_stats['upper_L']}, lower_L={l_stats['lower_L']}, "
                    f"straight={l_stats['straight']}, unknown={l_stats['unknown']}")

        # 4. Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if guide_wires:
            lc_guide = LineCollection(guide_wires, colors='blue', linewidths=0.5, 
                                      label=f'Route Guide (L={guide_length:.2f})', alpha=0.5)
            ax.add_collection(lc_guide)
            
        if steiner_segments:
            lc_steiner = LineCollection(steiner_segments, colors='green', linewidths=0.5, 
                                        label=f'EGR Steiner Tree (L={steiner_length:.2f})', alpha=0.7)
            ax.add_collection(lc_steiner)
            
        ax.autoscale()
        ax.set_aspect('equal')
        plt.legend(loc='upper right')
        
        plt.title(f"Route Guide vs EGR-built Steiner Tree\n"
                 f"Guide Length: {guide_length:.2f}, EGR Steiner Length: {steiner_length:.2f}")
        plt.xlabel("X (microns)")
        plt.ylabel("Y (microns)")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"EGR Steiner plot saved. Guide Length: {guide_length}, EGR Steiner Length: {steiner_length}")

    def __init__(self, params, placedb):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        super(NonLinearPlace, self).__init__(params, placedb)
        self._egr_padding_state = None

    def _apply_egr_padding(self, params, placedb):
        if not getattr(params, "egr_padding_flag", 0):
            return
        if self._egr_padding_state is not None:
            logging.warning("EGR padding is already active; skip duplicate apply")
            return

        congestion_map_op = getattr(
            self.op_collections, "irt_egr_congestion_map_op", None
        )
        if congestion_map_op is None:
            raise RuntimeError("EGR padding requested but iRT EGR op was not built")

        tt = time.time()
        with torch.no_grad():
            try:
                route_map = congestion_map_op(
                    self.pos[0], stage="egr3D", resolve_congestion="high"
                )
            except Exception as exc:
                raise RuntimeError(
                    "EGR padding congestion map generation failed"
                ) from exc

            self._egr_padding_state = apply_egr_padding(
                placedb=placedb,
                pos=self.pos[0],
                node_size_x=self.data_collections.node_size_x,
                node_size_y=self.data_collections.node_size_y,
                pin_offset_x=self.data_collections.pin_offset_x,
                pin2node_map=self.data_collections.pin2node_map,
                movable_macro_mask=self.data_collections.movable_macro_mask,
                route_map=route_map,
            )

        if self._egr_padding_state is None:
            logging.info("EGR padding selected no cells")
            return

        state = self._egr_padding_state
        logging.info(
            "EGR padding applied: selected %d/%d cells, threshold %.4g, "
            "max_congestion %.4g, padding_area %.6g, "
            "padding_area_ratio_movable %.6g, elapsed %.3fs",
            state.num_selected,
            placedb.num_movable_nodes,
            state.threshold,
            state.max_congestion,
            state.padding_area,
            state.padding_area_ratio,
            time.time() - tt,
        )

    def _restore_egr_padding(self):
        state = self._egr_padding_state
        if state is None:
            return
        with torch.no_grad():
            restore_egr_padding(
                state,
                self.pos[0],
                self.data_collections.node_size_x,
                self.data_collections.pin_offset_x,
            )
        logging.info(
            "EGR padding restored: selected %d cells, threshold %.4g",
            state.num_selected,
            state.threshold,
        )
        self._egr_padding_state = None

    def _run_standard_legalization(self, params, placedb, iteration, all_metrics):
        tt = time.time()
        self.pos[0].data.copy_(self.op_collections.legalize_op(self.pos[0]))
        logging.info("legalization takes %.3f seconds" % (time.time() - tt))
        cur_metric = EvalMetrics.EvalMetrics(iteration)
        all_metrics.append(cur_metric)
        cur_metric.evaluate(
            placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
        )
        logging.info(cur_metric)
        return iteration + 1

    def __call__(self, params, placedb):
        """
        @brief Top API to solve placement.
        @param params parameters
        @param placedb placement database
        """
        iteration = 0
        all_metrics = []
        inflation_legalization.validate_params(params)
        validate_ggr_l_shape_topology_params(params)
        if params.timing_opt_flag:
            timing_op = self.op_collections.timing_op
            time_unit = timing_op.timer.time_unit()

        # global placement
        if params.global_place_flag:

            global_place_stages = params.global_place_stages
            # macro place use external 1 stage to place macros
            if params.macro_place_flag:
                first_place_params = global_place_stages[0]
                if params.two_stage_flag:
                    global_place_stages.insert(0, first_place_params)
                macro_placed = False

                # add macro halo
                # if params.macro_halo_x > 0 or params.macro_halo_y > 0:
                #     with torch.no_grad():
                #         movable_macro_mask = self.data_collections.movable_macro_mask
                #         movable_macro_pins = self.data_collections.movable_macro_pins
                #         # node sizes
                #         self.data_collections.node_size_x[: placedb.num_movable_nodes][movable_macro_mask] += (
                #             2 * params.macro_halo_x)
                #         self.data_collections.node_size_y[: placedb.num_movable_nodes][movable_macro_mask] += (
                #             2 * params.macro_halo_y)
                #         # pin offsets
                #         self.data_collections.pin_offset_x[movable_macro_pins] += params.macro_halo_x
                #         self.data_collections.pin_offset_y[movable_macro_pins] += params.macro_halo_y
                #         # macro locations
                #         self.pos[0][: placedb.num_movable_nodes][movable_macro_mask] -= params.macro_halo_x
                #         self.pos[0][placedb.num_nodes: placedb.num_nodes +
                #                     placedb.num_movable_nodes][movable_macro_mask] -= params.macro_halo_y

            # global placement may run in multiple stages according to user specification
            for cur_stage, global_place_params in enumerate(global_place_stages):

                # we formulate each stage as a 3-nested optimization problem
                # f_gamma(g_density(h(x) ; density weight) ; gamma)
                # Lgamma      Llambda        Lsub
                # When optimizing an inner problem, the outer parameters are fixed.
                # This is a generalization to the eplace/RePlAce approach

                # As global placement may easily diverge, we record the position of best overflow
                best_metric = [None]
                best_pos = [None]

                if params.gpu:
                    torch.cuda.synchronize()
                tt = time.time()
                # construct model and optimizer
                density_weight = 0.0
                if params.macro_place_flag and cur_stage == 1:
                    density_weight = all_metrics[-1][-1][-1].density_weight.item(
                    ) / params.two_stage_density_scaler
                    # at the 2nd stage, total_movable_node_area should exclude movable macro area to enable more aggresive spreading of cells
                    placedb.total_movable_node_area = placedb.total_movable_cell_area
                # construct placement model
                model = PlaceObj.PlaceObj(
                    density_weight,
                    params,
                    placedb,
                    self.data_collections,
                    self.op_collections,
                    global_place_params,
                ).to(self.data_collections.pos[0].device)
                model.compile()
                if params.routability_opt_flag:
                    inflation_state = enhanced_inflation_controller.ensure_inflation_state(
                        params,
                        placedb,
                        self.data_collections,
                        stage_idx=cur_stage,
                    )
                    model.inflation_state = inflation_state
                    if getattr(params, "enhanced_inflation_flag", False):
                        logging.info(
                            "Initialized enhanced inflation controller for stage %d; "
                            "inflation rounds will restore best_pos before the route-driven "
                            "outer-loop area adjust.",
                            cur_stage,
                        )
                else:
                    model.inflation_state = None

                if params.macro_place_flag and macro_placed:
                    movable_macro_mask = self.data_collections.movable_macro_mask
                    model.fix_nodes_mask = movable_macro_mask.new_zeros(
                        placedb.num_nodes)
                    model.fix_nodes_mask[placedb.num_movable_nodes:placedb.num_physical_nodes] = 1
                    model.fix_nodes_mask[:placedb.num_movable_nodes] = movable_macro_mask[:placedb.num_movable_nodes]
                    # params.use_bb = False
                    # pdb.set_trace()

                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd_momentum":
                    optimizer = torch.optim.SGD(
                        self.parameters(), lr=0, momentum=0.9, nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov":
                    optimizer = torch.optim.SGD(
                        self.parameters(), lr=0, momentum=0.9, nesterov=True)
                elif optimizer_name.lower() == "nesterov":
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(
                        self.parameters(),
                        lr=0,
                        obj_and_grad_fn=model.obj_and_grad_fn,
                        constraint_fn=self.op_collections.move_boundary_op,
                        use_bb=params.use_bb
                    )
                else:
                    assert 0, "unknown optimizer %s" % (optimizer_name)

                logging.info("use %s optimizer" % (optimizer_name))
                model.train()
                # defining evaluation ops
                eval_ops = {
                    # "wirelength" : self.op_collections.wirelength_op,
                    # "density" : self.op_collections.density_op,
                    # "objective" : model.obj_fn,
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op,
                }
                # [DISABLED] route_overflow and pin_overflow computation disabled
                # if params.routability_opt_flag:
                #     eval_ops.update(
                #         {
                #             "route_utilization": self.op_collections.route_utilization_map_op,
                #             "pin_utilization": self.op_collections.pin_utilization_map_op,
                #         }
                #     )
                if len(placedb.regions) > 0:
                    eval_ops.update(
                        {
                            "density": self.op_collections.fence_region_density_merged_op,
                            "overflow": self.op_collections.fence_region_density_overflow_merged_op,
                            "goverflow": self.op_collections.density_overflow_op,
                        }
                    )

                # a function to initialize learning rate
                def initialize_learning_rate(pos):
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)
                    learning_rate = model.estimate_initial_learning_rate(
                        pos, global_place_params["learning_rate"]
                    )
                    # update learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate.data

                if iteration == 0 or (params.macro_place_flag and cur_stage == 1):
                    if iteration == 0 and params.gp_noise_ratio > 0.0:
                        logging.info("add %g%% noise" %
                                     (params.gp_noise_ratio * 100))
                        model.op_collections.noise_op(
                            model.data_collections.pos[0], params.gp_noise_ratio)
                    initialize_learning_rate(model.data_collections.pos[0])
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if params.gpu:
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" %
                             (optimizer_name, (time.time() - tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov"}:
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def Lgamma_stop_criterion(Lgamma_step, metrics, stop_mask=None):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-2][-1][-1]
                            # update stop mask for each fence region
                            # if(stop_mask is not None):
                            #     stop_mask.copy_(cur_metric.overflow < params.stop_overflow)

                            if Lgamma_step > 100 and (
                                # for fence region, the outer cell overflow decides the stopping of GP
                                (
                                    cur_metric.overflow[-1] < params.stop_overflow
                                    and cur_metric.hpwl > prev_metric.hpwl
                                )
                                or cur_metric.max_density[-1] < params.target_density
                            ):
                                logging.debug(
                                    "Lgamma stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Lgamma_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                            if len(placedb.regions) > 0 and model.update_mask.sum() == 0:
                                logging.debug(
                                    "All regions stop updating, finish global placement")
                                return True
                        # a heuristic to detect divergence and stop early
                        if len(metrics) > 50:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-50][-1][-1]
                            # record HPWL and overflow increase, and check divergence
                            if (
                                cur_metric.overflow[-1] > prev_metric.overflow[-1]
                                and cur_metric.hpwl > best_metric[0].hpwl * 2
                            ):
                                return True
                        return False

                def Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1]
                            prev_metric = metrics[-2][-1]
                            # for fence regions, the outer cell overflow and max_density decides whether to stop
                            if (
                                cur_metric.overflow[-1] < params.stop_overflow
                                and cur_metric.hpwl > prev_metric.hpwl
                            ) or cur_metric.max_density[-1] < 1.0:
                                logging.debug(
                                    "Llambda stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Llambda_density_weight_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                    return False

                # use a moving average window for stopping criteria, for an example window of 3
                # 0, 1, 2, 3, 4, 5, 6
                #    window2
                #             window1
                moving_avg_window = max(min(model.Lsub_iteration // 2, 3), 1)

                def attach_l_shape_telemetry(cur_metric):
                    l_shape_metric_active = (
                        getattr(model, "use_l_shape_routability", False)
                        or getattr(model, "l_shape_last_cost", None) is not None
                        or getattr(model, "l_shape_last_grad_norm", None) is not None
                    )
                    if not l_shape_metric_active:
                        return
                    field_map = {
                        "l_shape_fast_mode": "l_shape_fast_mode",
                        "l_shape_energy_valid": "l_shape_energy_valid",
                        "l_shape_cost": "l_shape_last_cost",
                        "l_shape_weighted_cost": "l_shape_last_weighted_cost",
                        "l_shape_weight": "l_shape_last_weight",
                        "l_shape_target_weight": "l_shape_last_target_weight",
                        "l_shape_weight_candidate": "l_shape_last_weight_candidate",
                        "l_shape_cap_active": "l_shape_last_cap_active",
                        "l_shape_base_grad_norm": "l_shape_last_base_grad_norm",
                        "l_shape_grad_raw_norm": "l_shape_last_grad_raw_norm",
                        "l_shape_grad_norm": "l_shape_last_grad_norm",
                        "l_shape_grad_ratio": "l_shape_last_grad_ratio",
                    }
                    for metric_field, model_field in field_map.items():
                        value = getattr(model, model_field, None)
                        if value is not None:
                            setattr(cur_metric, metric_field, value)

                    cur_metric.l_shape_log_verbose = l_shape_log_verbose(params)
                    cur_metric.l_shape_target_ratio = float(
                        model.l_shape_grad_target_ratio
                    )

                    overflow_ema = getattr(model, "_l_shape_overflow_ema", None)
                    if overflow_ema is not None:
                        cur_metric.l_shape_overflow_ema = float(overflow_ema)

                    al_summary = getattr(
                        model, "l_shape_capacity_al_last_summary", None
                    ) or {}
                    al_field_map = {
                        "l_shape_capacity_al_enabled": "enabled",
                        "l_shape_capacity_al_updated": "updated",
                        "l_shape_capacity_al_g_h_max": "g_h_max",
                        "l_shape_capacity_al_g_v_max": "g_v_max",
                        "l_shape_capacity_al_g_h_sum": "g_h_sum",
                        "l_shape_capacity_al_g_v_sum": "g_v_sum",
                        "l_shape_capacity_al_g_h_pos_ratio": "g_h_pos_ratio",
                        "l_shape_capacity_al_g_v_pos_ratio": "g_v_pos_ratio",
                        "l_shape_capacity_al_q_h_max": "q_h_max",
                        "l_shape_capacity_al_q_v_max": "q_v_max",
                        "l_shape_capacity_al_q_h_sum": "q_h_sum",
                        "l_shape_capacity_al_q_v_sum": "q_v_sum",
                        "l_shape_capacity_al_lambda_h_max": "lambda_h_max",
                        "l_shape_capacity_al_lambda_v_max": "lambda_v_max",
                        "l_shape_capacity_al_lambda_h_sum": "lambda_h_sum",
                        "l_shape_capacity_al_lambda_v_sum": "lambda_v_sum",
                        "l_shape_capacity_al_energy_h": "E_cap_smooth_h",
                        "l_shape_capacity_al_energy_v": "E_cap_smooth_v",
                        "l_shape_capacity_al_energy_total": "E_cap_smooth_total",
                        "l_shape_capacity_al_pq_h_min": "Pq_h_min",
                        "l_shape_capacity_al_pq_h_max": "Pq_h_max",
                        "l_shape_capacity_al_pq_h_sum": "Pq_h_sum",
                        "l_shape_capacity_al_pq_v_min": "Pq_v_min",
                        "l_shape_capacity_al_pq_v_max": "Pq_v_max",
                        "l_shape_capacity_al_pq_v_sum": "Pq_v_sum",
                        "l_shape_capacity_al_active_memory_bins_h": "active_memory_bins_h",
                        "l_shape_capacity_al_active_memory_bins_v": "active_memory_bins_v",
                    }
                    for metric_field, summary_field in al_field_map.items():
                        value = al_summary.get(summary_field)
                        if value is not None or summary_field in al_summary:
                            setattr(cur_metric, metric_field, value)

                    macro_summary = getattr(
                        model, "l_shape_macro_exclusion_last_summary", None
                    ) or {}
                    macro_field_map = {
                        "l_shape_macro_exclusion_enabled": "macro_exclusion_enabled",
                        "l_shape_macro_exclusion_macro_count": "macro_count",
                        "l_shape_macro_exclusion_body_bins": "macro_body_bins",
                        "l_shape_macro_exclusion_halo_bins": "macro_halo_bins",
                        "l_shape_macro_exclusion_active_bins": "macro_source_active_bins",
                        "l_shape_macro_exclusion_source_max": "macro_source_max",
                        "l_shape_macro_exclusion_source_sum": "macro_source_sum",
                        "l_shape_macro_exclusion_body_source_max": "macro_body_source_max",
                        "l_shape_macro_exclusion_body_source_sum": "macro_body_source_sum",
                        "l_shape_macro_exclusion_halo_source_max": "macro_halo_source_max",
                        "l_shape_macro_exclusion_halo_source_sum": "macro_halo_source_sum",
                        "l_shape_macro_exclusion_usage_max": "macro_usage_max",
                        "l_shape_macro_exclusion_usage_sum": "macro_usage_sum",
                        "l_shape_macro_exclusion_usage_bins": "macro_usage_active_bins",
                        "l_shape_macro_exclusion_dominates_bins": "macro_dominates_bins",
                        "l_shape_macro_exclusion_routing_dominates_bins": "routing_dominates_macro_bins",
                    }
                    for metric_field, summary_field in macro_field_map.items():
                        value = macro_summary.get(summary_field)
                        if value is not None:
                            setattr(cur_metric, metric_field, value)

                    soft_summary = getattr(model, "soft_l_last_summary", None) or {}
                    soft_field_map = {
                        "soft_l_diag_count": "diag_edge_count",
                        "soft_l_mean_cost_gap": "mean_cost_gap",
                        "soft_l_raw_cost_gap_p50": "raw_cost_gap_p50",
                        "soft_l_biased_cost_gap_p50": "biased_cost_gap_p50",
                        "soft_l_tau_source_gap": "tau_source_gap",
                        "soft_l_mean_max_prob": "mean_max_prob",
                        "soft_l_mean_entropy": "mean_entropy",
                        "soft_l_near_tie_ratio": "near_tie_ratio",
                        "soft_l_tau": "tau",
                        "soft_l_effective_hotspot_weight": "effective_hotspot_weight",
                        "soft_l_resolver_agreement_ratio": "resolver_agreement_ratio",
                        "soft_l_target_demand_supply_ratio": "target_demand_supply_ratio",
                        "soft_l_current_demand_supply_ratio": "current_demand_supply_ratio",
                        "soft_l_same_net_topo_nets": "same_net_topo_nets",
                        "soft_l_same_net_topo_segments_h": "same_net_topo_segments_h",
                        "soft_l_same_net_topo_segments_v": "same_net_topo_segments_v",
                        "soft_l_same_net_topo_diag_edges": "same_net_topo_diag_edges",
                        "soft_l_same_net_topo_edges_with_topology": "same_net_topo_edges_with_topology",
                        "soft_l_same_net_topo_edges_with_observed_intervals": "same_net_topo_edges_with_observed_intervals",
                        "soft_l_same_net_topo_mean_gap": "same_net_topo_mean_gap",
                        "soft_l_same_net_topo_tie_ratio": "same_net_topo_tie_ratio",
                    }
                    for metric_field, summary_field in soft_field_map.items():
                        value = soft_summary.get(summary_field)
                        if value is not None:
                            setattr(cur_metric, metric_field, value)

                def reset_l_shape_auto_disable_state():
                    model._l_shape_auto_disabled = False
                    model._l_shape_auto_disable_state = {
                        "update_count": 0,
                        "best_lcost": None,
                        "best_iteration": None,
                        "prev_ov_ema": None,
                        "last_ov_ema_improvement": None,
                        "overshoot_streak": 0,
                        "rebound_streak": 0,
                        "plateau_streak": 0,
                    }

                def reset_l_shape_reenable_state():
                    base_threshold = float(
                        getattr(params, "l_shape_overflow_threshold", 0.2)
                    )
                    model._l_shape_reenable_threshold_base = base_threshold
                    model._l_shape_reenable_threshold = base_threshold
                    model._l_shape_reenable_count = 0
                    model._l_shape_reenable_last_overflow = None
                    model._l_shape_reenable_descend_streak = 0
                    model._l_shape_ratio_last = None
                    model._l_shape_ratio_rise_streak = 0
                    model._l_shape_ratio_guard_disabled = False
                    model._l_shape_inflation_guard_disabled = False

                def disable_l_shape_for_recovery(
                    iteration,
                    reason,
                    update_threshold=False,
                    inflation_round=None,
                    ratio_info=None,
                ):
                    if not getattr(model, "use_l_shape_routability", False):
                        return

                    model.use_l_shape_routability = False
                    model.enable_l_shape_routability = False
                    if hasattr(model, "reset_l_shape_weight_state"):
                        model.reset_l_shape_weight_state()
                    reset_l_shape_auto_disable_state()

                    base_threshold = float(
                        getattr(
                            model,
                            "_l_shape_reenable_threshold_base",
                            getattr(params, "l_shape_overflow_threshold", 0.2),
                        )
                    )
                    current_threshold = float(
                        getattr(
                            model,
                            "_l_shape_reenable_threshold",
                            base_threshold,
                        )
                    )
                    reenable_count = int(
                        getattr(model, "_l_shape_reenable_count", 0)
                    )
                    next_threshold = current_threshold
                    if update_threshold:
                        reenable_count += 1
                        next_threshold = current_threshold * 0.7
                        model._l_shape_reenable_count = reenable_count
                        model._l_shape_reenable_threshold = next_threshold

                    model._l_shape_reenable_last_overflow = None
                    model._l_shape_reenable_descend_streak = 0
                    model._l_shape_ratio_last = None
                    model._l_shape_ratio_rise_streak = 0

                    if reason == "inflation":
                        model._l_shape_inflation_guard_disabled = True
                        logging.info(
                            "L-shape disabled due to inflation (round %d) and permanently disabled "
                            "for the remaining placement "
                            "(next_threshold=%.4f, base_threshold=%.4f, reenable_count=%d)",
                            inflation_round,
                            next_threshold,
                            base_threshold,
                            reenable_count,
                        )
                    elif reason == "demand_supply_ratio_limit":
                        model._l_shape_ratio_guard_disabled = True
                        logging.info(
                            "L-shape disabled due to demand/supply ratio limit at iteration %d: "
                            "ratio=%.4f > 0.6500; permanently disabled for the remaining placement "
                            "(reenable_threshold=%.4f, reenable_count=%d)",
                            iteration,
                            float((ratio_info or {}).get("current_ratio", float("nan"))),
                            current_threshold,
                            reenable_count,
                        )
                    elif reason == "demand_supply_ratio_rise":
                        model._l_shape_ratio_guard_disabled = True
                        logging.info(
                            "L-shape disabled due to demand/supply ratio surge at iteration %d: "
                            "ratio %.4f -> %.4f (rise_streak=%d, rel_increase=%.2f%%, "
                            "permanently disabled for the remaining placement; "
                            "reenable_threshold=%.4f, reenable_count=%d)",
                            iteration,
                            float((ratio_info or {}).get("prev_ratio", float("nan"))),
                            float((ratio_info or {}).get("current_ratio", float("nan"))),
                            int((ratio_info or {}).get("rise_streak", 0)),
                            float((ratio_info or {}).get("rel_increase_pct", 0.0)),
                            current_threshold,
                            reenable_count,
                        )

                def maybe_auto_disable_l_shape(iteration, outer_update=False):
                    if not getattr(params, "l_shape_auto_disable_flag", False):
                        return
                    if not getattr(model, "use_l_shape_routability", False):
                        return
                    if getattr(model, "_l_shape_auto_disabled", False):
                        return

                    l_shape_op = getattr(model, "l_shape_routability_op", None)
                    if l_shape_op is None or not getattr(l_shape_op, "soft_l_assignment", False):
                        return
                    if bool(getattr(model, "l_shape_fast_mode", False)) and not bool(
                        getattr(model, "l_shape_energy_valid", True)
                    ):
                        if outer_update and l_shape_log_verbose(params) >= 1:
                            logging.info(
                                "Skip L-shape auto-disable cost-rebound check because l_shape_fast_mode invalidates scalar energy"
                            )
                        return

                    current_cost = getattr(model, "l_shape_last_cost", None)
                    current_grad_ratio = getattr(model, "l_shape_last_grad_ratio", None)
                    current_ov_ema = getattr(model, "_l_shape_overflow_ema", None)
                    if current_cost is None or current_grad_ratio is None or current_ov_ema is None:
                        return
                    if not math.isfinite(current_cost) or not math.isfinite(current_grad_ratio):
                        return

                    state = getattr(model, "_l_shape_auto_disable_state", None)
                    if not isinstance(state, dict):
                        reset_l_shape_auto_disable_state()
                        state = model._l_shape_auto_disable_state

                    if outer_update:
                        state["update_count"] += 1

                    best_cost = state.get("best_lcost")
                    if best_cost is None or current_cost < best_cost:
                        state["best_lcost"] = current_cost
                        state["best_iteration"] = iteration
                        state["rebound_streak"] = 0
                    else:
                        rebound_ratio = float(
                            getattr(
                                params,
                                "l_shape_auto_disable_lcost_rebound_ratio",
                                0.05,
                            )
                        )
                        rebound_threshold = best_cost * (1.0 + rebound_ratio)
                        if current_cost > rebound_threshold:
                            state["rebound_streak"] += 1
                        else:
                            state["rebound_streak"] = 0

                    target_ratio = float(model.l_shape_grad_target_ratio)
                    overshoot_margin = float(
                        getattr(
                            params,
                            "l_shape_auto_disable_grad_overshoot_margin",
                            0.01,
                        )
                    )
                    if current_grad_ratio > target_ratio + overshoot_margin:
                        state["overshoot_streak"] += 1
                    else:
                        state["overshoot_streak"] = 0

                    prev_ov_ema = state.get("prev_ov_ema")
                    ema_improvement = None
                    if outer_update:
                        if prev_ov_ema is not None and math.isfinite(prev_ov_ema):
                            plateau_eps = float(
                                getattr(
                                    params,
                                    "l_shape_auto_disable_ov_ema_plateau_eps",
                                    1e-3,
                                )
                            )
                            ema_improvement = prev_ov_ema - current_ov_ema
                            if ema_improvement <= plateau_eps:
                                state["plateau_streak"] += 1
                            else:
                                state["plateau_streak"] = 0
                            state["last_ov_ema_improvement"] = ema_improvement
                        state["prev_ov_ema"] = float(current_ov_ema)

                    warmup_updates = max(
                        0,
                        int(
                            getattr(
                                params,
                                "l_shape_auto_disable_warmup_updates",
                                3,
                            )
                        ),
                    )
                    patience = max(
                        1,
                        int(getattr(params, "l_shape_auto_disable_patience", 2)),
                    )
                    if state["update_count"] <= warmup_updates:
                        return
                    if state["overshoot_streak"] < patience:
                        return
                    if state["rebound_streak"] < patience:
                        return
                    if state["plateau_streak"] < patience:
                        return

                    model.use_l_shape_routability = False
                    model.enable_l_shape_routability = False
                    model._l_shape_auto_disabled = True
                    state["disabled_at"] = iteration

                    rebound_pct = 0.0
                    if state.get("best_lcost"):
                        rebound_pct = (
                            (current_cost - state["best_lcost"])
                            / max(state["best_lcost"], 1e-12)
                            * 100.0
                        )
                    logging.info(
                        "L-shape auto-disabled at iteration %d: "
                        "LShapeCostRaw %.6e rebounded from best %.6e@iter=%s by %.2f%%, "
                        "LGradRatio %.4f > LTargetRatio %.4f + %.4f, "
                        "ov_ema improvement %.3e.",
                        iteration,
                        current_cost,
                        state.get("best_lcost", float("nan")),
                        state.get("best_iteration"),
                        rebound_pct,
                        current_grad_ratio,
                        target_ratio,
                        overshoot_margin,
                        state.get("last_ov_ema_improvement", float("nan")),
                    )

                def maybe_disable_l_shape_by_ratio(iteration):
                    if not getattr(model, "use_l_shape_routability", False):
                        return
                    if int(getattr(model, "_l_shape_reenable_count", 0)) <= 0:
                        return

                    soft_summary = getattr(model, "soft_l_last_summary", None) or {}
                    current_ratio = soft_summary.get("current_demand_supply_ratio")
                    if current_ratio is None:
                        current_ratio = soft_summary.get("target_demand_supply_ratio")
                    if current_ratio is None or not math.isfinite(float(current_ratio)):
                        return
                    current_ratio = float(current_ratio)

                    absolute_limit = 0.65
                    relative_rise_limit = 0.03
                    prev_ratio = getattr(model, "_l_shape_ratio_last", None)
                    rise_streak = int(getattr(model, "_l_shape_ratio_rise_streak", 0))

                    rel_increase = None
                    if prev_ratio is not None and math.isfinite(prev_ratio) and prev_ratio > 1e-9:
                        rel_increase = (current_ratio - prev_ratio) / prev_ratio
                        if current_ratio > prev_ratio and rel_increase > relative_rise_limit:
                            rise_streak += 1
                        else:
                            rise_streak = 0
                    else:
                        rise_streak = 0

                    model._l_shape_ratio_last = current_ratio
                    model._l_shape_ratio_rise_streak = rise_streak

                    if current_ratio > absolute_limit:
                        disable_l_shape_for_recovery(
                            iteration,
                            reason="demand_supply_ratio_limit",
                            update_threshold=False,
                            ratio_info={"current_ratio": current_ratio},
                        )
                        return

                    if rise_streak >= 2 and rel_increase is not None:
                        disable_l_shape_for_recovery(
                            iteration,
                            reason="demand_supply_ratio_rise",
                            update_threshold=False,
                            ratio_info={
                                "prev_ratio": prev_ratio,
                                "current_ratio": current_ratio,
                                "rise_streak": rise_streak,
                                "rel_increase_pct": rel_increase * 100.0,
                            },
                        )

                def Lsub_stop_criterion(Lgamma_step, Llambda_density_weight_step, Lsub_step, metrics):
                    with torch.no_grad():
                        if len(metrics) >= moving_avg_window * 2:
                            cur_avg_obj = 0
                            prev_avg_obj = 0
                            for i in range(moving_avg_window):
                                cur_avg_obj += metrics[-1 - i].objective
                                prev_avg_obj += metrics[-1 -
                                                        moving_avg_window - i].objective
                            cur_avg_obj /= moving_avg_window
                            prev_avg_obj /= moving_avg_window
                            threshold = 0.999
                            if cur_avg_obj >= prev_avg_obj * threshold:
                                logging.debug(
                                    "Lsub stopping criteria: %d and %g > %g * %g"
                                    % (Lsub_step, cur_avg_obj, prev_avg_obj, threshold)
                                )
                                return True
                    return False

                def one_descent_step(
                    Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, metrics, stop_mask=None
                ):
                    t0 = time.time()

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(
                        iteration, (Lgamma_step,
                                    Llambda_density_weight_step, Lsub_step)
                    )
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    pos = model.data_collections.pos[0]
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)

                    # move any out-of-bound cell back to placement region
                    self.op_collections.move_boundary_op(pos)

                    # handle multiple density weights for multi-electric field
                    if torch.eq(model.density_weight.mean(), 0.0):
                        model.initialize_density_weight(params, placedb)
                        if model.density_weight.size(0) == 1:
                            logging.info("density_weight = %.6E" %
                                         (model.density_weight.data))
                        else:
                            logging.info(
                                "density_weight = [%s]"
                                % ", ".join(["%.3E" % i for i in model.density_weight.cpu().numpy().tolist()])
                            )

                    # For backward compatibility
                    # PyTorch 1.7 introduced zero_grad(set_to_none=False)
                    # PyTorch 2.0 changed set_to_none=True
                    if "set_to_none" in inspect.signature(optimizer.zero_grad).parameters:
                        optimizer.zero_grad(set_to_none=False)
                    else:
                        optimizer.zero_grad()

                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops,
                                        pos, model.data_collections)
                    model.overflow = cur_metric.overflow.data.clone()
                    # logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    # t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov"]:
                        obj, grad = model.obj_and_grad_fn(pos)
                        cur_metric.objective = obj.data.clone()
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # diff tdp
                    if params.with_sta and (iteration % 10 == 0 and iteration >= 100):
                        t_steiner = time.time()
                        with torch.no_grad():
                            pin_pos = self.op_collections.pin_pos_op(pos)
                            if pin_pos.is_cuda:
                                pin_pos = pin_pos.cpu()
                            self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(
                                    pin_pos)
                            model.use_timing_obj = True
                        logging.info("Update steiner topo %.3f ms" %
                                     ((time.time() - t_steiner) * 1000))


                    # ========== L形Routability Density Objective ==========
                    # 根据overflow条件启用L形routability
                    l_shape_routability_enabled = (
                        params.routability_opt_flag
                        and getattr(params, "l_shape_routability_flag", 0)
                    )
                    if l_shape_routability_enabled:
                        
                        if (
                            not model.enable_l_shape_routability
                            and not getattr(model, "_l_shape_auto_disabled", False)
                            and not getattr(model, "_l_shape_ratio_guard_disabled", False)
                            and not getattr(model, "_l_shape_inflation_guard_disabled", False)
                        ):
                            current_overflow = float(cur_metric.overflow[-1])
                            overflow_threshold = float(
                                getattr(
                                    model,
                                    "_l_shape_reenable_threshold",
                                    getattr(params, "l_shape_overflow_threshold", 0.2),
                                )
                            )
                            reenable_count = int(
                                getattr(model, "_l_shape_reenable_count", 0)
                            )
                            if reenable_count <= 0:
                                if current_overflow < overflow_threshold:
                                    model.enable_l_shape_routability = True
                            else:
                                last_overflow = getattr(
                                    model,
                                    "_l_shape_reenable_last_overflow",
                                    None,
                                )
                                descend_streak = int(
                                    getattr(
                                        model,
                                        "_l_shape_reenable_descend_streak",
                                        0,
                                    )
                                )
                                if (
                                    last_overflow is not None
                                    and math.isfinite(last_overflow)
                                    and current_overflow < last_overflow - 1e-6
                                ):
                                    descend_streak += 1
                                else:
                                    descend_streak = 0
                                model._l_shape_reenable_last_overflow = current_overflow
                                model._l_shape_reenable_descend_streak = descend_streak
                                if (
                                    descend_streak >= 5
                                    and current_overflow < overflow_threshold
                                ):
                                    model.enable_l_shape_routability = True
                            
                            # # 条件2: 也可以根据iteration启用
                            # if iteration >= getattr(params, 'l_shape_start_iteration', 100):
                            #     model.enable_l_shape_routability = True
                        
                        if model.enable_l_shape_routability and not model.use_l_shape_routability:
                            # 首次启用L形routability
                            t_l_shape_init = time.time()
                            
                            L_shape_num_bins_x = params.num_bins_x
                            L_shape_num_bins_y = params.num_bins_y

                            ggr_topology_mode = use_ggr_l_shape_topology(params)
                            gpugr_inputs = None
                            l_shape_inputs = None
                            if ggr_topology_mode:
                                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                    params,
                                    placedb,
                                    pos,
                                    model=model,
                                )
                                l_shape_inputs = gpugr_inputs
                                l_directions = _load_l_shape_topology_pack_from_gpugr(
                                    params,
                                    pos,
                                    self.op_collections.pin_pos_op,
                                    self.op_collections.steiner_topo_op,
                                    self.data_collections,
                                    l_shape_inputs,
                                    "l_shape_init.load_ggr_topology_pack",
                                    iteration,
                                )
                            else:
                                with profile_scope(params, "l_shape_init.rebuild_tree", tensor=pos, iteration=iteration):
                                    with torch.no_grad():
                                        pin_pos = self.op_collections.pin_pos_op(pos)
                                        if pin_pos.is_cuda:
                                            pin_pos = pin_pos.cpu()
                                        self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                            self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                            self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(pin_pos)
                                if getattr(params, "l_direction_use_gpugr", False):
                                    gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                    l_shape_inputs = gpugr_inputs
                                else:
                                    l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )

                            supply_map = l_shape_inputs["supply_map"]
                            demand_map = l_shape_inputs["demand_map"]
                            wire_width = l_shape_inputs["wire_width"]

                            # Resolve L directions from the selected topology source.
                            steiner_topo_op = self.op_collections.steiner_topo_op
                            if ggr_topology_mode:
                                l_directions = steiner_topo_op.edge_l_directions
                            elif _should_skip_resolver_l_direction_for_soft(params):
                                steiner_topo_op.edge_l_directions = None
                                l_directions = None
                                if l_shape_log_verbose(params) >= 2:
                                    logging.info(
                                        "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                                        "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                                        "will be refreshed."
                                    )
                            else:
                                l_directions = _resolve_l_directions_for_l_shape(
                                    params,
                                    placedb,
                                    pos,
                                    steiner_topo_op,
                                    gpugr_route_entries=(
                                        gpugr_inputs["route_entries"]
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                    gpugr_metrics=(
                                        gpugr_inputs["metrics"]
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                    gpugr_route_grid=(
                                        (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                )

                            # # ========== Plot edges with L-shape by l_direction ==========
                            # import matplotlib.pyplot as plt
                            # import matplotlib.collections as mc
                            
                            # # 获取坐标和边信息
                            # newx = steiner_topo_op.newx.cpu().numpy()
                            # newy = steiner_topo_op.newy.cpu().numpy()
                            # flat_pin_from = self.data_collections.flat_pin_from.cpu().numpy()
                            # flat_pin_to = self.data_collections.flat_pin_to.cpu().numpy()
                            # l_dirs = l_directions.cpu().numpy()
                            
                            # # 颜色映射: H_FIRST=0(红), V_FIRST=1(蓝), STRAIGHT=2(绿), FAKE_STRAIGHT=3(橙)
                            # color_map = {
                            #     0: 'red',      # H_FIRST: 先水平后垂直
                            #     1: 'blue',     # V_FIRST: 先垂直后水平
                            #     2: 'green',    # STRAIGHT: 直线
                            #     3: 'orange'    # FAKE_STRAIGHT: 伪直线
                            # }
                            # label_map = {
                            #     0: 'H_FIRST (H→V)',
                            #     1: 'V_FIRST (V→H)',
                            #     2: 'STRAIGHT',
                            #     3: 'FAKE_STRAIGHT'
                            # }
                            
                            # # 按l_direction分组收集线段
                            # # L形边变成两段，直线保持一段
                            # edges_by_dir = {0: [], 1: [], 2: [], 3: []}
                            # edge_count_by_dir = {0: 0, 1: 0, 2: 0, 3: 0}
                            
                            # for i in range(len(l_dirs)):
                            #     from_idx = flat_pin_from[i]
                            #     to_idx = flat_pin_to[i]
                            #     if from_idx == -1 or to_idx == -1:
                            #         continue
                            #     x1, y1 = newx[from_idx], newy[from_idx]
                            #     x2, y2 = newx[to_idx], newy[to_idx]
                            #     direction = int(l_dirs[i])
                            #     if direction not in edges_by_dir:
                            #         direction = 1  # 默认用V_FIRST
                                
                            #     edge_count_by_dir[direction] += 1
                                
                            #     # 根据方向生成路径
                            #     if direction == 0:  # H_FIRST: 水平优先 (x1,y1) -> (x2,y1) -> (x2,y2)
                            #         corner = (x2, y1)
                            #         edges_by_dir[direction].append([(x1, y1), corner])
                            #         edges_by_dir[direction].append([corner, (x2, y2)])
                            #     elif direction == 2:  # STRAIGHT: 直线
                            #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
                            #     elif direction == 3:  # FAKE_STRAIGHT: 伪直线（画成直线）
                            #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
                            #     elif direction == 1:  # V_FIRST: 垂直优先 (x1,y1) -> (x1,y2) -> (x2,y2)
                            #         # (x1,y1) -> (x1,y2) -> (x2,y2)
                            #         corner = (x1, y2)
                            #         edges_by_dir[direction].append([(x1, y1), corner])
                            #         edges_by_dir[direction].append([corner, (x2, y2)])
                            
                            # # 绘图
                            # fig, ax = plt.subplots(figsize=(12, 10))
                            
                            # for direction, edges in edges_by_dir.items():
                            #     if len(edges) > 0:
                            #         lc = mc.LineCollection(edges, colors=color_map[direction], 
                            #                               linewidths=0.5, alpha=0.7,
                            #                               label=f'{label_map[direction]} ({edge_count_by_dir[direction]})')
                            #         ax.add_collection(lc)
                            
                            # ax.autoscale()
                            # ax.set_aspect('equal')
                            # ax.set_xlabel('X')
                            # ax.set_ylabel('Y')
                            # ax.set_title('Steiner Tree L-Shape Edges')
                            # ax.legend(loc='upper right')
                            
                            # # 保存图片
                            # plot_path = os.path.join(params.result_dir, 'l_direction_edges.png')
                            # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                            # plt.close()
                            # logging.info(f"L-direction edge plot saved to {plot_path}")
                            
                            # exit(0)
                            
                            # Initialize the L-shape routability operator.

                            with profile_scope(params, "l_shape_init.construct_op", tensor=pos, iteration=iteration):
                                model.init_l_shape_routability(
                                    wire_width=wire_width,
                                    num_bins_x=L_shape_num_bins_x,
                                    num_bins_y=L_shape_num_bins_y,
                                    target_density=supply_map,
                                    target_demand=demand_map,
                                    raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                                    supply_original=l_shape_inputs.get("supply_original"),
                                    target_density_h=l_shape_inputs.get("supply_map_h"),
                                    target_density_v=l_shape_inputs.get("supply_map_v"),
                                    target_demand_h=l_shape_inputs.get("demand_map_h"),
                                    target_demand_v=l_shape_inputs.get("demand_map_v"),
                                    raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                                    raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                                    supply_original_h=l_shape_inputs.get("supply_original_h"),
                                    supply_original_v=l_shape_inputs.get("supply_original_v"),
                                    fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                                    fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                                    fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
                                )
                            if model.l_shape_routability_op is not None:
                                model.l_shape_routability_op.update_same_net_topology(
                                    topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                                    topo_stats=l_shape_inputs.get("same_net_topo_stats"),
                                )
                            if hasattr(model, "start_l_shape_weight_controller"):
                                model.start_l_shape_weight_controller(iteration)
                            # 初始化基于L-shape overflow的外环状态
                            model._l_shape_overflow_ema = None
                            model._l_shape_overflow_last = None
                            
                            if l_shape_log_verbose(params) >= 1:
                                logging.info(f"L-shape routability enabled at iteration {iteration}, "
                                            f"overflow={cur_metric.overflow[-1]:.4f}, "
                                            f"threshold={float(getattr(model, '_l_shape_reenable_threshold', getattr(params, 'l_shape_overflow_threshold', 0.2))):.4f}, "
                                            f"descend_streak={int(getattr(model, '_l_shape_reenable_descend_streak', 0))}, "
                                            f"init time={((time.time() - t_l_shape_init) * 1000):.2f}ms")
                            model._l_shape_reenable_last_overflow = None
                            model._l_shape_reenable_descend_streak = 0
                            
                            # ========== 梯度正确性检查 (可选) ==========
                            if getattr(params, 'l_shape_gradient_check', False):
                                # 1. 运行梯度链诊断
                                logging.info("Running L-shape gradient chain diagnosis...")
                                model.diagnose_l_shape_gradient_chain(pos)
                                
                                # 2. 运行梯度问题诊断（检查边界问题）
                                logging.info("Running L-shape gradient issues diagnosis...")
                                model.diagnose_l_shape_gradient_issues(pos)
                                
                                # 3. 运行梯度方向检查（更实用）
                                logging.info("Running L-shape gradient direction check...")
                                direction_results = model.check_l_shape_gradient_direction(
                                    pos, 
                                    step_sizes=[0.1, 1.0, 10.0, 100.0]
                                )
                                
                                # 4. 可选：运行数值梯度检查（对bin-based函数可能失败）
                                logging.info("Running L-shape gradient numerical check...")
                                logging.info("Note: Numerical check may fail for bin-based density functions")
                                grad_check_results = model.check_l_shape_gradient_numerical(
                                    pos, 
                                    num_check=200,  # 检查200个位置
                                    eps=1e-3,       # 有限差分步长
                                    check_movable_only=True,
                                    verbose=True
                                )
                                
                                # 保存结果到文件
                                import json
                                results_to_save = {
                                    'direction_check': direction_results,
                                    'numerical_check': grad_check_results
                                }
                                grad_check_path = os.path.join(params.result_dir, "l_shape_grad_check.json")
                                with open(grad_check_path, 'w') as f:
                                    json.dump(results_to_save, f, indent=2)
                                logging.info(f"Gradient check results saved to {grad_check_path}")

                                exit(0)
                            # =============================================
                            
                            # 可视化L形密度图和segments
                            if params.l_shape_plot_flag:
                                try:
                                    from dreamplace.ops.routability.l_shape_routability import (
                                        plot_l_shape_electric_overflow_map,
                                        plot_l_shape_initial_density_map,
                                        plot_l_shape_macro_source_maps,
                                        plot_l_shape_electric_potential_map,
                                        plot_l_shape_supply_maps,
                                        plot_l_shape_true_source_maps,
                                        plot_segment_density_map,
                                        plot_soft_l_intermediate,
                                        plot_soft_l_scoring_maps,
                                    )
                                    
                                    # 获取密度图
                                    forward_source_snapshot = _snapshot_l_shape_forward_source()
                                    density_map = model.get_l_shape_density_map(pos, use_l_direction=True)
                                    _restore_l_shape_forward_source(forward_source_snapshot)
                                    if density_map is not None:
                                        density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_density_iter{iteration}.png"
                                        )
                                        plot_segment_density_map(
                                            density_map, density_plot_path,
                                            title=f"L-shape Density (iter={iteration})",
                                            colormap="binary"
                                        )
                                        logging.info(f"L-shape density plot saved to {density_plot_path}")
                                    
                                    # 绘制基于 electric potential 的 overflow map
                                    if model.l_shape_routability_op is not None and \
                                       model.l_shape_routability_op.cached_segments is not None:
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_overflow_map(
                                            l_shape_op,
                                            output_path=overflow_plot_path,
                                            title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                                        )
                                        logging.info(f"L-shape electric overflow plot saved to {overflow_plot_path}")
                                        supply_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_supply_iter{iteration}.png"
                                        )
                                        plot_l_shape_supply_maps(
                                            l_shape_op,
                                            output_path=supply_plot_path,
                                            title_prefix=f"L-shape Supply Debug (iter={iteration})",
                                        )
                                        logging.info(f"L-shape supply debug plot saved to {supply_plot_path}")
                                        initial_density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                                        )
                                        plot_l_shape_initial_density_map(
                                            l_shape_op,
                                            output_path=initial_density_plot_path,
                                            title_prefix=f"L-shape Initial Density (iter={iteration})",
                                        )
                                        logging.info(
                                            f"L-shape initial density plot saved to {initial_density_plot_path}"
                                        )
                                        source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_true_source_maps(
                                            l_shape_op,
                                            output_path=source_plot_path,
                                            title_prefix=f"L-shape True Source (iter={iteration})",
                                        )
                                        logging.info(f"L-shape true source plot saved to {source_plot_path}")
                                        macro_source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_macro_source_maps(
                                            l_shape_op,
                                            output_path=macro_source_plot_path,
                                            title_prefix=f"L-shape Macro Source (iter={iteration})",
                                        )
                                        logging.info(
                                            f"L-shape macro source plot saved to {macro_source_plot_path}"
                                        )
                                        potential_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_potential_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_potential_map(
                                            l_shape_op,
                                            output_path=potential_plot_path,
                                            title_prefix=f"L-shape Electric Potential (iter={iteration})",
                                        )
                                        logging.info(f"L-shape electric potential plot saved to {potential_plot_path}")
                                        if getattr(l_shape_op, "soft_l_assignment", False) and \
                                           'soft_l_weights' in l_shape_op.cached_segments:
                                            soft_plot_path = os.path.join(
                                                params.result_dir, f"l_shape_soft_iter{iteration}.png"
                                            )
                                            plot_soft_l_intermediate(
                                                l_shape_op.cached_segments,
                                                output_path=soft_plot_path,
                                            )
                                            logging.info(f"Soft L-shape plot saved to {soft_plot_path}")
                                            if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                                                soft_scoring_path = os.path.join(
                                                    params.result_dir,
                                                    f"l_shape_soft_scoring_iter{iteration}.png",
                                                )
                                                plot_soft_l_scoring_maps(
                                                    l_shape_op.cached_soft_debug,
                                                    output_path=soft_scoring_path,
                                                    title_prefix=f"Soft L Scoring (iter={iteration})",
                                                )
                                                logging.info(
                                                    f"Soft L-shape scoring plot saved to {soft_scoring_path}"
                                                )
                                except Exception as e:
                                    logging.warning(f"Failed to plot L-shape density/segments: {e}")
                                
                            # exit(0)
                        # 定期更新Steiner树和L方向（每N次迭代）
                        elif model.use_l_shape_routability and (iteration % getattr(params, 'l_shape_update_interval', 10) == 0):
                            t_l_shape_update = time.time()
                            
                            # 重置L形segment缓存（EGR将重新运行）
                            if model.l_shape_routability_op is not None:
                                model.l_shape_routability_op.segment_builder.reset_cache()
                            
                            ggr_topology_mode = use_ggr_l_shape_topology(params)
                            gpugr_inputs = None
                            l_shape_inputs = None
                            if ggr_topology_mode:
                                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                    params,
                                    placedb,
                                    pos,
                                    model=model,
                                )
                                l_shape_inputs = gpugr_inputs
                                l_directions = _load_l_shape_topology_pack_from_gpugr(
                                    params,
                                    pos,
                                    self.op_collections.pin_pos_op,
                                    self.op_collections.steiner_topo_op,
                                    self.data_collections,
                                    l_shape_inputs,
                                    "l_shape_update.load_ggr_topology_pack",
                                    iteration,
                                )
                            else:
                                with profile_scope(params, "l_shape_update.rebuild_tree", tensor=pos, iteration=iteration):
                                    with torch.no_grad():
                                        pin_pos = self.op_collections.pin_pos_op(pos)
                                        if pin_pos.is_cuda:
                                            pin_pos = pin_pos.cpu()
                                        self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                            self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                            self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(pin_pos)
                                if not getattr(params, "l_direction_use_gpugr", False):
                                    l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                else:
                                    gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                    l_shape_inputs = gpugr_inputs

                            steiner_topo_op = self.op_collections.steiner_topo_op
                            if ggr_topology_mode:
                                l_directions = steiner_topo_op.edge_l_directions
                            elif _should_skip_resolver_l_direction_for_soft(params):
                                steiner_topo_op.edge_l_directions = None
                                l_directions = None
                                if l_shape_log_verbose(params) >= 2:
                                    logging.info(
                                        "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                                        "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                                        "will be refreshed."
                                    )
                            else:
                                l_directions = _resolve_l_directions_for_l_shape(
                                    params,
                                    placedb,
                                    pos,
                                    steiner_topo_op,
                                    gpugr_route_entries=(
                                        gpugr_inputs["route_entries"]
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                    gpugr_metrics=(
                                        gpugr_inputs["metrics"]
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                    gpugr_route_grid=(
                                        (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                )
                            if model.l_shape_routability_op is not None and l_shape_inputs is not None:
                                with profile_scope(params, "l_shape_update.update_targets", tensor=pos, iteration=iteration):
                                    model.l_shape_routability_op.update_targets(
                                        target_density=l_shape_inputs["supply_map"],
                                        target_demand=l_shape_inputs["demand_map"],
                                        raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                                        target_density_h=l_shape_inputs.get("supply_map_h"),
                                        target_density_v=l_shape_inputs.get("supply_map_v"),
                                        target_demand_h=l_shape_inputs.get("demand_map_h"),
                                        target_demand_v=l_shape_inputs.get("demand_map_v"),
                                        raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                                        raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                                        supply_original=l_shape_inputs.get("supply_original"),
                                        supply_original_h=l_shape_inputs.get("supply_original_h"),
                                        supply_original_v=l_shape_inputs.get("supply_original_v"),
                                        fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                                        fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                                        fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
                                    )
                                    model.l_shape_routability_op.update_same_net_topology(
                                        topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                                        topo_stats=l_shape_inputs.get("same_net_topo_stats"),
                                    )
                                if (
                                    getattr(model.l_shape_routability_op, "wire_width_h", None) is None
                                    and getattr(model.l_shape_routability_op, "wire_width_v", None) is None
                                ):
                                    updated_wire_width = float(l_shape_inputs["wire_width"])
                                    current_wire_width = float(model.l_shape_routability_op.wire_width)
                                    if abs(updated_wire_width - current_wire_width) > 1e-6:
                                        logging.info(
                                            "L-shape periodic target update kept existing wire_width %.4f while refreshed maps imply %.4f. "
                                            "Segment width is not updated online after initialization.",
                                            current_wire_width,
                                            updated_wire_width,
                                        )
                            density_map = None

                            # 基于L-shape overflow变化更新target_ratio（外环慢速更新）
                            if getattr(params, 'l_shape_overflow_update_flag', True):
                                try:
                                    with torch.no_grad():
                                        density_map = model.get_l_shape_density_map(
                                            pos, use_l_direction=True
                                        )
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_op = (
                                            getattr(l_shape_op, "overflow_op", None)
                                            if l_shape_op is not None
                                            else None
                                        )
                                        if density_map is not None and overflow_op is not None:
                                            l_shape_overflow = None
                                            overflow_ratio = None
                                            l_shape_max_density = None

                                            # 优先使用potential中的当前场源口径：
                                            # blockage_initial_density 主线走 track-space，
                                            # legacy residual 路径仍走 tracks + area_per_track。
                                            density_driver = getattr(l_shape_op, "density_op", None)
                                            if density_driver is not None:
                                                supply_map = getattr(
                                                    density_driver, "target_density", None
                                                )
                                                supply_map_h = getattr(
                                                    density_driver, "target_density_h", None
                                                )
                                                supply_map_v = getattr(
                                                    density_driver, "target_density_v", None
                                                )
                                                demand_map = getattr(
                                                    density_driver, "target_demand", None
                                                )
                                                demand_map_h = getattr(
                                                    density_driver, "target_demand_h", None
                                                )
                                                demand_map_v = getattr(
                                                    density_driver, "target_demand_v", None
                                                )
                                                density_map_h = getattr(
                                                    l_shape_op, "cached_density_map_h", None
                                                )
                                                density_map_v = getattr(
                                                    l_shape_op, "cached_density_map_v", None
                                                )
                                                if (
                                                    isinstance(supply_map, torch.Tensor)
                                                    and supply_map.dim() == 2
                                                ):
                                                    supply_map = supply_map.to(
                                                        density_map.device,
                                                        dtype=density_map.dtype,
                                                    )
                                                    blockage_initial_density = bool(
                                                        getattr(density_driver, "blockage_initial_density", False)
                                                    )
                                                    supply_original_map = getattr(
                                                        density_driver, "supply_original", None
                                                    )
                                                    supply_original_map_h = getattr(
                                                        density_driver, "supply_original_h", None
                                                    )
                                                    supply_original_map_v = getattr(
                                                        density_driver, "supply_original_v", None
                                                    )
                                                    fix_usage_map = getattr(
                                                        density_driver, "fix_usage_map", None
                                                    )
                                                    fix_usage_map_h = getattr(
                                                        density_driver, "fix_usage_map_h", None
                                                    )
                                                    fix_usage_map_v = getattr(
                                                        density_driver, "fix_usage_map_v", None
                                                    )

                                                    if (
                                                        blockage_initial_density
                                                        and isinstance(supply_original_map, torch.Tensor)
                                                        and isinstance(fix_usage_map, torch.Tensor)
                                                    ):
                                                        bin_area_local = float(
                                                            density_driver.bin_size_x
                                                            * density_driver.bin_size_y
                                                        )
                                                        split_available = (
                                                            isinstance(density_map_h, torch.Tensor)
                                                            and isinstance(density_map_v, torch.Tensor)
                                                            and isinstance(supply_original_map_h, torch.Tensor)
                                                            and isinstance(supply_original_map_v, torch.Tensor)
                                                            and isinstance(fix_usage_map_h, torch.Tensor)
                                                            and isinstance(fix_usage_map_v, torch.Tensor)
                                                        )
                                                        if split_available:
                                                            density_map_h = density_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_map_v = density_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            supply_original_map_h = supply_original_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            supply_original_map_v = supply_original_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map_h = fix_usage_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map_v = fix_usage_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_h_in_tracks = density_map_h / bin_area_local
                                                            density_v_in_tracks = density_map_v / bin_area_local
                                                            occupancy_h = density_h_in_tracks + fix_usage_map_h.clamp(min=0.0)
                                                            occupancy_v = density_v_in_tracks + fix_usage_map_v.clamp(min=0.0)
                                                            overflow_h_in_tracks = (
                                                                occupancy_h - supply_original_map_h
                                                            ).clamp(min=0.0)
                                                            overflow_v_in_tracks = (
                                                                occupancy_v - supply_original_map_v
                                                            ).clamp(min=0.0)
                                                            utilization_h = occupancy_h / supply_original_map_h.clamp(
                                                                min=1e-6
                                                            )
                                                            utilization_v = occupancy_v / supply_original_map_v.clamp(
                                                                min=1e-6
                                                            )
                                                            l_shape_overflow = float(
                                                                (
                                                                    overflow_h_in_tracks.sum()
                                                                    + overflow_v_in_tracks.sum()
                                                                ).item()
                                                            )
                                                            overflow_ratio = float(
                                                                (
                                                                    overflow_h_in_tracks.sum()
                                                                    + overflow_v_in_tracks.sum()
                                                                )
                                                                / (
                                                                    supply_original_map_h.sum()
                                                                    + supply_original_map_v.sum()
                                                                ).clamp(min=1e-12)
                                                            )
                                                            l_shape_max_density = float(
                                                                torch.maximum(
                                                                    utilization_h.max(),
                                                                    utilization_v.max(),
                                                                ).item()
                                                            )
                                                        else:
                                                            supply_original_map = supply_original_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map = fix_usage_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_in_tracks = density_map / bin_area_local
                                                            occupancy = density_in_tracks + fix_usage_map.clamp(min=0.0)
                                                            overflow_in_tracks = (
                                                                occupancy - supply_original_map
                                                            ).clamp(min=0.0)
                                                            utilization = occupancy / supply_original_map.clamp(
                                                                min=1e-6
                                                            )
                                                            l_shape_overflow = float(
                                                                overflow_in_tracks.sum().item()
                                                            )
                                                            overflow_ratio = float(
                                                                (
                                                                    overflow_in_tracks.sum()
                                                                    / supply_original_map.sum().clamp(min=1e-12)
                                                                ).item()
                                                            )
                                                            l_shape_max_density = float(
                                                                utilization.max().item()
                                                            )
                                                    else:
                                                        area_per_track_buf = getattr(
                                                            density_driver, "area_per_track", None
                                                        )
                                                        total_density = density_map.sum()
                                                        calibrated_area_per_track = None

                                                        if (
                                                            isinstance(demand_map, torch.Tensor)
                                                            and demand_map.dim() == 2
                                                        ):
                                                            demand_map = demand_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            total_demand = demand_map.sum()
                                                            if total_demand > 0 and total_density > 0:
                                                                if (
                                                                    isinstance(
                                                                        area_per_track_buf,
                                                                        torch.Tensor,
                                                                    )
                                                                    and area_per_track_buf.numel() == 1
                                                                ):
                                                                    if (
                                                                        float(
                                                                            area_per_track_buf.item()
                                                                        )
                                                                        <= 0
                                                                    ):
                                                                        area_per_track_buf.fill_(
                                                                            total_density / total_demand
                                                                        )
                                                                    calibrated_area_per_track = area_per_track_buf.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    calibrated_area_per_track = (
                                                                        total_density / total_demand
                                                                    )
                                                        else:
                                                            target_utilization = float(
                                                                getattr(
                                                                    params,
                                                                    "l_shape_target_utilization",
                                                                    0.8,
                                                                )
                                                            )
                                                            total_supply = supply_map.sum()
                                                            if total_supply > 0 and total_density > 0:
                                                                calibrated_area_per_track = total_density / (
                                                                    target_utilization
                                                                    * total_supply.clamp(min=1e-12)
                                                                )

                                                        if calibrated_area_per_track is not None:
                                                            split_available = (
                                                                isinstance(density_map_h, torch.Tensor)
                                                                and isinstance(density_map_v, torch.Tensor)
                                                                and isinstance(supply_map_h, torch.Tensor)
                                                                and isinstance(supply_map_v, torch.Tensor)
                                                            )
                                                            if split_available:
                                                                density_map_h = density_map_h.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                density_map_v = density_map_v.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                supply_map_h = supply_map_h.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                supply_map_v = supply_map_v.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                if (
                                                                    isinstance(demand_map_h, torch.Tensor)
                                                                    and demand_map_h.dim() == 2
                                                                ):
                                                                    demand_map_h = demand_map_h.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    demand_map_h = None
                                                                if (
                                                                    isinstance(demand_map_v, torch.Tensor)
                                                                    and demand_map_v.dim() == 2
                                                                ):
                                                                    demand_map_v = demand_map_v.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    demand_map_v = None

                                                                calibrated_area_per_track_h = calibrated_area_per_track
                                                                calibrated_area_per_track_v = calibrated_area_per_track
                                                                if demand_map_h is not None:
                                                                    total_density_h = density_map_h.sum()
                                                                    total_demand_h = demand_map_h.sum()
                                                                    if total_density_h > 0 and total_demand_h > 0:
                                                                        calibrated_area_per_track_h = (
                                                                            total_density_h / total_demand_h
                                                                        )
                                                                if demand_map_v is not None:
                                                                    total_density_v = density_map_v.sum()
                                                                    total_demand_v = demand_map_v.sum()
                                                                    if total_density_v > 0 and total_demand_v > 0:
                                                                        calibrated_area_per_track_v = (
                                                                            total_density_v / total_demand_v
                                                                        )

                                                                demand_h_in_tracks = (
                                                                    density_map_h
                                                                    / calibrated_area_per_track_h
                                                                )
                                                                demand_v_in_tracks = (
                                                                    density_map_v
                                                                    / calibrated_area_per_track_v
                                                                )
                                                                overflow_h_in_tracks = (
                                                                    demand_h_in_tracks - supply_map_h
                                                                ).clamp(min=0.0)
                                                                overflow_v_in_tracks = (
                                                                    demand_v_in_tracks - supply_map_v
                                                                ).clamp(min=0.0)
                                                                utilization_h = demand_h_in_tracks / supply_map_h.clamp(
                                                                    min=1e-6
                                                                )
                                                                utilization_v = demand_v_in_tracks / supply_map_v.clamp(
                                                                    min=1e-6
                                                                )
                                                                overflow_area = (
                                                                    overflow_h_in_tracks
                                                                    * calibrated_area_per_track_h
                                                                    + overflow_v_in_tracks
                                                                    * calibrated_area_per_track_v
                                                                )
                                                                l_shape_overflow = float(
                                                                    overflow_area.sum().item()
                                                                )
                                                                overflow_ratio = float(
                                                                    (
                                                                        overflow_h_in_tracks.sum()
                                                                        + overflow_v_in_tracks.sum()
                                                                    )
                                                                    / (
                                                                        supply_map_h.sum()
                                                                        + supply_map_v.sum()
                                                                    ).clamp(min=1e-12)
                                                                )
                                                                l_shape_max_density = float(
                                                                    torch.maximum(
                                                                        utilization_h.max(),
                                                                        utilization_v.max(),
                                                                    ).item()
                                                                )
                                                            else:
                                                                demand_in_tracks = (
                                                                    density_map
                                                                    / calibrated_area_per_track
                                                                )
                                                                overflow_in_tracks = (
                                                                    demand_in_tracks - supply_map
                                                                ).clamp(min=0.0)
                                                                utilization = demand_in_tracks / supply_map.clamp(
                                                                    min=1e-6
                                                                )
                                                                l_shape_overflow = float(
                                                                    (
                                                                        overflow_in_tracks
                                                                        * calibrated_area_per_track
                                                                    )
                                                                    .sum()
                                                                    .item()
                                                                )
                                                                overflow_ratio = float(
                                                                    (
                                                                        overflow_in_tracks.sum()
                                                                        / supply_map.sum().clamp(min=1e-12)
                                                                    )
                                                                    .item()
                                                                )
                                                                l_shape_max_density = float(
                                                                    utilization.max().item()
                                                                )

                                            # 若potential口径不可用，退化为overflow_op口径
                                            if (
                                                l_shape_overflow is None
                                                or overflow_ratio is None
                                                or l_shape_max_density is None
                                            ):
                                                cached_segments = getattr(
                                                    l_shape_op, "cached_segments", None
                                                )
                                                if (
                                                    cached_segments is not None
                                                    and int(
                                                        cached_segments.get(
                                                            "num_segments", 0
                                                        )
                                                    )
                                                    > 0
                                                ):
                                                    seg_pos = cached_segments.get(
                                                        "segment_pos", None
                                                    )
                                                    seg_size_x = cached_segments.get(
                                                        "segment_size_x", None
                                                    )
                                                    seg_size_y = cached_segments.get(
                                                        "segment_size_y", None
                                                    )
                                                    if (
                                                        seg_pos is not None
                                                        and seg_size_x is not None
                                                        and seg_size_y is not None
                                                    ):
                                                        ov_cost, ov_max_density = overflow_op(
                                                            seg_pos,
                                                            seg_size_x,
                                                            seg_size_y,
                                                        )
                                                        l_shape_overflow = float(
                                                            ov_cost.item()
                                                        )
                                                        l_shape_max_density = float(
                                                            ov_max_density.item()
                                                        )

                                                bin_area = float(
                                                    overflow_op.bin_size_x
                                                    * overflow_op.bin_size_y
                                                )
                                                target_density = overflow_op.target_density
                                                if isinstance(target_density, torch.Tensor):
                                                    target_total = float(
                                                        (
                                                            target_density.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            * bin_area
                                                        )
                                                        .sum()
                                                        .item()
                                                    )
                                                else:
                                                    target_total = float(
                                                        float(target_density)
                                                        * bin_area
                                                        * overflow_op.num_bins_x
                                                        * overflow_op.num_bins_y
                                                    )
                                                overflow_ratio = float(
                                                    l_shape_overflow / (target_total + 1e-12)
                                                )
                                            model.l_shape_overflow = l_shape_overflow
                                            model.l_shape_overflow_ratio = overflow_ratio
                                            model.l_shape_overflow_max_density = (
                                                l_shape_max_density
                                            )
                                            cur_metric.l_shape_overflow = l_shape_overflow
                                            cur_metric.l_shape_overflow_ratio = overflow_ratio
                                            cur_metric.l_shape_overflow_max_density = (
                                                l_shape_max_density
                                            )
                                            if l_shape_log_verbose(params) >= 1:
                                                logging.info(
                                                    "L-shape refresh iter=%d: "
                                                    "ov_raw=%.6e, ov_ratio=%.6e, max_density=%.6f",
                                                    iteration,
                                                    l_shape_overflow,
                                                    overflow_ratio,
                                                    l_shape_max_density,
                                                )

                                            beta = float(
                                                getattr(
                                                    params,
                                                    "l_shape_overflow_ema_beta",
                                                    0.8,
                                                )
                                            )
                                            beta = max(0.0, min(0.999, beta))
                                            if (
                                                not hasattr(model, "_l_shape_overflow_ema")
                                                or model._l_shape_overflow_ema is None
                                            ):
                                                model._l_shape_overflow_ema = overflow_ratio
                                                model._l_shape_overflow_last = overflow_ratio
                                                if l_shape_log_verbose(params) >= 1:
                                                    logging.info(
                                                        "L-shape overflow outer-loop initialized: "
                                                        "ov_raw=%.6e, ov_ratio=%.6e, target_ratio=%.4f",
                                                        l_shape_overflow,
                                                        overflow_ratio,
                                                        model.l_shape_grad_target_ratio,
                                                    )
                                            else:
                                                prev_ema = float(
                                                    model._l_shape_overflow_ema
                                                )
                                                ema = beta * prev_ema + (1.0 - beta) * overflow_ratio
                                                delta = ema - prev_ema
                                                model._l_shape_overflow_last = overflow_ratio
                                                model._l_shape_overflow_ema = ema

                                                deadband = float(
                                                    getattr(
                                                        params,
                                                        "l_shape_overflow_deadband",
                                                        1e-4,
                                                    )
                                                )
                                                if abs(delta) > deadband:
                                                    k = float(
                                                        getattr(
                                                            params,
                                                            "l_shape_overflow_update_k",
                                                            2.0,
                                                        )
                                                    )
                                                    old_ratio = float(
                                                        model.l_shape_grad_target_ratio
                                                    )
                                                    ratio_min = float(
                                                        getattr(
                                                            params,
                                                            "l_shape_grad_target_ratio_min",
                                                            0.05,
                                                        )
                                                    )
                                                    ratio_max = float(
                                                        getattr(
                                                            params,
                                                            "l_shape_grad_target_ratio_max",
                                                            0.2,
                                                        )
                                                    )
                                                    if ratio_min > ratio_max:
                                                        ratio_min, ratio_max = ratio_max, ratio_min
                                                    new_ratio = old_ratio * math.exp(k * delta)
                                                    new_ratio = max(
                                                        ratio_min, min(ratio_max, new_ratio)
                                                    )
                                                    model.l_shape_grad_target_ratio = new_ratio
                                                    if l_shape_log_verbose(params) >= 1:
                                                        logging.info(
                                                            "L-shape overflow outer-loop iter=%d: "
                                                            "ov_raw=%.6e, ov_ratio=%.6e, ov_ema %.6e->%.6e, delta=%.3e, "
                                                            "target_ratio %.4f->%.4f",
                                                            iteration,
                                                            l_shape_overflow,
                                                            overflow_ratio,
                                                            prev_ema,
                                                            ema,
                                                            delta,
                                                            old_ratio,
                                                            new_ratio,
                                                        )
                                                else:
                                                    logging.debug(
                                                        "L-shape overflow outer-loop iter=%d: "
                                                        "ov_raw=%.6e, ov_ratio=%.6e, ov_ema %.6e->%.6e, delta=%.3e "
                                                        "(deadband=%.3e), target_ratio=%.4f",
                                                        iteration,
                                                        l_shape_overflow,
                                                        overflow_ratio,
                                                        prev_ema,
                                                        ema,
                                                        delta,
                                                        deadband,
                                                        model.l_shape_grad_target_ratio,
                                                    )
                                            maybe_auto_disable_l_shape(
                                                iteration, outer_update=True
                                            )
                                except Exception as e:
                                    logging.warning(
                                        f"L-shape overflow outer-loop update failed at iter {iteration}: {e}"
                                    )
                            
                            logging.debug(f"L-shape routability updated at iteration {iteration}, "
                                         f"time={((time.time() - t_l_shape_update) * 1000):.2f}ms")
                            
                            # 定期可视化L形密度图和segments
                            if params.l_shape_plot_flag:
                                try:
                                    from dreamplace.ops.routability.l_shape_routability import (
                                        plot_l_shape_electric_overflow_map,
                                        plot_l_shape_initial_density_map,
                                        plot_l_shape_macro_source_maps,
                                        plot_l_shape_electric_potential_map,
                                        plot_l_shape_supply_maps,
                                        plot_l_shape_true_source_maps,
                                        plot_segment_density_map,
                                        plot_soft_l_intermediate,
                                        plot_soft_l_scoring_maps,
                                    )
                                    
                                    if density_map is None:
                                        forward_source_snapshot = _snapshot_l_shape_forward_source()
                                        density_map = model.get_l_shape_density_map(
                                            pos, use_l_direction=True
                                        )
                                        _restore_l_shape_forward_source(forward_source_snapshot)
                                    if density_map is not None:
                                        density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_density_iter{iteration}.png"
                                        )
                                        plot_segment_density_map(
                                            density_map, density_plot_path,
                                            title=f"L-shape Density (iter={iteration})",
                                            colormap="binary"
                                        )
                                    
                                    # 绘制基于 electric potential 的 overflow map
                                    if model.l_shape_routability_op is not None and \
                                       model.l_shape_routability_op.cached_segments is not None:
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_overflow_map(
                                            l_shape_op,
                                            output_path=overflow_plot_path,
                                            title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                                        )
                                        supply_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_supply_iter{iteration}.png"
                                        )
                                        plot_l_shape_supply_maps(
                                            l_shape_op,
                                            output_path=supply_plot_path,
                                            title_prefix=f"L-shape Supply Debug (iter={iteration})",
                                        )
                                        initial_density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                                        )
                                        plot_l_shape_initial_density_map(
                                            l_shape_op,
                                            output_path=initial_density_plot_path,
                                            title_prefix=f"L-shape Initial Density (iter={iteration})",
                                        )
                                        source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_true_source_maps(
                                            l_shape_op,
                                            output_path=source_plot_path,
                                            title_prefix=f"L-shape True Source (iter={iteration})",
                                        )
                                        macro_source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_macro_source_maps(
                                            l_shape_op,
                                            output_path=macro_source_plot_path,
                                            title_prefix=f"L-shape Macro Source (iter={iteration})",
                                        )
                                        potential_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_potential_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_potential_map(
                                            l_shape_op,
                                            output_path=potential_plot_path,
                                            title_prefix=f"L-shape Electric Potential (iter={iteration})",
                                        )
                                        if getattr(l_shape_op, "soft_l_assignment", False) and \
                                           'soft_l_weights' in l_shape_op.cached_segments:
                                            soft_plot_path = os.path.join(
                                                params.result_dir, f"l_shape_soft_iter{iteration}.png"
                                            )
                                            plot_soft_l_intermediate(
                                                l_shape_op.cached_segments,
                                                output_path=soft_plot_path,
                                            )
                                            if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                                                soft_scoring_path = os.path.join(
                                                    params.result_dir,
                                                    f"l_shape_soft_scoring_iter{iteration}.png",
                                                )
                                                plot_soft_l_scoring_maps(
                                                    l_shape_op.cached_soft_debug,
                                                    output_path=soft_scoring_path,
                                                    title_prefix=f"Soft L Scoring (iter={iteration})",
                                                )
                                except Exception as e:
                                    logging.warning(f"Failed to plot L-shape density/segments: {e}")
                    # ======================================================

                    # plot placement
                    if params.plot_flag and (iteration % 30 == 0 or iteration == 999):
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    # stop updating fence regions that are marked stop, exclude the outer cell !
                    t3 = time.time()
                    if model.update_mask is not None:
                        pos_bk = pos.data.clone()
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        optimizer.step()

                        for region_id, fence_region_update_flag in enumerate[Any](model.update_mask):
                            if fence_region_update_flag == 0:
                                # don't update cell location in that region
                                mask = self.op_collections.fence_region_density_ops[region_id].pos_mask
                                pos.data.masked_scatter_(mask, pos_bk[mask])
                    else:
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        optimizer.step()

                    logging.info("optimizer step %.3f ms" %
                                 ((time.time() - t3) * 1000))

                    # Perform timing-opt.
                    if params.global_place_flag and params.timing_opt_flag and \
                            params.enable_net_weighting and \
                            iteration > 500 and iteration % 15 == 0:
                        # Take the timing operator from the operator collections.
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        # The timing operator has already integrated timer as its
                        # instance variable, so it only takes one argument.
                        timing_op(self.pos[0].data.clone().cpu())
                        timing_op.timer.update_timing()
                        npaths = max(1, int(placedb.num_nets * 0.03))

                        # Report timing step.
                        # Temporary solution: modify net weights
                        beg = time.time()
                        timing_op.update_net_weights(
                            max_net_weight=placedb.max_net_weight,
                            n=npaths)
                        if self.device != torch.device("cpu"):
                            # Copy weights from placedb.net_weights to device.
                            self.data_collections.net_weights.copy_(
                                torch.from_numpy(placedb.net_weights))
                        logging.info("net-weight update step %.3f ms" %
                                     ((time.time() - beg) * 1000))

                        # Report tns and wns in each timing feedback call.
                        # Note that OpenTimer considers early,late,rise,fall for tns/wns.
                        # The following values are for reference.
                        cur_metric.tns = timing_op.timer.report_tns_elw(
                            split=1) / (time_unit * 1e17)
                        cur_metric.wns = timing_op.timer.report_wns(
                            split=1) / (time_unit * 1e15)

                    # nesterov has already computed the objective of the next step
                    if optimizer_name.lower() == "nesterov":
                        cur_metric.objective = optimizer.param_groups[0]["obj_k_1"][0].data.clone(
                        )

                    maybe_auto_disable_l_shape(iteration, outer_update=False)
                    maybe_disable_l_shape_by_ratio(iteration)
                    attach_l_shape_telemetry(cur_metric)
                    # actually reports the metric before step
                    logging.info(cur_metric)
                    # record the best outer cell overflow
                    if best_metric[0] is None or best_metric[0].overflow[-1] > cur_metric.overflow[-1]:
                        best_metric[0] = cur_metric
                        if best_pos[0] is None:
                            best_pos[0] = self.pos[0].data.clone()
                        else:
                            best_pos[0].data.copy_(self.pos[0].data)

                    logging.info("full step %.3f ms" %
                                 ((time.time() - t0) * 1000))

                def check_plateau(x, window=10, threshold=0.001):
                    if len(x) < window:
                        return False
                    x = x[-window:]
                    return (np.max(x) - np.min(x)) / np.mean(x) < threshold

                def check_divergence(x, window=50, threshold=0.05):
                    if len(x) < window or best_metric[0] is None:
                        return False
                    x = np.array(x[-window:])
                    overflow_mean = np.mean(x[:, 1])
                    overflow_diff = np.maximum(0, np.sign(
                        x[1:, 1] - x[:-1, 1])).astype(np.float32)
                    overflow_diff = np.sum(
                        overflow_diff) / overflow_diff.shape[0]
                    overflow_range = np.max(x[:, 1]) - np.min(x[:, 1])
                    wl_mean = np.mean(x[:, 0])
                    wl_ratio, overflow_ratio = (wl_mean - best_metric[0].hpwl.item()) / best_metric[
                        0
                    ].hpwl.item(), (
                        overflow_mean -
                        max(params.stop_overflow,
                            best_metric[0].overflow.item())
                    ) / best_metric[
                        0
                    ].overflow.item()
                    if wl_ratio > threshold * 1.2:
                        # this condition is not suitable for routability-driven opt with cell inflation
                        if (not params.routability_opt_flag) and overflow_ratio > threshold:
                            logging.warning(
                                f"Divergence detected: overflow increases too much than best overflow ({overflow_ratio:.4f} > {threshold:.4f})"
                            )
                            return True
                        elif overflow_range / overflow_mean < threshold:
                            logging.warning(
                                f"Divergence detected: overflow plateau ({overflow_range/overflow_mean:.4f} < {threshold:.4f})"
                            )
                            return True
                        elif overflow_diff > 0.6:
                            logging.warning(
                                f"Divergence detected: overflow fluctuate too frequently ({overflow_diff:.2f} > 0.6)"
                            )
                            return True
                        else:
                            return False
                    else:
                        return False

                def entropy_injection(
                    pos, placedb, shrink_factor=1, noise_intensity=1, mode="random", iteration=1
                ):
                    if mode == "random":
                        # print(pos[: placedb.num_movable_nodes].mean())
                        xc = pos[: placedb.num_movable_nodes].data.mean()
                        yc = pos.data[
                            placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                        ].mean()
                        num_movable_nodes = placedb.num_movable_nodes
                        num_nodes = placedb.num_nodes
                        num_filler_nodes = placedb.num_filler_nodes
                        num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes

                        fixed_pos_x = pos.data[
                            num_movable_nodes: num_movable_nodes + num_fixed_nodes
                        ].clone()
                        fixed_pos_y = pos.data[
                            num_nodes + num_movable_nodes: num_nodes + num_movable_nodes + num_fixed_nodes
                        ].clone()
                        if shrink_factor != 1:
                            pos.data[:num_nodes] = (
                                pos.data[:num_nodes] - xc) * shrink_factor + xc
                            pos.data[num_nodes:] = (
                                pos.data[num_nodes:] - yc) * shrink_factor + yc
                        if noise_intensity > 0.01:
                            # pos.data.add_(noise_intensity * torch.rand(num_nodes*2, device=pos.device).sub_(0.5))
                            pos.data.add_(
                                noise_intensity * torch.randn(num_nodes * 2, device=pos.device))

                        pos.data[num_movable_nodes: num_movable_nodes +
                                 num_fixed_nodes] = fixed_pos_x
                        pos.data[
                            num_nodes + num_movable_nodes: num_nodes + num_movable_nodes + num_fixed_nodes
                        ] = fixed_pos_y
                        # print(pos[: placedb.num_movable_nodes].mean())
                    else:
                        raise NotImplementedError

                Lgamma_metrics = all_metrics

                if params.routability_opt_flag:
                    _ensure_modularity_inflation_contract(params)
                    adjust_area_flag = True
                    adjust_route_area_flag = (
                        getattr(params, "adjust_gpugr_area_flag", False)
                        or params.adjust_nctugr_area_flag
                        or params.adjust_rudy_area_flag
                    )
                    adjust_pin_area_flag = params.adjust_pin_area_flag
                    num_area_adjust = 0
                    max_area_adjust_rounds = enhanced_inflation_controller.get_inflation_round_limit(params)
                    if getattr(model, "inflation_state", None) is not None:
                        model.inflation_state.num_area_adjust = 0
                    if inflation_legalization.is_enabled(params):
                        logging.info(
                            "Ordinary inflation will trigger at stop_overflow=%.6g "
                            "and legalize physical cell geometry before every round",
                            inflation_legalization.legacy_trigger_threshold(params),
                        )

                Llambda_flat_iteration = 0

                # L-shape routability 状态初始化
                model.enable_l_shape_routability = False
                reset_l_shape_auto_disable_state()
                reset_l_shape_reenable_state()

                # preparation for self-adaptive divergence check
                overflow_list = [1]
                divergence_list = []
                min_perturb_interval = 50
                stop_placement = 0
                last_perturb_iter = -min_perturb_interval
                perturb_counter = 0

                for Lgamma_step in range(model.Lgamma_iteration):
                    Lgamma_metrics.append([])
                    Llambda_metrics = Lgamma_metrics[-1]
                    inflation_applied_this_gamma = False
                    for Llambda_density_weight_step in range(model.Llambda_density_weight_iteration):
                        Llambda_metrics.append([])
                        Lsub_metrics = Llambda_metrics[-1]
                        for Lsub_step in range(model.Lsub_iteration):
                            # divergence threshold should decrease as overflow decreases
                            # only detect divergence when overflow is relatively low but not too low
                            div_flag = check_divergence(
                                # sometimes maybe too aggressive...
                                divergence_list, window=50, threshold=overflow_list[-1])
                            if params.timing_opt_flag:
                                # currently do not check divergence in timing-driven placement
                                # TODO: a better way for divergence detection and roll-back for tdp.
                                div_flag = False
                            if (
                                len(placedb.regions) == 0
                                and params.stop_overflow * 1.1 < overflow_list[-1] < params.stop_overflow * 4
                                and div_flag
                            ):
                                self.pos[0].data.copy_(best_pos[0].data)
                                stop_placement = 1

                                logging.error(
                                    "possible DIVERGENCE detected, roll back to the best position recorded"
                                )

                            one_descent_step(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, Lsub_metrics
                            )

                            if len(placedb.regions) == 0:
                                overflow_list.append(
                                    Llambda_metrics[-1][-1].overflow.data.item())
                                divergence_list.append(
                                    [
                                        Llambda_metrics[-1][-1].hpwl.data.item(),
                                        Llambda_metrics[-1][-1].overflow.data.item(),
                                    ]
                                )

                            # quadratic penalty and entropy injection
                            # This heuristics makes placement unstable
                            if (
                                len(placedb.regions) == 0
                                and iteration - last_perturb_iter > min_perturb_interval
                                and check_plateau(overflow_list, window=15, threshold=0.001)
                            ):
                                if overflow_list[-1] > 0.9:  # stuck at high overflow
                                    model.quad_penalty = True
                                    model.density_factor *= 2
                                    logging.info(
                                        f"Stuck at early stage. Turn on quadratic penalty with double density factor to accelerate convergence"
                                    )
                                    # stuck at very high overflow
                                    if overflow_list[-1] > 0.95:
                                        noise_intensity = min(
                                            max(40 + (120 - 40) *
                                                (overflow_list[-1] - 0.95) * 10, 40), 90
                                        )
                                        entropy_injection(
                                            self.pos[0],
                                            placedb,
                                            shrink_factor=0.996,
                                            noise_intensity=noise_intensity,
                                            mode="random",
                                        )
                                        logging.info(
                                            f"Stuck at very early stage. Turn on entropy injection with noise intensity = {noise_intensity} to help convergence"
                                        )
                                    last_perturb_iter = iteration
                                    perturb_counter += 1

                            iteration += 1
                            # stopping criteria
                            if Lsub_stop_criterion(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, Lsub_metrics
                            ):
                                break
                        Llambda_flat_iteration += 1

                        # update density weight
                        if Llambda_flat_iteration > 1:
                            model.op_collections.update_density_weight_op(
                                Llambda_metrics[-1][-1],
                                Llambda_metrics[-2][-1]
                                if len(Llambda_metrics) > 1
                                else Lgamma_metrics[-2][-1][-1],
                                Llambda_flat_iteration,
                            )
                        # logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        llambda_should_stop = Llambda_stop_criterion(
                            Lgamma_step,
                            Llambda_density_weight_step,
                            Llambda_metrics,
                        )
                        defer_stop_for_inflation = (
                            params.routability_opt_flag
                            and inflation_legalization.is_enabled(params)
                            and inflation_legalization.should_trigger_legacy_inflation(
                                params,
                                num_area_adjust,
                                max_area_adjust_rounds,
                                Llambda_metrics[-1][-1].overflow,
                            )
                        )
                        if llambda_should_stop and not defer_stop_for_inflation:
                            break

                        # for routability optimization
                        if params.routability_opt_flag:
                            trigger_enhanced_inflation = enhanced_inflation_controller.should_trigger_enhanced_inflation(
                                params,
                                num_area_adjust=num_area_adjust,
                                overflow=Llambda_metrics[-1][-1].overflow,
                            )
                            trigger_legacy_inflation = inflation_legalization.should_trigger_legacy_inflation(
                                params,
                                num_area_adjust,
                                max_area_adjust_rounds,
                                Llambda_metrics[-1][-1].overflow,
                                trigger_enhanced_inflation=trigger_enhanced_inflation,
                            )
                            if trigger_enhanced_inflation or trigger_legacy_inflation:
                                use_enhanced_inflation = bool(trigger_enhanced_inflation)
                                round_flags = (
                                    enhanced_inflation_controller.get_area_adjust_flags(params)
                                    if use_enhanced_inflation
                                    else {
                                        "adjust_area_flag": adjust_area_flag,
                                        "adjust_route_area_flag": adjust_route_area_flag,
                                        "adjust_pin_area_flag": adjust_pin_area_flag,
                                    }
                                )
                                round_adjust_area_flag = round_flags["adjust_area_flag"]
                                round_adjust_route_area_flag = round_flags["adjust_route_area_flag"]
                                round_adjust_pin_area_flag = round_flags["adjust_pin_area_flag"]
                                content = (
                                    "routability optimization round %d: adjust area flags = (%d, %d, %d)"
                                    % (
                                        num_area_adjust,
                                        round_adjust_area_flag,
                                        round_adjust_route_area_flag,
                                        round_adjust_pin_area_flag,
                                    )
                                )
                                pos = model.data_collections.pos[0]
                                if use_enhanced_inflation and best_pos[0] is not None:
                                    pos.data.copy_(best_pos[0].data)
                                    content = (
                                        "enhanced inflation round %d: restore best_pos snapshot before gpugr/area-adjust | "
                                        % num_area_adjust
                                    ) + content
                                    logging.info(
                                        "enhanced inflation round %d uses best_pos snapshot with best overflow %.6f at iter %d",
                                        num_area_adjust,
                                        float(best_metric[0].overflow[-1]) if best_metric[0] is not None else float("nan"),
                                        int(best_metric[0].iteration) if best_metric[0] is not None else -1,
                                    )
                                elif use_enhanced_inflation:
                                    logging.info(
                                        "enhanced inflation round %d cannot find an earlier best_pos snapshot; use current position as trigger input",
                                        num_area_adjust,
                                    )
                                route_map_source = "none"
                                if round_adjust_route_area_flag:
                                    route_map_source = _resolve_route_map_source(params)
                                if round_adjust_route_area_flag:
                                    _ensure_modularity_inflation_contract(
                                        params,
                                        route_map_source=route_map_source,
                                    )
                                current_inflation_round = None
                                if getattr(model, "inflation_state", None) is not None:
                                    enhanced_inflation_controller.maybe_capture_model_density_state(
                                        model.inflation_state,
                                        model,
                                    )
                                    current_inflation_round = enhanced_inflation_controller.begin_inflation_round(
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        pos=pos,
                                        round_idx=num_area_adjust,
                                        stage_idx=cur_stage,
                                        iteration=iteration,
                                        overflow=Llambda_metrics[-1][-1].overflow,
                                        route_map_source=route_map_source,
                                        adjust_area_flag=round_adjust_area_flag,
                                        adjust_route_area_flag=round_adjust_route_area_flag,
                                        adjust_pin_area_flag=round_adjust_pin_area_flag,
                                        notes=(
                                            "enhanced_inflation_best_pos"
                                            if use_enhanced_inflation
                                            else (
                                                "legacy_area_adjust_after_legalization"
                                                if inflation_legalization.is_enabled(params)
                                                else "legacy_area_adjust"
                                            )
                                        ),
                                    )
                                physical_geometry_backup = None
                                if inflation_legalization.is_enabled(params):
                                    physical_geometry_backup = (
                                        inflation_legalization.use_physical_geometry(
                                            pos,
                                            self.data_collections,
                                        )
                                    )
                                    physical_pre_legalization_pos = (
                                        pos.detach().clone()
                                    )
                                    _log_inflation_macro_overlap(
                                        placedb,
                                        pos,
                                        self.data_collections,
                                        num_area_adjust,
                                        "after_gp_before_legalization",
                                    )
                                    legalization_start = time.time()
                                    pos.data.copy_(
                                        self.op_collections.legalize_op(pos)
                                    )
                                    logging.info(
                                        "Inflation round %d physical legalization "
                                        "takes %.3f seconds",
                                        num_area_adjust,
                                        time.time() - legalization_start,
                                    )
                                    _log_inflation_macro_overlap(
                                        placedb,
                                        pos,
                                        self.data_collections,
                                        num_area_adjust,
                                        "after_legalization",
                                        reference_pos=physical_pre_legalization_pos,
                                    )
                                _run_gpugr_before_first_area_adjust_and_exit(
                                    params,
                                    placedb,
                                    pos,
                                    num_area_adjust,
                                    model,
                                )
                                if round_adjust_route_area_flag:
                                    _ensure_modularity_active_clusters(
                                        params,
                                        placedb,
                                        model,
                                        pos,
                                        num_area_adjust,
                                    )

                                route_utilization_map = None
                                pin_utilization_map = None
                                modularity_maps = None
                                gpugr_metrics = {}
                                low_util_context = None
                                fixed_target_area = None
                                if round_adjust_route_area_flag:
                                    if route_map_source == "gpugr":
                                        _sync_gpugr_route_grid_to_autodmp(
                                            params,
                                            placedb,
                                            model=model,
                                        )
                                        route_utilization_map = model.op_collections.gpugr_congestion_map_op(
                                            pos
                                        )
                                        gpugr_metrics = getattr(
                                            model.op_collections.gpugr_congestion_map_op,
                                            "last_metrics",
                                            {},
                                        ) or {}
                                        if _is_modularity_inflation_enabled(params):
                                            modularity_maps = _prepare_modularity_maps_from_gpugr(
                                                placedb,
                                                model.op_collections.gpugr_congestion_map_op,
                                                pos,
                                                eps=getattr(params, "modularity_active_bin_overflow_eps", 1e-6),
                                            )
                                    elif route_map_source == "irt_egr":
                                        route_utilization_map = model.op_collections.irt_egr_congestion_map_op(
                                            pos, stage="egr3D", resolve_congestion="high")
                                    else:
                                        route_utilization_map = model.op_collections.route_utilization_map_op(
                                            pos)
                                if physical_geometry_backup is not None:
                                    inflation_legalization.restore_inflated_geometry(
                                        pos,
                                        self.data_collections,
                                        physical_geometry_backup,
                                    )
                                    logging.info(
                                        "Inflation round %d restored cumulative "
                                        "inflated geometry after congestion estimation",
                                        num_area_adjust,
                                    )
                                if round_adjust_pin_area_flag:
                                    pin_utilization_map = model.op_collections.pin_utilization_map_op(
                                        pos)
                                    if params.plot_flag:
                                        path = "%s/%s" % (params.result_dir,
                                                          params.design_name())
                                        figname = "%s/plot/pin%d.png" % (
                                            path, num_area_adjust)
                                        os.system("mkdir -p %s" %
                                                  (os.path.dirname(figname)))
                                        plt.imsave(
                                            figname, pin_utilization_map.data.cpu().numpy().T, origin="lower"
                                        )
                                if getattr(model, "inflation_state", None) is not None:
                                    if use_enhanced_inflation:
                                        fixed_target_area = getattr(
                                            model.inflation_state,
                                            "target_area",
                                            None,
                                        )
                                    low_util_context = enhanced_inflation_controller.prepare_low_util_inflation(
                                        params,
                                        model.inflation_state,
                                        placedb,
                                        self.data_collections,
                                        model.op_collections.adjust_node_area_op,
                                        gpugr_metrics,
                                    )
                                (
                                    adjust_area_flag,
                                    adjust_route_area_flag,
                                    adjust_pin_area_flag,
                                ) = model.op_collections.adjust_node_area_op(
                                    pos,
                                    route_utilization_map,
                                    pin_utilization_map,
                                    modularity_maps=modularity_maps,
                                    inflation_round=int(num_area_adjust),
                                    fixed_target_area=fixed_target_area,
                                )
                                enhanced_inflation_controller.restore_low_util_inflation(
                                    model.op_collections.adjust_node_area_op,
                                    low_util_context,
                                )
                                low_util_metrics = {}
                                if adjust_area_flag:
                                    low_util_metrics = enhanced_inflation_controller.apply_low_util_target_density(
                                        params,
                                        getattr(model, "inflation_state", None),
                                        self.data_collections,
                                        placedb,
                                        pos,
                                        low_util_context,
                                    )
                                min_area_inc_metrics = {}
                                if (
                                    adjust_area_flag
                                    and current_inflation_round is not None
                                    and use_enhanced_inflation
                                ):
                                    min_area_inc_result = enhanced_inflation_controller.enforce_min_area_increment(
                                        params,
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        pos,
                                    )
                                    movable_area_increment_ratio = min_area_inc_result.get(
                                        "movable_area_increment_ratio"
                                    )
                                    if movable_area_increment_ratio is not None:
                                        min_area_inc_metrics = {
                                            "enhanced_movable_area_increment_ratio": float(
                                                movable_area_increment_ratio
                                            ),
                                            "enhanced_min_area_increment_threshold": float(
                                                min_area_inc_result[
                                                    "min_area_increment_threshold"
                                                ]
                                            ),
                                        }
                                    if min_area_inc_result.get("triggered"):
                                        low_util_metrics = {}
                                        adjust_area_flag = False
                                        adjust_route_area_flag = False
                                        adjust_pin_area_flag = False
                                content += " -> (%d, %d, %d)" % (
                                    adjust_area_flag,
                                    adjust_route_area_flag,
                                    adjust_pin_area_flag,
                                )
                                logging.info(content)
                                if current_inflation_round is not None:
                                    round_status = "applied" if adjust_area_flag else "stopped"
                                    enhanced_inflation_controller.finish_inflation_round(
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        adjust_area_flag=adjust_area_flag,
                                        adjust_route_area_flag=adjust_route_area_flag,
                                        adjust_pin_area_flag=adjust_pin_area_flag,
                                        status=round_status,
                                        pos=pos,
                                        gr_metrics=dict(
                                            {
                                                "placement_overflow": Llambda_metrics[-1][-1].overflow,
                                            },
                                            **min_area_inc_metrics,
                                            **low_util_metrics,
                                            **gpugr_metrics,
                                        ),
                                    )
                                    selected_metric_name = getattr(
                                        params,
                                        "enhanced_inflation_select_metric",
                                        "est_shorts",
                                    )
                                    round_metric_value = enhanced_inflation_controller.get_round_metric(
                                        model.inflation_state.round_records[-1],
                                        selected_metric_name,
                                    )
                                    if round_metric_value is not None:
                                        logging.info(
                                            "Recorded %s inflation round %d: %s=%.4f trigger_overflow=%.6f",
                                            "enhanced"
                                            if use_enhanced_inflation
                                            else "ordinary",
                                            num_area_adjust,
                                            selected_metric_name,
                                            round_metric_value,
                                            float(Llambda_metrics[-1][-1].overflow),
                                        )
                                if adjust_area_flag:
                                    inflation_applied_this_gamma = True
                                    num_area_adjust += 1
                                    if getattr(model, "inflation_state", None) is not None:
                                        model.inflation_state.num_area_adjust = num_area_adjust
                                    # restart Llambda
                                    model.op_collections.density_op.reset()
                                    model.op_collections.density_overflow_op.reset()
                                    model.op_collections.pin_utilization_map_op.reset()
                                    model.initialize_density_weight(
                                        params, placedb)
                                    model.density_weight.mul_(
                                        0.1 / params.density_weight)
                                    logging.info("density_weight = %.6E" %
                                                 (model.density_weight.data))
                                    # load state to restart the optimizer
                                    optimizer.load_state_dict(initial_state)
                                    # must after loading the state
                                    initialize_learning_rate(pos)
                                    # increase iterations of the sub problem to slow down the search
                                    model.Lsub_iteration = model.routability_Lsub_iteration

                                    # reset best metric
                                    best_metric[0] = None
                                    best_pos[0] = None

                                    # disable L-shape during inflation recovery;
                                    # it will re-enable when overflow drops below threshold again
                                    # NOTE: can be overridden by l_shape_keep_during_inflation parameter
                                    keep_during_inflation = getattr(params, "l_shape_keep_during_inflation", False)
                                    if getattr(model, "use_l_shape_routability", False) and not keep_during_inflation:
                                        disable_l_shape_for_recovery(
                                            iteration,
                                            reason="inflation",
                                            update_threshold=True,
                                            inflation_round=num_area_adjust,
                                        )

                                    break
                                else:
                                    num_area_adjust = max_area_adjust_rounds
                                    if getattr(model, "inflation_state", None) is not None:
                                        model.inflation_state.num_area_adjust = num_area_adjust
                                    logging.info(
                                        "Terminate routability inflation after round %d because adjust_node_area reported no further area change",
                                        current_inflation_round.round_idx
                                        if current_inflation_round is not None
                                        else num_area_adjust,
                                    )

                    # gradually reduce gamma to tradeoff smoothness and accuracy
                    if len(placedb.regions) > 0 and Llambda_metrics[-1][-1].goverflow is not None:
                        model.op_collections.update_gamma_op(
                            Lgamma_step, Llambda_metrics[-1][-1].goverflow)
                    elif len(placedb.regions) == 0 and Llambda_metrics[-1][-1].overflow is not None:
                        model.op_collections.update_gamma_op(
                            Lgamma_step, Llambda_metrics[-1][-1].overflow)
                    else:
                        model.op_collections.precondition_op.set_overflow(
                            Llambda_metrics[-1][-1].overflow)
                    if (
                        not inflation_applied_this_gamma
                        and Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics)
                    ) or stop_placement == 1:
                        break

                    # update learning rate
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]:
                        if "learning_rate_decay" in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] *= global_place_params["learning_rate_decay"]

                # in case of divergence, use the best metric
                last_metric = all_metrics[-1][-1][-1]
                # if (
                #     last_metric.overflow[-1] > max(params.stop_overflow, best_metric[0].overflow[-1])
                #     and last_metric.hpwl > best_metric[0].hpwl
                # ):
                #     all_metrics.append([best_metric])
                # fix movable macros
                if params.macro_place_flag and not macro_placed:
                    macro_placed = True
                    # recover halo
                    if params.macro_halo_x > 0 or params.macro_halo_y > 0:
                        with torch.no_grad():
                            movable_macro_mask = self.data_collections.movable_macro_mask
                            movable_macro_pins = self.data_collections.movable_macro_pins
                            # node sizes
                            self.data_collections.node_size_x[: placedb.num_movable_nodes][movable_macro_mask] -= (
                                2 * params.macro_halo_x)
                            self.data_collections.node_size_y[: placedb.num_movable_nodes][movable_macro_mask] -= (
                                2 * params.macro_halo_y)
                            # pin offsets
                            self.data_collections.pin_offset_x[movable_macro_pins] -= params.macro_halo_x
                            self.data_collections.pin_offset_y[movable_macro_pins] -= params.macro_halo_y
                            # macro locations
                            self.pos[0][: placedb.num_movable_nodes][movable_macro_mask] += params.macro_halo_x
                            self.pos[0][placedb.num_nodes: placedb.num_nodes +
                                        placedb.num_movable_nodes][movable_macro_mask] += params.macro_halo_y
                            params.macro_halo_x = 0
                            params.macro_halo_y = 0
                    if params.macro_pin_halo_x >= 0:
                        with torch.no_grad():
                            self.data_collections.node_size_x[placedb.movable_macro_idx] -= torch.tensor(
                                placedb.is_pin_lower_x * params.macro_pin_halo_x + placedb.is_pin_upper_x * params.macro_pin_halo_x, device=self.pos[0].device)
                            self.data_collections.node_size_y[placedb.movable_macro_idx] -= torch.tensor(
                                placedb.is_pin_lower_y * params.macro_pin_halo_y + placedb.is_pin_upper_y * params.macro_pin_halo_y, device=self.pos[0].device)

                            self.data_collections.pin_offset_x[placedb.movable_macro_pins] -= torch.tensor(
                                placedb.is_pin_lower_x[placedb.pin2node_map[placedb.movable_macro_pins]] * params.macro_pin_halo_x, device=self.pos[0].device)
                            self.data_collections.pin_offset_y[placedb.movable_macro_pins] -= torch.tensor(
                                placedb.is_pin_lower_y[placedb.pin2node_map[placedb.movable_macro_pins]] * params.macro_pin_halo_y, device=self.pos[0].device)
                            # macro locations

                            self.pos[0][placedb.movable_slice][
                                placedb.movable_macro_mask
                            ] += torch.tensor(placedb.is_pin_lower_x * params.macro_pin_halo_x, device=self.pos[0].device)

                            self.pos[0][
                                placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                            ][placedb.movable_macro_mask] += torch.tensor(placedb.is_pin_lower_y * params.macro_pin_halo_y, device=self.pos[0].device)
                            params.macro_pin_halo_x = 0
                            params.macro_pin_halo_y = 0

                    if last_metric and (
                        last_metric.overflow[-1] > params.stop_overflow
                        or torch.isinf(last_metric.objective)
                        or torch.isnan(last_metric.objective)
                    ):
                        break

                    if params.plot_flag:
                        self.plot(params, placedb, iteration,
                                  self.pos[0].data.clone().cpu().numpy())
                    self.pos[0].data.copy_(
                        self.op_collections.macro_legalize_op(self.pos[0]))
                    iteration += 1
                    if params.plot_flag:
                        self.plot(params, placedb, iteration,
                                  self.pos[0].data.clone().cpu().numpy())

                logging.info("optimizer %s takes %.3f seconds" %
                             (optimizer_name, time.time() - tt))

            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                selected_inflation_round = None
                replay_best_inflation_round = bool(
                    getattr(params, "enhanced_inflation_replay_best_round_flag", False)
                )
                if (
                    replay_best_inflation_round
                    and enhanced_inflation_controller.is_enhanced_inflation_enabled(params)
                ):
                    selected_inflation_round = enhanced_inflation_controller.select_best_gr_solution(
                        getattr(self.data_collections, "inflation_state", None),
                        metric_name=getattr(
                            params,
                            "enhanced_inflation_select_metric",
                            "est_shorts",
                        ),
                    )
                with torch.no_grad():
                    # convert lower left to centers
                    self.pos[0][: placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_x[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_y[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.node_size_x.copy_(
                        self.data_collections.original_node_size_x)
                    self.data_collections.node_size_y.copy_(
                        self.data_collections.original_node_size_y)
                    enhanced_inflation_controller.sync_node_areas(
                        self.data_collections
                    )
                    # use fixed centers as the anchor
                    self.pos[0][: placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_x[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_y[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.pin_offset_x.copy_(
                        self.data_collections.original_pin_offset_x)
                    self.data_collections.pin_offset_y.copy_(
                        self.data_collections.original_pin_offset_y)
                    enhanced_inflation_controller.rollback_inflation_state(
                        self.data_collections
                    )
                    enhanced_inflation_controller.restore_model_density_state(
                        getattr(self.data_collections, "inflation_state", None),
                        model,
                    )
                if selected_inflation_round is not None:
                    with torch.no_grad():
                        self.pos[0].data.copy_(
                            selected_inflation_round.position_snapshot.data.to(
                                device=self.pos[0].device,
                                dtype=self.pos[0].dtype,
                            )
                        )
                    selected_metric_name = getattr(
                        params,
                        "enhanced_inflation_select_metric",
                        "est_shorts",
                    )
                    selected_metric_value = enhanced_inflation_controller.get_round_metric(
                        selected_inflation_round,
                        selected_metric_name,
                    )
                    logging.info(
                        "Replay enhanced best inflation round %d after rollback using %s=%.4f (trigger_overflow=%.6f, stage=%d, iter=%d)",
                        selected_inflation_round.round_idx,
                        selected_metric_name,
                        float(selected_metric_value)
                        if selected_metric_value is not None
                        else float("nan"),
                        selected_inflation_round.trigger_overflow,
                        selected_inflation_round.stage_idx,
                        selected_inflation_round.iteration,
                    )
                elif (
                    not replay_best_inflation_round
                    and enhanced_inflation_controller.is_enhanced_inflation_enabled(params)
                ):
                    logging.info(
                        "Skip enhanced best-round replay after rollback because enhanced_inflation_replay_best_round_flag is disabled"
                    )
                elif replay_best_inflation_round and enhanced_inflation_controller.is_enhanced_inflation_enabled(params):
                    logging.info(
                        "Skip enhanced best-round replay after rollback because no eligible inflation round was recorded"
                    )

        else:
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
        if params.plot_flag:
            self.plot(params, placedb, 9999,
                      self.pos[0].data.clone().cpu().numpy())

        processed_metrics = {
            "objective": [-1],
            "hpwl": [-1],
            "overflow": [-1],
            "density": [-1],
        }
        if params.global_place_flag:
            # dump global placement solution for legalization
            if params.dump_global_place_solution_flag:
                self.dump(params, placedb, self.pos[0].cpu(
                ), "%s.lg.pklz" % (params.design_name()))

            # process metrics
            def flatten(l): return sum(map(flatten, l), []
                                       ) if isinstance(l, list) else [l]
            metrics = flatten(all_metrics)
            objectives = [metric.objective.data.item() for metric in metrics]
            hpwls = [metric.hpwl.data.item() for metric in metrics]
            overflows = [metric.overflow.data.item() for metric in metrics]
            densities = [metric.max_density.data.item() for metric in metrics]
            processed_metrics = {
                "objective": objectives,
                "hpwl": hpwls,
                "overflow": overflows,
                "density": densities,
            }
            optional_metric_fields = [
                "l_shape_fast_mode",
                "l_shape_energy_valid",
                "l_shape_cost",
                "l_shape_weighted_cost",
                "l_shape_weight",
                "l_shape_target_weight",
                "l_shape_weight_candidate",
                "l_shape_base_grad_norm",
                "l_shape_grad_raw_norm",
                "l_shape_grad_norm",
                "l_shape_grad_ratio",
                "l_shape_target_ratio",
                "l_shape_overflow",
                "l_shape_overflow_ratio",
                "l_shape_overflow_ema",
                "l_shape_overflow_max_density",
                "l_shape_capacity_al_enabled",
                "l_shape_capacity_al_updated",
                "l_shape_capacity_al_g_h_max",
                "l_shape_capacity_al_g_v_max",
                "l_shape_capacity_al_g_h_sum",
                "l_shape_capacity_al_g_v_sum",
                "l_shape_capacity_al_g_h_pos_ratio",
                "l_shape_capacity_al_g_v_pos_ratio",
                "l_shape_capacity_al_q_h_max",
                "l_shape_capacity_al_q_v_max",
                "l_shape_capacity_al_q_h_sum",
                "l_shape_capacity_al_q_v_sum",
                "l_shape_capacity_al_lambda_h_max",
                "l_shape_capacity_al_lambda_v_max",
                "l_shape_capacity_al_lambda_h_sum",
                "l_shape_capacity_al_lambda_v_sum",
                "l_shape_capacity_al_energy_h",
                "l_shape_capacity_al_energy_v",
                "l_shape_capacity_al_energy_total",
                "l_shape_capacity_al_pq_h_min",
                "l_shape_capacity_al_pq_h_max",
                "l_shape_capacity_al_pq_h_sum",
                "l_shape_capacity_al_pq_v_min",
                "l_shape_capacity_al_pq_v_max",
                "l_shape_capacity_al_pq_v_sum",
                "l_shape_capacity_al_active_memory_bins_h",
                "l_shape_capacity_al_active_memory_bins_v",
                "l_shape_macro_exclusion_enabled",
                "l_shape_macro_exclusion_macro_count",
                "l_shape_macro_exclusion_body_bins",
                "l_shape_macro_exclusion_halo_bins",
                "l_shape_macro_exclusion_active_bins",
                "l_shape_macro_exclusion_source_max",
                "l_shape_macro_exclusion_source_sum",
                "l_shape_macro_exclusion_body_source_max",
                "l_shape_macro_exclusion_body_source_sum",
                "l_shape_macro_exclusion_halo_source_max",
                "l_shape_macro_exclusion_halo_source_sum",
                "l_shape_macro_exclusion_usage_max",
                "l_shape_macro_exclusion_usage_sum",
                "l_shape_macro_exclusion_usage_bins",
                "l_shape_macro_exclusion_dominates_bins",
                "l_shape_macro_exclusion_routing_dominates_bins",
                "soft_l_diag_count",
                "soft_l_mean_cost_gap",
                "soft_l_raw_cost_gap_p50",
                "soft_l_biased_cost_gap_p50",
                "soft_l_tau_source_gap",
                "soft_l_mean_max_prob",
                "soft_l_mean_entropy",
                "soft_l_near_tie_ratio",
                "soft_l_tau",
                "soft_l_effective_hotspot_weight",
                "soft_l_resolver_agreement_ratio",
                "soft_l_same_net_topo_nets",
                "soft_l_same_net_topo_segments_h",
                "soft_l_same_net_topo_segments_v",
                "soft_l_same_net_topo_diag_edges",
                "soft_l_same_net_topo_edges_with_topology",
                "soft_l_same_net_topo_edges_with_observed_intervals",
                "soft_l_same_net_topo_mean_gap",
                "soft_l_same_net_topo_tie_ratio",
            ]

            def scalarize_metric_value(value):
                if value is None:
                    return None
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return value.detach().cpu().view(-1).tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    return None
                return value

            fixed_when_l_shape_fields = {
                "l_shape_fast_mode",
                "l_shape_energy_valid",
            }
            energy_derived_metric_fields = {
                "l_shape_cost",
                "l_shape_weighted_cost",
                "l_shape_capacity_al_energy_h",
                "l_shape_capacity_al_energy_v",
                "l_shape_capacity_al_energy_total",
                "l_shape_capacity_al_pq_h_min",
                "l_shape_capacity_al_pq_h_max",
                "l_shape_capacity_al_pq_h_sum",
                "l_shape_capacity_al_pq_v_min",
                "l_shape_capacity_al_pq_v_max",
                "l_shape_capacity_al_pq_v_sum",
            }
            l_shape_seen = any(
                getattr(metric, "l_shape_fast_mode", None) is not None
                or getattr(metric, "l_shape_energy_valid", None) is not None
                for metric in metrics
            )
            l_shape_energy_invalid_seen = any(
                getattr(metric, "l_shape_energy_valid", None) is not None
                and not bool(getattr(metric, "l_shape_energy_valid"))
                for metric in metrics
            )
            for field_name in optional_metric_fields:
                series = [
                    scalarize_metric_value(getattr(metric, field_name, None))
                    for metric in metrics
                ]
                if (
                    any(value is not None for value in series)
                    or (l_shape_seen and field_name in fixed_when_l_shape_fields)
                    or (
                        l_shape_energy_invalid_seen
                        and field_name in energy_derived_metric_fields
                    )
                ):
                    processed_metrics[field_name] = series

            # plot placement
            if params.plot_flag:
                self.plot(params, placedb, iteration,
                          self.pos[0].data.clone().cpu().numpy())

            last_metric = copy.deepcopy(all_metrics)
            for idx in [-1, -1, -1]:
                try:
                    last_metric = last_metric[idx]
                except IndexError:
                    last_metric = False
                    break

            if not last_metric:
                cur_metric = EvalMetrics.EvalMetrics(iteration)
                all_metrics.append(cur_metric)
                cur_metric.evaluate(
                    placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
                )
                logging.info(cur_metric)

            # in case of significant divergence, no need to run legalizer
            if last_metric and (
                last_metric.overflow[-1] > params.stop_overflow
                or torch.isinf(last_metric.objective)
                or torch.isnan(last_metric.objective)
            ):
                logging.warn(
                    "overflow is significant %.3f or hpwl is infinity or nan, skip legalization and detail placement steps"
                    % (last_metric.overflow[-1])
                )
                self.plot(params, placedb, 9999,
                          self.pos[0].data.clone().cpu().numpy())
                return float("inf"), float("inf"), processed_metrics

            # recover node sizes, pins shifts, and positions of macros
            if params.macro_halo_x >= 0 and params.macro_halo_y >= 0:
                with torch.no_grad():
                    # node sizes
                    self.data_collections.node_size_x[placedb.movable_macro_idx] -= (
                        2 * params.macro_halo_x
                    )
                    self.data_collections.node_size_y[placedb.movable_macro_idx] -= (
                        2 * params.macro_halo_y
                    )
                    # self.data_collections.node_size_x[placedb.fixed_macro_idx] -= (
                    #     2 * params.macro_halo_x
                    # )
                    # self.data_collections.node_size_y[placedb.fixed_macro_idx] -= (
                    #     2 * params.macro_halo_y
                    # )

                    # pin offsets
                    self.data_collections.pin_offset_x[
                        placedb.movable_macro_pins
                    ] -= params.macro_halo_x
                    self.data_collections.pin_offset_y[
                        placedb.movable_macro_pins
                    ] -= params.macro_halo_y

                    self.pos[0][placedb.movable_slice][
                        placedb.movable_macro_mask
                    ] += params.macro_halo_x
                    self.pos[0][
                        placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                    ][placedb.movable_macro_mask] += params.macro_halo_y

                    self.pos[0][placedb.fixed_slice][
                        placedb.fixed_macro_mask
                    ] += params.macro_halo_x
                    self.pos[0][
                        placedb.num_nodes
                        + placedb.num_movable_nodes: placedb.num_nodes
                        + placedb.num_movable_nodes
                        + placedb.num_terminals
                    ][placedb.fixed_macro_mask] += params.macro_halo_y

                    # self.data_collections.pin_offset_x[
                    #     placedb.fixed_macro_pins
                    # ] -= params.macro_halo_x
                    # self.data_collections.pin_offset_y[
                    #     placedb.fixed_macro_pins
                    # ] -= params.macro_halo_y
            if params.macro_pin_halo_x >= 0:
                with torch.no_grad():
                    self.data_collections.node_size_x[placedb.movable_macro_idx] -= torch.tensor(
                        placedb.is_pin_lower_x * params.macro_pin_halo_x + placedb.is_pin_upper_x * params.macro_pin_halo_x, device=self.pos[0].device)
                    self.data_collections.node_size_y[placedb.movable_macro_idx] -= torch.tensor(
                        placedb.is_pin_lower_y * params.macro_pin_halo_y + placedb.is_pin_upper_y * params.macro_pin_halo_y, device=self.pos[0].device)

                    self.data_collections.pin_offset_x[placedb.movable_macro_pins] -= torch.tensor(
                        placedb.is_pin_lower_x[placedb.pin2node_map[placedb.movable_macro_pins]] * params.macro_pin_halo_x, device=self.pos[0].device)
                    self.data_collections.pin_offset_y[placedb.movable_macro_pins] -= torch.tensor(
                        placedb.is_pin_lower_y[placedb.pin2node_map[placedb.movable_macro_pins]] * params.macro_pin_halo_y, device=self.pos[0].device)
                    # macro locations

                    self.pos[0][placedb.movable_slice][
                        placedb.movable_macro_mask
                    ] += torch.tensor(placedb.is_pin_lower_x * params.macro_pin_halo_x, device=self.pos[0].device)

                    self.pos[0][
                        placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                    ][placedb.movable_macro_mask] += torch.tensor(placedb.is_pin_lower_y * params.macro_pin_halo_y, device=self.pos[0].device)
                    params.macro_pin_halo_x = 0
                    params.macro_pin_halo_y = 0

        fixed_macro_pre_legalization_pos = None
        try:
            fixed_macro_pre_legalization_pos = self.pos[0].detach().clone()
            fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                self.pos[0],
                self.data_collections.node_size_x,
                self.data_collections.node_size_y,
                placedb,
            )
            for key, value in fixed_macro_overlap_stats.items():
                processed_metrics["pre_legalization_%s" % key] = value
            logging.info(
                "Fixed macro overlap telemetry: stage=pre_legalization "
                "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                "coordinate_system=%s macro_set_source=%s",
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_macro_count", 0
                    )
                ),
                float(fixed_macro_overlap_stats.get("fixed_macro_overlap_area", 0.0)),
                float(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_area_ratio", 0.0
                    )
                ),
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_cell_count", 0
                    )
                ),
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_pair_count", 0
                    )
                ),
                float(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_max_area", 0.0
                    )
                ),
                str(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_coordinate_system", "unknown"
                    )
                ),
                str(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_macro_set_source", "unknown"
                    )
                ),
            )
        except Exception as e:
            logging.warning(
                "Failed to compute fixed macro overlap telemetry before legalization: %s",
                e,
            )

        # legalization
        if params.legalize_flag:
            if params.macro_place_flag:
                tt = time.time()
                self.pos[0].data.copy_(
                    self.op_collections.macro_legalize_op(self.pos[0]))
                logging.info("Macro legalization takes %.3f seconds" %
                             (time.time() - tt))
                try:
                    fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                        self.pos[0],
                        self.data_collections.node_size_x,
                        self.data_collections.node_size_y,
                        placedb,
                    )
                    for key, value in fixed_macro_overlap_stats.items():
                        processed_metrics["post_macro_legalization_%s" % key] = value
                    if fixed_macro_pre_legalization_pos is not None:
                        displacement_stats = compute_movable_displacement_stats(
                            fixed_macro_pre_legalization_pos, self.pos[0], placedb
                        )
                        for key, value in displacement_stats.items():
                            processed_metrics[
                                "post_macro_legalization_%s" % key
                            ] = value
                    else:
                        displacement_stats = {}
                    logging.info(
                        "Fixed macro overlap telemetry: stage=post_macro_legalization "
                        "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                        "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                        "legalization_moved=%d legalization_disp_max=%.6E "
                        "legalization_disp_mean=%.6E",
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_macro_count", 0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_area", 0.0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_area_ratio", 0.0
                            )
                        ),
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_cell_count", 0
                            )
                        ),
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_pair_count", 0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_max_area", 0.0
                            )
                        ),
                        int(
                            displacement_stats.get(
                                "movable_displacement_moved_count", 0
                            )
                        ),
                        float(
                            displacement_stats.get(
                                "movable_displacement_max", 0.0
                            )
                        ),
                        float(
                            displacement_stats.get(
                                "movable_displacement_mean", 0.0
                            )
                        ),
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to compute fixed macro overlap telemetry after macro legalization: %s",
                        e,
                    )
                cur_metric = EvalMetrics.EvalMetrics(iteration)
                all_metrics.append(cur_metric)
                cur_metric.evaluate(
                    placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
                logging.info(cur_metric)
                iteration += 1

        if params.legalize_flag:
            iteration = self._run_standard_legalization(
                params, placedb, iteration, all_metrics
            )
            try:
                fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                    self.pos[0],
                    self.data_collections.node_size_x,
                    self.data_collections.node_size_y,
                    placedb,
                )
                for key, value in fixed_macro_overlap_stats.items():
                    processed_metrics["post_legalization_%s" % key] = value
                if fixed_macro_pre_legalization_pos is not None:
                    displacement_stats = compute_movable_displacement_stats(
                        fixed_macro_pre_legalization_pos, self.pos[0], placedb
                    )
                    for key, value in displacement_stats.items():
                        processed_metrics["post_legalization_%s" % key] = value
                else:
                    displacement_stats = {}
                logging.info(
                    "Fixed macro overlap telemetry: stage=post_legalization "
                    "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                    "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                    "legalization_moved=%d legalization_disp_max=%.6E "
                    "legalization_disp_mean=%.6E legalization_disp_sum=%.6E",
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_macro_count", 0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_area", 0.0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_area_ratio", 0.0
                        )
                    ),
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_cell_count", 0
                        )
                    ),
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_pair_count", 0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_max_area", 0.0
                        )
                    ),
                    int(
                        displacement_stats.get(
                            "movable_displacement_moved_count", 0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_max", 0.0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_mean", 0.0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_sum", 0.0
                        )
                    ),
                )
            except Exception as e:
                logging.warning(
                    "Failed to compute fixed macro overlap telemetry after legalization: %s",
                    e,
                )
            if getattr(params, "egr_padding_flag", 0):
                self._apply_egr_padding(params, placedb)
                iteration = self._run_standard_legalization(
                    params, placedb, iteration, all_metrics
                )
                self._restore_egr_padding()

            # Perform any configured post-legalization STA on the final
            # legalized position. The public Placer rejects the removed
            # OpenTimer mode before this path is reachable.
            if params.timing_opt_flag:
                logging.info("additional sta after legalization")
                timing_op = self.op_collections.timing_op
                timing_op(self.pos[0].data.clone().cpu())
                timing_op.timer.update_timing()
                cur_metric = all_metrics[-1]
                cur_metric.tns = timing_op.timer.report_tns_elw(
                    split=1
                ) / (time_unit * 1e17)
                cur_metric.wns = timing_op.timer.report_wns(
                    split=1
                ) / (time_unit * 1e15)

        # after_legalization recover node sizes, pins shifts, and positions of cells
        if params.cell_padding_x >= 0:
            with torch.no_grad():
                # node sizes
                self.data_collections.node_size_x[:placedb.num_movable_nodes] -= (
                    2 * params.cell_padding_x
                )
                # self.data_collections.node_size_y[:placedb.num_movable_nodes] -= (
                #     2 * params.cell_padding_y
                # )
                movable_cell_tensor = np.arange(
                    0, placedb.num_movable_nodes, dtype=placedb.pin2node_map.dtype)
                # shift macro pins
                movable_cell_pins = np.isin(
                    placedb.pin2node_map, movable_cell_tensor)

                # pin offsets
                self.data_collections.pin_offset_x[movable_cell_pins] -= params.cell_padding_x
                # self.data_collections.pin_offset_y -= params.cell_padding_y

                self.pos[0][:placedb.num_movable_nodes] += params.cell_padding_x
                params.cell_padding_x = 0
                # self.pos[0][
                #     placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                # ] += params.cell_padding_y

        # # rescale everything
        # cur_scale_factor = self.data_collections.fp_info.scale_factor
        # gcd_site_scale_factor = 1 / math.gcd(
        #     placedb.origin_site_width, placedb.origin_row_height
        # )
        # if cur_scale_factor != gcd_site_scale_factor:
        #     logging.warn(
        #         f"Rescaling by GCD(site_width, row_height) = {gcd_site_scale_factor} before legalization and detailed placement"
        #     )
        #     params.scale_factor = gcd_site_scale_factor
        #     rescale_factor = gcd_site_scale_factor / cur_scale_factor
        #     with torch.no_grad():
        #         self.pos[0].mul_(rescale_factor).round_()
        #         self.data_collections.node_size_x.mul_(rescale_factor).round_()
        #         self.data_collections.node_size_y.mul_(rescale_factor).round_()
        #         self.data_collections.flat_region_boxes.mul_(
        #             rescale_factor).round_()
        #         self.data_collections.pin_offset_x.mul_(rescale_factor)
        #         self.data_collections.pin_offset_y.mul_(rescale_factor)
        #         # self.data_collections.node_areas.mul_(rescale_factor * rescale_factor)
        #         self.data_collections.fp_info.scale(rescale_factor)
        #         self.data_collections.fp_info.scale_factor = gcd_site_scale_factor
        #         params.macro_halo_x *= rescale_factor
        #         params.macro_halo_y *= rescale_factor
        #         params.macro_pin_halo_x *= rescale_factor
        #         params.macro_pin_halo_y *= rescale_factor
        #         params.cell_padding_x *= rescale_factor
        #         params.cell_padding_y *= rescale_factor
        #         # TODO: rescale fence regions

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration,
                      self.pos[0].data.clone().cpu().numpy())

        # dump legalization solution for detailed placement
        if params.dump_legalize_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(
            ), "%s.dp.pklz" % (params.design_name()))

        # detailed placement
        if params.detailed_place_flag:
            tt = time.time()
            self.pos[0].data.copy_(
                self.op_collections.detailed_place_op(self.pos[0]))
            logging.info("detailed placement takes %.3f seconds" %
                         (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        if self.op_collections.m2_pa_refine_op is not None:
            refined_pos, _ = self.op_collections.m2_pa_refine_op(self.pos[0])
            self.pos[0].data.copy_(refined_pos)
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
            )
            logging.info(cur_metric)
            iteration += 1

        adaptive_padding_pos, adaptive_padding_stats = (
            _run_post_legalization_adaptive_padding(
                params,
                placedb,
                self.pos[0],
                self,
            )
        )
        if adaptive_padding_stats is not None:
            self.pos[0].data.copy_(adaptive_padding_pos)
            for key, value in adaptive_padding_stats.items():
                processed_metrics[
                    "post_legalization_adaptive_padding_%s" % key
                ] = value
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
            )
            logging.info(cur_metric)
            iteration += 1

        # save results
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution
        placedb.apply(
            params,
            cur_pos[0: placedb.num_movable_nodes],
            cur_pos[placedb.num_nodes: placedb.num_nodes +
                    placedb.num_movable_nodes],
        )

        # update pin offsets of std cells
        # assume rows are FS = 0, N = 1, FS, ...
        # cur_orient = torch.from_numpy(
        #     np.where(placedb.node_orient == b"N", 1, 0)[: placedb.num_movable_nodes]
        # ).to(self.pos[0].device)
        # new_orient = (
        #     torch.div(
        #         self.pos[0][
        #             placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
        #         ],
        #         self.data_collections.fp_info.row_height,
        #         rounding_mode="floor",
        #     )
        #     % 2
        # )
        # flips = (
        #     (cur_orient != new_orient) & ~self.data_collections.movable_macro_mask
        # ).nonzero().view(-1)
        # self.data_collections.pin_offset_y[flips] = (
        #     self.data_collections.fp_info.row_height
        #     - self.data_collections.pin_offset_y[flips]
        # )

        # reset net weights
        self.data_collections.net_weights.fill_(1.0)

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, cur_pos)

        # update pin offsets of std cells
        # assume rows are FS = 0, N = 1, FS, ...
        # cur_orient = torch.from_numpy(
        #     np.where(placedb.node_orient == b"N", 1, 0)[: placedb.num_movable_nodes]
        # ).to(self.pos[0].device)
        # new_orient = (
        #     torch.div(
        #         self.pos[0][
        #             placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
        #         ],
        #         self.data_collections.fp_info.row_height,
        #         rounding_mode="floor",
        #     )
        #     % 2
        # )
        # flips = (
        #     (cur_orient != new_orient) & ~self.data_collections.movable_macro_mask
        # ).nonzero().view(-1)
        # self.data_collections.pin_offset_y[flips] = (
        #     self.data_collections.fp_info.row_height
        #     - self.data_collections.pin_offset_y[flips]
        # )

        # reset net weights
        self.data_collections.net_weights.fill_(1.0)

        # run RSMT
        if params.with_sta:
            with torch.no_grad():
                tt = time.time()
                self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                    self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                    self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(
                        self.op_collections.pin_pos_op(self.pos[0]))
                new_x, new_y = self.op_collections.steiner_topo_op(
                    self.op_collections.pin_pos_op(self.pos[0])
                )
                self.data_collections.net_flat_topo_sort,
                self.data_collections.net_flat_topo_sort_start,
                self.data_collections.pin_fa,
                self.data_collections.flat_pin_to_start,
                flat_pin_to = self.data_collections.flat_pin_to
                flat_pin_from = self.data_collections.flat_pin_from
                
                length = (torch.abs(new_x[flat_pin_from] - new_x[flat_pin_to])
                        + torch.abs(new_y[flat_pin_from] - new_y[flat_pin_to])) / params.scale_factor  / placedb.dbu
                flute_length_path = "%s/%s_flute_length.txt" % (
                    params.result_dir, params.design_name())
                # with open(flute_length_path, "w") as f:
                #     f.write("net_name, flute_length (um), cap (pF)\n")
                #     for net_id in range(placedb.num_nets):
                #         start = self.data_collections.net_flat_topo_sort_start[net_id]
                #         end = self.data_collections.net_flat_topo_sort_start[net_id + 1]
                #         net_name = placedb.net_names[net_id]
                #         net_flat_pins_nodes = self.data_collections.net_flat_topo_sort[start:end]
                #         net_flat_pin_start = self.data_collections.flat_pin_to_start[net_flat_pins_nodes]
                #         net_flat_pin_end = self.data_collections.flat_pin_to_start[net_flat_pins_nodes + 1]
                #         net_length = 0
                #         for i in range(len(net_flat_pins_nodes)):
                #             pin_start = net_flat_pin_start[i]
                #             pin_end = net_flat_pin_end[i]
                #             net_length += length[pin_start:pin_end].sum().item()
                #         # net_length = length[start:end].sum().item()
                #         f.write(f"{net_name}, {net_length}, {placedb.c_unit * net_length}\n")

                wns, tns, ws, ts = model.timing_obj(self.pos[0])
                model.check_log(wns, tns, ws, ts)
                logging.info("rsmt computation takes %.3f seconds" %
                            (time.time() - tt))

        # get HPWL
        with torch.no_grad():
            hpwl = self.op_collections.hpwl_op(self.pos[0])
            rsmt_wl = self.op_collections.rsmt_wl_op(self.pos[0]) / placedb.dbu
            logging.info("flute rsmt %.6E um" % rsmt_wl)
            logging.info("unweighted hpwl %.6E" % hpwl)

        _run_gpugr_final_eval(params, placedb, self.pos[0])

        # save nets degree, RSMT, HPWL
        # with torch.no_grad():
        #     degrees = torch.from_numpy(np.ediff1d(placedb.flat_net2pin_start_map))
        #     mask = torch.logical_and(2 <= degrees, degrees < params.ignore_net_degree)
        #     degrees = degrees[mask].long()
        #     steiners = self.op_collections.rsmt_wl_op(self.pos[0], False)[mask]
        #     wirelengths = (
        #         self.op_collections.hpwl_op(self.pos[0], False)
        #         .cpu()
        #         .detach()[mask]
        #     )
        #     weights = steiners / wirelengths
        #     # get new RISA weights
        #     degrees, indices = torch.sort(degrees)
        #     weights = weights[indices]
        #     c = torch.stack((degrees, weights))
        #     idxs, vals = torch.unique(c[0, :], return_counts=True)
        #     vs = torch.split_with_sizes(c[1, :], tuple(vals))
        #     weights_dict = {int(k.item()): float(v.mean()) for k, v in zip(idxs, vs)}
        #     path = "%s/%s" % (params.result_dir, params.design_name())
        #     with open("%s/risa_weights.pkl" % path, "wb") as f:
        #         pickle.dump(weights_dict, f)

        return float(rsmt_wl), float(hpwl), processed_metrics
