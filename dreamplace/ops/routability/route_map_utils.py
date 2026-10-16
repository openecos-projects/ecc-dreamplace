"""Pure map and routing-geometry helpers for routability placement."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from dreamplace.ops.routability.egr_resample import create_supply_map_from_placedb
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose


def normalize_placedb_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    if hasattr(name, "decode"):
        try:
            return name.decode("utf-8")
        except Exception:
            pass
    return str(name)


def resample_xy_map(map_xy, target_x, target_y):
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


def build_l_shape_supply_original_maps_from_placedb(
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


def split_gpugr_maps_by_direction(placedb, capacity_map, demand_map):
    """Aggregate per-layer GPUGR maps into horizontal and vertical maps."""
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
    h_caps = torch.as_tensor(
        raw_h_caps[:num_layers], device=capacity_map.device, dtype=capacity_map.dtype
    )
    v_caps = torch.as_tensor(
        raw_v_caps[:num_layers], device=capacity_map.device, dtype=capacity_map.dtype
    )
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


def fallback_l_shape_wire_width(
    placedb, route_xsize=None, route_ysize=None, fallback_wire_width=None
):
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


def resolve_l_shape_wire_width(
    placedb, route_xsize=None, route_ysize=None, fallback_wire_width=None
):
    raw_widths = getattr(placedb, "min_wire_widths", None)
    widths = (
        np.array(raw_widths, dtype=np.float32)
        if raw_widths is not None and len(raw_widths) > 0
        else np.array([], dtype=np.float32)
    )
    raw_h_caps = getattr(placedb, "unit_horizontal_capacities", None)
    raw_v_caps = getattr(placedb, "unit_vertical_capacities", None)
    h_caps = np.array(raw_h_caps, dtype=np.float32) if raw_h_caps is not None else np.array([], dtype=np.float32)
    v_caps = np.array(raw_v_caps, dtype=np.float32) if raw_v_caps is not None else np.array([], dtype=np.float32)

    def _finalize(value, source):
        wire_width = float(value)
        logging.info("Resolved L-shape wire width: source=%s value=%.4f", source, wire_width)
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
            "Cannot compute capacity_weighted(width) for L-shape wire width because "
            "routing-layer capacities are unavailable. Fall back to min_width."
        )
        return _finalize(positive_widths.min(), "min_wire_width(no_capacity)")

    fallback_value, fallback_source = fallback_l_shape_wire_width(
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


def prepare_modularity_maps_from_gpugr(placedb, gpugr_congestion_map_op, pos, eps=1e-6):
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
        return resample_xy_map(xy_map, target_x, target_y)

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


__all__ = [
    "normalize_placedb_name",
    "resample_xy_map",
    "build_l_shape_supply_original_maps_from_placedb",
    "split_gpugr_maps_by_direction",
    "fallback_l_shape_wire_width",
    "resolve_l_shape_wire_width",
    "prepare_modularity_maps_from_gpugr",
]
