import logging
from collections import Counter, defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _normalize_net_name(net_name):
    if isinstance(net_name, bytes):
        return net_name.decode("utf-8")
    return str(net_name)


def _build_net_name_to_id(placedb):
    mapping = {}
    for net_id, net_name in enumerate(getattr(placedb, "net_names", [])):
        mapping[_normalize_net_name(net_name)] = int(net_id)
    return mapping


def _merge_intervals(intervals, max_gap=1):
    if not intervals:
        return []
    normalized = sorted((min(lo, hi), max(lo, hi)) for lo, hi in intervals)
    merged = [[int(normalized[0][0]), int(normalized[0][1])]]
    for lo, hi in normalized[1:]:
        prev = merged[-1]
        if int(lo) <= prev[1] + int(max_gap):
            prev[1] = max(prev[1], int(hi))
        else:
            merged.append([int(lo), int(hi)])
    return [tuple(item) for item in merged]


def _interval_total_length(intervals):
    total = 0
    for lo, hi in intervals:
        total += int(hi) - int(lo)
    return int(total)


def _route_grid_shape(placedb):
    num_x = int(getattr(placedb, "num_routing_grids_x", 0))
    num_y = int(getattr(placedb, "num_routing_grids_y", 0))
    return (num_x, num_y)


def _build_net_topology_record(net_id, net_route, max_gap=1):
    entries = net_route.get("entries", []) or []
    horizontal_by_row = defaultdict(list)
    vertical_by_col = defaultdict(list)
    point_count = Counter()
    invalid_wire_count = 0
    raw_wire_count = 0

    bbox_x_lo = None
    bbox_y_lo = None
    bbox_x_hi = None
    bbox_y_hi = None

    for entry in entries:
        if entry.get("type", "") != "wire":
            continue
        raw_wire_count += 1
        x1 = int(entry.get("grid_x1", 0))
        y1 = int(entry.get("grid_y1", 0))
        x2 = int(entry.get("grid_x2", 0))
        y2 = int(entry.get("grid_y2", 0))
        orientation = entry.get("orientation", "")
        is_horizontal = orientation == "H" or y1 == y2
        is_vertical = orientation == "V" or x1 == x2

        bbox_x_lo = x1 if bbox_x_lo is None else min(bbox_x_lo, x1, x2)
        bbox_y_lo = y1 if bbox_y_lo is None else min(bbox_y_lo, y1, y2)
        bbox_x_hi = x1 if bbox_x_hi is None else max(bbox_x_hi, x1, x2)
        bbox_y_hi = y1 if bbox_y_hi is None else max(bbox_y_hi, y1, y2)

        if is_horizontal and y1 == y2:
            lo, hi = sorted((x1, x2))
            horizontal_by_row[int(y1)].append((int(lo), int(hi)))
            point_count[(int(x1), int(y1))] += 1
            point_count[(int(x2), int(y2))] += 1
            continue
        if is_vertical and x1 == x2:
            lo, hi = sorted((y1, y2))
            vertical_by_col[int(x1)].append((int(lo), int(hi)))
            point_count[(int(x1), int(y1))] += 1
            point_count[(int(x2), int(y2))] += 1
            continue
        invalid_wire_count += 1

    merged_h = {
        int(y_idx): _merge_intervals(intervals, max_gap=max_gap)
        for y_idx, intervals in horizontal_by_row.items()
        if intervals
    }
    merged_v = {
        int(x_idx): _merge_intervals(intervals, max_gap=max_gap)
        for x_idx, intervals in vertical_by_col.items()
        if intervals
    }

    if not merged_h and not merged_v:
        return None

    junction_points = sorted(
        (int(x), int(y)) for (x, y), count in point_count.items() if count > 1
    )
    junction_array = (
        np.asarray(junction_points, dtype=np.int32)
        if junction_points
        else np.zeros((0, 2), dtype=np.int32)
    )

    num_horizontal_intervals = sum(len(intervals) for intervals in merged_h.values())
    num_vertical_intervals = sum(len(intervals) for intervals in merged_v.values())
    total_horizontal_length = sum(_interval_total_length(intervals) for intervals in merged_h.values())
    total_vertical_length = sum(_interval_total_length(intervals) for intervals in merged_v.values())

    return {
        "net_id": int(net_id),
        "net_name": net_route.get("net_name", "") or f"net_{net_id}",
        "horizontal_by_row": merged_h,
        "vertical_by_col": merged_v,
        "horizontal_rows": np.asarray(sorted(merged_h.keys()), dtype=np.int32),
        "vertical_cols": np.asarray(sorted(merged_v.keys()), dtype=np.int32),
        "junction_points": junction_array,
        "bbox": (
            int(bbox_x_lo if bbox_x_lo is not None else 0),
            int(bbox_y_lo if bbox_y_lo is not None else 0),
            int(bbox_x_hi if bbox_x_hi is not None else 0),
            int(bbox_y_hi if bbox_y_hi is not None else 0),
        ),
        "num_horizontal_intervals": int(num_horizontal_intervals),
        "num_vertical_intervals": int(num_vertical_intervals),
        "total_horizontal_length": int(total_horizontal_length),
        "total_vertical_length": int(total_vertical_length),
        "raw_wire_count": int(raw_wire_count),
        "invalid_wire_count": int(invalid_wire_count),
        "route_failed": bool(net_route.get("route_failed", False)),
    }


def build_same_net_topology_cache(route_entries, placedb, max_gap=1):
    net_name_to_id = _build_net_name_to_id(placedb)
    net_topologies = {}
    num_route_failed_nets = 0
    unknown_name_count = 0
    total_segments_h = 0
    total_segments_v = 0
    total_length_h = 0
    total_length_v = 0
    max_intervals_per_net = 0
    invalid_wire_count = 0

    for fallback_net_id, net_route in enumerate(route_entries or []):
        if bool(net_route.get("route_failed", False)):
            num_route_failed_nets += 1
        net_name = net_route.get("net_name", "")
        if net_name in net_name_to_id:
            net_id = net_name_to_id[net_name]
        else:
            net_id = int(net_route.get("net_id", -1))
            if net_id < 0:
                unknown_name_count += 1
                continue
        record = _build_net_topology_record(int(net_id), net_route, max_gap=max_gap)
        if record is None:
            continue
        net_topologies[int(net_id)] = record
        total_segments_h += int(record["num_horizontal_intervals"])
        total_segments_v += int(record["num_vertical_intervals"])
        total_length_h += int(record["total_horizontal_length"])
        total_length_v += int(record["total_vertical_length"])
        invalid_wire_count += int(record["invalid_wire_count"])
        max_intervals_per_net = max(
            max_intervals_per_net,
            int(record["num_horizontal_intervals"]) + int(record["num_vertical_intervals"]),
        )

    stats = {
        "route_grid_shape": _route_grid_shape(placedb),
        "num_route_entry_nets": int(len(route_entries or [])),
        "num_nets_with_topology": int(len(net_topologies)),
        "num_route_failed_nets": int(num_route_failed_nets),
        "unknown_name_count": int(unknown_name_count),
        "num_segments_h": int(total_segments_h),
        "num_segments_v": int(total_segments_v),
        "total_horizontal_length": int(total_length_h),
        "total_vertical_length": int(total_length_v),
        "invalid_wire_count": int(invalid_wire_count),
        "max_intervals_per_net": int(max_intervals_per_net),
    }
    cache = {
        "route_grid_shape": stats["route_grid_shape"],
        "net_topologies": net_topologies,
        "num_nets_with_topology": stats["num_nets_with_topology"],
        "num_segments_h": stats["num_segments_h"],
        "num_segments_v": stats["num_segments_v"],
    }
    logger.info(
        "Built same-net topo cache: nets=%d/%d segments_h=%d segments_v=%d route_failed=%d unknown_names=%d invalid_wires=%d max_intervals_per_net=%d",
        stats["num_nets_with_topology"],
        stats["num_route_entry_nets"],
        stats["num_segments_h"],
        stats["num_segments_v"],
        stats["num_route_failed_nets"],
        stats["unknown_name_count"],
        stats["invalid_wire_count"],
        stats["max_intervals_per_net"],
    )
    return cache, stats


def _normalize_interval(lo, hi):
    return int(min(lo, hi)), int(max(lo, hi))


def _interval_length(lo, hi):
    lo, hi = _normalize_interval(lo, hi)
    return max(int(hi) - int(lo), 1)


def _interval_overlap_ratio(candidate_lo, candidate_hi, interval_lo, interval_hi):
    cand_lo, cand_hi = _normalize_interval(candidate_lo, candidate_hi)
    int_lo, int_hi = _normalize_interval(interval_lo, interval_hi)
    overlap = max(0, min(cand_hi, int_hi) - max(cand_lo, int_lo))
    return float(overlap) / float(_interval_length(cand_lo, cand_hi))


def _to_numpy_int_array(values):
    if isinstance(values, np.ndarray):
        return values.astype(np.int32, copy=False)
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int32, copy=False)
    return np.asarray(values, dtype=np.int32)


def _find_axis_window(axis_values, target_axis, max_offset):
    axis_values = _to_numpy_int_array(axis_values)
    if axis_values.size == 0:
        return axis_values
    target_axis = int(target_axis)
    max_offset = max(int(max_offset), 0)
    lo = target_axis - max_offset
    hi = target_axis + max_offset
    left = np.searchsorted(axis_values, lo, side="left")
    right = np.searchsorted(axis_values, hi, side="right")
    return axis_values[left:right]


def _best_overlap_ratio(intervals_by_axis, axis_values, target_axis, lo, hi, max_offset=0):
    candidate_axes = _find_axis_window(axis_values, target_axis, max_offset=max_offset)
    if candidate_axes.size == 0:
        return 0.0, False
    best_ratio = 0.0
    for axis in candidate_axes.tolist():
        for int_lo, int_hi in intervals_by_axis.get(int(axis), []):
            best_ratio = max(
                best_ratio,
                _interval_overlap_ratio(lo, hi, int_lo, int_hi),
            )
    return float(best_ratio), True


def _build_same_net_topo_zero_tensors(num_edges, device, dtype):
    zeros = torch.zeros((int(num_edges),), dtype=dtype, device=device)
    mask = torch.zeros((int(num_edges),), dtype=torch.bool, device=device)
    return zeros, zeros.clone(), mask


def compute_same_net_topology_scores(
    edge_net_ids,
    x1_idx,
    y1_idx,
    x2_idx,
    y2_idx,
    topo_cache,
    *,
    max_row_offset=0,
    max_col_offset=0,
    missing_dir_penalty=0.0,
    tie_delta=0.0,
    device=None,
    dtype=torch.float32,
):
    num_edges = len(edge_net_ids) if edge_net_ids is not None else 0
    topo_cost_h, topo_cost_v, topo_observed_mask = _build_same_net_topo_zero_tensors(
        num_edges, device=device, dtype=dtype
    )
    stats = {
        "diag_edges": int(num_edges),
        "edges_with_topology": 0,
        "edges_with_observed_intervals": 0,
        "leg_fallback_ratio": 1.0 if num_edges > 0 else 0.0,
        "mean_gap": 0.0,
        "tie_ratio": 1.0 if num_edges > 0 else 0.0,
        "missing_dir_penalty": float(missing_dir_penalty),
        "max_row_offset": int(max_row_offset),
        "max_col_offset": int(max_col_offset),
    }
    if num_edges == 0 or not isinstance(topo_cache, dict):
        return topo_cost_h, topo_cost_v, topo_observed_mask, stats

    net_topologies = topo_cache.get("net_topologies", {})
    if not net_topologies:
        return topo_cost_h, topo_cost_v, topo_observed_mask, stats

    edge_net_ids_np = _to_numpy_int_array(edge_net_ids)
    x1_idx_np = _to_numpy_int_array(x1_idx)
    y1_idx_np = _to_numpy_int_array(y1_idx)
    x2_idx_np = _to_numpy_int_array(x2_idx)
    y2_idx_np = _to_numpy_int_array(y2_idx)

    topo_cost_h_np = np.zeros((num_edges,), dtype=np.float32)
    topo_cost_v_np = np.zeros((num_edges,), dtype=np.float32)
    topo_observed_mask_np = np.zeros((num_edges,), dtype=np.bool_)

    observed_leg_count = 0
    gap_sum = 0.0
    tie_count = 0
    edges_with_topology = 0
    edges_with_observed_intervals = 0

    for edge_id in range(num_edges):
        net_id = int(edge_net_ids_np[edge_id])
        record = net_topologies.get(net_id)
        if record is None:
            continue
        edges_with_topology += 1

        x_lo = min(int(x1_idx_np[edge_id]), int(x2_idx_np[edge_id]))
        x_hi = max(int(x1_idx_np[edge_id]), int(x2_idx_np[edge_id]))
        y_lo = min(int(y1_idx_np[edge_id]), int(y2_idx_np[edge_id]))
        y_hi = max(int(y1_idx_np[edge_id]), int(y2_idx_np[edge_id]))

        hv_h_ratio, hv_h_observed = _best_overlap_ratio(
            record["horizontal_by_row"],
            record["horizontal_rows"],
            int(y1_idx_np[edge_id]),
            x_lo,
            x_hi,
            max_offset=max_row_offset,
        )
        hv_v_ratio, hv_v_observed = _best_overlap_ratio(
            record["vertical_by_col"],
            record["vertical_cols"],
            int(x2_idx_np[edge_id]),
            y_lo,
            y_hi,
            max_offset=max_col_offset,
        )
        vh_v_ratio, vh_v_observed = _best_overlap_ratio(
            record["vertical_by_col"],
            record["vertical_cols"],
            int(x1_idx_np[edge_id]),
            y_lo,
            y_hi,
            max_offset=max_col_offset,
        )
        vh_h_ratio, vh_h_observed = _best_overlap_ratio(
            record["horizontal_by_row"],
            record["horizontal_rows"],
            int(y2_idx_np[edge_id]),
            x_lo,
            x_hi,
            max_offset=max_row_offset,
        )

        hv_h_cost = -hv_h_ratio if hv_h_observed else float(missing_dir_penalty)
        hv_v_cost = -hv_v_ratio if hv_v_observed else float(missing_dir_penalty)
        vh_v_cost = -vh_v_ratio if vh_v_observed else float(missing_dir_penalty)
        vh_h_cost = -vh_h_ratio if vh_h_observed else float(missing_dir_penalty)

        topo_cost_h_np[edge_id] = hv_h_cost + hv_v_cost
        topo_cost_v_np[edge_id] = vh_v_cost + vh_h_cost

        observed_count = int(hv_h_observed) + int(hv_v_observed) + int(vh_v_observed) + int(vh_h_observed)
        observed_leg_count += observed_count
        if observed_count > 0:
            topo_observed_mask_np[edge_id] = True
            edges_with_observed_intervals += 1

        gap = abs(float(topo_cost_h_np[edge_id]) - float(topo_cost_v_np[edge_id]))
        gap_sum += gap
        if gap <= float(tie_delta):
            tie_count += 1

    if num_edges > 0:
        stats["leg_fallback_ratio"] = 1.0 - float(observed_leg_count) / float(4 * num_edges)
        stats["mean_gap"] = float(gap_sum) / float(num_edges)
        stats["tie_ratio"] = float(tie_count) / float(num_edges)
    stats["edges_with_topology"] = int(edges_with_topology)
    stats["edges_with_observed_intervals"] = int(edges_with_observed_intervals)

    topo_cost_h = torch.as_tensor(topo_cost_h_np, dtype=dtype, device=device)
    topo_cost_v = torch.as_tensor(topo_cost_v_np, dtype=dtype, device=device)
    topo_observed_mask = torch.as_tensor(topo_observed_mask_np, dtype=torch.bool, device=device)
    return topo_cost_h, topo_cost_v, topo_observed_mask, stats
