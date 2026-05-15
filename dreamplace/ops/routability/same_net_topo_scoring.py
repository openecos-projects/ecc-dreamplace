import logging
import time
from collections import Counter

import numpy as np
import torch

from dreamplace.ops.routability.profile_timing import (
    l_shape_profile_enabled,
    profile_end,
    profile_scope,
    profile_start,
)

logger = logging.getLogger(__name__)

_F32_EPS = np.float32(1e-12)
_F32_SIGN_TOL = np.float32(1e-9)

try:
    import dreamplace.ops.routability.same_net_topo_scoring_cpp as same_net_topo_scoring_cpp
except ImportError:
    same_net_topo_scoring_cpp = None

_CPP_PACK_CACHE_KEY = "_same_net_topo_cpp_packed"
_CPP_PACK_CACHE_META_KEY = "_same_net_topo_cpp_pack_key"


def _normalize_net_name(net_name):
    if isinstance(net_name, bytes):
        return net_name.decode("utf-8")
    return str(net_name)


def _build_net_name_to_id(placedb):
    net_names = getattr(placedb, "net_names", [])
    try:
        cache_key = (id(net_names), len(net_names))
    except TypeError:
        cache_key = None
    if cache_key is not None:
        cached_key = getattr(placedb, "_same_net_topology_name_to_id_key", None)
        cached_mapping = getattr(placedb, "_same_net_topology_name_to_id", None)
        if cached_key == cache_key and isinstance(cached_mapping, dict):
            return cached_mapping

    mapping = {}
    for net_id, net_name in enumerate(net_names):
        mapping[_normalize_net_name(net_name)] = int(net_id)
    if cache_key is not None:
        try:
            placedb._same_net_topology_name_to_id_key = cache_key
            placedb._same_net_topology_name_to_id = mapping
        except Exception:
            pass
    return mapping


def _merge_intervals(intervals, max_gap=1):
    if not intervals:
        return []
    if len(intervals) == 1:
        lo, hi = intervals[0]
        return [(int(min(lo, hi)), int(max(lo, hi)))]
    normalized = sorted((min(lo, hi), max(lo, hi)) for lo, hi in intervals)
    merged = [[int(normalized[0][0]), int(normalized[0][1])]]
    for lo, hi in normalized[1:]:
        prev = merged[-1]
        if int(lo) <= prev[1] + int(max_gap):
            prev[1] = max(prev[1], int(hi))
        else:
            merged.append([int(lo), int(hi)])
    return [tuple(item) for item in merged]


def _merge_normalized_int_intervals(intervals, max_gap=1):
    if not intervals:
        return []
    if len(intervals) == 1:
        return [intervals[0]]

    intervals.sort()
    max_gap = int(max_gap)
    cur_lo, cur_hi = intervals[0]
    merged = []
    for lo, hi in intervals[1:]:
        if lo <= cur_hi + max_gap:
            if hi > cur_hi:
                cur_hi = hi
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged


def _interval_total_length(intervals):
    total = 0
    for lo, hi in intervals:
        total += int(hi) - int(lo)
    return int(total)


def _route_grid_shape(placedb):
    num_x = int(getattr(placedb, "num_routing_grids_x", 0))
    num_y = int(getattr(placedb, "num_routing_grids_y", 0))
    return (num_x, num_y)


def _route_grid_pack_geometry(placedb):
    route_num_bins_x, route_num_bins_y = _route_grid_shape(placedb)
    if route_num_bins_x <= 0 or route_num_bins_y <= 0:
        return None
    try:
        xl = float(getattr(placedb, "routing_grid_xl", placedb.xl))
        yl = float(getattr(placedb, "routing_grid_yl", placedb.yl))
        xh = float(getattr(placedb, "routing_grid_xh", placedb.xh))
        yh = float(getattr(placedb, "routing_grid_yh", placedb.yh))
    except Exception:
        return None
    return (
        xl,
        yl,
        (xh - xl) / float(route_num_bins_x),
        (yh - yl) / float(route_num_bins_y),
    )


def _log_topology_cache_stats(stats):
    logger.info(
        "Built per-net topology cache: nets=%d/%d segments_h=%d segments_v=%d route_failed=%d unknown_names=%d invalid_wires=%d max_intervals_per_net=%d",
        int(stats.get("num_nets_with_topology", 0)),
        int(stats.get("num_route_entry_nets", 0)),
        int(stats.get("num_segments_h", 0)),
        int(stats.get("num_segments_v", 0)),
        int(stats.get("num_route_failed_nets", 0)),
        int(stats.get("unknown_name_count", 0)),
        int(stats.get("invalid_wire_count", 0)),
        int(stats.get("max_intervals_per_net", 0)),
    )


def _build_net_topology_record(
    net_id,
    net_route,
    max_gap=1,
    return_wire_count=False,
    *,
    entries=None,
    net_name=None,
    route_failed=None,
):
    entries = (net_route.get("entries", []) if entries is None else entries) or []
    horizontal_by_row = {}
    vertical_by_col = {}
    invalid_wire_count = 0
    raw_wire_count = 0

    bbox_x_lo = None
    bbox_y_lo = None
    bbox_x_hi = None
    bbox_y_hi = None

    for entry in entries:
        entry_get = entry.get
        if entry_get("type", "") != "wire":
            continue
        raw_wire_count += 1
        x1 = int(entry_get("grid_x1", 0))
        y1 = int(entry_get("grid_y1", 0))
        x2 = int(entry_get("grid_x2", 0))
        y2 = int(entry_get("grid_y2", 0))
        orientation = entry_get("orientation", "")
        is_horizontal = orientation == "H" or y1 == y2
        is_vertical = orientation == "V" or x1 == x2

        if bbox_x_lo is None:
            bbox_x_lo = x1
            bbox_y_lo = y1
            bbox_x_hi = x1
            bbox_y_hi = y1
        else:
            bbox_x_lo = min(bbox_x_lo, x1, x2)
            bbox_y_lo = min(bbox_y_lo, y1, y2)
            bbox_x_hi = max(bbox_x_hi, x1, x2)
            bbox_y_hi = max(bbox_y_hi, y1, y2)

        if is_horizontal and y1 == y2:
            if x1 <= x2:
                lo, hi = x1, x2
            else:
                lo, hi = x2, x1
            intervals = horizontal_by_row.get(y1)
            if intervals is None:
                horizontal_by_row[y1] = [(lo, hi)]
            else:
                intervals.append((lo, hi))
            continue
        if is_vertical and x1 == x2:
            if y1 <= y2:
                lo, hi = y1, y2
            else:
                lo, hi = y2, y1
            intervals = vertical_by_col.get(x1)
            if intervals is None:
                vertical_by_col[x1] = [(lo, hi)]
            else:
                intervals.append((lo, hi))
            continue
        invalid_wire_count += 1

    merged_h = {
        int(y_idx): _merge_normalized_int_intervals(intervals, max_gap=max_gap)
        for y_idx, intervals in horizontal_by_row.items()
        if intervals
    }
    merged_v = {
        int(x_idx): _merge_normalized_int_intervals(intervals, max_gap=max_gap)
        for x_idx, intervals in vertical_by_col.items()
        if intervals
    }

    if not merged_h and not merged_v:
        if return_wire_count:
            return None, int(raw_wire_count)
        return None

    num_horizontal_intervals = sum(len(intervals) for intervals in merged_h.values())
    num_vertical_intervals = sum(len(intervals) for intervals in merged_v.values())
    total_horizontal_length = sum(
        _interval_total_length(intervals) for intervals in merged_h.values()
    )
    total_vertical_length = sum(
        _interval_total_length(intervals) for intervals in merged_v.values()
    )

    record = {
        "net_id": int(net_id),
        "net_name": (net_route.get("net_name", "") if net_name is None else net_name) or f"net_{net_id}",
        "horizontal_by_row": merged_h,
        "vertical_by_col": merged_v,
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
        "route_failed": bool(net_route.get("route_failed", False) if route_failed is None else route_failed),
    }
    if return_wire_count:
        return record, int(raw_wire_count)
    return record


def _build_same_net_topology_cache_python(
    route_entries,
    placedb,
    max_gap=1,
    profile_enabled=False,
    net_name_to_id=None,
):
    profile_active = l_shape_profile_enabled(profile_enabled)
    if net_name_to_id is None:
        net_name_to_id = _build_net_name_to_id(placedb)
    net_topologies = {}
    route_entry_meta = {}
    num_route_failed_nets = 0
    unknown_name_count = 0
    total_segments_h = 0
    total_segments_v = 0
    total_length_h = 0
    total_length_v = 0
    max_intervals_per_net = 0
    invalid_wire_count = 0

    route_loop_timer = profile_start(profile_enabled)
    meta_elapsed_s = 0.0
    wire_count_elapsed_s = 0.0
    record_build_elapsed_s = 0.0
    stats_elapsed_s = 0.0
    for net_route in route_entries or []:
        meta_start = time.perf_counter() if profile_active else None
        net_route_get = net_route.get
        route_failed = bool(net_route_get("route_failed", False))
        if route_failed:
            num_route_failed_nets += 1
        net_name = net_route_get("net_name", "")
        if net_name in net_name_to_id:
            net_id = net_name_to_id[net_name]
        else:
            net_id = int(net_route_get("net_id", -1))
            if net_id < 0:
                unknown_name_count += 1
                continue
        entries = net_route_get("entries", []) or []
        if profile_active:
            meta_elapsed_s += time.perf_counter() - meta_start
            record_build_start = time.perf_counter()
        record, wire_entry_count = _build_net_topology_record(
            int(net_id),
            net_route,
            max_gap=max_gap,
            return_wire_count=True,
            entries=entries,
            net_name=net_name,
            route_failed=route_failed,
        )
        if profile_active:
            record_build_elapsed_s += time.perf_counter() - record_build_start
            meta_start = time.perf_counter()
        route_entry_meta[int(net_id)] = {
            "net_id": int(net_id),
            "net_name": net_name or f"net_{net_id}",
            "route_failed": route_failed,
            "entry_count": int(len(entries)),
            "wire_entry_count": int(wire_entry_count),
        }
        if profile_active:
            meta_elapsed_s += time.perf_counter() - meta_start
        if record is None:
            continue
        stats_start = time.perf_counter() if profile_active else None
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
        if profile_active:
            stats_elapsed_s += time.perf_counter() - stats_start
    profile_end(
        profile_enabled,
        route_loop_timer,
        "same_net_topology_cache.route_loop",
        logger=logger,
        route_nets=len(route_entries or []),
        topo_nets=len(net_topologies),
        meta_ms=f"{meta_elapsed_s * 1000.0:.3f}",
        wire_count_ms=f"{wire_count_elapsed_s * 1000.0:.3f}",
        record_build_ms=f"{record_build_elapsed_s * 1000.0:.3f}",
        stats_ms=f"{stats_elapsed_s * 1000.0:.3f}",
        backend="python",
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
        "backend": "python",
    }
    cache = {
        "route_grid_shape": stats["route_grid_shape"],
        "net_topologies": net_topologies,
        "route_entry_meta": route_entry_meta,
        "num_nets_with_topology": stats["num_nets_with_topology"],
        "num_segments_h": stats["num_segments_h"],
        "num_segments_v": stats["num_segments_v"],
    }
    _log_topology_cache_stats(stats)
    return cache, stats


def _build_same_net_topology_cache_cpp(
    route_entries,
    placedb,
    *,
    net_name_to_id,
    max_gap=1,
    profile_enabled=False,
):
    if same_net_topo_scoring_cpp is None or not hasattr(same_net_topo_scoring_cpp, "build_topology_cache"):
        return None

    route_entries = route_entries or []
    geometry = _route_grid_pack_geometry(placedb)
    build_packed = geometry is not None
    if geometry is None:
        xl, yl, route_bin_size_x, route_bin_size_y = 0.0, 0.0, 1.0, 1.0
    else:
        xl, yl, route_bin_size_x, route_bin_size_y = geometry
    build_python_cache = bool(getattr(profile_enabled, "l_shape_plot_flag", 0)) or not build_packed

    route_loop_timer = profile_start(profile_enabled)
    try:
        cache, stats = same_net_topo_scoring_cpp.build_topology_cache(
            route_entries,
            net_name_to_id,
            _route_grid_shape(placedb),
            int(max_gap),
            float(xl),
            float(yl),
            float(route_bin_size_x),
            float(route_bin_size_y),
            bool(build_packed),
            bool(build_python_cache),
        )
    except Exception:
        logger.exception("C++ per-net topology cache build failed; falling back to Python builder.")
        return None

    cache = dict(cache or {})
    stats = dict(stats or {})
    stats.setdefault("route_grid_shape", _route_grid_shape(placedb))
    stats.setdefault("num_route_entry_nets", int(len(route_entries)))
    stats.setdefault("num_nets_with_topology", int(cache.get("num_nets_with_topology", 0)))
    stats.setdefault("num_route_failed_nets", 0)
    stats.setdefault("unknown_name_count", 0)
    stats.setdefault("num_segments_h", int(cache.get("num_segments_h", 0)))
    stats.setdefault("num_segments_v", int(cache.get("num_segments_v", 0)))
    stats.setdefault("total_horizontal_length", 0)
    stats.setdefault("total_vertical_length", 0)
    stats.setdefault("invalid_wire_count", 0)
    stats.setdefault("max_intervals_per_net", 0)
    stats["backend"] = "cpp"
    profile_end(
        profile_enabled,
        route_loop_timer,
        "same_net_topology_cache.route_loop",
        logger=logger,
        route_nets=len(route_entries),
        topo_nets=int(stats.get("num_nets_with_topology", 0)),
        h_segments=int(stats.get("num_segments_h", 0)),
        v_segments=int(stats.get("num_segments_v", 0)),
        prepacked=int(build_packed),
        compact=int(not build_python_cache),
        backend="cpp",
    )
    _log_topology_cache_stats(stats)
    return cache, stats


def _build_same_net_topology_cache_prebuilt(
    prebuilt_cache,
    prebuilt_stats,
    placedb,
    *,
    profile_enabled=False,
):
    if same_net_topo_scoring_cpp is None or not isinstance(prebuilt_cache, dict):
        return None

    needs_python_cache = bool(getattr(profile_enabled, "l_shape_plot_flag", 0))
    if needs_python_cache and not prebuilt_cache.get("net_topologies"):
        return None

    packed = prebuilt_cache.get(_CPP_PACK_CACHE_KEY)
    if not isinstance(packed, dict):
        return None

    geometry = _route_grid_pack_geometry(placedb)
    if geometry is None:
        return None

    pack_key = tuple(float(v) for v in geometry)
    cached_key = prebuilt_cache.get(_CPP_PACK_CACHE_META_KEY)
    try:
        cached_key = tuple(float(v) for v in cached_key)
    except Exception:
        return None
    if cached_key != pack_key:
        logger.warning(
            "Ignore prebuilt same-net topology cache because geometry key mismatches: prebuilt=%s expected=%s",
            cached_key,
            pack_key,
        )
        return None

    cache = dict(prebuilt_cache)
    cache.setdefault("route_grid_shape", _route_grid_shape(placedb))
    cache.setdefault("net_topologies", {})
    cache.setdefault("route_entry_meta", {})
    cache.setdefault("num_nets_with_topology", int(packed["net_ids"].size))
    cache.setdefault("num_segments_h", int(packed["h_x1"].size))
    cache.setdefault("num_segments_v", int(packed["v_x"].size))
    cache.setdefault("compact_topology_cache", True)

    stats = dict(prebuilt_stats or {})
    stats.setdefault("route_grid_shape", _route_grid_shape(placedb))
    stats.setdefault("num_route_entry_nets", 0)
    stats.setdefault("num_nets_with_topology", int(cache.get("num_nets_with_topology", 0)))
    stats.setdefault("num_route_failed_nets", 0)
    stats.setdefault("unknown_name_count", 0)
    stats.setdefault("num_segments_h", int(cache.get("num_segments_h", 0)))
    stats.setdefault("num_segments_v", int(cache.get("num_segments_v", 0)))
    stats.setdefault("total_horizontal_length", 0)
    stats.setdefault("total_vertical_length", 0)
    stats.setdefault("total_entry_count", 0)
    stats.setdefault("wire_entry_count", int(stats.get("num_segments_h", 0)) + int(stats.get("num_segments_v", 0)))
    stats.setdefault("invalid_wire_count", 0)
    stats.setdefault("max_intervals_per_net", 0)
    stats["backend"] = stats.get("backend", "gpugr_packed")
    _log_topology_cache_stats(stats)
    return cache, stats


def build_same_net_topology_cache(
    route_entries,
    placedb,
    max_gap=1,
    profile_enabled=False,
    prebuilt_cache=None,
    prebuilt_stats=None,
):
    prebuilt_result = _build_same_net_topology_cache_prebuilt(
        prebuilt_cache,
        prebuilt_stats,
        placedb,
        profile_enabled=profile_enabled,
    )
    if prebuilt_result is not None:
        return prebuilt_result

    with profile_scope(
        profile_enabled,
        "same_net_topology_cache.build_net_name_to_id",
        logger=logger,
    ):
        net_name_to_id = _build_net_name_to_id(placedb)

    cpp_result = _build_same_net_topology_cache_cpp(
        route_entries,
        placedb,
        net_name_to_id=net_name_to_id,
        max_gap=max_gap,
        profile_enabled=profile_enabled,
    )
    if cpp_result is not None:
        return cpp_result

    return _build_same_net_topology_cache_python(
        route_entries,
        placedb,
        max_gap=max_gap,
        profile_enabled=profile_enabled,
        net_name_to_id=net_name_to_id,
    )


def pack_same_net_topology_cache_for_cpp(
    topo_cache,
    *,
    xl,
    yl,
    route_bin_size_x,
    route_bin_size_y,
    profile_enabled=False,
):
    if not isinstance(topo_cache, dict):
        return None

    pack_key = (
        float(xl),
        float(yl),
        float(route_bin_size_x),
        float(route_bin_size_y),
    )
    cached_key = topo_cache.get(_CPP_PACK_CACHE_META_KEY)
    cached_pack = topo_cache.get(_CPP_PACK_CACHE_KEY)
    if cached_key == pack_key and isinstance(cached_pack, dict):
        profile_end(
            profile_enabled,
            profile_start(profile_enabled),
            "same_net_topo_scoring.pack_cpp_cache_hit",
            logger=logger,
            nets=int(cached_pack["net_ids"].size),
            h_segments=int(cached_pack["h_x1"].size),
            v_segments=int(cached_pack["v_x"].size),
        )
        return cached_pack

    net_topologies = topo_cache.get("net_topologies", {})
    if not net_topologies:
        return None

    net_ids = []
    h_seg_offsets = [0]
    v_seg_offsets = [0]
    h_x1 = []
    h_y = []
    h_x2 = []
    v_x = []
    v_y1 = []
    v_y2 = []
    route_grid_shape = topo_cache.get("route_grid_shape", (0, 0))
    try:
        route_num_bins_x = max(int(route_grid_shape[0]), 0)
        route_num_bins_y = max(int(route_grid_shape[1]), 0)
    except Exception:
        route_num_bins_x = 0
        route_num_bins_y = 0
    x_centers = np.float32(xl) + (
        np.arange(route_num_bins_x, dtype=np.float32) + np.float32(0.5)
    ) * np.float32(route_bin_size_x)
    y_centers = np.float32(yl) + (
        np.arange(route_num_bins_y, dtype=np.float32) + np.float32(0.5)
    ) * np.float32(route_bin_size_y)
    x_center_count = int(x_centers.size)
    y_center_count = int(y_centers.size)

    pack_timer = profile_start(profile_enabled)
    loop_elapsed_s = 0.0
    array_elapsed_s = 0.0
    loop_start = time.perf_counter() if l_shape_profile_enabled(profile_enabled) else None
    for net_id in sorted(net_topologies.keys()):
        record = net_topologies[net_id]
        net_ids.append(int(net_id))

        h_count = 0
        for row_idx in sorted(record.get("horizontal_by_row", {}).keys()):
            seg_y = y_centers[row_idx] if 0 <= row_idx < y_center_count else _grid_center(row_idx, yl, route_bin_size_y)
            intervals = record["horizontal_by_row"][row_idx]
            for int_lo, int_hi in intervals:
                h_x1.append(x_centers[int_lo] if 0 <= int_lo < x_center_count else _grid_center(int_lo, xl, route_bin_size_x))
                h_y.append(seg_y)
                h_x2.append(x_centers[int_hi] if 0 <= int_hi < x_center_count else _grid_center(int_hi, xl, route_bin_size_x))
                h_count += 1
        h_seg_offsets.append(h_seg_offsets[-1] + h_count)

        v_count = 0
        for col_idx in sorted(record.get("vertical_by_col", {}).keys()):
            seg_x = x_centers[col_idx] if 0 <= col_idx < x_center_count else _grid_center(col_idx, xl, route_bin_size_x)
            intervals = record["vertical_by_col"][col_idx]
            for int_lo, int_hi in intervals:
                v_x.append(seg_x)
                v_y1.append(y_centers[int_lo] if 0 <= int_lo < y_center_count else _grid_center(int_lo, yl, route_bin_size_y))
                v_y2.append(y_centers[int_hi] if 0 <= int_hi < y_center_count else _grid_center(int_hi, yl, route_bin_size_y))
                v_count += 1
        v_seg_offsets.append(v_seg_offsets[-1] + v_count)
    if loop_start is not None:
        loop_elapsed_s = time.perf_counter() - loop_start

    array_start = time.perf_counter() if l_shape_profile_enabled(profile_enabled) else None
    net_ids_np = np.asarray(net_ids, dtype=np.int32)
    if net_ids_np.size:
        max_net_id = int(net_ids_np.max())
        net_index_by_id = np.full(max_net_id + 1, -1, dtype=np.int32)
        valid_net_ids = net_ids_np >= 0
        if valid_net_ids.any():
            net_index_by_id[net_ids_np[valid_net_ids]] = np.nonzero(valid_net_ids)[0].astype(np.int32)
    else:
        net_index_by_id = np.zeros(0, dtype=np.int32)

    packed = {
        "net_ids": net_ids_np,
        "net_index_by_id": net_index_by_id,
        "h_seg_offsets": np.asarray(h_seg_offsets, dtype=np.int32),
        "v_seg_offsets": np.asarray(v_seg_offsets, dtype=np.int32),
        "h_x1": np.asarray(h_x1, dtype=np.float32),
        "h_y": np.asarray(h_y, dtype=np.float32),
        "h_x2": np.asarray(h_x2, dtype=np.float32),
        "v_x": np.asarray(v_x, dtype=np.float32),
        "v_y1": np.asarray(v_y1, dtype=np.float32),
        "v_y2": np.asarray(v_y2, dtype=np.float32),
    }
    if array_start is not None:
        array_elapsed_s = time.perf_counter() - array_start
    topo_cache[_CPP_PACK_CACHE_META_KEY] = pack_key
    topo_cache[_CPP_PACK_CACHE_KEY] = packed
    profile_end(
        profile_enabled,
        pack_timer,
        "same_net_topo_scoring.pack_cpp",
        logger=logger,
        nets=int(packed["net_ids"].size),
        h_segments=int(packed["h_x1"].size),
        v_segments=int(packed["v_x"].size),
        loop_ms=f"{loop_elapsed_s * 1000.0:.3f}",
        array_ms=f"{array_elapsed_s * 1000.0:.3f}",
    )
    logger.info(
        "Packed per-net topology cache for C++: nets=%d h_segments=%d v_segments=%d",
        int(packed["net_ids"].size),
        int(packed["h_x1"].size),
        int(packed["v_x"].size),
    )
    return packed


def _to_numpy_int_array(values):
    if isinstance(values, np.ndarray):
        return values.astype(np.int32, copy=False)
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int32, copy=False)
    return np.asarray(values, dtype=np.int32)


def _to_numpy_float_array(values):
    if isinstance(values, np.ndarray):
        return values.astype(np.float32, copy=False)
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def _build_same_net_topo_zero_tensors(num_edges, device, dtype):
    zeros = torch.zeros((int(num_edges),), dtype=dtype, device=device)
    mask = torch.zeros((int(num_edges),), dtype=torch.bool, device=device)
    return zeros, zeros.clone(), mask


def _grid_center(coord_idx, origin, step):
    origin = np.float32(origin)
    coord_idx = np.float32(coord_idx)
    step = np.float32(step)
    return np.float32(origin + coord_idx * step + np.float32(0.5) * step)


def _interval_overlap_len(lo1, hi1, lo2, hi2):
    lo1, hi1 = sorted((np.float32(lo1), np.float32(hi1)))
    lo2, hi2 = sorted((np.float32(lo2), np.float32(hi2)))
    return np.float32(max(np.float32(0.0), min(hi1, hi2) - max(lo1, lo2)))


def _interval_gap(lo1, hi1, lo2, hi2):
    lo1, hi1 = sorted((np.float32(lo1), np.float32(hi1)))
    lo2, hi2 = sorted((np.float32(lo2), np.float32(hi2)))
    if _interval_overlap_len(lo1, hi1, lo2, hi2) > np.float32(0.0):
        return np.float32(0.0)
    if hi1 < lo2:
        return np.float32(lo2 - hi1)
    if hi2 < lo1:
        return np.float32(lo1 - hi2)
    return np.float32(0.0)


def _cross_value(dx, dy, x1, y1, qx, qy):
    dx = np.float32(dx)
    dy = np.float32(dy)
    x1 = np.float32(x1)
    y1 = np.float32(y1)
    qx = np.float32(qx)
    qy = np.float32(qy)
    return np.float32(dx * (qy - y1) - dy * (qx - x1))


def _sign_with_tol(value, tol=_F32_SIGN_TOL):
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def _segment_length(x1, y1, x2, y2):
    x1 = np.float32(x1)
    y1 = np.float32(y1)
    x2 = np.float32(x2)
    y2 = np.float32(y2)
    return np.float32(max(abs(x2 - x1) + abs(y2 - y1), np.float32(1e-9)))


def _split_horizontal_segment_by_line(sx1, sy, sx2, x1, y1, dx, dy):
    sx_lo, sx_hi = sorted((np.float32(sx1), np.float32(sx2)))
    cross1 = _cross_value(dx, dy, x1, y1, sx_lo, sy)
    cross2 = _cross_value(dx, dy, x1, y1, sx_hi, sy)
    sign1 = _sign_with_tol(cross1)
    sign2 = _sign_with_tol(cross2)

    if sign1 == 0 and sign2 == 0:
        return [(sx_lo, sy, sx_hi, sy)]
    if sign1 == sign2 or sign1 == 0 or sign2 == 0 or abs(np.float32(dy)) <= _F32_EPS:
        return [(sx_lo, sy, sx_hi, sy)]

    x_int = np.float32(x1) + np.float32(dx) * ((np.float32(sy) - np.float32(y1)) / np.float32(dy))
    x_int = min(max(x_int, sx_lo), sx_hi)
    if x_int <= sx_lo + _F32_EPS or x_int >= sx_hi - _F32_EPS:
        return [(sx_lo, sy, sx_hi, sy)]
    return [
        (sx_lo, sy, x_int, sy),
        (x_int, sy, sx_hi, sy),
    ]


def _split_vertical_segment_by_line(sx, sy1, sy2, x1, y1, dx, dy):
    sy_lo, sy_hi = sorted((np.float32(sy1), np.float32(sy2)))
    cross1 = _cross_value(dx, dy, x1, y1, sx, sy_lo)
    cross2 = _cross_value(dx, dy, x1, y1, sx, sy_hi)
    sign1 = _sign_with_tol(cross1)
    sign2 = _sign_with_tol(cross2)

    if sign1 == 0 and sign2 == 0:
        return [(sx, sy_lo, sx, sy_hi)]
    if sign1 == sign2 or sign1 == 0 or sign2 == 0 or abs(np.float32(dx)) <= _F32_EPS:
        return [(sx, sy_lo, sx, sy_hi)]

    y_int = np.float32(y1) + np.float32(dy) * ((np.float32(sx) - np.float32(x1)) / np.float32(dx))
    y_int = min(max(y_int, sy_lo), sy_hi)
    if y_int <= sy_lo + _F32_EPS or y_int >= sy_hi - _F32_EPS:
        return [(sx, sy_lo, sx, sy_hi)]
    return [
        (sx, sy_lo, sx, y_int),
        (sx, y_int, sx, sy_hi),
    ]


def _piece_side_affinity(piece_x1, piece_y1, piece_x2, piece_y2, x1, y1, dx, dy, sign_h):
    mid_x = np.float32(0.5) * (np.float32(piece_x1) + np.float32(piece_x2))
    mid_y = np.float32(0.5) * (np.float32(piece_y1) + np.float32(piece_y2))
    side_val = _cross_value(dx, dy, x1, y1, mid_x, mid_y)
    piece_sign = _sign_with_tol(side_val)
    if piece_sign == 0:
        return 0.5, 0.5
    if piece_sign == int(sign_h):
        return 1.0, 0.0
    return 0.0, 1.0


def _horizontal_leg_affinity(piece_x1, piece_y, piece_x2, leg_y, leg_x1, leg_x2, sigma, max_distance):
    piece_lo, piece_hi = sorted((np.float32(piece_x1), np.float32(piece_x2)))
    leg_lo, leg_hi = sorted((np.float32(leg_x1), np.float32(leg_x2)))
    x_gap = _interval_gap(piece_lo, piece_hi, leg_lo, leg_hi)
    y_offset = abs(np.float32(piece_y) - np.float32(leg_y))
    raw_dist = y_offset + x_gap
    if np.float32(max_distance) > np.float32(0.0) and raw_dist > np.float32(max_distance):
        return np.float32(0.0)
    piece_len = max(piece_hi - piece_lo, np.float32(1e-9))
    dist_norm = raw_dist / piece_len
    return np.float32(np.exp(-dist_norm / max(np.float32(sigma), np.float32(1e-9))))


def _vertical_leg_affinity(piece_x, piece_y1, piece_y2, leg_x, leg_y1, leg_y2, sigma, max_distance):
    piece_lo, piece_hi = sorted((np.float32(piece_y1), np.float32(piece_y2)))
    leg_lo, leg_hi = sorted((np.float32(leg_y1), np.float32(leg_y2)))
    y_gap = _interval_gap(piece_lo, piece_hi, leg_lo, leg_hi)
    x_offset = abs(np.float32(piece_x) - np.float32(leg_x))
    raw_dist = x_offset + y_gap
    if np.float32(max_distance) > np.float32(0.0) and raw_dist > np.float32(max_distance):
        return np.float32(0.0)
    piece_len = max(piece_hi - piece_lo, np.float32(1e-9))
    dist_norm = raw_dist / piece_len
    return np.float32(np.exp(-dist_norm / max(np.float32(sigma), np.float32(1e-9))))


def _append_sample(bucket, edge_id, net_id, cost_h, cost_v, gap, observed_count):
    if len(bucket) >= 12:
        return
    bucket.append(
        {
            "edge_id": int(edge_id),
            "net_id": int(net_id),
            "cost_h": float(cost_h),
            "cost_v": float(cost_v),
            "gap": float(gap),
            "observed_count": int(observed_count),
        }
    )


def _append_edge_reason_sample(
    bucket,
    *,
    edge_id,
    net_id,
    net_name,
    reason,
    x1,
    y1,
    x2,
    y2,
    observed_piece_count=0,
    score_h=0.0,
    score_v=0.0,
    total_support=0.0,
    route_failed=False,
    route_entry_present=False,
    wire_entry_count=0,
    bbox=None,
    limit=12,
):
    if len(bucket) >= int(limit):
        return
    sample = {
        "edge_id": int(edge_id),
        "net_id": int(net_id),
        "net_name": _normalize_net_name(net_name),
        "reason": str(reason),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "observed_piece_count": int(observed_piece_count),
        "score_h": float(score_h),
        "score_v": float(score_v),
        "total_support": float(total_support),
        "route_failed": bool(route_failed),
        "route_entry_present": bool(route_entry_present),
        "wire_entry_count": int(wire_entry_count),
    }
    if bbox is not None:
        sample["bbox"] = [int(v) for v in bbox]
    bucket.append(sample)


def _counter_top_list(counter, net_name_lookup=None, topk=12):
    items = []
    for net_id, count in counter.most_common(int(topk)):
        net_name = None
        if isinstance(net_name_lookup, dict):
            net_name = net_name_lookup.get(int(net_id), None)
        items.append(
            {
                "net_id": int(net_id),
                "net_name": _normalize_net_name(net_name if net_name is not None else f"net_{int(net_id)}"),
                "count": int(count),
            }
        )
    return items


def _compute_diagonal_split_topo_costs_python(
    edge_net_ids,
    x1,
    y1,
    x2,
    y2,
    topo_cache,
    *,
    xl,
    yl,
    route_bin_size_x,
    route_bin_size_y,
    sigma=1.0,
    min_support=1e-6,
    max_distance=0.0,
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
        "edges_missing_topology": 0,
        "edges_missing_topology_no_route_entry": 0,
        "edges_missing_topology_route_failed": 0,
        "edges_missing_topology_no_wire_entries": 0,
        "edges_missing_topology_other": 0,
        "edges_with_zero_observed_pieces": 0,
        "edges_with_weak_support": 0,
        "mean_gap": 0.0,
        "tie_ratio": 1.0 if num_edges > 0 else 0.0,
        "sigma": float(sigma),
        "min_support": float(min_support),
        "max_distance": float(max_distance),
    }
    if num_edges == 0 or not isinstance(topo_cache, dict):
        return topo_cost_h, topo_cost_v, topo_observed_mask, stats

    net_topologies = topo_cache.get("net_topologies", {})
    route_entry_meta = topo_cache.get("route_entry_meta", {})
    net_name_lookup = {}
    for net_id, record in net_topologies.items():
        net_name_lookup[int(net_id)] = record.get("net_name", f"net_{int(net_id)}")
    for net_id, meta in route_entry_meta.items():
        net_name_lookup.setdefault(int(net_id), meta.get("net_name", f"net_{int(net_id)}"))
    if not net_topologies:
        return topo_cost_h, topo_cost_v, topo_observed_mask, stats

    edge_net_ids_np = _to_numpy_int_array(edge_net_ids)
    x1_np = _to_numpy_float_array(x1)
    y1_np = _to_numpy_float_array(y1)
    x2_np = _to_numpy_float_array(x2)
    y2_np = _to_numpy_float_array(y2)

    topo_cost_h_np = np.zeros((num_edges,), dtype=np.float32)
    topo_cost_v_np = np.zeros((num_edges,), dtype=np.float32)
    topo_observed_mask_np = np.zeros((num_edges,), dtype=np.bool_)

    gap_sum = 0.0
    tie_count = 0
    edges_with_topology = 0
    edges_with_observed_intervals = 0
    zero_zero_count = 0
    observed_zero_zero_count = 0
    exact_equal_count = 0
    hv_support_count = 0
    vh_support_count = 0
    both_paths_supported_count = 0
    gap_values = np.zeros((num_edges,), dtype=np.float32)
    sample_observed_zero_zero = []
    sample_unobserved_zero_zero = []
    sample_exact_equal = []
    sample_missing_topology = []
    sample_zero_support = []
    sample_weak_support = []
    missing_topology_counter = Counter()
    zero_support_counter = Counter()
    weak_support_counter = Counter()

    for edge_id in range(num_edges):
        net_id = int(edge_net_ids_np[edge_id])
        record = net_topologies.get(net_id)
        ex1 = float(x1_np[edge_id])
        ey1 = float(y1_np[edge_id])
        ex2 = float(x2_np[edge_id])
        ey2 = float(y2_np[edge_id])
        if record is None:
            stats["edges_missing_topology"] += 1
            missing_topology_counter[int(net_id)] += 1
            meta = route_entry_meta.get(int(net_id))
            route_entry_present = meta is not None
            route_failed = bool(meta.get("route_failed", False)) if isinstance(meta, dict) else False
            wire_entry_count = int(meta.get("wire_entry_count", 0)) if isinstance(meta, dict) else 0
            if not route_entry_present:
                reason = "no_route_entry"
                stats["edges_missing_topology_no_route_entry"] += 1
            elif route_failed:
                reason = "route_failed"
                stats["edges_missing_topology_route_failed"] += 1
            elif wire_entry_count <= 0:
                reason = "no_wire_entries"
                stats["edges_missing_topology_no_wire_entries"] += 1
            else:
                reason = "route_entry_without_topology"
                stats["edges_missing_topology_other"] += 1
            _append_edge_reason_sample(
                sample_missing_topology,
                edge_id=edge_id,
                net_id=net_id,
                net_name=(meta or {}).get("net_name", f"net_{net_id}"),
                reason=reason,
                x1=ex1,
                y1=ey1,
                x2=ex2,
                y2=ey2,
                route_failed=route_failed,
                route_entry_present=route_entry_present,
                wire_entry_count=wire_entry_count,
            )
            continue
        edges_with_topology += 1

        dx = ex2 - ex1
        dy = ey2 - ey1
        if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
            continue

        sign_h = _sign_with_tol(_cross_value(dx, dy, ex1, ey1, ex2, ey1))
        if sign_h == 0:
            sign_h = 1

        hv_h_leg = (ex1, ey1, ex2, ey1)
        hv_v_leg = (ex2, ey1, ex2, ey2)
        vh_v_leg = (ex1, ey1, ex1, ey2)
        vh_h_leg = (ex1, ey2, ex2, ey2)

        score_h = 0.0
        score_v = 0.0
        observed_piece_count = 0

        for row_idx, intervals in record.get("horizontal_by_row", {}).items():
            seg_y = _grid_center(row_idx, yl, route_bin_size_y)
            for int_lo, int_hi in intervals:
                seg_x1 = _grid_center(int_lo, xl, route_bin_size_x)
                seg_x2 = _grid_center(int_hi, xl, route_bin_size_x)
                for piece_x1, piece_y1, piece_x2, piece_y2 in _split_horizontal_segment_by_line(
                    seg_x1, seg_y, seg_x2, ex1, ey1, dx, dy
                ):
                    piece_len = _segment_length(piece_x1, piece_y1, piece_x2, piece_y2)
                    alpha_h, alpha_v = _piece_side_affinity(
                        piece_x1, piece_y1, piece_x2, piece_y2, ex1, ey1, dx, dy, sign_h
                    )
                    aff_h = _horizontal_leg_affinity(
                        piece_x1,
                        piece_y1,
                        piece_x2,
                        hv_h_leg[1],
                        hv_h_leg[0],
                        hv_h_leg[2],
                        sigma,
                        max_distance,
                    )
                    aff_v = _horizontal_leg_affinity(
                        piece_x1,
                        piece_y1,
                        piece_x2,
                        vh_h_leg[1],
                        vh_h_leg[0],
                        vh_h_leg[2],
                        sigma,
                        max_distance,
                    )
                    support_h_piece = piece_len * alpha_h * aff_h
                    support_v_piece = piece_len * alpha_v * aff_v
                    if support_h_piece > 0.0 or support_v_piece > 0.0:
                        observed_piece_count += 1
                    score_h += support_h_piece
                    score_v += support_v_piece

        for col_idx, intervals in record.get("vertical_by_col", {}).items():
            seg_x = _grid_center(col_idx, xl, route_bin_size_x)
            for int_lo, int_hi in intervals:
                seg_y1 = _grid_center(int_lo, yl, route_bin_size_y)
                seg_y2 = _grid_center(int_hi, yl, route_bin_size_y)
                for piece_x1, piece_y1, piece_x2, piece_y2 in _split_vertical_segment_by_line(
                    seg_x, seg_y1, seg_y2, ex1, ey1, dx, dy
                ):
                    piece_len = _segment_length(piece_x1, piece_y1, piece_x2, piece_y2)
                    alpha_h, alpha_v = _piece_side_affinity(
                        piece_x1, piece_y1, piece_x2, piece_y2, ex1, ey1, dx, dy, sign_h
                    )
                    aff_h = _vertical_leg_affinity(
                        piece_x1,
                        piece_y1,
                        piece_y2,
                        hv_v_leg[0],
                        hv_v_leg[1],
                        hv_v_leg[3],
                        sigma,
                        max_distance,
                    )
                    aff_v = _vertical_leg_affinity(
                        piece_x1,
                        piece_y1,
                        piece_y2,
                        vh_v_leg[0],
                        vh_v_leg[1],
                        vh_v_leg[3],
                        sigma,
                        max_distance,
                    )
                    support_h_piece = piece_len * alpha_h * aff_h
                    support_v_piece = piece_len * alpha_v * aff_v
                    if support_h_piece > 0.0 or support_v_piece > 0.0:
                        observed_piece_count += 1
                    score_h += support_h_piece
                    score_v += support_v_piece

        if score_h > float(min_support):
            hv_support_count += 1
        if score_v > float(min_support):
            vh_support_count += 1
        if score_h > float(min_support) and score_v > float(min_support):
            both_paths_supported_count += 1

        total_support = score_h + score_v
        if total_support > float(min_support):
            weight_h = score_h / total_support
            weight_v = score_v / total_support
            topo_cost_h_np[edge_id] = float(-np.log(weight_h + 1e-12))
            topo_cost_v_np[edge_id] = float(-np.log(weight_v + 1e-12))
            topo_observed_mask_np[edge_id] = True
            edges_with_observed_intervals += 1
        else:
            topo_cost_h_np[edge_id] = 0.0
            topo_cost_v_np[edge_id] = 0.0
            if observed_piece_count <= 0:
                stats["edges_with_zero_observed_pieces"] += 1
                zero_support_counter[int(net_id)] += 1
                _append_edge_reason_sample(
                    sample_zero_support,
                    edge_id=edge_id,
                    net_id=net_id,
                    net_name=record.get("net_name", f"net_{net_id}"),
                    reason="zero_observed_support",
                    x1=ex1,
                    y1=ey1,
                    x2=ex2,
                    y2=ey2,
                    observed_piece_count=observed_piece_count,
                    score_h=score_h,
                    score_v=score_v,
                    total_support=total_support,
                    route_entry_present=True,
                    wire_entry_count=int(record.get("raw_wire_count", 0)),
                    bbox=record.get("bbox", None),
                )
            else:
                stats["edges_with_weak_support"] += 1
                weak_support_counter[int(net_id)] += 1
                _append_edge_reason_sample(
                    sample_weak_support,
                    edge_id=edge_id,
                    net_id=net_id,
                    net_name=record.get("net_name", f"net_{net_id}"),
                    reason="weak_support_below_threshold",
                    x1=ex1,
                    y1=ey1,
                    x2=ex2,
                    y2=ey2,
                    observed_piece_count=observed_piece_count,
                    score_h=score_h,
                    score_v=score_v,
                    total_support=total_support,
                    route_entry_present=True,
                    wire_entry_count=int(record.get("raw_wire_count", 0)),
                    bbox=record.get("bbox", None),
                )

        gap = abs(float(topo_cost_h_np[edge_id]) - float(topo_cost_v_np[edge_id]))
        gap_values[edge_id] = gap
        gap_sum += gap
        if gap <= 1e-6:
            tie_count += 1
        if gap <= 1e-12:
            exact_equal_count += 1
            _append_sample(
                sample_exact_equal,
                edge_id,
                net_id,
                topo_cost_h_np[edge_id],
                topo_cost_v_np[edge_id],
                gap,
                observed_piece_count,
            )
        if abs(float(topo_cost_h_np[edge_id])) <= 1e-12 and abs(float(topo_cost_v_np[edge_id])) <= 1e-12:
            zero_zero_count += 1
            if observed_piece_count > 0:
                observed_zero_zero_count += 1
                _append_sample(
                    sample_observed_zero_zero,
                    edge_id,
                    net_id,
                    topo_cost_h_np[edge_id],
                    topo_cost_v_np[edge_id],
                    gap,
                    observed_piece_count,
                )
            else:
                _append_sample(
                    sample_unobserved_zero_zero,
                    edge_id,
                    net_id,
                    topo_cost_h_np[edge_id],
                    topo_cost_v_np[edge_id],
                    gap,
                    observed_piece_count,
                )

    if num_edges > 0:
        stats["mean_gap"] = float(gap_sum) / float(num_edges)
        stats["tie_ratio"] = float(tie_count) / float(num_edges)
        stats["zero_zero_edges"] = int(zero_zero_count)
        stats["observed_zero_zero_edges"] = int(observed_zero_zero_count)
        stats["unobserved_zero_zero_edges"] = int(zero_zero_count - observed_zero_zero_count)
        stats["exact_equal_edges"] = int(exact_equal_count)
        stats["hv_path_observed_edges"] = int(hv_support_count)
        stats["vh_path_observed_edges"] = int(vh_support_count)
        stats["both_paths_observed_edges"] = int(both_paths_supported_count)
        stats["gap_p50"] = float(np.quantile(gap_values, 0.5))
        stats["gap_p90"] = float(np.quantile(gap_values, 0.9))
        nonzero_gaps = gap_values[gap_values > 1e-12]
        stats["nonzero_gap_edges"] = int(nonzero_gaps.size)
        stats["nonzero_gap_p50"] = (
            float(np.quantile(nonzero_gaps, 0.5)) if nonzero_gaps.size > 0 else 0.0
        )
        stats["sample_observed_zero_zero"] = sample_observed_zero_zero
        stats["sample_unobserved_zero_zero"] = sample_unobserved_zero_zero
        stats["sample_exact_equal"] = sample_exact_equal
        stats["sample_missing_topology"] = sample_missing_topology
        stats["sample_zero_support"] = sample_zero_support
        stats["sample_weak_support"] = sample_weak_support
    stats["edges_with_topology"] = int(edges_with_topology)
    stats["edges_with_observed_intervals"] = int(edges_with_observed_intervals)
    stats["missing_topology_top_nets"] = _counter_top_list(missing_topology_counter, net_name_lookup)
    stats["zero_support_top_nets"] = _counter_top_list(zero_support_counter, net_name_lookup)
    stats["weak_support_top_nets"] = _counter_top_list(weak_support_counter, net_name_lookup)

    topo_cost_h = torch.as_tensor(topo_cost_h_np, dtype=dtype, device=device)
    topo_cost_v = torch.as_tensor(topo_cost_v_np, dtype=dtype, device=device)
    topo_observed_mask = torch.as_tensor(topo_observed_mask_np, dtype=torch.bool, device=device)
    return topo_cost_h, topo_cost_v, topo_observed_mask, stats


def _compute_diagonal_split_topo_costs_cpp(
    edge_net_ids,
    x1,
    y1,
    x2,
    y2,
    topo_cache,
    *,
    xl,
    yl,
    route_bin_size_x,
    route_bin_size_y,
    sigma=1.0,
    min_support=1e-6,
    max_distance=0.0,
    device=None,
    dtype=torch.float32,
    profile_enabled=False,
    collect_stats=True,
):
    if same_net_topo_scoring_cpp is None:
        return None

    with profile_scope(
        profile_enabled,
        "same_net_topo_scoring.pack_cpp_lookup",
        logger=logger,
    ):
        packed = pack_same_net_topology_cache_for_cpp(
            topo_cache,
            xl=xl,
            yl=yl,
            route_bin_size_x=route_bin_size_x,
            route_bin_size_y=route_bin_size_y,
            profile_enabled=profile_enabled,
        )
    if not isinstance(packed, dict):
        return None

    try:
        with profile_scope(
            profile_enabled,
            "same_net_topo_scoring.cpp_forward",
            logger=logger,
            diag_edges=len(edge_net_ids) if edge_net_ids is not None else 0,
            nets=int(packed["net_ids"].size),
            h_segments=int(packed["h_x1"].size),
            v_segments=int(packed["v_x"].size),
        ):
            topo_cost_h_np, topo_cost_v_np, topo_observed_mask_np, stats = same_net_topo_scoring_cpp.forward(
                packed["net_ids"],
                packed["net_index_by_id"],
                packed["h_seg_offsets"],
                packed["v_seg_offsets"],
                packed["h_x1"],
                packed["h_y"],
                packed["h_x2"],
                packed["v_x"],
                packed["v_y1"],
                packed["v_y2"],
                _to_numpy_int_array(edge_net_ids),
                _to_numpy_float_array(x1),
                _to_numpy_float_array(y1),
                _to_numpy_float_array(x2),
                _to_numpy_float_array(y2),
                float(sigma),
                float(min_support),
                float(max_distance),
                bool(collect_stats),
            )
    except Exception:
        logger.exception("C++ per-net topology scoring failed; falling back to Python kernel.")
        return None

    with profile_scope(
        profile_enabled,
        "same_net_topo_scoring.to_torch",
        logger=logger,
        diag_edges=len(edge_net_ids) if edge_net_ids is not None else 0,
    ):
        topo_cost_h = torch.as_tensor(np.asarray(topo_cost_h_np, dtype=np.float32), dtype=dtype, device=device)
        topo_cost_v = torch.as_tensor(np.asarray(topo_cost_v_np, dtype=np.float32), dtype=dtype, device=device)
        topo_observed_mask = torch.as_tensor(
            np.asarray(topo_observed_mask_np, dtype=np.bool_),
            dtype=torch.bool,
            device=device,
        )
    stats = dict(stats or {})
    stats.setdefault("diag_edges", int(len(edge_net_ids) if edge_net_ids is not None else 0))
    stats.setdefault("edges_with_topology", 0)
    stats.setdefault("edges_with_observed_intervals", 0)
    stats.setdefault("edges_missing_topology", 0)
    stats.setdefault("edges_missing_topology_no_route_entry", 0)
    stats.setdefault("edges_missing_topology_route_failed", 0)
    stats.setdefault("edges_missing_topology_no_wire_entries", 0)
    stats.setdefault("edges_missing_topology_other", 0)
    stats.setdefault("edges_with_zero_observed_pieces", 0)
    stats.setdefault("edges_with_weak_support", 0)
    stats.setdefault("mean_gap", 0.0)
    stats.setdefault("tie_ratio", 0.0)
    stats.setdefault("sigma", float(sigma))
    stats.setdefault("min_support", float(min_support))
    stats.setdefault("max_distance", float(max_distance))
    stats.setdefault("sample_missing_topology", [])
    stats.setdefault("sample_zero_support", [])
    stats.setdefault("sample_weak_support", [])
    stats.setdefault("missing_topology_top_nets", [])
    stats.setdefault("zero_support_top_nets", [])
    stats.setdefault("weak_support_top_nets", [])
    stats.setdefault("zero_zero_edges", 0)
    stats.setdefault("observed_zero_zero_edges", 0)
    stats.setdefault("unobserved_zero_zero_edges", 0)
    stats.setdefault("exact_equal_edges", 0)
    stats.setdefault("hv_path_observed_edges", 0)
    stats.setdefault("vh_path_observed_edges", 0)
    stats.setdefault("both_paths_observed_edges", 0)
    stats["backend"] = "cpp"
    return topo_cost_h, topo_cost_v, topo_observed_mask, stats


def compute_diagonal_split_topo_costs(
    edge_net_ids,
    x1,
    y1,
    x2,
    y2,
    topo_cache,
    *,
    xl,
    yl,
    route_bin_size_x,
    route_bin_size_y,
    sigma=1.0,
    min_support=1e-6,
    max_distance=0.0,
    device=None,
    dtype=torch.float32,
    use_cpp=True,
    profile_enabled=False,
    collect_stats=True,
):
    if use_cpp:
        cpp_result = _compute_diagonal_split_topo_costs_cpp(
            edge_net_ids,
            x1,
            y1,
            x2,
            y2,
            topo_cache,
            xl=xl,
            yl=yl,
            route_bin_size_x=route_bin_size_x,
            route_bin_size_y=route_bin_size_y,
            sigma=sigma,
            min_support=min_support,
            max_distance=max_distance,
            device=device,
            dtype=dtype,
            profile_enabled=profile_enabled,
            collect_stats=collect_stats,
        )
        if cpp_result is not None:
            return cpp_result

    with profile_scope(
        profile_enabled,
        "same_net_topo_scoring.python_kernel",
        logger=logger,
        diag_edges=len(edge_net_ids) if edge_net_ids is not None else 0,
    ):
        topo_cost_h, topo_cost_v, topo_observed_mask, stats = _compute_diagonal_split_topo_costs_python(
            edge_net_ids,
            x1,
            y1,
            x2,
            y2,
            topo_cache,
            xl=xl,
            yl=yl,
            route_bin_size_x=route_bin_size_x,
            route_bin_size_y=route_bin_size_y,
            sigma=sigma,
            min_support=min_support,
            max_distance=max_distance,
            device=device,
            dtype=dtype,
        )
    if isinstance(stats, dict):
        stats["backend"] = "python"
    return topo_cost_h, topo_cost_v, topo_observed_mask, stats
