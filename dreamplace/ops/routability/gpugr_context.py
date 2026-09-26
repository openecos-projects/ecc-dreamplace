"""Small GPUGR context helpers shared by routability flow adapters."""

import logging

import numpy as np

from dreamplace.ops.routability.route_map_utils import normalize_placedb_name


def compute_route_grid_like_xplace(params, placedb):
    """Resolve the GPUGR grid using the Xplace-compatible rule."""
    configured_route_x = int(getattr(params, "route_num_bins_x", 0) or 0)
    configured_route_y = int(getattr(params, "route_num_bins_y", 0) or 0)
    auto_adjust_bins = bool(getattr(params, "auto_adjust_bins", 0))
    if not auto_adjust_bins and configured_route_x > 0 and configured_route_y > 0:
        logging.info(
            "Use explicit gpugr route grid: route_num_bins=%dx%d (auto_adjust_bins=0)",
            configured_route_x,
            configured_route_y,
        )
        return configured_route_x, configured_route_y

    place_num_bins_y = int(
        getattr(placedb, "num_bins_y", getattr(params, "num_bins_y", 512))
    )
    if place_num_bins_y <= 0:
        place_num_bins_y = 512

    die_w = float(placedb.xh - placedb.xl)
    die_h = float(placedb.yh - placedb.yl)
    if die_w <= 0 or die_h <= 0:
        logging.warning(
            "Invalid die size for gpugr grid computation (die_w=%g, die_h=%g). "
            "Fallback to square %dx%d grid.",
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
        "Compute gpugr grid with Xplace rule: place_bins_y=%d die_ratio=%.4f "
        "-> route_xsize=%d route_ysize=%d",
        place_num_bins_y,
        die_ratio,
        route_xsize,
        route_ysize,
    )
    return route_xsize, route_ysize


def sync_route_grid_to_autodmp(params, placedb, model=None):
    """Apply the selected GPUGR grid and rebuild grid-dependent operators."""
    route_xsize, route_ysize = compute_route_grid_like_xplace(params, placedb)
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

    if grid_changed and model is not None:
        model.refresh_routability_operators()
        logging.info(
            "Rebuilt pin_utilization_map_op and adjust_node_area_op for the gpugr routing grid."
        )
    elif not grid_changed:
        logging.info("AutoDMP routing grid already matches the gpugr grid. Skip rebuilding related ops.")

    return route_xsize, route_ysize


def extract_movable_lpos(pos, params, placedb):
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


def write_back_movable_lpos(pos, params, placedb):
    node_x, node_y = extract_movable_lpos(pos, params, placedb)
    placedb.write_placement_back(node_x, node_y)


def build_parser_cache_inputs(pos, params, placedb):
    node_x, node_y = extract_movable_lpos(pos, params, placedb)

    def std_round(values):
        values = np.asarray(values)
        return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))

    node_lpos = np.stack([std_round(node_x), std_round(node_y)], axis=1).astype(
        np.float32, copy=False
    )
    return node_lpos, get_parser_cache_node_names(placedb)


def get_parser_cache_node_names(placedb):
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
        normalize_placedb_name(name)
        for name in placedb_node_names[: placedb.num_movable_nodes]
    )
    setattr(
        placedb,
        "_gpugr_parser_cache_node_names_cache",
        {"key": cache_key, "names": node_names},
    )
    return node_names


def get_cached_gpugr_operator(params, placedb):
    from dreamplace.ops.gpugr.backend_select import create_gpugr_backend

    gpugr_op = getattr(placedb, "_autodmp_gpugr_op", None)
    if gpugr_op is None:
        gpugr_op = create_gpugr_backend(params, placedb)
        setattr(placedb, "_autodmp_gpugr_op", gpugr_op)
    return gpugr_op


def build_topology_net_name_to_id(placedb):
    net_names = getattr(placedb, "net_names", [])
    native_name_map = getattr(placedb, "net_name2id_map", None)
    cache_key = (
        id(net_names),
        int(len(net_names)),
        id(native_name_map),
        int(len(native_name_map)) if hasattr(native_name_map, "__len__") else -1,
    )
    cache = getattr(placedb, "_gpugr_topology_net_name_to_id_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["mapping"]
    mapping = {
        normalize_placedb_name(net_name): int(net_id)
        for net_id, net_name in enumerate(net_names)
    }
    if isinstance(native_name_map, dict):
        for raw_name, raw_net_id in native_name_map.items():
            name = normalize_placedb_name(raw_name)
            net_id = int(raw_net_id)
            previous = mapping.get(name)
            if previous is not None and previous != net_id:
                raise RuntimeError(
                    "ECC net identity conflict for %r: net_names=%d, "
                    "net_name2id_map=%d" % (name, previous, net_id)
                )
            mapping[name] = net_id
    setattr(
        placedb,
        "_gpugr_topology_net_name_to_id_cache",
        {"key": cache_key, "mapping": mapping},
    )
    return mapping


def build_topology_pin_name_to_id(placedb):
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
    setattr(placedb, "_gpugr_topology_pin_name_to_id_cache", (cache_key, mapping))
    return mapping


def _as_cached_int64_array(placedb, attr_name, cache_name):
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
    setattr(placedb, cache_name, {"key": cache_key, "array": int64_array})
    return int64_array


def build_topology_flat_net2pin_inputs(placedb):
    return (
        _as_cached_int64_array(
            placedb,
            "flat_net2pin_map",
            "_gpugr_topology_flat_net2pin_map_int64_cache",
        ),
        _as_cached_int64_array(
            placedb,
            "flat_net2pin_start_map",
            "_gpugr_topology_flat_net2pin_start_map_int64_cache",
        ),
    )


def build_topology_pack_geometry(placedb, route_xsize, route_ysize):
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


__all__ = [
    "compute_route_grid_like_xplace",
    "sync_route_grid_to_autodmp",
    "extract_movable_lpos",
    "write_back_movable_lpos",
    "build_parser_cache_inputs",
    "get_parser_cache_node_names",
    "get_cached_gpugr_operator",
    "build_topology_net_name_to_id",
    "build_topology_pin_name_to_id",
    "build_topology_flat_net2pin_inputs",
    "build_topology_pack_geometry",
]
