"""Prepare L-shape objective inputs from GPUGR or iRT EGR."""

import logging
import os

import torch

from dreamplace.ops.routability.egr_resample import (
    create_directional_supply_and_demand_maps_from_egr,
    create_supply_and_demand_maps_from_egr,
)
from dreamplace.ops.routability.gpugr_context import (
    build_parser_cache_inputs,
    build_topology_flat_net2pin_inputs,
    build_topology_net_name_to_id,
    build_topology_pack_geometry,
    build_topology_pin_name_to_id,
    get_cached_gpugr_operator,
    sync_route_grid_to_autodmp,
    write_back_movable_lpos,
)
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose, profile_scope
from dreamplace.ops.routability.route_map_utils import (
    build_l_shape_supply_original_maps_from_placedb,
    resolve_l_shape_wire_width,
    resample_xy_map,
    split_gpugr_maps_by_direction,
)
from dreamplace.ops.routability.same_net_topo_scoring import build_same_net_topology_cache
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import use_ggr_l_shape_topology


def should_skip_resolver_l_direction_for_soft(params):
    return bool(getattr(params, "soft_l_assignment", False)) and not bool(
        getattr(params, "soft_l_use_resolver_prior", True)
    )


def prepare_l_shape_inputs_from_gpugr(params, placedb, pos, model=None):
    with profile_scope(params, "gpugr_prepare.sync_route_grid", tensor=pos):
        route_xsize, route_ysize = sync_route_grid_to_autodmp(
            params,
            placedb,
            model=model,
        )
    l_shape_num_bins_x = int(getattr(params, "num_bins_x", placedb.num_bins_x))
    l_shape_num_bins_y = int(getattr(params, "num_bins_y", placedb.num_bins_y))
    topology_xl, topology_yl, topology_bin_size_x, topology_bin_size_y = build_topology_pack_geometry(
        placedb,
        route_xsize,
        route_ysize,
    )
    need_l_shape_topology_pack = use_ggr_l_shape_topology(params)
    need_same_net_topology_pack = bool(getattr(params, "soft_l_assignment", False))
    need_route_entries = (
        bool(getattr(params, "l_direction_use_gpugr", False))
        and not should_skip_resolver_l_direction_for_soft(params)
        and not need_l_shape_topology_pack
    )
    need_route_entries = need_route_entries or bool(
        getattr(params, "gpugr_l_direction_save_artifacts", 0)
    )
    need_route_entries = need_route_entries or bool(getattr(params, "l_shape_plot_flag", 0))

    parser_cache_enable = bool(getattr(params, "gpugr_parser_cache_enable", True))
    parser_cache_node_lpos = None
    parser_cache_node_names = None
    gpugr_op = get_cached_gpugr_operator(params, placedb)
    if parser_cache_enable:
        with profile_scope(params, "gpugr_prepare.parser_cache_inputs", tensor=pos):
            parser_cache_node_lpos, parser_cache_node_names = build_parser_cache_inputs(
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
            write_back_movable_lpos(pos, params, placedb)

    parser_cache_fallback_before_export = None
    if parser_cache_hit_for_writeback:

        def parser_cache_fallback_before_export():
            with profile_scope(
                params,
                "gpugr_prepare.parser_cache_fallback_write_back_pos",
                tensor=pos,
            ):
                write_back_movable_lpos(pos, params, placedb)

    with profile_scope(params, "gpugr_prepare.topology_net_name_to_id", tensor=pos):
        topology_net_name_to_id = build_topology_net_name_to_id(placedb)
    with profile_scope(params, "gpugr_prepare.topology_pin_name_to_id", tensor=pos):
        topology_pin_name_to_id = build_topology_pin_name_to_id(placedb)
    with profile_scope(params, "gpugr_prepare.topology_flat_net2pin_inputs", tensor=pos):
        topology_flat_net2pin_map, topology_flat_net2pin_start_map = (
            build_topology_flat_net2pin_inputs(placedb)
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
            topology_ignored_net_names=getattr(placedb, "clock_net_names", ()),
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
    _, _, raw_wire_h_xy, raw_wire_v_xy = split_gpugr_maps_by_direction(
        placedb, capacity_map, raw_wire_layer_map
    )
    supply_h_xy, supply_v_xy, demand_h_xy, demand_v_xy = split_gpugr_maps_by_direction(
        placedb, capacity_map, total_demand_map
    )
    _, _, fix_usage_h_xy, fix_usage_v_xy = split_gpugr_maps_by_direction(
        placedb, capacity_map, fix_usage_layer_map
    )
    _, _, mov_usage_h_xy, mov_usage_v_xy = split_gpugr_maps_by_direction(
        placedb, capacity_map, mov_usage_layer_map
    )
    with profile_scope(
        params,
        "gpugr_prepare.split_resample_maps",
        tensor=pos,
        l_shape_bins=f"{l_shape_num_bins_x}x{l_shape_num_bins_y}",
    ):
        maps_by_name = {
            "supply_map": supply_xy,
            "raw_wire_demand_map": raw_wire_xy,
            "fix_usage_map": fix_usage_xy,
            "mov_usage_map": mov_usage_xy,
            "demand_map": demand_xy,
            "supply_map_h": supply_h_xy,
            "supply_map_v": supply_v_xy,
            "raw_wire_demand_map_h": raw_wire_h_xy,
            "raw_wire_demand_map_v": raw_wire_v_xy,
            "fix_usage_map_h": fix_usage_h_xy,
            "fix_usage_map_v": fix_usage_v_xy,
            "mov_usage_map_h": mov_usage_h_xy,
            "mov_usage_map_v": mov_usage_v_xy,
            "demand_map_h": demand_h_xy,
            "demand_map_v": demand_v_xy,
        }
        maps_by_name = {
            name: resample_xy_map(value, l_shape_num_bins_x, l_shape_num_bins_y)
            for name, value in maps_by_name.items()
        }
    supply_original_maps = {
        "supply_original": maps_by_name["supply_map"].clone(),
        "supply_original_h": maps_by_name["supply_map_h"].clone(),
        "supply_original_v": maps_by_name["supply_map_v"].clone(),
    }
    wire_width = resolve_l_shape_wire_width(
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
            maps_by_name["supply_map"].min().item(),
            maps_by_name["supply_map"].max().item(),
            maps_by_name["supply_map"].mean().item(),
            maps_by_name["demand_map"].min().item(),
            maps_by_name["demand_map"].max().item(),
            maps_by_name["demand_map"].mean().item(),
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
            float((supply_original_maps["supply_original"] - maps_by_name["supply_map"]).abs().sum().item()),
            float((supply_original_maps["supply_original_h"] - maps_by_name["supply_map_h"]).abs().sum().item()),
            float((supply_original_maps["supply_original_v"] - maps_by_name["supply_map_v"]).abs().sum().item()),
        )
    return {
        **maps_by_name,
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


def prepare_l_shape_inputs_from_egr(params, placedb, pos, model):
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
        layer="planar",
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
    wire_width = resolve_l_shape_wire_width(placedb, fallback_wire_width=wire_width)
    supply_original_maps = build_l_shape_supply_original_maps_from_placedb(
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


def get_default_egr_guide_path(params):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(params.result_dir))),
        "iEDA/data/rt/rt_temp_directory/early_router/route_planar.guide",
    )


def resolve_l_directions_for_l_shape(
    params,
    placedb,
    pos,
    steiner_topo_op,
    gpugr_route_entries=None,
    gpugr_metrics=None,
    gpugr_route_grid=None,
    prepare_gpugr_inputs=None,
):
    if steiner_topo_op.l_direction_resolver is None:
        steiner_topo_op.init_l_direction_resolver(placedb, params)

    if getattr(params, "l_direction_use_gpugr", False):
        if gpugr_route_entries is None:
            if prepare_gpugr_inputs is None:
                prepare_gpugr_inputs = prepare_l_shape_inputs_from_gpugr
            gpugr_inputs = prepare_gpugr_inputs(params, placedb, pos)
            route_entries = gpugr_inputs["route_entries"]
            metrics = gpugr_inputs["metrics"]
            route_xsize = gpugr_inputs["route_xsize"]
            route_ysize = gpugr_inputs["route_ysize"]
        else:
            route_entries = gpugr_route_entries
            metrics = gpugr_metrics or {"num_overflow_nets": -1, "gr_est_shorts": -1.0}
            if gpugr_route_grid is None:
                from dreamplace.ops.routability.gpugr_context import compute_route_grid_like_xplace

                route_xsize, route_ysize = compute_route_grid_like_xplace(params, placedb)
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

    egr_guide_path = getattr(params, "egr_guide_path", get_default_egr_guide_path(params))
    logging.info("Resolve L directions from EGR guide: %s", egr_guide_path)
    with profile_scope(params, "l_direction.resolve_from_egr", tensor=pos):
        return steiner_topo_op.resolve_l_directions_from_egr(egr_guide_path)


def load_l_shape_topology_pack_from_gpugr(
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
        raise RuntimeError("l_shape_use_ggr_topology requires GGR l_shape_topology_pack output")
    with profile_scope(params, profile_name, tensor=pos, iteration=iteration):
        with torch.no_grad():
            pin_pos = pin_pos_op(pos)
            if pin_pos.is_cuda:
                pin_pos = pin_pos.cpu()
            (
                data_collections.net_flat_topo_sort,
                data_collections.net_flat_topo_sort_start,
                data_collections.pin_fa,
                data_collections.flat_pin_to,
                data_collections.flat_pin_to_start,
                data_collections.flat_pin_from,
            ) = steiner_topo_op.load_ggr_topology_pack(topology_pack, pin_pos)
    if steiner_topo_op.edge_l_directions is None:
        raise RuntimeError("l_shape_use_ggr_topology requires edge_l_directions in GGR topology pack")
    return steiner_topo_op.edge_l_directions


__all__ = [
    "prepare_l_shape_inputs_from_gpugr",
    "prepare_l_shape_inputs_from_egr",
    "should_skip_resolver_l_direction_for_soft",
    "get_default_egr_guide_path",
    "resolve_l_directions_for_l_shape",
    "load_l_shape_topology_pack_from_gpugr",
]
