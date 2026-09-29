"""GPUGR evaluation and post-legalization route-driven helpers."""

import logging
import os

import numpy as np

from dreamplace.ops.routability.gpugr_context import (
    compute_route_grid_like_xplace,
    get_cached_gpugr_operator,
    sync_route_grid_to_autodmp,
    write_back_movable_lpos,
)
from dreamplace.ops.routability.post_legalization_adaptive_padding import (
    allocate_padding_sites,
    build_smoothed_overflow_map,
    compute_cell_box_overlap_stats,
    score_cells_from_overflow,
)


def run_gpugr_final_eval(params, placedb, pos):
    if not bool(getattr(params, "gpugr_final_eval_flag", 0)):
        return None

    write_back_movable_lpos(pos, params, placedb)
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
        route_xsize, route_ysize = compute_route_grid_like_xplace(params, placedb)

    out_dir = os.path.join(params.result_dir, "gpugr_final_eval")
    logging.info(
        "Run final gpugr eval after placement: route_grid=%dx%d rrr_iters=%d skip_m1_route=%s",
        route_xsize,
        route_ysize,
        int(getattr(params, "gpugr_final_eval_rrr_iters", 1)),
        str(bool(getattr(params, "gpugr_final_eval_skip_m1_route", 1))),
    )
    result = get_cached_gpugr_operator(params, placedb).run_gpugr(
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


def run_post_legalization_adaptive_padding(
    params,
    placedb,
    pos,
    model,
):
    if not bool(getattr(params, "post_legalization_adaptive_padding_flag", 0)):
        return pos, None
    if not bool(getattr(params, "legalize_flag", 0)):
        raise ValueError("post_legalization_adaptive_padding_flag requires legalize_flag=1")
    if len(placedb.regions) > 0:
        raise ValueError("post-legalization adaptive padding does not support fence regions")

    write_back_movable_lpos(pos, params, placedb)
    route_xsize, route_ysize = compute_route_grid_like_xplace(params, placedb)
    rrr_iters = int(getattr(params, "post_legalization_padding_rrr_iters", 0))
    out_dir = os.path.join(params.result_dir, "gpugr_post_legalization_padding")
    logging.info(
        "Run gpugr for post-legalization adaptive padding: route_grid=%dx%d rrr_iters=%d",
        route_xsize,
        route_ysize,
        rrr_iters,
    )
    result = get_cached_gpugr_operator(params, placedb).run_gpugr(
        out_dir=out_dir,
        design_name=params.design_name(),
        gpu=getattr(params, "gpu_id", 0),
        threads=params.num_threads,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        rrr_iters=rrr_iters,
        skip_m1_route=bool(getattr(params, "post_legalization_padding_skip_m1_route", 1)),
        verbose_parser_log=False,
        cpp_log_level=int(getattr(params, "gpugr_final_eval_cpp_log_level", 2)),
        keep_temp_def=False,
        save_artifacts=bool(getattr(params, "post_legalization_padding_save_artifacts", 0)),
        include_route_entries=False,
        backend=getattr(params, "gpugr_backend", "auto"),
    )
    maps = result["maps"]
    metrics = result["metrics"]
    overflow_xy = build_smoothed_overflow_map(
        maps["cg_map_h_overflow"],
        maps["cg_map_v_overflow"],
        smooth_kernel=int(getattr(params, "post_legalization_padding_smooth_kernel", 3)),
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
    movable_macro_mask = getattr(model.data_collections, "movable_macro_mask", None)
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
            getattr(placedb, "num_physical_nodes", placedb.num_nodes - placedb.num_filler_nodes)
        ),
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        site_width=placedb.site_width,
        row_height=placedb.row_height,
        hot_cell_ratio=float(getattr(params, "post_legalization_padding_hot_cell_ratio", 0.2)),
        row_free_ratio=float(getattr(params, "post_legalization_padding_row_free_ratio", 0.5)),
        max_padding_sites=int(getattr(params, "post_legalization_padding_max_sites", 1)),
        eligible_mask=eligible_mask,
    )
    logging.info(
        "Post-legalization adaptive padding plan: eligible=%d positive_score=%d requested_hot=%d allocated=%d "
        "added_sites=%d free_sites=%d budget_sites=%d max_score=%.6g min_allocated_score=%.6g "
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
        max_retries=int(getattr(params, "post_legalization_padding_max_retries", 4)),
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
        "Post-legalization adaptive padding result: requested=%d used=%d attempts=%d rollback=%d "
        "padded_legal=%d physical_legal=%d greedy_fallback=%d moved=%d displacement_total=%.6g "
        "displacement_max=%.6g HPWL=%.6g->%.6g delta=%.6g M2_overlap_count=%d->%d M2_overlap_area=%.6g->%.6g",
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


def run_gpugr_before_first_area_adjust_and_exit(
    params,
    placedb,
    pos,
    num_area_adjust,
    model=None,
):
    if not getattr(params, "gpugr_first_inflation_exit", False) or num_area_adjust != 0:
        return
    logging.info(
        "Run gpugr operator before the first AutoDMP area-adjust round and exit after it finishes."
    )
    write_back_movable_lpos(pos, params, placedb)
    route_xsize, route_ysize = sync_route_grid_to_autodmp(params, placedb, model=model)
    out_dir = os.path.join(params.result_dir, "gpugr_first_inflation")
    result = get_cached_gpugr_operator(params, placedb).run_gpugr(
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
        "gpugr finished before first area-adjust. #OvflNets=%d GR_WL=%.0f GR_Vias=%.0f EstShorts=%.0f",
        metrics["num_overflow_nets"],
        metrics["gr_wirelength"],
        metrics["gr_num_vias"],
        metrics["gr_est_shorts"],
    )
    raise SystemExit(0)


__all__ = [
    "run_gpugr_final_eval",
    "run_post_legalization_adaptive_padding",
    "run_gpugr_before_first_area_adjust_and_exit",
]
