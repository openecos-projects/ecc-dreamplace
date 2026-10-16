"""Incremental GGR-directed padding for legalization-only flows."""

import logging
import os

import numpy as np

from dreamplace.ops.routability import adaptive_padding_cpp
from dreamplace.ops.routability.gpugr_context import (
    compute_route_grid_like_xplace,
    get_cached_gpugr_operator,
    write_back_movable_lpos,
)


def _cpu_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def plan_iterative_padding(params, placedb, current, data, horizontal, vertical,
                           route_grid_edges_dbu, cumulative):
    physical_pos = np.ascontiguousarray(current.detach().cpu().numpy(), dtype=np.float64)
    logical_pos = physical_pos.copy()
    logical_pos[:placedb.num_movable_nodes] -= cumulative * placedb.site_width
    physical_size_x = np.ascontiguousarray(
        data.node_size_x.detach().cpu().numpy(), dtype=np.float64
    )
    logical_size_x = physical_size_x.copy()
    logical_size_x[:placedb.num_movable_nodes] += 2 * cumulative * placedb.site_width
    eligible = np.ones(placedb.num_movable_nodes, dtype=bool)
    movable_macro_mask = getattr(data, "movable_macro_mask", None)
    if movable_macro_mask is not None:
        eligible &= ~movable_macro_mask.detach().cpu().numpy().astype(bool)
    grid_x = (np.asarray(route_grid_edges_dbu[0], dtype=np.float64)
              - params.shift_factor[0]) * params.scale_factor
    grid_y = (np.asarray(route_grid_edges_dbu[1], dtype=np.float64)
              - params.shift_factor[1]) * params.scale_factor
    return adaptive_padding_cpp.plan_padding(
        np.ascontiguousarray(_cpu_numpy(horizontal), dtype=np.float64),
        np.ascontiguousarray(_cpu_numpy(vertical), dtype=np.float64),
        np.ascontiguousarray(grid_x), np.ascontiguousarray(grid_y),
        physical_pos, logical_pos, physical_size_x, logical_size_x,
        np.ascontiguousarray(data.node_size_y.detach().cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(placedb.rows, dtype=np.float64).reshape(-1, 4),
        eligible, placedb.num_nodes, placedb.num_movable_nodes,
        placedb.num_physical_nodes, placedb.xl, placedb.yl, placedb.xh,
        placedb.yh, placedb.site_width, placedb.row_height,
        float(params.post_legalization_padding_overflow_bin_ratio),
        float(params.post_legalization_padding_hot_cell_ratio),
    )


def run_iterative_legalization_padding(params, placedb, pos, model):
    rounds = int(params.post_legalization_padding_rounds)
    if rounds <= 0:
        return pos, {"completed_rounds": 0, "effective_cells_by_round": []}
    if len(placedb.regions):
        raise ValueError("iterative legalization padding does not support fence regions")
    if placedb.rows is None or len(placedb.rows) == 0:
        raise ValueError("iterative legalization padding requires placement rows")

    data = model.data_collections
    movable = placedb.num_movable_nodes
    cumulative = np.zeros(movable, dtype=np.int32)
    current = pos.detach().clone()
    completed = 0
    effective_counts = []

    def evaluate(pos_to_route, stage):
        write_back_movable_lpos(pos_to_route, params, placedb)
        route_x, route_y = compute_route_grid_like_xplace(params, placedb)
        return get_cached_gpugr_operator(params, placedb).run_gpugr(
            out_dir=os.path.join(params.result_dir, "gpugr_legalization_padding", stage),
            design_name=params.design_name(),
            gpu=getattr(params, "gpu_id", 0),
            threads=params.num_threads,
            route_xsize=route_x,
            route_ysize=route_y,
            rrr_iters=int(getattr(params, "post_legalization_padding_rrr_iters", 0)),
            skip_m1_route=bool(getattr(params, "post_legalization_padding_skip_m1_route", 1)),
            save_artifacts=bool(getattr(params, "post_legalization_padding_save_artifacts", 0)),
            backend=getattr(params, "gpugr_backend", "auto"),
            bottom_routing_layer=getattr(params, "gpugr_bottom_routing_layer", ""),
            top_routing_layer=getattr(params, "gpugr_top_routing_layer", ""),
        )

    result = evaluate(current, "initial")
    for round_id in range(1, rounds + 1):
        plan = plan_iterative_padding(
            params, placedb, current, data,
            result["maps"]["cg_map_h_overflow"],
            result["maps"]["cg_map_v_overflow"],
            result["route_grid_edges_dbu"], cumulative,
        )
        if not plan["hot_bins_count"]:
            effective_counts.append(0)
            logging.info(
                "Legalization padding round %d/%d: no overflow bins; "
                "requested_cells=0 effective_cells=0 cumulative_padded_cells=%d",
                round_id, rounds, int(np.count_nonzero(cumulative)),
            )
            break
        if plan["allocated_count"] == 0:
            effective_counts.append(0)
            logging.info(
                "Legalization padding round %d/%d: no free row capacity; "
                "requested_cells=0 effective_cells=0 cumulative_padded_cells=%d",
                round_id, rounds, int(np.count_nonzero(cumulative)),
            )
            break
        proposed = cumulative + plan["padding_sites"]
        next_pos, stats = model.run_adaptive_padding_legalization(
            placedb=placedb, pos=current, padding_sites=proposed,
            scores=plan["scores"], max_retries=0, prefer_direct_abacus=True,
        )
        candidate_result = None
        accepted = False
        if not stats["rollback"]:
            candidate_result = evaluate(next_pos, str(round_id))
            before_short = float(result["metrics"]["gr_est_shorts"])
            after_short = float(candidate_result["metrics"]["gr_est_shorts"])
            accepted = np.isfinite(after_short) and after_short < before_short
        effective = int(np.count_nonzero(proposed > cumulative)) if accepted else 0
        effective_counts.append(effective)
        logging.info(
            "Legalization padding round %d/%d: overflow_bins=%d candidates=%d "
            "requested_cells=%d effective_cells=%d cumulative_padded_cells=%d "
            "cumulative_sites=%d free_sites=%d legal=%d accepted=%d "
            "overflow_nets=%d est_shorts=%.6g candidate_est_shorts=%.6g "
            "candidate_moved_cells=%d candidate_max_displacement=%.6g",
            round_id, rounds, plan["hot_bins_count"],
            plan["positive_score_count"], plan["allocated_count"], effective,
            int(np.count_nonzero(proposed if accepted else cumulative)),
            int((proposed if accepted else cumulative).sum()),
            plan["total_free_sites"], int(not stats["rollback"]), int(accepted),
            result["metrics"]["num_overflow_nets"],
            result["metrics"]["gr_est_shorts"],
            candidate_result["metrics"]["gr_est_shorts"] if candidate_result else float("nan"),
            stats.get("moved_count", 0), stats.get("max_displacement", 0.0),
        )
        if not accepted:
            write_back_movable_lpos(current, params, placedb)
            break
        current, cumulative = next_pos, proposed
        result = candidate_result
        completed += 1

    return current, {
        "completed_rounds": completed,
        "cumulative_padding_sites": int(cumulative.sum()),
        "effective_cells_by_round": effective_counts,
    }
