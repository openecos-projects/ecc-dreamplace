import logging
import time

import numpy as np
import torch

import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import dreamplace.ops.legality_check.legality_check as legality_check
import dreamplace.ops.m2_pa_refine.m2_pa_refine as m2_pa_refine
import dreamplace.ops.m2_soft_legalize.m2_soft_legalize as m2_soft_legalize
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
from dreamplace.m2_pg_rail_hybrid_legalization import (
    M2PgRailHybridLegalizationView,
)


def _resolve_boolean_like_flag(params, name, default=0):
    value = getattr(params, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    elif isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(
        "%s must be a boolean-like value "
        "(0/1, true/false, yes/no, on/off)" % name
    )


def _resolve_m2_pg_rail_legalization_mode(params):
    value = getattr(params, "m2_pg_rail_legalization_mode", "soft")
    if not isinstance(value, str):
        raise ValueError(
            "m2_pg_rail_legalization_mode must be 'soft', 'hybrid_hard', "
            "or 'subset_hard'"
        )
    value = value.strip().lower()
    if value not in ("soft", "hybrid_hard", "subset_hard"):
        raise ValueError(
            "m2_pg_rail_legalization_mode must be 'soft', 'hybrid_hard', "
            "or 'subset_hard'"
        )
    return value


def build_m2_soft_legalization(self, params, placedb, data_collections):
    enabled = _resolve_boolean_like_flag(
        params, "m2_pg_rail_legalization_blockage_flag"
    )
    if not enabled or _resolve_m2_pg_rail_legalization_mode(params) != "soft":
        return None
    if _resolve_boolean_like_flag(
        params, "ieda_m2_pg_rail_blockage_flag"
    ):
        logging.info(
            "M2 soft legalization skipped because global hard M2 "
            "blockages are already enabled"
        )
        return None
    if len(placedb.regions) > 0:
        raise ValueError(
            "m2_pg_rail_legalization_blockage_flag does not support "
            "fence regions"
        )

    rail_boxes = data_collections.m2_pg_rail_density_boxes
    if rail_boxes.numel() == 0:
        logging.info(
            "M2 soft legalization enabled with no rail boxes; treating "
            "the stage as a no-op"
        )
        return None

    displacement_weight = float(
        getattr(
            params,
            "m2_pg_rail_legalization_displacement_weight",
            0.01,
        )
    )
    soft_legalize = m2_soft_legalize.M2SoftLegalize(
        node_size_x=data_collections.node_size_x,
        node_size_y=data_collections.node_size_y,
        rail_boxes=rail_boxes,
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        site_width=placedb.site_width,
        row_height=placedb.row_height,
        num_movable_nodes=placedb.num_movable_nodes,
        num_terminals=placedb.num_terminals,
        displacement_weight=displacement_weight,
    )

    movable_area = float(
        getattr(
            placedb,
            "total_movable_node_area",
            (
                data_collections.node_size_x[:placedb.num_movable_nodes]
                * data_collections.node_size_y[:placedb.num_movable_nodes]
            ).sum().item(),
        )
    )
    placeable_area = float(
        getattr(
            placedb,
            "total_space_area",
            (placedb.xh - placedb.xl) * (placedb.yh - placedb.yl),
        )
    )
    clipped_xl = rail_boxes[:, 0].clamp(
        min=float(placedb.xl), max=float(placedb.xh)
    )
    clipped_yl = rail_boxes[:, 1].clamp(
        min=float(placedb.yl), max=float(placedb.yh)
    )
    clipped_xh = rail_boxes[:, 2].clamp(
        min=float(placedb.xl), max=float(placedb.xh)
    )
    clipped_yh = rail_boxes[:, 3].clamp(
        min=float(placedb.yl), max=float(placedb.yh)
    )
    rail_area = float(
        (
            (clipped_xh - clipped_xl).clamp(min=0)
            * (clipped_yh - clipped_yl).clamp(min=0)
        ).sum().item()
    )
    available_area = placeable_area - rail_area
    effective_utilization = (
        movable_area / available_area
        if available_area > 0
        else float("inf")
    )
    unavoidable_overlap_lower_bound = max(
        0.0, movable_area - available_area
    )
    logging.info(
        "M2 soft legalization capacity telemetry: requested=1 "
        "rail_boxes=%d movable_area=%.6g placeable_area=%.6g "
        "rail_area=%.6g effective_utilization=%.6g "
        "overlap_lower_bound=%.6g advisory_only=1 "
        "displacement_weight=%.6g",
        int(rail_boxes.size(0)),
        movable_area,
        placeable_area,
        rail_area,
        effective_utilization,
        unavoidable_overlap_lower_bound,
        displacement_weight,
    )

    def build_m2_soft_legalization_op(pos):
        hpwl_before = float(self.op_collections.hpwl_op(pos).item())
        result, stats = soft_legalize(pos)
        legal = bool(self.op_collections.legality_check_op(result))
        rollback = not legal
        if rollback:
            result = pos.detach().clone()
        hpwl_after = float(self.op_collections.hpwl_op(result).item())
        stats.update(
            {
                "rail_count": int(rail_boxes.size(0)),
                "legal": legal,
                "rollback": rollback,
                "hpwl_before": hpwl_before,
                "hpwl_after": hpwl_after,
                "hpwl_delta": hpwl_after - hpwl_before,
            }
        )
        logging.info(
            "M2 soft legalization completed: requested=1 used=%d "
            "rail_boxes=%d overlap_count=%.0f->%.0f "
            "overlap_area=%.6g->%.6g moved=%d move_events=%.0f "
            "displacement_total=%.6g displacement_max=%.6g "
            "accepted=%.0f rejected=%.0f skipped_illegal_rows=%.0f "
            "HPWL=%.6g->%.6g delta=%.6g cell_legal=%d rollback=%d",
            int(not rollback),
            stats["rail_count"],
            stats["overlap_count_before"],
            stats["overlap_count_after"],
            stats["overlap_area_before"],
            stats["overlap_area_after"],
            int(stats["moved_count"]),
            stats["move_events"],
            stats["total_displacement"],
            stats["max_displacement"],
            stats["accepted_candidates"],
            stats["rejected_candidates"],
            stats["skipped_illegal_rows"],
            hpwl_before,
            hpwl_after,
            stats["hpwl_delta"],
            int(legal),
            int(rollback),
        )
        if rollback:
            logging.error(
                "M2 soft legalization violated ordinary legality; "
                "rolling back to the legal input placement"
            )
        return result, stats

    return build_m2_soft_legalization_op

def build_m2_pg_rail_hybrid_legalization_view(
    self, params, placedb, data_collections
):
    enabled = _resolve_boolean_like_flag(
        params, "m2_pg_rail_legalization_blockage_flag"
    )
    mode = _resolve_m2_pg_rail_legalization_mode(params)
    if (
        not enabled
        or mode not in ("hybrid_hard", "subset_hard")
    ):
        return None
    if _resolve_boolean_like_flag(
        params, "ieda_m2_pg_rail_blockage_flag"
    ):
        logging.info(
            "M2 %s legalization skipped because global hard M2 "
            "blockages are already enabled",
            mode.replace("_", "-"),
        )
        return None
    if len(placedb.regions) > 0:
        raise ValueError(
            "%s M2 legalization does not support fence regions" % mode
        )
    if (
        mode == "hybrid_hard"
        and _resolve_boolean_like_flag(params, "detailed_place_flag")
    ):
        raise ValueError(
            "%s M2 legalization requires detailed_place_flag=0" % mode
        )

    hard_rail_start = None
    hard_rail_end = None
    if mode == "subset_hard":
        hard_rail_start = getattr(
            params, "m2_pg_rail_legalization_hard_rail_start", 1
        )
        hard_rail_end = getattr(
            params, "m2_pg_rail_legalization_hard_rail_end", 0
        )

    view = M2PgRailHybridLegalizationView.create(
        placedb,
        data_collections,
        hard_rail_start=hard_rail_start,
        hard_rail_end=hard_rail_end,
        legalization_mode=mode,
    )
    if view is None:
        logging.info(
            "M2 %s legalization enabled with no rail boxes; "
            "treating the mode as an ordinary-legalization no-op",
            mode,
        )
    return view

def build_legalization(
    self,
    params,
    placedb,
    data_collections,
    device,
):
    """
    @brief legalization
    @param params parameters
    @param placedb placement database
    @param data_collections a collection of all data and variables required for constructing the ops
    @param device cpu or cuda
    """
    def build_legalizer_ops(
        node_size_x,
        node_size_y,
        node_weights,
        node2fence_region_map,
        num_movable_nodes,
        num_terminal_NIs,
        num_filler_nodes,
        check_op,
    ):
        # The number of bins controls the search granularity.
        macro_op = macro_legalize.MacroLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=node_weights,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=num_movable_nodes,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_filler_nodes,
        )
        greedy_op = greedy_legalize.GreedyLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=node_weights,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=num_movable_nodes,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_filler_nodes,
        )
        abacus_op = abacus_legalize.AbacusLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=node_weights,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=num_movable_nodes,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_filler_nodes,
        )
        return macro_op, greedy_op, abacus_op, check_op

    ordinary_ops = build_legalizer_ops(
        node_size_x=data_collections.node_size_x,
        node_size_y=data_collections.node_size_y,
        node_weights=data_collections.num_pins_in_nodes,
        node2fence_region_map=data_collections.node2fence_region_map,
        num_movable_nodes=placedb.num_movable_nodes,
        num_terminal_NIs=placedb.num_terminal_NIs,
        num_filler_nodes=placedb.num_filler_nodes,
        check_op=self.op_collections.legality_check_op,
    )

    def run_legalizer(pos, ops, label):
        macro_op, greedy_op, abacus_op, check_op = ops
        pos1 = macro_op(pos, pos)
        pos2 = greedy_op(pos1, pos1)
        if not check_op(pos2):
            logging.error(
                "%s legality check failed in greedy legalization",
                label,
            )
            return pos2, False

        pos3 = abacus_op(pos1, pos2)
        if not check_op(pos3):
            logging.error(
                "%s legality check failed in abacus legalization; "
                "using legal greedy result",
                label,
            )
            return pos2, True
        return pos3, True

    def build_legalization_op(pos):
        logging.info("Start legalization")
        hybrid_view = getattr(
            self, "m2_pg_rail_hybrid_legalization_view", None
        )
        if hybrid_view is not None:
            problem = hybrid_view.prepare(pos, data_collections)
            hybrid_check_op = legality_check.LegalityCheck(
                node_size_x=problem.node_size_x,
                node_size_y=problem.node_size_y,
                flat_region_boxes=data_collections.flat_region_boxes,
                flat_region_boxes_start=data_collections.flat_region_boxes_start,
                node2fence_region_map=problem.node2fence_region_map,
                fp_info=data_collections.fp_info,
                num_terminals=problem.num_terminals,
                num_movable_nodes=problem.num_movable_nodes,
            )
            hybrid_ops = build_legalizer_ops(
                node_size_x=problem.node_size_x,
                node_size_y=problem.node_size_y,
                node_weights=problem.node_weights,
                node2fence_region_map=problem.node2fence_region_map,
                num_movable_nodes=problem.num_movable_nodes,
                num_terminal_NIs=problem.num_terminal_NIs,
                num_filler_nodes=problem.num_filler_nodes,
                check_op=hybrid_check_op,
            )
            hybrid_result, legal = run_legalizer(
                problem.packed_pos,
                hybrid_ops,
                "M2 %s" % hybrid_view.legalization_mode.replace("_", "-"),
            )
            if not legal:
                raise RuntimeError(
                    "%s M2 legalization failed; refusing to "
                    "publish an illegal placement"
                    % hybrid_view.legalization_mode
                )
            result = problem.restore_original_order(pos, hybrid_result)
            if not self.op_collections.legality_check_op(result):
                raise RuntimeError(
                    "%s M2 legalization failed ordinary "
                    "legality after restoring original node order"
                    % hybrid_view.legalization_mode
                )
            audit = hybrid_view.audit(
                result,
                data_collections.node_size_x,
                data_collections.node_size_y,
                problem,
            )
            logging.info(
                "M2 %s audit: hard_non_exempt_overlap_count=%d "
                "hard_non_exempt_overlap_area=%.6g "
                "full_non_exempt_overlap_count=%d "
                "full_non_exempt_overlap_area=%.6g "
                "reserved_overlap_count=%d reserved_overlap_area=%.6g "
                "reserved_max_area_delta=%.6g ordinary_legal=1",
                hybrid_view.legalization_mode.replace("_", "-"),
                audit["non_exempt_overlap_count"],
                audit["non_exempt_overlap_area"],
                audit["full_non_exempt_overlap_count"],
                audit["full_non_exempt_overlap_area"],
                audit["reserved_overlap_count"],
                audit["reserved_overlap_area"],
                audit["reserved_max_area_delta"],
            )
            audit_tolerance = max(
                1.0,
                abs(audit["reserved_overlap_area"]),
            ) * 1e-7
            if audit["non_exempt_overlap_count"] != 0:
                raise RuntimeError(
                    "%s M2 legalization left non-exempt "
                    "cell-to-rail overlap"
                    % hybrid_view.legalization_mode
                )
            if audit["reserved_max_area_delta"] > audit_tolerance:
                raise RuntimeError(
                    "%s M2 legalization changed the reserved "
                    "cell's minimum rail-overlap area"
                    % hybrid_view.legalization_mode
                )
            return result

        result, legal = run_legalizer(pos, ordinary_ops, "ordinary")
        if not legal:
            raise RuntimeError(
                "ordinary legalization failed before M2 soft legalization"
            )
        if self.op_collections.m2_soft_legalize_op is not None:
            result, _ = self.op_collections.m2_soft_legalize_op(result)
        return result

    return build_legalization_op

def run_adaptive_padding_legalization(
    self,
    placedb,
    pos,
    padding_sites,
    scores,
    max_retries=4,
):
    if len(placedb.regions) > 0:
        raise ValueError(
            "post-legalization adaptive padding does not support fence regions"
        )

    data_collections = self.data_collections
    device = pos.device
    active_sites = torch.as_tensor(
        padding_sites,
        device=device,
        dtype=data_collections.node_size_x.dtype,
    ).reshape(-1)
    if active_sites.numel() != placedb.num_movable_nodes:
        raise ValueError(
            "padding_sites must have one entry per movable node"
        )
    active_sites = active_sites.clamp(min=0).clone()
    score_tensor = torch.as_tensor(
        scores,
        device=device,
        dtype=data_collections.node_size_x.dtype,
    ).reshape(-1)
    if score_tensor.numel() != placedb.num_movable_nodes:
        raise ValueError("scores must have one entry per movable node")

    input_pos = pos.detach().clone()
    requested_count = int((active_sites > 0).sum().item())
    if requested_count == 0:
        return input_pos, {
            "requested_count": 0,
            "used_count": 0,
            "attempts": 0,
            "rollback": False,
            "padded_legal": True,
            "physical_legal": bool(
                self.op_collections.legality_check_op(input_pos)
            ),
            "greedy_fallback": False,
            "moved_count": 0,
            "max_displacement": 0.0,
            "total_displacement": 0.0,
        }

    max_retries = max(0, int(max_retries))
    attempts = 0
    greedy_fallback = False
    while attempts <= max_retries and bool((active_sites > 0).any()):
        attempts += 1
        full_padding = torch.zeros_like(data_collections.node_size_x)
        full_padding[: placedb.num_movable_nodes] = (
            active_sites * float(placedb.site_width)
        )
        padded_size_x = data_collections.node_size_x + 2 * full_padding
        padded_pos = input_pos.detach().clone()
        padded_pos[: placedb.num_movable_nodes].sub_(
            full_padding[: placedb.num_movable_nodes]
        )

        padded_check_op = legality_check.LegalityCheck(
            node_size_x=padded_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes,
        )
        macro_op = macro_legalize.MacroLegalize(
            node_size_x=padded_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )
        greedy_op = greedy_legalize.GreedyLegalize(
            node_size_x=padded_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )
        abacus_op = abacus_legalize.AbacusLegalize(
            node_size_x=padded_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )

        macro_result = macro_op(padded_pos, padded_pos)
        greedy_result = greedy_op(macro_result, macro_result)
        padded_legal = bool(padded_check_op(greedy_result))
        if padded_legal:
            abacus_result = abacus_op(macro_result, greedy_result)
            if bool(padded_check_op(abacus_result)):
                padded_result = abacus_result
                greedy_fallback = False
            else:
                padded_result = greedy_result
                greedy_fallback = True

            physical_result = padded_result.detach().clone()
            physical_result[: placedb.num_movable_nodes].add_(
                full_padding[: placedb.num_movable_nodes]
            )
            physical_legal = bool(
                self.op_collections.legality_check_op(physical_result)
            )
            if physical_legal:
                displacement = (
                    physical_result[: placedb.num_movable_nodes]
                    - input_pos[: placedb.num_movable_nodes]
                ).abs() + (
                    physical_result[
                        placedb.num_nodes : placedb.num_nodes
                        + placedb.num_movable_nodes
                    ]
                    - input_pos[
                        placedb.num_nodes : placedb.num_nodes
                        + placedb.num_movable_nodes
                    ]
                ).abs()
                return physical_result, {
                    "requested_count": requested_count,
                    "used_count": int((active_sites > 0).sum().item()),
                    "attempts": attempts,
                    "rollback": False,
                    "padded_legal": True,
                    "physical_legal": True,
                    "greedy_fallback": greedy_fallback,
                    "moved_count": int((displacement > 0).sum().item()),
                    "max_displacement": float(displacement.max().item()),
                    "total_displacement": float(displacement.sum().item()),
                }

        active_ids = torch.nonzero(active_sites > 0, as_tuple=False).reshape(-1)
        keep_count = active_ids.numel() // 2
        if keep_count <= 0:
            active_sites.zero_()
            break
        ranked = active_ids[
            torch.argsort(score_tensor.index_select(0, active_ids), descending=True)
        ]
        keep_ids = ranked[:keep_count]
        reduced_sites = torch.zeros_like(active_sites)
        reduced_sites.index_copy_(
            0, keep_ids, active_sites.index_select(0, keep_ids)
        )
        active_sites = reduced_sites
        logging.warning(
            "Adaptive padded legalization attempt %d failed; "
            "reduce padded cells from %d to %d",
            attempts,
            int(active_ids.numel()),
            int(keep_count),
        )

    return input_pos, {
        "requested_count": requested_count,
        "used_count": 0,
        "attempts": attempts,
        "rollback": True,
        "padded_legal": False,
        "physical_legal": bool(
            self.op_collections.legality_check_op(input_pos)
        ),
        "greedy_fallback": False,
        "moved_count": 0,
        "max_displacement": 0.0,
        "total_displacement": 0.0,
    }

def build_m2_pa_refine(self, params, placedb, data_collections):
    enabled = _resolve_boolean_like_flag(params, "m2_pa_refine_flag")
    if not enabled:
        return None
    if len(placedb.regions) > 0:
        raise ValueError("m2_pa_refine_flag does not support fence regions")

    refine = m2_pa_refine.M2PARefine(
        node_size_x=data_collections.node_size_x,
        node_size_y=data_collections.node_size_y,
        rail_boxes=data_collections.m2_pg_rail_boxes,
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        site_width=placedb.site_width,
        row_height=placedb.row_height,
        num_movable_nodes=placedb.num_movable_nodes,
        num_terminals=placedb.num_terminals,
        max_neighbors=params.m2_pa_refine_max_neighbors,
        max_displacement_sites=params.m2_pa_refine_max_displacement_sites,
    )

    def build_m2_pa_refine_op(pos):
        start_time = time.time()
        hpwl_before = float(self.op_collections.hpwl_op(pos).item())
        refined_pos, stats = refine(pos)
        moved = int(stats["moved_count"])
        legal = True
        rollback = False
        if moved:
            legal = bool(self.op_collections.legality_check_op(refined_pos))
            if not legal:
                rollback = True
                refined_pos = pos.detach().clone()
        hpwl_after = float(self.op_collections.hpwl_op(refined_pos).item())
        stats.update(
            {
                "rail_count": int(data_collections.m2_pg_rail_boxes.size(0)),
                "legal": legal,
                "rollback": rollback,
                "hpwl_before": hpwl_before,
                "hpwl_after": hpwl_after,
                "hpwl_delta": hpwl_after - hpwl_before,
                "runtime_seconds": time.time() - start_time,
            }
        )
        logging.info(
            "M2 PA-Refine rails=%d overlap_count=%.0f->%.0f "
            "overlap_area=%.6g->%.6g moved=%d move_events=%.0f "
            "displacement_total=%.6g displacement_max=%.6g "
            "HPWL=%.6g->%.6g delta=%.6g legal=%d rollback=%d "
            "time=%.3fs",
            stats["rail_count"],
            stats["overlap_count_before"],
            stats["overlap_count_after"],
            stats["overlap_area_before"],
            stats["overlap_area_after"],
            moved,
            stats["move_events"],
            stats["total_displacement"],
            stats["max_displacement"],
            hpwl_before,
            hpwl_after,
            stats["hpwl_delta"],
            int(legal),
            int(rollback),
            stats["runtime_seconds"],
        )
        return refined_pos, stats

    return build_m2_pa_refine_op
