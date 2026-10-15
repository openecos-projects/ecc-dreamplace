"""Evaluate current route supply/overflow and update the existing policy."""
import logging
import torch
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose

def update_overflow(params, model, pos, iteration, l_shape_policy, cur_metric):
    density_map = None
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

                    l_shape_policy.update_overflow_target(
                        model, iteration, l_shape_overflow, overflow_ratio
                    )
        except Exception as e:
            logging.warning(
                f"L-shape overflow outer-loop update failed at iter {iteration}: {e}"
            )

    return density_map
