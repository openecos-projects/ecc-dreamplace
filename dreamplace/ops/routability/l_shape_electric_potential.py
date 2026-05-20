##
# @file   l_shape_electric_potential.py
# @brief  Compute electric potential for L-shape routing segments
#         Uses C++/CUDA backends for density map and force computation
#         Gradients flow back through segment positions to cell positions
#

from doctest import FAIL_FAST
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
import logging
import os

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform
import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

from .l_shape_electric_overflow import SegmentDensityMapFunction
from .plot_map import plot_density_map, plot_potential_map
from .profile_timing import l_shape_log_verbose, profile_end, profile_scope, profile_start

import torch.nn.functional as F

logger = logging.getLogger(__name__)
_PLOT_ITER = 0


def _log_topk_bins(map_tensor, k, name, xl, yl, bin_size_x, bin_size_y):
    if not torch.is_tensor(map_tensor):
        return
    flat = map_tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return
    k = min(k, flat.numel())
    values, idx = torch.topk(flat, k)
    values = values.cpu().tolist()
    idx = idx.cpu().tolist()
    num_bins_y = map_tensor.shape[1]
    entries = []
    for v, i in zip(values, idx):
        ix = i // num_bins_y
        iy = i % num_bins_y
        x = xl + (ix + 0.5) * bin_size_x
        y = yl + (iy + 0.5) * bin_size_y
        entries.append(f"({ix},{iy}) xy=({x:.3f},{y:.3f}) v={v:.3e}")
    logger.info(f"[L-shape topk] {name} top{len(entries)}: " + "; ".join(entries))


class SegmentElectricPotentialFunction(Function):
    """
    Compute electric potential for routing segments.
    
    Forward: Compute density map and potential energy using DCT.
    Backward: Compute gradients using electric field from Poisson equation.
    """
    # Class-level storage for the last computed field maps (for filler reverse force)
    last_field_map_x = None
    last_field_map_y = None
    last_overflow_map = None  # for pseudo wire force
    last_density_map = None
    last_density_map_h = None
    last_density_map_v = None
    last_rho_map = None
    last_rho_map_h = None
    last_rho_map_v = None
    last_energy = None
    last_energy_h = None
    last_energy_v = None
    
    @staticmethod
    def forward(
        ctx,
        segment_pos,
        segment_size_x,
        segment_size_y,
        segment_is_horizontal,
        segment_weight,
        directional_targets,
        segment_size_x_clamped,
        segment_size_y_clamped,
        offset_x,
        offset_y,
        ratio,
        bin_center_x,
        bin_center_y,
        initial_density_map,
        target_density,
        target_demand,
        raw_wire_demand_map,
        supply_original,
        fix_usage_map,
        area_per_track,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_segments,
        padding,
        padding_mask,
        num_bins_x, num_bins_y,
        num_impacted_bins_x,
        num_impacted_bins_y,
        deterministic_flag,
        sorted_segment_map,
        exact_expkM,
        exact_expkN,
        inv_wu2_plus_wv2,
        wu_by_wu2_plus_wv2_half,
        wv_by_wu2_plus_wv2_half,
        dct2,
        idct2,
        idct_idxst,
        idxst_idct,
        fast_mode
    ):
        """
        Compute electric potential energy for segments.
        """
        tt = time.time()
        profile_enabled = bool(getattr(SegmentElectricPotentialFunction, "profile_enabled", False))
        ctx.profile_enabled = profile_enabled
        profile_timer = profile_start(profile_enabled, tensor=segment_pos)
        SegmentElectricPotentialFunction.last_density_map = None
        SegmentElectricPotentialFunction.last_density_map_h = None
        SegmentElectricPotentialFunction.last_density_map_v = None
        SegmentElectricPotentialFunction.last_rho_map = None
        SegmentElectricPotentialFunction.last_rho_map_h = None
        SegmentElectricPotentialFunction.last_rho_map_v = None
        SegmentElectricPotentialFunction.last_energy = None
        SegmentElectricPotentialFunction.last_energy_h = None
        SegmentElectricPotentialFunction.last_energy_v = None

        def _prepare_optional_map(value):
            if not isinstance(value, torch.Tensor):
                return None
            if value.dim() != 2:
                return None
            return value.to(device=segment_pos.device, dtype=segment_pos.dtype)

        def _build_subset(mask):
            if mask is None or mask.numel() == 0 or mask.sum().item() == 0:
                return None
            idx = torch.nonzero(mask, as_tuple=True)[0]
            seg_llx = segment_pos[:num_segments][idx]
            seg_lly = segment_pos[num_segments:][idx]
            seg_pos = torch.cat([seg_llx, seg_lly], dim=0)
            seg_sx = segment_size_x[idx]
            seg_sy = segment_size_y[idx]
            seg_sw = segment_weight[idx] if isinstance(segment_weight, torch.Tensor) else None

            sqrt2 = math.sqrt(2)
            seg_sx_clamped = seg_sx
            seg_sy_clamped = seg_sy
            seg_off_x = (seg_sx - seg_sx_clamped).mul(0.5)
            seg_off_y = (seg_sy - seg_sy_clamped).mul(0.5)
            seg_area = seg_sx * seg_sy
            seg_clamped_area = seg_sx_clamped * seg_sy_clamped
            seg_ratio = seg_area / seg_clamped_area.clamp(min=1e-10)
            if isinstance(seg_sw, torch.Tensor):
                seg_ratio = seg_ratio * seg_sw
            sqrt2_bin_x = sqrt2 * bin_size_x
            sqrt2_bin_y = sqrt2 * bin_size_y
            seg_num_impacted_bins_x = 0
            seg_num_impacted_bins_y = 0
            if seg_sx.numel() > 0:
                seg_num_impacted_bins_x = int(
                    ((seg_sx.max() + 2 * sqrt2_bin_x) / bin_size_x)
                    .ceil().clamp(max=num_bins_x).item()
                )
                seg_num_impacted_bins_y = int(
                    ((seg_sy.max() + 2 * sqrt2_bin_y) / bin_size_y)
                    .ceil().clamp(max=num_bins_y).item()
                )
                seg_sorted_map = torch.argsort(seg_sx).to(torch.int32)
            else:
                seg_sorted_map = torch.tensor([], dtype=torch.int32, device=seg_pos.device)
            return {
                "indices": idx,
                "segment_pos": seg_pos,
                "segment_size_x": seg_sx,
                "segment_size_y": seg_sy,
                "segment_size_x_clamped": seg_sx_clamped,
                "segment_size_y_clamped": seg_sy_clamped,
                "offset_x": seg_off_x,
                "offset_y": seg_off_y,
                "ratio": seg_ratio,
                "num_impacted_bins_x": seg_num_impacted_bins_x,
                "num_impacted_bins_y": seg_num_impacted_bins_y,
                "sorted_segment_map": seg_sorted_map,
                "num_segments": int(seg_sx.numel()),
            }

        def _compute_density_map(prepared):
            if prepared is None or prepared["num_segments"] == 0:
                return torch.zeros_like(initial_density_map)
            return SegmentDensityMapFunction.forward(
                prepared["segment_pos"],
                prepared["segment_size_x"],
                prepared["segment_size_y"],
                prepared["segment_size_x_clamped"],
                prepared["segment_size_y_clamped"],
                prepared["offset_x"],
                prepared["offset_y"],
                prepared["ratio"],
                bin_center_x,
                bin_center_y,
                initial_density_map,
                target_density,
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                prepared["num_segments"],
                padding,
                padding_mask,
                num_bins_x, num_bins_y,
                prepared["num_impacted_bins_x"],
                prepared["num_impacted_bins_y"],
                deterministic_flag,
                prepared["sorted_segment_map"],
            )

        def _compute_blockage_track_rho_components(
            density_seg_area_local,
            supply_original_local,
            fix_usage_local,
        ):
            density_seg_tracks_local = density_seg_area_local / bin_area
            initial_density_tracks_local = fix_usage_local.clamp(min=0)
            capacity_tracks_local = supply_original_local
            occupancy_tracks_local = initial_density_tracks_local + density_seg_tracks_local
            rho_tracks_local = occupancy_tracks_local
            rho_centered_tracks_local = rho_tracks_local - rho_tracks_local.mean()
            rho_map_local = rho_centered_tracks_local
            overflow_map_local = (occupancy_tracks_local - capacity_tracks_local).clamp(min=0)
            utilization = occupancy_tracks_local / capacity_tracks_local.clamp(min=1e-6)
            return (
                rho_map_local,
                overflow_map_local,
                utilization,
                initial_density_tracks_local,
                density_seg_tracks_local,
                occupancy_tracks_local,
                capacity_tracks_local,
                rho_centered_tracks_local,
            )

        def _compute_field_and_energy(rho_map_local):
            rho_map_normalized_local = rho_map_local.clone()
            rho_map_normalized_local.mul_(1.0 / bin_area)
            auv_local = dct2.forward(rho_map_normalized_local)
            field_map_x_local = idxst_idct.forward(
                auv_local.mul(wu_by_wu2_plus_wv2_half)
            )
            field_map_y_local = idct_idxst.forward(
                auv_local.mul(wv_by_wu2_plus_wv2_half)
            )
            if fast_mode:
                energy_local = torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device)
            else:
                potential_map_local = idct2.forward(auv_local.mul(inv_wu2_plus_wv2))
                potential_map_local.mul_(bin_area)
                energy_local = potential_map_local.mul(rho_map_normalized_local).sum()
            return rho_map_normalized_local, field_map_x_local, field_map_y_local, energy_local

        target_density_h = None
        target_density_v = None
        target_demand_h = None
        target_demand_v = None
        supply_original_h = None
        supply_original_v = None
        raw_wire_demand_map_h = None
        raw_wire_demand_map_v = None
        fix_usage_map_h = None
        fix_usage_map_v = None
        if isinstance(directional_targets, (tuple, list)) and len(directional_targets) == 10:
            target_density_h = _prepare_optional_map(directional_targets[0])
            target_density_v = _prepare_optional_map(directional_targets[1])
            target_demand_h = _prepare_optional_map(directional_targets[2])
            target_demand_v = _prepare_optional_map(directional_targets[3])
            raw_wire_demand_map_h = _prepare_optional_map(directional_targets[4])
            raw_wire_demand_map_v = _prepare_optional_map(directional_targets[5])
            supply_original_h = _prepare_optional_map(directional_targets[6])
            supply_original_v = _prepare_optional_map(directional_targets[7])
            fix_usage_map_h = _prepare_optional_map(directional_targets[8])
            fix_usage_map_v = _prepare_optional_map(directional_targets[9])

        density_map_h = None
        density_map_v = None
        ctx.hv_split_active = False
        ctx.h_split_data = None
        ctx.v_split_data = None

        hv_split = isinstance(segment_is_horizontal, torch.Tensor) and segment_is_horizontal.numel() == num_segments
        directional_split = (
            hv_split
            and isinstance(target_density_h, torch.Tensor)
            and target_density_h.dim() == 2
            and isinstance(target_density_v, torch.Tensor)
            and target_density_v.dim() == 2
        )

        density_timer = profile_start(profile_enabled, tensor=segment_pos)
        if hv_split:
            mask_h = segment_is_horizontal.to(torch.bool)
            mask_v = ~mask_h
            prepared_h = _build_subset(mask_h)
            prepared_v = _build_subset(mask_v)
            density_map_h = _compute_density_map(prepared_h)
            density_map_v = _compute_density_map(prepared_v)
            density_map = density_map_h + density_map_v
            SegmentElectricPotentialFunction.last_density_map_h = density_map_h.detach()
            SegmentElectricPotentialFunction.last_density_map_v = density_map_v.detach()
        else:
            prepared_h = None
            prepared_v = None
            density_map = SegmentDensityMapFunction.forward(
                segment_pos,
                segment_size_x,
                segment_size_y,
                segment_size_x_clamped,
                segment_size_y_clamped,
                offset_x,
                offset_y,
                ratio,
                bin_center_x,
                bin_center_y,
                initial_density_map,
                target_density,
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_segments,
                padding,
                padding_mask,
                num_bins_x, num_bins_y,
                num_impacted_bins_x,
                num_impacted_bins_y,
                deterministic_flag,
                sorted_segment_map
            )
        SegmentElectricPotentialFunction.last_density_map = density_map.detach()
        profile_end(
            profile_enabled,
            density_timer,
            "electric.forward.density_map",
            tensor=segment_pos,
            logger=logger,
            segments=num_segments,
            hv_split=int(hv_split),
        )

        # plot density map
        # plot_density_map(density_map, 
        #     title="Segment Density Map", 
        #     save_path=f"/home/sxr/workspace/test-benchmark/benchmark/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/segment_density_map.png")
        
        # exit(0)
        
        # Save for backward
        ctx.segment_pos = segment_pos
        ctx.segment_size_x_clamped = segment_size_x_clamped
        ctx.segment_size_y_clamped = segment_size_y_clamped
        ctx.offset_x = offset_x
        ctx.offset_y = offset_y
        ctx.ratio = ratio
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.target_density = target_density
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_segments = num_segments
        ctx.padding = padding
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_impacted_bins_x = num_impacted_bins_x
        ctx.num_impacted_bins_y = num_impacted_bins_y
        ctx.deterministic_flag = deterministic_flag
        ctx.sorted_segment_map = sorted_segment_map
        
        # DCT-based Poisson solver
        M = num_bins_x
        N = num_bins_y
        
        # Compute a signed rho map based on routing demand vs supply.
        # Overflow remains as telemetry, but the Poisson solver should see
        # both positive residuals (capacity pressure) and negative residuals
        # (available resource).
        bin_area = bin_size_x * bin_size_y
        if not isinstance(target_density, torch.Tensor) or target_density.dim() != 2:
            raise TypeError(
                "L-shape electric potential requires target_density to be a 2D routing supply map tensor"
            )
        if not isinstance(target_demand, torch.Tensor) or target_demand.dim() != 2:
            raise TypeError(
                "L-shape electric potential requires target_demand to be a 2D routing demand map tensor"
            )
        planar_fix_usage_map = _prepare_optional_map(fix_usage_map)

        supply_map = target_density.to(device=segment_pos.device, dtype=segment_pos.dtype)
        supply_original_map = None
        if isinstance(supply_original, torch.Tensor) and supply_original.dim() == 2:
            supply_original_map = supply_original.to(device=segment_pos.device, dtype=segment_pos.dtype)
        if isinstance(supply_original_map, torch.Tensor):
            current_demand_supply_ratio = float(
                (
                    (density_map.sum() / bin_area)
                    / supply_original_map.sum().clamp(min=1e-6)
                ).detach().item()
            )
        else:
            current_demand_supply_ratio = float(
                (
                    density_map.sum()
                    / supply_map.sum().clamp(min=1e-6)
                ).detach().item()
            )
        SegmentElectricPotentialFunction.last_demand_supply_ratio = current_demand_supply_ratio
        log_verbose = l_shape_log_verbose(
            getattr(SegmentElectricPotentialFunction, "log_verbose", 0)
        )
        if log_verbose >= 2 and isinstance(supply_original_map, torch.Tensor):
            logger.info(
                f"[L-shape supply/demand] density_seg_tracks_sum={(density_map.sum() / bin_area).item():.3e}, "
                f"supply_original_sum={supply_original_map.sum().item():.3e}, "
                f"ratio={current_demand_supply_ratio:.2f}"
            )
        elif log_verbose >= 2:
            logger.info(
                f"[L-shape supply/demand] demand_sum={density_map.sum().item():.3e}, "
                f"supply_sum={supply_map.sum().item():.3e}, "
                f"ratio={current_demand_supply_ratio:.2f}"
            )

        if (
            directional_split
            and isinstance(supply_original_h, torch.Tensor)
            and isinstance(supply_original_v, torch.Tensor)
            and isinstance(fix_usage_map_h, torch.Tensor)
            and isinstance(fix_usage_map_v, torch.Tensor)
        ):
            (
                rho_map_h,
                overflow_map_h,
                utilization_h,
                initial_density_tracks_h,
                density_seg_tracks_h,
                occupancy_tracks_h,
                capacity_tracks_h,
                rho_centered_tracks_h,
            ) = _compute_blockage_track_rho_components(
                density_map_h,
                supply_original_h,
                fix_usage_map_h,
            )
            (
                rho_map_v,
                overflow_map_v,
                utilization_v,
                initial_density_tracks_v,
                density_seg_tracks_v,
                occupancy_tracks_v,
                capacity_tracks_v,
                rho_centered_tracks_v,
            ) = _compute_blockage_track_rho_components(
                density_map_v,
                supply_original_v,
                fix_usage_map_v,
            )

            with profile_scope(
                profile_enabled,
                "electric.forward.field_energy",
                tensor=segment_pos,
                logger=logger,
                mode="hv_split",
            ):
                _, field_map_x_h, field_map_y_h, energy_h = _compute_field_and_energy(rho_map_h)
                _, field_map_x_v, field_map_y_v, energy_v = _compute_field_and_energy(rho_map_v)

            ctx.hv_split_active = True
            ctx.h_split_data = prepared_h
            ctx.v_split_data = prepared_v
            ctx.h_field_map_x = field_map_x_h
            ctx.h_field_map_y = field_map_y_h
            ctx.v_field_map_x = field_map_x_v
            ctx.v_field_map_y = field_map_y_v
            ctx.field_map_x = None
            ctx.field_map_y = None

            energy = energy_h + energy_v
            rho_map = rho_map_h + rho_map_v
            overflow_map = overflow_map_h + overflow_map_v
            SegmentElectricPotentialFunction.last_rho_map_h = rho_map_h.detach()
            SegmentElectricPotentialFunction.last_rho_map_v = rho_map_v.detach()
            SegmentElectricPotentialFunction.last_rho_map = rho_map.detach()
            SegmentElectricPotentialFunction.last_energy_h = energy_h.detach()
            SegmentElectricPotentialFunction.last_energy_v = energy_v.detach()
            SegmentElectricPotentialFunction.last_energy = energy.detach()

            logger.debug(
                "Blockage rho(split,track): occ_ratio_h=%.4f occ_ratio_v=%.4f util_h_max=%.2f util_v_max=%.2f "
                "rho_center_pos_bins_h=%d/%d rho_center_neg_bins_h=%d/%d overflow_bins_h=%d/%d "
                "rho_center_pos_bins_v=%d/%d rho_center_neg_bins_v=%d/%d overflow_bins_v=%d/%d",
                float(occupancy_tracks_h.sum().item() / capacity_tracks_h.sum().clamp(min=1e-6).item()) if capacity_tracks_h.numel() > 0 else 0.0,
                float(occupancy_tracks_v.sum().item() / capacity_tracks_v.sum().clamp(min=1e-6).item()) if capacity_tracks_v.numel() > 0 else 0.0,
                float(utilization_h.max().item()) if utilization_h.numel() > 0 else 0.0,
                float(utilization_v.max().item()) if utilization_v.numel() > 0 else 0.0,
                int((rho_centered_tracks_h > 0).sum().item()),
                int(rho_centered_tracks_h.numel()),
                int((rho_centered_tracks_h < 0).sum().item()),
                int(rho_centered_tracks_h.numel()),
                int((overflow_map_h > 0).sum().item()),
                int(overflow_map_h.numel()),
                int((rho_centered_tracks_v > 0).sum().item()),
                int(rho_centered_tracks_v.numel()),
                int((rho_centered_tracks_v < 0).sum().item()),
                int(rho_centered_tracks_v.numel()),
                int((overflow_map_v > 0).sum().item()),
                int(overflow_map_v.numel()),
            )
        elif isinstance(supply_original_map, torch.Tensor) and isinstance(planar_fix_usage_map, torch.Tensor):
            (
                rho_map,
                overflow_map,
                utilization,
                initial_density_tracks,
                density_seg_tracks,
                occupancy_tracks,
                capacity_tracks,
                rho_centered_tracks,
            ) = _compute_blockage_track_rho_components(
                density_map,
                supply_original_map,
                planar_fix_usage_map,
            )
            with profile_scope(
                profile_enabled,
                "electric.forward.field_energy",
                tensor=segment_pos,
                logger=logger,
                mode="planar",
            ):
                rho_map_normalized, field_map_x, field_map_y, energy = _compute_field_and_energy(rho_map)
            ctx.field_map_x = field_map_x
            ctx.field_map_y = field_map_y
            SegmentElectricPotentialFunction.last_rho_map = rho_map.detach()
            SegmentElectricPotentialFunction.last_energy = energy.detach()

            logger.debug(
                "Blockage rho(planar,track): occ_ratio=%.4f util_mean=%.2f util_max=%.2f "
                "rho_center_pos_bins=%d/%d rho_center_neg_bins=%d/%d overflow_bins=%d/%d",
                float(occupancy_tracks.sum().item() / capacity_tracks.sum().clamp(min=1e-6).item()) if capacity_tracks.numel() > 0 else 0.0,
                float(utilization.mean().item()) if utilization.numel() > 0 else 0.0,
                float(utilization.max().item()) if utilization.numel() > 0 else 0.0,
                int((rho_centered_tracks > 0).sum().item()),
                int(rho_centered_tracks.numel()),
                int((rho_centered_tracks < 0).sum().item()),
                int(rho_centered_tracks.numel()),
                int((overflow_map > 0).sum().item()),
                int(overflow_map.numel()),
            )
        else:
            raise TypeError(
                "L-shape blockage initial density requires supply_original and fix_usage maps for either planar or H/V split routing data"
            )
        

        # plot overflow map
        # global _PLOT_ITER
        # iter = _PLOT_ITER
        # _PLOT_ITER += 1
        # plot_root = "/home/sxr/workspace/test-benchmark/benchmark/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot"
        # if logger.isEnabledFor(logging.INFO):
        #     ov_mean = overflow_map_normalized.mean().item()
        #     ov_sum = overflow_map_normalized.sum().item()
        #     logger.info(f"[L-shape stats] overflow_norm_iter{iter}: mean={ov_mean:.3e}, sum={ov_sum:.3e}")
        #     # log a few low-frequency DCT coefficients (real)
        #     if auv.numel() >= 4:
        #         a01 = auv[0, 1].item()
        #         a10 = auv[1, 0].item()
        #         a11 = auv[1, 1].item()
        #         logger.info(
        #             f"[L-shape stats] dct_lowfreq_iter{iter}: "
        #             f"a01={a01:.3e}, a10={a10:.3e}, a11={a11:.3e}"
        #         )
        #     _log_topk_bins(
        #         overflow_map_normalized, 10,
        #         f"overflow_norm_iter{iter}",
        #         xl, yl, bin_size_x, bin_size_y
        #     )
        #     if not fast_mode:
        #         _log_topk_bins(
        #             potential_map, 10,
        #             f"potential_iter{iter}",
        #             xl, yl, bin_size_x, bin_size_y
        #         )
        # plot_dirs = [
        #     "overflow",
        #     "overflow_h",
        #     "overflow_v",
        #     "density",
        #     "egr_supply",
        #     "egr_overflow",
        #     "egr_overflow_resampled",
        #     "egr_netmap",
        #     "egr_supply_original",
        #     "potential",
        # ]
        # for d in plot_dirs:
        #     os.makedirs(os.path.join(plot_root, d), exist_ok=True)

        # plot_density_map(overflow_map_normalized, 
        #     title="Overflow Map", 
        #     save_path=f"{plot_root}/overflow/overflow_map_iter{iter}.png")

        # if hv_split and 'overflow_in_tracks_h' in locals() and 'overflow_in_tracks_v' in locals():
        #     overflow_map_h = overflow_in_tracks_h * calibrated_area_per_track
        #     overflow_map_v = overflow_in_tracks_v * calibrated_area_per_track
        #     plot_density_map(overflow_map_h, 
        #         title="Overflow Map (H)", 
        #         save_path=f"{plot_root}/overflow_h/overflow_map_h_iter{iter}.png")
        #     plot_density_map(overflow_map_v, 
        #         title="Overflow Map (V)", 
        #         save_path=f"{plot_root}/overflow_v/overflow_map_v_iter{iter}.png")
        
        # # plot density map (segment demand)
        # plot_density_map(density_map, 
        #     title="Density Map (Segment Demand)", 
        #     save_path=f"{plot_root}/density/density_map_iter{iter}.png")
        
        # # Debug: print statistics
        # # logger.info(f"=== Debug Statistics ===")
        # # logger.info(f"density_map: min={density_map.min():.2f}, max={density_map.max():.2f}, sum={density_map.sum():.2f}")
        # # logger.info(f"supply_map (target_density): min={supply_map.min():.2f}, max={supply_map.max():.2f}, sum={supply_map.sum():.2f}")
        # # logger.info(f"supply_map shape: {supply_map.shape}, density_map shape: {density_map.shape}")
        
        # # plot EGR supply map
        # plot_density_map(supply_map, 
        #     title="EGR Supply Map", 
        #     save_path=f"{plot_root}/egr_supply/egr_supply_map_iter{iter}.png")
        
        # # plot EGR overflow map
        # from .egr_resample import load_egr_csv_map
        # egr_overflow_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/APU/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/overflow_map_planar.csv"
        # egr_overflow_map = load_egr_csv_map(egr_overflow_path)
        # egr_overflow_tensor = torch.from_numpy(egr_overflow_map).to(segment_pos.device)
        # plot_density_map(egr_overflow_tensor, 
        #     title="EGR Overflow Map (from CSV)", 
        #     save_path=f"{plot_root}/egr_overflow/egr_overflow_map_iter{iter}.png")
        # egr_overflow_resampled = F.interpolate(
        #     egr_overflow_tensor.unsqueeze(0).unsqueeze(0),
        #     size=(num_bins_x, num_bins_y),
        #     mode="bilinear",
        #     align_corners=False
        # ).squeeze(0).squeeze(0)
        # plot_density_map(egr_overflow_resampled, 
        #     title="EGR Overflow Map (resampled)", 
        #     save_path=f"{plot_root}/egr_overflow_resampled/egr_overflow_resampled_iter{iter}.png")

        # egr_netmap_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/APU/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/net_map_planar.csv"
        # egr_netmap = load_egr_csv_map(egr_netmap_path)
        # egr_netmap_tensor = torch.from_numpy(egr_netmap).to(segment_pos.device)
        # plot_density_map(egr_netmap_tensor, 
        #     title="EGR Netmap (from CSV)", 
        #     save_path=f"{plot_root}/egr_netmap/egr_netmap_iter{iter}.png")
        
        # # plot EGR ORIGINAL supply map (before resample) for comparison
        # # egr_supply_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/APU/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/supply_map_planar.csv"
        # # egr_supply_original = load_egr_csv_map(egr_supply_path)
        # # egr_supply_original_tensor = torch.from_numpy(egr_supply_original).to(segment_pos.device)
        # # logger.info(f"EGR ORIGINAL supply: shape={egr_supply_original.shape}, min={egr_supply_original.min():.2f}, max={egr_supply_original.max():.2f}")
        # # plot_density_map(egr_supply_original_tensor, 
        # #     title="EGR Supply Map (ORIGINAL from CSV)", 
        # #     save_path=f"{plot_root}/egr_supply_original/egr_supply_original_iter{iter}.png")
        
        if segment_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug(f"Segment electric potential forward: {(time.time() - tt) * 1000:.2f} ms")

        # Save field maps for filler reverse force (detached, no autograd)
        if ctx.hv_split_active:
            if ctx.h_field_map_x is not None and ctx.v_field_map_x is not None:
                SegmentElectricPotentialFunction.last_field_map_x = (ctx.h_field_map_x + ctx.v_field_map_x).detach()
                SegmentElectricPotentialFunction.last_field_map_y = (ctx.h_field_map_y + ctx.v_field_map_y).detach()
            elif ctx.h_field_map_x is not None:
                SegmentElectricPotentialFunction.last_field_map_x = ctx.h_field_map_x.detach()
                SegmentElectricPotentialFunction.last_field_map_y = ctx.h_field_map_y.detach()
            elif ctx.v_field_map_x is not None:
                SegmentElectricPotentialFunction.last_field_map_x = ctx.v_field_map_x.detach()
                SegmentElectricPotentialFunction.last_field_map_y = ctx.v_field_map_y.detach()
            # Save overflow map for pseudo wire force (HV split: sum of h and v)
            if 'overflow_map_h' in locals() and 'overflow_map_v' in locals():
                SegmentElectricPotentialFunction.last_overflow_map = (overflow_map_h + overflow_map_v).detach()
            elif 'overflow_map_h' in locals():
                SegmentElectricPotentialFunction.last_overflow_map = overflow_map_h.detach()
            elif 'overflow_map_v' in locals():
                SegmentElectricPotentialFunction.last_overflow_map = overflow_map_v.detach()
        elif ctx.field_map_x is not None:
            SegmentElectricPotentialFunction.last_field_map_x = ctx.field_map_x.detach()
            SegmentElectricPotentialFunction.last_field_map_y = ctx.field_map_y.detach()
            # Save overflow map for pseudo wire force (planar)
            if 'overflow_map' in locals():
                SegmentElectricPotentialFunction.last_overflow_map = overflow_map.detach()
        
        profile_end(
            profile_enabled,
            profile_timer,
            "electric.forward.total",
            tensor=segment_pos,
            logger=logger,
            segments=num_segments,
            hv_split=int(hv_split),
            directional_split=int(directional_split),
        )
        return energy
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients using electric force.
        """
        tt = time.time()
        profile_enabled = bool(getattr(ctx, "profile_enabled", False))
        profile_timer = profile_start(profile_enabled, tensor=grad_output)

        def _electric_force_subset(field_map_x, field_map_y, split_data):
            if split_data is None or split_data["num_segments"] == 0:
                return None
            num_movable_nodes = split_data["num_segments"]
            num_filler_nodes = 0
            if grad_output.is_cuda:
                return -electric_potential_cuda.electric_force(
                    grad_output,
                    ctx.num_bins_x, ctx.num_bins_y,
                    split_data["num_impacted_bins_x"], split_data["num_impacted_bins_y"],
                    0, 0,
                    field_map_x.view([-1]),
                    field_map_y.view([-1]),
                    split_data["segment_pos"],
                    split_data["segment_size_x_clamped"],
                    split_data["segment_size_y_clamped"],
                    split_data["offset_x"],
                    split_data["offset_y"],
                    split_data["ratio"],
                    ctx.bin_center_x,
                    ctx.bin_center_y,
                    ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                    ctx.bin_size_x, ctx.bin_size_y,
                    num_movable_nodes, num_filler_nodes,
                    ctx.deterministic_flag,
                    split_data["sorted_segment_map"],
                )
            return -electric_potential_cpp.electric_force(
                grad_output,
                ctx.num_bins_x, ctx.num_bins_y,
                split_data["num_impacted_bins_x"], split_data["num_impacted_bins_y"],
                0, 0,
                field_map_x.view([-1]),
                field_map_y.view([-1]),
                split_data["segment_pos"],
                split_data["segment_size_x_clamped"],
                split_data["segment_size_y_clamped"],
                split_data["offset_x"],
                split_data["offset_y"],
                split_data["ratio"],
                ctx.bin_center_x,
                ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                num_movable_nodes, num_filler_nodes,
            )

        if getattr(ctx, "hv_split_active", False):
            output = torch.zeros_like(ctx.segment_pos)

            for split_data, field_map_x, field_map_y in (
                (ctx.h_split_data, getattr(ctx, "h_field_map_x", None), getattr(ctx, "h_field_map_y", None)),
                (ctx.v_split_data, getattr(ctx, "v_field_map_x", None), getattr(ctx, "v_field_map_y", None)),
            ):
                branch_output = _electric_force_subset(field_map_x, field_map_y, split_data)
                if branch_output is None:
                    continue
                indices = split_data["indices"]
                num_branch = split_data["num_segments"]
                output[indices] = branch_output[:num_branch]
                output[indices + ctx.num_segments] = branch_output[num_branch:]
        else:
            num_movable_nodes = ctx.num_segments
            num_filler_nodes = 0

            if grad_output.is_cuda:
                output = -electric_potential_cuda.electric_force(
                    grad_output,
                    ctx.num_bins_x, ctx.num_bins_y,
                    ctx.num_impacted_bins_x, ctx.num_impacted_bins_y,
                    0, 0,  # filler impacted bins
                    ctx.field_map_x.view([-1]),
                    ctx.field_map_y.view([-1]),
                    ctx.segment_pos,
                    ctx.segment_size_x_clamped,
                    ctx.segment_size_y_clamped,
                    ctx.offset_x,
                    ctx.offset_y,
                    ctx.ratio,
                    ctx.bin_center_x,
                    ctx.bin_center_y,
                    ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                    ctx.bin_size_x, ctx.bin_size_y,
                    num_movable_nodes, num_filler_nodes,
                    ctx.deterministic_flag,
                    ctx.sorted_segment_map
                )
            else:
                output = -electric_potential_cpp.electric_force(
                    grad_output,
                    ctx.num_bins_x, ctx.num_bins_y,
                    ctx.num_impacted_bins_x, ctx.num_impacted_bins_y,
                    0, 0,  # filler impacted bins
                    ctx.field_map_x.view([-1]),
                    ctx.field_map_y.view([-1]),
                    ctx.segment_pos,
                    ctx.segment_size_x_clamped,
                    ctx.segment_size_y_clamped,
                    ctx.offset_x,
                    ctx.offset_y,
                    ctx.ratio,
                    ctx.bin_center_x,
                    ctx.bin_center_y,
                    ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                    ctx.bin_size_x, ctx.bin_size_y,
                    num_movable_nodes, num_filler_nodes
                )
        
        if grad_output.is_cuda:
            torch.cuda.synchronize()
        logger.debug(f"Segment electric potential backward: {(time.time() - tt) * 1000:.2f} ms")
        profile_end(
            profile_enabled,
            profile_timer,
            "electric.backward.total",
            tensor=grad_output,
            logger=logger,
            segments=ctx.num_segments,
            hv_split=int(getattr(ctx, "hv_split_active", False)),
        )
        
        # Return gradients (only for segment_pos, others are None)
        return (output,) + (None,) * 46


class LShapeElectricPotential(nn.Module):
    """
    Compute electric potential for L-shape routing segments.
    
    This module:
    1. Computes density map for segments using C++/CUDA
    2. Solves Poisson equation using DCT
    3. Computes gradients using electric force with C++/CUDA
    
    The target_density must be a 2D per-bin routing supply tensor.
    The target_demand (optional) must be a 2D per-bin routing demand tensor.
    """
    
    def __init__(
        self,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        target_density=None,
        target_demand=None,
        raw_wire_demand_map=None,
        supply_original=None,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
        raw_wire_demand_map_h=None,
        raw_wire_demand_map_v=None,
        supply_original_h=None,
        supply_original_v=None,
        fix_usage_map=None,
        fix_usage_map_h=None,
        fix_usage_map_v=None,
        padding=0,
        deterministic_flag=True,
        fast_mode=False,
        profile_enabled=False,
        log_verbose=0,
    ):
        """
        Initialize L-shape electric potential module.
        
        Args:
            xl, yl, xh, yh: die boundaries
            bin_size_x, bin_size_y: bin sizes
            num_bins_x, num_bins_y: number of bins
            target_density: 2D routing supply tensor
            target_demand: optional 2D routing demand tensor
            padding: bin padding
            deterministic_flag: whether to use deterministic routine
            fast_mode: if True, skip energy computation (only gradients)
        """
        super(LShapeElectricPotential, self).__init__()
        
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.padding = padding
        self.deterministic_flag = deterministic_flag
        self.last_demand_supply_ratio = None
        self.fast_mode = fast_mode
        self.profile_enabled = bool(profile_enabled)
        self.log_verbose = l_shape_log_verbose(log_verbose)
        self.blockage_initial_density = True
        
        if isinstance(target_density, torch.Tensor):
            self.register_buffer('target_density', target_density)
        else:
            raise TypeError(
                "LShapeElectricPotential requires target_density to be a 2D routing supply tensor"
            )

        # Store target_demand (2D tensor from the current routing-oracle demand map)
        if isinstance(target_demand, torch.Tensor):
            self.register_buffer('target_demand', target_demand)
        else:
            raise TypeError(
                "LShapeElectricPotential requires target_demand to be a 2D routing demand tensor"
            )
        self.register_buffer(
            'raw_wire_demand_map',
            raw_wire_demand_map if isinstance(raw_wire_demand_map, torch.Tensor) else None,
        )
        self.register_buffer(
            'supply_original',
            supply_original if isinstance(supply_original, torch.Tensor) else None,
        )

        self.register_buffer(
            'target_density_h',
            target_density_h if isinstance(target_density_h, torch.Tensor) else None,
        )
        self.register_buffer(
            'target_density_v',
            target_density_v if isinstance(target_density_v, torch.Tensor) else None,
        )
        self.register_buffer(
            'target_demand_h',
            target_demand_h if isinstance(target_demand_h, torch.Tensor) else None,
        )
        self.register_buffer(
            'target_demand_v',
            target_demand_v if isinstance(target_demand_v, torch.Tensor) else None,
        )
        self.register_buffer(
            'raw_wire_demand_map_h',
            raw_wire_demand_map_h if isinstance(raw_wire_demand_map_h, torch.Tensor) else None,
        )
        self.register_buffer(
            'raw_wire_demand_map_v',
            raw_wire_demand_map_v if isinstance(raw_wire_demand_map_v, torch.Tensor) else None,
        )
        self.register_buffer(
            'supply_original_h',
            supply_original_h if isinstance(supply_original_h, torch.Tensor) else None,
        )
        self.register_buffer(
            'supply_original_v',
            supply_original_v if isinstance(supply_original_v, torch.Tensor) else None,
        )
        self.register_buffer(
            'fix_usage_map',
            fix_usage_map if isinstance(fix_usage_map, torch.Tensor) else None,
        )
        self.register_buffer(
            'fix_usage_map_h',
            fix_usage_map_h if isinstance(fix_usage_map_h, torch.Tensor) else None,
        )
        self.register_buffer(
            'fix_usage_map_v',
            fix_usage_map_v if isinstance(fix_usage_map_v, torch.Tensor) else None,
        )

        # Persistent calibration factor (area per track), initialized on first forward
        self.register_buffer('area_per_track', torch.tensor(0.0))
        
        # Will be initialized on first forward pass
        self.bin_center_x = None
        self.bin_center_y = None
        self.padding_mask = None
        self.initial_density_map = None
        
        # DCT related
        self.exact_expkM = None
        self.exact_expkN = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None
        self.dct2 = None
        self.idct2 = None
        self.idct_idxst = None
        self.idxst_idct = None
    
    def _init_bins(self, device, dtype):
        """Initialize bin centers and padding mask."""
        # Bin centers
        self.bin_center_x = torch.arange(
            self.num_bins_x, device=device, dtype=dtype
        ).mul_(self.bin_size_x).add_(self.xl + self.bin_size_x / 2)
        
        self.bin_center_y = torch.arange(
            self.num_bins_y, device=device, dtype=dtype
        ).mul_(self.bin_size_y).add_(self.yl + self.bin_size_y / 2)
        
        # Padding mask (使用bool类型以兼容新版PyTorch的masked_fill_)
        if self.padding > 0:
            self.padding_mask = torch.ones(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.bool, device=device
            )
            self.padding_mask[
                self.padding:self.num_bins_x - self.padding,
                self.padding:self.num_bins_y - self.padding
            ] = False
        else:
            self.padding_mask = torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.bool, device=device
            )
        
        # Initial density map
        self.initial_density_map = torch.zeros(
            self.num_bins_x, self.num_bins_y,
            dtype=dtype, device=device
        )
        
        # Ensure target_density is properly initialized as tensor
        self._init_target_density(device, dtype)
        self._init_target_demand(device, dtype)
        self._init_optional_target_map('raw_wire_demand_map', device, dtype)
        self._init_optional_target_map('supply_original', device, dtype)
        self._init_optional_target_map('target_density_h', device, dtype)
        self._init_optional_target_map('target_density_v', device, dtype)
        self._init_optional_target_map('target_demand_h', device, dtype)
        self._init_optional_target_map('target_demand_v', device, dtype)
        self._init_optional_target_map('raw_wire_demand_map_h', device, dtype)
        self._init_optional_target_map('raw_wire_demand_map_v', device, dtype)
        self._init_optional_target_map('supply_original_h', device, dtype)
        self._init_optional_target_map('supply_original_v', device, dtype)
        self._init_optional_target_map('fix_usage_map', device, dtype)
        self._init_optional_target_map('fix_usage_map_h', device, dtype)
        self._init_optional_target_map('fix_usage_map_v', device, dtype)
        self.area_per_track = self.area_per_track.to(device=device, dtype=dtype)
    
    def _init_target_density(self, device, dtype):
        """Initialize or convert target_density to proper tensor format."""
        if not isinstance(self.target_density, torch.Tensor):
            raise TypeError(
                "LShapeElectricPotential requires target_density to be a 2D routing supply tensor"
            )
        if self.target_density.shape != (self.num_bins_x, self.num_bins_y):
            logger.warning(
                f"target_density shape {self.target_density.shape} != "
                f"expected ({self.num_bins_x}, {self.num_bins_y}), resizing..."
            )
            from torch.nn.functional import interpolate
            td = self.target_density.unsqueeze(0).unsqueeze(0)
            td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            self.target_density = td.squeeze(0).squeeze(0).to(device=device, dtype=dtype)
        else:
            self.target_density = self.target_density.to(device=device, dtype=dtype)

    def _init_target_demand(self, device, dtype):
        """Initialize or convert target_demand to proper tensor format."""
        if isinstance(self.target_demand, torch.Tensor):
            if self.target_demand.shape != (self.num_bins_x, self.num_bins_y):
                logger.warning(f"target_demand shape {self.target_demand.shape} != "
                              f"expected ({self.num_bins_x}, {self.num_bins_y}), resizing...")
                from torch.nn.functional import interpolate
                td = self.target_demand.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                self.target_demand = td.squeeze(0).squeeze(0).to(device=device, dtype=dtype)
            else:
                self.target_demand = self.target_demand.to(device=device, dtype=dtype)
        else:
            raise TypeError(
                "LShapeElectricPotential requires target_demand to be a 2D routing demand tensor"
            )

    def _init_optional_target_map(self, name, device, dtype):
        target_map = getattr(self, name, None)
        if target_map is None:
            return
        if not isinstance(target_map, torch.Tensor):
            raise TypeError(f"{name} must be a 2D routing tensor or None")
        if target_map.shape != (self.num_bins_x, self.num_bins_y):
            logger.warning(
                "%s shape %s != expected (%d, %d), resizing...",
                name,
                tuple(target_map.shape),
                self.num_bins_x,
                self.num_bins_y,
            )
            td = target_map.unsqueeze(0).unsqueeze(0)
            td = F.interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            target_map = td.squeeze(0).squeeze(0)
        setattr(self, name, target_map.to(device=device, dtype=dtype))
    
    def set_target_density(self, target_density):
        """
        Set target density (routing supply map).

        Args:
            target_density: 2D tensor (num_bins_x, num_bins_y)
        """
        if not isinstance(target_density, torch.Tensor):
            raise TypeError(
                "LShapeElectricPotential requires target_density to be a 2D routing supply tensor"
            )
        if target_density.shape != (self.num_bins_x, self.num_bins_y):
            from torch.nn.functional import interpolate
            td = target_density.unsqueeze(0).unsqueeze(0)
            td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            target_density = td.squeeze(0).squeeze(0)

        if self.bin_center_x is not None:
            target_density = target_density.to(
                device=self.bin_center_x.device,
                dtype=self.bin_center_x.dtype
            )
        self.target_density = target_density
        if self.log_verbose >= 2:
            logger.info(
                f"Set target_density from routing supply map: "
                f"min={target_density.min():.3f}, max={target_density.max():.3f}, "
                f"mean={target_density.mean():.3f}"
            )

    def set_target_demand(self, target_demand):
        """
        Set target demand (routing demand map).
        
        Args:
            target_demand: 2D tensor (num_bins_x, num_bins_y)
        """
        if isinstance(target_demand, torch.Tensor):
            if target_demand.shape != (self.num_bins_x, self.num_bins_y):
                from torch.nn.functional import interpolate
                td = target_demand.unsqueeze(0).unsqueeze(0)
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                target_demand = td.squeeze(0).squeeze(0)
            
            if self.bin_center_x is not None:
                target_demand = target_demand.to(
                    device=self.bin_center_x.device, 
                    dtype=self.bin_center_x.dtype
                )
            self.target_demand = target_demand
            if isinstance(self.area_per_track, torch.Tensor):
                self.area_per_track.zero_()
            if self.log_verbose >= 2:
                logger.info(f"Set target_demand from routing demand map: "
                           f"min={target_demand.min():.3f}, max={target_demand.max():.3f}, "
                           f"mean={target_demand.mean():.3f}")
        else:
            raise TypeError(
                "LShapeElectricPotential requires target_demand to be a 2D routing demand tensor"
            )

    def set_raw_wire_demand_map(self, raw_wire_demand_map):
        if not isinstance(raw_wire_demand_map, torch.Tensor):
            raise TypeError(
                "LShapeElectricPotential requires raw_wire_demand_map to be a 2D routing demand tensor"
            )
        if raw_wire_demand_map.shape != (self.num_bins_x, self.num_bins_y):
            td = raw_wire_demand_map.unsqueeze(0).unsqueeze(0)
            td = F.interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            raw_wire_demand_map = td.squeeze(0).squeeze(0)
        if self.bin_center_x is not None:
            raw_wire_demand_map = raw_wire_demand_map.to(
                device=self.bin_center_x.device,
                dtype=self.bin_center_x.dtype,
            )
        self.raw_wire_demand_map = raw_wire_demand_map
        if isinstance(self.area_per_track, torch.Tensor):
            self.area_per_track.zero_()
        if self.log_verbose >= 2:
            logger.info(
                "Set raw_wire_demand_map from pure-wire routing demand map: min=%.3f, max=%.3f, mean=%.3f",
                float(raw_wire_demand_map.min().item()),
                float(raw_wire_demand_map.max().item()),
                float(raw_wire_demand_map.mean().item()),
            )

    def set_supply_original(self, supply_original):
        if not isinstance(supply_original, torch.Tensor):
            raise TypeError(
                "LShapeElectricPotential requires supply_original to be a 2D routing capacity tensor"
            )
        if supply_original.shape != (self.num_bins_x, self.num_bins_y):
            td = supply_original.unsqueeze(0).unsqueeze(0)
            td = F.interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            supply_original = td.squeeze(0).squeeze(0)
        if self.bin_center_x is not None:
            supply_original = supply_original.to(
                device=self.bin_center_x.device,
                dtype=self.bin_center_x.dtype,
            )
        self.supply_original = supply_original
        if self.log_verbose >= 2:
            logger.info(
                "Set supply_original from theoretical routing capacity map: min=%.3f, max=%.3f, mean=%.3f",
                float(supply_original.min().item()),
                float(supply_original.max().item()),
                float(supply_original.mean().item()),
            )

    def set_fix_usage_map(self, fix_usage_map):
        if not isinstance(fix_usage_map, torch.Tensor):
            raise TypeError(
                "LShapeElectricPotential requires fix_usage_map to be a 2D routing usage tensor"
            )
        if fix_usage_map.shape != (self.num_bins_x, self.num_bins_y):
            td = fix_usage_map.unsqueeze(0).unsqueeze(0)
            td = F.interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
            fix_usage_map = td.squeeze(0).squeeze(0)
        if self.bin_center_x is not None:
            fix_usage_map = fix_usage_map.to(
                device=self.bin_center_x.device,
                dtype=self.bin_center_x.dtype,
            )
        self.fix_usage_map = fix_usage_map
        if self.log_verbose >= 2:
            logger.info(
                "Set fix_usage_map from fixed routing usage map: min=%.3f, max=%.3f, mean=%.3f",
                float(fix_usage_map.min().item()),
                float(fix_usage_map.max().item()),
                float(fix_usage_map.mean().item()),
            )

    def set_directional_targets(
        self,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
        raw_wire_demand_map_h=None,
        raw_wire_demand_map_v=None,
        supply_original_h=None,
        supply_original_v=None,
        fix_usage_map_h=None,
        fix_usage_map_v=None,
    ):
        updates = {
            "target_density_h": target_density_h,
            "target_density_v": target_density_v,
            "target_demand_h": target_demand_h,
            "target_demand_v": target_demand_v,
            "raw_wire_demand_map_h": raw_wire_demand_map_h,
            "raw_wire_demand_map_v": raw_wire_demand_map_v,
            "supply_original_h": supply_original_h,
            "supply_original_v": supply_original_v,
            "fix_usage_map_h": fix_usage_map_h,
            "fix_usage_map_v": fix_usage_map_v,
        }
        for name, value in updates.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a 2D routing tensor")
            if value.shape != (self.num_bins_x, self.num_bins_y):
                td = value.unsqueeze(0).unsqueeze(0)
                td = F.interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                value = td.squeeze(0).squeeze(0)
            if self.bin_center_x is not None:
                value = value.to(
                    device=self.bin_center_x.device,
                    dtype=self.bin_center_x.dtype,
                )
            setattr(self, name, value)
    
    def _init_dct(self, device, dtype):
        """Initialize DCT related parameters."""
        M = self.num_bins_x
        N = self.num_bins_y
        
        # Precompute exp(-j*pi*k/M) and exp(-j*pi*k/N)
        self.exact_expkM = precompute_expk(M, dtype=dtype, device=device)
        self.exact_expkN = precompute_expk(N, dtype=dtype, device=device)
        
        # DCT operators
        self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
        if not self.fast_mode:
            self.idct2 = dct.IDCT2(self.exact_expkM, self.exact_expkN)
        self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM, self.exact_expkN)
        self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM, self.exact_expkN)
        
        # wu and wv (angular frequencies)
        wu = torch.arange(M, dtype=dtype, device=device).mul(2 * np.pi / M).view([M, 1])
        # Scale wv for non-square bins
        wv = torch.arange(N, dtype=dtype, device=device).mul(2 * np.pi / N).view([1, N])
        wv = wv.mul_(self.bin_size_x / self.bin_size_y)
        
        # Precompute inverse of (wu^2 + wv^2)
        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0, 0] = 1.0  # Avoid division by zero
        
        self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        self.inv_wu2_plus_wv2[0, 0] = 0.0  # DC component has zero potential
        
        self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(0.5)
        self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(0.5)
    
    def _prepare_segment_data(self, segment_pos, segment_size_x, segment_size_y, segment_weight=None):
        """Prepare segment data for density computation."""
        sqrt2 = math.sqrt(2)
        
        # Clamp sizes to minimum bin size
        # Using sqrt(2) ensures that the segment "splats" cover enough bins to form a smooth
        # continuous potential field, which is critical for the optimizer to spread cells.
        # Reducing this factor makes the force too local (sharp), causing cells to clump/gather
        # because they don't feel the repulsion until they are right on top of each other.
        # segment_size_x_clamped = segment_size_x.clamp(min=self.bin_size_x * sqrt2)
        # segment_size_y_clamped = segment_size_y.clamp(min=self.bin_size_y * sqrt2)
        segment_size_x_clamped = segment_size_x
        segment_size_y_clamped = segment_size_y
        
        # Compute offsets
        offset_x = (segment_size_x - segment_size_x_clamped).mul(0.5)
        offset_y = (segment_size_y - segment_size_y_clamped).mul(0.5)
        
        # Compute area ratio
        segment_area = segment_size_x * segment_size_y
        clamped_area = segment_size_x_clamped * segment_size_y_clamped
        ratio = segment_area / clamped_area.clamp(min=1e-10)
        if isinstance(segment_weight, torch.Tensor):
            ratio = ratio * segment_weight
        
        # Compute maximum impacted bins
        sqrt2_bin_x = sqrt2 * self.bin_size_x
        sqrt2_bin_y = sqrt2 * self.bin_size_y
        
        if segment_size_x.numel() > 0:
            num_impacted_bins_x = int(
                ((segment_size_x.max() + 2 * sqrt2_bin_x) / self.bin_size_x)
                .ceil().clamp(max=self.num_bins_x).item()
            )
            num_impacted_bins_y = int(
                ((segment_size_y.max() + 2 * sqrt2_bin_y) / self.bin_size_y)
                .ceil().clamp(max=self.num_bins_y).item()
            )
        else:
            num_impacted_bins_x = 0
            num_impacted_bins_y = 0
        
        # Sort segments by size
        if segment_size_x.numel() > 0:
            sorted_segment_map = torch.argsort(segment_size_x).to(torch.int32)
        else:
            sorted_segment_map = torch.tensor([], dtype=torch.int32, device=segment_pos.device)
        
        return (
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            num_impacted_bins_x,
            num_impacted_bins_y,
            sorted_segment_map
        )
    
    def forward(self, segment_pos, segment_size_x, segment_size_y, segment_is_horizontal=None, segment_weight=None):
        """
        Compute electric potential energy for routing segments.
        
        Args:
            segment_pos: [num_segments * 2] positions (llx then lly)
            segment_size_x: [num_segments] segment widths
            segment_size_y: [num_segments] segment heights
            segment_is_horizontal: [num_segments] bool tensor, optional
            
        Returns:
            energy: electric potential energy (scalar)
        """
        num_segments = segment_size_x.numel()
        
        if num_segments == 0:
            return torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device, requires_grad=True)
        
        # Initialize on first call
        if self.bin_center_x is None:
            self._init_bins(segment_pos.device, segment_pos.dtype)
        if self.dct2 is None:
            self._init_dct(segment_pos.device, segment_pos.dtype)
        
        # Prepare segment data
        (
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            num_impacted_bins_x,
            num_impacted_bins_y,
            sorted_segment_map
        ) = self._prepare_segment_data(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_weight=segment_weight,
        )

        if isinstance(segment_is_horizontal, torch.Tensor) and segment_is_horizontal.device != segment_pos.device:
            segment_is_horizontal = segment_is_horizontal.to(segment_pos.device)
        if isinstance(segment_weight, torch.Tensor) and segment_weight.device != segment_pos.device:
            segment_weight = segment_weight.to(segment_pos.device)
        
        # Compute electric potential
        SegmentElectricPotentialFunction.profile_enabled = self.profile_enabled
        SegmentElectricPotentialFunction.log_verbose = self.log_verbose
        energy = SegmentElectricPotentialFunction.apply(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_is_horizontal,
            segment_weight,
            (
                self.target_density_h,
                self.target_density_v,
                self.target_demand_h,
                self.target_demand_v,
                self.raw_wire_demand_map_h,
                self.raw_wire_demand_map_v,
                self.supply_original_h,
                self.supply_original_v,
                self.fix_usage_map_h,
                self.fix_usage_map_v,
            ),
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            self.bin_center_x,
            self.bin_center_y,
            self.initial_density_map,
            self.target_density,
            self.target_demand,
            self.raw_wire_demand_map,
            self.supply_original,
            self.fix_usage_map,
            self.area_per_track,
            self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y,
            num_segments,
            self.padding,
            self.padding_mask,
            self.num_bins_x, self.num_bins_y,
            num_impacted_bins_x,
            num_impacted_bins_y,
            self.deterministic_flag,
            sorted_segment_map,
            self.exact_expkM,
            self.exact_expkN,
            self.inv_wu2_plus_wv2,
            self.wu_by_wu2_plus_wv2_half,
            self.wv_by_wu2_plus_wv2_half,
            self.dct2,
            self.idct2,
            self.idct_idxst,
            self.idxst_idct,
            self.fast_mode
        )
        self.last_demand_supply_ratio = getattr(
            SegmentElectricPotentialFunction,
            "last_demand_supply_ratio",
            None,
        )
        
        return energy


class LShapeRoutabilityPotentialOp(nn.Module):
    """
    Complete L-shape routability operator using electric potential.
    
    This operator:
    1. Builds L-shape segments from Steiner tree edges
    2. Computes segment density using electric potential model
    3. Returns differentiable cost for optimization
    """
    
    def __init__(
        self,
        placedb,
        data_collections,
        steiner_topo_op,
        l_direction_resolver,
        num_bins_x=64,
        num_bins_y=64,
        target_density=None,
        target_demand=None,
        wire_width=0.0
    ):
        """
        Initialize routability operator.
        
        Args:
            placedb: placement database
            data_collections: data collections with pin/net info
            steiner_topo_op: Steiner topology operator
            l_direction_resolver: L-direction resolver
            num_bins_x, num_bins_y: number of bins
            target_density: 2D target routing supply tensor
        target_demand: optional 2D target routing demand tensor
            wire_width: wire width for segments
        """
        super(LShapeRoutabilityPotentialOp, self).__init__()
        
        self.placedb = placedb
        self.data_collections = data_collections
        self.steiner_topo_op = steiner_topo_op
        self.l_direction_resolver = l_direction_resolver
        self.wire_width = wire_width
        
        # Compute bin sizes
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        
        # Create electric potential operator
        self.electric_potential_op = LShapeElectricPotential(
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            target_density=target_density,
            target_demand=target_demand,
            padding=0,
            deterministic_flag=True,
            fast_mode=False
        )
        
        # Import segment builder
        from .l_shape_segment import LShapeSegmentOp
        self.segment_op = LShapeSegmentOp(wire_width=wire_width, use_vectorized=True)
        
        # Cached L-directions
        self.l_directions = None
    
    def resolve_l_directions(self):
        """Resolve L-directions for all edges."""
        if self.l_direction_resolver is not None:
            self.l_directions = self.l_direction_resolver.resolve_l_directions()
        else:
            # Default to V_FIRST
            num_edges = self.data_collections.flat_pin_from.numel()
            self.l_directions = torch.ones(
                num_edges,
                dtype=torch.int32,
                device=self.data_collections.flat_pin_from.device
            )
    
    def forward(self, pos):
        """
        Compute routability cost.
        
        Args:
            pos: cell positions [num_nodes * 2]
            
        Returns:
            cost: routability cost (scalar, differentiable)
        """
        # Resolve L-directions if not done
        if self.l_directions is None:
            self.resolve_l_directions()
        
        # Compute Steiner point coordinates
        newx, newy = self.steiner_topo_op(pos)
        
        # Build segments
        segment_result = self.segment_op(
            newx, newy,
            self.data_collections.flat_pin_from,
            self.data_collections.flat_pin_to,
            self.l_directions
        )
        
        if segment_result['num_segments'] == 0:
            return torch.zeros(1, dtype=pos.dtype, device=pos.device, requires_grad=True)
        
        # Compute electric potential cost
        segment_pos = segment_result['segment_pos']
        segment_size_x = segment_result['segment_size_x']
        segment_size_y = segment_result['segment_size_y']
        segment_is_horizontal = segment_result.get('segment_is_horizontal', None)
        
        cost = self.electric_potential_op(
            segment_pos, segment_size_x, segment_size_y, segment_is_horizontal
        )
        
        return cost


def create_l_shape_electric_potential(
    placedb,
    num_bins_x=64,
    num_bins_y=64,
    target_density=None,
    target_demand=None,
    raw_wire_demand_map=None,
    supply_original=None,
    target_density_h=None,
    target_density_v=None,
    target_demand_h=None,
    target_demand_v=None,
    raw_wire_demand_map_h=None,
    raw_wire_demand_map_v=None,
    supply_original_h=None,
    supply_original_v=None,
    fix_usage_map=None,
    fix_usage_map_h=None,
    fix_usage_map_v=None,
    padding=0,
    deterministic_flag=True,
    fast_mode=False,
    profile_enabled=False,
    log_verbose=0,
):
    """
    Factory function to create LShapeElectricPotential.
    
    Args:
        placedb: placement database
        num_bins_x, num_bins_y: number of bins
        target_density: 2D target routing supply tensor
        target_demand: optional 2D target routing demand tensor
        padding: bin padding
        deterministic_flag: whether to use deterministic routine
        fast_mode: if True, skip energy computation
        
    Returns:
        LShapeElectricPotential
    """
    bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
    bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
    
    return LShapeElectricPotential(
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        target_density=target_density,
        target_demand=target_demand,
        raw_wire_demand_map=raw_wire_demand_map,
        supply_original=supply_original,
        target_density_h=target_density_h,
        target_density_v=target_density_v,
        target_demand_h=target_demand_h,
        target_demand_v=target_demand_v,
        raw_wire_demand_map_h=raw_wire_demand_map_h,
        raw_wire_demand_map_v=raw_wire_demand_map_v,
        supply_original_h=supply_original_h,
        supply_original_v=supply_original_v,
        fix_usage_map=fix_usage_map,
        fix_usage_map_h=fix_usage_map_h,
        fix_usage_map_v=fix_usage_map_v,
        padding=padding,
        deterministic_flag=deterministic_flag,
        fast_mode=fast_mode,
        profile_enabled=profile_enabled,
        log_verbose=log_verbose,
    )
