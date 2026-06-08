##
# @file   l_shape_electric_potential.py
# @brief  Compute electric potential for L-shape routing segments
#         Uses C++/CUDA backends for density map and force computation
#         Gradients flow back through segment positions to cell positions
#

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
_MACRO_BODY_SOURCE_STRENGTH = 1.0
_MACRO_HALO_SOURCE_STRENGTH = 0.25
_MACRO_HALO_X = 0.0
_MACRO_HALO_Y = 0.0
_MACRO_USAGE_REFERENCE_QUANTILE = 0.95
_MACRO_SET_SOURCE = "fixed_macro_mask_heuristic"
_MACRO_COORDINATE_SYSTEM = "autodmp_scaled"
_BOUNDARY_SOURCE_REFERENCE_QUANTILE = 0.95


def _scalar_stat(tensor, op_name, default=0.0):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return default
    value = tensor.detach()
    if op_name == "sum":
        return float(value.sum().item())
    if op_name == "max":
        return float(value.max().item())
    if op_name == "mean":
        return float(value.mean().item())
    if op_name == "min":
        return float(value.min().item())
    raise ValueError("unsupported stat op %s" % op_name)


def _positive_count(tensor):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return 0
    return int((tensor.detach() > 0).sum().item())


def _finite_quantile(tensor, quantile, default=0.0):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return default
    values = tensor.detach().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return default
    return float(torch.quantile(values.to(dtype=torch.float32), float(quantile)).item())


def compute_macro_fixed_usage(
    macro_source_map,
    capacity_tracks,
    quantile=_MACRO_USAGE_REFERENCE_QUANTILE,
):
    if not torch.is_tensor(macro_source_map) or not torch.is_tensor(capacity_tracks):
        return None, 0.0
    if macro_source_map.shape != capacity_tracks.shape:
        raise RuntimeError(
            "macro source map shape %s does not match capacity map shape %s"
            % (tuple(macro_source_map.shape), tuple(capacity_tracks.shape))
        )
    capacity_local = capacity_tracks.clamp(min=0)
    macro_fraction = macro_source_map.to(
        device=capacity_tracks.device, dtype=capacity_tracks.dtype
    ).clamp(min=0, max=1)
    if not bool((macro_fraction > 0).any().item()):
        return torch.zeros_like(capacity_tracks), 0.0
    positive_capacity = capacity_local[capacity_local > 0]
    reference_p95 = _finite_quantile(
        positive_capacity,
        quantile,
        default=0.0,
    )
    if reference_p95 <= 0.0:
        return torch.zeros_like(capacity_tracks), reference_p95
    cap_value = torch.as_tensor(
        reference_p95,
        dtype=capacity_tracks.dtype,
        device=capacity_tracks.device,
    )
    usage_per_full_macro_bin = torch.minimum(capacity_local, cap_value)
    return macro_fraction * usage_per_full_macro_bin, reference_p95


def build_boundary_source_map(
    num_bins_x,
    num_bins_y,
    *,
    width_bins=0,
    strength=0.0,
    dtype=torch.float32,
):
    shape = (int(num_bins_x), int(num_bins_y))
    source_map = torch.zeros(shape, dtype=dtype)
    width = max(int(width_bins), 0)
    strength = max(float(strength), 0.0)
    enabled = width > 0 and strength > 0.0 and shape[0] > 0 and shape[1] > 0
    if enabled:
        width_x = min(width, shape[0])
        width_y = min(width, shape[1])
        source_map[:width_x, :] = strength
        source_map[shape[0] - width_x :, :] = strength
        source_map[:, :width_y] = strength
        source_map[:, shape[1] - width_y :] = strength
    stats = {
        "boundary_source_enabled": bool(enabled),
        "boundary_source_width_bins": width,
        "boundary_source_strength": strength,
        "boundary_source_active_bins": _positive_count(source_map),
        "boundary_source_max": _scalar_stat(source_map, "max"),
        "boundary_source_sum": _scalar_stat(source_map, "sum"),
        "boundary_source_grid_shape": shape,
    }
    return source_map, stats


def compute_boundary_fixed_usage(
    boundary_source_map,
    capacity_tracks,
    quantile=_BOUNDARY_SOURCE_REFERENCE_QUANTILE,
):
    if not torch.is_tensor(boundary_source_map) or not torch.is_tensor(capacity_tracks):
        return None, 0.0
    if boundary_source_map.shape != capacity_tracks.shape:
        raise RuntimeError(
            "boundary source map shape %s does not match capacity map shape %s"
            % (tuple(boundary_source_map.shape), tuple(capacity_tracks.shape))
        )
    capacity_local = capacity_tracks.clamp(min=0)
    boundary_fraction = boundary_source_map.to(
        device=capacity_tracks.device, dtype=capacity_tracks.dtype
    ).clamp(min=0, max=1)
    if not bool((boundary_fraction > 0).any().item()):
        return torch.zeros_like(capacity_tracks), 0.0
    positive_capacity = capacity_local[capacity_local > 0]
    reference_p95 = _finite_quantile(
        positive_capacity,
        quantile,
        default=0.0,
    )
    if reference_p95 <= 0.0:
        return torch.zeros_like(capacity_tracks), reference_p95
    cap_value = torch.as_tensor(
        reference_p95,
        dtype=capacity_tracks.dtype,
        device=capacity_tracks.device,
    )
    usage_per_full_boundary_bin = torch.minimum(capacity_local, cap_value)
    return boundary_fraction * usage_per_full_boundary_bin, reference_p95


def compute_track_rho_components(
    density_seg_area,
    supply_original,
    fix_usage,
    bin_area,
    macro_source_map=None,
    boundary_source_map=None,
):
    density_seg_tracks = density_seg_area / bin_area
    base_fixed_usage = fix_usage.clamp(min=0)
    capacity_tracks = supply_original
    macro_usage, macro_usage_reference_p95 = compute_macro_fixed_usage(
        macro_source_map,
        capacity_tracks,
    )
    if macro_usage is None:
        macro_usage = torch.zeros_like(base_fixed_usage)
        macro_usage_reference_p95 = 0.0
    boundary_usage, boundary_usage_reference_p95 = compute_boundary_fixed_usage(
        boundary_source_map,
        capacity_tracks,
    )
    if boundary_usage is None:
        boundary_usage = torch.zeros_like(base_fixed_usage)
        boundary_usage_reference_p95 = 0.0
    initial_density_tracks = base_fixed_usage + macro_usage + boundary_usage
    occupancy_tracks = initial_density_tracks + density_seg_tracks
    residual_tracks = (
        occupancy_tracks - capacity_tracks
    ) / capacity_tracks.clamp(min=1e-6)
    residual_centered_tracks = residual_tracks - residual_tracks.mean()
    overflow_map = (occupancy_tracks - capacity_tracks).clamp(min=0)
    utilization = occupancy_tracks / capacity_tracks.clamp(min=1e-6)
    return {
        "rho_map": residual_tracks,
        "residual_tracks": residual_tracks,
        "overflow_map": overflow_map,
        "utilization": utilization,
        "initial_density_tracks": initial_density_tracks,
        "density_seg_tracks": density_seg_tracks,
        "occupancy_tracks": occupancy_tracks,
        "capacity_tracks": capacity_tracks,
        "residual_centered_tracks": residual_centered_tracks,
        "base_fixed_usage": base_fixed_usage,
        "macro_usage": macro_usage,
        "macro_usage_reference_p95": float(macro_usage_reference_p95),
        "boundary_usage": boundary_usage,
        "boundary_usage_reference_p95": float(boundary_usage_reference_p95),
    }


def _tensor_identity(value):
    if not torch.is_tensor(value):
        return None
    return (
        tuple(int(dim) for dim in value.shape),
        str(value.device),
        str(value.dtype),
        int(value.data_ptr()),
        int(getattr(value, "_version", 0)),
    )


def _to_numpy_1d(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _placedb_scalar_array_value(value, index):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[int(index)].item())
    return float(np.asarray(value).reshape(-1)[int(index)])


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


def _fixed_macro_indices_from_placedb(placedb):
    macro_idx = _to_numpy_1d(getattr(placedb, "fixed_macro_idx", None))
    if macro_idx is None:
        raise RuntimeError(
            "fixed macro exclusion requires placedb.fixed_macro_idx from the fixed macro heuristic"
        )
    macro_idx = macro_idx.astype(np.int64, copy=False)

    fixed_mask = _to_numpy_1d(getattr(placedb, "fixed_macro_mask", None))
    fixed_slice = getattr(placedb, "fixed_slice", None)
    if fixed_mask is None or fixed_slice is None:
        raise RuntimeError(
            "fixed macro exclusion requires placedb.fixed_macro_mask and placedb.fixed_slice"
        )
    expected_idx = (
        int(fixed_slice.start)
        + np.nonzero(fixed_mask.astype(bool, copy=False))[0].astype(np.int64)
    )
    if macro_idx.shape != expected_idx.shape or not np.array_equal(macro_idx, expected_idx):
        raise RuntimeError(
            "placedb.fixed_macro_idx must match fixed_macro_mask indexed relative to fixed_slice"
        )
    return macro_idx


def build_fixed_macro_source_maps(
    placedb,
    xl,
    yl,
    bin_size_x,
    bin_size_y,
    num_bins_x,
    num_bins_y,
    *,
    body_strength=_MACRO_BODY_SOURCE_STRENGTH,
    halo_strength=_MACRO_HALO_SOURCE_STRENGTH,
    halo_x=_MACRO_HALO_X,
    halo_y=_MACRO_HALO_Y,
    dtype=torch.float32,
):
    macro_idx = _fixed_macro_indices_from_placedb(placedb)
    shape = (int(num_bins_x), int(num_bins_y))
    body_map = torch.zeros(shape, dtype=dtype)
    halo_map = torch.zeros(shape, dtype=dtype)
    bin_area = max(float(bin_size_x) * float(bin_size_y), 1.0e-30)

    node_x = getattr(placedb, "node_x")
    node_y = getattr(placedb, "node_y")
    node_size_x = getattr(placedb, "node_size_x")
    node_size_y = getattr(placedb, "node_size_y")

    def apply_box(box, target):
        x_l, y_l, x_h, y_h = box
        if x_h <= x_l or y_h <= y_l:
            return
        ix0 = max(int(math.floor((x_l - xl) / bin_size_x)), 0)
        ix1 = min(int(math.ceil((x_h - xl) / bin_size_x)), num_bins_x)
        iy0 = max(int(math.floor((y_l - yl) / bin_size_y)), 0)
        iy1 = min(int(math.ceil((y_h - yl) / bin_size_y)), num_bins_y)
        for ix in range(ix0, ix1):
            bin_x_l = xl + ix * bin_size_x
            bin_x_h = bin_x_l + bin_size_x
            overlap_x = max(0.0, min(x_h, bin_x_h) - max(x_l, bin_x_l))
            if overlap_x <= 0.0:
                continue
            for iy in range(iy0, iy1):
                bin_y_l = yl + iy * bin_size_y
                bin_y_h = bin_y_l + bin_size_y
                overlap_y = max(0.0, min(y_h, bin_y_h) - max(y_l, bin_y_l))
                if overlap_y <= 0.0:
                    continue
                fraction = min(max((overlap_x * overlap_y) / bin_area, 0.0), 1.0)
                if fraction > float(target[ix, iy].item()):
                    target[ix, iy] = fraction

    for macro_id in macro_idx:
        macro_id = int(macro_id)
        x_l = _placedb_scalar_array_value(node_x, macro_id)
        y_l = _placedb_scalar_array_value(node_y, macro_id)
        x_h = x_l + _placedb_scalar_array_value(node_size_x, macro_id)
        y_h = y_l + _placedb_scalar_array_value(node_size_y, macro_id)
        apply_box((x_l, y_l, x_h, y_h), body_map)
        if halo_x > 0.0 or halo_y > 0.0:
            halo_box = (
                x_l - float(halo_x),
                y_l - float(halo_y),
                x_h + float(halo_x),
                y_h + float(halo_y),
            )
            apply_box(halo_box, halo_map)

    body_mask = body_map.clone()
    halo_map = (halo_map - body_mask).clamp(min=0.0)
    body_map.mul_(float(body_strength))
    halo_map.mul_(float(halo_strength))
    source_map = torch.maximum(body_map, halo_map)
    stats = {
        "macro_exclusion_enabled": True,
        "macro_count": int(macro_idx.shape[0]),
        "macro_body_bins": _positive_count(body_map),
        "macro_halo_bins": _positive_count(halo_map),
        "macro_body_source_max": _scalar_stat(body_map, "max"),
        "macro_body_source_sum": _scalar_stat(body_map, "sum"),
        "macro_halo_source_max": _scalar_stat(halo_map, "max"),
        "macro_halo_source_sum": _scalar_stat(halo_map, "sum"),
        "macro_source_max": _scalar_stat(source_map, "max"),
        "macro_source_sum": _scalar_stat(source_map, "sum"),
        "macro_source_active_bins": _positive_count(source_map),
        "macro_source_grid_shape": shape,
        "macro_source_coordinate_system": _MACRO_COORDINATE_SYSTEM,
        "macro_set_source": _MACRO_SET_SOURCE,
    }
    return body_map, halo_map, source_map, stats


def _tensor_like(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _empty_fixed_macro_overlap_stats(placedb, macro_count, movable_count):
    return {
        "fixed_macro_overlap_enabled": True,
        "fixed_macro_overlap_macro_count": int(macro_count),
        "fixed_macro_overlap_movable_count": int(movable_count),
        "fixed_macro_overlap_area": 0.0,
        "fixed_macro_overlap_area_ratio": 0.0,
        "fixed_macro_overlap_cell_count": 0,
        "fixed_macro_overlap_pair_count": 0,
        "fixed_macro_overlap_max_area": 0.0,
        "fixed_macro_overlap_mean_area": 0.0,
        "fixed_macro_overlap_coordinate_system": _MACRO_COORDINATE_SYSTEM,
        "fixed_macro_overlap_macro_set_source": _MACRO_SET_SOURCE,
    }


def compute_fixed_macro_overlap_stats(pos, node_size_x, node_size_y, placedb):
    macro_idx = _fixed_macro_indices_from_placedb(placedb)
    movable_count = int(getattr(placedb, "num_movable_nodes", 0))
    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos, dtype=torch.float32)
    num_pos_nodes = int(pos.numel() // 2)
    movable_count = min(movable_count, num_pos_nodes)
    if movable_count <= 0 or macro_idx.shape[0] == 0:
        return _empty_fixed_macro_overlap_stats(
            placedb, macro_idx.shape[0], movable_count
        )

    sizes_x = _tensor_like(node_size_x, pos).reshape(-1)
    sizes_y = _tensor_like(node_size_y, pos).reshape(-1)
    movable_count = min(movable_count, int(sizes_x.numel()), int(sizes_y.numel()))
    if movable_count <= 0:
        return _empty_fixed_macro_overlap_stats(
            placedb, macro_idx.shape[0], movable_count
        )

    movable_x_l = pos[:movable_count].reshape(-1)
    movable_y_l = pos[num_pos_nodes : num_pos_nodes + movable_count].reshape(-1)
    movable_x_h = movable_x_l + sizes_x[:movable_count]
    movable_y_h = movable_y_l + sizes_y[:movable_count]
    movable_area = (sizes_x[:movable_count] * sizes_y[:movable_count]).clamp(
        min=0.0
    ).sum()

    per_cell_overlap = torch.zeros_like(movable_x_l)
    total_overlap = torch.zeros((), device=pos.device, dtype=pos.dtype)
    max_pair_overlap = torch.zeros((), device=pos.device, dtype=pos.dtype)
    pair_count = 0

    node_x = getattr(placedb, "node_x")
    node_y = getattr(placedb, "node_y")
    macro_size_x = getattr(placedb, "node_size_x")
    macro_size_y = getattr(placedb, "node_size_y")
    for macro_id in macro_idx:
        macro_id = int(macro_id)
        macro_x_l = _placedb_scalar_array_value(node_x, macro_id)
        macro_y_l = _placedb_scalar_array_value(node_y, macro_id)
        macro_x_h = macro_x_l + _placedb_scalar_array_value(macro_size_x, macro_id)
        macro_y_h = macro_y_l + _placedb_scalar_array_value(macro_size_y, macro_id)
        overlap_x = (
            torch.minimum(movable_x_h, torch.tensor(macro_x_h, device=pos.device, dtype=pos.dtype))
            - torch.maximum(movable_x_l, torch.tensor(macro_x_l, device=pos.device, dtype=pos.dtype))
        ).clamp(min=0.0)
        overlap_y = (
            torch.minimum(movable_y_h, torch.tensor(macro_y_h, device=pos.device, dtype=pos.dtype))
            - torch.maximum(movable_y_l, torch.tensor(macro_y_l, device=pos.device, dtype=pos.dtype))
        ).clamp(min=0.0)
        overlap = overlap_x * overlap_y
        positive = overlap > 0
        if bool(positive.any().item()):
            pair_count += int(positive.sum().item())
            total_overlap = total_overlap + overlap.sum()
            max_pair_overlap = torch.maximum(max_pair_overlap, overlap.max())
            per_cell_overlap = per_cell_overlap + overlap

    overlap_cells = int((per_cell_overlap > 0).sum().item())
    overlap_area = float(total_overlap.detach().cpu().item())
    movable_area_value = float(movable_area.detach().cpu().item())
    mean_area = overlap_area / overlap_cells if overlap_cells > 0 else 0.0
    area_ratio = (
        overlap_area / movable_area_value if movable_area_value > 0.0 else 0.0
    )
    return {
        "fixed_macro_overlap_enabled": True,
        "fixed_macro_overlap_macro_count": int(macro_idx.shape[0]),
        "fixed_macro_overlap_movable_count": int(movable_count),
        "fixed_macro_overlap_area": overlap_area,
        "fixed_macro_overlap_area_ratio": area_ratio,
        "fixed_macro_overlap_cell_count": overlap_cells,
        "fixed_macro_overlap_pair_count": int(pair_count),
        "fixed_macro_overlap_max_area": float(max_pair_overlap.detach().cpu().item()),
        "fixed_macro_overlap_mean_area": mean_area,
        "fixed_macro_overlap_coordinate_system": _MACRO_COORDINATE_SYSTEM,
        "fixed_macro_overlap_macro_set_source": _MACRO_SET_SOURCE,
    }


def compute_movable_displacement_stats(pos_before, pos_after, placedb):
    if not torch.is_tensor(pos_before):
        pos_before = torch.as_tensor(pos_before, dtype=torch.float32)
    pos_after = _tensor_like(pos_after, pos_before).reshape(-1)
    pos_before = pos_before.reshape(-1)
    num_pos_nodes = int(min(pos_before.numel(), pos_after.numel()) // 2)
    movable_count = min(int(getattr(placedb, "num_movable_nodes", 0)), num_pos_nodes)
    if movable_count <= 0:
        return {
            "movable_displacement_movable_count": 0,
            "movable_displacement_moved_count": 0,
            "movable_displacement_max": 0.0,
            "movable_displacement_mean": 0.0,
            "movable_displacement_sum": 0.0,
            "movable_displacement_rms": 0.0,
        }

    dx = pos_after[:movable_count] - pos_before[:movable_count]
    dy = (
        pos_after[num_pos_nodes : num_pos_nodes + movable_count]
        - pos_before[num_pos_nodes : num_pos_nodes + movable_count]
    )
    distance = torch.sqrt(dx * dx + dy * dy)
    return {
        "movable_displacement_movable_count": int(movable_count),
        "movable_displacement_moved_count": int((distance > 1.0e-6).sum().item()),
        "movable_displacement_max": float(distance.max().detach().cpu().item()),
        "movable_displacement_mean": float(distance.mean().detach().cpu().item()),
        "movable_displacement_sum": float(distance.sum().detach().cpu().item()),
        "movable_displacement_rms": float(
            torch.sqrt((distance * distance).mean()).detach().cpu().item()
        ),
    }


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
        fast_mode,
        capacity_al_owner,
        update_capacity_al_lambda,
        placement_iteration_id,
        capacity_al_reset_key,
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
            macro_source_local = None
            if capacity_al_owner is not None and hasattr(capacity_al_owner, "_macro_source_for"):
                macro_source_local = capacity_al_owner._macro_source_for(supply_original_local)
            boundary_source_local = None
            if capacity_al_owner is not None and hasattr(capacity_al_owner, "_boundary_source_for"):
                boundary_source_local = capacity_al_owner._boundary_source_for(supply_original_local)
            components = compute_track_rho_components(
                density_seg_area_local,
                supply_original_local,
                fix_usage_local,
                bin_area,
                macro_source_map=macro_source_local,
                boundary_source_map=boundary_source_local,
            )
            return (
                components["rho_map"],
                components["overflow_map"],
                components["utilization"],
                components["initial_density_tracks"],
                components["density_seg_tracks"],
                components["occupancy_tracks"],
                components["capacity_tracks"],
                components["residual_centered_tracks"],
                components["base_fixed_usage"],
                components["macro_usage"],
                components["macro_usage_reference_p95"],
                components["boundary_usage"],
                components["boundary_usage_reference_p95"],
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
                potential_map_local = torch.zeros_like(rho_map_local)
            else:
                potential_map_local = idct2.forward(auv_local.mul(inv_wu2_plus_wv2))
                potential_map_local.mul_(bin_area)
                energy_local = potential_map_local.mul(rho_map_normalized_local).sum()
            return (
                rho_map_normalized_local,
                field_map_x_local,
                field_map_y_local,
                potential_map_local,
                energy_local,
            )

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
        
        # Compute a capacity-residual source from routing demand vs supply.
        # Overflow remains as absolute-track telemetry, while the Poisson
        # solver sees positive normalized capacity violation only.
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
                residual_centered_tracks_h,
                base_fixed_usage_h,
                macro_usage_h,
                macro_usage_reference_p95_h,
                boundary_usage_h,
                boundary_usage_reference_p95_h,
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
                residual_centered_tracks_v,
                base_fixed_usage_v,
                macro_usage_v,
                macro_usage_reference_p95_v,
                boundary_usage_v,
                boundary_usage_reference_p95_v,
            ) = _compute_blockage_track_rho_components(
                density_map_v,
                supply_original_v,
                fix_usage_map_v,
            )
            if getattr(capacity_al_owner, "capacity_al_enable", False):
                rho_map_h, rho_map_v = capacity_al_owner._capacity_al_source_maps(
                    g_h=rho_map_h,
                    g_v=rho_map_v,
                    update_lambda=bool(update_capacity_al_lambda),
                    placement_iteration_id=placement_iteration_id,
                    reset_key=capacity_al_reset_key,
                )
            elif capacity_al_owner is not None:
                capacity_al_owner.last_al_stats = {"enabled": False}
            rho_map_h = rho_map_h.clamp(min=0)
            rho_map_v = rho_map_v.clamp(min=0)
            if capacity_al_owner is not None and hasattr(
                capacity_al_owner, "_record_macro_occupancy_stats"
            ):
                capacity_al_owner._record_macro_occupancy_stats(
                    source_name="h,v",
                    base_fixed_usage_h=base_fixed_usage_h,
                    base_fixed_usage_v=base_fixed_usage_v,
                    macro_usage_h=macro_usage_h,
                    macro_usage_v=macro_usage_v,
                    boundary_usage_h=boundary_usage_h,
                    boundary_usage_v=boundary_usage_v,
                    initial_density_tracks_h=initial_density_tracks_h,
                    initial_density_tracks_v=initial_density_tracks_v,
                    occupancy_tracks_h=occupancy_tracks_h,
                    occupancy_tracks_v=occupancy_tracks_v,
                    residual_h=rho_map_h,
                    residual_v=rho_map_v,
                    macro_usage_reference_p95_h=macro_usage_reference_p95_h,
                    macro_usage_reference_p95_v=macro_usage_reference_p95_v,
                    boundary_usage_reference_p95_h=boundary_usage_reference_p95_h,
                    boundary_usage_reference_p95_v=boundary_usage_reference_p95_v,
                )

            with profile_scope(
                profile_enabled,
                "electric.forward.field_energy",
                tensor=segment_pos,
                logger=logger,
                mode="hv_split",
            ):
                _, field_map_x_h, field_map_y_h, potential_map_h, energy_h = _compute_field_and_energy(rho_map_h)
                _, field_map_x_v, field_map_y_v, potential_map_v, energy_v = _compute_field_and_energy(rho_map_v)

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
            if getattr(capacity_al_owner, "capacity_al_enable", False):
                capacity_al_owner.last_al_stats.update(
                    {
                        "E_cap_smooth_h": float(energy_h.detach().item()),
                        "E_cap_smooth_v": float(energy_v.detach().item()),
                        "E_cap_smooth_total": float(energy.detach().item()),
                        "Pq_h_min": _scalar_stat(potential_map_h, "min"),
                        "Pq_h_max": _scalar_stat(potential_map_h, "max"),
                        "Pq_h_sum": _scalar_stat(potential_map_h, "sum"),
                        "Pq_v_min": _scalar_stat(potential_map_v, "min"),
                        "Pq_v_max": _scalar_stat(potential_map_v, "max"),
                        "Pq_v_sum": _scalar_stat(potential_map_v, "sum"),
                    }
                )
                if log_verbose >= 2:
                    logger.info(
                        "L-shape capacity AL: iter=%s update=%s reset=%s "
                        "g_h_max=%.4e g_v_max=%.4e g_h_sum=%.4e g_v_sum=%.4e "
                        "g_h_pos_bins=%d/%d g_v_pos_bins=%d/%d "
                        "g_h_pos_ratio=%.4f g_v_pos_ratio=%.4f "
                        "q_h_max=%.4e q_v_max=%.4e "
                        "lambda_h_max=%.4e lambda_v_max=%.4e E_h=%.4e E_v=%.4e "
                        "Pq_h_min=%.4e Pq_v_min=%.4e "
                        "active_memory_bins_h=%d active_memory_bins_v=%d "
                        "(negative Pq values can occur from the DCT/Poisson gauge)",
                        str(placement_iteration_id),
                        str(capacity_al_owner.last_al_stats.get("updated")),
                        str(capacity_al_owner.last_al_stats.get("reset_reason")),
                        float(capacity_al_owner.last_al_stats.get("g_h_max", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("g_v_max", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("g_h_sum", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("g_v_sum", 0.0)),
                        int(capacity_al_owner.last_al_stats.get("g_h_pos_bins", 0)),
                        int(capacity_al_owner.last_al_stats.get("g_h_bins", 0)),
                        int(capacity_al_owner.last_al_stats.get("g_v_pos_bins", 0)),
                        int(capacity_al_owner.last_al_stats.get("g_v_bins", 0)),
                        float(capacity_al_owner.last_al_stats.get("g_h_pos_ratio", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("g_v_pos_ratio", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("q_h_max", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("q_v_max", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("lambda_h_max", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("lambda_v_max", 0.0)),
                        float(energy_h.detach().item()),
                        float(energy_v.detach().item()),
                        float(capacity_al_owner.last_al_stats.get("Pq_h_min", 0.0)),
                        float(capacity_al_owner.last_al_stats.get("Pq_v_min", 0.0)),
                        int(capacity_al_owner.last_al_stats.get("active_memory_bins_h", 0)),
                        int(capacity_al_owner.last_al_stats.get("active_memory_bins_v", 0)),
                    )

            logger.debug(
                "Blockage residual-source(split,track): occ_ratio_h=%.4f occ_ratio_v=%.4f util_h_max=%.2f util_v_max=%.2f "
                "residual_center_pos_bins_h=%d/%d residual_center_neg_bins_h=%d/%d overflow_bins_h=%d/%d "
                "residual_center_pos_bins_v=%d/%d residual_center_neg_bins_v=%d/%d overflow_bins_v=%d/%d",
                float(occupancy_tracks_h.sum().item() / capacity_tracks_h.sum().clamp(min=1e-6).item()) if capacity_tracks_h.numel() > 0 else 0.0,
                float(occupancy_tracks_v.sum().item() / capacity_tracks_v.sum().clamp(min=1e-6).item()) if capacity_tracks_v.numel() > 0 else 0.0,
                float(utilization_h.max().item()) if utilization_h.numel() > 0 else 0.0,
                float(utilization_v.max().item()) if utilization_v.numel() > 0 else 0.0,
                int((residual_centered_tracks_h > 0).sum().item()),
                int(residual_centered_tracks_h.numel()),
                int((residual_centered_tracks_h < 0).sum().item()),
                int(residual_centered_tracks_h.numel()),
                int((overflow_map_h > 0).sum().item()),
                int(overflow_map_h.numel()),
                int((residual_centered_tracks_v > 0).sum().item()),
                int(residual_centered_tracks_v.numel()),
                int((residual_centered_tracks_v < 0).sum().item()),
                int(residual_centered_tracks_v.numel()),
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
                residual_centered_tracks,
                base_fixed_usage,
                macro_usage,
                macro_usage_reference_p95,
                boundary_usage,
                boundary_usage_reference_p95,
            ) = _compute_blockage_track_rho_components(
                density_map,
                supply_original_map,
                planar_fix_usage_map,
            )
            if getattr(capacity_al_owner, "capacity_al_enable", False):
                raise RuntimeError(
                    "l_shape_capacity_al_enable requires H/V capacity and fixed-usage maps"
                )
            elif capacity_al_owner is not None:
                capacity_al_owner.last_al_stats = {"enabled": False}
            rho_map = rho_map.clamp(min=0)
            if capacity_al_owner is not None and hasattr(
                capacity_al_owner, "_record_macro_occupancy_stats"
            ):
                capacity_al_owner._record_macro_occupancy_stats(
                    source_name="planar",
                    base_fixed_usage_h=base_fixed_usage,
                    base_fixed_usage_v=None,
                    macro_usage_h=macro_usage,
                    macro_usage_v=None,
                    boundary_usage_h=boundary_usage,
                    boundary_usage_v=None,
                    initial_density_tracks_h=initial_density_tracks,
                    initial_density_tracks_v=None,
                    occupancy_tracks_h=occupancy_tracks,
                    occupancy_tracks_v=None,
                    residual_h=rho_map,
                    residual_v=None,
                    macro_usage_reference_p95_h=macro_usage_reference_p95,
                    macro_usage_reference_p95_v=0.0,
                    boundary_usage_reference_p95_h=boundary_usage_reference_p95,
                    boundary_usage_reference_p95_v=0.0,
                )
            with profile_scope(
                profile_enabled,
                "electric.forward.field_energy",
                tensor=segment_pos,
                logger=logger,
                mode="planar",
            ):
                rho_map_normalized, field_map_x, field_map_y, potential_map, energy = _compute_field_and_energy(rho_map)
            ctx.field_map_x = field_map_x
            ctx.field_map_y = field_map_y
            SegmentElectricPotentialFunction.last_rho_map = rho_map.detach()
            SegmentElectricPotentialFunction.last_energy = energy.detach()

            logger.debug(
                "Blockage residual-source(planar,track): occ_ratio=%.4f util_mean=%.2f util_max=%.2f "
                "residual_center_pos_bins=%d/%d residual_center_neg_bins=%d/%d overflow_bins=%d/%d",
                float(occupancy_tracks.sum().item() / capacity_tracks.sum().clamp(min=1e-6).item()) if capacity_tracks.numel() > 0 else 0.0,
                float(utilization.mean().item()) if utilization.numel() > 0 else 0.0,
                float(utilization.max().item()) if utilization.numel() > 0 else 0.0,
                int((residual_centered_tracks > 0).sum().item()),
                int(residual_centered_tracks.numel()),
                int((residual_centered_tracks < 0).sum().item()),
                int(residual_centered_tracks.numel()),
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
        return (output,) + (None,) * 48


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
        capacity_al_enable=False,
        placedb=None,
        boundary_source_enable=False,
        boundary_source_width_bins=0,
        boundary_source_strength=0.0,
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
        self.capacity_al_enable = bool(capacity_al_enable)
        self._capacity_al_rho = 1.0
        self._capacity_al_lambda_max = 1.0
        self._capacity_al_positive_step = 0.10
        self._capacity_al_negative_step = 0.50
        self.lambda_h = None
        self.lambda_v = None
        self._capacity_al_reset_key = None
        self._capacity_al_last_update_iter = None
        self._capacity_al_last_reset_reason = None
        self._capacity_al_reset_count = 0
        self._capacity_al_update_count = 0
        self.last_al_stats = {}
        self.last_macro_exclusion_stats = {}
        self.last_boundary_source_stats = {}
        
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
        if placedb is None:
            macro_body_source_map = torch.zeros(
                self.num_bins_x, self.num_bins_y, dtype=torch.float32
            )
            macro_halo_source_map = torch.zeros_like(macro_body_source_map)
            macro_source_map = torch.zeros_like(macro_body_source_map)
            self.macro_exclusion_stats = {
                "macro_exclusion_enabled": False,
                "macro_count": 0,
                "macro_body_bins": 0,
                "macro_halo_bins": 0,
                "macro_body_source_max": 0.0,
                "macro_body_source_sum": 0.0,
                "macro_halo_source_max": 0.0,
                "macro_halo_source_sum": 0.0,
                "macro_source_max": 0.0,
                "macro_source_sum": 0.0,
                "macro_source_active_bins": 0,
                "macro_source_grid_shape": (self.num_bins_x, self.num_bins_y),
                "macro_source_coordinate_system": _MACRO_COORDINATE_SYSTEM,
                "macro_set_source": _MACRO_SET_SOURCE,
            }
        else:
            (
                macro_body_source_map,
                macro_halo_source_map,
                macro_source_map,
                self.macro_exclusion_stats,
            ) = build_fixed_macro_source_maps(
                placedb,
                self.xl,
                self.yl,
                self.bin_size_x,
                self.bin_size_y,
                self.num_bins_x,
                self.num_bins_y,
            )
        self.register_buffer('macro_body_source_map', macro_body_source_map)
        self.register_buffer('macro_halo_source_map', macro_halo_source_map)
        self.register_buffer('macro_source_map', macro_source_map)
        self.last_macro_exclusion_stats = dict(self.macro_exclusion_stats)
        if bool(boundary_source_enable):
            boundary_source_map, self.boundary_source_stats = build_boundary_source_map(
                self.num_bins_x,
                self.num_bins_y,
                width_bins=boundary_source_width_bins,
                strength=boundary_source_strength,
            )
        else:
            boundary_source_map, self.boundary_source_stats = build_boundary_source_map(
                self.num_bins_x,
                self.num_bins_y,
            )
        self.register_buffer('boundary_source_map', boundary_source_map)
        self.last_boundary_source_stats = dict(self.boundary_source_stats)
        logger.info(
            "L-shape macro exclusion: macro_exclusion_enabled=%s macro_count=%d "
            "macro_body_bins=%d macro_halo_bins=%d macro_source_active_bins=%d "
            "macro_body_source_max=%.4e macro_body_source_sum=%.4e "
            "macro_halo_source_max=%.4e macro_halo_source_sum=%.4e "
            "macro_source_max=%.4e macro_source_sum=%.4e "
            "macro_source_grid_shape=%s macro_source_coordinate_system=%s "
            "macro_set_source=%s",
            str(self.macro_exclusion_stats.get("macro_exclusion_enabled")),
            int(self.macro_exclusion_stats.get("macro_count", 0)),
            int(self.macro_exclusion_stats.get("macro_body_bins", 0)),
            int(self.macro_exclusion_stats.get("macro_halo_bins", 0)),
            int(self.macro_exclusion_stats.get("macro_source_active_bins", 0)),
            float(self.macro_exclusion_stats.get("macro_body_source_max", 0.0)),
            float(self.macro_exclusion_stats.get("macro_body_source_sum", 0.0)),
            float(self.macro_exclusion_stats.get("macro_halo_source_max", 0.0)),
            float(self.macro_exclusion_stats.get("macro_halo_source_sum", 0.0)),
            float(self.macro_exclusion_stats.get("macro_source_max", 0.0)),
            float(self.macro_exclusion_stats.get("macro_source_sum", 0.0)),
            str(self.macro_exclusion_stats.get("macro_source_grid_shape")),
            str(self.macro_exclusion_stats.get("macro_source_coordinate_system")),
            str(self.macro_exclusion_stats.get("macro_set_source")),
        )
        if self.log_verbose >= 1 or self.boundary_source_stats.get("boundary_source_enabled"):
            logger.info(
                "L-shape boundary source: enabled=%s width_bins=%d strength=%.4e "
                "active_bins=%d source_max=%.4e source_sum=%.4e grid_shape=%s",
                str(self.boundary_source_stats.get("boundary_source_enabled")),
                int(self.boundary_source_stats.get("boundary_source_width_bins", 0)),
                float(self.boundary_source_stats.get("boundary_source_strength", 0.0)),
                int(self.boundary_source_stats.get("boundary_source_active_bins", 0)),
                float(self.boundary_source_stats.get("boundary_source_max", 0.0)),
                float(self.boundary_source_stats.get("boundary_source_sum", 0.0)),
                str(self.boundary_source_stats.get("boundary_source_grid_shape")),
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

    def _macro_source_for(self, reference):
        if not isinstance(self.macro_source_map, torch.Tensor):
            return None
        if self.macro_source_map.shape != reference.shape:
            raise RuntimeError(
                "macro source map shape %s does not match routing source shape %s"
                % (tuple(self.macro_source_map.shape), tuple(reference.shape))
            )
        return self.macro_source_map.to(device=reference.device, dtype=reference.dtype)

    def _boundary_source_for(self, reference):
        if not isinstance(self.boundary_source_map, torch.Tensor):
            return None
        if self.boundary_source_map.shape != reference.shape:
            raise RuntimeError(
                "boundary source map shape %s does not match routing source shape %s"
                % (tuple(self.boundary_source_map.shape), tuple(reference.shape))
            )
        return self.boundary_source_map.to(device=reference.device, dtype=reference.dtype)

    def _record_macro_occupancy_stats(
        self,
        *,
        source_name,
        base_fixed_usage_h,
        base_fixed_usage_v,
        macro_usage_h,
        macro_usage_v,
        boundary_usage_h,
        boundary_usage_v,
        initial_density_tracks_h,
        initial_density_tracks_v,
        occupancy_tracks_h,
        occupancy_tracks_v,
        residual_h,
        residual_v,
        macro_usage_reference_p95_h,
        macro_usage_reference_p95_v,
        boundary_usage_reference_p95_h,
        boundary_usage_reference_p95_v,
    ):
        def _sum_optional(*values):
            tensors = [value.detach() for value in values if torch.is_tensor(value)]
            if not tensors:
                return None
            out = tensors[0]
            for item in tensors[1:]:
                out = out + item
            return out

        macro_usage_total = _sum_optional(macro_usage_h, macro_usage_v)
        boundary_usage_total = _sum_optional(boundary_usage_h, boundary_usage_v)
        base_fixed_total = _sum_optional(base_fixed_usage_h, base_fixed_usage_v)
        initial_total = _sum_optional(initial_density_tracks_h, initial_density_tracks_v)
        occupancy_total = _sum_optional(occupancy_tracks_h, occupancy_tracks_v)
        residual_total = _sum_optional(residual_h, residual_v)
        macro_active = (
            macro_usage_total.detach() > 0
            if torch.is_tensor(macro_usage_total)
            else torch.zeros((), dtype=torch.bool)
        )
        stats = dict(self.macro_exclusion_stats)
        stats.update(
            {
                "source_name": str(source_name),
                "macro_injection_mode": "fixed_usage",
                "macro_usage_reference_quantile": float(_MACRO_USAGE_REFERENCE_QUANTILE),
                "macro_usage_reference_p95_h": float(macro_usage_reference_p95_h),
                "macro_usage_reference_p95_v": float(macro_usage_reference_p95_v),
                "macro_usage_reference_p95": max(
                    float(macro_usage_reference_p95_h),
                    float(macro_usage_reference_p95_v),
                ),
                "macro_usage_h_max": _scalar_stat(macro_usage_h, "max"),
                "macro_usage_v_max": _scalar_stat(macro_usage_v, "max"),
                "macro_usage_h_sum": _scalar_stat(macro_usage_h, "sum"),
                "macro_usage_v_sum": _scalar_stat(macro_usage_v, "sum"),
                "macro_usage_max": _scalar_stat(macro_usage_total, "max"),
                "macro_usage_sum": _scalar_stat(macro_usage_total, "sum"),
                "macro_usage_active_bins": _positive_count(macro_usage_total),
                "boundary_source_enabled": bool(
                    self.boundary_source_stats.get("boundary_source_enabled", False)
                ),
                "boundary_usage_reference_quantile": float(_BOUNDARY_SOURCE_REFERENCE_QUANTILE),
                "boundary_usage_reference_p95_h": float(boundary_usage_reference_p95_h),
                "boundary_usage_reference_p95_v": float(boundary_usage_reference_p95_v),
                "boundary_usage_reference_p95": max(
                    float(boundary_usage_reference_p95_h),
                    float(boundary_usage_reference_p95_v),
                ),
                "boundary_usage_h_max": _scalar_stat(boundary_usage_h, "max"),
                "boundary_usage_v_max": _scalar_stat(boundary_usage_v, "max"),
                "boundary_usage_h_sum": _scalar_stat(boundary_usage_h, "sum"),
                "boundary_usage_v_sum": _scalar_stat(boundary_usage_v, "sum"),
                "boundary_usage_max": _scalar_stat(boundary_usage_total, "max"),
                "boundary_usage_sum": _scalar_stat(boundary_usage_total, "sum"),
                "boundary_usage_active_bins": _positive_count(boundary_usage_total),
                "base_fixed_usage_sum": _scalar_stat(base_fixed_total, "sum"),
                "initial_density_tracks_sum": _scalar_stat(initial_total, "sum"),
                "occupancy_tracks_sum": _scalar_stat(occupancy_total, "sum"),
                "residual_macro_bins_sum": (
                    float(residual_total.detach()[macro_active].sum().item())
                    if torch.is_tensor(residual_total)
                    and torch.is_tensor(macro_active)
                    and macro_active.numel() == residual_total.numel()
                    and bool(macro_active.any().item())
                    else 0.0
                ),
                "macro_dominates_bins": 0,
                "routing_dominates_macro_bins": 0,
            }
        )
        self.last_macro_exclusion_stats = stats
        if self.log_verbose >= 2 and stats.get("macro_exclusion_enabled"):
            logger.info(
                "L-shape macro occupancy injection: source=%s macro_count=%d "
                "macro_usage_bins=%d macro_usage_max=%.4e macro_usage_sum=%.4e "
                "macro_usage_p95_h=%.4e macro_usage_p95_v=%.4e "
                "boundary_usage_bins=%d boundary_usage_max=%.4e boundary_usage_sum=%.4e "
                "base_fixed_sum=%.4e initial_fixed_sum=%.4e occupancy_sum=%.4e",
                str(source_name),
                int(stats.get("macro_count", 0)),
                int(stats.get("macro_usage_active_bins", 0)),
                float(stats.get("macro_usage_max", 0.0)),
                float(stats.get("macro_usage_sum", 0.0)),
                float(stats.get("macro_usage_reference_p95_h", 0.0)),
                float(stats.get("macro_usage_reference_p95_v", 0.0)),
                int(stats.get("boundary_usage_active_bins", 0)),
                float(stats.get("boundary_usage_max", 0.0)),
                float(stats.get("boundary_usage_sum", 0.0)),
                float(stats.get("base_fixed_usage_sum", 0.0)),
                float(stats.get("initial_density_tracks_sum", 0.0)),
                float(stats.get("occupancy_tracks_sum", 0.0)),
            )

    def _merge_macro_source(self, source_map, source_name):
        """Deprecated: macro exclusion is injected as fixed occupancy before AL."""
        macro_source = self._macro_source_for(source_map)
        if macro_source is None:
            return source_map
        source_before = source_map
        macro_detached = macro_source.detach()
        before_detached = source_before.detach()
        macro_active = macro_detached > 0
        routing_dominates = macro_active & (before_detached > 0)
        stats = dict(self.macro_exclusion_stats)
        stats.update(
            {
                "source_name": str(source_name),
                "macro_injection_mode": "fixed_usage",
                "source_before_macro_max": _scalar_stat(before_detached, "max"),
                "source_before_macro_sum": _scalar_stat(before_detached, "sum"),
                "source_after_macro_max": _scalar_stat(source_before.detach(), "max"),
                "source_after_macro_sum": _scalar_stat(source_before.detach(), "sum"),
                "macro_source_effective_min": 0.0,
                "macro_source_effective_max": 0.0,
                "macro_source_effective_sum": 0.0,
                "macro_source_effective_scale": 0.0,
                "macro_source_reference_p99": 0.0,
                "macro_source_reference_quantile": 0.0,
                "macro_source_reference_scale": 0.0,
                "macro_source_min_effective_strength": 0.0,
                "macro_dominates_bins": 0,
                "routing_dominates_macro_bins": int(routing_dominates.sum().item()),
                "macro_source_device": str(macro_source.device),
                "macro_source_dtype": str(macro_source.dtype),
            }
        )
        if isinstance(self.last_macro_exclusion_stats, dict):
            sources = dict(self.last_macro_exclusion_stats.get("sources", {}))
        else:
            sources = {}
        sources[str(source_name)] = dict(stats)
        if len(sources) > 1:
            stats["source_name"] = ",".join(sorted(sources))
            stats["macro_dominates_bins"] = sum(
                int(item.get("macro_dominates_bins", 0))
                for item in sources.values()
            )
            stats["routing_dominates_macro_bins"] = sum(
                int(item.get("routing_dominates_macro_bins", 0))
                for item in sources.values()
            )
            stats["macro_source_effective_scale"] = max(
                float(item.get("macro_source_effective_scale", 0.0))
                for item in sources.values()
            )
            stats["macro_source_reference_p99"] = max(
                float(item.get("macro_source_reference_p99", 0.0))
                for item in sources.values()
            )
            stats["macro_source_effective_max"] = max(
                float(item.get("macro_source_effective_max", 0.0))
                for item in sources.values()
            )
            stats["macro_source_effective_sum"] = sum(
                float(item.get("macro_source_effective_sum", 0.0))
                for item in sources.values()
            )
        stats["sources"] = sources
        self.last_macro_exclusion_stats = stats
        if self.log_verbose >= 2:
            logger.info(
                "L-shape macro exclusion source merge skipped: source=%s macro_exclusion_enabled=%s "
                "macro_count=%d macro_body_bins=%d macro_halo_bins=%d "
                "macro_source_active_bins=%d macro_source_max=%.4e macro_source_sum=%.4e "
                "macro_effective_scale=%.4e macro_effective_max=%.4e "
                "macro_effective_sum=%.4e source_p99=%.4e "
                "source_before_max=%.4e source_after_max=%.4e "
                "macro_dominates_bins=%d routing_dominates_macro_bins=%d "
                "macro_source_grid_shape=%s macro_source_coordinate_system=%s "
                "macro_set_source=%s",
                str(source_name),
                str(stats.get("macro_exclusion_enabled")),
                int(stats.get("macro_count", 0)),
                int(stats.get("macro_body_bins", 0)),
                int(stats.get("macro_halo_bins", 0)),
                int(stats.get("macro_source_active_bins", 0)),
                float(stats.get("macro_source_max", 0.0)),
                float(stats.get("macro_source_sum", 0.0)),
                float(stats.get("macro_source_effective_scale", 0.0)),
                float(stats.get("macro_source_effective_max", 0.0)),
                float(stats.get("macro_source_effective_sum", 0.0)),
                float(stats.get("macro_source_reference_p99", 0.0)),
                float(stats.get("source_before_macro_max", 0.0)),
                float(stats.get("source_after_macro_max", 0.0)),
                int(stats.get("macro_dominates_bins", 0)),
                int(stats.get("routing_dominates_macro_bins", 0)),
                str(stats.get("macro_source_grid_shape")),
                str(stats.get("macro_source_coordinate_system")),
                str(stats.get("macro_set_source")),
            )
        return source_map

    def reset_capacity_al_state(self, reason="manual"):
        self.lambda_h = None
        self.lambda_v = None
        self._capacity_al_reset_key = None
        self._capacity_al_last_update_iter = None
        self._capacity_al_last_reset_reason = str(reason)
        self._capacity_al_reset_count += 1
        if self.log_verbose >= 1:
            logger.info("L-shape capacity AL reset: reason=%s", reason)

    def _capacity_al_key_from_maps(
        self,
        supply_h,
        supply_v,
        fix_usage_h,
        fix_usage_v,
        macro_source_map=None,
        extra_key=None,
    ):
        return (
            _tensor_identity(supply_h),
            _tensor_identity(supply_v),
            _tensor_identity(fix_usage_h),
            _tensor_identity(fix_usage_v),
            _tensor_identity(macro_source_map),
            extra_key,
        )

    def _ensure_capacity_al_state(self, g_h, g_v, reset_key):
        if reset_key != self._capacity_al_reset_key:
            self.lambda_h = None
            self.lambda_v = None
            self._capacity_al_reset_key = reset_key
            self._capacity_al_last_update_iter = None
            self._capacity_al_last_reset_reason = "constraint_identity_changed"
            self._capacity_al_reset_count += 1
            if self.log_verbose >= 1:
                logger.info(
                    "L-shape capacity AL reset: reason=constraint_identity_changed"
                )

        need_reset = (
            self.lambda_h is None
            or self.lambda_v is None
            or self.lambda_h.shape != g_h.shape
            or self.lambda_v.shape != g_v.shape
            or self.lambda_h.device != g_h.device
            or self.lambda_v.device != g_v.device
            or self.lambda_h.dtype != g_h.dtype
            or self.lambda_v.dtype != g_v.dtype
        )
        if need_reset:
            self.lambda_h = torch.zeros_like(g_h, requires_grad=False).detach()
            self.lambda_v = torch.zeros_like(g_v, requires_grad=False).detach()
            self._capacity_al_last_update_iter = None
            if self._capacity_al_last_reset_reason is None:
                self._capacity_al_last_reset_reason = "shape_device_dtype_changed"
            if self.log_verbose >= 1:
                logger.info(
                    "L-shape capacity AL reset: reason=shape_device_dtype_changed "
                    "shape_h=%s shape_v=%s dtype=%s device=%s",
                    tuple(int(dim) for dim in g_h.shape),
                    tuple(int(dim) for dim in g_v.shape),
                    str(g_h.dtype),
                    str(g_h.device),
                )
        else:
            self.lambda_h = self.lambda_h.detach()
            self.lambda_v = self.lambda_v.detach()

    def _capacity_al_state_matches(self, g_h, g_v, reset_key):
        return (
            reset_key == self._capacity_al_reset_key
            and self.lambda_h is not None
            and self.lambda_v is not None
            and self.lambda_h.shape == g_h.shape
            and self.lambda_v.shape == g_v.shape
            and self.lambda_h.device == g_h.device
            and self.lambda_v.device == g_v.device
            and self.lambda_h.dtype == g_h.dtype
            and self.lambda_v.dtype == g_v.dtype
        )

    def _update_capacity_al_lambda(self, g_h, g_v, placement_iteration_id):
        if placement_iteration_id is None:
            raise RuntimeError(
                "capacity AL lambda update requires placement_iteration_id"
            )
        iteration = int(placement_iteration_id)
        if self._capacity_al_last_update_iter == iteration:
            return False

        with torch.no_grad():
            for lambda_map, residual in (
                (self.lambda_h, g_h.detach()),
                (self.lambda_v, g_v.detach()),
            ):
                clipped = residual.clamp(min=-1.0, max=1.0)
                delta = (
                    self._capacity_al_positive_step * torch.relu(clipped)
                    + self._capacity_al_negative_step * torch.minimum(
                        clipped, torch.zeros_like(clipped)
                    )
                )
                lambda_map.add_(self._capacity_al_rho * delta)
                lambda_map.clamp_(min=0.0, max=self._capacity_al_lambda_max)

        self._capacity_al_last_update_iter = iteration
        self._capacity_al_update_count += 1
        return True

    def _capacity_al_source_maps(
        self,
        g_h,
        g_v,
        *,
        update_lambda=False,
        placement_iteration_id=None,
        reset_key=None,
    ):
        g_h_detached = g_h.detach()
        g_v_detached = g_v.detach()
        if not self.capacity_al_enable:
            self.last_al_stats = {
                "enabled": False,
                "source_h_max": _scalar_stat(torch.relu(g_h_detached), "max"),
                "source_v_max": _scalar_stat(torch.relu(g_v_detached), "max"),
            }
            return torch.relu(g_h), torch.relu(g_v)

        if update_lambda:
            self._ensure_capacity_al_state(g_h_detached, g_v_detached, reset_key)
            lambda_h_t = self.lambda_h.detach().clone()
            lambda_v_t = self.lambda_v.detach().clone()
            reset_reason = self._capacity_al_last_reset_reason
        elif self._capacity_al_state_matches(g_h_detached, g_v_detached, reset_key):
            lambda_h_t = self.lambda_h.detach().clone()
            lambda_v_t = self.lambda_v.detach().clone()
            reset_reason = self._capacity_al_last_reset_reason
        else:
            lambda_h_t = torch.zeros_like(g_h_detached)
            lambda_v_t = torch.zeros_like(g_v_detached)
            reset_reason = "read_only_missing_or_stale_state"
        q_h = torch.relu(lambda_h_t + self._capacity_al_rho * g_h)
        q_v = torch.relu(lambda_v_t + self._capacity_al_rho * g_v)

        updated = False
        if update_lambda:
            updated = self._update_capacity_al_lambda(
                g_h_detached,
                g_v_detached,
                placement_iteration_id,
            )

        q_h_detached = q_h.detach()
        q_v_detached = q_v.detach()
        active_memory_h = (q_h_detached > 0) & (g_h_detached <= 0)
        active_memory_v = (q_v_detached > 0) & (g_v_detached <= 0)
        g_h_pos_bins = _positive_count(g_h_detached)
        g_v_pos_bins = _positive_count(g_v_detached)
        g_h_bins = int(g_h_detached.numel())
        g_v_bins = int(g_v_detached.numel())
        self.last_al_stats = {
            "enabled": True,
            "rho": float(self._capacity_al_rho),
            "lambda_max": float(self._capacity_al_lambda_max),
            "updated": bool(updated),
            "placement_iteration_id": (
                None if placement_iteration_id is None else int(placement_iteration_id)
            ),
            "last_update_iter": self._capacity_al_last_update_iter,
            "update_count": int(self._capacity_al_update_count),
            "reset_count": int(self._capacity_al_reset_count),
            "reset_reason": reset_reason,
            "g_h_max": _scalar_stat(torch.relu(g_h_detached), "max"),
            "g_v_max": _scalar_stat(torch.relu(g_v_detached), "max"),
            "g_h_sum": _scalar_stat(torch.relu(g_h_detached), "sum"),
            "g_v_sum": _scalar_stat(torch.relu(g_v_detached), "sum"),
            "g_h_pos_bins": g_h_pos_bins,
            "g_v_pos_bins": g_v_pos_bins,
            "g_h_bins": g_h_bins,
            "g_v_bins": g_v_bins,
            "g_h_pos_ratio": float(g_h_pos_bins) / max(float(g_h_bins), 1.0),
            "g_v_pos_ratio": float(g_v_pos_bins) / max(float(g_v_bins), 1.0),
            "q_h_max": _scalar_stat(q_h_detached, "max"),
            "q_v_max": _scalar_stat(q_v_detached, "max"),
            "q_h_sum": _scalar_stat(q_h_detached, "sum"),
            "q_v_sum": _scalar_stat(q_v_detached, "sum"),
            "q_h_pos_bins": _positive_count(q_h_detached),
            "q_v_pos_bins": _positive_count(q_v_detached),
            "lambda_h_max": _scalar_stat(lambda_h_t, "max"),
            "lambda_v_max": _scalar_stat(lambda_v_t, "max"),
            "lambda_h_sum": _scalar_stat(lambda_h_t, "sum"),
            "lambda_v_sum": _scalar_stat(lambda_v_t, "sum"),
            "lambda_h_next_max": _scalar_stat(self.lambda_h, "max"),
            "lambda_v_next_max": _scalar_stat(self.lambda_v, "max"),
            "lambda_h_next_sum": _scalar_stat(self.lambda_h, "sum"),
            "lambda_v_next_sum": _scalar_stat(self.lambda_v, "sum"),
            "active_memory_bins_h": int(active_memory_h.sum().item()),
            "active_memory_bins_v": int(active_memory_v.sum().item()),
        }
        return q_h, q_v
    
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
        self.macro_body_source_map = self.macro_body_source_map.to(device=device, dtype=dtype)
        self.macro_halo_source_map = self.macro_halo_source_map.to(device=device, dtype=dtype)
        self.macro_source_map = self.macro_source_map.to(device=device, dtype=dtype)
        self.boundary_source_map = self.boundary_source_map.to(device=device, dtype=dtype)
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
        self.reset_capacity_al_state("target_density_changed")
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
            self.reset_capacity_al_state("target_demand_changed")
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
        self.reset_capacity_al_state("raw_wire_demand_changed")
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
        self.reset_capacity_al_state("supply_original_changed")
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
        self.reset_capacity_al_state("fix_usage_changed")
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
        if any(value is not None for value in updates.values()):
            self.reset_capacity_al_state("directional_targets_changed")
    
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
    
    def forward(
        self,
        segment_pos,
        segment_size_x,
        segment_size_y,
        segment_is_horizontal=None,
        segment_weight=None,
        update_capacity_al_lambda=False,
        placement_iteration_id=None,
    ):
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
        if self.capacity_al_enable:
            required_maps = (
                self.supply_original_h,
                self.supply_original_v,
                self.fix_usage_map_h,
                self.fix_usage_map_v,
            )
            if not all(isinstance(value, torch.Tensor) for value in required_maps):
                raise RuntimeError(
                    "l_shape_capacity_al_enable requires H/V capacity and fixed-usage maps"
                )
            if not (
                isinstance(segment_is_horizontal, torch.Tensor)
                and int(segment_is_horizontal.numel()) == int(num_segments)
            ):
                raise RuntimeError(
                    "l_shape_capacity_al_enable requires H/V segment directions"
                )
        capacity_al_reset_key = self._capacity_al_key_from_maps(
            self.supply_original_h,
            self.supply_original_v,
            self.fix_usage_map_h,
            self.fix_usage_map_v,
            macro_source_map=self.macro_source_map,
            extra_key=(
                bool(self.capacity_al_enable),
                _tensor_identity(self.boundary_source_map),
            ),
        )
        
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
            self.fast_mode,
            self,
            bool(update_capacity_al_lambda),
            placement_iteration_id,
            capacity_al_reset_key,
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
            fast_mode=False,
            placedb=placedb,
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
    capacity_al_enable=False,
    boundary_source_enable=False,
    boundary_source_width_bins=0,
    boundary_source_strength=0.0,
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
        capacity_al_enable=capacity_al_enable,
        boundary_source_enable=boundary_source_enable,
        boundary_source_width_bins=boundary_source_width_bins,
        boundary_source_strength=boundary_source_strength,
        placedb=placedb,
    )
