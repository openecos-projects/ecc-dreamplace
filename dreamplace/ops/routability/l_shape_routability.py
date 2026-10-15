##
# @file   l_shape_routability.py
# @brief  L-shape guided routability optimization module
#         Uses routing-oracle L-direction information to build accurate routing density model
#         Supports both Python-based RUDY density and C++/CUDA electric potential
#

import torch
import torch.nn as nn
import logging
import time
import json
import os
import hashlib

import dreamplace.ops.routability.l_shape_segment as l_shape_segment_module
from dreamplace.ops.routability.l_shape_segment import (
    LShapeSegmentOp,
    build_segment_pos_tensor,
    build_l_shape_segments_vectorized,
    _build_gather_plan,
    _deterministic_gather_1d,
    maybe_save_segment_snapshot,
    H_FIRST, V_FIRST, STRAIGHT, FAKE_STRAIGHT, UNKNOWN
)
from dreamplace.ops.routability.segment_density import (
    SegmentDensityOp,
    compute_segment_rudy_density,
    create_segment_density_op
)
from dreamplace.ops.routability.l_shape_electric_potential import (
    LShapeElectricPotential,
    SegmentElectricPotentialFunction,
    compute_track_rho_components,
    create_l_shape_electric_potential,
)
from dreamplace.ops.routability.l_shape_electric_overflow import (
    LShapeElectricOverflow,
    create_l_shape_electric_overflow
)
from dreamplace.ops.routability.same_net_topo_scoring import (
    compute_diagonal_split_topo_costs,
)
from dreamplace.ops.routability.profile_timing import (
    l_shape_log_verbose,
    profile_end,
    profile_scope,
    profile_start,
)

logger = logging.getLogger(__name__)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _module_file_digest(module):
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return "<unknown>", "<unknown>"

    digest = hashlib.blake2b(digest_size=16)
    try:
        with open(module_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return module_path, f"error:{exc}"
    return module_path, digest.hexdigest()


class LShapeRoutabilityOp(nn.Module):
    """
    使用EGR L方向信息的可布线性优化模块
    
    核心流程:
    1. 从steiner_topo_op获取Steiner树边和L方向
    2. 根据L方向将边拆分为segments
    3. 计算segment密度作为routing congestion的代理
    4. 返回可微的routability cost
    
    使用示例:
        op = LShapeRoutabilityOp(placedb, params)
        cost = op(pos, steiner_topo_op, pin_pos_op)
    
    支持两种密度计算模式:
        - "rudy": Python实现的RUDY密度 (默认)
        - "electric": C++/CUDA实现的电势场模型 (更快，梯度更准确)
    """
    
    def __init__(
        self,
        placedb,
        params,
        wire_width=None,
        wire_width_h=None,
        wire_width_v=None,
        num_bins_x=64,
        num_bins_y=64,
        density_mode="electric",
        target_density=1.0,
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
    ):
        """
        Args:
            placedb: placement database
            params: parameters
            wire_width: legacy segment线宽（默认从params获取或使用0）
            num_bins_x, num_bins_y: 密度计算的bin数量
            density_mode: 密度计算模式 ("rudy" 或 "electric")
            target_density: 2D routing supply map (用于electric模式)
            target_demand: 2D routing demand map (用于electric模式的单位标定)
        """
        super(LShapeRoutabilityOp, self).__init__()
        
        self.placedb = placedb
        self.params = params
        self.gradient_mode = getattr(params, "l_shape_gradient_mode", "translation")
        if self.gradient_mode not in ("translation", "full"):
            raise ValueError(f"Unknown L-shape gradient mode: {self.gradient_mode}")
        if self.gradient_mode == "full" and (
            density_mode != "electric" or bool(getattr(params, "soft_l_assignment", False))
        ):
            raise ValueError(
                "Full L-shape gradients require electric density and hard L assignment"
            )
        logger.info("L-shape gradient mode=%s", self.gradient_mode)
        segment_module_path, segment_module_digest = _module_file_digest(l_shape_segment_module)
        segment_reference_env = "DREAMPLACE_L_SHAPE_SEGMENT_COMPACTION_REFERENCE"
        legacy_segment_reference_env = "DREAMPLACE_L_SHAPE_HARD_SEGMENT_COMPACTION_REFERENCE"
        segment_reference_enabled = l_shape_segment_module._env_any_flag_enabled(
            segment_reference_env,
            legacy_segment_reference_env,
        )
        logger.info(
            "LShapeSegmentProvenance module_path=%s blake2b16=%s "
            "segment_compaction_reference_env=%s legacy_segment_compaction_reference_env=%s "
            "segment_compaction_reference_enabled=%s",
            segment_module_path,
            segment_module_digest,
            segment_reference_env,
            legacy_segment_reference_env,
            segment_reference_enabled,
        )
        self.density_mode = density_mode
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        
        # 线宽设置
        if wire_width is None:
            wire_width = getattr(params, 'route_wire_width', 0.0)
        self.wire_width = wire_width
        self.soft_l_assignment = bool(getattr(params, "soft_l_assignment", False))
        self.soft_l_use_resolver_prior = bool(getattr(params, "soft_l_use_resolver_prior", True))
        self.soft_l_temperature = max(float(getattr(params, "soft_l_temperature", 1.0)), 1e-6)
        self.soft_l_prior_bias = float(getattr(params, "soft_l_prior_bias", 1.0))
        self.soft_l_min_weight = max(float(getattr(params, "soft_l_min_weight", 0.05)), 0.0)
        self.soft_l_tie_break_delta = max(float(getattr(params, "soft_l_tie_break_delta", 0.10)), 0.0)
        self.soft_l_background_weight = max(float(getattr(params, "soft_l_background_weight", 0.05)), 0.0)
        self.soft_l_self_overflow_weight = max(float(getattr(params, "soft_l_self_overflow_weight", 0.5)), 0.0)
        self.soft_l_hotspot_weight = max(float(getattr(params, "soft_l_hotspot_weight", 2.0)), 0.0)
        self.soft_l_hotspot_ramp_ratio = max(float(getattr(params, "soft_l_hotspot_ramp_ratio", 0.03)), 0.0)
        self.soft_l_adaptive_tau = bool(getattr(params, "soft_l_adaptive_tau", False))
        self.soft_l_adaptive_scale = max(float(getattr(params, "soft_l_adaptive_scale", 1.0)), 1e-6)
        self.per_net_topology_mode = True
        self.soft_l_same_net_diag_split_sigma = max(
            float(getattr(params, "soft_l_same_net_diag_split_sigma", 1.0)), 1e-6
        )
        self.soft_l_same_net_diag_split_min_support = max(
            float(getattr(params, "soft_l_same_net_diag_split_min_support", 1e-6)), 0.0
        )
        self.soft_l_same_net_diag_split_max_distance = max(
            float(getattr(params, "soft_l_same_net_diag_split_max_distance", 0.0)), 0.0
        )
        self.profile_enabled = bool(getattr(params, "l_shape_profile_flag", False))
        self.log_verbose = l_shape_log_verbose(params)
        self.l_shape_fast_mode = _as_bool(getattr(params, "l_shape_fast_mode", False))
        self.capacity_al_enable = bool(
            getattr(params, "l_shape_capacity_al_enable", False)
        )
        self.boundary_source_enable = bool(
            getattr(params, "l_shape_boundary_source_enable", False)
        )
        self.boundary_source_width_bins = max(
            int(getattr(params, "l_shape_boundary_source_width_bins", 0)),
            0,
        )
        self.boundary_source_strength = max(
            float(getattr(params, "l_shape_boundary_source_strength", 0.0)),
            0.0,
        )
        self.deterministic_flag = _as_bool(getattr(params, "deterministic_flag", False))
        self.debug_hash_enabled = _as_bool(getattr(params, "l_shape_debug_hash_flag", False))
        self.debug_hash_start_iter = int(getattr(params, "l_shape_debug_hash_start_iter", -1))
        self.debug_hash_end_iter = int(getattr(params, "l_shape_debug_hash_end_iter", -1))
        self.debug_hash_sample = max(int(getattr(params, "l_shape_debug_hash_sample", 4096)), 0)
        self._debug_hash_iteration = None
        self.segment_snapshot_path = os.environ.get(
            "DREAMPLACE_L_SHAPE_SEGMENT_SNAPSHOT_PATH",
            os.environ.get("DREAMPLACE_L_SHAPE_HARD_SEGMENT_SNAPSHOT_PATH"),
        )
        self.segment_snapshot_iter = os.environ.get(
            "DREAMPLACE_L_SHAPE_SEGMENT_SNAPSHOT_ITER",
            os.environ.get("DREAMPLACE_L_SHAPE_HARD_SEGMENT_SNAPSHOT_ITER"),
        )
        self._segment_snapshot_saved = False
        if self.segment_snapshot_path:
            logger.info(
                "LShapeSegmentSnapshot path=%s iteration=%s",
                self.segment_snapshot_path,
                self.segment_snapshot_iter or "<first>",
            )
        self.per_net_topology_use_cpp = True
        cache_flag = getattr(params, "l_shape_edge_net_ids_cache_flag", 1)
        if isinstance(cache_flag, str):
            self.l_shape_edge_net_ids_cache_flag = cache_flag.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        else:
            self.l_shape_edge_net_ids_cache_flag = bool(cache_flag)
        self.soft_l_debug_update_interval = max(
            int(getattr(params, "soft_l_debug_update_interval", 10)),
            1,
        )
        self._soft_l_debug_call_count = 0
        self.blockage_initial_density = True
        self._cached_fallback_l_directions = None
        self._cached_fallback_l_directions_key = None

        directional_track_widths = (
            self.density_mode == "electric"
            and self.blockage_initial_density
        )
        if directional_track_widths:
            self.wire_width_h = float(self.bin_size_y if wire_width_h is None else wire_width_h)
            self.wire_width_v = float(self.bin_size_x if wire_width_v is None else wire_width_v)
        else:
            self.wire_width_h = None if wire_width_h is None else float(wire_width_h)
            self.wire_width_v = None if wire_width_v is None else float(wire_width_v)

        # L形segment构建器
        self.segment_builder = LShapeSegmentOp(
            wire_width=wire_width,
            wire_width_h=self.wire_width_h,
            wire_width_v=self.wire_width_v,
            use_vectorized=True,
            soft_min_weight=self.soft_l_min_weight,
            deterministic_backward=self.deterministic_flag,
            log_verbose=self.log_verbose,
            differentiate_sizes=self.gradient_mode == "full",
        )
        if directional_track_widths and self.log_verbose >= 1:
            logger.info(
                "Use directional track-space segment thickness for blockage initial density: width_h=%.4f width_v=%.4f",
                self.wire_width_h,
                self.wire_width_v,
            )
        if self.soft_l_assignment and self.log_verbose >= 1:
            logger.info(
                "Soft L-assignment enabled: tau=%.3f adaptive_tau=%s adaptive_scale=%.3f prior=%s bias=%.3f min_weight=%.3f tie_delta=%.3f bg=%.3f self_ov=%.3f hotspot=%.3f hotspot_ramp=%.3f debug_interval=%d edge_net_ids_cache=%s",
                self.soft_l_temperature,
                self.soft_l_adaptive_tau,
                self.soft_l_adaptive_scale,
                self.soft_l_use_resolver_prior,
                self.soft_l_prior_bias,
                self.soft_l_min_weight,
                self.soft_l_tie_break_delta,
                self.soft_l_background_weight,
                self.soft_l_self_overflow_weight,
                self.soft_l_hotspot_weight,
                self.soft_l_hotspot_ramp_ratio,
                self.soft_l_debug_update_interval,
                self.l_shape_edge_net_ids_cache_flag,
            )
        if self.log_verbose >= 1:
            logger.info(
                "Per-net topology scoring enabled: kernel=diag_split_geometric topo_only=1 sigma=%.4f min_support=%.3e max_distance=%.4f cpp=1 overflow_density_deterministic=%s",
                self.soft_l_same_net_diag_split_sigma,
                self.soft_l_same_net_diag_split_min_support,
                self.soft_l_same_net_diag_split_max_distance,
                self.deterministic_flag,
            )
        
        # 根据模式选择密度计算器
        if density_mode == "electric":

            self.density_op = create_l_shape_electric_potential(
                placedb,
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
                # padding=1,  # 边界填充
                fast_mode=self.l_shape_fast_mode,
                gradient_mode=self.gradient_mode,
                require_field_maps=(
                    (float(getattr(params, "l_shape_filler_reverse_force", 0)) > 0
                     and placedb.num_filler_nodes > 0)
                    or bool(getattr(params, "l_shape_plot_flag", False))
                ),
                profile_enabled=self.profile_enabled,
                log_verbose=self.log_verbose,
                capacity_al_enable=self.capacity_al_enable,
                boundary_source_enable=self.boundary_source_enable,
                boundary_source_width_bins=self.boundary_source_width_bins,
                boundary_source_strength=self.boundary_source_strength,
            )
            self.overflow_op = create_l_shape_electric_overflow(
                placedb,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                target_density=target_density,
                deterministic_flag=self.deterministic_flag,
                log_verbose=self.log_verbose,
                # padding=1  # 边界填充
            )
            if self.log_verbose >= 1:
                logger.info(
                    "Using C++/CUDA electric potential for routability (fast_mode=%d)",
                    1 if self.l_shape_fast_mode else 0,
                )
        else:
            if self.l_shape_fast_mode and self.log_verbose >= 1:
                logger.info(
                    "l_shape_fast_mode is ignored for non-electric L-shape density mode"
                )
            # Python RUDY密度
            self.density_op = SegmentDensityOp(
                xl=placedb.xl,
                yl=placedb.yl,
                xh=placedb.xh,
                yh=placedb.yh,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y
            )
            self.overflow_op = None
            if self.log_verbose >= 1:
                logger.info(f"Using Python RUDY density for routability")
        
        # 存储边界信息用于OOB检查（优先 routing grid）
        self.xl = getattr(placedb, "routing_grid_xl", placedb.xl)
        self.yl = getattr(placedb, "routing_grid_yl", placedb.yl)
        self.xh = getattr(placedb, "routing_grid_xh", placedb.xh)
        self.yh = getattr(placedb, "routing_grid_yh", placedb.yh)
        
        # 缓存
        self.cached_segments = None
        self.cached_soft_debug = None
        self._cached_edge_net_ids = None
        self._cached_edge_net_ids_key = None
        self._cached_soft_edge_topology = None
        self._cached_soft_edge_topology_key = None
        self.cached_density_map = None
        self.cached_density_map_h = None
        self.cached_density_map_v = None
        self.per_net_topology_cache = None
        self.per_net_topology_stats = None
        self._ggr_fast_path_log_emitted = False
        self._ggr_fast_path_warning_emitted = False
        self._ggr_vertex_x_cache = None
        self._ggr_vertex_y_cache = None
        self._ggr_flat_from_cache = None
        self._ggr_flat_to_cache = None
        self._ggr_l_direction_cache = None
        self.target_density_h = target_density_h
        self.target_density_v = target_density_v
        self.target_demand_h = target_demand_h
        self.target_demand_v = target_demand_v
        self.raw_wire_demand_map = raw_wire_demand_map
        self.raw_wire_demand_map_h = raw_wire_demand_map_h
        self.raw_wire_demand_map_v = raw_wire_demand_map_v
        self.supply_original = supply_original
        self.supply_original_h = supply_original_h
        self.supply_original_v = supply_original_v
        self.fix_usage_map = fix_usage_map
        self.fix_usage_map_h = fix_usage_map_h
        self.fix_usage_map_v = fix_usage_map_v

    def set_debug_iteration(self, iteration):
        self._debug_hash_iteration = None if iteration is None else int(iteration)

    def _debug_hash_active(self):
        if not self.debug_hash_enabled:
            return False
        iteration = self._debug_hash_iteration
        if iteration is None:
            return False
        start_iter = self.debug_hash_start_iter
        end_iter = self.debug_hash_end_iter
        if start_iter >= 0 and iteration < start_iter:
            return False
        if end_iter >= 0 and iteration > end_iter:
            return False
        return True

    def _tensor_debug_hash(self, tensor):
        if tensor is None:
            return "none"
        if not isinstance(tensor, torch.Tensor):
            return str(type(tensor).__name__)
        values = tensor.detach()
        if values.device.type != "cpu":
            values = values.cpu()
        values = values.contiguous()
        numel = int(values.numel())
        if numel == 0:
            return "empty"
        sample = self.debug_hash_sample
        shape = tuple(int(dim) for dim in values.shape)
        dtype = str(values.dtype)
        if sample > 0 and numel > sample:
            half = max(sample // 2, 1)
            values = torch.cat((values.reshape(-1)[:half], values.reshape(-1)[-half:]))
        raw_values = values.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        header = ("%s|%s|%d|" % (dtype, shape, numel)).encode("ascii")
        return hashlib.sha1(header + raw_values).hexdigest()[:16]

    def _tensor_debug_stats(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return ""
        values = tensor.detach()
        if values.device.type != "cpu":
            values = values.cpu()
        values = values.reshape(-1)
        if values.numel() == 0:
            return " dtype=%s shape=%s numel=0" % (
                str(tensor.dtype),
                tuple(int(dim) for dim in tensor.shape),
            )
        prefix = " dtype=%s shape=%s" % (
            str(tensor.dtype),
            tuple(int(dim) for dim in tensor.shape),
        )
        if not torch.is_floating_point(values):
            return prefix + " numel=%d min=%d max=%d sum=%d" % (
                int(values.numel()),
                int(values.min().item()),
                int(values.max().item()),
                int(values.to(dtype=torch.int64).sum().item()),
            )
        values_f = values.to(dtype=torch.float32)
        return prefix + " numel=%d min=%.9e max=%.9e sum=%.9e mean=%.9e" % (
            int(values.numel()),
            float(values_f.min().item()),
            float(values_f.max().item()),
            float(values_f.sum().item()),
            float(values_f.mean().item()),
        )

    def _log_debug_hash(self, name, tensor, **fields):
        if not self._debug_hash_active():
            return
        extra = " ".join("%s=%s" % (key, value) for key, value in fields.items())
        if extra:
            extra = " " + extra
        logger.info(
            "LShapeHash iter=%s name=%s hash=%s%s%s",
            self._debug_hash_iteration,
            name,
            self._tensor_debug_hash(tensor),
            self._tensor_debug_stats(tensor),
            extra,
        )

    def log_debug_hash(self, name, tensor, **fields):
        self._log_debug_hash(name, tensor, **fields)

    def _maybe_save_segment_snapshot(
        self, newx, newy, flat_pin_from, flat_pin_to, l_directions
    ):
        if not self.segment_snapshot_path or self._segment_snapshot_saved:
            return
        if self.soft_l_assignment:
            return
        if self.segment_snapshot_iter:
            if self._debug_hash_iteration is None:
                return
            if int(self.segment_snapshot_iter) != int(self._debug_hash_iteration):
                return
        try:
            saved = maybe_save_segment_snapshot(
                self.segment_snapshot_path,
                self.segment_snapshot_iter,
                self._debug_hash_iteration,
                self.soft_l_assignment,
                self._segment_snapshot_saved,
                newx,
                newy,
                flat_pin_from,
                flat_pin_to,
                l_directions,
                wire_width=self.wire_width,
                wire_width_h=self.wire_width_h,
                wire_width_v=self.wire_width_v,
                deterministic_backward=self.deterministic_flag,
            )
            if saved:
                self._segment_snapshot_saved = True
                logger.info(
                    "Saved L-shape segment snapshot path=%s iteration=%s",
                    self.segment_snapshot_path,
                    self._debug_hash_iteration,
                )
        except Exception as exc:
            self._segment_snapshot_saved = True
            logger.warning(
                "Failed to save L-shape segment snapshot path=%s: %s",
                self.segment_snapshot_path,
                exc,
            )

    def _gather_vertices(self, values, indices, gather_plan=None):
        if indices.device != values.device:
            indices = indices.to(values.device)
        if not self.deterministic_flag or not values.requires_grad:
            return values[indices]
        if gather_plan is None:
            gather_plan = _build_gather_plan(indices)
        return _deterministic_gather_1d(values, gather_plan)

    def update_targets(
        self,
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
    ):
        """
        Refresh external routing targets used by the electric L-shape model.

        Args:
            target_density: 2D routing supply map
            target_demand: 2D routing demand map or None
        """
        if self.density_mode != "electric":
            logger.debug("Skip L-shape target update because density_mode=%s", self.density_mode)
            return

        updated = False
        if target_density is not None and hasattr(self.density_op, "set_target_density"):
            self.density_op.set_target_density(target_density)
            updated = True
        if target_density is not None and self.overflow_op is not None and hasattr(self.overflow_op, "set_target_density"):
            self.overflow_op.set_target_density(target_density)
            updated = True
        if target_demand is not None and hasattr(self.density_op, "set_target_demand"):
            self.density_op.set_target_demand(target_demand)
            updated = True
        if raw_wire_demand_map is not None and hasattr(self.density_op, "set_raw_wire_demand_map"):
            self.density_op.set_raw_wire_demand_map(raw_wire_demand_map)
            updated = True
        if supply_original is not None and hasattr(self.density_op, "set_supply_original"):
            self.density_op.set_supply_original(supply_original)
            updated = True
        if hasattr(self.density_op, "set_directional_targets"):
            self.density_op.set_directional_targets(
                target_density_h=target_density_h,
                target_density_v=target_density_v,
                target_demand_h=target_demand_h,
                target_demand_v=target_demand_v,
                raw_wire_demand_map_h=raw_wire_demand_map_h,
                raw_wire_demand_map_v=raw_wire_demand_map_v,
                supply_original_h=supply_original_h,
                supply_original_v=supply_original_v,
                fix_usage_map_h=fix_usage_map_h,
                fix_usage_map_v=fix_usage_map_v,
            )
            updated = True
        if fix_usage_map is not None and hasattr(self.density_op, "set_fix_usage_map"):
            self.density_op.set_fix_usage_map(fix_usage_map)
            updated = True

        if target_density_h is not None:
            self.target_density_h = target_density_h
            updated = True
        if target_density_v is not None:
            self.target_density_v = target_density_v
            updated = True
        if target_demand_h is not None:
            self.target_demand_h = target_demand_h
            updated = True
        if target_demand_v is not None:
            self.target_demand_v = target_demand_v
            updated = True
        if raw_wire_demand_map is not None:
            self.raw_wire_demand_map = raw_wire_demand_map
            updated = True
        if raw_wire_demand_map_h is not None:
            self.raw_wire_demand_map_h = raw_wire_demand_map_h
            updated = True
        if raw_wire_demand_map_v is not None:
            self.raw_wire_demand_map_v = raw_wire_demand_map_v
            updated = True
        if supply_original is not None:
            self.supply_original = supply_original
            updated = True
        if supply_original_h is not None:
            self.supply_original_h = supply_original_h
            updated = True
        if supply_original_v is not None:
            self.supply_original_v = supply_original_v
            updated = True
        if fix_usage_map is not None:
            self.fix_usage_map = fix_usage_map
            updated = True
        if fix_usage_map_h is not None:
            self.fix_usage_map_h = fix_usage_map_h
            updated = True
        if fix_usage_map_v is not None:
            self.fix_usage_map_v = fix_usage_map_v
            updated = True

        if updated:
            if hasattr(self.segment_builder, "reset_cache"):
                self.segment_builder.reset_cache()
            self._clear_soft_edge_topology_cache()
            self._clear_fallback_l_directions_cache()
            self._clear_ggr_fast_path_cache()
            self.cached_soft_debug = None
            self._soft_l_debug_call_count = 0
            self.cached_density_map = None
            self.cached_density_map_h = None
            self.cached_density_map_v = None
            if self.log_verbose >= 2:
                logger.info(
                    "Updated L-shape routing targets (density=%s, demand=%s, raw_wire=%s, density_h=%s, density_v=%s, demand_h=%s, demand_v=%s, raw_wire_h=%s, raw_wire_v=%s, supply_original=%s, supply_original_h=%s, supply_original_v=%s, fix_usage=%s, fix_usage_h=%s, fix_usage_v=%s).",
                    "set" if target_density is not None else "keep",
                    "set" if target_demand is not None else "clear",
                    "set" if raw_wire_demand_map is not None else "keep",
                    "set" if target_density_h is not None else "keep",
                    "set" if target_density_v is not None else "keep",
                    "set" if target_demand_h is not None else "keep",
                    "set" if target_demand_v is not None else "keep",
                    "set" if raw_wire_demand_map_h is not None else "keep",
                    "set" if raw_wire_demand_map_v is not None else "keep",
                    "set" if supply_original is not None else "keep",
                    "set" if supply_original_h is not None else "keep",
                    "set" if supply_original_v is not None else "keep",
                    "set" if fix_usage_map is not None else "keep",
                    "set" if fix_usage_map_h is not None else "keep",
                    "set" if fix_usage_map_v is not None else "keep",
                )

    def update_per_net_topology(self, topo_cache=None, topo_stats=None):
        self.per_net_topology_cache = topo_cache
        self.per_net_topology_stats = topo_stats
        self._soft_l_debug_call_count = 0
        self._cached_edge_net_ids = None
        self._cached_edge_net_ids_key = None
        self._clear_soft_edge_topology_cache()
        self._clear_fallback_l_directions_cache()
        self._clear_ggr_fast_path_cache()
        if hasattr(self.segment_builder, "reset_cache"):
            self.segment_builder.reset_cache()
        if isinstance(self.cached_soft_debug, dict):
            self.cached_soft_debug.update(
                {
                    "per_net_topology_cache_present": topo_cache is not None,
                    "per_net_topology_nets": None if topo_stats is None else int(topo_stats.get("num_nets_with_topology", 0)),
                    "per_net_topology_segments_h": None if topo_stats is None else int(topo_stats.get("num_segments_h", 0)),
                    "per_net_topology_segments_v": None if topo_stats is None else int(topo_stats.get("num_segments_v", 0)),
                }
            )
        if topo_stats is None:
            if self.log_verbose >= 2:
                logger.info("Cleared per-net topology cache for L-shape routability op.")
            return
        if self.log_verbose >= 2:
            logger.info(
                "Updated per-net topology cache for L-shape routability op: nets=%d segments_h=%d segments_v=%d invalid_wires=%d",
                int(topo_stats.get("num_nets_with_topology", 0)),
                int(topo_stats.get("num_segments_h", 0)),
                int(topo_stats.get("num_segments_v", 0)),
                int(topo_stats.get("invalid_wire_count", 0)),
            )

    def update_same_net_topology(self, topo_cache=None, topo_stats=None):
        self.update_per_net_topology(topo_cache=topo_cache, topo_stats=topo_stats)

    def _clear_ggr_fast_path_cache(self):
        self._ggr_vertex_x_cache = None
        self._ggr_vertex_y_cache = None
        self._ggr_flat_from_cache = None
        self._ggr_flat_to_cache = None
        self._ggr_l_direction_cache = None

    def _tensor_device_cache_key(self, tensor, device, dtype=None):
        if not isinstance(tensor, torch.Tensor):
            return None
        values = tensor.detach()
        target_dtype = dtype if dtype is not None else values.dtype
        return (
            str(values.dtype),
            str(values.device),
            int(values.numel()),
            int(values.data_ptr()) if values.numel() > 0 else 0,
            int(getattr(tensor, "_version", 0)),
            str(device),
            str(target_dtype),
        )

    def _cached_tensor_to_device(self, tensor, device, cache_attr, dtype=None):
        key = self._tensor_device_cache_key(tensor, device, dtype=dtype)
        cache = getattr(self, cache_attr, None)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache["tensor"]
        value = tensor.detach().to(
            device=device,
            dtype=dtype if dtype is not None else tensor.dtype,
        ).contiguous()
        setattr(self, cache_attr, {"key": key, "tensor": value})
        return value

    def _cached_vertex_gather(self, relate, device, cache_attr):
        key = self._tensor_device_cache_key(relate, device, dtype=torch.long)
        key = key + (bool(self.deterministic_flag),)
        cache = getattr(self, cache_attr, None)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache

        indices = relate.detach().to(device=device, dtype=torch.long).contiguous()
        plan = _build_gather_plan(indices) if self.deterministic_flag else None
        cache = {"key": key, "indices": indices, "plan": plan}
        setattr(self, cache_attr, cache)
        return cache

    def _gather_vertices_from_pins(self, pin_coord, relate, cache_attr):
        cache = self._cached_vertex_gather(relate, pin_coord.device, cache_attr)
        if self.deterministic_flag:
            return _deterministic_gather_1d(pin_coord, cache["plan"]).contiguous()
        return pin_coord.index_select(0, cache["indices"]).contiguous()

    def _try_ggr_topology_fast_path(self, pin_pos, steiner_topo_op):
        if not bool(getattr(self.params, "l_shape_use_ggr_topology", False)):
            return None

        pin_relate_x = getattr(steiner_topo_op, "pin_relate_x", None)
        pin_relate_y = getattr(steiner_topo_op, "pin_relate_y", None)
        flat_pin_from = getattr(steiner_topo_op, "flat_pin_from", None)
        flat_pin_to = getattr(steiner_topo_op, "flat_pin_to", None)
        edge_l_directions = getattr(steiner_topo_op, "edge_l_directions", None)
        num_vertices = getattr(steiner_topo_op, "num_vertices", None)
        if (
            not isinstance(pin_relate_x, torch.Tensor)
            or not isinstance(pin_relate_y, torch.Tensor)
            or not isinstance(flat_pin_from, torch.Tensor)
            or not isinstance(flat_pin_to, torch.Tensor)
            or edge_l_directions is not None and not isinstance(edge_l_directions, torch.Tensor)
            or num_vertices is None
        ):
            return None

        num_vertices = int(num_vertices)
        num_pins = pin_pos.numel() // 2
        if num_vertices <= 0 or pin_relate_x.numel() < num_vertices or pin_relate_y.numel() < num_vertices:
            return None

        try:
            with profile_scope(
                self.profile_enabled,
                "l_shape_op.ggr_topology_fast_path",
                tensor=pin_pos,
                logger=logger,
                vertices=num_vertices,
            ):
                relate_x = pin_relate_x[:num_vertices]
                relate_y = pin_relate_y[:num_vertices]
                pin_x = pin_pos[:num_pins]
                pin_y = pin_pos[num_pins:]
                newx = self._gather_vertices_from_pins(
                    pin_x,
                    relate_x,
                    "_ggr_vertex_x_cache",
                )
                newy = self._gather_vertices_from_pins(
                    pin_y,
                    relate_y,
                    "_ggr_vertex_y_cache",
                )

                device = pin_pos.device
                flat_pin_from = self._cached_tensor_to_device(
                    flat_pin_from,
                    device,
                    "_ggr_flat_from_cache",
                )
                flat_pin_to = self._cached_tensor_to_device(
                    flat_pin_to,
                    device,
                    "_ggr_flat_to_cache",
                )
                if edge_l_directions is not None:
                    edge_l_directions = self._cached_tensor_to_device(
                        edge_l_directions,
                        device,
                        "_ggr_l_direction_cache",
                    )

            if self.log_verbose >= 1 and not self._ggr_fast_path_log_emitted:
                logger.info(
                    "Using GGR topology GPU fast path for L-shape vertex gather: vertices=%d edges=%d deterministic=%s",
                    num_vertices,
                    int(flat_pin_from.numel()),
                    self.deterministic_flag,
                )
                self._ggr_fast_path_log_emitted = True
            return newx, newy, flat_pin_from, flat_pin_to, edge_l_directions
        except Exception as exc:
            if not self._ggr_fast_path_warning_emitted:
                logger.warning(
                    "Falling back to CPU Steiner topology path after GGR topology fast path failed: %s",
                    exc,
                )
                self._ggr_fast_path_warning_emitted = True
            return None

    def _build_vertex_to_net(self, steiner_topo_op, num_vertices):
        """Build vertex->net mapping for pins and Steiner points."""
        vertex_to_net = torch.full((num_vertices,), -1, dtype=torch.int32)
        num_pins = self.placedb.num_pins
        pin2net = self.placedb.pin2net_map
        if hasattr(pin2net, "__len__") and len(pin2net) >= num_pins:
            vertex_to_net[:num_pins] = torch.tensor(pin2net[:num_pins], dtype=torch.int32)

        net_steiner_start = getattr(steiner_topo_op, "net_steiner_start", None)
        if net_steiner_start is not None:
            if hasattr(net_steiner_start, "cpu"):
                ns = net_steiner_start.cpu().numpy()
            else:
                ns = net_steiner_start
            if len(ns) >= 2:
                for net_id in range(len(ns) - 1):
                    s = int(ns[net_id])
                    e = int(ns[net_id + 1])
                    if s < e and s < num_vertices:
                        vertex_to_net[s:min(e, num_vertices)] = net_id
        return vertex_to_net

    def _tensor_sample_signature(self, tensor, sample_size=8):
        if not isinstance(tensor, torch.Tensor):
            if not hasattr(tensor, "__len__"):
                return (0, 0, 0, 0, 0)
            numel = int(len(tensor))
            if numel == 0:
                return (0, 0, 0, 0, 0)
            sample = min(int(sample_size), numel)
            front = sum(int(tensor[idx]) for idx in range(sample))
            back = sum(int(tensor[numel - sample + idx]) for idx in range(sample))
            return (numel, 0, 0, int(front), int(back))

        values = tensor.detach()
        numel = int(values.numel())
        if numel == 0:
            return (0, int(values.data_ptr()), int(getattr(tensor, "_version", 0)), 0, 0)

        sample = min(int(sample_size), numel)
        front_values = values[:sample]
        back_values = values[-sample:]
        if front_values.device.type != "cpu":
            front_values = front_values.cpu()
        if back_values.device.type != "cpu":
            back_values = back_values.cpu()
        return (
            numel,
            int(values.data_ptr()),
            int(getattr(tensor, "_version", 0)),
            int(front_values.to(dtype=torch.int64).sum().item()),
            int(back_values.to(dtype=torch.int64).sum().item()),
        )

    def _edge_net_ids_cache_key(self, steiner_topo_op, flat_pin_from, flat_pin_to, num_vertices):
        net_steiner_start = getattr(steiner_topo_op, "net_steiner_start", None)
        return (
            int(num_vertices),
            self._tensor_sample_signature(flat_pin_from),
            self._tensor_sample_signature(flat_pin_to),
            self._tensor_sample_signature(net_steiner_start),
        )

    def _clear_soft_edge_topology_cache(self):
        self._cached_soft_edge_topology = None
        self._cached_soft_edge_topology_key = None

    def _clear_fallback_l_directions_cache(self):
        self._cached_fallback_l_directions = None
        self._cached_fallback_l_directions_key = None

    def _get_fallback_l_directions(self, num_edges, fallback_direction, device):
        key = (int(num_edges), int(fallback_direction), str(device))
        cached = self._cached_fallback_l_directions
        if (
            key == self._cached_fallback_l_directions_key
            and isinstance(cached, torch.Tensor)
            and int(cached.numel()) == int(num_edges)
            and cached.device == device
        ):
            return cached
        l_directions = torch.full(
            (int(num_edges),),
            int(fallback_direction),
            dtype=torch.int32,
            device=device,
        )
        self._cached_fallback_l_directions_key = key
        self._cached_fallback_l_directions = l_directions
        return l_directions

    def _soft_edge_topology_cache_key(self, flat_pin_from, flat_pin_to, num_vertices, device):
        return (
            int(num_vertices),
            str(device),
            self._tensor_sample_signature(flat_pin_from),
            self._tensor_sample_signature(flat_pin_to),
        )

    def _get_soft_edge_topology(self, flat_pin_from, flat_pin_to, num_vertices, device):
        cache_key = self._soft_edge_topology_cache_key(
            flat_pin_from,
            flat_pin_to,
            num_vertices,
            device,
        )
        cached = self._cached_soft_edge_topology
        if (
            cache_key == self._cached_soft_edge_topology_key
            and isinstance(cached, dict)
            and int(cached.get("num_edges", -1)) == int(flat_pin_from.numel())
        ):
            return cached

        if flat_pin_from.device != device:
            flat_pin_from = flat_pin_from.to(device)
        if flat_pin_to.device != device:
            flat_pin_to = flat_pin_to.to(device)

        valid_mask = (
            (flat_pin_from >= 0)
            & (flat_pin_to >= 0)
            & (flat_pin_from < num_vertices)
            & (flat_pin_to < num_vertices)
        )
        if not valid_mask.any():
            topology = {
                "valid": False,
                "num_edges": int(flat_pin_from.numel()),
                "num_valid": 0,
                "all_valid": False,
            }
            self._cached_soft_edge_topology_key = cache_key
            self._cached_soft_edge_topology = topology
            return topology

        all_valid = bool(valid_mask.all().item())
        if all_valid:
            valid_from = flat_pin_from
            valid_to = flat_pin_to
            valid_mask_cpu = None
        else:
            valid_from = flat_pin_from[valid_mask]
            valid_to = flat_pin_to[valid_mask]
            valid_mask_cpu = valid_mask.cpu() if valid_mask.device.type != "cpu" else valid_mask

        topology = {
            "valid": True,
            "num_edges": int(flat_pin_from.numel()),
            "num_valid": int(valid_from.numel()),
            "all_valid": all_valid,
            "valid_mask": valid_mask,
            "valid_mask_cpu": valid_mask_cpu,
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
        if self.deterministic_flag:
            topology["valid_from_gather_plan"] = _build_gather_plan(valid_from)
            topology["valid_to_gather_plan"] = _build_gather_plan(valid_to)

        self._cached_soft_edge_topology_key = cache_key
        self._cached_soft_edge_topology = topology
        return topology

    def _build_edge_net_ids(self, steiner_topo_op, flat_pin_from, flat_pin_to, num_vertices):
        vertex_to_net = self._build_vertex_to_net(steiner_topo_op, num_vertices)
        flat_pin_from_cpu = flat_pin_from.cpu() if isinstance(flat_pin_from, torch.Tensor) and flat_pin_from.device.type != "cpu" else flat_pin_from
        flat_pin_to_cpu = flat_pin_to.cpu() if isinstance(flat_pin_to, torch.Tensor) and flat_pin_to.device.type != "cpu" else flat_pin_to
        edge_net_ids = torch.full((flat_pin_from_cpu.numel(),), -1, dtype=torch.int32)
        valid_mask = (
            (flat_pin_from_cpu >= 0)
            & (flat_pin_to_cpu >= 0)
            & (flat_pin_from_cpu < num_vertices)
            & (flat_pin_to_cpu < num_vertices)
        )
        if not valid_mask.any():
            return edge_net_ids

        net_from = vertex_to_net[flat_pin_from_cpu[valid_mask]]
        net_to = vertex_to_net[flat_pin_to_cpu[valid_mask]]
        chosen_net = torch.where(net_from >= 0, net_from, net_to)
        edge_net_ids[valid_mask] = chosen_net
        return edge_net_ids

    def _get_edge_net_ids(self, steiner_topo_op, flat_pin_from, flat_pin_to, num_vertices):
        cache_key = self._edge_net_ids_cache_key(
            steiner_topo_op,
            flat_pin_from,
            flat_pin_to,
            num_vertices,
        )
        cached = self._cached_edge_net_ids
        if (
            cache_key == self._cached_edge_net_ids_key
            and isinstance(cached, torch.Tensor)
            and int(cached.numel()) == int(flat_pin_from.numel())
        ):
            return cached

        edge_net_ids = self._build_edge_net_ids(
            steiner_topo_op,
            flat_pin_from,
            flat_pin_to,
            num_vertices,
        )
        self._cached_edge_net_ids_key = cache_key
        self._cached_edge_net_ids = edge_net_ids.detach().cpu()
        return self._cached_edge_net_ids

    def _prepare_soft_map(self, candidate, device, dtype):
        if not isinstance(candidate, torch.Tensor):
            return None
        tensor = candidate.detach()
        if tensor.device != device:
            tensor = tensor.to(device)
        tensor = tensor.to(dtype=dtype)
        if tensor.shape != (self.num_bins_x, self.num_bins_y):
            return None
        return torch.clamp(tensor, min=0.0)

    def _compose_soft_l_cost_map(self, supply, target_demand, cached_density, device, dtype):
        if supply is None:
            candidates = [cached_density, target_demand]
            for candidate in candidates:
                if candidate is not None:
                    return candidate, torch.zeros_like(candidate), torch.tensor(0.0, dtype=dtype, device=device)
            zero = torch.zeros((self.num_bins_x, self.num_bins_y), dtype=dtype, device=device)
            return zero, zero, torch.tensor(0.0, dtype=dtype, device=device)

        denom = supply.clamp_min(1e-6)
        cost_map = torch.zeros((self.num_bins_x, self.num_bins_y), dtype=dtype, device=device)
        hotspot_map = torch.zeros_like(cost_map)
        overflow_ratio = torch.tensor(0.0, dtype=dtype, device=device)

        if target_demand is not None:
            demand_ratio = target_demand / denom
            demand_overflow = torch.relu(target_demand - supply) / denom
            cost_map = cost_map + demand_overflow + self.soft_l_background_weight * demand_ratio
            hotspot_map = hotspot_map + (demand_overflow > 0).to(dtype)
            overflow_ratio = torch.maximum(
                overflow_ratio,
                torch.sum(torch.relu(target_demand - supply)) / supply.sum().clamp_min(1e-6),
            )

        if cached_density is not None and self.soft_l_self_overflow_weight > 0.0:
            density_ratio = cached_density / denom
            density_overflow = torch.relu(cached_density - supply) / denom
            cost_map = cost_map + self.soft_l_self_overflow_weight * (
                density_overflow + self.soft_l_background_weight * density_ratio
            )
            hotspot_map = hotspot_map + self.soft_l_self_overflow_weight * (density_overflow > 0).to(dtype)
            overflow_ratio = torch.maximum(
                overflow_ratio,
                torch.sum(torch.relu(cached_density - supply)) / supply.sum().clamp_min(1e-6),
            )

        return cost_map, hotspot_map, overflow_ratio

    def _get_soft_l_scoring_maps(self, device, dtype):
        """
        Build legacy Soft-L heuristic maps for missing per-net topology fallback.

        Each leg is scored against a direction-matched resource map:
        - horizontal legs use H maps
        - vertical legs use V maps

        The fallback keeps the old Soft-L congestion heuristic:
        - routing-oracle overflow/background term
        - cached soft-L density self-feedback
        - hotspot count penalty derived from overflow bins

        Resolver prior and near-tie hard switching are still disabled
        by the per-net topology mode gates outside this helper.
        """
        previous_soft_debug = self.cached_soft_debug if isinstance(self.cached_soft_debug, dict) else {}
        total_supply = self._prepare_soft_map(getattr(self.density_op, "target_density", None), device, dtype)
        total_demand = self._prepare_soft_map(getattr(self.density_op, "target_demand", None), device, dtype)
        total_cached = self._prepare_soft_map(self.cached_density_map, device, dtype)
        supply_h = self._prepare_soft_map(self.target_density_h, device, dtype)
        supply_v = self._prepare_soft_map(self.target_density_v, device, dtype)
        demand_h = self._prepare_soft_map(self.target_demand_h, device, dtype)
        demand_v = self._prepare_soft_map(self.target_demand_v, device, dtype)
        cached_h = self._prepare_soft_map(self.cached_density_map_h, device, dtype)
        cached_v = self._prepare_soft_map(self.cached_density_map_v, device, dtype)

        total_cost, total_hotspot, total_ratio = self._compose_soft_l_cost_map(
            total_supply, total_demand, total_cached, device, dtype
        )

        target_demand_supply_ratio = None
        if total_demand is not None and total_supply is not None:
            total_supply_sum = total_supply.sum().clamp_min(1e-6)
            target_demand_supply_ratio = float(
                (total_demand.sum() / total_supply_sum).detach().item()
            )

        if supply_h is None and demand_h is None and cached_h is None:
            cost_map_h, hotspot_map_h, ratio_h = total_cost, total_hotspot, total_ratio
        else:
            cost_map_h, hotspot_map_h, ratio_h = self._compose_soft_l_cost_map(
                supply_h, demand_h, cached_h, device, dtype
            )

        if supply_v is None and demand_v is None and cached_v is None:
            cost_map_v, hotspot_map_v, ratio_v = total_cost, total_hotspot, total_ratio
        else:
            cost_map_v, hotspot_map_v, ratio_v = self._compose_soft_l_cost_map(
                supply_v, demand_v, cached_v, device, dtype
            )

        effective_hotspot_weight = self.soft_l_hotspot_weight
        overflow_ratio = torch.maximum(total_ratio, torch.maximum(ratio_h, ratio_v))
        if self.soft_l_hotspot_ramp_ratio > 0.0:
            scale = torch.clamp(overflow_ratio / self.soft_l_hotspot_ramp_ratio, min=0.0, max=1.0)
            effective_hotspot_weight = self.soft_l_hotspot_weight * float(scale.item())

        def build_overflow_map(demand, supply, fallback):
            if demand is None or supply is None:
                return fallback.detach() if isinstance(fallback, torch.Tensor) else None
            return (torch.relu(demand - supply) / supply.clamp_min(1e-6)).detach()

        total_ggr_overflow = None
        if total_demand is not None and total_supply is not None:
            total_ggr_overflow = torch.relu(total_demand - total_supply) / total_supply.clamp_min(1e-6)
        ggr_overflow_h = build_overflow_map(demand_h, supply_h, total_ggr_overflow)
        ggr_overflow_v = build_overflow_map(demand_v, supply_v, total_ggr_overflow)

        self.cached_soft_debug = {
            "cost_map_h": cost_map_h.detach(),
            "cost_map_v": cost_map_v.detach(),
            "hotspot_map_h": hotspot_map_h.detach(),
            "hotspot_map_v": hotspot_map_v.detach(),
            "ggr_overflow_h": ggr_overflow_h,
            "ggr_overflow_v": ggr_overflow_v,
            "effective_hotspot_weight": effective_hotspot_weight,
            "overflow_ratio": float(overflow_ratio.detach().item()),
            "target_demand_supply_ratio": target_demand_supply_ratio,
            "per_net_topology_fallback_mode": "legacy_soft_l_heuristic",
        }
        for key, value in previous_soft_debug.items():
            if key not in self.cached_soft_debug and not isinstance(value, torch.Tensor):
                self.cached_soft_debug[key] = value

        return cost_map_h, cost_map_v, hotspot_map_h, hotspot_map_v, effective_hotspot_weight

    def _coord_to_bin_index(self, coord, low, bin_size, num_bins):
        idx = torch.floor((coord - low) / bin_size).to(torch.long)
        return idx.clamp_(0, num_bins - 1)

    def _horizontal_cost(self, prefix_x, y_idx, x_idx1, x_idx2):
        x_lo = torch.minimum(x_idx1, x_idx2)
        x_hi = torch.maximum(x_idx1, x_idx2)
        hi = prefix_x[x_hi, y_idx]
        lo = torch.where(x_lo > 0, prefix_x[x_lo - 1, y_idx], torch.zeros_like(hi))
        return hi - lo

    def _vertical_cost(self, prefix_y, x_idx, y_idx1, y_idx2):
        y_lo = torch.minimum(y_idx1, y_idx2)
        y_hi = torch.maximum(y_idx1, y_idx2)
        hi = prefix_y[x_idx, y_hi]
        lo = torch.where(y_lo > 0, prefix_y[x_idx, y_lo - 1], torch.zeros_like(hi))
        return hi - lo

    def _per_net_topology_diag_dir(self):
        result_dir = str(getattr(self.params, "result_dir", "."))
        diag_dir = os.path.join(result_dir, "per_net_topology_diag")
        os.makedirs(diag_dir, exist_ok=True)
        return diag_dir

    def _plot_per_net_topology_diagnostic_samples(
        self,
        topo_debug_stats,
        *,
        route_bin_size_x,
        route_bin_size_y,
        output_path,
        max_samples=6,
    ):
        import matplotlib.pyplot as plt

        topo_cache = self.per_net_topology_cache if isinstance(self.per_net_topology_cache, dict) else {}
        net_topologies = topo_cache.get("net_topologies", {})
        samples = []
        for bucket_name in ("sample_missing_topology", "sample_zero_support", "sample_weak_support"):
            for sample in topo_debug_stats.get(bucket_name, []) or []:
                sample = dict(sample)
                sample["bucket"] = bucket_name
                samples.append(sample)
                if len(samples) >= int(max_samples):
                    break
            if len(samples) >= int(max_samples):
                break
        if not samples:
            return None

        ncols = 2
        nrows = (len(samples) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows), constrained_layout=True)
        if hasattr(axes, "flat"):
            axes_list = list(axes.flat)
        else:
            axes_list = [axes]

        margin_x = 2.5 * float(route_bin_size_x)
        margin_y = 2.5 * float(route_bin_size_y)

        for ax, sample in zip(axes_list, samples):
            x1 = float(sample.get("x1", 0.0))
            y1 = float(sample.get("y1", 0.0))
            x2 = float(sample.get("x2", 0.0))
            y2 = float(sample.get("y2", 0.0))
            net_id = int(sample.get("net_id", -1))
            record = net_topologies.get(net_id)

            ax.plot([x1, x2], [y1, y2], "k--", linewidth=1.0, alpha=0.8, label="diag")
            ax.plot([x1, x2], [y1, y1], color="tab:red", linewidth=1.6, alpha=0.8, label="HV-H")
            ax.plot([x2, x2], [y1, y2], color="tab:red", linewidth=1.6, alpha=0.8, label="HV-V")
            ax.plot([x1, x1], [y1, y2], color="tab:purple", linewidth=1.6, alpha=0.8, label="VH-V")
            ax.plot([x1, x2], [y2, y2], color="tab:purple", linewidth=1.6, alpha=0.8, label="VH-H")

            view_x_lo = min(x1, x2)
            view_x_hi = max(x1, x2)
            view_y_lo = min(y1, y2)
            view_y_hi = max(y1, y2)

            if isinstance(record, dict):
                for row_idx, intervals in record.get("horizontal_by_row", {}).items():
                    seg_y = self.yl + (float(row_idx) + 0.5) * float(route_bin_size_y)
                    for int_lo, int_hi in intervals:
                        seg_x1 = self.xl + (float(int_lo) + 0.5) * float(route_bin_size_x)
                        seg_x2 = self.xl + (float(int_hi) + 0.5) * float(route_bin_size_x)
                        ax.plot([seg_x1, seg_x2], [seg_y, seg_y], color="tab:blue", linewidth=2.0, alpha=0.9)
                        view_x_lo = min(view_x_lo, seg_x1, seg_x2)
                        view_x_hi = max(view_x_hi, seg_x1, seg_x2)
                        view_y_lo = min(view_y_lo, seg_y)
                        view_y_hi = max(view_y_hi, seg_y)
                for col_idx, intervals in record.get("vertical_by_col", {}).items():
                    seg_x = self.xl + (float(col_idx) + 0.5) * float(route_bin_size_x)
                    for int_lo, int_hi in intervals:
                        seg_y1 = self.yl + (float(int_lo) + 0.5) * float(route_bin_size_y)
                        seg_y2 = self.yl + (float(int_hi) + 0.5) * float(route_bin_size_y)
                        ax.plot([seg_x, seg_x], [seg_y1, seg_y2], color="tab:green", linewidth=2.0, alpha=0.9)
                        view_x_lo = min(view_x_lo, seg_x)
                        view_x_hi = max(view_x_hi, seg_x)
                        view_y_lo = min(view_y_lo, seg_y1, seg_y2)
                        view_y_hi = max(view_y_hi, seg_y1, seg_y2)

            ax.set_xlim(view_x_lo - margin_x, view_x_hi + margin_x)
            ax.set_ylim(view_y_lo - margin_y, view_y_hi + margin_y)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            title = (
                f"net={sample.get('net_name', f'net_{net_id}')} edge={int(sample.get('edge_id', -1))}\n"
                f"reason={sample.get('reason', 'unknown')} obs={int(sample.get('observed_piece_count', 0))} "
                f"support={float(sample.get('total_support', 0.0)):.3e}"
            )
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        for ax in axes_list[len(samples):]:
            ax.axis("off")
        fig.suptitle("Same-Net Topology Missing-Support Samples")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _emit_per_net_topology_diagnostics(
        self,
        *,
        topo_debug_stats,
        missing_count,
        total_count,
        route_bin_size_x,
        route_bin_size_y,
    ):
        diag_dir = self._per_net_topology_diag_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(diag_dir, f"per_net_topology_missing_{stamp}.json")
        png_path = os.path.join(diag_dir, f"per_net_topology_missing_{stamp}.png")

        payload = {
            "timestamp": stamp,
            "missing_count": int(missing_count),
            "total_count": int(total_count),
            "observed_count": int(total_count) - int(missing_count),
            "per_net_topology_mode": True,
            "per_net_topology_sigma": float(self.soft_l_same_net_diag_split_sigma),
            "per_net_topology_min_support": float(self.soft_l_same_net_diag_split_min_support),
            "per_net_topology_max_distance": float(self.soft_l_same_net_diag_split_max_distance),
            "per_net_topology_use_cpp": True,
            "route_bin_size_x": float(route_bin_size_x),
            "route_bin_size_y": float(route_bin_size_y),
            "topo_debug_stats": topo_debug_stats,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        plot_path = self._plot_per_net_topology_diagnostic_samples(
            topo_debug_stats,
            route_bin_size_x=route_bin_size_x,
            route_bin_size_y=route_bin_size_y,
            output_path=png_path,
        )
        return json_path, plot_path

    def _compute_soft_l_weights(self, newx, newy, flat_pin_from, flat_pin_to, l_directions, edge_net_ids=None):
        """
        Build soft H/V weights for each edge.

        All diagonal edges are scored by cost_h / cost_v.
        Resolver labels only provide optional soft bias.
        """
        device = newx.device
        dtype = newx.dtype
        num_edges = flat_pin_from.numel()
        weights = torch.zeros((num_edges, 2), dtype=dtype, device=device)
        profile_timer = profile_start(self.profile_enabled, tensor=newx)

        def _finish(result, **fields):
            profile_end(
                self.profile_enabled,
                profile_timer,
                "l_shape_op.soft_l_weights",
                tensor=newx,
                logger=logger,
                edges=num_edges,
                **fields,
            )
            return result

        if num_edges == 0:
            return _finish(weights, diag_edges=0)

        with profile_scope(self.profile_enabled, "soft_l_weights.prepare_edges", tensor=newx, logger=logger):
            if l_directions.device != device:
                l_directions = l_directions.to(device)

            soft_topology = self._get_soft_edge_topology(
                flat_pin_from,
                flat_pin_to,
                len(newx),
                device,
            )
            if not soft_topology.get("valid", False):
                return _finish(weights, valid_edges=0, diag_edges=0)

            valid_mask = soft_topology["valid_mask"]
            valid_from = soft_topology["valid_from"]
            valid_to = soft_topology["valid_to"]
            all_valid_edges = bool(soft_topology.get("all_valid", False))
            valid_l_dir = l_directions if all_valid_edges else l_directions[valid_mask]
            valid_edge_net_ids = None
            if isinstance(edge_net_ids, torch.Tensor):
                edge_net_ids_cpu = edge_net_ids.cpu() if edge_net_ids.device.type != "cpu" else edge_net_ids
                if all_valid_edges:
                    valid_edge_net_ids = edge_net_ids_cpu
                else:
                    valid_mask_cpu = soft_topology.get("valid_mask_cpu")
                    if valid_mask_cpu is None:
                        valid_mask_cpu = valid_mask.cpu() if valid_mask.device.type != "cpu" else valid_mask
                    valid_edge_net_ids = edge_net_ids_cpu[valid_mask_cpu]

            x1 = self._gather_vertices(newx, valid_from, soft_topology.get("valid_from_gather_plan"))
            y1 = self._gather_vertices(newy, valid_from, soft_topology.get("valid_from_gather_plan"))
            x2 = self._gather_vertices(newx, valid_to, soft_topology.get("valid_to_gather_plan"))
            y2 = self._gather_vertices(newy, valid_to, soft_topology.get("valid_to_gather_plan"))

            is_horizontal_line = torch.abs(y1 - y2) < 1e-4
            is_vertical_line = torch.abs(x1 - x2) < 1e-4
            is_straight = is_horizontal_line | is_vertical_line
            is_diagonal = ~is_straight
            diag_edge_count = int(is_diagonal.sum().item())

            valid_weights = torch.zeros((valid_from.numel(), 2), dtype=dtype, device=device)
            valid_weights[:, 0] = 1.0  # straight edge fallback; soft mode only reads diagonal weights

        if is_diagonal.any():
            with profile_scope(
                self.profile_enabled,
                "soft_l_weights.diag_indexing",
                tensor=newx,
                logger=logger,
                diag_edges=diag_edge_count,
            ):
                dx1 = x1[is_diagonal]
                dy1 = y1[is_diagonal]
                dx2 = x2[is_diagonal]
                dy2 = y2[is_diagonal]

                x1_idx = self._coord_to_bin_index(dx1, self.xl, self.bin_size_x, self.num_bins_x)
                y1_idx = self._coord_to_bin_index(dy1, self.yl, self.bin_size_y, self.num_bins_y)
                x2_idx = self._coord_to_bin_index(dx2, self.xl, self.bin_size_x, self.num_bins_x)
                y2_idx = self._coord_to_bin_index(dy2, self.yl, self.bin_size_y, self.num_bins_y)
                diag_l_dir = valid_l_dir[is_diagonal]

            next_soft_l_debug_call_count = self._soft_l_debug_call_count + 1
            should_update_soft_debug = (
                next_soft_l_debug_call_count == 1
                or next_soft_l_debug_call_count % self.soft_l_debug_update_interval == 0
            )

            topo_debug_stats = None
            topo_cost_h = None
            topo_cost_v = None
            topo_observed_mask = None
            missing_count = int(is_diagonal.sum().item())
            total_count = int(is_diagonal.sum().item())
            if isinstance(self.per_net_topology_cache, dict) and valid_edge_net_ids is not None:
                route_grid_shape = self.per_net_topology_cache.get("route_grid_shape", None)
                if route_grid_shape is None:
                    route_num_bins_x = int(getattr(self.placedb, "num_routing_grids_x", self.num_bins_x))
                    route_num_bins_y = int(getattr(self.placedb, "num_routing_grids_y", self.num_bins_y))
                else:
                    route_num_bins_x = int(route_grid_shape[0])
                    route_num_bins_y = int(route_grid_shape[1])
                route_num_bins_x = max(route_num_bins_x, 1)
                route_num_bins_y = max(route_num_bins_y, 1)
                route_bin_size_x = (self.xh - self.xl) / float(route_num_bins_x)
                route_bin_size_y = (self.yh - self.yl) / float(route_num_bins_y)

                diag_mask_cpu = is_diagonal.cpu() if is_diagonal.device.type != "cpu" else is_diagonal
                with profile_scope(
                    self.profile_enabled,
                    "l_shape_op.same_net_topo_scoring",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                    cpp=int(self.per_net_topology_use_cpp),
                ):
                    topo_cost_h, topo_cost_v, topo_observed_mask, topo_debug_stats = compute_diagonal_split_topo_costs(
                        edge_net_ids=valid_edge_net_ids[diag_mask_cpu],
                        x1=dx1.cpu(),
                        y1=dy1.cpu(),
                        x2=dx2.cpu(),
                        y2=dy2.cpu(),
                        topo_cache=self.per_net_topology_cache,
                        xl=self.xl,
                        yl=self.yl,
                        route_bin_size_x=route_bin_size_x,
                        route_bin_size_y=route_bin_size_y,
                        sigma=self.soft_l_same_net_diag_split_sigma,
                        min_support=self.soft_l_same_net_diag_split_min_support,
                        max_distance=self.soft_l_same_net_diag_split_max_distance,
                        device=device,
                        dtype=dtype,
                        use_cpp=self.per_net_topology_use_cpp,
                        profile_enabled=self.profile_enabled,
                        log_verbose=self.log_verbose,
                        collect_stats=should_update_soft_debug,
                    )
                with profile_scope(
                    self.profile_enabled,
                    "soft_l_weights.topo_apply",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                ):
                    topo_observed_mask_cpu = topo_observed_mask.cpu() if topo_observed_mask.device.type != "cpu" else topo_observed_mask
                    missing_count = int((~topo_observed_mask_cpu).sum().item())
                    total_count = int(topo_observed_mask_cpu.numel())

            need_full_base_cost = topo_observed_mask is None or self._debug_hash_active()
            topo_missing_mask = None
            need_missing_base_cost = False
            if topo_observed_mask is not None and missing_count > 0 and not need_full_base_cost:
                topo_missing_mask = ~topo_observed_mask
                need_missing_base_cost = True

            cost_h = topo_cost_h
            cost_v = topo_cost_v
            base_cost_h = None
            base_cost_v = None
            if need_full_base_cost or need_missing_base_cost:
                with profile_scope(
                    self.profile_enabled,
                    "soft_l_weights.map_prefix",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                    fallback_edges=diag_edge_count if need_full_base_cost else missing_count,
                ):
                    cost_map_h, cost_map_v, hotspot_map_h, hotspot_map_v, effective_hotspot_weight = self._get_soft_l_scoring_maps(device, dtype)
                    prefix_x_h = cost_map_h.cumsum(dim=0)
                    prefix_y_v = cost_map_v.cumsum(dim=1)
                    hotspot_prefix_x_h = hotspot_map_h.cumsum(dim=0) if effective_hotspot_weight > 0.0 else None
                    hotspot_prefix_y_v = hotspot_map_v.cumsum(dim=1) if effective_hotspot_weight > 0.0 else None
                    self._log_debug_hash("soft_l.cost_map_h", cost_map_h, diag_edges=diag_edge_count)
                    self._log_debug_hash("soft_l.cost_map_v", cost_map_v, diag_edges=diag_edge_count)

                with profile_scope(
                    self.profile_enabled,
                    "soft_l_weights.base_cost",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                    fallback_edges=diag_edge_count if need_full_base_cost else missing_count,
                ):
                    if need_full_base_cost:
                        bx1_idx = x1_idx
                        by1_idx = y1_idx
                        bx2_idx = x2_idx
                        by2_idx = y2_idx
                    else:
                        bx1_idx = x1_idx[topo_missing_mask]
                        by1_idx = y1_idx[topo_missing_mask]
                        bx2_idx = x2_idx[topo_missing_mask]
                        by2_idx = y2_idx[topo_missing_mask]

                    base_cost_h = self._horizontal_cost(prefix_x_h, by1_idx, bx1_idx, bx2_idx) + self._vertical_cost(
                        prefix_y_v, bx2_idx, by1_idx, by2_idx
                    )
                    base_cost_v = self._vertical_cost(prefix_y_v, bx1_idx, by1_idx, by2_idx) + self._horizontal_cost(
                        prefix_x_h, by2_idx, bx1_idx, bx2_idx
                    )

                    if hotspot_prefix_x_h is not None and hotspot_prefix_y_v is not None:
                        hotspot_h = self._horizontal_cost(hotspot_prefix_x_h, by1_idx, bx1_idx, bx2_idx) + self._vertical_cost(
                            hotspot_prefix_y_v, bx2_idx, by1_idx, by2_idx
                        )
                        hotspot_v = self._vertical_cost(hotspot_prefix_y_v, bx1_idx, by1_idx, by2_idx) + self._horizontal_cost(
                            hotspot_prefix_x_h, by2_idx, bx1_idx, bx2_idx
                        )
                        base_cost_h = base_cost_h + effective_hotspot_weight * hotspot_h
                        base_cost_v = base_cost_v + effective_hotspot_weight * hotspot_v

                    if need_full_base_cost:
                        cost_h = base_cost_h
                        cost_v = base_cost_v
                        self._log_debug_hash("soft_l.base_cost_h", cost_h, diag_edges=diag_edge_count)
                        self._log_debug_hash("soft_l.base_cost_v", cost_v, diag_edges=diag_edge_count)
                        self._log_debug_hash("soft_l.diag_l_dir", diag_l_dir, diag_edges=diag_edge_count)

            if topo_observed_mask is not None:
                with profile_scope(
                    self.profile_enabled,
                    "soft_l_weights.topo_merge",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                    fallback_edges=missing_count,
                ):
                    self._log_debug_hash("soft_l.topo_cost_h", topo_cost_h, diag_edges=diag_edge_count)
                    self._log_debug_hash("soft_l.topo_cost_v", topo_cost_v, diag_edges=diag_edge_count)
                    self._log_debug_hash("soft_l.topo_observed_mask", topo_observed_mask, diag_edges=diag_edge_count)
                    if need_full_base_cost:
                        cost_h = torch.where(topo_observed_mask, topo_cost_h, cost_h)
                        cost_v = torch.where(topo_observed_mask, topo_cost_v, cost_v)
                    elif missing_count > 0:
                        cost_h = topo_cost_h.clone()
                        cost_v = topo_cost_v.clone()
                        cost_h[topo_missing_mask] = base_cost_h
                        cost_v[topo_missing_mask] = base_cost_v
                    else:
                        cost_h = topo_cost_h
                        cost_v = topo_cost_v
                    self._log_debug_hash("soft_l.merged_cost_h", cost_h, diag_edges=diag_edge_count)
                    self._log_debug_hash("soft_l.merged_cost_v", cost_v, diag_edges=diag_edge_count)

            self._soft_l_debug_call_count = next_soft_l_debug_call_count

            raw_cost_h = cost_h
            raw_cost_v = cost_v
            with profile_scope(
                self.profile_enabled,
                "soft_l_weights.softmax_core",
                tensor=newx,
                logger=logger,
                diag_edges=diag_edge_count,
            ):
                if self.soft_l_use_resolver_prior and not self.per_net_topology_mode:
                    cost_h = cost_h - self.soft_l_prior_bias * (diag_l_dir == H_FIRST).to(dtype)
                    cost_v = cost_v - self.soft_l_prior_bias * (diag_l_dir == V_FIRST).to(dtype)

                if self.soft_l_adaptive_tau:
                    if cost_h.numel() == 0:
                        tau = torch.tensor(
                            self.soft_l_temperature, dtype=dtype, device=device
                        )
                    else:
                        # Per-edge adaptive tau: scale by each edge's own cost magnitude
                        # instead of global median gap, avoiding the "chasing-tail" problem
                        # where tau ~ median(gap) guarantees ~50% near-tie edges.
                        per_edge_cost_scale = ((cost_h + cost_v) / 2).clamp(min=1e-6)
                        tau = self.soft_l_adaptive_scale * per_edge_cost_scale
                else:
                    tau = self.soft_l_temperature

                logits = torch.stack((-cost_h / tau, -cost_v / tau), dim=1)
                diag_weights = torch.softmax(logits, dim=1)
                self._log_debug_hash("soft_l.tau", tau, diag_edges=diag_edge_count)
                self._log_debug_hash("soft_l.diag_weights", diag_weights, diag_edges=diag_edge_count)
                max_prob = None
                near_tie_ratio = None
                if self.soft_l_tie_break_delta > 0.0 and not self.per_net_topology_mode:
                    max_prob = diag_weights.max(dim=1).values
                    near_tie = max_prob <= (0.5 + self.soft_l_tie_break_delta)
                    if should_update_soft_debug:
                        near_tie_ratio = float(near_tie.to(dtype).mean().item())
                    if near_tie.any():
                        choose_h = cost_h <= cost_v
                        tie_weights = torch.stack(
                            [choose_h.to(dtype=dtype), (~choose_h).to(dtype=dtype)], dim=1
                        )
                        diag_weights = torch.where(near_tie.unsqueeze(1), tie_weights, diag_weights)

            if isinstance(self.cached_soft_debug, dict) and not should_update_soft_debug:
                self.cached_soft_debug.update(
                    {
                        "diag_edge_count": int(diag_edge_count),
                        "per_net_topology_missing_edges": missing_count,
                        "per_net_topology_fallback_edges": missing_count,
                        "per_net_topology_total_diag_edges": total_count,
                    }
                )

            if should_update_soft_debug and isinstance(self.cached_soft_debug, dict):
                with profile_scope(
                    self.profile_enabled,
                    "soft_l_weights.softmax_stats",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                ):
                    raw_cost_gap = (raw_cost_h - raw_cost_v).abs()
                    biased_cost_gap = (cost_h - cost_v).abs()
                    tau_source_gap = None
                    if self.soft_l_adaptive_tau and raw_cost_gap.numel() > 0:
                        tau_source_gap = torch.quantile(raw_cost_gap, 0.5).clamp(min=1e-6)
                    if max_prob is None:
                        max_prob = diag_weights.max(dim=1).values
                    safe_weights = diag_weights.clamp_min(1e-12)
                    entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
                    pred_dir = torch.where(
                        diag_weights[:, 0] >= diag_weights[:, 1],
                        torch.full_like(diag_l_dir, H_FIRST),
                        torch.full_like(diag_l_dir, V_FIRST),
                    )
                    known_resolver = (diag_l_dir == H_FIRST) | (diag_l_dir == V_FIRST)
                    resolver_agreement_ratio = None
                    if known_resolver.any():
                        resolver_agreement_ratio = float(
                            (pred_dir[known_resolver] == diag_l_dir[known_resolver])
                            .to(dtype)
                            .mean()
                            .item()
                        )

                debug_timer = profile_start(self.profile_enabled, tensor=newx)
                topo_kernel_name = "diag_split_geometric"
                if topo_debug_stats is not None:
                    backend = topo_debug_stats.get("backend", None)
                    if backend == "cpp":
                        topo_kernel_name = "diag_split_geometric_cpp"
                    elif backend == "python":
                        topo_kernel_name = "diag_split_geometric_python"
                self.cached_soft_debug.update(
                    {
                        "diag_edge_count": int(diag_edge_count),
                        "mean_cost_gap": float(biased_cost_gap.mean().item()),
                        "raw_cost_gap_p50": float(torch.quantile(raw_cost_gap, 0.5).item()),
                        "biased_cost_gap_p50": float(torch.quantile(biased_cost_gap, 0.5).item()),
                        "tau_source_gap": None
                        if tau_source_gap is None
                        else float(tau_source_gap.item()),
                        "mean_max_prob": float(max_prob.mean().item()),
                        "mean_entropy": float(entropy.mean().item()),
                        "near_tie_ratio": near_tie_ratio,
                        "resolver_agreement_ratio": resolver_agreement_ratio,
                        "tau": float(tau) if isinstance(tau, (int, float)) else float(tau.mean().item()),
                        "tau_min": float(tau) if isinstance(tau, (int, float)) else float(tau.min().item()),
                        "tau_max": float(tau) if isinstance(tau, (int, float)) else float(tau.max().item()),
                        "adaptive_tau": self.soft_l_adaptive_tau,
                        "per_net_topology_mode": self.per_net_topology_mode,
                        "per_net_topology_kernel": topo_kernel_name,
                        "per_net_topology_sigma": self.soft_l_same_net_diag_split_sigma,
                        "per_net_topology_min_support": self.soft_l_same_net_diag_split_min_support,
                        "per_net_topology_max_distance": self.soft_l_same_net_diag_split_max_distance,
                        "per_net_topology_use_cpp": self.per_net_topology_use_cpp,
                        "soft_l_debug_update_interval": self.soft_l_debug_update_interval,
                        "soft_l_debug_call_count": self._soft_l_debug_call_count,
                    }
                )
                if topo_debug_stats is not None:
                    self.cached_soft_debug.update(
                        {
                            "per_net_topology_diag_edges": int(topo_debug_stats.get("diag_edges", 0)),
                            "per_net_topology_edges_with_topology": int(topo_debug_stats.get("edges_with_topology", 0)),
                            "per_net_topology_edges_with_observed_intervals": int(
                                topo_debug_stats.get("edges_with_observed_intervals", 0)
                            ),
                            "per_net_topology_missing_edges": missing_count,
                            "per_net_topology_fallback_edges": missing_count,
                            "per_net_topology_total_diag_edges": total_count,
                            "per_net_topology_mean_gap": float(topo_debug_stats.get("mean_gap", 0.0)),
                            "per_net_topology_tie_ratio": float(topo_debug_stats.get("tie_ratio", 0.0)),
                            "per_net_topology_zero_zero_edges": int(
                                topo_debug_stats.get("zero_zero_edges", 0)
                            ),
                            "per_net_topology_exact_equal_edges": int(
                                topo_debug_stats.get("exact_equal_edges", 0)
                            ),
                        }
                    )
                else:
                    self.cached_soft_debug.update(
                        {
                            "per_net_topology_missing_edges": missing_count,
                            "per_net_topology_fallback_edges": missing_count,
                            "per_net_topology_total_diag_edges": total_count,
                        }
                    )
                profile_end(
                    self.profile_enabled,
                    debug_timer,
                    "soft_l_weights.debug_update",
                    tensor=newx,
                    logger=logger,
                    diag_edges=diag_edge_count,
                )
            with profile_scope(
                self.profile_enabled,
                "soft_l_weights.scatter_diag",
                tensor=newx,
                logger=logger,
                diag_edges=diag_edge_count,
            ):
                valid_weights[is_diagonal] = diag_weights

        with profile_scope(
            self.profile_enabled,
            "soft_l_weights.scatter_valid",
            tensor=newx,
            logger=logger,
            valid_edges=int(valid_from.numel()),
            diag_edges=diag_edge_count,
        ):
            weights[valid_mask] = valid_weights
        return _finish(weights, valid_edges=int(valid_from.numel()), diag_edges=diag_edge_count)

    def _prepare_segment_weights(self, segment_result, device):
        segment_weight = segment_result.get('segment_weight', None)
        if segment_weight is None:
            return None
        if segment_weight.device != device:
            segment_weight = segment_weight.to(device)
        return segment_weight

    def _compute_density_map_from_subset(self, segment_result, segment_size_x, segment_size_y, segment_weight=None, mask=None):
        if mask is None:
            llx = segment_result['segment_llx']
            lly = segment_result['segment_lly']
            sx = segment_size_x
            sy = segment_size_y
            sw = segment_weight
        else:
            if not mask.any():
                return torch.zeros(
                    self.num_bins_x, self.num_bins_y,
                    dtype=segment_size_x.dtype, device=segment_size_x.device
                )
            llx = segment_result['segment_llx'][mask]
            lly = segment_result['segment_lly'][mask]
            sx = segment_size_x[mask]
            sy = segment_size_y[mask]
            sw = segment_weight[mask] if segment_weight is not None else None

        segment_pos = build_segment_pos_tensor(llx, lly)
        if self.density_mode == "electric":
            return self.overflow_op.compute_density_map(segment_pos, sx, sy, segment_weight=sw)
        return self.density_op(segment_pos, sx, sy, mode="density", segment_weight=sw)

    def forward(
        self,
        pos,
        steiner_topo_op,
        pin_pos_op,
        use_l_direction=True,
        update_capacity_al_lambda=False,
        placement_iteration_id=None,
    ):
        """
        计算L形segment密度代价
        
        Args:
            pos: cell位置 [num_nodes * 2]
            steiner_topo_op: Steiner树操作对象
            pin_pos_op: pin位置计算操作
            use_l_direction: 是否使用EGR的L方向信息
            
        Returns:
            routability_cost: 可微的代价值
        """
        tt = time.time()
        original_device = pos.device
        profile_timer = profile_start(self.profile_enabled, tensor=pos)

        def _finish(cost, **fields):
            profile_end(
                self.profile_enabled,
                profile_timer,
                "l_shape_op.forward.total",
                tensor=pos,
                logger=logger,
                **fields,
            )
            return cost
        
        # 1. 检查pos是否越界（仅检查movable节点）
        num_nodes = pos.numel() // 2
        num_movable = min(self.placedb.num_movable_nodes, num_nodes)
        pos_x = pos[:num_nodes][:num_movable]
        pos_y = pos[num_nodes:][:num_movable]
        oob_xl = (pos_x < self.xl)
        oob_xh = (pos_x > self.xh)
        oob_yl = (pos_y < self.yl)
        oob_yh = (pos_y > self.yh)
        oob_cnt = (oob_xl | oob_xh | oob_yl | oob_yh).sum().item()
        if oob_cnt > 0:
            max_dx = torch.max(
                torch.clamp(self.xl - pos_x, min=0),
                torch.clamp(pos_x - self.xh, min=0),
            ).max().item()
            max_dy = torch.max(
                torch.clamp(self.yl - pos_y, min=0),
                torch.clamp(pos_y - self.yh, min=0),
            ).max().item()
            logger.warning(
                f"L-shape: pos OOB {oob_cnt}/{num_movable} (max_dx={max_dx:.3f}, max_dy={max_dy:.3f})"
            )

        # 2. 计算pin位置并clamp到边界（防止out-of-bound）
        with profile_scope(self.profile_enabled, "l_shape_op.pin_pos_clamp", tensor=pos, logger=logger):
            pin_pos = pin_pos_op(pos)
            num_pins = pin_pos.numel() // 2
            pin_pos_x = pin_pos[:num_pins].clamp(self.xl, self.xh)
            pin_pos_y = pin_pos[num_pins:].clamp(self.yl, self.yh)
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
        self._log_debug_hash("forward.pos", pos)
        self._log_debug_hash("forward.pin_pos", pin_pos)

        fast_path = self._try_ggr_topology_fast_path(pin_pos, steiner_topo_op)
        if fast_path is not None:
            newx, newy, flat_pin_from, flat_pin_to, fast_l_directions = fast_path
        else:
            fast_l_directions = None
            # steiner_topo_op only supports CPU in the legacy path. The CPU copy
            # keeps autograd connected through ToCopyBackward.
            if pin_pos.is_cuda:
                pin_pos_cpu = pin_pos.cpu()
            else:
                pin_pos_cpu = pin_pos

            with profile_scope(self.profile_enabled, "l_shape_op.steiner_topo", tensor=pos, logger=logger):
                newx, newy = steiner_topo_op(pin_pos_cpu)
            flat_pin_from = steiner_topo_op.flat_pin_from
            flat_pin_to = steiner_topo_op.flat_pin_to
        self._log_debug_hash("forward.pin_relate_x", getattr(steiner_topo_op, "pin_relate_x", None))
        self._log_debug_hash("forward.pin_relate_y", getattr(steiner_topo_op, "pin_relate_y", None))
        self._log_debug_hash("forward.newx", newx)
        self._log_debug_hash("forward.newy", newy)

        if flat_pin_from is None or flat_pin_to is None:
            logger.warning("Steiner edges not available, returning zero cost")
            return _finish(
                torch.zeros(1, dtype=pos.dtype, device=original_device, requires_grad=True),
                segments=0,
            )

        # 5. 获取L方向信息
        if use_l_direction and fast_l_directions is not None:
            l_directions = fast_l_directions
            logger.debug(f"Using GGR L-directions: {len(l_directions)} edges")
        elif use_l_direction and hasattr(steiner_topo_op, 'edge_l_directions') and steiner_topo_op.edge_l_directions is not None:
            l_directions = steiner_topo_op.edge_l_directions
            logger.debug(f"Using EGR L-directions: {len(l_directions)} edges")
        else:
            fallback_direction = UNKNOWN if self.soft_l_assignment else H_FIRST
            l_directions = self._get_fallback_l_directions(
                flat_pin_from.numel(),
                fallback_direction,
                newx.device,
            )
            logger.debug(
                "No L-direction info, using default %s",
                "UNKNOWN" if fallback_direction == UNKNOWN else "H_FIRST",
            )
        self._log_debug_hash("forward.flat_pin_from", flat_pin_from)
        self._log_debug_hash("forward.flat_pin_to", flat_pin_to)
        self._log_debug_hash("forward.l_directions", l_directions)
        self._maybe_save_segment_snapshot(
            newx,
            newy,
            flat_pin_from,
            flat_pin_to,
            l_directions,
        )
        
        soft_l_weights = None
        edge_net_ids = None
        if self.soft_l_assignment:
            if isinstance(self.per_net_topology_cache, dict):
                with profile_scope(
                    self.profile_enabled,
                    "l_shape_op.edge_net_ids",
                    tensor=pos,
                    logger=logger,
                    edges=int(flat_pin_from.numel()),
                ):
                    if self.l_shape_edge_net_ids_cache_flag:
                        edge_net_ids = self._get_edge_net_ids(
                            steiner_topo_op,
                            flat_pin_from,
                            flat_pin_to,
                            newx.numel(),
                        )
                    else:
                        edge_net_ids = self._build_edge_net_ids(
                            steiner_topo_op,
                            flat_pin_from,
                            flat_pin_to,
                            newx.numel(),
                        )
                    self._log_debug_hash("forward.edge_net_ids", edge_net_ids)
            soft_l_weights = self._compute_soft_l_weights(
                newx, newy, flat_pin_from, flat_pin_to, l_directions, edge_net_ids=edge_net_ids
            )
            self._log_debug_hash("forward.soft_l_weights", soft_l_weights)
        else:
            self.cached_soft_debug = None

        # 6. 构建L形segments（在CPU上）
        with profile_scope(self.profile_enabled, "l_shape_op.segment_builder", tensor=pos, logger=logger):
            segment_result = self.segment_builder(
                newx, newy, flat_pin_from, flat_pin_to, l_directions, soft_l_weights=soft_l_weights
            )
        self._log_debug_hash("forward.segment_pos", segment_result.get("segment_pos"))
        self._log_debug_hash("forward.segment_size_x", segment_result.get("segment_size_x"))
        self._log_debug_hash("forward.segment_size_y", segment_result.get("segment_size_y"))
        self._log_debug_hash("forward.segment_is_horizontal", segment_result.get("segment_is_horizontal"))

        num_segments = segment_result['num_segments']
        if num_segments == 0:
            logger.warning("No valid segments built")
            return _finish(
                torch.zeros(1, dtype=pos.dtype, device=original_device, requires_grad=True),
                segments=0,
            )

        # 保存缓存（包含绘图所需的原始数据）
        self.cached_segments = segment_result
        self.cached_segments['newx'] = newx.detach()
        self.cached_segments['newy'] = newy.detach()
        self.cached_segments['flat_from'] = flat_pin_from.detach()
        self.cached_segments['flat_to'] = flat_pin_to.detach()
        self.cached_segments['l_directions'] = l_directions.detach()
        if edge_net_ids is not None:
            self.cached_segments['edge_net_ids'] = edge_net_ids.detach()
        if soft_l_weights is not None:
            self.cached_segments['soft_l_weights'] = soft_l_weights.detach()
        
        # 7. 计算密度代价（在CPU上）
        segment_pos = segment_result['segment_pos']
        segment_size_x = segment_result['segment_size_x']
        segment_size_y = segment_result['segment_size_y']
        segment_weight = self._prepare_segment_weights(segment_result, segment_pos.device)
        self._log_debug_hash("forward.segment_weight", segment_weight)
        
        # 根据模式选择计算方式
        if self.density_mode == "electric":
            # 使用C++/CUDA电势场模型
            # 如果density_op支持CUDA，将数据移到CUDA
            with profile_scope(self.profile_enabled, "l_shape_op.density_cost", tensor=pos, logger=logger, segments=num_segments):
                if original_device.type == 'cuda':
                    segment_pos = segment_pos.to(original_device)
                    segment_size_x = segment_size_x.to(original_device)
                    segment_size_y = segment_size_y.to(original_device)
                    if segment_weight is not None:
                        segment_weight = segment_weight.to(original_device)
                segment_is_horizontal = segment_result.get('segment_is_horizontal', None)
                cost = self.density_op(
                    segment_pos,
                    segment_size_x,
                    segment_size_y,
                    segment_is_horizontal,
                    segment_weight=segment_weight,
                    update_capacity_al_lambda=update_capacity_al_lambda,
                    placement_iteration_id=placement_iteration_id,
                )
                if self._debug_hash_active():
                    try:
                        from dreamplace.ops.routability.l_shape_electric_potential import SegmentElectricPotentialFunction
                        self._log_debug_hash("electric.density_map", SegmentElectricPotentialFunction.last_density_map)
                        self._log_debug_hash("electric.density_map_h", SegmentElectricPotentialFunction.last_density_map_h)
                        self._log_debug_hash("electric.density_map_v", SegmentElectricPotentialFunction.last_density_map_v)
                        self._log_debug_hash("electric.rho_map", SegmentElectricPotentialFunction.last_rho_map)
                        self._log_debug_hash("electric.rho_map_h", SegmentElectricPotentialFunction.last_rho_map_h)
                        self._log_debug_hash("electric.rho_map_v", SegmentElectricPotentialFunction.last_rho_map_v)
                        self._log_debug_hash("electric.field_map_x", SegmentElectricPotentialFunction.last_field_map_x)
                        self._log_debug_hash("electric.field_map_y", SegmentElectricPotentialFunction.last_field_map_y)
                        self._log_debug_hash("electric.overflow_map", SegmentElectricPotentialFunction.last_overflow_map)
                        self._log_debug_hash("electric.energy", SegmentElectricPotentialFunction.last_energy)
                        self._log_debug_hash("electric.energy_h", SegmentElectricPotentialFunction.last_energy_h)
                        self._log_debug_hash("electric.energy_v", SegmentElectricPotentialFunction.last_energy_v)
                    except Exception as exc:
                        logger.warning("LShapeHash iter=%s electric map hash failed: %s", self._debug_hash_iteration, exc)
        else:
            # 使用Python RUDY密度的energy模式
            with profile_scope(self.profile_enabled, "l_shape_op.density_cost", tensor=pos, logger=logger, segments=num_segments):
                cost = self.density_op(
                    segment_pos,
                    segment_size_x,
                    segment_size_y,
                    mode="energy",
                    segment_weight=segment_weight,
                )
        
        # 确保cost在原始设备上
        if cost.device != original_device:
            cost = cost.to(original_device)
        self._log_debug_hash("forward.cost", cost, segments=num_segments)
        
        logger.debug(f"L-shape routability cost: {cost.item():.4f}, "
                    f"{num_segments} segments, {(time.time() - tt) * 1000:.2f} ms")
        
        return _finish(cost, segments=num_segments)
    
    def get_density_map(self, pos, steiner_topo_op, pin_pos_op, use_l_direction=True):
        """
        获取密度图
        
        Returns:
            density_map: [num_bins_x, num_bins_y]
        """
        device = pos.device
        
        # 复用forward的逻辑，但使用density模式
        num_nodes = pos.numel() // 2
        num_movable = min(self.placedb.num_movable_nodes, num_nodes)
        pos_x = pos[:num_nodes][:num_movable]
        pos_y = pos[num_nodes:][:num_movable]
        oob_xl = (pos_x < self.xl)
        oob_xh = (pos_x > self.xh)
        oob_yl = (pos_y < self.yl)
        oob_yh = (pos_y > self.yh)
        oob_cnt = (oob_xl | oob_xh | oob_yl | oob_yh).sum().item()
        if oob_cnt > 0:
            max_dx = torch.max(
                torch.clamp(self.xl - pos_x, min=0),
                torch.clamp(pos_x - self.xh, min=0),
            ).max().item()
            max_dy = torch.max(
                torch.clamp(self.yl - pos_y, min=0),
                torch.clamp(pos_y - self.yh, min=0),
            ).max().item()
            logger.warning(
                f"L-shape: pos OOB {oob_cnt}/{num_movable} (max_dx={max_dx:.3f}, max_dy={max_dy:.3f})"
            )

        pin_pos = pin_pos_op(pos)
        num_pins = pin_pos.numel() // 2
        pin_pos_x = pin_pos[:num_pins].clamp(self.xl, self.xh)
        pin_pos_y = pin_pos[num_pins:].clamp(self.yl, self.yh)
        pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
        
        fast_path = self._try_ggr_topology_fast_path(pin_pos, steiner_topo_op)
        if fast_path is not None:
            newx, newy, flat_pin_from, flat_pin_to, fast_l_directions = fast_path
        else:
            fast_l_directions = None
            # steiner_topo_op requires CPU tensors in the legacy path.
            pin_pos_cpu = pin_pos.cpu() if pin_pos.is_cuda else pin_pos
            newx, newy = steiner_topo_op(pin_pos_cpu)
            flat_pin_from = steiner_topo_op.flat_pin_from
            flat_pin_to = steiner_topo_op.flat_pin_to
        self._log_debug_hash("get_density_map.pin_relate_x", getattr(steiner_topo_op, "pin_relate_x", None))
        self._log_debug_hash("get_density_map.pin_relate_y", getattr(steiner_topo_op, "pin_relate_y", None))
        self._log_debug_hash("get_density_map.newx", newx)
        self._log_debug_hash("get_density_map.newy", newy)

        # 将 legacy CPU topology 结果移回原设备；fast path 已在目标设备上。
        if fast_path is None and device.type == 'cuda':
            newx = newx.to(device)
            newy = newy.to(device)

        if flat_pin_from is None or flat_pin_to is None:
            return torch.zeros(
                self.density_op.num_bins_x, self.density_op.num_bins_y,
                dtype=pos.dtype, device=device
            )
        
        # Keep edge topology tensors on their stable source device. Soft scoring
        # moves local views as needed; this lets overflow probing and forward
        # share edge-net and segment-topology caches across a refresh.
        if use_l_direction and fast_l_directions is not None:
            l_directions = fast_l_directions
        elif use_l_direction and hasattr(steiner_topo_op, 'edge_l_directions') and steiner_topo_op.edge_l_directions is not None:
            l_directions = steiner_topo_op.edge_l_directions
            if l_directions.device != flat_pin_from.device:
                l_directions = l_directions.to(flat_pin_from.device)
        else:
            fallback_direction = UNKNOWN if self.soft_l_assignment else H_FIRST
            l_directions = self._get_fallback_l_directions(
                flat_pin_from.numel(),
                fallback_direction,
                flat_pin_from.device,
            )
        
        soft_l_weights = None
        edge_net_ids = None
        if self.soft_l_assignment:
            if isinstance(self.per_net_topology_cache, dict):
                with profile_scope(
                    self.profile_enabled,
                    "l_shape_op.edge_net_ids",
                    tensor=pos,
                    logger=logger,
                    edges=int(flat_pin_from.numel()),
                ):
                    if self.l_shape_edge_net_ids_cache_flag:
                        edge_net_ids = self._get_edge_net_ids(
                            steiner_topo_op,
                            flat_pin_from,
                            flat_pin_to,
                            newx.numel(),
                        )
                    else:
                        edge_net_ids = self._build_edge_net_ids(
                            steiner_topo_op,
                            flat_pin_from,
                            flat_pin_to,
                            newx.numel(),
                        )
            soft_l_weights = self._compute_soft_l_weights(
                newx, newy, flat_pin_from, flat_pin_to, l_directions, edge_net_ids=edge_net_ids
            )
        else:
            self.cached_soft_debug = None

        segment_result = self.segment_builder(
            newx, newy, flat_pin_from, flat_pin_to, l_directions, soft_l_weights=soft_l_weights
        )
        
        if segment_result['num_segments'] == 0:
            return torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=pos.dtype, device=pos.device
            )
        
        segment_size_x = segment_result['segment_size_x']
        segment_size_y = segment_result['segment_size_y']
        segment_weight = self._prepare_segment_weights(segment_result, segment_size_x.device)
        density_map = self._compute_density_map_from_subset(
            segment_result,
            segment_size_x,
            segment_size_y,
            segment_weight=segment_weight,
            mask=None,
        )
        self._log_debug_hash("get_density_map.total", density_map)
        segment_is_horizontal = segment_result.get('segment_is_horizontal', None)
        if isinstance(segment_is_horizontal, torch.Tensor) and segment_is_horizontal.numel() == segment_result['num_segments']:
            segment_is_horizontal = segment_is_horizontal.to(torch.bool)
            density_map_h = self._compute_density_map_from_subset(
                segment_result,
                segment_size_x,
                segment_size_y,
                segment_weight=segment_weight,
                mask=segment_is_horizontal,
            )
            density_map_v = self._compute_density_map_from_subset(
                segment_result,
                segment_size_x,
                segment_size_y,
                segment_weight=segment_weight,
                mask=~segment_is_horizontal,
            )
            self._log_debug_hash("get_density_map.h", density_map_h)
            self._log_debug_hash("get_density_map.v", density_map_v)
        else:
            density_map_h = None
            density_map_v = None

        # 保存缓存用于绘图（包含原始数据，确保绘图一致性）
        self.cached_segments = segment_result
        self.cached_segments['newx'] = newx.detach()
        self.cached_segments['newy'] = newy.detach()
        self.cached_segments['flat_from'] = flat_pin_from.detach()
        self.cached_segments['flat_to'] = flat_pin_to.detach()
        self.cached_segments['l_directions'] = l_directions.detach()
        if edge_net_ids is not None:
            self.cached_segments['edge_net_ids'] = edge_net_ids.detach()
        if soft_l_weights is not None:
            self.cached_segments['soft_l_weights'] = soft_l_weights.detach()
        self.cached_density_map = density_map
        self.cached_density_map_h = density_map_h
        self.cached_density_map_v = density_map_v
        self._log_debug_hash("density_map.cached_total", self.cached_density_map)
        self._log_debug_hash("density_map.cached_h", self.cached_density_map_h)
        self._log_debug_hash("density_map.cached_v", self.cached_density_map_v)
        return density_map
    
    def get_segments_for_plot(self):
        """
        获取segments信息用于绘图
        
        Returns:
            list of (llx, lly, urx, ury) tuples
        """
        if self.cached_segments is None:
            return []
        
        seg_llx = self.cached_segments['segment_llx'].detach().cpu().numpy()
        seg_lly = self.cached_segments['segment_lly'].detach().cpu().numpy()
        seg_size_x = self.cached_segments['segment_size_x']
        seg_size_y = self.cached_segments['segment_size_y']
        seg_size_x = seg_size_x.detach().cpu().numpy()
        seg_size_y = seg_size_y.detach().cpu().numpy()
        
        segments = []
        for i in range(len(seg_llx)):
            segments.append((
                seg_llx[i],
                seg_lly[i],
                seg_llx[i] + seg_size_x[i],
                seg_lly[i] + seg_size_y[i]
            ))
        return segments


class LShapeRoutabilityMixin:
    """
    Mixin类，用于将L形routability功能集成到PlaceObj中
    
    使用方式:
        class PlaceObj(LShapeRoutabilityMixin, nn.Module):
            def __init__(self, ...):
                ...
                self.init_l_shape_routability(placedb, params)
            
            def obj_fn(self, pos):
                ...
                if self.use_l_shape_routability:
                    cost += self.l_shape_routability_weight * self.l_shape_routability_obj(pos)
    """
    
    def init_l_shape_routability(self, placedb, params, wire_width=None, 
                                  num_bins_x=64, num_bins_y=64,
                                  density_mode="electric", target_density=1.0,
                                  target_demand=None):
        """
        初始化L形routability模块
        
        Args:
            placedb: placement database
            params: parameters
            wire_width: segment线宽
            num_bins_x, num_bins_y: 密度计算的bin数量
            density_mode: 密度计算模式 ("rudy" 或 "electric")
            target_density: 目标密度
            target_demand: 目标需求 (EGR net map)
        """
        self.l_shape_routability_op = LShapeRoutabilityOp(
            placedb, params, wire_width, num_bins_x, num_bins_y,
            density_mode=density_mode, target_density=target_density,
            target_demand=target_demand
        )
        self.use_l_shape_routability = True
        # Placeholder only. Real external weight is calibrated in PlaceObj.
        self.l_shape_routability_weight = 0.0
        logger.info(
            "L-shape routability initialized with placeholder weight %.1e, mode=%s",
            self.l_shape_routability_weight,
            density_mode,
        )
    
    def l_shape_routability_obj(
        self,
        pos,
        use_l_direction=True,
        update_capacity_al_lambda=False,
        placement_iteration_id=None,
    ):
        """
        计算L形routability目标
        
        需要self.op_collections中有steiner_topo_op和pin_pos_op
        """
        if not hasattr(self, 'l_shape_routability_op'):
            raise RuntimeError("L-shape routability not initialized. Call init_l_shape_routability first.")
        
        steiner_topo_op = self.op_collections.steiner_topo_op
        pin_pos_op = self.op_collections.pin_pos_op
        
        return self.l_shape_routability_op(
            pos,
            steiner_topo_op,
            pin_pos_op,
            use_l_direction=use_l_direction,
            update_capacity_al_lambda=update_capacity_al_lambda,
            placement_iteration_id=placement_iteration_id,
        )
    
    def get_l_shape_density_map(self, pos, use_l_direction=True):
        """获取L形密度图"""
        if not hasattr(self, 'l_shape_routability_op'):
            return None
        
        steiner_topo_op = self.op_collections.steiner_topo_op
        pin_pos_op = self.op_collections.pin_pos_op
        
        return self.l_shape_routability_op.get_density_map(
            pos, steiner_topo_op, pin_pos_op, use_l_direction
        )


def create_l_shape_routability_op(placedb, params, wire_width=None, 
                                   num_bins_x=64, num_bins_y=64,
                                   density_mode="electric", target_density=1.0,
                                   target_demand=None):
    """
    工厂函数：创建LShapeRoutabilityOp
    
    Args:
        placedb: placement database
        params: parameters
        wire_width: segment线宽
        num_bins_x, num_bins_y: 密度计算的bin数量
        density_mode: 密度计算模式 ("rudy" 或 "electric")
        target_density: 目标密度
        target_demand: 目标需求 (EGR net map)
        
    Returns:
        LShapeRoutabilityOp
    """
    return LShapeRoutabilityOp(
        placedb, params, wire_width, num_bins_x, num_bins_y,
        density_mode=density_mode, target_density=target_density,
        target_demand=target_demand
    )
