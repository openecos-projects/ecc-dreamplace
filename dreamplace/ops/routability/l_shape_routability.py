##
# @file   l_shape_routability.py
# @brief  L-shape guided routability optimization module
#         Uses EGR L-direction information to build accurate routing density model
#         Supports both Python-based RUDY density and C++/CUDA electric potential
#

import torch
import torch.nn as nn
import logging
import time

from dreamplace.ops.routability.l_shape_segment import (
    LShapeSegmentOp,
    build_segment_pos_tensor,
    build_l_shape_segments_vectorized,
    H_FIRST, V_FIRST, STRAIGHT, FAKE_STRAIGHT, UNKNOWN
)
from dreamplace.ops.routability.segment_density import (
    SegmentDensityOp,
    compute_segment_rudy_density,
    create_segment_density_op
)
from dreamplace.ops.routability.l_shape_electric_potential import (
    LShapeElectricPotential,
    create_l_shape_electric_potential
)
from dreamplace.ops.routability.l_shape_electric_overflow import (
    LShapeElectricOverflow,
    create_l_shape_electric_overflow
)
from dreamplace.ops.routability.same_net_topo_scoring import (
    compute_diagonal_split_topo_costs,
)

logger = logging.getLogger(__name__)


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
        num_bins_x=64,
        num_bins_y=64,
        density_mode="electric",
        target_density=1.0,
        target_demand=None,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
    ):
        """
        Args:
            placedb: placement database
            params: parameters
            wire_width: segment线宽（默认从params获取或使用0）
            num_bins_x, num_bins_y: 密度计算的bin数量
            density_mode: 密度计算模式 ("rudy" 或 "electric")
            target_density: 2D routing supply map (用于electric模式)
            target_demand: 2D routing demand map (EGR net map，用于electric模式的单位标定)
        """
        super(LShapeRoutabilityOp, self).__init__()
        
        self.placedb = placedb
        self.params = params
        self.density_mode = density_mode
        
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
        self.soft_l_use_same_net_topo_scoring = bool(
            getattr(params, "soft_l_use_same_net_topo_scoring", False)
        )
        self.soft_l_same_net_topo_weight = float(
            getattr(params, "soft_l_same_net_topo_weight", 1.0)
        )
        self.soft_l_same_net_topo_only_mode = bool(
            getattr(params, "soft_l_same_net_topo_only_mode", False)
        )
        self.soft_l_same_net_diag_split_sigma = max(
            float(getattr(params, "soft_l_same_net_diag_split_sigma", 1.0)), 1e-6
        )
        self.soft_l_same_net_diag_split_min_support = max(
            float(getattr(params, "soft_l_same_net_diag_split_min_support", 1e-6)), 0.0
        )
        self.soft_l_same_net_diag_split_max_distance = max(
            float(getattr(params, "soft_l_same_net_diag_split_max_distance", 0.0)), 0.0
        )
        self.soft_l_same_net_topo_cpp_accel = bool(
            getattr(params, "soft_l_same_net_topo_cpp_accel", True)
        )

        # L形segment构建器
        self.segment_builder = LShapeSegmentOp(
            wire_width=wire_width,
            use_vectorized=True,
            soft_min_weight=self.soft_l_min_weight,
        )
        if self.soft_l_assignment:
            logger.info(
                "Soft L-assignment enabled: tau=%.3f adaptive_tau=%s adaptive_scale=%.3f prior=%s bias=%.3f min_weight=%.3f tie_delta=%.3f bg=%.3f self_ov=%.3f hotspot=%.3f hotspot_ramp=%.3f",
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
            )
        if self.soft_l_use_same_net_topo_scoring:
            logger.info(
                "Same-net topo scoring enabled: kernel=diag_split_geometric weight=%.3f topo_only=%s sigma=%.4f min_support=%.3e max_distance=%.4f cpp_accel=%s",
                self.soft_l_same_net_topo_weight,
                self.soft_l_same_net_topo_only_mode,
                self.soft_l_same_net_diag_split_sigma,
                self.soft_l_same_net_diag_split_min_support,
                self.soft_l_same_net_diag_split_max_distance,
                self.soft_l_same_net_topo_cpp_accel,
            )
        
        # 根据模式选择密度计算器
        if density_mode == "electric":

            self.density_op = create_l_shape_electric_potential(
                placedb,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                target_density=target_density,
                target_demand=target_demand,
                target_density_h=target_density_h,
                target_density_v=target_density_v,
                target_demand_h=target_demand_h,
                target_demand_v=target_demand_v,
                # padding=1,  # 边界填充
                fast_mode=False
            )
            self.overflow_op = create_l_shape_electric_overflow(
                placedb,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                target_density=target_density,
                # padding=1  # 边界填充
            )
            logger.info(f"Using C++/CUDA electric potential for routability")
        else:
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
            logger.info(f"Using Python RUDY density for routability")
        
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        
        # 存储边界信息用于OOB检查（优先 routing grid）
        self.xl = getattr(placedb, "routing_grid_xl", placedb.xl)
        self.yl = getattr(placedb, "routing_grid_yl", placedb.yl)
        self.xh = getattr(placedb, "routing_grid_xh", placedb.xh)
        self.yh = getattr(placedb, "routing_grid_yh", placedb.yh)
        
        # 缓存
        self.cached_segments = None
        self.cached_soft_debug = None
        self.cached_density_map = None
        self.cached_density_map_h = None
        self.cached_density_map_v = None
        self.same_net_topo_cache = None
        self.same_net_topo_stats = None
        self.target_density_h = target_density_h
        self.target_density_v = target_density_v
        self.target_demand_h = target_demand_h
        self.target_demand_v = target_demand_v

    def update_targets(
        self,
        target_density=None,
        target_demand=None,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
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
        if hasattr(self.density_op, "set_target_demand"):
            self.density_op.set_target_demand(target_demand)
            updated = True
        if hasattr(self.density_op, "set_directional_targets"):
            self.density_op.set_directional_targets(
                target_density_h=target_density_h,
                target_density_v=target_density_v,
                target_demand_h=target_demand_h,
                target_demand_v=target_demand_v,
            )
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

        if updated:
            self.cached_soft_debug = None
            self.cached_density_map = None
            self.cached_density_map_h = None
            self.cached_density_map_v = None
            logger.info(
                "Updated L-shape routing targets (density=%s, demand=%s, density_h=%s, density_v=%s, demand_h=%s, demand_v=%s).",
                "set" if target_density is not None else "keep",
                "set" if target_demand is not None else "clear",
                "set" if target_density_h is not None else "keep",
                "set" if target_density_v is not None else "keep",
                "set" if target_demand_h is not None else "keep",
                "set" if target_demand_v is not None else "keep",
            )

    def update_same_net_topology(self, topo_cache=None, topo_stats=None):
        self.same_net_topo_cache = topo_cache
        self.same_net_topo_stats = topo_stats
        if isinstance(self.cached_soft_debug, dict):
            self.cached_soft_debug.update(
                {
                    "same_net_topo_cache_present": topo_cache is not None,
                    "same_net_topo_nets": None if topo_stats is None else int(topo_stats.get("num_nets_with_topology", 0)),
                    "same_net_topo_segments_h": None if topo_stats is None else int(topo_stats.get("num_segments_h", 0)),
                    "same_net_topo_segments_v": None if topo_stats is None else int(topo_stats.get("num_segments_v", 0)),
                }
            )
        if topo_stats is None:
            logger.info("Cleared same-net topology cache for L-shape routability op.")
            return
        logger.info(
            "Updated same-net topo cache for L-shape routability op: nets=%d segments_h=%d segments_v=%d invalid_wires=%d",
            int(topo_stats.get("num_nets_with_topology", 0)),
            int(topo_stats.get("num_segments_h", 0)),
            int(topo_stats.get("num_segments_v", 0)),
            int(topo_stats.get("invalid_wire_count", 0)),
        )

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
        Build overflow-aware directional maps for soft L path scoring.

        Each leg is scored against a direction-matched resource map:
        - horizontal legs use H maps
        - vertical legs use V maps

        Each `cost_map_*` focuses on excess demand over supply with a small background term
        to break ties in non-overflow regions.
        `hotspot_map_*` counts bins that are currently over capacity, which helps avoid
        traversing any overflow hotspot even if the total excess sum is similar.
        The hotspot penalty is scaled by the current global overflow ratio so it fades
        out automatically once the design is close to feasible.
        """
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
        }

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
        if num_edges == 0:
            return weights

        if flat_pin_from.device != device:
            flat_pin_from = flat_pin_from.to(device)
        if flat_pin_to.device != device:
            flat_pin_to = flat_pin_to.to(device)
        if l_directions.device != device:
            l_directions = l_directions.to(device)

        valid_mask = (flat_pin_from >= 0) & (flat_pin_to >= 0) & (flat_pin_from < len(newx)) & (flat_pin_to < len(newx))
        if not valid_mask.any():
            return weights

        valid_from = flat_pin_from[valid_mask]
        valid_to = flat_pin_to[valid_mask]
        valid_l_dir = l_directions[valid_mask]
        valid_edge_net_ids = None
        if isinstance(edge_net_ids, torch.Tensor):
            valid_mask_cpu = valid_mask.cpu() if valid_mask.device.type != "cpu" else valid_mask
            edge_net_ids_cpu = edge_net_ids.cpu() if edge_net_ids.device.type != "cpu" else edge_net_ids
            valid_edge_net_ids = edge_net_ids_cpu[valid_mask_cpu]

        x1 = newx[valid_from]
        y1 = newy[valid_from]
        x2 = newx[valid_to]
        y2 = newy[valid_to]

        is_horizontal_line = torch.abs(y1 - y2) < 1e-6
        is_vertical_line = torch.abs(x1 - x2) < 1e-6
        is_straight = is_horizontal_line | is_vertical_line
        is_diagonal = ~is_straight

        valid_weights = torch.zeros((valid_from.numel(), 2), dtype=dtype, device=device)
        valid_weights[:, 0] = 1.0  # straight edge fallback; soft mode only reads diagonal weights

        if is_diagonal.any():
            cost_map_h, cost_map_v, hotspot_map_h, hotspot_map_v, effective_hotspot_weight = self._get_soft_l_scoring_maps(device, dtype)
            prefix_x_h = cost_map_h.cumsum(dim=0)
            prefix_y_v = cost_map_v.cumsum(dim=1)
            hotspot_prefix_x_h = hotspot_map_h.cumsum(dim=0) if effective_hotspot_weight > 0.0 else None
            hotspot_prefix_y_v = hotspot_map_v.cumsum(dim=1) if effective_hotspot_weight > 0.0 else None

            dx1 = x1[is_diagonal]
            dy1 = y1[is_diagonal]
            dx2 = x2[is_diagonal]
            dy2 = y2[is_diagonal]

            x1_idx = self._coord_to_bin_index(dx1, self.xl, self.bin_size_x, self.num_bins_x)
            y1_idx = self._coord_to_bin_index(dy1, self.yl, self.bin_size_y, self.num_bins_y)
            x2_idx = self._coord_to_bin_index(dx2, self.xl, self.bin_size_x, self.num_bins_x)
            y2_idx = self._coord_to_bin_index(dy2, self.yl, self.bin_size_y, self.num_bins_y)
            diag_l_dir = valid_l_dir[is_diagonal]

            cost_h = self._horizontal_cost(prefix_x_h, y1_idx, x1_idx, x2_idx) + self._vertical_cost(prefix_y_v, x2_idx, y1_idx, y2_idx)
            cost_v = self._vertical_cost(prefix_y_v, x1_idx, y1_idx, y2_idx) + self._horizontal_cost(prefix_x_h, y2_idx, x1_idx, x2_idx)

            if hotspot_prefix_x_h is not None and hotspot_prefix_y_v is not None:
                hotspot_h = self._horizontal_cost(hotspot_prefix_x_h, y1_idx, x1_idx, x2_idx) + self._vertical_cost(
                    hotspot_prefix_y_v, x2_idx, y1_idx, y2_idx
                )
                hotspot_v = self._vertical_cost(hotspot_prefix_y_v, x1_idx, y1_idx, y2_idx) + self._horizontal_cost(
                    hotspot_prefix_x_h, y2_idx, x1_idx, x2_idx
                )
                cost_h = cost_h + effective_hotspot_weight * hotspot_h
                cost_v = cost_v + effective_hotspot_weight * hotspot_v

            topo_debug_stats = None
            if (
                self.soft_l_use_same_net_topo_scoring
                and isinstance(self.same_net_topo_cache, dict)
                and valid_edge_net_ids is not None
            ):
                route_grid_shape = self.same_net_topo_cache.get("route_grid_shape", None)
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
                topo_cost_h, topo_cost_v, topo_observed_mask, topo_debug_stats = compute_diagonal_split_topo_costs(
                    edge_net_ids=valid_edge_net_ids[diag_mask_cpu],
                    x1=dx1.cpu(),
                    y1=dy1.cpu(),
                    x2=dx2.cpu(),
                    y2=dy2.cpu(),
                    topo_cache=self.same_net_topo_cache,
                    xl=self.xl,
                    yl=self.yl,
                    route_bin_size_x=route_bin_size_x,
                    route_bin_size_y=route_bin_size_y,
                    sigma=self.soft_l_same_net_diag_split_sigma,
                    min_support=self.soft_l_same_net_diag_split_min_support,
                    max_distance=self.soft_l_same_net_diag_split_max_distance,
                    device=device,
                    dtype=dtype,
                    use_cpp=self.soft_l_same_net_topo_cpp_accel,
                )
                if self.soft_l_same_net_topo_only_mode:
                    cost_h = torch.where(topo_observed_mask, topo_cost_h, cost_h)
                    cost_v = torch.where(topo_observed_mask, topo_cost_v, cost_v)
                else:
                    cost_h = cost_h + self.soft_l_same_net_topo_weight * topo_cost_h
                    cost_v = cost_v + self.soft_l_same_net_topo_weight * topo_cost_v

            raw_cost_h = cost_h
            raw_cost_v = cost_v
            if self.soft_l_use_resolver_prior:
                cost_h = cost_h - self.soft_l_prior_bias * (diag_l_dir == H_FIRST).to(dtype)
                cost_v = cost_v - self.soft_l_prior_bias * (diag_l_dir == V_FIRST).to(dtype)

            raw_cost_gap = (raw_cost_h - raw_cost_v).abs()
            biased_cost_gap = (cost_h - cost_v).abs()
            tau_source_gap = None
            if self.soft_l_adaptive_tau:
                if raw_cost_gap.numel() == 0:
                    tau = torch.tensor(
                        self.soft_l_temperature, dtype=dtype, device=device
                    )
                else:
                    tau_source_gap = torch.quantile(raw_cost_gap, 0.5).clamp(min=1e-6)
                    tau = self.soft_l_adaptive_scale * tau_source_gap
            else:
                tau = self.soft_l_temperature

            logits = torch.stack((-cost_h / tau, -cost_v / tau), dim=1)
            diag_weights = torch.softmax(logits, dim=1)
            max_prob = diag_weights.max(dim=1).values
            near_tie_ratio = None
            if self.soft_l_tie_break_delta > 0.0:
                near_tie = max_prob <= (0.5 + self.soft_l_tie_break_delta)
                near_tie_ratio = float(near_tie.to(dtype).mean().item())
                if near_tie.any():
                    choose_h = cost_h <= cost_v
                    tie_weights = torch.stack(
                        [choose_h.to(dtype=dtype), (~choose_h).to(dtype=dtype)], dim=1
                    )
                    diag_weights = torch.where(near_tie.unsqueeze(1), tie_weights, diag_weights)
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
            if isinstance(self.cached_soft_debug, dict):
                topo_kernel_name = "diag_split_geometric"
                if topo_debug_stats is not None:
                    backend = topo_debug_stats.get("backend", None)
                    if backend == "cpp":
                        topo_kernel_name = "diag_split_geometric_cpp"
                    elif backend == "python":
                        topo_kernel_name = "diag_split_geometric_python"
                self.cached_soft_debug.update(
                    {
                        "diag_edge_count": int(is_diagonal.sum().item()),
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
                        "tau": float(tau) if isinstance(tau, (int, float)) else float(tau.item()),
                        "adaptive_tau": self.soft_l_adaptive_tau,
                        "same_net_topo_scoring_enabled": self.soft_l_use_same_net_topo_scoring,
                        "same_net_topo_only_mode": self.soft_l_same_net_topo_only_mode,
                        "same_net_topo_weight": self.soft_l_same_net_topo_weight,
                        "same_net_topo_kernel": topo_kernel_name,
                        "same_net_topo_sigma": self.soft_l_same_net_diag_split_sigma,
                        "same_net_topo_min_support": self.soft_l_same_net_diag_split_min_support,
                        "same_net_topo_max_distance": self.soft_l_same_net_diag_split_max_distance,
                        "same_net_topo_cpp_accel": self.soft_l_same_net_topo_cpp_accel,
                    }
                )
                if topo_debug_stats is not None:
                    self.cached_soft_debug.update(
                        {
                            "same_net_topo_diag_edges": int(topo_debug_stats.get("diag_edges", 0)),
                            "same_net_topo_edges_with_topology": int(topo_debug_stats.get("edges_with_topology", 0)),
                            "same_net_topo_edges_with_observed_intervals": int(
                                topo_debug_stats.get("edges_with_observed_intervals", 0)
                            ),
                            "same_net_topo_mean_gap": float(topo_debug_stats.get("mean_gap", 0.0)),
                            "same_net_topo_tie_ratio": float(topo_debug_stats.get("tie_ratio", 0.0)),
                            "same_net_topo_zero_zero_edges": int(
                                topo_debug_stats.get("zero_zero_edges", 0)
                            ),
                            "same_net_topo_exact_equal_edges": int(
                                topo_debug_stats.get("exact_equal_edges", 0)
                            ),
                        }
                    )
            valid_weights[is_diagonal] = diag_weights

        weights[valid_mask] = valid_weights
        return weights

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

    def _debug_abnormal_segments(self, segment_result, steiner_topo_op, flat_pin_from, flat_pin_to, newx, newy):
        """Log segments that look abnormally 'fat' to locate bad edges/nets."""
        if not getattr(self.params, "l_shape_debug_segments", False):
            return
        seg_size_x = segment_result["segment_size_x"]
        seg_size_y = segment_result["segment_size_y"]
        seg_edge_idx = segment_result.get("segment_edge_idx", None)
        if seg_edge_idx is None or seg_edge_idx.numel() == 0:
            return
        min_side = torch.minimum(seg_size_x, seg_size_y)
        min_bin = min(self.bin_size_x, self.bin_size_y)
        thresh = max(3.0 * float(self.wire_width), 0.5 * float(min_bin))
        abnormal_mask = min_side > thresh
        if abnormal_mask.sum().item() == 0:
            logger.info(f"L-shape debug: no abnormal segments (min_side > {thresh:.3f})")
            return

        idx = torch.nonzero(abnormal_mask, as_tuple=True)[0]
        area = seg_size_x * seg_size_y
        topk = int(getattr(self.params, "l_shape_debug_topk", 20))
        if idx.numel() > topk:
            _, sel = torch.topk(area[idx], k=topk)
            idx = idx[sel]

        vertex_to_net = self._build_vertex_to_net(steiner_topo_op, newx.numel())
        net_names = getattr(self.placedb, "net_names", [])
        seg_llx = segment_result.get("segment_llx", None)
        seg_lly = segment_result.get("segment_lly", None)

        logger.warning(
            f"L-shape debug: {idx.numel()} abnormal segments (min_side > {thresh:.3f}); "
            f"wire_width={self.wire_width:.3f}, bin=({self.bin_size_x:.3f},{self.bin_size_y:.3f})"
        )
        for seg_i in idx.tolist():
            edge_idx = int(seg_edge_idx[seg_i].item())
            if edge_idx < 0 or edge_idx >= flat_pin_from.numel():
                continue
            v1 = int(flat_pin_from[edge_idx].item())
            v2 = int(flat_pin_to[edge_idx].item())
            net_id = int(vertex_to_net[v1].item()) if v1 < vertex_to_net.numel() else -1
            net_name = ""
            if 0 <= net_id < len(net_names):
                net_name = net_names[net_id]
                if isinstance(net_name, bytes):
                    net_name = net_name.decode("utf-8")
            llx = float(seg_llx[seg_i].item()) if seg_llx is not None else float("nan")
            lly = float(seg_lly[seg_i].item()) if seg_lly is not None else float("nan")
            logger.warning(
                f"[L-shape debug] seg={seg_i} edge={edge_idx} net={net_id}({net_name}) "
                f"v1={v1} v2={v2} size=({seg_size_x[seg_i]:.3f},{seg_size_y[seg_i]:.3f}) "
                f"ll=({llx:.3f},{lly:.3f}) "
                f"p1=({newx[v1]:.3f},{newy[v1]:.3f}) p2=({newx[v2]:.3f},{newy[v2]:.3f})"
            )
    
    def forward(self, pos, steiner_topo_op, pin_pos_op, use_l_direction=True):
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
        
        # ========== 关键修复: 保持梯度链完整 ==========
        # steiner_topo_op 只支持CPU，所以我们需要：
        # 1. 检查pos是否越界（仅movable）
        # 2. 计算pin_pos（不再clamp）
        # 3. 将pin_pos移到CPU (保持梯度)
        # 4. 在CPU上调用steiner_topo_op
        # 5. 在CPU上计算segments和density
        # 6. 将结果移回CUDA
        
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
        pin_pos = pin_pos_op(pos)
        num_pins = pin_pos.numel() // 2
        pin_pos_x = pin_pos[:num_pins].clamp(self.xl, self.xh)
        pin_pos_y = pin_pos[num_pins:].clamp(self.yl, self.yh)
        pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
        
        # 3. 将pin_pos移到CPU（保持梯度连接）
        if pin_pos.is_cuda:
            pin_pos_cpu = pin_pos.cpu()  # .cpu() 会创建 ToCopyBackward，梯度可以流回
        else:
            pin_pos_cpu = pin_pos
        
        # 4. 调用steiner_topo_op (在CPU上)
        newx, newy = steiner_topo_op(pin_pos_cpu)
                
        # 5. 获取边信息（确保在CPU上）
        flat_pin_from = steiner_topo_op.flat_pin_from
        flat_pin_to = steiner_topo_op.flat_pin_to
        
        if flat_pin_from is None or flat_pin_to is None:
            logger.warning("Steiner edges not available, returning zero cost")
            return torch.zeros(1, dtype=pos.dtype, device=original_device, requires_grad=True)
        
        # 确保边信息在CPU上（与newx, newy一致）
        if flat_pin_from.is_cuda:
            flat_pin_from = flat_pin_from.cpu()
        if flat_pin_to.is_cuda:
            flat_pin_to = flat_pin_to.cpu()
        
        # 5. 获取L方向信息
        if use_l_direction and hasattr(steiner_topo_op, 'edge_l_directions') and steiner_topo_op.edge_l_directions is not None:
            l_directions = steiner_topo_op.edge_l_directions
            # 确保L方向在CPU上
            if l_directions.is_cuda:
                l_directions = l_directions.cpu()
            logger.debug(f"Using EGR L-directions: {len(l_directions)} edges")
        else:
            fallback_direction = UNKNOWN if self.soft_l_assignment else H_FIRST
            l_directions = torch.full(
                (flat_pin_from.numel(),), fallback_direction,
                dtype=torch.int32, device=newx.device  # CPU
            )
            logger.debug(
                "No L-direction info, using default %s",
                "UNKNOWN" if fallback_direction == UNKNOWN else "H_FIRST",
            )
        
        soft_l_weights = None
        if self.soft_l_assignment:
            edge_net_ids = None
            if self.soft_l_use_same_net_topo_scoring and isinstance(self.same_net_topo_cache, dict):
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

        # 6. 构建L形segments（在CPU上）
        segment_result = self.segment_builder(
            newx, newy, flat_pin_from, flat_pin_to, l_directions, soft_l_weights=soft_l_weights
        )

        num_segments = segment_result['num_segments']
        if num_segments == 0:
            logger.warning("No valid segments built")
            return torch.zeros(1, dtype=pos.dtype, device=original_device, requires_grad=True)

        # Debug abnormal segments (fat rectangles)
        self._debug_abnormal_segments(
            segment_result, steiner_topo_op, flat_pin_from, flat_pin_to, newx, newy
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
        
        # 根据模式选择计算方式
        if self.density_mode == "electric":
            # 使用C++/CUDA电势场模型
            # 如果density_op支持CUDA，将数据移到CUDA
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
            )
        else:
            # 使用Python RUDY密度的energy模式
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
        
        logger.debug(f"L-shape routability cost: {cost.item():.4f}, "
                    f"{num_segments} segments, {(time.time() - tt) * 1000:.2f} ms")
        
        return cost
    
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
        
        # steiner_topo_op要求CPU tensor
        pin_pos_cpu = pin_pos.cpu() if pin_pos.is_cuda else pin_pos
        newx, newy = steiner_topo_op(pin_pos_cpu)
        
        # 将结果移回原设备
        if device.type == 'cuda':
            newx = newx.to(device)
            newy = newy.to(device)
        
        flat_pin_from = steiner_topo_op.flat_pin_from
        flat_pin_to = steiner_topo_op.flat_pin_to
        
        if flat_pin_from is None or flat_pin_to is None:
            return torch.zeros(
                self.density_op.num_bins_x, self.density_op.num_bins_y,
                dtype=pos.dtype, device=device
            )
        
        # 确保边信息在正确的设备上
        if flat_pin_from.device != device:
            flat_pin_from = flat_pin_from.to(device)
        if flat_pin_to.device != device:
            flat_pin_to = flat_pin_to.to(device)
        
        if use_l_direction and hasattr(steiner_topo_op, 'edge_l_directions') and steiner_topo_op.edge_l_directions is not None:
            l_directions = steiner_topo_op.edge_l_directions
            if l_directions.device != device:
                l_directions = l_directions.to(device)
        else:
            fallback_direction = UNKNOWN if self.soft_l_assignment else H_FIRST
            l_directions = torch.full(
                (flat_pin_from.numel(),), fallback_direction,
                dtype=torch.int32, device=device
            )
        
        soft_l_weights = None
        if self.soft_l_assignment:
            edge_net_ids = None
            if self.soft_l_use_same_net_topo_scoring and isinstance(self.same_net_topo_cache, dict):
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
        self.l_shape_routability_weight = getattr(params, 'l_shape_routability_weight', 1.0)
        logger.info(f"L-shape routability initialized with weight {self.l_shape_routability_weight}, "
                   f"mode={density_mode}")
    
    def l_shape_routability_obj(self, pos, use_l_direction=True):
        """
        计算L形routability目标
        
        需要self.op_collections中有steiner_topo_op和pin_pos_op
        """
        if not hasattr(self, 'l_shape_routability_op'):
            raise RuntimeError("L-shape routability not initialized. Call init_l_shape_routability first.")
        
        steiner_topo_op = self.op_collections.steiner_topo_op
        pin_pos_op = self.op_collections.pin_pos_op
        
        return self.l_shape_routability_op(
            pos, steiner_topo_op, pin_pos_op, use_l_direction
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


# ==================== 绘图工具 ====================

def plot_l_shape_segments(segments, newx=None, newy=None, flat_from=None, flat_to=None, l_directions=None,
                          output_path=None, placedb=None, params=None):
    """
    绘制L形segments
    
    Args:
        segments: LShapeSegmentOp的输出（包含缓存的原始数据）
        newx, newy: 顶点坐标（可选，优先使用segments中缓存的数据）
        flat_from, flat_to: 边信息（可选，优先使用segments中缓存的数据）
        l_directions: L方向（可选，优先使用segments中缓存的数据）
        output_path: 输出路径
        placedb, params: 用于坐标转换
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    # 优先使用segments中缓存的原始数据（确保一致性）
    if 'newx' in segments:
        newx = segments['newx']
    if 'newy' in segments:
        newy = segments['newy']
    if 'flat_from' in segments:
        flat_from = segments['flat_from']
    if 'flat_to' in segments:
        flat_to = segments['flat_to']
    if 'l_directions' in segments:
        l_directions = segments['l_directions']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制segments
    if segments['num_segments'] > 0:
        seg_llx = segments['segment_llx'].detach().cpu().numpy()
        seg_lly = segments['segment_lly'].detach().cpu().numpy()
        seg_size_x = segments['segment_size_x'].detach().cpu().numpy()
        seg_size_y = segments['segment_size_y'].detach().cpu().numpy()
        seg_is_h = segments['segment_is_horizontal'].detach().cpu().numpy()
        
        patches = []
        colors = []
        
        for i in range(len(seg_llx)):
            rect = Rectangle(
                (seg_llx[i], seg_lly[i]),
                seg_size_x[i], seg_size_y[i]
            )
            patches.append(rect)
            # 水平segment用蓝色，垂直segment用绿色
            colors.append('blue' if seg_is_h[i] else 'green')
        
        pc = PatchCollection(patches, alpha=0.3, edgecolor='black', linewidth=0.5)
        pc.set_facecolor(colors)
        ax.add_collection(pc)
    
    # 绘制原始边（与segment构建逻辑保持一致）
    if newx is None or newy is None or flat_from is None or flat_to is None or l_directions is None:
        logger.warning("Missing edge data for plotting L-shape lines")
    else:
        newx_np = newx.detach().cpu().numpy()
        newy_np = newy.detach().cpu().numpy()
        flat_from_np = flat_from.detach().cpu().numpy()
        flat_to_np = flat_to.detach().cpu().numpy()
        l_dir_np = l_directions.detach().cpu().numpy()
        
        for i in range(len(flat_from_np)):
            f, t = flat_from_np[i], flat_to_np[i]
            if f < 0 or t < 0 or f >= len(newx_np) or t >= len(newx_np):
                continue
            
            x1, y1 = newx_np[f], newy_np[f]
            x2, y2 = newx_np[t], newy_np[t]
            l_dir = l_dir_np[i]
            
            # 与segment构建逻辑保持一致：先检查几何上是否是直线
            is_horizontal_line = abs(y1 - y2) < 1e-6
            is_vertical_line = abs(x1 - x2) < 1e-6
            is_straight = is_horizontal_line or is_vertical_line or (l_dir == STRAIGHT)
            
            if is_straight:
                # 几何上是水平或垂直线，直接画直线
                ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5, alpha=0.5)
            elif l_dir == V_FIRST:
                # Lower-L: 先垂直后水平，拐点在 (x1, y2)
                corner_x, corner_y = x1, y2
                ax.plot([x1, corner_x], [y1, corner_y], 'm-', linewidth=0.8, alpha=0.7)
                ax.plot([corner_x, x2], [corner_y, y2], 'm-', linewidth=0.8, alpha=0.7)
            elif l_dir == UNKNOWN:
                # 残余 UNKNOWN 与 segment 构建保持一致：跳过，不画 L 形
                continue
            else:
                # Upper-L (包括 H_FIRST 和残余 FAKE_STRAIGHT): 先水平后垂直
                corner_x, corner_y = x2, y1
                ax.plot([x1, corner_x], [y1, corner_y], 'r-', linewidth=0.8, alpha=0.7)
                ax.plot([corner_x, x2], [corner_y, y2], 'r-', linewidth=0.8, alpha=0.7)
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title('L-shape Segments (blue=horizontal, green=vertical)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', label='Upper-L (H→V)'),
        Line2D([0], [0], color='m', label='Lower-L (V→H)'),
        Line2D([0], [0], color='k', linestyle='--', label='Straight'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"L-shape segments plot saved to {output_path}")


def plot_soft_l_intermediate(segments, output_path, max_diagonal_edges=4000):
    """Plot soft-L candidate paths and edge-level probability statistics."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    required = ("newx", "newy", "flat_from", "flat_to", "soft_l_weights")
    missing = [key for key in required if key not in segments]
    if missing:
        raise ValueError(f"Missing soft-L plotting data: {missing}")

    newx = segments["newx"].detach().cpu().numpy()
    newy = segments["newy"].detach().cpu().numpy()
    flat_from = segments["flat_from"].detach().cpu().numpy()
    flat_to = segments["flat_to"].detach().cpu().numpy()
    soft_l_weights = segments["soft_l_weights"].detach().cpu().numpy()

    valid = (
        (flat_from >= 0)
        & (flat_to >= 0)
        & (flat_from < len(newx))
        & (flat_to < len(newx))
    )
    if not np.any(valid):
        raise ValueError("No valid edges available for soft-L plotting")

    edge_idx = np.nonzero(valid)[0]
    x1 = newx[flat_from[valid]]
    y1 = newy[flat_from[valid]]
    x2 = newx[flat_to[valid]]
    y2 = newy[flat_to[valid]]
    weights = np.clip(soft_l_weights[valid], 0.0, None)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = np.where(weight_sum > 1e-12, weights / np.clip(weight_sum, 1e-12, None), 0.5)

    is_diagonal = (np.abs(x1 - x2) >= 1e-6) & (np.abs(y1 - y2) >= 1e-6)
    diag_idx = np.nonzero(is_diagonal)[0]
    if diag_idx.size == 0:
        raise ValueError("No diagonal edges available for soft-L plotting")

    if diag_idx.size > max_diagonal_edges:
        take = np.linspace(0, diag_idx.size - 1, max_diagonal_edges, dtype=int)
        diag_idx = diag_idx[take]

    diag_x1 = x1[diag_idx]
    diag_y1 = y1[diag_idx]
    diag_x2 = x2[diag_idx]
    diag_y2 = y2[diag_idx]
    diag_weights = weights[diag_idx]

    full_diag_weights = weights[is_diagonal]
    full_p_h = full_diag_weights[:, 0]
    full_entropy = -np.sum(
        np.where(full_diag_weights > 1e-12, full_diag_weights * np.log2(np.clip(full_diag_weights, 1e-12, None)), 0.0),
        axis=1,
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.2], height_ratios=[1.0, 1.0])
    ax_paths = fig.add_subplot(gs[:, 0])
    ax_prob = fig.add_subplot(gs[0, 1])
    ax_entropy = fig.add_subplot(gs[1, 1])

    for i in range(diag_idx.size):
        h_weight = float(diag_weights[i, 0])
        v_weight = float(diag_weights[i, 1])
        x1_i, y1_i = diag_x1[i], diag_y1[i]
        x2_i, y2_i = diag_x2[i], diag_y2[i]

        h_alpha = min(0.95, 0.05 + 0.90 * h_weight)
        v_alpha = min(0.95, 0.05 + 0.90 * v_weight)
        h_width = 0.4 + 1.4 * h_weight
        v_width = 0.4 + 1.4 * v_weight

        ax_paths.plot([x1_i, x2_i], [y1_i, y1_i], color="tab:red", alpha=h_alpha, linewidth=h_width)
        ax_paths.plot([x2_i, x2_i], [y1_i, y2_i], color="tab:red", alpha=h_alpha, linewidth=h_width)

        ax_paths.plot([x1_i, x1_i], [y1_i, y2_i], color="tab:purple", alpha=v_alpha, linewidth=v_width)
        ax_paths.plot([x1_i, x2_i], [y2_i, y2_i], color="tab:purple", alpha=v_alpha, linewidth=v_width)

    ax_paths.set_aspect("equal")
    ax_paths.autoscale()
    ax_paths.set_title(
        f"Soft L Candidate Paths (sampled {diag_idx.size}/{np.count_nonzero(is_diagonal)} diagonal edges)"
    )
    ax_paths.set_xlabel("X")
    ax_paths.set_ylabel("Y")
    ax_paths.legend(
        handles=[
            Line2D([0], [0], color="tab:red", label="H-path candidate"),
            Line2D([0], [0], color="tab:purple", label="V-path candidate"),
        ],
        loc="upper right",
    )

    ax_prob.hist(full_p_h, bins=20, color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_prob.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax_prob.set_title("p(H-first) Distribution")
    ax_prob.set_xlabel("p_H")
    ax_prob.set_ylabel("Edge Count")

    ax_entropy.hist(full_entropy, bins=20, color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_entropy.set_title("Soft-L Entropy")
    ax_entropy.set_xlabel("Entropy (bits)")
    ax_entropy.set_ylabel("Edge Count")
    ambiguous_ratio = float(np.mean((full_p_h >= 0.4) & (full_p_h <= 0.6)))
    decisive_ratio = float(np.mean((full_p_h <= 0.1) | (full_p_h >= 0.9)))
    stats_text = "\n".join(
        [
            f"diagonal_edges = {int(np.count_nonzero(is_diagonal))}",
            f"sampled_edges = {int(diag_idx.size)}",
            f"mean_p_H = {float(full_p_h.mean()):.3f}",
            f"mean_entropy = {float(full_entropy.mean()):.3f}",
            f"ambiguous[0.4,0.6] = {ambiguous_ratio:.1%}",
            f"decisive<=0.1|>=0.9 = {decisive_ratio:.1%}",
        ]
    )
    ax_entropy.text(
        0.98,
        0.98,
        stats_text,
        transform=ax_entropy.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Soft L intermediate plot saved to %s", output_path)


def plot_soft_l_scoring_maps(soft_debug, output_path, title_prefix="Soft L Scoring"):
    """Plot cached directional cost and hotspot maps used for soft-L scoring."""
    import matplotlib.pyplot as plt
    import numpy as np

    required = ("cost_map_h", "cost_map_v", "hotspot_map_h", "hotspot_map_v")
    missing = [key for key in required if key not in soft_debug]
    if missing:
        raise ValueError(f"Missing soft-L scoring maps: {missing}")

    has_ggr_overflow = ("ggr_overflow_h" in soft_debug and isinstance(soft_debug["ggr_overflow_h"], torch.Tensor) and
                        "ggr_overflow_v" in soft_debug and isinstance(soft_debug["ggr_overflow_v"], torch.Tensor))
    nrows = 3 if has_ggr_overflow else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows), constrained_layout=True)
    plots = [
        ("cost_map_h", "Horizontal-Leg Cost", "viridis"),
        ("cost_map_v", "Vertical-Leg Cost", "viridis"),
        ("hotspot_map_h", "Horizontal Hotspot Map", "magma"),
        ("hotspot_map_v", "Vertical Hotspot Map", "magma"),
    ]
    if has_ggr_overflow:
        plots.extend([
            ("ggr_overflow_h", "GGR H Overflow Ratio", "plasma"),
            ("ggr_overflow_v", "GGR V Overflow Ratio", "plasma"),
        ])

    for ax, (key, title, cmap) in zip(axes.flat, plots):
        value = soft_debug[key]
        if value.requires_grad:
            value = value.detach()
        if value.is_cuda:
            value = value.cpu()
        kwargs = {}
        if key.startswith("ggr_overflow_"):
            vmax = max(1e-6, float(np.percentile(value.numpy(), 99)))
            kwargs = {"vmin": 0.0, "vmax": vmax}
        im = ax.imshow(value.numpy().T, origin="lower", cmap=cmap, aspect="equal", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    extra = []
    if "effective_hotspot_weight" in soft_debug:
        extra.append(f"effective_hotspot_weight={soft_debug['effective_hotspot_weight']:.3f}")
    if "overflow_ratio" in soft_debug:
        extra.append(f"overflow_ratio={soft_debug['overflow_ratio']:.4f}")
    fig.suptitle(f"{title_prefix}" + (f" ({', '.join(extra)})" if extra else ""))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Soft L scoring map plot saved to %s", output_path)


def plot_l_shape_electric_overflow_map(l_shape_op, output_path, title_prefix="L-shape Electric Overflow"):
    """Plot GGR overflow and surrogate overflow in a two-row comparison."""
    import matplotlib.pyplot as plt
    import numpy as np

    plot_payload = _build_l_shape_electric_plot_maps(l_shape_op)
    ggr_overflow_map = plot_payload["ggr_overflow_map"]
    ggr_overflow_map_h = plot_payload["ggr_overflow_map_h"]
    ggr_overflow_map_v = plot_payload["ggr_overflow_map_v"]
    overflow_map = plot_payload["surrogate_overflow_map"]
    overflow_map_h = plot_payload["surrogate_overflow_map_h"]
    overflow_map_v = plot_payload["surrogate_overflow_map_v"]
    split_available = plot_payload["split_available"]

    def compute_limits(*map_tensors):
        valid_maps = [tensor.numpy() for tensor in map_tensors if isinstance(tensor, torch.Tensor)]
        if not valid_maps:
            return 0.0, 1.0
        stacked = np.concatenate([m.reshape(-1) for m in valid_maps])
        if float(np.min(stacked)) < -1e-9:
            vmax = max(1e-6, float(np.percentile(np.abs(stacked), 99)))
            return -vmax, vmax
        vmax = max(1e-6, float(np.percentile(stacked, 99)))
        return 0.0, vmax

    def render_map(ax, map_tensor, title, vmin, vmax):
        map_np = map_tensor.numpy()
        if vmin < 0.0:
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap="coolwarm",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap="plasma",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
        positive_ratio = float(np.mean(map_np > 0))
        ax.set_title(
            f"{title}\nmean={float(map_np.mean()):.4e} "
            f"max={float(map_np.max()):.4e} pos={positive_ratio:.1%}"
        )
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        return im

    if split_available and isinstance(overflow_map_h, torch.Tensor) and isinstance(overflow_map_v, torch.Tensor):
        fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
        limits_h = compute_limits(ggr_overflow_map_h, overflow_map_h)
        limits_v = compute_limits(ggr_overflow_map_v, overflow_map_v)
        limits_t = compute_limits(ggr_overflow_map, overflow_map)
        im_h_ggr = render_map(axes[0, 0], ggr_overflow_map_h, "GGR Overflow H", *limits_h)
        im_v_ggr = render_map(axes[0, 1], ggr_overflow_map_v, "GGR Overflow V", *limits_v)
        im_t_ggr = render_map(axes[0, 2], ggr_overflow_map, "GGR Overflow Total", *limits_t)
        render_map(axes[1, 0], overflow_map_h, "Surrogate Overflow H", *limits_h)
        render_map(axes[1, 1], overflow_map_v, "Surrogate Overflow V", *limits_v)
        render_map(axes[1, 2], overflow_map, "Surrogate Overflow Total", *limits_t)
        axes[0, 0].set_ylabel("Bin Y\nGGR")
        axes[1, 0].set_ylabel("Bin Y\nSurrogate")
        fig.colorbar(im_h_ggr, ax=axes[:, 0], fraction=0.025, pad=0.02, label="Overflow (area units)")
        fig.colorbar(im_v_ggr, ax=axes[:, 1], fraction=0.025, pad=0.02, label="Overflow (area units)")
        fig.colorbar(im_t_ggr, ax=axes[:, 2], fraction=0.025, pad=0.02, label="Overflow (area units)")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 18), constrained_layout=True)
        limits = compute_limits(ggr_overflow_map, overflow_map)
        im_ggr = render_map(axes[0], ggr_overflow_map, "GGR Overflow", *limits)
        render_map(axes[1], overflow_map, "Surrogate Overflow", *limits)
        fig.colorbar(im_ggr, ax=axes, label="Overflow (area units)")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape electric overflow plot saved to %s", output_path)


def _build_l_shape_electric_plot_maps(l_shape_op):
    density_map = getattr(l_shape_op, "cached_density_map", None)
    density_map_h = getattr(l_shape_op, "cached_density_map_h", None)
    density_map_v = getattr(l_shape_op, "cached_density_map_v", None)
    density_op = getattr(l_shape_op, "density_op", None)
    if density_map is None or density_op is None:
        raise ValueError("Missing cached density map or density op for electric plotting")

    def prepare_tensor(value):
        if not isinstance(value, torch.Tensor):
            return None
        if value.requires_grad:
            value = value.detach()
        if value.is_cuda:
            value = value.cpu()
        return value.to(dtype=torch.float32)

    density_map = prepare_tensor(density_map)
    density_map_h = prepare_tensor(density_map_h)
    density_map_v = prepare_tensor(density_map_v)
    target_density = prepare_tensor(getattr(density_op, "target_density", None))
    target_demand = prepare_tensor(getattr(density_op, "target_demand", None))
    target_density_h = prepare_tensor(getattr(density_op, "target_density_h", None))
    target_density_v = prepare_tensor(getattr(density_op, "target_density_v", None))
    target_demand_h = prepare_tensor(getattr(density_op, "target_demand_h", None))
    target_demand_v = prepare_tensor(getattr(density_op, "target_demand_v", None))
    area_per_track = prepare_tensor(getattr(density_op, "area_per_track", None))

    if not isinstance(target_density, torch.Tensor) or target_density.dim() != 2:
        raise TypeError(
            "L-shape electric plotting expects a 2D routing supply tensor in density_op.target_density"
        )
    if not isinstance(target_demand, torch.Tensor) or target_demand.dim() != 2:
        raise TypeError(
            "L-shape electric plotting expects a 2D routing demand tensor in density_op.target_demand"
        )

    surrogate_overflow_map = density_map.clone()
    surrogate_overflow_map_h = None
    surrogate_overflow_map_v = None
    ggr_overflow_map = torch.zeros_like(density_map)
    ggr_overflow_map_h = None
    ggr_overflow_map_v = None
    supply_map = target_density

    if isinstance(area_per_track, torch.Tensor) and area_per_track.numel() == 1 and float(area_per_track.item()) > 0:
        calibrated_area_per_track = float(area_per_track.item())
    else:
        total_density = float(density_map.sum().item())
        total_demand = float(target_demand.sum().item())
        calibrated_area_per_track = (
            total_density / total_demand if total_density > 0 and total_demand > 0 else 0.0
        )

    split_available = (
        isinstance(density_map_h, torch.Tensor)
        and isinstance(density_map_v, torch.Tensor)
        and isinstance(target_density_h, torch.Tensor)
        and isinstance(target_density_v, torch.Tensor)
        and density_map_h.shape == target_density_h.shape
        and density_map_v.shape == target_density_v.shape
    )

    if calibrated_area_per_track > 0:
        if split_available:
            calibrated_area_per_track_h = calibrated_area_per_track
            calibrated_area_per_track_v = calibrated_area_per_track
            if isinstance(target_demand_h, torch.Tensor) and target_demand_h.dim() == 2:
                total_density_h = float(density_map_h.sum().item())
                total_demand_h = float(target_demand_h.sum().item())
                if total_density_h > 0 and total_demand_h > 0:
                    calibrated_area_per_track_h = total_density_h / total_demand_h
            if isinstance(target_demand_v, torch.Tensor) and target_demand_v.dim() == 2:
                total_density_v = float(density_map_v.sum().item())
                total_demand_v = float(target_demand_v.sum().item())
                if total_density_v > 0 and total_demand_v > 0:
                    calibrated_area_per_track_v = total_density_v / total_demand_v

            demand_h_in_tracks = density_map_h / calibrated_area_per_track_h
            demand_v_in_tracks = density_map_v / calibrated_area_per_track_v
            surrogate_overflow_map_h = torch.relu(demand_h_in_tracks - target_density_h) * calibrated_area_per_track_h
            surrogate_overflow_map_v = torch.relu(demand_v_in_tracks - target_density_v) * calibrated_area_per_track_v
            surrogate_overflow_map = surrogate_overflow_map_h + surrogate_overflow_map_v
            if isinstance(target_demand_h, torch.Tensor) and target_demand_h.dim() == 2:
                ggr_overflow_map_h = torch.relu(target_demand_h - target_density_h) * calibrated_area_per_track_h
            else:
                ggr_overflow_map_h = torch.zeros_like(surrogate_overflow_map_h)
            if isinstance(target_demand_v, torch.Tensor) and target_demand_v.dim() == 2:
                ggr_overflow_map_v = torch.relu(target_demand_v - target_density_v) * calibrated_area_per_track_v
            else:
                ggr_overflow_map_v = torch.zeros_like(surrogate_overflow_map_v)
            ggr_overflow_map = ggr_overflow_map_h + ggr_overflow_map_v
        else:
            demand_in_tracks = density_map / calibrated_area_per_track
            overflow_in_tracks = torch.relu(demand_in_tracks - supply_map)
            surrogate_overflow_map = overflow_in_tracks * calibrated_area_per_track
            ggr_overflow_map = torch.relu(target_demand - supply_map) * calibrated_area_per_track

    return {
        "density_map": density_map,
        "density_map_h": density_map_h,
        "density_map_v": density_map_v,
        "surrogate_overflow_map": surrogate_overflow_map,
        "surrogate_overflow_map_h": surrogate_overflow_map_h,
        "surrogate_overflow_map_v": surrogate_overflow_map_v,
        "ggr_overflow_map": ggr_overflow_map,
        "ggr_overflow_map_h": ggr_overflow_map_h,
        "ggr_overflow_map_v": ggr_overflow_map_v,
        "split_available": split_available,
        "density_op": density_op,
    }


def plot_l_shape_electric_potential_map(l_shape_op, output_path, title_prefix="L-shape Electric Potential"):
    """Plot the potential map(s) reconstructed from the electric overflow maps."""
    import matplotlib.pyplot as plt
    import numpy as np

    plot_payload = _build_l_shape_electric_plot_maps(l_shape_op)
    overflow_map = plot_payload["surrogate_overflow_map"]
    overflow_map_h = plot_payload["surrogate_overflow_map_h"]
    overflow_map_v = plot_payload["surrogate_overflow_map_v"]
    split_available = plot_payload["split_available"]
    density_op = plot_payload["density_op"]

    ref_tensor = getattr(density_op, "bin_center_x", None)
    if not isinstance(ref_tensor, torch.Tensor):
        target_density = getattr(density_op, "target_density", None)
        if not isinstance(target_density, torch.Tensor):
            raise ValueError("L-shape electric potential plot requires density_op.target_density")
        density_op._init_bins(target_density.device, target_density.dtype)
        ref_tensor = getattr(density_op, "bin_center_x", None)
    if not isinstance(ref_tensor, torch.Tensor):
        raise ValueError("L-shape electric potential plot requires initialized density_op bin centers")

    if (
        getattr(density_op, "idct2", None) is None
        or getattr(density_op, "dct2", None) is None
        or getattr(density_op, "inv_wu2_plus_wv2", None) is None
    ):
        density_op._init_dct(ref_tensor.device, ref_tensor.dtype)
    device = ref_tensor.device
    dtype = ref_tensor.dtype
    bin_area = float(density_op.bin_size_x * density_op.bin_size_y)

    def compute_potential(map_tensor):
        map_tensor = map_tensor.to(device=device, dtype=dtype)
        overflow_map_normalized = map_tensor * (1.0 / bin_area)
        auv = density_op.dct2.forward(overflow_map_normalized)
        potential_map = density_op.idct2.forward(auv * density_op.inv_wu2_plus_wv2)
        potential_map = potential_map * bin_area
        return potential_map.detach().cpu().to(dtype=torch.float32)

    def render_map(ax, map_tensor, title):
        map_np = map_tensor.numpy()
        has_negative = float(np.min(map_np)) < -1e-9
        if has_negative:
            vmax = max(1e-6, float(np.percentile(np.abs(map_np), 99)))
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap="coolwarm",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            vmin = float(np.percentile(map_np, 1))
            vmax = max(vmin + 1e-6, float(np.percentile(map_np, 99)))
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap="viridis",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
        ax.set_title(
            f"{title}\nmean={float(map_np.mean()):.4e} "
            f"max={float(map_np.max()):.4e} min={float(map_np.min()):.4e}"
        )
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        return im

    if split_available and isinstance(overflow_map_h, torch.Tensor) and isinstance(overflow_map_v, torch.Tensor):
        potential_map_h = compute_potential(overflow_map_h)
        potential_map_v = compute_potential(overflow_map_v)
        potential_map = potential_map_h + potential_map_v
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        im_h = render_map(axes[0], potential_map_h, f"{title_prefix} H")
        im_v = render_map(axes[1], potential_map_v, f"{title_prefix} V")
        im_t = render_map(axes[2], potential_map, f"{title_prefix} Total")
        fig.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.04, label="Potential")
        fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04, label="Potential")
        fig.colorbar(im_t, ax=axes[2], fraction=0.046, pad=0.04, label="Potential")
    else:
        potential_map = compute_potential(overflow_map)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = render_map(ax, potential_map, title_prefix)
        fig.colorbar(im, ax=ax, label="Potential")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape electric potential plot saved to %s", output_path)


def plot_segment_density_map(density_map, output_path, title="Segment Density Map", colormap="hot"):
    """绘制segment密度图"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 确保tensor可以转换为numpy
    if density_map.requires_grad:
        density_map = density_map.detach()
    if density_map.is_cuda:
        density_map = density_map.cpu()
    
    im = ax.imshow(
        density_map.numpy().T,  # 转置使x轴在水平方向
        origin='lower',
        cmap=colormap,
        aspect='equal'
    )
    
    ax.set_title(title)
    ax.set_xlabel('Bin X')
    ax.set_ylabel('Bin Y')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Density')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Density map saved to {output_path}")
