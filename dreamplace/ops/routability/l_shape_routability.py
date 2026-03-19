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
    
    def __init__(self, placedb, params, wire_width=None, num_bins_x=64, num_bins_y=64,
                 density_mode="electric", target_density=1.0, target_demand=None):
        """
        Args:
            placedb: placement database
            params: parameters
            wire_width: segment线宽（默认从params获取或使用0）
            num_bins_x, num_bins_y: 密度计算的bin数量
            density_mode: 密度计算模式 ("rudy" 或 "electric")
            target_density: 目标密度 (用于electric模式)
            target_demand: 目标需求 (EGR net map，用于electric模式的单位标定)
        """
        super(LShapeRoutabilityOp, self).__init__()
        
        self.placedb = placedb
        self.params = params
        self.density_mode = density_mode
        
        # 线宽设置
        if wire_width is None:
            wire_width = getattr(params, 'route_wire_width', 0.0)
        self.wire_width = wire_width
        
        # L形segment构建器
        self.segment_builder = LShapeSegmentOp(
            wire_width=wire_width,
            use_vectorized=True
        )
        
        # 根据模式选择密度计算器
        if density_mode == "electric":

            self.density_op = create_l_shape_electric_potential(
                placedb,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                target_density=target_density,
                target_demand=target_demand,
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
        self.cached_density_map = None

    def update_targets(self, target_density=None, target_demand=None):
        """
        Refresh external routing targets used by the electric L-shape model.

        Args:
            target_density: 2D routing supply map or scalar
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

        if updated:
            self.cached_density_map = None
            logger.info("Updated L-shape routing targets (density=%s, demand=%s).",
                        "set" if target_density is not None else "keep",
                        "set" if target_demand is not None else "clear")

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
            # 没有L方向信息，使用默认（全部UNKNOWN，会被当作H_FIRST处理）
            l_directions = torch.full(
                (flat_pin_from.numel(),), UNKNOWN,
                dtype=torch.int32, device=newx.device  # CPU
            )
            logger.debug("No L-direction info, using default H_FIRST")
        
        # 6. 构建L形segments（在CPU上）
        segment_result = self.segment_builder(
            newx, newy, flat_pin_from, flat_pin_to, l_directions
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
        
        # 7. 计算密度代价（在CPU上）
        segment_pos = segment_result['segment_pos']
        segment_size_x = segment_result['segment_size_x']
        segment_size_y = segment_result['segment_size_y']
        
        # 根据模式选择计算方式
        if self.density_mode == "electric":
            # 使用C++/CUDA电势场模型
            # 如果density_op支持CUDA，将数据移到CUDA
            if original_device.type == 'cuda':
                segment_pos = segment_pos.to(original_device)
                segment_size_x = segment_size_x.to(original_device)
                segment_size_y = segment_size_y.to(original_device)
            segment_is_horizontal = segment_result.get('segment_is_horizontal', None)
            cost = self.density_op(segment_pos, segment_size_x, segment_size_y, segment_is_horizontal)
        else:
            # 使用Python RUDY密度的energy模式
            cost = self.density_op(segment_pos, segment_size_x, segment_size_y, mode="energy")
        
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
            l_directions = torch.full(
                (flat_pin_from.numel(),), UNKNOWN,
                dtype=torch.int32, device=device
            )
        
        segment_result = self.segment_builder(
            newx, newy, flat_pin_from, flat_pin_to, l_directions
        )
        
        if segment_result['num_segments'] == 0:
            return torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=pos.dtype, device=pos.device
            )
        
        # 根据模式选择计算方式
        if self.density_mode == "electric":
            # 使用overflow_op获取density_map
            density_map = self.overflow_op.compute_density_map(
                segment_result['segment_pos'],
                segment_result['segment_size_x'],
                segment_result['segment_size_y']
            )
        else:
            density_map = self.density_op(
                segment_result['segment_pos'],
                segment_result['segment_size_x'],
                segment_result['segment_size_y'],
                mode="density"
            )
        
        # 保存缓存用于绘图（包含原始数据，确保绘图一致性）
        self.cached_segments = segment_result
        self.cached_segments['newx'] = newx.detach()
        self.cached_segments['newy'] = newy.detach()
        self.cached_segments['flat_from'] = flat_pin_from.detach()
        self.cached_segments['flat_to'] = flat_pin_to.detach()
        self.cached_segments['l_directions'] = l_directions.detach()
        self.cached_density_map = density_map
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
        seg_size_x = self.cached_segments['segment_size_x'].detach().cpu().numpy()
        seg_size_y = self.cached_segments['segment_size_y'].detach().cpu().numpy()
        
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
            else:
                # Upper-L (包括 H_FIRST 和 UNKNOWN): 先水平后垂直，拐点在 (x2, y1)
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
