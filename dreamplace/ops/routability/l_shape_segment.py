##
# @file   l_shape_segment.py
# @brief  Build L-shape segments from Steiner tree edges with EGR L-direction
#         Each L-shape edge is split into two segments (horizontal + vertical)
#         This enables accurate density modeling for routing congestion
#

import torch
import logging

logger = logging.getLogger(__name__)

# L方向常量 (与steiner_topo.py保持一致)
H_FIRST = 0       # 先水平后垂直, 拐点在 (x2, y1)
V_FIRST = 1       # 先垂直后水平, 拐点在 (x1, y2)
STRAIGHT = 2      # 直线（水平或垂直）
FAKE_STRAIGHT = 3 # 伪直线（在gcell下只有一条wire）
UNKNOWN = -1      # 未知，segment阶段跳过


class LShapeSegmentBuilder:
    """
    将Steiner树的边根据L方向拆分为segments
    
    核心思想：
    - H_FIRST: p1 -> corner(p2.x, p1.y) -> p2 (先水平后垂直)
    - V_FIRST: p1 -> corner(p1.x, p2.y) -> p2 (先垂直后水平)
    - STRAIGHT: p1 -> p2 (直线)
    
    每个segment是一个矩形，用于后续密度计算
    """
    
    def __init__(self, wire_width=0.0):
        """
        Args:
            wire_width: 线宽，用于给segment增加宽度
        """
        self.wire_width = wire_width
    
    def build_segments(self, newx, newy, flat_from, flat_to, l_directions):
        """
        根据L方向将边拆分为segments
        
        Args:
            newx: [num_vertices] 所有顶点的x坐标 (pins + Steiner points)
            newy: [num_vertices] 所有顶点的y坐标
            flat_from: [num_edges] 边的起点索引
            flat_to: [num_edges] 边的终点索引
            l_directions: [num_edges] 每条边的L方向 (H_FIRST/V_FIRST/STRAIGHT/UNKNOWN)
            
        Returns:
            dict with:
                segment_llx: [num_segments] segment左下角x
                segment_lly: [num_segments] segment左下角y
                segment_size_x: [num_segments] segment宽度
                segment_size_y: [num_segments] segment高度
                segment_edge_idx: [num_segments] 每个segment对应的原始边索引
                segment_is_horizontal: [num_segments] 是否是水平segment
        """
        device = newx.device
        dtype = newx.dtype
        
        num_edges = flat_from.numel()
        
        # 预分配（最多每条边拆分为2个segment）
        max_segments = num_edges * 2
        
        segment_llx_list = []
        segment_lly_list = []
        segment_size_x_list = []
        segment_size_y_list = []
        segment_edge_idx_list = []
        segment_is_horizontal_list = []
        
        # 获取numpy用于循环（但保持tensor用于梯度）
        flat_from_np = flat_from.cpu().numpy() if flat_from.is_cuda else flat_from.numpy()
        flat_to_np = flat_to.cpu().numpy() if flat_to.is_cuda else flat_to.numpy()
        l_dir_np = l_directions.cpu().numpy() if l_directions.is_cuda else l_directions.numpy()
        
        half_width = self.wire_width / 2.0
        
        for edge_idx in range(num_edges):
            from_idx = flat_from_np[edge_idx]
            to_idx = flat_to_np[edge_idx]
            
            if from_idx < 0 or to_idx < 0:
                continue
            if from_idx >= len(newx) or to_idx >= len(newx):
                continue
            
            # 获取端点坐标（保持tensor以保持梯度）
            x1, y1 = newx[from_idx], newy[from_idx]
            x2, y2 = newx[to_idx], newy[to_idx]
            
            l_dir = l_dir_np[edge_idx]
            
            # 判断是否是直线（水平或垂直）
            is_horizontal_line = torch.abs(y1 - y2) < 1e-6
            is_vertical_line = torch.abs(x1 - x2) < 1e-6
            # 如果l_dir标记为STRAIGHT但几何上是斜线，强制按L形处理
            is_straight = (is_horizontal_line or is_vertical_line) or (
                l_dir == STRAIGHT and (is_horizontal_line or is_vertical_line)
            )
            
            if is_straight:
                # 直线段：创建一个segment
                seg_llx, seg_lly, seg_sx, seg_sy, seg_is_h = self._create_segment(
                    x1, y1, x2, y2, half_width
                )
                segment_llx_list.append(seg_llx)
                segment_lly_list.append(seg_lly)
                segment_size_x_list.append(seg_sx)
                segment_size_y_list.append(seg_sy)
                segment_edge_idx_list.append(edge_idx)
                segment_is_horizontal_list.append(seg_is_h)
            else:
                # L形：根据方向确定拐点，拆分为两个segment
                # STRAIGHT但为斜线时，默认按H_FIRST处理
                if l_dir == STRAIGHT:
                    l_dir = H_FIRST
                if l_dir == H_FIRST:
                    # 先水平后垂直，拐点在 (x2, y1)
                    corner_x, corner_y = x2, y1
                elif l_dir == V_FIRST:
                    # 先垂直后水平，拐点在 (x1, y2)
                    corner_x, corner_y = x1, y2
                elif l_dir == FAKE_STRAIGHT:
                    # 残余 FAKE_STRAIGHT 仍按 H_FIRST fallback 处理
                    corner_x, corner_y = x2, y1
                elif l_dir == UNKNOWN:
                    # 解析后仍未知的边不参与 L-shape segment 构建
                    continue
                else:
                    # 非法方向值不参与 L-shape segment 构建
                    continue
                
                # Segment 1: p1 -> corner
                seg1_llx, seg1_lly, seg1_sx, seg1_sy, seg1_is_h = self._create_segment(
                    x1, y1, corner_x, corner_y, half_width
                )
                if seg1_sx > 1e-6 or seg1_sy > 1e-6:  # 过滤零尺寸segment
                    segment_llx_list.append(seg1_llx)
                    segment_lly_list.append(seg1_lly)
                    segment_size_x_list.append(seg1_sx)
                    segment_size_y_list.append(seg1_sy)
                    segment_edge_idx_list.append(edge_idx)
                    segment_is_horizontal_list.append(seg1_is_h)
                
                # Segment 2: corner -> p2
                seg2_llx, seg2_lly, seg2_sx, seg2_sy, seg2_is_h = self._create_segment(
                    corner_x, corner_y, x2, y2, half_width
                )
                if seg2_sx > 1e-6 or seg2_sy > 1e-6:  # 过滤零尺寸segment
                    segment_llx_list.append(seg2_llx)
                    segment_lly_list.append(seg2_lly)
                    segment_size_x_list.append(seg2_sx)
                    segment_size_y_list.append(seg2_sy)
                    segment_edge_idx_list.append(edge_idx)
                    segment_is_horizontal_list.append(seg2_is_h)
        
        # 转换为tensor
        if len(segment_llx_list) == 0:
            # 没有有效segment
            empty = torch.tensor([], dtype=dtype, device=device)
            return {
                'segment_llx': empty,
                'segment_lly': empty,
                'segment_size_x': empty,
                'segment_size_y': empty,
                'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
                'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
                'num_segments': 0
            }
        
        segment_llx = torch.stack(segment_llx_list)
        segment_lly = torch.stack(segment_lly_list)
        segment_size_x = torch.stack(segment_size_x_list)
        segment_size_y = torch.stack(segment_size_y_list)
        segment_edge_idx = torch.tensor(segment_edge_idx_list, dtype=torch.long, device=device)
        segment_is_horizontal = torch.tensor(segment_is_horizontal_list, dtype=torch.bool, device=device)
        
        num_segments = len(segment_llx_list)
        logger.info(f"Built {num_segments} segments from {num_edges} edges")
        
        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'num_segments': num_segments
        }
    
    def _create_segment(self, x1, y1, x2, y2, half_width):
        """
        创建一个segment（矩形）
        
        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
            half_width: 半线宽
            
        Returns:
            llx, lly: 左下角坐标
            size_x, size_y: 尺寸
            is_horizontal: 是否是水平segment
        """
        min_x = torch.minimum(x1, x2)
        max_x = torch.maximum(x1, x2)
        min_y = torch.minimum(y1, y2)
        max_y = torch.maximum(y1, y2)
        
        # 判断是水平还是垂直
        dx = torch.abs(x2 - x1)
        dy = torch.abs(y2 - y1)
        is_horizontal = dx >= dy
        
        # 添加线宽
        llx = min_x - half_width
        lly = min_y - half_width
        size_x = (max_x - min_x) + 2 * half_width
        size_y = (max_y - min_y) + 2 * half_width
        
        # 确保最小尺寸
        min_size = half_width * 2 if half_width > 0 else 1e-6
        size_x = torch.maximum(size_x, torch.tensor(min_size, dtype=size_x.dtype, device=size_x.device))
        size_y = torch.maximum(size_y, torch.tensor(min_size, dtype=size_y.dtype, device=size_y.device))
        
        return llx, lly, size_x, size_y, is_horizontal


def build_l_shape_segments_vectorized(newx, newy, flat_from, flat_to, l_directions, wire_width=0.0):
    """
    向量化版本的L形segment构建（更快但需要更多内存）
    
    Args:
        newx: [num_vertices] 所有顶点的x坐标
        newy: [num_vertices] 所有顶点的y坐标
        flat_from: [num_edges] 边的起点索引
        flat_to: [num_edges] 边的终点索引
        l_directions: [num_edges] 每条边的L方向
        wire_width: 线宽
        
    Returns:
        dict with segment information
    """
    device = newx.device
    dtype = newx.dtype
    
    num_edges = flat_from.numel()
    half_width = wire_width / 2.0
    
    # 确保所有tensor在同一设备上
    if flat_from.device != device:
        flat_from = flat_from.to(device)
    if flat_to.device != device:
        flat_to = flat_to.to(device)
    if l_directions.device != device:
        l_directions = l_directions.to(device)
    
    # 过滤无效边
    valid_mask = (flat_from >= 0) & (flat_to >= 0) & (flat_from < len(newx)) & (flat_to < len(newx))
    valid_from = flat_from[valid_mask]
    valid_to = flat_to[valid_mask]
    valid_l_dir = l_directions[valid_mask]
    valid_edge_idx = torch.arange(num_edges, device=device)[valid_mask]
    
    num_valid = valid_from.numel()
    if num_valid == 0:
        empty = torch.tensor([], dtype=dtype, device=device)
        return {
            'segment_llx': empty,
            'segment_lly': empty,
            'segment_size_x': empty,
            'segment_size_y': empty,
            'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
            'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
            'num_segments': 0
        }
    
    # 获取端点坐标
    x1 = newx[valid_from]
    y1 = newy[valid_from]
    x2 = newx[valid_to]
    y2 = newy[valid_to]
    
    # 判断边的类型
    is_horizontal_line = torch.abs(y1 - y2) < 1e-6
    is_vertical_line = torch.abs(x1 - x2) < 1e-6
    # 如果l_dir标记为STRAIGHT但几何上为斜线，强制按L形处理
    straight_by_dir = (valid_l_dir == STRAIGHT)
    diag_straight = straight_by_dir & ~(is_horizontal_line | is_vertical_line)
    is_straight = (is_horizontal_line | is_vertical_line) | (straight_by_dir & ~diag_straight)
    # 对斜线但被标记为STRAIGHT的边，默认按H_FIRST处理。
    # 残余 UNKNOWN 在 segment 阶段跳过，不生成 L-shape。
    is_upper_l = (~is_straight) & (
        (valid_l_dir == H_FIRST) | (valid_l_dir == FAKE_STRAIGHT) | diag_straight
    )
    is_lower_l = (~is_straight) & (valid_l_dir == V_FIRST)
    
    # 计算拐点坐标
    # H_FIRST: corner = (x2, y1)
    # V_FIRST: corner = (x1, y2)
    corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
    corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))
    
    # ===== 构建所有segments =====
    # 对于直线边：1个segment (p1 -> p2)
    # 对于L形边：2个segments (p1 -> corner, corner -> p2)
    
    # Segment类型1: 直线边 或 L形边的第一段
    seg1_x1 = x1
    seg1_y1 = y1
    seg1_x2 = torch.where(is_straight, x2, corner_x)
    seg1_y2 = torch.where(is_straight, y2, corner_y)
    
    seg1_min_x = torch.minimum(seg1_x1, seg1_x2)
    seg1_max_x = torch.maximum(seg1_x1, seg1_x2)
    seg1_min_y = torch.minimum(seg1_y1, seg1_y2)
    seg1_max_y = torch.maximum(seg1_y1, seg1_y2)
    
    seg1_llx = seg1_min_x - half_width
    seg1_lly = seg1_min_y - half_width
    seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
    seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
    seg1_is_h = torch.abs(seg1_x2 - seg1_x1) >= torch.abs(seg1_y2 - seg1_y1)
    
    # Segment类型2: L形边的第二段 (corner -> p2)
    # 只对L形边有效
    seg2_x1 = corner_x
    seg2_y1 = corner_y
    seg2_x2 = x2
    seg2_y2 = y2
    
    seg2_min_x = torch.minimum(seg2_x1, seg2_x2)
    seg2_max_x = torch.maximum(seg2_x1, seg2_x2)
    seg2_min_y = torch.minimum(seg2_y1, seg2_y2)
    seg2_max_y = torch.maximum(seg2_y1, seg2_y2)
    
    seg2_llx = seg2_min_x - half_width
    seg2_lly = seg2_min_y - half_width
    seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
    seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width
    seg2_is_h = torch.abs(seg2_x2 - seg2_x1) >= torch.abs(seg2_y2 - seg2_y1)
    
    # 过滤有效segment2（只有L形边有第二段）
    is_l_shape = is_upper_l | is_lower_l
    seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
    seg1_valid = is_straight | is_l_shape
    
    # 合并所有segments
    # Segment 1 (直线边或L形边的第一段)
    all_llx = [seg1_llx[seg1_valid]]
    all_lly = [seg1_lly[seg1_valid]]
    all_size_x = [seg1_size_x[seg1_valid]]
    all_size_y = [seg1_size_y[seg1_valid]]
    all_edge_idx = [valid_edge_idx[seg1_valid]]
    all_is_h = [seg1_is_h[seg1_valid]]
    
    # Segment 2 (只有L形边)
    if seg2_valid.any():
        all_llx.append(seg2_llx[seg2_valid])
        all_lly.append(seg2_lly[seg2_valid])
        all_size_x.append(seg2_size_x[seg2_valid])
        all_size_y.append(seg2_size_y[seg2_valid])
        all_edge_idx.append(valid_edge_idx[seg2_valid])
        all_is_h.append(seg2_is_h[seg2_valid])
    
    segment_llx = torch.cat(all_llx)
    segment_lly = torch.cat(all_lly)
    segment_size_x = torch.cat(all_size_x)
    segment_size_y = torch.cat(all_size_y)
    segment_edge_idx = torch.cat(all_edge_idx)
    segment_is_horizontal = torch.cat(all_is_h)
    
    # 过滤零尺寸segment
    min_size = max(half_width * 2, 1e-6)
    valid_seg = (segment_size_x > min_size) | (segment_size_y > min_size)
    
    segment_llx = segment_llx[valid_seg]
    segment_lly = segment_lly[valid_seg]
    segment_size_x = segment_size_x[valid_seg]
    segment_size_y = segment_size_y[valid_seg]
    segment_edge_idx = segment_edge_idx[valid_seg]
    segment_is_horizontal = segment_is_horizontal[valid_seg]
    
    num_segments = segment_llx.numel()
    logger.info(f"Built {num_segments} segments from {num_edges} edges (vectorized)")
    
    return {
        'segment_llx': segment_llx,
        'segment_lly': segment_lly,
        'segment_size_x': segment_size_x,
        'segment_size_y': segment_size_y,
        'segment_edge_idx': segment_edge_idx,
        'segment_is_horizontal': segment_is_horizontal,
        'num_segments': num_segments
    }


def build_segment_pos_tensor(segment_llx, segment_lly):
    """
    将segment坐标转换为pos tensor格式 (与BBoxElectricPotential兼容)
    
    Args:
        segment_llx: [num_segments]
        segment_lly: [num_segments]
        
    Returns:
        pos: [num_segments * 2] 格式为 [llx1, llx2, ..., lly1, lly2, ...]
    """
    return torch.cat([segment_llx, segment_lly])


class LShapeSegmentOp:
    """
    可微的L形segment操作
    
    使用方式：
        op = LShapeSegmentOp(wire_width=100.0)
        result = op(newx, newy, flat_from, flat_to, l_directions)
        segment_pos = result['segment_pos']  # 用于密度计算
        
    优化：预计算拓扑结构，只在坐标更新时重新计算segment位置
    """
    
    def __init__(self, wire_width=0.0, use_vectorized=True, soft_min_weight=0.0):
        self.wire_width = wire_width
        self.use_vectorized = use_vectorized
        self.soft_min_weight = float(soft_min_weight)
        self.builder = LShapeSegmentBuilder(wire_width)
        
        # 缓存拓扑结构（不随pos变化）
        self._cached_topology = None
        # 缓存输入的hash，用于检测EGR更新
        self._cached_input_hash = None
    
    def reset_cache(self):
        """重置拓扑缓存，在EGR重新运行后调用"""
        self._cached_topology = None
        self._cached_input_hash = None
        logger.info("LShapeSegmentOp cache reset")
    
    def _compute_input_hash(self, flat_from, flat_to, l_directions):
        """计算输入的hash值，用于检测拓扑变化"""
        # 使用边的部分数据来快速检测变化
        # 检查：边数量 + 前/后几条边的内容 + l_directions的sum
        num_edges = flat_from.numel()
        if num_edges == 0:
            return (0, 0, 0, 0)
        
        # 采样检查（避免对整个tensor计算hash）
        sample_size = min(10, num_edges)
        from_sample = flat_from[:sample_size].sum().item()
        to_sample = flat_to[:sample_size].sum().item()
        l_dir_sum = l_directions.sum().item()
        
        return (num_edges, from_sample, to_sample, l_dir_sum)

    def _empty_segment_result(self, dtype, device):
        empty = torch.tensor([], dtype=dtype, device=device)
        return {
            'segment_llx': empty,
            'segment_lly': empty,
            'segment_size_x': empty,
            'segment_size_y': empty,
            'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
            'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
            'segment_weight': empty,
            'num_segments': 0
        }

    def _create_segment_batch(self, x1, y1, x2, y2, half_width):
        min_x = torch.minimum(x1, x2)
        max_x = torch.maximum(x1, x2)
        min_y = torch.minimum(y1, y2)
        max_y = torch.maximum(y1, y2)

        llx = min_x - half_width
        lly = min_y - half_width
        size_x = (max_x - min_x) + 2 * half_width
        size_y = (max_y - min_y) + 2 * half_width
        is_horizontal = torch.abs(x2 - x1) >= torch.abs(y2 - y1)
        return llx, lly, size_x, size_y, is_horizontal
    
    def _compute_topology(self, flat_from, flat_to, l_directions, num_vertices, device):
        """
        预计算拓扑结构（只需要计算一次）
        
        Args:
            flat_from, flat_to: 边的端点索引
            l_directions: L方向
            num_vertices: 顶点数量
            device: 目标设备（应与newx/newy一致）
        
        Returns:
            dict with precomputed topology info
        """
        num_edges = flat_from.numel()
        half_width = self.wire_width / 2.0
        
        # 确保所有输入在同一设备上
        if flat_from.device != device:
            flat_from = flat_from.to(device)
        if flat_to.device != device:
            flat_to = flat_to.to(device)
        if l_directions.device != device:
            l_directions = l_directions.to(device)
        
        # 过滤无效边
        valid_mask = (flat_from >= 0) & (flat_to >= 0) & (flat_from < num_vertices) & (flat_to < num_vertices)
        valid_from = flat_from[valid_mask]
        valid_to = flat_to[valid_mask]
        valid_l_dir = l_directions[valid_mask]
        valid_edge_idx = torch.arange(num_edges, device=device)[valid_mask]
        
        num_valid = valid_from.numel()
        
        if num_valid == 0:
            return {
                'valid': False,
                'num_valid': 0
            }

        return {
            'valid': True,
            'num_valid': num_valid,
            'valid_mask': valid_mask,
            'valid_from': valid_from,
            'valid_to': valid_to,
            'valid_edge_idx': valid_edge_idx,
            'half_width': half_width,
            'num_edges': num_edges
        }

    def _compute_segments_hard(self, newx, newy, topo, l_directions):
        """
        使用 hard L-direction 计算 segment
        """
        device = newx.device
        dtype = newx.dtype
        half_width = topo['half_width']

        if not topo['valid']:
            return self._empty_segment_result(dtype, device)

        # 确保索引在同一设备上
        valid_from = topo['valid_from']
        valid_to = topo['valid_to']
        valid_edge_idx = topo['valid_edge_idx']
        # 关键：将索引移到与newx相同的设备
        if valid_from.device != device:
            valid_from = valid_from.to(device)
            valid_to = valid_to.to(device)
            valid_edge_idx = valid_edge_idx.to(device)
        valid_mask = topo['valid_mask']
        if valid_mask.device != device:
            valid_mask = valid_mask.to(device)
        if l_directions.device != device:
            l_directions = l_directions.to(device)
        valid_l_dir = l_directions[valid_mask]
        is_h_first = (valid_l_dir == H_FIRST) | (valid_l_dir == FAKE_STRAIGHT)
        is_v_first = (valid_l_dir == V_FIRST)
        is_straight_by_dir = (valid_l_dir == STRAIGHT)

        # 获取端点坐标
        x1 = newx[valid_from]
        y1 = newy[valid_from]
        x2 = newx[valid_to]
        y2 = newy[valid_to]
        
        # 判断几何上的直线
        is_horizontal_line = torch.abs(y1 - y2) < 1e-6
        is_vertical_line = torch.abs(x1 - x2) < 1e-6
        # 如果l_dir标记为STRAIGHT但几何上为斜线，强制按L形处理
        diag_straight = is_straight_by_dir & ~(is_horizontal_line | is_vertical_line)
        is_straight = (is_horizontal_line | is_vertical_line) | (is_straight_by_dir & ~diag_straight)
        
        # 对斜线但被标记为STRAIGHT的边，默认按H_FIRST处理。
        # 残余 UNKNOWN 在 segment 阶段跳过，不生成 L-shape。
        is_upper_l = (~is_straight) & (is_h_first | diag_straight)
        is_lower_l = (~is_straight) & is_v_first
        
        # 计算拐点坐标
        corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
        corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))
        
        # Segment 1: 所有边都有 (p1 -> corner/p2)
        seg1_x2 = torch.where(is_straight, x2, corner_x)
        seg1_y2 = torch.where(is_straight, y2, corner_y)
        
        seg1_min_x = torch.minimum(x1, seg1_x2)
        seg1_max_x = torch.maximum(x1, seg1_x2)
        seg1_min_y = torch.minimum(y1, seg1_y2)
        seg1_max_y = torch.maximum(y1, seg1_y2)
        
        seg1_llx = seg1_min_x - half_width
        seg1_lly = seg1_min_y - half_width
        seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
        seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
        seg1_is_h = torch.abs(seg1_x2 - x1) >= torch.abs(seg1_y2 - y1)
        
        # Segment 2: 只有L形边有 (corner -> p2)
        is_l_shape = is_upper_l | is_lower_l
        seg1_valid = is_straight | is_l_shape
        
        seg2_min_x = torch.minimum(corner_x, x2)
        seg2_max_x = torch.maximum(corner_x, x2)
        seg2_min_y = torch.minimum(corner_y, y2)
        seg2_max_y = torch.maximum(corner_y, y2)
        
        seg2_llx = seg2_min_x - half_width
        seg2_lly = seg2_min_y - half_width
        seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
        seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width
        seg2_is_h = torch.abs(x2 - corner_x) >= torch.abs(y2 - corner_y)
        
        seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
        
        # 合并所有segments
        all_llx = [seg1_llx[seg1_valid]]
        all_lly = [seg1_lly[seg1_valid]]
        all_size_x = [seg1_size_x[seg1_valid]]
        all_size_y = [seg1_size_y[seg1_valid]]
        all_edge_idx = [valid_edge_idx[seg1_valid]]
        all_is_h = [seg1_is_h[seg1_valid]]
        all_weight = [torch.ones_like(seg1_size_x[seg1_valid])]
        
        if seg2_valid.any():
            all_llx.append(seg2_llx[seg2_valid])
            all_lly.append(seg2_lly[seg2_valid])
            all_size_x.append(seg2_size_x[seg2_valid])
            all_size_y.append(seg2_size_y[seg2_valid])
            all_edge_idx.append(valid_edge_idx[seg2_valid])
            all_is_h.append(seg2_is_h[seg2_valid])
            all_weight.append(torch.ones_like(seg2_size_x[seg2_valid]))
        
        segment_llx = torch.cat(all_llx)
        segment_lly = torch.cat(all_lly)
        segment_size_x = torch.cat(all_size_x)
        segment_size_y = torch.cat(all_size_y)
        segment_edge_idx = torch.cat(all_edge_idx)
        segment_is_horizontal = torch.cat(all_is_h)
        segment_weight = torch.cat(all_weight)
        
        # 过滤零尺寸segment
        min_size = max(half_width * 2, 1e-6)
        valid_seg = (segment_size_x > min_size) | (segment_size_y > min_size)
        
        segment_llx = segment_llx[valid_seg]
        segment_lly = segment_lly[valid_seg]
        segment_size_x = segment_size_x[valid_seg]
        segment_size_y = segment_size_y[valid_seg]
        segment_edge_idx = segment_edge_idx[valid_seg]
        segment_is_horizontal = segment_is_horizontal[valid_seg]
        segment_weight = segment_weight[valid_seg]
        
        num_segments = segment_llx.numel()
        
        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'segment_weight': segment_weight,
            'num_segments': num_segments
        }

    def _compute_segments_soft(self, newx, newy, topo, soft_l_weights):
        """
        使用 soft H/V 权重生成候选 segment
        """
        device = newx.device
        dtype = newx.dtype
        half_width = topo['half_width']

        if not topo['valid']:
            return self._empty_segment_result(dtype, device)

        valid_from = topo['valid_from']
        valid_to = topo['valid_to']
        valid_edge_idx = topo['valid_edge_idx']
        valid_mask = topo['valid_mask']
        if valid_from.device != device:
            valid_from = valid_from.to(device)
            valid_to = valid_to.to(device)
            valid_edge_idx = valid_edge_idx.to(device)
        if valid_mask.device != device:
            valid_mask = valid_mask.to(device)
        if soft_l_weights.device != device:
            soft_l_weights = soft_l_weights.to(device)

        valid_soft = soft_l_weights[valid_mask].to(dtype=dtype)
        if valid_soft.dim() != 2 or valid_soft.size(1) != 2:
            raise ValueError("soft_l_weights must have shape [num_edges, 2]")

        valid_soft = torch.clamp(valid_soft, min=0.0)
        weight_sum = valid_soft.sum(dim=1, keepdim=True)
        fallback = torch.full_like(valid_soft, 0.5)
        valid_soft = torch.where(weight_sum > 1e-12, valid_soft / weight_sum.clamp_min(1e-12), fallback)

        x1 = newx[valid_from]
        y1 = newy[valid_from]
        x2 = newx[valid_to]
        y2 = newy[valid_to]

        is_horizontal_line = torch.abs(y1 - y2) < 1e-6
        is_vertical_line = torch.abs(x1 - x2) < 1e-6
        is_straight = is_horizontal_line | is_vertical_line
        is_diagonal = ~is_straight

        all_llx = []
        all_lly = []
        all_size_x = []
        all_size_y = []
        all_edge_idx = []
        all_is_h = []
        all_weight = []

        def append_group(llx, lly, size_x, size_y, edge_idx, is_h, weight, valid):
            if valid.any():
                all_llx.append(llx[valid])
                all_lly.append(lly[valid])
                all_size_x.append(size_x[valid])
                all_size_y.append(size_y[valid])
                all_edge_idx.append(edge_idx[valid])
                all_is_h.append(is_h[valid])
                all_weight.append(weight[valid])

        straight_llx, straight_lly, straight_size_x, straight_size_y, straight_is_h = self._create_segment_batch(
            x1, y1, x2, y2, half_width
        )
        append_group(
            straight_llx,
            straight_lly,
            straight_size_x,
            straight_size_y,
            valid_edge_idx,
            straight_is_h,
            torch.ones_like(straight_size_x),
            is_straight,
        )

        h_weight = valid_soft[:, 0]
        v_weight = valid_soft[:, 1]
        min_weight = max(self.soft_min_weight, 0.0)

        h_corner_x = x2
        h_corner_y = y1
        h_seg1_llx, h_seg1_lly, h_seg1_size_x, h_seg1_size_y, h_seg1_is_h = self._create_segment_batch(
            x1, y1, h_corner_x, h_corner_y, half_width
        )
        h_seg2_llx, h_seg2_lly, h_seg2_size_x, h_seg2_size_y, h_seg2_is_h = self._create_segment_batch(
            h_corner_x, h_corner_y, x2, y2, half_width
        )
        h_valid = is_diagonal & (h_weight > min_weight)
        append_group(h_seg1_llx, h_seg1_lly, h_seg1_size_x, h_seg1_size_y, valid_edge_idx, h_seg1_is_h, h_weight, h_valid)
        append_group(h_seg2_llx, h_seg2_lly, h_seg2_size_x, h_seg2_size_y, valid_edge_idx, h_seg2_is_h, h_weight, h_valid)

        v_corner_x = x1
        v_corner_y = y2
        v_seg1_llx, v_seg1_lly, v_seg1_size_x, v_seg1_size_y, v_seg1_is_h = self._create_segment_batch(
            x1, y1, v_corner_x, v_corner_y, half_width
        )
        v_seg2_llx, v_seg2_lly, v_seg2_size_x, v_seg2_size_y, v_seg2_is_h = self._create_segment_batch(
            v_corner_x, v_corner_y, x2, y2, half_width
        )
        v_valid = is_diagonal & (v_weight > min_weight)
        append_group(v_seg1_llx, v_seg1_lly, v_seg1_size_x, v_seg1_size_y, valid_edge_idx, v_seg1_is_h, v_weight, v_valid)
        append_group(v_seg2_llx, v_seg2_lly, v_seg2_size_x, v_seg2_size_y, valid_edge_idx, v_seg2_is_h, v_weight, v_valid)

        if not all_llx:
            return self._empty_segment_result(dtype, device)

        segment_llx = torch.cat(all_llx)
        segment_lly = torch.cat(all_lly)
        segment_size_x = torch.cat(all_size_x)
        segment_size_y = torch.cat(all_size_y)
        segment_edge_idx = torch.cat(all_edge_idx)
        segment_is_horizontal = torch.cat(all_is_h)
        segment_weight = torch.cat(all_weight)

        min_size = max(half_width * 2, 1e-6)
        valid_seg = ((segment_size_x > min_size) | (segment_size_y > min_size)) & (segment_weight > min_weight)

        segment_llx = segment_llx[valid_seg]
        segment_lly = segment_lly[valid_seg]
        segment_size_x = segment_size_x[valid_seg]
        segment_size_y = segment_size_y[valid_seg]
        segment_edge_idx = segment_edge_idx[valid_seg]
        segment_is_horizontal = segment_is_horizontal[valid_seg]
        segment_weight = segment_weight[valid_seg]

        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'segment_weight': segment_weight,
            'num_segments': segment_llx.numel()
        }

    def __call__(self, newx, newy, flat_from, flat_to, l_directions, soft_l_weights=None):
        """
        构建L形segments
        
        Returns:
            dict with:
                segment_pos: [num_segments * 2] 位置tensor
                segment_size_x: [num_segments]
                segment_size_y: [num_segments]
                ...
        """
        device = newx.device
        
        # 计算输入hash，检测EGR是否重新运行
        current_hash = self._compute_input_hash(flat_from, flat_to, l_directions)
        
        # 检查是否需要重新计算拓扑
        need_recompute_topo = (
            self._cached_topology is None or
            self._cached_input_hash != current_hash
        )
        
        if need_recompute_topo:
            # 预计算拓扑（只需一次，或EGR更新后重新计算）
            self._cached_topology = self._compute_topology(
                flat_from, flat_to, l_directions, len(newx), device
            )
            self._cached_input_hash = current_hash
            logger.info(f"Computed topology: {self._cached_topology['num_valid']} valid edges (hash={current_hash[0]})")

        # 快速计算segment坐标
        if soft_l_weights is None:
            result = self._compute_segments_hard(newx, newy, self._cached_topology, l_directions)
            mode_name = "hard"
        else:
            result = self._compute_segments_soft(newx, newy, self._cached_topology, soft_l_weights)
            mode_name = "soft"

        # 只在第一次打印详细日志
        if need_recompute_topo and result['num_segments'] > 0:
            logger.info(
                f"Built {result['num_segments']} segments from {flat_from.numel()} edges (vectorized, mode={mode_name})"
            )

        # 添加pos tensor
        if result['num_segments'] > 0:
            result['segment_pos'] = build_segment_pos_tensor(
                result['segment_llx'], result['segment_lly']
            )
        else:
            result['segment_pos'] = torch.tensor([], dtype=newx.dtype, device=newx.device)

        return result
