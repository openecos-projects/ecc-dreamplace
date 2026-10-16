##
# @file   segment_density.py
# @brief  Compute routing density from L-shape segments
#         Supports multiple density models: RUDY, overlap, electric potential
#

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import logging
import time

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform
import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

logger = logging.getLogger(__name__)


def compute_segment_rudy_density(
    segment_pos, segment_size_x, segment_size_y,
    segment_weight,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y
):
    """
    计算segment的RUDY密度图
    
    RUDY (Rectangular Uniform Density Yield):
    每个segment的demand = (width + height) / area
    然后按面积重叠分布到bins
    
    Args:
        segment_pos: [num_segments * 2] 格式为 [llx1, ..., lly1, ...]
        segment_size_x: [num_segments]
        segment_size_y: [num_segments]
        xl, yl, xh, yh: die边界
        bin_size_x, bin_size_y: bin尺寸
        num_bins_x, num_bins_y: bin数量
        
    Returns:
        density_map: [num_bins_x, num_bins_y]
    """
    device = segment_pos.device
    dtype = segment_pos.dtype
    
    num_segments = segment_size_x.numel()
    if num_segments == 0:
        return torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)
    
    # 提取坐标
    seg_llx = segment_pos[:num_segments]
    seg_lly = segment_pos[num_segments:]
    seg_urx = seg_llx + segment_size_x
    seg_ury = seg_lly + segment_size_y
    
    # 计算RUDY demand
    seg_area = segment_size_x * segment_size_y
    seg_demand = (segment_size_x + segment_size_y) / torch.clamp(seg_area, min=1e-8)
    if isinstance(segment_weight, torch.Tensor):
        if segment_weight.device != seg_demand.device:
            segment_weight = segment_weight.to(seg_demand.device)
        seg_demand = seg_demand * segment_weight
    
    # 预计算bin边界
    bin_xl_coords = torch.arange(num_bins_x, device=device, dtype=dtype) * bin_size_x + xl
    bin_yl_coords = torch.arange(num_bins_y, device=device, dtype=dtype) * bin_size_y + yl
    bin_xh_coords = bin_xl_coords + bin_size_x
    bin_yh_coords = bin_yl_coords + bin_size_y
    
    # 初始化density map
    density_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)
    
    # 批量处理
    batch_size = 128
    for batch_start in range(0, num_segments, batch_size):
        batch_end = min(batch_start + batch_size, num_segments)
        
        batch_llx = seg_llx[batch_start:batch_end]
        batch_lly = seg_lly[batch_start:batch_end]
        batch_urx = seg_urx[batch_start:batch_end]
        batch_ury = seg_ury[batch_start:batch_end]
        batch_demand = seg_demand[batch_start:batch_end]
        batch_area = seg_area[batch_start:batch_end]
        
        # 计算每个segment影响的bin范围
        bin_xl_idx = torch.clamp(((batch_llx - xl) / bin_size_x).floor().long(), 0, num_bins_x - 1)
        bin_yl_idx = torch.clamp(((batch_lly - yl) / bin_size_y).floor().long(), 0, num_bins_y - 1)
        bin_xh_idx = torch.clamp(((batch_urx - xl) / bin_size_x).ceil().long(), 0, num_bins_x)
        bin_yh_idx = torch.clamp(((batch_ury - yl) / bin_size_y).ceil().long(), 0, num_bins_y)
        
        # 逐segment处理（避免索引冲突）
        for i in range(batch_end - batch_start):
            bx_start = bin_xl_idx[i].item()
            bx_end = bin_xh_idx[i].item()
            by_start = bin_yl_idx[i].item()
            by_end = bin_yh_idx[i].item()
            
            if bx_start >= bx_end or by_start >= by_end:
                continue
            
            # 计算重叠
            affected_bin_xl = bin_xl_coords[bx_start:bx_end]
            affected_bin_yl = bin_yl_coords[by_start:by_end]
            affected_bin_xh = bin_xh_coords[bx_start:bx_end]
            affected_bin_yh = bin_yh_coords[by_start:by_end]
            
            # 向量化计算重叠面积
            overlap_xl = torch.maximum(batch_llx[i], affected_bin_xl.unsqueeze(1))
            overlap_yl = torch.maximum(batch_lly[i], affected_bin_yl.unsqueeze(0))
            overlap_xh = torch.minimum(batch_urx[i], affected_bin_xh.unsqueeze(1))
            overlap_yh = torch.minimum(batch_ury[i], affected_bin_yh.unsqueeze(0))
            
            overlap_area = torch.clamp(overlap_xh - overlap_xl, min=0) * torch.clamp(overlap_yh - overlap_yl, min=0)
            
            # 按面积比例分配demand
            overlap_ratio = overlap_area / torch.clamp(batch_area[i], min=1e-8)
            density_map[bx_start:bx_end, by_start:by_end] += overlap_ratio * batch_demand[i]
    
    return density_map


def compute_segment_overlap_density(
    segment_pos, segment_size_x, segment_size_y,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y
):
    """
    计算segment的面积重叠密度图（更简单的密度模型）
    
    每个segment按其与bin的重叠面积贡献密度
    
    Returns:
        density_map: [num_bins_x, num_bins_y]
    """
    device = segment_pos.device
    dtype = segment_pos.dtype
    
    num_segments = segment_size_x.numel()
    if num_segments == 0:
        return torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)
    
    seg_llx = segment_pos[:num_segments]
    seg_lly = segment_pos[num_segments:]
    seg_urx = seg_llx + segment_size_x
    seg_ury = seg_lly + segment_size_y
    
    # 预计算bin边界
    bin_xl_coords = torch.arange(num_bins_x, device=device, dtype=dtype) * bin_size_x + xl
    bin_yl_coords = torch.arange(num_bins_y, device=device, dtype=dtype) * bin_size_y + yl
    bin_xh_coords = bin_xl_coords + bin_size_x
    bin_yh_coords = bin_yl_coords + bin_size_y
    
    density_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)
    
    # 批量处理
    for i in range(num_segments):
        bx_start = max(0, int((seg_llx[i] - xl) / bin_size_x))
        bx_end = min(num_bins_x, int(np.ceil((seg_urx[i] - xl) / bin_size_x)))
        by_start = max(0, int((seg_lly[i] - yl) / bin_size_y))
        by_end = min(num_bins_y, int(np.ceil((seg_ury[i] - yl) / bin_size_y)))
        
        if bx_start >= bx_end or by_start >= by_end:
            continue
        
        for bx in range(bx_start, bx_end):
            for by in range(by_start, by_end):
                ovlp_xl = max(seg_llx[i].item(), bin_xl_coords[bx].item())
                ovlp_xh = min(seg_urx[i].item(), bin_xh_coords[bx].item())
                ovlp_yl = max(seg_lly[i].item(), bin_yl_coords[by].item())
                ovlp_yh = min(seg_ury[i].item(), bin_yh_coords[by].item())
                
                ovlp_area = max(0, ovlp_xh - ovlp_xl) * max(0, ovlp_yh - ovlp_yl)
                density_map[bx, by] += ovlp_area
    
    return density_map


class SegmentDensityFunction(Function):
    """
    可微的segment密度计算（用于优化）
    
    Forward: 计算密度图和电势场能量
    Backward: 计算梯度
    """
    
    @staticmethod
    def forward(
        ctx,
        segment_pos,
        segment_size_x,
        segment_size_y,
        segment_weight,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        exact_expkM, exact_expkN,
        inv_wu2_plus_wv2,
        wu_by_wu2_plus_wv2_half,
        wv_by_wu2_plus_wv2_half,
        dct2_op, idct_idxst_op, idxst_idct_op
    ):
        """
        计算segment密度和电势场能量
        """
        tt = time.time()
        
        # 计算密度图
        density_map = compute_segment_rudy_density(
            segment_pos, segment_size_x, segment_size_y, segment_weight,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y
        )
        
        # 保存用于backward
        ctx.save_for_tensor = segment_pos
        ctx.segment_size_x = segment_size_x
        ctx.segment_size_y = segment_size_y
        ctx.segment_weight = segment_weight
        ctx.xl, ctx.yl = xl, yl
        ctx.xh, ctx.yh = xh, yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        
        # 使用DCT计算电势场
        M, N = num_bins_x, num_bins_y
        
        # DCT变换
        density_map_dct = dct2_op(density_map)
        
        # 求解泊松方程 (频域)
        potential_map_dct = density_map_dct * inv_wu2_plus_wv2
        
        # 计算电场 (频域)
        field_map_x_dct = density_map_dct * wu_by_wu2_plus_wv2_half
        field_map_y_dct = density_map_dct * wv_by_wu2_plus_wv2_half
        
        # IDCT/IDXST变换得到电场
        field_map_x = idxst_idct_op(field_map_x_dct)
        field_map_y = idct_idxst_op(field_map_y_dct)
        
        # 保存电场用于backward
        ctx.field_map_x = field_map_x
        ctx.field_map_y = field_map_y
        ctx.density_map = density_map
        
        # 计算能量 (sum of density * potential)
        # 简化：使用密度图的平方和作为cost
        energy = (density_map ** 2).sum()
        
        logger.debug(f"Segment density forward: {(time.time() - tt) * 1000:.2f} ms")
        
        return energy
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        计算梯度
        """
        tt = time.time()
        
        segment_pos = ctx.save_for_tensor
        segment_size_x = ctx.segment_size_x
        segment_size_y = ctx.segment_size_y
        field_map_x = ctx.field_map_x
        field_map_y = ctx.field_map_y
        
        xl, yl = ctx.xl, ctx.yl
        bin_size_x = ctx.bin_size_x
        bin_size_y = ctx.bin_size_y
        num_bins_x = ctx.num_bins_x
        num_bins_y = ctx.num_bins_y
        
        num_segments = segment_size_x.numel()
        if num_segments == 0:
            # forward有20个参数: segment_pos, segment_size_x, segment_size_y, segment_weight,
            # xl, yl, xh, yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y,
            # exact_expkM, exact_expkN, inv_wu2_plus_wv2, wu_by_wu2_plus_wv2_half,
            # wv_by_wu2_plus_wv2_half, dct2_op, idct_idxst_op, idxst_idct_op
            return (torch.zeros_like(segment_pos),) + (None,) * 19
        
        device = segment_pos.device
        dtype = segment_pos.dtype
        
        # 计算每个segment中心对应的电场
        seg_llx = segment_pos[:num_segments]
        seg_lly = segment_pos[num_segments:]
        seg_cx = seg_llx + segment_size_x / 2
        seg_cy = seg_lly + segment_size_y / 2
        
        # 计算segment中心所在的bin
        bin_x = ((seg_cx - xl) / bin_size_x).long().clamp(0, num_bins_x - 1)
        bin_y = ((seg_cy - yl) / bin_size_y).long().clamp(0, num_bins_y - 1)
        
        # 获取电场值
        grad_x = field_map_x[bin_x, bin_y]
        grad_y = field_map_y[bin_x, bin_y]
        segment_weight = ctx.segment_weight
        if isinstance(segment_weight, torch.Tensor):
            if segment_weight.device != grad_x.device:
                segment_weight = segment_weight.to(grad_x.device)
            grad_x = grad_x * segment_weight
            grad_y = grad_y * segment_weight
        
        # 组合梯度
        grad_pos = torch.cat([grad_x, grad_y]) * grad_output
        
        logger.debug(f"Segment density backward: {(time.time() - tt) * 1000:.2f} ms")
        
        # forward有20个参数，返回20个梯度
        return (grad_pos,) + (None,) * 19


class SegmentDensityOp(nn.Module):
    """
    Segment密度计算模块
    
    将L形segments的密度转换为可微的电势场能量
    """
    
    def __init__(
        self,
        xl, yl, xh, yh,
        num_bins_x=64, num_bins_y=64
    ):
        super(SegmentDensityOp, self).__init__()
        
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        
        # DCT相关（延迟初始化）
        self.exact_expkM = None
        self.exact_expkN = None
        self.dct2 = None
        self.idct_idxst = None
        self.idxst_idct = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None
    
    def _init_dct(self, device, dtype):
        """初始化DCT相关参数"""
        M = self.num_bins_x
        N = self.num_bins_y
        
        self.exact_expkM = precompute_expk(M, dtype=dtype, device=device)
        self.exact_expkN = precompute_expk(N, dtype=dtype, device=device)
        
        self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
        self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM, self.exact_expkN)
        self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM, self.exact_expkN)
        
        # wu和wv
        wu = torch.arange(M, dtype=dtype, device=device).mul(2 * np.pi / M).view([M, 1])
        wv = torch.arange(N, dtype=dtype, device=device).mul(2 * np.pi / N).view([1, N])
        wv = wv * (self.bin_size_x / self.bin_size_y)
        
        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0, 0] = 1.0  # 避免除零
        
        self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        self.inv_wu2_plus_wv2[0, 0] = 0.0
        
        self.wu_by_wu2_plus_wv2_half = wu * self.inv_wu2_plus_wv2 * 0.5
        self.wv_by_wu2_plus_wv2_half = wv * self.inv_wu2_plus_wv2 * 0.5
    
    def forward(self, segment_pos, segment_size_x, segment_size_y, mode="density", segment_weight=None):
        """
        计算segment密度
        
        Args:
            segment_pos: [num_segments * 2]
            segment_size_x: [num_segments]
            segment_size_y: [num_segments]
            mode: "density" 或 "energy"
            
        Returns:
            如果mode=="density": 返回density_map
            如果mode=="energy": 返回可微的能量值
        """
        if segment_size_x.numel() == 0:
            if mode == "density":
                return torch.zeros(
                    self.num_bins_x, self.num_bins_y,
                    dtype=segment_pos.dtype, device=segment_pos.device
                )
            else:
                return torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device)
        
        if mode == "density":
            return compute_segment_rudy_density(
                segment_pos, segment_size_x, segment_size_y, segment_weight,
                self.xl, self.yl, self.xh, self.yh,
                self.bin_size_x, self.bin_size_y,
                self.num_bins_x, self.num_bins_y
            )
        else:
            # 初始化DCT
            if self.dct2 is None:
                self._init_dct(segment_pos.device, segment_pos.dtype)
            
            return SegmentDensityFunction.apply(
                segment_pos, segment_size_x, segment_size_y, segment_weight,
                self.xl, self.yl, self.xh, self.yh,
                self.bin_size_x, self.bin_size_y,
                self.num_bins_x, self.num_bins_y,
                self.exact_expkM, self.exact_expkN,
                self.inv_wu2_plus_wv2,
                self.wu_by_wu2_plus_wv2_half,
                self.wv_by_wu2_plus_wv2_half,
                self.dct2, self.idct_idxst, self.idxst_idct
            )


def create_segment_density_op(placedb, num_bins_x=64, num_bins_y=64):
    """
    工厂函数：创建SegmentDensityOp
    
    Args:
        placedb: placement database
        num_bins_x, num_bins_y: bin数量
        
    Returns:
        SegmentDensityOp
    """
    return SegmentDensityOp(
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y
    )
