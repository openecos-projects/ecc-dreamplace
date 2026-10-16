##
# @file   bbox_electric_potential.py
# @author Shawn
# @date   Aug 2025
# @brief 
#

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import logging
# import cProfile
# import pstats
from functools import wraps

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform

import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

#import dreamplace.ops.dct.dct as dct
#from dreamplace.ops.dct.discrete_spectral_transform import get_expk as precompute_expk

from dreamplace.ops.electric_potential.electric_overflow import ElectricDensityMapFunction as ElectricDensityMapFunction
from dreamplace.ops.electric_potential.electric_overflow import ElectricOverflow as ElectricOverflow

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

import pdb
import matplotlib
# matplotlib.use('Agg')

from dreamplace.ops.routability.plot_map import plot_density_map
from dreamplace.ops.routability.plot_map import plot_bboxes_on_die
from dreamplace.ops.routability.plot_map import plot_density_with_bboxes

logger = logging.getLogger(__name__)

# global variable for plot
plot_count = 0

# Performance profiling utilities (DISABLED)
# def profile_method(func):
#     """Decorator for profiling methods"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         profiler = cProfile.Profile()
#         profiler.enable()
#         try:
#             result = func(*args, **kwargs)
#         finally:
#             profiler.disable()
#             # Save profile to file
#             profile_path = f"/tmp/bbox_profile_{func.__name__}.prof"
#             profiler.dump_stats(profile_path)
#             
#             # Print top 20 time-consuming functions
#             stats = pstats.Stats(profiler)
#             stats.sort_stats('cumulative')
#             print(f"\n=== Profile for {func.__name__} ===")
#             stats.print_stats(20)
#             print(f"Profile saved to: {profile_path}")
#         return result
#     return wrapper

# def time_method(func):
#     """Decorator for timing methods"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"{func.__name__} took {(end_time - start_time)*1000:.2f} ms")
#         return result
#     return wrapper

# def torch_profile_method(func):
#     """Decorator for PyTorch profiling"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             from torch.profiler import profile, record_function, ProfilerActivity
#             with profile(
#                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                 record_shapes=True,
#                 profile_memory=True,
#                 with_stack=True
#             ) as prof:
#                 with record_function(f"{func.__name__}"):
#                     result = func(*args, **kwargs)
#             
#             # Save profile
#             profile_path = f"/tmp/torch_profile_{func.__name__}.json"
#             prof.export_chrome_trace(profile_path)
#             print(f"PyTorch profile saved to: {profile_path}")
#             
#             # Print summary
#             print(f"\n=== PyTorch Profile for {func.__name__} ===")
#             print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))
#             
#         except ImportError:
#             print("PyTorch profiler not available, using basic timing")
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
#             print(f"{func.__name__} took {(end_time - start_time)*1000:.2f} ms")
#         
#         return result
#     return wrapper




def compute_bbox_rudy_density_map_vectorized(
    pos,
    node_size_x_clamped,
    node_size_y_clamped,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y,
):
    device = pos.device
    dtype = pos.dtype
    num_bboxes = pos.numel() // 2
    
    # Bbox位置和尺寸
    bbox_xl = pos[:num_bboxes]
    bbox_yl = pos[num_bboxes:]
    bbox_size_x = node_size_x_clamped
    bbox_size_y = node_size_y_clamped
    bbox_xh = bbox_xl + bbox_size_x
    bbox_yh = bbox_yl + bbox_size_y
    
    # 计算每个bbox的routing demand
    bbox_width = bbox_size_x
    bbox_height = bbox_size_y
    bbox_area = bbox_width * bbox_height
    bbox_demand = (bbox_width + bbox_height) / torch.clamp(bbox_area, min=1e-8)    

    # 预计算bin边界
    bin_xl_coords = torch.arange(num_bins_x, device=device, dtype=dtype) * bin_size_x + xl
    bin_yl_coords = torch.arange(num_bins_y, device=device, dtype=dtype) * bin_size_y + yl
    bin_xh_coords = bin_xl_coords + bin_size_x
    bin_yh_coords = bin_yl_coords + bin_size_y
    
    # 初始化routing demand map
    routing_demand_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)
    
    # 批量处理bbox，避免内存爆炸
    batch_size = 64  # 可调参数
    for batch_start in range(0, num_bboxes, batch_size):
        batch_end = min(batch_start + batch_size, num_bboxes)
        batch_size_actual = batch_end - batch_start
        
        # 当前批次的bbox
        batch_bbox_xl = bbox_xl[batch_start:batch_end]  # [batch_size]
        batch_bbox_yl = bbox_yl[batch_start:batch_end]  # [batch_size]
        batch_bbox_xh = bbox_xh[batch_start:batch_end]  # [batch_size]
        batch_bbox_yh = bbox_yh[batch_start:batch_end]  # [batch_size]
        batch_demand = bbox_demand[batch_start:batch_end]  # [batch_size]
        batch_area = bbox_area[batch_start:batch_end]  # [batch_size]
        
        # 计算每个bbox影响的bin范围
        bin_xl_idx = torch.clamp(((batch_bbox_xl - xl) / bin_size_x).floor().long(), 0, num_bins_x-1)
        bin_yl_idx = torch.clamp(((batch_bbox_yl - yl) / bin_size_y).floor().long(), 0, num_bins_y-1)
        bin_xh_idx = torch.clamp(((batch_bbox_xh - xl) / bin_size_x).ceil().long(), 0, num_bins_x)
        bin_yh_idx = torch.clamp(((batch_bbox_yh - yl) / bin_size_y).ceil().long(), 0, num_bins_y)
        
        # 处理每个bbox（这部分暂时保持循环，因为每个bbox的bin范围不同）
        for i in range(batch_size_actual):
            bx_start, bx_end = bin_xl_idx[i].item(), bin_xh_idx[i].item()
            by_start, by_end = bin_yl_idx[i].item(), bin_yh_idx[i].item()
            
            if bx_start >= bx_end or by_start >= by_end:
                continue
                
            # 获取影响的bin坐标
            affected_bin_xl = bin_xl_coords[bx_start:bx_end]  # [n_bx]
            affected_bin_yl = bin_yl_coords[by_start:by_end]  # [n_by]
            affected_bin_xh = bin_xh_coords[bx_start:bx_end]  # [n_bx]
            affected_bin_yh = bin_yh_coords[by_start:by_end]  # [n_by]
            
            # 广播计算重叠（向量化）
            bin_xl_grid = affected_bin_xl.unsqueeze(1)  # [n_bx, 1]
            bin_yl_grid = affected_bin_yl.unsqueeze(0)  # [1, n_by]
            bin_xh_grid = affected_bin_xh.unsqueeze(1)  # [n_bx, 1]
            bin_yh_grid = affected_bin_yh.unsqueeze(0)  # [1, n_by]
            
            # 计算重叠区域
            overlap_xl = torch.maximum(batch_bbox_xl[i], bin_xl_grid)
            overlap_yl = torch.maximum(batch_bbox_yl[i], bin_yl_grid)
            overlap_xh = torch.minimum(batch_bbox_xh[i], bin_xh_grid)
            overlap_yh = torch.minimum(batch_bbox_yh[i], bin_yh_grid)
            
            # 重叠面积
            overlap_area = (torch.clamp(overlap_xh - overlap_xl, min=0) * 
                           torch.clamp(overlap_yh - overlap_yl, min=0))
            
            area_ratio = overlap_area / torch.clamp(batch_area[i], min=1e-8)
            bin_demand_contrib = batch_demand[i] *  area_ratio

            routing_demand_map[bx_start:bx_end, by_start:by_end] += bin_demand_contrib
    
    return routing_demand_map


def triangle_density_function(x, node_size, xl, k, bin_size):
    bin_xl = xl + k * bin_size
    right = torch.tensor(bin_xl + bin_size, device=x.device, dtype=x.dtype)
    left = torch.tensor(bin_xl, device=x.device, dtype=x.dtype)
    return torch.minimum(x + node_size, right) - torch.maximum(x, left)

def compute_triangle_density_map_vectorized(
    pos,
    node_size_x_clamped,
    node_size_y_clamped,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y,
    ratio=None
):
    """
    仿照C++ triangle模型，计算density map
    pos: [2*N] tensor, 前N为x，后N为y
    node_size_x_clamped: [N]
    node_size_y_clamped: [N]
    ratio: [N] 可选，默认为1
    返回: [num_bins_x, num_bins_y] density map
    """
    device = pos.device
    dtype = pos.dtype
    num_nodes = pos.numel() // 2

    node_x = pos[:num_nodes]
    node_y = pos[num_nodes:]
    node_w = node_size_x_clamped
    node_h = node_size_y_clamped
    if ratio is None:
        ratio = torch.ones(num_nodes, device=device, dtype=dtype)

    density_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)

    for i in range(num_nodes):
        # 计算影响的bin范围
        bin_index_xl = int((node_x[i] - xl) / bin_size_x)
        bin_index_xh = int(((node_x[i] + node_w[i] - xl) / bin_size_x)) + 1  # exclusive
        bin_index_xl = max(bin_index_xl, 0)
        bin_index_xh = min(bin_index_xh, num_bins_x)

        bin_index_yl = int((node_y[i] - yl) / bin_size_y)
        bin_index_yh = int(((node_y[i] + node_h[i] - yl) / bin_size_y)) + 1  # exclusive
        bin_index_yl = max(bin_index_yl, 0)
        bin_index_yh = min(bin_index_yh, num_bins_y)

        for k in range(bin_index_xl, bin_index_xh):
            px = triangle_density_function(node_x[i], node_w[i], xl, k, bin_size_x)
            px_by_ratio = px * ratio[i]
            for h in range(bin_index_yl, bin_index_yh):
                py = triangle_density_function(node_y[i], node_h[i], yl, h, bin_size_y)
                area = px_by_ratio * py
                density_map[k, h] += area

    return density_map


def compute_bbox_density_map_vectorized(
    pos,
    node_size_x_clamped,
    node_size_y_clamped,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y,
):
    """
    计算所有bbox在die上的density map（面积覆盖），向量化实现
    :param pos: [2*N] tensor, 前N为x，后N为y
    :param node_size_x_clamped: [N] tensor
    :param node_size_y_clamped: [N] tensor
    :param xl, yl, xh, yh: die边界
    :param bin_size_x, bin_size_y: bin尺寸
    :param num_bins_x, num_bins_y: bin数量
    :return: [num_bins_x, num_bins_y] tensor, 每个bin的density
    """
    device = pos.device
    dtype = pos.dtype
    num_bboxes = pos.numel() // 2

    logger.info(f"[BBoxDensityMap] Total input bbox count: {num_bboxes}")

    bbox_xl = pos[:num_bboxes]
    bbox_yl = pos[num_bboxes:]
    bbox_xh = bbox_xl + node_size_x_clamped
    bbox_yh = bbox_yl + node_size_y_clamped

    # bin边界
    bin_xl_coords = torch.arange(num_bins_x, device=device, dtype=dtype) * bin_size_x + xl
    bin_yl_coords = torch.arange(num_bins_y, device=device, dtype=dtype) * bin_size_y + yl
    bin_xh_coords = bin_xl_coords + bin_size_x
    bin_yh_coords = bin_yl_coords + bin_size_y

    density_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)

    processed_bbox_count = 0
    batch_size = 64
    for batch_start in range(0, num_bboxes, batch_size):
        batch_end = min(batch_start + batch_size, num_bboxes)
        processed_bbox_count += (batch_end - batch_start)
        batch_bbox_xl = bbox_xl[batch_start:batch_end]
        batch_bbox_yl = bbox_yl[batch_start:batch_end]
        batch_bbox_xh = bbox_xh[batch_start:batch_end]
        batch_bbox_yh = bbox_yh[batch_start:batch_end]
        batch_area = (batch_bbox_xh - batch_bbox_xl) * (batch_bbox_yh - batch_bbox_yl)

        # 计算每个bbox影响的bin范围
        bin_xl_idx = torch.clamp(((batch_bbox_xl - xl) / bin_size_x).floor().long(), 0, num_bins_x-1)
        bin_yl_idx = torch.clamp(((batch_bbox_yl - yl) / bin_size_y).floor().long(), 0, num_bins_y-1)
        bin_xh_idx = torch.clamp(((batch_bbox_xh - xl) / bin_size_x).ceil().long(), 0, num_bins_x)
        bin_yh_idx = torch.clamp(((batch_bbox_yh - yl) / bin_size_y).ceil().long(), 0, num_bins_y)

        for i in range(batch_end - batch_start):
            bx_start, bx_end = bin_xl_idx[i].item(), bin_xh_idx[i].item()
            by_start, by_end = bin_yl_idx[i].item(), bin_yh_idx[i].item()
            if bx_start >= bx_end or by_start >= by_end:
                continue

            affected_bin_xl = bin_xl_coords[bx_start:bx_end]
            affected_bin_yl = bin_yl_coords[by_start:by_end]
            affected_bin_xh = bin_xh_coords[bx_start:bx_end]
            affected_bin_yh = bin_yh_coords[by_start:by_end]

            bin_xl_grid = affected_bin_xl.unsqueeze(1)  # [n_bx, 1]
            bin_yl_grid = affected_bin_yl.unsqueeze(0)  # [1, n_by]
            bin_xh_grid = affected_bin_xh.unsqueeze(1)  # [n_bx, 1]
            bin_yh_grid = affected_bin_yh.unsqueeze(0)  # [1, n_by]

            overlap_xl = torch.maximum(batch_bbox_xl[i], bin_xl_grid)
            overlap_yl = torch.maximum(batch_bbox_yl[i], bin_yl_grid)
            overlap_xh = torch.minimum(batch_bbox_xh[i], bin_xh_grid)
            overlap_yh = torch.minimum(batch_bbox_yh[i], bin_yh_grid)

            overlap_area = (torch.clamp(overlap_xh - overlap_xl, min=0) *
                            torch.clamp(overlap_yh - overlap_yl, min=0))

            density_map[bx_start:bx_end, by_start:by_end] += overlap_area

    logger.info(f"[BBoxDensityMap] Processed bbox count: {processed_bbox_count}")
    return density_map


def compute_density_map(
    pos,
    node_size_x_clamped,
    node_size_y_clamped,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    num_bins_x, num_bins_y,
):
    device = pos.device
    dtype = pos.dtype
    num_bboxes = pos.numel() // 2

    bbox_xl = pos[:num_bboxes]
    bbox_yl = pos[num_bboxes:]
    bbox_xh = bbox_xl + node_size_x_clamped
    bbox_yh = bbox_yl + node_size_y_clamped

    density_map = torch.zeros(num_bins_x, num_bins_y, dtype=dtype, device=device)

    bin_has_rect = [[[] for _ in range(num_bins_y)] for _ in range(num_bins_x)]

    for i in range(num_bboxes):
        bin_index_xl = int((bbox_xl[i] - xl) / bin_size_x)
        bin_index_xh = int(((bbox_xh[i] - xl) / bin_size_x + 0.5)) 
        bin_index_xl = max(bin_index_xl, 0)
        bin_index_xh = min(bin_index_xh, num_bins_x)

        bin_index_yl = int((bbox_yl[i] - yl) / bin_size_y)
        bin_index_yh = int(((bbox_yh[i] - yl) / bin_size_y + 0.5))
        bin_index_yl = max(bin_index_yl, 0)
        bin_index_yh = min(bin_index_yh, num_bins_y)

        for bx in range(bin_index_xl, bin_index_xh):
            for by in range(bin_index_yl, bin_index_yh):
                ovlp_x = min(bbox_xh[i], xl + (bx + 1) * bin_size_x) - max(bbox_xl[i], xl + bx * bin_size_x)
                ovlp_y = min(bbox_yh[i], yl + (by + 1) * bin_size_y) - max(bbox_yl[i], yl + by * bin_size_y)
                if ovlp_x > 0 and ovlp_y > 0:
                    density_map[bx][by] += ovlp_x * ovlp_y
                    bin_has_rect[bx][by].append(i)

    logging.info(f"num_bin: {num_bins_x} x {num_bins_y}, num_bboxes: {num_bboxes}")

    # print density map values by order (left-top to right-bottom) 
    with open("density_map.txt", "w") as f:
        for j in range(num_bins_y-1, -1, -1):
            for i in range(num_bins_x):
                f.write(f"[{i:2d}][{j:2d}]: {density_map[i][j].item():6.1f} ")
            f.write("\n")

    with open("bin_overlaps.txt", "w") as f:
        for i in range(num_bins_x):
            for j in range(num_bins_y):
                for k in bin_has_rect[i][j]:                
                        f.write(f"Bin ({i},{j}) overlaps with bbox {k}\n")

    with open("bbox_positions.txt", "w") as f:
        for i in range(num_bboxes):
            f.write(f"BBox {i}: xl={bbox_xl[i].item()}, yl={bbox_yl[i].item()}, xh={bbox_xh[i].item()}, yh={bbox_yh[i].item()}\n")

    return density_map


def path_count_density_torch(x, y, W, H):
    eps = 1e-8
    n1 = x + y
    k1 = x
    log_comb_1 = torch.lgamma(n1 + 1 + eps) - torch.lgamma(k1 + 1 + eps) - torch.lgamma(n1 - k1 + 1 + eps)

    n2 = (W - x) + (H - y)
    k2 = W - x
    log_comb_2 = torch.lgamma(n2 + 1 + eps) - torch.lgamma(k2 + 1 + eps) - torch.lgamma(n2 - k2 + 1 + eps)

    n3 = W + H
    k3 = W
    log_comb_3 = torch.lgamma(n3 + 1 + eps) - torch.lgamma(k3 + 1 + eps) - torch.lgamma(n3 - k3 + 1 + eps)

    demand = log_comb_1 + log_comb_2 - log_comb_3
    demand = torch.exp(demand)
    return torch.nan_to_num(demand, nan=-np.inf, posinf=0, neginf=-np.inf)

def compute_probability_demand_map(
    pos, node_size_x, node_size_y, direction_info,
    xl, yl, xh, yh,
    bin_num_x, bin_num_y):
    
    device = pos.device
    dtype = pos.dtype

    num_edges = node_size_x.numel()

    bin_size_x = (xh - xl) / bin_num_x
    bin_size_y = (yh - yl) / bin_num_y

    bin_centers_x = xl + (torch.arange(bin_num_x, device=device, dtype=dtype) + 0.5) * bin_size_x
    bin_centers_y = yl + (torch.arange(bin_num_y, device=device, dtype=dtype) + 0.5) * bin_size_y

    llx = pos[:num_edges]
    lly = pos[num_edges:]
    urx = llx + node_size_x
    ury = lly + node_size_y
    W_real = node_size_x
    H_real = node_size_y


    norm_x_base = (bin_centers_x.unsqueeze(0) - llx.unsqueeze(1)) / torch.clamp(W_real.unsqueeze(1), min=1e-6)
    norm_y_base = (bin_centers_y.unsqueeze(0) - lly.unsqueeze(1)) / torch.clamp(H_real.unsqueeze(1), min=1e-6)

    # direction_info: 0=LL->UR, 1=UL->LR, 2=LR->UL, 3=UR->LL
    flip_x_mask = (direction_info == 2) | (direction_info == 3)
    flip_y_mask = (direction_info == 1) | (direction_info == 3)

    norm_x = torch.where(flip_x_mask.unsqueeze(1), 1.0 - norm_x_base, norm_x_base)
    norm_y = torch.where(flip_y_mask.unsqueeze(1), 1.0 - norm_y_base, norm_y_base)
    
    
    unit_W = torch.tensor(1.0, dtype=dtype, device=device)
    unit_H = torch.tensor(1.0, dtype=dtype, device=device)

    # 扩展维度以进行最终计算
    # norm_x: [num_edges, num_bins_x] -> [num_edges, 1, num_bins_x]
    # norm_y: [num_edges, num_bins_y] -> [num_edges, num_bins_y, 1]
    demand_shape = path_count_density_torch(
        norm_x.unsqueeze(1), norm_y.unsqueeze(2), unit_W, unit_H
    )

    # --- 4. 创建BBox掩码并应用 ---
    # is_in_x: [num_edges, num_bins_x]
    is_in_x = (bin_centers_x.unsqueeze(0) >= llx.unsqueeze(1)) & (bin_centers_x.unsqueeze(0) <= urx.unsqueeze(1))
    is_in_y = (bin_centers_y.unsqueeze(0) >= lly.unsqueeze(1)) & (bin_centers_y.unsqueeze(0) <= ury.unsqueeze(1))
    # mask: [num_edges, num_bins_y, num_bins_x]
    mask = is_in_y.unsqueeze(2) & is_in_x.unsqueeze(1)
    
    demand_shape[~mask] = 0.0
    demand_map = torch.sum(demand_shape, dim=0)

    return demand_map

def compute_probability_demand_map_batched(
    pos, node_size_x, node_size_y, direction_info,
    xl, yl, xh, yh,
    bin_num_x, bin_num_y,
    batch_size=4096 # 可根据GPU显存调整的批次大小
):
    
    device = pos.device
    dtype = pos.dtype
    num_total_edges = node_size_x.numel()
    
    # 最终的demand map
    total_demand_map = torch.zeros(bin_num_y, bin_num_x, dtype=dtype, device=device)

    # --- 将所有输入按edge的维度进行分批 ---
    pos_chunks = torch.split(pos, batch_size)
    node_size_x_chunks = torch.split(node_size_x, batch_size)
    node_size_y_chunks = torch.split(node_size_y, batch_size)
    direction_info_chunks = torch.split(direction_info, batch_size)
    
    # 创建一次bin中心点坐标，在所有批次中复用
    bin_centers_x = xl + (torch.arange(bin_num_x, device=device, dtype=dtype) + 0.5) * ((xh - xl) / bin_num_x)
    bin_centers_y = yl + (torch.arange(bin_num_y, device=device, dtype=dtype) + 0.5) * ((yh - yl) / bin_num_y)

    # --- 循环处理每个批次 ---
    for i in range(len(pos_chunks)):
        # 获取当前批次的数据
        batch_pos = pos_chunks[i]
        batch_node_size_x = node_size_x_chunks[i]
        batch_node_size_y = node_size_y_chunks[i]
        batch_direction_info = direction_info_chunks[i]
        
        num_edges_in_batch = batch_node_size_x.numel()

        # --------------------------------------------------------------------
        #  【核心计算逻辑与您之前的代码完全相同，只是操作对象从 all_edges 变成了 batch_edges】
        # --------------------------------------------------------------------
        llx = batch_pos[:num_edges_in_batch]
        lly = batch_pos[num_edges_in_batch:]
        urx = llx + batch_node_size_x
        ury = lly + batch_node_size_y
        W_real = batch_node_size_x
        H_real = batch_node_size_y
        
        norm_x_base = (bin_centers_x.unsqueeze(0) - llx.unsqueeze(1)) / torch.clamp(W_real.unsqueeze(1), min=1e-6)
        norm_y_base = (bin_centers_y.unsqueeze(0) - lly.unsqueeze(1)) / torch.clamp(H_real.unsqueeze(1), min=1e-6)

        flip_x_mask = (batch_direction_info == 2) | (batch_direction_info == 3)
        flip_y_mask = (batch_direction_info == 1) | (batch_direction_info == 3)

        norm_x = torch.where(flip_x_mask.unsqueeze(1), 1.0 - norm_x_base, norm_x_base)
        norm_y = torch.where(flip_y_mask.unsqueeze(1), 1.0 - norm_y_base, norm_y_base)
        
        unit_W = torch.tensor(1.0, dtype=dtype, device=device)
        unit_H = torch.tensor(1.0, dtype=dtype, device=device)

        demand_shape = path_count_density_torch( # 假设这个函数已定义
            norm_x.unsqueeze(1), norm_y.unsqueeze(2), unit_W, unit_H
        )

        is_in_x = (bin_centers_x.unsqueeze(0) >= llx.unsqueeze(1)) & (bin_centers_x.unsqueeze(0) <= urx.unsqueeze(1))
        is_in_y = (bin_centers_y.unsqueeze(0) >= lly.unsqueeze(1)) & (bin_centers_y.unsqueeze(0) <= ury.unsqueeze(1))
        mask = is_in_y.unsqueeze(2) & is_in_x.unsqueeze(1)
        
        demand_shape[~mask] = 0.0

        total_demand_map += torch.sum(demand_shape, dim=0)

    return total_demand_map

class BBoxElectricPotentialFunction(Function):
    """
    @brief compute electric potential according to e-place.
    """
    @staticmethod
    # @profile_method
    def forward(
        ctx,
        pos,
        bbox_direction,
        node_size_x_clamped,
        node_size_y_clamped,
        offset_x,
        offset_y,
        ratio,
        bin_center_x,
        bin_center_y,
        initial_density_map,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_filler_nodes,
        padding,
        padding_mask,  # same dimensions as density map, with padding regions to be 1
        num_bins_x,
        num_bins_y,
        num_movable_impacted_bins_x,
        num_movable_impacted_bins_y,
        num_filler_impacted_bins_x,
        num_filler_impacted_bins_y,
        deterministic_flag,
        sorted_node_map,
        exact_expkM=None,  # exp(-j*pi*k/M)
        exact_expkN=None,  # exp(-j*pi*k/N)
        inv_wu2_plus_wv2=None,  # 1.0/(wu^2 + wv^2)
        wu_by_wu2_plus_wv2_half=None,  # wu/(wu^2 + wv^2)/2
        wv_by_wu2_plus_wv2_half=None,  # wv/(wu^2 + wv^2)/2
        dct2=None,
        idct2=None,
        idct_idxst=None,
        idxst_idct=None,
        fast_mode=True  # fast mode will discard some computation
    ):

        tt = time.time()

        # density_map = ElectricDensityMapFunction.forward(
        #     pos, node_size_x_clamped, node_size_y_clamped, offset_x, offset_y,
        #     ratio, bin_center_x, bin_center_y, initial_density_map,
        #     target_density, xl, yl, xh, yh, bin_size_x, bin_size_y,
        #     num_movable_nodes, num_filler_nodes, padding, padding_mask,
        #     num_bins_x, num_bins_y, num_movable_impacted_bins_x,
        #     num_movable_impacted_bins_y, num_filler_impacted_bins_x,
        #     num_filler_impacted_bins_y, deterministic_flag, sorted_node_map)

        global plot_count

        density_map = compute_probability_demand_map(
            pos, node_size_x_clamped, node_size_y_clamped, bbox_direction,
            xl, yl, xh, yh,
            num_bins_x, num_bins_y)
        
        # plot_density_map(density_map=density_map.cpu().numpy(), title="Probability Demand Map", 
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/probability_demand_map_{plot_count}.png")
        # exit(0)



        # density_map = compute_gamma_demand_map(pos, node_size_x_clamped, node_size_y_clamped,
        #                                        xl, yl, xh, yh,
        #                                        num_bins_x, num_bins_y)
        
        # plot_density_map(density_map=density_map.cpu().numpy(), title="Gamma Demand Map", 
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/gamma_demand_map_{plot_count}.png")

        # exit(0)

        # plot_bboxes_on_die(pos, node_size_x_clamped, node_size_y_clamped, 
        #                    xl, yl, xh, yh, bin_size_x, bin_size_y, title="BBoxes on Die", 
        #                    save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/bbox_map_{plot_count}.png")
        # plot_density_map(density_map=density_map.cpu().numpy(), title="Density Map", 
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/density_map_{plot_count}.png")
       

        # density_map_py = compute_bbox_density_map_vectorized(
        #     pos,
        #     node_size_x_clamped,
        #     node_size_y_clamped,
        #     xl, yl, xh, yh,
        #     bin_size_x,
        #     bin_size_y,
        #     num_bins_x,
        #     num_bins_y,
        # )

        # plot_density_map(density_map=density_map_py.cpu().numpy(), title="Python Density Map", 
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/python_density_map_{plot_count}.png")

        # triangle_density_map_py = compute_triangle_density_map_vectorized(
        #     pos, node_size_x_clamped, node_size_y_clamped,
        #     xl, yl, xh, yh, bin_size_x, bin_size_y, num_bins_x, num_bins_y
        # )

        # plot_density_map(density_map=triangle_density_map_py.cpu().numpy(), title="Triangle Density Map",
        #                 save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/triangle_density_map_{plot_count}.png")

        # density_map_rudy = compute_bbox_rudy_density_map_vectorized(
        #     pos,
        #     node_size_x_clamped,
        #     node_size_y_clamped,
        #     xl, yl, xh, yh,
        #     bin_size_x,
        #     bin_size_y,
        #     num_bins_x,
        #     num_bins_y,
        # )      

        # plot_density_map(density_map=density_map_rudy.cpu().numpy(), title="RUDY Density Map", 
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/rudy_density_map_{plot_count}.png")

        # plot_density_with_bboxes(density_map=density_map_py.cpu().numpy(), pos=pos, node_size_x=node_size_x_clamped, node_size_y=node_size_y_clamped,
        #                         xl=xl, yl=yl, xh=xh, yh=yh, bin_size_x=bin_size_x, bin_size_y=bin_size_y,
        #                         title="Density Map with BBoxes",
        #                         save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/density_with_bboxes_{plot_count}.png")

        # density_map_py_2 = compute_density_map(
        #     pos,
        #     node_size_x_clamped,
        #     node_size_y_clamped,
        #     xl, yl, xh, yh,
        #     bin_size_x, bin_size_y,
        #     num_bins_x, num_bins_y)
        
        # plot_density_map(density_map=density_map_py_2.cpu().numpy(), title="Density Map V2",
        #                  save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/density_map_v2_{plot_count}.png")

        

        # print("=== Debugging density maps ===")
        # print(f"C++ density map sum: {density_map.sum():.6f}")
        # print(f"Python density map sum: {density_map_py.sum():.6f}")
        # print(f"Maps equal: {torch.allclose(density_map, density_map_py, rtol=1e-5)}")

        # if not torch.allclose(density_map, density_map_py, rtol=1e-5):
        #     diff = (density_map - density_map_py).abs()
        #     print(f"Max absolute difference: {diff.max():.6f}")
        #     print(f"Mean absolute difference: {diff.mean():.6f}")
            
        #     max_diff_idx = torch.argmax(diff)
        #     max_diff_coords = torch.unravel_index(max_diff_idx, diff.shape)
        #     print(f"Max diff at bin ({max_diff_coords[0]}, {max_diff_coords[1]}): "
        #         f"C++={density_map[max_diff_coords]:.6f}, "
        #         f"Python={density_map_py[max_diff_coords]:.6f}")
            
        #     exit(0)

        # output consists of (density_cost, density_map, max_density)
        ctx.node_size_x_clamped = node_size_x_clamped
        ctx.node_size_y_clamped = node_size_y_clamped
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
        ctx.num_movable_nodes = num_movable_nodes
        ctx.num_filler_nodes = num_filler_nodes
        ctx.padding = padding
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_movable_impacted_bins_x = num_movable_impacted_bins_x
        ctx.num_movable_impacted_bins_y = num_movable_impacted_bins_y
        ctx.num_filler_impacted_bins_x = num_filler_impacted_bins_x
        ctx.num_filler_impacted_bins_y = num_filler_impacted_bins_y
        ctx.deterministic_flag = deterministic_flag
        ctx.pos = pos
        ctx.sorted_node_map = sorted_node_map
        #density_map = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_x = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_y = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        # return torch.zeros(1, dtype=pos.dtype, device=pos.device)

        # for DCT
        M = num_bins_x
        N = num_bins_y

        # wu and wv
        if inv_wu2_plus_wv2 is None:
            wu = torch.arange(M,
                              dtype=density_map.dtype,
                              device=density_map.device).mul(2 * np.pi /
                                                             M).view([M, 1])
            wv = torch.arange(N,
                              dtype=density_map.dtype,
                              device=density_map.device).mul(2 * np.pi /
                                                             N).view([1, N])
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0,
                         0] = 1.0  # avoid zero-division, it will be zeroed out
            inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
            inv_wu2_plus_wv2[0, 0] = 0.0
            wu_by_wu2_plus_wv2_half = wu.mul(inv_wu2_plus_wv2).mul_(1. / 2)
            wv_by_wu2_plus_wv2_half = wv.mul(inv_wu2_plus_wv2).mul_(1. / 2)


        # density_map = (density_map - ctx.target_density * (ctx.bin_size_x * ctx.bin_size_y)).clamp(min=0)
        

        # plot_density_map(density_map=density_map.cpu().numpy(), title="Overflow Density Map",
        #                 save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/overflow_density_map_{plot_count}.png")

        # compute auv
        density_map.mul_(1.0 / (ctx.bin_size_x * ctx.bin_size_y))

        # plot_density_map(density_map=density_map.cpu().numpy(), title="Normalized Density Map",
        #                 save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/normalized_density_map_{plot_count}.png")

        
        # exit(0) 

        # density_map = (density_map - ctx.target_density * (ctx.bin_size_x * ctx.bin_size_y)).clamp(min=0)

        #auv = discrete_spectral_transform.dct2_2N(density_map, expk0=exact_expkM, expk1=exact_expkN)
        auv = dct2.forward(density_map)

        # compute field xi
        auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_half)
        auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_half)

        #ctx.field_map_x = discrete_spectral_transform.idsct2(auv_by_wu2_plus_wv2_wu, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_x = idxst_idct.forward(auv_by_wu2_plus_wv2_wu)
        #ctx.field_map_y = discrete_spectral_transform.idcst2(auv_by_wu2_plus_wv2_wv, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_y = idct_idxst.forward(auv_by_wu2_plus_wv2_wv)


        # compute potential phi for size gradient
        # auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2)
        # potential_map = idct2.forward(auv_by_wu2_plus_wv2)
        
        # ctx.potential_map = potential_map

        # energy = \sum q*phi
        # it takes around 80% of the computation time
        # so I will not always evaluate it
        if fast_mode:  # dummy for invoking backward propagation
            energy = torch.zeros(1, dtype=pos.dtype, device=pos.device)
        else:
            # compute potential phi
            # auv / (wu**2 + wv**2)
            auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2)
            #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, exact_expkM, exact_expkN)
            potential_map = idct2.forward(auv_by_wu2_plus_wv2)

            # compute potential phi for size gradient
            ctx.potential_map = potential_map

            # compute energy
            energy = potential_map.mul(density_map).sum()

        
        # plot_density_map(density_map=potential_map.cpu().numpy(), title="Potential Map",
        #                   save_path=f"/home/sxr/workspace/benchmark-rp/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/potential_map_{plot_count}.png")

        # exit(0)

        # torch.set_printoptions(precision=10)
        # logger.debug("initial_density_map")
        # logger.debug(initial_density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("density_map")
        # logger.debug(density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("auv_by_wu2_plus_wv2")
        # logger.debug(auv_by_wu2_plus_wv2)
        # logger.debug("potential_map")
        # logger.debug(potential_map)
        # logger.debug("field_map_x")
        # logger.debug(ctx.field_map_x)
        # logger.debug("field_map_y")
        # logger.debug(ctx.field_map_y)

        # global plot_count
        # if plot_count >= 600 and plot_count % 1 == 0:
        #    logger.debug("density_map")
        #    plot(plot_count, density_map.clone().div(bin_size_x*bin_size_y).cpu().numpy(), padding, "summary/%d.density_map" % (plot_count))
        #    logger.debug("potential_map")
        #    plot(plot_count, potential_map.clone().cpu().numpy(), padding, "summary/%d.potential_map" % (plot_count))
        #    logger.debug("field_map_x")
        #    plot(plot_count, ctx.field_map_x.clone().cpu().numpy(), padding, "summary/%d.field_map_x" % (plot_count))
        #    logger.debug("field_map_y")
        #    plot(plot_count, ctx.field_map_y.clone().cpu().numpy(), padding, "summary/%d.field_map_y" % (plot_count))
        # plot_count += 1

        plot_count += 1

        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("bbox density forward %.3f ms" % ((time.time() - tt) * 1000))
        return energy
    
    # @profile_method
    def compute_size_gradients_vectorized(ctx, grad_pos):
        tt = time.time()
        
        num_nodes = ctx.pos.numel() // 2
        device = ctx.pos.device
        
        # 预计算所有bin的边界
        bin_xl_grid = torch.arange(ctx.num_bins_x, device=device, dtype=ctx.pos.dtype) * ctx.bin_size_x + ctx.xl
        bin_yl_grid = torch.arange(ctx.num_bins_y, device=device, dtype=ctx.pos.dtype) * ctx.bin_size_y + ctx.yl
        bin_xh_grid = bin_xl_grid + ctx.bin_size_x
        bin_yh_grid = bin_yl_grid + ctx.bin_size_y
        
        # 节点信息向量化
        node_xl = ctx.pos[:num_nodes]  # [num_nodes]
        node_yl = ctx.pos[num_nodes:]  # [num_nodes]
        node_size_x = ctx.node_size_x_clamped  # [num_nodes]
        node_size_y = ctx.node_size_y_clamped  # [num_nodes]
        node_xh = node_xl + node_size_x
        node_yh = node_yl + node_size_y
        
        # 计算每个节点影响的bin范围
        bin_index_xl = ((node_xl - ctx.xl) / ctx.bin_size_x).floor().clamp(0, ctx.num_bins_x-1).long()
        bin_index_yl = ((node_yl - ctx.yl) / ctx.bin_size_y).floor().clamp(0, ctx.num_bins_y-1).long()  
        bin_index_xh = ((node_xh - ctx.xl) / ctx.bin_size_x).ceil().clamp(0, ctx.num_bins_x).long()
        bin_index_yh = ((node_yh - ctx.yl) / ctx.bin_size_y).ceil().clamp(0, ctx.num_bins_y).long()
        
        size_x_grad = torch.zeros(num_nodes, device=device, dtype=ctx.pos.dtype)
        size_y_grad = torch.zeros(num_nodes, device=device, dtype=ctx.pos.dtype)
        
        # 向量化计算overlap和梯度
        for i in range(num_nodes):
            if bin_index_xl[i] >= bin_index_xh[i] or bin_index_yl[i] >= bin_index_yh[i]:
                continue
                
            # 获取影响的bin范围
            bx_start, bx_end = bin_index_xl[i], bin_index_xh[i] 
            by_start, by_end = bin_index_yl[i], bin_index_yh[i]
            
            # 创建bin网格
            bin_x_range = torch.arange(bx_start, bx_end, device=device)
            bin_y_range = torch.arange(by_start, by_end, device=device)
            
            # 广播计算overlap
            bin_xl = bin_xl_grid[bin_x_range].unsqueeze(1)  # [n_bx, 1]
            bin_yl = bin_yl_grid[bin_y_range].unsqueeze(0)  # [1, n_by]
            bin_xh = bin_xh_grid[bin_x_range].unsqueeze(1)  # [n_bx, 1] 
            bin_yh = bin_yh_grid[bin_y_range].unsqueeze(0)  # [1, n_by]
            
            # 计算overlap - 向量化
            ovlp_xl = torch.maximum(node_xl[i], bin_xl)
            ovlp_yl = torch.maximum(node_yl[i], bin_yl) 
            ovlp_xh = torch.minimum(node_xh[i], bin_xh)
            ovlp_yh = torch.minimum(node_yh[i], bin_yh)
            
            ovlp_area = (ovlp_xh - ovlp_xl).clamp(min=0) * (ovlp_yh - ovlp_yl).clamp(min=0)
            
            # 提取相应的potential值
            potential_slice = ctx.potential_map[bx_start:bx_end, by_start:by_end]
            
            # 向量化梯度计算
            grad_contrib_x = (potential_slice * ovlp_area * node_size_y[i]).sum()
            grad_contrib_y = (potential_slice * ovlp_area * node_size_x[i]).sum()
            
            size_x_grad[i] = grad_contrib_x
            size_y_grad[i] = grad_contrib_y
        
        # 应用缩放因子
        combined_factor = ctx.ratio * grad_pos
        size_x_grad.mul_(combined_factor)
        size_y_grad.mul_(combined_factor)
        
        logger.debug("bbox density size grad (vectorized) %.3f ms" % ((time.time() - tt) * 1000))
        return size_x_grad, size_y_grad

    def compute_size_gradients(ctx, grad_pos):
        # compute size gradients
        tt = time.time()

        num_nodes = ctx.pos.numel() // 2

        size_x_grad = torch.zeros_like(ctx.pos[:num_nodes])
        size_y_grad = torch.zeros_like(ctx.pos[:num_nodes])
        
        for i in range(num_nodes):
            node_xl = ctx.pos[i]
            node_yl = ctx.pos[i + num_nodes]
            node_size_x = ctx.node_size_x_clamped[i]
            node_size_y = ctx.node_size_y_clamped[i]
            bin_index_xl = int((node_xl - ctx.xl) / ctx.bin_size_x)
            bin_index_yl = int((node_yl - ctx.yl) / ctx.bin_size_y)
            bin_index_xh = int((node_xl + node_size_x) / ctx.bin_size_x) + 1
            bin_index_yh = int((node_yl + node_size_y) / ctx.bin_size_y) + 1
            bin_index_xl = max(bin_index_xl, 0)
            bin_index_yl = max(bin_index_yl, 0)
            bin_index_xh = min(bin_index_xh, ctx.num_bins_x)    
            bin_index_yh = min(bin_index_yh, ctx.num_bins_y)
            if bin_index_xl >= bin_index_xh or bin_index_yl >= bin_index_yh:
                continue
            
            size_x_grad_i = 0.0
            size_y_grad_i = 0.0
            for bin_x in range(bin_index_xl, bin_index_xh):
                for bin_y in range(bin_index_yl, bin_index_yh):
                    # compute overlap
                    bin_xl = ctx.xl + bin_x * ctx.bin_size_x
                    bin_yl = ctx.yl + bin_y * ctx.bin_size_y
                    bin_xh = bin_xl + ctx.bin_size_x
                    bin_yh = bin_yl + ctx.bin_size_y
                    ovlp_xl = max(node_xl, bin_xl)
                    ovlp_yl = max(node_yl, bin_yl)
                    ovlp_xh = min(node_xl + node_size_x, bin_xh)
                    ovlp_yh = min(node_yl + node_size_y, bin_yh)
                    ovlp = max(ovlp_xh - ovlp_xl, 0.0) * max(ovlp_yh - ovlp_yl, 0.0)
                    if ovlp <= 0.0:
                        continue

                    # compute gradient
                    size_x_grad_i += ctx.potential_map[bin_x, bin_y] * ovlp * node_size_y
                    size_y_grad_i += ctx.potential_map[bin_x, bin_y] * ovlp * node_size_x
            
            size_x_grad[i] = size_x_grad_i
            size_y_grad[i] = size_y_grad_i

        logger.debug("bbox density size grad %.3f ms" % ((time.time() - tt) * 1000))

        size_x_grad.mul_(ctx.ratio)
        size_y_grad.mul_(ctx.ratio)

        size_x_grad.mul_(grad_pos)
        size_y_grad.mul_(grad_pos)

        return size_x_grad, size_y_grad

    @staticmethod
    # @time_method
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            output = -electric_potential_cuda.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x,
                ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view(
                    [-1]), ctx.pos, ctx.node_size_x_clamped,
                ctx.node_size_y_clamped, ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh,
                ctx.yh, ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                ctx.num_filler_nodes, ctx.deterministic_flag, ctx.sorted_node_map)
        else:
            output = -electric_potential_cpp.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x,
                ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view(
                    [-1]), ctx.pos, ctx.node_size_x_clamped,
                ctx.node_size_y_clamped, ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh,
                ctx.yh, ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                ctx.num_filler_nodes)


        num_nodes = ctx.pos.numel() // 2
        area = ctx.node_size_x_clamped * ctx.node_size_y_clamped  # [num_nodes]
        area = area.clamp(min=1.0) 

        output[:num_nodes] = output[:num_nodes] / area
        output[num_nodes:] = output[num_nodes:] / area
            
        # size_x_grad, size_y_grad = BBoxElectricPotentialFunction.compute_size_gradients_vectorized(ctx, grad_pos)


        #global plot_count
        # if plot_count >= 300:
        #    indices = (ctx.pos[ctx.pos.numel()/2-ctx.num_filler_nodes:ctx.pos.numel()/2] < ctx.xl+ctx.bin_size_x).nonzero()
        #    pdb.set_trace()

        #pgradx = []
        #pgrady = []
        # with open("/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/gradient.csv", "r") as f:
        #    for line in f:
        #        tokens = line.strip().split(" ")
        #        pgradx.append(float(tokens[3].strip()))
        #        pgrady.append(float(tokens[4].strip()))
        #pgrad = np.concatenate([np.array(pgradx), np.array(pgrady)])

        #output = torch.empty_like(ctx.pos).uniform_(0.0, 0.1)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("bbox density backward %.3f ms" % ((time.time() - tt) * 1000))
        return output, \
            None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None


class BBoxElectricPotential(ElectricOverflow):
    """
    @brief Compute electric potential according to e-place
    """
    def __init__(
        self,
        node_size_x,
        node_size_y,
        bin_center_x,
        bin_center_y,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_terminals,
        num_filler_nodes,
        padding,
        deterministic_flag,  # control whether to use deterministic routine
        sorted_node_map,
        movable_macro_mask=None,
        fast_mode=False,
        region_id=None,
        fence_regions=None, # [n_subregion, 4] as dummy macros added to initial density. (xl,yl,xh,yh) rectangles
        node2fence_region_map=None,
        placedb=None
        ):
        """
        @brief initialization
        Be aware that all scalars must be python type instead of tensors.
        Otherwise, GPU version can be weirdly slow.
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order
        @param movable_macro_mask some large movable macros need to be scaled to avoid halos
        @param bin_center_x bin center x locations
        @param bin_center_y bin center y locations
        @param target_density target density
        @param xl left boundary
        @param yl bottom boundary
        @param xh right boundary
        @param yh top boundary
        @param bin_size_x bin width
        @param bin_size_y bin height
        @param num_movable_nodes number of movable cells
        @param num_terminals number of fixed cells
        @param num_filler_nodes number of filler cells
        @param padding bin padding to boundary of placement region
        @param deterministic_flag control whether to use deterministic routine
        @param fast_mode if true, only gradient is computed, while objective computation is skipped
        @param region_id id for fence region, from 0 to N if there are N fence regions
        @param fence_regions # [n_subregion, 4] as dummy macros added to initial density. (xl,yl,xh,yh) rectangles
        @param node2fence_region_map node to region id map, non fence region is set to INT_MAX
        @param placedb
        """

        if(region_id is not None):
            ### reconstruct data structure
            num_nodes = placedb.num_nodes
            if(region_id < len(placedb.regions)):
                self.fence_region_mask = node2fence_region_map[:num_movable_nodes] == region_id
            else:
                self.fence_region_mask = node2fence_region_map[:num_movable_nodes] >= len(placedb.regions)

            node_size_x = torch.cat([node_size_x[:num_movable_nodes][self.fence_region_mask],
                                    node_size_x[num_movable_nodes:num_nodes-num_filler_nodes],
                                    node_size_x[num_nodes-num_filler_nodes+placedb.filler_start_map[region_id]:num_nodes-num_filler_nodes+placedb.filler_start_map[region_id+1]]], 0)
            node_size_y = torch.cat([node_size_y[:num_movable_nodes][self.fence_region_mask],
                                    node_size_y[num_movable_nodes:num_nodes-num_filler_nodes],
                                    node_size_y[num_nodes-num_filler_nodes+placedb.filler_start_map[region_id]:num_nodes-num_filler_nodes+placedb.filler_start_map[region_id+1]]], 0)

            num_movable_nodes = (self.fence_region_mask).long().sum().item()
            num_filler_nodes = placedb.filler_start_map[region_id+1]-placedb.filler_start_map[region_id]
            if(movable_macro_mask is not None):
                movable_macro_mask = movable_macro_mask[self.fence_region_mask]
            ## sorted cell is recomputed
            sorted_node_map = torch.sort(node_size_x[:num_movable_nodes])[1].to(torch.int32)
            ## make pos mask for fast forward
            self.pos_mask = torch.zeros(2, placedb.num_nodes, dtype=torch.bool, device=node_size_x.device)
            self.pos_mask[0,:placedb.num_movable_nodes].masked_fill_(self.fence_region_mask, 1)
            self.pos_mask[1,:placedb.num_movable_nodes].masked_fill_(self.fence_region_mask, 1)
            self.pos_mask[:,placedb.num_movable_nodes:placedb.num_nodes-placedb.num_filler_nodes] = 1
            self.pos_mask[:,placedb.num_nodes-placedb.num_filler_nodes+placedb.filler_start_map[region_id]:placedb.num_nodes-placedb.num_filler_nodes+placedb.filler_start_map[region_id+1]] = 1
            self.pos_mask = self.pos_mask.view(-1)

        super(BBoxElectricPotential,
              self).__init__(node_size_x=node_size_x,
                             node_size_y=node_size_y,
                             bin_center_x=bin_center_x,
                             bin_center_y=bin_center_y,
                             target_density=target_density,
                             xl=xl,
                             yl=yl,
                             xh=xh,
                             yh=yh,
                             bin_size_x=bin_size_x,
                             bin_size_y=bin_size_y,
                             num_movable_nodes=num_movable_nodes,
                             num_terminals=num_terminals,
                             num_filler_nodes=num_filler_nodes,
                             padding=padding,
                             deterministic_flag=deterministic_flag,
                             sorted_node_map=sorted_node_map,
                             movable_macro_mask=movable_macro_mask)
        self.fast_mode = fast_mode
        self.fence_regions = fence_regions
        self.node2fence_region_map = node2fence_region_map
        self.placedb = placedb
        self.target_density = target_density
        self.region_id = region_id
        ## set by build_density_op func
        self.filler_start_map = None
        self.filler_beg = None
        self.filler_end = None


    def compute_fence_region_map(self, fence_region, macro_pos_x=None, macro_pos_y=None, macro_size_x=None, macro_size_y=None):
        if(macro_pos_x is not None):
            pos = torch.cat([fence_region[:,0], macro_pos_x, fence_region[:,1], macro_pos_y], 0)
            num_terminals = fence_region.size(0) + macro_size_x.size(0)
            node_size_x = torch.cat([fence_region[:,2] - fence_region[:,0], macro_size_x], 0)
            node_size_y = torch.cat([fence_region[:,3] - fence_region[:,1], macro_size_y], 0)
        else:
            pos = fence_region[:,:2].t().contiguous().view(-1)
            num_terminals = fence_region.size(0)
            node_size_x = fence_region[:,2] - fence_region[:,0]
            node_size_y = fence_region[:,3] - fence_region[:,1]
        max_size_x = node_size_x.max()
        max_size_y = node_size_y.max()
        num_fixed_impacted_bins_x = ((max_size_x + self.bin_size_x) /
                                        self.bin_size_x).ceil().clamp(
                                            max=self.num_bins_x)
        num_fixed_impacted_bins_y = ((max_size_y + self.bin_size_y) /
                                        self.bin_size_y).ceil().clamp(
                                            max=self.num_bins_y)

        if pos.is_cuda:
            func = electric_potential_cuda.fixed_density_map
        else:
            func = electric_potential_cpp.fixed_density_map

        fence_region_map = func(
            pos, node_size_x, node_size_y, self.bin_center_x,
            self.bin_center_y, self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y, 0,
            num_terminals, self.num_bins_x, self.num_bins_y,
            num_fixed_impacted_bins_x, num_fixed_impacted_bins_y,
            self.deterministic_flag)

        fence_region_map.mul_(self.target_density)
        self.fence_region_map = fence_region_map
        return fence_region_map

    def reset(self):
        """ Compute members derived from input
        """
        super(BBoxElectricPotential, self).reset()
        logger.info("regard %d cells as movable macros in global placement" %
                    (self.num_movable_macros))

        self.exact_expkM = None
        self.exact_expkN = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None

        # dct2, idct2, idct_idxst, idxst_idct functions
        self.dct2 = None
        self.idct2 = None
        self.idct_idxst = None
        self.idxst_idct = None

    def forward(self, pos, bbox_direction, mode="density"):
        assert mode in {"density", "overflow"}, "Only support density mode or overflow mode"
        # if(self.region_id is not None):
        #     ### reconstruct pos, only extract cells in this electric field
        #     pos = pos[self.pos_mask]

        # if self.initial_density_map is None:
        #     num_nodes = pos.size(0)//2
        #     if(self.fence_regions is not None):
        #         if(self.placedb.num_terminals > 0):
        #             ### merge fence region density and macro density together as initial density map
        #             ### pay attention to the number of nodes, must use data from self
        #             ### here pos is reconstructed pos !
        #             self.initial_density_map = self.compute_fence_region_map(
        #                 self.fence_regions,
        #                 pos[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals],
        #                 pos[num_nodes+self.num_movable_nodes:num_nodes+self.num_movable_nodes+self.num_terminals],
        #                 self.node_size_x[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals],
        #                 self.node_size_y[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals]
        #                 )
        #         else:
        #             self.initial_density_map = self.compute_fence_region_map(self.fence_regions)
        #     else:
        #         self.compute_initial_density_map(pos)
        #     ## sync the initial density map with
        #     # self.compute_initial_density_map(pos)
        #     # plot(0, self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'summary/initial_potential_map')
        #     logger.info("fixed density map: average %g, max %g, bin area %g" %
        #                 (self.initial_density_map.mean(),
        #                  self.initial_density_map.max(),
        #                  self.bin_size_x * self.bin_size_y))

        # TODO : iniitial density map

            # expk
        M = self.num_bins_x
        N = self.num_bins_y
        self.exact_expkM = precompute_expk(M,
                                            dtype=pos.dtype,
                                            device=pos.device)
        self.exact_expkN = precompute_expk(N,
                                            dtype=pos.dtype,
                                            device=pos.device)

        # init dct2, idct2, idct_idxst, idxst_idct with expkM and expkN
        self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
        if not self.fast_mode:
            self.idct2 = dct.IDCT2(self.exact_expkM, self.exact_expkN)
        self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM,
                                            self.exact_expkN)
        self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM,
                                            self.exact_expkN)

        # wu and wv
        wu = torch.arange(M, dtype=pos.dtype, device=pos.device).mul(
            2 * np.pi / M).view([M, 1])
        # scale wv because the aspect ratio of a bin may not be 1
        wv = torch.arange(N, dtype=pos.dtype,
                            device=pos.device).mul(2 * np.pi / N).view(
                                [1,
                                N]).mul_(self.bin_size_x / self.bin_size_y)
        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0,
                        0] = 1.0  # avoid zero-division, it will be zeroed out
        self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        self.inv_wu2_plus_wv2[0, 0] = 0.0
        self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(
            1. / 2)
        self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(
            1. / 2)

        if(mode == "density"):
            return BBoxElectricPotentialFunction.apply(
                pos, bbox_direction, self.node_size_x_clamped, self.node_size_y_clamped,
                self.offset_x, self.offset_y, self.ratio, self.bin_center_x,
                self.bin_center_y, self.initial_density_map, self.target_density,
                self.xl, self.yl, self.xh, self.yh, self.bin_size_x,
                self.bin_size_y, self.num_movable_nodes, self.num_filler_nodes,
                self.padding, self.padding_mask, self.num_bins_x, self.num_bins_y,
                self.num_movable_impacted_bins_x, self.num_movable_impacted_bins_y,
                self.num_filler_impacted_bins_x, self.num_filler_impacted_bins_y,
                self.deterministic_flag, self.sorted_node_map, self.exact_expkM,
                self.exact_expkN, self.inv_wu2_plus_wv2,
                self.wu_by_wu2_plus_wv2_half, self.wv_by_wu2_plus_wv2_half,
                self.dct2, self.idct2, self.idct_idxst, self.idxst_idct,
                self.fast_mode)
        elif(mode == "overflow"):
            ### num_filler_nodes is set 0
            density_map = ElectricDensityMapFunction.forward(
                pos, self.node_size_x_clamped, self.node_size_y_clamped,
                self.offset_x, self.offset_y, self.ratio, self.bin_center_x,
                self.bin_center_y, self.initial_density_map, self.target_density,
                self.xl, self.yl, self.xh, self.yh, self.bin_size_x,
                self.bin_size_y, self.num_movable_nodes, 0,
                self.padding, self.padding_mask, self.num_bins_x, self.num_bins_y,
                self.num_movable_impacted_bins_x, self.num_movable_impacted_bins_y,
                self.num_filler_impacted_bins_x, self.num_filler_impacted_bins_y,
                self.deterministic_flag, self.sorted_node_map)

            bin_area = self.bin_size_x * self.bin_size_y
            density_cost = (density_map -
                            self.target_density * bin_area).clamp_(min=0.0).sum()

            return density_cost, density_map.max() / bin_area

