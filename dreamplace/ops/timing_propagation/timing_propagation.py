#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : timing_propagation.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        :
@version      : 0.1
@date         : 2025-04-18 20:07:53
@copyright    : Copyright (c) 2021-2022 PCNL EDA.
'''

import time
import torch
from torch import nn
from torch.autograd import Function
import logging
from torch.func import vmap
import unittest
from dataclasses import dataclass, field
# torch._dynamo.config.suppress_errors = True
# Mock Timing Arc representation (can be a simple dict or class)


'''
arc_idx -> lut_idx
len(flat_luts_values_start) == len(flat_arcs)
'''

'''
    flat_luts_values: torch.Tensor      # [N, MaxT, MaxC]
    flat_luts_trans_table : torch.Tensor  # [N, MaxT]
    flat_luts_cap_table : torch.Tensor   # [N, MaxC]
    flat_luts_dim: torch.Tensor        # [N, 2] - Actual dims [trans_dim, cap_dim]
'''


class LUTS_INFO:
    """
    一个数据类，用于持有和预计算一个批次的 LUTs 所需的信息。
    此类封装了为特定批次创建掩码的逻辑。
    """
    def __init__(self,
                 flat_luts_values: torch.Tensor,
                 flat_luts_trans_table: torch.Tensor,
                 flat_luts_cap_table: torch.Tensor,
                 flat_luts_dim: torch.Tensor):
        
        # --- 1. 存储批次数据 ---
        self.flat_luts_values = flat_luts_values
        self.flat_luts_trans_table = flat_luts_trans_table
        self.flat_luts_cap_table = flat_luts_cap_table
        self.flat_luts_dim = flat_luts_dim
        
        # --- 2. 计算实际维度 ---
        self.trans_dims_actual = flat_luts_dim[:, 0].long()
        self.cap_dims_actual = flat_luts_dim[:, 1].long()

        # --- 3. 预计算不同 LUT 类型的布尔掩码 ---
        self.valid_arc_mask = (self.trans_dims_actual > 0) & (self.cap_dims_actual > 0)
        self.is_scalar = (self.trans_dims_actual <= 1) & (self.cap_dims_actual <= 1)
        self.is_trans_1d = (self.trans_dims_actual > 1) & (self.cap_dims_actual <= 1)
        self.is_cap_1d = (self.trans_dims_actual <= 1) & (self.cap_dims_actual > 1)
        self.is_2d = (self.trans_dims_actual > 1) & (self.cap_dims_actual > 1)

'''
each arc has only one f_delay_lut/r_delay_lut

need arc_idx to calc something
'''


@dataclass
class ARCS_INFO:
    # Use default_factory to ensure a new LUTS_INFO instance is created for each ARCS_INFO instance
    f_delay_luts: LUTS_INFO = field(default_factory=LUTS_INFO)
    r_delay_luts: LUTS_INFO = field(default_factory=LUTS_INFO)
    f_trans_luts: LUTS_INFO = field(default_factory=LUTS_INFO)
    r_trans_luts: LUTS_INFO = field(default_factory=LUTS_INFO)


'''
flat_luts_values
flat_luts_values_start
flat_luts_trans
flat_luts_trans_start
flat_luts_cap
flat_luts_cap_start

flat_luts_dim

'''

'''
inst_flat_arcs: [inpin, outpin, lib_cell_idx, lib_cell_arc_idx, timing_sense]
arc_type: 0 for neg, 1 for postive

net_flat_arcs: [inpin, outpin]

'''


class TimingPropagation(nn.Module):

    def __init__(self,
                 inrdelays,
                 infdelays,
                 inrtrans,
                 inftrans,
                 outcaps,
                 pin_net,
                 start_points,
                 end_points,
                 clock_pins,
                 FF_ids,
                 clk_pin_rtran,
                 clk_pin_ftran,
                 net_flat_arcs_start,
                 net_flat_arcs,
                 net2driver_pin_map,
                 arcs_info: ARCS_INFO,
                 inst_flat_arcs_start,
                 inst_flat_arcs,
                 endpoints_constraint_arcs,
                 flat_cells_by_level,
                 flat_cells_by_level_start,
                 flat_cells_by_reverse_level,
                 flat_cells_by_reverse_level_start,
                 endpoints_rRAT,
                 endpoints_fRAT):
        super(TimingPropagation, self).__init__()

        self.num_pins = pin_net.shape[0]
        self.inrdelays = inrdelays
        self.infdelays = infdelays
        self.inrtrans = inrtrans
        self.inftrans = inftrans
        self.outcaps = outcaps
        self.pin_net = pin_net
        self.start_points = start_points
        self.end_points = end_points
        self.clock_pins = clock_pins
        self.FF_ids = FF_ids
        self.clk_pin_rtran = clk_pin_rtran
        self.clk_pin_ftran = clk_pin_ftran
        self.net_flat_arcs_start = net_flat_arcs_start
        self.net_flat_arcs = net_flat_arcs
        self.net2driver_pin_map = net2driver_pin_map
        self.inst_flat_arcs_start = inst_flat_arcs_start
        self.inst_flat_arcs = inst_flat_arcs
        self.endpoints_constraint_arcs = endpoints_constraint_arcs
        self.flat_cells_by_level = flat_cells_by_level
        self.flat_cells_by_level_start = flat_cells_by_level_start
        self.flat_cells_by_reverse_level = flat_cells_by_reverse_level
        self.flat_cells_by_reverse_level_start = flat_cells_by_reverse_level_start
        self.arcs_info = arcs_info
        self.device = inrdelays.device
        self.dtype = inrdelays.dtype  # Use dtype from inputs

        self.endpoints_rRAT = endpoints_rRAT
        self.endpoints_fRAT = endpoints_fRAT

        self.pin_rAAT = None
        self.pin_fAAT = None
        self.pin_rRAT = None
        self.pin_fRAT = None
        self.pin_rtran = None
        self.pin_ftran = None
        self.pin_net_cap = None
        self.cell_arc_r_delays = None
        self.cell_arc_f_delays = None
        
    # @staticmethod
    def _lut_entry_1d_vectorized(self, x,           # [B] - Input values
                                 x_table,     # [B, MaxDim] - Table axes
                                 y_table,     # [B, MaxDim] - Table values
                                 actual_dims  # [B] - Actual table dimensions
                                ):
        """
        Performs vectorized 1D linear interpolation for a full batch.
        (Handles one dimension, e.g., transition or capacitance).
        """
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        # Find neighboring indices using searchsorted on the padded table
        x_b = x.unsqueeze(1)  # [B, 1]
        idx_padded = torch.searchsorted(x_table, x_b, right=True).squeeze(1) # [B]

        # Clamp indices to the actual valid range for each item
        max_idx_actual = (actual_dims - 1).clamp(min=0) # [B]
        idx_high = idx_padded.clamp(min=1).clamp(max=max_idx_actual)
        idx_low = (idx_high - 1).clamp(min=0)

        # Gather boundary points (x0, x1, y0, y1)
        batch_indices = torch.arange(batch_size, device=device)
        x0 = x_table[batch_indices, idx_low]
        x1 = x_table[batch_indices, idx_high]
        y0 = y_table[batch_indices, idx_low]
        y1 = y_table[batch_indices, idx_high]

        # Perform linear interpolation
        interval = x1 - x0
        denom_epsilon = torch.tensor(1e-12, device=device, dtype=dtype)
        is_degenerate = torch.abs(interval) < denom_epsilon

        x_clamped = x.clamp(min=x0, max=x1)
        safe_interval = torch.where(is_degenerate, denom_epsilon, interval)
        factor = ((x_clamped - x0) / safe_interval).clamp(min=0.0).clamp(max=1.0)

        interp_val = torch.lerp(y0, y1, factor)
        final_val = torch.where(is_degenerate, y0, interp_val)
        return final_val

    # @staticmethod
    # @torch.compile
    def _lut_entry_2d_vectorized(self, input_trans,
                                 output_caps,
                                 trans_tables_batch,
                                 cap_tables_batch,
                                 lut_values_batch,
                                 trans_dims_actual,
                                 cap_dims_actual
                                ):
        """
        Performs vectorized 2D bilinear interpolation for a full batch.
        """
        device = input_trans.device
        dtype = input_trans.dtype
        batch_size = input_trans.shape[0]
        denom_epsilon = torch.tensor(1e-12, device=device, dtype=dtype)

        # Find Neighboring Indices
        trans_idx_padded = torch.searchsorted(trans_tables_batch, input_trans.unsqueeze(1), right=True).squeeze(1)
        cap_idx_padded = torch.searchsorted(cap_tables_batch, output_caps.unsqueeze(1), right=True).squeeze(1)

        max_trans_idx_actual = (trans_dims_actual - 1).clamp(min=0)
        max_cap_idx_actual = (cap_dims_actual - 1).clamp(min=0)

        trans_idx = trans_idx_padded.clamp(min=1).clamp(max=max_trans_idx_actual)
        cap_idx = cap_idx_padded.clamp(min=1).clamp(max=max_cap_idx_actual)
        trans_idx_low = (trans_idx - 1).clamp(min=0)
        cap_idx_low = (cap_idx - 1).clamp(min=0)

        # Gather Boundary Coordinates (t0, t1, c0, c1)
        batch_indices = torch.arange(batch_size, device=device)
        t0 = trans_tables_batch[batch_indices, trans_idx_low]
        t1 = trans_tables_batch[batch_indices, trans_idx]
        c0 = cap_tables_batch[batch_indices, cap_idx_low]
        c1 = cap_tables_batch[batch_indices, cap_idx]

        # Gather Corner Values (v00, v01, v10, v11)
        # WARNING: Using `cap_dims_actual` as the stride assumes that each LUT in
        # `lut_values_batch` is tightly packed without padding. If `lut_values_batch`
        # is derived from a tensor where each row is padded to a uniform width (MaxC),
        # then using `MaxC` as the stride is the correct approach. This implementation
        # follows the user's request but may lead to incorrect indexing for padded data.
        stride = cap_dims_actual # [B]
        idx00 = trans_idx_low * stride + cap_idx_low
        idx01 = trans_idx_low * stride + cap_idx
        idx10 = trans_idx * stride + cap_idx_low
        idx11 = trans_idx * stride + cap_idx

        # === gather 操作优化方案 ===
        # 原始代码使用4个独立的gather操作，这会造成性能瓶颈
        
        # # 优化方案2: 简单批量gather (备选)
        # corner_indices = torch.stack([idx00, idx01, idx10, idx11], dim=1)  # [B, 4]
        # corner_values = lut_values_batch.gather(1, corner_indices)  # [B, 4]
        # v00, v01, v10, v11 = corner_values[:, 0], corner_values[:, 1], corner_values[:, 2], corner_values[:, 3]
        
        # 优化方案3: 高级索引 (某些情况下可能更快，但内存使用可能更高)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
        corner_indices_2d = torch.stack([
            torch.stack([batch_indices.squeeze(1), idx00], dim=1),
            torch.stack([batch_indices.squeeze(1), idx01], dim=1), 
            torch.stack([batch_indices.squeeze(1), idx10], dim=1),
            torch.stack([batch_indices.squeeze(1), idx11], dim=1)
        ], dim=1)  # [B, 4, 2]
        corner_values = lut_values_batch[corner_indices_2d[:, :, 0], corner_indices_2d[:, :, 1]]
        v00, v01, v10, v11 = corner_values[:, 0], corner_values[:, 1], corner_values[:, 2], corner_values[:, 3]

        # Perform Bilinear Interpolation
        t_interval = t1 - t0
        c_interval = c1 - c0
        is_t_degenerate = torch.abs(t_interval) < denom_epsilon
        is_c_degenerate = torch.abs(c_interval) < denom_epsilon

        input_trans_clamped = input_trans.clamp(min=t0, max=t1)
        output_caps_clamped = output_caps.clamp(min=c0, max=c1)

        t_interval_safe = torch.where(is_t_degenerate, denom_epsilon, t_interval)
        c_interval_safe = torch.where(is_c_degenerate, denom_epsilon, c_interval)
        safe_denominator = t_interval_safe * c_interval_safe

        wa = (t1 - input_trans_clamped) * (c1 - output_caps_clamped)
        wb = (t1 - input_trans_clamped) * (output_caps_clamped - c0)
        wc = (input_trans_clamped - t0) * (c1 - output_caps_clamped)
        wd = (input_trans_clamped - t0) * (output_caps_clamped - c0)

        bilinear_val = (v00 * wa + v01 * wb + v10 * wc + v11 * wd) / safe_denominator

        # Handle degenerate cases using linear interpolation (lerp)
        lerp_c_factor = ((output_caps_clamped - c0) / c_interval_safe).clamp(min=0.0).clamp(max=1.0)
        lerp_t_factor = ((input_trans_clamped - t0) / t_interval_safe).clamp(min=0.0).clamp(max=1.0)
        val_t_degenerate = torch.lerp(v00, v01, lerp_c_factor)
        val_c_degenerate = torch.lerp(v00, v10, lerp_t_factor)

        # Combine results based on which dimensions are degenerate
        return torch.where(
            is_t_degenerate & is_c_degenerate, v00,
            torch.where(is_t_degenerate, val_t_degenerate,
                        torch.where(is_c_degenerate, val_c_degenerate, bilinear_val))
        )

    def lut_entry_vectorized(self,
                             lib_cell_idxs,
                             input_trans,
                             output_caps,
                             arc_idxs,
                             luts_info: LUTS_INFO
                             ):
        """
        执行 LUT 插值的顶层函数。
        此版本通过掩码将数据分批，然后分别处理。
        注意：此方法在逻辑上更清晰，但在GPU上可能因数据重组和同步而比
              完全向量化的版本慢。
        """
        device = input_trans.device
        dtype = input_trans.dtype
        batch_size = arc_idxs.shape[0]

        # --- 1. 获取当前批次对应的掩码和维度 ---
        is_scalar_batch = luts_info.is_scalar[arc_idxs]
        is_trans_1d_batch = luts_info.is_trans_1d[arc_idxs]
        is_cap_1d_batch = luts_info.is_cap_1d[arc_idxs]
        is_2d_batch = luts_info.is_2d[arc_idxs]
        valid_arc_mask_batch = luts_info.valid_arc_mask[arc_idxs]

        final_value = torch.zeros(batch_size, device=device, dtype=dtype)

        # --- 2. 分批处理不同情况 ---

        # 情况：2D LUT (双线性插值)
        idx_2d = is_2d_batch.nonzero().squeeze(-1)
        if idx_2d.numel() > 0:
            idx_2d_batch = arc_idxs[idx_2d]
            val_2d = self._lut_entry_2d_vectorized(
                input_trans=input_trans[idx_2d],
                output_caps=output_caps[idx_2d],
                trans_tables_batch=luts_info.flat_luts_trans_table[idx_2d_batch],
                cap_tables_batch=luts_info.flat_luts_cap_table[idx_2d_batch],
                lut_values_batch=luts_info.flat_luts_values[idx_2d_batch],
                trans_dims_actual=luts_info.trans_dims_actual[idx_2d_batch],
                cap_dims_actual=luts_info.cap_dims_actual[idx_2d_batch]
            )
            final_value[idx_2d] = val_2d

        # 情况：依赖于 input_trans 的 1D LUT
        idx_trans_1d = is_trans_1d_batch.nonzero().squeeze(-1)
        if idx_trans_1d.numel() > 0:
            val_1d_t = self._lut_entry_1d_vectorized(
                x=input_trans[idx_trans_1d],
                x_table=luts_info.flat_luts_trans_table[arc_idxs[idx_trans_1d]],
                y_table=luts_info.flat_luts_values[arc_idxs[idx_trans_1d]],
                actual_dims=luts_info.trans_dims_actual[arc_idxs[idx_trans_1d]]
            )
            final_value[idx_trans_1d] = val_1d_t

        # 情况：依赖于 output_caps 的 1D LUT
        idx_cap_1d = is_cap_1d_batch.nonzero().squeeze(-1)
        if idx_cap_1d.numel() > 0:
            val_1d_c = self._lut_entry_1d_vectorized(
                x=output_caps[idx_cap_1d],
                x_table=luts_info.flat_luts_cap_table[arc_idxs[idx_cap_1d]],
                y_table=luts_info.flat_luts_values[arc_idxs[idx_cap_1d]],
                actual_dims=luts_info.cap_dims_actual[arc_idxs[idx_cap_1d]]
            )
            final_value[idx_cap_1d] = val_1d_c
            
        # 情况：标量 LUT (0D)
        idx_scalar = is_scalar_batch.nonzero().squeeze(-1)
        if idx_scalar.numel() > 0:
            # 对于标量，值就是 LUT values 表的第一个元素
            final_value[idx_scalar] = luts_info.flat_luts_values[arc_idxs[idx_scalar], 0]

        # --- 3. 应用最终掩码处理有效性和非有限值 ---
        return torch.where(
            valid_arc_mask_batch & torch.isfinite(final_value),
            final_value,
            torch.tensor(0.0, device=device, dtype=dtype)
        )

    def r_setup_entry(self, lib_cell_idxs, clk_pin_rtrans, data_pin_trans, lib_arc_idxs):
        luts = self.arcs_info.r_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, clk_pin_rtrans, data_pin_trans, lib_arc_idxs, luts
        )

    def f_setup_entry(self, lib_cell_idxs, clk_pin_rtrans, data_pin_trans, lib_arc_idxs):
        luts = self.arcs_info.f_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, clk_pin_rtrans, data_pin_trans, lib_arc_idxs, luts
        )

    # --- Vectorized LUT Entry Functions (Updated to call vectorized lut_entry) ---
    # These now directly call the vectorized function, no vmap needed here.
    def r_delay_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.r_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs, luts
        )

    def f_delay_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.f_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs, luts
        )

    def r_tran_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.r_trans_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs, luts
        )

    def f_tran_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.f_trans_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs, luts
        )
    
    def calculate_clk2q_aat(self, 
                            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                            pin_net_cap_rise, pin_net_cap_fall,
                            cell_arc_r_delays, cell_arc_f_delays):
        device = pin_rAAT.device
        
        level_cells = self.FF_ids

        # Get outpin and inpin ranges for all cells
        cell_arcs_start = self.inst_flat_arcs_start[level_cells]
        cell_arcs_end = self.inst_flat_arcs_start[level_cells + 1]

        # Compute number of outpins and inpins per cell
        num_arcs = cell_arcs_end - cell_arcs_start

        # Generate all outpin indices
        max_arcs = num_arcs.max().item()
        if max_arcs == 0:
            return torch.tensor([], device=device, dtype=torch.long), pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays

        arcs_indices = torch.arange(max_arcs, device=device).unsqueeze(0)
        arcs_mask = arcs_indices < num_arcs.unsqueeze(1)
        arcs_global_indices = arcs_indices + cell_arcs_start.unsqueeze(1)
        level_inst_arcs = self.inst_flat_arcs[arcs_global_indices[arcs_mask]]

        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]
        lib_cell_idxs = level_inst_arcs[:, 2]
        lib_arc_idxs = level_inst_arcs[:, 3]
        timing_senses = level_inst_arcs[:, 4]
        timing_types = level_inst_arcs[:, 5]

        # --- 1. 准备输入条件 ---
        pin_r_slew_in = pin_rtran[arc_in_pins] 
        pin_f_slew_in = pin_ftran[arc_in_pins]
        pin_r_load_out = pin_net_cap_rise[arc_out_pins]
        pin_f_load_out = pin_net_cap_fall[arc_out_pins]

        # --- 2. 计算所有可能的延时和转换时间 ---
        delay_rr = self.r_delay_entry(lib_cell_idxs, pin_r_slew_in, pin_r_load_out, lib_arc_idxs)
        delay_rf = self.f_delay_entry(lib_cell_idxs, pin_r_slew_in, pin_f_load_out, lib_arc_idxs)
        delay_fr = self.r_delay_entry(lib_cell_idxs, pin_f_slew_in, pin_r_load_out, lib_arc_idxs)
        delay_ff = self.f_delay_entry(lib_cell_idxs, pin_f_slew_in, pin_f_load_out, lib_arc_idxs)
        
        tran_rr = self.r_tran_entry(lib_cell_idxs, pin_r_slew_in, pin_r_load_out, lib_arc_idxs)
        tran_rf = self.f_tran_entry(lib_cell_idxs, pin_r_slew_in, pin_f_load_out, lib_arc_idxs)
        tran_fr = self.r_tran_entry(lib_cell_idxs, pin_f_slew_in, pin_r_load_out, lib_arc_idxs)
        tran_ff = self.f_tran_entry(lib_cell_idxs, pin_f_slew_in, pin_f_load_out, lib_arc_idxs)
        
        # --- 3. 创建 TimingSense 和 TimingType 的掩码 ---
        # timing_senses: 1 for positive_unate, -1 for negative_unate, 0 for non_unate
        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)
        # is_non_unate = (timing_senses == 0) # non_unate is where not pos and not neg

        # timing_types: 1 for rising_edge, -1 for falling_edge, 0 for both_edge
        # 假设下降沿用-1表示，如果不是，请相应修改
        is_rising_edge = (timing_types == 1)
        is_falling_edge = (timing_types == -1)
        is_both_edge = (timing_types == 0)

        # --- 4. 根据 Unateness 计算四种可能的延迟分量 ---
        # 为了在后续max操作中正确处理无效路径，将无效延迟设置为-inf
        NEG_INF = -torch.inf

        # 由 Rising Edge Clock 触发
        # 输出Rise延时: 发生在 pos_unate 或 non_unate
        r_delays_re = torch.where(is_neg_unate, NEG_INF, delay_rr)
        # 输出Fall延时: 发生在 neg_unate 或 non_unate
        f_delays_re = torch.where(is_pos_unate, NEG_INF, delay_rf)

        # 由 Falling Edge Clock 触发
        # 输出Rise延时: 发生在 neg_unate 或 non_unate
        r_delays_fe = torch.where(is_pos_unate, NEG_INF, delay_fr)
        # 输出Fall延时: 发生在 pos_unate 或 non_unate
        f_delays_fe = torch.where(is_neg_unate, NEG_INF, delay_ff)

        # 对 Slew/Transition 应用相同的逻辑
        r_trans_re = torch.where(is_neg_unate, NEG_INF, tran_rr)
        f_trans_re = torch.where(is_pos_unate, NEG_INF, tran_rf)
        r_trans_fe = torch.where(is_pos_unate, NEG_INF, tran_fr)
        f_trans_fe = torch.where(is_neg_unate, NEG_INF, tran_ff)

        # --- 5. 计算四种核心的 AAT 和 Tran 更新量 ---
        # AAT_out = AAT_in + delay
        r_aat_update_from_re = pin_rAAT[arc_in_pins] + r_delays_re
        f_aat_update_from_re = pin_rAAT[arc_in_pins] + f_delays_re
        r_aat_update_from_fe = pin_fAAT[arc_in_pins] + r_delays_fe
        f_aat_update_from_fe = pin_fAAT[arc_in_pins] + f_delays_fe

        r_tran_update_from_re = r_trans_re
        f_tran_update_from_re = f_trans_re
        r_tran_update_from_fe = r_trans_fe
        f_tran_update_from_fe = f_trans_fe

        # --- 6. 根据 TimingType 组合最终的更新值 ---
        # 对于双边沿触发 (is_both_edge)，我们需要取两种可能触发情况下的最大值，以符合STA的最差情况分析
        
        # Rise AAT 更新
        r_aat_updates = torch.where(
            is_rising_edge, 
            r_aat_update_from_re, 
            torch.where(
                is_falling_edge,
                r_aat_update_from_fe,
                torch.max(r_aat_update_from_re, r_aat_update_from_fe) # both_edge case
            )
        )

        # Fall AAT 更新
        f_aat_updates = torch.where(
            is_rising_edge,
            f_aat_update_from_re,
            torch.where(
                is_falling_edge,
                f_aat_update_from_fe,
                torch.max(f_aat_update_from_re, f_aat_update_from_fe) # both_edge case
            )
        )

        # Rise Tran 更新
        r_tran_updates = torch.where(
            is_rising_edge,
            r_tran_update_from_re,
            torch.where(
                is_falling_edge,
                r_tran_update_from_fe,
                torch.max(r_tran_update_from_re, r_tran_update_from_fe) # both_edge case
            )
        )

        # Fall Tran 更新
        f_tran_updates = torch.where(
            is_rising_edge,
            f_tran_update_from_re,
            torch.where(
                is_falling_edge,
                f_tran_update_from_fe,
                torch.max(f_tran_update_from_re, f_tran_update_from_fe) # both_edge case
            )
        )

        # --- 7. 存储和更新 AAT/Tran (scatter_reduce) ---
        # 存储每个弧的最终有效延迟 (可选，用于调试或报告)
        # 注意：这里的delay只是最终选择的那个，而不是AAT的增量
        final_r_delays = torch.where(r_aat_updates > NEG_INF, r_aat_updates - torch.where(r_aat_updates == r_aat_update_from_re, pin_rAAT[arc_in_pins], pin_fAAT[arc_in_pins]), 0.0)
        final_f_delays = torch.where(f_aat_updates > NEG_INF, f_aat_updates - torch.where(f_aat_updates == f_aat_update_from_re, pin_rAAT[arc_in_pins], pin_fAAT[arc_in_pins]), 0.0)
        cell_arc_r_delays.scatter_(0, lib_arc_idxs.long(), final_r_delays)
        cell_arc_f_delays.scatter_(0, lib_arc_idxs.long(), final_f_delays)

        # 使用 scatter_reduce 更新下游引脚的 AAT 和 Tran
        # 'amax' 确保了汇聚点(fan-in)的 AAT 是所有到达路径中的最大值
        pin_rAAT = torch.scatter_reduce(pin_rAAT, 0, arc_out_pins.long(), r_aat_updates, reduce="amax", include_self=False)
        pin_fAAT = torch.scatter_reduce(pin_fAAT, 0, arc_out_pins.long(), f_aat_updates, reduce="amax", include_self=False)
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_out_pins.long(), r_tran_updates, reduce="amax", include_self=False)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_out_pins.long(), f_tran_updates, reduce="amax", include_self=False)
        
        net_in_pins = torch.unique(self.pin_net[arc_out_pins])
        return net_in_pins, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays
    
    # @torch.compile    
    def calculate_cell_aat_level(self, level_cells,
                                pin_rAAT, pin_fAAT,
                                pin_rtran, pin_ftran,
                                pin_net_cap_rise, pin_net_cap_fall,
                                cell_arc_r_delays, cell_arc_f_delays):
        """
        Optimized version of the AAT calculation function.
        It groups arcs by their 'unate' property (timing sense) to avoid
        redundant delay and transition calculations.
        """
        start_time = time.time()
        device = pin_rAAT.device

        # Get outpin and inpin ranges for all cells
        cell_arcs_start = self.inst_flat_arcs_start[level_cells]
        cell_arcs_end = self.inst_flat_arcs_start[level_cells + 1]

        # Compute number of arcs per cell
        num_arcs = cell_arcs_end - cell_arcs_start

        # Generate all arc indices for the current level
        max_arcs = num_arcs.max().item()
        if max_arcs == 0:
            return torch.tensor([], device=device, dtype=torch.long), pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays

        arcs_indices = torch.arange(max_arcs, device=device).unsqueeze(0)
        arcs_mask = arcs_indices < num_arcs.unsqueeze(1)
        arcs_global_indices = arcs_indices + cell_arcs_start.unsqueeze(1)
        level_inst_arcs = self.inst_flat_arcs[arcs_global_indices[arcs_mask]]

        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]
        lib_cell_idxs = level_inst_arcs[:, 2]
        lib_arc_idxs = level_inst_arcs[:, 3]
        timing_senses = level_inst_arcs[:, 4]

        if (arc_out_pins == 332754 ).any():
            logging.warning(f"lib_arc_idxs: {lib_arc_idxs}")

        # 1. 准备输入条件 (Prepare inputs)
        pin_r_slew_in = pin_rtran[arc_in_pins]
        pin_f_slew_in = pin_ftran[arc_in_pins]
        pin_r_load_out = pin_net_cap_rise[arc_out_pins]
        pin_f_load_out = pin_net_cap_fall[arc_out_pins]

        # 2. 根据 Unate 类型进行分组 (Group arcs by unate type)
        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)
        is_non_unate = ~ (is_pos_unate | is_neg_unate)

        num_level_arcs = level_inst_arcs.shape[0]
        r_delays = torch.zeros(num_level_arcs, device=device, dtype=torch.float32)
        f_delays = torch.zeros(num_level_arcs, device=device, dtype=torch.float32)
        r_trans = torch.zeros(num_level_arcs, device=device, dtype=torch.float32)
        f_trans = torch.zeros(num_level_arcs, device=device, dtype=torch.float32)

        # --- 3. 按需计算每种 Unate 类型的 Delay 和 Tran ---

        # A. Positive Unate (r->r, f->f)
        if torch.any(is_pos_unate):
            mask = is_pos_unate
            # Rise -> Rise
            delay_rr = self.r_delay_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            tran_rr = self.r_tran_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            # Fall -> Fall
            delay_ff = self.f_delay_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            tran_ff = self.f_tran_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])

            r_delays[mask] = delay_rr
            f_delays[mask] = delay_ff
            r_trans[mask] = tran_rr
            f_trans[mask] = tran_ff

        # B. Negative Unate (f->r, r->f)
        if torch.any(is_neg_unate):
            mask = is_neg_unate
            # Fall -> Rise
            delay_fr = self.r_delay_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            tran_fr = self.r_tran_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            # Rise -> Fall
            delay_rf = self.f_delay_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            tran_rf = self.f_tran_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])

            r_delays[mask] = delay_fr
            f_delays[mask] = delay_rf
            r_trans[mask] = tran_fr
            f_trans[mask] = tran_rf

        # C. Non-Unate (worst of all applicable cases)
        if torch.any(is_non_unate):
            mask = is_non_unate
            # Rise output calculation (worst of r->r and f->r)
            delay_rr_non = self.r_delay_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            tran_rr_non = self.r_tran_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            delay_fr_non = self.r_delay_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            tran_fr_non = self.r_tran_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_r_load_out[mask], lib_arc_idxs[mask])
            r_delays[mask] = torch.maximum(delay_rr_non, delay_fr_non)
            r_trans[mask] = torch.maximum(tran_rr_non, tran_fr_non)

            # Fall output calculation (worst of f->f and r->f)
            delay_ff_non = self.f_delay_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            tran_ff_non = self.f_tran_entry(lib_cell_idxs[mask], pin_f_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            delay_rf_non = self.f_delay_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            tran_rf_non = self.f_tran_entry(lib_cell_idxs[mask], pin_r_slew_in[mask], pin_f_load_out[mask], lib_arc_idxs[mask])
            f_delays[mask] = torch.maximum(delay_ff_non, delay_rf_non)
            f_trans[mask] = torch.maximum(tran_ff_non, tran_rf_non)

        # --- 4. 存储和更新 (Store and Update) ---
        # The logic from here remains the same, as r_delays, f_delays, etc. are now fully populated.

        scatter_indices = arcs_global_indices[arcs_mask]
        cell_arc_r_delays.scatter_(0, scatter_indices.long(), r_delays)
        cell_arc_f_delays.scatter_(0, scatter_indices.long(), f_delays)

        r_aat_updates = pin_rAAT[arc_in_pins] + r_delays
        f_aat_updates = pin_fAAT[arc_in_pins] + f_delays

        pin_rAAT = torch.scatter_reduce(pin_rAAT, 0, arc_out_pins.long(), r_aat_updates, reduce="amax", include_self=False)
        pin_fAAT = torch.scatter_reduce(pin_fAAT, 0, arc_out_pins.long(), f_aat_updates, reduce="amax", include_self=False)
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_out_pins.long(), r_trans, reduce="amax", include_self=False)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_out_pins.long(), f_trans, reduce="amax", include_self=False)

        net_in_pins = torch.unique(self.pin_net[arc_out_pins])
        logging.debug(f"Cell AAT Level Time: {time.time() - start_time:.4f}s")
        return net_in_pins, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays

    def calculate_net_aat_level(self, curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                                pin_net_delay_rise, pin_net_delay_fall,
                                pin_net_impulse_rise, pin_net_impulse_fall):
        start_time = time.time()
        net_arcs_starts = self.net_flat_arcs_start[curnets]
        net_arcs_ends = self.net_flat_arcs_start[curnets + 1]
        num_net_fopins = net_arcs_ends - net_arcs_starts
        if num_net_fopins.numel() == 0:
            return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

        max_net_fopins = num_net_fopins.max().item()
        net_arcs_indices = torch.arange(
            max_net_fopins, device=pin_rAAT.device).unsqueeze(0)
        net_arcs_mask = net_arcs_indices < num_net_fopins.unsqueeze(1)
        net_arcs_global_indices = net_arcs_indices + \
            net_arcs_starts.unsqueeze(1)
        net_arcs_level = self.net_flat_arcs[net_arcs_global_indices[net_arcs_mask]]

        arc_fopins = net_arcs_level[:, 1]
        arc_fipins = net_arcs_level[:, 0]

        wire_delays_rise = pin_net_delay_rise[arc_fopins]
        wire_delays_fall = pin_net_delay_fall[arc_fopins]
        impulse_rise = pin_net_impulse_rise[arc_fopins]
        impulse_fall = pin_net_impulse_fall[arc_fopins]

        r_aat_updates = pin_rAAT[arc_fipins] + wire_delays_rise
        f_aat_updates = pin_fAAT[arc_fipins] + wire_delays_fall
        r_tran_updates = torch.sqrt(pin_rtran[arc_fipins]**2 + impulse_rise)
        f_tran_updates = torch.sqrt(pin_ftran[arc_fipins]**2 + impulse_fall)

        pin_rAAT = torch.scatter_reduce(pin_rAAT, 0, arc_fopins.long(
        ), r_aat_updates, reduce="amax", include_self=True)
        pin_fAAT = torch.scatter_reduce(pin_fAAT, 0, arc_fopins.long(
        ), f_aat_updates, reduce="amax", include_self=True)
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_fopins.long(
        ), r_tran_updates, reduce="amax", include_self=True)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_fopins.long(
        ), f_tran_updates, reduce="amax", include_self=True)
        logging.debug(f"Net AAT Level Time: {time.time() - start_time:.4f}s")
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def calculate_cell_rat_level(self, level_cells, pin_rRAT, pin_fRAT, cell_arc_r_delays, cell_arc_f_delays):
        start_time = time.time()
        device = pin_rRAT.device

        cell_arcs_start = self.inst_flat_arcs_start[level_cells]
        cell_arcs_end = self.inst_flat_arcs_start[level_cells + 1]

        num_arcs = cell_arcs_end - cell_arcs_start

        max_arcs = num_arcs.max().item()
        arcs_indices = torch.arange(max_arcs, device=device).unsqueeze(0)
        arcs_mask = arcs_indices < num_arcs.unsqueeze(1)
        arcs_global_indices = arcs_indices + cell_arcs_start.unsqueeze(1)
        level_inst_arcs = self.inst_flat_arcs[arcs_global_indices[arcs_mask]]

        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]

        scatter_indices = arcs_global_indices[arcs_mask]

        f_delays = cell_arc_f_delays[scatter_indices]
        r_delays = cell_arc_r_delays[scatter_indices]

        # if (arc_in_pins == 7850   ).any():
        #     logging.warning(torch.where(arc_in_pins == 7850  ))

        assert pin_rRAT[arc_out_pins].max(
            ) <= 1e8, "Negative r_rat_updates detected"
        r_rat_updates = pin_rRAT[arc_out_pins] - r_delays
        f_rat_updates = pin_fRAT[arc_out_pins] - f_delays

        pin_rRAT = torch.scatter_reduce(pin_rRAT, 0, arc_in_pins.long(
        ), r_rat_updates, reduce="amin", include_self=True)
        pin_fRAT = torch.scatter_reduce(pin_fRAT, 0, arc_in_pins.long(
        ), f_rat_updates, reduce="amin", include_self=True)
        # assert pin_rRAT[arc_in_pins].min(
        # ) >= 0, "Negative r_rat_updates detected"
        # assert pin_fRAT[arc_in_pins].min(
        # ) >= 0, "Negative f_rat_updates detected"

        cur_endpoints = torch.unique(arc_in_pins)
        logging.debug(f"Cell RAT Level Time: {time.time() - start_time:.4f}s")
        return cur_endpoints, pin_rRAT, pin_fRAT

    def calculate_net_rat_level(self, cur_endpoint, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall):

        curnets = self.pin_net[cur_endpoint]

        arc_fipins = self.net2driver_pin_map[curnets]
        wire_delays_rise = pin_net_delay_rise[cur_endpoint]
        wire_delays_fall = pin_net_delay_fall[cur_endpoint]
        assert pin_rRAT[cur_endpoint].max(
            ) <= 1e8, "Negative r_rat_updates detected"
        r_rat_updates = pin_rRAT[cur_endpoint] - wire_delays_rise
        f_rat_updates = pin_fRAT[cur_endpoint] - wire_delays_fall
        # assert r_rat_updates.max() < 5e4, "r_rat_updates exceed expected range"
        # assert f_rat_updates.max() < 5e4, "f_rat_updates exceed expected range"

        pin_rRAT = torch.scatter_reduce(pin_rRAT, 0, arc_fipins.long(
        ), r_rat_updates, reduce="amin", include_self=True)
        pin_fRAT = torch.scatter_reduce(pin_fRAT, 0, arc_fipins.long(
        ), f_rat_updates, reduce="amin", include_self=True)
        assert pin_rRAT[arc_fipins].max(
            ) <= 1e8, "Negative r_rat_updates detected after scatter_reduce"
        return pin_rRAT, pin_fRAT

    def calculate_setup_rat(self, pin_rRAT, pin_fRAT, clk_pin_rtran, clk_pin_ftran, pin_rtran, pin_ftran):

        arc_in_pins = self.endpoints_constraint_arcs[:, 0]
        arc_out_pins = self.endpoints_constraint_arcs[:, 1]
        lib_cell_idxs = self.endpoints_constraint_arcs[:, 2]
        lib_arc_idxs = self.endpoints_constraint_arcs[:, 3]
        timing_senses = self.endpoints_constraint_arcs[:, 4]

        pin_clk_rtran_in = clk_pin_rtran[arc_in_pins]
        pin_clk_ftran_in = clk_pin_ftran[arc_in_pins]
        pin_data_rtran_in = pin_rtran[arc_out_pins]
        pin_data_ftran_in = pin_ftran[arc_out_pins]

        # if (arc_out_pins == 4720 ).any():
        #     logging.info(torch.where(arc_out_pins == 4720))
        #     logging.warning(f"lib_arc_idxs: {lib_arc_idxs}")
            
        # R->R
        rr_setup_time = self.r_setup_entry(lib_cell_idxs, pin_clk_rtran_in,
                                           pin_data_rtran_in, lib_arc_idxs)
        ff_setup_time = self.f_setup_entry(lib_cell_idxs, pin_clk_ftran_in,
                                           pin_data_ftran_in, lib_arc_idxs)

        fr_setup_time = self.r_setup_entry(lib_cell_idxs, pin_clk_ftran_in,
                                           pin_data_rtran_in, lib_arc_idxs)
        rf_setup_time = self.f_setup_entry(lib_cell_idxs, pin_clk_rtran_in,
                                           pin_data_ftran_in, lib_arc_idxs)

        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)

        # --- 计算最终的上升输出延时 (r_delays) ---
        # Positive unate : r->r / f->f
        # Negative unate : f->r / r->f
        # Non-unate : worst of r->r, f->r, f->f, r->f
        r_delays_non_unate = torch.maximum(rr_setup_time, fr_setup_time)
        f_delays_non_unate = torch.maximum(ff_setup_time, rf_setup_time)
        r_setup_time = torch.where(is_pos_unate, rr_setup_time,
                                   torch.where(is_neg_unate, fr_setup_time, r_delays_non_unate))
        f_setup_time = torch.where(is_pos_unate, ff_setup_time,
                                   torch.where(is_neg_unate, rf_setup_time, f_delays_non_unate))

        # assert pin_rRAT[arc_out_pins].min(
        # ) >= 0, "Negative r_rat_updates detected"

        pin_rRAT[arc_out_pins] = pin_rRAT[arc_out_pins] - r_setup_time
        pin_fRAT[arc_out_pins] = pin_fRAT[arc_out_pins] - f_setup_time

        return pin_rRAT, pin_fRAT

    def forward(self, pin_net_delays, pin_net_impulses, pin_net_caps):
        start_time = time.time()
        pin_net_delay_rise = pin_net_delays['rise'].clone()
        pin_net_delay_fall = pin_net_delays['fall'].clone()
        pin_net_impulse_rise = pin_net_impulses['rise'].clone()
        pin_net_impulse_fall = pin_net_impulses['fall'].clone()
        pin_net_cap_rise = pin_net_caps['rise'].clone()
        pin_net_cap_fall = pin_net_caps['fall'].clone()

        device = pin_net_delay_rise.device
        dtype = pin_net_delay_rise.dtype

        # --- Initialization ---
        pin_rAAT = torch.zeros(self.num_pins, device=device, dtype=dtype)
        pin_fAAT = torch.zeros(self.num_pins, device=device, dtype=dtype)
        pin_rtran = torch.zeros(self.num_pins, device=device, dtype=dtype)
        pin_ftran = torch.zeros(self.num_pins, device=device, dtype=dtype)
        inf_val = torch.tensor(2e8, device=self.device, dtype=self.dtype)
        pin_rRAT = torch.full((self.num_pins,), inf_val,
                              device=self.device, dtype=self.dtype)
        pin_fRAT = torch.full((self.num_pins,), inf_val,
                              device=self.device, dtype=self.dtype)
        pin_rRAT[self.end_points] = torch.tensor(
            self.endpoints_rRAT, device=self.device, dtype=self.dtype)
        pin_fRAT[self.end_points] = torch.tensor(
            self.endpoints_fRAT, device=self.device, dtype=self.dtype)

        num_arcs_total = self.inst_flat_arcs[:, 3].shape[0] + \
            1 if self.inst_flat_arcs.numel() > 0 else 1
        cell_arc_r_delays = torch.zeros(
            num_arcs_total, device=device, dtype=dtype)
        cell_arc_f_delays = torch.zeros(
            num_arcs_total, device=device, dtype=dtype)

        pin_rAAT[self.start_points] = self.inrdelays
        pin_fAAT[self.start_points] = self.infdelays
        pin_rtran[self.start_points] = self.inrtrans
        pin_ftran[self.start_points] = self.inftrans
        # pin_net_cap = pin_net_cap.clone()  # Clone to avoid modifying the original tensor
        # pin_net_cap[self.end_points] = pin_net_cap[self.end_points] + self.outcaps

        cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays = self.calculate_clk2q_aat(
            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
            pin_net_cap_rise, pin_net_cap_fall,
            cell_arc_r_delays, cell_arc_f_delays
        )

        pi_nets = torch.unique(self.pin_net[self.start_points])
        pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = self.calculate_net_aat_level(
            pi_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
            pin_net_delay_rise, pin_net_delay_fall,
            pin_net_impulse_rise, pin_net_impulse_fall
        )

        for level in range(len(self.flat_cells_by_level_start) - 1):
            start = self.flat_cells_by_level_start[level]
            end = self.flat_cells_by_level_start[level + 1]
            level_cells = self.flat_cells_by_level[start: end]
            cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays = self.calculate_cell_aat_level(
                level_cells, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                pin_net_cap_rise, pin_net_cap_fall,
                cell_arc_r_delays, cell_arc_f_delays
            )
            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = self.calculate_net_aat_level(
                cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                pin_net_delay_rise, pin_net_delay_fall,
                pin_net_impulse_rise, pin_net_impulse_fall
            )
        # --- Calculate RAT Levels ---

        pin_rRAT, pin_fRAT = self.calculate_setup_rat(
            pin_rRAT, pin_fRAT, self.clk_pin_rtran, self.clk_pin_ftran, pin_rtran, pin_ftran)

        pin_rRAT, pin_fRAT = self.calculate_net_rat_level(
            self.end_points, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall
        )

        for level in range(len(self.flat_cells_by_reverse_level_start) - 1):
            start = self.flat_cells_by_reverse_level_start[level]
            end = self.flat_cells_by_reverse_level_start[level + 1]
            level_cells = self.flat_cells_by_reverse_level[start: end]
            cur_endpoints, pin_rRAT, pin_fRAT = self.calculate_cell_rat_level(
                level_cells, pin_rRAT, pin_fRAT, cell_arc_r_delays, cell_arc_f_delays
            )
            pin_rRAT, pin_fRAT = self.calculate_net_rat_level(
                cur_endpoints, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall
            )
        rslack = pin_rRAT - pin_rAAT
        fslack = pin_fRAT - pin_fAAT
        slack = torch.min(rslack, fslack)
        RAT_THRESHOLD = 8e7 
        # valid_mask = (pin_rRAT < RAT_THRESHOLD) & (pin_fRAT < RAT_THRESHOLD)
        # all_valid_slacks = slack[valid_mask]
        endpoints_slack = slack[self.end_points]
        # neg_slack = torch.clamp(all_valid_slacks, max=0)
        neg_endpoint_slack = torch.clamp(endpoints_slack, max=0)
        ws = torch.min(endpoints_slack)
        ts = 0
        wns = torch.min(neg_endpoint_slack)
        tns = torch.sum(neg_endpoint_slack)

        self.pin_rAAT, self.pin_fAAT, self.pin_rRAT, self.pin_fRAT = (
            t.clone().detach() for t in [pin_rAAT, pin_fAAT, pin_rRAT, pin_fRAT])
        self.pin_rtran, self.pin_ftran = (
            t.clone().detach() for t in [pin_rtran, pin_ftran])
        self.pin_net_cap_rise, self.pin_net_cap_fall = (
            t.clone().detach() for t in [pin_net_cap_rise, pin_net_cap_fall])
        self.cell_arc_r_delays, self.cell_arc_f_delays = (
            t.clone().detach() for t in [cell_arc_r_delays, cell_arc_f_delays])
        logging.info(f"WNS: {wns:.4f}, TNS: {tns:.4f}")
        logging.info(f"Total Timing propagation Time: {time.time() - start_time:.4f}s")
        return wns, tns, ws, ts
