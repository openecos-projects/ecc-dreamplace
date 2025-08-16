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


@dataclass
class LUTS_INFO:
    # Provide default values or use field(default_factory=...)
    flat_luts_values: torch.Tensor = None
    flat_luts_trans_table: torch.Tensor = None
    flat_luts_cap_table: torch.Tensor = None
    flat_luts_dim: torch.Tensor = None


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
                 cell_flat_clk_arcs,
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
        self.cell_flat_clk_arcs = cell_flat_clk_arcs
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

    def lut_entry_vectorized(self,
                             # Batched inputs
                             # [B] - For potential future use
                             lib_cell_idxs,
                             input_trans,   # [B]
                             output_caps,   # [B]
                             # [B] - LongTensor indices for the LUTs
                             arc_idxs,
                             # Padded LUT Tensors (Constant for the call)
                             flat_luts_values,      # [N, MaxT * MaxC]
                             flat_luts_trans_table,  # [N, MaxT]
                             flat_luts_cap_table,   # [N, MaxC]
                             flat_luts_dim):        # [N, 2] - Actual dims [trans_dim, cap_dim]
        """
        Performs bilinear interpolation for a batch of arcs using padded LUT tensors.
        Uses vectorized tensor operations instead of vmap.
        """
        device = input_trans.device
        dtype = input_trans.dtype
        batch_size = arc_idxs.shape[0]

        # --- 1. Gather data for the batch of arcs ---
        # Ensure arc_idxs are valid before gathering
        assert arc_idxs.min() >= 0, "Negative arc_idx detected"
        assert arc_idxs.max(
        ) < flat_luts_dim.shape[0], "arc_idx out of bounds for flat_luts_dim"
        # Add similar checks for other flat tensors if necessary

        lut_dims_batch = flat_luts_dim[arc_idxs]             # [B, 2]
        trans_tables_batch = flat_luts_trans_table[arc_idxs]  # [B, MaxT]
        cap_tables_batch = flat_luts_cap_table[arc_idxs]     # [B, MaxC]
        lut_values_batch = flat_luts_values[arc_idxs]        # [B, MaxT, MaxC]

        # --- 2. Get actual dimensions and create masks ---
        # [B] - Actual T dimension for each arc
        trans_dims_actual = lut_dims_batch[:, 0].long()
        # [B] - Actual C dimension for each arc
        cap_dims_actual = lut_dims_batch[:, 1].long()

        # Mask for valid arcs (dimension > 0)
        valid_arc_mask = (trans_dims_actual > 0) & (cap_dims_actual > 0)  # [B]
        if not torch.any(valid_arc_mask):
            # Return all zeros if no valid arcs
            return torch.zeros(batch_size, device=device, dtype=dtype)

        # --- 3. Find Neighboring Indices (Vectorized Searchsorted on Padded) ---
        # Unsqueeze inputs for broadcasting comparison with tables
        input_trans_b = input_trans.unsqueeze(1)  # [B, 1]
        output_caps_b = output_caps.unsqueeze(1)  # [B, 1]

        # Perform searchsorted on the padded tables
        # Note: This finds indices within the padded dimension length (MaxT or MaxC)
        # We need to clamp based on actual dimensions later.
        # Using left=True for lower bound index might simplify low/high index calculation later
        # Let's stick to right=True and calculate low index as idx-1 for now.
        trans_idx_padded = torch.searchsorted(
            trans_tables_batch, input_trans_b, right=True).squeeze(-1)  # [B]
        cap_idx_padded = torch.searchsorted(
            cap_tables_batch, output_caps_b, right=True).squeeze(-1)     # [B]

        # Calculate actual max indices (ensure non-negative)
        max_trans_idx_actual = (trans_dims_actual - 1).clamp(min=0)  # [B]
        max_cap_idx_actual = (cap_dims_actual - 1).clamp(min=0)     # [B]

        # Clamp the indices found by searchsorted to the *actual* valid range [1, max_actual_idx]
        # This ensures t1/c1 indices are within the real data bounds
        trans_idx = trans_idx_padded.clamp(
            min=1).clamp(max=max_trans_idx_actual)  # [B]
        cap_idx = cap_idx_padded.clamp(
            min=1).clamp(max=max_cap_idx_actual)         # [B]

        # Calculate low indices [0, max_actual_idx - 1]
        trans_idx_low = (trans_idx - 1).clamp(min=0)  # [B]
        cap_idx_low = (cap_idx - 1).clamp(min=0)     # [B]

        # --- 4. Gather Boundary Values (t0, t1, c0, c1) ---
        # Use gather or advanced indexing with batch indices
        batch_indices = torch.arange(batch_size, device=device)  # [B]
        t0 = trans_tables_batch[batch_indices, trans_idx_low]  # [B]
        t1 = trans_tables_batch[batch_indices, trans_idx]     # [B]
        c0 = cap_tables_batch[batch_indices, cap_idx_low]     # [B]
        c1 = cap_tables_batch[batch_indices, cap_idx]         # [B]
        # --- 5. Calculate Flattened Indices for lut_values ---
        # ★★★ 修正开始 ★★★
        # 从 cap 坐标轴张量获取物理（填充后）的列数 MaxC
        # 这是获取内存步长(stride)的正确方法
        padded_max_cols = flat_luts_cap_table.shape[1]

        # 使用正确的物理列数来计算一维索引
        idx00 = trans_idx_low * padded_max_cols + cap_idx_low  # [B]
        idx01 = trans_idx_low * padded_max_cols + cap_idx     # [B]
        idx10 = trans_idx * padded_max_cols + cap_idx_low     # [B]
        idx11 = trans_idx * padded_max_cols + cap_idx         # [B]
        # ★★★ 修正结束 ★★★

        # --- 6. Gather Corner Values (v00, v01, v10, v11) ---
        # ★★★ 修正开始 ★★★
        # lut_values_batch 已经是 [B, MaxFlatSize] 的二维张量，无需再做 view()
        lut_values_batch_flat = lut_values_batch

        # 使用 gather 从已经压平的张量中安全地提取数值
        v00 = lut_values_batch_flat.gather(
            1, idx00.unsqueeze(1)).squeeze(1)  # [B]
        v01 = lut_values_batch_flat.gather(
            1, idx01.unsqueeze(1)).squeeze(1)  # [B]
        v10 = lut_values_batch_flat.gather(
            1, idx10.unsqueeze(1)).squeeze(1)  # [B]
        v11 = lut_values_batch_flat.gather(
            1, idx11.unsqueeze(1)).squeeze(1)  # [B]
        # ★★★ 修正结束 ★★★
        # # --- 5. Calculate Flattened Indices for lut_values ---
        # # num_cols is the *actual* cap dimension for each arc
        # num_cols_batch = cap_dims_actual  # [B]
        # idx00 = trans_idx_low * num_cols_batch + cap_idx_low  # [B]
        # idx01 = trans_idx_low * num_cols_batch + cap_idx     # [B]
        # idx10 = trans_idx * num_cols_batch + cap_idx_low     # [B]
        # idx11 = trans_idx * num_cols_batch + cap_idx         # [B]

        # # --- 6. Gather Corner Values (v00, v01, v10, v11) ---
        # # Reshape the batch of 2D LUT values to be flat for 1D indexing
        # # TODO:
        # # B, MaxT, MaxC = lut_values_batch.shape
        # lut_values_batch_flat = lut_values_batch

        # # Check bounds before gathering (indices should be < actual_T * actual_C)
        # # This check is complex with varying dimensions. Trusting the index calculation for now.
        # # A robust implementation might involve masking indices that fall outside the true area.

        # # Gather using batch_indices and the calculated 1D indices
        # v00 = lut_values_batch_flat[batch_indices, idx00]  # [B]
        # v01 = lut_values_batch_flat[batch_indices, idx01]  # [B]
        # v10 = lut_values_batch_flat[batch_indices, idx10]  # [B]
        # v11 = lut_values_batch_flat[batch_indices, idx11]  # [B]

        # --- 7. Perform Interpolation (Vectorized) ---
        t_interval = t1 - t0  # [B]
        c_interval = c1 - c0  # [B]
        denom_epsilon = torch.tensor(1e-12, device=device, dtype=dtype)

        is_t_degenerate = torch.abs(t_interval) < denom_epsilon  # [B]
        is_c_degenerate = torch.abs(c_interval) < denom_epsilon  # [B]

        input_trans_clamped = input_trans.clamp(min=t0, max=t1)  # [B]
        output_caps_clamped = output_caps.clamp(min=c0, max=c1)  # [B]

        wa = (t1 - input_trans_clamped) * (c1 - output_caps_clamped)  # [B]
        wb = (t1 - input_trans_clamped) * (output_caps_clamped - c0)  # [B]
        wc = (input_trans_clamped - t0) * (c1 - output_caps_clamped)  # [B]
        wd = (input_trans_clamped - t0) * (output_caps_clamped - c0)  # [B]

        t_interval_safe = torch.where(
            is_t_degenerate, denom_epsilon, t_interval)  # [B]
        c_interval_safe = torch.where(
            is_c_degenerate, denom_epsilon, c_interval)  # [B]
        safe_denominator = t_interval_safe * c_interval_safe  # [B]

        # Use torch.lerp for linear interpolation parts
        lerp_c_factor = ((output_caps_clamped - c0) / c_interval_safe.clamp(
            min=denom_epsilon)).clamp(0.0, 1.0)  # Clamp factor to [0, 1]
        lerp_t_factor = ((input_trans_clamped - t0) / t_interval_safe.clamp(
            min=denom_epsilon)).clamp(0.0, 1.0)  # Clamp factor to [0, 1]
        val_t_degenerate = torch.lerp(v00, v01, lerp_c_factor)  # [B]
        val_c_degenerate = torch.lerp(v00, v10, lerp_t_factor)  # [B]

        # Bilinear calculation (avoid division by zero)
        # Initialize bilinear_val with a fallback (e.g., v00)
        bilinear_val = torch.full_like(wa, float('nan'))  # Initialize with NaN
        # Mask where division is safe
        valid_denom_mask = torch.abs(safe_denominator) >= denom_epsilon
        bilinear_val[valid_denom_mask] = (v00 * wa + v01 * wb + v10 * wc + v11 * wd)[
            valid_denom_mask] / safe_denominator[valid_denom_mask]
        # Handle cases where denominator was zero but shouldn't have been (fallback)
        bilinear_val = torch.where(torch.isfinite(
            bilinear_val), bilinear_val, v00)  # Replace NaN with v00

        # Combine results using torch.where
        final_value = torch.where(
            is_t_degenerate & is_c_degenerate, v00,
            torch.where(is_t_degenerate, val_t_degenerate,
                        torch.where(is_c_degenerate, val_c_degenerate, bilinear_val))
        )

        # --- 8. Apply Mask and Handle Non-finite ---
        # Ensure results for invalid arcs (dim <= 0) are zero
        # Also handle any NaNs produced during calculation
        final_value = torch.where(valid_arc_mask & torch.isfinite(
            final_value), final_value, torch.tensor(0.0, device=device, dtype=dtype))

        return final_value

    # --- Vectorized LUT Entry Functions (Updated to call vectorized lut_entry) ---
    # These now directly call the vectorized function, no vmap needed here.
    def r_delay_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.r_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs,
            luts.flat_luts_values, luts.flat_luts_trans_table,
            luts.flat_luts_cap_table, luts.flat_luts_dim
        )

    def f_delay_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.f_delay_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs,
            luts.flat_luts_values, luts.flat_luts_trans_table,
            luts.flat_luts_cap_table, luts.flat_luts_dim
        )

    def r_tran_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.r_trans_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs,
            luts.flat_luts_values, luts.flat_luts_trans_table,
            luts.flat_luts_cap_table, luts.flat_luts_dim
        )

    def f_tran_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs):
        luts = self.arcs_info.f_trans_luts
        return self.lut_entry_vectorized(
            lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs,
            luts.flat_luts_values, luts.flat_luts_trans_table,
            luts.flat_luts_cap_table, luts.flat_luts_dim
        )

    def calculate_cell_aat_level(self, level_cells,
                                 pin_rAAT, pin_fAAT,
                                 pin_rtran, pin_ftran,
                                 pin_net_cap_rise, pin_net_cap_fall,
                                 cell_arc_r_delays, cell_arc_f_delays, is_clk2q=False, clk_pin_rtran=None, clk_pin_ftran=None):
        device = pin_rAAT.device

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

        if (arc_in_pins == 15754).any() and (arc_out_pins == 15771).any():
            logging.warning(f"lib_arc_idxs: {lib_arc_idxs}")

        # 1. 准备输入条件
        #    - Rise Slew/Load at Input for Rise Delay/Tran calculation
        #    - Fall Slew/Load at Input for Fall Delay/Tran calculation
        if is_clk2q:
            # clk2q only has one direction
            pin_r_slew_in = clk_pin_rtran[arc_in_pins]
            pin_f_slew_in = clk_pin_ftran[arc_in_pins]
        else:
            pin_r_slew_in = pin_rtran[arc_in_pins]
            pin_f_slew_in = pin_ftran[arc_in_pins]

        pin_r_load_out = pin_net_cap_rise[arc_out_pins]
        pin_f_load_out = pin_net_cap_fall[arc_out_pins]

        # --- 2. 计算四种基础情况的延时(Delay)和转换时间(Transition) ---

        # Rise -> Rise (r->r)
        delay_rr = self.r_delay_entry(
            lib_cell_idxs, pin_r_slew_in, pin_r_load_out, lib_arc_idxs)
        tran_rr = self.r_tran_entry(
            lib_cell_idxs, pin_r_slew_in, pin_r_load_out, lib_arc_idxs)

        # Fall -> Fall (f->f)
        delay_ff = self.f_delay_entry(
            lib_cell_idxs, pin_f_slew_in, pin_f_load_out, lib_arc_idxs)
        tran_ff = self.f_tran_entry(
            lib_cell_idxs, pin_f_slew_in, pin_f_load_out, lib_arc_idxs)

        # Rise -> Fall (r->f)
        delay_rf = self.f_delay_entry(
            lib_cell_idxs, pin_r_slew_in, pin_f_load_out, lib_arc_idxs)
        tran_rf = self.f_tran_entry(
            lib_cell_idxs, pin_r_slew_in, pin_f_load_out, lib_arc_idxs)

        # Fall -> Rise (f->r)
        delay_fr = self.r_delay_entry(
            lib_cell_idxs, pin_f_slew_in, pin_r_load_out, lib_arc_idxs)
        tran_fr = self.r_tran_entry(
            lib_cell_idxs, pin_f_slew_in, pin_r_load_out, lib_arc_idxs)

        # --- 3. 根据 Timing Sense 组合最终结果 ---
        # is_non_unate mask is implicitly `not (is_pos_unate or is_neg_unate)`
        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)

        # --- 计算最终的上升输出延时 (r_delays) ---
        # Positive unate : r->r / f->f
        # Negative unate : f->r / r->f
        # Non-unate : worst of r->r, f->r, f->f, r->f
        r_delays_non_unate = torch.maximum(delay_rr, delay_fr)
        f_delays_non_unate = torch.maximum(delay_ff, delay_rf)
        r_delays = torch.where(is_pos_unate, delay_rr,
                               torch.where(is_neg_unate, delay_fr, r_delays_non_unate))
        f_delays = torch.where(is_pos_unate, delay_ff,
                               torch.where(is_neg_unate, delay_rf, f_delays_non_unate))

        # --- 计算最终的上升输出转换时间 (r_trans) ---
        r_trans_non_unate = torch.maximum(tran_rr, tran_fr)
        r_trans = torch.where(is_pos_unate, tran_rr,
                              torch.where(is_neg_unate, tran_fr, r_trans_non_unate))

        # --- 计算最终的下降输出转换时间 (f_trans) ---
        f_trans_non_unate = torch.maximum(tran_ff, tran_rf)
        f_trans = torch.where(is_pos_unate, tran_ff,
                              torch.where(is_neg_unate, tran_rf, f_trans_non_unate))

        # --- 4. 存储和更新 (最终修正版) ---

        # ★★★ 核心修正: 获取当前批次中所有有效弧的、扁平化的全局索引 ★★★
        # 这个操作会生成一个一维张量，其大小 (5104) 与 r_delays 的大小完全匹配
        scatter_indices = arcs_global_indices[arcs_mask]

        # 使用这个正确的一维索引张量进行scatter操作
        cell_arc_r_delays.scatter_(0, scatter_indices.long(), r_delays)
        cell_arc_f_delays.scatter_(0, scatter_indices.long(), f_delays)

        # 后续的AAT和Slew更新逻辑保持不变
        r_aat_updates = pin_rAAT[arc_in_pins] + r_delays
        f_aat_updates = pin_fAAT[arc_in_pins] + f_delays

        pin_rAAT = torch.scatter_reduce(pin_rAAT, 0, arc_out_pins.long(
        ), r_aat_updates, reduce="amax", include_self=False)
        pin_fAAT = torch.scatter_reduce(pin_fAAT, 0, arc_out_pins.long(
        ), f_aat_updates, reduce="amax", include_self=False)
        pin_rtran = torch.scatter_reduce(
            pin_rtran, 0, arc_out_pins.long(), r_trans, reduce="amax", include_self=False)
        pin_ftran = torch.scatter_reduce(
            pin_ftran, 0, arc_out_pins.long(), f_trans, reduce="amax", include_self=False)

        net_in_pins = torch.unique(self.pin_net[arc_out_pins])

        return net_in_pins, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays

    def calculate_net_aat_level(self, curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                                pin_net_delay_rise, pin_net_delay_fall,
                                pin_net_impulse_rise, pin_net_impulse_fall):
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
        r_tran_updates = torch.sqrt(pin_rtran[arc_fipins]**2 + impulse_rise**2)
        f_tran_updates = torch.sqrt(pin_ftran[arc_fipins]**2 + impulse_fall**2)

        pin_rAAT = torch.scatter_reduce(pin_rAAT, 0, arc_fopins.long(
        ), r_aat_updates, reduce="amax", include_self=True)
        pin_fAAT = torch.scatter_reduce(pin_fAAT, 0, arc_fopins.long(
        ), f_aat_updates, reduce="amax", include_self=True)
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_fopins.long(
        ), r_tran_updates, reduce="amax", include_self=True)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_fopins.long(
        ), f_tran_updates, reduce="amax", include_self=True)

        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def calculate_cell_rat_level(self, level_cells, pin_rRAT, pin_fRAT, cell_arc_r_delays, cell_arc_f_delays):
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

        assert pin_rRAT[arc_out_pins].min(
        ) >= 0, "Negative r_rat_updates detected"
        r_rat_updates = pin_rRAT[arc_out_pins] - r_delays
        f_rat_updates = pin_fRAT[arc_out_pins] - f_delays

        pin_rRAT = torch.scatter_reduce(pin_rRAT, 0, arc_in_pins.long(
        ), r_rat_updates, reduce="amin", include_self=True)
        pin_fRAT = torch.scatter_reduce(pin_fRAT, 0, arc_in_pins.long(
        ), f_rat_updates, reduce="amin", include_self=True)

        cur_endpoints = torch.unique(arc_in_pins)
        return cur_endpoints, pin_rRAT, pin_fRAT

    def calculate_net_rat_level(self, cur_endpoint, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall):

        curnets = self.pin_net[cur_endpoint]

        arc_fipins = self.net2driver_pin_map[curnets]
        wire_delays_rise = pin_net_delay_rise[cur_endpoint]
        wire_delays_fall = pin_net_delay_fall[cur_endpoint]
        assert pin_rRAT[cur_endpoint].min(
        ) >= 0, "Negative r_rat_updates detected"
        r_rat_updates = pin_rRAT[cur_endpoint] - wire_delays_rise
        f_rat_updates = pin_fRAT[cur_endpoint] - wire_delays_fall
        pin_rRAT = torch.scatter_reduce(pin_rRAT, 0, arc_fipins.long(
        ), r_rat_updates, reduce="amin", include_self=True)
        pin_fRAT = torch.scatter_reduce(pin_fRAT, 0, arc_fipins.long(
        ), f_rat_updates, reduce="amin", include_self=True)
        assert pin_rRAT[arc_fipins].min(
        ) >= 0, "Negative r_rat_updates detected after scatter_reduce"
        return pin_rRAT, pin_fRAT

    def forward(self, pin_net_delays, pin_net_impulses, pin_net_caps):
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

        cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, cell_arc_r_delays, cell_arc_f_delays = self.calculate_cell_aat_level(
            self.FF_ids, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
            pin_net_cap_rise, pin_net_cap_fall,
            cell_arc_r_delays, cell_arc_f_delays, True, self.clk_pin_rtran, self.clk_pin_ftran)

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
        neg_slack = torch.clamp(slack, max=0)
        wns = torch.min(neg_slack)
        tns = torch.sum(neg_slack)

        self.pin_rAAT, self.pin_fAAT, self.pin_rRAT, self.pin_fRAT = (
            t.clone().detach() for t in [pin_rAAT, pin_fAAT, pin_rRAT, pin_fRAT])
        self.pin_rtran, self.pin_ftran = (
            t.clone().detach() for t in [pin_rtran, pin_ftran])
        self.pin_net_cap_rise, self.pin_net_cap_fall = (
            t.clone().detach() for t in [pin_net_cap_rise, pin_net_cap_fall])
        self.cell_arc_r_delays, self.cell_arc_f_delays = (
            t.clone().detach() for t in [cell_arc_r_delays, cell_arc_f_delays])

        return wns, tns
