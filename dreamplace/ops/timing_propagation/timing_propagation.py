#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : timing_propagation.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        :
@version      : 0.1
@date         : 2025-04-18 20:07:53
@copyright    : Copyright (c) 2024-2026 ICT CAS.
'''

import csv
import copy
import sys
import json
import os
import time
import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import logging
from torch.func import vmap
import unittest
from dataclasses import dataclass, field
from typing import Optional
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
from dreamplace.ops.timing_propagation.critical_endpoint_pruning import (
    DynamicCriticalEndpointSelector,
)
from dreamplace.ops.timing_propagation.endpoint_qualification import (
    endpoint_metrics, qualify_check_candidates, qualify_pin_slacks,
)

try:
    from dreamplace.ops.timing_propagation import lut_entry_2d_op
except ImportError:
    lut_entry_2d_op = None

def _load_timing_propagation_cpp():
    for module_name, module in sys.modules.items():
        if module_name.endswith(".timing_propagation_cpp"):
            return module
    try:
        # the compiled extension will be named timing_propagation_cpp and installed
        # into the same package (dreamplace.ops.timing_propagation)
        from . import timing_propagation_cpp as cpp_module  # type: ignore
        return cpp_module
    except Exception:
        return None


# Try to import compiled C++ extension (built by CMake via add_pytorch_extension).
_tp_cpp = _load_timing_propagation_cpp()
timing_propagation_cpp = _tp_cpp

# Smooth maximum (log-sum-exp) helpers
# Use a module-level alpha so it's easy to tune globally.
# Larger alpha -> closer to hard max; smaller alpha -> smoother
SMOOTH_MAX_ALPHA = 20.0
SMOOTH_MAX_ENABLED = True
TIMING_CHECK_CLASS_RECOVERY = 3
# Slew propagation updates a transition by sqrt(tran^2 + impulse).  Pins whose
# transition is still exactly zero (no lib default, no driver yet) make the
# radicand zero, and sqrt's backward pass is 0.5/sqrt(x) = inf, which poisons
# the whole timing gradient.  Clamp the radicand: the forward value moves by
# sqrt(eps) ~ 1e-4 ps (negligible against lib slews) and clamp_min's zero
# derivative at the boundary keeps those arcs out of the gradient instead of
# injecting an infinite one.
SLEW_SQRT_EPS = 1e-8


def _force_hard_pairwise_max_enabled() -> bool:
    value = os.environ.get("AIMP_FORCE_HARD_PAIRWISE_MAX", "")
    return str(value).strip().lower() not in ("", "0", "false", "off", "no")

PIN_VIOLATION_DETAIL_FIELDNAMES = (
    "pin_name",
    "check_type",
    "is_output_pin",
    "limit",
    "rise_value",
    "fall_value",
    "rise_violation",
    "fall_violation",
    "total_violation",
    "unit",
)


def _normalize_pin_name(pin_name) -> str:
    if isinstance(pin_name, bytes):
        return pin_name.decode("utf-8", errors="replace")
    return str(pin_name)


def build_pin_violation_detail_rows(
    pin_names,
    pin_mask,
    rise_values,
    fall_values,
    limits,
    check_type: str,
    unit: str,
    output_pin_mask=None,
):
    pin_mask_t = torch.as_tensor(pin_mask, dtype=torch.bool).detach().cpu()
    rise_values_t = torch.as_tensor(rise_values, dtype=torch.float32).detach().cpu()
    fall_values_t = torch.as_tensor(fall_values, dtype=torch.float32).detach().cpu()
    limits_t = torch.as_tensor(limits, dtype=torch.float32).detach().cpu()
    if output_pin_mask is None:
        output_pin_mask_t = torch.zeros_like(pin_mask_t, dtype=torch.bool)
    else:
        output_pin_mask_t = torch.as_tensor(output_pin_mask, dtype=torch.bool).detach().cpu()

    expected_len = min(
        len(pin_names),
        pin_mask_t.numel(),
        rise_values_t.numel(),
        fall_values_t.numel(),
        limits_t.numel(),
        output_pin_mask_t.numel(),
    )
    if expected_len <= 0:
        return []

    valid_mask = (
        pin_mask_t[:expected_len]
        & torch.isfinite(limits_t[:expected_len])
        & (limits_t[:expected_len] > 0)
    )
    rows = []
    for pin_idx in torch.where(valid_mask)[0].tolist():
        limit_value = float(limits_t[pin_idx].item())
        rise_value = float(rise_values_t[pin_idx].item())
        fall_value = float(fall_values_t[pin_idx].item())
        rise_violation = max(0.0, rise_value - limit_value)
        fall_violation = max(0.0, fall_value - limit_value)
        rows.append(
            {
                "pin_name": _normalize_pin_name(pin_names[pin_idx]),
                "check_type": check_type,
                "is_output_pin": int(bool(output_pin_mask_t[pin_idx].item())),
                "limit": limit_value,
                "rise_value": rise_value,
                "fall_value": fall_value,
                "rise_violation": rise_violation,
                "fall_violation": fall_violation,
                "total_violation": rise_violation + fall_violation,
                "unit": unit,
            }
        )
    return rows


def write_pin_violation_detail_csv(output_path, rows) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PIN_VIOLATION_DETAIL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_smooth_max_enabled(enabled: bool) -> None:
    """全局开关，可控制是否使用平滑最大值。"""
    global SMOOTH_MAX_ENABLED
    SMOOTH_MAX_ENABLED = bool(enabled)

def smooth_max(a: torch.Tensor, b: torch.Tensor, alpha: float = SMOOTH_MAX_ALPHA) -> torch.Tensor:
    """
    Pairwise smooth maximum using log-sum-exp:
        smooth_max(a, b) = (1/alpha) * log( exp(alpha * a) + exp(alpha * b) )
    This is numerically stable when implemented with torch.logsumexp over a stacked tensor.
    Works for tensors of identical shape and broadcasts like torch.maximum.
    """
    if not SMOOTH_MAX_ENABLED or _force_hard_pairwise_max_enabled():
        return torch.maximum(a, b)
    # stack on a new dim and reduce
    return torch.logsumexp(torch.stack([alpha * a, alpha * b], dim=0), dim=0) / alpha

def smooth_max_reduce(x: torch.Tensor, dim: int = None, alpha: float = SMOOTH_MAX_ALPHA, keepdim: bool = False) -> torch.Tensor:
    """
    Generalized smooth max over a dimension (log-sum-exp reduction).
    If dim is None, reduce over all elements.
    """
    if not SMOOTH_MAX_ENABLED:
        if dim is None:
            result = torch.amax(x)
            return result.reshape(1, 1) if keepdim else result.reshape(1)
        return torch.amax(x, dim=dim, keepdim=keepdim)
    if dim is None:
        x_flat = x.view(1, -1)
        return torch.logsumexp(alpha * x_flat, dim=1, keepdim=keepdim) / alpha
    return torch.logsumexp(alpha * x, dim=dim, keepdim=keepdim) / alpha


def smooth_scatter_max(dest: torch.Tensor, index: torch.Tensor, src: torch.Tensor, include_self: bool = True, alpha: float = SMOOTH_MAX_ALPHA, eps: float = 1e-30) -> torch.Tensor:
    """
    Vectorized scatter-reduce that performs a smooth log-sum-exp (soft max) over contributions
    for each destination index.

    dest: target tensor (1D) whose shape defines the number of destinations (N)
    index: long tensor of shape (M,) providing destination indices for each src element
    src: tensor of shape (M,) with values to reduce
    include_self: whether to include existing values from dest in the reduction

    Implementation outline (vectorized):
    1) Optionally append the dest values as additional contributions (to include_self).
    2) Compute per-index max using scatter_reduce(..., reduce='amax') for numerical stability.
    3) Compute exp(alpha*(vals - max[idx])) per contribution and scatter_add to accumulate per-index sums.
    4) result = max + log(sum_exp) ; return result / alpha
    """
    if not SMOOTH_MAX_ENABLED:
        result = dest.clone()
        if not include_self:
            result.fill_(float('-inf'))
        return torch.scatter_reduce(result, 0, index.long(), src, reduce="amax", include_self=include_self)
    alpha_value = float(alpha)
    if alpha_value <= 0.0:
        raise ValueError("alpha must be positive for smooth scatter max")
    return smooth_scatter_max_tau(
        dest,
        index,
        src,
        include_self=include_self,
        tau=1.0 / alpha_value,
        eps=eps,
    )


def smooth_scatter_max_tau(
    dest: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    include_self: bool = True,
    tau: float = 1.0,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Scatter log-sum-exp max with timing-unit temperature tau."""
    tau_value = float(tau)
    if tau_value <= 0.0:
        raise ValueError("tau must be positive for smooth scatter max")

    idx = index.long()
    vals = src
    target_size = dest.shape[0]
    device = dest.device
    dtype = dest.dtype

    if include_self:
        idx = torch.cat([idx, torch.arange(target_size, device=device)], dim=0)
        vals = torch.cat([vals, dest], dim=0)

    max_per_index = torch.full((target_size,), -torch.inf, device=device, dtype=dtype)
    max_per_index = torch.scatter_reduce(
        max_per_index,
        0,
        idx,
        vals,
        reduce="amax",
        include_self=False,
    )
    stable_max = max_per_index.detach()
    finite_group = torch.isfinite(stable_max)
    finite_vals = finite_group[idx]
    safe_group_max = torch.where(
        finite_group,
        stable_max,
        torch.zeros_like(stable_max),
    )
    scaled = (vals - safe_group_max[idx]) / tau_value
    scaled = torch.where(finite_vals, scaled, torch.full_like(scaled, -torch.inf))
    exp_scaled = torch.exp(scaled)
    sum_exp = torch.zeros_like(max_per_index).scatter_add_(0, idx, exp_scaled)
    lse = safe_group_max + tau_value * torch.log(sum_exp.clamp(min=eps))
    reduced = torch.where(finite_group, lse, max_per_index)
    if include_self:
        return reduced
    touched = torch.zeros_like(dest, dtype=torch.bool).scatter_(0, idx, True)
    return torch.where(touched, reduced, dest)


def smooth_scatter_min_tau(
    dest: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    include_self: bool = True,
    tau: float = 1.0,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Scatter smooth min implemented as negative smooth max of negatives."""
    return -smooth_scatter_max_tau(
        -dest,
        index,
        -src,
        include_self=include_self,
        tau=tau,
        eps=eps,
    )


def scatter_timing_max(
    dest: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    include_self: bool = True,
    mode: str = "hard",
    tau_ps: float = 1.0,
) -> torch.Tensor:
    if str(mode) == "smooth":
        return smooth_scatter_max_tau(
            dest,
            index,
            src,
            include_self=include_self,
            tau=tau_ps,
        )
    return torch.scatter_reduce(
        dest,
        0,
        index.long(),
        src,
        reduce="amax",
        include_self=include_self,
    )


def scatter_timing_min(
    dest: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    include_self: bool = True,
    mode: str = "hard",
    tau_ps: float = 1.0,
) -> torch.Tensor:
    if str(mode) == "smooth":
        return smooth_scatter_min_tau(
            dest,
            index,
            src,
            include_self=include_self,
            tau=tau_ps,
        )
    return torch.scatter_reduce(
        dest,
        0,
        index.long(),
        src,
        reduce="amin",
        include_self=include_self,
    )
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
        # A LUT is valid as long as it has at least one timing axis.
        # 2D tables have both dims > 0, while 1D tables intentionally keep one dim at 0.
        # Only the [0, 0] placeholder means "missing table".
        self.valid_arc_mask = (self.trans_dims_actual > 0) | (self.cap_dims_actual > 0)
        self.is_scalar = (self.trans_dims_actual <= 1) & (self.cap_dims_actual <= 1)
        self.is_trans_1d = (self.trans_dims_actual > 1) & (self.cap_dims_actual <= 1)
        self.is_cap_1d = (self.trans_dims_actual <= 1) & (self.cap_dims_actual > 1)
        self.is_2d = (self.trans_dims_actual > 1) & (self.cap_dims_actual > 1)
        self.has_2d_coeff_cache = False
        self.coeff_a = None
        self.coeff_b = None
        self.coeff_c = None
        self.coeff_d = None
        self.coeff_valid = None
        self.coeff_table = None
        self.has_full_2d_coeff_cache = False
        self.use_2d_coeff_cache = False
        self.use_2d_coeff_op = False
        self.log_stage_progress = True
        self.log_violation_shape_mismatch = True
        self._build_2d_coeff_cache()

    def _build_2d_coeff_cache(self):
        if self.flat_luts_values.numel() == 0:
            return
        if self.flat_luts_trans_table.dim() != 2 or self.flat_luts_cap_table.dim() != 2:
            return
        if self.flat_luts_trans_table.shape[1] < 2 or self.flat_luts_cap_table.shape[1] < 2:
            return

        device = self.flat_luts_values.device
        dtype = self.flat_luts_values.dtype
        num_luts = self.flat_luts_values.shape[0]
        trans_cells = self.flat_luts_trans_table.shape[1] - 1
        cap_cells = self.flat_luts_cap_table.shape[1] - 1
        flat_values = self.flat_luts_values.reshape(num_luts, -1)

        t0 = self.flat_luts_trans_table[:, :-1].unsqueeze(2)
        t1 = self.flat_luts_trans_table[:, 1:].unsqueeze(2)
        c0 = self.flat_luts_cap_table[:, :-1].unsqueeze(1)
        c1 = self.flat_luts_cap_table[:, 1:].unsqueeze(1)

        trans_offsets = torch.arange(trans_cells, device=device).view(1, trans_cells, 1)
        cap_offsets = torch.arange(cap_cells, device=device).view(1, 1, cap_cells)
        stride = self.cap_dims_actual.view(num_luts, 1, 1)
        idx00 = trans_offsets * stride + cap_offsets
        idx01 = idx00 + 1
        idx10 = idx00 + stride
        idx11 = idx10 + 1

        max_flat_idx = flat_values.shape[1] - 1
        v00 = flat_values.gather(1, idx00.clamp(min=0, max=max_flat_idx).reshape(num_luts, -1)).reshape(num_luts, trans_cells, cap_cells)
        v01 = flat_values.gather(1, idx01.clamp(min=0, max=max_flat_idx).reshape(num_luts, -1)).reshape(num_luts, trans_cells, cap_cells)
        v10 = flat_values.gather(1, idx10.clamp(min=0, max=max_flat_idx).reshape(num_luts, -1)).reshape(num_luts, trans_cells, cap_cells)
        v11 = flat_values.gather(1, idx11.clamp(min=0, max=max_flat_idx).reshape(num_luts, -1)).reshape(num_luts, trans_cells, cap_cells)

        trans_idx = torch.arange(trans_cells, device=device).view(1, trans_cells, 1)
        cap_idx = torch.arange(cap_cells, device=device).view(1, 1, cap_cells)
        required_cell = (
            self.is_2d.view(num_luts, 1, 1)
            & (trans_idx < (self.trans_dims_actual - 1).view(num_luts, 1, 1))
            & (cap_idx < (self.cap_dims_actual - 1).view(num_luts, 1, 1))
        )
        valid_cell = (
            required_cell
            & torch.isfinite(t0)
            & torch.isfinite(t1)
            & torch.isfinite(c0)
            & torch.isfinite(c1)
            & torch.isfinite(v00)
            & torch.isfinite(v01)
            & torch.isfinite(v10)
            & torch.isfinite(v11)
        )
        denom = (t1 - t0) * (c1 - c0)
        eps = torch.tensor(1e-12, device=device, dtype=dtype)
        valid_cell = valid_cell & (torch.abs(denom) >= eps)
        safe_denom = torch.where(valid_cell, denom, torch.ones_like(denom))

        coeff_a = (v00 - v01 - v10 + v11) / safe_denom
        coeff_b = (-v00 * c1 + v01 * c0 + v10 * c1 - v11 * c0) / safe_denom
        coeff_c = (-v00 * t1 + v01 * t1 + v10 * t0 - v11 * t0) / safe_denom
        coeff_d = (v00 * t1 * c1 - v01 * t1 * c0 - v10 * t0 * c1 + v11 * t0 * c0) / safe_denom

        self.coeff_a = torch.where(valid_cell, coeff_a, torch.zeros_like(coeff_a))
        self.coeff_b = torch.where(valid_cell, coeff_b, torch.zeros_like(coeff_b))
        self.coeff_c = torch.where(valid_cell, coeff_c, torch.zeros_like(coeff_c))
        self.coeff_d = torch.where(valid_cell, coeff_d, torch.zeros_like(coeff_d))
        self.coeff_valid = valid_cell
        self.has_2d_coeff_cache = bool(torch.any(valid_cell).detach().cpu().item())
        self.has_full_2d_coeff_cache = bool(torch.all((~required_cell) | valid_cell).detach().cpu().item())
        coeff_op_requested = (
            os.environ.get("AIMP_USE_LUT_2D_COEFF_OP", "").lower()
            in ("1", "true", "yes", "on")
        )
        if lut_entry_2d_op is not None and self.has_2d_coeff_cache and coeff_op_requested:
            self.coeff_table = lut_entry_2d_op.build_2d_coefficients(
                self.flat_luts_trans_table,
                self.flat_luts_cap_table,
                flat_values,
                self.trans_dims_actual,
                self.cap_dims_actual,
            )
        self.use_2d_coeff_cache = (
            os.environ.get("AIMP_USE_LUT_2D_COEFF_CACHE", "").lower()
            in ("1", "true", "yes", "on")
            and self.has_2d_coeff_cache
            and self.has_full_2d_coeff_cache
        )
        self.use_2d_coeff_op = (
            coeff_op_requested
            and self.coeff_table is not None
            and self.has_full_2d_coeff_cache
        )

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


def _resolve_native_net_subgraph_timing_device(*, timing_device, requested_device=None):
    if requested_device is not None:
        return torch.device(requested_device)
    return torch.device("cpu")


def _move_dynamic_net_arc_inputs_to_device(dynamic_net_arc_inputs, device):
    if dynamic_net_arc_inputs is None:
        return None
    device = torch.device(device)

    def move_value(value):
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        return value

    return {key: move_value(value) for key, value in dynamic_net_arc_inputs.items()}


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
                 flat_inst_arcs_by_level,
                 flat_inst_arcs_by_level_start, # level 0 is clk->Q
                 endpoints_rRAT,
                 endpoints_fRAT,
                 flat_pin_to_graph,
                 flat_pin_to_graph_start,
                 flat_pin_to_graph_reverse,
                 flat_pin_to_graph_start_reverse,
                 endpoints_timing_check_arcs=None,
                 pin_pair_arc_keys=None,
                 flat_pin_pair_arc_start=None,
                 flat_pin_pair_arc_indices=None,
                 pin_pred_start=None,
                 pin_pred_pin=None,
                 pin_pred_arc_id=None,
                 pin_succ_start=None,
                 pin_succ_pin=None,
                 pin_succ_arc_id=None,
                 arc_level_start=None,
                 arc_src_pin=None,
                 arc_dst_pin=None,
                 arc_inst_id=None,
                 endpoint_pin_ids=None,
                 start_pin_ids=None,
                 pin_to_inst_id=None,
                 pin_to_node_id=None,
                 inst_topo_start=None,
                 inst_topo_ids=None,
                 cell_modeling_op=None,
                 lib_arc_offsets=None,
                 pin2node_map=None,
                 inst_main_id=None,
                 inst_is_sizeable=None,
                 inst_size_init=None,
                 inst_vt_init=None,
                 size_var_getter=None,
                 vt_var_getter=None,
                 critical_endpoint_pruning_mode="off",
                 critical_endpoint_top_k=256,
                 critical_endpoint_slack_window_ps=100.0,
                 critical_endpoint_refresh_interval=10,
                 critical_endpoint_hysteresis_interval=3,
                 critical_endpoint_full_refresh_interval=50,
                 timing_propagation_profile=False,
                 timing_propagation_device="inherit",
                 timing_propagation_global_gpu=0,
                 timing_propagation_global_gpu_id=0,
                 timing_propagation_parity_check=False,
                 timing_propagation_parity_atol_ps=1e-3,
                 timing_propagation_parity_rtol=1e-5,
                 timing_aggregation_mode="smooth",
                 timing_aggregation_tau_ps=2.0,
                 production_fast_loop=False,
                 timing_lut_2d_native_op="auto",
                 endpoints_max_valid=None,
                 endpoints_constraint_max_valid=None,
                 endpoints_timing_check_max_valid=None,
                 ):
        super(TimingPropagation, self).__init__()
        write_crash_stage_marker(
            "timing_propagation_init",
            "start",
            timing_propagation_device=timing_propagation_device,
            timing_propagation_global_gpu=timing_propagation_global_gpu,
            timing_propagation_global_gpu_id=timing_propagation_global_gpu_id,
        )

        self.num_pins = pin_net.shape[0]
        # TimingPropagation keeps the Python timing state on the same exported
        # basis as PyPlaceDB:
        #   - delay / AAT / RAT / slew / slack are in ps
        #   - pin load / cap limits are in pF
        #   - graph indices in this module are design-pin ids unless a specific
        #     RC tensor is documented as including an appended Steiner suffix.
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
        self.endpoints_timing_check_arcs = endpoints_timing_check_arcs
        self.endpoints_max_valid = endpoints_max_valid
        self.endpoints_constraint_max_valid = endpoints_constraint_max_valid
        self.endpoints_timing_check_max_valid = endpoints_timing_check_max_valid
        self.endpoint_max_valid_by_pin = None
        if endpoints_max_valid is not None:
            self.endpoint_max_valid_by_pin = torch.zeros((self.num_pins, 2), dtype=torch.bool, device=end_points.device)
            self.endpoint_max_valid_by_pin[end_points.long()] = endpoints_max_valid
        self.flat_inst_arcs_by_level = flat_inst_arcs_by_level
        self.flat_inst_arcs_by_level_start = flat_inst_arcs_by_level_start
        self.arcs_info = arcs_info
        self.dtype = inrdelays.dtype  # Use dtype from inputs

        self.endpoints_rRAT = endpoints_rRAT
        self.endpoints_fRAT = endpoints_fRAT

        self.flat_pin_to_graph = flat_pin_to_graph
        self.flat_pin_to_graph_start = flat_pin_to_graph_start
        self.flat_pin_to_graph_reverse = flat_pin_to_graph_reverse
        self.flat_pin_to_graph_start_reverse = flat_pin_to_graph_start_reverse
        self.pin_pair_arc_keys = pin_pair_arc_keys
        self.flat_pin_pair_arc_start = flat_pin_pair_arc_start
        self.flat_pin_pair_arc_indices = flat_pin_pair_arc_indices
        self.pin_pred_start = pin_pred_start
        self.pin_pred_pin = pin_pred_pin
        self.pin_pred_arc_id = pin_pred_arc_id
        self.pin_succ_start = pin_succ_start
        self.pin_succ_pin = pin_succ_pin
        self.pin_succ_arc_id = pin_succ_arc_id
        self.arc_level_start = arc_level_start
        self.arc_src_pin = arc_src_pin
        self.arc_dst_pin = arc_dst_pin
        self.arc_inst_id = arc_inst_id
        self.endpoint_pin_ids = endpoint_pin_ids
        self.start_pin_ids = start_pin_ids
        self.pin_to_inst_id = pin_to_inst_id
        self.pin_to_node_id = pin_to_node_id
        self.inst_topo_start = inst_topo_start
        self.inst_topo_ids = inst_topo_ids
        self.cell_modeling_op = cell_modeling_op
        self.lib_arc_offsets = lib_arc_offsets
        self.pin2node_map = pin2node_map
        self.inst_main_id = inst_main_id
        self.inst_is_sizeable = inst_is_sizeable
        self.inst_size_init = inst_size_init
        self.inst_vt_init = inst_vt_init
        self.size_var_getter = size_var_getter
        self.vt_var_getter = vt_var_getter
        self.critical_endpoint_pruning_mode = str(critical_endpoint_pruning_mode)
        self.critical_endpoint_top_k = int(critical_endpoint_top_k)
        self.critical_endpoint_slack_window_ps = float(critical_endpoint_slack_window_ps)
        self.critical_endpoint_refresh_interval = int(critical_endpoint_refresh_interval)
        self.critical_endpoint_hysteresis_interval = int(critical_endpoint_hysteresis_interval)
        self.critical_endpoint_full_refresh_interval = int(critical_endpoint_full_refresh_interval)
        self.critical_endpoint_selector = None
        self.timing_forward_count = 0
        self.timing_propagation_profile = bool(timing_propagation_profile)
        self.timing_propagation_device = str(timing_propagation_device or "inherit")
        self.timing_propagation_global_gpu = int(timing_propagation_global_gpu or 0)
        self.timing_propagation_global_gpu_id = int(timing_propagation_global_gpu_id or 0)
        self.timing_propagation_parity_check = bool(timing_propagation_parity_check)
        self.timing_propagation_parity_atol_ps = float(timing_propagation_parity_atol_ps)
        self.timing_propagation_parity_rtol = float(timing_propagation_parity_rtol)
        self.timing_aggregation_mode = str(timing_aggregation_mode or "smooth")
        if self.timing_aggregation_mode not in ("hard", "smooth"):
            raise ValueError("timing_aggregation_mode must be one of: hard, smooth")
        self.timing_aggregation_tau_ps = float(timing_aggregation_tau_ps)
        if self.timing_aggregation_tau_ps <= 0.0:
            raise ValueError("timing_aggregation_tau_ps must be positive")
        self.production_fast_loop = bool(production_fast_loop)
        self.timing_lut_2d_native_op = self._resolve_timing_lut_2d_native_op_mode(
            timing_lut_2d_native_op
        )
        self.use_cell_aat_static_cache = (
            os.environ.get("AIMP_USE_CELL_AAT_STATIC_CACHE", "").lower()
            in ("1", "true", "yes", "on")
        )
        self.resolved_timing_propagation_device = self._resolve_timing_device()
        self.device = self.resolved_timing_propagation_device
        logging.info("TIMING_AGGREGATION_EFFECTIVE %s", json.dumps({
            "mode": self.timing_aggregation_mode,
            "tau_ps": self.timing_aggregation_tau_ps,
            "device": str(self.device),
        }))
        write_crash_stage_marker(
            "timing_propagation_materialize_device_tensors",
            "start",
            resolved_device=str(self.device),
        )
        self._materialize_timing_device_tensors()
        write_crash_stage_marker(
            "timing_propagation_materialize_device_tensors",
            "done",
            resolved_device=str(self.device),
        )
        self.last_device_contract = self._build_device_contract(
            {
                "inrdelays": self.inrdelays,
                "infdelays": self.infdelays,
                "pin_net": self.pin_net,
                "start_points": self.start_points,
                "end_points": self.end_points,
                "flat_inst_arcs_by_level": self.flat_inst_arcs_by_level,
                "flat_inst_arcs_by_level_start": self.flat_inst_arcs_by_level_start,
            },
            scope="init",
        )
        self._assert_device_contract(self.last_device_contract)
        write_crash_stage_marker(
            "timing_propagation_init",
            "done",
            resolved_device=str(self.device),
            device_contract_status=self.last_device_contract.get("status"),
        )
        self.last_profile_payload = None
        self.last_parity_payload = self._build_parity_payload(
            enabled=self.timing_propagation_parity_check,
            checked=False,
            reason="not_run",
        )
        
        self.pin_rAAT = None
        self.pin_fAAT = None
        self.pin_rRAT = None
        self.pin_fRAT = None
        self.pin_rtran = None
        self.pin_ftran = None
        self.pin_net_cap = None
        self.pin_rtran_live = None
        self.pin_ftran_live = None
        self.pin_net_cap_rise_live = None
        self.pin_net_cap_fall_live = None
        self.pin_rAAT_live_snapshot = None
        self.pin_fAAT_live_snapshot = None
        self.pin_rRAT_live_snapshot = None
        self.pin_fRAT_live_snapshot = None
        self.pin_rslack_live_snapshot = None
        self.pin_fslack_live_snapshot = None
        self.pin_slack_live_snapshot = None
        self.last_endpoint_slack_tensor = None
        self.last_endpoint_ids_tensor = None
        self.last_active_endpoint_ids = []
        self._cell_aat_static_cache = {}
        self._cell_aat_static_cache_hits = 0
        self._cell_aat_static_cache_misses = 0
        self.last_critical_endpoint_pruning_stats = {
            "mode": self.critical_endpoint_pruning_mode,
            "applied": False,
            "iteration": None,
        }
        self.traversal_pruning_refresh_interval = max(
            1,
            int(self.critical_endpoint_refresh_interval or 10),
        )
        self._traversal_pruner = None
        self._traversal_pruner_topology_source = None
        self.last_traversal_pruning_result = None
        self.last_traversal_pruning_iteration = None
        self._dynamic_net_level_view_epoch = 0
        self._dynamic_net_level_view_signature = None
        self.last_traversal_pruning_stats = self._empty_traversal_pruning_stats(
            iteration=None,
            reason="not_run",
        )
        self._surrogate_state_cache = None
        self._surrogate_runtime_state_active = False
        self._active_cell_aat_profile = None
        self._surrogate_support_cache_status = "unknown"
        self.last_fast_loop_skipped_full_rat = False
        self._critical_path_snapshot_requested = False
        self._critical_path_snapshot = None
        self._critical_path_constraint_state = None
        self._setup_critical_path_extractor = None
        self._setup_critical_path_extractor_epoch = None
        self.critical_path_topology_epoch = 0
        self.last_critical_path_extraction_stats = None

    def _skip_full_rat_propagation_in_fast_loop(self):
        return bool(getattr(self, "production_fast_loop", False))

    def request_critical_path_snapshot(self):
        if self.timing_aggregation_mode != "hard":
            raise RuntimeError(
                "critical path extraction requires hard timing aggregation"
            )
        if getattr(self, "_critical_path_snapshot_requested", False) or getattr(
            self, "_critical_path_snapshot", None
        ) is not None:
            raise RuntimeError("critical path snapshot request is already active")
        self._critical_path_snapshot_requested = True
        self._critical_path_constraint_state = None

    def clear_critical_path_snapshot_request(self):
        self._critical_path_snapshot_requested = False
        self._critical_path_snapshot = None
        self._critical_path_constraint_state = None

    def consume_critical_path_snapshot(self):
        if getattr(self, "_critical_path_snapshot", None) is None:
            raise RuntimeError("critical path snapshot has not been captured")
        snapshot = self._critical_path_snapshot
        self._critical_path_snapshot = None
        self._critical_path_snapshot_requested = False
        self._critical_path_constraint_state = None
        return snapshot

    def invalidate_critical_path_topology(self):
        self.critical_path_topology_epoch += 1
        self._setup_critical_path_extractor = None
        self._setup_critical_path_extractor_epoch = None
        self.clear_critical_path_snapshot_request()

    def _capture_critical_path_snapshot(
        self,
        *,
        pin_rAAT,
        pin_fAAT,
        pin_rRAT,
        pin_fRAT,
        pin_net_delay_rise,
        pin_net_delay_fall,
        cell_arc_rr_delays,
        cell_arc_fr_delays,
        cell_arc_rf_delays,
        cell_arc_ff_delays,
    ):
        if not getattr(self, "_critical_path_snapshot_requested", False):
            return
        if getattr(self, "_critical_path_snapshot", None) is not None:
            raise RuntimeError("critical path snapshot was captured more than once")
        cell_delays = (
            cell_arc_rr_delays,
            cell_arc_fr_delays,
            cell_arc_rf_delays,
            cell_arc_ff_delays,
        )
        if any(tensor is None for tensor in cell_delays):
            raise RuntimeError("critical path snapshot requires cell arc delay tensors")

        endpoint_ids = self.end_points.long()
        qualification = getattr(self, "endpoints_max_valid", None)
        if qualification is not None:
            endpoint_ids = endpoint_ids[qualification.any(dim=1)]
        constraint_state = getattr(self, "_critical_path_constraint_state", None)
        if constraint_state is None:
            endpoint_test_pins = endpoint_ids
            endpoint_test_ids = torch.full_like(endpoint_ids, -1)
            endpoint_rise_slack = pin_rRAT[endpoint_ids] - pin_rAAT[endpoint_ids]
            endpoint_fall_slack = pin_fRAT[endpoint_ids] - pin_fAAT[endpoint_ids]
        else:
            constraint_pins = constraint_state["endpoint_pins"].long()
            timing_check_arcs = getattr(self, "endpoints_timing_check_arcs", None)
            if (
                timing_check_arcs is not None
                and torch.is_tensor(timing_check_arcs)
                and timing_check_arcs.numel() > 0
            ):
                timing_check_pins = timing_check_arcs[:, 1].long()
            else:
                timing_check_pins = constraint_pins
            # Setup path selection keeps primary-output fallbacks, but must not
            # mix recovery/removal async-check endpoints into the setup view.
            fallback_pins = endpoint_ids[
                ~torch.isin(endpoint_ids, timing_check_pins)
            ]
            endpoint_test_pins = torch.cat((constraint_pins, fallback_pins))
            endpoint_test_ids = torch.cat(
                (
                    constraint_state["test_ids"].long(),
                    torch.full_like(fallback_pins, -1),
                )
            )
            endpoint_rise_slack = torch.cat(
                (
                    constraint_state["rise_rat"] - pin_rAAT[constraint_pins],
                    pin_rRAT[fallback_pins] - pin_rAAT[fallback_pins],
                )
            )
            endpoint_fall_slack = torch.cat(
                (
                    constraint_state["fall_rat"] - pin_fAAT[constraint_pins],
                    pin_fRAT[fallback_pins] - pin_fAAT[fallback_pins],
                )
            )

        endpoint_rise_slack, endpoint_fall_slack = qualify_pin_slacks(
            self, endpoint_test_pins, endpoint_rise_slack, endpoint_fall_slack,
        )

        # Recovery is a separate max-check production lane. It is never folded
        # into the setup-only fidelity state arrays above.
        recovery_endpoint_pins = torch.empty(
            0,
            dtype=endpoint_ids.dtype,
            device=endpoint_ids.device,
        )
        timing_check_arcs = getattr(self, "endpoints_timing_check_arcs", None)
        if (
            timing_check_arcs is not None
            and torch.is_tensor(timing_check_arcs)
            and timing_check_arcs.numel() > 0
            and timing_check_arcs.dim() == 2
            and timing_check_arcs.size(1) > 6
        ):
            recovery_mask = (
                timing_check_arcs[:, 6] == TIMING_CHECK_CLASS_RECOVERY
            )
            check_valid = getattr(self, "endpoints_timing_check_max_valid", None)
            if check_valid is not None:
                recovery_mask = recovery_mask & check_valid.any(dim=1)
            recovery_endpoint_pins = torch.unique(
                timing_check_arcs[recovery_mask, 1].to(endpoint_ids.dtype)
            )
            recovery_endpoint_pins = recovery_endpoint_pins[
                torch.isin(recovery_endpoint_pins, endpoint_ids)
                & ~torch.isin(recovery_endpoint_pins, endpoint_test_pins)
            ]
        recovery_rise_slack = (
            pin_rRAT[recovery_endpoint_pins]
            - pin_rAAT[recovery_endpoint_pins]
        )
        recovery_fall_slack = (
            pin_fRAT[recovery_endpoint_pins]
            - pin_fAAT[recovery_endpoint_pins]
        )
        recovery_rise_slack, recovery_fall_slack = qualify_pin_slacks(
            self, recovery_endpoint_pins, recovery_rise_slack, recovery_fall_slack,
        )
        if recovery_endpoint_pins.numel() > 0:
            rise_is_worse = recovery_rise_slack <= recovery_fall_slack
            inactive = torch.full_like(recovery_rise_slack, float("inf"))
            recovery_rise_slack = torch.where(
                rise_is_worse,
                recovery_rise_slack,
                inactive,
            )
            recovery_fall_slack = torch.where(
                rise_is_worse,
                inactive,
                recovery_fall_slack,
            )
        parts = (
            pin_rAAT,
            pin_fAAT,
            endpoint_rise_slack,
            endpoint_fall_slack,
            recovery_rise_slack,
            recovery_fall_slack,
            pin_net_delay_rise[: self.num_pins],
            pin_net_delay_fall[: self.num_pins],
            *cell_delays,
        )
        lengths = [int(tensor.numel()) for tensor in parts]
        transfer_started_at = time.perf_counter()
        packed = torch.cat(
            [tensor.detach().reshape(-1) for tensor in parts],
            dim=0,
        ).cpu().contiguous()
        transfer_ms = (time.perf_counter() - transfer_started_at) * 1000.0
        names = (
            "pin_rise_aat",
            "pin_fall_aat",
            "endpoint_rise_slack",
            "endpoint_fall_slack",
            "recovery_endpoint_rise_slack",
            "recovery_endpoint_fall_slack",
            "pin_net_delay_rise",
            "pin_net_delay_fall",
            "cell_delay_rr",
            "cell_delay_fr",
            "cell_delay_rf",
            "cell_delay_ff",
        )
        snapshot = {
            "packed": packed,
            "endpoint_pins": endpoint_test_pins.detach().cpu().to(torch.int32).contiguous(),
            "endpoint_test_ids": endpoint_test_ids.detach().cpu().to(torch.int64).contiguous(),
            "recovery_endpoint_pins": recovery_endpoint_pins.detach()
            .cpu()
            .to(torch.int32)
            .contiguous(),
            "topology_epoch": int(self.critical_path_topology_epoch),
            "aggregation_mode": self.timing_aggregation_mode,
            "transfer_ms": transfer_ms,
            "bytes": int(packed.numel() * packed.element_size()),
        }
        offset = 0
        for name, length in zip(names, lengths):
            snapshot[name] = packed.narrow(0, offset, length)
            offset += length
        self._critical_path_snapshot = snapshot

    def _get_setup_critical_path_extractor(self):
        if _tp_cpp is None or not hasattr(_tp_cpp, "SetupCriticalPathExtractor"):
            raise RuntimeError(
                "native SetupCriticalPathExtractor is required for pin2pin production flow"
            )
        epoch = int(self.critical_path_topology_epoch)
        if (
            self._setup_critical_path_extractor is not None
            and self._setup_critical_path_extractor_epoch == epoch
        ):
            return self._setup_critical_path_extractor
        required = {
            "flat_inst_arcs_by_level": self.flat_inst_arcs_by_level,
            "pin_pred_start": self.pin_pred_start,
            "pin_pred_pin": self.pin_pred_pin,
            "pin_pred_arc_id": self.pin_pred_arc_id,
            "start_points": self.start_points,
        }
        missing = [
            name
            for name, value in required.items()
            if value is None or not torch.is_tensor(value) or value.numel() == 0
        ]
        if missing:
            raise RuntimeError(
                "critical path extractor missing compact topology tensors: "
                + ", ".join(missing)
            )
        self._setup_critical_path_extractor = _tp_cpp.SetupCriticalPathExtractor(
            required["flat_inst_arcs_by_level"].detach().cpu().to(torch.int32).contiguous(),
            required["pin_pred_start"].detach().cpu().to(torch.int32).contiguous(),
            required["pin_pred_pin"].detach().cpu().to(torch.int32).contiguous(),
            required["pin_pred_arc_id"].detach().cpu().to(torch.int32).contiguous(),
            required["start_points"].detach().cpu().to(torch.int32).contiguous(),
            epoch,
        )
        self._setup_critical_path_extractor_epoch = epoch
        return self._setup_critical_path_extractor

    def _extract_critical_paths_from_snapshot(
        self,
        snapshot,
        *,
        global_k,
        max_depth,
        residual_tolerance_ps,
        include_recovery,
    ):
        if snapshot["aggregation_mode"] != "hard":
            raise RuntimeError("critical path snapshot does not use hard aggregation")
        if snapshot["topology_epoch"] != int(self.critical_path_topology_epoch):
            raise RuntimeError("critical path snapshot topology epoch is stale")
        endpoint_pins = snapshot["endpoint_pins"]
        endpoint_test_ids = snapshot["endpoint_test_ids"]
        endpoint_rise_slack = snapshot["endpoint_rise_slack"]
        endpoint_fall_slack = snapshot["endpoint_fall_slack"]
        recovery_count = 0
        if include_recovery:
            recovery_pins = snapshot["recovery_endpoint_pins"]
            recovery_count = int(recovery_pins.numel())
            if recovery_count:
                endpoint_pins = torch.cat((endpoint_pins, recovery_pins))
                endpoint_test_ids = torch.cat(
                    (
                        endpoint_test_ids,
                        torch.full(
                            (recovery_count,),
                            -1,
                            dtype=endpoint_test_ids.dtype,
                        ),
                    )
                )
                endpoint_rise_slack = torch.cat(
                    (
                        endpoint_rise_slack,
                        snapshot["recovery_endpoint_rise_slack"],
                    )
                )
                endpoint_fall_slack = torch.cat(
                    (
                        endpoint_fall_slack,
                        snapshot["recovery_endpoint_fall_slack"],
                    )
                )
        extractor = self._get_setup_critical_path_extractor()
        batch = extractor.extract(
            endpoint_pins,
            endpoint_test_ids,
            endpoint_rise_slack,
            endpoint_fall_slack,
            snapshot["pin_rise_aat"],
            snapshot["pin_fall_aat"],
            snapshot["pin_net_delay_rise"],
            snapshot["pin_net_delay_fall"],
            snapshot["cell_delay_rr"],
            snapshot["cell_delay_fr"],
            snapshot["cell_delay_rf"],
            snapshot["cell_delay_ff"],
            int(global_k),
            int(max_depth),
            float(residual_tolerance_ps),
        )
        self.last_critical_path_extraction_stats = {
            "backend": "cpp_openmp_transition_aware",
            "topology_epoch": int(batch.topology_epoch),
            "failing_state_count": int(batch.failing_state_count),
            "selected_state_count": int(batch.selected_state_count),
            "valid_path_count": int(batch.valid_path_count),
            "invalid_path_count": int(batch.invalid_path_count),
            "max_residual_ps": (
                float(batch.max_residual_ps.max().item())
                if batch.max_residual_ps.numel()
                else 0.0
            ),
            "snapshot_transfer_ms": float(snapshot["transfer_ms"]),
            "snapshot_bytes": int(snapshot["bytes"]),
            "extraction_runtime_ms": float(batch.extraction_runtime_ms),
            "state_domain": (
                "setup_plus_recovery" if include_recovery else "setup_only"
            ),
            "recovery_endpoint_count": recovery_count,
        }
        return batch

    def extract_setup_critical_paths(
        self,
        global_k=0,
        max_depth=0,
        residual_tolerance_ps=1.0e-3,
    ):
        snapshot = self.consume_critical_path_snapshot()
        return self._extract_critical_paths_from_snapshot(
            snapshot,
            global_k=global_k,
            max_depth=max_depth,
            residual_tolerance_ps=residual_tolerance_ps,
            include_recovery=False,
        )

    def extract_pin2pin_critical_paths(
        self,
        global_k=0,
        max_depth=0,
        residual_tolerance_ps=1.0e-3,
    ):
        snapshot = self.consume_critical_path_snapshot()
        return self._extract_critical_paths_from_snapshot(
            snapshot,
            global_k=global_k,
            max_depth=max_depth,
            residual_tolerance_ps=residual_tolerance_ps,
            include_recovery=True,
        )

    def _resolve_timing_lut_2d_native_op_mode(self, value):
        mode = str(value or "auto").strip().lower()
        if mode in ("1", "true", "yes", "on", "enable", "enabled"):
            mode = "on"
        elif mode in ("0", "false", "no", "off", "disable", "disabled"):
            mode = "off"
        if mode not in ("auto", "on", "off"):
            raise ValueError(
                "timing_lut_2d_native_op must be one of: auto, on, off"
            )
        return mode

    def _use_lut_2d_native_op_for_tensor(self, tensor):
        native_mode = getattr(self, "timing_lut_2d_native_op", "auto")
        if native_mode == "off":
            return False
        if native_mode == "on":
            return True
        return bool(
            getattr(self, "production_fast_loop", False)
            and getattr(tensor, "is_cuda", False)
        )

    def _surrogate_slew_input_tensor(self, pin_slew):
        if not getattr(self, "production_fast_loop", False):
            return pin_slew
        if (
            os.environ.get("AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS", "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        ):
            return pin_slew
        return pin_slew.detach()

    def _surrogate_cap_input_tensor(self, pin_net_caps):
        if not getattr(self, "production_fast_loop", False):
            return pin_net_caps
        if (
            os.environ.get("AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS", "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        ):
            return pin_net_caps
        return pin_net_caps.detach()

    def _build_cell_aat_surrogate_support_cache(
        self,
        working_flat_arcs_by_level,
        working_flat_arc_level_start,
        profile=None,
    ):
        self._surrogate_support_cache_status = "unknown"
        if (
            not self._cell_modeling_enabled()
            or self.cell_modeling_op is None
            or not hasattr(self.cell_modeling_op, "supports_arc_types")
            or self.pin2node_map is None
            or self.inst_main_id is None
        ):
            return
        cache_timer = (
            self._sync_profile_clock(self.device)
            if isinstance(profile, dict)
            else None
        )
        try:
            level_count = int(len(working_flat_arc_level_start))
            if level_count <= 2:
                self._surrogate_support_cache_status = "all_supported"
                return
            starts = []
            ends = []
            for level in range(1, level_count - 1):
                start = working_flat_arc_level_start[level]
                end = working_flat_arc_level_start[level + 1]
                if torch.is_tensor(start):
                    start = int(start.detach().item())
                else:
                    start = int(start)
                if torch.is_tensor(end):
                    end = int(end.detach().item())
                else:
                    end = int(end)
                if start < end:
                    starts.append(start)
                    ends.append(end)
            if not starts:
                self._surrogate_support_cache_status = "all_supported"
                return
            arcs = torch.cat(
                [
                    working_flat_arcs_by_level[start:end]
                    for start, end in zip(starts, ends)
                ],
                dim=0,
            )
            arc_out_pins = arcs[:, 1]
            lib_arc_idxs = arcs[:, 3].long()
            timing_senses = arcs[:, 4]
            node_ids = self.pin2node_map[arc_out_pins.long()].long()
            main_ids = self.inst_main_id[node_ids].long()
            valid_main_id_mask = main_ids >= 0
            if self.inst_is_sizeable is None:
                sizeable_mask = torch.ones_like(valid_main_id_mask, dtype=torch.bool)
            else:
                sizeable_mask = self.inst_is_sizeable[node_ids].bool()
            fixed_non_sizeable_mask = torch.zeros_like(valid_main_id_mask, dtype=torch.bool)
            if self._fixed_non_sizeable_state_available():
                fixed_non_sizeable_mask = valid_main_id_mask & ~sizeable_mask
            valid_mask = valid_main_id_mask & (sizeable_mask | fixed_non_sizeable_mask)
            if profile is not None:
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_support_cache_entry_count",
                    int(arcs.shape[0]),
                )
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_support_cache_fixed_non_sizeable_count",
                    int(
                        torch.count_nonzero(fixed_non_sizeable_mask)
                        .detach()
                        .cpu()
                        .item()
                    ),
                )
            if not torch.any(valid_mask):
                self._surrogate_support_cache_status = "unsupported"
                return
            lib_arc_offsets = lib_arc_idxs
            if self.lib_arc_offsets is not None:
                lib_arc_offsets = self.lib_arc_offsets[lib_arc_offsets].long()
            pos_mask = timing_senses == 1
            neg_mask = timing_senses == -1
            non_mask = ~(pos_mask | neg_mask)
            support_inputs = []
            for mask, arc_types in (
                (pos_mask, (1, 3, 0, 2)),
                (neg_mask, (1, 3, 0, 2)),
                (non_mask, (1, 1, 3, 3, 0, 0, 2, 2)),
            ):
                local_mask = mask & valid_mask
                if not torch.any(local_mask):
                    continue
                local_main_ids = main_ids[local_mask]
                local_arc_offsets = lib_arc_offsets[local_mask]
                for arc_type in arc_types:
                    support_inputs.append(
                        (
                            local_main_ids,
                            local_arc_offsets,
                            torch.full_like(local_main_ids, int(arc_type)),
                        )
                    )
            if not support_inputs:
                self._surrogate_support_cache_status = "all_supported"
                return
            support_main_ids = torch.cat([item[0] for item in support_inputs], dim=0)
            support_arc_offsets = torch.cat([item[1] for item in support_inputs], dim=0)
            support_arc_types = torch.cat([item[2] for item in support_inputs], dim=0)
            support_mask = self.cell_modeling_op.supports_arc_types(
                support_main_ids,
                support_arc_offsets,
                support_arc_types,
            )
            support_missing_count = int(
                torch.count_nonzero(~support_mask).detach().cpu().item()
            )
            if profile is not None:
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_support_cache_support_missing_count",
                    support_missing_count,
                )
                if support_missing_count:
                    self._record_local_surrogate_unsupported_keys(
                        profile,
                        support_main_ids[~support_mask],
                        support_arc_offsets[~support_mask],
                        support_arc_types[~support_mask],
                    )
            self._surrogate_support_cache_status = (
                "all_supported" if support_missing_count == 0 else "unsupported"
            )
        finally:
            if isinstance(profile, dict):
                cache_done = self._sync_profile_clock(self.device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "surrogate_support_cache_ms",
                    cache_done - cache_timer,
                )
                profile["surrogate_support_cache_status"] = (
                    self._surrogate_support_cache_status
                )

    def _resolve_timing_device(self):
        mode = self.timing_propagation_device
        if mode not in ("inherit", "cpu", "cuda"):
            raise ValueError(
                "timing_propagation_device must be one of: inherit, cpu, cuda"
            )
        cuda_available = torch.cuda.is_available()
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "timing_propagation_device=cuda requested but CUDA is unavailable"
                )
            gpu_id = self._resolved_gpu_id_for_cuda()
            return torch.device(f"cuda:{gpu_id}")
        if self.timing_propagation_global_gpu and cuda_available:
            gpu_id = self._resolved_gpu_id_for_cuda()
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")

    def _resolved_gpu_id_for_cuda(self):
        gpu_id = max(0, int(self.timing_propagation_global_gpu_id))
        if not torch.cuda.is_available():
            return gpu_id
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA is reported available but no CUDA devices are visible")
        if gpu_id >= device_count:
            if self.timing_propagation_device == "cuda":
                raise RuntimeError(
                    "timing_propagation_device=cuda requested gpu_id "
                    f"{gpu_id}, but only {device_count} CUDA devices are available"
                )
            return 0
        return gpu_id

    @staticmethod
    def _tensor_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        return value

    @staticmethod
    def _move_luts_info_to_device(luts_info, device):
        if luts_info is None:
            return
        for name, value in list(vars(luts_info).items()):
            if torch.is_tensor(value):
                setattr(luts_info, name, value.to(device))

    def _materialize_timing_device_tensors(self):
        tensor_attrs = (
            "inrdelays",
            "infdelays",
            "inrtrans",
            "inftrans",
            "outcaps",
            "pin_net",
            "start_points",
            "end_points",
            "clock_pins",
            "FF_ids",
            "clk_pin_rtran",
            "clk_pin_ftran",
            "net_flat_arcs_start",
            "net_flat_arcs",
            "net2driver_pin_map",
            "inst_flat_arcs_start",
            "inst_flat_arcs",
            "endpoints_constraint_arcs",
            "endpoints_timing_check_arcs",
            "endpoints_max_valid",
            "endpoints_constraint_max_valid",
            "endpoints_timing_check_max_valid",
            "endpoint_max_valid_by_pin",
            "flat_inst_arcs_by_level",
            "flat_inst_arcs_by_level_start",
            "flat_pin_to_graph",
            "flat_pin_to_graph_start",
            "flat_pin_to_graph_reverse",
            "flat_pin_to_graph_start_reverse",
            "pin_pair_arc_keys",
            "flat_pin_pair_arc_start",
            "flat_pin_pair_arc_indices",
            "pin_pred_start",
            "pin_pred_pin",
            "pin_pred_arc_id",
            "pin_succ_start",
            "pin_succ_pin",
            "pin_succ_arc_id",
            "arc_level_start",
            "arc_src_pin",
            "arc_dst_pin",
            "arc_inst_id",
            "endpoint_pin_ids",
            "start_pin_ids",
            "pin_to_inst_id",
            "pin_to_node_id",
            "inst_topo_start",
            "inst_topo_ids",
            "lib_arc_offsets",
            "pin2node_map",
            "inst_main_id",
            "inst_is_sizeable",
            "inst_size_init",
            "inst_vt_init",
        )
        for attr_name in tensor_attrs:
            if hasattr(self, attr_name):
                setattr(
                    self,
                    attr_name,
                    self._tensor_to_device(getattr(self, attr_name), self.device),
                )
        self.endpoints_rRAT = torch.as_tensor(
            self.endpoints_rRAT,
            device=self.device,
            dtype=self.dtype,
        )
        self.endpoints_fRAT = torch.as_tensor(
            self.endpoints_fRAT,
            device=self.device,
            dtype=self.dtype,
        )
        if self.arcs_info is not None:
            for attr_name in (
                "f_delay_luts",
                "r_delay_luts",
                "f_trans_luts",
                "r_trans_luts",
            ):
                self._move_luts_info_to_device(
                    getattr(self.arcs_info, attr_name, None),
                    self.device,
                )
        if self.cell_modeling_op is not None and hasattr(self.cell_modeling_op, "device"):
            self.cell_modeling_op.device = self.device

    def _build_device_contract(self, tensors_by_name, scope):
        expected_device = torch.device(self.resolved_timing_propagation_device)
        mismatches = []
        checked_tensors = []
        checked = 0
        for name, value in tensors_by_name.items():
            if not torch.is_tensor(value):
                continue
            checked += 1
            device = value.device
            expected_index = expected_device.index
            actual_index = device.index
            if expected_device.type == "cuda":
                expected_index = 0 if expected_index is None else expected_index
                actual_index = 0 if actual_index is None else actual_index
            checked_tensors.append(
                {
                    "name": name,
                    "device": str(device),
                    "expected_device": str(expected_device),
                }
            )
            if device.type != expected_device.type or actual_index != expected_index:
                mismatches.append(
                    {
                        "name": name,
                        "device": str(device),
                        "expected_device": str(expected_device),
                    }
                )
        return {
            "scope": scope,
            "requested_device": self.timing_propagation_device,
            "resolved_device": str(expected_device),
            "resolved_device_type": expected_device.type,
            "global_gpu": self.timing_propagation_global_gpu,
            "global_gpu_id": self.timing_propagation_global_gpu_id,
            "cuda_available": bool(torch.cuda.is_available()),
            "checked_tensor_count": checked,
            "checked_tensors": checked_tensors,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "status": "ok" if not mismatches else "mismatch",
        }

    @staticmethod
    def _assert_device_contract(contract):
        if contract.get("mismatch_count", 0):
            first = (contract.get("mismatches") or [{}])[0]
            raise RuntimeError(
                "timing propagation device contract mismatch: "
                f"{first.get('name')} is on {first.get('device')}, "
                f"expected {first.get('expected_device')}"
            )

    def _build_parity_payload(
        self,
        *,
        enabled,
        checked,
        reason,
        wns_abs_diff_ps=None,
        tns_abs_diff_ps=None,
        max_endpoint_slack_abs_diff_ps=None,
        passed=None,
        error=None,
        extra_fields=None,
    ):
        cuda_timing_safe = None
        if enabled:
            cuda_timing_safe = bool(checked and passed)
        payload = {
            "artifact": "timing_propagation_parity_latest",
            "artifact_version": 1,
            "enabled": bool(enabled),
            "checked": bool(checked),
            "reason": reason,
            "wns_abs_diff_ps": wns_abs_diff_ps,
            "tns_abs_diff_ps": tns_abs_diff_ps,
            "max_endpoint_slack_abs_diff_ps": max_endpoint_slack_abs_diff_ps,
            "passed": passed,
            "atol_ps": self.timing_propagation_parity_atol_ps,
            "rtol": self.timing_propagation_parity_rtol,
            "cuda_timing_safe": cuda_timing_safe,
            "requested_device": self.timing_propagation_device,
            "resolved_device": str(self.resolved_timing_propagation_device),
            "resolved_device_type": self.resolved_timing_propagation_device.type,
            "global_gpu": self.timing_propagation_global_gpu,
            "global_gpu_id": self.timing_propagation_global_gpu_id,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if error is not None:
            payload["error"] = str(error)[:500]
        if extra_fields:
            payload.update(extra_fields)
        return payload

    def _try_run_cpu_cuda_parity(
        self,
        pin_net_delays,
        pin_net_impulses,
        pin_net_caps,
        surrogate_mode,
        reference_wns,
        reference_tns,
        reference_endpoint_slack,
    ):
        if not self.timing_propagation_parity_check:
            return self._build_parity_payload(
                enabled=False,
                checked=False,
                reason="disabled",
            )
        if (
            isinstance(self.last_parity_payload, dict)
            and self.last_parity_payload.get("reason") != "not_run"
        ):
            return self.last_parity_payload
        if not torch.cuda.is_available():
            return self._build_parity_payload(
                enabled=True,
                checked=False,
                reason="cuda_unavailable",
            )
        if self._cell_modeling_enabled():
            return self._build_parity_payload(
                enabled=True,
                checked=False,
                reason="parity_blocked_by_cell_modeling_surrogate_duplicate_state",
                extra_fields={
                    "blocker_file": (
                        "AiEDA/third_party/AutoDMP/dreamplace/ops/"
                        "timing_propagation/timing_propagation.py"
                    ),
                    "blocker_function": "TimingPropagation._run_cpu_cuda_parity",
                    "blocker_object": (
                        "cell_modeling_op with size_var_getter/vt_var_getter state; "
                        "deepcopy plus device materialization is not proven safe for "
                        "duplicate CPU/CUDA surrogate forwards"
                    ),
                    "next_fix": (
                        "Construct explicit CPU/CUDA TimingPropagation parity duplicates "
                        "with independently materialized cell_modeling_op, "
                        "size_var_getter, vt_var_getter, pin2node_map, and inst_main_id "
                        "state, then compare WNS/TNS/endpoint slack under torch.no_grad()."
                    ),
                },
            )
        try:
            return self._run_cpu_cuda_parity(
                pin_net_delays,
                pin_net_impulses,
                pin_net_caps,
                surrogate_mode,
                reference_wns,
                reference_tns,
                reference_endpoint_slack,
            )
        except Exception as exc:
            logging.warning("Timing propagation CPU/CUDA parity check failed: %s", exc)
            return self._build_parity_payload(
                enabled=True,
                checked=False,
                reason=f"parity_error:{type(exc).__name__}",
                error=exc,
            )

    def _run_cpu_cuda_parity(
        self,
        pin_net_delays,
        pin_net_impulses,
        pin_net_caps,
        surrogate_mode,
        reference_wns,
        reference_tns,
        reference_endpoint_slack,
    ):
        parity_device = torch.device(f"cuda:{self._resolved_gpu_id_for_cuda()}")
        saved_traversal_pruner = self._traversal_pruner
        saved_traversal_result = self.last_traversal_pruning_result
        self._traversal_pruner = None
        self.last_traversal_pruning_result = None
        try:
            cpu_op = copy.deepcopy(self)
            cuda_op = copy.deepcopy(self)
        finally:
            self._traversal_pruner = saved_traversal_pruner
            self.last_traversal_pruning_result = saved_traversal_result
        for op, device in ((cpu_op, torch.device("cpu")), (cuda_op, parity_device)):
            op.timing_propagation_parity_check = False
            op.timing_propagation_profile = False
            op.resolved_timing_propagation_device = device
            op.device = device
            op._traversal_pruner = None
            op._materialize_timing_device_tensors()
        cpu_inputs = self._move_forward_inputs(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            torch.device("cpu"),
        )
        cuda_inputs = self._move_forward_inputs(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            parity_device,
        )
        with torch.no_grad():
            cpu_wns, cpu_tns, _, _ = cpu_op(*cpu_inputs, surrogate_mode=surrogate_mode)
            cuda_wns, cuda_tns, _, _ = cuda_op(*cuda_inputs, surrogate_mode=surrogate_mode)
        cpu_slack = cpu_op.last_endpoint_slack_tensor.detach().cpu()
        cuda_slack = cuda_op.last_endpoint_slack_tensor.detach().cpu()
        wns_abs_diff = abs(float(cpu_wns.detach().cpu()) - float(cuda_wns.detach().cpu()))
        tns_abs_diff = abs(float(cpu_tns.detach().cpu()) - float(cuda_tns.detach().cpu()))
        if cpu_slack.numel() and cuda_slack.numel():
            max_slack_diff = float(torch.max(torch.abs(cpu_slack - cuda_slack)).item())
        else:
            max_slack_diff = 0.0
        passed = (
            self._within_parity_tolerance(wns_abs_diff, float(cpu_wns.detach().cpu()))
            and self._within_parity_tolerance(tns_abs_diff, float(cpu_tns.detach().cpu()))
            and self._within_parity_tolerance(
                max_slack_diff,
                float(torch.max(torch.abs(cpu_slack)).item()) if cpu_slack.numel() else 0.0,
            )
        )
        return self._build_parity_payload(
            enabled=True,
            checked=True,
            reason="passed" if passed else "diff_exceeds_tolerance",
            wns_abs_diff_ps=wns_abs_diff,
            tns_abs_diff_ps=tns_abs_diff,
            max_endpoint_slack_abs_diff_ps=max_slack_diff,
            passed=bool(passed),
        )

    def _within_parity_tolerance(self, abs_diff, reference_abs):
        return abs_diff <= (
            self.timing_propagation_parity_atol_ps
            + self.timing_propagation_parity_rtol * abs(reference_abs)
        )

    def _build_profile_payload(
        self,
        *,
        total_runtime_ms,
        stage_runtime_ms,
        surrogate_mode,
        device_contract,
        post_timing_boundary=None,
        cell_aat_profile=None,
        dynamic_provider_metadata=None,
    ):
        num_level_offsets = int(len(self.flat_inst_arcs_by_level_start))
        num_topological_levels = max(0, num_level_offsets - 1)
        pruning_stats = (
            dict(self.last_critical_endpoint_pruning_stats)
            if isinstance(self.last_critical_endpoint_pruning_stats, dict)
            else {}
        )
        return {
            "artifact": "timing_propagation_profile_latest",
            "artifact_version": 1,
            "enabled": bool(self.timing_propagation_profile),
            "forward_index": int(self.timing_forward_count),
            "requested_device": self.timing_propagation_device,
            "resolved_device": str(self.resolved_timing_propagation_device),
            "resolved_device_type": self.resolved_timing_propagation_device.type,
            "global_gpu": self.timing_propagation_global_gpu,
            "global_gpu_id": self.timing_propagation_global_gpu_id,
            "timing_aggregation": {
                "mode": self.timing_aggregation_mode,
                "tau_ps": self.timing_aggregation_tau_ps,
                "aat_rat_only": True,
                "transition_aggregation_mode": "hard",
            },
            "cuda_available": bool(torch.cuda.is_available()),
            "device_contract": device_contract,
            "post_timing_boundary": post_timing_boundary
            or {
                "core_device": str(self.resolved_timing_propagation_device),
                "return_device": str(self.resolved_timing_propagation_device),
                "returns_moved_to_return_device": False,
                "returned_tensors": ["wns", "tns", "ws", "ts"],
                "excluded_from_runtime_ms": True,
            },
            "parity": self.last_parity_payload,
            "surrogate_mode": surrogate_mode,
            "runtime_ms": float(total_runtime_ms),
            "stage_runtime_ms": {
                str(name): float(value) for name, value in stage_runtime_ms.items()
            },
            "cell_rat_levels_skipped_by_fast_loop": bool(
                getattr(self, "last_fast_loop_skipped_full_rat", False)
            ),
            "cell_aat_levels_detail": cell_aat_profile
            or self._empty_cell_aat_profile(enabled=False),
            "dynamic_provider_profile": (
                dict(dynamic_provider_metadata)
                if isinstance(dynamic_provider_metadata, dict)
                else None
            ),
            "num_topological_levels": num_topological_levels,
            "num_cell_aat_levels": max(0, num_level_offsets - 2),
            "num_cell_rat_levels": max(0, num_level_offsets - 2),
            "num_pins": int(self.num_pins),
            "num_inst_arcs": int(self.flat_inst_arcs_by_level.shape[0]),
            "critical_endpoint_pruning": {
                "mode": self.critical_endpoint_pruning_mode,
                "refresh_interval": self.critical_endpoint_refresh_interval,
                "active_endpoint_count": pruning_stats.get("active_endpoint_count"),
                "applied": pruning_stats.get("applied", False),
                "traversal_pruning": dict(self.last_traversal_pruning_stats),
            },
        }

    def build_timing_propagation_profile_artifact(self):
        if not self.timing_propagation_profile:
            return None
        return self.last_profile_payload

    def build_timing_propagation_parity_artifact(self):
        if not self.timing_propagation_parity_check:
            return None
        return self.last_parity_payload

    def _sync_profile_clock(self, device=None):
        if not getattr(self, "timing_propagation_profile", False):
            return time.perf_counter()
        sync_device = device if device is not None else self.device
        if torch.device(sync_device).type == "cuda":
            torch.cuda.synchronize(sync_device)
        return time.perf_counter()

    def _empty_cell_aat_profile(self, enabled=False):
        return {
            "enabled": bool(enabled),
            "num_levels": 0,
            "num_arcs": 0,
            "query_count": 0,
            "total_ms": 0.0,
            "query_build_ms": 0.0,
            "query_eval_ms": 0.0,
            "query_pack_ms": 0.0,
            "surrogate_batch_ms": 0.0,
            "fallback_lut_ms": 0.0,
            "fallback_lut_select_ms": 0.0,
            "fallback_lut_eval_ms": 0.0,
            "surrogate_support_cache_ms": 0.0,
            "result_split_ms": 0.0,
            "result_all_valid_check_ms": 0.0,
            "result_clone_ms": 0.0,
            "result_fallback_mask_ms": 0.0,
            "result_fallback_assign_ms": 0.0,
            "result_split_non_lut_ms": 0.0,
            "surrogate_state_ms": 0.0,
            "surrogate_index_ms": 0.0,
            "surrogate_support_ms": 0.0,
            "surrogate_forward_ms": 0.0,
            "surrogate_scatter_ms": 0.0,
            "cell_update_ms": 0.0,
            "unique_net_ms": 0.0,
            "arc_delay_scatter_ms": 0.0,
            "net_aat_ms": 0.0,
            "surrogate_candidate_count": 0,
            "surrogate_supported_count": 0,
            "surrogate_unsupported_count": 0,
            "surrogate_non_sizeable_candidate_count": 0,
            "surrogate_fixed_non_sizeable_count": 0,
            "surrogate_invalid_main_id_count": 0,
            "surrogate_non_sizeable_count": 0,
            "surrogate_support_missing_count": 0,
            "surrogate_support_skipped_count": 0,
            "surrogate_support_cache_entry_count": 0,
            "surrogate_support_cache_support_missing_count": 0,
            "surrogate_support_cache_fixed_non_sizeable_count": 0,
            "fallback_lut_query_count": 0,
            "fallback_lut_entry_count": 0,
            "fallback_lut_scalar_entry_count": 0,
            "fallback_lut_trans_1d_entry_count": 0,
            "fallback_lut_cap_1d_entry_count": 0,
            "fallback_lut_2d_entry_count": 0,
            "fallback_lut_f_delay_entry_count": 0,
            "fallback_lut_r_delay_entry_count": 0,
            "fallback_lut_f_trans_entry_count": 0,
            "fallback_lut_r_trans_entry_count": 0,
            "direct_lut_query_count": 0,
            "direct_lut_entry_count": 0,
            "surrogate_support_missing_unique_key_count": 0,
            "surrogate_support_missing_top_keys": [],
            "surrogate_unsupported_unique_key_count": 0,
            "surrogate_unsupported_top_keys": [],
            "surrogate_support_cache_status": "unknown",
            "top_levels": [],
        }

    @staticmethod
    def _cell_aat_detail_timing_keys():
        return (
            "query_build_ms",
            "query_eval_ms",
            "query_pack_ms",
            "surrogate_batch_ms",
            "fallback_lut_ms",
            "fallback_lut_select_ms",
            "fallback_lut_eval_ms",
            "surrogate_support_cache_ms",
            "result_split_ms",
            "result_all_valid_check_ms",
            "result_clone_ms",
            "result_fallback_mask_ms",
            "result_fallback_assign_ms",
            "result_split_non_lut_ms",
            "surrogate_state_ms",
            "surrogate_index_ms",
            "surrogate_support_ms",
            "surrogate_forward_ms",
            "surrogate_scatter_ms",
            "cell_update_ms",
            "unique_net_ms",
        )

    @staticmethod
    def _cell_aat_detail_count_keys():
        return (
            "surrogate_candidate_count",
            "surrogate_supported_count",
            "surrogate_unsupported_count",
            "surrogate_non_sizeable_candidate_count",
            "surrogate_fixed_non_sizeable_count",
            "surrogate_invalid_main_id_count",
            "surrogate_non_sizeable_count",
            "surrogate_support_missing_count",
            "surrogate_support_skipped_count",
            "surrogate_support_cache_entry_count",
            "surrogate_support_cache_support_missing_count",
            "surrogate_support_cache_fixed_non_sizeable_count",
            "fallback_lut_query_count",
            "fallback_lut_entry_count",
            "fallback_lut_scalar_entry_count",
            "fallback_lut_trans_1d_entry_count",
            "fallback_lut_cap_1d_entry_count",
            "fallback_lut_2d_entry_count",
            "fallback_lut_f_delay_entry_count",
            "fallback_lut_r_delay_entry_count",
            "fallback_lut_f_trans_entry_count",
            "fallback_lut_r_trans_entry_count",
            "direct_lut_query_count",
            "direct_lut_entry_count",
        )

    def _record_local_surrogate_unsupported_keys(
        self,
        profile,
        main_ids,
        arc_offsets,
        arc_types,
    ):
        if not isinstance(profile, dict):
            return
        if not (
            torch.is_tensor(main_ids)
            and torch.is_tensor(arc_offsets)
            and torch.is_tensor(arc_types)
        ):
            return
        if main_ids.numel() == 0:
            return
        key_counts = profile.setdefault("_surrogate_unsupported_key_counts", {})
        main_ids_cpu = main_ids.detach().to("cpu", dtype=torch.long).reshape(-1)
        arc_offsets_cpu = arc_offsets.detach().to("cpu", dtype=torch.long).reshape(-1)
        arc_types_cpu = arc_types.detach().to("cpu", dtype=torch.long).reshape(-1)
        for main_id, arc_offset, arc_type in zip(
            main_ids_cpu.tolist(),
            arc_offsets_cpu.tolist(),
            arc_types_cpu.tolist(),
        ):
            key = (int(main_id), int(arc_offset), int(arc_type))
            key_counts[key] = int(key_counts.get(key, 0)) + 1

    @staticmethod
    def _format_surrogate_unsupported_top_keys(profile, limit=20, remove_raw=True):
        if not isinstance(profile, dict):
            return
        raw_counts = profile.get("_surrogate_unsupported_key_counts")
        if remove_raw:
            profile.pop("_surrogate_unsupported_key_counts", None)
        if not isinstance(raw_counts, dict):
            profile.setdefault("surrogate_support_missing_unique_key_count", 0)
            profile.setdefault("surrogate_support_missing_top_keys", [])
            profile.setdefault("surrogate_unsupported_unique_key_count", 0)
            profile.setdefault("surrogate_unsupported_top_keys", [])
            return
        top_items = sorted(
            raw_counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:limit]
        top_key_payload = [
            {
                "main_id": int(key[0]),
                "arc_offset": int(key[1]),
                "arc_type": int(key[2]),
                "count": int(count),
            }
            for key, count in top_items
        ]
        profile["surrogate_support_missing_unique_key_count"] = len(raw_counts)
        profile["surrogate_support_missing_top_keys"] = top_key_payload
        profile["surrogate_unsupported_unique_key_count"] = len(raw_counts)
        profile["surrogate_unsupported_top_keys"] = top_key_payload

    def _record_cell_aat_profile_value(self, name, elapsed_seconds):
        profile = getattr(self, "_active_cell_aat_profile", None)
        if not isinstance(profile, dict):
            return
        profile[name] = float(profile.get(name, 0.0)) + float(elapsed_seconds) * 1000.0

    def _record_local_cell_aat_profile_value(self, profile, name, elapsed_seconds):
        if not isinstance(profile, dict):
            return
        profile[name] = float(profile.get(name, 0.0)) + float(elapsed_seconds) * 1000.0

    def _record_local_cell_aat_profile_count(self, profile, name, count):
        if not isinstance(profile, dict):
            return
        profile[name] = int(profile.get(name, 0)) + int(count)

    def _record_cell_aat_level_profile(self, level_profile):
        profile = getattr(self, "_active_cell_aat_profile", None)
        if not isinstance(profile, dict) or not isinstance(level_profile, dict):
            return
        if level_profile.get("level") is None:
            level_profile = dict(level_profile)
            level_profile["level"] = int(profile.get("num_levels", 0))
        for key in self._cell_aat_detail_timing_keys():
            level_profile.setdefault(key, 0.0)
        for key in self._cell_aat_detail_count_keys():
            level_profile.setdefault(key, 0)
        raw_key_counts = level_profile.get("_surrogate_unsupported_key_counts")
        if isinstance(raw_key_counts, dict):
            global_key_counts = profile.setdefault("_surrogate_unsupported_key_counts", {})
            for key, count in raw_key_counts.items():
                global_key_counts[key] = int(global_key_counts.get(key, 0)) + int(count)
        else:
            global_key_counts = profile.setdefault("_surrogate_unsupported_key_counts", {})
            for item in level_profile.get("surrogate_unsupported_top_keys", []):
                key = (
                    int(item.get("main_id", -1)),
                    int(item.get("arc_offset", -1)),
                    int(item.get("arc_type", -1)),
                )
                global_key_counts[key] = int(global_key_counts.get(key, 0)) + int(
                    item.get("count", 0)
                )
        self._format_surrogate_unsupported_top_keys(level_profile)
        profile["num_levels"] = int(profile.get("num_levels", 0)) + 1
        profile["num_arcs"] = int(profile.get("num_arcs", 0)) + int(
            level_profile.get("num_arcs", 0)
        )
        profile["query_count"] = int(profile.get("query_count", 0)) + int(
            level_profile.get("query_count", 0)
        )
        for key in self._cell_aat_detail_timing_keys():
            profile[key] = float(profile.get(key, 0.0)) + float(level_profile.get(key, 0.0))
        for key in self._cell_aat_detail_count_keys():
            profile[key] = int(profile.get(key, 0)) + int(level_profile.get(key, 0))
        profile["top_levels"].append(level_profile)

    def _get_cell_aat_static_level_data(self, inst_arcs):
        cache = getattr(self, "_cell_aat_static_cache", None)
        if cache is None:
            cache = {}
            self._cell_aat_static_cache = cache
        key = (int(inst_arcs.data_ptr()), tuple(inst_arcs.shape), str(inst_arcs.device))
        cached = cache.get(key)
        if cached is not None:
            self._cell_aat_static_cache_hits = int(
                getattr(self, "_cell_aat_static_cache_hits", 0)
            ) + 1
            return cached

        timing_senses = inst_arcs[:, 4]
        is_pos_unate = timing_senses == 1
        is_neg_unate = timing_senses == -1
        is_non_unate = ~(is_pos_unate | is_neg_unate)
        static_data = {
            "arc_in_pins": inst_arcs[:, 0],
            "arc_out_pins": inst_arcs[:, 1],
            "lib_cell_idxs": inst_arcs[:, 2],
            "lib_arc_idxs": inst_arcs[:, 3],
            "is_pos_unate": is_pos_unate,
            "is_neg_unate": is_neg_unate,
            "is_non_unate": is_non_unate,
            "has_pos_unate": bool(torch.any(is_pos_unate).detach().cpu().item()),
            "has_neg_unate": bool(torch.any(is_neg_unate).detach().cpu().item()),
            "has_non_unate": bool(torch.any(is_non_unate).detach().cpu().item()),
        }
        cache[key] = static_data
        self._cell_aat_static_cache_misses = int(
            getattr(self, "_cell_aat_static_cache_misses", 0)
        ) + 1
        return static_data

    @staticmethod
    def _move_forward_inputs(pin_net_delays, pin_net_impulses, pin_net_caps, device):
        return (
            {key: value.to(device) for key, value in pin_net_delays.items()},
            {key: value.to(device) for key, value in pin_net_impulses.items()},
            {key: value.to(device) for key, value in pin_net_caps.items()},
        )

    def _forward_input_tensor(self, tensor):
        tensor = tensor.to(self.device)
        if getattr(self, "production_fast_loop", False):
            return tensor
        return tensor.clone()

    def _apply_dynamic_net_arc_inputs(
        self,
        pin_net_delays,
        pin_net_impulses,
        pin_net_caps,
        dynamic_net_arc_inputs,
    ):
        if dynamic_net_arc_inputs is None:
            self.last_dynamic_net_arc_input_status = {"status": "not_requested"}
            return pin_net_delays, pin_net_impulses, pin_net_caps

        required_groups = (
            ("pin_net_delays", pin_net_delays),
            ("pin_net_impulses", pin_net_impulses),
            ("pin_net_caps", pin_net_caps),
        )
        resolved = {}
        for group_name, reference_group in required_groups:
            if group_name not in dynamic_net_arc_inputs:
                raise ValueError(f"dynamic_net_arc_inputs missing {group_name}")
            dynamic_group = dynamic_net_arc_inputs[group_name]
            resolved_group = {}
            for lane in ("rise", "fall"):
                if lane not in dynamic_group:
                    raise ValueError(f"dynamic_net_arc_inputs[{group_name}] missing {lane}")
                dynamic_tensor = dynamic_group[lane]
                reference_tensor = reference_group[lane]
                if dynamic_tensor.shape != reference_tensor.shape:
                    raise ValueError(
                        f"dynamic_net_arc_inputs[{group_name}][{lane}] shape "
                        f"{tuple(dynamic_tensor.shape)} does not match "
                        f"{tuple(reference_tensor.shape)}"
                    )
                resolved_group[lane] = dynamic_tensor
            resolved[group_name] = resolved_group

        self.last_dynamic_net_arc_input_status = {
            "status": "applied",
            "source": str(dynamic_net_arc_inputs.get("source", "explicit")),
        }
        if "net_subgraph_timing_mode" in dynamic_net_arc_inputs:
            self.last_dynamic_net_arc_input_status["net_subgraph_timing_mode"] = str(
                dynamic_net_arc_inputs["net_subgraph_timing_mode"]
            )
        if "net_subgraph_native_device" in dynamic_net_arc_inputs:
            self.last_dynamic_net_arc_input_status["net_subgraph_native_device"] = str(
                dynamic_net_arc_inputs["net_subgraph_native_device"]
            )
        for status_key in (
            "gradient_chain_level",
            "buffer_coordinate_source",
            "strong_gradient_chain_claim_allowed",
        ):
            if status_key in dynamic_net_arc_inputs:
                self.last_dynamic_net_arc_input_status[status_key] = str(
                    dynamic_net_arc_inputs[status_key]
                )
        if "metadata" in dynamic_net_arc_inputs:
            self.last_dynamic_net_arc_input_status["metadata"] = dynamic_net_arc_inputs[
                "metadata"
            ]
        return (
            resolved["pin_net_delays"],
            resolved["pin_net_impulses"],
            resolved["pin_net_caps"],
        )

    def _critical_endpoint_pruning_enabled(self):
        return self.critical_endpoint_pruning_mode == "dynamic"

    def _traversal_pruning_enabled(self):
        return self._critical_endpoint_pruning_enabled()

    def _empty_traversal_pruning_stats(self, iteration, reason):
        return {
            "enabled": bool(self._traversal_pruning_enabled()),
            "applied": False,
            "iteration": None if iteration is None else int(iteration),
            "refresh_interval": int(getattr(self, "traversal_pruning_refresh_interval", 10)),
            "refreshed_this_iteration": False,
            "active_endpoint_count": 0,
            "active_inst_count": 0,
            "active_arc_count": 0,
            "dropped_arc_count": 0,
            "parallel_task_count": 0,
            "per_level_kept_counts": [],
            "preparation_runtime_ms": None,
            "ordering": "original_flat_topological_order",
            "topology_source": self._traversal_pruner_topology_source,
            "reason": reason,
        }

    @staticmethod
    def _to_cpu_int_tensor(value, *, name, empty_shape=None):
        if value is None:
            if empty_shape is None:
                return None
            return torch.empty(empty_shape, dtype=torch.int32, device="cpu")
        if torch.is_tensor(value):
            if value.numel() == 0 and empty_shape is not None:
                return torch.empty(empty_shape, dtype=torch.int32, device="cpu")
            return value.detach().cpu().to(torch.int32).contiguous()
        try:
            array_value = np.asarray(value)
        except Exception as exc:
            raise RuntimeError(f"cannot convert {name} to tensor") from exc
        if array_value.size == 0 and empty_shape is not None:
            return torch.empty(empty_shape, dtype=torch.int32, device="cpu")
        return torch.as_tensor(array_value, dtype=torch.int32, device="cpu").contiguous()

    def _ensure_traversal_pruner(self):
        if self._traversal_pruner is not None:
            return self._traversal_pruner
        if _tp_cpp is None or not hasattr(_tp_cpp, "CriticalEndpointTraversalPruner"):
            return None
        base_required = (
            self.flat_inst_arcs_by_level,
            self.flat_inst_arcs_by_level_start,
            self.start_points,
            self.pin2node_map,
        )
        if any(value is None for value in base_required):
            return None

        compact_required = (
            self.pin_pred_start,
            self.pin_pred_pin,
            self.pin_pred_arc_id,
        )
        has_compact_topology = (
            all(value is not None for value in compact_required)
            and torch.as_tensor(self.pin_pred_start).numel() > 0
        )
        try:
            flat_arcs_cpu = self._to_cpu_int_tensor(
                self.flat_inst_arcs_by_level,
                name="flat_inst_arcs_by_level",
            )
            level_start_cpu = self._to_cpu_int_tensor(
                self.flat_inst_arcs_by_level_start,
                name="flat_inst_arcs_by_level_start",
            )
            start_points_cpu = self._to_cpu_int_tensor(self.start_points, name="start_points")
            pin2node_cpu = self._to_cpu_int_tensor(self.pin2node_map, name="pin2node_map")

            if has_compact_topology:
                self._traversal_pruner = _tp_cpp.CriticalEndpointTraversalPruner(
                    flat_arcs_cpu,
                    level_start_cpu,
                    self._to_cpu_int_tensor(self.pin_pred_start, name="pin_pred_start"),
                    self._to_cpu_int_tensor(self.pin_pred_pin, name="pin_pred_pin"),
                    self._to_cpu_int_tensor(self.pin_pred_arc_id, name="pin_pred_arc_id"),
                    start_points_cpu,
                    pin2node_cpu,
                )
                self._traversal_pruner_topology_source = "compact_csr"
            else:
                if (
                    self.flat_pin_to_graph_reverse is None
                    or self.flat_pin_to_graph_start_reverse is None
                    or self.pin_pair_arc_keys is None
                    or self.flat_pin_pair_arc_start is None
                    or self.flat_pin_pair_arc_indices is None
                ):
                    return None
                self._traversal_pruner = _tp_cpp.CriticalEndpointTraversalPruner(
                    flat_arcs_cpu,
                    level_start_cpu,
                    self._to_cpu_int_tensor(
                        self.flat_pin_to_graph_reverse,
                        name="flat_pin_to_graph_reverse",
                    ),
                    self._to_cpu_int_tensor(
                        self.flat_pin_to_graph_start_reverse,
                        name="flat_pin_to_graph_start_reverse",
                    ),
                    self._to_cpu_int_tensor(
                        self.pin_pair_arc_keys,
                        name="pin_pair_arc_keys",
                        empty_shape=(0, 2),
                    ).reshape(-1, 2),
                    self._to_cpu_int_tensor(
                        self.flat_pin_pair_arc_start,
                        name="flat_pin_pair_arc_start",
                        empty_shape=(0,),
                    ),
                    self._to_cpu_int_tensor(
                        self.flat_pin_pair_arc_indices,
                        name="flat_pin_pair_arc_indices",
                        empty_shape=(0,),
                    ),
                    start_points_cpu,
                    pin2node_cpu,
                )
                self._traversal_pruner_topology_source = "pair_key_compat"
        except Exception as exc:
            logging.warning("Traversal pruning pruner initialization failed: %s", exc)
            self._traversal_pruner = None
            self._traversal_pruner_topology_source = None
        return self._traversal_pruner

    def _refresh_traversal_pruning_state(self, iteration):
        if not self._traversal_pruning_enabled():
            self.last_traversal_pruning_result = None
            self.last_traversal_pruning_stats = self._empty_traversal_pruning_stats(
                iteration=iteration,
                reason="disabled",
            )
            return None

        active_endpoint_ids = list(self.last_active_endpoint_ids or [])
        if not active_endpoint_ids:
            self.last_traversal_pruning_result = None
            self.last_traversal_pruning_stats = self._empty_traversal_pruning_stats(
                iteration=iteration,
                reason="no_active_endpoints",
            )
            return None

        pruner = self._ensure_traversal_pruner()
        if pruner is None:
            self.last_traversal_pruning_result = None
            reason = "cpp_extension_or_static_inputs_unavailable"
            self.last_traversal_pruning_stats = self._empty_traversal_pruning_stats(
                iteration=iteration,
                reason=reason,
            )
            self.last_traversal_pruning_stats["active_endpoint_count"] = len(active_endpoint_ids)
            return None

        selector_refreshed = bool(
            isinstance(self.last_critical_endpoint_pruning_stats, dict)
            and self.last_critical_endpoint_pruning_stats.get("refreshed") is True
        )
        interval_due = (
            self.last_traversal_pruning_iteration is None
            or int(iteration) - int(self.last_traversal_pruning_iteration) >= self.traversal_pruning_refresh_interval
        )
        should_refresh = (
            self.last_traversal_pruning_result is None
            or selector_refreshed
            or interval_due
        )

        if should_refresh:
            active_tensor = torch.as_tensor(
                active_endpoint_ids,
                dtype=torch.int32,
                device="cpu",
            )
            result = pruner.refresh(active_tensor)
            self.last_traversal_pruning_result = result
            self.last_traversal_pruning_iteration = int(iteration)
            refreshed = True
        else:
            result = self.last_traversal_pruning_result
            refreshed = False

        per_level_counts = []
        if result is not None and getattr(result, "kept_counts_by_level", None) is not None:
            per_level_counts = [
                int(value)
                for value in result.kept_counts_by_level.detach().cpu().reshape(-1).tolist()
            ]
        self.last_traversal_pruning_stats = {
            "enabled": True,
            "applied": result is not None,
            "iteration": int(iteration),
            "refresh_interval": int(self.traversal_pruning_refresh_interval),
            "refreshed_this_iteration": bool(refreshed),
            "active_endpoint_count": int(len(active_endpoint_ids)),
            "active_inst_count": int(getattr(result, "active_inst_count", 0) if result is not None else 0),
            "active_arc_count": int(getattr(result, "active_arc_count", 0) if result is not None else 0),
            "dropped_arc_count": int(getattr(result, "dropped_arc_count", 0) if result is not None else 0),
            "parallel_task_count": int(getattr(result, "parallel_task_count", 0) if result is not None else 0),
            "per_level_kept_counts": per_level_counts,
            "preparation_runtime_ms": (
                float(getattr(result, "preparation_runtime_ms", 0.0))
                if result is not None and refreshed
                else 0.0
            ),
            "ordering": "original_flat_topological_order",
            "topology_source": self._traversal_pruner_topology_source,
            "reason": "refreshed" if refreshed else "reused",
        }
        return result

    def _build_traversal_pruning_working_view(self, device):
        result = self.last_traversal_pruning_result
        stats = self.last_traversal_pruning_stats
        if not (
            self._traversal_pruning_enabled()
            and isinstance(stats, dict)
            and stats.get("applied")
            and result is not None
        ):
            num_arcs = int(self.flat_inst_arcs_by_level.shape[0])
            return (
                self.flat_inst_arcs_by_level,
                self.flat_inst_arcs_by_level_start,
                torch.arange(num_arcs, device=device, dtype=torch.long),
                False,
            )
        kept_idx = result.kept_flat_arc_indices.to(device=device, dtype=torch.long)
        kept_offsets = result.kept_level_offsets.to(device=device, dtype=torch.long)
        if kept_idx.numel() == 0:
            working_flat_arcs = self.flat_inst_arcs_by_level[:0]
        else:
            working_flat_arcs = self.flat_inst_arcs_by_level.index_select(0, kept_idx)
        return working_flat_arcs, kept_offsets, kept_idx, True

    def _ensure_critical_endpoint_selector(self):
        if self.critical_endpoint_selector is not None:
            return self.critical_endpoint_selector
        self.critical_endpoint_selector = DynamicCriticalEndpointSelector(
            top_k=self.critical_endpoint_top_k,
            slack_window_ps=self.critical_endpoint_slack_window_ps,
            refresh_interval=self.critical_endpoint_refresh_interval,
            hysteresis_interval=self.critical_endpoint_hysteresis_interval,
            full_refresh_interval=self.critical_endpoint_full_refresh_interval,
        )
        return self.critical_endpoint_selector

    def update_critical_endpoint_pruning_state(self, iteration):
        if not self._critical_endpoint_pruning_enabled():
            self.last_active_endpoint_ids = []
            self.last_critical_endpoint_pruning_stats = {
                "mode": self.critical_endpoint_pruning_mode,
                "applied": False,
                "iteration": int(iteration),
            }
            return

        endpoint_slack = self.last_endpoint_slack_tensor
        endpoint_ids = self.last_endpoint_ids_tensor
        if endpoint_slack is None or endpoint_ids is None:
            self.last_active_endpoint_ids = []
            self.last_critical_endpoint_pruning_stats = {
                "mode": "dynamic",
                "applied": False,
                "iteration": int(iteration),
                "missing_reason": "endpoint_slack_or_ids_unavailable",
            }
            return

        selector = self._ensure_critical_endpoint_selector()
        if selector.should_refresh(iteration) or not selector.active_endpoint_ids:
            active_endpoint_ids, stats = selector.refresh(
                iteration=iteration,
                endpoint_ids=endpoint_ids.to(endpoint_slack.device),
                endpoint_slack_ps=endpoint_slack.detach(),
            )
            stats["refreshed"] = True
        else:
            active_endpoint_ids = set(selector.active_endpoint_ids)
            stats = dict(selector.last_stats)
            stats.update({"iteration": int(iteration), "refreshed": False})
        stats["mode"] = "dynamic"
        stats["applied"] = True
        self.last_active_endpoint_ids = sorted(active_endpoint_ids)
        self.last_critical_endpoint_pruning_stats = stats

    def select_critical_endpoint_timing(self, full_wns, full_tns):
        mode = self.critical_endpoint_pruning_mode
        metadata = {
            "mode": mode,
            "applied": False,
            "active_endpoint_count": 0,
            "selected_endpoint_wns_ps": None,
            "selected_endpoint_tns_ps": None,
            "full_endpoint_wns_ps": float(full_wns.detach().item()),
            "full_endpoint_tns_ps": float(full_tns.detach().item()),
        }
        if mode != "dynamic":
            return full_wns, full_tns, metadata

        endpoint_slack = self.last_endpoint_slack_tensor
        endpoint_ids = self.last_endpoint_ids_tensor
        active_endpoint_ids = self.last_active_endpoint_ids
        if endpoint_slack is None or endpoint_ids is None or not active_endpoint_ids:
            return full_wns, full_tns, metadata

        endpoint_ids = endpoint_ids.to(device=endpoint_slack.device).reshape(-1)
        active_ids_tensor = torch.as_tensor(
            list(active_endpoint_ids),
            dtype=endpoint_ids.dtype,
            device=endpoint_ids.device,
        ).reshape(-1)
        if active_ids_tensor.numel() == 0:
            return full_wns, full_tns, metadata

        active_mask = (endpoint_ids.unsqueeze(1) == active_ids_tensor.unsqueeze(0)).any(dim=1)
        if active_mask.numel() != endpoint_slack.reshape(-1).numel() or not torch.any(active_mask):
            return full_wns, full_tns, metadata

        selected_slack = endpoint_slack.reshape(-1)[active_mask]
        selected_negative_slack = torch.clamp(selected_slack, max=0.0)
        selected_wns = torch.min(selected_negative_slack)
        selected_tns = torch.sum(selected_negative_slack)
        metadata.update(
            {
                "applied": True,
                "active_endpoint_count": int(selected_slack.numel()),
                "selected_endpoint_wns_ps": float(selected_wns.detach().item()),
                "selected_endpoint_tns_ps": float(selected_tns.detach().item()),
            }
        )
        if isinstance(self.last_critical_endpoint_pruning_stats, dict):
            metadata["selector_stats"] = dict(self.last_critical_endpoint_pruning_stats)
        return selected_wns, selected_tns, metadata

    def _setup_endpoint_objective_mask(self, device):
        """Return the per-endpoint mask covered by exported setup check arcs."""
        num_endpoints = int(self.end_points.numel())
        if num_endpoints == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        constraints = getattr(self, "endpoints_constraint_arcs", None)
        if constraints is None or constraints.numel() == 0:
            return torch.ones(num_endpoints, dtype=torch.bool, device=device)
        setup_endpoint_pin_ids = constraints[:, 1].to(
            device=device,
            dtype=self.end_points.dtype,
        )
        return torch.isin(self.end_points.to(device), setup_endpoint_pin_ids)

    def build_critical_endpoint_pruning_artifact(self, iteration):
        if not self._critical_endpoint_pruning_enabled():
            return None
        active_endpoint_ids = list(self.last_active_endpoint_ids or [])
        stats = self.last_critical_endpoint_pruning_stats
        endpoint_summary = self._build_endpoint_pruning_timing_summary(active_endpoint_ids)
        refreshed = None
        if isinstance(stats, dict):
            refreshed = stats.get("refreshed")
        if refreshed is True:
            refresh_decision = "refreshed"
        elif refreshed is False:
            refresh_decision = "reused"
        else:
            refresh_decision = "unknown"
        return {
            "artifact": "critical_endpoint_pruning_latest",
            "iteration": int(iteration),
            "mode": self.critical_endpoint_pruning_mode,
            "active_endpoint_count": len(active_endpoint_ids),
            "full_endpoint_count": endpoint_summary["full_endpoint_count"],
            "selected_endpoint_wns_ps": endpoint_summary["selected_endpoint_wns_ps"],
            "selected_endpoint_tns_ps": endpoint_summary["selected_endpoint_tns_ps"],
            "full_endpoint_wns_ps": endpoint_summary["full_endpoint_wns_ps"],
            "full_endpoint_tns_ps": endpoint_summary["full_endpoint_tns_ps"],
            "refresh_interval": int(self.critical_endpoint_refresh_interval),
            "refresh_decision": refresh_decision,
            "selection_provenance": {
                "owner": "TimingPropagation",
                "source": "last_endpoint_slack_tensor",
                "top_k": int(self.critical_endpoint_top_k),
                "slack_window_ps": float(self.critical_endpoint_slack_window_ps),
                "hysteresis_interval": int(self.critical_endpoint_hysteresis_interval),
                "full_refresh_interval": int(self.critical_endpoint_full_refresh_interval),
                "active_endpoint_id_limit": 1024,
            },
            "active_endpoint_ids": [int(endpoint_id) for endpoint_id in active_endpoint_ids[:1024]],
            "active_endpoint_ids_truncated": len(active_endpoint_ids) > 1024,
            "selector_stats": dict(stats) if isinstance(stats, dict) else {},
            "traversal_pruning_enabled": bool(
                self.last_traversal_pruning_stats.get("enabled", False)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else False
            ),
            "traversal_refresh_interval": int(
                self.last_traversal_pruning_stats.get(
                    "refresh_interval",
                    self.traversal_pruning_refresh_interval,
                )
                if isinstance(self.last_traversal_pruning_stats, dict)
                else self.traversal_pruning_refresh_interval
            ),
            "traversal_refreshed_this_iteration": bool(
                self.last_traversal_pruning_stats.get("refreshed_this_iteration", False)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else False
            ),
            "traversal_active_endpoint_count": int(
                self.last_traversal_pruning_stats.get("active_endpoint_count", 0)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else 0
            ),
            "traversal_active_inst_count": int(
                self.last_traversal_pruning_stats.get("active_inst_count", 0)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else 0
            ),
            "traversal_active_arc_count": int(
                self.last_traversal_pruning_stats.get("active_arc_count", 0)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else 0
            ),
            "traversal_dropped_arc_count": int(
                self.last_traversal_pruning_stats.get("dropped_arc_count", 0)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else 0
            ),
            "traversal_parallel_task_count": int(
                self.last_traversal_pruning_stats.get("parallel_task_count", 0)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else 0
            ),
            "traversal_per_level_kept_counts": list(
                self.last_traversal_pruning_stats.get("per_level_kept_counts", [])
                if isinstance(self.last_traversal_pruning_stats, dict)
                else []
            ),
            "traversal_preparation_runtime_ms": (
                self.last_traversal_pruning_stats.get("preparation_runtime_ms")
                if isinstance(self.last_traversal_pruning_stats, dict)
                else None
            ),
            "traversal_ordering": (
                self.last_traversal_pruning_stats.get(
                    "ordering",
                    "original_flat_topological_order",
                )
                if isinstance(self.last_traversal_pruning_stats, dict)
                else "original_flat_topological_order"
            ),
            "traversal_pruning": (
                dict(self.last_traversal_pruning_stats)
                if isinstance(self.last_traversal_pruning_stats, dict)
                else {}
            ),
        }

    def compute_active_endpoint_incidence(self, active_endpoint_ids=None):
        if _tp_cpp is None:
            return None
        pruner = self._ensure_traversal_pruner()
        if pruner is None or not hasattr(pruner, "endpoint_incidence"):
            return None
        endpoint_ids = (
            list(active_endpoint_ids)
            if active_endpoint_ids is not None
            else list(self.last_active_endpoint_ids or [])
        )
        if not endpoint_ids:
            return None
        active_tensor = torch.as_tensor(endpoint_ids, dtype=torch.int32, device="cpu")
        return pruner.endpoint_incidence(active_tensor)

    def _build_endpoint_pruning_timing_summary(self, active_endpoint_ids):
        endpoint_slack = self.last_endpoint_slack_tensor
        endpoint_ids = self.last_endpoint_ids_tensor
        summary = {
            "full_endpoint_count": 0,
            "selected_endpoint_wns_ps": None,
            "selected_endpoint_tns_ps": None,
            "full_endpoint_wns_ps": None,
            "full_endpoint_tns_ps": None,
        }
        if endpoint_slack is None or endpoint_ids is None:
            return summary

        endpoint_slack = endpoint_slack.detach().reshape(-1)
        endpoint_ids = endpoint_ids.detach().to(endpoint_slack.device).reshape(-1)
        if endpoint_slack.numel() == 0 or endpoint_ids.numel() != endpoint_slack.numel():
            return summary

        finite_mask = torch.isfinite(endpoint_slack)
        if not torch.any(finite_mask):
            return summary

        finite_endpoint_slack = endpoint_slack[finite_mask]
        finite_endpoint_ids = endpoint_ids[finite_mask]
        negative_slack = torch.clamp(finite_endpoint_slack, max=0.0)
        summary["full_endpoint_count"] = int(finite_endpoint_slack.numel())
        summary["full_endpoint_wns_ps"] = float(torch.min(negative_slack).detach().item())
        summary["full_endpoint_tns_ps"] = float(torch.sum(negative_slack).detach().item())

        if not active_endpoint_ids:
            return summary
        active_ids = torch.as_tensor(
            list(active_endpoint_ids),
            dtype=finite_endpoint_ids.dtype,
            device=finite_endpoint_ids.device,
        ).reshape(-1)
        active_mask = (finite_endpoint_ids.unsqueeze(1) == active_ids.unsqueeze(0)).any(dim=1)
        if not torch.any(active_mask):
            return summary
        selected_negative_slack = torch.clamp(finite_endpoint_slack[active_mask], max=0.0)
        summary["selected_endpoint_wns_ps"] = float(
            torch.min(selected_negative_slack).detach().item()
        )
        summary["selected_endpoint_tns_ps"] = float(
            torch.sum(selected_negative_slack).detach().item()
        )
        return summary

    def _resolve_surrogate_mode(self, surrogate_mode: Optional[str] = None) -> tuple[bool, bool]:
        """
        Resolve per-call timing model mode.

        Returns:
            (clk2q_use_surrogate, cell_use_surrogate)

        Supported modes:
            - None / "auto" / "mixed": current default behavior
              clk2q stays on accurate LUT, later cell arcs use surrogate when available.
            - "lut_only": force accurate liberty LUT for all timing arc evaluations.
            - "surrogate_only": force surrogate on all timing arc evaluations when available.
        """
        mode = (surrogate_mode or "mixed").lower()
        surrogate_available = self._cell_modeling_enabled()
        if mode in ("mixed", "auto", "default"):
            return False, surrogate_available
        if mode == "lut_only":
            return False, False
        if mode == "surrogate_only":
            return surrogate_available, surrogate_available
        raise ValueError(
            f"unsupported surrogate_mode={surrogate_mode!r}; "
            "expected one of: auto, mixed, lut_only, surrogate_only"
        )
        self.cell_arc_r_delays = None
        self.cell_arc_f_delays = None
        self.pin_slack = None
        self.pin_net_delay_rise = None
        self.pin_net_delay_fall = None

    def _check_index_range(self, name, tensor, upper_bound):
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        if upper_bound <= 0:
            return
        tensor = tensor.detach()
        min_val = int(tensor.min().item())
        max_val = int(tensor.max().item())
        if min_val < 0 or max_val >= upper_bound:
            raise RuntimeError(
                f"TimingPropagation invalid index range for {name}: "
                f"min={min_val}, max={max_val}, upper_bound={upper_bound}"
            )

    def _validate_runtime_indices(self):
        num_pins = int(self.num_pins)
        self._check_index_range("start_points", self.start_points, num_pins)
        self._check_index_range("end_points", self.end_points, num_pins)
        self._check_index_range("clock_pins", self.clock_pins, num_pins)
        self._check_index_range("FF_ids", self.FF_ids, num_pins)
        self._check_index_range("pin_net", self.pin_net, self.net2driver_pin_map.numel())
        # The timing snapshot uses -1 for nets without a timing arc.
        drivers = self.net2driver_pin_map[self.net2driver_pin_map != -1]
        self._check_index_range("net2driver_pin_map", drivers, num_pins)
        self._check_index_range("flat_pin_to_graph", self.flat_pin_to_graph, num_pins)
        self._check_index_range("flat_pin_to_graph_reverse", self.flat_pin_to_graph_reverse, num_pins)

        if self.endpoints_constraint_arcs is not None and self.endpoints_constraint_arcs.numel() > 0:
            self._check_index_range(
                "endpoints_constraint_arcs.pin_cols",
                self.endpoints_constraint_arcs[:, :2],
                num_pins,
            )

        if self.flat_inst_arcs_by_level is not None and self.flat_inst_arcs_by_level.numel() > 0:
            self._check_index_range(
                "flat_inst_arcs_by_level.pin_cols",
                self.flat_inst_arcs_by_level[:, :2],
                num_pins,
            )

    def _cell_modeling_enabled(self):
        return (
            self.cell_modeling_op is not None
            and self.pin2node_map is not None
            and self.inst_main_id is not None
            and self.size_var_getter is not None
            and self.vt_var_getter is not None
        )

    def _fixed_non_sizeable_state_available(self):
        return (
            getattr(self, "inst_size_init", None) is not None
            and getattr(self, "inst_vt_init", None) is not None
        )

    def _begin_forward_runtime_state(self):
        self._surrogate_runtime_state_active = True
        self._surrogate_state_cache = None

    def _end_forward_runtime_state(self):
        self._surrogate_runtime_state_active = False
        self._surrogate_state_cache = None

    def _get_surrogate_state(self):
        runtime_cache_active = bool(
            getattr(self, "_surrogate_runtime_state_active", False)
        )
        if runtime_cache_active:
            cached = getattr(self, "_surrogate_state_cache", None)
            if isinstance(cached, dict):
                return cached
        size_var = self.size_var_getter()
        vt_var = self.vt_var_getter()
        if size_var is None or vt_var is None:
            return None
        state = {
            "size_var": size_var.to(self.device),
            "vt_var": vt_var.to(self.device),
        }
        if self._fixed_non_sizeable_state_available():
            state["inst_size_init"] = self.inst_size_init.to(self.device)
            state["inst_vt_init"] = self.inst_vt_init.to(self.device)
        if runtime_cache_active:
            self._surrogate_state_cache = state
        return state

    def _vt_scalar(self, vt_var):
        vt_codes = torch.arange(vt_var.shape[1], device=vt_var.device, dtype=vt_var.dtype)
        return torch.sum(vt_var * vt_codes.unsqueeze(0), dim=1)

    def _surrogate_entry(self, node_ids, pin_slew, pin_net_caps, lib_arc_idxs, arc_type):
        arc_types = torch.full(
            (pin_slew.numel(),),
            int(arc_type),
            device=pin_slew.device,
            dtype=torch.long,
        )
        surrogate = self._surrogate_batch_entry(
            node_ids,
            pin_slew,
            pin_net_caps,
            lib_arc_idxs,
            arc_types,
        )
        if isinstance(surrogate, tuple) and len(surrogate) == 3:
            return surrogate[0], surrogate[1]
        return surrogate

    def _surrogate_batch_entry(
        self,
        node_ids,
        pin_slew,
        pin_net_caps,
        lib_arc_idxs,
        arc_types,
        profile=None,
    ):
        if not self._cell_modeling_enabled():
            return None
        if node_ids is None or node_ids.numel() == 0:
            return None
        profile_enabled = isinstance(profile, dict)
        profile_timer = self._sync_profile_clock(pin_slew.device) if profile_enabled else None
        pin_ids = node_ids.long()
        if self.pin2node_map is None:
            return None
        node_ids = self.pin2node_map[pin_ids].long()
        if profile_enabled:
            profile_now = self._sync_profile_clock(pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_index_ms",
                profile_now - profile_timer,
            )
            profile_timer = profile_now

        surrogate_state = self._get_surrogate_state()
        if surrogate_state is None:
            return None
        if profile_enabled:
            profile_now = self._sync_profile_clock(pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_state_ms",
                profile_now - profile_timer,
            )
            profile_timer = profile_now

        size_var = surrogate_state["size_var"]
        vt_var = surrogate_state["vt_var"]
        main_ids = self.inst_main_id[node_ids].long()
        arc_types = arc_types.long()
        valid_main_id_mask = main_ids >= 0
        sizeable_mask = torch.ones_like(valid_main_id_mask, dtype=torch.bool)
        if self.inst_is_sizeable is not None:
            sizeable_mask = self.inst_is_sizeable[node_ids].bool()
        fixed_non_sizeable_mask = torch.zeros_like(valid_main_id_mask, dtype=torch.bool)
        inst_size_init = surrogate_state.get("inst_size_init")
        inst_vt_init = surrogate_state.get("inst_vt_init")
        if inst_size_init is not None and inst_vt_init is not None:
            fixed_non_sizeable_mask = valid_main_id_mask & ~sizeable_mask
        valid_mask = valid_main_id_mask & (sizeable_mask | fixed_non_sizeable_mask)
        if profile_enabled:
            invalid_main_id_count = int(
                torch.count_nonzero(~valid_main_id_mask).detach().cpu().item()
            )
            non_sizeable_candidate_count = int(
                torch.count_nonzero(valid_main_id_mask & ~sizeable_mask)
                .detach()
                .cpu()
                .item()
            )
            fixed_non_sizeable_count = int(
                torch.count_nonzero(fixed_non_sizeable_mask).detach().cpu().item()
            )
            non_sizeable_count = int(
                torch.count_nonzero(
                    valid_main_id_mask & ~sizeable_mask & ~fixed_non_sizeable_mask
                )
                .detach()
                .cpu()
                .item()
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_non_sizeable_candidate_count",
                non_sizeable_candidate_count,
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_fixed_non_sizeable_count",
                fixed_non_sizeable_count,
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_invalid_main_id_count",
                invalid_main_id_count,
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_non_sizeable_count",
                non_sizeable_count,
            )
        if not torch.any(valid_mask):
            if profile_enabled:
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_candidate_count",
                    int(pin_slew.numel()),
                )
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_unsupported_count",
                    int(pin_slew.numel()),
                )
            return valid_mask, torch.zeros_like(pin_slew), False
        surrogate_values = torch.zeros_like(pin_slew)
        size_values = size_var[node_ids]
        vt_values = vt_var[node_ids]
        if torch.any(fixed_non_sizeable_mask):
            size_values = size_values.clone()
            vt_values = vt_values.clone()
            size_values[fixed_non_sizeable_mask] = inst_size_init[node_ids][
                fixed_non_sizeable_mask
            ].detach()
            vt_values[fixed_non_sizeable_mask] = inst_vt_init[node_ids][
                fixed_non_sizeable_mask
            ].detach()
        vt_scalar = self._vt_scalar(vt_values[valid_mask])
        arc_offsets = lib_arc_idxs[valid_mask].long()
        selected_arc_types = arc_types[valid_mask]
        if self.lib_arc_offsets is not None:
            arc_offsets = self.lib_arc_offsets[arc_offsets].long()
        skip_support_check = (
            getattr(self, "_surrogate_support_cache_status", "unknown")
            == "all_supported"
        )
        if skip_support_check:
            if profile_enabled:
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_support_skipped_count",
                    int(selected_arc_types.numel()),
                )
            support_mask = None
        elif hasattr(self.cell_modeling_op, "supports_arc_types"):
            support_mask = self.cell_modeling_op.supports_arc_types(
                main_ids[valid_mask],
                arc_offsets,
                selected_arc_types,
            )
        else:
            support_mask = None
        if support_mask is not None:
            if profile_enabled:
                support_missing_count = int(
                    torch.count_nonzero(~support_mask).detach().cpu().item()
                )
                self._record_local_cell_aat_profile_count(
                    profile,
                    "surrogate_support_missing_count",
                    support_missing_count,
                )
            if profile_enabled and torch.any(~support_mask):
                self._record_local_surrogate_unsupported_keys(
                    profile,
                    main_ids[valid_mask][~support_mask],
                    arc_offsets[~support_mask],
                    selected_arc_types[~support_mask],
                )
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            valid_mask = valid_mask.clone()
            valid_mask[valid_indices] = support_mask
            arc_offsets = arc_offsets[support_mask]
            vt_scalar = vt_scalar[support_mask]
            selected_arc_types = selected_arc_types[support_mask]
            if not torch.any(valid_mask):
                if profile_enabled:
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "surrogate_candidate_count",
                        int(pin_slew.numel()),
                    )
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "surrogate_unsupported_count",
                        int(pin_slew.numel()),
                    )
                return valid_mask, surrogate_values, False
        if profile_enabled:
            supported_count = int(torch.count_nonzero(valid_mask).detach().cpu().item())
            candidate_count = int(pin_slew.numel())
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_candidate_count",
                candidate_count,
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_supported_count",
                supported_count,
            )
            self._record_local_cell_aat_profile_count(
                profile,
                "surrogate_unsupported_count",
                candidate_count - supported_count,
            )
        if profile_enabled:
            profile_now = self._sync_profile_clock(pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_support_ms",
                profile_now - profile_timer,
            )
            profile_timer = profile_now

        cell_model_values = self.cell_modeling_op(
            main_ids[valid_mask],
            arc_offsets,
            vt_scalar,
            size_values[valid_mask],
            self._surrogate_slew_input_tensor(pin_slew)[valid_mask],
            self._surrogate_cap_input_tensor(pin_net_caps)[valid_mask],
            selected_arc_types,
        )
        if profile_enabled:
            profile_now = self._sync_profile_clock(pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_forward_ms",
                profile_now - profile_timer,
            )
            profile_timer = profile_now

        surrogate_values[valid_mask] = cell_model_values
        if profile_enabled:
            profile_now = self._sync_profile_clock(pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_scatter_ms",
                profile_now - profile_timer,
            )
        return valid_mask, surrogate_values

    def _entry_queries_with_optional_surrogate(self, queries, use_surrogate, profile=None):
        if not queries:
            return []

        results = [None] * len(queries)
        profile_enabled = isinstance(profile, dict)
        if profile_enabled:
            for key in self._cell_aat_detail_count_keys():
                profile.setdefault(key, 0)
            for key in self._cell_aat_detail_timing_keys():
                profile.setdefault(key, 0.0)

        def finalize_results():
            if profile_enabled:
                self._format_surrogate_unsupported_top_keys(
                    profile,
                    remove_raw=False,
                )
            return results

        def run_lut_query(query, fallback_mask=None):
            lut_started_at = (
                self._sync_profile_clock(query["pin_slew"].device)
                if profile_enabled
                else None
            )
            lib_cell_idxs = query["lib_cell_idxs"]
            pin_slew = query["pin_slew"]
            pin_net_caps = query["pin_net_caps"]
            lib_arc_idxs = query["lib_arc_idxs"]
            luts = query["luts"]
            if fallback_mask is not None:
                lib_cell_idxs = lib_cell_idxs[fallback_mask]
                pin_slew = pin_slew[fallback_mask]
                pin_net_caps = pin_net_caps[fallback_mask]
                lib_arc_idxs = lib_arc_idxs[fallback_mask]
            if profile_enabled:
                lut_selected_at = self._sync_profile_clock(query["pin_slew"].device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "fallback_lut_select_ms",
                    lut_selected_at - lut_started_at,
                )
            else:
                lut_selected_at = None
            result = self.lut_entry_vectorized(
                lib_cell_idxs,
                pin_slew,
                pin_net_caps,
                lib_arc_idxs,
                luts,
            )
            if profile_enabled:
                lut_eval_done_at = self._sync_profile_clock(query["pin_slew"].device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "fallback_lut_eval_ms",
                    lut_eval_done_at - lut_selected_at,
                )
                if fallback_mask is None:
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "direct_lut_query_count",
                        1,
                    )
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "direct_lut_entry_count",
                        int(pin_slew.numel()),
                    )
                else:
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "fallback_lut_query_count",
                        1,
                    )
                    self._record_local_cell_aat_profile_count(
                        profile,
                        "fallback_lut_entry_count",
                        int(pin_slew.numel()),
                    )
                    for lut_mask_name, count_key in (
                        ("is_scalar", "fallback_lut_scalar_entry_count"),
                        ("is_trans_1d", "fallback_lut_trans_1d_entry_count"),
                        ("is_cap_1d", "fallback_lut_cap_1d_entry_count"),
                        ("is_2d", "fallback_lut_2d_entry_count"),
                    ):
                        lut_type_mask = getattr(luts, lut_mask_name, None)
                        if lut_type_mask is None:
                            continue
                        lut_type_entries = lut_type_mask[lib_arc_idxs]
                        if isinstance(lut_type_entries, torch.Tensor):
                            lut_type_count = int(torch.count_nonzero(lut_type_entries).item())
                        else:
                            lut_type_count = int(sum(bool(value) for value in lut_type_entries))
                        self._record_local_cell_aat_profile_count(
                            profile,
                            count_key,
                            lut_type_count,
                        )
                    arc_type_count_key = {
                        0: "fallback_lut_f_delay_entry_count",
                        1: "fallback_lut_r_delay_entry_count",
                        2: "fallback_lut_f_trans_entry_count",
                        3: "fallback_lut_r_trans_entry_count",
                    }.get(int(query.get("arc_type", -1)))
                    if arc_type_count_key is not None:
                        self._record_local_cell_aat_profile_count(
                            profile,
                            arc_type_count_key,
                            int(pin_slew.numel()),
                        )
                self._record_local_cell_aat_profile_value(
                    profile,
                    "fallback_lut_ms",
                    lut_eval_done_at - lut_started_at,
                )
            return result

        if not use_surrogate:
            for query_idx, query in enumerate(queries):
                results[query_idx] = run_lut_query(query)
            return finalize_results()

        batch_query_indices = []
        batch_query_lengths = []
        batch_node_ids = []
        batch_pin_slew = []
        batch_pin_net_caps = []
        batch_lib_arc_idxs = []
        batch_arc_types = []

        pack_started_at = (
            self._sync_profile_clock(queries[0]["pin_slew"].device)
            if profile_enabled
            else None
        )
        for query_idx, query in enumerate(queries):
            local_length = int(query["pin_slew"].numel())
            if local_length == 0:
                results[query_idx] = torch.zeros_like(query["pin_slew"])
                continue
            if query.get("node_ids") is None:
                results[query_idx] = run_lut_query(query)
                continue
            batch_query_indices.append(query_idx)
            batch_query_lengths.append(local_length)
            batch_node_ids.append(query["node_ids"].long())
            batch_pin_slew.append(query["pin_slew"])
            batch_pin_net_caps.append(query["pin_net_caps"])
            batch_lib_arc_idxs.append(query["lib_arc_idxs"].long())
            batch_arc_types.append(
                torch.full(
                    (local_length,),
                    int(query["arc_type"]),
                    device=query["pin_slew"].device,
                    dtype=torch.long,
                )
            )

        if not batch_query_indices:
            return finalize_results()

        batch_node_ids = torch.cat(batch_node_ids, dim=0)
        batch_pin_slew = torch.cat(batch_pin_slew, dim=0)
        batch_pin_net_caps = torch.cat(batch_pin_net_caps, dim=0)
        batch_lib_arc_idxs = torch.cat(batch_lib_arc_idxs, dim=0)
        batch_arc_types = torch.cat(batch_arc_types, dim=0)
        if profile_enabled:
            pack_now = self._sync_profile_clock(batch_pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "query_pack_ms",
                pack_now - pack_started_at,
            )
        else:
            pack_now = None

        surrogate_started_at = pack_now
        surrogate = self._surrogate_batch_entry(
            batch_node_ids,
            batch_pin_slew,
            batch_pin_net_caps,
            batch_lib_arc_idxs,
            batch_arc_types,
            profile=profile,
        )
        if profile_enabled:
            surrogate_now = self._sync_profile_clock(batch_pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "surrogate_batch_ms",
                surrogate_now - surrogate_started_at,
            )
        if surrogate is None:
            for query_idx in batch_query_indices:
                results[query_idx] = run_lut_query(queries[query_idx])
            return finalize_results()

        if isinstance(surrogate, tuple) and len(surrogate) >= 2:
            batched_valid_mask, batched_surrogate_values = surrogate[:2]
        else:
            batched_valid_mask, batched_surrogate_values = surrogate
        if torch.all(batched_valid_mask):
            split_started_at = (
                self._sync_profile_clock(batch_pin_slew.device)
                if profile_enabled
                else None
            )
            offset = 0
            for query_idx, local_length in zip(
                batch_query_indices,
                batch_query_lengths,
            ):
                results[query_idx] = batched_surrogate_values[
                    offset: offset + local_length
                ]
                offset += local_length
            if profile_enabled:
                split_now = self._sync_profile_clock(batch_pin_slew.device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "result_all_valid_check_ms",
                    split_now - split_started_at,
                )
                self._record_local_cell_aat_profile_value(
                    profile,
                    "result_split_ms",
                    split_now - split_started_at,
                )
                profile["result_split_non_lut_ms"] = (
                    float(profile.get("result_split_non_lut_ms", 0.0))
                    + (split_now - split_started_at) * 1000.0
                )
            return finalize_results()

        split_started_at = (
            self._sync_profile_clock(batch_pin_slew.device)
            if profile_enabled
            else None
        )
        split_lut_ms_before = float(profile.get("fallback_lut_ms", 0.0)) if profile_enabled else 0.0
        offset = 0
        for query_idx, local_length in zip(
            batch_query_indices,
            batch_query_lengths,
        ):
            split_step_started_at = (
                self._sync_profile_clock(batch_pin_slew.device)
                if profile_enabled
                else None
            )
            local_surrogate_values = batched_surrogate_values[offset: offset + local_length]
            local_valid_mask = batched_valid_mask[offset: offset + local_length]
            offset += local_length
            if profile_enabled:
                split_step_done_at = self._sync_profile_clock(batch_pin_slew.device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "result_all_valid_check_ms",
                    split_step_done_at - split_step_started_at,
                )
                split_step_started_at = split_step_done_at
            if torch.all(local_valid_mask):
                results[query_idx] = local_surrogate_values
                continue
            local_result = local_surrogate_values.clone()
            if profile_enabled:
                split_step_done_at = self._sync_profile_clock(batch_pin_slew.device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "result_clone_ms",
                    split_step_done_at - split_step_started_at,
                )
                split_step_started_at = split_step_done_at
            fallback_mask = ~local_valid_mask
            has_fallback = torch.any(fallback_mask)
            if profile_enabled:
                split_step_done_at = self._sync_profile_clock(batch_pin_slew.device)
                self._record_local_cell_aat_profile_value(
                    profile,
                    "result_fallback_mask_ms",
                    split_step_done_at - split_step_started_at,
                )
                split_step_started_at = split_step_done_at
            if has_fallback:
                fallback_values = run_lut_query(
                    queries[query_idx],
                    fallback_mask=fallback_mask,
                )
                assign_started_at = (
                    self._sync_profile_clock(batch_pin_slew.device)
                    if profile_enabled
                    else None
                )
                local_result[fallback_mask] = fallback_values
                if profile_enabled:
                    assign_done_at = self._sync_profile_clock(batch_pin_slew.device)
                    self._record_local_cell_aat_profile_value(
                        profile,
                        "result_fallback_assign_ms",
                        assign_done_at - assign_started_at,
                    )
            results[query_idx] = local_result
        if profile_enabled:
            split_now = self._sync_profile_clock(batch_pin_slew.device)
            self._record_local_cell_aat_profile_value(
                profile,
                "result_split_ms",
                split_now - split_started_at,
            )
            split_lut_ms = float(profile.get("fallback_lut_ms", 0.0)) - split_lut_ms_before
            split_non_lut_ms = max(0.0, (split_now - split_started_at) * 1000.0 - split_lut_ms)
            profile["result_split_non_lut_ms"] = (
                float(profile.get("result_split_non_lut_ms", 0.0)) + split_non_lut_ms
            )
        return finalize_results()

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
                                    cap_dims_actual,
                                    coeff_a_batch=None,
                                    coeff_b_batch=None,
                                    coeff_c_batch=None,
                                    coeff_d_batch=None,
                                    coeff_valid_batch=None,
                                    coeff_table_batch=None,
                                    ):
        """
        Performs vectorized 2D interpolation with LINEAR EXTRAPOLATION for a full batch.
        """
        if (
            lut_entry_2d_op is not None
            and coeff_table_batch is not None
        ):
            return lut_entry_2d_op.lut_entry_2d_coeff(
                input_trans,
                output_caps,
                trans_tables_batch,
                cap_tables_batch,
                coeff_table_batch,
                trans_dims_actual,
                cap_dims_actual,
            )
        if (
            lut_entry_2d_op is not None
            and self._use_lut_2d_native_op_for_tensor(input_trans)
            and coeff_a_batch is None
            and coeff_b_batch is None
            and coeff_c_batch is None
            and coeff_d_batch is None
            and coeff_valid_batch is None
        ):
            return lut_entry_2d_op.lut_entry_2d(
                input_trans,
                output_caps,
                trans_tables_batch,
                cap_tables_batch,
                lut_values_batch,
                trans_dims_actual,
                cap_dims_actual,
            )
        device = input_trans.device
        dtype = input_trans.dtype
        batch_size = input_trans.shape[0]
        denom_epsilon = torch.tensor(1e-12, device=device, dtype=dtype)

        # === 修改点 1: 确定每个输入点所在的区域 (低区外插, 内部插值, 高区外插) ===
        # 获取每个LUT的边界
        # 注意: .gather() 用于处理每个batch item的实际维度可能不同的情况
        trans_min = trans_tables_batch[:, 0]
        trans_max = trans_tables_batch.gather(1, (trans_dims_actual - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        cap_min = cap_tables_batch[:, 0]
        cap_max = cap_tables_batch.gather(1, (cap_dims_actual - 1).clamp(min=0).unsqueeze(1)).squeeze(1)

        # 判断是否需要外插
        is_trans_low_extrap = input_trans < trans_min
        is_trans_high_extrap = input_trans > trans_max
        is_cap_low_extrap = output_caps < cap_min
        is_cap_high_extrap = output_caps > cap_max

        # 找到用于插值的索引 (即使在外插区域，也先按常规方法找，后续会修正)
        trans_search_table = (
            trans_tables_batch
            if trans_tables_batch.is_contiguous()
            else trans_tables_batch.contiguous()
        )
        cap_search_table = (
            cap_tables_batch
            if cap_tables_batch.is_contiguous()
            else cap_tables_batch.contiguous()
        )
        trans_idx_padded = torch.searchsorted(trans_search_table, input_trans.unsqueeze(1), right=True).squeeze(1)
        cap_idx_padded = torch.searchsorted(cap_search_table, output_caps.unsqueeze(1), right=True).squeeze(1)

        max_trans_idx_actual = (trans_dims_actual - 1).clamp(min=0)
        max_cap_idx_actual = (cap_dims_actual - 1).clamp(min=0)

        # === 修改点 2: 根据区域动态选择用于计算的索引 ===
        # 对于高区外插，我们使用最后两个点 (N-1, N-2) 来定义斜率，所以高位索引是 max_idx
        # 对于低区外插，我们使用前两个点 (0, 1) 来定义斜率，所以高位索引是 1
        trans_idx_high = trans_idx_padded.clamp(min=1).clamp(max=max_trans_idx_actual)
        trans_idx_high = torch.where(is_trans_low_extrap, 1, trans_idx_high)
        trans_idx_low = (trans_idx_high - 1).clamp(min=0)

        cap_idx_high = cap_idx_padded.clamp(min=1).clamp( max=max_cap_idx_actual)
        cap_idx_high = torch.where(is_cap_low_extrap, 1, cap_idx_high)
        cap_idx_low = (cap_idx_high - 1).clamp(min=0)

        batch_indices = torch.arange(batch_size, device=device)
        if (
            coeff_a_batch is not None
            and coeff_b_batch is not None
            and coeff_c_batch is not None
            and coeff_d_batch is not None
            and coeff_valid_batch is not None
        ):
            coeff_value = (
                coeff_a_batch[batch_indices, trans_idx_low, cap_idx_low] * input_trans * output_caps
                + coeff_b_batch[batch_indices, trans_idx_low, cap_idx_low] * input_trans
                + coeff_c_batch[batch_indices, trans_idx_low, cap_idx_low] * output_caps
                + coeff_d_batch[batch_indices, trans_idx_low, cap_idx_low]
            )
            return coeff_value
        else:
            coeff_value = None

        # Gather Boundary Coordinates (t0, t1, c0, c1)
        # 这部分代码无需修改，因为它会根据上面计算出的索引自动选择正确的参考点
        t0 = trans_tables_batch[batch_indices, trans_idx_low]
        t1 = trans_tables_batch[batch_indices, trans_idx_high]
        c0 = cap_tables_batch[batch_indices, cap_idx_low]
        c1 = cap_tables_batch[batch_indices, cap_idx_high]

        # Gather Corner Values (v00, v01, v10, v11)
        # 这部分代码也无需修改
        stride = cap_dims_actual
        idx00 = trans_idx_low * stride + cap_idx_low
        idx01 = trans_idx_low * stride + cap_idx_high
        idx10 = trans_idx_high * stride + cap_idx_low
        idx11 = trans_idx_high * stride + cap_idx_high

        # 使用 gather 避免高级索引可能带来的性能问题，保持与原代码结构类似
        corner_indices = torch.stack([idx00, idx01, idx10, idx11], dim=1)
        corner_values = lut_values_batch.gather(1, corner_indices)
        v00, v01, v10, v11 = corner_values[:, 0], corner_values[:, 1], corner_values[:, 2], corner_values[:, 3]

        # === 修改点 3: 使用原始输入值进行计算，不再clamp ===
        # Perform Bilinear Interpolation / Extrapolation
        t_interval = t1 - t0
        c_interval = c1 - c0
        is_t_degenerate = torch.abs(t_interval) < denom_epsilon
        is_c_degenerate = torch.abs(c_interval) < denom_epsilon

        # 使用安全的间隔值避免除以零
        t_interval_safe = torch.where(is_t_degenerate, denom_epsilon, t_interval)
        c_interval_safe = torch.where(is_c_degenerate, denom_epsilon, c_interval)
        safe_denominator = t_interval_safe * c_interval_safe

        # 关键：这里使用原始的 input_trans 和 output_caps，而不是被 clamp 后的值
        # 这使得当输入超出 [t0, t1] 或 [c0, c1] 范围时，公式自动执行线性外插
        wa = (t1 - input_trans) * (c1 - output_caps)
        wb = (t1 - input_trans) * (output_caps - c0)
        wc = (input_trans - t0) * (c1 - output_caps)
        wd = (input_trans - t0) * (output_caps - c0)

        bilinear_val = (v00 * wa + v01 * wb + v10 * wc + v11 * wd) / safe_denominator

        # Handle degenerate cases using linear interpolation (lerp)
        # 这部分逻辑保持不变，用于处理参考点重合的特殊情况
        # 注意：这里的 lerp_factor 可能会超出 [0, 1]，这正是外插所需要的
        lerp_c_factor = (output_caps - c0) / c_interval_safe
        lerp_t_factor = (input_trans - t0) / t_interval_safe
        val_t_degenerate = torch.lerp(v00, v01, lerp_c_factor)
        val_c_degenerate = torch.lerp(v00, v10, lerp_t_factor)

        # Combine results based on which dimensions are degenerate
        legacy_value = torch.where(
            is_t_degenerate & is_c_degenerate, v00,
            torch.where(is_t_degenerate, val_t_degenerate,
                        torch.where(is_c_degenerate, val_c_degenerate, bilinear_val))
        )
        return legacy_value

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
                cap_dims_actual=luts_info.cap_dims_actual[idx_2d_batch],
                coeff_a_batch=(
                    luts_info.coeff_a[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_cache", False)
                    else None
                ),
                coeff_b_batch=(
                    luts_info.coeff_b[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_cache", False)
                    else None
                ),
                coeff_c_batch=(
                    luts_info.coeff_c[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_cache", False)
                    else None
                ),
                coeff_d_batch=(
                    luts_info.coeff_d[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_cache", False)
                    else None
                ),
                coeff_valid_batch=(
                    luts_info.coeff_valid[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_cache", False)
                    else None
                ),
                coeff_table_batch=(
                    luts_info.coeff_table[idx_2d_batch]
                    if getattr(luts_info, "use_2d_coeff_op", False)
                    else None
                ),
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

    def _entry_with_optional_surrogate(
        self,
        lib_cell_idxs,
        pin_slew,
        pin_net_caps,
        lib_arc_idxs,
        luts,
        node_ids,
        arc_type,
        use_surrogate,
    ):
        if not use_surrogate:
            return self.lut_entry_vectorized(
                lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts
            )

        surrogate = self._surrogate_entry(
            node_ids, pin_slew, pin_net_caps, lib_arc_idxs, arc_type
        )
        if surrogate is None:
            return self.lut_entry_vectorized(
                lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts
            )

        valid_mask, surrogate_values = surrogate
        if torch.all(valid_mask):
            return surrogate_values

        result = surrogate_values.clone()
        fallback_mask = ~valid_mask
        if torch.any(fallback_mask):
            result[fallback_mask] = self.lut_entry_vectorized(
                lib_cell_idxs[fallback_mask],
                pin_slew[fallback_mask],
                pin_net_caps[fallback_mask],
                lib_arc_idxs[fallback_mask],
                luts,
            )
        return result

    # --- Vectorized LUT Entry Functions (Updated to call vectorized lut_entry) ---
    # These now directly call the vectorized function, no vmap needed here.
    def r_delay_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs, node_ids=None, use_surrogate=True):
        luts = self.arcs_info.r_delay_luts
        return self._entry_with_optional_surrogate(
            lib_cell_idxs,
            pin_rtrans,
            pin_net_caps,
            lib_arc_idxs,
            luts,
            node_ids,
            1,
            use_surrogate,
        )

    def f_delay_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs, node_ids=None, use_surrogate=True):
        luts = self.arcs_info.f_delay_luts
        return self._entry_with_optional_surrogate(
            lib_cell_idxs,
            pin_ftrans,
            pin_net_caps,
            lib_arc_idxs,
            luts,
            node_ids,
            0,
            use_surrogate,
        )

    def r_tran_entry(self, lib_cell_idxs, pin_rtrans, pin_net_caps, lib_arc_idxs, node_ids=None, use_surrogate=True):
        luts = self.arcs_info.r_trans_luts
        return self._entry_with_optional_surrogate(
            lib_cell_idxs,
            pin_rtrans,
            pin_net_caps,
            lib_arc_idxs,
            luts,
            node_ids,
            3,
            use_surrogate,
        )

    def f_tran_entry(self, lib_cell_idxs, pin_ftrans, pin_net_caps, lib_arc_idxs, node_ids=None, use_surrogate=True):
        luts = self.arcs_info.f_trans_luts
        return self._entry_with_optional_surrogate(
            lib_cell_idxs,
            pin_ftrans,
            pin_net_caps,
            lib_arc_idxs,
            luts,
            node_ids,
            2,
            use_surrogate,
        )

    def calculate_clk2q_aat(self,
                        pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                        pin_net_cap_rise, pin_net_cap_fall,
                        use_surrogate=False):
        device = pin_rAAT.device

        # --- 数据准备部分 (与您的风格完全一致) ---
        level_cells = self.FF_ids
        start = self.flat_inst_arcs_by_level_start[0]
        end = self.flat_inst_arcs_by_level_start[1]
        inst_arcs = self.flat_inst_arcs_by_level[start: end]
            
        level_inst_arcs = inst_arcs

        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]
        lib_cell_idxs = level_inst_arcs[:, 2]
        lib_arc_idxs = level_inst_arcs[:, 3]
        timing_senses = level_inst_arcs[:, 4]
        timing_types = level_inst_arcs[:, 5]

        # Level-0 clk->Q arcs should use exported clock-pin slew, not the
        # runtime data-pin transition tensor. The latter is uninitialized for
        # clock pins during AAT forward propagation and collapses clk2q delay.
        pin_r_slew_in = self.clk_pin_rtran[arc_in_pins]
        pin_f_slew_in = self.clk_pin_ftran[arc_in_pins]
        pin_r_load_out = pin_net_cap_rise[arc_out_pins]
        pin_f_load_out = pin_net_cap_fall[arc_out_pins]

        # --- 初始化更新张量 (与您的风格一致) ---
        num_level_arcs = level_inst_arcs.shape[0]
        r_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
        f_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
        r_aat_updates = torch.full((num_level_arcs,), -torch.inf, device=device, dtype=self.dtype)
        f_aat_updates = torch.full((num_level_arcs,), -torch.inf, device=device, dtype=self.dtype)
        # --- 创建 Unate 和 TimingType 的掩码 ---
        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)
        is_non_unate = ~(is_pos_unate | is_neg_unate)

        is_rising_edge = (timing_types == 1)
        is_falling_edge = (timing_types == -1)
        is_both_edge = (timing_types == 0)

        # --- 按 Unate 类型分块计算 ---

        # A. Positive Unate (clk->Q)
        if torch.any(is_pos_unate):
            mask = is_pos_unate
            
            # Path 1: Rise->Rise (triggered by rising or both edge)
            sub_mask_rr = mask & (is_rising_edge | is_both_edge)
            if torch.any(sub_mask_rr):
                delay_rr = self.r_delay_entry(
                    lib_cell_idxs[sub_mask_rr],
                    pin_r_slew_in[sub_mask_rr],
                    pin_r_load_out[sub_mask_rr],
                    lib_arc_idxs[sub_mask_rr],
                    arc_out_pins[sub_mask_rr],
                    use_surrogate=use_surrogate,
                )
                tran_rr = self.r_tran_entry(
                    lib_cell_idxs[sub_mask_rr],
                    pin_r_slew_in[sub_mask_rr],
                    pin_r_load_out[sub_mask_rr],
                    lib_arc_idxs[sub_mask_rr],
                    arc_out_pins[sub_mask_rr],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_rr_delays[scatter_indices[sub_mask_rr]] = delay_rr
                r_aat_updates[sub_mask_rr] = 0 + delay_rr
                r_trans[sub_mask_rr] = tran_rr

            # Path 2: Fall->Fall (triggered by falling or both edge)
            sub_mask_ff = mask & (is_falling_edge | is_both_edge)
            if torch.any(sub_mask_ff):
                delay_ff = self.f_delay_entry(
                    lib_cell_idxs[sub_mask_ff],
                    pin_f_slew_in[sub_mask_ff],
                    pin_f_load_out[sub_mask_ff],
                    lib_arc_idxs[sub_mask_ff],
                    arc_out_pins[sub_mask_ff],
                    use_surrogate=use_surrogate,
                )
                tran_ff = self.f_tran_entry(
                    lib_cell_idxs[sub_mask_ff],
                    pin_f_slew_in[sub_mask_ff],
                    pin_f_load_out[sub_mask_ff],
                    lib_arc_idxs[sub_mask_ff],
                    arc_out_pins[sub_mask_ff],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_ff_delays[scatter_indices[sub_mask_ff]] = delay_ff
                f_aat_updates[sub_mask_ff] = 0 + delay_ff
                f_trans[sub_mask_ff] = tran_ff

        # B. Negative Unate (clk->QN)
        if torch.any(is_neg_unate):
            mask = is_neg_unate
            
            # Path 1: Rise->Fall (triggered by rising or both edge)
            sub_mask_rf = mask & (is_rising_edge | is_both_edge)
            if torch.any(sub_mask_rf):
                delay_rf = self.f_delay_entry(
                    lib_cell_idxs[sub_mask_rf],
                    pin_r_slew_in[sub_mask_rf],
                    pin_f_load_out[sub_mask_rf],
                    lib_arc_idxs[sub_mask_rf],
                    arc_out_pins[sub_mask_rf],
                    use_surrogate=use_surrogate,
                )
                tran_rf = self.f_tran_entry(
                    lib_cell_idxs[sub_mask_rf],
                    pin_r_slew_in[sub_mask_rf],
                    pin_f_load_out[sub_mask_rf],
                    lib_arc_idxs[sub_mask_rf],
                    arc_out_pins[sub_mask_rf],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_rf_delays[scatter_indices[sub_mask_rf]] = delay_rf
                f_aat_updates[sub_mask_rf] = 0 + delay_rf
                f_trans[sub_mask_rf] = tran_rf

            # Path 2: Fall->Rise (triggered by falling or both edge)
            sub_mask_fr = mask & (is_falling_edge | is_both_edge)
            if torch.any(sub_mask_fr):
                delay_fr = self.r_delay_entry(
                    lib_cell_idxs[sub_mask_fr],
                    pin_f_slew_in[sub_mask_fr],
                    pin_r_load_out[sub_mask_fr],
                    lib_arc_idxs[sub_mask_fr],
                    arc_out_pins[sub_mask_fr],
                    use_surrogate=use_surrogate,
                )
                tran_fr = self.r_tran_entry(
                    lib_cell_idxs[sub_mask_fr],
                    pin_f_slew_in[sub_mask_fr],
                    pin_r_load_out[sub_mask_fr],
                    lib_arc_idxs[sub_mask_fr],
                    arc_out_pins[sub_mask_fr],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_fr_delays[scatter_indices[sub_mask_fr]] = delay_fr
                r_aat_updates[sub_mask_fr] = 0 + delay_fr
                r_trans[sub_mask_fr] = tran_fr

        # C. Non-Unate
        if torch.any(is_non_unate):
            mask = is_non_unate
            
            # --- 计算所有可能的 AAT 和 Tran 更新值 ---
            # 1. 由 Rising Edge Clock 触发
            aat_rr_re = torch.full_like(r_trans, -torch.inf)
            aat_rf_re = torch.full_like(r_trans, -torch.inf)
            tran_rr_re = torch.zeros_like(r_trans)
            tran_rf_re = torch.zeros_like(r_trans)
            sub_mask_re = mask & (is_rising_edge | is_both_edge)
            if torch.any(sub_mask_re):
                delay_rr = self.r_delay_entry(
                    lib_cell_idxs[sub_mask_re],
                    pin_r_slew_in[sub_mask_re],
                    pin_r_load_out[sub_mask_re],
                    lib_arc_idxs[sub_mask_re],
                    arc_out_pins[sub_mask_re],
                    use_surrogate=use_surrogate,
                )
                delay_rf = self.f_delay_entry(
                    lib_cell_idxs[sub_mask_re],
                    pin_r_slew_in[sub_mask_re],
                    pin_f_load_out[sub_mask_re],
                    lib_arc_idxs[sub_mask_re],
                    arc_out_pins[sub_mask_re],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_rr_delays[scatter_indices[sub_mask_re]] = delay_rr
                # cell_arc_rf_delays[scatter_indices[sub_mask_re]] = delay_rf
                aat_rr_re[sub_mask_re] = 0 + delay_rr
                aat_rf_re[sub_mask_re] = 0+ delay_rf
                tran_rr_re[sub_mask_re] = self.r_tran_entry(
                    lib_cell_idxs[sub_mask_re],
                    pin_r_slew_in[sub_mask_re],
                    pin_r_load_out[sub_mask_re],
                    lib_arc_idxs[sub_mask_re],
                    arc_out_pins[sub_mask_re],
                    use_surrogate=use_surrogate,
                )
                tran_rf_re[sub_mask_re] = self.f_tran_entry(
                    lib_cell_idxs[sub_mask_re],
                    pin_r_slew_in[sub_mask_re],
                    pin_f_load_out[sub_mask_re],
                    lib_arc_idxs[sub_mask_re],
                    arc_out_pins[sub_mask_re],
                    use_surrogate=use_surrogate,
                )

            # 2. 由 Falling Edge Clock 触发
            aat_fr_fe = torch.full_like(r_trans, -torch.inf)
            aat_ff_fe = torch.full_like(r_trans, -torch.inf)
            tran_fr_fe = torch.zeros_like(r_trans)
            tran_ff_fe = torch.zeros_like(r_trans)
            sub_mask_fe = mask & (is_falling_edge | is_both_edge)
            if torch.any(sub_mask_fe):
                delay_fr = self.r_delay_entry(
                    lib_cell_idxs[sub_mask_fe],
                    pin_f_slew_in[sub_mask_fe],
                    pin_r_load_out[sub_mask_fe],
                    lib_arc_idxs[sub_mask_fe],
                    arc_out_pins[sub_mask_fe],
                    use_surrogate=use_surrogate,
                )
                delay_ff = self.f_delay_entry(
                    lib_cell_idxs[sub_mask_fe],
                    pin_f_slew_in[sub_mask_fe],
                    pin_f_load_out[sub_mask_fe],
                    lib_arc_idxs[sub_mask_fe],
                    arc_out_pins[sub_mask_fe],
                    use_surrogate=use_surrogate,
                )
                # cell_arc_fr_delays[scatter_indices[sub_mask_fe]] = delay_fr
                # cell_arc_ff_delays[scatter_indices[sub_mask_fe]] = delay_ff
                aat_fr_fe[sub_mask_fe] = 0 + delay_fr
                aat_ff_fe[sub_mask_fe] = 0 + delay_ff
                tran_fr_fe[sub_mask_fe] = self.r_tran_entry(
                    lib_cell_idxs[sub_mask_fe],
                    pin_f_slew_in[sub_mask_fe],
                    pin_r_load_out[sub_mask_fe],
                    lib_arc_idxs[sub_mask_fe],
                    arc_out_pins[sub_mask_fe],
                    use_surrogate=use_surrogate,
                )
                tran_ff_fe[sub_mask_fe] = self.f_tran_entry(
                    lib_cell_idxs[sub_mask_fe],
                    pin_f_slew_in[sub_mask_fe],
                    pin_f_load_out[sub_mask_fe],
                    lib_arc_idxs[sub_mask_fe],
                    arc_out_pins[sub_mask_fe],
                    use_surrogate=use_surrogate,
                )

            # --- 合并 Non-Unate 结果 ---
            # Rise AAT 更新: worst of (r->r from rising_clk, f->r from falling_clk)
            r_aat_updates[mask] = smooth_max(aat_rr_re[mask], aat_fr_fe[mask], alpha=SMOOTH_MAX_ALPHA)
            # Fall AAT 更新: worst of (r->f from rising_clk, f->f from falling_clk)
            f_aat_updates[mask] = smooth_max(aat_rf_re[mask], aat_ff_fe[mask], alpha=SMOOTH_MAX_ALPHA)
            # Rise Tran 更新
            r_trans[mask] = smooth_max(tran_rr_re[mask], tran_fr_fe[mask], alpha=SMOOTH_MAX_ALPHA)
            # Fall Tran 更新
            f_trans[mask] = smooth_max(tran_rf_re[mask], tran_ff_fe[mask], alpha=SMOOTH_MAX_ALPHA)

        # --- 4. 聚合更新 (与您的风格一致) ---
        pin_rAAT = scatter_timing_max(
            pin_rAAT,
            arc_out_pins,
            r_aat_updates,
            include_self=False,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fAAT = scatter_timing_max(
            pin_fAAT,
            arc_out_pins,
            f_aat_updates,
            include_self=False,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_out_pins.long(), r_trans, reduce="amax", include_self=False)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_out_pins.long(), f_trans, reduce="amax", include_self=False)

        net_in_pins = torch.unique(self.pin_net[arc_out_pins])
        return (net_in_pins, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran)
        
    # @torch.compile
    def calculate_cell_aat_level(self, inst_arcs,
                                pin_rAAT, pin_fAAT,
                                pin_rtran, pin_ftran,
                                pin_net_cap_rise, pin_net_cap_fall
                                , use_surrogate=None, profile_level=None,
                                need_delay_updates=True):
        """
        Optimized version of the AAT calculation function.
        It groups arcs by their 'unate' property (timing sense) to avoid
        redundant delay and transition calculations.
        """
        start_time = time.time()
        device = pin_rAAT.device

        # Get outpin and inpin ranges for all cells

        level_inst_arcs = inst_arcs
        static_level_data = (
            self._get_cell_aat_static_level_data(level_inst_arcs)
            if getattr(self, "use_cell_aat_static_cache", False)
            else None
        )

        if static_level_data is None:
            arc_in_pins = level_inst_arcs[:, 0]
            arc_out_pins = level_inst_arcs[:, 1]
            lib_cell_idxs = level_inst_arcs[:, 2]
            lib_arc_idxs = level_inst_arcs[:, 3]
            timing_senses = level_inst_arcs[:, 4]
        else:
            arc_in_pins = static_level_data["arc_in_pins"]
            arc_out_pins = static_level_data["arc_out_pins"]
            lib_cell_idxs = static_level_data["lib_cell_idxs"]
            lib_arc_idxs = static_level_data["lib_arc_idxs"]

        if torch.logical_and(arc_out_pins == 2485, arc_in_pins == 3200).any():
            logging.warning(f"flat arc index is: {torch.where(torch.logical_and(arc_out_pins == 2485, arc_in_pins == 3200))}")

        # 1. 准备输入条件 (Prepare inputs)
        pin_r_slew_in = pin_rtran[arc_in_pins]
        pin_f_slew_in = pin_ftran[arc_in_pins]
        pin_r_load_out = pin_net_cap_rise[arc_out_pins]
        pin_f_load_out = pin_net_cap_fall[arc_out_pins]

        # 2. 根据 Unate 类型进行分组 (Group arcs by unate type)
        if static_level_data is None:
            is_pos_unate = timing_senses == 1
            is_neg_unate = timing_senses == -1
            is_non_unate = ~(is_pos_unate | is_neg_unate)
            has_pos_unate = torch.any(is_pos_unate)
            has_neg_unate = torch.any(is_neg_unate)
            has_non_unate = torch.any(is_non_unate)
        else:
            is_pos_unate = static_level_data["is_pos_unate"]
            is_neg_unate = static_level_data["is_neg_unate"]
            is_non_unate = static_level_data["is_non_unate"]
            has_pos_unate = static_level_data["has_pos_unate"]
            has_neg_unate = static_level_data["has_neg_unate"]
            has_non_unate = static_level_data["has_non_unate"]

        num_level_arcs = level_inst_arcs.shape[0]

        r_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
        f_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
        # 初始化 AAT 更新张量
        r_aat_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
        f_aat_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
        # --- 3. 按需计算每种 Unate 类型的 Delay 和 Tran ---
        if need_delay_updates:
            delay_rr_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
            delay_rf_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
            delay_fr_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
            delay_ff_updates = torch.full_like(r_trans, -float('inf'), device=device, dtype=self.dtype)
        else:
            delay_rr_updates = delay_rf_updates = None
            delay_fr_updates = delay_ff_updates = None
        if use_surrogate is None:
            use_surrogate = self._cell_modeling_enabled()

        profile_enabled = bool(getattr(self, "timing_propagation_profile", False))
        level_profile = None
        if profile_enabled:
            level_profile = {
                "level": None if profile_level is None else int(profile_level),
                "num_arcs": int(num_level_arcs),
                "query_count": 0,
                "query_build_ms": 0.0,
                "query_eval_ms": 0.0,
                "cell_update_ms": 0.0,
                "unique_net_ms": 0.0,
            }
            for key in self._cell_aat_detail_count_keys():
                level_profile[key] = 0
            profile_timer = self._sync_profile_clock(device)

        level_queries = []

        def register_level_query(name, mask, pin_slew, pin_load, arc_type, luts):
            level_queries.append(
                {
                    "name": name,
                    "lib_cell_idxs": lib_cell_idxs[mask],
                    "pin_slew": pin_slew[mask],
                    "pin_net_caps": pin_load[mask],
                    "lib_arc_idxs": lib_arc_idxs[mask],
                    "luts": luts,
                    "node_ids": arc_out_pins[mask],
                    "arc_type": arc_type,
                }
            )

        if has_pos_unate:
            register_level_query(
                "pos_delay_rr",
                is_pos_unate,
                pin_r_slew_in,
                pin_r_load_out,
                1,
                self.arcs_info.r_delay_luts,
            )
            register_level_query(
                "pos_tran_rr",
                is_pos_unate,
                pin_r_slew_in,
                pin_r_load_out,
                3,
                self.arcs_info.r_trans_luts,
            )
            register_level_query(
                "pos_delay_ff",
                is_pos_unate,
                pin_f_slew_in,
                pin_f_load_out,
                0,
                self.arcs_info.f_delay_luts,
            )
            register_level_query(
                "pos_tran_ff",
                is_pos_unate,
                pin_f_slew_in,
                pin_f_load_out,
                2,
                self.arcs_info.f_trans_luts,
            )

        if has_neg_unate:
            register_level_query(
                "neg_delay_fr",
                is_neg_unate,
                pin_f_slew_in,
                pin_r_load_out,
                1,
                self.arcs_info.r_delay_luts,
            )
            register_level_query(
                "neg_tran_fr",
                is_neg_unate,
                pin_f_slew_in,
                pin_r_load_out,
                3,
                self.arcs_info.r_trans_luts,
            )
            register_level_query(
                "neg_delay_rf",
                is_neg_unate,
                pin_r_slew_in,
                pin_f_load_out,
                0,
                self.arcs_info.f_delay_luts,
            )
            register_level_query(
                "neg_tran_rf",
                is_neg_unate,
                pin_r_slew_in,
                pin_f_load_out,
                2,
                self.arcs_info.f_trans_luts,
            )

        if has_non_unate:
            register_level_query(
                "non_delay_rr",
                is_non_unate,
                pin_r_slew_in,
                pin_r_load_out,
                1,
                self.arcs_info.r_delay_luts,
            )
            register_level_query(
                "non_delay_fr",
                is_non_unate,
                pin_f_slew_in,
                pin_r_load_out,
                1,
                self.arcs_info.r_delay_luts,
            )
            register_level_query(
                "non_tran_rr",
                is_non_unate,
                pin_r_slew_in,
                pin_r_load_out,
                3,
                self.arcs_info.r_trans_luts,
            )
            register_level_query(
                "non_tran_fr",
                is_non_unate,
                pin_f_slew_in,
                pin_r_load_out,
                3,
                self.arcs_info.r_trans_luts,
            )
            register_level_query(
                "non_delay_ff",
                is_non_unate,
                pin_f_slew_in,
                pin_f_load_out,
                0,
                self.arcs_info.f_delay_luts,
            )
            register_level_query(
                "non_delay_rf",
                is_non_unate,
                pin_r_slew_in,
                pin_f_load_out,
                0,
                self.arcs_info.f_delay_luts,
            )
            register_level_query(
                "non_tran_ff",
                is_non_unate,
                pin_f_slew_in,
                pin_f_load_out,
                2,
                self.arcs_info.f_trans_luts,
            )
            register_level_query(
                "non_tran_rf",
                is_non_unate,
                pin_r_slew_in,
                pin_f_load_out,
                2,
                self.arcs_info.f_trans_luts,
            )

        if level_profile is not None:
            profile_now = self._sync_profile_clock(device)
            level_profile["query_build_ms"] = (profile_now - profile_timer) * 1000.0
            level_profile["query_count"] = len(level_queries)
            profile_timer = profile_now

        query_results = {
            query["name"]: result
            for query, result in zip(
                level_queries,
                self._entry_queries_with_optional_surrogate(
                    level_queries,
                    use_surrogate,
                    profile=level_profile,
                ),
            )
        }
        if level_profile is not None:
            profile_now = self._sync_profile_clock(device)
            level_profile["query_eval_ms"] = (profile_now - profile_timer) * 1000.0
            profile_timer = profile_now
        
        # A. Positive Unate (r->r, f->f)
        if has_pos_unate:
            mask = is_pos_unate
            delay_rr = query_results["pos_delay_rr"]
            tran_rr = query_results["pos_tran_rr"]
            delay_ff = query_results["pos_delay_ff"]
            tran_ff = query_results["pos_tran_ff"]

            if need_delay_updates:
                delay_rr_updates[mask] = delay_rr
                delay_ff_updates[mask] = delay_ff
            r_trans[mask] = tran_rr
            f_trans[mask] = tran_ff
            r_aat_updates[mask] = pin_rAAT[arc_in_pins[mask]] + delay_rr
            f_aat_updates[mask] = pin_fAAT[arc_in_pins[mask]] + delay_ff

        # B. Negative Unate (f->r, r->f)
        if has_neg_unate:
            mask = is_neg_unate
            delay_fr = query_results["neg_delay_fr"]
            tran_fr = query_results["neg_tran_fr"]
            delay_rf = query_results["neg_delay_rf"]
            tran_rf = query_results["neg_tran_rf"]

            if need_delay_updates:
                delay_fr_updates[mask] = delay_fr
                delay_rf_updates[mask] = delay_rf
            r_trans[mask] = tran_fr
            f_trans[mask] = tran_rf
            r_aat_updates[mask] = pin_fAAT[arc_in_pins[mask]] + delay_fr # <-- 使用 pin_fAAT
            f_aat_updates[mask] = pin_rAAT[arc_in_pins[mask]] + delay_rf # <-- 使用 pin_rAAT

        # C. Non-Unate (worst of all applicable cases)
        if has_non_unate:
            mask = is_non_unate
            delay_rr_non = query_results["non_delay_rr"]
            delay_fr_non = query_results["non_delay_fr"]
            tran_rr_non = query_results["non_tran_rr"]
            tran_fr_non = query_results["non_tran_fr"]
            if need_delay_updates:
                delay_fr_updates[mask] = delay_fr_non
                delay_rr_updates[mask] = delay_rr_non

            # unate_r_delays = torch.maximum(delay_rr_non, delay_fr_non)
            r_trans[mask] = smooth_max(tran_rr_non, tran_fr_non, alpha=SMOOTH_MAX_ALPHA)

            aat_rr = pin_rAAT[arc_in_pins[mask]] + delay_rr_non
            aat_fr = pin_fAAT[arc_in_pins[mask]] + delay_fr_non
            r_aat_updates[mask] = smooth_max(aat_rr, aat_fr, alpha=SMOOTH_MAX_ALPHA)

            # Fall output calculation (worst of f->f and r->f)
            delay_ff_non = query_results["non_delay_ff"]
            delay_rf_non = query_results["non_delay_rf"]
            tran_ff_non = query_results["non_tran_ff"]
            tran_rf_non = query_results["non_tran_rf"]

            f_trans[mask] = smooth_max(tran_ff_non, tran_rf_non, alpha=SMOOTH_MAX_ALPHA)

            if need_delay_updates:
                delay_ff_updates[mask] = delay_ff_non
                delay_rf_updates[mask] = delay_rf_non

            aat_ff = pin_fAAT[arc_in_pins[mask]] + delay_ff_non
            aat_rf = pin_rAAT[arc_in_pins[mask]] + delay_rf_non
            f_aat_updates[mask] = smooth_max(aat_ff, aat_rf, alpha=SMOOTH_MAX_ALPHA)

        # --- 4. 存储和更新 (Store and Update) ---
        # The logic from here remains the same, as r_delays, f_delays, etc. are now fully populated.

        # cell_arc_r_delays.scatter_(0, scatter_indices.long(), r_delays)
        # cell_arc_f_delays.scatter_(0, scatter_indices.long(), f_delays)

        pin_rAAT = scatter_timing_max(
            pin_rAAT,
            arc_out_pins,
            r_aat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fAAT = scatter_timing_max(
            pin_fAAT,
            arc_out_pins,
            f_aat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_out_pins.long(), r_trans, reduce="amax", include_self=True)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_out_pins.long(), f_trans, reduce="amax", include_self=True)
        if level_profile is not None:
            profile_now = self._sync_profile_clock(device)
            level_profile["cell_update_ms"] = (profile_now - profile_timer) * 1000.0
            profile_timer = profile_now
        
        # t_cell_arc_rr_delays = torch.max(cell_arc_rr_delays, delay_rr_updates)
        # t_cell_arc_fr_delays = torch.max(cell_arc_fr_delays, delay_fr_updates)
        # t_cell_arc_rf_delays = torch.max(cell_arc_rf_delays, delay_rf_updates)
        # t_cell_arc_ff_delays = torch.max(cell_arc_ff_delays, delay_ff_updates)

        net_in_pins = torch.unique(self.pin_net[arc_out_pins])
        if level_profile is not None:
            profile_now = self._sync_profile_clock(device)
            level_profile["unique_net_ms"] = (profile_now - profile_timer) * 1000.0
            self._record_cell_aat_level_profile(level_profile)
        logging.debug(f"Cell AAT Level Time: {time.time() - start_time:.4f}s")
        return (
            net_in_pins,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            delay_rr_updates,
            delay_fr_updates,
            delay_rf_updates,
            delay_ff_updates,
        )

    def calculate_net_aat_level(self, curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                                pin_net_delay_rise, pin_net_delay_fall,
                                pin_net_impulse_rise, pin_net_impulse_fall):
        start_time = time.time()
        net_arcs_starts = self.net_flat_arcs_start[curnets]
        net_arcs_ends = self.net_flat_arcs_start[curnets + 1]
        num_net_fopins = net_arcs_ends - net_arcs_starts
        if num_net_fopins.numel() == 0:
            return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

        # Pack the CSR ranges in net order without padding to maximum fanout.
        packed_starts = num_net_fopins.cumsum(0) - num_net_fopins
        net_arcs_global_indices = torch.repeat_interleave(
            net_arcs_starts - packed_starts, num_net_fopins)
        net_arcs_global_indices += torch.arange(
            net_arcs_global_indices.numel(), device=pin_rAAT.device)
        net_arcs_level = self.net_flat_arcs[net_arcs_global_indices]

        arc_fopins = net_arcs_level[:, 1]
        arc_fipins = net_arcs_level[:, 0]

        wire_delays_rise = pin_net_delay_rise[arc_fopins]
        wire_delays_fall = pin_net_delay_fall[arc_fopins]
        impulse_rise = pin_net_impulse_rise[arc_fopins]
        impulse_fall = pin_net_impulse_fall[arc_fopins]

        r_aat_updates = pin_rAAT[arc_fipins] + wire_delays_rise
        f_aat_updates = pin_fAAT[arc_fipins] + wire_delays_fall
        r_tran_updates = torch.sqrt(
            torch.clamp_min(pin_rtran[arc_fipins]**2 + impulse_rise, SLEW_SQRT_EPS))
        f_tran_updates = torch.sqrt(
            torch.clamp_min(pin_ftran[arc_fipins]**2 + impulse_fall, SLEW_SQRT_EPS))

        pin_rAAT = scatter_timing_max(
            pin_rAAT,
            arc_fopins,
            r_aat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fAAT = scatter_timing_max(
            pin_fAAT,
            arc_fopins,
            f_aat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_rtran = torch.scatter_reduce(pin_rtran, 0, arc_fopins.long(
        ), r_tran_updates, reduce="amax", include_self=True)
        pin_ftran = torch.scatter_reduce(pin_ftran, 0, arc_fopins.long(
        ), f_tran_updates, reduce="amax", include_self=True)
        logging.debug(f"Net AAT Level Time: {time.time() - start_time:.4f}s")
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def _calculate_net_aat_level_dynamic(
        self,
        curnets,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        pin_net_delay_rise,
        pin_net_delay_fall,
        pin_net_impulse_rise,
        pin_net_impulse_fall,
        dynamic_net_provider,
        level_id=0,
        level_view_epoch=0,
    ):
        if dynamic_net_provider is None or not dynamic_net_provider.has_dynamic_nets(
            curnets,
            level_id=level_id,
            level_view_epoch=level_view_epoch,
        ):
            return self.calculate_net_aat_level(
                curnets,
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                pin_net_delay_rise,
                pin_net_delay_fall,
                pin_net_impulse_rise,
                pin_net_impulse_fall,
            )
        return dynamic_net_provider.propagate_net_aat_level(
            net_ids=curnets,
            pin_rAAT=pin_rAAT,
            pin_fAAT=pin_fAAT,
            pin_rtran=pin_rtran,
            pin_ftran=pin_ftran,
            base_pin_net_delay_rise=pin_net_delay_rise,
            base_pin_net_delay_fall=pin_net_delay_fall,
            base_pin_net_impulse_rise=pin_net_impulse_rise,
            base_pin_net_impulse_fall=pin_net_impulse_fall,
            static_calculate_net_aat_level=self.calculate_net_aat_level,
            level_id=level_id,
            level_view_epoch=level_view_epoch,
        )

    # @torch.compile
    def calculate_cell_rat_level(
        self,
        inst_arcs,
        pin_rRAT,
        pin_fRAT,
        cell_arc_rr_delays,
        cell_arc_fr_delays,
        cell_arc_rf_delays,
        cell_arc_ff_delays,
    ):
        start_time = time.time()
        device = pin_rRAT.device

        # --- 1. 获取当前层级的所有时序弧信息 (与AAT函数中相同) ---
        level_inst_arcs = inst_arcs

        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]
        timing_senses = level_inst_arcs[:, 4]

        # --- 2. 获取预先计算好的延迟 ---

        rr_delays = cell_arc_rr_delays
        ff_delays = cell_arc_ff_delays
        rf_delays = cell_arc_rf_delays
        fr_delays = cell_arc_fr_delays

        # --- 3. 根据 Unate 类型计算 RAT 更新值 ---

        # 初始化RAT更新张量为一个极大值，因为我们要取最小值
        r_rat_updates = torch.full_like(rr_delays, float('inf'), device=self.device, dtype=self.dtype)
        f_rat_updates = torch.full_like(rr_delays, float('inf'), device=self.device, dtype=self.dtype)

        # 根据Unate类型进行分组
        is_pos_unate = (timing_senses == 1)
        is_neg_unate = (timing_senses == -1)

        # A. Positive Unate (r_out -> r_in, f_out -> f_in)
        if torch.any(is_pos_unate):
            mask = is_pos_unate
            # r_delays[mask] 对应 delay_rr
            r_rat_updates[mask] = pin_rRAT[arc_out_pins[mask]] - rr_delays[mask]
            # f_delays[mask] 对应 delay_ff
            f_rat_updates[mask] = pin_fRAT[arc_out_pins[mask]] - ff_delays[mask]

        # B. Negative Unate (r_out -> f_in, f_out -> r_in)
        if torch.any(is_neg_unate):
            mask = is_neg_unate
            # 要计算输入端的 rise RAT (r_rat_updates)，需要看输出端的 fall RAT
            # 对应的延迟是 rise-to-fall，在前向计算时存在 f_delays 中
            r_rat_updates[mask] = pin_fRAT[arc_out_pins[mask]] - rf_delays[mask]

            # 要计算输入端的 fall RAT (f_rat_updates)，需要看输出端的 rise RAT
            # 对应的延迟是 fall-to-rise，在前向计算时存在 r_delays 中
            f_rat_updates[mask] = pin_rRAT[arc_out_pins[mask]] - fr_delays[mask]

        # C. Non-Unate (需要同时满足所有情况)
        # 对于RAT，一个输入引脚的要求是最坏（早）的。
        # 一个输入引脚的 rise RAT，需要满足它所驱动的所有弧（r->r, r->f）的要求
        # scatter_reduce 的 amin 操作会自动处理 Non-Unate 的聚合，所以我们只需为每条弧计算
        is_non_unate = ~(is_pos_unate | is_neg_unate)
        if torch.any(is_non_unate):
            mask = is_non_unate
            # r_in -> r_out 弧的要求

            r_rat_updates[mask] = torch.minimum(
                pin_rRAT[arc_out_pins[mask]] - rr_delays[mask],
                pin_fRAT[arc_out_pins[mask]] - rf_delays[mask],
            )  # r_delays for non-unate is max(delay_rr, delay_fr)
            # r_in -> f_out 弧的要求
            f_rat_updates[mask] = torch.minimum(
                pin_rRAT[arc_out_pins[mask]] - fr_delays[mask],
                pin_fRAT[arc_out_pins[mask]] - ff_delays[mask],
            )  # f_delays for non-unate is max(delay_ff, delay_rf)

        # --- 4. 聚合更新 ---
        pin_rRAT = scatter_timing_min(
            pin_rRAT,
            arc_in_pins,
            r_rat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fRAT = scatter_timing_min(
            pin_fRAT,
            arc_in_pins,
            f_rat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )

        cur_endpoints = torch.unique(arc_in_pins)
        logging.debug(f"Cell RAT Level Time: {time.time() - start_time:.4f}s")
        return cur_endpoints, pin_rRAT, pin_fRAT

    def calculate_net_rat_level(self, cur_endpoint, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall):

        curnets = self.pin_net[cur_endpoint]

        arc_fipins = self.net2driver_pin_map[curnets]
        has_driver = arc_fipins != -1
        cur_endpoint = cur_endpoint[has_driver]
        arc_fipins = arc_fipins[has_driver]
        wire_delays_rise = pin_net_delay_rise[cur_endpoint]
        wire_delays_fall = pin_net_delay_fall[cur_endpoint]
        # assert pin_rRAT[cur_endpoint].max(
        #     ) <= 1e8, "Negative r_rat_updates detected"
        r_rat_updates = pin_rRAT[cur_endpoint] - wire_delays_rise
        f_rat_updates = pin_fRAT[cur_endpoint] - wire_delays_fall
        # assert r_rat_updates.max() < 5e4, "r_rat_updates exceed expected range"
        # assert f_rat_updates.max() < 5e4, "f_rat_updates exceed expected range"

        pin_rRAT = scatter_timing_min(
            pin_rRAT,
            arc_fipins,
            r_rat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fRAT = scatter_timing_min(
            pin_fRAT,
            arc_fipins,
            f_rat_updates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        # assert pin_rRAT[arc_fipins].max(
        #     ) <= 1e8, "Negative r_rat_updates detected after scatter_reduce"
        return pin_rRAT, pin_fRAT

    def calculate_setup_rat(self, pin_rRAT, pin_fRAT, clk_pin_rtran, clk_pin_ftran, pin_rtran, pin_ftran):

        if getattr(self, "_critical_path_snapshot_requested", False):
            self._critical_path_constraint_state = None
        if self.endpoints_constraint_arcs.numel() == 0:
            return pin_rRAT, pin_fRAT

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
        r_delays_non_unate = smooth_max(rr_setup_time, fr_setup_time, alpha=SMOOTH_MAX_ALPHA)
        f_delays_non_unate = smooth_max(ff_setup_time, rf_setup_time, alpha=SMOOTH_MAX_ALPHA)
        r_setup_time = torch.where(is_pos_unate, rr_setup_time,
                                   torch.where(is_neg_unate, fr_setup_time, r_delays_non_unate))
        f_setup_time = torch.where(is_pos_unate, ff_setup_time,
                                   torch.where(is_neg_unate, rf_setup_time, f_delays_non_unate))

        rise_rat_candidates = pin_rRAT[arc_out_pins] - r_setup_time
        fall_rat_candidates = pin_fRAT[arc_out_pins] - f_setup_time
        rise_rat_candidates, fall_rat_candidates = qualify_check_candidates(
            rise_rat_candidates, fall_rat_candidates,
            getattr(self, "endpoints_constraint_max_valid", None),
        )
        if getattr(self, "_critical_path_snapshot_requested", False):
            qualified_rows = getattr(self, "endpoints_constraint_max_valid", None)
            qualified_rows = torch.ones_like(arc_out_pins, dtype=torch.bool) if qualified_rows is None else qualified_rows.any(dim=1)
            self._critical_path_constraint_state = {
                "endpoint_pins": arc_out_pins[qualified_rows].detach(),
                "test_ids": torch.arange(
                    arc_out_pins.numel(),
                    dtype=torch.int64,
                    device=arc_out_pins.device,
                )[qualified_rows],
                "rise_rat": rise_rat_candidates[qualified_rows].detach(),
                "fall_rat": fall_rat_candidates[qualified_rows].detach(),
            }

        # assert pin_rRAT[arc_out_pins].min(
        # ) >= 0, "Negative r_rat_updates detected"

        pin_rRAT = scatter_timing_min(
            pin_rRAT,
            arc_out_pins,
            rise_rat_candidates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )
        pin_fRAT = scatter_timing_min(
            pin_fRAT,
            arc_out_pins,
            fall_rat_candidates,
            include_self=True,
            mode=self.timing_aggregation_mode,
            tau_ps=self.timing_aggregation_tau_ps,
        )

        return pin_rRAT, pin_fRAT

    def forward(
        self,
        pin_net_delays,
        pin_net_impulses,
        pin_net_caps,
        surrogate_mode: Optional[str] = None,
        dynamic_net_arc_inputs=None,
        dynamic_net_subgraph_inputs=None,
        dynamic_net_provider=None,
    ):
        start_time = time.time()

        def profile_clock():
            if self.timing_propagation_profile:
                return self._sync_profile_clock(self.device)
            return time.perf_counter()

        profile_started_at = profile_clock()
        stage_runtime_ms = {}
        return_device = pin_net_delays["rise"].device
        if dynamic_net_arc_inputs is not None and dynamic_net_subgraph_inputs is not None:
            raise ValueError(
                "dynamic_net_arc_inputs and dynamic_net_subgraph_inputs are mutually exclusive"
            )
        if dynamic_net_provider is not None and (
            dynamic_net_arc_inputs is not None or dynamic_net_subgraph_inputs is not None
        ):
            raise ValueError(
                "dynamic_net_provider is mutually exclusive with dynamic_net_arc_inputs "
                "and dynamic_net_subgraph_inputs"
            )
        if dynamic_net_subgraph_inputs is not None:
            dynamic_net_subgraph_inputs = dict(dynamic_net_subgraph_inputs)
            build_dynamic_net_arc_inputs = dynamic_net_subgraph_inputs.pop(
                "build_dynamic_net_arc_inputs",
                dynamic_net_subgraph_inputs.pop("builder", None),
            )
            if build_dynamic_net_arc_inputs is None:
                raise ValueError(
                    "dynamic_net_subgraph_inputs requires a caller-provided "
                    "build_dynamic_net_arc_inputs callable; TimingPropagation "
                    "does not import net_subgraph_timing directly"
                )
            dynamic_net_subgraph_inputs.setdefault("num_pins", pin_net_delays["rise"].numel())
            dynamic_net_subgraph_inputs.setdefault("dtype", pin_net_delays["rise"].dtype)
            native_device = _resolve_native_net_subgraph_timing_device(
                timing_device=return_device,
                requested_device=dynamic_net_subgraph_inputs.get("device"),
            )
            dynamic_net_subgraph_inputs["device"] = native_device
            dynamic_net_arc_inputs = build_dynamic_net_arc_inputs(
                **dynamic_net_subgraph_inputs
            )
            dynamic_net_arc_inputs["net_subgraph_native_device"] = str(native_device)
            dynamic_net_arc_inputs = _move_dynamic_net_arc_inputs_to_device(
                dynamic_net_arc_inputs,
                return_device,
            )
        pin_net_delays, pin_net_impulses, pin_net_caps = self._apply_dynamic_net_arc_inputs(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            dynamic_net_arc_inputs,
        )
        self._begin_forward_runtime_state()
        write_crash_stage_marker(
            "timing_propagation_forward",
            "start",
            resolved_device=str(self.device),
            return_device=str(return_device),
            surrogate_mode=surrogate_mode,
        )

        def record_stage(name, stage_started_at):
            profile_now = profile_clock()
            if self.timing_propagation_profile:
                stage_runtime_ms[name] = (profile_now - stage_started_at) * 1000.0
            return profile_now

        stage_started_at = profile_clock()
        self._validate_runtime_indices()
        clk2q_use_surrogate, cell_use_surrogate = self._resolve_surrogate_mode(surrogate_mode)
        # RC timing ops feed net delay / impulse in ps. The cap vectors are in
        # pF and may live on the larger RC-graph domain (design pins followed by
        # appended Steiner vertices).
        pin_net_delay_rise = self._forward_input_tensor(pin_net_delays['rise'])
        pin_net_delay_fall = self._forward_input_tensor(pin_net_delays['fall'])
        pin_net_impulse_rise = self._forward_input_tensor(pin_net_impulses['rise'])
        pin_net_impulse_fall = self._forward_input_tensor(pin_net_impulses['fall'])
        pin_net_cap_rise = self._forward_input_tensor(pin_net_caps['rise'])
        pin_net_cap_fall = self._forward_input_tensor(pin_net_caps['fall'])
        if dynamic_net_provider is not None and hasattr(dynamic_net_provider, "apply_dynamic_net_cap_overlay"):
            pin_net_cap_rise, pin_net_cap_fall = dynamic_net_provider.apply_dynamic_net_cap_overlay(
                pin_net_cap_rise=pin_net_cap_rise,
                pin_net_cap_fall=pin_net_cap_fall,
            )
        critical_path_snapshot_requested = bool(
            getattr(self, "_critical_path_snapshot_requested", False)
        )
        if dynamic_net_provider is not None:
            dynamic_net_provider.begin_critical_path_net_delay_snapshot(
                base_pin_net_delay_rise=pin_net_delay_rise,
                base_pin_net_delay_fall=pin_net_delay_fall,
            )
        self.last_fast_loop_skipped_full_rat = False
        stage_started_at = record_stage("input_clone", stage_started_at)

        device = self.device
        dtype = pin_net_delay_rise.dtype
        forward_contract = self._build_device_contract(
            {
                "pin_net_delay_rise": pin_net_delay_rise,
                "pin_net_delay_fall": pin_net_delay_fall,
                "pin_net_impulse_rise": pin_net_impulse_rise,
                "pin_net_impulse_fall": pin_net_impulse_fall,
                "pin_net_cap_rise": pin_net_cap_rise,
                "pin_net_cap_fall": pin_net_cap_fall,
                "inrdelays": self.inrdelays,
                "end_points": self.end_points,
                "flat_inst_arcs_by_level": self.flat_inst_arcs_by_level,
            },
            scope="forward",
        )
        self.last_device_contract = forward_contract
        self._assert_device_contract(forward_contract)

        # --- Initialization ---
        # Sentinel AAT/RAT values also live in ps, matching all propagated time
        # tensors in this module.
        inf_val = torch.tensor(2e8, device=self.device, dtype=self.dtype)
        pin_rAAT = torch.full((self.num_pins,), -inf_val, device=device, dtype=dtype)
        pin_fAAT = torch.full((self.num_pins,), -inf_val, device=device, dtype=dtype)
        pin_rtran = torch.zeros(self.num_pins, device=device, dtype=dtype)
        pin_ftran = torch.zeros(self.num_pins, device=device, dtype=dtype)
        pin_rRAT = torch.full((self.num_pins,), inf_val,
                              device=self.device, dtype=self.dtype)
        pin_fRAT = torch.full((self.num_pins,), inf_val,
                              device=self.device, dtype=self.dtype)
        pin_rRAT[self.end_points] = self.endpoints_rRAT.to(
            device=self.device,
            dtype=self.dtype,
        )
        pin_fRAT[self.end_points] = self.endpoints_fRAT.to(
            device=self.device,
            dtype=self.dtype,
        )

        skip_full_rat = self._skip_full_rat_propagation_in_fast_loop()
        need_cell_arc_delay_cache = (
            not skip_full_rat
            or getattr(self, "_critical_path_snapshot_requested", False)
        )
        num_arcs_total = self.flat_inst_arcs_by_level.shape[0]
        # Cell arc delay tensors are cached in ps and indexed by
        # flat_inst_arcs_by_level order.
        if need_cell_arc_delay_cache:
            cell_arc_rr_delays = torch.zeros(
                num_arcs_total, device=device, dtype=dtype)
            cell_arc_fr_delays = torch.zeros(
                num_arcs_total, device=device, dtype=dtype)
            cell_arc_rf_delays = torch.zeros(
                num_arcs_total, device=device, dtype=dtype)
            cell_arc_ff_delays = torch.zeros(
                num_arcs_total, device=device, dtype=dtype)
        else:
            cell_arc_rr_delays = cell_arc_fr_delays = None
            cell_arc_rf_delays = cell_arc_ff_delays = None

        pin_rAAT[self.start_points] = self.inrdelays
        pin_fAAT[self.start_points] = self.infdelays
        pin_rtran[self.start_points] = self.inrtrans
        pin_ftran[self.start_points] = self.inftrans
        # pin_net_cap = pin_net_cap.clone()  # Clone to avoid modifying the original tensor
        # pin_net_cap[self.end_points] = pin_net_cap[self.end_points] + self.outcaps
        stage_started_at = record_stage("initialization", stage_started_at)

        if getattr(self, "log_stage_progress", True):
            logging.info("TimingPropagation stage: clk2q")
        cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = self.calculate_clk2q_aat(
            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
            pin_net_cap_rise, pin_net_cap_fall,
            use_surrogate=clk2q_use_surrogate,
        )
        stage_started_at = record_stage("clk2q_aat", stage_started_at)

        pi_nets = torch.unique(self.pin_net[self.start_points])
        if getattr(self, "log_stage_progress", True):
            logging.info("TimingPropagation stage: pi_net_aat")
        pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = self._calculate_net_aat_level_dynamic(
            pi_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
            pin_net_delay_rise, pin_net_delay_fall,
            pin_net_impulse_rise, pin_net_impulse_fall,
            dynamic_net_provider,
            level_id=0,
            level_view_epoch=getattr(self, "_dynamic_net_level_view_epoch", 0),
        )
        stage_started_at = record_stage("pi_net_aat", stage_started_at)

        (
            working_flat_arcs_by_level,
            working_flat_arc_level_start,
            working_flat_arc_indices,
            traversal_pruning_active,
        ) = self._build_traversal_pruning_working_view(device)

        traversal_signature = None
        if traversal_pruning_active:
            pruning_stats = getattr(self, "last_traversal_pruning_stats", {}) or {}
            traversal_signature = (
                int(pruning_stats.get("iteration", -1) or -1),
                bool(pruning_stats.get("refreshed_this_iteration", False)),
                int(working_flat_arc_indices.numel()),
            )
        if traversal_signature != getattr(self, "_dynamic_net_level_view_signature", None):
            self._dynamic_net_level_view_epoch = int(
                getattr(self, "_dynamic_net_level_view_epoch", 0)
            ) + 1
            self._dynamic_net_level_view_signature = traversal_signature

        def level_offset(offsets, index):
            value = offsets[index]
            if torch.is_tensor(value):
                return int(value.detach().item())
            return int(value)

        cell_aat_started_at = profile_clock()
        cell_aat_profile = self._empty_cell_aat_profile(enabled=self.timing_propagation_profile)
        self._active_cell_aat_profile = cell_aat_profile if self.timing_propagation_profile else None
        self._build_cell_aat_surrogate_support_cache(
            working_flat_arcs_by_level,
            working_flat_arc_level_start,
            profile=cell_aat_profile if self.timing_propagation_profile else None,
        )
        for level in range(1, len(working_flat_arc_level_start) - 1):
            start = level_offset(working_flat_arc_level_start, level)
            end = level_offset(working_flat_arc_level_start, level + 1)
            if start >= end:
                continue
            inst_arcs = working_flat_arcs_by_level[start: end]
            abs_arc_indices = working_flat_arc_indices[start: end]
            if level == 1 or level % 50 == 0:
                if getattr(self, "log_stage_progress", True):
                    logging.info("TimingPropagation stage: cell_aat_level %d", level)
            (
                cur_nets,
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                t_cell_arc_rr_delays,
                t_cell_arc_fr_delays,
                t_cell_arc_rf_delays,
                t_cell_arc_ff_delays,
            ) = self.calculate_cell_aat_level(
                inst_arcs,
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                pin_net_cap_rise,
                pin_net_cap_fall,
                use_surrogate=cell_use_surrogate,
                profile_level=level,
                need_delay_updates=need_cell_arc_delay_cache,
            )

            profile_timer = self._sync_profile_clock(device)
            if need_cell_arc_delay_cache:
                cell_arc_rr_delays.scatter_(0, abs_arc_indices, t_cell_arc_rr_delays)
                cell_arc_fr_delays.scatter_(0, abs_arc_indices, t_cell_arc_fr_delays)
                cell_arc_rf_delays.scatter_(0, abs_arc_indices, t_cell_arc_rf_delays)
                cell_arc_ff_delays.scatter_(0, abs_arc_indices, t_cell_arc_ff_delays)
            if self.timing_propagation_profile:
                profile_now = self._sync_profile_clock(device)
                if need_cell_arc_delay_cache:
                    self._record_cell_aat_profile_value(
                        "arc_delay_scatter_ms",
                        profile_now - profile_timer,
                    )
                profile_timer = profile_now
            
            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = self._calculate_net_aat_level_dynamic(
                cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran,
                pin_net_delay_rise, pin_net_delay_fall,
                pin_net_impulse_rise, pin_net_impulse_fall,
                dynamic_net_provider,
                level_id=level,
                level_view_epoch=getattr(self, "_dynamic_net_level_view_epoch", 0),
            )
            if self.timing_propagation_profile:
                profile_now = self._sync_profile_clock(device)
                self._record_cell_aat_profile_value(
                    "net_aat_ms",
                    profile_now - profile_timer,
                )
        if self.timing_propagation_profile:
            profile_now = self._sync_profile_clock(device)
            stage_runtime_ms["cell_aat_levels"] = (
                profile_now - cell_aat_started_at
            ) * 1000.0
            cell_aat_profile["total_ms"] = stage_runtime_ms["cell_aat_levels"]
            top_levels = sorted(
                cell_aat_profile["top_levels"],
                key=lambda item: (
                    float(item.get("query_build_ms", 0.0))
                    + float(item.get("query_eval_ms", 0.0))
                    + float(item.get("cell_update_ms", 0.0))
                    + float(item.get("unique_net_ms", 0.0))
                ),
                reverse=True,
            )[:20]
            cell_aat_profile["top_levels"] = top_levels
            self._format_surrogate_unsupported_top_keys(cell_aat_profile)
        self._active_cell_aat_profile = None
        stage_started_at = profile_clock()
        # --- Calculate RAT Levels ---

        self.virtual_buffer_drv = (
            None if dynamic_net_provider is None
            else dynamic_net_provider.virtual_buffer_drv_tensors()
        )

        if dynamic_net_provider is not None:
            # AAT, RAT and path extraction must use the same buffered net delay.
            pin_net_delay_rise, pin_net_delay_fall = (
                dynamic_net_provider.consume_critical_path_net_delays(
                    base_pin_net_delay_rise=pin_net_delay_rise,
                    base_pin_net_delay_fall=pin_net_delay_fall,
                )
            )

        if getattr(self, "log_stage_progress", True):
            logging.info("TimingPropagation stage: setup_rat")
        pin_rRAT, pin_fRAT = self.calculate_setup_rat(
            pin_rRAT, pin_fRAT, self.clk_pin_rtran, self.clk_pin_ftran, pin_rtran, pin_ftran)
        stage_started_at = record_stage("setup_rat", stage_started_at)

        if getattr(self, "log_stage_progress", True):
            logging.info("TimingPropagation stage: endpoint_net_rat")
        pin_rRAT, pin_fRAT = self.calculate_net_rat_level(
            self.end_points, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall
        )
        stage_started_at = record_stage("endpoint_net_rat", stage_started_at)
        num_arcs_level = len(working_flat_arc_level_start) - 1
        cell_rat_started_at = profile_clock()
        self.last_fast_loop_skipped_full_rat = bool(skip_full_rat)
        if skip_full_rat:
            if getattr(self, "log_stage_progress", True):
                logging.info("TimingPropagation stage: cell_rat_levels skipped by production_fast_loop")
        else:
            for level in range(len(working_flat_arc_level_start) - 2):
                start = level_offset(working_flat_arc_level_start, num_arcs_level - level - 1)
                end = level_offset(working_flat_arc_level_start, num_arcs_level - level)
                if start >= end:
                    continue
                inst_arcs = working_flat_arcs_by_level[start: end]
                abs_arc_indices = working_flat_arc_indices[start: end]
                if level == 0 or level % 50 == 0:
                    if getattr(self, "log_stage_progress", True):
                        logging.info("TimingPropagation stage: cell_rat_level %d", level)
                cur_endpoints, pin_rRAT, pin_fRAT = self.calculate_cell_rat_level(
                    inst_arcs,
                    pin_rRAT,
                    pin_fRAT,
                    cell_arc_rr_delays[abs_arc_indices],
                    cell_arc_fr_delays[abs_arc_indices],
                    cell_arc_rf_delays[abs_arc_indices],
                    cell_arc_ff_delays[abs_arc_indices],
                )
                pin_rRAT, pin_fRAT = self.calculate_net_rat_level(
                    cur_endpoints, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall
                )
        if self.timing_propagation_profile:
            stage_runtime_ms["cell_rat_levels"] = (
                profile_clock() - cell_rat_started_at
            ) * 1000.0
        stage_started_at = profile_clock()
        if skip_full_rat:
            endpoint_rslack = pin_rRAT[self.end_points] - pin_rAAT[self.end_points]
            endpoint_fslack = pin_fRAT[self.end_points] - pin_fAAT[self.end_points]
            rslack = fslack = slack = None
        else:
            rslack = pin_rRAT - pin_rAAT
            fslack = pin_fRAT - pin_fAAT
            slack = torch.min(rslack, fslack)
            endpoint_rslack = rslack[self.end_points]
            endpoint_fslack = fslack[self.end_points]
        endpoint_rslack, endpoint_fslack = qualify_pin_slacks(self, self.end_points, endpoint_rslack, endpoint_fslack)
        endpoints_slack = torch.min(endpoint_rslack, endpoint_fslack)
        RAT_THRESHOLD = 8e7 
        # valid_mask = (pin_rRAT < RAT_THRESHOLD) & (pin_fRAT < RAT_THRESHOLD)
        # all_valid_slacks = slack[valid_mask]
        self.last_endpoint_slack_tensor = endpoints_slack
        self.last_endpoint_ids_tensor = self.end_points.detach().clone()
        self.update_critical_endpoint_pruning_state(iteration=self.timing_forward_count)
        self._refresh_traversal_pruning_state(iteration=self.timing_forward_count)
        self.timing_forward_count += 1
        # neg_slack = torch.clamp(all_valid_slacks, max=0)
        ts = 0
        wns, tns, ws = endpoint_metrics(endpoints_slack, pin_net_cap_rise.sum() * 0)
        stage_started_at = record_stage("slack_finalize", stage_started_at)

        critical_path_pin_net_delay_rise = None
        critical_path_pin_net_delay_fall = None
        if critical_path_snapshot_requested and dynamic_net_provider is not None:
            critical_path_pin_net_delay_rise = pin_net_delay_rise
            critical_path_pin_net_delay_fall = pin_net_delay_fall
        self._store_forward_state(
            pin_rAAT=pin_rAAT,
            pin_fAAT=pin_fAAT,
            pin_rRAT=pin_rRAT,
            pin_fRAT=pin_fRAT,
            pin_rtran=pin_rtran,
            pin_ftran=pin_ftran,
            pin_net_cap_rise=pin_net_cap_rise,
            pin_net_cap_fall=pin_net_cap_fall,
            pin_net_delay_rise=pin_net_delay_rise,
            pin_net_delay_fall=pin_net_delay_fall,
            pin_net_impulse_rise=pin_net_impulse_rise,
            pin_net_impulse_fall=pin_net_impulse_fall,
            rslack=rslack,
            fslack=fslack,
            slack=slack,
            cell_arc_rr_delays=cell_arc_rr_delays,
            cell_arc_fr_delays=cell_arc_fr_delays,
            cell_arc_rf_delays=cell_arc_rf_delays,
            cell_arc_ff_delays=cell_arc_ff_delays,
            critical_path_pin_net_delay_rise=critical_path_pin_net_delay_rise,
            critical_path_pin_net_delay_fall=critical_path_pin_net_delay_fall,
        )
        total_runtime_ms = (profile_clock() - profile_started_at) * 1000.0
        post_timing_boundary = {
            "core_device": str(self.device),
            "return_device": str(return_device),
            "returns_moved_to_return_device": str(return_device) != str(self.device),
            "returned_tensors": ["wns", "tns", "ws", "ts"],
            "excluded_from_runtime_ms": True,
            "note": (
                "runtime_ms and stage_runtime_ms stop before scalar timing outputs "
                "are moved back to the caller return_device"
            ),
        }
        write_crash_stage_marker(
            "timing_propagation_core_forward",
            "done",
            resolved_device=str(self.device),
        )
        self.last_parity_payload = self._try_run_cpu_cuda_parity(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            surrogate_mode,
            wns,
            tns,
            endpoints_slack,
        )
        self.last_profile_payload = self._build_profile_payload(
            total_runtime_ms=total_runtime_ms,
            stage_runtime_ms=stage_runtime_ms,
            surrogate_mode=surrogate_mode,
            device_contract=forward_contract,
            post_timing_boundary=post_timing_boundary,
            cell_aat_profile=cell_aat_profile,
            dynamic_provider_metadata=getattr(dynamic_net_provider, "metadata", None),
        )
        if getattr(self, "log_stage_progress", True):
            logging.info(f"WNS: {wns:.4f}, TNS: {tns:.4f}")
            logging.info(f"Total Timing propagation Time: {time.time() - start_time:.4f}s")
        if torch.is_tensor(wns):
            wns = wns.to(return_device)
        if torch.is_tensor(tns):
            tns = tns.to(return_device)
        if torch.is_tensor(ws):
            ws = ws.to(return_device)
        if torch.is_tensor(ts):
            ts = ts.to(return_device)
        self._end_forward_runtime_state()
        write_crash_stage_marker(
            "timing_propagation_forward",
            "done",
            resolved_device=str(self.device),
            return_device=str(return_device),
        )
        return wns, tns, ws, ts

    def get_pin_slack(self):
        """返回最新一次时序传播得到的 pin slack 向量。"""
        pin_slack = self.pin_slack
        if pin_slack is None:
            pin_slack = getattr(self, "pin_slack_live_snapshot", None)
        if pin_slack is None:
            raise RuntimeError("尚未执行时序传播，无法获取 pin slack")
        return pin_slack.detach().cpu()

    def _store_forward_state(
        self,
        *,
        pin_rAAT,
        pin_fAAT,
        pin_rRAT,
        pin_fRAT,
        pin_rtran,
        pin_ftran,
        pin_net_cap_rise,
        pin_net_cap_fall,
        pin_net_delay_rise,
        pin_net_delay_fall,
        rslack,
        fslack,
        slack,
        cell_arc_rr_delays,
        cell_arc_fr_delays,
        cell_arc_rf_delays,
        cell_arc_ff_delays,
        pin_net_impulse_rise=None,
        pin_net_impulse_fall=None,
        critical_path_pin_net_delay_rise=None,
        critical_path_pin_net_delay_fall=None,
    ):
        self._capture_critical_path_snapshot(
            pin_rAAT=pin_rAAT,
            pin_fAAT=pin_fAAT,
            pin_rRAT=pin_rRAT,
            pin_fRAT=pin_fRAT,
            pin_net_delay_rise=(
                pin_net_delay_rise
                if critical_path_pin_net_delay_rise is None
                else critical_path_pin_net_delay_rise
            ),
            pin_net_delay_fall=(
                pin_net_delay_fall
                if critical_path_pin_net_delay_fall is None
                else critical_path_pin_net_delay_fall
            ),
            cell_arc_rr_delays=cell_arc_rr_delays,
            cell_arc_fr_delays=cell_arc_fr_delays,
            cell_arc_rf_delays=cell_arc_rf_delays,
            cell_arc_ff_delays=cell_arc_ff_delays,
        )
        self.pin_rtran_live, self.pin_ftran_live = pin_rtran, pin_ftran
        self.pin_net_cap_rise_live, self.pin_net_cap_fall_live = (
            pin_net_cap_rise,
            pin_net_cap_fall,
        )
        self.pin_net_delay_rise_live, self.pin_net_delay_fall_live = (
            pin_net_delay_rise.detach(),
            pin_net_delay_fall.detach(),
        )
        self.pin_net_impulse_rise_live = (
            pin_net_impulse_rise.detach()
            if pin_net_impulse_rise is not None
            else None
        )
        self.pin_net_impulse_fall_live = (
            pin_net_impulse_fall.detach()
            if pin_net_impulse_fall is not None
            else None
        )
        self.pin_rAAT_live_snapshot, self.pin_fAAT_live_snapshot = (
            t.clone().detach() for t in [pin_rAAT, pin_fAAT]
        )
        self.pin_rRAT_live_snapshot, self.pin_fRAT_live_snapshot = (
            t.clone().detach() for t in [pin_rRAT, pin_fRAT]
        )
        if rslack is None:
            rslack = pin_rRAT - pin_rAAT
        if fslack is None:
            fslack = pin_fRAT - pin_fAAT
        if slack is None:
            slack = torch.min(rslack, fslack)
        self.pin_rslack_live_snapshot, self.pin_fslack_live_snapshot, self.pin_slack_live_snapshot = (
            t.clone().detach() for t in [rslack, fslack, slack]
        )
        if getattr(self, "production_fast_loop", False):
            self.pin_rAAT = self.pin_fAAT = self.pin_rRAT = self.pin_fRAT = None
            self.pin_rtran = self.pin_ftran = None
            self.pin_net_cap_rise = self.pin_net_cap_fall = None
            self.pin_net_delay_rise = self.pin_net_delay_fall = None
            self.pin_rslack = self.pin_fslack = self.pin_slack = None
            self.cell_arc_rr_delays = self.cell_arc_fr_delays = None
            self.cell_arc_rf_delays = self.cell_arc_ff_delays = None
            return

        self.pin_rAAT, self.pin_fAAT, self.pin_rRAT, self.pin_fRAT = (
            t.clone().detach() for t in [pin_rAAT, pin_fAAT, pin_rRAT, pin_fRAT])
        self.pin_rtran, self.pin_ftran = (
            t.clone().detach() for t in [pin_rtran, pin_ftran])
        # pin_net_cap_* stays on the RC-graph domain. When Steiner points are
        # appended by steiner_topo/rc_timing, these vectors can be longer than
        # self.num_pins even though all per-pin timing state above remains in
        # the design-pin domain.
        self.pin_net_cap_rise, self.pin_net_cap_fall = (
            t.clone().detach() for t in [pin_net_cap_rise, pin_net_cap_fall])
        self.pin_net_delay_rise = pin_net_delay_rise.clone().detach()
        self.pin_net_delay_fall = pin_net_delay_fall.clone().detach()
        self.pin_rslack = rslack.clone().detach()
        self.pin_fslack = fslack.clone().detach()
        self.pin_slack = slack.clone().detach()
        self.cell_arc_rr_delays, self.cell_arc_fr_delays, self.cell_arc_rf_delays, self.cell_arc_ff_delays  = (
            t.clone().detach() for t in [cell_arc_rr_delays, cell_arc_fr_delays, cell_arc_rf_delays, cell_arc_ff_delays ])

    def _prepare_violation_inputs(self, values, pin_mask, limits, metric_name: str):
        values = values.reshape(-1)
        mask = pin_mask.bool().to(values.device).reshape(-1)
        limits = limits.to(values.device).float().reshape(-1)
        # mask/limits are always design-pin tensors. values may come from the
        # larger RC graph and therefore include a trailing Steiner-only suffix.

        if mask.numel() != limits.numel():
            raise RuntimeError(
                f"{metric_name} mask/limit length mismatch: "
                f"mask={mask.numel()} limits={limits.numel()}"
            )

        expected_len = mask.numel()
        if values.numel() == expected_len:
            return values, mask, limits

        if values.numel() > expected_len:
            if getattr(self, "log_violation_shape_mismatch", True):
                logging.warning(
                    "%s length mismatch: values=%d mask=%d limits=%d; "
                    "trimming Steiner suffix and using design-pin prefix=%d",
                    metric_name,
                    values.numel(),
                    mask.numel(),
                    limits.numel(),
                    expected_len,
                )
            return values[:expected_len], mask, limits

        raise RuntimeError(
            f"{metric_name} values shorter than design-pin mask: "
            f"values={values.numel()} mask={mask.numel()} limits={limits.numel()}"
        )

    def get_total_slew_violation(self, pin_mask, slew_limits_ps):
        total_violation = self.get_total_slew_violation_tensor(pin_mask, slew_limits_ps)
        return float(total_violation.detach().item())

    def get_total_slew_violation_tensor(self, pin_mask, slew_limits_ps):
        virtual_drv = getattr(self, "virtual_buffer_drv", None)
        pin_rtran_live = getattr(self, "pin_rtran_live", None)
        pin_ftran_live = getattr(self, "pin_ftran_live", None)
        pin_rtran = pin_rtran_live if pin_rtran_live is not None else self.pin_rtran
        pin_ftran = pin_ftran_live if pin_ftran_live is not None else self.pin_ftran
        if pin_rtran is None or pin_ftran is None:
            raise RuntimeError("尚未执行时序传播，无法统计 slew violation")

        rise_values, mask, limits = self._prepare_violation_inputs(
            pin_rtran,
            pin_mask,
            slew_limits_ps,
            "slew violation inputs",
        )
        fall_values, _, _ = self._prepare_violation_inputs(
            pin_ftran,
            pin_mask,
            slew_limits_ps,
            "slew violation inputs",
        )
        valid_mask = mask & torch.isfinite(limits) & (limits > 0)
        if not torch.any(valid_mask):
            return rise_values.new_zeros(()) if virtual_drv is None else virtual_drv[0]

        rise_violation = torch.clamp(rise_values[valid_mask].float() - limits[valid_mask], min=0.0)
        fall_violation = torch.clamp(fall_values[valid_mask].float() - limits[valid_mask], min=0.0)
        total_violation_per_pin = torch.maximum(rise_violation, fall_violation)

        # Debug: log top 20 slew violations
        if getattr(self, "log_top_violations", True) and total_violation_per_pin.numel() > 0:
            top_k = min(20, total_violation_per_pin.numel())
            top_values, top_indices = torch.topk(total_violation_per_pin, top_k)
            valid_indices = torch.where(valid_mask)[0]
            logging.info("=== Top 20 Slew Violations (AutoDMP) ===")
            logging.info(f"Total pins checked: {valid_mask.sum().item()}, pins with violation: {(total_violation_per_pin > 0).sum().item()}")
            logging.info("pin_idx, rise_slew_ps, fall_slew_ps, limit_ps, violation_ps")
            for i in range(top_k):
                pin_idx = valid_indices[top_indices[i]].item()
                rv = rise_values[valid_indices[top_indices[i]]].item()
                fv = fall_values[valid_indices[top_indices[i]]].item()
                lv = limits[valid_indices[top_indices[i]]].item()
                tv = top_values[i].item()
                logging.info(f"{pin_idx}, {rv:.3f}, {fv:.3f}, {lv:.3f}, {tv:.3f}")
            logging.info(f"Total slew violation: {total_violation_per_pin.sum().item() / 1000.0:.3f} ns")

        # Helper/golden-compatible slew aggregation uses the worst rise/fall
        # excess per pin, with the exported scalar kept in ns.
        total = total_violation_per_pin.sum() / 1000.0
        return total if virtual_drv is None else total + virtual_drv[0]

    def get_total_cap_violation(self, output_pin_mask, cap_limits_pf):
        total_violation = self.get_total_cap_violation_tensor(output_pin_mask, cap_limits_pf)
        return float(total_violation.detach().item() * 1000.0)

    def get_total_cap_violation_tensor(self, output_pin_mask, cap_limits_pf):
        virtual_drv = getattr(self, "virtual_buffer_drv", None)
        pin_net_cap_rise_live = getattr(self, "pin_net_cap_rise_live", None)
        pin_net_cap_fall_live = getattr(self, "pin_net_cap_fall_live", None)
        pin_net_cap_rise = (
            pin_net_cap_rise_live
            if pin_net_cap_rise_live is not None
            else self.pin_net_cap_rise
        )
        pin_net_cap_fall = (
            pin_net_cap_fall_live
            if pin_net_cap_fall_live is not None
            else self.pin_net_cap_fall
        )
        if pin_net_cap_rise is None or pin_net_cap_fall is None:
            raise RuntimeError("尚未执行时序传播，无法统计 cap violation")

        rise_values, mask, limits = self._prepare_violation_inputs(
            pin_net_cap_rise,
            output_pin_mask,
            cap_limits_pf,
            "cap violation inputs",
        )
        fall_values, _, _ = self._prepare_violation_inputs(
            pin_net_cap_fall,
            output_pin_mask,
            cap_limits_pf,
            "cap violation inputs",
        )
        valid_mask = mask & torch.isfinite(limits) & (limits > 0)
        if not torch.any(valid_mask):
            return rise_values.new_zeros(()) if virtual_drv is None else virtual_drv[1]

        rise_violation = torch.clamp(
            rise_values[valid_mask].float() - limits[valid_mask],
            min=0.0,
        )
        fall_violation = torch.clamp(
            fall_values[valid_mask].float() - limits[valid_mask],
            min=0.0,
        )
        total_violation_per_pin = torch.maximum(rise_violation, fall_violation)

        # Debug: log top 20 cap violations
        if getattr(self, "log_top_violations", True) and total_violation_per_pin.numel() > 0:
            top_k = min(20, total_violation_per_pin.numel())
            top_values, top_indices = torch.topk(total_violation_per_pin, top_k)
            valid_indices = torch.where(valid_mask)[0]
            logging.info("=== Top 20 Cap Violations (AutoDMP) ===")
            logging.info(f"Total pins checked: {valid_mask.sum().item()}, pins with violation: {(total_violation_per_pin > 0).sum().item()}")
            logging.info("pin_idx, rise_cap_pf, fall_cap_pf, limit_pf, violation_fF")
            for i in range(top_k):
                pin_idx = valid_indices[top_indices[i]].item()
                rv = rise_values[valid_indices[top_indices[i]]].item()
                fv = fall_values[valid_indices[top_indices[i]]].item()
                lv = limits[valid_indices[top_indices[i]]].item()
                tv = top_values[i].item()
                logging.info(f"{pin_idx}, {rv:.6f}, {fv:.6f}, {lv:.6f}, {tv * 1000:.3f}")
            logging.info(f"Total cap violation: {total_violation_per_pin.sum().item() * 1000.0:.3f} fF")

        # Helper/golden-compatible cap aggregation uses the worst rise/fall
        # excess per pin, with the exported scalar kept in fF.
        total = total_violation_per_pin.sum()
        return total if virtual_drv is None else total + virtual_drv[1]

    def get_critical_paths(self, endpoints, K: int = 1, max_depth: Optional[int] = None,
                           slack_epsilon: float = 1e-3):
        pin_slack_source = self.pin_slack
        if pin_slack_source is None:
            pin_slack_source = getattr(self, "pin_slack_live_snapshot", None)
        pin_rslack_source = self.pin_rslack
        if pin_rslack_source is None:
            pin_rslack_source = getattr(self, "pin_rslack_live_snapshot", None)
        pin_fslack_source = self.pin_fslack
        if pin_fslack_source is None:
            pin_fslack_source = getattr(self, "pin_fslack_live_snapshot", None)
        pin_rAAT_source = self.pin_rAAT
        if pin_rAAT_source is None:
            pin_rAAT_source = getattr(self, "pin_rAAT_live_snapshot", None)
        pin_fAAT_source = self.pin_fAAT
        if pin_fAAT_source is None:
            pin_fAAT_source = getattr(self, "pin_fAAT_live_snapshot", None)
        if (
            pin_slack_source is None
            or pin_rslack_source is None
            or pin_fslack_source is None
            or pin_rAAT_source is None
            or pin_fAAT_source is None
        ):
            raise RuntimeError("尚未执行时序传播，无法提取关键路径")

        endpoints_tensor = torch.as_tensor(endpoints, dtype=torch.int32, device="cpu")
        endpoints_tensor = endpoints_tensor.reshape(-1)
        if endpoints_tensor.numel() == 0:
            return []

        if max_depth is None:
            max_depth = 0
        slack_eps = slack_epsilon if slack_epsilon > 0 else 1e-3

        def _to_cpu_int(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None or not torch.is_tensor(t) or t.numel() == 0:
                return None
            return t.detach().cpu().to(torch.int32).contiguous()

        def _to_cpu_tensor_general(obj, dtype=torch.int32) -> Optional[torch.Tensor]:
            """Convert a variety of python/np/torch objects into a cpu torch tensor of given dtype.

            Returns None if conversion is not possible or object is empty.
            """
            if obj is None:
                return None
            if torch.is_tensor(obj):
                if obj.numel() == 0:
                    return None
                return obj.detach().cpu().to(dtype).contiguous()
            try:
                arr = np.asarray(obj)
            except Exception:
                return None
            if arr.size == 0:
                return None
            try:
                return torch.from_numpy(arr.astype(np.int32)).to(dtype).contiguous()
            except Exception:
                try:
                    return torch.as_tensor(arr, dtype=dtype).cpu().contiguous()
                except Exception:
                    return None

        def _extract_python() -> list[list[int]]:
            rev_offsets = self.flat_pin_to_graph_start_reverse.detach().cpu().to(torch.int64)
            rev_edges = self.flat_pin_to_graph_reverse.detach().cpu().to(torch.int64)
            num_pins = rev_offsets.numel() - 1
            rslack = pin_rslack_source.detach().cpu().to(torch.float64)
            fslack = pin_fslack_source.detach().cpu().to(torch.float64)
            raat = pin_rAAT_source.detach().cpu().to(torch.float64)
            faat = pin_fAAT_source.detach().cpu().to(torch.float64)
            start_mask = torch.zeros(num_pins, dtype=torch.bool)
            if self.start_points is not None:
                start_idx = self.start_points.detach().cpu().to(torch.int64)
                start_idx = start_idx[(start_idx >= 0) & (start_idx < num_pins)]
                if start_idx.numel() > 0:
                    start_mask[start_idx] = True

            depth_limit = max_depth if max_depth > 0 else num_pins
            paths: list[list[int]] = []
            endpoints_list = endpoints_tensor.tolist()
            for sink in endpoints_list:
                if sink < 0 or sink >= num_pins:
                    paths.append([])
                    continue

                use_rise = rslack[sink] <= fslack[sink]
                current_slack = rslack[sink] if use_rise else fslack[sink]
                current_pin = int(sink)
                path = [current_pin]
                visited = {current_pin}

                steps = 0
                while steps < depth_limit:
                    if current_pin < 0 or current_pin >= num_pins:
                        break
                    if start_mask[current_pin]:
                        break

                    begin = int(rev_offsets[current_pin])
                    end = int(rev_offsets[current_pin + 1])
                    if begin >= end:
                        break

                    best_pred = None
                    best_slack = float("inf")
                    best_gap = float("inf")
                    best_arrival = float("-inf")

                    for idx in range(begin, end):
                        pred = int(rev_edges[idx])
                        if pred < 0 or pred >= num_pins or pred in visited:
                            continue

                        pred_slack = rslack[pred] if use_rise else fslack[pred]
                        slack_gap = abs(float(pred_slack - current_slack))
                        arrival = float(raat[pred] if use_rise else faat[pred])

                        improved = False
                        if float(pred_slack) < best_slack - slack_eps:
                            improved = True
                        elif abs(float(pred_slack) - best_slack) <= slack_eps:
                            if slack_gap < best_gap - slack_eps:
                                improved = True
                            elif abs(slack_gap - best_gap) <= slack_eps and arrival > best_arrival + slack_eps:
                                improved = True

                        if improved:
                            best_pred = pred
                            best_slack = float(pred_slack)
                            best_gap = slack_gap
                            best_arrival = arrival

                    if best_pred is None:
                        break

                    current_pin = best_pred
                    current_slack = rslack[best_pred] if use_rise else fslack[best_pred]
                    path.append(current_pin)
                    visited.add(current_pin)
                    steps += 1

                paths.append(list(reversed(path)))

            return paths

        if _tp_cpp is not None and hasattr(_tp_cpp, "extract_critical_paths"):
            try:
                pin_pair_keys = _to_cpu_tensor_general(self.pin_pair_arc_keys, dtype=torch.int32)
                pin_pair_start = _to_cpu_tensor_general(self.flat_pin_pair_arc_start, dtype=torch.int32)
                pin_pair_indices = _to_cpu_tensor_general(self.flat_pin_pair_arc_indices, dtype=torch.int32)
                paths, _ = _tp_cpp.extract_critical_paths(
                    endpoints_tensor,
                    int(K),
                    self.start_points.cpu(),
                    pin_rAAT_source.detach().cpu(),
                    pin_fAAT_source.detach().cpu(),
                    pin_rslack_source.detach().cpu(),
                    pin_fslack_source.detach().cpu(),
                    pin_slack_source.detach().cpu(),
                    self.flat_pin_to_graph_reverse.cpu(),
                    self.flat_pin_to_graph_start_reverse.cpu(),
                    pin_pair_keys,
                    pin_pair_start,
                    pin_pair_indices,
                    int(max_depth),
                    float(slack_eps),
                )
                return [list(map(int, path)) for path in paths]
            except Exception as exc:  # pragma: no cover - fallback path
                logging.warning("关键路径提取回退到Python实现: %s", exc)

        if K > 1:
            logging.warning("当前Python回退版本仅返回每个端点一条路径")
        return _extract_python()
