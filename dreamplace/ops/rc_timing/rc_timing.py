#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : rc_timing.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2025-05-06 16:03:14
@copyright    : Copyright (c) 2023-2025 ICT, CAS.
'''

import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.rc_timing.rc_timing_cpp as rc_timing_cpp
import dreamplace.configure as configure

# ==========================
# Load Operator
# ==========================


class LoadOpFunction(Function):
    """
    Computes Load = Cap + sum(Load_children) (Bottom-up)
    Corresponds to rc_timing_cpp.load_forward / load_backward
    """
    @staticmethod
    def forward(ctx,
                cap,              # Input: Capacitance (requires grad)
                pin_start,        # Input: Structure (no grad)
                pin_to,           # Input: Structure (no grad)
                net_flat_topo,    # Input: Structure (no grad)
                net_flat_topo_start  # Input: Structure (no grad)
                # Removed pin_fa, net_driver_pin as they seem unused in C++ forward/backward
                ):

        # Call C++ Forward:
        # Inputs: cap_tensor, pin_fa_tensor (unused), net_driver_pin_tensor (unused),
        #         pin_start_tensor, pin_to_tensor, net_flat_topo, net_flat_topo_start
        load = rc_timing_cpp.load_forward_cpp(  # Or load_forward_cpp
            cap,
            pin_start,
            pin_to,
            net_flat_topo,
            net_flat_topo_start
        )

        # Save tensors needed for C++ backward:
        # C++ Backward Inputs: grad_load, pin_fa(unused), net_driver(unused),
        #                      pin_start, pin_to, net_flat_topo, net_flat_topo_start
        # We only need the structure tensors.
        ctx.save_for_backward(
            pin_start, pin_to, net_flat_topo, net_flat_topo_start)
        # Note: We don't save 'cap' as it's not needed for backward calculation itself.

        return load

    @staticmethod
    def backward(ctx, grad_load):
        # grad_load is the gradient dF/dLoad

        # Check if context is valid
        # Check if gradient w.r.t. 'cap' is needed
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None  # Match number of forward inputs

        # Retrieve saved tensors
        pin_start, pin_to, net_flat_topo, net_flat_topo_start = ctx.saved_tensors

        # Call C++ Backward:
        # Inputs: grad_output, pin_fa_tensor (unused), net_driver_pin_tensor (unused),
        #         pin_start_tensor, pin_to_tensor, net_flat_topo, net_flat_topo_start
        # Output: grad_input_cap
        grad_cap = rc_timing_cpp.load_backward_cpp(
            grad_load.contiguous(),  # Ensure contiguous
            pin_start,
            pin_to,
            net_flat_topo,
            net_flat_topo_start
        )

        # Return gradients for inputs of forward:
        # cap, pin_start, pin_to, net_flat_topo, net_flat_topo_start
        return grad_cap, None, None, None, None


# ==========================
# Delay Operator
# ==========================
class DelayOpFunction(Function):
    """
    Computes Delay = Delay(fa) + Res * Load (Top-down)
    Corresponds to rc_timing_cpp.delay_forward / delay_backward
    """
    @staticmethod
    def forward(ctx,
                res,              # Input: Resistance (requires grad)
                load,             # Input: Load (requires grad)
                pin_fa,           # Input: Structure (no grad)
                net_flat_topo,    # Input: Structure (no grad)
                net_flat_topo_start  # Input: Structure (no grad)
                # Removed pin_start, pin_to as they are not needed for C++ forward
                ):

        # Call C++ Forward:
        # Inputs: resistance_tensor, load_tensor, pin_fa_tensor,
        #         net_driver_pin_tensor (unused), net_flat_topo, net_flat_topo_start
        delay = rc_timing_cpp.delay_forward_cpp(
            res,
            load,
            pin_fa,
            net_flat_topo,
            net_flat_topo_start
        )

        # Save tensors needed for C++ backward:
        # C++ Backward Inputs: grad_delay, res, load, pin_fa,
        #                      net_driver(unused), net_flat_topo, net_flat_topo_start
        ctx.save_for_backward(
            res, load, pin_fa, net_flat_topo, net_flat_topo_start)

        return delay

    @staticmethod
    def backward(ctx, grad_delay):
        # grad_delay is the gradient dF/dDelay

        # Check if gradients are needed for res or load
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None, None, None  # Match number of forward inputs

        # Retrieve saved tensors
        res, load, pin_fa, net_flat_topo, net_flat_topo_start = ctx.saved_tensors

        # Call C++ Backward:
        # Inputs: grad_output_delay, resistance_tensor, load_tensor, pin_fa_tensor,
        #         net_driver_pin_tensor (unused), net_flat_topo, net_flat_topo_start
        # Outputs: [grad_input_res, grad_input_load]
        grad_res_load_list = rc_timing_cpp.delay_backward_cpp(
            grad_delay.contiguous(),  # Ensure contiguous
            res,
            load,
            pin_fa,
            net_flat_topo,
            net_flat_topo_start
        )
        grad_res = grad_res_load_list[0]
        grad_load = grad_res_load_list[1]

        # Return gradients for inputs of forward:
        # res, load, pin_fa, net_flat_topo, net_flat_topo_start
        # Only return grads if needed
        grad_res_out = grad_res if ctx.needs_input_grad[0] else None
        grad_load_out = grad_load if ctx.needs_input_grad[1] else None

        return grad_res_out, grad_load_out, None, None, None


# ==========================
# LDelay Operator
# ==========================
class LDelayOpFunction(Function):
    """
    Computes LDelay = Cap * Delay + sum(LDelay_children) (Bottom-up)
    Corresponds to rc_timing_cpp.ldelay_forward / ldelay_backward
    """
    @staticmethod
    def forward(ctx,
                cap,              # Input: Capacitance (requires grad)
                delay,            # Input: Delay (requires grad)
                pin_start,        # Input: Structure (no grad)
                pin_to,           # Input: Structure (no grad)
                net_flat_topo,    # Input: Structure (no grad)
                net_flat_topo_start  # Input: Structure (no grad)
                ):

        # Call C++ Forward:
        # Inputs: cap_tensor, delay_tensor, pin_start_tensor, pin_to_tensor,
        #         net_flat_topo, net_flat_topo_start
        ldelay = rc_timing_cpp.ldelay_forward_cpp(
            cap,
            delay,
            pin_start,
            pin_to,
            net_flat_topo,
            net_flat_topo_start
        )

        # Save tensors needed for C++ backward:
        # C++ Backward Inputs: grad_ldelay, cap, delay, pin_start, pin_to,
        #                      net_flat_topo, net_flat_topo_start
        ctx.save_for_backward(cap, delay, pin_start, pin_to,
                              net_flat_topo, net_flat_topo_start)

        return ldelay

    @staticmethod
    def backward(ctx, grad_ldelay):
        # grad_ldelay is the gradient dF/dLDelay

        # Check if gradients are needed for cap or delay
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None, None, None, None  # Match number of forward inputs

        # Retrieve saved tensors
        cap, delay, pin_start, pin_to, net_flat_topo, net_flat_topo_start = ctx.saved_tensors

        # Call C++ Backward:
        # Inputs: grad_output_ldelay, cap_tensor, delay_tensor, pin_start_tensor,
        #         pin_to_tensor, net_flat_topo, net_flat_topo_start
        # Outputs: [grad_input_cap, grad_input_delay]
        grad_cap_delay_list = rc_timing_cpp.ldelay_backward_cpp(
            grad_ldelay.contiguous(),  # Ensure contiguous
            cap,
            delay,
            pin_start,
            pin_to,
            net_flat_topo,
            net_flat_topo_start
        )
        grad_cap = grad_cap_delay_list[0]
        grad_delay = grad_cap_delay_list[1]

        # Return gradients for inputs of forward:
        # cap, delay, pin_start, pin_to, net_flat_topo, net_flat_topo_start
        grad_cap_out = grad_cap if ctx.needs_input_grad[0] else None
        grad_delay_out = grad_delay if ctx.needs_input_grad[1] else None

        return grad_cap_out, grad_delay_out, None, None, None, None


# ==========================
# Beta Operator
# ==========================
class BetaOpFunction(Function):
    """
    Computes Beta = Beta(fa) + Res * LDelay (Top-down)
    Corresponds to rc_timing_cpp.beta_forward / beta_backward
    """
    @staticmethod
    def forward(ctx,
                res,               # Input: Resistance (requires grad)
                ldelay,            # Input: LDelay (requires grad)
                pin_fa,            # Input: Structure (no grad)
                net_flat_topo,     # Input: Structure (no grad)
                net_flat_topo_start  # Input: Structure (no grad)
                ):

        # Call C++ Forward:
        # Inputs: resistance_tensor, ldelay_tensor, pin_fa_tensor,
        #         net_driver_pin_tensor (unused), net_flat_topo, net_flat_topo_start
        beta = rc_timing_cpp.beta_forward_cpp(
            res,
            ldelay,
            pin_fa,
            net_flat_topo,
            net_flat_topo_start
        )

        # Save tensors needed for C++ backward:
        # C++ Backward Inputs: grad_beta, res, ldelay, pin_fa,
        #                      net_driver(unused), net_flat_topo, net_flat_topo_start
        ctx.save_for_backward(res, ldelay, pin_fa,
                              net_flat_topo, net_flat_topo_start)

        return beta

    @staticmethod
    def backward(ctx, grad_beta):
        # grad_beta is the gradient dF/dBeta

        # Check if gradients are needed for res or ldelay
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None, None, None  # Match number of forward inputs

        # Retrieve saved tensors
        res, ldelay, pin_fa, net_flat_topo, net_flat_topo_start = ctx.saved_tensors

        # Call C++ Backward:
        # Inputs: grad_output_beta, resistance_tensor, ldelay_tensor, pin_fa_tensor,
        #         net_driver_pin_tensor (unused), net_flat_topo, net_flat_topo_start
        # Outputs: [grad_input_res, grad_input_ldelay]
        grad_res_ldelay_list = rc_timing_cpp.beta_backward_cpp(
            grad_beta.contiguous(),  # Ensure contiguous
            res,
            ldelay,
            pin_fa,
            net_flat_topo,
            net_flat_topo_start
        )
        grad_res = grad_res_ldelay_list[0]
        grad_ldelay = grad_res_ldelay_list[1]

        # Return gradients for inputs of forward:
        # res, ldelay, pin_fa, net_flat_topo, net_flat_topo_start
        grad_res_out = grad_res if ctx.needs_input_grad[0] else None
        grad_ldelay_out = grad_ldelay if ctx.needs_input_grad[1] else None

        return grad_res_out, grad_ldelay_out, None, None, None


'''
net_flat_topo_sort: 给每个net分配一个拓扑排序的索引
net_flat_topo_sort_start: 每个net的起始索引
pin_fa: bfs后每个pin的父节点索引, 不区分net，
flat_pin_to： bfs后，每个节点的孩子，不区分net
flat_pin_to_start: 每个pin的起始索引
flat_pin_from: 展开的from形式，最后会形如 [1,1,1,2,2,3,...] 

'''


class RCTiming(nn.Module):
    def __init__(self,
                 r_unit=1.0,
                 c_unit=1.0, 
                 scale_factor=1.0, 
                 dbu=1.0):
        """

        @param pin2node_map 引脚到节点的映射
        @param pin_caps 每个引脚的电容值
        @param driver_pin_indices 每个网络的驱动引脚索引
        @param r_unit 单位长度电阻值
        @param c_unit 单位长度电容值
        @param ignore_net_degree 忽略网络的度数阈值
        """
        super(RCTiming, self).__init__()

        # 设置RC参数
        self.flat_pin_to_res = None
        self.net_cap = None
        self.delays = None
        self.edge_resistance = None
        self.loads = None
        self.ldelays = None
        self.betas = None
        self.impulses = None
        self.r_unit = r_unit
        self.c_unit = c_unit
        self.scale_factor = scale_factor
        self.dbu = dbu

    def forward(self, new_x, new_y,
                net_flat_topo_sort,
                net_flat_topo_sort_start,
                pin_fa,
                flat_pin_to_start,
                flat_pin_to,
                flat_pin_from,
                pin_caps_base, 
                pin_rcaps_base, 
                pin_fcaps_base):

        # # the length is um
        self.dbu = torch.tensor(self.dbu, dtype=new_x.dtype, device=new_x.device)
        length = (torch.abs(new_x[flat_pin_from] - new_x[flat_pin_to])
                    + torch.abs(new_y[flat_pin_from] - new_y[flat_pin_to])) / self.scale_factor / self.dbu
        
        cap = length * self.c_unit
        net_caps = torch.zeros_like(new_x, dtype=pin_caps_base.dtype)
        net_caps = torch.scatter_add(net_caps, 0, flat_pin_from.long(), cap / 2)
        net_caps = torch.scatter_add(net_caps, 0, flat_pin_to.long(),   cap / 2)

        edge_resistance  = length * self.r_unit
        flat_pin_to_res = torch.zeros_like(new_x, dtype=edge_resistance.dtype)
        flat_pin_to_res = torch.scatter(flat_pin_to_res, 0, flat_pin_to.long(), edge_resistance)

        modes = ['generic', 'rise', 'fall']
        base_caps_map = {
            'generic': pin_caps_base,
            'rise': pin_rcaps_base,
            'fall': pin_fcaps_base
        }
        pin_caps, loads, delays, ldelays, betas, impulses = {}, {}, {}, {}, {}, {}
        for mode in modes:
            base_cap = base_caps_map[mode]
            num_base = base_cap.size(0)
            num_total = new_x.size(0)

            if num_base < num_total:
                padding = torch.zeros(
                    num_total - num_base,
                    dtype=base_cap.dtype,
                    device=base_cap.device
                )
                caps_padded = torch.cat([base_cap, padding], dim=0)
            else:
                caps_padded = base_cap[:num_total]

            pin_caps[mode] = caps_padded + net_caps

            # 后续所有计算都依赖于 autograd.Function 或标准的 torch 操作，它们是 autograd 友好的
            loads[mode] = LoadOpFunction.apply(
                pin_caps[mode], flat_pin_to_start, flat_pin_to,
                net_flat_topo_sort, net_flat_topo_sort_start)

            delays[mode] = DelayOpFunction.apply(
                flat_pin_to_res, loads[mode], pin_fa,
                net_flat_topo_sort, net_flat_topo_sort_start)

            ldelays[mode] = LDelayOpFunction.apply(
                pin_caps[mode], delays[mode], flat_pin_to_start, flat_pin_to,
                net_flat_topo_sort, net_flat_topo_sort_start)

            betas[mode] = BetaOpFunction.apply(
                flat_pin_to_res, ldelays[mode], pin_fa,
                net_flat_topo_sort, net_flat_topo_sort_start)

            # 计算 impulse
            inner_term = 2 * betas[mode] - delays[mode] ** 2
            inner_term_stable = torch.clamp(inner_term, min=1e-12)
            impulses[mode] = torch.sqrt(inner_term_stable)

        load, rload, fload = loads['generic'], loads['rise'], loads['fall']
        delay, rdelay, fdelay = delays['generic'], delays['rise'], delays['fall']
        impulse, rimpulse, fimpulse = impulses['generic'], impulses['rise'], impulses['fall']
        # pin_caps = torch.zeros_like(new_x, dtype=pin_caps_base.dtype)
        # pin_caps[:pin_caps_base.size(0)] = pin_caps_base
        # pin_caps = pin_caps + net_caps
        # pin_rcaps = torch.zeros_like(new_x, dtype=pin_caps_base.dtype)
        # pin_rcaps[:pin_rcaps_base.size(0)] = pin_rcaps_base
        # pin_rcaps = pin_rcaps + net_caps
        # pin_fcaps = torch.zeros_like(new_x, dtype=pin_caps_base.dtype)
        # pin_fcaps[:pin_fcaps_base.size(0)] = pin_fcaps_base
        # pin_fcaps = pin_fcaps + net_caps


        # load = LoadOpFunction.apply(
        #     pin_caps, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # rload = LoadOpFunction.apply(
        #     pin_rcaps, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # fload = LoadOpFunction.apply(
        #     pin_fcaps, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        
        # delay = DelayOpFunction.apply(
        #     flat_pin_to_res, load, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # rdelay = DelayOpFunction.apply(
        #     flat_pin_to_res, rload, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # fdelay = DelayOpFunction.apply(
        #     flat_pin_to_res, fload, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        
        # ldelay = LDelayOpFunction.apply(
        #     pin_caps, delay, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # rldelay = LDelayOpFunction.apply(
        #     pin_rcaps, rdelay, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # fldelay = LDelayOpFunction.apply(
        #     pin_fcaps, fdelay, flat_pin_to_start, flat_pin_to,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        
        # beta = BetaOpFunction.apply(
        #     flat_pin_to_res, ldelay, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # rbeta = BetaOpFunction.apply(
        #     flat_pin_to_res, rldelay, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)
        # fbeta = BetaOpFunction.apply(
        #     flat_pin_to_res, fldelay, pin_fa,
        #     net_flat_topo_sort, net_flat_topo_sort_start)

        # inner_term = 2 * beta - delay * delay
        # rinner_term = 2 * rbeta - rdelay * rdelay
        # finner_term = 2 * fbeta - fdelay * fdelay
        # # Clamp to avoid sqrt of negative numbers due to potential floating point noise
        # inner_term_stable = torch.clamp(inner_term, min=1e-12)
        # rinner_term_stable = torch.clamp(rinner_term, min=1e-12)
        # finner_term_stable = torch.clamp(finner_term, min=1e-12)
        # impulse = torch.sqrt(inner_term_stable)
        # rimpulse = torch.sqrt(rinner_term_stable)
        # fimpulse = torch.sqrt(finner_term_stable)

        self.flat_pin_to_res = flat_pin_to_res.clone().detach()
        self.net_cap = net_caps.clone().detach()
        self.edge_resistance = edge_resistance.clone().detach()
        self.delays = delays
        self.loads = loads
        self.ldelays = ldelays
        self.betas = betas
        self.impulses = impulses

        return pin_caps, loads, delays, ldelays, betas, impulses

    @property
    def output_names(self):
        """输出张量描述信息"""
        return [
            "rc_values"  # [num_edges, 2] 每条边的电阻和电容值
        ]
