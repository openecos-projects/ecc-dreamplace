# @file   rc_timing.py
# @author
# @date   Mar 2025
# @brief  Convert Steiner tree to RC tree for timing analysis
#

import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.rc_timing.rc_timing_cpp as rc_timing_cpp
import dreamplace.configure as configure


class RCTimingFunction(Function):
        
    @staticmethod
    def forward(ctx, pos, 
                branch_u, branch_v, net_branch_start, 
                pin_caps, driver_pin_indices, 
                r_unit, c_unit, ignore_net_degree):
        """
        @brief 前向传播: 从斯坦纳树构建RC树
        @param pos 节点坐标张量 [num_pins*2]
        @param branch_u 斯坦纳树边的起点 [num_edges]
        @param branch_v 斯坦纳树边的终点 [num_edges]
        @param net_branch_start 每个网络边的起始索引 [num_nets+1]
        @param pin_caps 引脚电容 [num_pins]
        @param driver_pin_indices 每个网络的驱动引脚索引 [num_nets]
        @param r_unit 单位长度电阻
        @param c_unit 单位长度电容
        @param ignore_net_degree 忽略网络的度数阈值
        @return rc_values 网络的RC参数 [num_edges, 2]
        """
        # 构建边张量 [num_edges, 2]
        steiner_edges = torch.stack([branch_u, branch_v], dim=1)
        
        if pos.is_cuda:
            # 如果有CUDA支持，则调用CUDA实现
            if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
                rc_values = rc_timing_cuda.forward(
                    pos.view(-1), 
                    steiner_edges,
                    net_branch_start,
                    pin_caps,
                    driver_pin_indices,
                    r_unit,
                    c_unit,
                    ignore_net_degree
                )
            else:
                raise NotImplementedError("CUDA version not implemented")
        else:
            # 调用C++扩展
            rc_values = rc_timing_cpp.forward(
                pos.view(-1),
                steiner_edges,
                net_branch_start,
                pin_caps,
                driver_pin_indices,
                r_unit,
                c_unit,
                ignore_net_degree
            )
            
        # 保存反向传播所需信息
        ctx.save_for_backward(pos, steiner_edges, net_branch_start)
        ctx.r_unit = r_unit
        ctx.c_unit = c_unit
        
        return rc_values

    @staticmethod
    def backward(ctx, grad_output):
        """
        @brief 反向传播计算梯度
        """
        pos, steiner_edges, net_branch_start = ctx.saved_tensors
        r_unit = ctx.r_unit
        c_unit = ctx.c_unit
        
        if pos.is_cuda:
            if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
                grad_pos = rc_timing_cuda.backward(
                    grad_output,
                    pos.view(-1),
                    steiner_edges,
                    net_branch_start,
                    r_unit,
                    c_unit
                )
            else:
                raise NotImplementedError("CUDA version not implemented")
        else:
            grad_pos = rc_timing_cpp.backward(
                grad_output,
                pos.view(-1),
                steiner_edges,
                net_branch_start,
                r_unit,
                c_unit
            )
            
        return (grad_pos, None, None, None, None, None, None, None, None)

class RCTiming(nn.Module):
    def __init__(self, 
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 pin2node_map,
                 driver_pin_indices,
                 r_unit=1.0,
                 c_unit=1.0,
                 ignore_net_degree=None):
        """
        @param flat_net2pin_map 网络到引脚的扁平化映射
        @param flat_net2pin_start_map 每个网络的起始索引
        @param pin2node_map 引脚到节点的映射
        @param pin_caps 每个引脚的电容值
        @param driver_pin_indices 每个网络的驱动引脚索引
        @param r_unit 单位长度电阻值
        @param c_unit 单位长度电容值
        @param ignore_net_degree 忽略网络的度数阈值
        """
        super(RCTiming, self).__init__()
        # 注册为buffer确保设备一致性
        self.register_buffer('flat_net2pin_map', flat_net2pin_map)
        self.register_buffer('flat_net2pin_start_map', flat_net2pin_start_map)
        self.register_buffer('pin2node_map', pin2node_map)
        self.register_buffer('driver_pin_indices', driver_pin_indices)
        
        # 设置RC参数
        self.r_unit = r_unit
        self.c_unit = c_unit
        
        # 设置忽略网络的度数阈值
        self.ignore_net_degree = ignore_net_degree if ignore_net_degree else flat_net2pin_map.numel()

    def forward(self, pos, steiner_output, branch_u, branch_v, net_branch_start, pin_caps):
        """
        @brief 前向传播主函数
        @param pos 节点位置张量 [num_nodes*2]
        @param steiner_output steiner_topo.forward()的输出结果（可选）
        @param branch_u 斯坦纳树边的起点 [num_edges]（可选，与steiner_output二选一）
        @param branch_v 斯坦纳树边的终点 [num_edges]（可选，与steiner_output二选一）
        @param net_branch_start 每个网络边的起始索引 [num_nets+1]（可选，与steiner_output二选一）
        @return rc_values 网络的RC参数 [num_edges, 2]
        """
        # 通过pin2node_map转换坐标
        pin_pos = pos[self.pin2node_map]
        
        # 处理输入参数
        if steiner_output is not None:
            # 如果提供了steiner_output，直接解包
            _, _, _, _, branch_u, branch_v, net_branch_start = steiner_output
        else:
            # 否则使用单独提供的参数
            assert branch_u is not None and branch_v is not None and net_branch_start is not None, \
                "必须提供steiner_output或branch_u/branch_v/net_branch_start"
            
        # 调用底层扩展
        return RCTimingFunction.apply(
            pin_pos,
            branch_u,
            branch_v,
            net_branch_start,
            pin_caps,
            self.driver_pin_indices,
            self.r_unit,
            self.c_unit,
            self.ignore_net_degree
        )

    @property
    def output_names(self):
        """输出张量描述信息"""
        return [
            "rc_values"  # [num_edges, 2] 每条边的电阻和电容值
        ] 