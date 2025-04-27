# @file   steiner_topo.py
# @author
# @date   Mar 2025
# @brief  Get steiner tree topology & steiner node locations
#

import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.steiner_topo.steiner_topo_cpp as steiner_topo_cpp
import dreamplace.configure as configure
# if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
#     import dreamplace.ops.steiner_topo.steiner_topo_cuda as steiner_topo_cuda
#     import dreamplace.ops.steiner_topo.steiner_topo_cuda_segment as steiner_topo_cuda_segment

class SteinerTopoFunction(Function):
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, ignore_net_degree,
                read_lut_flag):
        """
        @brief 前向传播：输入引脚位置，返回Steiner拓扑结构
        @param pos 引脚坐标张量 [num_pins*2]
        @param flat_netpin 网络到引脚的扁平化映射
        @param netpin_start 每个网络的起始索引
        @param ignore_net_degree 忽略网络的度数阈值
        @return (wl, nodes, pin_relate_x, pin_relate_y, branch_u, branch_v, net_branch_start)
        """
        # if pos.is_cuda:
        #     raise NotImplementedError("CUDA version not implemented")
        # else:
        # 调用C++扩展并接收所有输出张量
        outputs = steiner_topo_cpp.forward(
            pos.view(-1),  # 确保输入为扁平张量
            flat_netpin,
            netpin_start,
            ignore_net_degree
        )
            
        # 保存反向传播所需信息
        ctx.save_for_backward(pos, flat_netpin, netpin_start)
        ctx.num_nets = netpin_start.numel() - 1
        ctx.num_pins = pos.numel() // 2
        
        return outputs  # 返回所有7个输出张量

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        @brief 反向传播（占位实现）
        """
        # 实际反向传播需要根据具体应用实现
        pos, flat_netpin, netpin_start = ctx.saved_tensors
        grad_pos = torch.zeros_like(pos)
        return (grad_pos, None, None, None, None, None, None)

class SteinerTopo(nn.Module):
    def __init__(self, 
                 flat_net2pin_map,
                 flat_net2pin_start_map, 
                 ignore_net_degree=None,
                 algorithm="FLUTE"):
        """
        @param flat_net2pin_map 网络到引脚的扁平化映射
        @param flat_net2pin_start_map 每个网络的起始索引
        @param ignore_net_degree 忽略网络的度数阈值
        """
        super(SteinerTopo, self).__init__()
        # 注册为buffer确保设备一致性
        self.register_buffer('flat_net2pin_map', flat_net2pin_map)
        self.register_buffer('flat_net2pin_start_map', flat_net2pin_start_map)
        
        # 设置忽略网络的度数阈值
        self.ignore_net_degree = ignore_net_degree if ignore_net_degree else flat_net2pin_map.numel()
        
        # 算法参数（保留接口兼容性）
        self.algorithm = algorithm

    def update_topo(self):
        pass

    def forward(self, pos, read_lut_flag=False):
        """
        @brief 前向传播主函数
        @param pos 引脚位置张量 [num_nodes*2]
        @param read_lut_flag 预加载LUT标志（保留参数）
        @return 包含7个输出张量的元组
        """

        
        # 调用底层扩展
        return SteinerTopoFunction.apply(
            pos, 
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            self.ignore_net_degree,
            read_lut_flag
        )

    @property
    def output_names(self):
        """输出张量描述信息"""
        return [
            "wirelength",         # wl
            "steiner_nodes",      # nodes
            "x_pin_mapping",      # pin_relate_x
            "y_pin_mapping",      # pin_relate_y  
            "branch_u",           # branch_u
            "branch_v",           # branch_v
            "net_branch_indices"  # net_branch_start
        ]