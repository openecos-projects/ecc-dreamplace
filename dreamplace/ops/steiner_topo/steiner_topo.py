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
    def forward(ctx, pos, flat_netpin, netpin_start,
                # Topology information passed as arguments
                pin_relate_x, pin_relate_y, local2global_index,
                net_vertex_start, num_total_vertices):

        updated_newx, updated_newy = steiner_topo_cpp.forward(
            pos,
            pin_relate_x.contiguous(),
            pin_relate_y.contiguous(),
            local2global_index.contiguous(),
            netpin_start.contiguous(),
            net_vertex_start.contiguous(),
            num_total_vertices
        )

        ctx.save_for_backward(pos,
                              netpin_start,
                              net_vertex_start.contiguous(),
                              pin_relate_x.contiguous(),
                              pin_relate_y.contiguous(),
                              local2global_index.contiguous()
                              )

        return updated_newx, updated_newy

    @staticmethod
    def backward(ctx, grad_newx, grad_newy):

        grad_vertices = torch.cat([grad_newx, grad_newy], dim=0).contiguous()

        grad_pos = steiner_topo_cpp.backward(
            grad_vertices,
            ctx.pos,
            ctx.pin_relate_x,
            ctx.pin_relate_y,
            ctx.netpin_start,
            ctx.net_vertex_start,
            ctx.local2global_index
        )

        return grad_pos, None, None, None, None, None, None, None


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
        @param compute_interval int: Recompute topology every N iterations.
        """
        super(SteinerTopo, self).__init__()
        # Register buffers
        self.register_buffer('flat_net2pin_map', flat_net2pin_map.contiguous())
        self.register_buffer('flat_net2pin_start_map',
                             flat_net2pin_start_map.contiguous())

        # Set ignore degree threshold
        self.ignore_net_degree = ignore_net_degree if ignore_net_degree is not None else flat_net2pin_map.numel()

        self.newx = None
        self.newy = None
        self.pin_relate_x = None
        self.pin_relate_y = None
        self.local2global_index = None
        self.net_vertex_start = None
        self.net_steiner_start = None
        self.pin_fa = None
        self.flat_pin_to = None
        self.flat_pin_from = None
        self.flat_pin_to_start = None
        self.net_flat_topo_sort = None
        self.net_flat_topo_sort_start = None
        self.num_total_vertices = None

        self.algorithm = algorithm

    def forward(self, pos):

        if self.pin_relate_x is None or self.pin_relate_y is None \
           or self.local2global_index is None or self.net_vertex_start is None \
           or self.num_total_vertices is None:
            raise RuntimeError(
                "SteinerTopo topology not initialized. Call rebuild_tree and update_topology first.")

        outputs = SteinerTopoFunction.apply(
            pos,
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            # Topology info stored in self:
            self.pin_relate_x,
            self.pin_relate_y,
            self.local2global_index,
            self.net_vertex_start,
            self.num_total_vertices
        )
        return outputs

    def update_cache(self, build_tree_outputs):

        (self.newx, self.newy, self.pin_relate_x, self.pin_relate_y,
            self.local2global_index, self.net_vertex_start, self.net_steiner_start,
            self.pin_fa, self.flat_pin_to, self.flat_pin_from, self.flat_pin_to_start,
            self.net_flat_topo_sort, self.net_flat_topo_sort_start) = build_tree_outputs

        self.num_total_vertices = self.newx.numel()
        self.pin_relate_x = self.pin_relate_x.contiguous()
        self.pin_relate_y = self.pin_relate_y.contiguous()
        self.net_vertex_start = self.net_vertex_start.contiguous()
        self.local2global_index = self.local2global_index.contiguous()

    def rebuild_tree(self, pos):

        print("SteinerTopo: Rebuilding tree...")

        new_outputs_tuple = steiner_topo_cpp.build_tree(
            pos,
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            self.ignore_net_degree
        )

        self.update_cache(new_outputs_tuple)
        print("SteinerTopo: Tree rebuild complete and cache updated.")
        return self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from

    def clear_cache(self):
        """Clears the internal topology cache."""
        # print("Clearing SteinerTopo cache.")
        self.former_outputs['steiner'] = None

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
