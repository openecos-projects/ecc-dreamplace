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
    def forward(ctx, pos, pin_relate_x, pin_relate_y,
                net_vertex_start, num_vertices):

        updated_newx, updated_newy = steiner_topo_cpp.forward(
            pos,
            pin_relate_x.contiguous(),
            pin_relate_y.contiguous(),
            num_vertices
        )

        ctx.save_for_backward(pos,
                              net_vertex_start.contiguous(),
                              pin_relate_x.contiguous(),
                              pin_relate_y.contiguous()
                              )

        return updated_newx, updated_newy

    @staticmethod
    def backward(ctx, grad_newx, grad_newy):

        pos, net_vertex_start, pin_relate_x, pin_relate_y = ctx.saved_tensors
        grad_pos = steiner_topo_cpp.backward(
            grad_newx,
            grad_newy,
            pos,
            pin_relate_x,
            pin_relate_y
        )

        return grad_pos, None, None, None, None, None, None


class SteinerTopo(nn.Module):
    def __init__(self,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 ignore_net_degree=None,
                 algorithm="FLUTE"):
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
        self.net_vertex_start = None
        self.net_steiner_start = None
        self.pin_fa = None
        self.flat_pin_to = None
        self.flat_pin_from = None
        self.flat_pin_to_start = None
        self.net_flat_topo_sort = None
        self.net_flat_topo_sort_start = None
        self.num_vertices = None

        self.algorithm = algorithm

    def forward(self, pos):

        if self.pin_relate_x is None or self.pin_relate_y is None \
           or self.net_vertex_start is None \
           or self.num_vertices is None:
            raise RuntimeError(
                "SteinerTopo topology not initialized. Call rebuild_tree and update_topology first.")

        updated_newx, updated_newy = SteinerTopoFunction.apply(
            pos,
            self.pin_relate_x,
            self.pin_relate_y,
            self.net_vertex_start,
            self.num_vertices
        )
        # outputs = (
        #     updated_newx,
        #     updated_newy,
        #     self.net_flat_topo_sort,
        #     self.net_flat_topo_sort_start,
        #     self.pin_fa,
        #     self.flat_pin_to,
        #     self.flat_pin_to_start,
        #     self.flat_pin_from
        # )
        return updated_newx, updated_newy

    def update_cache(self, build_tree_outputs):

        (self.newx, self.newy, self.pin_relate_x, self.pin_relate_y,
            self.net_vertex_start, self.net_steiner_start,
            self.pin_fa, self.flat_pin_to, self.flat_pin_from, self.flat_pin_to_start,
            self.net_flat_topo_sort, self.net_flat_topo_sort_start) = build_tree_outputs

        self.num_vertices = self.newx.numel()
        self.pin_relate_x = self.pin_relate_x.contiguous()
        self.pin_relate_y = self.pin_relate_y.contiguous()
        self.net_vertex_start = self.net_vertex_start.contiguous()
        self.net_steiner_start = self.net_steiner_start.contiguous()
        self.pin_fa = self.pin_fa.contiguous()
        self.flat_pin_to = self.flat_pin_to.contiguous()
        self.flat_pin_from = self.flat_pin_from.contiguous()
        self.flat_pin_to_start = self.flat_pin_to_start.contiguous()
        self.net_flat_topo_sort = self.net_flat_topo_sort.contiguous()
        self.net_flat_topo_sort_start = self.net_flat_topo_sort_start.contiguous()

    def rebuild_tree(self, pos):

        new_outputs_tuple = steiner_topo_cpp.build_tree(
            pos,
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            self.ignore_net_degree
        )

        self.update_cache(new_outputs_tuple)
        return self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from


'''
self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from

'''
