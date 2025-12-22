##
# @file   nctugr_binary.py
# @author Yibo Lin
# @date   Jan 2020
#

import os
import stat
import sys
import logging
import torch
from torch.autograd import Function
from torch import nn
import pdb
import numpy as np

import dreamplace.ops.place_io.place_io as place_io

logger = logging.getLogger(__name__)


class IRT_eGR(object):
    def __init__(self, params, placedb):
        self.params = params
        self.placedb = placedb

    def __call__(self, pos, stage, resolve_congestion):
        return self.forward(pos, stage, resolve_congestion)

    def forward(self, pos, stage, resolve_congestion):
        if pos.is_cuda:
            pos_cpu = pos.cpu().data.numpy().copy()
        else:
            pos_cpu = pos.data.numpy().copy()

        num_nodes = pos.numel() // 2
        node_x = pos_cpu[: self.placedb.num_movable_nodes]
        node_y = pos_cpu[
            self.placedb.num_nodes : self.placedb.num_nodes
            + self.placedb.num_movable_nodes
        ]
        if self.params.cell_padding_x >= 0:
            node_x += self.params.cell_padding_x

        # unscale locations
        unscale_factor = 1.0 / self.params.scale_factor
        node_x = (
            node_x[: self.placedb.num_movable_nodes] * unscale_factor
            + self.params.shift_factor[0]
        )
        node_y = (
            node_y[: self.placedb.num_movable_nodes] * unscale_factor
            + self.params.shift_factor[1]
        )

        # update raw database
        self.placedb.write_placement_back(node_x, node_y)
        utilization_map_py = self.placedb.pydb.getCongestionMap("max", stage, resolve_congestion)
        utilization_map_np = np.array(utilization_map_py, dtype=np.float32)
        utilization_map = torch.from_numpy(utilization_map_np).to(pos.device).T
        utilization_map = utilization_map.contiguous()
        # congestion_map = torch.zeros(
        #     (
        #         self.placedb.num_routing_grids_x,
        #         self.placedb.num_routing_grids_y,
        #         self.placedb.num_routing_layers,
        #     ),
        #     dtype=pos.dtype,
        # )

        # utilization_map = (
        #     utilization_map + 1
        # )
        

        return utilization_map
