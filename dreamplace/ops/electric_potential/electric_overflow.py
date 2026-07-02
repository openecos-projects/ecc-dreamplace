##
# @file   electric_overflow.py
# @author Yibo Lin
# @date   Aug 2018
#

import math
import logging
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

import pdb
import matplotlib
# matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ElectricDensityMapFunction(Function):
    """
    @brief compute density overflow.
    @param ctx pytorch API to store data for backward proporgation
    @param pos location of cells, x and then y
    @param node_size_x_clamped stretched size, max(bin_size*sqrt2, node_size)
    @param node_size_y_clamped stretched size, max(bin_size*sqrt2, node_size)
    @param offset_x (stretched size - node_size) / 2
    @param offset_y (stretched size - node_size) / 2
    @param ratio original area / stretched area
    @param initial_density_map density_map for fixed cells
    @param target_density target density
    @param xl left boundary
    @param yl lower boundary
    @param xh right boundary
    @param yh upper boundary
    @param bin_size_x bin width
    @param bin_size_x bin height
    @param num_movable_nodes number of movable cells
    @param num_filler_nodes number of filler cells
    @param padding bin padding to boundary of placement region
    @param padding_mask padding mask with 0 and 1 to indicate padding bins with padding regions to be 1
    @param num_bins_x number of bins in horizontal direction
    @param num_bins_y number of bins in vertical direction
    @param num_movable_impacted_bins_x number of impacted bins for any movable cell in x direction
    @param num_movable_impacted_bins_y number of impacted bins for any movable cell in y direction
    @param num_filler_impacted_bins_x number of impacted bins for any filler cell in x direction
    @param num_filler_impacted_bins_y number of impacted bins for any filler cell in y direction
    @param sorted_node_map the indices of the movable node map
    """
    @staticmethod
    def forward(
        pos,
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
        sorted_node_map):

        if pos.is_cuda:
            output = electric_potential_cuda.density_map(
                pos.view(pos.numel()), node_size_x_clamped,
                node_size_y_clamped, offset_x, offset_y, ratio, bin_center_x,
                bin_center_y, initial_density_map, target_density, xl, yl, xh,
                yh, bin_size_x, bin_size_y, num_movable_nodes,
                num_filler_nodes, padding, num_bins_x, num_bins_y,
                num_movable_impacted_bins_x, num_movable_impacted_bins_y,
                num_filler_impacted_bins_x, num_filler_impacted_bins_y,
                deterministic_flag, sorted_node_map)
        else:
            output = electric_potential_cpp.density_map(
                pos.view(pos.numel()), node_size_x_clamped,
                node_size_y_clamped, offset_x, offset_y, ratio, bin_center_x,
                bin_center_y, initial_density_map, target_density, xl, yl, xh,
                yh, bin_size_x, bin_size_y, num_movable_nodes,
                num_filler_nodes, padding, num_bins_x, num_bins_y,
                num_movable_impacted_bins_x, num_movable_impacted_bins_y,
                num_filler_impacted_bins_x, num_filler_impacted_bins_y,
                deterministic_flag)

        density_map = output.view([num_bins_x, num_bins_y])
        # set padding density
        if padding > 0:
            density_map.masked_fill_(padding_mask,
                                     target_density * bin_size_x * bin_size_y)

        return density_map


class ElectricOverflow(nn.Module):
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
        num_terminal_NIs=0,
        iopin_density_weight=0.0,
        m2_pg_rail_density_boxes=None,
        m2_pg_rail_density_weight=1.0):
        super(ElectricOverflow, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y

        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y
        self.target_density = target_density
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_filler_nodes = num_filler_nodes
        self.padding = padding
        self.sorted_node_map = sorted_node_map
        self.movable_macro_mask = movable_macro_mask
        self.num_terminal_NIs = int(num_terminal_NIs)
        self.iopin_density_weight = float(iopin_density_weight)
        self.m2_pg_rail_density_boxes = self._normalize_m2_pg_rail_density_boxes(
            m2_pg_rail_density_boxes)
        self.m2_pg_rail_density_weight = self._normalize_m2_pg_rail_density_weight(
            m2_pg_rail_density_weight)

        self.deterministic_flag = deterministic_flag

        self.reset()

    def _normalize_m2_pg_rail_density_boxes(self, boxes):
        if boxes is None:
            return None
        boxes = torch.as_tensor(
            boxes,
            dtype=self.node_size_x.dtype,
            device=self.node_size_x.device,
        )
        if boxes.numel() == 0:
            return boxes.reshape(0, 4).contiguous()
        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError(
                "m2_pg_rail_density_boxes must have shape [N, 4], got %s"
                % (tuple(boxes.size()),)
            )
        if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError(
                "m2_pg_rail_density_boxes must contain positive-width and "
                "positive-height [xl, yl, xh, yh] rectangles"
            )
        return boxes.contiguous()

    def _normalize_m2_pg_rail_density_weight(self, weight):
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError("m2_pg_rail_density_weight must be numeric")
        if not math.isfinite(weight):
            raise ValueError("m2_pg_rail_density_weight must be finite")
        if weight < 0:
            raise ValueError("m2_pg_rail_density_weight must be non-negative")
        return weight

    def reset(self):
        sqrt2 = math.sqrt(2)
        # clamped means stretch a cell to bin size
        # clamped = max(bin_size*sqrt2, node_size)
        # offset means half of the stretch size
        # ratio means the original area over the stretched area
        self.node_size_x_clamped = self.node_size_x.clamp(min=self.bin_size_x *
                                                          sqrt2)
        self.offset_x = (self.node_size_x - self.node_size_x_clamped).mul(0.5)
        self.node_size_y_clamped = self.node_size_y.clamp(min=self.bin_size_y *
                                                          sqrt2)
        self.offset_y = (self.node_size_y - self.node_size_y_clamped).mul(0.5)
        node_areas = self.node_size_x * self.node_size_y
        self.ratio = node_areas / (self.node_size_x_clamped *
                                   self.node_size_y_clamped)

        # detect movable macros and scale down the density to avoid halos
        # the definition of movable macros should be different according to algorithms
        self.num_movable_macros = 0
        if self.target_density < 1 and self.movable_macro_mask is not None:
            self.num_movable_macros = self.movable_macro_mask.sum().data.item()
            self.ratio[:self.num_movable_nodes][
                self.movable_macro_mask] = self.target_density

        # compute maximum impacted bins
        self.num_bins_x = int(round((self.xh - self.xl) / self.bin_size_x))
        self.num_bins_y = int(round((self.yh - self.yl) / self.bin_size_y))
        if self.num_movable_nodes:
            self.num_movable_impacted_bins_x = int(
                ((self.node_size_x[:self.num_movable_nodes].max() +
                  2 * sqrt2 * self.bin_size_x) /
                 self.bin_size_x).ceil().clamp(max=self.num_bins_x))
            self.num_movable_impacted_bins_y = int(
                ((self.node_size_y[:self.num_movable_nodes].max() +
                  2 * sqrt2 * self.bin_size_y) /
                 self.bin_size_y).ceil().clamp(max=self.num_bins_y))
        else:
            self.num_movable_impacted_bins_x = 0
            self.num_movable_impacted_bins_y = 0
        if self.num_filler_nodes:
            self.num_filler_impacted_bins_x = (
                (self.node_size_x[-self.num_filler_nodes:].max() +
                 2 * sqrt2 * self.bin_size_x) /
                self.bin_size_x).ceil().clamp(max=self.num_bins_x)
            self.num_filler_impacted_bins_y = (
                (self.node_size_y[-self.num_filler_nodes:].max() +
                 2 * sqrt2 * self.bin_size_y) /
                self.bin_size_y).ceil().clamp(max=self.num_bins_y)
        else:
            self.num_filler_impacted_bins_x = 0
            self.num_filler_impacted_bins_y = 0
        if self.padding > 0:
            self.padding_mask = torch.ones(self.num_bins_x,
                                           self.num_bins_y,
                                           dtype=torch.uint8,
                                           device=self.node_size_x.device)
            self.padding_mask[self.padding:self.num_bins_x - self.padding,
                              self.padding:self.num_bins_y -
                              self.padding].fill_(0)
        else:
            self.padding_mask = torch.zeros(self.num_bins_x,
                                            self.num_bins_y,
                                            dtype=torch.uint8,
                                            device=self.node_size_x.device)
        # initial density_map due to fixed cells
        self.initial_density_map = None
        self.m2_pg_rail_fixed_without_rail_raw_map = None
        self.m2_pg_rail_combined_raw_map = None
        self.m2_pg_rail_raw_delta_map = None

    def _fixed_density_map(self, pos, node_size_x, node_size_y, num_movable_nodes,
                           num_terminals):
        if num_terminals == 0:
            num_fixed_impacted_bins_x = 0
            num_fixed_impacted_bins_y = 0
        else:
            fixed_beg = num_movable_nodes
            fixed_end = num_movable_nodes + num_terminals
            max_size_x = node_size_x[fixed_beg:fixed_end].max()
            max_size_y = node_size_y[fixed_beg:fixed_end].max()
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
        return func(
            pos, node_size_x, node_size_y, self.bin_center_x,
            self.bin_center_y, self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y, num_movable_nodes,
            num_terminals, self.num_bins_x, self.num_bins_y,
            num_fixed_impacted_bins_x, num_fixed_impacted_bins_y,
            self.deterministic_flag)

    def compute_iopin_density_map(self, pos):
        num_nodes = pos.numel() // 2
        iopin_start = self.num_movable_nodes + self.num_terminals
        iopin_end = iopin_start + self.num_terminal_NIs
        packed_pos = torch.cat([
            pos[iopin_start:iopin_end],
            pos[num_nodes + iopin_start:num_nodes + iopin_end],
        ]).contiguous()
        packed_node_size_x = self.node_size_x[iopin_start:iopin_end].contiguous()
        packed_node_size_y = self.node_size_y[iopin_start:iopin_end].contiguous()
        return self._fixed_density_map(
            packed_pos,
            packed_node_size_x,
            packed_node_size_y,
            num_movable_nodes=0,
            num_terminals=self.num_terminal_NIs)

    def _pack_combined_fixed_density_terms(self, pos):
        rail_boxes = self.m2_pg_rail_density_boxes
        if rail_boxes is None or rail_boxes.numel() == 0:
            return (
                pos,
                self.node_size_x,
                self.node_size_y,
                self.num_movable_nodes,
                self.num_terminals,
            )

        num_nodes = pos.numel() // 2
        fixed_beg = self.num_movable_nodes
        fixed_end = self.num_movable_nodes + self.num_terminals
        packed_pos = torch.cat(
            [
                pos[fixed_beg:fixed_end],
                rail_boxes[:, 0],
                pos[num_nodes + fixed_beg:num_nodes + fixed_end],
                rail_boxes[:, 1],
            ]
        ).contiguous()
        packed_node_size_x = torch.cat(
            [
                self.node_size_x[fixed_beg:fixed_end],
                rail_boxes[:, 2] - rail_boxes[:, 0],
            ]
        ).contiguous()
        packed_node_size_y = torch.cat(
            [
                self.node_size_y[fixed_beg:fixed_end],
                rail_boxes[:, 3] - rail_boxes[:, 1],
            ]
        ).contiguous()
        return (
            packed_pos,
            packed_node_size_x,
            packed_node_size_y,
            0,
            self.num_terminals + rail_boxes.size(0),
        )

    def compute_initial_density_map(self, pos):
        fixed_without_rail_raw_map = self._fixed_density_map(
            pos, self.node_size_x, self.node_size_y, self.num_movable_nodes,
            self.num_terminals)
        self.m2_pg_rail_fixed_without_rail_raw_map = fixed_without_rail_raw_map.clone()
        if (
            self.m2_pg_rail_density_boxes is not None
            and self.m2_pg_rail_density_boxes.numel() > 0
            and self.m2_pg_rail_density_weight > 0
        ):
            (
                packed_pos,
                packed_node_size_x,
                packed_node_size_y,
                packed_num_movable_nodes,
                packed_num_terminals,
            ) = self._pack_combined_fixed_density_terms(pos)
            combined_raw_map = self._fixed_density_map(
                packed_pos,
                packed_node_size_x,
                packed_node_size_y,
                packed_num_movable_nodes,
                packed_num_terminals,
            )
        else:
            combined_raw_map = fixed_without_rail_raw_map
        self.m2_pg_rail_combined_raw_map = combined_raw_map.clone()
        self.m2_pg_rail_raw_delta_map = (
            combined_raw_map - fixed_without_rail_raw_map
        ).clone()
        if (
            self.m2_pg_rail_density_boxes is not None
            and self.m2_pg_rail_density_boxes.numel() > 0
            and self.m2_pg_rail_density_weight > 0
        ):
            logger.info(
                "M2 PG rail soft density map: boxes=%d, weight=%g, combined_raw_sum=%g, "
                "combined_raw_max=%g, raw_delta_sum=%g, raw_delta_max=%g",
                int(self.m2_pg_rail_density_boxes.size(0)),
                self.m2_pg_rail_density_weight,
                float(self.m2_pg_rail_combined_raw_map.sum().item()),
                float(self.m2_pg_rail_combined_raw_map.max().item()),
                float(self.m2_pg_rail_raw_delta_map.sum().item()),
                float(self.m2_pg_rail_raw_delta_map.max().item()),
            )
        if self.m2_pg_rail_density_weight == 1.0:
            self.initial_density_map = combined_raw_map.clone()
        elif self.m2_pg_rail_density_weight > 0:
            self.initial_density_map = fixed_without_rail_raw_map.add(
                self.m2_pg_rail_raw_delta_map,
                alpha=self.m2_pg_rail_density_weight,
            )
        else:
            self.initial_density_map = fixed_without_rail_raw_map.clone()
        # scale density of fixed macros
        self.initial_density_map.mul_(self.target_density)
        if self.iopin_density_weight > 0 and self.num_terminal_NIs > 0:
            logger.info(
                "I/O pin density increment enabled: weight=%s, num_terminal_NIs=%d",
                self.iopin_density_weight,
                self.num_terminal_NIs,
            )
            self.initial_density_map.add_(
                self.compute_iopin_density_map(pos),
                alpha=self.iopin_density_weight)

    def forward(self, pos):
        if self.initial_density_map is None:
            self.compute_initial_density_map(pos)

        density_map = ElectricDensityMapFunction.forward(
            pos, self.node_size_x_clamped, self.node_size_y_clamped,
            self.offset_x, self.offset_y, self.ratio, self.bin_center_x,
            self.bin_center_y, self.initial_density_map, self.target_density,
            self.xl, self.yl, self.xh, self.yh, self.bin_size_x,
            self.bin_size_y, self.num_movable_nodes, self.num_filler_nodes,
            self.padding, self.padding_mask, self.num_bins_x, self.num_bins_y,
            self.num_movable_impacted_bins_x, self.num_movable_impacted_bins_y,
            self.num_filler_impacted_bins_x, self.num_filler_impacted_bins_y,
            self.deterministic_flag, self.sorted_node_map)
        bin_area = self.bin_size_x * self.bin_size_y
        density_cost = (density_map -
                        self.target_density * bin_area).clamp_(min=0.0).sum().unsqueeze(0)

        return density_cost, density_map.max().unsqueeze(0) / bin_area


def plot(plot_count, density_map, padding, name):
    """
    density map contour and heat map
    """
    density_map = density_map[padding:density_map.shape[0] - padding,
                              padding:density_map.shape[1] - padding]
    print("max density = %g @ %s" %
          (np.amax(density_map),
           np.unravel_index(np.argmax(density_map), density_map.shape)))
    print("mean density = %g" % (np.mean(density_map)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    # looks like x and y should be swapped
    ax.plot_surface(y, x, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    # plt.tight_layout()
    plt.savefig(name + ".3d.png")
    plt.close()

    # plt.clf()

    #fig, ax = plt.subplots()

    # ax.pcolor(density_map)

    # Loop over data dimensions and create text annotations.
    # for i in range(density_map.shape[0]):
    # for j in range(density_map.shape[1]):
    # text = ax.text(j, i, density_map[i, j],
    # ha="center", va="center", color="w")
    # fig.tight_layout()
    #plt.savefig(name+".2d.%d.png" % (plot_count))
    # plt.close()
