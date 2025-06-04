#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : MacroPlaceDB.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        :
@version      : 0.1
@date         : 2023-10-13 10:58:51
'''

import sys
import os
import re
import math
import time
import numpy as np
import logging
import pdb
import itertools

from tools.iEDA.data.design import IEDADesign
from tools.iEDA.data.idm import IEDADataManager
from tools.iEDA.module.io import IEDAIO
# import macro_placer.database.fence_region.fence_region as fence_region

datatypes = {
    'float32': np.float32,
    'float64': np.float64
}


class MacroPlaceDB(object):
    """
    @brief placement database
    """

    def __init__(self, data_manager: IEDAIO):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.data_manager = data_manager
        # self.rawdb = None # raw placement database, a C++ object

        # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_physical_nodes = 0
        self.num_terminals = 0  # number of terminals, essentially fixed macros
        # number of terminal_NIs that can be overlapped, essentially IO pins
        self.num_terminal_NIs = 0
        self.node_name2id_map = {}  # node name to id map, cell name
        self.node_names = None  # 1D array, cell name
        self.node_x = None  # 1D array, cell position x
        self.node_y = None  # 1D array, cell position y
        self.node_orient = None  # 1D array, cell orientation
        self.node_size_x = None  # 1D array, cell width
        self.node_size_y = None  # 1D array, cell height

        # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
        self.node2orig_node_map = None
        # this map maps the current multiple node ids into the original one

        self.pin_direct = None  # 1D array, pin direction IO
        self.pin_offset_x = None  # 1D array, pin offset x to its node
        self.pin_offset_y = None  # 1D array, pin offset y to its node

        self.net_name2id_map = {}  # net name to id map
        self.net_names = None  # net name
        self.net_weights = None  # weights for each net

        self.net2pin_map = None  # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None  # flatten version of net2pin_map
        # starting index of each net in flat_net2pin_map
        self.flat_net2pin_start_map = None

        self.node2pin_map = None  # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = None  # flatten version of node2pin_map
        # starting index of each node in flat_node2pin_map
        self.flat_node2pin_start_map = None

        self.pin2node_map = None  # 1D array, contain parent node id of each pin
        self.pin2net_map = None  # 1D array, contain parent net id of each pin

        self.rows = None  # NumRows x 4 array, stores xl, yl, xh, yh of each row

        self.regions = None  # array of 1D array, placement regions like FENCE and GUIDE
        self.flat_region_boxes = None  # flat version of regions
        # start indices of regions, length of num regions + 1
        self.flat_region_boxes_start = None
        # map cell to a region, maximum integer if no fence region
        self.node2fence_region_map = None

        self.xl = None
        self.yl = None
        self.xh = None
        self.yh = None

        self.row_height = None
        self.site_width = None

        self.bin_size_x = None
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None
        self.bin_center_x = None
        self.bin_center_y = None

        self.num_movable_pins = None

        self.total_movable_node_area = None  # total movable cell area
        self.total_fixed_node_area = None  # total fixed cell area
        self.total_space_area = None  # total placeable space area excluding fixed cells

        # enable filler cells
        # the Idea from e-place and RePlace
        self.total_filler_node_area = None
        self.num_filler_nodes = None

        self.routing_grid_xl = None
        self.routing_grid_yl = None
        self.routing_grid_xh = None
        self.routing_grid_yh = None
        self.num_routing_grids_x = None
        self.num_routing_grids_y = None
        self.num_routing_layers = None
        # per unit distance, projected to one layer
        self.unit_horizontal_capacity = None
        self.unit_vertical_capacity = None  # per unit distance, projected to one layer
        self.unit_horizontal_capacities = None  # per unit distance, layer by layer
        self.unit_vertical_capacities = None  # per unit distance, layer by layer
        # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer
        self.initial_horizontal_demand_map = None
        # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer
        self.initial_vertical_demand_map = None

        self.is_pin_lower_x = None  # macro pin is on the lower edge of the macro
        self.is_pin_upper_x = None  # macro pin is on the upper edge of the macro
        self.is_pin_lower_y = None
        self.is_pin_upper_y = None
        self.dtype = None
        self.pydb = None

        # Timing model
        self.start_points = None
        self.end_points = None
        # self.cells_by_level = None
        # self.cells_by_reverse_level = None

        self.flat_cells_by_level = None  # //
        self.flat_cells_by_reverse_level = None  # //
        self.flat_cells_by_level_start = None  # //
        self.flat_cells_by_reverse_level_start = None  # //

        self.inrdelays = None
        self.infdelays = None
        self.inrtrans = None
        self.inftrans = None
        self.outcaps = None

        self.net_flat_arcs_start = None
        self.net_flat_arcs = None
        self.cell_flat_arcs_start = None
        self.cell_flat_arcs = None

        self.main_id_2_cell_id_start = None
        self.cell_id_2_arc_id_start = None

        self.inst_main_id = None
        self.inst_size = None

        # LUTs table
        self.f_delay_flat_luts_values = None
        self.f_delay_flat_luts_trans_table = None
        self.f_delay_flat_luts_cap_table = None
        self.f_delay_flat_luts_dim = None

        self.r_delay_flat_luts_values = None
        self.r_delay_flat_luts_trans_table = None
        self.r_delay_flat_luts_cap_table = None
        self.r_delay_flat_luts_dim = None

        self.f_trans_flat_luts_values = None
        self.f_trans_flat_luts_trans_table = None
        self.f_trans_flat_luts_cap_table = None
        self.f_trans_flat_luts_dim = None

        self.r_trans_flat_luts_values = None
        self.r_trans_flat_luts_trans_table = None
        self.r_trans_flat_luts_cap_table = None
        self.r_trans_flat_luts_dim = None

    def scale_pl(self, scale_factor):
        """
        @brief scale placement solution only
        @param scale_factor scale factor 
        """
        self.node_x *= scale_factor
        self.node_y *= scale_factor

    def scale(self, shift_factor, scale_factor):
        """
        @brief shift and scale coordinates
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        logging.info(
            "shift coordinate system by (%g, %g), scale coordinate system by %g"
            % (shift_factor[0], shift_factor[1], scale_factor)
        )

        # node positions
        self.node_x -= shift_factor[0]
        self.node_x *= scale_factor
        self.node_y -= shift_factor[1]
        self.node_y *= scale_factor

        # node sizes
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor

        # pin offsets
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor

        # floorplan
        self.xl -= shift_factor[0]
        self.xl *= scale_factor
        self.yl -= shift_factor[1]
        self.yl *= scale_factor
        self.xh -= shift_factor[0]
        self.xh *= scale_factor
        self.yh -= shift_factor[1]
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor

        # # bin
        # self.bin_size_x *= scale_factor
        # self.bin_size_y *= scale_factor

        # routing
        self.routing_grid_xl -= shift_factor[0]
        self.routing_grid_xl *= scale_factor
        self.routing_grid_yl -= shift_factor[1]
        self.routing_grid_yl *= scale_factor
        self.routing_grid_xh -= shift_factor[0]
        self.routing_grid_xh *= scale_factor
        self.routing_grid_yh -= shift_factor[1]
        self.routing_grid_yh *= scale_factor
        self.routing_V *= scale_factor
        self.routing_H *= scale_factor
        self.macro_util_V *= scale_factor
        self.macro_util_H *= scale_factor

        # shift factor for rectangle
        box_shift_factor = np.array(
            [shift_factor, shift_factor], dtype=self.rows.dtype
        ).reshape(1, -1)

        # placement rows
        self.rows -= box_shift_factor
        self.rows *= scale_factor
        self.total_space_area *= scale_factor * scale_factor

        # regions
        if len(self.flat_region_boxes) > 0:
            self.flat_region_boxes -= box_shift_factor
            self.flat_region_boxes *= scale_factor
        for i in range(len(self.regions)):
            # may have performance issue
            self.regions[i] -= box_shift_factor
            self.regions[i] *= scale_factor

    def setup_rawdb(self, params):
        self.dtype = datatypes[params.dtype]
        if self.pydb is None:
            ieda_dm = IEDADataManager(self.data_manager.dir_workspace)
            self.get_dmInst_ptr = ieda_dm.get_dmInst_ptr()
            self.pydb = ieda_dm.pydb(self.get_dmInst_ptr, params.with_sta)

    def init_db(self, params):
        self.setup_rawdb(params)
        self.initialize_from_rawdb(self.pydb, params)
        # if params.with_sta:
        # self.virtual_net_init()
        self.initialize(params)
        self.params = params
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        print("net_degrees max{} min{}", max(net_degrees), min(net_degrees))

    def clustering(self, cluster_config):
        pass

    def sort(self):
        """
        @brief Sort net by degree. 
        Sort pin array such that pins belonging to the same net is abutting each other
        """
        logging.info("sort nets by degree and pins by net")

        # sort nets by degree
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        net_order = net_degrees.argsort()  # indexed by new net_id, content is old net_id
        self.net_names = self.net_names[net_order]
        self.net2pin_map = self.net2pin_map[net_order]
        for net_id, net_name in enumerate(self.net_names):
            self.net_name2id_map[net_name] = net_id
        for new_net_id in range(len(net_order)):
            for pin_id in self.net2pin_map[new_net_id]:
                self.pin2net_map[pin_id] = new_net_id
        # check
        # for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id

        # sort pins such that pins belonging to the same net is abutting each other
        # indexed new pin_id, content is old pin_id
        pin_order = self.pin2net_map.argsort()
        self.pin2net_map = self.pin2net_map[pin_order]
        self.pin2node_map = self.pin2node_map[pin_order]
        self.pin_direct = self.pin_direct[pin_order]
        self.pin_offset_x = self.pin_offset_x[pin_order]
        self.pin_offset_y = self.pin_offset_y[pin_order]
        old2new_pin_id_map = np.zeros(len(pin_order), dtype=np.int32)
        for new_pin_id in range(len(pin_order)):
            old2new_pin_id_map[pin_order[new_pin_id]] = new_pin_id
        for i in range(len(self.net2pin_map)):
            for j in range(len(self.net2pin_map[i])):
                self.net2pin_map[i][j] = old2new_pin_id_map[self.net2pin_map[i][j]]
        for i in range(len(self.node2pin_map)):
            for j in range(len(self.node2pin_map[i])):
                self.node2pin_map[i][j] = old2new_pin_id_map[self.node2pin_map[i][j]]
        # check
        # for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id
        # for node_id in range(len(self.node2pin_map)):
        #    for j in range(len(self.node2pin_map[node_id])):
        #        assert self.pin2node_map[self.node2pin_map[node_id][j]] == node_id

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes 
        """
        return self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs

    @property
    def num_nodes(self):
        """
        @return number of movable nodes, terminals, terminal_NIs, and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout 
        """
        return self.xh - self.xl

    @property
    def height(self):
        """
        @return height of layout 
        """
        return self.yh - self.yl

    @property
    def area(self):
        """
        @return area of layout 
        """
        return self.width * self.height

    def bin_index_x(self, x):
        """
        @param x horizontal location 
        @return bin index in x direction 
        """
        if x < self.xl:
            return 0
        elif x > self.xh:
            return int(np.floor((self.xh - self.xl) / self.bin_size_x))
        else:
            return int(np.floor((x - self.xl) / self.bin_size_x))

    def bin_index_y(self, y):
        """
        @param y vertical location 
        @return bin index in y direction 
        """
        if y < self.yl:
            return 0
        elif y > self.yh:
            return int(np.floor((self.yh - self.yl) / self.bin_size_y))
        else:
            return int(np.floor((y - self.yl) / self.bin_size_y))

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index 
        @return bin xl
        """
        return self.xl + id_x * self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index 
        @return bin xh
        """
        return min(self.bin_xl(id_x) + self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index 
        @return bin yl
        """
        return self.yl + id_y * self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index 
        @return bin yh
        """
        return min(self.bin_yl(id_y) + self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins 
        @param l lower bound 
        @param h upper bound 
        @param bin_size bin size 
        @return number of bins 
        """
        return int(np.ceil((h - l) / bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers 
        @param l lower bound 
        @param h upper bound 
        @param bin_size bin size 
        @return array of bin centers 
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins):
            bin_l = l + id_x * bin_size
            bin_h = min(bin_l + bin_size, h)
            centers[id_x] = (bin_l + bin_h) / 2
        return centers

    @property
    def routing_grid_size_x(self):
        return (self.routing_grid_xh - self.routing_grid_xl) / self.num_routing_grids_x

    @property
    def routing_grid_size_y(self):
        return (self.routing_grid_yh - self.routing_grid_yl) / self.num_routing_grids_y

    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net 
        @param x horizontal cell locations 
        @param y vertical cell locations
        @return hpwl of a net 
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes] + self.pin_offset_x[pins]) - \
            np.amin(x[nodes] + self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes] + self.pin_offset_y[pins]) - \
            np.amin(y[nodes] + self.pin_offset_y[pins])

        return (hpwl_x + hpwl_y) * self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes 
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2) - max(xl1, xl2), 0.0) * max(min(yh1, yh2) - max(yl1, yl2), 0.0)

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @return density map 
        """
        bin_index_xl = np.maximum(
            np.floor(x / self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(np.ceil(
            (x + self.node_size_x) / self.bin_size_x).astype(np.int32), self.num_bins_x - 1)
        bin_index_yl = np.maximum(
            np.floor(y / self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(np.ceil(
            (y + self.node_size_y) / self.bin_size_y).astype(np.int32), self.num_bins_y - 1)

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id] + 1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id] + 1):
                    density_map[ix, iy] += self.overlap(
                        self.bin_xl(ix), self.bin_yl(
                            iy), self.bin_xh(ix), self.bin_yh(iy),
                        x[node_id], y[node_id], x[node_id] +
                        self.node_size_x[node_id], y[node_id] +
                        self.node_size_y[node_id]
                    )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix) - self.bin_xl(ix)) * \
                    (self.bin_yh(iy) - self.bin_yl(iy))

        return density_map

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @param target_density target density 
        @return density overflow cost 
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map - target_density, 0.0)))

    def print_node(self, node_id):
        """
        @brief print node information 
        @param node_id cell index 
        """
        logging.debug("node %s(%d), size (%g, %g), pos (%g, %g)" % (
            self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]],
                                     self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index 
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]],
                                     self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_row(self, row_id):
        """
        @brief print row information 
        @param row_id row index 
        """
        logging.debug("row %d %s" % (row_id, self.rows[row_id]))

    def virtual_net_init(self):
        max_hop = 2
        print("build macro connections Begin")
        ieda_design = IEDADesign(self.data_manager.dir_workspace)
        macro_connections = ieda_design.build_macro_connection_map(max_hop)
        print("build macro connections finished")
        print(f" self.num_physical_nodes =  {self.num_physical_nodes}")
        print(f" self.row_height =  {self.row_height}")
        macro_id_set = set()
        for i in range(self.num_physical_nodes):
            if (self.node_size_y[i] > self.row_height):
                macro_id_set.add(i)

        print(f"macro_id_set: {len(macro_id_set)}")

        # macro_pair_weights = {}

        # # 遍历 macro_id_set 中的所有两两配对
        # for macro_id_pair in itertools.combinations(macro_id_set, 2):
        #     macro1_id, macro2_id = macro_id_pair
        #     # 对每个配对进行处理
        #     # print(f"Processing pair: {macro1_id}, {macro2_id}")
        #     macro_1_name, macro_2_name = self.node_names[macro1_id], self.node_names[macro2_id]
        #     if isinstance(macro_1_name, bytes):
        #         macro_1_name = macro_1_name.decode('utf-8')
        #     if isinstance(macro_2_name, bytes):
        #         macro_2_name = macro_2_name.decode('utf-8')
        #     # print(f"Name: {macro_1_name} for Macros {macro_2_name}")
        #     macro_1_parent = macro_1_name.rsplit('/', 1)[0] if '/' in macro_1_name else ''
        #     macro_2_parent = macro_2_name.rsplit('/', 1)[0] if '/' in macro_2_name else ''

        #     # 如果上一级名称不为空且相同，认为在同个module
        #     if macro_1_parent and macro_1_parent == macro_2_parent:
        #         print(f"Same parent: {macro_1_parent} for Macros {macro_2_parent}")
        #         macro_pair = tuple(sorted([macro_1_name, macro_2_name]))

        #         if macro_pair not in macro_pair_weights:
        #             macro_pair_weights[macro_pair] = 1

        # macro_group_weights = {}

        # # 遍历 macro_id_set 中的所有 macro_id
        # for macro_id in macro_id_set:
        #     macro_name = self.node_names[macro_id]
        #     if isinstance(macro_name, bytes):
        #         macro_name = macro_name.decode('utf-8')

        #     macro_parent = macro_name.rsplit('/', 1)[0] if '/' in macro_name else ''

        #     # 如果上一级名称不为空，则将其加入对应的分组
        #     if macro_parent:
        #         if macro_parent not in macro_group_weights:
        #             macro_group_weights[macro_parent] = {'macros': [], 'weight': 15000}
        #         macro_group_weights[macro_parent]['macros'].append(macro_name)

        # for macro_id in macro_id_set:
        #     macro_name = self.node_names[macro_id]
        #     if isinstance(macro_name, bytes):
        #         macro_name = macro_name.decode('utf-8')

        #     # 获取最前面的一级名称
        #     macro_parent = macro_name.split('/', 1)[0] if '/' in macro_name else ''

        #     # 如果上一级名称不为空，则将其加入对应的分组
        #     if macro_parent:
        #         if macro_parent not in macro_group_weights:
        #             macro_group_weights[macro_parent] = {'macros': [], 'weight': 5}
        #         macro_group_weights[macro_parent]['macros'].append(macro_name)

        # group_count = sum(1 for group in macro_group_weights.values() if len(group['macros']) > 1)
        # print(f"Groups with more than 1 macro: {group_count}")

        # self.net2pin_map = list(self.net2pin_map)
        # for parent, data in macro_group_weights.items():
        #     if len(data['macros']) > 1:
        #         weight = data['weight']
        #         macros = data['macros']
        #         net_id = len(self.flat_net2pin_start_map)

        #         pin_ids = []
        #         for macro_name in macros:
        #             inst_id = self.node_name2id_map[macro_name]

        #             pin_id = len(self.pin_offset_x)
        #             self.pin_offset_x = np.append(self.pin_offset_x, self.node_size_x[inst_id] / 2)
        #             self.pin_offset_y = np.append(self.pin_offset_y, self.node_size_y[inst_id] / 2)
        #             self.node2pin_map[inst_id] = np.array(np.append(self.node2pin_map[inst_id], pin_id), dtype=np.int32)
        #             self.pin2node_map = np.append(self.pin2node_map, inst_id)
        #             self.pin2net_map = np.append(self.pin2net_map, net_id)

        #             pin_ids.append(pin_id)

        #         self.flat_net2pin_start_map = np.append(self.flat_net2pin_start_map, int(len(self.flat_net2pin_map)))
        #         for pin_id in pin_ids:
        #             self.flat_net2pin_map = np.append(self.flat_net2pin_map, pin_id)
        #         self.net2pin_map.append(np.array(pin_ids, dtype=np.int32))
        #         self.net_weights = np.append(self.net_weights, weight)

        # 上面是根据层次化的

        if len(macro_connections) == 0:
            print("macro_connections 是空的")
        else:
            print(f"macro_connections {len(macro_connections)} 不是空的")
        for macro_connection in macro_connections:
            print(
                "src macro name {} -> snk macro name {} stages {} hop {}".format(
                    macro_connection.src_macro_name,
                    macro_connection.dst_macro_name,
                    " ".join([str(x)
                             for x in macro_connection.stages_each_hop]),
                    macro_connection.hop,
                )
            )

        macro_connection_dict = {}
        macro_set = set()
        macro_name2pin_id_map = {}
        for macro_connection in macro_connections:
            macro_connection.src_macro_name = macro_connection.src_macro_name.replace(
                '[', '\\[')
            macro_connection.src_macro_name = macro_connection.src_macro_name.replace(
                ']', '\\]')
            macro_connection.dst_macro_name = macro_connection.dst_macro_name.replace(
                '[', '\\[')
            macro_connection.dst_macro_name = macro_connection.dst_macro_name.replace(
                ']', '\\]')

            src = macro_connection.src_macro_name
            dst = macro_connection.dst_macro_name
            macro_set.add(src)
            macro_set.add(dst)
            if src not in macro_connection_dict:
                macro_connection_dict[src] = {dst: [0] * max_hop}
            elif dst not in macro_connection_dict[src]:
                macro_connection_dict[src][dst] = [0] * max_hop

        for macro_connection in macro_connections:
            src = macro_connection.src_macro_name
            dst = macro_connection.dst_macro_name
            macro_connection_dict[src][dst][macro_connection.hop - 1] += 1

        hop_counts = [0] * max_hop

        for src, dsts in macro_connection_dict.items():
            for dst, hops in dsts.items():
                for hop_index, count in enumerate(hops):
                    hop_counts[hop_index] += count

        # 打印每个hop的总条数
        for i, count in enumerate(hop_counts, start=1):
            print(f"Hop {i}: {count} connections")

        macro_pair_weights = {}

        for src, dsts in macro_connection_dict.items():
            for dst, hops in dsts.items():
                if src == dst:
                    continue
                macro_pair = tuple(sorted([src, dst]))

                if macro_pair not in macro_pair_weights:
                    macro_pair_weights[macro_pair] = 0

                for hop_index, count in enumerate(hops, start=1):
                    macro_pair_weights[macro_pair] += count / (hop_index ** 2)

        # diff_parent_connections_count = 0
        # print(f"macro_pair_weights size before adjusting same parent weights: {len(macro_pair_weights)}")
        # same_parent_connections_count = 0
        # for macro_pair, weight in macro_pair_weights.items():
        #     src, dst = macro_pair
        #     # 解析宏名称中的上一级名称
        #     src_parent = src.rsplit('/', 1)[0] if '/' in src else ''
        #     dst_parent = dst.rsplit('/', 1)[0] if '/' in dst else ''

        #     # 如果上一级名称不为空且相同，权重乘以2
        #     if src_parent and src_parent == dst_parent:
        #         print(f"Same parent: {src_parent} for Macros {macro_pair}")
        #         macro_pair_weights[macro_pair] *= 1.2
        #         same_parent_connections_count += 1
        #     elif src_parent and dst_parent and src_parent != dst_parent:
        #         diff_parent_connections_count += 1

        # self.node2pin_map = np.array(self.node2pin_map, dtype=np.int32)
        self.net2pin_map = list(self.net2pin_map)
        for macro_pair, weight in macro_pair_weights.items():
            net_id = len(self.flat_net2pin_start_map)
            src_name, dst_name = macro_pair

            src_inst_id = self.node_name2id_map[src_name]
            dst_inst_id = self.node_name2id_map[dst_name]

            src_pin_id = len(self.pin_offset_x)
            self.pin_offset_x = np.append(
                self.pin_offset_x, self.node_size_x[src_inst_id] / 2)
            self.pin_offset_y = np.append(
                self.pin_offset_y, self.node_size_y[src_inst_id] / 2)
            self.node2pin_map[src_inst_id] = np.array(
                np.append(self.node2pin_map[src_inst_id], src_pin_id), dtype=np.int32)
            self.pin2node_map = np.append(self.pin2node_map, src_inst_id)
            self.pin2net_map = np.append(self.pin2net_map, net_id)

            dst_pin_id = len(self.pin_offset_x)
            self.pin_offset_x = np.append(
                self.pin_offset_x, self.node_size_x[dst_inst_id] / 2)
            self.pin_offset_y = np.append(
                self.pin_offset_y, self.node_size_y[dst_inst_id] / 2)
            self.node2pin_map[dst_inst_id] = np.array(
                np.append(self.node2pin_map[dst_inst_id], dst_pin_id), dtype=np.int32)
            self.pin2node_map = np.append(self.pin2node_map, dst_inst_id)
            self.pin2net_map = np.append(self.pin2net_map, net_id)

            self.flat_net2pin_start_map = np.append(
                self.flat_net2pin_start_map, int(len(self.flat_net2pin_map)))
            self.flat_net2pin_map = np.append(
                self.flat_net2pin_map, src_pin_id)
            self.flat_net2pin_map = np.append(
                self.flat_net2pin_map, dst_pin_id)
            self.net2pin_map.append(
                np.array([src_pin_id, dst_pin_id], dtype=np.int32))
            self.net_weights = np.append(self.net_weights, weight)
            # self.pin2net_map[src_pin_id] = net_id
            # self.pin2net_map[dst_pin_id] = net_id

            print(
                f"Macros {macro_pair} have a total connection weight of {weight}")

        self.net2pin_map = np.array(self.net2pin_map, dtype=object)
        self.flat_net2pin_map = np.array(self.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(
            self.flat_net2pin_start_map, dtype=np.int32)
        self.net_weights = np.array(self.net_weights, dtype=self.dtype)
        self.pin2node_map = np.array(self.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(self.pin2net_map, dtype=np.int32)
        # for macro_pair, weight in macro_pair_weights.items():
        #     print(f"Macros {macro_pair} have a total connection weight of {weight}")

        # print(f"Number of connections with different parents: {diff_parent_connections_count}")
        # print(f"Number of connections with same parents: {same_parent_connections_count}")

    # def flatten_nested_map(self, net2pin_map):
    #    """
    #    @brief flatten an array of array to two arrays like CSV format
    #    @param net2pin_map array of array
    #    @return a pair of (elements, cumulative column indices of the beginning element of each row)
    #    """
    #    # flat netpin map, length of #pins
    #    flat_net2pin_map = np.zeros(len(pin2net_map), dtype=np.int32)
    #    # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
    #    flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
    #    count = 0
    #    for i in range(len(net2pin_map)):
    #        flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
    #        flat_net2pin_start_map[i] = count
    #        count += len(net2pin_map[i])
    #    assert flat_net2pin_map[-1] != 0
    #    flat_net2pin_start_map[len(net2pin_map)] = len(pin2net_map)

    #    return flat_net2pin_map, flat_net2pin_start_map

    def read(self, params):
        """
        @brief read using c++ 
        @param params parameters 
        """
        self.dtype = datatypes[params.dtype]
        # self.rawdb = place_io.PlaceIOFunction.read(params)
        self.initialize_from_rawdb(params)

    def set_net_weights(self):
        # with open("risa_weights.pkl", 'rb') as f:
        #     weights_dict = pickle.load(f)
        weights_dict = {
            1: 1.0000,
            2: 1.0000,
            3: 1.0000,
            4: 1.0828,
            5: 1.1536,
            6: 1.2206,
            7: 1.2823,
            8: 1.3385,
            9: 1.3991,
            10: 1.4493,
            11: 1.6899,
            12: 1.6899,
            13: 1.6899,
            14: 1.6899,
            15: 1.6899,
            16: 1.8924,
            17: 1.8924,
            18: 1.8924,
            19: 1.8924,
            20: 1.8924,
            21: 2.0743,
            22: 2.0743,
            23: 2.0743,
            24: 2.0743,
            25: 2.0743,
            26: 2.2334,
            27: 2.2334,
            28: 2.2334,
            29: 2.2334,
            30: 2.2334,
            31: 2.3892,
            32: 2.3892,
            33: 2.3892,
            34: 2.3892,
            35: 2.3892,
            36: 2.5356,
            37: 2.5356,
            38: 2.5356,
            39: 2.5356,
            40: 2.5356,
            41: 2.6625,
            42: 2.6625,
            43: 2.6625,
            44: 2.6625,
            45: 2.6625,
        }
        num_pins_in_net = np.ediff1d(self.flat_net2pin_start_map)
        weights = np.full(num_pins_in_net.shape, 2.7933)
        for k in weights_dict:
            weights[num_pins_in_net == k] = weights_dict[k]
        self.net_weights *= weights

    def get_inhomogeneous_list_to_ndarray(self, inhomogeneous_list, dtype=np.int32):
        res = inhomogeneous_list.copy()
        for i in range(len(res)):
            res[i] = np.array(
                res[i], dtype=dtype)
        res = np.array(res)
        return res

    def initialize_from_rawdb(self, pydb, params):
        """
        @brief initialize data members from raw database 
        @param params parameters 
        """
        # pydb = place_io.PlaceIOFunction.pydb(self.rawdb)

        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.num_terminal_NIs = pydb.num_terminal_NIs
        self.node_name2id_map = pydb.node_name2id_map
        self.node_names = np.array(pydb.node_names, dtype=np.string_)
        # If the placer directly takes a global placement solution,
        # the cell positions may still be floating point numbers.
        # It is not good to use the place_io OP to round the positions.
        # Currently we only support BOOKSHELF format.

        self.node_x = np.array(pydb.node_x, dtype=self.dtype)
        self.node_y = np.array(pydb.node_y, dtype=self.dtype)
        self.node_orient = np.array(pydb.node_orient, dtype=np.string_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node2orig_node_map = np.array(
            pydb.node2orig_node_map, dtype=np.int32)
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.string_)
        # BUG all the pin offsets are -1
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.net_name2id_map = pydb.net_name2id_map
        self.net_names = np.array(pydb.net_names, dtype=np.string_)
        self.net2pin_map = pydb.net2pin_map
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(
            pydb.flat_net2pin_start_map, dtype=np.int32)
        self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
        self.node2pin_map = pydb.node2pin_map
        self.flat_node2pin_map = np.array(
            pydb.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(
            pydb.flat_node2pin_start_map, dtype=np.int32)
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        self.rows = np.array(pydb.rows, dtype=self.dtype)
        self.regions = pydb.regions
        for i in range(len(self.regions)):
            self.regions[i] = np.array(self.regions[i], dtype=self.dtype)
        self.flat_region_boxes = np.array(
            pydb.flat_region_boxes, dtype=self.dtype)
        self.flat_region_boxes_start = np.array(
            pydb.flat_region_boxes_start, dtype=np.int32)
        self.node2fence_region_map = np.array(
            pydb.node2fence_region_map, dtype=np.int32)
        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)
        self.origin_row_height = int(pydb.row_height)
        self.origin_site_width = int(pydb.site_width)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)
        self.num_movable_pins = pydb.num_movable_pins
        self.total_space_area = float(pydb.total_space_area)

        self.routing_grid_xl = float(pydb.routing_grid_xl)
        self.routing_grid_yl = float(pydb.routing_grid_yl)
        self.routing_grid_xh = float(pydb.routing_grid_xh)
        self.routing_grid_yh = float(pydb.routing_grid_yh)
        if pydb.num_routing_grids_x:
            self.num_routing_grids_x = pydb.num_routing_grids_x
            self.num_routing_grids_y = pydb.num_routing_grids_y
            self.num_routing_layers = len(pydb.unit_horizontal_capacities)
            self.unit_horizontal_capacity = np.array(
                pydb.unit_horizontal_capacities, dtype=self.dtype).sum()
            self.unit_vertical_capacity = np.array(
                pydb.unit_vertical_capacities, dtype=self.dtype).sum()
            self.unit_horizontal_capacities = np.array(
                pydb.unit_horizontal_capacities, dtype=self.dtype)
            self.unit_vertical_capacities = np.array(
                pydb.unit_vertical_capacities, dtype=self.dtype)
            self.initial_horizontal_demand_map = np.array(pydb.initial_horizontal_demand_map, dtype=self.dtype).reshape(
                (-1, self.num_routing_grids_x, self.num_routing_grids_y)).sum(axis=0)
            self.initial_vertical_demand_map = np.array(pydb.initial_vertical_demand_map, dtype=self.dtype).reshape(
                (-1, self.num_routing_grids_x, self.num_routing_grids_y)).sum(axis=0)
        else:
            self.num_routing_grids_x = params.route_num_bins_x
            self.num_routing_grids_y = params.route_num_bins_y
            self.num_routing_layers = 1
            self.unit_horizontal_capacity = params.unit_horizontal_capacity
            self.unit_vertical_capacity = params.unit_vertical_capacity

        # convert node2pin_map to array of array
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(
                self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map)

        # convert net2pin_map to array of array
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map)

        if params.with_sta:
            self.start_points = np.array(pydb.start_points, dtype=np.int32)
            self.end_points = np.array(pydb.end_points, dtype=np.int32)

            self.flat_cells_by_level = np.array(
                pydb.flat_cells_by_level, dtype=np.int32)
            self.flat_cells_by_reverse_level = np.array(
                pydb.flat_cells_by_reverse_level, dtype=np.int32)
            self.flat_cells_by_level_start = np.array(
                pydb.flat_cells_by_level_start, dtype=np.int32)
            self.flat_cells_by_reverse_level_start = np.array(
                pydb.flat_cells_by_reverse_level_start, dtype=np.int32)
            # self.cells_by_level = self.get_inhomogeneous_list_to_ndarray(
            #     pydb.cells_by_level, dtype=np.int32)
            # self.cells_by_reverse_level = self.get_inhomogeneous_list_to_ndarray(
            #     pydb.cells_by_reverse_level, dtype=np.int32)
            self.net2driver_pin_map = np.array(
                pydb.net2driver_pin_map, dtype=np.int32)
            
            self.inrdelays = np.array(pydb.inrdelays, dtype=self.dtype)
            self.infdelays = np.array(pydb.infdelays, dtype=self.dtype)
            self.inrtrans = np.array(pydb.inrtrans, dtype=self.dtype)
            self.inftrans = np.array(pydb.inftrans, dtype=self.dtype)
            self.outcaps = np.array(pydb.outcaps, dtype=self.dtype)

            self.net_flat_arcs_start = np.array(
                pydb.net_flat_arcs_start, dtype=np.int32)
            self.net_flat_arcs = np.array(pydb.net_flat_arcs, dtype=np.int32)
            self.cell_flat_arcs_start = np.array(
                pydb.cell_flat_arcs_start, dtype=np.int32)
            self.cell_flat_arcs = np.array(pydb.cell_flat_arcs, dtype=np.int32)

            self.main_id_2_cell_id_start = np.array(
                pydb.main_id_2_cell_id_start, dtype=np.int32)
            self.cell_id_2_arc_id_start = np.array(
                pydb.cell_id_2_arc_id_start, dtype=np.int32)

            self.inst_main_id = np.array(pydb.inst_main_id, dtype=np.int32)
            self.inst_size = np.array(pydb.inst_size, dtype=self.dtype)

            # LUTs table
            self.f_delay_flat_luts_values = np.array(
                pydb.f_delay_flat_luts_values, dtype=self.dtype)
            self.f_delay_flat_luts_trans_table = np.array(
                pydb.f_delay_flat_luts_trans_table, dtype=self.dtype)
            self.f_delay_flat_luts_cap_table = np.array(
                pydb.f_delay_flat_luts_cap_table, dtype=self.dtype)
            self.f_delay_flat_luts_dim = np.array(
                pydb.f_delay_flat_luts_dim, dtype=np.int32)

            self.r_delay_flat_luts_values = np.array(
                pydb.r_delay_flat_luts_values, dtype=self.dtype)
            self.r_delay_flat_luts_trans_table = np.array(
                pydb.r_delay_flat_luts_trans_table, dtype=self.dtype)
            self.r_delay_flat_luts_cap_table = np.array(
                pydb.r_delay_flat_luts_cap_table, dtype=self.dtype)
            self.r_delay_flat_luts_dim = np.array(
                pydb.r_delay_flat_luts_dim, dtype=np.int32)

            self.f_trans_flat_luts_values = np.array(
                pydb.f_trans_flat_luts_values, dtype=self.dtype)
            self.f_trans_flat_luts_trans_table = np.array(
                pydb.f_trans_flat_luts_trans_table, dtype=self.dtype)
            self.f_trans_flat_luts_cap_table = np.array(
                pydb.f_trans_flat_luts_cap_table, dtype=self.dtype)
            self.f_trans_flat_luts_dim = np.array(
                pydb.f_trans_flat_luts_dim, dtype=np.int32)

            self.r_trans_flat_luts_values = np.array(
                pydb.r_trans_flat_luts_values, dtype=self.dtype)
            self.r_trans_flat_luts_trans_table = np.array(
                pydb.r_trans_flat_luts_trans_table, dtype=self.dtype)
            self.r_trans_flat_luts_cap_table = np.array(
                pydb.r_trans_flat_luts_cap_table, dtype=self.dtype)
            self.r_trans_flat_luts_dim = np.array(
                pydb.r_trans_flat_luts_dim, dtype=np.int32)

            self.cell_id_2_libpin_id_start = np.array(
                pydb.cell_id_2_libpin_id_start, dtype=np.int32)
            self.pin_2_libpin_offset = np.array(
                pydb.pin_2_libpin_offset, dtype=np.int32)
            self.flat_lib_pin_cap = np.array(
                pydb.flat_lib_pin_cap, dtype=self.dtype)
            self.flat_lib_pin_cap_limit = np.array(
                pydb.flat_lib_pin_cap_limit, dtype=self.dtype)
            self.flat_lib_pin_slew_limit = np.array(
                pydb.flat_lib_pin_slew_limit, dtype=self.dtype)

            # RC
            self.r_unit = float(pydb.r_unit)
            self.c_unit = float(pydb.c_unit)

    def __call__(self, params):
        """
        @brief top API to read placement files 
        @param params parameters 
        """
        tt = time.time()

        self.read(params)
        self.initialize(params)

        logging.info("reading benchmark takes %g seconds" % (time.time() - tt))

    def calc_num_filler_for_fence_region(
        self, region_id, node2fence_region_map, target_density
    ):
        """
        @description: calculate number of fillers for each fence region
        @param fence_regions{type}
        @return:
        """
        num_regions = len(self.regions)
        node2fence_region_map = node2fence_region_map[self.movable_slice]
        if region_id < len(self.regions):
            fence_region_mask = node2fence_region_map == region_id
        else:
            fence_region_mask = node2fence_region_map >= len(self.regions)
        if np.sum(fence_region_mask) == 0:
            return 0, 0, 1, 1, 0, 0
        num_movable_nodes = self.num_movable_nodes

        movable_node_size_x = self.node_size_x[:
                                               num_movable_nodes][fence_region_mask]
        # movable_node_size_y = self.node_size_y[:num_movable_nodes][fence_region_mask]

        lower_bound = np.percentile(movable_node_size_x, 5)
        upper_bound = np.percentile(movable_node_size_x, 95)
        filler_size_x = np.mean(
            movable_node_size_x[
                (movable_node_size_x >= lower_bound)
                & (movable_node_size_x <= upper_bound)
            ]
        )
        filler_size_y = self.row_height

        area = (self.xh - self.xl) * (self.yh - self.yl)

        total_movable_node_area = np.sum(
            self.node_size_x[:num_movable_nodes][fence_region_mask]
            * self.node_size_y[:num_movable_nodes][fence_region_mask]
        )

        if region_id < num_regions:
            # placeable area is not just fention region area. Macros can have overlap with fence region. But we approximate by this method temporarily
            region = self.regions[region_id]
            placeable_area = np.sum(
                (region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1])
            )
        else:
            # invalid area outside the region, excluding macros? ignore overlap between fence region and macro
            fence_regions = np.concatenate(self.regions, 0).astype(np.float32)
            fence_regions_size_x = fence_regions[:, 2] - fence_regions[:, 0]
            fence_regions_size_y = fence_regions[:, 3] - fence_regions[:, 1]
            fence_region_area = np.sum(
                fence_regions_size_x * fence_regions_size_y)

            placeable_area = (
                max(self.total_space_area, self.area - self.total_fixed_node_area)
                - fence_region_area
            )

        # recompute target density based on the region utilization
        utilization = min(total_movable_node_area / placeable_area, 1.0)
        if target_density < utilization:
            # add a few fillers to avoid divergence
            target_density_fence_region = min(1, utilization + 0.01)
        else:
            target_density_fence_region = target_density

        target_density_fence_region = max(0.35, target_density_fence_region)

        total_filler_node_area = max(
            placeable_area * target_density_fence_region - total_movable_node_area, 0.0
        )

        num_filler = int(
            round(total_filler_node_area / (filler_size_x * filler_size_y))
        )
        logging.info(
            "Region:%2d movable_node_area =%10.1f, placeable_area =%10.1f, utilization =%.3f, filler_node_area =%10.1f, #fillers =%8d, filler sizes =%2.4gx%g\n"
            % (
                region_id,
                total_movable_node_area,
                placeable_area,
                utilization,
                total_filler_node_area,
                num_filler,
                filler_size_x,
                filler_size_y,
            )
        )

        return (
            num_filler,
            target_density_fence_region,
            filler_size_x,
            filler_size_y,
            total_movable_node_area,
            np.sum(fence_region_mask.astype(np.float32)),
        )

    def initialize(self, params):
        """
        @brief initialize data members after reading 
        @param params parameters 
        """
        # setup utility slices
        self.movable_slice = slice(0, self.num_movable_nodes)
        self.fixed_slice = slice(
            self.num_movable_nodes, self.num_movable_nodes + self.num_terminals
        )
        self.io_slice = slice(
            self.num_movable_nodes + self.num_terminals,
            self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs,
        )

        # set macros
        self.update_macros(params)

        # set net weights for improved HPWL % RSMT correlation
        if params.risa_weights == 1:
            self.set_net_weights()

        # pin density inflation
        self.pin_density_inflation(params.pin_density)

        # routing information for congestion estimation
        if params.route_info_input == "default":
            aux_name = params.aux_input.rsplit(".", 1)[0]
            params.route_info_input = f"{aux_name}.route_info"
        self.set_routing_info(params.route_info_input)

        # shift and scale
        # adjust shift_factor and scale_factor if not set
        params.shift_factor[0] = self.xl
        params.shift_factor[1] = self.yl
        logging.info(
            "set shift_factor = (%g, %g), as original row bbox = (%g, %g, %g, %g)"
            % (
                params.shift_factor[0],
                params.shift_factor[1],
                self.xl,
                self.yl,
                self.xh,
                self.yh,
            )
        )

        if params.scale_factor == 0.0 or self.site_width != 1.0:
            params.scale_factor = 1.0 / self.site_width
        if self.row_height % self.site_width != 0:
            logging.warn(
                "row_height is not divisible by site_width, might create issues during legalization"
            )
        if not self.site_width.is_integer() or not self.row_height.is_integer():
            logging.warn(
                "site_width or row_height is not an integer, might create issues during legalization"
            )
        logging.info(
            "set scale_factor = %g, as site_width = %g"
            % (params.scale_factor, self.site_width)
        )
        self.scale(params.shift_factor, params.scale_factor)

        params.macro_halo_x *= params.scale_factor
        params.macro_halo_y *= params.scale_factor
        params.macro_pin_halo_x *= params.scale_factor
        params.macro_pin_halo_y *= params.scale_factor
        self.macro_padding_x *= params.scale_factor
        self.macro_padding_y *= params.scale_factor
        self.bndry_padding_x *= params.scale_factor
        self.bndry_padding_y *= params.scale_factor

        content = """
================================= Benchmark Statistics =================================
#nodes = %d, #terminals = %d, # terminal_NIs = %d, #movable = %d, #nets = %d
die area = (%g, %g, %g, %g) %g
row height = %g, site width = %g
""" % (
            self.num_physical_nodes, self.num_terminals, self.num_terminal_NIs, self.num_movable_nodes, len(
                self.net_names),
            self.xl, self.yl, self.xh, self.yh, self.area,
            self.row_height, self.site_width
        )

        # set number of bins
        # derive bin dimensions by keeping the aspect ratio
        aspect_ratio = (self.yh - self.yl) / (self.xh - self.xl)
        if params.auto_adjust_bins:
            num_bins = math.sqrt(math.pow(2, round(math.log2(self.num_physical_nodes)))) 
            num_bins_x = int(num_bins)
            num_bins_y = int(num_bins)
        else:    
            num_bins_x = params.num_bins_x
            num_bins_y = params.num_bins_y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        # set bin size
        self.bin_size_x = (self.xh - self.xl) / self.num_bins_x
        self.bin_size_y = (self.yh - self.yl) / self.num_bins_y

        # bin center array
        self.bin_center_x = self.bin_centers(self.xl, self.xh, self.bin_size_x)
        self.bin_center_y = self.bin_centers(self.yl, self.yh, self.bin_size_y)

        content += "num_bins = %dx%d, bin sizes = %gx%g\n" % (
            self.num_bins_x, self.num_bins_y, self.bin_size_x, self.bin_size_y)

        # set num_movable_pins
        if self.num_movable_pins is None:
            self.num_movable_pins = 0
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        content += "#pins = %d, #movable_pins = %d\n" % (
            self.num_pins, self.num_movable_pins)
        # set total cell area
        self.total_movable_node_area = float(np.sum(
            self.node_size_x[:self.num_movable_nodes] * self.node_size_y[:self.num_movable_nodes]))
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(np.sum(
            np.maximum(
                np.minimum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] +
                           self.node_size_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xh)
                - np.maximum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xl),
                0.0) * np.maximum(
                np.minimum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] +
                           self.node_size_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yh)
                - np.maximum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yl),
                0.0)
        ))
        self.total_space_area = self.area - self.total_fixed_node_area
        content += "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n" % (
            self.total_movable_node_area, self.total_fixed_node_area, self.total_space_area)

        # set movable macro area
        self.total_movable_macro_area = np.sum(
            (self.node_size_x * self.node_size_y)[self.movable_slice][
                self.movable_macro_mask
            ]
        )
        total_movable_cell_area = self.total_movable_node_area - \
            self.total_movable_macro_area
        total_cell_space_area = self.total_space_area - self.total_movable_macro_area
        cell_utilization = total_movable_cell_area / total_cell_space_area
        self.total_movable_cell_area = total_movable_cell_area
        if self.total_movable_macro_area <= 0:
            params.macro_place_flag = False

        content += "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\ntotal_movable_cell_area = %g, total_movable_macro_area = %g\n" % \
            (self.total_movable_node_area, self.total_fixed_node_area,
             self.total_space_area, total_movable_cell_area, self.total_movable_macro_area)

        # # check movable macros, adjust area to treat movable macros as fixed macros
        if self.num_movable_macros > 0:
            logging.info(
                "detect movable macros %d, area %g, reduce those area from movable_area",
                self.num_movable_macros,
                self.total_movable_macro_area,
            )
            # self.total_movable_node_area -= self.total_movable_macro_area
            # self.total_fixed_node_area += self.total_movable_macro_area
            # self.total_space_area -= self.total_movable_macro_area
            # content += (
            #     "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n"
            #     % (
            #         self.total_movable_node_area,
            #         self.total_fixed_node_area,
            #         self.total_space_area,
            #     )
            # )

        target_density = min(self.total_movable_node_area /
                             self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warn(
                "target_density %g is smaller than utilization %g, ignored"
                % (params.target_density, target_density)
            )
            params.target_density = target_density
        content += "utilization = %g, target_density = %g\n" % (
            self.total_movable_node_area / self.total_space_area,
            params.target_density,
        )

        # calculate fence region virtual macro
        if len(self.regions) > 0:
            virtual_macro_for_fence_region = [
                fence_region.slice_non_fence_region(
                    region,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    merge=True,
                    plot=False,
                    figname=f"vmacro_{region_id}_merged.png",
                    device="cpu",
                    macro_pos_x=self.node_x[self.fixed_slice],
                    macro_pos_y=self.node_y[self.fixed_slice],
                    macro_size_x=self.node_size_x[self.fixed_slice],
                    macro_size_y=self.node_size_y[self.fixed_slice],
                )
                .cpu()
                .numpy()
                for region_id, region in enumerate(self.regions)
            ]
            virtual_macro_for_non_fence_region = np.concatenate(
                self.regions, 0)
            self.virtual_macro_fence_region = virtual_macro_for_fence_region + [
                virtual_macro_for_non_fence_region
            ]

        # insert filler nodes
        if len(self.regions) > 0:
            # calculate fillers if there is fence region
            self.filler_size_x_fence_region = []
            self.filler_size_y_fence_region = []
            self.num_filler_nodes = 0
            self.num_filler_nodes_fence_region = []
            self.num_movable_nodes_fence_region = []
            self.total_movable_node_area_fence_region = []
            self.target_density_fence_region = []
            self.filler_start_map = None
            filler_node_size_x_list = []
            filler_node_size_y_list = []
            self.total_filler_node_area = 0
            for i in range(len(self.regions) + 1):
                (
                    num_filler_i,
                    target_density_i,
                    filler_size_x_i,
                    filler_size_y_i,
                    total_movable_node_area_i,
                    num_movable_nodes_i,
                ) = self.calc_num_filler_for_fence_region(
                    i, self.node2fence_region_map, params.target_density
                )
                self.num_movable_nodes_fence_region.append(num_movable_nodes_i)
                self.num_filler_nodes_fence_region.append(num_filler_i)
                self.total_movable_node_area_fence_region.append(
                    total_movable_node_area_i
                )
                self.target_density_fence_region.append(target_density_i)
                self.filler_size_x_fence_region.append(filler_size_x_i)
                self.filler_size_y_fence_region.append(filler_size_y_i)
                self.num_filler_nodes += num_filler_i
                filler_node_size_x_list.append(
                    np.full(
                        num_filler_i,
                        fill_value=filler_size_x_i,
                        dtype=self.node_size_x.dtype,
                    )
                )
                filler_node_size_y_list.append(
                    np.full(
                        num_filler_i,
                        fill_value=filler_size_y_i,
                        dtype=self.node_size_y.dtype,
                    )
                )
                filler_node_area_i = num_filler_i * \
                    (filler_size_x_i * filler_size_y_i)
                self.total_filler_node_area += filler_node_area_i
                content += (
                    "Region: %2d filler_node_area = %10.2f, #fillers = %8d, filler sizes = %2.4gx%g\n"
                    % (
                        i,
                        filler_node_area_i,
                        num_filler_i,
                        filler_size_x_i,
                        filler_size_y_i,
                    )
                )

            self.total_movable_node_area_fence_region = np.array(
                self.total_movable_node_area_fence_region
            )
            self.num_movable_nodes_fence_region = np.array(
                self.num_movable_nodes_fence_region
            )

        if params.enable_fillers:
            # the way to compute this is still tricky; we need to consider place_io together on how to
            # summarize the area of fixed cells, which may overlap with each other.
            if len(self.regions) > 0:
                self.filler_start_map = np.cumsum(
                    [0] + self.num_filler_nodes_fence_region
                )
                self.num_filler_nodes_fence_region = np.array(
                    self.num_filler_nodes_fence_region
                )
                self.node_size_x = np.concatenate(
                    [self.node_size_x] + filler_node_size_x_list
                )
                self.node_size_y = np.concatenate(
                    [self.node_size_y] + filler_node_size_y_list
                )
                content += (
                    "total_filler_node_area = %10.2f, #fillers = %8d, average filler sizes = %2.4gx%g\n"
                    % (
                        self.total_filler_node_area,
                        self.num_filler_nodes,
                        self.total_filler_node_area
                        / self.num_filler_nodes
                        / self.row_height,
                        self.row_height,
                    )
                )
            else:
                node_size_order = np.argsort(
                    self.node_size_x[self.movable_slice])
                filler_size_x = np.mean(
                    self.node_size_x[
                        node_size_order[
                            int(self.num_movable_nodes * 0.05): int(
                                self.num_movable_nodes * 0.95
                            )
                        ]
                    ]
                )
                filler_size_y = self.row_height
                placeable_area = max(
                    self.area - self.total_fixed_node_area, self.total_space_area
                )
                content += "use placeable_area = %g to compute fillers\n" % (
                    placeable_area
                )
                self.total_filler_node_area = max(
                    placeable_area * params.target_density
                    - self.total_movable_node_area,
                    0.0,
                )
                self.num_filler_nodes = int(
                    round(self.total_filler_node_area /
                          (filler_size_x * filler_size_y))
                )
                self.node_size_x = np.concatenate(
                    [
                        self.node_size_x,
                        np.full(
                            self.num_filler_nodes,
                            fill_value=filler_size_x,
                            dtype=self.node_size_x.dtype,
                        ),
                    ]
                )
                self.node_size_y = np.concatenate(
                    [
                        self.node_size_y,
                        np.full(
                            self.num_filler_nodes,
                            fill_value=filler_size_y,
                            dtype=self.node_size_y.dtype,
                        ),
                    ]
                )
                content += (
                    "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n"
                    % (
                        self.total_filler_node_area,
                        self.num_filler_nodes,
                        filler_size_x,
                        filler_size_y,
                    )
                )
        else:
            self.total_filler_node_area = 0
            self.num_filler_nodes = 0
            filler_size_x, filler_size_y = 0, 0
            if len(self.regions) > 0:
                self.filler_start_map = np.zeros(
                    len(self.regions) + 2, dtype=np.int32)
                self.num_filler_nodes_fence_region = np.zeros(
                    len(self.num_filler_nodes_fence_region)
                )

            content += (
                "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n"
                % (
                    self.total_filler_node_area,
                    self.num_filler_nodes,
                    filler_size_x,
                    filler_size_y,
                )
            )

        if params.routability_opt_flag:
            content += "================================== routing information =================================\n"
            content += "routing grids (%d, %d)\n" % (
                self.num_routing_grids_x,
                self.num_routing_grids_y,
            )
            content += "routing grid sizes (%g, %g)\n" % (
                self.routing_grid_size_x,
                self.routing_grid_size_y,
            )
            content += "routing capacity H/V (%g, %g) per tile\n" % (
                self.unit_horizontal_capacity * self.routing_grid_size_y,
                self.unit_vertical_capacity * self.routing_grid_size_x,
            )
        content += "========================================================================================"

        logging.info(content)

        # setup utility slices
        self.filler_slice = slice(
            self.num_nodes - self.num_filler_nodes, self.num_nodes
        )
        self.all_slice = slice(0, self.num_nodes)

    def pin_density_inflation(self, pin_density, pin_accessibility=2.0):
        # pin_accessibility: virtually increases # pins due to cell blockages/pin shapes
        if 0 < pin_density < 1:
            # assume 6.5-track high-density library ~ 5 M1/M2 tracks for signal routing
            num_tracks_height = 5
            num_pins_in_cell = np.ediff1d(self.flat_node2pin_start_map)
            inflated_widths = self.crop_to_site(
                np.minimum(
                    2.5 * self.node_size_x,
                    num_pins_in_cell
                    * pin_accessibility
                    * self.row_height
                    / (num_tracks_height**2 * pin_density),
                ),
                "x",
            )
            # inflate standard cells only
            self.node_size_x[self.movable_slice][~self.movable_macro_mask] = np.maximum(
                self.node_size_x[self.movable_slice][~self.movable_macro_mask],
                inflated_widths[self.movable_slice][~self.movable_macro_mask],
            )

    def set_routing_info(self, route_file):
        self.routing_V = (
            10 * self.area / (100 * self.site_width)
        )  # default: 10 layers and pitch is 100x site width
        self.routing_H = 10 * self.area / (100 * self.site_width)
        self.macro_util_V = np.zeros(
            self.num_movable_macros + self.num_fixed_macros, dtype=self.dtype
        )
        self.macro_util_H = np.zeros(
            self.num_movable_macros + self.num_fixed_macros, dtype=self.dtype
        )
        self.macros_routing = {}

        if os.path.isfile(route_file):
            with open(route_file, "r") as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) == 2:
                        self.routing_V = float(line[0])
                        self.routing_H = float(line[1])
                    elif len(line) == 3:
                        self.macros_routing[line[0]] = [
                            float(line[1]), float(line[2])]

        if self.macros_routing:
            movable_macros_indexes = np.where(self.movable_macro_mask)[0]
            fixed_macro_indexes = (
                self.num_movable_nodes + np.where(self.fixed_macro_mask)[0]
            )
            macros_indexes = np.concatenate(
                (movable_macros_indexes, fixed_macro_indexes)
            )
            for name, util in self.macros_routing.items():
                idx = np.where(macros_indexes ==
                               self.node_name2id_map[name])[0]
                self.macro_util_V[idx], self.macro_util_H[idx] = util

        self.routing_grid_xl = self.xl
        self.routing_grid_yl = self.yl
        self.routing_grid_xh = self.xh
        self.routing_grid_yh = self.yh

    def crop_to_site(self, v, axis: str = "x", mode: str = "up"):
        ops = {"close": np.round, "up": np.ceil, "down": np.floor}
        op = ops[mode]
        if axis == "x":
            return self.site_width * op(v / self.site_width)
        elif axis == "y":
            return self.row_height * op(v / self.row_height)

    def update_macros(self, params, area_threshold=10, height_threshold=2):
        # set large cells as macros
        node_areas = self.node_size_x * self.node_size_y
        mean_area = node_areas[self.movable_slice].mean() * area_threshold
        row_height = self.node_size_y[self.movable_slice].min(
        ) * height_threshold

        # movable macros
        self.movable_macro_mask = (node_areas[self.movable_slice] > mean_area) & (
            self.node_size_y[self.movable_slice] > row_height
        )
        self.movable_macro_pins = np.isin(self.pin2node_map, np.arange(
            0, self.num_movable_nodes)[self.movable_macro_mask])
        self.movable_macro_idx = np.where(self.movable_macro_mask)[0]
        self.num_movable_macros = self.movable_macro_idx.shape[0]
        # fixed macros
        self.fixed_macro_mask = (node_areas[self.fixed_slice] > mean_area) & (
            self.node_size_y[self.fixed_slice] > row_height
        )
        self.fixed_macro_idx = (
            self.num_movable_nodes + np.where(self.fixed_macro_mask)[0]
        )
        self.num_fixed_macros = self.fixed_macro_idx.shape[0]

        macro_pin_offset_x_mean = []
        macro_pin_offset_y_mean = []

        for macro_id in self.movable_macro_idx:
            pins = self.node2pin_map[macro_id]
            macro_pin_offset_x_mean.append(np.mean(self.pin_offset_x[pins]))
            macro_pin_offset_y_mean.append(np.mean(self.pin_offset_y[pins]))

        self.is_pin_lower_x = (np.array(macro_pin_offset_x_mean) <= 0.1 *
                               self.node_size_x[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_upper_x = (np.array(macro_pin_offset_x_mean) >= 0.9 *
                               self.node_size_x[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_lower_y = (np.array(macro_pin_offset_y_mean) <= 0.1 *
                               self.node_size_y[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_upper_y = (np.array(macro_pin_offset_y_mean) >= 0.9 *
                               self.node_size_y[self.movable_slice][self.movable_macro_mask]).astype('float64')

        # setup macro padding for overlap loss
        self.macro_padding_x = params.macro_padding_x
        self.macro_padding_y = params.macro_padding_y
        self.bndry_padding_x = params.bndry_padding_x
        self.bndry_padding_y = params.bndry_padding_y

        # make sure the macros & halo sizes are multiples of site
        params.macro_halo_x = self.crop_to_site(params.macro_halo_x, "x")
        params.macro_halo_y = self.crop_to_site(params.macro_halo_y, "y")
        params.macro_pin_halo_x = self.crop_to_site(
            params.macro_pin_halo_x, "x")
        params.macro_pin_halo_y = self.crop_to_site(
            params.macro_pin_halo_y, "y")
        self.node_size_x[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_x[self.movable_macro_idx], "x"
        )
        self.node_size_y[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_y[self.movable_macro_idx], "y"
        )

        # add halo around macros
        if params.macro_halo_x >= 0 and params.macro_halo_y >= 0:
            # increase macro sizes
            self.node_size_x[self.movable_macro_idx] += 2 * params.macro_halo_x
            self.node_size_y[self.movable_macro_idx] += 2 * params.macro_halo_y
            # self.node_size_x[self.fixed_macro_idx] += 2 * params.macro_halo_x
            # self.node_size_y[self.fixed_macro_idx] += 2 * params.macro_halo_y

            # shift macro positions
            self.node_x[self.movable_macro_idx] -= params.macro_halo_x
            self.node_y[self.movable_macro_idx] -= params.macro_halo_y
            # self.node_x[self.fixed_macro_idx] -= params.macro_halo_x
            # self.node_y[self.fixed_macro_idx] -= params.macro_halo_y

            # shift macro pins
            self.movable_macro_pins = np.isin(
                self.pin2node_map, self.movable_macro_idx)
            self.pin_offset_x[self.movable_macro_pins] += params.macro_halo_x
            self.pin_offset_y[self.movable_macro_pins] += params.macro_halo_y
            # self.fixed_macro_pins = np.isin(self.pin2node_map, self.fixed_macro_idx)
            # self.pin_offset_x[self.fixed_macro_pins] += params.macro_halo_x
            # self.pin_offset_y[self.fixed_macro_pins] += params.macro_halo_y
        if params.macro_pin_halo_x >= 0:
            self.node_size_x[self.movable_macro_idx] += self.is_pin_lower_x * \
                params.macro_pin_halo_x + self.is_pin_upper_x * params.macro_pin_halo_x
            self.node_size_y[self.movable_macro_idx] += self.is_pin_lower_y * \
                params.macro_pin_halo_y + self.is_pin_upper_y * params.macro_pin_halo_y

            self.node_x[self.movable_macro_idx] -= self.is_pin_lower_x * \
                params.macro_pin_halo_x
            self.node_y[self.movable_macro_idx] -= self.is_pin_lower_y * \
                params.macro_pin_halo_y

            self.pin_offset_x[self.movable_macro_pins] += self.is_pin_lower_x[self.pin2node_map[self.movable_macro_pins]
                                                                              ] * params.macro_pin_halo_x
            self.pin_offset_y[self.movable_macro_pins] += self.is_pin_lower_y[self.pin2node_map[self.movable_macro_pins]
                                                                              ] * params.macro_pin_halo_y

    def write(self, params, filename):
        """
        @brief write placement solution
        @param filename output file name 
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        """
        tt = time.time()
        logging.info("writing to %s" % (filename))
        # unscale locations
        unscale_factor = 1.0 / params.scale_factor
        if unscale_factor == 1.0:
            node_x = self.node_x
            node_y = self.node_y
        else:
            node_x = self.node_x * unscale_factor
            node_y = self.node_y * unscale_factor

        # Global placement may have floating point positions.
        # Currently only support BOOKSHELF format.
        # This is mainly for debug.
        self.write_nets(params, filename)

        logging.info("write %s takes %.3f seconds" % ('pl', time.time() - tt))

    def read_pl(self, params, pl_file):
        """
        @brief read .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("reading %s" % (pl_file))
        count = 0
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    continue
                # node positions
                pos = re.search(
                    r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*:\s*(\w+)", line)
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(6))
                    self.node_orient[node_id] = pos.group(10)
                    orient = pos.group(4)
        if params.scale_factor != 1.0:
            self.scale_pl(params.scale_factor)
        logging.info("read_pl takes %.3f seconds" % (time.time() - tt))

    def write_pl(self, params, pl_file, node_x, node_y):
        """
        @brief write .pl file
        @param pl_file .pl file 
        """
        tt = time.time()
        logging.info("writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_movable_nodes):
            content += "\n%s %g %g : %s" % (
                str_node_names[i],
                node_x[i],
                node_y[i],
                str_node_orient[i]
            )
        # use the original fixed cells, because they are expanded if they contain shapes
        fixed_node_indices = list(self.rawdb.fixedNodeIndices())
        for i, node_id in enumerate(fixed_node_indices):
            content += "\n%s %g %g : %s /FIXED" % (
                str(self.rawdb.nodeName(node_id)),
                float(self.rawdb.node(node_id).xl()),
                float(self.rawdb.node(node_id).yl()),
                "N"  # still hard-coded
            )
        for i in range(self.num_movable_nodes + self.num_terminals, self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs):
            content += "\n%s %g %g : %s /FIXED_NI" % (
                str_node_names[i],
                node_x[i],
                node_y[i],
                str_node_orient[i]
            )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write_pl takes %.3f seconds" % (time.time() - tt))

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters 
        @param net_file .net file 
        """
        tt = time.time()
        logging.info("writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins),
                                                self.net_names[net_id])
            for pin_id in pins:
                content += "\n\t%s %s : %d %d" % (self.node_names[self.pin2node_map[pin_id]], self.pin_direct[pin_id],
                                                  self.pin_offset_x[pin_id] / params.scale_factor, self.pin_offset_y[pin_id] / params.scale_factor)

        with open(net_file, "w") as f:
            f.write(content)
        logging.info("write_nets takes %.3f seconds" % (time.time() - tt))

    def write_placement_back(self, node_x, node_y):
        # unscale locations
        # TODO:
        ieda_io = IEDAIO(self.data_manager.dir_workspace)
        ieda_io.write_placement_back(self.get_dmInst_ptr, node_x, node_y)

    def unscale_pl(self, shift_factor, scale_factor):
        """
        @brief unscale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        unscale_factor = 1.0 / scale_factor
        node_x = self.node_x * unscale_factor + shift_factor[0]
        node_y = self.node_y * unscale_factor + shift_factor[1]
        return node_x, node_y

    # FIXME:
    def apply(self, params, node_x, node_y):
        """
        @brief apply placement solution and update database 
        """
        # assign solution
        self.node_x[:self.num_movable_nodes] = node_x[:self.num_movable_nodes]
        self.node_y[:self.num_movable_nodes] = node_y[:self.num_movable_nodes]

        # unscale locations
        unscale_factor = 1.0 / params.scale_factor
        node_x = self.node_x[:self.num_movable_nodes] * \
            unscale_factor + params.shift_factor[0]
        node_y = self.node_y[:self.num_movable_nodes] * \
            unscale_factor + params.shift_factor[1]
        # update raw database
        self.write_placement_back(node_x, node_y)
        # update raw database
        # place_io.PlaceIOFunction.apply(self.rawdb, node_x, node_y)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    logging.info("parameters = %s" % (params))

    db = PlaceDB()
    db(params)

    db.print_node(1)
    db.print_net(1)
    db.print_row(1)
