# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# @file   PlaceObj.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Placement model class defining the placement objective.
#

import os
import sys
import time
from matplotlib.pyplot import step
import numpy as np
import itertools
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gzip

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import dreamplace.ops.density_overflow.density_overflow as density_overflow
import dreamplace.ops.electric_potential.electric_overflow as electric_overflow
import dreamplace.ops.electric_potential.electric_potential as electric_potential
import dreamplace.ops.density_potential.density_potential as density_potential
import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.rudy.rudy_macros as rudy_macros
import dreamplace.ops.pin_utilization.pin_utilization as pin_utilization
import dreamplace.ops.nctugr_binary.nctugr_binary as nctugr_binary
import dreamplace.ops.irt_egr.irt_egr as eGR
import dreamplace.ops.gpugr.gpugr as gpugr_congestion
import dreamplace.ops.adjust_node_area.adjust_node_area as adjust_node_area
import dreamplace.ops.macro_overlap.macro_overlap as macro_overlap
import dreamplace.ops.macro_refinement.macro_refinement as macro_refinement
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
from dreamplace.ops.rc_timing.rc_timing import RCTiming
from dreamplace.BasicPlace import PlaceDataCollection
from tools.iEDA.module.sta import IEDASta
from dreamplace.ops.routability.plot_map import plot_node_grad_directions


class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """

    def __init__(self, placedb, data_collections, op_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []
        if len(placedb.regions) > 0:
            self.movablenode2fence_region_map_clamp = (
                data_collections.node2fence_region_map[: placedb.num_movable_nodes]
                .clamp(max=len(placedb.regions))
                .long()
            )
            self.filler2fence_region_map = torch.zeros(
                placedb.num_filler_nodes,
                device=data_collections.pos[0].device,
                dtype=torch.long,
            )
            for i in range(len(placedb.regions) + 1):
                filler_beg, filler_end = self.placedb.filler_start_map[i : i + 2]
                self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            # The preconditioning step in python is time-consuming, as in each gradient
            # pass, the total net weight should be re-calculated.
            # sum_pin_weights_in_nodes = self.op_collections.pws_op(
            #     self.data_collections.net_weights
            # )
            if density_weight.size(0) == 1:
                precond = (
                    # sum_pin_weights_in_nodes
                    0 + self.alpha * density_weight * self.data_collections.node_areas
                )
            else:
                # only precondition the non fence region
                node_areas = self.data_collections.node_areas.clone()

                mask = self.data_collections.node2fence_region_map[
                    : self.placedb.num_movable_nodes
                ] >= len(self.placedb.regions)
                node_areas[: self.placedb.num_movable_nodes].masked_scatter_(
                    mask,
                    node_areas[: self.placedb.num_movable_nodes][mask]
                    * density_weight[-1],
                )
                filler_beg, filler_end = self.placedb.filler_start_map[-2:]
                node_areas[
                    self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_beg : self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_end
                ] *= density_weight[-1]
                precond = sum_pin_weights_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[self.placedb.num_movable_nodes : self.placedb.num_physical_nodes] = 0
            grad[
                self.placedb.num_nodes
                + self.placedb.num_movable_nodes : self.placedb.num_nodes
                + self.placedb.num_physical_nodes
            ] = 0
            grad[0 : self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)
            # grad = grad.view(2, -1)
            # grad[0, self.placedb.num_movable_nodes:self.placedb.num_nodes] = 0
            # grad[1, self.placedb.num_movable_nodes:self.placedb.num_nodes] = 0
            # grad = grad.view(-1)
            # stop gradients for terminated electric field
            if update_mask is not None:
                grad = grad.view(2, -1)
                update_mask = ~update_mask
                movable_mask = update_mask[self.movablenode2fence_region_map_clamp]
                filler_mask = update_mask[self.filler2fence_region_map]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[
                    0, self.placedb.num_nodes - self.placedb.num_filler_nodes :
                ].masked_fill_(filler_mask, 0)
                grad[
                    1, self.placedb.num_nodes - self.placedb.num_filler_nodes :
                ].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)
            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(
                    fix_nodes_mask[: self.placedb.num_movable_nodes], 0
                )
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(
                    fix_nodes_mask[: self.placedb.num_movable_nodes], 0
                )
                grad = grad.view(-1)
            self.iteration += 1

            # only work in benchmarks without fence region, assume overflow has been updated
            if (
                len(self.placedb.regions) > 0
                and self.overflows
                and self.overflows[-1].max() < 0.3
                and self.alpha < 1024
            ):
                if (self.iteration % 20) == 0:
                    self.alpha *= 2
                    logging.info(
                        "preconditioning alpha = %g, best_overflow %g, overflow %g"
                        % (self.alpha, self.best_overflow, self.overflows[-1])
                    )

        return grad


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """

    def __init__(
        self,
        density_weight,
        params,
        placedb,
        data_collections: PlaceDataCollection,
        op_collections,
        global_place_params,
    ):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObj, self).__init__()

        # quadratic penalty
        self.density_quad_coeff = 2000
        self.init_density = None
        # increase density penalty if slow convergence
        self.density_factor = 1

        if len(placedb.regions) > 0:
            # fence region will enable quadratic penalty by default
            self.quad_penalty = True
        else:
            # non fence region will use first-order density penalty by default
            self.quad_penalty = False

        # timing diff
        self.use_timing_obj = False
        self.invoke_timing_count = 0
        self.timing_wns_coeff = 0.01
        self.timing_tns_coeff = 0.0001
        # fence region
        # update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None
        self.fix_nodes_mask = None
        if len(placedb.regions) > 0:
            # for subregion rough legalization, once stop updating, perform immediate greddy legalization once
            # this is to avoid repeated legalization
            # 1 represents already legal
            self.legal_mask = torch.zeros(len(placedb.regions) + 1)

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params

        self.gpu = params.gpu
        self.data_collections = data_collections
        self.op_collections = op_collections
        if len(placedb.regions) > 0:
            # different fence region needs different density weights in multi-electric field algorithm
            self.density_weight = torch.tensor(
                [density_weight] * (len(placedb.regions) + 1),
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device,
            )
        else:
            self.density_weight = torch.tensor(
                [density_weight],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device,
            )
        # Note: even for multi-electric fields, they use the same gamma
        num_bins_x = placedb.num_bins_x
        num_bins_y = placedb.num_bins_y
        name = "Global placement: %dx%d bins by default" % (num_bins_x, num_bins_y)
        logging.info(name)
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        self.gamma = torch.tensor(
            10 * self.base_gamma(params, placedb),
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device,
        )

        # compute weighted average wirelength from position

        name = "%dx%d bins" % (num_bins_x, num_bins_y)
        self.name = name

        if global_place_params["wirelength"] == "weighted_average":
            (
                self.op_collections.wirelength_op,
                self.op_collections.update_gamma_op,
            ) = self.build_weighted_average_wl(
                params, placedb, self.data_collections, self.op_collections.pin_pos_op
            )
        elif global_place_params["wirelength"] == "logsumexp":
            (
                self.op_collections.wirelength_op,
                self.op_collections.update_gamma_op,
            ) = self.build_logsumexp_wl(
                params, placedb, self.data_collections, self.op_collections.pin_pos_op
            )
        else:
            assert 0, "unknown wirelength model %s" % (
                global_place_params["wirelength"]
            )

        self.op_collections.density_overflow_op = self.build_electric_overflow(
            params, placedb, self.data_collections, self.num_bins_x, self.num_bins_y
        )

        self.op_collections.density_op = self.build_electric_potential(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=name,
        )
        if params.with_sta:
            self.op_collections.timing_propagation_op = (
                self.build_timing_propagation_op(params, placedb, self.data_collections)
            )
            self.op_collections.elmore_delay_op = self.build_elmore_delay_op(
                params,
                placedb,
                self.data_collections,
            )

        # build multiple density op for multi-electric field
        if len(self.placedb.regions) > 0:
            (
                self.op_collections.fence_region_density_ops,
                self.op_collections.fence_region_density_merged_op,
                self.op_collections.fence_region_density_overflow_merged_op,
            ) = self.build_multi_fence_region_density_op()
        self.op_collections.update_density_weight_op = self.build_update_density_weight(
            params, placedb
        )
        self.op_collections.precondition_op = self.build_precondition(
            params, placedb, self.data_collections, self.op_collections
        )
        self.op_collections.noise_op = self.build_noise(
            params, placedb, self.data_collections
        )
        if params.get_congestion_map:
            self.op_collections.get_congestion_map_op = (
                self.build_route_utilization_map(params, placedb, self.data_collections)
            )
        if params.routability_opt_flag:
            # compute congestion map, RISA/RUDY congestion map
            self.op_collections.route_utilization_map_op = (
                self.build_route_utilization_map(params, placedb, self.data_collections)
            )
            self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
                params, placedb, self.data_collections
            )
            self.op_collections.nctugr_congestion_map_op = (
                self.build_nctugr_congestion_map(params, placedb, self.data_collections)
            )
            self.op_collections.irt_egr_congestion_map_op = (
                self.build_irt_egr_congestion_map(
                    params, placedb, self.data_collections)
            )
            self.op_collections.gpugr_congestion_map_op = (
                self.build_gpugr_congestion_map(
                    params, placedb, self.data_collections)
            )
            # adjust instance area with congestion map
            self.op_collections.adjust_node_area_op = self.build_adjust_node_area(
                params, placedb, self.data_collections
            )
            

        self.Lgamma_iteration = global_place_params["iteration"]
        if "Llambda_density_weight_iteration" in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params[
                "Llambda_density_weight_iteration"
            ]
        else:
            self.Llambda_density_weight_iteration = 1
        if "Lsub_iteration" in global_place_params:
            self.Lsub_iteration = global_place_params["Lsub_iteration"]
        else:
            self.Lsub_iteration = 1
        if "routability_Lsub_iteration" in global_place_params:
            self.routability_Lsub_iteration = global_place_params[
                "routability_Lsub_iteration"
            ]
        else:
            self.routability_Lsub_iteration = self.Lsub_iteration
        self.start_fence_region_density = False

        # MFP macro overlap
        if self.params.macro_overlap_flag:
            self.op_collections.macro_overlap_op = self.build_macro_overlap(
                params, placedb, self.data_collections
            )

            self.macro_overlap_weight = torch.tensor(
                [0.0],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device,
            )

            self.op_collections.update_macro_overlap_weight_op = (
                self.build_update_macro_overlap_weight(params, placedb)
            )

        # # refine macro orientations
        # self.op_collections.macro_refinement_op = self.build_macro_refinement(
        #     params, placedb, self.data_collections, self.op_collections.hpwl_op
        # )

        # ========== L形Routability初始化 ==========
        self.use_l_shape_routability = False
        self.l_shape_routability_op = None
        
        self.l_shape_routability_weight = torch.tensor(
            [getattr(params, 'l_shape_routability_weight', 1e-5)],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device,
        )
        # 目标：L-shape梯度范数占density梯度范数的比例
        self.l_shape_grad_target_ratio = getattr(params, 'l_shape_grad_target_ratio', 0.3) 
        # 权重调整的平滑因子 (0~1, 越小越平滑)
        self.l_shape_weight_momentum = getattr(params, 'l_shape_weight_momentum', 0.1)
        self.l_shape_last_cost = None
        self.l_shape_last_weighted_cost = None
        self.l_shape_last_weight = None
        self.l_shape_last_target_weight = None
        self.l_shape_last_base_grad_norm = None
        self.l_shape_last_grad_norm = None
        self.l_shape_last_grad_ratio = None
        self.soft_l_last_summary = {}
        self._l_shape_auto_disabled = False
        self._l_shape_auto_disable_state = {}
        # ==========================================

    def reset_l_shape_weight_state(self):
        self._l_shape_weight_initialized = False
        self.l_shape_last_weight = None
        self.l_shape_last_target_weight = None
        self.l_shape_last_grad_ratio = None

    @staticmethod
    def _telemetry_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value

    def init_l_shape_routability(
        self,
        wire_width,
        num_bins_x,
        num_bins_y,
        target_density=1.0,
        target_demand=None,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
    ):
        """
        初始化L形routability模块
        应在steiner_topo_op有L方向信息后调用
        
        Args:
            wire_width: segment线宽
            num_bins_x, num_bins_y: 密度计算的bin数量
            target_density: 2D routing supply map
            target_demand: 2D routing demand map，用于标定L-shape密度单位
        """
        from dreamplace.ops.routability.l_shape_routability import LShapeRoutabilityOp
        
        if self.l_shape_routability_op is not None:
            self.l_shape_routability_op.update_targets(
                target_density=target_density,
                target_demand=target_demand,
                target_density_h=target_density_h,
                target_density_v=target_density_v,
                target_demand_h=target_demand_h,
                target_demand_v=target_demand_v,
            )
            self.use_l_shape_routability = True
            logging.info("L-shape routability already initialized; refreshed targets and re-enabled")
            return
        
        self.l_shape_routability_op = LShapeRoutabilityOp(
            placedb=self.placedb,
            params=self.params,
            wire_width=wire_width,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            target_density=target_density,
            target_demand=target_demand,
            target_density_h=target_density_h,
            target_density_v=target_density_v,
            target_demand_h=target_demand_h,
            target_demand_v=target_demand_v,
        )
        self.use_l_shape_routability = True
        
        if isinstance(target_density, torch.Tensor):
            if isinstance(target_demand, torch.Tensor):
                logging.info(f"L-shape routability initialized with routing supply/demand maps "
                            f"(supply min={target_density.min():.1f}, max={target_density.max():.1f}; "
                            f"demand min={target_demand.min():.1f}, max={target_demand.max():.1f}), "
                            f"init_weight={self.l_shape_routability_weight.item():.2e}, "
                            f"target_grad_ratio={self.l_shape_grad_target_ratio}")
            else:
                logging.info(f"L-shape routability initialized with routing supply map "
                            f"(min={target_density.min():.1f}, max={target_density.max():.1f}), "
                            f"init_weight={self.l_shape_routability_weight.item():.2e}, "
                            f"target_grad_ratio={self.l_shape_grad_target_ratio}")
        else:
            logging.info(f"L-shape routability initialized without a valid routing supply tensor, "
                        f"init_weight={self.l_shape_routability_weight.item():.2e}, "
                        f"target_grad_ratio={self.l_shape_grad_target_ratio}")
    
    def l_shape_routability_obj(self, pos, use_l_direction=True):
        """
        计算L形routability代价
        
        Args:
            pos: cell位置
            use_l_direction: 是否使用EGR的L方向信息
            
        Returns:
            可微的routability cost
        """
        if self.l_shape_routability_op is None:
            logging.warning("L-shape routability not initialized")
            return torch.zeros(1, dtype=pos.dtype, device=pos.device)
        
        return self.l_shape_routability_op(
            pos,
            self.op_collections.steiner_topo_op,
            self.op_collections.pin_pos_op,
            use_l_direction=use_l_direction,
        )
    
    def get_l_shape_density_map(self, pos, use_l_direction=True):
        """获取L形密度图用于可视化"""
        if self.l_shape_routability_op is None:
            return None
        return self.l_shape_routability_op.get_density_map(
            pos,
            self.op_collections.steiner_topo_op,
            self.op_collections.pin_pos_op,
            use_l_direction=use_l_direction,
        )

    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty + macro_overlap_weight * macro overlap penalty
        @param pos locations of cells
        @return objective value
        """
        self.wirelength = self.op_collections.wirelength_op(pos)
        if len(self.placedb.regions) > 0:
            self.density = self.op_collections.fence_region_density_merged_op(pos)
        else:
            self.density = self.op_collections.density_op(pos)

        if self.init_density is None:
            # record initial density
            self.init_density = self.density.data.clone()
            # density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(
                self.init_density > 0, 1 / self.init_density[self.init_density > 0]
            )
            self.quad_penalty_coeff = (
                self.density_quad_coeff / 2 * self.density_weight_grad_precond
            )
        if self.quad_penalty:
            # quadratic density penalty
            self.density = self.density * (1 + self.quad_penalty_coeff * self.density)
        if len(self.placedb.regions) > 0:
            result = self.wirelength + self.density_weight.dot(self.density)
        else:
            result = torch.add(
                self.wirelength,
                self.density,
                alpha=(self.density_factor * self.density_weight).item(),
            )

        if self.params.macro_overlap_flag:
            self.macro_overlap = self.op_collections.macro_overlap_op(pos)
            result = torch.add(
                result, self.macro_overlap, alpha=self.macro_overlap_weight.item()
            )
        if self.use_timing_obj:
            # log_dir = './log'
            # os.makedirs(log_dir, exist_ok=True)
            # with torch.profiler.profile(
            #     activities=[
            #         torch.profiler.ProfilerActivity.CPU,  # 追踪 CPU 上的操作
            #         # torch.profiler.ProfilerActivity.CUDA, # 追踪 GPU 上的操作 (如果可用)
            #     ],
            #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            #     record_shapes=True,  # 关闭形状记录以减少文件大小
            #     profile_memory=True, # 关闭内存分析以减少文件大小
            #     with_stack=True
            # ) as prof:
            #     wns, tns, ws, ts = self.timing_obj(pos)
            #     prof.step()  # 标记一个新的分析步骤
            #     exit(0)

            wns, tns, ws, ts = self.timing_obj(pos)
            slack = - (self.timing_wns_coeff * wns + self.timing_tns_coeff * tns)
            self.wns = wns
            self.tns = tns
            self.ws = ws
            self.ts = ts
            # logging.info(f"Timing slack: {slack}")
            result = torch.add(result, slack)

        return result

    def pin_2_libpin_ids(self, inst_size: torch.tensor, data_collections):
        nodes_id = data_collections.pin2node_map
        pins_main_id = data_collections.inst_main_id[nodes_id]
        inst_pins_mask = pins_main_id >= 0

        pins_cell_id = (
            data_collections.main_id_2_cell_id_start[pins_main_id[inst_pins_mask]]
            + inst_size[nodes_id[inst_pins_mask]].long()
        )
        libpin_ids = (
            data_collections.cell_id_2_libpin_id_start[pins_cell_id]
            + data_collections.pin_2_libpin_offset[inst_pins_mask]
        )
        return inst_pins_mask, libpin_ids

    def pin_caps_op(self, inst_size, data_collections):
        self.pin2libpin_flat_ids = torch.zeros(
            data_collections.pin2node_map.size()[0],
            dtype=data_collections.cell_id_2_libpin_id_start.dtype,
            device=data_collections.cell_id_2_libpin_id_start.device,
        )  # 初始化为-1
        pin_cap_base = torch.zeros(data_collections.pin2node_map.size()[0])
        pin_rcap_base = torch.zeros(data_collections.pin2node_map.size()[0])
        pin_fcap_base = torch.zeros(data_collections.pin2node_map.size()[0])
        self.inst_pins_mask, inst_pin2libpin_flat_ids = self.pin_2_libpin_ids(
            inst_size, data_collections
        )
        inst_pin_cap_base = data_collections.flat_lib_pin_cap[inst_pin2libpin_flat_ids]
        inst_pin_rcap_base = data_collections.flat_lib_pin_rcap[
            inst_pin2libpin_flat_ids
        ]
        inst_pin_fcap_base = data_collections.flat_lib_pin_fcap[
            inst_pin2libpin_flat_ids
        ]
        # TODO: cap limit
        self.pin2libpin_flat_ids[self.inst_pins_mask] = inst_pin2libpin_flat_ids
        pin_cap_base[self.inst_pins_mask] = inst_pin_cap_base
        pin_rcap_base[self.inst_pins_mask] = inst_pin_rcap_base
        pin_fcap_base[self.inst_pins_mask] = inst_pin_fcap_base

        # fill in the pin capacitance for non-pin nodes
        non_inst_pins_mask = ~self.inst_pins_mask
        pin_cap_base[data_collections.end_points] += data_collections.outcaps
        pin_rcap_base[data_collections.end_points] += data_collections.outcaps
        pin_fcap_base[data_collections.end_points] += data_collections.outcaps
        return pin_cap_base, pin_rcap_base, pin_fcap_base

    def build_ieda_rct(self):
        # ==============================================================================
        # --- 步骤 2: 初始化iEDA并使用正确的线电容为其构建RC树 ---
        # ==============================================================================
        logging.info("正在初始化iEDA STA引擎...")
        ieda_sta = IEDASta(self.placedb.data_manager.dir_workspace)
        num_pins = len(self.placedb.pin_names)
        self.id2net_name_map = {v: k for k, v in self.placedb.net_name2id_map.items()}

        pin_fa = self.data_collections.pin_fa.clone().detach().cpu().numpy()
        flat_pin_from = (
            self.data_collections.flat_pin_from.clone().detach().cpu().numpy()
        )
        flat_pin_to = self.data_collections.flat_pin_to.clone().detach().cpu().numpy()

        # 关键：使用纯粹的线电容 node_wire_caps_np 来构建iEDA的RC树
        node_wire_caps_np = self.op_collections.elmore_delay_op.net_cap.cpu().numpy()
        edge_resistance = (
            self.op_collections.elmore_delay_op.edge_resistance.cpu().numpy()
        )
        edge_to_res_map = {
            (u, v): r for u, v, r in zip(flat_pin_from, flat_pin_to, edge_resistance)
        }

        logging.info("开始为iEDA构建所有网络的RC树...")
        for net_id, net_name in self.id2net_name_map.items():
            # (RC树构建循环逻辑保持不变，确保传递的是 node_wire_caps)
            # ... 此处省略您已验证通过的RC树构建循环代码 ...
            net_pins_start = self.data_collections.flat_net2pin_start_map[net_id]
            net_pins_end = self.data_collections.flat_net2pin_start_map[net_id + 1]
            seed_pins = (
                self.data_collections.flat_net2pin_map[net_pins_start:net_pins_end]
                .cpu()
                .numpy()
            )
            if len(seed_pins) < 2:
                continue
            net_nodes_set = set()
            queue = [seed_pins[0]]
            visited_in_queue = {seed_pins[0]}
            head = 0
            while head < len(queue):
                current_node_idx = queue[head]
                head += 1
                net_nodes_set.add(current_node_idx)
                children_start = self.data_collections.flat_pin_to_start[
                    current_node_idx
                ]
                children_end = self.data_collections.flat_pin_to_start[
                    current_node_idx + 1
                ]
                for child_idx_tensor in self.data_collections.flat_pin_to[
                    children_start:children_end
                ]:
                    child_idx = child_idx_tensor.item()
                    if child_idx not in visited_in_queue:
                        queue.append(child_idx)
                        visited_in_queue.add(child_idx)
            net_nodes_global_indices = list(net_nodes_set)
            global_to_local_idx_map = {
                global_idx: i for i, global_idx in enumerate(net_nodes_global_indices)
            }
            true_driver_global_idx = self.data_collections.flat_net2pin_map[
                net_pins_start
            ].item()
            node_sta_names, node_is_pin, steiner_indices = [], [], []
            parent_indices, node_wire_caps, edge_resistances_net = [], [], []
            for global_idx in net_nodes_global_indices:
                if global_idx < num_pins:
                    node_is_pin.append(True)
                    node_sta_names.append(self.placedb.pin_names[global_idx])
                    steiner_indices.append(-1)
                else:
                    node_is_pin.append(False)
                    node_sta_names.append(f"S_{net_name}_{global_idx}")
                    steiner_indices.append(global_idx - num_pins)
                if global_idx == true_driver_global_idx:
                    parent_indices.append(-1)
                    edge_resistances_net.append(0.0)
                else:
                    parent_idx = pin_fa[global_idx]
                    parent_indices.append(global_to_local_idx_map.get(parent_idx, -1))
                    edge_resistances_net.append(
                        edge_to_res_map.get((parent_idx, global_idx), 0.0)
                    )
                node_wire_caps.append(node_wire_caps_np[global_idx])
            ieda_sta.build_rc_tree_from_flat_data(
                net_name,
                node_sta_names,
                node_is_pin,
                steiner_indices,
                parent_indices,
                node_wire_caps,
                edge_resistances_net,
                net_nodes_global_indices,
            )
        logging.info("所有网络的RC树构建完成。")

        # ==============================================================================
        # --- 步骤 3: 调用iEDA执行分析并获取所有调试信息 ---
        # ==============================================================================
        logging.info("调用iEDA执行时序分析并获取详细数据...")
        at_late_cpp, at_early_cpp, rt_late_cpp, rt_early_cpp = [], [], [], []
        pin_net_delay_cpp, cell_arc_delays_cpp, net_timing_details_cpp = [], [], []

        ieda_sta.update_and_get_all_pin_timings(
            self.placedb.pin_names,
            at_late_cpp,
            at_early_cpp,
            rt_late_cpp,
            rt_early_cpp,
            pin_net_delay_cpp,
            cell_arc_delays_cpp,
            net_timing_details_cpp,
        )
        logging.info(
            f"成功获取iEDA数据: {len(cell_arc_delays_cpp)}条CellArc, {len(net_timing_details_cpp)}条NetPin记录。"
        )
        return (
            at_late_cpp,
            at_early_cpp,
            rt_late_cpp,
            rt_early_cpp,
            pin_net_delay_cpp,
            cell_arc_delays_cpp,
            net_timing_details_cpp,
        )

    def write_timing_pin_all(
        self,
        at_late_cpp,
        at_early_cpp,
        rt_late_cpp,
        rt_early_cpp,
        pin_net_delay_cpp,
        cell_arc_delays_cpp,
        net_timing_details_cpp,
    ):
        # # ======================================================================
        # # --- 步骤 4: 生成所有Pin的详细时序参数对比报告并写入CSV文件 ---
        # # ======================================================================
        num_pins = len(self.placedb.pin_names)
        # 定义报告文件名（CSV）
        report_filename = "%s/%s_timing_pin_all_report.csv" % (
                self.params.result_dir, self.params.design_name())
        logging.info(f"\n正在生成详细时序对比报告，结果将写入CSV文件: {report_filename}")

        # 在函数内部导入所需模块
        import math
        import csv

        # 4a. 预处理iEDA返回的Net Pin数据，过滤掉 slew_ns 为 nan 的条目
        net_timing_details_cpp_filtered = [
            info
            for info in net_timing_details_cpp
            if not math.isnan(info.get("slew_ns", float("nan")))
        ]

        # 使用过滤后的干净数据来创建 ieda_pin_map
        ieda_pin_map = {
            (info["pin_name"], info["mode"], info["transition"]): info
            for info in net_timing_details_cpp_filtered
        }

        # 4b. 预处理Python端计算的所有Pin的数据
        op_elmore = self.op_collections.elmore_delay_op
        py_pin_r_load = op_elmore.loads["rise"].clone().detach().cpu().numpy()
        py_pin_f_load = op_elmore.loads["fall"].clone().detach().cpu().numpy()
        py_pin_r_delay = op_elmore.delays["rise"].clone().detach().cpu().numpy()
        py_pin_f_delay = op_elmore.delays["fall"].clone().detach().cpu().numpy()
        py_pin_r_ldelay = (
            op_elmore.ldelays["rise"].clone().detach().cpu().numpy()
        )
        py_pin_f_ldelay = (
            op_elmore.ldelays["fall"].clone().detach().cpu().numpy()
        )
        py_pin_r_beta = op_elmore.betas["rise"].clone().detach().cpu().numpy()
        py_pin_f_beta = op_elmore.betas["fall"].clone().detach().cpu().numpy()
        py_pin_r_impulse = (
            op_elmore.impulses["rise"].clone().detach().cpu().numpy()
        )
        py_pin_f_impulse = (
            op_elmore.impulses["fall"].clone().detach().cpu().numpy()
        )

        op_timing = self.op_collections.timing_propagation_op
        py_pin_r_slew = op_timing.pin_rtran.clone().detach().cpu().numpy()
        py_pin_f_slew = op_timing.pin_ftran.clone().detach().cpu().numpy()

        # 新增: 获取Python端的AT和RT数据
        py_r_aat_late = (
            op_timing.pin_rAAT
            .clone()
            .detach()
            .cpu()
            .numpy()
        )
        py_f_aat_late = (
            op_timing.pin_fAAT
            .clone()
            .detach()
            .cpu()
            .numpy()
        )
        py_r_rat_late = (
            op_timing.pin_rRAT
            .clone()
            .detach()
            .cpu()
            .numpy()
        )
        py_f_rat_late = (
            op_timing.pin_fRAT
            .clone()
            .detach()
            .cpu()
            .numpy()
        )
        
        python_pin_map = {}
        pin_names = self.placedb.pin_names

        for pin_id in range(num_pins):
            full_pin_name = pin_names[pin_id].decode("utf-8")

            # 为Rise和Fall transition分别创建数据条目
            key_rise = (full_pin_name, "Max", "Rise")
            python_pin_map[key_rise] = {
                "load": py_pin_r_load[pin_id],
                "delay": py_pin_r_delay[pin_id],
                "ldelay": py_pin_r_ldelay[pin_id],
                "beta": py_pin_r_beta[pin_id],
                "impulse": py_pin_r_impulse[pin_id],
                "slew": py_pin_r_slew[pin_id],
                # 新增AT/RT
                "at": py_r_aat_late[pin_id],
                "rt": py_r_rat_late[pin_id],
            }

            key_fall = (full_pin_name, "Max", "Fall")
            python_pin_map[key_fall] = {
                "load": py_pin_f_load[pin_id],
                "delay": py_pin_f_delay[pin_id],
                "ldelay": py_pin_f_ldelay[pin_id],
                "beta": py_pin_f_beta[pin_id],
                "impulse": py_pin_f_impulse[pin_id],
                "slew": py_pin_f_slew[pin_id],
                # 新增AT/RT
                "at": py_f_aat_late[pin_id],
                "rt": py_f_rat_late[pin_id],
            }

        # 4c. 构建用于排序和报告的中间列表
        report_data = []
        common_keys = set(python_pin_map.keys()).intersection(
            set(ieda_pin_map.keys())
        )

        # 新增: 创建一个从pin name到id的反向映射，以便查找AT/RT
        pin_name_to_id_map = {pin_names[i].decode("utf-8"): i for i in range(num_pins)}

        for key in common_keys:
            pin_name, _, _ = key
            py_data = python_pin_map[key]
            ieda_data = ieda_pin_map[key]

            # 获取pin_id，如果找不到则跳过 (更稳健)
            pin_id = pin_name_to_id_map.get(pin_name)
            if pin_id is None:
                continue

            # 从iEDA的列表中获取AT/RT
            if key[2] == "Rise":
                ieda_at_val = at_late_cpp[pin_id][0]
                ieda_rt_val = rt_late_cpp[pin_id][0]
            else:  # Fall                
                ieda_at_val = at_late_cpp[pin_id][1]
                ieda_rt_val = rt_late_cpp[pin_id][1]

            # 计算用于排序的差异值 (使用 RT 差异)
            diff = abs(py_data["rt"] - ieda_rt_val * 1000)
            report_data.append(
                {
                    "key": key,
                    "py_data": py_data,
                    "ieda_data": ieda_data,
                    "ieda_at": ieda_at_val * 1000,
                    "ieda_rt": ieda_rt_val * 1000,
                    "sort_diff": diff,
                }
            )

        # 4d. 按 Slew 差异进行降序排序
        report_data.sort(key=lambda item: item["sort_diff"], reverse=True)

        # 4e. 将报告写入CSV文件
        csv_header = [
            "Pin Name",
            "Mode",
            "Trans",
            "Py AT (ps)",
            "iEDA AT (ps)",
            "Py RT (ps)",
            "iEDA RT (ps)",
            "Py Slack(ps)",
            "iEDA Slack(ps)",
            "Py Slew (ps)",
            "iEDA Slew (ps)",
            "Py Delay (ps)",
            "iEDA Delay (ps)",
            "Py Load (pF)",
            "iEDA Load (pF)",
            "Py LDelay (ps)",
            "iEDA LDelay (ps)",
            "Py Beta",
            "iEDA Beta",
            "Py Impulse",
            "iEDA Impulse",
        ]

                
        with open(report_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            for item in report_data:
                pin_name, mode, trans = item["key"]
                py_data = item["py_data"]
                ieda_data = item["ieda_data"]

                # safe extraction and unit handling
                ieda_slew_val = ieda_data.get("slew_ns", float("nan"))
                ieda_delay_val = ieda_data.get("delay", float("nan"))
                ieda_load_val = ieda_data.get("load", float("nan"))
                ieda_ldelay_val = ieda_data.get("ldelay", float("nan"))
                ieda_beta_val = ieda_data.get("beta", float("nan"))
                ieda_impulse_val = (
                    ieda_data.get("impulse", 0.0)
                    if ieda_data.get("impulse", 0.0) >= 0
                    else 0.0
                )

                row = [
                    pin_name,
                    mode,
                    trans,
                    f"{py_data.get('at', float('nan')):.6f}",
                    f"{item.get('ieda_at', float('nan')):.6f}",
                    f"{py_data.get('rt', float('nan')):.6f}",
                    f"{item.get('ieda_rt', float('nan')):.6f}",
                    f"{py_data.get('rt', float('nan')) - py_data.get('at', float('nan')):.6f}",
                    f"{item.get('ieda_rt', float('nan')) - item.get('ieda_at', float('nan')):.6f}",
                    f"{py_data.get('slew', float('nan')):.6f}",
                    f"{1000 * ieda_slew_val:.6f}",
                    f"{py_data.get('delay', float('nan')):.6f}",
                    f"{ieda_delay_val:.6f}",
                    f"{py_data.get('load', float('nan')):.6f}",
                    f"{ieda_load_val:.6f}",
                    f"{py_data.get('ldelay', float('nan')):.6f}",
                    f"{ieda_ldelay_val:.6f}",
                    f"{py_data.get('beta', float('nan')):.6f}",
                    f"{ieda_beta_val:.6f}",
                    f"{py_data.get('impulse', float('nan')):.6f}",
                    f"{ieda_impulse_val:.6f}",
                ]

                writer.writerow(row)

        # 在写入完成后，向控制台打印一条确认信息
        logging.info(f"详细的对比报告已成功写入CSV文件: {report_filename}")

    def write_arc_all(
        self, cell_arc_delays_cpp
    ):

        # ==============================================================================
        # --- 步骤 4: 对齐Cell Arc Delay (新增Arc Sense列)，并写入CSV文件 ---
        # ==============================================================================
        logging.info("\n--- Cell Arc Delay 详细对比 (按差异绝对值降序排序) ---")
        import math
        import numpy as np
        import csv

        # 4a. 预处理iEDA返回的Cell Arc数据 (不变)
        ieda_arc_map = {
        }
        for arc in cell_arc_delays_cpp:
            key_t = (
                arc["inst_name"],
                arc["from_pin"],
                arc["to_pin"],
                arc["transition"],
                arc["arc_sense"],
            )
            if ieda_arc_map.get(key_t) is not None:
                tmp = ieda_arc_map[key_t]
                if tmp['delay_ns'] < arc['delay_ns']:
                    ieda_arc_map[key_t] = arc
                # logging.info(f"Warning: Duplicate arc key found: {key_t}. Overwriting previous entry.")
            else:
                ieda_arc_map[key_t] = arc
            

        # 4b. 预处理Python端计算的Cell Arc数据 (逻辑修正版)
        op_timing = self.op_collections.timing_propagation_op
        op_elmore = self.op_collections.elmore_delay_op

        py_cell_arc_rr_delays = (
            op_timing.cell_arc_rr_delays.cpu().numpy()
        )  # Delay for OUTPUT Rise
        py_cell_arc_ff_delays = (
            op_timing.cell_arc_ff_delays.cpu().numpy()
        )  # Delay for OUTPUT Fall

        py_cell_arc_rf_delays = (
            op_timing.cell_arc_rf_delays.cpu().numpy()
        )  # Delay for OUTPUT Rise
        py_cell_arc_fr_delays = (
            op_timing.cell_arc_fr_delays.cpu().numpy()
        )  # Delay for OUTPUT Fall

        py_pin_r_slew = op_timing.pin_rtran.cpu().numpy()
        py_pin_f_slew = op_timing.pin_ftran.cpu().numpy()
        py_pin_r_load = op_elmore.loads["rise"].clone().detach().cpu().numpy()
        py_pin_f_load = op_elmore.loads["fall"].clone().detach().cpu().numpy()

        python_arc_map = {}
        id2cell_name_map = {v: k for k, v in self.placedb.node_name2id_map.items()}
        pin_names = self.placedb.pin_names

        cell_arcs = self.data_collections.inst_flat_arcs.cpu().numpy()
        cell_arcs_start = self.data_collections.inst_flat_arcs_start.cpu().numpy()

        for cell_id, cell_name in id2cell_name_map.items():
            start_index = cell_arcs_start[cell_id]
            end_index = cell_arcs_start[cell_id + 1]
            if start_index == end_index:
                continue

            for inst_arc_idx in range(start_index, end_index):
                arc_info = cell_arcs[inst_arc_idx]

                in_pin_id, out_pin_id, _, _, arc_sense, _ = arc_info # there shouldnt be fall if timingtype is rise

                from_pin_name = pin_names[in_pin_id].decode("utf-8")
                to_pin_name = pin_names[out_pin_id].decode("utf-8")

                is_inverting = arc_sense == -1  # negative_unate
                is_unate = arc_sense == 0      # non_unate

                # --- 情况一: 报告中对应 output "Rise" 的行 ---
                delay_for_output_rise = (
                    py_cell_arc_fr_delays[inst_arc_idx]
                    if is_inverting
                    else (
                        py_cell_arc_rr_delays[inst_arc_idx]
                        if not is_unate
                        else max(
                            py_cell_arc_rr_delays[inst_arc_idx],
                            py_cell_arc_fr_delays[inst_arc_idx],
                        )
                    )
                )

                key_rise = (cell_name, from_pin_name, to_pin_name, "Rise", arc_sense)
                input_slew_rise = (
                    py_pin_f_slew[in_pin_id]
                    if is_inverting
                    else py_pin_r_slew[in_pin_id]
                )
                output_load_rise = py_pin_r_load[out_pin_id]
                if python_arc_map.get(key_rise) is not None:
                    tmp = python_arc_map[key_rise]
                    if tmp['delay'] < delay_for_output_rise:
                        python_arc_map[key_rise] = {
                            "delay": delay_for_output_rise,
                            "slew": input_slew_rise,
                            "load": output_load_rise,
                            "arc_sense": arc_sense,
                        }
                    # logging.info(f"Warning: Duplicate arc key found: {key_rise}. Overwriting previous entry.")
                else:
                    python_arc_map[key_rise] = {
                        "delay": delay_for_output_rise,
                        "slew": input_slew_rise,
                        "load": output_load_rise,
                        "arc_sense": arc_sense,
                    }

                # --- 情况二: 报告中对应 output "Fall" 的行 ---
                delay_for_output_fall = (
                    py_cell_arc_rf_delays[inst_arc_idx]
                    if is_inverting
                    else (
                        py_cell_arc_ff_delays[inst_arc_idx]
                        if not is_unate
                        else max(
                            py_cell_arc_ff_delays[inst_arc_idx],
                            py_cell_arc_rf_delays[inst_arc_idx],
                        )
                    )
                )
                
                key_fall = (cell_name, from_pin_name, to_pin_name, "Fall", arc_sense)
                input_slew_fall = (
                    py_pin_r_slew[in_pin_id]
                    if is_inverting
                    else py_pin_f_slew[in_pin_id]
                )
                output_load_fall = py_pin_f_load[out_pin_id]
                if python_arc_map.get(key_fall) is not None:
                    tmp= python_arc_map[key_fall]
                    if tmp['delay'] < delay_for_output_fall:
                        python_arc_map[key_fall] = {
                            "delay": delay_for_output_fall,
                            "slew": input_slew_fall,
                            "load": output_load_fall,
                            "arc_sense": arc_sense,
                        }
                    # logging.info(f"Warning: Duplicate arc key found: {key_fall}. Overwriting previous entry.")
                else:
                    python_arc_map[key_fall] = {
                        "delay": delay_for_output_fall,
                        "slew": input_slew_fall,
                        "load": output_load_fall,
                        "arc_sense": arc_sense,
                    }

        # 4c. 构建用于排序和报告的中间列表 (不变)
        report_data = []
        common_keys = set(python_arc_map.keys()).intersection(set(ieda_arc_map.keys()))

        for key in common_keys:
            py_data = python_arc_map[key]
            ieda_data = ieda_arc_map[key]
            delay_diff = py_data["delay"] - ieda_data["delay_ns"] * 1000
            report_item = {
                "key": key,
                "py_data": py_data,
                "ieda_data": ieda_data,
                "delay_diff": delay_diff,
            }
            report_data.append(report_item)

        # 4d. 按需排序 (当前为按Delay差异)
        report_data.sort(key=lambda item: abs(item["delay_diff"]), reverse=True)

        # 4e. 将 Arc 对比写入 CSV
        csv_filename = "%s/%s_cell_arc_delay_report.csv" % (
                self.params.result_dir, self.params.design_name())
        csv_header = [
            "Instance",
            "Arc",
            "Trans",
            "Sense",
            "Py Delay (ps)",
            "iEDA Delay (ps)",
            "Diff (ps)",
            "Py Slew (ps)",
            "iEDA Slew (ps)",
            "Py Load (pF)",
            "iEDA Load (pF)",
        ]

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            for item in report_data:
                inst, from_p, to_p, trans, arc_sense = item["key"]
                py_data = item["py_data"]
                ieda_data = item["ieda_data"]

                ieda_delay_ps = ieda_data.get("delay_ns", float("nan")) * 1000
                ieda_in_slew_ps = ieda_data.get("in_slew_ns", float("nan")) * 1000
                ieda_load = ieda_data.get("load_cap", float("nan"))

                row = [
                    inst,
                    f"{from_p}->{to_p}",
                    trans,
                    arc_sense,
                    f"{py_data['delay']:.6f}",
                    f"{ieda_delay_ps:.6f}",
                    f"{item['delay_diff']:+.6f}",
                    f"{py_data['slew']:.6f}",
                    f"{ieda_in_slew_ps:.6f}",
                    f"{py_data['load']:.6f}",
                    f"{ieda_load:.6f}",
                ]

                writer.writerow(row)

        logging.info(f"Cell arc 对比已写入CSV文件: {csv_filename}")

    def show_slack_compare(self, at_late_cpp, rt_late_cpp, wns, tns, ws, ts):
        # ==============================================================================
        # --- 步骤 6: 计算并返回最终的目标函数值 ---
        # ==============================================================================
        # (这部分计算WNS/TNS的逻辑保持不变)
        num_pins = len(self.placedb.pin_names)
        logging.info("\n--- 全局指标对比 (WNS/TNS) ---")
        t_dtype = self.op_collections.timing_propagation_op.pin_rAAT.dtype
        t_device = self.op_collections.timing_propagation_op.pin_rAAT.device
        # at_late_py = torch.max(
        #     self.op_collections.timing_propagation_op.pin_fAAT,
        # )
        # rt_late_py = torch.min(
        #     self.op_collections.timing_propagation_op.pin_rRAT,
        #     self.op_collections.timing_propagation_op.pin_fRAT,
        # )
        # setup_slack_py = rt_late_py - at_late_py
        # setup_slack_py_pins_only = setup_slack_py[:num_pins]

        at_late_cpp_tensor = torch.tensor(
            at_late_cpp, dtype=t_dtype, device=t_device
        )
        rt_late_cpp_tensor = torch.tensor(
            rt_late_cpp, dtype=t_dtype, device=t_device
        )
        setup_slack_cpp = torch.min(rt_late_cpp_tensor[:,0] - at_late_cpp_tensor[:,0], rt_late_cpp_tensor[:,1] - at_late_cpp_tensor[:,1])

        # wns_py_calc = torch.min(torch.clamp(setup_slack_py_pins_only, max=0)).item()
        # tns_py_calc = torch.sum(torch.clamp(setup_slack_py_pins_only, max=0)).item()

        slack_endpoints = setup_slack_cpp[self.data_collections.end_points]
        valid_setup_mask = torch.isfinite(slack_endpoints)
        setup_slack_cpp_valid = slack_endpoints[valid_setup_mask]
        if setup_slack_cpp_valid.numel() > 0:
            wns_cpp_calc = torch.min(setup_slack_cpp_valid).item()
            tns_cpp_calc = torch.sum(setup_slack_cpp_valid.clamp(max=0)).item()
        else:
            wns_cpp_calc = 0.0
            tns_cpp_calc = 0.0

        logging.info(
            f"WNS (Python Calculated): {wns / 1000:<15.4f} | WNS (iEDA): {wns_cpp_calc:<15.4f}"
        )
        logging.info(
            f"TNS (Python Calculated): {tns / 1000:<15.4f} | TNS (iEDA): {tns_cpp_calc:<15.4f}"
        )
        logging.info("-" * 60)
        logging.info(f"flow 输出的 WNS (orig wns): {wns.item():.4f}")
        logging.info(f"flow 输出的 TNS (orig tns): {tns.item():.4f}")
        logging.info(f"flow 输出的 WS (orig ws): {ws.item():.4f}")
        # logging.info(f"flow 输出的 TS (orig ts): {ts.item():.4f}")

    def write_first_level_pin_timing_log(self, net_timing_details_cpp):
        """
        @brief [修改后] 识别第一层传播引脚，并将详细的Slew/Impulse时序对比报告
            (精度为9位小数) 写入到一个名为 "timing_first_level_pins_report.csv" 的文件中。
        @param net_timing_details_cpp: 从 C++/iEDA 获取的包含 net pin 时序细节的列表。
        """
        import math
        import csv
        # ==============================================================================
        # --- 步骤 1: 定义文件名并准备数据 ---
        # ==============================================================================
        report_filename = "%s/%s_timing_first_level_pins_report.csv" % (
                self.params.result_dir, self.params.design_name())
        logging.info(f"\n--- [专属报告] 正在生成第一层传播引脚的详细报告 -> {report_filename}")

        # 1a. 找出所有“第一层传播”的引脚及其驱动源
        start_pin_ids = self.data_collections.start_points.cpu().numpy()

        pin_id_to_net_id_map = {}
        for net_id in range(len(self.data_collections.flat_net2pin_start_map) - 1):
            start_idx = self.data_collections.flat_net2pin_start_map[net_id]
            end_idx = self.data_collections.flat_net2pin_start_map[net_id + 1]
            for pin_id_tensor in self.data_collections.flat_net2pin_map[start_idx:end_idx]:
                pin_id_to_net_id_map[pin_id_tensor.item()] = net_id

        start_pin_to_sinks_map = {}
        for start_pin_id in start_pin_ids:
            net_id = pin_id_to_net_id_map.get(start_pin_id)
            if net_id is not None:
                sinks = []
                start_idx = self.data_collections.flat_net2pin_start_map[net_id]
                end_idx = self.data_collections.flat_net2pin_start_map[net_id + 1]
                for pin_id_tensor in self.data_collections.flat_net2pin_map[start_idx:end_idx]:
                    pin_id = pin_id_tensor.item()
                    if pin_id != start_pin_id:
                        sinks.append(pin_id)
                if sinks:
                    start_pin_to_sinks_map[start_pin_id] = sinks

        # 1b. 预处理Python和iEDA的数据
        op_timing = self.op_collections.timing_propagation_op
        op_elmore = self.op_collections.elmore_delay_op

        py_pin_r_impulse = op_elmore.impulses['rise'].clone().detach().cpu().numpy()
        py_pin_f_impulse = op_elmore.impulses['fall'].clone().detach().cpu().numpy()
        py_pin_r_slew = op_timing.pin_rtran.clone().detach().cpu().numpy()
        py_pin_f_slew = op_timing.pin_ftran.clone().detach().cpu().numpy()

        ieda_pin_map = {
            (info['pin_name'], info['mode'], info['transition']): info
            for info in net_timing_details_cpp
        }
        pin_names = self.placedb.pin_names

        # ==============================================================================
        # --- 步骤 2: 将报告写入CSV文件 ---
        # ==============================================================================
        csv_header = [
            "Group", "Pin Type", "Pin Name",
            "Py Rise Slew (ps)", "iEDA Rise Slew (ns)",
            "Py Fall Slew (ps)", "iEDA Fall Slew (ns)",
            "Py Rise Impulse", "iEDA Rise Impulse",
            "Py Fall Impulse", "iEDA Fall Impulse"
        ]

        with open(report_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            # 按 start_pin_id 排序，确保报告顺序一致
            for start_pin_id, sink_pin_ids in sorted(start_pin_to_sinks_map.items()):
                # --- 处理 Start Pin ---
                start_pin_name = pin_names[start_pin_id].decode('utf-8')
                py_r_slew_start = py_pin_r_slew[start_pin_id]
                py_f_slew_start = py_pin_f_slew[start_pin_id]
                py_r_impulse_start = py_pin_r_impulse[start_pin_id]
                py_f_impulse_start = py_pin_f_impulse[start_pin_id]

                key_rise_start = (start_pin_name, "Max", "Rise")
                key_fall_start = (start_pin_name, "Max", "Fall")
                ieda_data_rise = ieda_pin_map.get(key_rise_start, {})
                ieda_data_fall = ieda_pin_map.get(key_fall_start, {})

                ieda_r_slew_start = ieda_data_rise.get('slew_ns', float('nan'))
                ieda_f_slew_start = ieda_data_fall.get('slew_ns', float('nan'))

                ieda_r_impulse_sq = ieda_data_rise.get('impulse', -1.0)
                ieda_r_impulse_start = math.sqrt(ieda_r_impulse_sq) if ieda_r_impulse_sq >= 0 else float('nan')
                ieda_f_impulse_sq = ieda_data_fall.get('impulse', -1.0)
                ieda_f_impulse_start = math.sqrt(ieda_f_impulse_sq) if ieda_f_impulse_sq >= 0 else float('nan')

                start_pin_row = [
                    start_pin_name, "Start Pin", start_pin_name,
                    f"{py_r_slew_start:.9f}", f"{ieda_r_slew_start:.9f}",
                    f"{py_f_slew_start:.9f}", f"{ieda_f_slew_start:.9f}",
                    f"{py_r_impulse_start:.9f}", f"{ieda_r_impulse_start:.9f}",
                    f"{py_f_impulse_start:.9f}", f"{ieda_f_impulse_start:.9f}"
                ]
                writer.writerow(start_pin_row)

                # --- 处理该 Start Pin 驱动的所有 Sink Pin ---
                for sink_pin_id in sorted(sink_pin_ids):
                    sink_pin_name = pin_names[sink_pin_id].decode('utf-8')
                    py_r_slew_sink = py_pin_r_slew[sink_pin_id]
                    py_f_slew_sink = py_pin_f_slew[sink_pin_id]
                    py_r_impulse_sink = py_pin_r_impulse[sink_pin_id]
                    py_f_impulse_sink = py_pin_f_impulse[sink_pin_id]

                    key_rise_sink = (sink_pin_name, "Max", "Rise")
                    key_fall_sink = (sink_pin_name, "Max", "Fall")
                    ieda_data_rise_sink = ieda_pin_map.get(key_rise_sink, {})
                    ieda_data_fall_sink = ieda_pin_map.get(key_fall_sink, {})

                    ieda_r_slew_sink = ieda_data_rise_sink.get('slew_ns', float('nan'))
                    ieda_f_slew_sink = ieda_data_fall_sink.get('slew_ns', float('nan'))

                    ieda_r_impulse_sq_sink = ieda_data_rise_sink.get('impulse', -1.0)
                    ieda_r_impulse_sink = math.sqrt(ieda_r_impulse_sq_sink) if ieda_r_impulse_sq_sink >= 0 else float('nan')
                    ieda_f_impulse_sq_sink = ieda_data_fall_sink.get('impulse', -1.0)
                    ieda_f_impulse_sink = math.sqrt(ieda_f_impulse_sq_sink) if ieda_f_impulse_sq_sink >= 0 else float('nan')

                    sink_pin_row = [
                        start_pin_name, "Sink Pin", sink_pin_name,
                        f"{py_r_slew_sink:.9f}", f"{ieda_r_slew_sink:.9f}",
                        f"{py_f_slew_sink:.9f}", f"{ieda_f_slew_sink:.9f}",
                        f"{py_r_impulse_sink:.9f}", f"{ieda_r_impulse_sink:.9f}",
                        f"{py_f_impulse_sink:.9f}", f"{ieda_f_impulse_sink:.9f}"
                    ]
                    writer.writerow(sink_pin_row)

        logging.info(f"第一层传播引脚的详细报告已成功写入: {report_filename}")

    def timing_obj(self, pos):
        """
        @brief Compute objective and perform detailed timing analysis for debugging.
        @param pos locations of cells
        @return objective value
        """
        # ==============================================================================
        # --- 步骤 1: Python端计算，获取所有时序参数 ---
        # ==============================================================================
        import math
        import numpy as np

        new_x, new_y = self.op_collections.steiner_topo_op(
            self.op_collections.pin_pos_op(pos)
        )

        pin_caps_base, pin_rcaps_base, pin_fcaps_base = self.pin_caps_op(
            self.data_collections.inst_size, self.data_collections
        )

        # Elmore Delay算子
        pin_caps, loads, delays, ldelays, betas, impulses = (
            self.op_collections.elmore_delay_op(
                new_x,
                new_y,
                self.data_collections.net_flat_topo_sort,
                self.data_collections.net_flat_topo_sort_start,
                self.data_collections.pin_fa,
                self.data_collections.flat_pin_to_start,
                self.data_collections.flat_pin_to,
                self.data_collections.flat_pin_from,
                pin_caps_base,
                pin_rcaps_base,
                pin_fcaps_base,
            )
        )

        # 时序传播算子 (为获取WNS/TNS和完整的slew/load值，仍然需要运行)
        wns, tns, ws, ts = self.op_collections.timing_propagation_op(delays, impulses, loads)

        # if self.invoke_timing_count % 280 == 0:
        #     self.check_log(wns, tns, ws, ts)
        #     logging.info(f"\n--- [Timing Debug] 第 {self.invoke_timing_count} 次调用 timing_obj ---")
        #     logging.info(f"当前 WNS: {wns.item():.4f}, TNS: {tns.item():.4f}, WS: {ws.item():.4f}, TS: {ts:.4f}")
        self.invoke_timing_count += 1
        return wns, tns, ws, ts

    def check_log(self, wns, tns, ws, ts):
        # ==============================================================================
        # --- 步骤 2: 初始化iEDA并使用正确的线电容为其构建RC树 ---
        # ==============================================================================

        (
            at_late_cpp,
            at_early_cpp,
            rt_late_cpp,
            rt_early_cpp,
            pin_net_delay_cpp,
            cell_arc_delays_cpp,
            net_timing_details_cpp,
        ) = self.build_ieda_rct()

        self.write_timing_pin_all(
            at_late_cpp,
            at_early_cpp,
            rt_late_cpp,
            rt_early_cpp,
            pin_net_delay_cpp,
            cell_arc_delays_cpp,
            net_timing_details_cpp,
        )

        self.write_arc_all(cell_arc_delays_cpp)
        self.show_slack_compare(at_late_cpp, rt_late_cpp, wns, tns, ws, ts)

        # DEBUG
        self.write_first_level_pin_timing_log(net_timing_details_cpp)
        # DEBUG
        # ==============================================================================
        # --- 额外调试步骤: 输出特定引脚所在网络的完整信息 ---
        # ==============================================================================
        logging.info("\n--- [特定网络拓扑] 'DFF_659/Q_reg:Q' 所在网络的详细信息 ---")

        # ★★★ 您可以在这里修改想追踪的目标引脚名称 ★★★
        target_pin_full_name = "U4339:ZN"
        endpoints_str = self.placedb.pin_names[self.placedb.end_points] # .cpu().numpy().tolist()
        # logging.info(f"设计中的所有端点引脚共有 {len(endpoints_str)} 个，包括：")
        # for ep in endpoints_str:
        #     logging.info(f" - {ep.decode('utf-8')}")
        # self.debug_target_pin_net_info(target_pin_full_name)

    def debug_target_pin_net_info(self, target_pin_full_name):

        # 注意：请确保这个名字与 self.placedb.pin_names 中的某个条目完全匹配
        # 1. 准备必要的映射关系
        self.pin_names = self.placedb.pin_names
        self.name_to_id_map = {
            name.decode("utf-8"): i for i, name in enumerate(self.pin_names)
        }
        # 2. 查找目标引脚及其所在的Net
        if target_pin_full_name in self.name_to_id_map:
            target_pin_id = self.name_to_id_map[target_pin_full_name]

            # a. 构建 pin_id -> net_id 的反向映射 (如果尚未构建)
            pin_id_to_net_id_map = {}
            for net_id, net_name in self.id2net_name_map.items():
                start = self.data_collections.flat_net2pin_start_map[net_id]
                end = self.data_collections.flat_net2pin_start_map[net_id + 1]
                for pin_id_tensor in self.data_collections.flat_net2pin_map[start:end]:
                    pin_id_to_net_id_map[pin_id_tensor.item()] = net_id

            target_net_id = pin_id_to_net_id_map.get(target_pin_id)

            if target_net_id is not None:
                target_net_name = self.id2net_name_map[target_net_id]
                logging.info(
                    f"引脚 '{target_pin_full_name}' (ID: {target_pin_id}) 位于网络 '{target_net_name}' (ID: {target_net_id})。"
                )
                logging.info("该网络包含以下所有引脚：")
                logging.info(f"{'Pin ID':<15} | {'Pin Name'}")
                logging.info("-" * 60)

                # b. 根据net_id获取该网络的所有引脚
                start_idx = self.data_collections.flat_net2pin_start_map[target_net_id]
                end_idx = self.data_collections.flat_net2pin_start_map[
                    target_net_id + 1
                ]
                all_pin_ids_in_net = self.data_collections.flat_net2pin_map[
                    start_idx:end_idx
                ]

                # c. 遍历并打印所有引脚信息
                for pin_id_tensor in all_pin_ids_in_net:
                    pin_id = pin_id_tensor.item()
                    pin_name = self.pin_names[pin_id].decode("utf-8")
                    # 如果是目标引脚，特殊标记出来
                    marker = "★" if pin_id == target_pin_id else " "
                    logging.info(f"{marker} {pin_id:<13} | {pin_name}")
            else:
                logging.info(f"错误：在网络映射中未找到引脚 '{target_pin_full_name}'。")
        else:
            logging.info(f"错误：在设计中未找到名为 '{target_pin_full_name}' 的引脚。")

    def obj_and_grad_fn_old(self, pos_w, pos_g=None, admm_multiplier=None):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        if not self.start_fence_region_density:
            obj = self.obj_fn(pos_w, pos_g, admm_multiplier)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            obj.backward()
        else:
            num_nodes = self.placedb.num_nodes
            num_movable_nodes = self.placedb.num_movable_nodes
            num_filler_nodes = self.placedb.num_filler_nodes

            wl = self.op_collections.wirelength_op(pos_w)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            wl.backward()
            wl_grad = pos_w.grad.data.clone()
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.init_density is None:
                self.init_density = self.op_collections.density_op(
                    pos_w.data
                ).data.item()

            if self.quad_penalty:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)
                inner_density = (
                    inner_density
                    + self.density_quad_coeff / 2 / self.init_density * inner_density**2
                )
            else:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)

            inner_density.backward()
            inner_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map > 1e3
            inner_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes : num_nodes + num_movable_nodes].masked_fill_(
                mask, 0
            )
            inner_density_grad[num_nodes - num_filler_nodes : num_nodes].mul_(0.5)
            inner_density_grad[-num_filler_nodes:].mul_(0.5)
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.quad_penalty:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)
                outer_density = (
                    outer_density
                    + self.density_quad_coeff / 2 / self.init_density * outer_density**2
                )
            else:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)

            outer_density.backward()
            outer_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map < 1e3
            outer_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes : num_nodes + num_movable_nodes].masked_fill_(
                mask, 0
            )
            outer_density_grad[num_nodes - num_filler_nodes : num_nodes].mul_(0.5)
            outer_density_grad[-num_filler_nodes:].mul_(0.5)

            if self.quad_penalty:
                density = self.op_collections.density_op(pos_w.data)
                obj = wl.data.item() + self.density_weight * (
                    density
                    + self.density_quad_coeff / 2 / self.init_density * density**2
                )
            else:
                obj = (
                    wl.data.item()
                    + self.density_weight * self.op_collections.density_op(pos_w.data)
                )

            pos_w.grad.data.copy_(
                wl_grad
                + self.density_weight * (inner_density_grad + outer_density_grad)
            )

        self.op_collections.precondition_op(pos_w.grad, self.density_weight, 0)

        return obj, pos_w.grad

    def _apply_gradient_masks_only(self, grad):
        """
        Apply the same fixed/update masks as precondition, without extra scaling.
        This is used after adding L-shape gradients so fixed/terminated nodes stay zero.
        """
        with torch.no_grad():
            debug_mask = bool(getattr(self.params, "l_shape_mask_debug", True))
            before_nonzero = 0
            if debug_mask:
                before_nonzero = (grad.abs() > 0).sum().item()

            fixed_zero_xy = 2 * max(
                0, self.placedb.num_physical_nodes - self.placedb.num_movable_nodes
            )
            update_zero_xy = 0
            fix_nodes_zero_xy = 0

            # Keep fixed physical nodes zero (x and y parts).
            grad[self.placedb.num_movable_nodes : self.placedb.num_physical_nodes] = 0
            grad[
                self.placedb.num_nodes
                + self.placedb.num_movable_nodes : self.placedb.num_nodes
                + self.placedb.num_physical_nodes
            ] = 0

            # For multi-fence flow: stop gradients for terminated electric fields.
            if self.update_mask is not None and len(self.placedb.regions) > 0:
                precond_op = self.op_collections.precondition_op
                if (
                    hasattr(precond_op, "movablenode2fence_region_map_clamp")
                    and hasattr(precond_op, "filler2fence_region_map")
                ):
                    grad2 = grad.view(2, -1)
                    stop_mask = ~self.update_mask
                    movable_mask = stop_mask[
                        precond_op.movablenode2fence_region_map_clamp
                    ]
                    filler_mask = stop_mask[precond_op.filler2fence_region_map]
                    update_zero_xy = int(movable_mask.sum().item()) * 2 + int(
                        filler_mask.sum().item()
                    ) * 2
                    grad2[0, : self.placedb.num_movable_nodes].masked_fill_(
                        movable_mask, 0
                    )
                    grad2[1, : self.placedb.num_movable_nodes].masked_fill_(
                        movable_mask, 0
                    )
                    grad2[
                        0, self.placedb.num_nodes - self.placedb.num_filler_nodes :
                    ].masked_fill_(filler_mask, 0)
                    grad2[
                        1, self.placedb.num_nodes - self.placedb.num_filler_nodes :
                    ].masked_fill_(filler_mask, 0)

            # User-specified mask for temporarily fixed movable nodes.
            if self.fix_nodes_mask is not None:
                grad2 = grad.view(2, -1)
                fix_nodes_zero_xy = int(
                    self.fix_nodes_mask[: self.placedb.num_movable_nodes].sum().item()
                ) * 2
                grad2[0, : self.placedb.num_movable_nodes].masked_fill_(
                    self.fix_nodes_mask[: self.placedb.num_movable_nodes], 0
                )
                grad2[1, : self.placedb.num_movable_nodes].masked_fill_(
                    self.fix_nodes_mask[: self.placedb.num_movable_nodes], 0
                )

            if debug_mask:
                after_nonzero = (grad.abs() > 0).sum().item()
                removed_nonzero = before_nonzero - after_nonzero
                if not hasattr(self, "_l_shape_mask_debug_iter"):
                    self._l_shape_mask_debug_iter = 0
                self._l_shape_mask_debug_iter += 1
                interval = int(getattr(self.params, "l_shape_mask_debug_interval", 20))
                if interval <= 0:
                    interval = 1
                if self._l_shape_mask_debug_iter % interval == 0:
                    logging.info(
                        "L-shape mask debug iter=%d nonzero %d->%d (removed=%d), "
                        "target_zero_xy: fixed=%d, update=%d, fix_nodes=%d",
                        self._l_shape_mask_debug_iter,
                        before_nonzero,
                        after_nonzero,
                        removed_nonzero,
                        fixed_zero_xy,
                        update_zero_xy,
                        fix_nodes_zero_xy,
                    )

    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty + l_shape_routability
        @param pos locations of cells
        @return objective value
        """

        if pos.grad is not None:
            pos.grad.zero_()
        obj = self.obj_fn(pos)

        obj.backward()
        assert torch.isnan(pos.grad).any() == False, "Gradient contains NaN"
        self.op_collections.precondition_op(
            pos.grad, self.density_weight, self.update_mask, self.fix_nodes_mask
        )
        self.l_shape_last_cost = None
        self.l_shape_last_weighted_cost = None
        self.l_shape_last_weight = None
        self.l_shape_last_target_weight = None
        self.l_shape_last_base_grad_norm = None
        self.l_shape_last_grad_norm = None
        self.l_shape_last_grad_ratio = None
        self.soft_l_last_summary = {}
        
        # ========== L形Routability梯度 ==========
        if self.use_l_shape_routability and self.l_shape_routability_op is not None:
            # 保存 wirelength + density 的梯度
            base_grad = pos.grad.data.clone()
            base_grad_norm = base_grad.norm(p=2)
            base_grad_norm_value = float(base_grad_norm.item())
            
            pos.grad.zero_()
            
            # 计算L形routability cost 
            l_shape_cost = self.l_shape_routability_obj(pos, use_l_direction=True)
            l_shape_cost.backward()
            
            # 获取原始 L-shape 梯度范数
            l_shape_grad_raw = pos.grad.data.clone()
            l_shape_grad_norm = l_shape_grad_raw.norm(p=2)
            l_shape_grad_norm_value = float(l_shape_grad_norm.item())
            target_weight_value = None
            
            # 自适应调整权重
            # 目标: l_shape_grad_norm * weight ≈ target_ratio * base_grad_norm
            if l_shape_grad_norm > 1e-10 and base_grad_norm > 1e-10:
                target_weight = (self.l_shape_grad_target_ratio * base_grad_norm / l_shape_grad_norm).item()
                target_weight_value = float(target_weight)
                
                # 使用动量平滑更新权重
                old_weight = self.l_shape_routability_weight.item()
                # new_weight = (1 - self.l_shape_weight_momentum) * old_weight + \
                #              self.l_shape_weight_momentum * target_weight

                if not hasattr(self, '_l_shape_weight_initialized') or not self._l_shape_weight_initialized:
                    new_weight = target_weight
                    self._l_shape_weight_initialized = True
                    logging.info(f"L-shape weight auto-initialized: {new_weight:.4e} "
                                f"(base_grad={base_grad_norm:.4e}, l_shape_grad={l_shape_grad_norm:.4e})")
                else:
                    new_weight = (1 - self.l_shape_weight_momentum) * old_weight + \
                                 self.l_shape_weight_momentum * target_weight
                
                # 限制权重范围，防止过大或过小
                new_weight = max(1e-12, min(1, new_weight))
                self.l_shape_routability_weight.data.fill_(new_weight)
                
                # 应用权重到梯度
                pos.grad.data.mul_(new_weight)
                
                logging.debug(f"L-shape: cost={l_shape_cost.item():.4e}, "
                             f"grad_norm={l_shape_grad_norm:.4e}, "
                             f"base_grad_norm={base_grad_norm:.4e}, "
                             f"weight={old_weight:.4e}->{new_weight:.4e}")
            else:
                # 梯度太小，直接用当前权重
                pos.grad.data.mul_(self.l_shape_routability_weight.item())
            
            l_shape_weighted = l_shape_cost * self.l_shape_routability_weight.item()
            current_weight = float(self.l_shape_routability_weight.item())
            grad_ratio_value = None
            if l_shape_grad_norm_value > 1e-10 and base_grad_norm_value > 1e-10:
                grad_ratio_value = (
                    current_weight * l_shape_grad_norm_value / (base_grad_norm_value + 1e-12)
                )
            obj = obj + l_shape_weighted
            pos.grad.data.add_(base_grad)
            self._apply_gradient_masks_only(pos.grad.data)
            self.l_shape_last_cost = float(l_shape_cost.item())
            self.l_shape_last_weighted_cost = float(l_shape_weighted.item())
            self.l_shape_last_weight = current_weight
            self.l_shape_last_target_weight = target_weight_value
            self.l_shape_last_base_grad_norm = base_grad_norm_value
            self.l_shape_last_grad_norm = l_shape_grad_norm_value
            self.l_shape_last_grad_ratio = grad_ratio_value
            soft_debug = getattr(self.l_shape_routability_op, "cached_soft_debug", None)
            if isinstance(soft_debug, dict):
                self.soft_l_last_summary = {
                    key: self._telemetry_scalar(soft_debug.get(key))
                    for key in (
                        "diag_edge_count",
                        "mean_cost_gap",
                        "raw_cost_gap_p50",
                        "biased_cost_gap_p50",
                        "tau_source_gap",
                        "mean_max_prob",
                        "mean_entropy",
                        "near_tie_ratio",
                        "tau",
                        "effective_hotspot_weight",
                        "resolver_agreement_ratio",
                        "target_demand_supply_ratio",
                        "same_net_topo_cache_present",
                        "same_net_topo_nets",
                        "same_net_topo_segments_h",
                        "same_net_topo_segments_v",
                        "same_net_topo_diag_edges",
                        "same_net_topo_edges_with_topology",
                        "same_net_topo_edges_with_observed_intervals",
                        "same_net_topo_mean_gap",
                        "same_net_topo_tie_ratio",
                        "same_net_topo_zero_zero_edges",
                        "same_net_topo_exact_equal_edges",
                    )
                }
                density_op = getattr(self.l_shape_routability_op, "density_op", None)
                self.soft_l_last_summary["current_demand_supply_ratio"] = (
                    self._telemetry_scalar(
                        getattr(density_op, "last_demand_supply_ratio", None)
                    )
                )
            
            # self.check_gradient(pos)
        # ==========================================
        
        return obj, pos.grad

    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_collections.pos[0])

    def check_gradient(self, tpos):
        """
        @brief check gradient for debug
        @param pos locations of cells
        """

        pos = tpos.detach()
        pos.requires_grad_(True)

        # === 1. Wirelength梯度 ===
        wirelength = self.op_collections.wirelength_op(pos)
        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        # === 2. Density梯度 ===
        pos.grad.zero_()
        density = self.density_weight * self.op_collections.density_op(pos)
        density.backward()
        density_grad = pos.grad.clone()

        # plot_node_grad_directions(pos, density_grad, save_path="density_grad_directions.png")

        # === 3. Routability梯度 (新增) ===
        routability_grad = None
        if self.use_l_shape_routability:
            pos.grad.zero_()
            routability_cost = self.l_shape_routability_obj(pos)
            routability_weighted = routability_cost * self.l_shape_routability_weight.item()
            routability_weighted.backward()
            routability_grad = pos.grad.clone()

            # plot_node_grad_directions(pos, -wirelength_grad, -density_grad, -routability_grad,
            #                            title="Cell Move Directions",
            #                             save_path="cell_move_directions.png")

            num_nodes = self.placedb.num_nodes
            num_movable_nodes = self.placedb.num_movable_nodes
            
            def get_movable_part(tensor):
                x_part = tensor[:num_movable_nodes]
                y_part = tensor[num_nodes : num_nodes + num_movable_nodes]
                return torch.cat([x_part, y_part], dim=0)
            
            self.step = self.step + 1 if hasattr(self, 'step') else 0
            plot_node_grad_directions(get_movable_part(pos), 
                                      get_movable_part(-wirelength_grad), 
                                      get_movable_part(-density_grad), 
                                      get_movable_part(-routability_grad),
                                       title="Movable Cell Move Directions",
                                        save_path=f"cell_move_directions_movable_{self.step}.png")
            plot_node_grad_directions(pos, - (wirelength_grad + density_grad + routability_grad),
                                        title="Total Move Directions",
                                        save_path=f"total_move_directions_{self.step}.png")
            exit(0)

        # === 4. 计算梯度范数 ===
        wirelength_grad_norm = wirelength_grad.norm(p=1)
        density_grad_norm = density_grad.norm(p=1)

        # === 5. 输出对比结果 ===
        logging.info("=" * 60)
        logging.info("GRADIENT ANALYSIS:")
        
        logging.info("wirelength_grad norm = %.6E" % wirelength_grad_norm)
        logging.info("density_grad norm    = %.6E" % density_grad_norm)
        if routability_grad is not None:
            routability_grad_norm = routability_grad.norm(p=1)
            logging.info("routability_grad norm = %.6E" % routability_grad_norm)

            # adjust routability weight to let  routability grad norm be 0.001 ~ 0.1 of density grad norm
            # if routability_grad_norm > 0 and density_grad_norm > 0:
            #     ratio = routability_grad_norm /density_grad_norm
            #     if ratio < 0.001:
            #         new_weight = self.routability_weight.item() * 1.1
            #         self.routability_weight.data.fill_(new_weight)
            #         logging.info("Increase routability weight to %.6E" % new_weight)
            #     elif ratio > 0.1:
            #         new_weight = self.routability_weight.item() * 0.8
            #         self.routability_weight.data.fill_(new_weight)
            #         logging.info("Decrease routability weight to %.6E" % new_weight)
            
        
        logging.info("=" * 60)

        pos.grad.zero_()

    def check_l_shape_gradient_direction(self, tpos, step_sizes=[0.1, 1.0, 10.0]):
        """
        @brief 验证L-shape梯度方向是否能降低cost（比数值梯度检查更实用）
        @param tpos: cell位置  
        @param step_sizes: 测试的步长列表
        @return: dict with results
        """
        if not self.use_l_shape_routability or self.l_shape_routability_op is None:
            logging.warning("L-shape routability not enabled")
            return None
        
        logging.info("=" * 60)
        logging.info("L-SHAPE GRADIENT DIRECTION CHECK")
        logging.info("=" * 60)
        
        pos = tpos.detach().clone()
        pos.requires_grad_(True)
        
        # 计算当前cost和梯度
        if pos.grad is not None:
            pos.grad.zero_()
        cost0 = self.l_shape_routability_obj(pos)
        cost0.backward()
        grad = pos.grad.clone()
        
        logging.info(f"Initial cost: {cost0.item():.6e}")
        logging.info(f"Gradient norm: {grad.norm().item():.6e}")
        
        results = {'initial_cost': cost0.item(), 'step_results': []}
        
        for step in step_sizes:
            # 沿负梯度方向移动
            with torch.no_grad():
                pos_new = pos - step * grad / grad.norm()
                cost_new = self.l_shape_routability_obj(pos_new)
            
            improvement = cost0.item() - cost_new.item()
            pct = improvement / cost0.item() * 100
            
            status = "✓ cost decreased" if improvement > 0 else "✗ cost increased"
            logging.info(f"  Step={step:.1f}: cost={cost_new.item():.6e}, "
                        f"change={improvement:+.6e} ({pct:+.2f}%) {status}")
            
            results['step_results'].append({
                'step': step,
                'new_cost': cost_new.item(),
                'improvement': improvement,
                'success': improvement > 0
            })
        
        # 判断梯度是否有效
        num_success = sum(1 for r in results['step_results'] if r['success'])
        if num_success >= len(step_sizes) // 2 + 1:
            logging.info("  ✓ Gradient direction is VALID (cost decreases along -grad)")
            results['valid'] = True
        else:
            logging.warning("  ✗ Gradient direction may be INVALID")
            results['valid'] = False
        
        logging.info("=" * 60)
        return results

    def diagnose_l_shape_gradient_issues(self, tpos):
        """
        @brief 诊断L-shape梯度可能的问题
        检查：
        1. pos是否在边界内
        2. pin_pos是否在边界内  
        3. 梯度方向是否把cell推向边界外
        4. L-shape梯度与wirelength梯度的关系
        """
        if not self.use_l_shape_routability or self.l_shape_routability_op is None:
            logging.warning("L-shape routability not enabled")
            return
        
        logging.info("=" * 60)
        logging.info("L-SHAPE GRADIENT ISSUES DIAGNOSIS")
        logging.info("=" * 60)
        
        pos = tpos.detach().clone()
        pos.requires_grad_(True)
        
        num_nodes = self.placedb.num_nodes
        num_movable = self.placedb.num_movable_nodes
        xl, yl = self.placedb.xl, self.placedb.yl
        xh, yh = self.placedb.xh, self.placedb.yh
        
        # ========== 1. 检查pos是否在边界内 ==========
        logging.info("\n[1] Cell position boundary check:")
        pos_x = pos[:num_nodes].detach()
        pos_y = pos[num_nodes:2*num_nodes].detach()
        
        out_left = (pos_x[:num_movable] < xl).sum().item()
        out_right = (pos_x[:num_movable] > xh).sum().item()
        out_bottom = (pos_y[:num_movable] < yl).sum().item()
        out_top = (pos_y[:num_movable] > yh).sum().item()
        
        logging.info(f"  Chip boundary: x=[{xl}, {xh}], y=[{yl}, {yh}]")
        logging.info(f"  Movable cells out of bounds: left={out_left}, right={out_right}, "
                    f"bottom={out_bottom}, top={out_top}")
        
        if out_left + out_right + out_bottom + out_top > 0:
            # 找出超出最多的cell
            x_min = pos_x[:num_movable].min().item()
            x_max = pos_x[:num_movable].max().item()
            y_min = pos_y[:num_movable].min().item()
            y_max = pos_y[:num_movable].max().item()
            logging.warning(f"  ⚠ Cell range: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
            logging.warning(f"  ⚠ Overflow: left={xl-x_min:.1f}, right={x_max-xh:.1f}, "
                          f"bottom={yl-y_min:.1f}, top={y_max-yh:.1f}")
        else:
            logging.info("  ✓ All movable cells within bounds")
        
        # ========== 2. 检查pin_pos ==========
        logging.info("\n[2] Pin position boundary check:")
        pin_pos = self.op_collections.pin_pos_op(pos)
        num_pins = pin_pos.numel() // 2
        pin_x = pin_pos[:num_pins].detach()
        pin_y = pin_pos[num_pins:].detach()
        
        pin_x_min = pin_x.min().item()
        pin_x_max = pin_x.max().item()
        pin_y_min = pin_y.min().item()
        pin_y_max = pin_y.max().item()
        
        logging.info(f"  Pin range: x=[{pin_x_min:.1f}, {pin_x_max:.1f}], y=[{pin_y_min:.1f}, {pin_y_max:.1f}]")
        
        if pin_x_min < xl or pin_x_max > xh or pin_y_min < yl or pin_y_max > yh:
            logging.warning(f"  ⚠ Pins out of bounds!")
        else:
            logging.info("  ✓ All pins within bounds")
        
        # ========== 3. 计算L-shape梯度并分析方向 ==========
        logging.info("\n[3] L-shape gradient direction analysis:")
        if pos.grad is not None:
            pos.grad.zero_()
        
        l_shape_cost = self.l_shape_routability_obj(pos)
        l_shape_cost.backward()
        l_grad = pos.grad.clone()
        
        # 分析movable cells的梯度
        grad_x = l_grad[:num_movable]
        grad_y = l_grad[num_nodes:num_nodes+num_movable]
        
        # 统计梯度方向：正梯度意味着cost随位置增加而增加，所以优化应该往负方向走
        # 如果cell在左边界，梯度为正，则会往左推（出界）
        # 如果cell在右边界，梯度为负，则会往右推（出界）
        
        cells_near_left = pos_x[:num_movable] < (xl + (xh-xl)*0.1)
        cells_near_right = pos_x[:num_movable] > (xh - (xh-xl)*0.1)
        cells_near_bottom = pos_y[:num_movable] < (yl + (yh-yl)*0.1)
        cells_near_top = pos_y[:num_movable] > (yh - (yh-yl)*0.1)
        
        # 检查边界附近的cell梯度方向
        push_out_left = (cells_near_left & (grad_x > 0)).sum().item()  # 正梯度 -> 往左推
        push_out_right = (cells_near_right & (grad_x < 0)).sum().item()  # 负梯度 -> 往右推
        push_out_bottom = (cells_near_bottom & (grad_y > 0)).sum().item()
        push_out_top = (cells_near_top & (grad_y < 0)).sum().item()
        
        total_boundary_cells = (cells_near_left | cells_near_right | cells_near_bottom | cells_near_top).sum().item()
        total_push_out = push_out_left + push_out_right + push_out_bottom + push_out_top
        
        logging.info(f"  Cells near boundary: {total_boundary_cells}")
        logging.info(f"  Cells being pushed outward: {total_push_out} "
                    f"(left={push_out_left}, right={push_out_right}, "
                    f"bottom={push_out_bottom}, top={push_out_top})")
        
        if total_push_out > total_boundary_cells * 0.3:
            logging.warning(f"  ⚠ {total_push_out/total_boundary_cells*100:.1f}% of boundary cells "
                          f"are being pushed outward!")
        
        # ========== 4. 比较L-shape梯度与wirelength梯度 ==========
        logging.info("\n[4] L-shape vs Wirelength gradient comparison:")
        pos.grad.zero_()
        wirelength = self.op_collections.wirelength_op(pos)
        wirelength.backward()
        wl_grad = pos.grad.clone()
        
        # 计算夹角
        l_grad_flat = l_grad[:2*num_movable].flatten()
        wl_grad_flat = wl_grad[:2*num_movable].flatten()
        
        l_norm = l_grad_flat.norm()
        wl_norm = wl_grad_flat.norm()
        
        if l_norm > 1e-10 and wl_norm > 1e-10:
            cosine = (l_grad_flat @ wl_grad_flat) / (l_norm * wl_norm)
            angle = torch.acos(cosine.clamp(-1, 1)) * 180 / 3.14159
            logging.info(f"  L-shape grad norm: {l_norm.item():.4e}")
            logging.info(f"  Wirelength grad norm: {wl_norm.item():.4e}")
            logging.info(f"  Cosine similarity: {cosine.item():.4f}")
            logging.info(f"  Angle between gradients: {angle.item():.1f}°")
            
            if cosine.item() < -0.5:
                logging.warning("  ⚠ Gradients are nearly opposite! This may cause oscillation.")
            elif cosine.item() < 0:
                logging.info("  ⚠ Gradients have negative correlation (some conflict)")
            else:
                logging.info("  ✓ Gradients are compatible")
        
        logging.info("=" * 60)

    def check_l_shape_gradient_numerical(self, tpos, num_check=100, eps=1e-3, 
                                          check_movable_only=True, verbose=True):
        """
        @brief 数值验证L-shape routability梯度的正确性
        注意：对于bin-based密度函数，数值梯度检查可能失败是正常的，
        建议使用 check_l_shape_gradient_direction 来验证梯度方向。
        @param tpos: cell位置
        @param num_check: 检查的位置数量
        @param eps: 有限差分步长
        @param check_movable_only: 是否只检查movable cells
        @param verbose: 是否输出详细信息
        @return: dict with gradient check results
        """
        if not self.use_l_shape_routability or self.l_shape_routability_op is None:
            logging.warning("L-shape routability not enabled, skip gradient check")
            return None
        
        pos = tpos.detach().clone()
        pos.requires_grad_(True)
        
        num_nodes = self.placedb.num_nodes
        num_movable_nodes = self.placedb.num_movable_nodes
        
        # 确定要检查的索引
        if check_movable_only:
            # 只检查movable cells的x和y坐标
            check_indices_x = torch.randperm(num_movable_nodes)[:num_check//2]
            check_indices_y = num_nodes + torch.randperm(num_movable_nodes)[:num_check//2]
            check_indices = torch.cat([check_indices_x, check_indices_y])
        else:
            check_indices = torch.randperm(pos.numel())[:num_check]
        
        # === 1. 计算自动微分梯度 ===
        if pos.grad is not None:
            pos.grad.zero_()
        
        cost = self.l_shape_routability_obj(pos)
        cost.backward()
        auto_grad = pos.grad.clone()
        
        # === 2. 计算数值梯度 ===
        numerical_grad = torch.zeros_like(pos)
        
        logging.info("=" * 60)
        logging.info("L-SHAPE GRADIENT NUMERICAL CHECK")
        logging.info(f"Checking {len(check_indices)} positions with eps={eps}")
        logging.info("=" * 60)
        
        for i, idx in enumerate(check_indices):
            idx = idx.item()
            
            # f(x + eps)
            pos_plus = pos.detach().clone()
            pos_plus[idx] += eps
            with torch.no_grad():
                cost_plus = self.l_shape_routability_obj(pos_plus)
            
            # f(x - eps)
            pos_minus = pos.detach().clone()
            pos_minus[idx] -= eps
            with torch.no_grad():
                cost_minus = self.l_shape_routability_obj(pos_minus)
            
            # 中心差分
            numerical_grad[idx] = (cost_plus - cost_minus) / (2 * eps)
            
            if verbose and i < 20:  # 只打印前20个
                coord_type = "x" if idx < num_nodes else "y"
                node_idx = idx if idx < num_nodes else idx - num_nodes
                is_movable = node_idx < num_movable_nodes
                
                diff = abs(auto_grad[idx].item() - numerical_grad[idx].item())
                rel_err = diff / (abs(auto_grad[idx].item()) + 1e-10)
                
                logging.info(f"  [{i:3d}] node={node_idx:5d}({coord_type}), movable={is_movable}, "
                           f"auto={auto_grad[idx].item():+.6e}, num={numerical_grad[idx].item():+.6e}, "
                           f"diff={diff:.2e}, rel_err={rel_err:.2%}")
        
        # === 3. 统计分析 ===
        checked_auto = auto_grad[check_indices]
        checked_num = numerical_grad[check_indices]
        
        abs_diff = (checked_auto - checked_num).abs()
        rel_diff = abs_diff / (checked_auto.abs() + 1e-10)
        
        # 计算相关系数
        if checked_auto.std() > 1e-10 and checked_num.std() > 1e-10:
            correlation = torch.corrcoef(torch.stack([checked_auto, checked_num]))[0, 1].item()
        else:
            correlation = float('nan')
        
        # 计算cosine相似度
        if checked_auto.norm() > 1e-10 and checked_num.norm() > 1e-10:
            cosine_sim = (checked_auto @ checked_num) / (checked_auto.norm() * checked_num.norm())
            cosine_sim = cosine_sim.item()
        else:
            cosine_sim = float('nan')
        
        results = {
            'abs_diff_mean': abs_diff.mean().item(),
            'abs_diff_max': abs_diff.max().item(),
            'rel_diff_mean': rel_diff.mean().item(),
            'rel_diff_max': rel_diff.max().item(),
            'correlation': correlation,
            'cosine_similarity': cosine_sim,
            'auto_grad_norm': checked_auto.norm().item(),
            'numerical_grad_norm': checked_num.norm().item(),
            'auto_grad_mean': checked_auto.mean().item(),
            'numerical_grad_mean': checked_num.mean().item(),
            'num_zero_auto': (checked_auto.abs() < 1e-10).sum().item(),
            'num_zero_num': (checked_num.abs() < 1e-10).sum().item(),
        }
        
        logging.info("-" * 60)
        logging.info("SUMMARY:")
        logging.info(f"  Auto-diff grad norm:    {results['auto_grad_norm']:.6e}")
        logging.info(f"  Numerical grad norm:    {results['numerical_grad_norm']:.6e}")
        logging.info(f"  Abs diff (mean/max):    {results['abs_diff_mean']:.6e} / {results['abs_diff_max']:.6e}")
        logging.info(f"  Rel diff (mean/max):    {results['rel_diff_mean']:.2%} / {results['rel_diff_max']:.2%}")
        logging.info(f"  Correlation:            {results['correlation']:.4f}")
        logging.info(f"  Cosine similarity:      {results['cosine_similarity']:.4f}")
        logging.info(f"  Zero gradients (auto/num): {results['num_zero_auto']}/{results['num_zero_num']}")
        
        # 判断梯度是否正确
        if results['cosine_similarity'] > 0.99 and results['rel_diff_mean'] < 0.05:
            logging.info("  ✓ Gradient check PASSED")
        elif results['cosine_similarity'] > 0.9 and results['rel_diff_mean'] < 0.1:
            logging.info("  ~ Gradient check MARGINAL (may have minor issues)")
        else:
            logging.warning("  ✗ Gradient check FAILED (gradient may be incorrect)")
        
        logging.info("=" * 60)
        
        pos.grad.zero_()
        return results

    def diagnose_l_shape_gradient_chain(self, tpos):
        """
        @brief 诊断L-shape routability梯度链路，找出梯度断裂点
        @param tpos: cell位置
        """
        if not self.use_l_shape_routability or self.l_shape_routability_op is None:
            logging.warning("L-shape routability not enabled")
            return
        
        logging.info("=" * 60)
        logging.info("L-SHAPE GRADIENT CHAIN DIAGNOSIS")
        logging.info("=" * 60)
        
        pos = tpos.detach().clone()
        pos.requires_grad_(True)
        original_device = pos.device
        
        steiner_topo_op = self.op_collections.steiner_topo_op
        pin_pos_op = self.op_collections.pin_pos_op
        l_shape_op = self.l_shape_routability_op
        
        # === Step 1: pos → pin_pos ===
        logging.info("\n[Step 1] pos → pin_pos_op → pin_pos")
        pin_pos = pin_pos_op(pos)
        logging.info(f"  pin_pos.requires_grad: {pin_pos.requires_grad}")
        logging.info(f"  pin_pos.grad_fn: {pin_pos.grad_fn}")
        logging.info(f"  pin_pos.device: {pin_pos.device}")
        
        # === Step 2: pin_pos.cpu() → pin_pos_cpu ===
        logging.info("\n[Step 2] pin_pos → pin_pos.cpu() → pin_pos_cpu")
        if pin_pos.is_cuda:
            pin_pos_cpu = pin_pos.cpu()
            logging.info(f"  pin_pos_cpu.requires_grad: {pin_pos_cpu.requires_grad}")
            logging.info(f"  pin_pos_cpu.grad_fn: {pin_pos_cpu.grad_fn}")
        else:
            pin_pos_cpu = pin_pos
            logging.info(f"  pin_pos already on CPU")
        
        # === Step 3: pin_pos_cpu → steiner_topo_op → newx, newy ===
        logging.info("\n[Step 3] pin_pos_cpu → steiner_topo_op → newx, newy (on CPU)")
        newx, newy = steiner_topo_op(pin_pos_cpu)
        logging.info(f"  newx.requires_grad: {newx.requires_grad}")
        logging.info(f"  newy.requires_grad: {newy.requires_grad}")
        logging.info(f"  newx.grad_fn: {newx.grad_fn}")
        logging.info(f"  newx.device: {newx.device}")
        
        # === Step 4: newx, newy → segments (on CPU) ===
        logging.info("\n[Step 4] newx, newy → segment_builder → segments (on CPU)")
        
        flat_pin_from = steiner_topo_op.flat_pin_from
        flat_pin_to = steiner_topo_op.flat_pin_to
        if flat_pin_from.is_cuda:
            flat_pin_from = flat_pin_from.cpu()
            flat_pin_to = flat_pin_to.cpu()
        
        if steiner_topo_op.edge_l_directions is not None:
            l_directions = steiner_topo_op.edge_l_directions
            if l_directions.is_cuda:
                l_directions = l_directions.cpu()
        else:
            l_directions = torch.full((flat_pin_from.numel(),), -1, dtype=torch.int32, device=newx.device)
        
        segment_result = l_shape_op.segment_builder(newx, newy, flat_pin_from, flat_pin_to, l_directions)
        
        logging.info(f"  num_segments: {segment_result['num_segments']}")
        if segment_result['num_segments'] > 0:
            seg_llx = segment_result['segment_llx']
            seg_lly = segment_result['segment_lly']
            seg_size_x = segment_result['segment_size_x']
            seg_size_y = segment_result['segment_size_y']
            
            logging.info(f"  segment_llx.requires_grad: {seg_llx.requires_grad}")
            logging.info(f"  segment_size_x.requires_grad: {seg_size_x.requires_grad}")
            logging.info(f"  segment_llx.grad_fn: {seg_llx.grad_fn}")
            logging.info(f"  segment_llx.device: {seg_llx.device}")
            
            # === Step 5: segments → density_op → cost ===
            logging.info("\n[Step 5] segments → density_op → cost")
            segment_pos = segment_result['segment_pos']
            logging.info(f"  segment_pos.requires_grad: {segment_pos.requires_grad}")
            
            # 移到CUDA（如果需要）
            if original_device.type == 'cuda':
                segment_pos = segment_pos.to(original_device)
                seg_size_x = seg_size_x.to(original_device)
                seg_size_y = seg_size_y.to(original_device)
                logging.info(f"  Moved segments to CUDA for density_op")
            
            cost = l_shape_op.density_op(segment_pos, seg_size_x, seg_size_y)
            logging.info(f"  cost.requires_grad: {cost.requires_grad}")
            logging.info(f"  cost.grad_fn: {cost.grad_fn}")
            logging.info(f"  cost.item(): {cost.item():.6e}")
            
            # === Step 6: 尝试backward ===
            logging.info("\n[Step 6] Backward pass")
            if pos.grad is not None:
                pos.grad.zero_()
            
            try:
                cost.backward()
                grad_norm = pos.grad.norm().item() if pos.grad is not None else 0
                num_nonzero = (pos.grad.abs() > 1e-10).sum().item() if pos.grad is not None else 0
                logging.info(f"  ✓ backward succeeded")
                logging.info(f"  pos.grad norm: {grad_norm:.6e}")
                logging.info(f"  pos.grad non-zero count: {num_nonzero}/{pos.numel()}")
            except Exception as e:
                logging.error(f"  ✗ backward failed: {e}")
        
        # === 额外检查: 哪些cell有pin ===
        logging.info("\n[Extra] Cell-Pin connectivity check")
        num_movable_nodes = self.placedb.num_movable_nodes
        pin2node_map = self.data_collections.pin2node_map
        
        # 统计每个movable cell有多少pin
        movable_pin_counts = torch.zeros(num_movable_nodes, dtype=torch.long, device=pos.device)
        for node_id in pin2node_map:
            if node_id < num_movable_nodes:
                movable_pin_counts[node_id] += 1
        
        cells_with_pins = (movable_pin_counts > 0).sum().item()
        cells_without_pins = num_movable_nodes - cells_with_pins
        logging.info(f"  Movable cells with pins: {cells_with_pins}")
        logging.info(f"  Movable cells without pins: {cells_without_pins}")
        
        logging.info("=" * 60)

    def estimate_initial_learning_rate(self, x_k, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)
        new_lr = (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

        if torch.isnan(new_lr) or torch.isinf(new_lr):
            # backtracking line search (w. Armijo condition)
            def backtrack_line_search(f, df, x, alpha, beta):
                assert (0 < alpha < 0.5) and (0 < beta < 1.0)
                t = 1.0
                x1 = torch.autograd.Variable(x - t * df(x), requires_grad=True)
                while f(x1) > f(x) - alpha * t * df(x).norm(p=2):
                    t *= beta
                    x1 = x - t * df(x)
                return t, x1

            def f(x):
                return self.obj_and_grad_fn(x)[0]

            def df(x):
                return self.obj_and_grad_fn(x)[1]

            _, x_k_1 = backtrack_line_search(f, df, x_k, 0.3, 0.8)
            _, g_k_1 = self.obj_and_grad_fn(x_k_1)

            new_lr = (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

        return new_lr

    def build_weighted_average_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        # use WeightedAverageWirelength atomic
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm="merged",
        )

        # wirelength for position
        def build_wirelength_op(pos):
            pos1 = pin_pos_op(pos)
            return wirelength_for_pin_op(pos1)

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            # logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_logsumexp_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        wirelength_for_pin_op = logsumexp_wirelength.LogSumExpWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm="merged",
        )

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            # logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_elmore_delay_op(self, params, placedb, data_collections):
        rc_timing = RCTiming(
            r_unit=placedb.r_unit,
            c_unit=placedb.c_unit,
            scale_factor=params.scale_factor,
            dbu=placedb.dbu,
        )
        return rc_timing

    def build_timing_propagation_op(self, params, placedb, data_collections):

        return TimingPropagation(
            data_collections.inrdelays,
            data_collections.infdelays,
            data_collections.inrtrans,
            data_collections.inftrans,
            data_collections.outcaps,
            data_collections.pin2net_map,
            data_collections.start_points,
            data_collections.end_points,
            data_collections.clock_pins,
            data_collections.FF_ids,
            data_collections.clk_pin_rtran,
            data_collections.clk_pin_ftran,
            data_collections.net_flat_arcs_start,
            data_collections.net_flat_arcs,
            data_collections.net2driver_pin_map,
            data_collections.arcs_info,
            data_collections.inst_flat_arcs_start,
            data_collections.inst_flat_arcs,
            data_collections.endpoints_constraint_arcs,
            data_collections.flat_cells_by_level,
            data_collections.flat_cells_by_level_start,
            data_collections.flat_cells_by_reverse_level,
            data_collections.flat_cells_by_reverse_level_start,
            placedb.endpoints_rRAT,
            placedb.endpoints_fRAT,
        )

    def build_density_overflow(
        self, params, placedb, data_collections, num_bins_x, num_bins_y
    ):
        """
        @brief compute density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return density_overflow.DensityOverflow(
            data_collections.node_size_x,
            data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
        )

    def build_electric_overflow(
        self, params, placedb, data_collections, num_bins_x, num_bins_y
    ):
        """
        @brief compute electric density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return electric_overflow.ElectricOverflow(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
        )

    def build_density_potential(
        self, params, placedb, data_collections, num_bins_x, num_bins_y, padding, name
    ):
        """
        @brief NTUPlace3 density potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        local_num_bins_x = num_bins_x + 2 * padding
        local_num_bins_y = num_bins_y + 2 * padding
        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x) + 4 * bin_size_x) / bin_size_x
        )
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y) + 4 * bin_size_y) / bin_size_y
        )
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (
                name,
                local_num_bins_x,
                local_num_bins_y,
                bin_size_x / placedb.row_height,
                bin_size_y / placedb.row_height,
                max_num_bins,
                padding,
            )
        )
        if local_num_bins_x < max_num_bins:
            logging.warning(
                "local_num_bins_x (%d) < max_num_bins (%d)"
                % (local_num_bins_x, max_num_bins)
            )
        if local_num_bins_y < max_num_bins:
            logging.warning(
                "local_num_bins_y (%d) < max_num_bins (%d)"
                % (local_num_bins_y, max_num_bins)
            )

        node_size_x = placedb.node_size_x
        node_size_y = placedb.node_size_y

        # coefficients
        ax = (
            (4 / (node_size_x + 2 * bin_size_x) / (node_size_x + 4 * bin_size_x))
            .astype(placedb.dtype)
            .reshape([placedb.num_nodes, 1])
        )
        bx = (
            (2 / bin_size_x / (node_size_x + 4 * bin_size_x))
            .astype(placedb.dtype)
            .reshape([placedb.num_nodes, 1])
        )
        ay = (
            (4 / (node_size_y + 2 * bin_size_y) / (node_size_y + 4 * bin_size_y))
            .astype(placedb.dtype)
            .reshape([placedb.num_nodes, 1])
        )
        by = (
            (2 / bin_size_y / (node_size_y + 4 * bin_size_y))
            .astype(placedb.dtype)
            .reshape([placedb.num_nodes, 1])
        )

        # bell shape overlap function
        def npfx1(dist):
            # ax will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return 1.0 - ax.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfx2(dist):
            # bx will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return bx.reshape([placedb.num_nodes, 1]) * np.square(
                dist - node_size_x / 2 - 2 * bin_size_x
            ).reshape([placedb.num_nodes, 1])

        def npfy1(dist):
            # ay will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return 1.0 - ay.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfy2(dist):
            # by will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return by.reshape([placedb.num_nodes, 1]) * np.square(
                dist - node_size_y / 2 - 2 * bin_size_y
            ).reshape([placedb.num_nodes, 1])

        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_x = (
            npfx1(0) + 2 * npfx1(bin_size_x) + 2 * npfx2(2 * bin_size_x)
        )
        cx = (
            node_size_x.reshape([placedb.num_nodes, 1]) / integral_potential_x
        ).reshape([placedb.num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_y = (
            npfy1(0) + 2 * npfy1(bin_size_y) + 2 * npfy2(2 * bin_size_y)
        )
        cy = (
            node_size_y.reshape([placedb.num_nodes, 1]) / integral_potential_y
        ).reshape([placedb.num_nodes, 1])

        return density_potential.DensityPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            ax=torch.tensor(
                ax.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            bx=torch.tensor(
                bx.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            cx=torch.tensor(
                cx.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            ay=torch.tensor(
                ay.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            by=torch.tensor(
                by.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            cy=torch.tensor(
                cy.ravel(),
                dtype=data_collections.pos[0].dtype,
                device=data_collections.pos[0].device,
            ),
            bin_center_x=data_collections.bin_center_x_padded(
                placedb, padding, num_bins_x
            ),
            bin_center_y=data_collections.bin_center_y_padded(
                placedb, padding, num_bins_y
            ),
            target_density=data_collections.target_density,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            padding=padding,
            sigma=(1.0 / 16) * placedb.width / bin_size_x,
            delta=2.0,
        )

    def build_electric_potential(
        self,
        params,
        placedb,
        data_collections,
        num_bins_x,
        num_bins_y,
        name,
        region_id=None,
        fence_regions=None,
    ):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param name string for printing
        @param fence_regions a [n_subregions, 4] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        max_num_bins_x = np.ceil(
            (
                np.amax(placedb.node_size_x[0 : placedb.num_movable_nodes])
                + 2 * bin_size_x
            )
            / bin_size_x
        )
        max_num_bins_y = np.ceil(
            (
                np.amax(placedb.node_size_y[0 : placedb.num_movable_nodes])
                + 2 * bin_size_y
            )
            / bin_size_y
        )
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (
                name,
                num_bins_x,
                num_bins_y,
                bin_size_x / placedb.row_height,
                bin_size_y / placedb.row_height,
                max_num_bins,
                0,
            )
        )
        if num_bins_x < max_num_bins:
            logging.warning(
                "num_bins_x (%d) < max_num_bins (%d)" % (num_bins_x, max_num_bins)
            )
        if num_bins_y < max_num_bins:
            logging.warning(
                "num_bins_y (%d) < max_num_bins (%d)" % (num_bins_y, max_num_bins)
            )
        # for fence region, the target density is different from different regions
        target_density = (
            data_collections.target_density.item()
            if fence_regions is None
            else placedb.target_density_fence_region[region_id]
        )
        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            fast_mode=params.RePlAce_skip_energy_flag,
            region_id=region_id,
            fence_regions=fence_regions,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb,
        )

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        self.data_collections.pos[0].grad.zero_()

        if len(self.placedb.regions) > 0:
            density_list = []
            density_grad_list = []
            for density_op in self.op_collections.fence_region_density_ops:
                density_i = density_op(self.data_collections.pos[0])
                density_list.append(density_i.data.clone())
                density_i.backward()
                density_grad_list.append(self.data_collections.pos[0].grad.data.clone())
                self.data_collections.pos[0].grad.zero_()

            # record initial density
            self.init_density = torch.stack(density_list)
            # density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(
                self.init_density > 0, 1 / self.init_density[self.init_density > 0]
            )
            # compute u
            self.density_weight_u = self.init_density * self.density_weight_grad_precond
            self.density_weight_u += (
                0.5 * self.density_quad_coeff * self.density_weight_u**2
            )
            # compute s
            density_weight_s = (
                1
                + self.density_quad_coeff
                * self.init_density
                * self.density_weight_grad_precond
            )
            # compute density grad L1 norm
            density_grad_norm = sum(
                self.density_weight_u[i]
                * density_weight_s[i]
                * density_grad_list[i].norm(p=1)
                for i in range(density_weight_s.size(0))
            )

            self.density_weight_u *= (
                params.density_weight * wirelength_grad_norm / density_grad_norm
            )
            # set initial step size for density weight update
            self.density_weight_step_size_inc_low = 1.03
            self.density_weight_step_size_inc_high = 1.04
            self.density_weight_step_size = (
                self.density_weight_step_size_inc_low - 1
            ) * self.density_weight_u.norm(p=2)
            # commit initial density weight
            self.density_weight = self.density_weight_u * density_weight_s

        else:
            density = self.op_collections.density_op(self.data_collections.pos[0])
            # record initial density
            self.init_density = density.data.clone()
            density.backward()
            density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

            grad_norm_ratio = wirelength_grad_norm / density_grad_norm
            self.density_weight = torch.tensor(
                [params.density_weight * grad_norm_ratio],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device,
            )

        return self.density_weight

    def build_update_density_weight(self, params, placedb, algo="overflow"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        # params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl / params.scale_factor
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        # params for overflow mode from elfPlace
        assert algo in {"hpwl", "overflow"}, logging.error(
            "density weight update not supports hpwl mode or overflow mode"
        )

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            # based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(
                        np.power(0.9999, float(iteration)), 0.98
                    )
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl
                    ).clamp(min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu
                self.timing_tns_coeff *= 1.01
                self.timing_wns_coeff *= 1.01

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            assert (
                self.quad_penalty == True
            ), "[Error] density weight update based on overflow only works for quadratic density penalty"
            # based on overflow
            # stop updating if a region has lower overflow than stop overflow
            with torch.no_grad():
                density_norm = cur_metric.density * self.density_weight_grad_precond
                density_weight_grad = (
                    density_norm + self.density_quad_coeff / 2 * density_norm**2
                )
                density_weight_grad /= density_weight_grad.norm(p=2)

                self.density_weight_u += (
                    self.density_weight_step_size * density_weight_grad
                )
                density_weight_s = 1 + self.density_quad_coeff * density_norm

                density_weight_new = (self.density_weight_u * density_weight_s).clamp(
                    max=10
                )

                # conditional update if this region's overflow is higher than stop overflow
                if self.update_mask is None:
                    self.update_mask = cur_metric.overflow >= self.params.stop_overflow
                else:
                    # restart updating is not allowed
                    self.update_mask &= cur_metric.overflow >= self.params.stop_overflow
                self.density_weight.masked_scatter_(
                    self.update_mask, density_weight_new[self.update_mask]
                )

                # update density weight step size
                rate = torch.log(
                    self.density_quad_coeff * density_norm.norm(p=2)
                ).clamp(min=0)
                rate = rate / (1 + rate)
                rate = (
                    rate
                    * (
                        self.density_weight_step_size_inc_high
                        - self.density_weight_step_size_inc_low
                    )
                    + self.density_weight_step_size_inc_low
                )
                self.density_weight_step_size *= rate
                self.timing_tns_coeff *= 1.01
                self.timing_wns_coeff *= 1.01

        if not self.quad_penalty and algo == "overflow":
            logging.warn(
                "quadratic density penalty is disabled, density weight update is forced to be based on HPWL"
            )
            algo = "hpwl"
        if len(self.placedb.regions) == 0 and algo == "overflow":
            logging.warn(
                "for benchmark without fence region, density weight update is forced to be based on HPWL"
            )
            algo = "hpwl"

        update_density_weight_op = {
            "hpwl": update_density_weight_op_hpwl,
            "overflow": update_density_weight_op_overflow,
        }[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        return params.gamma * (self.bin_size_x + self.bin_size_y)

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        # overflow can have multiple values for fence regions, use their weighted average based on movable node number
        if overflow.numel() == 1:
            overflow_avg = overflow
        else:
            overflow_avg = overflow
        coef = torch.pow(10, (overflow_avg - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_((base_gamma * coef).item())
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y], dim=0
        ).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                if self.fix_nodes_mask is not None:
                    noise = noise.view(2, -1)
                    noise[0, : placedb.num_movable_nodes].masked_fill_(
                        self.fix_nodes_mask[: placedb.num_movable_nodes], 0
                    )
                    noise[1, : placedb.num_movable_nodes].masked_fill_(
                        self.fix_nodes_mask[: placedb.num_movable_nodes], 0
                    )
                    noise = noise.view(-1)
                noise[
                    placedb.num_movable_nodes : placedb.num_nodes
                    - placedb.num_filler_nodes
                ].zero_()
                noise[
                    placedb.num_nodes
                    + placedb.num_movable_nodes : 2 * placedb.num_nodes
                    - placedb.num_filler_nodes
                ].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb, data_collections, op_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """

        return PreconditionOp(placedb, data_collections, op_collections)

    def build_route_utilization_map(self, params, placedb, data_collections):
        """
        @brief routing congestion map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        congestion_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            initial_horizontal_utilization_map=data_collections.initial_horizontal_utilization_map,
            initial_vertical_utilization_map=data_collections.initial_vertical_utilization_map,
            deterministic_flag=params.deterministic_flag,
        )

        # congestion_macros_op = rudy_macros.RudyWithMacros(
        #     netpin_start=data_collections.flat_net2pin_start_map,
        #     flat_netpin=data_collections.flat_net2pin_map,
        #     net_weights=data_collections.net_weights,
        #     fp_info=data_collections.fp_info,
        #     num_bins_x=placedb.num_routing_grids_x,
        #     num_bins_y=placedb.num_routing_grids_y,
        #     node_size_x=data_collections.node_size_x,
        #     node_size_y=data_collections.node_size_y,
        #     num_movable_nodes=placedb.num_movable_nodes,
        #     movable_macro_mask=data_collections.movable_macro_mask,
        #     num_terminals=placedb.num_terminals,
        #     fixed_macro_mask=data_collections.fixed_macro_mask,
        #     params=params,
        # )

        def route_utilization_map_op(pos):
            pin_pos = self.op_collections.pin_pos_op(pos)
            return congestion_op(pin_pos)

        return route_utilization_map_op

    def build_pin_utilization_map(self, params, placedb, data_collections):
        """
        @brief pin density map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        return pin_utilization.PinUtilization(
            pin_weights=data_collections.pin_weights,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            pin_stretch_ratio=params.pin_stretch_ratio,
            deterministic_flag=params.deterministic_flag,
        )

    def build_nctugr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call NCTUgr for congestion estimation
        """
        path = "%s/%s" % (params.result_dir, params.design_name())
        return nctugr_binary.NCTUgr(
            aux_input_file=os.path.realpath(params.aux_input),
            param_setting_file="%s/../thirdparty/NCTUgr.ICCAD2012/DAC12.set"
            % (os.path.dirname(os.path.realpath(__file__))),
            tmp_pl_file="%s/%s.NCTUgr.pl"
            % (os.path.realpath(path), params.design_name()),
            tmp_output_file="%s/%s.NCTUgr"
            % (os.path.realpath(path), params.design_name()),
            horizontal_routing_capacities=torch.from_numpy(
                placedb.unit_horizontal_capacities * placedb.routing_grid_size_y
            ),
            vertical_routing_capacities=torch.from_numpy(
                placedb.unit_vertical_capacities * placedb.routing_grid_size_x
            ),
            params=params,
            placedb=placedb,
        )
    
    def build_irt_egr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call iRT egr for congestion estimation
        """
        # path = "%s/%s" % (params.result_dir, params.design_name())
        return eGR.IRT_eGR(
            params=params,
            placedb=placedb,
        )

    def build_gpugr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call Xplace gpugr for congestion estimation
        """
        return gpugr_congestion.GPUGR(
            params=params,
            placedb=placedb,
        )
        

    def build_adjust_node_area(self, params, placedb, data_collections):
        """
        @brief adjust cell area according to routing congestion and pin utilization map
        """
        total_movable_area = (
            data_collections.node_size_x[: placedb.num_movable_nodes]
            * data_collections.node_size_y[: placedb.num_movable_nodes]
        ).sum()
        total_filler_area = (
            data_collections.node_size_x[-placedb.num_filler_nodes :]
            * data_collections.node_size_y[-placedb.num_filler_nodes :]
        ).sum()
        total_place_area = (
            total_movable_area + total_filler_area
        ) / data_collections.target_density
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin_weights=data_collections.pin_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            route_num_bins_x=placedb.num_routing_grids_x,
            route_num_bins_y=placedb.num_routing_grids_y,
            pin_num_bins_x=placedb.num_routing_grids_x,
            pin_num_bins_y=placedb.num_routing_grids_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_place_area - total_movable_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity,
        )

        def build_adjust_node_area_op(pos, route_utilization_map, pin_utilization_map):
            return adjust_node_area_op(
                pos,
                data_collections.node_size_x,
                data_collections.node_size_y,
                data_collections.pin_offset_x,
                data_collections.pin_offset_y,
                data_collections.target_density,
                route_utilization_map,
                pin_utilization_map,
            )

        build_adjust_node_area_op._xplace_adjust_node_area_impl = adjust_node_area_op
        return build_adjust_node_area_op

    def build_fence_region_density_op(self, fence_region_list, node2fence_region_map):
        assert (
            type(fence_region_list) == list and len(fence_region_list) == 2
        ), "Unsupported fence region list"
        self.data_collections.node2fence_region_map = torch.from_numpy(
            self.placedb.node2fence_region_map[: self.placedb.num_movable_nodes]
        ).to(fence_region_list[0].device)
        self.op_collections.inner_fence_region_density_op = (
            self.build_electric_potential(
                self.params,
                self.placedb,
                self.data_collections,
                self.num_bins_x,
                self.num_bins_y,
                name=self.name,
                fence_regions=fence_region_list[0],
                fence_region_mask=self.data_collections.node2fence_region_map > 1e3,
            )
        )  # density penalty for inner cells
        self.op_collections.outer_fence_region_density_op = (
            self.build_electric_potential(
                self.params,
                self.placedb,
                self.data_collections,
                self.num_bins_x,
                self.num_bins_y,
                name=self.name,
                fence_regions=fence_region_list[1],
                fence_region_mask=self.data_collections.node2fence_region_map < 1e3,
            )
        )  # density penalty for outer cells

    def build_multi_fence_region_density_op(self):
        # region 0, ..., region n, non_fence_region
        self.op_collections.fence_region_density_ops = []

        for i, fence_region in enumerate(
            self.data_collections.virtual_macro_fence_region[:-1]
        ):
            self.op_collections.fence_region_density_ops.append(
                self.build_electric_potential(
                    self.params,
                    self.placedb,
                    self.data_collections,
                    self.num_bins_x,
                    self.num_bins_y,
                    name=self.name,
                    region_id=i,
                    fence_regions=fence_region,
                )
            )

        self.op_collections.fence_region_density_ops.append(
            self.build_electric_potential(
                self.params,
                self.placedb,
                self.data_collections,
                self.num_bins_x,
                self.num_bins_y,
                name=self.name,
                region_id=len(self.placedb.regions),
                fence_regions=self.data_collections.virtual_macro_fence_region[-1],
            )
        )

        def merged_density_op(pos):
            # stop mask is to stop forward of density
            # 1 represents stop flag
            res = torch.stack(
                [
                    density_op(pos, mode="density")
                    for density_op in self.op_collections.fence_region_density_ops
                ]
            )
            return res

        def merged_density_overflow_op(pos):
            # stop mask is to stop forward of density
            # 1 represents stop flag
            overflow_list, max_density_list = [], []
            for density_op in self.op_collections.fence_region_density_ops:
                overflow, max_density = density_op(pos, mode="overflow")
                overflow_list.append(overflow)
                max_density_list.append(max_density)
            overflow_list, max_density_list = torch.stack(overflow_list), torch.stack(
                max_density_list
            )

            return overflow_list, max_density_list

        self.op_collections.fence_region_density_merged_op = merged_density_op

        self.op_collections.fence_region_density_overflow_merged_op = (
            merged_density_overflow_op
        )
        return (
            self.op_collections.fence_region_density_ops,
            self.op_collections.fence_region_density_merged_op,
            self.op_collections.fence_region_density_overflow_merged_op,
        )

    def build_macro_overlap(self, params, placedb, data_collections):
        """
        @brief MFP macro overlap
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        return macro_overlap.MacroOverlap(
            fp_info=data_collections.fp_info,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            movable_macro_mask=data_collections.movable_macro_mask,
        )

    def initialize_macro_overlap_weight(self, params, placedb):
        # with torch.no_grad():
        #     wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])
        #     macro_overlap = self.op_collections.macro_overlap_op(
        #         self.data_collections.pos[0]
        #     )

        # ratio = wirelength / macro_overlap
        ratio = 1.0
        self.macro_overlap_weight = torch.tensor(
            [params.macro_overlap_weight * ratio],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device,
        )

        return self.macro_overlap_weight

    def build_update_macro_overlap_weight(self, params, placedb, algo="static"):
        ref_hpwl = params.RePlAce_ref_hpwl / params.scale_factor
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF

        assert algo in {"static", "dynamic"}, logging.error(
            "macro overlap weight update only supports static and dynamic modes"
        )

        def get_coeff(scaled_diff_hpwl, cofmax, cofmin):
            mu = cofmax * torch.pow(cofmax, 1.0 - scaled_diff_hpwl)
            return torch.clamp(mu, min=cofmin, max=cofmax)

        def update_macro_overlap_weight_op_dynamic(cur_metric, prev_metric, iteration):
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                mu = get_coeff(delta_hpwl / ref_hpwl, UPPER_PCOF, LOWER_PCOF)
                self.macro_overlap_weight *= mu

        def update_macro_overlap_weight_op_static(cur_metric, prev_metric, iteration):
            self.macro_overlap_weight *= params.macro_overlap_mult_weight

        update_macro_overlap_weight_op = {
            "static": update_macro_overlap_weight_op_static,
            "dynamic": update_macro_overlap_weight_op_dynamic,
        }[algo]

        return update_macro_overlap_weight_op

    def build_macro_refinement(self, params, placedb, data_collections, hpwl_op):
        """
        @brief macro orientation refinement
        """
        return macro_refinement.MacroRefinement(
            node_orient=placedb.node_orient,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            hpwl_op=hpwl_op,
            hpwl_op_net_mask=data_collections.net_mask_all,
        )
