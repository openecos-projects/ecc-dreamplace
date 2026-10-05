##
# @file   NonLinearPlace.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Nonlinear placement engine to be called with parameters and placement database
#

import os
import sys
import time
import pickle
from typing import Any
import numpy as np
import logging
import torch
import copy
import matplotlib.pyplot as plt
import inspect

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.BasicPlace as BasicPlace
import dreamplace.PlaceObj as PlaceObj
import dreamplace.NesterovAcceleratedGradientOptimizer as NesterovAcceleratedGradientOptimizer
import dreamplace.EvalMetrics as EvalMetrics
import math

from dreamplace.ops.routability.profile_timing import l_shape_log_verbose, profile_scope
from dreamplace.ops.routability import gpugr_context
from dreamplace.ops.routability import route_map_utils
from dreamplace.ops.routability.l_shape_inputs import (
    load_l_shape_topology_pack_from_gpugr as _load_l_shape_topology_pack_from_gpugr,
    prepare_l_shape_inputs_from_egr as _prepare_l_shape_inputs_from_egr,
    prepare_l_shape_inputs_from_gpugr as _prepare_l_shape_inputs_from_gpugr,
    resolve_l_directions_for_l_shape as _resolve_l_directions_for_l_shape,
    should_skip_resolver_l_direction_for_soft as _should_skip_resolver_l_direction_for_soft,
)
from dreamplace.ops.routability import route_evaluation
from dreamplace.ops.routability.l_shape_policy import LShapePolicy
from dreamplace.ops.routability.l_shape_electric_potential import (
    compute_fixed_macro_overlap_stats,
    compute_movable_displacement_stats,
)
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
    use_ggr_l_shape_topology,
    validate_ggr_l_shape_topology_params,
)
from dreamplace.ops.routability.routability_controller import RoutabilityController
from dreamplace.ops.irt_egr.egr_padding import apply_egr_padding, restore_egr_padding
from dreamplace.ops.routability import inflation_legalization


def _log_inflation_macro_overlap(
    placedb,
    pos,
    data_collections,
    round_idx,
    stage,
    reference_pos=None,
):
    try:
        overlap = compute_fixed_macro_overlap_stats(
            pos,
            data_collections.node_size_x,
            data_collections.node_size_y,
            placedb,
        )
        displacement = (
            compute_movable_displacement_stats(reference_pos, pos, placedb)
            if reference_pos is not None
            else {}
        )
        logging.info(
            "Inflation legalization telemetry: round=%d stage=%s "
            "macro_overlap_cells=%d macro_overlap_pairs=%d "
            "macro_overlap_area=%.6E macro_overlap_area_ratio=%.6E "
            "moved=%d displacement_max=%.6E displacement_mean=%.6E",
            int(round_idx),
            stage,
            int(overlap.get("fixed_macro_overlap_cell_count", 0)),
            int(overlap.get("fixed_macro_overlap_pair_count", 0)),
            float(overlap.get("fixed_macro_overlap_area", 0.0)),
            float(overlap.get("fixed_macro_overlap_area_ratio", 0.0)),
            int(displacement.get("movable_displacement_moved_count", 0)),
            float(displacement.get("movable_displacement_max", 0.0)),
            float(displacement.get("movable_displacement_mean", 0.0)),
        )
    except Exception as error:
        logging.warning(
            "Failed to compute inflation legalization telemetry for round %d "
            "at stage %s: %s",
            int(round_idx),
            stage,
            error,
        )


class NonLinearPlace(BasicPlace.BasicPlace):
    """
    @brief Nonlinear placement engine.
    It takes parameters and placement database and runs placement flow.
    """

    def __init__(self, params, placedb):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        super(NonLinearPlace, self).__init__(params, placedb)
        self._egr_padding_state = None
        self._routability_controller = None
        self._routability_model = None

    def _apply_egr_padding(self, params, placedb):
        if not getattr(params, "egr_padding_flag", 0):
            return
        if self._egr_padding_state is not None:
            logging.warning("EGR padding is already active; skip duplicate apply")
            return

        congestion_map_op = getattr(
            self.op_collections, "irt_egr_congestion_map_op", None
        )
        if congestion_map_op is None:
            raise RuntimeError("EGR padding requested but iRT EGR op was not built")

        tt = time.time()
        with torch.no_grad():
            try:
                route_map = congestion_map_op(
                    self.pos[0], stage="egr3D", resolve_congestion="high"
                )
            except Exception as exc:
                raise RuntimeError(
                    "EGR padding congestion map generation failed"
                ) from exc

            self._egr_padding_state = apply_egr_padding(
                placedb=placedb,
                pos=self.pos[0],
                node_size_x=self.data_collections.node_size_x,
                node_size_y=self.data_collections.node_size_y,
                pin_offset_x=self.data_collections.pin_offset_x,
                pin2node_map=self.data_collections.pin2node_map,
                movable_macro_mask=self.data_collections.movable_macro_mask,
                route_map=route_map,
            )

        if self._egr_padding_state is None:
            logging.info("EGR padding selected no cells")
            return

        state = self._egr_padding_state
        logging.info(
            "EGR padding applied: selected %d/%d cells, threshold %.4g, "
            "max_congestion %.4g, padding_area %.6g, "
            "padding_area_ratio_movable %.6g, elapsed %.3fs",
            state.num_selected,
            placedb.num_movable_nodes,
            state.threshold,
            state.max_congestion,
            state.padding_area,
            state.padding_area_ratio,
            time.time() - tt,
        )

    def _restore_egr_padding(self):
        state = self._egr_padding_state
        if state is None:
            return
        with torch.no_grad():
            restore_egr_padding(
                state,
                self.pos[0],
                self.data_collections.node_size_x,
                self.data_collections.pin_offset_x,
            )
        logging.info(
            "EGR padding restored: selected %d cells, threshold %.4g",
            state.num_selected,
            state.threshold,
        )
        self._egr_padding_state = None

    def _run_standard_legalization(self, params, placedb, iteration, all_metrics):
        controller = self._routability_controller
        if controller is not None:
            controller.before_legalization(model=self._routability_model, position=self.pos[0])
        tt = time.time()
        self.pos[0].data.copy_(self.op_collections.legalize_op(self.pos[0]))
        logging.info("legalization takes %.3f seconds" % (time.time() - tt))
        cur_metric = EvalMetrics.EvalMetrics(iteration)
        all_metrics.append(cur_metric)
        cur_metric.evaluate(
            placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
        )
        logging.info(cur_metric)
        if controller is not None:
            controller.after_legalization(model=self._routability_model, position=self.pos[0])
        return iteration + 1

    def __call__(self, params, placedb):
        """
        @brief Top API to solve placement.
        @param params parameters
        @param placedb placement database
        """
        iteration = 0
        all_metrics = []
        self._routability_controller = RoutabilityController(params)
        self._routability_model = None
        routability_controller = self._routability_controller
        inflation_legalization.validate_params(params)
        validate_ggr_l_shape_topology_params(params)
        if params.timing_opt_flag:
            timing_op = self.op_collections.timing_op
            time_unit = timing_op.timer.time_unit()

        # global placement
        if params.global_place_flag:

            global_place_stages = params.global_place_stages
            # macro place use external 1 stage to place macros
            if params.macro_place_flag:
                first_place_params = global_place_stages[0]
                if params.two_stage_flag:
                    global_place_stages.insert(0, first_place_params)
                macro_placed = False

                # add macro halo
                # if params.macro_halo_x > 0 or params.macro_halo_y > 0:
                #     with torch.no_grad():
                #         movable_macro_mask = self.data_collections.movable_macro_mask
                #         movable_macro_pins = self.data_collections.movable_macro_pins
                #         # node sizes
                #         self.data_collections.node_size_x[: placedb.num_movable_nodes][movable_macro_mask] += (
                #             2 * params.macro_halo_x)
                #         self.data_collections.node_size_y[: placedb.num_movable_nodes][movable_macro_mask] += (
                #             2 * params.macro_halo_y)
                #         # pin offsets
                #         self.data_collections.pin_offset_x[movable_macro_pins] += params.macro_halo_x
                #         self.data_collections.pin_offset_y[movable_macro_pins] += params.macro_halo_y
                #         # macro locations
                #         self.pos[0][: placedb.num_movable_nodes][movable_macro_mask] -= params.macro_halo_x
                #         self.pos[0][placedb.num_nodes: placedb.num_nodes +
                #                     placedb.num_movable_nodes][movable_macro_mask] -= params.macro_halo_y

            # global placement may run in multiple stages according to user specification
            for cur_stage, global_place_params in enumerate(global_place_stages):

                # we formulate each stage as a 3-nested optimization problem
                # f_gamma(g_density(h(x) ; density weight) ; gamma)
                # Lgamma      Llambda        Lsub
                # When optimizing an inner problem, the outer parameters are fixed.
                # This is a generalization to the eplace/RePlAce approach

                # As global placement may easily diverge, we record the position of best overflow
                best_metric = [None]
                best_pos = [None]

                if params.gpu:
                    torch.cuda.synchronize()
                tt = time.time()
                # construct model and optimizer
                density_weight = 0.0
                if params.macro_place_flag and cur_stage == 1:
                    density_weight = all_metrics[-1][-1][-1].density_weight.item(
                    ) / params.two_stage_density_scaler
                    # at the 2nd stage, total_movable_node_area should exclude movable macro area to enable more aggresive spreading of cells
                    placedb.total_movable_node_area = placedb.total_movable_cell_area
                # construct placement model
                model = PlaceObj.PlaceObj(
                    density_weight,
                    params,
                    placedb,
                    self.data_collections,
                    self.op_collections,
                    global_place_params,
                ).to(self.data_collections.pos[0].device)
                model.compile()
                self._routability_model = model
                routability_controller.before_stage(model=model, stage_idx=cur_stage)
                if params.routability_opt_flag:
                    inflation_state = routability_controller.inflation.ensure_inflation_state(
                        params,
                        placedb,
                        self.data_collections,
                        stage_idx=cur_stage,
                    )
                    model.inflation_state = inflation_state
                    if getattr(params, "enhanced_inflation_flag", False):
                        logging.info(
                            "Initialized enhanced inflation controller for stage %d; "
                            "inflation rounds will restore best_pos before the route-driven "
                            "outer-loop area adjust.",
                            cur_stage,
                        )
                else:
                    model.inflation_state = None

                if params.macro_place_flag and macro_placed:
                    movable_macro_mask = self.data_collections.movable_macro_mask
                    model.fix_nodes_mask = movable_macro_mask.new_zeros(
                        placedb.num_nodes)
                    model.fix_nodes_mask[placedb.num_movable_nodes:placedb.num_physical_nodes] = 1
                    model.fix_nodes_mask[:placedb.num_movable_nodes] = movable_macro_mask[:placedb.num_movable_nodes]
                    # params.use_bb = False
                    # pdb.set_trace()

                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd_momentum":
                    optimizer = torch.optim.SGD(
                        self.parameters(), lr=0, momentum=0.9, nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov":
                    optimizer = torch.optim.SGD(
                        self.parameters(), lr=0, momentum=0.9, nesterov=True)
                elif optimizer_name.lower() == "nesterov":
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(
                        self.parameters(),
                        lr=0,
                        obj_and_grad_fn=model.obj_and_grad_fn,
                        constraint_fn=self.op_collections.move_boundary_op,
                        use_bb=params.use_bb
                    )
                else:
                    assert 0, "unknown optimizer %s" % (optimizer_name)

                logging.info("use %s optimizer" % (optimizer_name))
                model.train()
                # defining evaluation ops
                eval_ops = {
                    # "wirelength" : self.op_collections.wirelength_op,
                    # "density" : self.op_collections.density_op,
                    # "objective" : model.obj_fn,
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op,
                }
                # [DISABLED] route_overflow and pin_overflow computation disabled
                # if params.routability_opt_flag:
                #     eval_ops.update(
                #         {
                #             "route_utilization": self.op_collections.route_utilization_map_op,
                #             "pin_utilization": self.op_collections.pin_utilization_map_op,
                #         }
                #     )
                if len(placedb.regions) > 0:
                    eval_ops.update(
                        {
                            "density": self.op_collections.fence_region_density_merged_op,
                            "overflow": self.op_collections.fence_region_density_overflow_merged_op,
                            "goverflow": self.op_collections.density_overflow_op,
                        }
                    )

                # a function to initialize learning rate
                def initialize_learning_rate(pos):
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)
                    learning_rate = model.estimate_initial_learning_rate(
                        pos, global_place_params["learning_rate"]
                    )
                    # update learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate.data

                if iteration == 0 or (params.macro_place_flag and cur_stage == 1):
                    if iteration == 0 and params.gp_noise_ratio > 0.0:
                        logging.info("add %g%% noise" %
                                     (params.gp_noise_ratio * 100))
                        model.op_collections.noise_op(
                            model.data_collections.pos[0], params.gp_noise_ratio)
                    initialize_learning_rate(model.data_collections.pos[0])
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if params.gpu:
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" %
                             (optimizer_name, (time.time() - tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov"}:
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def Lgamma_stop_criterion(Lgamma_step, metrics, stop_mask=None):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-2][-1][-1]
                            # update stop mask for each fence region
                            # if(stop_mask is not None):
                            #     stop_mask.copy_(cur_metric.overflow < params.stop_overflow)

                            if Lgamma_step > 100 and (
                                # for fence region, the outer cell overflow decides the stopping of GP
                                (
                                    cur_metric.overflow[-1] < params.stop_overflow
                                    and cur_metric.hpwl > prev_metric.hpwl
                                )
                                or cur_metric.max_density[-1] < params.target_density
                            ):
                                logging.debug(
                                    "Lgamma stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Lgamma_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                            if len(placedb.regions) > 0 and model.update_mask.sum() == 0:
                                logging.debug(
                                    "All regions stop updating, finish global placement")
                                return True
                        # a heuristic to detect divergence and stop early
                        if len(metrics) > 50:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-50][-1][-1]
                            # record HPWL and overflow increase, and check divergence
                            if (
                                cur_metric.overflow[-1] > prev_metric.overflow[-1]
                                and cur_metric.hpwl > best_metric[0].hpwl * 2
                            ):
                                return True
                        return False

                def Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1]
                            prev_metric = metrics[-2][-1]
                            # for fence regions, the outer cell overflow and max_density decides whether to stop
                            if (
                                cur_metric.overflow[-1] < params.stop_overflow
                                and cur_metric.hpwl > prev_metric.hpwl
                            ) or cur_metric.max_density[-1] < 1.0:
                                logging.debug(
                                    "Llambda stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Llambda_density_weight_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                    return False

                # use a moving average window for stopping criteria, for an example window of 3
                # 0, 1, 2, 3, 4, 5, 6
                #    window2
                #             window1
                moving_avg_window = max(min(model.Lsub_iteration // 2, 3), 1)
                l_shape_policy = LShapePolicy(params)

                def Lsub_stop_criterion(Lgamma_step, Llambda_density_weight_step, Lsub_step, metrics):
                    with torch.no_grad():
                        if len(metrics) >= moving_avg_window * 2:
                            cur_avg_obj = 0
                            prev_avg_obj = 0
                            for i in range(moving_avg_window):
                                cur_avg_obj += metrics[-1 - i].objective
                                prev_avg_obj += metrics[-1 -
                                                        moving_avg_window - i].objective
                            cur_avg_obj /= moving_avg_window
                            prev_avg_obj /= moving_avg_window
                            threshold = 0.999
                            if cur_avg_obj >= prev_avg_obj * threshold:
                                logging.debug(
                                    "Lsub stopping criteria: %d and %g > %g * %g"
                                    % (Lsub_step, cur_avg_obj, prev_avg_obj, threshold)
                                )
                                return True
                    return False

                def one_descent_step(
                    Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, metrics, stop_mask=None
                ):
                    t0 = time.time()
                    routability_controller.before_iteration(
                        model=model,
                        iteration=iteration,
                        position=model.data_collections.pos[0],
                    )

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(
                        iteration, (Lgamma_step,
                                    Llambda_density_weight_step, Lsub_step)
                    )
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    pos = model.data_collections.pos[0]
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)

                    # move any out-of-bound cell back to placement region
                    self.op_collections.move_boundary_op(pos)

                    # handle multiple density weights for multi-electric field
                    if torch.eq(model.density_weight.mean(), 0.0):
                        model.initialize_density_weight(params, placedb)
                        if model.density_weight.size(0) == 1:
                            logging.info("density_weight = %.6E" %
                                         (model.density_weight.data))
                        else:
                            logging.info(
                                "density_weight = [%s]"
                                % ", ".join(["%.3E" % i for i in model.density_weight.cpu().numpy().tolist()])
                            )

                    # For backward compatibility
                    # PyTorch 1.7 introduced zero_grad(set_to_none=False)
                    # PyTorch 2.0 changed set_to_none=True
                    if "set_to_none" in inspect.signature(optimizer.zero_grad).parameters:
                        optimizer.zero_grad(set_to_none=False)
                    else:
                        optimizer.zero_grad()

                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops,
                                        pos, model.data_collections)
                    model.overflow = cur_metric.overflow.data.clone()
                    # logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    # t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov"]:
                        obj, grad = model.obj_and_grad_fn(pos)
                        cur_metric.objective = obj.data.clone()
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # diff tdp
                    if params.with_sta and (iteration % 10 == 0 and iteration >= 100):
                        t_steiner = time.time()
                        with torch.no_grad():
                            pin_pos = self.op_collections.pin_pos_op(pos)
                            if pin_pos.is_cuda:
                                pin_pos = pin_pos.cpu()
                            self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(
                                    pin_pos)
                            model.use_timing_obj = True
                        logging.info("Update steiner topo %.3f ms" %
                                     ((time.time() - t_steiner) * 1000))


                    # ========== L形Routability Density Objective ==========
                    # 根据overflow条件启用L形routability
                    l_shape_routability_enabled = (
                        params.routability_opt_flag
                        and getattr(params, "l_shape_routability_flag", 0)
                    )
                    if l_shape_routability_enabled:
                        
                        l_shape_policy.maybe_reenable(
                            model,
                            float(cur_metric.overflow[-1]),
                        )
                            
                            # # 条件2: 也可以根据iteration启用
                            # if iteration >= getattr(params, 'l_shape_start_iteration', 100):
                            #     model.enable_l_shape_routability = True
                        
                        if model.enable_l_shape_routability and not model.use_l_shape_routability:
                            # 首次启用L形routability
                            t_l_shape_init = time.time()
                            
                            L_shape_num_bins_x = params.num_bins_x
                            L_shape_num_bins_y = params.num_bins_y

                            ggr_topology_mode = use_ggr_l_shape_topology(params)
                            gpugr_inputs = None
                            l_shape_inputs = None
                            if ggr_topology_mode:
                                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                    params,
                                    placedb,
                                    pos,
                                    model=model,
                                )
                                l_shape_inputs = gpugr_inputs
                                l_directions = _load_l_shape_topology_pack_from_gpugr(
                                    params,
                                    pos,
                                    self.op_collections.pin_pos_op,
                                    self.op_collections.steiner_topo_op,
                                    self.data_collections,
                                    l_shape_inputs,
                                    "l_shape_init.load_ggr_topology_pack",
                                    iteration,
                                )
                            else:
                                with profile_scope(params, "l_shape_init.rebuild_tree", tensor=pos, iteration=iteration):
                                    with torch.no_grad():
                                        pin_pos = self.op_collections.pin_pos_op(pos)
                                        if pin_pos.is_cuda:
                                            pin_pos = pin_pos.cpu()
                                        self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                            self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                            self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(pin_pos)
                                if getattr(params, "l_direction_use_gpugr", False):
                                    gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                    l_shape_inputs = gpugr_inputs
                                else:
                                    l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )

                            supply_map = l_shape_inputs["supply_map"]
                            demand_map = l_shape_inputs["demand_map"]
                            wire_width = l_shape_inputs["wire_width"]

                            # Resolve L directions from the selected topology source.
                            steiner_topo_op = self.op_collections.steiner_topo_op
                            if ggr_topology_mode:
                                l_directions = steiner_topo_op.edge_l_directions
                            elif _should_skip_resolver_l_direction_for_soft(params):
                                steiner_topo_op.edge_l_directions = None
                                l_directions = None
                                if l_shape_log_verbose(params) >= 2:
                                    logging.info(
                                        "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                                        "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                                        "will be refreshed."
                                    )
                            else:
                                l_directions = _resolve_l_directions_for_l_shape(
                                    params,
                                    placedb,
                                    pos,
                                    steiner_topo_op,
                                    gpugr_route_entries=(
                                        gpugr_inputs["route_entries"]
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                    gpugr_metrics=(
                                        gpugr_inputs["metrics"]
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                    gpugr_route_grid=(
                                        (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                                        if getattr(params, "l_direction_use_gpugr", False)
                                        else None
                                    ),
                                )

                            # # ========== Plot edges with L-shape by l_direction ==========
                            # import matplotlib.pyplot as plt
                            # import matplotlib.collections as mc
                            
                            # # 获取坐标和边信息
                            # newx = steiner_topo_op.newx.cpu().numpy()
                            # newy = steiner_topo_op.newy.cpu().numpy()
                            # flat_pin_from = self.data_collections.flat_pin_from.cpu().numpy()
                            # flat_pin_to = self.data_collections.flat_pin_to.cpu().numpy()
                            # l_dirs = l_directions.cpu().numpy()
                            
                            # # 颜色映射: H_FIRST=0(红), V_FIRST=1(蓝), STRAIGHT=2(绿), FAKE_STRAIGHT=3(橙)
                            # color_map = {
                            #     0: 'red',      # H_FIRST: 先水平后垂直
                            #     1: 'blue',     # V_FIRST: 先垂直后水平
                            #     2: 'green',    # STRAIGHT: 直线
                            #     3: 'orange'    # FAKE_STRAIGHT: 伪直线
                            # }
                            # label_map = {
                            #     0: 'H_FIRST (H→V)',
                            #     1: 'V_FIRST (V→H)',
                            #     2: 'STRAIGHT',
                            #     3: 'FAKE_STRAIGHT'
                            # }
                            
                            # # 按l_direction分组收集线段
                            # # L形边变成两段，直线保持一段
                            # edges_by_dir = {0: [], 1: [], 2: [], 3: []}
                            # edge_count_by_dir = {0: 0, 1: 0, 2: 0, 3: 0}
                            
                            # for i in range(len(l_dirs)):
                            #     from_idx = flat_pin_from[i]
                            #     to_idx = flat_pin_to[i]
                            #     if from_idx == -1 or to_idx == -1:
                            #         continue
                            #     x1, y1 = newx[from_idx], newy[from_idx]
                            #     x2, y2 = newx[to_idx], newy[to_idx]
                            #     direction = int(l_dirs[i])
                            #     if direction not in edges_by_dir:
                            #         direction = 1  # 默认用V_FIRST
                                
                            #     edge_count_by_dir[direction] += 1
                                
                            #     # 根据方向生成路径
                            #     if direction == 0:  # H_FIRST: 水平优先 (x1,y1) -> (x2,y1) -> (x2,y2)
                            #         corner = (x2, y1)
                            #         edges_by_dir[direction].append([(x1, y1), corner])
                            #         edges_by_dir[direction].append([corner, (x2, y2)])
                            #     elif direction == 2:  # STRAIGHT: 直线
                            #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
                            #     elif direction == 3:  # FAKE_STRAIGHT: 伪直线（画成直线）
                            #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
                            #     elif direction == 1:  # V_FIRST: 垂直优先 (x1,y1) -> (x1,y2) -> (x2,y2)
                            #         # (x1,y1) -> (x1,y2) -> (x2,y2)
                            #         corner = (x1, y2)
                            #         edges_by_dir[direction].append([(x1, y1), corner])
                            #         edges_by_dir[direction].append([corner, (x2, y2)])
                            
                            # # 绘图
                            # fig, ax = plt.subplots(figsize=(12, 10))
                            
                            # for direction, edges in edges_by_dir.items():
                            #     if len(edges) > 0:
                            #         lc = mc.LineCollection(edges, colors=color_map[direction], 
                            #                               linewidths=0.5, alpha=0.7,
                            #                               label=f'{label_map[direction]} ({edge_count_by_dir[direction]})')
                            #         ax.add_collection(lc)
                            
                            # ax.autoscale()
                            # ax.set_aspect('equal')
                            # ax.set_xlabel('X')
                            # ax.set_ylabel('Y')
                            # ax.set_title('Steiner Tree L-Shape Edges')
                            # ax.legend(loc='upper right')
                            
                            # # 保存图片
                            # plot_path = os.path.join(params.result_dir, 'l_direction_edges.png')
                            # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                            # plt.close()
                            # logging.info(f"L-direction edge plot saved to {plot_path}")
                            
                            # exit(0)
                            
                            # Initialize the L-shape routability operator.

                            with profile_scope(params, "l_shape_init.construct_op", tensor=pos, iteration=iteration):
                                model.init_l_shape_routability(
                                    wire_width=wire_width,
                                    num_bins_x=L_shape_num_bins_x,
                                    num_bins_y=L_shape_num_bins_y,
                                    target_density=supply_map,
                                    target_demand=demand_map,
                                    raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                                    supply_original=l_shape_inputs.get("supply_original"),
                                    target_density_h=l_shape_inputs.get("supply_map_h"),
                                    target_density_v=l_shape_inputs.get("supply_map_v"),
                                    target_demand_h=l_shape_inputs.get("demand_map_h"),
                                    target_demand_v=l_shape_inputs.get("demand_map_v"),
                                    raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                                    raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                                    supply_original_h=l_shape_inputs.get("supply_original_h"),
                                    supply_original_v=l_shape_inputs.get("supply_original_v"),
                                    fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                                    fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                                    fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
                                )
                            if model.l_shape_routability_op is not None:
                                model.l_shape_routability_op.update_same_net_topology(
                                    topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                                    topo_stats=l_shape_inputs.get("same_net_topo_stats"),
                                )
                            if hasattr(model, "start_l_shape_weight_controller"):
                                model.start_l_shape_weight_controller(iteration)
                            # 初始化基于L-shape overflow的外环状态
                            l_shape_policy.reset_overflow_state(model)
                            
                            if l_shape_log_verbose(params) >= 1:
                                logging.info(f"L-shape routability enabled at iteration {iteration}, "
                                            f"overflow={cur_metric.overflow[-1]:.4f}, "
                                            f"threshold={float(getattr(model, '_l_shape_reenable_threshold', getattr(params, 'l_shape_overflow_threshold', 0.2))):.4f}, "
                                            f"descend_streak={int(getattr(model, '_l_shape_reenable_descend_streak', 0))}, "
                                            f"init time={((time.time() - t_l_shape_init) * 1000):.2f}ms")
                            l_shape_policy.reset_reenable_progress(model)
                            
                            # ========== 梯度正确性检查 (可选) ==========
                            if getattr(params, 'l_shape_gradient_check', False):
                                # 1. 运行梯度链诊断
                                logging.info("Running L-shape gradient chain diagnosis...")
                                model.diagnose_l_shape_gradient_chain(pos)
                                
                                # 2. 运行梯度问题诊断（检查边界问题）
                                logging.info("Running L-shape gradient issues diagnosis...")
                                model.diagnose_l_shape_gradient_issues(pos)
                                
                                # 3. 运行梯度方向检查（更实用）
                                logging.info("Running L-shape gradient direction check...")
                                direction_results = model.check_l_shape_gradient_direction(
                                    pos, 
                                    step_sizes=[0.1, 1.0, 10.0, 100.0]
                                )
                                
                                # 4. 可选：运行数值梯度检查（对bin-based函数可能失败）
                                logging.info("Running L-shape gradient numerical check...")
                                logging.info("Note: Numerical check may fail for bin-based density functions")
                                grad_check_results = model.check_l_shape_gradient_numerical(
                                    pos, 
                                    num_check=200,  # 检查200个位置
                                    eps=1e-3,       # 有限差分步长
                                    check_movable_only=True,
                                    verbose=True
                                )
                                
                                # 保存结果到文件
                                import json
                                results_to_save = {
                                    'direction_check': direction_results,
                                    'numerical_check': grad_check_results
                                }
                                grad_check_path = os.path.join(params.result_dir, "l_shape_grad_check.json")
                                with open(grad_check_path, 'w') as f:
                                    json.dump(results_to_save, f, indent=2)
                                logging.info(f"Gradient check results saved to {grad_check_path}")

                                exit(0)
                            # =============================================
                            
                            # 可视化L形密度图和segments
                            if params.l_shape_plot_flag:
                                try:
                                    from dreamplace.ops.routability.l_shape_routability import (
                                        plot_l_shape_electric_overflow_map,
                                        plot_l_shape_initial_density_map,
                                        plot_l_shape_macro_source_maps,
                                        plot_l_shape_electric_potential_map,
                                        plot_l_shape_supply_maps,
                                        plot_l_shape_true_source_maps,
                                        plot_segment_density_map,
                                        plot_soft_l_intermediate,
                                        plot_soft_l_scoring_maps,
                                    )
                                    
                                    # 获取密度图
                                    forward_source_snapshot = model.snapshot_l_shape_forward_state()
                                    density_map = model.get_l_shape_density_map(pos, use_l_direction=True)
                                    model.restore_l_shape_forward_state(forward_source_snapshot)
                                    if density_map is not None:
                                        density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_density_iter{iteration}.png"
                                        )
                                        plot_segment_density_map(
                                            density_map, density_plot_path,
                                            title=f"L-shape Density (iter={iteration})",
                                            colormap="binary"
                                        )
                                        logging.info(f"L-shape density plot saved to {density_plot_path}")
                                    
                                    # 绘制基于 electric potential 的 overflow map
                                    if model.l_shape_routability_op is not None and \
                                       model.l_shape_routability_op.cached_segments is not None:
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_overflow_map(
                                            l_shape_op,
                                            output_path=overflow_plot_path,
                                            title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                                        )
                                        logging.info(f"L-shape electric overflow plot saved to {overflow_plot_path}")
                                        supply_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_supply_iter{iteration}.png"
                                        )
                                        plot_l_shape_supply_maps(
                                            l_shape_op,
                                            output_path=supply_plot_path,
                                            title_prefix=f"L-shape Supply Debug (iter={iteration})",
                                        )
                                        logging.info(f"L-shape supply debug plot saved to {supply_plot_path}")
                                        initial_density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                                        )
                                        plot_l_shape_initial_density_map(
                                            l_shape_op,
                                            output_path=initial_density_plot_path,
                                            title_prefix=f"L-shape Initial Density (iter={iteration})",
                                        )
                                        logging.info(
                                            f"L-shape initial density plot saved to {initial_density_plot_path}"
                                        )
                                        source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_true_source_maps(
                                            l_shape_op,
                                            output_path=source_plot_path,
                                            title_prefix=f"L-shape True Source (iter={iteration})",
                                        )
                                        logging.info(f"L-shape true source plot saved to {source_plot_path}")
                                        macro_source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_macro_source_maps(
                                            l_shape_op,
                                            output_path=macro_source_plot_path,
                                            title_prefix=f"L-shape Macro Source (iter={iteration})",
                                        )
                                        logging.info(
                                            f"L-shape macro source plot saved to {macro_source_plot_path}"
                                        )
                                        potential_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_potential_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_potential_map(
                                            l_shape_op,
                                            output_path=potential_plot_path,
                                            title_prefix=f"L-shape Electric Potential (iter={iteration})",
                                        )
                                        logging.info(f"L-shape electric potential plot saved to {potential_plot_path}")
                                        if getattr(l_shape_op, "soft_l_assignment", False) and \
                                           'soft_l_weights' in l_shape_op.cached_segments:
                                            soft_plot_path = os.path.join(
                                                params.result_dir, f"l_shape_soft_iter{iteration}.png"
                                            )
                                            plot_soft_l_intermediate(
                                                l_shape_op.cached_segments,
                                                output_path=soft_plot_path,
                                            )
                                            logging.info(f"Soft L-shape plot saved to {soft_plot_path}")
                                            if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                                                soft_scoring_path = os.path.join(
                                                    params.result_dir,
                                                    f"l_shape_soft_scoring_iter{iteration}.png",
                                                )
                                                plot_soft_l_scoring_maps(
                                                    l_shape_op.cached_soft_debug,
                                                    output_path=soft_scoring_path,
                                                    title_prefix=f"Soft L Scoring (iter={iteration})",
                                                )
                                                logging.info(
                                                    f"Soft L-shape scoring plot saved to {soft_scoring_path}"
                                                )
                                except Exception as e:
                                    logging.warning(f"Failed to plot L-shape density/segments: {e}")
                                
                            # exit(0)
                        # 定期更新Steiner树和L方向（每N次迭代）
                        elif model.use_l_shape_routability and (iteration % params.l_shape_update_interval == 0):
                            t_l_shape_update = time.time()
                            
                            # 重置L形segment缓存（EGR将重新运行）
                            if model.l_shape_routability_op is not None:
                                model.l_shape_routability_op.segment_builder.reset_cache()
                            
                            ggr_topology_mode = use_ggr_l_shape_topology(params)
                            gpugr_inputs = None
                            l_shape_inputs = None
                            if ggr_topology_mode:
                                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                    params,
                                    placedb,
                                    pos,
                                    model=model,
                                )
                                l_shape_inputs = gpugr_inputs
                                l_directions = _load_l_shape_topology_pack_from_gpugr(
                                    params,
                                    pos,
                                    self.op_collections.pin_pos_op,
                                    self.op_collections.steiner_topo_op,
                                    self.data_collections,
                                    l_shape_inputs,
                                    "l_shape_update.load_ggr_topology_pack",
                                    iteration,
                                )
                            else:
                                with profile_scope(params, "l_shape_update.rebuild_tree", tensor=pos, iteration=iteration):
                                    with torch.no_grad():
                                        pin_pos = self.op_collections.pin_pos_op(pos)
                                        if pin_pos.is_cuda:
                                            pin_pos = pin_pos.cpu()
                                        self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                                            self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                                            self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(pin_pos)
                                if not getattr(params, "l_direction_use_gpugr", False):
                                    l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                else:
                                    gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                                        params,
                                        placedb,
                                        pos,
                                        model=model,
                                    )
                                    l_shape_inputs = gpugr_inputs

                            steiner_topo_op = self.op_collections.steiner_topo_op
                            if ggr_topology_mode:
                                l_directions = steiner_topo_op.edge_l_directions
                            elif _should_skip_resolver_l_direction_for_soft(params):
                                steiner_topo_op.edge_l_directions = None
                                l_directions = None
                                if l_shape_log_verbose(params) >= 2:
                                    logging.info(
                                        "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                                        "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                                        "will be refreshed."
                                    )
                            else:
                                l_directions = _resolve_l_directions_for_l_shape(
                                    params,
                                    placedb,
                                    pos,
                                    steiner_topo_op,
                                    gpugr_route_entries=(
                                        gpugr_inputs["route_entries"]
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                    gpugr_metrics=(
                                        gpugr_inputs["metrics"]
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                    gpugr_route_grid=(
                                        (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                                        if gpugr_inputs is not None
                                        else None
                                    ),
                                )
                            if model.l_shape_routability_op is not None and l_shape_inputs is not None:
                                with profile_scope(params, "l_shape_update.update_targets", tensor=pos, iteration=iteration):
                                    model.l_shape_routability_op.update_targets(
                                        target_density=l_shape_inputs["supply_map"],
                                        target_demand=l_shape_inputs["demand_map"],
                                        raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                                        target_density_h=l_shape_inputs.get("supply_map_h"),
                                        target_density_v=l_shape_inputs.get("supply_map_v"),
                                        target_demand_h=l_shape_inputs.get("demand_map_h"),
                                        target_demand_v=l_shape_inputs.get("demand_map_v"),
                                        raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                                        raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                                        supply_original=l_shape_inputs.get("supply_original"),
                                        supply_original_h=l_shape_inputs.get("supply_original_h"),
                                        supply_original_v=l_shape_inputs.get("supply_original_v"),
                                        fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                                        fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                                        fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
                                    )
                                    model.l_shape_routability_op.update_same_net_topology(
                                        topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                                        topo_stats=l_shape_inputs.get("same_net_topo_stats"),
                                    )
                                if (
                                    getattr(model.l_shape_routability_op, "wire_width_h", None) is None
                                    and getattr(model.l_shape_routability_op, "wire_width_v", None) is None
                                ):
                                    updated_wire_width = float(l_shape_inputs["wire_width"])
                                    current_wire_width = float(model.l_shape_routability_op.wire_width)
                                    if abs(updated_wire_width - current_wire_width) > 1e-6:
                                        logging.info(
                                            "L-shape periodic target update kept existing wire_width %.4f while refreshed maps imply %.4f. "
                                            "Segment width is not updated online after initialization.",
                                            current_wire_width,
                                            updated_wire_width,
                                        )
                            density_map = None

                            # 基于L-shape overflow变化更新target_ratio（外环慢速更新）
                            if getattr(params, 'l_shape_overflow_update_flag', True):
                                try:
                                    with torch.no_grad():
                                        density_map = model.get_l_shape_density_map(
                                            pos, use_l_direction=True
                                        )
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_op = (
                                            getattr(l_shape_op, "overflow_op", None)
                                            if l_shape_op is not None
                                            else None
                                        )
                                        if density_map is not None and overflow_op is not None:
                                            l_shape_overflow = None
                                            overflow_ratio = None
                                            l_shape_max_density = None

                                            # 优先使用potential中的当前场源口径：
                                            # blockage_initial_density 主线走 track-space，
                                            # legacy residual 路径仍走 tracks + area_per_track。
                                            density_driver = getattr(l_shape_op, "density_op", None)
                                            if density_driver is not None:
                                                supply_map = getattr(
                                                    density_driver, "target_density", None
                                                )
                                                supply_map_h = getattr(
                                                    density_driver, "target_density_h", None
                                                )
                                                supply_map_v = getattr(
                                                    density_driver, "target_density_v", None
                                                )
                                                demand_map = getattr(
                                                    density_driver, "target_demand", None
                                                )
                                                demand_map_h = getattr(
                                                    density_driver, "target_demand_h", None
                                                )
                                                demand_map_v = getattr(
                                                    density_driver, "target_demand_v", None
                                                )
                                                density_map_h = getattr(
                                                    l_shape_op, "cached_density_map_h", None
                                                )
                                                density_map_v = getattr(
                                                    l_shape_op, "cached_density_map_v", None
                                                )
                                                if (
                                                    isinstance(supply_map, torch.Tensor)
                                                    and supply_map.dim() == 2
                                                ):
                                                    supply_map = supply_map.to(
                                                        density_map.device,
                                                        dtype=density_map.dtype,
                                                    )
                                                    blockage_initial_density = bool(
                                                        getattr(density_driver, "blockage_initial_density", False)
                                                    )
                                                    supply_original_map = getattr(
                                                        density_driver, "supply_original", None
                                                    )
                                                    supply_original_map_h = getattr(
                                                        density_driver, "supply_original_h", None
                                                    )
                                                    supply_original_map_v = getattr(
                                                        density_driver, "supply_original_v", None
                                                    )
                                                    fix_usage_map = getattr(
                                                        density_driver, "fix_usage_map", None
                                                    )
                                                    fix_usage_map_h = getattr(
                                                        density_driver, "fix_usage_map_h", None
                                                    )
                                                    fix_usage_map_v = getattr(
                                                        density_driver, "fix_usage_map_v", None
                                                    )

                                                    if (
                                                        blockage_initial_density
                                                        and isinstance(supply_original_map, torch.Tensor)
                                                        and isinstance(fix_usage_map, torch.Tensor)
                                                    ):
                                                        bin_area_local = float(
                                                            density_driver.bin_size_x
                                                            * density_driver.bin_size_y
                                                        )
                                                        split_available = (
                                                            isinstance(density_map_h, torch.Tensor)
                                                            and isinstance(density_map_v, torch.Tensor)
                                                            and isinstance(supply_original_map_h, torch.Tensor)
                                                            and isinstance(supply_original_map_v, torch.Tensor)
                                                            and isinstance(fix_usage_map_h, torch.Tensor)
                                                            and isinstance(fix_usage_map_v, torch.Tensor)
                                                        )
                                                        if split_available:
                                                            density_map_h = density_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_map_v = density_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            supply_original_map_h = supply_original_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            supply_original_map_v = supply_original_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map_h = fix_usage_map_h.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map_v = fix_usage_map_v.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_h_in_tracks = density_map_h / bin_area_local
                                                            density_v_in_tracks = density_map_v / bin_area_local
                                                            occupancy_h = density_h_in_tracks + fix_usage_map_h.clamp(min=0.0)
                                                            occupancy_v = density_v_in_tracks + fix_usage_map_v.clamp(min=0.0)
                                                            overflow_h_in_tracks = (
                                                                occupancy_h - supply_original_map_h
                                                            ).clamp(min=0.0)
                                                            overflow_v_in_tracks = (
                                                                occupancy_v - supply_original_map_v
                                                            ).clamp(min=0.0)
                                                            utilization_h = occupancy_h / supply_original_map_h.clamp(
                                                                min=1e-6
                                                            )
                                                            utilization_v = occupancy_v / supply_original_map_v.clamp(
                                                                min=1e-6
                                                            )
                                                            l_shape_overflow = float(
                                                                (
                                                                    overflow_h_in_tracks.sum()
                                                                    + overflow_v_in_tracks.sum()
                                                                ).item()
                                                            )
                                                            overflow_ratio = float(
                                                                (
                                                                    overflow_h_in_tracks.sum()
                                                                    + overflow_v_in_tracks.sum()
                                                                )
                                                                / (
                                                                    supply_original_map_h.sum()
                                                                    + supply_original_map_v.sum()
                                                                ).clamp(min=1e-12)
                                                            )
                                                            l_shape_max_density = float(
                                                                torch.maximum(
                                                                    utilization_h.max(),
                                                                    utilization_v.max(),
                                                                ).item()
                                                            )
                                                        else:
                                                            supply_original_map = supply_original_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            fix_usage_map = fix_usage_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            density_in_tracks = density_map / bin_area_local
                                                            occupancy = density_in_tracks + fix_usage_map.clamp(min=0.0)
                                                            overflow_in_tracks = (
                                                                occupancy - supply_original_map
                                                            ).clamp(min=0.0)
                                                            utilization = occupancy / supply_original_map.clamp(
                                                                min=1e-6
                                                            )
                                                            l_shape_overflow = float(
                                                                overflow_in_tracks.sum().item()
                                                            )
                                                            overflow_ratio = float(
                                                                (
                                                                    overflow_in_tracks.sum()
                                                                    / supply_original_map.sum().clamp(min=1e-12)
                                                                ).item()
                                                            )
                                                            l_shape_max_density = float(
                                                                utilization.max().item()
                                                            )
                                                    else:
                                                        area_per_track_buf = getattr(
                                                            density_driver, "area_per_track", None
                                                        )
                                                        total_density = density_map.sum()
                                                        calibrated_area_per_track = None

                                                        if (
                                                            isinstance(demand_map, torch.Tensor)
                                                            and demand_map.dim() == 2
                                                        ):
                                                            demand_map = demand_map.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            total_demand = demand_map.sum()
                                                            if total_demand > 0 and total_density > 0:
                                                                if (
                                                                    isinstance(
                                                                        area_per_track_buf,
                                                                        torch.Tensor,
                                                                    )
                                                                    and area_per_track_buf.numel() == 1
                                                                ):
                                                                    if (
                                                                        float(
                                                                            area_per_track_buf.item()
                                                                        )
                                                                        <= 0
                                                                    ):
                                                                        area_per_track_buf.fill_(
                                                                            total_density / total_demand
                                                                        )
                                                                    calibrated_area_per_track = area_per_track_buf.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    calibrated_area_per_track = (
                                                                        total_density / total_demand
                                                                    )
                                                        else:
                                                            target_utilization = float(
                                                                getattr(
                                                                    params,
                                                                    "l_shape_target_utilization",
                                                                    0.8,
                                                                )
                                                            )
                                                            total_supply = supply_map.sum()
                                                            if total_supply > 0 and total_density > 0:
                                                                calibrated_area_per_track = total_density / (
                                                                    target_utilization
                                                                    * total_supply.clamp(min=1e-12)
                                                                )

                                                        if calibrated_area_per_track is not None:
                                                            split_available = (
                                                                isinstance(density_map_h, torch.Tensor)
                                                                and isinstance(density_map_v, torch.Tensor)
                                                                and isinstance(supply_map_h, torch.Tensor)
                                                                and isinstance(supply_map_v, torch.Tensor)
                                                            )
                                                            if split_available:
                                                                density_map_h = density_map_h.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                density_map_v = density_map_v.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                supply_map_h = supply_map_h.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                supply_map_v = supply_map_v.to(
                                                                    density_map.device,
                                                                    dtype=density_map.dtype,
                                                                )
                                                                if (
                                                                    isinstance(demand_map_h, torch.Tensor)
                                                                    and demand_map_h.dim() == 2
                                                                ):
                                                                    demand_map_h = demand_map_h.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    demand_map_h = None
                                                                if (
                                                                    isinstance(demand_map_v, torch.Tensor)
                                                                    and demand_map_v.dim() == 2
                                                                ):
                                                                    demand_map_v = demand_map_v.to(
                                                                        density_map.device,
                                                                        dtype=density_map.dtype,
                                                                    )
                                                                else:
                                                                    demand_map_v = None

                                                                calibrated_area_per_track_h = calibrated_area_per_track
                                                                calibrated_area_per_track_v = calibrated_area_per_track
                                                                if demand_map_h is not None:
                                                                    total_density_h = density_map_h.sum()
                                                                    total_demand_h = demand_map_h.sum()
                                                                    if total_density_h > 0 and total_demand_h > 0:
                                                                        calibrated_area_per_track_h = (
                                                                            total_density_h / total_demand_h
                                                                        )
                                                                if demand_map_v is not None:
                                                                    total_density_v = density_map_v.sum()
                                                                    total_demand_v = demand_map_v.sum()
                                                                    if total_density_v > 0 and total_demand_v > 0:
                                                                        calibrated_area_per_track_v = (
                                                                            total_density_v / total_demand_v
                                                                        )

                                                                demand_h_in_tracks = (
                                                                    density_map_h
                                                                    / calibrated_area_per_track_h
                                                                )
                                                                demand_v_in_tracks = (
                                                                    density_map_v
                                                                    / calibrated_area_per_track_v
                                                                )
                                                                overflow_h_in_tracks = (
                                                                    demand_h_in_tracks - supply_map_h
                                                                ).clamp(min=0.0)
                                                                overflow_v_in_tracks = (
                                                                    demand_v_in_tracks - supply_map_v
                                                                ).clamp(min=0.0)
                                                                utilization_h = demand_h_in_tracks / supply_map_h.clamp(
                                                                    min=1e-6
                                                                )
                                                                utilization_v = demand_v_in_tracks / supply_map_v.clamp(
                                                                    min=1e-6
                                                                )
                                                                overflow_area = (
                                                                    overflow_h_in_tracks
                                                                    * calibrated_area_per_track_h
                                                                    + overflow_v_in_tracks
                                                                    * calibrated_area_per_track_v
                                                                )
                                                                l_shape_overflow = float(
                                                                    overflow_area.sum().item()
                                                                )
                                                                overflow_ratio = float(
                                                                    (
                                                                        overflow_h_in_tracks.sum()
                                                                        + overflow_v_in_tracks.sum()
                                                                    )
                                                                    / (
                                                                        supply_map_h.sum()
                                                                        + supply_map_v.sum()
                                                                    ).clamp(min=1e-12)
                                                                )
                                                                l_shape_max_density = float(
                                                                    torch.maximum(
                                                                        utilization_h.max(),
                                                                        utilization_v.max(),
                                                                    ).item()
                                                                )
                                                            else:
                                                                demand_in_tracks = (
                                                                    density_map
                                                                    / calibrated_area_per_track
                                                                )
                                                                overflow_in_tracks = (
                                                                    demand_in_tracks - supply_map
                                                                ).clamp(min=0.0)
                                                                utilization = demand_in_tracks / supply_map.clamp(
                                                                    min=1e-6
                                                                )
                                                                l_shape_overflow = float(
                                                                    (
                                                                        overflow_in_tracks
                                                                        * calibrated_area_per_track
                                                                    )
                                                                    .sum()
                                                                    .item()
                                                                )
                                                                overflow_ratio = float(
                                                                    (
                                                                        overflow_in_tracks.sum()
                                                                        / supply_map.sum().clamp(min=1e-12)
                                                                    )
                                                                    .item()
                                                                )
                                                                l_shape_max_density = float(
                                                                    utilization.max().item()
                                                                )

                                            # 若potential口径不可用，退化为overflow_op口径
                                            if (
                                                l_shape_overflow is None
                                                or overflow_ratio is None
                                                or l_shape_max_density is None
                                            ):
                                                cached_segments = getattr(
                                                    l_shape_op, "cached_segments", None
                                                )
                                                if (
                                                    cached_segments is not None
                                                    and int(
                                                        cached_segments.get(
                                                            "num_segments", 0
                                                        )
                                                    )
                                                    > 0
                                                ):
                                                    seg_pos = cached_segments.get(
                                                        "segment_pos", None
                                                    )
                                                    seg_size_x = cached_segments.get(
                                                        "segment_size_x", None
                                                    )
                                                    seg_size_y = cached_segments.get(
                                                        "segment_size_y", None
                                                    )
                                                    if (
                                                        seg_pos is not None
                                                        and seg_size_x is not None
                                                        and seg_size_y is not None
                                                    ):
                                                        ov_cost, ov_max_density = overflow_op(
                                                            seg_pos,
                                                            seg_size_x,
                                                            seg_size_y,
                                                        )
                                                        l_shape_overflow = float(
                                                            ov_cost.item()
                                                        )
                                                        l_shape_max_density = float(
                                                            ov_max_density.item()
                                                        )

                                                bin_area = float(
                                                    overflow_op.bin_size_x
                                                    * overflow_op.bin_size_y
                                                )
                                                target_density = overflow_op.target_density
                                                if isinstance(target_density, torch.Tensor):
                                                    target_total = float(
                                                        (
                                                            target_density.to(
                                                                density_map.device,
                                                                dtype=density_map.dtype,
                                                            )
                                                            * bin_area
                                                        )
                                                        .sum()
                                                        .item()
                                                    )
                                                else:
                                                    target_total = float(
                                                        float(target_density)
                                                        * bin_area
                                                        * overflow_op.num_bins_x
                                                        * overflow_op.num_bins_y
                                                    )
                                                overflow_ratio = float(
                                                    l_shape_overflow / (target_total + 1e-12)
                                                )
                                            model.l_shape_overflow = l_shape_overflow
                                            model.l_shape_overflow_ratio = overflow_ratio
                                            model.l_shape_overflow_max_density = (
                                                l_shape_max_density
                                            )
                                            cur_metric.l_shape_overflow = l_shape_overflow
                                            cur_metric.l_shape_overflow_ratio = overflow_ratio
                                            cur_metric.l_shape_overflow_max_density = (
                                                l_shape_max_density
                                            )
                                            if l_shape_log_verbose(params) >= 1:
                                                logging.info(
                                                    "L-shape refresh iter=%d: "
                                                    "ov_raw=%.6e, ov_ratio=%.6e, max_density=%.6f",
                                                    iteration,
                                                    l_shape_overflow,
                                                    overflow_ratio,
                                                    l_shape_max_density,
                                                )

                                            l_shape_policy.update_overflow_target(
                                                model, iteration, l_shape_overflow, overflow_ratio
                                            )
                                except Exception as e:
                                    logging.warning(
                                        f"L-shape overflow outer-loop update failed at iter {iteration}: {e}"
                                    )
                            
                            logging.debug(f"L-shape routability updated at iteration {iteration}, "
                                         f"time={((time.time() - t_l_shape_update) * 1000):.2f}ms")
                            
                            # 定期可视化L形密度图和segments
                            if params.l_shape_plot_flag:
                                try:
                                    from dreamplace.ops.routability.l_shape_routability import (
                                        plot_l_shape_electric_overflow_map,
                                        plot_l_shape_initial_density_map,
                                        plot_l_shape_macro_source_maps,
                                        plot_l_shape_electric_potential_map,
                                        plot_l_shape_supply_maps,
                                        plot_l_shape_true_source_maps,
                                        plot_segment_density_map,
                                        plot_soft_l_intermediate,
                                        plot_soft_l_scoring_maps,
                                    )
                                    
                                    if density_map is None:
                                        forward_source_snapshot = model.snapshot_l_shape_forward_state()
                                        density_map = model.get_l_shape_density_map(
                                            pos, use_l_direction=True
                                        )
                                        model.restore_l_shape_forward_state(forward_source_snapshot)
                                    if density_map is not None:
                                        density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_density_iter{iteration}.png"
                                        )
                                        plot_segment_density_map(
                                            density_map, density_plot_path,
                                            title=f"L-shape Density (iter={iteration})",
                                            colormap="binary"
                                        )
                                    
                                    # 绘制基于 electric potential 的 overflow map
                                    if model.l_shape_routability_op is not None and \
                                       model.l_shape_routability_op.cached_segments is not None:
                                        l_shape_op = model.l_shape_routability_op
                                        overflow_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_overflow_map(
                                            l_shape_op,
                                            output_path=overflow_plot_path,
                                            title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                                        )
                                        supply_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_supply_iter{iteration}.png"
                                        )
                                        plot_l_shape_supply_maps(
                                            l_shape_op,
                                            output_path=supply_plot_path,
                                            title_prefix=f"L-shape Supply Debug (iter={iteration})",
                                        )
                                        initial_density_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                                        )
                                        plot_l_shape_initial_density_map(
                                            l_shape_op,
                                            output_path=initial_density_plot_path,
                                            title_prefix=f"L-shape Initial Density (iter={iteration})",
                                        )
                                        source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_true_source_maps(
                                            l_shape_op,
                                            output_path=source_plot_path,
                                            title_prefix=f"L-shape True Source (iter={iteration})",
                                        )
                                        macro_source_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                                        )
                                        plot_l_shape_macro_source_maps(
                                            l_shape_op,
                                            output_path=macro_source_plot_path,
                                            title_prefix=f"L-shape Macro Source (iter={iteration})",
                                        )
                                        potential_plot_path = os.path.join(
                                            params.result_dir, f"l_shape_potential_iter{iteration}.png"
                                        )
                                        plot_l_shape_electric_potential_map(
                                            l_shape_op,
                                            output_path=potential_plot_path,
                                            title_prefix=f"L-shape Electric Potential (iter={iteration})",
                                        )
                                        if getattr(l_shape_op, "soft_l_assignment", False) and \
                                           'soft_l_weights' in l_shape_op.cached_segments:
                                            soft_plot_path = os.path.join(
                                                params.result_dir, f"l_shape_soft_iter{iteration}.png"
                                            )
                                            plot_soft_l_intermediate(
                                                l_shape_op.cached_segments,
                                                output_path=soft_plot_path,
                                            )
                                            if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                                                soft_scoring_path = os.path.join(
                                                    params.result_dir,
                                                    f"l_shape_soft_scoring_iter{iteration}.png",
                                                )
                                                plot_soft_l_scoring_maps(
                                                    l_shape_op.cached_soft_debug,
                                                    output_path=soft_scoring_path,
                                                    title_prefix=f"Soft L Scoring (iter={iteration})",
                                                )
                                except Exception as e:
                                    logging.warning(f"Failed to plot L-shape density/segments: {e}")
                    # ======================================================

                    # plot placement
                    if params.plot_flag and (iteration % 30 == 0 or iteration == 999):
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    # stop updating fence regions that are marked stop, exclude the outer cell !
                    t3 = time.time()
                    if model.update_mask is not None:
                        pos_bk = pos.data.clone()
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        optimizer.step()

                        for region_id, fence_region_update_flag in enumerate[Any](model.update_mask):
                            if fence_region_update_flag == 0:
                                # don't update cell location in that region
                                mask = self.op_collections.fence_region_density_ops[region_id].pos_mask
                                pos.data.masked_scatter_(mask, pos_bk[mask])
                    else:
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        optimizer.step()

                    logging.info("optimizer step %.3f ms" %
                                 ((time.time() - t3) * 1000))

                    # Perform timing-opt.
                    if params.global_place_flag and params.timing_opt_flag and \
                            params.enable_net_weighting and \
                            iteration > 500 and iteration % 15 == 0:
                        # Take the timing operator from the operator collections.
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        # The timing operator has already integrated timer as its
                        # instance variable, so it only takes one argument.
                        timing_op(self.pos[0].data.clone().cpu())
                        timing_op.timer.update_timing()
                        npaths = max(1, int(placedb.num_nets * 0.03))

                        # Report timing step.
                        # Temporary solution: modify net weights
                        beg = time.time()
                        timing_op.update_net_weights(
                            max_net_weight=placedb.max_net_weight,
                            n=npaths)
                        if self.device != torch.device("cpu"):
                            # Copy weights from placedb.net_weights to device.
                            self.data_collections.net_weights.copy_(
                                torch.from_numpy(placedb.net_weights))
                        logging.info("net-weight update step %.3f ms" %
                                     ((time.time() - beg) * 1000))

                        # Report tns and wns in each timing feedback call.
                        # Note that OpenTimer considers early,late,rise,fall for tns/wns.
                        # The following values are for reference.
                        cur_metric.tns = timing_op.timer.report_tns_elw(
                            split=1) / (time_unit * 1e17)
                        cur_metric.wns = timing_op.timer.report_wns(
                            split=1) / (time_unit * 1e15)

                    # nesterov has already computed the objective of the next step
                    if optimizer_name.lower() == "nesterov":
                        cur_metric.objective = optimizer.param_groups[0]["obj_k_1"][0].data.clone(
                        )

                    l_shape_policy.maybe_auto_disable(model, iteration, outer_update=False)
                    l_shape_policy.maybe_disable_by_ratio(model, iteration)
                    model.collect_l_shape_telemetry(cur_metric)
                    routability_controller.after_iteration(
                        model=model,
                        iteration=iteration,
                        metrics=cur_metric,
                    )
                    # actually reports the metric before step
                    logging.info(cur_metric)
                    # record the best outer cell overflow
                    if best_metric[0] is None or best_metric[0].overflow[-1] > cur_metric.overflow[-1]:
                        best_metric[0] = cur_metric
                        if best_pos[0] is None:
                            best_pos[0] = self.pos[0].data.clone()
                        else:
                            best_pos[0].data.copy_(self.pos[0].data)

                    logging.info("full step %.3f ms" %
                                 ((time.time() - t0) * 1000))

                def check_plateau(x, window=10, threshold=0.001):
                    if len(x) < window:
                        return False
                    x = x[-window:]
                    return (np.max(x) - np.min(x)) / np.mean(x) < threshold

                def check_divergence(x, window=50, threshold=0.05):
                    if len(x) < window or best_metric[0] is None:
                        return False
                    x = np.array(x[-window:])
                    overflow_mean = np.mean(x[:, 1])
                    overflow_diff = np.maximum(0, np.sign(
                        x[1:, 1] - x[:-1, 1])).astype(np.float32)
                    overflow_diff = np.sum(
                        overflow_diff) / overflow_diff.shape[0]
                    overflow_range = np.max(x[:, 1]) - np.min(x[:, 1])
                    wl_mean = np.mean(x[:, 0])
                    wl_ratio, overflow_ratio = (wl_mean - best_metric[0].hpwl.item()) / best_metric[
                        0
                    ].hpwl.item(), (
                        overflow_mean -
                        max(params.stop_overflow,
                            best_metric[0].overflow.item())
                    ) / best_metric[
                        0
                    ].overflow.item()
                    if wl_ratio > threshold * 1.2:
                        # this condition is not suitable for routability-driven opt with cell inflation
                        if (not params.routability_opt_flag) and overflow_ratio > threshold:
                            logging.warning(
                                f"Divergence detected: overflow increases too much than best overflow ({overflow_ratio:.4f} > {threshold:.4f})"
                            )
                            return True
                        elif overflow_range / overflow_mean < threshold:
                            logging.warning(
                                f"Divergence detected: overflow plateau ({overflow_range/overflow_mean:.4f} < {threshold:.4f})"
                            )
                            return True
                        elif overflow_diff > 0.6:
                            logging.warning(
                                f"Divergence detected: overflow fluctuate too frequently ({overflow_diff:.2f} > 0.6)"
                            )
                            return True
                        else:
                            return False
                    else:
                        return False

                def entropy_injection(
                    pos, placedb, shrink_factor=1, noise_intensity=1, mode="random", iteration=1
                ):
                    if mode == "random":
                        # print(pos[: placedb.num_movable_nodes].mean())
                        xc = pos[: placedb.num_movable_nodes].data.mean()
                        yc = pos.data[
                            placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                        ].mean()
                        num_movable_nodes = placedb.num_movable_nodes
                        num_nodes = placedb.num_nodes
                        num_filler_nodes = placedb.num_filler_nodes
                        num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes

                        fixed_pos_x = pos.data[
                            num_movable_nodes: num_movable_nodes + num_fixed_nodes
                        ].clone()
                        fixed_pos_y = pos.data[
                            num_nodes + num_movable_nodes: num_nodes + num_movable_nodes + num_fixed_nodes
                        ].clone()
                        if shrink_factor != 1:
                            pos.data[:num_nodes] = (
                                pos.data[:num_nodes] - xc) * shrink_factor + xc
                            pos.data[num_nodes:] = (
                                pos.data[num_nodes:] - yc) * shrink_factor + yc
                        if noise_intensity > 0.01:
                            # pos.data.add_(noise_intensity * torch.rand(num_nodes*2, device=pos.device).sub_(0.5))
                            pos.data.add_(
                                noise_intensity * torch.randn(num_nodes * 2, device=pos.device))

                        pos.data[num_movable_nodes: num_movable_nodes +
                                 num_fixed_nodes] = fixed_pos_x
                        pos.data[
                            num_nodes + num_movable_nodes: num_nodes + num_movable_nodes + num_fixed_nodes
                        ] = fixed_pos_y
                        # print(pos[: placedb.num_movable_nodes].mean())
                    else:
                        raise NotImplementedError

                Lgamma_metrics = all_metrics

                if params.routability_opt_flag:
                    routability_controller.ensure_modularity_inflation_contract(params)
                    adjust_area_flag = True
                    adjust_route_area_flag = (
                        getattr(params, "adjust_gpugr_area_flag", False)
                        or params.adjust_nctugr_area_flag
                        or params.adjust_rudy_area_flag
                    )
                    adjust_pin_area_flag = params.adjust_pin_area_flag
                    num_area_adjust = 0
                    max_area_adjust_rounds = routability_controller.inflation.get_inflation_round_limit(params)
                    if getattr(model, "inflation_state", None) is not None:
                        model.inflation_state.num_area_adjust = 0
                    if inflation_legalization.is_enabled(params):
                        logging.info(
                            "Ordinary inflation will trigger at stop_overflow=%.6g "
                            "and legalize physical cell geometry before every round",
                            inflation_legalization.legacy_trigger_threshold(params),
                        )

                Llambda_flat_iteration = 0

                # L-shape routability 状态初始化
                model.enable_l_shape_routability = False
                l_shape_policy.reset_auto_disable_state(model)
                l_shape_policy.reset_reenable_state(model)

                # preparation for self-adaptive divergence check
                overflow_list = [1]
                divergence_list = []
                min_perturb_interval = 50
                stop_placement = 0
                last_perturb_iter = -min_perturb_interval
                perturb_counter = 0

                for Lgamma_step in range(model.Lgamma_iteration):
                    Lgamma_metrics.append([])
                    Llambda_metrics = Lgamma_metrics[-1]
                    inflation_applied_this_gamma = False
                    for Llambda_density_weight_step in range(model.Llambda_density_weight_iteration):
                        Llambda_metrics.append([])
                        Lsub_metrics = Llambda_metrics[-1]
                        for Lsub_step in range(model.Lsub_iteration):
                            # divergence threshold should decrease as overflow decreases
                            # only detect divergence when overflow is relatively low but not too low
                            div_flag = check_divergence(
                                # sometimes maybe too aggressive...
                                divergence_list, window=50, threshold=overflow_list[-1])
                            if params.timing_opt_flag:
                                # currently do not check divergence in timing-driven placement
                                # TODO: a better way for divergence detection and roll-back for tdp.
                                div_flag = False
                            if (
                                len(placedb.regions) == 0
                                and params.stop_overflow * 1.1 < overflow_list[-1] < params.stop_overflow * 4
                                and div_flag
                            ):
                                self.pos[0].data.copy_(best_pos[0].data)
                                stop_placement = 1

                                logging.error(
                                    "possible DIVERGENCE detected, roll back to the best position recorded"
                                )

                            one_descent_step(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, Lsub_metrics
                            )

                            if len(placedb.regions) == 0:
                                overflow_list.append(
                                    Llambda_metrics[-1][-1].overflow.data.item())
                                divergence_list.append(
                                    [
                                        Llambda_metrics[-1][-1].hpwl.data.item(),
                                        Llambda_metrics[-1][-1].overflow.data.item(),
                                    ]
                                )

                            # quadratic penalty and entropy injection
                            # This heuristics makes placement unstable
                            if (
                                len(placedb.regions) == 0
                                and iteration - last_perturb_iter > min_perturb_interval
                                and check_plateau(overflow_list, window=15, threshold=0.001)
                            ):
                                if overflow_list[-1] > 0.9:  # stuck at high overflow
                                    model.quad_penalty = True
                                    model.density_factor *= 2
                                    logging.info(
                                        f"Stuck at early stage. Turn on quadratic penalty with double density factor to accelerate convergence"
                                    )
                                    # stuck at very high overflow
                                    if overflow_list[-1] > 0.95:
                                        noise_intensity = min(
                                            max(40 + (120 - 40) *
                                                (overflow_list[-1] - 0.95) * 10, 40), 90
                                        )
                                        entropy_injection(
                                            self.pos[0],
                                            placedb,
                                            shrink_factor=0.996,
                                            noise_intensity=noise_intensity,
                                            mode="random",
                                        )
                                        logging.info(
                                            f"Stuck at very early stage. Turn on entropy injection with noise intensity = {noise_intensity} to help convergence"
                                        )
                                    last_perturb_iter = iteration
                                    perturb_counter += 1

                            iteration += 1
                            # stopping criteria
                            if Lsub_stop_criterion(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, Lsub_metrics
                            ):
                                break
                        Llambda_flat_iteration += 1

                        # update density weight
                        if Llambda_flat_iteration > 1:
                            model.op_collections.update_density_weight_op(
                                Llambda_metrics[-1][-1],
                                Llambda_metrics[-2][-1]
                                if len(Llambda_metrics) > 1
                                else Lgamma_metrics[-2][-1][-1],
                                Llambda_flat_iteration,
                            )
                        # logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        llambda_should_stop = Llambda_stop_criterion(
                            Lgamma_step,
                            Llambda_density_weight_step,
                            Llambda_metrics,
                        )
                        defer_stop_for_inflation = (
                            params.routability_opt_flag
                            and routability_controller.should_defer_stop_for_legacy_inflation(
                                params,
                                num_area_adjust,
                                max_area_adjust_rounds,
                                Llambda_metrics[-1][-1].overflow,
                            )
                        )
                        if llambda_should_stop and not defer_stop_for_inflation:
                            break

                        # for routability optimization
                        if params.routability_opt_flag:
                            trigger_enhanced_inflation, trigger_legacy_inflation = routability_controller.inflation_triggers(
                                params,
                                num_area_adjust,
                                max_area_adjust_rounds,
                                Llambda_metrics[-1][-1].overflow,
                            )
                            if trigger_enhanced_inflation or trigger_legacy_inflation:
                                routability_controller.before_area_adjust(
                                    model=model,
                                    position=model.data_collections.pos[0],
                                )
                                use_enhanced_inflation = bool(trigger_enhanced_inflation)
                                round_flags = routability_controller.area_adjust_flags(
                                    params,
                                    use_enhanced=use_enhanced_inflation,
                                    default_flags={
                                        "adjust_area_flag": adjust_area_flag,
                                        "adjust_route_area_flag": adjust_route_area_flag,
                                        "adjust_pin_area_flag": adjust_pin_area_flag,
                                    },
                                )
                                round_adjust_area_flag = round_flags["adjust_area_flag"]
                                round_adjust_route_area_flag = round_flags["adjust_route_area_flag"]
                                round_adjust_pin_area_flag = round_flags["adjust_pin_area_flag"]
                                content = (
                                    "routability optimization round %d: adjust area flags = (%d, %d, %d)"
                                    % (
                                        num_area_adjust,
                                        round_adjust_area_flag,
                                        round_adjust_route_area_flag,
                                        round_adjust_pin_area_flag,
                                    )
                                )
                                pos = model.data_collections.pos[0]
                                if use_enhanced_inflation and best_pos[0] is not None:
                                    pos.data.copy_(best_pos[0].data)
                                    content = (
                                        "enhanced inflation round %d: restore best_pos snapshot before gpugr/area-adjust | "
                                        % num_area_adjust
                                    ) + content
                                    logging.info(
                                        "enhanced inflation round %d uses best_pos snapshot with best overflow %.6f at iter %d",
                                        num_area_adjust,
                                        float(best_metric[0].overflow[-1]) if best_metric[0] is not None else float("nan"),
                                        int(best_metric[0].iteration) if best_metric[0] is not None else -1,
                                    )
                                elif use_enhanced_inflation:
                                    logging.info(
                                        "enhanced inflation round %d cannot find an earlier best_pos snapshot; use current position as trigger input",
                                        num_area_adjust,
                                    )
                                route_map_source = "none"
                                if round_adjust_route_area_flag:
                                    route_map_source = routability_controller.resolve_route_map_source(params)
                                if round_adjust_route_area_flag:
                                    routability_controller.ensure_modularity_inflation_contract(
                                        params,
                                        route_map_source=route_map_source,
                                    )
                                current_inflation_round = None
                                if getattr(model, "inflation_state", None) is not None:
                                    routability_controller.inflation.maybe_capture_model_density_state(
                                        model.inflation_state,
                                        model,
                                    )
                                    current_inflation_round = routability_controller.inflation.begin_inflation_round(
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        pos=pos,
                                        round_idx=num_area_adjust,
                                        stage_idx=cur_stage,
                                        iteration=iteration,
                                        overflow=Llambda_metrics[-1][-1].overflow,
                                        route_map_source=route_map_source,
                                        adjust_area_flag=round_adjust_area_flag,
                                        adjust_route_area_flag=round_adjust_route_area_flag,
                                        adjust_pin_area_flag=round_adjust_pin_area_flag,
                                        notes=(
                                            "enhanced_inflation_best_pos"
                                            if use_enhanced_inflation
                                            else (
                                                "legacy_area_adjust_after_legalization"
                                                if inflation_legalization.is_enabled(params)
                                                else "legacy_area_adjust"
                                            )
                                        ),
                                    )
                                physical_geometry_backup = None
                                if inflation_legalization.is_enabled(params):
                                    physical_geometry_backup = (
                                        inflation_legalization.use_physical_geometry(
                                            pos,
                                            self.data_collections,
                                        )
                                    )
                                    physical_pre_legalization_pos = (
                                        pos.detach().clone()
                                    )
                                    _log_inflation_macro_overlap(
                                        placedb,
                                        pos,
                                        self.data_collections,
                                        num_area_adjust,
                                        "after_gp_before_legalization",
                                    )
                                    legalization_start = time.time()
                                    routability_controller.before_legalization(
                                        model=model,
                                        position=pos,
                                    )
                                    pos.data.copy_(
                                        self.op_collections.legalize_op(pos)
                                    )
                                    routability_controller.after_legalization(
                                        model=model,
                                        position=pos,
                                    )
                                    logging.info(
                                        "Inflation round %d physical legalization "
                                        "takes %.3f seconds",
                                        num_area_adjust,
                                        time.time() - legalization_start,
                                    )
                                    _log_inflation_macro_overlap(
                                        placedb,
                                        pos,
                                        self.data_collections,
                                        num_area_adjust,
                                        "after_legalization",
                                        reference_pos=physical_pre_legalization_pos,
                                    )
                                route_evaluation.run_gpugr_before_first_area_adjust_and_exit(
                                    params,
                                    placedb,
                                    pos,
                                    num_area_adjust,
                                    model,
                                )
                                if round_adjust_route_area_flag:
                                    routability_controller.ensure_active_modularity_clusters(
                                        params,
                                        placedb,
                                        model,
                                        pos,
                                        num_area_adjust,
                                    )

                                route_utilization_map = None
                                pin_utilization_map = None
                                modularity_maps = None
                                gpugr_metrics = {}
                                low_util_context = None
                                fixed_target_area = None
                                if round_adjust_route_area_flag:
                                    if route_map_source == "gpugr":
                                        gpugr_context.sync_route_grid_to_autodmp(
                                            params,
                                            placedb,
                                            model=model,
                                        )
                                        route_utilization_map = model.op_collections.gpugr_congestion_map_op(
                                            pos
                                        )
                                        gpugr_metrics = getattr(
                                            model.op_collections.gpugr_congestion_map_op,
                                            "last_metrics",
                                            {},
                                        ) or {}
                                        if routability_controller.is_modularity_inflation_enabled(params):
                                            modularity_maps = route_map_utils.prepare_modularity_maps_from_gpugr(
                                                placedb,
                                                model.op_collections.gpugr_congestion_map_op,
                                                pos,
                                                eps=getattr(params, "modularity_active_bin_overflow_eps", 1e-6),
                                            )
                                    elif route_map_source == "irt_egr":
                                        route_utilization_map = model.op_collections.irt_egr_congestion_map_op(
                                            pos, stage="egr3D", resolve_congestion="high")
                                    else:
                                        route_utilization_map = model.op_collections.route_utilization_map_op(
                                            pos)
                                if physical_geometry_backup is not None:
                                    inflation_legalization.restore_inflated_geometry(
                                        pos,
                                        self.data_collections,
                                        physical_geometry_backup,
                                    )
                                    logging.info(
                                        "Inflation round %d restored cumulative "
                                        "inflated geometry after congestion estimation",
                                        num_area_adjust,
                                    )
                                if round_adjust_pin_area_flag:
                                    pin_utilization_map = model.op_collections.pin_utilization_map_op(
                                        pos)
                                    if params.plot_flag:
                                        path = "%s/%s" % (params.result_dir,
                                                          params.design_name())
                                        figname = "%s/plot/pin%d.png" % (
                                            path, num_area_adjust)
                                        os.system("mkdir -p %s" %
                                                  (os.path.dirname(figname)))
                                        plt.imsave(
                                            figname, pin_utilization_map.data.cpu().numpy().T, origin="lower"
                                        )
                                if getattr(model, "inflation_state", None) is not None:
                                    if use_enhanced_inflation:
                                        fixed_target_area = getattr(
                                            model.inflation_state,
                                            "target_area",
                                            None,
                                        )
                                    low_util_context = routability_controller.inflation.prepare_low_util_inflation(
                                        params,
                                        model.inflation_state,
                                        placedb,
                                        self.data_collections,
                                        model.op_collections.adjust_node_area_op,
                                        gpugr_metrics,
                                    )
                                (
                                    adjust_area_flag,
                                    adjust_route_area_flag,
                                    adjust_pin_area_flag,
                                ) = model.op_collections.adjust_node_area_op(
                                    pos,
                                    route_utilization_map,
                                    pin_utilization_map,
                                    modularity_maps=modularity_maps,
                                    inflation_round=int(num_area_adjust),
                                    fixed_target_area=fixed_target_area,
                                )
                                routability_controller.inflation.restore_low_util_inflation(
                                    model.op_collections.adjust_node_area_op,
                                    low_util_context,
                                )
                                low_util_metrics = {}
                                if adjust_area_flag:
                                    low_util_metrics = routability_controller.inflation.apply_low_util_target_density(
                                        params,
                                        getattr(model, "inflation_state", None),
                                        self.data_collections,
                                        placedb,
                                        pos,
                                        low_util_context,
                                    )
                                min_area_inc_metrics = {}
                                if (
                                    adjust_area_flag
                                    and current_inflation_round is not None
                                    and use_enhanced_inflation
                                ):
                                    min_area_inc_result = routability_controller.inflation.enforce_min_area_increment(
                                        params,
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        pos,
                                    )
                                    movable_area_increment_ratio = min_area_inc_result.get(
                                        "movable_area_increment_ratio"
                                    )
                                    if movable_area_increment_ratio is not None:
                                        min_area_inc_metrics = {
                                            "enhanced_movable_area_increment_ratio": float(
                                                movable_area_increment_ratio
                                            ),
                                            "enhanced_min_area_increment_threshold": float(
                                                min_area_inc_result[
                                                    "min_area_increment_threshold"
                                                ]
                                            ),
                                        }
                                    if min_area_inc_result.get("triggered"):
                                        low_util_metrics = {}
                                        adjust_area_flag = False
                                        adjust_route_area_flag = False
                                        adjust_pin_area_flag = False
                                content += " -> (%d, %d, %d)" % (
                                    adjust_area_flag,
                                    adjust_route_area_flag,
                                    adjust_pin_area_flag,
                                )
                                logging.info(content)
                                if current_inflation_round is not None:
                                    round_status = "applied" if adjust_area_flag else "stopped"
                                    routability_controller.inflation.finish_inflation_round(
                                        model.inflation_state,
                                        self.data_collections,
                                        placedb,
                                        adjust_area_flag=adjust_area_flag,
                                        adjust_route_area_flag=adjust_route_area_flag,
                                        adjust_pin_area_flag=adjust_pin_area_flag,
                                        status=round_status,
                                        pos=pos,
                                        gr_metrics=dict(
                                            {
                                                "placement_overflow": Llambda_metrics[-1][-1].overflow,
                                            },
                                            **min_area_inc_metrics,
                                            **low_util_metrics,
                                            **gpugr_metrics,
                                        ),
                                    )
                                    selected_metric_name = getattr(
                                        params,
                                        "enhanced_inflation_select_metric",
                                        "est_shorts",
                                    )
                                    round_metric_value = routability_controller.inflation.get_round_metric(
                                        model.inflation_state.round_records[-1],
                                        selected_metric_name,
                                    )
                                    if round_metric_value is not None:
                                        logging.info(
                                            "Recorded %s inflation round %d: %s=%.4f trigger_overflow=%.6f",
                                            "enhanced"
                                            if use_enhanced_inflation
                                            else "ordinary",
                                            num_area_adjust,
                                            selected_metric_name,
                                            round_metric_value,
                                            float(Llambda_metrics[-1][-1].overflow),
                                        )
                                if adjust_area_flag:
                                    inflation_applied_this_gamma = True
                                    num_area_adjust += 1
                                    if getattr(model, "inflation_state", None) is not None:
                                        model.inflation_state.num_area_adjust = num_area_adjust
                                    # restart Llambda
                                    model.refresh_after_geometry_change()
                                    model.initialize_density_weight(
                                        params, placedb)
                                    model.density_weight.mul_(
                                        0.1 / params.density_weight)
                                    logging.info("density_weight = %.6E" %
                                                 (model.density_weight.data))
                                    # load state to restart the optimizer
                                    optimizer.load_state_dict(initial_state)
                                    # must after loading the state
                                    initialize_learning_rate(pos)
                                    # increase iterations of the sub problem to slow down the search
                                    model.Lsub_iteration = model.routability_Lsub_iteration
                                    routability_controller.after_geometry_change(
                                        model=model,
                                        position=pos,
                                    )

                                    # reset best metric
                                    best_metric[0] = None
                                    best_pos[0] = None

                                    # disable L-shape during inflation recovery;
                                    # it will re-enable when overflow drops below threshold again
                                    # NOTE: can be overridden by l_shape_keep_during_inflation parameter
                                    keep_during_inflation = getattr(params, "l_shape_keep_during_inflation", False)
                                    if getattr(model, "use_l_shape_routability", False) and not keep_during_inflation:
                                        l_shape_policy.disable_for_recovery(
                                            model,
                                            iteration,
                                            reason="inflation",
                                            update_threshold=True,
                                            inflation_round=num_area_adjust,
                                        )

                                    break
                                else:
                                    num_area_adjust = max_area_adjust_rounds
                                    if getattr(model, "inflation_state", None) is not None:
                                        model.inflation_state.num_area_adjust = num_area_adjust
                                    logging.info(
                                        "Terminate routability inflation after round %d because adjust_node_area reported no further area change",
                                        current_inflation_round.round_idx
                                        if current_inflation_round is not None
                                        else num_area_adjust,
                                    )

                    # gradually reduce gamma to tradeoff smoothness and accuracy
                    if len(placedb.regions) > 0 and Llambda_metrics[-1][-1].goverflow is not None:
                        model.op_collections.update_gamma_op(
                            Lgamma_step, Llambda_metrics[-1][-1].goverflow)
                    elif len(placedb.regions) == 0 and Llambda_metrics[-1][-1].overflow is not None:
                        model.op_collections.update_gamma_op(
                            Lgamma_step, Llambda_metrics[-1][-1].overflow)
                    else:
                        model.op_collections.precondition_op.set_overflow(
                            Llambda_metrics[-1][-1].overflow)
                    if (
                        not inflation_applied_this_gamma
                        and Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics)
                    ) or stop_placement == 1:
                        break

                    # update learning rate
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]:
                        if "learning_rate_decay" in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] *= global_place_params["learning_rate_decay"]

                # in case of divergence, use the best metric
                last_metric = all_metrics[-1][-1][-1]
                # if (
                #     last_metric.overflow[-1] > max(params.stop_overflow, best_metric[0].overflow[-1])
                #     and last_metric.hpwl > best_metric[0].hpwl
                # ):
                #     all_metrics.append([best_metric])
                # fix movable macros
                if params.macro_place_flag and not macro_placed:
                    macro_placed = True
                    # recover halo
                    if params.macro_halo_x > 0 or params.macro_halo_y > 0:
                        with torch.no_grad():
                            movable_macro_mask = self.data_collections.movable_macro_mask
                            movable_macro_pins = self.data_collections.movable_macro_pins
                            # node sizes
                            self.data_collections.node_size_x[: placedb.num_movable_nodes][movable_macro_mask] -= (
                                2 * params.macro_halo_x)
                            self.data_collections.node_size_y[: placedb.num_movable_nodes][movable_macro_mask] -= (
                                2 * params.macro_halo_y)
                            # pin offsets
                            self.data_collections.pin_offset_x[movable_macro_pins] -= params.macro_halo_x
                            self.data_collections.pin_offset_y[movable_macro_pins] -= params.macro_halo_y
                            # macro locations
                            self.pos[0][: placedb.num_movable_nodes][movable_macro_mask] += params.macro_halo_x
                            self.pos[0][placedb.num_nodes: placedb.num_nodes +
                                        placedb.num_movable_nodes][movable_macro_mask] += params.macro_halo_y
                            params.macro_halo_x = 0
                            params.macro_halo_y = 0
                    if params.macro_pin_halo_x >= 0:
                        with torch.no_grad():
                            macro_pin_to_macro = np.searchsorted(
                                placedb.movable_macro_idx,
                                placedb.pin2node_map[placedb.movable_macro_pins],
                            )
                            self.data_collections.node_size_x[placedb.movable_macro_idx] -= torch.tensor(
                                placedb.is_pin_lower_x * params.macro_pin_halo_x + placedb.is_pin_upper_x * params.macro_pin_halo_x, device=self.pos[0].device)
                            self.data_collections.node_size_y[placedb.movable_macro_idx] -= torch.tensor(
                                placedb.is_pin_lower_y * params.macro_pin_halo_y + placedb.is_pin_upper_y * params.macro_pin_halo_y, device=self.pos[0].device)

                            self.data_collections.pin_offset_x[placedb.movable_macro_pins] -= torch.tensor(
                                placedb.is_pin_lower_x[macro_pin_to_macro] * params.macro_pin_halo_x, device=self.pos[0].device)
                            self.data_collections.pin_offset_y[placedb.movable_macro_pins] -= torch.tensor(
                                placedb.is_pin_lower_y[macro_pin_to_macro] * params.macro_pin_halo_y, device=self.pos[0].device)
                            # macro locations

                            self.pos[0][placedb.movable_slice][
                                placedb.movable_macro_mask
                            ] += torch.tensor(placedb.is_pin_lower_x * params.macro_pin_halo_x, device=self.pos[0].device)

                            self.pos[0][
                                placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                            ][placedb.movable_macro_mask] += torch.tensor(placedb.is_pin_lower_y * params.macro_pin_halo_y, device=self.pos[0].device)
                            params.macro_pin_halo_x = 0
                            params.macro_pin_halo_y = 0

                    if last_metric and (
                        last_metric.overflow[-1] > params.stop_overflow
                        or torch.isinf(last_metric.objective)
                        or torch.isnan(last_metric.objective)
                    ):
                        break

                    if params.plot_flag:
                        self.plot(params, placedb, iteration,
                                  self.pos[0].data.clone().cpu().numpy())
                    self.pos[0].data.copy_(
                        self.op_collections.macro_legalize_op(self.pos[0]))
                    iteration += 1
                    if params.plot_flag:
                        self.plot(params, placedb, iteration,
                                  self.pos[0].data.clone().cpu().numpy())

                logging.info("optimizer %s takes %.3f seconds" %
                             (optimizer_name, time.time() - tt))

            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                selected_inflation_round = None
                replay_best_inflation_round = bool(
                    getattr(params, "enhanced_inflation_replay_best_round_flag", False)
                )
                if (
                    replay_best_inflation_round
                    and routability_controller.inflation.is_enhanced_inflation_enabled(params)
                ):
                    selected_inflation_round = routability_controller.inflation.select_best_gr_solution(
                        getattr(self.data_collections, "inflation_state", None),
                        metric_name=getattr(
                            params,
                            "enhanced_inflation_select_metric",
                            "est_shorts",
                        ),
                    )
                with torch.no_grad():
                    # convert lower left to centers
                    self.pos[0][: placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_x[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_y[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.node_size_x.copy_(
                        self.data_collections.original_node_size_x)
                    self.data_collections.node_size_y.copy_(
                        self.data_collections.original_node_size_y)
                    routability_controller.inflation.sync_node_areas(
                        self.data_collections
                    )
                    # use fixed centers as the anchor
                    self.pos[0][: placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_x[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_y[:
                                                          placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.pin_offset_x.copy_(
                        self.data_collections.original_pin_offset_x)
                    self.data_collections.pin_offset_y.copy_(
                        self.data_collections.original_pin_offset_y)
                    routability_controller.inflation.rollback_inflation_state(
                        self.data_collections
                    )
                    routability_controller.inflation.restore_model_density_state(
                        getattr(self.data_collections, "inflation_state", None),
                        model,
                    )
                if selected_inflation_round is not None:
                    with torch.no_grad():
                        self.pos[0].data.copy_(
                            selected_inflation_round.position_snapshot.data.to(
                                device=self.pos[0].device,
                                dtype=self.pos[0].dtype,
                            )
                        )
                    selected_metric_name = getattr(
                        params,
                        "enhanced_inflation_select_metric",
                        "est_shorts",
                    )
                    selected_metric_value = routability_controller.inflation.get_round_metric(
                        selected_inflation_round,
                        selected_metric_name,
                    )
                    logging.info(
                        "Replay enhanced best inflation round %d after rollback using %s=%.4f (trigger_overflow=%.6f, stage=%d, iter=%d)",
                        selected_inflation_round.round_idx,
                        selected_metric_name,
                        float(selected_metric_value)
                        if selected_metric_value is not None
                        else float("nan"),
                        selected_inflation_round.trigger_overflow,
                        selected_inflation_round.stage_idx,
                        selected_inflation_round.iteration,
                    )
                elif (
                    not replay_best_inflation_round
                    and routability_controller.inflation.is_enhanced_inflation_enabled(params)
                ):
                    logging.info(
                        "Skip enhanced best-round replay after rollback because enhanced_inflation_replay_best_round_flag is disabled"
                    )
                elif replay_best_inflation_round and routability_controller.inflation.is_enhanced_inflation_enabled(params):
                    logging.info(
                        "Skip enhanced best-round replay after rollback because no eligible inflation round was recorded"
                    )

        else:
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
        if params.plot_flag:
            self.plot(params, placedb, 9999,
                      self.pos[0].data.clone().cpu().numpy())

        processed_metrics = {
            "objective": [-1],
            "hpwl": [-1],
            "overflow": [-1],
            "density": [-1],
        }
        if params.global_place_flag:
            # dump global placement solution for legalization
            if params.dump_global_place_solution_flag:
                self.dump(params, placedb, self.pos[0].cpu(
                ), "%s.lg.pklz" % (params.design_name()))

            # process metrics
            def flatten(l): return sum(map(flatten, l), []
                                       ) if isinstance(l, list) else [l]
            metrics = flatten(all_metrics)
            objectives = [metric.objective.data.item() for metric in metrics]
            hpwls = [metric.hpwl.data.item() for metric in metrics]
            overflows = [metric.overflow.data.item() for metric in metrics]
            densities = [metric.max_density.data.item() for metric in metrics]
            processed_metrics = {
                "objective": objectives,
                "hpwl": hpwls,
                "overflow": overflows,
                "density": densities,
            }
            optional_metric_fields = [
                "l_shape_fast_mode",
                "l_shape_energy_valid",
                "l_shape_cost",
                "l_shape_weighted_cost",
                "l_shape_weight",
                "l_shape_target_weight",
                "l_shape_weight_candidate",
                "l_shape_base_grad_norm",
                "l_shape_grad_raw_norm",
                "l_shape_grad_norm",
                "l_shape_grad_ratio",
                "l_shape_target_ratio",
                "l_shape_overflow",
                "l_shape_overflow_ratio",
                "l_shape_overflow_ema",
                "l_shape_overflow_max_density",
                "l_shape_capacity_al_enabled",
                "l_shape_capacity_al_updated",
                "l_shape_capacity_al_g_h_max",
                "l_shape_capacity_al_g_v_max",
                "l_shape_capacity_al_g_h_sum",
                "l_shape_capacity_al_g_v_sum",
                "l_shape_capacity_al_g_h_pos_ratio",
                "l_shape_capacity_al_g_v_pos_ratio",
                "l_shape_capacity_al_q_h_max",
                "l_shape_capacity_al_q_v_max",
                "l_shape_capacity_al_q_h_sum",
                "l_shape_capacity_al_q_v_sum",
                "l_shape_capacity_al_lambda_h_max",
                "l_shape_capacity_al_lambda_v_max",
                "l_shape_capacity_al_lambda_h_sum",
                "l_shape_capacity_al_lambda_v_sum",
                "l_shape_capacity_al_energy_h",
                "l_shape_capacity_al_energy_v",
                "l_shape_capacity_al_energy_total",
                "l_shape_capacity_al_pq_h_min",
                "l_shape_capacity_al_pq_h_max",
                "l_shape_capacity_al_pq_h_sum",
                "l_shape_capacity_al_pq_v_min",
                "l_shape_capacity_al_pq_v_max",
                "l_shape_capacity_al_pq_v_sum",
                "l_shape_capacity_al_active_memory_bins_h",
                "l_shape_capacity_al_active_memory_bins_v",
                "l_shape_macro_exclusion_enabled",
                "l_shape_macro_exclusion_macro_count",
                "l_shape_macro_exclusion_body_bins",
                "l_shape_macro_exclusion_halo_bins",
                "l_shape_macro_exclusion_active_bins",
                "l_shape_macro_exclusion_source_max",
                "l_shape_macro_exclusion_source_sum",
                "l_shape_macro_exclusion_body_source_max",
                "l_shape_macro_exclusion_body_source_sum",
                "l_shape_macro_exclusion_halo_source_max",
                "l_shape_macro_exclusion_halo_source_sum",
                "l_shape_macro_exclusion_usage_max",
                "l_shape_macro_exclusion_usage_sum",
                "l_shape_macro_exclusion_usage_bins",
                "l_shape_macro_exclusion_dominates_bins",
                "l_shape_macro_exclusion_routing_dominates_bins",
                "soft_l_diag_count",
                "soft_l_mean_cost_gap",
                "soft_l_raw_cost_gap_p50",
                "soft_l_biased_cost_gap_p50",
                "soft_l_tau_source_gap",
                "soft_l_mean_max_prob",
                "soft_l_mean_entropy",
                "soft_l_near_tie_ratio",
                "soft_l_tau",
                "soft_l_effective_hotspot_weight",
                "soft_l_resolver_agreement_ratio",
                "soft_l_same_net_topo_nets",
                "soft_l_same_net_topo_segments_h",
                "soft_l_same_net_topo_segments_v",
                "soft_l_same_net_topo_diag_edges",
                "soft_l_same_net_topo_edges_with_topology",
                "soft_l_same_net_topo_edges_with_observed_intervals",
                "soft_l_same_net_topo_mean_gap",
                "soft_l_same_net_topo_tie_ratio",
            ]

            def scalarize_metric_value(value):
                if value is None:
                    return None
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return value.detach().cpu().view(-1).tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    return None
                return value

            fixed_when_l_shape_fields = {
                "l_shape_fast_mode",
                "l_shape_energy_valid",
            }
            energy_derived_metric_fields = {
                "l_shape_cost",
                "l_shape_weighted_cost",
                "l_shape_capacity_al_energy_h",
                "l_shape_capacity_al_energy_v",
                "l_shape_capacity_al_energy_total",
                "l_shape_capacity_al_pq_h_min",
                "l_shape_capacity_al_pq_h_max",
                "l_shape_capacity_al_pq_h_sum",
                "l_shape_capacity_al_pq_v_min",
                "l_shape_capacity_al_pq_v_max",
                "l_shape_capacity_al_pq_v_sum",
            }
            l_shape_seen = any(
                getattr(metric, "l_shape_fast_mode", None) is not None
                or getattr(metric, "l_shape_energy_valid", None) is not None
                for metric in metrics
            )
            l_shape_energy_invalid_seen = any(
                getattr(metric, "l_shape_energy_valid", None) is not None
                and not bool(getattr(metric, "l_shape_energy_valid"))
                for metric in metrics
            )
            for field_name in optional_metric_fields:
                series = [
                    scalarize_metric_value(getattr(metric, field_name, None))
                    for metric in metrics
                ]
                if (
                    any(value is not None for value in series)
                    or (l_shape_seen and field_name in fixed_when_l_shape_fields)
                    or (
                        l_shape_energy_invalid_seen
                        and field_name in energy_derived_metric_fields
                    )
                ):
                    processed_metrics[field_name] = series

            # plot placement
            if params.plot_flag:
                self.plot(params, placedb, iteration,
                          self.pos[0].data.clone().cpu().numpy())

            last_metric = copy.deepcopy(all_metrics)
            for idx in [-1, -1, -1]:
                try:
                    last_metric = last_metric[idx]
                except IndexError:
                    last_metric = False
                    break

            if not last_metric:
                cur_metric = EvalMetrics.EvalMetrics(iteration)
                all_metrics.append(cur_metric)
                cur_metric.evaluate(
                    placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
                )
                logging.info(cur_metric)

            # in case of significant divergence, no need to run legalizer
            if last_metric and (
                last_metric.overflow[-1] > params.stop_overflow
                or torch.isinf(last_metric.objective)
                or torch.isnan(last_metric.objective)
            ):
                logging.warn(
                    "overflow is significant %.3f or hpwl is infinity or nan, skip legalization and detail placement steps"
                    % (last_metric.overflow[-1])
                )
                self.plot(params, placedb, 9999,
                          self.pos[0].data.clone().cpu().numpy())
                return float("inf"), float("inf"), processed_metrics

            # recover node sizes, pins shifts, and positions of macros
            if params.macro_halo_x >= 0 and params.macro_halo_y >= 0:
                with torch.no_grad():
                    # node sizes
                    self.data_collections.node_size_x[placedb.movable_macro_idx] -= (
                        2 * params.macro_halo_x
                    )
                    self.data_collections.node_size_y[placedb.movable_macro_idx] -= (
                        2 * params.macro_halo_y
                    )
                    # self.data_collections.node_size_x[placedb.fixed_macro_idx] -= (
                    #     2 * params.macro_halo_x
                    # )
                    # self.data_collections.node_size_y[placedb.fixed_macro_idx] -= (
                    #     2 * params.macro_halo_y
                    # )

                    # pin offsets
                    self.data_collections.pin_offset_x[
                        placedb.movable_macro_pins
                    ] -= params.macro_halo_x
                    self.data_collections.pin_offset_y[
                        placedb.movable_macro_pins
                    ] -= params.macro_halo_y

                    self.pos[0][placedb.movable_slice][
                        placedb.movable_macro_mask
                    ] += params.macro_halo_x
                    self.pos[0][
                        placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                    ][placedb.movable_macro_mask] += params.macro_halo_y

                    self.pos[0][placedb.fixed_slice][
                        placedb.fixed_macro_mask
                    ] += params.macro_halo_x
                    self.pos[0][
                        placedb.num_nodes
                        + placedb.num_movable_nodes: placedb.num_nodes
                        + placedb.num_movable_nodes
                        + placedb.num_terminals
                    ][placedb.fixed_macro_mask] += params.macro_halo_y

                    # self.data_collections.pin_offset_x[
                    #     placedb.fixed_macro_pins
                    # ] -= params.macro_halo_x
                    # self.data_collections.pin_offset_y[
                    #     placedb.fixed_macro_pins
                    # ] -= params.macro_halo_y
            if params.macro_pin_halo_x >= 0:
                with torch.no_grad():
                    macro_pin_to_macro = np.searchsorted(
                        placedb.movable_macro_idx,
                        placedb.pin2node_map[placedb.movable_macro_pins],
                    )
                    self.data_collections.node_size_x[placedb.movable_macro_idx] -= torch.tensor(
                        placedb.is_pin_lower_x * params.macro_pin_halo_x + placedb.is_pin_upper_x * params.macro_pin_halo_x, device=self.pos[0].device)
                    self.data_collections.node_size_y[placedb.movable_macro_idx] -= torch.tensor(
                        placedb.is_pin_lower_y * params.macro_pin_halo_y + placedb.is_pin_upper_y * params.macro_pin_halo_y, device=self.pos[0].device)

                    self.data_collections.pin_offset_x[placedb.movable_macro_pins] -= torch.tensor(
                        placedb.is_pin_lower_x[macro_pin_to_macro] * params.macro_pin_halo_x, device=self.pos[0].device)
                    self.data_collections.pin_offset_y[placedb.movable_macro_pins] -= torch.tensor(
                        placedb.is_pin_lower_y[macro_pin_to_macro] * params.macro_pin_halo_y, device=self.pos[0].device)
                    # macro locations

                    self.pos[0][placedb.movable_slice][
                        placedb.movable_macro_mask
                    ] += torch.tensor(placedb.is_pin_lower_x * params.macro_pin_halo_x, device=self.pos[0].device)

                    self.pos[0][
                        placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                    ][placedb.movable_macro_mask] += torch.tensor(placedb.is_pin_lower_y * params.macro_pin_halo_y, device=self.pos[0].device)
                    params.macro_pin_halo_x = 0
                    params.macro_pin_halo_y = 0

        fixed_macro_pre_legalization_pos = None
        try:
            fixed_macro_pre_legalization_pos = self.pos[0].detach().clone()
            fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                self.pos[0],
                self.data_collections.node_size_x,
                self.data_collections.node_size_y,
                placedb,
            )
            for key, value in fixed_macro_overlap_stats.items():
                processed_metrics["pre_legalization_%s" % key] = value
            logging.info(
                "Fixed macro overlap telemetry: stage=pre_legalization "
                "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                "coordinate_system=%s macro_set_source=%s",
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_macro_count", 0
                    )
                ),
                float(fixed_macro_overlap_stats.get("fixed_macro_overlap_area", 0.0)),
                float(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_area_ratio", 0.0
                    )
                ),
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_cell_count", 0
                    )
                ),
                int(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_pair_count", 0
                    )
                ),
                float(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_max_area", 0.0
                    )
                ),
                str(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_coordinate_system", "unknown"
                    )
                ),
                str(
                    fixed_macro_overlap_stats.get(
                        "fixed_macro_overlap_macro_set_source", "unknown"
                    )
                ),
            )
        except Exception as e:
            logging.warning(
                "Failed to compute fixed macro overlap telemetry before legalization: %s",
                e,
            )

        # legalization
        if params.legalize_flag:
            if params.macro_place_flag:
                tt = time.time()
                routability_controller.before_legalization(
                    model=self._routability_model,
                    position=self.pos[0],
                )
                self.pos[0].data.copy_(
                    self.op_collections.macro_legalize_op(self.pos[0]))
                routability_controller.after_legalization(
                    model=self._routability_model,
                    position=self.pos[0],
                )
                logging.info("Macro legalization takes %.3f seconds" %
                             (time.time() - tt))
                try:
                    fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                        self.pos[0],
                        self.data_collections.node_size_x,
                        self.data_collections.node_size_y,
                        placedb,
                    )
                    for key, value in fixed_macro_overlap_stats.items():
                        processed_metrics["post_macro_legalization_%s" % key] = value
                    if fixed_macro_pre_legalization_pos is not None:
                        displacement_stats = compute_movable_displacement_stats(
                            fixed_macro_pre_legalization_pos, self.pos[0], placedb
                        )
                        for key, value in displacement_stats.items():
                            processed_metrics[
                                "post_macro_legalization_%s" % key
                            ] = value
                    else:
                        displacement_stats = {}
                    logging.info(
                        "Fixed macro overlap telemetry: stage=post_macro_legalization "
                        "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                        "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                        "legalization_moved=%d legalization_disp_max=%.6E "
                        "legalization_disp_mean=%.6E",
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_macro_count", 0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_area", 0.0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_area_ratio", 0.0
                            )
                        ),
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_cell_count", 0
                            )
                        ),
                        int(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_pair_count", 0
                            )
                        ),
                        float(
                            fixed_macro_overlap_stats.get(
                                "fixed_macro_overlap_max_area", 0.0
                            )
                        ),
                        int(
                            displacement_stats.get(
                                "movable_displacement_moved_count", 0
                            )
                        ),
                        float(
                            displacement_stats.get(
                                "movable_displacement_max", 0.0
                            )
                        ),
                        float(
                            displacement_stats.get(
                                "movable_displacement_mean", 0.0
                            )
                        ),
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to compute fixed macro overlap telemetry after macro legalization: %s",
                        e,
                    )
                cur_metric = EvalMetrics.EvalMetrics(iteration)
                all_metrics.append(cur_metric)
                cur_metric.evaluate(
                    placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
                logging.info(cur_metric)
                iteration += 1

        if params.legalize_flag:
            iteration = self._run_standard_legalization(
                params, placedb, iteration, all_metrics
            )
            try:
                fixed_macro_overlap_stats = compute_fixed_macro_overlap_stats(
                    self.pos[0],
                    self.data_collections.node_size_x,
                    self.data_collections.node_size_y,
                    placedb,
                )
                for key, value in fixed_macro_overlap_stats.items():
                    processed_metrics["post_legalization_%s" % key] = value
                if fixed_macro_pre_legalization_pos is not None:
                    displacement_stats = compute_movable_displacement_stats(
                        fixed_macro_pre_legalization_pos, self.pos[0], placedb
                    )
                    for key, value in displacement_stats.items():
                        processed_metrics["post_legalization_%s" % key] = value
                else:
                    displacement_stats = {}
                logging.info(
                    "Fixed macro overlap telemetry: stage=post_legalization "
                    "macro_count=%d overlap_area=%.6E overlap_area_ratio=%.6E "
                    "overlap_cells=%d overlap_pairs=%d max_overlap_area=%.6E "
                    "legalization_moved=%d legalization_disp_max=%.6E "
                    "legalization_disp_mean=%.6E legalization_disp_sum=%.6E",
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_macro_count", 0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_area", 0.0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_area_ratio", 0.0
                        )
                    ),
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_cell_count", 0
                        )
                    ),
                    int(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_pair_count", 0
                        )
                    ),
                    float(
                        fixed_macro_overlap_stats.get(
                            "fixed_macro_overlap_max_area", 0.0
                        )
                    ),
                    int(
                        displacement_stats.get(
                            "movable_displacement_moved_count", 0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_max", 0.0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_mean", 0.0
                        )
                    ),
                    float(
                        displacement_stats.get(
                            "movable_displacement_sum", 0.0
                        )
                    ),
                )
            except Exception as e:
                logging.warning(
                    "Failed to compute fixed macro overlap telemetry after legalization: %s",
                    e,
                )
            if getattr(params, "egr_padding_flag", 0):
                self._apply_egr_padding(params, placedb)
                iteration = self._run_standard_legalization(
                    params, placedb, iteration, all_metrics
                )
                self._restore_egr_padding()

            # Perform any configured post-legalization STA on the final
            # legalized position. The public Placer rejects the removed
            # OpenTimer mode before this path is reachable.
            if params.timing_opt_flag:
                logging.info("additional sta after legalization")
                timing_op = self.op_collections.timing_op
                timing_op(self.pos[0].data.clone().cpu())
                timing_op.timer.update_timing()
                cur_metric = all_metrics[-1]
                cur_metric.tns = timing_op.timer.report_tns_elw(
                    split=1
                ) / (time_unit * 1e17)
                cur_metric.wns = timing_op.timer.report_wns(
                    split=1
                ) / (time_unit * 1e15)

        # after_legalization recover node sizes, pins shifts, and positions of cells
        if params.cell_padding_x >= 0:
            with torch.no_grad():
                # node sizes
                self.data_collections.node_size_x[:placedb.num_movable_nodes] -= (
                    2 * params.cell_padding_x
                )
                # self.data_collections.node_size_y[:placedb.num_movable_nodes] -= (
                #     2 * params.cell_padding_y
                # )
                movable_cell_tensor = np.arange(
                    0, placedb.num_movable_nodes, dtype=placedb.pin2node_map.dtype)
                # shift macro pins
                movable_cell_pins = np.isin(
                    placedb.pin2node_map, movable_cell_tensor)

                # pin offsets
                self.data_collections.pin_offset_x[movable_cell_pins] -= params.cell_padding_x
                # self.data_collections.pin_offset_y -= params.cell_padding_y

                self.pos[0][:placedb.num_movable_nodes] += params.cell_padding_x
                params.cell_padding_x = 0
                # self.pos[0][
                #     placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
                # ] += params.cell_padding_y

        # # rescale everything
        # cur_scale_factor = self.data_collections.fp_info.scale_factor
        # gcd_site_scale_factor = 1 / math.gcd(
        #     placedb.origin_site_width, placedb.origin_row_height
        # )
        # if cur_scale_factor != gcd_site_scale_factor:
        #     logging.warn(
        #         f"Rescaling by GCD(site_width, row_height) = {gcd_site_scale_factor} before legalization and detailed placement"
        #     )
        #     params.scale_factor = gcd_site_scale_factor
        #     rescale_factor = gcd_site_scale_factor / cur_scale_factor
        #     with torch.no_grad():
        #         self.pos[0].mul_(rescale_factor).round_()
        #         self.data_collections.node_size_x.mul_(rescale_factor).round_()
        #         self.data_collections.node_size_y.mul_(rescale_factor).round_()
        #         self.data_collections.flat_region_boxes.mul_(
        #             rescale_factor).round_()
        #         self.data_collections.pin_offset_x.mul_(rescale_factor)
        #         self.data_collections.pin_offset_y.mul_(rescale_factor)
        #         # self.data_collections.node_areas.mul_(rescale_factor * rescale_factor)
        #         self.data_collections.fp_info.scale(rescale_factor)
        #         self.data_collections.fp_info.scale_factor = gcd_site_scale_factor
        #         params.macro_halo_x *= rescale_factor
        #         params.macro_halo_y *= rescale_factor
        #         params.macro_pin_halo_x *= rescale_factor
        #         params.macro_pin_halo_y *= rescale_factor
        #         params.cell_padding_x *= rescale_factor
        #         params.cell_padding_y *= rescale_factor
        #         # TODO: rescale fence regions

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration,
                      self.pos[0].data.clone().cpu().numpy())

        # dump legalization solution for detailed placement
        if params.dump_legalize_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(
            ), "%s.dp.pklz" % (params.design_name()))

        # detailed placement
        if params.detailed_place_flag:
            tt = time.time()
            self.pos[0].data.copy_(
                self.op_collections.detailed_place_op(self.pos[0]))
            logging.info("detailed placement takes %.3f seconds" %
                         (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        if self.op_collections.m2_pa_refine_op is not None:
            refined_pos, _ = self.op_collections.m2_pa_refine_op(self.pos[0])
            self.pos[0].data.copy_(refined_pos)
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
            )
            logging.info(cur_metric)
            iteration += 1

        if (getattr(params, "post_legalization_adaptive_padding_flag", 0)
                and not params.global_place_flag
                and int(getattr(params, "post_legalization_padding_rounds", 1)) > 1):
            from dreamplace.ops.routability.iterative_legalization_padding import (
                run_iterative_legalization_padding,
            )
            adaptive_padding_pos, adaptive_padding_stats = (
                run_iterative_legalization_padding(params, placedb, self.pos[0], self)
            )
        else:
            adaptive_padding_pos, adaptive_padding_stats = (
                route_evaluation.run_post_legalization_adaptive_padding(
                    params, placedb, self.pos[0], self
                )
            )
        if adaptive_padding_stats is not None:
            self.pos[0].data.copy_(adaptive_padding_pos)
            for key, value in adaptive_padding_stats.items():
                processed_metrics[
                    "post_legalization_adaptive_padding_%s" % key
                ] = value
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(
                placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
            )
            logging.info(cur_metric)
            iteration += 1

        # save results
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution
        placedb.apply(
            params,
            cur_pos[0: placedb.num_movable_nodes],
            cur_pos[placedb.num_nodes: placedb.num_nodes +
                    placedb.num_movable_nodes],
        )

        # update pin offsets of std cells
        # assume rows are FS = 0, N = 1, FS, ...
        # cur_orient = torch.from_numpy(
        #     np.where(placedb.node_orient == b"N", 1, 0)[: placedb.num_movable_nodes]
        # ).to(self.pos[0].device)
        # new_orient = (
        #     torch.div(
        #         self.pos[0][
        #             placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
        #         ],
        #         self.data_collections.fp_info.row_height,
        #         rounding_mode="floor",
        #     )
        #     % 2
        # )
        # flips = (
        #     (cur_orient != new_orient) & ~self.data_collections.movable_macro_mask
        # ).nonzero().view(-1)
        # self.data_collections.pin_offset_y[flips] = (
        #     self.data_collections.fp_info.row_height
        #     - self.data_collections.pin_offset_y[flips]
        # )

        # reset net weights
        self.data_collections.net_weights.fill_(1.0)

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, cur_pos)

        # update pin offsets of std cells
        # assume rows are FS = 0, N = 1, FS, ...
        # cur_orient = torch.from_numpy(
        #     np.where(placedb.node_orient == b"N", 1, 0)[: placedb.num_movable_nodes]
        # ).to(self.pos[0].device)
        # new_orient = (
        #     torch.div(
        #         self.pos[0][
        #             placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
        #         ],
        #         self.data_collections.fp_info.row_height,
        #         rounding_mode="floor",
        #     )
        #     % 2
        # )
        # flips = (
        #     (cur_orient != new_orient) & ~self.data_collections.movable_macro_mask
        # ).nonzero().view(-1)
        # self.data_collections.pin_offset_y[flips] = (
        #     self.data_collections.fp_info.row_height
        #     - self.data_collections.pin_offset_y[flips]
        # )

        # reset net weights
        self.data_collections.net_weights.fill_(1.0)

        # run RSMT
        if params.with_sta:
            with torch.no_grad():
                tt = time.time()
                self.data_collections.net_flat_topo_sort, self.data_collections.net_flat_topo_sort_start, \
                    self.data_collections.pin_fa, self.data_collections.flat_pin_to, self.data_collections.flat_pin_to_start, \
                    self.data_collections.flat_pin_from = self.op_collections.steiner_topo_op.rebuild_tree(
                        self.op_collections.pin_pos_op(self.pos[0]))
                new_x, new_y = self.op_collections.steiner_topo_op(
                    self.op_collections.pin_pos_op(self.pos[0])
                )
                self.data_collections.net_flat_topo_sort,
                self.data_collections.net_flat_topo_sort_start,
                self.data_collections.pin_fa,
                self.data_collections.flat_pin_to_start,
                flat_pin_to = self.data_collections.flat_pin_to
                flat_pin_from = self.data_collections.flat_pin_from
                
                length = (torch.abs(new_x[flat_pin_from] - new_x[flat_pin_to])
                        + torch.abs(new_y[flat_pin_from] - new_y[flat_pin_to])) / params.scale_factor  / placedb.dbu
                flute_length_path = "%s/%s_flute_length.txt" % (
                    params.result_dir, params.design_name())
                # with open(flute_length_path, "w") as f:
                #     f.write("net_name, flute_length (um), cap (pF)\n")
                #     for net_id in range(placedb.num_nets):
                #         start = self.data_collections.net_flat_topo_sort_start[net_id]
                #         end = self.data_collections.net_flat_topo_sort_start[net_id + 1]
                #         net_name = placedb.net_names[net_id]
                #         net_flat_pins_nodes = self.data_collections.net_flat_topo_sort[start:end]
                #         net_flat_pin_start = self.data_collections.flat_pin_to_start[net_flat_pins_nodes]
                #         net_flat_pin_end = self.data_collections.flat_pin_to_start[net_flat_pins_nodes + 1]
                #         net_length = 0
                #         for i in range(len(net_flat_pins_nodes)):
                #             pin_start = net_flat_pin_start[i]
                #             pin_end = net_flat_pin_end[i]
                #             net_length += length[pin_start:pin_end].sum().item()
                #         # net_length = length[start:end].sum().item()
                #         f.write(f"{net_name}, {net_length}, {placedb.c_unit * net_length}\n")

                wns, tns, ws, ts = model.timing_obj(self.pos[0])
                model.check_log(wns, tns, ws, ts)
                logging.info("rsmt computation takes %.3f seconds" %
                            (time.time() - tt))

        # get HPWL
        with torch.no_grad():
            hpwl = self.op_collections.hpwl_op(self.pos[0])
            rsmt_wl = self.op_collections.rsmt_wl_op(self.pos[0]) / placedb.dbu
            logging.info("flute rsmt %.6E um" % rsmt_wl)
            logging.info("unweighted hpwl %.6E" % hpwl)

        routability_controller.finalize(
            model=self._routability_model,
            position=self.pos[0],
        )
        route_evaluation.run_gpugr_final_eval(params, placedb, self.pos[0])

        # save nets degree, RSMT, HPWL
        # with torch.no_grad():
        #     degrees = torch.from_numpy(np.ediff1d(placedb.flat_net2pin_start_map))
        #     mask = torch.logical_and(2 <= degrees, degrees < params.ignore_net_degree)
        #     degrees = degrees[mask].long()
        #     steiners = self.op_collections.rsmt_wl_op(self.pos[0], False)[mask]
        #     wirelengths = (
        #         self.op_collections.hpwl_op(self.pos[0], False)
        #         .cpu()
        #         .detach()[mask]
        #     )
        #     weights = steiners / wirelengths
        #     # get new RISA weights
        #     degrees, indices = torch.sort(degrees)
        #     weights = weights[indices]
        #     c = torch.stack((degrees, weights))
        #     idxs, vals = torch.unique(c[0, :], return_counts=True)
        #     vs = torch.split_with_sizes(c[1, :], tuple(vals))
        #     weights_dict = {int(k.item()): float(v.mean()) for k, v in zip(idxs, vs)}
        #     path = "%s/%s" % (params.result_dir, params.design_name())
        #     with open("%s/risa_weights.pkl" % path, "wb") as f:
        #         pickle.dump(weights_dict, f)

        return float(rsmt_wl), float(hpwl), processed_metrics
