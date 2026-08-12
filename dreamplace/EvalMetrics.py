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
# @file   EvalMetrics.py
# @author Yibo Lin
# @date   Sep 2018
# @brief  Evaluation metrics
#

import time
import torch
import pdb


def _as_log_verbose(value, default=0):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "0", "false", "no", "off"):
            return 0
        if normalized in ("true", "yes", "on"):
            return 1
        try:
            return int(float(normalized))
        except ValueError:
            return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class EvalMetrics(object):
    """
    @brief evaluation metrics at one step
    """

    def __init__(self, iteration=None, detailed_step=None):
        """
        @brief initialization
        @param iteration optimization step
        """
        self.iteration = iteration
        self.detailed_step = detailed_step
        self.objective = None
        self.wirelength = None
        self.density = None
        self.density_weight = None
        self.hpwl = None
        self.rsmt_wl = None
        self.overflow = None
        self.goverflow = None
        self.route_utilization = None
        self.pin_utilization = None
        self.max_density = None
        self.gmax_density = None
        self.gamma = None
        self.eval_time = None
        self.weight_hpwl = None
        self.macro_overlap = None
        self.macro_overlap_weight = None
        self.l_shape_fast_mode = None
        self.l_shape_energy_valid = None
        self.l_shape_cost = None
        self.l_shape_weighted_cost = None
        self.l_shape_weight = None
        self.l_shape_target_weight = None
        self.l_shape_weight_candidate = None
        self.l_shape_cap_active = None
        self.l_shape_base_grad_norm = None
        self.l_shape_grad_raw_norm = None
        self.l_shape_grad_norm = None
        self.l_shape_grad_ratio = None
        self.l_shape_target_ratio = None
        self.l_shape_overflow = None
        self.l_shape_overflow_ratio = None
        self.l_shape_overflow_ema = None
        self.l_shape_overflow_max_density = None
        self.l_shape_capacity_al_enabled = None
        self.l_shape_capacity_al_updated = None
        self.l_shape_capacity_al_g_h_max = None
        self.l_shape_capacity_al_g_v_max = None
        self.l_shape_capacity_al_g_h_sum = None
        self.l_shape_capacity_al_g_v_sum = None
        self.l_shape_capacity_al_g_h_pos_ratio = None
        self.l_shape_capacity_al_g_v_pos_ratio = None
        self.l_shape_capacity_al_q_h_max = None
        self.l_shape_capacity_al_q_v_max = None
        self.l_shape_capacity_al_q_h_sum = None
        self.l_shape_capacity_al_q_v_sum = None
        self.l_shape_capacity_al_lambda_h_max = None
        self.l_shape_capacity_al_lambda_v_max = None
        self.l_shape_capacity_al_lambda_h_sum = None
        self.l_shape_capacity_al_lambda_v_sum = None
        self.l_shape_capacity_al_energy_h = None
        self.l_shape_capacity_al_energy_v = None
        self.l_shape_capacity_al_energy_total = None
        self.l_shape_capacity_al_pq_h_min = None
        self.l_shape_capacity_al_pq_h_max = None
        self.l_shape_capacity_al_pq_v_min = None
        self.l_shape_capacity_al_pq_v_max = None
        self.l_shape_capacity_al_active_memory_bins_h = None
        self.l_shape_capacity_al_active_memory_bins_v = None
        self.l_shape_macro_exclusion_enabled = None
        self.l_shape_macro_exclusion_macro_count = None
        self.l_shape_macro_exclusion_body_bins = None
        self.l_shape_macro_exclusion_halo_bins = None
        self.l_shape_macro_exclusion_active_bins = None
        self.l_shape_macro_exclusion_source_max = None
        self.l_shape_macro_exclusion_source_sum = None
        self.l_shape_macro_exclusion_body_source_max = None
        self.l_shape_macro_exclusion_body_source_sum = None
        self.l_shape_macro_exclusion_halo_source_max = None
        self.l_shape_macro_exclusion_halo_source_sum = None
        self.l_shape_macro_exclusion_usage_max = None
        self.l_shape_macro_exclusion_usage_sum = None
        self.l_shape_macro_exclusion_usage_bins = None
        self.l_shape_macro_exclusion_dominates_bins = None
        self.l_shape_macro_exclusion_routing_dominates_bins = None
        self.l_shape_log_verbose = 0
        self.soft_l_diag_count = None
        self.soft_l_mean_cost_gap = None
        self.soft_l_raw_cost_gap_p50 = None
        self.soft_l_biased_cost_gap_p50 = None
        self.soft_l_tau_source_gap = None
        self.soft_l_mean_max_prob = None
        self.soft_l_mean_entropy = None
        self.soft_l_near_tie_ratio = None
        self.soft_l_tau = None
        self.soft_l_effective_hotspot_weight = None
        self.soft_l_resolver_agreement_ratio = None
        self.soft_l_target_demand_supply_ratio = None
        self.soft_l_current_demand_supply_ratio = None
        self.soft_l_same_net_topo_nets = None
        self.soft_l_same_net_topo_segments_h = None
        self.soft_l_same_net_topo_segments_v = None
        self.soft_l_same_net_topo_diag_edges = None
        self.soft_l_same_net_topo_edges_with_topology = None
        self.soft_l_same_net_topo_edges_with_observed_intervals = None
        self.soft_l_same_net_topo_mean_gap = None
        self.soft_l_same_net_topo_tie_ratio = None

    def __str__(self):
        """
        @brief convert to string
        """
        content = ""
        if self.iteration is not None:
            content = "iteration %4d" % (self.iteration)
        if self.detailed_step is not None:
            content += ", (%4d, %2d, %2d)" % (
                self.detailed_step[0],
                self.detailed_step[1],
                self.detailed_step[2],
            )
        if self.objective is not None:
            content += ", Obj %.6E" % (self.objective)
        if self.wirelength is not None:
            content += ", WL %.3E" % (self.wirelength)
        if self.density is not None:
            if self.density.numel() == 1:
                content += ", Density %.3E" % (self.density)
            else:
                content += ", Density [%s]" % ", ".join(
                    ["%.3E" % i for i in self.density]
                )
        if self.density_weight is not None:
            if self.density_weight.numel() == 1:
                content += ", DensityWeight %.6E" % (self.density_weight)
            else:
                content += ", DensityWeight [%s]" % ", ".join(
                    ["%.3E" % i for i in self.density_weight]
                )
        if self.hpwl is not None:
            content += ", HPWL %.6E" % (self.hpwl)
        if self.weight_hpwl is not None:
            content += ", weight HPWL %.6E" % (self.weight_hpwl)
        if self.rsmt_wl is not None:
            content += ", RSMT %.3E" % (self.rsmt_wl)
        if self.overflow is not None:
            if self.overflow.numel() == 1:
                content += ", Overflow %.6E" % (self.overflow)
            else:
                content += ", Overflow [%s]" % ", ".join(
                    ["%.3E" % i for i in self.overflow]
                )
        if self.goverflow is not None:
            content += ", Global Overflow %.6E" % (self.goverflow)
        if self.max_density is not None:
            if self.max_density.numel() == 1:
                content += ", MaxDensity %.3E" % (self.max_density)
            else:
                content += ", MaxDensity [%s]" % ", ".join(
                    ["%.3E" % i for i in self.max_density]
                )
        if self.route_utilization is not None:
            content += ", RouteOverflow %.6E" % (self.route_utilization)
        if self.pin_utilization is not None:
            content += ", PinOverflow %.6E" % (self.pin_utilization)
        if self.macro_overlap is not None:
            content += ", MacroOverlap %.3E" % (self.macro_overlap)
        if self.macro_overlap_weight is not None:
            content += ", MacroOverlapWeight %.6E" % (
                self.macro_overlap_weight)
        l_shape_energy_invalid = (
            self.l_shape_fast_mode is not None
            and bool(self.l_shape_fast_mode)
            and self.l_shape_energy_valid is not None
            and not bool(self.l_shape_energy_valid)
        )
        if l_shape_energy_invalid:
            content += ", LShapeCostRaw N/A(fast_mode)"
            content += ", LShapeCostWeighted N/A(fast_mode)"
        elif self.l_shape_cost is not None:
            content += ", LShapeCostRaw %.6E" % (self.l_shape_cost)
            if self.l_shape_weighted_cost is not None:
                content += ", LShapeCostWeighted %.6E" % (self.l_shape_weighted_cost)
        elif self.l_shape_weighted_cost is not None:
            content += ", LShapeCostWeighted %.6E" % (self.l_shape_weighted_cost)
        if _as_log_verbose(self.l_shape_log_verbose) >= 2:
            if self.l_shape_weight is not None:
                content += ", LWeight %.6E" % (self.l_shape_weight)
            if self.l_shape_weight_candidate is not None:
                content += ", LWCandidate %.6E" % (self.l_shape_weight_candidate)
            if self.l_shape_target_weight is not None:
                content += ", LWCap %.6E" % (self.l_shape_target_weight)
            if self.l_shape_cap_active is not None:
                content += ", LWCapAct %d" % (1 if self.l_shape_cap_active else 0)
            if self.l_shape_base_grad_norm is not None:
                content += ", LBaseGrad %.6E" % (self.l_shape_base_grad_norm)
            if self.l_shape_grad_norm is not None:
                content += ", LGrad %.6E" % (self.l_shape_grad_norm)
            if self.l_shape_grad_raw_norm is not None:
                content += ", LGradRaw %.6E" % (self.l_shape_grad_raw_norm)
            if self.l_shape_grad_ratio is not None:
                content += ", LGradRatio %.4f" % (self.l_shape_grad_ratio)
            if self.l_shape_target_ratio is not None:
                content += ", LTargetRatio %.4f" % (self.l_shape_target_ratio)
            if self.l_shape_overflow is not None:
                content += ", LOvRaw %.6E" % (self.l_shape_overflow)
            if self.l_shape_overflow_ratio is not None:
                content += ", LOvRatio %.6E" % (self.l_shape_overflow_ratio)
            if self.l_shape_overflow_ema is not None:
                content += ", LOvEma %.6E" % (self.l_shape_overflow_ema)
            if self.l_shape_overflow_max_density is not None:
                content += ", LMaxDen %.6E" % (self.l_shape_overflow_max_density)
            if self.l_shape_capacity_al_energy_total is not None:
                content += ", LCapALE %.6E" % (
                    self.l_shape_capacity_al_energy_total
                )
            if self.l_shape_capacity_al_q_h_max is not None:
                content += ", LCapALQHMax %.6E" % (
                    self.l_shape_capacity_al_q_h_max
                )
            if self.l_shape_capacity_al_lambda_h_max is not None:
                content += ", LCapALLamHMax %.6E" % (
                    self.l_shape_capacity_al_lambda_h_max
                )
            if self.l_shape_capacity_al_g_h_sum is not None:
                content += ", LCapALGHSum %.6E" % (
                    self.l_shape_capacity_al_g_h_sum
                )
            if self.l_shape_capacity_al_g_h_pos_ratio is not None:
                content += ", LCapALGHRatio %.4f" % (
                    self.l_shape_capacity_al_g_h_pos_ratio
                )
            if self.l_shape_capacity_al_g_v_sum is not None:
                content += ", LCapALGVSum %.6E" % (
                    self.l_shape_capacity_al_g_v_sum
                )
            if self.l_shape_capacity_al_g_v_pos_ratio is not None:
                content += ", LCapALGVRatio %.4f" % (
                    self.l_shape_capacity_al_g_v_pos_ratio
                )
            if self.l_shape_macro_exclusion_macro_count is not None:
                content += ", LMacroCnt %d" % (
                    self.l_shape_macro_exclusion_macro_count
                )
            if self.l_shape_macro_exclusion_active_bins is not None:
                content += ", LMacroBins %d" % (
                    self.l_shape_macro_exclusion_active_bins
                )
            if self.l_shape_macro_exclusion_source_max is not None:
                content += ", LMacroMax %.6E" % (
                    self.l_shape_macro_exclusion_source_max
                )
            if self.l_shape_macro_exclusion_usage_sum is not None:
                content += ", LMacroUsage %.6E" % (
                    self.l_shape_macro_exclusion_usage_sum
                )
            if self.l_shape_macro_exclusion_usage_bins is not None:
                content += ", LMacroUsageBins %d" % (
                    self.l_shape_macro_exclusion_usage_bins
                )
            if self.soft_l_diag_count is not None:
                content += ", SoftDiag %d" % (self.soft_l_diag_count)
            if self.soft_l_mean_cost_gap is not None:
                content += ", SoftGapMean %.4f" % (self.soft_l_mean_cost_gap)
            if self.soft_l_raw_cost_gap_p50 is not None:
                content += ", SoftGapP50Raw %.8f" % (self.soft_l_raw_cost_gap_p50)
            if self.soft_l_biased_cost_gap_p50 is not None:
                content += ", SoftGapP50Bias %.8f" % (self.soft_l_biased_cost_gap_p50)
            if self.soft_l_tau_source_gap is not None:
                content += ", SoftTauSrc %.8f" % (self.soft_l_tau_source_gap)
            if self.soft_l_mean_max_prob is not None:
                content += ", SoftConf %.4f" % (self.soft_l_mean_max_prob)
            if self.soft_l_mean_entropy is not None:
                content += ", SoftEnt %.4f" % (self.soft_l_mean_entropy)
            if self.soft_l_near_tie_ratio is not None:
                content += ", SoftTie %.4f" % (self.soft_l_near_tie_ratio)
            if self.soft_l_tau is not None:
                content += ", SoftTau %.8f" % (self.soft_l_tau)
            if self.soft_l_effective_hotspot_weight is not None:
                content += ", SoftHot %.4f" % (self.soft_l_effective_hotspot_weight)
            if self.soft_l_resolver_agreement_ratio is not None:
                content += ", SoftAgree %.4f" % (
                    self.soft_l_resolver_agreement_ratio
                )
            if self.soft_l_target_demand_supply_ratio is not None:
                content += ", SoftDSRatio %.4f" % (
                    self.soft_l_target_demand_supply_ratio
                )
            if self.soft_l_current_demand_supply_ratio is not None:
                content += ", SoftLoadRatio %.4f" % (
                    self.soft_l_current_demand_supply_ratio
                )
            if self.soft_l_same_net_topo_nets is not None:
                content += ", TopoNets %d" % (self.soft_l_same_net_topo_nets)
            if self.soft_l_same_net_topo_diag_edges is not None:
                content += ", TopoDiag %d" % (self.soft_l_same_net_topo_diag_edges)
            if self.soft_l_same_net_topo_edges_with_topology is not None:
                content += ", TopoHit %d" % (
                    self.soft_l_same_net_topo_edges_with_topology
                )
            if self.soft_l_same_net_topo_edges_with_observed_intervals is not None:
                content += ", TopoObs %d" % (
                    self.soft_l_same_net_topo_edges_with_observed_intervals
                )
            if self.soft_l_same_net_topo_mean_gap is not None:
                content += ", TopoGap %.4f" % (self.soft_l_same_net_topo_mean_gap)
            if self.soft_l_same_net_topo_tie_ratio is not None:
                content += ", TopoTie %.4f" % (self.soft_l_same_net_topo_tie_ratio)
        if self.gamma is not None:
            content += ", gamma %.6E" % (self.gamma)
        if self.eval_time is not None:
            content += ", time %.3fms" % (self.eval_time * 1000)

        return content

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def evaluate(self, placedb, ops, var, data_collections=None):
        """
        @brief evaluate metrics
        @param placedb placement database
        @param ops a list of ops
        @param var variables
        @param data_collections placement data collections
        """
        tt = time.time()
        with torch.no_grad():
            if "objective" in ops:
                self.objective = ops["objective"](var).data
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data
            if "density" in ops:
                self.density = ops["density"](var).data
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data
            if "weight_hpwl" in ops:
                self.weight_hpwl = ops["weight_hpwl"](var).data
            if "rsmt_wl" in ops:
                self.rsmt_wl = ops["rsmt_wl"](var).data
            if "overflow" in ops:
                overflow, max_density = ops["overflow"](var)
                if overflow.numel() == 1:
                    self.overflow = overflow.data / placedb.total_movable_node_area
                    self.max_density = max_density.data
                else:
                    self.overflow = (
                        overflow.data
                        / data_collections.total_movable_node_area_fence_region
                    )
                    self.max_density = max_density.data
            if "goverflow" in ops:
                overflow, max_density = ops["goverflow"](var)
                self.goverflow = overflow.data / placedb.total_movable_node_area
                self.gmax_density = max_density.data
            if "route_utilization" in ops:
                route_utilization_map = ops["route_utilization"](var)
                route_utilization_map_sum = route_utilization_map.sum()
                self.route_utilization = (
                    route_utilization_map.sub_(1).clamp_(min=0).sum()
                    / route_utilization_map_sum
                )
            if "pin_utilization" in ops:
                pin_utilization_map = ops["pin_utilization"](var)
                pin_utilization_map_sum = pin_utilization_map.sum()
                self.pin_utilization = (
                    pin_utilization_map.sub_(1).clamp_(min=0).sum()
                    / pin_utilization_map_sum
                )
            if "macro_overlap" in ops:
                self.macro_overlap = ops["macro_overlap"](var).data
        self.eval_time = time.time() - tt
