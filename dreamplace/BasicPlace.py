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
# @file   BasicPlace.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Base placement class
#

from dataclasses import dataclass, fields
import os
from pathlib import Path
import sys
import time
import gzip

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import re
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dreamplace.ops.move_boundary.move_boundary as move_boundary
import dreamplace.ops.hpwl.hpwl as hpwl
import dreamplace.ops.rmst_wl.rmst_wl as rmst_wl
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
import dreamplace.ops.legality_check.legality_check as legality_check
import dreamplace.ops.draw_place.draw_place as draw_place
import dreamplace.ops.pin_pos.pin_pos as pin_pos
import dreamplace.ops.global_swap.global_swap as global_swap
import dreamplace.ops.k_reorder.k_reorder as k_reorder
import dreamplace.ops.independent_set_matching.independent_set_matching as independent_set_matching
import dreamplace.ops.pin_weight_sum.pin_weight_sum as pws
import dreamplace.ops.irt_egr.irt_egr as irt_egr
import dreamplace.ops.m2_legalize.m2_rail_legalization as m2_rail_legalization
# import dreamplace.ops.pin_weight_sum.pin_weight_sum as pws
# import dreamplace.ops.timing.timing as timingimport
import dreamplace.ops.steiner_topo.steiner_topo as steiner_topo
from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
from dreamplace.ops.gate_projection.gate_projection import (
    ArgminProjectionResolver,
    GateProjectionOp,
    NearestSizeProjectionResolver,
    StableCurrentCellResolver,
    VectorizedMainIdCandidateProvider,
)
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
from dreamplace.ops.timing_propagation.timing_propagation import ARCS_INFO, LUTS_INFO
import pdb

DEFAULT_SIZE_INIT_NORM_LOWER_CLIP = 1e-4
SIZE_INIT_NORM_LOWER_CLIP_ENV = "AIMP_SIZE_INIT_NORM_LOWER_CLIP"
SIZE_PARAMETERIZATIONS = ("logits", "real_size")


def resolve_size_init_norm_lower_clip(default=DEFAULT_SIZE_INIT_NORM_LOWER_CLIP):
    raw_value = os.environ.get(SIZE_INIT_NORM_LOWER_CLIP_ENV, "").strip()
    if not raw_value:
        return default
    try:
        lower_clip = float(raw_value)
    except ValueError:
        logging.warning(
            "Ignore invalid %s=%r; fallback to default %.6g",
            SIZE_INIT_NORM_LOWER_CLIP_ENV,
            raw_value,
            default,
        )
        return default
    if not 0.0 < lower_clip < 0.5:
        logging.warning(
            "Ignore out-of-range %s=%r; fallback to default %.6g",
            SIZE_INIT_NORM_LOWER_CLIP_ENV,
            raw_value,
            default,
        )
        return default
    logging.info(
        "Override size init norm lower clip to %.6g via %s",
        lower_clip,
        SIZE_INIT_NORM_LOWER_CLIP_ENV,
    )
    return lower_clip


def compute_init_size_logits(
    inst_size_init,
    inst_size_lower,
    inst_size_upper,
    dtype,
    lower_clip=None,
):
    # Historical "size" variables now represent the unified timing coordinate.
    if lower_clip is None:
        lower_clip = resolve_size_init_norm_lower_clip()
    size_denom = np.maximum(
        inst_size_upper - inst_size_lower,
        np.finfo(dtype).eps,
    )
    size_norm = np.clip(
        (inst_size_init - inst_size_lower) / size_denom,
        lower_clip,
        1.0 - lower_clip,
    )
    return np.log(size_norm / (1.0 - size_norm)).astype(dtype)


def compute_effective_initial_size(
    inst_size_init,
    inst_size_lower,
    inst_size_upper,
    dtype,
    lower_clip=None,
):
    """Return the bounded continuous size coordinate used at initialization."""
    if lower_clip is None:
        lower_clip = resolve_size_init_norm_lower_clip()
    init = np.asarray(inst_size_init, dtype=dtype)
    lower = np.asarray(inst_size_lower, dtype=dtype)
    upper = np.asarray(inst_size_upper, dtype=dtype)
    if init.shape != lower.shape or init.shape != upper.shape:
        raise ValueError("size initialization and bounds must have identical shapes")
    if not np.all(np.isfinite(init)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("size initialization and bounds must be finite")
    if np.any(upper < lower):
        raise ValueError("inst_size_upper must be greater than or equal to inst_size_lower")
    size_denom = np.maximum(upper - lower, np.finfo(dtype).eps)
    size_norm = np.clip(
        (init - lower) / size_denom,
        lower_clip,
        1.0 - lower_clip,
    )
    return (lower + size_norm * (upper - lower)).astype(dtype)


def size_var_to_logits(size_var, inst_size_lower, inst_size_upper, lower_clip=None):
    """Convert a continuous size tensor to the bounded legacy logit coordinate."""
    if lower_clip is None:
        lower_clip = DEFAULT_SIZE_INIT_NORM_LOWER_CLIP
    size_var = torch.as_tensor(size_var)
    lower = torch.as_tensor(inst_size_lower, device=size_var.device, dtype=size_var.dtype)
    upper = torch.as_tensor(inst_size_upper, device=size_var.device, dtype=size_var.dtype)
    if size_var.shape != lower.shape or size_var.shape != upper.shape:
        raise ValueError("size value and bounds must have identical shapes")
    if bool(torch.any(upper < lower)):
        raise ValueError("inst_size_upper must be greater than or equal to inst_size_lower")
    denom = upper - lower
    valid = denom > 0
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    normalized = (size_var - lower) / safe_denom
    normalized = torch.clamp(normalized, min=float(lower_clip), max=1.0 - float(lower_clip))
    logits = torch.log(normalized) - torch.log1p(-normalized)
    return torch.where(valid, logits, torch.zeros_like(logits))


def _plot_state(placedb):
    state = getattr(placedb, "_dreamplace_plot_clock", None)
    if state is None:
        state = {
            "generation": None,
            "next_global": 0,
            "last_local": None,
        }
        setattr(placedb, "_dreamplace_plot_clock", state)
    return state


def _plot_generation(placedb):
    generation = getattr(placedb, "runtimedb_generation", None)
    if generation is None:
        session = getattr(placedb, "handoff_session", None)
        generation = getattr(session, "runtimedb_generation", 0)
    try:
        return int(generation)
    except (TypeError, ValueError):
        return 0


def next_plot_filename(path, placedb, local_iteration, tag=None):
    state = _plot_state(placedb)
    generation = _plot_generation(placedb)
    local_iteration = int(local_iteration)
    if state["generation"] != generation:
        state["generation"] = generation
        state["last_local"] = None

    preferred_global = local_iteration
    if local_iteration == 9999 or tag:
        preferred_global = state["next_global"]
    global_iteration = max(state["next_global"], preferred_global)
    state["next_global"] = global_iteration + 1
    state["last_local"] = local_iteration

    suffix = ""
    if tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(tag)).strip("_")
        if safe_tag:
            suffix = "_" + safe_tag
    return "%s/plot/iter%06d_g%02d_local%04d%s.png" % (
        path,
        global_iteration,
        generation,
        local_iteration,
        suffix,
    )


@dataclass
class FloorplanInfo:
    xl: float
    yl: float
    xh: float
    yh: float
    site_width: float
    row_height: float
    scale_factor: float
    routing_grid_xl: float
    routing_grid_yl: float
    routing_grid_xh: float
    routing_grid_yh: float
    routing_V: float
    routing_H: float
    macro_util_V: torch.Tensor
    macro_util_H: torch.Tensor
    cell_padding_x: float
    # cell_padding_y: float
    bndry_padding_x: float
    bndry_padding_y: float

    def scale(self, factor):
        for field in fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, value * factor)


class PlaceDataCollection(object):
    """
    @brief A wraper for all data tensors on device for building ops
    """

    def _initialize_buffering_database_handles(self, placedb):
        pydb = getattr(placedb, "pydb", None)
        self.pydb = pydb
        self.buffering_metadata = pydb if pydb is not None else placedb
        self.dbu = int(getattr(placedb, "dbu", getattr(pydb, "dbu", 1000)) or 1000)
        self.buffer_main_type_index = int(getattr(placedb, "buffer_main_type_index", -1))
        self.buffer_main_type_status = str(getattr(placedb, "buffer_main_type_status", "unsupported"))
        if self.buffering_metadata is not None:
            for field_name in ("buffer_main_type_index", "buffer_main_type_status"):
                if not hasattr(self.buffering_metadata, field_name) and hasattr(placedb, field_name):
                    try:
                        setattr(self.buffering_metadata, field_name, getattr(placedb, field_name))
                    except AttributeError:
                        pass

    def __init__(self, pos, params, placedb, device):
        """
        @brief initialization
        @param pos locations of cells
        @param params parameters
        @param placedb placement database
        @param device cpu or cuda
        """
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter
        self.pos = pos
        self.sizing_parameterization = str(
            getattr(params, "sizing_parameterization", "logits") or "logits"
        ).strip().lower()
        if self.sizing_parameterization not in SIZE_PARAMETERIZATIONS:
            raise ValueError(
                "unsupported sizing_parameterization: "
                f"{self.sizing_parameterization!r}"
            )
        self.size_logits = None
        self.real_size = None
        self.effective_initial_size = None
        self.continuous_size_trainable_mask = None
        self.vt_logits = None
        self.buffer_optimization_state = None
        self.buffer_bu_logits = None
        self.buffer_z_param = None
        self.buffer_bsu_index_param = None
        self.buffer_segment_timing_topology_epoch = 0
        self.buffer_segment_timing_topology_epoch_reason = "initial"
        self.runtime_cell_state_generation = 0
        self.timing_model_generation = 0
        # Disk cache for the cell surrogate fit is opt-in; default off so runs
        # never write a CWD-relative surrogate_cache directory.
        self.surrogate_cache_root = getattr(params, "surrogate_cache_root", None) or None
        self.cell_modeling_num_threads = int(getattr(params, "cell_modeling_num_threads", 1))
        self.cell_model_schema = getattr(params, "cell_model_schema", None)
        self.piecewise_gradient_mode = getattr(
            params, "piecewise_gradient_mode", "native_piecewise"
        )
        self.cell_model_lut_boundary_mode = getattr(params, "cell_model_lut_boundary_mode", None)
        self.projection_resolver = getattr(params, "projection_resolver", "argmin")
        self.size_interpolated_pin_native_op = getattr(
            params, "size_interpolated_pin_native_op", "auto"
        )
        self._initialize_buffering_database_handles(placedb)

        with torch.no_grad():
            # other tensors required to build ops

            self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)
            
            k = 500000
            self.pairs = torch.zeros(2*k, dtype=torch.int32, device=device)
            self.weights = torch.zeros(k, dtype=torch.float32, device=device)

            m2_pg_rail_density_boxes = getattr(
                placedb,
                "m2_pg_rail_density_boxes",
                np.zeros((0, 4), dtype=placedb.dtype),
            )
            self.m2_pg_rail_density_boxes = torch.as_tensor(
                m2_pg_rail_density_boxes,
                dtype=self.pos[0].dtype,
                device=device,
            ).reshape(-1, 4)
            m2_pg_rail_boxes = getattr(
                placedb,
                "m2_pg_rail_boxes",
                np.zeros((0, 4), dtype=placedb.dtype),
            )
            self.m2_pg_rail_boxes = torch.as_tensor(
                m2_pg_rail_boxes,
                dtype=self.pos[0].dtype,
                device=device,
            ).reshape(-1, 4)
            # original node size for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_node_size_x = self.node_size_x.clone()
                self.original_node_size_y = self.node_size_y.clone()

            self.pin_offset_x = torch.tensor(
                placedb.pin_offset_x, dtype=self.pos[0].dtype, device=device
            )
            self.pin_offset_y = torch.tensor(
                placedb.pin_offset_y, dtype=self.pos[0].dtype, device=device
            )
            # original pin offset for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_pin_offset_x = self.pin_offset_x.clone()
                self.original_pin_offset_y = self.pin_offset_y.clone()

            self.target_density = torch.empty(
                1, dtype=self.pos[0].dtype, device=device)
            self.target_density.data.fill_(params.target_density)
            self.inflation_state = None
            if params.routability_opt_flag:
                self.original_target_density = self.target_density.clone()
                self.original_num_filler_nodes = int(placedb.num_filler_nodes)

            self.node_areas = self.node_size_x * self.node_size_y
            if params.routability_opt_flag:
                self.original_total_movable_area = float(
                    self.node_areas[: placedb.num_movable_nodes].sum().item()
                )
                if placedb.num_filler_nodes > 0:
                    self.original_total_filler_area = float(
                        self.node_areas[-placedb.num_filler_nodes :].sum().item()
                    )
                else:
                    self.original_total_filler_area = 0.0
                original_target_density = max(float(self.original_target_density.item()), 1e-12)
                self.original_total_place_area = (
                    self.original_total_movable_area + self.original_total_filler_area
                ) / original_target_density
                self.original_total_whitespace_area = (
                    self.original_total_place_area - self.original_total_movable_area
                )

            self.movable_macro_mask = torch.from_numpy(placedb.movable_macro_mask).to(
                device
            )
            self.movable_macro_pins = torch.tensor(
                placedb.movable_macro_pins, dtype=int, device=device)
            self.fixed_macro_mask = torch.from_numpy(placedb.fixed_macro_mask).to(
                device
            )

            self.pin2node_map = torch.from_numpy(
                placedb.pin2node_map).to(device)
            self.flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(
                device
            )
            self.flat_node2pin_start_map = torch.from_numpy(
                placedb.flat_node2pin_start_map
            ).to(device)
            # number of pins for each cell
            self.pin_weights = (
                self.flat_node2pin_start_map[1:] -
                self.flat_node2pin_start_map[:-1]
            ).to(self.node_size_x.dtype)

            self.unit_pin_capacity = torch.empty(
                1, dtype=self.pos[0].dtype, device=device
            )
            self.unit_pin_capacity.data.fill_(params.unit_pin_capacity)
            if params.routability_opt_flag:
                unit_pin_capacity = (
                    self.pin_weights[: placedb.num_movable_nodes]
                    / self.node_areas[: placedb.num_movable_nodes]
                )
                avg_pin_capacity = unit_pin_capacity.mean() * self.target_density
                # min(computed, params.unit_pin_capacity)
                self.unit_pin_capacity = avg_pin_capacity.clamp_(
                    max=params.unit_pin_capacity
                )
                logging.info("unit_pin_capacity = %g" %
                             (self.unit_pin_capacity))

            # routing information
            # project initial routing utilization map to one layer
            self.initial_horizontal_utilization_map = None
            self.initial_vertical_utilization_map = None
            if (
                params.routability_opt_flag
                and placedb.initial_horizontal_demand_map is not None
            ):
                self.initial_horizontal_utilization_map = (
                    torch.from_numpy(placedb.initial_horizontal_demand_map)
                    .to(device)
                    .div_(
                        placedb.routing_grid_size_y * placedb.unit_horizontal_capacity
                    )
                )
                self.initial_vertical_utilization_map = (
                    torch.from_numpy(placedb.initial_vertical_demand_map)
                    .to(device)
                    .div_(placedb.routing_grid_size_x * placedb.unit_vertical_capacity)
                )

            self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
            self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map).to(
                device
            )
            self.flat_net2pin_start_map = torch.from_numpy(
                placedb.flat_net2pin_start_map
            ).to(device)
            self.net_weights = torch.from_numpy(placedb.net_weights).to(device)
            self.modularity_cluster_ids_by_level = []
            self.modularity_num_clusters_by_level = []
            self.modularity_resolutions_used = []
            self.modularity_cluster_source = "none"
            self.refresh_modularity_clusters_from_placedb(placedb)

            if params.with_sta and placedb.flat_pin_to_graph is not None:
                self.flat_pin_to_graph = torch.from_numpy(
                    placedb.flat_pin_to_graph
                ).to(device)
                self.flat_pin_to_graph_start = torch.from_numpy(
                    placedb.flat_pin_to_graph_start
                ).to(device)
                self.flat_pin_to_graph_reverse = torch.from_numpy(
                    placedb.flat_pin_to_graph_reverse
                ).to(device)
                self.flat_pin_to_graph_start_reverse = torch.from_numpy(
                    placedb.flat_pin_to_graph_start_reverse
                ).to(device)

            # regions
            self.flat_region_boxes = torch.from_numpy(placedb.flat_region_boxes).to(
                device
            )
            self.flat_region_boxes_start = torch.from_numpy(
                placedb.flat_region_boxes_start
            ).to(device)
            self.node2fence_region_map = torch.from_numpy(
                placedb.node2fence_region_map
            ).to(device)
            if len(placedb.regions) > 0:
                # This is for multi-electric potential and legalization
                # boxes defined as left-bottm point and top-right point
                self.virtual_macro_fence_region = [
                    torch.from_numpy(region).to(device)
                    for region in placedb.virtual_macro_fence_region
                ]
                # this is for overflow op
                self.total_movable_node_area_fence_region = torch.from_numpy(
                    placedb.total_movable_node_area_fence_region
                ).to(device)
                # this is for gamma update
                self.num_movable_nodes_fence_region = torch.from_numpy(
                    placedb.num_movable_nodes_fence_region
                ).to(device)
                # this is not used yet
                self.num_filler_nodes_fence_region = torch.from_numpy(
                    placedb.num_filler_nodes_fence_region
                ).to(device)

            self.net_mask_all = torch.from_numpy(
                np.ones(placedb.num_nets, dtype=np.uint8)
            ).to(
                device
            )  # all nets included
            net_degrees = np.ediff1d(placedb.flat_net2pin_start_map)
            net_mask = np.logical_and(
                2 <= net_degrees, net_degrees < params.ignore_net_degree
            ).astype(np.uint8)
            self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(
                device
            )  # nets with large degrees are ignored
            large_weight_net_mask = placedb.net_weights < params.ignore_net_weight
            self.net_mask_ignore_large_weights = torch.from_numpy(
                large_weight_net_mask.astype(np.uint8)
            ).to(device)
            # number of pins for each node
            self.num_pins_in_nodes = torch.zeros_like(self.node_size_x)
            self.num_pins_in_nodes[: placedb.num_physical_nodes] = self.pin_weights

            # avoid computing gradient for fixed macros
            # 1 is for fixed macros
            self.pin_mask_ignore_fixed_macros = (
                self.pin2node_map >= placedb.num_movable_nodes
            )

            # pin pair arc lookup tables for timing graph
            self.pin_pair_arc_keys = None
            self.flat_pin_pair_arc_start = None
            self.flat_pin_pair_arc_indices = None
            self.arc_level_start = None
            self.arc_src_pin = None
            self.arc_dst_pin = None
            self.arc_inst_id = None
            self.arc_libcell_id = None
            self.arc_libarc_id = None
            self.arc_sense = None
            self.arc_type = None
            self.arc_offset = None
            self.pin_pred_start = None
            self.pin_pred_pin = None
            self.pin_pred_arc_id = None
            self.pin_succ_start = None
            self.pin_succ_pin = None
            self.pin_succ_arc_id = None
            self.endpoint_pin_ids = None
            self.start_pin_ids = None
            self.pin_to_inst_id = None
            self.pin_to_node_id = None
            self.inst_topo_start = None
            self.inst_topo_ids = None

            # sort nodes by size, return their sorted indices, designed for memory coalesce in electrical force
            movable_size_x = self.node_size_x[: placedb.num_movable_nodes]
            _, self.sorted_node_map = torch.sort(movable_size_x)
            self.sorted_node_map = self.sorted_node_map.to(torch.int32)
            # self.sorted_node_map = torch.arange(0, placedb.num_movable_nodes, dtype=torch.int32, device=device)

            # store floorplan info for later rescaling during legalization/detailed placement
            self.fp_info = FloorplanInfo(
                placedb.xl,
                placedb.yl,
                placedb.xh,
                placedb.yh,
                placedb.site_width,
                placedb.row_height,
                params.scale_factor,
                placedb.routing_grid_xl,
                placedb.routing_grid_yl,
                placedb.routing_grid_xh,
                placedb.routing_grid_yh,
                placedb.routing_V,
                placedb.routing_H,
                torch.from_numpy(placedb.macro_util_V).to(device),
                torch.from_numpy(placedb.macro_util_H).to(device),
                placedb.cell_padding_x,
                # placedb.cell_padding_y,
                placedb.bndry_padding_x,
                placedb.bndry_padding_y,
            )

            if params.with_sta:
                # diff timing opt
                self.inrdelays = torch.from_numpy(
                    placedb.inrdelays
                ).to(device)
                self.infdelays = torch.from_numpy(placedb.infdelays).to(device)
                self.inrtrans = torch.from_numpy(placedb.inrtrans).to(device)
                self.inftrans = torch.from_numpy(placedb.inftrans).to(device)
                self.outcaps = torch.from_numpy(placedb.outcaps).to(device)
                # self.pin_net = torch.from_numpy(placedb.pin_net).to(device)

                self.flat_inst_arcs_by_level = torch.from_numpy(
                    placedb.flat_inst_arcs_by_level).to(device)
                self.flat_inst_arcs_by_level_start = torch.from_numpy(
                    placedb.flat_inst_arcs_by_level_start).to(device)
                self.flat_pin_to_graph = torch.from_numpy(
                    placedb.flat_pin_to_graph).to(device)
                self.flat_pin_to_graph_start = torch.from_numpy(
                    placedb.flat_pin_to_graph_start).to(device)
                self.flat_pin_to_graph_reverse = torch.from_numpy(
                    placedb.flat_pin_to_graph_reverse).to(device)
                self.flat_pin_to_graph_start_reverse = torch.from_numpy(
                    placedb.flat_pin_to_graph_start_reverse).to(device)
                if placedb.pin_pair_arc_keys is not None:
                    self.pin_pair_arc_keys = torch.from_numpy(
                        placedb.pin_pair_arc_keys).to(device)
                    self.flat_pin_pair_arc_start = torch.from_numpy(
                        placedb.flat_pin_pair_arc_start).to(device)
                    self.flat_pin_pair_arc_indices = torch.from_numpy(
                        placedb.flat_pin_pair_arc_indices).to(device)
                compact_timing_attrs = (
                    "arc_level_start",
                    "arc_src_pin",
                    "arc_dst_pin",
                    "arc_inst_id",
                    "arc_libcell_id",
                    "arc_libarc_id",
                    "arc_sense",
                    "arc_type",
                    "arc_offset",
                    "pin_pred_start",
                    "pin_pred_pin",
                    "pin_pred_arc_id",
                    "pin_succ_start",
                    "pin_succ_pin",
                    "pin_succ_arc_id",
                    "endpoint_pin_ids",
                    "start_pin_ids",
                    "pin_to_inst_id",
                    "pin_to_node_id",
                    "inst_topo_start",
                    "inst_topo_ids",
                )
                for attr_name in compact_timing_attrs:
                    value = getattr(placedb, attr_name, None)
                    if value is not None and getattr(value, "size", 0) > 0:
                        setattr(self, attr_name, torch.from_numpy(value).to(device))
                # self.flat_cells_by_level = torch.from_numpy(
                #     placedb.flat_cells_by_level).to(device)
                # self.flat_cells_by_reverse_level = torch.from_numpy(
                #     placedb.flat_cells_by_reverse_level).to(device)
                # self.flat_cells_by_level_start = torch.from_numpy(
                #     placedb.flat_cells_by_level_start).to(device)
                # self.flat_cells_by_reverse_level_start = torch.from_numpy(
                #     placedb.flat_cells_by_reverse_level_start).to(device)

                # self.cells_by_level = torch.from_numpy(
                #     placedb.cells_by_level).to(device)
                # self.cells_by_reverse_level = torch.from_numpy(
                #     placedb.cells_by_reverse_level).to(device)
                self.start_points = torch.from_numpy(
                    placedb.start_points).to(device)
                self.end_points = torch.from_numpy(placedb.end_points).to(device)
                self.clock_pins = torch.from_numpy(placedb.clock_pins).to(device)
                self.FF_ids = torch.from_numpy(placedb.FF_ids).to(device)
                self.clk_pin_r_aat = torch.from_numpy(placedb.clk_pin_r_aat).to(device)
                self.clk_pin_f_aat = torch.from_numpy(placedb.clk_pin_f_aat).to(device)
                self.clk_pin_rtran = torch.from_numpy(placedb.clk_pin_rtran).to(device)
                self.clk_pin_ftran = torch.from_numpy(placedb.clk_pin_ftran).to(device)
                self.clk_pin_names = placedb.clk_pin_names
                self.net_flat_arcs_start = torch.from_numpy(
                    placedb.net_flat_arcs_start).to(device)
                self.net_flat_arcs = torch.from_numpy(
                    placedb.net_flat_arcs).to(device)
                self.inst_flat_arcs_start = torch.from_numpy(
                    placedb.inst_flat_arcs_start).to(device)
                self.inst_flat_arcs = torch.from_numpy(
                    placedb.inst_flat_arcs).to(device)
                self.endpoints_constraint_arcs = torch.from_numpy(
                    placedb.endpoints_constraint_arcs).to(device)
                self.endpoints_timing_check_arcs = torch.from_numpy(
                    placedb.endpoints_timing_check_arcs).to(device)
                self.net2driver_pin_map = torch.from_numpy(
                    placedb.net2driver_pin_map).to(device)
                # self.arcs_info = ARCS_INFO()

                # Initialize LUTS_INFO objects from placedb and construct ARCS_INFO
                f_delay_values = torch.from_numpy(placedb.f_delay_flat_luts_values).to(device)
                f_delay_trans = torch.from_numpy(placedb.f_delay_flat_luts_trans_table).to(device)
                f_delay_cap = torch.from_numpy(placedb.f_delay_flat_luts_cap_table).to(device)
                f_delay_dim = torch.from_numpy(placedb.f_delay_flat_luts_dim).to(device)
                f_delay_luts = LUTS_INFO(f_delay_values, f_delay_trans, f_delay_cap, f_delay_dim)

                r_delay_values = torch.from_numpy(placedb.r_delay_flat_luts_values).to(device)
                r_delay_trans = torch.from_numpy(placedb.r_delay_flat_luts_trans_table).to(device)
                r_delay_cap = torch.from_numpy(placedb.r_delay_flat_luts_cap_table).to(device)
                r_delay_dim = torch.from_numpy(placedb.r_delay_flat_luts_dim).to(device)
                r_delay_luts = LUTS_INFO(r_delay_values, r_delay_trans, r_delay_cap, r_delay_dim)

                f_trans_values = torch.from_numpy(placedb.f_trans_flat_luts_values).to(device)
                f_trans_trans = torch.from_numpy(placedb.f_trans_flat_luts_trans_table).to(device)
                f_trans_cap = torch.from_numpy(placedb.f_trans_flat_luts_cap_table).to(device)
                f_trans_dim = torch.from_numpy(placedb.f_trans_flat_luts_dim).to(device)
                f_trans_luts = LUTS_INFO(f_trans_values, f_trans_trans, f_trans_cap, f_trans_dim)

                r_trans_values = torch.from_numpy(placedb.r_trans_flat_luts_values).to(device)
                r_trans_trans = torch.from_numpy(placedb.r_trans_flat_luts_trans_table).to(device)
                r_trans_cap = torch.from_numpy(placedb.r_trans_flat_luts_cap_table).to(device)
                r_trans_dim = torch.from_numpy(placedb.r_trans_flat_luts_dim).to(device)
                r_trans_luts = LUTS_INFO(r_trans_values, r_trans_trans, r_trans_cap, r_trans_dim)

                self.arcs_info = ARCS_INFO(f_delay_luts, r_delay_luts, f_trans_luts, r_trans_luts)

                self.main_id_2_cell_id_start = torch.from_numpy(
                    placedb.main_id_2_cell_id_start).to(device)
                self.cell_id_2_arc_id_start = torch.from_numpy(
                    placedb.cell_id_2_arc_id_start).to(device)

                self.inst_main_id = torch.from_numpy(
                    placedb.inst_main_id).to(device)
                self.inst_libcell_offset = torch.from_numpy(
                    placedb.inst_libcell_offset).to(device)

                self.flat_libarc_info = torch.from_numpy(placedb.flat_libarc_info).to(device)
                self.flat_libcell_info = torch.from_numpy(placedb.flat_libcell_info).to(device)
                self.flat_libcell_width = (
                    None
                    if placedb.flat_libcell_width is None
                    else torch.from_numpy(placedb.flat_libcell_width).to(device)
                )
                self.flat_libcell_height = (
                    None
                    if placedb.flat_libcell_height is None
                    else torch.from_numpy(placedb.flat_libcell_height).to(device)
                )
                self.flat_libcell_leakage = torch.from_numpy(placedb.flat_libcell_leakage).to(device)
                self.flat_libcell_main_id2size_vt_limit = torch.from_numpy(placedb.flat_libcell_main_id2size_vt_limit).to(device)
                self.main_id_is_sizeable = torch.from_numpy(placedb.main_id_is_sizeable).to(device)
                self.inst_cell_id = torch.from_numpy(placedb.inst_cell_id).to(device)
                self.inst_size_init = torch.from_numpy(placedb.inst_size_init).to(device)
                self.inst_leakage_init = torch.from_numpy(placedb.inst_leakage_init).to(device)
                self.inst_vt_init = torch.from_numpy(placedb.inst_vt_init).to(device)
                self.inst_size_lower = torch.from_numpy(placedb.inst_size_lower).to(device)
                self.inst_size_upper = torch.from_numpy(placedb.inst_size_upper).to(device)
                self.inst_vt_mask = torch.from_numpy(placedb.inst_vt_mask).to(device)
                self.inst_is_sizeable = torch.from_numpy(placedb.inst_is_sizeable).to(device)

                self.cell_id_2_libpin_id_start = torch.from_numpy(
                    placedb.cell_id_2_libpin_id_start).to(device)
                self.pin_2_libpin_offset = torch.from_numpy(
                    placedb.pin_2_libpin_offset).to(device)
                self.flat_lib_pin_offset_x = torch.from_numpy(
                    placedb.flat_lib_pin_offset_x).to(device)
                self.flat_lib_pin_offset_y = torch.from_numpy(
                    placedb.flat_lib_pin_offset_y).to(device)
                self.flat_lib_pin_cap = torch.from_numpy(
                    placedb.flat_lib_pin_cap).to(device)
                self.flat_lib_pin_rcap = torch.from_numpy(
                    placedb.flat_lib_pin_rcap).to(device)
                self.flat_lib_pin_fcap = torch.from_numpy(
                    placedb.flat_lib_pin_fcap).to(device)
                self.flat_lib_pin_cap_limit = torch.from_numpy(
                    placedb.flat_lib_pin_cap_limit).to(device)
                self.flat_lib_pin_slew_limit = torch.from_numpy(
                    placedb.flat_lib_pin_slew_limit).to(device)


                self.net_flat_topo_sort = None
                self.net_flat_topo_sort_start = None
                self.pin_fa = None
                self.flat_pin_to = None
                self.flat_pin_to_start = None
                self.flat_pin_from = None

    def get_size_var(self):
        if self.sizing_parameterization == "real_size":
            return self.real_size
        if self.size_logits is None:
            return None
        # Historical "size" variables now represent the unified timing coordinate.
        size_norm = torch.sigmoid(self.size_logits)
        return self.inst_size_lower + size_norm * (self.inst_size_upper - self.inst_size_lower)

    def get_continuous_size_parameter(self):
        if self.sizing_parameterization == "real_size":
            return self.real_size
        return self.size_logits

    def get_continuous_size_value(self):
        return self.get_size_var()

    def get_continuous_size_gradient(self):
        parameter = self.get_continuous_size_parameter()
        if parameter is None:
            return None
        return parameter.grad

    def capture_continuous_size_state(self):
        parameter = self.get_continuous_size_parameter()
        value = self.get_continuous_size_value()
        return {
            "parameterization": self.sizing_parameterization,
            "parameter": None if parameter is None else parameter.detach().clone(),
            "value": None if value is None else value.detach().clone(),
        }

    def restore_continuous_size_state(self, state):
        if not isinstance(state, dict):
            return False
        parameterization = str(
            state.get("parameterization", self.sizing_parameterization)
        )
        if parameterization != self.sizing_parameterization:
            raise ValueError(
                "continuous size snapshot parameterization mismatch: "
                f"snapshot={parameterization!r}, "
                f"current={self.sizing_parameterization!r}"
            )
        parameter = self.get_continuous_size_parameter()
        saved_parameter = state.get("parameter")
        if saved_parameter is None:
            # Compatibility with snapshots written before the adapter stored
            # the owner coordinate explicitly.
            value = state.get("value")
            if value is None or parameter is None:
                return value is None and parameter is None
            if self.sizing_parameterization == "logits":
                saved_parameter = size_var_to_logits(
                    value,
                    self.inst_size_lower,
                    self.inst_size_upper,
                )
            else:
                saved_parameter = value
        if parameter is None:
            return False
        with torch.no_grad():
            parameter.copy_(
                saved_parameter.to(device=parameter.device, dtype=parameter.dtype)
            )
        return True

    def mask_continuous_size_gradient(self):
        if self.sizing_parameterization != "real_size":
            return 0
        parameter = self.real_size
        mask = self.continuous_size_trainable_mask
        if parameter is None or parameter.grad is None or mask is None:
            return 0
        fixed_mask = ~mask.to(device=parameter.grad.device)
        if not bool(torch.any(fixed_mask).item()):
            return 0
        parameter.grad.masked_fill_(fixed_mask, 0.0)
        return int(fixed_mask.sum().item())

    def project_continuous_size(self):
        if self.sizing_parameterization != "real_size":
            return {
                "applied": False,
                "reason": "logit_parameterization",
                "lower_clamped_count": 0,
                "upper_clamped_count": 0,
                "fixed_instance_count": 0,
                "max_projection_delta": 0.0,
            }
        parameter = self.real_size
        if parameter is None:
            raise RuntimeError("real_size parameter is unavailable")
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise ValueError("real_size parameter must contain only finite values")
        lower = self.inst_size_lower.to(device=parameter.device, dtype=parameter.dtype)
        upper = self.inst_size_upper.to(device=parameter.device, dtype=parameter.dtype)
        if bool(torch.any(upper < lower).item()):
            raise ValueError("inst_size_upper must be greater than or equal to inst_size_lower")
        before = parameter.detach().clone()
        projected = torch.minimum(torch.maximum(before, lower), upper)
        trainable_mask = self.continuous_size_trainable_mask
        fixed_count = 0
        if trainable_mask is not None:
            trainable_mask = trainable_mask.to(device=parameter.device).bool()
            fixed_mask = ~trainable_mask
            fixed_count = int(fixed_mask.sum().item())
            fixed_values = self.inst_size_init.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            fixed_values = torch.minimum(torch.maximum(fixed_values, lower), upper)
            projected = torch.where(fixed_mask, fixed_values, projected)
        lower_clamped = ((before < lower) & (projected == lower)).sum().item()
        upper_clamped = ((before > upper) & (projected == upper)).sum().item()
        max_delta = 0.0
        if projected.numel() > 0:
            max_delta = float((projected - before).abs().max().item())
        with torch.no_grad():
            parameter.copy_(projected)
        return {
            "applied": True,
            "reason": "post_step_projection",
            "lower_clamped_count": int(lower_clamped),
            "upper_clamped_count": int(upper_clamped),
            "fixed_instance_count": fixed_count,
            "max_projection_delta": max_delta,
        }

    def get_vt_var(self):
        if self.vt_logits is None:
            return None
        logits = self.vt_logits.masked_fill(~self.inst_vt_mask, -1e9)
        vt_var = F.softmax(logits, dim=1)
        if hasattr(self, "inst_is_sizeable"):
            vt_var = torch.where(
                self.inst_is_sizeable.unsqueeze(1),
                vt_var,
                self.inst_vt_init,
            )
        return vt_var

    def set_buffer_optimization_state(self, state):
        self.buffer_optimization_state = state
        self.buffer_bu_logits = None if state is None else getattr(state, "bu_logits", None)
        self.buffer_candidate_activation_param = (
            None if state is None else getattr(state, "activation_param", None)
        )
        self.buffer_z_param = None if state is None else getattr(state, "z_param", None)
        self.buffer_bsu_index_param = (
            None if state is None else getattr(state, "bsu_index_param", None)
        )
        return state

    def advance_buffer_segment_timing_topology_epoch(self, reason):
        reason = str(reason or "unspecified")
        self.buffer_segment_timing_topology_epoch = int(
            getattr(self, "buffer_segment_timing_topology_epoch", 0)
        ) + 1
        self.buffer_segment_timing_topology_epoch_reason = reason
        return self.buffer_segment_timing_topology_epoch

    def install_buffer_segment_timing_state(self, state, payload, *, reason):
        if (
            getattr(self, "buffer_segment_count_state", None) is state
            and getattr(self, "buffer_segment_count_payload", None) is payload
        ):
            return False
        self.set_buffer_optimization_state(state)
        self.buffer_relaxed_timing_payload = payload
        self.buffer_segment_count_state = state
        self.buffer_segment_count_payload = payload
        self.buffer_segment_count_nets = payload.get("nets", ())
        self.buffer_segment_count_per_size_input_cap = payload.get("per_size_input_cap")
        self.buffer_segment_count_per_size_delay = payload.get("per_size_delay")
        self.buffer_segment_count_per_size_output_slew = payload.get(
            "per_size_output_slew"
        )
        self.advance_buffer_segment_timing_topology_epoch(reason)
        return True

    def get_buffer_bu_var(self):
        state = getattr(self, "buffer_optimization_state", None)
        if state is None or not hasattr(state, "relaxed_bu"):
            return None
        return state.relaxed_bu()

    def get_buffer_z_var(self):
        state = getattr(self, "buffer_optimization_state", None)
        if state is None or not hasattr(state, "z_value"):
            return None
        return state.z_value()

    def get_buffer_bsu_index_var(self):
        state = getattr(self, "buffer_optimization_state", None)
        if state is None:
            return None
        return state.bsu_index()

    def get_buffer_candidate_node_id(self):
        state = getattr(self, "buffer_optimization_state", None)
        if state is None or not hasattr(state, "candidate_node_id"):
            return None
        return state.candidate_node_id

    def get_libcell_leakage(self):
        return getattr(self, "flat_libcell_leakage", None)

    def get_inst_leakage(self):
        return getattr(self, "inst_leakage_init", None)
    def refresh_modularity_clusters_from_placedb(self, placedb):
        self.modularity_cluster_ids_by_level = []
        self.modularity_num_clusters_by_level = []
        self.modularity_resolutions_used = []
        self.modularity_cluster_source = "none"

        result = getattr(placedb, "modularity_active_clustering_result", None)
        if result is not None:
            self.modularity_cluster_source = "active"
        else:
            result = getattr(placedb, "modularity_topology_clustering_result", None)
            if result is not None:
                self.modularity_cluster_source = "topology"

        if result is None:
            return

        cluster_ids_by_level = []
        num_clusters_by_level = []
        expected_num_nodes = int(placedb.num_movable_nodes)
        for level_idx, cluster_ids in enumerate(result.cluster_ids_by_level):
            if not isinstance(cluster_ids, torch.Tensor):
                cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.int64)
            cluster_ids = cluster_ids.to(device=self.device, dtype=torch.int64)
            if cluster_ids.numel() != expected_num_nodes:
                raise ValueError(
                    "modularity cluster size mismatch at level %d: got %d nodes, expected %d"
                    % (level_idx, cluster_ids.numel(), expected_num_nodes)
                )
            cluster_ids_by_level.append(cluster_ids)
            if level_idx < len(result.num_clusters_by_level):
                num_clusters = int(result.num_clusters_by_level[level_idx])
            else:
                num_clusters = int(cluster_ids.max().item()) + 1 if cluster_ids.numel() else 0
            num_clusters_by_level.append(num_clusters)

        self.modularity_cluster_ids_by_level = cluster_ids_by_level
        self.modularity_num_clusters_by_level = num_clusters_by_level
        self.modularity_resolutions_used = list(getattr(result, "resolutions_used", []))
        logging.info(
            "Loaded modularity %s clusters on %s: levels=%d counts=%s",
            self.modularity_cluster_source,
            str(self.device),
            len(self.modularity_cluster_ids_by_level),
            self.modularity_num_clusters_by_level,
        )

    def bin_center_x_padded(self, placedb, padding, num_bins_x):
        """
        @brief compute array of bin center horizontal coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_x = (self.fp_info.xh - self.fp_info.xl) / num_bins_x
        xl = self.fp_info.xl - padding * bin_size_x
        xh = self.fp_info.xh + padding * bin_size_x
        bin_center_x = torch.from_numpy(placedb.bin_centers(xl, xh, bin_size_x)).to(
            self.device
        )
        return bin_center_x

    def build_cell_modeling_op(self, placedb, padding, num_bins_y):
        """
        @brief build online cell surrogate used by sizing-enabled timing flow
        """
        return CellModeling(self)

    def build_gate_projection_op(
        self,
        scorer=None,
        validator=None,
        candidate_provider=None,
        resolver=None,
    ):
        """
        @brief build discrete gate projection operator for post-optimization mapping
        """
        if resolver is None:
            resolver_name = getattr(self, "projection_resolver", "argmin")
            if resolver_name == "argmin":
                resolver = ArgminProjectionResolver()
            elif resolver_name == "stable_current_cell":
                resolver = StableCurrentCellResolver()
            elif resolver_name == "nearest_size_round":
                resolver = NearestSizeProjectionResolver()
            else:
                raise ValueError(f"unsupported projection_resolver: {resolver_name}")
        return GateProjectionOp(
            self,
            scorer=scorer,
            validator=validator,
            candidate_provider=candidate_provider,
            resolver=resolver,
        )

    def bin_center_y_padded(self, placedb, padding, num_bins_y):
        """
        @brief compute array of bin center vertical coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_y = (self.fp_info.yh - self.fp_info.yl) / num_bins_y
        yl = self.fp_info.yl - padding * bin_size_y
        yh = self.fp_info.yh + padding * bin_size_y
        bin_center_y = torch.from_numpy(placedb.bin_centers(yl, yh, bin_size_y)).to(
            self.device
        )
        return bin_center_y

    # def scale(self, factor):
    #     # TODO: add regions
    #     with torch.no_grad():
    #         self.pos.mul_(factor).round_()
    #         self.node_size_x.mul_(factor).round_()
    #         self.node_size_y.mul_(factor).round_()
    #         self.original_node_size_x.mul_(factor).round_()
    #         self.original_node_size_y.mul_(factor).round_()
    #         self.pin_offset_x.mul_(factor)
    #         self.pin_offset_y.mul_(factor)
    #         self.original_pin_offset_x.mul_(factor)
    #         self.original_pin_offset_y.mul_(factor)
    #         self.unit_pin_capacity.div_(factor * factor)
    #         self.flat_region_boxes.mul_(factor).round_()
    #         self.node_areas.mul_(factor * factor)
    #         self.fp_info.scale(factor)
    #         self.fp_info.scale_factor = factor


class PlaceOpCollection(object):
    """
    @brief A wrapper for all ops
    """

    def __init__(self):
        """
        @brief initialization
        """
        self.pin_pos_op = None
        self.move_boundary_op = None
        self.hpwl_op = None
        self.weight_hpwl_op = None
        self.rsmt_wl_op = None
        self.density_overflow_op = None
        self.legality_check_op = None
        self.legalize_op = None
        self.detailed_place_op = None
        self.m2_soft_legalize_op = None
        self.m2_pa_refine_op = None
        self.wirelength_op = None
        self.update_gamma_op = None
        self.density_op = None
        self.virtual_cell_density_op = None
        self.update_density_weight_op = None
        self.precondition_op = None
        self.noise_op = None
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None
        self.irt_egr_congestion_map_op = None
        self.adjust_node_area_op = None
        self.macro_overlap_op = None
        self.update_macro_overlap_weight_op = None
        self.macro_refinement_op = None
        self.pws_op = None

        # diff timing obj
        self.steiner_topo_op = None
        self.elmore_delay_op = None
        self.timing_propagation_op = None
        self.cell_modeling_op = None


class BasicPlace(nn.Module):
    """
    @brief Base placement class.
    All placement engines should be derived from this class.
    """

    def _placement_sizing_mode(self, params):
        return getattr(params, "placement_sizing_mode", "place_only")

    def register_buffer_optimization_state(self, state):
        self.buffer_params = nn.ParameterList()
        if state is not None:
            if hasattr(state, "z_param"):
                self.buffer_params.append(state.z_param)
            elif hasattr(state, "bu_logits"):
                self.buffer_params.append(state.bu_logits)
            elif hasattr(state, "activation_param"):
                self.buffer_params.append(state.activation_param)
            if hasattr(state, "bsu_index_param"):
                self.buffer_params.append(state.bsu_index_param)
        if hasattr(self, "data_collections"):
            self.data_collections.set_buffer_optimization_state(state)
        return state

    def _should_randomize_init_pos(self, params):
        return (
            params.global_place_flag
            and params.random_center_init_flag
            and self._placement_sizing_mode(params) != "size_only"
        )

    def _copy_continuation_seed_fillers(
        self,
        continuation_seed,
        num_nodes,
        num_physical_nodes,
        capacity,
        dtype,
    ):
        filler_x = np.asarray(
            continuation_seed.get("filler_x", []),
            dtype=dtype,
        ).reshape(-1)
        filler_y = np.asarray(
            continuation_seed.get("filler_y", []),
            dtype=dtype,
        ).reshape(-1)
        count = min(int(capacity), filler_x.size, filler_y.size)
        if count > 0:
            self.init_pos[
                num_physical_nodes:num_physical_nodes + count
            ] = filler_x[:count]
            self.init_pos[
                num_nodes + num_physical_nodes:num_nodes + num_physical_nodes + count
            ] = filler_y[:count]
        return count

    @staticmethod
    def _continuation_seed_node_name_key(name):
        if isinstance(name, bytes):
            return name.decode()
        if hasattr(name, "decode"):
            return name.decode()
        return str(name)

    def _current_movable_node_name_to_id(self, placedb):
        num_movable_nodes = int(placedb.num_movable_nodes)
        num_physical_nodes = int(placedb.num_physical_nodes)
        movable_limit = min(num_movable_nodes, num_physical_nodes)
        name_to_id = {}

        node_names = getattr(placedb, "node_names", None)
        if node_names is not None:
            for node_id, name in enumerate(list(node_names)[:movable_limit]):
                name_to_id[self._continuation_seed_node_name_key(name)] = node_id

        for name, node_id in getattr(placedb, "node_name2id_map", {}).items():
            try:
                node_id = int(node_id)
            except (TypeError, ValueError):
                continue
            if 0 <= node_id < movable_limit:
                name_to_id[self._continuation_seed_node_name_key(name)] = node_id

        return name_to_id

    def _copy_named_continuation_seed_movables(
        self,
        placedb,
        continuation_seed,
        num_nodes,
        dtype,
    ):
        movable_node_names = continuation_seed.get("movable_node_names")
        if movable_node_names is None:
            return

        movable_x = np.asarray(
            continuation_seed.get("movable_x", []),
            dtype=dtype,
        ).reshape(-1)
        movable_y = np.asarray(
            continuation_seed.get("movable_y", []),
            dtype=dtype,
        ).reshape(-1)
        count = min(len(movable_node_names), movable_x.size, movable_y.size)
        if count <= 0:
            return

        current_name_to_id = self._current_movable_node_name_to_id(placedb)
        for seed_id, name in enumerate(list(movable_node_names)[:count]):
            node_id = current_name_to_id.get(
                self._continuation_seed_node_name_key(name)
            )
            if node_id is None:
                continue
            self.init_pos[node_id] = movable_x[seed_id]
            self.init_pos[num_nodes + node_id] = movable_y[seed_id]

    def _copy_available_filler_positions_from_placedb(
        self,
        placedb,
        filler_beg,
        filler_end,
        dtype,
    ):
        if filler_beg >= filler_end:
            return filler_beg

        num_physical_nodes = int(placedb.num_physical_nodes)
        num_nodes = int(placedb.num_nodes)
        node_x = np.asarray(placedb.node_x, dtype=dtype).reshape(-1)
        node_y = np.asarray(placedb.node_y, dtype=dtype).reshape(-1)
        source_beg = num_physical_nodes + filler_beg
        source_end = min(num_physical_nodes + filler_end, node_x.size, node_y.size)
        if source_end <= source_beg:
            return filler_beg

        count = source_end - source_beg
        target_beg = num_physical_nodes + filler_beg
        target_end = target_beg + count
        self.init_pos[target_beg:target_end] = node_x[source_beg:source_end]
        self.init_pos[
            num_nodes + target_beg:num_nodes + target_end
        ] = node_y[source_beg:source_end]
        return filler_beg + count

    def _initialize_fresh_filler_positions(self, placedb, filler_beg, filler_end):
        if filler_beg >= filler_end:
            return

        num_nodes = int(placedb.num_nodes)
        num_physical_nodes = int(placedb.num_physical_nodes)
        if len(placedb.regions) > 0:
            for i, region in enumerate(placedb.regions):
                region_filler_beg, region_filler_end = placedb.filler_start_map[i: i + 2]
                subregion_areas = (region[:, 2] - region[:, 0]) * (
                    region[:, 3] - region[:, 1]
                )
                total_area = np.sum(subregion_areas)
                subregion_area_ratio = subregion_areas / total_area
                subregion_num_filler = np.round(
                    (region_filler_end - region_filler_beg) * subregion_area_ratio
                )
                subregion_num_filler[-1] = (
                    region_filler_end
                    - region_filler_beg
                    - np.sum(subregion_num_filler[:-1])
                )
                subregion_num_filler_start_map = np.concatenate(
                    [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                ).astype(np.int32)
                for j, subregion in enumerate(region):
                    sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[
                        j: j + 2
                    ]
                    target_beg = max(
                        int(region_filler_beg + sub_filler_beg),
                        int(filler_beg),
                    )
                    target_end = min(
                        int(region_filler_beg + sub_filler_end),
                        int(filler_end),
                    )
                    if target_beg >= target_end:
                        continue
                    self.init_pos[
                        num_physical_nodes + target_beg:num_physical_nodes + target_end
                    ] = np.random.uniform(
                        low=subregion[0],
                        high=subregion[2] -
                        placedb.filler_size_x_fence_region[i],
                        size=target_end - target_beg,
                    )
                    self.init_pos[
                        num_nodes + num_physical_nodes + target_beg:
                        num_nodes + num_physical_nodes + target_end
                    ] = np.random.uniform(
                        low=subregion[1],
                        high=subregion[3] -
                        placedb.filler_size_y_fence_region[i],
                        size=target_end - target_beg,
                    )

            region_filler_beg, region_filler_end = placedb.filler_start_map[-2:]
            target_beg = max(int(region_filler_beg), int(filler_beg))
            target_end = min(int(region_filler_end), int(filler_end))
            if target_beg < target_end:
                self.init_pos[
                    num_physical_nodes + target_beg:num_physical_nodes + target_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.filler_size_x_fence_region[-1],
                    size=target_end - target_beg,
                )
                self.init_pos[
                    num_nodes + num_physical_nodes + target_beg:
                    num_nodes + num_physical_nodes + target_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.filler_size_y_fence_region[-1],
                    size=target_end - target_beg,
                )
        else:
            self.init_pos[
                num_physical_nodes + filler_beg:num_physical_nodes + filler_end
            ] = np.random.uniform(
                low=placedb.xl,
                high=placedb.xh - placedb.node_size_x[-placedb.num_filler_nodes],
                size=filler_end - filler_beg,
            )
            self.init_pos[
                num_nodes + num_physical_nodes + filler_beg:
                num_nodes + num_physical_nodes + filler_end
            ] = np.random.uniform(
                low=placedb.yl,
                high=placedb.yh - placedb.node_size_y[-placedb.num_filler_nodes],
                size=filler_end - filler_beg,
            )

    def _initialize_init_pos_from_continuation_seed(self, placedb, continuation_seed):
        num_nodes = int(placedb.num_nodes)
        num_physical_nodes = int(placedb.num_physical_nodes)
        num_filler_nodes = int(getattr(placedb, "num_filler_nodes", 0))
        dtype = placedb.dtype

        node_x = np.asarray(placedb.node_x, dtype=dtype).reshape(-1)
        node_y = np.asarray(placedb.node_y, dtype=dtype).reshape(-1)
        physical_count = min(num_physical_nodes, node_x.size, node_y.size)
        self.init_pos[0:physical_count] = node_x[:physical_count]
        self.init_pos[num_nodes:num_nodes + physical_count] = node_y[:physical_count]
        self._copy_named_continuation_seed_movables(
            placedb,
            continuation_seed,
            num_nodes,
            dtype,
        )

        filler_capacity = min(
            num_filler_nodes,
            max(0, num_nodes - num_physical_nodes),
        )
        seeded_filler_count = self._copy_continuation_seed_fillers(
            continuation_seed,
            num_nodes,
            num_physical_nodes,
            filler_capacity,
            dtype,
        )
        fallback_filler_beg = self._copy_available_filler_positions_from_placedb(
            placedb,
            seeded_filler_count,
            filler_capacity,
            dtype,
        )
        self._initialize_fresh_filler_positions(
            placedb,
            fallback_filler_beg,
            filler_capacity,
        )

    def _initialize_fresh_init_pos(self, params, placedb):
        # initial location of cells
        init_loc_perc_x = params.init_loc_perc_x
        init_loc_perc_y = params.init_loc_perc_y
        if (
            not 0.0 < params.init_loc_perc_x < 1.0
            or not 0.0 < params.init_loc_perc_y < 1.0
        ):
            init_loc_perc_x = 0.5
            init_loc_perc_y = 0.5
            logging.warn(
                "incorrect initial location provided, choose center of layout")

        init_loc_x = (
            placedb.xl * 1.0 + (placedb.xh * 1.0 -
                                placedb.xl * 1.0) * init_loc_perc_x
        )
        init_loc_y = (
            placedb.yl * 1.0 + (placedb.yh * 1.0 -
                                placedb.yl * 1.0) * init_loc_perc_y
        )
        randomize_init_pos = self._should_randomize_init_pos(params)

        # x position
        self.init_pos[0: placedb.num_physical_nodes] = placedb.node_x
        if randomize_init_pos:
            logging.info(
                f"move cells to location {init_loc_x, init_loc_y} with random noise"
            )
            self.init_pos[0: placedb.num_movable_nodes] = (
                np.random.normal(
                    loc=init_loc_x,
                    scale=(placedb.xh - placedb.xl) * 0.001,
                    size=placedb.num_movable_nodes,
                )
                - placedb.node_size_x[0: placedb.num_movable_nodes] / 2
            )

        # y position
        self.init_pos[
            placedb.num_nodes: placedb.num_nodes + placedb.num_physical_nodes
        ] = placedb.node_y
        if randomize_init_pos:
            self.init_pos[
                placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes
            ] = (
                np.random.normal(
                    loc=init_loc_y,
                    scale=(placedb.yh - placedb.yl) * 0.001,
                    size=placedb.num_movable_nodes,
                )
                - placedb.node_size_y[0: placedb.num_movable_nodes] / 2
            )

        if placedb.num_filler_nodes:  # uniformly distribute filler cells in the layout
            if len(placedb.regions) > 0:
                # uniformly spread fillers in fence region
                # for cells in the fence region
                for i, region in enumerate(placedb.regions):
                    filler_beg, filler_end = placedb.filler_start_map[i: i + 2]
                    subregion_areas = (region[:, 2] - region[:, 0]) * (
                        region[:, 3] - region[:, 1]
                    )
                    total_area = np.sum(subregion_areas)
                    subregion_area_ratio = subregion_areas / total_area
                    subregion_num_filler = np.round(
                        (filler_end - filler_beg) * subregion_area_ratio
                    )
                    subregion_num_filler[-1] = (filler_end - filler_beg) - np.sum(
                        subregion_num_filler[:-1]
                    )
                    subregion_num_filler_start_map = np.concatenate(
                        [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                    ).astype(np.int32)
                    for j, subregion in enumerate(region):
                        sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[
                            j: j + 2
                        ]
                        self.init_pos[
                            placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg: placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[0],
                            high=subregion[2] -
                            placedb.filler_size_x_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )
                        self.init_pos[
                            placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg: placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[1],
                            high=subregion[3] -
                            placedb.filler_size_y_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )

                # for cells outside fence region
                filler_beg, filler_end = placedb.filler_start_map[-2:]
                self.init_pos[
                    placedb.num_physical_nodes
                    + filler_beg: placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.filler_size_x_fence_region[-1],
                    size=filler_end - filler_beg,
                )
                self.init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_beg: placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.filler_size_y_fence_region[-1],
                    size=filler_end - filler_beg,
                )

            else:
                self.init_pos[
                    placedb.num_physical_nodes: placedb.num_nodes
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh -
                    placedb.node_size_x[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )
                self.init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes: placedb.num_nodes * 2
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh -
                    placedb.node_size_y[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )

    def __init__(self, params, placedb, timer):
        """
        @brief initialization
        @param params parameter
        @param placedb placement database
        """
        super(BasicPlace, self).__init__()
        self.timer = timer
        write_crash_stage_marker(
            "basicplace_database_available",
            "done",
            num_nodes=getattr(placedb, "num_nodes", None),
            num_pins=getattr(placedb, "num_pins", None),
            num_nets=getattr(placedb, "num_nets", None),
            with_sta=getattr(params, "with_sta", None),
        )

        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

        continuation_seed = getattr(placedb, "pending_continuation_seed", None)
        if continuation_seed is not None:
            self._initialize_init_pos_from_continuation_seed(
                placedb,
                continuation_seed,
            )
            placedb.pending_continuation_seed = None
        else:
            self._initialize_fresh_init_pos(params, placedb)

        logging.debug("prepare init_pos takes %.2f seconds" %
                      (time.time() - tt))
        prepare_init_pos_ms = (time.time() - tt) * 1000.0

        # setting device
        write_crash_stage_marker(
            "basicplace_device_selection",
            "start",
            params_gpu=getattr(params, "gpu", None),
            params_gpu_id=getattr(params, "gpu_id", None),
            cuda_available=torch.cuda.is_available(),
        )
        if params.gpu and torch.cuda.is_available():
            if params.gpu_id >= torch.cuda.device_count():
                params.gpu_id = 0
            torch.cuda.set_device(params.gpu_id)
            self.device = torch.device("cuda")
            logging.info(
                f"Using Torch GPU device # {params.gpu_id}: {torch.cuda.get_device_name(params.gpu_id)}"
            )
        else:
            self.device = torch.device("cpu")
            logging.info("Using Torch CPU device")
        write_crash_stage_marker(
            "basicplace_device_selection",
            "done",
            selected_device=str(self.device),
            params_gpu=getattr(params, "gpu", None),
            params_gpu_id=getattr(params, "gpu_id", None),
        )

        # position should be parameter
        # must be defined in BasicPlace
        tt = time.time()
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))]
        )
        self.size_params = nn.ParameterList()
        self.vt_params = nn.ParameterList()
        self.buffer_params = nn.ParameterList()
        self.placement_sizing_mode = self._placement_sizing_mode(params)
        
        # inst size Parameter
        # self.init_size = placedb.inst_libcell_offset
        # self.inst_libcell_offset = nn.ParameterList(
        #     [
        #         nn.Parameter(
        #             torch.from_numpy(
        #                 self.init_size
        #             ).to(self.device)
        #         )
        #     ]
        # )
        
        logging.debug("build pos takes %.2f seconds" % (time.time() - tt))
        pos_parameter_ms = (time.time() - tt) * 1000.0
        # shared data on device for building ops
        # I do not want to construct the data from placedb again and again for each op
        tt = time.time()
        write_crash_stage_marker(
            "basicplace_data_collection",
            "start",
            selected_device=str(self.device),
        )
        self.data_collections = PlaceDataCollection(
            self.pos, params, placedb, self.device
        )
        if (
            params.with_sta
            and getattr(placedb, "inst_size_init", None) is not None
            and (
                self.placement_sizing_mode != "place_only"
                or bool(getattr(params, "timing_opt_enabled", False))
            )
        ):
            effective_initial_size = compute_effective_initial_size(
                inst_size_init=placedb.inst_size_init,
                inst_size_lower=placedb.inst_size_lower,
                inst_size_upper=placedb.inst_size_upper,
                dtype=placedb.dtype,
            )
            self.data_collections.effective_initial_size = torch.from_numpy(
                effective_initial_size
            ).to(self.device)
            if self.data_collections.sizing_parameterization == "real_size":
                real_size = nn.Parameter(
                    torch.from_numpy(effective_initial_size).to(self.device)
                )
                self.size_params.append(real_size)
                self.data_collections.real_size = real_size
                if getattr(self.data_collections, "inst_is_sizeable", None) is not None:
                    self.data_collections.continuous_size_trainable_mask = (
                        self.data_collections.inst_is_sizeable.detach().bool()
                    )
            else:
                init_size_logits = compute_init_size_logits(
                    inst_size_init=placedb.inst_size_init,
                    inst_size_lower=placedb.inst_size_lower,
                    inst_size_upper=placedb.inst_size_upper,
                    dtype=placedb.dtype,
                )
                size_logits = nn.Parameter(
                    torch.from_numpy(init_size_logits).to(self.device)
                )
                self.size_params.append(size_logits)
                self.data_collections.size_logits = size_logits

            vt_init = np.clip(placedb.inst_vt_init, 1e-6, 1.0)
            init_vt_logits = np.log(vt_init).astype(placedb.dtype)
            vt_logits = nn.Parameter(torch.from_numpy(init_vt_logits).to(self.device))
            self.vt_params.append(vt_logits)
            self.data_collections.vt_logits = vt_logits
            if bool(getattr(params, "timing_opt_enabled", False)):
                for parameter in (*self.size_params, *self.vt_params):
                    parameter.requires_grad_(requires_grad=False)
        write_crash_stage_marker(
            "basicplace_data_collection",
            "done",
            selected_device=str(self.device),
        )
        logging.debug("build data_collections takes %.2f seconds" %
                      (time.time() - tt))
        data_collection_ms = (time.time() - tt) * 1000.0

        # similarly I wrap all ops
        tt = time.time()
        self.op_collections = PlaceOpCollection()
        logging.debug("build op_collections takes %.2f seconds" %
                      (time.time() - tt))
        op_collection_shell_ms = (time.time() - tt) * 1000.0

        tt = time.time()
        write_crash_stage_marker(
            "basicplace_operator_build",
            "start",
            selected_device=str(self.device),
            with_sta=getattr(params, "with_sta", None),
        )
        # position to pin position
        self.op_collections.pin_pos_op = self.build_pin_pos(
            params, placedb, self.data_collections, self.device
        )
        l_shape_routability_enabled = (
            params.routability_opt_flag
            and getattr(params, "l_shape_routability_flag", 0)
        )
        if params.with_sta or l_shape_routability_enabled:
            self.op_collections.steiner_topo_op = self.build_steiner_topo(
                params, placedb, self.data_collections, self.device)

        # bound nodes to layout region
        self.op_collections.move_boundary_op = self.build_move_boundary(
            params, placedb, self.data_collections, self.device
        )
        # hpwl and density overflow ops for evaluation
        self.op_collections.hpwl_op = self.build_hpwl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            self.device,
        )
        # rectilinear minimum steiner tree wirelength from flute
        # can only be called once
        self.op_collections.pws_op = self.build_pws(
            placedb, self.data_collections)

        # hpwl for nets with smaller weight than ignore_net_weight
        self.op_collections.weight_hpwl_op = self.build_weight_hpwl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            self.device,
        )
        # rectilinear minimum steiner tree wirelength from flute
        # can only be called once
        self.op_collections.rsmt_wl_op = self.build_rsmt_wl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            torch.device("cpu"),
        )
        # legality check
        self.op_collections.legality_check_op = self.build_legality_check(
            params, placedb, self.data_collections, self.device
        )
        self.op_collections.m2_soft_legalize_op = self.build_m2_soft_legalization(
            params, placedb, self.data_collections
        )
        self.m2_pg_rail_hybrid_legalization_view = (
            self.build_m2_pg_rail_hybrid_legalization_view(
                params, placedb, self.data_collections
            )
        )
        self.op_collections.m2_pa_refine_op = self.build_m2_pa_refine(
            params, placedb, self.data_collections
        )
        # legalization
        if len(placedb.regions) > 0:
            (
                self.op_collections.legalize_op,
                self.op_collections.individual_legalize_op,
            ) = self.build_multi_fence_region_legalization(
                params, placedb, self.data_collections, self.device
            )
        else:
            self.op_collections.legalize_op = self.build_legalization(
                params,
                placedb,
                self.data_collections,
                self.device,
            )
        if params.macro_place_flag:
            self.op_collections.macro_legalize_op = self.build_macro_legalization(
                params, placedb, self.data_collections, self.device)
        # detailed placement
        self.op_collections.detailed_place_op = self.build_detailed_placement(
            params, placedb, self.data_collections, self.device
        )
        if (getattr(params, "egr_padding_flag", 0)
                and getattr(params, "legalize_flag", 0)):
            self.op_collections.irt_egr_congestion_map_op = (
                self.build_irt_egr_congestion_map(params, placedb)
            )
        # draw placement
        self.op_collections.draw_place_op = self.build_draw_placement(
            params, placedb, self.data_collections
        )
        self.op_collections.gate_projection_op = self.data_collections.build_gate_projection_op()

        # flag for rsmt_wl_op
        # can only read once
        self.read_lut_flag = True

        logging.debug("build BasicPlace ops takes %.2f seconds" %
                      (time.time() - tt))
        operator_build_ms = (time.time() - tt) * 1000.0
        write_crash_stage_marker(
            "basicplace_operator_build",
            "done",
            selected_device=str(self.device),
            with_sta=getattr(params, "with_sta", None),
        )
        self.initialization_profile = {
            "prepare_init_pos_ms": prepare_init_pos_ms,
            "pos_parameter_ms": pos_parameter_ms,
            "data_collection_ms": data_collection_ms,
            "op_collection_shell_ms": op_collection_shell_ms,
            "operator_build_ms": operator_build_ms,
            "basicplace_init_ms": (
                prepare_init_pos_ms
                + pos_parameter_ms
                + data_collection_ms
                + op_collection_shell_ms
                + operator_build_ms
            ),
            "selected_device": str(self.device),
        }

    def __call__(self, params, placedb):
        """
        @brief Solve placement.
        placeholder for derived classes.
        @param params parameters
        @param placedb placement database
        """
        pass

    def build_pin_pos(self, params, placedb, data_collections, device):
        """
        @brief sum up the pins for each cell
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return pin_pos.PinPos(
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            pin2node_map=data_collections.pin2node_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            num_physical_nodes=placedb.num_physical_nodes,
            algorithm="node-by-node",
        )

    def build_steiner_topo(self, params, placedb, data_collections, device):
        """
        @brief build the operator for Steiner tree topology construction
        """
        steiner_topo_for_pin_op = steiner_topo.SteinerTopo(
            flat_net2pin_map=data_collections.flat_net2pin_map.to(
                device).cpu(),
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map.to(
                device).cpu(),
            # pin2node_map=data_collections.pin2node_map,
            ignore_net_degree=params.ignore_net_degree,
            deterministic_flag=getattr(params, "deterministic_flag", False),
            collect_edge_geometry_stats=(
                bool(getattr(params, "l_shape_profile_flag", False))
                or bool(getattr(params, "l_shape_collect_edge_geometry_stats", False))
            ))

        return steiner_topo_for_pin_op

    def build_move_boundary(self, params, placedb, data_collections, device):
        """
        @brief bound nodes into layout region
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return move_boundary.MoveBoundary(
            data_collections.node_size_x,
            data_collections.node_size_y,
            fp_info=data_collections.fp_info,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
        )

    def build_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """

        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,  # net_mask_all
            algorithm="net-by-net",  # "atomic"
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            hpwls = wirelength_for_pin_op(pin_pos_op(pos))
            if reduction:
                return hpwls.sum() / data_collections.fp_info.scale_factor
            else:
                return hpwls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_pws(self, placedb, data_collections):
        """
        @brief accumulate pin weights of a node
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        # CPU version by default...
        pws_op = pws.PinWeightSum(
            flat_nodepin=data_collections.flat_node2pin_map,
            nodepin_start=data_collections.flat_node2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            num_nodes=placedb.num_nodes,
            algorithm='node-by-node')

        return pws_op

    def build_weight_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength for weights less than ignore_net_weight
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """

        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_weights,
            algorithm="net-by-net",  # "atomic"
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            hpwls = wirelength_for_pin_op(pin_pos_op(pos))
            if reduction:
                return hpwls.sum() / data_collections.fp_info.scale_factor
            else:
                return hpwls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_rsmt_wl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute rectilinear steiner minimal tree wirelength with flute
        @param params parameters
        @param placedb placement database
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """
        # wirelength cost
        POWVFILE = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../thirdparty/flute/lut.ICCAD2015/POWV9.dat"
            )
        )
        POSTFILE = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../thirdparty/flute/lut.ICCAD2015/POST9.dat"
            )
        )
        logging.info("POWVFILE = %s" % (POWVFILE))
        logging.info("POSTFILE = %s" % (POSTFILE))
        wirelength_for_pin_op = rmst_wl.RmstWL(
            flat_netpin=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            netpin_start=torch.from_numpy(
                placedb.flat_net2pin_start_map).to(device),
            ignore_net_degree=params.ignore_net_degree,
            POWVFILE=POWVFILE,
            POSTFILE=POSTFILE,
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            pin_pos = pin_pos_op(pos)
            wls = wirelength_for_pin_op(
                pin_pos.clone().cpu(), self.read_lut_flag)
            self.read_lut_flag = False
            if reduction:
                return wls.sum() / data_collections.fp_info.scale_factor
            else:
                return wls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_legality_check(self, params, placedb, data_collections, device):
        """
        @brief legality check
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return legality_check.LegalityCheck(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes,
        )

    def build_macro_legalization(self, params, placedb, data_collections, device):
        """
        @brief legalization
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # for movable macro legalization
        # the number of bins control the search granularity
        ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)

        def build_macro_legalization_op(pos):
            logging.info("Start macro legalization")
            return ml(pos.clone(), pos)
        return build_macro_legalization_op

    build_m2_soft_legalization = m2_rail_legalization.build_m2_soft_legalization
    build_m2_pg_rail_hybrid_legalization_view = (
        m2_rail_legalization.build_m2_pg_rail_hybrid_legalization_view
    )
    build_legalization = m2_rail_legalization.build_legalization
    run_adaptive_padding_legalization = (
        m2_rail_legalization.run_adaptive_padding_legalization
    )

    def build_multi_fence_region_legalization(
        self, params, placedb, data_collections, device
    ):
        legal_ops = [
            self.build_fence_region_legalization(
                region_id, params, placedb, data_collections, device
            )
            for region_id in range(len(placedb.regions) + 1)
        ]

        pos_ml_list = []
        pos_gl_list = []

        def build_legalization_op(pos):
            for i in range(len(placedb.regions) + 1):
                pos, pos_ml, pos_gl = legal_ops[i][0](pos)
                pos_ml_list.append(pos_ml)
                pos_gl_list.append(pos_gl)
            legal = self.op_collections.legality_check_op(pos)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos
            else:
                # start abacus legalizer
                for i in range(len(placedb.regions) + 1):
                    pos = legal_ops[i][1](pos, pos_ml_list[i], pos_gl_list[i])
            return pos

        def build_individual_legalization_ops(pos, region_id):
            pos = legal_ops[region_id][0](pos)[0]
            return pos

        return build_legalization_op, build_individual_legalization_ops

    def build_fence_region_legalization(
        self, region_id, params, placedb, data_collections, device
    ):
        # reconstruct node size
        # extract necessary nodes in the electric field and insert virtual macros to replace fence region
        num_nodes = placedb.num_nodes
        num_movable_nodes = placedb.num_movable_nodes
        num_filler_nodes = placedb.num_filler_nodes
        num_terminals = placedb.num_terminals
        num_terminal_NIs = placedb.num_terminal_NIs
        if region_id < len(placedb.regions):
            fence_region_mask = (
                data_collections.node2fence_region_map[:num_movable_nodes] == region_id
            )
        else:
            fence_region_mask = data_collections.node2fence_region_map[
                :num_movable_nodes
            ] >= len(placedb.regions)

        virtual_macros = data_collections.virtual_macro_fence_region[region_id]
        virtual_macros_center_x = (
            virtual_macros[:, 2] + virtual_macros[:, 0]) / 2
        virtual_macros_center_y = (
            virtual_macros[:, 3] + virtual_macros[:, 1]) / 2
        virtual_macros_size_x = (virtual_macros[:, 2] - virtual_macros[:, 0]).clamp(
            min=30
        )

        virtual_macros_size_y = (virtual_macros[:, 3] - virtual_macros[:, 1]).clamp(
            min=30
        )
        virtual_macros[:, 0] = virtual_macros_center_x - \
            virtual_macros_size_x / 2
        virtual_macros[:, 1] = virtual_macros_center_y - \
            virtual_macros_size_y / 2
        virtual_macros_pos = virtual_macros[:, 0:2].t().contiguous()

        # node size
        node_size_x, node_size_y = (
            data_collections.node_size_x,
            data_collections.node_size_y,
        )
        filler_beg, filler_end = placedb.filler_start_map[region_id: region_id + 2]
        node_size_x = torch.cat(
            [
                node_size_x[:num_movable_nodes][fence_region_mask],  # movable
                node_size_x[
                    num_movable_nodes: num_movable_nodes + num_terminals
                ],  # terminals
                virtual_macros_size_x,  # virtual macros
                node_size_x[
                    num_movable_nodes
                    + num_terminals: num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  # terminal NIs
                node_size_x[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg: num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  # fillers
            ],
            0,
        )
        node_size_y = torch.cat(
            [
                node_size_y[:num_movable_nodes][fence_region_mask],  # movable
                node_size_y[
                    num_movable_nodes: num_movable_nodes + num_terminals
                ],  # terminals
                virtual_macros_size_y,  # virtual macros
                node_size_y[
                    num_movable_nodes
                    + num_terminals: num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  # terminal NIs
                node_size_y[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg: num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  # fillers
            ],
            0,
        )

        # num pins in nodes
        # 0 for virtual macros and fillers
        num_pins_in_nodes = data_collections.num_pins_in_nodes
        num_pins_in_nodes = torch.cat(
            [
                # movable
                num_pins_in_nodes[:num_movable_nodes][fence_region_mask],
                num_pins_in_nodes[
                    num_movable_nodes: num_movable_nodes + num_terminals
                ],  # terminals
                torch.zeros(
                    virtual_macros_size_x.size(0),
                    dtype=num_pins_in_nodes.dtype,
                    device=device,
                ),  # virtual macros
                num_pins_in_nodes[
                    num_movable_nodes
                    + num_terminals: num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  # terminal NIs
                num_pins_in_nodes[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg: num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  # fillers
            ],
            0,
        )
        # num movable nodes and num filler nodes
        num_movable_nodes_fence_region = fence_region_mask.long().sum().item()
        num_filler_nodes_fence_region = filler_end - filler_beg
        num_terminals_fence_region = num_terminals + \
            virtual_macros_size_x.size(0)
        assert (
            node_size_x.size(0)
            == node_size_y.size(0)
            == num_movable_nodes_fence_region
            + num_terminals_fence_region
            + num_terminal_NIs
            + num_filler_nodes_fence_region
        )

        # flat region boxes
        flat_region_boxes = torch.tensor(
            [],
            device=node_size_x.device,
            dtype=data_collections.flat_region_boxes.dtype,
        )
        # flat region boxes start
        flat_region_boxes_start = torch.tensor(
            [0],
            device=node_size_x.device,
            dtype=data_collections.flat_region_boxes_start.dtype,
        )
        # node2fence region map: movable + terminal
        node2fence_region_map = torch.zeros(
            num_movable_nodes_fence_region + num_terminals_fence_region,
            dtype=data_collections.node2fence_region_map.dtype,
            device=node_size_x.device,
        ).fill_(data_collections.node2fence_region_map.max().item())

        ml = macro_legalize.MacroLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            # fp_info=data_collections.fp_info,
            num_bins_x=params.num_bins_x,
            num_bins_y=params.num_bins_y,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )

        gl = greedy_legalize.GreedyLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            # fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            # num_bins_x=64, num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            # fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )

        def build_greedy_legalization_op(pos):
            # reconstruct pos for fence region
            pos_total = pos.data.clone()
            pos = pos.view(2, -1)
            pos = (
                torch.cat(
                    [
                        pos[:, :num_movable_nodes][:,
                                                   fence_region_mask],  # movable
                        pos[
                            :, num_movable_nodes: num_movable_nodes + num_terminals
                        ],  # terminals
                        virtual_macros_pos,  # virtual macros
                        pos[
                            :,
                            num_movable_nodes
                            + num_terminals: num_movable_nodes
                            + num_terminals
                            + num_terminal_NIs,
                        ],  # terminal NIs
                        pos[
                            :,
                            num_nodes
                            - num_filler_nodes
                            + filler_beg: num_nodes
                            - num_filler_nodes
                            + filler_end,
                        ],  # fillers
                    ],
                    1,
                )
                .view(-1)
                .contiguous()
            )
            assert pos.size(0) == 2 * node_size_x.size(0)

            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            result = gl(pos1, pos1)
            # commit legal solution for movable cells in fence region
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[0, :num_movable_nodes_fence_region]
            )
            pos_total[1, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[1, :num_movable_nodes_fence_region]
            )
            pos_total = pos_total.view(-1).contiguous()
            result = result.view(-1).contiguous()
            return pos_total, pos1, result

        def build_abacus_legalization_op(pos_total, pos_ref, pos):
            result = al(pos_ref, pos)
            # commit abacus results to pos_total
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[0, :num_movable_nodes_fence_region]
            )
            pos_total[1, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[1, :num_movable_nodes_fence_region]
            )
            pos_total = pos_total.view(-1).contiguous()
            return pos_total

        return build_greedy_legalization_op, build_abacus_legalization_op

    build_m2_pa_refine = m2_rail_legalization.build_m2_pa_refine

    def build_detailed_placement(self, params, placedb, data_collections,
                                 device):
        """
        @brief detailed placement consisting of global swap and independent set matching
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        gs = global_swap.GlobalSwap(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x // 2,
            num_bins_y=placedb.num_bins_y // 2,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=256,
            max_iters=2,
            algorithm='concurrent')
        kr = k_reorder.KReorder(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            K=4,
            max_iters=2)
        ism = independent_set_matching.IndependentSetMatching(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=2048,
            set_size=128,
            max_iters=50,
            algorithm='concurrent')

        # wirelength for position
        def build_detailed_placement_op(pos):
            logging.info("Start ABCDPlace for refinement")

            if placedb.num_movable_nodes < 2:
                logging.info("Too few movable cells, skip detailed placement")
                return pos

            pos1 = pos
            legal = self.op_collections.legality_check_op(pos1)
            logging.info("ABCDPlace input legal flag = %d" %
                         (legal))
            if not legal:
                return pos1

            # integer factorization to prime numbers
            def prime_factorization(num):
                lt = []
                while num != 1:
                    for i in range(2, int(num + 1)):
                        if num % i == 0:  # i is a prime factor
                            lt.append(i)
                            num = num / i  # get the quotient for further factorization
                            break
                return lt

            # compute the scale factor for detailed placement
            # as the algorithms prefer integer coordinate systems
            scale_factor = params.scale_factor
            if params.scale_factor != 1.0:
                inv_scale_factor = int(round(1.0 / params.scale_factor))
                prime_factors = prime_factorization(inv_scale_factor)
                target_inv_scale_factor = 1
                for factor in prime_factors:
                    if factor != 2 and factor != 5:
                        target_inv_scale_factor = inv_scale_factor
                        break
                scale_factor = 1.0 / target_inv_scale_factor
                logging.info("Deriving from system scale factor %g (1/%d)" %
                             (params.scale_factor, inv_scale_factor))
                logging.info("Use scale factor %g (1/%d) for detailed placement" %
                             (scale_factor, target_inv_scale_factor))

            for i in range(1):
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = ism(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Independent set matching legal flag = %d" %
                             (legal))
                if not legal:
                    return pos1
                pos1 = gs(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Global swap legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
            return pos1

        return build_detailed_placement_op

    def build_draw_placement(self, params, placedb, data_collections):
        """
        @brief plot placement
        @param params parameters
        @param placedb placement database
        """
        return draw_place.DrawPlace(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            pin2node_map=data_collections.pin2node_map,
            fp_info=data_collections.fp_info,
            bin_size_x=placedb.bin_size_x,
            bin_size_y=placedb.bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
        )

    def validate(self, placedb, pos, iteration):
        """
        @brief validate placement
        @param placedb placement database
        @param pos locations of cells
        @param iteration optimization step
        """
        pos = torch.from_numpy(pos).to(self.device)
        hpwl = self.op_collections.hpwl_op(pos)
        overflow, max_density = self.op_collections.density_overflow_op(pos)

        return hpwl, overflow, max_density

    def plot(self, params, placedb, iteration, pos):
        """
        @brief plot layout
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        """
        tt = time.time()
        path = "%s/%s" % (params.result_dir, params.design_name())
        tag = "final" if int(iteration) == 9999 else None
        figname = next_plot_filename(path, placedb, iteration, tag=tag)
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        logging.info("plotting to %s takes %.3f seconds" %
                     (figname, time.time() - tt))

    def dump(self, params, placedb, pos, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "wb") as f:
            pickle.dump(
                (
                    self.data_collections.node_size_x.cpu(),
                    self.data_collections.node_size_y.cpu(),
                    self.data_collections.flat_net2pin_map.cpu(),
                    self.data_collections.flat_net2pin_start_map.cpu(),
                    self.data_collections.pin2net_map.cpu(),
                    self.data_collections.flat_node2pin_map.cpu(),
                    self.data_collections.flat_node2pin_start_map.cpu(),
                    self.data_collections.pin2node_map.cpu(),
                    self.data_collections.pin_offset_x.cpu(),
                    self.data_collections.pin_offset_y.cpu(),
                    self.data_collections.net_mask_ignore_large_degrees.cpu(),
                    self.data_collections.fp_info.xl,
                    self.data_collections.fp_info.yl,
                    self.data_collections.fp_info.xh,
                    self.data_collections.fp_info.yh,
                    self.data_collections.fp_info.site_width,
                    self.data_collections.fp_info.row_height,
                    placedb.num_bins_x,
                    placedb.num_bins_y,
                    placedb.num_movable_nodes,
                    placedb.num_terminal_NIs,
                    placedb.num_filler_nodes,
                    pos,
                ),
                f,
            )

    def build_irt_egr_congestion_map(self, params, placedb):
        """
        @brief call iRT EGR for congestion estimation.
        """
        return irt_egr.IRT_eGR(
            params=params,
            placedb=placedb,
        )

    def load(self, params, placedb, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
            self.data_collections.node_size_x.data = data[0].data.to(
                self.device)
            self.data_collections.node_size_y.data = data[1].data.to(
                self.device)
            self.data_collections.flat_net2pin_map.data = data[2].data.to(
                self.device)
            self.data_collections.flat_net2pin_start_map.data = data[3].data.to(
                self.device
            )
            self.data_collections.pin2net_map.data = data[4].data.to(
                self.device)
            self.data_collections.flat_node2pin_map.data = data[5].data.to(
                self.device)
            self.data_collections.flat_node2pin_start_map.data = data[6].data.to(
                self.device
            )
            self.data_collections.pin2node_map.data = data[7].data.to(
                self.device)
            self.data_collections.pin_offset_x.data = data[8].data.to(
                self.device)
            self.data_collections.pin_offset_y.data = data[9].data.to(
                self.device)
            self.data_collections.net_mask_ignore_large_degrees.data = data[10].data.to(
                self.device
            )

            placedb.xl = data[11]
            placedb.yl = data[12]
            placedb.xh = data[13]
            placedb.yh = data[14]
            placedb.site_width = data[15]
            placedb.row_height = data[16]
            placedb.num_bins_x = data[17]
            placedb.num_bins_y = data[18]
            num_movable_nodes = data[19]
            num_nodes = data[0].numel()
            placedb.num_terminal_NIs = data[20]
            placedb.num_filler_nodes = data[21]
            placedb.num_physical_nodes = num_nodes - placedb.num_filler_nodes
            placedb.num_terminals = (
                placedb.num_physical_nodes
                - placedb.num_terminal_NIs
                - num_movable_nodes
            )
            self.data_collections.pos[0].data = data[22].data.to(self.device)
