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
import csv
import re
import math
import time
import copy
import json
import csv
import hashlib
from contextlib import contextmanager
from numbers import Integral
import numpy as np
import logging
import pdb
import itertools

import dreamplace.ops.fence_region.fence_region as fence_region
from dreamplace.ops.openroad_handoff import OpenRoadHandoffController
from dreamplace.ops.openroad_handoff import PlacementHandoffSession
from dreamplace.ops.placeio_ecc.export_options import (
    PyDbExportOptions,
    resolve_flag,
    resolve_m2_density_weight,
)

datatypes = {'float32': np.float32, 'float64': np.float64}

MAX_MOVABLE_UTILIZATION = 0.99
TARGET_MOVABLE_COVERAGE = 0.95


def minimum_target_density_for_coverage(utilization):
    return min(utilization / TARGET_MOVABLE_COVERAGE, 1.0)


SUPPORTED_PLACEDB_REBUILD_MODES = ("topo", "all")


def _validate_rebuild_mode(rebuild_mode):
    mode = str(rebuild_mode or "topo").strip().lower()
    if mode not in SUPPORTED_PLACEDB_REBUILD_MODES:
        raise RuntimeError(
            "unsupported macroPlaceDB rebuild_mode=%r; expected one of %s"
            % (rebuild_mode, ", ".join(SUPPORTED_PLACEDB_REBUILD_MODES))
        )
    return mode


def _load_placeio_ieda():
    import dreamplace.ops.placeio_ieda.place_io as placeio_ieda

    return placeio_ieda


def _load_placeio_ecc():
    import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

    return placeio_ecc


def _openroad_backend_caps_dict():
    from dreamplace.ops.placeio_common.backend_contract import openroad_backend_caps

    return openroad_backend_caps().to_dict()


@contextmanager
def _profile_openroad_init_stage(enabled, label):
    if not enabled:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        logging.info(
            "openroad placement init stage %-42s %.3f s",
            label,
            time.time() - start,
        )


class OpenRoadHandoffExecutionError(RuntimeError):
    def __init__(self, failure_stage, message, handoff_payload=None):
        super(OpenRoadHandoffExecutionError, self).__init__(message)
        self.failure_stage = failure_stage
        self.handoff_payload = dict(handoff_payload or {})


def _maybe_decode_libcell_name(raw_name):
    if isinstance(raw_name, (bytes, np.bytes_)):
        return raw_name.decode("utf-8")
    if isinstance(raw_name, np.ndarray):
        if raw_name.ndim == 0:
            return _maybe_decode_libcell_name(raw_name.item())
        if raw_name.size == 1:
            return _maybe_decode_libcell_name(raw_name.reshape(-1)[0])
    return str(raw_name)


def _write_libcell_id_artifact(output_path, flat_libcell_names, flat_libcell_info):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(
        "libcell_id export: path=%s rows=%d",
        output_path,
        min(len(flat_libcell_names), len(flat_libcell_info)),
    )
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for cell_name, row in zip(flat_libcell_names, flat_libcell_info):
            writer.writerow([_maybe_decode_libcell_name(cell_name), int(row[1])])


def _gate_sizeable_families_by_runtime_arc_schema(
    main_id_2_cell_id_start,
    cell_id_2_arc_id_start,
    flat_libarc_info,
    main_id_is_sizeable,
    flat_libcell_names=None,
):
    """Disable family resizing that a frozen instance-arc graph cannot represent."""
    family_starts = np.asarray(main_id_2_cell_id_start, dtype=np.int64)
    arc_starts = np.asarray(cell_id_2_arc_id_start, dtype=np.int64)
    arc_info = np.asarray(flat_libarc_info, dtype=np.int64)
    sizeable = np.asarray(main_id_is_sizeable, dtype=np.bool_).copy()
    num_families = max(int(family_starts.size) - 1, 0)
    num_cells = max(int(arc_starts.size) - 1, 0)
    if sizeable.ndim != 1 or int(sizeable.size) != num_families:
        raise ValueError("main_id_is_sizeable length must match number of families")
    if arc_info.ndim != 2 or int(arc_info.shape[1]) < 6:
        raise ValueError("flat_libarc_info must contain at least six columns")
    if family_starts.size == 0 or int(family_starts[0]) != 0:
        raise ValueError("main_id_2_cell_id_start must start at zero")
    if arc_starts.size == 0 or int(arc_starts[0]) != 0:
        raise ValueError("cell_id_2_arc_id_start must start at zero")
    if int(family_starts[-1]) != num_cells:
        raise ValueError("family boundaries must cover every Liberty cell")
    if int(arc_starts[-1]) != int(arc_info.shape[0]):
        raise ValueError("arc boundaries must cover every Liberty arc")

    def cell_schema(cell_id):
        counts = {}
        for row in arc_info[arc_starts[cell_id] : arc_starts[cell_id + 1]]:
            signature = (int(row[0]), int(row[1]), int(row[4]), int(row[5]))
            counts[signature] = counts.get(signature, 0) + 1
        return counts

    def cell_name(cell_id):
        if flat_libcell_names is None or cell_id >= len(flat_libcell_names):
            return ""
        return _maybe_decode_libcell_name(flat_libcell_names[cell_id])

    disabled = []
    input_sizeable_count = int(np.count_nonzero(sizeable))
    for main_id in range(num_families):
        if not sizeable[main_id]:
            continue
        cell_begin = int(family_starts[main_id])
        cell_end = int(family_starts[main_id + 1])
        if cell_end - cell_begin <= 1:
            continue
        reference_cell = cell_begin
        reference_schema = cell_schema(reference_cell)
        for cell_id in range(cell_begin + 1, cell_end):
            candidate_schema = cell_schema(cell_id)
            if candidate_schema == reference_schema:
                continue
            signatures = sorted(set(reference_schema) | set(candidate_schema))
            differences = [
                {
                    "signature": list(signature),
                    "reference_count": int(reference_schema.get(signature, 0)),
                    "candidate_count": int(candidate_schema.get(signature, 0)),
                }
                for signature in signatures
                if reference_schema.get(signature, 0)
                != candidate_schema.get(signature, 0)
            ]
            sizeable[main_id] = False
            disabled.append(
                {
                    "main_id": main_id,
                    "reference_cell_id": reference_cell,
                    "reference_cell_name": cell_name(reference_cell),
                    "candidate_cell_id": cell_id,
                    "candidate_cell_name": cell_name(cell_id),
                    "differences": differences,
                }
            )
            break

    return sizeable, {
        "status": "degraded" if disabled else "pass",
        "policy": "identical_runtime_arc_signature_multiset",
        "input_sizeable_family_count": input_sizeable_count,
        "output_sizeable_family_count": int(np.count_nonzero(sizeable)),
        "disabled_family_count": len(disabled),
        "disabled_families": disabled,
    }

MAX_MOVABLE_UTILIZATION = 0.99


def _is_enabled_param(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _pow2_floor(value):
    return int(math.pow(2, math.floor(math.log2(value))))


def _compute_enhanced_auto_adjust_bins(
    preset_num_bins_x,
    preset_num_bins_y,
    layout_height,
    row_height,
):
    preset_num_bins_x = int(preset_num_bins_x)
    preset_num_bins_y = int(preset_num_bins_y)
    if preset_num_bins_x <= 0 or preset_num_bins_y <= 0:
        raise ValueError("preset bin counts must be positive")
    if layout_height <= 0 or row_height <= 0:
        return preset_num_bins_x, preset_num_bins_y

    num_rows = int(math.floor(float(layout_height) / float(row_height)))
    if num_rows <= 0 or preset_num_bins_y <= num_rows:
        return preset_num_bins_x, preset_num_bins_y

    new_num_bins_y = _pow2_floor(num_rows)
    new_num_bins_x = max(
        1,
        int(round(float(preset_num_bins_x) / float(preset_num_bins_y) * new_num_bins_y)),
    )
    return new_num_bins_x, new_num_bins_y


def compute_auto_bin_counts(movable_area, movable_count, target_density, width, height):
    """Use GPL's area and aspect-ratio rule within DreamPlace's 512-bin limit."""
    average_area = movable_area / movable_count
    ideal_bin_count = max(4, width * height * target_density / average_area)
    ratio = 2 ** math.floor(math.log2(max(width, height) / min(width, height)))
    # The shortest axis has at least two bins, so the ratio cannot exceed 256.
    ratio = min(ratio, 256)

    bin_count = 2
    while bin_count < 512 and 4 * bin_count * bin_count * ratio <= ideal_bin_count:
        bin_count *= 2

    if width > height:
        return min(512, bin_count * ratio), bin_count
    return bin_count, min(512, bin_count * ratio)


class MacroPlaceDB(object):
    """
    @brief placement database
    """

    def __init__(self, ecc_module):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.data_manager = ecc_module
        self.rawdb = None  # raw placement database, a C++ object
        self.openroad_bridge = None  # OpenROAD bridge object for OpenROAD mode
        self.backend_caps = {}
        self.rawdb_initialization_profile = {}
        self.ecc_module = ecc_module

        # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_physical_nodes = 0
        self.num_terminals = 0  # number of terminals, essentially fixed macros
        # number of terminal_NIs that can be overlapped, essentially IO pins
        self.num_terminal_NIs = 0
        self.num_place_blockages = 0  # synthetic placement blockages appended to fixed terminals
        self.num_fixed_macro_excluded_place_blockages = 0
        self.m2_pg_rail_boxes = np.zeros((0, 4), dtype=np.float32)
        self.m2_pg_rail_density_boxes = np.zeros((0, 4), dtype=np.float32)
        self.m2_pg_rail_legalization_blockage_flag = False
        self.m2_pa_refine_flag = False
        self.node_name2id_map = {}  # node name to id map, cell name
        self.node_names = None  # 1D array, cell name
        # Unless noted otherwise, geometry arrays in this DB live in the
        # placer's scaled coordinate system after scale(), not raw DBU.
        self.node_x = None  # 1D array, cell position x
        self.node_y = None  # 1D array, cell position y
        self.node_orient = None  # 1D array, cell orientation
        self.node_size_x = None  # 1D array, cell width
        self.node_size_y = None  # 1D array, cell height
        self.node_is_hard_macro = None
        self.macro_writeback_candidate = None

        # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
        self.node2orig_node_map = None
        # this map maps the current multiple node ids into the original one

        self.pin_direct = None  # 1D array, pin direction IO
        self.pin_offset_x = None  # 1D array, pin offset x to its node
        self.pin_offset_y = None  # 1D array, pin offset y to its node
        self.pin_names = None  # 1D array, pin name
        self.pin_name2id_map = {}

        self.net_name2id_map = {}  # net name to id map
        self.net_names = None  # net name
        self.net_weights = None  # weights for each net
        self.net_weight_deltas = None
        self.net_criticality = None
        self.net_criticality_deltas = None

        self.net2pin_map = None  # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None  # flatten version of net2pin_map
        # starting index of each net in flat_net2pin_map
        self.flat_net2pin_start_map = None

        self.node2pin_map = None  # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = None  # flatten version of node2pin_map
        # starting index of each node in flat_node2pin_map
        self.flat_node2pin_start_map = None
        # timing graph adjacency (forward and reverse) in flattened CSR style;
        # indices are design-pin ids aligned with TimingPropagation pin vectors.
        self.flat_pin_to_graph = None
        self.flat_pin_to_graph_start = None
        self.flat_pin_to_graph_reverse = None
        self.flat_pin_to_graph_start_reverse = None
        self.pin_pair_arc_keys = None
        self.flat_pin_pair_arc_start = None
        self.flat_pin_pair_arc_indices = None

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

        self.dbu = None  # database unit, used for scaling back to raw DBU/DEF

        self.num_movable_pins = None

        self.total_movable_node_area = None  # total movable cell area
        self.total_fixed_node_area = None  # native union area of fixed geometry
        self.total_space_area = None  # native placeable core area after fixed geometry

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
        self.min_wire_widths = None  # min wire width per routing layer (scaled coords)
        self.min_wire_spacings = None  # min wire spacing per routing layer (scaled coords)
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
        
        
        self.max_net_weight = None # maximum net weight in timing opt
        self.pin2pin_net_weight = {}
        self.length = [0]

        self.modularity_topology_clustering_result = None
        self.modularity_active_clustering_result = None

        # Timing model
        self.start_points = None
        self.end_points = None
        self.clock_pins = None
        self.FF_ids = None
        self.clk_pin_r_aat = None
        self.clk_pin_f_aat = None
        self.clk_pin_rtran = None
        self.clk_pin_ftran = None
        self.clk_pin_names = None  # // added for clk pin names
        # self.cells_by_level = None
        # self.cells_by_reverse_level = None

        # self.flat_cells_by_level = None  # //
        # self.flat_cells_by_reverse_level = None  # //
        # self.flat_cells_by_level_start = None  # //
        # self.flat_cells_by_reverse_level_start = None  # //
        self.flat_inst_arcs_by_level = None  # //
        self.flat_inst_arcs_by_level_start = None  

        self.inrdelays = None
        self.infdelays = None
        self.inrtrans = None
        self.inftrans = None
        self.outcaps = None
        self.backend_endpoint_rAAT = None
        self.backend_endpoint_fAAT = None
        self.backend_endpoint_rRAT = None
        self.backend_endpoint_fRAT = None
        self.backend_endpoint_rSlew = None
        self.backend_endpoint_fSlew = None
        self.backend_endpoint_min_rAAT = None
        self.backend_endpoint_min_fAAT = None
        self.backend_endpoint_min_rRAT = None
        self.backend_endpoint_min_fRAT = None

        self.net_flat_arcs_start = None
        self.net_flat_arcs = None
        self.inst_flat_arcs_start = None
        self.inst_flat_arcs = None
        self.endpoints_constraint_arcs = None
        self.endpoints_timing_check_arcs = None

        self.main_id_2_cell_id_start = None
        self.cell_id_2_arc_id_start = None
        self.buffer_main_type_index = -1
        self.buffer_main_type_status = "unsupported"
        self.buffer_main_type_candidate_indices = None

        self.inst_main_id = None
        self.inst_libcell_offset = None
        self.inst_cell_id = None
        self.inst_size_init = None
        self.inst_leakage_init = None
        self.inst_vt_init = None
        self.inst_size_lower = None
        self.inst_size_upper = None
        self.inst_vt_mask = None
        self.inst_is_sizeable = None
        self.main_id_is_sizeable = None
        self.flat_libcell_names = None
        self.flat_libcell_width = None
        self.flat_libcell_height = None
        self.flat_libcell_leakage = None
        self.flat_libarc_names = None
        self.flat_libcell_main_type_names = None
        self.cell_id_2_libpin_id_start = None
        self.pin_2_libpin_offset = None
        self.flat_lib_pin_offset_x = None
        self.flat_lib_pin_offset_y = None
        self.flat_lib_pin_cap = None
        self.flat_lib_pin_rcap = None
        self.flat_lib_pin_fcap = None
        self.flat_lib_pin_cap_limit = None
        self.flat_lib_pin_slew_limit = None

        # delay LUTs table
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

        self.f_check_flat_luts_values = None
        self.f_check_flat_luts_trans_table = None
        self.f_check_flat_luts_cap_table = None
        self.f_check_flat_luts_dim = None
        self.r_check_flat_luts_values = None
        self.r_check_flat_luts_trans_table = None
        self.r_check_flat_luts_cap_table = None
        self.r_check_flat_luts_dim = None
        self._topology_refresh_param_baseline = None
        self.handoff_session = None
        self.pending_continuation_seed = None
        self.topology_generation = 0
        self.last_openroad_refresh_summary = None
        self.last_rebuild_summary = None

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

        # This converts raw database coordinates into the normalized placer
        # coordinate space used by density, macro, and timing-related tensors.
        # Every geometry-like field below must stay on the same scale.
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
        if self.flat_libcell_width is not None:
            self.flat_libcell_width *= scale_factor
        if self.flat_libcell_height is not None:
            self.flat_libcell_height *= scale_factor
        if self.flat_lib_pin_offset_x is not None:
            self.flat_lib_pin_offset_x *= scale_factor
        if self.flat_lib_pin_offset_y is not None:
            self.flat_lib_pin_offset_y *= scale_factor

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
        if getattr(self, "total_fixed_node_area", None) is not None:
            self.total_fixed_node_area *= scale_factor * scale_factor
        if self.min_wire_widths is not None:
            self.min_wire_widths *= scale_factor
        if self.min_wire_spacings is not None:
            self.min_wire_spacings *= scale_factor

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
        if self.rows.size > 0:
            box_shift_factor = np.array(
                [shift_factor, shift_factor], dtype=self.rows.dtype
            ).reshape(1, -1)
            # placement rows
            self.rows -= box_shift_factor
            self.rows *= scale_factor
        else:
            box_shift_factor = np.zeros((1, 4), dtype=self.dtype)
        self.total_space_area *= scale_factor * scale_factor

        # regions
        if len(self.flat_region_boxes) > 0:
            self.flat_region_boxes -= box_shift_factor
            self.flat_region_boxes *= scale_factor
        for i in range(len(self.regions)):
            # may have performance issue
            self.regions[i] -= box_shift_factor
            self.regions[i] *= scale_factor
        for rail_box_field in (
            "m2_pg_rail_boxes",
            "m2_pg_rail_density_boxes",
        ):
            rail_boxes = getattr(self, rail_box_field, None)
            if rail_boxes is None or not rail_boxes.size:
                continue
            rail_xl = rail_boxes[:, 0].copy()
            rail_yl = rail_boxes[:, 1].copy()
            rail_w = (rail_boxes[:, 2] - rail_boxes[:, 0]).copy()
            rail_h = (rail_boxes[:, 3] - rail_boxes[:, 1]).copy()
            rail_boxes[:, 0] = (rail_xl - shift_factor[0]) * scale_factor
            rail_boxes[:, 1] = (rail_yl - shift_factor[1]) * scale_factor
            rail_boxes[:, 2] = rail_boxes[:, 0] + rail_w * scale_factor
            rail_boxes[:, 3] = rail_boxes[:, 1] + rail_h * scale_factor

    def _validate_place_blockage_bookkeeping(self):
        blockage_count = int(getattr(self, "num_place_blockages", 0))
        terminal_count = int(getattr(self, "num_terminals", 0))
        if blockage_count < 0 or blockage_count > terminal_count:
            raise RuntimeError(
                "Invalid ecc-tools placement blockage bookkeeping: "
                "num_place_blockages=%d num_terminals=%d"
                % (blockage_count, terminal_count)
            )

    def _import_m2_pg_rail_density_boxes(self, pydb, include_m2_pg_rail_density):
        dtype = getattr(self, "dtype", None) or np.float32
        if not hasattr(pydb, "m2_pg_rail_density_boxes"):
            if include_m2_pg_rail_density:
                raise RuntimeError(
                    "M2 PG rail geometry collection is enabled, but pydb has no "
                    "m2_pg_rail_density_boxes field. Rebuild ecc-tools pybind to avoid "
                    "using stale hard-node M2 PG rail behavior."
                )
            logging.warning(
                "pydb has no m2_pg_rail_density_boxes field; M2 PG rail geometry "
                "collection is disabled, so using empty rail boxes"
            )
            return np.zeros((0, 4), dtype=dtype)

        boxes = np.array(pydb.m2_pg_rail_density_boxes, dtype=dtype)
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                "m2_pg_rail_density_boxes must have shape [N, 4], got %s"
                % (boxes.shape,)
            )
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError(
                "m2_pg_rail_density_boxes must contain positive-width and "
                "positive-height [xl, yl, xh, yh] rectangles"
            )
        return boxes

    def _import_m2_pg_rail_boxes(self, pydb, include_m2_pg_rail_geometry):
        dtype = getattr(self, "dtype", None) or np.float32
        if not include_m2_pg_rail_geometry:
            return np.zeros((0, 4), dtype=dtype)
        if not hasattr(pydb, "m2_pg_rail_boxes"):
            raise RuntimeError(
                "M2 PA-Refine rail geometry collection is enabled, but pydb has "
                "no m2_pg_rail_boxes field. Rebuild ecc-tools pybind before enabling "
                "m2_pa_refine_flag."
            )

        boxes = np.array(pydb.m2_pg_rail_boxes, dtype=dtype)
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                "m2_pg_rail_boxes must have shape [N, 4], got %s"
                % (boxes.shape,)
            )
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError(
                "m2_pg_rail_boxes must contain positive-width and "
                "positive-height [xl, yl, xh, yh] rectangles"
            )
        return boxes

    @staticmethod
    def _log_ieda_m2_pg_rail_blockage_effect(
        include_m2_pg_rail_blockage,
        include_m2_pg_rail_density,
        pydb,
        soft_density_enabled=None,
        legalization_blockage_enabled=False,
        pa_refine_enabled=False,
    ):
        if include_m2_pg_rail_blockage:
            logging.info(
                "PyPlaceDB M2 PG rail blockage rectangles added before union: %d",
                int(getattr(pydb, "m2_pg_rail_blockage_rects", 0)),
            )
        else:
            rail_boxes = getattr(pydb, "m2_pg_rail_density_boxes", [])
            logging.info("PyPlaceDB M2 PG rail hard blockage conversion skipped")
            logging.info(
                "PyPlaceDB M2 PG rail density boxes exported: %d",
                len(rail_boxes),
            )
            if soft_density_enabled is None:
                soft_density_enabled = include_m2_pg_rail_density
            if soft_density_enabled:
                logging.info("PyPlaceDB M2 PG rail soft density collection enabled")
            else:
                logging.info("PyPlaceDB M2 PG rail soft density collection skipped")
            if legalization_blockage_enabled:
                logging.info(
                    "PyPlaceDB M2 PG rail soft legalization geometry enabled"
                )
        if pa_refine_enabled:
            logging.info(
                "PyPlaceDB M2 PG rail PA-Refine geometry boxes exported: %d",
                len(getattr(pydb, "m2_pg_rail_boxes", [])),
            )

    def setup_rawdb(self, params):
        self.dtype = datatypes[params.dtype]
        if self.pydb is None:
            profile = {}
            place_io_engine = getattr(params, "place_io_engine", "ieda")
            if place_io_engine == "openroad":
                import dreamplace.ops.placeio_openroad.place_io as placeio_openroad
                stage_start = time.perf_counter()
                self.openroad_bridge = placeio_openroad.PlaceIOFunction.read(params)
                profile["setup_rawdb_openroad_read_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                read_profile = getattr(
                    placeio_openroad.PlaceIOFunction,
                    "last_read_profile",
                    None,
                )
                if isinstance(read_profile, dict):
                    profile.update(read_profile)
                self.rawdb = self.openroad_bridge
                stage_start = time.perf_counter()
                self.pydb = placeio_openroad.PlaceIOFunction.pydb(self.openroad_bridge)
                profile["setup_rawdb_pydb_export_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                read_profile = getattr(
                    placeio_openroad.PlaceIOFunction,
                    "last_read_profile",
                    None,
                )
                if isinstance(read_profile, dict):
                    profile.update(read_profile)
                self.backend_caps = _openroad_backend_caps_dict()
            elif place_io_engine == "ecc":
                placeio_ecc = _load_placeio_ecc()
                stage_start = time.perf_counter()
                self.rawdb = placeio_ecc.PlaceIOFunction.read(
                    params,
                    self.data_manager,
                )
                profile["setup_rawdb_ecc_read_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                self.get_dmInst_ptr = placeio_ecc.PlaceIOFunction.dm_inst(self.rawdb)
                stage_start = time.perf_counter()
                self.pydb = placeio_ecc.PlaceIOFunction.pydb(self.rawdb)
                profile["setup_rawdb_ecc_pydb_export_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                self.backend_caps = placeio_ecc.PlaceIOFunction.backend_caps(self.rawdb)
            else:
                placeio_ieda = _load_placeio_ieda()
                stage_start = time.perf_counter()
                self.rawdb = placeio_ieda.PlaceIOFunction.read(
                    params,
                    self.data_manager.dir_workspace,
                )
                profile["setup_rawdb_ieda_read_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                self.get_dmInst_ptr = placeio_ieda.PlaceIOFunction.dm_inst(self.rawdb)
                stage_start = time.perf_counter()
                self.pydb = placeio_ieda.PlaceIOFunction.pydb(self.rawdb)
                profile["setup_rawdb_pydb_export_ms"] = (
                    time.perf_counter() - stage_start
                ) * 1000.0
                self.backend_caps = placeio_ieda.PlaceIOFunction.backend_caps(self.rawdb)
            self.rawdb_initialization_profile = profile

    def _openroad_aimp_debug_artifacts_enabled(self, params=None):
        if params is None:
            params = getattr(self, "params", None)
        return bool(getattr(params, "openroad_aimp_debug_artifacts", False))

    def _topology_counts_from_pydb(self, pydb):
        if pydb is None:
            return {}
        return {
            "nodes": int(getattr(pydb, "num_nodes", 0) or 0),
            "pins": int(getattr(pydb, "num_pins", len(getattr(pydb, "pin_names", ()))) or 0),
            "nets": int(getattr(pydb, "num_nets", len(getattr(pydb, "net_names", ()))) or 0),
        }

    def _topology_counts(self):
        pin2net_map = getattr(self, "pin2net_map", None)
        pin_names = getattr(self, "pin_names", None)
        net2pin_map = getattr(self, "net2pin_map", None)
        net_names = getattr(self, "net_names", None)
        return {
            "nodes": int(getattr(self, "num_physical_nodes", 0) or 0),
            "pins": int(
                len(pin2net_map)
                if pin2net_map is not None
                else (len(pin_names) if pin_names is not None else 0)
            ),
            "nets": int(
                len(net2pin_map)
                if net2pin_map is not None
                else (len(net_names) if net_names is not None else 0)
            ),
        }

    def rebuild_from_pydb(self, pydb, rebuild_mode="topo"):
        from dreamplace.ops.routability.gpugr_context import invalidate_route_state

        mode = _validate_rebuild_mode(rebuild_mode)
        if pydb is None:
            raise RuntimeError("cannot rebuild macroPlaceDB from empty pydb")
        pre_counts = self._topology_counts()
        invalidate_route_state(self)
        self.pydb = pydb
        self._restore_topology_refresh_params(self.params)
        self.initialize_from_rawdb(pydb, self.params)
        self.initialize(self.params)
        self.topology_generation = int(getattr(self, "topology_generation", 0) or 0) + 1
        summary = {
            "status": "ok",
            "requested_rebuild_mode": mode,
            "effective_rebuild_mode": mode,
            "source": "macroPlaceDB.initialize_from_rawdb+initialize",
            "topology_generation": self.topology_generation,
            "pre_counts": pre_counts,
            "post_counts": self._topology_counts(),
        }
        self.last_rebuild_summary = dict(summary)
        return summary

    def refresh_from_openroad_bridge(self, refresh_mode="topo", rebuild_mode="topo"):
        if self.openroad_bridge is None:
            raise RuntimeError("OpenROAD refresh requested but macroPlaceDB has no openroad_bridge")
        import dreamplace.ops.placeio_openroad.place_io as placeio_openroad
        self.pydb, refresh_summary = placeio_openroad.PlaceIOFunction.sync_from_openroad(
            self.openroad_bridge,
            refresh_mode=refresh_mode,
            return_summary=True,
        )
        rebuild_summary = self.rebuild_from_pydb(self.pydb, rebuild_mode=rebuild_mode)
        refresh_summary = dict(refresh_summary)
        refresh_summary["pydb_counts"] = self._topology_counts_from_pydb(self.pydb)
        refresh_summary["rebuild"] = dict(rebuild_summary)
        self.last_openroad_refresh_summary = dict(refresh_summary)
        return refresh_summary

    def refresh_from_ecc_backend(
        self, refresh_mode="full_rebuild", rebuild_mode="full_rebuild"
    ):
        if self.rawdb is None or getattr(self.rawdb, "timing_enabled", False) is not True:
            raise RuntimeError("ECC refresh requires a timing-enabled native backend")
        import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

        old_pydb = self.pydb
        new_pydb, refresh_summary = placeio_ecc.PlaceIOFunction.refresh(
            self.rawdb,
            refresh_mode=refresh_mode,
            rebuild_mode=rebuild_mode,
        )
        rebuild_summary = self.rebuild_from_pydb(
            new_pydb,
            rebuild_mode="all" if rebuild_mode == "full_rebuild" else rebuild_mode,
        )
        self.backend_caps = placeio_ecc.PlaceIOFunction.backend_caps(self.rawdb)
        refresh_summary = dict(refresh_summary or {})
        refresh_summary["rebuild"] = dict(rebuild_summary)
        refresh_summary["old_macro_pydb_id"] = id(old_pydb)
        refresh_summary["new_macro_pydb_id"] = id(new_pydb)
        refresh_summary["macro_pydb_replaced"] = old_pydb is not new_pydb
        self.last_ecc_refresh_summary = dict(refresh_summary)
        return refresh_summary

    def _capture_topology_refresh_param_baseline(self, params):
        if self._topology_refresh_param_baseline is not None:
            return
        self._topology_refresh_param_baseline = {
            "shift_factor": list(getattr(params, "shift_factor", [0.0, 0.0])),
            "scale_factor": getattr(params, "scale_factor", 0.0),
            "macro_halo_x": getattr(params, "macro_halo_x", 0.0),
            "macro_halo_y": getattr(params, "macro_halo_y", 0.0),
            "macro_pin_halo_x": getattr(params, "macro_pin_halo_x", 0.0),
            "macro_pin_halo_y": getattr(params, "macro_pin_halo_y", 0.0),
            "cell_padding_x": getattr(params, "cell_padding_x", 0.0),
            "bndry_padding_x": getattr(params, "bndry_padding_x", 0.0),
            "bndry_padding_y": getattr(params, "bndry_padding_y", 0.0),
            "target_density": getattr(params, "target_density", 0.0),
            "route_info_input": getattr(params, "route_info_input", None),
            "num_bins_x": getattr(params, "num_bins_x", None),
            "num_bins_y": getattr(params, "num_bins_y", None),
            "macro_place_flag": getattr(params, "macro_place_flag", False),
        }

    def _restore_topology_refresh_params(self, params):
        baseline = self._topology_refresh_param_baseline
        if baseline is None:
            return
        params.shift_factor[0] = baseline["shift_factor"][0]
        params.shift_factor[1] = baseline["shift_factor"][1]
        params.scale_factor = baseline["scale_factor"]
        params.macro_halo_x = baseline["macro_halo_x"]
        params.macro_halo_y = baseline["macro_halo_y"]
        params.macro_pin_halo_x = baseline["macro_pin_halo_x"]
        params.macro_pin_halo_y = baseline["macro_pin_halo_y"]
        params.cell_padding_x = baseline["cell_padding_x"]
        params.bndry_padding_x = baseline["bndry_padding_x"]
        params.bndry_padding_y = baseline["bndry_padding_y"]
        params.target_density = baseline["target_density"]
        params.route_info_input = baseline["route_info_input"]
        params.num_bins_x = baseline["num_bins_x"]
        params.num_bins_y = baseline["num_bins_y"]
        params.macro_place_flag = baseline["macro_place_flag"]

    def _decode_name(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _normalize_snapshot_value(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple)):
            return tuple(self._normalize_snapshot_value(item) for item in value)
        if hasattr(value, "tolist"):
            return self._normalize_snapshot_value(value.tolist())
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    def _node_names_as_str_list(self):
        if self.node_names is None:
            return []
        return [self._decode_name(name) for name in self.node_names]

    def _pydb_names_as_str_list(self, pydb, attr_name):
        return [self._decode_name(value) for value in getattr(pydb, attr_name, [])]

    def _hash_name_list(self, values):
        digest = hashlib.sha1()
        for value in values:
            digest.update(self._decode_name(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:12]

    def _hash_snapshot_payload(self, payload):
        digest = hashlib.sha1()
        digest.update(repr(self._normalize_snapshot_value(payload)).encode("utf-8"))
        return digest.hexdigest()[:12]

    def _snapshot_sequence_value(self, values, index):
        try:
            if index < 0 or index >= len(values):
                return None
            return self._normalize_snapshot_value(values[index])
        except (IndexError, TypeError):
            return None

    def _snapshot_id_value(self, values, index):
        value = self._snapshot_sequence_value(values, index)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _canonical_pin_to_node_names(self, pydb):
        if not hasattr(pydb, "pin2node_map"):
            return None
        pin_names = self._pydb_names_as_str_list(pydb, "pin_names")
        node_names = self._pydb_names_as_str_list(pydb, "node_names")
        pin2node_map = getattr(pydb, "pin2node_map")
        if len(pin2node_map) != len(pin_names):
            return None
        canonical_map = {}
        for pin_id, pin_name in enumerate(pin_names):
            node_id = self._snapshot_id_value(pin2node_map, pin_id)
            if node_id is None or node_id < 0 or node_id >= len(node_names):
                return None
            canonical_map[pin_name] = node_names[node_id]
        return canonical_map

    def _canonical_pin_to_net_names(self, pydb):
        if not hasattr(pydb, "pin2net_map"):
            return None
        pin_names = self._pydb_names_as_str_list(pydb, "pin_names")
        net_names = self._pydb_names_as_str_list(pydb, "net_names")
        pin2net_map = getattr(pydb, "pin2net_map")
        if len(pin2net_map) != len(pin_names):
            return None
        canonical_map = {}
        for pin_id, pin_name in enumerate(pin_names):
            net_id = self._snapshot_id_value(pin2net_map, pin_id)
            if net_id is None or net_id < 0 or net_id >= len(net_names):
                return None
            canonical_map[pin_name] = net_names[net_id]
        return canonical_map

    def _canonical_net_to_pin_names(self, pydb):
        if not (
            hasattr(pydb, "flat_net2pin_map")
            and hasattr(pydb, "flat_net2pin_start_map")
        ):
            return None
        pin_names = self._pydb_names_as_str_list(pydb, "pin_names")
        net_names = self._pydb_names_as_str_list(pydb, "net_names")
        flat_net2pin_map = getattr(pydb, "flat_net2pin_map")
        flat_net2pin_start_map = getattr(pydb, "flat_net2pin_start_map")
        if len(flat_net2pin_start_map) != len(net_names) + 1:
            return None
        canonical_map = {}
        for net_id, net_name in enumerate(net_names):
            start_id = self._snapshot_id_value(flat_net2pin_start_map, net_id)
            end_id = self._snapshot_id_value(flat_net2pin_start_map, net_id + 1)
            if (
                start_id is None
                or end_id is None
                or end_id < start_id
                or end_id > len(flat_net2pin_map)
            ):
                return None
            net_pin_names = []
            for flat_id in range(start_id, end_id):
                pin_id = self._snapshot_id_value(flat_net2pin_map, flat_id)
                if pin_id is None or pin_id < 0 or pin_id >= len(pin_names):
                    return None
                net_pin_names.append(pin_names[pin_id])
            canonical_map[net_name] = tuple(sorted(net_pin_names))
        return canonical_map

    def _connectivity_fact_changed(self, old_pydb, new_pydb, attr_names, canonicalizer):
        old_canonical = canonicalizer(old_pydb)
        new_canonical = canonicalizer(new_pydb)
        if old_canonical is not None and new_canonical is not None:
            return old_canonical != new_canonical
        if not all(
            hasattr(old_pydb, attr_name) and hasattr(new_pydb, attr_name)
            for attr_name in attr_names
        ):
            return False
        old_value = tuple(
            self._normalize_snapshot_value(getattr(old_pydb, attr_name))
            for attr_name in attr_names
        )
        new_value = tuple(
            self._normalize_snapshot_value(getattr(new_pydb, attr_name))
            for attr_name in attr_names
        )
        return old_value != new_value

    def _canonical_node_metadata(self, pydb):
        node_names = self._pydb_names_as_str_list(pydb, "node_names")
        metadata = []
        for node_id, node_name in enumerate(node_names):
            metadata.append(
                (
                    node_name,
                    self._snapshot_sequence_value(
                        getattr(pydb, "node_master_names", None), node_id
                    ),
                    self._snapshot_sequence_value(
                        getattr(pydb, "node_size_x", None), node_id
                    ),
                    self._snapshot_sequence_value(
                        getattr(pydb, "node_size_y", None), node_id
                    ),
                    self._snapshot_sequence_value(
                        getattr(pydb, "node_orient", None), node_id
                    ),
                )
            )
        return tuple(sorted(metadata))

    def _raw_snapshot_attrs(self, pydb, attr_names):
        return tuple(
            self._normalize_snapshot_value(getattr(pydb, attr_name, None))
            for attr_name in attr_names
        )

    def _canonical_connectivity_metadata(self, pydb):
        connectivity = []
        for label, attr_names, canonicalizer in (
            ("pin_to_node", ("pin2node_map",), self._canonical_pin_to_node_names),
            ("pin_to_net", ("pin2net_map",), self._canonical_pin_to_net_names),
            (
                "net_to_pins",
                ("flat_net2pin_map", "flat_net2pin_start_map"),
                self._canonical_net_to_pin_names,
            ),
        ):
            canonical = canonicalizer(pydb)
            if canonical is None:
                canonical = self._raw_snapshot_attrs(pydb, attr_names)
            connectivity.append((label, canonical))
        return tuple(connectivity)

    def _snapshot_node_positions(self, pos):
        old_node_x = np.array(self.node_x, copy=True)
        old_node_y = np.array(self.node_y, copy=True)
        if pos is None:
            return old_node_x, old_node_y

        if hasattr(pos, "detach"):
            pos = pos.detach()
        if hasattr(pos, "cpu"):
            pos = pos.cpu()
        if hasattr(pos, "numpy"):
            pos = pos.numpy()

        movable_count = self.num_movable_nodes
        old_node_x[:movable_count] = pos[:movable_count]
        old_node_y[:movable_count] = pos[self.num_nodes:self.num_nodes + movable_count]
        return old_node_x, old_node_y

    def _snapshot_node_is_buffer(self, pydb, node_name_to_id, node_name):
        values = getattr(pydb, "node_is_buffer", None)
        node_id = node_name_to_id.get(node_name)
        if values is None or node_id is None:
            return False
        value = self._snapshot_sequence_value(values, node_id)
        if value is None:
            return False
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)

    def _build_buffer_churn_counts(
        self,
        old_pydb,
        new_pydb,
        old_node_name_to_id,
        new_node_name_to_id,
        added_node_names,
        removed_node_names,
        surviving_node_names,
    ):
        def _old_is_buffer(node_name):
            return self._snapshot_node_is_buffer(
                old_pydb, old_node_name_to_id, node_name
            )

        def _new_is_buffer(node_name):
            return self._snapshot_node_is_buffer(
                new_pydb, new_node_name_to_id, node_name
            )

        # Identity rule: buffers removed by unbuffer are non-surviving, buffers
        # inserted later in the same transaction are added, and unchanged
        # non-buffer instances remain ordinary surviving objects.
        added_buffer_count = sum(
            1 for node_name in added_node_names if _new_is_buffer(node_name)
        )
        removed_buffer_count = sum(
            1 for node_name in removed_node_names if _old_is_buffer(node_name)
        )
        surviving_buffer_count = sum(
            1
            for node_name in surviving_node_names
            if _old_is_buffer(node_name) and _new_is_buffer(node_name)
        )
        return {
            "added_buffer_count": int(added_buffer_count),
            "removed_buffer_count": int(removed_buffer_count),
            "surviving_buffer_count": int(surviving_buffer_count),
        }

    _BUFFER_ONLY_POLICY_SAMPLE_LIMIT = 50

    def _snapshot_field_value(self, pydb, attr_name, name_to_id, node_name):
        values = getattr(pydb, attr_name, None)
        node_id = name_to_id.get(node_name)
        if values is None or node_id is None:
            return False, None
        try:
            if node_id >= len(values):
                return False, None
            return True, self._normalize_snapshot_value(values[node_id])
        except TypeError:
            return False, None

    def _snapshot_node_buffer_status(self, pydb, node_name_to_id, node_name):
        values = getattr(pydb, "node_is_buffer", None)
        node_id = node_name_to_id.get(node_name)
        if values is None or node_id is None:
            return None
        try:
            if node_id >= len(values):
                return None
        except TypeError:
            return None
        value = self._snapshot_sequence_value(values, node_id)
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes"):
                return True
            if normalized in ("0", "false", "no"):
                return False
            return None
        return bool(value)

    def _empty_buffer_only_policy(self):
        return {
            "status": "clean",
            "allowed_change_counts": {
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "resized_buffer_count": 0,
                "reoriented_buffer_count": 0,
                "moved_instance_count": 0,
            },
            "violation_counts": {
                "non_buffer_master_change_count": 0,
                "non_buffer_size_change_count": 0,
                "non_buffer_orient_change_count": 0,
                "added_non_buffer_count": 0,
                "removed_non_buffer_count": 0,
                "connectivity_violation_count": 0,
                "total_violation_count": 0,
            },
            "summary": {
                "non_buffer_changed_instance_count": 0,
                "buffer_changed_instance_count": 0,
                "unknown_reason_count": 0,
            },
            "violation_samples": [],
            "unknown_reasons": [],
            "sample_limit": self._BUFFER_ONLY_POLICY_SAMPLE_LIMIT,
            "samples_truncated": False,
        }

    def _record_buffer_only_unknown(self, policy, reason, node_name=None, field_name=None):
        entry = {"reason": reason}
        if node_name is not None:
            entry["node_name"] = node_name
        if field_name is not None:
            entry["field_name"] = field_name
        policy["unknown_reasons"].append(entry)
        policy["summary"]["unknown_reason_count"] = len(policy["unknown_reasons"])

    def _record_buffer_only_violation(
        self,
        policy,
        node_name,
        change_kinds,
        old_pydb,
        new_pydb,
        old_node_name_to_id,
        new_node_name_to_id,
        old_is_buffer,
        new_is_buffer,
    ):
        for change_kind in change_kinds:
            if change_kind == "master":
                policy["violation_counts"]["non_buffer_master_change_count"] += 1
            elif change_kind == "size":
                policy["violation_counts"]["non_buffer_size_change_count"] += 1
            elif change_kind == "orient":
                policy["violation_counts"]["non_buffer_orient_change_count"] += 1
        policy["violation_counts"]["total_violation_count"] = sum(
            policy["violation_counts"][field_name]
            for field_name in (
                "non_buffer_master_change_count",
                "non_buffer_size_change_count",
                "non_buffer_orient_change_count",
                "added_non_buffer_count",
                "removed_non_buffer_count",
                "connectivity_violation_count",
            )
        )
        if len(policy["violation_samples"]) >= policy["sample_limit"]:
            policy["samples_truncated"] = True
            return

        def field(pydb, attr_name, name_to_id):
            has_value, value = self._snapshot_field_value(
                pydb, attr_name, name_to_id, node_name
            )
            return value if has_value else None

        policy["violation_samples"].append(
            {
                "node_name": node_name,
                "change_kinds": list(change_kinds),
                "old_master": field(old_pydb, "node_master_names", old_node_name_to_id),
                "new_master": field(new_pydb, "node_master_names", new_node_name_to_id),
                "old_size_x": field(old_pydb, "node_size_x", old_node_name_to_id),
                "old_size_y": field(old_pydb, "node_size_y", old_node_name_to_id),
                "new_size_x": field(new_pydb, "node_size_x", new_node_name_to_id),
                "new_size_y": field(new_pydb, "node_size_y", new_node_name_to_id),
                "old_orient": field(old_pydb, "node_orient", old_node_name_to_id),
                "new_orient": field(new_pydb, "node_orient", new_node_name_to_id),
                "old_is_buffer": old_is_buffer,
                "new_is_buffer": new_is_buffer,
            }
        )

    def _record_buffer_only_topology_violation_sample(
        self,
        policy,
        node_name,
        change_kind,
        old_pydb,
        new_pydb,
        old_node_name_to_id,
        new_node_name_to_id,
        old_is_buffer,
        new_is_buffer,
    ):
        if len(policy["violation_samples"]) >= policy["sample_limit"]:
            policy["samples_truncated"] = True
            return

        def field(pydb, attr_name, name_to_id):
            has_value, value = self._snapshot_field_value(
                pydb, attr_name, name_to_id, node_name
            )
            return value if has_value else None

        policy["violation_samples"].append(
            {
                "node_name": node_name,
                "change_kinds": [change_kind],
                "old_master": field(old_pydb, "node_master_names", old_node_name_to_id),
                "new_master": field(new_pydb, "node_master_names", new_node_name_to_id),
                "old_size_x": field(old_pydb, "node_size_x", old_node_name_to_id),
                "old_size_y": field(old_pydb, "node_size_y", old_node_name_to_id),
                "new_size_x": field(new_pydb, "node_size_x", new_node_name_to_id),
                "new_size_y": field(new_pydb, "node_size_y", new_node_name_to_id),
                "old_orient": field(old_pydb, "node_orient", old_node_name_to_id),
                "new_orient": field(new_pydb, "node_orient", new_node_name_to_id),
                "old_is_buffer": old_is_buffer,
                "new_is_buffer": new_is_buffer,
            }
        )

    def _record_buffer_only_connectivity_violation_sample(self, policy, sample):
        if len(policy["violation_samples"]) >= policy["sample_limit"]:
            policy["samples_truncated"] = True
            return
        policy["violation_samples"].append(sample)

    def _snapshot_buffer_node_names(self, pydb, node_name_to_id, node_names):
        buffer_node_names = set()
        for node_name in node_names:
            if self._snapshot_node_buffer_status(pydb, node_name_to_id, node_name):
                buffer_node_names.add(node_name)
        return buffer_node_names

    def _buffer_churn_affected_net_names(
        self,
        old_pin_to_node,
        new_pin_to_node,
        old_pin_to_net,
        new_pin_to_net,
        added_buffer_node_names,
        removed_buffer_node_names,
        surviving_buffer_node_names=None,
        maybe_added_buffer_node_names=None,
        maybe_removed_buffer_node_names=None,
    ):
        surviving_buffer_node_names = surviving_buffer_node_names or set()
        maybe_added_buffer_node_names = maybe_added_buffer_node_names or set()
        maybe_removed_buffer_node_names = maybe_removed_buffer_node_names or set()
        affected_net_names = set()
        for pin_name, node_name in old_pin_to_node.items():
            if (
                node_name in removed_buffer_node_names
                or node_name in maybe_removed_buffer_node_names
            ):
                net_name = old_pin_to_net.get(pin_name)
                if net_name is not None:
                    affected_net_names.add(net_name)
        for pin_name, node_name in new_pin_to_node.items():
            if (
                node_name in added_buffer_node_names
                or node_name in maybe_added_buffer_node_names
            ):
                net_name = new_pin_to_net.get(pin_name)
                if net_name is not None:
                    affected_net_names.add(net_name)
        common_pin_names = set(old_pin_to_node) & set(new_pin_to_node)
        for pin_name in common_pin_names:
            old_node_name = old_pin_to_node.get(pin_name)
            new_node_name = new_pin_to_node.get(pin_name)
            if (
                old_node_name not in surviving_buffer_node_names
                or new_node_name not in surviving_buffer_node_names
                or old_node_name != new_node_name
            ):
                continue
            old_net_name = old_pin_to_net.get(pin_name)
            new_net_name = new_pin_to_net.get(pin_name)
            if old_net_name == new_net_name:
                continue
            if old_net_name is not None:
                affected_net_names.add(old_net_name)
            if new_net_name is not None:
                affected_net_names.add(new_net_name)
        return affected_net_names

    def _buffer_churn_affected_net_source_samples(
        self,
        old_pin_to_node,
        new_pin_to_node,
        old_pin_to_net,
        new_pin_to_net,
        added_buffer_node_names,
        removed_buffer_node_names,
        surviving_buffer_node_names=None,
        maybe_added_buffer_node_names=None,
        maybe_removed_buffer_node_names=None,
        sample_limit=20,
    ):
        surviving_buffer_node_names = surviving_buffer_node_names or set()
        maybe_added_buffer_node_names = maybe_added_buffer_node_names or set()
        maybe_removed_buffer_node_names = maybe_removed_buffer_node_names or set()
        samples = []
        for pin_name in sorted(old_pin_to_node):
            node_name = old_pin_to_node.get(pin_name)
            if (
                node_name not in removed_buffer_node_names
                and node_name not in maybe_removed_buffer_node_names
            ):
                continue
            samples.append(
                {
                    "source": (
                        "removed_buffer_pin"
                        if node_name in removed_buffer_node_names
                        else "maybe_removed_buffer_pin"
                    ),
                    "pin_name": pin_name,
                    "node_name": node_name,
                    "net_name": old_pin_to_net.get(pin_name),
                }
            )
            if len(samples) >= sample_limit:
                return samples
        for pin_name in sorted(new_pin_to_node):
            node_name = new_pin_to_node.get(pin_name)
            if (
                node_name not in added_buffer_node_names
                and node_name not in maybe_added_buffer_node_names
            ):
                continue
            samples.append(
                {
                    "source": (
                        "added_buffer_pin"
                        if node_name in added_buffer_node_names
                        else "maybe_added_buffer_pin"
                    ),
                    "pin_name": pin_name,
                    "node_name": node_name,
                    "net_name": new_pin_to_net.get(pin_name),
                }
            )
            if len(samples) >= sample_limit:
                return samples
        common_pin_names = sorted(set(old_pin_to_node) & set(new_pin_to_node))
        for pin_name in common_pin_names:
            old_node_name = old_pin_to_node.get(pin_name)
            new_node_name = new_pin_to_node.get(pin_name)
            if (
                old_node_name not in surviving_buffer_node_names
                or new_node_name not in surviving_buffer_node_names
                or old_node_name != new_node_name
            ):
                continue
            old_net_name = old_pin_to_net.get(pin_name)
            new_net_name = new_pin_to_net.get(pin_name)
            if old_net_name == new_net_name:
                continue
            samples.append(
                {
                    "source": "surviving_buffer_pin_rewire",
                    "pin_name": pin_name,
                    "node_name": old_node_name,
                    "old_net_name": old_net_name,
                    "new_net_name": new_net_name,
                }
            )
            if len(samples) >= sample_limit:
                return samples
        return samples

    def _buffer_only_connectivity_has_non_buffer_violation(
        self,
        old_pydb,
        new_pydb,
        old_node_name_to_id,
        new_node_name_to_id,
        added_node_names,
        removed_node_names,
        surviving_node_names,
    ):
        old_pin_to_node = self._canonical_pin_to_node_names(old_pydb)
        new_pin_to_node = self._canonical_pin_to_node_names(new_pydb)
        old_pin_to_net = self._canonical_pin_to_net_names(old_pydb)
        new_pin_to_net = self._canonical_pin_to_net_names(new_pydb)
        if (
            old_pin_to_node is None
            or new_pin_to_node is None
            or old_pin_to_net is None
            or new_pin_to_net is None
        ):
            return None

        added_buffer_node_names = self._snapshot_buffer_node_names(
            new_pydb, new_node_name_to_id, added_node_names
        )
        removed_buffer_node_names = self._snapshot_buffer_node_names(
            old_pydb, old_node_name_to_id, removed_node_names
        )
        maybe_added_buffer_node_names = set()
        for node_name in added_node_names:
            if self._snapshot_node_buffer_status(
                new_pydb, new_node_name_to_id, node_name
            ) is None:
                maybe_added_buffer_node_names.add(node_name)
        maybe_removed_buffer_node_names = set()
        for node_name in removed_node_names:
            if self._snapshot_node_buffer_status(
                old_pydb, old_node_name_to_id, node_name
            ) is None:
                maybe_removed_buffer_node_names.add(node_name)
        surviving_buffer_node_names = set()
        for node_name in surviving_node_names:
            old_is_buffer = self._snapshot_node_buffer_status(
                old_pydb, old_node_name_to_id, node_name
            )
            new_is_buffer = self._snapshot_node_buffer_status(
                new_pydb, new_node_name_to_id, node_name
            )
            if old_is_buffer is True and new_is_buffer is True:
                surviving_buffer_node_names.add(node_name)
        affected_net_names = self._buffer_churn_affected_net_names(
            old_pin_to_node,
            new_pin_to_node,
            old_pin_to_net,
            new_pin_to_net,
            added_buffer_node_names,
            removed_buffer_node_names,
            surviving_buffer_node_names,
            maybe_added_buffer_node_names,
            maybe_removed_buffer_node_names,
        )
        if not affected_net_names and (
            added_buffer_node_names
            or removed_buffer_node_names
            or maybe_added_buffer_node_names
            or maybe_removed_buffer_node_names
        ):
            return None

        affected_net_source_samples = self._buffer_churn_affected_net_source_samples(
            old_pin_to_node,
            new_pin_to_node,
            old_pin_to_net,
            new_pin_to_net,
            added_buffer_node_names,
            removed_buffer_node_names,
            surviving_buffer_node_names,
            maybe_added_buffer_node_names,
            maybe_removed_buffer_node_names,
        )
        affected_net_names_sample = sorted(affected_net_names)[:20]
        affected_net_count = len(affected_net_names)

        def sample(
            reason,
            pin_name,
            old_node_name=None,
            new_node_name=None,
            old_net_name=None,
            new_net_name=None,
            old_node_is_buffer=None,
            new_node_is_buffer=None,
        ):
            return {
                "change_kinds": ["connectivity"],
                "reason": reason,
                "pin_name": pin_name,
                "old_node_name": old_node_name,
                "new_node_name": new_node_name,
                "old_net_name": old_net_name,
                "new_net_name": new_net_name,
                "old_node_is_buffer": old_node_is_buffer,
                "new_node_is_buffer": new_node_is_buffer,
                "old_net_is_buffer_churn_affected": old_net_name in affected_net_names,
                "new_net_is_buffer_churn_affected": new_net_name in affected_net_names,
                "affected_net_names_sample": affected_net_names_sample,
                "affected_net_count": affected_net_count,
                "affected_net_source_samples": affected_net_source_samples,
            }

        surviving_non_buffer_node_names = set()
        unknown_surviving_node_names = set()
        surviving_node_buffer_status = {}
        for node_name in surviving_node_names:
            old_is_buffer = self._snapshot_node_buffer_status(
                old_pydb, old_node_name_to_id, node_name
            )
            new_is_buffer = self._snapshot_node_buffer_status(
                new_pydb, new_node_name_to_id, node_name
            )
            surviving_node_buffer_status[node_name] = (old_is_buffer, new_is_buffer)
            if old_is_buffer is None or new_is_buffer is None:
                unknown_surviving_node_names.add(node_name)
            if old_is_buffer is False and new_is_buffer is False:
                surviving_non_buffer_node_names.add(node_name)

        def unexplained_net_change(old_net_name, new_net_name):
            return (
                old_net_name != new_net_name
                and not (
                    old_net_name in affected_net_names
                    and new_net_name in affected_net_names
                )
            )

        common_pin_names = set(old_pin_to_node) & set(new_pin_to_node)
        for pin_name in common_pin_names:
            old_node_name = old_pin_to_node.get(pin_name)
            new_node_name = new_pin_to_node.get(pin_name)
            if (
                old_node_name in unknown_surviving_node_names
                or new_node_name in unknown_surviving_node_names
            ):
                if old_node_name != new_node_name or old_pin_to_net.get(
                    pin_name
                ) != new_pin_to_net.get(pin_name):
                    return None
            if (
                old_node_name in surviving_non_buffer_node_names
                or new_node_name in surviving_non_buffer_node_names
            ):
                old_is_buffer, _ = surviving_node_buffer_status.get(
                    old_node_name, (None, None)
                )
                _, new_is_buffer = surviving_node_buffer_status.get(
                    new_node_name, (None, None)
                )
                if old_node_name != new_node_name:
                    return sample(
                        "non_buffer_pin_node_changed",
                        pin_name,
                        old_node_name=old_node_name,
                        new_node_name=new_node_name,
                        old_net_name=old_pin_to_net.get(pin_name),
                        new_net_name=new_pin_to_net.get(pin_name),
                        old_node_is_buffer=old_is_buffer,
                        new_node_is_buffer=new_is_buffer,
                    )
                if unexplained_net_change(
                    old_pin_to_net.get(pin_name), new_pin_to_net.get(pin_name)
                ):
                    return sample(
                        "non_buffer_pin_net_changed",
                        pin_name,
                        old_node_name=old_node_name,
                        new_node_name=new_node_name,
                        old_net_name=old_pin_to_net.get(pin_name),
                        new_net_name=new_pin_to_net.get(pin_name),
                        old_node_is_buffer=old_is_buffer,
                        new_node_is_buffer=new_is_buffer,
                    )

        for pin_name in set(new_pin_to_node) - set(old_pin_to_node):
            node_name = new_pin_to_node.get(pin_name)
            if node_name in unknown_surviving_node_names:
                return None
            if node_name in surviving_non_buffer_node_names:
                _, new_is_buffer = surviving_node_buffer_status.get(
                    node_name, (None, None)
                )
                return sample(
                    "non_buffer_pin_added",
                    pin_name,
                    new_node_name=node_name,
                    new_net_name=new_pin_to_net.get(pin_name),
                    new_node_is_buffer=new_is_buffer,
                )
        for pin_name in set(old_pin_to_node) - set(new_pin_to_node):
            node_name = old_pin_to_node.get(pin_name)
            if node_name in unknown_surviving_node_names:
                return None
            if node_name in surviving_non_buffer_node_names:
                old_is_buffer, _ = surviving_node_buffer_status.get(
                    node_name, (None, None)
                )
                return sample(
                    "non_buffer_pin_removed",
                    pin_name,
                    old_node_name=node_name,
                    old_net_name=old_pin_to_net.get(pin_name),
                    old_node_is_buffer=old_is_buffer,
                )
        return False

    def _build_buffer_only_policy_audit(
        self,
        old_pydb,
        new_pydb,
        old_node_name_to_id,
        new_node_name_to_id,
        added_node_names,
        removed_node_names,
        surviving_node_names,
        connectivity_changed,
        connectivity_audit_needed,
    ):
        policy = self._empty_buffer_only_policy()
        changed_buffer_nodes = set()
        changed_non_buffer_nodes = set()
        unknown_topology_buffer_status = False

        for node_name in added_node_names:
            is_buffer = self._snapshot_node_buffer_status(
                new_pydb, new_node_name_to_id, node_name
            )
            if is_buffer is None:
                unknown_topology_buffer_status = True
                self._record_buffer_only_unknown(
                    policy, "missing_added_node_buffer_status", node_name, "node_is_buffer"
                )
            elif is_buffer:
                policy["allowed_change_counts"]["added_buffer_count"] += 1
            else:
                policy["violation_counts"]["added_non_buffer_count"] += 1
                changed_non_buffer_nodes.add(node_name)
                self._record_buffer_only_topology_violation_sample(
                    policy,
                    node_name,
                    "added_non_buffer",
                    old_pydb,
                    new_pydb,
                    old_node_name_to_id,
                    new_node_name_to_id,
                    None,
                    is_buffer,
                )

        for node_name in removed_node_names:
            is_buffer = self._snapshot_node_buffer_status(
                old_pydb, old_node_name_to_id, node_name
            )
            if is_buffer is None:
                unknown_topology_buffer_status = True
                self._record_buffer_only_unknown(
                    policy, "missing_removed_node_buffer_status", node_name, "node_is_buffer"
                )
            elif is_buffer:
                policy["allowed_change_counts"]["removed_buffer_count"] += 1
            else:
                policy["violation_counts"]["removed_non_buffer_count"] += 1
                changed_non_buffer_nodes.add(node_name)
                self._record_buffer_only_topology_violation_sample(
                    policy,
                    node_name,
                    "removed_non_buffer",
                    old_pydb,
                    new_pydb,
                    old_node_name_to_id,
                    new_node_name_to_id,
                    is_buffer,
                    None,
                )

        for node_name in surviving_node_names:
            change_kinds = []
            for attr_name, change_kind in (
                ("node_master_names", "master"),
                ("node_size_x", "size"),
                ("node_size_y", "size"),
                ("node_orient", "orient"),
            ):
                old_has_value, old_value = self._snapshot_field_value(
                    old_pydb, attr_name, old_node_name_to_id, node_name
                )
                new_has_value, new_value = self._snapshot_field_value(
                    new_pydb, attr_name, new_node_name_to_id, node_name
                )
                if not old_has_value or not new_has_value:
                    self._record_buffer_only_unknown(
                        policy, "missing_surviving_node_metadata", node_name, attr_name
                    )
                    continue
                if old_value != new_value and change_kind not in change_kinds:
                    change_kinds.append(change_kind)

            moved = False
            for attr_name in ("node_x", "node_y"):
                old_has_value, old_value = self._snapshot_field_value(
                    old_pydb, attr_name, old_node_name_to_id, node_name
                )
                new_has_value, new_value = self._snapshot_field_value(
                    new_pydb, attr_name, new_node_name_to_id, node_name
                )
                if not old_has_value or not new_has_value:
                    self._record_buffer_only_unknown(
                        policy, "missing_surviving_node_metadata", node_name, attr_name
                    )
                    continue
                if old_has_value and new_has_value and old_value != new_value:
                    moved = True
            if moved:
                policy["allowed_change_counts"]["moved_instance_count"] += 1

            if not change_kinds:
                continue

            old_is_buffer = self._snapshot_node_buffer_status(
                old_pydb, old_node_name_to_id, node_name
            )
            new_is_buffer = self._snapshot_node_buffer_status(
                new_pydb, new_node_name_to_id, node_name
            )
            if old_is_buffer is None or new_is_buffer is None:
                self._record_buffer_only_unknown(
                    policy, "missing_surviving_node_buffer_status", node_name, "node_is_buffer"
                )
                continue
            if old_is_buffer and new_is_buffer:
                if "master" in change_kinds or "size" in change_kinds:
                    policy["allowed_change_counts"]["resized_buffer_count"] += 1
                if "orient" in change_kinds:
                    policy["allowed_change_counts"]["reoriented_buffer_count"] += 1
                changed_buffer_nodes.add(node_name)
            else:
                changed_non_buffer_nodes.add(node_name)
                self._record_buffer_only_violation(
                    policy,
                    node_name,
                    change_kinds,
                    old_pydb,
                    new_pydb,
                    old_node_name_to_id,
                    new_node_name_to_id,
                    old_is_buffer,
                    new_is_buffer,
                )

        if connectivity_changed or connectivity_audit_needed:
            unknown_surviving_buffer_status = False
            for node_name in surviving_node_names:
                old_is_buffer = self._snapshot_node_buffer_status(
                    old_pydb, old_node_name_to_id, node_name
                )
                new_is_buffer = self._snapshot_node_buffer_status(
                    new_pydb, new_node_name_to_id, node_name
                )
                if old_is_buffer is None or new_is_buffer is None:
                    unknown_surviving_buffer_status = True
                    self._record_buffer_only_unknown(
                        policy,
                        "missing_surviving_node_buffer_status",
                        node_name,
                        "node_is_buffer",
                    )
            connectivity_violation_sample = (
                self._buffer_only_connectivity_has_non_buffer_violation(
                    old_pydb,
                    new_pydb,
                    old_node_name_to_id,
                    new_node_name_to_id,
                    added_node_names,
                    removed_node_names,
                    surviving_node_names,
                )
            )
            if connectivity_violation_sample is None:
                self._record_buffer_only_unknown(
                    policy,
                    "unattributed_connectivity_change",
                    field_name="connectivity",
                )
            elif connectivity_violation_sample:
                policy["violation_counts"]["connectivity_violation_count"] += 1
                if isinstance(connectivity_violation_sample, dict):
                    self._record_buffer_only_connectivity_violation_sample(
                        policy, connectivity_violation_sample
                    )
            elif unknown_topology_buffer_status or unknown_surviving_buffer_status:
                self._record_buffer_only_unknown(
                    policy,
                    "unattributed_connectivity_change",
                    field_name="connectivity",
                )

        policy["violation_counts"]["total_violation_count"] = sum(
            policy["violation_counts"][field_name]
            for field_name in (
                "non_buffer_master_change_count",
                "non_buffer_size_change_count",
                "non_buffer_orient_change_count",
                "added_non_buffer_count",
                "removed_non_buffer_count",
                "connectivity_violation_count",
            )
        )
        policy["samples_truncated"] = (
            policy["violation_counts"]["total_violation_count"]
            > policy["sample_limit"]
        )
        policy["summary"]["non_buffer_changed_instance_count"] = len(
            changed_non_buffer_nodes
        )
        policy["summary"]["buffer_changed_instance_count"] = len(
            changed_buffer_nodes
        )
        policy["summary"]["unknown_reason_count"] = len(policy["unknown_reasons"])
        if policy["violation_counts"]["total_violation_count"] > 0:
            policy["status"] = "violated"
        elif policy["unknown_reasons"]:
            policy["status"] = "unknown"
        else:
            policy["status"] = "clean"
        return policy

    def _build_topology_sync_contract(self, old_pydb, new_pydb, mutation_hint=None):
        old_node_name_list = self._pydb_names_as_str_list(old_pydb, "node_names")
        new_node_name_list = self._pydb_names_as_str_list(new_pydb, "node_names")
        old_node_names = set(old_node_name_list)
        new_node_names = set(new_node_name_list)
        old_pin_names = set(self._pydb_names_as_str_list(old_pydb, "pin_names"))
        new_pin_names = set(self._pydb_names_as_str_list(new_pydb, "pin_names"))
        old_net_names = set(self._pydb_names_as_str_list(old_pydb, "net_names"))
        new_net_names = set(self._pydb_names_as_str_list(new_pydb, "net_names"))
        old_node_name_to_id = {
            name: node_id for node_id, name in enumerate(old_node_name_list)
        }
        new_node_name_to_id = {
            name: node_id for node_id, name in enumerate(new_node_name_list)
        }
        for name, node_id in getattr(new_pydb, "node_name2id_map", {}).items():
            new_node_name_to_id[self._decode_name(name)] = int(node_id)

        added_node_names = sorted(new_node_names - old_node_names)
        added_pin_names = sorted(new_pin_names - old_pin_names)
        added_net_names = sorted(new_net_names - old_net_names)
        removed_node_names = sorted(old_node_names - new_node_names)
        removed_pin_names = sorted(old_pin_names - new_pin_names)
        removed_net_names = sorted(old_net_names - new_net_names)
        old_counts = {
            "nodes": int(getattr(old_pydb, "num_nodes", len(old_node_name_list))),
            "pins": len(getattr(old_pydb, "pin_names", [])),
            "nets": len(getattr(old_pydb, "net_names", [])),
        }
        new_counts = {
            "nodes": int(getattr(new_pydb, "num_nodes", len(new_node_name_list))),
            "pins": len(getattr(new_pydb, "pin_names", [])),
            "nets": len(getattr(new_pydb, "net_names", [])),
        }
        counts_changed = any(
            (
                old_counts["nodes"] != new_counts["nodes"],
                old_counts["pins"] != new_counts["pins"],
                old_counts["nets"] != new_counts["nets"],
            )
        )
        topology_changed = counts_changed or any(
            (
                added_node_names,
                added_pin_names,
                added_net_names,
                removed_node_names,
                removed_pin_names,
                removed_net_names,
            )
        )
        connectivity_audit_needed = bool(
            added_pin_names
            or added_net_names
            or removed_pin_names
            or removed_net_names
            or old_counts["pins"] != new_counts["pins"]
            or old_counts["nets"] != new_counts["nets"]
        )
        connectivity_changed = False
        for attr_names, canonicalizer in (
            (("pin2node_map",), self._canonical_pin_to_node_names),
            (("pin2net_map",), self._canonical_pin_to_net_names),
            (
                ("flat_net2pin_map", "flat_net2pin_start_map"),
                self._canonical_net_to_pin_names,
            ),
        ):
            if self._connectivity_fact_changed(
                old_pydb, new_pydb, attr_names, canonicalizer
            ):
                topology_changed = True
                connectivity_changed = True
                connectivity_audit_needed = True
                break

        geometry_changed = False
        coordinates_changed = False
        surviving_node_names = sorted(old_node_names & new_node_names)
        buffer_churn_counts = self._build_buffer_churn_counts(
            old_pydb,
            new_pydb,
            old_node_name_to_id,
            new_node_name_to_id,
            added_node_names,
            removed_node_names,
            surviving_node_names,
        )
        for node_name in surviving_node_names:
            for attr_name in (
                "node_master_names",
                "node_size_x",
                "node_size_y",
                "node_orient",
            ):
                old_has_value, old_value = self._snapshot_field_value(
                    old_pydb, attr_name, old_node_name_to_id, node_name
                )
                new_has_value, new_value = self._snapshot_field_value(
                    new_pydb, attr_name, new_node_name_to_id, node_name
                )
                if old_has_value and new_has_value and old_value != new_value:
                    geometry_changed = True
                    break
            for attr_name in ("node_x", "node_y"):
                old_has_value, old_value = self._snapshot_field_value(
                    old_pydb, attr_name, old_node_name_to_id, node_name
                )
                new_has_value, new_value = self._snapshot_field_value(
                    new_pydb, attr_name, new_node_name_to_id, node_name
                )
                if old_has_value and new_has_value and old_value != new_value:
                    coordinates_changed = True
                    break

        if topology_changed:
            mutation_kind = "topology_changed"
        elif geometry_changed:
            mutation_kind = "geometry_changed"
        elif coordinates_changed:
            mutation_kind = "placement_only"
        else:
            mutation_kind = "no_mutation"

        buffer_only_policy = self._build_buffer_only_policy_audit(
            old_pydb,
            new_pydb,
            old_node_name_to_id,
            new_node_name_to_id,
            added_node_names,
            removed_node_names,
            surviving_node_names,
            connectivity_changed,
            connectivity_audit_needed,
        )

        return {
            "mutation_kind": mutation_kind,
            "requires_runtimedb_rebuild": mutation_kind in (
                "geometry_changed",
                "topology_changed",
            ),
            "old_counts": old_counts,
            "new_counts": new_counts,
            "added_names": {
                "nodes": added_node_names,
                "pins": added_pin_names,
                "nets": added_net_names,
            },
            "removed_names": {
                "nodes": removed_node_names,
                "pins": removed_pin_names,
                "nets": removed_net_names,
            },
            "surviving_name_to_new_id": {
                node_name: int(new_node_name_to_id[node_name])
                for node_name in surviving_node_names
            },
            "added_buffer_count": buffer_churn_counts["added_buffer_count"],
            "removed_buffer_count": buffer_churn_counts["removed_buffer_count"],
            "surviving_buffer_count": buffer_churn_counts[
                "surviving_buffer_count"
            ],
            "buffer_only_policy": buffer_only_policy,
            "stable_identity_key": "name",
            "legacy_ids_stable": False,
        }

    def export_openroad_topology_snapshot(self):
        if self.openroad_bridge is None:
            return None
        import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

        return placeio_openroad.PlaceIOFunction.export_topology_snapshot(
            self.openroad_bridge
        )

    def refresh_topology_from_openroad_snapshot(self, params, new_pydb, old_positions=None):
        self.pydb = new_pydb
        self.params = params
        self._restore_topology_refresh_params(params)
        self.initialize_from_rawdb(self.pydb, params)
        self.node_name2id_map = {
            self._decode_name(name): node_id
            for name, node_id in getattr(self, "node_name2id_map", {}).items()
        }
        self.initialize(params)

        if not old_positions:
            return
        for node_name, (node_x, node_y) in old_positions.items():
            node_id = self.node_name2id_map.get(node_name)
            if node_id is None or node_id >= self.num_physical_nodes:
                continue
            self.node_x[node_id] = node_x
            self.node_y[node_id] = node_y

    def synchronize_topology_from_openroad(self, params, pos=None, mutation_hint=None):
        if self.openroad_bridge is None:
            return {
                "mutation_kind": "placement_only",
                "requires_runtimedb_rebuild": False,
            }

        old_pydb = self.pydb
        old_node_names = self._node_names_as_str_list()
        old_node_x, old_node_y = self._snapshot_node_positions(pos)
        old_positions = {
            name: (old_node_x[node_id], old_node_y[node_id])
            for node_id, name in enumerate(old_node_names[: len(old_node_x)])
        }

        new_pydb = self.export_openroad_topology_snapshot()
        contract = self._build_topology_sync_contract(old_pydb, new_pydb)
        self.refresh_topology_from_openroad_snapshot(params, new_pydb, old_positions)
        contract["snapshot_exported"] = True
        contract["runtimedb_rebuild_owner"] = "dreamplace_runtimedb"
        return contract

    def create_openroad_handoff_controller(self, params):
        if getattr(params, "place_io_engine", "ieda") != "openroad":
            return None
        if self.openroad_bridge is None:
            return None
        controller = OpenRoadHandoffController.from_params(params)
        if not controller.enabled:
            return None
        return controller

    def _get_openroad_session_identity(self):
        if self.openroad_bridge is None:
            return None
        for attr_name in ("session_identity", "session_id"):
            attr_value = getattr(self.openroad_bridge, attr_name, None)
            if attr_value:
                return attr_value
        return "openroad-bridge-%s" % id(self.openroad_bridge)

    def _build_handoff_counts_summary(self):
        pydb = getattr(self, "pydb", None)
        if pydb is not None:
            return {
                "nodes": int(getattr(pydb, "num_nodes", 0)),
                "pins": len(getattr(pydb, "pin_names", [])),
                "nets": len(getattr(pydb, "net_names", [])),
            }
        return {
            "nodes": int(getattr(self, "num_physical_nodes", 0)),
            "pins": len(getattr(self, "pin_names", []) or []),
            "nets": len(getattr(self, "net_names", []) or []),
        }

    def current_topology_counts(self):
        return self._build_handoff_counts_summary()

    def _build_handoff_snapshot_fingerprint(self, counts):
        pydb = getattr(self, "pydb", None)
        if pydb is not None:
            node_names = self._pydb_names_as_str_list(pydb, "node_names")
            pin_names = self._pydb_names_as_str_list(pydb, "pin_names")
            net_names = self._pydb_names_as_str_list(pydb, "net_names")
            node_metadata = self._canonical_node_metadata(pydb)
            connectivity_metadata = self._canonical_connectivity_metadata(pydb)
        else:
            node_names = getattr(self, "node_names", []) or []
            pin_names = getattr(self, "pin_names", []) or []
            net_names = getattr(self, "net_names", []) or []
            node_metadata = self._canonical_node_metadata(self)
            connectivity_metadata = self._canonical_connectivity_metadata(self)
        return "nodes=%s|pins=%s|nets=%s" % (
            counts.get("nodes", 0),
            counts.get("pins", 0),
            counts.get("nets", 0),
        ) + "|node_digest=%s|pin_digest=%s|net_digest=%s|node_meta_digest=%s|connectivity_digest=%s" % (
            self._hash_name_list(node_names),
            self._hash_name_list(pin_names),
            self._hash_name_list(net_names),
            self._hash_snapshot_payload(node_metadata),
            self._hash_snapshot_payload(connectivity_metadata),
        )

    def current_snapshot_fingerprint(self, counts=None):
        if counts is None:
            counts = self.current_topology_counts()
        return self._build_handoff_snapshot_fingerprint(counts)

    def _handoff_buffer_churn_count(self, handoff_result, sync_contract, field_name, mutation_kind):
        if field_name in sync_contract:
            return self._validate_buffer_churn_count(
                field_name, sync_contract[field_name], mutation_kind
            )
        if field_name in handoff_result:
            return self._validate_buffer_churn_count(
                field_name, handoff_result[field_name], mutation_kind
            )
        if mutation_kind == "topology_changed":
            raise RuntimeError(
                "topology_changed handoff missing buffer churn count: %s"
                % field_name
            )
        return 0

    def _validate_buffer_churn_count(self, field_name, value, mutation_kind):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise RuntimeError(
                "%s handoff invalid buffer churn count: %s"
                % (mutation_kind, field_name)
            )
        return int(value)

    def _build_mutation_result_from_handoff(self, handoff_result, pre_counts):
        sync_contract = handoff_result.get("sync_contract", {}) or {}
        mutation_kind = (
            sync_contract.get("mutation_kind")
            or handoff_result.get("mutation_kind")
            or ("placement_only" if handoff_result.get("executed") else "no_mutation")
        )
        requires_runtimedb_rebuild = bool(
            sync_contract.get(
                "requires_runtimedb_rebuild",
                handoff_result.get("requires_runtimedb_rebuild", False),
            )
        )
        post_counts = sync_contract.get("new_counts")
        if mutation_kind == "placement_only" and post_counts is None:
            post_counts = dict(pre_counts)
        added_names = sync_contract.get("added_names", {}) or {}
        removed_names = sync_contract.get("removed_names", {}) or {}
        added_buffer_count = self._handoff_buffer_churn_count(
            handoff_result,
            sync_contract,
            "added_buffer_count",
            mutation_kind,
        )
        removed_buffer_count = self._handoff_buffer_churn_count(
            handoff_result,
            sync_contract,
            "removed_buffer_count",
            mutation_kind,
        )
        surviving_buffer_count = self._handoff_buffer_churn_count(
            handoff_result,
            sync_contract,
            "surviving_buffer_count",
            mutation_kind,
        )
        identity_summary = {
            "stable_identity_key": sync_contract.get("stable_identity_key"),
            "legacy_ids_stable": sync_contract.get("legacy_ids_stable"),
            "surviving_name_to_new_id": dict(
                sync_contract.get("surviving_name_to_new_id", {}) or {}
            ),
        }
        return {
            "mutation_kind": mutation_kind,
            "requires_runtimedb_rebuild": requires_runtimedb_rebuild,
            "post_counts": post_counts,
            "post_snapshot_fingerprint": handoff_result.get("post_snapshot_fingerprint"),
            "identity_summary": identity_summary,
            "added_names": {
                "nodes": list(added_names.get("nodes", []) or []),
                "pins": list(added_names.get("pins", []) or []),
                "nets": list(added_names.get("nets", []) or []),
            },
            "removed_names": {
                "nodes": list(removed_names.get("nodes", []) or []),
                "pins": list(removed_names.get("pins", []) or []),
                "nets": list(removed_names.get("nets", []) or []),
            },
            "added_buffer_count": int(added_buffer_count),
            "removed_buffer_count": int(removed_buffer_count),
            "surviving_buffer_count": int(surviving_buffer_count),
            "buffer_only_policy": copy.deepcopy(
                sync_contract.get("buffer_only_policy")
            ),
            "runtimedb_rebuild_owner": sync_contract.get("runtimedb_rebuild_owner"),
        }

    def create_or_reset_handoff_session(self, session_id=None):
        if session_id is None:
            session_id = self._get_openroad_session_identity()
            if session_id is None:
                session_id = "macroplacedb-handoff"
        self.handoff_session = PlacementHandoffSession(session_id=session_id)
        return self.handoff_session

    def execute_openroad_handoff(self, params, pos, handoff_event):
        pre_counts = self.current_topology_counts()
        pre_snapshot_fingerprint = self.current_snapshot_fingerprint(pre_counts)
        handoff_result = self.run_openroad_buffer_insertion(params, pos, handoff_event)
        handoff_result["mutation_attempted"] = bool(handoff_result.get("executed", False))
        handoff_result["pre_counts"] = dict(pre_counts)
        handoff_result["pre_snapshot_fingerprint"] = pre_snapshot_fingerprint
        handoff_result["post_snapshot_fingerprint"] = self.current_snapshot_fingerprint()
        try:
            handoff_result.update(
                self._build_mutation_result_from_handoff(handoff_result, pre_counts)
            )
        except Exception as exc:
            handoff_payload = {}
            for field_name in (
                "buffer_insertion_strategy",
                "buffer_insertion_command",
                "buffer_insertion_strategy_profile",
                "buffer_insertion_strategy_kind",
                "buffer_insertion_strategy_experimental",
                "buffer_insertion_strategy_validation",
                "buffer_insertion_strategy_non_buffer_mutation_free",
            ):
                if field_name in handoff_result:
                    handoff_payload[field_name] = handoff_result[field_name]
            raise OpenRoadHandoffExecutionError(
                "mutation",
                str(exc),
                handoff_payload=handoff_payload,
            ) from exc
        return handoff_result

    def _extract_unscaled_movable_positions(self, pos):
        if hasattr(pos, "detach"):
            pos = pos.detach()
        if hasattr(pos, "cpu"):
            pos = pos.cpu()
        if hasattr(pos, "numpy"):
            pos = pos.numpy()

        movable_count = self.num_movable_nodes
        node_x = pos[:movable_count]
        node_y = pos[self.num_nodes:self.num_nodes + movable_count]
        return self.unscale_pl_positions(node_x, node_y)

    def unscale_pl_positions(self, node_x, node_y):
        unscale_factor = 1.0 / self.params.scale_factor
        return (
            node_x * unscale_factor + self.params.shift_factor[0],
            node_y * unscale_factor + self.params.shift_factor[1],
        )

    def run_openroad_buffer_insertion(self, params, pos, handoff_event):
        result = {
            "triggered": False,
            "executed": False,
            "requires_sync_back": False,
            "requires_runtimedb_rebuild": False,
            "handoff_event": handoff_event,
        }
        if getattr(params, "place_io_engine", "ieda") != "openroad":
            return result
        if self.openroad_bridge is None:
            return result

        handoff_config = getattr(params, "openroad_handoff", {}) or {}
        buffer_cfg = handoff_config.get("buffer_insertion", {}) or {}
        if not buffer_cfg.get("enabled", False):
            logging.info("OpenROAD handoff skipped because buffer insertion is disabled")
            return result

        import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

        node_x, node_y = self._extract_unscaled_movable_positions(pos)
        try:
            placeio_openroad.PlaceIOFunction.sync_to_openroad(
                self.openroad_bridge, node_x, node_y
            )
        except Exception as exc:
            raise OpenRoadHandoffExecutionError("sync", str(exc)) from exc
        result["triggered"] = True
        result["sync_to_openroad_done"] = True

        strategy = buffer_cfg.get("strategy", "repair_design")
        options = buffer_cfg.get("options", {}) or {}
        strategy_profile = (
            placeio_openroad.PlaceIOFunction.describe_buffer_insertion_strategy(
                strategy
            )
        )
        buffer_insertion_command = (
            placeio_openroad.PlaceIOFunction.build_buffer_insertion_command(
                strategy,
                options,
            )
        )
        result["buffer_insertion_strategy"] = strategy
        result["buffer_insertion_command"] = buffer_insertion_command
        result["buffer_insertion_strategy_profile"] = strategy_profile["profile"]
        result["buffer_insertion_strategy_kind"] = strategy_profile["kind"]
        result["buffer_insertion_strategy_experimental"] = strategy_profile[
            "experimental"
        ]
        result["buffer_insertion_strategy_validation"] = strategy_profile[
            "validation"
        ]
        result["buffer_insertion_strategy_non_buffer_mutation_free"] = None
        buffer_insertion_payload = {
            "buffer_insertion_strategy": strategy,
            "buffer_insertion_command": buffer_insertion_command,
            "buffer_insertion_strategy_profile": strategy_profile["profile"],
            "buffer_insertion_strategy_kind": strategy_profile["kind"],
            "buffer_insertion_strategy_experimental": strategy_profile[
                "experimental"
            ],
            "buffer_insertion_strategy_validation": strategy_profile["validation"],
            "buffer_insertion_strategy_non_buffer_mutation_free": None,
        }
        try:
            tcl_output = placeio_openroad.PlaceIOFunction.buffer_insertion(
                self.openroad_bridge,
                strategy=strategy,
                options=options,
            )
        except Exception as exc:
            raise OpenRoadHandoffExecutionError(
                "mutation",
                str(exc),
                handoff_payload=buffer_insertion_payload,
            ) from exc
        result["executed"] = True
        result["requires_sync_back"] = True
        result["tcl_output"] = tcl_output
        try:
            sync_contract = self.synchronize_topology_from_openroad(
                params,
                pos=pos,
            )
        except Exception as exc:
            raise OpenRoadHandoffExecutionError(
                "refresh",
                str(exc),
                handoff_payload=buffer_insertion_payload,
            ) from exc
        result["sync_contract"] = sync_contract
        result["mutation_kind"] = sync_contract.get("mutation_kind")
        result["requires_runtimedb_rebuild"] = sync_contract.get(
            "requires_runtimedb_rebuild", False
        )

        def_output = buffer_cfg.get("write_def_after")
        if def_output:
            try:
                self.openroad_bridge.write_def(def_output)
            except Exception as exc:
                raise OpenRoadHandoffExecutionError(
                    "refresh",
                    str(exc),
                    handoff_payload=buffer_insertion_payload,
                ) from exc
            result["write_def_after"] = def_output

        logging.info(
            "OpenROAD buffer insertion executed: strategy=%s requires_sync_back=%s",
            strategy,
            result["requires_sync_back"],
        )
        return result

    def init_db(self, params):
        profile_openroad_init = getattr(params, "place_io_engine", "ieda") == "openroad"
        with _profile_openroad_init_stage(profile_openroad_init, "setup_rawdb"):
            self.setup_rawdb(params)
        with _profile_openroad_init_stage(profile_openroad_init, "capture_topology_refresh_param_baseline"):
            self._capture_topology_refresh_param_baseline(params)
        with _profile_openroad_init_stage(profile_openroad_init, "initialize_from_rawdb"):
            self.initialize_from_rawdb(self.pydb, params)
        # if params.with_sta:
        # self.virtual_net_init()
        with _profile_openroad_init_stage(profile_openroad_init, "initialize"):
            self.initialize(params)
        self.params = params
        self.build_modularity_topology_clusters(params)
        with _profile_openroad_init_stage(profile_openroad_init, "net_degree_summary"):
            net_degrees = np.array([len(pins) for pins in self.net2pin_map])
            print("net_degrees max{} min{}", max(net_degrees), min(net_degrees))

    @staticmethod
    def _require_numeric_matrix(values, field_name, width=None, min_width=None):
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError(f"{field_name} must be a 2D matrix, got shape {array.shape}")
        if width is not None and array.shape[1] != width:
            raise ValueError(
                f"{field_name} must have width {width}, got {array.shape[1]}"
            )
        if min_width is not None and array.shape[1] < min_width:
            raise ValueError(
                f"{field_name} must have width at least {min_width}, got {array.shape[1]}"
            )
        return array

    @staticmethod
    def _ensure_non_decreasing(values, field_name):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{field_name} must be 1D, got shape {values.shape}")
        if np.any(values[1:] < values[:-1]):
            raise ValueError(f"{field_name} must be non-decreasing")

    def _normalize_flat_libcell_info(self, flat_libcell_info):
        rows = [list(row) for row in flat_libcell_info]
        if not rows:
            return np.empty((0,), dtype=np.bytes_), np.empty((0, 4), dtype=self.dtype)
        widths = [len(row) for row in rows]
        if any(width != 4 for width in widths):
            raise ValueError(f"flat_libcell_info must have width 4, got widths {widths}")

        flat_libcell_names = np.asarray([str(row[0]).encode() for row in rows], dtype=np.bytes_)
        numeric = np.empty((len(rows), 4), dtype=self.dtype)
        try:
            numeric[:, 0] = np.arange(len(rows), dtype=self.dtype)
            numeric[:, 1] = np.asarray([int(row[1]) for row in rows], dtype=np.int32)
            numeric[:, 2] = np.asarray([self.dtype(row[2]) for row in rows], dtype=self.dtype)
            vt_tokens = [row[3] for row in rows]
            try:
                numeric[:, 3] = np.asarray(vt_tokens, dtype=np.int32)
            except (TypeError, ValueError):
                vt_labels = [str(token) for token in vt_tokens]
                vt_to_index = {
                    vt_label: idx for idx, vt_label in enumerate(sorted(set(vt_labels)))
                }
                numeric[:, 3] = np.asarray(
                    [vt_to_index[vt_label] for vt_label in vt_labels], dtype=np.int32
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "flat_libcell_info must be numerically convertible as "
                "[cell_name_id, main_id, timing_coordinate, vt]"
            ) from exc
        return flat_libcell_names, numeric

    def _normalize_flat_libcell_leakage(self, flat_libcell_leakage, expected_num_cells):
        values = np.asarray(flat_libcell_leakage, dtype=self.dtype)
        if values.ndim != 1:
            raise ValueError(
                f"flat_libcell_leakage must be 1D, got shape {values.shape}"
            )
        if expected_num_cells is not None and len(values) != int(expected_num_cells):
            raise ValueError(
                "flat_libcell_leakage length must match number of lib cells, "
                f"got {len(values)} vs {expected_num_cells}"
            )
        return values

    def _normalize_flat_libcell_geometry(self, values, name, expected_num_cells):
        values = np.asarray(values, dtype=self.dtype)
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {values.shape}")
        if expected_num_cells is not None and len(values) != int(expected_num_cells):
            raise ValueError(
                f"{name} length must match number of lib cells, got {len(values)} vs {expected_num_cells}"
            )
        return values

    def _log_leakage_debug_summary(self):
        values = getattr(self, "flat_libcell_leakage", None)
        if values is None:
            logging.info("LeakageDebug: flat_libcell_leakage is missing")
            return
        values = np.asarray(values, dtype=self.dtype)
        if values.size == 0:
            logging.info("LeakageDebug: flat_libcell_leakage is empty")
            return
        nonzero_mask = values > 0
        nonzero_count = int(np.count_nonzero(nonzero_mask))
        logging.info(
            "LeakageDebug: flat_libcell_leakage stats count=%d nonzero=%d min=%.6e max=%.6e sum=%.6e",
            int(values.size),
            nonzero_count,
            float(values.min()),
            float(values.max()),
            float(values.sum()),
        )
        if nonzero_count > 0 and getattr(self, "flat_libcell_names", None) is not None:
            sample_indices = np.flatnonzero(nonzero_mask)[:5]
            sample_entries = []
            for idx in sample_indices:
                raw_name = self.flat_libcell_names[int(idx)]
                if isinstance(raw_name, np.ndarray):
                    raw_name = raw_name[0]
                if isinstance(raw_name, bytes):
                    cell_name = raw_name.decode(errors="ignore")
                else:
                    cell_name = str(raw_name)
                sample_entries.append(f"{idx}:{cell_name}={float(values[int(idx)]):.6e}")
            logging.info(
                "LeakageDebug: flat_libcell_leakage sample_nonzero=%s",
                "; ".join(sample_entries),
            )

        inst_values = getattr(self, "inst_leakage_init", None)
        if inst_values is not None:
            inst_values = np.asarray(inst_values, dtype=self.dtype)
            inst_nonzero_count = int(np.count_nonzero(inst_values > 0))
            logging.info(
                "LeakageDebug: inst_leakage_init stats count=%d nonzero=%d min=%.6e max=%.6e sum=%.6e",
                int(inst_values.size),
                inst_nonzero_count,
                float(inst_values.min()) if inst_values.size else 0.0,
                float(inst_values.max()) if inst_values.size else 0.0,
                float(inst_values.sum()) if inst_values.size else 0.0,
            )

    def _as_int_array_1d(self, value):
        if value is None:
            return np.array([], dtype=np.int64)
        arr = np.asarray(value, dtype=np.int64)
        return arr.reshape(-1)

    def _as_int_array_2d(self, value, width=2):
        if value is None:
            return np.empty((0, width), dtype=np.int64)
        arr = np.asarray(value, dtype=np.int64)
        if arr.size == 0:
            return np.empty((0, width), dtype=np.int64)
        if arr.ndim == 1:
            if width <= 0 or arr.size % width != 0:
                return np.empty((0, width), dtype=np.int64)
            return arr.reshape((-1, width))
        if arr.ndim == 2 and arr.shape[1] >= width:
            return arr[:, :width]
        return np.empty((0, width), dtype=np.int64)

    def _pin_name_for_summary(self, pin_id):
        pin_names = getattr(self, "pin_names", None)
        try:
            pin_id = int(pin_id)
        except (TypeError, ValueError):
            return None
        if pin_names is None or pin_id < 0 or pin_id >= len(pin_names):
            return None
        return self._decode_name(pin_names[pin_id])

    def _net_name_for_summary(self, net_id):
        net_names = getattr(self, "net_names", None)
        try:
            net_id = int(net_id)
        except (TypeError, ValueError):
            return None
        if net_names is None or net_id < 0 or net_id >= len(net_names):
            return None
        return self._decode_name(net_names[net_id])

    def _first_level_fanout_samples(self, sample_limit=12):
        pin2net = self._as_int_array_1d(getattr(self, "pin2net_map", None))
        flat_net2pin = self._as_int_array_1d(getattr(self, "flat_net2pin_map", None))
        flat_net2pin_start = self._as_int_array_1d(
            getattr(self, "flat_net2pin_start_map", None)
        )
        net2driver = self._as_int_array_1d(getattr(self, "net2driver_pin_map", None))
        start_points = self._as_int_array_1d(getattr(self, "start_points", None))
        if (
            pin2net.size == 0
            or flat_net2pin.size == 0
            or flat_net2pin_start.size < 2
            or net2driver.size == 0
            or start_points.size == 0
        ):
            return []

        samples = []
        seen_nets = set()
        for raw_pin_id in start_points:
            pin_id = int(raw_pin_id)
            if pin_id < 0 or pin_id >= pin2net.size:
                continue
            net_id = int(pin2net[pin_id])
            if net_id in seen_nets or net_id < 0 or net_id + 1 >= flat_net2pin_start.size:
                continue
            start = int(flat_net2pin_start[net_id])
            end = int(flat_net2pin_start[net_id + 1])
            if start < 0 or end < start or end > flat_net2pin.size:
                continue
            driver_pin_id = int(net2driver[net_id]) if net_id < net2driver.size else None
            net_pin_ids = [int(pin) for pin in flat_net2pin[start:end]]
            sink_pin_ids = [pin for pin in net_pin_ids if pin != driver_pin_id]
            samples.append(
                {
                    "start_pin_id": pin_id,
                    "start_pin_name": self._pin_name_for_summary(pin_id),
                    "net_id": net_id,
                    "net_name": self._net_name_for_summary(net_id),
                    "driver_pin_id": driver_pin_id,
                    "driver_pin_name": self._pin_name_for_summary(driver_pin_id),
                    "sink_pin_ids": sink_pin_ids,
                    "sink_pin_names": [
                        self._pin_name_for_summary(sink_pin_id)
                        for sink_pin_id in sink_pin_ids
                    ],
                    "fanout": len(sink_pin_ids),
                }
            )
            seen_nets.add(net_id)
            if len(samples) >= sample_limit:
                break
        return samples

    def _probe_pin_backward_traces(self):
        probe_pin_names = [
            "g44249__69005:B2",
            "g44248__63144:B2",
            "g44175__85370:B1",
            "FE_OFC19_u_NV_NVDLA_cmac_u_reg_n_1479:Y",
            "g65266:Y",
        ]
        pin_names = getattr(self, "pin_names", None)
        if pin_names is None:
            pin_names = []
        pin_name_to_id = {
            self._decode_name(name): pin_id
            for pin_id, name in enumerate(pin_names)
        }
        pin2net = self._as_int_array_1d(getattr(self, "pin2net_map", None))
        net2driver = self._as_int_array_1d(getattr(self, "net2driver_pin_map", None))
        pin_pred_start = self._as_int_array_1d(getattr(self, "pin_pred_start", None))
        pin_pred_pin = self._as_int_array_1d(getattr(self, "pin_pred_pin", None))
        pin_pred_arc_id = self._as_int_array_1d(getattr(self, "pin_pred_arc_id", None))
        arc_src_pin = self._as_int_array_1d(getattr(self, "arc_src_pin", None))
        arc_dst_pin = self._as_int_array_1d(getattr(self, "arc_dst_pin", None))

        traces = {}
        for pin_name in probe_pin_names:
            pin_id = pin_name_to_id.get(pin_name)
            if pin_id is None:
                traces[pin_name] = {
                    "pin_id": None,
                    "pin_name": pin_name,
                    "found": False,
                    "predecessors": [],
                }
                continue
            net_id = int(pin2net[pin_id]) if 0 <= pin_id < pin2net.size else None
            driver_pin_id = (
                int(net2driver[net_id])
                if net_id is not None and 0 <= net_id < net2driver.size
                else None
            )
            predecessors = []
            if 0 <= pin_id and pin_id + 1 < pin_pred_start.size:
                start = int(pin_pred_start[pin_id])
                end = int(pin_pred_start[pin_id + 1])
                if 0 <= start <= end <= pin_pred_pin.size:
                    for edge_idx in range(start, end):
                        pred_pin_id = int(pin_pred_pin[edge_idx])
                        arc_id = (
                            int(pin_pred_arc_id[edge_idx])
                            if edge_idx < pin_pred_arc_id.size
                            else -1
                        )
                        edge_type = "net_arc" if arc_id < 0 else "cell_arc"
                        arc_src_pin_id = (
                            int(arc_src_pin[arc_id])
                            if arc_id >= 0 and arc_id < arc_src_pin.size
                            else None
                        )
                        arc_dst_pin_id = (
                            int(arc_dst_pin[arc_id])
                            if arc_id >= 0 and arc_id < arc_dst_pin.size
                            else None
                        )
                        predecessors.append(
                            {
                                "pred_pin_id": pred_pin_id,
                                "pred_pin_name": self._pin_name_for_summary(pred_pin_id),
                                "edge_type": edge_type,
                                "arc_id": arc_id,
                                "arc_src_pin_id": arc_src_pin_id,
                                "arc_src_pin_name": self._pin_name_for_summary(arc_src_pin_id),
                                "arc_dst_pin_id": arc_dst_pin_id,
                                "arc_dst_pin_name": self._pin_name_for_summary(arc_dst_pin_id),
                            }
                        )
            traces[pin_name] = {
                "pin_id": int(pin_id),
                "pin_name": pin_name,
                "found": True,
                "net_id": net_id,
                "net_name": self._net_name_for_summary(net_id),
                "driver_pin_id": driver_pin_id,
                "driver_pin_name": self._pin_name_for_summary(driver_pin_id),
                "predecessors": predecessors,
            }
        return traces

    def _build_first_level_graph_contract_summary(self, sample_limit=12):
        pin_names = getattr(self, "pin_names", None)
        net_names = getattr(self, "net_names", None)
        flat_net2pin = self._as_int_array_1d(getattr(self, "flat_net2pin_map", None))
        flat_net2pin_start = self._as_int_array_1d(
            getattr(self, "flat_net2pin_start_map", None)
        )
        net2driver = self._as_int_array_1d(getattr(self, "net2driver_pin_map", None))
        net_flat_arcs = self._as_int_array_2d(getattr(self, "net_flat_arcs", None), width=2)
        net_flat_arcs_start = self._as_int_array_1d(
            getattr(self, "net_flat_arcs_start", None)
        )
        start_points = self._as_int_array_1d(getattr(self, "start_points", None))
        end_points = self._as_int_array_1d(getattr(self, "end_points", None))
        clock_pins = self._as_int_array_1d(getattr(self, "clock_pins", None))
        flat_inst_arcs_by_level = self._as_int_array_1d(
            getattr(self, "flat_inst_arcs_by_level", None)
        )
        flat_inst_arcs_by_level_start = self._as_int_array_1d(
            getattr(self, "flat_inst_arcs_by_level_start", None)
        )
        pin_pred_start = self._as_int_array_1d(getattr(self, "pin_pred_start", None))
        pin_pred_pin = self._as_int_array_1d(getattr(self, "pin_pred_pin", None))
        pin_succ_start = self._as_int_array_1d(getattr(self, "pin_succ_start", None))
        pin_succ_pin = self._as_int_array_1d(getattr(self, "pin_succ_pin", None))
        pin_pair_arc_keys = self._as_int_array_2d(
            getattr(self, "pin_pair_arc_keys", None), width=2
        )
        pin2net = self._as_int_array_1d(getattr(self, "pin2net_map", None))

        inst_level_counts = []
        if flat_inst_arcs_by_level_start.size >= 2:
            inst_level_counts = [
                int(flat_inst_arcs_by_level_start[idx + 1] - flat_inst_arcs_by_level_start[idx])
                for idx in range(flat_inst_arcs_by_level_start.size - 1)
            ]
        inst_arc_count = int(sum(inst_level_counts))

        checked_nets = 0
        driver_first_failures = 0
        net_arc_source_failures = 0
        failure_samples = []
        num_nets = 0 if net_names is None else int(len(net_names))
        for net_id in range(num_nets):
            if net_id >= net2driver.size or net_id + 1 >= flat_net2pin_start.size:
                continue
            start = int(flat_net2pin_start[net_id])
            end = int(flat_net2pin_start[net_id + 1])
            if start < 0 or end <= start or end > flat_net2pin.size:
                continue
            driver_pin_id = int(net2driver[net_id])
            flat_first_pin = int(flat_net2pin[start])
            checked_nets += 1
            driver_first_ok = flat_first_pin == driver_pin_id
            if not driver_first_ok:
                driver_first_failures += 1

            arc_sources_ok = True
            if net_flat_arcs_start.size >= num_nets + 1:
                arc_start = int(net_flat_arcs_start[net_id])
                arc_end = int(net_flat_arcs_start[net_id + 1])
                if (
                    arc_start < 0
                    or arc_end < arc_start
                    or arc_end > net_flat_arcs.shape[0]
                ):
                    arc_sources_ok = False
                elif arc_end > arc_start:
                    arc_sources_ok = bool(np.all(net_flat_arcs[arc_start:arc_end, 0] == driver_pin_id))
            elif net_flat_arcs.shape[0] > 0:
                sink_pin_ids = set(
                    int(pin)
                    for pin in flat_net2pin[start:end]
                    if int(pin) != driver_pin_id
                )
                local_arcs = [
                    row
                    for row in net_flat_arcs
                    if int(row[1]) in sink_pin_ids or int(row[0]) == driver_pin_id
                ]
                if local_arcs:
                    arc_sources_ok = all(int(row[0]) == driver_pin_id for row in local_arcs)
            if not arc_sources_ok:
                net_arc_source_failures += 1

            if (not driver_first_ok or not arc_sources_ok) and len(failure_samples) < sample_limit:
                failure_samples.append(
                    {
                        "net_id": net_id,
                        "net_name": self._net_name_for_summary(net_id),
                        "driver_pin_id": driver_pin_id,
                        "driver_pin_name": self._pin_name_for_summary(driver_pin_id),
                        "flat_first_pin_id": flat_first_pin,
                        "flat_first_pin_name": self._pin_name_for_summary(flat_first_pin),
                        "driver_first_ok": driver_first_ok,
                        "net_arc_sources_ok": arc_sources_ok,
                    }
                )

        startpoints_missing_net = 0
        startpoints_without_successors = 0
        startpoint_top_level_count = 0
        startpoint_instance_count = 0
        for raw_pin_id in start_points:
            pin_id = int(raw_pin_id)
            pin_name = self._pin_name_for_summary(pin_id)
            if pin_name is not None and ":" in pin_name:
                startpoint_instance_count += 1
            else:
                startpoint_top_level_count += 1
            if pin_id < 0 or pin_id >= pin2net.size:
                startpoints_missing_net += 1
            if pin_id < 0 or pin_id + 1 >= pin_succ_start.size:
                startpoints_without_successors += 1
            elif int(pin_succ_start[pin_id + 1]) <= int(pin_succ_start[pin_id]):
                startpoints_without_successors += 1

        return {
            "graph_shape": {
                "pins": 0 if pin_names is None else int(len(pin_names)),
                "nets": num_nets,
                "net_arcs": int(net_flat_arcs.shape[0]),
                "inst_arcs_total": inst_arc_count,
                "inst_arcs_flat_scalars": int(flat_inst_arcs_by_level.size),
                "inst_arc_levels": inst_level_counts,
                "start_points": int(start_points.size),
                "end_points": int(end_points.size),
                "clock_pins": int(clock_pins.size),
                "pin_pred_edges": int(pin_pred_pin.size),
                "pin_succ_edges": int(pin_succ_pin.size),
                "pin_pair_arc_key_count": int(pin_pair_arc_keys.shape[0]),
            },
            "net_driver_contract": {
                "checked_nets": int(checked_nets),
                "driver_first_failures": int(driver_first_failures),
                "net_arc_source_failures": int(net_arc_source_failures),
                "sample_failures": failure_samples,
            },
            "startpoint_contract": {
                "num_start_points": int(start_points.size),
                "num_top_level_startpoints": int(startpoint_top_level_count),
                "num_instance_startpoints": int(startpoint_instance_count),
                "startpoints_missing_net": int(startpoints_missing_net),
                "startpoints_without_successors": int(startpoints_without_successors),
            },
            "clk2q_contract": {
                "clock_pins": int(clock_pins.size),
                "level0_inst_arcs": int(inst_level_counts[0]) if inst_level_counts else 0,
            },
            "first_fanout_samples": self._first_level_fanout_samples(
                sample_limit=sample_limit
            ),
            "probe_pin_backward_traces": self._probe_pin_backward_traces(),
        }

    def _write_openroad_first_level_graph_audit(self, params, audit_payload):
        design_name_attr = getattr(params, "design_name", None)
        if callable(design_name_attr):
            design_name = design_name_attr()
        else:
            design_name = getattr(params, "base_design_name", "design")
        result_dir = getattr(params, "result_dir", "")
        output_path = os.path.join(
            result_dir,
            f"{design_name}_openroad_first_level_graph_audit.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(audit_payload, json_file, indent=2, sort_keys=True)
        logging.info("Wrote OpenROAD-backed first-level graph audit to %s", output_path)
        return output_path

    def _write_first_level_alignment_debug_csv(self, params, audit_payload):
        design_name_attr = getattr(params, "design_name", None)
        if callable(design_name_attr):
            design_name = design_name_attr()
        else:
            design_name = getattr(params, "base_design_name", "design")
        result_dir = getattr(params, "result_dir", "")
        output_path = os.path.join(
            result_dir,
            f"{design_name}_first_level_alignment_debug.csv",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pin_succ_start = self._as_int_array_1d(getattr(self, "pin_succ_start", None))
        pin_succ_pin = self._as_int_array_1d(getattr(self, "pin_succ_pin", None))
        pin_succ_arc_id = self._as_int_array_1d(getattr(self, "pin_succ_arc_id", None))
        arc_dst_pin = self._as_int_array_1d(getattr(self, "arc_dst_pin", None))

        def _first_cell_successors(pin_id):
            try:
                pin_id = int(pin_id)
            except (TypeError, ValueError):
                return []
            if pin_id < 0 or pin_id + 1 >= pin_succ_start.size:
                return []
            start = int(pin_succ_start[pin_id])
            end = int(pin_succ_start[pin_id + 1])
            if start < 0 or end < start or end > pin_succ_pin.size:
                return []
            successors = []
            for edge_idx in range(start, end):
                arc_id = -1
                if edge_idx < pin_succ_arc_id.size:
                    arc_id = int(pin_succ_arc_id[edge_idx])
                if arc_id < 0:
                    continue
                dst_pin_id = None
                if arc_id < arc_dst_pin.size:
                    dst_pin_id = int(arc_dst_pin[arc_id])
                if dst_pin_id is None and edge_idx < pin_succ_pin.size:
                    dst_pin_id = int(pin_succ_pin[edge_idx])
                successors.append((arc_id, dst_pin_id))
            return successors

        fieldnames = [
            "start_pin_id",
            "start_pin_name",
            "net_id",
            "net_name",
            "driver_pin_id",
            "driver_pin_name",
            "sink_pin_id",
            "sink_pin_name",
            "fanout",
            "has_first_level_cell_successor",
            "first_level_cell_successor_pin_ids",
            "first_level_cell_successor_pin_names",
            "first_level_cell_arc_ids",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for sample in audit_payload.get("first_fanout_samples", []):
                sink_ids = sample.get("sink_pin_ids", []) or []
                for sink_pin_id in sink_ids:
                    successors = _first_cell_successors(sink_pin_id)
                    successor_pin_ids = [dst_pin_id for _, dst_pin_id in successors]
                    successor_arc_ids = [arc_id for arc_id, _ in successors]
                    writer.writerow(
                        {
                            "start_pin_id": sample.get("start_pin_id"),
                            "start_pin_name": sample.get("start_pin_name"),
                            "net_id": sample.get("net_id"),
                            "net_name": sample.get("net_name"),
                            "driver_pin_id": sample.get("driver_pin_id"),
                            "driver_pin_name": sample.get("driver_pin_name"),
                            "sink_pin_id": sink_pin_id,
                            "sink_pin_name": self._pin_name_for_summary(sink_pin_id),
                            "fanout": sample.get("fanout"),
                            "has_first_level_cell_successor": str(bool(successors)).lower(),
                            "first_level_cell_successor_pin_ids": ";".join(
                                str(pin_id) for pin_id in successor_pin_ids
                            ),
                            "first_level_cell_successor_pin_names": ";".join(
                                self._pin_name_for_summary(pin_id) or ""
                                for pin_id in successor_pin_ids
                            ),
                            "first_level_cell_arc_ids": ";".join(
                                str(arc_id) for arc_id in successor_arc_ids
                            ),
                        }
                    )
        logging.info("Wrote first-level alignment debug CSV to %s", output_path)
        return output_path

    def _write_openroad_aimp_db_summary(self, params):
        def _len_attr(name):
            value = getattr(self, name, None)
            return 0 if value is None else int(len(value))

        def _count_positive_attr(name):
            value = getattr(self, name, None)
            if value is None:
                return 0
            return int(np.count_nonzero(np.asarray(value, dtype=np.float64) > 0))

        def _levelized_arc_count():
            starts = self._as_int_array_1d(getattr(self, "flat_inst_arcs_by_level_start", None))
            if starts.size < 2:
                return 0
            return int(np.sum(np.diff(starts)))

        def _decode(value):
            return self._decode_name(value)

        def _finite_or_none(value):
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                return None
            return scalar if math.isfinite(scalar) else None

        def _pin_name(pin_id):
            pin_names = getattr(self, "pin_names", None)
            try:
                pin_id = int(pin_id)
                if pin_names is None or pin_id < 0 or pin_id >= len(pin_names):
                    return None
                return _decode(pin_names[pin_id])
            except (TypeError, ValueError, IndexError):
                return None

        endpoint_contract = {
            "num_endpoints": _len_attr("end_points"),
            "num_start_points": _len_attr("start_points"),
            "num_constraint_arcs": _len_attr("endpoints_constraint_arcs"),
            "num_endpoint_r_rat": _len_attr("endpoints_rRAT"),
            "num_endpoint_f_rat": _len_attr("endpoints_fRAT"),
            "endpoint_rat_length_match": (
                _len_attr("end_points")
                == _len_attr("endpoints_rRAT")
                == _len_attr("endpoints_fRAT")
            ),
            "num_top_level_output_endpoints": 0,
            "num_top_level_output_r_rat_sentinel": 0,
            "num_top_level_output_f_rat_sentinel": 0,
            "num_endpoint_r_rat_sentinel": 0,
            "num_endpoint_f_rat_sentinel": 0,
            "sample_top_level_output_endpoints": [],
            "probe_pins": {},
        }
        sentinel_threshold = 5.0e7
        end_points = getattr(self, "end_points", None)
        r_rat = getattr(self, "endpoints_rRAT", None)
        f_rat = getattr(self, "endpoints_fRAT", None)
        pin2net_map = getattr(self, "pin2net_map", None)
        net2driver_pin_map = getattr(self, "net2driver_pin_map", None)
        net_names = getattr(self, "net_names", None)
        if end_points is not None and r_rat is not None and f_rat is not None:
            for endpoint_idx, raw_pin_id in enumerate(end_points):
                try:
                    pin_id = int(raw_pin_id)
                except (TypeError, ValueError):
                    continue
                r_value = _finite_or_none(r_rat[endpoint_idx]) if endpoint_idx < len(r_rat) else None
                f_value = _finite_or_none(f_rat[endpoint_idx]) if endpoint_idx < len(f_rat) else None
                r_is_sentinel = r_value is None or r_value >= sentinel_threshold
                f_is_sentinel = f_value is None or f_value >= sentinel_threshold
                endpoint_contract["num_endpoint_r_rat_sentinel"] += int(r_is_sentinel)
                endpoint_contract["num_endpoint_f_rat_sentinel"] += int(f_is_sentinel)
                name = _pin_name(pin_id)
                if name is None or ":" in name:
                    continue
                endpoint_contract["num_top_level_output_endpoints"] += 1
                endpoint_contract["num_top_level_output_r_rat_sentinel"] += int(r_is_sentinel)
                endpoint_contract["num_top_level_output_f_rat_sentinel"] += int(f_is_sentinel)
                if len(endpoint_contract["sample_top_level_output_endpoints"]) < 8:
                    net_id = None
                    net_name = None
                    driver_pin_id = None
                    driver_pin_name = None
                    if pin2net_map is not None and 0 <= pin_id < len(pin2net_map):
                        net_id = int(pin2net_map[pin_id])
                        if net_names is not None and 0 <= net_id < len(net_names):
                            net_name = _decode(net_names[net_id])
                        if (
                            net2driver_pin_map is not None
                            and 0 <= net_id < len(net2driver_pin_map)
                        ):
                            driver_pin_id = int(net2driver_pin_map[net_id])
                            driver_pin_name = _pin_name(driver_pin_id)
                    endpoint_contract["sample_top_level_output_endpoints"].append(
                        {
                            "endpoint_index": int(endpoint_idx),
                            "pin_id": pin_id,
                            "pin_name": name,
                            "r_rat": r_value,
                            "f_rat": f_value,
                            "net_id": net_id,
                            "net_name": net_name,
                            "driver_pin_id": driver_pin_id,
                            "driver_pin_name": driver_pin_name,
                        }
                    )
        pin_name2id_map = getattr(self, "pin_name2id_map", {}) or {}
        for probe_pin_name in ("cmac_a2csb_resp_pd[33]",):
            probe_pin_id = pin_name2id_map.get(probe_pin_name)
            if probe_pin_id is None:
                probe_pin_id = pin_name2id_map.get(probe_pin_name.encode("utf-8"))
            probe = {"pin_id": None, "endpoint_index": None, "r_rat": None, "f_rat": None}
            try:
                probe_pin_id = None if probe_pin_id is None else int(probe_pin_id)
            except (TypeError, ValueError):
                probe_pin_id = None
            if probe_pin_id is not None:
                probe["pin_id"] = probe_pin_id
                if end_points is not None:
                    for endpoint_idx, raw_pin_id in enumerate(end_points):
                        if int(raw_pin_id) == probe_pin_id:
                            probe["endpoint_index"] = int(endpoint_idx)
                            if r_rat is not None and endpoint_idx < len(r_rat):
                                probe["r_rat"] = _finite_or_none(r_rat[endpoint_idx])
                            if f_rat is not None and endpoint_idx < len(f_rat):
                                probe["f_rat"] = _finite_or_none(f_rat[endpoint_idx])
                            break
            endpoint_contract["probe_pins"][probe_pin_name] = probe

        design_name_attr = getattr(params, "design_name", None)
        if callable(design_name_attr):
            design_name = design_name_attr()
        else:
            design_name = getattr(params, "base_design_name", "design")
        result_dir = getattr(params, "result_dir", "")
        output_path = os.path.join(
            result_dir,
            f"{design_name}_openroad_aimp_db_summary.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        design_inputs = getattr(params, "design_inputs", {}) or {}
        rc_tcl = design_inputs.get("rc_tcl", "")
        parasitics_initialization = design_inputs.get(
            "parasitics_initialization",
            "placement" if rc_tcl else "none",
        )
        first_level_graph_contract = self._build_first_level_graph_contract_summary()
        first_level_audit_path = None
        first_level_alignment_path = None
        if self._openroad_aimp_debug_artifacts_enabled(params):
            first_level_audit_path = self._write_openroad_first_level_graph_audit(
                params,
                first_level_graph_contract,
            )
            first_level_alignment_path = self._write_first_level_alignment_debug_csv(
                params,
                first_level_graph_contract,
            )

        summary = {
            "aimp_db_source": "openroad",
            "timing_data_source": getattr(params, "timing_data_source", "openroad"),
            "openroad_backed_aimp_db": True,
            "design_name": design_name,
            "counts": {
                "instances": int(getattr(self, "num_physical_nodes", 0)),
                "movable_instances": int(getattr(self, "num_movable_nodes", 0)),
                "pins": _len_attr("pin_names"),
                "nets": _len_attr("net_names"),
                "libcells": _len_attr("flat_libcell_info"),
                "libarcs": _len_attr("flat_libarc_info"),
                "libpins": _len_attr("flat_lib_pin_cap"),
                "timing_edges": _levelized_arc_count(),
                "equiv_classes": max(0, _len_attr("main_id_2_cell_id_start") - 1),
                "sizeable_instances": int(
                    np.count_nonzero(getattr(self, "inst_is_sizeable", []))
                ),
            },
            "coverage": {
                "lib_pins_with_cap": _count_positive_attr("flat_lib_pin_cap"),
                "lib_pins_with_rcap": _count_positive_attr("flat_lib_pin_rcap"),
                "lib_pins_with_fcap": _count_positive_attr("flat_lib_pin_fcap"),
                "lib_pins_with_cap_limit": _count_positive_attr(
                    "flat_lib_pin_cap_limit"
                ),
                "lib_pins_with_slew_limit": _count_positive_attr(
                    "flat_lib_pin_slew_limit"
                ),
            },
            "runtime_sizing_arc_schema_gate": getattr(
                self, "runtime_sizing_arc_schema_gate", None
            ),
            "endpoint_contract": endpoint_contract,
            "first_level_graph_contract": first_level_graph_contract,
            "parasitics": {
                "initialization": parasitics_initialization,
                "rc_tcl_configured": bool(design_inputs.get("rc_tcl_configured", bool(rc_tcl))),
                "r_unit": _finite_or_none(getattr(self, "r_unit", None)),
                "c_unit_pf": _finite_or_none(getattr(self, "c_unit", None)),
            },
            "paths": {
                "workspace": getattr(getattr(self, "data_manager", None), "dir_workspace", ""),
                "result_dir": result_dir,
                "def": design_inputs.get("def", ""),
                "sdc": design_inputs.get("sdc", ""),
                "rc_tcl": rc_tcl,
                "first_level_graph_audit": first_level_audit_path,
                "first_level_alignment_debug": first_level_alignment_path,
            },
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2, sort_keys=True)
        logging.info("Wrote OpenROAD-backed AIMP DB summary to %s", output_path)
        return output_path

    def _normalize_flat_libpin_offsets(self, values, name, expected_num_libpins):
        arr = np.asarray(values, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
        if expected_num_libpins is not None and len(arr) != int(expected_num_libpins):
            raise ValueError(
                f"{name} length must match number of lib pins, got {len(arr)} vs {expected_num_libpins}"
            )
        return arr

    def _normalize_flat_libarc_info(self, flat_libarc_info):
        rows = [list(row) for row in flat_libarc_info]
        if not rows:
            return np.empty((0, 2), dtype=np.bytes_), np.empty((0, 6), dtype=np.int32)
        widths = [len(row) for row in rows]
        if any(width < 4 for width in widths):
            raise ValueError(
                f"flat_libarc_info must have width at least 4, got widths {widths}"
            )

        arc_name_rows = []
        numeric = np.zeros((len(rows), 6), dtype=np.int32)
        pin_name_to_id = {}
        try:
            for idx, row in enumerate(rows):
                src_name = str(row[0])
                dst_name = str(row[1])
                if src_name not in pin_name_to_id:
                    pin_name_to_id[src_name] = len(pin_name_to_id)
                if dst_name not in pin_name_to_id:
                    pin_name_to_id[dst_name] = len(pin_name_to_id)
                arc_name_rows.append([src_name.encode(), dst_name.encode()])
                numeric[idx, 0] = pin_name_to_id[src_name]
                numeric[idx, 1] = pin_name_to_id[dst_name]
                numeric[idx, 2] = int(row[2])
                numeric[idx, 3] = int(row[3])
                if len(row) >= 6:
                    numeric[idx, 4] = int(row[4])
                    numeric[idx, 5] = int(row[5])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "flat_libarc_info must be numerically convertible as "
                "[src_pin_name_id, dst_pin_name_id, libcell_id, arc_offset, timing_sense, timing_type]"
            ) from exc
        flat_libarc_names = np.asarray(arc_name_rows, dtype=np.bytes_)
        return flat_libarc_names, numeric

    def _normalize_flat_libcell_limit_info(self, flat_libcell_main_id2size_vt_limit):
        rows = [list(row) for row in flat_libcell_main_id2size_vt_limit]
        if not rows:
            return np.empty((0,), dtype=np.bytes_), np.empty((0, 3), dtype=np.int32)
        widths = [len(row) for row in rows]
        if any(width != 3 for width in widths):
            raise ValueError(
                f"flat_libcell_main_id2size_vt_limit must have width 3, got widths {widths}"
            )

        main_type_names = np.asarray([str(row[0]).encode() for row in rows], dtype=np.bytes_)
        numeric = np.empty((len(rows), 3), dtype=np.int32)
        try:
            numeric[:, 0] = np.arange(len(rows), dtype=np.int32)
            numeric[:, 1] = np.asarray([int(row[1]) for row in rows], dtype=np.int32)
            numeric[:, 2] = np.asarray([int(row[2]) for row in rows], dtype=np.int32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "flat_libcell_main_id2size_vt_limit must be numerically convertible as "
                "[main_id, size_limit, vt_limit]"
            ) from exc
        return main_type_names, numeric

    def _build_inst_sizing_metadata(self):
        self._ensure_non_decreasing(self.main_id_2_cell_id_start, "main_id_2_cell_id_start")
        self._ensure_non_decreasing(self.cell_id_2_arc_id_start, "cell_id_2_arc_id_start")

        if self.flat_libcell_info is None or self.flat_libcell_main_id2size_vt_limit is None:
            raise ValueError("Library cell sizing metadata is missing")

        flat_libcell_info = self._require_numeric_matrix(
            self.flat_libcell_info, "flat_libcell_info", width=4
        )
        limits = self._require_numeric_matrix(
            self.flat_libcell_main_id2size_vt_limit,
            "flat_libcell_main_id2size_vt_limit",
            width=3,
        ).astype(np.int32)
        num_main_ids = len(self.main_id_2_cell_id_start) - 1
        if num_main_ids < 0:
            raise ValueError("main_id_2_cell_id_start must contain at least one boundary")
        if self.main_id_is_sizeable is None:
            main_id_is_sizeable = np.ones(num_main_ids, dtype=np.bool_)
        else:
            main_id_is_sizeable = np.asarray(self.main_id_is_sizeable, dtype=np.bool_)
            if main_id_is_sizeable.ndim != 1:
                raise ValueError(
                    f"main_id_is_sizeable must be 1D, got shape {main_id_is_sizeable.shape}"
                )
            if len(main_id_is_sizeable) != num_main_ids:
                raise ValueError(
                    "main_id_is_sizeable length must match number of main_id families"
                )

        num_insts = len(self.inst_main_id)
        num_vt_types = int(max(1, np.max(flat_libcell_info[:, 3].astype(np.int32)) + 1))
        inst_cell_id = np.full(num_insts, -1, dtype=np.int32)
        inst_size_init = np.ones(num_insts, dtype=self.dtype)
        inst_leakage_init = np.zeros(num_insts, dtype=self.dtype)
        inst_size_lower = np.ones(num_insts, dtype=self.dtype)
        inst_size_upper = np.ones(num_insts, dtype=self.dtype)
        inst_vt_init = np.zeros((num_insts, num_vt_types), dtype=self.dtype)
        inst_vt_mask = np.zeros((num_insts, num_vt_types), dtype=np.bool_)
        inst_is_sizeable = np.zeros(num_insts, dtype=np.bool_)

        # Column 2 is the unified timing coordinate exported by the iEDA bridge.
        # Historical "size" names are kept because the downstream sizing path
        # already consumes this coordinate consistently.
        cell_sizes = flat_libcell_info[:, 2].astype(self.dtype)
        cell_vts = flat_libcell_info[:, 3].astype(np.int32)
        if self.flat_libcell_leakage is None:
            raise ValueError("flat_libcell_leakage is required for sizing metadata")
        cell_leakages = np.asarray(self.flat_libcell_leakage, dtype=self.dtype)
        if len(cell_leakages) != len(flat_libcell_info):
            raise ValueError(
                "flat_libcell_leakage length must match flat_libcell_info rows"
            )
        limit_by_main_id = {int(row[0]): row for row in limits}

        for inst_idx, main_id in enumerate(self.inst_main_id.astype(np.int32)):
            if main_id < 0:
                inst_vt_mask[inst_idx, 0] = True
                inst_vt_init[inst_idx, 0] = 1.0
                continue
            if main_id + 1 >= len(self.main_id_2_cell_id_start):
                raise ValueError(f"inst_main_id[{inst_idx}]={main_id} exceeds main_id_2_cell_id_start")
            family_is_sizeable = bool(main_id_is_sizeable[main_id])
            limit_row = limit_by_main_id.get(int(main_id))
            if limit_row is None:
                raise ValueError(f"Missing size/vt limit for main_id {main_id}")
            size_limit = int(limit_row[1])
            vt_limit = int(limit_row[2])
            if size_limit <= 0 or vt_limit <= 0:
                raise ValueError(f"Invalid size/vt limit for main_id {main_id}: {limit_row.tolist()}")
            cell_begin = int(self.main_id_2_cell_id_start[main_id])
            cell_end = int(self.main_id_2_cell_id_start[main_id + 1])
            if cell_end <= cell_begin:
                raise ValueError(f"main_id {main_id} has no candidate library cells")
            candidate_cell_ids = np.arange(cell_begin, cell_end, dtype=np.int32)
            candidate_vts = cell_vts[candidate_cell_ids]
            candidate_sizes = cell_sizes[candidate_cell_ids]
            inst_vt_mask[inst_idx, np.unique(candidate_vts)] = True
            inst_size_lower[inst_idx] = candidate_sizes.min()
            inst_size_upper[inst_idx] = candidate_sizes.max()
            if np.count_nonzero(np.unique(candidate_sizes)) > size_limit:
                raise ValueError(f"main_id {main_id} has more sizes than declared size limit")
            if np.count_nonzero(inst_vt_mask[inst_idx]) > vt_limit:
                raise ValueError(f"main_id {main_id} has more VT types than declared vt limit")

            offset = int(self.inst_libcell_offset[inst_idx])
            if offset < 0:
                continue
            cell_id = cell_begin + offset
            if cell_id >= cell_end:
                raise ValueError(
                    f"inst_libcell_offset[{inst_idx}]={offset} is outside candidate range for main_id {main_id}"
                )
            inst_cell_id[inst_idx] = cell_id
            inst_size_init[inst_idx] = cell_sizes[cell_id]
            inst_leakage_init[inst_idx] = cell_leakages[cell_id]
            inst_vt_init[inst_idx, cell_vts[cell_id]] = 1.0
            inst_is_sizeable[inst_idx] = family_is_sizeable

        return (
            inst_cell_id,
            inst_size_init,
            inst_leakage_init,
            inst_vt_init,
            inst_size_lower,
            inst_size_upper,
            inst_vt_mask,
            inst_is_sizeable,
        )

    def _write_libcell_semantic_drift_artifacts(self, report_csv_path, summary_json_path):
        self._ensure_non_decreasing(self.main_id_2_cell_id_start, "main_id_2_cell_id_start")
        self._ensure_non_decreasing(self.cell_id_2_arc_id_start, "cell_id_2_arc_id_start")

        flat_libcell_info = self._require_numeric_matrix(
            self.flat_libcell_info, "flat_libcell_info", width=4
        )
        flat_libarc_info = self._require_numeric_matrix(
            self.flat_libarc_info, "flat_libarc_info", width=6
        ).astype(np.int32)

        num_cells = len(flat_libcell_info)
        num_families = len(self.main_id_2_cell_id_start) - 1
        if len(self.flat_libcell_names) != num_cells:
            raise ValueError("flat_libcell_names length must match flat_libcell_info rows")
        if len(self.flat_libarc_names) != len(flat_libarc_info):
            raise ValueError("flat_libarc_names length must match flat_libarc_info rows")
        if len(self.cell_id_2_arc_id_start) != num_cells + 1:
            raise ValueError("cell_id_2_arc_id_start must have num_cells + 1 entries")
        if len(self.main_id_is_sizeable) != num_families:
            raise ValueError("main_id_is_sizeable length must match number of families")

        os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)

        report_rows = []
        arc_offset_checks = []
        logical_arc_checks = []
        sizeable_arc_offset_violations = []
        sizeable_logical_arc_violations = []

        for main_id in range(num_families):
            cell_begin = int(self.main_id_2_cell_id_start[main_id])
            cell_end = int(self.main_id_2_cell_id_start[main_id + 1])
            family_is_sizeable = bool(self.main_id_is_sizeable[main_id])
            representative_cell = (
                _maybe_decode_libcell_name(self.flat_libcell_names[cell_begin])
                if cell_begin < cell_end
                else ""
            )
            main_type = representative_cell
            if (
                self.flat_libcell_main_type_names is not None
                and main_id < len(self.flat_libcell_main_type_names)
            ):
                main_type = _maybe_decode_libcell_name(
                    self.flat_libcell_main_type_names[main_id]
                )

            per_offset_signatures = {}
            per_logical_arc_senses = {}
            per_logical_arc_sense_profiles = {}
            family_rows = []
            for libcell_id in range(cell_begin, cell_end):
                cell_name = _maybe_decode_libcell_name(self.flat_libcell_names[libcell_id])
                arc_begin = int(self.cell_id_2_arc_id_start[libcell_id])
                arc_end = int(self.cell_id_2_arc_id_start[libcell_id + 1])
                cell_logical_arc_senses = {}
                for arc_id in range(arc_begin, arc_end):
                    src_pin = _maybe_decode_libcell_name(self.flat_libarc_names[arc_id][0])
                    dst_pin = _maybe_decode_libcell_name(self.flat_libarc_names[arc_id][1])
                    arc_offset = int(flat_libarc_info[arc_id, 3])
                    timing_sense = int(flat_libarc_info[arc_id, 4])
                    timing_type = int(flat_libarc_info[arc_id, 5])
                    signature = (src_pin, dst_pin, timing_type, timing_sense)
                    logical_arc_key = (src_pin, dst_pin)
                    per_offset_signatures.setdefault(arc_offset, set()).add(signature)
                    per_logical_arc_senses.setdefault(logical_arc_key, set()).add(timing_sense)
                    cell_logical_arc_senses.setdefault(logical_arc_key, []).append(timing_sense)
                    family_rows.append(
                        {
                            "main_id": main_id,
                            "main_type": main_type,
                            "family_is_sizeable": family_is_sizeable,
                            "libcell_id": libcell_id,
                            "cell_name": cell_name,
                            "arc_offset": arc_offset,
                            "src_pin": src_pin,
                            "dst_pin": dst_pin,
                            "timing_type": timing_type,
                            "timing_sense": timing_sense,
                        }
                    )
                for logical_arc_key, timing_senses in cell_logical_arc_senses.items():
                    per_logical_arc_sense_profiles.setdefault(logical_arc_key, set()).add(
                        tuple(sorted(timing_senses))
                    )

            violating_offsets = []
            for arc_offset in sorted(per_offset_signatures):
                signatures = sorted(per_offset_signatures[arc_offset])
                is_single_valued = len(signatures) == 1
                if not is_single_valued:
                    violating_offsets.append(arc_offset)
                arc_offset_checks.append(
                    {
                        "main_id": main_id,
                        "main_type": main_type,
                        "representative_cell": representative_cell,
                        "family_is_sizeable": family_is_sizeable,
                        "arc_offset": arc_offset,
                        "signature_count": len(signatures),
                        "is_single_valued": is_single_valued,
                        "signatures": [
                            {
                                "src_pin": src_pin,
                                "dst_pin": dst_pin,
                                "timing_type": timing_type,
                                "timing_sense": timing_sense,
                            }
                            for src_pin, dst_pin, timing_type, timing_sense in signatures
                        ],
                    }
                )
            if family_is_sizeable and violating_offsets:
                sizeable_arc_offset_violations.append(
                    {
                        "main_id": main_id,
                        "main_type": main_type,
                        "representative_cell": representative_cell,
                        "violating_arc_offsets": violating_offsets,
                    }
                )

            violating_logical_arcs = []
            for logical_arc_key in sorted(per_logical_arc_senses):
                timing_senses = sorted(per_logical_arc_senses[logical_arc_key])
                sense_profiles = sorted(
                    per_logical_arc_sense_profiles.get(logical_arc_key, set())
                )
                is_single_valued = len(timing_senses) == 1
                is_profile_consistent = len(sense_profiles) == 1
                if not is_profile_consistent:
                    violating_logical_arcs.append(
                        {
                            "src_pin": logical_arc_key[0],
                            "dst_pin": logical_arc_key[1],
                            "timing_senses": timing_senses,
                            "timing_sense_profiles": [
                                list(profile) for profile in sense_profiles
                            ],
                        }
                    )
                logical_arc_checks.append(
                    {
                        "main_id": main_id,
                        "main_type": main_type,
                        "representative_cell": representative_cell,
                        "family_is_sizeable": family_is_sizeable,
                        "src_pin": logical_arc_key[0],
                        "dst_pin": logical_arc_key[1],
                        "timing_sense_count": len(timing_senses),
                        "timing_senses": timing_senses,
                        "timing_sense_is_single_valued": is_single_valued,
                        "timing_sense_profile_count": len(sense_profiles),
                        "timing_sense_profiles": [
                            list(profile) for profile in sense_profiles
                        ],
                        "timing_sense_profile_is_consistent": is_profile_consistent,
                    }
                )
            if family_is_sizeable and violating_logical_arcs:
                sizeable_logical_arc_violations.append(
                    {
                        "main_id": main_id,
                        "main_type": main_type,
                        "representative_cell": representative_cell,
                        "violating_logical_arcs": violating_logical_arcs,
                    }
                )

            for row in family_rows:
                offset_signatures = per_offset_signatures.get(row["arc_offset"], set())
                logical_arc_senses = per_logical_arc_senses.get(
                    (row["src_pin"], row["dst_pin"]), set()
                )
                logical_arc_sense_profiles = per_logical_arc_sense_profiles.get(
                    (row["src_pin"], row["dst_pin"]), set()
                )
                row["main_id_arc_offset_is_single_valued"] = len(offset_signatures) == 1
                row["logical_arc_timing_sense_is_single_valued"] = (
                    len(logical_arc_senses) == 1
                )
                row["logical_arc_timing_sense_profile_is_consistent"] = (
                    len(logical_arc_sense_profiles) == 1
                )
                report_rows.append(row)

        with open(report_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "main_id",
                    "main_type",
                    "family_is_sizeable",
                    "libcell_id",
                    "cell_name",
                    "arc_offset",
                    "src_pin",
                    "dst_pin",
                    "timing_type",
                    "timing_sense",
                    "main_id_arc_offset_is_single_valued",
                    "logical_arc_timing_sense_is_single_valued",
                    "logical_arc_timing_sense_profile_is_consistent",
                ],
            )
            writer.writeheader()
            for row in report_rows:
                writer.writerow(row)

        needs_degraded_handling = bool(
            sizeable_arc_offset_violations or sizeable_logical_arc_violations
        )
        summary = {
            "report_csv_path": report_csv_path,
            "summary_json_path": summary_json_path,
            "num_families": num_families,
            "num_sizeable_families": int(np.count_nonzero(self.main_id_is_sizeable)),
            "num_report_rows": len(report_rows),
            "per_main_id_arc_offset_checks": arc_offset_checks,
            "per_logical_arc_timing_sense_checks": logical_arc_checks,
            "sizeable_family_violations": {
                "arc_offset_signature_drift": sizeable_arc_offset_violations,
                "logical_arc_timing_sense_drift": sizeable_logical_arc_violations,
            },
            "needs_degraded_handling": needs_degraded_handling,
            "decision": (
                "sizeable-family semantic drift detected; degraded handling is required"
                if needs_degraded_handling
                else "no sizeable-family semantic drift detected; no degraded handling is needed yet"
            ),
        }
        with open(summary_json_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2, sort_keys=True)

        logging.info(
            "libcell semantic drift export: report=%s summary=%s rows=%d arc_offset_checks=%d logical_arc_checks=%d needs_degraded_handling=%s",
            report_csv_path,
            summary_json_path,
            len(report_rows),
            len(arc_offset_checks),
            len(logical_arc_checks),
            needs_degraded_handling,
        )

    def _validate_modularity_inflation_contract(self, params):
        if not getattr(params, "modularity_inflation_flag", False):
            return
        if not getattr(params, "routability_opt_flag", False):
            raise RuntimeError(
                "modularity_inflation_flag=1 requires routability_opt_flag=1"
            )
        if not getattr(params, "modularity_require_gpugr_flag", 1):
            return
        if not getattr(params, "adjust_gpugr_area_flag", False):
            raise RuntimeError(
                "modularity_inflation_flag=1 requires adjust_gpugr_area_flag=1"
            )
        if getattr(params, "adjust_nctugr_area_flag", False):
            raise RuntimeError(
                "modularity_inflation_flag=1 does not support adjust_nctugr_area_flag=1"
            )
        if getattr(params, "adjust_rudy_area_flag", False):
            raise RuntimeError(
                "modularity_inflation_flag=1 does not support adjust_rudy_area_flag=1"
            )

    def build_modularity_topology_clusters(self, params):
        if not getattr(params, "modularity_inflation_flag", False):
            self.modularity_topology_clustering_result = None
            self.modularity_active_clustering_result = None
            return

        self._validate_modularity_inflation_contract(params)
        from dreamplace.ops.routability.leiden_clustering import (
            build_topology_leiden_clusters,
        )

        self.modularity_topology_clustering_result = build_topology_leiden_clusters(
            self, params
        )
        self.modularity_active_clustering_result = None

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
        placeio_ieda = _load_placeio_ieda()
        ieda_design = placeio_ieda.PlaceIOFunction.make_design(
            self.rawdb if self.rawdb is not None else self.data_manager.dir_workspace
        )
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

    def pad_sequences(self, sequences, dtype, pad_value=0.0):
        """Pad sequences to the same length"""
        if not sequences:
            return np.array([])
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padded_seq = list(seq) + [pad_value] * (max_len - len(seq))
            else:
                padded_seq = list(seq)
            padded.append(padded_seq)
        return np.array(padded, dtype=dtype)

    def initialize_from_rawdb(self, pydb, params):
        """
        @brief initialize data members from raw database 
        @param params parameters 
        """
        # pydb = place_io.PlaceIOFunction.pydb(self.rawdb)
        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.num_terminal_NIs = pydb.num_terminal_NIs
        self.num_place_blockages = int(getattr(pydb, "num_place_blockages", 0))
        self._validate_place_blockage_bookkeeping()
        export_options = getattr(getattr(self, "rawdb", None), "export_options", None)
        if export_options is None:
            export_options = PyDbExportOptions.from_params(params)
        include_m2_pg_rail_blockage = export_options.include_m2_pg_rail_blockage
        self.m2_pg_rail_legalization_blockage_flag = (
            resolve_flag(params, "m2_pg_rail_legalization_blockage_flag")
        )
        self.m2_pa_refine_flag = resolve_flag(params, "m2_pa_refine_flag")
        include_m2_pg_rail_density = (
            (
                resolve_m2_density_weight(params) > 0
                or self.m2_pg_rail_legalization_blockage_flag
            )
            and not include_m2_pg_rail_blockage
        )
        if getattr(params, "place_io_engine", "ieda") == "openroad" and not hasattr(
            pydb, "m2_pg_rail_density_boxes"
        ):
            if include_m2_pg_rail_blockage or self.m2_pg_rail_legalization_blockage_flag or self.m2_pa_refine_flag:
                raise RuntimeError("OpenROAD PyDB does not support native M2 PG rail geometry export")
            if include_m2_pg_rail_density:
                logging.warning(
                    "OpenROAD PyDB has no M2 PG rail geometry export; soft rail density is unavailable"
                )
            include_m2_pg_rail_density = False
        self.m2_pg_rail_density_boxes = self._import_m2_pg_rail_density_boxes(
            pydb,
            include_m2_pg_rail_density=include_m2_pg_rail_density,
        )
        self.m2_pg_rail_boxes = self._import_m2_pg_rail_boxes(
            pydb,
            include_m2_pg_rail_geometry=self.m2_pa_refine_flag,
        )
        self.node_name2id_map = {
            self._decode_name(name): node_id
            for name, node_id in pydb.node_name2id_map.items()
        }
        self.node_names = np.array(pydb.node_names, dtype=np.bytes_)
        # If the placer directly takes a global placement solution,
        # the cell positions may still be floating point numbers.
        # It is not good to use the place_io OP to round the positions.
        # Currently we only support BOOKSHELF format.

        self.node_x = np.array(pydb.node_x, dtype=self.dtype)
        self.node_y = np.array(pydb.node_y, dtype=self.dtype)
        self.node_orient = np.array(pydb.node_orient, dtype=np.bytes_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node_is_hard_macro = np.array(
            pydb.node_is_hard_macro, dtype=np.bool_, copy=True)
        self.macro_writeback_candidate = np.array(
            pydb.macro_writeback_candidate, dtype=np.bool_, copy=True)
        if (self.node_is_hard_macro.shape != (self.num_physical_nodes,)
                or self.macro_writeback_candidate.shape != (self.num_physical_nodes,)):
            raise ValueError("Native macro masks must align with DreamPlace nodes")
        self.node_is_hard_macro.setflags(write=False)
        self.macro_writeback_candidate.setflags(write=False)
        self.node2orig_node_map = np.array(
            pydb.node2orig_node_map, dtype=np.int32)
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.bytes_)
        # BUG all the pin offsets are -1
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.pin_names = np.array(pydb.pin_names, dtype=np.bytes_)
        self.net_name2id_map = pydb.net_name2id_map
        self.pin_name2id_map = getattr(pydb, "pin_name2id_map", {})
        self.net_names = np.array(pydb.net_names, dtype=np.bytes_)
        self.clock_net_names = tuple(
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in getattr(pydb, "clock_net_names", [])
        )
        self.net2pin_map = pydb.net2pin_map
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(
            pydb.flat_net2pin_start_map, dtype=np.int32)
        self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
        self.net_weight_deltas = np.array(
            getattr(pydb, "net_weight_deltas", [0.0] * len(self.net_names)),
            dtype=self.dtype)
        self.net_criticality = np.array(
            getattr(pydb, "net_criticality", [0.0] * len(self.net_names)),
            dtype=self.dtype)
        self.net_criticality_deltas = np.array(
            getattr(pydb, "net_criticality_deltas", [0.0] * len(self.net_names)),
            dtype=self.dtype)
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
        self.total_fixed_node_area = float(pydb.total_fixed_node_area)
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
            if hasattr(pydb, "min_wire_widths") and len(pydb.min_wire_widths):
                self.min_wire_widths = np.array(pydb.min_wire_widths, dtype=self.dtype)
            if hasattr(pydb, "min_wire_spacings") and len(pydb.min_wire_spacings):
                self.min_wire_spacings = np.array(pydb.min_wire_spacings, dtype=self.dtype)
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
            self.unit_horizontal_capacities = np.array(
                [self.unit_horizontal_capacity], dtype=self.dtype)
            self.unit_vertical_capacities = np.array(
                [self.unit_vertical_capacity], dtype=self.dtype)
            self.initial_horizontal_demand_map = np.zeros(
                (self.num_routing_grids_x, self.num_routing_grids_y),
                dtype=self.dtype)
            self.initial_vertical_demand_map = np.zeros(
                (self.num_routing_grids_x, self.num_routing_grids_y),
                dtype=self.dtype)

        # convert node2pin_map to array of array
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(
                self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map, dtype=object)

        # convert net2pin_map to array of array
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map, dtype=object)

        # convert the max_net_weight from params
        # note that infinity may be included so we need a type cast
        self.max_net_weight = np.float64(params.max_net_weight)
        self.dbu = float(pydb.dbu)
        
        if params.with_sta:
            self.start_points = np.array(pydb.start_points, dtype=np.int32)
            self.end_points = np.array(pydb.end_points, dtype=np.int32)
            from dreamplace.ops.placeio_ecc.timing_schema import read_qualification_arrays
            for name, values in read_qualification_arrays(pydb).items():
                setattr(self, name, values)
            self.clock_pins = np.array(pydb.clock_pins, dtype=np.int32)
            self.FF_ids = np.array(pydb.FF_ids, dtype=np.int32)
            self.clk_pin_r_aat = np.array(pydb.clk_pin_r_aat, dtype=self.dtype)
            self.clk_pin_f_aat = np.array(pydb.clk_pin_f_aat, dtype=self.dtype)
            self.clk_pin_rtran = np.array(pydb.clk_pin_rtran, dtype=self.dtype)
            self.clk_pin_ftran = np.array(pydb.clk_pin_ftran, dtype=self.dtype)
            self.clk_pin_names = np.array(
                pydb.clk_pin_names, dtype=np.bytes_)
            self.flat_inst_arcs_by_level = np.array(
                getattr(pydb, "flat_inst_arcs_by_level", []), dtype=np.int32)
            self.flat_inst_arcs_by_level_start = np.array(
                getattr(pydb, "flat_inst_arcs_by_level_start", []), dtype=np.int32)
            self.flat_pin_to_graph = np.array(
                getattr(pydb, "flat_pin_to_graph", []), dtype=np.int32)
            self.flat_pin_to_graph_start = np.array(
                getattr(pydb, "flat_pin_to_graph_start", []), dtype=np.int32)
            self.flat_pin_to_graph_reverse = np.array(
                getattr(pydb, "flat_pin_to_graph_reverse", []), dtype=np.int32)
            self.flat_pin_to_graph_start_reverse = np.array(
                getattr(pydb, "flat_pin_to_graph_start_reverse", []), dtype=np.int32)
            pin_pair_arc_keys = np.array(
                getattr(pydb, "pin_pair_arc_keys", []), dtype=np.int32)
            self.pin_pair_arc_keys = pin_pair_arc_keys.reshape(-1, 2)
            self.flat_pin_pair_arc_start = np.array(
                getattr(pydb, "flat_pin_pair_arc_start", []), dtype=np.int32)
            self.flat_pin_pair_arc_indices = np.array(
                getattr(pydb, "flat_pin_pair_arc_indices", []), dtype=np.int32)

            self.arc_level_start = np.array(
                getattr(pydb, "arc_level_start", []), dtype=np.int32)
            self.arc_src_pin = np.array(
                getattr(pydb, "arc_src_pin", []), dtype=np.int32)
            self.arc_dst_pin = np.array(
                getattr(pydb, "arc_dst_pin", []), dtype=np.int32)
            self.arc_inst_id = np.array(
                getattr(pydb, "arc_inst_id", []), dtype=np.int32)
            self.arc_libcell_id = np.array(
                getattr(pydb, "arc_libcell_id", []), dtype=np.int32)
            self.arc_libarc_id = np.array(
                getattr(pydb, "arc_libarc_id", []), dtype=np.int32)
            self.arc_sense = np.array(
                getattr(pydb, "arc_sense", []), dtype=np.int32)
            self.arc_type = np.array(
                getattr(pydb, "arc_type", []), dtype=np.int32)
            self.arc_offset = np.array(
                getattr(pydb, "arc_offset", []), dtype=np.int32)
            self.pin_pred_start = np.array(
                getattr(pydb, "pin_pred_start", []), dtype=np.int32)
            self.pin_pred_pin = np.array(
                getattr(pydb, "pin_pred_pin", []), dtype=np.int32)
            self.pin_pred_arc_id = np.array(
                getattr(pydb, "pin_pred_arc_id", []), dtype=np.int32)
            self.pin_succ_start = np.array(
                getattr(pydb, "pin_succ_start", []), dtype=np.int32)
            self.pin_succ_pin = np.array(
                getattr(pydb, "pin_succ_pin", []), dtype=np.int32)
            self.pin_succ_arc_id = np.array(
                getattr(pydb, "pin_succ_arc_id", []), dtype=np.int32)
            self.endpoint_pin_ids = np.array(
                getattr(pydb, "endpoint_pin_ids", []), dtype=np.int32)
            self.start_pin_ids = np.array(
                getattr(pydb, "start_pin_ids", []), dtype=np.int32)
            self.pin_to_inst_id = np.array(
                getattr(pydb, "pin_to_inst_id", []), dtype=np.int32)
            self.pin_to_node_id = np.array(
                getattr(pydb, "pin_to_node_id", []), dtype=np.int32)
            self.inst_topo_start = np.array(
                getattr(pydb, "inst_topo_start", []), dtype=np.int32)
            self.inst_topo_ids = np.array(
                getattr(pydb, "inst_topo_ids", []), dtype=np.int32)
            
            # self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
            self.net_weight_deltas = np.zeros_like(pydb.net_weights, dtype=self.dtype)
            self.net_criticality = np.zeros_like(pydb.net_weights, dtype=self.dtype)
            self.net_criticality_deltas = np.zeros_like(pydb.net_weights, dtype=self.dtype)
            
            self.flat_cells_by_level = np.array(
                getattr(pydb, "flat_cells_by_level", []), dtype=np.int32)
            self.flat_cells_by_reverse_level = np.array(
                getattr(pydb, "flat_cells_by_reverse_level", []), dtype=np.int32)
            self.flat_cells_by_level_start = np.array(
                getattr(pydb, "flat_cells_by_level_start", []), dtype=np.int32)
            self.flat_cells_by_reverse_level_start = np.array(
                getattr(pydb, "flat_cells_by_reverse_level_start", []), dtype=np.int32)
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
            self.endpoints_rRAT = np.array(pydb.endpoints_rRAT, dtype=self.dtype)
            self.endpoints_fRAT = np.array(pydb.endpoints_fRAT, dtype=self.dtype)
            self.backend_endpoint_rAAT = np.array(
                getattr(pydb, "backend_endpoint_rAAT", []), dtype=self.dtype)
            self.backend_endpoint_fAAT = np.array(
                getattr(pydb, "backend_endpoint_fAAT", []), dtype=self.dtype)
            self.backend_endpoint_rRAT = np.array(
                getattr(pydb, "backend_endpoint_rRAT", []), dtype=self.dtype)
            self.backend_endpoint_fRAT = np.array(
                getattr(pydb, "backend_endpoint_fRAT", []), dtype=self.dtype)
            self.backend_endpoint_rSlew = np.array(
                getattr(pydb, "backend_endpoint_rSlew", []), dtype=self.dtype)
            self.backend_endpoint_fSlew = np.array(
                getattr(pydb, "backend_endpoint_fSlew", []), dtype=self.dtype)
            self.backend_endpoint_min_rAAT = np.array(
                getattr(pydb, "backend_endpoint_min_rAAT", []), dtype=self.dtype)
            self.backend_endpoint_min_fAAT = np.array(
                getattr(pydb, "backend_endpoint_min_fAAT", []), dtype=self.dtype)
            self.backend_endpoint_min_rRAT = np.array(
                getattr(pydb, "backend_endpoint_min_rRAT", []), dtype=self.dtype)
            self.backend_endpoint_min_fRAT = np.array(
                getattr(pydb, "backend_endpoint_min_fRAT", []), dtype=self.dtype)

            self.net_flat_arcs_start = np.array(
                pydb.net_flat_arcs_start, dtype=np.int32)
            self.net_flat_arcs = np.array(pydb.net_flat_arcs, dtype=np.int32)
            self.inst_flat_arcs_start = np.array(
                pydb.inst_flat_arcs_start, dtype=np.int32)
            self.inst_flat_arcs = np.array(pydb.inst_flat_arcs, dtype=np.int32)
            self.endpoints_constraint_arcs = np.array(pydb.endpoints_constraint_arcs, dtype=np.int32)
            self.endpoints_timing_check_arcs = np.array(
                getattr(pydb, "endpoints_timing_check_arcs", []), dtype=np.int32)

            if self._supports_diff_optimizer_metadata():
                self._initialize_diff_optimizer_metadata_from_pydb(pydb, params)
            else:
                logging.info(
                    "Skipping diff optimizer metadata initialization for backend=%s; backend_caps.has_diff_sizing_metadata=False",
                    dict(getattr(self, "backend_caps", {}) or {}).get("backend", "unknown"),
                )

    def _supports_diff_optimizer_metadata(self):
        caps = dict(getattr(self, "backend_caps", {}) or {})
        return bool(caps.get("has_diff_sizing_metadata", caps.get("has_diff_optimizer_metadata", True)))

    def _initialize_diff_optimizer_metadata_from_pydb(self, pydb, params):
        self.main_id_2_cell_id_start = np.array(
            pydb.main_id_2_cell_id_start, dtype=np.int32)
        self.cell_id_2_arc_id_start = np.array(
            pydb.cell_id_2_arc_id_start, dtype=np.int32)
        self.buffer_main_type_index = int(getattr(pydb, "buffer_main_type_index", -1))
        self.buffer_main_type_status = str(getattr(pydb, "buffer_main_type_status", "unsupported"))
        self.buffer_main_type_candidate_indices = np.array(
            getattr(pydb, "buffer_main_type_candidate_indices", []), dtype=np.int32
        )

        self.inst_main_id = np.array(pydb.inst_main_id, dtype=np.int32)
        self.inst_libcell_offset = np.array(pydb.inst_libcell_offset, dtype=np.int32)

        self.flat_libarc_names, self.flat_libarc_info = self._normalize_flat_libarc_info(
            pydb.flat_libarc_info
        )
        self.flat_libcell_names, self.flat_libcell_info = self._normalize_flat_libcell_info(
            pydb.flat_libcell_info
        )
        if hasattr(pydb, "flat_libcell_width") and hasattr(pydb, "flat_libcell_height"):
            self.flat_libcell_width = self._normalize_flat_libcell_geometry(
                pydb.flat_libcell_width,
                "flat_libcell_width",
                expected_num_cells=len(self.flat_libcell_info),
            )
            self.flat_libcell_height = self._normalize_flat_libcell_geometry(
                pydb.flat_libcell_height,
                "flat_libcell_height",
                expected_num_cells=len(self.flat_libcell_info),
            )
        else:
            self.flat_libcell_width = None
            self.flat_libcell_height = None
            logging.warning(
                "pydb.flat_libcell_width/height are unavailable; projected runtime geometry refresh will be disabled"
            )
        if self._openroad_aimp_debug_artifacts_enabled(params):
            logging.info(
                "libcell_id export trigger: result_dir=%s design=%s num_cells=%d",
                params.result_dir,
                params.design_name(),
                len(self.flat_libcell_info),
            )
            _write_libcell_id_artifact(
                os.path.join(params.result_dir, f"{params.design_name()}_libcell_id.csv"),
                self.flat_libcell_names,
                self.flat_libcell_info,
            )
        if not hasattr(pydb, "flat_libcell_leakage"):
            raise ValueError("pydb.flat_libcell_leakage is required for gate sizing metadata")
        self.flat_libcell_leakage = self._normalize_flat_libcell_leakage(
            pydb.flat_libcell_leakage,
            expected_num_cells=len(self.flat_libcell_info),
        )
        (
            self.flat_libcell_main_type_names,
            self.flat_libcell_main_id2size_vt_limit,
        ) = self._normalize_flat_libcell_limit_info(
            pydb.flat_libcell_main_id2size_vt_limit
        )
        if hasattr(pydb, "main_id_is_sizeable") and len(pydb.main_id_is_sizeable) > 0:
            self.main_id_is_sizeable = np.array(pydb.main_id_is_sizeable, dtype=np.bool_)
        else:
            self.main_id_is_sizeable = np.ones(
                len(self.main_id_2_cell_id_start) - 1, dtype=np.bool_
            )
        if self._openroad_aimp_debug_artifacts_enabled(params):
            self._write_libcell_semantic_drift_artifacts(
                os.path.join(
                    params.result_dir,
                    f"{params.design_name()}_libcell_semantic_drift_report.csv",
                ),
                os.path.join(
                    params.result_dir,
                    f"{params.design_name()}_libcell_semantic_drift_summary.json",
                ),
            )
        (
            self.main_id_is_sizeable,
            self.runtime_sizing_arc_schema_gate,
        ) = _gate_sizeable_families_by_runtime_arc_schema(
            self.main_id_2_cell_id_start,
            self.cell_id_2_arc_id_start,
            self.flat_libarc_info,
            self.main_id_is_sizeable,
            self.flat_libcell_names,
        )
        if self.runtime_sizing_arc_schema_gate["disabled_family_count"]:
            logging.warning(
                "disabled %d frozen-topology sizing families with incompatible "
                "Liberty arc schemas: %s",
                self.runtime_sizing_arc_schema_gate["disabled_family_count"],
                ", ".join(
                    str(row["main_id"])
                    for row in self.runtime_sizing_arc_schema_gate["disabled_families"]
                ),
            )
        (
            self.inst_cell_id,
            self.inst_size_init,
            self.inst_leakage_init,
            self.inst_vt_init,
            self.inst_size_lower,
            self.inst_size_upper,
            self.inst_vt_mask,
            self.inst_is_sizeable,
        ) = self._build_inst_sizing_metadata()
        self._log_leakage_debug_summary()

        # LUTs table
        self.f_delay_flat_luts_values = self.pad_sequences(
            pydb.f_delay_flat_luts_values, dtype=self.dtype)
        self.f_delay_flat_luts_trans_table = self.pad_sequences(
            pydb.f_delay_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.f_delay_flat_luts_cap_table = self.pad_sequences(
            pydb.f_delay_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.f_delay_flat_luts_dim = np.array(
            pydb.f_delay_flat_luts_dim, dtype=np.int32)

        self.r_delay_flat_luts_values = self.pad_sequences(
            pydb.r_delay_flat_luts_values, dtype=self.dtype)
        self.r_delay_flat_luts_trans_table = self.pad_sequences(
            pydb.r_delay_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.r_delay_flat_luts_cap_table = self.pad_sequences(
            pydb.r_delay_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.r_delay_flat_luts_dim = np.array(
            pydb.r_delay_flat_luts_dim, dtype=np.int32)

        self.f_trans_flat_luts_values = self.pad_sequences(
            pydb.f_trans_flat_luts_values, dtype=self.dtype)
        self.f_trans_flat_luts_trans_table = self.pad_sequences(
            pydb.f_trans_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.f_trans_flat_luts_cap_table = self.pad_sequences(
            pydb.f_trans_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.f_trans_flat_luts_dim = np.array(
            pydb.f_trans_flat_luts_dim, dtype=np.int32)

        self.r_trans_flat_luts_values = self.pad_sequences(
            pydb.r_trans_flat_luts_values, dtype=self.dtype)
        self.r_trans_flat_luts_trans_table = self.pad_sequences(
            pydb.r_trans_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.r_trans_flat_luts_cap_table = self.pad_sequences(
            pydb.r_trans_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.r_trans_flat_luts_dim = np.array(
            pydb.r_trans_flat_luts_dim, dtype=np.int32)

        self.f_check_flat_luts_values = self.pad_sequences(
            pydb.f_check_flat_luts_values, dtype=self.dtype)
        self.f_check_flat_luts_trans_table = self.pad_sequences(
            pydb.f_check_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.f_check_flat_luts_cap_table = self.pad_sequences(
            pydb.f_check_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.f_check_flat_luts_dim = np.array(
            pydb.f_check_flat_luts_dim, dtype=np.int32)

        self.r_check_flat_luts_values = self.pad_sequences(
            pydb.r_check_flat_luts_values, dtype=self.dtype)
        self.r_check_flat_luts_trans_table = self.pad_sequences(
            pydb.r_check_flat_luts_trans_table, dtype=self.dtype, pad_value=np.inf)
        self.r_check_flat_luts_cap_table = self.pad_sequences(
            pydb.r_check_flat_luts_cap_table, dtype=self.dtype, pad_value=np.inf)
        self.r_check_flat_luts_dim = np.array(
            pydb.r_check_flat_luts_dim, dtype=np.int32)

        self.cell_id_2_libpin_id_start = np.array(
            pydb.cell_id_2_libpin_id_start, dtype=np.int32)
        self.pin_2_libpin_offset = np.array(
            pydb.pin_2_libpin_offset, dtype=np.int32)
        if not hasattr(pydb, "flat_lib_pin_offset_x") or not hasattr(pydb, "flat_lib_pin_offset_y"):
            raise ValueError("pydb.flat_lib_pin_offset_x/y are required for size-aware lib pin offsets")
        expected_num_libpins = len(self.cell_id_2_libpin_id_start) and int(self.cell_id_2_libpin_id_start[-1])
        self.flat_lib_pin_offset_x = self._normalize_flat_libpin_offsets(
            pydb.flat_lib_pin_offset_x,
            "flat_lib_pin_offset_x",
            expected_num_libpins=expected_num_libpins,
        )
        self.flat_lib_pin_offset_y = self._normalize_flat_libpin_offsets(
            pydb.flat_lib_pin_offset_y,
            "flat_lib_pin_offset_y",
            expected_num_libpins=expected_num_libpins,
        )
        self.flat_lib_pin_cap = np.array(
            pydb.flat_lib_pin_cap, dtype=self.dtype)
        self.flat_lib_pin_rcap = np.array(
            pydb.flat_lib_pin_rcap, dtype=self.dtype)
        self.flat_lib_pin_fcap = np.array(
            pydb.flat_lib_pin_fcap, dtype=self.dtype)
        self.flat_lib_pin_cap_limit = np.array(
            pydb.flat_lib_pin_cap_limit, dtype=self.dtype)
        self.flat_lib_pin_slew_limit = np.array(
            pydb.flat_lib_pin_slew_limit, dtype=self.dtype)

        # RC
        self.r_unit = float(pydb.r_unit)
        self.c_unit = float(pydb.c_unit)
        if getattr(params, "place_io_engine", "ieda") == "openroad":
            self._write_openroad_aimp_db_summary(params)

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
        params.cell_padding_x *= params.scale_factor
        self.cell_padding_x *= params.scale_factor
        # self.cell_padding_y *= params.scale_factor
        self.bndry_padding_x *= params.scale_factor
        self.bndry_padding_y *= params.scale_factor

        # Apply cell padding only after all movable geometry inflation has
        # completed.  In particular, pin-density inflation runs before this
        # point; checking the area budget earlier could accept padding that
        # later makes the placement too large for the placeable core.
        self._apply_cell_padding(params)

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

        self.total_movable_node_area = float(np.sum(
            self.node_size_x[:self.num_movable_nodes] * self.node_size_y[:self.num_movable_nodes]))
        target_density = min(self.total_movable_node_area / self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warn(
                "target_density %g is smaller than utilization %g, ignored"
                % (params.target_density, target_density)
            )
            params.target_density = target_density

        # set number of bins
        if _is_enabled_param(getattr(params, "enhanced_auto_adjust_bins", 0)):
            preset_num_bins_x = int(params.num_bins_x)
            preset_num_bins_y = int(params.num_bins_y)
            num_bins_x, num_bins_y = _compute_enhanced_auto_adjust_bins(
                preset_num_bins_x,
                preset_num_bins_y,
                self.yh - self.yl,
                self.row_height,
            )
            if (num_bins_x, num_bins_y) != (preset_num_bins_x, preset_num_bins_y):
                logging.warning(
                    "enhanced_auto_adjust_bins caps preset num_bins %dx%d to %dx%d by row count"
                    % (preset_num_bins_x, preset_num_bins_y, num_bins_x, num_bins_y)
                )
            params.num_bins_x = num_bins_x
            params.num_bins_y = num_bins_y
        elif _is_enabled_param(getattr(params, "auto_adjust_bins", 0)):
            num_bins_x, num_bins_y = compute_auto_bin_counts(
                self.total_movable_node_area,
                self.num_movable_nodes,
                params.target_density,
                self.xh - self.xl,
                self.yh - self.yl,
            )
            logging.info(
                "auto placement bins from padded movable area %g, movable nodes %d, "
                "target density %g: %dx%d",
                self.total_movable_node_area,
                self.num_movable_nodes,
                params.target_density,
                num_bins_x,
                num_bins_y,
            )
            params.num_bins_x = num_bins_x
            params.num_bins_y = num_bins_y

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
        # Fixed geometry is unioned by ecc-tools before it reaches Python.  Do
        # not sum terminal rectangles here: overlapping bodies and synthetic
        # obstacles would otherwise be counted more than once and overwrite
        # the native placeable-area calculation.
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

        utilization = self.total_movable_node_area / self.total_space_area
        target_density = minimum_target_density_for_coverage(utilization)
        if target_density > params.target_density:
            logging.warn(
                "target_density %g is smaller than %g required for %g movable coverage, ignored"
                % (params.target_density, target_density, TARGET_MOVABLE_COVERAGE)
            )
            params.target_density = target_density
        utilization = self.total_movable_node_area / self.total_space_area
        content += "utilization = %g, target_density = %g\n" % (
            self.total_movable_node_area / self.total_space_area,
            params.target_density,
        )

        if utilization > MAX_MOVABLE_UTILIZATION:
            raise RuntimeError(
                "utilization is larger than %g. Please change the core size."
                % MAX_MOVABLE_UTILIZATION
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
            # the way to compute this is still tricky; preserve the ECC DB contract here.
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
                    math.floor(self.total_filler_node_area /
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
        # v is expected in the current scaled placer coordinate system.
        # site_width/row_height have already gone through scale(), so snapping
        # here preserves site/row alignment after normalization.
        ops = {"close": np.round, "up": np.ceil, "down": np.floor}
        op = ops[mode]
        if axis == "x":
            return self.site_width * op(v / self.site_width)
        elif axis == "y":
            return self.row_height * op(v / self.row_height)

    def _apply_cell_padding(self, params):
        requested_padding = float(params.cell_padding_x)
        padding = requested_padding
        self.cell_padding_x = padding
        if padding <= 0:
            return

        movable_slice = slice(0, self.num_movable_nodes)
        movable_size_x = self.node_size_x[movable_slice]
        movable_size_y = self.node_size_y[movable_slice]
        movable_area = float(np.sum(movable_size_x * movable_size_y))
        padded_movable_area = float(
            np.sum((movable_size_x + 2 * padding) * movable_size_y)
        )
        placeable_area = getattr(self, "total_space_area", None)
        if placeable_area is not None and np.isfinite(placeable_area):
            max_movable_area = MAX_MOVABLE_UTILIZATION * float(placeable_area)
            if padded_movable_area > max_movable_area:
                total_movable_height = float(np.sum(movable_size_y))
                if total_movable_height > 0:
                    max_padding = max(
                        (max_movable_area - movable_area)
                        / (2 * total_movable_height),
                        0.0,
                    )
                    padding = min(
                        padding,
                        self.crop_to_site(max_padding, "x", mode="down"),
                    )
                else:
                    padding = 0.0

                logging.warning(
                    "cell_padding_x %g would increase movable area to %g, "
                    "above the %g * placeable-area limit (%g); reducing it to %g",
                    requested_padding,
                    padded_movable_area,
                    MAX_MOVABLE_UTILIZATION,
                    max_movable_area,
                    padding,
                )
                params.cell_padding_x = padding
                self.cell_padding_x = padding

        if padding == 0:
            logging.info(
                "cell padding geometry: requested=%.6g effective=0 "
                "movable_area_before=%.6g movable_area_after=%.6g "
                "placeable_area=%.6g",
                requested_padding,
                movable_area,
                movable_area,
                float(placeable_area) if placeable_area is not None else float("nan"),
            )
            return

        self.node_size_x[movable_slice] += 2 * padding
        self.node_x[movable_slice] -= padding
        movable_cell_tensor = np.arange(
            0, self.num_movable_nodes, dtype=self.pin2node_map.dtype
        )
        movable_cell_pins = np.isin(self.pin2node_map, movable_cell_tensor)
        self.pin_offset_x[movable_cell_pins] += padding
        logging.info(
            "cell padding geometry: requested=%.6g effective=%.6g "
            "movable_area_before=%.6g movable_area_after=%.6g "
            "placeable_area=%.6g",
            requested_padding,
            padding,
            movable_area,
            float(np.sum(
                self.node_size_x[movable_slice] * self.node_size_y[movable_slice]
            )),
            float(placeable_area) if placeable_area is not None else float("nan"),
        )

    def _dump_fixed_macro_mask_debug(
        self,
        params,
        node_areas,
        movable_mean_area,
        movable_min_height,
        area_threshold_value,
        height_threshold_value,
        area_threshold,
        height_threshold,
    ):
        def node_name(node_id):
            if self.node_names is None:
                return str(node_id)
            name = self.node_names[node_id]
            if isinstance(name, bytes):
                return name.decode("utf-8", errors="replace")
            return str(name)

        area_den = max(float(movable_mean_area), 1.0e-30)
        height_den = max(float(movable_min_height), 1.0e-30)
        fixed_start = int(self.fixed_slice.start)
        fixed_stop = int(self.fixed_slice.stop)
        selected_rel = np.where(self.fixed_macro_mask)[0]
        selected_count = int(selected_rel.shape[0])
        fixed_count = int(max(fixed_stop - fixed_start, 0))

        placement_blockage_mask = self._fixed_placement_blockage_mask()
        excluded_place_blockages = int(
            getattr(self, "num_fixed_macro_excluded_place_blockages", 0)
        )

        logging.info(
            "Fixed macro heuristic mask: fixed_terminals=%d selected=%d "
            "placement_blockages=%d excluded_place_blockages=%d "
            "movable_mean_area=%.6g area_threshold_scale=%.6g "
            "area_threshold=%.6g movable_min_height=%.6g "
            "height_threshold_scale=%.6g height_threshold=%.6g",
            fixed_count,
            selected_count,
            int(np.count_nonzero(placement_blockage_mask)),
            excluded_place_blockages,
            float(movable_mean_area),
            float(area_threshold),
            float(area_threshold_value),
            float(movable_min_height),
            float(height_threshold),
            float(height_threshold_value),
        )

        if selected_count > 0:
            preview = []
            for rel_idx in selected_rel[:20]:
                node_id = fixed_start + int(rel_idx)
                area_ratio = float(node_areas[node_id]) / area_den
                height_ratio = float(self.node_size_y[node_id]) / height_den
                preview.append(
                    "%d:%s area_ratio=%.3g height_ratio=%.3g "
                    "xy=(%.6g,%.6g) size=(%.6g,%.6g)"
                    % (
                        node_id,
                        node_name(node_id),
                        area_ratio,
                        height_ratio,
                        float(self.node_x[node_id]),
                        float(self.node_y[node_id]),
                        float(self.node_size_x[node_id]),
                        float(self.node_size_y[node_id]),
                    )
                )
            logging.info(
                "Fixed macro heuristic mask selected preview first %d/%d: %s",
                len(preview),
                selected_count,
                "; ".join(preview),
            )

        result_dir = getattr(params, "result_dir", None)
        if not result_dir:
            return

        out_dir = os.path.join(result_dir, "debug")
        out_path = os.path.join(out_dir, "fixed_macro_mask_debug.csv")
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "node_id",
                        "fixed_rel_id",
                        "node_name",
                        "x",
                        "y",
                        "width",
                        "height",
                        "area",
                        "area_ratio_to_movable_mean",
                        "height_ratio_to_movable_min",
                        "is_placement_blockage",
                        "excluded_place_blockage",
                        "is_fixed_macro",
                    ]
                )
                for rel_idx, node_id in enumerate(range(fixed_start, fixed_stop)):
                    is_placement_blockage = bool(placement_blockage_mask[rel_idx])
                    writer.writerow(
                        [
                            node_id,
                            rel_idx,
                            node_name(node_id),
                            float(self.node_x[node_id]),
                            float(self.node_y[node_id]),
                            float(self.node_size_x[node_id]),
                            float(self.node_size_y[node_id]),
                            float(node_areas[node_id]),
                            float(node_areas[node_id]) / area_den,
                            float(self.node_size_y[node_id]) / height_den,
                            int(is_placement_blockage),
                            int(is_placement_blockage and not self.fixed_macro_mask[rel_idx]),
                            int(bool(self.fixed_macro_mask[rel_idx])),
                        ]
                    )
            logging.info("Fixed macro heuristic mask debug CSV written to %s", out_path)
        except Exception as e:
            logging.warning("Failed to write fixed macro heuristic mask debug CSV: %s", e)

    def _fixed_placement_blockage_mask(self):
        fixed_count = max(int(self.fixed_slice.stop - self.fixed_slice.start), 0)
        blockage_count = min(
            max(int(getattr(self, "num_place_blockages", 0)), 0),
            fixed_count,
        )
        mask = np.zeros(fixed_count, dtype=bool)
        if blockage_count:
            mask[fixed_count - blockage_count:] = True
        return mask

    def update_macros(self, params, area_threshold=10, height_threshold=2):
        # set large cells as macros
        node_areas = self.node_size_x * self.node_size_y
        movable_mean_area = node_areas[self.movable_slice].mean()
        movable_min_height = self.node_size_y[self.movable_slice].min()
        mean_area = movable_mean_area * area_threshold
        row_height = movable_min_height * height_threshold

        # movable macros
        self.movable_macro_mask = (node_areas[self.movable_slice] > mean_area) & (
            self.node_size_y[self.movable_slice] > row_height
        )
        if params.macro_only:
            self.movable_macro_mask = self.macro_writeback_candidate[
                self.movable_slice].copy()
        self.movable_macro_pins = np.isin(self.pin2node_map, np.arange(
            0, self.num_movable_nodes)[self.movable_macro_mask])
        self.movable_macro_idx = np.where(self.movable_macro_mask)[0]
        self.num_movable_macros = self.movable_macro_idx.shape[0]
        # fixed macros
        self.fixed_macro_mask = (node_areas[self.fixed_slice] > mean_area) & (
            self.node_size_y[self.fixed_slice] > row_height
        )
        if params.macro_only:
            self.fixed_macro_mask = self.node_is_hard_macro[self.fixed_slice].copy()
        placement_blockage_mask = self._fixed_placement_blockage_mask()
        self.num_fixed_macro_excluded_place_blockages = int(
            np.count_nonzero(self.fixed_macro_mask & placement_blockage_mask)
        )
        self.fixed_macro_mask = self.fixed_macro_mask & ~placement_blockage_mask
        self.fixed_macro_idx = (
            self.num_movable_nodes + np.where(self.fixed_macro_mask)[0]
        )
        self.num_fixed_macros = self.fixed_macro_idx.shape[0]
        self._dump_fixed_macro_mask_debug(
            params,
            node_areas,
            movable_mean_area,
            movable_min_height,
            mean_area,
            row_height,
            area_threshold,
            height_threshold,
        )

        macro_pin_offset_x_mean = []
        macro_pin_offset_y_mean = []

        for macro_id in self.movable_macro_idx:
            pins = np.asarray(self.node2pin_map[macro_id], dtype=np.int32)
            # Pin offsets are measured relative to the macro origin in the same
            # scaled coordinate system as node_size_{x,y}.
            if pins.size:
                macro_pin_offset_x_mean.append(np.mean(self.pin_offset_x[pins]))
                macro_pin_offset_y_mean.append(np.mean(self.pin_offset_y[pins]))
            else:
                macro_pin_offset_x_mean.append(0.5 * self.node_size_x[macro_id])
                macro_pin_offset_y_mean.append(0.5 * self.node_size_y[macro_id])

        self.is_pin_lower_x = (np.array(macro_pin_offset_x_mean) <= 0.1 *
                               self.node_size_x[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_upper_x = (np.array(macro_pin_offset_x_mean) >= 0.9 *
                               self.node_size_x[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_lower_y = (np.array(macro_pin_offset_y_mean) <= 0.1 *
                               self.node_size_y[self.movable_slice][self.movable_macro_mask]).astype('float64')
        self.is_pin_upper_y = (np.array(macro_pin_offset_y_mean) >= 0.9 *
                               self.node_size_y[self.movable_slice][self.movable_macro_mask]).astype('float64')

        # setup macro padding for overlap loss
        # self.cell_padding_x = params.cell_padding_x
        # self.cell_padding_y = params.cell_padding_y
        self.bndry_padding_x = params.bndry_padding_x
        self.bndry_padding_y = params.bndry_padding_y

        # make sure the macros & halo sizes are multiples of site
        # All halo/padding parameters are snapped in scaled units so downstream
        # legalization still sees exact site/row multiples.
        params.macro_halo_x = self.crop_to_site(params.macro_halo_x, "x")
        params.macro_halo_y = self.crop_to_site(params.macro_halo_y, "y")
        params.macro_pin_halo_x = self.crop_to_site(
            params.macro_pin_halo_x, "x")
        params.macro_pin_halo_y = self.crop_to_site(
            params.macro_pin_halo_y, "y")
        params.cell_padding_x = self.crop_to_site(
            params.cell_padding_x, "x")
        self.cell_padding_x = params.cell_padding_x
        # params.cell_padding_y = self.crop_to_site(
        #     params.cell_padding_y, "y")
        self.node_size_x[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_x[self.movable_macro_idx], "x"
        )
        self.node_size_y[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_y[self.movable_macro_idx], "y"
        )

        # add halo around macros
        if params.macro_halo_x > 0 and params.macro_halo_y >= 0:
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
            # pin_offset_* must move with the expanded halo so absolute pin
            # locations stay unchanged after the macro origin shifts.
            self.movable_macro_pins = np.isin(
                self.pin2node_map, self.movable_macro_idx)
            self.pin_offset_x[self.movable_macro_pins] += params.macro_halo_x
            self.pin_offset_y[self.movable_macro_pins] += params.macro_halo_y
            # self.fixed_macro_pins = np.isin(self.pin2node_map, self.fixed_macro_idx)
            # self.pin_offset_x[self.fixed_macro_pins] += params.macro_halo_x
            # self.pin_offset_y[self.fixed_macro_pins] += params.macro_halo_y
        if params.macro_pin_halo_x >= 0:
            # Directional macro-pin halo grows only the sides that host pins,
            # so origin shifts and pin-offset compensation are side-dependent.
            self.node_size_x[self.movable_macro_idx] += self.is_pin_lower_x * \
                params.macro_pin_halo_x + self.is_pin_upper_x * params.macro_pin_halo_x
            self.node_size_y[self.movable_macro_idx] += self.is_pin_lower_y * \
                params.macro_pin_halo_y + self.is_pin_upper_y * params.macro_pin_halo_y

            self.node_x[self.movable_macro_idx] -= self.is_pin_lower_x * \
                params.macro_pin_halo_x
            self.node_y[self.movable_macro_idx] -= self.is_pin_lower_y * \
                params.macro_pin_halo_y

            macro_pin_list = self.pin2node_map[self.movable_macro_pins]
            move_macro_idx_list = np.where(
                macro_pin_list == self.movable_macro_idx[:, None]
            )
            self.pin_offset_x[self.movable_macro_pins] += (
                self.is_pin_lower_x[move_macro_idx_list[0]] * params.macro_pin_halo_x
            )
            self.pin_offset_y[self.movable_macro_pins] += (
                self.is_pin_lower_y[move_macro_idx_list[0]] * params.macro_pin_halo_y
            )

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
        if getattr(params, "place_io_engine", "ieda") in {"openroad", "ecc"}:
            for i in range(self.num_movable_nodes, self.num_movable_nodes + self.num_terminals):
                content += "\n%s %g %g : %s /FIXED" % (
                    str_node_names[i],
                    self.node_x[i],
                    self.node_y[i],
                    str_node_orient[i]
                )
        else:
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

    def write_placement_back(
        self, node_x, node_y, refresh_parasitics=True, *, refresh_pydb=True
    ):
        if self.params.macro_only:
            return self.pydb.write_macro_placement_back(node_x, node_y)
        # unscale locations
        # TODO:
        place_io_engine = getattr(self.params, "place_io_engine", "ieda")
        if place_io_engine == "openroad":
            import dreamplace.ops.placeio_openroad.place_io as placeio_openroad
            placeio_openroad.PlaceIOFunction.apply(self.openroad_bridge, node_x, node_y)
            # OpenROAD placement-coordinate writes invalidate placement parasitics.
            # Repair stages maintain their own timing state; this refresh belongs
            # only to the coordinate sync boundary.
            if refresh_parasitics:
                self.openroad_bridge.eval_tcl_string("estimate_parasitics -placement")
            if refresh_pydb:
                self.refresh_from_openroad_bridge()
        elif place_io_engine == "ecc":
            placeio_ecc = _load_placeio_ecc()
            placeio_ecc.PlaceIOFunction.apply(self.rawdb, node_x, node_y)
        else:
            placeio_ieda = _load_placeio_ieda()
            placeio_ieda.PlaceIOFunction.apply(self.rawdb, node_x, node_y)

    def write_sizing_back(self):
        self.last_sizing_writeback_summary = None
        if self.inst_cell_id is None or self.flat_libcell_names is None:
            return
        cell_ids = np.asarray(self.inst_cell_id, dtype=np.int32)
        cell_master_names = [_maybe_decode_libcell_name(name) for name in self.flat_libcell_names]
        preview = [
            (int(cell_ids[idx]), cell_master_names[int(cell_ids[idx])])
            for idx in range(min(10, len(cell_ids)))
            if 0 <= int(cell_ids[idx]) < len(cell_master_names)
        ]
        logging.info("write_sizing_back preview placedb_id=%s first_cells=%s", id(self), preview)
        place_io_engine = getattr(self.params, "place_io_engine", "ieda")
        if place_io_engine == "openroad":
            if self.openroad_bridge is None:
                raise RuntimeError("OpenROAD sizing writeback requires openroad_bridge")
            import dreamplace.ops.placeio_openroad.place_io as placeio_openroad
            result = placeio_openroad.PlaceIOFunction.apply_sizing(
                self.openroad_bridge,
                cell_ids,
                cell_master_names,
            )
            self.last_sizing_writeback_summary = result
            logging.info("openroad write_sizing_back result=%s", result)
            return result
        if place_io_engine == "ecc":
            placeio_ecc = _load_placeio_ecc()
            result = placeio_ecc.PlaceIOFunction.apply_sizing(
                self.rawdb,
                cell_ids,
                cell_master_names,
            )
            self.last_sizing_writeback_summary = result
            self.backend_caps = placeio_ecc.PlaceIOFunction.backend_caps(self.rawdb)
            logging.info(
                "ecc write_sizing_back ok=%s accepted=%s requested=%s rejected=%s",
                result.get("ok"), result.get("accepted_count"),
                result.get("requested_count"), result.get("rejected_count"),
            )
            return result

        placeio_ieda = _load_placeio_ieda()
        result = placeio_ieda.PlaceIOFunction.apply_sizing(
            self.rawdb,
            cell_ids,
            cell_master_names,
        )
        self.last_sizing_writeback_summary = result
        return result

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
        # Native coordinates are integer DBU. Float32 scaling can otherwise
        # turn a fixed coordinate such as 63872 into 63871 after truncation.
        node_x = np.rint(node_x).astype(np.int64)
        node_y = np.rint(node_y).astype(np.int64)
        sizing_result = None
        if getattr(params, "placement_sizing_mode", "place_only") != "place_only":
            sizing_result = self.write_sizing_back()
            if getattr(params, "place_io_engine", "ieda") == "ecc" and (
                not isinstance(sizing_result, dict)
                or not sizing_result.get("ok")
                or int(sizing_result.get("accepted_count", 0))
                != int(sizing_result.get("requested_count", -1))
            ):
                raise RuntimeError(f"ECC sizing writeback rejected: {sizing_result}")
        else:
            self.last_sizing_writeback_summary = None
        # update raw database
        self.write_placement_back(
            node_x,
            node_y,
            refresh_parasitics=(
                getattr(params, "placement_sizing_mode", "place_only") != "size_only"
            ),
        )
        if (
            getattr(params, "place_io_engine", "ieda") == "ecc"
            and getattr(self.rawdb, "timing_enabled", False)
            and isinstance(sizing_result, dict)
            and bool(sizing_result.get("ok"))
            and int(sizing_result.get("accepted_count", 0) or 0) > 0
        ):
            self.refresh_from_ecc_backend(
                refresh_mode="full_rebuild",
                rebuild_mode="full_rebuild",
            )
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
