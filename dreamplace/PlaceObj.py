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

import copy
import math
import os
import sys
import time
import json
import csv
from matplotlib.pyplot import step
import numpy as np
import itertools
import logging
import math
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
import dreamplace.ops.pin_utilization.pin_utilization as pin_utilization
import dreamplace.ops.nctugr_binary.nctugr_binary as nctugr_binary
import dreamplace.ops.irt_egr.irt_egr as eGR
import dreamplace.ops.adjust_node_area.adjust_node_area as adjust_node_area
import dreamplace.ops.macro_overlap.macro_overlap as macro_overlap
import dreamplace.ops.macro_refinement.macro_refinement as macro_refinement
from dreamplace.ops.timing_propagation.timing_propagation import (
    TimingPropagation,
    SMOOTH_MAX_ALPHA,
    build_pin_violation_detail_rows,
    smooth_max,
    write_pin_violation_detail_csv,
)
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
from dreamplace.ops.timing_propagation.timing_path_report import (
    build_critical_path_rows,
    timing_index_tensor,
    write_critical_path_report,
)
from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
try:
    from dreamplace.ops.cell_modeling import cell_modeling_op
except ImportError:
    cell_modeling_op = None
from dreamplace.ops.rc_timing.rc_timing import RCTiming
from dreamplace.BasicPlace import PlaceDataCollection
from dreamplace.ops.routability.plot_map import plot_node_grad_directions
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose, profile_scope
from dreamplace.ops.routability.l_shape_gradient import apply_l_shape_gradient
from dreamplace.ops.routability.l_shape_telemetry import reset_l_shape_gradient_telemetry
from dreamplace.ops.routability import enhanced_inflation_controller


def _effective_iopin_density_weight(params, placedb, region_id=None,
                                    fence_regions=None):
    if len(placedb.regions) > 0 or region_id is not None or fence_regions is not None:
        return 0.0
    return float(getattr(params, "iopin_density_weight", 3.0))


def _bool_like(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _effective_m2_pg_rail_density_weight(params, placedb, region_id=None,
                                         fence_regions=None):
    if len(placedb.regions) > 0 or region_id is not None or fence_regions is not None:
        return 0.0
    if _bool_like(getattr(params, "ieda_m2_pg_rail_blockage_flag", 0)):
        return 0.0
    return float(getattr(params, "m2_pg_rail_density_weight", 1.0))


def _effective_m2_pg_rail_density_boxes(
    params,
    placedb,
    data_collections,
    region_id=None,
    fence_regions=None,
):
    boxes = getattr(
        data_collections,
        "m2_pg_rail_density_boxes",
        getattr(placedb, "m2_pg_rail_density_boxes", None),
    )
    if boxes is None:
        return None

    if _effective_m2_pg_rail_density_weight(
        params, placedb, region_id=region_id, fence_regions=fence_regions
    ) <= 0:
        has_boxes = boxes.numel() > 0 if isinstance(boxes, torch.Tensor) else len(boxes) > 0
        if has_boxes:
            reason = (
                "fence-region electric field"
                if len(placedb.regions) > 0 or region_id is not None or fence_regions is not None
                else "disabled soft-density weight or enabled hard blockage flag"
            )
            logging.info(
                "M2 PG rail soft density skipped for %s",
                reason,
            )
        return None

    return boxes
import dreamplace.ops.pin2pin_attraction.pin2pin_attraction as pin2pin_attraction
from dreamplace.ops.size_interpolated_pin.sizing_limit_utils import (
    compute_size_interpolated_pin_properties,
    warmup_size_interpolated_pin_native_cache,
)
from dreamplace.flows.joint_proximal import JointProximalObjective

_AIMP_TRUE_VALUES = {"1", "true", "yes", "on"}
_AIMP_FALSE_VALUES = {"0", "false", "no", "off"}
_FIRST_LEVEL_NUMERIC_PROBE_PINS = (
    "g44249__69005:B2",
    "g44248__63144:B2",
    "g44175__85370:B1",
    "FE_OFC19_u_NV_NVDLA_cmac_u_reg_n_1479:Y",
    "g65266:Y",
)
_FIRST_LEVEL_NUMERIC_ALIGNMENT_FIELDNAMES = (
    "row_role",
    "pin_id",
    "pin_name",
    "normalized_pin_name",
    "net_id",
    "net_name",
    "driver_pin_id",
    "driver_pin_name",
    "source_start_pin_id",
    "source_start_pin_name",
    "reachable_static_first_fanout",
    "is_output_pin",
    "py_r_aat_ps",
    "py_f_aat_ps",
    "py_r_rat_ps",
    "py_f_rat_ps",
    "py_min_slack_ps",
    "py_r_slew_ps",
    "py_f_slew_ps",
    "py_r_cap_pf",
    "py_f_cap_pf",
    "slew_limit_ps",
    "cap_limit_pf",
    "iteration",
)


def _load_placeio_ieda():
    import dreamplace.ops.placeio_ieda.place_io as placeio_ieda

    return placeio_ieda
_PYTHON_ENDPOINT_SLACK_FIELDNAMES = (
    "pin_id",
    "pin_name",
    "normalized_pin_name",
    "py_r_aat_ps",
    "py_f_aat_ps",
    "py_r_rat_ps",
    "py_f_rat_ps",
    "py_r_slack_ps",
    "py_f_slack_ps",
    "py_min_slack_ps",
    "python_arrival_sentinel",
    "python_required_sentinel",
    "python_nonfinite",
    "python_negative_slack",
    "iteration",
)
_ENDPOINT_CONSTRAINT_DEBUG_FIELDNAMES = (
    "pin_id",
    "pin_name",
    "normalized_pin_name",
    "endpoint_check_class",
    "endpoint_pin_role",
    "exported_r_rat_ps",
    "exported_f_rat_ps",
    "py_r_rat_ps",
    "py_f_rat_ps",
    "setup_r_subtract_ps",
    "setup_f_subtract_ps",
    "constraint_arc_count",
    "timing_check_arc_count",
    "timing_check_classes",
    "query_r_clk_slew_ps",
    "query_f_clk_slew_ps",
    "query_r_data_slew_ps",
    "query_f_data_slew_ps",
    "r_setup_value_ps",
    "f_setup_value_ps",
    "constraint_lib_arc_indices",
    "constraint_lib_arc_offsets",
    "constraint_r_setup_values_ps",
    "constraint_f_setup_values_ps",
    "constraint_lib_cell_names",
    "constraint_lib_arc_from_pins",
    "constraint_lib_arc_to_pins",
    "constraint_timing_senses",
    "constraint_timing_types",
    "iteration",
)
_INPUT_BOUNDARY_FIELDNAMES = (
    "pin_id",
    "pin_name",
    "normalized_pin_name",
    "net_id",
    "net_name",
    "driver_pin_id",
    "driver_pin_name",
    "is_start_point",
    "is_output_pin",
    "exported_inrdelay_ps",
    "exported_infdelay_ps",
    "exported_inrtrans_ps",
    "exported_inftrans_ps",
    "py_r_aat_ps",
    "py_f_aat_ps",
    "py_r_slew_ps",
    "py_f_slew_ps",
    "iteration",
)
_FIRST_LEVEL_LUT_PROBE_FIELDNAMES = (
    "arc_index",
    "arc_join_key",
    "in_pin_id",
    "in_pin_name",
    "in_pin_normalized",
    "out_pin_id",
    "out_pin_name",
    "out_pin_normalized",
    "lib_cell_idx",
    "lib_cell_name",
    "lib_arc_idx",
    "lib_arc_from_pin",
    "lib_arc_to_pin",
    "arc_offset",
    "lut_boundary_mode",
    "timing_sense",
    "timing_sense_name",
    "timing_type",
    "timing_type_name",
    "query_r_input_slew_ps",
    "query_f_input_slew_ps",
    "query_r_output_cap_pf",
    "query_f_output_cap_pf",
    "r_delay_value_ps",
    "f_delay_value_ps",
    "r_transition_value_ps",
    "f_transition_value_ps",
    "r_delay_trans_min_ps",
    "r_delay_trans_max_ps",
    "r_delay_cap_min_pf",
    "r_delay_cap_max_pf",
    "f_delay_trans_min_ps",
    "f_delay_trans_max_ps",
    "f_delay_cap_min_pf",
    "f_delay_cap_max_pf",
    "r_transition_trans_min_ps",
    "r_transition_trans_max_ps",
    "r_transition_cap_min_pf",
    "r_transition_cap_max_pf",
    "f_transition_trans_min_ps",
    "f_transition_trans_max_ps",
    "f_transition_cap_min_pf",
    "f_transition_cap_max_pf",
    "iteration",
)
_CELL_ARC_PY_REPORT_FIELDNAMES = (
    "arc_index",
    "arc_join_key",
    "in_pin_id",
    "in_pin_name",
    "in_pin_normalized",
    "out_pin_id",
    "out_pin_name",
    "out_pin_normalized",
    "lib_cell_idx",
    "lib_cell_name",
    "lib_arc_idx",
    "lib_arc_from_pin",
    "lib_arc_to_pin",
    "arc_offset",
    "timing_sense",
    "timing_sense_name",
    "timing_type",
    "timing_type_name",
    "py_output_rise_delay_ps",
    "py_output_fall_delay_ps",
    "py_output_rise_input_slew_ps",
    "py_output_fall_input_slew_ps",
    "py_output_rise_load_pf",
    "py_output_fall_load_pf",
    "r_delay_value_ps",
    "f_delay_value_ps",
    "r_transition_value_ps",
    "f_transition_value_ps",
    "iteration",
)
_CELL_ARC_PY_SEMANTIC_SUMMARY_FIELDNAMES = (
    "lib_cell_name",
    "lib_arc_from_pin",
    "lib_arc_to_pin",
    "timing_sense",
    "timing_sense_name",
    "arc_count",
    "mean_py_output_rise_delay_ps",
    "mean_py_output_fall_delay_ps",
    "mean_py_output_rise_input_slew_ps",
    "mean_py_output_fall_input_slew_ps",
    "mean_py_output_rise_load_pf",
    "mean_py_output_fall_load_pf",
    "mean_query_r_input_slew_ps",
    "mean_query_f_input_slew_ps",
    "mean_query_r_output_cap_pf",
    "mean_query_f_output_cap_pf",
    "mean_r_delay_value_ps",
    "mean_f_delay_value_ps",
    "mean_r_transition_value_ps",
    "mean_f_transition_value_ps",
    "iteration",
)
_CELL_ARC_LUT_FINGERPRINT_FIELDNAMES = (
    "lib_cell_name",
    "lib_arc_from_pin",
    "lib_arc_to_pin",
    "timing_sense",
    "timing_sense_name",
    "lib_arc_idx",
    "arc_offset",
    "r_delay_trans_dim",
    "r_delay_cap_dim",
    "r_delay_trans_min_ps",
    "r_delay_trans_max_ps",
    "r_delay_cap_min_pf",
    "r_delay_cap_max_pf",
    "r_delay_value_min_ps",
    "r_delay_value_max_ps",
    "r_delay_value_sum_ps",
    "r_delay_value_sample0_ps",
    "r_delay_value_sample_mid_ps",
    "r_delay_value_sample_last_ps",
    "f_delay_trans_dim",
    "f_delay_cap_dim",
    "f_delay_trans_min_ps",
    "f_delay_trans_max_ps",
    "f_delay_cap_min_pf",
    "f_delay_cap_max_pf",
    "f_delay_value_min_ps",
    "f_delay_value_max_ps",
    "f_delay_value_sum_ps",
    "f_delay_value_sample0_ps",
    "f_delay_value_sample_mid_ps",
    "f_delay_value_sample_last_ps",
    "r_transition_trans_dim",
    "r_transition_cap_dim",
    "r_transition_trans_min_ps",
    "r_transition_trans_max_ps",
    "r_transition_cap_min_pf",
    "r_transition_cap_max_pf",
    "r_transition_value_min_ps",
    "r_transition_value_max_ps",
    "r_transition_value_sum_ps",
    "r_transition_value_sample0_ps",
    "r_transition_value_sample_mid_ps",
    "r_transition_value_sample_last_ps",
    "f_transition_trans_dim",
    "f_transition_cap_dim",
    "f_transition_trans_min_ps",
    "f_transition_trans_max_ps",
    "f_transition_cap_min_pf",
    "f_transition_cap_max_pf",
    "f_transition_value_min_ps",
    "f_transition_value_max_ps",
    "f_transition_value_sum_ps",
    "f_transition_value_sample0_ps",
    "f_transition_value_sample_mid_ps",
    "f_transition_value_sample_last_ps",
    "iteration",
)
_STARTPOINT_ARC_REWRITE_FIELDNAMES = (
    "pin_id",
    "pin_name",
    "normalized_pin_name",
    "net_id",
    "net_name",
    "is_output_pin",
    "exported_inrdelay_ps",
    "exported_infdelay_ps",
    "py_r_aat_ps",
    "py_f_aat_ps",
    "max_r_aat_delta_ps",
    "max_f_aat_delta_ps",
    "level0_as_inst_out_arc_count",
    "non_level0_as_inst_out_arc_count",
    "total_as_inst_out_arc_count",
    "as_inst_in_arc_count",
    "as_net_out_arc_count",
    "as_net_self_loop_count",
    "as_net_in_arc_count",
    "first_non_level0_inst_arc_index",
    "first_non_level0_inst_arc_level",
    "first_non_level0_inst_arc_join_key",
    "first_non_level0_in_pin_name",
    "first_non_level0_out_pin_name",
    "first_non_level0_lib_cell_name",
    "first_non_level0_lib_arc_from_pin",
    "first_non_level0_lib_arc_to_pin",
    "first_non_level0_timing_sense_name",
    "first_non_level0_timing_type_name",
    "first_net_out_arc_index",
    "first_net_out_in_pin_name",
    "first_net_out_out_pin_name",
    "iteration",
)
_NET_TOPOLOGY_DEBUG_FIELDNAMES = (
    "net_id",
    "net_name",
    "driver_pin_id",
    "driver_pin_name",
    "normalized_driver_pin_name",
    "flat_net_first_pin_id",
    "flat_net_first_pin_name",
    "driver_is_flat_net_first_pin",
    "rc_traversal_seed_pin_id",
    "rc_traversal_seed_pin_name",
    "fanout_count",
    "sink_count",
    "unreached_net_pin_count",
    "net_pin_ids",
    "net_pin_names",
    "sink_pin_ids",
    "sink_pin_names",
    "driver_py_r_aat_ps",
    "driver_py_f_aat_ps",
    "driver_py_r_slew_ps",
    "driver_py_f_slew_ps",
    "driver_py_r_cap_pf",
    "driver_py_f_cap_pf",
    "driver_cap_limit_pf",
    "sum_sink_cap_base_pf",
    "sum_sink_rcap_base_pf",
    "sum_sink_fcap_base_pf",
    "sum_node_wire_cap_pf",
    "estimated_total_cap_plus_wire_pf",
    "estimated_total_rcap_plus_wire_pf",
    "estimated_total_fcap_plus_wire_pf",
    "max_sink_cap_base_pf",
    "max_sink_rcap_base_pf",
    "max_sink_fcap_base_pf",
    "iteration",
)
_NET_SINK_CAP_DEBUG_FIELDNAMES = (
    "net_id",
    "net_name",
    "driver_pin_id",
    "driver_pin_name",
    "normalized_driver_pin_name",
    "sink_pin_id",
    "sink_pin_name",
    "normalized_sink_pin_name",
    "sink_node_id",
    "sink_main_id",
    "sink_libcell_idx",
    "sink_libcell_name",
    "sink_libpin_flat_id",
    "sink_libpin_name",
    "sink_pin_offset",
    "sink_cap_base_pf",
    "sink_rcap_base_pf",
    "sink_fcap_base_pf",
    "sink_py_r_aat_ps",
    "sink_py_f_aat_ps",
    "sink_py_r_slew_ps",
    "sink_py_f_slew_ps",
    "iteration",
)


def _env_flag(name, default=False):
    value = os.environ.get(name, "")
    normalized = str(value).strip().lower()
    if normalized in _AIMP_TRUE_VALUES:
        return True
    if normalized in _AIMP_FALSE_VALUES:
        return False
    return bool(default)


def _decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_join_pin_name(value):
    return _decode_name(value).replace("/", ":").replace("\\[", "[").replace("\\]", "]")


def _tensor_to_cpu_float_array(values):
    if values is None:
        return None
    if hasattr(values, "detach"):
        return values.detach().cpu().float().numpy()
    return np.asarray(values, dtype=np.float32)


def _tensor_to_cpu_long_array(values):
    if values is None:
        return None
    if hasattr(values, "detach"):
        return values.detach().cpu().long().numpy()
    return np.asarray(values, dtype=np.int64)


def _timing_tensor_to_cpu_float_array(timing_op, *names):
    for name in names:
        array = _tensor_to_cpu_float_array(getattr(timing_op, name, None))
        if array is not None:
            return array
    return None


def _timing_tensor_ref(timing_op, *names):
    for name in names:
        value = getattr(timing_op, name, None)
        if value is not None:
            return value
    return None


def _finite_or_blank(value):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(number) else number


def _numeric_delta(lhs, rhs):
    lhs_value = _finite_or_blank(lhs)
    rhs_value = _finite_or_blank(rhs)
    if lhs_value == "" or rhs_value == "":
        return None
    return float(lhs_value) - float(rhs_value)


def _timing_sense_name(value):
    if value == 1:
        return "positive_unate"
    if value == -1:
        return "negative_unate"
    return "non_unate"


def _timing_type_name(value):
    if value == 1:
        return "rising_edge"
    if value == -1:
        return "falling_edge"
    return "both_edge"


def _endpoint_pin_role(pin_name):
    normalized = _normalize_join_pin_name(pin_name)
    leaf = normalized.rsplit(":", 1)[-1].upper()
    if leaf in {"D", "DI", "DATA", "DIN"}:
        return "data"
    if leaf in {"CLK", "CK", "CP", "CLOCK", "GCLK"}:
        return "clock"
    if any(token in leaf for token in ("RESET", "RST", "CLR", "CLEAR", "SET", "PRESET")):
        return "async_control"
    return "unknown"


def _endpoint_check_class(pin_name, constraint_arc_count):
    if constraint_arc_count > 0:
        return "setup_exported"
    role = _endpoint_pin_role(pin_name)
    if role == "async_control":
        return "async_control_unmodeled"
    return "unsupported_unmodeled"


def _endpoint_timing_check_bucket(check_classes):
    classes = {int(value) for value in check_classes}
    if 1 in classes:
        return "setup"
    if 2 in classes and (3 in classes or 4 in classes):
        return "hold_async"
    if 2 in classes:
        return "hold"
    if 3 in classes or 4 in classes:
        return "async_recovery_removal"
    return "unmodeled"


def _summarize_endpoint_compare_buckets(rows):
    buckets = {}

    def update(bucket_name, row):
        bucket = buckets.setdefault(
            bucket_name,
            {
                "count": 0,
                "python_negative_slack": 0,
                "backend_negative_slack": 0,
                "backend_check_negative_slack": 0,
                "python_wns": None,
                "backend_wns": None,
                "backend_check_wns": None,
                "python_tns": 0.0,
                "backend_tns": 0.0,
                "backend_check_tns": 0.0,
            },
        )
        bucket["count"] += 1
        for prefix, key in (("python", "py_slack"), ("backend", "cpp_slack")):
            value = row.get(key)
            if value is None or not np.isfinite(value):
                continue
            bucket[f"{prefix}_wns"] = (
                value
                if bucket[f"{prefix}_wns"] is None
                else min(bucket[f"{prefix}_wns"], value)
            )
            if value < 0.0:
                bucket[f"{prefix}_negative_slack"] += 1
                bucket[f"{prefix}_tns"] += float(value)
        check_value = _endpoint_backend_check_slack(row)
        if check_value is not None and np.isfinite(check_value):
            bucket["backend_check_wns"] = (
                check_value
                if bucket["backend_check_wns"] is None
                else min(bucket["backend_check_wns"], check_value)
            )
            if check_value < 0.0:
                bucket["backend_check_negative_slack"] += 1
                bucket["backend_check_tns"] += float(check_value)

    for row in rows:
        update("all", row)
        update(f"check_bucket:{row.get('timing_check_bucket', 'unknown')}", row)
        for cls in row.get("timing_check_classes_list", []):
            update(f"check_class:{int(cls)}", row)

    return buckets


def _endpoint_backend_check_slack(row):
    classes = {int(value) for value in row.get("timing_check_classes_list", [])}
    values = []
    if classes & {1, 3}:
        values.append(row.get("cpp_slack"))
    if classes & {2, 4}:
        values.append(row.get("cpp_min_slack"))
    finite_values = [
        value for value in values if value is not None and np.isfinite(value)
    ]
    if finite_values:
        return min(finite_values)

    bucket = row.get("timing_check_bucket", "unmodeled")
    if bucket in {"hold", "hold_async"}:
        return row.get("cpp_min_slack")
    return row.get("cpp_slack")


def _backend_min_slack_from_endpoint_values(
    min_r_aat, min_f_aat, min_r_rat, min_f_rat, sentinel_threshold=5e7
):
    values = []
    for arrival, required in ((min_r_aat, min_r_rat), (min_f_aat, min_f_rat)):
        if not np.isfinite(arrival) or not np.isfinite(required):
            continue
        if arrival <= -sentinel_threshold or required >= sentinel_threshold:
            continue
        values.append(arrival - required)
    if not values:
        return np.nan
    return min(values)


class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """

    def __init__(
        self, placedb, data_collections, op_collections, precond_pin_count_flag=False,
        use_net_weights=False,
    ):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.precond_pin_count_flag = bool(precond_pin_count_flag)
        self.use_net_weights = bool(use_net_weights)
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
            precond = self._build_precondition(density_weight)
            self._apply_precondition_to_grad(
                grad, precond, update_mask=update_mask, fix_nodes_mask=fix_nodes_mask
            )
            self._advance_state()
        return grad

    def apply_components(
        self, grads, density_weight, update_mask=None, fix_nodes_mask=None
    ):
        """Precondition multiple gradient components with one shared state step."""
        with torch.no_grad():
            precond = self._build_precondition(density_weight)
            outputs = []
            for grad in grads:
                component = grad.clone()
                self._apply_precondition_to_grad(
                    component,
                    precond,
                    update_mask=update_mask,
                    fix_nodes_mask=fix_nodes_mask,
                )
                outputs.append(component)
            self._advance_state()
        return outputs

    def _build_precondition(self, density_weight):
        if density_weight.size(0) == 1:
            density_precond = (
                self.alpha * density_weight * self.data_collections.node_areas
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
            density_precond = self.alpha * node_areas

        precond = density_precond
        if self.use_net_weights:
            precond = precond + self.op_collections.pws_op(
                self.data_collections.net_weights
            )
        elif self.precond_pin_count_flag:
            # Original DreamPlace pin counts, including zero counts for fillers.
            precond = precond + self.data_collections.num_pins_in_nodes
        return precond.clamp(min=1.0)

    def _apply_precondition_to_grad(
        self, grad, precond, update_mask=None, fix_nodes_mask=None
    ):
        grad[self.placedb.num_movable_nodes : self.placedb.num_physical_nodes] = 0
        grad[
            self.placedb.num_nodes
            + self.placedb.num_movable_nodes : self.placedb.num_nodes
            + self.placedb.num_physical_nodes
        ] = 0
        grad[0 : self.placedb.num_nodes].div_(precond)
        grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)
        # stop gradients for terminated electric field
        if update_mask is not None:
            grad2 = grad.view(2, -1)
            stop_mask = ~update_mask
            movable_mask = stop_mask[self.movablenode2fence_region_map_clamp]
            filler_mask = stop_mask[self.filler2fence_region_map]
            grad2[0, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
            grad2[1, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
            grad2[
                0, self.placedb.num_nodes - self.placedb.num_filler_nodes :
            ].masked_fill_(filler_mask, 0)
            grad2[
                1, self.placedb.num_nodes - self.placedb.num_filler_nodes :
            ].masked_fill_(filler_mask, 0)
        if fix_nodes_mask is not None:
            grad2 = grad.view(2, -1)
            grad2[0, : self.placedb.num_movable_nodes].masked_fill_(
                fix_nodes_mask[: self.placedb.num_movable_nodes], 0
            )
            grad2[1, : self.placedb.num_movable_nodes].masked_fill_(
                fix_nodes_mask[: self.placedb.num_movable_nodes], 0
            )
        return grad

    def _advance_state(self):
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
        self.timing_wns_coeff = float(getattr(params, "timing_wns_coeff", 0.01))
        self.timing_tns_coeff = float(getattr(params, "timing_tns_coeff", 0.0001))
        self.timing_grad_balance_target_ratio = float(
            getattr(params, "timing_grad_balance_target_ratio", 0.0)
        )
        if self.timing_grad_balance_target_ratio < 0.0:
            raise ValueError("timing_grad_balance_target_ratio must be nonnegative")
        self.timing_grad_balance_weight = 1.0
        self.timing_grad_balance_summary = {
            "artifact": "timing_grad_balance",
            "artifact_version": 1,
            "status": (
                "pending"
                if self.timing_grad_balance_target_ratio > 0.0
                else "disabled"
            ),
            "initialized": self.timing_grad_balance_target_ratio <= 0.0,
            "target_ratio": self.timing_grad_balance_target_ratio,
            "norm": "l1",
            "weight_applied": 1.0,
        }
        self.timing_slew_weight = float(getattr(params, "timing_slew_weight", 1.0))
        self.timing_cap_weight = float(getattr(params, "timing_cap_weight", 1.0))
        self.timing_leakage_weight = float(getattr(params, "timing_leakage_weight", 0.0))
        self.last_timing_objective_terms = {}
        self.size_density_area_weight = float(
            getattr(params, "size_density_area_weight", 0.1)
        )
        self.size_density_area_penalty = None
        self.wns = None
        self.tns = None
        self.ws = None
        self.ts = None
        self.max_slew_violation = None
        self.max_load_cap_violation = None
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
        self.joint_proximal_objective = JointProximalObjective(params)
        self.last_joint_proximal_summary = {
            "enabled": bool(self.joint_proximal_objective.enabled)
        }
        self.data_collections.pin_slack = None
        self.op_collections = op_collections
        self.global_place_params = global_place_params
        self._relaxed_buffer_dynamic_net_provider = None
        self._relaxed_buffer_dynamic_net_provider_key = None
        self._timing_geometry_cache = None
        self._virtual_cell_density_view = None
        self.pin2pin_weight = params.pin2pin_weight

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
        
        if params.enable_net_weighting and params.net_weighting_scheme =="pin2pin":
            self.op_collections.pin2pin_net_weight_op = self.build_pin2pin_net_weight(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op
            )
            
        if params.with_sta:
            size_parameter_getter = getattr(
                self.data_collections,
                "get_continuous_size_parameter",
                None,
            )
            size_parameter = (
                size_parameter_getter()
                if callable(size_parameter_getter)
                else getattr(self.data_collections, "size_logits", None)
            )
            if size_parameter is not None:
                self.op_collections.cell_modeling_op = CellModeling(self.data_collections)
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

        # Placeholder only. Real l-shape weight is calibrated from gradient ratio
        # the first time a valid l-shape gradient is observed.
        self.l_shape_routability_weight_init = 0.0
        self.l_shape_routability_weight = torch.tensor(
            [self.l_shape_routability_weight_init],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device,
        )
        # 目标：L-shape梯度范数占density梯度范数的比例
        l_shape_grad_target_ratio = float(
            getattr(params, "l_shape_grad_target_ratio", 0.2)
        )
        l_shape_grad_target_ratio_max = max(
            0.0, float(getattr(params, "l_shape_grad_target_ratio_max", 0.2))
        )
        self.l_shape_grad_target_ratio = max(
            0.0, min(l_shape_grad_target_ratio_max, l_shape_grad_target_ratio)
        )
        # Filler reverse force: push fillers toward congested areas using L-shape field
        self.l_shape_filler_reverse_force = float(
            getattr(params, "l_shape_filler_reverse_force", 0.0)
        )
        # Filler pseudo wire force: pull random fillers to most congested point.
        self.l_shape_filler_pseudo_wire_ratio = float(
            getattr(params, "l_shape_filler_pseudo_wire_ratio", 0.0)
        )
        # 权重调整的平滑因子 (0~1, 越小越平滑)
        self.l_shape_weight_momentum = getattr(params, 'l_shape_weight_momentum', 0.1)
        self.l_shape_weight_min = float(
            getattr(params, "l_shape_weight_min", 1e-12)
        )
        self.l_shape_weight_max = float(
            getattr(params, "l_shape_weight_max", 1.0)
        )
        self._l_shape_weight_initialized = False
        self._l_shape_outer_iteration = None
        reset_l_shape_gradient_telemetry(self)
        self.l_shape_macro_exclusion_last_summary = {}
        self._l_shape_auto_disabled = False
        self._l_shape_auto_disable_state = {}
        # ==========================================
        if self._should_warmup_size_interpolated_pin_native_cache():
            warmup_size_interpolated_pin_native_cache(self.data_collections)

    def _timing_objective_lane(self):
        return getattr(self.params, "timing_objective_lane", "timing_only")

    def _timing_objective_lane_enabled_terms(self):
        lane = self._timing_objective_lane()
        mapping = {
            "timing_only": ("timing",),
            "timing_slew": ("timing", "slew"),
            "timing_cap": ("timing", "cap"),
            "timing_slew_cap": ("timing", "slew", "cap"),
            "timing_slew_cap_leakage": ("timing", "slew", "cap", "leakage"),
        }
        return mapping.get(lane, ("timing",))

    def _should_warmup_size_interpolated_pin_native_cache(self):
        if not self._production_fast_loop_enabled():
            return False
        if not bool(getattr(self.params, "differentiable_timing_obj", False)):
            return False
        return "slew" in self._timing_objective_lane_enabled_terms() or "cap" in self._timing_objective_lane_enabled_terms()

    def _should_compute_timing_aux_tensor(self, term_name):
        if not self._production_fast_loop_enabled():
            return True
        return term_name in self._timing_objective_lane_enabled_terms()

    def _active_cone_aux_pin_masks(self, timing_op):
        result = getattr(timing_op, "last_traversal_pruning_result", None)
        stats = getattr(timing_op, "last_traversal_pruning_stats", None)
        arcs = getattr(timing_op, "flat_inst_arcs_by_level", None)
        if not (
            isinstance(stats, dict)
            and stats.get("applied")
            and result is not None
            and torch.is_tensor(getattr(result, "kept_flat_arc_indices", None))
            and torch.is_tensor(arcs)
        ):
            return None, None, {
                "active_cone_aux_pin_mask_applied": False,
                "reason": "traversal_pruning_unavailable",
            }

        kept_idx = result.kept_flat_arc_indices.to(device=arcs.device, dtype=torch.long)
        if kept_idx.numel() == 0:
            return None, None, {
                "active_cone_aux_pin_mask_applied": False,
                "reason": "empty_kept_arcs",
            }
        kept_arcs = arcs.index_select(0, kept_idx)
        active_pins = torch.unique(kept_arcs[:, :2].reshape(-1).long())
        active_pins = active_pins[
            (active_pins >= 0) & (active_pins < int(self.inst_pins_mask.numel()))
        ]
        if active_pins.numel() == 0:
            return None, None, {
                "active_cone_aux_pin_mask_applied": False,
                "reason": "empty_active_pins",
            }

        pin_device = self.inst_pins_mask.device
        active_pins = active_pins.to(device=pin_device)
        active_pin_mask = torch.zeros_like(self.inst_pins_mask, dtype=torch.bool)
        active_pin_mask[active_pins] = True
        slew_mask = self.inst_pins_mask & active_pin_mask
        cap_mask = self.output_pin_mask & active_pin_mask.to(device=self.output_pin_mask.device)
        return slew_mask, cap_mask, {
            "active_cone_aux_pin_mask_applied": True,
            "active_aux_pin_count": int(torch.count_nonzero(slew_mask).detach().item()),
            "active_aux_output_pin_count": int(torch.count_nonzero(cap_mask).detach().item()),
            "active_aux_kept_arc_count": int(kept_idx.numel()),
        }

    def _should_use_active_cone_aux_pin_masks(self):
        return (
            self._production_fast_loop_enabled()
            and os.environ.get("AIMP_ACTIVE_CONE_AUX_PIN_MASK", "").strip().lower()
            in _AIMP_TRUE_VALUES
        )

    def _should_reuse_zero_slew_aux(self, timing_op):
        if (
            self._relaxed_buffer_timing_enabled()
            or "slew" in self._timing_objective_lane_enabled_terms()
        ):
            return False
        if not self._production_fast_loop_enabled():
            return False
        if _env_flag("AIMP_DISABLE_REUSE_ZERO_SLEW_AUX"):
            return False
        scalar_value = getattr(timing_op, "last_total_slew_violation", None)
        if scalar_value is None:
            return False
        try:
            return abs(float(scalar_value)) == 0.0
        except (TypeError, ValueError):
            return False

    def _reused_zero_slew_aux_tensor(self, timing_op):
        existing = getattr(timing_op, "last_total_slew_violation_tensor", None)
        if torch.is_tensor(existing):
            return existing.detach() * 0.0
        pos = getattr(self, "pos", None)
        if pos is not None and len(pos) > 0:
            return torch.zeros((), dtype=pos[0].dtype, device=pos[0].device)
        return torch.tensor(0.0, dtype=torch.float32)

    def _timing_rc_cap_input_tensor(self, pin_cap_tensor):
        if not self._production_fast_loop_enabled():
            return pin_cap_tensor
        if "cap" in self._timing_objective_lane_enabled_terms():
            return pin_cap_tensor
        if (
            os.environ.get("AIMP_DETACH_TIMING_RC_CAP_INPUTS", "").strip().lower()
            not in _AIMP_TRUE_VALUES
        ):
            return pin_cap_tensor
        return pin_cap_tensor.detach()

    def _maybe_write_rc_root_load_probe(
        self,
        *,
        pin_rcaps_base,
        pin_caps_rise,
        loads_rise,
    ):
        probe_path = os.environ.get("AIMP_RC_TIMING_ROOT_LOAD_PROBE_JSON")
        if not probe_path or getattr(self, "_rc_root_load_probe_written", False):
            return
        target_net_ids = []
        raw_target_net_ids = os.environ.get("AIMP_RC_TIMING_ROOT_LOAD_PROBE_NET_IDS", "")
        if raw_target_net_ids:
            for item in raw_target_net_ids.split(","):
                item = item.strip()
                if item:
                    target_net_ids.append(int(item))
        cap_overlay_path = os.environ.get("AIMP_BUFFERING_SEGMENT_CAP_OVERLAY_PROBE_JSON")
        if not target_net_ids and cap_overlay_path and os.path.exists(cap_overlay_path):
            try:
                with open(cap_overlay_path, "r", encoding="utf-8") as f:
                    cap_overlay = json.load(f)
                target_net_ids = [
                    int(row["net_id"])
                    for row in cap_overlay.get("top_abs_delta", [])
                    if row.get("net_id") is not None
                ]
            except Exception:
                target_net_ids = []
        if not target_net_ids:
            target_net_ids = list(range(20))

        topo = self.data_collections.net_flat_topo_sort.detach().cpu().tolist()
        topo_start = self.data_collections.net_flat_topo_sort_start.detach().cpu().tolist()
        flat_pin_from = self.data_collections.flat_pin_from.detach().cpu().tolist()
        flat_pin_to = self.data_collections.flat_pin_to.detach().cpu().tolist()
        pin_rcaps_base_cpu = pin_rcaps_base.detach().cpu()
        pin_caps_rise_cpu = pin_caps_rise.detach().cpu()
        loads_rise_cpu = loads_rise.detach().cpu()
        net_cap_cpu = getattr(self.op_collections.elmore_delay_op, "net_cap", None)
        net_cap_cpu = None if net_cap_cpu is None else net_cap_cpu.detach().cpu()

        rows = []
        def _sum_by_valid_node_ids(tensor, node_ids):
            valid = [int(node) for node in node_ids if int(node) < int(tensor.numel())]
            if not valid:
                return 0.0, 0
            return float(tensor[valid].sum().item()), int(torch.count_nonzero(tensor[valid]).item())

        for net_id in target_net_ids[:20]:
            net_id = int(net_id)
            if net_id < 0 or net_id + 1 >= len(topo_start):
                continue
            begin = int(topo_start[net_id])
            end = int(topo_start[net_id + 1])
            nodes = [int(node) for node in topo[begin:end]]
            if not nodes:
                continue
            root = nodes[0]
            node_set = set(nodes)
            base_sum, nonzero_base_count = _sum_by_valid_node_ids(
                pin_rcaps_base_cpu,
                nodes,
            )
            cap_sum = float(pin_caps_rise_cpu[nodes].sum().item())
            net_cap_sum = (
                float(net_cap_cpu[nodes].sum().item())
                if net_cap_cpu is not None
                else None
            )
            rows.append(
                {
                    "net_id": net_id,
                    "root_node_id": int(root),
                    "topology_node_count": int(len(nodes)),
                    "root_load_rise": float(loads_rise_cpu[root].item()),
                    "pin_rcaps_base_slice_sum": base_sum,
                    "pin_caps_rise_slice_sum": cap_sum,
                    "net_half_cap_slice_sum": net_cap_sum,
                    "pin_caps_minus_base_slice_sum": (
                        None if net_cap_sum is None else float(cap_sum - base_sum)
                    ),
                    "nonzero_base_cap_node_count": int(nonzero_base_count),
                    "nonzero_pin_cap_node_count": int(
                        torch.count_nonzero(pin_caps_rise_cpu[nodes]).item()
                    ),
                    "steiner_or_padded_node_count": int(
                        sum(1 for node in nodes if int(node) >= int(pin_rcaps_base_cpu.numel()))
                    ),
                    "edge_count": int(
                        sum(
                            1
                            for src, dst in zip(flat_pin_from, flat_pin_to)
                            if int(src) in node_set and int(dst) in node_set
                        )
                    ),
                }
            )
        payload = {
            "artifact": "rc_timing_root_load_probe",
            "artifact_version": 1,
            "rows": rows,
        }
        os.makedirs(os.path.dirname(probe_path), exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._rc_root_load_probe_written = True

    def _timing_objective_term_tensor(self, attr_name, scalar_attr_name, default_scale=1.0):
        op_collections = getattr(self, "op_collections", None)
        timing_op = None if op_collections is None else getattr(op_collections, "timing_propagation_op", None)
        if timing_op is None:
            return None
        tensor_value = getattr(timing_op, attr_name, None)
        if torch.is_tensor(tensor_value):
            return tensor_value
        scalar_value = getattr(timing_op, scalar_attr_name, None)
        if scalar_value is None:
            return None
        pos = getattr(self, "pos", None)
        if pos is not None and len(pos) > 0:
            dtype = pos[0].dtype
            device = pos[0].device
        else:
            dtype = torch.float32
            device = None
        return torch.as_tensor(
            float(scalar_value) * float(default_scale),
            dtype=dtype,
            device=device,
        )

    def _should_detach_zero_timing_aux_term(self, term_name, raw_term, scalar_value=None):
        if term_name == "timing" or not self._production_fast_loop_enabled():
            return False
        if scalar_value is not None:
            try:
                return abs(float(scalar_value)) == 0.0
            except (TypeError, ValueError):
                return False
        try:
            return float(raw_term.detach().abs().item()) == 0.0
        except (RuntimeError, TypeError, ValueError):
            return False

    def _critical_endpoint_selected_timing(self, full_wns, full_tns):
        mode = getattr(self.params, "critical_endpoint_pruning_mode", "off")
        metadata = {
            "mode": mode,
            "applied": False,
            "active_endpoint_count": 0,
            "selected_endpoint_wns_ps": None,
            "selected_endpoint_tns_ps": None,
            "full_endpoint_wns_ps": float(full_wns.detach().item()),
            "full_endpoint_tns_ps": float(full_tns.detach().item()),
        }
        op_collections = getattr(self, "op_collections", None)
        timing_op = None if op_collections is None else getattr(op_collections, "timing_propagation_op", None)
        selector = None if timing_op is None else getattr(timing_op, "select_critical_endpoint_timing", None)
        if not callable(selector):
            return full_wns, full_tns, metadata
        return selector(full_wns, full_tns)

    def _timing_objective_term_components(self, wns, tns):
        enabled_terms = self._timing_objective_lane_enabled_terms()
        effective_wns, effective_tns, pruning_metadata = (
            self._critical_endpoint_selected_timing(wns, tns)
        )
        timing_term = -(
            self.timing_wns_coeff * effective_wns
            + self.timing_tns_coeff * effective_tns
        )
        zero = timing_term * 0.0
        raw_terms = {
            "timing": timing_term,
            "slew": self._timing_objective_term_tensor(
                "last_total_slew_violation_tensor",
                "last_total_slew_violation",
                default_scale=1.0,
            ),
            "cap": self._timing_objective_term_tensor(
                "last_total_cap_violation_tensor",
                "last_total_cap_violation",
                default_scale=1.0,
            ),
            "leakage": self._timing_objective_term_tensor(
                "last_total_leakage_tensor",
                "last_total_leakage",
                default_scale=1.0,
            ),
        }
        op_collections = getattr(self, "op_collections", None)
        timing_op = None if op_collections is None else getattr(op_collections, "timing_propagation_op", None)
        scalar_terms = {
            "slew": None if timing_op is None else getattr(timing_op, "last_total_slew_violation", None),
            "cap": None if timing_op is None else getattr(timing_op, "last_total_cap_violation", None),
            "leakage": None if timing_op is None else getattr(timing_op, "last_total_leakage", None),
        }
        weights = {
            "timing": float(getattr(self, "timing_grad_balance_weight", 1.0)),
            "slew": float(
                getattr(
                    self,
                    "timing_slew_weight",
                    getattr(self.params, "timing_slew_weight", 1.0),
                )
            ),
            "cap": float(
                getattr(
                    self,
                    "timing_cap_weight",
                    getattr(self.params, "timing_cap_weight", 1.0),
                )
            ),
            "leakage": float(
                getattr(
                    self,
                    "timing_leakage_weight",
                    getattr(self.params, "timing_leakage_weight", 0.0),
                )
            ),
        }
        weighted_terms = {}
        total_loss = zero
        detached_zero_terms = []
        for term_name, raw_term in raw_terms.items():
            if raw_term is None:
                weighted_terms[term_name] = zero
                continue
            if self._should_detach_zero_timing_aux_term(
                term_name,
                raw_term,
                scalar_value=scalar_terms.get(term_name),
            ):
                raw_term = raw_term.detach()
                detached_zero_terms.append(term_name)
            if term_name == "timing" or term_name in enabled_terms:
                weighted_terms[term_name] = raw_term * float(weights[term_name])
                total_loss = total_loss + weighted_terms[term_name]
            else:
                weighted_terms[term_name] = raw_term * 0.0

        self.last_timing_objective_terms = {
            "lane": self._timing_objective_lane(),
            "enabled_terms": list(enabled_terms),
            "units": {
                "timing": "loss_proxy",
                "slew": "ns",
                "cap": "pF",
                "leakage": "mW",
            },
            "weights": {name: float(value) for name, value in weights.items()},
            "raw_terms": {
                name: None if raw_terms[name] is None else float(raw_terms[name].detach().item())
                for name in ("timing", "slew", "cap", "leakage")
            },
            "weighted_terms": {
                name: float(weighted_terms[name].detach().item())
                for name in ("timing", "slew", "cap", "leakage")
            },
            "critical_endpoint_pruning": pruning_metadata,
            "detached_zero_terms": detached_zero_terms,
        }
        self.last_timing_objective_terms["report_aligned_terms"] = {
            "slew_violation_ns": self.last_timing_objective_terms["raw_terms"]["slew"],
            "cap_violation_fF": (
                None
                if self.last_timing_objective_terms["raw_terms"]["cap"] is None
                else self.last_timing_objective_terms["raw_terms"]["cap"] * 1000.0
            ),
            "leakage_mW": self.last_timing_objective_terms["raw_terms"]["leakage"],
        }
        self.last_timing_objective_terms["total_loss"] = float(total_loss.detach().item())
        self.last_timing_objective_terms["grad_balance"] = copy.deepcopy(
            getattr(self, "timing_grad_balance_summary", {})
        )
        self._last_timing_objective_component_tensors = dict(weighted_terms)
        return total_loss

    def _timing_loss(self, wns, tns, ws, ts):
        # Positive WS should not be rewarded in size-only optimization; otherwise
        # already-nonviolating directions inject extra timing gradient.
        return self._timing_objective_term_components(wns, tns)

    @staticmethod
    def _timing_grad_balance_from_norms(
        *,
        wirelength_grad_norm,
        timing_grad_norm,
        target_ratio,
        min_weight=1.0e-6,
        max_weight=1.0e6,
        eps=1.0e-12,
    ):
        wirelength_grad_norm = float(wirelength_grad_norm)
        timing_grad_norm = float(timing_grad_norm)
        target_ratio = float(target_ratio)
        result = {
            "artifact": "timing_grad_balance",
            "artifact_version": 1,
            "initialized": False,
            "target_ratio": target_ratio,
            "norm": "l1",
            "wirelength_grad_norm": wirelength_grad_norm,
            "timing_grad_norm": timing_grad_norm,
            "timing_to_wirelength_grad_ratio": None,
            "weight_raw": None,
            "weight_applied": 1.0,
            "weight_clamped": False,
            "weighted_timing_to_wirelength_grad_ratio": None,
        }
        if target_ratio <= 0.0:
            result.update(status="disabled", initialized=True)
            return result
        if not math.isfinite(wirelength_grad_norm) or not math.isfinite(
            timing_grad_norm
        ):
            result["status"] = "deferred_nonfinite_grad"
            return result
        if wirelength_grad_norm <= eps:
            result["status"] = "deferred_zero_wirelength_grad"
            return result
        result["timing_to_wirelength_grad_ratio"] = (
            timing_grad_norm / wirelength_grad_norm
        )
        if timing_grad_norm <= eps:
            result["status"] = "deferred_zero_timing_grad"
            return result
        weight_raw = target_ratio * wirelength_grad_norm / timing_grad_norm
        weight_applied = min(max(weight_raw, float(min_weight)), float(max_weight))
        result.update(
            status="initialized",
            initialized=True,
            weight_raw=weight_raw,
            weight_applied=weight_applied,
            weight_clamped=not math.isclose(
                weight_raw,
                weight_applied,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ),
            weighted_timing_to_wirelength_grad_ratio=(
                weight_applied * timing_grad_norm / wirelength_grad_norm
            ),
        )
        return result

    def apply_timing_grad_balance_state(self, state):
        state = copy.deepcopy(dict(state or {}))
        if state.get("initialized"):
            self.timing_grad_balance_weight = float(
                state.get("weight_applied", 1.0)
            )
        self.timing_grad_balance_summary = state
        return copy.deepcopy(state)

    def initialize_timing_grad_balance(self, pos, *, iteration, overflow):
        if bool(self.timing_grad_balance_summary.get("initialized")):
            return copy.deepcopy(self.timing_grad_balance_summary)

        target_ratio = float(self.timing_grad_balance_target_ratio)
        if target_ratio <= 0.0:
            result = self._timing_grad_balance_from_norms(
                wirelength_grad_norm=0.0,
                timing_grad_norm=0.0,
                target_ratio=target_ratio,
            )
        else:
            wirelength = self.op_collections.wirelength_op(pos)
            wirelength_grad = torch.autograd.grad(
                wirelength,
                pos,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            try:
                wns, tns, ws, ts = self.timing_obj(pos)
                timing_loss = self._timing_loss(wns, tns, ws, ts)
                timing_grad = torch.autograd.grad(
                    timing_loss,
                    pos,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            finally:
                self._timing_geometry_cache = None
            wirelength_grad_norm = (
                0.0
                if wirelength_grad is None
                else float(wirelength_grad.detach().float().norm(p=1).item())
            )
            timing_grad_norm = (
                0.0
                if timing_grad is None
                else float(timing_grad.detach().float().norm(p=1).item())
            )
            result = self._timing_grad_balance_from_norms(
                wirelength_grad_norm=wirelength_grad_norm,
                timing_grad_norm=timing_grad_norm,
                target_ratio=target_ratio,
            )

        result.update(
            iteration=int(iteration),
            overflow=float(torch.as_tensor(overflow).detach().cpu().item()),
            timing_wns_coeff=float(self.timing_wns_coeff),
            timing_tns_coeff=float(self.timing_tns_coeff),
        )
        self.apply_timing_grad_balance_state(result)
        logging.info(
            "Timing gradient balance status=%s iteration=%d overflow=%.6f "
            "wl_grad_l1=%s timing_grad_l1=%s target_ratio=%.6g weight=%.6g",
            result["status"],
            int(iteration),
            result["overflow"],
            result.get("wirelength_grad_norm"),
            result.get("timing_grad_norm"),
            target_ratio,
            float(result.get("weight_applied", 1.0)),
        )
        return copy.deepcopy(result)

    def _write_critical_endpoint_pruning_artifact(self, iteration):
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        builder = None if timing_op is None else getattr(timing_op, "build_critical_endpoint_pruning_artifact", None)
        if not callable(builder):
            return None
        payload = builder(iteration)
        if payload is None:
            return None
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        design_name_attr = getattr(self.params, "design_name", None)
        design_name = design_name_attr() if callable(design_name_attr) else "design"
        artifact_path = os.path.join(
            result_dir,
            f"{design_name}_critical_endpoint_pruning_latest.json",
        )
        os.makedirs(result_dir, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return artifact_path

    def _write_timing_propagation_profile_artifact(self):
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        builder = None if timing_op is None else getattr(timing_op, "build_timing_propagation_profile_artifact", None)
        if not callable(builder):
            write_crash_stage_marker(
                "timing_propagation_profile_artifact",
                "skipped",
                reason="builder_missing",
            )
            return None
        write_crash_stage_marker("timing_propagation_profile_artifact", "start")
        payload = builder()
        if payload is None:
            write_crash_stage_marker(
                "timing_propagation_profile_artifact",
                "skipped",
                reason="payload_missing",
            )
            return None
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            write_crash_stage_marker(
                "timing_propagation_profile_artifact",
                "skipped",
                reason="result_dir_missing",
            )
            return None
        design_name_attr = getattr(self.params, "design_name", None)
        design_name = design_name_attr() if callable(design_name_attr) else "design"
        artifact_path = os.path.join(
            result_dir,
            f"{design_name}_timing_propagation_profile_latest.json",
        )
        os.makedirs(result_dir, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        write_crash_stage_marker(
            "timing_propagation_profile_artifact",
            "done",
            artifact_path=artifact_path,
        )
        return artifact_path

    def _write_timing_propagation_parity_artifact(self):
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        builder = None if timing_op is None else getattr(timing_op, "build_timing_propagation_parity_artifact", None)
        if not callable(builder):
            write_crash_stage_marker(
                "timing_propagation_parity_artifact",
                "skipped",
                reason="builder_missing",
            )
            return None
        write_crash_stage_marker("timing_propagation_parity_artifact", "start")
        payload = builder()
        if payload is None:
            write_crash_stage_marker(
                "timing_propagation_parity_artifact",
                "skipped",
                reason="payload_missing",
            )
            return None
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            write_crash_stage_marker(
                "timing_propagation_parity_artifact",
                "skipped",
                reason="result_dir_missing",
            )
            return None
        design_name_attr = getattr(self.params, "design_name", None)
        design_name = design_name_attr() if callable(design_name_attr) else "design"
        artifact_path = os.path.join(
            result_dir,
            f"{design_name}_timing_propagation_parity_latest.json",
        )
        os.makedirs(result_dir, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        write_crash_stage_marker(
            "timing_propagation_parity_artifact",
            "done",
            artifact_path=artifact_path,
        )
        return artifact_path

    def _production_fast_loop_enabled(self):
        return bool(getattr(getattr(self, "params", None), "production_fast_loop", False))

    def _interval_value(self, name, default):
        try:
            return int(getattr(self.params, name, default))
        except (TypeError, ValueError):
            return int(default)

    def _interval_should_write(self, interval, iteration):
        try:
            interval = int(interval)
            iteration = int(iteration)
        except (TypeError, ValueError):
            return False
        return interval > 0 and iteration % interval == 0

    def _timing_obj_profile_enabled(self):
        return bool(getattr(self.params, "timing_obj_profile", False))

    def _timing_obj_profile_interval(self):
        return max(
            1,
            self._interval_value("timing_obj_profile_interval", 1),
        )

    def _should_write_timing_obj_profile(self, iteration):
        if not self._timing_obj_profile_enabled():
            return False
        return self._interval_should_write(
            self._timing_obj_profile_interval(),
            iteration,
        )

    def _should_write_timing_artifacts(self, iteration):
        default_interval = 0 if self._production_fast_loop_enabled() else 1
        interval = self._interval_value(
            "timing_artifact_write_interval",
            default_interval,
        )
        return self._interval_should_write(interval, iteration)

    def _should_write_pin_violation_detail(self, iteration):
        default_interval = 0 if self._production_fast_loop_enabled() else 1
        interval = self._interval_value(
            "pin_violation_detail_interval",
            default_interval,
        )
        return self._interval_should_write(interval, iteration)

    def _should_copy_pin_slack_after_timing(self, timing_artifact_enabled):
        params = getattr(self, "params", None)
        flow_kind = getattr(params, "flow_kind", None)
        flow_value = getattr(flow_kind, "value", flow_kind)
        mode = str(getattr(params, "buffering_mode", "") or "")
        data_collections = getattr(self, "data_collections", None)
        needs_buffer_projection_slack = (
            str(flow_value) in ("buffering", "joint")
            and mode == "segment"
            and getattr(data_collections, "buffer_segment_count_state", None) is not None
        )
        return (
            not self._production_fast_loop_enabled()
            or bool(timing_artifact_enabled)
            or bool(needs_buffer_projection_slack)
        )

    def _should_refresh_debug_pin_caps_base(self, pin_caps_base, timing_artifact_enabled):
        device = getattr(pin_caps_base, "device", None)
        if device is not None and getattr(device, "type", None) == "cpu":
            return True
        if not self._production_fast_loop_enabled():
            return True
        if bool(timing_artifact_enabled):
            return True

        params = getattr(self, "params", None)
        return bool(
            getattr(params, "timing_propagation_parity_check", False)
            or getattr(params, "openroad_aimp_debug_artifacts", False)
        )

    def _cap_limit_interpolation_pin_mask(self, aux_cap_pin_mask, output_pin_mask):
        if aux_cap_pin_mask is not None:
            return aux_cap_pin_mask
        limit_flag = os.environ.get("AIMP_LIMIT_CAP_LIMIT_TO_OUTPUT_PINS", "")
        if (
            self._production_fast_loop_enabled()
            and str(limit_flag).strip().lower() in _AIMP_TRUE_VALUES
        ):
            return output_pin_mask
        return None

    def _zero_slew_reuse_cap_limit_interpolation_pin_mask(
        self,
        aux_cap_pin_mask,
        output_pin_mask,
    ):
        if aux_cap_pin_mask is not None:
            return aux_cap_pin_mask
        return None

    def _timing_obj_profile_paths(self):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None, None
        design_name_attr = getattr(self.params, "design_name", None)
        design_name = design_name_attr() if callable(design_name_attr) else "design"
        latest_path = os.path.join(
            result_dir,
            f"{design_name}_timing_obj_profile_latest.json",
        )
        trace_path = os.path.join(
            result_dir,
            f"{design_name}_timing_obj_profile.jsonl",
        )
        return latest_path, trace_path

    def _write_timing_obj_profile_record(self, record):
        if not self._timing_obj_profile_enabled():
            return None
        latest_path, trace_path = self._timing_obj_profile_paths()
        if not latest_path or not trace_path:
            return None
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        payload = dict(record)
        payload["artifact"] = "timing_obj_profile_latest"
        payload["artifact_version"] = 1
        payload["timestamp"] = time.time()
        with open(latest_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        trace_payload = dict(payload)
        trace_payload["artifact"] = "timing_obj_profile_record"
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
        return latest_path

    def _timing_pruning_profile_fields(self, timing_op):
        critical_stats = getattr(
            timing_op,
            "last_critical_endpoint_pruning_stats",
            {},
        )
        traversal_stats = getattr(
            timing_op,
            "last_traversal_pruning_stats",
            {},
        )
        if not isinstance(critical_stats, dict):
            critical_stats = {}
        if not isinstance(traversal_stats, dict):
            traversal_stats = {}
        propagation_profile = getattr(timing_op, "last_profile_payload", {})
        if not isinstance(propagation_profile, dict):
            propagation_profile = {}
        propagation_stage_runtime = propagation_profile.get("stage_runtime_ms", {})
        if not isinstance(propagation_stage_runtime, dict):
            propagation_stage_runtime = {}
        cell_aat_detail = propagation_profile.get("cell_aat_levels_detail", {})
        if not isinstance(cell_aat_detail, dict):
            cell_aat_detail = {}
        dynamic_provider_profile = propagation_profile.get(
            "dynamic_provider_profile",
            None,
        )
        if not isinstance(dynamic_provider_profile, dict):
            dynamic_provider_profile = None

        def _int_field(name, default=0):
            try:
                return int(traversal_stats.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        def _float_or_none(name):
            value = traversal_stats.get(name)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return {
            "critical_endpoint_pruning": dict(critical_stats),
            "traversal_pruning": dict(traversal_stats),
            "active_endpoint_count": _int_field(
                "active_endpoint_count",
                critical_stats.get("active_endpoint_count", 0),
            ),
            "active_inst_count": _int_field("active_inst_count"),
            "active_arc_count": _int_field("active_arc_count"),
            "dropped_arc_count": _int_field("dropped_arc_count"),
            "parallel_task_count": _int_field("parallel_task_count"),
            "traversal_pruning_preparation_runtime_ms": _float_or_none(
                "preparation_runtime_ms"
            ),
            "timing_propagation_stage_runtime_ms": dict(propagation_stage_runtime),
            "timing_propagation_cell_aat_detail": dict(cell_aat_detail),
            "dynamic_provider_profile": (
                copy.deepcopy(dynamic_provider_profile)
                if dynamic_provider_profile is not None
                else None
            ),
        }

    def _is_size_only_mode(self):
        return getattr(self.params, "placement_sizing_mode", "place_only") == "size_only"

    def _use_targeted_sizing_backward(self):
        flag = getattr(self.params, "targeted_sizing_backward", False)
        env_flag = os.environ.get("AIMP_TARGETED_SIZING_BACKWARD", "")
        env_enabled = str(env_flag).strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        return (
            self._is_size_only_mode()
            and self._production_fast_loop_enabled()
            and (bool(flag) or env_enabled)
        )

    def _sizing_autograd_params(self):
        params = []
        size_parameter_getter = getattr(
            self.data_collections,
            "get_continuous_size_parameter",
            None,
        )
        size_parameter = (
            size_parameter_getter()
            if callable(size_parameter_getter)
            else getattr(self.data_collections, "size_logits", None)
        )
        for param in (
            size_parameter,
            getattr(self.data_collections, "vt_logits", None),
        ):
            if param is not None and torch.is_tensor(param) and param.requires_grad:
                params.append(param)
        return params

    def _buffer_autograd_params(self):
        state = getattr(self.data_collections, "buffer_optimization_state", None)
        if state is None:
            state = getattr(self.data_collections, "buffer_segment_count_state", None)
        if state is None:
            return []
        params = []
        seen = set()
        for attr_name in (
            "z_param",
            "bu_logits",
            "activation_param",
            "bsu_index_param",
        ):
            param = getattr(state, attr_name, None)
            if (
                param is not None
                and torch.is_tensor(param)
                and param.requires_grad
                and id(param) not in seen
            ):
                params.append(param)
                seen.add(id(param))
        return params

    def _targeted_sizing_backward(self, obj):
        sizing_params = self._sizing_autograd_params()
        if not sizing_params or not torch.is_tensor(obj) or not obj.requires_grad:
            return 0
        grads = autograd.grad(
            obj,
            sizing_params,
            allow_unused=True,
        )
        assigned = 0
        for param, grad in zip(sizing_params, grads):
            if grad is None:
                param.grad = None
                continue
            param.grad = grad.detach()
            assigned += 1
        return assigned

    def _size_debug_component_grad_stats_enabled(self):
        flag = os.environ.get("AIMP_SIZE_DEBUG_TRACE", "")
        return str(flag).strip().lower() not in ("", "0", "false", "off", "no")

    def _timing_backward_component_profile_enabled(self):
        flag = os.environ.get("AIMP_TIMING_BACKWARD_PROFILE", "")
        return str(flag).strip().lower() not in ("", "0", "false", "off", "no")

    def _sync_autograd_profile_clock(self, tensor):
        if torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        return time.perf_counter()

    def _timing_backward_component_profile(self):
        size_parameter_getter = getattr(
            self.data_collections,
            "get_continuous_size_parameter",
            None,
        )
        size_logits = (
            size_parameter_getter()
            if callable(size_parameter_getter)
            else getattr(self.data_collections, "size_logits", None)
        )
        component_tensors = getattr(
            self,
            "_last_timing_objective_component_tensors",
            None,
        )
        if size_logits is None or not component_tensors:
            return {}
        profile = {}
        for component_name, component in component_tensors.items():
            if component is None or not torch.is_tensor(component):
                continue
            if not component.requires_grad:
                continue
            started_at = self._sync_autograd_profile_clock(component)
            grad = autograd.grad(
                component,
                size_logits,
                retain_graph=True,
                allow_unused=True,
            )[0]
            elapsed_ms = (
                self._sync_autograd_profile_clock(component) - started_at
            ) * 1000.0
            if grad is None:
                profile[component_name] = {
                    "grad_ms": elapsed_ms,
                    "grad_norm": None,
                    "grad_max_abs": None,
                    "grad_nonzero_count": 0,
                }
                continue
            grad_detached = grad.detach()
            profile[component_name] = {
                "grad_ms": elapsed_ms,
                "grad_norm": float(torch.linalg.vector_norm(grad_detached).item()),
                "grad_max_abs": float(torch.max(torch.abs(grad_detached)).item()),
                "grad_nonzero_count": int(torch.count_nonzero(grad_detached).item()),
            }
        return profile

    def _size_debug_grad_stats_for_component(self, component):
        size_parameter_getter = getattr(
            self.data_collections,
            "get_continuous_size_parameter",
            None,
        )
        size_logits = (
            size_parameter_getter()
            if callable(size_parameter_getter)
            else getattr(self.data_collections, "size_logits", None)
        )
        if size_logits is None or component is None:
            return {
                "norm": None,
                "max_abs": None,
                "mean": None,
                "positive_frac": None,
                "negative_frac": None,
            }
        grad = autograd.grad(
            component,
            size_logits,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return {
                "norm": None,
                "max_abs": None,
                "mean": None,
                "positive_frac": None,
                "negative_frac": None,
            }
        grad = grad.detach()
        if grad.numel() == 0:
            return {
                "norm": 0.0,
                "max_abs": 0.0,
                "mean": 0.0,
                "positive_frac": 0.0,
                "negative_frac": 0.0,
            }
        positive_frac = float((grad > 0).float().mean().item())
        negative_frac = float((grad < 0).float().mean().item())
        return {
            "norm": float(grad.norm().item()),
            "max_abs": float(grad.abs().max().item()),
            "mean": float(grad.mean().item()),
            "positive_frac": positive_frac,
            "negative_frac": negative_frac,
        }

    def _continuous_density_area_penalty(self):
        mode = getattr(self.params, "placement_sizing_mode", "place_only")
        if mode not in ("size_only", "joint"):
            return None
        area_weight = float(getattr(self, "size_density_area_weight", 0.1))
        if area_weight <= 0.0:
            return None

        size_var_getter = getattr(self.data_collections, "get_size_var", None)
        if size_var_getter is None:
            return None
        size_var = size_var_getter()
        inst_size_init = getattr(self.data_collections, "inst_size_init", None)
        node_areas = getattr(self.data_collections, "node_areas", None)
        inst_is_sizeable = getattr(self.data_collections, "inst_is_sizeable", None)
        if (
            size_var is None
            or inst_size_init is None
            or node_areas is None
            or inst_is_sizeable is None
        ):
            return None

        num_movable_nodes = min(
            int(getattr(self.placedb, "num_movable_nodes", 0)),
            int(size_var.numel()),
            int(inst_size_init.numel()),
            int(node_areas.numel()),
            int(inst_is_sizeable.numel()),
        )
        if num_movable_nodes <= 0:
            return None

        movable_mask = inst_is_sizeable[:num_movable_nodes].bool()
        if not torch.any(movable_mask):
            return None

        base_sizes = inst_size_init[:num_movable_nodes].float().clamp_min(1e-6)
        base_areas = node_areas[:num_movable_nodes].float()
        area_scale = size_var[:num_movable_nodes].float() / base_sizes
        movable_base_area = base_areas[movable_mask].sum().clamp_min(1.0)
        movable_projected_area = torch.sum(base_areas[movable_mask] * area_scale[movable_mask])
        normalized_area_delta = (
            movable_projected_area - movable_base_area
        ) / movable_base_area

        density_signal = self.density.float()
        if density_signal.numel() > 1:
            density_signal = density_signal.mean()
        if self.init_density is not None:
            init_density = self.init_density.float()
            if init_density.numel() > 1:
                init_density = init_density.mean()
            density_signal = density_signal / init_density.clamp_min(1e-6)

        return area_weight * density_signal * normalized_area_delta


    def reset_l_shape_weight_state(self):
        self.l_shape_routability_weight.data.fill_(self.l_shape_routability_weight_init)
        self._l_shape_weight_initialized = False
        self.l_shape_last_weight = None
        self.l_shape_last_target_weight = None
        self.l_shape_last_weight_candidate = None
        self.l_shape_last_cap_active = None
        self.l_shape_last_base_grad_norm = None
        self.l_shape_last_grad_raw_norm = None
        self.l_shape_last_grad_norm = None
        self.l_shape_last_grad_ratio = None
        self.l_shape_capacity_al_last_summary = {}
        self.l_shape_macro_exclusion_last_summary = {}

    def collect_l_shape_telemetry(self, metric):
        from dreamplace.ops.routability.l_shape_telemetry import collect_l_shape_telemetry

        collect_l_shape_telemetry(self, metric)

    def snapshot_l_shape_forward_state(self):
        """Capture diagnostic forward maps before a non-objective query."""
        from dreamplace.ops.routability.l_shape_electric_potential import (
            SegmentElectricPotentialFunction,
        )

        return (
            SegmentElectricPotentialFunction.last_rho_map,
            SegmentElectricPotentialFunction.last_rho_map_h,
            SegmentElectricPotentialFunction.last_rho_map_v,
        )

    def restore_l_shape_forward_state(self, snapshot):
        """Restore diagnostic forward maps after a non-objective query."""
        from dreamplace.ops.routability.l_shape_electric_potential import (
            SegmentElectricPotentialFunction,
        )

        (
            SegmentElectricPotentialFunction.last_rho_map,
            SegmentElectricPotentialFunction.last_rho_map_h,
            SegmentElectricPotentialFunction.last_rho_map_v,
        ) = snapshot

    def refresh_routability_operators(self):
        """Rebuild the operators whose bins follow the active routing grid."""
        self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
            self.params, self.placedb, self.data_collections
        )
        self.op_collections.adjust_node_area_op = self.build_adjust_node_area(
            self.params, self.placedb, self.data_collections
        )

    def refresh_after_geometry_change(self):
        """Invalidate density and pin caches after area or geometry changes."""
        self.op_collections.density_op.reset()
        self.op_collections.density_overflow_op.reset()
        self.op_collections.pin_utilization_map_op.reset()
        self._timing_geometry_cache = None
        if self._virtual_cell_density_view is not None:
            self._virtual_cell_density_view._density_ops.clear()

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

    def set_l_shape_outer_iteration(self, iteration):
        if iteration is None:
            self._l_shape_outer_iteration = None
        else:
            self._l_shape_outer_iteration = int(iteration)
        if self.l_shape_routability_op is not None and hasattr(
            self.l_shape_routability_op, "set_debug_iteration"
        ):
            self.l_shape_routability_op.set_debug_iteration(self._l_shape_outer_iteration)

    @staticmethod
    def _compute_l_shape_target_weight(
        base_grad_norm_value, l_shape_grad_norm_value, target_ratio
    ):
        if base_grad_norm_value <= 1e-10 or l_shape_grad_norm_value <= 1e-10:
            return None
        return (
            float(target_ratio)
            * float(base_grad_norm_value)
            / (float(l_shape_grad_norm_value) + 1e-12)
        )

    def start_l_shape_weight_controller(self, iteration):
        self.set_l_shape_outer_iteration(iteration)
        self._l_shape_weight_initialized = False
        self.l_shape_routability_weight.data.fill_(self.l_shape_routability_weight_init)
        self.l_shape_last_weight_candidate = None

    def init_l_shape_routability(
        self,
        wire_width,
        num_bins_x,
        num_bins_y,
        target_density=1.0,
        target_demand=None,
        raw_wire_demand_map=None,
        supply_original=None,
        target_density_h=None,
        target_density_v=None,
        target_demand_h=None,
        target_demand_v=None,
        raw_wire_demand_map_h=None,
        raw_wire_demand_map_v=None,
        supply_original_h=None,
        supply_original_v=None,
        fix_usage_map=None,
        fix_usage_map_h=None,
        fix_usage_map_v=None,
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
            with profile_scope(
                self.params,
                "place_obj.init_l_shape.update_existing_targets",
                logger=logging,
            ):
                self.l_shape_routability_op.update_targets(
                    target_density=target_density,
                    target_demand=target_demand,
                    raw_wire_demand_map=raw_wire_demand_map,
                    supply_original=supply_original,
                    target_density_h=target_density_h,
                    target_density_v=target_density_v,
                    target_demand_h=target_demand_h,
                    target_demand_v=target_demand_v,
                    raw_wire_demand_map_h=raw_wire_demand_map_h,
                    raw_wire_demand_map_v=raw_wire_demand_map_v,
                    supply_original_h=supply_original_h,
                    supply_original_v=supply_original_v,
                    fix_usage_map=fix_usage_map,
                    fix_usage_map_h=fix_usage_map_h,
                    fix_usage_map_v=fix_usage_map_v,
            )
            self.use_l_shape_routability = True
            if l_shape_log_verbose(self.params) >= 1:
                logging.info("L-shape routability already initialized; refreshed targets and re-enabled")
            return
        
        with profile_scope(
            self.params,
            "place_obj.init_l_shape.construct_routability_op",
            logger=logging,
            bins=f"{num_bins_x}x{num_bins_y}",
        ):
            self.l_shape_routability_op = LShapeRoutabilityOp(
                placedb=self.placedb,
                params=self.params,
                wire_width=wire_width,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                target_density=target_density,
                target_demand=target_demand,
                raw_wire_demand_map=raw_wire_demand_map,
                supply_original=supply_original,
                target_density_h=target_density_h,
                target_density_v=target_density_v,
                target_demand_h=target_demand_h,
                target_demand_v=target_demand_v,
                raw_wire_demand_map_h=raw_wire_demand_map_h,
                raw_wire_demand_map_v=raw_wire_demand_map_v,
                supply_original_h=supply_original_h,
                supply_original_v=supply_original_v,
                fix_usage_map=fix_usage_map,
                fix_usage_map_h=fix_usage_map_h,
                fix_usage_map_v=fix_usage_map_v,
            )
        self.use_l_shape_routability = True
        
        if l_shape_log_verbose(self.params) >= 1:
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
    
    def l_shape_routability_obj(
        self,
        pos,
        use_l_direction=True,
        update_capacity_al_lambda=False,
        placement_iteration_id=None,
    ):
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
        if hasattr(self.l_shape_routability_op, "set_debug_iteration"):
            self.l_shape_routability_op.set_debug_iteration(self._l_shape_outer_iteration)
        
        with profile_scope(
            self.params,
            "place_obj.l_shape_routability_obj",
            tensor=pos,
            logger=logging,
            iteration=self._l_shape_outer_iteration,
        ):
            return self.l_shape_routability_op(
                pos,
                self.op_collections.steiner_topo_op,
                self.op_collections.pin_pos_op,
                use_l_direction=use_l_direction,
                update_capacity_al_lambda=update_capacity_al_lambda,
                placement_iteration_id=placement_iteration_id,
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
        objective_pos = pos.detach() if self._is_size_only_mode() else pos
        skip_placement_objective_ops = (
            self._is_size_only_mode()
            and self._production_fast_loop_enabled()
            and self.init_density is not None
        )
        if skip_placement_objective_ops:
            if not hasattr(self, "wirelength"):
                self.wirelength = torch.zeros((), dtype=pos.dtype, device=pos.device)
            if not hasattr(self, "density"):
                self.density = self.init_density.to(device=pos.device, dtype=pos.dtype)
        else:
            self.wirelength = self.op_collections.wirelength_op(objective_pos)
            self.density = self.placement_density(objective_pos)

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
        if self._is_size_only_mode():
            # In size_only mode we keep placement observability (WL/density snapshots) but
            # optimize only sizing-related terms so the objective does not backprop into pos.
            result = torch.zeros((), dtype=pos.dtype, device=pos.device)
        elif len(self.placedb.regions) > 0:
            result = self.wirelength + self.density_weight.dot(self.density)
        else:
            result = torch.add(
                self.wirelength,
                self.density,
                alpha=(self.density_factor * self.density_weight).item(),
            )
        self.size_density_area_penalty = self._continuous_density_area_penalty()
        if self.size_density_area_penalty is not None:
            result = torch.add(result, self.size_density_area_penalty)

        if (not self._is_size_only_mode()) and self.params.macro_overlap_flag:
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

            wns, tns, ws, ts = self.timing_obj(objective_pos)
            slack = self._timing_loss(wns, tns, ws, ts)
            self.wns = wns
            self.tns = tns
            self.ws = ws
            self.ts = ts
            # logging.info(f"Timing slack: {slack}")
            result = torch.add(result, slack)

        if (
            (not self._is_size_only_mode())
            and self.params.enable_net_weighting
            and self.params.net_weighting_scheme == "pin2pin"
        ):
            self.pin2pin_net_weight = self.op_collections.pin2pin_net_weight_op(pos)
            
            # if self.pin2pin_net_weight > 0:
            #     self.pin2pin_weight += 0.001
            result = torch.add(result, self.pin2pin_net_weight, alpha=self.pin2pin_weight)

        return result

    def pin_2_libpin_ids(self, inst_libcell_offset: torch.tensor, data_collections):
        nodes_id = data_collections.pin2node_map
        pins_main_id = data_collections.inst_main_id[nodes_id]
        inst_pins_mask = pins_main_id >= 0
        pin_offsets = data_collections.pin_2_libpin_offset
        if hasattr(pin_offsets, "to"):
            pin_offsets = pin_offsets.to(device=nodes_id.device)
        else:
            pin_offsets = torch.as_tensor(pin_offsets, device=nodes_id.device)
        inst_pins_mask = inst_pins_mask & (pin_offsets >= 0)
        if not torch.any(inst_pins_mask):
            empty_libpin_ids = torch.empty(
                0,
                dtype=data_collections.cell_id_2_libpin_id_start.dtype,
                device=data_collections.cell_id_2_libpin_id_start.device,
            )
            return inst_pins_mask, empty_libpin_ids

        pins_cell_id = (
            data_collections.main_id_2_cell_id_start[pins_main_id[inst_pins_mask]]
            + inst_libcell_offset[nodes_id[inst_pins_mask]].long()
        )
        libpin_ids = (
            data_collections.cell_id_2_libpin_id_start[pins_cell_id]
            + pin_offsets[inst_pins_mask]
        )
        return inst_pins_mask, libpin_ids

    def _pin_caps_from_offsets(self, inst_libcell_offset, data_collections):
        pin2libpin_flat_ids = torch.zeros(
            data_collections.pin2node_map.size()[0],
            dtype=data_collections.cell_id_2_libpin_id_start.dtype,
            device=data_collections.cell_id_2_libpin_id_start.device,
        )
        pin_cap_base = torch.zeros(
            data_collections.pin2node_map.size()[0],
            dtype=data_collections.flat_lib_pin_cap.dtype,
            device=data_collections.flat_lib_pin_cap.device,
        )
        pin_rcap_base = torch.zeros_like(pin_cap_base)
        pin_fcap_base = torch.zeros_like(pin_cap_base)
        inst_pins_mask, inst_pin2libpin_flat_ids = self.pin_2_libpin_ids(
            inst_libcell_offset, data_collections
        )
        inst_pin_cap_base = data_collections.flat_lib_pin_cap[inst_pin2libpin_flat_ids]
        inst_pin_rcap_base = data_collections.flat_lib_pin_rcap[inst_pin2libpin_flat_ids]
        inst_pin_fcap_base = data_collections.flat_lib_pin_fcap[inst_pin2libpin_flat_ids]
        pin2libpin_flat_ids[inst_pins_mask] = inst_pin2libpin_flat_ids
        pin_cap_base[inst_pins_mask] = inst_pin_cap_base
        pin_rcap_base[inst_pins_mask] = inst_pin_rcap_base
        pin_fcap_base[inst_pins_mask] = inst_pin_fcap_base
        return pin2libpin_flat_ids, inst_pins_mask, pin_cap_base, pin_rcap_base, pin_fcap_base

    def _current_timing_surrogate_mode(self):
        mode = getattr(self, "_active_timing_surrogate_mode", None)
        if mode is not None:
            return mode
        params = getattr(self, "params", None)
        if params is None:
            return None
        return getattr(params, "timing_surrogate_mode", None)

    def _sizing_pin_cap_static_cache_key(self, data_collections):
        names = (
            "pin2node_map",
            "inst_main_id",
            "inst_is_sizeable",
            "pin_2_libpin_offset",
            "main_id_2_cell_id_start",
            "cell_id_2_libpin_id_start",
            "flat_libcell_info",
            "flat_lib_pin_cap",
            "flat_lib_pin_rcap",
            "flat_lib_pin_fcap",
        )
        key = []
        for name in names:
            value = getattr(data_collections, name, None)
            if not isinstance(value, torch.Tensor):
                return None
            key.append(
                (
                    name,
                    tuple(value.shape),
                    value.device.type,
                    value.device.index,
                    value.data_ptr(),
                )
            )
        return tuple(key)

    def _get_sizing_pin_cap_static_cache(self, data_collections):
        cache_key = self._sizing_pin_cap_static_cache_key(data_collections)
        if cache_key is None:
            return None
        existing = getattr(data_collections, "_sizing_pin_cap_static_cache", None)
        if isinstance(existing, dict) and existing.get("key") == cache_key:
            return existing

        nodes_id = data_collections.pin2node_map
        pin_offsets_all = data_collections.pin_2_libpin_offset
        pin_main_ids_all = data_collections.inst_main_id[nodes_id]
        sizeable_mask = (
            (pin_main_ids_all >= 0)
            & data_collections.inst_is_sizeable[nodes_id]
            & (pin_offsets_all >= 0)
        )
        if not torch.any(sizeable_mask):
            cache = {
                "key": cache_key,
                "has_sizeable": False,
                "sizeable_mask": sizeable_mask,
                "groups": [],
            }
            setattr(data_collections, "_sizing_pin_cap_static_cache", cache)
            return cache

        cell_sizes = data_collections.flat_libcell_info[:, 2].float()
        cell_vts = data_collections.flat_libcell_info[:, 3].long()
        pin_nodes = nodes_id[sizeable_mask]
        pin_main_ids = pin_main_ids_all[sizeable_mask].long()
        pin_offsets = pin_offsets_all[sizeable_mask].long()
        sizeable_pin_ids = torch.nonzero(sizeable_mask, as_tuple=False).flatten()
        groups = []
        for main_id in torch.unique(pin_main_ids):
            main_mask = pin_main_ids == main_id
            if not torch.any(main_mask):
                continue
            cell_start = data_collections.main_id_2_cell_id_start[main_id].long()
            cell_end = data_collections.main_id_2_cell_id_start[main_id + 1].long()
            candidate_cell_ids = torch.arange(
                cell_start,
                cell_end,
                device=nodes_id.device,
                dtype=torch.long,
            )
            if candidate_cell_ids.numel() == 0:
                continue
            local_pin_offsets = pin_offsets[main_mask]
            candidate_libpin_ids = (
                data_collections.cell_id_2_libpin_id_start[candidate_cell_ids]
                .long()
                .unsqueeze(0)
                + local_pin_offsets.unsqueeze(1)
            )
            groups.append(
                {
                    "main_mask": main_mask,
                    "candidate_cell_ids": candidate_cell_ids,
                    "candidate_sizes": cell_sizes[candidate_cell_ids],
                    "candidate_vts": cell_vts[candidate_cell_ids],
                    "candidate_libpin_ids": candidate_libpin_ids,
                    "candidate_caps": data_collections.flat_lib_pin_cap[candidate_libpin_ids],
                    "candidate_rcaps": data_collections.flat_lib_pin_rcap[candidate_libpin_ids],
                    "candidate_fcaps": data_collections.flat_lib_pin_fcap[candidate_libpin_ids],
                }
            )

        cache = {
            "key": cache_key,
            "has_sizeable": bool(groups),
            "sizeable_mask": sizeable_mask,
            "pin_nodes": pin_nodes,
            "pin_main_ids": pin_main_ids,
            "pin_offsets": pin_offsets,
            "sizeable_pin_ids": sizeable_pin_ids,
            "groups": groups,
        }
        setattr(data_collections, "_sizing_pin_cap_static_cache", cache)
        return cache

    def _apply_sizing_driven_pin_caps(self, pin_cap_base, pin_rcap_base, pin_fcap_base, data_collections):
        if self._current_timing_surrogate_mode() == "lut_only":
            return pin_cap_base, pin_rcap_base, pin_fcap_base

        size_var = data_collections.get_size_var()
        vt_var = data_collections.get_vt_var()
        if size_var is None or vt_var is None:
            return pin_cap_base, pin_rcap_base, pin_fcap_base

        nodes_id = data_collections.pin2node_map
        pin_offsets = data_collections.pin_2_libpin_offset
        sizeable_mask = (
            (data_collections.inst_main_id[nodes_id] >= 0)
            & data_collections.inst_is_sizeable[nodes_id]
            & (pin_offsets >= 0)
        )
        if not torch.any(sizeable_mask):
            return pin_cap_base, pin_rcap_base, pin_fcap_base

        size_var = size_var.float()
        vt_var = vt_var.float()
        cache = self._get_sizing_pin_cap_static_cache(data_collections)
        if cache is not None:
            sizeable_mask = cache["sizeable_mask"]
            if not cache["has_sizeable"]:
                return pin_cap_base, pin_rcap_base, pin_fcap_base
            pin_nodes = cache["pin_nodes"]
            pin_main_ids = cache["pin_main_ids"]
            pin_offsets = cache["pin_offsets"]
            sizeable_pin_ids = cache["sizeable_pin_ids"]
            groups = cache["groups"]
        else:
            cell_sizes = data_collections.flat_libcell_info[:, 2].float()
            cell_vts = data_collections.flat_libcell_info[:, 3].long()
            pin_nodes = nodes_id[sizeable_mask]
            pin_main_ids = data_collections.inst_main_id[pin_nodes].long()
            pin_offsets = pin_offsets[sizeable_mask].long()
            sizeable_pin_ids = torch.nonzero(sizeable_mask, as_tuple=False).flatten()
            groups = None
        pin_sizes = size_var[pin_nodes]
        pin_vts = vt_var[pin_nodes]
        sized_pin_cap = torch.zeros_like(pin_sizes)
        sized_pin_rcap = torch.zeros_like(pin_sizes)
        sized_pin_fcap = torch.zeros_like(pin_sizes)
        size_temperature = 4.0
        debug_target_pins = (
            set() if self._production_fast_loop_enabled() else {"FE_DBTC149_n_2224:A"}
        )
        enable_sizing_pin_cap_debug = (
            os.environ.get("AUTODMP_ENABLE_SIZING_PIN_CAP_DEBUG", "0") == "1"
        )

        if groups is None:
            groups = []
            cell_sizes = data_collections.flat_libcell_info[:, 2].float()
            cell_vts = data_collections.flat_libcell_info[:, 3].long()
            for main_id in torch.unique(pin_main_ids):
                main_mask = pin_main_ids == main_id
                if not torch.any(main_mask):
                    continue
                cell_start = data_collections.main_id_2_cell_id_start[main_id].long()
                cell_end = data_collections.main_id_2_cell_id_start[main_id + 1].long()
                candidate_cell_ids = torch.arange(
                    cell_start,
                    cell_end,
                    device=pin_cap_base.device,
                    dtype=torch.long,
                )
                local_pin_offsets = pin_offsets[main_mask]
                candidate_libpin_ids = (
                    data_collections.cell_id_2_libpin_id_start[candidate_cell_ids].long().unsqueeze(0)
                    + local_pin_offsets.unsqueeze(1)
                )
                groups.append(
                    {
                        "main_mask": main_mask,
                        "candidate_cell_ids": candidate_cell_ids,
                        "candidate_sizes": cell_sizes[candidate_cell_ids],
                        "candidate_vts": cell_vts[candidate_cell_ids],
                        "candidate_libpin_ids": candidate_libpin_ids,
                        "candidate_caps": data_collections.flat_lib_pin_cap[candidate_libpin_ids],
                        "candidate_rcaps": data_collections.flat_lib_pin_rcap[candidate_libpin_ids],
                        "candidate_fcaps": data_collections.flat_lib_pin_fcap[candidate_libpin_ids],
                    }
                )

        for group in groups:
            main_mask = group["main_mask"]
            candidate_cell_ids = group["candidate_cell_ids"]
            candidate_sizes = group["candidate_sizes"]
            candidate_vts = group["candidate_vts"]
            candidate_libpin_ids = group["candidate_libpin_ids"]
            local_sizes = pin_sizes[main_mask]
            local_vt = pin_vts[main_mask]
            vt_scores = local_vt[:, candidate_vts].clamp_min(1e-9)
            size_scores = -size_temperature * torch.abs(
                local_sizes.unsqueeze(1) - candidate_sizes.unsqueeze(0)
            )
            logits = torch.log(vt_scores) + size_scores
            candidate_weights = F.softmax(logits, dim=1)
            sized_pin_cap[main_mask] = torch.sum(
                candidate_weights * group["candidate_caps"],
                dim=1,
            )
            sized_pin_rcap[main_mask] = torch.sum(
                candidate_weights * group["candidate_rcaps"],
                dim=1,
            )
            sized_pin_fcap[main_mask] = torch.sum(
                candidate_weights * group["candidate_fcaps"],
                dim=1,
            )

            if enable_sizing_pin_cap_debug:
                debug_local_rows = torch.nonzero(main_mask, as_tuple=False).flatten()
                for debug_local_row in debug_local_rows.tolist():
                    global_pin_id = int(sizeable_pin_ids[debug_local_row].item())
                    pin_name = self.placedb.pin_names[global_pin_id]
                    pin_name = pin_name.decode("utf-8") if isinstance(pin_name, bytes) else str(pin_name)
                    if pin_name not in debug_target_pins:
                        continue

                    pin_node_id = int(pin_nodes[debug_local_row].item())
                    local_pin_offset = int(pin_offsets[debug_local_row].item())
                    local_size = float(pin_sizes[debug_local_row].item())
                    local_vt_probs = [
                        round(float(x), 6) for x in pin_vts[debug_local_row].detach().cpu().tolist()
                    ]
                    raw_cap_pf = float(pin_cap_base[global_pin_id].item())
                    raw_rcap_pf = float(pin_rcap_base[global_pin_id].item())
                    raw_fcap_pf = float(pin_fcap_base[global_pin_id].item())
                    weighted_cap_pf = float(sized_pin_cap[debug_local_row].item())
                    weighted_rcap_pf = float(sized_pin_rcap[debug_local_row].item())
                    weighted_fcap_pf = float(sized_pin_fcap[debug_local_row].item())

                    row_weights = candidate_weights[debug_local_row].detach().cpu()
                    row_candidate_cell_ids = candidate_cell_ids.detach().cpu()
                    row_candidate_libpin_ids = candidate_libpin_ids[debug_local_row].detach().cpu()
                    row_candidate_caps = data_collections.flat_lib_pin_cap[
                        row_candidate_libpin_ids.to(candidate_libpin_ids.device)
                    ].detach().cpu()
                    row_candidate_rcaps = data_collections.flat_lib_pin_rcap[
                        row_candidate_libpin_ids.to(candidate_libpin_ids.device)
                    ].detach().cpu()
                    row_candidate_fcaps = data_collections.flat_lib_pin_fcap[
                        row_candidate_libpin_ids.to(candidate_libpin_ids.device)
                    ].detach().cpu()
                    row_candidate_sizes = candidate_sizes.detach().cpu()
                    row_candidate_vts = candidate_vts.detach().cpu()
                    top_k = min(5, row_weights.numel())
                    top_weights, top_indices = torch.topk(row_weights, k=top_k)
                    top_candidates = []
                    for rank in range(top_k):
                        cand_idx = int(top_indices[rank].item())
                        top_candidates.append(
                            {
                                "cell_id": int(row_candidate_cell_ids[cand_idx].item()),
                                "weight": round(float(top_weights[rank].item()), 6),
                                "size": round(float(row_candidate_sizes[cand_idx].item()), 6),
                                "vt": int(row_candidate_vts[cand_idx].item()),
                                "cap_pf": round(float(row_candidate_caps[cand_idx].item()), 9),
                                "rcap_pf": round(float(row_candidate_rcaps[cand_idx].item()), 9),
                                "fcap_pf": round(float(row_candidate_fcaps[cand_idx].item()), 9),
                            }
                        )

                    logging.info(
                        "SizingPinCapDebug pin=%s global_pin_id=%d node_id=%d "
                        "main_id=%d local_pin_offset=%d sizeable=True "
                        "raw_cap_pf=%.9f raw_rcap_pf=%.9f raw_fcap_pf=%.9f "
                        "weighted_cap_pf=%.9f weighted_rcap_pf=%.9f weighted_fcap_pf=%.9f "
                        "size_var=%.6f vt_var=%s top_candidates=%s",
                        pin_name,
                        global_pin_id,
                        pin_node_id,
                        int(main_id.item()),
                        local_pin_offset,
                        raw_cap_pf,
                        raw_rcap_pf,
                        raw_fcap_pf,
                        weighted_cap_pf,
                        weighted_rcap_pf,
                        weighted_fcap_pf,
                        local_size,
                        local_vt_probs,
                        top_candidates,
                    )

        pin_cap_base[sizeable_mask] = sized_pin_cap
        pin_rcap_base[sizeable_mask] = sized_pin_rcap
        pin_fcap_base[sizeable_mask] = sized_pin_fcap
        return pin_cap_base, pin_rcap_base, pin_fcap_base

    def pin_caps_op(self, inst_libcell_offset, data_collections):
        (
            self.pin2libpin_flat_ids,
            self.inst_pins_mask,
            pin_cap_base,
            pin_rcap_base,
            pin_fcap_base,
        ) = self._pin_caps_from_offsets(inst_libcell_offset, data_collections)
        enable_pin_cap_weight_debug = not self._production_fast_loop_enabled()
        raw_pin_cap_base = pin_cap_base.clone() if enable_pin_cap_weight_debug else None
        raw_pin_rcap_base = pin_rcap_base.clone() if enable_pin_cap_weight_debug else None
        raw_pin_fcap_base = pin_fcap_base.clone() if enable_pin_cap_weight_debug else None
        pin_cap_base, pin_rcap_base, pin_fcap_base = self._apply_sizing_driven_pin_caps(
            pin_cap_base, pin_rcap_base, pin_fcap_base, data_collections
        )
        debug_target_pins = (
            {"FE_DBTC149_n_2224:A"} if enable_pin_cap_weight_debug else set()
        )
        pin_name_to_idx = {}
        for idx, pin_name in enumerate(self.placedb.pin_names):
            decoded_name = pin_name.decode("utf-8") if isinstance(pin_name, bytes) else str(pin_name)
            if decoded_name in debug_target_pins:
                pin_name_to_idx[decoded_name] = idx
        for pin_name, global_pin_id in pin_name_to_idx.items():
            node_id = int(data_collections.pin2node_map[global_pin_id].item())
            pin_offset = int(data_collections.pin_2_libpin_offset[global_pin_id].item())
            main_id = int(data_collections.inst_main_id[node_id].item())
            is_sizeable = bool(data_collections.inst_is_sizeable[node_id].item()) if main_id >= 0 else False
            logging.info(
                "PinCapWeightCheck pin=%s global_pin_id=%d node_id=%d main_id=%d "
                "pin_offset=%d inst_pin=%d sizeable=%d raw_cap_pf=%.9f weighted_cap_pf=%.9f "
                "raw_rcap_pf=%.9f weighted_rcap_pf=%.9f raw_fcap_pf=%.9f weighted_fcap_pf=%.9f",
                pin_name,
                global_pin_id,
                node_id,
                main_id,
                pin_offset,
                int(self.inst_pins_mask[global_pin_id].item()),
                int(is_sizeable),
                float(raw_pin_cap_base[global_pin_id].item()),
                float(pin_cap_base[global_pin_id].item()),
                float(raw_pin_rcap_base[global_pin_id].item()),
                float(pin_rcap_base[global_pin_id].item()),
                float(raw_pin_fcap_base[global_pin_id].item()),
                float(pin_fcap_base[global_pin_id].item()),
            )
        self.output_pin_mask = self.inst_pins_mask & (pin_cap_base == 0)

        # fill in the pin capacitance for non-pin nodes
        non_inst_pins_mask = ~self.inst_pins_mask
        pin_cap_base[data_collections.end_points] += data_collections.outcaps
        pin_rcap_base[data_collections.end_points] += data_collections.outcaps
        pin_fcap_base[data_collections.end_points] += data_collections.outcaps
        return pin_cap_base, pin_rcap_base, pin_fcap_base

    def get_total_leakage_tensor(self):
        cell_leakage = getattr(self.data_collections, "flat_libcell_leakage", None)
        if cell_leakage is None:
            initial_leakage = getattr(self.data_collections, "inst_leakage_init", None)
            if initial_leakage is None:
                return None
            return initial_leakage.float().clamp(min=0.0).sum()

        size_var = self.data_collections.get_size_var()
        vt_var = self.data_collections.get_vt_var()
        if size_var is None or vt_var is None:
            initial_leakage = getattr(self.data_collections, "inst_leakage_init", None)
            if initial_leakage is None:
                return None
            return initial_leakage.float().clamp(min=0.0).sum()

        if (
            not self._production_fast_loop_enabled()
            and not getattr(self, "_leakage_debug_logged", False)
        ):
            leakage_min = float(cell_leakage.min().item()) if cell_leakage.numel() else 0.0
            leakage_max = float(cell_leakage.max().item()) if cell_leakage.numel() else 0.0
            leakage_sum = float(cell_leakage.float().sum().item()) if cell_leakage.numel() else 0.0
            leakage_nonzero = int((cell_leakage > 0).sum().item()) if cell_leakage.numel() else 0
            logging.info(
                "LeakageDebug: data_collections.flat_libcell_leakage stats count=%d nonzero=%d min=%.6e max=%.6e sum=%.6e",
                int(cell_leakage.numel()),
                leakage_nonzero,
                leakage_min,
                leakage_max,
                leakage_sum,
            )
            initial_leakage = getattr(self.data_collections, "inst_leakage_init", None)
            if initial_leakage is not None:
                init_min = float(initial_leakage.min().item()) if initial_leakage.numel() else 0.0
                init_max = float(initial_leakage.max().item()) if initial_leakage.numel() else 0.0
                init_sum = float(initial_leakage.float().sum().item()) if initial_leakage.numel() else 0.0
                init_nonzero = int((initial_leakage > 0).sum().item()) if initial_leakage.numel() else 0
                logging.info(
                    "LeakageDebug: data_collections.inst_leakage_init stats count=%d nonzero=%d min=%.6e max=%.6e sum=%.6e",
                    int(initial_leakage.numel()),
                    init_nonzero,
                    init_min,
                    init_max,
                    init_sum,
                )
            self._leakage_debug_logged = True

        # Early return if cell_leakage is all zeros (ASAP7 library has no leakage data)
        if torch.all(cell_leakage == 0):
            return cell_leakage.new_zeros(())

        inst_main_id = self.data_collections.inst_main_id.long()
        valid_mask = inst_main_id >= 0
        if not torch.any(valid_mask):
            return size_var.new_zeros(())

        size_var = size_var.float()
        vt_var = vt_var.float()
        cell_sizes = self.data_collections.flat_libcell_info[:, 2].float()
        cell_vts = self.data_collections.flat_libcell_info[:, 3].long()
        total_leakage = torch.zeros_like(size_var)
        size_temperature = 4.0

        for main_id in torch.unique(inst_main_id[valid_mask]):
            main_mask = inst_main_id == main_id
            if not torch.any(main_mask):
                continue
            cell_start = self.data_collections.main_id_2_cell_id_start[main_id].long()
            cell_end = self.data_collections.main_id_2_cell_id_start[main_id + 1].long()
            candidate_cell_ids = torch.arange(
                cell_start,
                cell_end,
                device=size_var.device,
                dtype=torch.long,
            )
            candidate_sizes = cell_sizes[candidate_cell_ids]
            candidate_vts = cell_vts[candidate_cell_ids]
            local_sizes = size_var[main_mask]
            local_vt = vt_var[main_mask]
            # Clamp vt_scores to avoid log(0) and numerical issues
            vt_scores = local_vt[:, candidate_vts].clamp(min=1e-9, max=1.0)
            # Check for invalid values
            if torch.any(torch.isnan(vt_scores)) or torch.any(torch.isinf(vt_scores)):
                logging.warning(f"Invalid vt_scores for main_id {main_id}, falling back to initial leakage")
                initial_leakage = getattr(self.data_collections, "inst_leakage_init", None)
                if initial_leakage is not None:
                    total_leakage[main_mask] = initial_leakage[main_mask].float()
                continue
            size_scores = -size_temperature * torch.abs(
                local_sizes.unsqueeze(1) - candidate_sizes.unsqueeze(0)
            )
            logits = torch.log(vt_scores) + size_scores
            # Check for invalid logits
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logging.warning(f"Invalid logits for main_id {main_id}, using size_scores only")
                logits = size_scores
            candidate_weights = F.softmax(logits, dim=1)
            # Check for invalid weights
            if torch.any(torch.isnan(candidate_weights)) or torch.any(torch.isinf(candidate_weights)):
                logging.warning(f"Invalid candidate_weights for main_id {main_id}, falling back to uniform")
                num_candidates = candidate_cell_ids.numel()
                candidate_weights = torch.ones_like(logits) / num_candidates
            total_leakage[main_mask] = torch.sum(
                candidate_weights * cell_leakage[candidate_cell_ids].float(),
                dim=1,
            )
        return total_leakage.clamp(min=0.0).sum()

    def get_total_leakage(self):
        total_leakage = self.get_total_leakage_tensor()
        if total_leakage is None:
            return None
        return float(total_leakage.detach().item())

    def timing_metric_snapshot(self):
        snapshot = {
            "wns": None,
            "tns": None,
            "ws": None,
            "ts": None,
            "max_slew_violation": None,
            "max_load_cap_violation": None,
        }
        for field_name in ("wns", "tns", "ws", "ts"):
            snapshot[field_name] = self._timing_metric_to_float(getattr(self, field_name, None))

        timing_op = getattr(self.op_collections, "timing_propagation_op", None)
        if timing_op is None:
            return snapshot

        inst_pins_mask = getattr(self, "inst_pins_mask", None)
        pin2libpin_flat_ids = getattr(self, "pin2libpin_flat_ids", None)
        if inst_pins_mask is None or pin2libpin_flat_ids is None:
            return snapshot
        if hasattr(inst_pins_mask, "to"):
            inst_pins_mask = inst_pins_mask.to(dtype=torch.bool)
        else:
            inst_pins_mask = torch.as_tensor(inst_pins_mask, dtype=torch.bool)
        if hasattr(pin2libpin_flat_ids, "to"):
            pin2libpin_flat_ids = pin2libpin_flat_ids.to(dtype=torch.long)
        else:
            pin2libpin_flat_ids = torch.as_tensor(pin2libpin_flat_ids, dtype=torch.long)

        mapped_slew_limits = self._mapped_instance_pin_limits(
            getattr(self.data_collections, "flat_lib_pin_slew_limit", None),
            inst_pins_mask,
            pin2libpin_flat_ids,
        )
        mapped_cap_limits = self._mapped_instance_pin_limits(
            getattr(self.data_collections, "flat_lib_pin_cap_limit", None),
            inst_pins_mask,
            pin2libpin_flat_ids,
        )

        max_slew_violation = self._max_positive_violation(
            (
                self._masked_instance_pin_values(getattr(timing_op, "pin_rtran", None), inst_pins_mask),
                self._masked_instance_pin_values(getattr(timing_op, "pin_ftran", None), inst_pins_mask),
            ),
            mapped_slew_limits,
        )
        max_load_cap_violation = self._max_positive_violation(
            (
                self._masked_instance_pin_values(
                    getattr(timing_op, "pin_net_cap_rise", None), inst_pins_mask
                ),
                self._masked_instance_pin_values(
                    getattr(timing_op, "pin_net_cap_fall", None), inst_pins_mask
                ),
            ),
            mapped_cap_limits,
        )
        snapshot["max_slew_violation"] = max_slew_violation
        snapshot["max_load_cap_violation"] = max_load_cap_violation
        self.max_slew_violation = max_slew_violation
        self.max_load_cap_violation = max_load_cap_violation
        return snapshot

    def _masked_instance_pin_values(self, values, inst_pins_mask):
        if values is None:
            return None
        if hasattr(values, "to"):
            value_tensor = values
        else:
            value_tensor = torch.as_tensor(values)
        mask = inst_pins_mask.to(device=value_tensor.device)
        if value_tensor.numel() != mask.numel():
            return None
        return value_tensor[mask]

    def _mapped_instance_pin_limits(self, flat_limits, inst_pins_mask, pin2libpin_flat_ids):
        if flat_limits is None:
            return None
        if hasattr(flat_limits, "to"):
            limit_table = flat_limits
        else:
            limit_table = torch.as_tensor(flat_limits)
        inst_pins_mask = inst_pins_mask.to(device=limit_table.device)
        pin2libpin_flat_ids = pin2libpin_flat_ids.to(device=limit_table.device)
        if pin2libpin_flat_ids.numel() != inst_pins_mask.numel():
            return None
        libpin_ids = pin2libpin_flat_ids[inst_pins_mask]
        if libpin_ids.numel() == 0:
            return limit_table.new_empty((0,), dtype=limit_table.dtype)
        if torch.any(libpin_ids < 0) or torch.any(libpin_ids >= limit_table.numel()):
            return None
        return limit_table[libpin_ids]

    def _mapped_all_pin_limits(self, flat_limits):
        pin2libpin_flat_ids = getattr(self, "pin2libpin_flat_ids", None)
        if flat_limits is None or pin2libpin_flat_ids is None:
            return None
        if hasattr(flat_limits, "detach"):
            flat_limit_values = flat_limits.detach().cpu().float()
        else:
            flat_limit_values = torch.as_tensor(flat_limits, dtype=torch.float32)
        if hasattr(pin2libpin_flat_ids, "detach"):
            flat_ids = pin2libpin_flat_ids.detach().cpu().long()
        else:
            flat_ids = torch.as_tensor(pin2libpin_flat_ids, dtype=torch.long)
        result = torch.full((flat_ids.numel(),), float("nan"), dtype=torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < flat_limit_values.numel())
        if valid.any():
            result[valid] = flat_limit_values[flat_ids[valid]]
        return result.numpy()

    def _max_positive_violation(self, observed_candidates, limits):
        if limits is None:
            return None
        valid_candidates = [candidate for candidate in observed_candidates if candidate is not None]
        if not valid_candidates:
            return None
        observed = torch.stack(valid_candidates).amax(dim=0)
        if observed.numel() == 0:
            return None
        if hasattr(limits, "to"):
            limit_tensor = limits.to(device=observed.device, dtype=observed.dtype)
        else:
            limit_tensor = torch.as_tensor(limits, device=observed.device, dtype=observed.dtype)
        if limit_tensor.numel() != observed.numel():
            return None
        valid_limit_mask = torch.isfinite(limit_tensor) & (limit_tensor > 0)
        if not torch.any(valid_limit_mask):
            return 0.0
        violations = observed[valid_limit_mask] - limit_tensor[valid_limit_mask]
        violations = torch.clamp(violations, min=0)
        if violations.numel() == 0:
            return 0.0
        return self._timing_metric_to_float(torch.max(violations))

    def _timing_metric_to_float(self, value):
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def build_ieda_rct(self):
        # ==============================================================================
        # --- 步骤 2: 初始化iEDA并使用正确的线电容为其构建RC树 ---
        # ==============================================================================
        logging.info("正在初始化iEDA STA引擎...")
        self._build_ieda_rct_debug_counter = (
            getattr(self, "_build_ieda_rct_debug_counter", 0) + 1
        )
        debug_build_idx = self._build_ieda_rct_debug_counter
        placeio_ieda = _load_placeio_ieda()
        ieda_sta = placeio_ieda.PlaceIOFunction.init_sta(
            self.placedb.rawdb
            if getattr(self.placedb, "rawdb", None) is not None
            else self.placedb.data_manager.dir_workspace
        )
        num_pins = len(self.placedb.pin_names)
        self.id2net_name_map = {v: k for k, v in self.placedb.net_name2id_map.items()}
        clock_net_ids = set()
        if (
            getattr(self.data_collections, "clock_pins", None) is not None
            and getattr(self.data_collections, "pin2net_map", None) is not None
            and self.data_collections.clock_pins.numel() > 0
        ):
            clock_net_ids = {
                int(net_id)
                for net_id in self.data_collections.pin2net_map[
                    self.data_collections.clock_pins
                ]
                .detach()
                .cpu()
                .tolist()
            }

        pin_fa = self.data_collections.pin_fa.clone().detach().cpu().numpy()
        flat_pin_from = (
            self.data_collections.flat_pin_from.clone().detach().cpu().numpy()
        )
        flat_pin_to = self.data_collections.flat_pin_to.clone().detach().cpu().numpy()

        # 关键：使用 Python RC timing 侧的数据网 RC 来驱动 iEDA 对照分析。
        # node_wire_caps_np follows elmore_delay_op.net_cap and is on the same
        # load-cap basis as Python timing (pF). edge_resistance is exported as
        # edge resistance (ohm) for the same RC tree topology.
        node_wire_caps_np = self.op_collections.elmore_delay_op.net_cap.cpu().numpy()
        edge_resistance = (
            self.op_collections.elmore_delay_op.edge_resistance.cpu().numpy()
        )
        edge_to_res_map = {
            (u, v): r for u, v, r in zip(flat_pin_from, flat_pin_to, edge_resistance)
        }
        debug_pin_caps_base = getattr(self, "_debug_last_pin_caps_base", None)
        debug_pin_rcaps_base = getattr(self, "_debug_last_pin_rcaps_base", None)
        debug_pin_fcaps_base = getattr(self, "_debug_last_pin_fcaps_base", None)

        logging.info("开始为iEDA构建所有网络的RC树...")
        skipped_clock_nets = 0
        debug_target_nets = {"FE_DBTN149_n_2224", "n_22053"}
        debug_target_pins = {
            "FE_DBTC149_n_2224:A",
            "FE_DBTC149_n_2224:Y",
            "g63378:A2",
            "g63378:Y",
        }
        for net_id, net_name in self.id2net_name_map.items():
            if int(net_id) in clock_net_ids:
                skipped_clock_nets += 1
                continue
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
                if global_idx < num_pins:
                    pin_name = (
                        self.placedb.pin_names[global_idx].decode("utf-8")
                        if isinstance(self.placedb.pin_names[global_idx], bytes)
                        else str(self.placedb.pin_names[global_idx])
                    )
                    if pin_name in debug_target_pins:
                        base_cap_pf = (
                            float(debug_pin_caps_base[global_idx])
                            if debug_pin_caps_base is not None
                            else float("nan")
                        )
                        base_rcap_pf = (
                            float(debug_pin_rcaps_base[global_idx])
                            if debug_pin_rcaps_base is not None
                            else float("nan")
                        )
                        base_fcap_pf = (
                            float(debug_pin_fcaps_base[global_idx])
                            if debug_pin_fcaps_base is not None
                            else float("nan")
                        )
                        wire_cap_pf = float(node_wire_caps_np[global_idx])
                        logging.info(
                            "PinCapBridgeDebugPy build=%d net=%s pin=%s "
                            "global_idx=%d base_cap_pf=%.9f base_rcap_pf=%.9f "
                            "base_fcap_pf=%.9f wire_cap_pf=%.9f total_cap_pf=%.9f "
                            "total_rcap_pf=%.9f total_fcap_pf=%.9f",
                            debug_build_idx,
                            net_name,
                            pin_name,
                            global_idx,
                            base_cap_pf,
                            base_rcap_pf,
                            base_fcap_pf,
                            wire_cap_pf,
                            base_cap_pf + wire_cap_pf,
                            base_rcap_pf + wire_cap_pf,
                            base_fcap_pf + wire_cap_pf,
                        )
            if net_name in debug_target_nets:
                decoded_names = [
                    name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    for name in node_sta_names
                ]
                logging.info(
                    "BuildIedaRCTDebug build=%d net=%s node_count=%d "
                    "cap_sum_pf=%.9f caps_pf=%s parents=%s edge_res_ohm=%s nodes=%s",
                    debug_build_idx,
                    net_name,
                    len(node_wire_caps),
                    float(sum(node_wire_caps)),
                    [round(float(cap), 9) for cap in node_wire_caps],
                    parent_indices,
                    [round(float(res), 9) for res in edge_resistances_net],
                    decoded_names,
                )
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
        logging.info(
            "所有网络的RC树构建完成。跳过了 %d 条 clock net，保留 %d 条 data net。",
            skipped_clock_nets,
            len(self.id2net_name_map) - skipped_clock_nets,
        )

        # ==============================================================================
        # --- 步骤 3: 调用iEDA执行分析并获取所有调试信息 ---
        # ==============================================================================
        logging.info("调用iEDA执行时序分析并获取详细数据...")
        refresh_mode = os.environ.get(
            "AUTODMP_ISTA_QUERY_REFRESH_MODE", "graph"
        ).strip()
        if refresh_mode:
            # This refresh is only for the retained-evaluation query path.
            # It should not affect the optimization-side timing engine state.
            logging.info(
                "check_log前执行iSTA query refresh，mode=%s",
                refresh_mode,
            )
            ieda_sta.refresh_timing_query_state(refresh_mode)
        # update_and_get_all_pin_timings returns raw iEDA/py_ista debug payloads.
        # In the current bridge:
        #   - arrival / required / delay / slew are reported in ns
        #   - load_cap stays on the exported capacitance basis (pF)
        # Downstream comparison helpers convert selected fields to ps when they
        # need to align with Python TimingPropagation tensors.
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
        self._last_ieda_sta = ieda_sta
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
        py_pin_r_slew = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_rtran_live", None)
        )
        py_pin_f_slew = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_ftran_live", None)
        )
        if py_pin_r_slew is None:
            py_pin_r_slew = _tensor_to_cpu_float_array(getattr(op_timing, "pin_rtran", None))
        if py_pin_f_slew is None:
            py_pin_f_slew = _tensor_to_cpu_float_array(getattr(op_timing, "pin_ftran", None))
        if py_pin_r_slew is None or py_pin_f_slew is None:
            logging.warning("skip timing_pin_all report: Python pin slew is unavailable")
            return

        # 新增: 获取Python端的AT和RT数据
        py_r_aat_late = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_rAAT_live_snapshot", None)
        )
        py_f_aat_late = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_fAAT_live_snapshot", None)
        )
        py_r_rat_late = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_rRAT_live_snapshot", None)
        )
        py_f_rat_late = _tensor_to_cpu_float_array(
            getattr(op_timing, "pin_fRAT_live_snapshot", None)
        )
        if py_r_aat_late is None:
            py_r_aat_late = _tensor_to_cpu_float_array(getattr(op_timing, "pin_rAAT", None))
        if py_f_aat_late is None:
            py_f_aat_late = _tensor_to_cpu_float_array(getattr(op_timing, "pin_fAAT", None))
        if py_r_rat_late is None:
            py_r_rat_late = _tensor_to_cpu_float_array(getattr(op_timing, "pin_rRAT", None))
        if py_f_rat_late is None:
            py_f_rat_late = _tensor_to_cpu_float_array(getattr(op_timing, "pin_fRAT", None))
        if (
            py_r_aat_late is None
            or py_f_aat_late is None
            or py_r_rat_late is None
            or py_f_rat_late is None
        ):
            logging.warning("skip timing_pin_all report: Python AAT/RAT is unavailable")
            return
        
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
        debug_ieda_arc_candidates = {}
        debug_target_keys = {
            (
                "FE_DBTC149_n_2224",
                "FE_DBTC149_n_2224:A",
                "FE_DBTC149_n_2224:Y",
            ),
            (
                "g63378",
                "g63378:A2",
                "g63378:Y",
            ),
        }
        for arc in cell_arc_delays_cpp:
            key_t = (
                arc["inst_name"],
                arc["from_pin"],
                arc["to_pin"],
                arc["transition"],
                arc["arc_sense"],
            )
            debug_key = (arc["inst_name"], arc["from_pin"], arc["to_pin"])
            if debug_key in debug_target_keys:
                debug_ieda_arc_candidates.setdefault(key_t, []).append(
                    {
                        "analysis_mode": arc["analysis_mode"],
                        "transition": arc["transition"],
                        "arc_sense": arc["arc_sense"],
                        "in_slew_ns": arc["in_slew_ns"],
                        "load_cap": arc["load_cap"],
                        "delay_ns": arc["delay_ns"],
                    }
                )
            if ieda_arc_map.get(key_t) is not None:
                tmp = ieda_arc_map[key_t]
                if tmp['delay_ns'] < arc['delay_ns']:
                    ieda_arc_map[key_t] = arc
                # logging.info(f"Warning: Duplicate arc key found: {key_t}. Overwriting previous entry.")
            else:
                ieda_arc_map[key_t] = arc

        for key_t, candidates in sorted(debug_ieda_arc_candidates.items()):
            logging.info(
                "ArcCsvDebug iEDA candidates key=%s count=%d",
                key_t,
                len(candidates),
            )
            for candidate in sorted(
                candidates, key=lambda item: item["delay_ns"], reverse=True
            ):
                logging.info(
                    "ArcCsvDebug candidate key=%s mode=%s trans=%s sense=%s "
                    "in_slew_ns=%.9f load_cap_pf=%.9f delay_ns=%.9f",
                    key_t,
                    candidate["analysis_mode"],
                    candidate["transition"],
                    candidate["arc_sense"],
                    candidate["in_slew_ns"],
                    candidate["load_cap"],
                    candidate["delay_ns"],
                )

        # 4b. 预处理Python端计算的Cell Arc数据 (逻辑修正版)
        op_timing = self.op_collections.timing_propagation_op
        op_elmore = self.op_collections.elmore_delay_op

        py_cell_arc_rr_delays = _tensor_to_cpu_float_array(getattr(op_timing, "cell_arc_rr_delays", None))
        py_cell_arc_ff_delays = _tensor_to_cpu_float_array(getattr(op_timing, "cell_arc_ff_delays", None))
        py_cell_arc_rf_delays = _tensor_to_cpu_float_array(getattr(op_timing, "cell_arc_rf_delays", None))
        py_cell_arc_fr_delays = _tensor_to_cpu_float_array(getattr(op_timing, "cell_arc_fr_delays", None))
        if (
            py_cell_arc_rr_delays is None
            or py_cell_arc_ff_delays is None
            or py_cell_arc_rf_delays is None
            or py_cell_arc_fr_delays is None
        ):
            logging.warning(
                "skip cell_arc_delay_report: Python cell arc delay tensors are unavailable "
                "(expected under production_fast_loop)"
            )
            return

        py_pin_r_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_rtran_live", "pin_rtran"
        )
        py_pin_f_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_ftran_live", "pin_ftran"
        )
        if py_pin_r_slew is None or py_pin_f_slew is None:
            logging.warning("skip cell_arc_delay_report: Python pin slew tensors are unavailable")
            return
        py_pin_r_load = op_elmore.loads["rise"].clone().detach().cpu().numpy()
        py_pin_f_load = op_elmore.loads["fall"].clone().detach().cpu().numpy()

        python_arc_map = {}
        id2cell_name_map = {v: k for k, v in self.placedb.node_name2id_map.items()}
        pin_names = self.placedb.pin_names

        cell_arcs = self.data_collections.inst_flat_arcs.cpu().numpy()
        cell_arcs_start = self.data_collections.inst_flat_arcs_start.cpu().numpy()
        level_arcs = self.data_collections.flat_inst_arcs_by_level.cpu().numpy()

        for cell_id, cell_name in id2cell_name_map.items():
            start_index = cell_arcs_start[cell_id]
            end_index = cell_arcs_start[cell_id + 1]
            if start_index == end_index:
                continue

            for inst_arc_idx in range(start_index, end_index):
                level_arc_idx = int(cell_arcs[inst_arc_idx])
                arc_info = level_arcs[level_arc_idx]
                if len(arc_info) < 5:
                    raise ValueError(
                        f"flat_inst_arcs_by_level entry expects at least 5 columns, got {len(arc_info)}"
                    )

                in_pin_id = int(arc_info[0])
                out_pin_id = int(arc_info[1])
                arc_sense = int(arc_info[4])

                from_pin_name = pin_names[in_pin_id].decode("utf-8")
                to_pin_name = pin_names[out_pin_id].decode("utf-8")

                is_inverting = arc_sense == -1  # negative_unate
                is_unate = arc_sense == 0      # non_unate

                # --- 情况一: 报告中对应 output "Rise" 的行 ---
                delay_for_output_rise = (
                    py_cell_arc_fr_delays[level_arc_idx]
                    if is_inverting
                    else (
                        py_cell_arc_rr_delays[level_arc_idx]
                        if not is_unate
                        else max(
                            py_cell_arc_rr_delays[level_arc_idx],
                            py_cell_arc_fr_delays[level_arc_idx],
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
                    py_cell_arc_rf_delays[level_arc_idx]
                    if is_inverting
                    else (
                        py_cell_arc_ff_delays[level_arc_idx]
                        if not is_unate
                        else max(
                            py_cell_arc_ff_delays[level_arc_idx],
                            py_cell_arc_rf_delays[level_arc_idx],
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
            # Python arc delay tensors are already in ps. iEDA debug arc delay is
            # exported in ns, so normalize it to ps before diffing/reporting.
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

                # Keep the CSV on a single basis:
                #   - Py Delay / Slew are already ps
                #   - iEDA delay_ns / in_slew_ns are converted to ps here
                #   - load_cap is emitted as pF on both sides
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
        op_timing = self.op_collections.timing_propagation_op
        reference_tensor = _timing_tensor_ref(
            op_timing,
            "pin_rAAT_live_snapshot",
            "pin_rAAT",
            "pin_rtran_live",
            "pin_rtran",
        )
        if reference_tensor is None:
            logging.warning("skip slack compare: Python timing tensors are unavailable")
            return
        t_dtype = reference_tensor.dtype
        t_device = reference_tensor.device
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
        # at_late_cpp / rt_late_cpp are the raw iEDA arrays returned by py_ista
        # and remain in ns in this helper. Python wns/tns inputs are still on
        # TimingPropagation's ps basis, so the log below converts only the
        # Python side to ns for an apples-to-apples printout.
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

    def _build_endpoint_timing_compare(self, at_late_cpp, rt_late_cpp, at_early_cpp=None, rt_early_cpp=None):
        op_timing = self.op_collections.timing_propagation_op
        pin_names = self.placedb.pin_names
        backend_caps = dict(getattr(self.placedb, "backend_caps", {}) or {})
        endpoint_ids = self.data_collections.end_points.detach().cpu().tolist()
        py_r_aat = _timing_tensor_ref(op_timing, "pin_rAAT_live_snapshot", "pin_rAAT")
        py_f_aat = _timing_tensor_ref(op_timing, "pin_fAAT_live_snapshot", "pin_fAAT")
        py_r_rat = _timing_tensor_ref(op_timing, "pin_rRAT_live_snapshot", "pin_rRAT")
        py_f_rat = _timing_tensor_ref(op_timing, "pin_fRAT_live_snapshot", "pin_fRAT")
        if py_r_aat is None or py_f_aat is None or py_r_rat is None or py_f_rat is None:
            logging.warning("skip endpoint timing compare: Python AAT/RAT tensors are unavailable")
            return [], {"status": "skipped", "reason": "python_aat_rat_unavailable"}
        py_r_aat = py_r_aat.detach().cpu()
        py_f_aat = py_f_aat.detach().cpu()
        py_r_rat = py_r_rat.detach().cpu()
        py_f_rat = py_f_rat.detach().cpu()
        sentinel_threshold = 5e7
        timing_check_arcs = _tensor_to_cpu_long_array(
            getattr(self.data_collections, "endpoints_timing_check_arcs", None)
        )
        timing_checks_by_endpoint = {}
        if timing_check_arcs is not None and timing_check_arcs.size > 0:
            for arc in timing_check_arcs:
                if len(arc) <= 6:
                    continue
                out_pin_id = int(arc[1])
                timing_checks_by_endpoint.setdefault(out_pin_id, []).append(int(arc[6]))

        def _pin_name(pin_id):
            value = pin_names[pin_id]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        rows = []
        for pin_id in endpoint_ids:
            py_r_aat_v = float(py_r_aat[pin_id].item())
            py_f_aat_v = float(py_f_aat[pin_id].item())
            py_r_rat_v = float(py_r_rat[pin_id].item())
            py_f_rat_v = float(py_f_rat[pin_id].item())
            timing_check_classes_list = sorted(set(timing_checks_by_endpoint.get(pin_id, [])))
            timing_check_bucket = _endpoint_timing_check_bucket(timing_check_classes_list)
            # Endpoint compare is intentionally emitted on the Python timing
            # basis (ps). The incoming iEDA arrays are ns, so convert them here
            # before computing slack deltas or serializing the JSON summary.
            cpp_r_aat_v = float(at_late_cpp[pin_id][0]) * 1000.0
            cpp_f_aat_v = float(at_late_cpp[pin_id][1]) * 1000.0
            cpp_r_rat_v = float(rt_late_cpp[pin_id][0]) * 1000.0
            cpp_f_rat_v = float(rt_late_cpp[pin_id][1]) * 1000.0
            cpp_min_r_aat_v = (
                float(at_early_cpp[pin_id][0]) * 1000.0
                if at_early_cpp is not None
                else np.nan
            )
            cpp_min_f_aat_v = (
                float(at_early_cpp[pin_id][1]) * 1000.0
                if at_early_cpp is not None
                else np.nan
            )
            cpp_min_r_rat_v = (
                float(rt_early_cpp[pin_id][0]) * 1000.0
                if rt_early_cpp is not None
                else np.nan
            )
            cpp_min_f_rat_v = (
                float(rt_early_cpp[pin_id][1]) * 1000.0
                if rt_early_cpp is not None
                else np.nan
            )
            py_slack = min(py_r_rat_v - py_r_aat_v, py_f_rat_v - py_f_aat_v)
            cpp_slack = min(cpp_r_rat_v - cpp_r_aat_v, cpp_f_rat_v - cpp_f_aat_v)
            cpp_min_slack = _backend_min_slack_from_endpoint_values(
                cpp_min_r_aat_v,
                cpp_min_f_aat_v,
                cpp_min_r_rat_v,
                cpp_min_f_rat_v,
                sentinel_threshold=sentinel_threshold,
            )
            rows.append(
                {
                    "pin_id": int(pin_id),
                    "pin_name": _pin_name(pin_id),
                    "timing_check_classes": "|".join(str(value) for value in timing_check_classes_list),
                    "timing_check_bucket": timing_check_bucket,
                    "timing_check_classes_list": timing_check_classes_list,
                    "py_r_aat": py_r_aat_v,
                    "py_f_aat": py_f_aat_v,
                    "py_r_rat": py_r_rat_v,
                    "py_f_rat": py_f_rat_v,
                    "cpp_r_aat": cpp_r_aat_v,
                    "cpp_f_aat": cpp_f_aat_v,
                    "cpp_r_rat": cpp_r_rat_v,
                    "cpp_f_rat": cpp_f_rat_v,
                    "cpp_min_r_aat": cpp_min_r_aat_v,
                    "cpp_min_f_aat": cpp_min_f_aat_v,
                    "cpp_min_r_rat": cpp_min_r_rat_v,
                    "cpp_min_f_rat": cpp_min_f_rat_v,
                    "py_slack": py_slack,
                    "cpp_slack": cpp_slack,
                    "cpp_min_slack": cpp_min_slack,
                    "slack_delta": py_slack - cpp_slack,
                    "python_arrival_sentinel": bool(
                        py_r_aat_v <= -sentinel_threshold or py_f_aat_v <= -sentinel_threshold
                    ),
                    "python_required_sentinel": bool(
                        py_r_rat_v >= sentinel_threshold or py_f_rat_v >= sentinel_threshold
                    ),
                    "python_nonfinite": not np.isfinite(py_slack),
                    "backend_nonfinite": not np.isfinite(cpp_slack),
                    "backend_min_nonfinite": not np.isfinite(cpp_min_slack),
                    "ieda_nonfinite": not np.isfinite(cpp_slack),
                    "python_negative_slack": bool(np.isfinite(py_slack) and py_slack < 0.0),
                    "backend_negative_slack": bool(np.isfinite(cpp_slack) and cpp_slack < 0.0),
                    "backend_min_negative_slack": bool(np.isfinite(cpp_min_slack) and cpp_min_slack < 0.0),
                    "ieda_negative_slack": bool(np.isfinite(cpp_slack) and cpp_slack < 0.0),
                }
            )

        rows.sort(key=lambda item: abs(item["slack_delta"]), reverse=True)

        def _finite_values(key):
            return [row[key] for row in rows if np.isfinite(row[key])]

        py_slacks = _finite_values("py_slack")
        cpp_slacks = _finite_values("cpp_slack")
        cpp_min_slacks = _finite_values("cpp_min_slack")
        cpp_check_slacks = [
            _endpoint_backend_check_slack(row)
            for row in rows
            if _endpoint_backend_check_slack(row) is not None
            and np.isfinite(_endpoint_backend_check_slack(row))
        ]
        # Summary WNS/TNS in this file are therefore stored in ps as well,
        # unlike OpenROAD metric JSONs which are commonly reported in ns.
        summary = {
            "num_endpoints": len(rows),
            "backend_metadata": {
                "backend": backend_caps.get("backend", "unknown"),
                "sta_reference_role": backend_caps.get("sta_reference_role", "unspecified"),
                "sta_is_golden": backend_caps.get("sta_reference_role") == "golden_opensta",
                "parasitics_initialization": backend_caps.get(
                    "parasitics_initialization",
                    "unspecified",
                ),
                "sta_state_status": backend_caps.get("sta_state_status", "unspecified"),
            },
            "num_python_negative_slack": sum(row["python_negative_slack"] for row in rows),
            "num_backend_negative_slack": sum(row["backend_negative_slack"] for row in rows),
            "num_backend_min_negative_slack": sum(row["backend_min_negative_slack"] for row in rows),
            "num_backend_check_negative_slack": sum(
                _endpoint_backend_check_slack(row) is not None
                and np.isfinite(_endpoint_backend_check_slack(row))
                and _endpoint_backend_check_slack(row) < 0.0
                for row in rows
            ),
            "num_ieda_negative_slack": sum(row["ieda_negative_slack"] for row in rows),
            "num_python_arrival_sentinel": sum(row["python_arrival_sentinel"] for row in rows),
            "num_python_required_sentinel": sum(row["python_required_sentinel"] for row in rows),
            "num_python_nonfinite_slack": sum(row["python_nonfinite"] for row in rows),
            "num_backend_nonfinite_slack": sum(row["backend_nonfinite"] for row in rows),
            "num_backend_min_nonfinite_slack": sum(row["backend_min_nonfinite"] for row in rows),
            "num_ieda_nonfinite_slack": sum(row["ieda_nonfinite"] for row in rows),
            "python_wns": min(py_slacks) if py_slacks else None,
            "backend_wns": min(cpp_slacks) if cpp_slacks else None,
            "backend_min_wns": min(cpp_min_slacks) if cpp_min_slacks else None,
            "backend_check_wns": min(cpp_check_slacks) if cpp_check_slacks else None,
            "ieda_wns": min(cpp_slacks) if cpp_slacks else None,
            "python_tns": float(sum(min(value, 0.0) for value in py_slacks)) if py_slacks else None,
            "backend_tns": float(sum(min(value, 0.0) for value in cpp_slacks)) if cpp_slacks else None,
            "backend_min_tns": float(sum(min(value, 0.0) for value in cpp_min_slacks)) if cpp_min_slacks else None,
            "backend_check_tns": float(sum(min(value, 0.0) for value in cpp_check_slacks)) if cpp_check_slacks else None,
            "ieda_tns": float(sum(min(value, 0.0) for value in cpp_slacks)) if cpp_slacks else None,
            "timing_check_bucket_summary": _summarize_endpoint_compare_buckets(rows),
            "worst_slack_delta_pin": rows[0]["pin_name"] if rows else None,
            "worst_slack_delta": rows[0]["slack_delta"] if rows else None,
        }
        return rows, summary

    def write_endpoint_timing_compare(
        self, at_late_cpp, rt_late_cpp, at_early_cpp=None, rt_early_cpp=None
    ):
        rows, summary = self._build_endpoint_timing_compare(
            at_late_cpp, rt_late_cpp, at_early_cpp, rt_early_cpp
        )
        csv_filename = "%s/%s_endpoint_timing_compare.csv" % (
            self.params.result_dir, self.params.design_name()
        )
        summary_filename = "%s/%s_endpoint_timing_compare_summary.json" % (
            self.params.result_dir, self.params.design_name()
        )

        import csv

        if summary.get("status") == "skipped":
            with open(summary_filename, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
            logging.warning(
                "endpoint timing compare skipped: %s",
                summary.get("reason", "unknown"),
            )
            return rows, summary

        fieldnames = [
            "pin_id",
            "pin_name",
            "timing_check_classes",
            "timing_check_bucket",
            "py_r_aat",
            "py_f_aat",
            "py_r_rat",
            "py_f_rat",
            "cpp_r_aat",
            "cpp_f_aat",
            "cpp_r_rat",
            "cpp_f_rat",
            "cpp_min_r_aat",
            "cpp_min_f_aat",
            "cpp_min_r_rat",
            "cpp_min_f_rat",
            "py_slack",
            "cpp_slack",
            "cpp_min_slack",
            "slack_delta",
            "python_arrival_sentinel",
            "python_required_sentinel",
            "python_nonfinite",
            "backend_nonfinite",
            "backend_min_nonfinite",
            "ieda_nonfinite",
            "python_negative_slack",
            "backend_negative_slack",
            "backend_min_negative_slack",
            "ieda_negative_slack",
        ]
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row.pop("timing_check_classes_list", None)
                writer.writerow(csv_row)
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logging.info(f"endpoint 时序对比已写入CSV文件: {csv_filename}")
        logging.info(f"endpoint 时序对比摘要已写入JSON文件: {summary_filename}")
        logging.info(
            "endpoint 对齐摘要: python_neg=%d, backend_neg=%d, py_arrival_sentinel=%d, py_required_sentinel=%d",
            summary["num_python_negative_slack"],
            summary["num_backend_negative_slack"],
            summary["num_python_arrival_sentinel"],
            summary["num_python_required_sentinel"],
        )
        return rows, summary

    def write_backend_endpoint_timing_compare_if_available(self):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is None:
            return None
        late_attrs = (
            "backend_endpoint_rAAT",
            "backend_endpoint_fAAT",
            "backend_endpoint_rRAT",
            "backend_endpoint_fRAT",
        )
        arrays = {
            name: _tensor_to_cpu_float_array(getattr(self.placedb, name, None))
            for name in late_attrs
        }
        if any(value is None or len(value) == 0 for value in arrays.values()):
            return None
        endpoint_ids = _tensor_to_cpu_long_array(getattr(data_collections, "end_points", None))
        if endpoint_ids is None or len(endpoint_ids) != len(arrays["backend_endpoint_rAAT"]):
            return None

        min_attrs = (
            "backend_endpoint_min_rAAT",
            "backend_endpoint_min_fAAT",
            "backend_endpoint_min_rRAT",
            "backend_endpoint_min_fRAT",
        )
        min_arrays = {
            name: _tensor_to_cpu_float_array(getattr(self.placedb, name, None))
            for name in min_attrs
        }
        has_min_arrays = all(
            value is not None and len(value) == len(endpoint_ids)
            for value in min_arrays.values()
        )

        num_pins = len(getattr(self.placedb, "pin_names", []))
        at_late = np.full((num_pins, 2), np.nan, dtype=np.float64)
        rt_late = np.full((num_pins, 2), np.nan, dtype=np.float64)
        at_early = np.full((num_pins, 2), np.nan, dtype=np.float64) if has_min_arrays else None
        rt_early = np.full((num_pins, 2), np.nan, dtype=np.float64) if has_min_arrays else None
        for endpoint_index, pin_id in enumerate(endpoint_ids.tolist()):
            at_late[pin_id, 0] = arrays["backend_endpoint_rAAT"][endpoint_index] / 1000.0
            at_late[pin_id, 1] = arrays["backend_endpoint_fAAT"][endpoint_index] / 1000.0
            rt_late[pin_id, 0] = arrays["backend_endpoint_rRAT"][endpoint_index] / 1000.0
            rt_late[pin_id, 1] = arrays["backend_endpoint_fRAT"][endpoint_index] / 1000.0
            if has_min_arrays:
                at_early[pin_id, 0] = min_arrays["backend_endpoint_min_rAAT"][endpoint_index] / 1000.0
                at_early[pin_id, 1] = min_arrays["backend_endpoint_min_fAAT"][endpoint_index] / 1000.0
                rt_early[pin_id, 0] = min_arrays["backend_endpoint_min_rRAT"][endpoint_index] / 1000.0
                rt_early[pin_id, 1] = min_arrays["backend_endpoint_min_fRAT"][endpoint_index] / 1000.0
        return self.write_endpoint_timing_compare(at_late, rt_late, at_early, rt_early)

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
        py_pin_r_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_rtran_live", "pin_rtran"
        )
        py_pin_f_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_ftran_live", "pin_ftran"
        )
        if py_pin_r_slew is None or py_pin_f_slew is None:
            logging.warning("skip first-level pin timing log: Python pin slew tensors are unavailable")
            return None

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

    def write_clk2q_level0_arc_report(self, at_late_cpp, cell_arc_delays_cpp, net_timing_details_cpp):
        import csv

        op_timing = self.op_collections.timing_propagation_op
        pin_names = self.placedb.pin_names
        clk_pin_names = getattr(self.placedb, "clk_pin_names", None)
        if clk_pin_names is None:
            logging.warning("clk2q level-0 report skipped: placedb.clk_pin_names is missing")
            return None, None

        level_start = int(self.data_collections.flat_inst_arcs_by_level_start[0].item())
        level_end = int(self.data_collections.flat_inst_arcs_by_level_start[1].item())
        level0_arcs = self.data_collections.flat_inst_arcs_by_level[level_start:level_end].detach().cpu().numpy()
        start_point_set = set(self.data_collections.start_points.detach().cpu().tolist())

        py_pin_r_aat = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_rAAT_live_snapshot", "pin_rAAT"
        )
        py_pin_f_aat = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_fAAT_live_snapshot", "pin_fAAT"
        )
        py_pin_r_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_rtran_live", "pin_rtran"
        )
        py_pin_f_slew = _timing_tensor_to_cpu_float_array(
            op_timing, "pin_ftran_live", "pin_ftran"
        )
        if (
            py_pin_r_aat is None
            or py_pin_f_aat is None
            or py_pin_r_slew is None
            or py_pin_f_slew is None
        ):
            logging.warning("clk2q level-0 report skipped: Python pin AAT/slew tensors are unavailable")
            return None, None
        py_pin_net_cap_rise = _timing_tensor_ref(
            op_timing, "pin_net_cap_rise_live", "pin_net_cap_rise"
        )
        py_pin_net_cap_fall = _timing_tensor_ref(
            op_timing, "pin_net_cap_fall_live", "pin_net_cap_fall"
        )
        exported_clk_r_aat = getattr(self.placedb, "clk_pin_r_aat", None)
        exported_clk_f_aat = getattr(self.placedb, "clk_pin_f_aat", None)
        py_clk_r_slew = _tensor_to_cpu_float_array(getattr(op_timing, "clk_pin_rtran", None))
        py_clk_f_slew = _tensor_to_cpu_float_array(getattr(op_timing, "clk_pin_ftran", None))
        if py_clk_r_slew is None or py_clk_f_slew is None:
            logging.warning("clk2q level-0 report skipped: exported clock slew tensors are unavailable")
            return None, None

        def _decode_name(value):
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        def _timing_sense_name(value):
            if value == 1:
                return "positive_unate"
            if value == -1:
                return "negative_unate"
            return "non_unate"

        def _timing_type_name(value):
            if value == 1:
                return "rising_edge"
            if value == -1:
                return "falling_edge"
            return "both_edge"

        ieda_pin_map = {
            (info["pin_name"], info["mode"], info["transition"]): info
            for info in net_timing_details_cpp
        }
        ieda_arc_map = {}
        for arc in cell_arc_delays_cpp:
            key = (arc["inst_name"], arc["from_pin"], arc["to_pin"], arc["transition"])
            prev = ieda_arc_map.get(key)
            if prev is None or prev.get("delay_ns", float("-inf")) < arc.get("delay_ns", float("-inf")):
                ieda_arc_map[key] = arc

        golden_r_delay = None
        golden_f_delay = None
        golden_r_slew = None
        golden_f_slew = None
        golden_load_r = None
        golden_load_f = None
        if py_pin_net_cap_rise is not None and py_pin_net_cap_fall is not None and len(level0_arcs):
            arc_in_pins_t = self.data_collections.flat_inst_arcs_by_level[level_start:level_end, 0]
            arc_out_pins_t = self.data_collections.flat_inst_arcs_by_level[level_start:level_end, 1]
            lib_cell_idxs_t = self.data_collections.flat_inst_arcs_by_level[level_start:level_end, 2]
            lib_arc_idxs_t = self.data_collections.flat_inst_arcs_by_level[level_start:level_end, 3]
            load_r_t = py_pin_net_cap_rise[arc_out_pins_t]
            load_f_t = py_pin_net_cap_fall[arc_out_pins_t]
            # Level-0 golden checks stay entirely on the Python timing basis:
            # zero input slew in ps, output load in pF, and LUT outputs in ps.
            zero_r_slew_t = torch.zeros_like(load_r_t)
            zero_f_slew_t = torch.zeros_like(load_f_t)
            with torch.no_grad():
                golden_r_delay = (
                    op_timing.r_delay_entry(
                        lib_cell_idxs_t,
                        zero_r_slew_t,
                        load_r_t,
                        lib_arc_idxs_t,
                        arc_out_pins_t,
                        use_surrogate=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                golden_f_delay = (
                    op_timing.f_delay_entry(
                        lib_cell_idxs_t,
                        zero_f_slew_t,
                        load_f_t,
                        lib_arc_idxs_t,
                        arc_out_pins_t,
                        use_surrogate=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                golden_r_slew = (
                    op_timing.r_tran_entry(
                        lib_cell_idxs_t,
                        zero_r_slew_t,
                        load_r_t,
                        lib_arc_idxs_t,
                        arc_out_pins_t,
                        use_surrogate=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                golden_f_slew = (
                    op_timing.f_tran_entry(
                        lib_cell_idxs_t,
                        zero_f_slew_t,
                        load_f_t,
                        lib_arc_idxs_t,
                        arc_out_pins_t,
                        use_surrogate=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            golden_load_r = load_r_t.detach().cpu().numpy()
            golden_load_f = load_f_t.detach().cpu().numpy()

        rows = []
        for arc_index, arc in enumerate(level0_arcs):
            clk_local_id = int(arc[0])
            q_pin_id = int(arc[1])
            lib_cell_idx = int(arc[2])
            lib_arc_idx = int(arc[3])
            timing_sense = int(arc[4])
            timing_type = int(arc[5])

            q_pin_name = _decode_name(pin_names[q_pin_id])
            clk_pin_name = _decode_name(clk_pin_names[clk_local_id]) if clk_local_id < len(clk_pin_names) else f"<invalid_clk_local_{clk_local_id}>"
            inst_name, to_pin = q_pin_name.split(":", 1)
            from_pin = clk_pin_name.split(":", 1)[1] if ":" in clk_pin_name else clk_pin_name

            ieda_q_r = ieda_pin_map.get((q_pin_name, "Max", "Rise"), {})
            ieda_q_f = ieda_pin_map.get((q_pin_name, "Max", "Fall"), {})
            ieda_arc_r = ieda_arc_map.get((inst_name, from_pin, to_pin, "Rise"), {})
            ieda_arc_f = ieda_arc_map.get((inst_name, from_pin, to_pin, "Fall"), {})

            rows.append(
                {
                    "level0_arc_index": arc_index,
                    "clk_local_id": clk_local_id,
                    "clk_pin_name": clk_pin_name,
                    "q_pin_id": q_pin_id,
                    "q_pin_name": q_pin_name,
                    "q_is_start_point": q_pin_id in start_point_set,
                    "lib_cell_idx": lib_cell_idx,
                    "lib_arc_idx": lib_arc_idx,
                    "timing_sense": timing_sense,
                    "timing_sense_name": _timing_sense_name(timing_sense),
                    "timing_type": timing_type,
                    "timing_type_name": _timing_type_name(timing_type),
                    "ieda_clk_r_aat_ps": float(exported_clk_r_aat[clk_local_id]) if exported_clk_r_aat is not None and clk_local_id < len(exported_clk_r_aat) else float("nan"),
                    "ieda_clk_f_aat_ps": float(exported_clk_f_aat[clk_local_id]) if exported_clk_f_aat is not None and clk_local_id < len(exported_clk_f_aat) else float("nan"),
                    "py_clk_r_slew_ps": float(py_clk_r_slew[clk_local_id]) if clk_local_id < len(py_clk_r_slew) else None,
                    "py_clk_f_slew_ps": float(py_clk_f_slew[clk_local_id]) if clk_local_id < len(py_clk_f_slew) else None,
                    "py_q_r_aat_ps": float(py_pin_r_aat[q_pin_id]),
                    "py_q_f_aat_ps": float(py_pin_f_aat[q_pin_id]),
                    "py_q_r_slew_ps": float(py_pin_r_slew[q_pin_id]),
                    "py_q_f_slew_ps": float(py_pin_f_slew[q_pin_id]),
                    "golden_load_r_pf": float(golden_load_r[arc_index]) if golden_load_r is not None else float("nan"),
                    "golden_load_f_pf": float(golden_load_f[arc_index]) if golden_load_f is not None else float("nan"),
                    "golden_clk2q_r_delay_ps": float(golden_r_delay[arc_index]) if golden_r_delay is not None else float("nan"),
                    "golden_clk2q_f_delay_ps": float(golden_f_delay[arc_index]) if golden_f_delay is not None else float("nan"),
                    "golden_q_r_slew_ps": float(golden_r_slew[arc_index]) if golden_r_slew is not None else float("nan"),
                    "golden_q_f_slew_ps": float(golden_f_slew[arc_index]) if golden_f_slew is not None else float("nan"),
                    "ieda_q_r_aat_ps": float(ieda_q_r.get("at", float("nan"))) * 1000.0 if "at" in ieda_q_r else float("nan"),
                    "ieda_q_f_aat_ps": float(ieda_q_f.get("at", float("nan"))) * 1000.0 if "at" in ieda_q_f else float("nan"),
                    "ieda_q_r_slew_ps": float(ieda_q_r.get("slew_ns", float("nan"))) * 1000.0 if "slew_ns" in ieda_q_r else float("nan"),
                    "ieda_q_f_slew_ps": float(ieda_q_f.get("slew_ns", float("nan"))) * 1000.0 if "slew_ns" in ieda_q_f else float("nan"),
                    "ieda_clk2q_r_delay_ps": float(ieda_arc_r.get("delay_ns", float("nan"))) * 1000.0 if "delay_ns" in ieda_arc_r else float("nan"),
                    "ieda_clk2q_f_delay_ps": float(ieda_arc_f.get("delay_ns", float("nan"))) * 1000.0 if "delay_ns" in ieda_arc_f else float("nan"),
                    "ieda_clk2q_r_in_slew_ps": float(ieda_arc_r.get("in_slew_ns", float("nan"))) * 1000.0 if "in_slew_ns" in ieda_arc_r else float("nan"),
                    "ieda_clk2q_f_in_slew_ps": float(ieda_arc_f.get("in_slew_ns", float("nan"))) * 1000.0 if "in_slew_ns" in ieda_arc_f else float("nan"),
                }
            )

        rows.sort(
            key=lambda row: abs(row["py_q_r_aat_ps"] - row["ieda_q_r_aat_ps"])
            + abs(row["py_q_f_aat_ps"] - row["ieda_q_f_aat_ps"]),
            reverse=True,
        )

        csv_filename = "%s/%s_clk2q_level0_arc_report.csv" % (
            self.params.result_dir,
            self.params.design_name(),
        )
        summary_filename = "%s/%s_clk2q_level0_arc_summary.json" % (
            self.params.result_dir,
            self.params.design_name(),
        )
        fieldnames = list(rows[0].keys()) if rows else []
        if rows:
            with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        summary = {
            "num_level0_arcs": len(rows),
            "num_q_start_points": sum(1 for row in rows if row["q_is_start_point"]),
            "num_zero_ieda_clk_r_aat": sum(1 for row in rows if abs(row["ieda_clk_r_aat_ps"]) < 1e-6),
            "num_zero_ieda_clk_f_aat": sum(1 for row in rows if abs(row["ieda_clk_f_aat_ps"]) < 1e-6),
            "num_q_zero_py_r_aat": sum(1 for row in rows if abs(row["py_q_r_aat_ps"]) < 1e-6),
            "num_q_zero_py_f_aat": sum(1 for row in rows if abs(row["py_q_f_aat_ps"]) < 1e-6),
            "num_missing_ieda_q_slew": sum(
                1
                for row in rows
                if not np.isfinite(row["ieda_q_r_slew_ps"]) or not np.isfinite(row["ieda_q_f_slew_ps"])
            ),
            "num_missing_ieda_clk2q_delay": sum(
                1
                for row in rows
                if not np.isfinite(row["ieda_clk2q_r_delay_ps"]) and not np.isfinite(row["ieda_clk2q_f_delay_ps"])
            ),
            "q_startpoint_py_vs_golden_slew_mae_ps": (
                float(
                    np.mean(
                        [
                            0.5
                            * (
                                abs(row["py_q_r_slew_ps"] - row["golden_q_r_slew_ps"])
                                + abs(row["py_q_f_slew_ps"] - row["golden_q_f_slew_ps"])
                            )
                            for row in rows
                            if row["q_is_start_point"]
                            and np.isfinite(row["golden_q_r_slew_ps"])
                            and np.isfinite(row["golden_q_f_slew_ps"])
                        ]
                    )
                )
                if any(
                    row["q_is_start_point"]
                    and np.isfinite(row["golden_q_r_slew_ps"])
                    and np.isfinite(row["golden_q_f_slew_ps"])
                    for row in rows
                )
                else None
            ),
            "q_startpoint_ieda_vs_golden_slew_mae_ps": (
                float(
                    np.mean(
                        [
                            0.5
                            * (
                                abs(row["ieda_q_r_slew_ps"] - row["golden_q_r_slew_ps"])
                                + abs(row["ieda_q_f_slew_ps"] - row["golden_q_f_slew_ps"])
                            )
                            for row in rows
                            if row["q_is_start_point"]
                            and np.isfinite(row["ieda_q_r_slew_ps"])
                            and np.isfinite(row["ieda_q_f_slew_ps"])
                            and np.isfinite(row["golden_q_r_slew_ps"])
                            and np.isfinite(row["golden_q_f_slew_ps"])
                        ]
                    )
                )
                if any(
                    row["q_is_start_point"]
                    and np.isfinite(row["ieda_q_r_slew_ps"])
                    and np.isfinite(row["ieda_q_f_slew_ps"])
                    and np.isfinite(row["golden_q_r_slew_ps"])
                    and np.isfinite(row["golden_q_f_slew_ps"])
                    for row in rows
                )
                else None
            ),
            "worst_q_aat_delta_pin": rows[0]["q_pin_name"] if rows else None,
            "worst_q_aat_delta_ps": (
                abs(rows[0]["py_q_r_aat_ps"] - rows[0]["ieda_q_r_aat_ps"])
                + abs(rows[0]["py_q_f_aat_ps"] - rows[0]["ieda_q_f_aat_ps"])
            )
            if rows
            else None,
        }
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logging.info(f"clk2q level-0 arc report 已写入CSV文件: {csv_filename}")
        logging.info(f"clk2q level-0 arc 摘要已写入JSON文件: {summary_filename}")
        return rows, summary

    def write_clk2q_clock_pin_debug_dump(self, clk2q_rows, limit=16):
        import json

        ieda_sta = getattr(self, "_last_ieda_sta", None)
        if ieda_sta is None or not hasattr(ieda_sta.ieda, "dump_clock_pin_timing"):
            logging.warning("clk2q clock-pin debug dump skipped: iEDA debug API unavailable")
            return None

        selected = []
        seen = set()
        target_clk_name = "do_re[8]_reg_p:CK"

        for row in clk2q_rows:
            clk_name = row["clk_pin_name"]
            if clk_name == target_clk_name and clk_name not in seen:
                selected.append(row)
                seen.add(clk_name)
                break

        for row in clk2q_rows:
            clk_name = row["clk_pin_name"]
            if clk_name in seen:
                continue
            selected.append(row)
            seen.add(clk_name)
            if len(selected) >= limit:
                break

        dump_rows = []
        for row in selected:
            clk_name = row["clk_pin_name"]
            dump = dict(ieda_sta.ieda.dump_clock_pin_timing(clk_name))
            dump_rows.append(
                {
                    "clk_pin_name": clk_name,
                    "q_pin_name": row["q_pin_name"],
                    "q_is_start_point": row["q_is_start_point"],
                    "py_q_r_aat_ps": row["py_q_r_aat_ps"],
                    "py_q_f_aat_ps": row["py_q_f_aat_ps"],
                    "ieda_q_r_aat_ps": row["ieda_q_r_aat_ps"],
                    "ieda_q_f_aat_ps": row["ieda_q_f_aat_ps"],
                    "ieda_clk_r_aat_export_ps": row["ieda_clk_r_aat_ps"],
                    "ieda_clk_f_aat_export_ps": row["ieda_clk_f_aat_ps"],
                    "py_clk_r_slew_export_ps": row["py_clk_r_slew_ps"],
                    "py_clk_f_slew_export_ps": row["py_clk_f_slew_ps"],
                    "ieda_live_dump": dump,
                }
            )

        summary = {
            "num_dumped_clock_pins": len(dump_rows),
            "num_zero_export_clk_r_aat": sum(
                1 for row in dump_rows if abs(row["ieda_clk_r_aat_export_ps"]) < 1e-6
            ),
            "num_zero_export_clk_f_aat": sum(
                1 for row in dump_rows if abs(row["ieda_clk_f_aat_export_ps"]) < 1e-6
            ),
            "num_zero_live_max_rise_clock_arrival": sum(
                1
                for row in dump_rows
                if row["ieda_live_dump"].get("max_rise_clock_arrival_ps") in (None, 0.0)
            ),
            "num_zero_live_max_fall_clock_arrival": sum(
                1
                for row in dump_rows
                if row["ieda_live_dump"].get("max_fall_clock_arrival_ps") in (None, 0.0)
            ),
            "target_clk_pin_present": any(row["clk_pin_name"] == target_clk_name for row in dump_rows),
        }

        json_filename = "%s/%s_clk2q_clock_pin_debug.json" % (
            self.params.result_dir,
            self.params.design_name(),
        )
        summary_filename = "%s/%s_clk2q_clock_pin_debug_summary.json" % (
            self.params.result_dir,
            self.params.design_name(),
        )
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(dump_rows, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logging.info(f"clk2q clock-pin debug dump 已写入JSON文件: {json_filename}")
        logging.info(f"clk2q clock-pin debug 摘要已写入JSON文件: {summary_filename}")
        return dump_rows, summary

    def _relaxed_buffer_timing_enabled(self):
        enabled = bool(getattr(self.params, "enable_relaxed_buffer_timing", False))
        if enabled and bool(getattr(self.params, "timing_opt_enabled", False)):
            state = self._segment_count_state_for_virtual_density()
            return state is not None and (
                state.z_param.requires_grad or bool((state.z_param.detach() > 0).any())
            )
        return enabled

    def _segment_live_geometry_enabled(self):
        return bool(
            getattr(self.params, "buffering_segment_live_geometry", False)
        )

    def _segment_virtual_density_enabled(self):
        return bool(
            getattr(self.params, "joint_segment_virtual_density_enabled", False)
            and (
                str(getattr(self.params, "joint_quality_profile", "") or "")
                == "segment_count_direct_joint_v1"
                or bool(getattr(self.params, "timing_opt_enabled", False))
            )
            and len(self.placedb.regions) == 0
        )

    def _segment_count_state_for_virtual_density(self):
        state = getattr(self.data_collections, "buffer_segment_count_state", None)
        if state is None:
            state = getattr(self.data_collections, "buffer_optimization_state", None)
        if state is None or not hasattr(state, "z_param"):
            return None
        return state

    def _active_segment_count_state_for_virtual_density(self):
        if not self._segment_virtual_density_enabled():
            return None
        state = self._segment_count_state_for_virtual_density()
        if state is None:
            return None
        return state if bool((state.z_param.detach() > 0.0).any().item()) else None

    def _timing_geometry_for_pos(self, pos):
        if getattr(self.placedb, "gr_sizing", None) is not None:
            pin_pos = self.op_collections.pin_pos_op(pos)
            return pin_pos, *pin_pos.chunk(2)
        cacheable = bool(torch.is_grad_enabled() and pos.requires_grad)
        position_version = getattr(pos, "_version", None)
        topology_generation = getattr(
            getattr(self.op_collections, "steiner_topo_op", None),
            "topology_generation",
            None,
        )
        cached = getattr(self, "_timing_geometry_cache", None)
        if (
            cacheable
            and isinstance(cached, dict)
            and cached.get("pos") is pos
            and cached.get("position_version") == position_version
            and cached.get("topology_generation") == topology_generation
        ):
            return cached["pin_pos"], cached["new_x"], cached["new_y"]
        pin_pos = self.op_collections.pin_pos_op(pos)
        pin_pos_for_steiner = pin_pos.cpu().contiguous()
        new_x, new_y = self.op_collections.steiner_topo_op(pin_pos_for_steiner)
        if cacheable:
            self._timing_geometry_cache = {
                "pos": pos,
                "position_version": position_version,
                "topology_generation": topology_generation,
                "pin_pos": pin_pos,
                "pin_pos_for_steiner": pin_pos_for_steiner,
                "new_x": new_x,
                "new_y": new_y,
            }
        return pin_pos, new_x, new_y

    def _segment_virtual_buffer_size(self):
        payload = getattr(self.data_collections, "buffer_segment_count_payload", None)
        if payload is None:
            payload = getattr(self.data_collections, "buffer_relaxed_timing_payload", None)
        legal_table = {} if payload is None else dict(payload.get("buffer_legal_table", {}) or {})
        legal_cell_ids = list(legal_table.get("legal_cell_ids", ()) or ())
        state = self._segment_count_state_for_virtual_density()
        if state is None or state.fixed_bsu_index is None:
            raise ValueError("virtual density requires the resolved fixed buffer master")
        bsu_index = int(state.fixed_bsu_index)
        if bsu_index < 0 or bsu_index >= len(legal_cell_ids):
            raise ValueError(
                "virtual density requires buffering_fixed_bsu_index in buffer_legal_table"
            )
        cell_id = int(legal_cell_ids[bsu_index])
        width_table = getattr(self.data_collections, "flat_libcell_width", None)
        height_table = getattr(self.data_collections, "flat_libcell_height", None)
        if width_table is None or height_table is None:
            raise ValueError("virtual density requires flat buffer cell dimensions")
        width = float(torch.as_tensor(width_table)[cell_id].detach().cpu().item())
        height = float(torch.as_tensor(height_table)[cell_id].detach().cpu().item())
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
            raise ValueError("virtual buffer cell dimensions must be finite and positive")
        return width, height

    def combined_density_overflow(self, pos):
        state = self._active_segment_count_state_for_virtual_density()
        if state is None:
            return self.op_collections.density_overflow_op(pos)
        _, x, y = self._timing_geometry_for_pos(pos)
        view = self._build_segment_virtual_density_view(state)
        overflow, maximum = view(pos, x, y, mode="overflow")
        return overflow.reshape(1), maximum.reshape(1)

    def placement_density(self, pos):
        """Use the same density view for GP and density-weight initialization."""
        if len(self.placedb.regions) > 0:
            return self.op_collections.fence_region_density_merged_op(pos)
        state = self._active_segment_count_state_for_virtual_density()
        if state is None:
            return self.op_collections.density_op(pos)
        _, x, y = self._timing_geometry_for_pos(pos)
        return self._build_segment_virtual_density_view(state)(pos, x, y)

    def _build_segment_virtual_density_view(self, state):
        topology_generation = (
            self.op_collections.steiner_topo_op.topology_generation
            if bool(getattr(self.params, "timing_opt_enabled", False)) else None
        )
        if (
            self._virtual_cell_density_view is not None
            and bool(getattr(self.params, "timing_opt_enabled", False))
            and self._virtual_cell_density_view.topology_generation != topology_generation
        ):
            self._virtual_cell_density_view = None
        if self._virtual_cell_density_view is not None:
            return self._virtual_cell_density_view
        from dreamplace.ops.buffer_insertion.virtual_cell_density import (
            VirtualCellDensityOp,
        )

        prepared = getattr(state, "prepared_timing_inputs", None)
        if prepared is None:
            payload = getattr(self.data_collections, "buffer_segment_count_payload", None)
            if payload is None:
                payload = getattr(self.data_collections, "buffer_relaxed_timing_payload", None)
            prepared = None if payload is None else payload.get("prepared_timing_inputs")
        if prepared is None and payload is not None:
            nets = payload.get("nets", ())
            if nets:
                from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
                    build_segment_count_timing_inputs,
                )

                prepared = build_segment_count_timing_inputs(
                    nets,
                    state,
                    dtype=state.z_param.dtype,
                    device=state.z_param.device,
                )
        if prepared is None:
            raise ValueError("virtual density requires prepared segment timing inputs")
        buffer_size_x, buffer_size_y = self._segment_virtual_buffer_size()
        params = self.params

        def density_op_factory(**kwargs):
            return self.build_electric_potential(
                params,
                self.placedb,
                self.data_collections,
                self.num_bins_x,
                self.num_bins_y,
                name="Global placement: virtual buffer density",
                **kwargs,
            )

        self._virtual_cell_density_view = VirtualCellDensityOp(
            segment_state=state,
            prepared_timing_inputs=prepared,
            num_movable_nodes=self.placedb.num_movable_nodes,
            num_terminals=self.placedb.num_terminals,
            num_filler_nodes=self.placedb.num_filler_nodes,
            num_fixed_nodes=(
                self.placedb.num_physical_nodes - self.placedb.num_movable_nodes
            ),
            node_size_x=self.data_collections.node_size_x,
            node_size_y=self.data_collections.node_size_y,
            buffer_size_x=buffer_size_x,
            buffer_size_y=buffer_size_y,
            movable_macro_mask=self.data_collections.movable_macro_mask,
            density_op_factory=density_op_factory,
        )
        self._virtual_cell_density_view.topology_generation = topology_generation
        self.op_collections.virtual_cell_density_op = self._virtual_cell_density_view
        return self._virtual_cell_density_view

    def _bind_segment_live_geometry(self, provider, *, new_x, new_y, pin_capacitance_by_sense):
        if provider is None or not self._segment_live_geometry_enabled():
            return None
        bind = getattr(provider, "bind_live_geometry", None)
        if not callable(bind):
            raise RuntimeError(
                "segment live geometry requires a compatible dynamic provider"
            )
        rc_timing = self.op_collections.elmore_delay_op
        topology_epoch = int(
            getattr(
                self.data_collections,
                "buffer_segment_timing_topology_epoch",
                0,
            )
        )
        forward_id = (
            topology_epoch,
            int(self.invoke_timing_count),
            id(new_x),
            id(new_y),
        )
        bind(
            new_x=new_x,
            new_y=new_y,
            r_unit=rc_timing.r_unit,
            c_unit=rc_timing.c_unit,
            scale_factor=rc_timing.scale_factor,
            dbu=rc_timing.dbu,
            topology_epoch=topology_epoch,
            forward_id=forward_id,
            require_axis_aligned=False,
            pin_capacitance_by_sense=pin_capacitance_by_sense,
        )
        return forward_id

    def _relaxed_buffer_timing_integration_mode(self):
        return str(
            getattr(
                self.params,
                "relaxed_buffer_timing_integration_mode",
                "precomputed_overlay",
            )
            or "precomputed_overlay"
        )

    def _build_relaxed_buffer_dynamic_net_arc_inputs(self, pin_net_delays):
        if not self._relaxed_buffer_timing_enabled():
            return None
        data_collections = self.data_collections
        buffer_state = getattr(data_collections, "buffer_optimization_state", None)
        if buffer_state is None:
            raise ValueError(
                "enable_relaxed_buffer_timing requires buffer_optimization_state"
            )
        payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
        if payload is None:
            raise ValueError(
                "enable_relaxed_buffer_timing requires buffer_relaxed_timing_payload"
            )

        from dreamplace.ops.net_subgraph_timing import (
            build_relaxed_buffer_dynamic_net_arc_inputs,
        )

        payload = dict(payload)
        payload.setdefault("num_pins", pin_net_delays["rise"].numel())
        return build_relaxed_buffer_dynamic_net_arc_inputs(
            buffer_state=buffer_state,
            **payload,
        )

    def _build_relaxed_buffer_dynamic_net_provider(self):
        if not self._relaxed_buffer_timing_enabled():
            return None
        data_collections = self.data_collections
        buffer_state = getattr(data_collections, "buffer_optimization_state", None)
        if buffer_state is None:
            raise ValueError(
                "enable_relaxed_buffer_timing requires buffer_optimization_state"
            )
        payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
        if payload is None:
            raise ValueError(
                "enable_relaxed_buffer_timing requires buffer_relaxed_timing_payload"
            )
        metadata = dict(payload.get("metadata", {}) or {}) if isinstance(payload, dict) else {}
        if metadata.get("state_kind") == "segment_count" or hasattr(buffer_state, "z_param"):
            from dreamplace.ops.buffer_insertion.contract import BufferFamilyContract
            from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
                SegmentCountDynamicNetProvider,
            )
            from dreamplace.ops.net_subgraph_timing.segment_transfer import (
                build_buffer_device_lut_from_metadata,
                summarize_metadata_liberty_arc_parity,
            )

            segment_state = getattr(
                data_collections,
                "buffer_segment_count_state",
                buffer_state,
            )
            nets = getattr(data_collections, "buffer_segment_count_nets", None)
            if nets is None:
                nets = payload.get("nets") if isinstance(payload, dict) else None
            prepared_timing_inputs = getattr(
                segment_state,
                "prepared_timing_inputs",
                None,
            )
            if prepared_timing_inputs is None and isinstance(payload, dict):
                prepared_timing_inputs = payload.get("prepared_timing_inputs")
            per_size_input_cap = getattr(
                data_collections,
                "buffer_segment_count_per_size_input_cap",
                None,
            )
            if per_size_input_cap is None:
                per_size_input_cap = payload.get("per_size_input_cap")
            per_size_delay = getattr(
                data_collections,
                "buffer_segment_count_per_size_delay",
                None,
            )
            if per_size_delay is None:
                per_size_delay = payload.get("per_size_delay")
            per_size_output_slew = getattr(
                data_collections,
                "buffer_segment_count_per_size_output_slew",
                None,
            )
            if per_size_output_slew is None:
                per_size_output_slew = payload.get("per_size_output_slew")
            if nets is None and prepared_timing_inputs is None:
                raise ValueError(
                    "segment-count relaxed buffer timing requires nets or prepared inputs"
                )
            if (
                per_size_input_cap is None
                or per_size_delay is None
                or per_size_output_slew is None
            ):
                raise ValueError(
                    "segment-count relaxed buffer timing requires per-size tensors"
                )
            affected_net_ids = None
            target_net_raw = os.environ.get("AIMP_BUFFERING_SEGMENT_TRANSFER_TARGET_NET_IDS")
            if target_net_raw:
                affected_net_ids = {
                    int(part.strip())
                    for part in str(target_net_raw).split(",")
                    if part.strip().isdigit()
                }
            timing_backend = str(
                getattr(
                    self.params,
                    "buffering_segment_count_timing_backend",
                    metadata.get(
                        "segment_count_timing_backend",
                        "cpp_cpu_explicit_autograd",
                    ),
                )
                or "cpp_cpu_explicit_autograd"
            )
            segment_transfer_backend = str(
                getattr(
                    self.params,
                    "buffering_segment_transfer_backend",
                    metadata.get(
                        "segment_transfer_backend",
                        "static_size_table",
                    ),
                )
                or "static_size_table"
            )
            profile_enabled = bool(
                getattr(self.params, "buffering_dynamic_provider_profile", False)
            )
            topology_epoch = int(
                getattr(data_collections, "buffer_segment_timing_topology_epoch", 0)
            )
            topology_reason = str(
                getattr(
                    data_collections,
                    "buffer_segment_timing_topology_epoch_reason",
                    "legacy_no_epoch",
                )
            )
            driver_cap_mode = str(
                getattr(self.params, "buffering_segment_driver_cap_mode", "residual")
            )
            provider_cache_key = (
                "segment_count",
                driver_cap_mode,
                topology_epoch,
                str(segment_state.z_param.device),
                str(segment_state.z_param.dtype),
                timing_backend,
                segment_transfer_backend,
                tuple(sorted(affected_net_ids)) if affected_net_ids is not None else None,
                profile_enabled,
                (
                    self.op_collections.steiner_topo_op.topology_generation
                    if bool(getattr(self.params, "timing_opt_enabled", False)) else None
                ),
            )
            cached_provider = getattr(self, "_relaxed_buffer_dynamic_net_provider", None)
            if (
                cached_provider is not None
                and getattr(self, "_relaxed_buffer_dynamic_net_provider_key", None)
                == provider_cache_key
            ):
                cached_provider.update_segment_state(segment_state)
                cached_provider.reset_runtime_metadata()
                cached_provider.metadata.update(
                    {
                        "topology_epoch": topology_epoch,
                        "topology_epoch_reason": topology_reason,
                        "provider_reused": True,
                    }
                )
                if isinstance(payload, dict):
                    payload_metadata = dict(payload.get("metadata", {}) or {})
                    payload_metadata.update(dict(cached_provider.metadata))
                    payload_metadata["segment_transfer_device_lut"] = dict(
                        getattr(
                            cached_provider,
                            "device_lut_metadata",
                            {"status": "unknown"},
                        )
                    )
                    payload["metadata"] = payload_metadata
                    data_collections.buffer_relaxed_timing_payload = payload
                return cached_provider
            buffer_device_lut = None
            device_lut_metadata = {"status": "skipped", "reason": "missing_metadata"}
            source_metadata = getattr(data_collections, "buffering_metadata", None)
            if source_metadata is None:
                source_metadata = getattr(data_collections, "metadata", None)
            if source_metadata is None:
                source_metadata = getattr(data_collections, "pydb", None)
            if source_metadata is not None:
                try:
                    contract = BufferFamilyContract.from_buffering_params(
                        source_metadata,
                        self.params,
                    ).build_artifact()
                    if contract.get("status") == "ok":
                        dtype = torch.as_tensor(per_size_input_cap).dtype
                        buffer_device_lut = build_buffer_device_lut_from_metadata(
                            source_metadata,
                            contract,
                            dtype=dtype,
                        )
                        parity_path = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_JSON"
                        )
                        parity_master = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_MASTER"
                        )
                        parity_slew = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_INPUT_SLEW_PS"
                        )
                        parity_load = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_OUTPUT_LOAD_PF"
                        )
                        parity_delay = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_OPENSTA_DELAY_PS"
                        )
                        parity_out_slew = os.environ.get(
                            "AIMP_BUFFERING_SEGMENT_METADATA_LIBERTY_PARITY_OPENSTA_SLEW_PS"
                        )
                        if (
                            parity_path
                            and parity_master
                            and parity_slew
                            and parity_load
                            and parity_delay
                            and parity_out_slew
                        ):
                            def parse_list(raw):
                                return [
                                    float(part.strip())
                                    for part in str(raw).split(",")
                                    if part.strip()
                                ]

                            opensta_delays = parse_list(parity_delay)
                            opensta_slews = parse_list(parity_out_slew)
                            opensta_samples = [
                                {
                                    "liberty_gate_delay": opensta_delays[index],
                                    "liberty_output_slew": opensta_slews[
                                        min(index, len(opensta_slews) - 1)
                                    ],
                                }
                                for index in range(len(opensta_delays))
                            ]
                            parity = summarize_metadata_liberty_arc_parity(
                                metadata=source_metadata,
                                master_name=str(parity_master),
                                input_slew=float(parity_slew),
                                output_load=float(parity_load),
                                opensta_arc_samples=opensta_samples,
                            )
                            os.makedirs(
                                os.path.dirname(os.path.abspath(parity_path)),
                                exist_ok=True,
                            )
                            with open(parity_path, "w", encoding="utf-8") as stream:
                                json.dump(
                                    {
                                        "artifact": "segment_transfer_metadata_liberty_parity",
                                        "artifact_version": 1,
                                        "summary": parity,
                                    },
                                    stream,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                                stream.write("\n")
                        device_lut_metadata = {
                            "status": "ok",
                            "source": buffer_device_lut.source,
                            "legal_size_count": int(
                                buffer_device_lut.tensors(
                                    dtype=dtype,
                                    device=torch.as_tensor(per_size_input_cap).device,
                                )["input_cap_by_size"].numel()
                            ),
                        }
                    else:
                        device_lut_metadata = {
                            "status": "skipped",
                            "reason": "buffer_family_contract_not_ok",
                            "contract": contract,
                        }
                except Exception as exc:
                    device_lut_metadata = {
                        "status": "skipped",
                        "reason": "buffer_device_lut_build_failed",
                        "error": str(exc),
                    }
            provider = SegmentCountDynamicNetProvider(
                nets=nets,
                segment_state=segment_state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                affected_net_ids=affected_net_ids,
                buffer_device_lut=buffer_device_lut,
                backend=timing_backend,
                segment_transfer_backend=segment_transfer_backend,
                profile_enabled=profile_enabled,
                prepared_timing_inputs=prepared_timing_inputs,
                driver_cap_mode=driver_cap_mode,
            )
            provider.device_lut_metadata = dict(device_lut_metadata)
            provider.metadata.update(
                {
                    "topology_epoch": topology_epoch,
                    "topology_epoch_reason": topology_reason,
                    "provider_reused": False,
                }
            )
            self._relaxed_buffer_dynamic_net_provider = provider
            self._relaxed_buffer_dynamic_net_provider_key = provider_cache_key
            if isinstance(payload, dict):
                metadata = dict(payload.get("metadata", {}) or {})
                metadata.update(dict(provider.metadata))
                metadata["segment_transfer_device_lut"] = device_lut_metadata
                payload["metadata"] = metadata
                data_collections.buffer_relaxed_timing_payload = payload
            logging.info(
                "segment_transfer_device_lut metadata: %s",
                device_lut_metadata,
            )
            return provider

        from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
            RelaxedBufferDynamicNetProvider,
        )
        from dreamplace.ops.buffer_insertion.contract import BufferFamilyContract
        from dreamplace.ops.net_subgraph_timing.segment_transfer import (
            build_buffer_device_lut_from_metadata,
        )

        buffer_device_lut = None
        device_lut_metadata = {"status": "skipped", "reason": "missing_metadata"}
        source_metadata = getattr(data_collections, "buffering_metadata", None)
        if source_metadata is None:
            source_metadata = getattr(data_collections, "metadata", None)
        if source_metadata is None:
            source_metadata = getattr(data_collections, "pydb", None)
        if source_metadata is not None:
            try:
                contract = BufferFamilyContract.from_buffering_params(
                    source_metadata,
                    self.params,
                ).build_artifact()
                if contract.get("status") == "ok":
                    per_size_input_cap = payload.get("per_size_input_cap")
                    dtype = torch.as_tensor(per_size_input_cap).dtype
                    buffer_device_lut = build_buffer_device_lut_from_metadata(
                        source_metadata,
                        contract,
                        dtype=dtype,
                    )
                    device_lut_metadata = {
                        "status": "ok",
                        "source": buffer_device_lut.source,
                        "legal_size_count": len(contract.get("legal_cell_ids", [])),
                    }
                else:
                    device_lut_metadata = {
                        "status": "skipped",
                        "reason": "buffer_family_contract_not_ok",
                    }
            except Exception as exc:
                device_lut_metadata = {
                    "status": "skipped",
                    "reason": "buffer_device_lut_build_failed",
                    "error": str(exc),
                }

        candidate_activation = buffer_state.candidate_bu()
        candidate_forward_backend = (
            "cuda_explicit_autograd"
            if candidate_activation.is_cuda
            else "native_explicit_autograd"
        )
        provider_cache_key = (
            "candidate",
            id(payload),
            id(buffer_state),
            str(candidate_activation.device),
            str(candidate_activation.dtype),
            candidate_forward_backend,
        )
        cached_provider = getattr(self, "_relaxed_buffer_dynamic_net_provider", None)
        if (
            isinstance(cached_provider, RelaxedBufferDynamicNetProvider)
            and getattr(self, "_relaxed_buffer_dynamic_net_provider_key", None)
            == provider_cache_key
        ):
            cached_provider.update_buffer_state(buffer_state)
            cached_provider.reset_runtime_metadata()
            cached_provider.metadata["provider_reused"] = True
            if isinstance(payload, dict):
                payload_metadata = dict(payload.get("metadata", {}) or {})
                payload_metadata.update(dict(cached_provider.metadata))
                payload["metadata"] = payload_metadata
                data_collections.buffer_relaxed_timing_payload = payload
            return cached_provider

        provider = RelaxedBufferDynamicNetProvider(
            static_payload=dict(payload),
            buffer_state=buffer_state,
            buffer_device_lut=buffer_device_lut,
            forward_backend=candidate_forward_backend,
        )
        provider.metadata["provider_reused"] = False
        self._relaxed_buffer_dynamic_net_provider = provider
        self._relaxed_buffer_dynamic_net_provider_key = provider_cache_key
        if isinstance(payload, dict):
            payload_metadata = dict(payload.get("metadata", {}) or {})
            payload_metadata.update(dict(provider.metadata))
            payload_metadata["candidate_buffer_device_lut"] = device_lut_metadata
            payload["metadata"] = payload_metadata
            data_collections.buffer_relaxed_timing_payload = payload
        return provider

    def timing_obj(self, pos, surrogate_mode=None):
        """
        @brief Compute objective and perform detailed timing analysis for debugging.
        @param pos locations of cells
        @param surrogate_mode optional per-call timing model mode:
            - mixed/auto/default: clk2q uses LUT, later cell arcs use surrogate when available
            - lut_only: all timing arc evaluations use LUT
            - surrogate_only: all timing arc evaluations use surrogate when available
        @return objective value
        """
        # ==============================================================================
        # --- 步骤 1: Python端计算，获取所有时序参数 ---
        # ==============================================================================
        timing_started_at = time.time()
        profile_started_at = time.perf_counter()
        profile_iteration = int(self.invoke_timing_count)
        timing_profile_enabled = self._should_write_timing_obj_profile(profile_iteration)
        timing_artifact_enabled = self._should_write_timing_artifacts(profile_iteration)
        pin_detail_enabled = self._should_write_pin_violation_detail(profile_iteration)
        violation_log_enabled = not self._production_fast_loop_enabled()
        profile_record = {
            "iteration": profile_iteration,
            "surrogate_mode": surrogate_mode,
            "production_fast_loop": self._production_fast_loop_enabled(),
            "timing_obj_profile_interval": self._timing_obj_profile_interval(),
            "timing_artifact_write_enabled": bool(timing_artifact_enabled),
            "pin_violation_detail_enabled": bool(pin_detail_enabled),
            "violation_log_enabled": bool(violation_log_enabled),
            "steiner_forward_ms": 0.0,
            "pin_caps_ms": 0.0,
            "pin_caps_debug_copy_ms": 0.0,
            "elmore_delay_ms": 0.0,
            "timing_propagation_ms": 0.0,
            "pin_slack_get_ms": 0.0,
            "critical_endpoint_artifact_ms": 0.0,
            "timing_profile_artifact_ms": 0.0,
            "timing_parity_artifact_ms": 0.0,
            "slew_cap_limit_ms": 0.0,
            "size_interpolated_pin_profile": {},
            "slew_violation_ms": 0.0,
            "cap_violation_ms": 0.0,
            "leakage_ms": 0.0,
            "violation_detail_csv_ms": 0.0,
            "artifact_write_skipped_by_fast_loop": False,
            "pin_violation_detail_skipped_by_fast_loop": False,
            "active_cone_aux_pin_mask": {
                "enabled": self._should_use_active_cone_aux_pin_masks(),
                "active_cone_aux_pin_mask_applied": False,
            },
            "zero_slew_aux_reuse": {
                "enabled": (
                    self._production_fast_loop_enabled()
                    and not _env_flag("AIMP_DISABLE_REUSE_ZERO_SLEW_AUX")
                ),
                "applied": False,
            },
        }

        def _profile_now():
            return time.perf_counter()

        def _profile_add(name, started_at):
            if timing_profile_enabled:
                profile_record[name] = (
                    float(profile_record.get(name, 0.0))
                    + (_profile_now() - started_at) * 1000.0
                )

        log_timing_stage = not self._production_fast_loop_enabled()

        if surrogate_mode is None:
            surrogate_mode = getattr(self.params, "timing_surrogate_mode", None)
            profile_record["surrogate_mode"] = surrogate_mode
        self._active_timing_surrogate_mode = surrogate_mode
        timing_op = self.op_collections.timing_propagation_op
        previous_log_top_violations = getattr(timing_op, "log_top_violations", True)
        previous_log_stage_progress = getattr(timing_op, "log_stage_progress", True)
        previous_log_violation_shape_mismatch = getattr(
            timing_op,
            "log_violation_shape_mismatch",
            True,
        )
        timing_op.log_top_violations = violation_log_enabled
        timing_op.log_stage_progress = not self._production_fast_loop_enabled()
        timing_op.log_violation_shape_mismatch = not self._production_fast_loop_enabled()
        critical_path_batch = None

        try:
            if log_timing_stage:
                logging.info("timing_obj stage: steiner_forward start")
            write_crash_stage_marker(
                "timing_obj_steiner_forward",
                "start",
                pos_device=str(pos.device),
            )
            steiner_started_at = time.time()
            profile_stage_start = _profile_now()
            pin_pos, new_x, new_y = self._timing_geometry_for_pos(pos)
            geometry_cache = getattr(self, "_timing_geometry_cache", None)
            if isinstance(geometry_cache, dict) and geometry_cache.get("pin_pos") is pin_pos:
                pin_pos_for_steiner = geometry_cache["pin_pos_for_steiner"]
            else:
                pin_pos_for_steiner = pin_pos.cpu().contiguous()
            _profile_add("steiner_forward_ms", profile_stage_start)
            write_crash_stage_marker(
                "timing_obj_steiner_forward",
                "done",
                pos_device=str(pos.device),
                pin_pos_device=str(pin_pos.device),
                steiner_input_device=str(pin_pos_for_steiner.device),
                new_x_device=str(new_x.device),
                new_y_device=str(new_y.device),
            )
            if log_timing_stage:
                logging.info(
                    "timing_obj stage: steiner_forward done %.3f ms",
                    (time.time() - steiner_started_at) * 1000,
                )

            profile_stage_start = _profile_now()
            pin_caps_base, pin_rcaps_base, pin_fcaps_base = self.pin_caps_op(
                self.data_collections.inst_libcell_offset, self.data_collections
            )
            _profile_add("pin_caps_ms", profile_stage_start)
            profile_stage_start = _profile_now()
            refresh_debug_pin_caps_base = self._should_refresh_debug_pin_caps_base(
                pin_caps_base,
                timing_artifact_enabled,
            )
            profile_record["pin_caps_debug_copy_skipped"] = not refresh_debug_pin_caps_base
            if refresh_debug_pin_caps_base:
                self._debug_last_pin_caps_base = pin_caps_base.detach().cpu().numpy()
                self._debug_last_pin_rcaps_base = pin_rcaps_base.detach().cpu().numpy()
                self._debug_last_pin_fcaps_base = pin_fcaps_base.detach().cpu().numpy()
            _profile_add("pin_caps_debug_copy_ms", profile_stage_start)
            elmore_device = new_x.device
            pin_caps_base_for_elmore = self._timing_rc_cap_input_tensor(
                pin_caps_base.to(elmore_device)
            )
            pin_rcaps_base_for_elmore = self._timing_rc_cap_input_tensor(
                pin_rcaps_base.to(elmore_device)
            )
            pin_fcaps_base_for_elmore = self._timing_rc_cap_input_tensor(
                pin_fcaps_base.to(elmore_device)
            )

            # Elmore Delay算子
            if log_timing_stage:
                logging.info("timing_obj stage: elmore_delay start")
            elmore_started_at = time.time()
            profile_stage_start = _profile_now()
            if getattr(self.placedb, "gr_sizing", None) is not None:
                pin_caps, loads, delays, ldelays, betas, impulses = self.op_collections.elmore_delay_op(
                    pin_caps_base_for_elmore, pin_rcaps_base_for_elmore, pin_fcaps_base_for_elmore
                )
            else:
                pin_caps, loads, delays, ldelays, betas, impulses = self.op_collections.elmore_delay_op(
                    new_x,
                    new_y,
                    self.data_collections.net_flat_topo_sort,
                    self.data_collections.net_flat_topo_sort_start,
                    self.data_collections.pin_fa,
                    self.data_collections.flat_pin_to_start,
                    self.data_collections.flat_pin_to,
                    self.data_collections.flat_pin_from,
                    pin_caps_base_for_elmore,
                    pin_rcaps_base_for_elmore,
                    pin_fcaps_base_for_elmore,
                )
            _profile_add("elmore_delay_ms", profile_stage_start)
            if log_timing_stage:
                logging.info(
                    "timing_obj stage: elmore_delay done %.3f ms",
                    (time.time() - elmore_started_at) * 1000,
                )
            def _rise_or_tensor(value):
                return value["rise"] if isinstance(value, dict) else value

            self._maybe_write_rc_root_load_probe(
                pin_rcaps_base=pin_rcaps_base_for_elmore,
                pin_caps_rise=_rise_or_tensor(pin_caps),
                loads_rise=_rise_or_tensor(loads),
            )

            # 时序传播算子 (为获取WNS/TNS和完整的slew/load值，仍然需要运行)
            if log_timing_stage:
                logging.info("timing_obj stage: timing_propagation start")
            write_crash_stage_marker(
                "timing_obj_timing_propagation_forward",
                "start",
                surrogate_mode=surrogate_mode,
            )
            propagation_started_at = time.time()
            profile_stage_start = _profile_now()
            dynamic_net_arc_inputs = None
            dynamic_net_provider = None
            live_geometry_forward_id = None
            if self._relaxed_buffer_timing_enabled():
                integration_mode = self._relaxed_buffer_timing_integration_mode()
                if integration_mode == "dynamic_net_provider":
                    dynamic_net_provider = self._build_relaxed_buffer_dynamic_net_provider()
                    live_geometry_forward_id = self._bind_segment_live_geometry(
                        dynamic_net_provider,
                        new_x=new_x,
                        new_y=new_y,
                        pin_capacitance_by_sense={
                            "base": pin_caps_base_for_elmore,
                            "rise": pin_rcaps_base_for_elmore,
                            "fall": pin_fcaps_base_for_elmore,
                        },
                    )
                elif integration_mode == "precomputed_overlay":
                    dynamic_net_arc_inputs = self._build_relaxed_buffer_dynamic_net_arc_inputs(
                        delays
                    )
                else:
                    raise ValueError(
                        "unsupported relaxed_buffer_timing_integration_mode: "
                        f"{integration_mode}"
                    )
            timing_kwargs = {
                "surrogate_mode": surrogate_mode,
                "dynamic_net_arc_inputs": dynamic_net_arc_inputs,
            }
            if dynamic_net_provider is not None:
                timing_kwargs["dynamic_net_provider"] = dynamic_net_provider
            critical_path_snapshot_owned = False
            if (
                pin_detail_enabled
                and not getattr(timing_op, "_critical_path_snapshot_requested", False)
                and getattr(timing_op, "_critical_path_snapshot", None) is None
            ):
                try:
                    timing_op.request_critical_path_snapshot()
                    critical_path_snapshot_owned = True
                except RuntimeError as exc:
                    logging.warning("skip critical path debug snapshot: %s", exc)
            try:
                wns, tns, ws, ts = timing_op(
                    delays,
                    impulses,
                    loads,
                    **timing_kwargs,
                )
                if critical_path_snapshot_owned:
                    try:
                        critical_path_batch = timing_op.extract_setup_critical_paths(
                            global_k=16,
                            residual_tolerance_ps=1.0e-3,
                        )
                    except Exception:
                        logging.exception("critical path debug extraction failed")
                        timing_op.clear_critical_path_snapshot_request()
                    finally:
                        critical_path_snapshot_owned = False
            except Exception:
                if critical_path_snapshot_owned:
                    timing_op.clear_critical_path_snapshot_request()
                    critical_path_snapshot_owned = False
                raise
            finally:
                if live_geometry_forward_id is not None:
                    dynamic_net_provider.release_live_geometry(
                        forward_id=live_geometry_forward_id,
                    )
            _profile_add("timing_propagation_ms", profile_stage_start)
            write_crash_stage_marker(
                "timing_obj_timing_propagation_forward",
                "done",
                surrogate_mode=surrogate_mode,
            )
            if log_timing_stage:
                logging.info(
                    "timing_obj stage: timing_propagation done %.3f ms",
                    (time.time() - propagation_started_at) * 1000,
                )
            if timing_profile_enabled:
                profile_record.update(self._timing_pruning_profile_fields(timing_op))
            profile_stage_start = _profile_now()
            if self._should_copy_pin_slack_after_timing(timing_artifact_enabled):
                self.data_collections.pin_slack = timing_op.get_pin_slack()
            _profile_add("pin_slack_get_ms", profile_stage_start)
        finally:
            self._active_timing_surrogate_mode = None
        profile_stage_start = _profile_now()
        if timing_artifact_enabled:
            self._write_critical_endpoint_pruning_artifact(
                iteration=self.invoke_timing_count,
            )
        elif self._production_fast_loop_enabled():
            profile_record["artifact_write_skipped_by_fast_loop"] = True
        _profile_add("critical_endpoint_artifact_ms", profile_stage_start)
        profile_stage_start = _profile_now()
        if timing_artifact_enabled:
            self._write_timing_propagation_profile_artifact()
        _profile_add("timing_profile_artifact_ms", profile_stage_start)
        profile_stage_start = _profile_now()
        if timing_artifact_enabled:
            self._write_timing_propagation_parity_artifact()
        _profile_add("timing_parity_artifact_ms", profile_stage_start)
        profile_stage_start = _profile_now()
        aux_slew_pin_mask = None
        aux_cap_pin_mask = None
        if self._should_use_active_cone_aux_pin_masks():
            aux_slew_pin_mask, aux_cap_pin_mask, aux_mask_metadata = (
                self._active_cone_aux_pin_masks(timing_op)
            )
            profile_record["active_cone_aux_pin_mask"].update(aux_mask_metadata)
        aux_limit_pin_mask = None
        if aux_slew_pin_mask is not None and aux_cap_pin_mask is not None:
            aux_limit_pin_mask = aux_slew_pin_mask | aux_cap_pin_mask.to(
                device=aux_slew_pin_mask.device
            )
        reuse_zero_slew_aux = self._should_reuse_zero_slew_aux(timing_op)
        profile_record["zero_slew_aux_reuse"]["applied"] = bool(reuse_zero_slew_aux)
        if timing_profile_enabled:
            self.data_collections._size_interpolated_pin_profile_enabled = True
        if reuse_zero_slew_aux:
            cap_limits = compute_size_interpolated_pin_properties(
                self.data_collections,
                [self.data_collections.flat_lib_pin_cap_limit],
                pin_mask=self._zero_slew_reuse_cap_limit_interpolation_pin_mask(
                    aux_cap_pin_mask,
                    self.output_pin_mask,
                ),
            )[0]
            slew_limits = None
        else:
            slew_limits, cap_limits = compute_size_interpolated_pin_properties(
                self.data_collections,
                [
                    self.data_collections.flat_lib_pin_slew_limit,
                    self.data_collections.flat_lib_pin_cap_limit,
                ],
                pin_mask=aux_limit_pin_mask,
            )
            if slew_limits is None:
                slew_limits = self.data_collections.flat_lib_pin_slew_limit[
                    self.pin2libpin_flat_ids
                ]
        if cap_limits is None:
            cap_limits = self.data_collections.flat_lib_pin_cap_limit[self.pin2libpin_flat_ids]
        _profile_add("slew_cap_limit_ms", profile_stage_start)
        if timing_profile_enabled:
            profile_record["size_interpolated_pin_profile"] = dict(
                getattr(
                    self.data_collections,
                    "_size_interpolated_pin_last_profile",
                    {},
                )
                or {}
            )
        profile_stage_start = _profile_now()
        if reuse_zero_slew_aux:
            timing_op.last_total_slew_violation_tensor = (
                self._reused_zero_slew_aux_tensor(timing_op)
            )
            timing_op.last_total_slew_violation = 0.0
        else:
            timing_op.last_total_slew_violation_tensor = timing_op.get_total_slew_violation_tensor(
                aux_slew_pin_mask if aux_slew_pin_mask is not None else self.inst_pins_mask,
                slew_limits,
            )
            timing_op.last_total_slew_violation = float(
                timing_op.last_total_slew_violation_tensor.detach().item()
            )
        _profile_add("slew_violation_ms", profile_stage_start)
        profile_stage_start = _profile_now()
        timing_op.last_total_cap_violation_tensor = timing_op.get_total_cap_violation_tensor(
            aux_cap_pin_mask if aux_cap_pin_mask is not None else self.output_pin_mask,
            cap_limits,
        )
        timing_op.last_total_cap_violation = float(
            timing_op.last_total_cap_violation_tensor.detach().item() * 1000.0
        )
        _profile_add("cap_violation_ms", profile_stage_start)
        profile_stage_start = _profile_now()
        if self._should_compute_timing_aux_tensor("leakage"):
            timing_op.last_total_leakage_tensor = self.get_total_leakage_tensor()
        else:
            previous_leakage = getattr(timing_op, "last_total_leakage", None)
            if previous_leakage is None:
                initial_leakage = getattr(self.data_collections, "inst_leakage_init", None)
                if initial_leakage is None:
                    timing_op.last_total_leakage_tensor = None
                else:
                    timing_op.last_total_leakage_tensor = initial_leakage.float().clamp(min=0.0).sum().detach()
            else:
                timing_op.last_total_leakage_tensor = torch.as_tensor(
                    float(previous_leakage),
                    device=pos.device,
                    dtype=pos.dtype,
                )
        timing_op.last_total_leakage = (
            None
            if timing_op.last_total_leakage_tensor is None
            else float(timing_op.last_total_leakage_tensor.detach().item())
        )
        _profile_add("leakage_ms", profile_stage_start)

        # Debug: Log top violation pins with names
        profile_stage_start = _profile_now()
        if pin_detail_enabled:
            self._log_violation_details(
                timing_op,
                slew_limits,
                cap_limits,
                critical_path_batch=critical_path_batch,
            )
        elif self._production_fast_loop_enabled():
            profile_record["pin_violation_detail_skipped_by_fast_loop"] = True
        _profile_add("violation_detail_csv_ms", profile_stage_start)
        timing_op.last_runtime_seconds = time.time() - timing_started_at
        if timing_profile_enabled:
            profile_record["timing_obj_ms"] = (time.perf_counter() - profile_started_at) * 1000.0
            self._write_timing_obj_profile_record(profile_record)
        timing_op.log_top_violations = previous_log_top_violations
        timing_op.log_stage_progress = previous_log_stage_progress
        timing_op.log_violation_shape_mismatch = previous_log_violation_shape_mismatch

        # if self.invoke_timing_count % 280 == 0:
        #     self.check_log(wns, tns, ws, ts)
        #     logging.info(f"\n--- [Timing Debug] 第 {self.invoke_timing_count} 次调用 timing_obj ---")
        #     logging.info(f"当前 WNS: {wns.item():.4f}, TNS: {tns.item():.4f}, WS: {ws.item():.4f}, TS: {ts:.4f}")
        self.invoke_timing_count += 1
        return wns, tns, ws, ts

    def _log_violation_details(
        self,
        timing_op,
        slew_limits,
        cap_limits,
        critical_path_batch=None,
    ):
        """Export full per-pin slew/cap detail with a schema aligned to OpenROAD."""
        if not getattr(self.params, "export_pin_violation_csv", False):
            return
        if not hasattr(self, 'placedb') or self.placedb.pin_names is None:
            logging.warning("[Violation Debug] placedb.pin_names not available")
            return

        pin_names = self.placedb.pin_names
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        iteration = int(getattr(self, "invoke_timing_count", 0))

        pin_rtran = getattr(timing_op, "pin_rtran_live", None)
        pin_ftran = getattr(timing_op, "pin_ftran_live", None)
        if pin_rtran is None:
            pin_rtran = timing_op.pin_rtran
        if pin_ftran is None:
            pin_ftran = timing_op.pin_ftran
        if pin_rtran is not None and pin_ftran is not None:
            slew_rows = build_pin_violation_detail_rows(
                pin_names=pin_names,
                pin_mask=self.inst_pins_mask,
                rise_values=pin_rtran,
                fall_values=pin_ftran,
                limits=slew_limits,
                check_type="slew",
                unit="ps",
                output_pin_mask=self.output_pin_mask,
            )
            slew_path = os.path.join(result_dir, f"{self.params.design_name()}_slew_pin_detail.csv")
            iter_slew_path = os.path.join(
                result_dir,
                f"{self.params.design_name()}_slew_pin_detail_iter{iteration:04d}.csv",
            )
            write_pin_violation_detail_csv(slew_path, slew_rows)
            write_pin_violation_detail_csv(iter_slew_path, slew_rows)
            logging.info("[Violation Debug] Wrote %d slew rows to %s", len(slew_rows), slew_path)

        pin_net_cap_rise = getattr(timing_op, "pin_net_cap_rise_live", None)
        pin_net_cap_fall = getattr(timing_op, "pin_net_cap_fall_live", None)
        if pin_net_cap_rise is None:
            pin_net_cap_rise = timing_op.pin_net_cap_rise
        if pin_net_cap_fall is None:
            pin_net_cap_fall = timing_op.pin_net_cap_fall
        if pin_net_cap_rise is not None and pin_net_cap_fall is not None:
            cap_rows = build_pin_violation_detail_rows(
                pin_names=pin_names,
                pin_mask=self.output_pin_mask,
                rise_values=pin_net_cap_rise,
                fall_values=pin_net_cap_fall,
                limits=cap_limits,
                check_type="cap",
                unit="pF",
                output_pin_mask=self.output_pin_mask,
            )
            cap_path = os.path.join(result_dir, f"{self.params.design_name()}_cap_pin_detail.csv")
            iter_cap_path = os.path.join(
                result_dir,
                f"{self.params.design_name()}_cap_pin_detail_iter{iteration:04d}.csv",
            )
            write_pin_violation_detail_csv(cap_path, cap_rows)
            write_pin_violation_detail_csv(iter_cap_path, cap_rows)
            logging.info("[Violation Debug] Wrote %d cap rows to %s", len(cap_rows), cap_path)

        self.write_cell_arc_py_report()
        if critical_path_batch is not None:
            self.write_setup_critical_path_report(critical_path_batch)
        self.write_cell_arc_py_semantic_summary()
        self.write_cell_arc_lut_fingerprint()
        self.write_python_endpoint_slack()
        self.write_endpoint_constraint_debug()
        self.write_backend_endpoint_timing_compare_if_available()
        self.write_first_level_numeric_alignment(slew_limits=slew_limits, cap_limits=cap_limits)
        self.write_input_boundary_alignment()
        self.write_startpoint_arc_rewrite_debug()
        self.write_net_topology_debug(cap_limits=cap_limits)
        self.write_net_sink_cap_debug()
        self.write_first_level_lut_probe()

    def write_setup_critical_path_report(self, batch):
        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        rows = build_critical_path_rows(
            batch=batch,
            timing_op=self.op_collections.timing_propagation_op,
            pin_names=self.placedb.pin_names,
            flat_inst_arcs_by_level=self.data_collections.flat_inst_arcs_by_level,
            flat_libcell_names=getattr(self.placedb, "flat_libcell_names", []),
            flat_libarc_names=getattr(self.placedb, "flat_libarc_names", []),
        )
        output_path = os.path.join(result_dir, f"{design_name}_setup_critical_paths.csv")
        summary_path = os.path.join(
            result_dir,
            f"{design_name}_setup_critical_paths_summary.json",
        )
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_setup_critical_paths_iter{iteration:04d}.csv",
        )
        iter_summary_path = os.path.join(
            result_dir,
            f"{design_name}_setup_critical_paths_summary_iter{iteration:04d}.json",
        )
        summary = write_critical_path_report(
            output_path,
            summary_path,
            batch=batch,
            rows=rows,
        )
        write_critical_path_report(
            iter_output_path,
            iter_summary_path,
            batch=batch,
            rows=rows,
        )
        logging.info(
            "[SetupCriticalPathReport] Wrote %d rows across %d paths to %s",
            len(rows),
            summary["selected_state_count"],
            output_path,
        )
        return output_path

    def write_python_endpoint_slack(self):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_python_endpoint_slack.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_python_endpoint_slack_iter{iteration:04d}.csv",
        )

        rows = self._build_python_endpoint_slack_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_PYTHON_ENDPOINT_SLACK_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[PythonEndpointSlack] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_python_endpoint_slack_rows(self):
        data_collections = getattr(self, "data_collections", None)
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        end_points = _tensor_to_cpu_long_array(getattr(data_collections, "end_points", None))
        if end_points is None or timing_op is None:
            logging.warning("skip python_endpoint_slack: endpoint or timing tensors are unavailable")
            return []

        py_r_aat = _timing_tensor_to_cpu_float_array(timing_op, "pin_rAAT_live_snapshot", "pin_rAAT")
        py_f_aat = _timing_tensor_to_cpu_float_array(timing_op, "pin_fAAT_live_snapshot", "pin_fAAT")
        py_r_rat = _timing_tensor_to_cpu_float_array(timing_op, "pin_rRAT_live_snapshot", "pin_rRAT")
        py_f_rat = _timing_tensor_to_cpu_float_array(timing_op, "pin_fRAT_live_snapshot", "pin_fRAT")
        if py_r_aat is None or py_f_aat is None or py_r_rat is None or py_f_rat is None:
            logging.warning("skip python_endpoint_slack: Python AAT/RAT tensors are unavailable")
            return []

        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        sentinel_threshold = 5e7
        iteration = int(getattr(self, "invoke_timing_count", 0))

        def _array_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return ""
            value = float(values[index])
            return "" if not np.isfinite(value) else value

        rows = []
        for pin_id_value in end_points.tolist():
            pin_id = int(pin_id_value)
            pin_name = pin_names[pin_id] if 0 <= pin_id < len(pin_names) else ""
            py_r_aat_v = _array_value(py_r_aat, pin_id)
            py_f_aat_v = _array_value(py_f_aat, pin_id)
            py_r_rat_v = _array_value(py_r_rat, pin_id)
            py_f_rat_v = _array_value(py_f_rat, pin_id)
            if py_r_aat_v == "" or py_r_rat_v == "":
                py_r_slack = ""
            else:
                py_r_slack = py_r_rat_v - py_r_aat_v
            if py_f_aat_v == "" or py_f_rat_v == "":
                py_f_slack = ""
            else:
                py_f_slack = py_f_rat_v - py_f_aat_v
            finite_slacks = [
                value for value in (py_r_slack, py_f_slack)
                if value != "" and np.isfinite(value)
            ]
            py_min_slack = min(finite_slacks) if finite_slacks else ""
            arrival_values = [value for value in (py_r_aat_v, py_f_aat_v) if value != ""]
            required_values = [value for value in (py_r_rat_v, py_f_rat_v) if value != ""]
            python_arrival_sentinel = any(value <= -sentinel_threshold for value in arrival_values)
            python_required_sentinel = any(value >= sentinel_threshold for value in required_values)
            python_nonfinite = not finite_slacks
            python_negative_slack = bool(py_min_slack != "" and py_min_slack < 0.0)
            rows.append(
                {
                    "pin_id": pin_id,
                    "pin_name": pin_name,
                    "normalized_pin_name": _normalize_join_pin_name(pin_name),
                    "py_r_aat_ps": py_r_aat_v,
                    "py_f_aat_ps": py_f_aat_v,
                    "py_r_rat_ps": py_r_rat_v,
                    "py_f_rat_ps": py_f_rat_v,
                    "py_r_slack_ps": py_r_slack,
                    "py_f_slack_ps": py_f_slack,
                    "py_min_slack_ps": py_min_slack,
                    "python_arrival_sentinel": int(python_arrival_sentinel),
                    "python_required_sentinel": int(python_required_sentinel),
                    "python_nonfinite": int(python_nonfinite),
                    "python_negative_slack": int(python_negative_slack),
                    "iteration": iteration,
                }
            )
        rows.sort(
            key=lambda row: (
                float("inf") if row["py_min_slack_ps"] == "" else float(row["py_min_slack_ps"]),
                row["normalized_pin_name"],
            )
        )
        return rows

    def write_endpoint_constraint_debug(self):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_endpoint_constraint_debug.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_endpoint_constraint_debug_iter{iteration:04d}.csv",
        )

        rows = self._build_endpoint_constraint_debug_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_ENDPOINT_CONSTRAINT_DEBUG_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[EndpointConstraintDebug] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_endpoint_constraint_debug_rows(self):
        data_collections = getattr(self, "data_collections", None)
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        end_points = _tensor_to_cpu_long_array(getattr(data_collections, "end_points", None))
        if end_points is None or timing_op is None:
            logging.warning("skip endpoint_constraint_debug: endpoint or timing tensors are unavailable")
            return []

        exported_r_rat = _tensor_to_cpu_float_array(
            getattr(timing_op, "endpoints_rRAT", None)
            if getattr(timing_op, "endpoints_rRAT", None) is not None
            else getattr(data_collections, "endpoints_rRAT", None)
        )
        exported_f_rat = _tensor_to_cpu_float_array(
            getattr(timing_op, "endpoints_fRAT", None)
            if getattr(timing_op, "endpoints_fRAT", None) is not None
            else getattr(data_collections, "endpoints_fRAT", None)
        )
        py_r_rat = _timing_tensor_to_cpu_float_array(timing_op, "pin_rRAT_live_snapshot", "pin_rRAT")
        py_f_rat = _timing_tensor_to_cpu_float_array(timing_op, "pin_fRAT_live_snapshot", "pin_fRAT")
        constraints = _tensor_to_cpu_long_array(
            getattr(data_collections, "endpoints_constraint_arcs", None)
            if getattr(data_collections, "endpoints_constraint_arcs", None) is not None
            else getattr(timing_op, "endpoints_constraint_arcs", None)
        )
        timing_check_arcs = _tensor_to_cpu_long_array(
            getattr(data_collections, "endpoints_timing_check_arcs", None)
            if getattr(data_collections, "endpoints_timing_check_arcs", None) is not None
            else getattr(timing_op, "endpoints_timing_check_arcs", None)
        )
        if exported_r_rat is None or exported_f_rat is None or py_r_rat is None or py_f_rat is None:
            logging.warning("skip endpoint_constraint_debug: RAT tensors are unavailable")
            return []

        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])
        iteration = int(getattr(self, "invoke_timing_count", 0))

        def _array_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return ""
            return _finite_or_blank(values[index])

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        constraint_by_endpoint = {}
        if constraints is not None and constraints.size > 0:
            for arc in constraints:
                if len(arc) < 5:
                    continue
                out_pin_id = int(arc[1])
                constraint_by_endpoint.setdefault(out_pin_id, []).append(
                    {
                        "clk_pin_id": int(arc[0]),
                        "endpoint_pin_id": out_pin_id,
                        "lib_cell_idx": int(arc[2]),
                        "lib_arc_idx": int(arc[3]),
                        "timing_sense": int(arc[4]),
                    }
                )
        timing_checks_by_endpoint = {}
        if timing_check_arcs is not None and timing_check_arcs.size > 0:
            for arc in timing_check_arcs:
                if len(arc) <= 6:
                    continue
                out_pin_id = int(arc[1])
                timing_checks_by_endpoint.setdefault(out_pin_id, []).append(int(arc[6]))

        setup_debug = self._endpoint_setup_constraint_values(constraint_by_endpoint)
        rows = []
        for endpoint_index, pin_id_value in enumerate(end_points.tolist()):
            pin_id = int(pin_id_value)
            pin_name = pin_names[pin_id] if 0 <= pin_id < len(pin_names) else ""
            exported_r = _array_value(exported_r_rat, endpoint_index)
            exported_f = _array_value(exported_f_rat, endpoint_index)
            py_r = _array_value(py_r_rat, pin_id)
            py_f = _array_value(py_f_rat, pin_id)
            constraint_rows = constraint_by_endpoint.get(pin_id, [])
            timing_check_classes = timing_checks_by_endpoint.get(pin_id, [])
            setup_row = setup_debug.get(pin_id, {})
            constraint_arc_count = len(constraint_rows)
            lib_cell_names = []
            lib_arc_from_pins = []
            lib_arc_to_pins = []
            timing_senses = []
            timing_types = []
            lib_arc_indices = []
            lib_arc_offsets = []
            for constraint in constraint_rows:
                lib_cell_idx = constraint["lib_cell_idx"]
                lib_arc_idx = constraint["lib_arc_idx"]
                lib_arc_indices.append(str(lib_arc_idx))
                lib_arc_offsets.append(str(setup_row.get("arc_offsets_by_idx", {}).get(lib_arc_idx, "")))
                lib_cell_names.append(
                    flat_libcell_names[lib_cell_idx]
                    if 0 <= lib_cell_idx < len(flat_libcell_names)
                    else ""
                )
                lib_arc_from_pins.append(_arc_pin_name(lib_arc_idx, 0))
                lib_arc_to_pins.append(_arc_pin_name(lib_arc_idx, 1))
                timing_senses.append(_timing_sense_name(constraint["timing_sense"]))
                timing_types.append(str(setup_row.get("timing_type_name", "")))
            rows.append(
                {
                    "pin_id": pin_id,
                    "pin_name": pin_name,
                    "normalized_pin_name": _normalize_join_pin_name(pin_name),
                    "endpoint_check_class": _endpoint_check_class(pin_name, constraint_arc_count),
                    "endpoint_pin_role": _endpoint_pin_role(pin_name),
                    "exported_r_rat_ps": exported_r,
                    "exported_f_rat_ps": exported_f,
                    "py_r_rat_ps": py_r,
                    "py_f_rat_ps": py_f,
                    "setup_r_subtract_ps": _numeric_delta(exported_r, py_r),
                    "setup_f_subtract_ps": _numeric_delta(exported_f, py_f),
                    "constraint_arc_count": constraint_arc_count,
                    "timing_check_arc_count": len(timing_check_classes),
                    "timing_check_classes": "|".join(str(value) for value in sorted(set(timing_check_classes))),
                    "query_r_clk_slew_ps": setup_row.get("query_r_clk_slew_ps", ""),
                    "query_f_clk_slew_ps": setup_row.get("query_f_clk_slew_ps", ""),
                    "query_r_data_slew_ps": setup_row.get("query_r_data_slew_ps", ""),
                    "query_f_data_slew_ps": setup_row.get("query_f_data_slew_ps", ""),
                    "r_setup_value_ps": setup_row.get("r_setup_value_ps", ""),
                    "f_setup_value_ps": setup_row.get("f_setup_value_ps", ""),
                    "constraint_lib_arc_indices": "|".join(lib_arc_indices),
                    "constraint_lib_arc_offsets": "|".join(lib_arc_offsets),
                    "constraint_r_setup_values_ps": "|".join(
                        str(
                            setup_row.get("r_setup_values_by_idx", {}).get(
                                constraint["lib_arc_idx"], ""
                            )
                        )
                        for constraint in constraint_rows
                    ),
                    "constraint_f_setup_values_ps": "|".join(
                        str(
                            setup_row.get("f_setup_values_by_idx", {}).get(
                                constraint["lib_arc_idx"], ""
                            )
                        )
                        for constraint in constraint_rows
                    ),
                    "constraint_lib_cell_names": "|".join(lib_cell_names),
                    "constraint_lib_arc_from_pins": "|".join(lib_arc_from_pins),
                    "constraint_lib_arc_to_pins": "|".join(lib_arc_to_pins),
                    "constraint_timing_senses": "|".join(timing_senses),
                    "constraint_timing_types": "|".join(value for value in timing_types if value),
                    "iteration": iteration,
                }
            )
        rows.sort(key=lambda row: (row["normalized_pin_name"], row["pin_id"]))
        return rows

    def _endpoint_setup_constraint_values(self, constraint_by_endpoint):
        if not constraint_by_endpoint:
            return {}
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return {}
        clk_r_slew_ref = _timing_tensor_ref(timing_op, "clk_pin_rtran", "clk_pin_rtran_live")
        clk_f_slew_ref = _timing_tensor_ref(timing_op, "clk_pin_ftran", "clk_pin_ftran_live")
        data_r_slew_ref = _timing_tensor_ref(timing_op, "pin_rtran_live", "pin_rtran")
        data_f_slew_ref = _timing_tensor_ref(timing_op, "pin_ftran_live", "pin_ftran")
        if clk_r_slew_ref is None or clk_f_slew_ref is None or data_r_slew_ref is None or data_f_slew_ref is None:
            return {}

        flat_libarc_info = _tensor_to_cpu_long_array(getattr(self.data_collections, "flat_libarc_info", None))

        def _timing_type_for_arc(lib_arc_idx):
            if flat_libarc_info is None or lib_arc_idx < 0 or lib_arc_idx >= len(flat_libarc_info):
                return ""
            arc_info = flat_libarc_info[lib_arc_idx]
            if len(arc_info) <= 5:
                return ""
            return _timing_type_name(int(arc_info[5]))

        def _arc_offset_for_arc(lib_arc_idx):
            if flat_libarc_info is None or lib_arc_idx < 0 or lib_arc_idx >= len(flat_libarc_info):
                return ""
            arc_info = flat_libarc_info[lib_arc_idx]
            if len(arc_info) <= 3:
                return ""
            return int(arc_info[3])

        result = {}
        for endpoint_pin_id, constraint_rows in constraint_by_endpoint.items():
            if not constraint_rows:
                continue
            try:
                clk_pin_ids = torch.as_tensor(
                    [row["clk_pin_id"] for row in constraint_rows],
                    dtype=torch.long,
                    device=clk_r_slew_ref.device,
                )
                endpoint_pin_ids = torch.as_tensor(
                    [row["endpoint_pin_id"] for row in constraint_rows],
                    dtype=torch.long,
                    device=data_r_slew_ref.device,
                )
                lib_cell_idxs = torch.as_tensor(
                    [row["lib_cell_idx"] for row in constraint_rows],
                    dtype=torch.long,
                    device=data_r_slew_ref.device,
                )
                lib_arc_idxs = torch.as_tensor(
                    [row["lib_arc_idx"] for row in constraint_rows],
                    dtype=torch.long,
                    device=data_r_slew_ref.device,
                )
                timing_senses = torch.as_tensor(
                    [row["timing_sense"] for row in constraint_rows],
                    dtype=torch.long,
                    device=data_r_slew_ref.device,
                )
                query_r_clk = clk_r_slew_ref[clk_pin_ids]
                query_f_clk = clk_f_slew_ref[clk_pin_ids]
                query_r_data = data_r_slew_ref[endpoint_pin_ids]
                query_f_data = data_f_slew_ref[endpoint_pin_ids]
                with torch.no_grad():
                    rr_setup = timing_op.r_setup_entry(
                        lib_cell_idxs,
                        query_r_clk.to(device=data_r_slew_ref.device),
                        query_r_data,
                        lib_arc_idxs,
                    )
                    ff_setup = timing_op.f_setup_entry(
                        lib_cell_idxs,
                        query_f_clk.to(device=data_r_slew_ref.device),
                        query_f_data,
                        lib_arc_idxs,
                    )
                    fr_setup = timing_op.r_setup_entry(
                        lib_cell_idxs,
                        query_f_clk.to(device=data_r_slew_ref.device),
                        query_r_data,
                        lib_arc_idxs,
                    )
                    rf_setup = timing_op.f_setup_entry(
                        lib_cell_idxs,
                        query_r_clk.to(device=data_r_slew_ref.device),
                        query_f_data,
                        lib_arc_idxs,
                    )
                    is_pos_unate = timing_senses == 1
                    is_neg_unate = timing_senses == -1
                    r_non_unate = smooth_max(rr_setup, fr_setup, alpha=SMOOTH_MAX_ALPHA)
                    f_non_unate = smooth_max(ff_setup, rf_setup, alpha=SMOOTH_MAX_ALPHA)
                    r_setup = torch.where(
                        is_pos_unate,
                        rr_setup,
                        torch.where(is_neg_unate, fr_setup, r_non_unate),
                    )
                    f_setup = torch.where(
                        is_pos_unate,
                        ff_setup,
                        torch.where(is_neg_unate, rf_setup, f_non_unate),
                    )
                r_setup_np = r_setup.detach().cpu().float().numpy()
                f_setup_np = f_setup.detach().cpu().float().numpy()
                query_r_clk_np = query_r_clk.detach().cpu().float().numpy()
                query_f_clk_np = query_f_clk.detach().cpu().float().numpy()
                query_r_data_np = query_r_data.detach().cpu().float().numpy()
                query_f_data_np = query_f_data.detach().cpu().float().numpy()
                lib_arc_idx_values = [row["lib_arc_idx"] for row in constraint_rows]
                result[endpoint_pin_id] = {
                    "query_r_clk_slew_ps": _finite_or_blank(np.max(query_r_clk_np)),
                    "query_f_clk_slew_ps": _finite_or_blank(np.max(query_f_clk_np)),
                    "query_r_data_slew_ps": _finite_or_blank(np.max(query_r_data_np)),
                    "query_f_data_slew_ps": _finite_or_blank(np.max(query_f_data_np)),
                    "r_setup_value_ps": _finite_or_blank(np.max(r_setup_np)),
                    "f_setup_value_ps": _finite_or_blank(np.max(f_setup_np)),
                    "arc_offsets_by_idx": {
                        lib_arc_idx: _arc_offset_for_arc(lib_arc_idx)
                        for lib_arc_idx in lib_arc_idx_values
                    },
                    "r_setup_values_by_idx": {
                        lib_arc_idx: _finite_or_blank(r_setup_np[index])
                        for index, lib_arc_idx in enumerate(lib_arc_idx_values)
                    },
                    "f_setup_values_by_idx": {
                        lib_arc_idx: _finite_or_blank(f_setup_np[index])
                        for index, lib_arc_idx in enumerate(lib_arc_idx_values)
                    },
                    "timing_type_name": "|".join(
                        sorted(
                            {
                                _timing_type_for_arc(row["lib_arc_idx"])
                                for row in constraint_rows
                                if _timing_type_for_arc(row["lib_arc_idx"])
                            }
                        )
                    ),
                }
            except Exception:
                logging.exception("endpoint_constraint_debug setup value probe failed for endpoint %s", endpoint_pin_id)
        return result

    def write_cell_arc_py_report(self, max_rows=4096):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_cell_arc_py_report.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_cell_arc_py_report_iter{iteration:04d}.csv",
        )

        rows = self._build_cell_arc_py_report_rows(max_rows=max_rows)
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_CELL_ARC_PY_REPORT_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[CellArcPyReport] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_cell_arc_py_report_rows(self, max_rows=4096):
        data_collections = getattr(self, "data_collections", None)
        timing_op = self.op_collections.timing_propagation_op
        op_elmore = self.op_collections.elmore_delay_op
        flat_arcs = getattr(data_collections, "flat_inst_arcs_by_level", None)
        if flat_arcs is None:
            return []
        flat_arcs_np = flat_arcs.detach().cpu().numpy() if hasattr(flat_arcs, "detach") else np.asarray(flat_arcs)
        if flat_arcs_np.size == 0:
            return []

        py_cell_arc_rr_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_rr_delays", None))
        py_cell_arc_ff_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_ff_delays", None))
        py_cell_arc_rf_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_rf_delays", None))
        py_cell_arc_fr_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_fr_delays", None))
        if (
            py_cell_arc_rr_delays is None
            or py_cell_arc_ff_delays is None
            or py_cell_arc_rf_delays is None
            or py_cell_arc_fr_delays is None
        ):
            logging.warning("skip cell_arc_py_report: Python cell arc delay tensors are unavailable")
            return []

        py_pin_r_slew = _timing_tensor_to_cpu_float_array(timing_op, "pin_rtran_live", "pin_rtran")
        py_pin_f_slew = _timing_tensor_to_cpu_float_array(timing_op, "pin_ftran_live", "pin_ftran")
        py_pin_r_load = _tensor_to_cpu_float_array(getattr(op_elmore.loads, "get", lambda _name: None)("rise"))
        py_pin_f_load = _tensor_to_cpu_float_array(getattr(op_elmore.loads, "get", lambda _name: None)("fall"))
        if py_pin_r_slew is None or py_pin_f_slew is None or py_pin_r_load is None or py_pin_f_load is None:
            logging.warning("skip cell_arc_py_report: Python pin slew/load tensors are unavailable")
            return []

        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])
        flat_libarc_info = _tensor_to_cpu_long_array(getattr(data_collections, "flat_libarc_info", None))
        max_rows = max(1, int(max_rows))

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        rows = []
        for arc_index, arc in enumerate(flat_arcs_np):
            if len(arc) < 6:
                continue
            if arc_index >= len(py_cell_arc_rr_delays):
                continue
            in_pin_id = int(arc[0])
            out_pin_id = int(arc[1])
            lib_cell_idx = int(arc[2])
            lib_arc_idx = int(arc[3])
            timing_sense = int(arc[4])
            timing_type = int(arc[5])
            in_pin_name = pin_names[in_pin_id] if 0 <= in_pin_id < len(pin_names) else ""
            out_pin_name = pin_names[out_pin_id] if 0 <= out_pin_id < len(pin_names) else ""
            is_inverting = timing_sense == -1
            is_non_unate = timing_sense == 0
            if is_inverting:
                output_rise_delay = py_cell_arc_fr_delays[arc_index]
                output_fall_delay = py_cell_arc_rf_delays[arc_index]
                output_rise_slew = py_pin_f_slew[in_pin_id]
                output_fall_slew = py_pin_r_slew[in_pin_id]
            elif is_non_unate:
                output_rise_delay = max(py_cell_arc_rr_delays[arc_index], py_cell_arc_fr_delays[arc_index])
                output_fall_delay = max(py_cell_arc_ff_delays[arc_index], py_cell_arc_rf_delays[arc_index])
                output_rise_slew = max(py_pin_r_slew[in_pin_id], py_pin_f_slew[in_pin_id])
                output_fall_slew = max(py_pin_r_slew[in_pin_id], py_pin_f_slew[in_pin_id])
            else:
                output_rise_delay = py_cell_arc_rr_delays[arc_index]
                output_fall_delay = py_cell_arc_ff_delays[arc_index]
                output_rise_slew = py_pin_r_slew[in_pin_id]
                output_fall_slew = py_pin_f_slew[in_pin_id]
            arc_offset = ""
            if flat_libarc_info is not None and 0 <= lib_arc_idx < len(flat_libarc_info) and flat_libarc_info.shape[1] > 3:
                arc_offset = int(flat_libarc_info[lib_arc_idx, 3])
            timing_sense_label = _timing_sense_name(timing_sense)
            timing_type_label = _timing_type_name(timing_type)
            rows.append(
                {
                    "arc_index": arc_index,
                    "arc_join_key": (
                        f"{_normalize_join_pin_name(in_pin_name)}->"
                        f"{_normalize_join_pin_name(out_pin_name)}:"
                        f"{timing_sense_label}:{timing_type_label}"
                    ),
                    "in_pin_id": in_pin_id,
                    "in_pin_name": in_pin_name,
                    "in_pin_normalized": _normalize_join_pin_name(in_pin_name),
                    "out_pin_id": out_pin_id,
                    "out_pin_name": out_pin_name,
                    "out_pin_normalized": _normalize_join_pin_name(out_pin_name),
                    "lib_cell_idx": lib_cell_idx,
                    "lib_cell_name": flat_libcell_names[lib_cell_idx] if 0 <= lib_cell_idx < len(flat_libcell_names) else "",
                    "lib_arc_idx": lib_arc_idx,
                    "lib_arc_from_pin": _arc_pin_name(lib_arc_idx, 0),
                    "lib_arc_to_pin": _arc_pin_name(lib_arc_idx, 1),
                    "arc_offset": arc_offset,
                    "timing_sense": timing_sense,
                    "timing_sense_name": timing_sense_label,
                    "timing_type": timing_type,
                    "timing_type_name": timing_type_label,
                    "py_output_rise_delay_ps": _finite_or_blank(output_rise_delay),
                    "py_output_fall_delay_ps": _finite_or_blank(output_fall_delay),
                    "py_output_rise_input_slew_ps": _finite_or_blank(output_rise_slew),
                    "py_output_fall_input_slew_ps": _finite_or_blank(output_fall_slew),
                    "py_output_rise_load_pf": _finite_or_blank(py_pin_r_load[out_pin_id]),
                    "py_output_fall_load_pf": _finite_or_blank(py_pin_f_load[out_pin_id]),
                    "r_delay_value_ps": "",
                    "f_delay_value_ps": "",
                    "r_transition_value_ps": "",
                    "f_transition_value_ps": "",
                    "iteration": int(getattr(self, "invoke_timing_count", 0)),
                    "_source_arc_index": arc_index,
                }
            )
        rows.sort(key=lambda row: (row["arc_join_key"], row["arc_index"]))
        rows = rows[:max_rows]

        if rows:
            source_indices = [int(row["_source_arc_index"]) for row in rows]
            try:
                if hasattr(flat_arcs, "index_select"):
                    index_tensor = torch.as_tensor(source_indices, dtype=torch.long, device=flat_arcs.device)
                    selected_tensor = flat_arcs.index_select(0, index_tensor).long()
                else:
                    selected_tensor = torch.as_tensor(flat_arcs_np[source_indices], dtype=torch.long)
                with torch.no_grad():
                    in_pins = selected_tensor[:, 0]
                    out_pins = selected_tensor[:, 1]
                    lib_cell_idxs = selected_tensor[:, 2]
                    lib_arc_idxs = selected_tensor[:, 3]
                    r_slew_ref = _timing_tensor_ref(timing_op, "pin_rtran_live", "pin_rtran")
                    f_slew_ref = _timing_tensor_ref(timing_op, "pin_ftran_live", "pin_ftran")
                    query_r_cap_ref = _timing_tensor_ref(timing_op, "pin_net_cap_rise_live", "pin_net_cap_rise")
                    query_f_cap_ref = _timing_tensor_ref(timing_op, "pin_net_cap_fall_live", "pin_net_cap_fall")
                    if (
                        r_slew_ref is not None
                        and f_slew_ref is not None
                        and query_r_cap_ref is not None
                        and query_f_cap_ref is not None
                    ):
                        selected_tensor = timing_index_tensor(selected_tensor, r_slew_ref)
                        in_pins = selected_tensor[:, 0]
                        out_pins = selected_tensor[:, 1]
                        lib_cell_idxs = selected_tensor[:, 2]
                        lib_arc_idxs = selected_tensor[:, 3]
                        query_r_slew = r_slew_ref[in_pins]
                        query_f_slew = f_slew_ref[in_pins]
                        query_r_cap = query_r_cap_ref[out_pins]
                        query_f_cap = query_f_cap_ref[out_pins]
                        r_delay = timing_op.r_delay_entry(
                            lib_cell_idxs,
                            query_r_slew,
                            query_r_cap,
                            lib_arc_idxs,
                            out_pins,
                            use_surrogate=False,
                        ).detach().cpu().numpy()
                        f_delay = timing_op.f_delay_entry(
                            lib_cell_idxs,
                            query_f_slew,
                            query_f_cap,
                            lib_arc_idxs,
                            out_pins,
                            use_surrogate=False,
                        ).detach().cpu().numpy()
                        r_transition = timing_op.r_tran_entry(
                            lib_cell_idxs,
                            query_r_slew,
                            query_r_cap,
                            lib_arc_idxs,
                            out_pins,
                            use_surrogate=False,
                        ).detach().cpu().numpy()
                        f_transition = timing_op.f_tran_entry(
                            lib_cell_idxs,
                            query_f_slew,
                            query_f_cap,
                            lib_arc_idxs,
                            out_pins,
                            use_surrogate=False,
                        ).detach().cpu().numpy()
                        for local_idx, row in enumerate(rows):
                            row["r_delay_value_ps"] = _finite_or_blank(r_delay[local_idx])
                            row["f_delay_value_ps"] = _finite_or_blank(f_delay[local_idx])
                            row["r_transition_value_ps"] = _finite_or_blank(r_transition[local_idx])
                            row["f_transition_value_ps"] = _finite_or_blank(f_transition[local_idx])
            except Exception:
                logging.exception("cell_arc_py_report LUT value probe failed; keeping propagated arc values")

        for row in rows:
            row.pop("_source_arc_index", None)
        return rows

    def write_cell_arc_py_semantic_summary(self):
        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_cell_arc_py_semantic_summary.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_cell_arc_py_semantic_summary_iter{iteration:04d}.csv",
        )

        rows = self._build_cell_arc_py_semantic_summary_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_CELL_ARC_PY_SEMANTIC_SUMMARY_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[CellArcPySemanticSummary] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_cell_arc_py_semantic_summary_rows(self):
        data_collections = getattr(self, "data_collections", None)
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if data_collections is None or timing_op is None:
            return []
        flat_arcs = getattr(data_collections, "flat_inst_arcs_by_level", None)
        if flat_arcs is None:
            return []
        flat_arcs_np = flat_arcs.detach().cpu().numpy() if hasattr(flat_arcs, "detach") else np.asarray(flat_arcs)
        if flat_arcs_np.size == 0:
            return []

        py_cell_arc_rr_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_rr_delays", None))
        py_cell_arc_ff_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_ff_delays", None))
        py_cell_arc_rf_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_rf_delays", None))
        py_cell_arc_fr_delays = _tensor_to_cpu_float_array(getattr(timing_op, "cell_arc_fr_delays", None))
        if (
            py_cell_arc_rr_delays is None
            or py_cell_arc_ff_delays is None
            or py_cell_arc_rf_delays is None
            or py_cell_arc_fr_delays is None
        ):
            logging.warning("skip cell_arc_py_semantic_summary: Python cell arc delay tensors are unavailable")
            return []

        op_elmore = getattr(getattr(self, "op_collections", None), "elmore_delay_op", None)
        py_pin_r_slew = _timing_tensor_to_cpu_float_array(timing_op, "pin_rtran_live", "pin_rtran")
        py_pin_f_slew = _timing_tensor_to_cpu_float_array(timing_op, "pin_ftran_live", "pin_ftran")
        py_pin_r_load = _tensor_to_cpu_float_array(getattr(getattr(op_elmore, "loads", None), "get", lambda _name: None)("rise"))
        py_pin_f_load = _tensor_to_cpu_float_array(getattr(getattr(op_elmore, "loads", None), "get", lambda _name: None)("fall"))
        if py_pin_r_slew is None or py_pin_f_slew is None or py_pin_r_load is None or py_pin_f_load is None:
            logging.warning("skip cell_arc_py_semantic_summary: Python pin slew/load tensors are unavailable")
            return []

        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        def _accumulate(groups, key, field, value):
            if value is None:
                return
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if not np.isfinite(number):
                return
            groups[key][field].append(number)

        numeric_fields = [
            "py_output_rise_delay_ps",
            "py_output_fall_delay_ps",
            "py_output_rise_input_slew_ps",
            "py_output_fall_input_slew_ps",
            "py_output_rise_load_pf",
            "py_output_fall_load_pf",
            "query_r_input_slew_ps",
            "query_f_input_slew_ps",
            "query_r_output_cap_pf",
            "query_f_output_cap_pf",
            "r_delay_value_ps",
            "f_delay_value_ps",
            "r_transition_value_ps",
            "f_transition_value_ps",
        ]
        groups = {}
        for arc_index, arc in enumerate(flat_arcs_np):
            if len(arc) < 6 or arc_index >= len(py_cell_arc_rr_delays):
                continue
            in_pin_id = int(arc[0])
            out_pin_id = int(arc[1])
            lib_cell_idx = int(arc[2])
            lib_arc_idx = int(arc[3])
            timing_sense = int(arc[4])
            if not (0 <= in_pin_id < len(py_pin_r_slew) and 0 <= out_pin_id < len(py_pin_r_load)):
                continue
            lib_cell_name = flat_libcell_names[lib_cell_idx] if 0 <= lib_cell_idx < len(flat_libcell_names) else ""
            key = (
                lib_cell_name,
                _arc_pin_name(lib_arc_idx, 0),
                _arc_pin_name(lib_arc_idx, 1),
                timing_sense,
                _timing_sense_name(timing_sense),
            )
            groups.setdefault(key, {field: [] for field in numeric_fields})

            if timing_sense == -1:
                output_rise_delay = py_cell_arc_fr_delays[arc_index]
                output_fall_delay = py_cell_arc_rf_delays[arc_index]
                output_rise_slew = py_pin_f_slew[in_pin_id]
                output_fall_slew = py_pin_r_slew[in_pin_id]
            elif timing_sense == 0:
                output_rise_delay = max(py_cell_arc_rr_delays[arc_index], py_cell_arc_fr_delays[arc_index])
                output_fall_delay = max(py_cell_arc_ff_delays[arc_index], py_cell_arc_rf_delays[arc_index])
                output_rise_slew = max(py_pin_r_slew[in_pin_id], py_pin_f_slew[in_pin_id])
                output_fall_slew = max(py_pin_r_slew[in_pin_id], py_pin_f_slew[in_pin_id])
            else:
                output_rise_delay = py_cell_arc_rr_delays[arc_index]
                output_fall_delay = py_cell_arc_ff_delays[arc_index]
                output_rise_slew = py_pin_r_slew[in_pin_id]
                output_fall_slew = py_pin_f_slew[in_pin_id]

            _accumulate(groups, key, "py_output_rise_delay_ps", output_rise_delay)
            _accumulate(groups, key, "py_output_fall_delay_ps", output_fall_delay)
            _accumulate(groups, key, "py_output_rise_input_slew_ps", output_rise_slew)
            _accumulate(groups, key, "py_output_fall_input_slew_ps", output_fall_slew)
            _accumulate(groups, key, "py_output_rise_load_pf", py_pin_r_load[out_pin_id])
            _accumulate(groups, key, "py_output_fall_load_pf", py_pin_f_load[out_pin_id])

        try:
            r_slew_ref = _timing_tensor_ref(timing_op, "pin_rtran_live", "pin_rtran")
            f_slew_ref = _timing_tensor_ref(timing_op, "pin_ftran_live", "pin_ftran")
            query_r_cap_ref = _timing_tensor_ref(timing_op, "pin_net_cap_rise_live", "pin_net_cap_rise")
            query_f_cap_ref = _timing_tensor_ref(timing_op, "pin_net_cap_fall_live", "pin_net_cap_fall")
            if (
                r_slew_ref is not None
                and f_slew_ref is not None
                and query_r_cap_ref is not None
                and query_f_cap_ref is not None
            ):
                selected_tensor = timing_index_tensor(flat_arcs, r_slew_ref)
                in_pins = selected_tensor[:, 0]
                out_pins = selected_tensor[:, 1]
                lib_cell_idxs = selected_tensor[:, 2]
                lib_arc_idxs = selected_tensor[:, 3]
                query_r_slew = r_slew_ref[in_pins]
                query_f_slew = f_slew_ref[in_pins]
                query_r_cap = query_r_cap_ref[out_pins]
                query_f_cap = query_f_cap_ref[out_pins]
                with torch.no_grad():
                    r_delay = timing_op.r_delay_entry(
                        lib_cell_idxs,
                        query_r_slew,
                        query_r_cap,
                        lib_arc_idxs,
                        out_pins,
                        use_surrogate=False,
                    ).detach().cpu().numpy()
                    f_delay = timing_op.f_delay_entry(
                        lib_cell_idxs,
                        query_f_slew,
                        query_f_cap,
                        lib_arc_idxs,
                        out_pins,
                        use_surrogate=False,
                    ).detach().cpu().numpy()
                    r_transition = timing_op.r_tran_entry(
                        lib_cell_idxs,
                        query_r_slew,
                        query_r_cap,
                        lib_arc_idxs,
                        out_pins,
                        use_surrogate=False,
                    ).detach().cpu().numpy()
                    f_transition = timing_op.f_tran_entry(
                        lib_cell_idxs,
                        query_f_slew,
                        query_f_cap,
                        lib_arc_idxs,
                        out_pins,
                        use_surrogate=False,
                    ).detach().cpu().numpy()
                query_r_slew_np = query_r_slew.detach().cpu().float().numpy()
                query_f_slew_np = query_f_slew.detach().cpu().float().numpy()
                query_r_cap_np = query_r_cap.detach().cpu().float().numpy()
                query_f_cap_np = query_f_cap.detach().cpu().float().numpy()
                for arc_index, arc in enumerate(flat_arcs_np):
                    if len(arc) < 6:
                        continue
                    lib_cell_idx = int(arc[2])
                    lib_arc_idx = int(arc[3])
                    timing_sense = int(arc[4])
                    lib_cell_name = flat_libcell_names[lib_cell_idx] if 0 <= lib_cell_idx < len(flat_libcell_names) else ""
                    key = (
                        lib_cell_name,
                        _arc_pin_name(lib_arc_idx, 0),
                        _arc_pin_name(lib_arc_idx, 1),
                        timing_sense,
                        _timing_sense_name(timing_sense),
                    )
                    if key not in groups:
                        continue
                    _accumulate(groups, key, "query_r_input_slew_ps", query_r_slew_np[arc_index])
                    _accumulate(groups, key, "query_f_input_slew_ps", query_f_slew_np[arc_index])
                    _accumulate(groups, key, "query_r_output_cap_pf", query_r_cap_np[arc_index])
                    _accumulate(groups, key, "query_f_output_cap_pf", query_f_cap_np[arc_index])
                    _accumulate(groups, key, "r_delay_value_ps", r_delay[arc_index])
                    _accumulate(groups, key, "f_delay_value_ps", f_delay[arc_index])
                    _accumulate(groups, key, "r_transition_value_ps", r_transition[arc_index])
                    _accumulate(groups, key, "f_transition_value_ps", f_transition[arc_index])
        except Exception:
            logging.exception("cell_arc_py_semantic_summary LUT value probe failed; keeping propagated arc aggregates")

        rows = []
        for key, values in groups.items():
            row = {
                "lib_cell_name": key[0],
                "lib_arc_from_pin": key[1],
                "lib_arc_to_pin": key[2],
                "timing_sense": key[3],
                "timing_sense_name": key[4],
                "arc_count": len(values["py_output_rise_delay_ps"]),
                "iteration": int(getattr(self, "invoke_timing_count", 0)),
            }
            for field in numeric_fields:
                entries = values[field]
                row[f"mean_{field}"] = _finite_or_blank(sum(entries) / len(entries)) if entries else ""
            rows.append(row)
        rows.sort(key=lambda row: (row["lib_cell_name"], row["lib_arc_from_pin"], row["lib_arc_to_pin"], row["timing_sense"]))
        return rows

    def write_cell_arc_lut_fingerprint(self):
        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_cell_arc_lut_fingerprint.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_cell_arc_lut_fingerprint_iter{iteration:04d}.csv",
        )

        rows = self._build_cell_arc_lut_fingerprint_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_CELL_ARC_LUT_FINGERPRINT_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[CellArcLUTFingerprint] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_cell_arc_lut_fingerprint_rows(self):
        data_collections = getattr(self, "data_collections", None)
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if data_collections is None or timing_op is None:
            return []
        flat_libarc_info = _tensor_to_cpu_long_array(getattr(data_collections, "flat_libarc_info", None))
        if flat_libarc_info is None or flat_libarc_info.size == 0:
            return []

        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        def _lut_stats(luts_info, arc_id, prefix):
            stats = {
                f"{prefix}_trans_dim": "",
                f"{prefix}_cap_dim": "",
                f"{prefix}_trans_min_ps": "",
                f"{prefix}_trans_max_ps": "",
                f"{prefix}_cap_min_pf": "",
                f"{prefix}_cap_max_pf": "",
                f"{prefix}_value_min_ps": "",
                f"{prefix}_value_max_ps": "",
                f"{prefix}_value_sum_ps": "",
                f"{prefix}_value_sample0_ps": "",
                f"{prefix}_value_sample_mid_ps": "",
                f"{prefix}_value_sample_last_ps": "",
            }
            if luts_info is None or arc_id < 0:
                return stats
            trans_dims = getattr(luts_info, "trans_dims_actual", None)
            cap_dims = getattr(luts_info, "cap_dims_actual", None)
            trans_table = getattr(luts_info, "flat_luts_trans_table", None)
            cap_table = getattr(luts_info, "flat_luts_cap_table", None)
            values = getattr(luts_info, "flat_luts_values", None)
            if (
                trans_dims is None
                or cap_dims is None
                or trans_table is None
                or cap_table is None
                or values is None
            ):
                return stats
            if arc_id >= int(trans_dims.shape[0]) or arc_id >= int(cap_dims.shape[0]) or arc_id >= int(values.shape[0]):
                return stats

            trans_dim = int(trans_dims[arc_id].detach().item())
            cap_dim = int(cap_dims[arc_id].detach().item())
            stats[f"{prefix}_trans_dim"] = trans_dim
            stats[f"{prefix}_cap_dim"] = cap_dim
            if trans_dim > 0:
                trans_values = trans_table[arc_id, :trans_dim].detach().cpu().float().numpy()
                stats[f"{prefix}_trans_min_ps"] = _finite_or_blank(trans_values[0])
                stats[f"{prefix}_trans_max_ps"] = _finite_or_blank(trans_values[-1])
            if cap_dim > 0:
                cap_values = cap_table[arc_id, :cap_dim].detach().cpu().float().numpy()
                stats[f"{prefix}_cap_min_pf"] = _finite_or_blank(cap_values[0])
                stats[f"{prefix}_cap_max_pf"] = _finite_or_blank(cap_values[-1])

            value_tensor = values[arc_id].reshape(-1)
            if trans_dim > 0 and cap_dim > 0:
                valid_count = trans_dim * cap_dim
            else:
                valid_count = max(trans_dim, cap_dim, 1)
            valid_count = min(valid_count, int(value_tensor.numel()))
            valid_values = value_tensor[:valid_count]
            value_np = valid_values.detach().cpu().float().numpy()
            value_np = value_np[np.isfinite(value_np)]
            if value_np.size > 0:
                mid_idx = int(value_np.size // 2)
                stats[f"{prefix}_value_min_ps"] = _finite_or_blank(np.min(value_np))
                stats[f"{prefix}_value_max_ps"] = _finite_or_blank(np.max(value_np))
                stats[f"{prefix}_value_sum_ps"] = _finite_or_blank(np.sum(value_np))
                stats[f"{prefix}_value_sample0_ps"] = _finite_or_blank(value_np[0])
                stats[f"{prefix}_value_sample_mid_ps"] = _finite_or_blank(value_np[mid_idx])
                stats[f"{prefix}_value_sample_last_ps"] = _finite_or_blank(value_np[-1])
            return stats

        rows = []
        arcs_info = getattr(timing_op, "arcs_info", None)
        for lib_arc_idx, arc_info in enumerate(flat_libarc_info):
            if len(arc_info) < 6:
                continue
            lib_cell_idx = int(arc_info[2])
            arc_offset = int(arc_info[3])
            timing_sense = int(arc_info[4])
            row = {
                "lib_cell_name": (
                    flat_libcell_names[lib_cell_idx]
                    if 0 <= lib_cell_idx < len(flat_libcell_names)
                    else ""
                ),
                "lib_arc_from_pin": _arc_pin_name(lib_arc_idx, 0),
                "lib_arc_to_pin": _arc_pin_name(lib_arc_idx, 1),
                "timing_sense": timing_sense,
                "timing_sense_name": _timing_sense_name(timing_sense),
                "lib_arc_idx": lib_arc_idx,
                "arc_offset": arc_offset,
                "iteration": int(getattr(self, "invoke_timing_count", 0)),
            }
            row.update(_lut_stats(getattr(arcs_info, "r_delay_luts", None), lib_arc_idx, "r_delay"))
            row.update(_lut_stats(getattr(arcs_info, "f_delay_luts", None), lib_arc_idx, "f_delay"))
            row.update(_lut_stats(getattr(arcs_info, "r_trans_luts", None), lib_arc_idx, "r_transition"))
            row.update(_lut_stats(getattr(arcs_info, "f_trans_luts", None), lib_arc_idx, "f_transition"))
            rows.append(row)
        rows.sort(key=lambda row: (row["lib_cell_name"], row["lib_arc_from_pin"], row["lib_arc_to_pin"], row["timing_sense"], row["lib_arc_idx"]))
        return rows

    def write_first_level_numeric_alignment(self, slew_limits=None, cap_limits=None):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_first_level_numeric_alignment.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_first_level_numeric_alignment_iter{iteration:04d}.csv",
        )

        rows = self._build_first_level_numeric_alignment_rows(slew_limits, cap_limits)
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_FIRST_LEVEL_NUMERIC_ALIGNMENT_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[FirstLevelNumeric] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def write_input_boundary_alignment(self):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_input_boundary.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_input_boundary_iter{iteration:04d}.csv",
        )

        rows = self._build_input_boundary_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_INPUT_BOUNDARY_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[InputBoundary] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_input_boundary_rows(self):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        net2driver_pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "net2driver_pin_map", None))
        net_names = [_decode_name(value) for value in getattr(self.placedb, "net_names", [])]
        output_pin_mask = _tensor_to_cpu_long_array(getattr(self, "output_pin_mask", None))
        data_collections = getattr(self, "data_collections", None)
        timing_op = self.op_collections.timing_propagation_op
        start_points = _tensor_to_cpu_long_array(getattr(data_collections, "start_points", None))
        inrdelays = _tensor_to_cpu_float_array(getattr(data_collections, "inrdelays", None))
        infdelays = _tensor_to_cpu_float_array(getattr(data_collections, "infdelays", None))
        inrtrans = _tensor_to_cpu_float_array(getattr(data_collections, "inrtrans", None))
        inftrans = _tensor_to_cpu_float_array(getattr(data_collections, "inftrans", None))
        py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT_live_snapshot", None))
        py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT_live_snapshot", None))
        py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran_live", None))
        py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran_live", None))
        if py_r_aat is None:
            py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT", None))
        if py_f_aat is None:
            py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT", None))
        if py_r_slew is None:
            py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran", None))
        if py_f_slew is None:
            py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran", None))
        if start_points is None:
            return []

        rows = []
        for local_idx, pin_id in enumerate(start_points.tolist()):
            pin_id = int(pin_id)
            net_id = int(pin2net_map[pin_id]) if pin2net_map is not None and 0 <= pin_id < pin2net_map.size else -1
            driver_pin_id = -1
            if net2driver_pin_map is not None and 0 <= net_id < net2driver_pin_map.size:
                driver_pin_id = int(net2driver_pin_map[net_id])
            rows.append(
                {
                    "pin_id": pin_id,
                    "pin_name": pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "",
                    "normalized_pin_name": _normalize_join_pin_name(pin_names[pin_id]) if 0 <= pin_id < len(pin_names) else "",
                    "net_id": net_id if net_id >= 0 else "",
                    "net_name": net_names[net_id] if 0 <= net_id < len(net_names) else "",
                    "driver_pin_id": driver_pin_id if driver_pin_id >= 0 else "",
                    "driver_pin_name": pin_names[driver_pin_id] if 0 <= driver_pin_id < len(pin_names) else "",
                    "is_start_point": 1,
                    "is_output_pin": int(bool(output_pin_mask is not None and pin_id < len(output_pin_mask) and output_pin_mask[pin_id])),
                    "exported_inrdelay_ps": _finite_or_blank(inrdelays[local_idx] if inrdelays is not None and local_idx < len(inrdelays) else None),
                    "exported_infdelay_ps": _finite_or_blank(infdelays[local_idx] if infdelays is not None and local_idx < len(infdelays) else None),
                    "exported_inrtrans_ps": _finite_or_blank(inrtrans[local_idx] if inrtrans is not None and local_idx < len(inrtrans) else None),
                    "exported_inftrans_ps": _finite_or_blank(inftrans[local_idx] if inftrans is not None and local_idx < len(inftrans) else None),
                    "py_r_aat_ps": _finite_or_blank(py_r_aat[pin_id] if py_r_aat is not None and pin_id < len(py_r_aat) else None),
                    "py_f_aat_ps": _finite_or_blank(py_f_aat[pin_id] if py_f_aat is not None and pin_id < len(py_f_aat) else None),
                    "py_r_slew_ps": _finite_or_blank(py_r_slew[pin_id] if py_r_slew is not None and pin_id < len(py_r_slew) else None),
                    "py_f_slew_ps": _finite_or_blank(py_f_slew[pin_id] if py_f_slew is not None and pin_id < len(py_f_slew) else None),
                    "iteration": int(getattr(self, "invoke_timing_count", 0)),
                }
            )
        return rows

    def write_startpoint_arc_rewrite_debug(self):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_startpoint_arc_rewrite_debug.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_startpoint_arc_rewrite_debug_iter{iteration:04d}.csv",
        )
        summary_path = os.path.join(result_dir, f"{design_name}_startpoint_arc_rewrite_debug_summary.json")
        iter_summary_path = os.path.join(
            result_dir,
            f"{design_name}_startpoint_arc_rewrite_debug_summary_iter{iteration:04d}.json",
        )

        rows, summary = self._build_startpoint_arc_rewrite_debug_rows()
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_STARTPOINT_ARC_REWRITE_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        for path in (summary_path, iter_summary_path):
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(summary, json_file, indent=2, sort_keys=True)
        logging.info("[StartpointRewrite] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_startpoint_arc_rewrite_debug_rows(self):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        net_names = [_decode_name(value) for value in getattr(self.placedb, "net_names", [])]
        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        output_pin_mask = _tensor_to_cpu_long_array(getattr(self, "output_pin_mask", None))
        data_collections = getattr(self, "data_collections", None)
        timing_op = self.op_collections.timing_propagation_op
        start_points = _tensor_to_cpu_long_array(getattr(data_collections, "start_points", None))
        inrdelays = _tensor_to_cpu_float_array(getattr(data_collections, "inrdelays", None))
        infdelays = _tensor_to_cpu_float_array(getattr(data_collections, "infdelays", None))
        py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT_live_snapshot", None))
        py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT_live_snapshot", None))
        if py_r_aat is None:
            py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT", None))
        if py_f_aat is None:
            py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT", None))
        if start_points is None:
            return [], {"status": "missing_start_points"}

        flat_arcs = getattr(data_collections, "flat_inst_arcs_by_level", None)
        level_starts = getattr(data_collections, "flat_inst_arcs_by_level_start", None)
        flat_arcs_np = (
            flat_arcs.detach().cpu().long().numpy()
            if hasattr(flat_arcs, "detach")
            else np.asarray(flat_arcs, dtype=np.int64)
            if flat_arcs is not None
            else np.empty((0, 0), dtype=np.int64)
        )
        level_starts_np = (
            level_starts.detach().cpu().long().numpy()
            if hasattr(level_starts, "detach")
            else np.asarray(level_starts, dtype=np.int64)
            if level_starts is not None
            else np.empty((0,), dtype=np.int64)
        )
        net_flat_arcs = getattr(data_collections, "net_flat_arcs", None)
        net_arcs_np = (
            net_flat_arcs.detach().cpu().long().numpy()
            if hasattr(net_flat_arcs, "detach")
            else np.asarray(net_flat_arcs, dtype=np.int64)
            if net_flat_arcs is not None
            else np.empty((0, 0), dtype=np.int64)
        )

        inst_out_map = {}
        inst_in_map = {}
        if flat_arcs_np is not None and flat_arcs_np.size:
            arc_levels = {}
            if level_starts_np is not None and level_starts_np.size >= 2:
                for level in range(int(level_starts_np.size) - 1):
                    start = int(level_starts_np[level])
                    end = int(level_starts_np[level + 1])
                    for arc_index in range(start, end):
                        arc_levels[arc_index] = level
            for arc_index, arc in enumerate(flat_arcs_np):
                if len(arc) < 2:
                    continue
                in_pin = int(arc[0])
                out_pin = int(arc[1])
                level = int(arc_levels.get(arc_index, -1))
                inst_out_map.setdefault(out_pin, []).append((arc_index, level, arc))
                inst_in_map.setdefault(in_pin, []).append((arc_index, level, arc))

        net_out_map = {}
        net_in_map = {}
        if net_arcs_np is not None and net_arcs_np.size:
            for arc_index, arc in enumerate(net_arcs_np):
                if len(arc) < 2:
                    continue
                in_pin = int(arc[0])
                out_pin = int(arc[1])
                net_out_map.setdefault(out_pin, []).append((arc_index, in_pin, out_pin))
                net_in_map.setdefault(in_pin, []).append((arc_index, in_pin, out_pin))

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        def _pin_name(pin_id):
            return pin_names[pin_id] if 0 <= pin_id < len(pin_names) else ""

        rows = []
        for local_idx, pin_id in enumerate(start_points.tolist()):
            pin_id = int(pin_id)
            inst_out_entries = inst_out_map.get(pin_id, [])
            inst_in_entries = inst_in_map.get(pin_id, [])
            non_level0_entries = [entry for entry in inst_out_entries if int(entry[1]) > 0]
            level0_entries = [entry for entry in inst_out_entries if int(entry[1]) == 0]
            net_out_entries = net_out_map.get(pin_id, [])
            net_in_entries = net_in_map.get(pin_id, [])
            net_self_loop_entries = [entry for entry in net_out_entries if int(entry[1]) == int(entry[2])]
            first_non_level0 = non_level0_entries[0] if non_level0_entries else None
            first_net_out = net_out_entries[0] if net_out_entries else None
            net_id = int(pin2net_map[pin_id]) if pin2net_map is not None and 0 <= pin_id < pin2net_map.size else -1
            exported_r = inrdelays[local_idx] if inrdelays is not None and local_idx < len(inrdelays) else None
            exported_f = infdelays[local_idx] if infdelays is not None and local_idx < len(infdelays) else None
            current_r = py_r_aat[pin_id] if py_r_aat is not None and pin_id < len(py_r_aat) else None
            current_f = py_f_aat[pin_id] if py_f_aat is not None and pin_id < len(py_f_aat) else None
            first_arc = first_non_level0[2] if first_non_level0 is not None else None
            first_arc_index = int(first_non_level0[0]) if first_non_level0 is not None else ""
            first_arc_level = int(first_non_level0[1]) if first_non_level0 is not None else ""
            first_in_pin = int(first_arc[0]) if first_arc is not None and len(first_arc) > 0 else -1
            first_out_pin = int(first_arc[1]) if first_arc is not None and len(first_arc) > 1 else -1
            first_lib_cell_idx = int(first_arc[2]) if first_arc is not None and len(first_arc) > 2 else -1
            first_lib_arc_idx = int(first_arc[3]) if first_arc is not None and len(first_arc) > 3 else -1
            first_timing_sense = int(first_arc[4]) if first_arc is not None and len(first_arc) > 4 else 0
            first_timing_type = int(first_arc[5]) if first_arc is not None and len(first_arc) > 5 else 0
            first_net_arc_index = int(first_net_out[0]) if first_net_out is not None else ""
            first_net_in_pin = int(first_net_out[1]) if first_net_out is not None else -1
            first_net_out_pin = int(first_net_out[2]) if first_net_out is not None else -1
            row = {
                "pin_id": pin_id,
                "pin_name": _pin_name(pin_id),
                "normalized_pin_name": _normalize_join_pin_name(_pin_name(pin_id)),
                "net_id": net_id if net_id >= 0 else "",
                "net_name": net_names[net_id] if 0 <= net_id < len(net_names) else "",
                "is_output_pin": int(bool(output_pin_mask is not None and pin_id < len(output_pin_mask) and output_pin_mask[pin_id])),
                "exported_inrdelay_ps": _finite_or_blank(exported_r),
                "exported_infdelay_ps": _finite_or_blank(exported_f),
                "py_r_aat_ps": _finite_or_blank(current_r),
                "py_f_aat_ps": _finite_or_blank(current_f),
                "max_r_aat_delta_ps": _finite_or_blank(_numeric_delta(current_r, exported_r)),
                "max_f_aat_delta_ps": _finite_or_blank(_numeric_delta(current_f, exported_f)),
                "level0_as_inst_out_arc_count": len(level0_entries),
                "non_level0_as_inst_out_arc_count": len(non_level0_entries),
                "total_as_inst_out_arc_count": len(inst_out_entries),
                "as_inst_in_arc_count": len(inst_in_entries),
                "as_net_out_arc_count": len(net_out_entries),
                "as_net_self_loop_count": len(net_self_loop_entries),
                "as_net_in_arc_count": len(net_in_entries),
                "first_non_level0_inst_arc_index": first_arc_index,
                "first_non_level0_inst_arc_level": first_arc_level,
                "first_non_level0_inst_arc_join_key": (
                    f"{_normalize_join_pin_name(_pin_name(first_in_pin))}->"
                    f"{_normalize_join_pin_name(_pin_name(first_out_pin))}:"
                    f"{_timing_sense_name(first_timing_sense)}:{_timing_type_name(first_timing_type)}"
                    if first_arc is not None
                    else ""
                ),
                "first_non_level0_in_pin_name": _pin_name(first_in_pin),
                "first_non_level0_out_pin_name": _pin_name(first_out_pin),
                "first_non_level0_lib_cell_name": (
                    flat_libcell_names[first_lib_cell_idx]
                    if 0 <= first_lib_cell_idx < len(flat_libcell_names)
                    else ""
                ),
                "first_non_level0_lib_arc_from_pin": _arc_pin_name(first_lib_arc_idx, 0),
                "first_non_level0_lib_arc_to_pin": _arc_pin_name(first_lib_arc_idx, 1),
                "first_non_level0_timing_sense_name": _timing_sense_name(first_timing_sense) if first_arc is not None else "",
                "first_non_level0_timing_type_name": _timing_type_name(first_timing_type) if first_arc is not None else "",
                "first_net_out_arc_index": first_net_arc_index,
                "first_net_out_in_pin_name": _pin_name(first_net_in_pin),
                "first_net_out_out_pin_name": _pin_name(first_net_out_pin),
                "iteration": int(getattr(self, "invoke_timing_count", 0)),
            }
            rows.append(row)

        rows.sort(
            key=lambda row: (
                -int(row["non_level0_as_inst_out_arc_count"]),
                -int(row["as_net_self_loop_count"]),
                -float(row["max_r_aat_delta_ps"] or 0.0),
                row["normalized_pin_name"],
            )
        )
        summary = {
            "status": "ok",
            "row_count": len(rows),
            "non_level0_inst_out_startpoint_count": sum(1 for row in rows if int(row["non_level0_as_inst_out_arc_count"]) > 0),
            "net_out_startpoint_count": sum(1 for row in rows if int(row["as_net_out_arc_count"]) > 0),
            "net_self_loop_startpoint_count": sum(1 for row in rows if int(row["as_net_self_loop_count"]) > 0),
            "max_r_aat_delta_ps": max((float(row["max_r_aat_delta_ps"]) for row in rows if row["max_r_aat_delta_ps"] != ""), default=None),
            "max_f_aat_delta_ps": max((float(row["max_f_aat_delta_ps"]) for row in rows if row["max_f_aat_delta_ps"] != ""), default=None),
            "iteration": int(getattr(self, "invoke_timing_count", 0)),
        }
        return rows, summary

    def _selected_net_ids_for_alignment_debug(self, max_cap_drivers=64):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        output_pin_mask = _tensor_to_cpu_long_array(getattr(self, "output_pin_mask", None))
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        py_r_cap = None
        py_f_cap = None
        if timing_op is not None:
            py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise_live", None))
            py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall_live", None))
            if py_r_cap is None:
                py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise", None))
            if py_f_cap is None:
                py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall", None))
        if pin2net_map is None:
            return set()

        selected_net_ids = set()
        first_fanout_pin_ids, _ = self._first_level_fanout_pin_ids()
        for pin_id in first_fanout_pin_ids:
            if 0 <= int(pin_id) < pin2net_map.size:
                net_id = int(pin2net_map[int(pin_id)])
                if net_id >= 0:
                    selected_net_ids.add(net_id)
        name_to_id = {name: idx for idx, name in enumerate(pin_names)}
        for probe_name in _FIRST_LEVEL_NUMERIC_PROBE_PINS:
            pin_id = name_to_id.get(probe_name)
            if pin_id is not None and 0 <= pin_id < pin2net_map.size:
                net_id = int(pin2net_map[pin_id])
                if net_id >= 0:
                    selected_net_ids.add(net_id)

        if py_r_cap is not None and output_pin_mask is not None:
            output_pin_ids = [
                pin_id
                for pin_id in range(min(len(output_pin_mask), len(py_r_cap), pin2net_map.size))
                if output_pin_mask[pin_id]
            ]
            output_pin_ids.sort(
                key=lambda pin_id: max(
                    float(py_r_cap[pin_id]) if py_r_cap is not None and np.isfinite(py_r_cap[pin_id]) else 0.0,
                    float(py_f_cap[pin_id]) if py_f_cap is not None and np.isfinite(py_f_cap[pin_id]) else 0.0,
                ),
                reverse=True,
            )
            for pin_id in output_pin_ids[:max_cap_drivers]:
                net_id = int(pin2net_map[pin_id])
                if net_id >= 0:
                    selected_net_ids.add(net_id)
        return selected_net_ids

    def write_net_topology_debug(self, cap_limits=None, max_cap_drivers=64):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_net_topology_debug.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_net_topology_debug_iter{iteration:04d}.csv",
        )
        summary_path = os.path.join(result_dir, f"{design_name}_net_topology_debug_summary.json")
        iter_summary_path = os.path.join(
            result_dir,
            f"{design_name}_net_topology_debug_summary_iter{iteration:04d}.json",
        )

        rows, summary = self._build_net_topology_debug_rows(cap_limits=cap_limits, max_cap_drivers=max_cap_drivers)
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_NET_TOPOLOGY_DEBUG_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        for path in (summary_path, iter_summary_path):
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(summary, json_file, indent=2, sort_keys=True)
        logging.info("[NetTopologyDebug] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_net_topology_debug_rows(self, cap_limits=None, max_cap_drivers=64):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        net_names = [_decode_name(value) for value in getattr(self.placedb, "net_names", [])]
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        flat_net2pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_map", None))
        flat_net2pin_start = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_start_map", None))
        net2driver_pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "net2driver_pin_map", None))
        output_pin_mask = _tensor_to_cpu_long_array(getattr(self, "output_pin_mask", None))
        if (
            pin2net_map is None
            or flat_net2pin_map is None
            or flat_net2pin_start is None
            or net2driver_pin_map is None
        ):
            return [], {"status": "missing_net_topology_inputs"}

        timing_op = self.op_collections.timing_propagation_op
        py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT_live_snapshot", None))
        py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT_live_snapshot", None))
        py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran_live", None))
        py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran_live", None))
        py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise_live", None))
        py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall_live", None))
        if py_r_aat is None:
            py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT", None))
        if py_f_aat is None:
            py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT", None))
        if py_r_slew is None:
            py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran", None))
        if py_f_slew is None:
            py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran", None))
        if py_r_cap is None:
            py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise", None))
        if py_f_cap is None:
            py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall", None))
        if cap_limits is None:
            cap_limits = self._mapped_all_pin_limits(
                getattr(getattr(self, "data_collections", None), "flat_lib_pin_cap_limit", None)
            )
        else:
            cap_limits = _tensor_to_cpu_float_array(cap_limits)

        pin_cap_base = getattr(self, "_debug_last_pin_caps_base", None)
        pin_rcap_base = getattr(self, "_debug_last_pin_rcaps_base", None)
        pin_fcap_base = getattr(self, "_debug_last_pin_fcaps_base", None)
        wire_cap = None
        elmore_op = getattr(getattr(self, "op_collections", None), "elmore_delay_op", None)
        if elmore_op is not None:
            wire_cap = _tensor_to_cpu_float_array(getattr(elmore_op, "net_cap", None))

        selected_net_ids = self._selected_net_ids_for_alignment_debug(max_cap_drivers=max_cap_drivers)

        def _array_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return ""
            value = values[index]
            return "" if not np.isfinite(value) else float(value)

        def _sum_array(values, indices):
            total = 0.0
            seen = False
            for index in indices:
                value = _array_value(values, index)
                if value == "":
                    continue
                total += float(value)
                seen = True
            return total if seen else ""

        def _max_array(values, indices):
            numbers = []
            for index in indices:
                value = _array_value(values, index)
                if value != "":
                    numbers.append(float(value))
            return max(numbers) if numbers else ""

        rows = []
        for net_id in sorted(selected_net_ids):
            if net_id < 0 or net_id + 1 >= flat_net2pin_start.size:
                continue
            net_start = int(flat_net2pin_start[net_id])
            net_end = int(flat_net2pin_start[net_id + 1])
            net_pin_ids = [int(pin_id) for pin_id in flat_net2pin_map[net_start:net_end].tolist()]
            if not net_pin_ids:
                continue
            driver_pin_id = int(net2driver_pin_map[net_id]) if 0 <= net_id < net2driver_pin_map.size else -1
            if driver_pin_id < 0 and output_pin_mask is not None:
                for candidate in net_pin_ids:
                    if 0 <= candidate < len(output_pin_mask) and output_pin_mask[candidate]:
                        driver_pin_id = candidate
                        break
            if driver_pin_id < 0:
                continue
            flat_first_pin_id = int(net_pin_ids[0])
            sink_pin_ids = [pin_id for pin_id in net_pin_ids if pin_id != driver_pin_id]
            node_indices = list(net_pin_ids)
            if wire_cap is not None:
                # Include Steiner nodes reachable from the net's flat-pin seed if the
                # timing RC tree stores wire capacitance on internal nodes.
                data_collections = getattr(self, "data_collections", None)
                pin_fa = _tensor_to_cpu_long_array(getattr(data_collections, "pin_fa", None))
                flat_pin_to = _tensor_to_cpu_long_array(getattr(data_collections, "flat_pin_to", None))
                flat_pin_to_start = _tensor_to_cpu_long_array(getattr(data_collections, "flat_pin_to_start", None))
                if pin_fa is not None and flat_pin_to is not None and flat_pin_to_start is not None:
                    queue = [flat_first_pin_id]
                    visited = {flat_first_pin_id}
                    head = 0
                    while head < len(queue):
                        current = queue[head]
                        head += 1
                        if current + 1 >= flat_pin_to_start.size:
                            continue
                        child_start = int(flat_pin_to_start[current])
                        child_end = int(flat_pin_to_start[current + 1])
                        for child in flat_pin_to[child_start:child_end].tolist():
                            child = int(child)
                            if child not in visited:
                                visited.add(child)
                                queue.append(child)
                    node_indices = sorted(visited)
            unreached_net_pin_count = len(set(net_pin_ids) - set(node_indices))
            sum_sink_cap = _sum_array(pin_cap_base, sink_pin_ids)
            sum_sink_rcap = _sum_array(pin_rcap_base, sink_pin_ids)
            sum_sink_fcap = _sum_array(pin_fcap_base, sink_pin_ids)
            sum_wire = _sum_array(wire_cap, node_indices)
            estimated_total_cap = (
                sum_sink_cap + sum_wire
                if sum_sink_cap != "" and sum_wire != ""
                else ""
            )
            estimated_total_rcap = (
                sum_sink_rcap + sum_wire
                if sum_sink_rcap != "" and sum_wire != ""
                else ""
            )
            estimated_total_fcap = (
                sum_sink_fcap + sum_wire
                if sum_sink_fcap != "" and sum_wire != ""
                else ""
            )
            rows.append(
                {
                    "net_id": net_id,
                    "net_name": net_names[net_id] if 0 <= net_id < len(net_names) else "",
                    "driver_pin_id": driver_pin_id,
                    "driver_pin_name": pin_names[driver_pin_id] if 0 <= driver_pin_id < len(pin_names) else "",
                    "normalized_driver_pin_name": (
                        _normalize_join_pin_name(pin_names[driver_pin_id])
                        if 0 <= driver_pin_id < len(pin_names)
                        else ""
                    ),
                    "flat_net_first_pin_id": flat_first_pin_id,
                    "flat_net_first_pin_name": pin_names[flat_first_pin_id] if 0 <= flat_first_pin_id < len(pin_names) else "",
                    "driver_is_flat_net_first_pin": int(driver_pin_id == flat_first_pin_id),
                    "rc_traversal_seed_pin_id": flat_first_pin_id,
                    "rc_traversal_seed_pin_name": pin_names[flat_first_pin_id] if 0 <= flat_first_pin_id < len(pin_names) else "",
                    "fanout_count": max(len(net_pin_ids) - 1, 0),
                    "sink_count": len(sink_pin_ids),
                    "unreached_net_pin_count": unreached_net_pin_count,
                    "net_pin_ids": ";".join(str(pin_id) for pin_id in net_pin_ids),
                    "net_pin_names": ";".join(pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "" for pin_id in net_pin_ids),
                    "sink_pin_ids": ";".join(str(pin_id) for pin_id in sink_pin_ids),
                    "sink_pin_names": ";".join(pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "" for pin_id in sink_pin_ids),
                    "driver_py_r_aat_ps": _array_value(py_r_aat, driver_pin_id),
                    "driver_py_f_aat_ps": _array_value(py_f_aat, driver_pin_id),
                    "driver_py_r_slew_ps": _array_value(py_r_slew, driver_pin_id),
                    "driver_py_f_slew_ps": _array_value(py_f_slew, driver_pin_id),
                    "driver_py_r_cap_pf": _array_value(py_r_cap, driver_pin_id),
                    "driver_py_f_cap_pf": _array_value(py_f_cap, driver_pin_id),
                    "driver_cap_limit_pf": _array_value(cap_limits, driver_pin_id),
                    "sum_sink_cap_base_pf": sum_sink_cap,
                    "sum_sink_rcap_base_pf": sum_sink_rcap,
                    "sum_sink_fcap_base_pf": sum_sink_fcap,
                    "sum_node_wire_cap_pf": sum_wire,
                    "estimated_total_cap_plus_wire_pf": estimated_total_cap,
                    "estimated_total_rcap_plus_wire_pf": estimated_total_rcap,
                    "estimated_total_fcap_plus_wire_pf": estimated_total_fcap,
                    "max_sink_cap_base_pf": _max_array(pin_cap_base, sink_pin_ids),
                    "max_sink_rcap_base_pf": _max_array(pin_rcap_base, sink_pin_ids),
                    "max_sink_fcap_base_pf": _max_array(pin_fcap_base, sink_pin_ids),
                    "iteration": int(getattr(self, "invoke_timing_count", 0)),
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row["driver_py_r_cap_pf"] or 0.0),
                row["normalized_driver_pin_name"],
            )
        )
        summary = {
            "status": "ok",
            "row_count": len(rows),
            "driver_not_first_pin_count": sum(1 for row in rows if int(row["driver_is_flat_net_first_pin"]) == 0),
            "max_driver_py_r_cap_pf": max((float(row["driver_py_r_cap_pf"]) for row in rows if row["driver_py_r_cap_pf"] != ""), default=None),
            "max_sum_node_wire_cap_pf": max((float(row["sum_node_wire_cap_pf"]) for row in rows if row["sum_node_wire_cap_pf"] != ""), default=None),
            "max_unreached_net_pin_count": max((int(row["unreached_net_pin_count"]) for row in rows), default=0),
            "iteration": int(getattr(self, "invoke_timing_count", 0)),
        }
        return rows, summary

    def write_net_sink_cap_debug(self, max_cap_drivers=64):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_net_sink_cap_debug.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_net_sink_cap_debug_iter{iteration:04d}.csv",
        )
        summary_path = os.path.join(result_dir, f"{design_name}_net_sink_cap_debug_summary.json")
        iter_summary_path = os.path.join(
            result_dir,
            f"{design_name}_net_sink_cap_debug_summary_iter{iteration:04d}.json",
        )

        rows, summary = self._build_net_sink_cap_debug_rows(max_cap_drivers=max_cap_drivers)
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_NET_SINK_CAP_DEBUG_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        for path in (summary_path, iter_summary_path):
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(summary, json_file, indent=2, sort_keys=True)
        logging.info("[NetSinkCapDebug] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_net_sink_cap_debug_rows(self, max_cap_drivers=64):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        net_names = [_decode_name(value) for value in getattr(self.placedb, "net_names", [])]
        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        flat_net2pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_map", None))
        flat_net2pin_start = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_start_map", None))
        net2driver_pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "net2driver_pin_map", None))
        pin2node_map = _tensor_to_cpu_long_array(getattr(getattr(self, "data_collections", None), "pin2node_map", None))
        inst_main_id = _tensor_to_cpu_long_array(getattr(getattr(self, "data_collections", None), "inst_main_id", None))
        inst_libcell_offset = _tensor_to_cpu_long_array(getattr(getattr(self, "data_collections", None), "inst_libcell_offset", None))
        main_id_2_cell_id_start = _tensor_to_cpu_long_array(
            getattr(getattr(self, "data_collections", None), "main_id_2_cell_id_start", None)
        )
        pin_offsets = _tensor_to_cpu_long_array(getattr(getattr(self, "data_collections", None), "pin_2_libpin_offset", None))
        pin2libpin_flat_ids = _tensor_to_cpu_long_array(getattr(self, "pin2libpin_flat_ids", None))
        if (
            pin2net_map is None
            or flat_net2pin_map is None
            or flat_net2pin_start is None
            or net2driver_pin_map is None
        ):
            return [], {"status": "missing_net_topology_inputs"}

        timing_op = self.op_collections.timing_propagation_op
        py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT_live_snapshot", None))
        py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT_live_snapshot", None))
        py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran_live", None))
        py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran_live", None))
        if py_r_aat is None:
            py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT", None))
        if py_f_aat is None:
            py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT", None))
        if py_r_slew is None:
            py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran", None))
        if py_f_slew is None:
            py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran", None))

        pin_cap_base = getattr(self, "_debug_last_pin_caps_base", None)
        pin_rcap_base = getattr(self, "_debug_last_pin_rcaps_base", None)
        pin_fcap_base = getattr(self, "_debug_last_pin_fcaps_base", None)
        flat_lib_pin_names = getattr(getattr(self, "data_collections", None), "flat_lib_pin_names", None)
        if flat_lib_pin_names is None:
            flat_lib_pin_names = getattr(self.placedb, "flat_lib_pin_names", None)

        def _array_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return ""
            value = values[index]
            return "" if not np.isfinite(value) else float(value)

        def _long_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return -1
            return int(values[index])

        def _libpin_name(libpin_id, sink_pin_name):
            if flat_lib_pin_names is None or libpin_id < 0:
                return sink_pin_name.rsplit(":", 1)[-1] if ":" in sink_pin_name else ""
            try:
                raw = flat_lib_pin_names[libpin_id]
                return _decode_name(raw)
            except Exception:
                return sink_pin_name.rsplit(":", 1)[-1] if ":" in sink_pin_name else ""

        selected_net_ids = self._selected_net_ids_for_alignment_debug(max_cap_drivers=max_cap_drivers)
        rows = []
        for net_id in sorted(selected_net_ids):
            if net_id < 0 or net_id + 1 >= flat_net2pin_start.size:
                continue
            net_start = int(flat_net2pin_start[net_id])
            net_end = int(flat_net2pin_start[net_id + 1])
            net_pin_ids = [int(pin_id) for pin_id in flat_net2pin_map[net_start:net_end].tolist()]
            if not net_pin_ids:
                continue
            driver_pin_id = int(net2driver_pin_map[net_id]) if 0 <= net_id < net2driver_pin_map.size else -1
            if driver_pin_id < 0:
                continue
            driver_pin_name = pin_names[driver_pin_id] if 0 <= driver_pin_id < len(pin_names) else ""
            for sink_pin_id in net_pin_ids:
                if sink_pin_id == driver_pin_id:
                    continue
                sink_pin_name = pin_names[sink_pin_id] if 0 <= sink_pin_id < len(pin_names) else ""
                sink_node_id = _long_value(pin2node_map, sink_pin_id)
                sink_main_id = _long_value(inst_main_id, sink_node_id)
                sink_libpin_flat_id = _long_value(pin2libpin_flat_ids, sink_pin_id)
                sink_cell_offset = _long_value(inst_libcell_offset, sink_node_id)
                sink_libcell_idx = -1
                if (
                    sink_main_id >= 0
                    and sink_cell_offset >= 0
                    and main_id_2_cell_id_start is not None
                    and sink_main_id < main_id_2_cell_id_start.size
                ):
                    sink_libcell_idx = int(main_id_2_cell_id_start[sink_main_id]) + sink_cell_offset
                rows.append(
                    {
                        "net_id": net_id,
                        "net_name": net_names[net_id] if 0 <= net_id < len(net_names) else "",
                        "driver_pin_id": driver_pin_id,
                        "driver_pin_name": driver_pin_name,
                        "normalized_driver_pin_name": _normalize_join_pin_name(driver_pin_name),
                        "sink_pin_id": sink_pin_id,
                        "sink_pin_name": sink_pin_name,
                        "normalized_sink_pin_name": _normalize_join_pin_name(sink_pin_name),
                        "sink_node_id": sink_node_id if sink_node_id >= 0 else "",
                        "sink_main_id": sink_main_id if sink_main_id >= 0 else "",
                        "sink_libcell_idx": sink_libcell_idx if sink_libcell_idx >= 0 else "",
                        "sink_libcell_name": (
                            flat_libcell_names[sink_libcell_idx]
                            if 0 <= sink_libcell_idx < len(flat_libcell_names)
                            else ""
                        ),
                        "sink_libpin_flat_id": sink_libpin_flat_id if sink_libpin_flat_id >= 0 else "",
                        "sink_libpin_name": _libpin_name(sink_libpin_flat_id, sink_pin_name),
                        "sink_pin_offset": _long_value(pin_offsets, sink_pin_id) if pin_offsets is not None else "",
                        "sink_cap_base_pf": _array_value(pin_cap_base, sink_pin_id),
                        "sink_rcap_base_pf": _array_value(pin_rcap_base, sink_pin_id),
                        "sink_fcap_base_pf": _array_value(pin_fcap_base, sink_pin_id),
                        "sink_py_r_aat_ps": _array_value(py_r_aat, sink_pin_id),
                        "sink_py_f_aat_ps": _array_value(py_f_aat, sink_pin_id),
                        "sink_py_r_slew_ps": _array_value(py_r_slew, sink_pin_id),
                        "sink_py_f_slew_ps": _array_value(py_f_slew, sink_pin_id),
                        "iteration": int(getattr(self, "invoke_timing_count", 0)),
                    }
                )
        rows.sort(key=lambda row: (row["normalized_driver_pin_name"], row["normalized_sink_pin_name"]))
        summary = {
            "status": "ok",
            "row_count": len(rows),
            "unique_driver_count": len({row["normalized_driver_pin_name"] for row in rows}),
            "max_sink_rcap_base_pf": max((float(row["sink_rcap_base_pf"]) for row in rows if row["sink_rcap_base_pf"] != ""), default=None),
            "iteration": int(getattr(self, "invoke_timing_count", 0)),
        }
        return rows, summary

    def write_first_level_lut_probe(self, max_rows=4096):
        if not hasattr(self, "placedb") or getattr(self.placedb, "pin_names", None) is None:
            return None
        timing_op = getattr(getattr(self, "op_collections", None), "timing_propagation_op", None)
        if timing_op is None:
            return None

        result_dir = getattr(getattr(self, "params", None), "result_dir", None)
        if not result_dir:
            result_dir = os.path.join(
                self.placedb.data_manager.dir_workspace,
                "output",
                "dreamplace",
                "result",
            )
        os.makedirs(result_dir, exist_ok=True)
        design_name = self.params.design_name()
        iteration = int(getattr(self, "invoke_timing_count", 0))
        output_path = os.path.join(result_dir, f"{design_name}_first_level_lut_probe.csv")
        iter_output_path = os.path.join(
            result_dir,
            f"{design_name}_first_level_lut_probe_iter{iteration:04d}.csv",
        )

        rows = self._build_first_level_lut_probe_rows(max_rows=max_rows)
        for path in (output_path, iter_output_path):
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=_FIRST_LEVEL_LUT_PROBE_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        logging.info("[FirstLevelLUTProbe] Wrote %d rows to %s", len(rows), output_path)
        return output_path

    def _build_first_level_lut_probe_rows(self, max_rows=4096):
        data_collections = getattr(self, "data_collections", None)
        timing_op = self.op_collections.timing_propagation_op
        flat_arcs = getattr(data_collections, "flat_inst_arcs_by_level", None)
        level_starts = getattr(data_collections, "flat_inst_arcs_by_level_start", None)
        if flat_arcs is None or level_starts is None:
            return []
        flat_arcs_np = flat_arcs.detach().cpu().numpy() if hasattr(flat_arcs, "detach") else np.asarray(flat_arcs)
        level_starts_np = level_starts.detach().cpu().long().numpy() if hasattr(level_starts, "detach") else np.asarray(level_starts, dtype=np.int64)
        if level_starts_np.size < 3 or flat_arcs_np.size == 0:
            return []

        first_level_start = int(level_starts_np[1])
        first_level_end = int(level_starts_np[2])
        if first_level_end <= first_level_start:
            return []

        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        flat_libcell_names = [_decode_name(value) for value in getattr(self.placedb, "flat_libcell_names", [])]
        raw_libarc_names = getattr(self.placedb, "flat_libarc_names", [])
        flat_libarc_info = _tensor_to_cpu_long_array(getattr(data_collections, "flat_libarc_info", None))
        py_r_slew = getattr(timing_op, "pin_rtran_live", None)
        py_f_slew = getattr(timing_op, "pin_ftran_live", None)
        py_r_cap = getattr(timing_op, "pin_net_cap_rise_live", None)
        py_f_cap = getattr(timing_op, "pin_net_cap_fall_live", None)
        if py_r_slew is None:
            py_r_slew = getattr(timing_op, "pin_rtran", None)
        if py_f_slew is None:
            py_f_slew = getattr(timing_op, "pin_ftran", None)
        if py_r_cap is None:
            py_r_cap = getattr(timing_op, "pin_net_cap_rise", None)
        if py_f_cap is None:
            py_f_cap = getattr(timing_op, "pin_net_cap_fall", None)
        if py_r_slew is None or py_f_slew is None or py_r_cap is None or py_f_cap is None:
            return []

        rows = []
        selected_arcs = flat_arcs[first_level_start:first_level_end]
        if selected_arcs.shape[0] > max_rows:
            selected_arcs = selected_arcs[:max_rows]
        selected_arcs = timing_index_tensor(selected_arcs, py_r_slew)
        if selected_arcs.numel() == 0:
            return []
        in_pins = selected_arcs[:, 0]
        out_pins = selected_arcs[:, 1]
        lib_cell_idxs = selected_arcs[:, 2]
        lib_arc_idxs = selected_arcs[:, 3]
        query_r_slew = py_r_slew[in_pins]
        query_f_slew = py_f_slew[in_pins]
        query_r_cap = py_r_cap[out_pins]
        query_f_cap = py_f_cap[out_pins]

        with torch.no_grad():
            r_delay = timing_op.r_delay_entry(
                lib_cell_idxs,
                query_r_slew,
                query_r_cap,
                lib_arc_idxs,
                out_pins,
                use_surrogate=False,
            ).detach().cpu().numpy()
            f_delay = timing_op.f_delay_entry(
                lib_cell_idxs,
                query_f_slew,
                query_f_cap,
                lib_arc_idxs,
                out_pins,
                use_surrogate=False,
            ).detach().cpu().numpy()
            r_transition = timing_op.r_tran_entry(
                lib_cell_idxs,
                query_r_slew,
                query_r_cap,
                lib_arc_idxs,
                out_pins,
                use_surrogate=False,
            ).detach().cpu().numpy()
            f_transition = timing_op.f_tran_entry(
                lib_cell_idxs,
                query_f_slew,
                query_f_cap,
                lib_arc_idxs,
                out_pins,
                use_surrogate=False,
            ).detach().cpu().numpy()

        query_r_slew_np = query_r_slew.detach().cpu().float().numpy()
        query_f_slew_np = query_f_slew.detach().cpu().float().numpy()
        query_r_cap_np = query_r_cap.detach().cpu().float().numpy()
        query_f_cap_np = query_f_cap.detach().cpu().float().numpy()
        selected_np = selected_arcs.detach().cpu().numpy()

        def _arc_pin_name(arc_id, col):
            if 0 <= arc_id < len(raw_libarc_names):
                raw = raw_libarc_names[arc_id]
                try:
                    return _decode_name(raw[col])
                except Exception:
                    return ""
            return ""

        def _axis_bounds(luts_info, arc_id):
            if luts_info is None or arc_id < 0:
                return "", "", "", ""
            trans_dims = getattr(luts_info, "trans_dims_actual", None)
            cap_dims = getattr(luts_info, "cap_dims_actual", None)
            trans_table = getattr(luts_info, "flat_luts_trans_table", None)
            cap_table = getattr(luts_info, "flat_luts_cap_table", None)
            if trans_dims is None or cap_dims is None or trans_table is None or cap_table is None:
                return "", "", "", ""
            if arc_id >= int(trans_dims.shape[0]) or arc_id >= int(cap_dims.shape[0]):
                return "", "", "", ""
            trans_dim = int(trans_dims[arc_id].detach().item())
            cap_dim = int(cap_dims[arc_id].detach().item())
            trans_min = trans_max = cap_min = cap_max = ""
            if trans_dim > 0:
                trans_min = _finite_or_blank(trans_table[arc_id, 0].detach().item())
                trans_max = _finite_or_blank(trans_table[arc_id, trans_dim - 1].detach().item())
            if cap_dim > 0:
                cap_min = _finite_or_blank(cap_table[arc_id, 0].detach().item())
                cap_max = _finite_or_blank(cap_table[arc_id, cap_dim - 1].detach().item())
            return trans_min, trans_max, cap_min, cap_max

        for local_idx, arc in enumerate(selected_np):
            if len(arc) < 6:
                continue
            arc_index = first_level_start + local_idx
            in_pin_id = int(arc[0])
            out_pin_id = int(arc[1])
            lib_cell_idx = int(arc[2])
            lib_arc_idx = int(arc[3])
            timing_sense = int(arc[4])
            timing_type = int(arc[5])
            in_pin_name = pin_names[in_pin_id] if 0 <= in_pin_id < len(pin_names) else ""
            out_pin_name = pin_names[out_pin_id] if 0 <= out_pin_id < len(pin_names) else ""
            timing_sense_label = _timing_sense_name(timing_sense)
            timing_type_label = _timing_type_name(timing_type)
            r_delay_bounds = _axis_bounds(timing_op.arcs_info.r_delay_luts, lib_arc_idx)
            f_delay_bounds = _axis_bounds(timing_op.arcs_info.f_delay_luts, lib_arc_idx)
            r_tran_bounds = _axis_bounds(timing_op.arcs_info.r_trans_luts, lib_arc_idx)
            f_tran_bounds = _axis_bounds(timing_op.arcs_info.f_trans_luts, lib_arc_idx)
            arc_offset = ""
            if flat_libarc_info is not None and 0 <= lib_arc_idx < len(flat_libarc_info) and flat_libarc_info.shape[1] > 3:
                arc_offset = int(flat_libarc_info[lib_arc_idx, 3])
            rows.append(
                {
                    "arc_index": arc_index,
                    "arc_join_key": (
                        f"{_normalize_join_pin_name(in_pin_name)}->"
                        f"{_normalize_join_pin_name(out_pin_name)}:"
                        f"{timing_sense_label}:{timing_type_label}"
                    ),
                    "in_pin_id": in_pin_id,
                    "in_pin_name": in_pin_name,
                    "in_pin_normalized": _normalize_join_pin_name(in_pin_name),
                    "out_pin_id": out_pin_id,
                    "out_pin_name": out_pin_name,
                    "out_pin_normalized": _normalize_join_pin_name(out_pin_name),
                    "lib_cell_idx": lib_cell_idx,
                    "lib_cell_name": flat_libcell_names[lib_cell_idx] if 0 <= lib_cell_idx < len(flat_libcell_names) else "",
                    "lib_arc_idx": lib_arc_idx,
                    "lib_arc_from_pin": _arc_pin_name(lib_arc_idx, 0),
                    "lib_arc_to_pin": _arc_pin_name(lib_arc_idx, 1),
                    "arc_offset": arc_offset,
                    "lut_boundary_mode": "timing_propagation_1d_clamp_2d_extrapolate",
                    "timing_sense": timing_sense,
                    "timing_sense_name": timing_sense_label,
                    "timing_type": timing_type,
                    "timing_type_name": timing_type_label,
                    "query_r_input_slew_ps": _finite_or_blank(query_r_slew_np[local_idx]),
                    "query_f_input_slew_ps": _finite_or_blank(query_f_slew_np[local_idx]),
                    "query_r_output_cap_pf": _finite_or_blank(query_r_cap_np[local_idx]),
                    "query_f_output_cap_pf": _finite_or_blank(query_f_cap_np[local_idx]),
                    "r_delay_value_ps": _finite_or_blank(r_delay[local_idx]),
                    "f_delay_value_ps": _finite_or_blank(f_delay[local_idx]),
                    "r_transition_value_ps": _finite_or_blank(r_transition[local_idx]),
                    "f_transition_value_ps": _finite_or_blank(f_transition[local_idx]),
                    "r_delay_trans_min_ps": r_delay_bounds[0],
                    "r_delay_trans_max_ps": r_delay_bounds[1],
                    "r_delay_cap_min_pf": r_delay_bounds[2],
                    "r_delay_cap_max_pf": r_delay_bounds[3],
                    "f_delay_trans_min_ps": f_delay_bounds[0],
                    "f_delay_trans_max_ps": f_delay_bounds[1],
                    "f_delay_cap_min_pf": f_delay_bounds[2],
                    "f_delay_cap_max_pf": f_delay_bounds[3],
                    "r_transition_trans_min_ps": r_tran_bounds[0],
                    "r_transition_trans_max_ps": r_tran_bounds[1],
                    "r_transition_cap_min_pf": r_tran_bounds[2],
                    "r_transition_cap_max_pf": r_tran_bounds[3],
                    "f_transition_trans_min_ps": f_tran_bounds[0],
                    "f_transition_trans_max_ps": f_tran_bounds[1],
                    "f_transition_cap_min_pf": f_tran_bounds[2],
                    "f_transition_cap_max_pf": f_tran_bounds[3],
                    "iteration": int(getattr(self, "invoke_timing_count", 0)),
                }
            )
        return rows

    def _build_first_level_numeric_alignment_rows(self, slew_limits=None, cap_limits=None):
        pin_names = [_decode_name(value) for value in self.placedb.pin_names]
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        net2driver_pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "net2driver_pin_map", None))
        net_names = [_decode_name(value) for value in getattr(self.placedb, "net_names", [])]

        timing_op = self.op_collections.timing_propagation_op
        py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT_live_snapshot", None))
        py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT_live_snapshot", None))
        py_r_rat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rRAT_live_snapshot", None))
        py_f_rat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fRAT_live_snapshot", None))
        if py_r_aat is None:
            py_r_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rAAT", None))
        if py_f_aat is None:
            py_f_aat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fAAT", None))
        if py_r_rat is None:
            py_r_rat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rRAT", None))
        if py_f_rat is None:
            py_f_rat = _tensor_to_cpu_float_array(getattr(timing_op, "pin_fRAT", None))
        py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran_live", None))
        py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran_live", None))
        py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise_live", None))
        py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall_live", None))
        if py_r_slew is None:
            py_r_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_rtran", None))
        if py_f_slew is None:
            py_f_slew = _tensor_to_cpu_float_array(getattr(timing_op, "pin_ftran", None))
        if py_r_cap is None:
            py_r_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_rise", None))
        if py_f_cap is None:
            py_f_cap = _tensor_to_cpu_float_array(getattr(timing_op, "pin_net_cap_fall", None))

        if slew_limits is None:
            slew_limits = self._mapped_all_pin_limits(
                getattr(getattr(self, "data_collections", None), "flat_lib_pin_slew_limit", None)
            )
        else:
            slew_limits = _tensor_to_cpu_float_array(slew_limits)
        if cap_limits is None:
            cap_limits = self._mapped_all_pin_limits(
                getattr(getattr(self, "data_collections", None), "flat_lib_pin_cap_limit", None)
            )
        else:
            cap_limits = _tensor_to_cpu_float_array(cap_limits)
        output_pin_mask = _tensor_to_cpu_long_array(getattr(self, "output_pin_mask", None))

        selected = []
        first_fanout_pin_ids, start_pin_for_sink = self._first_level_fanout_pin_ids()
        for pin_id in first_fanout_pin_ids:
            role = "first_fanout_start" if start_pin_for_sink.get(pin_id) == pin_id else "first_fanout_sink"
            selected.append((role, pin_id))
        name_to_id = {name: idx for idx, name in enumerate(pin_names)}
        for probe_name in _FIRST_LEVEL_NUMERIC_PROBE_PINS:
            pin_id = name_to_id.get(probe_name)
            if pin_id is not None:
                selected.append(("top_mismatch_anchor", pin_id))

        rows = []
        seen = set()
        for role, pin_id in selected:
            key = (role, int(pin_id))
            if key in seen:
                continue
            seen.add(key)
            source_start_pin_id = int(start_pin_for_sink.get(pin_id, -1))
            rows.append(
                self._first_level_numeric_alignment_row(
                    row_role=role,
                    pin_id=int(pin_id),
                    source_start_pin_id=source_start_pin_id,
                    pin_names=pin_names,
                    pin2net_map=pin2net_map,
                    net2driver_pin_map=net2driver_pin_map,
                    net_names=net_names,
                    first_fanout_pin_ids=first_fanout_pin_ids,
                    py_r_aat=py_r_aat,
                    py_f_aat=py_f_aat,
                    py_r_rat=py_r_rat,
                    py_f_rat=py_f_rat,
                    py_r_slew=py_r_slew,
                    py_f_slew=py_f_slew,
                    py_r_cap=py_r_cap,
                    py_f_cap=py_f_cap,
                    slew_limits=slew_limits,
                    cap_limits=cap_limits,
                    output_pin_mask=output_pin_mask,
                )
            )
        return rows

    def _first_level_fanout_pin_ids(self, max_startpoints=12):
        data_collections = getattr(self, "data_collections", None)
        start_points = _tensor_to_cpu_long_array(getattr(data_collections, "start_points", None))
        flat_net2pin_map = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_map", None))
        flat_net2pin_start = _tensor_to_cpu_long_array(getattr(self.placedb, "flat_net2pin_start_map", None))
        pin2net_map = _tensor_to_cpu_long_array(getattr(self.placedb, "pin2net_map", None))
        if start_points is None or flat_net2pin_map is None or flat_net2pin_start is None or pin2net_map is None:
            return [], {}
        selected = []
        start_pin_for_sink = {}
        for start_pin_id in start_points[:max_startpoints].tolist():
            if start_pin_id < 0 or start_pin_id >= pin2net_map.size:
                continue
            net_id = int(pin2net_map[start_pin_id])
            if net_id < 0 or net_id + 1 >= flat_net2pin_start.size:
                continue
            net_start = int(flat_net2pin_start[net_id])
            net_end = int(flat_net2pin_start[net_id + 1])
            net_pin_ids = [int(pin_id) for pin_id in flat_net2pin_map[net_start:net_end].tolist()]
            if not net_pin_ids:
                continue
            if start_pin_id not in selected:
                selected.append(start_pin_id)
            start_pin_for_sink[start_pin_id] = start_pin_id
            for sink_pin_id in net_pin_ids:
                if sink_pin_id == start_pin_id:
                    continue
                if sink_pin_id not in selected:
                    selected.append(sink_pin_id)
                start_pin_for_sink[sink_pin_id] = start_pin_id
        return selected, start_pin_for_sink

    def _first_level_numeric_alignment_row(
        self,
        row_role,
        pin_id,
        source_start_pin_id,
        pin_names,
        pin2net_map,
        net2driver_pin_map,
        net_names,
        first_fanout_pin_ids,
        py_r_aat,
        py_f_aat,
        py_r_rat,
        py_f_rat,
        py_r_slew,
        py_f_slew,
        py_r_cap,
        py_f_cap,
        slew_limits,
        cap_limits,
        output_pin_mask,
    ):
        def _array_value(values, index):
            if values is None or index < 0 or index >= len(values):
                return ""
            value = values[index]
            return "" if not np.isfinite(value) else float(value)

        net_id = int(pin2net_map[pin_id]) if pin2net_map is not None and pin_id < pin2net_map.size else -1
        driver_pin_id = -1
        if net2driver_pin_map is not None and 0 <= net_id < net2driver_pin_map.size:
            driver_pin_id = int(net2driver_pin_map[net_id])
        py_r_aat_v = _array_value(py_r_aat, pin_id)
        py_f_aat_v = _array_value(py_f_aat, pin_id)
        py_r_rat_v = _array_value(py_r_rat, pin_id)
        py_f_rat_v = _array_value(py_f_rat, pin_id)
        if py_r_aat_v == "" or py_f_aat_v == "" or py_r_rat_v == "" or py_f_rat_v == "":
            py_min_slack = ""
        else:
            py_min_slack = min(py_r_rat_v - py_r_aat_v, py_f_rat_v - py_f_aat_v)
        return {
            "row_role": row_role,
            "pin_id": pin_id,
            "pin_name": pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "",
            "normalized_pin_name": _normalize_join_pin_name(pin_names[pin_id]) if 0 <= pin_id < len(pin_names) else "",
            "net_id": net_id if net_id >= 0 else "",
            "net_name": net_names[net_id] if 0 <= net_id < len(net_names) else "",
            "driver_pin_id": driver_pin_id if driver_pin_id >= 0 else "",
            "driver_pin_name": pin_names[driver_pin_id] if 0 <= driver_pin_id < len(pin_names) else "",
            "source_start_pin_id": source_start_pin_id if source_start_pin_id >= 0 else "",
            "source_start_pin_name": pin_names[source_start_pin_id] if 0 <= source_start_pin_id < len(pin_names) else "",
            "reachable_static_first_fanout": str(pin_id in set(first_fanout_pin_ids)).lower(),
            "is_output_pin": int(bool(output_pin_mask is not None and pin_id < len(output_pin_mask) and output_pin_mask[pin_id])),
            "py_r_aat_ps": py_r_aat_v,
            "py_f_aat_ps": py_f_aat_v,
            "py_r_rat_ps": py_r_rat_v,
            "py_f_rat_ps": py_f_rat_v,
            "py_min_slack_ps": py_min_slack,
            "py_r_slew_ps": _array_value(py_r_slew, pin_id),
            "py_f_slew_ps": _array_value(py_f_slew, pin_id),
            "py_r_cap_pf": _array_value(py_r_cap, pin_id),
            "py_f_cap_pf": _array_value(py_f_cap, pin_id),
            "slew_limit_ps": _array_value(slew_limits, pin_id),
            "cap_limit_pf": _array_value(cap_limits, pin_id),
            "iteration": int(getattr(self, "invoke_timing_count", 0)),
        }

    def check_log(self, wns, tns, ws, ts):
        if getattr(self.params, "place_io_engine", "ieda") == "ecc":
            logging.info("skip iEDA reference check_log for ECC native timing backend")
            return
        # ==============================================================================
        # --- 步骤 2: 初始化iEDA并使用正确的线电容为其构建RC树 ---
        # ==============================================================================
        self._check_log_invocation_count = (
            getattr(self, "_check_log_invocation_count", 0) + 1
        )
        check_log_idx = self._check_log_invocation_count

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
        self.write_endpoint_timing_compare(at_late_cpp, rt_late_cpp)

        # DEBUG
        self.write_first_level_pin_timing_log(net_timing_details_cpp)
        clk2q_rows, _ = self.write_clk2q_level0_arc_report(at_late_cpp, cell_arc_delays_cpp, net_timing_details_cpp)
        self.write_clk2q_clock_pin_debug_dump(clk2q_rows or [])
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

        exit_after_first_check_log = os.environ.get(
            "AUTODMP_EXIT_AFTER_FIRST_CHECK_LOG", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if exit_after_first_check_log and check_log_idx == 1:
            logging.info(
                "AUTODMP_EXIT_AFTER_FIRST_CHECK_LOG is enabled; exiting after "
                "the first completed check_log() pass."
            )
            raise SystemExit(0)

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
            if num_filler_nodes > 0:
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
            if num_filler_nodes > 0:
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
            debug_mask = (
                bool(getattr(self.params, "l_shape_mask_debug", False))
                or l_shape_log_verbose(self.params) >= 2
            )
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
        cell_modeling_op = getattr(self.op_collections, "cell_modeling_op", None)
        profile_start = time.perf_counter()
        zero_grad_ms = 0.0
        obj_fn_ms = 0.0
        component_grad_stats_ms = 0.0
        backward_ms = 0.0
        backward_mode = "full_backward"
        backward_grad_param_count = None
        size_only_grad_zero_ms = 0.0
        precondition_ms = 0.0
        if (
            cell_modeling_op is not None
            and hasattr(cell_modeling_op, "reset_piecewise_size_backend_profile")
        ):
            cell_modeling_op.reset_piecewise_size_backend_profile()
        cell_model_profile_enabled = bool(getattr(self.params, "full_step_profile", False))
        cell_model_forward_profile = {}
        cell_modeling_runtime_op = getattr(self.op_collections, "cell_modeling_op", None)
        if (
            cell_model_profile_enabled
            and cell_modeling_runtime_op is not None
            and hasattr(cell_modeling_runtime_op, "reset_forward_profile")
        ):
            cell_modeling_runtime_op.reset_forward_profile(enabled=True)
        # self.check_gradient(pos)
        stage_start = time.perf_counter()
        if pos.grad is not None:
            pos.grad.zero_()
        size_parameter_getter = getattr(
            self.data_collections,
            "get_continuous_size_parameter",
            None,
        )
        size_parameter = (
            size_parameter_getter()
            if callable(size_parameter_getter)
            else getattr(self.data_collections, "size_logits", None)
        )
        if size_parameter is not None and size_parameter.grad is not None:
            size_parameter.grad.zero_()
        if getattr(self.data_collections, "vt_logits", None) is not None and self.data_collections.vt_logits.grad is not None:
            self.data_collections.vt_logits.grad.zero_()
        for param in self._buffer_autograd_params():
            if param.grad is not None:
                param.grad.zero_()
        zero_grad_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        obj = self.obj_fn(pos)
        obj, joint_proximal_summary = self.joint_proximal_objective.add_to_objective(
            self,
            obj,
            pos,
        )
        self.last_joint_proximal_summary = joint_proximal_summary
        obj_fn_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        self.debug_component_grad_stats = {}
        if self._size_debug_component_grad_stats_enabled():
            timing_loss = None
            if self.use_timing_obj and hasattr(self, "wns") and hasattr(self, "tns"):
                timing_loss = self._timing_loss(
                    self.wns,
                    self.tns,
                    getattr(self, "ws", None),
                    getattr(self, "ts", None),
                )
            self.debug_component_grad_stats = {
                "timing_loss_size_grad": self._size_debug_grad_stats_for_component(timing_loss),
                "size_density_area_penalty_grad": self._size_debug_grad_stats_for_component(
                    self.size_density_area_penalty
                ),
            }
        component_grad_stats_ms = (time.perf_counter() - stage_start) * 1000.0

        backward_component_profile = {}
        if self._timing_backward_component_profile_enabled():
            backward_component_profile = self._timing_backward_component_profile()

        stage_start = time.perf_counter()
        if self._use_targeted_sizing_backward():
            backward_mode = "targeted_sizing_autograd_grad"
            backward_grad_param_count = self._targeted_sizing_backward(obj)
        else:
            obj.backward()
        # Do not reuse an autograd-connected Steiner graph across optimizer
        # evaluations; the next objective call must build a fresh graph.
        self._timing_geometry_cache = None
        backward_ms = (time.perf_counter() - stage_start) * 1000.0
        total_ms = (time.perf_counter() - profile_start) * 1000.0
        piecewise_backend_profile = {}
        if (
            cell_modeling_op is not None
            and hasattr(cell_modeling_op, "get_piecewise_size_backend_profile")
        ):
            piecewise_backend_profile = (
                cell_modeling_op.get_piecewise_size_backend_profile()
            )
        if (
            cell_model_profile_enabled
            and cell_modeling_runtime_op is not None
            and hasattr(cell_modeling_runtime_op, "get_forward_profile")
        ):
            cell_model_forward_profile = cell_modeling_runtime_op.get_forward_profile()
        if self._is_size_only_mode():
            stage_start = time.perf_counter()
            if pos.grad is not None:
                pos.grad.zero_()
            size_only_grad_zero_ms = (time.perf_counter() - stage_start) * 1000.0
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            self.debug_obj_and_grad_profile = {
                "obj_and_grad_zero_grad_ms": zero_grad_ms,
                "obj_and_grad_obj_fn_ms": obj_fn_ms,
                "obj_and_grad_component_grad_stats_ms": component_grad_stats_ms,
                "objective_backward_ms": backward_ms,
                "objective_backward_mode": backward_mode,
                "objective_backward_grad_param_count": backward_grad_param_count,
                "obj_and_grad_size_only_grad_zero_ms": size_only_grad_zero_ms,
                "obj_and_grad_precondition_ms": precondition_ms,
                "obj_and_grad_total_ms": total_ms,
                "joint_proximal": self.last_joint_proximal_summary,
            }
            if backward_component_profile:
                self.debug_obj_and_grad_profile["timing_backward_component_profile"] = (
                    backward_component_profile
                )
            if piecewise_backend_profile:
                self.debug_obj_and_grad_profile["piecewise_size_backend_profile"] = (
                    piecewise_backend_profile
                )
            if cell_model_forward_profile:
                self.debug_obj_and_grad_profile["cell_model_forward_profile"] = (
                    cell_model_forward_profile
                )
            return obj, torch.zeros_like(pos)

        assert torch.isnan(pos.grad).any() == False, "Gradient contains NaN"
        stage_start = time.perf_counter()
        l_shape_active = self.use_l_shape_routability and self.l_shape_routability_op is not None
        if not l_shape_active:
            self.op_collections.precondition_op(
                pos.grad, self.density_weight, self.update_mask, self.fix_nodes_mask
            )
        precondition_ms = (time.perf_counter() - stage_start) * 1000.0
        total_ms = (time.perf_counter() - profile_start) * 1000.0
        self.debug_obj_and_grad_profile = {
            "obj_and_grad_zero_grad_ms": zero_grad_ms,
            "obj_and_grad_obj_fn_ms": obj_fn_ms,
            "obj_and_grad_component_grad_stats_ms": component_grad_stats_ms,
            "objective_backward_ms": backward_ms,
            "objective_backward_mode": backward_mode,
            "objective_backward_grad_param_count": backward_grad_param_count,
            "obj_and_grad_size_only_grad_zero_ms": size_only_grad_zero_ms,
            "obj_and_grad_precondition_ms": precondition_ms,
            "obj_and_grad_total_ms": total_ms,
            "joint_proximal": self.last_joint_proximal_summary,
        }
        if backward_component_profile:
            self.debug_obj_and_grad_profile["timing_backward_component_profile"] = (
                backward_component_profile
            )
        if piecewise_backend_profile:
            self.debug_obj_and_grad_profile["piecewise_size_backend_profile"] = (
                piecewise_backend_profile
            )
        if cell_model_forward_profile:
            self.debug_obj_and_grad_profile["cell_model_forward_profile"] = (
                cell_model_forward_profile
            )
        reset_l_shape_gradient_telemetry(self)
        
        # ========== L形Routability梯度 ==========
        if l_shape_active:
            obj = apply_l_shape_gradient(self, pos, obj)
            
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
        density_op = getattr(self.l_shape_routability_op, "density_op", None)
        if bool(getattr(density_op, "fast_mode", False)):
            raise RuntimeError(
                "check_l_shape_gradient_direction requires full L-shape energy; disable l_shape_fast_mode"
            )
        
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
        density_op = getattr(self.l_shape_routability_op, "density_op", None)
        if bool(getattr(density_op, "fast_mode", False)):
            raise RuntimeError(
                "check_l_shape_gradient_numerical requires full L-shape energy; disable l_shape_fast_mode"
            )
        
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
        if getattr(placedb, "gr_sizing", None) is not None:
            return placedb.gr_sizing.op
        rc_timing = RCTiming(
            r_unit=placedb.r_unit,
            c_unit=placedb.c_unit,
            scale_factor=params.scale_factor,
            dbu=placedb.dbu,
        )
        return rc_timing

    def build_timing_propagation_op(self, params, placedb, data_collections):

        write_crash_stage_marker(
            "timing_propagation_op_construction",
            "start",
            timing_propagation_device=getattr(params, "timing_propagation_device", "inherit"),
            params_gpu=getattr(params, "gpu", 0),
            params_gpu_id=getattr(params, "gpu_id", 0),
        )
        timing_propagation_op = TimingPropagation(
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
            data_collections.flat_inst_arcs_by_level, # //
            data_collections.flat_inst_arcs_by_level_start, 
            placedb.endpoints_rRAT,
            placedb.endpoints_fRAT,
            data_collections.flat_pin_to_graph,
            data_collections.flat_pin_to_graph_start,
            data_collections.flat_pin_to_graph_reverse,
            data_collections.flat_pin_to_graph_start_reverse,
            endpoints_timing_check_arcs=getattr(data_collections, "endpoints_timing_check_arcs", None),
            pin_pair_arc_keys=data_collections.pin_pair_arc_keys,
            flat_pin_pair_arc_start=data_collections.flat_pin_pair_arc_start,
            flat_pin_pair_arc_indices=data_collections.flat_pin_pair_arc_indices,
            pin_pred_start=getattr(data_collections, "pin_pred_start", None),
            pin_pred_pin=getattr(data_collections, "pin_pred_pin", None),
            pin_pred_arc_id=getattr(data_collections, "pin_pred_arc_id", None),
            pin_succ_start=getattr(data_collections, "pin_succ_start", None),
            pin_succ_pin=getattr(data_collections, "pin_succ_pin", None),
            pin_succ_arc_id=getattr(data_collections, "pin_succ_arc_id", None),
            arc_level_start=getattr(data_collections, "arc_level_start", None),
            arc_src_pin=getattr(data_collections, "arc_src_pin", None),
            arc_dst_pin=getattr(data_collections, "arc_dst_pin", None),
            arc_inst_id=getattr(data_collections, "arc_inst_id", None),
            endpoint_pin_ids=getattr(data_collections, "endpoint_pin_ids", None),
            start_pin_ids=getattr(data_collections, "start_pin_ids", None),
            pin_to_inst_id=getattr(data_collections, "pin_to_inst_id", None),
            pin_to_node_id=getattr(data_collections, "pin_to_node_id", None),
            inst_topo_start=getattr(data_collections, "inst_topo_start", None),
            inst_topo_ids=getattr(data_collections, "inst_topo_ids", None),
            cell_modeling_op=self.op_collections.cell_modeling_op,
            lib_arc_offsets=(
                data_collections.flat_libarc_info[:, 3].long()
                if getattr(data_collections, "flat_libarc_info", None) is not None
                else None
            ),
            pin2node_map=data_collections.pin2node_map,
            inst_main_id=data_collections.inst_main_id,
            inst_is_sizeable=getattr(data_collections, "inst_is_sizeable", None),
            inst_size_init=getattr(data_collections, "inst_size_init", None),
            inst_vt_init=getattr(data_collections, "inst_vt_init", None),
            size_var_getter=getattr(data_collections, "get_size_var", None),
            vt_var_getter=getattr(data_collections, "get_vt_var", None),
            critical_endpoint_pruning_mode=getattr(
                params, "critical_endpoint_pruning_mode", "off"
            ),
            critical_endpoint_top_k=getattr(params, "critical_endpoint_top_k", 256),
            critical_endpoint_slack_window_ps=getattr(
                params, "critical_endpoint_slack_window_ps", 100.0
            ),
            critical_endpoint_refresh_interval=getattr(
                params, "critical_endpoint_refresh_interval", 10
            ),
            critical_endpoint_hysteresis_interval=getattr(
                params, "critical_endpoint_hysteresis_interval", 3
            ),
            critical_endpoint_full_refresh_interval=getattr(
                params, "critical_endpoint_full_refresh_interval", 50
            ),
            timing_propagation_profile=getattr(
                params, "timing_propagation_profile", False
            ),
            timing_propagation_device=getattr(
                params, "timing_propagation_device", "inherit"
            ),
            timing_propagation_global_gpu=getattr(params, "gpu", 0),
            timing_propagation_global_gpu_id=getattr(params, "gpu_id", 0),
            timing_propagation_parity_check=getattr(
                params, "timing_propagation_parity_check", False
            ),
            timing_propagation_parity_atol_ps=getattr(
                params, "timing_propagation_parity_atol_ps", 1e-3
            ),
            timing_propagation_parity_rtol=getattr(
                params, "timing_propagation_parity_rtol", 1e-5
            ),
            timing_aggregation_mode=getattr(
                params, "timing_aggregation_mode", "hard"
            ),
            timing_aggregation_tau_ps=getattr(
                params, "timing_aggregation_tau_ps", 1.0
            ),
            production_fast_loop=getattr(params, "production_fast_loop", False),
            timing_lut_2d_native_op=getattr(
                params, "timing_lut_2d_native_op", "auto"
            ),
        )
        write_crash_stage_marker(
            "timing_propagation_op_construction",
            "done",
            resolved_device=str(getattr(timing_propagation_op, "device", "")),
        )
        return timing_propagation_op

    def build_pin2pin_net_weight(self, params, placedb, data_collections,
                                    pin_pos_op):
        
        pin2pin_attraction_op = pin2pin_attraction.Pin2PinAttraction(
            pin2pin_net_weight=placedb.pin2pin_net_weight,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            pairs=data_collections.pairs,
            weights=data_collections.weights,
            length=placedb.length)
        def build_pin2pin_op(pos):
            return pin2pin_attraction_op(pin_pos_op(pos))
        return build_pin2pin_op

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
            num_terminal_NIs=placedb.num_terminal_NIs,
            iopin_density_weight=_effective_iopin_density_weight(params, placedb),
            m2_pg_rail_density_boxes=_effective_m2_pg_rail_density_boxes(
                params,
                placedb,
                data_collections,
            ),
            m2_pg_rail_density_weight=_effective_m2_pg_rail_density_weight(
                params,
                placedb,
            ),
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
        node_size_x=None,
        node_size_y=None,
        num_movable_nodes=None,
        num_terminals=None,
        num_filler_nodes=None,
        sorted_node_map=None,
        movable_macro_mask=None,
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

        node_size_x = data_collections.node_size_x if node_size_x is None else node_size_x
        node_size_y = data_collections.node_size_y if node_size_y is None else node_size_y
        num_movable_nodes = (
            placedb.num_movable_nodes
            if num_movable_nodes is None
            else int(num_movable_nodes)
        )
        num_terminals = (
            placedb.num_terminals if num_terminals is None else int(num_terminals)
        )
        num_filler_nodes = (
            placedb.num_filler_nodes
            if num_filler_nodes is None
            else int(num_filler_nodes)
        )
        sorted_node_map = (
            data_collections.sorted_node_map
            if sorted_node_map is None
            else sorted_node_map
        )
        movable_macro_mask = (
            data_collections.movable_macro_mask
            if movable_macro_mask is None
            else movable_macro_mask
        )
        movable_size_x = node_size_x[:num_movable_nodes]
        movable_size_y = node_size_y[:num_movable_nodes]
        max_num_bins_x = np.ceil(
            (float(movable_size_x.max().detach().cpu().item()) + 2 * bin_size_x)
            / bin_size_x
        ) if int(movable_size_x.numel()) else 0
        max_num_bins_y = np.ceil(
            (float(movable_size_y.max().detach().cpu().item()) + 2 * bin_size_y)
            / bin_size_y
        ) if int(movable_size_y.numel()) else 0
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
        # Keep the shared tensor because inflation updates target density in place.
        # For fence regions, target density is different across regions.
        target_density = (
            data_collections.target_density
            if fence_regions is None
            else placedb.target_density_fence_region[region_id]
        )
        return electric_potential.ElectricPotential(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=num_movable_nodes,
            num_terminals=num_terminals,
            num_filler_nodes=num_filler_nodes,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=sorted_node_map,
            movable_macro_mask=movable_macro_mask,
            num_terminal_NIs=placedb.num_terminal_NIs,
            iopin_density_weight=_effective_iopin_density_weight(
                params, placedb, region_id=region_id, fence_regions=fence_regions),
            m2_pg_rail_density_boxes=_effective_m2_pg_rail_density_boxes(
                params,
                placedb,
                data_collections,
                region_id=region_id,
                fence_regions=fence_regions,
            ),
            m2_pg_rail_density_weight=_effective_m2_pg_rail_density_weight(
                params,
                placedb,
                region_id=region_id,
                fence_regions=fence_regions,
            ),
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
            self._timing_geometry_cache = None
            density = self.placement_density(self.data_collections.pos[0])
            # record initial density
            self.init_density = density.data.clone()
            self.density_weight_grad_precond = self.init_density.masked_scatter(
                self.init_density > 0, 1 / self.init_density[self.init_density > 0]
            )
            self.quad_penalty_coeff = (
                self.density_quad_coeff / 2 * self.density_weight_grad_precond
            )
            try:
                density.backward()
            finally:
                self._timing_geometry_cache = None
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

        joint_density_weight_max = None
        if (
            str(getattr(params, "flow_kind", "") or "").lower() == "joint"
            and str(getattr(params, "joint_quality_profile", "") or "")
            == "segment_count_direct_joint_v1"
            and not bool(getattr(params, "joint_segment_sizing_enabled", True))
        ):
            candidate_max = getattr(params, "joint_density_weight_max", None)
            if candidate_max is not None:
                candidate_max = float(candidate_max)
                if math.isfinite(candidate_max) and candidate_max > 0.0:
                    joint_density_weight_max = candidate_max
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
                if joint_density_weight_max is not None:
                    self.density_weight.clamp_(max=joint_density_weight_max)

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
                if joint_density_weight_max is not None:
                    self.density_weight.clamp_(max=joint_density_weight_max)

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

        pin_count_flag = bool(getattr(params, "precond_pin_count_flag", 0))
        logging.info("Gradient preconditioner: precond_pin_count_flag=%d", pin_count_flag)
        return PreconditionOp(
            placedb, data_collections, op_collections,
            precond_pin_count_flag=pin_count_flag,
            use_net_weights=not bool(params.routability_opt_flag) and not pin_count_flag,
        )

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
        try:
            import dreamplace.ops.gpugr.gpugr as gpugr_congestion
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GPUGR routability is enabled, but the optional DreamPlace "
                "GPUGR operator is not available"
            ) from exc
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
        if placedb.num_filler_nodes > 0:
            total_filler_area = (
                data_collections.node_size_x[-placedb.num_filler_nodes :]
                * data_collections.node_size_y[-placedb.num_filler_nodes :]
            ).sum()
        else:
            total_filler_area = data_collections.node_size_x.new_tensor(0.0)
        total_place_area = (
            total_movable_area + total_filler_area
        ) / data_collections.target_density
        virtual_area = 0.0
        if bool(getattr(params, "timing_opt_enabled", False)):
            from dreamplace.ops.routability.cooptimization_area import capture_area
            state = self._active_segment_count_state_for_virtual_density()
            if state is not None:
                width, height = self._segment_virtual_buffer_size()
                virtual_area = float(state.z_param.detach().sum()) * width * height
            total_place_area = capture_area(data_collections, placedb, virtual_area).capacity
        modularity_config = None
        if getattr(params, "modularity_inflation_flag", False):
            modularity_config = {
                "enabled": True,
                "params": params,
                "placedb": placedb,
                "data_collections": data_collections,
                "node_weights": data_collections.pin_weights[: placedb.num_movable_nodes],
            }
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
            total_whitespace_area=total_place_area - total_movable_area - virtual_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            inflation_area_budget_ratio=getattr(
                params, "inflation_area_budget_ratio", 0.1
            ),
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            modularity_config=modularity_config,
            params=params,
            virtual_cell_area=virtual_area,
        )

        def build_adjust_node_area_op(
            pos,
            route_utilization_map,
            pin_utilization_map,
            modularity_maps=None,
            inflation_round=0,
            fixed_target_area=None,
        ):
            result = adjust_node_area_op(
                pos,
                data_collections.node_size_x,
                data_collections.node_size_y,
                data_collections.pin_offset_x,
                data_collections.pin_offset_y,
                data_collections.target_density,
                route_utilization_map,
                pin_utilization_map,
                modularity_maps=modularity_maps,
                inflation_round=inflation_round,
                fixed_target_area=fixed_target_area,
            )
            if result[0]:
                enhanced_inflation_controller.sync_node_areas(data_collections)
            return result

        build_adjust_node_area_op._enhanced_adjust_node_area_impl = adjust_node_area_op
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
