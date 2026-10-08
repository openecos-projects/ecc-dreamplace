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
import json
import shutil
from typing import Any
import numpy as np
import logging
import torch
import copy
import matplotlib.pyplot as plt
import inspect
import hashlib
from dataclasses import replace
from dreamplace.flows import sizing_dynamics_policy, continuous_early_stop
from dreamplace.flows import l_shape_iteration
from dreamplace.flows import projected_pin_refresh
from dreamplace.flows import placement_step_profile
from dreamplace.ops.routability import l_shape_telemetry
from dreamplace.flows import sizing_debug, placement_metrics, metric_state, timing_artifacts, projection_context
from dreamplace.flows import timing_iteration, joint_iteration, placement_topology_iteration
from dreamplace.flows import segment_buffer_gradient, segment_joint_terminal, joint_commit_control
from dreamplace.flows import sizing_dynamics, discrete_sizing_step, segment_sizing_frame
from dreamplace.flows.inflation_s5b1 import InflationS5B1

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.BasicPlace as BasicPlace
import dreamplace.PlaceObj as PlaceObj
import dreamplace.NesterovAcceleratedGradientOptimizer as NesterovAcceleratedGradientOptimizer
import dreamplace.EvalMetrics as EvalMetrics
from dreamplace.ops.macro_overlap.macro_pin_geometry import restore_macro_pin_halo
import pdb
import dreamplace.ops.fence_region.fence_region as fence_region
from dreamplace.ops.irt_egr.egr_padding import apply_egr_padding, restore_egr_padding
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
from dreamplace.ops.timing_net_weighting.net_weighting import (
    PIN2PIN_PAIR_RESET_INTERVAL,
    clear_pin2pin_pair_accumulation,
    pin2pin_pair_accumulation_reset_due,
)
from dreamplace.ops.gate_projection.gate_projection import (
    GateProjectionOp,
    MainIdCandidateProvider,
    VectorizedMainIdCandidateProvider,
)
from dreamplace.ops.discrete_gradient_topk import (
    DiscreteVtCommit,
    build_discrete_gradient_topk_candidate_cache,
)
from dreamplace.ops.discrete_gradient_topk.runtime_cells import sync_runtime_cells
from dreamplace.ops.discrete_gradient_topk.runtime_timing_arcs import sync_timing_arc_lut_rows
from dreamplace.ops.size_interpolated_pin.sizing_limit_utils import (
    build_size_interpolated_pin_property_debug,
)


try:
    import torch_optimizer
except ImportError:
    torch_optimizer = None
try:
    import ncg_optimizer
except ImportError:
    ncg_optimizer = None

_AIMP_TRUE_VALUES = {"1", "true", "yes", "on"}
_AIMP_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name, default=False):
    value = os.environ.get(name, "")
    normalized = str(value).strip().lower()
    if normalized in _AIMP_TRUE_VALUES:
        return True
    if normalized in _AIMP_FALSE_VALUES:
        return False
    return bool(default)


def _param_flag(params, name, default=False):
    value = getattr(params, name, default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _AIMP_TRUE_VALUES:
        return True
    if normalized in _AIMP_FALSE_VALUES:
        return False
    return bool(default)

from dreamplace.ops.routability import gpugr_context
from dreamplace.ops.routability import route_map_utils
from dreamplace.ops.routability import route_evaluation
from dreamplace.ops.routability.l_shape_policy import LShapePolicy
from dreamplace.ops.routability.l_shape_electric_potential import (
    compute_fixed_macro_overlap_stats,
    compute_movable_displacement_stats,
)
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
    validate_ggr_l_shape_topology_params,
)
from dreamplace.ops.routability.routability_controller import RoutabilityController
from dreamplace.ops.irt_egr.egr_padding import apply_egr_padding, restore_egr_padding
from dreamplace.ops.routability import inflation_legalization
from dreamplace.flows.handoff_state import (
    build_continuation_seed,
    call_restart_after_topology_sync,
    apply_warm_schedule_state,
    capture_warm_schedule_state,
    clone_restart_value,
    resolve_restart_policy,
    restart_after_topology_sync,
    restart_probe_payload,
    restart_scalar,
)
from dreamplace.flows.handoff_transaction import (
    HandoffTransactionContext,
    build_handoff_command_failure_payload,
    build_session_mutation_result,
    execute_openroad_handoff_transaction,
)
from dreamplace.flows.metric_state import (
    apply_metric_snapshot,
    combined_timing_loss,
    finite_timing_scalar,
    flatten_metrics,
    last_metric_from_metrics,
    metric_has_nonfinite_value,
    metric_objective_is_nonfinite,
    metric_snapshot,
    should_skip_post_global_placement,
    timing_scalar,
    update_metric_timing,
    value_has_nonfinite,
)
from dreamplace.flows.real_size_transition import (
    RealSizeTransitionContext,
    TransitionProjection,
    capture_transaction,
    clone_transition_value,
    restore_transaction,
    sync_projected_real_size,
    transition_to_discrete,
)
from dreamplace.flows.real_size_transition_profile import (
    finalize_transition_accounting,
    process_peak_rss_bytes,
    record_transition_artifacts,
    record_transition_bytes,
    record_transition_stage,
    synchronize_transition_device,
    transition_profile,
    transition_profile_device,
    write_transition_profile,
)
from dreamplace.flows.timing_objective_policy import (
    diff_tdp_controls_direct_timing_objective,
    diff_tdp_enabled,
    diff_tdp_gate_status,
    diff_tdp_uses_direct_loss,
    diff_tdp_uses_gradient_net_weight,
    freeze_diff_tdp_step_status,
    legacy_net_weight_gate_status,
    net_weighting_nendpoints,
    net_weighting_npaths_budget,
    net_weighting_npaths_percent,
    new_diff_tdp_gate_summary,
    record_diff_tdp_gate_status,
    reset_legacy_net_weight_gate_summary,
    should_refresh_live_timing_before_objective,
    should_use_live_timing_loss,
    timing_placement_carrier,
)
from dreamplace.flows.timing_net_weighting import (
    TimingNetWeightingContext, LegacyWeightContext, update_legacy_weights,
    rebuild_pin2pin_pair_weights,
)
from dreamplace.flows.timing_gradient_projection import (
    TimingGradientProjectionContext,
    project_timing_gradient_to_net_weights,
)
from dreamplace.flows.timing_topology import (
    TimingTopologyContext,
    freeze_live_timing_topology,
    prepare_legacy_net_weight_timing_topology,
    publish_live_timing_topology,
    refresh_live_timing_topology,
)
from dreamplace.flows.segment_joint import (
    SegmentJointContext,
    apply_segment_overflow_milestone_transaction,
    apply_segment_rebootstrap_milestone_transaction,
    apply_segment_virtual_window_boundary,
    prepare_segment_direct_joint_iteration,
    rebuild_and_install_segment_pin2pin,
    run_segment_milestone_sizing_micro_loop,
)
from dreamplace.flows.segment_joint_terminal import (
    TerminalLaneContext,
    TerminalRouteBContext,
    rebuild_terminal_segment_route_b_state,
    reselect_terminal_segment_route_b,
    terminal_arc_cell_state_snapshot,
    terminal_route_b_round_budget,
    terminal_static_timing_snapshot,
)
from dreamplace.flows.timing_artifacts import (
    StageCompareWriterContext,
    artifact_bool,
    artifact_float,
    artifact_value,
    build_changed_cell_summary,
    build_stage_compare_stage_record,
    load_csv_rows_if_exists,
    load_json_artifact_if_exists,
    refresh_stage_compare_latest_artifact,
    write_stage_compare_latest_artifact,
    write_timing_stage_summary,
    write_post_projection_timing_summary,
)
from dreamplace.flows.projection_context import (
    ProjectionScoringContext,
    aggregate_violation_per_instance,
    build_projection_context,
    compute_local_timing_scores,
    current_pin_limit_context,
    endpoint_focus_penalty,
    prepare_design_pin_values,
    projection_objective_profile,
    projection_profile_term_weights,
)
from dreamplace.flows.projection_artifacts import (
    ProjectionFrameContext,
    build_projection_frame,
    projected_vt_distribution,
    projection_artifact_policy,
    projection_changed_inst_mask,
    write_projection_summary_artifact,
)
from dreamplace.flows.projection_runtime_refresh import (
    ProjectionTimingSnapshotContext,
    compute_post_projection_timing_snapshot,
)


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

    def __init__(self, params, placedb, timer):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        @param timer the timing analysis engine
        """
        super(NonLinearPlace, self).__init__(params, placedb, timer)
        # Keep the Python DB handle on the placer for debug/report helpers that
        # need stable access to exported pin names after initialization.
        self.placedb = placedb
        self.params = params
        self.last_discrete_gradient_topk_summary = {}
        self.discrete_gradient_topk_lambda_state = None
        self.discrete_gradient_topk_candidate_cache = None
        self.discrete_gradient_topk_candidate_cache_key = None
        self._quad_oscillation_state = None
        self._live_timing_topology_initialized = False
        self._stage_compare_run_started_at = time.time()
        self._diff_tdp_gate_summary = self._new_diff_tdp_gate_summary()
        self._legacy_net_weight_update_count = 0
        self._legacy_net_weight_gate_summary = {}
        self._legacy_net_weight_gate_open = False
        self._real_size_transition_done = False
        self._active_real_size_transition_profile = None
        self._active_real_size_transition = False
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

    def _placement_sizing_mode(self, params):
        return getattr(params, "placement_sizing_mode", "place_only")

    def _sizing_parameterization(self, params=None):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is not None:
            value = getattr(data_collections, "sizing_parameterization", None)
            if value:
                return str(value).strip().lower()
        return str(
            getattr(params, "sizing_parameterization", "logits") or "logits"
        ).strip().lower()

    def _real_size_enabled(self, params=None):
        return self._sizing_parameterization(params) == "real_size"

    def _effective_sizing_learning_rate(self, params, fallback):
        if self._real_size_enabled(params):
            return float(getattr(params, "real_size_learning_rate", 0.1))
        return float(getattr(params, "sizing_learning_rate", fallback))

    def _continuous_size_parameter(self):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is None:
            return None
        getter = getattr(data_collections, "get_continuous_size_parameter", None)
        if callable(getter):
            return getter()
        return getattr(data_collections, "size_logits", None)

    def _continuous_size_gradient(self):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is None:
            return None
        getter = getattr(data_collections, "get_continuous_size_gradient", None)
        if callable(getter):
            return getter()
        parameter = self._continuous_size_parameter()
        return None if parameter is None else parameter.grad

    def _mask_continuous_size_gradient(self):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is None:
            return 0
        masker = getattr(data_collections, "mask_continuous_size_gradient", None)
        if callable(masker):
            return int(masker())
        return 0

    def _project_real_size_after_optimizer_step(self, params):
        data_collections = getattr(self, "data_collections", None)
        if not self._real_size_enabled(params) or data_collections is None:
            return None
        projector = getattr(data_collections, "project_continuous_size", None)
        if not callable(projector):
            raise RuntimeError("real_size mode requires project_continuous_size")
        summary = projector()
        summary["parameterization"] = "real_size"
        summary["learning_rate"] = float(
            getattr(params, "real_size_learning_rate", 0.1)
        )
        return summary

    def _reset_sizing_optimizer_state(self, optimizer):
        if optimizer is None:
            return
        sizing_parameters = set()
        for group in optimizer.param_groups:
            if group.get("group_name") == "sizing":
                sizing_parameters.update(group.get("params", ()))
        for parameter in list(sizing_parameters):
            optimizer.state.pop(parameter, None)

    def _real_size_transition_profile_enabled(self, params):
        return _param_flag(params, "real_size_transition_profile", False)

    def _real_size_transition_backend(self, params):
        backend = str(
            getattr(params, "real_size_transition_backend", "reference")
            or "reference"
        ).strip().lower()
        if backend not in {"reference", "vectorized", "native"}:
            raise ValueError(
                "unsupported real_size_transition_backend: "
                f"{backend!r}; expected reference, vectorized, or native"
            )
        return backend

    def _real_size_transition_profile_latest_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_real_size_transition_profile_latest.json",
        )

    @staticmethod
    def _clone_transition_value(value):
        return clone_transition_value(value)

    def _real_size_transition_context(self):
        return RealSizeTransitionContext(
            data_collections=self.data_collections,
            placedb=getattr(self, "placedb", None),
            op_collections=getattr(self, "op_collections", None),
            size_params=getattr(self, "size_params", ()),
            last_global_place_model=getattr(self, "last_global_place_model", None),
            last_projection_frame=getattr(self, "last_projection_frame", None),
            discrete_gradient_topk_candidate_cache=getattr(
                self, "discrete_gradient_topk_candidate_cache", None
            ),
            discrete_gradient_topk_candidate_cache_key=getattr(
                self, "discrete_gradient_topk_candidate_cache_key", None
            ),
            active_profile=getattr(
                self, "_active_real_size_transition_profile", None
            ),
            defer_placedb_sync=self._defer_real_size_transition_placedb_sync(),
        )


    def _capture_real_size_transition_transaction(self, optimizer, frame):
        return capture_transaction(self._real_size_transition_context(), optimizer, frame)

    def _restore_real_size_transition_transaction(self, snapshot, optimizer):
        restored = restore_transaction(
            self._real_size_transition_context(),
            snapshot,
            optimizer,
        )
        self.size_params = restored["size_params"]
        self.last_projection_frame = restored["projection_frame"]
        for name, value in restored.get("projection_runtime_state", {}).items():
            setattr(self, name, value)
        self.discrete_gradient_topk_candidate_cache = restored[
            "discrete_gradient_topk_candidate_cache"
        ]
        self.discrete_gradient_topk_candidate_cache_key = restored[
            "discrete_gradient_topk_candidate_cache_key"
        ]

    def _real_size_transition_profile_trace_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_real_size_transition_profile.jsonl",
        )

    def _transition_profile_device(self):
        return transition_profile_device(getattr(self, "data_collections", None))

    def _synchronize_transition_device(self):
        synchronize_transition_device(getattr(self, "data_collections", None))

    def _record_real_size_transition_stage(self, name, started_at):
        record_transition_stage(
            getattr(self, "_active_real_size_transition_profile", None),
            name,
            started_at,
        )

    def _record_real_size_transition_bytes(self, name, *tensors):
        record_transition_bytes(
            getattr(self, "_active_real_size_transition_profile", None),
            name,
            *tensors,
        )

    def _record_real_size_transition_artifacts(self, paths):
        record_transition_artifacts(
            getattr(self, "_active_real_size_transition_profile", None),
            paths,
        )

    @staticmethod
    def _process_peak_rss_bytes():
        return process_peak_rss_bytes()

    @staticmethod
    def _finalize_real_size_transition_accounting(profile):
        finalize_transition_accounting(profile)

    def _defer_real_size_transition_placedb_sync(self):
        return bool(getattr(self, "_active_real_size_transition", False))

    def _write_real_size_transition_profile(self, params, profile):
        return write_transition_profile(
            profile,
            self._real_size_transition_profile_latest_path(params),
            self._real_size_transition_profile_trace_path(params),
        )

    def _maybe_transition_real_size_to_discrete(
        self,
        *,
        params,
        optimizer,
        iteration,
        model,
    ):
        warmup_steps = int(getattr(params, "real_size_warmup_steps", 0) or 0)
        transition_due = (
            self._real_size_enabled(params)
            and str(getattr(params, "real_size_execution_mode", "continuous_only"))
            == "warmup_to_discrete"
            and not getattr(self, "_real_size_transition_done", False)
            and int(iteration) + 1 >= warmup_steps
            and warmup_steps > 0
        )
        if not transition_due:
            return self._maybe_transition_real_size_to_discrete_impl(
                params=params,
                optimizer=optimizer,
                iteration=iteration,
                model=model,
            )

        if not self._real_size_transition_profile_enabled(params):
            self._active_real_size_transition = True
            try:
                return self._maybe_transition_real_size_to_discrete_impl(
                    params=params,
                    optimizer=optimizer,
                    iteration=iteration,
                    model=model,
                )
            finally:
                self._active_real_size_transition = False

        with transition_profile(
            self.data_collections,
            params,
            iteration,
            self._real_size_transition_backend(params),
            self._real_size_transition_profile_latest_path(params),
            self._real_size_transition_profile_trace_path(params),
        ) as profile:
            self._active_real_size_transition = True
            self._active_real_size_transition_profile = profile
            try:
                return self._maybe_transition_real_size_to_discrete_impl(
                    params=params,
                    optimizer=optimizer,
                    iteration=iteration,
                    model=model,
                )
            finally:
                self._active_real_size_transition_profile = None
                self._active_real_size_transition = False

    def _maybe_transition_real_size_to_discrete_impl(
        self,
        *,
        params,
        optimizer,
        iteration,
        model,
    ):
        """Delegate the ECC real-size transaction to its explicit state owner."""
        context = self._real_size_transition_context()
        context.last_global_place_model = model
        context.parameterization = self._sizing_parameterization(params)

        def project(project_params, *, stage_timing_summary):
            previous_owner_state = {
                name: getattr(self, name, None)
                for name in (
                    "last_projection_result",
                    "last_projection_frame",
                    "last_projection_artifact_paths",
                    "last_projection_runtime_refresh_summary",
                    "last_projection_runtime_refresh_summary_path",
                    "last_projection_pin_offset_runtime_summary_path",
                )
            }
            context.projection_runtime_state = previous_owner_state
            projection_paths = self._write_projection_artifacts(
                project_params,
                stage_timing_summary=stage_timing_summary,
            )
            projection_result = getattr(self, "last_projection_result", None)
            projection_frame = getattr(self, "last_projection_frame", None)
            return TransitionProjection(
                paths=projection_paths,
                result=projection_result,
                frame=projection_frame,
            )

        def refresh(refresh_params, placedb):
            return self._apply_projected_runtime_refresh(refresh_params, placedb)

        try:
            summary = transition_to_discrete(
                context,
                params,
                optimizer,
                iteration,
                project=project,
                refresh=refresh,
            )
        except Exception:
            restored = context.restored_owner_state
            if restored is not None:
                self.size_params = restored["size_params"]
                self.last_projection_frame = restored["projection_frame"]
                self.discrete_gradient_topk_candidate_cache = restored[
                    "discrete_gradient_topk_candidate_cache"
                ]
                self.discrete_gradient_topk_candidate_cache_key = restored[
                    "discrete_gradient_topk_candidate_cache_key"
                ]
                owner_state = restored.get("projection_runtime_state", {})
            else:
                owner_state = context.projection_runtime_state
            for name, value in owner_state.items():
                setattr(self, name, value)
            self._real_size_transition_done = False
            raise
        if summary is None:
            return None
        self._real_size_transition_done = context.transition_done
        self.size_params = context.size_params
        self.last_projection_frame = context.last_projection_frame
        self.discrete_gradient_topk_candidate_cache = (
            context.discrete_gradient_topk_candidate_cache
        )
        self.discrete_gradient_topk_candidate_cache_key = (
            context.discrete_gradient_topk_candidate_cache_key
        )
        profile = getattr(self, "_active_real_size_transition_profile", None)
        projection = getattr(self, "last_projection_result", None)
        frame = context.last_projection_frame
        runtime_summary = summary.get("runtime_refresh", {})
        if profile is not None and projection is not None and frame is not None:
            profile.update(
                {
                    "candidate_instance_count": int(projection.inst_ids.numel()),
                    "changed_instance_count": int(frame.changed_inst_ids.numel()),
                    "changed_pin_count": int(frame.changed_pin_ids.numel()),
                    "state_digest": frame.state_digest,
                    "topology_generation_before": int(runtime_summary.get(
                        "projection_frame_topology_generation_before",
                        frame.topology_generation_before,
                    )),
                    "topology_generation_after": int(runtime_summary.get(
                        "projection_frame_topology_generation_after",
                        frame.topology_generation_before,
                    )),
                    "timing_model_generation_before": int(runtime_summary.get(
                        "projection_frame_timing_model_generation_before",
                        frame.timing_model_generation_before,
                    )),
                    "timing_model_generation_after": int(runtime_summary.get(
                        "projection_frame_timing_model_generation_after",
                        frame.timing_model_generation_before,
                    )),
                    "topology_update_kind": runtime_summary.get(
                        "runtime_topology_refresh_kind"
                    ),
                    "topology_rebuilt": bool(
                        runtime_summary.get("runtime_topology_rebuilt", False)
                    ),
                    "placedb_cpu_mirror_updated": not self._defer_real_size_transition_placedb_sync(),
                }
            )
        logging.info(
            "real_size warm-up transition at iteration=%d: projected %d instances "
            "and switched to discrete logits",
            int(iteration),
            int(projection.inst_ids.numel()) if projection is not None else 0,
        )
        return summary

    def _sync_projected_real_size(self, projection_result):
        result = sync_projected_real_size(
            getattr(self, "data_collections", None),
            self._sizing_parameterization(),
            projection_result,
        )
        if result.get("applied"):
            self.discrete_gradient_topk_candidate_cache = None
            self.discrete_gradient_topk_candidate_cache_key = None
        return result

    @staticmethod
    def _validate_optimizer_parameter_groups(optimizer_name, parameter_groups):
        if str(optimizer_name).lower() != "nesterov":
            return
        unsupported = sorted(
            {
                str(group.get("group_name", "unknown"))
                for group in parameter_groups
                if group.get("group_name") != "placement"
            }
        )
        if unsupported:
            raise ValueError(
                "nesterov mode supports only placement parameters; "
                f"found groups={unsupported}"
            )

    def _openroad_backed_aimp_db_enabled(self, params):
        if params is None:
            return False
        return (
            bool(getattr(params, "openroad_backed_aimp_db", False))
            or getattr(params, "place_io_engine", None) in {"openroad", "ecc"}
            or getattr(params, "aimp_db_source", None) == "openroad"
        )

    def _size_debug_trace_enabled(self):
        flag = os.environ.get("AIMP_SIZE_DEBUG_TRACE", "")
        return str(flag).strip().lower() not in ("", "0", "false", "off", "no")

    def _size_debug_trace_path(self, params):
        return os.path.join(params.result_dir, f"{params.design_name()}_size_debug_trace.jsonl")

    def _discrete_gradient_topk_latest_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_discrete_gradient_topk_latest.json",
        )

    def _discrete_gradient_topk_trace_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_discrete_gradient_topk_trace.jsonl",
        )

    def _full_step_profile_enabled(self, params):
        return placement_step_profile.full_step_profile_enabled(params)

    def _full_step_profile_interval(self, params):
        return placement_step_profile.full_step_profile_interval(params)

    def _full_step_profile_latest_path(self, params):
        return placement_step_profile.full_step_profile_latest_path(params)

    def _full_step_profile_trace_path(self, params):
        return placement_step_profile.full_step_profile_trace_path(params)

    def _timing_obj_profile_enabled(self, params):
        return bool(getattr(params, "timing_obj_profile", False))

    def _joint_milestone_runtime_profile_enabled(self, params):
        return bool(
            getattr(params, "timing_obj_profile", False)
            or getattr(params, "timing_propagation_profile", False)
        )

    @staticmethod
    def _synchronized_milestone_profile_clock(enabled, reference):
        if enabled and torch.is_tensor(reference) and reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter()

    def _timing_obj_profile_latest_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_timing_obj_profile_latest.json",
        )

    def _timing_obj_profile_trace_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_timing_obj_profile.jsonl",
        )

    def _timing_objective_terms_artifact_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_timing_objective_terms_latest.json",
        )

    def _size_limit_samples_artifact_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_size_limit_samples.json",
        )

    def _reset_size_debug_trace(self, params):
        if not self._size_debug_trace_enabled():
            return
        path = self._size_debug_trace_path(params)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _reset_discrete_gradient_topk_trace(self, params):
        mode = str(getattr(params, "continuous_size_dynamics_mode", "none") or "none")
        if mode != "discrete_gradient_topk":
            return
        self.discrete_gradient_topk_candidate_cache = None
        self.discrete_gradient_topk_candidate_cache_key = None
        paths = [
            self._discrete_gradient_topk_latest_path(params),
            self._discrete_gradient_topk_trace_path(params),
        ]
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _reset_full_step_profile(self, params):
        return placement_step_profile.reset_full_step_profile(params)

    def _reset_timing_obj_profile(self, params):
        if not self._timing_obj_profile_enabled(params):
            return
        for path in (
            self._timing_obj_profile_latest_path(params),
            self._timing_obj_profile_trace_path(params),
        ):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _should_write_full_step_profile(self, params, iteration):
        return placement_step_profile.should_write_full_step_profile(params, iteration)

    def _write_full_step_profile_record(self, params, record):
        return placement_step_profile.write_full_step_profile_record(params, record)

    def _write_discrete_gradient_topk_artifacts(self, params, summary):
        if not isinstance(summary, dict) or summary.get("mode") != "discrete_gradient_topk":
            return None
        payload = dict(summary)
        payload["artifact"] = "discrete_gradient_topk_latest"
        payload["artifact_version"] = 1
        payload["design_name"] = params.design_name()
        latest_path = self._discrete_gradient_topk_latest_path(params)
        trace_path = self._discrete_gradient_topk_trace_path(params)
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        with open(latest_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        trace_payload = dict(payload)
        trace_payload["artifact"] = "discrete_gradient_topk_trace"
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
        self.last_discrete_gradient_topk_summary = payload
        self.last_discrete_gradient_topk_latest_path = latest_path
        self.last_discrete_gradient_topk_trace_path = trace_path
        return {"latest_path": latest_path, "trace_path": trace_path, "payload": payload}

    def _discrete_gradient_topk_cache_key(self, config, data_collections):
        inst_cell_id = getattr(data_collections, "inst_cell_id", None)
        flat_libcell_info = getattr(data_collections, "flat_libcell_info", None)
        inst_size_lower = getattr(data_collections, "inst_size_lower", None)
        inst_size_upper = getattr(data_collections, "inst_size_upper", None)
        if any(value is None for value in (inst_cell_id, flat_libcell_info, inst_size_lower, inst_size_upper)):
            return None
        return (
            bool(config["discrete_gradient_topk_preserve_vt"]),
            str(inst_cell_id.device),
            str(inst_size_lower.dtype),
            int(inst_cell_id.numel()),
            tuple(int(dim) for dim in flat_libcell_info.shape),
            tuple(int(dim) for dim in inst_size_lower.shape),
            tuple(int(dim) for dim in inst_size_upper.shape),
        )

    def _get_discrete_gradient_topk_candidate_cache(self, config, data_collections, device, dtype):
        cache_key = self._discrete_gradient_topk_cache_key(config, data_collections)
        if cache_key is None:
            return None
        if (
            self.discrete_gradient_topk_candidate_cache is not None
            and self.discrete_gradient_topk_candidate_cache_key == cache_key
        ):
            return self.discrete_gradient_topk_candidate_cache

        cache = build_discrete_gradient_topk_candidate_cache(
            inst_cell_id=data_collections.inst_cell_id.to(device=device).long(),
            flat_libcell_info=data_collections.flat_libcell_info.to(device=device, dtype=dtype),
            inst_size_lower=data_collections.inst_size_lower.to(device=device, dtype=dtype),
            inst_size_upper=data_collections.inst_size_upper.to(device=device, dtype=dtype),
            preserve_vt=config["discrete_gradient_topk_preserve_vt"],
        )
        self.discrete_gradient_topk_candidate_cache = cache
        self.discrete_gradient_topk_candidate_cache_key = cache_key
        return cache

    def _write_timing_objective_terms_artifact(self, params, model):
        payload = dict(getattr(model, "last_timing_objective_terms", {}) or {})
        model_params = getattr(model, "params", params)
        payload.setdefault("lane", getattr(model_params, "timing_objective_lane", "timing_only"))
        payload.setdefault("enabled_terms", ["timing"])
        payload.setdefault(
            "units",
            {
                "timing": "loss_proxy",
                "slew": "ns",
                "cap": "pF",
                "leakage": "mW",
            },
        )
        payload.setdefault("weights", {})
        payload.setdefault("raw_terms", {})
        payload.setdefault("weighted_terms", {})
        payload.setdefault("report_aligned_terms", {})
        payload["metadata"] = {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": self._placement_sizing_mode(model_params),
        }
        path = self._timing_objective_terms_artifact_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_timing_objective_terms_artifact = payload
        self.last_timing_objective_terms_artifact_path = path
        return {"path": path, "payload": payload}

    def _write_size_limit_samples_artifact(self, params, pin_ids=None):
        data_collections = getattr(self, "data_collections", None)
        if data_collections is None:
            return None
        pin2node_map = getattr(data_collections, "pin2node_map", None)
        if pin2node_map is None:
            return None
        pin_count = int(torch.as_tensor(pin2node_map).numel())
        if pin_count <= 0:
            return None
        flat_slew_limit = getattr(data_collections, "flat_lib_pin_slew_limit", None)
        flat_cap_limit = getattr(data_collections, "flat_lib_pin_cap_limit", None)
        if flat_slew_limit is None or flat_cap_limit is None:
            return None
        if pin_ids is None:
            pin_ids = torch.arange(min(pin_count, 8), dtype=torch.long)
        payload = build_size_interpolated_pin_property_debug(
            data_collections,
            pin_ids=pin_ids,
            property_specs={
                "slew_limit": {
                    "flat_pin_values": flat_slew_limit,
                    "unit": "ps",
                },
                "cap_limit": {
                    "flat_pin_values": flat_cap_limit,
                    "unit": "pF",
                },
            },
        )
        payload["metadata"] = {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": self._placement_sizing_mode(params),
        }
        path = self._size_limit_samples_artifact_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_size_limit_samples_artifact = payload
        self.last_size_limit_samples_artifact_path = path
        return {"path": path, "payload": payload}

    def _sizing_debug_context(self):
        return sizing_debug.SizingDebugContext(
            getattr(self, "data_collections", None), self._continuous_size_parameter,
            self._sizing_parameterization, self._size_debug_trace_enabled,
            self._append_size_debug_trace,
            getattr(self, "last_continuous_size_dynamics_summary", {}),
        )

    def _capture_optional_tensor(self, tensor):
        return sizing_debug.capture_optional_tensor(tensor)

    def _capture_size_debug_snapshot(self):
        return sizing_debug.capture_size_debug_snapshot(self._sizing_debug_context())

    def _tensor_delta_stats(self, before, after):
        return sizing_debug.tensor_delta_stats(before, after)

    def _size_debug_grad_stats(self):
        return sizing_debug.size_debug_grad_stats(self._sizing_debug_context())

    def _size_debug_learning_rates(self, optimizer):
        return sizing_debug.size_debug_learning_rates(optimizer)

    def _append_size_debug_trace(self, params, record):
        if not self._size_debug_trace_enabled():
            return
        path = self._size_debug_trace_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _size_debug_objective_terms(self, model):
        return sizing_debug.size_debug_objective_terms(model)

    def _continuous_size_dynamics_config(self, params):
        return sizing_dynamics_policy.continuous_size_dynamics_config(params)

    def _continuous_size_dynamics_phase(self, iteration, total_iterations, config):
        return sizing_dynamics_policy.continuous_size_dynamics_phase(iteration, total_iterations, config)

    def _sizing_dynamics_context(self):
        return sizing_dynamics.SizingDynamicsContext(
            accept_reject=self._apply_late_accept_reject_size_dynamics,
            apply_metric=self._apply_metric_snapshot,
            combined_loss=self._combined_timing_loss,
            commit_feedback=self._apply_late_commit_feedback_size_dynamics,
            config=self._continuous_size_dynamics_config,
            data_collections=getattr(self, "data_collections", None),
            discrete_step=self._apply_discrete_sizing_dynamics,
            evaluate_candidate=self._evaluate_continuous_size_dynamics_candidate_metric,
            metric_snapshot=self._metric_snapshot_from_metric,
            metric_values=self._continuous_size_dynamics_metric_values,
            op_collections=getattr(self, "op_collections", None),
            phase=self._continuous_size_dynamics_phase,
            record_timing=self._record_timing_metrics,
            timing_scalar=self._timing_scalar,
            update_guard=self._update_continuous_size_dynamics_state,
        )

    def _discrete_sizing_step_context(self):
        return discrete_sizing_step.DiscreteSizingStepContext(
            cache=lambda: self.discrete_gradient_topk_candidate_cache,
            cache_key=lambda: self.discrete_gradient_topk_candidate_cache_key,
            candidate_table=self._get_discrete_gradient_topk_candidate_cache,
            config=self._continuous_size_dynamics_config,
            data_collections=getattr(self, "data_collections", None),
            oscillation=lambda: getattr(self, "_quad_oscillation_state", None),
            publish_candidate_cache=self._publish_discrete_sizing_cache,
            publish_oscillation=self._publish_sizing_oscillation,
            select=self._select_discrete_gradient_topk,
            sizing_mode=self._placement_sizing_mode,
            skip_artifact=self._should_skip_discrete_gradient_topk_artifact_write,
            sync_cells=self._sync_discrete_gradient_topk_runtime_cells,
            write_artifact=self._write_discrete_gradient_topk_artifacts,
        )

    def _segment_sizing_frame_context(self):
        return segment_sizing_frame.SegmentSizingFrameContext(
            coordinator=getattr(self, "joint_coordinator", None),
            data_collections=getattr(self, "data_collections", None),
            digest=self._digest_named_tensors,
            freeze_frame=self._freeze_segment_milestone_sizing_frame,
            profile_clock=self._synchronized_milestone_profile_clock,
            profile_enabled=self._joint_milestone_runtime_profile_enabled,
            select=self._select_discrete_gradient_topk,
            sync_cells=self._sync_discrete_gradient_topk_runtime_cells,
        )

    def _publish_discrete_sizing_cache(self, table, key):
        self.discrete_gradient_topk_candidate_cache = table
        self.discrete_gradient_topk_candidate_cache_key = key

    def _publish_sizing_oscillation(self, state):
        self._quad_oscillation_state = state

    def _apply_discrete_sizing_dynamics(self, params, before_snapshot, iteration, total_iterations, config, summary, size_logits, before_logits, phase, progress, raw_delta_max_abs):
        context = self._discrete_sizing_step_context()
        result = discrete_sizing_step.apply_discrete_sizing_dynamics(context, params, before_snapshot, iteration, total_iterations, config, summary, size_logits, before_logits, phase, progress, raw_delta_max_abs)
        self.last_continuous_size_dynamics_summary = context.summary
        if context.topk_summary is not discrete_sizing_step.UNSET:
            self.last_discrete_gradient_topk_summary = context.topk_summary
        return result

    def _build_continuous_size_dynamics_state(self, params):
        return sizing_dynamics.build_continuous_size_dynamics_state(self._sizing_dynamics_context(), params)

    def _continuous_size_dynamics_metric_values(self, metric):
        return sizing_dynamics.continuous_size_dynamics_metric_values(self._sizing_dynamics_context(), metric)

    def _update_continuous_size_dynamics_state(self, state, metric, iteration, phase):
        return sizing_dynamics.update_continuous_size_dynamics_state(self._sizing_dynamics_context(), state, metric, iteration, phase)

    def _apply_late_accept_reject_size_dynamics(
        self,
        state,
        before_logits,
        metric,
        phase,
        phase_lr_scale,
        config,
        candidate_metric_snapshot=None,
    ):
        return sizing_dynamics.apply_late_accept_reject_size_dynamics(self._sizing_dynamics_context(), state, before_logits, metric, phase, phase_lr_scale, config, candidate_metric_snapshot)

    def _apply_late_commit_feedback_size_dynamics(
        self,
        state,
        before_logits,
        metric,
        phase,
        phase_lr_scale,
        config,
        candidate_metric_snapshot=None,
        baseline_metric_snapshot=None,
    ):
        return sizing_dynamics.apply_late_commit_feedback_size_dynamics(self._sizing_dynamics_context(), state, before_logits, metric, phase, phase_lr_scale, config, candidate_metric_snapshot, baseline_metric_snapshot)

    @staticmethod
    def _digest_named_tensors(named_tensors):
        digest = hashlib.sha256()
        for name, value in named_tensors:
            digest.update(str(name).encode("utf-8"))
            if value is None:
                digest.update(b"<none>")
                continue
            tensor = torch.as_tensor(value).detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def _select_discrete_gradient_topk(
        self,
        *,
        params,
        size_logits,
        size_grad,
        policy,
    ):
        return discrete_sizing_step.select_discrete_gradient_topk(self._discrete_sizing_step_context(), params=params, size_logits=size_logits, size_grad=size_grad, policy=policy)

    def _freeze_segment_milestone_sizing_frame(
        self,
        *,
        params,
        model,
        pos,
        event,
    ):
        return segment_sizing_frame.freeze_segment_milestone_sizing_frame(self._segment_sizing_frame_context(), params=params, model=model, pos=pos, event=event)

    def _capture_segment_milestone_sizing_frame(
        self,
        *,
        params,
        model,
        pos,
        event,
        evaluate_objective,
        advance_stage=True,
    ):
        return segment_sizing_frame.capture_segment_milestone_sizing_frame(self._segment_sizing_frame_context(), params=params, model=model, pos=pos, event=event, evaluate_objective=evaluate_objective, advance_stage=advance_stage)

    def _apply_segment_milestone_discrete_sizing(
        self,
        *,
        params,
        event,
        sizing_frame,
    ):
        return segment_sizing_frame.apply_segment_milestone_discrete_sizing(self._segment_sizing_frame_context(), params=params, event=event, sizing_frame=sizing_frame)

    def _apply_continuous_size_logits_dynamics(
        self,
        params,
        before_snapshot,
        iteration,
        total_iterations,
        runtime_state=None,
        metric=None,
        model=None,
        pos=None,
    ):
        context = self._sizing_dynamics_context()
        result = sizing_dynamics.apply_continuous_size_logits_dynamics(context, params, before_snapshot, iteration, total_iterations, runtime_state, metric, model, pos)
        if context.summary is not sizing_dynamics.UNSET:
            self.last_continuous_size_dynamics_summary = context.summary
        return result


    @staticmethod
    def _runtime_liberty_arc_transition_map(
        source_cell,
        target_cell,
        arc_start,
        flat_arc_info,
    ):
        from dreamplace.ops.discrete_gradient_topk.runtime_timing_arcs import (
            map_liberty_arc_transition,
        )

        return map_liberty_arc_transition(
            source_cell,
            target_cell,
            arc_start,
            flat_arc_info,
        )


    def _sync_runtime_timing_arc_lut_rows(self, changed_inst_ids):
        return sync_timing_arc_lut_rows(
            self.data_collections,
            getattr(self, "placedb", None),
            getattr(getattr(self, "op_collections", None), "timing_propagation_op", None),
            changed_inst_ids,
        )

    def _sync_discrete_gradient_topk_runtime_cells(self, params, summary):
        return sync_runtime_cells(
            params, summary,
            data_collections=getattr(self, "data_collections", None),
            placedb=getattr(self, "placedb", None),
            timing_op=getattr(getattr(self, "op_collections", None), "timing_propagation_op", None),
            model=getattr(self, "last_global_place_model", None),
            direct_joint=self._segment_direct_joint_enabled(),
        )

    def _maybe_write_size_debug_trace(
        self,
        params,
        iteration,
        metric,
        model,
        optimizer,
        before_snapshot,
        after_snapshot,
        grad_stats,
    ):
        return sizing_debug.maybe_write_size_debug_trace(self._sizing_debug_context(), params, iteration, metric, model, optimizer, before_snapshot, after_snapshot, grad_stats)

    def _should_use_live_timing_loss(self, params):
        return should_use_live_timing_loss(
            params,
            getattr(self, "_diff_tdp_step_status", None),
        )

    def _should_refresh_live_timing_before_objective(self, params):
        return should_refresh_live_timing_before_objective(
            params,
            getattr(self, "_diff_tdp_step_status", None),
        )

    def _should_skip_live_timing_topology_refresh(self, params):
        coordinator = getattr(self, "joint_coordinator", None)
        if (
            coordinator is not None
            and bool(getattr(coordinator, "is_segment_direct_joint", False))
            and getattr(self.op_collections, "steiner_topo_op", None) is not None
            and bool(self.op_collections.steiner_topo_op.topology_frozen)
        ):
            return True
        return (
            self._placement_sizing_mode(params) == "size_only"
            and bool(getattr(params, "production_fast_loop", False))
            and bool(getattr(self, "_live_timing_topology_initialized", False))
            and not _env_flag("AIMP_DISABLE_SIZE_ONLY_TOPOLOGY_REFRESH_SKIP")
        )

    def _should_skip_discrete_gradient_topk_artifact_write(self, params):
        return (
            self._placement_sizing_mode(params) == "size_only"
            and bool(getattr(params, "production_fast_loop", False))
            and str(getattr(params, "continuous_size_dynamics_mode", "none") or "none")
            == "discrete_gradient_topk"
            and not _env_flag("AIMP_DISABLE_DISCRETE_TOPK_ARTIFACT_SKIP")
        )

    def _diff_tdp_enabled(self, params):
        return diff_tdp_enabled(params)

    def _timing_placement_carrier(self, params):
        return timing_placement_carrier(params)

    def _diff_tdp_uses_direct_loss(self, params):
        return diff_tdp_uses_direct_loss(params)

    def _diff_tdp_uses_gradient_net_weight(self, params):
        return diff_tdp_uses_gradient_net_weight(params)

    def _diff_tdp_controls_direct_timing_objective(self, params):
        return diff_tdp_controls_direct_timing_objective(params)

    def _new_diff_tdp_gate_summary(self):
        return new_diff_tdp_gate_summary()

    def _reset_diff_tdp_gate_summary(self, params):
        summary = new_diff_tdp_gate_summary(params)
        self._diff_tdp_gate_summary = summary
        self._timing_grad_balance_state = None
        self._diff_tdp_step_status = {
            "enabled": False,
            "timing_objective_active": False,
            "topology_refresh_due": False,
            "reason": "not_evaluated",
        }
        self._diff_tdp_net_weights_are_projected = False
        return summary

    def _record_diff_tdp_gate_status(self, params, gate_status):
        summary = getattr(self, "_diff_tdp_gate_summary", None)
        if not isinstance(summary, dict):
            summary = self._reset_diff_tdp_gate_summary(params)
        summary = record_diff_tdp_gate_status(summary, gate_status)
        return summary

    def _reset_diff_tdp_net_weights_if_needed(self, params, placedb, *, iteration, reason):
        if not self._diff_tdp_uses_gradient_net_weight(params):
            return False
        if not bool(getattr(self, "_diff_tdp_net_weights_are_projected", False)):
            return False
        net_weights = getattr(self.data_collections, "net_weights", None)
        if net_weights is not None:
            net_weights.fill_(1.0)
        if hasattr(placedb, "net_weights"):
            placedb.net_weights[:] = 1.0
        self._diff_tdp_net_weights_are_projected = False
        summary = getattr(self, "_diff_tdp_gate_summary", None)
        if isinstance(summary, dict):
            summary["net_weight_reset_count"] = int(
                summary.get("net_weight_reset_count", 0)
            ) + 1
            summary["last_reset_iteration"] = int(iteration)
            summary["last_reset_reason"] = str(reason)
        logging.info(
            "Diff TDP gate reset projected net weights at iteration=%d reason=%s",
            int(iteration),
            str(reason),
        )
        return True

    def _diff_tdp_gate_status(self, params, iteration, overflow):
        return diff_tdp_gate_status(params, iteration, overflow)

    def _freeze_diff_tdp_step_status(self, params, iteration, overflow):
        previous_status = getattr(self, "_diff_tdp_step_status", None) or {}
        status = freeze_diff_tdp_step_status(
            params,
            iteration,
            overflow,
            previous_status,
            getattr(self, "_live_timing_topology_initialized", False),
        )
        self._diff_tdp_step_status = dict(status)
        self._record_diff_tdp_gate_status(params, status)
        return status

    def _initialize_timing_grad_balance_if_needed(
        self,
        params,
        model,
        pos,
        *,
        iteration,
        overflow,
    ):
        if (
            not self._diff_tdp_enabled(params)
            or not self._diff_tdp_uses_direct_loss(params)
            or self._placement_sizing_mode(params) == "size_only"
            or not bool(
                getattr(self, "_diff_tdp_step_status", {}).get(
                    "timing_objective_active",
                    False,
                )
            )
        ):
            return copy.deepcopy(
                getattr(self, "_timing_grad_balance_state", None)
            )

        state = getattr(self, "_timing_grad_balance_state", None)
        if state is not None:
            model.apply_timing_grad_balance_state(state)
            if state.get("initialized"):
                return copy.deepcopy(state)

        state = model.initialize_timing_grad_balance(
            pos,
            iteration=iteration,
            overflow=overflow,
        )
        self._timing_grad_balance_state = copy.deepcopy(state)
        return copy.deepcopy(state)

    def _diff_tdp_net_weight_artifact_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_gradient_projected_net_weight_latest.json",
        )

    def _write_diff_tdp_net_weight_artifact(self, params, payload):
        path = self._diff_tdp_net_weight_artifact_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_diff_tdp_net_weight_artifact = payload
        self.last_diff_tdp_net_weight_artifact_path = path
        return path

    def _legacy_net_weight_artifact_path(self, params):
        return os.path.join(
            params.result_dir,
            f"{params.design_name()}_legacy_net_weight_latest.json",
        )

    def _write_legacy_net_weight_artifact(self, params, payload):
        path = self._legacy_net_weight_artifact_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_legacy_net_weight_artifact = payload
        self.last_legacy_net_weight_artifact_path = path
        return path

    def _reset_legacy_net_weight_gate_summary(self, params):
        summary = reset_legacy_net_weight_gate_summary(params)
        self._legacy_net_weight_gate_summary = summary
        self._legacy_net_weight_gate_open = False
        self._legacy_net_weight_update_count = 0
        self.last_legacy_net_weight_artifact = None
        self.last_legacy_net_weight_artifact_path = None
        return summary

    def _legacy_net_weight_gate_status(self, params, iteration, overflow):
        summary = getattr(self, "_legacy_net_weight_gate_summary", None)
        status, gate_open, summary = legacy_net_weight_gate_status(
            params,
            iteration,
            overflow,
            summary,
            getattr(self, "_legacy_net_weight_gate_open", False),
        )
        self._legacy_net_weight_gate_open = gate_open
        self._legacy_net_weight_gate_summary = summary
        return status

    @staticmethod
    def _net_weighting_npaths_percent(params):
        return net_weighting_npaths_percent(params)

    @staticmethod
    def _net_weighting_npaths_budget(params, placedb):
        return net_weighting_npaths_budget(params, placedb)

    @staticmethod
    def _net_weighting_nendpoints(params, placedb=None):
        return net_weighting_nendpoints(params, placedb)

    def _maybe_reset_pin2pin_pair_accumulation(
        self, placedb, next_update_count
    ):
        if not pin2pin_pair_accumulation_reset_due(next_update_count):
            return False
        self._clear_pin2pin_pair_accumulation(
            placedb,
            next_update_count=next_update_count,
            reason="periodic_accumulation_reset",
        )
        return True

    def _clear_pin2pin_pair_accumulation(
        self,
        placedb,
        *,
        next_update_count,
        reason,
    ):
        reset = clear_pin2pin_pair_accumulation(placedb)
        summary = self._legacy_net_weight_gate_summary
        summary["pin2pin_pair_reset_count"] = int(
            summary.get("pin2pin_pair_reset_count", 0)
        ) + 1
        summary.setdefault("pin2pin_pair_reset_update_counts", []).append(
            int(next_update_count)
        )
        summary["last_pin2pin_pair_reset_before_update"] = int(next_update_count)
        summary.setdefault("pin2pin_pair_reset_reasons", []).append(str(reason))
        logging.info(
            "Reset Pin2Pin pair accumulation before update count=%d; "
            "cleared_pairs=%d interval=%d reason=%s",
            int(next_update_count),
            int(reset["cleared_pair_count"]),
            PIN2PIN_PAIR_RESET_INTERVAL,
            str(reason),
        )
        return reset

    def _record_legacy_net_weight_update(
        self,
        params,
        placedb,
        *,
        iteration,
        npaths,
        wns,
        tns,
        update_ms,
        gate_status,
        pin2pin_objective,
        pin2pin_path_pair_backend,
        pin2pin_attraction_backend,
        pin2pin_pair_reset_before_update=False,
        update_reason=None,
    ):
        self._legacy_net_weight_update_count = int(
            getattr(self, "_legacy_net_weight_update_count", 0)
        ) + 1
        context = timing_artifacts.LegacyWeightReportContext(
            self._legacy_net_weight_update_count,
            self._legacy_net_weight_gate_summary, self._write_legacy_net_weight_artifact,
        )
        return timing_artifacts.record_legacy_net_weight_update(
            context, params, placedb, iteration=iteration, npaths=npaths, wns=wns, tns=tns, update_ms=update_ms, gate_status=gate_status, pin2pin_objective=pin2pin_objective, pin2pin_path_pair_backend=pin2pin_path_pair_backend, pin2pin_attraction_backend=pin2pin_attraction_backend, pin2pin_pair_reset_before_update=pin2pin_pair_reset_before_update, update_reason=update_reason,
        )

    def _legacy_weight_context(self):
        return LegacyWeightContext(
            self.data_collections, self.device, self.pos,
            getattr(self.op_collections, "timing_propagation_op", None),
            self._maybe_reset_pin2pin_pair_accumulation,
            self._net_weighting_nendpoints, self._net_weighting_npaths_budget,
            self._net_weighting_npaths_percent,
            self._prepare_legacy_net_weight_timing_topology,
            self._record_legacy_net_weight_update, self._record_timing_metrics,
            self._write_legacy_net_weight_artifact,
            lambda: int(self._legacy_net_weight_update_count) + 1,
        )

    def _rebuild_pin2pin_pair_weights(
        self,
        params,
        placedb,
        model,
        pos,
        *,
        iteration,
        gate_status,
        update_reason,
        force_clear_accumulation,
    ):
        context = TimingNetWeightingContext(
            params=params,
            placedb=placedb,
            model=model,
            pos=pos,
            data_collections=self.data_collections,
            op_collections=self.op_collections,
            iteration=int(iteration),
            gate_status=dict(gate_status),
            update_reason=str(update_reason),
            force_clear_accumulation=bool(force_clear_accumulation),
            profile_enabled=self._joint_milestone_runtime_profile_enabled(params),
            profile_clock=lambda: self._synchronized_milestone_profile_clock(
                self._joint_milestone_runtime_profile_enabled(params),
                pos,
            ),
            prepare_timing_topology=self._prepare_legacy_net_weight_timing_topology,
            record_timing_metrics=self._record_timing_metrics,
            timing_scalar=self._timing_scalar,
            clear_pair_accumulation=self._clear_pin2pin_pair_accumulation,
            maybe_reset_pair_accumulation=self._maybe_reset_pin2pin_pair_accumulation,
            record_update=self._record_legacy_net_weight_update,
            write_artifact=lambda payload: self._write_legacy_net_weight_artifact(
                params, payload
            ),
            endpoint_budget=lambda: self._net_weighting_nendpoints(params, placedb),
            next_update_count=lambda: int(self._legacy_net_weight_update_count) + 1,
        )
        return rebuild_pin2pin_pair_weights(context)

    def _project_timing_gradient_to_net_weights(
        self,
        params,
        placedb,
        model,
        pos,
        *,
        iteration,
        gate_status,
        topology_prepared=False,
    ):
        context = TimingGradientProjectionContext(
            params=params,
            placedb=placedb,
            model=model,
            pos=pos,
            data_collections=self.data_collections,
            op_collections=self.op_collections,
            iteration=int(iteration),
            gate_status=dict(gate_status),
            topology_prepared=topology_prepared,
            timing_carrier=self._timing_placement_carrier(params),
            record_timing_metrics=self._record_timing_metrics,
            write_artifact=lambda payload: self._write_diff_tdp_net_weight_artifact(
                params, payload
            ),
        )
        payload, projected = project_timing_gradient_to_net_weights(context)
        if payload.get("status") != "skipped":
            self._diff_tdp_net_weights_are_projected = projected
        return payload

    def _should_apply_gp_noise(self, params):
        return (
            params.gp_noise_ratio > 0.0
            and self._placement_sizing_mode(params) != "size_only"
        )

    def _overflow_blocks_placement_success(self, params):
        # Joint no-sizing still owns the segment Route-B/finalization path;
        # its overflow is a joint-flow gate, not standalone placement failure.
        return (
            self._placement_sizing_mode(params) == "place_only"
            and not self._joint_flow_enabled(params)
        )

    def _timing_topology_context(self):
        return TimingTopologyContext(
            data_collections=self.data_collections,
            op_collections=self.op_collections,
            placedb=getattr(self, "placedb", None),
            params=getattr(self, "params", None),
            segment_direct_joint_enabled=self._segment_direct_joint_enabled(),
            record_stage=self._record_real_size_transition_stage,
            record_bytes=self._record_real_size_transition_bytes,
            refresh_callback=self._refresh_live_timing_topology,
        )


    def _publish_live_timing_topology(self, new_x, new_y, *, update_kind):
        topology = publish_live_timing_topology(
            self._timing_topology_context(),
            new_x,
            new_y,
            update_kind=update_kind,
        )
        self._live_timing_topology_initialized = True
        return topology

    def _refresh_live_timing_topology(self, pos):
        topology = refresh_live_timing_topology(
            self._timing_topology_context(),
            pos,
        )
        self._live_timing_topology_initialized = True
        return topology

    def _freeze_live_timing_topology(self, pos):
        return freeze_live_timing_topology(self._timing_topology_context(), pos)

    def _record_segment_joint_initial_inst_cell_id(self, model):
        self.segment_joint_initial_inst_cell_id = (
            model.data_collections.inst_cell_id.detach()
            .cpu()
            .long()
            .contiguous()
            .clone()
        )


    def _segment_joint_context(self):
        return SegmentJointContext(
            coordinator=getattr(self, "joint_coordinator", None),
            freeze_live_timing_topology=self._freeze_live_timing_topology,
            refresh_live_timing_topology=self._refresh_live_timing_topology,
            record_initial_inst_cell_id=self._record_segment_joint_initial_inst_cell_id,
            data_collections=getattr(self, "data_collections", None),
            apply_discrete_sizing=self._apply_segment_milestone_discrete_sizing,
            compact_sizing=self._compact_segment_milestone_sizing_summary,
            compute_buffer_gradient=self._compute_refreshed_segment_buffer_gradient,
            profile_enabled=self._joint_milestone_runtime_profile_enabled,
            profile_clock=self._synchronized_milestone_profile_clock,
            capture_sizing_frame=self._capture_segment_milestone_sizing_frame,
            timing_scalar=self._timing_scalar,
            rebuild_pair_weights=self._rebuild_pin2pin_pair_weights,
            write_net_weight_artifact=self._write_legacy_net_weight_artifact,
        )


    def _size_only_frozen_topology_enabled(self, params):
        return (
            getattr(getattr(self, "placedb", None), "gr_sizing", None) is None
            and self._placement_sizing_mode(params) == "size_only"
            and bool(getattr(params, "with_sta", False))
            and getattr(getattr(self, "op_collections", None), "steiner_topo_op", None)
            is not None
        )

    def _activate_size_only_frozen_topology(self, params, pos):
        if not self._size_only_frozen_topology_enabled(params):
            return None
        topo_op = self.op_collections.steiner_topo_op
        if bool(getattr(topo_op, "topology_frozen", False)):
            return int(topo_op.frozen_topology_generation)
        if not bool(getattr(self, "_live_timing_topology_initialized", False)):
            self._refresh_live_timing_topology(pos)
        generation = int(topo_op.freeze_topology())
        topology = getattr(self.data_collections, "buffering_timing_topology", None)
        if isinstance(topology, dict):
            topology["topology_frozen"] = True
            topology["frozen_topology_generation"] = generation
            topology["topology_update_kind"] = "size_only_stage_freeze"
        self._size_only_frozen_topology_generation = generation
        return generation

    def _release_size_only_frozen_topology(self, params):
        if not self._size_only_frozen_topology_enabled(params):
            return None
        topo_op = self.op_collections.steiner_topo_op
        if not bool(getattr(topo_op, "topology_frozen", False)):
            return None
        generation = topo_op.unfreeze_topology()
        topology = getattr(self.data_collections, "buffering_timing_topology", None)
        if isinstance(topology, dict):
            topology["topology_frozen"] = False
            topology["frozen_topology_generation"] = None
            topology["topology_update_kind"] = "size_only_stage_release"
        self._size_only_frozen_topology_generation = None
        return generation

    def _segment_direct_joint_enabled(self):
        coordinator = getattr(self, "joint_coordinator", None)
        return bool(coordinator is not None and coordinator.is_segment_direct_joint)

    def _prepare_legacy_net_weight_timing_topology(self, pos):
        return prepare_legacy_net_weight_timing_topology(
            self._timing_topology_context(),
            pos,
        )

    @staticmethod
    def _compact_segment_milestone_sizing_summary(summary):
        summary = dict(summary or {})
        return {
            key: copy.deepcopy(summary.get(key))
            for key in (
                "status",
                "reason",
                "event_id",
                "iteration",
                "milestone",
                "frame",
                "policy",
                "num_sizeable_instances",
                "num_up_eligible",
                "num_up_improving_instances",
                "num_up_selected",
                "num_down_improving_instances",
                "num_changed_instances",
                "ranking",
                "applied_instance_ids",
                "applied_cell_ids",
                "applied_sizes",
                "applied_legal_index_deltas",
                "action_digest",
                "runtime_cell_state_synced",
                "runtime_synced_instances",
                "runtime_sync_reason",
                "runtime_cell_state_generation",
                "timing_cache_generation",
                "runtime_geometry_synced",
                "runtime_pin_geometry_synced",
                "runtime_pin_cap_view",
                "movable_area_before_internal",
                "movable_area_after_internal",
                "area_delta_internal",
                "density_visible_area_delta_internal",
                "legal_size_max_abs_gap",
                "timing_cache_invalidated",
                "input_state_digest",
                "output_state_digest",
                "runtime_ms",
            )
            if key in summary
        }

    def _segment_gradient_context(self):
        return segment_buffer_gradient.SegmentGradientContext(
            self.joint_coordinator, self.data_collections, self._digest_named_tensors,
            self._joint_milestone_runtime_profile_enabled,
            self._synchronized_milestone_profile_clock, self._refresh_live_timing_topology,
        )

    def _joint_commit_context(self):
        return joint_commit_control.JointCommitContext(
            joint_coordinator=getattr(self, "joint_coordinator", None),
            trace=getattr(self, "joint_buffer_commit_trace", None),
            artifact=getattr(self, "last_joint_flow_artifact", None),
            last_metrics=getattr(self, "last_metrics", []),
            physical_context=self._final_joint_physical_context,
            projection_context=self._final_segment_joint_projection_context,
            publish_request=self._publish_joint_buffer_request,
            update_artifact=self._update_joint_flow_artifact,
            write_artifact=self._write_joint_flow_artifact,
            write_metrics=self._write_metrics_artifacts,
            write_debug=self._write_placement_debug_summary,
        )

    def _compute_refreshed_segment_buffer_gradient(
        self,
        *,
        params,
        model,
        pos,
        event,
        refresh_topology=True,
        frame="post_placement_post_sizing_x_next",
        expected_topology_generation=None,
    ):
        return segment_buffer_gradient.compute_refreshed_segment_buffer_gradient(self._segment_gradient_context(), params=params, model=model, pos=pos, event=event, refresh_topology=refresh_topology, frame=frame, expected_topology_generation=expected_topology_generation)

    def _apply_segment_overflow_milestone_transaction(
        self,
        *,
        params,
        model,
        pos,
        optimizer,
        event,
        sizing_frame,
    ):
        return apply_segment_overflow_milestone_transaction(
            self._segment_joint_context(),
            params=params,
            model=model,
            pos=pos,
            optimizer=optimizer,
            event=event,
            sizing_frame=sizing_frame,
        )

    def _rebuild_and_install_segment_pin2pin(
        self,
        *,
        params,
        placedb,
        model,
        pos,
        optimizer,
        iteration,
        overflow,
        reason,
        force_clear_accumulation,
    ):
        return rebuild_and_install_segment_pin2pin(
            self._segment_joint_context(),
            params=params,
            placedb=placedb,
            model=model,
            pos=pos,
            optimizer=optimizer,
            iteration=iteration,
            overflow=overflow,
            reason=reason,
            force_clear_accumulation=force_clear_accumulation,
        )

    def _run_segment_milestone_sizing_micro_loop(
        self, *, params, model, pos, event
    ):
        return run_segment_milestone_sizing_micro_loop(
            self._segment_joint_context(),
            params=params,
            model=model,
            pos=pos,
            event=event,
        )

    def _apply_segment_rebootstrap_milestone_transaction(
        self, *, params, placedb, model, pos, optimizer, event
    ):
        return apply_segment_rebootstrap_milestone_transaction(
            self._segment_joint_context(),
            params=params,
            placedb=placedb,
            model=model,
            pos=pos,
            optimizer=optimizer,
            event=event,
        )

    def _terminal_static_timing_snapshot(self, *, params, model, pos, frame):
        return terminal_static_timing_snapshot(
            getattr(self, "data_collections", None),
            params=params, model=model, pos=pos, frame=frame,
        )

    def _terminal_arc_cell_state_snapshot(self):
        return terminal_arc_cell_state_snapshot(
            getattr(self, "data_collections", None)
        )

    def _publish_terminal_segment_lane(self, lane, state):
        self.last_buffering_lane = lane
        self.buffer_optimization_state = state
        self.buffer_relaxed_timing_payload = lane.buffer_relaxed_timing_payload


    def _rebuild_terminal_segment_route_b_state(self, *, params, model, pos):
        context = TerminalLaneContext(
            coordinator=self.joint_coordinator,
            data_collections=self.data_collections,
            op_collections=self.op_collections,
            refresh_topology=self._refresh_live_timing_topology,
            publish_lane=self._publish_terminal_segment_lane,
        )
        return rebuild_terminal_segment_route_b_state(
            context, params=params, model=model, pos=pos
        )

    def _apply_terminal_segment_route_b_if_needed(
        self, params, *, iteration, optimizer=None
    ):
        coordinator = getattr(self, "joint_coordinator", None)
        round_budget = terminal_route_b_round_budget(coordinator)
        if round_budget == 0:
            return None
        context = TerminalRouteBContext(
            coordinator=coordinator,
            model=getattr(self, "last_global_place_model", None),
            pos=self.pos[0],
            digest_tensors=self._digest_named_tensors,
            static_snapshot=self._terminal_static_timing_snapshot,
            arc_snapshot=self._terminal_arc_cell_state_snapshot,
            rebuild_state=self._rebuild_terminal_segment_route_b_state,
            compute_gradient=self._compute_refreshed_segment_buffer_gradient,
        )
        return reselect_terminal_segment_route_b(
            context, params, iteration=iteration, optimizer=optimizer,
            round_budget=round_budget,
        )

    def _final_segment_joint_projection_context(self, pos, *, iteration):
        context = segment_joint_terminal.FinalProjectionContext(
            getattr(self, "joint_coordinator", None), self.data_collections,
            self.params, self._refresh_live_timing_topology,
        )
        result = segment_joint_terminal.final_segment_joint_projection_context(
            context, pos, iteration=iteration,
        )
        if context.summary is not None:
            self.last_segment_joint_projection_context_summary = context.summary
        return result

    def _final_joint_physical_context(self, pos, *, iteration):
        from dreamplace.flows.joint_physical_actions import final_physical_context

        return final_physical_context(self, pos, iteration=iteration)

    def _joint_iteration_context(self):
        return joint_iteration.JointIterationContext(
            timing_op=getattr(self.op_collections, "timing_propagation_op", None),
            prepare=self._prepare_segment_direct_joint_iteration,
            rebuild_pin2pin=self._rebuild_and_install_segment_pin2pin,
            capture_frame=self._capture_segment_milestone_sizing_frame,
            rebootstrap=self._apply_segment_rebootstrap_milestone_transaction,
            overflow_transaction=self._apply_segment_overflow_milestone_transaction,
            physical_context=self._final_joint_physical_context,
            update_artifact=self._update_joint_flow_artifact,
            requires_commit=self._joint_buffer_request_requires_outer_commit,
            virtual_window=self._maybe_apply_segment_virtual_window_boundary,
            publish_request=self._publish_joint_buffer_request,
        )

    def _publish_joint_buffer_request(self, request, result, *, reset_trace=False):
        self.last_buffer_commit_request = request
        trace = getattr(self, "joint_buffer_commit_trace", None)
        if reset_trace or trace is None:
            trace = []
            self.joint_buffer_commit_trace = trace
        trace.append(dict(result))

    def _timing_iteration_context(self):
        return timing_iteration.TimingIterationContext(
            enabled=self._diff_tdp_enabled,
            uses_net_weight=self._diff_tdp_uses_gradient_net_weight,
            project_weights=self._project_timing_gradient_to_net_weights,
            refresh_topology=self._refresh_live_timing_topology,
            carrier=self._timing_placement_carrier,
            reset_weights=self._reset_diff_tdp_net_weights_if_needed,
        )

    def _prepare_segment_direct_joint_iteration(
        self,
        *,
        model,
        pos,
        overflow,
        iteration,
    ):
        return prepare_segment_direct_joint_iteration(
            self._segment_joint_context(),
            model=model,
            pos=pos,
            overflow=overflow,
            iteration=iteration,
        )

    def _maybe_apply_segment_virtual_window_boundary(
        self,
        *,
        params,
        model,
        pos,
        optimizer,
        iteration,
    ):
        return apply_segment_virtual_window_boundary(
            self._segment_joint_context(),
            params=params,
            model=model,
            pos=pos,
            optimizer=optimizer,
            iteration=iteration,
        )

    def _record_timing_metrics(self, wns, tns, ws=None, ts=None):
        self.last_timing_metrics = {
            "wns": float(wns),
            "tns": float(tns),
            "ws": None if ws is None else float(ws),
            "ts": None if ts is None else float(ts),
        }

    def _write_placement_debug_summary(
        self,
        params,
        *,
        iteration=None,
        optimizer_steps=None,
        processed_metrics=None,
        last_metric=None,
        status="unknown",
        stop_reason=None,
        entered_legalization=False,
        skipped_legalization_reason=None,
    ):
        context = placement_metrics.DebugSummaryContext(
            getattr(self, "last_timing_metrics", None), getattr(self, "_diff_tdp_gate_summary", None),
            getattr(self, "_timing_grad_balance_state", None),
            getattr(self, "_legacy_net_weight_gate_summary", None),
            getattr(self, "last_legacy_net_weight_artifact", None),
        )
        result = placement_metrics.write_placement_debug_summary(
            context, params, iteration=iteration, optimizer_steps=optimizer_steps,
            processed_metrics=processed_metrics, last_metric=last_metric, status=status,
            stop_reason=stop_reason, entered_legalization=entered_legalization,
            skipped_legalization_reason=skipped_legalization_reason,
        )
        if context.path is not None:
            self.last_placement_debug_summary = context.summary
            self.last_placement_debug_summary_path = context.path
        return result

    def _buffering_flow_enabled(self, params):
        flow_kind = getattr(params, "flow_kind", None)
        flow_value = getattr(flow_kind, "value", flow_kind)
        return str(flow_value) == "buffering"

    def _joint_flow_enabled(self, params):
        flow_kind = getattr(params, "flow_kind", None)
        flow_value = getattr(flow_kind, "value", flow_kind)
        return str(flow_value) == "joint"

    def _buffering_lane_flow_enabled(self, params):
        return self._buffering_flow_enabled(params) or self._joint_flow_enabled(params)

    def _build_buffering_lane(self, params):
        from dreamplace.ops.buffer_insertion.buffering_config import (
            build_buffering_config_from_params,
        )
        from dreamplace.ops.buffer_insertion.buffering_lane import (
            BufferingOptimizationLane,
        )

        config = build_buffering_config_from_params(
            params,
            flow_kind=getattr(params, "flow_kind", None),
        )
        return BufferingOptimizationLane(
            config,
            buffer_optimization_state=getattr(self, "buffer_optimization_state", None),
            buffer_relaxed_timing_payload=getattr(
                self,
                "buffer_relaxed_timing_payload",
                None,
            ),
            placedb=getattr(self, "placedb", None),
        )

    def _write_buffering_inner_loop_artifact(self, params, summary):
        design_name_getter = getattr(params, "design_name", None)
        design_name = (
            design_name_getter()
            if callable(design_name_getter)
            else getattr(params, "base_design_name", "unknown_design")
        )
        result_dir = getattr(params, "result_dir", "results")
        os.makedirs(result_dir, exist_ok=True)
        summary_path = os.path.join(
            result_dir,
            f"{design_name}_buffering_inner_loop_summary.json",
        )
        omitted_fields = (
            "candidate_ids",
            "candidate_net_id",
            "compact_node_to_net_id",
            "compact_node_to_original",
            "net_ids",
            "original_node_to_compact",
            "original_node_to_compact_by_net",
            "synthetic_segment_splits",
        )
        artifact_summary = dict(summary)
        omission_summary = {}
        for field in omitted_fields:
            value = artifact_summary.pop(field, None)
            if value is not None:
                omission_summary[field] = {
                    "entry_count": len(value) if hasattr(value, "__len__") else None,
                }
        state_build_metadata = artifact_summary.get("state_build_metadata")
        if isinstance(state_build_metadata, dict):
            compact_metadata = dict(state_build_metadata)
            for field in omitted_fields:
                compact_metadata.pop(field, None)
            artifact_summary["state_build_metadata"] = compact_metadata
        if omission_summary:
            artifact_summary["artifact_omitted_fields"] = omission_summary
        payload = {
            "artifact_version": 1,
            "metadata": {
                "artifact_scope": "buffering_inner_loop",
                "flow_kind": getattr(params, "flow_kind", None),
                "canonical_entry": "NonLinearPlace._maybe_run_buffering_inner_loop",
                "physical_commit_policy": "final_request_via_placement_engine",
            },
            "summary": artifact_summary,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_buffering_inner_loop_artifact = payload
        self.last_buffering_inner_loop_artifact_path = summary_path
        return {"summary_path": summary_path}

    def _write_joint_flow_artifact(self, params, summary):
        design_name_getter = getattr(params, "design_name", None)
        design_name = (
            design_name_getter()
            if callable(design_name_getter)
            else getattr(params, "base_design_name", "unknown_design")
        )
        result_dir = getattr(params, "result_dir", "results")
        os.makedirs(result_dir, exist_ok=True)
        summary_path = os.path.join(result_dir, f"{design_name}_joint_flow_summary.json")
        sizing_enabled = bool(
            getattr(params, "joint_segment_sizing_enabled", True)
        )
        method = (
            "proximal_alternating_joint"
            if str(getattr(params, "joint_quality_profile", "") or "")
            == "proximal_alternating_v1"
            else ("joint_place_sizing_buffering" if sizing_enabled else "joint_place_buffering")
        )
        payload = {
            "artifact_version": 1,
            "metadata": {
                "artifact_scope": method,
                "flow_kind": getattr(params, "flow_kind", None),
                "canonical_entry": "NonLinearPlace",
                "physical_commit_policy": "fixed_period_requests_via_placement_engine",
                "method": method,
                "sizing_enabled": sizing_enabled,
            },
            "summary": summary,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_joint_flow_artifact = payload
        self.last_joint_flow_artifact_path = summary_path
        return {"summary_path": summary_path}

    def _update_joint_flow_artifact(self, params):
        coordinator = getattr(self, "joint_coordinator", None)
        if coordinator is None:
            return None
        summary = coordinator.summarize()
        trace = list(getattr(self, "joint_buffer_commit_trace", []) or [])
        summary["buffer_commit_trace"] = trace
        summary["buffer_commit_trace_count"] = len(trace)
        model = getattr(self, "last_global_place_model", None)
        proximal = getattr(model, "last_joint_proximal_summary", None)
        if isinstance(proximal, dict) and proximal:
            summary["proximal"] = dict(proximal)
            summary["anchors"] = dict(proximal.get("anchors", {}) or {})
            summary["lambda_scale_probe"] = dict(
                proximal.get("lambda_scale_probe", {}) or {}
            )
            summary["gates"] = dict(proximal.get("gates", {}) or {})
        summary["transactions"] = {
            "buffer_commit_trace_count": len(trace),
            "buffer_commit_enabled": bool(
                getattr(params, "buffering_commit_enabled", 0)
            ),
            "default_no_hard_buffer_commit": not bool(
                getattr(params, "buffering_commit_enabled", 0)
            ),
        }
        return self._write_joint_flow_artifact(params, summary)

    def _finalize_joint_buffer_commit(
        self,
        params,
        *,
        iteration,
        optimizer=None,
        final_pos=None,
    ):
        return joint_commit_control.finalize_joint_buffer_commit(self._joint_commit_context(), params, iteration=iteration, optimizer=optimizer, final_pos=final_pos)

    def _joint_buffer_request_requires_outer_commit(self, request_result):
        return joint_commit_control.joint_buffer_request_requires_outer_commit(request_result, getattr(self, "last_buffer_commit_request", None))

    def _joint_post_commit_control_result(self, commit_result):
        return joint_commit_control.joint_post_commit_control_result(commit_result)

    def record_outer_buffer_commit_result(self, commit_result):
        trace = getattr(self, "joint_buffer_commit_trace", None)
        if trace is None:
            trace = []
            self.joint_buffer_commit_trace = trace
        if trace:
            trace[-1].update(dict(commit_result or {}))
        else:
            trace.append(dict(commit_result or {}))
        control_result = self._joint_post_commit_control_result(commit_result)
        request_summary = (
            dict(commit_result.get("commit_request", {}) or {})
            if isinstance(commit_result, dict)
            else {}
        )
        if request_summary:
            control_result["commit_request"] = request_summary
        self.last_post_commit_control_result = dict(control_result)
        self._update_joint_flow_artifact(self.params)
        summary_payload = getattr(self, "last_joint_flow_artifact", None)
        if isinstance(summary_payload, dict):
            summary = summary_payload.get("summary")
            if isinstance(summary, dict):
                summary["post_commit_control_result"] = dict(control_result)
                summary["post_commit_control_policy"] = {
                    "status": control_result.get("status"),
                    "reason": "placement_engine_executed_buffer_commit",
                    "accepted_action_count": control_result.get(
                        "accepted_action_count"
                    ),
                }
                self._write_joint_flow_artifact(self.params, summary)
        if self._buffering_flow_enabled(self.params):
            buffering_summary = dict(
                getattr(self, "last_buffering_lane_summary", {}) or {}
            )
            if buffering_summary:
                buffering_summary["commit"] = dict(
                    commit_result.get("commit", {}) or {}
                )
                buffering_summary["post_commit_control_result"] = dict(
                    control_result
                )
                if buffering_summary.get("strategy") == "discrete_net_gradient":
                    if buffering_summary.get("mode") == "candidate":
                        from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
                            update_discrete_candidate_scheduler_trace_commit as update_trace_commit,
                        )
                    else:
                        from dreamplace.ops.buffer_insertion.discrete_virtual_buffer_runner import (
                            update_discrete_virtual_scheduler_trace_commit as update_trace_commit,
                        )

                    trace_path = str(buffering_summary.get("trace_path") or "")
                    trace_sha256 = update_trace_commit(
                        trace_path,
                        buffering_summary["commit"],
                    )
                    buffering_summary["trace_sha256"] = trace_sha256
                    inner_loop = buffering_summary.get("inner_loop")
                    if isinstance(inner_loop, dict):
                        inner_loop["trace_sha256"] = trace_sha256
                self.last_buffering_lane_summary = buffering_summary
                self._write_buffering_inner_loop_artifact(
                    self.params,
                    buffering_summary,
                )
        return control_result

    def _safe_stop_after_joint_buffer_request(
        self,
        params,
        *,
        iteration,
        place_stage_metrics,
        processed_metrics,
        commit_result,
    ):
        context = self._joint_commit_context()
        result = joint_commit_control.safe_stop_after_joint_buffer_request(
            context, params, iteration=iteration, place_stage_metrics=place_stage_metrics,
            processed_metrics=processed_metrics, commit_result=commit_result,
        )
        self.last_post_commit_control_result = context.post_commit_result
        return result

    def _compact_buffering_inner_loop_log_summary(self, summary):
        compact = {}
        for key in (
            "status",
            "mode",
            "state_kind",
            "candidate_count",
            "segment_count",
            "affected_net_count",
            "net_count",
            "segment_count_timing_backend_requested",
            "segment_count_timing_backend_used",
            "segment_count_timing_backend_fallback_reason",
            "segment_transfer_backend_requested",
            "segment_transfer_backend",
            "segment_transfer_backend_used",
            "periodic_integer_projection_enabled",
            "periodic_integer_projection_interval",
            "periodic_integer_projection_start_step",
            "periodic_integer_projection_project_bsu",
            "periodic_integer_projection_reset_optimizer_state",
            "periodic_integer_projection_count",
            "strategy",
            "terminal_reason",
            "trace_path",
            "trace_sha256",
        ):
            if key in summary:
                compact[key] = summary[key]
        inner_loop = summary.get("inner_loop") if isinstance(summary, dict) else None
        if isinstance(inner_loop, dict):
            compact["inner_loop"] = {
                "iterations": inner_loop.get("iterations"),
                "projection_count": inner_loop.get("projection_count"),
                "final_projection_wall_ms": inner_loop.get("final_projection_wall_ms"),
            }
            metrics_trace = inner_loop.get("metrics_trace")
            if isinstance(metrics_trace, list) and metrics_trace:
                last_metrics = dict(metrics_trace[-1])
                # Detailed selected-action rows are retained only in the Route B
                # trace artifact, never expanded into the main execution log.
                last_metrics.pop("selected_rows", None)
                compact["last_metrics"] = last_metrics
        commit = summary.get("commit") if isinstance(summary, dict) else None
        if isinstance(commit, dict):
            compact["commit"] = {
                "status": commit.get("status"),
                "accepted_action_count": commit.get("accepted_action_count"),
                "delta_tns": commit.get("delta_tns"),
            }
        return compact

    def _maybe_run_buffering_inner_loop(self, params, model):
        from dreamplace.flows.buffering_coordinator import run_buffering_inner_loop

        return run_buffering_inner_loop(self, params, model)

    def _timing_scalar(self, value):
        return timing_scalar(value)

    def _update_metric_timing(self, metric, wns, tns, ws=None):
        timing_op = getattr(self.op_collections, "timing_propagation_op", None)
        update_metric_timing(metric, wns, tns, ws, timing_op)

    def _metric_snapshot_from_metric(self, metric):
        return metric_snapshot(metric)

    def _apply_metric_snapshot(self, metric, snapshot):
        return apply_metric_snapshot(metric, snapshot)

    def _evaluate_continuous_size_dynamics_candidate_metric(self, model, pos, metric):
        return sizing_dynamics.evaluate_continuous_size_dynamics_candidate_metric(self._sizing_dynamics_context(), model, pos, metric)

    def _combined_timing_loss(self, wns, tns):
        return combined_timing_loss(wns, tns)

    def _finite_timing_scalar(self, value):
        return finite_timing_scalar(value)

    def _early_stop_context(self):
        params = getattr(self, "params", None)
        return continuous_early_stop.EarlyStopContext(
            data_collections=getattr(self, "data_collections", None),
            placedb=getattr(self, "placedb", None),
            params=params,
            oscillation_state=getattr(self, "_quad_oscillation_state", None),
            sizing_parameterization=self._sizing_parameterization,
            continuous_size_parameter=self._continuous_size_parameter,
            discrete_actions_owned=(
                self._segment_direct_joint_enabled()
                or bool(getattr(params, "timing_opt_enabled", False))
            ),
            sync_runtime_cells=self._sync_discrete_gradient_topk_runtime_cells,
            sizing_mode=self._placement_sizing_mode,
            finite_scalar=self._finite_timing_scalar,
            capture_best=self._capture_continuous_early_stop_best_state,
            loss_and_source=self._continuous_early_stop_loss_and_source,
        )

    def _continuous_early_stop_loss_and_source(self, params, metric):
        return continuous_early_stop.continuous_early_stop_loss_and_source(self._early_stop_context(), params, metric)

    def _continuous_early_stop_loss(self, params, metric):
        return continuous_early_stop.continuous_early_stop_loss(self._early_stop_context(), params, metric)

    def _build_continuous_early_stop_state(self, params):
        return continuous_early_stop.build_continuous_early_stop_state(params)

    def _capture_continuous_early_stop_best_state(self, snapshot=None):
        return continuous_early_stop.capture_continuous_early_stop_best_state(self._early_stop_context(), snapshot)

    def _restore_continuous_early_stop_best_state(self, state, reason="continuous_best_state_restore"):
        context = self._early_stop_context()
        result = continuous_early_stop.restore_continuous_early_stop_best_state(context, state, reason)
        if result.get("restored") and "quad_oscillation_state" in state["best_state"]:
            self._quad_oscillation_state = context.oscillation_state
        return result

    def _update_continuous_early_stop_state(
        self,
        state,
        combined_timing_loss,
        iteration,
        loss_source="combined_timing_loss",
    ):
        return continuous_early_stop.update_continuous_early_stop_state(self._early_stop_context(), state, combined_timing_loss, iteration, loss_source)

    def _continuous_early_stop_summary(self, state):
        return continuous_early_stop.continuous_early_stop_summary(state)

    def _augment_metric_observability(self, metric):
        last_timing = getattr(self, "last_timing_metrics", None)
        if last_timing is not None:
            metric.wns = last_timing.get("wns")
            metric.tns = last_timing.get("tns")
            metric.timing_objective = last_timing.get("ws")
        timing_op = getattr(self.op_collections, "timing_propagation_op", None)
        if timing_op is not None:
            metric.slew_violation = getattr(timing_op, "last_total_slew_violation", None)
            metric.cap_violation = getattr(timing_op, "last_total_cap_violation", None)
            metric.leakage = getattr(timing_op, "last_total_leakage", None)

        size_var = self.data_collections.get_size_var()
        if size_var is not None and size_var.numel() > 0:
            metric.size_min = float(size_var.min().item())
            metric.size_mean = float(size_var.mean().item())
            metric.size_max = float(size_var.max().item())

        vt_var = self.data_collections.get_vt_var()
        if vt_var is not None and vt_var.numel() > 0:
            metric.vt_class_proportions = (
                vt_var.mean(dim=0).detach().cpu().numpy().astype(float).tolist()
            )

    def _metric_to_record(self, metric):
        return metric_state.metric_to_record(metric)

    def _enrich_timing_record(self, record):
        return metric_state.enrich_timing_record(record)

    def _timing_stage_delta(self, place_record, legalization_record):
        return metric_state.timing_stage_delta(place_record, legalization_record)

    def _retained_timing_improvement(self, place_record, legalization_record):
        return metric_state.retained_timing_improvement(place_record, legalization_record)

    def _apply_final_check_timing_stage_record(self, timing_stage_summary):
        final_record = getattr(self, "last_final_check_timing_record", None)
        if not isinstance(timing_stage_summary, dict) or not isinstance(final_record, dict):
            return timing_stage_summary

        refreshed = dict(timing_stage_summary)
        legalization_record = dict(refreshed.get("post_legalization") or {})
        legalization_record.update(final_record)
        refreshed["post_legalization"] = self._enrich_timing_record(legalization_record)
        if "legalization" in refreshed:
            refreshed["legalization"] = refreshed["post_legalization"]

        place_record = refreshed.get("pre_projection") or refreshed.get("place") or {}
        refreshed["delta"] = self._timing_stage_delta(
            place_record,
            refreshed["post_legalization"],
        )
        refreshed["retained_timing_improvement"] = self._retained_timing_improvement(
            place_record,
            refreshed["post_legalization"],
        )
        refreshed["post_legalization_full_timing_check_applied"] = True
        return refreshed

    def _flatten_metrics(self, items):
        return flatten_metrics(items)

    def _processed_global_place_metrics(self, all_metrics):
        return metric_state.processed_global_place_metrics(all_metrics)

    @staticmethod
    def _metric_objective_is_nonfinite(metric):
        return metric_objective_is_nonfinite(metric)

    @classmethod
    def _last_metric_from_metrics(cls, metrics):
        return last_metric_from_metrics(metrics)

    def _metrics_stage_name(self, params):
        if getattr(params, "global_place_flag", False):
            return "place"
        if getattr(params, "legalize_flag", False):
            return "legalization"
        return "runtime"

    def _write_metrics_artifacts(self, params, all_metrics, stage_name=None):
        return placement_metrics.write_metrics_artifacts(placement_metrics.MetricsWriterContext(self._flatten_metrics, self._metric_to_record, self._metrics_stage_name), params, all_metrics, stage_name)

    def _density_snapshot(self):
        density_overflow_op = getattr(self.op_collections, "density_overflow_op", None)
        if density_overflow_op is None:
            return {}
        with torch.no_grad():
            overflow, max_density = density_overflow_op(self.pos[0])
        return {
            "overflow": float(torch.as_tensor(overflow).reshape(-1)[0].item()),
            "max_density": float(torch.as_tensor(max_density).reshape(-1)[0].item()),
        }

    def _write_timing_stage_artifact(self, params, place_metrics, legalization_metrics):
        mode = self._placement_sizing_mode(params)
        if mode not in ("size_only", "joint"):
            return None

        place_flat = [metric for metric in self._flatten_metrics(place_metrics) if metric is not None]
        legalization_flat = [metric for metric in self._flatten_metrics(legalization_metrics) if metric is not None]
        if not place_flat or not legalization_flat:
            return None

        place_record = self._enrich_timing_record(self._metric_to_record(place_flat[-1]))
        legalization_record = self._enrich_timing_record(self._metric_to_record(legalization_flat[-1]))
        projection_summary_path = os.path.join(params.result_dir, f"{params.design_name()}_projection_summary.json")
        projection_summary = self._load_json_artifact_if_exists(projection_summary_path)
        post_projection_record, stage_compare_overrides = self._resolve_post_projection_stage_data(
            params,
            place_record,
            legalization_record,
            projection_summary,
        )
        delta = self._timing_stage_delta(place_record, legalization_record)
        retained_timing_improvement = self._retained_timing_improvement(
            place_record,
            legalization_record,
        )

        summary = write_timing_stage_summary(
            params,
            mode=mode,
            place_record=place_record,
            post_projection_record=post_projection_record,
            legalization_record=legalization_record,
            delta=delta,
            retained_timing_improvement=retained_timing_improvement,
            continuous_early_stop=self._continuous_early_stop_summary(
                getattr(self, "last_continuous_early_stop_state", None)
            ),
            stage_compare_overrides=stage_compare_overrides,
        )
        self._write_stage_compare_latest_artifact(params, summary)
        self.last_timing_stage_summary = summary
        return summary

    def _write_sta_only_artifact(self, params, metric):
        record = self._enrich_timing_record(self._metric_to_record(metric))
        summary = {
            "artifact_version": 1,
            "metadata": {
                "artifact_scope": "sta_only",
                "flow_kind": "sta",
                "placement_sizing_mode": self._placement_sizing_mode(params),
                "strict_stage_order": ["sta"],
            },
            "sta": record,
            "pre_projection": record,
            "post_legalization": record,
            "place": record,
        }
        summary_path = os.path.join(
            params.result_dir,
            f"{params.design_name()}_timing_stage_summary.json",
        )
        os.makedirs(params.result_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_timing_stage_summary = summary
        return summary

    def _load_json_artifact_if_exists(self, path):
        return load_json_artifact_if_exists(path)

    def _load_csv_rows_if_exists(self, path):
        return load_csv_rows_if_exists(path)

    def _artifact_float(self, value):
        return artifact_float(value)

    def _artifact_bool(self, value):
        return artifact_bool(value)

    def _artifact_value(self, value):
        return artifact_value(value)

    def _normalize_pin_name(self, value):
        return timing_artifacts.normalize_pin_name(value)

    def _ps_to_ns(self, value):
        return timing_artifacts.ps_to_ns(value)

    def _current_pin_limit_context(self):
        return current_pin_limit_context(getattr(self, "data_collections", None))

    def _projected_vt_distribution(self, vt_tensor, legal_inst_ids, projected_vt):
        return projected_vt_distribution(vt_tensor, legal_inst_ids, projected_vt)

    def _prepare_design_pin_values(self, values, expected_len):
        return prepare_design_pin_values(values, expected_len)

    def _aggregate_violation_per_instance(
        self, rise_values, fall_values, pin_mask, limits, inst_ids
    ):
        return aggregate_violation_per_instance(
            self._projection_scoring_context(),
            rise_values, fall_values, pin_mask, limits, inst_ids,
        )

    def _restore_projection_snapshot_timing_metrics(self, metrics):
        self.last_timing_metrics = metrics

    def _compute_post_projection_timing_snapshot(
        self, params, place_record, legalization_record, projection_summary,
    ):
        if self._openroad_backed_aimp_db_enabled(params):
            return None
        context = ProjectionTimingSnapshotContext(
            data_collections=getattr(self, "data_collections", None),
            op_collections=getattr(self, "op_collections", None),
            model=getattr(self, "last_global_place_model", None),
            pos=self.pos[0].data if hasattr(self, "pos") else None,
            projection_result=getattr(self, "last_projection_result", None),
            capture_timing_metrics=lambda: copy.deepcopy(
                getattr(self, "last_timing_metrics", None)
            ),
            restore_timing_metrics=self._restore_projection_snapshot_timing_metrics,
            enrich_record=self._enrich_timing_record,
        )
        return compute_post_projection_timing_snapshot(
            context, params, place_record, legalization_record, projection_summary
        )

    def _run_final_check_log_after_writeback(self, model):
        if model is None or not hasattr(model, "timing_obj"):
            return None

        params = getattr(model, "params", None)
        if self._openroad_backed_aimp_db_enabled(params):
            logging.info("skip iEDA final check_log for external native timing backend")
            return None

        op_collections = getattr(self, "op_collections", None)
        if (
            getattr(op_collections, "steiner_topo_op", None) is not None
            and getattr(op_collections, "pin_pos_op", None) is not None
            and hasattr(self, "pos")
        ):
            pin_pos_for_topology = (
                op_collections.pin_pos_op(self.pos[0].data)
                .detach()
                .cpu()
                .contiguous()
            )
            (
                self.data_collections.net_flat_topo_sort,
                self.data_collections.net_flat_topo_sort_start,
                self.data_collections.pin_fa,
                self.data_collections.flat_pin_to,
                self.data_collections.flat_pin_to_start,
                self.data_collections.flat_pin_from,
            ) = op_collections.steiner_topo_op.rebuild_tree(
                pin_pos_for_topology
            )

        pos = self.pos[0].data if hasattr(self, "pos") else None
        timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
        previous_fast_loop = getattr(params, "production_fast_loop", None)
        previous_timing_fast_loop = getattr(timing_op, "production_fast_loop", None)
        if params is not None and previous_fast_loop is not None:
            params.production_fast_loop = False
        if timing_op is not None and previous_timing_fast_loop is not None:
            timing_op.production_fast_loop = False
        try:
            wns, tns, ws, ts = model.timing_obj(pos)
            self._record_timing_metrics(wns, tns, ws, ts)
            record = {
                "wns": self._timing_scalar(wns),
                "tns": self._timing_scalar(tns),
                "timing_objective": self._timing_scalar(ws),
                "slew_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_slew_violation", None),
                "cap_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_cap_violation", None),
                "leakage": None
                if timing_op is None
                else getattr(timing_op, "last_total_leakage", None),
                "runtime_seconds": None
                if timing_op is None
                else getattr(timing_op, "last_runtime_seconds", None),
                "stage_note": "post_runtime_projection_full_timing_check",
            }
            self.last_final_check_timing_record = self._enrich_timing_record(record)
            if hasattr(model, "check_log"):
                model.check_log(wns, tns, ws, ts)
        finally:
            if params is not None and previous_fast_loop is not None:
                params.production_fast_loop = previous_fast_loop
            if timing_op is not None and previous_timing_fast_loop is not None:
                timing_op.production_fast_loop = previous_timing_fast_loop
        return wns, tns, ws, ts

    def _finalize_after_runtime_projection_refresh(
        self,
        params,
        placedb,
        cur_pos,
        model,
        projection_runtime_summary,
    ):
        op_collections = getattr(self, "op_collections", None)
        if projection_runtime_summary and projection_runtime_summary.get("num_changed_instances", 0) > 0:
            logging.info(
                "re-applying placedb after runtime projection refresh to commit projected cell masters"
            )
            placedb.apply(
                params,
                cur_pos[0: placedb.num_movable_nodes],
                cur_pos[placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes],
            )

        if (
            getattr(params, "with_sta", False)
            and not self._openroad_backed_aimp_db_enabled(params)
            and getattr(op_collections, "steiner_topo_op", None) is not None
            and model is not None
            and hasattr(model, "timing_obj")
        ):
            self._run_final_check_log_after_writeback(model)

    def _refresh_post_projection_timing_stage_artifact(self, params, timing_stage_summary=None):
        if self._openroad_backed_aimp_db_enabled(params):
            return timing_stage_summary
        if timing_stage_summary is None:
            summary_path = os.path.join(params.result_dir, f"{params.design_name()}_timing_stage_summary.json")
            timing_stage_summary = self._load_json_artifact_if_exists(summary_path)
        if not isinstance(timing_stage_summary, dict) or not timing_stage_summary:
            return timing_stage_summary

        projection_summary_path = os.path.join(
            params.result_dir, f"{params.design_name()}_projection_summary.json"
        )
        projection_summary = self._load_json_artifact_if_exists(projection_summary_path)
        snapshot = self._compute_post_projection_timing_snapshot(
            params=params,
            place_record=timing_stage_summary.get("pre_projection") or timing_stage_summary.get("place") or {},
            legalization_record=timing_stage_summary.get("post_legalization") or {},
            projection_summary=projection_summary,
        )
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("record"), dict):
            return timing_stage_summary

        refreshed_summary = write_post_projection_timing_summary(
            params, timing_stage_summary, snapshot,
            enrich_record=self._enrich_timing_record,
            add_endpoint_override=self._add_stage_endpoint_override,
        )
        self.last_timing_stage_summary = refreshed_summary
        return refreshed_summary

    def _resolve_post_projection_stage_data(
        self,
        params,
        place_record,
        legalization_record,
        projection_summary,
    ):
        snapshot_fn = getattr(self, "_compute_post_projection_timing_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn(
                params=params,
                place_record=place_record,
                legalization_record=legalization_record,
                projection_summary=projection_summary,
            )
            if isinstance(snapshot, dict) and isinstance(snapshot.get("record"), dict):
                post_projection_record = self._enrich_timing_record(dict(snapshot["record"]))
                stage_override = {}
                self._add_stage_endpoint_override(
                    stage_override,
                    snapshot.get("endpoint_payload"),
                    snapshot.get("top_endpoints"),
                )
                if stage_override:
                    return post_projection_record, {"post_projection": stage_override}
                return post_projection_record, {}

        post_projection_record = dict(place_record)
        projected_leakage = projection_summary.get("total_projected_leakage")
        if projected_leakage is not None:
            post_projection_record["leakage"] = float(projected_leakage)
        post_projection_record["stage_note"] = (
            "projection_discrete_state_not_retimed; timing mirrors pre_projection "
            "until a dedicated post_projection timing snapshot is available"
        )
        return post_projection_record, {}

    def _timing_compare_snapshot_suffixes(self):
        return [
            "_endpoint_timing_compare.csv",
            "_endpoint_timing_compare_summary.json",
            "_timing_pin_all_report.csv",
            "_cell_arc_delay_report.csv",
            "_timing_first_level_pins_report.csv",
            "_clk2q_level0_arc_report.csv",
            "_clk2q_level0_arc_summary.json",
            "_clk2q_clock_pin_debug.json",
            "_clk2q_clock_pin_debug_summary.json",
        ]

    def _capture_default_timing_compare_artifacts(self, params):
        captured = {}
        for suffix in self._timing_compare_snapshot_suffixes():
            path = os.path.join(params.result_dir, f"{params.design_name()}{suffix}")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    captured[suffix] = f.read()
            else:
                captured[suffix] = None
        return captured

    def _restore_default_timing_compare_artifacts(self, params, captured_artifacts):
        if not isinstance(captured_artifacts, dict):
            return
        for suffix in self._timing_compare_snapshot_suffixes():
            path = os.path.join(params.result_dir, f"{params.design_name()}{suffix}")
            payload = captured_artifacts.get(suffix)
            if payload is None:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                continue
            with open(path, "wb") as f:
                f.write(payload)

    def _build_endpoint_stage_payload(self, params, stage_name):
        return timing_artifacts.build_endpoint_stage_payload(timing_artifacts.EndpointArtifactContext(self._endpoint_stage_artifact_stale_reason), params, stage_name)

    def _endpoint_payload_can_be_primary(self, payload):
        if not isinstance(payload, dict):
            return False
        if payload.get("source") != "backend_endpoint_debug":
            return True
        return (
            payload.get("sta_is_golden") is True
            or payload.get("sta_reference_role") == "golden_opensta"
        )

    def _add_stage_endpoint_override(self, stage_override, endpoint_payload, top_endpoints=None):
        if not isinstance(endpoint_payload, dict):
            return
        is_primary = self._endpoint_payload_can_be_primary(endpoint_payload)
        payload_key = "primary" if is_primary else "diagnostic"
        endpoint_key = "top_endpoints" if is_primary else "diagnostic_top_endpoints"
        stage_override[payload_key] = dict(endpoint_payload)
        if isinstance(top_endpoints, list):
            rows = [dict(row) for row in top_endpoints if isinstance(row, dict)]
            if rows:
                stage_override[endpoint_key] = rows

    def _endpoint_stage_artifact_stale_reason(self, summary_path, csv_path):
        run_started_at = getattr(self, "_stage_compare_run_started_at", None)
        if run_started_at is None:
            return None
        existing_paths = [
            path for path in (summary_path, csv_path)
            if path and os.path.exists(path)
        ]
        if not existing_paths:
            return None
        if any(os.path.getmtime(path) < float(run_started_at) for path in existing_paths):
            return "older_than_current_run"
        return None

    def _build_secondary_stage_payload(self, timing_stage_summary, stage_name):
        return timing_artifacts.build_secondary_stage_payload(timing_stage_summary, stage_name)

    def _endpoint_pin_name_to_id_map(self):
        return projection_context.endpoint_pin_name_to_id_map(projection_context.CriticalConeContext(getattr(self, "data_collections", None), getattr(self, "placedb", None)))

    def _reverse_reachable_pin_ids(self, seed_pin_ids, reverse_offsets, reverse_edges, num_pins):
        return projection_context.reverse_reachable_pin_ids(seed_pin_ids, reverse_offsets, reverse_edges, num_pins)

    def _collect_critical_endpoint_cone_nodes(self, top_critical_endpoints):
        return projection_context.collect_critical_endpoint_cone_nodes(projection_context.CriticalConeContext(getattr(self, "data_collections", None), getattr(self, "placedb", None)), top_critical_endpoints)

    def _build_changed_cell_summary(self, params, top_critical_endpoints=None):
        csv_path = os.path.join(params.result_dir, f"{params.design_name()}_projection.csv")
        summary_path = os.path.join(params.result_dir, f"{params.design_name()}_projection_summary.json")
        rows = self._load_csv_rows_if_exists(csv_path)
        cone_node_ids, stage_hits = self._collect_critical_endpoint_cone_nodes(
            top_critical_endpoints
        )
        return build_changed_cell_summary(
            params, rows=rows,
            load_summary=lambda: self._load_json_artifact_if_exists(summary_path),
            cone_node_ids=cone_node_ids, stage_hits=stage_hits,
        )

    def _build_stage_compare_stage_record(self, timing_stage_summary, params, stage_name):
        endpoint_payload, top_endpoints = self._build_endpoint_stage_payload(params, stage_name)
        return build_stage_compare_stage_record(
            timing_stage_summary, stage_name,
            endpoint_payload=endpoint_payload, top_endpoints=top_endpoints,
            build_secondary=lambda: self._build_secondary_stage_payload(
                timing_stage_summary, stage_name
            ),
            can_be_primary=self._endpoint_payload_can_be_primary,
        )

    def _write_stage_compare_latest_artifact(self, params, timing_stage_summary):
        context = StageCompareWriterContext(
            build_stage_record=self._build_stage_compare_stage_record,
            build_changed_cells=self._build_changed_cell_summary,
        )
        latest, latest_path = write_stage_compare_latest_artifact(
            context, params, timing_stage_summary,
            mode=self._placement_sizing_mode(params),
        )
        self.last_stage_compare_latest_summary = latest
        self.last_stage_compare_latest_path = latest_path
        return latest

    def _refresh_stage_compare_latest_artifact(self, params, timing_stage_summary=None):
        return refresh_stage_compare_latest_artifact(
            params, timing_stage_summary,
            write_artifact=self._write_stage_compare_latest_artifact,
        )

    def _snapshot_timing_compare_artifacts(self, params, stage_label):
        stage_prefix = f"{params.design_name()}_{stage_label}"
        for suffix in self._timing_compare_snapshot_suffixes():
            src = os.path.join(params.result_dir, f"{params.design_name()}{suffix}")
            if not os.path.exists(src):
                continue
            dst = os.path.join(params.result_dir, f"{stage_prefix}{suffix}")
            shutil.copyfile(src, dst)

    def _compute_local_timing_scores(self):
        return compute_local_timing_scores(self._projection_scoring_context())

    def _projection_objective_profile(self, params):
        return projection_objective_profile(params)

    def _projection_profile_term_weights(self, params):
        return projection_profile_term_weights(params)

    def _endpoint_focus_penalty(self, params, inst_ids, local_timing_scores, downsize_penalty):
        return endpoint_focus_penalty(
            params, inst_ids, local_timing_scores, downsize_penalty,
            build_endpoint_payload=self._build_endpoint_stage_payload,
            collect_cone_nodes=self._collect_critical_endpoint_cone_nodes,
        )

    def _projection_scoring_context(self):
        return ProjectionScoringContext(
            data_collections=self.data_collections,
            op_collections=getattr(self, "op_collections", None),
            pos=self.pos[0],
            build_endpoint_payload=self._build_endpoint_stage_payload,
            collect_cone_nodes=self._collect_critical_endpoint_cone_nodes,
        )

    def _build_projection_context(
        self, projection_op, inst_ids, size_var, vt_var,
        params=None, stage_timing_summary=None, candidates=None,
    ):
        return build_projection_context(
            self._projection_scoring_context(), projection_op, inst_ids,
            size_var, vt_var, params=params,
            stage_timing_summary=stage_timing_summary, candidates=candidates,
        )

    def _projection_artifact_policy(self, params, stage_timing_summary=None):
        return projection_artifact_policy(params, stage_timing_summary)

    def _write_projection_summary_artifact(
        self, params, projection_result, metadata=None,
    ):
        return write_projection_summary_artifact(
            self.op_collections.gate_projection_op, params, projection_result,
            metadata=metadata,
            projection_frame=getattr(self, "last_projection_frame", None),
        )

    def _build_projection_frame(self, params, projection_result):
        context = ProjectionFrameContext(
            data_collections=self.data_collections,
            op_collections=self.op_collections,
            changed_inst_mask=self._projection_changed_inst_mask,
            record_bytes=self._record_real_size_transition_bytes,
            digest_tensors=self._digest_named_tensors,
        )
        return build_projection_frame(context, params, projection_result)

    def _configure_real_size_projection_backend(self, params, projection_op):
        backend = self._real_size_transition_backend(params)
        if backend == "native":
            raise RuntimeError(
                "real_size_transition_backend='native' is not implemented; "
                "select reference or vectorized"
            )
        provider_type = (
            MainIdCandidateProvider
            if backend == "reference"
            else VectorizedMainIdCandidateProvider
        )
        if type(projection_op.candidate_provider) is not provider_type:
            projection_op.candidate_provider = provider_type()
        return backend

    def _write_projection_artifacts(self, params, stage_timing_summary=None):
        size_var = self.data_collections.get_size_var()
        vt_var = self.data_collections.get_vt_var()
        inst_is_sizeable = getattr(self.data_collections, "inst_is_sizeable", None)
        if size_var is None or vt_var is None or inst_is_sizeable is None:
            return None

        inst_ids = torch.nonzero(inst_is_sizeable, as_tuple=False).flatten()
        if inst_ids.numel() == 0:
            logging.info("skip gate projection artifacts because there are no sizeable instances")
            return None

        projection_op = getattr(self.op_collections, "gate_projection_op", None)
        if projection_op is None:
            projection_op = self.data_collections.build_gate_projection_op()
            self.op_collections.gate_projection_op = projection_op
        if not isinstance(projection_op, GateProjectionOp):
            raise TypeError("gate_projection_op must be a GateProjectionOp instance")
        backend = self._configure_real_size_projection_backend(params, projection_op)

        candidate_started_at = time.perf_counter()
        candidates = projection_op.candidate_provider.enumerate(
            self.data_collections,
            inst_ids,
        )
        self._record_real_size_transition_stage(
            "candidate_enumerate_ms",
            candidate_started_at,
        )

        context_started_at = time.perf_counter()
        context = self._build_projection_context(
            projection_op,
            inst_ids,
            size_var,
            vt_var,
            params=params,
            stage_timing_summary=stage_timing_summary,
            candidates=candidates,
        )
        self._record_real_size_transition_stage(
            "projection_context_ms",
            context_started_at,
        )

        projection_started_at = time.perf_counter()
        with torch.no_grad():
            projection_result = projection_op(
                inst_ids=inst_ids,
                size_var=size_var,
                vt_var=vt_var,
                context=context,
                candidates=candidates,
                profile=getattr(self, "_active_real_size_transition_profile", None),
            )
        self._record_real_size_transition_stage(
            "projection_compute_ms",
            projection_started_at,
        )

        self.last_projection_result = projection_result
        frame_started_at = time.perf_counter()
        self.last_projection_frame = self._build_projection_frame(
            params,
            projection_result,
        )
        self._record_real_size_transition_stage(
            "projection_frame_build_ms",
            frame_started_at,
        )
        self.last_projection_artifact_paths = {}
        metadata = {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": self._placement_sizing_mode(params),
            "timing_stage_summary": stage_timing_summary or {},
            "density_reference": self._density_snapshot(),
            "projection_objective_profile": self._projection_objective_profile(params),
            "projection_term_weights": context.term_weights,
            "projection_backend": backend,
        }
        artifact_policy = self._projection_artifact_policy(
            params,
            stage_timing_summary=stage_timing_summary,
        )
        artifact_started_at = time.perf_counter()
        if artifact_policy == "summary_only":
            artifact_paths = self._write_projection_summary_artifact(
                params,
                projection_result,
                metadata=metadata,
            )
        else:
            artifact_paths = projection_op.write_artifacts(
                projection_result,
                result_dir=params.result_dir,
                design_name=params.design_name(),
                metadata=metadata,
                profile=getattr(self, "_active_real_size_transition_profile", None),
            )
        self._record_real_size_transition_artifacts(artifact_paths)
        self._record_real_size_transition_stage(
            "artifact_write_wrapper_ms",
            artifact_started_at,
        )
        self.last_projection_artifact_paths = artifact_paths

        if projection_result.validation.is_valid:
            logging.info(
                "wrote gate projection artifacts to %s",
                ", ".join(artifact_paths.values()),
            )
        else:
            logging.warning(
                "gate projection artifact generated with validation issues: %s",
                "; ".join(projection_result.validation.issues),
            )
        return artifact_paths

    def _projection_changed_inst_mask(self, projection_result):
        return projection_changed_inst_mask(projection_result)

    def _write_projection_runtime_refresh_summary(self, params, summary):
        summary_path = os.path.join(
            params.result_dir,
            f"{params.design_name()}_projection_runtime_refresh_summary.json",
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._record_real_size_transition_artifacts({"runtime_summary": summary_path})
        self.last_projection_runtime_refresh_summary = summary
        self.last_projection_runtime_refresh_summary_path = summary_path
        return summary_path

    def _write_projection_pin_offset_runtime_summary(
        self,
        params,
        runtime_pin_offset_consistency_applied,
        num_runtime_updated_pins,
        num_runtime_updated_instances,
    ):
        summary = {
            "runtime_pin_offset_consistency_applied": bool(
                runtime_pin_offset_consistency_applied
            ),
            "num_runtime_updated_pins": int(num_runtime_updated_pins),
            "num_runtime_updated_instances": int(num_runtime_updated_instances),
            "metadata": {
                "artifact_scope": "post_projection_runtime",
                "placement_sizing_mode": self._placement_sizing_mode(params),
            },
        }
        runtime_summary_path = os.path.join(
            params.result_dir,
            f"{params.design_name()}_projection_pin_offsets_runtime_summary.json",
        )
        with open(runtime_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._record_real_size_transition_artifacts(
            {"pin_runtime_summary": runtime_summary_path}
        )
        self.last_projection_pin_offset_runtime_summary = summary
        self.last_projection_pin_offset_runtime_summary_path = runtime_summary_path
        return summary

    def _apply_projected_pin_offset_consistency(
        self,
        params,
        placedb,
        projection_result=None,
        changed_inst_ids=None,
        projection_frame=None,
    ):
        context = projected_pin_refresh.ProjectedPinContext(
            self.data_collections,
            getattr(self.op_collections, "gate_projection_op", None),
            getattr(self, "last_projection_result", None),
            self._write_projection_pin_offset_runtime_summary,
            self._record_real_size_transition_stage,
            self._defer_real_size_transition_placedb_sync,
        )
        return projected_pin_refresh.apply_projected_pin_offset_consistency(
            context, params, placedb, projection_result, changed_inst_ids, projection_frame,
        )

    def _apply_projected_runtime_refresh(self, params, placedb):
        from dreamplace.flows.projection_runtime_refresh import (
            apply_projected_runtime_refresh,
        )

        return apply_projected_runtime_refresh(self, params, placedb)

    def _write_surrogate_cache_artifact(self, params, model=None):
        mode = self._placement_sizing_mode(params)
        if mode not in ("size_only", "joint"):
            return None
        cell_modeling_op = None
        if model is not None:
            cell_modeling_op = getattr(getattr(model, "op_collections", None), "cell_modeling_op", None)
        if cell_modeling_op is None:
            return None

        summary = cell_modeling_op.cache_summary()
        summary["metadata"] = {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": mode,
        }
        summary_path = os.path.join(params.result_dir, f"{params.design_name()}_surrogate_cache_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.last_surrogate_cache_summary = summary
        return summary_path

    def _validate_sizing_mode(self, params):
        mode = self._placement_sizing_mode(params)
        sizing_parameterization = self._sizing_parameterization(params)
        has_sizing_params = any(True for _ in self.size_params.parameters()) or any(
            True for _ in self.vt_params.parameters()
        )
        if mode in ("size_only", "joint") and not has_sizing_params:
            raise ValueError(
                f"placement_sizing_mode={mode} requires size/vt parameters, but sizing tensors were not created"
            )
        if sizing_parameterization == "real_size":
            if mode not in ("size_only", "joint") and not bool(
                getattr(params, "timing_opt_enabled", False)
            ):
                raise ValueError(
                    "sizing_parameterization='real_size' is supported only by "
                    "size_only, joint and timing optimization flows"
                )
            if self._segment_direct_joint_enabled() or str(
                getattr(params, "joint_quality_profile", "") or ""
            ) == "segment_count_direct_joint_v1":
                raise ValueError(
                    "real_size is incompatible with segment direct joint milestone sizing"
                )
            if getattr(self.data_collections, "real_size", None) is None:
                raise ValueError("real_size parameterization has no real_size owner")
            if getattr(self.data_collections, "size_logits", None) is not None:
                raise ValueError(
                    "real_size parameterization must not register size_logits as a second owner"
                )
            execution_mode = str(
                getattr(params, "real_size_execution_mode", "continuous_only")
            ).strip().lower()
            if execution_mode not in ("continuous_only", "warmup_to_discrete"):
                raise ValueError(
                    "unsupported real_size_execution_mode: " + execution_mode
                )
            if execution_mode == "warmup_to_discrete":
                warmup_steps = int(
                    getattr(params, "real_size_warmup_steps", 0) or 0
                )
                if warmup_steps <= 0:
                    raise ValueError(
                        "real_size_warmup_steps must be positive for warmup_to_discrete"
                    )
            stages = getattr(params, "global_place_stages", None) or ()
            unsupported_optimizers = [
                str(stage.get("optimizer", "")).lower()
                for stage in stages
                if str(stage.get("optimizer", "")).lower() != "adam"
            ]
            if unsupported_optimizers and mode in ("size_only", "joint"):
                raise ValueError(
                    "real_size parameterization requires Adam; found optimizers="
                    + ",".join(unsupported_optimizers)
                )

    def _validate_backend_capabilities(self, params, placedb):
        caps = dict(getattr(placedb, "backend_caps", {}) or {})
        backend = caps.get("backend", "unknown")
        timing_active = bool(
            getattr(params, "with_sta", 0)
            or getattr(params, "diff_timing_driven_placement", 0)
            or getattr(params, "differentiable_timing_obj", 0)
            or getattr(params, "timing_eval_flag", 0)
            or str(getattr(params, "flow_kind", "placement"))
            in {"sta", "sizing", "buffering", "joint"}
        )
        if timing_active and not bool(caps.get("has_sta", True)):
            raise RuntimeError(
                "timing-enabled placement requires BackendCaps.has_sta=True; "
                f"backend={backend} is geometry-only"
            )
        if timing_active and not bool(caps.get("supports_sta_pydb", True)):
            raise RuntimeError(
                "timing-enabled placement requires BackendCaps.supports_sta_pydb=True; "
                f"backend={backend} does not export timing data"
            )
        has_diff_sizing_metadata = bool(caps.get("has_diff_sizing_metadata", caps.get("has_diff_optimizer_metadata", True)))
        has_buffer_optimizer_metadata = bool(caps.get("has_buffer_optimizer_metadata", caps.get("has_diff_optimizer_metadata", True)))
        sizing_mode = self._placement_sizing_mode(params)
        if sizing_mode in ("size_only", "joint") and not has_diff_sizing_metadata:
            raise RuntimeError(
                f"placement_sizing_mode={sizing_mode} requires "
                "BackendCaps.has_diff_sizing_metadata=True; "
                f"backend={backend} reports has_diff_sizing_metadata=False"
            )
        if self._buffering_flow_enabled(params) and not has_buffer_optimizer_metadata:
            raise RuntimeError(
                "flow_kind=buffering requires "
                "BackendCaps.has_buffer_optimizer_metadata=True; "
                f"backend={backend} reports has_buffer_optimizer_metadata=False"
            )

    def _write_backend_capability_artifact(self, params, placedb):
        result_dir = getattr(params, "result_dir", None)
        if not result_dir:
            return None

        design_name_attr = getattr(params, "design_name", None)
        design_name = design_name_attr() if callable(design_name_attr) else design_name_attr
        if not design_name:
            design_name = "design"

        caps = dict(getattr(placedb, "backend_caps", {}) or {})
        payload = {
            "design_name": str(design_name),
            "place_io_engine": getattr(params, "place_io_engine", "ieda"),
            "placement_sizing_mode": self._placement_sizing_mode(params),
            "flow_kind": str(getattr(params, "flow_kind", "")),
            "buffering_mode": str(getattr(params, "buffering_mode", "")),
            "backend_caps": caps,
            "capability_interpretation": {
                "diff_sizing_metadata_available": bool(
                    caps.get("has_diff_sizing_metadata", caps.get("has_diff_optimizer_metadata", True))
                ),
                "buffer_optimizer_metadata_available": bool(
                    caps.get("has_buffer_optimizer_metadata", caps.get("has_diff_optimizer_metadata", True))
                ),
                "sta_pydb_available": bool(caps.get("supports_sta_pydb", True)),
                "sta_reference_role": caps.get("sta_reference_role", "unspecified"),
                "sta_is_golden": caps.get("sta_reference_role") == "golden_opensta",
                "parasitics_initialization": caps.get("parasitics_initialization", "unspecified"),
                "sta_state_status": caps.get("sta_state_status", "unspecified"),
                "buffer_commit_available": bool(caps.get("supports_buffer_commit", False)),
                "committed_refresh_available": bool(
                    caps.get("supports_committed_refresh", False)
                ),
            },
        }
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, f"{design_name}_backend_caps.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        self.last_backend_capability_artifact_path = output_path
        self.last_backend_capability_artifact = payload
        return output_path

    def _optimizer_parameters(self, params):
        mode = self._placement_sizing_mode(params)
        buffer_params = list(getattr(self, "buffer_params", torch.nn.ParameterList()).parameters())
        sizing_params = list(self.size_params.parameters()) + list(self.vt_params.parameters())
        if buffer_params and mode != "joint":
            sizing_params = sizing_params + buffer_params
        if mode == "place_only":
            groups = [{"params": list(self.pos.parameters()), "group_name": "placement"}]
            if buffer_params:
                groups.append({"params": buffer_params, "group_name": "buffering"})
            return groups
        if mode == "size_only":
            return [
                {
                    "params": sizing_params,
                    "group_name": "sizing",
                }
            ]
        if mode == "joint":
            coordinator = getattr(self, "joint_coordinator", None)
            if coordinator is not None:
                return coordinator.collect_param_groups(
                    placement_params=list(self.pos.parameters()),
                    sizing_params=sizing_params,
                )
            groups = [{"params": list(self.pos.parameters()), "group_name": "placement"}]
            if sizing_params:
                groups.append({"params": sizing_params, "group_name": "sizing"})
            if buffer_params:
                groups.append({"params": buffer_params, "group_name": "buffering"})
            return groups
        raise ValueError(f"Unsupported placement_sizing_mode: {mode}")

    def _placement_pos_to_numpy(self, pos):
        if pos is None:
            runtime_pos = getattr(self, "pos", None)
            if runtime_pos is not None and len(runtime_pos):
                pos = runtime_pos[0]
        if pos is None:
            raise RuntimeError(
                "live placement position is required for runtimedb restart"
            )
        if hasattr(pos, "detach"):
            pos = pos.detach()
        if hasattr(pos, "cpu"):
            pos = pos.cpu()
        if hasattr(pos, "numpy"):
            pos = pos.numpy()
        return np.asarray(pos).reshape(-1).copy()

    def _current_placement_counts_for_continuation_seed(self, placedb):
        counts = {}
        for attr_name in (
            "num_nodes",
            "num_movable_nodes",
            "num_physical_nodes",
            "num_filler_nodes",
        ):
            try:
                counts[attr_name] = int(getattr(placedb, attr_name))
            except (AttributeError, TypeError, ValueError):
                pass
        if "num_nodes" not in counts and "num_physical_nodes" in counts:
            counts["num_nodes"] = counts["num_physical_nodes"] + counts.get(
                "num_filler_nodes",
                0,
            )
        try:
            num_movable_nodes = counts["num_movable_nodes"]
            node_names = getattr(placedb, "node_names")
            counts["movable_node_names"] = [
                self._decode_continuation_node_name(name)
                for name in list(node_names)[:num_movable_nodes]
            ]
        except (AttributeError, KeyError, TypeError):
            pass
        return counts

    @staticmethod
    def _decode_continuation_node_name(name):
        if isinstance(name, bytes):
            return name.decode()
        if hasattr(name, "decode"):
            return name.decode()
        return str(name)

    @staticmethod
    def _value_has_nonfinite(value):
        return value_has_nonfinite(value)

    @staticmethod
    def _handoff_restart_policy(params):
        return resolve_restart_policy(params)

    @staticmethod
    def _clone_restart_value(value):
        return clone_restart_value(value)

    @classmethod
    def _capture_warm_schedule_state(cls, model):
        return capture_warm_schedule_state(model)

    @staticmethod
    def _shape_tuple(value):
        shape = getattr(value, "shape", None)
        return () if shape is None else tuple(shape)

    @classmethod
    def _apply_warm_schedule_state(cls, model, state):
        return apply_warm_schedule_state(model, state)

    @staticmethod
    def _restart_scalar(value):
        return restart_scalar(value)

    @classmethod
    def _restart_probe_payload(cls, policy, phase, handoff_seq, metric):
        return restart_probe_payload(policy, phase, handoff_seq, metric)

    def _prepare_restart_state(
        self,
        params,
        placedb,
        handoff_result,
        pos=None,
        handoff_seq=None,
        model=None,
    ):
        restart_policy = self._handoff_restart_policy(params)
        placedb.pending_continuation_seed = self._build_continuation_seed(
            placedb,
            handoff_result,
            pos,
            handoff_seq,
        )
        placedb.pending_warm_schedule_state = None
        warm_schedule_state_captured = False
        if restart_policy == "warm_schedule" and model is not None:
            warm_schedule_state = self._capture_warm_schedule_state(model)
            if warm_schedule_state:
                placedb.pending_warm_schedule_state = warm_schedule_state
                warm_schedule_state_captured = True
        return {
            "restart_policy": restart_policy,
            "warm_schedule_state_captured": warm_schedule_state_captured,
        }

    @classmethod
    def _metric_has_nonfinite_value(cls, metric):
        return metric_has_nonfinite_value(metric)

    @classmethod
    def _should_skip_post_global_placement(cls, stop_reason, last_metric):
        return should_skip_post_global_placement(stop_reason, last_metric)

    @staticmethod
    def _is_gradient_nan_assertion(exc):
        return (
            isinstance(exc, AssertionError)
            and len(exc.args) == 1
            and exc.args[0] == "Gradient contains NaN"
        )

    @staticmethod
    def _mark_metric_gradient_nan_failure(metric):
        reference_value = getattr(metric, "objective", None)
        if reference_value is None:
            reference_value = getattr(metric, "hpwl", None)
        if hasattr(reference_value, "detach"):
            metric.objective = torch.full_like(reference_value.detach(), float("nan"))
        else:
            metric.objective = torch.tensor(float("nan"))

    @staticmethod
    def _refresh_timing_metrics(model, pos):
        if model is None or pos is None:
            return False
        timing_obj = getattr(model, "timing_obj", None)
        if timing_obj is None:
            return False
        timing_values = timing_obj(pos)
        if isinstance(timing_values, (tuple, list)) and len(timing_values) >= 4:
            for field_name, value in zip(
                ("wns", "tns", "ws", "ts"),
                timing_values[:4],
            ):
                setattr(model, field_name, value)
        return True

    @staticmethod
    def _copy_timing_metrics_to_eval_metric(model, metric):
        if model is None or metric is None:
            return
        snapshot_builder = getattr(model, "timing_metric_snapshot", None)
        if snapshot_builder is None:
            return
        snapshot = snapshot_builder()
        if not isinstance(snapshot, dict):
            return
        for field_name, value in snapshot.items():
            if value is None:
                continue
            setattr(metric, field_name, value)

    def _build_continuation_seed(self, placedb, handoff_result, pos, handoff_seq):
        return build_continuation_seed(
            placedb, handoff_result, pos, handoff_seq,
            pos_to_numpy=self._placement_pos_to_numpy,
            decode_node_name=self._decode_continuation_node_name,
        )

    def _restart_after_topology_sync(
        self, params, placedb, handoff_result, pos=None, handoff_seq=None, model=None,
    ):
        return restart_after_topology_sync(
            params, placedb, handoff_result, pos=pos,
            handoff_seq=handoff_seq, model=model,
            prepare_restart_state=self._prepare_restart_state,
            create_placer=lambda params, placedb: self.__class__(
                params, placedb, self.timer
            ),
        )

    def _call_restart_after_topology_sync(
        self, params, placedb, handoff_result, pos=None, handoff_seq=None, model=None,
    ):
        return call_restart_after_topology_sync(
            self._restart_after_topology_sync, params, placedb, handoff_result,
            pos=pos, handoff_seq=handoff_seq, model=model,
        )

    def _build_session_mutation_result(self, handoff_result):
        return build_session_mutation_result(handoff_result)

    def _build_handoff_command_failure_payload(self, handoff_result):
        return build_handoff_command_failure_payload(handoff_result)

    def _execute_openroad_handoff_transaction(
        self, params, placedb, pos, run_event, trigger_decision, handoff_event, model=None,
    ):
        context = HandoffTransactionContext(
            placement_counts=self._current_placement_counts_for_continuation_seed,
            mutation_payload=self._build_session_mutation_result,
            failure_payload=self._build_handoff_command_failure_payload,
            restart_after_sync=self._call_restart_after_topology_sync,
            restart_policy=self._handoff_restart_policy,
        )
        return execute_openroad_handoff_transaction(
            context, params, placedb, pos, run_event, trigger_decision,
            handoff_event, model=model,
        )

    def run_sta_only(self, params, placedb):
        self._validate_sizing_mode(params)
        self._write_backend_capability_artifact(params, placedb)
        self._validate_backend_capabilities(params, placedb)
        os.makedirs(params.result_dir, exist_ok=True)

        stage_params = (getattr(params, "global_place_stages", None) or [{}])[0]
        model = PlaceObj.PlaceObj(
            0.0,
            params,
            placedb,
            self.data_collections,
            self.op_collections,
            stage_params,
        ).to(self.data_collections.pos[0].device)
        self.last_global_place_model = model
        if not hasattr(model, "timing_obj"):
            raise RuntimeError("STA-only flow requires PlaceObj.timing_obj")
        if getattr(self.op_collections, "timing_propagation_op", None) is None:
            raise RuntimeError("STA-only flow requires with_sta timing propagation op")

        started_at = time.time()
        pos = self.data_collections.pos[0]
        with torch.no_grad():
            if hasattr(self, "_refresh_live_timing_topology"):
                self._refresh_live_timing_topology(pos)
            wns, tns, ws, ts = model.timing_obj(pos)
        self._record_timing_metrics(wns, tns, ws, ts)

        metric = EvalMetrics.EvalMetrics(iteration=0, detailed_step=None)
        self._update_metric_timing(metric, wns, tns, ws)
        metric.eval_time = time.time() - started_at
        metric.combined_timing_loss = self._combined_timing_loss(metric.wns, metric.tns)
        summary = self._write_sta_only_artifact(params, metric)
        logging.info("STA-only timing summary: %s", summary.get("sta"))
        return float("inf"), float("inf"), {
            "objective": [],
            "hpwl": [],
            "overflow": [],
            "density": [],
            "timing_stage_summary": summary,
        }

    def __call__(self, params, placedb):
        """
        @brief Top API to solve placement.
        @param params parameters
        @param placedb placement database
        """
        self._validate_sizing_mode(params)
        self._write_backend_capability_artifact(params, placedb)
        self._validate_backend_capabilities(params, placedb)
        self._reset_size_debug_trace(params)
        self._reset_discrete_gradient_topk_trace(params)
        self._reset_full_step_profile(params)
        self._reset_timing_obj_profile(params)
        self._reset_diff_tdp_gate_summary(params)
        self._real_size_transition_done = False
        self._reset_legacy_net_weight_gate_summary(params)
        if str(getattr(params, "joint_quality_profile", "") or "") == (
            "segment_count_direct_joint_v1"
        ):
            self.joint_buffer_commit_trace = []
            self.last_buffer_commit_request = None
            self.last_post_commit_control_result = None
        # List with optimizers that don't rebuild the model, i.e. they call the function "step()" instead of "step(closure)".
        optimizers_no_closure = ["nesterov", "sgd", "sgd_momentum", "sgd_nesterov", "adam", "adamax", "nadam", "adamw", "adadelta", "rmsprop", "aggmo", "qhadam", "yogi", "radam", "adabelief", "adabound", "adafactor", "diffgrad", "novograd"]
        
        # List with the available optimizers. ATTENTION: Do not add "nesterov" in this list.
        optimizers_list = ["sgd", "sgd_momentum", "sgd_nesterov", "adam", "adamax", "nadam", "adamw", "adadelta", "rmsprop", "aggmo", "qhadam", "yogi", "radam", "adabelief", "adabound", "adafactor", "diffgrad", "novograd", "cg_fr", "cg_prp", "cg_hs", "cg_cd", "cg_ls", "cg_dy", "cg_hz", "cg_hs-dy"]

        iteration = 0
        all_metrics = []
        original_stop_overflow = params.stop_overflow
        if getattr(params, "macro_only", False) and getattr(
            params, "macro_place_flag", False
        ):
            params.stop_overflow = min(
                0.1,
                placedb.total_movable_cell_area * 0.2
                / placedb.total_movable_node_area,
            )
            logging.info(
                "macro-only stop_overflow = %.6E (cell_area=%.6E, movable_area=%.6E)",
                params.stop_overflow,
                placedb.total_movable_cell_area,
                placedb.total_movable_node_area,
            )
        place_stage_metrics = []
        legalization_stage_metrics = []
        continuous_early_stop_state = self._build_continuous_early_stop_state(params)
        self.last_continuous_early_stop_state = continuous_early_stop_state
        continuous_size_dynamics_state = self._build_continuous_size_dynamics_state(params)
        self.last_continuous_size_dynamics_state = continuous_size_dynamics_state
        model = None
        pending_joint_topology_commit_result = None
        self._routability_controller = RoutabilityController(params)
        self._routability_model = None
        routability_controller = self._routability_controller
        cooptimization = None
        if bool(getattr(params, "timing_opt_enabled", False)):
            cooptimization = getattr(self, "inflation_s5b1", None)
            if cooptimization is None:
                cooptimization = InflationS5B1(
                    params, placedb, self.data_collections, self.op_collections,
                    self._refresh_live_timing_topology,
                )
                self.inflation_s5b1 = cooptimization
        inflation_legalization.validate_params(params)
        validate_ggr_l_shape_topology_params(params)

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

                if params.gpu and self.data_collections.pos[0].is_cuda:
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
                # Construct the model and the optimizer
                def construct_model():
                    write_crash_stage_marker(
                        "place_objective_construction",
                        "start",
                        cur_stage=cur_stage,
                        global_place_stage=cur_stage,
                        with_sta=getattr(params, "with_sta", None),
                        selected_device=str(self.data_collections.pos[0].device),
                    )
                    model = PlaceObj.PlaceObj(
                        density_weight,
                        params,
                        placedb,
                        self.data_collections,
                        self.op_collections,
                        global_place_params,
                    ).to(self.data_collections.pos[0].device)
                    timing_grad_balance_state = getattr(
                        self,
                        "_timing_grad_balance_state",
                        None,
                    )
                    if timing_grad_balance_state is not None:
                        model.apply_timing_grad_balance_state(
                            timing_grad_balance_state
                        )
                    if self._should_use_live_timing_loss(params):
                        model.use_timing_obj = True
                    write_crash_stage_marker(
                        "place_objective_construction",
                        "done",
                        cur_stage=cur_stage,
                        global_place_stage=cur_stage,
                        with_sta=getattr(params, "with_sta", None),
                        selected_device=str(self.data_collections.pos[0].device),
                    )
                    if hasattr(model, "compile"):
                        model.compile()
                    return model
                
                density_weight = 0.0
                model = construct_model()
                self.last_global_place_model = model
                warm_schedule_state = getattr(
                    placedb,
                    "pending_warm_schedule_state",
                    None,
                )
                if warm_schedule_state is not None:
                    warm_schedule_report = self._apply_warm_schedule_state(
                        model,
                        warm_schedule_state,
                    )
                    logging.info(
                        "handoff warm_schedule_state=%s",
                        warm_schedule_report,
                    )
                    placedb.pending_warm_schedule_state = None
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
                handoff_controller = placedb.create_openroad_handoff_controller(params)
                if self._joint_flow_enabled(params):
                    from dreamplace.flows.joint_coordinator import JointCoordinator

                    previous_lane = getattr(self, "last_buffering_lane", None)
                    reuse_segment_lane = (
                        str(getattr(params, "joint_quality_profile", "") or "")
                        == "segment_count_direct_joint_v1"
                        and previous_lane is not None
                        and getattr(previous_lane, "data_collections", None)
                        is getattr(model, "data_collections", None)
                        and getattr(previous_lane, "_current_state", lambda: None)()
                        is not None
                    )
                    buffering_lane = (
                        previous_lane
                        if reuse_segment_lane
                        else self._build_buffering_lane(params)
                    )
                    self.joint_coordinator = JointCoordinator(
                        params,
                        buffering_lane=buffering_lane,
                    )
                    self.joint_coordinator.prepare_lanes(
                        model=model,
                        data_collections=getattr(model, "data_collections", None),
                    )
                    if (
                        buffering_lane is not None
                        and not self.joint_coordinator.is_segment_direct_joint
                    ):
                        buffering_lane.register_model_parameters(self)
                        self.last_buffering_lane = buffering_lane
                        self.last_buffering_lane_summary = buffering_lane.summarize()
                    elif buffering_lane is not None:
                        self.last_buffering_lane = buffering_lane
                buffering_inner_loop_summary = self._maybe_run_buffering_inner_loop(
                    params,
                    model,
                )
                if buffering_inner_loop_summary is not None:
                    processed_metrics = {
                        "objective": [],
                        "hpwl": [],
                        "overflow": [],
                        "density": [],
                        "buffering_inner_loop_summary": buffering_inner_loop_summary,
                    }
                    control_result = getattr(
                        self,
                        "last_post_commit_control_result",
                        None,
                    )
                    if isinstance(control_result, dict):
                        processed_metrics["post_commit_control_result"] = dict(
                            control_result
                        )
                    write_crash_stage_marker(
                        "nonlinearplace_call_return",
                        "done",
                        reason="buffering_inner_loop_completed",
                        iteration=iteration,
                    )
                    return float("inf"), float("inf"), processed_metrics
                if params.macro_place_flag and macro_placed:
                    movable_macro_mask = self.data_collections.movable_macro_mask
                    model.fix_nodes_mask = movable_macro_mask.new_zeros(
                        placedb.num_nodes)
                    model.fix_nodes_mask[placedb.num_movable_nodes:placedb.num_physical_nodes] = 1
                    model.fix_nodes_mask[:placedb.num_movable_nodes] = movable_macro_mask[:placedb.num_movable_nodes]
                    # params.use_bb = False
                    # pdb.set_trace()

                optimizer_name = global_place_params["optimizer"]
                trainable_params = self._optimizer_parameters(params)
                if not trainable_params:
                    raise ValueError(
                        f"No trainable parameters selected for placement_sizing_mode={self._placement_sizing_mode(params)}"
                    )
                self._validate_optimizer_parameter_groups(
                    optimizer_name,
                    trainable_params,
                )

                # determine optimizer
                try:                
                    write_crash_stage_marker(
                        "nonlinearplace_optimizer_setup",
                        "start",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                    optimizer_name = global_place_params["optimizer"]

                    # Determine optimizer
                    # 1. The official pytorch package
                    if optimizer_name.lower() == "sgd":
                        optimizer = torch.optim.SGD(trainable_params, lr=0)
                    elif optimizer_name.lower() == "sgd_momentum":
                        optimizer = torch.optim.SGD(trainable_params, lr=0, momentum=0.9, nesterov=False)
                    elif optimizer_name.lower() == "sgd_nesterov":
                        optimizer = torch.optim.SGD(trainable_params, lr=0, momentum=0.9, nesterov=True)
                    elif optimizer_name.lower() == "adadelta":
                        optimizer = torch.optim.Adadelta(trainable_params, lr=0)
                    elif optimizer_name.lower() == "rmsprop":
                        optimizer = torch.optim.RMSprop(trainable_params, lr=0)
                    elif optimizer_name.lower() == "adam":
                        optimizer = torch.optim.Adam(trainable_params, lr=0)
                    elif optimizer_name.lower() == "adamax":
                        optimizer = torch.optim.Adamax(trainable_params, lr=0)
                    elif optimizer_name.lower() == "adamw":
                        optimizer = torch.optim.AdamW(trainable_params, lr=0)
                    elif optimizer_name.lower() == "nadam":
                        optimizer = torch.optim.NAdam(trainable_params, lr=0)
                    elif optimizer_name.lower() == "nesterov":
                        optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(
                            trainable_params,
                            lr=0,
                            obj_and_grad_fn=model.obj_and_grad_fn,
                            constraint_fn=self.op_collections.move_boundary_op,
                            use_bb=params.use_bb,
                            max_step_size=getattr(
                                params,
                                "placement_initial_learning_rate_max",
                                None,
                            ),
                        )
                    
                    # 2. The torch_optimizer package
                    elif optimizer_name.lower() == "aggmo":
                        optimizer = torch_optimizer.AggMo(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "qhadam":
                        optimizer = torch_optimizer.QHAdam(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "yogi":
                        optimizer = torch_optimizer.Yogi(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "radam":
                        optimizer = torch_optimizer.RAdam(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "adabelief":
                        optimizer = torch_optimizer.AdaBelief(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "adabound":
                        optimizer = torch_optimizer.AdaBound(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "adafactor":
                        optimizer = torch_optimizer.Adafactor(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "diffgrad":
                        optimizer = torch_optimizer.DiffGrad(trainable_params, lr=params.learning_rate_value)
                    elif optimizer_name.lower() == "novograd":
                        optimizer = torch_optimizer.NovoGrad(trainable_params, lr=params.learning_rate_value)           
                
                    # 3. The ncg_optimizer package
                    elif optimizer_name.lower() == "cg_fr":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'FR', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.5, lr = 0.2, max_ls = 25,)
                    elif optimizer_name.lower() == "cg_prp":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'PRP', line_search = 'Armijo', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5,)
                    elif optimizer_name.lower() == "cg_hs":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'HS', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.4, lr = 0.2, max_ls = 25,)
                    elif optimizer_name.lower() == "cg_cd":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'CD', line_search = 'Armijo', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5,)
                    elif optimizer_name.lower() == "cg_ls":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'LS', line_search = 'Armijo', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5,)
                    elif optimizer_name.lower() == "cg_dy":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'DY', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.9, lr = 0.2, max_ls = 25,)
                    elif optimizer_name.lower() == "cg_hz":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'HZ', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.9, lr = 0.2, max_ls = 25,)
                    elif optimizer_name.lower() == "cg_hs-dy":
                        optimizer = ncg_optimizer.BASIC(trainable_params, method = 'HS-DY', line_search = 'Armijo', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5,)

                    #4. Else
                    else:
                        assert 0, "unknown optimizer %s" % (optimizer_name)
                    write_crash_stage_marker(
                        "nonlinearplace_optimizer_setup",
                        "done",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                except Exception as e:
                    write_crash_stage_marker(
                        "nonlinearplace_optimizer_setup",
                        "failed",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                        error=repr(e),
                    )
                    logging.error(f"Error initializing optimizer %s: %s. Optimizer {optimizer_name} is not supported by torch (version {torch.__version__}) or torch_optimizer or ncg_optimizer. Please check the version of torch, torch_optimizer, and ncg_optimizer.")
                    raise e

                # Set the run_closure flag
                if optimizer_name.lower() in optimizers_no_closure:
                  run_closure = 0
                  logging.info("Optimizer '%s' does not rebuild the model. "
                               "Every training epoch calls the function 'step()'." % (optimizer_name))
                else:
                  run_closure = 1
                  logging.info("Optimizer '%s' rebuilds the model. "
                               "Every training epoch calls the function 'step(closure)'." % (optimizer_name))



                logging.info("use %s optimizer" % (optimizer_name))
                write_crash_stage_marker(
                    "nonlinearplace_model_train",
                    "start",
                    cur_stage=cur_stage,
                    optimizer=optimizer_name,
                )
                model.train()
                write_crash_stage_marker(
                    "nonlinearplace_model_train",
                    "done",
                    cur_stage=cur_stage,
                    optimizer=optimizer_name,
                )
                # defining evaluation ops
                eval_ops = {
                    # "wirelength" : self.op_collections.wirelength_op,
                    # "density" : self.op_collections.density_op,
                    # "objective" : model.obj_fn,
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op,
                }
                if cooptimization is not None:
                    eval_ops["overflow"] = model.combined_density_overflow
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
                    use_live_timing_loss = self._should_use_live_timing_loss(params)
                    if use_live_timing_loss:
                        write_crash_stage_marker(
                            "nonlinearplace_lr_live_timing_topology_refresh",
                            "start",
                            cur_stage=cur_stage,
                            iteration=iteration,
                            optimizer=optimizer_name,
                            use_live_timing_loss=use_live_timing_loss,
                        )
                        try:
                            with torch.no_grad():
                                self._refresh_live_timing_topology(pos)
                        except Exception as e:
                            write_crash_stage_marker(
                                "nonlinearplace_lr_live_timing_topology_refresh",
                                "failed",
                                cur_stage=cur_stage,
                                iteration=iteration,
                                optimizer=optimizer_name,
                                use_live_timing_loss=use_live_timing_loss,
                                error=repr(e),
                            )
                            raise
                        write_crash_stage_marker(
                            "nonlinearplace_lr_live_timing_topology_refresh",
                            "done",
                            cur_stage=cur_stage,
                            iteration=iteration,
                            optimizer=optimizer_name,
                            use_live_timing_loss=use_live_timing_loss,
                        )
                    else:
                        write_crash_stage_marker(
                            "nonlinearplace_lr_live_timing_topology_refresh",
                            "skipped",
                            cur_stage=cur_stage,
                            iteration=iteration,
                            optimizer=optimizer_name,
                            use_live_timing_loss=use_live_timing_loss,
                        )
                    write_crash_stage_marker(
                        "nonlinearplace_lr_estimate_initial_learning_rate",
                        "start",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                        use_live_timing_loss=use_live_timing_loss,
                    )
                    try:
                        placement_lr = model.estimate_initial_learning_rate(
                            pos, global_place_params["learning_rate"]
                        )
                    except Exception as e:
                        write_crash_stage_marker(
                            "nonlinearplace_lr_estimate_initial_learning_rate",
                            "failed",
                            cur_stage=cur_stage,
                            iteration=iteration,
                            optimizer=optimizer_name,
                            use_live_timing_loss=use_live_timing_loss,
                            error=repr(e),
                        )
                        raise
                    estimated_placement_lr = float(placement_lr.detach().item())
                    placement_lr_max = getattr(
                        params,
                        "placement_initial_learning_rate_max",
                        None,
                    )
                    if placement_lr_max is not None:
                        placement_lr = placement_lr.clamp(
                            max=float(placement_lr_max)
                        )
                    logging.info(
                        "initial placement learning rate estimated=%.9g effective=%.9g max=%s",
                        estimated_placement_lr,
                        float(placement_lr.detach().item()),
                        placement_lr_max,
                    )
                    write_crash_stage_marker(
                        "nonlinearplace_lr_estimate_initial_learning_rate",
                        "done",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                        use_live_timing_loss=use_live_timing_loss,
                    )
                    sizing_lr = self._effective_sizing_learning_rate(
                        params,
                        global_place_params["learning_rate"],
                    )
                    buffering_lr = getattr(
                        params,
                        "buffering_continuous_lr",
                        sizing_lr,
                    )
                    write_crash_stage_marker(
                        "nonlinearplace_lr_param_group_apply",
                        "start",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                        use_live_timing_loss=use_live_timing_loss,
                    )
                    try:
                        for param_group in optimizer.param_groups:
                            group_name = param_group.get("group_name")
                            if group_name == "placement":
                                param_group["lr"] = placement_lr.data
                            elif group_name == "sizing":
                                param_group["lr"] = sizing_lr
                            elif group_name == "buffering":
                                param_group["lr"] = buffering_lr
                            else:
                                param_group["lr"] = placement_lr.data
                        coordinator = getattr(self, "joint_coordinator", None)
                        if coordinator is not None:
                            coordinator.apply_activation_to_optimizer(
                                optimizer,
                                overflow=None,
                                sizing_lr=sizing_lr,
                                buffering_lr=buffering_lr,
                            )
                    except Exception as e:
                        write_crash_stage_marker(
                            "nonlinearplace_lr_param_group_apply",
                            "failed",
                            cur_stage=cur_stage,
                            iteration=iteration,
                            optimizer=optimizer_name,
                            use_live_timing_loss=use_live_timing_loss,
                            error=repr(e),
                        )
                        raise
                    write_crash_stage_marker(
                        "nonlinearplace_lr_param_group_apply",
                        "done",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                        use_live_timing_loss=use_live_timing_loss,
                    )

                if iteration == 0 or (params.macro_place_flag and cur_stage == 1):
                    write_crash_stage_marker(
                        "nonlinearplace_learning_rate_initialization",
                        "start",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                    )
                    if iteration == 0 and self._should_apply_gp_noise(params):
                        logging.info("add %g%% noise" %
                                     (params.gp_noise_ratio * 100))
                        model.op_collections.noise_op(
                            model.data_collections.pos[0], params.gp_noise_ratio)
                    initialize_learning_rate(model.data_collections.pos[0])
                    frozen_generation = self._activate_size_only_frozen_topology(
                        params,
                        model.data_collections.pos[0],
                    )
                    if frozen_generation is not None:
                        logging.info(
                            "froze size-only timing topology generation=%d",
                            frozen_generation,
                        )
                    write_crash_stage_marker(
                        "nonlinearplace_learning_rate_initialization",
                        "done",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                    )
                else:
                    write_crash_stage_marker(
                        "nonlinearplace_learning_rate_initialization",
                        "skipped",
                        cur_stage=cur_stage,
                        iteration=iteration,
                        optimizer=optimizer_name,
                    )
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if params.gpu and self.data_collections.pos[0].is_cuda:
                    write_crash_stage_marker(
                        "nonlinearplace_cuda_synchronize_before_opt",
                        "start",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                    torch.cuda.synchronize()
                    write_crash_stage_marker(
                        "nonlinearplace_cuda_synchronize_before_opt",
                        "done",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                else:
                    write_crash_stage_marker(
                        "nonlinearplace_cuda_synchronize_before_opt",
                        "skipped",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                logging.info("%s initialization takes %g seconds" %
                             (optimizer_name, (time.time() - tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in optimizers_list:
                    write_crash_stage_marker(
                        "nonlinearplace_preloop_obj_and_grad_fn",
                        "start",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                    if self._should_use_live_timing_loss(params):
                        with torch.no_grad():
                            self._refresh_live_timing_topology(model.data_collections.pos[0])
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                    write_crash_stage_marker(
                        "nonlinearplace_preloop_obj_and_grad_fn",
                        "done",
                        cur_stage=cur_stage,
                        optimizer=optimizer_name,
                    )
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def segment_rebootstrap_has_pending_work():
                    coordinator = getattr(self, "joint_coordinator", None)
                    return bool(
                        coordinator is not None
                        and coordinator.uses_pin2pin_rebootstrap_schedule
                        and (
                            coordinator.segment_milestones.has_pending_work
                            or coordinator.pin2pin_pair_generations.pending
                        )
                    )

                def Lgamma_stop_criterion(Lgamma_step, metrics, stop_mask=None):
                    with torch.no_grad():
                        if segment_rebootstrap_has_pending_work():
                            return False
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
                        if segment_rebootstrap_has_pending_work():
                            return False
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
                    nonlocal pending_joint_topology_commit_result
                    t0 = time.time()
                    routability_controller.before_iteration(
                        model=model,
                        iteration=iteration,
                        position=model.data_collections.pos[0],
                    )
                    profile_enabled = self._should_write_full_step_profile(params, iteration)
                    profile_step_start = time.perf_counter()
                    timing_backward_component_profile = {}
                    piecewise_size_backend_profile = {}
                    cell_model_forward_profile = {}
                    profile_stage_ms = {
                        "move_boundary_ms": 0.0,
                        "initialize_density_weight_ms": 0.0,
                        "zero_grad_ms": 0.0,
                        "eval_metrics_evaluate_ms": 0.0,
                        "refresh_live_timing_topology_ms": 0.0,
                        "obj_and_grad_fn_ms": 0.0,
                        "obj_and_grad_zero_grad_ms": 0.0,
                        "obj_and_grad_obj_fn_ms": 0.0,
                        "obj_and_grad_component_grad_stats_ms": 0.0,
                        "objective_backward_ms": None,
                        "obj_and_grad_size_only_grad_zero_ms": 0.0,
                        "obj_and_grad_precondition_ms": 0.0,
                        "obj_and_grad_total_ms": 0.0,
                        "record_timing_metrics_ms": 0.0,
                        "size_debug_grad_stats_ms": 0.0,
                        "shared_topology_update_ms": 0.0,
                        "plot_ms": 0.0,
                        "optimizer_step_ms": 0.0,
                        "optimizer_parameter_update_ms": 0.0,
                        "real_size_projection_ms": 0.0,
                        "discrete_gradient_topk_update_ms": 0.0,
                        "size_debug_trace_ms": 0.0,
                        "net_weighting_update_ms": 0.0,
                        "metric_logging_ms": 0.0,
                        "best_pos_update_ms": 0.0,
                    }

                    def profile_now():
                        return time.perf_counter()

                    def profile_add(stage_name, start_time):
                        if profile_enabled:
                            profile_stage_ms[stage_name] = (
                                profile_stage_ms.get(stage_name, 0.0)
                                + (profile_now() - start_time) * 1000.0
                            )

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(
                        iteration, (Lgamma_step,
                                    Llambda_density_weight_step, Lsub_step)
                    )
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    self._augment_metric_observability(cur_metric)
                    pos = model.data_collections.pos[0]
                    if hasattr(model, "set_l_shape_outer_iteration"):
                        model.set_l_shape_outer_iteration(iteration)

                    # move any out-of-bound cell back to placement region
                    _profile_t = profile_now()
                    self.op_collections.move_boundary_op(pos)
                    profile_add("move_boundary_ms", _profile_t)

                    # handle multiple density weights for multi-electric field
                    _profile_t = profile_now()
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
                    profile_add("initialize_density_weight_ms", _profile_t)

                    # For backward compatibility
                    # PyTorch 1.7 introduced zero_grad(set_to_none=False)
                    # PyTorch 2.0 changed set_to_none=True
                    _profile_t = profile_now()
                    if "set_to_none" in inspect.signature(optimizer.zero_grad).parameters:
                        optimizer.zero_grad(set_to_none=False)
                    else:
                        optimizer.zero_grad()
                    profile_add("zero_grad_ms", _profile_t)

                    # t1 = time.time()
                    _profile_t = profile_now()
                    cur_metric.evaluate(placedb, eval_ops,
                                        pos, model.data_collections)
                    model.overflow = cur_metric.overflow.data.clone()
                    coordinator = getattr(self, "joint_coordinator", None)
                    if coordinator is not None:
                        sizing_lr = self._effective_sizing_learning_rate(
                            params,
                            global_place_params["learning_rate"],
                        )
                        buffering_lr = getattr(
                            params,
                            "buffering_continuous_lr",
                            sizing_lr,
                        )
                        coordinator.apply_activation_to_optimizer(
                            optimizer,
                            overflow=float(model.overflow[-1]),
                            sizing_lr=sizing_lr,
                            buffering_lr=buffering_lr,
                        )
                    profile_add("eval_metrics_evaluate_ms", _profile_t)
                    # logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    # t2 = time.time()

                    # Freeze timing activation once per outer placement step. All
                    # objective evaluations inside this optimizer step share it.
                    diff_tdp_gate_status = self._freeze_diff_tdp_step_status(
                        params,
                        iteration,
                        model.overflow,
                    )
                    if self._diff_tdp_controls_direct_timing_objective(params):
                        model.use_timing_obj = bool(
                            diff_tdp_gate_status["timing_objective_active"]
                        )

                    joint_step = joint_iteration.JointStepState()
                    coordinator = getattr(self, "joint_coordinator", None)
                    joint_iteration.prepare_step(
                        self._joint_iteration_context(), coordinator, params, placedb,
                        model, pos, optimizer, optimizer_name, iteration, cur_metric,
                        joint_step, profile_now, profile_add,
                    )

                    _profile_t = profile_now()
                    placement_topology_iteration.update_before_objective(
                        self._timing_iteration_context(),
                        l_shape_iteration.LShapeIterationContext(
                            self.data_collections, self.op_collections,
                        ),
                        params, placedb, model, pos, iteration, cur_metric,
                        l_shape_policy, diff_tdp_gate_status, joint_step,
                        self._publish_live_timing_topology,
                    )
                    profile_add("shared_topology_update_ms", _profile_t)

                    if cooptimization is not None:
                        cooptimization.before_step(
                            model, pos, optimizer, cur_metric, eval_ops,
                            iteration=iteration, stage=cur_stage,
                        )

                    self._initialize_timing_grad_balance_if_needed(
                        params,
                        model,
                        pos,
                        iteration=iteration,
                        overflow=model.overflow,
                    )

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    size_debug_before = self._capture_size_debug_snapshot()
                    size_debug_grad_stats = {}
                    if optimizer_name.lower() in optimizers_list:
                        if (
                            self._should_refresh_live_timing_before_objective(params)
                            and not joint_step.topology_prepared
                        ):
                            _profile_t = profile_now()
                            if not self._should_skip_live_timing_topology_refresh(params):
                                with torch.no_grad():
                                    self._refresh_live_timing_topology(pos)
                            profile_add("refresh_live_timing_topology_ms", _profile_t)
                        _profile_t = profile_now()
                        obj, grad = model.obj_and_grad_fn(pos)
                        profile_add("obj_and_grad_fn_ms", _profile_t)
                        if (
                            coordinator is not None
                            and coordinator.uses_pin2pin_rebootstrap_schedule
                            and coordinator.pin2pin_pair_generations.active
                        ):
                            joint_step.consumed_generation = (
                                coordinator.pin2pin_pair_generations.current_generation_id
                            )
                            coordinator.record_pin2pin_objective_evaluation(
                                iteration=iteration,
                                generation_id=joint_step.consumed_generation,
                            )
                            joint_step.objective_recorded = True
                            setattr(
                                cur_metric,
                                "pin2pin_pair_generation",
                                joint_step.consumed_generation,
                            )
                        obj_grad_profile = getattr(model, "debug_obj_and_grad_profile", {}) or {}
                        for profile_key in (
                            "obj_and_grad_zero_grad_ms",
                            "obj_and_grad_obj_fn_ms",
                            "obj_and_grad_component_grad_stats_ms",
                            "objective_backward_ms",
                            "objective_backward_mode",
                            "objective_backward_grad_param_count",
                            "obj_and_grad_size_only_grad_zero_ms",
                            "obj_and_grad_precondition_ms",
                            "obj_and_grad_total_ms",
                        ):
                            if profile_key in obj_grad_profile:
                                profile_stage_ms[profile_key] = obj_grad_profile[profile_key]
                        timing_backward_component_profile = obj_grad_profile.get(
                            "timing_backward_component_profile"
                        )
                        piecewise_size_backend_profile = obj_grad_profile.get(
                            "piecewise_size_backend_profile"
                        )
                        cell_model_forward_profile = obj_grad_profile.get(
                            "cell_model_forward_profile"
                        )
                        cur_metric.objective = obj.data.clone()
                        if model.use_timing_obj and hasattr(model, "wns") and hasattr(model, "tns"):
                            _profile_t = profile_now()
                            self._record_timing_metrics(
                                model.wns,
                                model.tns,
                                getattr(model, "ws", None),
                                getattr(model, "ts", None),
                            )
                            self._update_metric_timing(
                                cur_metric,
                                model.wns,
                                model.tns,
                                getattr(model, "ws", None),
                            )
                            cur_metric.combined_timing_loss = self._combined_timing_loss(
                                cur_metric.wns,
                                cur_metric.tns,
                            )
                            profile_add("record_timing_metrics_ms", _profile_t)
                        _profile_t = profile_now()
                        size_debug_grad_stats = self._size_debug_grad_stats()
                        profile_add("size_debug_grad_stats_ms", _profile_t)
                        coordinator = getattr(self, "joint_coordinator", None)
                        if coordinator is not None:
                            for key, value in coordinator.gradient_metrics(
                                optimizer
                            ).items():
                                setattr(cur_metric, f"joint_{key}", value)
                            if coordinator.is_segment_direct_joint:
                                coordinator.record_segment_joint_backward_evidence(
                                    model=model,
                                    pos=pos,
                                    iteration=iteration,
                                    overflow=float(model.overflow[-1]),
                                )
                            if (
                                coordinator.uses_overflow_milestone_actions
                                and joint_step.event is not None
                            ):
                                _profile_t = profile_now()
                                joint_step.sizing_frame = (
                                    self._capture_segment_milestone_sizing_frame(
                                        params=params,
                                    model=model,
                                    pos=pos,
                                    event=joint_step.event,
                                    evaluate_objective=True,
                                )
                                )
                                profile_add(
                                    "segment_milestone_sizing_frame_ms",
                                    _profile_t,
                                )
                            if (
                                coordinator.uses_fixed_window_actions
                                and joint_step.event is not None
                            ):
                                _profile_t = profile_now()
                                calibration = coordinator.calibrate_segment_actions(
                                    model=model,
                                    pos=pos,
                                    event=joint_step.event,
                                )
                                profile_add("segment_action_calibration_ms", _profile_t)
                                if calibration is not None:
                                    setattr(
                                        cur_metric,
                                        "joint_segment_action_calibration_samples",
                                        calibration.get("sample_count", 0),
                                    )
                                    setattr(
                                        cur_metric,
                                        "joint_segment_action_sign_agreement",
                                        calibration.get("sign_agreement_ratio"),
                                    )
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # plot placement
                    _profile_t = profile_now()
                    plot_interval = int(getattr(params, "plot_interval", 30) or 30)
                    should_plot_interval = plot_interval > 0 and iteration % plot_interval == 0
                    if params.plot_flag and (should_plot_interval or iteration == 999):
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)
                    profile_add("plot_ms", _profile_t)

                    # Clear the grads, reconstruct the placement model, compute the loss value, and the gradients
                    def closure():
                      density_weight = 0.0
                      model = construct_model()
                      obj, grad = model.obj_and_grad_fn(pos)
                      return obj

                    # Make a parameter update after having checked if closure must be called
                    def make_parameter_update():
                        if run_closure == 0:
                          optimizer.step()
                        else:
                          optimizer.step(closure)
                    
                    # stop updating fence regions that are marked stop, exclude the outer cell !
                    t3 = time.time()
                    optimizer_step_profile_start = profile_now()
                    if (
                        optimizer_name.lower() != "nesterov"
                        and isinstance(continuous_early_stop_state, dict)
                        and continuous_early_stop_state.get("enabled")
                        and continuous_early_stop_state.get("restore_best")
                    ):
                        continuous_early_stop_state["candidate_state"] = (
                            self._capture_continuous_early_stop_best_state(
                                snapshot=self._capture_size_debug_snapshot()
                            )
                        )
                        continuous_early_stop_state["candidate_state"][
                            "capture_phase"
                        ] = "pre_optimizer_step"
                    if (
                        optimizer_name.lower() == "nesterov"
                        and coordinator is not None
                        and coordinator.uses_pin2pin_rebootstrap_schedule
                        and coordinator.pin2pin_pair_generations.active
                    ):
                        joint_step.consumed_generation = (
                            coordinator.pin2pin_pair_generations.current_generation_id
                        )
                        joint_step.eval_count_before = sum(
                            int(group.get("obj_eval_count", 0) or 0)
                            for group in optimizer.param_groups
                        )
                    _profile_t = profile_now()
                    masked_size_grad_count = self._mask_continuous_size_gradient()
                    setattr(
                        cur_metric,
                        "real_size_masked_gradient_count",
                        int(masked_size_grad_count),
                    )
                    if model.update_mask is not None:
                        pos_bk = pos.data.clone()
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        make_parameter_update()

                        for region_id, fence_region_update_flag in enumerate[Any](model.update_mask):
                            if fence_region_update_flag == 0:
                                # don't update cell location in that region
                                mask = self.op_collections.fence_region_density_ops[region_id].pos_mask
                                pos.data.masked_scatter_(mask, pos_bk[mask])
                    else:
                        if hasattr(model, "set_l_shape_outer_iteration"):
                            model.set_l_shape_outer_iteration(iteration)
                        make_parameter_update()
                    _profile_t = profile_now()
                    real_size_projection_summary = (
                        self._project_real_size_after_optimizer_step(params)
                    )
                    real_size_transition_summary = (
                        self._maybe_transition_real_size_to_discrete(
                            params=params,
                            optimizer=optimizer,
                            iteration=iteration,
                            model=model,
                        )
                    )
                    if real_size_projection_summary is not None:
                        for key, value in real_size_projection_summary.items():
                            setattr(cur_metric, f"real_size_{key}", value)
                    setattr(
                        cur_metric,
                        "sizing_parameterization",
                        self._sizing_parameterization(params),
                    )
                    if real_size_transition_summary is not None:
                        setattr(
                            cur_metric,
                            "real_size_transition",
                            real_size_transition_summary,
                        )
                    profile_add("real_size_projection_ms", _profile_t)
                    coordinator = getattr(self, "joint_coordinator", None)
                    pending_request = joint_iteration.after_step(
                        self._joint_iteration_context(), coordinator, params, placedb,
                        model, pos, optimizer, optimizer_name, iteration, cur_metric,
                        joint_step,
                    )
                    if pending_request is not None:
                        pending_joint_topology_commit_result = pending_request
                        return cur_metric
                    profile_add("optimizer_parameter_update_ms", _profile_t)

                    _profile_t = profile_now()
                    self._apply_continuous_size_logits_dynamics(
                        params=params,
                        before_snapshot=size_debug_before,
                        iteration=iteration,
                        total_iterations=global_place_params["iteration"],
                        runtime_state=continuous_size_dynamics_state,
                        metric=cur_metric,
                        model=model,
                        pos=pos,
                    )
                    profile_add("discrete_gradient_topk_update_ms", _profile_t)

                    _profile_t = profile_now()
                    self._maybe_write_size_debug_trace(
                        params,
                        iteration,
                        cur_metric,
                        model,
                        optimizer,
                        size_debug_before,
                        self._capture_size_debug_snapshot(),
                        size_debug_grad_stats,
                    )
                    if (
                        optimizer_name.lower() == "nesterov"
                        and isinstance(continuous_early_stop_state, dict)
                        and continuous_early_stop_state.get("enabled")
                        and continuous_early_stop_state.get("restore_best")
                    ):
                        continuous_early_stop_state["candidate_state"] = (
                            self._capture_continuous_early_stop_best_state(
                                snapshot=self._capture_size_debug_snapshot()
                            )
                        )
                        continuous_early_stop_state["candidate_state"][
                            "capture_phase"
                        ] = "post_optimizer_step"
                    profile_add("size_debug_trace_ms", _profile_t)
                    profile_stage_ms["optimizer_step_ms"] = (
                        profile_now() - optimizer_step_profile_start
                    ) * 1000.0

                    logging.info("optimizer step %.3f ms" %
                                 ((time.time() - t3) * 1000))

                    # Perform timing-opt.
                    _profile_t = profile_now()
                    managed_pin2pin_schedule = bool(
                        coordinator is not None
                        and coordinator.uses_pin2pin_rebootstrap_schedule
                    )
                    if (
                        managed_pin2pin_schedule
                        and coordinator.pin2pin_pair_generations.periodic_update_due
                        and not coordinator.segment_milestones.transaction_active
                        and not coordinator.pin2pin_pair_generations.install_count_at_iteration(
                            iteration
                        )
                    ):
                        joint_step.pin2pin_update = (
                            self._rebuild_and_install_segment_pin2pin(
                                params=params,
                                placedb=placedb,
                                model=model,
                                pos=pos,
                                optimizer=optimizer,
                                iteration=iteration,
                                overflow=model.overflow[-1],
                                reason="periodic",
                                force_clear_accumulation=False,
                            )
                        )
                        self._update_metric_timing(
                            cur_metric,
                            *joint_step.pin2pin_update["timing"][:3],
                        )
                        cur_metric.nvp = 1
                    legacy_net_weight_gate_status = (
                        {
                            "enabled": True,
                            "update_due": False,
                            "iteration": int(iteration),
                            "overflow": float(model.overflow[-1]),
                            "threshold": float(
                                getattr(
                                    params,
                                    "timing_topology_enable_overflow_threshold",
                                    0.35,
                                )
                            ),
                            "update_interval": int(
                                getattr(params, "net_weighting_update_interval", 15)
                                or 15
                            ),
                            "reason": "managed_pin2pin_generation_schedule",
                        }
                        if managed_pin2pin_schedule
                        else self._legacy_net_weight_gate_status(
                            params,
                            iteration,
                            model.overflow[-1],
                        )
                    )
                    update_legacy_weights(
                        self._legacy_weight_context(), params, placedb, model, pos,
                        iteration, cur_metric, managed_pin2pin_schedule,
                        legacy_net_weight_gate_status,
                    )
                    profile_add("net_weighting_update_ms", _profile_t)


                    # nesterov has already computed the objective of the next step
                    if (
                        optimizer_name.lower() == "nesterov"
                        and optimizer.param_groups[0]["obj_k_1"]
                    ):
                        cur_metric.objective = optimizer.param_groups[0]["obj_k_1"][0].data.clone(
                        )

                    self._copy_timing_metrics_to_eval_metric(model, cur_metric)

                    l_shape_policy.maybe_auto_disable(model, iteration, outer_update=False)
                    l_shape_policy.maybe_disable_by_ratio(model, iteration)
                    model.collect_l_shape_telemetry(cur_metric)
                    routability_controller.after_iteration(
                        model=model,
                        iteration=iteration,
                        metrics=cur_metric,
                    )
                    # actually reports the metric before step
                    _profile_t = profile_now()
                    logging.info(cur_metric)
                    profile_add("metric_logging_ms", _profile_t)
                    # record the best outer cell overflow
                    _profile_t = profile_now()
                    if best_metric[0] is None or best_metric[0].overflow[-1] > cur_metric.overflow[-1]:
                        best_metric[0] = cur_metric
                        if cooptimization is None:
                            if best_pos[0] is None:
                                best_pos[0] = self.pos[0].data.clone()
                            else:
                                best_pos[0].data.copy_(self.pos[0].data)
                    profile_add("best_pos_update_ms", _profile_t)

                    if profile_enabled:
                        placement_step_profile.record_iteration(
                            self._write_full_step_profile_record, params, iteration,
                            Lgamma_step, Llambda_density_weight_step, Lsub_step,
                            profile_now, profile_step_start, profile_stage_ms,
                            timing_backward_component_profile, piecewise_size_backend_profile,
                            cell_model_forward_profile,
                        )

                    logging.info("full step %.3f ms" %
                                 ((time.time() - t0) * 1000))
                    return cur_metric

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
                stop_placement_reason = None
                last_perturb_iter = -min_perturb_interval
                perturb_counter = 0
                joint_no_sizing_full_budget = bool(
                    str(getattr(params, "flow_kind", "") or "").lower()
                    == "joint"
                    and str(
                        getattr(params, "joint_quality_profile", "") or ""
                    )
                    == "segment_count_direct_joint_v1"
                    and not bool(
                        getattr(params, "joint_segment_sizing_enabled", True)
                    )
                )
                if joint_no_sizing_full_budget:
                    logging.info(
                        "joint no-sizing placement budget: disable Lsub early stop"
                    )

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
                            if params.with_sta:
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

                            try:
                                cur_metric = one_descent_step(
                                    Lgamma_step,
                                    Llambda_density_weight_step,
                                    Lsub_step,
                                    iteration,
                                    Lsub_metrics,
                                )
                            except AssertionError as exc:
                                if not self._is_gradient_nan_assertion(exc):
                                    raise
                                if Lsub_metrics:
                                    self._mark_metric_gradient_nan_failure(
                                        Lsub_metrics[-1]
                                    )
                                stop_placement_reason = "gradient_nan"
                                stop_placement = 1
                                logging.warning(
                                    "gradient contains NaN at iteration %s; stop global placement before OpenROAD handoff",
                                    iteration,
                                )
                                break

                            if self._metric_has_nonfinite_value(cur_metric):
                                stop_placement_reason = "nonfinite_metric"
                                stop_placement = 1
                                logging.warning(
                                    "non-finite placement metric detected at iteration %s; stop global placement before OpenROAD handoff",
                                    iteration,
                                )
                                break

                            if handoff_controller is not None:
                                run_event = {
                                    "absolute_iteration": iteration,
                                    "stage": cur_stage,
                                    "iteration": iteration,
                                    "phase": cur_stage,
                                    "iteration_in_stage": Llambda_flat_iteration,
                                    "Lgamma_step": Lgamma_step,
                                    "Llambda_density_weight_step": Llambda_density_weight_step,
                                    "Lsub_step": Lsub_step,
                                    "metric": cur_metric,
                                }
                                handoff_requested, trigger_decision = handoff_controller.should_handoff(
                                    run_event
                                )
                                if handoff_requested:
                                    handoff_event = handoff_controller.build_handoff_event(
                                        run_event,
                                        trigger_decision,
                                    )
                                    handoff_result = self._execute_openroad_handoff_transaction(
                                        params,
                                        placedb,
                                        model.data_collections.pos[0],
                                        run_event,
                                        trigger_decision,
                                        handoff_event,
                                        model=model,
                                    )
                                    if handoff_result is not None:
                                        return handoff_result

                            if len(placedb.regions) == 0:
                                overflow_list.append(
                                    Llambda_metrics[-1][-1].overflow.data.item())
                                divergence_list.append(
                                    [
                                        Llambda_metrics[-1][-1].hpwl.data.item(),
                                        Llambda_metrics[-1][-1].overflow.data.item(),
                                    ]
                                )

                            early_stop_loss, early_stop_loss_source = (
                                self._continuous_early_stop_loss_and_source(
                                    params,
                                    Llambda_metrics[-1][-1],
                                )
                            )
                            if self._update_continuous_early_stop_state(
                                continuous_early_stop_state,
                                early_stop_loss,
                                iteration,
                                loss_source=early_stop_loss_source,
                            ):
                                stop_placement = 1
                                logging.info(
                                    "continuous early stop triggered at iteration %d after %d non-improving rounds "
                                    "(best_iter=%s best_loss=%s last_loss=%s loss_source=%s min_delta=%s)",
                                    iteration,
                                    continuous_early_stop_state.get("no_improve_rounds"),
                                    continuous_early_stop_state.get("best_iteration"),
                                    continuous_early_stop_state.get("best_loss"),
                                    continuous_early_stop_state.get("last_loss"),
                                    continuous_early_stop_state.get("last_loss_source"),
                                    continuous_early_stop_state.get("min_delta"),
                                )
                                break

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
                            if (
                                not joint_no_sizing_full_budget
                                and Lsub_stop_criterion(
                                    Lgamma_step,
                                    Llambda_density_weight_step,
                                    Lsub_step,
                                    Lsub_metrics,
                                )
                            ):
                                break
                        fatal_no_sizing_stop = stop_placement_reason in {
                            "gradient_nan",
                            "nonfinite_metric",
                        }
                        if pending_joint_topology_commit_result is not None or (
                            stop_placement == 1
                            and (
                                not joint_no_sizing_full_budget
                                or fatal_no_sizing_stop
                            )
                        ):
                            break
                        if stop_placement == 1 and joint_no_sizing_full_budget:
                            logging.info(
                                "joint no-sizing placement budget: ignoring non-fatal stop signal reason=%s",
                                stop_placement_reason,
                            )
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
                        if not joint_no_sizing_full_budget and llambda_should_stop and not defer_stop_for_inflation:
                            break
                        if pending_joint_topology_commit_result is not None or (
                            stop_placement == 1
                            and (
                                not joint_no_sizing_full_budget
                                or stop_placement_reason
                                in {"gradient_nan", "nonfinite_metric"}
                            )
                        ):
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
                                if cooptimization is not None:
                                    cooptimization.before_inflation(
                                        model, model.data_collections.pos[0],
                                        iteration=iteration, stage=cur_stage,
                                    )
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
                                elif use_enhanced_inflation and cooptimization is None:
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
                                if cooptimization is not None:
                                    cooptimization.publish_geometry(model)
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
                                    routability_controller.record_inflation()
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
                                    model.reset_l_shape_weight_state()
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

                    if pending_joint_topology_commit_result is not None or (
                        stop_placement == 1
                        and (
                            not joint_no_sizing_full_budget
                            or stop_placement_reason
                            in {"gradient_nan", "nonfinite_metric"}
                        )
                    ):
                        break
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
                        (
                            not joint_no_sizing_full_budget
                            and not inflation_applied_this_gamma
                            and Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics)
                        )
                        or (
                            stop_placement == 1
                            and (
                                not joint_no_sizing_full_budget
                                or stop_placement_reason
                                in {"gradient_nan", "nonfinite_metric"}
                            )
                        )
                        or pending_joint_topology_commit_result is not None
                    ):
                        break

                    # update learning rate
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]:
                        if "learning_rate_decay" in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] *= global_place_params["learning_rate_decay"]

                # in case of divergence, use the best metric
                last_metric = self._last_metric_from_metrics(all_metrics)
                if pending_joint_topology_commit_result is not None:
                    break
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
                    restore_macro_pin_halo(params, placedb, self.data_collections, self.pos[0])

                    if last_metric and (
                        last_metric.overflow[-1] > params.stop_overflow
                        or self._metric_objective_is_nonfinite(last_metric)
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
                write_crash_stage_marker(
                    "nonlinearplace_optimizer_loop_exit",
                    "done",
                    cur_stage=cur_stage,
                    optimizer=optimizer_name,
                    iteration=iteration,
                )

            coordinator = getattr(self, "joint_coordinator", None)
            if (
                coordinator is not None
                and coordinator.uses_pin2pin_rebootstrap_schedule
                and (
                    coordinator.segment_milestones.has_pending_work
                    or coordinator.pin2pin_pair_generations.pending
                )
            ):
                incomplete = coordinator.mark_incomplete_pending_milestone(
                    iteration=iteration,
                    reason="placement_step_budget_exhausted",
                )
                self._update_joint_flow_artifact(params)
                raise RuntimeError(
                    "incomplete_pending_milestone: "
                    f"queue={incomplete['queue']} "
                    f"transaction={incomplete['active_transaction']} "
                    f"pair_generation_pending={incomplete['pair_generation_pending']}"
                )

            self._restore_continuous_early_stop_best_state(
                continuous_early_stop_state,
                reason="before_projection",
            )

            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                selected_inflation_round = None
                replay_best_inflation_round = bool(
                    getattr(params, "enhanced_inflation_replay_best_round_flag", False)
                ) and cooptimization is None
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
                    physical_end = placedb.num_physical_nodes if cooptimization is not None else placedb.num_nodes
                    self.data_collections.node_size_x[:physical_end].copy_(
                        self.data_collections.original_node_size_x[:physical_end])
                    self.data_collections.node_size_y[:physical_end].copy_(
                        self.data_collections.original_node_size_y[:physical_end])
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
                if cooptimization is not None:
                    cooptimization.publish_geometry(model)
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
            self._augment_metric_observability(cur_metric)
            logging.info(cur_metric)
        if params.plot_flag:
            write_crash_stage_marker(
                "nonlinearplace_final_plot",
                "start",
                iteration=9999,
            )
            self.plot(params, placedb, 9999,
                      self.pos[0].data.clone().cpu().numpy())
            write_crash_stage_marker(
                "nonlinearplace_final_plot",
                "done",
                iteration=9999,
            )

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
            processed_metrics = self._processed_global_place_metrics(all_metrics)
            metrics = [metric for metric in self._flatten_metrics(all_metrics) if metric is not None]
            l_shape_telemetry.append_processed_metric_series(processed_metrics, metrics)
            place_stage_metrics = copy.deepcopy(all_metrics)

            if (
                params.with_sta
                and getattr(self.op_collections, "steiner_topo_op", None) is not None
                and "model" in locals()
                and hasattr(model, "timing_obj")
                and pending_joint_topology_commit_result is None
                and cooptimization is None
            ):
                write_crash_stage_marker(
                    "nonlinearplace_post_global_timing_refresh",
                    "start",
                    pos_device=str(self.pos[0].device),
                )
                with torch.no_grad():
                    final_global_topology = self._refresh_live_timing_topology(
                        self.pos[0].data
                    )
                    wns_gp, tns_gp, ws_gp, ts_gp = model.timing_obj(self.pos[0].data)
                    self._record_timing_metrics(wns_gp, tns_gp, ws_gp, ts_gp)
                write_crash_stage_marker(
                    "nonlinearplace_post_global_timing_refresh",
                    "done",
                    pos_device=str(self.pos[0].device),
                    steiner_input_device=str(final_global_topology["node_x"].device),
                )

            if pending_joint_topology_commit_result is not None:
                return self._safe_stop_after_joint_buffer_request(
                    params,
                    iteration=iteration,
                    place_stage_metrics=place_stage_metrics,
                    processed_metrics=processed_metrics,
                    commit_result=pending_joint_topology_commit_result,
                )

            # plot placement
            if params.plot_flag:
                self.plot(params, placedb, iteration,
                          self.pos[0].data.clone().cpu().numpy())

            last_metric = self._last_metric_from_metrics(all_metrics)

            if self._should_skip_post_global_placement(
                stop_placement_reason,
                last_metric,
            ):
                logging.warning(
                    "skip legalization and detail placement steps because global placement stopped due to %s",
                    stop_placement_reason,
                )
                self.plot(params, placedb, 9999,
                          self.pos[0].data.clone().cpu().numpy())
                self._write_placement_debug_summary(
                    params,
                    iteration=iteration,
                    processed_metrics=processed_metrics,
                    last_metric=last_metric,
                    status="failed",
                    stop_reason=stop_placement_reason,
                    entered_legalization=False,
                    skipped_legalization_reason=stop_placement_reason,
                )
                return float("inf"), float("inf"), processed_metrics

            if not last_metric:
                cur_metric = EvalMetrics.EvalMetrics(iteration)
                all_metrics.append(cur_metric)
                cur_metric.evaluate(
                    placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0]
                )
                logging.info(cur_metric)

            # in case of significant divergence, no need to run legalizer
            placement_overflow_tolerance = max(
                0.0,
                float(getattr(params, "placement_overflow_tolerance", 0.0) or 0.0),
            )
            if last_metric and (
                (
                    self._overflow_blocks_placement_success(params)
                    and last_metric.overflow[-1]
                    > params.stop_overflow + placement_overflow_tolerance
                )
                or self._metric_objective_is_nonfinite(last_metric)
            ):
                logging.warn(
                    "overflow is significant %.3f (threshold %.3f + tolerance %.3g) "
                    "or hpwl is infinity or nan, skip legalization and detail placement steps"
                    % (
                        last_metric.overflow[-1],
                        params.stop_overflow,
                        placement_overflow_tolerance,
                    )
                )
                write_crash_stage_marker(
                    "nonlinearplace_call_return",
                    "start",
                    reason="skip_legalization_significant_overflow",
                    iteration=iteration,
                )
                self.plot(params, placedb, 9999,
                          self.pos[0].data.clone().cpu().numpy())
                write_crash_stage_marker(
                    "nonlinearplace_call_return",
                    "done",
                    reason="skip_legalization_significant_overflow",
                    iteration=iteration,
                    rsmt_wl=float("inf"),
                    hpwl=float("inf"),
                )
                self._write_placement_debug_summary(
                    params,
                    iteration=iteration,
                    processed_metrics=processed_metrics,
                    last_metric=last_metric,
                    status="failed",
                    stop_reason=stop_placement_reason,
                    entered_legalization=False,
                    skipped_legalization_reason="skip_legalization_significant_overflow",
                )
                return float("inf"), float("inf"), processed_metrics

            if cooptimization is not None:
                commit_result = cooptimization.prepare_terminal_request(
                    model, self.pos[0], iteration=iteration,
                )
                self.last_buffer_commit_request = cooptimization.terminal_request
                self.last_buffering_lane = cooptimization.lane
                return self._safe_stop_after_joint_buffer_request(
                    params,
                    iteration=iteration,
                    place_stage_metrics=place_stage_metrics,
                    processed_metrics=processed_metrics,
                    commit_result=commit_result,
                )

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
            restore_macro_pin_halo(params, placedb, self.data_collections, self.pos[0])

        if self._joint_flow_enabled(params) and not self._segment_direct_joint_enabled():
            write_crash_stage_marker(
                "nonlinearplace_final_joint_buffer_commit",
                "start",
                iteration=iteration,
            )
            final_joint_commit = self._finalize_joint_buffer_commit(
                params,
                iteration=iteration,
                optimizer=optimizer if "optimizer" in locals() else None,
                final_pos=self.pos[0],
            )
            write_crash_stage_marker(
                "nonlinearplace_final_joint_buffer_commit",
                "done",
                iteration=iteration,
                accepted_action_count=(
                    (final_joint_commit or {}).get("commit", {}) or {}
                ).get("accepted_action_count")
                if isinstance(final_joint_commit, dict)
                else None,
            )
            if final_joint_commit is not None:
                logging.info("joint final buffer commit summary: %s", final_joint_commit)
                if self._joint_buffer_request_requires_outer_commit(final_joint_commit):
                    return self._safe_stop_after_joint_buffer_request(
                        params,
                        iteration=iteration,
                        place_stage_metrics=place_stage_metrics,
                        processed_metrics=processed_metrics,
                        commit_result=final_joint_commit,
                    )

        legalization_stage_start = len([metric for metric in self._flatten_metrics(all_metrics) if metric is not None])
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
            cur_metric = all_metrics[-1]
            self._augment_metric_observability(cur_metric)

            # perform an additional timing analysis on the legalized solution.
            # sta after legalization is not needed anymore.
            if (
                params.with_sta
                and getattr(self.op_collections, "steiner_topo_op", None) is not None
                and "model" in locals()
                and hasattr(model, "timing_obj")
            ):
                # TODO:
                logging.info("additional sta after legalization")
                with torch.no_grad():
                    self._refresh_live_timing_topology(self.pos[0].data)
                    wns_tp, tns_tp, ws_tp, ts_tp = model.timing_obj(self.pos[0].data)
                    self._record_timing_metrics(wns_tp, tns_tp, ws_tp, ts_tp)

                # Report tns and wns in each timing feedback call.
                # Note that OpenTimer considers early,late,rise,fall for tns/wns.
                # The following values are for reference.
                self._update_metric_timing(cur_metric, wns_tp, tns_tp, ws_tp)

            if (
                getattr(params, "with_sta", False)
                and not getattr(params, "differentiable_timing_obj", 0)
            ):
                with torch.no_grad():
                    self._refresh_timing_metrics(model, self.pos[0])
                    self._copy_timing_metrics_to_eval_metric(model, cur_metric)
            logging.info(cur_metric)

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

        legalization_stage_metrics = copy.deepcopy(
            [metric for metric in self._flatten_metrics(all_metrics)[legalization_stage_start:] if metric is not None]
        )

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
            self._augment_metric_observability(cur_metric)
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
        if self._segment_direct_joint_enabled():
            # Keep the local PlaceDB snapshot coherent while deferring every
            # OpenROAD side effect to the combined physical-action executor.
            placedb.node_x[: placedb.num_movable_nodes] = cur_pos[
                : placedb.num_movable_nodes
            ]
            placedb.node_y[: placedb.num_movable_nodes] = cur_pos[
                placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
            ]
            placedb.last_sizing_writeback_summary = None
        else:
            placedb.apply(
                params,
                cur_pos[0: placedb.num_movable_nodes],
                cur_pos[
                    placedb.num_nodes : placedb.num_nodes
                    + placedb.num_movable_nodes
                ],
            )

        if self._segment_direct_joint_enabled():
            boundary_route_b = self._apply_terminal_segment_route_b_if_needed(
                params,
                iteration=iteration,
                optimizer=optimizer if "optimizer" in locals() else None,
            )
            if boundary_route_b is not None:
                logging.info(
                    "joint segment terminal Route-B reselection historical=%d "
                    "final=%d iterations=%d reason=%s",
                    int(boundary_route_b.get("historical_buffer_count", 0) or 0),
                    int(boundary_route_b.get("final_buffer_count", 0) or 0),
                    int(boundary_route_b.get("iterations", 0) or 0),
                    boundary_route_b.get("terminal_reason"),
                )
            write_crash_stage_marker(
                "nonlinearplace_final_joint_buffer_commit",
                "start",
                iteration=iteration,
                coordinate_source="final_live_geometry",
            )
            final_joint_commit = self._finalize_joint_buffer_commit(
                params,
                iteration=iteration,
                optimizer=optimizer if "optimizer" in locals() else None,
                final_pos=self.pos[0],
            )
            write_crash_stage_marker(
                "nonlinearplace_final_joint_buffer_commit",
                "done",
                iteration=iteration,
                request_status=(final_joint_commit or {}).get("status")
                if isinstance(final_joint_commit, dict)
                else None,
            )
            if final_joint_commit is not None:
                logging.info("joint final buffer commit summary: %s", final_joint_commit)
                if self._joint_buffer_request_requires_outer_commit(final_joint_commit):
                    return self._safe_stop_after_joint_buffer_request(
                        params,
                        iteration=iteration,
                        place_stage_metrics=place_stage_metrics,
                        processed_metrics=processed_metrics,
                        commit_result=final_joint_commit,
                    )

        # apply macro orientations solution
        if False:
            orients_map = {
                "N": 0,
                "S": 1,
                "W": 2,
                "E": 3,
                "FN": 4,
                "FS": 5,
                "FW": 6,
                "FE": 7,
                "UNKNOWN": 8,
            }
            for macro, orient in macro_orients:
                placedb.rawdb.setNodeOrient(
                    int(macro), place_io_cpp.OrientEnum.OrientType(
                        orients_map[orient])
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

        # apply macro orientations solution
        if False:
            orients_map = {
                "N": 0,
                "S": 1,
                "W": 2,
                "E": 3,
                "FN": 4,
                "FS": 5,
                "FW": 6,
                "FE": 7,
                "UNKNOWN": 8,
            }
            for macro, orient in macro_orients:
                placedb.rawdb.setNodeOrient(
                    int(macro), place_io_cpp.OrientEnum.OrientType(
                        orients_map[orient])
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

        # run final RSMT/timing dump only when the live timing stack is available
        rsmt_wl = float("nan")
        if (
            params.with_sta
            and getattr(placedb, "gr_sizing", None) is None
            and getattr(self.op_collections, "steiner_topo_op", None) is not None
            and "model" in locals()
            and hasattr(model, "timing_obj")
        ):
            write_crash_stage_marker(
                "nonlinearplace_final_timing_dump",
                "start",
                pos_device=str(self.pos[0].device),
            )
            with torch.no_grad():
                tt = time.time()
                final_topology = self._refresh_live_timing_topology(self.pos[0])
                new_x = final_topology["node_x"]
                new_y = final_topology["node_y"]
                flat_pin_to = self.data_collections.flat_pin_to
                flat_pin_from = self.data_collections.flat_pin_from

                length = (torch.abs(new_x[flat_pin_from] - new_x[flat_pin_to])
                        + torch.abs(new_y[flat_pin_from] - new_y[flat_pin_to])) / params.scale_factor  / placedb.dbu
                flute_length_path = "%s/%s_flute_length.txt" % (
                    params.result_dir, params.design_name())
                with open(flute_length_path, "w") as f:
                    f.write("net_name, flute_length (um), cap (pF)\n")
                    for net_id in range(placedb.num_nets):
                        start = self.data_collections.net_flat_topo_sort_start[net_id]
                        end = self.data_collections.net_flat_topo_sort_start[net_id + 1]
                        net_name = placedb.net_names[net_id]
                        net_flat_pins_nodes = self.data_collections.net_flat_topo_sort[start:end]
                        net_flat_pin_start = self.data_collections.flat_pin_to_start[net_flat_pins_nodes]
                        net_flat_pin_end = self.data_collections.flat_pin_to_start[net_flat_pins_nodes + 1]
                        net_length = 0
                        for i in range(len(net_flat_pins_nodes)):
                            pin_start = net_flat_pin_start[i]
                            pin_end = net_flat_pin_end[i]
                            net_length += length[pin_start:pin_end].sum().item()
                        f.write(f"{net_name}, {net_length}, {placedb.c_unit * net_length}\n")

                wns, tns, ws, ts = model.timing_obj(self.pos[0])
                if (
                    hasattr(model, "check_log")
                    and not self._openroad_backed_aimp_db_enabled(params)
                ):
                    model.check_log(wns, tns, ws, ts)
                self._record_timing_metrics(wns, tns, ws, ts)
            rsmt_wl = self.op_collections.rsmt_wl_op(self.pos[0]) / placedb.dbu
            logging.info("rsmt computation takes %.3f seconds" %
                         (time.time() - tt))
            logging.info("flute rsmt %.6E um" % rsmt_wl)
            write_crash_stage_marker(
                "nonlinearplace_final_timing_dump",
                "done",
                pos_device=str(self.pos[0].device),
                steiner_input_device=str(new_x.device),
                rsmt_wl=float(rsmt_wl),
            )
        elif (
            params.global_place_flag
            and getattr(placedb, "gr_sizing", None) is None
            and getattr(self.op_collections, "rsmt_wl_op", None) is not None
        ):
            write_crash_stage_marker(
                "nonlinearplace_final_timing_dump",
                "skipped",
                reason="live_timing_stack_unavailable",
                pos_device=str(self.pos[0].device),
            )
            with torch.no_grad():
                rsmt_wl = self.op_collections.rsmt_wl_op(self.pos[0]) / placedb.dbu
                logging.info("flute rsmt %.6E um" % rsmt_wl)
        else:
            write_crash_stage_marker(
                "nonlinearplace_final_timing_dump",
                "skipped",
                reason="rsmt_op_unavailable",
                pos_device=str(self.pos[0].device),
            )

        # get HPWL
        write_crash_stage_marker(
            "nonlinearplace_final_hpwl",
            "start",
            pos_device=str(self.pos[0].device),
        )
        with torch.no_grad():
            hpwl = self.op_collections.hpwl_op(self.pos[0])
            rsmt_wl = self.op_collections.rsmt_wl_op(self.pos[0]) / placedb.dbu
            logging.info("flute rsmt %.6E um" % rsmt_wl)
            logging.info("unweighted hpwl %.6E" % hpwl)
        write_crash_stage_marker(
            "nonlinearplace_final_hpwl",
            "done",
            pos_device=str(self.pos[0].device),
            hpwl=float(hpwl.detach().cpu().item() if torch.is_tensor(hpwl) else hpwl),
        )

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

        if not place_stage_metrics:
            place_stage_metrics = copy.deepcopy(
                [metric for metric in self._flatten_metrics(all_metrics) if metric is not None]
            )
        write_crash_stage_marker(
            "nonlinearplace_metrics_artifacts_write",
            "start",
        )
        self._write_metrics_artifacts(params, place_stage_metrics, stage_name="place")
        if legalization_stage_metrics:
            self._write_metrics_artifacts(params, legalization_stage_metrics, stage_name="legalization")
        write_crash_stage_marker(
            "nonlinearplace_metrics_artifacts_write",
            "done",
            place_metric_count=len(place_stage_metrics),
            legalization_metric_count=len(legalization_stage_metrics),
        )
        write_crash_stage_marker(
            "nonlinearplace_timing_artifacts_write",
            "start",
        )
        timing_stage_summary = self._write_timing_stage_artifact(
            params,
            place_stage_metrics,
            legalization_stage_metrics,
        )
        if "model" in locals():
            self._write_timing_objective_terms_artifact(params, model)
        self._write_surrogate_cache_artifact(
            params,
            model=model if "model" in locals() else None,
        )
        write_crash_stage_marker(
            "nonlinearplace_timing_artifacts_write",
            "done",
        )
        write_crash_stage_marker(
            "nonlinearplace_projection_artifacts_write",
            "start",
        )
        self._write_projection_artifacts(params, stage_timing_summary=timing_stage_summary)
        self._write_size_limit_samples_artifact(params)
        write_crash_stage_marker(
            "nonlinearplace_projection_artifacts_write",
            "done",
        )
        write_crash_stage_marker(
            "nonlinearplace_runtime_projection_refresh",
            "start",
        )
        projection_runtime_summary = self._apply_projected_runtime_refresh(params, placedb)
        write_crash_stage_marker(
            "nonlinearplace_runtime_projection_refresh",
            "done",
            has_summary=projection_runtime_summary is not None,
        )
        write_crash_stage_marker(
            "nonlinearplace_runtime_projection_finalize",
            "start",
        )
        self.last_final_check_timing_record = None
        self._finalize_after_runtime_projection_refresh(
            params,
            placedb,
            cur_pos,
            model if "model" in locals() else None,
            projection_runtime_summary,
        )
        write_crash_stage_marker(
            "nonlinearplace_runtime_projection_finalize",
            "done",
        )
        write_crash_stage_marker(
            "nonlinearplace_post_projection_timing_refresh",
            "start",
        )
        timing_stage_summary = self._refresh_post_projection_timing_stage_artifact(
            params,
            timing_stage_summary=timing_stage_summary,
        )
        timing_stage_summary = self._apply_final_check_timing_stage_record(
            timing_stage_summary
        )
        if isinstance(timing_stage_summary, dict):
            summary_path = os.path.join(
                params.result_dir,
                f"{params.design_name()}_timing_stage_summary.json",
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(timing_stage_summary, f, ensure_ascii=False, indent=2)
                f.write("\n")
            self.last_timing_stage_summary = timing_stage_summary
        write_crash_stage_marker(
            "nonlinearplace_post_projection_timing_refresh",
            "done",
        )
        write_crash_stage_marker(
            "nonlinearplace_stage_compare_refresh",
            "start",
        )
        self._refresh_stage_compare_latest_artifact(params, timing_stage_summary=timing_stage_summary)
        write_crash_stage_marker(
            "nonlinearplace_stage_compare_refresh",
            "done",
        )

        final_rsmt_wl = float(rsmt_wl)
        final_hpwl = float(hpwl)
        released_generation = self._release_size_only_frozen_topology(params)
        if released_generation is not None:
            logging.info(
                "released size-only timing topology generation=%d after final reports",
                released_generation,
            )
        self._write_placement_debug_summary(
            params,
            iteration=iteration,
            optimizer_steps=len(place_stage_metrics),
            processed_metrics=processed_metrics,
            last_metric=self._last_metric_from_metrics(all_metrics),
            status="ok",
            stop_reason=stop_placement_reason if "stop_placement_reason" in locals() else None,
            entered_legalization=bool(legalization_stage_metrics),
            skipped_legalization_reason=None,
        )
        write_crash_stage_marker(
            "nonlinearplace_call_return",
            "done",
            reason="normal_return",
            iteration=iteration,
            rsmt_wl=final_rsmt_wl,
            hpwl=final_hpwl,
        )
        params.stop_overflow = original_stop_overflow
        return final_rsmt_wl, final_hpwl, processed_metrics
