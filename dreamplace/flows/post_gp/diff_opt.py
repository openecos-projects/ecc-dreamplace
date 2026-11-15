"""Standalone placement-parasitic diff optimization controller."""


import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
import numpy as np
from dreamplace.flows.discrete.buffering_filter import refresh_topology_if_needed
from dreamplace.flows.discrete.buffer_commit import (
    clear_virtual_buffer_state,
    commit_buffer_request_to_ecc,
    project_window_buffer_request,
    supports_direct_buffer_commit,
)
from dreamplace.flows.discrete.owner import DiscreteTimingOwner
from dreamplace.flows.config.specs import TIMING_OPT_COEFFICIENT_FIELDS
from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
from dreamplace.flows.config.specs import DiffFlowSpec, parse_diff_opt_flows


__all__ = [
    "DiffOptController",
    "run_diff_opt",
]


class DiffOptController:
    """Run an ordered diff_opt tuple list on a fixed placement.

    The controller owns virtual sizing/candidate state for one run.  Native
    sizing and buffer materialization are exposed as one terminal request only
    after all phases have completed.
    """

    def __init__(
        self,
        params,
        placedb,
        data,
        ops,
        refresh_topology,
        *,
        lane=None,
        rebuild_runtime_fn=None,
    ):
        self.params = params
        self.placedb = placedb
        self.data = data
        self.ops = ops
        self._rebuild_runtime_fn = rebuild_runtime_fn
        self.direct_commit_enabled = supports_direct_buffer_commit(params, placedb)
        self.owner = DiscreteTimingOwner(
            params,
            placedb,
            data,
            ops,
            refresh_topology,
            refresh_candidates_after_sizing=True,
        )
        self.owner.lane = lane
        self.owner.direct_commit_enabled = self.direct_commit_enabled
        self.direct_commit_records = []
        self.full_rebuild_count = 0
        self.trace = []
        self.phase_count = 0
        self.geometry_refresh_count = 0
        self.timing_refresh_count = 0
        self.terminal_request = None
        self.status = "created"

    @contextmanager
    def _phase_coefficients(self, model, coefficients):
        original = {
            name: float(getattr(model, field))
            for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items()
        }
        original_weight = model.timing_grad_balance_weight
        hold = getattr(model, "hold_objective", None)
        try:
            for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
                setattr(model, field, float(getattr(coefficients, name)))
            if hold is None and (
                coefficients.hold_wns or coefficients.hold_tns
            ):
                raise ValueError(
                    "nonzero diff_opt hold weights require timing_hold_enabled"
                )
            if hold is not None:
                model.hold_objective = replace(
                    hold,
                    wns=float(coefficients.hold_wns),
                    tns=float(coefficients.hold_tns),
                )
            model.timing_grad_balance_weight = 1.0
            yield
        finally:
            for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
                setattr(model, field, original[name])
            if hold is not None:
                model.hold_objective = hold
            model.timing_grad_balance_weight = original_weight

    @contextmanager
    def _phase_config(self, spec: DiffFlowSpec):
        names = {
            "timing_opt_sizing_rounds": int(spec.steps),
            "buffering_continuous_steps": int(spec.steps),
        }
        previous = {name: getattr(self.params, name, None) for name in names}
        for name, value in names.items():
            setattr(self.params, name, value)
        lane = self.owner.lane
        previous_lane_config = getattr(lane, "config", None) if lane is not None else None
        if lane is not None and previous_lane_config is not None:
            lane.config = replace(
                previous_lane_config,
                continuous_steps=int(spec.steps),
                route_b_selection_fraction=float(
                    getattr(self.params, "diff_buffering_selection_fraction", 0.003)
                ),
            )
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(self.params, name, value)
            if lane is not None and previous_lane_config is not None:
                lane.config = previous_lane_config

    def _state_digest(self):
        values = [("inst_cell_id", self.data.inst_cell_id)]
        if getattr(self.data, "vt_logits", None) is not None:
            values.append(("vt_logits", self.data.vt_logits))
        lane = self.owner.lane
        state = lane._current_state() if lane is not None else None
        if state is not None:
            for name in ("activation_param", "z_param", "candidate_ids", "net_ids"):
                value = getattr(state, name, None)
                if value is not None:
                    values.append((name, value))
        return _tensor_digest(values)

    def _ensure_lane(self, model, pos):
        # Every buffering phase enables the complete candidate search domain.
        # The lane reuses its retained domain when the structural topology is
        # unchanged and only toggles the timing-active view.
        self.owner.prepare(model, pos, active_only=False)
        lane = self.owner.lane
        if lane is None or lane._current_state() is None:
            raise RuntimeError("diff_opt failed to prepare its candidate lane")
        lane_config = getattr(lane, "config", None)
        if lane_config is not None and is_dataclass(lane_config):
            steps = getattr(self.params, "buffering_continuous_steps", None)
            if steps is not None:
                lane.config = replace(
                    lane_config,
                    continuous_steps=int(steps),
                    route_b_selection_fraction=float(
                        getattr(self.params, "diff_buffering_selection_fraction", 0.003)
                    ),
                )
        timing_loss = getattr(model, "_timing_loss", None)
        if not callable(timing_loss):
            raise RuntimeError("diff_opt requires the model timing loss callback")
        lane.timing_loss = timing_loss
        self.params.enable_relaxed_buffer_timing = True

    def run(self, model, pos, *, flows=None, iteration=0):
        mode, specs = parse_diff_opt_flows(
            getattr(self.params, "diff_optimization_mode", "buffering"),
            flows if flows is not None else getattr(self.params, "diff_optimization_flows", None),
        )
        started_at = time.perf_counter()
        self.status = "active"
        trailing_committed_without_rebuild = False
        for phase_index, spec in enumerate(specs):
            phase_started_at = time.perf_counter()
            topology_refresh_start = int(
                getattr(self.owner, "topology_refresh_count", 0)
            )
            topology_reuse_start = int(
                getattr(self.owner, "topology_refresh_reuse_count", 0)
            )
            phase_runtime_profile = (
                {}
                if bool(getattr(self.params, "buffering_runtime_profile", False))
                else None
            )

            def record_phase_stage(
                name,
                started_at,
                runtime_profile=phase_runtime_profile,
            ):
                if runtime_profile is not None:
                    runtime_profile[name] = (
                        time.perf_counter() - started_at
                    ) * 1000.0

            input_state_digest = self._state_digest()
            with (
                self.owner.electrical_window(),
                self._phase_config(spec),
                self._phase_coefficients(model, spec.coefficients),
            ):
                if spec.action == "sz":
                    prepare_started_at = time.perf_counter()
                    if self.direct_commit_enabled:
                        self.params.enable_relaxed_buffer_timing = False
                        refresh_topology_if_needed(self.owner, pos)
                        prepare_runtime_profile = {}
                    else:
                        # S only needs the buffers already retained in the design.
                        # The next BF phase enables the complete action domain.
                        self.owner.prepare(model, pos, active_only=True)
                        prepare_runtime_profile = dict(
                            getattr(self.owner, "last_prepare_runtime_profile", {}) or {}
                        )
                    record_phase_stage("prepare_ms", prepare_started_at)
                    from dreamplace.ops.routability.cooptimization_area import SizingWindowGeometry

                    geometry = SizingWindowGeometry(self.data, self.placedb, pos)
                    operation_started_at = time.perf_counter()
                    result = self.owner.run_sizing(model, pos, geometry, self.owner.capacity)
                    record_phase_stage("operation_ms", operation_started_at)
                else:
                    prepare_started_at = time.perf_counter()
                    self._ensure_lane(model, pos)
                    prepare_runtime_profile = dict(
                        getattr(self.owner, "last_prepare_runtime_profile", {}) or {}
                    )
                    record_phase_stage("prepare_ms", prepare_started_at)
                    operation_started_at = time.perf_counter()
                    result = self.owner.run_buffering(model, pos, self.owner.capacity)
                    record_phase_stage("operation_ms", operation_started_at)

            result = dict(result or {})
            lane = self.owner.lane
            summarize = getattr(lane, "summarize", None)
            lane_summary = summarize() if callable(summarize) else None

            if self.direct_commit_enabled and spec.action == "bf":
                accepted_count = int(result.get("accepted", 0) or 0)
                result["cumulative_count"] = (
                    int(self.owner.committed_buffer_count) + accepted_count
                )
                if accepted_count > 0:
                    commit_started_at = time.perf_counter()
                    req = project_window_buffer_request(
                        self.owner,
                        model,
                        pos,
                        iteration=int(iteration),
                        reason=f"diff_opt_phase_{phase_index:03d}_bf",
                    )
                    clear_virtual_buffer_state(self.owner, model)
                    commit_info = commit_buffer_request_to_ecc(
                        self.owner,
                        self.placedb,
                        self.params,
                        req,
                        pos=pos,
                        output_dir=getattr(self.params, "result_dir", None),
                    )
                    result["direct_buffer_commit"] = commit_info
                    self.direct_commit_records.append(commit_info)
                    self.full_rebuild_count += 1
                    record_phase_stage("retain_ms", commit_started_at)
                    if phase_index + 1 < len(specs) and callable(self._rebuild_runtime_fn):
                        publish_started_at = time.perf_counter()
                        prev_committed_count = int(self.owner.committed_buffer_count)
                        prev_committed_records = list(self.owner.committed_buffer_records)
                        prev_osc_snapshot = self.owner.oscillation.snapshot()
                        (
                            model,
                            pos,
                            new_data,
                            new_ops,
                            new_refresh_topology,
                            new_lane,
                        ) = self._rebuild_runtime_fn()
                        self.data = new_data
                        self.ops = new_ops
                        self.owner = DiscreteTimingOwner(
                            self.params,
                            self.placedb,
                            new_data,
                            new_ops,
                            new_refresh_topology,
                            refresh_candidates_after_sizing=True,
                        )
                        self.owner.lane = new_lane
                        self.owner.direct_commit_enabled = True
                        self.owner.committed_buffer_count = prev_committed_count
                        self.owner.committed_buffer_records = prev_committed_records
                        self.owner.oscillation.restore(prev_osc_snapshot)
                        trailing_committed_without_rebuild = False
                        geometry_refresh = self.owner.publish_geometry(model)
                        record_phase_stage("publish_geometry_ms", publish_started_at)
                    else:
                        trailing_committed_without_rebuild = True
                        geometry_refresh = {
                            "status": "committed_to_ecc_db",
                            "accepted_count": accepted_count,
                        }
                else:
                    trailing_committed_without_rebuild = False
                    clear_virtual_buffer_state(self.owner, model)
                    publish_started_at = time.perf_counter()
                    geometry_refresh = self.owner.publish_geometry(model)
                    record_phase_stage("publish_geometry_ms", publish_started_at)
            else:
                trailing_committed_without_rebuild = False
                # A final buffering phase is already consumed by the terminal
                # projection below.  Freezing and immediately repacking its
                # candidates adds no state needed by a later phase and can make
                # the native topology packer observe a partially frozen tree.
                # Retain accepted nets only when another explicit phase follows.
                if spec.action == "bf" and phase_index + 1 < len(specs):
                    retain_started_at = time.perf_counter()
                    self.owner.retain_buffered_nets(model)
                    record_phase_stage("retain_ms", retain_started_at)
                publish_started_at = time.perf_counter()
                geometry_refresh = self.owner.publish_geometry(model)
                record_phase_stage("publish_geometry_ms", publish_started_at)
            self.geometry_refresh_count += 1
            self.timing_refresh_count += 1
            if lane_summary is not None:
                result["candidate_state"] = {
                    key: lane_summary.get(key)
                    for key in (
                        "candidate_count",
                        "net_count",
                        "state_kind",
                        "candidate_geometry_refresh",
                        "candidate_geometry_runtime_profile",
                        "candidate_geometry_runtime_profile_history",
                    )
                    if lane_summary.get(key) is not None
                }
                refresh = result["candidate_state"].get("candidate_geometry_refresh")
                if isinstance(refresh, dict):
                    result["candidate_state"]["candidate_geometry_refresh"] = {
                        key: refresh.get(key)
                        for key in (
                            "candidate_topology_generation",
                            "candidate_geometry_generation",
                            "candidate_geometry_digest",
                            "candidate_topology_digest",
                            "candidate_geometry_invalidated_count",
                            "candidate_activation_remap",
                            "candidate_geometry_refresh_reason",
                            "candidate_geometry_refresh_status",
                        )
                        if refresh.get(key) is not None
                    }
            if result.get("trace_path"):
                source = result["trace_path"]
                directory, name = os.path.split(source)
                phase_trace = os.path.join(directory, f"phase_{phase_index:03d}_{name}")
                shutil.copyfile(source, phase_trace)
                result["trace_path"] = phase_trace
            restore = result.get("restore") or result.get("best_restore")
            result.update(
                {
                    "phase_index": int(phase_index),
                    "action": spec.action,
                    "steps": int(spec.steps),
                    "coefficients": spec.coefficients.to_dict(),
                    "input_state_digest": input_state_digest,
                    "output_state_digest": self._state_digest(),
                    "restore": restore,
                    "phase_status": (
                        "no_valid_action"
                        if int(result.get("accepted", 0) or 0) == 0
                        and not result.get("rounds")
                        and int(result.get("scheduled", 0) or 0) == 0
                        else "published"
                    ),
                    "geometry_refresh": geometry_refresh,
                    "topology_refresh": {
                        "refresh_count": int(
                            getattr(self.owner, "topology_refresh_count", 0)
                        )
                        - topology_refresh_start,
                        "reuse_count": int(
                            getattr(self.owner, "topology_refresh_reuse_count", 0)
                        )
                        - topology_reuse_start,
                        "last_state": dict(
                            getattr(
                                self.owner,
                                "last_topology_refresh_state",
                                {},
                            )
                            or {}
                        ),
                    },
                    "prepare_runtime_profile": prepare_runtime_profile,
                    "runtime_ms": (time.perf_counter() - phase_started_at) * 1000.0,
                }
            )
            if phase_runtime_profile is not None:
                accounted_ms = sum(
                    float(phase_runtime_profile.get(name, 0.0) or 0.0)
                    for name in (
                        "prepare_ms",
                        "operation_ms",
                        "retain_ms",
                        "publish_geometry_ms",
                    )
                )
                phase_runtime_profile["total_ms"] = float(result["runtime_ms"])
                phase_runtime_profile["accounted_ms"] = accounted_ms
                phase_runtime_profile["unattributed_ms"] = max(
                    0.0,
                    float(result["runtime_ms"]) - accounted_ms,
                )
                result["phase_runtime_profile"] = phase_runtime_profile
            self.trace.append(result)
            self.phase_count += 1

        if self.direct_commit_enabled and trailing_committed_without_rebuild:
            sizing_commit = self.direct_commit_records[-1]["sizing"]
            if self.owner.lane is None:
                from dreamplace.ops.buffer_insertion.buffering_config import (
                    build_buffering_config_from_params,
                )
                from dreamplace.ops.buffer_insertion.buffering_lane import (
                    BufferingOptimizationLane,
                )

                self.owner.lane = BufferingOptimizationLane(
                    build_buffering_config_from_params(
                        self.params, flow_kind="buffering"
                    ),
                    placedb=self.placedb,
                )
            noop_req = self.owner.lane.build_commit_request(
                None,
                iteration=int(iteration),
                reason="diff_opt_direct_commit_final_noop",
                commit_enabled=True,
                refresh_mode="python_carrier",
                rebuild_mode="full_rebuild",
            )
            self.owner.terminal_request = noop_req
            self.terminal_request = {
                "status": "request_ready",
                "iteration": int(iteration),
                "projected_buffer_count": 0,
                "direct_committed_count": int(self.owner.committed_buffer_count),
                "commit_request": noop_req.to_summary(),
            }
        else:
            # Publish sizing to the native database exactly once, after every
            # phase has restored/published its virtual state and before the final
            # buffer request is materialized.
            sizing_commit = self.owner.sync_native_sizing()
            self.terminal_request = self.owner.prepare_terminal_request(
                model,
                pos,
                iteration=int(iteration),
                refresh_geometry=False,
            )
        self.status = "request_ready"
        summary = {
            "artifact": "diff_opt",
            "artifact_version": 1,
            "status": self.status,
            "mode": mode,
            "parasitic_source": "placement",
            "phase_count": int(self.phase_count),
            "phases": list(self.trace),
            "sizing_commit": sizing_commit,
            "terminal_request": self.terminal_request,
            "direct_committed_buffer_count": int(
                getattr(self.owner, "committed_buffer_count", 0) or 0
            ),
            "direct_committed_buffers": list(
                getattr(self.owner, "committed_buffer_records", ()) or ()
            ),
            "refresh_counts": {
                "geometry_refresh_count": int(self.geometry_refresh_count),
                "timing_refresh_count": int(self.timing_refresh_count),
                "full_rebuild_count": int(self.full_rebuild_count),
                "topology_refresh_count": int(
                    getattr(self.owner, "topology_refresh_count", 0)
                ),
                "topology_refresh_reuse_count": int(
                    getattr(self.owner, "topology_refresh_reuse_count", 0)
                ),
            },
            "runtime_ms": (time.perf_counter() - started_at) * 1000.0,
        }
        result_dir = getattr(self.params, "result_dir", None)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            path = os.path.join(result_dir, "diff_opt_flow_summary.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
                stream.write("\n")
            self.summary_path = path
        return summary


def _build_diff_opt_runtime(engine, *, rebuild: bool = False):
    saved_flow_kind = getattr(engine.params, "flow_kind", None)
    saved_build_model_only = getattr(engine.params, "build_model_only", False)
    saved_relaxed_timing = bool(
        getattr(engine.params, "enable_relaxed_buffer_timing", False)
    )
    engine.params.flow_kind = "placement"
    engine.params.global_place_flag = 1
    engine.params.legalize_flag = 0
    engine.params.detailed_place_flag = 0
    engine.params.build_model_only = True
    if rebuild:
        engine.params.enable_relaxed_buffer_timing = False
        engine.placedb.cooptimization_geometry = None
        engine.placer = None
    try:
        engine.place()
    finally:
        engine.params.flow_kind = saved_flow_kind
        engine.params.build_model_only = saved_build_model_only
        if rebuild:
            engine.params.enable_relaxed_buffer_timing = saved_relaxed_timing
    model = getattr(engine.placer, "last_global_place_model", None)
    if model is None:
        raise RuntimeError("diff_opt did not construct a placement timing model")
    topology = engine.placer.op_collections.steiner_topo_op
    if not bool(getattr(topology, "topology_frozen", False)):
        engine.placer._refresh_live_timing_topology(
            engine.placer.data_collections.pos[0]
        )
        topology.freeze_topology()
    engine.params.flow_kind = "buffering"
    lane = engine.placer._build_buffering_lane(engine.params)
    return (
        model,
        engine.placer.data_collections.pos[0],
        engine.placer.data_collections,
        engine.placer.op_collections,
        engine.placer._refresh_live_timing_topology,
        lane,
    )


def run_diff_opt(engine):
    """Run the explicit placement-parasitic diff_opt tuple flow."""

    engine.setup_placedb()
    # diff_opt is a post-placement step.  Reject an unplaced/partially
    # initialized DEF before the timing topology path sees native garbage
    # coordinates (the ECC backend can otherwise expose large sentinel
    # values for cells that have no PLACED record).
    node_x = np.asarray(engine.placedb.node_x, dtype=np.float64)
    node_y = np.asarray(engine.placedb.node_y, dtype=np.float64)
    die_span = max(
        float(engine.placedb.xh - engine.placedb.xl),
        float(engine.placedb.yh - engine.placedb.yl),
        1.0,
    )
    coordinate_limit = max(1.0e6, 100.0 * die_span)
    invalid_x = (~np.isfinite(node_x)) | (np.abs(node_x) > coordinate_limit)
    invalid_y = (~np.isfinite(node_y)) | (np.abs(node_y) > coordinate_limit)
    if bool(invalid_x.any() or invalid_y.any()):
        bad_axis = "x" if bool(invalid_x.any()) else "y"
        bad_values = node_x[invalid_x] if bad_axis == "x" else node_y[invalid_y]
        raise ValueError(
            "diff_opt requires a placed input DEF with finite coordinates; "
            f"invalid {bad_axis} coordinate count={len(bad_values)} "
            f"max_abs={float(np.max(np.abs(bad_values))):.6g} "
            f"limit={coordinate_limit:.6g}"
        )
    # Build the timing model without moving coordinates.  The ECC builder
    # sets this stage to zero for joint diff_opt flows; keep the guard here
    # so a stale workspace config cannot silently run GP iterations.
    stages = list(getattr(engine.params, "global_place_stages", None) or [{}])
    if any(int(stage.get("iteration", 0) or 0) != 0 for stage in stages):
        raise ValueError("diff_opt flow requires zero GP iterations")
    model, pos, data, ops, refresh_topology, lane = _build_diff_opt_runtime(
        engine, rebuild=False
    )
    controller = DiffOptController(
        engine.params,
        engine.placedb,
        data,
        ops,
        refresh_topology,
        lane=lane,
        rebuild_runtime_fn=lambda: _build_diff_opt_runtime(engine, rebuild=True),
    )
    summary = controller.run(
        model,
        pos,
        flows=getattr(engine.params, "diff_optimization_flows", None),
        iteration=0,
    )
    engine.placer.diff_opt_controller = controller
    engine.placer.last_buffer_commit_request = controller.owner.terminal_request
    engine.placer.last_buffering_lane = controller.owner.lane
    request = controller.owner.terminal_request
    control_result = {
        "status": "pending_buffer_commit_request",
        "commit_request": request.to_summary(),
    }
    engine.placer.last_post_commit_control_result = control_result
    commit_result = engine._execute_pending_joint_buffer_commit(control_result)
    if isinstance(commit_result, dict) and commit_result.get("status") == "failed":
        engine.last_run_result = {
            "status": "failed",
            "flow_kind": "diff_opt",
            "diff_opt": summary,
            "commit": commit_result,
        }
        return engine.last_run_result
    commit_payload = dict((commit_result or {}).get("commit", {}) or {})
    committed_count = int(
        getattr(controller.owner, "committed_buffer_count", 0) or 0
    )
    if committed_count > 0:
        committed_records = list(
            getattr(controller.owner, "committed_buffer_records", ()) or ()
        )
        committed_names = [
            rec["inserted_buffer_name"]
            for rec in committed_records
            if isinstance(rec, dict) and rec.get("inserted_buffer_name")
        ]
        commit_payload = {
            **commit_payload,
            "status": "accepted",
            "requested_action_count": committed_count,
            "accepted_action_count": committed_count,
            "committed_buffer_instance_names": committed_names,
            "results": committed_records,
            "phase_commits": list(controller.direct_commit_records),
            "direct_commit_after_b1": True,
        }
        if isinstance(commit_result, dict):
            commit_result = {**commit_result, "commit": commit_payload}
    engine.placer.last_buffering_lane_summary = {
        "status": "completed",
        "flow_kind": "diff_opt",
        "parasitic_source": "placement",
        "inner_loop": summary,
        "commit": commit_payload,
    }
    engine.last_run_result = {
        "status": "ok",
        "flow_kind": "diff_opt",
        "parasitic_source": "placement",
        "phase_count": summary.get("phase_count", 0),
        "commit": commit_result,
        "diff_opt_summary_path": getattr(controller, "summary_path", None),
    }
    return engine.last_run_result
