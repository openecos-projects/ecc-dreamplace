"""Standalone placement-parasitic diff optimization controller."""

import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.flows.timing_opt_config import TIMING_OPT_COEFFICIENT_FIELDS
from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest

from .diff_opt_spec import DiffFlowSpec, parse_diff_opt_flows


class DiffOptController:
    """Run an ordered diff_opt tuple list on a fixed placement.

    The controller owns virtual sizing/candidate state for one run.  Native
    sizing and buffer materialization are exposed as one terminal request only
    after all phases have completed.
    """

    def __init__(self, params, placedb, data, ops, refresh_topology, *, lane=None):
        self.params = params
        self.placedb = placedb
        self.data = data
        self.ops = ops
        self.owner = DiscreteTimingOwner(
            params,
            placedb,
            data,
            ops,
            refresh_topology,
            refresh_candidates_after_sizing=True,
        )
        self.owner.lane = lane
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
        try:
            for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
                setattr(model, field, float(getattr(coefficients, name)))
            model.timing_grad_balance_weight = 1.0
            yield
        finally:
            for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
                setattr(model, field, original[name])
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
        # Every buffering phase must rebuild the complete candidate domain.
        # ``active_only=True`` is appropriate while ordinary GP follows only
        # retained virtual buffers, but it would permanently exclude nets
        # that were not selected by an earlier BF phase.
        self.owner.prepare(model, pos, active_only=False)
        lane = self.owner.lane
        if lane is None or lane._current_state() is None:
            raise RuntimeError("diff_opt failed to prepare its candidate lane")
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
        for phase_index, spec in enumerate(specs):
            phase_started_at = time.perf_counter()
            input_state_digest = self._state_digest()
            with (
                self.owner.electrical_window(),
                self._phase_config(spec),
                self._phase_coefficients(model, spec.coefficients),
            ):
                if spec.action == "sz":
                    # S only needs the buffers already retained in the design;
                    # the next BF phase rebuilds the complete action domain.
                    self.owner.prepare(model, pos, active_only=True)
                    from dreamplace.ops.routability.cooptimization_area import SizingWindowGeometry

                    geometry = SizingWindowGeometry(self.data, self.placedb, pos)
                    result = self.owner.run_sizing(model, pos, geometry, self.owner.capacity)
                else:
                    self._ensure_lane(model, pos)
                    result = self.owner.run_buffering(model, pos, self.owner.capacity)
            # A final buffering phase is already consumed by the terminal
            # projection below.  Freezing and immediately repacking its
            # candidates adds no state needed by a later phase and can make
            # the native topology packer observe a partially frozen tree.
            # Retain accepted nets only when another explicit phase follows.
            if spec.action == "bf" and phase_index + 1 < len(specs):
                self.owner.retain_buffered_nets(model)
            geometry_refresh = self.owner.publish_geometry(model)
            self.geometry_refresh_count += 1
            self.timing_refresh_count += 1
            result = dict(result or {})
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
                    "runtime_ms": (time.perf_counter() - phase_started_at) * 1000.0,
                }
            )
            self.trace.append(result)
            self.phase_count += 1

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
            "refresh_counts": {
                "geometry_refresh_count": int(self.geometry_refresh_count),
                "timing_refresh_count": int(self.timing_refresh_count),
                "full_rebuild_count": 0,
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
    # ``apply_diff_opt_defaults`` keeps the original mode-specific
    # ``flow_kind`` (``buffering`` for buffering-only and the historical
    # joint profile) so timing metadata remains active at the native
    # boundary.  The zero-iteration model construction below must still
    # go through the ordinary placement path: ``flow_kind=buffering``
    # makes ``NonLinearPlace.__call__`` enter its legacy buffering inner
    # loop before the explicit tuple controller gets a chance to run.
    # Build only the timing model here, then restore the mode for the
    # standalone controller and terminal commit.
    saved_flow_kind = getattr(engine.params, "flow_kind", None)
    engine.params.flow_kind = "placement"
    engine.params.global_place_flag = 1
    engine.params.legalize_flag = 0
    engine.params.detailed_place_flag = 0
    try:
        engine.place()
    finally:
        engine.params.flow_kind = saved_flow_kind
    model = getattr(engine.placer, "last_global_place_model", None)
    if model is None:
        raise RuntimeError("diff_opt did not construct a placement timing model")
    # Candidate payloads and live geometry must use one topology identity
    # for the whole diff_opt transaction.  Rebuilding FLUTE after a sizing
    # action can change the vertex count/order by one or more nodes, which
    # would bind a frozen candidate row to a different coordinate domain.
    # Freeze the placement snapshot before constructing the candidate lane;
    # subsequent forwards still update coordinates through the frozen
    # Steiner relations while preserving node IDs and net slices.
    topology = engine.placer.op_collections.steiner_topo_op
    if not bool(getattr(topology, "topology_frozen", False)):
        engine.placer._refresh_live_timing_topology(engine.placer.data_collections.pos[0])
        topology.freeze_topology()
    # The model was constructed in the placement coordinate flow; the
    # explicit controller needs the joint lane contract while it prepares
    # typed candidates, without changing the placement-parasitic source.
    # The candidate lane's standalone validation intentionally requires
    # ``flow_kind=buffering``.  This is only the controller/runtime mode;
    # the model above was already built on the ordinary placement path,
    # and the terminal finalizer switches back to placement after commit.
    engine.params.flow_kind = "buffering"
    lane = engine.placer._build_buffering_lane(engine.params)
    controller = DiffOptController(
        engine.params,
        engine.placedb,
        engine.placer.data_collections,
        engine.placer.op_collections,
        engine.placer._refresh_live_timing_topology,
        lane=lane,
    )
    summary = controller.run(
        model,
        engine.placer.data_collections.pos[0],
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
    engine.placer.last_buffering_lane_summary = {
        "status": "completed",
        "flow_kind": "diff_opt",
        "parasitic_source": "placement",
        "inner_loop": summary,
        "commit": dict((commit_result or {}).get("commit", {}) or {}),
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


__all__ = ["DiffOptController", "run_diff_opt"]
