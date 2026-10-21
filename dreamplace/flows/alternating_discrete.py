"""Discrete sizing/candidate-buffering alternating placement profile."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, is_dataclass, replace

from dreamplace.ops.buffer_insertion.candidate_density import install_candidate_density
from dreamplace.ops.routability.cooptimization_area import SizingWindowGeometry

from .inflation_s5b1 import InflationS5B1

PROFILE = "discrete_alternating_sizing_buffering_v1"


def _sha256_file(path):
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError):
        return None


def _effective_parameters(params, config):
    names = (
        "flow_kind",
        "joint_quality_profile",
        "placement_sizing_mode",
        "sizing_parameterization",
        "buffering_mode",
        "buffering_candidate_strategy",
        "buffering_candidate_drive_type_mode",
        "joint_alternating_outer_rounds",
        "joint_alternating_sizing_rounds",
        "joint_alternating_sizing_shared_budget",
        "joint_alternating_sizing_shared_budget_percent",
        "joint_alternating_buffering_rounds",
        "joint_alternating_buffering_selection_fraction",
        "joint_alternating_buffer_commit_policy",
        "timing_opt_overflow_milestones",
        "cell_padding_x",
        "target_density",
    )
    values = {}
    for name in names:
        value = getattr(params, name, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            values[name] = value
        elif isinstance(value, (list, tuple)):
            values[name] = list(value)
        else:
            values[name] = str(value)
    values.update(
        {
            "profile": PROFILE,
            "placement_optimizer_scope": "placement_coordinates_only",
            "sizing_update_mode": "discrete_window",
            "buffering_update_mode": "discrete_window",
            "resolved_outer_rounds": int(config.outer_rounds),
            "resolved_sizing_rounds": int(config.sizing_rounds),
            "resolved_buffering_rounds": int(config.buffering_rounds),
            "resolved_sizing_budget_percent": float(config.sizing_budget_percent),
            "resolved_buffering_selection_fraction": float(config.buffering_selection_fraction),
        }
    )
    return values


def _input_hashes(params):
    design_inputs = getattr(params, "design_inputs", {}) or {}
    if not isinstance(design_inputs, dict):
        design_inputs = {}
    aliases = {
        "def": ("def", "def_input"),
        "verilog": ("verilog", "verilog_input"),
        "sdc": ("sdc", "sdc_input"),
    }
    hashes = {}
    paths = {}
    for label, candidates in aliases.items():
        path = next(
            (design_inputs.get(key) for key in candidates if design_inputs.get(key)),
            getattr(params, candidates[-1], None),
        )
        paths[label] = path
        hashes[label] = _sha256_file(path)
    return {"paths": paths, "sha256": hashes}


def _output_hashes(finalize_result):
    terminal = {}
    if isinstance(finalize_result, dict):
        terminal = next(
            (
                value
                for key, value in finalize_result.items()
                if str(key).endswith("_terminal") and isinstance(value, dict)
            ),
            {},
        )
    paths = {
        "def": terminal.get("final_def"),
        "verilog": terminal.get("final_verilog"),
    }
    return {"paths": paths, "sha256": {key: _sha256_file(path) for key, path in paths.items()}}


class AlternatingSizingOwner(InflationS5B1):
    """Refresh accepted virtual actions whenever S changes masters or pins."""

    def __init__(
        self, params, placedb, data, ops, refresh_topology, *, execute_window, window_limit
    ):
        super().__init__(
            params,
            placedb,
            data,
            ops,
            refresh_topology,
            window_limit=window_limit,
            refresh_candidates_after_sizing=True,
        )
        self.execute_window = execute_window

    def run(self, model, pos, *, iteration):
        return self.execute_window(model, pos, iteration=iteration)

    def prepare(
        self,
        model,
        pos,
        *,
        active_only=True,
        refresh_records_all=False,
    ):
        return super().prepare(
            model,
            pos,
            active_only=active_only,
            refresh_records_all=refresh_records_all,
        )


@dataclass(frozen=True)
class AlternatingDiscreteConfig:
    outer_rounds: int = 5
    sizing_rounds: int = 5
    buffering_rounds: int = 1
    sizing_budget_percent: float = 1.0
    buffering_selection_fraction: float = 0.001
    commit_policy: str = "final_only"

    @classmethod
    def from_params(cls, params):
        return cls(
            outer_rounds=int(getattr(params, "joint_alternating_outer_rounds", 5)),
            sizing_rounds=int(getattr(params, "joint_alternating_sizing_rounds", 5)),
            buffering_rounds=int(getattr(params, "joint_alternating_buffering_rounds", 1)),
            sizing_budget_percent=float(
                getattr(params, "joint_alternating_sizing_shared_budget_percent", 1.0)
            ),
            buffering_selection_fraction=float(
                getattr(params, "joint_alternating_buffering_selection_fraction", 0.001)
            ),
            commit_policy=str(
                getattr(params, "joint_alternating_buffer_commit_policy", "final_only")
            ),
        )


@dataclass(frozen=True)
class AlternatingRoundRecord:
    outer_round: int
    iteration: int
    trigger: str
    sizing: dict
    buffering: dict
    runtime_ms: float

    def to_dict(self):
        return {
            "outer_round": int(self.outer_round),
            "iteration": int(self.iteration),
            "trigger": str(self.trigger),
            "sizing": self.sizing,
            "buffering": self.buffering,
            "runtime_ms": float(self.runtime_ms),
        }


class AlternatingDiscreteController:
    """Own one persistent virtual candidate lane across alternating rounds.

    The ordinary GP optimizer remains the owner of placement coordinates.  This
    controller runs discrete S/B transactions through the existing overflow or
    inflation window owner and leaves buffer materialization to the final bridge.
    """

    def __init__(self, params, placedb, data, ops, refresh_topology, *, lane=None):
        self.params = params
        self.placedb = placedb
        self.data = data
        self.ops = ops
        self.config = AlternatingDiscreteConfig.from_params(params)
        self.owner = AlternatingSizingOwner(
            params,
            placedb,
            data,
            ops,
            refresh_topology,
            execute_window=self.run_outer_round,
            window_limit=self.config.outer_rounds,
        )
        self.owner.lane = lane
        if lane is not None:
            state = lane._current_state()
            if state is not None:
                state.timing_active_only = True
                state.activation_param.requires_grad_(False)
        self.outer_rounds = self.config.outer_rounds
        self.sizing_rounds = self.config.sizing_rounds
        self.buffering_rounds = self.config.buffering_rounds
        self.sizing_budget_percent = self.config.sizing_budget_percent
        self.buffering_fraction = self.config.buffering_selection_fraction
        self.placement_step_count = 0
        self.completed_rounds = 0
        self.trace = []
        self.last_result = None
        self.last_finalize_result = None
        self._candidate_payload_topology_generation = None

    @property
    def active(self):
        return self.completed_rounds < self.outer_rounds and self.last_finalize_result is None

    @contextmanager
    def _profile_params(self):
        names = {
            "timing_opt_sizing_rounds": self.sizing_rounds,
            "discrete_gradient_topk_shared_budget_percent": self.sizing_budget_percent,
        }
        previous = {name: getattr(self.params, name, None) for name in names}
        for name, value in names.items():
            setattr(self.params, name, value)
        self.params.enable_relaxed_buffer_timing = True
        lane = self.owner.lane
        previous_config = None
        if lane is not None:
            previous_config = lane.config
            # BufferingConfig is frozen so the alternating controller must
            # replace the phase configuration rather than mutate fields in
            # place.  Restoring the original object also preserves all
            # unrelated buffering defaults and explicit provenance.
            if is_dataclass(previous_config):
                lane.config = replace(
                    previous_config,
                    continuous_steps=self.buffering_rounds,
                    route_b_selection_fraction=self.buffering_fraction,
                )
            else:
                lane.config = copy(previous_config)
                lane.config.continuous_steps = self.buffering_rounds
                lane.config.route_b_selection_fraction = self.buffering_fraction
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(self.params, name, value)
            if self.owner.lane is not None:
                self.params.enable_relaxed_buffer_timing = True
            if lane is not None and previous_config is not None:
                lane.config = previous_config

    def _ensure_lane(self, model, pos):
        self.owner.prepare(model, pos)
        if self.owner.lane is None or self.owner.lane._current_state() is None:
            raise RuntimeError("alternating discrete profile failed to prepare candidate lane")
        self.params.enable_relaxed_buffer_timing = True
        state = self.owner.lane._current_state()
        state.timing_active_only = True
        state.activation_param.requires_grad_(False)

    def run_outer_round(self, model, pos, *, iteration, trigger="timing_opt_window"):
        if self.completed_rounds >= self.outer_rounds:
            return None
        started_at = time.perf_counter()
        self._ensure_lane(model, pos)
        self.owner.lane.timing_loss = model._timing_loss
        with (
            self._profile_params(),
            self.owner.electrical_window(),
            self.owner._sizing_coefficients(model) as coefficients,
        ):
            sizing = run_sizing_phase(self, model, pos)
            sizing["coefficients"] = dict(coefficients)
            self.owner.prepare(model, pos, active_only=False)
            buffering = run_buffering_phase(self, model, pos)
        self.owner.retain_buffered_nets(model)
        install_candidate_density(model, self.owner.lane._current_state())
        self.owner.publish_geometry(model)
        self._candidate_payload_topology_generation = None
        self.completed_rounds += 1
        result = AlternatingRoundRecord(
            outer_round=self.completed_rounds,
            iteration=iteration,
            trigger=trigger,
            sizing=sizing,
            buffering=buffering,
            runtime_ms=(time.perf_counter() - started_at) * 1000.0,
        ).to_dict()
        lane_summary = self.owner.lane.summarize()
        result["candidate_state"] = {
            key: lane_summary.get(key)
            for key in (
                "candidate_count",
                "affected_net_count",
                "state_kind",
                "candidate_drive_type_table",
                "candidate_geometry_refresh",
            )
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
                )
            }
        self.trace.append(result)
        self.last_result = result
        return result

    def after_placement_step(self, model, pos, *, iteration):
        self.placement_step_count += 1

    def before_step(self, model, pos, optimizer, metric, eval_ops, *, iteration, stage):
        if not self.active:
            return
        previous_count = self.completed_rounds
        self.owner.before_step(
            model,
            pos,
            optimizer,
            metric,
            eval_ops,
            iteration=iteration,
            stage=stage,
        )
        if self.completed_rounds != previous_count:
            self._write_summary(status="completed" if not self.active else "active")

    def before_inflation(self, model, pos, *, iteration, stage, optimizer=None):
        if not self.active:
            return None
        result = self.owner.before_inflation(
            model,
            pos,
            iteration=iteration,
            stage=stage,
            optimizer=optimizer,
        )
        if result is not None:
            self._write_summary(status="completed" if not self.active else "active")
        return result

    def publish_geometry(self, model):
        return self.owner.publish_geometry(model)

    def refresh_live_geometry(self, model, pos, *, iteration):
        """Refresh active virtual candidates after the live topology forward."""

        lane = self.owner.lane
        if lane is None or self.completed_rounds == 0:
            return None
        state = lane._current_state()
        if state is None or not hasattr(state, "activation_param"):
            return None
        generation = getattr(self.ops.steiner_topo_op, "topology_generation", None)
        if (
            generation is not None
            and generation == self._candidate_payload_topology_generation
            and bool(getattr(self.params, "buffering_segment_live_geometry", False))
        ):
            install_candidate_density(model, state)
            return {"status": "ok", "reason": "live_rc_bound_per_objective"}
        result = lane.refresh_candidate_geometry(
            model=model,
            params=self.params,
            reason=f"placement_geometry_iteration_{int(iteration)}",
            active_only=True,
        )
        self.last_alternating_geometry_refresh = dict(result)
        self._candidate_payload_topology_generation = generation
        self.ops.steiner_topo_op.frozen_nets.candidate_state = lane._current_state()
        install_candidate_density(model, lane._current_state())
        return result

    def finalize(self, model, pos, *, iteration):
        result = materialize_final_buffering(self, model, pos, iteration=iteration)
        self.last_finalize_result = dict(result)
        self._write_summary(status="request_ready")
        return result

    def _write_summary(self, *, status):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        os.makedirs(result_dir, exist_ok=True)
        design_attr = getattr(self.params, "design_name", "design")
        design_name = design_attr() if callable(design_attr) else str(design_attr)
        path = os.path.join(
            result_dir,
            f"{design_name}_discrete_alternating_sizing_buffering.json",
        )
        payload = {
            "artifact": "discrete_alternating_sizing_buffering",
            "artifact_version": 1,
            "profile": PROFILE,
            "status": str(status),
            "outer_rounds_requested": int(self.outer_rounds),
            "outer_rounds_completed": int(self.completed_rounds),
            "trigger_mode": self.owner.schedule.mode,
            "pending_overflow_milestones": list(self.owner.schedule.pending_milestones),
            "untriggered_window_count": max(self.outer_rounds - self.completed_rounds, 0),
            "sizing_rounds": int(self.sizing_rounds),
            "buffering_rounds": int(self.buffering_rounds),
            "sizing_budget_percent": float(self.sizing_budget_percent),
            "buffering_selection_fraction": float(self.buffering_fraction),
            "sizing_attempt_count": sum(
                len(record.get("sizing", {}).get("rounds", ())) for record in self.trace
            ),
            "buffering_attempt_count": sum(
                int(record.get("buffering", {}).get("scheduled", 0) or 0) for record in self.trace
            ),
            "virtual_buffer_count": (
                self.trace[-1].get("buffering", {}).get("cumulative_count", 0) if self.trace else 0
            ),
            "effective_parameters": _effective_parameters(self.params, self.config),
            "input_artifacts": _input_hashes(self.params),
            "output_artifacts": _output_hashes(self.last_finalize_result),
            "rounds": self.trace,
            "finalize": self.last_finalize_result,
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        self.summary_path = path
        return path

    def summarize(self):
        return {
            "profile": PROFILE,
            "status": "active" if self.active else "completed",
            "outer_rounds_requested": int(self.outer_rounds),
            "outer_rounds_completed": int(self.completed_rounds),
            "sizing_rounds": int(self.sizing_rounds),
            "buffering_rounds": int(self.buffering_rounds),
            "sizing_budget_percent": float(self.sizing_budget_percent),
            "buffering_selection_fraction": float(self.buffering_fraction),
            "sizing_attempt_count": sum(
                len(record.get("sizing", {}).get("rounds", ())) for record in self.trace
            ),
            "buffering_attempt_count": sum(
                int(record.get("buffering", {}).get("scheduled", 0) or 0) for record in self.trace
            ),
            "placement_step_count": int(self.placement_step_count),
            "summary_path": getattr(self, "summary_path", None),
        }


def run_sizing_phase(controller, model, pos):
    geometry = SizingWindowGeometry(controller.data, controller.placedb, pos)
    return controller.owner.run_sizing(
        model,
        pos,
        geometry,
        controller.owner.capacity,
    )


def run_buffering_phase(controller, model, pos):
    activation = controller.owner.lane._current_state().activation_param
    requires_grad = activation.requires_grad
    activation.requires_grad_(True)
    try:
        return controller.owner.run_buffering(
            model,
            pos,
            controller.owner.capacity,
        )
    finally:
        activation.requires_grad_(requires_grad)


def materialize_final_buffering(controller, model, pos, *, iteration):
    return controller.owner.prepare_terminal_request(
        model,
        pos,
        iteration=iteration,
    )


def run_alternating_discrete_flow(controller, model, pos, *, iteration, finalize=True):
    while controller.active:
        controller.run_outer_round(
            model,
            pos,
            iteration=iteration,
            trigger="standalone_flow",
        )
    if not finalize:
        controller._write_summary(status="completed")
        return controller.summarize()
    result = materialize_final_buffering(controller, model, pos, iteration=iteration)
    controller.last_finalize_result = dict(result)
    controller._write_summary(status="request_ready")
    return result


__all__ = [
    "AlternatingDiscreteConfig",
    "AlternatingDiscreteController",
    "AlternatingRoundRecord",
    "PROFILE",
    "materialize_final_buffering",
    "run_alternating_discrete_flow",
    "run_buffering_phase",
    "run_sizing_phase",
]
