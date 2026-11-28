"""Experimental mid-placement rebuffer repair session, tracker, and diagnostics."""


import math
from dataclasses import asdict, dataclass
from dreamplace.ops.placeio_ecc.export_options import resolve_flag
from dreamplace.flows.gp.continuation import (
    MidplaceBoundedStop,
    MidplaceRestart,
    PlacementCursor,
    remaining_steps,
    stage_budgets,
)
import hashlib
import json
import numpy as np
import torch
from dreamplace.EvalMetrics import EvalMetrics
from dreamplace.ops.rebuffer.pydb_adapter import _buffer_metadata
from dreamplace.flows.gp.continuation import (
    capture_fillers,
    committed_seed,
    fixed_positions,
    names,
)
import logging
import resource
from pathlib import Path
from dreamplace.flows.gp.continuation import capture_warm_schedule_state
from dreamplace.flows.gp.continuation import MidplaceSchedule
from dreamplace.flows.baselines.repair_buffering import run_repair_buffering_stage
from dreamplace.ops.placeio_ecc.repair_profile import RepairProfile
from dreamplace.ops.routability.gpugr_context import write_back_movable_lpos


@dataclass(frozen=True)
class MidplaceConfig:
    overflow: float
    mode: str
    resume_policy: str
    continue_placement: bool
    max_passes: int
    bounded_steps: int
    artifact: str
    max_buffer_fraction: float = 0.001

    @classmethod
    def from_params(cls, params):
        config = cls(
            overflow=float(getattr(params, "repair_buffering_midplace_overflow", 0.4)),
            mode=str(getattr(params, "repair_buffering_midplace_mode", "repair_design")),
            resume_policy=str(getattr(params, "repair_buffering_midplace_resume_policy", "cold")),
            continue_placement=resolve_flag(params, "repair_buffering_midplace_continue", default=1),
            max_passes=int(getattr(params, "repair_buffering_midplace_max_passes", 20)),
            bounded_steps=int(getattr(params, "repair_buffering_midplace_bounded_steps", 0)),
            artifact=str(getattr(params, "repair_buffering_midplace_artifact", "midplace_handoff.json")),
            max_buffer_fraction=float(
                getattr(params, "repair_buffering_midplace_max_buffer_fraction", 0.001)
            ),
        )
        if not math.isfinite(config.overflow) or config.overflow <= 0:
            raise ValueError("midplace overflow must be finite and positive")
        if (
            not math.isfinite(config.max_buffer_fraction)
            or not 0.0 <= config.max_buffer_fraction <= 1.0
        ):
            raise ValueError("midplace requires 0.0 <= max_buffer_fraction <= 1.0")
        if config.mode not in {"repair_design", "repair_design_then_timing"}:
            raise ValueError("midplace supports repair_design or repair_design_then_timing")
        if config.resume_policy not in {"cold", "warm_schedule"}:
            raise ValueError("midplace resume policy must be cold or warm_schedule")
        if config.bounded_steps < 0 or config.max_passes < 1:
            raise ValueError("midplace requires bounded_steps >= 0 and max_passes >= 1")
        if not resolve_flag(params, "repair_buffering_midplace_once", default=1):
            raise ValueError("midplace currently requires once per placement session")
        if (
            getattr(params, "repair_buffering_midplace_initial_policy", "first_stable_step")
            != "first_stable_step"
        ):
            raise ValueError("midplace initial policy must be first_stable_step")
        if getattr(params, "repair_buffering_midplace_pending_policy", "defer") != "defer":
            raise ValueError("midplace pending policy must be defer")
        if config.mode == "repair_design_then_timing" and not resolve_flag(
            params, "repair_buffering_midplace_timing_enable"
        ):
            raise ValueError("midplace combined mode requires explicit timing_enable")
        if resolve_flag(params, "repair_buffering_enable"):
            raise ValueError("midplace and terminal repair triggers are mutually exclusive")
        if not params.with_sta or str(params.place_io_engine) != "ecc":
            raise ValueError("midplace requires Python timing with the ECC placeio backend")
        if not params.global_place_flag or params.macro_place_flag:
            raise ValueError("midplace requires cell global placement")
        if str(getattr(params, "placement_sizing_mode", "place_only")) != "place_only":
            raise ValueError(
                "midplace currently requires the placement owner, without a joint ECO owner"
            )
        if resolve_flag(params, "openroad_handoff_enable"):
            raise ValueError("native midplace and external OpenROAD handoff are mutually exclusive")
        return config


@dataclass(frozen=True)
class OverflowObservation:
    overflow: float
    topology_generation: int
    cursor: PlacementCursor
    completed_effective_steps: int
    total_effective_steps: int


@dataclass(frozen=True)
class MidplaceRequest:
    request_id: str
    threshold: float
    previous_overflow: float | None
    crossing: OverflowObservation
    execution: OverflowObservation

    def as_dict(self):
        return asdict(self)


class OverflowTracker:
    """Latch the first crossing, including a first stable step below threshold.

    A pending placement transaction may change both overflow and topology. Only
    observations are retained while waiting: planning uses the execution input.
    Consumption belongs to this session, never to a PyDB or placer generation.
    """

    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.previous_overflow = None
        self.pending = None
        self.consumed = False

    def observe_initial(self, observation):
        """Latch an initially low metric, but execute only after a stable step."""
        if not math.isfinite(observation.overflow):
            raise ValueError("midplace initial overflow is not finite")
        self.previous_overflow = observation.overflow
        if observation.overflow <= self.threshold:
            self.pending = (None, observation)

    def observe(self, observation, *, pending_transaction=False):
        overflow = observation.overflow
        if not math.isfinite(overflow):
            raise ValueError("midplace overflow observation is not finite")
        if self.consumed:
            return None
        if self.pending is None and overflow <= self.threshold:
            self.pending = (self.previous_overflow, observation)
        self.previous_overflow = overflow
        if self.pending is None or pending_transaction:
            return None
        previous, crossing = self.pending
        self.consumed = True
        return MidplaceRequest("midplace-1", self.threshold, previous, crossing, observation)


def plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"unsupported handoff artifact value: {type(value).__name__}")


def topology_counts(placedb):
    pydb = placedb.rawdb.pydb
    buffer_mask, source = _buffer_metadata(pydb, int(pydb.num_nodes))
    if buffer_mask is None:
        raise RuntimeError("midplace requires qualified per-node buffer metadata")
    return {
        "nodes": int(pydb.num_nodes),
        "pins": len(pydb.pin_names),
        "nets": len(pydb.net_names),
        "cell_arcs": len(pydb.flat_inst_arcs_by_level),
        "movable": int(placedb.num_movable_nodes),
        "movable_density_area": float(
            np.dot(
                placedb.node_size_x[: placedb.num_movable_nodes],
                placedb.node_size_y[: placedb.num_movable_nodes],
            )
        ),
        "buffers": int(np.count_nonzero(buffer_mask)),
        "buffer_metadata_source": source,
        "topology_generation": int(getattr(placedb, "topology_generation", 0)),
        "refresh_generation": int(placedb.rawdb.refresh_generation),
    }


def action_identity(stage):
    payload = {
        name: {
            key: result["plan"].get(key, []) for key in ("actions", "resizes", "accepted_actions")
        }
        for name, result in stage["stages"].items()
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=plain).encode()).hexdigest()


def validate_position_seed(placer, placedb, seed):
    count = int(placedb.num_movable_nodes)
    if names(placedb.node_names[:count]) != seed["movable_node_names"]:
        raise RuntimeError("committed position seed has a different movable namespace")
    actual = placer.pos[0].detach().cpu().numpy()
    expected = np.concatenate((seed["movable_x"], seed["movable_y"]))
    observed = np.concatenate(
        (actual[:count], actual[placedb.num_nodes : placedb.num_nodes + count])
    )
    if not np.array_equal(expected.astype(observed.dtype), observed):
        raise RuntimeError("fresh placer did not preserve the committed position seed")
    if fixed_positions(placedb, actual) != seed["fixed_positions"]:
        raise RuntimeError("fresh placer did not preserve a fixed position")
    return {
        "movable_positions_checked": count,
        "fixed_positions_checked": len(seed["fixed_positions"]),
        "max_seed_displacement": 0.0,
    }


def runtime_topology(placer, placedb, previous_names):
    """Validate CSR identity and insertion participation, without running legalize."""
    data, pydb = placer.data_collections, placedb.rawdb.pydb
    if placer.pos[0].numel() != 2 * placedb.num_nodes:
        raise RuntimeError("fresh placer position size does not match the committed topology")
    if not bool(torch.isfinite(placer.pos[0]).all()):
        raise RuntimeError("fresh placer has non-finite committed positions")
    for key in (
        "pin2node_map",
        "pin2net_map",
        "flat_node2pin_map",
        "flat_node2pin_start_map",
        "flat_net2pin_map",
        "flat_net2pin_start_map",
        "node2fence_region_map",
    ):
        actual = getattr(data, key).detach().cpu().numpy()
        if not np.array_equal(actual, getattr(placedb, key)):
            raise RuntimeError(f"fresh placer has stale {key}")
    all_names = names(placedb.node_names)
    native_ids = {name: i for i, name in enumerate(names(pydb.node_names))}
    buffer_mask, _ = _buffer_metadata(pydb, int(pydb.num_nodes))
    if buffer_mask is None:
        raise RuntimeError("committed topology lost per-node buffer metadata")
    records = []
    for node_id, name in enumerate(all_names):
        # Synthetic blockages are renumbered when buffers are inserted; they
        # are obstacles, never inserted timing cells.
        if name in previous_names or name.startswith("blockage"):
            continue
        raw_id = native_ids[name]
        if not bool(buffer_mask[raw_id]) or node_id >= placedb.num_movable_nodes:
            raise RuntimeError(f"inserted node is not a movable buffer: {name}")
        original = int(placedb.node2orig_node_map[node_id])
        region = int(placedb.node2fence_region_map[node_id])
        if original < 0 or original != int(pydb.node2orig_node_map[raw_id]):
            raise RuntimeError(f"inserted buffer has stale node2orig_node_map: {name}")
        if region < 0 or region != int(pydb.node2fence_region_map[raw_id]):
            raise RuntimeError(f"inserted buffer has stale node2fence_region_map: {name}")
        pin_start = int(placedb.flat_node2pin_start_map[node_id])
        pin_end = int(placedb.flat_node2pin_start_map[node_id + 1])
        if pin_end - pin_start < 2 or float(data.node_areas[node_id]) <= 0:
            raise RuntimeError(f"inserted buffer is missing pins or density area: {name}")
        records.append(
            {
                "name": name,
                "node_id": node_id,
                "native_node_id": raw_id,
                "node2orig_node_map": original,
                "fence_region": region if region < len(placedb.regions) else None,
                "movable": True,
                "node_is_buffer": True,
                "pin_count": pin_end - pin_start,
                "size": [float(data.node_size_x[node_id]), float(data.node_size_y[node_id])],
                "area": float(data.node_areas[node_id]),
                "seed_position": [
                    float(placer.pos[0][node_id]),
                    float(placer.pos[0][placedb.num_nodes + node_id]),
                ],
            }
        )
    return {
        "csr_matches_committed_placedb": True,
        "new_buffers": records,
        "physical_nodes": int(placedb.num_physical_nodes),
        "movable_nodes": int(placedb.num_movable_nodes),
        "timing_consumer_validation": "pending_first_optimizer_step",
        "legalization_input_nodes": int(data.node_size_x.numel()),
    }


def timing_validation(stage):
    """Report measured evaluation modes; fresh commit TP is a separate check."""
    timing = stage["stages"].get("repair_timing")
    if timing is None:
        return {"status": "not_requested", "incremental_full_fallback": None}
    plan = timing["plan"]
    native = plan.get("native_timing_graph_statistics", {})
    full_fallbacks = native.get("full_fallback_count")
    persistent = plan.get("persistent_timing_statistics", {})
    python_fallbacks = int(persistent.get("full_overlay_fallbacks", 0))
    fallback = bool(full_fallbacks or python_fallbacks) if full_fallbacks is not None else None
    return {
        "status": timing["status"],
        "native_candidate_planner": bool(plan.get("native_candidate_planner", False)),
        "evaluation_counts": native.get("evaluation_counts", {}),
        "python_materialization_count": native.get("python_materialization_count"),
        "python_full_overlay_fallbacks": python_fallbacks,
        "native_full_fallbacks": full_fallbacks,
        "incremental_full_fallback": fallback,
        "fresh_commit_verification": timing.get("commit", {}).get("timing_evaluator"),
    }


def post_step_overflow(placer, placedb, eval_ops, iteration):
    metric = EvalMetrics(iteration)
    with torch.no_grad():
        metric.evaluate(
            placedb, {"overflow": eval_ops["overflow"]}, placer.pos[0], placer.data_collections
        )
    return float(metric.overflow.detach().max().cpu())


def step_record(placer, placedb, model, metric, overflow, step, elapsed_ms, buffers):
    result = {
        "step": step,
        "raw_iteration": int(metric.iteration),
        "post_step_overflow": overflow,
        "wall_ms": elapsed_ms,
    }
    for key in (
        "objective",
        "hpwl",
        "wns",
        "tns",
        "timing_objective",
        "slew_violation",
        "cap_violation",
        "l_shape_cost",
        "l_shape_weighted_cost",
    ):
        value = getattr(metric, key, None)
        if value is not None:
            array = np.asarray(
                plain(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
            )
            if not np.isfinite(array).all():
                raise RuntimeError(f"post-handoff step has non-finite {key}")
            result[key] = array.tolist()
    result["timing_objective_enabled"] = bool(model.use_timing_obj)
    result["routability_objective_enabled"] = bool(model.use_l_shape_routability)
    if result.get("objective") is None:
        raise RuntimeError("post-handoff step has no executed objective")
    if model.use_timing_obj:
        terms = model.last_timing_objective_terms
        loss = terms.get("total_loss")
        if loss is None or not np.isfinite(loss):
            raise RuntimeError("enabled timing objective has no finite executed loss")
        result["timing_objective_state"] = terms
        if step == 1:
            timing = model.op_collections.timing_propagation_op
            pin_count = len(placedb.pin2net_map)
            if (
                timing is None
                or timing.num_pins < pin_count
                or not np.array_equal(
                    timing.pin_net[:pin_count].detach().cpu().numpy(), placedb.pin2net_map
                )
            ):
                raise RuntimeError("post-handoff timing consumer has stale physical pin mapping")
            result["timing_physical_pins_checked"] = pin_count
    if model.use_l_shape_routability:
        if result.get("l_shape_cost") is None:
            raise RuntimeError("enabled routability objective has no executed cost")
        if step == 1:
            route_db = model.l_shape_routability_op.placedb
            if not np.array_equal(route_db.pin2node_map, placedb.pin2node_map):
                raise RuntimeError("post-handoff routability consumer has stale pin mapping")
            result["routability_physical_pins_checked"] = len(placedb.pin2node_map)
    owner = getattr(placer, "inflation_s5b1", None)
    if owner is not None:
        result["timing_window_state"] = {
            "completed": owner.schedule.window_count,
            "pending_milestones": list(owner.schedule.pending_milestones),
            "virtual_lane_active": owner.lane is not None,
        }
    inflation = getattr(model, "inflation_state", None)
    if inflation is not None:
        result["inflation_rounds_completed"] = int(inflation.num_area_adjust)
    result["buffer_positions"] = [
        {
            "name": entry["name"],
            "position": [
                float(placer.pos[0][entry["node_id"]]),
                float(placer.pos[0][placedb.num_nodes + entry["node_id"]]),
            ],
            "moved": any(
                float(placer.pos[0][offset + entry["node_id"]]) != seed
                for offset, seed in zip((0, placedb.num_nodes), entry["seed_position"], strict=True)
            ),
            "inside_placement_boundary": (
                placedb.xl
                <= float(placer.pos[0][entry["node_id"]])
                <= placedb.xh - float(placer.data_collections.node_size_x[entry["node_id"]])
                and placedb.yl
                <= float(placer.pos[0][placedb.num_nodes + entry["node_id"]])
                <= placedb.yh - float(placer.data_collections.node_size_y[entry["node_id"]])
            ),
        }
        for entry in buffers
    ]
    if not bool(torch.isfinite(placer.pos[0]).all()):
        raise RuntimeError("post-handoff optimizer produced non-finite positions")
    expected_fixed = fixed_positions(placedb)
    if fixed_positions(placedb, placer.pos[0].detach().cpu().numpy()) != expected_fixed:
        raise RuntimeError("post-handoff optimizer moved a fixed position")
    result["fixed_positions_checked"] = len(expected_fixed)
    return result


class MidplaceRepairSession:
    def __init__(self, engine):
        self.engine = engine
        self.config = MidplaceConfig.from_params(engine.params)
        engine.params._midplace_objectives_pending = resolve_flag(
            engine.params, "repair_buffering_midplace_objectives_after_repair"
        )
        self.tracker = OverflowTracker(self.config.overflow)
        self.profile = RepairProfile()
        self.schedule = MidplaceSchedule()
        self.resume_cursor = PlacementCursor()
        self.resume_sub_iteration = None
        self.completed_steps = 0
        self.initial_observed = False
        self.total_steps = sum(stage_budgets(engine.params.global_place_stages))
        self.continuation_steps = 0
        self.active = False
        self.restarted = False
        self.runtime_pending = False
        self.previous_names = set()
        self.buffers = []
        self.seed = None
        self.latest_step = None
        self.path = (
            Path(engine.params.result_dir) / engine.params.design_name() / self.config.artifact
        )
        self.record = {
            "artifact": "midplace_repair_handoff",
            "artifact_version": 2,
            "design_name": engine.params.design_name(),
            "status": "placement_running",
            "config": asdict(self.config),
            "repair_mode": self.config.mode,
            "timing_source": "dreamplace_timing_propagation",
            "ecc_sta_run": False,
            "terminal_status": "not_requested",
            "gate_tier": "A",
            "gate_status": "pending",
            "failure_stage": None,
            "legality_status": "not_checked_pre_legalize",
            "optimizer_restart_policy": "cold",
            "continuation_steps_requested": self.config.bounded_steps,
            "continuation_steps_completed": 0,
            "steps": [],
            "artifact_path": str(self.path),
        }
        self.write()

    @property
    def needs_bounded_steps(self):
        return self.active and self.config.bounded_steps > self.continuation_steps

    def write(self):
        self.record["phase_timings_ms"] = dict(self.profile.timings_ms)
        self.record["phase_rss_peak_kib"] = dict(self.profile.phase_rss_peak_kib)
        self.record["phase_rss_scope"] = "process_high_water_mark_at_phase_exit"
        self.record["refresh_timing_breakdown"] = {
            key: value
            for key, value in self.profile.timings_ms.items()
            if key in {"physical_pydb_export_ms", "timing_carrier_rebuild_ms"}
        }
        self.record["placedb_rebuild_breakdown"] = {
            key: value
            for key, value in self.profile.timings_ms.items()
            if key in {"placedb_rebuild_ms", "fresh_placer_build_ms", "continuation_seed_ms"}
        }
        self.record["native_commits"] = dict(self.profile.commits)
        self.record["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.record, indent=2, sort_keys=True, default=plain) + "\n"
        )
        temporary.replace(self.path)

    def build_placer(self, factory):
        phase = "fresh_placer_build_ms" if self.restarted else "initial_placer_build_ms"
        with self.profile.measure(phase):
            placer = factory()
        placer.midplace_repair = self
        return placer

    def runtime_ready(self, placer, cooptimization, controller):
        if not self.runtime_pending:
            return
        with self.profile.measure("runtime_view_validation_ms"):
            views = runtime_topology(placer, self.engine.placedb, self.previous_names)
            if len(views["new_buffers"]) != self.record["repair_inserted_count"]:
                raise RuntimeError(
                    "new movable buffer count differs from committed insertion count"
                )
            self.buffers = views["new_buffers"]
            self.record["runtime_views"] = views
        with self.profile.measure("continuation_seed_validation_ms"):
            self.record["position_continuity"] = validate_position_seed(
                placer, self.engine.placedb, self.seed
            )
        with self.profile.measure("virtual_state_validation_ms"):
            self.record["s5b1_inflation_state"] = self.schedule.restore(cooptimization, controller)
            self.record["inflated_geometry_restore"] = self.schedule.restore_inflated_geometry(
                placer, self.engine.placedb, cooptimization
            )
            self.record["new_placer_id"] = id(placer)
            self.record["position_seed_consumed"] = (
                self.engine.placedb.pending_continuation_seed is None
            )
        self.runtime_pending = False
        self.record["status"] = "placement_continuing"
        self.write()

    def after_step(
        self, placer, model, metric, eval_ops, cursor, elapsed_ms, *, pending_transaction=False
    ):
        placedb = self.engine.placedb
        overflow = post_step_overflow(placer, placedb, eval_ops, cursor.raw_iteration)
        self.observe_initial(metric, cursor)
        self.completed_steps += 1
        if self.active:
            self.continuation_steps += 1
            with self.profile.measure("post_handoff_step_validation_ms"):
                record = step_record(
                    placer,
                    placedb,
                    model,
                    metric,
                    overflow,
                    self.continuation_steps,
                    elapsed_ms,
                    self.buffers,
                )
            self.latest_step = record
            checkpoint = bool(
                self.config.bounded_steps
                or self.continuation_steps in {1, 5, 20}
                or self.continuation_steps % 50 == 0
            )
            if checkpoint:
                self.record["steps"].append(record)
            self.record["continuation_steps_completed"] = self.continuation_steps
            if self.continuation_steps == 1:
                self.profile.timings_ms["post_handoff_first_step_ms"] = elapsed_ms
                self.profile.phase_rss_peak_kib["post_handoff_first_step_ms"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                self.record.update(first_post_handoff_step=record, gate_status="passed")
            self.record["timing_objective_state"] = record.get(
                "timing_objective_state", {"enabled": False}
            )
            self.record["routability_state"] = {
                "enabled": record["routability_objective_enabled"],
                "cost": record.get("l_shape_cost"),
                "weighted_cost": record.get("l_shape_weighted_cost"),
                "cache": "fresh" if self.restarted else "retained_noop",
            }
            if checkpoint or self.continuation_steps % 10 == 0:
                self.write()
            if self.config.bounded_steps and self.continuation_steps >= self.config.bounded_steps:
                if self.buffers and not any(
                    entry["moved"]
                    for step in self.record["steps"]
                    for entry in step["buffer_positions"]
                ):
                    raise RuntimeError(
                        "inserted buffers did not participate in bounded position updates"
                    )
                if any(
                    not entry["inside_placement_boundary"] for entry in record["buffer_positions"]
                ):
                    self.profile.failure_stage = "post_handoff_boundary_failed"
                    raise RuntimeError("inserted buffer left the placement boundary")
                self.record.update(status="bounded_completed", gate_tier="B", gate_status="passed")
                self.write()
                raise MidplaceBoundedStop()
            return
        if self.defers_to_first_inflation_window(placer):
            self.tracker.previous_overflow = overflow
            return
        observation = OverflowObservation(
            overflow,
            int(getattr(placedb, "topology_generation", 0)),
            cursor,
            self.completed_steps,
            self.total_steps,
        )
        request = self.tracker.observe(observation, pending_transaction=pending_transaction)
        if request is not None:
            self._execute(placer, model, request, metric=metric)
        elif self.tracker.pending is not None:
            self.record.update(
                status="repair_handoff_pending", pending_crossing=asdict(self.tracker.pending[1])
            )
            self.write()

    def defers_to_first_inflation_window(self, placer):
        if self.active:
            return False
        params = self.engine.params
        if not bool(getattr(params, "routability_opt_flag", False)):
            return False
        owner = getattr(placer, "inflation_s5b1", None)
        schedule = getattr(owner, "schedule", None)
        return schedule is not None and schedule.mode == "inflation"

    def supports_inflation_window_repair_timing(self, placer):
        params = self.engine.params
        if not bool(getattr(params, "routability_opt_flag", False)):
            return False
        if not bool(getattr(params, "timing_opt_buffering_enabled", False)):
            return False
        owner = getattr(placer, "inflation_s5b1", None)
        schedule = getattr(owner, "schedule", None)
        return schedule is not None and schedule.mode == "inflation"

    def before_first_inflation(self, placer, model, metric, eval_ops, cursor):
        if self.active or self.tracker.consumed:
            return False
        placedb = self.engine.placedb
        overflow = float(metric.overflow.detach().max().cpu())
        threshold = float(
            getattr(self.engine.params, "node_area_adjust_overflow", self.config.overflow)
        )
        observation = OverflowObservation(
            overflow,
            int(getattr(placedb, "topology_generation", 0)),
            cursor,
            self.completed_steps,
            self.total_steps,
        )
        previous = self.tracker.previous_overflow
        self.tracker.previous_overflow = overflow
        self.tracker.consumed = True
        request = MidplaceRequest(
            "midplace-1",
            threshold,
            previous,
            observation,
            observation,
        )
        self._execute(
            placer,
            model,
            request,
            consume_inflation_window=True,
            metric=metric,
        )
        return True

    def after_inflation_window(
        self, placer, model, metric, eval_ops, cursor, *, window_index
    ):
        if not self.supports_inflation_window_repair_timing(placer):
            return False
        placedb = self.engine.placedb
        overflow = float(metric.overflow.detach().max().cpu())
        threshold = float(
            getattr(self.engine.params, "node_area_adjust_overflow", self.config.overflow)
        )
        observation = OverflowObservation(
            overflow,
            int(getattr(placedb, "topology_generation", 0)),
            cursor,
            self.completed_steps,
            self.total_steps,
        )
        previous = self.tracker.previous_overflow
        self.tracker.previous_overflow = overflow
        request = MidplaceRequest(
            f"midplace-w{int(window_index)}",
            threshold,
            previous,
            observation,
            observation,
        )
        self._execute(
            placer,
            model,
            request,
            consume_inflation_window=False,
            metric=metric,
            mode_override="repair_timing",
            artifact_subdir=f"midplace-repair-w{int(window_index)}",
            window_index=int(window_index),
        )
        return True

    def observe_initial(self, metric, cursor):
        """Capture the first metric before an S5B1 window can replace it."""
        if self.initial_observed or self.completed_steps:
            return
        self.initial_observed = True
        observation = OverflowObservation(
            float(metric.overflow.detach().max().cpu()),
            int(getattr(self.engine.placedb, "topology_generation", 0)),
            cursor,
            0,
            self.total_steps,
        )
        self.tracker.observe_initial(observation)
        self.record["initial_observation"] = asdict(observation)

    def _execute(
        self,
        placer,
        model,
        request,
        *,
        consume_inflation_window=False,
        metric=None,
        mode_override=None,
        artifact_subdir=None,
        window_index=None,
    ):
        placedb, params = self.engine.placedb, self.engine.params
        repair_mode = mode_override or self.config.mode
        self.record.update(
            status="repair_executing",
            request=request.as_dict(),
            trigger_progress=request.crossing.completed_effective_steps / max(1, self.total_steps),
            trigger_effective_step=request.crossing.completed_effective_steps,
            trigger_stage=request.crossing.cursor.stage,
            trigger_raw_iteration=request.crossing.cursor.raw_iteration,
            trigger_previous_overflow=request.previous_overflow,
            trigger_overflow=request.crossing.overflow,
            trigger_threshold=request.threshold,
            pre_repair_objective_state={
                "timing": bool(model.use_timing_obj),
                "routability": bool(model.use_l_shape_routability),
            },
            pre_counts=topology_counts(placedb),
            pre_topology_generation=request.execution.topology_generation,
            old_placer_id=id(placer),
            pre_pydb_identity=id(placedb.rawdb.pydb),
        )
        self.previous_names = set(names(placedb.node_names))
        self.schedule.capture(
            placer,
            model,
            request.execution.cursor.stage,
            cursor=request.execution.cursor,
            metric=metric,
        )
        old_seed = capture_fillers(placedb, placer.pos[0])
        warm = (
            capture_warm_schedule_state(model)
            if self.config.resume_policy == "warm_schedule"
            else None
        )
        self.write()
        with self.profile.measure("live_position_sync_ms"):
            owner = getattr(placer, "inflation_s5b1", None)
            if owner is not None:
                self.record["placement_sizing_sync"] = owner.sync_native_sizing()
            write_back_movable_lpos(placer.pos[0], params, placedb)
        # The terminal repair path reads this with a default; placement
        # workspaces that only set the midplace switches leave it absent.
        config = dict(getattr(params, "repair_buffering_config", None) or {})
        if repair_mode in ("repair_timing", "repair_design_then_timing"):
            for flag in (
                "use_native_timing_graph",
                "use_native_timing_queue",
                "use_native_timing_evaluation",
                "use_native_timing_planning",
            ):
                config.setdefault(flag, True)
            if (
                window_index is not None
                and self.config.max_buffer_fraction > 0.0
                and int(config.get("max_buffers", 0) or 0) <= 0
                and float(config.get("max_buffer_fraction", 0.0) or 0.0) <= 0.0
            ):
                config["max_buffer_fraction"] = self.config.max_buffer_fraction
        can_inplace_size = _can_sync_size_only_inplace(placer, placedb)
        stage_kwargs = (
            {"skip_size_only_rebuild": True} if can_inplace_size else {}
        )
        stage = run_repair_buffering_stage(
            placedb,
            mode=repair_mode,
            max_passes=self.config.max_passes,
            config=config,
            artifact_dir=self.path.parent / (artifact_subdir or "midplace-repair"),
            topology_generation=request.execution.topology_generation,
            profile=self.profile,
            **stage_kwargs,
        )
        self.record["repair"] = stage
        if stage["status"] != "ok":
            self.profile.failure_stage = (
                "native_commit_failed"
                if any(
                    result.get("status") == "commit_failed" for result in stage["stages"].values()
                )
                else "repair_plan_failed"
            )
            raise RuntimeError(f"midplace repair failed: {stage['status']}")
        design = stage["stages"].get("repair_design", {})
        inserted = int(design.get("inserted_count", 0) or 0)
        resized = int(design.get("resized_count", 0) or 0)
        timing = stage["stages"].get("repair_timing")
        if timing is not None:
            inserted += int(timing.get("verified_kept_count", 0) or 0)
        mutation = (
            "resize+insert"
            if inserted and resized
            else ("insert" if inserted else "resize-only" if resized else "noop")
        )
        self.record.update(
            mutation_kind=mutation,
            repair_inserted_count=inserted,
            repair_resized_count=resized,
            repair_action_digest=action_identity(stage),
            post_counts=topology_counts(placedb),
            post_topology_generation=int(placedb.topology_generation),
            post_pydb_identity=id(placedb.rawdb.pydb),
            fresh_tp_status="passed" if mutation != "noop" else "not_required",
        )
        self.record["timing_validation"] = timing_validation(stage)
        self.record["incremental_full_fallback"] = self.record["timing_validation"][
            "incremental_full_fallback"
        ]
        if window_index is not None:
            timing_plan = (timing or {}).get("plan") or {}
            max_buffers = int(
                (timing or {}).get(
                    "resolved_max_buffers",
                    timing_plan.get("max_buffers", 0),
                )
                or 0
            )
            stop_reason = timing_plan.get("stop_reason")
            window_entry = {
                "window": int(window_index),
                "mode": repair_mode,
                "mutation_kind": mutation,
                "inserted_count": inserted,
                "resized_count": resized,
                "max_buffers": max_buffers,
                "stop_reason": stop_reason,
                "timing_before": (timing or design).get("timing_before"),
                "timing_after": (timing or design).get("timing_after"),
                "artifact_path": stage.get("artifact_path"),
            }
            self.record.setdefault("window_repairs", []).append(window_entry)
            owner = getattr(placer, "inflation_s5b1", None)
            cumulative_inserted = sum(
                int(item.get("inserted_count", 0) or 0)
                for item in self.record.get("window_repairs", ())
            )
            buffering_summary = {
                "status": "committed" if inserted > 0 else "noop",
                "mode": repair_mode,
                "scheduled": int((timing or {}).get("accepted_count", 0) or 0),
                "accepted": inserted,
                "max_buffers": max_buffers,
                "stop_reason": stop_reason,
                "cumulative_count": cumulative_inserted,
                "timing_before": (timing or {}).get("timing_before"),
                "timing_after": (timing or {}).get("timing_after"),
            }
            if owner is not None and owner.summaries:
                owner.summaries[-1]["buffering"] = dict(buffering_summary)
                if callable(getattr(owner, "_write_summaries", None)):
                    owner._write_summaries()
            if self.schedule.timing_summaries:
                self.schedule.timing_summaries[-1]["buffering"] = dict(buffering_summary)
        self.active = True
        params._midplace_objectives_pending = False
        self.record["objectives_ready_after_repair"] = True
        if consume_inflation_window:
            self.schedule.record_inflation_window(
                stage=request.execution.cursor.stage,
                iteration=request.execution.cursor.raw_iteration + 1,
                window_summary={
                    "stage": int(request.execution.cursor.stage),
                    "iteration": int(request.execution.cursor.raw_iteration + 1),
                    "mode": repair_mode,
                    "mutation_kind": mutation,
                    "inserted_count": inserted,
                    "resized_count": resized,
                },
            )
        if mutation == "noop":
            if consume_inflation_window:
                self.record["s5b1_inflation_state"] = self.schedule.restore(
                    getattr(placer, "inflation_s5b1", None),
                    placer._routability_controller,
                )
            self.record.update(status="placement_continuing", runtime_reused=True)
            self.write()
            if not self.config.continue_placement:
                self.record.update(status="handoff_completed", gate_status="passed")
                self.write()
                raise MidplaceBoundedStop()
            return
        if mutation == "resize-only" and can_inplace_size:
            with self.profile.measure("inplace_runtime_sizing_sync_ms"):
                runtime_sync = _sync_size_only_runtime(
                    placer, model, params, placedb, design
                )
            if consume_inflation_window:
                self.record["s5b1_inflation_state"] = self.schedule.restore(
                    getattr(placer, "inflation_s5b1", None),
                    placer._routability_controller,
                )
            self.record.update(
                status="placement_continuing",
                runtime_reused=True,
                runtime_sizing_sync=runtime_sync,
                post_counts=topology_counts(placedb),
            )
            self.write()
            if not self.config.continue_placement:
                self.record.update(
                    status="handoff_completed_without_continuation", gate_status="pending"
                )
                self.write()
                raise MidplaceBoundedStop()
            return
        if not self.config.continue_placement:
            self.record.update(
                status="handoff_completed_without_continuation", gate_status="pending"
            )
            self.write()
            raise MidplaceBoundedStop()
        with self.profile.measure("continuation_seed_ms"):
            self.seed = committed_seed(placedb, old_seed)
            placedb.pending_continuation_seed = self.seed
            placedb.pending_warm_schedule_state = warm
            placedb.cooptimization_geometry = None
            self.resume_cursor = request.execution.cursor.after_step(
                gamma_count=model.Lgamma_iteration,
                density_count=model.Llambda_density_weight_iteration,
                sub_count=model.Lsub_iteration,
            )
            if self.resume_cursor.stage == request.execution.cursor.stage:
                self.resume_sub_iteration = int(
                    getattr(model, "routability_Lsub_iteration", model.Lsub_iteration)
                    if (
                        consume_inflation_window
                        or bool(getattr(params, "routability_opt_flag", False))
                    )
                    else model.Lsub_iteration
                )
        self.record.update(
            resume_cursor=asdict(self.resume_cursor),
            remaining_effective_steps=remaining_steps(
                params.global_place_stages,
                self.resume_cursor,
                sub_iteration=self.resume_sub_iteration,
            ),
            continuation_policy=self.config.resume_policy,
            status="placement_runtime_rebuild_pending",
            carrier_rebuild_mode="full",
        )
        self.restarted = self.runtime_pending = True
        self.write()
        logging.info(
            "Midplace repair committed: %s; inserted=%d resized=%d; resume=%s",
            mutation,
            inserted,
            resized,
            self.resume_cursor,
        )
        raise MidplaceRestart()

    def fail(self, error):
        raw = self.engine.placedb.rawdb
        self.record["failure_native_state"] = {
            "pydb_identity": id(raw.pydb),
            "refresh_generation": int(raw.refresh_generation),
            "native_mutation_epoch": int(raw.pydb.native_mutation_epoch),
            "native_state_dirty": bool(raw.pydb.native_state_dirty),
        }
        sizing = self.profile.commits.get("sizing", {})
        buffer = self.profile.commits.get("buffer", {})
        partial = bool(sizing.get("accepted_count", 0)) and (
            self.profile.failure_stage == "native_buffer_commit_ms"
            or buffer.get("status") not in {None, "accepted", "ok"}
        )
        self.record.update(
            status="partial_commit" if partial else "failed",
            gate_status="failed",
            failure_stage=self.profile.failure_stage
            or ("post_handoff_step_failed" if self.active else "trigger_failed"),
            error=str(error),
        )
        self.record["repair_plans_at_failure"] = dict(self.profile.plans)
        self.write()

    def placement_finished(self):
        if self.latest_step is not None and (
            not self.record["steps"] or self.record["steps"][-1]["step"] != self.latest_step["step"]
        ):
            self.record["steps"].append(self.latest_step)
        if self.config.bounded_steps:
            self.record.update(
                status="incomplete",
                gate_status="incomplete",
                stop_reason=(
                    "placement_ended_before_bounded_steps"
                    if self.active
                    else "overflow_threshold_not_reached"
                ),
            )
        else:
            self.record.update(
                status="placement_completed", terminal_status="global_placement_completed"
            )
        self.write()
        return dict(self.record)

    def finish_terminal(self, finalize):
        self.record["terminal_status"] = "materialization_and_legalization_running"
        self.write()
        try:
            with self.profile.measure("terminal_materialization_legalization_ms"):
                result = finalize()
            terminal = result["inflation_s5b1_terminal"]
            if result["status"] != "ok" or not terminal["legal"]:
                raise RuntimeError("midplace terminal materialization or legalization failed")
        except Exception as error:
            self.record["terminal_status"] = "failed"
            self.fail(error)
            raise
        self.record.update(
            status="full_flow_completed",
            terminal_status="legalization_completed",
            legality_status="passed",
            terminal_outputs={
                key: terminal[key] for key in ("final_def", "final_verilog", "prelegal_def")
            },
            terminal_counts=topology_counts(self.engine.placedb),
            terminal_qor_gate="not_evaluated",
        )
        self.write()
        return result


def _can_sync_size_only_inplace(placer, placedb):
    dc = getattr(placer, "data_collections", None)
    if dc is None or placedb is None:
        return False
    return all(
        getattr(dc, attr, None) is not None
        for attr in (
            "inst_cell_id",
            "node_size_x",
            "node_size_y",
            "node_areas",
            "flat_libcell_width",
            "flat_libcell_height",
        )
    )


def _sync_size_only_runtime(placer, model, params, placedb, design):
    import torch
    from dreamplace.ops.discrete_gradient_topk.runtime_cells import sync_runtime_cells
    from dreamplace.ops.routability.cooptimization_area import (
        SizingWindowGeometry,
        capture_area,
        coordinate_area,
    )

    resizes = list((design.get("plan") or {}).get("resizes", ()) or ())
    inst_ids = [int(item["node_id"]) for item in resizes]
    cell_ids = [
        int(
            item["target_cell_index"]
            if "target_cell_index" in item
            else item["new_cell_id"]
        )
        for item in resizes
    ]
    dc = placer.data_collections
    owner = getattr(placer, "inflation_s5b1", None)
    virtual_area = owner.virtual_area(model) if owner is not None else 0.0
    capacity = getattr(owner, "capacity", None)
    before = capture_area(dc, placedb, virtual_area, capacity=capacity)
    geometry = SizingWindowGeometry(dc, placedb, placer.pos[0])
    pin_inflate_x = (
        (dc.pin_offset_x - dc.original_pin_offset_x).detach().clone()
        if getattr(dc, "original_pin_offset_x", None) is not None
        else None
    )
    pin_inflate_y = (
        (dc.pin_offset_y - dc.original_pin_offset_y).detach().clone()
        if getattr(dc, "original_pin_offset_y", None) is not None
        else None
    )
    op_collections = getattr(placer, "op_collections", None)
    timing_op = getattr(op_collections, "timing_op", None)
    result = sync_runtime_cells(
        params,
        {
            "applied_instance_ids": inst_ids,
            "applied_cell_ids": cell_ids,
        },
        data_collections=dc,
        placedb=placedb,
        timing_op=timing_op,
        model=model,
        direct_joint=False,
        geometry_window=geometry,
        pin_offset_mode="delta",
    )
    with torch.no_grad():
        if pin_inflate_x is not None:
            dc.original_pin_offset_x.copy_(dc.pin_offset_x - pin_inflate_x)
        if pin_inflate_y is not None:
            dc.original_pin_offset_y.copy_(dc.pin_offset_y - pin_inflate_y)
        if owner is not None and getattr(owner, "native_cell_ids", None) is not None:
            owner.native_cell_ids.copy_(dc.inst_cell_id.detach().cpu())
    after = coordinate_area(before, dc, placedb, placer.pos[0], virtual_area)
    inflation_state = getattr(model, "inflation_state", None)
    if inflation_state is not None and inflation_state.target_area is not None:
        inflation_state.target_area = after.movable + after.filler
    if hasattr(model, "refresh_routability_operators"):
        model.refresh_routability_operators()
    if hasattr(model, "refresh_after_geometry_change"):
        model.refresh_after_geometry_change()
    result["area"] = {"before": before.__dict__, "after": after.__dict__}
    return result


def create_session(engine):
    from dreamplace.Placer import create_session as _create_session

    return _create_session(engine)
