"""Placement-owned native repair and fresh-runtime continuation.

The post-step hook delegates to this session. A noop keeps the live optimizer;
a physical mutation unwinds the old placement call before constructing its
replacement. No optimizer, timing graph or virtual topology crosses that seam.
"""

import json
import logging
import resource
from dataclasses import asdict
from pathlib import Path

from dreamplace.flows.handoff_state import capture_warm_schedule_state
from dreamplace.flows.midplace_observation import (
    action_identity,
    capture_fillers,
    committed_seed,
    names,
    plain,
    post_step_overflow,
    runtime_topology,
    step_record,
    timing_validation,
    topology_counts,
    validate_position_seed,
)
from dreamplace.flows.midplace_schedule import MidplaceSchedule
from dreamplace.flows.midplace_state import (
    MidplaceBoundedStop,
    MidplaceConfig,
    MidplaceRestart,
    OverflowObservation,
    OverflowTracker,
    PlacementCursor,
    remaining_steps,
    stage_budgets,
)
from dreamplace.flows.midplace_state import create_session as create_session
from dreamplace.flows.repair_buffering import run_repair_buffering_stage
from dreamplace.ops.placeio_ecc.export_options import resolve_flag
from dreamplace.ops.placeio_ecc.repair_profile import RepairProfile
from dreamplace.ops.routability.gpugr_context import write_back_movable_lpos


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
        observation = OverflowObservation(
            overflow,
            int(getattr(placedb, "topology_generation", 0)),
            cursor,
            self.completed_steps,
            self.total_steps,
        )
        request = self.tracker.observe(observation, pending_transaction=pending_transaction)
        if request is not None:
            self._execute(placer, model, request)
        elif self.tracker.pending is not None:
            self.record.update(
                status="repair_handoff_pending", pending_crossing=asdict(self.tracker.pending[1])
            )
            self.write()

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

    def _execute(self, placer, model, request):
        placedb, params = self.engine.placedb, self.engine.params
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
        self.schedule.capture(placer, model, request.execution.cursor.stage)
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
        if self.config.mode == "repair_design_then_timing":
            config.update(
                use_native_timing_graph=True,
                use_native_timing_queue=True,
                use_native_timing_evaluation=True,
                use_native_timing_planning=True,
            )
        stage = run_repair_buffering_stage(
            placedb,
            mode=self.config.mode,
            max_passes=self.config.max_passes,
            config=config,
            artifact_dir=self.path.parent / "midplace-repair",
            topology_generation=request.execution.topology_generation,
            profile=self.profile,
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
        design = stage["stages"]["repair_design"]
        inserted = int(design.get("inserted_count", 0) or 0)
        resized = int(design.get("resized_count", 0) or 0)
        timing = stage["stages"].get("repair_timing")
        if timing is not None:
            inserted += int(timing.get("verified_kept_count", 0))
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
        self.active = True
        params._midplace_objectives_pending = False
        self.record["objectives_ready_after_repair"] = True
        if mutation == "noop":
            self.record.update(status="placement_continuing", runtime_reused=True)
            self.write()
            if not self.config.continue_placement:
                self.record.update(status="handoff_completed", gate_status="passed")
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
                self.resume_sub_iteration = int(model.Lsub_iteration)
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
