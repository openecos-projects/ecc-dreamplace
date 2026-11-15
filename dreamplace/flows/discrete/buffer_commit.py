"""Project retained virtual buffers once from the final timing frame."""


import torch
from dreamplace.ops.buffer_insertion.buffering_config import build_buffering_config_from_params
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
from dreamplace.ops.routability.cooptimization_area import capture_area
from types import SimpleNamespace
from dreamplace.flows.discrete.common import parse_enabled_flag
from dreamplace.flows.gp.continuation import capture_warm_schedule_state
from dreamplace.flows.gp.continuation import capture_fillers, committed_seed
from dreamplace.flows.gp.continuation import MidplaceSchedule
from dreamplace.flows.gp.continuation import MidplaceRestart
from dreamplace.ops.placeio_common.physical_mutation import ECCBufferMutationBackend
from dreamplace.ops.routability.gpugr_context import write_back_movable_lpos


def prepare_terminal_request(owner, model, pos, *, iteration, refresh_geometry=True):
    """Materialize all retained counts from the final frame, without reselection."""
    from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
    from dreamplace.ops.buffer_insertion.projection import project_segment_count_buffer_state

    if owner.terminal_request is not None:
        raise RuntimeError("S5B1 final materialization was already prepared")
    if not bool(torch.isfinite(pos).all()):
        raise RuntimeError("S5B1 final placement coordinates are non-finite")
    state = None if owner.lane is None else owner.lane._current_state()
    candidate_mode = state is not None and hasattr(state, "activation_param")
    cumulative_count = 0
    if state is not None:
        count_tensor = state.activation_param if candidate_mode else state.z_param
        cumulative_count = int(count_tensor.detach().sum().cpu().item())
    if owner.lane is None:
        owner.lane = BufferingOptimizationLane(
            build_buffering_config_from_params(owner.params, flow_kind="placement"),
            placedb=owner.placedb,
        )
        projection = None
    elif cumulative_count == 0 and getattr(owner, "direct_commit_enabled", False):
        projection = None
    elif candidate_mode:
        # Rebuild the live candidate geometry before final projection so
        # no coordinate from an earlier placement generation is committed.
        # The unified diff_opt controller has already refreshed the lane
        # after its last explicit phase; skipping a second rebuild there
        # avoids repacking a partially frozen final topology.
        if refresh_geometry:
            owner.prepare(model, pos, refresh_records_all=True)
        projection = owner.lane.project(iteration=iteration, final=True)
        if projection.projected_buffer_count != cumulative_count:
            raise RuntimeError("final projection did not preserve every cumulative buffer")
    else:
        # Rebuild the ordinary nets and repack current retained references.
        # Frozen buffered nets retain their identity; no new S or B occurs.
        owner.prepare(model, pos)
        topology = owner.ops.steiner_topo_op
        generation = topology.freeze_topology()
        with torch.no_grad():
            _, x, y = model._timing_geometry_for_pos(pos)
        scale, shift = float(owner.params.scale_factor), owner.params.shift_factor
        context = {
            "coordinate_source": "final_live_geometry",
            "placement_snapshot_identity": _tensor_digest((("pos", pos),)),
            "placement_snapshot_iteration": int(iteration),
            "topology_generation": generation,
            "frozen_topology_generation": generation,
            "expected_frozen_topology_generation": generation,
            "rectilinear_path_policy": "x_then_y",
            "node_x_dbu": x.double() / scale + shift[0],
            "node_y_dbu": y.double() / scale + shift[1],
        }
        payload = owner.lane.buffer_relaxed_timing_payload
        projection = project_segment_count_buffer_state(
            owner.lane._current_state(),
            nets=payload["nets"],
            legal_table=payload["buffer_legal_table"],
            live_projection_context=context,
        )
        if projection.projected_buffer_count != cumulative_count:
            raise RuntimeError("final projection did not preserve every cumulative buffer")
    owner.terminal_request = owner.lane.build_commit_request(
        projection,
        iteration=iteration,
        reason="inflation_s5b1_final_commit",
        commit_enabled=True,
        refresh_mode="full_rebuild",
        rebuild_mode="full_rebuild",
    )
    model._timing_geometry_cache = None
    candidate_provider_was_enabled = bool(
        getattr(owner.params, "enable_relaxed_buffer_timing", False)
    )
    if candidate_mode:
        # Terminal reporting must include any accepted typed actions while
        # preserving the provider setting used by the regular GP loop.
        owner.params.enable_relaxed_buffer_timing = True
    try:
        with owner.electrical_window(), torch.no_grad():
            outputs = model.timing_obj(pos)
            loss = model._timing_loss(*outputs)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("S5B1 final virtual timing loss is non-finite")
            gp_timing = dict(
                zip(("wns", "tns", "ws", "ts"), (float(value) for value in outputs), strict=True)
            )
            timing = owner.ops.timing_propagation_op
            raw_cap = timing.last_total_cap_violation
            gp_timing["slew_violation_ns"] = timing.last_total_slew_violation
            gp_timing["cap_violation_pf"] = raw_cap
            gp_timing["cap_violation_ff"] = None if raw_cap is None else float(raw_cap) * 1000.0
            gp_timing["objective_terms"] = dict(model.last_timing_objective_terms)
    finally:
        owner.params.enable_relaxed_buffer_timing = candidate_provider_was_enabled
    model._timing_geometry_cache = None
    owner.terminal_summary = {
        "virtual_count": cumulative_count,
        "direct_committed_count": int(getattr(owner, "committed_buffer_count", 0) or 0),
        "scale_factor": float(owner.params.scale_factor),
        "dbu": float(owner.placedb.dbu),
        "gp_timing": gp_timing,
        "gp_area": capture_area(
            owner.data, owner.placedb, owner.virtual_area(model), capacity=owner.capacity
        ).__dict__,
        "request": owner.terminal_request.to_summary(),
    }
    if candidate_mode:
        owner.lane.retain_candidate_python_state = False
        from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
            _release_candidate_python_state,
        )

        _release_candidate_python_state(owner.lane)
    return {
        "status": "request_ready",
        "iteration": int(iteration),
        "projected_buffer_count": cumulative_count,
        "commit_request": owner.terminal_request.to_summary(),
    }


def supports_direct_buffer_commit(params, placedb) -> bool:
    """Return True when post-B1 buffers can commit directly to an ECC carrier DB."""
    if params is None or placedb is None:
        return False
    if not parse_enabled_flag(params, "timing_buffering_direct_commit", default=True):
        return False
    if str(getattr(params, "place_io_engine", "") or "") != "ecc":
        return False
    rawdb = getattr(placedb, "rawdb", None)
    if rawdb is None or getattr(rawdb, "timing_enabled", False) is not True:
        return False
    return callable(getattr(placedb, "refresh_from_ecc_backend", None))


def project_window_buffer_request(owner, model, pos, *, iteration: int, reason: str):
    """Project accepted window buffers at live coordinates before clearing virtual state."""
    from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
    from dreamplace.ops.buffer_insertion.projection import project_segment_count_buffer_state

    if not bool(torch.isfinite(pos).all()):
        raise RuntimeError("Direct B1 commit placement coordinates are non-finite")
    lane = owner.lane
    state = None if lane is None else lane._current_state()
    if state is None:
        return None
    candidate_mode = hasattr(state, "activation_param")
    count_tensor = state.activation_param if candidate_mode else state.z_param
    active_count = int(count_tensor.detach().sum().cpu().item())
    if active_count <= 0:
        return None
    if candidate_mode:
        projection = lane.project(iteration=int(iteration), final=True)
    else:
        topology = owner.ops.steiner_topo_op
        generation = int(getattr(topology, "topology_generation", 0) or 0)
        with torch.no_grad():
            _, x, y = model._timing_geometry_for_pos(pos)
        scale, shift = float(owner.params.scale_factor), owner.params.shift_factor
        context = {
            "coordinate_source": "final_live_geometry",
            "placement_snapshot_identity": _tensor_digest((("pos", pos),)),
            "placement_snapshot_iteration": int(iteration),
            "topology_generation": generation,
            "frozen_topology_generation": generation,
            "expected_frozen_topology_generation": generation,
            "rectilinear_path_policy": "x_then_y",
            "node_x_dbu": x.double() / scale + shift[0],
            "node_y_dbu": y.double() / scale + shift[1],
        }
        payload = lane.buffer_relaxed_timing_payload
        projection = project_segment_count_buffer_state(
            state,
            nets=payload["nets"],
            legal_table=payload["buffer_legal_table"],
            live_projection_context=context,
        )
    if projection.projected_buffer_count != active_count:
        raise RuntimeError("direct B1 projection did not preserve every accepted buffer")
    return lane.build_commit_request(
        projection,
        iteration=int(iteration),
        reason=str(reason),
        commit_enabled=True,
        refresh_mode="python_carrier",
        rebuild_mode="full_rebuild",
    )


def clear_virtual_buffer_state(owner, model) -> None:
    """Drop all virtual buffer/candidate state so subsequent phases run on real DB only."""
    params = getattr(owner, "params", None)
    if params is not None:
        params.enable_relaxed_buffer_timing = False
    lane = getattr(owner, "lane", None)
    if lane is not None:
        lane.buffer_optimization_state = None
        lane.buffer_relaxed_timing_payload = None
        lane.last_projection_result = None
        lane.retain_candidate_python_state = False
        from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
            _release_candidate_python_state,
        )

        _release_candidate_python_state(lane)
    data = getattr(owner, "data", None)
    if data is not None:
        data.buffer_optimization_state = None
        data.buffer_segment_count_state = None
        data.buffer_segment_count_payload = None
        data.buffer_relaxed_timing_payload = None
    ops = getattr(owner, "ops", None)
    if ops is not None:
        ops.virtual_cell_density_op = None
        topo = getattr(ops, "steiner_topo_op", None)
        frozen = getattr(topo, "frozen_nets", None)
        if frozen is not None:
            frozen.candidate_state = None
            frozen.segment_state = None
    if model is not None:
        model._virtual_cell_density_view = None
        model._candidate_density_view = None
        model._timing_geometry_cache = None
    owner._pending_dirty_instance_ids = set()
    owner._pending_dirty_net_ids = set()


def _extract_inserted_records(commit: dict) -> list[dict]:
    records = []
    for item in commit.get("results", ()) or ():
        if not isinstance(item, dict) or not item.get("accepted"):
            continue
        name = item.get("inserted_buffer_name")
        if not name and isinstance(item.get("connectivity"), dict):
            name = item["connectivity"].get("inserted_buffer_name")
        rec = dict(item)
        if name:
            rec["inserted_buffer_name"] = str(name)
        records.append(rec)
    if not records:
        for name in commit.get("committed_buffer_instance_names", ()) or ():
            records.append({"inserted_buffer_name": str(name), "accepted": True})
    return records


def commit_buffer_request_to_ecc(
    owner, placedb, params, request, *, pos=None, output_dir=None
) -> dict:
    """Sync sizing, commit projected buffers to ECC DB, and refresh Python timing carrier."""
    sizing_sync = owner.sync_native_sizing()
    if pos is not None:
        write_back_movable_lpos(pos, params, placedb)
    caps = dict(
        getattr(placedb, "backend_caps", None)
        or {
            "backend": "ecc",
            "supports_buffer_commit": True,
            "supports_committed_refresh": True,
        }
    )
    backend = ECCBufferMutationBackend(placedb, caps)
    commit = backend.commit_buffers(request, output_dir=output_dir)
    accepted = int(commit.get("accepted_action_count", 0) or 0)
    if accepted != len(request.actions) or commit.get("status") != "accepted":
        raise RuntimeError(f"Direct post-B1 buffer commit failed: {commit}")
    fillers_enabled = getattr(params, "enable_fillers", 0)
    refresh = backend.refresh(
        refresh_mode="python_carrier",
        rebuild_mode="full_rebuild",
    )
    params.enable_fillers = fillers_enabled
    if refresh.get("status") != "ok":
        raise RuntimeError(f"Direct post-B1 carrier refresh failed: {refresh}")

    inserted_records = _extract_inserted_records(commit)
    owner.committed_buffer_count = int(getattr(owner, "committed_buffer_count", 0)) + accepted
    existing_records = list(getattr(owner, "committed_buffer_records", ()) or ())
    existing_records.extend(inserted_records)
    owner.committed_buffer_records = existing_records
    owner._incremental_sizing_cache = None
    return {
        "status": "committed",
        "accepted_count": accepted,
        "cumulative_committed_count": owner.committed_buffer_count,
        "sizing": sizing_sync,
        "commit": commit,
        "refresh": refresh,
    }


class DirectBufferCommitSession:
    """Mid-placement session that commits accepted B1 buffers immediately to ECC DB."""

    def __init__(self, engine):
        self.engine = engine
        self.config = SimpleNamespace(bounded_steps=0)
        self.schedule = MidplaceSchedule()
        self.resume_cursor = None
        self.resume_sub_iteration = None
        self.restarted = False
        self.runtime_pending = False
        self.alternating_pending = False
        self.active = False
        self.needs_bounded_steps = False
        self.record = {
            "artifact": "direct_buffer_commit",
            "status": "placement_running",
            "window_commits": [],
        }

    def build_placer(self, factory):
        placer = factory()
        placer.midplace_repair = self
        return placer

    def runtime_ready(self, placer, cooptimization, routability_controller):
        if cooptimization is not None:
            owner = getattr(cooptimization, "owner", cooptimization)
            owner.direct_commit_enabled = True
        if not self.runtime_pending:
            return
        self.schedule.restore(cooptimization, routability_controller)
        self.schedule.restore_inflated_geometry(
            placer, self.engine.placedb, cooptimization
        )
        self.runtime_pending = False
        self.record["status"] = "placement_continuing"

    def bind_alternating_controller(self, placer, controller):
        del placer
        if controller is None:
            return
        controller.owner.direct_commit_enabled = True
        if self.alternating_pending:
            self.schedule.restore_alternating(controller)
            self.alternating_pending = False

    def observe_initial(self, metric, cursor):
        del metric, cursor

    def after_step(self, *args, **kwargs):
        del args, kwargs

    def defers_to_first_inflation_window(self, placer):
        del placer
        return False

    def placement_finished(self):
        self.record["status"] = "completed"
        return dict(self.record)

    def finish_terminal(self, fn):
        return fn()

    def fail(self, error):
        self.record["status"] = "failed"
        self.record["error"] = str(error)

    def _commit_and_restart(
        self,
        placer,
        model,
        cooptimization,
        metric,
        cursor,
        *,
        resume_cursor,
        use_restart_density_weight: bool,
        resume_sub_iteration: int,
    ) -> bool:
        owner = getattr(cooptimization, "owner", cooptimization)
        request = getattr(owner, "_pending_window_commit_request", None)
        if request is None:
            return False
        owner._pending_window_commit_request = None
        placedb, params = self.engine.placedb, self.engine.params
        old_seed = capture_fillers(placedb, placer.pos[0])
        warm = None if use_restart_density_weight else capture_warm_schedule_state(model)

        commit_summary = commit_buffer_request_to_ecc(
            owner,
            placedb,
            params,
            request,
            pos=placer.pos[0],
            output_dir=getattr(params, "result_dir", None),
        )
        if getattr(owner, "summaries", None):
            owner.summaries[-1]["direct_buffer_commit"] = commit_summary
            if callable(getattr(owner, "_write_summaries", None)):
                owner._write_summaries()
        alternating = getattr(placer, "alternating_discrete", None)
        if alternating is not None and getattr(alternating, "trace", None):
            alternating.trace[-1]["direct_buffer_commit"] = commit_summary
            if callable(getattr(alternating, "_write_summary", None)):
                alternating._write_summary(
                    status="completed" if not alternating.active else "active"
                )

        self.schedule.capture(
            placer,
            model,
            cursor.stage,
            cursor=cursor,
            metric=metric,
        )
        self.schedule.use_restart_density_weight = bool(use_restart_density_weight)

        placedb.pending_continuation_seed = committed_seed(placedb, old_seed)
        placedb.pending_warm_schedule_state = warm
        placedb.cooptimization_geometry = None
        self.resume_cursor = resume_cursor
        self.resume_sub_iteration = int(resume_sub_iteration)
        self.restarted = True
        self.runtime_pending = True
        self.alternating_pending = True
        self.record["window_commits"].append(
            {
                "stage": int(cursor.stage),
                "raw_iteration": int(cursor.raw_iteration),
                "accepted_count": int(commit_summary["accepted_count"]),
                "cumulative_committed_count": int(
                    commit_summary["cumulative_committed_count"]
                ),
            }
        )
        raise MidplaceRestart()

    def after_timing_window_step(self, placer, model, cooptimization, metric, cursor):
        if cooptimization is None:
            return False
        return self._commit_and_restart(
            placer,
            model,
            cooptimization,
            metric,
            cursor,
            resume_cursor=cursor,
            use_restart_density_weight=False,
            resume_sub_iteration=int(model.Lsub_iteration),
        )

    def after_inflation_round(
        self, placer, model, cooptimization, metric, cursor, *, adjust_area_flag: bool
    ):
        if cooptimization is None:
            return False
        next_cursor = cursor.after_step(
            gamma_count=model.Lgamma_iteration,
            density_count=model.Llambda_density_weight_iteration,
            sub_count=model.Lsub_iteration,
        )
        sub_iter = int(
            model.routability_Lsub_iteration
            if adjust_area_flag
            else model.Lsub_iteration
        )
        return self._commit_and_restart(
            placer,
            model,
            cooptimization,
            metric,
            cursor,
            resume_cursor=next_cursor,
            use_restart_density_weight=bool(adjust_area_flag),
            resume_sub_iteration=sub_iter,
        )

def create_direct_commit_session(engine):
    """Return DirectBufferCommitSession when direct post-B1 ECC commit is active."""
    if getattr(engine.params, "_midplace_objectives_pending", False):
        engine.params._midplace_objectives_pending = False
    buffering_active = bool(
        getattr(engine.params, "timing_opt_enabled", False)
        and getattr(engine.params, "timing_opt_buffering_enabled", False)
    ) or (
        str(getattr(engine.params, "joint_quality_profile", "") or "")
        == "discrete_alternating_sizing_buffering_v1"
    )
    if buffering_active and supports_direct_buffer_commit(engine.params, engine.placedb):
        return DirectBufferCommitSession(engine)
    return None
