"""Fixed-center sizing with an optional persistent virtual-buffer batch per window.

This owner publishes one window's electrical, native master and area state.
The caller keeps the original routing/inflation/GP schedule.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from dreamplace.flows.candidate_refresh import (
    accepted_candidate_net_ids,
    prepare_candidate_owner,
    refresh_owner_candidate_geometry_after_topology,
    refresh_topology_if_needed,
)
from dreamplace.flows.discrete_timing_common import (
    segment_counts,
    timing_area_fits,
    timing_electrical_window,
    timing_sizing_action_mask,
    timing_sizing_coefficients,
    transfer_segment_counts,
    virtual_timing_area,
)
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.flows.timing_opt_schedule import TimingOptWindowSchedule
from dreamplace.ops.buffer_insertion.buffering_config import build_buffering_config_from_params
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
from dreamplace.ops.buffer_insertion.discrete_virtual_scheduler import (
    schedule_net_gradient_actions,
    take_transition_prefix,
    transition_backtracking_sizes,
)
from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state
from dreamplace.ops.discrete_gradient_topk import (
    apply_quad_gradient_from_data_collections,
    build_quad_candidate_table,
)
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState
from dreamplace.ops.discrete_gradient_topk.runtime_cells import sync_runtime_cells
from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
    active_candidate_net_ids,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.routability.cooptimization_area import (
    SizingWindowGeometry,
    capture_area,
    coordinate_area,
    publish_area,
)


@dataclass
class SizingState:
    size: torch.Tensor
    vt: torch.Tensor
    masters: torch.Tensor
    oscillation: dict


class InflationS5B1:
    def __init__(
        self,
        params,
        placedb,
        data,
        ops,
        refresh_topology,
        *,
        window_limit=None,
        refresh_candidates_after_sizing=False,
    ):
        self.params, self.placedb, self.data, self.ops = params, placedb, data, ops
        self.refresh_topology = refresh_topology
        self.window_limit = window_limit
        self.refresh_candidates_after_sizing = bool(refresh_candidates_after_sizing)
        self.action_mode = str(
            getattr(params, "timing_action_gradient_mode", "off") or "off"
        )
        if self.action_mode not in ("off", "lut_action"):
            raise ValueError(
                "timing_action_gradient_mode must be one of: off, lut_action"
            )
        self._lut_action_owner = (
            DiscreteTimingOwner(
                params,
                placedb,
                data,
                ops,
                refresh_topology,
                refresh_candidates_after_sizing=True,
            )
            if self.action_mode == "lut_action"
            else None
        )
        self.oscillation = CellOscillationState()
        self.candidates = build_quad_candidate_table(
            data.flat_libcell_info, data.flat_libcell_leakage
        )
        self.lane = None
        self.round_index = 0
        self.window_count = 0
        self.attempts = set()
        self.summaries = []
        self.schedule = TimingOptWindowSchedule(
            getattr(params, "timing_opt_overflow_milestones", ()),
            params.timing_opt_max_windows,
        )
        self.milestones = self.schedule.milestones
        self.pending_milestones = self.schedule.pending_milestones
        self.capacity = capture_area(data, placedb).capacity
        self.native_cell_ids = data.inst_cell_id.detach().cpu().clone()
        self.terminal_request = None
        self.terminal_commit_started = False
        self._pending_dirty_instance_ids = set()
        self._pending_dirty_net_ids = set()
        self.last_candidate_dirty_net_summary = {}
        self._candidate_refresh_in_progress = False
        self._topology_geometry_key = None
        self.topology_refresh_count = 0
        self.topology_refresh_reuse_count = 0
        self.last_topology_refresh_state = {}
        # The native write-back path consumes this geometry even when
        # routability inflation is disabled and no sizing window ran.
        if not hasattr(data, "original_node_size_x"):
            data.original_node_size_x = data.node_size_x.detach().clone()
        if not hasattr(data, "original_node_size_y"):
            data.original_node_size_y = data.node_size_y.detach().clone()
        placedb.cooptimization_geometry = data

    def electrical_window(self):
        """Public compatibility name shared by discrete timing owners."""
        return timing_electrical_window(self.params, self.data)

    def _electrical_window(self):
        return timing_electrical_window(self.params, self.data)

    def _sizing_coefficients(self, model):
        return timing_sizing_coefficients(self.params, model)

    def _capture(self):
        return SizingState(
            self.data.get_continuous_size_parameter().detach().clone(),
            self.data.vt_logits.detach().clone(),
            self.data.inst_cell_id.detach().clone(),
            self.oscillation.snapshot(),
        )

    def _sync(self, model, summary, geometry):
        result = sync_runtime_cells(
            self.params,
            summary,
            data_collections=self.data,
            placedb=self.placedb,
            timing_op=self.ops.timing_propagation_op,
            model=model,
            direct_joint=False,
            geometry_window=geometry,
        )
        self._pending_dirty_instance_ids.update(
            int(value) for value in summary.get("applied_instance_ids", ()) or ()
        )
        return result

    def _restore(self, model, best, geometry):
        data = self.data
        changed = torch.nonzero(data.inst_cell_id != best.masters).flatten()
        with torch.no_grad():
            data.get_continuous_size_parameter().copy_(best.size)
            data.vt_logits.copy_(best.vt)
        if changed.numel():
            self._sync(
                model,
                {
                    "applied_instance_ids": changed.cpu().tolist(),
                    "applied_cell_ids": best.masters[changed].cpu().tolist(),
                },
                geometry,
            )
        self.oscillation.restore(best.oscillation)
        model._timing_geometry_cache = None

    def _loss(self, model, pos):
        # Both gradient and acceptance use the same model, lane and weights.
        result = model.timing_obj(pos.detach(), surrogate_mode="surrogate_only")
        loss = model._timing_loss(*result)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("S5B1 electrical loss is non-finite")
        return loss

    def virtual_area(self, model, state=None):
        return virtual_timing_area(self, model, state)

    def _fits(self, model, capacity):
        return timing_area_fits(self, model, capacity)

    @staticmethod
    def _counts(state):
        return segment_counts(state)

    @staticmethod
    def _transfer_counts(state, counts):
        return transfer_segment_counts(state, counts)

    def _prepare(
        self,
        model,
        pos,
        *,
        active_only=False,
        refresh_records_all=False,
    ):
        return prepare_candidate_owner(
            self,
            model,
            pos,
            active_only=active_only,
            refresh_records_all=bool(refresh_records_all or not active_only),
            force_topology=True,
            lane_flow_kind="placement",
            force_rebuild_when_unbounded=True,
            clear_timing_geometry_cache=True,
            reason="inflation_live_candidate_refresh",
        )

    def prepare(
        self,
        model,
        pos,
        *,
        active_only=False,
        refresh_records_all=False,
    ):
        """Prepare the current candidate/segment lane for a timing window."""
        return self._prepare(
            model,
            pos,
            active_only=active_only,
            refresh_records_all=refresh_records_all,
        )

    def refresh_candidate_geometry_after_topology(self, model, *, reason):
        """Refresh retained candidate geometry after a live topology update."""

        return refresh_owner_candidate_geometry_after_topology(
            self,
            model,
            reason=reason,
        )

    def _sizing(self, model, pos, geometry, capacity, *, action_mask=None):
        data = self.data
        candidates = self.candidates.with_action_mask(
            timing_sizing_action_mask(self.params, action_mask)
        )
        size, vt = data.get_continuous_size_parameter(), data.vt_logits
        best = self._capture()
        loss = self._loss(model, pos)
        initial_loss = best_loss = float(loss.detach())
        rounds, best_round = [], 0
        for round_id in range(1, int(self.params.timing_opt_sizing_rounds) + 1):
            size_grad, vt_grad = torch.autograd.grad(loss, (size, vt), allow_unused=False)
            model._timing_geometry_cache = None
            if not bool(torch.isfinite(size_grad).all() & torch.isfinite(vt_grad).all()):
                raise RuntimeError("S5B1 size/VT gradient is non-finite")
            self.round_index += 1
            blocked = self.oscillation.blocked(
                phase="timing",
                iteration=self.round_index,
                count=data.inst_cell_id.numel(),
                device=size.device,
            )
            next_size, next_vt, summary = apply_quad_gradient_from_data_collections(
                data_collections=data,
                size_logits=size,
                size_grad=size_grad,
                vt_logits=vt,
                vt_grad=vt_grad,
                size_up_percent=0.0,
                size_down_percent=0.0,
                vt_percent=0.0,
                candidate_table=candidates,
                shared_budget_percent=float(
                    self.params.discrete_gradient_topk_shared_budget_percent
                ),
                blocked_instances=blocked,
                size_parameterization="real_size",
            )
            if not summary["num_changed_instances"]:
                # A fixed-budget timing window records a no-op attempt and
                # rebuilds the graph for the next round.  Ending early here
                # made S5B1 report fewer rounds than the configured budget.
                rounds.append(
                    {
                        "round": round_id,
                        "loss": float(loss.detach()),
                        "fits": self._fits(model, capacity),
                        "actions": summary.get("selected_scores", ()),
                        "changed_instance_count": 0,
                        "no_action": True,
                    }
                )
                loss = self._loss(model, pos)
                continue
            rows = summary["applied_instance_ids"]
            previous = data.inst_cell_id[rows].detach().cpu().tolist()
            with torch.no_grad():
                size.copy_(next_size)
                vt.copy_(next_vt)
            self._sync(model, summary, geometry)
            self.oscillation.record(
                phase="timing",
                iteration=self.round_index,
                instance_ids=rows,
                previous_ids=previous,
                target_ids=summary["applied_cell_ids"],
            )
            loss = self._loss(model, pos)
            actual = float(loss.detach())
            fits = self._fits(model, capacity)
            rounds.append(
                {
                    "round": round_id,
                    "loss": actual,
                    "fits": fits,
                    "actions": summary["selected_scores"],
                }
            )
            if fits and actual < best_loss:
                best, best_loss, best_round = self._capture(), actual, round_id
        self._restore(model, best, geometry)
        return {
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_round": best_round,
            "rounds": rounds,
        }

    def run_sizing(self, model, pos, geometry, capacity, *, action_mask=None):
        if self._lut_action_owner is None:
            return self._sizing(model, pos, geometry, capacity, action_mask=action_mask)
        result = self._lut_action_owner.run_sizing(
            model,
            pos,
            geometry,
            capacity,
            action_mask=action_mask,
        )
        pending_instances = getattr(
            self._lut_action_owner, "_pending_dirty_instance_ids", set()
        )
        if pending_instances:
            self._pending_dirty_instance_ids.update(pending_instances)
            pending_instances.clear()
        pending_nets = getattr(self._lut_action_owner, "_pending_dirty_net_ids", set())
        if pending_nets:
            self._pending_dirty_net_ids.update(pending_nets)
            pending_nets.clear()
        self.round_index += len(result.get("rounds", ()) or ())
        return result

    def _buffering(self, model, pos, capacity):
        state = self.lane._current_state()
        if hasattr(state, "activation_param"):
            from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
                run_discrete_candidate_buffer_scheduler,
            )

            activation_before = state.activation_param.detach().clone()
            try:
                loop_summary = run_discrete_candidate_buffer_scheduler(
                    lane=self.lane,
                    model=model,
                    rounds=int(getattr(self.lane.config, "continuous_steps", 0)),
                    allow_existing_activation=True,
                )
            finally:
                # Candidate timing is a window-local probe.  Leaving its
                # large expanded provider enabled during ordinary GP makes
                # every subsequent placement step pay for all candidate rows.
                self.params.enable_relaxed_buffer_timing = False
            count = int(state.activation_param.detach().sum().cpu().item())
            accepted_net_ids = accepted_candidate_net_ids(
                activation_before,
                state,
            )
            self._pending_dirty_net_ids.update(accepted_net_ids)
            terminal = dict(loop_summary.terminal_metrics or {})
            from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest

            return {
                "initial_loss": float(terminal.get("loss", 0.0)),
                "final_loss": float(terminal.get("loss", 0.0)),
                "scheduled": int(loop_summary.iterations),
                "accepted": count,
                "accepted_net_ids": accepted_net_ids,
                "gradient_norm": None,
                "cumulative_count": count,
                "trace_path": loop_summary.trace_path,
                "trace_sha256": loop_summary.trace_sha256,
                "accepted_action_digest": _tensor_digest(
                    (("candidate_activation", state.activation_param),)
                ),
                "strategy": "discrete_net_gradient",
                "best_restore": dict(loop_summary.best_restore or {}),
            }
        before = state.z_param.detach().clone()
        state.z_param.requires_grad_(requires_grad=True)
        try:
            loss = self._loss(model, pos)
            initial = float(loss.detach())
            gradient = torch.autograd.grad(loss, state.z_param, allow_unused=False)[0]
            model._timing_geometry_cache = None
        finally:
            state.z_param.requires_grad_(requires_grad=False)
        transition = schedule_net_gradient_actions(
            segment_net_index=state.segment_net_index,
            net_ids=state.net_ids,
            segment_ids=state.segment_ids,
            counts=before,
            raw_grad=gradient,
            max_count=state.max_repeater_count,
            selection_fraction=float(self.params.buffering_route_b_selection_fraction),
        )
        accepted, final = 0, initial
        for count in transition_backtracking_sizes(transition.accepted_count):
            trial = take_transition_prefix(transition, counts=before, selected_count=count)
            with torch.no_grad():
                state.z_param.copy_(trial.next_counts)
            if not self._fits(model, capacity):
                continue
            with torch.no_grad():
                actual = float(self._loss(model, pos))
            if actual < initial:
                accepted, final = count, actual
                break
        if not accepted:
            with torch.no_grad():
                state.z_param.copy_(before)
        model._timing_geometry_cache = None
        return {
            "initial_loss": initial,
            "final_loss": final,
            "scheduled": transition.accepted_count,
            "accepted": accepted,
            "gradient_norm": float(gradient.norm()),
            "cumulative_count": int(state.z_param.detach().sum()),
        }

    def run_buffering(self, model, pos, capacity):
        return self._buffering(model, pos, capacity)

    def _retain_buffered_nets(self, model):
        """Ordinary GP carries only buffered trees; B gets fresh full candidates."""
        state = self.lane._current_state()
        if hasattr(state, "activation_param"):
            net_ids = active_candidate_net_ids(state)
            if net_ids:
                topo = self.ops.steiner_topo_op
                topo.freeze_nets(net_ids)
                topo.frozen_nets.candidate_state = state
                self._candidate_refresh_in_progress = True
                try:
                    refresh_topology_if_needed(
                        self,
                        self.data.pos[0],
                        force=True,
                    )
                finally:
                    self._candidate_refresh_in_progress = False
                refresh = self.lane.refresh_candidate_geometry(
                    model=model,
                    params=self.params,
                    reason="retain_inflation_candidate_nets",
                    active_only=False,
                    dirty_net_ids=net_ids,
                )
                if refresh.get("status") != "ok":
                    raise RuntimeError(f"S5B1 candidate retention failed: {refresh}")
                topo.frozen_nets.candidate_state = self.lane._current_state()
            state.activation_param.requires_grad_(requires_grad=False)
            state.activation_param.grad = None
            state.timing_active_only = True
            self.lane.attach_state(self.data)
            self.params.enable_relaxed_buffer_timing = True
            model._virtual_cell_density_view = None
            model._timing_geometry_cache = None
            self._pending_dirty_instance_ids = set()
            self._pending_dirty_net_ids = set()
            return
        if not hasattr(state, "z_param"):
            return
        counts = self._counts(state)
        nets_with_buffers = {key[0] for key in counts}
        if not nets_with_buffers:
            self.ops.steiner_topo_op.frozen_nets.segment_state = None
            return
        payload = dict(self.lane.buffer_relaxed_timing_payload)
        nets = [net for net in payload["nets"] if int(net["net_id"]) in nets_with_buffers]
        if not nets and state.is_tensor_backed:
            nets = list(state.net_records(nets_with_buffers))
        retained = build_segment_count_state(
            nets,
            buffer_main_type_index=state.buffer_main_type_index,
            legal_buffer_count=state.legal_buffer_count,
            max_repeater_count=state.max_repeater_count,
            initial_z=0.0,
            fixed_bsu_index=state.fixed_bsu_index,
            dtype=state.z_param.dtype,
            device=state.z_param.device,
            retained_edges=counts,
        )
        self._transfer_counts(retained, counts)
        retained.z_param.requires_grad_(requires_grad=False)
        retained.bsu_index_param.requires_grad_(requires_grad=False)
        retained.prepared_timing_inputs = build_segment_count_timing_inputs(
            nets,
            retained,
            dtype=retained.z_param.dtype,
            device=retained.z_param.device,
        )
        payload.update(nets=nets, prepared_timing_inputs=retained.prepared_timing_inputs)
        self.lane.buffer_optimization_state = retained
        self.lane.buffer_relaxed_timing_payload = payload
        self.lane.attach_state(self.data)
        topo = self.ops.steiner_topo_op
        topo.freeze_nets(nets_with_buffers)
        topo.frozen_nets.segment_state = retained
        model._virtual_cell_density_view = None

    def retain_buffered_nets(self, model):
        return self._retain_buffered_nets(model)

    def run(self, model, pos, *, iteration):
        started = time.perf_counter()
        if self.data.sizing_parameterization != "real_size":
            raise ValueError("inflation S5B1 requires real_size")
        if len(self.placedb.regions):
            raise ValueError("inflation S5B1 does not support fence regions")
        # S uses the retained GP state. Full candidate packing is only needed
        # for B, after the selected masters and pin geometry are current.
        if self.lane is None:
            refresh_topology_if_needed(self, pos, force=True)
        model._timing_geometry_cache = None
        geometry = SizingWindowGeometry(self.data, self.placedb, pos)
        entry = capture_area(self.data, self.placedb, self.virtual_area(model))
        if not entry.fits():
            raise RuntimeError("S5B1 entry exceeds placement capacity")
        with self.electrical_window(), self._sizing_coefficients(model) as coefficients:
            sizing = self.run_sizing(model, pos, geometry, entry.capacity)
            sizing["coefficients"] = coefficients
        sizing_done = time.perf_counter()
        preparation_ms = buffering_ms = 0.0
        buffering = {"status": "disabled", "scheduled": 0, "accepted": 0, "cumulative_count": 0}
        if bool(getattr(self.params, "timing_opt_buffering_enabled", False)):
            self.prepare(model, pos)
            prepared = time.perf_counter()
            with self.electrical_window():
                buffering = self.run_buffering(model, pos, entry.capacity)
            buffering_done = time.perf_counter()
            self.retain_buffered_nets(model)
            preparation_ms = (prepared - sizing_done) * 1000.0
            buffering_ms = (buffering_done - prepared) * 1000.0
        return {
            "iteration": int(iteration),
            "sizing": sizing,
            "buffering": buffering,
            "sizing_ms": (sizing_done - started) * 1000.0,
            "preparation_ms": preparation_ms,
            "buffering_ms": buffering_ms,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        }

    def _sync_native_sizing(self):
        from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
        from dreamplace.ops.routability.gpugr_context import invalidate_route_state

        if str(self.params.place_io_engine) != "ecc":
            raise ValueError("inflation S5B1 requires the ECC native backend")
        targets = self.data.inst_cell_id.detach().cpu()
        changed = int((targets != self.native_cell_ids).sum())
        if not changed:
            return {"ok": True, "accepted_count": 0, "requested_count": 0}
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in self.placedb.flat_libcell_names
        ]
        summary = PlaceIOFunction.apply_sizing(
            self.placedb.rawdb,
            targets.numpy(),
            names,
            current_cell_ids=self.native_cell_ids.numpy(),
        )
        if not summary.get("ok") or int(summary.get("accepted_count", 0)) != changed:
            raise RuntimeError(f"S5B1 native sizing transaction failed: {summary}")
        self.native_cell_ids.copy_(targets)
        invalidate_route_state(self.placedb)
        return summary

    def sync_native_sizing(self):
        return self._sync_native_sizing()

    def prepare_terminal_request(self, model, pos, *, iteration):
        """Materialize all retained counts from the final frame, without reselection."""
        from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
        from dreamplace.ops.buffer_insertion.projection import project_segment_count_buffer_state

        if self.terminal_request is not None:
            raise RuntimeError("S5B1 final materialization was already prepared")
        if not bool(torch.isfinite(pos).all()):
            raise RuntimeError("S5B1 final placement coordinates are non-finite")
        state = None if self.lane is None else self.lane._current_state()
        candidate_mode = state is not None and hasattr(state, "activation_param")
        cumulative_count = 0
        if state is not None:
            count_tensor = state.activation_param if candidate_mode else state.z_param
            cumulative_count = int(count_tensor.detach().sum().cpu().item())
        if self.lane is None:
            self.lane = BufferingOptimizationLane(
                build_buffering_config_from_params(self.params, flow_kind="placement"),
                placedb=self.placedb,
            )
            projection = None
        elif candidate_mode:
            projection = self.lane.last_projection_result
            if projection is None:
                projection = self.lane.project(iteration=iteration, final=True)
        else:
            # Rebuild the ordinary nets and repack current retained references.
            # Frozen buffered nets retain their identity; no new S or B occurs.
            self._prepare(model, pos)
            topology = self.ops.steiner_topo_op
            generation = topology.freeze_topology()
            with torch.no_grad():
                _, x, y = model._timing_geometry_for_pos(pos)
            scale, shift = float(self.params.scale_factor), self.params.shift_factor
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
            payload = self.lane.buffer_relaxed_timing_payload
            projection = project_segment_count_buffer_state(
                self.lane._current_state(),
                nets=payload["nets"],
                legal_table=payload["buffer_legal_table"],
                live_projection_context=context,
            )
            if projection.projected_buffer_count != cumulative_count:
                raise RuntimeError("final projection did not preserve every cumulative buffer")
        self.terminal_request = self.lane.build_commit_request(
            projection,
            iteration=iteration,
            reason="inflation_s5b1_final_commit",
            commit_enabled=True,
            refresh_mode="full_rebuild",
            rebuild_mode="full_rebuild",
        )
        model._timing_geometry_cache = None
        candidate_provider_was_enabled = bool(
            getattr(self.params, "enable_relaxed_buffer_timing", False)
        )
        if candidate_mode:
            # The regular GP loop keeps the expanded candidate provider off,
            # but terminal reporting must include any accepted typed actions.
            self.params.enable_relaxed_buffer_timing = True
        try:
            with self._electrical_window(), torch.no_grad():
                outputs = model.timing_obj(pos)
                loss = model._timing_loss(*outputs)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("S5B1 final virtual timing loss is non-finite")
                gp_timing = dict(
                    zip(
                        ("wns", "tns", "ws", "ts"), (float(value) for value in outputs), strict=True
                    )
                )
                timing = self.ops.timing_propagation_op
                gp_timing["slew_violation_ns"] = timing.last_total_slew_violation
                gp_timing["cap_violation_ff"] = timing.last_total_cap_violation
                gp_timing["objective_terms"] = dict(model.last_timing_objective_terms)
        finally:
            self.params.enable_relaxed_buffer_timing = candidate_provider_was_enabled
        model._timing_geometry_cache = None
        self.terminal_summary = {
            "virtual_count": cumulative_count,
            "scale_factor": float(self.params.scale_factor),
            "dbu": float(self.placedb.dbu),
            "gp_timing": gp_timing,
            "gp_area": capture_area(
                self.data, self.placedb, self.virtual_area(model), capacity=self.capacity
            ).__dict__,
            "request": self.terminal_request.to_summary(),
        }
        return {
            "status": "request_ready",
            "iteration": int(iteration),
            "projected_buffer_count": cumulative_count,
            "commit_request": self.terminal_request.to_summary(),
        }

    def publish_geometry(self, model):
        state = publish_area(
            self.data,
            self.placedb,
            self.virtual_area(model),
            capacity=self.capacity,
        )
        model.refresh_after_geometry_change()
        return state

    def before_inflation(self, model, pos, *, iteration, stage, optimizer=None):
        """Consume a window once per original attempt, before its route/snapshot."""
        schedule = getattr(self, "schedule", None)
        if schedule is not None:
            event = schedule.inflation_event(iteration=iteration, stage=stage)
            if event is None:
                return None
            summary = self._run_window(model, pos, iteration=iteration, stage=stage)
            summary["trigger"] = event
            self._write_summaries()
            return summary
        attempt = (int(stage), int(iteration))
        if (
            self.milestones
            or attempt in self.attempts
            or self.window_count >= self.params.timing_opt_max_windows
        ):
            return None
        self.attempts.add(attempt)
        summary = self._run_window(model, pos, iteration=iteration, stage=stage)
        self._write_summaries()
        return summary

    def before_step(self, model, pos, optimizer, metric, eval_ops, *, iteration, stage):
        """Consume every newly crossed threshold before GP uses the changed objective.

        Pending thresholds belong to this placement owner, so inflation and GP
        restarts cannot rearm them. A jump across several thresholds runs them
        in descending order, including no-op and rolled-back electrical windows.
        """
        overflow = float(metric.overflow[-1])
        schedule = getattr(self, "schedule", None)
        pending = schedule.pending_milestones if schedule is not None else self.pending_milestones
        if not pending or overflow > pending[0]:
            return
        if not callable(getattr(optimizer, "rebase_objective_state", None)):
            raise RuntimeError("S5B1 overflow windows require an optimizer with objective rebasing")
        while pending and overflow <= pending[0]:
            event = (
                schedule.next_overflow_event(overflow, iteration=iteration, stage=stage)
                if schedule is not None
                else {"threshold": pending.pop(0), "overflow": overflow}
            )
            summary = self._run_window(model, pos, iteration=iteration, stage=stage)
            summary["trigger"] = event
            summary["optimizer_rebase"] = optimizer.rebase_objective_state(reason="s5b1_milestone")
        # Resizing also moves lower-left corners and changes density normalization.
        # Stopping/inflation decisions must see the published geometry.
        metric.evaluate(self.placedb, eval_ops, pos, self.data)
        model.overflow = metric.overflow.data.clone()
        self._write_summaries()

    def _run_window(self, model, pos, *, iteration, stage):
        if hasattr(self, "schedule"):
            self.window_count = self.schedule.window_count
        else:
            self.window_count += 1
        before = capture_area(
            self.data, self.placedb, self.virtual_area(model), capacity=self.capacity
        )
        summary = self.run(model, pos, iteration=iteration)
        after = coordinate_area(before, self.data, self.placedb, pos, self.virtual_area(model))
        summary.update(
            window=self.window_count,
            stage=int(stage),
            area={"before": before.__dict__, "after": after.__dict__},
            native_sizing=self.sync_native_sizing(),
        )
        inflation_state = getattr(model, "inflation_state", None)
        if inflation_state is not None and inflation_state.target_area is not None:
            inflation_state.target_area = after.movable + after.filler
        model.refresh_routability_operators()
        model.refresh_after_geometry_change()
        self.summaries.append(summary)
        logging.info(
            "S5B1 window=%d iteration=%d S_best=%d B_accepted=%d count=%d "
            "movable=%g filler=%g capacity=%g density=%g",
            self.window_count,
            iteration,
            summary["sizing"]["best_round"],
            summary["buffering"]["accepted"],
            summary["buffering"]["cumulative_count"],
            after.movable,
            after.filler,
            after.capacity,
            after.density,
        )
        return summary

    def _write_summaries(self):
        path = Path(self.params.result_dir) / f"{self.params.design_name()}_inflation_s5b1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"windows": self.summaries}, indent=2) + "\n")
