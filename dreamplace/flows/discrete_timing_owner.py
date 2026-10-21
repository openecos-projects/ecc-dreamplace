"""Discrete sizing and virtual buffering state shared by timing optimization flows."""

import math

import torch

from dreamplace.flows.candidate_refresh import (
    accepted_candidate_net_ids,
    prepare_candidate_owner,
    refresh_owner_candidate_geometry_after_topology,
    refresh_topology_if_needed,
)
from dreamplace.flows.discrete_sizing_state import SizingState
from dreamplace.flows.discrete_timing_common import (
    segment_counts,
    timing_area_fits,
    timing_electrical_window,
    timing_sizing_coefficients,
    transfer_segment_counts,
    virtual_timing_area,
)
from dreamplace.flows.discrete_timing_diagnostics import objective_snapshot, phase_metrics
from dreamplace.ops.buffer_insertion.buffering_lane import (
    _tensor_digest,
)
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
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.routability.cooptimization_area import (
    capture_area,
    publish_area,
)


class DiscreteTimingOwner:
    """Own electrical actions and terminal state independently of GP scheduling."""

    def __init__(
        self, params, placedb, data, ops, refresh_topology, *, refresh_candidates_after_sizing=False
    ):
        self.params, self.placedb, self.data, self.ops = params, placedb, data, ops
        self.refresh_topology = refresh_topology
        self.oscillation = CellOscillationState()
        self.candidates = build_quad_candidate_table(
            data.flat_libcell_info, data.flat_libcell_leakage
        )
        self.lane = None
        self.round_index = 0
        self.refresh_candidates_after_sizing = refresh_candidates_after_sizing
        self.capacity = capture_area(data, placedb).capacity
        self.native_cell_ids = data.inst_cell_id.detach().cpu().clone()
        self.terminal_request = None
        self.terminal_commit_started = False
        self._topology_geometry_key = None
        self.topology_refresh_count = 0
        self.topology_refresh_reuse_count = 0
        self._pending_dirty_instance_ids = set()
        self._pending_dirty_net_ids = set()
        self.last_candidate_dirty_net_summary = {}
        # ``cooptimization_geometry`` is consumed by the native placement
        # write-back path even when routability inflation is disabled.  The
        # legacy PlaceDataCollection initializer only creates these snapshots
        # for routability runs, so establish the same baseline before exposing
        # the object as shared geometry.  SizingWindowGeometry may update the
        # snapshots for a sizing window; keeping the invariant here prevents a
        # terminal CUDA run from dereferencing a missing attribute.
        if not hasattr(data, "original_node_size_x"):
            data.original_node_size_x = data.node_size_x.detach().clone()
        if not hasattr(data, "original_node_size_y"):
            data.original_node_size_y = data.node_size_y.detach().clone()
        placedb.cooptimization_geometry = data

    def electrical_window(self):
        return timing_electrical_window(self.params, self.data)

    def _sizing_coefficients(self, model):
        return timing_sizing_coefficients(self.params, model)

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
        if self.refresh_candidates_after_sizing:
            pending = getattr(self, "_pending_dirty_instance_ids", None)
            if pending is None:
                pending = self._pending_dirty_instance_ids = set()
            pending.update(int(value) for value in summary.get("applied_instance_ids", ()) or ())
            result = dict(result or {})
            result["candidate_refresh_deferred"] = True
            result["pending_dirty_instance_count"] = len(pending)
            return result
        return result

    def restore_sizing_state(self, model, best, geometry):
        data = self.data
        changed = best.restore_parameters(data)
        if changed.numel():
            self._sync(
                model,
                {
                    "applied_instance_ids": changed.cpu().tolist(),
                    "applied_cell_ids": best.masters[changed].cpu().tolist(),
                },
                geometry,
            )
        pins_changed = best.restore_pin_geometry(data, self.placedb)
        if pins_changed and self.refresh_candidates_after_sizing:
            pending = getattr(self, "_pending_dirty_instance_ids", None)
            if pending is None:
                pending = self._pending_dirty_instance_ids = set()
            pending.update(int(value) for value in changed.detach().cpu().tolist())
        self.oscillation.restore(best.oscillation)
        model._timing_geometry_cache = None

    def _loss(self, model, pos):
        # Both gradient and acceptance use the same model, lane and weights.
        result = model.timing_obj(pos.detach(), surrogate_mode="surrogate_only")
        try:
            model._diff_opt_last_timing_metrics = {
                "wns": float(result[0].detach()),
                "tns": float(result[1].detach()),
            }
        except (TypeError, ValueError, RuntimeError, IndexError):
            model._diff_opt_last_timing_metrics = {}
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

    def prepare(
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
            refresh_records_all=refresh_records_all,
            reason="s5b1_live_topology_refresh",
        )

    def refresh_candidate_geometry_after_topology(self, model, *, reason):
        """Repack retained candidate RC after a live topology publication."""

        return refresh_owner_candidate_geometry_after_topology(
            self,
            model,
            reason=reason,
        )

    def run_sizing(self, model, pos, geometry, capacity):
        data = self.data
        size, vt = data.get_continuous_size_parameter(), data.vt_logits
        best = SizingState.capture(data, self.oscillation)
        loss = self._loss(model, pos)
        initial_loss = best_loss = float(loss.detach())
        initial_objective = objective_snapshot(model, loss)
        best_objective = dict(initial_objective)
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
                candidate_table=self.candidates,
                shared_budget_percent=float(
                    self.params.discrete_gradient_topk_shared_budget_percent
                ),
                blocked_instances=blocked,
                size_parameterization="real_size",
            )
            if not summary["num_changed_instances"]:
                rounds.append(
                    {
                        "round": round_id,
                        "loss": float(loss.detach()),
                        "fits": self._fits(model, capacity),
                        "actions": summary.get("selected_scores", ()),
                        "action_counts": dict(summary.get("action_counts", {}) or {}),
                        "action_digest": summary.get("action_digest"),
                        "input_state_digest": summary.get("input_state_digest"),
                        "output_state_digest": summary.get("output_state_digest"),
                        "changed_instance_count": 0,
                        "no_action": True,
                    }
                )
                # Consume the no-op attempt while rebuilding the graph for the
                # next fixed-budget sizing round.
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
                    "objective_terms": objective_snapshot(model, loss),
                    "fits": fits,
                    "actions": summary["selected_scores"],
                    "action_counts": dict(summary.get("action_counts", {}) or {}),
                    "action_digest": summary.get("action_digest"),
                    "input_state_digest": summary.get("input_state_digest"),
                    "output_state_digest": summary.get("output_state_digest"),
                    "changed_instance_count": int(summary.get("num_changed_instances", 0) or 0),
                    "applied_instance_ids": list(summary.get("applied_instance_ids", ()) or ()),
                    "applied_cell_ids": list(summary.get("applied_cell_ids", ()) or ()),
                    "runtime_cell_state_generation": summary.get("runtime_cell_state_generation"),
                    "timing_cache_generation": summary.get("timing_cache_generation"),
                }
            )
            if fits and actual < best_loss:
                best, best_loss, best_round = (
                    SizingState.capture(data, self.oscillation),
                    actual,
                    round_id,
                )
                best_objective = objective_snapshot(model, loss)
        self.restore_sizing_state(model, best, geometry)
        with torch.no_grad():
            restored_loss = self._loss(model, pos)
        if not math.isclose(float(restored_loss), best_loss, rel_tol=1e-5, abs_tol=1e-4):
            raise RuntimeError(
                "sizing restore did not reproduce the phase best objective: "
                f"best={best_loss} restored={float(restored_loss)}"
            )
        restored_objective = objective_snapshot(model, restored_loss)
        return {
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_round": best_round,
            "initial_objective": initial_objective,
            "best_objective": best_objective,
            "restore": {
                "status": "restored",
                "best_round": int(best_round),
                "best_score": float(best_loss),
                "objective": best_objective,
                "restored_objective": restored_objective,
            },
            "phase_metrics": {
                "before": phase_metrics(initial_objective),
                "after": phase_metrics(restored_objective),
            },
            "rounds": rounds,
        }

    def run_buffering(self, model, pos, capacity):
        self.lane.timing_loss = model._timing_loss
        state = self.lane._current_state()
        if hasattr(state, "activation_param"):
            from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
                run_discrete_candidate_buffer_scheduler,
            )

            activation = state.activation_param
            activation_before = activation.detach().clone()
            requires_grad = activation.requires_grad
            activation.requires_grad_(requires_grad=True)
            try:
                loop_summary = run_discrete_candidate_buffer_scheduler(
                    lane=self.lane,
                    model=model,
                    rounds=int(getattr(self.lane.config, "continuous_steps", 0)),
                    allow_existing_activation=True,
                )
            finally:
                activation.requires_grad_(requires_grad)
            count = int(state.activation_param.detach().sum().cpu().item())
            accepted_net_ids = accepted_candidate_net_ids(
                activation_before,
                state,
            )
            pending = getattr(self, "_pending_dirty_net_ids", None)
            if pending is None:
                pending = self._pending_dirty_net_ids = set()
            pending.update(accepted_net_ids)
            terminal = dict(loop_summary.terminal_metrics or {})
            restore = dict(loop_summary.best_restore or {})
            result = {
                "initial_loss": float(restore["initial_snapshot"]["loss"]),
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
                "best_restore": restore,
            }
            result["phase_metrics"] = {
                "before": phase_metrics(restore.get("initial_snapshot")),
                "after": phase_metrics(restore.get("best_snapshot")),
            }
            return result
        before = state.z_param.detach().clone()
        state.z_param.requires_grad_(requires_grad=True)
        try:
            loss = self._loss(model, pos)
            initial = float(loss.detach())
            initial_objective = objective_snapshot(model, loss)
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
        accepted = 0
        for count in transition_backtracking_sizes(transition.accepted_count):
            trial = take_transition_prefix(transition, counts=before, selected_count=count)
            with torch.no_grad():
                state.z_param.copy_(trial.next_counts)
            if not self._fits(model, capacity):
                continue
            with torch.no_grad():
                actual = float(self._loss(model, pos))
            if actual < initial:
                accepted = count
                break
        if not accepted:
            with torch.no_grad():
                state.z_param.copy_(before)
        model._timing_geometry_cache = None
        with torch.no_grad():
            final_loss = self._loss(model, pos)
        final_objective = objective_snapshot(model, final_loss)
        return {
            "initial_loss": initial,
            "final_loss": float(final_loss.detach()),
            "scheduled": transition.accepted_count,
            "accepted": accepted,
            "gradient_norm": float(gradient.norm()),
            "cumulative_count": int(state.z_param.detach().sum()),
            "phase_metrics": {
                "before": phase_metrics(initial_objective),
                "after": phase_metrics(final_objective),
            },
        }

    def retain_buffered_nets(self, model):
        """Retain virtual segment or candidate timing state for ordinary GP."""
        state = self.lane._current_state()
        if hasattr(state, "activation_param"):
            activation = state.activation_param.detach()
            active = activation > 0 if activation.ndim == 1 else activation.sum(dim=1) > 0
            net_ids = state.candidate_net_id[active].cpu().tolist()
            if net_ids:
                topo = self.ops.steiner_topo_op
                if not bool(getattr(topo, "topology_frozen", False)):
                    topo.freeze_nets(net_ids)
                    # FrozenNets remaps retained candidate records while the new
                    # Steiner vertex domain is published.  Install the old state
                    # before refreshing so that its node identities are
                    # translated before the payload is rebuilt below.
                    topo.frozen_nets.candidate_state = state
                    # freeze_nets changes the live vertex/index domain.  Publish
                    # that domain before rebuilding the candidate payload; using
                    # the pre-freeze payload leaves compact_node_to_original
                    # pointing at removed Steiner ids after GP restarts.
                    pos = self.data.pos[0]
                    self._candidate_refresh_in_progress = True
                    try:
                        refresh_topology_if_needed(self, pos, force=True)
                    finally:
                        self._candidate_refresh_in_progress = False
                    self.lane.refresh_candidate_geometry(
                        model=model,
                        params=self.params,
                        # Keep the complete candidate search domain after the
                        # accepted buffers are frozen. Ordinary GP still
                        # evaluates only accepted nets through
                        # timing_active_only below.
                        active_only=False,
                        reason="retain_accepted_candidate_nets",
                        dirty_net_ids=net_ids,
                    )
                    topo.frozen_nets.candidate_state = self.lane._current_state()
                # A frozen topology is the diff-opt contract: accepted actions
                # are virtual activation changes, not a new vertex domain.
                # Re-freezing here would change the structure signature and
                # force the exact candidate-state rebuild this path avoids.
            retained_activation = self.lane._current_state().activation_param
            retained_activation.requires_grad_(requires_grad=False)
            retained_activation.grad = None
            self.lane._current_state().timing_active_only = True
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

    def sync_native_sizing(self):
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

    def prepare_terminal_request(self, model, pos, *, iteration, refresh_geometry=True):
        from dreamplace.flows.discrete_buffer_commit import prepare_terminal_request

        return prepare_terminal_request(
            self,
            model,
            pos,
            iteration=iteration,
            refresh_geometry=refresh_geometry,
        )

    def publish_geometry(self, model):
        state = publish_area(
            self.data,
            self.placedb,
            self.virtual_area(model),
            capacity=self.capacity,
        )
        model.refresh_after_geometry_change()
        return state
