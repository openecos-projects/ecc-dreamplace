"""Discrete sizing and virtual buffering state shared by timing optimization flows."""

import math
import time

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
    ACTION_NAMES,
    apply_quad_gradient_from_action_scores,
    apply_quad_gradient_from_data_collections,
    build_quad_candidate_table,
)
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState
from dreamplace.ops.discrete_gradient_topk.runtime_cells import sync_runtime_cells
from dreamplace.ops.timing_propagation.action_cell_aat import (
    refresh_action_valid_instances,
)
from dreamplace.ops.timing_propagation.discrete_actions import (
    build_discrete_action_context,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.routability.cooptimization_area import (
    capture_area,
    publish_area,
)


def _term_delta(after, before):
    """Return ``after - before`` for an optional scalar diagnostic term."""

    if after is None or before is None:
        return None
    return float(after) - float(before)


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
        self.action_mode = str(
            getattr(params, "timing_action_gradient_mode", "off") or "off"
        )
        if self.action_mode not in ("off", "lut_action"):
            raise ValueError(
                "timing_action_gradient_mode must be one of: off, lut_action"
            )
        self._last_action_context = None
        # Mapping tensors are detached runtime state.  A new beta/context is
        # still created for every forward, while unchanged rounds reuse the
        # expensive action arc and input-cap tables.
        self._action_context_cache = {}
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
            pin_offset_mode="delta",
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

    def _loss(self, model, pos, action_context=None):
        # Both gradient and acceptance use the same model, lane and weights.
        if action_context is not None:
            result = model.timing_obj(
                pos.detach(),
                surrogate_mode="lut_only",
                discrete_action_context=action_context,
            )
        elif self.action_mode == "lut_action":
            result = model.timing_obj(pos.detach(), surrogate_mode="lut_only")
        else:
            result = model.timing_obj(pos.detach(), surrogate_mode="surrogate_only")
        try:
            model._diff_opt_last_timing_metrics = {
                "wns": float(result[0].detach()),
                "tns": float(result[1].detach()),
            }
        except (TypeError, ValueError, RuntimeError, IndexError):
            model._diff_opt_last_timing_metrics = {}
        if action_context is not None:
            action_context.tns_score_loss = -result[1]
            action_context.wns_score_loss = -result[0]
            objective = str(
                getattr(self.params, "timing_action_score_objective", "tns")
            )
            if objective == "tns":
                margin = float(
                    getattr(self.params, "timing_action_tns_margin_ps", 0.0)
                    or 0.0
                )
                timing_op = getattr(
                    getattr(model, "op_collections", None),
                    "timing_propagation_op",
                    None,
                )
                slack = (
                    None
                    if timing_op is None
                    else getattr(timing_op, "last_endpoint_slack_tensor", None)
                )
                if margin > 0.0 and slack is not None:
                    # Endpoints inside the margin keep a unit slope, so an
                    # action that pushes a near-critical path negative is
                    # visible instead of falling into the clamp's zero region.
                    action_context.score_loss = -torch.clamp(
                        slack - margin, max=0.0
                    ).sum()
                else:
                    action_context.score_loss = action_context.tns_score_loss
            elif objective == "wns":
                action_context.score_loss = action_context.wns_score_loss
            else:
                action_context.score_loss = None
        loss = model._timing_loss(*result)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("S5B1 electrical loss is non-finite")
        return loss

    def _action_loss(self, model, pos):
        map_started_at = time.perf_counter()
        timing_op = self.ops.timing_propagation_op
        action_device = torch.device(getattr(timing_op, "device", pos.device))
        context = build_discrete_action_context(
            self.data,
            self.candidates,
            action_names=ACTION_NAMES,
            pin_names=(
                getattr(self.data, "flat_lib_pin_names", None)
                or getattr(self.placedb, "flat_lib_pin_names", None)
            ),
            device=action_device,
            dtype=pos.dtype,
            requires_grad=True,
            load_delta=bool(
                getattr(self.params, "timing_action_load_delta", 0)
            ),
            drive_load_feedback=bool(
                getattr(self.params, "timing_action_drive_load_feedback", 1)
            ) and bool(getattr(self.params, "timing_action_pin_cap_gradient", 1)),
            cache=self._action_context_cache,
        )
        context.mapping_summary["build_ms"] = (
            time.perf_counter() - map_started_at
        ) * 1000.0
        table_timing = dict(context.mapping_summary.get("timing_ms", {}) or {})
        context.mapping_summary["action_table_mapping_ms"] = float(
            context.mapping_summary.get(
                "action_table_total_ms", table_timing.get("total", 0.0)
            )
            or 0.0
        )
        context.mapping_summary["action_table_build_ms"] = (
            0.0
            if bool(context.mapping_summary.get("cache_hit", False))
            else float(
                context.mapping_summary.get(
                    "action_table_total_ms", table_timing.get("total", 0.0)
                )
                or 0.0
            )
        )
        if not context.mapping_summary["pin_identity_checked"]:
            raise RuntimeError(
                "lut_action requires Liberty pin names in cap-table order; "
                "rebuild the active placeio/ECC-Tools exporter"
            )
        forward_started_at = time.perf_counter()
        loss = self._loss(model, pos, action_context=context)
        context.mapping_summary["loss_forward_ms"] = (
            time.perf_counter() - forward_started_at
        ) * 1000.0
        self._last_action_context = context
        return loss, context

    def _probe_action_chain(
        self,
        model,
        pos,
        context,
        inst,
        action,
        old_cell,
        geometry,
        pending_snapshot,
    ):
        """Decompose one trial candidate into its measurable channels.

        ``base_recheck`` measures the drift accumulated since the round base,
        ``noop`` re-applies the current master, ``beta`` runs the forward-only
        pin-cap channel at ``beta=1``, and ``restored`` checks that the trial
        apply/restore cycle is reversible.
        """

        probe = {}
        def _tns():
            metrics = getattr(model, "_diff_opt_last_timing_metrics", {}) or {}
            value = metrics.get("tns")
            return None if value is None else float(value)

        def _restore_pending():
            if pending_snapshot is None:
                if hasattr(self, "_pending_dirty_instance_ids"):
                    del self._pending_dirty_instance_ids
            else:
                self._pending_dirty_instance_ids = set(pending_snapshot)

        self._loss(model, pos)
        probe["base_recheck_tns"] = _tns()
        probe["base_recheck_hash"] = self._timing_output_hash(model)
        self._sync(
            model,
            {
                "applied_instance_ids": [inst],
                "applied_cell_ids": [old_cell],
            },
            geometry,
        )
        try:
            self._loss(model, pos)
            probe["noop_tns"] = _tns()
            probe["noop_hash"] = self._timing_output_hash(model)
        finally:
            self._sync(
                model,
                {
                    "applied_instance_ids": [inst],
                    "applied_cell_ids": [old_cell],
                },
                geometry,
            )
            _restore_pending()
        from .action_gradient_debug import probe_action_interpolation

        def evaluate():
            self._loss(model, pos, action_context=context)
            return dict(model._diff_opt_last_timing_metrics)

        try:
            probe["interpolation"] = probe_action_interpolation(
                evaluate, context, inst, action
            )
            probe["beta_tns"] = probe["interpolation"][-1]["tns"]
        finally:
            _restore_pending()
        self._probe_snapshot = {"state_hash": self._timing_state_hash(pos)}
        return probe

    def _timing_output_hash(self, model):
        """Hash the forward's RC and endpoint outputs, not just the inputs."""

        import hashlib

        digest = hashlib.sha1()
        timing_op = getattr(
            getattr(model, "op_collections", None),
            "timing_propagation_op",
            None,
        )
        for name in ("last_endpoint_slack_tensor", "last_endpoint_ids_tensor"):
            value = getattr(timing_op, name, None)
            if value is None:
                continue
            digest.update(
                value.detach().cpu().contiguous().numpy().tobytes()
            )
        for name in (
            "pin_net_cap_rise_live",
            "pin_net_cap_fall_live",
            "pin_net_delay_rise_live",
            "pin_net_delay_fall_live",
            "pin_rAAT_live_snapshot",
            "pin_fAAT_live_snapshot",
        ):
            value = getattr(timing_op, name, None)
            if value is None:
                continue
            digest.update(
                value.detach().cpu().contiguous().numpy().tobytes()
            )
        return digest.hexdigest()

    def _timing_state_hash(self, pos):
        import hashlib

        data = self.data
        digest = hashlib.sha1()
        for name in (
            "inst_cell_id",
            "inst_libcell_offset",
            "flat_inst_arcs_by_level",
            "endpoints_constraint_arcs",
            "pin_offset_x",
            "pin_offset_y",
            "node_size_x",
            "node_size_y",
        ):
            value = getattr(data, name, None)
            if value is None:
                continue
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(array.tobytes())
        digest.update(pos.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _record_restore_integrity(self, inst, pos, probe):
        """Record whether the trial apply/restore left the timing state intact."""

        snapshot = getattr(self, "_probe_snapshot", None)
        if snapshot is None:
            return
        self._probe_snapshot = None
        probe["state_hash_equal"] = int(
            self._timing_state_hash(pos) == snapshot["state_hash"]
        )

    def _trial_action_features(self, model, context, inst, old_cell, target_cell):
        """Electrical features for one debug trial record."""

        timing_op = getattr(
            getattr(model, "op_collections", None),
            "timing_propagation_op",
            None,
        )
        if timing_op is None:
            return {}
        data = self.data
        arcs = data.flat_inst_arcs_by_level.long()
        arc_inst = context.inst_ids.long()
        num_inst = int(context.beta.shape[0])
        arc_rows = torch.arange(
            int(arcs.shape[0]),
            device=arc_inst.device,
            dtype=torch.long,
        )
        valid_arc = (arc_inst >= 0) & (arc_inst < num_inst)
        first_arc = torch.full(
            (num_inst,),
            int(arcs.shape[0]),
            device=arc_inst.device,
            dtype=torch.long,
        )
        first_arc.scatter_reduce_(
            0,
            arc_inst[valid_arc],
            arc_rows[valid_arc],
            reduce="amin",
            include_self=True,
        )
        row = int(first_arc[inst].item())
        if row >= int(arcs.shape[0]):
            return {}
        in_pin, out_pin = (int(arcs[row, 0].item()), int(arcs[row, 1].item()))
        pin_net = timing_op.pin_net.long()
        net = int(pin_net[out_pin].item())
        if net < 0:
            return {}
        fanout = int(
            torch.bincount(
                pin_net.clamp(min=0),
                minlength=net + 1,
            )[net].item()
        )

        def _pin_value(name):
            value = getattr(timing_op, name, None)
            if (
                not torch.is_tensor(value)
                or value.numel() <= max(in_pin, out_pin)
            ):
                return None
            return value

        load_r = _pin_value("pin_net_cap_rise_live")
        load_f = _pin_value("pin_net_cap_fall_live")
        slew_r = _pin_value("pin_rtran_live")
        slew_f = _pin_value("pin_ftran_live")
        slack = _pin_value("pin_slack_live_snapshot")
        output_load = None
        if load_r is not None and load_f is not None:
            output_load = float(
                torch.maximum(
                    load_r[out_pin].detach(),
                    load_f[out_pin].detach(),
                )
            )
        input_slew = None
        if slew_r is not None and slew_f is not None:
            input_slew = float(
                torch.maximum(
                    slew_r[in_pin].detach(),
                    slew_f[in_pin].detach(),
                )
            )
        output_slack = (
            None
            if slack is None
            else float(slack[out_pin].detach())
        )
        flat_info = data.flat_libcell_info
        old_size = float(flat_info[old_cell, 2].detach())
        target_size = float(flat_info[target_cell, 2].detach())
        features = {
            "fanout": fanout,
            "output_load_pf": output_load,
            "input_slew_ps": input_slew,
            "output_slack_ps": output_slack,
            "current_size": old_size,
            "target_size": target_size,
            "size_ratio": (
                None if old_size == 0.0 else target_size / old_size
            ),
        }
        if bool(
            getattr(self.params, "timing_action_trial_reachability", 0)
        ):
            features.update(
                self._trial_reachability_features(timing_op, out_pin)
            )
        return features

    def _trial_reachability_features(self, timing_op, out_pin):
        """Count endpoints reachable from one output pin (debug only)."""

        succ_start = getattr(timing_op, "pin_succ_start", None)
        succ_pin = getattr(timing_op, "pin_succ_pin", None)
        endpoints = getattr(timing_op, "end_points", None)
        slack = getattr(timing_op, "pin_slack_live_snapshot", None)
        if any(
            value is None
            for value in (succ_start, succ_pin, endpoints, slack)
        ):
            return {}
        cache = getattr(timing_op, "_action_reachability_cache", None)
        if cache is None:
            cache = (
                succ_start.detach().cpu().tolist(),
                succ_pin.detach().cpu().tolist(),
                set(int(value) for value in endpoints.detach().cpu().tolist()),
                slack.detach().cpu(),
            )
            timing_op._action_reachability_cache = cache
        starts, pins, endpoint_set, endpoint_slack = cache
        num_pins = int(timing_op.num_pins)
        visited = bytearray(num_pins)
        stack = [out_pin]
        reachable = 0
        critical = 0
        while stack:
            pin = stack.pop()
            if pin < 0 or pin >= num_pins or visited[pin]:
                continue
            visited[pin] = 1
            if pin in endpoint_set:
                reachable += 1
                if float(endpoint_slack[pin]) < 0.0:
                    critical += 1
                continue
            for index in range(starts[pin], starts[pin + 1]):
                stack.append(pins[index])
        return {
            "reachable_endpoints": reachable,
            "critical_endpoints": critical,
        }

    def _trial_tns_action_scores(
        self,
        model,
        pos,
        context,
        local_scores,
        geometry,
        base_loss,
        *,
        action_mask=None,
        selection_scores=None,
        wns_scores=None,
    ):
        """Record actual one-action TNS/WNS deltas for a debug shortlist.

        This is diagnostic only: the caller keeps the local ``dL/dbeta``
        scores for selection.  Each candidate is applied alone, evaluated on
        the real timing graph, and restored.
        """

        data = self.data
        valid = context.valid_instances.bool()
        if action_mask is not None:
            mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=valid.device,
            )
            valid = valid & mask.unsqueeze(0)
        if selection_scores is None:
            selection_scores = local_scores
        scores = torch.where(
            valid,
            selection_scores,
            torch.full_like(selection_scores, float("inf")),
        )
        action_count = int(scores.shape[1])
        sizeable_count = int(data.inst_is_sizeable.sum().item())
        percent = float(self.params.discrete_gradient_topk_shared_budget_percent)
        budget = (
            max(1, int(math.ceil(sizeable_count * percent / 100.0)))
            if percent > 0.0
            else 0
        )
        configured = int(
            getattr(self.params, "timing_action_trial_candidates", 0) or 0
        )
        limit = configured if configured > 0 else max(2 * budget, budget + 16)
        limit = max(0, min(limit, int(torch.isfinite(scores).sum().item())))
        metrics = getattr(model, "_diff_opt_last_timing_metrics", {}) or {}
        base_tns = metrics.get("tns")
        if base_tns is None:
            self._loss(model, pos)
            base_tns = (
                getattr(model, "_diff_opt_last_timing_metrics", {}) or {}
            ).get("tns")
        if base_tns is None:
            raise RuntimeError("trial_tns requires a timing TNS metric")
        base_tns = float(base_tns)
        base_wns = metrics.get("wns")
        if base_wns is None:
            self._loss(model, pos)
            base_wns = (
                getattr(model, "_diff_opt_last_timing_metrics", {}) or {}
            ).get("wns")
        records = []
        record_enabled = True
        probe_limit = int(
            getattr(self.params, "timing_action_trial_probe", 0) or 0
        )
        probed = 0
        pending = getattr(self, "_pending_dirty_instance_ids", None)
        pending_snapshot = None if pending is None else set(pending)
        trial_state = SizingState.capture(data, self.oscillation)
        trial_pos = pos.detach().clone()
        started_at = time.perf_counter()
        evaluated = 0
        if limit:
            top_per_action = int(
                getattr(
                    self.params,
                    "timing_action_trial_top_per_action",
                    0,
                )
                or 0
            )
            sample_per_action = int(
                getattr(self.params, "timing_action_trial_sample", 0) or 0
            )
            if top_per_action > 0 or sample_per_action > 0:
                generator = torch.Generator().manual_seed(
                    int(
                        getattr(
                            self.params,
                            "timing_action_trial_seed",
                            0,
                        )
                        or 0
                    )
                )
                top = []
                valid_cpu = valid.detach().cpu()
                scores_cpu = scores.detach().cpu()
                for action in range(action_count):
                    finite = torch.isfinite(scores_cpu[:, action])
                    valid_indices = torch.nonzero(
                        valid_cpu[:, action] & finite, as_tuple=False
                    ).flatten()
                    chosen = []
                    if top_per_action > 0 and valid_indices.numel():
                        count = min(top_per_action, int(valid_indices.numel()))
                        ranked = torch.topk(
                            scores_cpu[valid_indices, action],
                            k=count,
                            largest=False,
                        ).indices
                        chosen.extend(valid_indices[ranked].tolist())
                    if sample_per_action > 0:
                        chosen_set = set(chosen)
                        remaining = [
                            int(index)
                            for index in valid_indices.tolist()
                            if int(index) not in chosen_set
                        ]
                        if remaining:
                            count = min(sample_per_action, len(remaining))
                            order = torch.randperm(
                                len(remaining),
                                generator=generator,
                            )[:count]
                            chosen.extend(remaining[int(i)] for i in order)
                    top.extend(
                        (index * action_count + action) for index in chosen
                    )
            elif bool(getattr(self.params, "timing_action_trial_stratified", 0)):
                per_action = max(1, limit // action_count)
                top = []
                for action in range(action_count):
                    column = scores[:, action]
                    finite = torch.isfinite(column)
                    count = min(per_action, int(finite.sum().item()))
                    if count:
                        ranked = torch.topk(
                            torch.where(
                                finite,
                                column,
                                torch.full_like(column, float("inf")),
                            ),
                            k=count,
                            largest=False,
                        ).indices
                        top.extend(
                            (ranked * action_count + action).tolist()
                        )
            else:
                flat = scores.reshape(-1)
                top = torch.topk(flat, k=limit, largest=False).indices.tolist()
            for flat_index in top:
                inst = int(flat_index) // action_count
                action = int(flat_index) % action_count
                if not bool(valid[inst, action]):
                    continue
                target_cell = int(
                    context.target_cell_ids[inst, action].item()
                )
                if target_cell < 0:
                    continue
                old_cell = int(data.inst_cell_id[inst].item())
                if target_cell == old_cell:
                    continue
                features = self._trial_action_features(
                    model,
                    context,
                    inst,
                    old_cell,
                    target_cell,
                )
                local_score = float(local_scores[inst, action])
                selection_score = float(selection_scores[inst, action])
                local_wns_score = (
                    None
                    if wns_scores is None
                    else float(wns_scores[inst, action])
                )
                probe = None
                probe_stride = max(1, limit // max(probe_limit, 1))
                if (
                    probe_limit
                    and probed < probe_limit
                    and evaluated % probe_stride == 0
                ):
                    probed += 1
                    probe = self._probe_action_chain(
                        model,
                        pos,
                        context,
                        inst,
                        action,
                        old_cell,
                        geometry,
                        pending_snapshot,
                    )
                self._sync(
                    model,
                    {
                        "applied_instance_ids": [inst],
                        "applied_cell_ids": [target_cell],
                    },
                    geometry,
                )
                try:
                    # Exact trials only consume scalar TNS/WNS values.  Avoid
                    # building an autograd graph for every candidate.
                    with torch.no_grad():
                        trial_loss = self._loss(model, pos)
                    trial_loss_value = float(trial_loss.detach())
                    trial_metrics = (
                        getattr(model, "_diff_opt_last_timing_metrics", {}) or {}
                    )
                    trial_tns = trial_metrics.get("tns")
                    trial_wns = trial_metrics.get("wns")
                    if trial_tns is None:
                        raise RuntimeError("trial_tns produced no TNS metric")
                    trial_features = self._trial_action_features(
                        model,
                        context,
                        inst,
                        target_cell,
                        target_cell,
                    )
                    evaluated += 1
                    if record_enabled:
                        def _cell_name(cell_id):
                            names = getattr(
                                self.placedb,
                                "flat_libcell_names",
                                None,
                            )
                            if (
                                names is None
                                or cell_id < 0
                                or cell_id >= len(names)
                            ):
                                return None
                            value = names[cell_id]
                            return (
                                value.decode("utf-8", errors="replace")
                                if isinstance(value, bytes)
                                else str(value)
                            )

                        records.append(
                            {
                                "instance_id": int(inst),
                                "action": ACTION_NAMES[int(action)],
                                "local_score": local_score,
                                "selection_score": selection_score,
                                "local_wns_score": local_wns_score,
                                "old_cell": int(old_cell),
                                "target_cell": int(target_cell),
                                "old_cell_name": _cell_name(int(old_cell)),
                                "target_cell_name": _cell_name(int(target_cell)),
                                "base_tns": base_tns,
                                "trial_tns": float(trial_tns),
                                "delta_tns": base_tns - float(trial_tns),
                                "base_wns": base_wns,
                                "trial_wns": (
                                    None if trial_wns is None else float(trial_wns)
                                ),
                                "delta_wns": (
                                    None
                                    if base_wns is None or trial_wns is None
                                    else float(trial_wns) - float(base_wns)
                                ),
                                "delta_wns_loss": (
                                    None
                                    if base_wns is None or trial_wns is None
                                    else float(base_wns) - float(trial_wns)
                                ),
                                "base_loss": float(base_loss),
                                "trial_loss": trial_loss_value,
                                "delta_loss": float(base_loss) - trial_loss_value,
                                "features": features,
                                "trial_features": trial_features,
                                **({"probe": probe} if probe is not None else {}),
                            }
                        )
                finally:
                    self.restore_sizing_state(model, trial_state, geometry)
                    with torch.no_grad():
                        pos.copy_(trial_pos)
                    if probe is not None:
                        self._loss(model, pos)
                        probe["restored_hash"] = self._timing_output_hash(model)
                        probe["restored_tns"] = (
                            getattr(
                                model, "_diff_opt_last_timing_metrics", {}
                            )
                            or {}
                        ).get("tns")
                        self._loss(model, pos)
                        probe["restored2_tns"] = (
                            getattr(
                                model, "_diff_opt_last_timing_metrics", {}
                            )
                            or {}
                        ).get("tns")
                        self._record_restore_integrity(inst, pos, probe)
        if pending_snapshot is None:
            if hasattr(self, "_pending_dirty_instance_ids"):
                del self._pending_dirty_instance_ids
        else:
            self._pending_dirty_instance_ids = pending_snapshot
        accuracy = {}
        for action_name in ACTION_NAMES:
            rows = [
                record
                for record in records
                if record["action"] == action_name
            ]
            nonzero = [
                record
                for record in rows
                if abs(float(record["delta_tns"])) > 1.0e-9
            ]
            sign_ok = sum(
                1
                for record in nonzero
                if float(record["local_score"]) != 0.0
                and (float(record["local_score"]) < 0.0)
                == (float(record["delta_tns"]) < 0.0)
            )
            sign_zero = sum(
                1
                for record in nonzero
                if float(record["local_score"]) == 0.0
            )
            sign_bad = sum(
                1
                for record in nonzero
                if float(record["local_score"]) != 0.0
                and (float(record["local_score"]) < 0.0)
                != (float(record["delta_tns"]) < 0.0)
            )
            abs_errors = sorted(
                abs(
                    float(record["local_score"])
                    - float(record["delta_tns"])
                )
                for record in nonzero
            )
            rel_errors = sorted(
                abs(
                    float(record["local_score"])
                    - float(record["delta_tns"])
                )
                / max(abs(float(record["delta_tns"])), 1.0)
                for record in nonzero
            )
            accuracy[action_name] = {
                "sample_count": len(rows),
                "nonzero_exact_count": len(nonzero),
                "sign_ok": int(sign_ok),
                "sign_zero": int(sign_zero),
                "sign_bad": int(sign_bad),
                "median_abs_error": (
                    abs_errors[len(abs_errors) // 2] if abs_errors else None
                ),
                "median_relative_error": (
                    rel_errors[len(rel_errors) // 2] if rel_errors else None
                ),
                "exact_improve_100_count": sum(
                    1
                    for record in rows
                    if float(record["delta_tns"]) < -100.0
                ),
                "exact_worsen_100_count": sum(
                    1
                    for record in rows
                    if float(record["delta_tns"]) > 100.0
                ),
                "local_zero_worsen_100_count": sum(
                    1
                    for record in rows
                    if float(record["local_score"]) == 0.0
                    and float(record["delta_tns"]) > 100.0
                ),
            }
        return {
            "mode": "debug_record",
            "trial_limit": int(limit),
            "trial_count": int(evaluated),
            "trial_ms": float((time.perf_counter() - started_at) * 1000.0),
            "trial_base_tns": base_tns,
            "trial_budget": int(budget),
            "accuracy": accuracy,
            **({"trial_records": records} if record_enabled else {}),
        }

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

    def run_sizing(self, model, pos, geometry, capacity, *, action_mask=None):
        data = self.data
        size, vt = data.get_continuous_size_parameter(), data.vt_logits
        sizing_started_at = time.perf_counter()
        capture_started_at = time.perf_counter()
        best = SizingState.capture(data, self.oscillation)
        capture_ms = [(time.perf_counter() - capture_started_at) * 1000.0]
        forward_ms = []
        timing_forward_ms = []
        backward_ms = []
        beta_backward_ms = []
        action_build_ms = []
        action_table_build_ms = []
        action_table_mapping_ms = []
        action_map_timing = []
        select_ms = []
        candidate_select_ms = []
        sync_ms = []
        state_sync_ms = []
        fits_ms = []
        blocked_ms = []
        record_ms = []
        snapshot_ms = []
        started_at = time.perf_counter()
        if self.action_mode == "lut_action":
            loss, action_context = self._action_loss(model, pos)
            action_build_ms.append(
                float(action_context.mapping_summary.get("build_ms", 0.0) or 0.0)
            )
            action_table_build_ms.append(
                float(
                    action_context.mapping_summary.get(
                        "action_table_build_ms", 0.0
                    )
                    or 0.0
                )
            )
            action_table_mapping_ms.append(
                float(
                    action_context.mapping_summary.get(
                        "action_table_mapping_ms", 0.0
                    )
                    or 0.0
                )
            )
            action_map_timing.append(
                {
                    "build_ms": float(
                        action_context.mapping_summary.get("build_ms", 0.0) or 0.0
                    ),
                    "timing_ms": dict(
                        action_context.mapping_summary.get("timing_ms", {}) or {}
                    ),
                    "pair_cache": dict(
                        action_context.mapping_summary.get("pair_cache", {}) or {}
                    ),
                }
            )
        else:
            loss = self._loss(model, pos)
            action_context = None
        initial_forward_ms = (time.perf_counter() - started_at) * 1000.0
        initial_timing_forward_ms = float(
            action_context.mapping_summary.get("loss_forward_ms", initial_forward_ms)
            if action_context is not None
            else initial_forward_ms
        )
        initial_action_table_build_ms = float(
            action_context.mapping_summary.get("action_table_build_ms", 0.0)
            if action_context is not None
            else 0.0
        )
        initial_action_table_mapping_ms = float(
            action_context.mapping_summary.get("action_table_mapping_ms", 0.0)
            if action_context is not None
            else 0.0
        )
        initial_action_mapping = None
        if action_context is not None:
            initial_action_mapping = dict(action_context.mapping_summary)
            initial_action_mapping["native_values_used"] = bool(
                action_context.native_values_used
            )
            initial_action_mapping["native_action_values_used"] = bool(
                action_context.native_action_values_used
            )
            initial_action_mapping["action_cell_aat_calls"] = int(
                action_context.action_cell_aat_calls
            )
            initial_action_mapping["pin_cap_gradient"] = bool(
                int(getattr(self.params, "timing_action_pin_cap_gradient", 1))
            )
            initial_action_mapping["drive_load_feedback"] = bool(
                int(getattr(self.params, "timing_action_drive_load_feedback", 1))
            )
            timing_op = self.ops.timing_propagation_op
            initial_action_mapping["runtime_context"] = {
                **dict(getattr(timing_op, "_lut_action_placeobj_debug", {}) or {}),
                **dict(getattr(timing_op, "_lut_action_runtime_debug", {}) or {}),
            }
        initial_loss = best_loss = float(loss.detach())
        snapshot_started_at = time.perf_counter()
        initial_objective = objective_snapshot(model, loss)
        snapshot_ms.append((time.perf_counter() - snapshot_started_at) * 1000.0)
        best_objective = dict(initial_objective)
        rounds, best_round = [], 0
        best_round_committed = False
        native_action_backward_used = False
        for round_id in range(1, int(self.params.timing_opt_sizing_rounds) + 1):
            self.round_index += 1
            if self.action_mode == "lut_action" and action_context is None:
                started_at = time.perf_counter()
                loss, action_context = self._action_loss(model, pos)
                forward_ms.append((time.perf_counter() - started_at) * 1000.0)
                timing_forward_ms.append(
                    float(
                        action_context.mapping_summary.get("loss_forward_ms", 0.0)
                        or 0.0
                    )
                )
                action_build_ms.append(
                    float(
                        action_context.mapping_summary.get("build_ms", 0.0) or 0.0
                    )
                )
                action_table_build_ms.append(
                    float(
                        action_context.mapping_summary.get(
                            "action_table_build_ms", 0.0
                        )
                        or 0.0
                    )
                )
                action_table_mapping_ms.append(
                    float(
                        action_context.mapping_summary.get(
                            "action_table_mapping_ms", 0.0
                        )
                        or 0.0
                    )
                )
                action_map_timing.append(
                    {
                        "build_ms": float(
                            action_context.mapping_summary.get("build_ms", 0.0) or 0.0
                        ),
                        "timing_ms": dict(
                            action_context.mapping_summary.get("timing_ms", {}) or {}
                        ),
                        "pair_cache": dict(
                            action_context.mapping_summary.get("pair_cache", {}) or {}
                        ),
                    }
                )
            blocked_started_at = time.perf_counter()
            blocked = self.oscillation.blocked(
                phase="timing",
                iteration=self.round_index,
                count=data.inst_cell_id.numel(),
                device=size.device,
            )
            blocked_ms.append((time.perf_counter() - blocked_started_at) * 1000.0)
            select_started_at = time.perf_counter()
            if action_context is not None:
                started_at = time.perf_counter()
                beta_started_at = time.perf_counter()
                score_loss = getattr(action_context, "score_loss", None)
                if score_loss is None:
                    score_loss = loss
                tns_score_loss = getattr(
                    action_context, "tns_score_loss", None
                )
                wns_score_loss = getattr(
                    action_context, "wns_score_loss", None
                )
                debug_gradients = bool(
                    getattr(self.params, "timing_action_trial_record_only", 0)
                )
                action_grad = torch.autograd.grad(
                    score_loss,
                    action_context.beta,
                    allow_unused=True,
                    retain_graph=debug_gradients,
                )[0]
                backward_passes = 1
                extra_grads = {}
                for name, value in (
                    ("tns", tns_score_loss),
                    ("wns", wns_score_loss),
                ):
                    if value is None:
                        continue
                    if value is score_loss:
                        extra_grads[name] = action_grad
                        continue
                    if not debug_gradients:
                        continue
                    extra_grads[name] = torch.autograd.grad(
                        value,
                        action_context.beta,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    backward_passes += 1
                beta_backward_ms.append(
                    (time.perf_counter() - beta_started_at) * 1000.0
                )
                action_context.mapping_summary["backward_passes"] = backward_passes
                action_context.mapping_summary["gradient_norms"] = {
                    "score": (
                        None
                        if action_grad is None
                        else float(action_grad.detach().abs().max())
                    ),
                    "tns": (
                        None
                        if extra_grads.get("tns") is None
                        else float(extra_grads["tns"].detach().abs().max())
                    ),
                    "wns": (
                        None
                        if extra_grads.get("wns") is None
                        else float(extra_grads["wns"].detach().abs().max())
                    ),
                }
                backward_ms.append((time.perf_counter() - started_at) * 1000.0)
                native_action_backward_used = native_action_backward_used or bool(
                    getattr(action_context, "native_action_backward_used", False)
                )
                if getattr(action_context, "native_action_backward_used", False):
                    refresh_action_valid_instances(action_context)
                action_scores = action_context.action_scores(action_grad)
                if not bool(torch.isfinite(action_scores).all()):
                    raise RuntimeError("lut_action beta gradient is non-finite")
                trial_summary = None
                if bool(
                    getattr(
                        self.params,
                        "timing_action_trial_record_only",
                        0,
                    )
                ):
                    trial_tns_scores = action_context.action_scores(extra_grads.get("tns"))
                    trial_wns_scores = action_context.action_scores(extra_grads.get("wns"))
                    trial_summary = self._trial_tns_action_scores(
                        model,
                        pos,
                        action_context,
                        trial_tns_scores,
                        geometry,
                        float(loss.detach()),
                        action_mask=action_mask,
                        selection_scores=action_scores,
                        wns_scores=trial_wns_scores,
                    )
                selection_valid = action_context.valid_instances
                if action_mask is not None:
                    mask = torch.as_tensor(
                        action_mask,
                        dtype=torch.bool,
                        device=selection_valid.device,
                    )
                    if mask.shape != (selection_valid.shape[1],):
                        raise ValueError(
                            "sizing action mask must match the four action columns"
                        )
                    selection_valid = selection_valid & mask.unsqueeze(0)
                candidate_select_started_at = time.perf_counter()
                next_size, next_vt, summary = apply_quad_gradient_from_action_scores(
                    action_scores=action_scores,
                    valid_instances=selection_valid,
                    data_collections=data,
                    size_logits=size,
                    vt_logits=vt,
                    candidate_table=self.candidates,
                    shared_budget_percent=float(
                        self.params.discrete_gradient_topk_shared_budget_percent
                    ),
                    blocked_instances=blocked,
                    size_parameterization="real_size",
                )
                candidate_select_ms.append(
                    (time.perf_counter() - candidate_select_started_at) * 1000.0
                )
                summary["action_mapping"] = action_context.mapping_summary
                # The post-commit forward replaces ``action_context``; keep
                # per-term predictions on the selected actions now so the
                # step diagnostics do not re-evaluate them on the next state.
                for term_name, term_key in (
                    ("tns", "predicted_delta_tns"),
                    ("wns", "predicted_delta_wns"),
                ):
                    term_gradient = extra_grads.get(term_name)
                    if term_gradient is None:
                        continue
                    term_scores = action_context.action_scores(term_gradient)
                    for item in summary.get("selected_scores", ()) or ():
                        instance_id = int(item.get("instance_id", -1))
                        action = str(item.get("action", ""))
                        if (
                            instance_id < 0
                            or action not in ACTION_NAMES
                        ):
                            continue
                        item[term_key] = -float(
                            term_scores[
                                instance_id,
                                ACTION_NAMES.index(action),
                            ].item()
                        )
                if trial_summary is not None:
                    summary["trial_tns"] = trial_summary
            else:
                started_at = time.perf_counter()
                size_grad, vt_grad = torch.autograd.grad(
                    loss, (size, vt), allow_unused=False
                )
                backward_ms.append((time.perf_counter() - started_at) * 1000.0)
                beta_backward_ms.append(0.0)
                if not bool(
                    torch.isfinite(size_grad).all()
                    & torch.isfinite(vt_grad).all()
                ):
                    raise RuntimeError("S5B1 size/VT gradient is non-finite")
                candidate_select_started_at = time.perf_counter()
                next_size, next_vt, summary = (
                    apply_quad_gradient_from_data_collections(
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
                )
                candidate_select_ms.append(
                    (time.perf_counter() - candidate_select_started_at) * 1000.0
                )
            select_ms.append((time.perf_counter() - select_started_at) * 1000.0)
            model._timing_geometry_cache = None
            if not summary["num_changed_instances"]:
                fits_started_at = time.perf_counter()
                fits = self._fits(model, capacity)
                fits_ms.append((time.perf_counter() - fits_started_at) * 1000.0)
                rounds.append(
                    {
                        "round": round_id,
                        "loss": float(loss.detach()),
                        # The next round compares against this entry's terms.
                        "objective_terms": objective_snapshot(model, loss),
                        "fits": fits,
                        "actions": summary.get("selected_scores", ()),
                        "action_counts": dict(summary.get("action_counts", {}) or {}),
                        "action_score_stats": summary.get("action_score_stats"),
                        "trial_tns": summary.get("trial_tns"),
                        "predicted_delta_sum": float(
                            sum(
                                float(item.get("predicted_delta_obj", 0.0))
                                for item in summary.get("selected_scores", ()) or ()
                            )
                        ),
                        "action_digest": summary.get("action_digest"),
                        "input_state_digest": summary.get("input_state_digest"),
                        "output_state_digest": summary.get("output_state_digest"),
                        "changed_instance_count": 0,
                        "no_action": True,
                    }
                )
                # Consume the no-op attempt while rebuilding the graph for the
                # next fixed-budget sizing round.
                if self.action_mode == "lut_action":
                    started_at = time.perf_counter()
                    loss, action_context = self._action_loss(model, pos)
                    forward_ms.append((time.perf_counter() - started_at) * 1000.0)
                    action_build_ms.append(
                        float(
                            action_context.mapping_summary.get("build_ms", 0.0) or 0.0
                        )
                    )
                    timing_forward_ms.append(
                        float(
                            action_context.mapping_summary.get(
                                "loss_forward_ms", 0.0
                            )
                            or 0.0
                        )
                    )
                    action_table_build_ms.append(
                        float(
                            action_context.mapping_summary.get(
                                "action_table_build_ms", 0.0
                            )
                            or 0.0
                        )
                    )
                    action_table_mapping_ms.append(
                        float(
                            action_context.mapping_summary.get(
                                "action_table_mapping_ms", 0.0
                            )
                            or 0.0
                        )
                    )
                    action_map_timing.append(
                        {
                            "build_ms": float(
                                action_context.mapping_summary.get("build_ms", 0.0)
                                or 0.0
                            ),
                            "timing_ms": dict(
                                action_context.mapping_summary.get("timing_ms", {})
                                or {}
                            ),
                            "pair_cache": dict(
                                action_context.mapping_summary.get("pair_cache", {})
                                or {}
                            ),
                        }
                    )
                else:
                    started_at = time.perf_counter()
                    loss = self._loss(model, pos)
                    forward_ms.append((time.perf_counter() - started_at) * 1000.0)
                    timing_forward_ms.append(forward_ms[-1])
                continue
            rows = summary["applied_instance_ids"]
            previous = data.inst_cell_id[rows].detach().cpu().tolist()
            with torch.no_grad():
                size.copy_(next_size)
                vt.copy_(next_vt)
            sync_started_at = time.perf_counter()
            self._sync(model, summary, geometry)
            sync_ms.append((time.perf_counter() - sync_started_at) * 1000.0)
            state_sync_ms.append(sync_ms[-1])
            record_started_at = time.perf_counter()
            self.oscillation.record(
                phase="timing",
                iteration=self.round_index,
                instance_ids=rows,
                previous_ids=previous,
                target_ids=summary["applied_cell_ids"],
            )
            record_ms.append((time.perf_counter() - record_started_at) * 1000.0)
            # The committed state is evaluated once.  In action mode this
            # forward also supplies the next round's beta graph, so the round
            # pays for one timing evaluation instead of a value-only pass plus
            # a second action pass.
            started_at = time.perf_counter()
            if self.action_mode == "lut_action":
                loss, action_context = self._action_loss(model, pos)
                action_build_ms.append(
                    float(action_context.mapping_summary.get("build_ms", 0.0) or 0.0)
                )
                timing_forward_ms.append(
                    float(
                        action_context.mapping_summary.get("loss_forward_ms", 0.0)
                        or 0.0
                    )
                )
                action_table_build_ms.append(
                    float(
                        action_context.mapping_summary.get(
                            "action_table_build_ms", 0.0
                        )
                        or 0.0
                    )
                )
                action_table_mapping_ms.append(
                    float(
                        action_context.mapping_summary.get(
                            "action_table_mapping_ms", 0.0
                        )
                        or 0.0
                    )
                )
                action_map_timing.append(
                    {
                        "build_ms": float(
                            action_context.mapping_summary.get("build_ms", 0.0) or 0.0
                        ),
                        "timing_ms": dict(
                            action_context.mapping_summary.get("timing_ms", {}) or {}
                        ),
                        "pair_cache": dict(
                            action_context.mapping_summary.get("pair_cache", {}) or {}
                        ),
                    }
                )
            else:
                loss = self._loss(model, pos)
                action_context = None
            forward_ms.append((time.perf_counter() - started_at) * 1000.0)
            if self.action_mode != "lut_action":
                timing_forward_ms.append(forward_ms[-1])
            actual = float(loss.detach())
            fits_started_at = time.perf_counter()
            fits = self._fits(model, capacity)
            fits_ms.append((time.perf_counter() - fits_started_at) * 1000.0)
            step_terms = objective_snapshot(model, loss)
            previous_terms = (
                rounds[-1]["objective_terms"] if rounds else initial_objective
            )
            selected_scores = summary.get("selected_scores", ())
            def _selected_sum(key):
                # Predictions are attached at selection time; summing them
                # avoids using the post-commit context's valid_instances.
                total = 0.0
                found = False
                for item in selected_scores or ():
                    if key not in item:
                        continue
                    total += float(item[key])
                    found = True
                return total if found else None

            # Only report derivatives of the named metric. A weighted loss
            # derivative cannot be reported as a TNS prediction.
            predicted_tns_raw = _selected_sum("predicted_delta_tns")
            calibration = 1.0
            predicted_tns = predicted_tns_raw
            predicted_wns = _selected_sum("predicted_delta_wns")
            actual_delta_tns = _term_delta(
                step_terms.get("tns"), previous_terms.get("tns")
            )
            actual_delta_wns = _term_delta(
                step_terms.get("wns"), previous_terms.get("wns")
            )
            rounds.append(
                {
                    "round": round_id,
                    "loss": actual,
                    "objective_terms": step_terms,
                    "predicted_delta_tns": predicted_tns,
                    "predicted_delta_tns_raw": predicted_tns_raw,
                    "first_order_sum_tns": predicted_tns_raw,
                    "step_tns_calibration": calibration,
                    "predicted_delta_wns": predicted_wns,
                    "actual_delta_tns": actual_delta_tns,
                    "actual_delta_wns": actual_delta_wns,
                    "fits": fits,
                    "actions": summary["selected_scores"],
                    "action_counts": dict(summary.get("action_counts", {}) or {}),
                    "action_score_stats": summary.get("action_score_stats"),
                    "gradient_norms": (
                        summary.get("action_mapping") or {}
                    ).get("gradient_norms"),
                    "trial_tns": summary.get("trial_tns"),
                    "predicted_delta_sum": float(
                        sum(
                            float(item.get("predicted_delta_obj", 0.0))
                            for item in summary.get("selected_scores", ()) or ()
                        )
                    ),
                    "actual_delta": actual
                    - (rounds[-1]["loss"] if rounds else initial_loss),
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
                capture_started_at = time.perf_counter()
                best, best_loss, best_round = (
                    SizingState.capture(data, self.oscillation),
                    actual,
                    round_id,
                )
                capture_ms.append((time.perf_counter() - capture_started_at) * 1000.0)
                snapshot_started_at = time.perf_counter()
                best_objective = objective_snapshot(model, loss)
                snapshot_ms.append(
                    (time.perf_counter() - snapshot_started_at) * 1000.0
                )
                best_round_committed = True
        # When the accepted best is the state that is already loaded, the
        # restore is a no-op and the last forward is the exact restored value.
        # Reuse it instead of paying for a duplicate timing evaluation.
        current_is_best = (
            best_round_committed
            and bool(rounds)
            and best_round == int(rounds[-1]["round"])
            and not rounds[-1].get("no_action", False)
        )
        restore_started_at = time.perf_counter()
        self.restore_sizing_state(model, best, geometry)
        restore_sync_ms = (time.perf_counter() - restore_started_at) * 1000.0
        with torch.no_grad():
            if current_is_best:
                restored_loss = loss.detach()
                restore_forward_ms = 0.0
            else:
                started_at = time.perf_counter()
                restored_loss = self._loss(model, pos)
                restore_forward_ms = (time.perf_counter() - started_at) * 1000.0
        if not math.isclose(float(restored_loss), best_loss, rel_tol=1e-5, abs_tol=1e-4):
            raise RuntimeError(
                "sizing restore did not reproduce the phase best objective: "
                f"best={best_loss} restored={float(restored_loss)}"
            )
        restored_objective = objective_snapshot(model, restored_loss)
        if initial_action_mapping is not None and action_context is not None:
            initial_action_mapping["native_action_backward_used"] = bool(
                native_action_backward_used
            )
        self._last_action_context = None
        return {
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_round": best_round,
            "initial_objective": initial_objective,
            "best_objective": best_objective,
            "timing_action_gradient_mode": self.action_mode,
            "run_sizing_ms": float(
                (time.perf_counter() - sizing_started_at) * 1000.0
            ),
            "action_mapping": initial_action_mapping,
            "runtime_profile": {
                "initial_forward_ms": float(initial_forward_ms),
                "initial_timing_forward_ms": float(initial_timing_forward_ms),
                "initial_action_table_build_ms": float(
                    initial_action_table_build_ms
                ),
                "initial_action_table_mapping_ms": float(
                    initial_action_table_mapping_ms
                ),
                "round_forward_ms": [float(value) for value in forward_ms],
                "timing_forward_ms": [
                    float(value) for value in timing_forward_ms
                ],
                "backward_ms": [float(value) for value in backward_ms],
                "beta_backward_ms": [float(value) for value in beta_backward_ms],
                "action_build_ms": [float(value) for value in action_build_ms],
                "action_table_build_ms": [
                    float(value) for value in action_table_build_ms
                ],
                "action_table_mapping_ms": [
                    float(value) for value in action_table_mapping_ms
                ],
                "action_map_timing": action_map_timing,
                "select_ms": [float(value) for value in select_ms],
                "candidate_select_ms": [
                    float(value) for value in candidate_select_ms
                ],
                "sync_ms": [float(value) for value in sync_ms],
                "state_sync_ms": [float(value) for value in state_sync_ms],
                "fits_ms": [float(value) for value in fits_ms],
                "capture_ms": [float(value) for value in capture_ms],
                "blocked_ms": [float(value) for value in blocked_ms],
                "record_ms": [float(value) for value in record_ms],
                "snapshot_ms": [float(value) for value in snapshot_ms],
                "restore_sync_ms": float(restore_sync_ms),
                "restore_forward_ms": float(restore_forward_ms),
            },
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
