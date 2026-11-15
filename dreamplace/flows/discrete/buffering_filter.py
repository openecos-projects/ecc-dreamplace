"""Shared candidate-state refresh orchestration for placement and diff-opt."""


import time
import torch
from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
    collect_candidate_dirty_nets,
)
from time import perf_counter
from dreamplace.flows.discrete.common import parse_enabled_flag
from dreamplace.flows.discrete.sizing_filter import (
    _build_engine_config,
    _build_rc_forest_payload,
    _phase_policy,
    common_incremental_sta_skip_reason,
)
from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
    DiscreteCandidateTransition,
    TypedDiscreteCandidateTransition,
)


__all__ = [
    "accepted_candidate_net_ids",
    "candidate_topology_geometry_key",
    "collect_owner_candidate_dirty_nets",
    "compact_candidate_dirty_summary",
    "prepare_candidate_owner",
    "refresh_owner_candidate_geometry_after_topology",
    "refresh_topology_if_needed",
    "IncrementalBufferingGate",
]


def candidate_topology_geometry_key(*, data, lane, ops, pos):
    """Identify the topology state whose candidate coordinates must stay live."""

    topology = getattr(ops, "steiner_topo_op", None)
    frozen_nets = getattr(topology, "frozen_nets", None)
    state = None if lane is None else lane._current_state()
    candidate_live_geometry = bool(state is not None and hasattr(state, "activation_param"))
    return _tensor_digest(
        (
            (
                "runtime_cell_state_generation",
                None
                if candidate_live_geometry
                else int(getattr(data, "runtime_cell_state_generation", 0) or 0),
            ),
            (
                "topology_generation",
                int(getattr(topology, "topology_generation", 0) or 0),
            ),
            (
                "frozen_topology_generation",
                getattr(topology, "frozen_topology_generation", None),
            ),
            (
                "frozen_net_ids",
                tuple(int(net_id) for net_id in getattr(frozen_nets, "net_ids", ()) or ()),
            ),
            ("pos", None if candidate_live_geometry else pos.detach()),
        )
    )


def refresh_topology_if_needed(owner, pos, *, force=False):
    """Refresh or reuse the live topology and publish a compact status."""

    if getattr(getattr(owner, "placedb", None), "gr_sizing", None) is not None:
        state = {"status": "skipped", "reason": "frozen_gr_snapshot"}
        owner.last_topology_refresh_state = state
        owner.data.buffer_topology_refresh_state = dict(state)
        return state

    geometry_key = (
        None
        if force
        else candidate_topology_geometry_key(
            data=owner.data,
            lane=owner.lane,
            ops=owner.ops,
            pos=pos,
        )
    )
    if geometry_key is not None and geometry_key == getattr(
        owner,
        "_topology_geometry_key",
        None,
    ):
        owner.topology_refresh_reuse_count = (
            int(getattr(owner, "topology_refresh_reuse_count", 0)) + 1
        )
        state = {
            "status": "reused",
            "geometry_key": geometry_key,
            "topology_generation": int(
                getattr(owner.ops.steiner_topo_op, "topology_generation", 0)
            ),
        }
    else:
        owner.refresh_topology(pos)
        owner.topology_refresh_reuse_count = int(getattr(owner, "topology_refresh_reuse_count", 0))
        owner._topology_geometry_key = candidate_topology_geometry_key(
            data=owner.data,
            lane=owner.lane,
            ops=owner.ops,
            pos=pos,
        )
        owner.topology_refresh_count = int(getattr(owner, "topology_refresh_count", 0)) + 1
        state = {
            "status": "refreshed",
            "geometry_key": owner._topology_geometry_key,
            "topology_generation": int(
                getattr(
                    getattr(owner.ops, "steiner_topo_op", None),
                    "topology_generation",
                    0,
                )
                or 0
            ),
            "forced": bool(force),
        }
    owner.last_topology_refresh_state = state
    owner.data.buffer_topology_refresh_state = dict(state)
    return state


def prepare_candidate_owner(
    owner,
    model,
    pos,
    *,
    active_only=False,
    refresh_records_all=False,
    force_topology=False,
    lane_flow_kind=None,
    force_rebuild_when_unbounded=False,
    clear_timing_geometry_cache=False,
    reason="live_topology_refresh",
):
    """Prepare or incrementally refresh an owner's buffering lane."""

    from dreamplace.ops.buffer_insertion.buffering_config import (
        build_buffering_config_from_params,
    )
    from dreamplace.ops.buffer_insertion.buffering_lane import (
        BufferingOptimizationLane,
    )

    prepare_started_at = time.perf_counter()
    runtime_profile = {"active_only": bool(active_only)}

    def record_stage(name, started_at):
        runtime_profile[name] = (time.perf_counter() - started_at) * 1000.0

    def finish(status):
        runtime_profile["status"] = str(status)
        runtime_profile["total_ms"] = (time.perf_counter() - prepare_started_at) * 1000.0
        owner.last_prepare_runtime_profile = runtime_profile
        owner.data.buffer_prepare_runtime_profile = dict(runtime_profile)

    stage_started_at = time.perf_counter()
    counts = {} if owner.lane is None else owner._counts(owner.lane._current_state())
    record_stage("counts_ms", stage_started_at)

    stage_started_at = time.perf_counter()
    owner._candidate_refresh_in_progress = True
    try:
        refresh_topology_if_needed(owner, pos, force=force_topology)
    finally:
        owner._candidate_refresh_in_progress = False
    record_stage("topology_ms", stage_started_at)
    runtime_profile["topology_status"] = str(
        (getattr(owner, "last_topology_refresh_state", {}) or {}).get(
            "status",
            "unknown",
        )
    )

    existing_state = None if owner.lane is None else owner.lane._current_state()
    if existing_state is not None and hasattr(
        existing_state,
        "activation_param",
    ):
        dirty_net_ids = collect_owner_candidate_dirty_nets(owner)
        runtime_profile["candidate_dirty_summary"] = compact_candidate_dirty_summary(
            owner.last_candidate_dirty_net_summary
        )
        owner.lane.retain_candidate_python_state = True
        stage_started_at = time.perf_counter()
        refresh_summary = owner.lane.refresh_candidate_geometry(
            model=model,
            params=owner.params,
            reason=reason,
            active_only=active_only,
            dirty_net_ids=dirty_net_ids,
            refresh_records_all=bool(refresh_records_all),
            force_rebuild=bool(force_rebuild_when_unbounded and dirty_net_ids is None),
        )
        record_stage("candidate_refresh_ms", stage_started_at)
        if refresh_summary.get("status") != "ok":
            raise RuntimeError(f"S5B1 candidate state refresh failed: {refresh_summary}")
        owner.data.buffer_retained_candidate_state = refresh_summary
        runtime_profile["dirty_net_count"] = int(len(dirty_net_ids or ()))
        runtime_profile["refresh_records_all"] = bool(refresh_records_all)
        owner._pending_dirty_instance_ids = set()
        owner._pending_dirty_net_ids = set()
        if active_only:
            owner.ops.steiner_topo_op.frozen_nets.candidate_state = owner.lane._current_state()
        model._virtual_cell_density_view = None
        if clear_timing_geometry_cache:
            model._timing_geometry_cache = None
        finish("candidate_state_refreshed")
        return

    # The topology refresh can remap retained state; rebuild the count
    # transfer payload only after it has completed.
    if owner.lane is not None:
        counts = owner._counts(owner.lane._current_state())
    owner.data.buffer_retained_segment_edges = tuple(counts)

    if lane_flow_kind is None:
        resolved_flow_kind = getattr(owner.params, "flow_kind", "placement")
        resolved_flow_kind = getattr(
            resolved_flow_kind,
            "value",
            resolved_flow_kind,
        )
        lane_flow_kind = (
            str(resolved_flow_kind)
            if str(resolved_flow_kind) in {"buffering", "joint"}
            else "placement"
        )

    stage_started_at = time.perf_counter()
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            owner.params,
            flow_kind=str(lane_flow_kind),
        ),
        placedb=owner.placedb,
    )
    lane.prepare(model=model, params=owner.params)
    record_stage("lane_prepare_ms", stage_started_at)
    if hasattr(lane._current_state(), "activation_param"):
        lane.retain_candidate_python_state = True
    state = lane._current_state()
    if state is None:
        raise RuntimeError(f"S5B1 state construction failed: {lane.summarize()}")
    owner._transfer_counts(state, counts)
    if hasattr(state, "activation_param"):
        state.timing_active_only = bool(active_only)
        lane._candidate_state_contains_full_domain = True
        if active_only:
            state.activation_param.requires_grad_(requires_grad=False)
            owner.ops.steiner_topo_op.frozen_nets.candidate_state = state
    if hasattr(state, "z_param"):
        state.z_param.requires_grad_(requires_grad=False)
        state.bsu_index_param.requires_grad_(requires_grad=False)
    lane.attach_state(owner.data)
    owner.lane = lane
    model._virtual_cell_density_view = None
    if clear_timing_geometry_cache:
        model._timing_geometry_cache = None
    owner._pending_dirty_instance_ids = set()
    owner._pending_dirty_net_ids = set()
    finish("lane_created")


def collect_owner_candidate_dirty_nets(owner):
    """Collect pending sizing and buffering changes into candidate-net IDs."""

    state = None if owner.lane is None else owner.lane._current_state()
    dirty_net_ids, summary = collect_candidate_dirty_nets(
        state=state,
        lane=owner.lane,
        placedb=owner.placedb,
        data=owner.data,
        changed_instance_ids=getattr(
            owner,
            "_pending_dirty_instance_ids",
            (),
        ),
        accepted_net_ids=getattr(
            owner,
            "_pending_dirty_net_ids",
            (),
        ),
    )
    owner.last_candidate_dirty_net_summary = summary
    owner.data.buffer_candidate_dirty_net_summary = dict(compact_candidate_dirty_summary(summary))
    return dirty_net_ids


def refresh_owner_candidate_geometry_after_topology(
    owner,
    model,
    *,
    reason,
):
    """Refresh active retained candidates after publishing live topology."""

    from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
        active_candidate_net_ids,
    )

    if owner.lane is None:
        return {"status": "skipped", "reason": "lane_uninitialized"}
    state = owner.lane._current_state()
    if state is None or not hasattr(state, "activation_param"):
        return {"status": "skipped", "reason": "non_candidate_state"}
    active_net_ids = active_candidate_net_ids(state)
    if not active_net_ids:
        return {
            "status": "ok",
            "reason": "no_active_candidates",
            "candidate_count": 0,
        }
    summary = owner.lane.refresh_candidate_geometry(
        model=model,
        params=owner.params,
        reason=reason,
        active_only=True,
        dirty_net_ids=active_net_ids,
    )
    owner.data.buffer_retained_candidate_state = summary
    owner.lane.attach_state(owner.data)
    owner.ops.steiner_topo_op.frozen_nets.candidate_state = owner.lane._current_state()
    owner.params.enable_relaxed_buffer_timing = True
    model._virtual_cell_density_view = None
    model._timing_geometry_cache = None
    return summary


def compact_candidate_dirty_summary(summary):
    """Keep profiling payloads small while retaining failure diagnostics."""

    summary = dict(summary or {})
    return {
        key: summary[key] for key in ("status", "reason", "counts", "metadata") if key in summary
    }


def accepted_candidate_net_ids(activation_before, state, *, candidate_ids_before=None):
    """Return candidate nets whose position became newly active."""

    active_after = (
        state.activation_param.detach().reshape(state.activation_param.shape[0], -1).sum(dim=1) > 0
    )
    candidate_net_id = getattr(state, "candidate_net_id", None)
    if candidate_net_id is None or not int(active_after.numel()):
        return []
    if (
        activation_before.shape[0] == state.activation_param.shape[0]
        and candidate_ids_before is None
    ):
        active_before = (
            activation_before.detach().reshape(activation_before.shape[0], -1).sum(dim=1) > 0
        )
        accepted_rows = torch.nonzero(
            active_before ^ active_after,
            as_tuple=False,
        ).flatten()
    else:
        before_active = (
            activation_before.detach().reshape(activation_before.shape[0], -1).sum(dim=1) > 0
        )
        before_ids = (
            candidate_ids_before.detach().cpu().tolist()
            if candidate_ids_before is not None
            else []
        )
        before_active_set = {
            int(before_ids[row])
            for row, flag in enumerate(before_active.cpu().tolist())
            if flag and row < len(before_ids)
        }
        after_ids = state.candidate_ids.detach().cpu().tolist()
        accepted_rows = torch.tensor(
            [
                row
                for row, flag in enumerate(active_after.cpu().tolist())
                if flag and int(after_ids[row]) not in before_active_set
            ],
            dtype=torch.long,
        )
    if not accepted_rows.numel():
        return []
    return (
        torch.unique(
            candidate_net_id.index_select(
                0,
                accepted_rows.to(candidate_net_id.device),
            ),
            sorted=True,
        )
        .detach()
        .cpu()
        .tolist()
    )


def _static_skip_reason(owner, model) -> str | None:
    params = getattr(owner, "params", None)
    if params is None:
        return "missing_params"
    if not parse_enabled_flag(params, "timing_buffering_incremental_commit", default=True):
        return "disabled"
    placedb = getattr(owner, "placedb", None)
    if getattr(placedb, "gr_sizing", None) is not None:
        return "gr_route_snapshot_active"
    return common_incremental_sta_skip_reason(params, model)


def _legal_cell_ids_from_lane(lane) -> list[int] | None:
    if lane is None:
        return None
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    if not isinstance(payload, dict):
        return None
    legal_table = payload.get("buffer_legal_table")
    if not isinstance(legal_table, dict):
        return None
    raw_ids = legal_table.get("legal_cell_ids")
    if not raw_ids:
        return None
    return [int(v) for v in raw_ids]


class IncrementalBufferingGate:
    """Window-scoped C++ incremental-STA filter for candidate and segment buffering."""

    def __init__(self, owner, model, pos, geometry, capacity: float, *, lane=None):
        self.owner = owner
        self.model = model
        self.pos = pos
        self.geometry = geometry
        self.capacity = float(capacity)
        self.lane = lane if lane is not None else getattr(owner, "lane", None)
        self._engine = None
        self._build_ms = 0.0
        self._static_check_done = False
        self._static_skip_reason: str | None = None
        self._rejected_typed_actions: torch.Tensor | None = None
        self._rejected_untyped_candidates: torch.Tensor | None = None
        self.last_area: float | None = None
        self.accepted_action_infos: list[dict] = []

    def check_available(self, *, preexisting_virtual_count: int = 0) -> str | None:
        """Return None when C++ incremental buffering gate can run, else a skip reason."""
        if not self._static_check_done:
            self._static_skip_reason = _static_skip_reason(self.owner, self.model)
            self._static_check_done = True
        if self._static_skip_reason is not None:
            return self._static_skip_reason
        if preexisting_virtual_count > 0:
            return "preexisting_virtual_buffers"
        return None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine, False, 0.0
        from dreamplace.ops.timing_propagation import timing_propagation_cpp

        started = perf_counter()
        timing = self.model.op_collections.timing_propagation_op
        num_pins = int(timing.num_pins)
        num_nets = int(timing.net2driver_pin_map.numel())
        _rc_key, rc_payload, _wc, _ir = _build_rc_forest_payload(
            self.owner, self.model, self.pos, num_pins, num_nets
        )
        engine_cfg = _build_engine_config(self.owner, self.model, rc_payload)
        self._engine = timing_propagation_cpp.IncrementalSizingEngine(engine_cfg)
        self._build_ms = (perf_counter() - started) * 1000.0
        return self._engine, False, self._build_ms

    def _policy(self, *, capacity: float | None = None) -> dict:
        timing = self.model.op_collections.timing_propagation_op
        num_pins = int(timing.num_pins)
        num_cells = int(self.owner.data.cell_id_2_arc_id_start.numel()) - 1
        policy = _phase_policy(
            self.owner,
            self.model,
            self.geometry,
            self.capacity if capacity is None else float(capacity),
            num_pins,
            num_cells,
        )
        # Accepted buffers in this window are tracked inside self._engine directly.
        policy["virtual_area"] = 0.0
        return policy

    def current_area(self) -> float | None:
        if self.last_area is not None:
            return float(self.last_area)
        return None

    def refresh_dirty_subnets(self, dirty_nets: list[int]) -> list[dict]:
        """Rebuild FLUTE Steiner trees & incremental STA on accepted split sub-nets."""
        if not dirty_nets or self._engine is None:
            return []
        return list(self._engine.refresh_dirty_subnet_steiner_trees([int(n) for n in dirty_nets]))

    def remap_rejection_masks(
        self,
        *,
        is_typed: bool,
        new_shape: tuple[int, ...],
        old_to_new_rows: dict[int, int],
        dirty_orig_nets: set[int],
        new_candidate_orig_nets: list[int],
    ) -> None:
        """Preserve untouched nets' rejection masks while clearing refreshed sub-net rows."""
        attr = "_rejected_typed_actions" if is_typed else "_rejected_untyped_candidates"
        old_mask = getattr(self, attr)
        if old_mask is None:
            return
        new_mask = torch.zeros(new_shape, dtype=torch.bool, device=old_mask.device)
        for old_row, new_row in old_to_new_rows.items():
            if int(new_candidate_orig_nets[int(new_row)]) in dirty_orig_nets:
                continue
            new_mask[int(new_row)] = old_mask[int(old_row)]
        setattr(self, attr, new_mask)

    def allowed_typed_actions(self, state):
        base = getattr(state, "drive_type_allowed", None)
        if self._rejected_typed_actions is None:
            return base
        rej = self._rejected_typed_actions.to(
            device=state.activation_param.device, dtype=torch.bool
        )
        if base is None:
            return ~rej
        return base.to(device=rej.device, dtype=torch.bool) & ~rej

    def allowed_untyped_candidates(self, state):
        if self._rejected_untyped_candidates is None:
            return None
        return ~self._rejected_untyped_candidates.to(
            device=state.activation_param.device, dtype=torch.bool
        )

    def filter_candidate_transition(
        self,
        state,
        transition,
        *,
        capacity_override: float | None = None,
    ):
        """Filter one round's gradient-selected candidate actions via C++ incremental STA."""
        reason = self.check_available()
        if reason is not None:
            report = (
                None
                if reason == "disabled"
                else {"status": "skipped", "reason": reason}
            )
            return transition, report

        is_typed = bool(getattr(state, "is_typed_candidate", False))
        legal_cell_ids = None if is_typed else _legal_cell_ids_from_lane(self.lane)
        if not is_typed and not legal_cell_ids:
            return transition, {"status": "skipped", "reason": "missing_legal_cell_ids"}

        if transition.accepted_count == 0:
            return transition, {
                "status": "ok",
                "input_count": 0,
                "evaluated_count": 0,
                "accepted_count": 0,
                "rejected_area_count": 0,
                "rejected_loss_count": 0,
                "elapsed_ms": 0.0,
            }

        started = perf_counter()
        rows = transition.selected_candidate_indices.detach().cpu().tolist()
        drive_types = (
            transition.selected_drive_type_ids.detach().cpu().tolist()
            if is_typed
            else [None] * len(rows)
        )
        specs = []
        for row_idx, dtype_id in zip(rows, drive_types, strict=True):
            rec = state.candidate_records[int(row_idx)]
            net_id = int(
                rec.get(
                    "cpp_net_id",
                    state.candidate_net_id[int(row_idx)].detach().cpu().item(),
                )
            )
            parent_v = rec.get(
                "cpp_parent_vertex",
                rec.get("parent_node_id"),
            )
            raw_children = rec.get("cpp_child_vertices", rec.get("child_node_ids")) or ()
            child_vs = [int(v) for v in raw_children]
            raw_ratio = rec.get("segment_split_ratio")
            split_ratio = -1.0 if raw_ratio is None else float(raw_ratio)
            if is_typed:
                target_cell = int(
                    state.drive_type_cell_ids[int(dtype_id)].detach().cpu().item()
                )
            else:
                assert legal_cell_ids is not None
                target_cell = int(legal_cell_ids[int(state.fixed_bsu_index)])
            spec = {
                "net_id": net_id,
                "parent_vertex": -1 if parent_v is None else int(parent_v),
                "child_vertices": child_vs,
                "split_ratio": split_ratio,
                "target_cell_id": target_cell,
            }
            if rec.get("x_dbu") is not None and rec.get("y_dbu") is not None:
                spec["x_dbu"] = float(rec["x_dbu"])
                spec["y_dbu"] = float(rec["y_dbu"])
            specs.append(spec)

        engine, cache_hit, build_ms = self._ensure_engine()
        eval_result = dict(
            engine.evaluate_ranked_buffer_candidates(
                specs,
                self._policy()
                if capacity_override is None
                else self._policy(capacity=capacity_override),
            )
        )
        eval_result["cache_hit"] = bool(cache_hit)
        eval_result["engine_build_ms"] = float(build_ms)
        eval_result["total_filter_ms"] = float((perf_counter() - started) * 1000.0)
        if "final_area" in eval_result:
            self.last_area = float(eval_result["final_area"])

        accepted_idx = [int(i) for i in eval_result["accepted_indices"]]
        accepted_set = set(accepted_idx)
        accepted_action_infos = list(eval_result.get("accepted_actions") or ())
        info_by_spec = {
            int(info["spec_index"]): dict(info)
            for info in accepted_action_infos
            if "spec_index" in info
        }
        batch_infos = []
        for spec_i in accepted_idx:
            row_i = int(rows[spec_i])
            rec = state.candidate_records[row_i]
            info = dict(info_by_spec.get(spec_i, {}))
            info.update(
                {
                    "row_idx": row_i,
                    "candidate_id": int(rec.get("candidate_id", row_i)),
                    "orig_net_id": int(rec.get("net_id", rec.get("affected_net_id", -1))),
                    "drive_type_id": (
                        None if drive_types[spec_i] is None else int(drive_types[spec_i])
                    ),
                }
            )
            batch_infos.append(info)
            self.accepted_action_infos.append(info)
        eval_result["accepted_action_infos"] = batch_infos
        device = state.activation_param.device
        keep = torch.tensor(accepted_idx, dtype=torch.long, device=device)

        next_activations = state.activation_param.detach().clone()
        if is_typed:
            sel_rows = transition.selected_candidate_indices.index_select(0, keep)
            sel_types = transition.selected_drive_type_ids.index_select(0, keep)
            if keep.numel() > 0:
                next_activations[sel_rows, sel_types] = 1.0
            filtered = TypedDiscreteCandidateTransition(
                next_activations=next_activations,
                selected_candidate_indices=sel_rows,
                selected_drive_type_ids=sel_types,
                selected_net_indices=transition.selected_net_indices.index_select(0, keep),
                selected_candidate_ids=transition.selected_candidate_ids.index_select(0, keep),
                selected_raw_grads=transition.selected_raw_grads.index_select(0, keep),
                selected_predicted_improvements=(
                    transition.selected_predicted_improvements.index_select(0, keep)
                ),
                affected_net_count=transition.affected_net_count,
                eligible_net_count=transition.eligible_net_count,
                prefix_size=transition.prefix_size,
                positive_prefix_count=transition.positive_prefix_count,
            )
        else:
            sel_rows = transition.selected_candidate_indices.index_select(0, keep)
            if keep.numel() > 0:
                next_activations[sel_rows] = 1.0
            filtered = DiscreteCandidateTransition(
                next_activations=next_activations,
                selected_candidate_indices=sel_rows,
                selected_net_indices=transition.selected_net_indices.index_select(0, keep),
                selected_candidate_ids=transition.selected_candidate_ids.index_select(0, keep),
                selected_raw_grads=transition.selected_raw_grads.index_select(0, keep),
                selected_predicted_improvements=(
                    transition.selected_predicted_improvements.index_select(0, keep)
                ),
                affected_net_count=transition.affected_net_count,
                eligible_net_count=transition.eligible_net_count,
                prefix_size=transition.prefix_size,
                positive_prefix_count=transition.positive_prefix_count,
            )

        if accepted_idx:
            self.owner._incremental_sizing_cache = None
        if len(accepted_set) < len(rows):
            if is_typed:
                if self._rejected_typed_actions is None:
                    self._rejected_typed_actions = torch.zeros(
                        state.activation_param.shape, dtype=torch.bool
                    )
                for spec_i, (row_idx, dtype_id) in enumerate(zip(rows, drive_types, strict=True)):
                    if spec_i not in accepted_set:
                        self._rejected_typed_actions[int(row_idx), int(dtype_id)] = True
            else:
                if self._rejected_untyped_candidates is None:
                    self._rejected_untyped_candidates = torch.zeros(
                        state.activation_param.shape, dtype=torch.bool
                    )
                for spec_i, row_idx in enumerate(rows):
                    if spec_i not in accepted_set:
                        self._rejected_untyped_candidates[int(row_idx)] = True

        return filtered, eval_result

    def filter_segment_transition(self, state, before_counts: torch.Tensor, transition):
        """Filter segment-count increments (+1 buffer at midpoint) via C++ incremental STA."""
        preexisting = int(before_counts.detach().sum().cpu().item())
        reason = self.check_available(preexisting_virtual_count=preexisting)
        if reason is not None:
            return None

        legal_cell_ids = _legal_cell_ids_from_lane(self.lane)
        if not legal_cell_ids:
            return None

        if transition.accepted_count == 0:
            engine, cache_hit, build_ms = self._ensure_engine()
            eval_result = dict(
                engine.evaluate_ranked_buffer_candidates([], self._policy())
            )
            eval_result["cache_hit"] = bool(cache_hit)
            eval_result["engine_build_ms"] = float(build_ms)
            eval_result["total_filter_ms"] = float(build_ms)
            return (
                before_counts.detach().clone(),
                0,
                float(eval_result["final_loss"]),
                eval_result,
            )

        started = perf_counter()
        selected_indices = [
            int(v)
            for v in transition.selected_segment_indices.detach().cpu().tolist()
        ]
        if getattr(state, "is_tensor_backed", False):
            segments = state.segment_records(
                selected_indices, materialization_kind="terminal"
            )
        else:
            segments = [state.segment_rows[idx] for idx in selected_indices]
        bsu_values = state.bsu_index().detach().cpu()
        max_bsu = int(state.legal_buffer_count) - 1

        specs = []
        for seg_idx, seg in zip(selected_indices, segments, strict=True):
            bsu_raw = float(bsu_values[seg_idx].item())
            bsu_idx = max(0, min(max_bsu, int(round(bsu_raw))))
            specs.append(
                {
                    "net_id": int(seg["net_id"]),
                    "parent_vertex": int(seg["parent_node_id"]),
                    "child_vertices": [int(seg["child_node_id"])],
                    "split_ratio": 0.5,
                    "target_cell_id": int(legal_cell_ids[bsu_idx]),
                }
            )

        engine, cache_hit, build_ms = self._ensure_engine()
        eval_result = dict(
            engine.evaluate_ranked_buffer_candidates(specs, self._policy())
        )
        eval_result["cache_hit"] = bool(cache_hit)
        eval_result["engine_build_ms"] = float(build_ms)
        eval_result["total_filter_ms"] = float((perf_counter() - started) * 1000.0)

        accepted_idx = [int(i) for i in eval_result["accepted_indices"]]
        next_counts = before_counts.detach().clone()
        for k in accepted_idx:
            next_counts[selected_indices[k]] += 1.0
        if accepted_idx:
            self.owner._incremental_sizing_cache = None

        return (
            next_counts,
            len(accepted_idx),
            float(eval_result["final_loss"]),
            eval_result,
        )
