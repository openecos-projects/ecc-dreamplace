"""Sequential C++ incremental-STA gate for discrete buffer insertion (B1)."""

from time import perf_counter

import torch

from dreamplace.flows.discrete_timing_common import parse_enabled_flag
from dreamplace.flows.incremental_sizing_filter import (
    _build_engine_config,
    _build_rc_forest_payload,
    _phase_policy,
    common_incremental_sta_skip_reason,
)
from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
    DiscreteCandidateTransition,
    TypedDiscreteCandidateTransition,
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

    def _policy(self) -> dict:
        timing = self.model.op_collections.timing_propagation_op
        num_pins = int(timing.num_pins)
        num_cells = int(self.owner.data.cell_id_2_arc_id_start.numel()) - 1
        policy = _phase_policy(
            self.owner,
            self.model,
            self.geometry,
            self.capacity,
            num_pins,
            num_cells,
        )
        # Accepted buffers in this window are tracked inside self._engine directly.
        policy["virtual_area"] = 0.0
        return policy

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

    def filter_candidate_transition(self, state, transition):
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
            net_id = int(state.candidate_net_id[int(row_idx)].detach().cpu().item())
            parent_v = rec.get("parent_node_id")
            child_vs = [int(v) for v in (rec.get("child_node_ids") or ())]
            raw_ratio = rec.get("segment_split_ratio")
            split_ratio = -1.0 if raw_ratio is None else float(raw_ratio)
            if is_typed:
                target_cell = int(
                    state.drive_type_cell_ids[int(dtype_id)].detach().cpu().item()
                )
            else:
                assert legal_cell_ids is not None
                target_cell = int(legal_cell_ids[int(state.fixed_bsu_index)])
            specs.append(
                {
                    "net_id": net_id,
                    "parent_vertex": -1 if parent_v is None else int(parent_v),
                    "child_vertices": child_vs,
                    "split_ratio": split_ratio,
                    "target_cell_id": target_cell,
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
            self._rejected_typed_actions = None
            self._rejected_untyped_candidates = None
            self.owner._incremental_sizing_cache = None
        else:
            if is_typed:
                if self._rejected_typed_actions is None:
                    self._rejected_typed_actions = torch.zeros(
                        state.activation_param.shape, dtype=torch.bool
                    )
                for row_idx, dtype_id in zip(rows, drive_types, strict=True):
                    self._rejected_typed_actions[int(row_idx), int(dtype_id)] = True
            else:
                if self._rejected_untyped_candidates is None:
                    self._rejected_untyped_candidates = torch.zeros(
                        state.activation_param.shape, dtype=torch.bool
                    )
                for row_idx in rows:
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


__all__ = ["IncrementalBufferingGate"]
