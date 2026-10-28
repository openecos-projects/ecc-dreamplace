import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path

from .buffering_inner_loop import BufferingInnerLoopSummary
from .typed_candidate_gradient import typed_candidate_objective_gradient as _typed_candidate_objective_gradient
from .buffering_lane import _tensor_digest
from .discrete_candidate_scheduler import (
    schedule_candidate_gradient_actions,
    schedule_typed_candidate_gradient_actions,
)
from .discrete_virtual_buffer_runner import _compact_round_record
from .candidate_objective import objective_snapshot as _objective_snapshot



def _assert_candidate_state(lane, *, require_zero):
    state = lane._current_state()
    if state is None or not hasattr(state, "activation_param"):
        raise ValueError("candidate discrete_net_gradient requires discrete candidate state")
    config = lane.config
    if str(getattr(config, "mode", "")) != "candidate":
        raise ValueError("candidate discrete_net_gradient requires candidate mode")
    strategy = str(
        getattr(
            config,
            "active_strategy",
            getattr(config, "candidate_strategy", "continuous"),
        )
    )
    if strategy != "discrete_net_gradient":
        raise ValueError("candidate discrete runner requires discrete_net_gradient")
    if bool(getattr(config, "segment_capacity_enabled", False)):
        raise ValueError("candidate discrete_net_gradient cannot use capacity control")
    if getattr(config, "max_selected_actions", None) is not None:
        raise ValueError("candidate discrete_net_gradient cannot use max_selected_actions")
    state.validate_binary_()
    if getattr(state, "is_typed_candidate", False):
        if str(getattr(config, "candidate_drive_type_mode", "")) != "four_strength_table":
            raise ValueError("typed candidate state requires four_strength_table mode")
        if require_zero and not bool(
            (state.activation_param.detach().sum(dim=1) == 0.0).all()
        ):
            raise ValueError("typed candidate discrete_net_gradient must start from zero")
    else:
        fixed_bsu_index = state.fixed_bsu_index
        if fixed_bsu_index is None:
            raise ValueError("candidate discrete_net_gradient requires fixed_bsu_index")
        if require_zero and not bool((state.activation_param.detach() == 0.0).all()):
            raise ValueError("candidate discrete_net_gradient must start from zero")
    return state


def _selected_rows(state, transition):
    result = []
    rows = transition.selected_candidate_indices.detach().cpu().tolist()
    selected_types = getattr(transition, "selected_drive_type_ids", None)
    selected_types = (
        selected_types.detach().cpu().tolist()
        if selected_types is not None
        else [None] * len(rows)
    )
    for row_index, drive_type_id, net_index, candidate_id, raw_grad, improvement in zip(
        rows,
        selected_types,
        transition.selected_net_indices.detach().cpu().tolist(),
        transition.selected_candidate_ids.detach().cpu().tolist(),
        transition.selected_raw_grads.detach().cpu().tolist(),
        transition.selected_predicted_improvements.detach().cpu().tolist(),
        strict=True,
    ):
        record = state.candidate_records[int(row_index)]
        drive_type = None
        if drive_type_id is not None and getattr(state, "is_typed_candidate", False):
            drive_type = state.drive_type_table_metadata["masters"][int(drive_type_id)]
        result.append(
            {
                "candidate_row_index": int(row_index),
                "candidate_id": int(candidate_id),
                "drive_type_id": (
                    None if drive_type_id is None else int(drive_type_id)
                ),
                "drive_type_master_name": (
                    "" if drive_type is None else str(drive_type["master_name"])
                ),
                "drive_type_cell_id": (
                    None if drive_type is None else int(drive_type["cell_id"])
                ),
                "net_id": int(state.net_ids[int(net_index)].detach().cpu().item()),
                "bu_before": 0,
                "bu_after": 1,
                "raw_grad": float(raw_grad),
                "predicted_improvement": float(improvement),
                "candidate_node_id": int(
                    state.candidate_node_id[int(row_index)].detach().cpu().item()
                ),
                "x_dbu": record.get("x_dbu", record.get("candidate_location_x_dbu")),
                "y_dbu": record.get("y_dbu", record.get("candidate_location_y_dbu")),
                "parent_node_id": record.get("parent_node_id"),
                "child_node_ids": list(record.get("child_node_ids", []) or []),
                "segment_split_ratio": record.get("segment_split_ratio"),
                "downstream_pin_count": int(
                    record.get("downstream_pin_count", 0) or 0
                ),
            }
        )
    return result


def _verify_exact_projection(state, projection):
    validation = dict(getattr(projection, "validation", {}) or {})
    if getattr(state, "is_typed_candidate", False):
        active_rows = torch.nonzero(
            state.activation_param.detach().sum(dim=1) == 1.0,
            as_tuple=False,
        ).flatten()
        selected_types = state.selected_drive_type().detach()
        active_action_keys = tuple(
            sorted(
                (
                    int(state.candidate_ids[row].detach().cpu().item()),
                    int(selected_types[row].detach().cpu().item()),
                )
                for row in active_rows.detach().cpu().tolist()
            )
        )
        projected_action_keys = tuple(
            tuple(int(value) for value in item)
            for item in validation.get("projected_action_keys", ())
        )
        if active_action_keys != projected_action_keys:
            raise ValueError("typed candidate projection changed the active action set")
        projected_count = int(getattr(projection, "projected_buffer_count", 0) or 0)
        if projected_count != len(active_action_keys):
            raise ValueError("typed candidate projection reports an inconsistent buffer count")
        serialized = json.dumps(active_action_keys, separators=(",", ":"))
        return {
            "status": "ok",
            "active_action_count": len(active_action_keys),
            "projected_action_count": projected_count,
            "projected_candidate_count": projected_count,
            "active_action_keys": active_action_keys,
            "projected_action_keys": projected_action_keys,
            "exact_active_action_match": True,
            "active_action_keys_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
        }
    active_candidate_ids = tuple(
        sorted(
            int(value)
            for value in state.candidate_ids[
                state.activation_param.detach() == 1.0
            ].detach().cpu().tolist()
        )
    )
    projected_candidate_ids = tuple(
        int(value) for value in validation.get("projected_candidate_ids", ())
    )
    if active_candidate_ids != projected_candidate_ids:
        raise ValueError("candidate projection changed the active candidate set")
    projected_count = int(getattr(projection, "projected_buffer_count", 0) or 0)
    if projected_count != len(active_candidate_ids):
        raise ValueError("candidate projection reports an inconsistent buffer count")
    serialized = json.dumps(
        active_candidate_ids,
        separators=(",", ":"),
    )
    return {
        "status": "ok",
        "active_candidate_count": len(active_candidate_ids),
        "projected_candidate_count": projected_count,
        "active_candidate_ids": active_candidate_ids,
        "projected_candidate_ids": projected_candidate_ids,
        "exact_active_candidate_match": True,
        "active_candidate_ids_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
    }


def _backend_provenance(lane, model):
    summarize = getattr(lane, "summarize", None)
    summary = dict(summarize() or {}) if callable(summarize) else {}
    provider = getattr(model, "_relaxed_buffer_dynamic_net_provider", None)
    provider_metadata = dict(getattr(provider, "metadata", {}) or {})
    state = getattr(lane, "_current_state", lambda: None)()
    params = getattr(lane, "params", None)
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    legal_table = (
        dict(payload.get("buffer_legal_table", {}) or {})
        if isinstance(payload, dict)
        else {}
    )
    raw_fixed_bsu_index = getattr(state, "fixed_bsu_index", -1)
    fixed_bsu_index = (
        -1 if raw_fixed_bsu_index is None else int(raw_fixed_bsu_index)
    )
    legal_master_names = list(legal_table.get("legal_master_names", ()) or ())
    fixed_master_name = (
        str(legal_master_names[fixed_bsu_index])
        if 0 <= fixed_bsu_index < len(legal_master_names)
        else ""
    )
    op_collections = getattr(model, "op_collections", None)
    timing_op = getattr(op_collections, "timing_propagation_op", None)
    return {
        "mode": "candidate",
        "candidate_strategy": "discrete_net_gradient",
        "timing_objective": "PlaceObj.timing_obj",
        "timing_propagation_backend": (
            type(timing_op).__name__ if timing_op is not None else "unknown"
        ),
        "dynamic_net_provider": (
            type(provider).__name__ if provider is not None else "unknown"
        ),
        "net_subgraph_forward_backend": provider_metadata.get(
            "net_subgraph_forward_backend",
            summary.get("net_subgraph_forward_backend"),
        ),
        "net_subgraph_execution_device": provider_metadata.get(
            "net_subgraph_execution_device"
        ),
        "active_view_selector_used": provider_metadata.get(
            "active_view_selector_used"
        ),
        "native_active_view_selector_count": provider_metadata.get(
            "native_active_view_selector_count"
        ),
        "fixed_bsu_fused_forward_supported": provider_metadata.get(
            "fixed_bsu_fused_forward_supported"
        ),
        "fixed_bsu_fused_forward_fallback_reason": provider_metadata.get(
            "fixed_bsu_fused_forward_fallback_reason"
        ),
        "fixed_bsu_fused_forward_count": provider_metadata.get(
            "fixed_bsu_fused_forward_count"
        ),
        "buffer_device_lut_status": provider_metadata.get(
            "buffer_device_lut_status"
        ),
        "buffer_device_lut_source": provider_metadata.get(
            "buffer_device_lut_source"
        ),
        "driver_cap_overlay_count": provider_metadata.get(
            "driver_cap_overlay_count"
        ),
        "driver_cap_overlay_path": provider_metadata.get(
            "driver_cap_overlay_path"
        ),
        "timing_surrogate_mode": str(
            getattr(params, "timing_surrogate_mode", "") or ""
        ),
        "state_device": str(getattr(getattr(state, "activation_param", None), "device", "")),
        "timing_active_only": bool(getattr(state, "timing_active_only", False)),
        "fixed_bsu_index": fixed_bsu_index,
        "fixed_buffer_master_name": fixed_master_name,
        "candidate_drive_type_mode": summary.get(
            "candidate_drive_type_mode",
            getattr(lane.config, "candidate_drive_type_mode", "single_fixed"),
        ),
        "candidate_drive_type_table": summary.get(
            "candidate_drive_type_table"
        ),
        "candidate_policy": str(
            getattr(params, "buffering_candidate_policy", "segment_only")
        ),
        "candidate_generation_mode": summary.get("candidate_generation_mode"),
        "max_candidates_per_segment": summary.get(
            "max_candidates_per_segment"
        ),
        "include_tree_node_candidates": bool(
            getattr(params, "buffering_include_tree_node_candidates", 0)
        ),
        "full_design_scope": bool(summary.get("full_design_scope", True)),
        "physical_commit_owner": "PlacementEngine",
        "physical_commit_backend": str(
            getattr(params, "place_io_engine", "openroad") or "openroad"
        ),
        "candidate_count": int(summary.get("candidate_count", 0) or 0),
        "candidate_source": summary.get("candidate_source"),
        "candidate_bearing_net_count": summary.get("candidate_bearing_net_count"),
        "candidate_net_coverage_ratio": summary.get("candidate_net_coverage_ratio"),
        "supported_tree_net_count": summary.get("supported_tree_net_count"),
        "total_tree_edge_count": summary.get("total_tree_edge_count"),
        "coordinate_supported_edge_count": summary.get(
            "coordinate_supported_edge_count"
        ),
        "nonzero_tree_edge_count": summary.get("nonzero_tree_edge_count"),
        "edge_with_candidate_count": summary.get("edge_with_candidate_count"),
        "edge_without_candidate_count": summary.get(
            "edge_without_candidate_count"
        ),
        "candidate_attempt_count": summary.get("candidate_attempt_count"),
        "rounded_endpoint_skip_count": summary.get(
            "rounded_endpoint_skip_count"
        ),
        "coordinate_dedup_count": summary.get("coordinate_dedup_count"),
    }


def _typed_candidate_snapshot_identity(lane, state):
    """Digest the immutable per-round candidate/timing input boundary."""

    digest = hashlib.sha256()
    for name in (
        "candidate_ids",
        "candidate_net_id",
        "candidate_net_index",
        "net_ids",
        "activation_param",
    ):
        value = getattr(state, name, None)
        if value is None:
            digest.update((name + ":<none>").encode("utf-8"))
            continue
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    metadata = dict(payload.get("metadata", {}) or {}) if isinstance(payload, dict) else {}
    digest.update(
        json.dumps(
            {
                "topology_generation": metadata.get("topology_generation"),
                "frozen_topology_generation": metadata.get("frozen_topology_generation"),
                "candidate_generation_mode": metadata.get("candidate_generation_mode"),
                "candidate_count": int(state.candidate_ids.numel()),
                "net_count": int(state.net_ids.numel()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _write_trace(output_dir, payload):
    path = Path(output_dir) / "candidate_discrete_scheduler_trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    contents = (serialized + "\n").encode("utf-8")
    if path.exists():
        window_index = (
            len(list(path.parent.glob("candidate_discrete_scheduler_trace.window*.json")))
            + 1
        )
        path.replace(
            path.with_name(
                f"candidate_discrete_scheduler_trace.window{window_index:03d}.json"
            )
        )
    path.write_bytes(contents)
    return str(path), hashlib.sha256(contents).hexdigest()


def update_discrete_candidate_scheduler_trace_commit(trace_path, commit):
    path = Path(trace_path)
    if not path.is_file():
        raise ValueError(f"candidate scheduler trace does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != "candidate_discrete_scheduler_trace":
        raise ValueError(f"not a candidate scheduler trace: {path}")
    payload["commit"] = dict(commit or {})
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    contents = (serialized + "\n").encode("utf-8")
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()



def _run_discrete_candidate_buffer_scheduler_impl(
    *, lane, model, rounds, allow_existing_activation=False
):
    """Run Candidate Route-B on one frozen full-design candidate topology."""
    rounds = int(rounds)
    if rounds <= 0:
        raise ValueError("candidate discrete_net_gradient requires positive rounds")
    state = _assert_candidate_state(
        lane,
        require_zero=not bool(allow_existing_activation),
    )
    selection_fraction = float(
        getattr(lane.config, "route_b_selection_fraction", 0.001)
    )
    trace_rounds = []
    pending_record = None
    terminal_reason = "max_rounds"
    iterations = 0
    best_activation = state.activation_param.detach().clone()
    best_score = float("inf")
    best_key = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_round = 0
    best_snapshot = None

    for round_index in range(rounds):
        _assert_candidate_state(lane, require_zero=False)
        activation_before = state.activation_param.detach()
        snapshot_id = _typed_candidate_snapshot_identity(lane, state)
        drive_type_order = list(range(int(getattr(state, "drive_type_count", 1))))
        active_count_before = int(
            (
                activation_before.sum(dim=1) > 0.0
                if getattr(state, "is_typed_candidate", False)
                else activation_before == 1.0
            )
            .sum()
            .cpu()
            .item()
        )
        activation_binary_before = bool(
            ((activation_before == 0.0) | (activation_before == 1.0))
            .all()
            .cpu()
            .item()
        )
        forward_started_at = time.perf_counter()
        typed_gradient_metadata = {}
        if getattr(state, "is_typed_candidate", False):
            snapshot, raw_grad, typed_gradient_metadata = (
                _typed_candidate_objective_gradient(
                    lane,
                    model,
                    state,
                    snapshot_id=snapshot_id,
                )
            )
            loss = snapshot["loss"]
            phase_times = typed_gradient_metadata["phase_times_ms"]
            objective_forward_ms = sum(item["forward_ms"] for item in phase_times)
            objective_grad_ms = sum(item["backward_ms"] for item in phase_times)
        else:
            loss, metrics = lane.objective(model)
            objective_forward_ms = (time.perf_counter() - forward_started_at) * 1000.0
            snapshot = _objective_snapshot(model, loss, metrics)
        snapshot_key = (
            float(snapshot["restore_score"]),
            float(snapshot.get("slew_violation") or 0.0)
            + float(snapshot.get("cap_violation") or 0.0),
            -float(snapshot.get("wns") or 0.0),
            int(round_index),
        )
        if snapshot_key < best_key:
            best_key = snapshot_key
            best_score = float(snapshot["restore_score"])
            best_round = int(round_index)
            best_snapshot = dict(snapshot)
            best_activation = state.activation_param.detach().clone()
        if pending_record is not None:
            pending_record["actual_delta_loss"] = (
                snapshot["loss"] - pending_record["loss_before"]
            )
            pending_record["actual_delta_tns"] = (
                snapshot["tns"] - pending_record["tns_before"]
            )
            pending_record = None

        if not getattr(state, "is_typed_candidate", False):
            gradient_started_at = time.perf_counter()
            raw_grad, = torch.autograd.grad(
                loss,
                state.activation_param,
                retain_graph=False,
                create_graph=False,
            )
            objective_grad_ms = (time.perf_counter() - gradient_started_at) * 1000.0
        scheduler_started_at = time.perf_counter()
        if getattr(state, "is_typed_candidate", False):
            transition = schedule_typed_candidate_gradient_actions(
                candidate_net_index=state.candidate_net_index,
                net_ids=state.net_ids,
                candidate_ids=state.candidate_ids,
                activations=state.activation_param.detach(),
                raw_grad=raw_grad,
                selection_fraction=selection_fraction,
            )
        else:
            transition = schedule_candidate_gradient_actions(
                candidate_net_index=state.candidate_net_index,
                net_ids=state.net_ids,
                candidate_ids=state.candidate_ids,
                activations=state.activation_param.detach(),
                raw_grad=raw_grad,
                selection_fraction=selection_fraction,
            )
        scheduler_wall_ms = (time.perf_counter() - scheduler_started_at) * 1000.0
        next_activation = transition.next_activations.detach()
        active_count_after = int(
            (
                next_activation.sum(dim=1) > 0.0
                if getattr(state, "is_typed_candidate", False)
                else next_activation == 1.0
            )
            .sum()
            .cpu()
            .item()
        )
        activation_binary_after = bool(
            ((next_activation == 0.0) | (next_activation == 1.0))
            .all()
            .cpu()
            .item()
        )
        record = {
            "round": int(round_index),
            "forward_phase": "cold" if round_index == 0 else "warm",
            "snapshot_id": str(snapshot_id),
            "snapshot_state": "frozen_per_round",
            "drive_type_order": drive_type_order,
            "refresh_between_types": False,
            **snapshot,
            "loss_before": snapshot["loss"],
            "tns_before": snapshot["tns"],
            "objective_forward_ms": float(objective_forward_ms),
            "objective_grad_ms": float(objective_grad_ms),
            "scheduler_wall_ms": float(scheduler_wall_ms),
            "activation_binary_before": activation_binary_before,
            "activation_binary_after": activation_binary_after,
            "active_candidate_count_before": active_count_before,
            "active_candidate_count_after": active_count_after,
            "total_virtual_buffer_count": int(
                state.activation_param.detach().sum().cpu().item()
            ),
            "raw_grad_finite_count": int(
                torch.isfinite(raw_grad).sum().detach().cpu().item()
            ),
            "raw_grad_nonfinite_count": int(
                (~torch.isfinite(raw_grad)).sum().detach().cpu().item()
            ),
            "raw_grad_nonzero_count": int(
                (raw_grad != 0.0).sum().detach().cpu().item()
            ),
            "raw_grad_negative_count": int(
                (raw_grad < 0.0).sum().detach().cpu().item()
            ),
            "raw_grad_positive_count": int(
                (raw_grad > 0.0).sum().detach().cpu().item()
            ),
            "eligible_net_count": int(transition.eligible_net_count),
            "affected_net_count": int(transition.affected_net_count),
            "prefix_size": int(transition.prefix_size),
            "positive_prefix_count": int(transition.positive_prefix_count),
            "accepted_action_count": int(transition.accepted_count),
            "drive_type_count": int(getattr(state, "drive_type_count", 1)),
            "typed_gradient_metadata": typed_gradient_metadata,
            "predicted_improvement_sum": float(
                transition.selected_predicted_improvements.sum().detach().cpu().item()
            ),
            "selected_rows": _selected_rows(state, transition),
        }
        trace_rounds.append(record)
        iterations += 1
        if transition.accepted_count == 0:
            terminal_reason = (
                "all_candidates_active"
                if transition.eligible_net_count == 0
                else "no_positive_action"
            )
            break
        with torch.no_grad():
            state.activation_param.copy_(transition.next_activations)
        state.validate_binary_()
        pending_record = record

    _assert_candidate_state(lane, require_zero=False)
    if getattr(state, "is_typed_candidate", False):
        # The remaining activation is already a discrete state.  The
        # preceding rounds required all candidate rows for gradients; this
        # terminal check only needs the accepted-net subset.
        state.timing_active_only = True
    terminal_started_at = time.perf_counter()
    with torch.no_grad():
        terminal_loss, terminal_metrics = lane.objective(model)
    terminal_forward_ms = (time.perf_counter() - terminal_started_at) * 1000.0
    terminal_snapshot = _objective_snapshot(model, terminal_loss, terminal_metrics)
    terminal_key = (
        float(terminal_snapshot["restore_score"]),
        float(terminal_snapshot.get("slew_violation") or 0.0)
        + float(terminal_snapshot.get("cap_violation") or 0.0),
        -float(terminal_snapshot.get("wns") or 0.0),
        int(iterations),
    )
    if terminal_key < best_key:
        best_key = terminal_key
        best_score = float(terminal_snapshot["restore_score"])
        best_round = int(iterations)
        best_snapshot = dict(terminal_snapshot)
        best_activation = state.activation_param.detach().clone()

    # The controller owns one tuple-level restore boundary.  Keep the best
    # virtual activation selected by this phase and only then project it.
    # This is still virtual state; no native buffer topology is committed here.
    restore_needed = not torch.equal(state.activation_param.detach(), best_activation)
    with torch.no_grad():
        state.activation_param.copy_(best_activation)
    state.validate_binary_()
    # All candidate gradients above deliberately used the full candidate
    # domain.  Once the discrete action set is frozen, ordinary timing only
    # needs to re-evaluate nets that actually carry an accepted virtual
    # buffer.  Keep the full state for projection/writeback, but narrow the
    # dynamic provider for the terminal and subsequent GP timing calls.
    if getattr(state, "is_typed_candidate", False):
        state.timing_active_only = True
    if restore_needed:
        with torch.no_grad():
            restored_loss, restored_metrics = lane.objective(model)
        terminal = _objective_snapshot(model, restored_loss, restored_metrics)
    else:
        terminal = terminal_snapshot
    if pending_record is not None:
        pending_record["actual_delta_loss"] = (
            terminal_snapshot["loss"] - pending_record["loss_before"]
        )
        pending_record["actual_delta_tns"] = (
            terminal_snapshot["tns"] - pending_record["tns_before"]
        )

    projection_started_at = time.perf_counter()
    final_projection = lane.project(iteration=iterations, final=True)
    projection_materialization_ms = (
        time.perf_counter() - projection_started_at
    ) * 1000.0
    projection_validation = _verify_exact_projection(state, final_projection)
    runtime_refresh_started_at = time.perf_counter()
    runtime_refresh = lane.refresh_runtime_state(final_projection)
    runtime_refresh_ms = (
        time.perf_counter() - runtime_refresh_started_at
    ) * 1000.0
    final_projection_wall_ms = (time.perf_counter() - projection_started_at) * 1000.0
    terminal = {
        **terminal,
        "terminal_forward_ms": float(terminal_forward_ms),
        "total_virtual_buffer_count": int(
            state.activation_param.detach().sum().cpu().item()
        ),
    }
    payload = {
        "artifact": "candidate_discrete_scheduler_trace",
        "artifact_version": 1,
        "status": "completed",
        "mode": "candidate",
        "strategy": "discrete_net_gradient",
        "candidate_count": int(state.candidate_ids.numel()),
        "action_count": int(state.activation_param.numel()),
        "affected_net_count": int(state.net_ids.numel()),
        "fixed_bsu_index": (
            None
            if getattr(state, "is_typed_candidate", False)
            else int(state.fixed_bsu_index)
        ),
        "candidate_drive_type_mode": (
            "four_strength_table"
            if getattr(state, "is_typed_candidate", False)
            else "single_fixed"
        ),
        "drive_type_count": int(getattr(state, "drive_type_count", 1)),
        "selection_fraction": selection_fraction,
        "affected_nets_per_action": 1.0 / selection_fraction,
        "rounds_requested": rounds,
        "iterations": iterations,
        "terminal_reason": terminal_reason,
        "snapshot_policy": {
            "mode": "frozen_per_round",
            "typed_evaluation_order": list(range(int(getattr(state, "drive_type_count", 1)))),
            "refresh_between_types": False,
            "round_snapshot_ids": [
                str(record["snapshot_id"])
                for record in trace_rounds
            ],
        },
        "rounds": trace_rounds,
        "terminal": terminal,
        "best_restore": {
            "status": "restored",
            "initial_snapshot": {
                key: trace_rounds[0][key]
                for key in (
                    "loss",
                    "restore_score",
                    "wns",
                    "tns",
                    "slew_violation",
                    "cap_violation",
                    "total_slew_violation",
                    "total_cap_violation",
                    "weighted_terms",
                    "coefficients",
                    "weights",
                )
            },
            "best_round": int(best_round),
            "best_score": float(best_score),
            "best_snapshot": best_snapshot,
            "restored_activation_digest": _tensor_digest(
                (("candidate_activation", state.activation_param),)
            ),
        },
        "projection": projection_validation,
        "runtime_refresh": dict(runtime_refresh or {}),
        "runtime": {
            "cold_forward_ms": (
                float(trace_rounds[0]["objective_forward_ms"])
                if trace_rounds
                else None
            ),
            "warm_forward_ms": [
                float(record["objective_forward_ms"])
                for record in trace_rounds[1:]
            ],
            "gradient_ms": [
                float(record["objective_grad_ms"]) for record in trace_rounds
            ],
            "scheduler_ms": [
                float(record["scheduler_wall_ms"]) for record in trace_rounds
            ],
            "terminal_forward_ms": float(terminal_forward_ms),
            "projection_materialization_ms": float(projection_materialization_ms),
            "runtime_refresh_ms": float(runtime_refresh_ms),
            "projection_and_refresh_ms": float(final_projection_wall_ms),
        },
        "backend_provenance": _backend_provenance(lane, model),
        "commit": {
            "status": (
                "pending_buffer_commit_request"
                if bool(getattr(lane.config, "commit_enabled", False))
                and projection_validation["projected_candidate_count"] > 0
                else "disabled_or_no_actions"
            ),
            "commit_enabled": bool(getattr(lane.config, "commit_enabled", False)),
            "action_count": int(projection_validation["projected_candidate_count"]),
        },
    }
    trace_path, trace_sha256 = _write_trace(lane.config.output_dir, payload)
    return BufferingInnerLoopSummary(
        iterations=iterations,
        final_projection=final_projection,
        final_projection_wall_ms=float(final_projection_wall_ms),
        metrics_trace=tuple(_compact_round_record(record) for record in trace_rounds),
        projection_trace=(final_projection,),
        periodic_integer_projection_trace=(),
        strategy="discrete_net_gradient",
        terminal_metrics=terminal,
        terminal_reason=terminal_reason,
        trace_path=trace_path,
        trace_sha256=trace_sha256,
        best_restore=payload["best_restore"],
    )


def _release_candidate_python_state(lane):
    if bool(getattr(lane, "retain_candidate_python_state", False)):
        return
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    restore_gc = bool(
        getattr(lane, "_candidate_gc_restore_required", False)
        or metadata.get("python_gc_suspended_until_projection", False)
    )
    if not restore_gc:
        return

    state = getattr(lane, "_current_state", lambda: None)()
    if state is not None and hasattr(state, "candidate_records"):
        state.candidate_records = ()
    data_collections = getattr(lane, "data_collections", None)
    if data_collections is not None:
        for name in (
            "buffering_nets",
            "buffer_nets",
            "buffering_candidate_rows",
            "buffering_candidates",
            "buffer_candidates",
        ):
            if hasattr(data_collections, name):
                setattr(data_collections, name, ())
    metadata["python_gc_suspended_until_projection"] = False
    metadata["candidate_python_state_released"] = True
    lane._candidate_gc_restore_required = False
    gc.enable()


def run_discrete_candidate_buffer_scheduler(
    *, lane, model, rounds, allow_existing_activation=False
):
    try:
        with buffer_timing_fast_path(model):
            return _run_discrete_candidate_buffer_scheduler_impl(
                lane=lane,
                model=model,
                rounds=rounds,
                allow_existing_activation=allow_existing_activation,
            )
    finally:
        _release_candidate_python_state(lane)
