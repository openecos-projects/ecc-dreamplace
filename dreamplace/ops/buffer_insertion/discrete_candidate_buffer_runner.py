import hashlib
import gc
import json
import time
from pathlib import Path

import torch

from .buffering_inner_loop import BufferingInnerLoopSummary
from .discrete_candidate_scheduler import schedule_candidate_gradient_actions
from .discrete_virtual_buffer_runner import _compact_round_record, _metric_snapshot
from .discrete_virtual_scheduler import ROUTE_B_AFFECTED_NETS_PER_ACTION


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
    fixed_bsu_index = state.fixed_bsu_index
    if fixed_bsu_index is None:
        raise ValueError("candidate discrete_net_gradient requires fixed_bsu_index")
    if bool(getattr(config, "segment_capacity_enabled", False)):
        raise ValueError("candidate discrete_net_gradient cannot use capacity control")
    if getattr(config, "max_selected_actions", None) is not None:
        raise ValueError("candidate discrete_net_gradient cannot use max_selected_actions")
    state.validate_binary_()
    if require_zero and not bool((state.activation_param.detach() == 0.0).all()):
        raise ValueError("candidate discrete_net_gradient must start from zero")
    return state


def _selected_rows(state, transition):
    result = []
    rows = transition.selected_candidate_indices.detach().cpu().tolist()
    for row_index, net_index, candidate_id, raw_grad, improvement in zip(
        rows,
        transition.selected_net_indices.detach().cpu().tolist(),
        transition.selected_candidate_ids.detach().cpu().tolist(),
        transition.selected_raw_grads.detach().cpu().tolist(),
        transition.selected_predicted_improvements.detach().cpu().tolist(),
    ):
        record = state.candidate_records[int(row_index)]
        result.append(
            {
                "candidate_row_index": int(row_index),
                "candidate_id": int(candidate_id),
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
    active_candidate_ids = tuple(
        sorted(
            int(value)
            for value in state.candidate_ids[
                state.activation_param.detach() == 1.0
            ].detach().cpu().tolist()
        )
    )
    validation = dict(getattr(projection, "validation", {}) or {})
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
    fixed_bsu_index = int(getattr(state, "fixed_bsu_index", -1))
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
        "fixed_bsu_index": fixed_bsu_index,
        "fixed_buffer_master_name": fixed_master_name,
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


def _write_trace(output_dir, payload):
    path = Path(output_dir) / "candidate_discrete_scheduler_trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    contents = (serialized + "\n").encode("utf-8")
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


def _run_discrete_candidate_buffer_scheduler_impl(*, lane, model, rounds):
    """Run Candidate Route-B on one frozen full-design candidate topology."""
    rounds = int(rounds)
    if rounds <= 0:
        raise ValueError("candidate discrete_net_gradient requires positive rounds")
    state = _assert_candidate_state(lane, require_zero=True)
    trace_rounds = []
    pending_record = None
    terminal_reason = "max_rounds"
    iterations = 0

    for round_index in range(rounds):
        _assert_candidate_state(lane, require_zero=False)
        activation_before = state.activation_param.detach()
        active_count_before = int((activation_before == 1.0).sum().cpu().item())
        activation_binary_before = bool(
            ((activation_before == 0.0) | (activation_before == 1.0))
            .all()
            .cpu()
            .item()
        )
        forward_started_at = time.perf_counter()
        loss, metrics = lane.objective(model)
        objective_forward_ms = (time.perf_counter() - forward_started_at) * 1000.0
        snapshot = _metric_snapshot(loss, metrics)
        if pending_record is not None:
            pending_record["actual_delta_loss"] = (
                snapshot["loss"] - pending_record["loss_before"]
            )
            pending_record["actual_delta_tns"] = (
                snapshot["tns"] - pending_record["tns_before"]
            )
            pending_record = None

        gradient_started_at = time.perf_counter()
        raw_grad, = torch.autograd.grad(
            loss,
            state.activation_param,
            retain_graph=False,
            create_graph=False,
        )
        objective_grad_ms = (time.perf_counter() - gradient_started_at) * 1000.0
        scheduler_started_at = time.perf_counter()
        transition = schedule_candidate_gradient_actions(
            candidate_net_index=state.candidate_net_index,
            net_ids=state.net_ids,
            candidate_ids=state.candidate_ids,
            activations=state.activation_param.detach(),
            raw_grad=raw_grad,
        )
        scheduler_wall_ms = (time.perf_counter() - scheduler_started_at) * 1000.0
        next_activation = transition.next_activations.detach()
        active_count_after = int((next_activation == 1.0).sum().cpu().item())
        activation_binary_after = bool(
            ((next_activation == 0.0) | (next_activation == 1.0))
            .all()
            .cpu()
            .item()
        )
        record = {
            "round": int(round_index),
            "forward_phase": "cold" if round_index == 0 else "warm",
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
        pending_record = record

    _assert_candidate_state(lane, require_zero=False)
    terminal_started_at = time.perf_counter()
    with torch.no_grad():
        terminal_loss, terminal_metrics = lane.objective(model)
    terminal_forward_ms = (time.perf_counter() - terminal_started_at) * 1000.0
    terminal_snapshot = _metric_snapshot(terminal_loss, terminal_metrics)
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
        **terminal_snapshot,
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
        "candidate_count": int(state.activation_param.numel()),
        "affected_net_count": int(state.net_ids.numel()),
        "fixed_bsu_index": int(state.fixed_bsu_index),
        "selection_fraction": 1.0 / float(ROUTE_B_AFFECTED_NETS_PER_ACTION),
        "affected_nets_per_action": int(ROUTE_B_AFFECTED_NETS_PER_ACTION),
        "rounds_requested": rounds,
        "iterations": iterations,
        "terminal_reason": terminal_reason,
        "rounds": trace_rounds,
        "terminal": terminal,
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
    )


def _release_candidate_python_state(lane):
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


def run_discrete_candidate_buffer_scheduler(*, lane, model, rounds):
    try:
        return _run_discrete_candidate_buffer_scheduler_impl(
            lane=lane,
            model=model,
            rounds=rounds,
        )
    finally:
        _release_candidate_python_state(lane)
