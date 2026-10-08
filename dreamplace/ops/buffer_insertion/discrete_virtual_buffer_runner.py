import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .buffering_inner_loop import BufferingInnerLoopSummary
from .discrete_virtual_scheduler import (
    schedule_net_gradient_actions,
    take_transition_prefix,
    transition_backtracking_sizes,
)


def _scalar(value, *, name):
    if value is None:
        return None
    if torch.is_tensor(value):
        if int(value.numel()) != 1:
            raise ValueError(f"{name} must be scalar")
        value = float(value.detach().cpu().item())
    else:
        value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _state_counts(state):
    counts = state.z_param.detach()
    if not bool(torch.isfinite(counts).all()):
        raise ValueError("discrete virtual state has nonfinite counts")
    if not bool((counts == torch.round(counts)).all()):
        raise ValueError("discrete virtual state must have integer counts")
    if not bool(
        ((counts >= 0.0) & (counts <= float(state.max_repeater_count))).all()
    ):
        raise ValueError("discrete virtual state exceeds its count bounds")
    return counts


def _assert_route_b_state(lane, *, require_zero):
    state = lane._current_state()
    if state is None or not hasattr(state, "z_param"):
        raise ValueError("discrete_net_gradient requires SegmentCountState")
    config = lane.config
    fixed_bsu_index = state.fixed_bsu_index
    if fixed_bsu_index is None:
        raise ValueError("discrete_net_gradient requires fixed_bsu_index")
    if bool(getattr(config, "segment_capacity_enabled", False)):
        raise ValueError("discrete_net_gradient cannot use segment capacity control")
    if getattr(config, "max_selected_actions", None) is not None:
        raise ValueError("discrete_net_gradient cannot use max_selected_actions")
    if int(getattr(config, "segment_integer_projection_interval", 0)) != 0:
        raise ValueError("discrete_net_gradient requires periodic projection disabled")
    if bool(getattr(config, "segment_projection_require_setup_criticality", False)):
        raise ValueError("discrete_net_gradient cannot use criticality filtering")
    if abs(float(getattr(config, "segment_projection_min_z_to_insert", 0.5)) - 0.5) > 1e-12:
        raise ValueError("discrete_net_gradient requires min_z_to_insert=0.5")
    counts = _state_counts(state)
    if require_zero and not bool((counts == 0.0).all()):
        raise ValueError("discrete_net_gradient must start from zero virtual buffers")
    bsu_value = state.bsu_index_param.detach()
    if not bool((bsu_value == float(fixed_bsu_index)).all()):
        raise ValueError("discrete_net_gradient requires every bsu entry to stay fixed")
    return state


def _metric_snapshot(loss, metrics):
    return {
        "loss": _scalar(loss, name="loss"),
        "wns": _scalar(metrics.get("wns"), name="wns"),
        "tns": _scalar(metrics.get("tns"), name="tns"),
    }


def _count_histogram(counts):
    values, frequencies = torch.unique(
        counts.detach().to(dtype=torch.long),
        return_counts=True,
    )
    return {
        str(key): int(value)
        for key, value in zip(
            values.detach().cpu().tolist(),
            frequencies.detach().cpu().tolist(),
        )
    }


def _selected_rows(state, transition):
    selected_indices = transition.selected_segment_indices.detach().cpu().tolist()
    if hasattr(state, "segment_records"):
        selected_records = state.segment_records(
            selected_indices,
            materialization_kind="selected",
        )
    else:
        selected_records = tuple(
            state.segment_rows[int(index)] for index in selected_indices
        )
    result = []
    for row, row_index, net_index, segment_id, raw_grad, improvement in zip(
        selected_records,
        selected_indices,
        transition.selected_net_indices.detach().cpu().tolist(),
        transition.selected_segment_ids.detach().cpu().tolist(),
        transition.selected_raw_grads.detach().cpu().tolist(),
        transition.selected_predicted_improvements.detach().cpu().tolist(),
    ):
        n_after = int(transition.next_counts[int(row_index)].detach().cpu().item())
        result.append(
            {
                "net_id": int(state.net_ids[int(net_index)].detach().cpu().item()),
                "segment_id": int(segment_id),
                "segment_row_index": int(row_index),
                "n_before": n_after - 1,
                "n_after": n_after,
                "raw_grad": float(raw_grad),
                "predicted_improvement": float(improvement),
                "parent_node_id": row.get("parent_node_id"),
                "child_node_id": row.get("child_node_id"),
            }
        )
    return result


def _compact_round_record(record):
    """Keep the canonical summary small; full action rows stay in the trace."""
    return {
        key: value
        for key, value in record.items()
        if key != "selected_rows"
    }


def _verify_exact_projection(state, projection):
    counts = _state_counts(state).to(dtype=torch.long)
    positive_index = torch.nonzero(counts > 0, as_tuple=False).flatten()
    positive_counts = counts.index_select(0, positive_index).detach().cpu()
    segment_ids = state.segment_ids.detach().index_select(
        0,
        positive_index.to(device=state.segment_ids.device),
    ).to(device="cpu", dtype=torch.long)
    expected = {
        int(segment_id): int(count)
        for segment_id, count in zip(segment_ids.tolist(), positive_counts.tolist())
    }
    actions = tuple(getattr(projection, "selected_actions", ()) or ())
    observed = Counter()
    for action in actions:
        if "segment_id" not in action:
            raise ValueError("Route B projection action is missing segment_id")
        segment_id = int(action["segment_id"])
        observed[segment_id] += 1
        projected_count = action.get("projected_repeater_count")
        if projected_count is not None and int(projected_count) != expected.get(segment_id):
            raise ValueError("Route B projection changed a segment repeater count")
    if dict(observed) != expected:
        raise ValueError(
            "Route B projection does not exactly expand the final integer segment state"
        )
    if int(getattr(projection, "projected_buffer_count", len(actions))) != len(actions):
        raise ValueError("Route B projection reports an inconsistent buffer count")
    serialized_counts = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    return {
        "status": "ok",
        "expected_total_buffer_count": int(sum(expected.values())),
        "projected_total_buffer_count": int(len(actions)),
        "active_segment_count": int(len(expected)),
        "per_segment_count_sha256": hashlib.sha256(
            serialized_counts.encode("utf-8")
        ).hexdigest(),
    }


def _write_trace(output_dir, payload):
    path = Path(output_dir) / "segment_discrete_scheduler_trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    contents = (serialized + "\n").encode("utf-8")
    path.write_bytes(contents)
    return str(path), hashlib.sha256(contents).hexdigest()


def update_discrete_virtual_scheduler_trace_commit(trace_path, commit):
    """Attach the PlacementEngine result to an existing Route B trace."""
    path = Path(trace_path)
    if not path.is_file():
        raise ValueError(f"Route B trace does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != "segment_discrete_scheduler_trace":
        raise ValueError(f"not a Route B scheduler trace: {path}")
    payload["commit"] = dict(commit or {})
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    contents = (serialized + "\n").encode("utf-8")
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def _backend_provenance(lane):
    summarize = getattr(lane, "summarize", None)
    summary = dict(summarize() or {}) if callable(summarize) else {}
    keys = (
        "segment_state_builder_backend_requested",
        "segment_state_builder_backend_used",
        "segment_state_builder_fallback_reason",
        "native_packer_thread_count",
        "segment_count_timing_backend_requested",
        "segment_count_timing_backend_used",
        "segment_count_timing_backend_fallback_reason",
        "segment_transfer_backend_requested",
        "segment_transfer_backend",
        "segment_transfer_backend_used",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _native_cap_backward_profile(model):
    provider = getattr(model, "_relaxed_buffer_dynamic_net_provider", None)
    metadata = getattr(provider, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    invocation_count = int(metadata.get("native_cap_backward_invocation_count", 0))
    if invocation_count <= 0:
        return None
    return {
        "invocation_count": invocation_count,
        "temporary_allocation_bytes": int(
            metadata.get("native_cap_backward_temporary_allocation_bytes", 0)
        ),
        "temporary_allocation_cpu_ms": float(
            metadata.get("native_cap_backward_temporary_allocation_cpu_ms", 0.0)
        ),
    }


def _discrete_count_proximal_action_grad(raw_grad, counts, lambda_z):
    lambda_z = float(lambda_z)
    if lambda_z == 0.0 or int(counts.numel()) == 0:
        return raw_grad.detach(), {
            "enabled": False,
            "lambda_z": lambda_z,
            "finite_increment_applied": False,
        }
    # Each standalone Route-B round is one outer buffer block, so its anchor
    # is the legal count state at the start of that round.
    correction = lambda_z / (2.0 * float(counts.numel()))
    action_grad = raw_grad.detach() + correction
    return action_grad, {
        "enabled": True,
        "lambda_z": lambda_z,
        "normalizer": int(counts.numel()),
        "finite_increment_applied": True,
        "correction": correction,
    }


def run_discrete_virtual_buffer_scheduler(*, lane, model, rounds):
    """Execute Route B over one frozen full-design segment state."""
    rounds = int(rounds)
    if rounds <= 0:
        raise ValueError("discrete_net_gradient requires a positive round count")
    state = _assert_route_b_state(lane, require_zero=True)
    trace_rounds = []
    terminal_reason = "max_rounds"
    iterations = 0
    selection_fraction = float(
        getattr(lane.config, "route_b_selection_fraction", 0.001)
    )

    for round_index in range(rounds):
        _assert_route_b_state(lane, require_zero=False)
        forward_started_at = time.perf_counter()
        loss, metrics = lane.objective(model)
        objective_forward_ms = float((time.perf_counter() - forward_started_at) * 1000.0)
        snapshot = _metric_snapshot(loss, metrics)

        gradient_started_at = time.perf_counter()
        raw_grad, = torch.autograd.grad(
            loss,
            state.z_param,
            retain_graph=False,
            create_graph=False,
        )
        objective_grad_ms = float((time.perf_counter() - gradient_started_at) * 1000.0)
        native_cap_backward_profile = _native_cap_backward_profile(model)
        counts_before = state.z_param.detach().clone()
        action_grad, proximal_adjustment = _discrete_count_proximal_action_grad(
            raw_grad,
            counts_before,
            lane.config.discrete_count_proximal_lambda,
        )
        scheduler_started_at = time.perf_counter()
        transition = schedule_net_gradient_actions(
            segment_net_index=state.segment_net_index,
            net_ids=state.net_ids,
            segment_ids=state.segment_ids,
            counts=counts_before,
            raw_grad=action_grad,
            max_count=state.max_repeater_count,
            selection_fraction=selection_fraction,
        )
        gradient_selected_action_count = int(transition.accepted_count)
        trial_prefixes = []
        accepted_snapshot = None
        if gradient_selected_action_count > 0:
            for selected_count in transition_backtracking_sizes(
                gradient_selected_action_count
            ):
                candidate = take_transition_prefix(
                    transition,
                    counts=counts_before,
                    selected_count=selected_count,
                )
                with torch.no_grad():
                    state.z_param.copy_(candidate.next_counts)
                    trial_loss, trial_metrics = lane.objective(model)
                trial_snapshot = _metric_snapshot(trial_loss, trial_metrics)
                delta_loss = trial_snapshot["loss"] - snapshot["loss"]
                delta_tns = trial_snapshot["tns"] - snapshot["tns"]
                trial_prefixes.append(
                    {
                        "selected_action_count": int(selected_count),
                        "loss": trial_snapshot["loss"],
                        "tns": trial_snapshot["tns"],
                        "delta_loss": delta_loss,
                        "delta_tns": delta_tns,
                        "accepted": bool(delta_loss < 0.0),
                    }
                )
                if delta_loss < 0.0:
                    transition = candidate
                    accepted_snapshot = trial_snapshot
                    break
            else:
                with torch.no_grad():
                    state.z_param.copy_(counts_before)
                transition = take_transition_prefix(
                    transition,
                    counts=counts_before,
                    selected_count=0,
                )
        scheduler_wall_ms = float((time.perf_counter() - scheduler_started_at) * 1000.0)
        selected_rows = _selected_rows(state, transition)
        record = {
            "round": int(round_index),
            **snapshot,
            "loss_before": snapshot["loss"],
            "tns_before": snapshot["tns"],
            "objective_forward_ms": objective_forward_ms,
            "objective_grad_ms": objective_grad_ms,
            "scheduler_wall_ms": scheduler_wall_ms,
            "integer_count_histogram": _count_histogram(counts_before),
            "total_virtual_buffer_count": int(counts_before.sum().cpu().item()),
            "raw_grad_nonzero_count": int((raw_grad != 0.0).sum().detach().cpu().item()),
            "raw_grad_negative_count": int((raw_grad < 0.0).sum().detach().cpu().item()),
            "raw_grad_positive_count": int((raw_grad > 0.0).sum().detach().cpu().item()),
            "action_grad_norm": float(action_grad.float().norm().detach().cpu().item()),
            "proximal_adjustment": proximal_adjustment,
            "eligible_net_count": int(transition.eligible_net_count),
            "affected_net_count": int(transition.affected_net_count),
            "selection_fraction": selection_fraction,
            "prefix_size": int(transition.prefix_size),
            "positive_prefix_count": int(transition.positive_prefix_count),
            "gradient_selected_action_count": gradient_selected_action_count,
            "accepted_action_count": int(transition.accepted_count),
            "predicted_improvement_sum": float(
                transition.selected_predicted_improvements.sum().detach().cpu().item()
            ),
            "batch_acceptance_policy": "monotonic_loss_prefix_backtracking",
            "trial_prefixes": trial_prefixes,
            "selected_rows": selected_rows,
        }
        if accepted_snapshot is not None:
            record["actual_delta_loss"] = (
                accepted_snapshot["loss"] - snapshot["loss"]
            )
            record["actual_delta_tns"] = (
                accepted_snapshot["tns"] - snapshot["tns"]
            )
        if native_cap_backward_profile is not None:
            record["native_cap_backward_profile"] = native_cap_backward_profile
        trace_rounds.append(record)
        iterations += 1
        if transition.accepted_count == 0:
            terminal_reason = (
                "no_improving_prefix"
                if gradient_selected_action_count > 0
                else "no_positive_action"
            )
            break

    _assert_route_b_state(lane, require_zero=False)
    terminal_started_at = time.perf_counter()
    with torch.no_grad():
        terminal_loss, terminal_metrics = lane.objective(model)
    terminal_forward_ms = float((time.perf_counter() - terminal_started_at) * 1000.0)
    terminal_snapshot = _metric_snapshot(terminal_loss, terminal_metrics)

    projection_started_at = time.perf_counter()
    final_projection = lane.project(iteration=iterations, final=True)
    projection_validation = _verify_exact_projection(state, final_projection)
    runtime_refresh = lane.refresh_runtime_state(final_projection)
    final_projection_wall_ms = float(
        (time.perf_counter() - projection_started_at) * 1000.0
    )
    state_summary = dict(getattr(state, "summary", {}) or {})
    payload = {
        "artifact": "segment_discrete_scheduler_trace",
        "artifact_version": 1,
        "status": "completed",
        "strategy": "discrete_net_gradient",
        "segment_count": int(state.z_param.numel()),
        "affected_net_count": int(state.net_ids.numel()),
        "max_repeater_count": int(state.max_repeater_count),
        "fixed_bsu_index": int(state.fixed_bsu_index),
        "rounds_requested": rounds,
        "iterations": iterations,
        "terminal_reason": terminal_reason,
        "rounds": trace_rounds,
        "terminal": {
            **terminal_snapshot,
            "terminal_forward_ms": terminal_forward_ms,
            "integer_count_histogram": _count_histogram(state.z_param),
            "total_virtual_buffer_count": int(state.z_param.detach().sum().cpu().item()),
        },
        "projection": projection_validation,
        "row_materialization": {
            "full_row_materialization_count": int(
                state_summary.get("full_row_materialization_count", 0)
            ),
            "selected_row_materialization_count": int(
                state_summary.get("selected_row_materialization_count", 0)
            ),
            "terminal_row_materialization_count": int(
                state_summary.get("terminal_row_materialization_count", 0)
            ),
        },
        "runtime_refresh": dict(runtime_refresh or {}),
        "backend_provenance": _backend_provenance(lane),
        "commit": {
            "status": (
                "pending_buffer_commit_request"
                if bool(getattr(lane.config, "commit_enabled", False))
                and int(projection_validation["projected_total_buffer_count"]) > 0
                else "disabled_or_no_actions"
            ),
            "commit_enabled": bool(getattr(lane.config, "commit_enabled", False)),
            "action_count": int(projection_validation["projected_total_buffer_count"]),
        },
    }
    trace_path, trace_sha256 = _write_trace(lane.config.output_dir, payload)
    return BufferingInnerLoopSummary(
        iterations=iterations,
        final_projection=final_projection,
        final_projection_wall_ms=final_projection_wall_ms,
        metrics_trace=tuple(_compact_round_record(record) for record in trace_rounds),
        projection_trace=(final_projection,),
        periodic_integer_projection_trace=(),
        strategy="discrete_net_gradient",
        terminal_metrics=payload["terminal"],
        terminal_reason=terminal_reason,
        trace_path=trace_path,
        trace_sha256=trace_sha256,
    )
