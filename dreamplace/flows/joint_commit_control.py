"""Joint request finalization and safe-stop reporting; no native mutation."""
from dataclasses import dataclass
from typing import Callable
import copy
import logging
from dreamplace.flows.metric_state import flatten_metrics, last_metric_from_metrics
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker

@dataclass
class JointCommitContext:
    joint_coordinator: object
    trace: object
    artifact: object
    last_metrics: object
    physical_context: Callable
    projection_context: Callable
    publish_request: Callable
    update_artifact: Callable
    write_artifact: Callable
    write_metrics: Callable
    write_debug: Callable
    post_commit_result: object = None

def finalize_joint_buffer_commit(
    context,
    params,
    *,
    iteration,
    optimizer=None,
    final_pos=None,
):
    coordinator = getattr(context, "joint_coordinator", None)
    if coordinator is None:
        return None
    trace = getattr(context, "trace", None)
    if trace:
        context.update_artifact(params)
        return None
    live_projection_context = None
    final_physical_context = None
    if final_pos is not None:
        final_physical_context = context.physical_context(
            final_pos, iteration=iteration,
        )
    if coordinator.is_segment_direct_joint:
        if final_pos is None:
            raise RuntimeError(
                "segment joint final commit requires the final placement snapshot"
            )
        live_projection_context = context.projection_context(
            final_pos,
            iteration=iteration,
        )
        if live_projection_context is None:
            result = {
                "status": "skipped",
                "reason": "segment_joint_window_not_activated",
                "iteration": int(iteration),
            }
            context.update_artifact(params)
            return result
    commit_result = coordinator.prepare_buffer_commit_request(
        iteration=iteration,
        optimizer=optimizer,
        reason="joint_buffer_final_commit",
        live_projection_context=live_projection_context,
        final_physical_context=final_physical_context,
    )
    context.publish_request(coordinator.last_buffer_commit_request, commit_result, reset_trace=True)
    context.update_artifact(params)
    return commit_result


def joint_post_commit_control_result(commit_result):
    if (
        isinstance(commit_result, dict)
        and commit_result.get("status") == "request_ready"
    ):
        request = dict(commit_result.get("commit_request", {}) or {})
        return {
            "status": "pending_buffer_commit_request",
            "topology_mutated": False,
            "refresh_required": False,
            "refresh_mode": request.get("refresh_mode", "topo"),
            "rebuild_mode": request.get("rebuild_mode", "topo"),
            "continuation_policy": "execute_in_placement_engine",
            "accepted_action_count": 0,
            "commit_request": request,
        }
    commit = {}
    if isinstance(commit_result, dict):
        commit = dict(commit_result.get("commit", {}) or {})
    accepted_action_count = commit.get("accepted_action_count")
    try:
        accepted_action_count = int(accepted_action_count or 0)
    except (TypeError, ValueError):
        accepted_action_count = 0
    topology_mutated = (
        commit.get("status") == "accepted" and accepted_action_count > 0
    )
    return {
        "status": (
            "safe_stopped_after_accepted_buffer_commit"
            if topology_mutated
            else "completed"
        ),
        "topology_mutated": bool(topology_mutated),
        "refresh_required": bool(topology_mutated),
        "refresh_mode": commit.get("refresh_mode", "topo"),
        "rebuild_mode": commit.get("rebuild_mode", "topo"),
        "continuation_policy": (
            "rebuild_placer_outer_loop"
            if topology_mutated
            else "continue_same_topology"
        ),
        "accepted_action_count": accepted_action_count,
        "commit": commit,
    }


def joint_buffer_request_requires_outer_commit(request_result, request):
    if not isinstance(request_result, dict):
        return False
    if request_result.get("status") != "request_ready":
        return False
    return bool(
        request is not None
        and bool(getattr(request, "commit_enabled", False))
        and not bool(getattr(request, "is_noop", False))
    )


def safe_stop_after_joint_buffer_request(
    context,
    params,
    *,
    iteration,
    place_stage_metrics,
    processed_metrics,
    commit_result,
):
    logging.info(
        "joint buffer commit request is ready; stop the inner window before "
        "physical mutation so PlacementEngine can own commit and rebuild"
    )
    if not place_stage_metrics:
        place_stage_metrics = copy.deepcopy(
            [
                metric
                for metric in flatten_metrics(getattr(context, "last_metrics", []))
                if metric is not None
            ]
        )
    write_crash_stage_marker(
        "nonlinearplace_joint_buffer_request_safe_stop",
        "start",
        iteration=iteration,
        projected_buffer_count=commit_result.get("projected_buffer_count"),
    )
    try:
        context.write_metrics(params, place_stage_metrics, stage_name="place")
    except Exception as exc:
        logging.warning(
            "failed to write place metrics before joint buffer commit safe-stop: %s",
            exc,
        )
    summary_payload = getattr(context, "artifact", None)
    if isinstance(summary_payload, dict):
        summary = summary_payload.get("summary")
        if isinstance(summary, dict):
            summary["post_commit_control_policy"] = {
                "status": "safe_stopped_before_buffer_commit",
                "reason": "physical_buffer_commit_owned_by_placement_engine",
                "projected_buffer_count": commit_result.get(
                    "projected_buffer_count"
                ),
            }
            summary["post_commit_control_result"] = (
                joint_post_commit_control_result(commit_result)
            )
            context.write_artifact(params, summary)
    write_crash_stage_marker(
        "nonlinearplace_joint_buffer_request_safe_stop",
        "done",
        iteration=iteration,
        reason="pending_buffer_commit_request",
    )
    processed_metrics = dict(processed_metrics or {})
    processed_metrics["joint_buffer_commit_safe_stop"] = {
        "status": "safe_stopped_before_buffer_commit",
        "projected_buffer_count": commit_result.get("projected_buffer_count"),
    }
    processed_metrics["post_commit_control_result"] = (
        joint_post_commit_control_result(commit_result)
    )
    context.post_commit_result = dict(
        processed_metrics["post_commit_control_result"]
    )
    context.write_debug(
        params,
        iteration=iteration,
        optimizer_steps=len(place_stage_metrics),
        processed_metrics=processed_metrics,
        last_metric=last_metric_from_metrics(place_stage_metrics),
        status="ok",
        stop_reason="pending_buffer_commit_request",
        entered_legalization=False,
        skipped_legalization_reason=None,
    )
    return float("nan"), float("nan"), processed_metrics

