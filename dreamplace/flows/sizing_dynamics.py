"""ECC sizing dynamics owner; live state is supplied per call."""
from dataclasses import dataclass
from typing import Callable
import torch

UNSET = object()

@dataclass
class SizingDynamicsContext:
    accept_reject: Callable
    apply_metric: Callable
    combined_loss: Callable
    commit_feedback: Callable
    config: Callable
    data_collections: object
    discrete_step: Callable
    evaluate_candidate: Callable
    metric_snapshot: Callable
    metric_values: Callable
    op_collections: object
    phase: Callable
    record_timing: Callable
    timing_scalar: Callable
    update_guard: Callable
    summary: object = UNSET

def build_continuous_size_dynamics_state(context, params):
    config = context.config(params)
    mode = config["mode"]
    enabled = mode in (
        "late_adaptive_guard",
        "late_zero_vio_guard",
        "late_accept_reject",
        "late_commit_feedback",
    )
    state = {
        "enabled": enabled,
        "mode": mode,
        "patience": config["trigger_patience"],
        "min_delta": config["trigger_min_delta"],
        "zero_vio_tol": config["zero_vio_tol"],
        "monitoring_ready": False,
        "monitoring_started": False,
        "monitoring_start_iteration": None,
        "best_loss": None,
        "best_iteration": None,
        "last_loss": None,
        "last_iteration": None,
        "no_improve_rounds": 0,
        "triggered": False,
        "trigger_iteration": None,
        "trigger_reason": None,
        "zero_vio_ready": False,
        "last_slew_violation": None,
        "last_cap_violation": None,
    }
    if mode in ("late_accept_reject", "late_commit_feedback"):
        state.update(
            {
                "accept_count": 0,
                "reject_count": 0,
                "rejection_streak": 0,
                "late_step_scale": float(config["late_lr_scale"]),
                "accept_min_delta": float(config["accept_min_delta"]),
                "reject_scale": float(config["reject_scale"]),
                "recover_scale": float(config["recover_scale"]),
                "base_late_step_scale": float(config["late_lr_scale"]),
            }
        )
    if mode == "late_accept_reject":
        state.update(
            {
                "anchor_logits": None,
                "anchor_loss": None,
                "anchor_metric_snapshot": None,
            }
        )
    if mode == "late_commit_feedback":
        state.update(
            {
                "committed_logits": None,
                "committed_loss": None,
                "committed_metric_snapshot": None,
                "previous_loss": None,
            }
        )
    return state


def continuous_size_dynamics_metric_values(context, metric):
    if metric is None:
        return None, None, None
    return (
        context.timing_scalar(getattr(metric, "combined_timing_loss", None)),
        context.timing_scalar(getattr(metric, "slew_violation", None)),
        context.timing_scalar(getattr(metric, "cap_violation", None)),
    )


def update_continuous_size_dynamics_state(context, state, metric, iteration, phase):
    summary = {
        "monitoring_ready": False,
        "monitoring_started": False,
        "best_loss": None,
        "best_iteration": None,
        "last_loss": None,
        "last_iteration": None,
        "no_improve_rounds": 0,
        "triggered": False,
        "trigger_iteration": None,
        "trigger_reason": None,
        "zero_vio_ready": None,
        "last_slew_violation": None,
        "last_cap_violation": None,
    }
    if not isinstance(state, dict) or not state.get("enabled"):
        return summary

    combined_timing_loss, slew_violation, cap_violation = context.metric_values(
        metric
    )
    if slew_violation is not None:
        state["last_slew_violation"] = float(slew_violation)
    if cap_violation is not None:
        state["last_cap_violation"] = float(cap_violation)

    zero_vio_tol = float(state.get("zero_vio_tol", 0.0))
    zero_vio_ready = (
        slew_violation is not None
        and cap_violation is not None
        and float(slew_violation) <= zero_vio_tol
        and float(cap_violation) <= zero_vio_tol
    )
    state["zero_vio_ready"] = zero_vio_ready

    monitoring_ready = phase == "late"
    if state.get("mode") == "late_zero_vio_guard":
        monitoring_ready = monitoring_ready and zero_vio_ready
    state["monitoring_ready"] = monitoring_ready

    if combined_timing_loss is not None:
        state["last_loss"] = float(combined_timing_loss)
        state["last_iteration"] = int(iteration)

    if not monitoring_ready:
        if not state.get("triggered"):
            state["monitoring_started"] = False
            state["monitoring_start_iteration"] = None
            state["best_loss"] = None
            state["best_iteration"] = None
            state["no_improve_rounds"] = 0
    elif combined_timing_loss is not None and not state.get("triggered"):
        loss = float(combined_timing_loss)
        if not state.get("monitoring_started"):
            state["monitoring_started"] = True
            state["monitoring_start_iteration"] = int(iteration)
            state["best_loss"] = loss
            state["best_iteration"] = int(iteration)
            state["no_improve_rounds"] = 0
        else:
            best_loss = state.get("best_loss")
            min_delta = float(state.get("min_delta", 0.0))
            if best_loss is None or loss < float(best_loss) - min_delta:
                state["best_loss"] = loss
                state["best_iteration"] = int(iteration)
                state["no_improve_rounds"] = 0
            else:
                state["no_improve_rounds"] = int(state.get("no_improve_rounds", 0)) + 1
                patience = int(state.get("patience") or 0)
                if patience > 0 and state["no_improve_rounds"] >= patience:
                    state["triggered"] = True
                    state["trigger_iteration"] = int(iteration)
                    state["trigger_reason"] = (
                        "late_no_improvement_after_zero_vio"
                        if state.get("mode") == "late_zero_vio_guard"
                        else "late_no_improvement"
                    )

    summary.update(
        {
            "monitoring_ready": bool(state.get("monitoring_ready")),
            "monitoring_started": bool(state.get("monitoring_started")),
            "best_loss": state.get("best_loss"),
            "best_iteration": state.get("best_iteration"),
            "last_loss": state.get("last_loss"),
            "last_iteration": state.get("last_iteration"),
            "no_improve_rounds": state.get("no_improve_rounds"),
            "triggered": bool(state.get("triggered")),
            "trigger_iteration": state.get("trigger_iteration"),
            "trigger_reason": state.get("trigger_reason"),
            "zero_vio_ready": bool(state.get("zero_vio_ready")),
            "last_slew_violation": state.get("last_slew_violation"),
            "last_cap_violation": state.get("last_cap_violation"),
        }
    )
    return summary


def apply_late_accept_reject_size_dynamics(
    context,
    state,
    before_logits,
    metric,
    phase,
    phase_lr_scale,
    config,
    candidate_metric_snapshot=None,
):
    summary = {
        "step_decision": "accepted",
        "rolled_back": False,
        "accept_count": 0,
        "reject_count": 0,
        "rejection_streak": 0,
        "anchor_loss": None,
        "late_step_scale": float(state.get("late_step_scale", config["late_lr_scale"])),
        "accept_min_delta": float(config["accept_min_delta"]),
        "effective_accept_min_delta": float(config["accept_min_delta"]),
        "reject_scale": float(config["reject_scale"]),
        "recover_scale": float(config["recover_scale"]),
        "applied_lr_scale": float(phase_lr_scale),
        "anchor_available": state.get("anchor_logits") is not None,
        "metric_snapshot": None,
    }
    if not isinstance(state, dict) or state.get("mode") != "late_accept_reject":
        return summary

    candidate_metric_snapshot = candidate_metric_snapshot or context.metric_snapshot(
        metric
    )
    current_loss = candidate_metric_snapshot.get("combined_timing_loss")
    current_late_step_scale = float(state.get("late_step_scale", config["late_lr_scale"]))
    summary["late_step_scale"] = current_late_step_scale
    base_accept_min_delta = float(state.get("accept_min_delta", config["accept_min_delta"]))
    base_late_step_scale = float(state.get("base_late_step_scale", config["late_lr_scale"]))
    effective_accept_min_delta = base_accept_min_delta
    if base_late_step_scale > 0.0:
        effective_accept_min_delta = base_accept_min_delta * (
            current_late_step_scale / base_late_step_scale
        )
    summary["effective_accept_min_delta"] = max(0.0, float(effective_accept_min_delta))
    current_logits = getattr(getattr(context, "data_collections", None), "size_logits", None)
    accepted_logits = before_logits.detach().clone()
    if current_logits is not None:
        accepted_logits = before_logits.detach() + (
            current_logits.detach() - before_logits.detach()
        ) * current_late_step_scale

    if phase != "late":
        summary["accept_count"] = int(state.get("accept_count", 0))
        summary["reject_count"] = int(state.get("reject_count", 0))
        summary["rejection_streak"] = int(state.get("rejection_streak", 0))
        summary["anchor_loss"] = state.get("anchor_loss")
        summary["metric_snapshot"] = candidate_metric_snapshot
        return summary

    anchor_logits = state.get("anchor_logits")
    anchor_loss = state.get("anchor_loss")
    if anchor_logits is None or anchor_loss is None or current_loss is None:
        state["anchor_logits"] = accepted_logits.detach().clone()
        state["anchor_loss"] = None if current_loss is None else float(current_loss)
        state["anchor_metric_snapshot"] = dict(candidate_metric_snapshot)
        state["accept_count"] = int(state.get("accept_count", 0)) + 1
        state["rejection_streak"] = 0
        summary["accept_count"] = int(state.get("accept_count", 0))
        summary["reject_count"] = int(state.get("reject_count", 0))
        summary["rejection_streak"] = 0
        summary["anchor_loss"] = state.get("anchor_loss")
        summary["applied_lr_scale"] = current_late_step_scale
        summary["anchor_available"] = True
        summary["metric_snapshot"] = dict(candidate_metric_snapshot)
        return summary

    if float(current_loss) <= float(anchor_loss) - summary["effective_accept_min_delta"]:
        state["anchor_logits"] = accepted_logits.detach().clone()
        state["anchor_loss"] = float(current_loss)
        state["anchor_metric_snapshot"] = dict(candidate_metric_snapshot)
        state["accept_count"] = int(state.get("accept_count", 0)) + 1
        state["rejection_streak"] = 0
        recover_scale = float(state.get("recover_scale", config["recover_scale"]))
        state["late_step_scale"] = min(
            base_late_step_scale,
            current_late_step_scale * recover_scale,
        )
        summary["accept_count"] = int(state.get("accept_count", 0))
        summary["reject_count"] = int(state.get("reject_count", 0))
        summary["rejection_streak"] = 0
        summary["anchor_loss"] = state.get("anchor_loss")
        summary["late_step_scale"] = float(state.get("late_step_scale"))
        summary["applied_lr_scale"] = current_late_step_scale
        summary["anchor_available"] = True
        summary["metric_snapshot"] = dict(candidate_metric_snapshot)
        return summary

    state["reject_count"] = int(state.get("reject_count", 0)) + 1
    state["rejection_streak"] = int(state.get("rejection_streak", 0)) + 1
    state["late_step_scale"] = current_late_step_scale * float(
        state.get("reject_scale", config["reject_scale"])
    )
    summary["step_decision"] = "rejected"
    summary["rolled_back"] = True
    summary["accept_count"] = int(state.get("accept_count", 0))
    summary["reject_count"] = int(state.get("reject_count", 0))
    summary["rejection_streak"] = int(state.get("rejection_streak", 0))
    summary["anchor_loss"] = state.get("anchor_loss")
    summary["late_step_scale"] = float(state.get("late_step_scale"))
    summary["applied_lr_scale"] = 0.0
    summary["anchor_available"] = True
    summary["metric_snapshot"] = state.get("anchor_metric_snapshot")
    return summary


def apply_late_commit_feedback_size_dynamics(
    context,
    state,
    before_logits,
    metric,
    phase,
    phase_lr_scale,
    config,
    candidate_metric_snapshot=None,
    baseline_metric_snapshot=None,
):
    summary = {
        "step_decision": "accepted",
        "rolled_back": False,
        "accept_count": 0,
        "reject_count": 0,
        "rejection_streak": 0,
        "anchor_loss": None,
        "committed_loss": state.get("committed_loss"),
        "previous_loss": None,
        "late_step_scale": float(state.get("late_step_scale", config["late_lr_scale"])),
        "accept_min_delta": float(config["accept_min_delta"]),
        "effective_accept_min_delta": float(config["accept_min_delta"]),
        "reject_scale": float(config["reject_scale"]),
        "recover_scale": float(config["recover_scale"]),
        "applied_lr_scale": float(phase_lr_scale),
        "anchor_available": state.get("committed_loss") is not None,
        "metric_snapshot": None,
    }
    if not isinstance(state, dict) or state.get("mode") != "late_commit_feedback":
        return summary

    candidate_metric_snapshot = candidate_metric_snapshot or context.metric_snapshot(
        metric
    )
    current_loss = candidate_metric_snapshot.get("combined_timing_loss")
    current_late_step_scale = float(state.get("late_step_scale", config["late_lr_scale"]))
    summary["late_step_scale"] = current_late_step_scale
    base_accept_min_delta = float(state.get("accept_min_delta", config["accept_min_delta"]))
    base_late_step_scale = float(state.get("base_late_step_scale", config["late_lr_scale"]))
    effective_accept_min_delta = base_accept_min_delta
    if base_late_step_scale > 0.0:
        effective_accept_min_delta = base_accept_min_delta * (
            current_late_step_scale / base_late_step_scale
        )
    summary["effective_accept_min_delta"] = max(0.0, float(effective_accept_min_delta))
    current_logits = getattr(getattr(context, "data_collections", None), "size_logits", None)
    committed_logits = before_logits.detach().clone()
    if current_logits is not None:
        committed_logits = before_logits.detach() + (
            current_logits.detach() - before_logits.detach()
        ) * current_late_step_scale

    if phase != "late":
        summary["accept_count"] = int(state.get("accept_count", 0))
        summary["reject_count"] = int(state.get("reject_count", 0))
        summary["rejection_streak"] = int(state.get("rejection_streak", 0))
        summary["committed_loss"] = state.get("committed_loss")
        summary["metric_snapshot"] = candidate_metric_snapshot
        return summary

    previous_loss = state.get("committed_loss")
    if previous_loss is None and isinstance(baseline_metric_snapshot, dict):
        previous_loss = baseline_metric_snapshot.get("combined_timing_loss")
    summary["previous_loss"] = previous_loss
    state["previous_loss"] = previous_loss

    improved = previous_loss is None or current_loss is None
    if previous_loss is not None and current_loss is not None:
        improved = float(current_loss) <= float(previous_loss) - summary["effective_accept_min_delta"]

    if improved:
        state["accept_count"] = int(state.get("accept_count", 0)) + 1
        state["rejection_streak"] = 0
        recover_scale = float(state.get("recover_scale", config["recover_scale"]))
        state["late_step_scale"] = min(
            base_late_step_scale,
            current_late_step_scale * recover_scale,
        )
        summary["step_decision"] = "accepted"
    else:
        state["reject_count"] = int(state.get("reject_count", 0)) + 1
        state["rejection_streak"] = int(state.get("rejection_streak", 0)) + 1
        state["late_step_scale"] = current_late_step_scale * float(
            state.get("reject_scale", config["reject_scale"])
        )
        summary["step_decision"] = "contracted"

    state["committed_logits"] = committed_logits.detach().clone()
    state["committed_loss"] = None if current_loss is None else float(current_loss)
    state["committed_metric_snapshot"] = dict(candidate_metric_snapshot)
    summary["accept_count"] = int(state.get("accept_count", 0))
    summary["reject_count"] = int(state.get("reject_count", 0))
    summary["rejection_streak"] = int(state.get("rejection_streak", 0))
    summary["committed_loss"] = state.get("committed_loss")
    summary["late_step_scale"] = float(state.get("late_step_scale"))
    summary["applied_lr_scale"] = current_late_step_scale
    summary["anchor_available"] = state.get("committed_loss") is not None
    summary["metric_snapshot"] = dict(candidate_metric_snapshot)
    return summary


def apply_continuous_size_logits_dynamics(
    context,
    params,
    before_snapshot,
    iteration,
    total_iterations,
    runtime_state=None,
    metric=None,
    model=None,
    pos=None,
):
    config = context.config(params)
    summary = {
        "enabled": False,
        "mode": config["mode"],
        "phase": "disabled",
        "phase_progress": None,
        "phase_lr_scale": 1.0,
        "raw_delta_max_abs": 0.0,
        "trust_ratio": 1.0,
        "applied_delta_max_abs": 0.0,
        "guard_patience": config["trigger_patience"],
        "guard_min_delta": config["trigger_min_delta"],
        "guard_monitoring_ready": False,
        "guard_monitoring_started": False,
        "guard_no_improve_rounds": 0,
        "guard_triggered": False,
        "guard_trigger_iteration": None,
        "guard_trigger_reason": None,
        "guard_best_loss": None,
        "guard_best_iteration": None,
        "guard_last_loss": None,
        "guard_last_iteration": None,
        "guard_zero_vio_ready": None,
        "guard_zero_vio_tol": config["zero_vio_tol"],
        "guard_last_slew_violation": None,
        "guard_last_cap_violation": None,
        "accept_reject_step_decision": "disabled",
        "accept_reject_rolled_back": False,
        "accept_reject_accept_count": 0,
        "accept_reject_reject_count": 0,
        "accept_reject_rejection_streak": 0,
        "accept_reject_anchor_loss": None,
        "accept_reject_committed_loss": None,
        "accept_reject_previous_loss": None,
        "accept_reject_late_step_scale": float(config["late_lr_scale"]),
        "accept_reject_accept_min_delta": float(config["accept_min_delta"]),
        "accept_reject_effective_accept_min_delta": float(config["accept_min_delta"]),
        "accept_reject_reject_scale": float(config["reject_scale"]),
        "accept_reject_recover_scale": float(config["recover_scale"]),
    }
    size_logits = getattr(getattr(context, "data_collections", None), "size_logits", None)
    before_logits = None if before_snapshot is None else before_snapshot.get("size_logits")
    supported_modes = {
        "phased_trust",
        "late_adaptive_guard",
        "late_zero_vio_guard",
        "late_accept_reject",
        "late_commit_feedback",
        "discrete_gradient_topk",
    }
    if config["mode"] not in supported_modes or size_logits is None or before_logits is None:
        context.summary = summary
        return summary

    phase, phase_lr_scale, progress = context.phase(
        iteration,
        total_iterations,
        config,
    )
    raw_delta = size_logits.detach() - before_logits.detach()
    raw_delta_max_abs = 0.0 if raw_delta.numel() == 0 else float(raw_delta.abs().max().item())
    if config["mode"] == "discrete_gradient_topk":
        return context.discrete_step(params, before_snapshot, iteration, total_iterations, config, summary, size_logits, before_logits, phase, progress, raw_delta_max_abs)
    guard_summary = context.update_guard(
        runtime_state,
        metric,
        iteration,
        phase,
    )
    accept_reject_summary = {
        "step_decision": "disabled",
        "rolled_back": False,
        "accept_count": 0,
        "reject_count": 0,
        "rejection_streak": 0,
        "anchor_loss": None,
        "committed_loss": None,
        "previous_loss": None,
        "late_step_scale": float(config["late_lr_scale"]),
        "accept_min_delta": float(config["accept_min_delta"]),
        "effective_accept_min_delta": float(config["accept_min_delta"]),
        "reject_scale": float(config["reject_scale"]),
        "recover_scale": float(config["recover_scale"]),
        "applied_lr_scale": float(phase_lr_scale),
    }
    if config["mode"] in ("late_adaptive_guard", "late_zero_vio_guard"):
        phase_lr_scale = (
            float(config["late_lr_scale"])
            if phase == "late" and guard_summary.get("triggered")
            else 1.0
        )
    elif config["mode"] in ("late_accept_reject", "late_commit_feedback"):
        candidate_metric_snapshot = None
        if phase == "late":
            current_late_step_scale = float(
                runtime_state.get("late_step_scale", config["late_lr_scale"])
            )
            candidate_logits = before_logits.detach() + raw_delta * current_late_step_scale
            with torch.no_grad():
                proposal_logits = size_logits.detach().clone()
                size_logits.copy_(candidate_logits)
            try:
                candidate_metric_snapshot = (
                    context.evaluate_candidate(
                        model, pos, metric
                    )
                )
            finally:
                with torch.no_grad():
                    size_logits.copy_(proposal_logits)
        baseline_metric_snapshot = None
        if config["mode"] == "late_commit_feedback" and phase == "late":
            if runtime_state.get("committed_loss") is None:
                with torch.no_grad():
                    proposal_logits = size_logits.detach().clone()
                    size_logits.copy_(before_logits.detach())
                try:
                    baseline_metric_snapshot = (
                        context.evaluate_candidate(
                            model, pos, metric
                        )
                    )
                finally:
                    with torch.no_grad():
                        size_logits.copy_(proposal_logits)
        if config["mode"] == "late_accept_reject":
            accept_reject_summary = context.accept_reject(
                runtime_state,
                before_logits,
                metric,
                phase,
                phase_lr_scale,
                config,
                candidate_metric_snapshot=candidate_metric_snapshot,
            )
        else:
            accept_reject_summary = context.commit_feedback(
                runtime_state,
                before_logits,
                metric,
                phase,
                phase_lr_scale,
                config,
                candidate_metric_snapshot=candidate_metric_snapshot,
                baseline_metric_snapshot=baseline_metric_snapshot,
            )
        phase_lr_scale = float(accept_reject_summary.get("applied_lr_scale", phase_lr_scale))
        context.apply_metric(metric, accept_reject_summary.get("metric_snapshot"))
    applied_delta = raw_delta * float(phase_lr_scale)
    applied_delta_max_abs = 0.0 if applied_delta.numel() == 0 else float(applied_delta.abs().max().item())
    trust_ratio = 1.0
    trust_radius = config["trust_radius"]
    trust_clip_enabled = (
        phase == "late"
        and trust_radius > 0.0
        and (
            config["mode"] == "phased_trust"
            or bool(guard_summary.get("triggered"))
        )
    )
    if trust_clip_enabled and applied_delta_max_abs > trust_radius:
        trust_ratio = trust_radius / applied_delta_max_abs
        applied_delta = applied_delta * trust_ratio
        applied_delta_max_abs = trust_radius
    with torch.no_grad():
        size_logits.copy_(before_logits.detach() + applied_delta)
    summary.update(
        {
            "enabled": True,
            "phase": phase,
            "phase_progress": progress,
            "phase_lr_scale": float(phase_lr_scale),
            "raw_delta_max_abs": raw_delta_max_abs,
            "trust_ratio": float(trust_ratio),
            "applied_delta_max_abs": float(applied_delta_max_abs),
            "guard_monitoring_ready": guard_summary.get("monitoring_ready"),
            "guard_monitoring_started": guard_summary.get("monitoring_started"),
            "guard_no_improve_rounds": guard_summary.get("no_improve_rounds"),
            "guard_triggered": guard_summary.get("triggered"),
            "guard_trigger_iteration": guard_summary.get("trigger_iteration"),
            "guard_trigger_reason": guard_summary.get("trigger_reason"),
            "guard_best_loss": guard_summary.get("best_loss"),
            "guard_best_iteration": guard_summary.get("best_iteration"),
            "guard_last_loss": guard_summary.get("last_loss"),
            "guard_last_iteration": guard_summary.get("last_iteration"),
            "guard_zero_vio_ready": guard_summary.get("zero_vio_ready"),
            "guard_last_slew_violation": guard_summary.get("last_slew_violation"),
            "guard_last_cap_violation": guard_summary.get("last_cap_violation"),
            "accept_reject_step_decision": accept_reject_summary.get("step_decision"),
            "accept_reject_rolled_back": accept_reject_summary.get("rolled_back"),
            "accept_reject_accept_count": accept_reject_summary.get("accept_count"),
            "accept_reject_reject_count": accept_reject_summary.get("reject_count"),
            "accept_reject_rejection_streak": accept_reject_summary.get("rejection_streak"),
            "accept_reject_anchor_loss": accept_reject_summary.get("anchor_loss"),
            "accept_reject_committed_loss": accept_reject_summary.get("committed_loss"),
            "accept_reject_previous_loss": accept_reject_summary.get("previous_loss"),
            "accept_reject_late_step_scale": accept_reject_summary.get("late_step_scale"),
            "accept_reject_accept_min_delta": accept_reject_summary.get("accept_min_delta"),
            "accept_reject_effective_accept_min_delta": accept_reject_summary.get("effective_accept_min_delta"),
            "accept_reject_reject_scale": accept_reject_summary.get("reject_scale"),
            "accept_reject_recover_scale": accept_reject_summary.get("recover_scale"),
        }
    )
    if config["mode"] == "late_accept_reject" and accept_reject_summary.get("rolled_back"):
        anchor_logits = None if not isinstance(runtime_state, dict) else runtime_state.get("anchor_logits")
        with torch.no_grad():
            if anchor_logits is not None:
                size_logits.copy_(anchor_logits.detach())
            else:
                size_logits.copy_(before_logits.detach())
        summary["phase_lr_scale"] = 0.0
        summary["trust_ratio"] = 1.0
        summary["applied_delta_max_abs"] = 0.0
    context.summary = summary
    return summary


def evaluate_continuous_size_dynamics_candidate_metric(context, model, pos, metric):
    if model is None or pos is None or not hasattr(model, "obj_fn"):
        return context.metric_snapshot(metric)

    with torch.no_grad():
        objective = model.obj_fn(pos)
    timing_op = getattr(context.op_collections, "timing_propagation_op", None)
    snapshot = {
        "objective": objective.detach().clone() if torch.is_tensor(objective) else objective,
        "wns": context.timing_scalar(getattr(model, "wns", None)),
        "tns": context.timing_scalar(getattr(model, "tns", None)),
        "timing_objective": context.timing_scalar(getattr(model, "ws", None)),
        "slew_violation": None
        if timing_op is None
        else context.timing_scalar(getattr(timing_op, "last_total_slew_violation", None)),
        "cap_violation": None
        if timing_op is None
        else context.timing_scalar(getattr(timing_op, "last_total_cap_violation", None)),
        "leakage": None
        if timing_op is None
        else context.timing_scalar(getattr(timing_op, "last_total_leakage", None)),
    }
    snapshot["combined_timing_loss"] = context.combined_loss(
        snapshot["wns"], snapshot["tns"]
    )
    if snapshot["wns"] is not None and snapshot["tns"] is not None:
        context.record_timing(
            snapshot["wns"],
            snapshot["tns"],
            snapshot["timing_objective"],
            context.timing_scalar(getattr(model, "ts", None)),
        )
    return snapshot

