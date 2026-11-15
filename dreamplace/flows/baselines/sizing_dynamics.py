"""ECC sizing dynamics owner; live state is supplied per call."""


from dataclasses import dataclass
from typing import Callable
import torch
import math
from dreamplace.flows.gp.metrics import timing_scalar, combined_timing_loss
from dreamplace.ops.buffer_insertion.optimization_state import (
    capture_buffer_optimization_snapshot, buffer_optimization_grad_stats,
)
import numpy as np
import dreamplace.BasicPlace as BasicPlace
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState
from dreamplace.ops.buffer_insertion.optimization_state import restore_buffer_optimization_snapshot


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


def continuous_size_dynamics_config(params):
    mode = str(getattr(params, "continuous_size_dynamics_mode", "none") or "none")
    early_frac = float(getattr(params, "continuous_size_dynamics_early_frac", 0.3))
    late_frac = float(getattr(params, "continuous_size_dynamics_late_frac", 0.8))
    mid_lr_scale = float(getattr(params, "continuous_size_dynamics_mid_lr_scale", 0.5))
    late_lr_scale = float(getattr(params, "continuous_size_dynamics_late_lr_scale", 0.25))
    trust_radius = float(getattr(params, "continuous_size_dynamics_trust_radius", 0.0))
    trigger_patience = int(getattr(params, "continuous_size_dynamics_trigger_patience", 0) or 0)
    trigger_min_delta = float(
        getattr(params, "continuous_size_dynamics_trigger_min_delta", 0.0) or 0.0
    )
    zero_vio_tol = float(getattr(params, "continuous_size_dynamics_zero_vio_tol", 0.0) or 0.0)
    accept_min_delta = float(
        getattr(params, "continuous_size_dynamics_accept_min_delta", 0.0) or 0.0
    )
    reject_scale = float(
        getattr(params, "continuous_size_dynamics_reject_scale", 0.5) or 0.0
    )
    recover_scale = float(
        getattr(params, "continuous_size_dynamics_recover_scale", 1.0) or 1.0
    )
    discrete_gradient_topk_up_percent = float(
        getattr(params, "discrete_gradient_topk_up_percent", 30.0)
    )
    discrete_gradient_topk_down_percent = float(
        getattr(params, "discrete_gradient_topk_down_percent", 0.0)
    )
    discrete_gradient_topk_vt_percent = float(
        getattr(params, "discrete_gradient_topk_vt_percent", 0.0)
    )
    shared_budget = bool(getattr(params, "discrete_gradient_topk_shared_budget", False))
    oscillation_veto = bool(getattr(params, "discrete_gradient_topk_oscillation_veto", False))
    if oscillation_veto and not shared_budget:
        raise ValueError("oscillation veto requires the shared cell budget")
    shared_budget_percent = float(
        getattr(params, "discrete_gradient_topk_shared_budget_percent", 1.0)
    )
    if shared_budget and (
        not math.isfinite(shared_budget_percent)
        or not 0.0 <= shared_budget_percent <= 100.0
    ):
        raise ValueError("shared cell budget percentage must be in [0, 100]")
    discrete_gradient_topk_preserve_vt = bool(
        getattr(params, "discrete_gradient_topk_preserve_vt", True)
    )
    discrete_gradient_topk_ranking_mode = str(
        getattr(params, "discrete_gradient_topk_ranking_mode", "real_size_taylor")
        or "real_size_taylor"
    )
    discrete_gradient_topk_step_mode = str(
        getattr(params, "discrete_gradient_topk_step_mode", "direct_one_step")
        or "direct_one_step"
    )
    discrete_gradient_topk_lambda_update_mode = str(
        getattr(params, "discrete_gradient_topk_lambda_update_mode", "fixed_gradient_price")
        or "fixed_gradient_price"
    )
    discrete_gradient_topk_lambda_update_scale = float(
        getattr(params, "discrete_gradient_topk_lambda_update_scale", 1.0) or 1.0
    )
    discrete_gradient_topk_lambda_min = float(
        getattr(params, "discrete_gradient_topk_lambda_min", 0.0) or 0.0
    )
    discrete_gradient_topk_lambda_max = float(
        getattr(params, "discrete_gradient_topk_lambda_max", 1.0e6) or 1.0e6
    )
    precise_improvement_search_mode = str(
        getattr(params, "precise_improvement_search_mode", "multistep")
        or "multistep"
    )
    precise_improvement_max_up_step = int(
        getattr(params, "precise_improvement_max_up_step", 3) or 3
    )
    precise_improvement_max_down_step = int(
        getattr(params, "precise_improvement_max_down_step", 0) or 0
    )
    precise_improvement_parallel_workers = int(
        getattr(params, "precise_improvement_parallel_workers", 1) or 1
    )
    timing_objective_lane = str(
        getattr(params, "timing_objective_lane", "timing_slew_cap")
        or "timing_slew_cap"
    )
    timing_slew_weight = float(getattr(params, "timing_slew_weight", 1.0))
    timing_cap_weight = float(getattr(params, "timing_cap_weight", 1.0))
    early_frac = min(max(early_frac, 0.0), 1.0)
    late_frac = min(max(late_frac, early_frac), 1.0)
    return {
        "mode": mode,
        "timing_objective_lane": timing_objective_lane,
        "timing_slew_weight": timing_slew_weight,
        "timing_cap_weight": timing_cap_weight,
        "early_frac": early_frac,
        "late_frac": late_frac,
        "mid_lr_scale": max(mid_lr_scale, 0.0),
        "late_lr_scale": max(late_lr_scale, 0.0),
        "trust_radius": max(trust_radius, 0.0),
        "trigger_patience": max(trigger_patience, 0),
        "trigger_min_delta": max(trigger_min_delta, 0.0),
        "zero_vio_tol": max(zero_vio_tol, 0.0),
        "accept_min_delta": max(accept_min_delta, 0.0),
        "reject_scale": min(max(reject_scale, 0.0), 1.0),
        "recover_scale": max(recover_scale, 1.0),
        "discrete_gradient_topk_up_percent": discrete_gradient_topk_up_percent,
        "discrete_gradient_topk_down_percent": discrete_gradient_topk_down_percent,
        "discrete_gradient_topk_vt_percent": discrete_gradient_topk_vt_percent,
        "discrete_gradient_topk_shared_budget": shared_budget,
        "discrete_gradient_topk_shared_budget_percent": shared_budget_percent,
        "discrete_gradient_topk_oscillation_veto": oscillation_veto,
        "discrete_gradient_topk_preserve_vt": discrete_gradient_topk_preserve_vt,
        "discrete_gradient_topk_ranking_mode": discrete_gradient_topk_ranking_mode,
        "discrete_gradient_topk_step_mode": discrete_gradient_topk_step_mode,
        "discrete_gradient_topk_lambda_update_mode": discrete_gradient_topk_lambda_update_mode,
        "discrete_gradient_topk_lambda_update_scale": discrete_gradient_topk_lambda_update_scale,
        "discrete_gradient_topk_lambda_min": discrete_gradient_topk_lambda_min,
        "discrete_gradient_topk_lambda_max": discrete_gradient_topk_lambda_max,
        "precise_improvement_search_mode": precise_improvement_search_mode,
        "precise_improvement_max_up_step": max(precise_improvement_max_up_step, 0),
        "precise_improvement_max_down_step": max(precise_improvement_max_down_step, 0),
        "precise_improvement_parallel_workers": max(precise_improvement_parallel_workers, 1),
    }


def continuous_size_dynamics_phase(iteration, total_iterations, config):
    denom = max(int(total_iterations) - 1, 1)
    progress = float(iteration) / float(denom)
    if progress < config["early_frac"]:
        return "early", 1.0, progress
    if progress < config["late_frac"]:
        return "mid", config["mid_lr_scale"], progress
    return "late", config["late_lr_scale"], progress


@dataclass(frozen=True)
class SizingDebugContext:
    data_collections: object
    parameter: Callable
    parameterization: Callable
    enabled: Callable
    append_trace: Callable
    dynamics_summary: object

def capture_optional_tensor(tensor):
    if tensor is None:
        return None
    return tensor.detach().clone()


def tensor_delta_stats(before, after):
    if before is None or after is None:
        return None, None
    delta = after.detach() - before.detach()
    if delta.numel() == 0:
        return 0.0, 0.0
    return float(delta.norm().item()), float(delta.abs().max().item())


def size_debug_learning_rates(optimizer):
    placement_lr = None
    sizing_lr = None
    for param_group in optimizer.param_groups:
        group_name = param_group.get("group_name")
        lr = param_group.get("lr")
        lr_value = None if lr is None else float(lr)
        if group_name == "placement":
            placement_lr = lr_value
        elif group_name == "sizing":
            sizing_lr = lr_value
    return placement_lr, sizing_lr


def size_debug_objective_terms(model):
    density_term = timing_scalar(getattr(model, "density", None))
    density_weight = timing_scalar(getattr(model, "density_weight", None))
    density_factor = getattr(model, "density_factor", None)
    density_objective_term = None
    if density_term is not None and density_weight is not None and density_factor is not None:
        density_objective_term = float(density_term * float(density_factor) * density_weight)
    wns = timing_scalar(getattr(model, "wns", None))
    tns = timing_scalar(getattr(model, "tns", None))
    return {
        "wirelength": timing_scalar(getattr(model, "wirelength", None)),
        "density": density_term,
        "density_weight": density_weight,
        "density_objective_term": density_objective_term,
        "size_density_area_penalty": timing_scalar(getattr(model, "size_density_area_penalty", None)),
        "wns": wns,
        "tns": tns,
        "combined_timing_loss": combined_timing_loss(wns, tns),
    }


def capture_size_debug_snapshot(context):
    size_var = context.data_collections.get_size_var()
    snapshot = {
        "size_logits": capture_optional_tensor(getattr(context.data_collections, "size_logits", None)),
        "real_size": capture_optional_tensor(getattr(context.data_collections, "real_size", None)),
        "continuous_size": capture_optional_tensor(size_var),
        "continuous_size_parameter": capture_optional_tensor(
            context.parameter()
        ),
        "sizing_parameterization": context.parameterization(),
        "effective_initial_size": capture_optional_tensor(
            getattr(context.data_collections, "effective_initial_size", None)
        ),
        "vt_logits": capture_optional_tensor(getattr(context.data_collections, "vt_logits", None)),
        "size_var": capture_optional_tensor(size_var),
    }
    snapshot.update(
        capture_buffer_optimization_snapshot(
            getattr(context.data_collections, "buffer_optimization_state", None)
        )
    )
    return snapshot


def size_debug_grad_stats(context):
    def grad_stats(tensor):
        if tensor is None or tensor.grad is None:
            return None, None
        grad = tensor.grad.detach()
        if grad.numel() == 0:
            return 0.0, 0.0
        return float(grad.norm().item()), float(grad.abs().max().item())

    continuous_size = context.parameter()
    stats = {
        "size_grad_norm": grad_stats(continuous_size)[0],
        "size_grad_max_abs": grad_stats(continuous_size)[1],
        "sizing_parameterization": context.parameterization(),
        "vt_grad_norm": grad_stats(getattr(context.data_collections, "vt_logits", None))[0],
        "vt_grad_max_abs": grad_stats(getattr(context.data_collections, "vt_logits", None))[1],
    }
    stats.update(
        buffer_optimization_grad_stats(
            getattr(context.data_collections, "buffer_optimization_state", None)
        )
    )
    return stats


def maybe_write_size_debug_trace(
    context,
    params,
    iteration,
    metric,
    model,
    optimizer,
    before_snapshot,
    after_snapshot,
    grad_stats,
):
    if not context.enabled():
        return
    placement_lr, sizing_lr = size_debug_learning_rates(optimizer)
    size_logits_delta_norm, size_logits_delta_max_abs = tensor_delta_stats(
        before_snapshot.get("size_logits"),
        after_snapshot.get("size_logits"),
    )
    vt_logits_delta_norm, vt_logits_delta_max_abs = tensor_delta_stats(
        before_snapshot.get("vt_logits"),
        after_snapshot.get("vt_logits"),
    )
    size_var_delta_norm, size_var_delta_max_abs = tensor_delta_stats(
        before_snapshot.get("size_var"),
        after_snapshot.get("size_var"),
    )
    continuous_size_delta_norm, continuous_size_delta_max_abs = tensor_delta_stats(
        before_snapshot.get("continuous_size"),
        after_snapshot.get("continuous_size"),
    )
    real_size_delta_norm, real_size_delta_max_abs = tensor_delta_stats(
        before_snapshot.get("real_size"),
        after_snapshot.get("real_size"),
    )
    size_before = before_snapshot.get("size_var")
    size_after = after_snapshot.get("size_var")
    record = {
        "iteration": int(iteration),
        "objective": timing_scalar(metric.objective),
        "overflow": timing_scalar(metric.overflow),
        "size_mean_before": None if size_before is None or size_before.numel() == 0 else float(size_before.mean().item()),
        "size_mean_after": None if size_after is None or size_after.numel() == 0 else float(size_after.mean().item()),
        "placement_learning_rate": placement_lr,
        "sizing_learning_rate": sizing_lr,
        "size_logits_delta_norm": size_logits_delta_norm,
        "size_logits_delta_max_abs": size_logits_delta_max_abs,
        "vt_logits_delta_norm": vt_logits_delta_norm,
        "vt_logits_delta_max_abs": vt_logits_delta_max_abs,
        "size_var_delta_norm": size_var_delta_norm,
        "size_var_delta_max_abs": size_var_delta_max_abs,
        "continuous_size_delta_norm": continuous_size_delta_norm,
        "continuous_size_delta_max_abs": continuous_size_delta_max_abs,
        "real_size_delta_norm": real_size_delta_norm,
        "real_size_delta_max_abs": real_size_delta_max_abs,
        "sizing_parameterization": after_snapshot.get(
            "sizing_parameterization",
            context.parameterization(),
        ),
    }
    record.update(grad_stats)
    record.update(size_debug_objective_terms(model))
    dynamics_summary = context.dynamics_summary or {}
    record.update(
        {
            "continuous_size_dynamics_enabled": dynamics_summary.get("enabled"),
            "continuous_size_dynamics_mode": dynamics_summary.get("mode"),
            "continuous_size_dynamics_phase": dynamics_summary.get("phase"),
            "continuous_size_dynamics_phase_progress": dynamics_summary.get("phase_progress"),
            "continuous_size_dynamics_phase_lr_scale": dynamics_summary.get("phase_lr_scale"),
            "continuous_size_dynamics_raw_delta_max_abs": dynamics_summary.get("raw_delta_max_abs"),
            "continuous_size_dynamics_trust_ratio": dynamics_summary.get("trust_ratio"),
            "continuous_size_dynamics_applied_delta_max_abs": dynamics_summary.get("applied_delta_max_abs"),
            "continuous_size_dynamics_guard_patience": dynamics_summary.get("guard_patience"),
            "continuous_size_dynamics_guard_min_delta": dynamics_summary.get("guard_min_delta"),
            "continuous_size_dynamics_guard_monitoring_ready": dynamics_summary.get("guard_monitoring_ready"),
            "continuous_size_dynamics_guard_monitoring_started": dynamics_summary.get("guard_monitoring_started"),
            "continuous_size_dynamics_guard_no_improve_rounds": dynamics_summary.get("guard_no_improve_rounds"),
            "continuous_size_dynamics_guard_triggered": dynamics_summary.get("guard_triggered"),
            "continuous_size_dynamics_guard_trigger_iteration": dynamics_summary.get("guard_trigger_iteration"),
            "continuous_size_dynamics_guard_trigger_reason": dynamics_summary.get("guard_trigger_reason"),
            "continuous_size_dynamics_guard_best_loss": dynamics_summary.get("guard_best_loss"),
            "continuous_size_dynamics_guard_best_iteration": dynamics_summary.get("guard_best_iteration"),
            "continuous_size_dynamics_guard_last_loss": dynamics_summary.get("guard_last_loss"),
            "continuous_size_dynamics_guard_last_iteration": dynamics_summary.get("guard_last_iteration"),
            "continuous_size_dynamics_guard_zero_vio_ready": dynamics_summary.get("guard_zero_vio_ready"),
            "continuous_size_dynamics_guard_zero_vio_tol": dynamics_summary.get("guard_zero_vio_tol"),
            "continuous_size_dynamics_guard_last_slew_violation": dynamics_summary.get("guard_last_slew_violation"),
            "continuous_size_dynamics_guard_last_cap_violation": dynamics_summary.get("guard_last_cap_violation"),
            "continuous_size_dynamics_accept_count": dynamics_summary.get("accept_reject_accept_count"),
            "continuous_size_dynamics_reject_count": dynamics_summary.get("accept_reject_reject_count"),
            "continuous_size_dynamics_rejection_streak": dynamics_summary.get("accept_reject_rejection_streak"),
            "continuous_size_dynamics_accept_min_delta": dynamics_summary.get("accept_reject_accept_min_delta"),
            "continuous_size_dynamics_effective_accept_min_delta": dynamics_summary.get("accept_reject_effective_accept_min_delta"),
            "continuous_size_dynamics_reject_scale": dynamics_summary.get("accept_reject_reject_scale"),
            "continuous_size_dynamics_recover_scale": dynamics_summary.get("accept_reject_recover_scale"),
            "continuous_size_dynamics_anchor_loss": dynamics_summary.get("accept_reject_anchor_loss"),
            "continuous_size_dynamics_committed_loss": dynamics_summary.get("accept_reject_committed_loss"),
            "continuous_size_dynamics_previous_loss": dynamics_summary.get("accept_reject_previous_loss"),
            "continuous_size_dynamics_step_decision": dynamics_summary.get("accept_reject_step_decision"),
            "continuous_size_dynamics_rolled_back": dynamics_summary.get("accept_reject_rolled_back"),
            "continuous_size_dynamics_late_step_scale": dynamics_summary.get("accept_reject_late_step_scale"),
            "discrete_gradient_topk_enabled": dynamics_summary.get("discrete_gradient_topk_enabled"),
            "discrete_gradient_topk_up_percent": dynamics_summary.get("discrete_gradient_topk_up_percent"),
            "discrete_gradient_topk_down_percent": dynamics_summary.get("discrete_gradient_topk_down_percent"),
            "discrete_gradient_topk_ranking_mode": dynamics_summary.get("discrete_gradient_topk_ranking_mode"),
            "discrete_gradient_topk_selection_backend": dynamics_summary.get("discrete_gradient_topk_selection_backend"),
            "discrete_gradient_topk_num_up_selected": dynamics_summary.get("discrete_gradient_topk_num_up_selected"),
            "discrete_gradient_topk_num_down_selected": dynamics_summary.get("discrete_gradient_topk_num_down_selected"),
            "discrete_gradient_topk_num_up_applied": dynamics_summary.get("discrete_gradient_topk_num_up_applied"),
            "discrete_gradient_topk_num_down_applied": dynamics_summary.get("discrete_gradient_topk_num_down_applied"),
            "discrete_gradient_topk_num_changed_instances": dynamics_summary.get("discrete_gradient_topk_num_changed_instances"),
            "discrete_gradient_topk_num_candidate_moves": dynamics_summary.get("discrete_gradient_topk_num_candidate_moves"),
            "discrete_gradient_topk_num_improving_candidates": dynamics_summary.get("discrete_gradient_topk_num_improving_candidates"),
            "discrete_gradient_topk_num_non_improving_selected": dynamics_summary.get("discrete_gradient_topk_num_non_improving_selected"),
            "discrete_gradient_topk_predicted_delta_obj_min": dynamics_summary.get("discrete_gradient_topk_predicted_delta_obj_min"),
            "discrete_gradient_topk_predicted_delta_obj_mean": dynamics_summary.get("discrete_gradient_topk_predicted_delta_obj_mean"),
            "discrete_gradient_topk_predicted_delta_obj_max": dynamics_summary.get("discrete_gradient_topk_predicted_delta_obj_max"),
            "discrete_gradient_topk_predicted_improvement_threshold": dynamics_summary.get("discrete_gradient_topk_predicted_improvement_threshold"),
            "discrete_gradient_topk_delta_logit_mean_abs": dynamics_summary.get("discrete_gradient_topk_delta_logit_mean_abs"),
            "discrete_gradient_topk_delta_logit_max_abs": dynamics_summary.get("discrete_gradient_topk_delta_logit_max_abs"),
            "discrete_gradient_topk_score_coordinate": dynamics_summary.get("discrete_gradient_topk_score_coordinate"),
            "discrete_gradient_topk_delta_value_name": dynamics_summary.get("discrete_gradient_topk_delta_value_name"),
            "discrete_gradient_topk_direction_coordinate": dynamics_summary.get("discrete_gradient_topk_direction_coordinate"),
            "discrete_gradient_topk_real_size_grad_min": dynamics_summary.get("discrete_gradient_topk_real_size_grad_min"),
            "discrete_gradient_topk_real_size_grad_mean": dynamics_summary.get("discrete_gradient_topk_real_size_grad_mean"),
            "discrete_gradient_topk_real_size_grad_max": dynamics_summary.get("discrete_gradient_topk_real_size_grad_max"),
            "discrete_gradient_topk_delta_real_size_mean_abs": dynamics_summary.get("discrete_gradient_topk_delta_real_size_mean_abs"),
            "discrete_gradient_topk_delta_real_size_max_abs": dynamics_summary.get("discrete_gradient_topk_delta_real_size_max_abs"),
            "discrete_gradient_topk_leakage_coord_mode": dynamics_summary.get("discrete_gradient_topk_leakage_coord_mode"),
            "discrete_gradient_topk_leakage_eps": dynamics_summary.get("discrete_gradient_topk_leakage_eps"),
            "discrete_gradient_topk_delta_log_leakage_mean_abs": dynamics_summary.get("discrete_gradient_topk_delta_log_leakage_mean_abs"),
            "discrete_gradient_topk_delta_log_leakage_max_abs": dynamics_summary.get("discrete_gradient_topk_delta_log_leakage_max_abs"),
            "discrete_gradient_topk_num_invalid_leakage_candidates": dynamics_summary.get("discrete_gradient_topk_num_invalid_leakage_candidates"),
            "discrete_gradient_topk_num_equal_leakage_candidates": dynamics_summary.get("discrete_gradient_topk_num_equal_leakage_candidates"),
            "discrete_gradient_topk_selected_up_mean_delta_log_leakage": dynamics_summary.get("discrete_gradient_topk_selected_up_mean_delta_log_leakage"),
            "discrete_gradient_topk_selected_down_mean_delta_log_leakage": dynamics_summary.get("discrete_gradient_topk_selected_down_mean_delta_log_leakage"),
            "discrete_gradient_topk_size_mean_after": dynamics_summary.get("discrete_gradient_topk_size_mean_after"),
            "discrete_gradient_topk_preserve_vt": dynamics_summary.get("discrete_gradient_topk_preserve_vt"),
            "discrete_gradient_topk_step_mode": dynamics_summary.get("discrete_gradient_topk_step_mode"),
            "discrete_gradient_topk_lambda_update_mode": dynamics_summary.get("discrete_gradient_topk_lambda_update_mode"),
            "discrete_gradient_topk_lagrangian_relaxation_enabled": dynamics_summary.get("discrete_gradient_topk_lagrangian_relaxation_enabled"),
            "discrete_gradient_topk_mean_abs_legal_index_delta": dynamics_summary.get("discrete_gradient_topk_mean_abs_legal_index_delta"),
            "discrete_gradient_topk_max_abs_legal_index_delta": dynamics_summary.get("discrete_gradient_topk_max_abs_legal_index_delta"),
            "discrete_gradient_topk_lambda_source": dynamics_summary.get("discrete_gradient_topk_lambda_source"),
            "discrete_gradient_topk_lambda_update_scale": dynamics_summary.get("discrete_gradient_topk_lambda_update_scale"),
            "discrete_gradient_topk_lambda_min": dynamics_summary.get("discrete_gradient_topk_lambda_min"),
            "discrete_gradient_topk_lambda_mean": dynamics_summary.get("discrete_gradient_topk_lambda_mean"),
            "discrete_gradient_topk_lambda_max": dynamics_summary.get("discrete_gradient_topk_lambda_max"),
            "discrete_gradient_topk_num_lambda_updated": dynamics_summary.get("discrete_gradient_topk_num_lambda_updated"),
            "discrete_gradient_topk_candidate_cost_min": dynamics_summary.get("discrete_gradient_topk_candidate_cost_min"),
            "discrete_gradient_topk_candidate_cost_mean": dynamics_summary.get("discrete_gradient_topk_candidate_cost_mean"),
            "discrete_gradient_topk_candidate_cost_max": dynamics_summary.get("discrete_gradient_topk_candidate_cost_max"),
        }
    )
    component_grad_stats = getattr(model, "debug_component_grad_stats", {})
    timing_grad = component_grad_stats.get("timing_loss_size_grad", {})
    area_grad = component_grad_stats.get("size_density_area_penalty_grad", {})
    record.update(
        {
            "timing_loss_size_grad_norm": timing_grad.get("norm"),
            "timing_loss_size_grad_max_abs": timing_grad.get("max_abs"),
            "timing_loss_size_grad_mean": timing_grad.get("mean"),
            "timing_loss_size_grad_positive_frac": timing_grad.get("positive_frac"),
            "timing_loss_size_grad_negative_frac": timing_grad.get("negative_frac"),
            "size_density_area_penalty_grad_norm": area_grad.get("norm"),
            "size_density_area_penalty_grad_max_abs": area_grad.get("max_abs"),
            "size_density_area_penalty_grad_mean": area_grad.get("mean"),
            "size_density_area_penalty_grad_positive_frac": area_grad.get("positive_frac"),
            "size_density_area_penalty_grad_negative_frac": area_grad.get("negative_frac"),
        }
    )
    context.append_trace(params, record)


@dataclass
class EarlyStopContext:
    data_collections: object
    placedb: object
    params: object
    oscillation_state: object
    sizing_parameterization: Callable
    continuous_size_parameter: Callable
    discrete_actions_owned: bool
    sync_runtime_cells: Callable
    sizing_mode: Callable
    finite_scalar: Callable
    capture_best: Callable
    loss_and_source: Callable

def continuous_early_stop_loss_and_source(context, params, metric):
    if metric is None:
        return None, None
    combined_timing_loss = context.finite_scalar(
        getattr(metric, "combined_timing_loss", None)
    )
    if combined_timing_loss is not None:
        return combined_timing_loss, "combined_timing_loss"
    if (
        context.sizing_mode(params) == "size_only"
        and bool(getattr(params, "with_sta", False))
        and bool(getattr(params, "differentiable_timing_obj", False))
    ):
        return None, None
    objective = context.finite_scalar(getattr(metric, "objective", None))
    if objective is not None:
        return objective, "objective"
    return None, None


def continuous_early_stop_loss(context, params, metric):
    loss, _source = context.loss_and_source(params, metric)
    return loss


def build_continuous_early_stop_state(params):
    patience = getattr(params, "early_stop_patience", None)
    min_delta = getattr(params, "early_stop_min_delta", None)
    restore_best = bool(getattr(params, "early_stop_restore_best", False))
    patience_value = None if patience is None else int(patience)
    stop_enabled = patience_value is not None and patience_value > 0
    enabled = stop_enabled or restore_best
    return {
        "enabled": enabled,
        "patience": patience_value,
        "min_delta": 0.0 if min_delta is None else max(0.0, float(min_delta)),
        "restore_best": restore_best,
        "best_loss": None,
        "best_iteration": None,
        "best_state": None,
        "loss_source": None,
        "last_loss": None,
        "last_iteration": None,
        "last_loss_source": None,
        "no_improve_rounds": 0,
        "triggered": False,
        "trigger_iteration": None,
        "trigger_reason": None,
        "restored": False,
        "restore_iteration": None,
        "restore_reason": None,
    }


def capture_continuous_early_stop_best_state(context, snapshot=None):
    data_collections = getattr(context, "data_collections", None)
    placedb = getattr(context, "placedb", None)

    def capture_tensor(name):
        if isinstance(snapshot, dict) and snapshot.get(name) is not None:
            return snapshot[name].detach().clone()
        tensor = None if data_collections is None else getattr(data_collections, name, None)
        if tensor is None:
            return None
        return tensor.detach().clone()

    def capture_array(name):
        value = None if placedb is None else getattr(placedb, name, None)
        if value is None:
            return None
        return np.array(value, copy=True)

    best_state = {
        "size_logits": capture_tensor("size_logits"),
        "real_size": capture_tensor("real_size"),
        "continuous_size": capture_tensor("continuous_size"),
        "continuous_size_parameter": capture_tensor(
            "continuous_size_parameter"
        ),
        "effective_initial_size": capture_tensor("effective_initial_size"),
        "sizing_parameterization": (
            (
                snapshot.get("sizing_parameterization")
                or context.sizing_parameterization()
            )
            if isinstance(snapshot, dict)
            else context.sizing_parameterization()
        ),
        "vt_logits": capture_tensor("vt_logits"),
        "inst_cell_id": capture_tensor("inst_cell_id"),
        "inst_libcell_offset": capture_tensor("inst_libcell_offset"),
        "placedb_inst_cell_id": capture_array("inst_cell_id"),
        "placedb_inst_libcell_offset": capture_array("inst_libcell_offset"),
    }
    best_state.update(
        capture_buffer_optimization_snapshot(
            getattr(data_collections, "buffer_optimization_state", None)
        )
    )
    oscillation_state = getattr(context, "oscillation_state", None)
    best_state["quad_oscillation_state"] = (
        None if oscillation_state is None else oscillation_state.snapshot()
    )
    return best_state


def restore_continuous_early_stop_best_state(context, state, reason="continuous_best_state_restore"):
    summary = {
        "enabled": bool(isinstance(state, dict) and state.get("enabled")),
        "restore_best": bool(isinstance(state, dict) and state.get("restore_best")),
        "restored": False,
        "restore_reason": None,
        "best_iteration": None,
        "best_loss": None,
    }
    if not isinstance(state, dict) or not state.get("enabled") or not state.get("restore_best"):
        return summary
    best_state = state.get("best_state")
    if not isinstance(best_state, dict):
        summary["restore_reason"] = "best_state_unavailable"
        return summary

    data_collections = getattr(context, "data_collections", None)
    placedb = getattr(context, "placedb", None)

    def restore_tensor(name):
        source = best_state.get(name)
        target = None if data_collections is None else getattr(data_collections, name, None)
        if source is None or target is None:
            return
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    restored_cell_updates = None
    sizing_runtime_sync = None
    if context.discrete_actions_owned:
        sizing_restore_summary = {
            "restored": False,
            "restore_reason": "milestone_owned_discrete_sizing_state",
        }
        buffer_restore_summary = {
            "restored": False,
            "restore_reason": "route_b_owned_discrete_state",
            "restored_keys": [],
        }
    else:
        current_cells = getattr(data_collections, "inst_cell_id", None)
        saved_cells = best_state.get("inst_cell_id")
        if current_cells is not None and saved_cells is not None:
            saved_cells = saved_cells.to(device=current_cells.device)
            changed_ids = torch.nonzero(
                current_cells != saved_cells, as_tuple=False
            ).flatten()
            if changed_ids.numel():
                restored_cell_updates = {
                    "applied_instance_ids": changed_ids.cpu().tolist(),
                    "applied_cell_ids": saved_cells[changed_ids].cpu().tolist(),
                }
        saved_parameterization = str(
            best_state.get(
                "sizing_parameterization",
                context.sizing_parameterization(),
            )
        ).strip().lower()
        current_parameterization = context.sizing_parameterization()
        saved_continuous_size = best_state.get("continuous_size")
        if saved_parameterization == current_parameterization:
            source = best_state.get("continuous_size_parameter")
            target = context.continuous_size_parameter()
            if source is not None and target is not None:
                with torch.no_grad():
                    target.copy_(
                        source.to(device=target.device, dtype=target.dtype)
                    )
            else:
                restore_tensor("size_logits")
                restore_tensor("real_size")
        elif saved_continuous_size is None:
            raise RuntimeError(
                "best sizing snapshot lacks continuous size for parameterization conversion"
            )
        elif current_parameterization == "logits":
            target = context.continuous_size_parameter()
            converted = BasicPlace.size_var_to_logits(
                saved_continuous_size,
                data_collections.inst_size_lower,
                data_collections.inst_size_upper,
            )
            with torch.no_grad():
                target.copy_(converted.to(device=target.device, dtype=target.dtype))
        elif current_parameterization == "real_size":
            target = context.continuous_size_parameter()
            with torch.no_grad():
                target.copy_(
                    saved_continuous_size.to(
                        device=target.device,
                        dtype=target.dtype,
                    )
                )
        else:
            raise RuntimeError(
                "unsupported sizing parameterization in best-state restore"
            )
        restore_tensor("effective_initial_size")
        restore_tensor("vt_logits")
        restore_tensor("inst_cell_id")
        restore_tensor("inst_libcell_offset")
        sizing_restore_summary = {
            "restored": True,
            "restore_reason": reason,
        }
        buffer_restore_summary = restore_buffer_optimization_snapshot(
            getattr(data_collections, "buffer_optimization_state", None),
            best_state,
        )

    if placedb is not None and not context.discrete_actions_owned:
        for state_key, attr_name in (
            ("placedb_inst_cell_id", "inst_cell_id"),
            ("placedb_inst_libcell_offset", "inst_libcell_offset"),
        ):
            source = best_state.get(state_key)
            target = getattr(placedb, attr_name, None)
            if source is not None and target is not None:
                target[...] = source

    if restored_cell_updates is not None:
        # Restore the selected masters and their physical/timing state as
        # one transaction. Final projection can report no changed cells,
        # so it cannot repair stale last-iteration geometry for us.
        sizing_runtime_sync = context.sync_runtime_cells(
            getattr(context, "params", None), restored_cell_updates
        )

    if "quad_oscillation_state" in best_state:
        saved_oscillation = best_state["quad_oscillation_state"]
        if saved_oscillation is None:
            context.oscillation_state = None
        else:
            oscillation_state = getattr(context, "oscillation_state", None)
            if oscillation_state is None:
                oscillation_state = CellOscillationState()
            oscillation_state.restore(saved_oscillation)
            context.oscillation_state = oscillation_state

    state["restored"] = True
    state["restore_iteration"] = state.get("best_iteration")
    state["restore_reason"] = reason
    summary.update(
        {
            "restored": True,
            "restore_reason": reason,
            "best_iteration": state.get("best_iteration"),
            "best_loss": state.get("best_loss"),
            "sizing_restored": bool(sizing_restore_summary.get("restored")),
            "sizing_restore_reason": sizing_restore_summary.get(
                "restore_reason"
            ),
            "sizing_runtime_sync": sizing_runtime_sync,
            "buffer_restored": bool(buffer_restore_summary.get("restored")),
            "buffer_restore_reason": buffer_restore_summary.get("restore_reason"),
        }
    )
    return summary


def update_continuous_early_stop_state(
    context,
    state,
    combined_timing_loss,
    iteration,
    loss_source="combined_timing_loss",
):
    if not state.get("enabled"):
        return False
    if combined_timing_loss is None:
        return False

    loss = float(combined_timing_loss)
    source = str(loss_source or "combined_timing_loss")
    state["last_loss"] = loss
    state["last_iteration"] = int(iteration)
    state["last_loss_source"] = source

    best_loss = state.get("best_loss")
    min_delta = float(state.get("min_delta", 0.0))
    if best_loss is None or loss < best_loss - min_delta:
        state["best_loss"] = loss
        state["best_iteration"] = int(iteration)
        state["loss_source"] = source
        if state.get("restore_best"):
            best_state = state.pop("candidate_state", None)
            if not isinstance(best_state, dict):
                best_state = context.capture_best()
            state["best_state"] = best_state
        state["no_improve_rounds"] = 0
        return False

    state["no_improve_rounds"] = int(state.get("no_improve_rounds", 0)) + 1
    patience = state.get("patience")
    if patience is not None and int(patience) > 0 and state["no_improve_rounds"] >= int(patience):
        state["triggered"] = True
        state["trigger_iteration"] = int(iteration)
        state["trigger_reason"] = "continuous_no_improvement"
        return True
    return False


def continuous_early_stop_summary(state):
    if not isinstance(state, dict):
        return None
    return {
        "enabled": bool(state.get("enabled")),
        "patience": state.get("patience"),
        "min_delta": state.get("min_delta"),
        "restore_best": bool(state.get("restore_best")),
        "best_loss": state.get("best_loss"),
        "best_iteration": state.get("best_iteration"),
        "loss_source": state.get("loss_source"),
        "best_state_available": isinstance(state.get("best_state"), dict),
        "last_loss": state.get("last_loss"),
        "last_iteration": state.get("last_iteration"),
        "last_loss_source": state.get("last_loss_source"),
        "no_improve_rounds": state.get("no_improve_rounds"),
        "triggered": bool(state.get("triggered")),
        "trigger_iteration": state.get("trigger_iteration"),
        "trigger_reason": state.get("trigger_reason"),
        "restored": bool(state.get("restored")),
        "restore_iteration": state.get("restore_iteration"),
        "restore_reason": state.get("restore_reason"),
    }
