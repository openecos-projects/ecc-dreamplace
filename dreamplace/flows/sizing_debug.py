"""Observe sizing/buffer parameters and write the existing debug trace.

Snapshots here preserve the early-stop/debug capture contract. This owner never
evaluates objectives, applies parameter updates or refreshes native state.
"""
from dataclasses import dataclass
from typing import Callable
from dreamplace.flows.metric_state import timing_scalar, combined_timing_loss
from dreamplace.ops.buffer_insertion.optimization_state import (
    capture_buffer_optimization_snapshot, buffer_optimization_grad_stats,
)

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

