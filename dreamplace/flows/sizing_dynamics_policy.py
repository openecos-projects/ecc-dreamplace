"""Pure configuration and phase policy for ECC sizing dynamics."""
import math

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
        getattr(params, "timing_objective_lane", "timing_only") or "timing_only"
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


