"""Pure ECC timing-placement gate and net-weighting policy helpers."""

import math

import torch

from dreamplace.ops.timing_net_weighting.net_weighting import (
    PIN2PIN_PAIR_RESET_INTERVAL,
)


def placement_sizing_mode(params):
    return getattr(params, "placement_sizing_mode", "place_only")


def timing_placement_carrier(params):
    return str(
        getattr(params, "timing_placement_carrier", "direct_loss")
        or "direct_loss"
    )


def diff_tdp_enabled(params):
    if placement_sizing_mode(params) == "size_only":
        return False
    if not bool(getattr(params, "with_sta", False)):
        return False
    if not bool(getattr(params, "differentiable_timing_obj", False)):
        return False
    return bool(getattr(params, "diff_timing_driven_placement", True))


def diff_tdp_uses_direct_loss(params):
    return timing_placement_carrier(params) == "direct_loss"


def diff_tdp_uses_gradient_net_weight(params):
    return timing_placement_carrier(params) == "gradient_net_weight"


def diff_tdp_controls_direct_timing_objective(params):
    return (
        placement_sizing_mode(params) != "size_only"
        and diff_tdp_uses_direct_loss(params)
    )


def should_use_live_timing_loss(params, step_status):
    mode = placement_sizing_mode(params)
    configured = (
        mode in ("size_only", "joint")
        and params.with_sta
        and params.differentiable_timing_obj
    )
    if mode == "size_only":
        return configured
    if not diff_tdp_enabled(params):
        return configured
    if not diff_tdp_uses_direct_loss(params):
        return False
    return bool((step_status or {}).get("timing_objective_active", False))


def should_refresh_live_timing_before_objective(params, step_status):
    return should_use_live_timing_loss(params, step_status) and not diff_tdp_enabled(
        params
    )


def new_diff_tdp_gate_summary(params=None):
    return {
        "artifact": "diff_tdp_gate_summary",
        "artifact_version": 1,
        "enabled": diff_tdp_enabled(params) if params is not None else False,
        "carrier": timing_placement_carrier(params) if params is not None else None,
        "threshold": float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
        ) if params is not None else None,
        "refresh_interval": int(
            getattr(params, "timing_topology_refresh_interval", 10) or 0
        ) if params is not None else None,
        "open_count": 0,
        "closed_count": 0,
        "interval_skip_count": 0,
        "warmup_skip_count": 0,
        "invalid_overflow_count": 0,
        "disabled_interval_count": 0,
        "disabled_tdp_count": 0,
        "net_weight_reset_count": 0,
        "timing_active_step_count": 0,
        "timing_inactive_step_count": 0,
        "topology_refresh_count": 0,
        "first_timing_active_iteration": None,
        "first_timing_active_overflow": None,
        "last_gate_status": None,
        "last_reset_iteration": None,
    }


def record_diff_tdp_gate_status(summary, status):
    summary = summary if isinstance(summary, dict) else {}
    status = dict(status or {})
    summary["last_gate_status"] = status
    if status.get("timing_objective_active"):
        summary["timing_active_step_count"] = int(
            summary.get("timing_active_step_count", 0)
        ) + 1
        if summary.get("first_timing_active_iteration") is None:
            summary["first_timing_active_iteration"] = status.get("iteration")
            summary["first_timing_active_overflow"] = status.get("overflow")
    else:
        summary["timing_inactive_step_count"] = int(
            summary.get("timing_inactive_step_count", 0)
        ) + 1
    if status.get("topology_refresh_due"):
        summary["topology_refresh_count"] = int(
            summary.get("topology_refresh_count", 0)
        ) + 1
    reason = str(status.get("reason") or "")
    counter_by_reason = {
        "overflow_gate": "closed_count",
        "interval": "interval_skip_count",
        "warmup_iteration": "warmup_skip_count",
        "invalid_overflow": "invalid_overflow_count",
        "disabled_interval": "disabled_interval_count",
        "disabled_tdp": "disabled_tdp_count",
    }
    if status.get("enabled"):
        summary["open_count"] = int(summary.get("open_count", 0)) + 1
    elif reason in counter_by_reason:
        key = counter_by_reason[reason]
        summary[key] = int(summary.get(key, 0)) + 1
    return summary


def diff_tdp_gate_status(params, iteration, overflow):
    interval = int(getattr(params, "timing_topology_refresh_interval", 10) or 0)
    threshold = float(
        getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
    )
    status = {
        "enabled": False,
        "timing_objective_active": False,
        "topology_refresh_due": False,
        "iteration": int(iteration),
        "interval": interval,
        "threshold": threshold,
        "overflow": None,
    }
    if not diff_tdp_enabled(params):
        return {**status, "reason": "disabled_tdp"}
    if int(iteration) <= 0:
        return {**status, "reason": "warmup_iteration"}
    try:
        overflow_value = float(torch.as_tensor(overflow).detach().cpu().item())
    except (RuntimeError, TypeError, ValueError):
        return {**status, "reason": "invalid_overflow"}
    status["overflow"] = overflow_value
    if not overflow_value < threshold:
        return {**status, "reason": "overflow_gate"}
    status["timing_objective_active"] = True
    if interval <= 0:
        return {**status, "reason": "disabled_interval"}
    if int(iteration) % interval != 0:
        return {**status, "reason": "interval"}
    status["enabled"] = True
    status["topology_refresh_due"] = True
    return {**status, "reason": "ok"}


def freeze_diff_tdp_step_status(
    params,
    iteration,
    overflow,
    previous_status,
    live_timing_topology_initialized,
):
    status = diff_tdp_gate_status(params, iteration, overflow)
    if status["timing_objective_active"] and (
        not bool((previous_status or {}).get("timing_objective_active", False))
        or not bool(live_timing_topology_initialized)
    ):
        status.update(
            enabled=True,
            topology_refresh_due=True,
            reason="activation_topology",
        )
    return status


def reset_legacy_net_weight_gate_summary(params):
    enabled = bool(
        getattr(params, "with_sta", False)
        and getattr(params, "enable_net_weighting", False)
        and not diff_tdp_enabled(params)
        and placement_sizing_mode(params) != "size_only"
    )
    return {
        "artifact": "legacy_net_weight_gate_summary",
        "artifact_version": 1,
        "enabled": enabled,
        "scheme": str(getattr(params, "net_weighting_scheme", "")),
        "trigger_policy": "overflow",
        "threshold": float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
        ),
        "update_interval": max(
            1,
            int(getattr(params, "net_weighting_update_interval", 15) or 15),
        ),
        "active_step_count": 0,
        "closed_step_count": 0,
        "update_count": 0,
        "pin2pin_pair_reset_interval": PIN2PIN_PAIR_RESET_INTERVAL,
        "pin2pin_pair_reset_count": 0,
        "pin2pin_pair_reset_update_counts": [],
        "last_pin2pin_pair_reset_before_update": None,
        "first_active_iteration": None,
        "first_active_overflow": None,
        "first_update_iteration": None,
        "first_update_overflow": None,
        "last_gate_status": None,
    }


def legacy_net_weight_gate_status(params, iteration, overflow, summary, gate_open):
    summary = summary if isinstance(summary, dict) and summary else reset_legacy_net_weight_gate_summary(params)
    status = {
        "enabled": False,
        "update_due": False,
        "iteration": int(iteration),
        "overflow": None,
        "threshold": float(summary["threshold"]),
        "update_interval": int(summary["update_interval"]),
    }
    if not summary["enabled"]:
        status["reason"] = "disabled"
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    try:
        overflow_value = float(torch.as_tensor(overflow).detach().cpu().item())
    except (RuntimeError, TypeError, ValueError):
        status["reason"] = "invalid_overflow"
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    status["overflow"] = overflow_value
    if not overflow_value < status["threshold"]:
        status["reason"] = "overflow_gate"
        summary["closed_step_count"] += 1
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    first_open_step = not gate_open
    status["enabled"] = True
    status["update_due"] = bool(
        first_open_step or int(iteration) % status["update_interval"] == 0
    )
    status["reason"] = "activation" if first_open_step else (
        "interval" if status["update_due"] else "between_updates"
    )
    summary["active_step_count"] += 1
    if summary["first_active_iteration"] is None:
        summary["first_active_iteration"] = int(iteration)
        summary["first_active_overflow"] = overflow_value
    summary["last_gate_status"] = dict(status)
    return status, True, summary


def net_weighting_npaths_percent(params):
    value = getattr(params, "net_weighting_npaths", 0.0)
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        raise ValueError("net_weighting_npaths must be a percentage in [0, 100]")
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("net_weighting_npaths must be a percentage in [0, 100]")
    return value


def net_weighting_npaths_budget(params, placedb):
    percent = net_weighting_npaths_percent(params)
    if percent <= 0.0:
        return 0
    net_count = int(getattr(placedb, "num_nets", 0) or 0)
    if net_count <= 0:
        return 0
    return max(1, int(net_count * percent / 100.0))


def net_weighting_nendpoints(params, placedb=None):
    canonical = getattr(params, "net_weighting_nendpoints", None)
    legacy = getattr(params, "net_weighting_npaths", None)
    if bool(getattr(params, "_net_weighting_nendpoints_explicit", False)):
        try:
            value = int(canonical or 0)
        except (TypeError, ValueError):
            raise ValueError(
                "net_weighting_nendpoints must be a non-negative integer"
            )
        if value < 0:
            raise ValueError(
                "net_weighting_nendpoints must be a non-negative integer"
            )
        return value
    if legacy is not None:
        return net_weighting_npaths_budget(params, placedb)
    try:
        value = int(canonical or 0)
    except (TypeError, ValueError):
        raise ValueError("net_weighting_nendpoints must be a non-negative integer")
    if value < 0:
        raise ValueError("net_weighting_nendpoints must be a non-negative integer")
    return value
