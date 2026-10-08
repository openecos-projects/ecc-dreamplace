"""Pure metric and timing-state conversions used by the placement flow."""

import math

import numpy as np
import torch


def timing_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def combined_timing_loss(wns, tns):
    if wns is None or tns is None:
        return None
    return float(-(0.01 * float(wns) + 0.0001 * float(tns)))


def finite_timing_scalar(value):
    if value is None:
        return None
    try:
        scalar = timing_scalar(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return scalar if math.isfinite(scalar) else None


def update_metric_timing(metric, wns, tns, ws, timing_op):
    metric.wns = timing_scalar(wns)
    metric.tns = timing_scalar(tns)
    metric.timing_objective = timing_scalar(ws)
    if timing_op is not None:
        metric.slew_violation = getattr(timing_op, "last_total_slew_violation", None)
        metric.cap_violation = getattr(timing_op, "last_total_cap_violation", None)
        metric.leakage = getattr(timing_op, "last_total_leakage", None)


def metric_snapshot(metric):
    objective = getattr(metric, "objective", None)
    if torch.is_tensor(objective):
        objective = objective.detach().clone()
    snapshot = {
        "objective": objective,
        "wns": timing_scalar(getattr(metric, "wns", None)),
        "tns": timing_scalar(getattr(metric, "tns", None)),
        "timing_objective": timing_scalar(getattr(metric, "timing_objective", None)),
        "slew_violation": timing_scalar(getattr(metric, "slew_violation", None)),
        "cap_violation": timing_scalar(getattr(metric, "cap_violation", None)),
        "leakage": timing_scalar(getattr(metric, "leakage", None)),
        "combined_timing_loss": timing_scalar(
            getattr(metric, "combined_timing_loss", None)
        ),
    }
    if snapshot["combined_timing_loss"] is None:
        snapshot["combined_timing_loss"] = combined_timing_loss(
            snapshot["wns"], snapshot["tns"]
        )
    return snapshot


def apply_metric_snapshot(metric, snapshot):
    if metric is None or not isinstance(snapshot, dict):
        return
    for attr_name in (
        "objective", "wns", "tns", "timing_objective",
        "slew_violation", "cap_violation", "leakage", "combined_timing_loss",
    ):
        if attr_name not in snapshot:
            continue
        value = snapshot.get(attr_name)
        if attr_name == "objective" and value is not None and not torch.is_tensor(value):
            reference = getattr(metric, "objective", None)
            if torch.is_tensor(reference):
                value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
            else:
                value = torch.tensor(value, dtype=torch.float32)
        setattr(metric, attr_name, value)


def flatten_metrics(items):
    return sum(map(flatten_metrics, items), []) if isinstance(items, list) else [items]


def metric_objective_is_nonfinite(metric):
    objective = getattr(metric, "objective", None)
    if objective is None:
        return False
    if torch.is_tensor(objective):
        return not bool(torch.isfinite(objective).all().item())
    return not math.isfinite(float(objective))


def last_metric_from_metrics(metrics):
    if isinstance(metrics, list):
        for item in reversed(metrics):
            metric = last_metric_from_metrics(item)
            if metric is not None:
                return metric
        return None
    return metrics


def value_has_nonfinite(value):
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach()
    if isinstance(value, torch.Tensor):
        return not bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return not bool(np.isfinite(value).all())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    return False


def metric_has_nonfinite_value(metric):
    for field_name, value in (getattr(metric, "__dict__", {}) or {}).items():
        if field_name not in {"iteration", "detailed_step", "eval_time"} and value_has_nonfinite(value):
            return True
    return False


def should_skip_post_global_placement(stop_reason, last_metric):
    if stop_reason in {"gradient_nan", "nonfinite_metric"}:
        return True
    return bool(last_metric and metric_has_nonfinite_value(last_metric))

def metric_to_record(metric):
    def scalar_or_none(value):
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "numel"):
            if value.numel() == 1:
                return float(value.item())
            return [float(v) for v in value.detach().cpu().numpy().tolist()]
        return float(value)

    return {
        "iteration": metric.iteration,
        "detailed_step": list(metric.detailed_step) if metric.detailed_step is not None else None,
        "objective": scalar_or_none(metric.objective),
        "wirelength": scalar_or_none(metric.wirelength),
        "density": scalar_or_none(metric.density),
        "density_weight": scalar_or_none(metric.density_weight),
        "hpwl": scalar_or_none(metric.hpwl),
        "overflow": scalar_or_none(metric.overflow),
        "max_density": scalar_or_none(metric.max_density),
        "gamma": scalar_or_none(metric.gamma),
        "wns": scalar_or_none(metric.wns),
        "tns": scalar_or_none(metric.tns),
        "timing_objective": scalar_or_none(metric.timing_objective),
        "slew_violation": scalar_or_none(metric.slew_violation),
        "cap_violation": scalar_or_none(metric.cap_violation),
        "leakage": scalar_or_none(metric.leakage),
        "runtime_seconds": scalar_or_none(metric.eval_time),
        "size_min": scalar_or_none(metric.size_min),
        "size_mean": scalar_or_none(metric.size_mean),
        "size_max": scalar_or_none(metric.size_max),
        "sizing_parameterization": getattr(metric, "sizing_parameterization", None),
        "real_size_masked_gradient_count": getattr(
            metric, "real_size_masked_gradient_count", None
        ),
        "real_size_applied": getattr(metric, "real_size_applied", None),
        "real_size_lower_clamped_count": getattr(
            metric, "real_size_lower_clamped_count", None
        ),
        "real_size_upper_clamped_count": getattr(
            metric, "real_size_upper_clamped_count", None
        ),
        "real_size_fixed_instance_count": getattr(
            metric, "real_size_fixed_instance_count", None
        ),
        "real_size_max_projection_delta": getattr(
            metric, "real_size_max_projection_delta", None
        ),
        "real_size_transition": getattr(metric, "real_size_transition", None),
        "vt_class_proportions": metric.vt_class_proportions,
        "joint_placement_grad_norm": scalar_or_none(
            getattr(metric, "joint_placement_grad_norm", None)
        ),
        "joint_sizing_grad_norm": scalar_or_none(
            getattr(metric, "joint_sizing_grad_norm", None)
        ),
        "joint_buffering_grad_norm": scalar_or_none(
            getattr(metric, "joint_buffering_grad_norm", None)
        ),
        "joint_placement_grad_param_count": scalar_or_none(
            getattr(metric, "joint_placement_grad_param_count", None)
        ),
        "joint_sizing_grad_param_count": scalar_or_none(
            getattr(metric, "joint_sizing_grad_param_count", None)
        ),
        "joint_buffering_grad_param_count": scalar_or_none(
            getattr(metric, "joint_buffering_grad_param_count", None)
        ),
        "joint_z_param_min": scalar_or_none(
            getattr(metric, "joint_z_param_min", None)
        ),
        "joint_z_param_max": scalar_or_none(
            getattr(metric, "joint_z_param_max", None)
        ),
        "joint_buffer_commit_status": getattr(
            metric,
            "joint_buffer_commit_status",
            None,
        ),
        "joint_projected_buffer_count": scalar_or_none(
            getattr(metric, "joint_projected_buffer_count", None)
        ),
    }


def enrich_timing_record(record):
    enriched = dict(record)
    wns = enriched.get("wns")
    tns = enriched.get("tns")
    if isinstance(wns, (float, int)) and isinstance(tns, (float, int)):
        enriched["combined_timing_loss"] = combined_timing_loss(wns, tns)
    else:
        enriched["combined_timing_loss"] = None
    return enriched


def timing_stage_delta(place_record, legalization_record):
    delta = {}
    for key in (
        "wns",
        "tns",
        "timing_objective",
        "combined_timing_loss",
        "slew_violation",
        "cap_violation",
        "leakage",
        "runtime_seconds",
        "hpwl",
        "overflow",
        "max_density",
    ):
        place_value = place_record.get(key)
        legalization_value = legalization_record.get(key)
        if isinstance(place_value, (float, int)) and isinstance(legalization_value, (float, int)):
            delta[key] = legalization_value - place_value
        else:
            delta[key] = None
    return delta


def retained_timing_improvement(place_record, legalization_record):
    if (
        isinstance(place_record.get("combined_timing_loss"), (float, int))
        and isinstance(legalization_record.get("combined_timing_loss"), (float, int))
    ):
        return legalization_record["combined_timing_loss"] <= place_record["combined_timing_loss"]
    if (
        isinstance(place_record.get("timing_objective"), (float, int))
        and isinstance(legalization_record.get("timing_objective"), (float, int))
    ):
        return legalization_record["timing_objective"] >= place_record["timing_objective"]
    return None


def processed_global_place_metrics(all_metrics):
    fields = ("objective", "hpwl", "overflow", "max_density")
    values = {name: [] for name in fields}
    incomplete = 0
    for metric in flatten_metrics(all_metrics):
        if metric is None or any(
            getattr(metric, name, None) is None for name in fields
        ):
            incomplete += 1
            continue
        for name in fields:
            value = getattr(metric, name)
            values[name].append(
                value.detach().cpu().item()
                if torch.is_tensor(value)
                else float(value)
            )
    return {
        "objective": values["objective"],
        "hpwl": values["hpwl"],
        "overflow": values["overflow"],
        "density": values["max_density"],
        "incomplete_metric_count": incomplete,
    }
