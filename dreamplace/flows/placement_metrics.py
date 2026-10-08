"""Placement metrics files and terminal debug reporting, without evaluation."""
import os
import json
import copy
from dataclasses import dataclass
from typing import Callable
from dreamplace.flows.metric_state import timing_scalar
from dreamplace.flows.timing_objective_policy import placement_sizing_mode
from dreamplace.flows.timing_artifacts import ps_to_ns

@dataclass(frozen=True)
class MetricsWriterContext:
    flatten: Callable
    record: Callable
    stage: Callable

@dataclass
class DebugSummaryContext:
    last_timing_metrics: object
    diff_tdp_gate: object
    timing_grad_balance: object
    legacy_gate: object
    legacy_artifact: object
    summary: object = None
    path: object = None

def write_metrics_artifacts(context, params, all_metrics, stage_name=None):
    metrics = [metric for metric in context.flatten(all_metrics) if metric is not None]
    if not metrics:
        return None

    records = [context.record(metric) for metric in metrics]
    stage_name = stage_name or context.stage(params)
    metrics_jsonl_paths = [
        os.path.join(params.result_dir, f"{params.design_name()}_{stage_name}_metrics.jsonl")
    ]
    metrics_csv_paths = [
        os.path.join(params.result_dir, f"{params.design_name()}_{stage_name}_metrics.csv")
    ]
    if stage_name == "place":
        metrics_jsonl_paths.append(
            os.path.join(params.result_dir, f"{params.design_name()}_metrics.jsonl")
        )
        metrics_csv_paths.append(
            os.path.join(params.result_dir, f"{params.design_name()}_metrics.csv")
        )

    for metrics_jsonl in metrics_jsonl_paths:
        with open(metrics_jsonl, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    import csv

    fieldnames = [
        "iteration",
        "detailed_step",
        "objective",
        "wirelength",
        "density",
        "density_weight",
        "hpwl",
        "overflow",
        "max_density",
        "gamma",
        "wns",
        "tns",
        "timing_objective",
        "slew_violation",
        "cap_violation",
        "leakage",
        "runtime_seconds",
        "size_min",
        "size_mean",
        "size_max",
        "vt_class_proportions",
        "joint_placement_grad_norm",
        "joint_sizing_grad_norm",
        "joint_buffering_grad_norm",
        "joint_placement_grad_param_count",
        "joint_sizing_grad_param_count",
        "joint_buffering_grad_param_count",
        "joint_z_param_min",
        "joint_z_param_max",
        "joint_buffer_commit_status",
        "joint_projected_buffer_count",
    ]
    # Keep the CSV schema aligned with metric records that carry
    # optional flow-specific diagnostics such as real-size projection.
    fieldnames.extend(
        sorted(
            {
                key
                for record in records
                for key in record
                if key not in fieldnames
            }
        )
    )
    for metrics_csv in metrics_csv_paths:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    return {
        "jsonl": metrics_jsonl_paths[0],
        "csv": metrics_csv_paths[0],
        "stage_name": stage_name,
    }


def write_placement_debug_summary(
    context,
    params,
    *,
    iteration=None,
    optimizer_steps=None,
    processed_metrics=None,
    last_metric=None,
    status="unknown",
    stop_reason=None,
    entered_legalization=False,
    skipped_legalization_reason=None,
):
    if params is None:
        return None
    os.makedirs(params.result_dir, exist_ok=True)

    def scalar_or_none(value):
        if value is None:
            return None
        try:
            return timing_scalar(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    timing = dict(getattr(context, "last_timing_metrics", None) or {})
    final_overflow = None
    final_hpwl = None
    final_max_density = None
    if last_metric is not None:
        final_overflow = scalar_or_none(getattr(last_metric, "overflow", None))
        final_hpwl = scalar_or_none(getattr(last_metric, "hpwl", None))
        final_max_density = scalar_or_none(getattr(last_metric, "max_density", None))
    if final_overflow is None and isinstance(processed_metrics, dict):
        values = processed_metrics.get("overflow") or []
        if values:
            final_overflow = scalar_or_none(values[-1])
    if final_hpwl is None and isinstance(processed_metrics, dict):
        values = processed_metrics.get("hpwl") or []
        if values:
            final_hpwl = scalar_or_none(values[-1])
    if final_max_density is None and isinstance(processed_metrics, dict):
        values = processed_metrics.get("density") or []
        if values:
            final_max_density = scalar_or_none(values[-1])

    gate_summary = copy.deepcopy(
        getattr(context, "diff_tdp_gate", None) or {}
    )
    timing_grad_balance = copy.deepcopy(
        getattr(context, "timing_grad_balance", None) or {}
    )
    legacy_net_weight_gate = copy.deepcopy(
        getattr(context, "legacy_gate", None) or {}
    )
    legacy_net_weight = copy.deepcopy(
        getattr(context, "legacy_artifact", None) or {}
    )
    summary = {
        "artifact": "placement_debug_summary",
        "artifact_version": 1,
        "design_name": params.design_name(),
        "flow_kind": getattr(params, "flow_kind", None),
        "placement_sizing_mode": placement_sizing_mode(params),
        "status": status,
        "stop_reason": stop_reason,
        "actual_iterations": None if iteration is None else int(iteration),
        "optimizer_steps": (
            None if optimizer_steps is None else int(optimizer_steps)
        ),
        "final_overflow": final_overflow,
        "final_hpwl": final_hpwl,
        "final_max_density": final_max_density,
        "entered_legalization": bool(entered_legalization),
        "skipped_legalization_reason": skipped_legalization_reason,
        "divergence_detected": bool(
            stop_reason in {"gradient_nan", "nonfinite_metric"}
            or skipped_legalization_reason is not None
        ),
        "stage1_final_pysta_wns_raw": timing.get("wns"),
        "stage1_final_pysta_tns_raw": timing.get("tns"),
        "stage1_final_pysta_wns_ns": ps_to_ns(timing.get("wns")),
        "stage1_final_pysta_tns_ns": ps_to_ns(timing.get("tns")),
        "stage1_final_pysta_ws_raw": timing.get("ws"),
        "stage1_final_pysta_ts_raw": timing.get("ts"),
        "diff_tdp_gate_summary": gate_summary,
        "timing_grad_balance": timing_grad_balance,
        "legacy_net_weight_gate_summary": legacy_net_weight_gate,
        "legacy_net_weight_summary": legacy_net_weight,
    }
    path = os.path.join(
        params.result_dir,
        f"{params.design_name()}_placement_debug_summary.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    context.summary = summary
    context.path = path
    return path


