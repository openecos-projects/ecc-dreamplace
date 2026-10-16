"""Timing-stage and endpoint comparison artifacts for ECC optimization flows.

Inputs are evaluated stage records or read-only endpoint/projection summaries.
This module assembles and writes artifacts; STA/mutation scheduling and engine
publication stay with the caller. Writer callbacks only assemble report data.
"""

import csv
import json
import logging
import numpy as np
import torch
from dreamplace.ops.timing_net_weighting.net_weighting import PIN2PIN_PAIR_RESET_INTERVAL
import os
from dataclasses import dataclass
from typing import Callable


@dataclass
class StageCompareWriterContext:
    build_stage_record: Callable
    build_changed_cells: Callable


def load_json_artifact_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows_if_exists(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def artifact_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def artifact_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def artifact_value(value):
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in ("0", "1", "true", "false", "yes", "no"):
        return artifact_bool(value)
    scalar = artifact_float(value)
    if scalar is not None:
        return scalar
    return value


def build_changed_cell_summary(
    params, *, rows, load_summary, cone_node_ids, stage_hits
):
    csv_path = os.path.join(params.result_dir, f"{params.design_name()}_projection.csv")
    summary_path = os.path.join(
        params.result_dir, f"{params.design_name()}_projection_summary.json"
    )
    changed_rows = []
    for row in rows:
        if not artifact_bool(row.get("changed_cell")):
            continue
        inst_id = int(float(row["inst_id"]))
        current_size = artifact_float(row.get("current_size"))
        continuous_size = artifact_float(row.get("continuous_size"))
        projected_size = artifact_float(row.get("projected_size"))
        timing_penalty = artifact_float(row.get("term_timing_penalty"))
        downsize = None
        if continuous_size is not None and projected_size is not None:
            downsize = max(continuous_size - projected_size, 0.0)
        local_timing_score = 0.0
        if downsize is not None and downsize > 1e-12 and timing_penalty is not None:
            local_timing_score = timing_penalty / downsize
        if cone_node_ids:
            critical_endpoint_cone_membership = inst_id in cone_node_ids
        else:
            critical_endpoint_cone_membership = bool(local_timing_score > 0.0)
        changed_row = {
            "inst_id": inst_id,
            "original_size": current_size,
            "projected_size": projected_size,
            "continuous_size": continuous_size,
            "local_timing_score": local_timing_score,
            "critical_endpoint_cone_membership": critical_endpoint_cone_membership,
            "critical_endpoint_stage_hits": stage_hits.get(inst_id, []),
        }
        for key, value in row.items():
            if key.startswith("metadata_current_candidate_"):
                changed_row[key] = artifact_value(value)
        changed_rows.append(changed_row)
    changed_rows.sort(
        key=lambda item: (
            -(item.get("local_timing_score") or 0.0),
            -abs(
                (item.get("projected_size") or 0.0) - (item.get("original_size") or 0.0)
            ),
        )
    )
    summary = load_summary()
    return {
        "num_changed_cells": int(summary.get("num_changed_cells", len(changed_rows))),
        "projection_csv_path": csv_path if os.path.exists(csv_path) else None,
        "projection_summary_path": summary_path
        if os.path.exists(summary_path)
        else None,
        "rows": changed_rows[:50],
    }


def build_stage_compare_stage_record(
    timing_stage_summary,
    stage_name,
    *,
    endpoint_payload,
    top_endpoints,
    build_secondary,
    can_be_primary,
):
    diagnostic = None
    diagnostic_top_endpoints = []
    if can_be_primary(endpoint_payload):
        primary = endpoint_payload
    else:
        primary = {}
        if isinstance(endpoint_payload, dict) and endpoint_payload.get("available"):
            diagnostic = dict(endpoint_payload)
            diagnostic_top_endpoints = top_endpoints
        top_endpoints = []
    stage_compare_override = (
        timing_stage_summary.get("stage_compare_overrides") or {}
    ).get(stage_name) or {}
    override_primary = stage_compare_override.get("primary")
    if isinstance(override_primary, dict):
        if can_be_primary(override_primary):
            primary = dict(override_primary)
        else:
            diagnostic = dict(override_primary)
            primary = {}
    override_diagnostic = stage_compare_override.get("diagnostic")
    if isinstance(override_diagnostic, dict):
        diagnostic = dict(override_diagnostic)
    override_top_endpoints = stage_compare_override.get("top_endpoints")
    if isinstance(override_top_endpoints, list):
        top_endpoints = [
            dict(row) for row in override_top_endpoints if isinstance(row, dict)
        ]
    override_diagnostic_top_endpoints = stage_compare_override.get(
        "diagnostic_top_endpoints"
    )
    if isinstance(override_diagnostic_top_endpoints, list):
        diagnostic_top_endpoints = [
            dict(row)
            for row in override_diagnostic_top_endpoints
            if isinstance(row, dict)
        ]
    secondary = build_secondary()
    if not primary:
        primary = dict(secondary)
    stage_record = {
        "WNS": primary.get("wns")
        if primary.get("wns") is not None
        else secondary.get("wns"),
        "TNS": primary.get("tns")
        if primary.get("tns") is not None
        else secondary.get("tns"),
        "SlewVio": secondary.get("slew_violation"),
        "CapVio": secondary.get("cap_violation"),
        "Leakage": secondary.get("leakage"),
        "wns": primary.get("wns")
        if primary.get("wns") is not None
        else secondary.get("wns"),
        "tns": primary.get("tns")
        if primary.get("tns") is not None
        else secondary.get("tns"),
        "slew_violation": secondary.get("slew_violation"),
        "cap_violation": secondary.get("cap_violation"),
        "leakage": secondary.get("leakage"),
        "primary": primary,
        "secondary": secondary,
    }
    if diagnostic is not None:
        stage_record["diagnostic"] = diagnostic
    if diagnostic_top_endpoints:
        stage_record["diagnostic_top_endpoints"] = diagnostic_top_endpoints
    if secondary.get("stage_note"):
        stage_record["note"] = secondary.get("stage_note")
    return stage_record, top_endpoints


def write_stage_compare_latest_artifact(context, params, timing_stage_summary, *, mode):
    strict_stage_order = ["pre_projection", "post_projection", "post_legalization"]
    top_critical_endpoints = {}
    latest = {
        "artifact_version": 1,
        "metadata": {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": mode,
            "strict_stage_order": strict_stage_order,
        },
    }
    for stage_name in strict_stage_order:
        stage_record, top_rows = context.build_stage_record(
            timing_stage_summary,
            params,
            stage_name,
        )
        latest[stage_name] = stage_record
        top_critical_endpoints[stage_name] = top_rows
    latest["top_critical_endpoints"] = top_critical_endpoints
    latest["changed_cell_summary"] = context.build_changed_cells(
        params,
        top_critical_endpoints=top_critical_endpoints,
    )
    latest_path = os.path.join(
        params.result_dir, f"{params.design_name()}_stage_compare_latest.json"
    )
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(latest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return latest, latest_path


def refresh_stage_compare_latest_artifact(
    params, timing_stage_summary, *, write_artifact
):
    if timing_stage_summary is None:
        summary_path = os.path.join(
            params.result_dir, f"{params.design_name()}_timing_stage_summary.json"
        )
        timing_stage_summary = load_json_artifact_if_exists(summary_path)
    if not isinstance(timing_stage_summary, dict) or not timing_stage_summary:
        return None
    return write_artifact(params, timing_stage_summary)


def write_timing_stage_summary(
    params,
    *,
    mode,
    place_record,
    post_projection_record,
    legalization_record,
    delta,
    retained_timing_improvement,
    continuous_early_stop,
    stage_compare_overrides,
):
    """Write already evaluated stage records without invoking timing or mutation."""
    summary = {
        "artifact_version": 2,
        "metadata": {
            "artifact_scope": "post_optimization",
            "placement_sizing_mode": mode,
            "strict_stage_order": [
                "pre_projection",
                "post_projection",
                "post_legalization",
            ],
        },
        "pre_projection": place_record,
        "post_projection": post_projection_record,
        "post_legalization": legalization_record,
        "place": place_record,
        "delta": delta,
        "retained_timing_improvement": retained_timing_improvement,
    }
    if continuous_early_stop is not None:
        summary["continuous_early_stop"] = continuous_early_stop
    if stage_compare_overrides:
        summary["stage_compare_overrides"] = stage_compare_overrides
    summary_path = os.path.join(
        params.result_dir, f"{params.design_name()}_timing_stage_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def write_post_projection_timing_summary(
    params, timing_stage_summary, snapshot, *, enrich_record, add_endpoint_override
):
    """Write a prepared post-projection snapshot; timing evaluation stays outside."""
    refreshed_summary = dict(timing_stage_summary)
    refreshed_summary["post_projection"] = enrich_record(dict(snapshot["record"]))
    stage_compare_overrides = dict(
        refreshed_summary.get("stage_compare_overrides") or {}
    )
    stage_override = {}
    add_endpoint_override(
        stage_override,
        snapshot.get("endpoint_payload"),
        snapshot.get("top_endpoints"),
    )
    if stage_override:
        stage_compare_overrides["post_projection"] = stage_override
        refreshed_summary["stage_compare_overrides"] = stage_compare_overrides

    summary_path = os.path.join(
        params.result_dir, f"{params.design_name()}_timing_stage_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(refreshed_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return refreshed_summary

def normalize_pin_name(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="ignore")
    return str(value)


def ps_to_ns(value):
    scalar = artifact_float(value)
    if scalar is None:
        return None
    return scalar / 1000.0


@dataclass(frozen=True)
class EndpointArtifactContext:
    stale_reason: Callable

def build_endpoint_stage_payload(context, params, stage_name):
    design_name = params.design_name()
    if stage_name == "pre_projection":
        prefix = f"{design_name}_place_stage"
    elif stage_name == "post_projection":
        prefix = f"{design_name}_post_projection_stage"
    elif stage_name == "post_legalization":
        prefix = design_name
    else:
        return {
            "source": "backend_endpoint_debug",
            "available": False,
        }, []

    summary_path = os.path.join(params.result_dir, f"{prefix}_endpoint_timing_compare_summary.json")
    csv_path = os.path.join(params.result_dir, f"{prefix}_endpoint_timing_compare.csv")
    stale_reason = context.stale_reason(summary_path, csv_path)
    if stale_reason is not None:
        return (
            {
                "source": "backend_endpoint_debug",
                "available": False,
                "summary_path": summary_path if os.path.exists(summary_path) else None,
                "csv_path": csv_path if os.path.exists(csv_path) else None,
                "wns": None,
                "tns": None,
                "num_negative_endpoints": None,
                "python_wns": None,
                "python_tns": None,
                "stale_reason": stale_reason,
            },
            [],
        )
    summary = load_json_artifact_if_exists(summary_path)
    rows = load_csv_rows_if_exists(csv_path)
    backend_metadata = dict(summary.get("backend_metadata", {}) or {})
    top_rows = []
    for row in rows:
        cpp_slack = artifact_float(row.get("cpp_slack"))
        py_slack = artifact_float(row.get("py_slack"))
        slack_delta = artifact_float(row.get("slack_delta"))
        ranking_slack = cpp_slack if cpp_slack is not None else py_slack
        top_rows.append(
            {
                "pin_name": row.get("pin_name"),
                "backend_slack_ns": ps_to_ns(cpp_slack),
                "ieda_slack_ns": ps_to_ns(cpp_slack),
                "autodmp_slack_ns": ps_to_ns(py_slack),
                "slack_delta_ns": ps_to_ns(slack_delta),
                "_ranking_slack_ps": ranking_slack if ranking_slack is not None else float("inf"),
            }
        )
    top_rows.sort(key=lambda item: item["_ranking_slack_ps"])
    top_rows = top_rows[:10]
    for row in top_rows:
        row.pop("_ranking_slack_ps", None)
    return (
        {
            "source": "backend_endpoint_debug",
            "available": bool(summary or rows),
            "summary_path": summary_path if os.path.exists(summary_path) else None,
            "csv_path": csv_path if os.path.exists(csv_path) else None,
            "wns": ps_to_ns(summary.get("backend_wns", summary.get("ieda_wns"))),
            "tns": ps_to_ns(summary.get("backend_tns", summary.get("ieda_tns"))),
            "num_negative_endpoints": summary.get(
                "num_backend_negative_slack",
                summary.get("num_ieda_negative_slack"),
            ),
            "python_wns": ps_to_ns(summary.get("python_wns")),
            "python_tns": ps_to_ns(summary.get("python_tns")),
            "backend_metadata": backend_metadata,
            "sta_reference_role": backend_metadata.get("sta_reference_role"),
            "sta_is_golden": backend_metadata.get("sta_is_golden"),
            "parasitics_initialization": backend_metadata.get("parasitics_initialization"),
            "sta_state_status": backend_metadata.get("sta_state_status"),
        },
        top_rows,
    )


def build_secondary_stage_payload(timing_stage_summary, stage_name):
    record = timing_stage_summary.get(stage_name) or {}
    return {
        "source": "AutoDMP",
        "available": bool(record),
        "wns": ps_to_ns(record.get("wns")),
        "tns": ps_to_ns(record.get("tns")),
        "slew_violation": artifact_float(record.get("slew_violation")),
        "cap_violation": artifact_float(record.get("cap_violation")),
        "leakage": artifact_float(record.get("leakage")),
        "combined_timing_loss": artifact_float(record.get("combined_timing_loss")),
        "timing_objective": artifact_float(record.get("timing_objective")),
        "stage_note": record.get("stage_note"),
    }

@dataclass(frozen=True)
class LegacyWeightReportContext:
    update_count: int
    gate_summary: dict
    write_artifact: Callable

def record_legacy_net_weight_update(
    context,
    params,
    placedb,
    *,
    iteration,
    npaths,
    wns,
    tns,
    update_ms,
    gate_status,
    pin2pin_objective,
    pin2pin_path_pair_backend,
    pin2pin_attraction_backend,
    pin2pin_pair_reset_before_update=False,
    update_reason=None,
):
    net_weights = np.asarray(getattr(placedb, "net_weights", []), dtype=np.float64)
    finite = net_weights[np.isfinite(net_weights)]
    if finite.size:
        max_weight = float(np.max(finite))
        min_weight = float(np.min(finite))
        non_unit_count = int(np.count_nonzero(np.abs(finite - 1.0) > 1e-6))
    else:
        max_weight = 1.0
        min_weight = 1.0
        non_unit_count = 0
    pair_tensor = getattr(placedb, "_pin2pin_pair_weights_tensor", None)
    if torch.is_tensor(pair_tensor):
        pair_values = pair_tensor.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        pair_count = int(pair_values.size)
    else:
        pair_mapping = getattr(placedb, "pin2pin_net_weight", {}) or {}
        pair_count = int(len(pair_mapping))
        pair_values = np.asarray(list(pair_mapping.values()), dtype=np.float64)
    pair_weight_min = float(np.min(pair_values)) if pair_values.size else None
    pair_weight_max = float(np.max(pair_values)) if pair_values.size else None
    pair_weight_mean = float(np.mean(pair_values)) if pair_values.size else None
    native_pin2pin_summary = dict(
        getattr(placedb, "_pin2pin_last_native_summary", {}) or {}
    )
    gate_summary = context.gate_summary
    gate_summary["update_count"] = int(context.update_count)
    if gate_summary["first_update_iteration"] is None:
        gate_summary["first_update_iteration"] = int(iteration)
        gate_summary["first_update_overflow"] = gate_status.get("overflow")
    gate_summary["last_pair_count"] = pair_count
    gate_summary["last_pin2pin_objective"] = pin2pin_objective
    gate_summary["pin2pin_backend"] = pin2pin_attraction_backend
    gate_summary["pin2pin_path_pair_backend"] = pin2pin_path_pair_backend
    payload = {
        "artifact": "legacy_net_weight_summary",
        "artifact_version": 2,
        "status": "updated",
        "carrier": "legacy_net_weight",
        "iteration": int(iteration),
        "update_count": int(context.update_count),
        "scheme": str(getattr(params, "net_weighting_scheme", "")),
        "npaths": int(npaths),
        "pin2pin_endpoint_limit": int(npaths),
        "pin2pin_endpoint_limit_policy": (
            "all_violating" if int(npaths) <= 0 else "worst_k_endpoint_states"
        ),
        # Keep the old fields for consumers of existing experiment tables.
        "pin2pin_path_limit": int(npaths),
        "pin2pin_path_limit_policy": (
            "all_violating" if int(npaths) <= 0 else "worst_k"
        ),
        "wns": float(wns.detach().cpu().item()) if torch.is_tensor(wns) else float(wns),
        "tns": float(tns.detach().cpu().item()) if torch.is_tensor(tns) else float(tns),
        "net_count": int(net_weights.size),
        "non_unit_net_weight_count": non_unit_count,
        "max_net_weight": max_weight,
        "min_net_weight": min_weight,
        "pin2pin_pair_count": pair_count,
        "pin2pin_weight": float(getattr(params, "pin2pin_weight", 0.0)),
        "pin2pin_min_weight": float(
            getattr(params, "pin2pin_min_weight", 0.0)
        ),
        "pin2pin_max_weight": float(
            getattr(params, "pin2pin_max_weight", float("inf"))
        ),
        "pin2pin_accumulate_weight": float(
            getattr(params, "pin2pin_accumulate_weight", 0.0)
        ),
        "pin2pin_pair_reset_interval": PIN2PIN_PAIR_RESET_INTERVAL,
        "pin2pin_pair_reset_before_update": bool(
            pin2pin_pair_reset_before_update
        ),
        "pin2pin_pair_reset_count": int(
            gate_summary.get("pin2pin_pair_reset_count", 0)
        ),
        "pin2pin_pair_reset_update_counts": list(
            gate_summary.get("pin2pin_pair_reset_update_counts", [])
        ),
        "pin2pin_observed_weight_min": pair_weight_min,
        "pin2pin_observed_weight_max": pair_weight_max,
        "pin2pin_observed_weight_mean": pair_weight_mean,
        "pin2pin_objective": pin2pin_objective,
        "pin2pin_weighted_objective": (
            None
            if pin2pin_objective is None
            else pin2pin_objective * float(getattr(params, "pin2pin_weight", 0.0))
        ),
        "pin2pin_backend": pin2pin_attraction_backend,
        "pin2pin_path_pair_backend": pin2pin_path_pair_backend,
        "pin2pin_native_summary": native_pin2pin_summary,
        "update_ms": float(update_ms),
        "trigger_policy": "overflow",
        "trigger_threshold": gate_status.get("threshold"),
        "trigger_overflow": gate_status.get("overflow"),
        "update_interval": gate_status.get("update_interval"),
        "update_reason": str(update_reason or gate_status.get("reason") or ""),
    }
    context.write_artifact(params, payload)
    logging.info(
        "Legacy net-weight update at iteration=%d count=%d "
        "pairs=%d overflow=%.6f pin2pin_objective=%s "
        "path_pair_backend=%s attraction_backend=%s "
        "wns=%.6f tns=%.6f update_ms=%.3f",
        int(iteration),
        payload["update_count"],
        payload["pin2pin_pair_count"],
        payload["trigger_overflow"],
        payload["pin2pin_objective"],
        payload["pin2pin_path_pair_backend"],
        payload["pin2pin_backend"],
        payload["wns"],
        payload["tns"],
        payload["update_ms"],
    )
    return payload
