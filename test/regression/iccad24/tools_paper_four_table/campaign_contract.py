#!/usr/bin/env python3
"""Frozen profile, identity, matrix, and row contracts for the tools campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROFILE_ARTIFACT = "autodmp_tools_paper_release_profile"
PROFILE_VERSION = 1
EXECUTION_ARTIFACT = "autodmp_tools_paper_execution_manifest"
EXECUTION_VERSION = 1
PRIMARY_TABLE_ORDER = ("placement", "buffering", "sizing", "full_flow")
ICCAD24_CASES = (
    "NV_NVDLA_partition_m",
    "NV_NVDLA_partition_p",
    "aes_256",
    "ariane136",
    "hidden1",
    "hidden2",
    "hidden3",
    "hidden4",
    "hidden5",
    "mempool_tile_wrap",
)
PRIMARY_METHODS = {
    "placement": (
        "or_gpl_plain",
        "or_tdp_reweight_only",
        "autodmp_p_ctrl",
        "autodmp_pin2pin",
    ),
    "buffering": (
        "or_repair_design_buffer_only",
        "autodmp_candidate_route_b",
        "autodmp_equal_spaced_route_b",
    ),
    "sizing": ("or_pure_sizeup", "autodmp_diff_sizing"),
    "full_flow": (
        "or_native_full_flow",
        "autodmp_staged_psb",
        "autodmp_coordinated_psb",
    ),
}
TERMINAL_STATUSES = frozenset(
    {
        "pass",
        "execution_failed",
        "artifact_contract_failed",
        "physical_invalid",
        "evaluation_failed",
        "provenance_mismatch",
        "cuda_unavailable",
    }
)
NONTERMINAL_STATUSES = frozenset({"planned", "not_run", "plan_only"})
EXPECTED_PRIMARY_ROWS = sum(len(methods) for methods in PRIMARY_METHODS.values()) * len(
    ICCAD24_CASES
)
NORMALIZED_ROW_ARTIFACT = "autodmp_tools_paper_normalized_row"
NORMALIZED_ROW_VERSION = 1
REQUIRED_TRACK_A_METRICS = (
    "raw_hpwl_dbu",
    "legalized_hpwl_dbu",
    "wns_ns",
    "tns_ns",
    "violating_endpoint_count",
    "slew_violation_count",
    "slew_violation_total",
    "cap_violation_count",
    "cap_violation_total",
    "total_cell_area_um2",
    "power_total_w",
    "placement_valid",
)
REQUIRED_RUNTIME_FIELDS = (
    "initialization_runtime_sec",
    "core_optimizer_runtime_sec",
    "physical_transaction_runtime_sec",
    "legalization_runtime_sec",
    "track_a_runtime_sec",
    "end_to_end_runtime_sec",
    "peak_rss_kb",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
)
REQUIRED_EXECUTION_FIELDS = (
    "selected_cases",
    "selected_tables",
    "autodmp_revision",
    "autodmp_tracked_diff_sha256",
    "source_file_sha256",
    "native_library_sha256",
    "openroad_path",
    "openroad_sha256",
    "python_path",
    "torch_version",
    "cuda_runtime_version",
    "cuda_visible_devices",
    "logical_gpu_index",
    "gpu_model",
    "gpu_uuid",
    "gpu_preflight_compute_processes",
    "cpu_model",
    "cpu_count",
    "omp_num_threads",
    "torch_num_threads",
    "torch_num_interop_threads",
    "openroad_num_threads",
    "case_concurrency",
    "measurement_policy",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _assert_close(actual: Any, expected: float, name: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} must be {expected}, got {actual}")


def _contains_key(payload: Any, forbidden: str) -> bool:
    if isinstance(payload, dict):
        return forbidden in payload or any(
            _contains_key(value, forbidden) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(_contains_key(value, forbidden) for value in payload)
    return False


def load_release_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    validate_release_profile(profile)
    return profile


def validate_release_profile(profile: dict[str, Any]) -> None:
    if profile.get("artifact") != PROFILE_ARTIFACT:
        raise ValueError("invalid release profile artifact")
    if int(profile.get("artifact_version", -1)) != PROFILE_VERSION:
        raise ValueError("unsupported release profile artifact_version")
    if tuple(profile.get("suite", ())) != ICCAD24_CASES:
        raise ValueError("release profile must contain the canonical ten-case suite")
    if tuple(profile.get("table_order", ())) != PRIMARY_TABLE_ORDER:
        raise ValueError("release profile table order mismatch")
    methods = dict(profile.get("methods") or {})
    for table_id in PRIMARY_TABLE_ORDER:
        if tuple(methods.get(table_id, ())) != PRIMARY_METHODS[table_id]:
            raise ValueError(f"release profile method mismatch for {table_id}")
    if set(methods) != set(PRIMARY_TABLE_ORDER):
        raise ValueError("release profile contains unknown table methods")
    flat_methods = [method for table in PRIMARY_TABLE_ORDER for method in methods[table]]
    if len(flat_methods) != len(set(flat_methods)):
        raise ValueError("release profile method IDs must be unique")
    specs = dict(profile.get("method_specs") or {})
    if set(specs) != set(flat_methods):
        raise ValueError("method_specs must match the primary method matrix")
    for method, spec in specs.items():
        if spec.get("input_domain") not in {"R0", "D_post"}:
            raise ValueError(f"{method}: invalid input domain")
        if not isinstance(spec.get("required_cuda"), bool):
            raise ValueError(f"{method}: required_cuda must be boolean")
        if not spec.get("runner"):
            raise ValueError(f"{method}: missing runner")
    if _contains_key(profile, "gpu_id"):
        raise ValueError("release profile must not contain a physical gpu_id")
    if "per_case" in profile or "case_overrides" in profile:
        raise ValueError("per-case release profile overrides are forbidden")

    r0 = dict(dict(profile.get("input_domains") or {}).get("R0") or {})
    expected_r0 = {
        "seed": 3000,
        "target_density": 0.8,
        "num_bins_x": 256,
        "num_bins_y": 256,
        "auto_adjust_bins": False,
        "enable_fillers": True,
    }
    if r0 != expected_r0:
        raise ValueError("R0 protocol mismatch")
    placement = dict(profile.get("placement") or {})
    if placement.get("optimizer") != "nesterov" or int(placement.get("max_steps", 0)) != 3000:
        raise ValueError("placement optimizer/max_steps mismatch")
    _assert_close(placement.get("stop_overflow"), 0.1, "placement.stop_overflow")
    pin2pin = dict(placement.get("pin2pin") or {})
    expected_pin2pin = {
        "backend": "cuda",
        "path_pair_backend": "cpp_openmp_transition_aware",
        "weight": 0.0005,
        "min_weight": 10.0,
        "max_weight": 50.0,
        "accumulate_weight": 0.2,
        "path_limit": 0,
        "state_domain": "setup_plus_recovery",
        "activation_overflow": 0.3,
        "update_interval": 15,
    }
    if pin2pin != expected_pin2pin:
        raise ValueError("Pin2Pin protocol mismatch")

    buffering = dict(profile.get("buffering") or {})
    expected_buffering = {
        "rounds": 5,
        "selection_fraction": 0.001,
        "max_repeaters_per_segment": 3,
        "fixed_bsu_index": 7,
        "buffer_master": "BUFx4f_ASAP7_75t_R",
        "candidate_count_per_segment": 3,
        "capacity_controller_enabled": False,
        "sizing_enabled": False,
        "final_commit_count": 1,
        "timing_backend": "cpp_cuda_segment_transfer_explicit_autograd",
        "transfer_backend": "segment_transfer_native",
        "candidate_backend": "cuda_explicit_autograd",
    }
    if buffering != expected_buffering:
        raise ValueError("buffering protocol mismatch")
    sizing = dict(profile.get("sizing") or {})
    expected_sizing = {
        "iterations": 50,
        "ranking_mode": "real_size_taylor",
        "up_percent": 30.0,
        "down_percent": 0.0,
        "restore_best": True,
        "buffering_enabled": False,
    }
    if sizing != expected_sizing:
        raise ValueError("sizing protocol mismatch")
    coordinated = dict(profile.get("coordinated_full_flow") or {})
    if coordinated.get("milestones") != [0.3, 0.25, 0.2, 0.15, 0.1]:
        raise ValueError("coordinated milestone protocol mismatch")
    expected_coordinated = {
        "milestones": [0.3, 0.25, 0.2, 0.15, 0.1],
        "sizing_top_percent_per_milestone": 10.0,
        "sizing_up_steps": 1,
        "sizing_down_enabled": False,
        "vt_change_enabled": False,
        "buffering_selection_fraction": 0.001,
        "integer_virtual_buffers": True,
        "virtual_buffer_density_in_placement": True,
        "density_gradient_in_buffer_selection": False,
        "final_combined_commit_count": 1,
    }
    if coordinated != expected_coordinated:
        raise ValueError("coordinated full-flow protocol mismatch")
    track_a = dict(profile.get("track_a") or {})
    expected_track_a = {
        "artifact": "canonical_def_only_track_a",
        "artifact_version": 1,
        "runner": "../../common/def_only_track_a.py",
        "read_verilog": False,
        "detailed_placement_count": 1,
        "detailed_placement_search_window": "full_core",
        "placement_validation_contract": "opendp_plus_exact_macro_keepout_v1",
        "repair_enabled": False,
        "power_input_activity": 0.1,
        "power_input_duty": 0.5,
        "tns_wns_tie_tolerance_ns": 0.001,
        "pareto_relative_tolerance": 0.001,
    }
    if track_a != expected_track_a:
        raise ValueError("canonical Track A protocol mismatch")
    efficient = dict(dict(profile.get("external_references") or {}).get("efficient_tdp") or {})
    if efficient != {"primary_matrix": False, "qualification_required": True}:
        raise ValueError("Efficient-TDP must remain an external qualified reference")
    if tuple(profile.get("execution_manifest_required_fields", ())) != REQUIRED_EXECUTION_FIELDS:
        raise ValueError("execution manifest field contract mismatch")


def validate_execution_manifest(
    manifest: dict[str, Any],
    *,
    profile: dict[str, Any],
) -> None:
    if manifest.get("artifact") != EXECUTION_ARTIFACT:
        raise ValueError("invalid execution manifest artifact")
    if int(manifest.get("artifact_version", -1)) != EXECUTION_VERSION:
        raise ValueError("unsupported execution manifest artifact_version")
    missing = [name for name in REQUIRED_EXECUTION_FIELDS if manifest.get(name) in (None, "")]
    if missing:
        raise ValueError("missing execution manifest fields: " + ", ".join(missing))
    selected = tuple(manifest["selected_cases"])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_cases must be nonempty and unique")
    canonical_selected = tuple(case for case in ICCAD24_CASES if case in set(selected))
    if selected != canonical_selected:
        raise ValueError("selected_cases must follow canonical suite order")
    selected_tables = tuple(manifest["selected_tables"])
    if not selected_tables or len(set(selected_tables)) != len(selected_tables):
        raise ValueError("selected_tables must be nonempty and unique")
    canonical_tables = tuple(
        table for table in PRIMARY_TABLE_ORDER if table in set(selected_tables)
    )
    if selected_tables != canonical_tables:
        raise ValueError("selected_tables must follow canonical table order")
    for field in (
        "cpu_count",
        "omp_num_threads",
        "torch_num_threads",
        "torch_num_interop_threads",
        "openroad_num_threads",
        "case_concurrency",
    ):
        if int(manifest[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    if int(manifest["case_concurrency"]) != 1:
        raise ValueError("tools release campaign requires serial case execution")
    if int(manifest["logical_gpu_index"]) < 0:
        raise ValueError("logical_gpu_index must be nonnegative")
    if manifest["measurement_policy"] not in {"cold_process_per_attempt"}:
        raise ValueError("unsupported measurement_policy")
    for field in ("source_file_sha256", "native_library_sha256"):
        values = manifest.get(field)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{field} must be a nonempty path-to-hash map")
        if any(not path or len(str(digest)) != 64 for path, digest in values.items()):
            raise ValueError(f"{field} contains an invalid path or SHA-256")
    if len(str(manifest["autodmp_tracked_diff_sha256"])) != 64:
        raise ValueError("autodmp_tracked_diff_sha256 must be a SHA-256")
    profile_digest = digest_payload(profile)
    if manifest.get("release_profile_digest") != profile_digest:
        raise ValueError("execution manifest release_profile_digest mismatch")


def campaign_identity(
    profile: dict[str, Any],
    execution_manifest: dict[str, Any],
) -> dict[str, str]:
    validate_release_profile(profile)
    validate_execution_manifest(execution_manifest, profile=profile)
    release_digest = digest_payload(profile)
    execution_digest = digest_payload(execution_manifest)
    campaign_digest = digest_payload(
        {
            "release_profile_digest": release_digest,
            "execution_manifest_digest": execution_digest,
        }
    )
    return {
        "release_profile_digest": release_digest,
        "execution_manifest_digest": execution_digest,
        "campaign_id": "tools4-" + campaign_digest[:20],
    }


def selected_cases(raw: Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ICCAD24_CASES
    requested = set(raw)
    unknown = sorted(requested - set(ICCAD24_CASES))
    if unknown:
        raise ValueError("unknown case(s): " + ", ".join(unknown))
    if not requested:
        raise ValueError("at least one case is required")
    return tuple(case for case in ICCAD24_CASES if case in requested)


def primary_matrix(cases: Iterable[str] = ICCAD24_CASES) -> list[dict[str, Any]]:
    cases_tuple = selected_cases(tuple(cases))
    rows: list[dict[str, Any]] = []
    for table_id in PRIMARY_TABLE_ORDER:
        for case in cases_tuple:
            for method_id in PRIMARY_METHODS[table_id]:
                rows.append(
                    {
                        "table_id": table_id,
                        "case": case,
                        "method_id": method_id,
                        "status": "not_run",
                        "terminal_reason": None,
                    }
                )
    return rows


def _selected_tables(raw: Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return PRIMARY_TABLE_ORDER
    requested = set(raw)
    unknown = sorted(requested - set(PRIMARY_TABLE_ORDER))
    if unknown:
        raise ValueError("unknown table(s): " + ", ".join(unknown))
    if not requested:
        raise ValueError("at least one table is required")
    return tuple(table for table in PRIMARY_TABLE_ORDER if table in requested)


def validate_normalized_row(
    row: dict[str, Any],
    *,
    campaign_id: str,
) -> None:
    if row.get("artifact") != NORMALIZED_ROW_ARTIFACT:
        raise ValueError("invalid normalized-row artifact")
    if int(row.get("artifact_version", -1)) != NORMALIZED_ROW_VERSION:
        raise ValueError("unsupported normalized-row artifact_version")
    if row.get("campaign_id") != campaign_id:
        raise ValueError("normalized row campaign identity mismatch")
    table_id = row.get("table_id")
    case = row.get("case")
    method_id = row.get("method_id")
    if table_id not in PRIMARY_METHODS or method_id not in PRIMARY_METHODS[table_id]:
        raise ValueError("normalized row method identity mismatch")
    if case not in ICCAD24_CASES:
        raise ValueError("normalized row case identity mismatch")
    if row.get("input_domain") not in {"R0", "D_post"}:
        raise ValueError("normalized row input domain is invalid")
    if not isinstance(row.get("required_cuda"), bool):
        raise ValueError("normalized row required_cuda must be boolean")
    if not row.get("attempt_id"):
        raise ValueError("normalized row is missing attempt_id")
    status = row.get("status")
    if status not in TERMINAL_STATUSES:
        raise ValueError("normalized row is not terminal")
    if status != "pass":
        if not row.get("terminal_reason"):
            raise ValueError("failed normalized row is missing terminal_reason")
        return
    input_metrics = row.get("input_metrics")
    if not isinstance(input_metrics, dict):
        raise ValueError("passing normalized row is missing input_metrics")
    for name in ("wns_ns", "tns_ns"):
        try:
            value = float(input_metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid input metric: {name}") from error
        if not math.isfinite(value):
            raise ValueError(f"invalid input metric: {name}")
    if row.get("terminal_reason") is not None:
        raise ValueError("passing normalized row has a terminal reason")
    final_metrics = row.get("final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError("passing normalized row is missing final_metrics")
    missing_metrics = [
        name for name in REQUIRED_TRACK_A_METRICS if final_metrics.get(name) is None
    ]
    if missing_metrics:
        raise ValueError(
            "passing normalized row is missing Track A metrics: "
            + ", ".join(missing_metrics)
        )
    if int(final_metrics["placement_valid"]) != 1:
        raise ValueError("passing normalized row is physically invalid")
    for name in ("wns_ns", "tns_ns", "power_total_w", "total_cell_area_um2"):
        try:
            value = float(final_metrics[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Track A metric: {name}") from error
        if not math.isfinite(value):
            raise ValueError(f"invalid Track A metric: {name}")
    runtime = row.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("passing normalized row is missing runtime")
    missing_runtime = [name for name in REQUIRED_RUNTIME_FIELDS if runtime.get(name) is None]
    if missing_runtime:
        raise ValueError(
            "passing normalized row is missing runtime fields: "
            + ", ".join(missing_runtime)
        )
    runtime_values = {}
    for name in REQUIRED_RUNTIME_FIELDS:
        try:
            value = float(runtime[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid runtime field: {name}") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid runtime field: {name}")
        runtime_values[name] = value
    component_total = (
        runtime_values["initialization_runtime_sec"]
        + runtime_values["core_optimizer_runtime_sec"]
        + runtime_values["physical_transaction_runtime_sec"]
        + runtime_values["legalization_runtime_sec"]
        + runtime_values["track_a_runtime_sec"]
    )
    if not math.isclose(
        runtime_values["end_to_end_runtime_sec"],
        component_total,
        rel_tol=1.0e-9,
        abs_tol=1.0e-6,
    ):
        raise ValueError("normalized row runtime components do not sum to end-to-end")
    if not row["runtime"].get("runtime_comparable_group"):
        raise ValueError("passing normalized row is missing runtime comparable group")
    if row["required_cuda"] and (
        runtime_values["gpu_peak_allocated_bytes"] <= 0
        or runtime_values["gpu_peak_reserved_bytes"] <= 0
    ):
        raise ValueError("required CUDA row has no GPU memory evidence")
    if not isinstance(row.get("mutation"), dict):
        raise ValueError("passing normalized row is missing mutation audit")
    source = row.get("source_artifact")
    if not isinstance(source, dict) or not source.get("path"):
        raise ValueError("passing normalized row is missing source artifact")
    track_a = row.get("track_a_artifact")
    if not isinstance(track_a, dict) or not track_a.get("path"):
        raise ValueError("passing normalized row is missing Track A artifact")
    if len(str(track_a.get("evaluated_def_sha256") or "")) != 64:
        raise ValueError("passing normalized row has invalid evaluated DEF SHA-256")


def validate_completed_matrix(
    rows: Iterable[dict[str, Any]],
    *,
    cases: Iterable[str] = ICCAD24_CASES,
    tables: Iterable[str] | None = None,
    campaign_id: str | None = None,
) -> None:
    selected_table_ids = _selected_tables(tables)
    expected = [
        row for row in primary_matrix(cases) if row["table_id"] in selected_table_ids
    ]
    expected_keys = {
        (row["table_id"], row["case"], row["method_id"]) for row in expected
    }
    rows_list = list(rows)
    actual_keys = [
        (row.get("table_id"), row.get("case"), row.get("method_id"))
        for row in rows_list
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("completed matrix contains duplicate row keys")
    if set(actual_keys) != expected_keys:
        raise ValueError("completed matrix does not match the required denominator")
    for row in rows_list:
        if campaign_id is not None:
            validate_normalized_row(row, campaign_id=campaign_id)
        status = row.get("status")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"primary row is not terminal: {actual_keys[rows_list.index(row)]}")
        if not row.get("attempt_id"):
            raise ValueError("completed primary row is missing attempt_id")
        if status != "pass" and not row.get("terminal_reason"):
            raise ValueError("failed primary row is missing terminal_reason")
