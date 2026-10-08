#!/usr/bin/env python3
"""Public artifact adapters for the AutoDMP tools-paper campaign."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


TRACK_A_ARTIFACT = "canonical_def_only_track_a"
INPUT_STATE_ARTIFACT = "canonical_def_only_input_state"
TRACK_A_VERSION = 1
NORMALIZED_ROW_ARTIFACT = "autodmp_tools_paper_normalized_row"
NORMALIZED_ROW_VERSION = 1
SIZING_RESULT_ARTIFACT = "diff_sizing_tools_method_result"
SIZING_RESULT_VERSION = 1
PLACEMENT_RESULT_ARTIFACT = "random_init_placement_buffering_case_result"
PLACEMENT_RESULT_VERSION = 1
EQUAL_BUFFERING_RESULT_ARTIFACT = "equal_spaced_buffering_method_result"
EQUAL_BUFFERING_RESULT_VERSION = 1
CANDIDATE_BUFFERING_RESULT_ARTIFACT = "buffering_inner_loop_regression_manifest"
CANDIDATE_BUFFERING_RESULT_VERSION = 2
FULL_FLOW_RESULT_ARTIFACT = "tools_release_full_flow_method_result"
FULL_FLOW_RESULT_VERSION = 1
_OPENROAD_CORE_RUNTIME_RE = re.compile(
    r"^METRIC\|core_optimizer_runtime_sec\|([-+0-9.eE]+)$"
)
_AUTODMP_OPTIMIZER_RUNTIME_RE = re.compile(
    r"optimizer\s+\S+\s+takes\s+([-+0-9.eE]+)\s+seconds"
)
_GPU_PEAK_ALLOCATED_RE = re.compile(
    r"TOOLS_METRIC\|gpu_peak_allocated_bytes\|(\d+)"
)
_GPU_PEAK_RESERVED_RE = re.compile(
    r"TOOLS_METRIC\|gpu_peak_reserved_bytes\|(\d+)"
)
_INITIALIZATION_RUNTIME_RES = (
    re.compile(r"setting up raw database takes\s+([-+0-9.eE]+)\s+seconds"),
    re.compile(r"setting up placement database takes\s+([-+0-9.eE]+)\s+seconds"),
    re.compile(r"non-linear placement initialization takes\s+([-+0-9.eE]+)\s+seconds"),
)


def read_json_artifact(
    path: Path,
    *,
    artifact: str,
    versions: set[int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != artifact:
        raise ValueError(
            f"{path}: expected artifact {artifact!r}, got {payload.get('artifact')!r}"
        )
    try:
        version = int(payload.get("artifact_version", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid artifact_version") from error
    if version not in versions:
        raise ValueError(f"{path}: unsupported artifact_version {version}")
    return payload


def normalize_track_a_row(
    *,
    track_a_path: Path,
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
    attempt_id: str,
    input_domain: str,
    required_cuda: bool,
    input_metrics: dict[str, Any],
    source_artifact: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    track_a = read_json_artifact(
        track_a_path,
        artifact=TRACK_A_ARTIFACT,
        versions={TRACK_A_VERSION},
    )
    if track_a.get("status") != "pass":
        raise ValueError(f"{track_a_path}: Track A did not pass")
    metrics = dict(track_a.get("metrics") or {})
    mutation = dict(track_a.get("mutation") or {})
    required_metrics = (
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
    )
    missing = [name for name in required_metrics if metrics.get(name) is None]
    if missing:
        raise ValueError(f"{track_a_path}: missing Track A metrics: {', '.join(missing)}")
    return {
        "artifact": NORMALIZED_ROW_ARTIFACT,
        "artifact_version": NORMALIZED_ROW_VERSION,
        "campaign_id": campaign_id,
        "table_id": table_id,
        "case": case,
        "method_id": method_id,
        "attempt_id": attempt_id,
        "input_domain": input_domain,
        "required_cuda": bool(required_cuda),
        "status": "pass",
        "terminal_reason": None,
        "input_metrics": input_metrics,
        "final_metrics": metrics,
        "mutation": mutation,
        "runtime": runtime,
        "source_artifact": source_artifact,
        "track_a_artifact": {
            "path": str(track_a_path),
            "artifact_version": TRACK_A_VERSION,
            "evaluated_def": dict(track_a.get("artifacts") or {}).get(
                "evaluated_def"
            ),
            "evaluated_def_sha256": dict(track_a.get("artifacts") or {}).get(
                "evaluated_def_sha256"
            ),
        },
    }


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: expected a finite number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field}: expected a nonnegative finite number")
    return result


def _sizing_core_runtime(log_path: Path, method_id: str) -> float:
    if not log_path.is_file():
        raise ValueError(f"{log_path}: missing sizing source log")
    patterns = (
        (_OPENROAD_CORE_RUNTIME_RE,)
        if method_id == "or_pure_sizeup"
        else (_AUTODMP_OPTIMIZER_RUNTIME_RE,)
    )
    matches: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                matches.append(_finite_float(match.group(1), field="core runtime"))
    if len(matches) != 1:
        raise ValueError(
            f"{log_path}: expected exactly one core runtime for {method_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _runtime_from_logs(
    log_paths: list[Path],
    *,
    required_cuda: bool,
) -> dict[str, float | int]:
    initialization = 0.0
    allocated: list[int] = []
    reserved: list[int] = []
    for path in log_paths:
        if not path.is_file():
            raise ValueError(f"{path}: missing runtime source log")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for pattern in _INITIALIZATION_RUNTIME_RES:
                match = pattern.search(line)
                if match:
                    initialization += _finite_float(
                        match.group(1), field="initialization runtime"
                    )
            match = _GPU_PEAK_ALLOCATED_RE.search(line)
            if match:
                allocated.append(int(match.group(1)))
            match = _GPU_PEAK_RESERVED_RE.search(line)
            if match:
                reserved.append(int(match.group(1)))
    if required_cuda and (not allocated or not reserved):
        raise ValueError("required CUDA runtime has no process GPU peak evidence")
    gpu_allocated = max(allocated, default=0)
    gpu_reserved = max(reserved, default=0)
    if required_cuda and (gpu_allocated <= 0 or gpu_reserved <= 0):
        raise ValueError("required CUDA runtime reported zero GPU peak memory")
    return {
        "initialization_runtime_sec": initialization,
        "gpu_peak_allocated_bytes": gpu_allocated,
        "gpu_peak_reserved_bytes": gpu_reserved,
    }


def _runtime_payload(
    *,
    table_id: str,
    source_runtime_sec: float,
    core_runtime_sec: float,
    track_a_runtime_sec: float,
    source_logs: list[Path],
    required_cuda: bool,
    legalization_runtime_sec: float = 0.0,
) -> dict[str, Any]:
    metadata = _runtime_from_logs(source_logs, required_cuda=required_cuda)
    initialization = float(metadata["initialization_runtime_sec"])
    legalization = _finite_float(
        legalization_runtime_sec, field="legalization runtime"
    )
    residual = source_runtime_sec - core_runtime_sec - initialization - legalization
    if residual < -0.1:
        raise ValueError("runtime components exceed source process runtime")
    if residual < 0.0:
        initialization = max(
            0.0,
            source_runtime_sec - core_runtime_sec - legalization,
        )
        residual = 0.0
    return {
        "initialization_runtime_sec": initialization,
        "core_optimizer_runtime_sec": core_runtime_sec,
        "physical_transaction_runtime_sec": residual,
        "legalization_runtime_sec": legalization,
        "track_a_runtime_sec": track_a_runtime_sec,
        "end_to_end_runtime_sec": source_runtime_sec + track_a_runtime_sec,
        "runtime_comparable_group": f"{table_id}:source_plus_track_a_v1",
        "gpu_peak_allocated_bytes": int(metadata["gpu_peak_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(metadata["gpu_peak_reserved_bytes"]),
    }


def normalize_sizing_result(
    *,
    result_path: Path,
    campaign_id: str,
    attempt_id: str,
    input_metrics: dict[str, Any],
    required_cuda: bool,
) -> dict[str, Any]:
    result = read_json_artifact(
        result_path,
        artifact=SIZING_RESULT_ARTIFACT,
        versions={SIZING_RESULT_VERSION},
    )
    case = str(result.get("case") or "")
    method_id = str(result.get("method_id") or "")
    if not case or method_id not in {"or_pure_sizeup", "autodmp_diff_sizing"}:
        raise ValueError(f"{result_path}: invalid sizing case/method identity")
    if result.get("status") != "pass" or result.get("failures"):
        raise ValueError(f"{result_path}: sizing source result did not pass")
    source_execution = dict(result.get("source_execution") or {})
    track_a_execution = dict(result.get("track_a_execution") or {})
    if source_execution.get("status") != "ok":
        raise ValueError(f"{result_path}: sizing source execution did not pass")
    if track_a_execution.get("status") != "ok":
        raise ValueError(f"{result_path}: sizing Track A execution did not pass")
    source_runtime = _finite_float(
        source_execution.get("runtime_sec"),
        field="source_execution.runtime_sec",
    )
    track_a_launcher_runtime = _finite_float(
        track_a_execution.get("runtime_sec"),
        field="track_a_execution.runtime_sec",
    )
    source_log = Path(str(source_execution.get("log_path") or ""))
    core_runtime = _sizing_core_runtime(source_log, method_id)
    if core_runtime > source_runtime + 1.0e-9:
        raise ValueError(f"{result_path}: core runtime exceeds source runtime")
    track_a_path = result_path.parent / "track_a" / "track_a_summary.json"
    row = normalize_track_a_row(
        track_a_path=track_a_path,
        campaign_id=campaign_id,
        table_id="sizing",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="D_post",
        required_cuda=required_cuda,
        input_metrics=dict(input_metrics),
        source_artifact={
            "path": str(result_path),
            "artifact": SIZING_RESULT_ARTIFACT,
            "artifact_version": SIZING_RESULT_VERSION,
        },
        runtime=_runtime_payload(
            table_id="sizing",
            source_runtime_sec=source_runtime,
            core_runtime_sec=core_runtime,
            track_a_runtime_sec=track_a_launcher_runtime,
            source_logs=[source_log],
            required_cuda=required_cuda,
        ),
    )
    row["source_artifact"]["source_log_path"] = str(source_log)
    return row


def normalize_placement_result(
    *,
    result_path: Path,
    track_a_path: Path,
    track_a_launcher_runtime_sec: float,
    campaign_id: str,
    attempt_id: str,
    method_id: str,
    required_cuda: bool,
) -> dict[str, Any]:
    result = read_json_artifact(
        result_path,
        artifact=PLACEMENT_RESULT_ARTIFACT,
        versions={PLACEMENT_RESULT_VERSION},
    )
    case = str(result.get("case") or "")
    source_method = {
        "or_gpl_plain": "OR-GPL-plain",
        "or_tdp_reweight_only": "OR-TDP-reweight-only",
        "autodmp_p_ctrl": "P",
        "autodmp_pin2pin": "TDP-pin2pin",
    }.get(method_id)
    if not case or source_method is None:
        raise ValueError(f"{result_path}: invalid placement case/method identity")
    method = dict(dict(result.get("methods") or {}).get(source_method) or {})
    if result.get("status") != "pass" or result.get("failures"):
        raise ValueError(f"{result_path}: placement source case did not pass")
    if method.get("status") != "pass":
        raise ValueError(f"{result_path}: placement source method did not pass")
    execution = dict(method.get("execution") or {})
    if execution.get("status") != "ok":
        raise ValueError(f"{result_path}: placement source execution did not pass")
    source_runtime = _finite_float(
        execution.get("runtime_sec"), field="placement source runtime"
    )
    metrics = dict(method.get("metrics") or {})
    source_logs: list[Path] = []
    if source_method in {"OR-GPL-plain", "OR-TDP-reweight-only"}:
        core_runtime = _finite_float(
            metrics.get("core_runtime_sec"), field="placement core runtime"
        )
        if method.get("r0_coordinate_match") is not True:
            raise ValueError(f"{result_path}: OpenROAD placement did not start from R0")
        actions = dict(method.get("native_timing_driven_actions") or {})
        if any(
            int(actions.get(name, 0) or 0)
            for name in (
                "explicit_resized_instances",
                "explicit_inserted_buffers",
                "repair_design_net_instance_delta",
            )
        ):
            raise ValueError(f"{result_path}: native sizing/buffering action detected")
    else:
        source_log = Path(str(execution.get("log_path") or ""))
        source_logs = [source_log]
        core_runtime = _sizing_core_runtime(source_log, "autodmp_diff_sizing")
        if method_id == "autodmp_pin2pin":
            milestones = dict(method.get("milestones") or {})
            update = dict(milestones.get("legacy_net_weight") or {})
            native = dict(update.get("pin2pin_native_summary") or {})
            if int(update.get("update_count", 0) or 0) <= 0:
                raise ValueError(f"{result_path}: Pin2Pin never updated")
            if update.get("pin2pin_backend") != "cuda":
                raise ValueError(f"{result_path}: Pin2Pin CUDA backend not proven")
            if update.get("pin2pin_path_pair_backend") != (
                "cpp_openmp_transition_aware"
            ):
                raise ValueError(f"{result_path}: Pin2Pin path-pair backend mismatch")
            if native.get("state_domain") != "setup_plus_recovery":
                raise ValueError(f"{result_path}: Pin2Pin state domain mismatch")
            if native.get("backend") != "cpp_openmp_transition_aware":
                raise ValueError(f"{result_path}: native path-pair backend mismatch")
    if core_runtime > source_runtime + 1.0e-9:
        raise ValueError(f"{result_path}: placement core runtime exceeds source runtime")
    track_a = read_json_artifact(
        track_a_path,
        artifact=TRACK_A_ARTIFACT,
        versions={TRACK_A_VERSION},
    )
    mutation = dict(track_a.get("mutation") or {})
    if any(
        int(mutation.get(name, 0) or 0)
        for name in (
            "resize_count",
            "inserted_buffer_count",
            "inserted_other_count",
            "deleted_instance_count",
        )
    ):
        raise ValueError(f"{track_a_path}: placement method changed topology or masters")
    track_a_runtime = _finite_float(
        track_a_launcher_runtime_sec,
        field="placement Track A launcher runtime",
    )
    r0_manifest = dict(dict(result.get("r0") or {}).get("manifest") or {})
    coordinate_identity = dict(r0_manifest.get("coordinate_identity") or {})
    r0_hash = coordinate_identity.get("movable_nodes_xy_sha256")
    if not r0_hash:
        raise ValueError(f"{result_path}: missing R0 coordinate identity")
    row = normalize_track_a_row(
        track_a_path=track_a_path,
        campaign_id=campaign_id,
        table_id="placement",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="R0",
        required_cuda=required_cuda,
        input_metrics={},
        source_artifact={
            "path": str(result_path),
            "artifact": PLACEMENT_RESULT_ARTIFACT,
            "artifact_version": PLACEMENT_RESULT_VERSION,
            "source_method": source_method,
            "raw_def": method.get("raw_def"),
            "raw_def_sha256": method.get("raw_def_sha256"),
            "r0_def": (
                dict(result.get("r0") or {}).get("manifest", {}).get("r0_def")
                or dict(result.get("r0") or {}).get("manifest", {}).get(
                    "output_def"
                )
            ),
            "r0_def_sha256": dict(result.get("r0") or {})
            .get("manifest", {})
            .get("r0_def_sha256"),
            "r0_movable_xy_sha256": r0_hash,
        },
        runtime=_runtime_payload(
            table_id="placement",
            source_runtime_sec=source_runtime,
            core_runtime_sec=core_runtime,
            track_a_runtime_sec=track_a_runtime,
            source_logs=source_logs,
            required_cuda=required_cuda,
        ),
    )
    return row


def _route_b_core_runtime(summary: dict[str, Any]) -> float:
    inner = dict(summary.get("inner_loop") or {})
    trace = list(inner.get("metrics_trace") or ())
    if not trace:
        raise ValueError("Route-B summary has no inner-loop metrics trace")
    total_ms = 0.0
    for index, raw in enumerate(trace):
        record = dict(raw or {})
        if record.get("step_wall_ms") is not None:
            total_ms += _finite_float(
                record["step_wall_ms"], field=f"Route-B step {index} wall time"
            )
            continue
        if record.get("objective_step_wall_ms") is not None:
            total_ms += _finite_float(
                record["objective_step_wall_ms"],
                field=f"Route-B step {index} objective wall time",
            )
            total_ms += _finite_float(
                record.get("projection_refresh_wall_ms", 0.0),
                field=f"Route-B step {index} projection wall time",
            )
            continue
        for name in (
            "objective_forward_ms",
            "objective_grad_ms",
            "scheduler_wall_ms",
        ):
            total_ms += _finite_float(
                record.get(name), field=f"Route-B step {index} {name}"
            )
    total_ms += _finite_float(
        inner.get("final_projection_wall_ms", 0.0),
        field="Route-B final projection wall time",
    )
    total_ms += _finite_float(
        dict(inner.get("terminal_metrics") or {}).get("terminal_forward_ms", 0.0),
        field="Route-B terminal forward wall time",
    )
    return total_ms / 1000.0


def _validate_buffer_track_a(
    track_a_path: Path,
    *,
    expected_buffer_count: int,
) -> None:
    track_a = read_json_artifact(
        track_a_path,
        artifact=TRACK_A_ARTIFACT,
        versions={TRACK_A_VERSION},
    )
    mutation = dict(track_a.get("mutation") or {})
    if int(mutation.get("inserted_buffer_count", -1)) != int(expected_buffer_count):
        raise ValueError(f"{track_a_path}: committed buffer/action count mismatch")
    if any(
        int(mutation.get(name, 0) or 0)
        for name in (
            "resize_count",
            "inserted_other_count",
            "deleted_instance_count",
        )
    ):
        raise ValueError(f"{track_a_path}: buffering method changed masters or topology")


def normalize_equal_spaced_buffering_result(
    *,
    result_path: Path,
    track_a_path: Path,
    track_a_launcher_runtime_sec: float,
    campaign_id: str,
    attempt_id: str,
    method_id: str,
    input_metrics: dict[str, Any],
    required_cuda: bool,
) -> dict[str, Any]:
    payload = read_json_artifact(
        result_path,
        artifact=EQUAL_BUFFERING_RESULT_ARTIFACT,
        versions={EQUAL_BUFFERING_RESULT_VERSION},
    )
    result = dict(payload.get("result") or {})
    case = str(result.get("case") or "")
    expected_source_method = {
        "or_repair_design_buffer_only": "b1_rd_rt",
        "autodmp_equal_spaced_route_b": "ours",
    }.get(method_id)
    if not case or result.get("method") != expected_source_method:
        raise ValueError(f"{result_path}: equal-spaced method identity mismatch")
    if result.get("status") != "pass" or result.get("failure"):
        raise ValueError(f"{result_path}: buffering source result did not pass")
    execution = dict(payload.get("execution") or {})
    if execution.get("status") != "ok":
        raise ValueError(f"{result_path}: buffering source execution did not pass")
    source_runtime = _finite_float(
        execution.get("runtime_sec"), field="buffering source runtime"
    )
    audit = dict(payload.get("audit") or payload.get("commit_audit") or {})
    expected_buffer_count = int(audit.get("added_buffer_count", -1))
    if audit.get("status") != "pass" or expected_buffer_count <= 0:
        raise ValueError(f"{result_path}: buffering action audit did not pass")
    if method_id == "or_repair_design_buffer_only":
        metrics = dict(payload.get("metrics") or {})
        core_runtime = _finite_float(
            metrics.get("repair_runtime_sec"), field="OpenROAD buffer core runtime"
        )
    else:
        summary = dict(payload.get("summary") or {})
        if summary.get("strategy") != "discrete_net_gradient":
            raise ValueError(f"{result_path}: equal-spaced Route-B strategy mismatch")
        if int(dict(summary.get("inner_loop") or {}).get("iterations", 0)) != 5:
            raise ValueError(f"{result_path}: equal-spaced Route-B round count mismatch")
        commit = dict(summary.get("commit") or {})
        if int(commit.get("accepted_action_count", -1)) != expected_buffer_count:
            raise ValueError(f"{result_path}: Route-B action/DEF count mismatch")
        core_runtime = _route_b_core_runtime(summary)
    if core_runtime > source_runtime + 1.0e-9:
        raise ValueError(f"{result_path}: buffering core runtime exceeds source runtime")
    _validate_buffer_track_a(
        track_a_path,
        expected_buffer_count=expected_buffer_count,
    )
    track_a_runtime = _finite_float(
        track_a_launcher_runtime_sec, field="buffering Track A launcher runtime"
    )
    source_log = Path(
        str(result.get("run_log") or execution.get("log_path") or "")
    )
    return normalize_track_a_row(
        track_a_path=track_a_path,
        campaign_id=campaign_id,
        table_id="buffering",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="D_post",
        required_cuda=required_cuda,
        input_metrics=dict(input_metrics),
        source_artifact={
            "path": str(result_path),
            "artifact": EQUAL_BUFFERING_RESULT_ARTIFACT,
            "artifact_version": EQUAL_BUFFERING_RESULT_VERSION,
            "source_method": expected_source_method,
            "committed_buffer_count": expected_buffer_count,
            "committed_def": result.get("final_def"),
            "committed_def_sha256": result.get("final_def_sha256"),
        },
        runtime=_runtime_payload(
            table_id="buffering",
            source_runtime_sec=source_runtime,
            core_runtime_sec=core_runtime,
            track_a_runtime_sec=track_a_runtime,
            source_logs=[source_log],
            required_cuda=required_cuda,
        ),
    )


def normalize_candidate_buffering_result(
    *,
    manifest_path: Path,
    track_a_path: Path,
    track_a_launcher_runtime_sec: float,
    campaign_id: str,
    attempt_id: str,
    case: str,
    input_metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = read_json_artifact(
        manifest_path,
        artifact=CANDIDATE_BUFFERING_RESULT_ARTIFACT,
        versions={CANDIDATE_BUFFERING_RESULT_VERSION},
    )
    case_result = dict(dict(payload.get("case_results") or {}).get(case) or {})
    row = dict(case_result.get("row") or {})
    if row.get("case") != case or row.get("status") != "pass":
        raise ValueError(f"{manifest_path}: candidate Route-B source did not pass")
    if row.get("buffering_mode") != "candidate":
        raise ValueError(f"{manifest_path}: candidate buffering mode mismatch")
    if row.get("candidate_strategy") != "discrete_net_gradient":
        raise ValueError(f"{manifest_path}: candidate Route-B strategy mismatch")
    if int(row.get("continuous_steps", 0)) != 5:
        raise ValueError(f"{manifest_path}: candidate Route-B round count mismatch")
    if row.get("trace_audit_status") != "pass":
        raise ValueError(f"{manifest_path}: candidate trace audit did not pass")
    expected_buffer_count = int(row.get("commit_accepted_action_count", -1))
    if expected_buffer_count <= 0:
        raise ValueError(f"{manifest_path}: candidate committed no buffers")
    summary_path = Path(str(row.get("summary_path") or ""))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = dict(summary_payload.get("summary") or {})
    core_runtime = _route_b_core_runtime(summary)
    source_runtime = _finite_float(
        row.get("autodmp_runtime_sec"), field="candidate source runtime"
    )
    if core_runtime > source_runtime + 1.0e-9:
        raise ValueError(f"{manifest_path}: candidate core runtime exceeds source runtime")
    _validate_buffer_track_a(
        track_a_path,
        expected_buffer_count=expected_buffer_count,
    )
    track_a_runtime = _finite_float(
        track_a_launcher_runtime_sec, field="candidate Track A launcher runtime"
    )
    return normalize_track_a_row(
        track_a_path=track_a_path,
        campaign_id=campaign_id,
        table_id="buffering",
        case=case,
        method_id="autodmp_candidate_route_b",
        attempt_id=attempt_id,
        input_domain="D_post",
        required_cuda=True,
        input_metrics=dict(input_metrics),
        source_artifact={
            "path": str(manifest_path),
            "artifact": CANDIDATE_BUFFERING_RESULT_ARTIFACT,
            "artifact_version": CANDIDATE_BUFFERING_RESULT_VERSION,
            "trace_audit_path": str(
                manifest_path.parent / case / "candidate_route_b_trace_audit.json"
            ),
            "committed_buffer_count": expected_buffer_count,
        },
        runtime=_runtime_payload(
            table_id="buffering",
            source_runtime_sec=source_runtime,
            core_runtime_sec=core_runtime,
            track_a_runtime_sec=track_a_runtime,
            source_logs=[Path(str(row.get("log_path") or ""))],
            required_cuda=True,
        ),
    )


def _pin2pin_source_evidence(
    case_result: dict[str, Any],
    *,
    source_method: str,
    source_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = dict(dict(case_result.get("methods") or {}).get(source_method) or {})
    if case_result.get("status") != "pass" or case_result.get("failures"):
        raise ValueError(f"{source_path}: placement source case did not pass")
    if method.get("status") != "pass":
        raise ValueError(f"{source_path}: placement source method did not pass")
    milestones = dict(method.get("milestones") or {})
    update = dict(milestones.get("legacy_net_weight") or {})
    native = dict(update.get("pin2pin_native_summary") or {})
    if int(update.get("update_count", 0) or 0) <= 0:
        raise ValueError(f"{source_path}: Pin2Pin never updated")
    if update.get("pin2pin_backend") != "cuda":
        raise ValueError(f"{source_path}: Pin2Pin CUDA backend not proven")
    if update.get("pin2pin_path_pair_backend") != "cpp_openmp_transition_aware":
        raise ValueError(f"{source_path}: Pin2Pin path-pair backend mismatch")
    if native.get("state_domain") != "setup_plus_recovery":
        raise ValueError(f"{source_path}: Pin2Pin state domain mismatch")
    return method, update


def normalize_full_flow_result(
    *,
    result_path: Path,
    track_a_path: Path,
    track_a_launcher_runtime_sec: float,
    campaign_id: str,
    attempt_id: str,
    method_id: str,
    required_cuda: bool,
) -> dict[str, Any]:
    if method_id not in {
        "or_native_full_flow",
        "autodmp_staged_psb",
        "autodmp_coordinated_psb",
    }:
        raise ValueError(f"unsupported full-flow method: {method_id}")

    source_details: dict[str, Any] = {}
    source_logs: list[Path] = []
    legalization_runtime = 0.0
    if method_id == "autodmp_staged_psb":
        result = read_json_artifact(
            result_path,
            artifact=FULL_FLOW_RESULT_ARTIFACT,
            versions={FULL_FLOW_RESULT_VERSION},
        )
        case = str(result.get("case") or "")
        if result.get("method_id") != method_id:
            raise ValueError(f"{result_path}: staged full-flow method mismatch")
        if result.get("status") != "pass" or result.get("failures"):
            raise ValueError(f"{result_path}: staged full flow did not pass")
        stages = dict(result.get("stages") or {})
        required_stages = (
            "pin2pin_placement",
            "placement_checkpoint_legalization",
            "fixed_position_sizing",
            "sizing_checkpoint_legalization",
            "equal_spaced_route_b",
        )
        if tuple(result.get("stage_order") or ()) != required_stages:
            raise ValueError(f"{result_path}: staged full-flow stage order mismatch")
        if any(dict(stages.get(name) or {}).get("status") != "ok" for name in required_stages):
            raise ValueError(f"{result_path}: staged full-flow stage did not pass")
        source_runtime = sum(
            _finite_float(
                dict(stages[name]).get("runtime_sec"),
                field=f"staged {name} runtime",
            )
            for name in required_stages
        )
        source_artifacts = dict(result.get("source_artifacts") or {})
        placement_summary_path = Path(
            str(source_artifacts.get("placement_case_summary") or "")
        )
        placement_summary = read_json_artifact(
            placement_summary_path,
            artifact=PLACEMENT_RESULT_ARTIFACT,
            versions={PLACEMENT_RESULT_VERSION},
        )
        placement_method, pin2pin_update = _pin2pin_source_evidence(
            placement_summary,
            source_method="TDP-pin2pin",
            source_path=placement_summary_path,
        )
        placement_log = Path(
            str(dict(placement_method.get("execution") or {}).get("log_path") or "")
        )
        placement_core = _sizing_core_runtime(
            placement_log,
            "autodmp_diff_sizing",
        )
        sizing_log = Path(
            str(dict(stages["fixed_position_sizing"]).get("log_path") or "")
        )
        sizing_core = _sizing_core_runtime(sizing_log, "autodmp_diff_sizing")
        buffering_result_path = Path(
            str(source_artifacts.get("buffering_result") or "")
        )
        buffering_payload = read_json_artifact(
            buffering_result_path,
            artifact=EQUAL_BUFFERING_RESULT_ARTIFACT,
            versions={EQUAL_BUFFERING_RESULT_VERSION},
        )
        buffering_result = dict(buffering_payload.get("result") or {})
        if buffering_result.get("status") != "pass":
            raise ValueError(f"{buffering_result_path}: staged buffering did not pass")
        buffering_core = _route_b_core_runtime(
            dict(buffering_payload.get("summary") or {})
        )
        buffering_log = Path(
            str(
                buffering_result.get("run_log")
                or dict(buffering_payload.get("execution") or {}).get("log_path")
                or ""
            )
        )
        source_logs = [placement_log, sizing_log, buffering_log]
        legalization_runtime = sum(
            _finite_float(
                dict(stages[name]).get("runtime_sec"),
                field=f"staged {name} legalization runtime",
            )
            for name in (
                "placement_checkpoint_legalization",
                "sizing_checkpoint_legalization",
            )
        )
        core_runtime = placement_core + sizing_core + buffering_core
        r0_manifest = dict(dict(placement_summary.get("r0") or {}).get("manifest") or {})
        r0_hash = dict(r0_manifest.get("coordinate_identity") or {}).get(
            "movable_nodes_xy_sha256"
        )
        source_details = {
            "stage_order": list(required_stages),
            "pin2pin_update": pin2pin_update,
            "r0_movable_xy_sha256": r0_hash,
            "r0_def": r0_manifest.get("r0_def") or result.get("r0_def"),
            "r0_def_sha256": r0_manifest.get("r0_def_sha256")
            or result.get("r0_def_sha256"),
            "raw_def": result.get("raw_def"),
            "raw_def_sha256": result.get("raw_def_sha256"),
        }
    else:
        case_result = read_json_artifact(
            result_path,
            artifact=PLACEMENT_RESULT_ARTIFACT,
            versions={PLACEMENT_RESULT_VERSION},
        )
        case = str(case_result.get("case") or "")
        source_method = (
            "OR-native-full-flow"
            if method_id == "or_native_full_flow"
            else "TDP-pin2pin-SB"
        )
        method = dict(dict(case_result.get("methods") or {}).get(source_method) or {})
        if case_result.get("status") != "pass" or case_result.get("failures"):
            raise ValueError(f"{result_path}: full-flow source case did not pass")
        if method.get("status") != "pass":
            raise ValueError(f"{result_path}: full-flow source method did not pass")
        execution = dict(method.get("execution") or {})
        if execution.get("status") != "ok":
            raise ValueError(f"{result_path}: full-flow source execution did not pass")
        source_runtime = _finite_float(
            execution.get("runtime_sec"), field="full-flow source runtime"
        )
        source_log = Path(str(execution.get("log_path") or ""))
        if method_id == "or_native_full_flow":
            core_runtime = _finite_float(
                dict(method.get("metrics") or {}).get("core_runtime_sec"),
                field="OpenROAD full-flow core runtime",
            )
            native_actions = dict(method.get("native_timing_driven_actions") or {})
            if native_actions.get("status") != "ok":
                raise ValueError(f"{result_path}: OpenROAD native action audit failed")
            source_details["native_timing_driven_actions"] = native_actions
        else:
            method, pin2pin_update = _pin2pin_source_evidence(
                case_result,
                source_method=source_method,
                source_path=result_path,
            )
            mechanism = dict(method.get("mechanism_audit") or {})
            if mechanism.get("status") != "pass" or mechanism.get("failures"):
                raise ValueError(f"{result_path}: coordinated mechanism audit failed")
            physical = dict(method.get("physical_commit") or {})
            if physical.get("status") != "ok":
                raise ValueError(f"{result_path}: coordinated physical commit failed")
            expected_action_order = [
                "coordinate_sync",
                "apply_sizing",
                "insert_buffers",
                "legalize",
                "check_placement",
                "estimate_parasitics",
                "timing_refresh",
            ]
            if physical.get("full_action_order") != expected_action_order:
                raise ValueError(
                    f"{result_path}: coordinated physical action order mismatch"
                )
            sizing_readback = dict(
                physical.get("sizing_master_verification") or {}
            )
            if (
                sizing_readback.get("status") != "ok"
                or int(sizing_readback.get("missing_count", -1)) != 0
                or int(sizing_readback.get("mismatch_count", -1)) != 0
                or int(sizing_readback.get("matched_count", -1))
                != int(sizing_readback.get("expected_count", -2))
            ):
                raise ValueError(f"{result_path}: coordinated sizing readback failed")
            if int(physical.get("accepted_buffer_count", -1)) != int(
                physical.get("buffer_count", -2)
            ):
                raise ValueError(f"{result_path}: coordinated buffer readback failed")
            for field in (
                "legalization_status",
                "parasitic_refresh_status",
                "def_only_opensta_verification_status",
            ):
                if physical.get(field) != "ok":
                    raise ValueError(
                        f"{result_path}: coordinated {field} audit failed"
                    )
            core_runtime = _sizing_core_runtime(
                source_log,
                "autodmp_diff_sizing",
            )
            source_logs = [source_log]
            source_details.update(
                {
                    "pin2pin_update": pin2pin_update,
                    "milestones": dict(method.get("milestones") or {}),
                    "physical_commit": physical,
                }
            )
        r0_manifest = dict(dict(case_result.get("r0") or {}).get("manifest") or {})
        r0_hash = dict(r0_manifest.get("coordinate_identity") or {}).get(
            "movable_nodes_xy_sha256"
        )
        source_details.update(
            {
                "source_method": source_method,
                "r0_movable_xy_sha256": r0_hash,
                "r0_def": r0_manifest.get("r0_def"),
                "r0_def_sha256": r0_manifest.get("r0_def_sha256"),
                "raw_def": method.get("raw_def"),
                "raw_def_sha256": method.get("raw_def_sha256"),
            }
        )

    if not case:
        raise ValueError(f"{result_path}: missing full-flow case identity")
    if core_runtime > source_runtime + 1.0e-9:
        raise ValueError(f"{result_path}: full-flow core runtime exceeds source runtime")
    if not source_details.get("r0_movable_xy_sha256"):
        raise ValueError(f"{result_path}: missing R0 coordinate identity")
    track_a_runtime = _finite_float(
        track_a_launcher_runtime_sec,
        field="full-flow Track A launcher runtime",
    )
    return normalize_track_a_row(
        track_a_path=track_a_path,
        campaign_id=campaign_id,
        table_id="full_flow",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="R0",
        required_cuda=required_cuda,
        input_metrics={},
        source_artifact={
            "path": str(result_path),
            "artifact": (
                FULL_FLOW_RESULT_ARTIFACT
                if method_id == "autodmp_staged_psb"
                else PLACEMENT_RESULT_ARTIFACT
            ),
            "artifact_version": 1,
            **source_details,
        },
        runtime=_runtime_payload(
            table_id="full_flow",
            source_runtime_sec=source_runtime,
            core_runtime_sec=core_runtime,
            track_a_runtime_sec=track_a_runtime,
            source_logs=source_logs,
            required_cuda=required_cuda,
            legalization_runtime_sec=legalization_runtime,
        ),
    )


def terminal_failure_row(
    *,
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
    attempt_id: str,
    input_domain: str,
    required_cuda: bool,
    status: str,
    terminal_reason: str,
    source_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status in {"pass", "planned", "not_run", "plan_only"}:
        raise ValueError(f"invalid failure status: {status}")
    if not terminal_reason:
        raise ValueError("terminal failure row requires a reason")
    return {
        "artifact": NORMALIZED_ROW_ARTIFACT,
        "artifact_version": NORMALIZED_ROW_VERSION,
        "campaign_id": campaign_id,
        "table_id": table_id,
        "case": case,
        "method_id": method_id,
        "attempt_id": attempt_id,
        "input_domain": input_domain,
        "required_cuda": bool(required_cuda),
        "status": status,
        "terminal_reason": terminal_reason,
        "input_metrics": {},
        "final_metrics": {},
        "mutation": {},
        "runtime": {},
        "source_artifact": source_artifact or {},
        "track_a_artifact": {},
    }
