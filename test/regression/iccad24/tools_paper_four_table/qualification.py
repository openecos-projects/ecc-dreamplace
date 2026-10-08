#!/usr/bin/env python3
"""Artifact sealing and explicit historical-row qualification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import campaign_contract as contract


REUSE_INDEX_ARTIFACT = "autodmp_tools_paper_reuse_index"
REUSE_INDEX_VERSION = 1
CAMPAIGN_MANIFEST_ARTIFACT = "autodmp_tools_paper_campaign_manifest"
ATTEMPT_MANIFEST_ARTIFACT = "autodmp_tools_paper_attempt_manifest"
ATTEMPT_QUALIFICATION_ARTIFACT = "autodmp_tools_paper_attempt_qualification"
SOURCE_IDENTITY_FIELDS = (
    "autodmp_revision",
    "autodmp_tracked_diff_sha256",
    "source_file_sha256",
    "native_library_sha256",
    "openroad_sha256",
    "openroad_version",
    "python_path",
    "python_version",
    "torch_version",
    "cuda_runtime_version",
    "gpu_model",
    "gpu_uuid",
    "gpu_preflight_compute_processes",
    "cpu_model",
    "omp_num_threads",
    "torch_num_threads",
    "torch_num_interop_threads",
    "openroad_num_threads",
    "measurement_policy",
    "benchmark_root",
)
INDEX_PATH_FIELDS = (
    "normalized_row",
    "campaign_manifest",
    "attempt_manifest",
    "qualification",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def seal_passing_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "pass":
        return copy.deepcopy(row)
    sealed = copy.deepcopy(row)
    source = dict(sealed.get("source_artifact") or {})
    source_path = Path(str(source.get("path") or ""))
    if not source_path.is_file():
        raise ValueError(f"missing source artifact: {source_path}")
    source["source_artifact_sha256"] = sha256(source_path)
    raw_def_value = source.get("raw_def")
    if raw_def_value:
        raw_def = Path(str(raw_def_value))
        if not raw_def.is_file():
            raise ValueError(f"missing source raw DEF: {raw_def}")
        raw_sha = sha256(raw_def)
        if raw_sha != source.get("raw_def_sha256"):
            raise ValueError("source raw DEF SHA-256 mismatch")
    r0_def_value = source.get("r0_def")
    if r0_def_value:
        r0_def = Path(str(r0_def_value))
        if not r0_def.is_file():
            raise ValueError(f"missing source R0 DEF: {r0_def}")
        r0_sha = sha256(r0_def)
        if r0_sha != source.get("r0_def_sha256"):
            raise ValueError("source R0 DEF SHA-256 mismatch")
    sealed["source_artifact"] = source

    track_a = dict(sealed.get("track_a_artifact") or {})
    summary_path = Path(str(track_a.get("path") or ""))
    if not summary_path.is_file():
        raise ValueError(f"missing Track A artifact: {summary_path}")
    track_a["summary_sha256"] = sha256(summary_path)
    evaluated_def = Path(str(track_a.get("evaluated_def") or ""))
    if not evaluated_def.is_file():
        raise ValueError(f"missing Track A evaluated DEF: {evaluated_def}")
    if sha256(evaluated_def) != track_a.get("evaluated_def_sha256"):
        raise ValueError("Track A evaluated DEF SHA-256 mismatch")
    sealed["track_a_artifact"] = track_a
    return sealed


def validate_sealed_passing_row(row: dict[str, Any], *, campaign_id: str) -> None:
    contract.validate_normalized_row(row, campaign_id=campaign_id)
    if row.get("status") != "pass":
        return
    source = dict(row["source_artifact"])
    source_path = Path(str(source["path"]))
    if not source_path.is_file() or sha256(source_path) != source.get(
        "source_artifact_sha256"
    ):
        raise ValueError("source artifact hash validation failed")
    raw_def_value = source.get("raw_def")
    if raw_def_value:
        raw_def = Path(str(raw_def_value))
        if not raw_def.is_file() or sha256(raw_def) != source.get("raw_def_sha256"):
            raise ValueError("source raw DEF hash validation failed")
    r0_def_value = source.get("r0_def")
    if r0_def_value:
        r0_def = Path(str(r0_def_value))
        if not r0_def.is_file() or sha256(r0_def) != source.get(
            "r0_def_sha256"
        ):
            raise ValueError("source R0 DEF hash validation failed")
    track_a = dict(row["track_a_artifact"])
    summary_path = Path(str(track_a["path"]))
    if not summary_path.is_file() or sha256(summary_path) != track_a.get(
        "summary_sha256"
    ):
        raise ValueError("Track A summary hash validation failed")
    evaluated_def = Path(str(track_a["evaluated_def"]))
    if not evaluated_def.is_file() or sha256(evaluated_def) != track_a.get(
        "evaluated_def_sha256"
    ):
        raise ValueError("Track A evaluated DEF hash validation failed")


def load_reuse_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("artifact") != REUSE_INDEX_ARTIFACT:
        raise ValueError("invalid reuse-index artifact")
    if int(payload.get("artifact_version", -1)) != REUSE_INDEX_VERSION:
        raise ValueError("unsupported reuse-index artifact_version")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("reuse-index entries must be a list")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("reuse-index entry must be an object")
        entry = dict(raw)
        key = (
            str(entry.get("table_id") or ""),
            str(entry.get("case") or ""),
            str(entry.get("method_id") or ""),
        )
        if key in result:
            raise ValueError(f"duplicate reuse-index row: {key}")
        table_id, case, method_id = key
        if (
            table_id not in contract.PRIMARY_METHODS
            or case not in contract.ICCAD24_CASES
            or method_id not in contract.PRIMARY_METHODS[table_id]
        ):
            raise ValueError(f"invalid reuse-index row identity: {key}")
        for field in INDEX_PATH_FIELDS:
            value = entry.get(field)
            expected_hash = entry.get(f"{field}_sha256")
            if not value or len(str(expected_hash or "")) != 64:
                raise ValueError(f"reuse-index entry is missing {field} or its SHA-256")
        result[key] = entry
    return result


def qualify_reuse_entry(
    entry: dict[str, Any],
    *,
    current_profile: dict[str, Any],
    current_execution: dict[str, Any],
    expected_key: tuple[str, str, str],
    current_dpost_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {field: Path(str(entry[field])).resolve() for field in INDEX_PATH_FIELDS}
    failures = []
    for field, path in paths.items():
        expected_hash = str(entry[f"{field}_sha256"])
        if not path.is_file():
            failures.append(f"missing_{field}")
        elif sha256(path) != expected_hash:
            failures.append(f"{field}_sha256_mismatch")
    if failures:
        raise ValueError(",".join(failures))

    row = _read_json(paths["normalized_row"])
    prior_manifest = _read_json(paths["campaign_manifest"])
    attempt = _read_json(paths["attempt_manifest"])
    prior_qualification = _read_json(paths["qualification"])
    prior_campaign_id = str(prior_manifest.get("campaign_id") or "")
    contract.validate_normalized_row(row, campaign_id=prior_campaign_id)
    validate_sealed_passing_row(row, campaign_id=prior_campaign_id)
    if (
        row.get("status") != "pass"
        or (row.get("table_id"), row.get("case"), row.get("method_id"))
        != expected_key
    ):
        failures.append("source_row_not_passing_or_wrong_identity")
    if prior_manifest.get("artifact") != CAMPAIGN_MANIFEST_ARTIFACT:
        failures.append("campaign_manifest_artifact_mismatch")
    current_release_digest = contract.digest_payload(current_profile)
    if prior_manifest.get("release_profile_digest") != current_release_digest:
        failures.append("release_profile_digest_mismatch")
    prior_execution = dict(prior_manifest.get("execution_manifest") or {})
    for field in SOURCE_IDENTITY_FIELDS:
        if prior_execution.get(field) != current_execution.get(field):
            failures.append(f"execution_identity_mismatch:{field}")
    if attempt.get("artifact") != ATTEMPT_MANIFEST_ARTIFACT:
        failures.append("attempt_manifest_artifact_mismatch")
    if (
        attempt.get("campaign_id") != prior_campaign_id
        or attempt.get("attempt_id") != row.get("attempt_id")
        or attempt.get("status") != "pass"
    ):
        failures.append("attempt_manifest_identity_or_status_mismatch")
    if prior_qualification.get("artifact") != ATTEMPT_QUALIFICATION_ARTIFACT:
        failures.append("qualification_artifact_mismatch")
    if (
        prior_qualification.get("attempt_id") != row.get("attempt_id")
        or prior_qualification.get("status") != "pass"
        or prior_qualification.get("terminal_status") != "pass"
    ):
        failures.append("qualification_identity_or_status_mismatch")
    table_id, _, method_id = expected_key
    spec = dict(current_profile["method_specs"][method_id])
    if row.get("input_domain") != spec.get("input_domain"):
        failures.append("input_domain_mismatch")
    if bool(row.get("required_cuda")) != bool(spec.get("required_cuda")):
        failures.append("cuda_requirement_mismatch")
    if row.get("input_domain") == "R0":
        source = dict(row.get("source_artifact") or {})
        r0_hash = source.get("r0_movable_xy_sha256")
        r0_def = Path(str(source.get("r0_def") or ""))
        r0_def_sha256 = str(source.get("r0_def_sha256") or "")
        if len(str(r0_hash or "")) != 64:
            failures.append("missing_r0_coordinate_identity")
        if (
            len(r0_def_sha256) != 64
            or not r0_def.is_file()
            or sha256(r0_def) != r0_def_sha256
        ):
            failures.append("missing_or_invalid_r0_def_identity")
    elif current_dpost_metrics is None:
        failures.append("missing_current_dpost_metrics")
    elif contract.digest_payload(row.get("input_metrics") or {}) != contract.digest_payload(
        current_dpost_metrics
    ):
        failures.append("dpost_input_metrics_mismatch")
    if failures:
        raise ValueError(",".join(failures))

    report = {
        "artifact": "autodmp_tools_paper_reuse_qualification",
        "artifact_version": 1,
        "status": "pass",
        "table_id": table_id,
        "case": expected_key[1],
        "method_id": method_id,
        "reused_from_campaign": prior_campaign_id,
        "original_attempt_id": row["attempt_id"],
        "source_files": {field: str(path) for field, path in paths.items()},
        "source_file_sha256": {
            field: str(entry[f"{field}_sha256"]) for field in INDEX_PATH_FIELDS
        },
        "source_identity_fields": list(SOURCE_IDENTITY_FIELDS),
    }
    return row, report
