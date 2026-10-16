"""Machine-readable evidence records for GPUGR qualification gates.

The backend returns runtime metrics, while this module records the boundary
evidence that cannot be inferred from those metrics: command provenance,
process termination, repository identities, and verified output artifacts.
Records are written atomically so an interrupted run cannot leave a plausible
but truncated qualification file.
"""

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

EVIDENCE_SCHEMA_VERSION = 1
VALID_STATUSES = frozenset({"pass", "fail", "incomplete", "blocked", "not_run"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def terminal_status(
    *, returncode: int | None, timed_out: bool = False, terminal_marker: bool = False
) -> str:
    """Classify a child-process outcome without upgrading an incomplete run."""

    if timed_out or returncode is None or returncode < 0 or not terminal_marker:
        return "incomplete"
    if returncode != 0:
        return "fail"
    return "pass"


def _validate_component(value: str, label: str) -> str:
    value = str(value)
    if not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be a single safe path component: {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _artifact_record(name: str, path_value, expected_sha256: str | None = None) -> dict:
    path = Path(path_value).expanduser().resolve()
    record = {
        "name": str(name),
        "path": str(path),
        "exists": path.exists(),
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": None,
        "expected_sha256": expected_sha256,
        "hash_matches": None,
    }
    if path.is_file():
        record["sha256"] = _sha256_file(path)
    elif path.is_dir():
        record["sha256"] = _sha256_directory(path)
    if expected_sha256 is not None:
        record["hash_matches"] = record["sha256"] == str(expected_sha256)
    return record


def _normalize_artifact_paths(artifact_paths) -> dict:
    if artifact_paths is None:
        return {}
    if not isinstance(artifact_paths, Mapping):
        raise TypeError("artifact_paths must be a mapping of names to paths")
    normalized = {}
    for name, value in artifact_paths.items():
        if isinstance(value, Mapping):
            if "path" not in value:
                raise ValueError(f"artifact {name!r} is missing its path")
            normalized[str(name)] = _artifact_record(str(name), value["path"], value.get("sha256"))
        else:
            normalized[str(name)] = _artifact_record(str(name), value)
    return normalized


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    return value


def write_evidence_record(
    evidence_dir,
    record: Mapping,
    *,
    artifact_paths: Mapping | None = None,
) -> Path:
    """Validate and atomically write one gate evidence record.

    A ``pass`` record raises when an artifact is missing or its expected hash
    does not match. Non-pass records retain the artifact validation details so
    a failed or incomplete run remains diagnosable.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    raw_gate = record.get("gate")
    if raw_gate is None:
        raise ValueError("record requires a gate")
    gate = _validate_component(raw_gate, "gate")
    run_id = _validate_component(record.get("run_id", uuid.uuid4().hex), "run_id")
    claim_level = str(record.get("claim_level", "")).strip()
    if not claim_level:
        raise ValueError("record requires a non-empty claim_level")
    status = str(record.get("status", "")).strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"unsupported evidence status {status!r}")

    artifacts = _normalize_artifact_paths(
        artifact_paths if artifact_paths is not None else record.get("artifact_paths")
    )
    artifact_errors = []
    for name, artifact in artifacts.items():
        if not artifact["exists"]:
            artifact_errors.append(f"{name}: missing {artifact['path']}")
        elif artifact["expected_sha256"] is not None and not artifact["hash_matches"]:
            artifact_errors.append(f"{name}: sha256 mismatch for {artifact['path']}")
    if status == "pass" and artifact_errors:
        raise ValueError(
            "cannot write pass evidence with invalid artifacts: " + "; ".join(artifact_errors)
        )

    output = dict(record)
    output.pop("artifact_paths", None)
    output.update(
        {
            "schema_version": int(record.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            "run_id": run_id,
            "gate": gate,
            "claim_level": claim_level,
            "status": status,
            "artifacts": artifacts,
        }
    )
    if artifact_errors:
        output["artifact_validation_errors"] = artifact_errors
    output = _json_safe(output)

    target_dir = Path(evidence_dir).expanduser().resolve() / gate
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{run_id}.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=f".{run_id}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(output, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "VALID_STATUSES",
    "terminal_status",
    "write_evidence_record",
]
