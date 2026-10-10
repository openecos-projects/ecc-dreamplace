"""Validate native ECC timing qualification at initial read and full refresh."""

import numpy as np

TIMING_SCHEMA_VERSION = 2
QUALIFICATION_FIELDS = (
    "endpoints_max_valid",
    "endpoints_max_reason",
    "endpoints_constraint_max_valid",
    "endpoints_constraint_max_reason",
    "endpoints_timing_check_max_valid",
    "endpoints_timing_check_max_reason",
)


def read_qualification_arrays(pydb):
    if not hasattr(pydb, QUALIFICATION_FIELDS[0]):
        return {}
    return {
        name: np.asarray(
            getattr(pydb, name), dtype=np.bool_ if name.endswith("_valid") else np.int32
        ).reshape(-1, 2)
        for name in QUALIFICATION_FIELDS
    }


def validate_timing_schema(pydb):
    version = int(getattr(pydb, "timing_schema_version", 0))
    if version != TIMING_SCHEMA_VERSION:
        raise RuntimeError(
            f"ECC timing requires schema {TIMING_SCHEMA_VERSION}, got {version}; "
            "rebuild/reinstall ecc-tools"
        )
    endpoints = np.asarray(pydb.end_points, dtype=np.int64)
    counts = (
        len(endpoints),
        len(pydb.endpoints_constraint_arcs),
        len(pydb.endpoints_timing_check_arcs),
    )
    for pair, count in zip(zip(QUALIFICATION_FIELDS[::2], QUALIFICATION_FIELDS[1::2]), counts):
        valid_name, reason_name = pair
        for name in pair:
            if not hasattr(pydb, name):
                raise RuntimeError(f"ECC timing schema {version} is missing {name}")
        valid = np.asarray(getattr(pydb, valid_name))
        reasons = np.asarray(getattr(pydb, reason_name))
        if count == 0 and valid.size == reasons.size == 0:
            valid = valid.reshape(0, 2).astype(bool)
            reasons = reasons.reshape(0, 2).astype(np.int32)
        if valid.shape != (count, 2) or reasons.shape != (count, 2):
            raise RuntimeError(
                f"ECC timing qualification rows do not align: {valid_name}/{reason_name}"
            )
        if valid.dtype != np.bool_ or not np.issubdtype(reasons.dtype, np.integer):
            raise RuntimeError(
                f"ECC timing qualification has invalid types: {valid_name}/{reason_name}"
            )
        # Stable schema-v2 reason codes are owned by TimingMaxQualificationReason.
        if np.any((reasons < 0) | (reasons > 6)) or not np.array_equal(valid, reasons == 0):
            raise RuntimeError(
                f"ECC timing qualification has inconsistent reason codes: {reason_name}"
            )
        if np.any(reasons == 6):
            raise RuntimeError(
                "ECC max timing qualification cannot represent a path-specific constraint: "
                f"{reason_name}"
            )
    pin_count = len(pydb.pin_names)
    if (
        endpoints.ndim != 1
        or len(np.unique(endpoints)) != len(endpoints)
        or np.any((endpoints < 0) | (endpoints >= pin_count))
    ):
        raise RuntimeError("ECC timing endpoint identities are invalid")
    endpoint_valid = read_qualification_arrays(pydb)["endpoints_max_valid"]
    by_pin = np.zeros((pin_count, 2), dtype=bool)
    by_pin[endpoints] = endpoint_valid
    for table, field, source_count in (
        (pydb.endpoints_constraint_arcs, "endpoints_constraint_max_valid", len(pydb.clock_pins)),
        (pydb.endpoints_timing_check_arcs, "endpoints_timing_check_max_valid", pin_count),
    ):
        arcs = np.asarray(table, dtype=np.int64)
        if not len(arcs):
            continue
        if (
            arcs.shape != (len(arcs), 8)
            or np.any((arcs[:, 0] < 0) | (arcs[:, 0] >= source_count))
            or np.any((arcs[:, 1] < 0) | (arcs[:, 1] >= pin_count))
        ):
            raise RuntimeError(f"ECC timing check identities/index domain are invalid: {field}")
        valid = np.asarray(getattr(pydb, field), dtype=bool)
        if np.any(valid & ~by_pin[arcs[:, 1]]):
            raise RuntimeError(
                f"ECC timing check qualifies an absent/unconstrained endpoint: {field}"
            )
