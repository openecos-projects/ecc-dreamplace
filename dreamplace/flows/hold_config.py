"""Validate the small public contract shared by every hold-capable flow."""

import math
from collections.abc import Mapping, Sequence


def _nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _hold_coefficients(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object with wns and tns")
    unknown = set(value) - {"wns", "tns"}
    if unknown:
        raise ValueError(f"{name} only allows wns and tns")
    return {
        field: _nonnegative_number(value.get(field, 0.0), f"{name}.{field}")
        for field in ("wns", "tns")
    }


def _phase_hold_coefficients(value, name):
    if value is None:
        return {"hold_wns": 0.0, "hold_tns": 0.0}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {
        field: _nonnegative_number(value.get(field, 0.0), f"{name}.{field}")
        for field in ("hold_wns", "hold_tns")
    }


def _is_placement_flow(flow_kind):
    value = getattr(flow_kind, "value", flow_kind)
    return value == "placement"


def validate_hold_configuration(params, flow_kind):
    """Reject unsupported hold combinations before database construction.

    This intentionally validates only the public configuration contract. Schema
    and Liberty identity checks remain at the producer/consumer boundary where
    the actual database is available.
    """
    enabled = getattr(params, "timing_hold_enabled", False)
    if enabled not in (False, True, 0, 1):
        raise ValueError("timing_hold_enabled must be a boolean")
    enabled = bool(enabled)
    margin = _nonnegative_number(
        getattr(params, "timing_hold_margin_ns", 0.0), "timing_hold_margin_ns"
    )
    coefficients = _hold_coefficients(
        getattr(params, "timing_hold_coefficients", {"wns": 0.0, "tns": 0.0}),
        "timing_hold_coefficients",
    )
    phase_values = []
    for name in ("timing_opt_coefficients", "diff_sizing_coefficients"):
        policy = getattr(params, name, None)
        if policy is not None:
            phase_values.append(_phase_hold_coefficients(policy, name))
    active = any(coefficients.values()) or any(
        value for phase in phase_values for value in phase.values()
    )
    if active and not enabled:
        raise ValueError("nonzero hold coefficients require timing_hold_enabled=true")
    if enabled and not bool(getattr(params, "with_sta", False)):
        raise ValueError("timing_hold_enabled requires with_sta=1")
    paths = getattr(params, "timing_hold_lib_paths", [])
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise ValueError("timing_hold_lib_paths must be a list of Liberty paths")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("timing_hold_lib_paths must contain non-empty strings")
    if paths and not enabled:
        raise ValueError("timing_hold_lib_paths requires timing_hold_enabled=true")
    if paths:
        raise ValueError(
            "timing_hold_lib_paths requires the independent early Liberty corner "
            "adapter, which is not available in this migration checkpoint"
        )
    if (
        active
        and _is_placement_flow(flow_kind)
        and getattr(params, "global_place_flag", True)
        and getattr(params, "placement_sizing_mode", "place_only") != "size_only"
        and getattr(params, "timing_placement_carrier", "direct_loss") != "direct_loss"
    ):
        raise ValueError(
            "active placement hold optimization currently requires "
            "timing_placement_carrier=direct_loss"
        )
    params.timing_hold_margin_ns = margin
    params.timing_hold_coefficients = coefficients
    return params
