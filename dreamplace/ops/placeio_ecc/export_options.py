"""Resolve native PyPlaceDB M2 geometry options once per ECC database session."""

import math
from dataclasses import dataclass
from numbers import Integral


def resolve_flag(params, name, default=0):
    value = getattr(params, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    elif isinstance(value, Integral) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"{name} must be a boolean-like value (0/1, true/false, yes/no, on/off)"
    )


def resolve_m2_density_weight(params):
    try:
        value = float(getattr(params, "m2_pg_rail_density_weight", 1.0))
    except (TypeError, ValueError):
        raise ValueError("m2_pg_rail_density_weight must be numeric") from None
    if not math.isfinite(value):
        raise ValueError("m2_pg_rail_density_weight must be finite")
    if value < 0:
        raise ValueError("m2_pg_rail_density_weight must be non-negative")
    return value


@dataclass(frozen=True)
class PyDbExportOptions:
    include_m2_pg_rail_blockage: bool = False
    include_m2_pg_rail_density: bool = True

    @classmethod
    def from_params(cls, params):
        blockage = resolve_flag(params, "ieda_m2_pg_rail_blockage_flag")
        density = (
            (
                resolve_m2_density_weight(params) > 0
                or resolve_flag(params, "m2_pg_rail_legalization_blockage_flag")
            )
            and not blockage
        ) or resolve_flag(params, "m2_pa_refine_flag")
        return cls(blockage, density)
