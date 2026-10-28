"""Configuration contract for the standalone ``diff_opt`` flow.

The ECC step owns the public mode/flow fields, while this module keeps the
DreamPlace side independent from chipcompiler.  A flow tuple always means one
discrete action phase; it never means a GP iteration or an optimizer step.
"""

from dataclasses import dataclass
import math


DIFF_OPT_SCHEMA_VERSION = 1
DIFF_OPT_MODES = frozenset(("sizing", "buffering", "joint"))
DIFF_OPT_ACTIONS = frozenset(("sz", "bf"))
DIFF_OPT_COEFFICIENT_NAMES = ("wns", "tns", "slew", "cap")
HOLD_COEFFICIENT_NAMES = ("hold_wns", "hold_tns")
DIFF_OPT_DEFAULT_COEFFICIENTS = {
    "wns": 500.0,
    "tns": 5.0,
    "slew": 1.0,
    "cap": 1.0,
}


@dataclass(frozen=True)
class TimingCoefficients:
    wns: float
    tns: float
    slew: float
    cap: float
    hold_wns: float = 0.0
    hold_tns: float = 0.0

    @classmethod
    def from_value(cls, value, *, field="coefficients"):
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError(f"{field} must contain [wns, tns, slew, cap]")
            value = dict(zip(DIFF_OPT_COEFFICIENT_NAMES, value, strict=True))
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a mapping or [wns, tns, slew, cap] list")
        unknown = set(value) - {*DIFF_OPT_COEFFICIENT_NAMES, *HOLD_COEFFICIENT_NAMES}
        if unknown:
            raise ValueError(f"{field} contains unknown keys: {sorted(unknown)}")
        result = {}
        for name in (*DIFF_OPT_COEFFICIENT_NAMES, *HOLD_COEFFICIENT_NAMES):
            raw = value.get(name, DIFF_OPT_DEFAULT_COEFFICIENTS.get(name, 0.0))
            if isinstance(raw, bool):
                raise ValueError(f"{field}.{name} must be a finite non-negative number")
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field}.{name} must be a finite non-negative number") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{field}.{name} must be a finite non-negative number")
            result[name] = number
        if not any(result.values()):
            raise ValueError(f"{field} must contain at least one positive coefficient")
        return cls(**result)

    def to_dict(self):
        values = {
            name: float(getattr(self, name)) for name in DIFF_OPT_COEFFICIENT_NAMES
        }
        values.update(
            {
                name: float(getattr(self, name))
                for name in HOLD_COEFFICIENT_NAMES
                if getattr(self, name)
            }
        )
        return values


@dataclass(frozen=True)
class DiffFlowSpec:
    action: str
    steps: int
    coefficients: TimingCoefficients

    def __post_init__(self):
        action = str(self.action).strip().lower()
        if action not in DIFF_OPT_ACTIONS:
            raise ValueError(f"unsupported diff_opt action: {self.action!r}")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ValueError("diff_opt steps must be a positive integer")
        if self.steps <= 0:
            raise ValueError("diff_opt steps must be a positive integer")
        object.__setattr__(self, "action", action)

    def to_list(self):
        return [self.action, int(self.steps), self.coefficients.to_dict()]


def _default_flows(mode):
    coefficients = TimingCoefficients.from_value(DIFF_OPT_DEFAULT_COEFFICIENTS)
    defaults = {
        "sizing": (("sz", 50),),
        "buffering": (("bf", 10),),
        "joint": (("sz", 5), ("bf", 1)),
    }
    return tuple(DiffFlowSpec(action, steps, coefficients) for action, steps in defaults[mode])


def parse_diff_opt_flows(mode="buffering", flows=None):
    """Parse and validate the canonical diff_opt mode and ordered flow list."""

    mode = str(mode or "buffering").strip().lower()
    if mode not in DIFF_OPT_MODES:
        raise ValueError(
            f"unsupported diff_optimization_mode: {mode!r}; "
            f"expected one of {sorted(DIFF_OPT_MODES)}"
        )
    if flows is None:
        parsed = _default_flows(mode)
    else:
        if not isinstance(flows, (list, tuple)) or not flows:
            raise ValueError("diff_optimization_flows must be a non-empty list")
        parsed = []
        for index, item in enumerate(flows):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    f"diff_optimization_flows[{index}] must be "
                    "[action, steps, [wns, tns, slew, cap]]"
                )
            action, steps, coefficients = item
            parsed.append(
                DiffFlowSpec(
                    str(action).strip().lower(),
                    steps,
                    TimingCoefficients.from_value(
                        coefficients,
                        field=f"diff_optimization_flows[{index}][2]",
                    ),
                )
            )
        parsed = tuple(parsed)

    allowed = {"sz"} if mode == "sizing" else {"bf"} if mode == "buffering" else DIFF_OPT_ACTIONS
    invalid = [spec.action for spec in parsed if spec.action not in allowed]
    if invalid:
        raise ValueError(
            f"diff_optimization_mode={mode!r} does not allow actions {sorted(set(invalid))}"
        )
    return mode, tuple(parsed)


def canonical_diff_opt_flows(mode="buffering", flows=None):
    mode, parsed = parse_diff_opt_flows(mode, flows)
    return mode, [spec.to_list() for spec in parsed]


__all__ = [
    "DIFF_OPT_ACTIONS",
    "DIFF_OPT_COEFFICIENT_NAMES",
    "DIFF_OPT_DEFAULT_COEFFICIENTS",
    "DIFF_OPT_MODES",
    "DIFF_OPT_SCHEMA_VERSION",
    "HOLD_COEFFICIENT_NAMES",
    "DiffFlowSpec",
    "TimingCoefficients",
    "canonical_diff_opt_flows",
    "parse_diff_opt_flows",
]
