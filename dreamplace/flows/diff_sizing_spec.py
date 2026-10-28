"""Configuration contract for ordered standalone ``diff_sizing`` phases.

A phase is one discrete action family, its round budget, its timing
coefficients, and its selection percentage.  The percentage is expressed in
percent of sizeable instances, matching the existing top-k budget.
"""

import math
from dataclasses import dataclass

from .diff_opt_spec import TimingCoefficients

DIFF_SIZING_ACTIONS = frozenset(("quad", "up", "down"))
DIFF_SIZING_ACTION_MASKS = {
    "quad": (True, True, True, True),
    "up": (True, False, False, True),
    "down": (False, True, True, False),
}


@dataclass(frozen=True)
class DiffSizingFlowSpec:
    action: str
    rounds: int
    coefficients: TimingCoefficients
    selection_percent: float

    def __post_init__(self):
        action = str(self.action).strip().lower()
        if action not in DIFF_SIZING_ACTIONS:
            raise ValueError(
                f"unsupported diff_sizing action: {self.action!r}; "
                f"expected one of {sorted(DIFF_SIZING_ACTIONS)}"
            )
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise ValueError("diff_sizing phase rounds must be a positive integer")
        if self.rounds <= 0:
            raise ValueError("diff_sizing phase rounds must be a positive integer")
        if isinstance(self.selection_percent, bool):
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            )
        try:
            percent = float(self.selection_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            ) from exc
        if not math.isfinite(percent) or not 0.0 < percent <= 100.0:
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "selection_percent", percent)

    @property
    def action_mask(self):
        return DIFF_SIZING_ACTION_MASKS[self.action]

    def to_list(self):
        return [
            self.action,
            int(self.rounds),
            self.coefficients.to_dict(),
            float(self.selection_percent),
        ]


def parse_diff_sizing_flows(value):
    """Parse ``[[action, rounds, coeffs, percent], ...]`` or return ``None``."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("diff_sizing_optimization_flows must be a non-empty list")
    parsed = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(
                f"diff_sizing_optimization_flows[{index}] must be "
                "[action, rounds, [wns, tns, slew, cap], selection_percent]"
            )
        action, rounds, coefficients, selection_percent = item
        parsed.append(
            DiffSizingFlowSpec(
                action=str(action),
                rounds=rounds,
                coefficients=TimingCoefficients.from_value(
                    coefficients,
                    field=f"diff_sizing_optimization_flows[{index}][2]",
                ),
                selection_percent=selection_percent,
            )
        )
    return tuple(parsed)


def canonical_diff_sizing_flows(value):
    parsed = parse_diff_sizing_flows(value)
    return None if parsed is None else [spec.to_list() for spec in parsed]


__all__ = [
    "DIFF_SIZING_ACTION_MASKS",
    "DIFF_SIZING_ACTIONS",
    "DiffSizingFlowSpec",
    "canonical_diff_sizing_flows",
    "parse_diff_sizing_flows",
]
