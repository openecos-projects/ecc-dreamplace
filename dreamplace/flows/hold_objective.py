"""Hold deficit and coefficients at the common timing objective boundary."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HoldObjective:
    wns: float
    tns: float
    margin_ps: float

    @property
    def active(self):
        return self.wns != 0 or self.tns != 0

    @classmethod
    def from_params(cls, params):
        coefficients = getattr(params, "timing_hold_coefficients", {"wns": 0.0, "tns": 0.0})
        if not isinstance(coefficients, dict) or set(coefficients) - {"wns", "tns"}:
            raise ValueError("timing_hold_coefficients must contain only named wns/tns weights")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in coefficients.values()
        ):
            raise ValueError("hold coefficients must be numeric")
        result = cls(
            float(coefficients.get("wns", 0)),
            float(coefficients.get("tns", 0)),
            1000 * float(getattr(params, "timing_hold_margin_ns", 0)),
        )
        if any(
            not math.isfinite(value) or value < 0
            for value in (result.wns, result.tns, result.margin_ps)
        ):
            raise ValueError("hold coefficients and margin must be finite and nonnegative")
        enabled = bool(getattr(params, "timing_hold_enabled", False))
        if (result.wns or result.tns) and not enabled:
            raise ValueError("nonzero hold coefficients require timing_hold_enabled")
        if not enabled:
            return None
        if not params.with_sta:
            raise ValueError("timing_hold_enabled requires STA")
        if (
            (result.wns or result.tns)
            and getattr(params, "placement_sizing_mode", "place_only") != "size_only"
            and getattr(params, "timing_placement_carrier", "direct_loss") != "direct_loss"
        ):
            raise ValueError("joint setup/hold placement currently requires direct_loss")
        return result

    def loss(self, evaluation):
        if evaluation is None or evaluation.hold is None:
            raise RuntimeError("joint hold loss requires the current complete timing evaluation")
        if not self.active:
            return evaluation.hold.hold_wns.new_zeros(())
        wns, tns = evaluation.hold.deficits(self.margin_ps)
        return self.wns * wns + self.tns * tns
