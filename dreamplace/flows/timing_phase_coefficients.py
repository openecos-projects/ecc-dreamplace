"""Freeze a phase's timing coefficients and restore the owning GP policy."""

from contextlib import contextmanager
from dataclasses import replace

from .timing_opt_config import TIMING_OPT_COEFFICIENT_FIELDS


@contextmanager
def timing_phase_coefficients(model, fixed=None):
    original = {
        name: float(getattr(model, field))
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items()
    }
    weight = model.timing_grad_balance_weight
    hold = getattr(model, "hold_objective", None)
    coefficients = dict(original if fixed is None else fixed)
    if fixed is None and hold is not None:
        coefficients.update(hold_wns=hold.wns, hold_tns=hold.tns)
    if hold is None and (
        coefficients.get("hold_wns", 0) or coefficients.get("hold_tns", 0)
    ):
        raise ValueError("nonzero phase hold weights require timing_hold_enabled")
    try:
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
            setattr(model, field, float(coefficients[name]))
        if hold is not None:
            model.hold_objective = replace(
                hold,
                wns=coefficients.get("hold_wns", 0),
                tns=coefficients.get("hold_tns", 0),
            )
        model.timing_grad_balance_weight = weight if fixed is None else 1.0
        yield coefficients | {
            "timing_grad_balance_weight": float(model.timing_grad_balance_weight)
        }
    finally:
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
            setattr(model, field, original[name])
        if hold is not None:
            model.hold_objective = hold
        model.timing_grad_balance_weight = weight
