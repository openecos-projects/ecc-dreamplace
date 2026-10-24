"""Consume max-analysis qualification without inferring constraints from slack."""

import torch


def qualify_check_candidates(rise, fall, valid):
    if valid is None:
        return rise, fall
    return (
        torch.where(valid[:, 0], rise, torch.full_like(rise, torch.inf)),
        torch.where(valid[:, 1], fall, torch.full_like(fall, torch.inf)),
    )


def qualify_pin_slacks(timing, pins, rise, fall):
    valid_by_pin = getattr(timing, "endpoint_max_valid_by_pin", None)
    valid = None if valid_by_pin is None else valid_by_pin[pins.long()]
    return qualify_check_candidates(rise, fall, valid)


def aggregate_endpoint_slack(rise, fall, *, mode="hard", tau_ps=2.0):
    """Combine rise/fall endpoint slack using the timing aggregation mode."""

    if mode == "smooth":
        tau = float(tau_ps)
        if tau <= 0.0:
            raise ValueError("endpoint aggregation tau must be positive")
        return -tau * torch.logsumexp(
            torch.stack((-rise / tau, -fall / tau), dim=0),
            dim=0,
        )
    return torch.minimum(rise, fall)


def endpoint_metrics(slack, zero):
    if slack.numel() == 0:
        return zero, zero, zero
    negative = torch.clamp(slack, max=0)
    worst = slack.min()
    return (
        negative.min() + zero,
        negative.sum() + zero,
        torch.where(torch.isfinite(worst), worst, zero),
    )


def reporting_endpoint_validity(timing, fallback):
    """Physical report rows are valid if any legal state/check qualifies.

    Schema-v2 path-specific qualification is intentionally invalid; schema-v3
    supplies the matching endpoint AAT/required projection instead.
    """
    runtime = getattr(timing, "_sparse_runtime", None)
    if runtime is None:
        return fallback
    return runtime.endpoint_validity[timing.end_points.long()]
