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
