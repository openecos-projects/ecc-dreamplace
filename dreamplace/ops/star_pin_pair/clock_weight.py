"""Clock-pair coverage in the shared timing-weight objective."""

from __future__ import annotations

from typing import Mapping

import torch


def launch_capture_clock_pair_adjoint(
    *,
    clock_gradients: Mapping[str, torch.Tensor],
    pair_sink_pin_ids: torch.Tensor,
    clock_row_to_compact_pin_ids: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, int | float | str]]:
    """Project setup-loss clock-arrival adjoints onto LCB-to-FF pairs.

    ``clock_gradients`` holds d(loss)/d(clk_pin_r/f_aat) from the timing VJP,
    indexed by clock row: the launch use through clk->Q plus the capture use
    through the setup RAT (negative).  Pair sinks are compact clock-Star pin
    IDs, mapped to rows with the inverse of ``clock_row_to_compact_pin_ids``.
    The pair wire delay shifts both edges, so the local derivative is the
    rise + fall sum.  A positive value means shortening the pair lowers the
    loss; capture-dominated (negative) pairs are clamped to zero.
    """
    required = ("clk_pin_r_aat", "clk_pin_f_aat")
    if not isinstance(clock_gradients, Mapping) or any(
        not torch.is_tensor(clock_gradients.get(name)) for name in required
    ):
        raise RuntimeError(
            "launch_capture clock weighting requires clock arrival adjoints"
        )
    sinks = torch.as_tensor(pair_sink_pin_ids).reshape(-1).long()
    rows_to_compact = torch.as_tensor(clock_row_to_compact_pin_ids).reshape(-1).long()
    rise = clock_gradients[required[0]].reshape(-1).to(
        device=sinks.device, dtype=torch.float32
    )
    fall = clock_gradients[required[1]].reshape(-1).to(
        device=sinks.device, dtype=torch.float32
    )
    if rise.shape != fall.shape or rise.numel() != rows_to_compact.numel():
        raise ValueError("clock arrival adjoints must match the clock-row domain")
    rows_to_compact = rows_to_compact.to(device=sinks.device)
    compact_count = int(max(int(rows_to_compact.max().item()) + 1, 0)) if rows_to_compact.numel() else 0
    compact_count = max(compact_count, int(sinks.max().item()) + 1 if sinks.numel() else 0)
    compact_to_row = torch.full((compact_count,), -1, dtype=torch.long, device=sinks.device)
    valid = rows_to_compact >= 0
    compact_to_row[rows_to_compact[valid]] = torch.nonzero(valid).reshape(-1)
    pair_rows = compact_to_row[sinks] if sinks.numel() else sinks
    if sinks.numel() and bool((pair_rows < 0).any().item()):
        raise ValueError("clock pair sink has no clock row")
    raw = (rise + fall).index_select(0, pair_rows)
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError("clock arrival adjoints contain non-finite values")
    weights = raw.clamp_min(0.0)
    receipt = {
        "mode": "launch_capture",
        "formula": "max(0, clk_pin_r_aat_vjp + clk_pin_f_aat_vjp)",
        "pair_count": int(sinks.numel()),
        "compact_pin_count": int(rise.numel()),
        "positive_pair_count": int((weights > 0.0).sum().item()),
        "negative_pair_count": int((raw < 0.0).sum().item()),
        "raw_l1": float(raw.abs().sum().item()),
        "published_l1": float(weights.sum().item()),
        "raw_min": float(raw.min().item()) if raw.numel() else 0.0,
        "raw_max": float(raw.max().item()) if raw.numel() else 0.0,
    }
    return weights, receipt


def capture_setup_clock_pair_coverage(
    *,
    endpoint_ids: torch.Tensor,
    endpoint_slack_ps: torch.Tensor,
    constraint_arcs: torch.Tensor,
    clock_row_to_compact_pin_ids: torch.Tensor,
    pair_sink_pin_ids: torch.Tensor,
    num_pins: int,
    num_compact_pins: int,
) -> tuple[torch.Tensor, Mapping[str, int | float]]:
    """Map violated setup endpoints onto compact LCB-to-FF clock pairs.

    The returned value is a non-negative graph-coverage count, matching the
    data-pair timing-weight unit. Each violated endpoint contributes one unit.
    Duplicate checks for the same endpoint/clock pin are deduplicated, and one
    endpoint contribution is split across multiple capture-clock candidates.
    """
    pair_sinks = torch.as_tensor(pair_sink_pin_ids).reshape(-1).long()
    pair_count = int(pair_sinks.numel())
    prototype = torch.as_tensor(endpoint_slack_ps)
    coverage = torch.zeros(
        pair_count, dtype=prototype.dtype, device=prototype.device
    )
    receipt = {
        "violating_endpoint_count": 0,
        "mapped_clock_constraint_count": 0,
        "weighted_pair_count": 0,
        "coverage_l1": 0.0,
        "max_coverage": 0.0,
    }
    if pair_count == 0:
        return coverage, receipt

    endpoint_ids = torch.as_tensor(endpoint_ids).reshape(-1).long()
    endpoint_slack = prototype.reshape(-1).to(
        device=prototype.device, dtype=prototype.dtype
    )
    constraints = torch.as_tensor(constraint_arcs, device=prototype.device)
    row_map = torch.as_tensor(
        clock_row_to_compact_pin_ids,
        device=prototype.device,
        dtype=torch.long,
    ).reshape(-1)
    if endpoint_ids.numel() != endpoint_slack.numel():
        raise ValueError("endpoint ids and slacks must have the same length")
    if constraints.dim() != 2 or constraints.shape[1] < 2:
        raise ValueError("constraint arcs must be a two-column-or-wider matrix")
    if int(num_pins) <= 0 or int(num_compact_pins) <= 0:
        raise ValueError("pin domains must be positive")
    if bool(((endpoint_ids < 0) | (endpoint_ids >= int(num_pins))).any().item()):
        raise ValueError("endpoint id is outside the timing pin domain")
    if bool(((pair_sinks < 0) | (pair_sinks >= int(num_compact_pins))).any().item()):
        raise ValueError("clock pair sink is outside the compact pin domain")
    if torch.unique(pair_sinks).numel() != pair_count:
        raise ValueError("clock pair sinks must be unique")

    endpoint_seed = torch.zeros(
        int(num_pins), dtype=prototype.dtype, device=prototype.device
    )
    violating = torch.isfinite(endpoint_slack) & (endpoint_slack < 0.0)
    endpoint_seed.index_copy_(
        0,
        endpoint_ids[violating],
        torch.ones_like(endpoint_slack[violating]),
    )
    receipt["violating_endpoint_count"] = int(violating.sum().item())
    if not bool(violating.any().item()):
        return coverage, receipt

    clock_rows = constraints[:, 0].long()
    data_pins = constraints[:, 1].long()
    unique_checks = torch.unique(
        torch.stack((data_pins, clock_rows), dim=1), dim=0
    )
    checked_data_pins = unique_checks[:, 0]
    checked_clock_rows = unique_checks[:, 1]
    active = (
        (checked_data_pins >= 0)
        & (checked_data_pins < int(num_pins))
        & (checked_clock_rows >= 0)
        & (checked_clock_rows < row_map.numel())
    )
    active &= endpoint_seed[checked_data_pins.clamp(0, int(num_pins) - 1)] != 0.0
    checked_data_pins = checked_data_pins[active]
    checked_clock_rows = checked_clock_rows[active]
    if checked_data_pins.numel() == 0:
        return coverage, receipt

    compact_pins = row_map[checked_clock_rows]
    active = (compact_pins >= 0) & (compact_pins < int(num_compact_pins))
    checked_data_pins = checked_data_pins[active]
    compact_pins = compact_pins[active]
    if checked_data_pins.numel() == 0:
        return coverage, receipt

    compact_to_pair = torch.full(
        (int(num_compact_pins),),
        -1,
        dtype=torch.long,
        device=prototype.device,
    )
    compact_to_pair.index_copy_(
        0,
        pair_sinks,
        torch.arange(pair_count, dtype=torch.long, device=prototype.device),
    )
    pair_indices = compact_to_pair[compact_pins]
    active = pair_indices >= 0
    checked_data_pins = checked_data_pins[active]
    pair_indices = pair_indices[active]
    if pair_indices.numel() == 0:
        return coverage, receipt

    unique_endpoints, inverse, counts = torch.unique(
        checked_data_pins, return_inverse=True, return_counts=True
    )
    contribution = endpoint_seed[unique_endpoints][inverse] / counts[inverse].to(
        prototype.dtype
    )
    coverage.scatter_add_(0, pair_indices, contribution)

    receipt.update(
        {
            "mapped_clock_constraint_count": int(pair_indices.numel()),
            "weighted_pair_count": int((coverage != 0.0).sum().item()),
            "coverage_l1": float(coverage.sum().item()),
            "max_coverage": (
                float(coverage.max().item()) if coverage.numel() else 0.0
            ),
        }
    )
    return coverage, receipt
