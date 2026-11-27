"""Drift-calibrated per-pin slack retention vector ``beta`` across sizing/buffering restarts.

Under subsequent placement/legalization timing drift of Gaussian scale ``sigma``,
a pin transition with current slack ``s = RAT - AAT`` has expected violation
severity proportional to::

    phi_tau(s) = tau * log(1 + exp(-s / tau)),   tau = sigma / 1.70

where ``sigma = 1.4826 * weighted_median(|Delta s|)`` is the exact same parameter-free
robust drift estimator used by ``AtMergeTemperature`` on internal AT merges.

Across a discrete sizing / buffering / routability restart, historical input
adjoints ``W_before`` in ``InputAccumulator`` are scaled componentwise by::

    beta_{p, e} = clip(phi_tau(s_after[p, e]) / (phi_tau(s_before[p, e]) + eps), 0, 1)

so deeply cleared pins (``s_after >> tau``) release to 0, still-violated pins
retain proportional stiffness, and newly zero-margin pins (``s_after ~ 0``) keep
a physically calibrated anti-rebound guardband without hand-tuned ps constants.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from dreamplace.ops.timing_propagation.at_merge_temperature import (
    LOGISTIC_SCALE,
    MAD_TO_SIGMA,
    weighted_median,
)

UNCONSTRAINED_SLACK_THRESHOLD_PS = 1.0e7


def valid_pin_slack_mask(slack: torch.Tensor) -> torch.Tensor:
    """True where ``slack`` represents a finite, timing-constrained pin transition."""
    return torch.isfinite(slack) & (slack < UNCONSTRAINED_SLACK_THRESHOLD_PS)


def soft_slack_severity(slack: torch.Tensor, *, tau_ps: float) -> torch.Tensor:
    """Smooth expected violation severity ``tau * softplus(-s / tau)`` in ps."""
    tau = max(1e-6, float(tau_ps))
    if not math.isfinite(tau):
        raise ValueError(f"tau_ps must be finite, got {tau_ps}")
    valid = valid_pin_slack_mask(slack)
    safe_s = torch.where(valid, slack.to(dtype=torch.float64), torch.zeros((), dtype=torch.float64, device=slack.device))
    sev = tau * F.softplus(-safe_s / tau)
    return torch.where(valid, sev, torch.zeros_like(sev)).to(dtype=torch.float32)


def compute_slack_retention_beta(
    slack_before: Mapping[str, torch.Tensor],
    slack_after: Mapping[str, torch.Tensor],
    *,
    tau_ps: float,
    eps: float = 1e-9,
) -> Dict[str, torch.Tensor]:
    """Compute per-edge pin retention vector ``beta[edge] in [0, 1]``."""
    tau = max(1e-6, float(tau_ps))
    denom_eps = max(1e-12, float(eps))
    out: Dict[str, torch.Tensor] = {}
    for edge in ("rise", "fall"):
        s_prev = slack_before[edge]
        s_curr = slack_after[edge]
        if s_prev.shape != s_curr.shape:
            raise ValueError(
                f"slack shape mismatch on edge {edge}: {tuple(s_prev.shape)} vs {tuple(s_curr.shape)}"
            )
        phi_prev = soft_slack_severity(s_prev, tau_ps=tau).to(dtype=torch.float64)
        phi_curr = soft_slack_severity(s_curr, tau_ps=tau).to(dtype=torch.float64)
        active_prev = phi_prev > denom_eps
        ratio = torch.where(
            active_prev,
            (phi_curr / (phi_prev + denom_eps)).clamp(min=0.0, max=1.0),
            torch.zeros_like(phi_curr),
        )
        out[edge] = ratio.to(device=s_curr.device, dtype=torch.float32)
    return out


def compute_delay_sigmoid_retention_beta(
    slack_after: Mapping[str, torch.Tensor],
    *,
    tau_ps: float,
    mutated_cap_pin_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Channel-aware post-mutation retention using derivative gate ``sigmoid(-s_after / tau)``.

    When ``mutated_cap_pin_mask`` is ``None`` (``delay_sigmoid`` mode), ``cap``
    historical adjoints are unconditionally zeroed on all pins.  When
    ``mutated_cap_pin_mask`` is provided (``selective_sigmoid`` mode), ``cap``
    historical adjoints are zeroed ONLY on pins whose driving cell was resized
    or whose net fanout mutated (e.g. buffer insertion), while unchanged pins
    apply the same derivative gate ``sigmoid(-s_after / tau)`` as ``delay``.
    """
    tau = max(1e-6, float(tau_ps))
    if not math.isfinite(tau):
        raise ValueError(f"tau_ps must be finite, got {tau_ps}")
    delay_beta: Dict[str, torch.Tensor] = {}
    cap_beta: Dict[str, torch.Tensor] = {}
    for edge in ("rise", "fall"):
        s_curr = slack_after[edge]
        valid = valid_pin_slack_mask(s_curr)
        safe_s = torch.where(
            valid,
            s_curr.to(dtype=torch.float64),
            torch.zeros((), dtype=torch.float64, device=s_curr.device),
        )
        gate = torch.sigmoid(-safe_s / tau)
        edge_gate = torch.where(
            valid, gate, torch.zeros_like(gate)
        ).to(device=s_curr.device, dtype=torch.float32)
        delay_beta[edge] = edge_gate
        if mutated_cap_pin_mask is None:
            cap_beta[edge] = torch.zeros_like(edge_gate)
        else:
            mut = mutated_cap_pin_mask.to(device=s_curr.device, dtype=torch.bool)
            if mut.shape != edge_gate.shape:
                raise ValueError(
                    f"mutated_cap_pin_mask shape {tuple(mut.shape)} != pin slack shape {tuple(edge_gate.shape)}"
                )
            cap_beta[edge] = torch.where(mut, torch.zeros_like(edge_gate), edge_gate)
    return {"delay": delay_beta, "cap": cap_beta}


def estimate_slack_drift_tau(
    slack_prev: Mapping[str, torch.Tensor],
    slack_curr: Mapping[str, torch.Tensor],
    *,
    pin_weights: Optional[Mapping[str, torch.Tensor]] = None,
    base_tau_ps: float = 1.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate ``(tau_ps, sigma_ps)`` from same-topology pin-slack drift across snapshots."""
    base = max(1e-6, float(base_tau_ps))
    drifts = []
    weights = []
    for edge in ("rise", "fall"):
        if edge not in slack_prev or edge not in slack_curr:
            continue
        s0 = slack_prev[edge]
        s1 = slack_curr[edge]
        if s0.shape != s1.shape or s0.numel() == 0:
            continue
        valid = valid_pin_slack_mask(s0) & valid_pin_slack_mask(s1)
        if not bool(valid.any().item()):
            continue
        if pin_weights is not None and edge in pin_weights:
            w = pin_weights[edge].detach().to(device=s0.device, dtype=torch.float64)
            w = torch.where(valid & torch.isfinite(w) & (w > 0.0), w, torch.zeros_like(w))
        else:
            # Fallback weight: active near-critical/violated support on either snapshot.
            near = (s0 < 50.0) | (s1 < 50.0)
            w = torch.where(valid & near, torch.ones_like(s0, dtype=torch.float64), torch.zeros_like(s0, dtype=torch.float64))
        pos_mask = w > 0.0
        if not bool(pos_mask.any().item()):
            continue
        delta = (s1[pos_mask].to(torch.float64) - s0[pos_mask].to(torch.float64)).abs()
        drifts.append(delta.cpu())
        weights.append(w[pos_mask].cpu())
    if not drifts:
        return None, None
    all_drift = torch.cat(drifts, dim=0)
    all_weight = torch.cat(weights, dim=0)
    total_w = float(all_weight.sum().item())
    if total_w <= 0.0 or all_drift.numel() == 0:
        return None, None
    med = weighted_median(all_drift, all_weight)
    sigma = float(MAD_TO_SIGMA * med)
    tau = max(base, float(sigma / LOGISTIC_SCALE))
    return tau, sigma


def summarize_drift_tau_history(
    slack_tau_history: Sequence[float],
    *,
    at_merge_tau_history: Sequence[float] = (),
    base_tau_ps: float = 1.0,
    scale: float = 1.0,
    min_tau_ps: float = 1.0,
) -> float:
    """Resolve parameter-free drift temperature ``tau`` from observed GP history."""
    floor = max(1e-6, float(min_tau_ps), float(base_tau_ps))
    mult = max(0.0, float(scale))
    samples = [float(v) for v in slack_tau_history if v is not None and math.isfinite(v) and float(v) > 0.0]
    if not samples:
        samples = [float(v) for v in at_merge_tau_history if v is not None and math.isfinite(v) and float(v) > float(base_tau_ps)]
    if not samples:
        return floor
    ordered = sorted(samples)
    n = len(ordered)
    mid = n // 2
    median_tau = ordered[mid] if (n % 2 == 1) else 0.5 * (ordered[mid - 1] + ordered[mid])
    return max(floor, mult * float(median_tau))
