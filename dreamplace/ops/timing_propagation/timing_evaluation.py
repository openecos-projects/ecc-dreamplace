"""Publish max and min metrics from one successful timing forward together."""

from dataclasses import dataclass

import torch

from .early_propagation import EarlyTimingResult


@dataclass(frozen=True)
class TimingEvaluation:
    generation: int
    setup_wns: torch.Tensor
    setup_tns: torch.Tensor
    setup_worst: torch.Tensor
    setup_total: torch.Tensor
    aggregation: str
    hold: EarlyTimingResult | None
    hold_liberty_paths: tuple[str, ...] = ()

    def hold_phase_snapshot(self):
        """Detached ps metrics shared by sizing and buffering phase artifacts."""
        if self.hold is None:
            return {}
        hold = self.hold
        return {
            "timing_generation": self.generation,
            "timing_aggregation": self.aggregation,
            "hold_wns": float(hold.hold_wns.detach()),
            "hold_tns": float(hold.hold_tns.detach()),
            "hold_worst": float(hold.hold_worst.detach()),
            "hold_violations": int(hold.hold_violations),
            "removal_wns": float(hold.removal_wns.detach()),
            "removal_tns": float(hold.removal_tns.detach()),
            "removal_violations": int(hold.removal_violations),
        }

    def hold_report(self):
        if self.hold is None:
            return None
        hold = self.hold
        return {
            "generation": self.generation,
            **({"early_liberty_paths": list(self.hold_liberty_paths)} if self.hold_liberty_paths else {}),
            "aggregation": self.aggregation,
            "wns_ns": float(hold.hold_wns.detach()) / 1000.0,
            "tns_ns": float(hold.hold_tns.detach()) / 1000.0,
            "worst_ns": float(hold.hold_worst.detach()) / 1000.0,
            "violation_endpoints": int(hold.hold_violations),
            "qualified_endpoint_count": int(torch.isfinite(hold.endpoint_slack).sum()),
            "context_count": hold.context_slack.numel(),
            "removal": {
                "wns_ns": float(hold.removal_wns.detach()) / 1000.0,
                "tns_ns": float(hold.removal_tns.detach()) / 1000.0,
                "violation_endpoints": int(hold.removal_violations),
            },
        }
