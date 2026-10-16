"""Evaluate a frozen GR tree with live, differentiable cell pin capacitances."""

import time

import torch
from torch import nn

from dreamplace.ops.rc_timing.rc_timing import evaluate_rc_tree

from .route_snapshot import RouteSnapshot, prepare_snapshot


class GRParasiticsOp(nn.Module):
    def __init__(self, snapshot: RouteSnapshot):
        super().__init__()
        self.snapshot = snapshot
        self.forward_calls = 0
        self.forward_seconds = 0.0
        self.loads, self.delays, self.ldelays, self.betas, self.impulses = ({}, {}, {}, {}, {})
        self.roundoff_corrections = 0

    @classmethod
    def prepare(cls, route_pack, rc_parameters, pin_mapping, identity, *, dtype=torch.float32):
        return cls(prepare_snapshot(route_pack, rc_parameters, pin_mapping, identity, dtype=dtype))

    def forward(self, pin_cap_base, pin_cap_rise, pin_cap_fall):
        """Return the six RC dictionaries described by ``evaluate_rc_tree``.

        Inputs and returned vectors use the physical-pin domain. Internally,
        pin C is scattered into independent RC vertices, then results are
        gathered through the snapshot's pin mapping. Wire/via RC stays fixed;
        pin C remains differentiable. Filtered pins keep local cap/load and
        ideal zero wire delay, and do not count toward route coverage.
        """
        started = time.perf_counter()
        tree = self.snapshot
        groups = tuple({} for _ in range(6))
        for mode, cap in zip(
            ("generic", "rise", "fall"), (pin_cap_base, pin_cap_rise, pin_cap_fall), strict=True
        ):
            if (
                cap.device.type != "cpu"
                or cap.dtype != tree.wire_cap.dtype
                or cap.ndim != 1
                or cap.numel() != tree.pin_to_vertex.numel()
            ):
                raise ValueError(
                    "GR pin cap must match the snapshot CPU dtype and physical pin domain"
                )
            if not bool(torch.isfinite(cap).all()) or bool((cap < 0).any()):
                raise ValueError("GR pin capacitance must be finite and nonnegative")
            local_cap = tree.wire_cap.scatter_add(
                0, tree.valid_vertex_ids, cap.index_select(0, tree.valid_pin_ids)
            )
            load, delay, ldelay, beta, impulse = evaluate_rc_tree(
                local_cap,
                tree.incoming_resistance,
                tree.parent,
                tree.child_start,
                tree.child_vertex,
                tree.topo_order,
                tree.net_topo_start,
            )
            # Only cancellation bounded by the arithmetic's own error scale
            # may be stabilized. Significant negative moments are model errors.
            bound = 8 * torch.finfo(cap.dtype).eps * (2 * beta.abs() + delay.square())
            if not bool(torch.isfinite(impulse).all()) or bool((impulse < -bound).any()):
                raise ValueError("GR RC produced a nonfinite or significantly negative impulse")
            negative = impulse < 0
            if bool(negative.any()):
                self.roundoff_corrections += int(negative.sum())
                impulse = torch.where(negative, torch.zeros_like(impulse), impulse)
            for index, value in enumerate((local_cap, load, delay, ldelay, beta, impulse)):
                # Filtered clock/PG/high-fanout pins keep local cap/load and ideal
                # zero wire delay; they are not counted as covered routing.
                base = cap if index < 2 else torch.zeros_like(cap)
                groups[index][mode] = base.index_copy(
                    0, tree.valid_pin_ids, value.index_select(0, tree.valid_vertex_ids)
                )
        _, self.loads, self.delays, self.ldelays, self.betas, self.impulses = groups
        self.forward_calls += 1
        self.forward_seconds += time.perf_counter() - started
        return groups

    @property
    def wire_capacitance(self):
        """Unique RC-vertex domain; summing a pin gather would double-count C."""
        return self.snapshot.wire_cap
