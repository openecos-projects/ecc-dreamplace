"""Per-net teacher homothety slope calibration and B^+ load clamp for Star TDP.

At every data Star publication, for each timing net ``n`` with Star pairs
``(driver d, sinks i)``:

    s_n = sum_i K_i * dD_teacher_i/dlam  /  sum_i K_i * dD_star_i/dlam

where ``lam`` is the homothety ``x -> x_d + lam * (x - x_d)`` of all pins of
the net about its driver, evaluated at ``lam = 1``.  The teacher is FLUTE
SteinerTopo + RCTiming Elmore with topology frozen at ``lam = 1``; its sink
wire delay is ``a * lam^2 + b * lam`` exactly, so
``dD/dlam = 3 * D(1) - 4 * D(1/2)``.  The Star branch delay
``r * L * (C + c * L / 2)`` has ``dD/dlam = r * L * (C + c * L)``.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import dreamplace.ops.steiner_topo.steiner_topo as steiner_topo
from dreamplace.ops.buffer_insertion.net_eligibility import clock_net_ids
from dreamplace.ops.rc_timing.rc_timing import RCTiming


def clamp_load_length_weight(
    raw_load_length_weight: torch.Tensor,
    *,
    mode: str = "clamp",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(published_load_weight, curvature_load_weight)`` for Star B^+.

    In ``R0_reference`` (``mode == "clamp"``), negative load sensitivities
    (beneficial load / slew effects) are clamped to zero in the objective force
    so no driver-sink pair is pushed apart, while the unclamped signed load
    weight is retained as ``curvature_load_length_weight``.
    """
    normalized = str(mode or "clamp").strip().lower()
    if normalized not in ("clamp", "signed", "zero"):
        raise ValueError(f"star_load_sign must be clamp, signed, or zero (got {mode!r})")
    raw = raw_load_length_weight.detach()
    if normalized == "clamp":
        return raw.clamp(min=0.0), raw
    if normalized == "zero":
        zeros = torch.zeros_like(raw)
        return zeros, zeros
    return raw, raw


def teacher_rc_like(rc_op) -> RCTiming:
    """Create an isolated RCTiming instance matching ``rc_op`` units."""
    return RCTiming(
        r_unit=float(rc_op.r_unit),
        c_unit=float(rc_op.c_unit),
        scale_factor=float(rc_op.scale_factor),
        dbu=float(rc_op.dbu),
    )


def net_roots(
    flat_net2pin_map: torch.Tensor,
    flat_net2pin_start_map: torch.Tensor,
) -> torch.Tensor:
    """Local pin 0 of every native net: the FLUTE root used by SteinerTopo."""
    start = flat_net2pin_start_map.long()
    degree = start[1:] - start[:-1]
    roots = torch.full((degree.numel(),), -1, dtype=torch.long)
    has = degree > 0
    roots[has] = flat_net2pin_map.long()[start[:-1][has]]
    return roots


def homothety_teacher_slopes(
    pin_pos: torch.Tensor,
    *,
    topo: steiner_topo.SteinerTopo,
    rc: RCTiming,
    caps: torch.Tensor,
    pin2net: torch.Tensor,
    roots: torch.Tensor,
    prebuilt_tree: Optional[tuple] = None,
) -> torch.Tensor:
    """Per-pin ``dD/dlam`` at ``lam = 1`` of teacher wire delay, frozen at ``lam = 1``."""
    n = int(pin_pos.numel() // 2)
    x, y = pin_pos[:n], pin_pos[n:]
    center = torch.arange(n, dtype=torch.long)
    valid = pin2net >= 0
    if bool(valid.any().item()):
        net_roots_for_pin = roots[pin2net[valid]]
        has_root = net_roots_for_pin >= 0
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        center[valid_indices[has_root]] = net_roots_for_pin[has_root]
    cx, cy = x[center], y[center]
    tree = (
        prebuilt_tree
        if prebuilt_tree is not None
        else topo.rebuild_tree(pin_pos.contiguous(), resolve_l_directions=False)
    )
    sort, sort_start, fa, to, to_start, frm = tree

    def delays(lam: float) -> torch.Tensor:
        scaled = torch.cat((cx + lam * (x - cx), cy + lam * (y - cy))).contiguous()
        nx, ny = topo(scaled)
        out = rc(nx, ny, sort, sort_start, fa, to_start, to, frm, caps, caps, caps)
        d = out[2]["generic"] if isinstance(out[2], dict) else out[2]
        return d[:n].detach().double()

    with torch.no_grad():
        full = delays(1.0)
        half = delays(0.5)
    return 3.0 * full - 4.0 * half


def net_scale(
    weight: torch.Tensor,
    teacher_slope_at_sink: torch.Tensor,
    star_slope: torch.Tensor,
    pair_net_ids: torch.Tensor,
    num_nets: int,
    *,
    invalid_nets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``s_n = sum K*teacher / sum K*star`` over pairs of net ``n``; 1 where undefined."""
    w = weight.double()
    num = torch.zeros(num_nets, dtype=torch.float64).index_add_(
        0, pair_net_ids, w * teacher_slope_at_sink
    )
    den = torch.zeros(num_nets, dtype=torch.float64).index_add_(
        0, pair_net_ids, w * star_slope
    )
    scale = torch.ones(num_nets, dtype=torch.float64)
    ok = den > 0
    scale[ok] = num[ok] / den[ok]
    if invalid_nets is not None:
        scale[invalid_nets] = 1.0
    return scale


def weighted_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    q: float,
) -> Optional[float]:
    if values.numel() == 0:
        return None
    order = torch.argsort(values)
    cumulative = torch.cumsum(weights[order], 0)
    total = float(cumulative[-1].item()) if cumulative.numel() else 0.0
    if total <= 0.0:
        return None
    idx = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(q * total, dtype=cumulative.dtype),
        ).clamp(max=len(order) - 1)
    )
    return float(values[order][idx].item())


class SlopeCalibration:
    """Per-design FLUTE teacher homothety slope calibrator for Star data pairs."""

    def __init__(self, *, placedb, params, data_collections, star, elmore_delay_op):
        if elmore_delay_op is None:
            raise RuntimeError("star slope calibration requires elmore_delay_op")
        self.rc = teacher_rc_like(elmore_delay_op)
        scale = float(self.rc.scale_factor) * float(self.rc.dbu)
        if not np.isclose(scale, float(star.coordinate_to_um), rtol=1e-9):
            raise RuntimeError("teacher and Star coordinate scales differ")
        if not (
            np.isclose(float(self.rc.r_unit), float(star.r_unit))
            and np.isclose(float(self.rc.c_unit), float(star.c_unit))
        ):
            raise RuntimeError("teacher and Star RC units differ")
        net2pin = data_collections.flat_net2pin_map.detach().cpu()
        start = data_collections.flat_net2pin_start_map.detach().cpu()
        ignored_clock = sorted(clock_net_ids(placedb, data_collections))
        self.topo = steiner_topo.SteinerTopo(
            net2pin,
            start,
            ignore_net_degree=getattr(params, "ignore_net_degree", None),
            ignored_net_ids=ignored_clock,
            deterministic_flag=bool(getattr(params, "deterministic_flag", False)),
        )
        self.roots = net_roots(net2pin, start)
        self.pin2net = data_collections.pin2net_map.detach().cpu().long()
        self.pair_net = star.pair_net_ids.detach().cpu().long()
        self.sink = star.sink_pin_ids.detach().cpu().long()
        self.driver = star.driver_pin_ids.detach().cpu().long()
        self.num_nets = int(star.num_nets)
        drivers = star.net_driver_pin_ids.detach().cpu().long()
        self.invalid = torch.ones(self.num_nets, dtype=torch.bool)
        has_driver = (drivers >= 0) & (drivers < int(self.pin2net.numel()))
        if bool(has_driver.any().item()):
            valid_net_idx = torch.nonzero(has_driver, as_tuple=False).reshape(-1)
            native = self.pin2net[drivers[valid_net_idx]]
            native_ok = (native >= 0) & (native < int(self.roots.numel()))
            matched_idx = valid_net_idx[native_ok]
            matched_native = native[native_ok]
            root_ok = self.roots[matched_native] == drivers[matched_idx]
            self.invalid[matched_idx[root_ok]] = False
        if ignored_clock:
            for cid in ignored_clock:
                if 0 <= int(cid) < self.num_nets:
                    self.invalid[int(cid)] = True
        self.r = float(star.r_unit)
        self.c = float(star.c_unit)
        self.unit = float(star.coordinate_to_um)
        self.pair_topology_generation = int(star.pair_topology_generation)

    def rescale_weights(
        self,
        *,
        pin_pos: torch.Tensor,
        base_caps_generic: torch.Tensor,
        raw_weight: torch.Tensor,
        iteration: int = 0,
        prebuilt_tree: Optional[tuple] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        started = time.time()
        if raw_weight.numel() == 0:
            return raw_weight, {
                "enabled": True,
                "iteration": int(iteration),
                "seconds": 0.0,
                "nets_with_k": 0,
                "k_l1_before": 0.0,
                "k_l1_after": 0.0,
            }
        cpu_pos = pin_pos.detach().to("cpu", dtype=torch.float32).contiguous()
        caps = base_caps_generic.detach().to("cpu", dtype=cpu_pos.dtype).contiguous()
        teacher = homothety_teacher_slopes(
            cpu_pos,
            topo=self.topo,
            rc=self.rc,
            caps=caps,
            pin2net=self.pin2net,
            roots=self.roots,
            prebuilt_tree=prebuilt_tree,
        )
        n = int(cpu_pos.numel() // 2)
        x, y = cpu_pos[:n].double(), cpu_pos[n:].double()
        length = (
            (x[self.driver] - x[self.sink]).abs()
            + (y[self.driver] - y[self.sink]).abs()
        ) / self.unit
        star_slope = self.r * length * (caps.double()[self.sink] + self.c * length)
        w = raw_weight.detach().to("cpu").double()
        scale = net_scale(
            w,
            teacher[self.sink],
            star_slope,
            self.pair_net,
            self.num_nets,
            invalid_nets=self.invalid,
        )
        factor = scale[self.pair_net]
        rescaled = (w * factor).to(device=raw_weight.device, dtype=raw_weight.dtype)
        mass = torch.zeros(self.num_nets, dtype=torch.float64).index_add_(
            0, self.pair_net, w.abs()
        )
        active = mass > 0
        receipt = {
            "enabled": True,
            "iteration": int(iteration),
            "seconds": time.time() - started,
            "nets_with_k": int(active.sum().item()),
            "invalid_root_nets": int(self.invalid.sum().item()),
            "k_l1_before": float(w.abs().sum().item()),
            "k_l1_after": float((w * factor).abs().sum().item()),
            "s_weighted_p10": weighted_quantile(scale[active], mass[active], 0.1),
            "s_weighted_p50": weighted_quantile(scale[active], mass[active], 0.5),
            "s_weighted_p90": weighted_quantile(scale[active], mass[active], 0.9),
        }
        return rescaled, receipt
