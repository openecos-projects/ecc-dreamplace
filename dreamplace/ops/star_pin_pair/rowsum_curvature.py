"""Gershgorin row-sum MM curvature diagonal, clock curvature, and Nesterov momentum transport.

Consolidates the production FULL8 preconditioner, clock curvature, momentum
transport, and input-adjoint accumulation formulas into ``dreamplace`` (E-0a).
"""

from __future__ import annotations

import math
import torch

CHANNELS = (("delay", "delays"), ("cap", "loads"))
EDGES = ("rise", "fall")


def linear_length_diagonal(
    weight,
    load,
    sink_cap,
    relative_xy,
    *,
    coefficient,
    r_unit,
    coordinate_to_um,
    gamma,
    driver_nodes,
    sink_nodes,
    movable_count,
    node_count,
    c_unit=None,
    length_um=None,
):
    """MM diagonal for the linear Star load/length slope with row-sum factor.

    The pair term is f(L) = K*r*L*(cap + c*L/2) + B*L.  The MM bound of the length
    curvature uses f'(L); ``length_um`` given (with ``c_unit``) adds the K*r*c*L
    part of f'(L), which the constant slope K*r*cap + B leaves out.
    """
    if load.numel() == 0:
        load = torch.zeros_like(weight)
    elif load.shape != weight.shape:
        raise RuntimeError("Star load coefficient shape differs from delay weight")
    slope = coefficient * (weight * r_unit * sink_cap + load)
    if length_um is not None:
        slope = slope + coefficient * weight * r_unit * float(c_unit) * length_um
    pair_diagonal = slope.clamp(min=0).unsqueeze(0) / (
        coordinate_to_um * relative_xy.abs().clamp(min=float(gamma))
    )
    distinct = driver_nodes != sink_nodes
    free_count = (driver_nodes < movable_count).to(pair_diagonal)
    free_count = free_count + (sink_nodes < movable_count).to(pair_diagonal)
    pair_diagonal = pair_diagonal * free_count.unsqueeze(0)
    diagonal = relative_xy.new_zeros((2, node_count))
    for endpoint in (driver_nodes, sink_nodes):
        mask = distinct & (endpoint < movable_count)
        for axis in range(2):
            diagonal[axis].scatter_add_(0, endpoint[mask], pair_diagonal[axis, mask])
    return diagonal


def elmore_rowsum_diagonal(
    weight,
    *,
    coefficient,
    r_unit,
    c_unit,
    coordinate_to_um,
    driver_nodes,
    sink_nodes,
    movable_count,
    node_count,
    axes=1,
):
    """Gershgorin row-sum diagonal for the Star Elmore quadratic term.

    The term is A*L^2 with L = |dx| + |dy|, so its 4x4 Hessian couples x and y:
    every entry is +-2A and an endpoint row sums to 2A per free endpoint per
    axis.  ``axes=2`` counts both axes (the full row sum); ``axes=1`` is the
    original per-axis count, half the bound.
    """
    curvature = (
        coefficient
        * weight.abs()
        * (r_unit * c_unit)
        / (coordinate_to_um * coordinate_to_um)
        * float(axes)
    )
    distinct = driver_nodes != sink_nodes
    free_count = (driver_nodes < movable_count).to(curvature)
    free_count = free_count + (sink_nodes < movable_count).to(curvature)
    pair_diagonal = curvature * distinct * free_count
    diagonal = weight.new_zeros((2, node_count))
    for endpoint in (driver_nodes, sink_nodes):
        mask = distinct & (endpoint < movable_count)
        for axis in range(2):
            diagonal[axis].scatter_add_(0, endpoint[mask], pair_diagonal[mask])
    return diagonal


def quadratic_rowsum_diagonal(
    weight,
    load,
    relative_xy,
    *,
    coefficient,
    coordinate_to_um,
    gamma,
    driver_nodes,
    sink_nodes,
    movable_count,
    node_count,
):
    """Gershgorin diagonal for K*||dx,dy||^2 + B*L.

    The squared-Euclidean delay term is separable with per-axis curvature
    2*lambda*|K|.  The separate net-weight load term is an L1 length and uses
    the same one-over-length MM bound as ``linear_length_diagonal``.
    """
    distinct = driver_nodes != sink_nodes
    free_count = (driver_nodes < movable_count).to(weight.dtype)
    free_count = free_count + (sink_nodes < movable_count).to(weight.dtype)
    pair_curvature = (
        coefficient * weight.abs() * 2.0 * distinct * free_count
    )
    if load.numel():
        pair_curvature = pair_curvature + (
            (coefficient * load.clamp(min=0)).unsqueeze(0)
            / (
                coordinate_to_um
                * relative_xy.abs().clamp(min=float(gamma))
            )
            * distinct
            * free_count
        )
    diagonal = weight.new_zeros((2, node_count))
    for endpoint in (driver_nodes, sink_nodes):
        mask = distinct & (endpoint < movable_count)
        for axis in range(2):
            diagonal[axis].scatter_add_(
                0,
                endpoint[mask],
                pair_curvature[axis, mask],
            )
    return diagonal


def rowsum_curvature_load(star, enabled):
    """Select the load tensor used by the row-sum C channel."""
    load = (
        getattr(star, "frozen_curvature_load_length_weight", None)
        if enabled
        else None
    )
    if load is None and enabled:
        load = getattr(star, "frozen_load_length_weight", None)
    if load is None:
        load = star.frozen_weight.new_empty(0)
    return load.detach()


def rowsum_hessian(owner, pos, *, elmore_axes=1, full_slope=False):
    """Build the exact row-sum diagonal for the data-path Star timing term."""
    star = owner._star_pin_pair_op()
    pinmap = owner.data_collections.pin2node_map.long()
    movable_count = int(owner.placedb.num_movable_nodes)
    node_count = int(owner.placedb.num_nodes)
    weight = star.frozen_weight.detach()
    curvature_enabled_fn = getattr(
        owner, "_segment_timing_star_load_curvature", None
    )
    curvature_enabled = (
        bool(curvature_enabled_fn()) if callable(curvature_enabled_fn) else True
    )
    load = rowsum_curvature_load(star, curvature_enabled)
    if load.numel() == 0:
        load = torch.zeros_like(weight)
    elif load.shape != weight.shape:
        raise RuntimeError("Star load coefficient shape differs from delay weight")
    coefficient = float(owner.segment_timing_coefficient)
    gamma_raw = owner._segment_timing_cut_effective_smoothing_gamma()
    if gamma_raw is None:
        gamma_raw = getattr(owner, "gamma", 1.0)
    if torch.is_tensor(gamma_raw):
        gamma = float(gamma_raw.detach().reshape(-1)[0].item())
    else:
        gamma = float(gamma_raw)
    pin_pos = owner.op_collections.pin_pos_op(pos)
    geometry = star.geometry(pin_pos)
    driver_nodes = pinmap[star.driver_pin_ids]
    sink_nodes = pinmap[star.sink_pin_ids]
    is_wa_net_load = getattr(star, "load_geometry", "star") == "wa_net"
    pair_load = torch.zeros_like(weight) if is_wa_net_load else load
    pair_objective_fn = getattr(
        owner, "_segment_timing_star_pair_objective", None
    )
    pair_objective = (
        str(pair_objective_fn()) if callable(pair_objective_fn) else "elmore"
    )
    if pair_objective == "efftdp_quadratic":
        diagonal = quadratic_rowsum_diagonal(
            weight,
            pair_load,
            torch.stack((geometry["raw_dx"], geometry["raw_dy"])),
            coefficient=coefficient,
            coordinate_to_um=star.coordinate_to_um,
            gamma=gamma,
            driver_nodes=driver_nodes,
            sink_nodes=sink_nodes,
            movable_count=movable_count,
            node_count=node_count,
        )
        if is_wa_net_load and bool((load != 0).any().item()) and coefficient > 0.0:
            wa_dx, wa_dy = star.net_wa_load_curvature_diagonal(
                pin_pos,
                load=load.clamp(min=0.0) * coefficient,
                smoothing_gamma=gamma,
                pin2node_map=pinmap,
                num_movable_nodes=movable_count,
                num_nodes=node_count,
                output_template=diagonal[0],
                mm_secant=True,
            )
            diagonal = diagonal + torch.stack((wa_dx, wa_dy))
        return diagonal, gamma
    base_caps = owner._driver_sink_star_base_cap_map()
    h_elmore = elmore_rowsum_diagonal(
        weight,
        coefficient=coefficient,
        r_unit=star.r_unit,
        c_unit=star.c_unit,
        coordinate_to_um=star.coordinate_to_um,
        driver_nodes=driver_nodes,
        sink_nodes=sink_nodes,
        movable_count=movable_count,
        node_count=node_count,
        axes=elmore_axes,
    )
    relative_xy = torch.stack((geometry["raw_dx"], geometry["raw_dy"]))
    sink_cap = base_caps["generic"][star.sink_pin_ids]
    h_linear = linear_length_diagonal(
        weight,
        pair_load,
        sink_cap,
        relative_xy,
        coefficient=coefficient,
        r_unit=star.r_unit,
        coordinate_to_um=star.coordinate_to_um,
        gamma=gamma,
        driver_nodes=driver_nodes,
        sink_nodes=sink_nodes,
        movable_count=movable_count,
        node_count=node_count,
        c_unit=star.c_unit,
        length_um=geometry["length_um"] if full_slope else None,
    )
    total_diag = h_elmore + h_linear
    if is_wa_net_load and bool((load != 0).any().item()) and coefficient > 0.0:
        wa_dx, wa_dy = star.net_wa_load_curvature_diagonal(
            pin_pos,
            load=load.clamp(min=0.0) * coefficient,
            smoothing_gamma=gamma,
            pin2node_map=pinmap,
            num_movable_nodes=movable_count,
            num_nodes=node_count,
            output_template=total_diag[0],
            mm_secant=True,
        )
        total_diag = total_diag + torch.stack((wa_dx, wa_dy))
    return total_diag, gamma


def clock_rowsum_hessian(owner, pos, *, gamma, elmore_axes=1, full_slope=False):
    """Row-sum diagonal of the LCB->FF clock Star term (zero when clock is off)."""
    star = getattr(owner.op_collections, "clock_star_pin_pair_op", None)
    mode_fn = getattr(owner, "_segment_timing_clock_weight_config", None)
    mode, scale = mode_fn() if callable(mode_fn) else ("off", 0.0)
    node_count = int(owner.placedb.num_nodes)
    if star is None or mode == "off" or scale == 0.0:
        return pos.new_zeros((2, node_count))
    weight = star.frozen_weight.detach()
    if int(weight.numel()) != int(star.sink_pin_ids.numel()):
        return pos.new_zeros((2, node_count))
    pinmap = owner.data_collections.pin2node_map.long()
    native = star.compact_to_native_pin_ids.long()
    driver_nodes = pinmap[native[star.driver_pin_ids.long()]]
    sink_nodes = pinmap[native[star.sink_pin_ids.long()]]
    movable_count = int(owner.placedb.num_movable_nodes)
    coefficient = float(owner.segment_timing_coefficient)
    compact = owner._lcb_ff_clock_compact_pin_pos(owner.op_collections.pin_pos_op(pos))
    geometry = star.geometry(compact)
    base_caps = owner._lcb_ff_clock_base_cap_map()
    load = getattr(star, "frozen_load_length_weight", None)
    load = weight.new_empty(0) if load is None else load.detach()
    h_elmore = elmore_rowsum_diagonal(
        weight,
        coefficient=coefficient,
        r_unit=star.r_unit,
        c_unit=star.c_unit,
        coordinate_to_um=star.coordinate_to_um,
        driver_nodes=driver_nodes,
        sink_nodes=sink_nodes,
        movable_count=movable_count,
        node_count=node_count,
        axes=elmore_axes,
    )
    h_linear = linear_length_diagonal(
        weight,
        load if load.numel() == weight.numel() else weight.new_empty(0),
        base_caps["generic"][star.sink_pin_ids.long()],
        torch.stack((geometry["raw_dx"], geometry["raw_dy"])),
        coefficient=coefficient,
        r_unit=star.r_unit,
        coordinate_to_um=star.coordinate_to_um,
        gamma=gamma,
        driver_nodes=driver_nodes,
        sink_nodes=sink_nodes,
        movable_count=movable_count,
        node_count=node_count,
        c_unit=star.c_unit,
        length_um=geometry["length_um"] if full_slope else None,
    )
    return (h_elmore + h_linear).to(pos.dtype)


def transport_momentum(u_next, v_next, diagonal_used, diagonal_next, envelope):
    """Rescale the stored Nesterov velocity w = v - u when the diagonal grows."""
    ratio = diagonal_next / diagonal_used
    hit = ratio > envelope
    if not bool(hit.any()):
        return 0, 1.0
    factor = torch.ones_like(ratio)
    factor[hit] = torch.sqrt(envelope / ratio[hit])
    factor = factor.to(u_next.dtype)
    u_next.copy_(v_next - factor * (v_next - u_next))
    return int(hit.sum()), float(factor.min())


def timing_transport_factor(
    base_used, timing_used, alpha_used, timing_next, alpha_next
):
    """Per-coordinate velocity factor of the timing-curvature transport."""
    used = (base_used + timing_used).clamp(min=1.0)
    hypothetical = (
        base_used + (alpha_used / alpha_next) * timing_next
    ).clamp(min=1.0)
    return torch.where(
        hypothetical > used,
        torch.sqrt(used / hypothetical),
        torch.ones_like(used),
    )


class InputAccumulator:
    """Accumulate, replace, or exponentially forget timing-input adjoints."""

    def __init__(
        self,
        gain=0.2,
        half_life=0.0,
        mode="accumulate",
        max_carry_ratio=0.0,
        reference_interval=0.0,
    ):
        self.gain = float(gain)
        self.half_life = float(half_life)
        self.mode = str(mode or "accumulate").strip().lower()
        self.max_carry_ratio = float(max_carry_ratio or 0.0)
        self.reference_interval = float(reference_interval or 0.0)
        if self.mode not in ("accumulate", "replace", "ema"):
            raise ValueError("history mode must be accumulate, replace, or ema")
        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("input gain must be finite and positive")
        if not math.isfinite(self.half_life) or self.half_life < 0:
            raise ValueError("half-life must be finite and nonnegative")
        if not math.isfinite(self.max_carry_ratio) or self.max_carry_ratio < 0:
            raise ValueError("max_carry_ratio must be finite and nonnegative")
        if not math.isfinite(self.reference_interval) or self.reference_interval < 0:
            raise ValueError("reference_interval must be finite and nonnegative")
        if self.mode == "ema" and self.half_life <= 0:
            raise ValueError("EMA history requires a positive half-life")
        self.identity = None
        self.iteration = None
        self.generation = 0
        self.values = None

    def compute_factors(self, elapsed, *, is_initial=False):
        if self.mode == "replace":
            return 0.0, 1.0
        if self.mode == "ema":
            rho = 1.0 if is_initial else 2.0 ** (-float(elapsed) / self.half_life)
            if (not is_initial) and self.reference_interval > 0.0:
                if float(elapsed) == self.reference_interval:
                    update_scale = self.gain
                else:
                    denom = 1.0 - 2.0 ** (-self.reference_interval / self.half_life)
                    if denom > 1.0e-15:
                        update_scale = self.gain * ((1.0 - rho) / denom)
                    else:
                        update_scale = self.gain * (
                            float(elapsed) / self.reference_interval
                        )
            else:
                update_scale = self.gain
            return rho, update_scale
        rho = 1.0
        if (not is_initial) and self.reference_interval > 0.0:
            if float(elapsed) == self.reference_interval:
                update_scale = self.gain
            else:
                update_scale = self.gain * (
                    float(elapsed) / self.reference_interval
                )
        else:
            update_scale = self.gain
        return rho, update_scale

    def prepare(
        self,
        fresh,
        *,
        identity,
        pin_count,
        iteration,
        carry_beta=None,
        allow_identity_change=False,
    ):
        if (
            self.identity is not None
            and identity != self.identity
            and not bool(allow_identity_change)
        ):
            raise ValueError("design-pin identity changed")
        if self.iteration is not None and iteration <= self.iteration:
            raise ValueError("history may advance only at a new scheduled iteration")
        is_initial = self.iteration is None
        elapsed = 0 if is_initial else iteration - self.iteration
        rho, update_scale = self.compute_factors(elapsed, is_initial=is_initial)
        values = {}
        old_total_l1 = 0.0
        carried_total_l1 = 0.0
        old_delay_l1 = 0.0
        carried_delay_l1 = 0.0
        for group, _ in CHANNELS:
            values[group] = {}
            group_beta_map = None
            if carry_beta is not None:
                if group in carry_beta and isinstance(carry_beta[group], dict):
                    group_beta_map = carry_beta[group]
                else:
                    group_beta_map = carry_beta
            for edge in EDGES:
                current = fresh[group][edge].detach()
                if current.ndim != 1 or current.numel() < pin_count:
                    raise ValueError("missing stable design-pin prefix")
                if not bool(torch.isfinite(current).all()):
                    raise ValueError("nonfinite timing input adjoint")
                if bool((current[pin_count:] != 0).any()):
                    raise ValueError("cannot discard nonzero Steiner-vertex adjoints")
                updated = current[:pin_count] * update_scale
                if self.values is not None:
                    old = self.values[group][edge]
                    if old.shape != updated.shape or old.device != updated.device:
                        raise ValueError("history shape or device changed")
                    carried = rho * old
                    if group_beta_map is not None and edge in group_beta_map:
                        beta_vec = group_beta_map[edge].detach().to(
                            device=carried.device, dtype=carried.dtype
                        )
                        if beta_vec.shape != carried.shape:
                            raise ValueError("carry_beta shape differs from history")
                        carried = carried * beta_vec
                    old_edge_l1 = float(old.abs().sum().item())
                    carried_edge_l1 = float(carried.abs().sum().item())
                    old_total_l1 += old_edge_l1
                    carried_total_l1 += carried_edge_l1
                    if group == "delay":
                        old_delay_l1 += old_edge_l1
                        carried_delay_l1 += carried_edge_l1
                    if self.max_carry_ratio > 0.0:
                        fresh_l1 = float(updated.abs().sum().item())
                        carried_l1 = float(carried.abs().sum().item())
                        if (
                            fresh_l1 > 0.0
                            and carried_l1 > self.max_carry_ratio * fresh_l1
                        ):
                            carried = carried * (
                                (self.max_carry_ratio * fresh_l1) / carried_l1
                            )
                    updated = updated + carried
                if not bool(torch.isfinite(updated).all()):
                    raise ValueError("input accumulation overflow")
                values[group][edge] = updated
        beta_carry_ratio = (
            carried_total_l1 / old_total_l1 if old_total_l1 > 0.0 else None
        )
        delay_beta_carry_ratio = (
            carried_delay_l1 / old_delay_l1 if old_delay_l1 > 0.0 else None
        )
        return dict(
            values=values,
            identity=identity,
            iteration=iteration,
            generation=self.generation + 1,
            rho=rho,
            elapsed=elapsed,
            beta_carry_ratio=beta_carry_ratio,
            delay_beta_carry_ratio=delay_beta_carry_ratio,
            allow_identity_change=bool(allow_identity_change),
        )

    def commit(self, candidate):
        if candidate["generation"] != self.generation + 1:
            raise ValueError("stale history publication")
        if (
            self.identity is not None
            and self.identity != candidate["identity"]
            and not bool(candidate.get("allow_identity_change", False))
        ):
            raise ValueError("history identity changed at publication")
        self.values = candidate["values"]
        self.identity = candidate["identity"]
        self.iteration = candidate["iteration"]
        self.generation = candidate["generation"]


def apply_satisfied_slack_release(
    fresh,
    values,
    slack,
    *,
    mode: str = "off",
    threshold_ps: float = 60.0,
    band_ps: float = 140.0,
    tau_mult: float = 2.0,
    merge_tau_ps=None,
    half_life_iters: float = 45.0,
    elapsed_iters: float = 15.0,
):
    """Step-invariant positive-slack KKT multiplier release on ``values`` (in place)."""
    rel_mode = str(mode or "off").strip().lower()
    if rel_mode == "off" or slack is None or not torch.is_tensor(slack):
        return None
    if rel_mode not in ("hard", "smooth"):
        raise ValueError(f"Unsupported satisfied slack release mode: {mode!r}")

    n = int(values["delay"]["rise"].numel())
    if n <= 0:
        return None
    q = sum(
        fresh[group][edge][:n].detach().double().abs()
        for group, _ in CHANNELS
        for edge in EDGES
        if group in fresh and edge in fresh[group]
    )
    s = slack.detach().double().reshape(-1)[:n].to(q.device)
    s_start = max(float(threshold_ps), 0.0)
    s_band = max(float(band_ps), 1.0e-6)
    if (
        merge_tau_ps is not None
        and math.isfinite(float(merge_tau_ps))
        and float(merge_tau_ps) > 0.0
        and float(tau_mult) > 0.0
    ):
        tau_f = float(merge_tau_ps)
        tau_cap = max(4.0 * max(float(threshold_ps), 50.0), 400.0)
        eff_tau_guard = min(float(tau_mult) * tau_f, tau_cap)
        s_start = max(s_start, eff_tau_guard)
        s_band = max(s_band, 1.5 * eff_tau_guard)

    sentinel = 8.0e7
    cand = (q == 0) & torch.isfinite(s) & (s.abs() < sentinel) & (s > s_start)
    before = float(sum(values["delay"][edge].double().abs().sum() for edge in EDGES))

    if rel_mode == "hard":
        keep = torch.where(cand, torch.zeros_like(s), torch.ones_like(s))
    else:
        u = ((s - s_start) / s_band).clamp(0.0, 1.0)
        smooth_u = u * u * (3.0 - 2.0 * u)
        dt = max(
            float(elapsed_iters if (elapsed_iters is not None and elapsed_iters > 0) else 15.0),
            1.0,
        )
        hl = max(float(half_life_iters), 1.0)
        max_drop = 1.0 - (2.0 ** (-dt / hl))
        keep = torch.where(cand, 1.0 - smooth_u * max_drop, torch.ones_like(s))

    for group, _ in CHANNELS:
        for edge in EDGES:
            target = values[group][edge]
            values[group][edge] = target * keep.to(device=target.device, dtype=target.dtype)

    after = float(sum(values["delay"][edge].double().abs().sum() for edge in EDGES))
    return dict(
        mode=rel_mode,
        s_start_ps=float(s_start),
        s_band_ps=float(s_band),
        elapsed_iters=float(elapsed_iters or 0.0),
        half_life_iters=float(half_life_iters),
        released_pins=int(cand.sum().item()),
        released_delay_share=(max(before - after, 0.0) / before if before > 0.0 else 0.0),
    )
