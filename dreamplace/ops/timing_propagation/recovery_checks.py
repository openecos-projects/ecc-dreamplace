"""Recovery/removal check handling for DreamPlace timing propagation."""

import torch

TIMING_CHECK_CLASS_SETUP = 1
TIMING_CHECK_CLASS_RECOVERY = 3


def timing_metric_endpoint_mask(timing, device):
    """Select endpoints included in the configured max timing metric."""
    endpoint_mask = torch.ones(timing.end_points.numel(), dtype=torch.bool, device=device)
    if timing.timing_metric_scope == "setup_plus_recovery":
        return endpoint_mask

    timing_check_arcs = getattr(timing, "endpoints_timing_check_arcs", None)
    if (
        timing_check_arcs is None
        or not torch.is_tensor(timing_check_arcs)
        or timing_check_arcs.numel() == 0
        or timing_check_arcs.dim() != 2
        or timing_check_arcs.size(1) <= 6
    ):
        return endpoint_mask

    recovery_mask = timing_check_arcs[:, 6] == TIMING_CHECK_CLASS_RECOVERY
    check_valid = getattr(timing, "endpoints_timing_check_max_valid", None)
    if check_valid is not None:
        recovery_mask = recovery_mask & check_valid.any(dim=1)
    recovery_pins = torch.unique(timing_check_arcs[recovery_mask, 1].long())
    if recovery_pins.numel() == 0:
        return endpoint_mask

    setup_pins = torch.unique(
        timing_check_arcs[timing_check_arcs[:, 6] == TIMING_CHECK_CLASS_SETUP, 1].long()
    )
    constraints = getattr(timing, "endpoints_constraint_arcs", None)
    if (
        constraints is not None
        and torch.is_tensor(constraints)
        and constraints.numel() > 0
        and constraints.dim() == 2
        and constraints.size(1) > 1
    ):
        setup_pins = torch.unique(torch.cat((setup_pins, constraints[:, 1].long())))

    recovery_only = torch.isin(timing.end_points.long(), recovery_pins)
    recovery_only &= ~torch.isin(timing.end_points.long(), setup_pins)
    return ~recovery_only


def _validate_native_endpoint_slew_contract(timing):
    """Require native endpoint slew data for recovery-aware timing loss."""
    if getattr(timing, "timing_metric_scope", "setup_plus_recovery") != "setup_plus_recovery":
        return

    expected = int(timing.end_points.numel())
    rise = getattr(timing, "backend_endpoint_rSlew", None)
    fall = getattr(timing, "backend_endpoint_fSlew", None)
    if not torch.is_tensor(rise) or not torch.is_tensor(fall):
        raise ValueError(
            "setup_plus_recovery requires native endpoint slew tensors "
            "backend_endpoint_rSlew and backend_endpoint_fSlew"
        )
    if rise.dim() != 1 or fall.dim() != 1 or rise.numel() != expected or fall.numel() != expected:
        raise ValueError(
            "setup_plus_recovery requires one native rise/fall slew per end_point "
            f"(expected {expected}, got rise={rise.numel()}, fall={fall.numel()})"
        )


def calculate_recovery_rat(timing, pin_rRAT, pin_fRAT, pin_rtran, pin_ftran, scatter_timing_min):
    """Recompute max recovery RAT from live data slew and check LUTs."""
    arcs = timing.endpoints_timing_check_arcs
    if arcs is None or arcs.numel() == 0 or arcs.shape[1] <= 6:
        return pin_rRAT, pin_fRAT
    recovery = arcs[:, 6] == TIMING_CHECK_CLASS_RECOVERY
    valid = getattr(timing, "endpoints_timing_check_max_valid", None)
    if valid is not None:
        recovery = recovery & valid.any(dim=1)
    if not torch.any(recovery):
        return pin_rRAT, pin_fRAT
    _validate_native_endpoint_slew_contract(timing)

    rows = arcs[recovery]
    clock_pins = rows[:, 0].long()
    endpoint_pins = rows[:, 1].long()
    lib_cells = rows[:, 2].long()
    lib_arcs = rows[:, 3].long()
    clock_type = rows[:, 5]

    clock_r = torch.zeros(timing.num_pins, device=timing.device, dtype=timing.dtype)
    clock_f = torch.zeros_like(clock_r)
    clock_r[timing.clock_pins.long()] = timing.clk_pin_rtran
    clock_f[timing.clock_pins.long()] = timing.clk_pin_ftran
    clock_r_slew = torch.where(clock_type < 0, clock_f[clock_pins], clock_r[clock_pins])
    clock_f_slew = torch.where(clock_type < 0, clock_f[clock_pins], clock_r[clock_pins])

    native_r = pin_rtran[endpoint_pins]
    native_f = pin_ftran[endpoint_pins]
    if getattr(timing, "timing_metric_scope", "setup_plus_recovery") == "setup_plus_recovery":
        native_r_by_pin = torch.zeros(timing.num_pins, device=timing.device, dtype=timing.dtype)
        native_f_by_pin = torch.zeros_like(native_r_by_pin)
        native_r_by_pin[timing.end_points.long()] = timing.backend_endpoint_rSlew
        native_f_by_pin[timing.end_points.long()] = timing.backend_endpoint_fSlew
        native_r = native_r_by_pin[endpoint_pins]
        native_f = native_f_by_pin[endpoint_pins]

    # All exported check LUTs use (related-clock, constrained-data) slew.
    native_check_r = timing.lut_entry_vectorized(
        lib_cells, clock_r_slew, native_r, lib_arcs, timing.arcs_info.r_check_luts
    )
    native_check_f = timing.lut_entry_vectorized(
        lib_cells, clock_f_slew, native_f, lib_arcs, timing.arcs_info.f_check_luts
    )
    # Max required time is capture anchor - check time, including recovery.
    capture_r = pin_rRAT[endpoint_pins] + native_check_r
    capture_f = pin_fRAT[endpoint_pins] + native_check_f

    live_check_r = timing.lut_entry_vectorized(
        lib_cells,
        clock_r_slew,
        pin_rtran[endpoint_pins],
        lib_arcs,
        timing.arcs_info.r_check_luts,
    )
    live_check_f = timing.lut_entry_vectorized(
        lib_cells,
        clock_f_slew,
        pin_ftran[endpoint_pins],
        lib_arcs,
        timing.arcs_info.f_check_luts,
    )
    candidates = (capture_r - live_check_r, capture_f - live_check_f)
    valid = (
        torch.ones((rows.size(0), 2), dtype=torch.bool, device=rows.device)
        if valid is None
        else valid[recovery]
    )
    # Replace the reference check: retaining it would clamp relaxed constraints
    # and count the same constraint twice in smooth mode.
    return tuple(
        scatter_timing_min(
            required,
            endpoint_pins[edge_valid],
            candidate[edge_valid],
            include_self=False,
            mode=timing.timing_aggregation_mode,
            tau_ps=timing.timing_aggregation_tau_ps,
        )
        for required, candidate, edge_valid in zip(
            (pin_rRAT, pin_fRAT), candidates, valid.unbind(dim=1)
        )
    )
