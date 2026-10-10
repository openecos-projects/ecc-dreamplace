"""Recovery/removal check handling for DreamPlace timing propagation."""

import torch


TIMING_CHECK_CLASS_SETUP = 1
TIMING_CHECK_CLASS_RECOVERY = 3


def timing_metric_endpoint_mask(timing, device):
    """Select endpoints included in the configured max timing metric."""
    endpoint_mask = torch.ones(
        timing.end_points.numel(), dtype=torch.bool, device=device
    )
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
        timing_check_arcs[
            timing_check_arcs[:, 6] == TIMING_CHECK_CLASS_SETUP, 1
        ].long()
    )
    constraints = getattr(timing, "endpoints_constraint_arcs", None)
    if (
        constraints is not None
        and torch.is_tensor(constraints)
        and constraints.numel() > 0
        and constraints.dim() == 2
        and constraints.size(1) > 1
    ):
        setup_pins = torch.unique(
            torch.cat((setup_pins, constraints[:, 1].long()))
        )

    recovery_only = torch.isin(timing.end_points.long(), recovery_pins)
    recovery_only &= ~torch.isin(timing.end_points.long(), setup_pins)
    return ~recovery_only


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
    if timing.backend_endpoint_rSlew is not None and timing.backend_endpoint_rSlew.numel() == timing.end_points.numel():
        native_r_by_pin = torch.zeros(timing.num_pins, device=timing.device, dtype=timing.dtype)
        native_f_by_pin = torch.zeros_like(native_r_by_pin)
        native_r_by_pin[timing.end_points.long()] = timing.backend_endpoint_rSlew
        native_f_by_pin[timing.end_points.long()] = timing.backend_endpoint_fSlew
        native_r = native_r_by_pin[endpoint_pins]
        native_f = native_f_by_pin[endpoint_pins]

    # Exported check LUTs use (related/data transition, constrained/clock
    # transition), matching native iSTA's TimingTable::findValue query order.
    native_check_r = timing.lut_entry_vectorized(
        lib_cells, native_r, clock_r_slew, lib_arcs, timing.arcs_info.r_check_luts
    )
    native_check_f = timing.lut_entry_vectorized(
        lib_cells, native_f, clock_f_slew, lib_arcs, timing.arcs_info.f_check_luts
    )
    # Native iSTA recovery RAT is capture_time + check_time.
    capture_r = pin_rRAT[endpoint_pins] - native_check_r
    capture_f = pin_fRAT[endpoint_pins] - native_check_f

    live_check_r = timing.lut_entry_vectorized(
        lib_cells, pin_rtran[endpoint_pins], clock_r_slew, lib_arcs, timing.arcs_info.r_check_luts
    )
    live_check_f = timing.lut_entry_vectorized(
        lib_cells, pin_ftran[endpoint_pins], clock_f_slew, lib_arcs, timing.arcs_info.f_check_luts
    )
    new_r = capture_r + live_check_r
    new_f = capture_f + live_check_f
    pin_rRAT = scatter_timing_min(
        pin_rRAT,
        endpoint_pins,
        new_r,
        include_self=True,
        mode=timing.timing_aggregation_mode,
        tau_ps=timing.timing_aggregation_tau_ps,
    )
    pin_fRAT = scatter_timing_min(
        pin_fRAT,
        endpoint_pins,
        new_f,
        include_self=True,
        mode=timing.timing_aggregation_mode,
        tau_ps=timing.timing_aggregation_tau_ps,
    )
    return pin_rRAT, pin_fRAT
