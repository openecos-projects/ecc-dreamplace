"""Opt-in raw-pointer CPU kernels for incremental hard timing propagation."""

import os

import torch

try:
    from . import timing_propagation_cpp
except ImportError:  # pragma: no cover - source trees may not have the extension built
    timing_propagation_cpp = None


def _enabled():
    return os.environ.get("REBUFFER_EVENT_NATIVE_KERNELS", "") == "1"


def _rat_enabled():
    return os.environ.get("REBUFFER_EVENT_NATIVE_RAT", "") == "1"


def _clk2q_enabled():
    return os.environ.get("REBUFFER_EVENT_NATIVE_CLK2Q", "") == "1"


def _hard_mode(timing):
    return str(getattr(timing, "timing_aggregation_mode", "hard")) == "hard"


def _cell_tables(timing):
    from .cell_aat_native import _tables

    try:
        return _tables(timing)
    except (AttributeError, TypeError):
        return None


def try_cell_aat_hard_level(timing, arcs, pin_state, caps):
    if (
        not _enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    tables = _cell_tables(timing)
    if tables is None:
        return None
    ra, fa, rs, fs = pin_state
    cap_rise, cap_fall = caps
    pin_count = ra.numel()
    from .timing_propagation import SMOOTH_MAX_ENABLED, _force_hard_pairwise_max_enabled

    return timing_propagation_cpp.cell_aat_hard_level(
        arcs.contiguous(),
        *(value.contiguous() for value in (ra, fa, rs, fs)),
        cap_rise[:pin_count].contiguous(),
        cap_fall[:pin_count].contiguous(),
        tables,
        SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled(),
        20.0,
    )


def try_net_aat_hard_level(timing, arcs, pin_state, net_state, epsilon):
    if (
        not _enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    ra, fa, rs, fs = pin_state
    delay_rise, delay_fall, impulse_rise, impulse_fall = net_state
    pin_count = ra.numel()
    return timing_propagation_cpp.net_aat_hard_level(
        arcs.contiguous(),
        *(value.contiguous() for value in (ra, fa, rs, fs)),
        delay_rise[:pin_count].contiguous(),
        delay_fall[:pin_count].contiguous(),
        impulse_rise[:pin_count].contiguous(),
        impulse_fall[:pin_count].contiguous(),
        float(epsilon),
    )


def try_cell_rat_hard_level(timing, arcs, pin_rRAT, pin_fRAT, delays):
    if (
        not _rat_enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    return timing_propagation_cpp.cell_rat_hard_level(
        arcs.contiguous(),
        pin_rRAT.contiguous(),
        pin_fRAT.contiguous(),
        *(value.contiguous() for value in delays),
    )


def try_net_rat_hard_level(timing, arcs, pin_rRAT, pin_fRAT, delays):
    if (
        not _rat_enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    pin_count = pin_rRAT.numel()
    return timing_propagation_cpp.net_rat_hard_level(
        arcs.contiguous(),
        pin_rRAT.contiguous(),
        pin_fRAT.contiguous(),
        *(value[:pin_count].contiguous() for value in delays),
    )


def try_setup_rat_hard_level(
    timing,
    arcs,
    pin_rRAT,
    pin_fRAT,
    clk_pin_rtran,
    clk_pin_ftran,
    pin_rtran,
    pin_ftran,
    valid,
):
    if (
        not _rat_enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or not torch.is_tensor(arcs)
        or arcs.numel() == 0
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    from .cell_aat_native import _tables
    from .timing_propagation import SMOOTH_MAX_ENABLED, _force_hard_pairwise_max_enabled

    try:
        all_tables = _tables(timing)
    except (AttributeError, TypeError):
        return None
    if all_tables is None:
        return None
    valid_tensor = (
        torch.empty(0, dtype=torch.bool)
        if valid is None
        else valid.contiguous()
    )
    return timing_propagation_cpp.setup_rat_hard_level(
        arcs.contiguous(),
        pin_rRAT.contiguous(),
        pin_fRAT.contiguous(),
        clk_pin_rtran.contiguous(),
        clk_pin_ftran.contiguous(),
        pin_rtran.contiguous(),
        pin_ftran.contiguous(),
        all_tables[:10],
        valid_tensor,
        SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled(),
        20.0,
    )


def try_recovery_rat_hard_level(
    timing,
    arcs,
    pin_rRAT,
    pin_fRAT,
    pin_rtran,
    pin_ftran,
    clock_rtran,
    clock_ftran,
    check_data_rtran,
    check_data_ftran,
    valid,
):
    if getattr(timing, "recovery_rat_is_capture_deadline", False):
        # The regular check adapter dispatches to the differentiable native
        # deadline kernel, also preserving mixed setup/recovery constraints.
        return None
    if (
        not _rat_enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or not torch.is_tensor(arcs)
        or arcs.numel() == 0
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    from .cell_aat_native import _tables

    try:
        all_tables = _tables(timing)
    except (AttributeError, TypeError):
        return None
    if all_tables is None:
        return None
    return timing_propagation_cpp.recovery_rat_hard_level(
        arcs.contiguous(),
        pin_rRAT.contiguous(),
        pin_fRAT.contiguous(),
        pin_rtran.contiguous(),
        pin_ftran.contiguous(),
        clock_rtran.contiguous(),
        clock_ftran.contiguous(),
        check_data_rtran.contiguous(),
        check_data_ftran.contiguous(),
        all_tables[:10],
        valid.contiguous(),
    )


def try_clk2q_hard_level(timing, arcs, pin_state, caps):
    if (
        not _clk2q_enabled()
        or timing_propagation_cpp is None
        or not _hard_mode(timing)
        or not torch.is_tensor(arcs)
        or arcs.numel() == 0
        or arcs.device.type != "cpu"
        or arcs.dtype != torch.long
    ):
        return None
    tables = _cell_tables(timing)
    if tables is None:
        return None
    from .timing_propagation import SMOOTH_MAX_ENABLED, _force_hard_pairwise_max_enabled

    ra, fa, rs, fs = pin_state
    cap_rise, cap_fall = caps
    pin_count = ra.numel()
    return timing_propagation_cpp.clk2q_hard_level(
        arcs.contiguous(),
        *(value.contiguous() for value in (ra, fa, rs, fs)),
        cap_rise[:pin_count].contiguous(),
        cap_fall[:pin_count].contiguous(),
        timing.clk_pin_rtran.contiguous(),
        timing.clk_pin_ftran.contiguous(),
        tables,
        SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled(),
        20.0,
    )


__all__ = [
    "try_cell_aat_hard_level",
    "try_cell_rat_hard_level",
    "try_net_aat_hard_level",
    "try_net_rat_hard_level",
    "try_setup_rat_hard_level",
    "try_recovery_rat_hard_level",
    "try_clk2q_hard_level",
]
