"""Transport immutable Python TP inputs to the native propagation owner."""

import numpy as np
import torch


def _array(value):
    if value is None:
        return None
    if value.device.type != "cpu":
        raise ValueError("native timing propagation requires CPU inputs")
    return np.array(value.detach().numpy(), order="C", copy=True)


def persistent_propagation_payload(owner):
    """Export original constraints and tables without performing propagation."""
    from dreamplace.ops.timing_propagation.timing_propagation import (
        SLEW_SQRT_EPS,
        SMOOTH_MAX_ALPHA,
        SMOOTH_MAX_ENABLED,
        _force_hard_pairwise_max_enabled,
    )

    timing = owner.model
    if int(owner.pydb.timing_schema_version) != 2 or timing._sparse_runtime is not None:
        raise ValueError("native timing propagation requires schema 2")
    if timing.num_pins != len(owner.pydb.pin_names) or timing.dtype != torch.float64:
        raise ValueError("native timing propagation requires the current float64 carrier")
    fields = {
        "cell_arcs": "flat_inst_arcs_by_level",
        "cell_level_start": "flat_inst_arcs_by_level_start",
        "net_arcs": "net_flat_arcs",
        "net_arc_start": "net_flat_arcs_start",
        "pin_net": "pin_net",
        "net_drivers": "net2driver_pin_map",
        "clock_pins": "clock_pins",
        "clock_rtran": "clk_pin_rtran",
        "clock_ftran": "clk_pin_ftran",
        "start_pins": "start_points",
        "start_rAAT": "inrdelays",
        "start_fAAT": "infdelays",
        "start_rtran": "inrtrans",
        "start_ftran": "inftrans",
        "endpoint_pins": "end_points",
        "endpoint_rRAT": "endpoints_rRAT",
        "endpoint_fRAT": "endpoints_fRAT",
        "setup_arcs": "endpoints_constraint_arcs",
        "setup_valid": "endpoints_constraint_max_valid",
        "check_arcs": "endpoints_timing_check_arcs",
        "check_valid": "endpoints_timing_check_max_valid",
        "endpoint_valid": "endpoint_max_valid_by_pin",
        "backend_rSlew": "backend_endpoint_rSlew",
        "backend_fSlew": "backend_endpoint_fSlew",
    }
    payload = {name: _array(getattr(timing, source)) for name, source in fields.items()}
    payload.update(
        schema_version=1,
        timing_schema_version=2,
        pin_names=[str(name) for name in owner.pydb.pin_names],
        aggregation_mode=timing.timing_aggregation_mode,
        metric_scope=timing.timing_metric_scope,
        recovery_uses_live_slew=bool(timing.recovery_check_uses_live_slew),
        recovery_rat_is_capture_deadline=bool(
            getattr(timing, "recovery_rat_is_capture_deadline", False)
        ),
        pairwise_smooth=bool(SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled()),
        pairwise_alpha=float(SMOOTH_MAX_ALPHA),
        slew_epsilon=float(SLEW_SQRT_EPS),
    )
    tables = []
    for name in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        table = getattr(timing.arcs_info, f"{name}_luts")
        if table.use_2d_coeff_cache or table.use_2d_coeff_op:
            raise ValueError("native timing propagation requires raw LUT tables")
        tables.append(
            [
                _array(table.flat_luts_values),
                _array(table.flat_luts_trans_table),
                _array(table.flat_luts_cap_table),
                _array(table.trans_dims_actual),
                _array(table.cap_dims_actual),
            ]
        )
    payload["luts"] = tables
    return payload
