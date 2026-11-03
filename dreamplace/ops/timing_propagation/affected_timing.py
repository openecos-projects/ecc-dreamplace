"""Execute native affected worklists using TimingPropagation's electrical ops.

The native graph owner computes closure, ordering and identity remaps. This
adapter gathers the old state, resets affected destinations, and invokes the
same cell/net/constraint operators as full TimingPropagation.
"""

import copy
import time

import torch

from .endpoint_qualification import endpoint_metrics, qualify_pin_slacks
from .net_aat_native import try_net_aat_native


def _remap(values, mapping, fill):
    if not values.numel():
        return values.new_full(mapping.shape, fill)
    return torch.where(mapping >= 0, values[mapping.clamp_min(0)], fill)


def _levels(schedule, prefix):
    cells, nets = schedule[f"{prefix}_cells"], schedule[f"{prefix}_nets"]
    cell_start, net_start = schedule[f"{prefix}_cell_start"], schedule[f"{prefix}_net_start"]
    for index in range(len(cell_start) - 1):
        yield (
            cells[int(cell_start[index]) : int(cell_start[index + 1])],
            nets[int(net_start[index]) : int(net_start[index + 1])],
        )


def _check_view(timing, schedule):
    view = copy.copy(timing)
    view._critical_path_snapshot_requested = False
    for field, indices in (
        ("endpoints_constraint", schedule["setup_rows"]),
        ("endpoints_timing_check", schedule["check_rows"]),
    ):
        setattr(view, f"{field}_arcs", getattr(timing, f"{field}_arcs")[indices])
        valid = getattr(timing, f"{field}_max_valid")
        setattr(view, f"{field}_max_valid", None if valid is None else valid[indices])
    return view


def forward_affected(timing, previous, schedule, delays, impulses, caps):
    """Update AAT/slew, reverse RAT and endpoint metrics on the current graph."""
    if timing.device.type != "cpu" or timing.timing_aggregation_mode != "hard":
        raise ValueError("affected timing currently requires CPU hard aggregation")
    if timing.sparse_timing_view is not None or previous.sparse_timing_view is not None:
        raise ValueError("affected timing currently requires a schema-2 physical pin graph")
    started = time.perf_counter()
    pin_map, arc_map = schedule["pin_from_previous"], schedule["arc_from_previous"]
    fwd, bwd = schedule["forward_pins"], schedule["backward_pins"]
    ra, fa, rs, fs = (
        _remap(getattr(previous, name), pin_map, fill)
        for name, fill in (
            ("pin_rAAT", -2e8),
            ("pin_fAAT", -2e8),
            ("pin_rtran", 0.0),
            ("pin_ftran", 0.0),
        )
    )
    rr, fr = (_remap(getattr(previous, name), pin_map, 2e8) for name in ("pin_rRAT", "pin_fRAT"))
    cell_delays = [
        _remap(getattr(previous, f"cell_arc_{name}_delays"), arc_map, 0.0)
        for name in ("rr", "fr", "rf", "ff")
    ]
    ra[fwd] = fa[fwd] = -2e8
    rs[fwd] = fs[fwd] = 0.0
    starts = schedule["forward_start_rows"]
    pins = timing.start_points[starts]
    ra[pins], fa[pins] = timing.inrdelays[starts], timing.infdelays[starts]
    rs[pins], fs[pins] = timing.inrtrans[starts], timing.inftrans[starts]
    clock_ids = schedule["clock_rows"]
    if clock_ids.numel():
        clock_view = copy.copy(timing)
        clock_view.flat_inst_arcs_by_level = timing.flat_inst_arcs_by_level[clock_ids]
        clock_view.flat_inst_arcs_by_level_start = [0, len(clock_ids)]
        _, ra, fa, rs, fs = clock_view.calculate_clk2q_aat(
            ra, fa, rs, fs, caps["rise"], caps["fall"], use_surrogate=False
        )
    for cell_ids, net_ids in _levels(schedule, "forward"):
        if cell_ids.numel():
            result = timing.calculate_cell_aat_level(
                timing.flat_inst_arcs_by_level[cell_ids],
                ra,
                fa,
                rs,
                fs,
                caps["rise"],
                caps["fall"],
                use_surrogate=False,
                need_delay_updates=True,
            )
            _, ra, fa, rs, fs, *updates = result
            for destination, values in zip(cell_delays, updates, strict=True):
                destination[cell_ids] = values
        if net_ids.numel():
            from .timing_propagation import SLEW_SQRT_EPS

            result = try_net_aat_native(
                timing,
                timing.net_flat_arcs[net_ids],
                (
                    ra,
                    fa,
                    rs,
                    fs,
                    delays["rise"],
                    delays["fall"],
                    impulses["rise"],
                    impulses["fall"],
                ),
                SLEW_SQRT_EPS,
            )
            if result is None:
                raise RuntimeError("affected timing requires the existing native net AAT operator")
            ra, fa, rs, fs = result
    rr[bwd] = fr[bwd] = 2e8
    endpoint_rows = schedule["backward_endpoint_rows"]
    endpoints = timing.end_points[endpoint_rows]
    rr[endpoints], fr[endpoints] = (
        timing.endpoints_rRAT[endpoint_rows],
        timing.endpoints_fRAT[endpoint_rows],
    )
    checks = _check_view(timing, schedule)
    rr, fr = checks.calculate_setup_rat(rr, fr, timing.clk_pin_rtran, timing.clk_pin_ftran, rs, fs)
    rr, fr = checks.calculate_recovery_rat(rr, fr, rs, fs)
    for cell_ids, net_ids in _levels(schedule, "backward"):
        if cell_ids.numel():
            _, rr, fr = timing.calculate_cell_rat_level(
                timing.flat_inst_arcs_by_level[cell_ids],
                rr,
                fr,
                *(values[cell_ids] for values in cell_delays),
            )
        if net_ids.numel():
            rr, fr = timing.calculate_net_rat_level(
                timing.net_flat_arcs[net_ids, 1], rr, fr, delays["rise"], delays["fall"]
            )
    rise_slack, fall_slack = rr - ra, fr - fa
    slack = torch.minimum(rise_slack, fall_slack)
    endpoint_ids = timing.end_points[timing._timing_metric_endpoint_mask(timing.device)]
    rise, fall = qualify_pin_slacks(
        timing, endpoint_ids, rise_slack[endpoint_ids], fall_slack[endpoint_ids]
    )
    timing.last_endpoint_slack_tensor = torch.minimum(rise, fall)
    timing.last_endpoint_ids_tensor = endpoint_ids.clone()
    timing.timing_forward_count = previous.timing_forward_count + 1
    timing.last_fast_loop_skipped_full_rat = False
    timing._store_forward_state(
        pin_rAAT=ra,
        pin_fAAT=fa,
        pin_rRAT=rr,
        pin_fRAT=fr,
        pin_rtran=rs,
        pin_ftran=fs,
        pin_net_cap_rise=caps["rise"],
        pin_net_cap_fall=caps["fall"],
        pin_net_delay_rise=delays["rise"],
        pin_net_delay_fall=delays["fall"],
        pin_net_impulse_rise=impulses["rise"],
        pin_net_impulse_fall=impulses["fall"],
        rslack=rise_slack,
        fslack=fall_slack,
        slack=slack,
        **{
            f"cell_arc_{name}_delays": values
            for name, values in zip(("rr", "fr", "rf", "ff"), cell_delays, strict=True)
        },
    )
    timing.last_incremental_changed_pins = torch.unique(
        torch.cat((fwd, bwd), dim=0)
    ).detach()
    timing.last_incremental_profile = {
        "forward_pins": fwd.numel(),
        "backward_pins": bwd.numel(),
        "total_pins": timing.num_pins,
        "forward_cell_arcs": schedule["forward_cells"].numel(),
        "backward_cell_arcs": schedule["backward_cells"].numel(),
        "total_cell_arcs": timing.flat_inst_arcs_by_level.shape[0],
        "runtime_ms": (time.perf_counter() - started) * 1000,
    }
    return (*endpoint_metrics(timing.last_endpoint_slack_tensor, ra.sum() * 0), 0)
