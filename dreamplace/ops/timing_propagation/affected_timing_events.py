"""Event-driven forward propagation for the CPU hard timing path.

This is the first slice of the incremental propagation plan.  It keeps the
existing Python cell/net kernels and the existing reverse RAT worklist, while
using native per-pin CSR information to avoid evaluating the whole forward
cone when only a small set of pins changed.  The entry point is opt-in and is
always compared against the existing full propagation path by the tests and
the timing oracle.
"""

import copy
import heapq
import os
import time

import torch

from .affected_timing import _remap
from .endpoint_qualification import endpoint_metrics, qualify_pin_slacks
from .net_aat_native import try_net_aat_native
from .timing_kernels_native import (
    try_cell_aat_hard_level,
    try_cell_rat_hard_level,
    try_clk2q_hard_level,
    try_net_aat_hard_level,
    try_net_rat_hard_level,
    try_recovery_rat_hard_level,
    try_setup_rat_hard_level,
)
from .timing_propagation import SLEW_SQRT_EPS


def _csr_view(schedule, prefix, count):
    starts = schedule.get(f"{prefix}_start")
    ids = schedule.get(f"{prefix}_ids")
    if starts is None or ids is None:
        return None
    if starts.numel() != count + 1:
        raise RuntimeError(f"invalid {prefix} CSR row count")
    return starts.long(), ids.long()


def _csr_row(view, index):
    starts, ids = view
    begin = int(starts[index].item())
    end = int(starts[index + 1].item())
    return ids[begin:end]


def _csr_gather(view, pins):
    """Gather all CSR values belonging to a batch of rows without Python lists."""
    if pins.numel() == 0:
        return view[1][:0]
    starts, ids = view
    begins = starts[pins]
    lengths = starts[pins + 1] - begins
    total = int(lengths.sum().item())
    if total == 0:
        return ids[:0]
    row_offsets = torch.cumsum(lengths, dim=0) - lengths
    positions = (
        torch.repeat_interleave(begins, lengths)
        + torch.arange(total, dtype=torch.long, device=ids.device)
        - torch.repeat_interleave(row_offsets, lengths)
    )
    return ids[positions]


def _changed(before, after):
    if before.numel() == 0:
        return before.new_empty((0,), dtype=torch.bool)
    return torch.any((before != after) | torch.isnan(before) | torch.isnan(after), dim=1)


def _changed_values(before, after):
    """Return whether two scalar/vector timing values differ, including NaN."""
    return bool(torch.any((before != after) | torch.isnan(before) | torch.isnan(after)).item())


def _unique_destinations(rows, column, device):
    if rows.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.unique(rows[:, column].long())


def _assert_native_rat_oracle(native_values, reference_values, label):
    if os.environ.get("REBUFFER_EVENT_NATIVE_RAT_ORACLE", "") != "1":
        return
    for native, reference in zip(native_values, reference_values, strict=True):
        if torch.equal(native, reference):
            continue
        difference = torch.abs(native - reference)
        difference = torch.nan_to_num(difference, nan=float("inf"))
        raise RuntimeError(
            f"native RAT mismatch in {label}: max_abs={float(difference.max().item())}"
        )


def _forward_events(timing, previous, schedule, delays, impulses, caps):
    pin_map = schedule["pin_from_previous"]
    arc_map = schedule["arc_from_previous"]
    count = int(timing.num_pins)
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

    seed = schedule.get("forward_seed_pins", schedule.get("forward_pins"))
    seed = seed.long() if seed is not None else torch.arange(count, device=ra.device)
    seed_set = {int(pin) for pin in seed.detach().cpu().tolist()}
    forward_changed = set(seed_set)
    changed_delay_sinks = set()
    seed_tensor = torch.tensor(sorted(seed_set), dtype=torch.long, device=ra.device)
    if seed_tensor.numel():
        ra[seed_tensor] = -2e8
        fa[seed_tensor] = -2e8
        rs[seed_tensor] = 0.0
        fs[seed_tensor] = 0.0

    # Startpoint values are constraints, so they must overwrite the remapped
    # predecessor state whenever a start pin is in the event seed set.
    starts = schedule["forward_start_rows"].long()
    start_pins = timing.start_points[starts]
    start_mask = torch.tensor(
        [int(pin) in seed_set for pin in start_pins.detach().cpu().tolist()],
        dtype=torch.bool,
        device=ra.device,
    )
    if torch.any(start_mask):
        selected_starts = starts[start_mask]
        selected_pins = start_pins[start_mask]
        ra[selected_pins], fa[selected_pins] = (
            timing.inrdelays[selected_starts],
            timing.infdelays[selected_starts],
        )
        rs[selected_pins], fs[selected_pins] = (
            timing.inrtrans[selected_starts],
            timing.inftrans[selected_starts],
        )

    rank = schedule.get("pin_rank")
    if rank is None:
        raise RuntimeError("event propagation requires native pin ranks")
    rank = rank.long()
    max_rank = int(rank.max().item()) if rank.numel() else 0
    cell_out = _csr_view(schedule, "cell_out", count)
    cell_in = _csr_view(schedule, "cell_in", count)
    net_out = _csr_view(schedule, "net_out", count)
    net_in = _csr_view(schedule, "net_in", count)
    if any(view is None for view in (cell_out, cell_in, net_out, net_in)):
        raise RuntimeError("event propagation requires native cell/net CSR")

    native_clk2q_enabled = os.environ.get("REBUFFER_EVENT_NATIVE_CLK2Q", "") == "1"
    native_clk2q_levels = 0
    clock_rows = schedule.get("clock_rows")
    if clock_rows is not None and clock_rows.numel():
        clock_rows = clock_rows.long()
        clock_view = copy.copy(timing)
        clock_view.flat_inst_arcs_by_level = timing.flat_inst_arcs_by_level[clock_rows]
        clock_view.flat_inst_arcs_by_level_start = [0, int(clock_rows.numel())]
        if not native_clk2q_enabled:
            _, ra, fa, rs, fs = clock_view.calculate_clk2q_aat(
                ra, fa, rs, fs, caps["rise"], caps["fall"], use_surrogate=False
            )
        else:
            clock_dest = timing.flat_inst_arcs_by_level[clock_rows][:, 1].long()
            before = torch.stack(
                (ra[clock_dest], fa[clock_dest], rs[clock_dest], fs[clock_dest]), dim=1
            ).clone()
            ra[clock_dest] = -2e8
            fa[clock_dest] = -2e8
            rs[clock_dest] = 0.0
            native_clk2q_result = try_clk2q_hard_level(
                timing,
                timing.flat_inst_arcs_by_level[clock_rows],
                (ra, fa, rs, fs),
                (caps["rise"], caps["fall"]),
            )
            if native_clk2q_result is None:
                _, ra, fa, rs, fs = clock_view.calculate_clk2q_aat(
                    ra, fa, rs, fs, caps["rise"], caps["fall"], use_surrogate=False
                )
            else:
                native_clk2q_levels += 1
                ra, fa, rs, fs = native_clk2q_result
            after = torch.stack(
                (ra[clock_dest], fa[clock_dest], rs[clock_dest], fs[clock_dest]), dim=1
            )
            for pin in clock_dest[_changed(before, after)].detach().cpu().tolist():
                seed_set.add(int(pin))
                forward_changed.add(int(pin))

    delta_pins = set(seed_set)
    clock_end = int(timing.flat_inst_arcs_by_level_start[1])
    by_level = [[] for _ in range(max_rank + 1)]
    for pin in seed_set:
        if 0 <= pin < count:
            by_level[min(int(rank[pin]), max_rank)].append(pin)
    scheduled = set(seed_set)
    processed = set()
    processed_cell_arcs = 0
    processed_net_ids = 0
    native_cell_levels = 0
    native_net_levels = 0
    native_rat_cell_levels = 0
    native_rat_net_levels = 0
    native_setup_levels = 0
    native_recovery_levels = 0
    changed_count = 0

    for level in range(max_rank + 1):
        pins = by_level[level]
        if not pins:
            continue
        new_pins = []
        for pin in pins:
            if pin in processed:
                continue
            processed.add(pin)
            new_pins.append(pin)
        new_pin_tensor = torch.tensor(new_pins, dtype=torch.long, device=ra.device)
        pin_tensor = torch.tensor(pins, dtype=torch.long, device=ra.device)
        cell_ids = _csr_gather(cell_out, new_pin_tensor)
        net_ids = _csr_gather(net_out, new_pin_tensor)
        # A changed pin can be the output of a cell or the sink of a net whose
        # upstream value did not change.  Recompute those incoming rows too;
        # this is the case that changes a driver's output load after a split.
        cell_ids = torch.cat((cell_ids, _csr_gather(cell_in, pin_tensor)))
        net_ids = torch.cat((net_ids, _csr_gather(net_in, pin_tensor)))
        if cell_ids.numel():
            ids = torch.unique(cell_ids)
            ids = ids[ids >= clock_end]
            if ids.numel():
                rows = timing.flat_inst_arcs_by_level[ids]
                destinations = _unique_destinations(rows, 1, ra.device)
                before = torch.stack(
                    (ra[destinations], fa[destinations], rs[destinations], fs[destinations]), dim=1
                ).clone()
                ra[destinations] = -2e8
                fa[destinations] = -2e8
                rs[destinations] = 0.0
                fs[destinations] = 0.0
                native_result = try_cell_aat_hard_level(
                    timing,
                    rows,
                    (ra, fa, rs, fs),
                    (caps["rise"], caps["fall"]),
                )
                if native_result is None:
                    result = timing.calculate_cell_aat_level(
                        rows,
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
                else:
                    native_cell_levels += 1
                    ra, fa, rs, fs, *updates = native_result
                delay_before = [destination[ids].clone() for destination in cell_delays]
                for destination, values in zip(cell_delays, updates, strict=True):
                    destination[ids] = values
                delay_changed = torch.any(
                    torch.stack(
                        [
                            (before_value != after_value)
                            | torch.isnan(before_value)
                            | torch.isnan(after_value)
                            for before_value, after_value in zip(
                                delay_before, updates, strict=True
                            )
                        ],
                        dim=1,
                    ),
                    dim=1,
                )
                for pin in rows[:, 1][delay_changed].detach().cpu().tolist():
                    changed_delay_sinks.add(int(pin))
                after = torch.stack(
                    (ra[destinations], fa[destinations], rs[destinations], fs[destinations]), dim=1
                )
                changed = destinations[_changed(before, after)]
                processed_cell_arcs += int(ids.numel())
                changed_count += int(changed.numel())
                delta_pins.update(int(pin) for pin in changed.detach().cpu().tolist())
                for pin in changed.detach().cpu().tolist():
                    pin = int(pin)
                    forward_changed.add(pin)
                    if pin not in scheduled:
                        scheduled.add(pin)
                        by_level[min(int(rank[pin]), max_rank)].append(pin)
        if net_ids.numel():
            net_arcs = torch.unique(net_ids)
            if net_arcs.numel():
                arc_rows = timing.net_flat_arcs[net_arcs]
                destinations = _unique_destinations(arc_rows, 1, ra.device)
                before = torch.stack(
                    (ra[destinations], fa[destinations], rs[destinations], fs[destinations]), dim=1
                ).clone()
                ra[destinations] = -2e8
                fa[destinations] = -2e8
                rs[destinations] = 0.0
                fs[destinations] = 0.0
                native_result = try_net_aat_hard_level(
                    timing,
                    arc_rows,
                    (ra, fa, rs, fs),
                    (
                        delays["rise"],
                        delays["fall"],
                        impulses["rise"],
                        impulses["fall"],
                    ),
                    SLEW_SQRT_EPS,
                )
                if native_result is None:
                    native_result = try_net_aat_native(
                        timing,
                        arc_rows,
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
                else:
                    native_net_levels += 1
                if native_result is None:
                    raise RuntimeError("event timing requires the existing native net AAT operator")
                ra, fa, rs, fs = native_result
                after = torch.stack(
                    (ra[destinations], fa[destinations], rs[destinations], fs[destinations]), dim=1
                )
                changed = destinations[_changed(before, after)]
                processed_net_ids += int(net_arcs.numel())
                changed_count += int(changed.numel())
                delta_pins.update(int(pin) for pin in changed.detach().cpu().tolist())
                for pin in changed.detach().cpu().tolist():
                    pin = int(pin)
                    forward_changed.add(pin)
                    if pin not in scheduled:
                        scheduled.add(pin)
                        by_level[min(int(rank[pin]), max_rank)].append(pin)

    # Refresh all endpoint constraints first.  This is small compared with
    # reverse graph propagation and avoids making the event path depend on
    # which setup/recovery check row happened to be in the native cone.
    before_checks_r, before_checks_f = rr.clone(), fr.clone()
    endpoints = timing.end_points.long()
    if endpoints.numel():
        rr[endpoints], fr[endpoints] = (
            timing.endpoints_rRAT.to(device=rr.device, dtype=rr.dtype),
            timing.endpoints_fRAT.to(device=fr.device, dtype=fr.dtype),
        )
    checks = copy.copy(timing)
    checks._critical_path_snapshot_requested = False
    native_rat_enabled = os.environ.get("REBUFFER_EVENT_NATIVE_RAT", "") == "1"
    native_setup_result = None
    if native_rat_enabled:
        native_setup_result = try_setup_rat_hard_level(
            timing,
            timing.endpoints_constraint_arcs,
            rr,
            fr,
            timing.clk_pin_rtran,
            timing.clk_pin_ftran,
            rs,
            fs,
            getattr(timing, "endpoints_constraint_max_valid", None),
        )
    if native_setup_result is None:
        rr, fr = checks.calculate_setup_rat(
            rr, fr, timing.clk_pin_rtran, timing.clk_pin_ftran, rs, fs
        )
    else:
        native_setup_levels += 1
        rr, fr = native_setup_result
    native_recovery_result = None
    check_arcs = getattr(timing, "endpoints_timing_check_arcs", None)
    if (
        native_rat_enabled
        and torch.is_tensor(check_arcs)
        and check_arcs.numel()
        and check_arcs.size(1) > 6
    ):
        recovery_mask = check_arcs[:, 6] == 3
        check_valid = getattr(timing, "endpoints_timing_check_max_valid", None)
        if check_valid is not None:
            recovery_mask = recovery_mask & check_valid.any(dim=1)
        if torch.any(recovery_mask):
            recovery_rows = check_arcs[recovery_mask]
            recovery_valid = (
                torch.ones(
                    (recovery_rows.size(0), 2),
                    dtype=torch.bool,
                    device=recovery_rows.device,
                )
                if check_valid is None
                else check_valid[recovery_mask]
            )
            clock_r = torch.zeros_like(rs)
            clock_f = torch.zeros_like(rs)
            clock_pins = timing.clock_pins.long()
            clock_r[clock_pins] = timing.clk_pin_rtran
            clock_f[clock_pins] = timing.clk_pin_ftran
            recovery_endpoints = recovery_rows[:, 1].long()
            check_data_r = rs[recovery_endpoints]
            check_data_f = fs[recovery_endpoints]
            if (
                getattr(timing, "timing_metric_scope", "setup_plus_recovery")
                == "setup_plus_recovery"
                and not getattr(timing, "recovery_check_uses_live_slew", False)
            ):
                backend_r = getattr(timing, "backend_endpoint_rSlew", None)
                backend_f = getattr(timing, "backend_endpoint_fSlew", None)
                if torch.is_tensor(backend_r) and torch.is_tensor(backend_f):
                    native_r_by_pin = torch.zeros_like(rs)
                    native_f_by_pin = torch.zeros_like(fs)
                    native_r_by_pin[timing.end_points.long()] = backend_r
                    native_f_by_pin[timing.end_points.long()] = backend_f
                    check_data_r = native_r_by_pin[recovery_endpoints]
                    check_data_f = native_f_by_pin[recovery_endpoints]
                else:
                    recovery_rows = None
            if recovery_rows is not None:
                native_recovery_result = try_recovery_rat_hard_level(
                    timing,
                    recovery_rows,
                    rr,
                    fr,
                    rs,
                    fs,
                    clock_r,
                    clock_f,
                    check_data_r,
                    check_data_f,
                    recovery_valid,
                )
    if native_recovery_result is None:
        rr, fr = checks.calculate_recovery_rat(rr, fr, rs, fs)
    else:
        if os.environ.get("REBUFFER_EVENT_NATIVE_RAT_ORACLE", "") == "1":
            reference_r, reference_f = checks.calculate_recovery_rat(
                rr.clone(), fr.clone(), rs, fs
            )
            _assert_native_rat_oracle(
                native_recovery_result,
                (reference_r, reference_f),
                "recovery",
            )
        native_recovery_levels += 1
        rr, fr = native_recovery_result
    changed_checks = (
        (before_checks_r != rr)
        | (before_checks_f != fr)
        | torch.isnan(before_checks_r)
        | torch.isnan(before_checks_f)
        | torch.isnan(rr)
        | torch.isnan(fr)
    )
    delta_pins.update(int(pin) for pin in torch.where(changed_checks)[0].detach().cpu().tolist())

    # Recompute a pin from every current outgoing cell/net contribution.  A
    # min reduction cannot remove a stale old winner by updating only the
    # changed edge; resetting the pin and replaying all its current fanout is
    # the exact required-time operation for one event.
    clock_end = int(timing.flat_inst_arcs_by_level_start[1])
    endpoint_set = {int(pin) for pin in endpoints.detach().cpu().tolist()}
    rank_cpu = schedule["pin_rank"].long().detach().cpu()
    reverse_stats = {"pins": set(), "cell_arcs": 0, "net_ids": 0}
    net_starts = timing.net_flat_arcs_start.long().to(device=rr.device)
    net_arc_to_net = torch.repeat_interleave(
        torch.arange(
            net_starts.numel() - 1,
            dtype=torch.long,
            device=rr.device,
        ),
        net_starts[1:] - net_starts[:-1],
    )

    def recompute_required_batch(pins):
        nonlocal rr, fr, native_rat_cell_levels, native_rat_net_levels
        pins = sorted({int(pin) for pin in pins if 0 <= int(pin) < count})
        if not pins:
            return []
        pin_tensor = torch.tensor(pins, dtype=torch.long, device=rr.device)
        before_r = rr[pin_tensor].clone()
        before_f = fr[pin_tensor].clone()
        rr[pin_tensor] = 2e8
        fr[pin_tensor] = 2e8
        endpoint_mask = torch.tensor(
            [pin in endpoint_set for pin in pins], dtype=torch.bool, device=rr.device
        )
        if torch.any(endpoint_mask):
            rr[pin_tensor[endpoint_mask]] = before_r[endpoint_mask]
            fr[pin_tensor[endpoint_mask]] = before_f[endpoint_mask]

        cell_ids = _csr_gather(cell_out, pin_tensor)
        cell_ids = torch.unique(cell_ids[cell_ids >= clock_end])
        if cell_ids.numel():
            cell_rows = timing.flat_inst_arcs_by_level[cell_ids]
            native_rat_result = try_cell_rat_hard_level(
                timing,
                cell_rows,
                rr,
                fr,
                tuple(values[cell_ids] for values in cell_delays),
            )
            if native_rat_result is None:
                _, rr, fr = timing.calculate_cell_rat_level(
                    cell_rows,
                    rr,
                    fr,
                    *(values[cell_ids] for values in cell_delays),
                )
            else:
                _, reference_r, reference_f = timing.calculate_cell_rat_level(
                    cell_rows,
                    rr.clone(),
                    fr.clone(),
                    *(values[cell_ids] for values in cell_delays),
                )
                _assert_native_rat_oracle(
                    native_rat_result,
                    (reference_r, reference_f),
                    "cell",
                )
                native_rat_cell_levels += 1
                rr, fr = native_rat_result
            reverse_stats["cell_arcs"] += int(cell_ids.numel())

        net_arc_ids = _csr_gather(net_out, pin_tensor)
        if net_arc_ids.numel():
            candidate_net_ids = torch.unique(net_arc_to_net[net_arc_ids])
            drivers = timing.net2driver_pin_map[candidate_net_ids]
            selected_net_ids = candidate_net_ids[
                (drivers >= 0) & torch.isin(drivers, pin_tensor)
            ]
            selected_arc_mask = torch.isin(
                net_arc_to_net[net_arc_ids], selected_net_ids
            )
            selected_arc_ids = net_arc_ids[selected_arc_mask]
        else:
            selected_arc_ids = net_arc_ids
        if selected_arc_ids.numel():
            net_rows = timing.net_flat_arcs[selected_arc_ids]
            sink_rows = net_rows[:, 1].long()
            native_rat_result = try_net_rat_hard_level(
                timing,
                net_rows,
                rr,
                fr,
                (delays["rise"], delays["fall"]),
            )
            if native_rat_result is None:
                rr, fr = timing.calculate_net_rat_level(
                    sink_rows, rr, fr, delays["rise"], delays["fall"]
                )
            else:
                reference_r, reference_f = timing.calculate_net_rat_level(
                    sink_rows,
                    rr.clone(),
                    fr.clone(),
                    delays["rise"],
                    delays["fall"],
                )
                _assert_native_rat_oracle(
                    native_rat_result,
                    (reference_r, reference_f),
                    "net",
                )
                native_rat_net_levels += 1
                rr, fr = native_rat_result
            reverse_stats["net_ids"] += int(selected_net_ids.numel())

        reverse_stats["pins"].update(pins)
        after_r = rr[pin_tensor]
        after_f = fr[pin_tensor]
        changed = (
            (before_r != after_r)
            | (before_f != after_f)
            | torch.isnan(before_r)
            | torch.isnan(before_f)
            | torch.isnan(after_r)
            | torch.isnan(after_f)
        )
        return [
            pin
            for pin, is_changed in zip(
                pins, changed.detach().cpu().tolist(), strict=True
            )
            if is_changed
        ]

    work = []
    sequence = 0
    force_pending = set()
    event_pending = set()
    delta_pins.update(int(pin) for pin in changed_delay_sinks)
    reverse_seed_pins = schedule.get("reverse_seed_pins", torch.empty(0, dtype=torch.long))
    delta_pins.update(int(pin) for pin in reverse_seed_pins.tolist())

    def enqueue_force(pin):
        nonlocal sequence
        pin = int(pin)
        if pin < 0 or pin >= count or pin in force_pending:
            return
        force_pending.add(pin)
        heapq.heappush(work, (-int(rank_cpu[pin]), sequence, 0, pin))
        sequence += 1

    def enqueue_event(pin):
        nonlocal sequence
        pin = int(pin)
        if pin < 0 or pin >= count or pin in event_pending:
            return
        event_pending.add(pin)
        heapq.heappush(work, (-int(rank_cpu[pin]), sequence, 1, pin))
        sequence += 1

    for pin in torch.where(changed_checks)[0].detach().cpu().tolist():
        enqueue_event(pin)
    for pin in forward_changed:
        enqueue_force(pin)
    for pin in reverse_seed_pins.tolist():
        enqueue_force(pin)
    for pin in changed_delay_sinks:
        enqueue_event(pin)

    while work:
        top_rank = -work[0][0]
        force_pins = []
        event_pins = []
        while work and -work[0][0] == top_rank:
            _, _, kind, pin = heapq.heappop(work)
            if kind == 0:
                if pin in force_pending:
                    force_pending.remove(pin)
                    force_pins.append(pin)
            elif pin in event_pending:
                event_pending.remove(pin)
                event_pins.append(pin)
        for pin in recompute_required_batch(force_pins):
            delta_pins.add(int(pin))
            enqueue_event(pin)
        for pin in event_pins:
            cell_arcs = _csr_row(cell_in, pin)
            cell_arcs = cell_arcs[cell_arcs >= clock_end]
            if cell_arcs.numel():
                sources = timing.flat_inst_arcs_by_level[cell_arcs, 0].long()
            else:
                sources = cell_arcs
            for source in sources.detach().cpu().tolist():
                enqueue_force(source)
            net_arcs = _csr_row(net_in, pin)
            if net_arcs.numel():
                sources = timing.net_flat_arcs[net_arcs, 0].long()
                for source in sources[sources >= 0].detach().cpu().tolist():
                    enqueue_force(source)

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
    timing.last_incremental_changed_pins = torch.tensor(
        sorted(delta_pins), dtype=torch.long, device=ra.device
    )
    timing.last_incremental_profile = {
        "event_forward": True,
        "forward_seed_pins": len(seed_set),
        "forward_pins": len(processed),
        "changed_forward_pins": changed_count,
        "forward_cell_arcs": processed_cell_arcs,
        "forward_net_ids": processed_net_ids,
        "native_cell_levels": native_cell_levels,
        "native_net_levels": native_net_levels,
        "native_rat_cell_levels": native_rat_cell_levels,
        "native_rat_net_levels": native_rat_net_levels,
        "native_setup_levels": native_setup_levels,
        "native_recovery_levels": native_recovery_levels,
        "native_clk2q_levels": native_clk2q_levels,
        "reverse_pins": len(reverse_stats["pins"]),
        "reverse_cell_arcs": reverse_stats["cell_arcs"],
        "reverse_net_ids": reverse_stats["net_ids"],
        "total_pins": timing.num_pins,
    }
    return (*endpoint_metrics(timing.last_endpoint_slack_tensor, ra.sum() * 0), 0)


__all__ = ["forward_event_affected"]


def forward_event_affected(timing, previous, schedule, delays, impulses, caps):
    started = time.perf_counter()
    result = _forward_events(timing, previous, schedule, delays, impulses, caps)
    timing.last_incremental_profile["runtime_ms"] = (time.perf_counter() - started) * 1000.0
    return result
