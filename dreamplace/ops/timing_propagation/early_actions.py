"""Base-point master partials on the live early timing graph."""

from dataclasses import replace

import torch


def _context(timing):
    context = getattr(timing, "_active_discrete_action_context", None)
    if context is not None and (torch.is_grad_enabled() or context.evaluate_targets):
        return context
    return None


def _probe(context, owners, valid, current, targets):
    """Keep master deltas constant while all current-state Jacobians stay live."""
    safe = owners.clamp(min=0, max=max(context.beta.shape[0] - 1, 0))
    valid = valid & (owners >= 0).unsqueeze(1) & context.valid_instances[safe]
    with torch.no_grad():
        finite_current = torch.isfinite(current).unsqueeze(1)
        finite_target = torch.isfinite(targets)
        invalid = valid & (finite_current & ~finite_target).any(dim=2)
        if bool(invalid.any()):
            rows, actions = torch.nonzero(invalid, as_tuple=True)
            context.valid_instances[safe[rows], actions] = False
        valid = valid & ~invalid
        delta = torch.where(finite_current & finite_target, targets - current.unsqueeze(1), 0)
    weights = context.beta[safe] * valid.to(context.beta.dtype)
    return (weights.unsqueeze(2) * delta).sum(dim=1)


def attach_cell_probes(
    timing, arcs, lanes, slews, caps, global_rows, *, clock=False,
    latch_data_arcs=None, physical_pins=None,
):
    context = _context(timing)
    if context is None or not len(arcs):
        return lanes
    from .action_cell_aat import _target_rows
    from .early_propagation import _cell_lanes

    targets = context.target_arc_ids[global_rows]
    target_rows = _target_rows(arcs, targets, context.target_senses[global_rows])
    with torch.no_grad():
        target_lanes = _cell_lanes(
            timing, target_rows, slews, caps, clock=clock,
            latch_data_arcs=latch_data_arcs, physical_pins=physical_pins,
        )
    valid = context.valid_arcs[global_rows] & (targets >= 0)
    owners = context.inst_ids[global_rows]
    result = []
    for lane, target in zip(lanes, target_lanes, strict=True):
        with torch.no_grad():
            values = caps[0].new_full((len(target_rows), 2), torch.inf)
            values[target.rows] = torch.stack((target.delay, target.slew), dim=1)
            values = values.reshape(len(arcs), targets.shape[1], 2)[lane.rows]
        current = torch.stack((lane.delay, lane.slew), dim=1)
        probe = _probe(context, owners[lane.rows], valid[lane.rows], current, values)
        # Interpolate each legal input/output branch before min/LSE reduction.
        # The winning branch can differ between current and target masters.
        result.append(
            replace(
                lane,
                delay=lane.delay + probe[:, 0],
                slew=lane.slew + probe[:, 1],
            )
        )
    return result


def attach_check_probes(timing, check_rows, rows, current, clock_slew, data_slew, lut):
    context = _context(timing)
    if context is None or not len(rows):
        return current
    if context.check_target_arc_ids is None or context.check_inst_ids is None:
        raise RuntimeError("hold actions require mapped endpoint check master identities")
    targets = context.check_target_arc_ids[check_rows]
    safe_targets = torch.where(targets >= 0, targets, rows[:, 3, None])
    valid = targets >= 0
    with torch.no_grad():
        values = timing.lut_entry_vectorized(
            rows[:, 2].long().repeat_interleave(targets.shape[1]),
            clock_slew.repeat_interleave(targets.shape[1]),
            data_slew.repeat_interleave(targets.shape[1]),
            safe_targets.reshape(-1).long(),
            lut,
        ).reshape(len(rows), targets.shape[1], 1)
        values = torch.where(lut.valid_arc_mask[safe_targets].unsqueeze(2), values, torch.inf)
    probe = _probe(context, context.check_inst_ids[check_rows], valid, current[:, None], values)
    return current + probe[:, 0]
