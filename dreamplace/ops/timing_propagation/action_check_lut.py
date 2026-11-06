"""Interpolate setup/recovery LUTs along the same master actions as cell arcs."""

import torch


def _index_on_device(value, device):
    return value.to(device=device, dtype=torch.long)


def action_check_values(timing, context, check_rows, transition, clock_slew, slew):
    device = timing.endpoints_timing_check_arcs.device
    check_rows = _index_on_device(check_rows, device)
    transition = transition.to(device=device)
    clock_slew = clock_slew.to(device=device)
    slew = slew.to(device=device)
    arcs = timing.endpoints_timing_check_arcs[check_rows]

    def query(rows):
        values = tuple(
            timing.lut_entry_vectorized(arcs[:, 2], clock_slew, slew, rows, info)
            for info in (timing.arcs_info.r_check_luts, timing.arcs_info.f_check_luts)
        )
        return torch.where(transition == 0, *values)

    current = query(arcs[:, 3])
    if context is None:
        return current
    if context.check_target_arc_ids is None:
        raise RuntimeError("lut_action requires endpoint check arc mappings")
    inst = context.check_inst_ids.to(device=device)[check_rows]
    targets = context.check_target_arc_ids.to(device=device)[check_rows]
    return _interpolate(current, query, arcs[:, 3], targets, inst, context)


def action_setup_values(timing, cells, clock_slew, slew, rows, lut):
    """Legacy constraint rows use clock-local input ids but physical outputs."""

    device = lut.flat_luts_values.device
    cells = _index_on_device(cells, device)
    clock_slew = clock_slew.to(device=device)
    slew = slew.to(device=device)
    rows = _index_on_device(rows, device)

    def query(indices):
        return timing.lut_entry_vectorized(
            cells, clock_slew, slew, _index_on_device(indices, device), lut
        )

    current = query(rows)
    context = getattr(timing, "_active_discrete_action_context", None)
    if context is None:
        return current
    if context.constraint_target_arc_ids is None:
        raise RuntimeError("lut_action requires setup constraint arc mappings")
    return _interpolate(
        current,
        query,
        rows,
        context.constraint_target_arc_ids,
        context.constraint_inst_ids,
        context,
    )


def _interpolate(current, query, rows, targets, inst, context):
    device = current.device
    rows = _index_on_device(rows, device)
    targets = _index_on_device(targets, device)
    inst = _index_on_device(inst, device)
    valid_instances = context.valid_instances.to(device=device)
    beta = context.beta.to(device=device)
    valid = (inst >= 0)[:, None] & (targets >= 0)
    valid &= valid_instances[inst.clamp(min=0)]
    result = current
    for action in range(beta.shape[1]):
        target_rows = torch.where(valid[:, action], targets[:, action], rows)
        delta = query(target_rows) - current
        weight = beta[inst.clamp(min=0), action] * valid[:, action]
        result = result + weight * delta
    return result
