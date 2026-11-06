"""Attach discrete master probes to either the late or early pin-cap tables."""

import torch


def target_pin_ids(data, context, action):
    device = context.beta.device
    nodes = data.pin2node_map.to(device=device, dtype=torch.long)
    offsets = data.pin_2_libpin_offset.to(device=device, dtype=torch.long)
    safe_nodes = nodes.clamp(min=0, max=max(context.target_cell_ids.shape[0] - 1, 0))
    cells = context.target_cell_ids[safe_nodes, action].long()
    starts = data.cell_id_2_libpin_id_start.to(device=device, dtype=torch.long)
    safe_cells = cells.clamp(min=0, max=max(starts.numel() - 2, 0))
    pins = starts[safe_cells] + offsets
    valid = (nodes >= 0) & (offsets >= 0) & (cells >= 0)
    valid &= cells + 1 < starts.numel()
    valid &= (pins >= 0) & (pins < starts[safe_cells + 1])
    valid &= context.valid_instances[safe_nodes, action]
    return pins, valid


def invalidate_instances(context, data, action, invalid_pins):
    if not bool(invalid_pins.any()):
        return
    nodes = data.pin2node_map.to(device=context.beta.device, dtype=torch.long)
    nodes = nodes[invalid_pins.to(device=nodes.device)]
    nodes = nodes[(nodes >= 0) & (nodes < context.valid_instances.shape[0])]
    if nodes.numel():
        context.valid_instances[nodes, action] = False


def apply_pin_caps(model, caps, data, tables):
    """Keep base-point cap Jacobians and add beta times each master delta."""
    context = getattr(model, "_active_discrete_action_context", None)
    if context is None or not int(getattr(model.params, "timing_action_pin_cap_gradient", 1)):
        return caps
    device = context.beta.device
    current = tuple(cap.to(device=device) for cap in caps)
    tables = tuple(table.to(device=device) for table in tables)
    result = current
    nodes = data.pin2node_map.to(device=device, dtype=torch.long)
    safe_nodes = nodes.clamp(min=0, max=max(context.beta.shape[0] - 1, 0))
    for action in range(len(context.action_names)):
        pins, valid = target_pin_ids(data, context, action)
        targets = tuple(table[pins.clamp(min=0, max=max(table.numel() - 1, 0))] for table in tables)
        in_bounds = valid & (pins >= 0)
        for table in tables:
            in_bounds &= pins < table.numel()
        finite = torch.stack([torch.isfinite(target) for target in targets]).all(0)
        invalidate_instances(context, data, action, in_bounds & ~finite)
        weight = context.beta[safe_nodes, action] * (in_bounds & finite).to(context.beta.dtype)
        result = tuple(
            value + weight * torch.nan_to_num(target - base)
            for value, target, base in zip(result, targets, current, strict=True)
        )
    return result
