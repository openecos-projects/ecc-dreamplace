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
    nodes = data.pin2node_map.to(device=device, dtype=torch.long)
    safe_nodes = nodes.clamp(min=0, max=max(context.beta.shape[0] - 1, 0))
    offsets = data.pin_2_libpin_offset.to(device=device, dtype=torch.long)
    starts = data.cell_id_2_libpin_id_start.to(device=device, dtype=torch.long)
    cells = context.target_cell_ids[safe_nodes].long()
    safe_cells = cells.clamp(min=0, max=max(starts.numel() - 2, 0))
    pins = starts[safe_cells] + offsets.unsqueeze(1)
    valid = ((nodes >= 0) & (offsets >= 0)).unsqueeze(1) & (cells >= 0)
    valid &= cells + 1 < starts.numel()
    valid &= (pins >= 0) & (pins < starts[safe_cells + 1])
    valid &= context.valid_instances[safe_nodes]
    in_bounds = valid.clone()
    for table in tables:
        in_bounds &= pins < table.numel()
    clamped_pins = tuple(
        pins.clamp(min=0, max=max(table.numel() - 1, 0)) for table in tables
    )
    targets = tuple(
        table[clamped] for table, clamped in zip(tables, clamped_pins, strict=True)
    )
    finite = torch.ones_like(in_bounds)
    for target in targets:
        finite &= torch.isfinite(target)
    invalid_mask = in_bounds & ~finite
    if bool(invalid_mask.any()):
        for action in range(len(context.action_names)):
            invalidate_instances(context, data, action, invalid_mask[:, action])
    mask = (in_bounds & finite).to(context.beta.dtype)
    weights = context.beta.index_select(0, safe_nodes) * mask
    return tuple(
        base + (weights * torch.nan_to_num(target - base.unsqueeze(1))).sum(dim=1)
        for base, target in zip(current, targets, strict=True)
    )
