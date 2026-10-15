import math

import torch


def build_live_edge_geometry(
    new_x,
    new_y,
    edge_parent_node_id,
    edge_child_node_id,
    *,
    r_unit,
    c_unit,
    scale_factor,
    dbu,
    require_axis_aligned=False,
):
    """Build differentiable edge length and RC in the supplied edge order."""

    if not torch.is_tensor(new_x) or not torch.is_tensor(new_y):
        raise TypeError("new_x and new_y must be torch tensors")
    if new_x.ndim != 1 or new_y.ndim != 1 or new_x.shape != new_y.shape:
        raise ValueError("new_x and new_y must be same-length one-dimensional tensors")
    if not new_x.is_floating_point() or not new_y.is_floating_point():
        raise TypeError("live topology coordinates must use a floating dtype")
    if new_x.device != new_y.device or new_x.dtype != new_y.dtype:
        raise ValueError("new_x and new_y must share device and dtype")

    parent = torch.as_tensor(
        edge_parent_node_id,
        dtype=torch.long,
        device=new_x.device,
    ).view(-1)
    child = torch.as_tensor(
        edge_child_node_id,
        dtype=torch.long,
        device=new_x.device,
    ).view(-1)
    if parent.shape != child.shape:
        raise ValueError("edge parent and child index tensors must have equal length")
    if int(parent.numel()):
        minimum = min(int(parent.min().item()), int(child.min().item()))
        maximum = max(int(parent.max().item()), int(child.max().item()))
        if minimum < 0 or maximum >= int(new_x.numel()):
            raise IndexError("live edge node ID is outside the coordinate domain")

    scale_factor = float(scale_factor)
    dbu = float(dbu)
    r_unit = float(r_unit)
    c_unit = float(c_unit)
    for name, value in (
        ("scale_factor", scale_factor),
        ("dbu", dbu),
        ("r_unit", r_unit),
        ("c_unit", c_unit),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if scale_factor <= 0.0 or dbu <= 0.0:
        raise ValueError("scale_factor and dbu must be positive")

    dx = new_x.index_select(0, child) - new_x.index_select(0, parent)
    dy = new_y.index_select(0, child) - new_y.index_select(0, parent)
    x_nonzero = dx != 0
    y_nonzero = dy != 0
    diagonal = x_nonzero & y_nonzero
    if require_axis_aligned and bool(diagonal.any().item()):
        raise ValueError("live segment topology contains a diagonal edge")

    length = (torch.abs(dx) + torch.abs(dy)) / scale_factor / dbu
    return {
        "length": length,
        "edge_resistance": length * r_unit,
        "edge_capacitance": length * c_unit,
        "edge_parent_node_id": parent,
        "edge_child_node_id": child,
        "rectilinear_path_policy": "x_then_y",
        "census": {
            "edge_count": int(parent.numel()),
            "horizontal_edge_count": int((x_nonzero & ~y_nonzero).sum().item()),
            "vertical_edge_count": int((~x_nonzero & y_nonzero).sum().item()),
            "diagonal_edge_count": int(diagonal.sum().item()),
            "zero_length_edge_count": int((~x_nonzero & ~y_nonzero).sum().item()),
        },
    }
