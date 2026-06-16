from dataclasses import dataclass

import torch
import torch.nn.functional as F


MIN_CONGESTION = 1.0
MAX_SITES = 3
BIN_RADIUS = 3
AREA_BUDGET_RATIO = 0.26
THRESHOLD_PERCENTILE = 50.0
SCALE_PERCENTILE = 90.0
SIGMOID_ALPHA = 3.0
MIN_SITES = 1.0


@dataclass
class EGRPaddingState:
    padding_x: torch.Tensor
    pin_padding: torch.Tensor
    num_selected: int
    threshold: float
    max_congestion: float
    padding_area: float
    padding_area_ratio: float


def _local_max_route_map(route_map, bin_radius):
    if bin_radius <= 0:
        return route_map
    x = route_map.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(
        x,
        kernel_size=2 * bin_radius + 1,
        stride=1,
        padding=bin_radius,
    )
    return x.squeeze(0).squeeze(0)


def _cell_congestion(placedb, pos, node_size_x, node_size_y, route_map, bin_radius):
    num_movable = placedb.num_movable_nodes
    num_nodes = placedb.num_nodes
    xl = pos[:num_movable]
    yl = pos[num_nodes:num_nodes + num_movable]
    xh = xl + node_size_x[:num_movable]
    yh = yl + node_size_y[:num_movable]

    grid_xl = torch.floor(
        (xl - placedb.routing_grid_xl) / placedb.routing_grid_size_x).long()
    grid_xh = torch.floor(
        (xh - placedb.routing_grid_xl) / placedb.routing_grid_size_x).long()
    grid_yl = torch.floor(
        (yl - placedb.routing_grid_yl) / placedb.routing_grid_size_y).long()
    grid_yh = torch.floor(
        (yh - placedb.routing_grid_yl) / placedb.routing_grid_size_y).long()

    valid = (
        (grid_xl <= route_map.shape[0] - 1)
        & (grid_xh >= 0)
        & (grid_yl <= route_map.shape[1] - 1)
        & (grid_yh >= 0)
    )
    grid_x = torch.div(grid_xl + grid_xh, 2, rounding_mode="floor")
    grid_y = torch.div(grid_yl + grid_yh, 2, rounding_mode="floor")
    grid_x = grid_x.clamp(0, route_map.shape[0] - 1)
    grid_y = grid_y.clamp(0, route_map.shape[1] - 1)

    local_max = _local_max_route_map(route_map, bin_radius)
    cell_congestion = torch.zeros(
        num_movable, dtype=route_map.dtype, device=route_map.device)
    cell_congestion[valid] = local_max[grid_x[valid], grid_y[valid]]
    return cell_congestion, valid


def apply_egr_padding(
        placedb,
        pos,
        node_size_x,
        node_size_y,
        pin_offset_x,
        pin2node_map,
        movable_macro_mask,
        route_map):
    num_movable = placedb.num_movable_nodes
    if route_map.numel() == 0:
        return None

    finite_map = route_map[torch.isfinite(route_map)]
    if finite_map.numel() == 0:
        return None

    route_map = route_map.to(pos.device)
    cell_congestion, valid = _cell_congestion(
        placedb, pos, node_size_x, node_size_y, route_map, BIN_RADIUS)
    eligible = (
        valid
        & torch.isfinite(cell_congestion)
        & (cell_congestion >= MIN_CONGESTION)
    )
    eligible &= ~movable_macro_mask[:num_movable].bool().to(eligible.device)

    eligible_values = cell_congestion[eligible]
    if eligible_values.numel() == 0:
        return None

    threshold = torch.quantile(
        eligible_values, THRESHOLD_PERCENTILE / 100.0)
    threshold = torch.maximum(
        threshold,
        torch.tensor(MIN_CONGESTION, dtype=threshold.dtype,
                     device=threshold.device),
    )
    scale_ref = torch.quantile(eligible_values, SCALE_PERCENTILE / 100.0)
    eps = torch.tensor(1e-6, dtype=threshold.dtype, device=threshold.device)
    scale = torch.maximum(scale_ref - threshold, eps)
    congestion_weight = 2.0 * torch.sigmoid(
        SIGMOID_ALPHA * ((cell_congestion - threshold) / scale)
    ) - 1.0
    congestion_weight = torch.clamp(congestion_weight, 0.0, 1.0)

    active = eligible & (cell_congestion >= threshold)
    padding_sites = MIN_SITES + (float(MAX_SITES) - MIN_SITES) * congestion_weight
    padding_sites[~active] = 0.0
    padding_sites = torch.round(padding_sites).to(torch.int32)
    padding_sites = torch.clamp(padding_sites, 0, MAX_SITES)
    selected = padding_sites > 0
    if not bool(selected.any()):
        return None

    site_width = float(placedb.site_width)
    node_area = (
        node_size_x[:num_movable].to(cell_congestion.dtype)
        * node_size_y[:num_movable].to(cell_congestion.dtype)
    )
    movable_area = torch.clamp(
        node_area.sum(),
        min=torch.tensor(1e-6, dtype=node_area.dtype, device=node_area.device),
    )
    padding_area = (
        2.0
        * padding_sites.to(cell_congestion.dtype)
        * site_width
        * node_size_y[:num_movable].to(cell_congestion.dtype)
    )
    area_budget = AREA_BUDGET_RATIO * movable_area
    total_padding_area = padding_area[selected].sum()
    if bool(total_padding_area > area_budget):
        selected_indices = torch.nonzero(selected, as_tuple=False).view(-1)
        _, order = torch.sort(
            cell_congestion[selected_indices], descending=True)
        ordered_indices = selected_indices[order]
        ordered_area = padding_area[ordered_indices]
        keep_mask_order = torch.cumsum(ordered_area, dim=0) <= area_budget
        if not bool(keep_mask_order.any()):
            keep_mask_order[0] = True
        keep_indices = ordered_indices[keep_mask_order]
        keep_mask = torch.zeros_like(selected, dtype=torch.bool)
        keep_mask[keep_indices] = True
        padding_sites[~keep_mask] = 0
        selected = padding_sites > 0
        if not bool(selected.any()):
            return None
        padding_area = (
            2.0
            * padding_sites.to(cell_congestion.dtype)
            * site_width
            * node_size_y[:num_movable].to(cell_congestion.dtype)
        )
        total_padding_area = padding_area[selected].sum()

    padding_x = torch.zeros_like(node_size_x[:num_movable])
    padding_x[selected] = (
        padding_sites[selected].to(padding_x.dtype) * site_width)

    pin_padding = torch.zeros_like(pin_offset_x)
    pin2node = pin2node_map.long().to(pin_padding.device)
    movable_pin_mask = pin2node < num_movable
    pin_padding[movable_pin_mask] = padding_x[
        pin2node[movable_pin_mask]].to(pin_padding.dtype)

    node_size_x[:num_movable] += 2 * padding_x
    pin_offset_x += pin_padding
    pos[:num_movable] -= padding_x

    return EGRPaddingState(
        padding_x=padding_x,
        pin_padding=pin_padding,
        num_selected=int(selected.sum().item()),
        threshold=float(threshold.item()),
        max_congestion=float(cell_congestion.max().item()),
        padding_area=float(total_padding_area.item()),
        padding_area_ratio=float((total_padding_area / movable_area).item()),
    )


def restore_egr_padding(state, pos, node_size_x, pin_offset_x):
    num_movable = state.padding_x.numel()
    padding_x = state.padding_x.to(pos.device)
    pos[:num_movable] += padding_x
    node_size_x[:num_movable] -= 2 * padding_x.to(node_size_x.device)
    pin_offset_x -= state.pin_padding.to(pin_offset_x.device)
