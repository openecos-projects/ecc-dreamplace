"""Window geometry and one-time area coordination for inflation S5B1.

Native/base indices stay unchanged. Integer virtual buffers contribute their
physical area to the same movable capacity, but never receive inflation.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AreaState:
    movable: float
    filler: float
    capacity: float

    @property
    def density(self):
        return (self.movable + self.filler) / self.capacity

    def fits(self):
        return self.movable <= self.capacity


def initialize_overflow_reference(placedb, continuation_seed=None, mode="initial"):
    """Initialize overflow normalization, carrying initial-area state forward."""
    mode = str(mode or "initial").strip().lower()
    if mode not in {"initial", "ordinary"}:
        raise ValueError("overflow_reference_mode must be 'initial' or 'ordinary'")
    placedb.overflow_reference_mode = mode
    seed = continuation_seed or {}
    placedb.overflow_reference_area = float(seed.get(
        "overflow_reference_area", placedb.total_movable_node_area,
    ))


def movable_area_for_metrics(data, placedb):
    """Freeze the GP-entry scale or preserve ordinary area publication behavior."""
    if getattr(placedb, "overflow_reference_mode", "initial") == "initial":
        return getattr(placedb, "overflow_reference_area", placedb.total_movable_node_area)
    state = getattr(data, "cooptimization_area", None)
    return placedb.total_movable_node_area if state is None else state.movable


def publish_area(data, placedb, virtual_area, *, capacity):
    """Publish current geometry after a window/inflation, without changing filler."""
    state = capture_area(data, placedb, virtual_area, capacity=capacity)
    _publish_geometry(state, data, placedb, virtual_area)
    return state


def _publish_geometry(state, data, placedb, virtual_area):
    data.cooptimization_area = state
    placedb.total_movable_node_area = state.movable - float(virtual_area)
    placedb.total_filler_node_area = state.filler
    with torch.no_grad():
        data.node_areas.copy_(data.node_size_x * data.node_size_y)
        # Occupancy is a lower bound on the density target, not the target
        # itself. Without fillers, the configured target leaves density
        # headroom that an electrical window must not remove.
        data.target_density.clamp_(min=state.density)
        data.sorted_node_map.copy_(torch.argsort(
            data.node_size_x[:placedb.num_movable_nodes]
        ).to(dtype=torch.int32))
    placedb.node_size_x[:] = data.node_size_x.detach().cpu().numpy()
    placedb.node_size_y[:] = data.node_size_y.detach().cpu().numpy()


def capture_area(data, placedb, virtual_area=0.0, *, capacity=None):
    """Use the existing PlaceDB filler capacity, independent of target density."""
    if capacity is None:
        capacity = max(placedb.area - placedb.total_fixed_node_area, placedb.total_space_area)
    capacity = float(capacity)
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("cooptimization requires positive placement capacity")
    count = placedb.num_movable_nodes
    movable = float(
        (data.node_size_x[:count].double() * data.node_size_y[:count].double()).sum().item()
    )
    fillers = placedb.num_filler_nodes
    filler = (
        float(
            (data.node_size_x[-fillers:].double() * data.node_size_y[-fillers:].double())
            .sum()
            .item()
        )
        if fillers
        else 0.0
    )
    return AreaState(movable + float(virtual_area), filler, capacity)


def coordinate_area(before, data, placedb, pos, virtual_area=0.0):
    """Consume the final window's net area change once, preserving filler centers."""
    after = capture_area(data, placedb, virtual_area, capacity=before.capacity)
    if not after.fits():
        raise ValueError("combined movable footprint exceeds placement capacity")
    filler_area = 0.0
    fillers = placedb.num_filler_nodes
    if fillers and before.filler > 0:
        filler_area = min(
            max(before.filler - (after.movable - before.movable), 0.0),
            before.capacity - after.movable,
        )
        scale = math.sqrt(filler_area / before.filler)
        nodes = data.node_size_x.numel()
        with torch.no_grad():
            x = pos[nodes - fillers : nodes]
            y = pos[2 * nodes - fillers : 2 * nodes]
            width, height = data.node_size_x[-fillers:], data.node_size_y[-fillers:]
            x.add_(width * 0.5)
            y.add_(height * 0.5)
            width.mul_(scale)
            height.mul_(scale)
            x.sub_(width * 0.5)
            y.sub_(height * 0.5)
    result = AreaState(after.movable, filler_area, before.capacity)
    _publish_geometry(result, data, placedb, virtual_area)
    return result


class SizingWindowGeometry:
    """Preserve entry centers and inflated footprints throughout all S rounds."""

    def __init__(self, data, placedb, pos):
        self.data = data
        self.pos = pos
        self.num_nodes = data.node_size_x.numel()
        count = placedb.num_movable_nodes
        self.width = data.node_size_x[:count].detach().clone()
        self.height = data.node_size_y[:count].detach().clone()
        self.inflated = (self.width > data.original_node_size_x[:count]) | (
            self.height > data.original_node_size_y[:count]
        )
        self.center_x = pos[:count].detach().clone() + self.width * 0.5
        self.center_y = (
            pos[self.num_nodes : self.num_nodes + count].detach().clone() + self.height * 0.5
        )

    def apply_sizes(self, node_ids, width, height):
        data = self.data
        with torch.no_grad():
            data.original_node_size_x[node_ids] = width
            data.original_node_size_y[node_ids] = height
            data.node_size_x[node_ids] = torch.where(
                self.inflated[node_ids], torch.maximum(self.width[node_ids], width), width
            )
            data.node_size_y[node_ids] = torch.where(
                self.inflated[node_ids], torch.maximum(self.height[node_ids], height), height
            )
            self.pos[node_ids] = self.center_x[node_ids] - data.node_size_x[node_ids] * 0.5
            self.pos[self.num_nodes + node_ids] = (
                self.center_y[node_ids] - data.node_size_y[node_ids] * 0.5
            )
            data.node_areas[node_ids] = data.node_size_x[node_ids] * data.node_size_y[node_ids]

    def apply_pins(self, pin_ids):
        data = self.data
        nodes = data.pin2node_map[pin_ids].long()
        with torch.no_grad():
            data.original_pin_offset_x[pin_ids] = data.pin_offset_x[pin_ids]
            data.original_pin_offset_y[pin_ids] = data.pin_offset_y[pin_ids]
            data.pin_offset_x[pin_ids] += (
                data.node_size_x[nodes] - data.original_node_size_x[nodes]
            ) * 0.5
            data.pin_offset_y[pin_ids] += (
                data.node_size_y[nodes] - data.original_node_size_y[nodes]
            ) * 0.5
