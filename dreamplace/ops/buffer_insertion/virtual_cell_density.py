"""Derived virtual-buffer cells for the placement density objective.

The view is intentionally separate from the placement database.  Active
integer segment counts select a compact set of virtual cells; their positions
are differentiable functions of the current Steiner endpoint coordinates.
"""

import math

import torch


def _require_1d_tensor(name, value):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch tensor")
    if not value.is_floating_point() and name != "counts":
        raise TypeError(f"{name} must use a floating dtype")


def _validate_counts(counts):
    _require_1d_tensor("counts", counts)
    detached = counts.detach()
    if not torch.isfinite(detached).all():
        raise ValueError("counts must be finite")
    if not torch.equal(detached, detached.round()):
        raise ValueError("virtual-cell counts must be integer-valued")
    if bool((detached < 0).any()):
        raise ValueError("virtual-cell counts must be nonnegative")
    return detached.to(dtype=torch.long)


def build_equal_spaced_virtual_positions(
    parent_x,
    parent_y,
    child_x,
    child_y,
    counts,
):
    """Return compact equal-spaced virtual-cell positions for each segment.

    Endpoint coordinates are in the same coordinate domain as the placement
    position tensor.  Count selection is deliberately detached:
    this view is used with integer ``z`` fixed during the placement block, so
    it supplies endpoint gradients but never a density-to-``z`` gradient.
    """

    for name, value in (
        ("parent_x", parent_x),
        ("parent_y", parent_y),
        ("child_x", child_x),
        ("child_y", child_y),
    ):
        _require_1d_tensor(name, value)
    if not (parent_x.shape == parent_y.shape == child_x.shape == child_y.shape):
        raise ValueError("segment endpoints must have equal shape")
    if len({parent_x.device, parent_y.device, child_x.device, child_y.device}) != 1:
        raise ValueError("segment endpoints must share device")
    if len({parent_x.dtype, parent_y.dtype, child_x.dtype, child_y.dtype}) != 1:
        raise ValueError("segment endpoints must share dtype")
    count_values = _validate_counts(counts)
    if count_values.device != parent_x.device:
        count_values = count_values.to(device=parent_x.device)
    if int(count_values.numel()) != int(parent_x.numel()):
        raise ValueError("counts must have one entry per segment")

    active_segment_indices = torch.nonzero(count_values > 0, as_tuple=False).flatten()
    if int(active_segment_indices.numel()) == 0:
        empty = parent_x[:0]
        return {
            "x": empty,
            "y": parent_y[:0],
            "segment_indices": active_segment_indices,
            "repeater_indices": torch.empty(
                0, dtype=torch.long, device=parent_x.device
            ),
            "fractions": empty,
        }

    active_counts = count_values.index_select(0, active_segment_indices)
    total_count = int(active_counts.sum().item())
    repeated_segments = torch.repeat_interleave(
        active_segment_indices,
        active_counts,
    )
    starts = torch.cumsum(active_counts, dim=0) - active_counts
    repeated_starts = torch.repeat_interleave(starts, active_counts)
    repeater_indices = (
        torch.arange(total_count, dtype=torch.long, device=parent_x.device)
        - repeated_starts
        + 1
    )
    repeated_counts = count_values.index_select(0, repeated_segments)
    fractions = repeater_indices.to(dtype=parent_x.dtype) / (
        repeated_counts.to(dtype=parent_x.dtype) + 1.0
    )
    repeated_parent_x = parent_x.index_select(0, repeated_segments)
    repeated_parent_y = parent_y.index_select(0, repeated_segments)
    repeated_child_x = child_x.index_select(0, repeated_segments)
    repeated_child_y = child_y.index_select(0, repeated_segments)
    return {
        "x": repeated_parent_x
        + (repeated_child_x - repeated_parent_x)
        * fractions,
        "y": repeated_parent_y
        + (repeated_child_y - repeated_parent_y)
        * fractions,
        "segment_indices": repeated_segments,
        "repeater_indices": repeater_indices,
        "fractions": fractions,
    }


def build_segment_endpoint_mapping(prepared_timing_inputs, segment_ids):
    """Map segment rows to the compact Steiner endpoint coordinate domain."""

    if not isinstance(prepared_timing_inputs, dict):
        raise ValueError("prepared_timing_inputs must be a dictionary")
    required = (
        "edge_to_segment_id",
        "edge_parent_compact_id",
        "edge_child_compact_id",
        "flat_topo_node_id",
    )
    missing = [name for name in required if name not in prepared_timing_inputs]
    if missing:
        raise ValueError(
            "prepared_timing_inputs is missing endpoint fields: " + ", ".join(missing)
        )
    segment_ids = torch.as_tensor(segment_ids, dtype=torch.long).flatten().cpu()
    if int(torch.unique(segment_ids).numel()) != int(segment_ids.numel()):
        raise ValueError("segment_ids must be unique")
    row_by_segment_id = {
        int(value): index for index, value in enumerate(segment_ids.tolist())
    }
    edge_to_segment = torch.as_tensor(
        prepared_timing_inputs["edge_to_segment_id"], dtype=torch.long
    ).flatten()
    edge_parent = torch.as_tensor(
        prepared_timing_inputs["edge_parent_compact_id"], dtype=torch.long
    ).flatten()
    edge_child = torch.as_tensor(
        prepared_timing_inputs["edge_child_compact_id"], dtype=torch.long
    ).flatten()
    if not (edge_to_segment.numel() == edge_parent.numel() == edge_child.numel()):
        raise ValueError("prepared endpoint fields must have equal length")
    parent = torch.full((segment_ids.numel(),), -1, dtype=torch.long)
    child = torch.full((segment_ids.numel(),), -1, dtype=torch.long)
    for segment_id, parent_id, child_id in zip(
        edge_to_segment.tolist(), edge_parent.tolist(), edge_child.tolist()
    ):
        row = row_by_segment_id.get(int(segment_id))
        if row is None:
            continue
        if parent[row] >= 0:
            if int(parent[row]) != int(parent_id) or int(child[row]) != int(child_id):
                raise ValueError(f"segment {segment_id} has conflicting endpoint mappings")
            continue
        parent[row] = int(parent_id)
        child[row] = int(child_id)
    if bool((parent < 0).any()) or bool((child < 0).any()):
        missing_ids = segment_ids[(parent < 0) | (child < 0)].tolist()
        raise ValueError(
            "prepared timing inputs do not map every segment to compact endpoints: "
            + ", ".join(str(int(value)) for value in missing_ids[:8])
        )
    topology_ids = torch.as_tensor(prepared_timing_inputs["flat_topo_node_id"], dtype=torch.long)
    return topology_ids.index_select(0, parent), topology_ids.index_select(0, child)


def build_combined_virtual_position(
    pos,
    virtual_x,
    virtual_y,
    virtual_size_x,
    virtual_size_y,
    *,
    num_movable_nodes,
    num_terminals,
    num_filler_nodes,
    num_fixed_nodes=None,
):
    """Append virtual cells before all fixed and filler nodes.

    ``num_terminals`` is the density-kernel terminal count.  A DREAMPlace
    position can also contain terminal-NIs, which occupy the fixed physical
    range but are not included in that kernel count.  Callers may provide the
    complete fixed-node count through ``num_fixed_nodes``.
    """

    if not torch.is_tensor(pos) or pos.ndim != 1 or not pos.is_floating_point():
        raise ValueError("pos must be a one-dimensional floating tensor")
    if pos.numel() % 2:
        raise ValueError("pos must contain concatenated x and y coordinates")
    num_nodes = int(pos.numel() // 2)
    num_movable_nodes = int(num_movable_nodes)
    num_terminals = int(num_terminals)
    num_filler_nodes = int(num_filler_nodes)
    num_fixed_nodes = (
        num_terminals if num_fixed_nodes is None else int(num_fixed_nodes)
    )
    if (
        num_movable_nodes < 0
        or num_terminals < 0
        or num_filler_nodes < 0
        or num_fixed_nodes < 0
    ):
        raise ValueError("node counts must be nonnegative")
    if num_movable_nodes + num_fixed_nodes + num_filler_nodes != num_nodes:
        raise ValueError(
            "node counts do not match pos: "
            f"pos_nodes={num_nodes}, movable={num_movable_nodes}, "
            f"terminals={num_terminals}, fixed={num_fixed_nodes}, "
            f"fillers={num_filler_nodes}"
        )
    if virtual_x.shape != virtual_y.shape:
        raise ValueError("virtual_x and virtual_y must have equal shape")
    if virtual_x.device != pos.device or virtual_y.device != pos.device:
        raise ValueError("virtual positions must share the pos device")
    if virtual_x.dtype != pos.dtype or virtual_y.dtype != pos.dtype:
        raise ValueError("virtual positions must share the pos dtype")
    virtual_size_x = torch.as_tensor(
        virtual_size_x, dtype=pos.dtype, device=pos.device
    ).reshape(-1)
    virtual_size_y = torch.as_tensor(
        virtual_size_y, dtype=pos.dtype, device=pos.device
    ).reshape(-1)
    if virtual_size_x.numel() == 1 and virtual_x.numel() != 1:
        virtual_size_x = virtual_size_x.expand_as(virtual_x)
    if virtual_size_y.numel() == 1 and virtual_y.numel() != 1:
        virtual_size_y = virtual_size_y.expand_as(virtual_y)
    if (
        virtual_size_x.numel() != virtual_x.numel()
        or virtual_size_y.numel() != virtual_x.numel()
    ):
        raise ValueError("virtual cell sizes must match virtual positions")

    x = pos[:num_nodes]
    y = pos[num_nodes:]
    physical_end = num_movable_nodes + num_fixed_nodes
    combined_x = torch.cat(
        (x[:num_movable_nodes], virtual_x, x[num_movable_nodes:physical_end], x[physical_end:]),
        dim=0,
    )
    combined_y = torch.cat(
        (y[:num_movable_nodes], virtual_y, y[num_movable_nodes:physical_end], y[physical_end:]),
        dim=0,
    )
    return torch.cat((combined_x, combined_y), dim=0)


class VirtualCellDensityOp:
    """Callable density view backed by the existing ElectricPotential op."""

    def __init__(
        self,
        *,
        segment_state,
        prepared_timing_inputs,
        num_movable_nodes,
        num_terminals,
        num_filler_nodes,
        node_size_x,
        node_size_y,
        buffer_size_x,
        buffer_size_y,
        movable_macro_mask,
        density_op_factory,
        num_fixed_nodes=None,
    ):
        self.segment_state = segment_state
        self.num_movable_nodes = int(num_movable_nodes)
        self.num_terminals = int(num_terminals)
        self.num_filler_nodes = int(num_filler_nodes)
        self.num_fixed_nodes = int(
            self.num_terminals if num_fixed_nodes is None else num_fixed_nodes
        )
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.buffer_size_x = float(buffer_size_x)
        self.buffer_size_y = float(buffer_size_y)
        self.movable_macro_mask = movable_macro_mask
        self.density_op_factory = density_op_factory
        self.segment_ids = segment_state.segment_ids.detach().cpu()
        self.parent_endpoint_id, self.child_endpoint_id = build_segment_endpoint_mapping(
            prepared_timing_inputs,
            self.segment_ids,
        )
        self._density_ops = {}
        self._layout_key = None
        self.last_metadata = {}

    def _active_rows(self):
        counts = _validate_counts(self.segment_state.z_param)
        active = torch.nonzero(counts > 0, as_tuple=False).flatten()
        return counts, active

    def _density_op(self, active_count, device, dtype):
        key = (tuple(int(value) for value in active_count), str(device), str(dtype))
        cached = self._density_ops.get(key)
        if cached is not None:
            return cached
        virtual_count = sum(int(value) for value in active_count)
        base_size_x = self.node_size_x.to(device=device, dtype=dtype)
        base_size_y = self.node_size_y.to(device=device, dtype=dtype)
        combined_size_x = torch.cat(
            (
                base_size_x[: self.num_movable_nodes],
                torch.full(
                    (virtual_count,), self.buffer_size_x, dtype=dtype, device=device
                ),
                base_size_x[self.num_movable_nodes :],
            ),
            dim=0,
        )
        combined_size_y = torch.cat(
            (
                base_size_y[: self.num_movable_nodes],
                torch.full(
                    (virtual_count,), self.buffer_size_y, dtype=dtype, device=device
                ),
                base_size_y[self.num_movable_nodes :],
            ),
            dim=0,
        )
        combined_movable_nodes = self.num_movable_nodes + virtual_count
        sorted_node_map = torch.argsort(combined_size_x[:combined_movable_nodes]).to(
            dtype=torch.int32
        )
        if self.movable_macro_mask is None:
            combined_macro_mask = None
        else:
            base_macro_mask = self.movable_macro_mask.to(device=device, dtype=torch.bool)
            combined_macro_mask = torch.cat(
                (
                    base_macro_mask[: self.num_movable_nodes],
                    torch.zeros(virtual_count, dtype=torch.bool, device=device),
                ),
                dim=0,
            )
        density_op = self.density_op_factory(
            node_size_x=combined_size_x,
            node_size_y=combined_size_y,
            num_movable_nodes=combined_movable_nodes,
            num_terminals=self.num_terminals,
            num_filler_nodes=self.num_filler_nodes,
            sorted_node_map=sorted_node_map,
            movable_macro_mask=combined_macro_mask,
        )
        self._density_ops[key] = density_op
        return density_op

    def _prepare_layout(self, device, dtype):
        z = self.segment_state.z_param
        key = (id(z), z._version, device, dtype)
        if key == self._layout_key:
            return
        counts, active = self._active_rows()
        active_counts = counts.index_select(0, active).to(device=device)
        dummy = torch.zeros(active.numel(), dtype=dtype, device=device)
        layout = build_equal_spaced_virtual_positions(
            dummy, dummy, dummy, dummy, active_counts
        )
        repeated_rows = active.to(device=device).index_select(0, layout["segment_indices"])
        self._parent = self.parent_endpoint_id.to(device=device).index_select(0, repeated_rows)
        self._child = self.child_endpoint_id.to(device=device).index_select(0, repeated_rows)
        self._fractions = layout["fractions"]
        self._active_counts = tuple(active_counts.detach().cpu().tolist())
        self.last_metadata = {
            "active_virtual_cell_count": int(repeated_rows.numel()),
            "active_segment_count": int(active.numel()),
            "active_segment_rows": [int(value) for value in active.cpu().tolist()],
            "density_to_z_gradient": False,
        }
        self._layout_key = key

    def __call__(self, pos, endpoint_x, endpoint_y, *, mode=None):
        self._prepare_layout(pos.device, pos.dtype)
        if self.last_metadata["active_virtual_cell_count"] == 0:
            return None
        # Layout/counts are static between B actions. Geometry and its autograd
        # graph are rebuilt each GP evaluation, preserving the endpoint chain.
        endpoint_x = endpoint_x.to(device=pos.device, dtype=pos.dtype)
        endpoint_y = endpoint_y.to(device=pos.device, dtype=pos.dtype)
        alpha = self._fractions
        virtual_x = (
            (1 - alpha) * endpoint_x.index_select(0, self._parent)
            + alpha * endpoint_x.index_select(0, self._child)
            - self.buffer_size_x * 0.5
        )
        virtual_y = (
            (1 - alpha) * endpoint_y.index_select(0, self._parent)
            + alpha * endpoint_y.index_select(0, self._child)
            - self.buffer_size_y * 0.5
        )
        combined_pos = build_combined_virtual_position(
            pos, virtual_x, virtual_y, self.buffer_size_x, self.buffer_size_y,
            num_movable_nodes=self.num_movable_nodes,
            num_terminals=self.num_terminals,
            num_filler_nodes=self.num_filler_nodes,
            num_fixed_nodes=self.num_fixed_nodes,
        )
        density_op = self._density_op(self._active_counts, pos.device, pos.dtype)
        return density_op(combined_pos) if mode is None else density_op(combined_pos, mode=mode)
