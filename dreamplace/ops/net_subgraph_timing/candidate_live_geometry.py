"""Differentiable RC for a fixed candidate tree at the current GP coordinates."""

from dataclasses import dataclass

import torch

from .segment_count_live_geometry import build_live_edge_geometry


@dataclass(frozen=True)
class CandidateSegmentColumns:
    """Original endpoints and ratios, in the native packer's candidate order."""

    parents: torch.Tensor
    children: torch.Tensor
    ratios: torch.Tensor


class CandidateLiveGeometry:
    def __init__(self, payload, state):
        self.payload = payload
        self.buffer_state = state
        self.original_nodes = torch.as_tensor(
            payload["metadata"]["compact_node_to_original"], dtype=torch.long
        )
        self.parents = payload["pin_fa"].long()
        self.sinks = payload["sink_node_id"].long()
        self.sink_pins = payload["sink_pin_id"].long()
        columns = payload.get("candidate_segment_columns")
        if columns is not None:
            # The packer preserves candidate row order and rejects unsupported
            # nets before publishing this payload. Remapping changes only IDs.
            self.split_nodes = payload["candidate_node_id"].long().clone()
            self.split_parents = columns.parents.clone()
            self.split_children = columns.children.clone()
            self.split_ratios = columns.ratios.clone()
            return
        rows = [
            index
            for index, record in enumerate(state.candidate_records)
            if record.get("parent_node_id") is not None
            and len(record.get("child_node_ids", ())) == 1
            and record.get("segment_split_ratio") is not None
        ]
        records = [state.candidate_records[index] for index in rows]
        self.split_nodes = payload["candidate_node_id"].long()[rows]
        self.split_parents = torch.tensor(
            [record["parent_node_id"] for record in records], dtype=torch.long
        )
        self.split_children = torch.tensor(
            [record["child_node_ids"][0] for record in records], dtype=torch.long
        )
        self.split_ratios = torch.tensor(
            [record["segment_split_ratio"] for record in records], dtype=torch.float64
        )

    def bind(self, *, new_x, new_y, pin_capacitance_by_sense=None, **units):
        if "base" not in (pin_capacitance_by_sense or {}):
            raise ValueError("live candidate RC requires current pin capacitances")
        # A discrete type probe holds placement and cell masters fixed. Their
        # cached geometry can still carry a graph from the preceding S phase.
        # Keep those values, while ordinary GP retains all geometry gradients.
        if getattr(self.buffer_state, "_gradient_drive_type", None) is not None or getattr(
            self.buffer_state, "_shared_gradient_probe", False
        ):
            new_x, new_y = new_x.detach(), new_y.detach()
            pin_capacitance_by_sense = {
                sense: caps.detach() for sense, caps in pin_capacitance_by_sense.items()
            }
        original = self.original_nodes.to(new_x.device)
        split_nodes = self.split_nodes.to(new_x.device)
        invalid_original = original >= int(new_x.numel())
        if bool(invalid_original.any().item()):
            invalid_indices = invalid_original.nonzero(as_tuple=False).view(-1)
            sample = invalid_indices[:8].detach().cpu().tolist()
            raise IndexError(
                "candidate topology node is outside live coordinate domain: "
                f"coordinate_count={int(new_x.numel())}, "
                f"original_node_min={int(original.min().item()) if original.numel() else None}, "
                f"original_node_max={int(original.max().item()) if original.numel() else None}, "
                f"invalid_indices={sample}, "
                f"invalid_nodes={[int(original[index].item()) for index in sample]}"
            )
        covered = original >= 0
        covered[split_nodes] = True
        if not bool(covered.all()):
            raise ValueError("candidate tree contains an unmapped synthetic node")
        parent = self.split_parents.to(new_x.device)
        child = self.split_children.to(new_x.device)
        invalid_split = (parent < 0) | (parent >= int(new_x.numel()))
        invalid_split |= (child < 0) | (child >= int(new_x.numel()))
        if bool(invalid_split.any().item()):
            invalid_indices = invalid_split.nonzero(as_tuple=False).view(-1)
            sample = invalid_indices[:8].detach().cpu().tolist()
            raise IndexError(
                "candidate segment endpoint is outside live coordinate domain: "
                f"coordinate_count={int(new_x.numel())}, "
                f"invalid_indices={sample}, "
                f"invalid_parents={[int(parent[index].item()) for index in sample]}, "
                f"invalid_children={[int(child[index].item()) for index in sample]}"
            )
        ratio = self.split_ratios.to(new_x)

        def coordinates(values):
            compact = values.index_select(0, original.clamp(min=0))
            splits = (1 - ratio) * values[parent] + ratio * values[child]
            return compact.index_copy(0, split_nodes, splits)

        x, y = coordinates(new_x), coordinates(new_y)
        parents = self.parents.to(x.device)
        geometry = build_live_edge_geometry(
            x,
            y,
            parents.clamp(min=0),
            torch.arange(x.numel(), device=x.device),
            **units,
        )
        nonroot = parents >= 0
        self.rc = {
            name: torch.where(nonroot, geometry[name], 0.0)
            for name in ("edge_resistance", "edge_capacitance")
        }
        self.node_caps = {}
        for sense, caps in pin_capacitance_by_sense.items():
            # Packed candidate trees already distribute wire C into node C.
            # Replace the frozen values with pin-only loads before the timing
            # operator distributes capacitance from the current edges.
            base = torch.zeros_like(
                self.payload["node_capacitance"], device=caps.device, dtype=caps.dtype
            )
            self.node_caps[sense] = base.index_copy(
                0, self.sinks.to(caps.device), caps[self.sink_pins.to(caps.device)]
            )

    def forward_overrides(self, *, active_view, device, dtype, sense):
        values = dict(self.rc)
        values["node_capacitance"] = self.node_caps.get(sense, self.node_caps["base"])
        if active_view is not None:
            indices = active_view["node_indices_cpu"]
            values = {
                name: value.index_select(0, indices.to(value.device))
                for name, value in values.items()
            }
        return {name: value.to(device=device, dtype=dtype) for name, value in values.items()}

    def release(self):
        self.rc = {}
        self.node_caps = {}
