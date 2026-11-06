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
        else:
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
        num_compact = int(self.original_nodes.numel())
        self._original_clamped = self.original_nodes.clamp(min=0)
        self._parents_clamped = self.parents.clamp(min=0)
        self._nonroot = self.parents >= 0
        self._node_is_split = torch.zeros(num_compact, dtype=torch.bool)
        self._node_parent_pin = self._original_clamped.clone()
        self._node_child_pin = self._original_clamped.clone()
        self._node_ratio = torch.zeros(num_compact, dtype=torch.float64)
        if int(self.split_nodes.numel()):
            self._node_is_split[self.split_nodes] = True
            self._node_parent_pin[self.split_nodes] = self.split_parents
            self._node_child_pin[self.split_nodes] = self.split_children
            self._node_ratio[self.split_nodes] = self.split_ratios.to(dtype=torch.float64)
        self._node_sink_pin = torch.full((num_compact,), -1, dtype=torch.long)
        if int(self.sinks.numel()):
            self._node_sink_pin[self.sinks] = self.sink_pins
        self._validated_coord_counts = set()
        self._rc = {}
        self._node_caps = {}
        self._bound_inputs = None
        self._active_override_cache = {}
        self._detached_rc_cache = None

    def _validate_domain(self, coord_count, device):
        key = (int(coord_count), device)
        if key in self._validated_coord_counts:
            return
        original = self.original_nodes.to(device)
        split_nodes = self.split_nodes.to(device)
        invalid_original = original >= int(coord_count)
        if bool(invalid_original.any().item()):
            invalid_indices = invalid_original.nonzero(as_tuple=False).view(-1)
            sample = invalid_indices[:8].detach().cpu().tolist()
            raise IndexError(
                "candidate topology node is outside live coordinate domain: "
                f"coordinate_count={int(coord_count)}, "
                f"original_node_min={int(original.min().item()) if original.numel() else None}, "
                f"original_node_max={int(original.max().item()) if original.numel() else None}, "
                f"invalid_indices={sample}, "
                f"invalid_nodes={[int(original[index].item()) for index in sample]}"
            )
        covered = original >= 0
        covered[split_nodes] = True
        if not bool(covered.all()):
            raise ValueError("candidate tree contains an unmapped synthetic node")
        parent = self.split_parents.to(device)
        child = self.split_children.to(device)
        invalid_split = (parent < 0) | (parent >= int(coord_count))
        invalid_split |= (child < 0) | (child >= int(coord_count))
        if bool(invalid_split.any().item()):
            invalid_indices = invalid_split.nonzero(as_tuple=False).view(-1)
            sample = invalid_indices[:8].detach().cpu().tolist()
            raise IndexError(
                "candidate segment endpoint is outside live coordinate domain: "
                f"coordinate_count={int(coord_count)}, "
                f"invalid_indices={sample}, "
                f"invalid_parents={[int(parent[index].item()) for index in sample]}, "
                f"invalid_children={[int(child[index].item()) for index in sample]}"
            )
        self._validated_coord_counts.add(key)

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
        self._validate_domain(int(new_x.numel()), new_x.device)
        self._bound_inputs = (new_x, new_y, pin_capacitance_by_sense, units)
        self._rc = {}
        self._node_caps = {}
        self._active_override_cache = {}

    def _ensure_full_rc(self):
        if self._rc or self._bound_inputs is None:
            return self._rc
        new_x, new_y, _pin_caps, units = self._bound_inputs
        is_detached = not new_x.requires_grad and not new_y.requires_grad
        cache_key = None
        if is_detached and not units.get("require_axis_aligned", False):
            cache_key = (
                new_x.data_ptr(),
                new_x._version,
                new_y.data_ptr(),
                new_y._version,
                new_x.device,
                new_x.dtype,
                float(units["r_unit"]),
                float(units["c_unit"]),
                float(units["scale_factor"]),
                float(units["dbu"]),
            )
            if self._detached_rc_cache is not None and self._detached_rc_cache[0] == cache_key:
                self._rc = self._detached_rc_cache[1]
                return self._rc
        original_clamped = self._original_clamped.to(new_x.device)
        split_nodes = self.split_nodes.to(new_x.device)
        parent = self.split_parents.to(new_x.device)
        child = self.split_children.to(new_x.device)
        ratio = self.split_ratios.to(new_x)

        def coordinates(values):
            compact = values.index_select(0, original_clamped)
            splits = (1 - ratio) * values[parent] + ratio * values[child]
            return compact.index_copy(0, split_nodes, splits)

        x, y = coordinates(new_x), coordinates(new_y)
        parents_clamped = self._parents_clamped.to(x.device)
        nonroot = self._nonroot.to(x.device)
        if units.get("require_axis_aligned", False):
            geometry = build_live_edge_geometry(
                x,
                y,
                parents_clamped,
                torch.arange(x.numel(), device=x.device),
                **units,
            )
            self._rc = {
                name: torch.where(nonroot, geometry[name], 0.0)
                for name in ("edge_resistance", "edge_capacitance")
            }
        else:
            scale_factor = float(units["scale_factor"])
            dbu = float(units["dbu"])
            r_unit = float(units["r_unit"])
            c_unit = float(units["c_unit"])
            dx = x - x.index_select(0, parents_clamped)
            dy = y - y.index_select(0, parents_clamped)
            length = (torch.abs(dx) + torch.abs(dy)) / scale_factor / dbu
            self._rc = {
                "edge_resistance": torch.where(nonroot, length * r_unit, 0.0),
                "edge_capacitance": torch.where(nonroot, length * c_unit, 0.0),
            }
        if cache_key is not None:
            self._detached_rc_cache = (cache_key, self._rc)
        return self._rc

    def _ensure_full_node_caps(self, sense=None):
        if self._bound_inputs is None:
            return self._node_caps
        _new_x, _new_y, pin_capacitance_by_sense, _units = self._bound_inputs
        senses = (
            pin_capacitance_by_sense.keys()
            if sense is None
            else (sense if sense in pin_capacitance_by_sense else "base",)
        )
        for key in senses:
            if key in self._node_caps:
                continue
            caps = pin_capacitance_by_sense[key]
            # Packed candidate trees already distribute wire C into node C.
            # Replace the frozen values with pin-only loads before the timing
            # operator distributes capacitance from the current edges.
            base = torch.zeros_like(
                self.payload["node_capacitance"], device=caps.device, dtype=caps.dtype
            )
            self._node_caps[key] = base.index_copy(
                0, self.sinks.to(caps.device), caps[self.sink_pins.to(caps.device)]
            )
        return self._node_caps

    @property
    def rc(self):
        return self._ensure_full_rc()

    @rc.setter
    def rc(self, value):
        self._rc = value

    @property
    def node_caps(self):
        return self._ensure_full_node_caps()

    @node_caps.setter
    def node_caps(self, value):
        self._node_caps = value

    def _coordinates_for_nodes(self, values, indices_cpu):
        p_pin = self._node_parent_pin.index_select(0, indices_cpu).to(values.device)
        base = values.index_select(0, p_pin)
        is_split_cpu = self._node_is_split.index_select(0, indices_cpu)
        if not bool(is_split_cpu.any()):
            return base
        c_pin = self._node_child_pin.index_select(0, indices_cpu).to(values.device)
        ratio = self._node_ratio.index_select(0, indices_cpu).to(values)
        splits = (1 - ratio) * base + ratio * values.index_select(0, c_pin)
        return torch.where(is_split_cpu.to(values.device), splits, base)

    def _active_forward_overrides(self, *, active_view, device, dtype, sense):
        new_x, new_y, pin_capacitance_by_sense, units = self._bound_inputs
        resolved_sense = sense if sense in pin_capacitance_by_sense else "base"
        view_key = active_view.get("cache_key")
        cache_key = (
            (view_key, device, dtype, resolved_sense)
            if view_key is not None
            else None
        )
        if cache_key is not None and cache_key in self._active_override_cache:
            return self._active_override_cache[cache_key]
        indices_cpu = active_view["node_indices_cpu"].detach().cpu()
        rc_key = (view_key, device, dtype, "_rc") if view_key is not None else None
        cached_rc = self._active_override_cache.get(rc_key) if rc_key is not None else None
        if cached_rc is None:
            x_sub = self._coordinates_for_nodes(new_x, indices_cpu)
            y_sub = self._coordinates_for_nodes(new_y, indices_cpu)
            local_pin_fa = (active_view.get("static") or {}).get("pin_fa")
            if local_pin_fa is not None:
                local_parents = local_pin_fa.to(device=x_sub.device, dtype=torch.long)
                parents_clamped = local_parents.clamp(min=0)
                nonroot = local_parents >= 0
                x_parent = x_sub.index_select(0, parents_clamped)
                y_parent = y_sub.index_select(0, parents_clamped)
            else:
                global_parents_cpu = self.parents.index_select(0, indices_cpu)
                parents_clamped_cpu = global_parents_cpu.clamp(min=0)
                nonroot = global_parents_cpu.to(device=x_sub.device) >= 0
                x_parent = self._coordinates_for_nodes(new_x, parents_clamped_cpu)
                y_parent = self._coordinates_for_nodes(new_y, parents_clamped_cpu)
            scale_factor = float(units["scale_factor"])
            dbu = float(units["dbu"])
            r_unit = float(units["r_unit"])
            c_unit = float(units["c_unit"])
            dx = x_sub - x_parent
            dy = y_sub - y_parent
            if units.get("require_axis_aligned", False):
                if bool(((dx != 0) & (dy != 0)).any().item()):
                    raise ValueError("live segment topology contains a diagonal edge")
            length = (torch.abs(dx) + torch.abs(dy)) / scale_factor / dbu
            cached_rc = {
                "edge_resistance": torch.where(nonroot, length * r_unit, 0.0).to(
                    device=device, dtype=dtype
                ),
                "edge_capacitance": torch.where(nonroot, length * c_unit, 0.0).to(
                    device=device, dtype=dtype
                ),
            }
            if rc_key is not None:
                self._active_override_cache[rc_key] = cached_rc
        caps = pin_capacitance_by_sense[resolved_sense]
        sink_pins_cpu = self._node_sink_pin.index_select(0, indices_cpu)
        sink_mask_cpu = sink_pins_cpu >= 0
        node_cap = torch.zeros(
            int(indices_cpu.numel()), device=caps.device, dtype=caps.dtype
        )
        if bool(sink_mask_cpu.any()):
            local_sink_pos = sink_mask_cpu.nonzero(as_tuple=False).view(-1)
            valid_pins = sink_pins_cpu.index_select(0, local_sink_pos).to(caps.device)
            node_cap = node_cap.index_copy(
                0,
                local_sink_pos.to(caps.device),
                caps.index_select(0, valid_pins),
            )
        result = {
            "edge_resistance": cached_rc["edge_resistance"],
            "edge_capacitance": cached_rc["edge_capacitance"],
            "node_capacitance": node_cap.to(device=device, dtype=dtype),
        }
        if cache_key is not None:
            self._active_override_cache[cache_key] = result
        return result

    def forward_overrides(self, *, active_view, device, dtype, sense):
        if active_view is not None and not self._rc and self._bound_inputs is not None:
            return self._active_forward_overrides(
                active_view=active_view,
                device=device,
                dtype=dtype,
                sense=sense,
            )
        values = dict(self.rc)
        node_caps = self._ensure_full_node_caps(sense)
        values["node_capacitance"] = node_caps.get(sense, node_caps["base"])
        if active_view is not None:
            indices = active_view["node_indices_cpu"]
            values = {
                name: value.index_select(0, indices.to(value.device))
                for name, value in values.items()
            }
        return {name: value.to(device=device, dtype=dtype) for name, value in values.items()}

    def release(self):
        self._rc = {}
        self._node_caps = {}
        self._bound_inputs = None
        self._active_override_cache = {}
