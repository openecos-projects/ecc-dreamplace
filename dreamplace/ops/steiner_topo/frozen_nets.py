"""Buffered-net lifetime and deterministic vertex mapping at tree refresh."""

import torch


class FrozenNets:
    def __init__(self):
        self.net_ids = ()
        self.vertex_map = None
        self.segment_state = None

    def freeze(self, topo, net_ids):
        if topo.num_vertices is None:
            raise RuntimeError("cannot retain nets before topology initialization")
        ids = tuple(sorted(set(self.net_ids).union(int(net) for net in net_ids)))
        nets = topo.flat_net2pin_start_map.numel() - 1
        if any(net < 0 or net >= nets for net in ids):
            raise ValueError("frozen net ID is outside the current net domain")
        self.net_ids = ids

    def build_options(self, topo):
        if not self.net_ids:
            return {}
        return {
            "frozen_net_ids": self.net_ids,
            "previous_cache": [
                topo.newx, topo.newy, topo.pin_relate_x, topo.pin_relate_y,
                topo.net_vertex_start, topo.net_steiner_start, topo.pin_fa,
                topo.flat_pin_to, topo.flat_pin_from, topo.flat_pin_to_start,
                topo.net_flat_topo_sort, topo.net_flat_topo_sort_start,
            ],
        }

    def unpack(self, outputs):
        if self.net_ids:
            self.vertex_map = outputs[-1]
            if self.segment_state is not None:
                self._remap_segments()
            return outputs[:-1]
        self.vertex_map = None
        return outputs

    def remap(self, vertex_ids):
        """Only physical pins and retained-net Steiner vertices have identities."""
        if self.vertex_map is None:
            return vertex_ids
        mapping = self.vertex_map.to(device=vertex_ids.device)
        if not bool(((vertex_ids >= 0) & (vertex_ids < mapping.numel())).all()):
            raise RuntimeError("retained segment vertex is outside the previous topology")
        result = mapping[vertex_ids.long()]
        if bool((result < 0).any()):
            raise RuntimeError("cannot map a Steiner vertex of an unfrozen net")
        return result.to(dtype=vertex_ids.dtype)

    def _remap_segments(self):
        state = self.segment_state
        prepared = state.prepared_timing_inputs
        # Validate every reference before publishing any changed tensor.
        state_values = {
            name: self.remap(getattr(state, name))
            for name in ("parent_node_id", "child_node_id")
        }
        prepared_values = {
            name: self.remap(prepared[name])
            for name in ("flat_topo_node_id", "edge_parent_node_id", "edge_child_node_id",
                         "sink_node_id", "driver_pin_id")
        }
        with torch.no_grad():
            for name, value in state_values.items():
                getattr(state, name).copy_(value)
            for name, value in prepared_values.items():
                prepared[name].copy_(value)
        parents = state.parent_node_id.detach().cpu().tolist()
        children = state.child_node_id.detach().cpu().tolist()
        state.segment_rows = tuple(
            dict(row, parent_node_id=parent, child_node_id=child)
            for row, parent, child in zip(state.segment_rows, parents, children, strict=True)
        )
