from dataclasses import dataclass

import torch
from torch import nn


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


def _edge_rc_record(edge_rc, parent, child):
    value = edge_rc.get((int(parent), int(child)))
    if value is None:
        value = edge_rc.get((str(int(parent)), str(int(child))))
    if value is None:
        return None
    if isinstance(value, dict):
        if "r" not in value and "c" not in value:
            return None
        return {
            "r": float(value.get("r", 0.0)),
            "c": float(value.get("c", 0.0)),
        }
    return {"r": float(value), "c": 0.0}


def _tree_topology_preorder(root_node_id, children_by_node):
    stack = [int(root_node_id)]
    seen = set()
    nodes = []
    edges = []
    while stack:
        parent = int(stack.pop())
        if parent in seen:
            continue
        seen.add(parent)
        nodes.append(parent)
        children = [int(child) for child in children_by_node.get(parent, [])]
        for child in children:
            edges.append((parent, child))
        stack.extend(reversed(children))
    return nodes, edges


def _length_sq(coordinates, parent, child):
    px, py = coordinates[int(parent)]
    cx, cy = coordinates[int(child)]
    dx = float(cx) - float(px)
    dy = float(cy) - float(py)
    return dx * dx + dy * dy


@dataclass
class SegmentCountState:
    segment_ids: torch.Tensor
    segment_net_id: torch.Tensor
    segment_net_index: torch.Tensor
    net_ids: torch.Tensor
    parent_node_id: torch.Tensor
    child_node_id: torch.Tensor
    z_param: nn.Parameter
    bsu_index_param: nn.Parameter
    legal_buffer_count: int
    buffer_main_type_index: int
    max_repeater_count: int
    fixed_bsu_index: int | None
    segment_rows: tuple | None
    net_to_segment_rows: dict | None
    summary: dict
    packed_segment_geometry: dict | None = None
    prepared_timing_inputs: dict | None = None
    net_record_resolver: object = None

    @property
    def num_segments(self):
        return int(self.z_param.numel())

    @property
    def is_tensor_backed(self):
        return self.packed_segment_geometry is not None

    def _record_materialization(self, kind, count):
        key = f"{kind}_row_materialization_count"
        self.summary[key] = int(self.summary.get(key, 0)) + int(count)

    def segment_record(self, index, *, materialization_kind="selected"):
        index = int(index)
        if index < 0 or index >= self.num_segments:
            raise IndexError("segment index out of range")
        if self.segment_rows is not None:
            row = dict(self.segment_rows[index])
        else:
            geometry = self.packed_segment_geometry

            def scalar(name):
                return geometry[name][index].item()

            row = {
                "segment_id": int(scalar("segment_ids")),
                "net_id": int(scalar("segment_net_id")),
                "parent_node_id": int(scalar("parent_node_id")),
                "child_node_id": int(scalar("child_node_id")),
                "parent_x_dbu": int(scalar("parent_x_dbu")),
                "parent_y_dbu": int(scalar("parent_y_dbu")),
                "child_x_dbu": int(scalar("child_x_dbu")),
                "child_y_dbu": int(scalar("child_y_dbu")),
                "edge_rc": {
                    "r": float(scalar("edge_resistance")),
                    "c": float(scalar("edge_capacitance")),
                },
                "buffer_main_type_index": int(self.buffer_main_type_index),
                "max_repeater_count": int(self.max_repeater_count),
            }
        self._record_materialization(materialization_kind, 1)
        return row

    def segment_records(self, indices, *, materialization_kind="selected"):
        return tuple(
            self.segment_record(index, materialization_kind=materialization_kind)
            for index in indices
        )

    def positive_segment_indices(self):
        return torch.nonzero(self.z_param.detach() > 0.0, as_tuple=False).flatten()

    def segment_net_ids_for_indices(self, indices):
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.segment_net_id.device,
        ).flatten()
        if int(index.numel()) == 0:
            return ()
        values = self.segment_net_id.index_select(0, index)
        return tuple(
            int(value)
            for value in torch.unique(values).detach().cpu().tolist()
        )

    def segment_indices_for_nets(self, net_ids):
        if not self.is_tensor_backed:
            rows = []
            for net_id in sorted(int(value) for value in net_ids):
                rows.extend((self.net_to_segment_rows or {}).get(net_id, ()))
            return tuple(int(value) for value in rows)
        geometry = self.packed_segment_geometry
        packed_net_ids = geometry["net_ids"].to(device="cpu", dtype=torch.long)
        starts = geometry["net_segment_start"].to(device="cpu", dtype=torch.long)
        query = torch.as_tensor(
            sorted(set(int(value) for value in net_ids)),
            dtype=torch.long,
        )
        if int(query.numel()) == 0 or int(packed_net_ids.numel()) == 0:
            return ()
        positions = torch.searchsorted(packed_net_ids, query)
        valid = positions < int(packed_net_ids.numel())
        if bool(valid.any()):
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            valid[valid_indices] &= packed_net_ids[positions[valid_indices]] == query[valid_indices]
        rows = []
        for position in positions[valid].tolist():
            rows.extend(range(int(starts[position]), int(starts[position + 1])))
        return tuple(rows)

    def net_records(self, net_ids):
        if not callable(self.net_record_resolver):
            return ()
        return tuple(self.net_record_resolver(tuple(sorted(set(int(value) for value in net_ids)))))

    def z_value(self):
        return torch.clamp(self.z_param, 0.0, float(self.max_repeater_count))

    def bsu_index(self):
        return torch.clamp(self.bsu_index_param, 0.0, float(self.legal_buffer_count - 1))

    def project_(self):
        with torch.no_grad():
            self.z_param.clamp_(0.0, float(self.max_repeater_count))
            self.bsu_index_param.clamp_(0.0, float(self.legal_buffer_count - 1))
        return self


def build_segment_count_state(
    nets,
    *,
    buffer_main_type_index,
    legal_buffer_count,
    max_repeater_count,
    initial_z=0.05,
    initial_bsu_index=7.0,
    fixed_bsu_index=None,
    dtype=torch.float32,
    device=None,
    max_nets=None,
    max_candidates=None,
    retained_edges=(),
):
    if max_nets is not None or max_candidates is not None:
        raise ValueError("segment-count state requires full-design segment scope")
    retained_edges = set(retained_edges)
    legal_buffer_count = int(legal_buffer_count)
    max_repeater_count = int(max_repeater_count)
    if legal_buffer_count < 2:
        raise ValueError("legal_buffer_count must be at least 2")
    if max_repeater_count < 1:
        raise ValueError("max_repeater_count must be at least 1")
    if fixed_bsu_index is not None:
        fixed_bsu_index = int(fixed_bsu_index)
        if not 0 <= fixed_bsu_index < legal_buffer_count:
            raise ValueError("fixed_bsu_index must reference a legal buffer")
    bsu_initial_value = (
        float(fixed_bsu_index)
        if fixed_bsu_index is not None
        else float(initial_bsu_index)
    )

    tensor_device = torch.device(device) if device is not None else None
    segment_rows = []
    net_to_segment_rows = {}
    skipped_missing_coordinate = 0
    skipped_missing_rc = 0
    skipped_zero_length = 0

    for net_index, net in enumerate(list(nets or [])):
        net_id = _net_id(net, net_index)
        rc_tree = dict(net.get("rc_tree", {}) or {})
        children_by_node = {
            int(node): [int(child) for child in children]
            for node, children in (rc_tree.get("children_by_node", {}) or {}).items()
        }
        root = rc_tree.get("root_node_id", net.get("driver_pin_id"))
        if root is None:
            continue
        coordinates = {
            int(node): tuple(value)
            for node, value in (net.get("coordinates", {}) or {}).items()
        }
        # Segment timing remains defined by ``coordinates``.  The state rows
        # are also consumed by final coordinate projection and fixed-placement
        # resource geometry, which require raw OpenDB/DBU locations whenever
        # the adapter exported them.
        coordinates_dbu = {
            int(node): tuple(value)
            for node, value in (net.get("coordinates_dbu", coordinates) or {}).items()
        }
        edge_rc = rc_tree.get("edge_rc", {}) or {}

        topo_nodes, topo_edges = _tree_topology_preorder(root, children_by_node)
        node_to_local = {
            int(node_id): int(local_id)
            for local_id, node_id in enumerate(topo_nodes)
        }
        static_edge_start = []
        static_edge_parent_node_id = []
        static_edge_child_node_id = []
        static_edge_parent_local_id = []
        static_edge_child_local_id = []
        static_edge_resistance = []
        static_edge_capacitance = []
        static_edge_global_segment_id = []
        static_node_capacitance = []
        for parent in topo_nodes:
            static_edge_start.append(len(static_edge_parent_node_id))
            static_node_capacitance.append(
                float((rc_tree.get("node_cap", {}) or {}).get(int(parent), 0.0))
            )
            for child in children_by_node.get(int(parent), []):
                rc_record = _edge_rc_record(edge_rc, parent, child)
                static_edge_parent_node_id.append(int(parent))
                static_edge_child_node_id.append(int(child))
                static_edge_parent_local_id.append(int(node_to_local[int(parent)]))
                static_edge_child_local_id.append(int(node_to_local[int(child)]))
                static_edge_resistance.append(
                    0.0 if rc_record is None else float(rc_record["r"])
                )
                static_edge_capacitance.append(
                    0.0 if rc_record is None else float(rc_record["c"])
                )
                static_edge_global_segment_id.append(-1)

        edge_offset_by_pair = {
            (int(parent), int(child)): int(edge_offset)
            for edge_offset, (parent, child) in enumerate(topo_edges)
        }
        for parent, child in topo_edges:
            if (
                parent not in coordinates
                or child not in coordinates
                or parent not in coordinates_dbu
                or child not in coordinates_dbu
            ):
                skipped_missing_coordinate += 1
                continue
            rc_record = _edge_rc_record(edge_rc, parent, child)
            if rc_record is None:
                skipped_missing_rc += 1
                continue
            if (
                _length_sq(coordinates, parent, child) <= 0.0
                and (net_id, parent, child) not in retained_edges
            ):
                skipped_zero_length += 1
                continue
            row = {
                "segment_id": len(segment_rows),
                "net_id": int(net_id),
                "net_name": net.get("net_name"),
                "parent_node_id": int(parent),
                "child_node_id": int(child),
                "parent_x_dbu": int(round(float(coordinates_dbu[parent][0]))),
                "parent_y_dbu": int(round(float(coordinates_dbu[parent][1]))),
                "child_x_dbu": int(round(float(coordinates_dbu[child][0]))),
                "child_y_dbu": int(round(float(coordinates_dbu[child][1]))),
                "edge_rc": dict(rc_record),
                "buffer_main_type_index": int(buffer_main_type_index),
                "max_repeater_count": int(max_repeater_count),
            }
            net_to_segment_rows.setdefault(int(net_id), []).append(len(segment_rows))
            static_edge_global_segment_id[edge_offset_by_pair[(int(parent), int(child))]] = (
                int(row["segment_id"])
            )
            segment_rows.append(row)

        static_edge_start.append(len(static_edge_parent_node_id))
        static_sink_node_id = []
        static_sink_local_id = []
        for sink_node_id in rc_tree.get("sink_nodes", []) or []:
            sink_node_id = int(sink_node_id)
            if sink_node_id in node_to_local:
                static_sink_node_id.append(sink_node_id)
                static_sink_local_id.append(int(node_to_local[sink_node_id]))
        net["_segment_count_static_topology"] = {
            "net_id": int(net_id),
            "root_node_id": int(root),
            "flat_topo_node_id": tuple(int(node_id) for node_id in topo_nodes),
            "edge_start": tuple(int(value) for value in static_edge_start),
            "edge_parent_node_id": tuple(static_edge_parent_node_id),
            "edge_child_node_id": tuple(static_edge_child_node_id),
            "edge_parent_local_id": tuple(static_edge_parent_local_id),
            "edge_child_local_id": tuple(static_edge_child_local_id),
            "edge_resistance": tuple(static_edge_resistance),
            "edge_capacitance": tuple(static_edge_capacitance),
            "edge_global_segment_id": tuple(static_edge_global_segment_id),
            "node_capacitance": tuple(static_node_capacitance),
            "sink_node_id": tuple(static_sink_node_id),
            "sink_local_id": tuple(static_sink_local_id),
        }

    segment_count = len(segment_rows)
    z_param = nn.Parameter(
        torch.full(
            (segment_count,),
            float(initial_z),
            dtype=dtype,
            device=tensor_device,
        )
    )
    bsu_index_param = nn.Parameter(
        torch.full(
            (segment_count,),
            bsu_initial_value,
            dtype=dtype,
            device=tensor_device,
        )
    )
    net_ids = sorted(net_to_segment_rows)
    net_id_to_index = {net_id: index for index, net_id in enumerate(net_ids)}
    summary = {
        "artifact": "segment_count_state_summary",
        "artifact_version": 1,
        "status": "ok",
        "full_design_scope": True,
        "max_nets": None,
        "max_candidates": None,
        "eligible_segment_count": int(segment_count),
        "segment_count_state_count": int(segment_count),
        "affected_net_count": int(len(net_to_segment_rows)),
        "compact_net_index_count": int(len(net_ids)),
        "skipped_missing_coordinate_count": int(skipped_missing_coordinate),
        "skipped_missing_rc_count": int(skipped_missing_rc),
        "skipped_zero_length_count": int(skipped_zero_length),
        "buffer_main_type_index": int(buffer_main_type_index),
        "legal_buffer_count": int(legal_buffer_count),
        "max_repeater_count": int(max_repeater_count),
        "fixed_bsu_index": fixed_bsu_index,
        "segment_shared_bsu": True,
    }

    return SegmentCountState(
        segment_ids=torch.tensor(
            [row["segment_id"] for row in segment_rows],
            dtype=torch.long,
            device=tensor_device,
        ),
        segment_net_id=torch.tensor(
            [row["net_id"] for row in segment_rows],
            dtype=torch.long,
            device=tensor_device,
        ),
        segment_net_index=torch.tensor(
            [net_id_to_index[int(row["net_id"])] for row in segment_rows],
            dtype=torch.long,
            device=tensor_device,
        ),
        net_ids=torch.tensor(
            net_ids,
            dtype=torch.long,
            device=tensor_device,
        ),
        parent_node_id=torch.tensor(
            [row["parent_node_id"] for row in segment_rows],
            dtype=torch.long,
            device=tensor_device,
        ),
        child_node_id=torch.tensor(
            [row["child_node_id"] for row in segment_rows],
            dtype=torch.long,
            device=tensor_device,
        ),
        z_param=z_param,
        bsu_index_param=bsu_index_param,
        legal_buffer_count=int(legal_buffer_count),
        buffer_main_type_index=int(buffer_main_type_index),
        max_repeater_count=int(max_repeater_count),
        fixed_bsu_index=fixed_bsu_index,
        segment_rows=tuple(segment_rows),
        net_to_segment_rows={
            int(net_id): list(rows)
            for net_id, rows in sorted(net_to_segment_rows.items())
        },
        summary=summary,
    )


def build_packed_segment_count_state(
    packed_result,
    *,
    buffer_main_type_index,
    legal_buffer_count,
    max_repeater_count,
    initial_z=0.0,
    initial_bsu_index=7.0,
    fixed_bsu_index=None,
    dtype=torch.float32,
    device=None,
    net_record_resolver=None,
):
    """Create Route-B optimization state without Python segment dictionaries."""

    geometry = dict(packed_result["packed_segment_geometry"])
    prepared = dict(packed_result["prepared_timing_inputs"])
    metadata = dict(packed_result.get("metadata") or {})
    segment_count = int(geometry["segment_ids"].numel())
    tensor_device = torch.device(device) if device is not None else torch.device("cpu")
    legal_buffer_count = int(legal_buffer_count)
    if fixed_bsu_index is not None:
        fixed_bsu_index = int(fixed_bsu_index)
        if not 0 <= fixed_bsu_index < legal_buffer_count:
            raise ValueError("fixed_bsu_index must reference a legal buffer")
    bsu_initial_value = (
        float(fixed_bsu_index)
        if fixed_bsu_index is not None
        else float(initial_bsu_index)
    )

    def state_long(name):
        return geometry[name].to(device=tensor_device, dtype=torch.long).contiguous()

    summary = {
        "artifact": "segment_count_state_summary",
        "artifact_version": 1,
        "status": "ok",
        "full_design_scope": True,
        "max_nets": None,
        "max_candidates": None,
        "eligible_segment_count": segment_count,
        "segment_count_state_count": segment_count,
        "affected_net_count": int(geometry["net_ids"].numel()),
        "compact_net_index_count": int(geometry["net_ids"].numel()),
        "buffer_main_type_index": int(buffer_main_type_index),
        "legal_buffer_count": int(legal_buffer_count),
        "max_repeater_count": int(max_repeater_count),
        "fixed_bsu_index": fixed_bsu_index,
        "segment_shared_bsu": True,
        "segment_state_storage": "native_packed_tensors",
        "full_row_materialization_count": 0,
        "selected_row_materialization_count": 0,
        "terminal_row_materialization_count": 0,
        "native_packer_thread_count": int(metadata.get("thread_count", 0)),
    }
    return SegmentCountState(
        segment_ids=state_long("segment_ids"),
        segment_net_id=state_long("segment_net_id"),
        segment_net_index=state_long("segment_net_index"),
        net_ids=state_long("net_ids"),
        parent_node_id=state_long("parent_node_id"),
        child_node_id=state_long("child_node_id"),
        z_param=nn.Parameter(
            torch.full(
                (segment_count,),
                float(initial_z),
                dtype=dtype,
                device=tensor_device,
            )
        ),
        bsu_index_param=nn.Parameter(
            torch.full(
                (segment_count,),
                bsu_initial_value,
                dtype=dtype,
                device=tensor_device,
            )
        ),
        legal_buffer_count=int(legal_buffer_count),
        buffer_main_type_index=int(buffer_main_type_index),
        max_repeater_count=int(max_repeater_count),
        fixed_bsu_index=fixed_bsu_index,
        segment_rows=None,
        net_to_segment_rows=None,
        summary=summary,
        packed_segment_geometry=geometry,
        prepared_timing_inputs=prepared,
        net_record_resolver=net_record_resolver,
    )
