"""In-process CPU CUGR backend for the DreamPlace routability contract.

The native module exposes primitive per-layer maps and route trees.  This
adapter owns the conversion to the existing DreamPlace map/metric/topology
contract; it never invokes the legacy CUGR CLI or parses guide/stdout files.
"""

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .cugr_evidence import write_cugr_evidence
from .result_maps import (
    aggregate_layer_maps,
    summarize_congestion_tensor,
    validate_map_contract,
)
from .xplace_backend import (
    XplaceGPUGR,
    _preload_libpython,
    normalize_gpugr_backend,
)

logger = logging.getLogger(__name__)


def _as_int32(values):
    return np.asarray(values, dtype=np.int32)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU map including shape and dtype for refresh evidence."""

    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class CugrGPUGR(XplaceGPUGR):
    """Run the source-built CUGR core and normalize its primitive result."""

    def _design_identity(self):
        """Return a stable identity for the cached immutable design graph.

        Placement coordinates are deliberately excluded so coordinate-refresh
        calls hit the native session cache. Names and connectivity are included
        because reusing a session for a different graph is unsafe even when the
        design name and component count happen to match.
        """

        digest = hashlib.sha256()

        def add_sequence(label, values):
            digest.update(label.encode("utf-8"))
            if values is None:
                digest.update(b"<none>\0")
                return
            for value in values:
                if isinstance(value, bytes):
                    encoded = value
                else:
                    encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)

        add_sequence("nodes\0", getattr(self.placedb, "node_names", None))
        add_sequence("nets\0", getattr(self.placedb, "net_names", None))
        add_sequence("flat_net2pin\0", getattr(self.placedb, "flat_net2pin_map", None))
        add_sequence(
            "flat_net2pin_start\0",
            getattr(self.placedb, "flat_net2pin_start_map", None),
        )
        for name in ("num_physical_nodes", "num_pins", "num_nets"):
            digest.update(name.encode("utf-8"))
            digest.update(str(int(getattr(self.placedb, name, 0))).encode("ascii"))
            digest.update(b"\0")
        return "sha256:" + digest.hexdigest()

    def _import_cugr(self):
        root = self._ensure_xplace_python_path()
        cpybin = root / "cpp_to_py" / "cpybin"
        if not cpybin.is_dir() or not any(cpybin.glob("cugr*.so")):
            raise RuntimeError(
                f"CUGR extension is not built (missing cugr*.so under {cpybin}); "
                "build Xplace with the CPU CUGR target first"
            )
        try:
            # CUGR is a pybind shared object with a DT_NEEDED entry for
            # libpython on the uv-managed interpreter.  The CUGR path can be
            # the first Xplace path used by a process, so it must establish
            # this loader invariant itself instead of relying on CUDA GGR or
            # another test to have run first.
            _preload_libpython()
            # Import the extension package explicitly.  The Xplace checkout
            # also contains a source directory named ``cpp_to_py/cugr``;
            # importing from the package root can therefore resolve that
            # directory instead of the compiled module.
            from cpp_to_py.cpybin import cugr
        except Exception as exc:
            raise RuntimeError("Failed to import the source-built CUGR extension") from exc
        if not bool(cugr.available()):
            raise RuntimeError("CUGR extension reported unavailable")
        return cugr

    @staticmethod
    def _primitive_maps(native, skip_m1_route):
        layers = int(native["num_layers"])
        grid_x = int(native["grid_x"])
        grid_y = int(native["grid_y"])
        directions = np.asarray(native["layer_directions"], dtype=np.int64)
        offsets = np.asarray(native["layer_edge_offsets"], dtype=np.int64)
        if directions.size != layers or offsets.size != layers + 1:
            raise RuntimeError("CUGR layer metadata has invalid direction or edge-offset sizes")
        if np.any(offsets[1:] < offsets[:-1]):
            raise RuntimeError("CUGR layer edge offsets are not monotonic")
        capacity = torch.zeros((layers, grid_x, grid_y), dtype=torch.float32)
        fixed = torch.zeros_like(capacity)
        movable = torch.zeros_like(capacity)
        wire = torch.zeros_like(capacity)
        capacity_values = np.asarray(native["capacity"], dtype=np.float32)
        fixed_values = np.asarray(native["fixed_usage"], dtype=np.float32)
        if "movable_usage" not in native:
            raise RuntimeError("CUGR result is missing the movable_usage map plane")
        movable_values = np.asarray(native["movable_usage"], dtype=np.float32)
        wire_values = np.asarray(native["wire_usage"], dtype=np.float32)
        if not (
            capacity_values.size
            == fixed_values.size
            == movable_values.size
            == wire_values.size
        ):
            raise RuntimeError("CUGR primitive wire/fixed/movable/capacity map planes have different sizes")

        for layer in range(layers):
            direction = int(directions[layer])
            gridline_count = grid_x if direction == 0 else grid_y
            edge_count = grid_y - 1 if direction == 0 else grid_x - 1
            begin, end = int(offsets[layer]), int(offsets[layer + 1])
            expected = gridline_count * edge_count
            if end - begin != expected:
                raise RuntimeError(
                    f"CUGR layer {layer} primitive map has {end - begin} values; "
                    f"expected {expected} for grid {grid_x}x{grid_y}"
                )
            cursor = begin
            for gridline in range(gridline_count):
                for cp in range(edge_count):
                    x, y = (gridline, cp) if direction == 0 else (cp, gridline)
                    capacity[layer, x, y] = float(capacity_values[cursor])
                    fixed[layer, x, y] = float(fixed_values[cursor])
                    movable[layer, x, y] = float(movable_values[cursor])
                    wire[layer, x, y] = float(wire_values[cursor])
                    cursor += 1

        via_values = np.asarray(native["via_usage"], dtype=np.float32)
        expected_via = layers * grid_x * grid_y
        if via_values.size != expected_via:
            raise RuntimeError(
                f"CUGR via map has {via_values.size} values; expected {expected_via}"
            )
        via = torch.from_numpy(via_values.reshape(layers, grid_x, grid_y).copy())
        routing_begin = int(native.get("routing_layer_begin", 0))
        routing_end = int(native.get("routing_layer_end", layers - 1))
        if routing_begin < 0 or routing_end >= layers or routing_begin > routing_end:
            raise RuntimeError(
                f"CUGR returned invalid routing-layer window [{routing_begin}, {routing_end}]"
            )
        layer_ids = torch.arange(layers)
        enabled = (layer_ids >= routing_begin) & (layer_ids <= routing_end)
        if skip_m1_route and layers:
            enabled &= layer_ids != 0
        layer_mask = enabled.view(-1, 1, 1)
        capacity = torch.where(layer_mask, capacity, torch.zeros_like(capacity))
        fixed = torch.where(layer_mask, fixed, torch.zeros_like(fixed))
        movable = torch.where(layer_mask, movable, torch.zeros_like(movable))
        wire = torch.where(layer_mask, wire, torch.zeros_like(wire))
        via_enabled = enabled & (layer_ids < routing_end)
        via = torch.where(via_enabled.view(-1, 1, 1), via, torch.zeros_like(via))
        demand = wire + fixed + movable + via
        effective_capacity = (capacity - fixed - movable).clamp_min(torch.finfo(torch.float32).eps)
        h_ids = layer_ids[(directions == 0) & enabled.numpy()]
        v_ids = layer_ids[(directions != 0) & enabled.numpy()]

        cg_h = aggregate_layer_maps(demand, capacity, h_ids)
        cg_v = aggregate_layer_maps(demand, capacity, v_ids)
        cg_union = aggregate_layer_maps(demand, capacity, layer_ids[enabled.numpy()])
        effective_h = aggregate_layer_maps(wire + via, effective_capacity, h_ids)
        effective_v = aggregate_layer_maps(wire + via, effective_capacity, v_ids)
        maps = {
            "dmd_map": demand,
            "demand_map": demand,
            "raw_wire_demand_map": wire,
            "wire_demand_map": wire + fixed + movable,
            "via_demand_map": via,
            "fix_usage_map": fixed,
            "mov_usage_map": movable,
            "capacity_map": capacity,
            "cg_map_h_raw": cg_h,
            "cg_map_v_raw": cg_v,
            "cg_map_union_raw": cg_union,
            "cg_map_h_overflow": (cg_h - 1.0).clamp_min(0.0),
            "cg_map_v_overflow": (cg_v - 1.0).clamp_min(0.0),
            "cg_map_union_overflow": (cg_union - 1.0).clamp_min(0.0),
            "cg_map_h_effective_raw": effective_h,
            "cg_map_v_effective_raw": effective_v,
            "cg_map_h_effective_overflow": (effective_h - 1.0).clamp_min(0.0),
            "cg_map_v_effective_overflow": (effective_v - 1.0).clamp_min(0.0),
        }
        validate_map_contract(maps, expected_layers=layers)
        return maps, directions

    @staticmethod
    def _route_entries(native, net_name_to_id, ignored_net_names=None):
        entries = []
        seen_ids = set()
        ignored_names = {str(name) for name in (ignored_net_names or ())}
        for fallback_id, net in enumerate(native.get("nets", [])):
            name = str(net.get("name", ""))
            if net_name_to_id and name not in net_name_to_id:
                if name in ignored_names:
                    continue
                raise RuntimeError(f"CUGR returned unknown ECC net '{name}'")
            net_id = int(net_name_to_id.get(name, fallback_id))
            if net_id in seen_ids:
                raise RuntimeError(f"CUGR returned duplicate route net id {net_id} for '{name}'")
            seen_ids.add(net_id)
            net_entries = []
            for order, raw in enumerate(net.get("entries", [])):
                if len(raw) != 6:
                    raise RuntimeError(f"CUGR route for net '{name}' has a malformed entry")
                entry_type, layer, x1, y1, x2, y2 = [int(v) for v in raw]
                if entry_type not in (0, 1):
                    raise RuntimeError(f"CUGR route for net '{name}' has an invalid entry type")
                item = {
                    "order": order,
                    "type": "wire" if entry_type == 0 else "via",
                    "layer_idx": layer,
                    "grid_x1": x1,
                    "grid_y1": y1,
                    "grid_x2": x2,
                    "grid_y2": y2,
                    "net_id": net_id,
                    "net_name": name,
                }
                net_entries.append(item)
            entries.append(
                {
                    "net_id": net_id,
                    "rawdb_net_id": net_id,
                    "net_name": name,
                    "route_failed": bool(net.get("route_failed", not bool(net.get("routed", False)))),
                    "route_failed_reason": str(net.get("route_failed_reason", "")),
                    "entries": net_entries,
                }
            )
        return entries

    @staticmethod
    def _validate_route_layer_window(native, route_entries):
        begin = int(native.get("routing_layer_begin", 0))
        end = int(native.get("routing_layer_end", native.get("num_layers", 1) - 1))
        grid_x = int(native.get("grid_x", 0))
        grid_y = int(native.get("grid_y", 0))
        num_layers = int(native.get("num_layers", 0))
        if grid_x <= 0 or grid_y <= 0 or num_layers <= 0:
            raise RuntimeError("CUGR returned invalid route-grid metadata")
        for net in native.get("nets", []):
            for node in net.get("nodes", []):
                if len(node) < 5:
                    raise RuntimeError(f"CUGR topology node for net '{net.get('name', '')}' is malformed")
                layer = int(node[0])
                x, y = int(node[1]), int(node[2])
                if layer < begin or layer > end or layer >= num_layers:
                    raise RuntimeError(
                        f"CUGR topology node for net '{net.get('name', '')}' is outside "
                        f"routing layer window [{begin}, {end}]"
                    )
                if x < 0 or x >= grid_x or y < 0 or y >= grid_y:
                    raise RuntimeError(
                        f"CUGR topology node for net '{net.get('name', '')}' is outside the route grid"
                    )
        for net in route_entries:
            for entry in net.get("entries", []):
                layer = int(entry["layer_idx"])
                x1, y1 = int(entry["grid_x1"]), int(entry["grid_y1"])
                x2, y2 = int(entry["grid_x2"]), int(entry["grid_y2"])
                if layer < begin or layer > end or layer >= num_layers:
                    raise RuntimeError(
                        f"CUGR route for net '{net.get('net_name', '')}' is outside "
                        f"routing layer window [{begin}, {end}]"
                    )
                if min(x1, x2) < 0 or max(x1, x2) >= grid_x or min(y1, y2) < 0 or max(y1, y2) >= grid_y:
                    raise RuntimeError(
                        f"CUGR route for net '{net.get('net_name', '')}' is outside the route grid"
                    )
                if entry["type"] == "via" and layer >= end:
                    raise RuntimeError(
                        f"CUGR via for net '{net.get('net_name', '')}' crosses above "
                        f"routing layer {end}"
                    )
                if entry["type"] == "wire" and x1 != x2 and y1 != y2:
                    raise RuntimeError(
                        f"CUGR route for net '{net.get('net_name', '')}' contains a diagonal wire"
                    )
                if entry["type"] == "via" and (x1 != x2 or y1 != y2):
                    raise RuntimeError(
                        f"CUGR route for net '{net.get('net_name', '')}' contains a non-vertical via"
                    )

    @staticmethod
    def _l_shape_pack(
        native,
        flat_net2pin,
        flat_net2pin_start,
        num_pins,
        num_nets,
        net_name_to_id,
        ignored_net_names=None,
    ):
        if num_pins <= 0 or num_nets <= 0:
            raise RuntimeError("CUGR L-shape packing requires positive pin and net counts")
        if not net_name_to_id:
            raise RuntimeError("CUGR L-shape packing requires the ECC net-name mapping")
        flat_net2pin = _as_int32(flat_net2pin)
        flat_net2pin_start = _as_int32(flat_net2pin_start)
        if (
            flat_net2pin_start.size != num_nets + 1
            or flat_net2pin_start[0] != 0
            or flat_net2pin_start[-1] != flat_net2pin.size
            or np.any(flat_net2pin_start[1:] < flat_net2pin_start[:-1])
        ):
            raise RuntimeError("CUGR L-shape net-to-pin inputs have invalid offsets")

        name_to_id = {str(name): int(net_id) for name, net_id in net_name_to_id.items()}
        if len(name_to_id) != len(net_name_to_id) or len(set(name_to_id.values())) != len(name_to_id):
            raise RuntimeError("CUGR L-shape net-name mapping is not bijective")
        id_to_name = {net_id: name for name, net_id in name_to_id.items()}
        if set(id_to_name) != set(range(num_nets)):
            raise RuntimeError("CUGR L-shape net-name mapping does not cover all ECC nets")
        ignored_names = {str(name) for name in (ignored_net_names or ())}
        native_by_name = {}
        for native_net in native.get("nets", []):
            name = str(native_net.get("name", ""))
            if name in native_by_name:
                raise RuntimeError(f"CUGR returned duplicate net '{name}'")
            if name not in name_to_id:
                if name in ignored_names:
                    continue
                raise RuntimeError(f"CUGR returned unknown ECC net '{name}'")
            native_by_name[name] = native_net

        missing_native = [name for name in name_to_id if name not in native_by_name]
        if missing_native:
            raise RuntimeError(
                "CUGR topology is missing ECC nets: " + ", ".join(sorted(missing_native))
            )

        # CUGR's native tree is grouped by net. DreamPlace reserves the first
        # num_pins vertex IDs for original pins, then appends one contiguous
        # Steiner range per net. Keep every array in that global ID space.
        cursor = num_pins
        pin_relate_x = list(range(num_pins))
        pin_relate_y = list(range(num_pins))
        pin_fa = [-1] * num_pins
        net_steiner_start = [num_pins]
        remapped_edges = []
        net_orders = []

        for net_id in range(num_nets):
            begin = int(flat_net2pin_start[net_id])
            end = int(flat_net2pin_start[net_id + 1])
            pins = [int(v) for v in flat_net2pin[begin:end]]
            if any(pin < 0 or pin >= num_pins for pin in pins):
                raise RuntimeError(f"CUGR L-shape net {net_id} contains an out-of-range pin")
            if len(set(pins)) != len(pins):
                raise RuntimeError(f"CUGR L-shape net {net_id} contains duplicate pin IDs")
            native_net = native_by_name.get(id_to_name[net_id], {})
            nodes = native_net.get("nodes", []) or []
            pin_access = native_net.get("pin_access", []) or []
            native_global = {}
            native_order = []
            first_native_node = {}
            for node_idx, node in enumerate(nodes):
                if len(node) < 5:
                    raise RuntimeError(f"CUGR net {id_to_name[net_id]} has a malformed topology node")
                pin_index = int(node[3])
                if pin_index < -1:
                    raise RuntimeError(
                        f"CUGR net {id_to_name[net_id]} has an invalid pin index {pin_index}"
                    )
                if pin_index >= len(pins):
                    raise RuntimeError(
                        f"CUGR net {id_to_name[net_id]} refers to pin index {pin_index}"
                    )
                parent_index = int(node[4])
                if parent_index < -1:
                    raise RuntimeError(
                        f"CUGR net {id_to_name[net_id]} has an invalid parent index {parent_index}"
                    )
                coordinate_pin = None
                if pin_access:
                    coordinate_matches = [
                        local_pin
                        for local_pin, access in enumerate(pin_access[: len(pins)])
                        if len(access) >= 3
                        and int(access[1]) == int(node[1])
                        and int(access[2]) == int(node[2])
                    ]
                    if coordinate_matches:
                        if pin_index in coordinate_matches:
                            coordinate_pin = pin_index
                        else:
                            coordinate_pin = coordinate_matches[0]
                # Pin indices in CUGR are tied to an access candidate and can
                # repeat across layer trees. Exact XY pin access coordinates
                # are the stable ECC identity; only roots or explicitly marked
                # pin nodes may claim a coordinate as an original pin.
                if coordinate_pin is not None and (pin_index >= 0 or parent_index < 0):
                    vertex = pins[coordinate_pin]
                elif pin_index >= 0:
                    vertex = pins[pin_index]
                else:
                    vertex = cursor
                    cursor += 1
                    # Relate each Steiner coordinate independently to the
                    # nearest native pin access coordinate. This preserves the
                    # differentiable pin-to-Steiner contract after placement
                    # updates without inventing a new coordinate source.
                    candidates = []
                    for local_pin, access in enumerate(pin_access[: len(pins)]):
                        if len(access) < 3:
                            continue
                        candidates.append((local_pin, int(access[1]), int(access[2])))
                    if candidates:
                        rel_x = min(candidates, key=lambda item: (abs(item[1] - int(node[1])), item[0]))[0]
                        rel_y = min(candidates, key=lambda item: (abs(item[2] - int(node[2])), item[0]))[0]
                        pin_relate_x.append(pins[rel_x])
                        pin_relate_y.append(pins[rel_y])
                    else:
                        fallback = pins[0] if pins else 0
                        pin_relate_x.append(fallback)
                        pin_relate_y.append(fallback)
                    pin_fa.append(-1)
                native_global[node_idx] = vertex
                native_order.append(vertex)
                first_native_node.setdefault(vertex, node_idx)

            for child_idx, node in enumerate(nodes):
                parent_idx = int(node[4])
                if parent_idx < 0:
                    continue
                if parent_idx not in native_global:
                    raise RuntimeError(f"CUGR net {id_to_name[net_id]} has an invalid topology parent")
                parent = native_global[parent_idx]
                child = native_global[child_idx]
                if parent == child:
                    continue
                # A multi-layer CUGR tree may revisit an original pin in a
                # second grid tree. The repeated node is not a second parent
                # in the DreamPlace topology; retain the first occurrence.
                if first_native_node.get(child) != child_idx:
                    continue
                dx = int(node[1]) - int(nodes[parent_idx][1])
                dy = int(node[2]) - int(nodes[parent_idx][2])
                if dx and dy:
                    raise RuntimeError(
                        f"CUGR net {id_to_name[net_id]} returned a diagonal topology edge"
                    )
                direction = 2
                remapped_edges.append((parent, child, direction))
                if pin_fa[child] not in (-1, parent):
                    raise RuntimeError(f"CUGR net {id_to_name[net_id]} returned multiple parents")
                pin_fa[child] = parent

            # Native collect_tree is pre-order, so parents precede children.
            # Remove duplicate pin nodes and append pins absent from an
            # unrouted/partially routed native tree exactly once.
            order = []
            seen = set()
            for vertex in native_order + pins:
                if vertex not in seen:
                    seen.add(vertex)
                    order.append(vertex)
            net_orders.append(order)
            net_steiner_start.append(cursor)

        num_vertices = cursor
        if len(pin_relate_x) != num_vertices or len(pin_fa) != num_vertices:
            raise RuntimeError("CUGR L-shape vertex arrays do not match num_vertices")
        # DreamPlace uses this array to partition the original pin IDs by net;
        # it is the same flat-net offset supplied by placedb, not a range over
        # the newly appended Steiner vertices.
        net_vertex_start = flat_net2pin_start.tolist()
        adjacency = [[] for _ in range(num_vertices)]
        for parent, child, direction in remapped_edges:
            adjacency[parent].append((parent, child, direction))
        flat_from, flat_to, directions, to_start = [], [], [], [0]
        for vertex_edges in adjacency:
            for parent, child, direction in vertex_edges:
                flat_from.append(parent)
                flat_to.append(child)
                directions.append(direction)
            to_start.append(len(flat_to))
        net_flat_topo_sort = [v for order in net_orders for v in order]
        net_flat_topo_sort_start = [0]
        for order in net_orders:
            net_flat_topo_sort_start.append(net_flat_topo_sort_start[-1] + len(order))
        metadata = {
            "schema_version": 1,
            "num_pins": num_pins,
            "num_vertices": num_vertices,
            "num_edges": len(flat_from),
            "num_nets": num_nets,
            "route_failed_count": sum(
                bool(net.get("route_failed", not bool(net.get("routed", False))))
                for net in native.get("nets", [])
            ),
        }
        return {
            "pin_relate_x": _as_int32(pin_relate_x),
            "pin_relate_y": _as_int32(pin_relate_y),
            "net_vertex_start": _as_int32(net_vertex_start),
            "net_steiner_start": _as_int32(net_steiner_start),
            "pin_fa": _as_int32(pin_fa),
            "flat_pin_from": _as_int32(flat_from),
            "flat_pin_to": _as_int32(flat_to),
            "flat_pin_to_start": _as_int32(to_start),
            "net_flat_topo_sort": _as_int32(net_flat_topo_sort),
            "net_flat_topo_sort_start": _as_int32(net_flat_topo_sort_start),
            "edge_l_directions": _as_int32(directions),
            "metadata": metadata,
        }

    def run_gpugr(
        self,
        input_def: str = "",
        out_dir: str = "",
        design_name: str = "",
        benchmark: str = "",
        gpu: int = 0,
        threads: int = 1,
        route_xsize: int = 0,
        route_ysize: int = 0,
        rrr_iters: int = 0,
        guide_path: str = "",
        skip_m1_route: bool = True,
        verbose_parser_log: bool = False,
        cpp_log_level: int = 2,
        export_current_db: bool = False,
        keep_temp_def: bool = False,
        save_artifacts: bool = False,
        include_route_entries: bool = False,
        include_topology_pack: bool = False,
        include_l_shape_topology_pack: bool = False,
        topology_net_name_to_id: dict = None,
        topology_pin_name_to_id: dict = None,
        topology_flat_net2pin_map=None,
        topology_flat_net2pin_start_map=None,
        topology_num_pins: int = 0,
        topology_num_nets: int = 0,
        topology_ignored_net_names=None,
        topology_max_gap: int = 1,
        topology_xl: float = 0.0,
        topology_yl: float = 0.0,
        topology_bin_size_x: float = 1.0,
        topology_bin_size_y: float = 1.0,
        parser_cache_enable: bool = False,
        parser_cache_node_lpos=None,
        parser_cache_node_names=None,
        parser_cache_fallback_before_export=None,
        profile_enabled: bool = False,
        profile_prefix: str = "gpugr.run",
        backend: str = "cugr",
        bottom_routing_layer: str = None,
        top_routing_layer: str = None,
        session_cache_enable: bool = True,
        design_identity: str = "",
        require_clean_source: bool = False,
    ):
        requested_backend = normalize_gpugr_backend(backend)
        if requested_backend != "cugr":
            raise RuntimeError("CugrGPUGR received a non-cugr backend request")
        if guide_path:
            raise RuntimeError("CUGR backend does not emit route guides; omit guide_path")
        if route_xsize <= 0 or route_ysize <= 0:
            raise ValueError(
                "CUGR requires positive route_xsize and route_ysize; "
                f"got {route_xsize}x{route_ysize}"
            )
        if threads <= 0:
            raise ValueError(f"CUGR requires a positive thread count; got {threads}")
        if rrr_iters < 0:
            raise ValueError(f"CUGR rrr_iters must be non-negative; got {rrr_iters}")
        if gpu not in (0, None):
            logger.info("Ignoring gpu=%s for the CPU CUGR backend", gpu)
        if parser_cache_enable:
            logger.debug(
                "CUGR uses its native coordinate-refresh session cache; "
                "DreamPlace parser-cache inputs are compatibility-only"
            )
        del (
            benchmark,
            verbose_parser_log,
            cpp_log_level,
            export_current_db,
            topology_pin_name_to_id,
            topology_xl,
            topology_yl,
            topology_bin_size_x,
            topology_bin_size_y,
            parser_cache_node_lpos,
            parser_cache_node_names,
            parser_cache_fallback_before_export,
            profile_enabled,
            profile_prefix,
        )
        bottom_value = (
            getattr(self.params, "gpugr_bottom_routing_layer", "")
            if bottom_routing_layer is None
            else bottom_routing_layer
        )
        top_value = (
            getattr(self.params, "gpugr_top_routing_layer", "")
            if top_routing_layer is None
            else top_routing_layer
        )
        bottom = str(bottom_value or "")
        top = str(top_value or "")
        cugr = self._import_cugr()
        source_dirty = bool(getattr(cugr, "source_dirty", lambda: True)())
        require_clean_source = bool(
            require_clean_source
            or getattr(self.params, "cugr_require_clean_source", False)
        )
        if require_clean_source and source_dirty:
            raise RuntimeError(
                "CUGR source metadata is dirty; a clean published source is required "
                "for qualification"
            )
        session_cache_enable = bool(
            session_cache_enable
            and getattr(self.params, "cugr_session_cache_enable", True)
        )
        if not session_cache_enable:
            try:
                cugr.reset()
            except Exception as exc:
                raise RuntimeError("Failed to reset the process-global CUGR database") from exc
        lefs = self._resolve_lefs()
        native_design_identity = str(design_identity or self._design_identity())
        result_dir = self._resolve_output_dir(out_dir)
        temp_dir = Path(tempfile.mkdtemp(prefix="cugr_", dir=str(result_dir)))
        native_output_capture = temp_dir / "cugr_native.log"
        evidence_path = None
        try:
            if input_def:
                materialized_def, _ = self._materialize_def(input_def, temp_dir)
            else:
                materialized_def = self._export_current_def(temp_dir, design_name or self.params.design_name())
            powv, post = self._resolve_flute_lut_paths()
            with self._capture_native_output(native_output_capture):
                native = cugr.run({
                    "lefs": lefs,
                    "def": str(materialized_def),
                    "powv_file": powv,
                    "post_file": post,
                    "design_identity": native_design_identity,
                    "threads": int(threads),
                    "rrr_iters": int(rrr_iters),
                    "route_num_bins_x": int(route_xsize),
                    "route_num_bins_y": int(route_ysize),
                    "bottom_routing_layer": bottom,
                    "top_routing_layer": top,
                })
            self._append_native_output_to_place_logs(native_output_capture)
            maps, directions = self._primitive_maps(native, bool(skip_m1_route))
            net_name_to_id = topology_net_name_to_id or {}
            route_entries = self._route_entries(
                native,
                net_name_to_id,
                ignored_net_names=topology_ignored_net_names,
            )
            self._validate_route_layer_window(native, route_entries)
            if not design_name:
                design_name = self._infer_design_name(str(materialized_def))
            metrics = {
                "num_overflow_nets": int(native["overflow_net_count"]),
                "gr_wirelength": float(native["wirelength_dbu"]),
                "gr_num_vias": int(native["via_count"]),
                "gr_est_shorts": float(native.get("short_vio_area", 0.0)),
                "gpugr_backend_requested": "cugr",
                "gpugr_backend": "cugr",
                "gpugr_backend_resolved": "cugr",
                "gpugr_rrr_iters": int(rrr_iters),
                "cugr_total_passes": int(native["total_passes"]),
                "cugr_threads": int(native["threads"]),
                "cugr_pass0_worker_count": int(native["pass0_worker_count"]),
                "cugr_route_grid_x": int(native["grid_x"]),
                "cugr_route_grid_y": int(native["grid_y"]),
                "cugr_design_identity": native_design_identity,
                "cugr_source_commit": str(cugr.source_commit()),
                "cugr_source_dirty": int(source_dirty),
                "cugr_cuda_enabled": 0,
                "cugr_require_clean_source": int(require_clean_source),
                "cugr_routing_layer_begin": int(native["routing_layer_begin"]),
                "cugr_routing_layer_end": int(native["routing_layer_end"]),
                "cugr_routing_layer_names": [str(name) for name in native["layer_names"]],
                "cugr_session_cache_hit": int(bool(native.get("session_cache_hit", False))),
                "cugr_session_mode": str(native.get("session_mode", "full_reparse_reset")),
                "cugr_session_cache_enabled": int(session_cache_enable),
                "elapsed_sec": float(native.get("total_sec", 0.0)),
                "cugr_parse_sec": float(native.get("parse_sec", 0.0)),
                "cugr_refresh_sec": float(native.get("refresh_sec", 0.0)),
                "cugr_route_sec": float(native.get("route_sec", 0.0)),
                "cugr_pack_sec": float(native.get("pack_sec", 0.0)),
                "cugr_peak_rss_mb": float(native.get("peak_rss_mb", 0.0)),
                "cugr_route_failed_count": sum(
                    bool(net.get("route_failed", not bool(net.get("routed", False))))
                    for net in native.get("nets", [])
                ),
                "cugr_num_nets": len(native.get("nets", [])),
                "cugr_fixed_usage_sha256": _tensor_sha256(maps["fix_usage_map"]),
                "cugr_movable_usage_sha256": _tensor_sha256(maps["mov_usage_map"]),
                "cugr_invalid_layer_warning_count": int(
                    native.get("invalid_layer_warning_count", 0)
                ),
                "gpugr_bottom_routing_layer": bottom,
                "gpugr_top_routing_layer": top,
            }
            for name, tensor in (
                ("cg_map_h_raw", maps["cg_map_h_raw"]),
                ("cg_map_v_raw", maps["cg_map_v_raw"]),
                ("cg_map_union_raw", maps["cg_map_union_raw"]),
            ):
                summary = summarize_congestion_tensor(tensor, 1.0)
                prefix = name
                metrics[f"{prefix}_max"] = summary["max"]
                metrics[f"{prefix}_mean"] = summary["mean"]
                metrics[f"{prefix}_top1pct_mean"] = summary["top1pct_mean"]
                metrics[f"{prefix}_overflow_bin_ratio"] = summary["overflow_bin_ratio"]
            result = {
                "metrics": metrics,
                "maps": maps,
                "route_entries": route_entries,
                "artifact_paths": {},
                "same_net_topology_cache": {},
                "same_net_topology_stats": {},
                "l_shape_topology_pack": {},
            }
            if include_topology_pack:
                from dreamplace.ops.routability.same_net_topo_scoring import (
                    build_same_net_topology_cache,
                )

                same_net_cache, same_net_stats = build_same_net_topology_cache(
                    route_entries,
                    self.placedb,
                    max_gap=int(topology_max_gap),
                )
                result["same_net_topology_cache"] = same_net_cache
                result["same_net_topology_stats"] = same_net_stats
            if include_l_shape_topology_pack:
                result["l_shape_topology_pack"] = self._l_shape_pack(
                    native,
                    topology_flat_net2pin_map,
                    topology_flat_net2pin_start_map,
                    int(topology_num_pins),
                    int(topology_num_nets),
                    net_name_to_id,
                    ignored_net_names=topology_ignored_net_names,
                )
            if save_artifacts:
                result_dir.mkdir(parents=True, exist_ok=True)
                maps_path = (result_dir / f"{design_name}_gpugr_map.npz").resolve()
                metrics_path = (result_dir / f"{design_name}_gpugr_metrics.json").resolve()
                self._save_maps(maps_path, maps)
                self._save_metrics(metrics_path, metrics)
                artifact_paths = {
                    "maps_path": str(maps_path),
                    "metrics_path": str(metrics_path),
                }
                native_log_path = self._persist_native_log(
                    native_output_capture,
                    result_dir,
                    design_name,
                    temp_dir.name,
                )
                if native_log_path:
                    artifact_paths["native_log_path"] = native_log_path
                if include_route_entries:
                    route_entries_path = (
                        result_dir / f"{design_name}_gpugr_routes.json"
                    ).resolve()
                    self._save_route_entries(route_entries_path, route_entries)
                    artifact_paths["route_entries_path"] = str(route_entries_path)
                if include_l_shape_topology_pack:
                    topology = result["l_shape_topology_pack"]
                    topology_arrays = {
                        key: value
                        for key, value in topology.items()
                        if key != "metadata" and isinstance(value, np.ndarray)
                    }
                    topology_path = (
                        result_dir / f"{design_name}_gpugr_l_shape_topology.npz"
                    ).resolve()
                    np.savez_compressed(topology_path, **topology_arrays)
                    topology_metadata_path = (
                        result_dir / f"{design_name}_gpugr_l_shape_topology.json"
                    ).resolve()
                    self._save_metrics(topology_metadata_path, topology["metadata"])
                    artifact_paths["l_shape_topology_path"] = str(topology_path)
                    artifact_paths["l_shape_topology_metadata_path"] = str(
                        topology_metadata_path
                    )
                result["artifact_paths"] = artifact_paths
                evidence_input_def = str(input_def or "")
                if not evidence_input_def and keep_temp_def:
                    evidence_input_def = str(materialized_def)
                evidence_path = write_cugr_evidence(
                    result_dir=result_dir,
                    run_id=temp_dir.name,
                    input_def=evidence_input_def,
                    lefs=lefs,
                    artifact_paths=artifact_paths,
                    metrics=metrics,
                    requested_backend="cugr",
                    resolved_backend="cugr",
                    source_commit=str(cugr.source_commit()),
                    source_dirty=source_dirty,
                    effective_params={
                        "threads": int(threads),
                        "rrr_iters": int(rrr_iters),
                        "cugr_total_passes": int(native["total_passes"]),
                        "route_num_bins_x": int(route_xsize),
                        "route_num_bins_y": int(route_ysize),
                        "bottom_routing_layer": bottom,
                        "top_routing_layer": top,
                        "skip_m1_route": bool(skip_m1_route),
                        "session_cache_enable": bool(session_cache_enable),
                        "require_clean_source": bool(require_clean_source),
                    },
                    returncode=0,
                    terminal_marker=True,
                )
                result["artifact_paths"]["evidence_path"] = str(evidence_path)
            elif keep_temp_def:
                result["artifact_paths"] = {"temp_def_path": str(materialized_def)}
            return result
        except Exception as exc:
            if save_artifacts:
                try:
                    self._append_native_output_to_place_logs(native_output_capture)
                    native_log_path = self._persist_native_log(
                        native_output_capture,
                        result_dir,
                        design_name or "cugr",
                        temp_dir.name,
                    )
                    write_cugr_evidence(
                        result_dir=result_dir,
                        run_id=temp_dir.name,
                        status="fail",
                        input_def=str(input_def or ""),
                        lefs=lefs if "lefs" in locals() else (),
                        artifact_paths={
                            "native_log_path": native_log_path,
                            **(
                                {"temp_def_path": str(materialized_def)}
                                if keep_temp_def and "materialized_def" in locals()
                                else {}
                            ),
                        },
                        requested_backend="cugr",
                        resolved_backend="cugr",
                        source_commit=str(cugr.source_commit()) if "cugr" in locals() else "",
                        source_dirty=source_dirty if "source_dirty" in locals() else None,
                        effective_params={
                            "threads": int(threads),
                            "rrr_iters": int(rrr_iters),
                            "route_num_bins_x": int(route_xsize),
                            "route_num_bins_y": int(route_ysize),
                            "bottom_routing_layer": bottom,
                            "top_routing_layer": top,
                        },
                        returncode=1,
                        terminal_marker=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    logger.exception("Failed to write CUGR failure evidence")
            raise
        finally:
            if not session_cache_enable:
                try:
                    cugr.reset()
                except Exception:
                    logger.exception("Failed to reset CUGR state after run")
            if not keep_temp_def:
                shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["CugrGPUGR"]
