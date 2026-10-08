from collections import Counter
import re
from pathlib import Path


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _as_int_list(value):
    return [int(item) for item in _as_list(value)]


def _get_index(values, index, default=None):
    return values[index] if 0 <= index < len(values) else default


def _pin_coordinate(pydb, pin_id):
    pin2node = _as_int_list(getattr(pydb, "pin2node_map", []))
    node_x = _as_list(getattr(pydb, "node_x", []))
    node_y = _as_list(getattr(pydb, "node_y", []))
    pin_offset_x = _as_list(getattr(pydb, "pin_offset_x", []))
    pin_offset_y = _as_list(getattr(pydb, "pin_offset_y", []))

    pin_id = int(pin_id)
    node_id = _get_index(pin2node, pin_id, -1)
    if node_id is None or node_id < 0:
        return None
    if node_id >= len(node_x) or node_id >= len(node_y):
        return None
    x = float(node_x[node_id]) + float(_get_index(pin_offset_x, pin_id, 0.0) or 0.0)
    y = float(node_y[node_id]) + float(_get_index(pin_offset_y, pin_id, 0.0) or 0.0)
    return int(round(x)), int(round(y))


def _iter_requested_net_ids(flat_net2pin_start, net_ids, max_nets):
    num_nets = max(0, len(flat_net2pin_start) - 1)
    requested = [int(net_id) for net_id in net_ids] if net_ids is not None else range(num_nets)
    count = 0
    for net_id in requested:
        if max_nets is not None and count >= int(max_nets):
            break
        count += 1
        yield net_id


def _collect_pydb_net_pin_groups(pydb, *, net_ids=None, max_nets=None, max_net_degree=None):
    flat_net2pin = _as_int_list(getattr(pydb, "flat_net2pin_map", []))
    flat_net2pin_start = _as_int_list(getattr(pydb, "flat_net2pin_start_map", []))
    net2driver = _as_int_list(getattr(pydb, "net2driver_pin_map", []))
    groups = []
    skipped = Counter()
    for net_id in _iter_requested_net_ids(flat_net2pin_start, net_ids, max_nets):
        if net_id < 0 or net_id + 1 >= len(flat_net2pin_start):
            skipped["invalid_net_id"] += 1
            continue
        begin = int(flat_net2pin_start[net_id])
        end = int(flat_net2pin_start[net_id + 1])
        if begin < 0 or end <= begin or end > len(flat_net2pin):
            skipped["invalid_net_pin_range"] += 1
            continue
        net_pin_ids = [int(pin_id) for pin_id in flat_net2pin[begin:end]]
        if len(net_pin_ids) <= 1:
            skipped["single_pin_net"] += 1
            continue
        if max_net_degree is not None and len(net_pin_ids) > int(max_net_degree):
            skipped["degree_over_limit"] += 1
            continue
        driver_pin_id = _get_index(net2driver, net_id, -1)
        if driver_pin_id is None or int(driver_pin_id) < 0 or int(driver_pin_id) not in set(net_pin_ids):
            skipped["invalid_driver_pin"] += 1
            continue
        groups.append((int(net_id), net_pin_ids, int(driver_pin_id)))
    return groups, skipped


def build_local_steiner_topology_from_pydb(
    pydb,
    *,
    net_ids=None,
    max_nets=None,
    max_net_degree=None,
):
    try:
        import torch
        import dreamplace.ops.steiner_topo.steiner_topo_cpp as steiner_topo_cpp
        from dreamplace.ops.steiner_topo.steiner_topo import _FLUTE_POST_FILE, _FLUTE_POWV_FILE
    except ImportError as exc:
        raise RuntimeError("steiner_topo_cpp is required for topology_source=steiner") from exc

    groups, skipped = _collect_pydb_net_pin_groups(
        pydb,
        net_ids=net_ids,
        max_nets=max_nets,
        max_net_degree=max_net_degree,
    )
    if not groups:
        return {
            "topology_source": "steiner",
            "status": "unsupported",
            "unsupported_reasons": ["no_supported_nets_for_steiner"],
            "net_ids": [],
            "skipped_reasons": dict(sorted(skipped.items())),
            "net_flat_topo_sort": [],
            "net_flat_topo_sort_start": [0],
            "pin_fa": [],
            "node_x": {},
            "node_y": {},
        }

    pin2node = _as_int_list(getattr(pydb, "pin2node_map", []))
    node_x_values = _as_list(getattr(pydb, "node_x", []))
    node_y_values = _as_list(getattr(pydb, "node_y", []))
    pin_offset_x_values = _as_list(getattr(pydb, "pin_offset_x", []))
    pin_offset_y_values = _as_list(getattr(pydb, "pin_offset_y", []))
    coordinate_cache = {}

    def pin_coordinate(pin_id):
        pin_id = int(pin_id)
        cached = coordinate_cache.get(pin_id)
        if cached is not None:
            return cached
        node_id = _get_index(pin2node, pin_id, -1)
        if node_id is None or node_id < 0:
            return None
        if node_id >= len(node_x_values) or node_id >= len(node_y_values):
            return None
        x = float(node_x_values[node_id]) + float(
            _get_index(pin_offset_x_values, pin_id, 0.0) or 0.0
        )
        y = float(node_y_values[node_id]) + float(
            _get_index(pin_offset_y_values, pin_id, 0.0) or 0.0
        )
        coord = (int(round(x)), int(round(y)))
        coordinate_cache[pin_id] = coord
        return coord

    global_to_local_pin = {}
    local_to_global_pin = []
    flat_netpin = []
    netpin_start = [0]
    for _net_id, net_pin_ids, _driver_pin_id in groups:
        for pin_id in net_pin_ids:
            if pin_id not in global_to_local_pin:
                coord = pin_coordinate(pin_id)
                if coord is None:
                    raise ValueError(f"missing coordinate for pin_id={pin_id}")
                global_to_local_pin[pin_id] = len(local_to_global_pin)
                local_to_global_pin.append(pin_id)
            flat_netpin.append(global_to_local_pin[pin_id])
        netpin_start.append(len(flat_netpin))

    local_coords = [pin_coordinate(pin_id) for pin_id in local_to_global_pin]
    xs = [float(coord[0]) for coord in local_coords]
    ys = [float(coord[1]) for coord in local_coords]
    pos = torch.tensor(xs + ys, dtype=torch.float32)
    flat_netpin_tensor = torch.tensor(flat_netpin, dtype=torch.int32)
    netpin_start_tensor = torch.tensor(netpin_start, dtype=torch.int32)
    build_tree_max_degree = (
        int(max_net_degree)
        if max_net_degree is not None
        else max(len(net_pin_ids) for _net_id, net_pin_ids, _driver in groups)
    )
    outputs = steiner_topo_cpp.build_tree(
        pos,
        flat_netpin_tensor,
        netpin_start_tensor,
        int(build_tree_max_degree),
        str(_FLUTE_POWV_FILE),
        str(_FLUTE_POST_FILE),
    )
    (
        newx,
        newy,
        _relate_x,
        _relate_y,
        _vertex_start,
        _steiner_start,
        pin_fa,
        _flat_to,
        _flat_from,
        _flat_to_start,
        topo,
        topo_start,
    ) = outputs

    num_local_pins = len(local_to_global_pin)
    original_pin_count = len(pin2node)
    max_pin_id = max(local_to_global_pin) if local_to_global_pin else -1
    steiner_base = max(original_pin_count, max_pin_id + 1)

    def local_node_to_topology_node(local_node_id):
        local_node_id = int(local_node_id)
        if local_node_id < num_local_pins:
            return int(local_to_global_pin[local_node_id])
        return int(steiner_base + local_node_id - num_local_pins)

    topo_local = _as_int_list(topo)
    topo_start_values = _as_int_list(topo_start)
    pin_fa_local = _as_int_list(pin_fa)
    num_vertices = len(_as_list(newx))

    node_x = {}
    node_y = {}
    xs_out = _as_list(newx)
    ys_out = _as_list(newy)
    for local_node_id in range(num_vertices):
        topology_node_id = local_node_to_topology_node(local_node_id)
        node_x[topology_node_id] = int(round(float(xs_out[local_node_id])))
        node_y[topology_node_id] = int(round(float(ys_out[local_node_id])))

    mapped_topo = [local_node_to_topology_node(node_id) for node_id in topo_local]
    mapped_pin_fa = [-1] * (steiner_base + max(0, num_vertices - num_local_pins))
    for local_node_id, parent in enumerate(pin_fa_local):
        topology_node_id = local_node_to_topology_node(local_node_id)
        if topology_node_id >= len(mapped_pin_fa):
            mapped_pin_fa.extend([-1] * (topology_node_id - len(mapped_pin_fa) + 1))
        mapped_pin_fa[topology_node_id] = (
            -1 if int(parent) < 0 else local_node_to_topology_node(parent)
        )

    return {
        "topology_source": "steiner",
        "status": "ok",
        "net_ids": [net_id for net_id, _pins, _driver in groups],
        "skipped_reasons": dict(sorted(skipped.items())),
        "max_nets": None if max_nets is None else int(max_nets),
        "max_net_degree": None if max_net_degree is None else int(max_net_degree),
        "build_tree_max_degree": int(build_tree_max_degree),
        "net_flat_topo_sort": mapped_topo,
        "net_flat_topo_sort_start": topo_start_values,
        "pin_fa": mapped_pin_fa,
        "node_x": node_x,
        "node_y": node_y,
        "local_pin_count": num_local_pins,
        "steiner_node_base": steiner_base,
        "steiner_node_count": max(0, num_vertices - num_local_pins),
    }


def topology_slice_index(net_id, topology):
    topo_start = topology.get("_net_flat_topo_sort_start_int")
    if topo_start is None:
        topo_start = _as_int_list(topology.get("net_flat_topo_sort_start", []))
        topology["_net_flat_topo_sort_start_int"] = topo_start
    net_ids = topology.get("_net_ids_int")
    if net_ids is None:
        net_ids = _as_int_list(topology.get("net_ids", []))
        topology["_net_ids_int"] = net_ids
    if net_ids:
        slice_by_net_id = topology.get("_slice_by_net_id")
        if slice_by_net_id is None:
            slice_by_net_id = {
                int(item): int(index)
                for index, item in enumerate(net_ids)
            }
            topology["_slice_by_net_id"] = slice_by_net_id
        slice_index = slice_by_net_id.get(int(net_id))
        if slice_index is None:
            return None
        return slice_index if slice_index + 1 < len(topo_start) else None
    return int(net_id) if 0 <= int(net_id) + 1 < len(topo_start) else None


def build_topology_edges(net_id, topology):
    topo = topology.get("_net_flat_topo_sort_int")
    if topo is None:
        topo = _as_int_list(topology.get("net_flat_topo_sort", []))
        topology["_net_flat_topo_sort_int"] = topo
    topo_start = topology.get("_net_flat_topo_sort_start_int")
    if topo_start is None:
        topo_start = _as_int_list(topology.get("net_flat_topo_sort_start", []))
        topology["_net_flat_topo_sort_start_int"] = topo_start
    pin_fa = topology.get("_pin_fa_int")
    if pin_fa is None:
        pin_fa = _as_int_list(topology.get("pin_fa", []))
        topology["_pin_fa_int"] = pin_fa
    slice_index = topology_slice_index(net_id, topology)
    if slice_index is None:
        return None
    begin = int(topo_start[slice_index])
    end = int(topo_start[slice_index + 1])
    if begin < 0 or end < begin or end > len(topo):
        return None
    node_ids = [int(node) for node in topo[begin:end]]
    edges = []
    for node_id in node_ids:
        parent = _get_index(pin_fa, node_id, -1)
        if parent is not None and int(parent) >= 0:
            edges.append((int(parent), int(node_id)))
    return node_ids, edges


def topology_coordinates(pydb, node_ids, topology):
    node_x = topology.get("node_x", {})
    node_y = topology.get("node_y", {})
    coordinates = {}
    num_pins = len(_as_list(getattr(pydb, "pin2node_map", [])))
    for node_id in node_ids:
        node_id = int(node_id)
        if node_id < num_pins:
            coord = _pin_coordinate(pydb, node_id)
        else:
            if isinstance(node_x, dict):
                x = node_x.get(node_id)
                y = node_y.get(node_id) if isinstance(node_y, dict) else None
            else:
                xs = _as_list(node_x)
                ys = _as_list(node_y)
                x = _get_index(xs, node_id)
                y = _get_index(ys, node_id)
            coord = None if x is None or y is None else (int(round(float(x))), int(round(float(y))))
        if coord is not None:
            coordinates[node_id] = coord
    return coordinates


def build_edge_rc_by_net_from_rc_timing_topology(
    topology,
    *,
    dbu,
    scale_factor=1.0,
    r_unit=1.0,
    c_unit=0.0,
):
    topo = _as_int_list(topology.get("net_flat_topo_sort", []))
    topo_start = _as_int_list(topology.get("net_flat_topo_sort_start", []))
    pin_fa = _as_int_list(topology.get("pin_fa", []))
    net_ids = _as_int_list(topology.get("net_ids", []))
    node_x = topology.get("node_x", {})
    node_y = topology.get("node_y", {})
    node_x_values = None if isinstance(node_x, dict) else _as_list(node_x)
    node_y_values = None if isinstance(node_y, dict) else _as_list(node_y)
    edge_rc_by_net = {}
    skipped = Counter()

    def coord(node_id):
        node_id = int(node_id)
        if isinstance(node_x, dict):
            x = node_x.get(node_id)
            y = node_y.get(node_id) if isinstance(node_y, dict) else None
        else:
            x = _get_index(node_x_values, node_id)
            y = _get_index(node_y_values, node_id)
        if x is None or y is None:
            return None
        return float(x), float(y)

    num_slices = max(0, len(topo_start) - 1)
    denom = max(1.0, float(scale_factor) * float(dbu))
    for slice_index in range(num_slices):
        net_id = int(net_ids[slice_index]) if slice_index < len(net_ids) else slice_index
        begin = int(topo_start[slice_index])
        end = int(topo_start[slice_index + 1])
        if begin < 0 or end < begin or end > len(topo):
            skipped["invalid_topology_range"] += 1
            continue
        edge_rc = {}
        for node_id in [int(node) for node in topo[begin:end]]:
            parent = _get_index(pin_fa, node_id, -1)
            if parent is None or int(parent) < 0:
                continue
            src_coord = coord(parent)
            dst_coord = coord(node_id)
            if src_coord is None or dst_coord is None:
                skipped["missing_coordinate"] += 1
                continue
            length_um = (
                abs(float(src_coord[0]) - float(dst_coord[0]))
                + abs(float(src_coord[1]) - float(dst_coord[1]))
            ) / denom
            edge_rc[(int(parent), int(node_id))] = {
                "r": float(length_um) * float(r_unit),
                "c": float(length_um) * float(c_unit),
                "length_um": float(length_um),
            }
        if edge_rc:
            edge_rc_by_net[net_id] = edge_rc

    return edge_rc_by_net, {
        "rc_source": "rc_timing_topology_formula",
        "edge_rc_map_net_count": len(edge_rc_by_net),
        "skipped_reasons": dict(sorted(skipped.items())),
        "r_unit": float(r_unit),
        "c_unit": float(c_unit),
        "dbu": int(dbu),
        "scale_factor": float(scale_factor),
        "length_denominator": float(denom),
    }


def parse_signal_wire_rc_from_setrc_tcl(path):
    layer_rc = {}
    signal_layer = None
    unit_note = ""
    layer_pattern = re.compile(
        r"set_layer_rc\s+.*?-layer\s+(\S+)\s+.*?-resistance\s+(\S+)\s+.*?-capacitance\s+(\S+)"
    )
    signal_pattern = re.compile(r"set_wire_rc\s+-signal\s+-layer\s+(\S+)")
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if "unit" in text.lower():
                unit_note = text
            continue
        layer_match = layer_pattern.search(line)
        if layer_match:
            layer, resistance, capacitance = layer_match.groups()
            layer_rc[str(layer)] = {
                "wire_resistance_per_micron": float(resistance),
                "wire_capacitance_per_micron": float(capacitance),
            }
            continue
        signal_match = signal_pattern.search(line)
        if signal_match:
            signal_layer = str(signal_match.group(1))

    if signal_layer is None:
        raise ValueError(f"missing set_wire_rc -signal -layer in {path}")
    if signal_layer not in layer_rc:
        raise ValueError(f"missing set_layer_rc entry for signal layer {signal_layer}")
    return {
        "rc_parameter_source": "set_rc_tcl_signal_layer_raw",
        "signal_layer": signal_layer,
        "unit_note": unit_note,
        **layer_rc[signal_layer],
    }
