import math
import logging
import time
from collections import Counter
from collections import defaultdict, deque

from dreamplace.ops.steiner_topo import (
    build_edge_rc_by_net_from_rc_timing_topology,
    build_local_steiner_topology_from_pydb,
    build_timing_aware_skeleton,
    build_timing_rooted_tree_view,
    build_topology_edges,
    parse_signal_wire_rc_from_setrc_tcl,
)
from dreamplace.ops.timing_propagation.criticality import (
    build_criticality_maps_from_timing_op,
    build_criticality_maps_from_timing_outputs,
)

from .net_eligibility import clock_net_ids


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


def _get_name(values, index):
    value = _get_index(values, int(index), "")
    return "" if value is None else str(value)


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


def _pin_lib_input_cap(pydb, pin_id):
    pin2node = _as_int_list(getattr(pydb, "pin2node_map", []))
    inst_main_id = _as_int_list(getattr(pydb, "inst_main_id", []))
    inst_libcell_offset = _as_int_list(getattr(pydb, "inst_libcell_offset", []))
    main_id_2_cell_id_start = _as_int_list(getattr(pydb, "main_id_2_cell_id_start", []))
    cell_id_2_libpin_id_start = _as_int_list(getattr(pydb, "cell_id_2_libpin_id_start", []))
    pin_2_libpin_offset = _as_int_list(getattr(pydb, "pin_2_libpin_offset", []))
    flat_lib_pin_cap = [float(value) for value in _as_list(getattr(pydb, "flat_lib_pin_cap", []))]
    flat_lib_pin_rcap = [
        float(value) for value in _as_list(getattr(pydb, "flat_lib_pin_rcap", []))
    ]

    pin_id = int(pin_id)
    node_id = _get_index(pin2node, pin_id, -1)
    libpin_offset = _get_index(pin_2_libpin_offset, pin_id, -1)
    if node_id is None or node_id < 0 or libpin_offset is None or libpin_offset < 0:
        return None
    main_id = _get_index(inst_main_id, node_id, -1)
    libcell_offset = _get_index(inst_libcell_offset, node_id, 0)
    if main_id is None or main_id < 0:
        return None
    cell_start = _get_index(main_id_2_cell_id_start, main_id, None)
    if cell_start is None:
        return None
    cell_id = int(cell_start) + int(libcell_offset or 0)
    libpin_start = _get_index(cell_id_2_libpin_id_start, cell_id, None)
    if libpin_start is None:
        return None
    libpin_id = int(libpin_start) + int(libpin_offset)
    active_caps = flat_lib_pin_rcap if flat_lib_pin_rcap else flat_lib_pin_cap
    if libpin_id < 0 or libpin_id >= len(active_caps):
        return None
    cap = float(active_caps[libpin_id])
    return cap if math.isfinite(cap) and cap > 0.0 else None


def _node_cap_summary(values, *, nondefault_count, default_count, io_sink_count=0):
    if not values:
        return {
            "node_cap_nondefault_count": int(nondefault_count),
            "node_cap_default_count": int(default_count),
            "node_cap_io_sink_count": int(io_sink_count),
            "node_cap_min": None,
            "node_cap_max": None,
            "node_cap_mean": None,
        }
    return {
        "node_cap_nondefault_count": int(nondefault_count),
        "node_cap_default_count": int(default_count),
        "node_cap_io_sink_count": int(io_sink_count),
        "node_cap_min": min(values),
        "node_cap_max": max(values),
        "node_cap_mean": sum(values) / float(len(values)),
    }


def _iter_requested_net_ids(flat_net2pin_start, net_ids, max_nets):
    num_nets = max(0, len(flat_net2pin_start) - 1)
    requested = [int(net_id) for net_id in net_ids] if net_ids is not None else range(num_nets)
    count = 0
    for net_id in requested:
        if max_nets is not None and count >= int(max_nets):
            break
        count += 1
        yield net_id


def _build_star_edges(driver_pin_id, net_pin_ids):
    driver_pin_id = int(driver_pin_id)
    return [
        (driver_pin_id, int(pin_id))
        for pin_id in net_pin_ids
        if int(pin_id) != driver_pin_id
    ]


def _edge_rc_from_coordinates(edges, coordinates, dbu):
    edge_rc = {}
    for src, dst in edges:
        src_coord = coordinates.get(int(src))
        dst_coord = coordinates.get(int(dst))
        if src_coord is None or dst_coord is None:
            continue
        length_um = (
            abs(float(src_coord[0]) - float(dst_coord[0]))
            + abs(float(src_coord[1]) - float(dst_coord[1]))
        ) / max(1.0, float(dbu))
        edge_rc[(int(src), int(dst))] = {"r": max(1.0, length_um)}
    return edge_rc


def _normalize_edge_rc_map(edge_rc):
    normalized = {}
    for edge, value in (edge_rc or {}).items():
        src, dst = edge
        item = {}
        for key, numeric in dict(value).items():
            item[str(key)] = float(numeric)
        normalized[(int(src), int(dst))] = item
    return normalized


def _children_by_node_from_edges(root_node_id, node_ids, edges):
    root_node_id = int(root_node_id)
    adjacency = defaultdict(list)
    for src, dst in edges:
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))

    children_by_node = {int(node_id): [] for node_id in node_ids}
    visited = {root_node_id}
    queue = deque([root_node_id])
    while queue:
        node_id = queue.popleft()
        for next_node in adjacency.get(node_id, []):
            next_node = int(next_node)
            if next_node in visited:
                continue
            visited.add(next_node)
            children_by_node.setdefault(node_id, []).append(next_node)
            children_by_node.setdefault(next_node, [])
            queue.append(next_node)
    return children_by_node


def _directed_edges_from_children(children_by_node):
    return [
        (int(parent), int(child))
        for parent, children in children_by_node.items()
        for child in children
    ]


def _driver_rooted_live_topology_view(
    *,
    net_id,
    net_pin_ids,
    driver_pin_id,
    directed_edges,
    coordinates,
    flat_first_pin_id,
):
    """Return the legacy rooted-tree fields when live topology already has the driver root.

    ``SteinerTopo`` emits directed parent-to-child edges.  Route-B still needs
    the legacy tree contract, but a second undirected adjacency build and BFS
    are redundant when the timing driver is already the emitted root.  Any
    malformed or non-rooted input returns ``None`` so callers retain the legacy
    `build_timing_rooted_tree_view` fallback.
    """
    driver_pin_id = int(driver_pin_id)
    node_ids = {
        int(node_id)
        for node_id in list(net_pin_ids or ())
    }
    children_by_node = {}
    parent_by_node = {}
    for parent, child in directed_edges or ():
        parent = int(parent)
        child = int(child)
        node_ids.add(parent)
        node_ids.add(child)
        if child == driver_pin_id or child in parent_by_node:
            return None
        parent_by_node[child] = parent
        children_by_node.setdefault(parent, []).append(child)
        children_by_node.setdefault(child, [])
    if driver_pin_id not in node_ids:
        return None
    for node_id in node_ids:
        children_by_node.setdefault(int(node_id), [])
    if len(parent_by_node) != max(0, len(node_ids) - 1):
        return None
    for children in children_by_node.values():
        children.sort()

    visited = set()
    stack = [driver_pin_id]
    while stack:
        node_id = int(stack.pop())
        if node_id in visited:
            return None
        visited.add(node_id)
        stack.extend(reversed(children_by_node[node_id]))
    if visited != node_ids:
        return None

    ordered_node_ids = sorted(node_ids)
    ordered_net_pins = sorted({int(pin_id) for pin_id in net_pin_ids or ()})
    return {
        "status": "ok",
        "unsupported_reasons": [],
        "net_id": int(net_id),
        "root_node_id": driver_pin_id,
        "flat_first_pin_id": int(flat_first_pin_id),
        "driver_is_flat_first_pin": int(flat_first_pin_id) == driver_pin_id,
        "reroot_applied": int(flat_first_pin_id) != driver_pin_id,
        "sink_pin_ids": [
            pin_id for pin_id in ordered_net_pins if pin_id != driver_pin_id
        ],
        "node_ids": ordered_node_ids,
        "edge_pairs": [
            (int(parent), int(child))
            for parent, child in directed_edges or ()
        ],
        "children_by_node": {
            int(node_id): list(children_by_node[int(node_id)])
            for node_id in ordered_node_ids
        },
        "coordinates": {
            int(node_id): tuple(value)
            for node_id, value in dict(coordinates or {}).items()
        },
        "is_connected": True,
        "has_cycle": False,
    }


def _record_runtime_profile_elapsed(runtime_profile, name, started_at):
    if runtime_profile is not None:
        runtime_profile[name] = float(runtime_profile.get(name, 0.0) or 0.0) + (
            time.perf_counter() - started_at
        ) * 1000.0


def build_buffering_nets_from_pydb(
    pydb,
    *,
    net_ids=None,
    max_nets=None,
    max_net_degree=None,
    topology=None,
    topology_source="flat_net_star",
    default_sink_slack=0.0,
    default_npath=1,
    sink_slack_by_pin=None,
    npath_by_pin=None,
    criticality_source=None,
    edge_rc_by_net=None,
    pin_cap_by_pin_id=None,
    pin_cap_source=None,
    enable_rebranching=False,
    max_rebranched_sink_ratio=0.2,
    rebranching_max_wirelength_delta_ratio=None,
    runtime_profile=None,
):
    adapter_started_at = time.perf_counter()
    runtime_stage_started_at = time.perf_counter()
    flat_net2pin = _as_int_list(getattr(pydb, "flat_net2pin_map", []))
    flat_net2pin_start = _as_int_list(getattr(pydb, "flat_net2pin_start_map", []))
    net2driver = _as_int_list(getattr(pydb, "net2driver_pin_map", []))
    net_names = list(getattr(pydb, "net_names", []) or [])
    excluded_clocks = clock_net_ids(pydb)
    pin_names = list(getattr(pydb, "pin_names", []) or [])
    pin2node = _as_int_list(getattr(pydb, "pin2node_map", []))
    node_x_values = _as_list(getattr(pydb, "node_x", []))
    node_y_values = _as_list(getattr(pydb, "node_y", []))
    num_terminals = int(getattr(pydb, "num_terminals", 0) or 0)
    num_terminal_NIs = int(getattr(pydb, "num_terminal_NIs", 0) or 0)
    inferred_movable_nodes = max(
        0,
        len(node_x_values) - int(num_terminals) - int(num_terminal_NIs),
    )
    num_movable_nodes = int(
        getattr(pydb, "num_movable_nodes", inferred_movable_nodes)
        or inferred_movable_nodes
    )
    pin_offset_x_values = _as_list(getattr(pydb, "pin_offset_x", []))
    pin_offset_y_values = _as_list(getattr(pydb, "pin_offset_y", []))
    inst_main_id_values = _as_int_list(getattr(pydb, "inst_main_id", []))
    inst_libcell_offset_values = _as_int_list(
        getattr(pydb, "inst_libcell_offset", [])
    )
    main_id_2_cell_id_start_values = _as_int_list(
        getattr(pydb, "main_id_2_cell_id_start", [])
    )
    cell_id_2_libpin_id_start_values = _as_int_list(
        getattr(pydb, "cell_id_2_libpin_id_start", [])
    )
    pin_2_libpin_offset_values = _as_int_list(
        getattr(pydb, "pin_2_libpin_offset", [])
    )
    flat_lib_pin_cap_values = [
        float(value) for value in _as_list(getattr(pydb, "flat_lib_pin_cap", []))
    ]
    flat_lib_pin_rcap_values = [
        float(value) for value in _as_list(getattr(pydb, "flat_lib_pin_rcap", []))
    ]
    active_lib_pin_cap_values = (
        flat_lib_pin_rcap_values
        if flat_lib_pin_rcap_values
        else flat_lib_pin_cap_values
    )
    active_lib_pin_cap_source = (
        "flat_lib_pin_rcap"
        if flat_lib_pin_rcap_values
        else "flat_lib_pin_cap"
    )
    dbu = int(getattr(pydb, "dbu", 1000) or 1000)
    _record_runtime_profile_elapsed(
        runtime_profile,
        "adapter_static_input_materialize_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    skipped = Counter()
    nets = []

    if not flat_net2pin or len(flat_net2pin_start) < 2 or not net2driver:
        return [], {
            "artifact": "buffering_net_adapter_summary",
            "artifact_version": 1,
            "status": "unsupported",
            "unsupported_reasons": ["missing_net_pin_or_driver_arrays"],
            "topology_source": topology_source,
            "built_net_count": 0,
            "skipped_reasons": {},
        }

    if topology is not None:
        topology_source = topology.get("topology_source", "steiner")
    external_sink_slack = {
        int(pin_id): float(slack)
        for pin_id, slack in (sink_slack_by_pin or {}).items()
    }
    external_npath = {
        int(pin_id): int(npath)
        for pin_id, npath in (npath_by_pin or {}).items()
    }
    if criticality_source is None:
        criticality_source = (
            "external_maps"
            if sink_slack_by_pin is not None or npath_by_pin is not None
            else "defaults"
        )
    raw_edge_rc_by_net = edge_rc_by_net or {}
    normalized_edge_rc_by_net = {}
    rc_source = "external_edge_rc" if raw_edge_rc_by_net else "manhattan_length_proxy"
    external_pin_cap_by_pin_id = (
        {
            int(pin_id): float(cap)
            for pin_id, cap in (pin_cap_by_pin_id or {}).items()
            if cap is not None and math.isfinite(float(cap)) and float(cap) >= 0.0
        }
        if pin_cap_by_pin_id is not None
        else None
    )
    effective_pin_cap_source = (
        str(pin_cap_source)
        if pin_cap_source is not None
        else "external_pin_cap_by_pin_id"
        if external_pin_cap_by_pin_id is not None
        else None
    )
    node_cap_values = []
    node_cap_nondefault_count = 0
    node_cap_default_count = 0
    node_cap_io_sink_count = 0
    node_cap_external_non_sink_count = 0
    external_edge_rc_missing_count = 0
    rebranching_summaries = []
    coordinate_cache = {}
    pin_cap_cache = {}
    topology_node_x = topology.get("node_x", {}) if topology is not None else {}
    topology_node_y = topology.get("node_y", {}) if topology is not None else {}
    topology_node_x_dbu = topology.get("node_x_dbu", {}) if topology is not None else {}
    topology_node_y_dbu = topology.get("node_y_dbu", {}) if topology is not None else {}
    topology_node_x_values = (
        None if isinstance(topology_node_x, dict) else _as_list(topology_node_x)
    )
    topology_node_y_values = (
        None if isinstance(topology_node_y, dict) else _as_list(topology_node_y)
    )
    topology_node_x_dbu_values = (
        None if isinstance(topology_node_x_dbu, dict) else _as_list(topology_node_x_dbu)
    )
    topology_node_y_dbu_values = (
        None if isinstance(topology_node_y_dbu, dict) else _as_list(topology_node_y_dbu)
    )
    prefer_topology_pin_coordinates = (
        topology is not None
        and raw_edge_rc_by_net
        and str(topology_source) == "live_timing_topology"
    )
    _record_runtime_profile_elapsed(
        runtime_profile,
        "adapter_static_setup_ms",
        runtime_stage_started_at,
    )

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

    def pin_lib_input_cap(pin_id):
        pin_id = int(pin_id)
        if external_pin_cap_by_pin_id is not None:
            cap = external_pin_cap_by_pin_id.get(pin_id)
            if cap is not None:
                return float(cap)
        if pin_id in pin_cap_cache:
            return pin_cap_cache[pin_id]
        node_id = _get_index(pin2node, pin_id, -1)
        libpin_offset = _get_index(pin_2_libpin_offset_values, pin_id, -1)
        if node_id is None or node_id < 0 or libpin_offset is None or libpin_offset < 0:
            pin_cap_cache[pin_id] = None
            return None
        main_id = _get_index(inst_main_id_values, node_id, -1)
        libcell_offset = _get_index(inst_libcell_offset_values, node_id, 0)
        if main_id is None or main_id < 0:
            pin_cap_cache[pin_id] = None
            return None
        cell_start = _get_index(main_id_2_cell_id_start_values, main_id, None)
        if cell_start is None:
            pin_cap_cache[pin_id] = None
            return None
        cell_id = int(cell_start) + int(libcell_offset or 0)
        libpin_start = _get_index(cell_id_2_libpin_id_start_values, cell_id, None)
        if libpin_start is None:
            pin_cap_cache[pin_id] = None
            return None
        libpin_id = int(libpin_start) + int(libpin_offset)
        if libpin_id < 0 or libpin_id >= len(active_lib_pin_cap_values):
            pin_cap_cache[pin_id] = None
            return None
        cap = float(active_lib_pin_cap_values[libpin_id])
        result = cap if math.isfinite(cap) and cap > 0.0 else None
        pin_cap_cache[pin_id] = result
        return result

    def node_coordinate(node_id):
        node_id = int(node_id)
        if node_id < len(pin2node) and not prefer_topology_pin_coordinates:
            return pin_coordinate(node_id)
        if isinstance(topology_node_x, dict):
            x = topology_node_x.get(node_id)
            y = topology_node_y.get(node_id) if isinstance(topology_node_y, dict) else None
        else:
            x = _get_index(topology_node_x_values, node_id)
            y = _get_index(topology_node_y_values, node_id)
        if x is not None and y is not None:
            return int(round(float(x))), int(round(float(y)))
        if node_id < len(pin2node):
            return pin_coordinate(node_id)
        if x is None or y is None:
            return None
        return int(round(float(x))), int(round(float(y)))

    def node_coordinate_dbu(node_id):
        node_id = int(node_id)
        if isinstance(topology_node_x_dbu, dict):
            x = topology_node_x_dbu.get(node_id)
            y = (
                topology_node_y_dbu.get(node_id)
                if isinstance(topology_node_y_dbu, dict)
                else None
            )
        else:
            x = _get_index(topology_node_x_dbu_values, node_id)
            y = _get_index(topology_node_y_dbu_values, node_id)
        if x is not None and y is not None:
            return int(round(float(x))), int(round(float(y)))
        return pin_coordinate(node_id) if node_id < len(pin2node) else node_coordinate(node_id)

    def edge_rc_for_net(net_id):
        net_id = int(net_id)
        if net_id in normalized_edge_rc_by_net:
            return normalized_edge_rc_by_net[net_id]
        raw_edge_rc = raw_edge_rc_by_net.get(net_id, {})
        normalized = _normalize_edge_rc_map(raw_edge_rc)
        normalized_edge_rc_by_net[net_id] = normalized
        return normalized

    adapter_start_time = time.time()
    processed_net_count = 0
    for net_id in _iter_requested_net_ids(flat_net2pin_start, net_ids, max_nets):
        runtime_stage_started_at = time.perf_counter()
        processed_net_count += 1
        if net_id in excluded_clocks:
            skipped["clock_net"] += 1
            continue
        if processed_net_count % 5000 == 0:
            logging.info(
                "build_buffering_nets_from_pydb progress processed=%d built=%d elapsed=%.3f ms",
                processed_net_count,
                len(nets),
                (time.time() - adapter_start_time) * 1000.0,
            )
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
        driver_pin_id = int(driver_pin_id)
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_validate_and_pin_slice_ms",
            runtime_stage_started_at,
        )

        topology_and_coordinates_started_at = time.perf_counter()
        fallback_star_started_at = time.perf_counter()
        node_ids = list(net_pin_ids)
        edges = _build_star_edges(driver_pin_id, net_pin_ids)
        coordinates = {pin_id: pin_coordinate(pin_id) for pin_id in net_pin_ids}
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_fallback_star_setup_ms",
            fallback_star_started_at,
        )
        rooted_children_by_node = None
        if topology is not None:
            runtime_stage_started_at = time.perf_counter()
            topology_result = build_topology_edges(net_id, topology)
            if topology_result is None:
                skipped["missing_topology"] += 1
                continue
            node_ids, edges = topology_result
            if runtime_profile is not None:
                pin_fa = topology.get("_pin_fa_int", ())
                driver_parent = _get_index(pin_fa, driver_pin_id, -1)
                root_key = (
                    "adapter_topology_driver_root_count"
                    if driver_parent is not None and int(driver_parent) < 0
                    else "adapter_topology_driver_nonroot_count"
                )
                runtime_profile[root_key] = int(
                    runtime_profile.get(root_key, 0) or 0
                ) + 1
            _record_runtime_profile_elapsed(
                runtime_profile,
                "adapter_net_topology_edge_slice_ms",
                runtime_stage_started_at,
            )
            runtime_stage_started_at = time.perf_counter()
            coordinates = {
                int(node_id): node_coordinate(node_id)
                for node_id in node_ids
            }
            coordinates_dbu = {
                int(node_id): node_coordinate_dbu(node_id)
                for node_id in node_ids
            }
            _record_runtime_profile_elapsed(
                runtime_profile,
                "adapter_net_topology_coordinate_build_ms",
                runtime_stage_started_at,
            )
            runtime_stage_started_at = time.perf_counter()
            tree_view = None
            if str(topology_source) == "live_timing_topology":
                pin_fa = topology.get("_pin_fa_int", ())
                driver_parent = _get_index(pin_fa, driver_pin_id, -1)
                if driver_parent is not None and int(driver_parent) < 0:
                    tree_view = _driver_rooted_live_topology_view(
                        net_id=net_id,
                        net_pin_ids=net_pin_ids,
                        driver_pin_id=driver_pin_id,
                        directed_edges=edges,
                        coordinates=coordinates,
                        flat_first_pin_id=net_pin_ids[0],
                    )
            if tree_view is None:
                tree_view = build_timing_rooted_tree_view(
                    net_id=net_id,
                    net_pin_ids=net_pin_ids,
                    driver_pin_id=driver_pin_id,
                    undirected_edges=edges,
                    coordinates=coordinates,
                    flat_first_pin_id=net_pin_ids[0],
                )
                if runtime_profile is not None:
                    runtime_profile["adapter_topology_legacy_reroot_count"] = int(
                        runtime_profile.get("adapter_topology_legacy_reroot_count", 0)
                        or 0
                    ) + 1
            elif runtime_profile is not None:
                runtime_profile["adapter_topology_direct_rooted_view_count"] = int(
                    runtime_profile.get("adapter_topology_direct_rooted_view_count", 0)
                    or 0
                ) + 1
            _record_runtime_profile_elapsed(
                runtime_profile,
                "adapter_net_topology_reroot_ms",
                runtime_stage_started_at,
            )
            if tree_view.get("status") != "ok":
                skipped["unsupported_topology_tree_view"] += 1
                continue
            runtime_stage_started_at = time.perf_counter()
            node_ids = [int(node_id) for node_id in tree_view.get("node_ids", node_ids)]
            rooted_children_by_node = {
                int(parent): [int(child) for child in children]
                for parent, children in tree_view.get("children_by_node", {}).items()
            }
            edges = _directed_edges_from_children(rooted_children_by_node)
            coordinates = {
                int(node_id): tuple(coord)
                for node_id, coord in tree_view.get("coordinates", coordinates).items()
            }
            coordinates_dbu = {
                int(node_id): tuple(coord)
                for node_id, coord in coordinates_dbu.items()
                if coord is not None and int(node_id) in coordinates
            }
            _record_runtime_profile_elapsed(
                runtime_profile,
                "adapter_net_topology_normalize_ms",
                runtime_stage_started_at,
            )
        else:
            coordinates_dbu = dict(coordinates)

        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_topology_and_coordinates_ms",
            topology_and_coordinates_started_at,
        )

        runtime_stage_started_at = time.perf_counter()
        if any(coordinates.get(int(node_id)) is None for node_id in node_ids):
            skipped["missing_coordinate"] += 1
            continue
        if any(coordinates_dbu.get(int(node_id)) is None for node_id in node_ids):
            skipped["missing_dbu_coordinate"] += 1
            continue
        sink_pin_ids = [pin_id for pin_id in net_pin_ids if pin_id != driver_pin_id]
        local_sink_slack_by_pin = {
            int(sink_pin_id): float(
                external_sink_slack.get(int(sink_pin_id), default_sink_slack)
            )
            for sink_pin_id in sink_pin_ids
        }
        local_npath_by_pin = {
            int(sink_pin_id): int(
                external_npath.get(int(sink_pin_id), default_npath)
            )
            for sink_pin_id in sink_pin_ids
        }
        pin_name_by_id = {
            int(pin_id): _get_name(pin_names, int(pin_id))
            for pin_id in net_pin_ids
        }
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_sink_metadata_ms",
            runtime_stage_started_at,
        )
        rebranching_summary = None
        if enable_rebranching:
            tree_view = build_timing_rooted_tree_view(
                net_id=net_id,
                net_pin_ids=net_pin_ids,
                driver_pin_id=driver_pin_id,
                undirected_edges=edges,
                coordinates=coordinates,
                flat_first_pin_id=net_pin_ids[0],
            )
            skeleton, rebranching_summary = build_timing_aware_skeleton(
                tree_view,
                sink_slack_by_pin=local_sink_slack_by_pin,
                npath_by_pin=local_npath_by_pin,
                max_sink_fraction=max_rebranched_sink_ratio,
                max_wirelength_delta_ratio=rebranching_max_wirelength_delta_ratio,
            )
            rebranching_summaries.append(rebranching_summary)
            if skeleton.get("status") == "ok":
                edges = [(int(src), int(dst)) for src, dst in skeleton.get("edge_pairs", [])]
                node_ids = [int(node_id) for node_id in skeleton.get("node_ids", node_ids)]
                coordinates = {
                    int(node_id): tuple(coord)
                    for node_id, coord in skeleton.get("coordinates", coordinates).items()
                }
                rooted_children_by_node = None
        runtime_stage_started_at = time.perf_counter()
        node_cap = {int(node_id): 0.0 for node_id in node_ids}
        sink_pin_ids = [pin_id for pin_id in net_pin_ids if pin_id != driver_pin_id]
        if external_pin_cap_by_pin_id is not None:
            sink_pin_id_set = {int(pin_id) for pin_id in sink_pin_ids}
            for node_id in node_ids:
                node_id = int(node_id)
                cap = external_pin_cap_by_pin_id.get(node_id)
                if cap is None:
                    continue
                node_cap[node_id] = float(cap)
                if node_id not in sink_pin_id_set:
                    node_cap_external_non_sink_count += 1
                    node_cap_nondefault_count += 1
                    node_cap_values.append(float(cap))
        for sink_pin_id in sink_pin_ids:
            sink_node_id = _get_index(pin2node, int(sink_pin_id), -1)
            if (
                sink_node_id is not None
                and int(sink_node_id) >= int(num_movable_nodes + num_terminals)
            ):
                cap = 0.0
                node_cap_io_sink_count += 1
                node_cap[int(sink_pin_id)] = float(cap)
                node_cap_values.append(float(cap))
                continue
            cap = pin_lib_input_cap(sink_pin_id)
            if cap is None:
                raise AssertionError(
                    "missing sink pin capacitance while building buffering net "
                    f"net_id={net_id} sink_pin_id={sink_pin_id} "
                    f"sink_pin_name={_get_name(pin_names, sink_pin_id)!r}"
                )
            node_cap_nondefault_count += 1
            node_cap[int(sink_pin_id)] = float(cap)
            node_cap_values.append(float(cap))
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_node_cap_build_ms",
            runtime_stage_started_at,
        )
        runtime_stage_started_at = time.perf_counter()
        if rooted_children_by_node is None:
            children_by_node = _children_by_node_from_edges(
                driver_pin_id,
                node_ids,
                edges,
            )
            directed_edges = _directed_edges_from_children(children_by_node)
        else:
            children_by_node = rooted_children_by_node
            directed_edges = list(edges)
        external_edge_rc = edge_rc_for_net(net_id)
        if raw_edge_rc_by_net:
            edge_rc = {}
            for src, dst in directed_edges:
                record = external_edge_rc.get((int(src), int(dst)))
                if record is None:
                    record = external_edge_rc.get((int(dst), int(src)))
                if record is None:
                    external_edge_rc_missing_count += 1
                    skipped["missing_external_edge_rc"] += 1
                    continue
                edge_rc[(int(src), int(dst))] = dict(record)
        else:
            edge_rc = _edge_rc_from_coordinates(directed_edges, coordinates, dbu)
        if len(edge_rc) != len(directed_edges):
            skipped["incomplete_edge_rc"] += 1
            continue
        net_record = {
                "net_id": int(net_id),
                "net_name": _get_name(net_names, net_id),
                "net_pin_ids": net_pin_ids,
                "driver_pin_id": driver_pin_id,
                "driver_pin_name": _get_name(pin_names, driver_pin_id),
                "pin_name_by_id": pin_name_by_id,
                "flat_first_pin_id": net_pin_ids[0],
                "undirected_edges": [(int(src), int(dst)) for src, dst in edges],
                "coordinates": {
                    int(node_id): (int(coord[0]), int(coord[1]))
                    for node_id, coord in coordinates.items()
                },
                "coordinates_dbu": {
                    int(node_id): (int(coord[0]), int(coord[1]))
                    for node_id, coord in coordinates_dbu.items()
                },
                "sink_slack_by_pin": {
                    int(pin_id): float(slack)
                    for pin_id, slack in local_sink_slack_by_pin.items()
                },
                "npath_by_pin": {
                    int(pin_id): int(npath)
                    for pin_id, npath in local_npath_by_pin.items()
                },
                "rc_tree": {
                    "root_node_id": driver_pin_id,
                    "children_by_node": children_by_node,
                    "edge_rc": edge_rc,
                    "node_cap": node_cap,
                    "sink_nodes": sink_pin_ids,
                },
            }
        if rebranching_summary is not None:
            net_record["rebranching_summary"] = rebranching_summary
        nets.append(net_record)
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_net_rc_tree_and_record_build_ms",
            runtime_stage_started_at,
        )

    summary = {
        "artifact": "buffering_net_adapter_summary",
        "artifact_version": 1,
        "status": "ok",
        "topology_source": topology_source,
        "requested_net_count": len(list(net_ids)) if net_ids is not None else len(flat_net2pin_start) - 1,
        "built_net_count": len(nets),
        "skipped_reasons": dict(sorted(skipped.items())),
        "max_nets": None if max_nets is None else int(max_nets),
        "max_net_degree": None if max_net_degree is None else int(max_net_degree),
        "dbu": dbu,
        "criticality_source": criticality_source,
        "sink_slack_map_entry_count": len(external_sink_slack),
        "npath_map_entry_count": len(external_npath),
        "rc_source": rc_source,
        "node_cap_libpin_cap_source": (
            effective_pin_cap_source or active_lib_pin_cap_source
        ),
        "node_cap_external_non_sink_count": int(node_cap_external_non_sink_count),
        "edge_rc_map_net_count": len(raw_edge_rc_by_net),
        "external_edge_rc_missing_count": int(external_edge_rc_missing_count),
        "rebranching_enabled": bool(enable_rebranching),
        "rebranching_net_count": len(rebranching_summaries),
        "rebranching_max_wirelength_delta_ratio": (
            None
            if rebranching_max_wirelength_delta_ratio is None
            else float(rebranching_max_wirelength_delta_ratio)
        ),
    }
    if runtime_profile is not None:
        _record_runtime_profile_elapsed(
            runtime_profile,
            "adapter_total_ms",
            adapter_started_at,
        )
        adapter_stage_names = (
            "adapter_static_input_materialize_ms",
            "adapter_static_setup_ms",
            "adapter_net_validate_and_pin_slice_ms",
            "adapter_net_topology_and_coordinates_ms",
            "adapter_net_sink_metadata_ms",
            "adapter_net_node_cap_build_ms",
            "adapter_net_rc_tree_and_record_build_ms",
        )
        runtime_profile["adapter_accounted_ms"] = sum(
            float(runtime_profile.get(name, 0.0) or 0.0)
            for name in adapter_stage_names
        )
        runtime_profile["adapter_unattributed_ms"] = max(
            0.0,
            float(runtime_profile["adapter_total_ms"])
            - float(runtime_profile["adapter_accounted_ms"]),
        )
        runtime_profile["adapter_processed_net_count"] = int(processed_net_count)
        runtime_profile["adapter_built_net_count"] = int(len(nets))
    if rebranching_summaries:
        status_counts = Counter(
            str(item.get("rebranching_status", item.get("status", "unknown")))
            for item in rebranching_summaries
        )
        skip_reason_counts = Counter(
            str(item.get("rebranching_skip_reason", ""))
            for item in rebranching_summaries
            if str(item.get("rebranching_skip_reason", ""))
        )
        candidate_sources = Counter(
            str(item.get("candidate_set_skeleton_source", "unknown"))
            for item in rebranching_summaries
        )
        wirelength_deltas = [
            float(item.get("rebranching_wirelength_delta", 0.0) or 0.0)
            for item in rebranching_summaries
        ]
        summary.update(
            {
                "rebranching_status": (
                    "ok"
                    if status_counts.get("ok", 0) == len(rebranching_summaries)
                    else "mixed"
                    if len(status_counts) > 1
                    else next(iter(status_counts))
                ),
                "rebranched_sink_count": sum(
                    int(item.get("rebranched_sink_count", 0))
                    for item in rebranching_summaries
                ),
                "rebranching_status_counts": dict(sorted(status_counts.items())),
                "rebranching_skip_reason_counts": dict(
                    sorted(skip_reason_counts.items())
                ),
                "wirelength_guard_triggered_count": sum(
                    1
                    for item in rebranching_summaries
                    if bool(item.get("wirelength_guard_triggered", False))
                ),
                "rebranching_wirelength_delta_sum": sum(wirelength_deltas),
                "rebranching_wirelength_delta_max": (
                    max(wirelength_deltas) if wirelength_deltas else 0.0
                ),
                "candidate_set_skeleton_source": (
                    "timing_aware_rebranched_skeleton"
                    if candidate_sources.get(
                        "timing_aware_rebranched_skeleton",
                        0,
                    )
                    == len(rebranching_summaries)
                    else "mixed"
                    if len(candidate_sources) > 1
                    else next(iter(candidate_sources))
                ),
                "candidate_set_skeleton_source_counts": dict(
                    sorted(candidate_sources.items())
                ),
            }
        )
    else:
        summary.update(
            {
                "rebranching_status": "disabled",
                "rebranched_sink_count": 0,
                "candidate_set_skeleton_source": topology_source,
            }
        )
    node_cap_stats = _node_cap_summary(
        node_cap_values,
        nondefault_count=node_cap_nondefault_count,
        default_count=node_cap_default_count,
        io_sink_count=node_cap_io_sink_count,
    )
    summary.update(node_cap_stats)
    if node_cap_nondefault_count > 0 and node_cap_default_count == 0:
        summary["node_cap_source"] = (
            "libpin_cap_and_io_zero_cap"
            if node_cap_io_sink_count > 0
            else "libpin_cap"
        )
    elif node_cap_io_sink_count > 0 and node_cap_default_count == 0:
        summary["node_cap_source"] = "io_zero_cap"
    elif node_cap_nondefault_count > 0:
        summary["node_cap_source"] = "mixed_libpin_cap_and_default"
    else:
        summary["node_cap_source"] = "default_unit_sink_cap"
    return nets, summary
