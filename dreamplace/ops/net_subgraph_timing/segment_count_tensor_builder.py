import time

import torch


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


def _edge_value(edge_rc, parent, child, key):
    value = edge_rc.get((int(parent), int(child)))
    if value is None:
        value = edge_rc.get((str(int(parent)), str(int(child))))
    if isinstance(value, dict):
        return float(value.get(key, 0.0))
    if key == "r" and value is not None:
        return float(value)
    return 0.0


def _tree_preorder(root_node_id, children_by_node):
    order = []
    stack = [int(root_node_id)]
    seen = set()
    while stack:
        node_id = int(stack.pop())
        if node_id in seen:
            continue
        seen.add(node_id)
        order.append(node_id)
        children = [int(child) for child in children_by_node.get(node_id, [])]
        stack.extend(reversed(children))
    return order


def _segment_lookup(segment_state):
    return {
        (
            int(row["net_id"]),
            int(row["parent_node_id"]),
            int(row["child_node_id"]),
        ): int(local_segment_id)
        for local_segment_id, row in enumerate(tuple(segment_state.segment_rows or ()))
    }


def _split_fractions(segment_row, repeater_count):
    count = int(repeater_count)
    if count <= 0:
        return [1.0]
    px = float(segment_row["parent_x_dbu"])
    py = float(segment_row["parent_y_dbu"])
    cx = float(segment_row["child_x_dbu"])
    cy = float(segment_row["child_y_dbu"])
    dx = cx - px
    dy = cy - py
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return [1.0 / float(count + 1) for _ in range(count + 1)]
    split_ratios = []
    for index in range(count):
        ideal = float(index + 1) / float(count + 1)
        qx = round(px + dx * ideal)
        qy = round(py + dy * ideal)
        ratio = ((float(qx) - px) * dx + (float(qy) - py) * dy) / length_sq
        ratio = min(1.0, max(0.0, float(ratio)))
        split_ratios.append(ratio)
    split_ratios = sorted(split_ratios)
    fractions = []
    previous = 0.0
    for ratio in split_ratios:
        fractions.append(max(0.0, ratio - previous))
        previous = ratio
    fractions.append(max(0.0, 1.0 - previous))
    return fractions


def _segment_fraction_tables(segment_state):
    max_count = int(segment_state.max_repeater_count)
    segment_count = int(segment_state.z_param.numel())
    parent = torch.zeros((segment_count, max_count + 1), dtype=torch.float64)
    child = torch.zeros_like(parent)
    sub = torch.zeros(
        (segment_count, max_count + 1, max_count + 1),
        dtype=torch.float64,
    )
    if segment_count == 0:
        return parent, child, sub

    rows = tuple(segment_state.segment_rows or ())
    if len(rows) != segment_count:
        raise ValueError("segment rows must match segment-count state size")
    coordinates = torch.tensor(
        [
            (
                float(row["parent_x_dbu"]),
                float(row["parent_y_dbu"]),
                float(row["child_x_dbu"]),
                float(row["child_y_dbu"]),
            )
            for row in rows
        ],
        dtype=torch.float64,
    )
    parent_xy = coordinates[:, :2]
    delta_xy = coordinates[:, 2:] - parent_xy
    length_sq = torch.sum(delta_xy * delta_xy, dim=1)
    nonzero_length = length_sq > 0.0
    safe_length_sq = torch.where(
        nonzero_length,
        length_sq,
        torch.ones_like(length_sq),
    )

    parent[:, 0] = 1.0
    child[:, 0] = 1.0
    sub[:, 0, 0] = 1.0
    for count in range(1, max_count + 1):
        ratios = []
        for index in range(count):
            ideal = float(index + 1) / float(count + 1)
            quantized = torch.round(parent_xy + delta_xy * ideal)
            numerator = torch.sum((quantized - parent_xy) * delta_xy, dim=1)
            ratio = torch.clamp(numerator / safe_length_sq, min=0.0, max=1.0)
            ratios.append(
                torch.where(
                    nonzero_length,
                    ratio,
                    torch.full_like(ratio, ideal),
                )
            )
        cuts = torch.sort(torch.stack(ratios, dim=1), dim=1).values
        fractions = torch.empty((segment_count, count + 1), dtype=torch.float64)
        fractions[:, 0] = cuts[:, 0]
        if count > 1:
            fractions[:, 1:count] = cuts[:, 1:] - cuts[:, :-1]
        fractions[:, count] = 1.0 - cuts[:, -1]
        parent[:, count] = fractions[:, 0]
        child[:, count] = fractions[:, -1]
        sub[:, count, : count + 1] = fractions
    return parent, child, sub


def _record_profile_elapsed(runtime_profile, key, started_at):
    if runtime_profile is None:
        return
    runtime_profile[key] = float(runtime_profile.get(key, 0.0)) + (
        time.perf_counter() - started_at
    ) * 1000.0


def build_segment_count_timing_inputs(
    nets,
    segment_state,
    *,
    dtype=None,
    device=None,
    runtime_profile=None,
):
    """Flatten segment-count RC trees into reusable tensor timing inputs."""

    dtype = dtype if dtype is not None else segment_state.z_param.dtype
    device = torch.device(device) if device is not None else segment_state.z_param.device
    profile_started_at = time.perf_counter() if runtime_profile is not None else None
    net_sequence = tuple(nets or ())
    static_topologies = tuple(
        net.get("_segment_count_static_topology")
        if isinstance(net, dict)
        else None
        for net in net_sequence
    )
    use_static_topology = bool(net_sequence) and all(
        isinstance(topology, dict) for topology in static_topologies
    )
    lookup_started_at = time.perf_counter() if runtime_profile is not None else None
    source_row_indices = (segment_state.summary or {}).get("source_row_indices")
    if source_row_indices is None:
        source_row_indices = tuple(range(int(segment_state.z_param.numel())))
    if len(source_row_indices) != int(segment_state.z_param.numel()):
        raise ValueError("source row indices must match segment-count state size")
    identity_segment_ids = all(
        int(global_id) == int(local_id)
        for local_id, global_id in enumerate(source_row_indices)
    )
    global_to_local_segment = (
        None
        if identity_segment_ids
        else {
            int(global_id): int(local_id)
            for local_id, global_id in enumerate(source_row_indices)
        }
    )
    segment_by_edge = None if use_static_topology else _segment_lookup(segment_state)
    _record_profile_elapsed(
        runtime_profile,
        "prepared_inputs_segment_lookup_cpu_ms",
        lookup_started_at,
    )

    net_ids = []
    net_topo_start = [0]
    flat_topo_node_id = []
    edge_start = []
    edge_parent_node_id = []
    edge_child_node_id = []
    edge_parent_compact_id = []
    edge_child_compact_id = []
    edge_net_index = []
    edge_resistance = []
    edge_capacitance = []
    edge_to_segment_id = []
    node_capacitance = []
    sink_node_id = []
    net_sink_start = [0]
    sink_net_id = []
    sink_node_compact_id = []
    driver_pin_id = []
    compact_node_to_original_by_net = {}

    flatten_started_at = time.perf_counter() if runtime_profile is not None else None
    if use_static_topology:
        for net_index, static_topology in enumerate(static_topologies):
            net_id = int(static_topology["net_id"])
            flat_nodes = tuple(static_topology["flat_topo_node_id"])
            node_base = int(net_topo_start[-1])
            edge_base = len(edge_parent_node_id)
            edge_count = len(static_topology["edge_parent_node_id"])
            net_ids.append(net_id)
            driver_pin_id.append(int(static_topology["root_node_id"]))
            compact_node_to_original_by_net[net_id] = list(flat_nodes)
            flat_topo_node_id.extend(flat_nodes)
            edge_start.extend(
                edge_base + int(offset)
                for offset in tuple(static_topology["edge_start"])[:-1]
            )
            edge_parent_node_id.extend(static_topology["edge_parent_node_id"])
            edge_child_node_id.extend(static_topology["edge_child_node_id"])
            edge_parent_compact_id.extend(
                node_base + int(local_id)
                for local_id in static_topology["edge_parent_local_id"]
            )
            edge_child_compact_id.extend(
                node_base + int(local_id)
                for local_id in static_topology["edge_child_local_id"]
            )
            edge_net_index.extend([len(net_ids) - 1] * edge_count)
            edge_resistance.extend(static_topology["edge_resistance"])
            edge_capacitance.extend(static_topology["edge_capacitance"])
            for global_segment_id in static_topology["edge_global_segment_id"]:
                if int(global_segment_id) < 0:
                    edge_to_segment_id.append(-1)
                elif global_to_local_segment is None:
                    edge_to_segment_id.append(int(global_segment_id))
                else:
                    edge_to_segment_id.append(
                        int(global_to_local_segment.get(int(global_segment_id), -1))
                    )
            node_capacitance.extend(static_topology["node_capacitance"])
            for sink, local_id in zip(
                static_topology["sink_node_id"],
                static_topology["sink_local_id"],
            ):
                sink_node_id.append(int(sink))
                sink_net_id.append(net_id)
                sink_node_compact_id.append(node_base + int(local_id))
            net_topo_start.append(node_base + len(flat_nodes))
            net_sink_start.append(len(sink_node_id))
    else:
        for net_index, net in enumerate(net_sequence):
            net_id = _net_id(net, net_index)
            rc_tree = dict(net.get("rc_tree", {}) or {})
            root = rc_tree.get("root_node_id", net.get("driver_pin_id"))
            if root is None:
                continue
            root = int(root)
            children_by_node = {
                int(node): [int(child) for child in children]
                for node, children in (rc_tree.get("children_by_node", {}) or {}).items()
            }
            topo = _tree_preorder(root, children_by_node)
            if not topo:
                continue
            node_to_local = {int(node): offset for offset, node in enumerate(topo)}
            edge_rc = rc_tree.get("edge_rc", {}) or {}
            raw_node_cap = {
                int(node): float(cap)
                for node, cap in (rc_tree.get("node_cap", {}) or {}).items()
            }

            net_ids.append(int(net_id))
            driver_pin_id.append(int(root))
            compact_node_to_original_by_net[int(net_id)] = list(topo)
            flat_topo_node_id.extend(int(node) for node in topo)
            for node in topo:
                edge_start.append(len(edge_parent_node_id))
                node_capacitance.append(float(raw_node_cap.get(int(node), 0.0)))
                for child in children_by_node.get(int(node), []):
                    edge_parent_node_id.append(int(node))
                    edge_child_node_id.append(int(child))
                    edge_parent_compact_id.append(net_topo_start[-1] + int(node_to_local[int(node)]))
                    edge_child_compact_id.append(net_topo_start[-1] + int(node_to_local[int(child)]))
                    edge_net_index.append(len(net_ids) - 1)
                    edge_resistance.append(_edge_value(edge_rc, node, child, "r"))
                    edge_capacitance.append(_edge_value(edge_rc, node, child, "c"))
                    edge_to_segment_id.append(
                        int(segment_by_edge.get((int(net_id), int(node), int(child)), -1))
                    )
            for sink in rc_tree.get("sink_nodes", []):
                sink = int(sink)
                if sink not in node_to_local:
                    continue
                sink_node_id.append(sink)
                sink_net_id.append(int(net_id))
                sink_node_compact_id.append(net_topo_start[-1] + int(node_to_local[sink]))
            net_topo_start.append(len(flat_topo_node_id))
            net_sink_start.append(len(sink_node_id))

    edge_start.append(len(edge_parent_node_id))
    _record_profile_elapsed(
        runtime_profile,
        "prepared_inputs_tree_flatten_cpu_ms",
        flatten_started_at,
    )

    def long_tensor(values):
        if torch.is_tensor(values):
            return values.to(dtype=torch.long, device=device).contiguous()
        return torch.tensor(values, dtype=torch.long).to(device=device)

    def value_tensor(values):
        if torch.is_tensor(values):
            return values.to(dtype=dtype, device=device).contiguous()
        return torch.tensor(values, dtype=dtype).to(device=device)

    fraction_started_at = time.perf_counter() if runtime_profile is not None else None
    parent_fraction, child_fraction, sub_fraction = _segment_fraction_tables(segment_state)
    _record_profile_elapsed(
        runtime_profile,
        "prepared_inputs_fraction_table_cpu_ms",
        fraction_started_at,
    )

    tensor_started_at = time.perf_counter() if runtime_profile is not None else None
    result = {
        "net_ids": long_tensor(net_ids),
        "net_topo_start": long_tensor(net_topo_start),
        "flat_topo_node_id": long_tensor(flat_topo_node_id),
        "edge_start": long_tensor(edge_start),
        "edge_parent_node_id": long_tensor(edge_parent_node_id),
        "edge_child_node_id": long_tensor(edge_child_node_id),
        "edge_parent_compact_id": long_tensor(edge_parent_compact_id),
        "edge_child_compact_id": long_tensor(edge_child_compact_id),
        "edge_net_index": long_tensor(edge_net_index),
        "edge_resistance": value_tensor(edge_resistance),
        "edge_capacitance": value_tensor(edge_capacitance),
        "node_capacitance": value_tensor(node_capacitance),
        "edge_to_segment_id": long_tensor(edge_to_segment_id),
        "sink_node_id": long_tensor(sink_node_id),
        "sink_net_id": long_tensor(sink_net_id),
        "sink_node_compact_id": long_tensor(sink_node_compact_id),
        "driver_pin_id": long_tensor(driver_pin_id),
        "parent_cap_fraction": value_tensor(parent_fraction),
        "child_cap_fraction": value_tensor(child_fraction),
        "segment_sub_resistance_fraction": value_tensor(sub_fraction),
        "metadata": {
            "net_ids": list(net_ids),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "compact_node_to_original_by_net": compact_node_to_original_by_net,
            "topology_source": "state_static" if use_static_topology else "rc_tree",
            "backend_input": "segment_count_prepared_tensor",
        },
    }
    # Publish the same per-net CSR partitions as the native packer so retained
    # Python-built trees can use its active-net selector during ordinary GP.
    result["net_edge_start"] = result["edge_start"][result["net_topo_start"]]
    result["net_sink_start"] = long_tensor(net_sink_start)
    _record_profile_elapsed(
        runtime_profile,
        "prepared_inputs_tensor_materialize_cpu_ms",
        tensor_started_at,
    )
    _record_profile_elapsed(
        runtime_profile,
        "prepared_inputs_total_cpu_ms",
        profile_started_at,
    )
    return result
