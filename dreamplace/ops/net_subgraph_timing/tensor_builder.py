import torch


NATIVE_INPUT_KEYS = (
    "net_flat_topo_sort",
    "net_flat_topo_sort_start",
    "pin_fa",
    "flat_pin_to_start",
    "flat_pin_to",
    "edge_resistance",
    "node_capacitance",
    "edge_capacitance",
    "driver_arrival",
    "driver_slew",
    "candidate_node_id",
    "candidate_bu",
    "buffer_input_cap",
    "buffer_delay",
    "buffer_output_slew",
    "sink_node_id",
    "sink_net_index",
)


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


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


def _edge_value(edge_rc, parent, child, key):
    value = edge_rc.get((int(parent), int(child)))
    if value is None:
        value = edge_rc.get((str(int(parent)), str(int(child))))
    if isinstance(value, dict):
        return float(value.get(key, 0.0))
    if key == "r" and value is not None:
        return float(value)
    return 0.0


def _edge_record(edge_rc, parent, child):
    return {
        "r": _edge_value(edge_rc, parent, child, "r"),
        "c": _edge_value(edge_rc, parent, child, "c"),
    }


def _single_segment_child(candidate):
    if candidate.get("tree_node_id") is not None or candidate.get("node_id") is not None:
        return None
    parent = candidate.get("parent_node_id")
    child_node_ids = candidate.get("child_node_ids") or []
    if parent is None or len(child_node_ids) != 1:
        return None
    return int(parent), int(child_node_ids[0])


def _candidate_net_id(candidate, net_ids):
    value = candidate.get("net_id", candidate.get("affected_net_id"))
    if value is not None:
        return int(value)
    if len(net_ids) == 1:
        return int(net_ids[0])
    return None


def _segment_split_ratio(candidate, coordinates, parent, child):
    explicit_ratio = candidate.get("segment_split_ratio")
    if explicit_ratio is not None:
        ratio = float(explicit_ratio)
        return ratio if 0.0 < ratio < 1.0 else None
    if "x_dbu" not in candidate or "y_dbu" not in candidate:
        return None
    if int(parent) not in coordinates or int(child) not in coordinates:
        return None
    px, py = coordinates[int(parent)]
    cx, cy = coordinates[int(child)]
    dx = float(cx) - float(px)
    dy = float(cy) - float(py)
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return None
    qx = float(candidate["x_dbu"]) - float(px)
    qy = float(candidate["y_dbu"]) - float(py)
    ratio = (qx * dx + qy * dy) / length_sq
    if ratio <= 0.0 or ratio >= 1.0:
        return None
    return ratio


def _collect_original_node_ids(nets):
    node_ids = []
    for net in nets:
        rc_tree = net.get("rc_tree", {})
        root = rc_tree.get("root_node_id", net.get("driver_pin_id"))
        if root is not None:
            node_ids.append(int(root))
        for node, children in rc_tree.get("children_by_node", {}).items():
            node_ids.append(int(node))
            node_ids.extend(int(child) for child in children)
        node_ids.extend(int(node) for node in rc_tree.get("sink_nodes", []))
        node_ids.extend(int(node) for node in (net.get("coordinates") or {}).keys())
    return node_ids


def _build_segment_split_specs(nets, candidates):
    net_ids = [_net_id(net, index) for index, net in enumerate(nets)]
    net_by_id = {_net_id(net, index): net for index, net in enumerate(nets)}
    min_node_id = min(_collect_original_node_ids(nets) or [0])
    next_synthetic_node_id = min(-1, min_node_id - 1)
    split_specs_by_net = {}
    split_node_by_candidate_index = {}

    for candidate_index, candidate in enumerate(candidates):
        segment = _single_segment_child(candidate)
        if segment is None:
            continue
        net_id = _candidate_net_id(candidate, net_ids)
        if net_id is None or net_id not in net_by_id:
            continue
        parent, child = segment
        net = net_by_id[net_id]
        ratio = _segment_split_ratio(
            candidate,
            net.get("coordinates") or {},
            parent,
            child,
        )
        if ratio is None:
            continue
        synthetic_node_id = int(candidate.get("synthetic_node_id", next_synthetic_node_id))
        next_synthetic_node_id = min(next_synthetic_node_id, synthetic_node_id) - 1
        spec = {
            "net_id": int(net_id),
            "synthetic_node_id": synthetic_node_id,
            "parent_node_id": int(parent),
            "child_node_id": int(child),
            "split_ratio": float(ratio),
            "candidate_index": int(candidate_index),
        }
        split_specs_by_net.setdefault(int(net_id), []).append(spec)
        split_node_by_candidate_index[int(candidate_index)] = synthetic_node_id

    for net_id, specs in split_specs_by_net.items():
        split_specs_by_net[int(net_id)] = sorted(
            specs,
            key=lambda spec: (
                int(spec["parent_node_id"]),
                int(spec["child_node_id"]),
                float(spec["split_ratio"]),
                int(spec["candidate_index"]),
            ),
        )

    return split_specs_by_net, split_node_by_candidate_index


def _apply_segment_splits(children_by_node, edge_rc, node_cap, coordinates, split_specs):
    children_by_node = {int(node): [int(child) for child in children] for node, children in children_by_node.items()}
    edge_rc = dict(edge_rc or {})
    node_cap = dict(node_cap or {})
    coordinates = dict(coordinates or {})
    specs_by_edge = {}
    for spec in split_specs:
        edge_key = (int(spec["parent_node_id"]), int(spec["child_node_id"]))
        specs_by_edge.setdefault(edge_key, []).append(spec)
    for (parent, child), edge_specs in specs_by_edge.items():
        edge_specs = sorted(
            edge_specs,
            key=lambda spec: (
                float(spec["split_ratio"]),
                int(spec["candidate_index"]),
            ),
        )
        children = list(children_by_node.get(parent, []))
        if child not in children:
            raise ValueError(
                f"segment split edge ({parent}, {child}) is not in selected net topology"
            )
        child_index = children.index(child)
        split_nodes = [int(spec["synthetic_node_id"]) for spec in edge_specs]
        split_ratios = [float(spec["split_ratio"]) for spec in edge_specs]
        children[child_index] = split_nodes[0]
        children_by_node[parent] = children
        for index, synthetic in enumerate(split_nodes):
            next_node = split_nodes[index + 1] if index + 1 < len(split_nodes) else child
            children_by_node[synthetic] = [next_node]
        children_by_node.setdefault(child, [])
        original_edge = _edge_record(edge_rc, parent, child)
        previous_node = parent
        previous_ratio = 0.0
        for synthetic, ratio in zip(split_nodes, split_ratios):
            fraction = ratio - previous_ratio
            edge_rc[(previous_node, synthetic)] = {
                "r": original_edge["r"] * fraction,
                "c": original_edge["c"] * fraction,
            }
            previous_node = synthetic
            previous_ratio = ratio
        edge_rc[(previous_node, child)] = {
            "r": original_edge["r"] * (1.0 - previous_ratio),
            "c": original_edge["c"] * (1.0 - previous_ratio),
        }
        for synthetic, ratio in zip(split_nodes, split_ratios):
            node_cap[synthetic] = 0.0
            if parent in coordinates and child in coordinates:
                px, py = coordinates[parent]
                cx, cy = coordinates[child]
                coordinates[synthetic] = (
                    int(round(float(px) + (float(cx) - float(px)) * ratio)),
                    int(round(float(py) + (float(cy) - float(py)) * ratio)),
                )
    return children_by_node, edge_rc, node_cap, coordinates


def _as_tensor(values, *, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


def _as_index_tensor(values, *, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


def build_net_subgraph_timing_inputs(
    nets,
    *,
    candidates=None,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    default_driver_arrival=0.0,
    default_driver_slew=0.0,
    dtype=torch.float32,
    device=None,
    compact_runtime_metadata=False,
):
    """Build dense tensor ABI inputs for the native net-subgraph timing op.

    `nets` follows the records emitted by buffer_insertion.real_design_adapter.
    Original pin/Steiner node ids can be sparse; this builder compacts only the
    selected net subgraphs and returns metadata for mapping outputs back.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    candidates = list(candidates or [])
    driver_arrival_by_net = {
        int(net_id): float(value)
        for net_id, value in (driver_arrival_by_net or {}).items()
    }
    driver_slew_by_net = {
        int(net_id): float(value)
        for net_id, value in (driver_slew_by_net or {}).items()
    }
    split_specs_by_net, split_node_by_candidate_index = _build_segment_split_specs(
        nets,
        candidates,
    )

    compact_node_to_original = []
    compact_node_to_net_id = []
    original_node_to_compact = {}
    original_node_to_compact_by_net = {}
    compact_key_to_id = {}
    net_flat_topo_sort = []
    net_flat_topo_sort_start = [0]
    pin_fa = []
    children_by_compact = []
    edge_resistance = []
    edge_capacitance = []
    node_capacitance = []
    driver_arrival = []
    driver_slew = []
    sink_node_id = []
    sink_pin_id = []
    sink_net_index = []

    def compact(net_id, original_node_id):
        net_id = int(net_id)
        original_node_id = int(original_node_id)
        key = (net_id, original_node_id)
        if key in compact_key_to_id:
            return compact_key_to_id[key]
        compact_id = len(compact_node_to_original)
        compact_key_to_id[key] = compact_id
        original_node_to_compact_by_net.setdefault(net_id, {})[original_node_id] = compact_id
        original_node_to_compact.setdefault(original_node_id, compact_id)
        compact_node_to_original.append(original_node_id)
        compact_node_to_net_id.append(net_id)
        pin_fa.append(-1)
        children_by_compact.append([])
        edge_resistance.append(0.0)
        edge_capacitance.append(0.0)
        node_capacitance.append(0.0)
        return compact_id

    for net_index, net in enumerate(nets):
        net_id = _net_id(net, net_index)
        rc_tree = net.get("rc_tree", {})
        root_node_id = int(rc_tree.get("root_node_id", net.get("driver_pin_id", -1)))
        children_by_node = {
            int(node): [int(child) for child in children]
            for node, children in rc_tree.get("children_by_node", {}).items()
        }
        edge_rc = rc_tree.get("edge_rc", {})
        node_cap = {int(node): float(cap) for node, cap in rc_tree.get("node_cap", {}).items()}
        children_by_node, edge_rc, node_cap, _coordinates = _apply_segment_splits(
            children_by_node,
            edge_rc,
            node_cap,
            net.get("coordinates") or {},
            split_specs_by_net.get(net_id, []),
        )
        topo_original = _tree_preorder(root_node_id, children_by_node)
        if not topo_original:
            net_flat_topo_sort_start.append(len(net_flat_topo_sort))
            continue

        for original_node in topo_original:
            compact_id = compact(net_id, original_node)
            net_flat_topo_sort.append(compact_id)
            node_capacitance[compact_id] = float(node_cap.get(int(original_node), 0.0))
            for child_original in children_by_node.get(int(original_node), []):
                child_compact = compact(net_id, child_original)
                pin_fa[child_compact] = compact_id
                children_by_compact[compact_id].append(child_compact)
                edge_resistance[child_compact] = _edge_value(edge_rc, original_node, child_original, "r")
                edge_capacitance[child_compact] = _edge_value(edge_rc, original_node, child_original, "c")

        for sink_original in rc_tree.get("sink_nodes", []):
            sink_compact = compact(net_id, int(sink_original))
            sink_node_id.append(sink_compact)
            sink_pin_id.append(int(sink_original))
            sink_net_index.append(net_id)

        driver_arrival.append(driver_arrival_by_net.get(net_id, float(default_driver_arrival)))
        driver_slew.append(driver_slew_by_net.get(net_id, float(default_driver_slew)))
        net_flat_topo_sort_start.append(len(net_flat_topo_sort))

    flat_pin_to = []
    flat_pin_to_start = [0]
    for children in children_by_compact:
        flat_pin_to.extend(children)
        flat_pin_to_start.append(len(flat_pin_to))

    candidate_node_id = []
    candidate_net_ids = []
    candidate_ids = []
    candidate_bu = []
    buffer_input_cap = []
    buffer_delay = []
    buffer_output_slew = []
    for candidate_index, candidate in enumerate(candidates):
        if int(candidate_index) in split_node_by_candidate_index:
            original_node = int(split_node_by_candidate_index[int(candidate_index)])
        else:
            if "node_id" not in candidate:
                raise ValueError(
                    "candidate missing node_id and did not produce a segment split: "
                    f"candidate_index={candidate_index}, "
                    f"candidate_id={candidate.get('candidate_id')}, "
                    f"net_id={candidate.get('net_id', candidate.get('affected_net_id'))}, "
                    f"parent_node_id={candidate.get('parent_node_id')}, "
                    f"child_node_ids={candidate.get('child_node_ids')}, "
                    f"synthetic_node_id={candidate.get('synthetic_node_id')}, "
                    f"x_dbu={candidate.get('x_dbu')}, y_dbu={candidate.get('y_dbu')}"
                )
            original_node = int(candidate["node_id"])
        candidate_net_id = candidate.get("net_id", candidate.get("affected_net_id"))
        if candidate_net_id is not None:
            resolved_net_id = int(candidate_net_id)
            net_map = original_node_to_compact_by_net.get(int(candidate_net_id), {})
            if original_node not in net_map:
                raise ValueError(
                    f"candidate node_id {original_node} is not in selected net subgraphs"
                )
            compact_node_id = net_map[original_node]
        else:
            matching_net_ids = [
                int(net_id)
                for net_id, node_map in original_node_to_compact_by_net.items()
                if original_node in node_map
            ]
            if not matching_net_ids:
                raise ValueError(
                    f"candidate node_id {original_node} is not in selected net subgraphs"
                )
            if len(matching_net_ids) > 1:
                raise ValueError(
                    f"ambiguous candidate node_id {original_node}; candidate net_id is required"
                )
            resolved_net_id = int(matching_net_ids[0])
            compact_node_id = original_node_to_compact_by_net[resolved_net_id][original_node]
        candidate_node_id.append(compact_node_id)
        candidate_net_ids.append(resolved_net_id)
        candidate_ids.append(int(candidate.get("candidate_id", candidate_index)))
        candidate_bu.append(float(candidate.get("bu", candidate.get("b_u", 0.0))))
        buffer_input_cap.append(float(candidate.get("buffer_input_cap", 0.0)))
        buffer_delay.append(float(candidate.get("buffer_delay", 0.0)))
        buffer_output_slew.append(float(candidate.get("buffer_output_slew", 0.0)))

    metadata = {
        "compact_node_to_original": compact_node_to_original,
        "candidate_count": len(candidate_node_id),
        "net_ids": [_net_id(net, index) for index, net in enumerate(nets)],
    }
    if not compact_runtime_metadata:
        metadata.update(
            {
                "compact_node_to_net_id": compact_node_to_net_id,
                "original_node_to_compact": dict(original_node_to_compact),
                "original_node_to_compact_by_net": {
                    int(net_id): dict(node_map)
                    for net_id, node_map in original_node_to_compact_by_net.items()
                },
                "synthetic_segment_splits": [
                    dict(spec)
                    for net_id in sorted(split_specs_by_net)
                    for spec in split_specs_by_net[net_id]
                ],
                "candidate_ids": candidate_ids,
                "candidate_net_id": candidate_net_ids,
            }
        )

    result = {
        "net_flat_topo_sort": _as_index_tensor(net_flat_topo_sort, device=device),
        "net_flat_topo_sort_start": _as_index_tensor(net_flat_topo_sort_start, device=device),
        "pin_fa": _as_index_tensor(pin_fa, device=device),
        "flat_pin_to_start": _as_index_tensor(flat_pin_to_start, device=device),
        "flat_pin_to": _as_index_tensor(flat_pin_to, device=device),
        "edge_resistance": _as_tensor(edge_resistance, dtype=dtype, device=device),
        "node_capacitance": _as_tensor(node_capacitance, dtype=dtype, device=device),
        "edge_capacitance": _as_tensor(edge_capacitance, dtype=dtype, device=device),
        "driver_arrival": _as_tensor(driver_arrival, dtype=dtype, device=device),
        "driver_slew": _as_tensor(driver_slew, dtype=dtype, device=device),
        "candidate_node_id": _as_index_tensor(candidate_node_id, device=device),
        "candidate_net_id": _as_index_tensor(candidate_net_ids, device=device),
        "candidate_bu": _as_tensor(candidate_bu, dtype=dtype, device=device),
        "buffer_input_cap": _as_tensor(buffer_input_cap, dtype=dtype, device=device),
        "buffer_delay": _as_tensor(buffer_delay, dtype=dtype, device=device),
        "buffer_output_slew": _as_tensor(buffer_output_slew, dtype=dtype, device=device),
        "sink_node_id": _as_index_tensor(sink_node_id, device=device),
        "sink_net_index": _as_index_tensor(sink_net_index, device=device),
        "sink_pin_id": _as_index_tensor(sink_pin_id, device=device),
        "native_input_keys": NATIVE_INPUT_KEYS,
        "metadata": metadata,
    }
    return result
