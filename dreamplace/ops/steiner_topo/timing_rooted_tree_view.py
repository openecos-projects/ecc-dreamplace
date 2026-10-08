from collections import defaultdict, deque


def _sorted_unique(values):
    return sorted({int(value) for value in values})


def build_timing_rooted_tree_view(
    *,
    net_id,
    net_pin_ids,
    driver_pin_id,
    undirected_edges,
    coordinates=None,
    flat_first_pin_id=None,
):
    coordinates = dict(coordinates or {})
    net_pin_ids = _sorted_unique(net_pin_ids)
    driver_pin_id = int(driver_pin_id)

    adjacency = defaultdict(list)
    node_ids = set()
    for src, dst in undirected_edges or []:
        src = int(src)
        dst = int(dst)
        adjacency[src].append(dst)
        adjacency[dst].append(src)
        node_ids.add(src)
        node_ids.add(dst)
    node_ids.update(net_pin_ids)

    if driver_pin_id not in node_ids:
        return {
            "status": "unsupported",
            "unsupported_reasons": ["driver_not_in_tree"],
            "net_id": int(net_id),
            "root_node_id": driver_pin_id,
            "sink_pin_ids": [pin_id for pin_id in net_pin_ids if pin_id != driver_pin_id],
        }

    parent_by_node = {driver_pin_id: None}
    children_by_node = defaultdict(list)
    depth_by_node = {driver_pin_id: 0}
    queue = deque([driver_pin_id])
    visited = {driver_pin_id}
    has_cycle = False

    while queue:
        node_id = queue.popleft()
        for next_node in sorted(adjacency.get(node_id, [])):
            if parent_by_node.get(node_id) == next_node:
                continue
            if next_node in visited:
                has_cycle = True
                continue
            visited.add(next_node)
            parent_by_node[next_node] = node_id
            children_by_node[node_id].append(next_node)
            depth_by_node[next_node] = depth_by_node[node_id] + 1
            queue.append(next_node)

    for node_id in node_ids:
        children_by_node.setdefault(node_id, [])

    flat_first_pin_id = (
        int(flat_first_pin_id)
        if flat_first_pin_id is not None
        else (net_pin_ids[0] if net_pin_ids else None)
    )
    sink_pin_ids = [pin_id for pin_id in net_pin_ids if pin_id != driver_pin_id]
    is_connected = node_ids.issubset(visited)
    unsupported_reasons = []
    if not is_connected:
        unsupported_reasons.append("tree_not_connected")
    if has_cycle:
        unsupported_reasons.append("tree_has_cycle")

    return {
        "status": "ok" if not unsupported_reasons else "unsupported",
        "unsupported_reasons": unsupported_reasons,
        "net_id": int(net_id),
        "root_node_id": driver_pin_id,
        "flat_first_pin_id": flat_first_pin_id,
        "driver_is_flat_first_pin": flat_first_pin_id == driver_pin_id,
        "reroot_applied": flat_first_pin_id != driver_pin_id,
        "sink_pin_ids": sink_pin_ids,
        "node_ids": sorted(node_ids),
        "edge_pairs": [(int(src), int(dst)) for src, dst in undirected_edges or []],
        "parent_by_node": parent_by_node,
        "children_by_node": {node_id: sorted(children) for node_id, children in children_by_node.items()},
        "depth_by_node": depth_by_node,
        "coordinates": {int(node_id): tuple(value) for node_id, value in coordinates.items()},
        "is_connected": is_connected,
        "has_cycle": has_cycle,
    }
