def _topo_order(root, children_by_node):
    order = []
    stack = [int(root)]
    while stack:
        node = stack.pop()
        order.append(node)
        for child in reversed(children_by_node.get(node, [])):
            stack.append(int(child))
    return order


def _buffer_entry(buffer_library, bsu):
    if buffer_library is None or int(bsu) not in buffer_library:
        raise ValueError(f"missing buffer library entry for bsu={bsu}")
    entry = buffer_library[int(bsu)]
    if "input_cap" not in entry or "delay" not in entry:
        raise ValueError(f"incomplete buffer library entry for bsu={bsu}")
    return entry


def compute_buffer_aware_rc_forward(
    tree,
    *,
    mode="original",
    virtual_buffers=None,
    buffer_library=None,
):
    if mode not in {"original", "buffer_forward", "buffer_gradient"}:
        raise ValueError(f"unsupported rc forward mode: {mode}")
    root = int(tree["root_node_id"])
    children_by_node = {
        int(node): [int(child) for child in children]
        for node, children in (tree.get("children_by_node") or {}).items()
    }
    edge_rc = {
        (int(src), int(dst)): dict(value)
        for (src, dst), value in (tree.get("edge_rc") or {}).items()
    }
    node_cap = {int(node): float(cap) for node, cap in (tree.get("node_cap") or {}).items()}
    effective_node_cap = dict(node_cap)
    for (src, dst), value in edge_rc.items():
        edge_cap = float(value.get("c", 0.0) or 0.0)
        if edge_cap <= 0.0:
            continue
        half_cap = 0.5 * edge_cap
        effective_node_cap[src] = float(effective_node_cap.get(src, 0.0)) + half_cap
        effective_node_cap[dst] = float(effective_node_cap.get(dst, 0.0)) + half_cap
    virtual_buffers = {int(node): dict(value) for node, value in (virtual_buffers or {}).items()}

    order = _topo_order(root, children_by_node)
    lout_by_node = {}
    lin_by_node = {}
    for node in reversed(order):
        lout = effective_node_cap.get(node, 0.0)
        for child in children_by_node.get(node, []):
            lout += lin_by_node[child]
        lout_by_node[node] = lout

        state = virtual_buffers.get(node, {})
        bu = int(state.get("bu", 0)) if mode != "original" else 0
        if bu:
            bsu = int(state.get("bsu", -1))
            lin_by_node[node] = float(_buffer_entry(buffer_library, bsu)["input_cap"])
        else:
            lin_by_node[node] = lout

    arrival_by_node = {root: 0.0}
    for node in order:
        input_arrival = arrival_by_node[node]
        state = virtual_buffers.get(node, {})
        bu = int(state.get("bu", 0)) if mode != "original" else 0
        output_arrival = input_arrival
        if bu:
            bsu = int(state.get("bsu", -1))
            output_arrival += float(_buffer_entry(buffer_library, bsu)["delay"])
            arrival_by_node[node] = output_arrival
        for child in children_by_node.get(node, []):
            resistance = float(edge_rc.get((node, child), {}).get("r", 0.0))
            arrival_by_node[child] = output_arrival + resistance * lin_by_node[child]

    return {
        "mode": mode,
        "effective_node_cap": effective_node_cap,
        "lout_by_node": lout_by_node,
        "lin_by_node": lin_by_node,
        "arrival_by_node": arrival_by_node,
    }
