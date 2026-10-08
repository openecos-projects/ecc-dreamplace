import math

import torch


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


def _as_tensor(value, *, dtype, device):
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _mapping_value(mapping, key, default, *, dtype, device):
    if mapping is None:
        return _as_tensor(default, dtype=dtype, device=device)
    value = mapping.get(int(key), default)
    return _as_tensor(value, dtype=dtype, device=device)


def _validate_size_table(name, table, *, bsu_index):
    table = torch.as_tensor(table, dtype=bsu_index.dtype, device=bsu_index.device)
    if table.ndim != 2:
        raise ValueError(f"{name} must have shape [segment_count, legal_buffer_count]")
    segment_count = bsu_index.numel()
    if int(table.shape[0]) != int(segment_count):
        raise ValueError(f"{name} segment dimension does not match segment state")
    if int(table.shape[1]) < 2:
        raise ValueError(f"{name} must contain at least two legal buffer sizes")
    return table


def _interpolate_size_tables(
    bsu_index,
    *,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
):
    bsu_index = torch.as_tensor(bsu_index)
    if bsu_index.ndim != 1:
        raise ValueError("bsu_index must have shape [segment_count]")
    per_size_input_cap = _validate_size_table(
        "per_size_input_cap",
        per_size_input_cap,
        bsu_index=bsu_index,
    )
    per_size_delay = _validate_size_table(
        "per_size_delay",
        per_size_delay,
        bsu_index=bsu_index,
    )
    per_size_output_slew = _validate_size_table(
        "per_size_output_slew",
        per_size_output_slew,
        bsu_index=bsu_index,
    )
    legal_buffer_count = int(per_size_delay.shape[1])
    clipped = torch.clamp(bsu_index, 0.0, float(legal_buffer_count - 1))
    lo_index = torch.floor(clipped).to(dtype=torch.long)
    hi_index = torch.clamp(lo_index + 1, max=legal_buffer_count - 1)
    alpha = clipped - lo_index.to(dtype=clipped.dtype)

    def gather(table):
        lo = torch.gather(table, 1, lo_index.view(-1, 1)).view(-1)
        hi = torch.gather(table, 1, hi_index.view(-1, 1)).view(-1)
        return (1.0 - alpha) * lo + alpha * hi

    return {
        "buffer_input_cap": gather(per_size_input_cap),
        "buffer_delay": gather(per_size_delay),
        "buffer_output_slew": gather(per_size_output_slew),
        "bsu_index": clipped,
    }


def _segment_lookup(segment_state):
    return {
        (
            int(row["net_id"]),
            int(row["parent_node_id"]),
            int(row["child_node_id"]),
        ): int(row["segment_id"])
        for row in segment_state.segment_rows
    }


def _interpolate_count_states(states, z_value, max_repeater_count):
    stacked = torch.stack(states)
    clipped = torch.clamp(z_value, 0.0, float(max_repeater_count))
    lo = torch.floor(clipped).to(dtype=torch.long)
    hi = torch.clamp(lo + 1, max=int(max_repeater_count))
    alpha = clipped - lo.to(dtype=clipped.dtype)
    return (1.0 - alpha) * stacked[lo] + alpha * stacked[hi]


def _edge_endpoint_half_cap(edge_cap, z_value, max_repeater_count, *, dtype, device):
    edge_cap = torch.as_tensor(edge_cap, dtype=dtype, device=device)
    states = [
        0.5 * edge_cap / float(k + 1)
        for k in range(int(max_repeater_count) + 1)
    ]
    return _interpolate_count_states(states, z_value, max_repeater_count)


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


def _edge_endpoint_half_caps(
    edge_cap,
    segment_row,
    z_value,
    max_repeater_count,
    *,
    dtype,
    device,
):
    edge_cap = torch.as_tensor(edge_cap, dtype=dtype, device=device)
    parent_states = []
    child_states = []
    for k in range(int(max_repeater_count) + 1):
        fractions = _split_fractions(segment_row, k)
        parent_states.append(0.5 * edge_cap * float(fractions[0]))
        child_states.append(0.5 * edge_cap * float(fractions[-1]))
    return (
        _interpolate_count_states(parent_states, z_value, max_repeater_count),
        _interpolate_count_states(child_states, z_value, max_repeater_count),
    )


def _edge_input_load_states(lin_child, buffer_input_cap, max_repeater_count):
    states = [lin_child]
    for _ in range(int(max_repeater_count)):
        states.append(buffer_input_cap)
    return states


def _edge_delay_slew_state(
    *,
    repeater_count,
    edge_resistance,
    lin_child,
    parent_slew,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    segment_row=None,
):
    dtype = lin_child.dtype
    device = lin_child.device
    edge_resistance = torch.as_tensor(
        edge_resistance,
        dtype=dtype,
        device=device,
    )
    count = int(repeater_count)
    fractions = (
        _split_fractions(segment_row, count)
        if segment_row is not None
        else [1.0 / float(count + 1) for _ in range(count + 1)]
    )
    log10 = torch.as_tensor(math.log(10.0), dtype=dtype, device=device)
    total_delay = torch.zeros((), dtype=dtype, device=device)
    slew = parent_slew
    for index in range(count):
        sub_resistance = edge_resistance * float(fractions[index])
        wire_delta = sub_resistance * buffer_input_cap
        total_delay = total_delay + wire_delta + buffer_delay
        slew = torch.sqrt(slew * slew + (log10 * wire_delta) * (log10 * wire_delta))
        slew = buffer_output_slew
    sub_resistance = edge_resistance * float(fractions[-1])
    wire_delta = sub_resistance * lin_child
    total_delay = total_delay + wire_delta
    slew = torch.sqrt(slew * slew + (log10 * wire_delta) * (log10 * wire_delta))
    return total_delay, slew


def _edge_delay_slew_states(
    *,
    edge_resistance,
    lin_child,
    parent_slew,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    max_repeater_count,
    segment_row=None,
):
    return [
        _edge_delay_slew_state(
            repeater_count=k,
            edge_resistance=edge_resistance,
            lin_child=lin_child,
            parent_slew=parent_slew,
            buffer_input_cap=buffer_input_cap,
            buffer_delay=buffer_delay,
            buffer_output_slew=buffer_output_slew,
            segment_row=segment_row,
        )
        for k in range(int(max_repeater_count) + 1)
    ]


def _empty_result(dtype, device):
    empty = torch.empty(0, dtype=dtype, device=device)
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    return {
        "segment_delay": empty,
        "segment_output_slew": empty,
        "segment_upstream_visible_input_cap": empty,
        "sink_arrival": empty,
        "sink_slew": empty,
        "sink_load": empty,
        "sink_node_id": empty_long,
        "sink_net_index": empty_long,
        "metadata": {
            "net_count": 0,
            "segment_count_state_count": 0,
            "max_repeater_count": 0,
            "segment_shared_bsu": True,
        },
    }


def segment_count_relaxed_timing(
    nets,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    default_driver_arrival=0.0,
    default_driver_slew=0.0,
):
    per_size_input_cap = torch.as_tensor(per_size_input_cap)
    dtype = per_size_input_cap.dtype
    device = per_size_input_cap.device
    per_size_delay = torch.as_tensor(per_size_delay, dtype=dtype, device=device)
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=dtype,
        device=device,
    )
    segment_count = int(segment_state.z_param.numel())
    if segment_count == 0:
        return _empty_result(dtype, device)

    buffer_tensors = _interpolate_size_tables(
        segment_state.bsu_index().to(dtype=dtype, device=device),
        per_size_input_cap=per_size_input_cap,
        per_size_delay=per_size_delay,
        per_size_output_slew=per_size_output_slew,
    )
    z_value = segment_state.z_value().to(dtype=dtype, device=device)
    buffer_input_cap = buffer_tensors["buffer_input_cap"]
    buffer_delay = buffer_tensors["buffer_delay"]
    buffer_output_slew = buffer_tensors["buffer_output_slew"]
    max_repeater_count = int(segment_state.max_repeater_count)
    segment_by_edge = (segment_state.summary or {}).get("segment_by_edge")
    if segment_by_edge is None:
        segment_by_edge = _segment_lookup(segment_state)
    segment_delay = [
        torch.zeros((), dtype=dtype, device=device)
        for _ in range(segment_count)
    ]
    segment_output_slew = [
        torch.zeros((), dtype=dtype, device=device)
        for _ in range(segment_count)
    ]
    segment_upstream_cap = [
        torch.zeros((), dtype=dtype, device=device)
        for _ in range(segment_count)
    ]
    sink_arrival = []
    sink_slew = []
    sink_load = []
    sink_node_id = []
    sink_net_index = []

    for net_index, net in enumerate(list(nets or [])):
        prepared = net.get("_segment_count_prepared")
        if prepared is None:
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
            edge_rc = rc_tree.get("edge_rc", {}) or {}
            raw_node_cap = {
                int(node): cap
                for node, cap in (rc_tree.get("node_cap", {}) or {}).items()
            }
            topo = _tree_preorder(root, children_by_node)
            sink_nodes = [int(sink) for sink in rc_tree.get("sink_nodes", [])]
        else:
            net_id = int(prepared["net_id"])
            root = prepared.get("root_node_id")
            if root is None:
                continue
            root = int(root)
            children_by_node = prepared["children_by_node"]
            edge_rc = prepared["edge_rc"]
            raw_node_cap = prepared["node_cap"]
            topo = prepared["topo"]
            sink_nodes = prepared["sink_nodes"]
        if not topo:
            continue
        node_cap = {
            int(node): _as_tensor(cap, dtype=dtype, device=device)
            for node, cap in raw_node_cap.items()
        }

        effective_node_cap = {
            int(node): node_cap.get(int(node), torch.zeros((), dtype=dtype, device=device))
            for node in topo
        }
        for parent in topo:
            for child in children_by_node.get(int(parent), []):
                edge_cap = _edge_value(edge_rc, parent, child, "c")
                seg_idx = segment_by_edge.get((net_id, int(parent), int(child)))
                if seg_idx is None:
                    half_cap = torch.as_tensor(0.5 * edge_cap, dtype=dtype, device=device)
                    parent_half_cap = half_cap
                    child_half_cap = half_cap
                else:
                    parent_half_cap = child_half_cap = _edge_endpoint_half_cap(
                        edge_cap,
                        z_value[seg_idx],
                        max_repeater_count,
                        dtype=dtype,
                        device=device,
                    )
                effective_node_cap[int(parent)] = effective_node_cap.get(
                    int(parent),
                    torch.zeros((), dtype=dtype, device=device),
                ) + parent_half_cap
                effective_node_cap[int(child)] = effective_node_cap.get(
                    int(child),
                    torch.zeros((), dtype=dtype, device=device),
                ) + child_half_cap

        lin = {}
        lout = {}
        for node in reversed(topo):
            children_load = torch.zeros((), dtype=dtype, device=device)
            for child in children_by_node.get(int(node), []):
                seg_idx = segment_by_edge.get((net_id, int(node), int(child)))
                if seg_idx is None:
                    edge_input = lin[int(child)]
                else:
                    states = _edge_input_load_states(
                        lin[int(child)],
                        buffer_input_cap[seg_idx],
                        max_repeater_count,
                    )
                    edge_input = _interpolate_count_states(
                        states,
                        z_value[seg_idx],
                        max_repeater_count,
                    )
                    segment_upstream_cap[seg_idx] = edge_input
                children_load = children_load + edge_input
            node_lout = effective_node_cap.get(
                int(node),
                torch.zeros((), dtype=dtype, device=device),
            ) + children_load
            lout[int(node)] = node_lout
            lin[int(node)] = node_lout

        arrival_in = {}
        arrival_out = {}
        slew_in = {}
        slew_out = {}
        arrival_in[root] = _mapping_value(
            driver_arrival_by_net,
            net_id,
            default_driver_arrival,
            dtype=dtype,
            device=device,
        )
        arrival_out[root] = arrival_in[root]
        slew_in[root] = _mapping_value(
            driver_slew_by_net,
            net_id,
            default_driver_slew,
            dtype=dtype,
            device=device,
        )
        slew_out[root] = slew_in[root]

        for parent in topo:
            for child in children_by_node.get(int(parent), []):
                edge_resistance = _edge_value(edge_rc, parent, child, "r")
                seg_idx = segment_by_edge.get((net_id, int(parent), int(child)))
                if seg_idx is None:
                    delay_state, child_slew = _edge_delay_slew_state(
                        repeater_count=0,
                        edge_resistance=edge_resistance,
                        lin_child=lin[int(child)],
                        parent_slew=slew_out[int(parent)],
                        buffer_input_cap=torch.zeros((), dtype=dtype, device=device),
                        buffer_delay=torch.zeros((), dtype=dtype, device=device),
                        buffer_output_slew=torch.zeros((), dtype=dtype, device=device),
                        segment_row=None,
                    )
                    edge_delay = delay_state
                else:
                    states = _edge_delay_slew_states(
                        edge_resistance=edge_resistance,
                        lin_child=lin[int(child)],
                        parent_slew=slew_out[int(parent)],
                        buffer_input_cap=buffer_input_cap[seg_idx],
                        buffer_delay=buffer_delay[seg_idx],
                        buffer_output_slew=buffer_output_slew[seg_idx],
                        max_repeater_count=max_repeater_count,
                        segment_row=None,
                    )
                    edge_delay = _interpolate_count_states(
                        [state[0] for state in states],
                        z_value[seg_idx],
                        max_repeater_count,
                    )
                    child_slew = _interpolate_count_states(
                        [state[1] for state in states],
                        z_value[seg_idx],
                        max_repeater_count,
                    )
                    segment_delay[seg_idx] = edge_delay
                    segment_output_slew[seg_idx] = child_slew
                arrival_in[int(child)] = arrival_out[int(parent)] + edge_delay
                arrival_out[int(child)] = arrival_in[int(child)]
                slew_in[int(child)] = child_slew
                slew_out[int(child)] = child_slew

        for sink in sink_nodes:
            sink = int(sink)
            if sink not in arrival_out:
                continue
            sink_arrival.append(arrival_out[sink])
            sink_slew.append(slew_out[sink])
            sink_load.append(lin[sink])
            sink_node_id.append(sink)
            sink_net_index.append(net_id)

    empty = torch.empty(0, dtype=dtype, device=device)
    return {
        "segment_delay": torch.stack(segment_delay) if segment_delay else empty,
        "segment_output_slew": torch.stack(segment_output_slew) if segment_output_slew else empty,
        "segment_upstream_visible_input_cap": torch.stack(segment_upstream_cap) if segment_upstream_cap else empty,
        "sink_arrival": torch.stack(sink_arrival) if sink_arrival else empty,
        "sink_slew": torch.stack(sink_slew) if sink_slew else empty,
        "sink_load": torch.stack(sink_load) if sink_load else empty,
        "sink_node_id": torch.tensor(sink_node_id, dtype=torch.long, device=device),
        "sink_net_index": torch.tensor(sink_net_index, dtype=torch.long, device=device),
        "relaxed_buffer_model": {
            "z_value": z_value,
            "bsu_index": buffer_tensors["bsu_index"],
            "buffer_input_cap": buffer_input_cap,
            "buffer_delay": buffer_delay,
            "buffer_output_slew": buffer_output_slew,
        },
        "metadata": {
            "net_count": len(list(nets or [])),
            "segment_count_state_count": int(segment_count),
            "max_repeater_count": int(max_repeater_count),
            "segment_shared_bsu": True,
        },
    }
