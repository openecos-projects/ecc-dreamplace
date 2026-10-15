import importlib
import time

import torch


def _cpu_contiguous(value, *, dtype=None):
    tensor = torch.as_tensor(value).detach()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device="cpu").contiguous()


def _cpu_floating_contiguous(value):
    tensor = torch.as_tensor(value).detach()
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device="cpu").contiguous()


def pack_segment_count_topology(
    *,
    flat_net2pin,
    flat_net2pin_start,
    net2driver,
    pin2node,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    node_x,
    node_y,
    node_x_dbu,
    node_y_dbu,
    pin_capacitance,
    num_movable_nodes,
    num_terminals,
    dbu,
    scale_factor,
    r_unit,
    c_unit,
    max_repeater_count,
    num_threads=None,
):
    """Pack live Route-B topology without constructing Python net/segment rows."""

    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    export_started_at = time.perf_counter()
    pin_capacitance = _cpu_contiguous(pin_capacitance)
    if pin_capacitance.dtype not in (torch.float32, torch.float64):
        pin_capacitance = pin_capacitance.to(dtype=torch.float32)
    native_inputs = (
        _cpu_contiguous(flat_net2pin),
        _cpu_contiguous(flat_net2pin_start),
        _cpu_contiguous(net2driver),
        _cpu_contiguous(pin2node),
        _cpu_contiguous(net_flat_topo_sort),
        _cpu_contiguous(net_flat_topo_sort_start),
        _cpu_contiguous(pin_fa),
        _cpu_floating_contiguous(node_x),
        _cpu_floating_contiguous(node_y),
        _cpu_floating_contiguous(node_x_dbu),
        _cpu_floating_contiguous(node_y_dbu),
        pin_capacitance,
    )
    native_input_export_ms = (time.perf_counter() - export_started_at) * 1000.0
    thread_count = torch.get_num_threads() if num_threads is None else int(num_threads)
    call_started_at = time.perf_counter()
    result = native.pack_segment_count_topology(
        *native_inputs,
        int(num_movable_nodes),
        int(num_terminals),
        float(dbu),
        float(scale_factor),
        float(r_unit),
        float(c_unit),
        int(max_repeater_count),
        thread_count,
    )
    native_pybind_call_ms = (time.perf_counter() - call_started_at) * 1000.0
    metadata = result["metadata"]
    native_cpp_accounted_ms = sum(
        float(metadata.get(name, 0.0) or 0.0)
        for name in (
            "validate_count_ms",
            "prefix_sum_ms",
            "allocate_ms",
            "fill_ms",
        )
    )
    metadata["native_input_export_ms"] = float(native_input_export_ms)
    metadata["native_pybind_call_ms"] = float(native_pybind_call_ms)
    metadata["native_return_attach_ms"] = max(
        0.0,
        float(native_pybind_call_ms) - native_cpp_accounted_ms,
    )
    return result


def select_packed_segment_count_inputs(
    prepared,
    geometry,
    *,
    active_net_ids,
    row_indices,
    dtype,
    device,
):
    """Select a deterministic active-net view from full native packed tensors."""

    if geometry is None:
        # Python-built prepared trees have the same CSR contract, without a
        # separate native geometry pack. Derive only its segment partition.
        net_ids = prepared["net_ids"]
        counts = torch.zeros_like(net_ids, dtype=torch.long)
        counts.scatter_add_(
            0, prepared["edge_net_index"].long(),
            (prepared["edge_to_segment_id"] >= 0).long(),
        )
        geometry = {
            "segment_ids": torch.arange(prepared["parent_cap_fraction"].shape[0]),
            "net_ids": net_ids,
            "net_segment_start": torch.cat((counts.new_zeros(1), counts.cumsum(0))),
        }
    prepared_net_ids = prepared["net_ids"].to(device="cpu", dtype=torch.long)
    query = torch.as_tensor(
        sorted(set(int(value) for value in active_net_ids)),
        dtype=torch.long,
    )
    positions = torch.searchsorted(prepared_net_ids, query)
    valid = positions < int(prepared_net_ids.numel())
    matched_positions = positions[valid]
    matched_query = query[valid]
    if int(matched_positions.numel()) > 0:
        matches = prepared_net_ids.index_select(0, matched_positions) == matched_query
        matched_positions = matched_positions[matches]
        matched_query = matched_query[matches]

    row_index = torch.as_tensor(row_indices, dtype=torch.long)
    full_net_view = (
        int(matched_positions.numel()) == int(prepared_net_ids.numel())
        and torch.equal(
            matched_positions,
            torch.arange(int(prepared_net_ids.numel()), dtype=torch.long),
        )
    )
    full_segment_view = (
        int(row_index.numel()) == int(geometry["segment_ids"].numel())
        and torch.equal(
            row_index,
            torch.arange(int(geometry["segment_ids"].numel()), dtype=torch.long),
        )
    )

    integer_fields = {
        "net_ids",
        "net_topo_start",
        "net_edge_start",
        "net_sink_start",
        "flat_topo_node_id",
        "edge_start",
        "edge_parent_node_id",
        "edge_child_node_id",
        "edge_parent_compact_id",
        "edge_child_compact_id",
        "edge_net_index",
        "edge_to_segment_id",
        "source_prepared_edge_index",
        "sink_node_id",
        "sink_net_id",
        "sink_node_compact_id",
        "driver_pin_id",
    }
    value_fields = {
        "edge_resistance",
        "edge_capacitance",
        "node_capacitance",
        "parent_cap_fraction",
        "child_cap_fraction",
        "segment_sub_resistance_fraction",
    }

    def to_runtime(name, tensor):
        target_dtype = torch.long if name in integer_fields else dtype
        return tensor.to(device=device, dtype=target_dtype).contiguous()

    if full_net_view and full_segment_view:
        result = {
            name: to_runtime(name, tensor)
            for name, tensor in prepared.items()
            if name in integer_fields or name in value_fields
        }
        result["metadata"] = {
            "topology_source": "native_packed_full_view",
            "backend_input": "segment_count_prepared_tensor",
            "segment_count_state_count": int(row_index.numel()),
        }
        result["source_prepared_edge_index"] = torch.arange(
            int(prepared["edge_resistance"].numel()),
            dtype=torch.long,
            device=device,
        )
        return result

    net_topo_start = prepared["net_topo_start"].to(device="cpu", dtype=torch.long)
    net_edge_start = prepared["net_edge_start"].to(device="cpu", dtype=torch.long)
    net_sink_start = prepared["net_sink_start"].to(device="cpu", dtype=torch.long)
    node_ranges = [
        (int(net_topo_start[index]), int(net_topo_start[index + 1]))
        for index in matched_positions.tolist()
    ]
    edge_ranges = [
        (int(net_edge_start[index]), int(net_edge_start[index + 1]))
        for index in matched_positions.tolist()
    ]
    sink_ranges = [
        (int(net_sink_start[index]), int(net_sink_start[index + 1]))
        for index in matched_positions.tolist()
    ]

    def cat_ranges(tensor, ranges):
        pieces = [tensor[start:end] for start, end in ranges if end > start]
        if not pieces:
            return tensor[:0]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    node_counts = [end - start for start, end in node_ranges]
    edge_counts = [end - start for start, end in edge_ranges]
    sink_counts = [end - start for start, end in sink_ranges]
    local_node_start = [0]
    local_edge_start = [0]
    local_sink_start = [0]
    for count in node_counts:
        local_node_start.append(local_node_start[-1] + count)
    for count in edge_counts:
        local_edge_start.append(local_edge_start[-1] + count)
    for count in sink_counts:
        local_sink_start.append(local_sink_start[-1] + count)

    edge_start_parts = []
    parent_compact_parts = []
    child_compact_parts = []
    sink_compact_parts = []
    edge_net_parts = []
    for local_net, ((node_begin, node_end), (edge_begin, edge_end), (sink_begin, sink_end)) in enumerate(
        zip(node_ranges, edge_ranges, sink_ranges)
    ):
        edge_start_parts.append(
            prepared["edge_start"][node_begin:node_end]
            - edge_begin
            + local_edge_start[local_net]
        )
        parent_compact_parts.append(
            prepared["edge_parent_compact_id"][edge_begin:edge_end]
            - node_begin
            + local_node_start[local_net]
        )
        child_compact_parts.append(
            prepared["edge_child_compact_id"][edge_begin:edge_end]
            - node_begin
            + local_node_start[local_net]
        )
        sink_compact_parts.append(
            prepared["sink_node_compact_id"][sink_begin:sink_end]
            - node_begin
            + local_node_start[local_net]
        )
        edge_net_parts.append(
            torch.full((edge_end - edge_begin,), local_net, dtype=torch.long)
        )

    affected_net_ids = geometry["net_ids"].to(device="cpu", dtype=torch.long)
    affected_positions = torch.searchsorted(affected_net_ids, matched_query)
    segment_starts = geometry["net_segment_start"].to(device="cpu", dtype=torch.long)
    local_segment_offset = 0
    edge_to_segment_parts = []
    for edge_range, affected_position in zip(edge_ranges, affected_positions.tolist()):
        edge_begin, edge_end = edge_range
        global_segment_begin = int(segment_starts[affected_position])
        global_segment_end = int(segment_starts[affected_position + 1])
        values = prepared["edge_to_segment_id"][edge_begin:edge_end].clone()
        mask = values >= 0
        values[mask] = values[mask] - global_segment_begin + local_segment_offset
        edge_to_segment_parts.append(values)
        local_segment_offset += global_segment_end - global_segment_begin

    def cat_or_empty(parts, template):
        nonempty = [part for part in parts if int(part.numel()) > 0]
        if not nonempty:
            return template[:0]
        return nonempty[0] if len(nonempty) == 1 else torch.cat(nonempty, dim=0)

    source_prepared_edge_index = cat_or_empty(
        [torch.arange(start, end, dtype=torch.long) for start, end in edge_ranges],
        prepared["edge_parent_node_id"],
    )

    result = {
        "net_ids": matched_query,
        "net_topo_start": torch.as_tensor(local_node_start, dtype=torch.long),
        "net_edge_start": torch.as_tensor(local_edge_start, dtype=torch.long),
        "net_sink_start": torch.as_tensor(local_sink_start, dtype=torch.long),
        "flat_topo_node_id": cat_ranges(prepared["flat_topo_node_id"], node_ranges),
        "edge_start": torch.cat(
            [
                cat_or_empty(edge_start_parts, prepared["edge_start"]),
                torch.as_tensor([local_edge_start[-1]], dtype=torch.long),
            ]
        ),
        "edge_parent_node_id": cat_ranges(prepared["edge_parent_node_id"], edge_ranges),
        "edge_child_node_id": cat_ranges(prepared["edge_child_node_id"], edge_ranges),
        "edge_parent_compact_id": cat_or_empty(
            parent_compact_parts,
            prepared["edge_parent_compact_id"],
        ),
        "edge_child_compact_id": cat_or_empty(
            child_compact_parts,
            prepared["edge_child_compact_id"],
        ),
        "edge_net_index": cat_or_empty(edge_net_parts, prepared["edge_net_index"]),
        "edge_resistance": cat_ranges(prepared["edge_resistance"], edge_ranges),
        "edge_capacitance": cat_ranges(prepared["edge_capacitance"], edge_ranges),
        "node_capacitance": cat_ranges(prepared["node_capacitance"], node_ranges),
        "edge_to_segment_id": cat_or_empty(
            edge_to_segment_parts,
            prepared["edge_to_segment_id"],
        ),
        "source_prepared_edge_index": source_prepared_edge_index,
        "sink_node_id": cat_ranges(prepared["sink_node_id"], sink_ranges),
        "sink_net_id": cat_ranges(prepared["sink_net_id"], sink_ranges),
        "sink_node_compact_id": cat_or_empty(
            sink_compact_parts,
            prepared["sink_node_compact_id"],
        ),
        "driver_pin_id": prepared["driver_pin_id"].index_select(0, matched_positions),
        "parent_cap_fraction": prepared["parent_cap_fraction"].index_select(0, row_index),
        "child_cap_fraction": prepared["child_cap_fraction"].index_select(0, row_index),
        "segment_sub_resistance_fraction": prepared[
            "segment_sub_resistance_fraction"
        ].index_select(0, row_index),
    }
    result = {name: to_runtime(name, tensor) for name, tensor in result.items()}
    result["metadata"] = {
        "topology_source": "native_packed_active_view",
        "backend_input": "segment_count_prepared_tensor",
        "segment_count_state_count": int(row_index.numel()),
    }
    return result


def select_packed_segment_count_inputs_native(
    prepared,
    geometry,
    *,
    active_net_ids,
    dtype,
    device,
    num_threads=None,
):
    """Select one Route-B timing level through the native packed CSR selector."""

    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    if torch.is_tensor(active_net_ids):
        active_net_ids = active_net_ids.detach().to(
            device="cpu", dtype=torch.long
        ).contiguous()
    else:
        active_net_ids = torch.as_tensor(
            list(active_net_ids or ()), dtype=torch.long
        ).contiguous()
    thread_count = torch.get_num_threads() if num_threads is None else int(num_threads)
    call_started_at = time.perf_counter()
    selected = native.select_packed_segment_count_inputs(
        prepared,
        geometry,
        active_net_ids,
        thread_count,
    )
    native_pybind_call_ms = (time.perf_counter() - call_started_at) * 1000.0

    integer_fields = {
        "net_ids",
        "net_topo_start",
        "net_edge_start",
        "net_sink_start",
        "flat_topo_node_id",
        "edge_start",
        "edge_parent_node_id",
        "edge_child_node_id",
        "edge_parent_compact_id",
        "edge_child_compact_id",
        "edge_net_index",
        "edge_to_segment_id",
        "source_prepared_edge_index",
        "sink_node_id",
        "sink_net_id",
        "sink_node_compact_id",
        "driver_pin_id",
    }
    prepared_inputs = {
        name: tensor.to(
            device=device,
            dtype=torch.long if name in integer_fields else dtype,
        ).contiguous()
        for name, tensor in selected["prepared_inputs"].items()
    }
    metadata = dict(selected["metadata"])
    native_cpp_accounted_ms = sum(
        float(metadata.get(name, 0.0) or 0.0)
        for name in (
            "native_active_view_match_count_ms",
            "native_active_view_prefix_sum_ms",
            "native_active_view_allocate_ms",
            "native_active_view_fill_ms",
        )
    )
    metadata.update(
        {
            "backend_input": "segment_count_prepared_tensor",
            "native_pybind_call_ms": float(native_pybind_call_ms),
            "native_return_attach_ms": max(
                0.0, float(native_pybind_call_ms) - native_cpp_accounted_ms
            ),
            "segment_count_state_count": int(
                selected["source_segment_row_index"].numel()
            ),
        }
    )
    prepared_inputs["metadata"] = metadata
    prepared_inputs["source_prepared_net_positions"] = selected[
        "source_prepared_net_positions"
    ].to(device=device, dtype=torch.long).contiguous()
    prepared_inputs["source_prepared_edge_index"] = selected[
        "source_prepared_edge_index"
    ].to(device=device, dtype=torch.long).contiguous()
    prepared_inputs["source_segment_row_index"] = selected[
        "source_segment_row_index"
    ].to(device=device, dtype=torch.long).contiguous()
    return prepared_inputs
