import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    batched_segment_transfer,
    segment_transfer_forward_native,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    _edge_delay_slew_state,
    _edge_delay_slew_states,
    _edge_endpoint_half_cap,
    _edge_endpoint_half_caps,
    _edge_input_load_states,
    _interpolate_count_states,
    _interpolate_size_tables,
    _split_fractions,
)


def _as_device_tensor(value, *, dtype, device):
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _buffer_device_tensors(buffer_device_lut, *, dtype, device):
    if buffer_device_lut is None:
        raise ValueError("buffer_device_lut is required for segment transfer backend")
    if hasattr(buffer_device_lut, "tensors"):
        tables = buffer_device_lut.tensors(dtype=dtype, device=device)
        return {
            "buffer_input_cap_by_size": tables["input_cap_by_size"],
            "buffer_slew_axis": tables["input_slew_axis"],
            "buffer_load_axis": tables["output_load_axis"],
            "buffer_delay_lut": tables["delay_lut"],
            "buffer_output_slew_lut": tables["output_slew_lut"],
        }
    required = (
        "buffer_input_cap_by_size",
        "buffer_slew_axis",
        "buffer_load_axis",
        "buffer_delay_lut",
        "buffer_output_slew_lut",
    )
    missing = [key for key in required if key not in buffer_device_lut]
    if missing:
        raise ValueError(f"buffer_device_lut is missing keys: {missing}")
    return {
        key: _as_device_tensor(buffer_device_lut[key], dtype=dtype, device=device)
        for key in required
    }


def _segment_row_by_id(segment_state):
    return {
        int(row["segment_id"]): row
        for row in tuple(segment_state.segment_rows or ())
    }


def _edge_override(value, *, name, edge_count, dtype, device):
    tensor = _as_device_tensor(value, dtype=dtype, device=device).view(-1)
    if int(tensor.numel()) != int(edge_count):
        raise ValueError(f"{name} length must match prepared edge count")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _empty_result(dtype, device):
    empty = torch.empty(0, dtype=dtype, device=device)
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    return {
        "segment_delay": empty,
        "segment_output_slew": empty,
        "segment_upstream_visible_input_cap": empty,
        "segment_downstream_load": empty,
        "segment_parent_arrival": empty,
        "segment_parent_slew": empty,
        "sink_arrival": empty,
        "sink_slew": empty,
        "sink_load": empty,
        "sink_node_id": empty_long,
        "sink_net_index": empty_long,
        "driver_pin_id": empty_long,
        "driver_net_cap": empty,
        "metadata": {
            "backend": "prepared_python",
            "net_count": 0,
            "segment_count_state_count": 0,
        },
    }


def segment_count_prepared_relaxed_timing(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival=None,
    driver_slew=None,
    default_driver_arrival=0.0,
    default_driver_slew=0.0,
    transfer_backend="static_size_table",
    buffer_device_lut=None,
    segment_retained_upstream_cap=None,
    edge_resistance_override=None,
    edge_capacitance_override=None,
    canonical_equal_spacing=False,
):
    transfer_backend = str(transfer_backend or "static_size_table")
    if transfer_backend not in {
        "static_size_table",
        "segment_transfer_python",
        "segment_transfer_native",
    }:
        raise ValueError(f"unsupported segment transfer backend: {transfer_backend}")
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

    net_ids = prepared_inputs["net_ids"].to(device=device, dtype=torch.long)
    net_topo_start = prepared_inputs["net_topo_start"].to(device=device, dtype=torch.long)
    flat_topo_node_id = prepared_inputs["flat_topo_node_id"].to(device=device, dtype=torch.long)
    edge_start = prepared_inputs["edge_start"].to(device=device, dtype=torch.long)
    edge_parent_node_id = prepared_inputs["edge_parent_node_id"].to(device=device, dtype=torch.long)
    edge_child_node_id = prepared_inputs["edge_child_node_id"].to(device=device, dtype=torch.long)
    edge_parent_compact_id = prepared_inputs["edge_parent_compact_id"].to(device=device, dtype=torch.long)
    edge_child_compact_id = prepared_inputs["edge_child_compact_id"].to(device=device, dtype=torch.long)
    edge_net_index = prepared_inputs["edge_net_index"].to(device=device, dtype=torch.long)
    edge_resistance = prepared_inputs["edge_resistance"].to(dtype=dtype, device=device)
    edge_capacitance = prepared_inputs["edge_capacitance"].to(dtype=dtype, device=device)
    if (edge_resistance_override is None) != (edge_capacitance_override is None):
        raise ValueError("live edge resistance and capacitance overrides are required together")
    live_edge_override = edge_resistance_override is not None
    if live_edge_override:
        edge_count = int(edge_parent_node_id.numel())
        edge_resistance = _edge_override(
            edge_resistance_override,
            name="edge_resistance_override",
            edge_count=edge_count,
            dtype=dtype,
            device=device,
        )
        edge_capacitance = _edge_override(
            edge_capacitance_override,
            name="edge_capacitance_override",
            edge_count=edge_count,
            dtype=dtype,
            device=device,
        )
    node_capacitance = prepared_inputs["node_capacitance"].to(dtype=dtype, device=device)
    edge_to_segment_id = prepared_inputs["edge_to_segment_id"].to(device=device, dtype=torch.long)
    sink_node_id_tensor = prepared_inputs["sink_node_id"].to(device=device, dtype=torch.long)
    sink_net_id_tensor = prepared_inputs["sink_net_id"].to(device=device, dtype=torch.long)
    sink_node_compact_id = prepared_inputs["sink_node_compact_id"].to(device=device, dtype=torch.long)
    driver_pin_id = prepared_inputs["driver_pin_id"].to(device=device, dtype=torch.long)

    driver_arrival = (
        torch.full((int(net_ids.numel()),), float(default_driver_arrival), dtype=dtype, device=device)
        if driver_arrival is None
        else _as_device_tensor(driver_arrival, dtype=dtype, device=device)
    )
    driver_slew = (
        torch.full((int(net_ids.numel()),), float(default_driver_slew), dtype=dtype, device=device)
        if driver_slew is None
        else _as_device_tensor(driver_slew, dtype=dtype, device=device)
    )

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
    segment_rows = _segment_row_by_id(segment_state)
    transfer_lut_tensors = (
        _buffer_device_tensors(buffer_device_lut, dtype=dtype, device=device)
        if transfer_backend in {"segment_transfer_python", "segment_transfer_native"}
        else None
    )
    segment_retained_upstream_cap = (
        torch.zeros((segment_count,), dtype=dtype, device=device)
        if segment_retained_upstream_cap is None
        else _as_device_tensor(segment_retained_upstream_cap, dtype=dtype, device=device)
    )
    if int(segment_retained_upstream_cap.numel()) != segment_count:
        raise ValueError("segment_retained_upstream_cap length must match segment state")

    zero = torch.zeros((), dtype=dtype, device=device)
    segment_delay = [zero for _ in range(segment_count)]
    segment_output_slew = [zero for _ in range(segment_count)]
    segment_upstream_cap = [zero for _ in range(segment_count)]
    segment_downstream_load = [zero for _ in range(segment_count)]
    segment_parent_arrival = [zero for _ in range(segment_count)]
    segment_parent_slew = [zero for _ in range(segment_count)]
    node_load = [zero for _ in range(int(flat_topo_node_id.numel()))]
    node_arrival = [zero for _ in range(int(flat_topo_node_id.numel()))]
    node_slew = [zero for _ in range(int(flat_topo_node_id.numel()))]
    edge_child_endpoint_cap = [zero for _ in range(int(edge_parent_node_id.numel()))]

    effective_node_cap = [node_capacitance[index] for index in range(int(node_capacitance.numel()))]
    for edge_index in range(int(edge_parent_node_id.numel())):
        parent_compact = int(edge_parent_compact_id[edge_index].detach().cpu().item())
        child_compact = int(edge_child_compact_id[edge_index].detach().cpu().item())
        seg_idx = int(edge_to_segment_id[edge_index].detach().cpu().item())
        edge_cap = edge_capacitance[edge_index]
        if seg_idx < 0:
            parent_half_cap = 0.5 * edge_cap
            child_half_cap = 0.5 * edge_cap
        else:
            if canonical_equal_spacing:
                parent_half_cap = child_half_cap = _edge_endpoint_half_cap(
                    edge_cap,
                    z_value[seg_idx],
                    max_repeater_count,
                    dtype=dtype,
                    device=device,
                )
            else:
                parent_half_cap, child_half_cap = _edge_endpoint_half_caps(
                    edge_cap,
                    segment_rows[seg_idx],
                    z_value[seg_idx],
                    max_repeater_count,
                    dtype=dtype,
                    device=device,
                )
        effective_node_cap[parent_compact] = effective_node_cap[parent_compact] + parent_half_cap
        effective_node_cap[child_compact] = effective_node_cap[child_compact] + child_half_cap
        edge_child_endpoint_cap[edge_index] = child_half_cap

    for net_index in range(int(net_ids.numel())):
        begin = int(net_topo_start[net_index].detach().cpu().item())
        end = int(net_topo_start[net_index + 1].detach().cpu().item())
        for compact in reversed(range(begin, end)):
            children_load = zero
            for edge_index in range(
                int(edge_start[compact].detach().cpu().item()),
                int(edge_start[compact + 1].detach().cpu().item()),
            ):
                child_compact = int(edge_child_compact_id[edge_index].detach().cpu().item())
                seg_idx = int(edge_to_segment_id[edge_index].detach().cpu().item())
                if seg_idx < 0:
                    edge_input = node_load[child_compact]
                else:
                    external_downstream_load = (
                        node_load[child_compact] - edge_child_endpoint_cap[edge_index]
                    )
                    states = _edge_input_load_states(
                        node_load[child_compact],
                        buffer_input_cap[seg_idx],
                        max_repeater_count,
                    )
                    if transfer_backend in {"segment_transfer_python", "segment_transfer_native"}:
                        edge_cap = edge_capacitance[edge_index]
                        states = [states[0]] + [
                            states[count] + 0.5 * edge_cap * (
                                1.0 / (count + 1) if canonical_equal_spacing
                                else _split_fractions(segment_rows[seg_idx], count)[0]
                            ) for count in range(1, max_repeater_count + 1)
                        ]
                    edge_input = _interpolate_count_states(
                        states,
                        z_value[seg_idx],
                        max_repeater_count,
                    )
                    segment_upstream_cap[seg_idx] = edge_input
                    segment_downstream_load[seg_idx] = external_downstream_load
                children_load = children_load + edge_input
            node_load[compact] = effective_node_cap[compact] + children_load

    for net_index in range(int(net_ids.numel())):
        begin = int(net_topo_start[net_index].detach().cpu().item())
        end = int(net_topo_start[net_index + 1].detach().cpu().item())
        if begin == end:
            continue
        node_arrival[begin] = driver_arrival[net_index]
        node_slew[begin] = driver_slew[net_index]
        for compact in range(begin, end):
            for edge_index in range(
                int(edge_start[compact].detach().cpu().item()),
                int(edge_start[compact + 1].detach().cpu().item()),
            ):
                child_compact = int(edge_child_compact_id[edge_index].detach().cpu().item())
                seg_idx = int(edge_to_segment_id[edge_index].detach().cpu().item())
                if seg_idx < 0:
                    edge_delay, child_slew = _edge_delay_slew_state(
                        repeater_count=0,
                        edge_resistance=edge_resistance[edge_index],
                        lin_child=node_load[child_compact],
                        parent_slew=node_slew[compact],
                        buffer_input_cap=zero,
                        buffer_delay=zero,
                        buffer_output_slew=zero,
                        segment_row=None,
                    )
                else:
                    if transfer_backend in {"segment_transfer_python", "segment_transfer_native"}:
                        counts = torch.arange(
                            max_repeater_count + 1,
                            dtype=torch.long,
                            device=device,
                        )
                        split_rows = []
                        for count in range(max_repeater_count + 1):
                            fractions = (
                                [1.0 / float(count + 1)] * (count + 1)
                                if canonical_equal_spacing
                                else _split_fractions(segment_rows[seg_idx], count)
                            )
                            split_rows.append(
                                fractions
                                + [0.0] * (max_repeater_count + 1 - len(fractions))
                            )
                        transfer_op = (
                            segment_transfer_forward_native
                            if transfer_backend == "segment_transfer_native"
                            else batched_segment_transfer
                        )
                        transfer_result = transfer_op(
                            input_arrival=node_arrival[compact].repeat(max_repeater_count + 1),
                            input_slew=node_slew[compact].repeat(max_repeater_count + 1),
                            downstream_load=(
                                node_load[child_compact]
                                - edge_child_endpoint_cap[edge_index]
                            ).repeat(max_repeater_count + 1),
                            edge_resistance=edge_resistance[edge_index].repeat(max_repeater_count + 1),
                            edge_capacitance=edge_capacitance[edge_index].repeat(max_repeater_count + 1),
                            repeater_count=counts,
                            split_fractions=torch.tensor(
                                split_rows,
                                dtype=dtype,
                                device=device,
                            ),
                            bsu_index=buffer_tensors["bsu_index"][seg_idx].repeat(max_repeater_count + 1),
                            upstream_retained_cap=segment_retained_upstream_cap[seg_idx].repeat(max_repeater_count + 1),
                            **transfer_lut_tensors,
                        )
                        edge_delay = _interpolate_count_states(
                            [
                                transfer_result["segment_delay"][count]
                                for count in range(max_repeater_count + 1)
                            ],
                            z_value[seg_idx],
                            max_repeater_count,
                        )
                        child_slew = _interpolate_count_states(
                            [
                                transfer_result["output_slew"][count]
                                for count in range(max_repeater_count + 1)
                            ],
                            z_value[seg_idx],
                            max_repeater_count,
                        )
                    else:
                        states = _edge_delay_slew_states(
                            edge_resistance=edge_resistance[edge_index],
                            lin_child=node_load[child_compact],
                            parent_slew=node_slew[compact],
                            buffer_input_cap=buffer_input_cap[seg_idx],
                            buffer_delay=buffer_delay[seg_idx],
                            buffer_output_slew=buffer_output_slew[seg_idx],
                            max_repeater_count=max_repeater_count,
                            segment_row=(
                                None
                                if canonical_equal_spacing
                                else segment_rows[seg_idx]
                            ),
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
                    segment_parent_arrival[seg_idx] = node_arrival[compact]
                    segment_parent_slew[seg_idx] = node_slew[compact]
                node_arrival[child_compact] = node_arrival[compact] + edge_delay
                node_slew[child_compact] = child_slew

    sink_arrival = []
    sink_slew = []
    sink_load = []
    for compact in sink_node_compact_id.detach().cpu().tolist():
        compact = int(compact)
        sink_arrival.append(node_arrival[compact])
        sink_slew.append(node_slew[compact])
        sink_load.append(node_load[compact])
    driver_net_cap = []
    for net_index in range(int(net_ids.numel())):
        begin = int(net_topo_start[net_index].detach().cpu().item())
        end = int(net_topo_start[net_index + 1].detach().cpu().item())
        if begin == end:
            driver_net_cap.append(zero)
        else:
            driver_net_cap.append(node_load[begin])

    empty = torch.empty(0, dtype=dtype, device=device)
    return {
        "segment_delay": torch.stack(segment_delay) if segment_delay else empty,
        "segment_output_slew": torch.stack(segment_output_slew) if segment_output_slew else empty,
        "segment_upstream_visible_input_cap": torch.stack(segment_upstream_cap) if segment_upstream_cap else empty,
        "segment_downstream_load": torch.stack(segment_downstream_load) if segment_downstream_load else empty,
        "segment_parent_arrival": torch.stack(segment_parent_arrival) if segment_parent_arrival else empty,
        "segment_parent_slew": torch.stack(segment_parent_slew) if segment_parent_slew else empty,
        "sink_arrival": torch.stack(sink_arrival) if sink_arrival else empty,
        "sink_slew": torch.stack(sink_slew) if sink_slew else empty,
        "sink_load": torch.stack(sink_load) if sink_load else empty,
        "sink_node_id": sink_node_id_tensor,
        "sink_net_index": sink_net_id_tensor,
        "driver_pin_id": driver_pin_id,
        "driver_net_cap": torch.stack(driver_net_cap) if driver_net_cap else empty,
        "relaxed_buffer_model": {
            "z_value": z_value,
            "bsu_index": buffer_tensors["bsu_index"],
            "buffer_input_cap": buffer_input_cap,
            "buffer_delay": buffer_delay,
            "buffer_output_slew": buffer_output_slew,
        },
        "metadata": {
            "backend": "prepared_python",
            "segment_transfer_backend": transfer_backend,
            "net_count": int(net_ids.numel()),
            "segment_count_state_count": int(segment_count),
            "max_repeater_count": int(max_repeater_count),
            "segment_shared_bsu": True,
            "edge_net_index_count": int(edge_net_index.numel()),
            "live_edge_override": bool(live_edge_override),
            "canonical_equal_spacing": bool(canonical_equal_spacing),
        },
    }
