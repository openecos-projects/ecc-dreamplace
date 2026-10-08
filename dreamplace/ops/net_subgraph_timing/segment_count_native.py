import importlib
from dataclasses import replace

import torch

from dreamplace.ops.net_subgraph_timing.segment_count_prepared_timing import (
    segment_count_prepared_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    _interpolate_size_tables,
)


def segment_count_forward_native(
    *,
    net_topo_start,
    flat_topo_node_id,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    node_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    sink_node_id,
    sink_net_index,
    sink_node_compact_id,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    return native.segment_count_forward(
        net_topo_start.contiguous(),
        flat_topo_node_id.contiguous(),
        edge_start.contiguous(),
        edge_parent_compact_id.contiguous(),
        edge_child_compact_id.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        node_capacitance.contiguous(),
        edge_to_segment_id.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
        sink_node_compact_id.contiguous(),
    )


def segment_count_backward_native(
    *,
    net_topo_start,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    bsu_index,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    sink_node_compact_id,
    node_load,
    node_slew,
    effective_node_cap,
    grad_segment_delay,
    grad_segment_output_slew,
    grad_segment_upstream_visible_input_cap,
    grad_sink_arrival,
    grad_sink_slew,
    grad_sink_load,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    return native.segment_count_backward(
        net_topo_start.contiguous(),
        edge_start.contiguous(),
        edge_parent_compact_id.contiguous(),
        edge_child_compact_id.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        edge_to_segment_id.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        bsu_index.contiguous(),
        per_size_input_cap.contiguous(),
        per_size_delay.contiguous(),
        per_size_output_slew.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        sink_node_compact_id.contiguous(),
        node_load.contiguous(),
        node_slew.contiguous(),
        effective_node_cap.contiguous(),
        grad_segment_delay.contiguous(),
        grad_segment_output_slew.contiguous(),
        grad_segment_upstream_visible_input_cap.contiguous(),
        grad_sink_arrival.contiguous(),
        grad_sink_slew.contiguous(),
        grad_sink_load.contiguous(),
    )


def _buffer_limit_tensor(value, reference):
    if value is None:
        return torch.full_like(reference, torch.finfo(reference.dtype).max)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device).contiguous()


def segment_count_transfer_forward_native(
    *,
    net_topo_start,
    flat_topo_node_id,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    node_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    bsu_index,
    load_input_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    segment_retained_upstream_cap,
    sink_node_id,
    sink_net_index,
    sink_node_compact_id,
    buffer_slew_limits=None,
    buffer_cap_limits=None,
    count_gradient_enabled=True,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    return native.segment_count_transfer_forward(
        net_topo_start.contiguous(),
        flat_topo_node_id.contiguous(),
        edge_start.contiguous(),
        edge_parent_compact_id.contiguous(),
        edge_child_compact_id.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        node_capacitance.contiguous(),
        edge_to_segment_id.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        bsu_index.contiguous(),
        load_input_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        segment_retained_upstream_cap.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
        sink_node_compact_id.contiguous(),
        _buffer_limit_tensor(buffer_slew_limits, buffer_input_cap_by_size),
        _buffer_limit_tensor(buffer_cap_limits, buffer_input_cap_by_size),
        bool(count_gradient_enabled),
    )


def segment_count_transfer_forward_native_cuda(
    *,
    net_topo_start,
    flat_topo_node_id,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    node_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    bsu_index,
    load_input_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    segment_retained_upstream_cap,
    sink_node_id,
    sink_net_index,
    sink_node_compact_id,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
    outputs = native.segment_count_transfer_forward(
        net_topo_start.to(dtype=torch.long).contiguous(),
        flat_topo_node_id.to(dtype=torch.long).contiguous(),
        edge_start.to(dtype=torch.long).contiguous(),
        edge_parent_compact_id.to(dtype=torch.long).contiguous(),
        edge_child_compact_id.to(dtype=torch.long).contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        node_capacitance.contiguous(),
        edge_to_segment_id.to(dtype=torch.long).contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        bsu_index.contiguous(),
        load_input_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        segment_retained_upstream_cap.contiguous(),
        sink_node_id.to(dtype=torch.long).contiguous(),
        sink_net_index.to(dtype=torch.long).contiguous(),
        sink_node_compact_id.to(dtype=torch.long).contiguous(),
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "node_load": outputs[3],
        "node_arrival": outputs[4],
        "node_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "effective_node_cap": outputs[9],
        "sink_node_id": sink_node_id,
        "sink_net_index": sink_net_index,
    }


def segment_count_cap_forward_native_cuda(
    *,
    net_topo_start,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_capacitance,
    node_capacitance,
    edge_to_segment_id,
    z_value,
    load_input_cap,
    parent_cap_fraction,
    child_cap_fraction,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
    outputs = native.segment_count_cap_forward(
        net_topo_start.to(dtype=torch.long).contiguous(),
        edge_start.to(dtype=torch.long).contiguous(),
        edge_parent_compact_id.to(dtype=torch.long).contiguous(),
        edge_child_compact_id.to(dtype=torch.long).contiguous(),
        edge_capacitance.contiguous(),
        node_capacitance.contiguous(),
        edge_to_segment_id.to(dtype=torch.long).contiguous(),
        z_value.contiguous(),
        load_input_cap.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
    )
    return {
        "node_load": outputs[0],
        "effective_node_cap": outputs[1],
    }


def segment_count_transfer_backward_native(
    *,
    net_topo_start,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    bsu_index,
    load_input_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    sink_node_compact_id,
    node_load,
    node_slew,
    effective_node_cap,
    grad_segment_delay,
    grad_segment_output_slew,
    grad_segment_upstream_visible_input_cap,
    grad_sink_arrival,
    grad_sink_slew,
    grad_sink_load,
    buffer_slew_limits=None,
    buffer_cap_limits=None,
    grad_buffer_slew_violation=None,
    grad_buffer_cap_violation=None,
    count_gradient_enabled=True,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    return native.segment_count_transfer_backward(
        net_topo_start.contiguous(),
        edge_start.contiguous(),
        edge_parent_compact_id.contiguous(),
        edge_child_compact_id.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        edge_to_segment_id.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        bsu_index.contiguous(),
        load_input_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        sink_node_compact_id.contiguous(),
        node_load.contiguous(),
        node_slew.contiguous(),
        effective_node_cap.contiguous(),
        grad_segment_delay.contiguous(),
        grad_segment_output_slew.contiguous(),
        grad_segment_upstream_visible_input_cap.contiguous(),
        grad_sink_arrival.contiguous(),
        grad_sink_slew.contiguous(),
        grad_sink_load.contiguous(),
        _buffer_limit_tensor(buffer_slew_limits, buffer_input_cap_by_size),
        _buffer_limit_tensor(buffer_cap_limits, buffer_input_cap_by_size),
        z_value.new_zeros((z_value.numel(), parent_cap_fraction.shape[1] - 1))
        if grad_buffer_slew_violation is None
        else grad_buffer_slew_violation.contiguous(),
        z_value.new_zeros((z_value.numel(), parent_cap_fraction.shape[1] - 1))
        if grad_buffer_cap_violation is None
        else grad_buffer_cap_violation.contiguous(),
        bool(count_gradient_enabled),
    )


def segment_count_transfer_backward_native_cuda(
    *,
    net_topo_start,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_resistance,
    edge_capacitance,
    edge_to_segment_id,
    driver_arrival,
    driver_slew,
    z_value,
    bsu_index,
    load_input_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
    parent_cap_fraction,
    child_cap_fraction,
    segment_sub_resistance_fraction,
    sink_node_compact_id,
    node_load,
    node_slew,
    effective_node_cap,
    grad_segment_delay,
    grad_segment_output_slew,
    grad_segment_upstream_visible_input_cap,
    grad_sink_arrival,
    grad_sink_slew,
    grad_sink_load,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
    return native.segment_count_transfer_backward(
        net_topo_start.to(dtype=torch.long).contiguous(),
        edge_start.to(dtype=torch.long).contiguous(),
        edge_parent_compact_id.to(dtype=torch.long).contiguous(),
        edge_child_compact_id.to(dtype=torch.long).contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        edge_to_segment_id.to(dtype=torch.long).contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        z_value.contiguous(),
        bsu_index.contiguous(),
        load_input_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        segment_sub_resistance_fraction.contiguous(),
        sink_node_compact_id.to(dtype=torch.long).contiguous(),
        node_load.contiguous(),
        node_slew.contiguous(),
        effective_node_cap.contiguous(),
        grad_segment_delay.contiguous(),
        grad_segment_output_slew.contiguous(),
        grad_segment_upstream_visible_input_cap.contiguous(),
        grad_sink_arrival.contiguous(),
        grad_sink_slew.contiguous(),
        grad_sink_load.contiguous(),
    )


def segment_count_cap_backward_native_cuda(
    *,
    net_topo_start,
    edge_start,
    edge_parent_compact_id,
    edge_child_compact_id,
    edge_capacitance,
    edge_to_segment_id,
    z_value,
    load_input_cap,
    parent_cap_fraction,
    child_cap_fraction,
    node_load,
    grad_driver_net_cap,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
    return native.segment_count_cap_backward(
        net_topo_start.to(dtype=torch.long).contiguous(),
        edge_start.to(dtype=torch.long).contiguous(),
        edge_parent_compact_id.to(dtype=torch.long).contiguous(),
        edge_child_compact_id.to(dtype=torch.long).contiguous(),
        edge_capacitance.contiguous(),
        edge_to_segment_id.to(dtype=torch.long).contiguous(),
        z_value.contiguous(),
        load_input_cap.contiguous(),
        parent_cap_fraction.contiguous(),
        child_cap_fraction.contiguous(),
        node_load.contiguous(),
        grad_driver_net_cap.contiguous(),
    )


class _SegmentCountForwardNativeRecomputeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        prepared_inputs,
        segment_state,
    ):
        runtime_state = replace(
            segment_state,
            z_param=z_param,
            bsu_index_param=bsu_index_param,
        )
        buffer_tensors = _interpolate_size_tables(
            runtime_state.bsu_index().to(dtype=z_param.dtype, device=z_param.device),
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        result = segment_count_forward_native(
            net_topo_start=prepared_inputs["net_topo_start"],
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"],
            edge_start=prepared_inputs["edge_start"],
            edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"],
            edge_child_compact_id=prepared_inputs["edge_child_compact_id"],
            edge_resistance=prepared_inputs["edge_resistance"],
            edge_capacitance=prepared_inputs["edge_capacitance"],
            node_capacitance=prepared_inputs["node_capacitance"],
            edge_to_segment_id=prepared_inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=runtime_state.z_value(),
            buffer_input_cap=buffer_tensors["buffer_input_cap"],
            buffer_delay=buffer_tensors["buffer_delay"],
            buffer_output_slew=buffer_tensors["buffer_output_slew"],
            parent_cap_fraction=prepared_inputs["parent_cap_fraction"],
            child_cap_fraction=prepared_inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared_inputs[
                "segment_sub_resistance_fraction"
            ],
            sink_node_id=prepared_inputs["sink_node_id"],
            sink_net_index=prepared_inputs["sink_net_id"],
            sink_node_compact_id=prepared_inputs["sink_node_compact_id"],
        )
        ctx.prepared_inputs = prepared_inputs
        ctx.segment_state = segment_state
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
        )
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_load"].device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_arrival"].device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_slew"].device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"],
            result["sink_net_index"],
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared_inputs, device=result["node_load"].device),
            ),
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            z_recompute = z_param.detach().requires_grad_(True)
            bsu_recompute = bsu_index_param.detach().requires_grad_(True)
            driver_arrival_recompute = driver_arrival.detach().requires_grad_(True)
            driver_slew_recompute = driver_slew.detach().requires_grad_(True)
            recompute_state = replace(
                ctx.segment_state,
                z_param=z_recompute,
                bsu_index_param=bsu_recompute,
            )
            recompute = segment_count_prepared_relaxed_timing(
                ctx.prepared_inputs,
                segment_state=recompute_state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                driver_arrival=driver_arrival_recompute,
                driver_slew=driver_slew_recompute,
            )
            objective = torch.zeros((), dtype=z_param.dtype, device=z_param.device)

            def add_vjp(output_key, grad_output):
                nonlocal objective
                if grad_output is None:
                    return
                output = recompute[output_key]
                if output.numel() == 0:
                    return
                objective = objective + torch.sum(output * grad_output.to(output))

            add_vjp("segment_delay", grad_segment_delay)
            add_vjp("segment_output_slew", grad_segment_output_slew)
            add_vjp(
                "segment_upstream_visible_input_cap",
                grad_segment_upstream_visible_input_cap,
            )
            add_vjp("sink_arrival", grad_sink_arrival)
            add_vjp("sink_slew", grad_sink_slew)
            add_vjp("sink_load", grad_sink_load)
            add_vjp("driver_net_cap", grad_driver_net_cap)
            grads = torch.autograd.grad(
                objective,
                (
                    z_recompute,
                    bsu_recompute,
                    driver_arrival_recompute,
                    driver_slew_recompute,
                ),
                allow_unused=True,
            )

        grad_z, grad_bsu, grad_driver_arrival, grad_driver_slew = grads
        return (
            grad_z,
            grad_bsu,
            grad_driver_arrival,
            grad_driver_slew,
            None,
            None,
            None,
            None,
            None,
        )


def segment_count_forward_native_recompute_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival,
    driver_slew,
):
    per_size_input_cap = torch.as_tensor(
        per_size_input_cap,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    per_size_delay = torch.as_tensor(
        per_size_delay,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    driver_arrival = torch.as_tensor(
        driver_arrival,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    driver_slew = torch.as_tensor(
        driver_slew,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    outputs = _SegmentCountForwardNativeRecomputeAutograd.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        prepared_inputs,
        segment_state,
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "segment_downstream_load": outputs[3],
        "segment_parent_arrival": outputs[4],
        "segment_parent_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "sink_node_id": outputs[9],
        "sink_net_index": outputs[10],
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=outputs[11].device,
            dtype=torch.long,
        ),
        "driver_net_cap": outputs[11],
        "metadata": {
            "backend": "cpp_cpu_recompute_autograd",
            "net_count": int(prepared_inputs["net_ids"].numel()),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
        },
    }


def _tensor_list(tensor):
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _float_list(tensor):
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _grad_list(grad, length, *, dtype, device):
    if grad is None:
        return [0.0 for _ in range(length)]
    if int(grad.numel()) == 0:
        return [0.0 for _ in range(length)]
    return [
        float(value)
        for value in grad.detach().to(dtype=dtype, device=device).cpu().tolist()
    ]


def _scatter_index_grad(target, index, grad, *, dtype, device):
    if grad is None or int(grad.numel()) == 0:
        return
    values = _grad_list(grad, int(index.numel()), dtype=dtype, device=device)
    for offset, compact in enumerate(_tensor_list(index)):
        target[int(compact)] += values[offset]


def _root_compact_index(prepared_inputs, *, device):
    net_topo_start = prepared_inputs["net_topo_start"].to(device=device, dtype=torch.long)
    if int(net_topo_start.numel()) <= 1:
        return torch.empty(0, dtype=torch.long, device=device)
    return net_topo_start[:-1]


def _segment_child_compact_index(prepared_inputs, segment_count, *, device):
    edge_to_segment_id = prepared_inputs["edge_to_segment_id"].to(
        device=device,
        dtype=torch.long,
    )
    edge_child_compact_id = prepared_inputs["edge_child_compact_id"].to(
        device=device,
        dtype=torch.long,
    )
    result = torch.zeros(int(segment_count), dtype=torch.long, device=device)
    if int(edge_to_segment_id.numel()) == 0 or int(segment_count) == 0:
        return result
    valid = (edge_to_segment_id >= 0) & (edge_to_segment_id < int(segment_count))
    if not bool(torch.any(valid).detach().cpu().item()):
        return result
    result[edge_to_segment_id[valid]] = edge_child_compact_id[valid]
    return result


def _segment_parent_compact_index(prepared_inputs, segment_count, *, device):
    edge_to_segment_id = prepared_inputs["edge_to_segment_id"].to(
        device=device,
        dtype=torch.long,
    )
    edge_parent_compact_id = prepared_inputs["edge_parent_compact_id"].to(
        device=device,
        dtype=torch.long,
    )
    result = torch.zeros(int(segment_count), dtype=torch.long, device=device)
    if int(edge_to_segment_id.numel()) == 0 or int(segment_count) == 0:
        return result
    valid = (edge_to_segment_id >= 0) & (edge_to_segment_id < int(segment_count))
    if not bool(torch.any(valid).detach().cpu().item()):
        return result
    result[edge_to_segment_id[valid]] = edge_parent_compact_id[valid]
    return result


def _clamp_backward_mask(value, low, high):
    return ((value >= float(low)) & (value <= float(high))).to(dtype=value.dtype)


def _prepared_with_canonical_equal_spacing(prepared_inputs):
    fractions = prepared_inputs["segment_sub_resistance_fraction"]
    cached = prepared_inputs.get("canonical_segment_sub_resistance_fraction")
    if cached is None:
        if fractions.ndim != 3 or int(fractions.shape[1]) != int(fractions.shape[2]):
            raise ValueError("segment split fractions must have shape [S, N+1, N+1]")
        cached = torch.zeros_like(fractions)
        for count in range(int(fractions.shape[1])):
            cached[:, count, : count + 1] = 1.0 / float(count + 1)
        prepared_inputs["canonical_segment_sub_resistance_fraction"] = cached
    elif cached.shape != fractions.shape:
        raise ValueError("cached canonical segment fractions have the wrong shape")
    parent_fraction = prepared_inputs["parent_cap_fraction"]
    child_fraction = prepared_inputs["child_cap_fraction"]
    canonical_endpoint = prepared_inputs.get("canonical_segment_endpoint_cap_fraction")
    if canonical_endpoint is None:
        canonical_endpoint = torch.empty_like(parent_fraction)
        for count in range(int(parent_fraction.shape[1])):
            canonical_endpoint[:, count] = 1.0 / float(count + 1)
        prepared_inputs["canonical_segment_endpoint_cap_fraction"] = canonical_endpoint
    elif canonical_endpoint.shape != parent_fraction.shape:
        raise ValueError("cached canonical endpoint fractions have the wrong shape")
    if child_fraction.shape != parent_fraction.shape:
        raise ValueError("parent and child cap fractions must have matching shapes")
    runtime_prepared = dict(prepared_inputs)
    runtime_prepared["segment_sub_resistance_fraction"] = cached
    runtime_prepared["parent_cap_fraction"] = canonical_endpoint
    runtime_prepared["child_cap_fraction"] = canonical_endpoint
    return runtime_prepared


def _recompute_backward_grads(
    *,
    prepared_inputs,
    segment_state,
    z_param,
    bsu_index_param,
    driver_arrival,
    driver_slew,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    grad_segment_delay,
    grad_segment_output_slew,
    grad_segment_upstream_visible_input_cap,
    grad_sink_arrival,
    grad_sink_slew,
    grad_sink_load,
    grad_driver_net_cap=None,
):
    with torch.enable_grad():
        z_recompute = z_param.detach().requires_grad_(True)
        bsu_recompute = bsu_index_param.detach().requires_grad_(True)
        driver_arrival_recompute = driver_arrival.detach().requires_grad_(True)
        driver_slew_recompute = driver_slew.detach().requires_grad_(True)
        recompute_state = replace(
            segment_state,
            z_param=z_recompute,
            bsu_index_param=bsu_recompute,
        )
        recompute = segment_count_prepared_relaxed_timing(
            prepared_inputs,
            segment_state=recompute_state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            driver_arrival=driver_arrival_recompute,
            driver_slew=driver_slew_recompute,
        )
        objective = torch.zeros((), dtype=z_param.dtype, device=z_param.device)

        def add_vjp(key, grad_output):
            nonlocal objective
            if grad_output is None:
                return
            output = recompute[key]
            if int(output.numel()) == 0:
                return
            objective = objective + torch.sum(output * grad_output.to(output))

        add_vjp("segment_delay", grad_segment_delay)
        add_vjp("segment_output_slew", grad_segment_output_slew)
        add_vjp("segment_upstream_visible_input_cap", grad_segment_upstream_visible_input_cap)
        add_vjp("sink_arrival", grad_sink_arrival)
        add_vjp("sink_slew", grad_sink_slew)
        add_vjp("sink_load", grad_sink_load)
        add_vjp("driver_net_cap", grad_driver_net_cap)
        return torch.autograd.grad(
            objective,
            (
                z_recompute,
                bsu_recompute,
                driver_arrival_recompute,
                driver_slew_recompute,
            ),
            allow_unused=True,
        )


class _SegmentCountTransferNativeRecomputeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        segment_retained_upstream_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        prepared_inputs,
        segment_state,
        buffer_device_lut,
    ):
        runtime_state = replace(
            segment_state,
            z_param=z_param,
            bsu_index_param=bsu_index_param,
        )
        load_buffer_tensors = _interpolate_size_tables(
            runtime_state.bsu_index().to(dtype=z_param.dtype, device=z_param.device),
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        result = segment_count_transfer_forward_native(
            net_topo_start=prepared_inputs["net_topo_start"],
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"],
            edge_start=prepared_inputs["edge_start"],
            edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"],
            edge_child_compact_id=prepared_inputs["edge_child_compact_id"],
            edge_resistance=prepared_inputs["edge_resistance"],
            edge_capacitance=prepared_inputs["edge_capacitance"],
            node_capacitance=prepared_inputs["node_capacitance"],
            edge_to_segment_id=prepared_inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=runtime_state.z_value(),
            bsu_index=runtime_state.bsu_index().to(
                dtype=z_param.dtype,
                device=z_param.device,
            ),
            load_input_cap=load_buffer_tensors["buffer_input_cap"],
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
            parent_cap_fraction=prepared_inputs["parent_cap_fraction"],
            child_cap_fraction=prepared_inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared_inputs[
                "segment_sub_resistance_fraction"
            ],
            segment_retained_upstream_cap=segment_retained_upstream_cap,
            sink_node_id=prepared_inputs["sink_node_id"],
            sink_net_index=prepared_inputs["sink_net_id"],
            sink_node_compact_id=prepared_inputs["sink_node_compact_id"],
        )
        ctx.prepared_inputs = prepared_inputs
        ctx.segment_state = segment_state
        ctx.buffer_device_lut = buffer_device_lut
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
            segment_retained_upstream_cap.detach(),
        )
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_load"].device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_arrival"].device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_slew"].device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"],
            result["sink_net_index"],
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared_inputs, device=result["node_load"].device),
            ),
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            segment_retained_upstream_cap,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            z_recompute = z_param.detach().requires_grad_(True)
            bsu_recompute = bsu_index_param.detach().requires_grad_(True)
            driver_arrival_recompute = driver_arrival.detach().requires_grad_(True)
            driver_slew_recompute = driver_slew.detach().requires_grad_(True)
            retained_recompute = segment_retained_upstream_cap.detach().requires_grad_(True)
            recompute_state = replace(
                ctx.segment_state,
                z_param=z_recompute,
                bsu_index_param=bsu_recompute,
            )
            recompute = segment_count_prepared_relaxed_timing(
                ctx.prepared_inputs,
                segment_state=recompute_state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                driver_arrival=driver_arrival_recompute,
                driver_slew=driver_slew_recompute,
                transfer_backend="segment_transfer_native",
                buffer_device_lut=ctx.buffer_device_lut,
                segment_retained_upstream_cap=retained_recompute,
            )
            objective = torch.zeros((), dtype=z_param.dtype, device=z_param.device)

            def add_vjp(key, grad_output):
                nonlocal objective
                if grad_output is None:
                    return
                output = recompute[key]
                if int(output.numel()) == 0:
                    return
                objective = objective + torch.sum(output * grad_output.to(output))

            add_vjp("segment_delay", grad_segment_delay)
            add_vjp("segment_output_slew", grad_segment_output_slew)
            add_vjp(
                "segment_upstream_visible_input_cap",
                grad_segment_upstream_visible_input_cap,
            )
            add_vjp("sink_arrival", grad_sink_arrival)
            add_vjp("sink_slew", grad_sink_slew)
            add_vjp("sink_load", grad_sink_load)
            add_vjp("driver_net_cap", grad_driver_net_cap)
            grads = torch.autograd.grad(
                objective,
                (
                    z_recompute,
                    bsu_recompute,
                    driver_arrival_recompute,
                    driver_slew_recompute,
                    retained_recompute,
                ),
                allow_unused=True,
            )

        grad_z, grad_bsu, grad_driver_arrival, grad_driver_slew, _ = grads
        return (
            grad_z,
            grad_bsu,
            grad_driver_arrival,
            grad_driver_slew,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SegmentCountTransferNativeExplicitAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        edge_resistance,
        edge_capacitance,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        segment_retained_upstream_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        prepared_inputs,
        segment_state,
        node_capacitance,
        buffer_slew_limits,
        buffer_cap_limits,
    ):
        runtime_state = replace(
            segment_state,
            z_param=z_param,
            bsu_index_param=bsu_index_param,
        )
        bsu_index = runtime_state.bsu_index().to(dtype=z_param.dtype, device=z_param.device)
        load_buffer_tensors = _interpolate_size_tables(
            bsu_index,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        result = segment_count_transfer_forward_native(
            net_topo_start=prepared_inputs["net_topo_start"],
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"],
            edge_start=prepared_inputs["edge_start"],
            edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"],
            edge_child_compact_id=prepared_inputs["edge_child_compact_id"],
            edge_resistance=edge_resistance,
            edge_capacitance=edge_capacitance,
            node_capacitance=node_capacitance,
            edge_to_segment_id=prepared_inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=runtime_state.z_value(),
            bsu_index=bsu_index,
            load_input_cap=load_buffer_tensors["buffer_input_cap"],
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
            parent_cap_fraction=prepared_inputs["parent_cap_fraction"],
            child_cap_fraction=prepared_inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared_inputs[
                "segment_sub_resistance_fraction"
            ],
            segment_retained_upstream_cap=segment_retained_upstream_cap,
            sink_node_id=prepared_inputs["sink_node_id"],
            sink_net_index=prepared_inputs["sink_net_id"],
            sink_node_compact_id=prepared_inputs["sink_node_compact_id"],
            buffer_slew_limits=buffer_slew_limits,
            buffer_cap_limits=buffer_cap_limits,
            count_gradient_enabled=z_param.requires_grad,
        )
        ctx.prepared_inputs = prepared_inputs
        ctx.segment_state = segment_state
        ctx.count_gradient_enabled = z_param.requires_grad
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            edge_resistance.detach(),
            edge_capacitance.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
            segment_retained_upstream_cap.detach(),
            buffer_input_cap_by_size.detach(),
            buffer_slew_axis.detach(),
            buffer_load_axis.detach(),
            buffer_delay_lut.detach(),
            buffer_output_slew_lut.detach(),
            load_buffer_tensors["buffer_input_cap"].detach(),
            result["node_load"].detach(),
            result["node_slew"].detach(),
            result["effective_node_cap"].detach(),
            buffer_slew_limits.detach(),
            buffer_cap_limits.detach(),
        )
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_load"].device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_arrival"].device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_slew"].device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"],
            result["sink_net_index"],
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared_inputs, device=result["node_load"].device),
            ),
            result["buffer_slew_violation"],
            result["buffer_cap_violation"],
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
        grad_buffer_slew_violation,
        grad_buffer_cap_violation,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            edge_resistance,
            edge_capacitance,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            segment_retained_upstream_cap,
            buffer_input_cap_by_size,
            buffer_slew_axis,
            buffer_load_axis,
            buffer_delay_lut,
            buffer_output_slew_lut,
            load_input_cap,
            node_load,
            node_slew,
            effective_node_cap,
            buffer_slew_limits,
            buffer_cap_limits,
        ) = ctx.saved_tensors
        prepared = ctx.prepared_inputs
        dtype = z_param.dtype
        device = z_param.device
        num_segments = int(z_param.numel())
        zero_segment_grad = torch.zeros(num_segments, dtype=dtype, device=device)
        sink_count = int(prepared["sink_node_compact_id"].numel())
        sink_load_grad = (
            grad_sink_load.to(dtype=dtype, device=device)
            if grad_sink_load is not None
            else torch.zeros(sink_count, dtype=dtype, device=device)
        )
        sink_arrival_grad = (
            grad_sink_arrival.to(dtype=dtype, device=device)
            if grad_sink_arrival is not None
            else torch.zeros(sink_count, dtype=dtype, device=device)
        )
        sink_slew_grad = (
            grad_sink_slew.to(dtype=dtype, device=device)
            if grad_sink_slew is not None
            else torch.zeros(sink_count, dtype=dtype, device=device)
        )
        if grad_driver_net_cap is not None and int(grad_driver_net_cap.numel()) > 0:
            root_index = _root_compact_index(prepared, device=device)
            sink_compact = torch.cat(
                [
                    prepared["sink_node_compact_id"].to(device=device, dtype=torch.long),
                    root_index,
                ],
                dim=0,
            )
            sink_load_grad = torch.cat(
                [
                    sink_load_grad,
                    grad_driver_net_cap.to(dtype=dtype, device=device),
                ],
                dim=0,
            )
            root_zeros = torch.zeros(
                int(root_index.numel()),
                dtype=dtype,
                device=device,
            )
            sink_arrival_grad = torch.cat([sink_arrival_grad, root_zeros], dim=0)
            sink_slew_grad = torch.cat([sink_slew_grad, root_zeros], dim=0)
        else:
            sink_compact = prepared["sink_node_compact_id"]

        bsu_clamped = torch.clamp(
            bsu_index_param,
            0.0,
            float(buffer_input_cap_by_size.numel() - 1),
        )
        cpp_grads = segment_count_transfer_backward_native(
            net_topo_start=prepared["net_topo_start"],
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_resistance=edge_resistance,
            edge_capacitance=edge_capacitance,
            edge_to_segment_id=prepared["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=torch.clamp(z_param, 0.0, float(ctx.segment_state.max_repeater_count)),
            bsu_index=bsu_clamped,
            load_input_cap=load_input_cap,
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared["segment_sub_resistance_fraction"],
            sink_node_compact_id=sink_compact,
            node_load=node_load,
            node_slew=node_slew,
            effective_node_cap=effective_node_cap,
            grad_segment_delay=(
                grad_segment_delay.to(dtype=dtype, device=device)
                if grad_segment_delay is not None
                else zero_segment_grad
            ),
            grad_segment_output_slew=(
                grad_segment_output_slew.to(dtype=dtype, device=device)
                if grad_segment_output_slew is not None
                else zero_segment_grad
            ),
            grad_segment_upstream_visible_input_cap=(
                grad_segment_upstream_visible_input_cap.to(dtype=dtype, device=device)
                if grad_segment_upstream_visible_input_cap is not None
                else zero_segment_grad
            ),
            grad_sink_arrival=sink_arrival_grad,
            grad_sink_slew=sink_slew_grad,
            grad_sink_load=sink_load_grad,
            buffer_slew_limits=buffer_slew_limits,
            buffer_cap_limits=buffer_cap_limits,
            grad_buffer_slew_violation=grad_buffer_slew_violation,
            grad_buffer_cap_violation=grad_buffer_cap_violation,
            count_gradient_enabled=ctx.count_gradient_enabled,
        )
        with torch.enable_grad():
            bsu_for_load = bsu_index_param.detach().requires_grad_(True)
            load_interp = _interpolate_size_tables(
                torch.clamp(
                    bsu_for_load,
                    0.0,
                    float(per_size_input_cap.shape[1] - 1),
                ),
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
            )["buffer_input_cap"]
            grad_load_bsu = torch.autograd.grad(
                torch.sum(
                    load_interp
                    * cpp_grads["grad_load_input_cap"].to(dtype=dtype, device=device)
                ),
                bsu_for_load,
                allow_unused=True,
            )[0]
        if grad_load_bsu is None:
            grad_load_bsu = torch.zeros_like(bsu_index_param)
        grad_bsu = cpp_grads["grad_bsu"] + grad_load_bsu
        return (
            cpp_grads["grad_z"] * _clamp_backward_mask(
                z_param,
                0.0,
                float(ctx.segment_state.max_repeater_count),
            ),
            grad_bsu * _clamp_backward_mask(
                bsu_index_param,
                0.0,
                float(per_size_input_cap.shape[1] - 1),
            ),
            cpp_grads["grad_driver_arrival"],
            cpp_grads["grad_driver_slew"],
            cpp_grads["grad_edge_resistance"],
            cpp_grads["grad_edge_capacitance"],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            cpp_grads["grad_node_capacitance"],
            None,
            None,
        )


class _SegmentCountTransferCudaExplicitAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        edge_resistance,
        edge_capacitance,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        segment_retained_upstream_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        prepared_inputs,
        segment_state,
    ):
        runtime_state = replace(
            segment_state,
            z_param=z_param,
            bsu_index_param=bsu_index_param,
        )
        dtype = z_param.dtype
        cpu_device = torch.device("cpu")
        bsu_index_cpu = runtime_state.bsu_index().to(dtype=dtype, device=cpu_device)
        load_buffer_tensors = _interpolate_size_tables(
            bsu_index_cpu,
            per_size_input_cap=per_size_input_cap.to(dtype=dtype, device=cpu_device),
            per_size_delay=per_size_delay.to(dtype=dtype, device=cpu_device),
            per_size_output_slew=per_size_output_slew.to(dtype=dtype, device=cpu_device),
        )

        def to_cuda(tensor, *, dtype_override=None):
            target_dtype = dtype if dtype_override is None else dtype_override
            return tensor.to(dtype=target_dtype, device="cuda")

        result_cuda = segment_count_transfer_forward_native_cuda(
            net_topo_start=prepared_inputs["net_topo_start"].to(device="cuda"),
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"].to(device="cuda"),
            edge_start=prepared_inputs["edge_start"].to(device="cuda"),
            edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"].to(device="cuda"),
            edge_child_compact_id=prepared_inputs["edge_child_compact_id"].to(device="cuda"),
            edge_resistance=to_cuda(edge_resistance),
            edge_capacitance=to_cuda(edge_capacitance),
            node_capacitance=to_cuda(prepared_inputs["node_capacitance"]),
            edge_to_segment_id=prepared_inputs["edge_to_segment_id"].to(device="cuda"),
            driver_arrival=driver_arrival.to(dtype=dtype, device="cuda"),
            driver_slew=driver_slew.to(dtype=dtype, device="cuda"),
            z_value=runtime_state.z_value().to(dtype=dtype, device="cuda"),
            bsu_index=runtime_state.bsu_index().to(dtype=dtype, device="cuda"),
            load_input_cap=load_buffer_tensors["buffer_input_cap"].to(device="cuda"),
            buffer_input_cap_by_size=buffer_input_cap_by_size.to(dtype=dtype, device="cuda"),
            buffer_slew_axis=buffer_slew_axis.to(dtype=dtype, device="cuda"),
            buffer_load_axis=buffer_load_axis.to(dtype=dtype, device="cuda"),
            buffer_delay_lut=buffer_delay_lut.to(dtype=dtype, device="cuda"),
            buffer_output_slew_lut=buffer_output_slew_lut.to(dtype=dtype, device="cuda"),
            parent_cap_fraction=to_cuda(prepared_inputs["parent_cap_fraction"]),
            child_cap_fraction=to_cuda(prepared_inputs["child_cap_fraction"]),
            segment_sub_resistance_fraction=to_cuda(
                prepared_inputs["segment_sub_resistance_fraction"]
            ),
            segment_retained_upstream_cap=segment_retained_upstream_cap.to(
                dtype=dtype,
                device="cuda",
            ),
            sink_node_id=prepared_inputs["sink_node_id"].to(device="cuda"),
            sink_net_index=prepared_inputs["sink_net_id"].to(device="cuda"),
            sink_node_compact_id=prepared_inputs["sink_node_compact_id"].to(device="cuda"),
        )
        result = {
            key: value.to(device=cpu_device)
            if torch.is_tensor(value) and value.is_cuda
            else value
            for key, value in result_cuda.items()
        }
        ctx.prepared_inputs = prepared_inputs
        ctx.segment_state = segment_state
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            edge_resistance.detach(),
            edge_capacitance.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
            segment_retained_upstream_cap.detach(),
            buffer_input_cap_by_size.detach(),
            buffer_slew_axis.detach(),
            buffer_load_axis.detach(),
            buffer_delay_lut.detach(),
            buffer_output_slew_lut.detach(),
            load_buffer_tensors["buffer_input_cap"].detach(),
            result["node_load"].detach(),
            result["node_slew"].detach(),
            result["effective_node_cap"].detach(),
        )
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_load"].device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_arrival"].device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_slew"].device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"].to(device=cpu_device, dtype=torch.long),
            result["sink_net_index"].to(device=cpu_device, dtype=torch.long),
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared_inputs, device=result["node_load"].device),
            ),
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            edge_resistance,
            edge_capacitance,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            segment_retained_upstream_cap,
            buffer_input_cap_by_size,
            buffer_slew_axis,
            buffer_load_axis,
            buffer_delay_lut,
            buffer_output_slew_lut,
            load_input_cap,
            node_load,
            node_slew,
            effective_node_cap,
        ) = ctx.saved_tensors
        del segment_retained_upstream_cap
        del grad_segment_downstream_load
        del grad_segment_parent_arrival
        del grad_segment_parent_slew
        del grad_sink_node_id
        del grad_sink_net_index

        prepared = ctx.prepared_inputs
        dtype = z_param.dtype
        cpu_device = z_param.device
        cuda_device = torch.device("cuda")
        zero_segment_grad_cpu = torch.zeros_like(z_param)
        sink_arrival_grad = (
            grad_sink_arrival
            if grad_sink_arrival is not None
            else torch.zeros(
                int(prepared["sink_node_compact_id"].numel()),
                dtype=dtype,
                device=cpu_device,
            )
        )
        sink_slew_grad = (
            grad_sink_slew
            if grad_sink_slew is not None
            else torch.zeros_like(sink_arrival_grad)
        )
        sink_load_grad = (
            grad_sink_load
            if grad_sink_load is not None
            else torch.zeros_like(sink_arrival_grad)
        )
        if grad_driver_net_cap is not None:
            root_index = _root_compact_index(prepared, device=cpu_device)
            sink_compact = torch.cat(
                [
                    prepared["sink_node_compact_id"].to(device=cpu_device, dtype=torch.long),
                    root_index,
                ],
                dim=0,
            )
            sink_load_grad = torch.cat(
                [
                    sink_load_grad,
                    grad_driver_net_cap.to(dtype=dtype, device=cpu_device),
                ],
                dim=0,
            )
            root_zeros = torch.zeros(
                int(root_index.numel()),
                dtype=dtype,
                device=cpu_device,
            )
            sink_arrival_grad = torch.cat([sink_arrival_grad, root_zeros], dim=0)
            sink_slew_grad = torch.cat([sink_slew_grad, root_zeros], dim=0)
        else:
            sink_compact = prepared["sink_node_compact_id"].to(
                device=cpu_device,
                dtype=torch.long,
            )

        bsu_clamped = torch.clamp(
            bsu_index_param,
            0.0,
            float(buffer_input_cap_by_size.numel() - 1),
        )

        def to_cuda(tensor, *, dtype_override=None):
            target_dtype = dtype if dtype_override is None else dtype_override
            return tensor.to(dtype=target_dtype, device=cuda_device)

        cuda_grads = segment_count_transfer_backward_native_cuda(
            net_topo_start=prepared["net_topo_start"].to(device=cuda_device),
            edge_start=prepared["edge_start"].to(device=cuda_device),
            edge_parent_compact_id=prepared["edge_parent_compact_id"].to(device=cuda_device),
            edge_child_compact_id=prepared["edge_child_compact_id"].to(device=cuda_device),
            edge_resistance=to_cuda(edge_resistance),
            edge_capacitance=to_cuda(edge_capacitance),
            edge_to_segment_id=prepared["edge_to_segment_id"].to(device=cuda_device),
            driver_arrival=driver_arrival.to(dtype=dtype, device=cuda_device),
            driver_slew=driver_slew.to(dtype=dtype, device=cuda_device),
            z_value=torch.clamp(
                z_param,
                0.0,
                float(ctx.segment_state.max_repeater_count),
            ).to(dtype=dtype, device=cuda_device),
            bsu_index=bsu_clamped.to(dtype=dtype, device=cuda_device),
            load_input_cap=load_input_cap.to(dtype=dtype, device=cuda_device),
            buffer_input_cap_by_size=buffer_input_cap_by_size.to(
                dtype=dtype,
                device=cuda_device,
            ),
            buffer_slew_axis=buffer_slew_axis.to(dtype=dtype, device=cuda_device),
            buffer_load_axis=buffer_load_axis.to(dtype=dtype, device=cuda_device),
            buffer_delay_lut=buffer_delay_lut.to(dtype=dtype, device=cuda_device),
            buffer_output_slew_lut=buffer_output_slew_lut.to(
                dtype=dtype,
                device=cuda_device,
            ),
            parent_cap_fraction=to_cuda(prepared["parent_cap_fraction"]),
            child_cap_fraction=to_cuda(prepared["child_cap_fraction"]),
            segment_sub_resistance_fraction=to_cuda(
                prepared["segment_sub_resistance_fraction"]
            ),
            sink_node_compact_id=sink_compact.to(device=cuda_device),
            node_load=node_load.to(dtype=dtype, device=cuda_device),
            node_slew=node_slew.to(dtype=dtype, device=cuda_device),
            effective_node_cap=effective_node_cap.to(dtype=dtype, device=cuda_device),
            grad_segment_delay=(
                grad_segment_delay.to(dtype=dtype, device=cuda_device)
                if grad_segment_delay is not None
                else zero_segment_grad_cpu.to(device=cuda_device)
            ),
            grad_segment_output_slew=(
                grad_segment_output_slew.to(dtype=dtype, device=cuda_device)
                if grad_segment_output_slew is not None
                else zero_segment_grad_cpu.to(device=cuda_device)
            ),
            grad_segment_upstream_visible_input_cap=(
                grad_segment_upstream_visible_input_cap.to(
                    dtype=dtype,
                    device=cuda_device,
                )
                if grad_segment_upstream_visible_input_cap is not None
                else zero_segment_grad_cpu.to(device=cuda_device)
            ),
            grad_sink_arrival=sink_arrival_grad.to(dtype=dtype, device=cuda_device),
            grad_sink_slew=sink_slew_grad.to(dtype=dtype, device=cuda_device),
            grad_sink_load=sink_load_grad.to(dtype=dtype, device=cuda_device),
        )
        cuda_grad_input_cap_cpu = cuda_grads["grad_load_input_cap"].to(
            dtype=dtype,
            device=cpu_device,
        )
        with torch.enable_grad():
            bsu_for_load = bsu_index_param.detach().requires_grad_(True)
            load_interp = _interpolate_size_tables(
                torch.clamp(
                    bsu_for_load,
                    0.0,
                    float(per_size_input_cap.shape[1] - 1),
                ),
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
            )["buffer_input_cap"]
            grad_load_bsu = torch.autograd.grad(
                torch.sum(load_interp * cuda_grad_input_cap_cpu),
                bsu_for_load,
                allow_unused=True,
            )[0]
        if grad_load_bsu is None:
            grad_load_bsu = torch.zeros_like(bsu_index_param)
        grad_bsu = cuda_grads["grad_bsu"].to(dtype=dtype, device=cpu_device) + grad_load_bsu
        return (
            cuda_grads["grad_z"].to(dtype=dtype, device=cpu_device)
            * _clamp_backward_mask(
                z_param,
                0.0,
                float(ctx.segment_state.max_repeater_count),
            ),
            grad_bsu
            * _clamp_backward_mask(
                bsu_index_param,
                0.0,
                float(per_size_input_cap.shape[1] - 1),
            ),
            cuda_grads["grad_driver_arrival"].to(dtype=dtype, device=cpu_device),
            cuda_grads["grad_driver_slew"].to(dtype=dtype, device=cpu_device),
            cuda_grads["grad_edge_resistance"].to(dtype=dtype, device=cpu_device),
            cuda_grads["grad_edge_capacitance"].to(dtype=dtype, device=cpu_device),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SegmentCountTransferCudaGpuResidentAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        edge_resistance,
        edge_capacitance,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        segment_retained_upstream_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        prepared_inputs,
        segment_state,
    ):
        if not z_param.is_cuda:
            raise RuntimeError("GPU-resident segment transfer requires CUDA segment state")
        device = z_param.device
        dtype = z_param.dtype

        def on_device(tensor, *, dtype_override=None):
            return tensor.to(
                device=device,
                dtype=dtype if dtype_override is None else dtype_override,
            )

        z_value = torch.clamp(
            z_param,
            0.0,
            float(segment_state.max_repeater_count),
        )
        bsu_index = torch.clamp(
            bsu_index_param,
            0.0,
            float(buffer_input_cap_by_size.numel() - 1),
        )
        per_size_input_cap = on_device(per_size_input_cap)
        per_size_delay = on_device(per_size_delay)
        per_size_output_slew = on_device(per_size_output_slew)
        load_input_cap = _interpolate_size_tables(
            bsu_index,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )["buffer_input_cap"]
        prepared = {
            "net_topo_start": prepared_inputs["net_topo_start"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_start": prepared_inputs["edge_start"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_parent_compact_id": prepared_inputs["edge_parent_compact_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_child_compact_id": prepared_inputs["edge_child_compact_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_resistance": on_device(edge_resistance),
            "edge_capacitance": on_device(edge_capacitance),
            "node_capacitance": on_device(prepared_inputs["node_capacitance"]),
            "edge_to_segment_id": prepared_inputs["edge_to_segment_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "parent_cap_fraction": on_device(prepared_inputs["parent_cap_fraction"]),
            "child_cap_fraction": on_device(prepared_inputs["child_cap_fraction"]),
            "segment_sub_resistance_fraction": on_device(
                prepared_inputs["segment_sub_resistance_fraction"]
            ),
            "sink_node_id": prepared_inputs["sink_node_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "sink_net_id": prepared_inputs["sink_net_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "sink_node_compact_id": prepared_inputs["sink_node_compact_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "driver_pin_id": prepared_inputs["driver_pin_id"].to(
                device=device,
                dtype=torch.long,
            ),
        }
        driver_arrival = on_device(driver_arrival)
        driver_slew = on_device(driver_slew)
        buffer_input_cap_by_size = on_device(buffer_input_cap_by_size)
        buffer_slew_axis = on_device(buffer_slew_axis)
        buffer_load_axis = on_device(buffer_load_axis)
        buffer_delay_lut = on_device(buffer_delay_lut)
        buffer_output_slew_lut = on_device(buffer_output_slew_lut)
        segment_retained_upstream_cap = on_device(segment_retained_upstream_cap)
        result = segment_count_transfer_forward_native_cuda(
            net_topo_start=prepared["net_topo_start"],
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"].to(
                device=device,
                dtype=torch.long,
            ),
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_resistance=prepared["edge_resistance"],
            edge_capacitance=prepared["edge_capacitance"],
            node_capacitance=prepared["node_capacitance"],
            edge_to_segment_id=prepared["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=z_value,
            bsu_index=bsu_index,
            load_input_cap=load_input_cap,
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared["segment_sub_resistance_fraction"],
            segment_retained_upstream_cap=segment_retained_upstream_cap,
            sink_node_id=prepared["sink_node_id"],
            sink_net_index=prepared["sink_net_id"],
            sink_node_compact_id=prepared["sink_node_compact_id"],
        )
        ctx.prepared_inputs = prepared
        ctx.max_repeater_count = int(segment_state.max_repeater_count)
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            edge_resistance.detach(),
            edge_capacitance.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
            buffer_input_cap_by_size.detach(),
            buffer_slew_axis.detach(),
            buffer_load_axis.detach(),
            buffer_delay_lut.detach(),
            buffer_output_slew_lut.detach(),
            load_input_cap.detach(),
            result["node_load"].detach(),
            result["node_slew"].detach(),
            result["effective_node_cap"].detach(),
        )
        segment_count = int(z_param.numel())
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared,
                    segment_count,
                    device=device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared,
                    segment_count,
                    device=device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared,
                    segment_count,
                    device=device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"],
            result["sink_net_index"],
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared, device=device),
            ),
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            edge_resistance,
            edge_capacitance,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            buffer_input_cap_by_size,
            buffer_slew_axis,
            buffer_load_axis,
            buffer_delay_lut,
            buffer_output_slew_lut,
            load_input_cap,
            node_load,
            node_slew,
            effective_node_cap,
        ) = ctx.saved_tensors
        del grad_segment_downstream_load
        del grad_segment_parent_arrival
        del grad_segment_parent_slew
        del grad_sink_node_id
        del grad_sink_net_index

        prepared = ctx.prepared_inputs
        dtype = z_param.dtype
        device = z_param.device
        zero_segment_grad = torch.zeros_like(z_param)
        sink_count = int(prepared["sink_node_compact_id"].numel())
        sink_arrival_grad = (
            grad_sink_arrival.to(device=device, dtype=dtype)
            if grad_sink_arrival is not None
            else torch.zeros(sink_count, dtype=dtype, device=device)
        )
        sink_slew_grad = (
            grad_sink_slew.to(device=device, dtype=dtype)
            if grad_sink_slew is not None
            else torch.zeros_like(sink_arrival_grad)
        )
        sink_load_grad = (
            grad_sink_load.to(device=device, dtype=dtype)
            if grad_sink_load is not None
            else torch.zeros_like(sink_arrival_grad)
        )
        root_index = _root_compact_index(prepared, device=device)
        if grad_driver_net_cap is not None:
            root_zeros = torch.zeros(
                int(root_index.numel()),
                dtype=dtype,
                device=device,
            )
            sink_compact = torch.cat(
                [prepared["sink_node_compact_id"], root_index],
                dim=0,
            )
            sink_arrival_grad = torch.cat([sink_arrival_grad, root_zeros], dim=0)
            sink_slew_grad = torch.cat([sink_slew_grad, root_zeros], dim=0)
            sink_load_grad = torch.cat(
                [
                    sink_load_grad,
                    grad_driver_net_cap.to(device=device, dtype=dtype),
                ],
                dim=0,
            )
        else:
            sink_compact = prepared["sink_node_compact_id"]

        cuda_grads = segment_count_transfer_backward_native_cuda(
            net_topo_start=prepared["net_topo_start"],
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_resistance=prepared["edge_resistance"],
            edge_capacitance=prepared["edge_capacitance"],
            edge_to_segment_id=prepared["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=torch.clamp(z_param, 0.0, float(ctx.max_repeater_count)),
            bsu_index=torch.clamp(
                bsu_index_param,
                0.0,
                float(buffer_input_cap_by_size.numel() - 1),
            ),
            load_input_cap=load_input_cap,
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared["segment_sub_resistance_fraction"],
            sink_node_compact_id=sink_compact,
            node_load=node_load,
            node_slew=node_slew,
            effective_node_cap=effective_node_cap,
            grad_segment_delay=(
                grad_segment_delay.to(device=device, dtype=dtype)
                if grad_segment_delay is not None
                else zero_segment_grad
            ),
            grad_segment_output_slew=(
                grad_segment_output_slew.to(device=device, dtype=dtype)
                if grad_segment_output_slew is not None
                else zero_segment_grad
            ),
            grad_segment_upstream_visible_input_cap=(
                grad_segment_upstream_visible_input_cap.to(device=device, dtype=dtype)
                if grad_segment_upstream_visible_input_cap is not None
                else zero_segment_grad
            ),
            grad_sink_arrival=sink_arrival_grad,
            grad_sink_slew=sink_slew_grad,
            grad_sink_load=sink_load_grad,
        )
        with torch.enable_grad():
            bsu_for_load = bsu_index_param.detach().requires_grad_(True)
            load_interp = _interpolate_size_tables(
                torch.clamp(
                    bsu_for_load,
                    0.0,
                    float(per_size_input_cap.shape[1] - 1),
                ),
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
            )["buffer_input_cap"]
            grad_load_bsu = torch.autograd.grad(
                torch.sum(load_interp * cuda_grads["grad_load_input_cap"]),
                bsu_for_load,
                allow_unused=True,
            )[0]
        if grad_load_bsu is None:
            grad_load_bsu = torch.zeros_like(bsu_index_param)
        return (
            cuda_grads["grad_z"]
            * _clamp_backward_mask(
                z_param,
                0.0,
                float(ctx.max_repeater_count),
            ),
            (cuda_grads["grad_bsu"] + grad_load_bsu)
            * _clamp_backward_mask(
                bsu_index_param,
                0.0,
                float(per_size_input_cap.shape[1] - 1),
            ),
            cuda_grads["grad_driver_arrival"],
            cuda_grads["grad_driver_slew"],
            cuda_grads["grad_edge_resistance"],
            cuda_grads["grad_edge_capacitance"],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _interpolate_input_cap_by_size(bsu_index, *, per_size_input_cap):
    if bsu_index.ndim != 1:
        raise ValueError("bsu_index must have shape [segment_count]")
    if per_size_input_cap.ndim != 2:
        raise ValueError("per_size_input_cap must have shape [segment_count, legal_buffer_count]")
    if int(per_size_input_cap.shape[0]) != int(bsu_index.numel()):
        raise ValueError("per_size_input_cap segment dimension does not match bsu_index")
    legal_buffer_count = int(per_size_input_cap.shape[1])
    if legal_buffer_count < 2:
        raise ValueError("per_size_input_cap must contain at least two legal buffer sizes")
    clipped = torch.clamp(bsu_index, 0.0, float(legal_buffer_count - 1))
    lo_index = torch.floor(clipped).to(dtype=torch.long)
    hi_index = torch.clamp(lo_index + 1, max=legal_buffer_count - 1)
    alpha = clipped - lo_index.to(dtype=clipped.dtype)
    lo = torch.gather(per_size_input_cap, 1, lo_index.view(-1, 1)).view(-1)
    hi = torch.gather(per_size_input_cap, 1, hi_index.view(-1, 1)).view(-1)
    return (1.0 - alpha) * lo + alpha * hi


class _SegmentCountCapCudaGpuResidentAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        per_size_input_cap,
        buffer_size_count,
        prepared_inputs,
        segment_state,
        runtime_profile,
    ):
        if not z_param.is_cuda:
            raise RuntimeError("CUDA cap-only transfer requires CUDA segment state")
        device = z_param.device
        dtype = z_param.dtype
        max_repeater_count = int(segment_state.max_repeater_count)
        buffer_size_count = int(buffer_size_count)
        per_size_input_cap = per_size_input_cap.to(device=device, dtype=dtype)
        if buffer_size_count < 2 or buffer_size_count > int(per_size_input_cap.shape[1]):
            raise ValueError("buffer_size_count must be within per_size_input_cap bounds")
        z_value = torch.clamp(z_param, 0.0, float(max_repeater_count))
        bsu_index = torch.clamp(
            bsu_index_param,
            0.0,
            float(buffer_size_count - 1),
        )
        load_input_cap = _interpolate_input_cap_by_size(
            bsu_index,
            per_size_input_cap=per_size_input_cap,
        )
        prepared = {
            "net_topo_start": prepared_inputs["net_topo_start"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_start": prepared_inputs["edge_start"].to(
                device=device,
                dtype=torch.long,
            ),
            "edge_parent_compact_id": prepared_inputs[
                "edge_parent_compact_id"
            ].to(device=device, dtype=torch.long),
            "edge_child_compact_id": prepared_inputs[
                "edge_child_compact_id"
            ].to(device=device, dtype=torch.long),
            "edge_capacitance": prepared_inputs["edge_capacitance"].to(
                device=device,
                dtype=dtype,
            ),
            "node_capacitance": prepared_inputs["node_capacitance"].to(
                device=device,
                dtype=dtype,
            ),
            "edge_to_segment_id": prepared_inputs["edge_to_segment_id"].to(
                device=device,
                dtype=torch.long,
            ),
            "parent_cap_fraction": prepared_inputs["parent_cap_fraction"].to(
                device=device,
                dtype=dtype,
            ),
            "child_cap_fraction": prepared_inputs["child_cap_fraction"].to(
                device=device,
                dtype=dtype,
            ),
        }
        result = segment_count_cap_forward_native_cuda(
            net_topo_start=prepared["net_topo_start"],
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_capacitance=prepared["edge_capacitance"],
            node_capacitance=prepared["node_capacitance"],
            edge_to_segment_id=prepared["edge_to_segment_id"],
            z_value=z_value,
            load_input_cap=load_input_cap,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
        )
        root_index = _root_compact_index(prepared, device=device)
        ctx.prepared_inputs = prepared
        ctx.max_repeater_count = max_repeater_count
        ctx.buffer_size_count = buffer_size_count
        ctx.runtime_profile = runtime_profile
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            per_size_input_cap.detach(),
            load_input_cap.detach(),
            result["node_load"].detach(),
        )
        return result["node_load"].index_select(0, root_index)

    @staticmethod
    def backward(ctx, grad_driver_net_cap):
        (
            z_param,
            bsu_index_param,
            per_size_input_cap,
            load_input_cap,
            node_load,
        ) = ctx.saved_tensors
        if grad_driver_net_cap is None:
            return (
                torch.zeros_like(z_param),
                torch.zeros_like(bsu_index_param),
                None,
                None,
                None,
                None,
                None,
            )
        dtype = z_param.dtype
        device = z_param.device
        prepared = ctx.prepared_inputs
        cuda_grads = segment_count_cap_backward_native_cuda(
            net_topo_start=prepared["net_topo_start"],
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_capacitance=prepared["edge_capacitance"],
            edge_to_segment_id=prepared["edge_to_segment_id"],
            z_value=torch.clamp(
                z_param,
                0.0,
                float(ctx.max_repeater_count),
            ),
            load_input_cap=load_input_cap,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
            node_load=node_load,
            grad_driver_net_cap=grad_driver_net_cap.to(device=device, dtype=dtype),
        )
        if isinstance(ctx.runtime_profile, dict):
            ctx.runtime_profile["native_cap_backward_invocation_count"] = int(
                ctx.runtime_profile.get("native_cap_backward_invocation_count", 0)
            ) + 1
            ctx.runtime_profile["native_cap_backward_temporary_allocation_bytes"] = (
                int(
                    ctx.runtime_profile.get(
                        "native_cap_backward_temporary_allocation_bytes",
                        0,
                    )
                )
                + int(cuda_grads.get("temporary_allocation_bytes", 0))
            )
            ctx.runtime_profile["native_cap_backward_temporary_allocation_cpu_ms"] = (
                float(
                    ctx.runtime_profile.get(
                        "native_cap_backward_temporary_allocation_cpu_ms",
                        0.0,
                    )
                )
                + float(cuda_grads.get("temporary_allocation_cpu_ms", 0.0))
            )
        with torch.enable_grad():
            bsu_for_load = bsu_index_param.detach().requires_grad_(True)
            load_interp = _interpolate_input_cap_by_size(
                torch.clamp(
                    bsu_for_load,
                    0.0,
                    float(per_size_input_cap.shape[1] - 1),
                ),
                per_size_input_cap=per_size_input_cap,
            )
            grad_load_bsu = torch.autograd.grad(
                torch.sum(load_interp * cuda_grads["grad_load_input_cap"]),
                bsu_for_load,
                allow_unused=True,
            )[0]
        if grad_load_bsu is None:
            grad_load_bsu = torch.zeros_like(bsu_index_param)
        return (
            cuda_grads["grad_z"]
            * _clamp_backward_mask(
                z_param,
                0.0,
                float(ctx.max_repeater_count),
            ),
            grad_load_bsu
            * _clamp_backward_mask(
                bsu_index_param,
                0.0,
                float(per_size_input_cap.shape[1] - 1),
            ),
            None,
            None,
            None,
            None,
            None,
        )


def segment_count_driver_cap_native_cuda_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    buffer_input_cap_by_size,
    runtime_profile=None,
):
    dtype = segment_state.z_param.dtype
    device = segment_state.z_param.device
    if device.type != "cuda":
        raise RuntimeError("segment-count cap-only transfer requires a CUDA segment state")
    per_size_input_cap = torch.as_tensor(
        per_size_input_cap,
        dtype=dtype,
        device=device,
    )
    buffer_size_count = int(torch.as_tensor(buffer_input_cap_by_size).numel())
    driver_net_cap = _SegmentCountCapCudaGpuResidentAutograd.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        per_size_input_cap,
        buffer_size_count,
        prepared_inputs,
        segment_state,
        runtime_profile,
    )
    return {
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=device,
            dtype=torch.long,
        ),
        "driver_net_cap": driver_net_cap,
        "metadata": {
            "backend": "cpp_cuda_segment_count_cap_autograd",
            "forward": "cpp_cuda_segment_count_cap_forward",
            "backward": "cpp_cuda_segment_count_cap_backward",
            "gpu_resident": True,
        },
    }


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
            "buffer_slew_limits": _buffer_limit_tensor(
                tables.get("buffer_slew_limits"), tables["input_cap_by_size"]
            ),
            "buffer_cap_limits": _buffer_limit_tensor(
                tables.get("buffer_cap_limits"), tables["input_cap_by_size"]
            ),
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
    result = {
        key: torch.as_tensor(buffer_device_lut[key], dtype=dtype, device=device)
        for key in required
    }
    for key in ("buffer_slew_limits", "buffer_cap_limits"):
        result[key] = _buffer_limit_tensor(
            buffer_device_lut.get(key), result["buffer_input_cap_by_size"]
        )
    return result


def segment_count_forward_native_segment_transfer_recompute_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival,
    driver_slew,
    buffer_device_lut,
    segment_retained_upstream_cap=None,
):
    dtype = segment_state.z_param.dtype
    device = segment_state.z_param.device
    per_size_input_cap = torch.as_tensor(per_size_input_cap, dtype=dtype, device=device)
    per_size_delay = torch.as_tensor(per_size_delay, dtype=dtype, device=device)
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=dtype,
        device=device,
    )
    driver_arrival = torch.as_tensor(driver_arrival, dtype=dtype, device=device)
    driver_slew = torch.as_tensor(driver_slew, dtype=dtype, device=device)
    retained_cap = (
        torch.zeros((int(segment_state.z_param.numel()),), dtype=dtype, device=device)
        if segment_retained_upstream_cap is None
        else torch.as_tensor(segment_retained_upstream_cap, dtype=dtype, device=device)
    )
    if int(retained_cap.numel()) != int(segment_state.z_param.numel()):
        raise ValueError("segment_retained_upstream_cap length must match segment state")
    lut_tensors = _buffer_device_tensors(buffer_device_lut, dtype=dtype, device=device)
    outputs = _SegmentCountTransferNativeRecomputeAutograd.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        retained_cap,
        lut_tensors["buffer_input_cap_by_size"],
        lut_tensors["buffer_slew_axis"],
        lut_tensors["buffer_load_axis"],
        lut_tensors["buffer_delay_lut"],
        lut_tensors["buffer_output_slew_lut"],
        prepared_inputs,
        segment_state,
        buffer_device_lut,
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "segment_downstream_load": outputs[3],
        "segment_parent_arrival": outputs[4],
        "segment_parent_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "sink_node_id": outputs[9],
        "sink_net_index": outputs[10],
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=outputs[11].device,
            dtype=torch.long,
        ),
        "driver_net_cap": outputs[11],
        "metadata": {
            "backend": "cpp_cpu_segment_transfer_recompute_autograd",
            "net_count": int(prepared_inputs["net_ids"].numel()),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
            "segment_transfer_backend": "segment_transfer_native",
            "backward": "prepared_python_segment_transfer_native_recompute",
        },
    }


def segment_count_forward_native_segment_transfer_explicit_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival,
    driver_slew,
    buffer_device_lut,
    segment_retained_upstream_cap=None,
    edge_resistance_override=None,
    edge_capacitance_override=None,
    canonical_equal_spacing=False,
):
    dtype = segment_state.z_param.dtype
    device = segment_state.z_param.device
    per_size_input_cap = torch.as_tensor(per_size_input_cap, dtype=dtype, device=device)
    per_size_delay = torch.as_tensor(per_size_delay, dtype=dtype, device=device)
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=dtype,
        device=device,
    )
    driver_arrival = torch.as_tensor(driver_arrival, dtype=dtype, device=device)
    driver_slew = torch.as_tensor(driver_slew, dtype=dtype, device=device)
    retained_cap = (
        torch.zeros((int(segment_state.z_param.numel()),), dtype=dtype, device=device)
        if segment_retained_upstream_cap is None
        else torch.as_tensor(segment_retained_upstream_cap, dtype=dtype, device=device)
    )
    if int(retained_cap.numel()) != int(segment_state.z_param.numel()):
        raise ValueError("segment_retained_upstream_cap length must match segment state")
    lut_tensors = _buffer_device_tensors(buffer_device_lut, dtype=dtype, device=device)
    edge_resistance = torch.as_tensor(
        prepared_inputs["edge_resistance"]
        if edge_resistance_override is None
        else edge_resistance_override,
        dtype=dtype,
        device=device,
    )
    edge_capacitance = torch.as_tensor(
        prepared_inputs["edge_capacitance"]
        if edge_capacitance_override is None
        else edge_capacitance_override,
        dtype=dtype,
        device=device,
    )
    if int(edge_resistance.numel()) != int(prepared_inputs["edge_resistance"].numel()):
        raise ValueError("edge_resistance_override length must match prepared edges")
    if int(edge_capacitance.numel()) != int(prepared_inputs["edge_capacitance"].numel()):
        raise ValueError("edge_capacitance_override length must match prepared edges")
    runtime_prepared_inputs = (
        _prepared_with_canonical_equal_spacing(prepared_inputs)
        if canonical_equal_spacing
        else prepared_inputs
    )
    outputs = _SegmentCountTransferNativeExplicitAutograd.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        driver_arrival,
        driver_slew,
        edge_resistance,
        edge_capacitance,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        retained_cap,
        lut_tensors["buffer_input_cap_by_size"],
        lut_tensors["buffer_slew_axis"],
        lut_tensors["buffer_load_axis"],
        lut_tensors["buffer_delay_lut"],
        lut_tensors["buffer_output_slew_lut"],
        runtime_prepared_inputs,
        segment_state,
        prepared_inputs["node_capacitance"],
        lut_tensors["buffer_slew_limits"],
        lut_tensors["buffer_cap_limits"],
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "segment_downstream_load": outputs[3],
        "segment_parent_arrival": outputs[4],
        "segment_parent_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "sink_node_id": outputs[9],
        "sink_net_index": outputs[10],
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=outputs[11].device,
            dtype=torch.long,
        ),
        "driver_net_cap": outputs[11],
        "buffer_slew_violation": outputs[12],
        "buffer_cap_violation": outputs[13],
        "metadata": {
            "backend": "cpp_cpu_segment_transfer_explicit_autograd",
            "net_count": int(prepared_inputs["net_ids"].numel()),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
            "segment_transfer_backend": "segment_transfer_native",
            "backward": "cpp_cpu_segment_transfer_explicit",
            "live_edge_override": bool(
                edge_resistance_override is not None
                and edge_capacitance_override is not None
            ),
            "canonical_equal_spacing": bool(canonical_equal_spacing),
        },
    }


def segment_count_forward_cuda_segment_transfer_explicit_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival,
    driver_slew,
    buffer_device_lut,
    segment_retained_upstream_cap=None,
    edge_resistance_override=None,
    edge_capacitance_override=None,
    canonical_equal_spacing=False,
):
    dtype = segment_state.z_param.dtype
    device = segment_state.z_param.device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cpp_cuda_segment_transfer_explicit_autograd")
    per_size_input_cap = torch.as_tensor(per_size_input_cap, dtype=dtype, device=device)
    per_size_delay = torch.as_tensor(per_size_delay, dtype=dtype, device=device)
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=dtype,
        device=device,
    )
    driver_arrival = torch.as_tensor(driver_arrival, dtype=dtype, device=device)
    driver_slew = torch.as_tensor(driver_slew, dtype=dtype, device=device)
    retained_cap = (
        torch.zeros((int(segment_state.z_param.numel()),), dtype=dtype, device=device)
        if segment_retained_upstream_cap is None
        else torch.as_tensor(segment_retained_upstream_cap, dtype=dtype, device=device)
    )
    if int(retained_cap.numel()) != int(segment_state.z_param.numel()):
        raise ValueError("segment_retained_upstream_cap length must match segment state")
    lut_tensors = _buffer_device_tensors(buffer_device_lut, dtype=dtype, device=device)
    edge_resistance = torch.as_tensor(
        prepared_inputs["edge_resistance"]
        if edge_resistance_override is None
        else edge_resistance_override,
        dtype=dtype,
        device=device,
    )
    edge_capacitance = torch.as_tensor(
        prepared_inputs["edge_capacitance"]
        if edge_capacitance_override is None
        else edge_capacitance_override,
        dtype=dtype,
        device=device,
    )
    if int(edge_resistance.numel()) != int(prepared_inputs["edge_resistance"].numel()):
        raise ValueError("edge_resistance_override length must match prepared edges")
    if int(edge_capacitance.numel()) != int(prepared_inputs["edge_capacitance"].numel()):
        raise ValueError("edge_capacitance_override length must match prepared edges")
    runtime_prepared_inputs = (
        _prepared_with_canonical_equal_spacing(prepared_inputs)
        if canonical_equal_spacing
        else prepared_inputs
    )
    autograd_op = (
        _SegmentCountTransferCudaGpuResidentAutograd
        if device.type == "cuda"
        else _SegmentCountTransferCudaExplicitAutograd
    )
    outputs = autograd_op.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        driver_arrival,
        driver_slew,
        edge_resistance,
        edge_capacitance,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        retained_cap,
        lut_tensors["buffer_input_cap_by_size"],
        lut_tensors["buffer_slew_axis"],
        lut_tensors["buffer_load_axis"],
        lut_tensors["buffer_delay_lut"],
        lut_tensors["buffer_output_slew_lut"],
        runtime_prepared_inputs,
        segment_state,
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "segment_downstream_load": outputs[3],
        "segment_parent_arrival": outputs[4],
        "segment_parent_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "sink_node_id": outputs[9],
        "sink_net_index": outputs[10],
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=outputs[11].device,
            dtype=torch.long,
        ),
        "driver_net_cap": outputs[11],
        "metadata": {
            "backend": "cpp_cuda_segment_transfer_explicit_autograd",
            "net_count": int(prepared_inputs["net_ids"].numel()),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
            "segment_transfer_backend": "segment_transfer_native",
            "forward": "cpp_cuda_segment_count_transfer_forward",
            "backward": "cpp_cuda_segment_count_transfer_explicit",
            "backward_fallback_reason": None,
            "gpu_resident": bool(device.type == "cuda"),
            "live_edge_override": bool(
                edge_resistance_override is not None
                and edge_capacitance_override is not None
            ),
            "canonical_equal_spacing": bool(canonical_equal_spacing),
        },
    }

class _SegmentCountForwardNativeExplicitAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_param,
        bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        prepared_inputs,
        segment_state,
    ):
        runtime_state = replace(
            segment_state,
            z_param=z_param,
            bsu_index_param=bsu_index_param,
        )
        buffer_tensors = _interpolate_size_tables(
            runtime_state.bsu_index().to(dtype=z_param.dtype, device=z_param.device),
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        result = segment_count_forward_native(
            net_topo_start=prepared_inputs["net_topo_start"],
            flat_topo_node_id=prepared_inputs["flat_topo_node_id"],
            edge_start=prepared_inputs["edge_start"],
            edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"],
            edge_child_compact_id=prepared_inputs["edge_child_compact_id"],
            edge_resistance=prepared_inputs["edge_resistance"],
            edge_capacitance=prepared_inputs["edge_capacitance"],
            node_capacitance=prepared_inputs["node_capacitance"],
            edge_to_segment_id=prepared_inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=runtime_state.z_value(),
            buffer_input_cap=buffer_tensors["buffer_input_cap"],
            buffer_delay=buffer_tensors["buffer_delay"],
            buffer_output_slew=buffer_tensors["buffer_output_slew"],
            parent_cap_fraction=prepared_inputs["parent_cap_fraction"],
            child_cap_fraction=prepared_inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared_inputs[
                "segment_sub_resistance_fraction"
            ],
            sink_node_id=prepared_inputs["sink_node_id"],
            sink_net_index=prepared_inputs["sink_net_id"],
            sink_node_compact_id=prepared_inputs["sink_node_compact_id"],
        )
        ctx.prepared_inputs = prepared_inputs
        ctx.segment_state = segment_state
        ctx.save_for_backward(
            z_param.detach(),
            bsu_index_param.detach(),
            driver_arrival.detach(),
            driver_slew.detach(),
            per_size_input_cap.detach(),
            per_size_delay.detach(),
            per_size_output_slew.detach(),
            buffer_tensors["buffer_input_cap"].detach(),
            buffer_tensors["buffer_delay"].detach(),
            buffer_tensors["buffer_output_slew"].detach(),
            result["node_load"].detach(),
            result["node_slew"].detach(),
            result["effective_node_cap"].detach(),
        )
        return (
            result["segment_delay"],
            result["segment_output_slew"],
            result["segment_upstream_visible_input_cap"],
            result["node_load"].index_select(
                0,
                _segment_child_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_load"].device,
                ),
            ),
            result["node_arrival"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_arrival"].device,
                ),
            ),
            result["node_slew"].index_select(
                0,
                _segment_parent_compact_index(
                    prepared_inputs,
                    int(z_param.numel()),
                    device=result["node_slew"].device,
                ),
            ),
            result["sink_arrival"],
            result["sink_slew"],
            result["sink_load"],
            result["sink_node_id"],
            result["sink_net_index"],
            result["node_load"].index_select(
                0,
                _root_compact_index(prepared_inputs, device=result["node_load"].device),
            ),
        )

    @staticmethod
    def backward(
        ctx,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_visible_input_cap,
        grad_segment_downstream_load,
        grad_segment_parent_arrival,
        grad_segment_parent_slew,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_node_id,
        grad_sink_net_index,
        grad_driver_net_cap,
    ):
        (
            z_param,
            bsu_index_param,
            driver_arrival,
            driver_slew,
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            node_load,
            node_slew,
            effective_node_cap,
        ) = ctx.saved_tensors
        prepared = ctx.prepared_inputs
        dtype = z_param.dtype
        device = z_param.device
        num_segments = int(z_param.numel())
        zero_segment_grad = torch.zeros(num_segments, dtype=dtype, device=device)
        sink_load_grad = (
            grad_sink_load.to(dtype=dtype, device=device)
            if grad_sink_load is not None
            else torch.zeros(
                int(prepared["sink_node_compact_id"].numel()),
                dtype=dtype,
                device=device,
            )
        )
        sink_arrival_grad = (
            grad_sink_arrival.to(dtype=dtype, device=device)
            if grad_sink_arrival is not None
            else torch.zeros(
                int(prepared["sink_node_compact_id"].numel()),
                dtype=dtype,
                device=device,
            )
        )
        sink_slew_grad = (
            grad_sink_slew.to(dtype=dtype, device=device)
            if grad_sink_slew is not None
            else torch.zeros(
                int(prepared["sink_node_compact_id"].numel()),
                dtype=dtype,
                device=device,
            )
        )
        if grad_driver_net_cap is not None and int(grad_driver_net_cap.numel()) > 0:
            root_index = _root_compact_index(prepared, device=device)
            root_as_sink = torch.cat(
                [
                    prepared["sink_node_compact_id"].to(device=device, dtype=torch.long),
                    root_index,
                ],
                dim=0,
            )
            sink_load_grad = torch.cat(
                [
                    sink_load_grad,
                    grad_driver_net_cap.to(dtype=dtype, device=device),
                ],
                dim=0,
            )
            root_zeros = torch.zeros(
                int(root_index.numel()),
                dtype=dtype,
                device=device,
            )
            sink_arrival_grad = torch.cat([sink_arrival_grad, root_zeros], dim=0)
            sink_slew_grad = torch.cat([sink_slew_grad, root_zeros], dim=0)
        else:
            root_as_sink = prepared["sink_node_compact_id"]
        cpp_grads = segment_count_backward_native(
            net_topo_start=prepared["net_topo_start"],
            edge_start=prepared["edge_start"],
            edge_parent_compact_id=prepared["edge_parent_compact_id"],
            edge_child_compact_id=prepared["edge_child_compact_id"],
            edge_resistance=prepared["edge_resistance"].to(dtype=dtype, device=device),
            edge_capacitance=prepared["edge_capacitance"].to(dtype=dtype, device=device),
            edge_to_segment_id=prepared["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=torch.clamp(z_param, 0.0, float(ctx.segment_state.max_repeater_count)),
            bsu_index=torch.clamp(
                bsu_index_param,
                0.0,
                float(per_size_delay.shape[1] - 1),
            ),
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            buffer_input_cap=buffer_input_cap,
            buffer_delay=buffer_delay,
            buffer_output_slew=buffer_output_slew,
            parent_cap_fraction=prepared["parent_cap_fraction"],
            child_cap_fraction=prepared["child_cap_fraction"],
            segment_sub_resistance_fraction=prepared["segment_sub_resistance_fraction"],
            sink_node_compact_id=root_as_sink,
            node_load=node_load,
            node_slew=node_slew,
            effective_node_cap=effective_node_cap,
            grad_segment_delay=(
                grad_segment_delay.to(dtype=dtype, device=device)
                if grad_segment_delay is not None
                else zero_segment_grad
            ),
            grad_segment_output_slew=(
                grad_segment_output_slew.to(dtype=dtype, device=device)
                if grad_segment_output_slew is not None
                else zero_segment_grad
            ),
            grad_segment_upstream_visible_input_cap=(
                grad_segment_upstream_visible_input_cap.to(dtype=dtype, device=device)
                if grad_segment_upstream_visible_input_cap is not None
                else zero_segment_grad
            ),
            grad_sink_arrival=(
                sink_arrival_grad
            ),
            grad_sink_slew=(
                sink_slew_grad
            ),
            grad_sink_load=(
                sink_load_grad
            ),
        )

        return (
            cpp_grads["grad_z"] * _clamp_backward_mask(
                z_param,
                0.0,
                float(ctx.segment_state.max_repeater_count),
            ),
            cpp_grads["grad_bsu"] * _clamp_backward_mask(
                bsu_index_param,
                0.0,
                float(per_size_delay.shape[1] - 1),
            ),
            cpp_grads["grad_driver_arrival"],
            cpp_grads["grad_driver_slew"],
            None,
            None,
            None,
            None,
            None,
        )


def segment_count_forward_native_explicit_autograd(
    prepared_inputs,
    *,
    segment_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    driver_arrival,
    driver_slew,
):
    per_size_input_cap = torch.as_tensor(
        per_size_input_cap,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    per_size_delay = torch.as_tensor(
        per_size_delay,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    per_size_output_slew = torch.as_tensor(
        per_size_output_slew,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    driver_arrival = torch.as_tensor(
        driver_arrival,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    driver_slew = torch.as_tensor(
        driver_slew,
        dtype=segment_state.z_param.dtype,
        device=segment_state.z_param.device,
    )
    outputs = _SegmentCountForwardNativeExplicitAutograd.apply(
        segment_state.z_param,
        segment_state.bsu_index_param,
        driver_arrival,
        driver_slew,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        prepared_inputs,
        segment_state,
    )
    return {
        "segment_delay": outputs[0],
        "segment_output_slew": outputs[1],
        "segment_upstream_visible_input_cap": outputs[2],
        "segment_downstream_load": outputs[3],
        "segment_parent_arrival": outputs[4],
        "segment_parent_slew": outputs[5],
        "sink_arrival": outputs[6],
        "sink_slew": outputs[7],
        "sink_load": outputs[8],
        "sink_node_id": outputs[9],
        "sink_net_index": outputs[10],
        "driver_pin_id": prepared_inputs["driver_pin_id"].to(
            device=outputs[11].device,
            dtype=torch.long,
        ),
        "driver_net_cap": outputs[11],
        "metadata": {
            "backend": "cpp_cpu_explicit_autograd",
            "net_count": int(prepared_inputs["net_ids"].numel()),
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
            "explicit_backward_z_range": "[0, Nmax]",
        },
    }
