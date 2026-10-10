import importlib

import torch

from .batched import batched_segment_transfer


def segment_transfer_forward_native_cpu(
    *,
    input_arrival,
    input_slew,
    downstream_load,
    edge_resistance,
    edge_capacitance,
    repeater_count,
    split_fractions,
    bsu_index,
    upstream_retained_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
):
    native = importlib.import_module(
        "dreamplace.ops.net_subgraph_timing.net_subgraph_timing_cpp"
    )
    return native.segment_transfer_forward(
        input_arrival.contiguous(),
        input_slew.contiguous(),
        downstream_load.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        repeater_count.contiguous(),
        split_fractions.contiguous(),
        bsu_index.contiguous(),
        upstream_retained_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
    )


def segment_transfer_forward_native_cuda(
    *,
    input_arrival,
    input_slew,
    downstream_load,
    edge_resistance,
    edge_capacitance,
    repeater_count,
    split_fractions,
    bsu_index,
    upstream_retained_cap,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
):
    native = importlib.import_module(
        "dreamplace.ops.net_subgraph_timing.net_subgraph_timing_cuda"
    )
    outputs = native.segment_transfer_forward(
        input_arrival.contiguous(),
        input_slew.contiguous(),
        downstream_load.contiguous(),
        edge_resistance.contiguous(),
        edge_capacitance.contiguous(),
        repeater_count.to(dtype=torch.long).contiguous(),
        split_fractions.contiguous(),
        bsu_index.contiguous(),
        upstream_retained_cap.contiguous(),
        buffer_input_cap_by_size.contiguous(),
        buffer_slew_axis.contiguous(),
        buffer_load_axis.contiguous(),
        buffer_delay_lut.contiguous(),
        buffer_output_slew_lut.contiguous(),
    )
    return {
        "upstream_visible_load": outputs[0],
        "segment_delay": outputs[1],
        "output_arrival": outputs[2],
        "output_slew": outputs[3],
        "first_buffer_input_slew": outputs[4],
        "first_buffer_output_load": outputs[5],
        "first_buffer_delay": outputs[6],
        "first_buffer_output_slew": outputs[7],
    }


def segment_transfer_forward_native(**kwargs):
    input_arrival = kwargs["input_arrival"]
    if torch.is_tensor(input_arrival) and input_arrival.is_cuda:
        return segment_transfer_forward_native_cuda(**kwargs)
    return segment_transfer_forward_native_cpu(**kwargs)


class _SegmentTransferNativeRecomputeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_arrival,
        input_slew,
        downstream_load,
        edge_resistance,
        edge_capacitance,
        repeater_count,
        split_fractions,
        bsu_index,
        upstream_retained_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
    ):
        result = segment_transfer_forward_native(
            input_arrival=input_arrival,
            input_slew=input_slew,
            downstream_load=downstream_load,
            edge_resistance=edge_resistance,
            edge_capacitance=edge_capacitance,
            repeater_count=repeater_count,
            split_fractions=split_fractions,
            bsu_index=bsu_index,
            upstream_retained_cap=upstream_retained_cap,
            buffer_input_cap_by_size=buffer_input_cap_by_size,
            buffer_slew_axis=buffer_slew_axis,
            buffer_load_axis=buffer_load_axis,
            buffer_delay_lut=buffer_delay_lut,
            buffer_output_slew_lut=buffer_output_slew_lut,
        )
        ctx.save_for_backward(
            input_arrival.detach(),
            input_slew.detach(),
            downstream_load.detach(),
            edge_resistance.detach(),
            edge_capacitance.detach(),
            repeater_count.detach(),
            split_fractions.detach(),
            bsu_index.detach(),
            upstream_retained_cap.detach(),
            buffer_input_cap_by_size.detach(),
            buffer_slew_axis.detach(),
            buffer_load_axis.detach(),
            buffer_delay_lut.detach(),
            buffer_output_slew_lut.detach(),
        )
        return (
            result["upstream_visible_load"],
            result["segment_delay"],
            result["output_arrival"],
            result["output_slew"],
            result["first_buffer_input_slew"],
            result["first_buffer_output_load"],
            result["first_buffer_delay"],
            result["first_buffer_output_slew"],
        )

    @staticmethod
    def backward(
        ctx,
        grad_upstream_visible_load,
        grad_segment_delay,
        grad_output_arrival,
        grad_output_slew,
        grad_first_buffer_input_slew,
        grad_first_buffer_output_load,
        grad_first_buffer_delay,
        grad_first_buffer_output_slew,
    ):
        (
            input_arrival,
            input_slew,
            downstream_load,
            edge_resistance,
            edge_capacitance,
            repeater_count,
            split_fractions,
            bsu_index,
            upstream_retained_cap,
            buffer_input_cap_by_size,
            buffer_slew_axis,
            buffer_load_axis,
            buffer_delay_lut,
            buffer_output_slew_lut,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            input_arrival = input_arrival.detach().requires_grad_(True)
            input_slew = input_slew.detach().requires_grad_(True)
            downstream_load = downstream_load.detach().requires_grad_(True)
            edge_resistance = edge_resistance.detach().requires_grad_(True)
            edge_capacitance = edge_capacitance.detach().requires_grad_(True)
            bsu_index = bsu_index.detach().requires_grad_(True)
            upstream_retained_cap = upstream_retained_cap.detach().requires_grad_(True)
            recompute = batched_segment_transfer(
                input_arrival=input_arrival,
                input_slew=input_slew,
                downstream_load=downstream_load,
                edge_resistance=edge_resistance,
                edge_capacitance=edge_capacitance,
                repeater_count=repeater_count,
                split_fractions=split_fractions,
                bsu_index=bsu_index,
                upstream_retained_cap=upstream_retained_cap,
                buffer_input_cap_by_size=buffer_input_cap_by_size,
                buffer_slew_axis=buffer_slew_axis,
                buffer_load_axis=buffer_load_axis,
                buffer_delay_lut=buffer_delay_lut,
                buffer_output_slew_lut=buffer_output_slew_lut,
            )
            objective = torch.zeros((), dtype=input_slew.dtype, device=input_slew.device)

            def add_vjp(key, grad):
                nonlocal objective
                if grad is None:
                    return
                objective = objective + torch.sum(recompute[key] * grad.to(recompute[key]))

            add_vjp("upstream_visible_load", grad_upstream_visible_load)
            add_vjp("segment_delay", grad_segment_delay)
            add_vjp("output_arrival", grad_output_arrival)
            add_vjp("output_slew", grad_output_slew)
            add_vjp("first_buffer_input_slew", grad_first_buffer_input_slew)
            add_vjp("first_buffer_output_load", grad_first_buffer_output_load)
            add_vjp("first_buffer_delay", grad_first_buffer_delay)
            add_vjp("first_buffer_output_slew", grad_first_buffer_output_slew)
            grads = torch.autograd.grad(
                objective,
                (
                    input_arrival,
                    input_slew,
                    downstream_load,
                    edge_resistance,
                    edge_capacitance,
                    bsu_index,
                    upstream_retained_cap,
                ),
                allow_unused=True,
            )
        (
            grad_input_arrival,
            grad_input_slew,
            grad_downstream_load,
            grad_edge_resistance,
            grad_edge_capacitance,
            grad_bsu_index,
            grad_upstream_retained_cap,
        ) = grads
        return (
            grad_input_arrival,
            grad_input_slew,
            grad_downstream_load,
            grad_edge_resistance,
            grad_edge_capacitance,
            None,
            None,
            grad_bsu_index,
            grad_upstream_retained_cap,
            None,
            None,
            None,
            None,
            None,
        )


def segment_transfer_native_recompute_autograd(**kwargs):
    outputs = _SegmentTransferNativeRecomputeAutograd.apply(
        kwargs["input_arrival"],
        kwargs["input_slew"],
        kwargs["downstream_load"],
        kwargs["edge_resistance"],
        kwargs["edge_capacitance"],
        kwargs["repeater_count"],
        kwargs["split_fractions"],
        kwargs["bsu_index"],
        kwargs["upstream_retained_cap"],
        kwargs["buffer_input_cap_by_size"],
        kwargs["buffer_slew_axis"],
        kwargs["buffer_load_axis"],
        kwargs["buffer_delay_lut"],
        kwargs["buffer_output_slew_lut"],
    )
    return {
        "upstream_visible_load": outputs[0],
        "segment_delay": outputs[1],
        "output_arrival": outputs[2],
        "output_slew": outputs[3],
        "first_buffer_input_slew": outputs[4],
        "first_buffer_output_load": outputs[5],
        "first_buffer_delay": outputs[6],
        "first_buffer_output_slew": outputs[7],
        "metadata": {
            "backend": "cpp_cpu_recompute_autograd",
            "backward": "python_recompute_autograd",
        },
    }
