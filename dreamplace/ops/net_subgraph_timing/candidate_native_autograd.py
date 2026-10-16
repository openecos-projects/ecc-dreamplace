"""Scalar and four-strength candidate derivatives through one CPU or CUDA reverse pass."""

import importlib

import torch
from torch.autograd import Function

from .candidate_rc_autograd import (
    candidate_node_coverage,
    candidate_rc_gradients_from_adjoint,
    mask_unvisited_candidate_adjoints,
)

CANDIDATE_OUTPUT_KEYS = (
    "lout",
    "lin",
    "effective_node_cap",
    "arrival_in",
    "arrival_out",
    "slew_in",
    "slew_out",
    "sink_arrival",
    "sink_slew",
    "sink_load",
    "sink_cap",
    "sink_net_delay",
    "sink_net_impulse",
)


class CandidateNativeAutogradFunction(Function):
    @staticmethod
    def forward(
        ctx,
        topo,
        starts,
        parents,
        child_starts,
        children,
        resistance,
        node_cap,
        edge_cap,
        driver_arrival,
        driver_slew,
        candidate_nodes,
        activation,
        input_cap,
        delay,
        output_slew,
        sinks,
        sink_nets,
    ):
        from .net_subgraph_timing import net_subgraph_forward_cuda, net_subgraph_forward_native

        forward = net_subgraph_forward_cuda if node_cap.is_cuda else net_subgraph_forward_native

        typed = activation.ndim == 2
        ctx.typed = typed
        ctx.has_edge_cap = bool(edge_cap.numel())
        ctx.set_materialize_grads(False)
        if typed:
            selected = activation.detach().argmax(dim=1, keepdim=True)
            bu = activation.sum(dim=1)
            cap = input_cap.gather(1, selected).squeeze(1)
            delay_value = delay.gather(1, selected).squeeze(1)
            slew_value = output_slew.gather(1, selected).squeeze(1)
        else:
            selected = candidate_nodes
            bu, cap, delay_value, slew_value = activation, input_cap, delay, output_slew
        result = forward(
            net_flat_topo_sort=topo,
            net_flat_topo_sort_start=starts,
            pin_fa=parents,
            flat_pin_to_start=child_starts,
            flat_pin_to=children,
            edge_resistance=resistance,
            node_capacitance=node_cap,
            edge_capacitance=edge_cap,
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            candidate_node_id=candidate_nodes,
            candidate_bu=bu,
            buffer_input_cap=cap,
            buffer_delay=delay_value,
            buffer_output_slew=slew_value,
            sink_node_id=sinks,
            sink_net_index=sink_nets,
        )
        if not node_cap.is_cuda:
            candidate_index = torch.full_like(parents, -1, dtype=torch.long)
            candidate_index[candidate_nodes.long()] = torch.arange(
                candidate_nodes.numel(), device=parents.device
            )
            result["candidate_index_by_node"] = candidate_index
        ctx.save_for_backward(
            topo.long(),
            starts.long(),
            parents.long(),
            child_starts.long(),
            children.long(),
            resistance,
            candidate_nodes.long(),
            activation,
            input_cap,
            delay,
            output_slew,
            sinks.long(),
            sink_nets.long(),
            result["candidate_index_by_node"],
            result["lin"],
            result["lout"],
            result["slew_in"],
            result["slew_out"],
            selected,
            bu,
            cap,
            delay_value,
            slew_value,
            topo,
            starts,
            parents,
            child_starts,
            children,
            candidate_nodes,
            sinks,
            sink_nets,
        )
        return tuple(result.get(key, node_cap.new_empty(0)) for key in CANDIDATE_OUTPUT_KEYS)

    @staticmethod
    def backward(ctx, *output_grads):
        (
            topo,
            starts,
            parents,
            child_starts,
            children,
            resistance,
            candidate_nodes,
            activation,
            input_cap,
            delay,
            output_slew,
            sinks,
            sink_nets,
            candidate_index,
            lin,
            lout,
            slew_in,
            slew_out,
            selected,
            bu,
            cap,
            delay_value,
            slew_value,
            *_original_indices,
        ) = ctx.saved_tensors
        gradients = dict(zip(CANDIDATE_OUTPUT_KEYS, output_grads, strict=True))

        zero_gradients = {}

        def grad(key, reference):
            value = gradients[key]
            if value is not None:
                return value.contiguous()
            shape = tuple(reference.shape)
            if shape not in zero_gradients:
                zero_gradients[shape] = torch.zeros_like(reference)
            return zero_gradients[shape]

        native = importlib.import_module(
            f"{__package__}.net_subgraph_timing_{'cuda' if lin.is_cuda else 'cpp'}"
        )
        sink_reference = lin.new_empty(sinks.numel())
        adjoints = native.candidate_net_subgraph_backward(
            topo,
            starts,
            parents,
            child_starts,
            children,
            resistance,
            candidate_index,
            bu,
            cap,
            delay_value,
            slew_value,
            sinks,
            sink_nets,
            lin,
            lout,
            slew_in,
            slew_out,
            grad("lout", lout),
            grad("lin", lin),
            grad("arrival_in", lin),
            grad("arrival_out", lin),
            grad("slew_in", slew_in),
            grad("slew_out", slew_out),
            grad("sink_arrival", sink_reference),
            grad("sink_slew", sink_reference),
            grad("sink_load", sink_reference),
            grad("sink_net_delay", sink_reference),
            grad("sink_net_impulse", sink_reference),
        )
        coverage = candidate_node_coverage(topo, lin.numel())
        unvisited_load = (
            adjoints["grad_lout"] + adjoints["grad_lin"]
            if coverage is not None and not lin.is_cuda
            else None
        )
        mask_unvisited_candidate_adjoints(adjoints, coverage)
        # The CPU forward retains effective capacitance at omitted padded nodes
        # in lout/lin. Those direct load derivatives still belong to node C.
        if unvisited_load is not None:
            adjoints["grad_lout"] = torch.where(coverage, adjoints["grad_lout"], unvisited_load)
        rc = candidate_rc_gradients_from_adjoint(
            parents=parents,
            edge_resistance=resistance,
            lin=lin,
            slew_in=slew_in,
            adjoints=adjoints,
            output_gradients=gradients,
            sink_node_id=sinks,
            required=(*ctx.needs_input_grad[5:7], ctx.needs_input_grad[7] and ctx.has_edge_cap),
            node_coverage=coverage if lin.is_cuda else None,
        )
        activation_grad = adjoints["grad_candidate_bu"]
        cap_grad = adjoints["grad_buffer_input_cap"]
        delay_grad = adjoints["grad_buffer_delay"]
        slew_grad = adjoints["grad_buffer_output_slew"]
        if ctx.typed:
            nodes = candidate_nodes.long()
            # At an inactive position every strength has the same forward state.
            # Each column nevertheless has its own device-cap/delay/slew tangent.
            activation_grad = (
                adjoints["grad_lin"][nodes, None] * (input_cap - lout[nodes, None])
                + adjoints["grad_arrival_out"][nodes, None] * delay
                + adjoints["grad_slew_out"][nodes, None] * (output_slew - slew_in[nodes, None])
            )
            active = bu > 0
            activation_grad = torch.where(
                active[:, None],
                adjoints["grad_candidate_bu"][:, None],
                activation_grad,
            )
            # Active buffers retain their selected master, including derivatives
            # through that master's LUT coordinate probe.
            cap_grad = (
                torch.zeros_like(input_cap).scatter(1, selected, cap_grad[:, None])
                if ctx.needs_input_grad[12]
                else None
            )
            delay_grad = (
                torch.zeros_like(delay).scatter(1, selected, delay_grad[:, None])
                if ctx.needs_input_grad[13]
                else None
            )
            slew_grad = (
                torch.zeros_like(output_slew).scatter(1, selected, slew_grad[:, None])
                if ctx.needs_input_grad[14]
                else None
            )
        return (
            None,
            None,
            None,
            None,
            None,
            *rc,
            adjoints["grad_driver_arrival"] if ctx.needs_input_grad[8] else None,
            adjoints["grad_driver_slew"] if ctx.needs_input_grad[9] else None,
            None,
            activation_grad if ctx.needs_input_grad[11] else None,
            cap_grad if ctx.needs_input_grad[12] else None,
            delay_grad if ctx.needs_input_grad[13] else None,
            slew_grad if ctx.needs_input_grad[14] else None,
            None,
            None,
        )
