import importlib
import math

import torch
from torch.autograd import Function


_NATIVE_RECOMPUTE_OUTPUT_KEYS = (
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

_EXPLICIT_OUTPUT_KEYS = _NATIVE_RECOMPUTE_OUTPUT_KEYS


def _candidate_lookup(candidate_node_id, values, num_nodes, *, dtype, device):
    result = torch.zeros(num_nodes, dtype=dtype, device=device)
    for index, node in enumerate(candidate_node_id.detach().cpu().tolist()):
        result[int(node)] = values[index].to(device=device, dtype=dtype)
    return result


def _candidate_node_values(values, candidate_node_id):
    node_index = candidate_node_id.to(device=values.device, dtype=torch.long)
    return values[node_index]


def _candidate_upstream_retained_cap(
    *,
    edge_capacitance,
    pin_fa,
    candidate_node_id,
    dtype,
    device,
):
    if edge_capacitance is None or int(edge_capacitance.numel()) == 0:
        return torch.zeros(
            int(candidate_node_id.numel()),
            dtype=dtype,
            device=device,
        )
    node_index = candidate_node_id.to(device=device, dtype=torch.long)
    parents = pin_fa.to(device=device, dtype=torch.long).index_select(0, node_index)
    incoming_cap = edge_capacitance.to(device=device, dtype=dtype).index_select(
        0,
        node_index,
    )
    return torch.where(
        parents >= 0,
        0.5 * incoming_cap,
        torch.zeros_like(incoming_cap),
    )


def _root_values_by_node(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    driver_values,
    num_nodes,
    dtype,
    device,
):
    if int(num_nodes) <= 0:
        return torch.empty((0,), dtype=dtype, device=device)
    topo = net_flat_topo_sort.detach().cpu().tolist()
    topo_start = net_flat_topo_sort_start.detach().cpu().tolist()
    root_values = [
        torch.zeros((), dtype=dtype, device=device) for _ in range(int(num_nodes))
    ]
    for net_idx in range(len(topo_start) - 1):
        value = driver_values[net_idx].to(device=device, dtype=dtype)
        for pos in range(topo_start[net_idx], topo_start[net_idx + 1]):
            root_values[int(topo[pos])] = value
    return torch.stack(root_values)


def _attach_sink_relative_outputs(
    result,
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    driver_arrival,
    driver_slew,
    sink_node_id,
):
    if sink_node_id is None or int(sink_node_id.numel()) == 0:
        return result
    sink_index = sink_node_id.to(
        device=result["arrival_out"].device,
        dtype=torch.long,
    )
    dtype = result["arrival_out"].dtype
    device = result["arrival_out"].device
    num_nodes = int(result["arrival_out"].numel())
    root_arrival = _root_values_by_node(
        net_flat_topo_sort=net_flat_topo_sort,
        net_flat_topo_sort_start=net_flat_topo_sort_start,
        driver_values=driver_arrival,
        num_nodes=num_nodes,
        dtype=dtype,
        device=device,
    )
    root_slew = _root_values_by_node(
        net_flat_topo_sort=net_flat_topo_sort,
        net_flat_topo_sort_start=net_flat_topo_sort_start,
        driver_values=driver_slew,
        num_nodes=num_nodes,
        dtype=dtype,
        device=device,
    )
    sink_root_arrival = root_arrival[sink_index]
    sink_root_slew = root_slew[sink_index]
    sink_arrival = result["arrival_out"][sink_index]
    sink_slew = result["slew_out"][sink_index]
    result["sink_root_arrival"] = sink_root_arrival
    result["sink_root_slew"] = sink_root_slew
    result["sink_net_delay"] = sink_arrival - sink_root_arrival
    result["sink_net_impulse"] = sink_slew * sink_slew - sink_root_slew * sink_root_slew
    return result


def net_subgraph_forward(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance=None,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id=None,
    sink_net_index=None,
):
    """Fixed-state buffer-aware net-subgraph forward.

    This is the Python reference path for the future tensor/C++ op. It treats
    each candidate as a pseudo-cell inside the net subgraph and propagates load,
    arrival, and slew over flattened topology tensors.
    """
    dtype = node_capacitance.dtype
    device = node_capacitance.device
    num_nodes = int(node_capacitance.numel())

    topo = net_flat_topo_sort.detach().cpu().tolist()
    topo_start = net_flat_topo_sort_start.detach().cpu().tolist()
    child_start = flat_pin_to_start.detach().cpu().tolist()
    children = flat_pin_to.detach().cpu().tolist()
    parents = pin_fa.detach().cpu().tolist()

    effective_node_capacitance = node_capacitance.clone()
    if edge_capacitance is not None:
        for child in range(num_nodes):
            parent = int(parents[child])
            if parent < 0:
                continue
            half_cap = 0.5 * edge_capacitance[child].to(device=device, dtype=dtype)
            effective_node_capacitance[parent] = effective_node_capacitance[parent] + half_cap
            effective_node_capacitance[child] = effective_node_capacitance[child] + half_cap

    bu_by_node = _candidate_lookup(
        candidate_node_id, candidate_bu, num_nodes, dtype=dtype, device=device
    )
    buffer_input_cap_by_node = _candidate_lookup(
        candidate_node_id, buffer_input_cap, num_nodes, dtype=dtype, device=device
    )
    buffer_delay_by_node = _candidate_lookup(
        candidate_node_id, buffer_delay, num_nodes, dtype=dtype, device=device
    )
    buffer_output_slew_by_node = _candidate_lookup(
        candidate_node_id, buffer_output_slew, num_nodes, dtype=dtype, device=device
    )
    lout = [effective_node_capacitance[node] for node in range(num_nodes)]
    lin = [effective_node_capacitance[node] for node in range(num_nodes)]
    for net_idx in range(len(topo_start) - 1):
        for pos in range(topo_start[net_idx + 1] - 1, topo_start[net_idx] - 1, -1):
            node = int(topo[pos])
            children_load = torch.zeros((), dtype=dtype, device=device)
            for edge_idx in range(child_start[node], child_start[node + 1]):
                children_load = children_load + lin[int(children[edge_idx])]
            node_lout = effective_node_capacitance[node] + children_load
            bu = bu_by_node[node]
            lout[node] = node_lout
            lin[node] = (1.0 - bu) * node_lout + bu * buffer_input_cap_by_node[node]

    lout_tensor = torch.stack(lout)
    lin_tensor = torch.stack(lin)

    arrival_in = [torch.zeros((), dtype=dtype, device=device) for _ in range(num_nodes)]
    arrival_out = [torch.zeros((), dtype=dtype, device=device) for _ in range(num_nodes)]
    slew_in = [torch.zeros((), dtype=dtype, device=device) for _ in range(num_nodes)]
    slew_out = [torch.zeros((), dtype=dtype, device=device) for _ in range(num_nodes)]

    for net_idx in range(len(topo_start) - 1):
        if topo_start[net_idx] == topo_start[net_idx + 1]:
            continue
        root = int(topo[topo_start[net_idx]])
        arrival_in[root] = driver_arrival[net_idx].to(device=device, dtype=dtype)
        arrival_out[root] = arrival_in[root]
        slew_in[root] = driver_slew[net_idx].to(device=device, dtype=dtype)
        slew_out[root] = slew_in[root]

        for pos in range(topo_start[net_idx], topo_start[net_idx + 1]):
            node = int(topo[pos])
            parent = int(parents[node])
            if parent >= 0:
                wire_delta = edge_resistance[node] * lin_tensor[node]
                arrival_in[node] = arrival_out[parent] + wire_delta
                slew_in[node] = torch.sqrt(
                    slew_out[parent] * slew_out[parent]
                    + (math.log(10.0) * wire_delta) * (math.log(10.0) * wire_delta)
                )

            bu = bu_by_node[node]
            arrival_out[node] = arrival_in[node] + bu * buffer_delay_by_node[node]
            slew_out[node] = (
                (1.0 - bu) * slew_in[node] + bu * buffer_output_slew_by_node[node]
            )

    result = {
        "lout": lout_tensor,
        "lin": lin_tensor,
        "effective_node_cap": effective_node_capacitance,
        "arrival_in": torch.stack(arrival_in),
        "arrival_out": torch.stack(arrival_out),
        "slew_in": torch.stack(slew_in),
        "slew_out": torch.stack(slew_out),
    }
    if sink_node_id is not None:
        sink_index = sink_node_id.to(device=device, dtype=torch.long)
        result["sink_arrival"] = result["arrival_out"][sink_index]
        result["sink_slew"] = result["slew_out"][sink_index]
        result["sink_load"] = result["lin"][sink_index]
        result["sink_cap"] = result["effective_node_cap"][sink_index]
        _attach_sink_relative_outputs(
            result,
            net_flat_topo_sort=net_flat_topo_sort,
            net_flat_topo_sort_start=net_flat_topo_sort_start,
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            sink_node_id=sink_node_id,
        )
        if sink_net_index is not None:
            result["sink_net_index"] = sink_net_index.to(device=device)
    return result


def net_subgraph_forward_relaxed(
    *,
    buffer_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    buffer_coordinate_callback=None,
    fixed_bsu_device_lut=None,
    fixed_bsu_index=None,
    coordinate_source="current_relaxed",
    forward_backend="python",
    **kwargs,
):
    from dreamplace.ops.buffer_insertion.optimization_state import (
        interpolate_buffer_values_by_bsu_index,
    )

    def interpolate(input_cap, delay, output_slew, source):
        return interpolate_buffer_values_by_bsu_index(
            buffer_state.bsu_index(),
            per_size_input_cap=input_cap,
            per_size_delay=delay,
            per_size_output_slew=output_slew,
            coordinate_source=source,
        )

    backend = str(forward_backend or "python")
    candidate_node_id = buffer_state.candidate_node_id.to(
        dtype=torch.long if backend == "cuda_explicit_autograd" else torch.int32
    )
    candidate_bu = buffer_state.candidate_bu()
    upstream_retained_cap = _candidate_upstream_retained_cap(
        edge_capacitance=kwargs.get("edge_capacitance"),
        pin_fa=kwargs["pin_fa"],
        candidate_node_id=candidate_node_id,
        dtype=candidate_bu.dtype,
        device=candidate_bu.device,
    )

    def retain_upstream_wire_cap(values):
        values = dict(values)
        values["buffer_input_cap"] = (
            values["buffer_input_cap"] + upstream_retained_cap
        )
        values["buffer_upstream_retained_cap"] = upstream_retained_cap
        return values

    buffer_tensors = retain_upstream_wire_cap(
        interpolate(
            per_size_input_cap,
            per_size_delay,
            per_size_output_slew,
            coordinate_source,
        )
    )
    if (
        backend == "cuda_explicit_autograd"
        and fixed_bsu_device_lut is not None
        and fixed_bsu_index is not None
    ):
        result = net_subgraph_forward_cuda_fixed_bsu_liberty_autograd(
            **kwargs,
            candidate_node_id=candidate_node_id,
            candidate_bu=candidate_bu,
            buffer_input_cap=buffer_tensors["buffer_input_cap"],
            probe_buffer_delay=buffer_tensors["buffer_delay"],
            probe_buffer_output_slew=buffer_tensors["buffer_output_slew"],
            upstream_retained_cap=upstream_retained_cap,
            buffer_device_lut=fixed_bsu_device_lut,
            fixed_bsu_index=fixed_bsu_index,
        )
        buffer_tensors.update(
            {
                "buffer_delay": result["buffer_delay"],
                "buffer_output_slew": result["buffer_output_slew"],
                "candidate_input_slew": result["candidate_input_slew"],
                "candidate_output_cap": result["candidate_output_load"],
            }
        )
        buffer_tensors["bsu_index_status"] = dict(
            buffer_tensors.get("bsu_index_status", {})
        )
        buffer_tensors["bsu_index_status"]["coordinate_source"] = (
            "liberty_lut_candidate_slew_load_fixed_bsu_fused"
        )
        result["relaxed_buffer_model"] = buffer_tensors
        return result
    if backend == "python":
        forward_fn = net_subgraph_forward
    elif backend == "native_explicit_autograd":
        forward_fn = net_subgraph_forward_native_explicit_autograd
    elif backend == "native_recompute_autograd":
        forward_fn = net_subgraph_forward_native_recompute_autograd
    elif backend == "cuda_explicit_autograd":
        forward_fn = net_subgraph_forward_cuda_explicit_autograd
    else:
        raise ValueError(f"unsupported net_subgraph forward_backend: {backend}")
    coordinate_probe = None
    if buffer_coordinate_callback is not None:
        coordinate_probe = forward_fn(
            **kwargs,
            candidate_node_id=candidate_node_id,
            candidate_bu=candidate_bu,
            buffer_input_cap=buffer_tensors["buffer_input_cap"],
            buffer_delay=buffer_tensors["buffer_delay"],
            buffer_output_slew=buffer_tensors["buffer_output_slew"],
        )
        candidate_input_slew = _candidate_node_values(
            coordinate_probe["slew_in"],
            buffer_state.candidate_node_id,
        )
        candidate_output_cap = _candidate_node_values(
            coordinate_probe["lout"],
            buffer_state.candidate_node_id,
        )
        candidate_output_cap = candidate_output_cap - upstream_retained_cap.to(
            device=candidate_output_cap.device,
            dtype=candidate_output_cap.dtype,
        )
        updated = buffer_coordinate_callback(
            buffer_state=buffer_state,
            candidate_input_slew=candidate_input_slew,
            candidate_output_cap=candidate_output_cap,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            coordinate_probe=coordinate_probe,
        )
        updated = dict(updated or {})
        per_size_input_cap = updated.get("per_size_input_cap", per_size_input_cap)
        per_size_delay = updated.get("per_size_delay", per_size_delay)
        per_size_output_slew = updated.get(
            "per_size_output_slew",
            per_size_output_slew,
        )
        coordinate_source = str(updated.get("coordinate_source", coordinate_source))
        direct_keys = (
            "buffer_input_cap",
            "buffer_delay",
            "buffer_output_slew",
        )
        if all(key in updated for key in direct_keys):
            direct_values = {key: updated[key] for key in direct_keys}
            for key in (
                "bsu_index",
                "bsu_index_lo",
                "bsu_index_hi",
                "bsu_index_alpha",
                "bsu_index_status",
            ):
                if key in buffer_tensors:
                    direct_values[key] = buffer_tensors[key]
            direct_values["bsu_index_status"] = dict(
                direct_values.get("bsu_index_status", {})
            )
            direct_values["bsu_index_status"]["coordinate_source"] = coordinate_source
            buffer_tensors = retain_upstream_wire_cap(direct_values)
        else:
            buffer_tensors = retain_upstream_wire_cap(
                interpolate(
                    per_size_input_cap,
                    per_size_delay,
                    per_size_output_slew,
                    coordinate_source,
                )
            )
        buffer_tensors["candidate_input_slew"] = candidate_input_slew
        buffer_tensors["candidate_output_cap"] = candidate_output_cap

    result = forward_fn(
        **kwargs,
        candidate_node_id=candidate_node_id,
        candidate_bu=candidate_bu,
        buffer_input_cap=buffer_tensors["buffer_input_cap"],
        buffer_delay=buffer_tensors["buffer_delay"],
        buffer_output_slew=buffer_tensors["buffer_output_slew"],
    )
    result["relaxed_buffer_model"] = buffer_tensors
    if coordinate_probe is not None:
        result["coordinate_probe"] = coordinate_probe
    return result


def net_subgraph_forward_native(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance=None,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id=None,
    sink_net_index=None,
):
    """Call the C++ fixed-state net-subgraph forward kernel.

    This is the production-facing wrapper for the native hot path. The Python
    reference above remains the equivalence oracle and should not be used for
    real-design net-arc recursion.
    """
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    if edge_capacitance is None:
        edge_capacitance = torch.empty(
            0,
            dtype=node_capacitance.dtype,
            device=node_capacitance.device,
        )
    if sink_node_id is None:
        sink_node_id = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    if sink_net_index is None:
        sink_net_index = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    result = native.forward(
        net_flat_topo_sort.contiguous(),
        net_flat_topo_sort_start.contiguous(),
        pin_fa.contiguous(),
        flat_pin_to_start.contiguous(),
        flat_pin_to.contiguous(),
        edge_resistance.contiguous(),
        node_capacitance.contiguous(),
        edge_capacitance.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        candidate_node_id.contiguous(),
        candidate_bu.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
    )
    _attach_sink_relative_outputs(
        result,
        net_flat_topo_sort=net_flat_topo_sort,
        net_flat_topo_sort_start=net_flat_topo_sort_start,
        driver_arrival=driver_arrival,
        driver_slew=driver_slew,
        sink_node_id=sink_node_id,
    )
    return result


class _NativeRecomputeAutogradFunction(Function):
    @staticmethod
    def forward(
        ctx,
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidate_node_id,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        sink_node_id,
        sink_net_index,
    ):
        ctx.save_for_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            node_capacitance,
            edge_capacitance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
        )
        with torch.no_grad():
            result = net_subgraph_forward_native(
                net_flat_topo_sort=net_flat_topo_sort,
                net_flat_topo_sort_start=net_flat_topo_sort_start,
                pin_fa=pin_fa,
                flat_pin_to_start=flat_pin_to_start,
                flat_pin_to=flat_pin_to,
                edge_resistance=edge_resistance,
                node_capacitance=node_capacitance,
                edge_capacitance=edge_capacitance,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                candidate_node_id=candidate_node_id,
                candidate_bu=candidate_bu,
                buffer_input_cap=buffer_input_cap,
                buffer_delay=buffer_delay,
                buffer_output_slew=buffer_output_slew,
                sink_node_id=sink_node_id,
                sink_net_index=sink_net_index,
            )
        outputs = []
        ctx.output_keys = []
        for key in _NATIVE_RECOMPUTE_OUTPUT_KEYS:
            value = result.get(key)
            if value is None:
                value = torch.empty(
                    0,
                    dtype=node_capacitance.dtype,
                    device=node_capacitance.device,
                )
            outputs.append(value)
            ctx.output_keys.append(key)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            node_capacitance,
            edge_capacitance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
        ) = ctx.saved_tensors
        driver_arrival_var = driver_arrival.detach().requires_grad_(
            bool(ctx.needs_input_grad[8])
        )
        driver_slew_var = driver_slew.detach().requires_grad_(
            bool(ctx.needs_input_grad[9])
        )
        candidate_bu_var = candidate_bu.detach().requires_grad_(
            bool(ctx.needs_input_grad[11])
        )
        buffer_input_cap_var = buffer_input_cap.detach().requires_grad_(
            bool(ctx.needs_input_grad[12])
        )
        buffer_delay_var = buffer_delay.detach().requires_grad_(
            bool(ctx.needs_input_grad[13])
        )
        buffer_output_slew_var = buffer_output_slew.detach().requires_grad_(
            bool(ctx.needs_input_grad[14])
        )
        with torch.enable_grad():
            recomputed = net_subgraph_forward(
                net_flat_topo_sort=net_flat_topo_sort,
                net_flat_topo_sort_start=net_flat_topo_sort_start,
                pin_fa=pin_fa,
                flat_pin_to_start=flat_pin_to_start,
                flat_pin_to=flat_pin_to,
                edge_resistance=edge_resistance,
                node_capacitance=node_capacitance,
                edge_capacitance=edge_capacitance,
                driver_arrival=driver_arrival_var,
                driver_slew=driver_slew_var,
                candidate_node_id=candidate_node_id,
                candidate_bu=candidate_bu_var,
                buffer_input_cap=buffer_input_cap_var,
                buffer_delay=buffer_delay_var,
                buffer_output_slew=buffer_output_slew_var,
                sink_node_id=sink_node_id,
                sink_net_index=sink_net_index,
            )
            output_values = []
            output_grads = []
            for key, grad in zip(ctx.output_keys, grad_outputs):
                value = recomputed.get(key)
                if (
                    value is None
                    or value.numel() == 0
                    or grad is None
                    or not value.requires_grad
                ):
                    continue
                output_values.append(value)
                output_grads.append(grad)
            if not output_values:
                return (
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
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            grad_inputs = []
            grad_input_names = []
            for name, value in (
                ("driver_arrival", driver_arrival_var),
                ("driver_slew", driver_slew_var),
                ("candidate_bu", candidate_bu_var),
                ("buffer_input_cap", buffer_input_cap_var),
                ("buffer_delay", buffer_delay_var),
                ("buffer_output_slew", buffer_output_slew_var),
            ):
                if value.requires_grad:
                    grad_inputs.append(value)
                    grad_input_names.append(name)
            grad_values = torch.autograd.grad(
                output_values,
                tuple(grad_inputs),
                grad_outputs=output_grads,
                allow_unused=True,
            )
        grad_by_name = dict(zip(grad_input_names, grad_values))
        grad_driver_arrival = grad_by_name.get("driver_arrival")
        grad_driver_slew = grad_by_name.get("driver_slew")
        grad_candidate_bu = grad_by_name.get("candidate_bu")
        grad_buffer_input_cap = grad_by_name.get("buffer_input_cap")
        grad_buffer_delay = grad_by_name.get("buffer_delay")
        grad_buffer_output_slew = grad_by_name.get("buffer_output_slew")
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_driver_arrival,
            grad_driver_slew,
            None,
            grad_candidate_bu,
            grad_buffer_input_cap,
            grad_buffer_delay,
            grad_buffer_output_slew,
            None,
            None,
        )


def net_subgraph_forward_native_recompute_autograd(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance=None,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id=None,
    sink_net_index=None,
):
    if edge_capacitance is None:
        edge_capacitance = torch.empty(
            0,
            dtype=node_capacitance.dtype,
            device=node_capacitance.device,
        )
    if sink_node_id is None:
        sink_node_id = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    if sink_net_index is None:
        sink_net_index = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    outputs = _NativeRecomputeAutogradFunction.apply(
        net_flat_topo_sort.contiguous(),
        net_flat_topo_sort_start.contiguous(),
        pin_fa.contiguous(),
        flat_pin_to_start.contiguous(),
        flat_pin_to.contiguous(),
        edge_resistance.contiguous(),
        node_capacitance.contiguous(),
        edge_capacitance.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        candidate_node_id.contiguous(),
        candidate_bu.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
    )
    result = {
        key: value
        for key, value in zip(_NATIVE_RECOMPUTE_OUTPUT_KEYS, outputs)
        if value.numel() > 0
    }
    if sink_net_index.numel() > 0:
        result["sink_net_index"] = sink_net_index.to(device=node_capacitance.device)
    return result


def _as_list(tensor):
    return tensor.detach().cpu().tolist()


def _candidate_node_mapping(candidate_node_id, num_nodes):
    mapping = [-1 for _ in range(int(num_nodes))]
    for index, node in enumerate(_as_list(candidate_node_id)):
        mapping[int(node)] = int(index)
    return mapping


def _float_list(tensor):
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _grad_output_list(grad, *, length, device, dtype):
    if grad is None or grad.numel() == 0:
        return None
    return [
        float(value)
        for value in grad.to(device=device, dtype=dtype).detach().cpu().tolist()
    ][: int(length)]


def _add_grad_output(target, grad, *, device, dtype):
    values = _grad_output_list(grad, length=len(target), device=device, dtype=dtype)
    if values is None:
        return
    for index, value in enumerate(values):
        target[index] += value


def _sink_net_indices_from_inputs(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    sink_node_id,
    sink_net_index,
):
    if sink_net_index is not None and int(sink_net_index.numel()) > 0:
        return [int(value) for value in sink_net_index.detach().cpu().tolist()]
    sinks = [int(value) for value in sink_node_id.detach().cpu().tolist()]
    if not sinks:
        return []
    sink_pos_by_node = {node: index for index, node in enumerate(sinks)}
    indices = [-1 for _ in sinks]
    topo = [int(value) for value in net_flat_topo_sort.detach().cpu().tolist()]
    topo_start = [
        int(value) for value in net_flat_topo_sort_start.detach().cpu().tolist()
    ]
    for net_idx in range(max(0, len(topo_start) - 1)):
        for pos in range(topo_start[net_idx], topo_start[net_idx + 1]):
            sink_pos = sink_pos_by_node.get(int(topo[pos]))
            if sink_pos is not None:
                indices[sink_pos] = int(net_idx)
    return indices


def net_subgraph_forward_cuda(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id,
    sink_net_index,
):
    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
    return native.candidate_net_subgraph_forward(
        net_flat_topo_sort.contiguous(),
        net_flat_topo_sort_start.contiguous(),
        pin_fa.contiguous(),
        flat_pin_to_start.contiguous(),
        flat_pin_to.contiguous(),
        edge_resistance.contiguous(),
        node_capacitance.contiguous(),
        edge_capacitance.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        candidate_node_id.contiguous(),
        candidate_bu.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
    )


_FIXED_BSU_OUTPUT_KEYS = _EXPLICIT_OUTPUT_KEYS + (
    "candidate_input_slew",
    "candidate_output_load",
    "buffer_delay",
    "buffer_output_slew",
    "probe_arrival_in",
    "probe_arrival_out",
    "probe_slew_in",
    "probe_slew_out",
)


class _CudaFixedBsuLibertyAutogradFunction(Function):
    @staticmethod
    def forward(
        ctx,
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidate_node_id,
        candidate_bu,
        buffer_input_cap,
        probe_buffer_delay,
        probe_buffer_output_slew,
        upstream_retained_cap,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        fixed_bsu_index,
        sink_node_id,
        sink_net_index,
    ):
        native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
        with torch.no_grad():
            result = native.candidate_net_subgraph_fixed_bsu_forward(
                net_flat_topo_sort,
                net_flat_topo_sort_start,
                pin_fa,
                flat_pin_to_start,
                flat_pin_to,
                edge_resistance,
                node_capacitance,
                edge_capacitance,
                driver_arrival,
                driver_slew,
                candidate_node_id,
                candidate_bu,
                buffer_input_cap,
                probe_buffer_delay,
                probe_buffer_output_slew,
                upstream_retained_cap,
                buffer_slew_axis,
                buffer_load_axis,
                buffer_delay_lut,
                buffer_output_slew_lut,
                int(fixed_bsu_index),
                sink_node_id,
                sink_net_index,
            )
        ctx.save_for_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            probe_buffer_delay,
            probe_buffer_output_slew,
            sink_node_id,
            sink_net_index,
            result["candidate_index_by_node"],
            result["lin"],
            result["lout"],
            result["probe_slew_in"],
            result["probe_slew_out"],
            result["buffer_delay"],
            result["buffer_output_slew"],
            result["buffer_delay_grad_slew"],
            result["buffer_delay_grad_load"],
            result["buffer_output_slew_grad_slew"],
            result["buffer_output_slew_grad_load"],
            result["slew_in"],
            result["slew_out"],
        )
        ctx.output_keys = _FIXED_BSU_OUTPUT_KEYS
        return tuple(result[key] for key in _FIXED_BSU_OUTPUT_KEYS)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            probe_buffer_delay,
            probe_buffer_output_slew,
            sink_node_id,
            sink_net_index,
            candidate_index_by_node,
            lin,
            lout,
            probe_slew_in,
            probe_slew_out,
            buffer_delay,
            buffer_output_slew,
            buffer_delay_grad_slew,
            buffer_delay_grad_load,
            buffer_output_slew_grad_slew,
            buffer_output_slew_grad_load,
            slew_in,
            slew_out,
        ) = ctx.saved_tensors
        grad_by_key = dict(zip(ctx.output_keys, grad_outputs))

        def grad_or_zero(key, reference):
            value = grad_by_key.get(key)
            if value is None or value.numel() == 0:
                return torch.zeros_like(reference)
            return value.to(
                device=reference.device,
                dtype=reference.dtype,
            ).contiguous()

        native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
        sink_reference = lin.new_zeros((int(sink_node_id.numel()),))
        final_grads = native.candidate_net_subgraph_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            candidate_index_by_node,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
            lin,
            lout,
            slew_in,
            slew_out,
            grad_or_zero("lout", lout),
            grad_or_zero("lin", lin),
            grad_or_zero("arrival_in", lin),
            grad_or_zero("arrival_out", lin),
            grad_or_zero("slew_in", slew_in),
            grad_or_zero("slew_out", slew_out),
            grad_or_zero("sink_arrival", sink_reference),
            grad_or_zero("sink_slew", sink_reference),
            grad_or_zero("sink_load", sink_reference),
            grad_or_zero("sink_net_delay", sink_reference),
            grad_or_zero("sink_net_impulse", sink_reference),
        )
        grad_buffer_delay = final_grads["grad_buffer_delay"] + grad_or_zero(
            "buffer_delay", buffer_delay
        )
        grad_buffer_output_slew = final_grads[
            "grad_buffer_output_slew"
        ] + grad_or_zero("buffer_output_slew", buffer_output_slew)
        coordinate_slew_grad = (
            grad_buffer_delay * buffer_delay_grad_slew
            + grad_buffer_output_slew * buffer_output_slew_grad_slew
            + grad_or_zero("candidate_input_slew", buffer_delay)
        )
        coordinate_load_grad = (
            grad_buffer_delay * buffer_delay_grad_load
            + grad_buffer_output_slew * buffer_output_slew_grad_load
            + grad_or_zero("candidate_output_load", buffer_delay)
        )
        probe_grad_lout = torch.zeros_like(lout)
        probe_grad_lout.index_add_(0, candidate_node_id, coordinate_load_grad)
        probe_grad_slew_in = grad_or_zero("probe_slew_in", probe_slew_in)
        probe_grad_slew_in.index_add_(0, candidate_node_id, coordinate_slew_grad)
        empty_sink = sink_node_id[:0]
        empty_sink_grad = sink_reference[:0]
        probe_grads = native.candidate_net_subgraph_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            candidate_index_by_node,
            candidate_bu,
            buffer_input_cap,
            probe_buffer_delay,
            probe_buffer_output_slew,
            empty_sink,
            empty_sink,
            lin,
            lout,
            probe_slew_in,
            probe_slew_out,
            probe_grad_lout,
            torch.zeros_like(lin),
            grad_or_zero("probe_arrival_in", lin),
            grad_or_zero("probe_arrival_out", lin),
            probe_grad_slew_in,
            grad_or_zero("probe_slew_out", probe_slew_out),
            empty_sink_grad,
            empty_sink_grad,
            empty_sink_grad,
            empty_sink_grad,
            empty_sink_grad,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            final_grads["grad_driver_arrival"]
            + probe_grads["grad_driver_arrival"],
            final_grads["grad_driver_slew"] + probe_grads["grad_driver_slew"],
            None,
            final_grads["grad_candidate_bu"] + probe_grads["grad_candidate_bu"],
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


def net_subgraph_forward_cuda_fixed_bsu_liberty_autograd(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    probe_buffer_delay,
    probe_buffer_output_slew,
    upstream_retained_cap,
    buffer_device_lut,
    fixed_bsu_index,
    sink_node_id,
    sink_net_index,
):
    if not node_capacitance.is_cuda:
        raise ValueError("fixed-BSu Liberty candidate backend requires CUDA tensors")
    device = node_capacitance.device
    dtype = node_capacitance.dtype

    def index_tensor(value):
        return value.to(device=device, dtype=torch.long).contiguous()

    def float_tensor(value):
        return value.to(device=device, dtype=dtype).contiguous()

    net_flat_topo_sort = index_tensor(net_flat_topo_sort)
    net_flat_topo_sort_start = index_tensor(net_flat_topo_sort_start)
    pin_fa = index_tensor(pin_fa)
    flat_pin_to_start = index_tensor(flat_pin_to_start)
    flat_pin_to = index_tensor(flat_pin_to)
    candidate_node_id = index_tensor(candidate_node_id)
    sink_node_id = index_tensor(sink_node_id)
    sink_net_index = index_tensor(sink_net_index)
    lut = {
        name: float_tensor(buffer_device_lut[name])
        for name in (
            "input_slew_axis",
            "output_load_axis",
            "delay_lut",
            "output_slew_lut",
        )
    }
    outputs = _CudaFixedBsuLibertyAutogradFunction.apply(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        float_tensor(edge_resistance),
        node_capacitance.contiguous(),
        float_tensor(edge_capacitance),
        float_tensor(driver_arrival),
        float_tensor(driver_slew),
        candidate_node_id,
        float_tensor(candidate_bu),
        float_tensor(buffer_input_cap),
        float_tensor(probe_buffer_delay),
        float_tensor(probe_buffer_output_slew),
        float_tensor(upstream_retained_cap),
        lut["input_slew_axis"],
        lut["output_load_axis"],
        lut["delay_lut"],
        lut["output_slew_lut"],
        int(fixed_bsu_index),
        sink_node_id,
        sink_net_index,
    )
    values = dict(zip(_FIXED_BSU_OUTPUT_KEYS, outputs))
    result = {key: values[key] for key in _EXPLICIT_OUTPUT_KEYS}
    result["sink_net_index"] = sink_net_index
    for key in (
        "candidate_input_slew",
        "candidate_output_load",
        "buffer_delay",
        "buffer_output_slew",
    ):
        result[key] = values[key]

    sink_arrival = values["probe_arrival_out"].index_select(0, sink_node_id)
    sink_slew = values["probe_slew_out"].index_select(0, sink_node_id)
    sink_root_arrival = driver_arrival.index_select(0, sink_net_index)
    sink_root_slew = driver_slew.index_select(0, sink_net_index)
    result["coordinate_probe"] = {
        "lout": values["lout"],
        "lin": values["lin"],
        "effective_node_cap": values["effective_node_cap"],
        "arrival_in": values["probe_arrival_in"],
        "arrival_out": values["probe_arrival_out"],
        "slew_in": values["probe_slew_in"],
        "slew_out": values["probe_slew_out"],
        "sink_arrival": sink_arrival,
        "sink_slew": sink_slew,
        "sink_load": values["lin"].index_select(0, sink_node_id),
        "sink_cap": values["effective_node_cap"].index_select(0, sink_node_id),
        "sink_net_delay": sink_arrival - sink_root_arrival,
        "sink_net_impulse": sink_slew * sink_slew - sink_root_slew * sink_root_slew,
        "sink_net_index": sink_net_index,
    }
    return result


class _CudaExplicitAutogradFunction(Function):
    @staticmethod
    def forward(
        ctx,
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidate_node_id,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        sink_node_id,
        sink_net_index,
    ):
        with torch.no_grad():
            result = net_subgraph_forward_cuda(
                net_flat_topo_sort=net_flat_topo_sort,
                net_flat_topo_sort_start=net_flat_topo_sort_start,
                pin_fa=pin_fa,
                flat_pin_to_start=flat_pin_to_start,
                flat_pin_to=flat_pin_to,
                edge_resistance=edge_resistance,
                node_capacitance=node_capacitance,
                edge_capacitance=edge_capacitance,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                candidate_node_id=candidate_node_id,
                candidate_bu=candidate_bu,
                buffer_input_cap=buffer_input_cap,
                buffer_delay=buffer_delay,
                buffer_output_slew=buffer_output_slew,
                sink_node_id=sink_node_id,
                sink_net_index=sink_net_index,
            )
        ctx.save_for_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
            result["candidate_index_by_node"],
            result["lin"],
            result["lout"],
            result["slew_in"],
            result["slew_out"],
        )
        ctx.output_keys = _EXPLICIT_OUTPUT_KEYS
        return tuple(result[key] for key in _EXPLICIT_OUTPUT_KEYS)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
            candidate_index_by_node,
            lin,
            lout,
            slew_in,
            slew_out,
        ) = ctx.saved_tensors
        grad_by_key = dict(zip(ctx.output_keys, grad_outputs))

        def grad_or_zero(key, reference):
            value = grad_by_key.get(key)
            if value is None or value.numel() == 0:
                return torch.zeros_like(reference)
            return value.to(device=reference.device, dtype=reference.dtype).contiguous()

        sink_reference = lin.new_zeros((int(sink_node_id.numel()),))
        native = importlib.import_module(f"{__package__}.net_subgraph_timing_cuda")
        grads = native.candidate_net_subgraph_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            candidate_index_by_node,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            sink_net_index,
            lin,
            lout,
            slew_in,
            slew_out,
            grad_or_zero("lout", lout),
            grad_or_zero("lin", lin),
            grad_or_zero("arrival_in", lin),
            grad_or_zero("arrival_out", lin),
            grad_or_zero("slew_in", slew_in),
            grad_or_zero("slew_out", slew_out),
            grad_or_zero("sink_arrival", sink_reference),
            grad_or_zero("sink_slew", sink_reference),
            grad_or_zero("sink_load", sink_reference),
            grad_or_zero("sink_net_delay", sink_reference),
            grad_or_zero("sink_net_impulse", sink_reference),
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grads["grad_driver_arrival"] if ctx.needs_input_grad[8] else None,
            grads["grad_driver_slew"] if ctx.needs_input_grad[9] else None,
            None,
            grads["grad_candidate_bu"] if ctx.needs_input_grad[11] else None,
            grads["grad_buffer_input_cap"] if ctx.needs_input_grad[12] else None,
            grads["grad_buffer_delay"] if ctx.needs_input_grad[13] else None,
            grads["grad_buffer_output_slew"] if ctx.needs_input_grad[14] else None,
            None,
            None,
        )


def net_subgraph_forward_cuda_explicit_autograd(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance=None,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id=None,
    sink_net_index=None,
):
    if not node_capacitance.is_cuda:
        raise ValueError("cuda_explicit_autograd requires CUDA tensors")
    device = node_capacitance.device
    dtype = node_capacitance.dtype

    def index_tensor(value):
        return value.to(device=device, dtype=torch.long).contiguous()

    def float_tensor(value):
        return value.to(device=device, dtype=dtype).contiguous()

    net_flat_topo_sort = index_tensor(net_flat_topo_sort)
    net_flat_topo_sort_start = index_tensor(net_flat_topo_sort_start)
    pin_fa = index_tensor(pin_fa)
    flat_pin_to_start = index_tensor(flat_pin_to_start)
    flat_pin_to = index_tensor(flat_pin_to)
    candidate_node_id = index_tensor(candidate_node_id)
    if edge_capacitance is None or int(edge_capacitance.numel()) == 0:
        edge_capacitance = torch.zeros_like(node_capacitance)
    if sink_node_id is None:
        sink_node_id = torch.empty(0, dtype=torch.long, device=device)
    else:
        sink_node_id = index_tensor(sink_node_id)
    if sink_net_index is None or int(sink_net_index.numel()) == 0:
        indices = _sink_net_indices_from_inputs(
            net_flat_topo_sort=net_flat_topo_sort,
            net_flat_topo_sort_start=net_flat_topo_sort_start,
            sink_node_id=sink_node_id,
            sink_net_index=None,
        )
        sink_net_index = torch.as_tensor(indices, dtype=torch.long, device=device)
    else:
        sink_net_index = index_tensor(sink_net_index)
    outputs = _CudaExplicitAutogradFunction.apply(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        float_tensor(edge_resistance),
        node_capacitance.contiguous(),
        float_tensor(edge_capacitance),
        float_tensor(driver_arrival),
        float_tensor(driver_slew),
        candidate_node_id,
        float_tensor(candidate_bu),
        float_tensor(buffer_input_cap),
        float_tensor(buffer_delay),
        float_tensor(buffer_output_slew),
        sink_node_id,
        sink_net_index,
    )
    result = dict(zip(_EXPLICIT_OUTPUT_KEYS, outputs))
    result["sink_net_index"] = sink_net_index
    return result


class _NativeExplicitAutogradFunction(Function):
    @staticmethod
    def forward(
        ctx,
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidate_node_id,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        sink_node_id,
        sink_net_index,
    ):
        with torch.no_grad():
            result = net_subgraph_forward_native(
                net_flat_topo_sort=net_flat_topo_sort,
                net_flat_topo_sort_start=net_flat_topo_sort_start,
                pin_fa=pin_fa,
                flat_pin_to_start=flat_pin_to_start,
                flat_pin_to=flat_pin_to,
                edge_resistance=edge_resistance,
                node_capacitance=node_capacitance,
                edge_capacitance=edge_capacitance,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                candidate_node_id=candidate_node_id,
                candidate_bu=candidate_bu,
                buffer_input_cap=buffer_input_cap,
                buffer_delay=buffer_delay,
                buffer_output_slew=buffer_output_slew,
                sink_node_id=sink_node_id,
                sink_net_index=sink_net_index,
            )
        ctx.save_for_backward(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            result["lin"],
            result["lout"],
            result["arrival_in"],
            result["arrival_out"],
            result["slew_in"],
            result["slew_out"],
        )
        ctx.sink_net_indices = _sink_net_indices_from_inputs(
            net_flat_topo_sort=net_flat_topo_sort,
            net_flat_topo_sort_start=net_flat_topo_sort_start,
            sink_node_id=sink_node_id,
            sink_net_index=sink_net_index,
        )
        outputs = []
        ctx.output_keys = []
        for key in _EXPLICIT_OUTPUT_KEYS:
            value = result.get(key)
            if value is None:
                value = torch.empty(
                    0,
                    dtype=node_capacitance.dtype,
                    device=node_capacitance.device,
                )
            outputs.append(value)
            ctx.output_keys.append(key)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_node_id,
            candidate_bu,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
            sink_node_id,
            lin,
            lout,
            arrival_in,
            arrival_out,
            slew_in,
            slew_out,
        ) = ctx.saved_tensors
        device = lin.device
        dtype = lin.dtype
        num_nodes = int(lin.numel())
        num_candidates = int(candidate_node_id.numel())
        num_nets = int(driver_arrival.numel())

        topo = [int(value) for value in _as_list(net_flat_topo_sort)]
        topo_start = [int(value) for value in _as_list(net_flat_topo_sort_start)]
        parents = [int(value) for value in _as_list(pin_fa)]
        child_start = [int(value) for value in _as_list(flat_pin_to_start)]
        children = [int(value) for value in _as_list(flat_pin_to)]
        sinks = [int(value) for value in _as_list(sink_node_id)]
        candidate_by_node = _candidate_node_mapping(candidate_node_id, num_nodes)
        candidate_bu_values = _float_list(candidate_bu)
        buffer_input_cap_values = _float_list(buffer_input_cap)
        buffer_delay_values = _float_list(buffer_delay)
        buffer_output_slew_values = _float_list(buffer_output_slew)
        edge_resistance_values = _float_list(edge_resistance)
        lin_values = _float_list(lin)
        lout_values = _float_list(lout)
        slew_in_values = _float_list(slew_in)
        slew_out_values = _float_list(slew_out)

        grad_lout = [0.0 for _ in range(num_nodes)]
        grad_lin = [0.0 for _ in range(num_nodes)]
        grad_arrival_in = [0.0 for _ in range(num_nodes)]
        grad_arrival_out = [0.0 for _ in range(num_nodes)]
        grad_slew_in = [0.0 for _ in range(num_nodes)]
        grad_slew_out = [0.0 for _ in range(num_nodes)]
        grad_driver_arrival = [0.0 for _ in range(num_nets)]
        grad_driver_slew = [0.0 for _ in range(num_nets)]
        grad_candidate_bu = [0.0 for _ in range(num_candidates)]
        grad_buffer_input_cap = [0.0 for _ in range(num_candidates)]
        grad_buffer_delay = [0.0 for _ in range(num_candidates)]
        grad_buffer_output_slew = [0.0 for _ in range(num_candidates)]

        grad_by_key = {
            key: grad
            for key, grad in zip(ctx.output_keys, grad_outputs)
            if grad is not None and grad.numel() > 0
        }
        for key, target in (
            ("lout", grad_lout),
            ("lin", grad_lin),
            ("arrival_in", grad_arrival_in),
            ("arrival_out", grad_arrival_out),
            ("slew_in", grad_slew_in),
            ("slew_out", grad_slew_out),
        ):
            _add_grad_output(
                target,
                grad_by_key.get(key),
                device=device,
                dtype=dtype,
            )

        if sinks:
            for key, target in (
                ("sink_arrival", grad_arrival_out),
                ("sink_slew", grad_slew_out),
                ("sink_load", grad_lin),
            ):
                values = _grad_output_list(
                    grad_by_key.get(key),
                    length=len(sinks),
                    device=device,
                    dtype=dtype,
                )
                if values is not None:
                    for sink_pos, sink_node in enumerate(sinks):
                        target[int(sink_node)] += values[sink_pos]
            grad = grad_by_key.get("sink_net_delay")
            if grad is not None:
                values = _grad_output_list(
                    grad,
                    length=len(sinks),
                    device=device,
                    dtype=dtype,
                )
                for sink_pos, sink_node in enumerate(sinks):
                    sink_node = int(sink_node)
                    grad_value = values[sink_pos]
                    grad_arrival_out[sink_node] += grad_value
                    net_idx = int(ctx.sink_net_indices[sink_pos])
                    if 0 <= net_idx + 1 < len(topo_start) and topo_start[net_idx] != topo_start[net_idx + 1]:
                        root = int(topo[topo_start[net_idx]])
                        grad_arrival_out[root] -= grad_value
            grad = grad_by_key.get("sink_net_impulse")
            if grad is not None:
                values = _grad_output_list(
                    grad,
                    length=len(sinks),
                    device=device,
                    dtype=dtype,
                )
                for sink_pos, sink_node in enumerate(sinks):
                    sink_node = int(sink_node)
                    grad_value = values[sink_pos]
                    grad_slew_out[sink_node] += (
                        2.0 * slew_out_values[sink_node] * grad_value
                    )
                    net_idx = int(ctx.sink_net_indices[sink_pos])
                    if 0 <= net_idx + 1 < len(topo_start) and topo_start[net_idx] != topo_start[net_idx + 1]:
                        root = int(topo[topo_start[net_idx]])
                        grad_slew_out[root] -= (
                            2.0 * slew_out_values[root] * grad_value
                        )

        log10 = math.log(10.0)
        for net_idx in range(max(0, len(topo_start) - 1)):
            begin = topo_start[net_idx]
            end = topo_start[net_idx + 1]
            if begin == end:
                continue
            root = int(topo[begin])
            for pos in range(end - 1, begin - 1, -1):
                node = int(topo[pos])
                candidate_index = candidate_by_node[node]
                bu = candidate_bu_values[candidate_index] if candidate_index >= 0 else 0.0
                delay = buffer_delay_values[candidate_index] if candidate_index >= 0 else 0.0
                out_slew = buffer_output_slew_values[candidate_index] if candidate_index >= 0 else 0.0

                grad_arrival_in[node] += grad_arrival_out[node]
                if candidate_index >= 0:
                    grad_candidate_bu[candidate_index] += grad_arrival_out[node] * delay
                    grad_buffer_delay[candidate_index] += grad_arrival_out[node] * bu

                grad_slew_in[node] += grad_slew_out[node] * (1.0 - bu)
                if candidate_index >= 0:
                    grad_candidate_bu[candidate_index] += (
                        grad_slew_out[node] * (out_slew - slew_in_values[node])
                    )
                    grad_buffer_output_slew[candidate_index] += grad_slew_out[node] * bu

                parent = parents[node]
                if parent >= 0:
                    wire_delta = edge_resistance_values[node] * lin_values[node]
                    grad_arrival_out[parent] += grad_arrival_in[node]
                    grad_lin[node] += grad_arrival_in[node] * edge_resistance_values[node]

                    slew_value = slew_in_values[node]
                    if abs(slew_value) > 0.0:
                        grad_slew_out[parent] += (
                            grad_slew_in[node] * slew_out_values[parent] / slew_value
                        )
                        grad_wire_delta = (
                            grad_slew_in[node] * (log10 * log10) * wire_delta / slew_value
                        )
                        grad_lin[node] += grad_wire_delta * edge_resistance_values[node]
                else:
                    grad_driver_arrival[net_idx] += grad_arrival_in[node]
                    grad_driver_slew[net_idx] += grad_slew_in[node]

        for net_idx in range(max(0, len(topo_start) - 1)):
            begin = topo_start[net_idx]
            end = topo_start[net_idx + 1]
            for pos in range(begin, end):
                node = int(topo[pos])
                candidate_index = candidate_by_node[node]
                bu = candidate_bu_values[candidate_index] if candidate_index >= 0 else 0.0
                input_cap = (
                    buffer_input_cap_values[candidate_index]
                    if candidate_index >= 0
                    else 0.0
                )

                if candidate_index >= 0:
                    grad_candidate_bu[candidate_index] += (
                        grad_lin[node] * (input_cap - lout_values[node])
                    )
                    grad_buffer_input_cap[candidate_index] += grad_lin[node] * bu
                    grad_lout[node] += grad_lin[node] * (1.0 - bu)
                else:
                    grad_lout[node] += grad_lin[node]

                grad_children_load = grad_lout[node]
                for edge_idx in range(child_start[node], child_start[node + 1]):
                    child = int(children[edge_idx])
                    grad_lin[child] += grad_children_load

        def as_grad(values):
            return torch.as_tensor(values, dtype=dtype, device=device)

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            as_grad(grad_driver_arrival) if ctx.needs_input_grad[8] else None,
            as_grad(grad_driver_slew) if ctx.needs_input_grad[9] else None,
            None,
            as_grad(grad_candidate_bu) if ctx.needs_input_grad[11] else None,
            as_grad(grad_buffer_input_cap) if ctx.needs_input_grad[12] else None,
            as_grad(grad_buffer_delay) if ctx.needs_input_grad[13] else None,
            as_grad(grad_buffer_output_slew) if ctx.needs_input_grad[14] else None,
            None,
            None,
        )


def net_subgraph_forward_native_explicit_autograd(
    *,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    flat_pin_to_start,
    flat_pin_to,
    edge_resistance,
    node_capacitance,
    edge_capacitance=None,
    driver_arrival,
    driver_slew,
    candidate_node_id,
    candidate_bu,
    buffer_input_cap,
    buffer_delay,
    buffer_output_slew,
    sink_node_id=None,
    sink_net_index=None,
):
    if edge_capacitance is None:
        edge_capacitance = torch.empty(
            0,
            dtype=node_capacitance.dtype,
            device=node_capacitance.device,
        )
    if sink_node_id is None:
        sink_node_id = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    if sink_net_index is None:
        sink_net_index = torch.empty(
            0,
            dtype=torch.int32,
            device=net_flat_topo_sort.device,
        )
    outputs = _NativeExplicitAutogradFunction.apply(
        net_flat_topo_sort.contiguous(),
        net_flat_topo_sort_start.contiguous(),
        pin_fa.contiguous(),
        flat_pin_to_start.contiguous(),
        flat_pin_to.contiguous(),
        edge_resistance.contiguous(),
        node_capacitance.contiguous(),
        edge_capacitance.contiguous(),
        driver_arrival.contiguous(),
        driver_slew.contiguous(),
        candidate_node_id.contiguous(),
        candidate_bu.contiguous(),
        buffer_input_cap.contiguous(),
        buffer_delay.contiguous(),
        buffer_output_slew.contiguous(),
        sink_node_id.contiguous(),
        sink_net_index.contiguous(),
    )
    result = {
        key: value
        for key, value in zip(_EXPLICIT_OUTPUT_KEYS, outputs)
        if value.numel() > 0
    }
    if sink_net_index.numel() > 0:
        result["sink_net_index"] = sink_net_index.to(device=node_capacitance.device)
    return result
