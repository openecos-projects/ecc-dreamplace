import math

import torch

from .schema import SegmentTransferInput, SegmentTransferResult


def _interp_axis(axis, point):
    if axis.numel() == 1:
        index = torch.zeros((), dtype=torch.long, device=axis.device)
        alpha = torch.zeros((), dtype=point.dtype, device=point.device)
        return index, index, alpha, {
            "clamped_low": bool((point < axis[0]).detach().cpu().item()),
            "clamped_high": bool((point > axis[0]).detach().cpu().item()),
        }
    hi = torch.searchsorted(axis.detach(), point.detach(), right=True)
    hi = torch.clamp(hi, 1, int(axis.numel() - 1)).to(dtype=torch.long)
    lo = hi - 1
    denom = axis[hi] - axis[lo]
    alpha = (point - axis[lo]) / denom
    return lo, hi, alpha, {
        "clamped_low": bool((point < axis[0]).detach().cpu().item()),
        "clamped_high": bool((point > axis[-1]).detach().cpu().item()),
    }


def _interp_size(values, bsu_index):
    size_count = int(values.shape[0])
    clipped = torch.clamp(bsu_index, 0.0, float(size_count - 1))
    lo = torch.floor(clipped.detach()).to(dtype=torch.long)
    hi = torch.clamp(lo + 1, max=size_count - 1)
    alpha = clipped - lo.to(dtype=clipped.dtype)
    return (1.0 - alpha) * values[lo] + alpha * values[hi]


def _lookup_device(buffer_device, *, bsu_index, input_slew, output_load, dtype, device):
    tables = buffer_device.tensors(dtype=dtype, device=device)
    input_cap = _interp_size(tables["input_cap_by_size"], bsu_index)
    slew_lo, slew_hi, slew_alpha, slew_status = _interp_axis(
        tables["input_slew_axis"],
        input_slew,
    )
    load_lo, load_hi, load_alpha, load_status = _interp_axis(
        tables["output_load_axis"],
        output_load,
    )

    def lookup(table):
        v00 = table[:, slew_lo, load_lo]
        v01 = table[:, slew_lo, load_hi]
        v10 = table[:, slew_hi, load_lo]
        v11 = table[:, slew_hi, load_hi]
        v0 = (1.0 - load_alpha) * v00 + load_alpha * v01
        v1 = (1.0 - load_alpha) * v10 + load_alpha * v11
        per_size = (1.0 - slew_alpha) * v0 + slew_alpha * v1
        return _interp_size(per_size, bsu_index)

    return {
        "buffer_input_cap": input_cap,
        "buffer_delay": lookup(tables["delay_lut"]),
        "buffer_output_slew": lookup(tables["output_slew_lut"]),
        "slew_status": slew_status,
        "load_status": load_status,
    }


def _wire_slew(input_slew, wire_delay):
    log10 = torch.as_tensor(math.log(10.0), dtype=input_slew.dtype, device=input_slew.device)
    return torch.sqrt(input_slew * input_slew + (log10 * wire_delay) * (log10 * wire_delay))


def analytic_segment_transfer(transfer_input: SegmentTransferInput) -> SegmentTransferResult:
    tensors, fractions = transfer_input.validate()
    input_arrival = tensors["input_arrival"]
    input_slew = tensors["input_slew"]
    downstream_load = tensors["downstream_load"]
    edge_resistance = tensors["edge_resistance"]
    edge_capacitance = tensors["edge_capacitance"]
    upstream_retained_capacitance = tensors["upstream_retained_capacitance"]
    bsu_index = tensors["bsu_index"]
    repeater_count = int(transfer_input.repeater_count)

    if repeater_count == 0:
        wire_delay = edge_resistance * (downstream_load + 0.5 * edge_capacitance)
        output_slew = _wire_slew(input_slew, wire_delay)
        upstream_visible_load = downstream_load + edge_capacitance
        diagnostics = {
            "repeater_count": 0,
            "wire_delay": wire_delay,
            "edge_resistance": edge_resistance,
            "edge_capacitance": edge_capacitance,
            "downstream_load": downstream_load,
            "split_fractions": fractions,
        }
        return SegmentTransferResult(
            upstream_visible_load=upstream_visible_load,
            segment_delay=wire_delay,
            output_slew=output_slew,
            output_arrival=input_arrival + wire_delay,
            diagnostics=diagnostics,
        )

    segment_resistances = [
        edge_resistance * float(fraction)
        for fraction in fractions
    ]
    segment_capacitances = [
        edge_capacitance * float(fraction)
        for fraction in fractions
    ]
    buffer_input_cap = _lookup_device(
        transfer_input.buffer_device,
        bsu_index=bsu_index,
        input_slew=input_slew,
        output_load=downstream_load,
        dtype=input_slew.dtype,
        device=input_slew.device,
    )["buffer_input_cap"]
    downstream_load_by_segment = []
    current_load = downstream_load
    for index in reversed(range(repeater_count + 1)):
        downstream_load_by_segment.append(current_load)
        if index > 0:
            current_load = buffer_input_cap
    downstream_load_by_segment = list(reversed(downstream_load_by_segment))

    total_delay = torch.zeros((), dtype=input_slew.dtype, device=input_slew.device)
    slew = input_slew
    buffer_input_slews = []
    buffer_output_loads = []
    buffer_delays = []
    buffer_output_slews = []
    wire_delays = []
    for index in range(repeater_count):
        wire_load = buffer_input_cap + 0.5 * segment_capacitances[index]
        wire_delay = segment_resistances[index] * wire_load
        wire_delays.append(wire_delay)
        total_delay = total_delay + wire_delay
        slew = _wire_slew(slew, wire_delay)
        buffer_input_slews.append(slew)

        output_load = downstream_load_by_segment[index + 1] + segment_capacitances[index + 1]
        buffer_output_loads.append(output_load)
        device = _lookup_device(
            transfer_input.buffer_device,
            bsu_index=bsu_index,
            input_slew=slew,
            output_load=output_load,
            dtype=input_slew.dtype,
            device=input_slew.device,
        )
        total_delay = total_delay + device["buffer_delay"]
        buffer_delays.append(device["buffer_delay"])
        buffer_output_slews.append(device["buffer_output_slew"])
        slew = device["buffer_output_slew"]

    final_index = repeater_count
    downstream_wire_delay = segment_resistances[final_index] * (
        downstream_load + 0.5 * segment_capacitances[final_index]
    )
    wire_delays.append(downstream_wire_delay)
    segment_delay = total_delay + downstream_wire_delay
    output_slew = _wire_slew(slew, downstream_wire_delay)
    upstream_visible_load = (
        buffer_input_cap
        + segment_capacitances[0]
        + upstream_retained_capacitance
    )
    first_buffer_input_slew = buffer_input_slews[0]
    first_buffer_output_load = buffer_output_loads[0]
    first_buffer_delay = buffer_delays[0]
    first_buffer_output_slew = buffer_output_slews[0]
    diagnostics = {
        "repeater_count": repeater_count,
        "split_fractions": fractions,
        "segment_resistances": segment_resistances,
        "segment_capacitances": segment_capacitances,
        "upstream_resistance": segment_resistances[0],
        "downstream_resistance": segment_resistances[-1],
        "upstream_capacitance": segment_capacitances[0],
        "upstream_retained_capacitance": upstream_retained_capacitance,
        "downstream_capacitance": segment_capacitances[-1],
        "wire_delays": wire_delays,
        "upstream_wire_delay": wire_delays[0],
        "downstream_wire_delay": downstream_wire_delay,
        "buffer_input_slew": first_buffer_input_slew,
        "buffer_input_slews": buffer_input_slews,
        "buffer_output_load": first_buffer_output_load,
        "buffer_output_loads": buffer_output_loads,
        "buffer_input_cap": buffer_input_cap,
        "buffer_delay": first_buffer_delay,
        "buffer_delays": buffer_delays,
        "buffer_output_slew": first_buffer_output_slew,
        "buffer_output_slews": buffer_output_slews,
        "device_source": transfer_input.buffer_device.source,
    }
    return SegmentTransferResult(
        upstream_visible_load=upstream_visible_load,
        segment_delay=segment_delay,
        output_slew=output_slew,
        output_arrival=input_arrival + segment_delay,
        diagnostics=diagnostics,
    )
