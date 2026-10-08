import math

import torch


def _as_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
        return tensor
    return torch.as_tensor(value, dtype=dtype, device=device)


def _require_vector(name, value, *, dtype=None, device=None):
    tensor = _as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [S]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _require_nonnegative(name, tensor):
    if torch.any(tensor < 0):
        raise ValueError(f"{name} must be non-negative")


def _validate_axis(name, axis):
    if axis.ndim != 1 or int(axis.numel()) == 0:
        raise ValueError(f"{name} must have shape [N]")
    if not torch.isfinite(axis).all():
        raise ValueError(f"{name} must be finite")
    _require_nonnegative(name, axis)
    if int(axis.numel()) > 1 and torch.any(torch.diff(axis) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def _interp_size(values, bsu_index):
    size_count = int(values.shape[0])
    clipped = torch.clamp(bsu_index, 0.0, float(size_count - 1))
    lo = torch.floor(clipped.detach()).to(dtype=torch.long)
    hi = torch.clamp(lo + 1, max=size_count - 1)
    alpha = clipped - lo.to(dtype=clipped.dtype)
    if values.ndim == 1:
        return (1.0 - alpha) * values[lo] + alpha * values[hi]
    if values.ndim == 2 and int(values.shape[1]) == int(bsu_index.numel()):
        sample_index = torch.arange(int(bsu_index.numel()), device=bsu_index.device)
        lo_value = values[lo, sample_index]
        hi_value = values[hi, sample_index]
        return (1.0 - alpha) * lo_value + alpha * hi_value
    raise ValueError("size interpolation values must have shape [B] or [B, S]")


def _interp_axis(axis, point):
    if int(axis.numel()) == 1:
        index = torch.zeros_like(point, dtype=torch.long)
        alpha = torch.zeros_like(point)
        return index, index, alpha
    hi = torch.searchsorted(axis.detach(), point.detach(), right=True)
    hi = torch.clamp(hi, 1, int(axis.numel() - 1)).to(dtype=torch.long)
    lo = hi - 1
    alpha = (point - axis[lo]) / (axis[hi] - axis[lo])
    return lo, hi, alpha


def _lookup_device(
    *,
    bsu_index,
    input_slew,
    output_load,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
):
    per_size = _lookup_device_per_size(
        input_slew=input_slew,
        output_load=output_load,
        buffer_input_cap_by_size=buffer_input_cap_by_size,
        buffer_slew_axis=buffer_slew_axis,
        buffer_load_axis=buffer_load_axis,
        buffer_delay_lut=buffer_delay_lut,
        buffer_output_slew_lut=buffer_output_slew_lut,
    )
    return {
        "buffer_input_cap": _interp_size(
            per_size["per_size_input_cap"].transpose(0, 1),
            bsu_index,
        ),
        "buffer_delay": _interp_size(
            per_size["per_size_delay"].transpose(0, 1),
            bsu_index,
        ),
        "buffer_output_slew": _interp_size(
            per_size["per_size_output_slew"].transpose(0, 1),
            bsu_index,
        ),
    }


def _lookup_device_per_size(
    *,
    input_slew,
    output_load,
    buffer_input_cap_by_size,
    buffer_slew_axis,
    buffer_load_axis,
    buffer_delay_lut,
    buffer_output_slew_lut,
):
    slew_lo, slew_hi, slew_alpha = _interp_axis(buffer_slew_axis, input_slew)
    load_lo, load_hi, load_alpha = _interp_axis(buffer_load_axis, output_load)

    def lookup(table):
        table_by_axis = table.permute(1, 2, 0)
        v00 = table_by_axis[slew_lo, load_lo]
        v01 = table_by_axis[slew_lo, load_hi]
        v10 = table_by_axis[slew_hi, load_lo]
        v11 = table_by_axis[slew_hi, load_hi]
        v0 = (1.0 - load_alpha).unsqueeze(1) * v00 + load_alpha.unsqueeze(1) * v01
        v1 = (1.0 - load_alpha).unsqueeze(1) * v10 + load_alpha.unsqueeze(1) * v11
        per_size = (1.0 - slew_alpha).unsqueeze(1) * v0 + slew_alpha.unsqueeze(1) * v1
        return per_size

    return {
        "per_size_input_cap": buffer_input_cap_by_size.view(1, -1).expand(
            int(input_slew.numel()),
            -1,
        ),
        "per_size_delay": lookup(buffer_delay_lut),
        "per_size_output_slew": lookup(buffer_output_slew_lut),
    }


def lookup_buffer_device_per_size(buffer_device_lut, *, input_slew, output_load):
    """Evaluate every legal buffer size at each candidate slew/load point."""

    input_slew = _require_vector("input_slew", input_slew)
    output_load = _require_vector(
        "output_load",
        output_load,
        dtype=input_slew.dtype,
        device=input_slew.device,
    )
    if int(input_slew.numel()) != int(output_load.numel()):
        raise ValueError("input_slew and output_load must have the same length")
    _require_nonnegative("input_slew", input_slew)
    _require_nonnegative("output_load", output_load)
    tables = buffer_device_lut.tensors(
        dtype=input_slew.dtype,
        device=input_slew.device,
    )
    return _lookup_device_per_size(
        input_slew=input_slew,
        output_load=output_load,
        buffer_input_cap_by_size=tables["input_cap_by_size"],
        buffer_slew_axis=tables["input_slew_axis"],
        buffer_load_axis=tables["output_load_axis"],
        buffer_delay_lut=tables["delay_lut"],
        buffer_output_slew_lut=tables["output_slew_lut"],
    )


def lookup_buffer_device_fixed_size(
    buffer_device_lut,
    *,
    size_index,
    input_slew,
    output_load,
):
    """Evaluate one legal buffer size without materializing every size row."""

    input_slew = _require_vector("input_slew", input_slew)
    output_load = _require_vector(
        "output_load",
        output_load,
        dtype=input_slew.dtype,
        device=input_slew.device,
    )
    if int(input_slew.numel()) != int(output_load.numel()):
        raise ValueError("input_slew and output_load must have the same length")
    _require_nonnegative("input_slew", input_slew)
    _require_nonnegative("output_load", output_load)
    tables = buffer_device_lut.tensors(
        dtype=input_slew.dtype,
        device=input_slew.device,
    )
    size_index = int(size_index)
    size_count = int(tables["input_cap_by_size"].numel())
    if not 0 <= size_index < size_count:
        raise ValueError("size_index is out of range")

    slew_lo, slew_hi, slew_alpha = _interp_axis(
        tables["input_slew_axis"], input_slew
    )
    load_lo, load_hi, load_alpha = _interp_axis(
        tables["output_load_axis"], output_load
    )

    def lookup(table):
        table = table[size_index]
        v00 = table[slew_lo, load_lo]
        v01 = table[slew_lo, load_hi]
        v10 = table[slew_hi, load_lo]
        v11 = table[slew_hi, load_hi]
        v0 = (1.0 - load_alpha) * v00 + load_alpha * v01
        v1 = (1.0 - load_alpha) * v10 + load_alpha * v11
        return (1.0 - slew_alpha) * v0 + slew_alpha * v1

    return {
        "buffer_input_cap": tables["input_cap_by_size"][size_index].expand_as(
            input_slew
        ),
        "buffer_delay": lookup(tables["delay_lut"]),
        "buffer_output_slew": lookup(tables["output_slew_lut"]),
    }


def _wire_slew(input_slew, wire_delay):
    log10 = torch.as_tensor(math.log(10.0), dtype=input_slew.dtype, device=input_slew.device)
    return torch.sqrt(input_slew * input_slew + (log10 * wire_delay) * (log10 * wire_delay))


def _validate_inputs(
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
    dtype = None
    device = None
    for value in (
        input_arrival,
        input_slew,
        downstream_load,
        edge_resistance,
        edge_capacitance,
        bsu_index,
        upstream_retained_cap,
    ):
        if torch.is_tensor(value):
            dtype = value.dtype if value.is_floating_point() else torch.float64
            device = value.device
            break
    dtype = dtype or torch.float64
    input_arrival = _require_vector("input_arrival", input_arrival, dtype=dtype, device=device)
    input_slew = _require_vector("input_slew", input_slew, dtype=dtype, device=input_arrival.device)
    device = input_arrival.device
    downstream_load = _require_vector("downstream_load", downstream_load, dtype=dtype, device=device)
    edge_resistance = _require_vector("edge_resistance", edge_resistance, dtype=dtype, device=device)
    edge_capacitance = _require_vector("edge_capacitance", edge_capacitance, dtype=dtype, device=device)
    bsu_index = _require_vector("bsu_index", bsu_index, dtype=dtype, device=device)
    upstream_retained_cap = _require_vector(
        "upstream_retained_cap",
        upstream_retained_cap,
        dtype=dtype,
        device=device,
    )
    sample_count = int(input_arrival.numel())
    for name, tensor in (
        ("input_slew", input_slew),
        ("downstream_load", downstream_load),
        ("edge_resistance", edge_resistance),
        ("edge_capacitance", edge_capacitance),
        ("bsu_index", bsu_index),
        ("upstream_retained_cap", upstream_retained_cap),
    ):
        _require_nonnegative(name, tensor)
        if int(tensor.numel()) != sample_count:
            raise ValueError(f"{name} length must match input_arrival")

    repeater_count = _as_tensor(repeater_count, device=device)
    if repeater_count.ndim != 1 or int(repeater_count.numel()) != sample_count:
        raise ValueError("repeater_count must have shape [S]")
    if repeater_count.dtype == torch.bool:
        raise ValueError("repeater_count must be an integer tensor")
    if torch.any(repeater_count < 0):
        raise ValueError("repeater_count must be non-negative")
    if torch.is_floating_point(repeater_count):
        if torch.any(repeater_count != torch.floor(repeater_count)):
            raise ValueError("repeater_count must be integer-valued")
    repeater_count = repeater_count.to(dtype=torch.long, device=device)

    split_fractions = _as_tensor(split_fractions, dtype=dtype, device=device)
    if split_fractions.ndim != 2 or int(split_fractions.shape[0]) != sample_count:
        raise ValueError("split_fractions must have shape [S, Nmax + 1]")
    if not torch.isfinite(split_fractions).all():
        raise ValueError("split_fractions must be finite")
    _require_nonnegative("split_fractions", split_fractions)
    nmax = int(split_fractions.shape[1]) - 1
    if nmax < 0:
        raise ValueError("split_fractions must have at least one column")
    if int(torch.max(repeater_count).detach().cpu().item()) > nmax:
        raise ValueError("repeater_count cannot exceed split_fractions.shape[1] - 1")
    used_mask = (
        torch.arange(nmax + 1, device=device).view(1, -1)
        <= repeater_count.view(-1, 1)
    )
    used_sum = torch.sum(split_fractions * used_mask.to(dtype=dtype), dim=1)
    if torch.any(torch.abs(used_sum - 1.0) > 1e-6):
        raise ValueError("used split_fractions must sum to one for each segment")

    buffer_input_cap_by_size = _as_tensor(buffer_input_cap_by_size, dtype=dtype, device=device)
    buffer_slew_axis = _as_tensor(buffer_slew_axis, dtype=dtype, device=device)
    buffer_load_axis = _as_tensor(buffer_load_axis, dtype=dtype, device=device)
    buffer_delay_lut = _as_tensor(buffer_delay_lut, dtype=dtype, device=device)
    buffer_output_slew_lut = _as_tensor(buffer_output_slew_lut, dtype=dtype, device=device)
    if torch.any(repeater_count > 0):
        if buffer_input_cap_by_size.ndim != 1 or int(buffer_input_cap_by_size.numel()) == 0:
            raise ValueError("buffer_input_cap_by_size must have shape [B]")
        _validate_axis("buffer_slew_axis", buffer_slew_axis)
        _validate_axis("buffer_load_axis", buffer_load_axis)
        expected = (
            int(buffer_input_cap_by_size.numel()),
            int(buffer_slew_axis.numel()),
            int(buffer_load_axis.numel()),
        )
        if tuple(buffer_delay_lut.shape) != expected:
            raise ValueError("buffer_delay_lut must have shape [B, Ts, Tl]")
        if tuple(buffer_output_slew_lut.shape) != expected:
            raise ValueError("buffer_output_slew_lut must have shape [B, Ts, Tl]")
        for name, tensor in (
            ("buffer_input_cap_by_size", buffer_input_cap_by_size),
            ("buffer_delay_lut", buffer_delay_lut),
            ("buffer_output_slew_lut", buffer_output_slew_lut),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
            _require_nonnegative(name, tensor)

    return {
        "input_arrival": input_arrival,
        "input_slew": input_slew,
        "downstream_load": downstream_load,
        "edge_resistance": edge_resistance,
        "edge_capacitance": edge_capacitance,
        "repeater_count": repeater_count,
        "split_fractions": split_fractions,
        "bsu_index": bsu_index,
        "upstream_retained_cap": upstream_retained_cap,
        "buffer_input_cap_by_size": buffer_input_cap_by_size,
        "buffer_slew_axis": buffer_slew_axis,
        "buffer_load_axis": buffer_load_axis,
        "buffer_delay_lut": buffer_delay_lut,
        "buffer_output_slew_lut": buffer_output_slew_lut,
        "nmax": nmax,
        "sample_count": sample_count,
    }


def batched_segment_transfer(
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
    return_debug=False,
):
    tensors = _validate_inputs(
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
    input_arrival = tensors["input_arrival"]
    input_slew = tensors["input_slew"]
    downstream_load = tensors["downstream_load"]
    edge_resistance = tensors["edge_resistance"]
    edge_capacitance = tensors["edge_capacitance"]
    repeater_count = tensors["repeater_count"]
    split_fractions = tensors["split_fractions"]
    bsu_index = tensors["bsu_index"]
    upstream_retained_cap = tensors["upstream_retained_cap"]
    nmax = tensors["nmax"]
    sample_count = tensors["sample_count"]
    dtype = input_slew.dtype
    device = input_slew.device

    segment_resistance = edge_resistance.view(-1, 1) * split_fractions
    segment_capacitance = edge_capacitance.view(-1, 1) * split_fractions
    zero = torch.zeros(sample_count, dtype=dtype, device=device)

    no_buffer_delay = edge_resistance * (downstream_load + 0.5 * edge_capacitance)
    no_buffer_slew = _wire_slew(input_slew, no_buffer_delay)
    no_buffer_load = downstream_load + edge_capacitance

    if nmax == 0:
        return {
            "upstream_visible_load": no_buffer_load,
            "segment_delay": no_buffer_delay,
            "output_arrival": input_arrival + no_buffer_delay,
            "output_slew": no_buffer_slew,
        }

    device_lookup = _lookup_device(
        bsu_index=bsu_index,
        input_slew=input_slew,
        output_load=downstream_load,
        buffer_input_cap_by_size=tensors["buffer_input_cap_by_size"],
        buffer_slew_axis=tensors["buffer_slew_axis"],
        buffer_load_axis=tensors["buffer_load_axis"],
        buffer_delay_lut=tensors["buffer_delay_lut"],
        buffer_output_slew_lut=tensors["buffer_output_slew_lut"],
    )
    buffer_input_cap = device_lookup["buffer_input_cap"]

    total_delay = torch.zeros(sample_count, dtype=dtype, device=device)
    slew = input_slew
    wire_delays = []
    buffer_input_slews = []
    buffer_output_loads = []
    buffer_delays = []
    buffer_output_slews = []
    for step in range(nmax):
        active = repeater_count > step
        wire_delay = segment_resistance[:, step] * (
            buffer_input_cap + 0.5 * segment_capacitance[:, step]
        )
        wire_delay = torch.where(active, wire_delay, zero)
        total_delay = total_delay + wire_delay
        next_slew = _wire_slew(slew, wire_delay)
        buffer_input_slews.append(torch.where(active, next_slew, zero))

        has_next_buffer = repeater_count > (step + 1)
        output_load = (
            torch.where(has_next_buffer, buffer_input_cap, downstream_load)
            + segment_capacitance[:, step + 1]
        )
        buffer_output_loads.append(torch.where(active, output_load, zero))
        lookup = _lookup_device(
            bsu_index=bsu_index,
            input_slew=next_slew,
            output_load=output_load,
            buffer_input_cap_by_size=tensors["buffer_input_cap_by_size"],
            buffer_slew_axis=tensors["buffer_slew_axis"],
            buffer_load_axis=tensors["buffer_load_axis"],
            buffer_delay_lut=tensors["buffer_delay_lut"],
            buffer_output_slew_lut=tensors["buffer_output_slew_lut"],
        )
        buffer_delay = torch.where(active, lookup["buffer_delay"], zero)
        buffer_output_slew = torch.where(active, lookup["buffer_output_slew"], slew)
        total_delay = total_delay + buffer_delay
        buffer_delays.append(buffer_delay)
        buffer_output_slews.append(torch.where(active, lookup["buffer_output_slew"], zero))
        slew = torch.where(active, buffer_output_slew, slew)
        wire_delays.append(wire_delay)

    final_index = torch.clamp(repeater_count, max=nmax)
    row_index = torch.arange(sample_count, device=device)
    final_resistance = segment_resistance[row_index, final_index]
    final_capacitance = segment_capacitance[row_index, final_index]
    downstream_wire_delay = final_resistance * (downstream_load + 0.5 * final_capacitance)
    segment_delay = total_delay + downstream_wire_delay
    output_slew = _wire_slew(slew, downstream_wire_delay)
    buffered_load = buffer_input_cap + segment_capacitance[:, 0] + upstream_retained_cap
    upstream_visible_load = torch.where(repeater_count > 0, buffered_load, no_buffer_load)
    segment_delay = torch.where(repeater_count > 0, segment_delay, no_buffer_delay)
    output_slew = torch.where(repeater_count > 0, output_slew, no_buffer_slew)
    output_arrival = input_arrival + segment_delay

    result = {
        "upstream_visible_load": upstream_visible_load,
        "segment_delay": segment_delay,
        "output_arrival": output_arrival,
        "output_slew": output_slew,
        "first_buffer_input_slew": buffer_input_slews[0],
        "first_buffer_output_load": buffer_output_loads[0],
        "first_buffer_delay": buffer_delays[0],
        "first_buffer_output_slew": buffer_output_slews[0],
    }
    if return_debug:
        wire_delays.append(downstream_wire_delay)
        result.update(
            {
                "wire_delays": torch.stack(wire_delays, dim=1),
                "buffer_input_slews": torch.stack(buffer_input_slews, dim=1),
                "buffer_output_loads": torch.stack(buffer_output_loads, dim=1),
                "buffer_delays": torch.stack(buffer_delays, dim=1),
                "buffer_output_slews": torch.stack(buffer_output_slews, dim=1),
                "segment_resistances": segment_resistance,
                "segment_capacitances": segment_capacitance,
            }
        )
    return result
