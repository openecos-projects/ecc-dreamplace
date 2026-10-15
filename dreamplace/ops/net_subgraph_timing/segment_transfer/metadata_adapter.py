import torch

from dreamplace.ops.buffer_insertion.buffer_library import (
    lookup_lut_value_with_status_mode,
)

from .schema import BufferDeviceLut


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _positive_input_cap(metadata, cell_id, timing_sense=None):
    pin_start = [int(value) for value in _as_list(getattr(metadata, "cell_id_2_libpin_id_start", []))]
    field = {"rise": "flat_lib_pin_rcap", "fall": "flat_lib_pin_fcap"}.get(
        timing_sense, "flat_lib_pin_cap"
    )
    # Older metadata exports only the mean input cap; the table source records it.
    cap_values = getattr(metadata, field, None)
    if cap_values is None:
        cap_values = getattr(metadata, "flat_lib_pin_cap", [])
    pin_caps = [float(value) for value in _as_list(cap_values)]
    if int(cell_id) < 0 or int(cell_id) + 1 >= len(pin_start):
        raise ValueError(f"missing libpin range for cell_id={int(cell_id)}")
    begin = int(pin_start[int(cell_id)])
    end = int(pin_start[int(cell_id) + 1])
    caps = [
        float(pin_caps[index])
        for index in range(begin, min(end, len(pin_caps)))
        if float(pin_caps[index]) > 0.0
    ]
    if not caps:
        raise ValueError(f"missing positive input cap for cell_id={int(cell_id)}")
    return min(caps)


def _arc_ids_for_cell(metadata, cell_id):
    arc_start = [int(value) for value in _as_list(getattr(metadata, "cell_id_2_arc_id_start", []))]
    if int(cell_id) < 0 or int(cell_id) + 1 >= len(arc_start):
        raise ValueError(f"missing libarc range for cell_id={int(cell_id)}")
    return list(range(int(arc_start[int(cell_id)]), int(arc_start[int(cell_id) + 1])))


def _output_limit(metadata, cell_id, field):
    limits = _as_list(getattr(metadata, field, []))
    starts = _as_list(metadata.cell_id_2_libpin_id_start)
    if not limits:
        return None
    caps = _as_list(metadata.flat_lib_pin_cap)
    # The buffer contract contains one input with positive capacitance and one
    # zero-cap output. Raw native arc endpoints are names, not libpin offsets.
    output_pins = [index for index in range(int(starts[cell_id]), int(starts[cell_id + 1]))
                   if float(caps[index]) == 0.0]
    if len(output_pins) != 1:
        raise ValueError(f"buffer cell {cell_id} must have one zero-cap output pin")
    pin_id = output_pins[0]
    if pin_id >= len(limits):
        raise ValueError(f"buffer output limit outside Liberty pin domain: {cell_id}")
    value = float(limits[pin_id])
    return value if value > 0.0 and torch.isfinite(torch.tensor(value)) else None


def _arc_lut(metadata, prefix, table_kind, arc_id):
    values = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_values", []))
    trans = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_trans_table", []))
    cap = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_cap_table", []))
    dims = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_dim", []))
    arc_id = int(arc_id)
    if arc_id < 0 or arc_id >= len(values):
        raise ValueError(f"missing {prefix} {table_kind} LUT for arc_id={arc_id}")
    dim = [int(value) for value in _as_list(dims[arc_id] if arc_id < len(dims) else [])]
    if len(dim) < 2 or dim[0] <= 0 or dim[1] <= 0:
        raise ValueError(f"missing 2-D {prefix} {table_kind} LUT for arc_id={arc_id}")
    trans_axis = [float(value) for value in _as_list(trans[arc_id] if arc_id < len(trans) else [])]
    cap_axis = [float(value) for value in _as_list(cap[arc_id] if arc_id < len(cap) else [])]
    flat_values = [float(value) for value in _as_list(values[arc_id])]
    expected = int(dim[0]) * int(dim[1])
    if len(trans_axis) != int(dim[0]) or len(cap_axis) != int(dim[1]):
        raise ValueError(f"{prefix} {table_kind} LUT axes do not match dim for arc_id={arc_id}")
    if len(flat_values) < expected:
        raise ValueError(f"{prefix} {table_kind} LUT values shorter than dim for arc_id={arc_id}")
    return trans_axis, cap_axis, flat_values[:expected], tuple(dim[:2])


def _same_axis(left, right):
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(left, right))


def _cell_average_lut(metadata, cell_id, table_kind, timing_sense=None):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    axis_slew = None
    axis_cap = None
    tables = []
    for arc_id in arc_ids:
        for prefix in ({"rise": ("r",), "fall": ("f",)}.get(timing_sense, ("f", "r"))):
            slew_axis, cap_axis, values, dim = _arc_lut(metadata, prefix, table_kind, arc_id)
            if axis_slew is None:
                axis_slew = slew_axis
                axis_cap = cap_axis
            elif not _same_axis(axis_slew, slew_axis) or not _same_axis(axis_cap, cap_axis):
                raise ValueError(
                    f"{table_kind} LUT axes must match across rise/fall arcs for cell_id={int(cell_id)}"
                )
            tables.append(torch.tensor(values, dtype=torch.float64).view(int(dim[0]), int(dim[1])))
    if not tables:
        raise ValueError(f"missing {table_kind} LUTs for cell_id={int(cell_id)}")
    return axis_slew, axis_cap, torch.stack(tables).mean(dim=0)


def _cell_arc_luts(metadata, cell_id):
    rows = []
    for arc_id in _arc_ids_for_cell(metadata, cell_id):
        for prefix in ("f", "r"):
            delay_slew_axis, delay_cap_axis, delay_values, delay_dim = _arc_lut(
                metadata,
                prefix,
                "delay",
                arc_id,
            )
            trans_slew_axis, trans_cap_axis, trans_values, trans_dim = _arc_lut(
                metadata,
                prefix,
                "trans",
                arc_id,
            )
            if (
                not _same_axis(delay_slew_axis, trans_slew_axis)
                or not _same_axis(delay_cap_axis, trans_cap_axis)
            ):
                raise ValueError(
                    f"delay/trans LUT axes must match for arc_id={int(arc_id)}"
                )
            rows.append(
                {
                    "arc_id": int(arc_id),
                    "prefix": str(prefix),
                    "input_slew_axis": list(delay_slew_axis),
                    "output_load_axis": list(delay_cap_axis),
                    "delay_lut": torch.tensor(
                        delay_values,
                        dtype=torch.float64,
                    ).view(int(delay_dim[0]), int(delay_dim[1])),
                    "output_slew_lut": torch.tensor(
                        trans_values,
                        dtype=torch.float64,
                    ).view(int(trans_dim[0]), int(trans_dim[1])),
                }
            )
    return rows


def _merge_axis(axes):
    values = sorted(
        {
            float(value)
            for axis in axes
            for value in axis
            if torch.isfinite(torch.tensor(float(value)))
        }
    )
    if not values:
        raise ValueError("cannot build shared LUT axis from empty axes")
    return values


def _resample_lut(lut, source_slew_axis, source_cap_axis, target_slew_axis, target_cap_axis):
    flat = [float(value) for value in lut.reshape(-1).tolist()]
    dim = [len(source_slew_axis), len(source_cap_axis)]
    rows = []
    for slew in target_slew_axis:
        row = []
        for cap in target_cap_axis:
            value, _status = lookup_lut_value_with_status_mode(
                values=flat,
                trans_axis=source_slew_axis,
                cap_axis=source_cap_axis,
                dim=dim,
                input_slew=float(slew),
                output_cap=float(cap),
                boundary_mode="extrapolate",
            )
            row.append(value)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def build_buffer_device_lut_from_metadata(
    metadata, contract_artifact, *, dtype=torch.float64, timing_sense=None
):
    if timing_sense not in (None, "rise", "fall"):
        raise ValueError("buffer timing sense must be rise or fall")
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before building device LUT")
    legal_cell_ids = [int(value) for value in contract_artifact.get("legal_cell_ids", [])]
    if not legal_cell_ids:
        raise ValueError("buffer legal cell table is empty")

    input_caps = []
    raw_delay_tables = []
    raw_output_slew_tables = []
    arc_luts_by_size = []
    slew_limits = []
    cap_limits = []
    slew_axes = []
    cap_axes = []
    for cell_id in legal_cell_ids:
        input_caps.append(_positive_input_cap(metadata, cell_id, timing_sense))
        arc_luts_by_size.append(_cell_arc_luts(metadata, cell_id))
        slew_limits.append(_output_limit(metadata, cell_id, "flat_lib_pin_slew_limit"))
        cap_limits.append(_output_limit(metadata, cell_id, "flat_lib_pin_cap_limit"))
        delay_slew_axis, delay_cap_axis, delay_lut = _cell_average_lut(
            metadata,
            cell_id,
            "delay",
            timing_sense,
        )
        trans_slew_axis, trans_cap_axis, trans_lut = _cell_average_lut(
            metadata,
            cell_id,
            "trans",
            timing_sense,
        )
        if not _same_axis(delay_slew_axis, trans_slew_axis) or not _same_axis(delay_cap_axis, trans_cap_axis):
            raise ValueError(f"delay/trans LUT axes must match for cell_id={int(cell_id)}")
        slew_axes.append(delay_slew_axis)
        cap_axes.append(delay_cap_axis)
        raw_delay_tables.append((delay_slew_axis, delay_cap_axis, delay_lut))
        raw_output_slew_tables.append((trans_slew_axis, trans_cap_axis, trans_lut))

    shared_slew_axis = _merge_axis(slew_axes)
    shared_cap_axis = _merge_axis(cap_axes)
    delay_tables = [
        _resample_lut(lut, slew_axis, cap_axis, shared_slew_axis, shared_cap_axis)
        for slew_axis, cap_axis, lut in raw_delay_tables
    ]
    output_slew_tables = [
        _resample_lut(lut, slew_axis, cap_axis, shared_slew_axis, shared_cap_axis)
        for slew_axis, cap_axis, lut in raw_output_slew_tables
    ]

    phase_luts = (
        None if timing_sense is not None else {
            sense: build_buffer_device_lut_from_metadata(
                metadata, contract_artifact, dtype=dtype, timing_sense=sense
            ) for sense in ("rise", "fall")
        }
    )
    cap_source = (
        "phase_input_cap" if timing_sense is not None and hasattr(
            metadata,
            {"rise": "flat_lib_pin_rcap", "fall": "flat_lib_pin_fcap"}[timing_sense]
        ) else "mean_input_cap"
    )
    return BufferDeviceLut(
        input_cap_by_size=torch.tensor(input_caps, dtype=dtype),
        input_slew_axis=torch.tensor(shared_slew_axis, dtype=dtype),
        output_load_axis=torch.tensor(shared_cap_axis, dtype=dtype),
        delay_lut=torch.stack(delay_tables).to(dtype=dtype),
        output_slew_lut=torch.stack(output_slew_tables).to(dtype=dtype),
        source=(
            "metadata_contract_lut" if timing_sense is None
            else f"metadata_{timing_sense}_lut:{cap_source}"
        ),
        arc_luts_by_size=arc_luts_by_size,
        phase_luts=phase_luts,
        output_slew_limit_by_size=torch.tensor([
            torch.finfo(dtype).max if limit is None else limit for limit in slew_limits
        ], dtype=dtype),
        output_cap_limit_by_size=torch.tensor([
            torch.finfo(dtype).max if limit is None else limit for limit in cap_limits
        ], dtype=dtype),
        drv_limit_coverage={
            "slew": [value is not None for value in slew_limits],
            "cap": [value is not None for value in cap_limits],
            "missing_limit_policy": "unbounded_and_reported",
        },
    )
