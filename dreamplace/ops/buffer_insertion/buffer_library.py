import math


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _flatten_numeric(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_numeric(item))
        return values
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return []
    return [numeric] if math.isfinite(numeric) else []


def _finite_numeric_list(value):
    return [float(item) for item in _flatten_numeric(value)]


def _axis_bracket(axis, point):
    lo, hi, alpha, _status = _axis_bracket_with_status(axis, point)
    return lo, hi, alpha


def _axis_bracket_with_status(axis, point, *, boundary_mode="clamp"):
    axis = [float(value) for value in axis if math.isfinite(float(value))]
    empty_status = {
        "clamped": False,
        "clamp": "none",
        "axis_min": None,
        "axis_max": None,
    }
    if not axis:
        return 0, 0, 0.0, empty_status
    axis_min = float(axis[0])
    axis_max = float(axis[-1])
    point = float(point)
    extrapolate = str(boundary_mode or "clamp").strip().lower() == "extrapolate"
    if point <= axis_min:
        if extrapolate and len(axis) > 1 and point < axis_min:
            denom = axis[1] - axis[0]
            alpha = 0.0 if denom == 0.0 else (point - axis[0]) / denom
            return 0, 1, alpha, {
                "clamped": True,
                "clamp": "low",
                "axis_min": axis_min,
                "axis_max": axis_max,
            }
        return 0, 0, 0.0, {
            "clamped": point < axis_min,
            "clamp": "low" if point < axis_min else "none",
            "axis_min": axis_min,
            "axis_max": axis_max,
        }
    if point >= axis_max:
        last = len(axis) - 1
        if extrapolate and len(axis) > 1 and point > axis_max:
            lo = last - 1
            denom = axis[last] - axis[lo]
            alpha = 0.0 if denom == 0.0 else (point - axis[lo]) / denom
            return lo, last, alpha, {
                "clamped": True,
                "clamp": "high",
                "axis_min": axis_min,
                "axis_max": axis_max,
            }
        return last, last, 0.0, {
            "clamped": point > axis_max,
            "clamp": "high" if point > axis_max else "none",
            "axis_min": axis_min,
            "axis_max": axis_max,
        }
    for index in range(len(axis) - 1):
        lo = axis[index]
        hi = axis[index + 1]
        if lo <= point <= hi:
            denom = hi - lo
            alpha = 0.0 if denom == 0.0 else (point - lo) / denom
            return index, index + 1, alpha, {
                "clamped": False,
                "clamp": "none",
                "axis_min": axis_min,
                "axis_max": axis_max,
            }
    last = len(axis) - 1
    return last, last, 0.0, {
        "clamped": False,
        "clamp": "none",
        "axis_min": axis_min,
        "axis_max": axis_max,
    }


def lookup_lut_value(*, values, trans_axis, cap_axis, dim, input_slew, output_cap):
    value, _status = lookup_lut_value_with_status(
        values=values,
        trans_axis=trans_axis,
        cap_axis=cap_axis,
        dim=dim,
        input_slew=input_slew,
        output_cap=output_cap,
    )
    return value


def _lut_status(trans_status, cap_status):
    return {
        "input_slew_clamped": bool(trans_status["clamped"]),
        "input_slew_clamp": str(trans_status["clamp"]),
        "input_slew_axis_min": trans_status["axis_min"],
        "input_slew_axis_max": trans_status["axis_max"],
        "output_cap_clamped": bool(cap_status["clamped"]),
        "output_cap_clamp": str(cap_status["clamp"]),
        "output_cap_axis_min": cap_status["axis_min"],
        "output_cap_axis_max": cap_status["axis_max"],
    }


def lookup_lut_value_with_status(*, values, trans_axis, cap_axis, dim, input_slew, output_cap):
    return lookup_lut_value_with_status_mode(
        values=values,
        trans_axis=trans_axis,
        cap_axis=cap_axis,
        dim=dim,
        input_slew=input_slew,
        output_cap=output_cap,
        boundary_mode="clamp",
    )


def lookup_lut_value_with_status_mode(
    *,
    values,
    trans_axis,
    cap_axis,
    dim,
    input_slew,
    output_cap,
    boundary_mode="clamp",
):
    values = _finite_numeric_list(values)
    dims = [int(value) for value in _as_list(dim)]
    trans_dim = dims[0] if len(dims) >= 1 else len(_as_list(trans_axis))
    cap_dim = dims[1] if len(dims) >= 2 else len(_as_list(cap_axis))
    if not values:
        raise ValueError("missing LUT values")

    empty_status = _lut_status(
        {"clamped": False, "clamp": "none", "axis_min": None, "axis_max": None},
        {"clamped": False, "clamp": "none", "axis_min": None, "axis_max": None},
    )
    if trans_dim <= 0 and cap_dim <= 0:
        return float(values[0]), empty_status
    if trans_dim <= 0:
        cap_axis = _as_list(cap_axis)
        cap0, cap1, alpha, cap_status = _axis_bracket_with_status(
            cap_axis,
            output_cap,
            boundary_mode=boundary_mode,
        )
        if cap0 >= len(values) or cap1 >= len(values):
            raise ValueError("cap LUT axis does not match values")
        return float(values[cap0] * (1.0 - alpha) + values[cap1] * alpha), _lut_status(
            _empty_axis_status(),
            cap_status,
        )
    if cap_dim <= 0:
        trans_axis = _as_list(trans_axis)
        tr0, tr1, alpha, trans_status = _axis_bracket_with_status(
            trans_axis,
            input_slew,
            boundary_mode=boundary_mode,
        )
        if tr0 >= len(values) or tr1 >= len(values):
            raise ValueError("transition LUT axis does not match values")
        return float(values[tr0] * (1.0 - alpha) + values[tr1] * alpha), _lut_status(
            trans_status,
            _empty_axis_status(),
        )

    expected = int(trans_dim) * int(cap_dim)
    if len(values) < expected:
        raise ValueError("2-D LUT values shorter than dimensions")
    tr0, tr1, tr_alpha, trans_status = _axis_bracket_with_status(
        _as_list(trans_axis),
        input_slew,
        boundary_mode=boundary_mode,
    )
    cap0, cap1, cap_alpha, cap_status = _axis_bracket_with_status(
        _as_list(cap_axis),
        output_cap,
        boundary_mode=boundary_mode,
    )
    if tr0 >= trans_dim or tr1 >= trans_dim or cap0 >= cap_dim or cap1 >= cap_dim:
        raise ValueError("2-D LUT axis does not match dimensions")

    def at(tr_index, cap_index):
        return values[int(tr_index) * int(cap_dim) + int(cap_index)]

    v00 = at(tr0, cap0)
    v01 = at(tr0, cap1)
    v10 = at(tr1, cap0)
    v11 = at(tr1, cap1)
    v0 = v00 * (1.0 - cap_alpha) + v01 * cap_alpha
    v1 = v10 * (1.0 - cap_alpha) + v11 * cap_alpha
    return float(v0 * (1.0 - tr_alpha) + v1 * tr_alpha), _lut_status(
        trans_status,
        cap_status,
    )


def _empty_axis_status():
    return {
        "clamped": False,
        "clamp": "none",
        "axis_min": None,
        "axis_max": None,
    }


def _combine_lut_status(statuses):
    statuses = list(statuses or [])
    if not statuses:
        return _lut_status(_empty_axis_status(), _empty_axis_status())

    def combine_axis(prefix):
        clamps = [str(status.get(f"{prefix}_clamp", "none")) for status in statuses]
        clamp = "none"
        if "low" in clamps and "high" in clamps:
            clamp = "mixed"
        elif "low" in clamps:
            clamp = "low"
        elif "high" in clamps:
            clamp = "high"
        mins = [
            status.get(f"{prefix}_axis_min")
            for status in statuses
            if status.get(f"{prefix}_axis_min") is not None
        ]
        maxs = [
            status.get(f"{prefix}_axis_max")
            for status in statuses
            if status.get(f"{prefix}_axis_max") is not None
        ]
        return {
            f"{prefix}_clamped": any(
                bool(status.get(f"{prefix}_clamped", False))
                for status in statuses
            ),
            f"{prefix}_clamp": clamp,
            f"{prefix}_axis_min": min(mins) if mins else None,
            f"{prefix}_axis_max": max(maxs) if maxs else None,
        }

    combined = {}
    combined.update(combine_axis("input_slew"))
    combined.update(combine_axis("output_cap"))
    return combined


def _positive_input_cap(metadata, cell_id):
    pin_start = [int(value) for value in _as_list(getattr(metadata, "cell_id_2_libpin_id_start", []))]
    pin_caps = [float(value) for value in _as_list(getattr(metadata, "flat_lib_pin_cap", []))]
    cell_id = int(cell_id)
    if cell_id < 0 or cell_id + 1 >= len(pin_start):
        raise ValueError(f"missing libpin range for cell_id={cell_id}")
    begin = int(pin_start[cell_id])
    end = int(pin_start[cell_id + 1])
    caps = [
        float(pin_caps[idx])
        for idx in range(begin, min(end, len(pin_caps)))
        if float(pin_caps[idx]) > 0.0 and math.isfinite(float(pin_caps[idx]))
    ]
    if not caps:
        raise ValueError(f"missing positive input cap for cell_id={cell_id}")
    return min(caps)


def _arc_ids_for_cell(metadata, cell_id):
    arc_start = [int(value) for value in _as_list(getattr(metadata, "cell_id_2_arc_id_start", []))]
    cell_id = int(cell_id)
    if cell_id < 0 or cell_id + 1 >= len(arc_start):
        raise ValueError(f"missing libarc range for cell_id={cell_id}")
    return list(range(int(arc_start[cell_id]), int(arc_start[cell_id + 1])))


def _delay_values_for_arc_ids(metadata, arc_ids):
    fields = (
        "f_delay_flat_luts_values",
        "r_delay_flat_luts_values",
    )
    values = []
    for field_name in fields:
        table = _as_list(getattr(metadata, field_name, []))
        for arc_id in arc_ids:
            if 0 <= int(arc_id) < len(table):
                values.extend(_flatten_numeric(table[int(arc_id)]))
    return [value for value in values if math.isfinite(value)]


def _delay_proxy(metadata, cell_id):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    values = _delay_values_for_arc_ids(metadata, arc_ids)
    if not values:
        raise ValueError(f"missing delay LUT values for cell_id={cell_id}")
    return sum(values) / float(len(values))


def _lookup_table(metadata, prefix, table_kind, arc_id, *, input_slew, output_cap):
    values = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_values", []))
    trans = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_trans_table", []))
    cap = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_cap_table", []))
    dim = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_dim", []))
    arc_id = int(arc_id)
    if arc_id < 0 or arc_id >= len(values):
        raise ValueError(f"missing {prefix} {table_kind} LUT for arc_id={arc_id}")
    arc_dim = dim[arc_id] if arc_id < len(dim) else []
    arc_dims = [int(value) for value in _as_list(arc_dim)]
    if len(arc_dims) < 2 or arc_dims[0] <= 0 or arc_dims[1] <= 0:
        raise ValueError(f"missing 2-D {prefix} {table_kind} LUT for arc_id={arc_id}")
    return lookup_lut_value(
        values=values[arc_id],
        trans_axis=trans[arc_id] if arc_id < len(trans) else [],
        cap_axis=cap[arc_id] if arc_id < len(cap) else [],
        dim=arc_dim,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def _lookup_table_with_status(metadata, prefix, table_kind, arc_id, *, input_slew, output_cap):
    values = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_values", []))
    trans = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_trans_table", []))
    cap = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_cap_table", []))
    dim = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_dim", []))
    arc_id = int(arc_id)
    if arc_id < 0 or arc_id >= len(values):
        raise ValueError(f"missing {prefix} {table_kind} LUT for arc_id={arc_id}")
    arc_dim = dim[arc_id] if arc_id < len(dim) else []
    arc_dims = [int(value) for value in _as_list(arc_dim)]
    if len(arc_dims) < 2 or arc_dims[0] <= 0 or arc_dims[1] <= 0:
        raise ValueError(f"missing 2-D {prefix} {table_kind} LUT for arc_id={arc_id}")
    return lookup_lut_value_with_status(
        values=values[arc_id],
        trans_axis=trans[arc_id] if arc_id < len(trans) else [],
        cap_axis=cap[arc_id] if arc_id < len(cap) else [],
        dim=arc_dim,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def _lookup_delay_table(metadata, prefix, arc_id, *, input_slew, output_cap):
    return _lookup_table(
        metadata,
        prefix,
        "delay",
        arc_id,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def _lookup_delay_table_with_status(metadata, prefix, arc_id, *, input_slew, output_cap):
    return _lookup_table_with_status(
        metadata,
        prefix,
        "delay",
        arc_id,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def _lookup_transition_table(metadata, prefix, arc_id, *, input_slew, output_cap):
    return _lookup_table(
        metadata,
        prefix,
        "trans",
        arc_id,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def _lookup_transition_table_with_status(metadata, prefix, arc_id, *, input_slew, output_cap):
    return _lookup_table_with_status(
        metadata,
        prefix,
        "trans",
        arc_id,
        input_slew=input_slew,
        output_cap=output_cap,
    )


def lookup_buffer_delay_from_metadata(
    metadata,
    *,
    cell_id,
    input_slew,
    output_cap,
):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    values = []
    for arc_id in arc_ids:
        values.append(
            _lookup_delay_table(
                metadata,
                "f",
                arc_id,
                input_slew=input_slew,
                output_cap=output_cap,
            )
        )
        values.append(
            _lookup_delay_table(
                metadata,
                "r",
                arc_id,
                input_slew=input_slew,
                output_cap=output_cap,
            )
        )
    finite_values = [value for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError(f"missing finite delay LUT values for cell_id={cell_id}")
    return sum(finite_values) / float(len(finite_values)), "lut_bilinear_interpolation"


def lookup_buffer_delay_from_metadata_with_status(
    metadata,
    *,
    cell_id,
    input_slew,
    output_cap,
):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    values = []
    statuses = []
    for arc_id in arc_ids:
        value, status = _lookup_delay_table_with_status(
            metadata,
            "f",
            arc_id,
            input_slew=input_slew,
            output_cap=output_cap,
        )
        values.append(value)
        statuses.append(status)
        value, status = _lookup_delay_table_with_status(
            metadata,
            "r",
            arc_id,
            input_slew=input_slew,
            output_cap=output_cap,
        )
        values.append(value)
        statuses.append(status)
    finite_values = [value for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError(f"missing finite delay LUT values for cell_id={cell_id}")
    return (
        sum(finite_values) / float(len(finite_values)),
        "lut_bilinear_interpolation",
        _combine_lut_status(statuses),
    )


def lookup_buffer_transition_from_metadata(
    metadata,
    *,
    cell_id,
    input_slew,
    output_cap,
):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    values = []
    for arc_id in arc_ids:
        values.append(
            _lookup_transition_table(
                metadata,
                "f",
                arc_id,
                input_slew=input_slew,
                output_cap=output_cap,
            )
        )
        values.append(
            _lookup_transition_table(
                metadata,
                "r",
                arc_id,
                input_slew=input_slew,
                output_cap=output_cap,
            )
        )
    finite_values = [value for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError(f"missing finite transition LUT values for cell_id={cell_id}")
    return sum(finite_values) / float(len(finite_values)), "lut_bilinear_interpolation"


def lookup_buffer_transition_from_metadata_with_status(
    metadata,
    *,
    cell_id,
    input_slew,
    output_cap,
):
    arc_ids = _arc_ids_for_cell(metadata, cell_id)
    values = []
    statuses = []
    for arc_id in arc_ids:
        value, status = _lookup_transition_table_with_status(
            metadata,
            "f",
            arc_id,
            input_slew=input_slew,
            output_cap=output_cap,
        )
        values.append(value)
        statuses.append(status)
        value, status = _lookup_transition_table_with_status(
            metadata,
            "r",
            arc_id,
            input_slew=input_slew,
            output_cap=output_cap,
        )
        values.append(value)
        statuses.append(status)
    finite_values = [value for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError(f"missing finite transition LUT values for cell_id={cell_id}")
    return (
        sum(finite_values) / float(len(finite_values)),
        "lut_bilinear_interpolation",
        _combine_lut_status(statuses),
    )


def lookup_buffer_delay_for_bsu(
    metadata,
    contract_artifact,
    *,
    bsu,
    input_slew,
    output_cap,
):
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before delay lookup")
    legal_cell_ids = [int(cell_id) for cell_id in contract_artifact.get("legal_cell_ids", [])]
    bsu = int(bsu)
    if bsu < 0 or bsu >= len(legal_cell_ids):
        raise ValueError(f"bsu={bsu} is outside legal buffer table")
    return lookup_buffer_delay_from_metadata(
        metadata,
        cell_id=legal_cell_ids[bsu],
        input_slew=input_slew,
        output_cap=output_cap,
    )


def lookup_buffer_delay_for_bsu_with_status(
    metadata,
    contract_artifact,
    *,
    bsu,
    input_slew,
    output_cap,
):
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before delay lookup")
    legal_cell_ids = [int(cell_id) for cell_id in contract_artifact.get("legal_cell_ids", [])]
    bsu = int(bsu)
    if bsu < 0 or bsu >= len(legal_cell_ids):
        raise ValueError(f"bsu={bsu} is outside legal buffer table")
    return lookup_buffer_delay_from_metadata_with_status(
        metadata,
        cell_id=legal_cell_ids[bsu],
        input_slew=input_slew,
        output_cap=output_cap,
    )


def lookup_buffer_transition_for_bsu(
    metadata,
    contract_artifact,
    *,
    bsu,
    input_slew,
    output_cap,
):
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before transition lookup")
    legal_cell_ids = [int(cell_id) for cell_id in contract_artifact.get("legal_cell_ids", [])]
    bsu = int(bsu)
    if bsu < 0 or bsu >= len(legal_cell_ids):
        raise ValueError(f"bsu={bsu} is outside legal buffer table")
    return lookup_buffer_transition_from_metadata(
        metadata,
        cell_id=legal_cell_ids[bsu],
        input_slew=input_slew,
        output_cap=output_cap,
    )


def lookup_buffer_transition_for_bsu_with_status(
    metadata,
    contract_artifact,
    *,
    bsu,
    input_slew,
    output_cap,
):
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before transition lookup")
    legal_cell_ids = [int(cell_id) for cell_id in contract_artifact.get("legal_cell_ids", [])]
    bsu = int(bsu)
    if bsu < 0 or bsu >= len(legal_cell_ids):
        raise ValueError(f"bsu={bsu} is outside legal buffer table")
    return lookup_buffer_transition_from_metadata_with_status(
        metadata,
        cell_id=legal_cell_ids[bsu],
        input_slew=input_slew,
        output_cap=output_cap,
    )


def build_buffer_library_from_metadata(metadata, contract_artifact):
    if contract_artifact.get("status") != "ok":
        raise ValueError("buffer family contract must be ok before building buffer library")

    legal_cell_ids = [int(cell_id) for cell_id in contract_artifact.get("legal_cell_ids", [])]
    legal_master_names = [str(name) for name in contract_artifact.get("legal_master_names", [])]
    legal_size_values = [float(value) for value in contract_artifact.get("legal_size_values", [])]
    if not legal_cell_ids:
        raise ValueError("buffer legal cell table is empty")

    library = {}
    delay_source_counts = {}
    for bsu, cell_id in enumerate(legal_cell_ids):
        master_name = legal_master_names[bsu] if bsu < len(legal_master_names) else ""
        size_value = legal_size_values[bsu] if bsu < len(legal_size_values) else float(bsu)
        input_cap = _positive_input_cap(metadata, cell_id)
        try:
            delay, delay_source = lookup_buffer_delay_from_metadata(
                metadata,
                cell_id=cell_id,
                input_slew=0.0,
                output_cap=input_cap,
            )
        except ValueError:
            delay = _delay_proxy(metadata, cell_id)
            delay_source = "contract_metadata_proxy_average"
        try:
            output_slew, transition_source = lookup_buffer_transition_from_metadata(
                metadata,
                cell_id=cell_id,
                input_slew=0.0,
                output_cap=input_cap,
            )
        except ValueError:
            output_slew = 0.0
            transition_source = "missing_transition_lut"
        delay_source_counts[delay_source] = delay_source_counts.get(delay_source, 0) + 1
        library[int(bsu)] = {
            "cell_id": int(cell_id),
            "master_name": master_name,
            "legal_size_value": float(size_value),
            "input_cap": input_cap,
            "delay": delay,
            "output_slew": output_slew,
            "delay_source": delay_source,
            "transition_source": transition_source,
            "source": "contract_metadata_proxy",
        }

    return library, {
        "artifact": "buffering_buffer_library_summary",
        "artifact_version": 1,
        "status": "ok",
        "source": "contract_metadata_proxy",
        "entry_count": len(library),
        "bsu_values": sorted(library),
        "delay_source_counts": delay_source_counts,
    }
