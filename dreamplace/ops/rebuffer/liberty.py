"""Liberty timing tables extracted from the canonical ECC PyPlaceDB.

The native rebuffer planner owns every selection and repair decision.  This
module only flattens the per-arc LUT lists exported by ``ecc-tools`` into
CSR-style contiguous arrays and attaches the cell attributes needed by the
planner.  No candidate filtering or timing lookup is implemented here.
"""

from __future__ import annotations

import numpy as np

ARTIFACT = "rebuffer_liberty_view"
ARTIFACT_VERSION = 2


def _list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _floats(value):
    return [float(item) for item in _list(value)]


def _ints(value):
    return [int(item) for item in _list(value)]


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dimension(value):
    dims = _ints(value)
    if len(dims) < 2:
        return None
    rows, cols = int(dims[0]), int(dims[1])
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _valid_table(values, trans_axis, cap_axis, dim, expected):
    if dim is None:
        return None
    rows, cols = dim
    if rows * cols != expected:
        return None
    if len(values) != expected or len(trans_axis) != rows or len(cap_axis) != cols:
        return None
    return values


def build_liberty_view_from_pydb(pydb, *, dbu=None):
    """Return the native Liberty view for ``pydb``.

    The view contains every Liberty cell and every combinational timing arc
    with complete rise/fall delay and transition tables.  It is intentionally
    independent of the buffer candidate list: the native planner re-runs
    OpenROAD's ``findBuffers``/``findFastBuffers`` characterization.
    """

    if pydb is None:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["pydb_is_none"],
        }

    dbu = int(dbu if dbu is not None else getattr(pydb, "dbu", 0) or 0)
    if dbu <= 0:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["invalid_dbu"],
        }

    try:
        cell_info = _list(getattr(pydb, "flat_libcell_info", None))
        names = [str(item) for item in _list(getattr(pydb, "flat_libcell_names", None))]
        widths = _floats(getattr(pydb, "flat_libcell_width", ()))
        heights = _floats(getattr(pydb, "flat_libcell_height", ()))
        leakage = _floats(getattr(pydb, "flat_libcell_leakage", ()))
        pin_names = [_text(item) for item in _list(getattr(pydb, "flat_lib_pin_names", ()))]
        pin_start = _ints(getattr(pydb, "cell_id_2_libpin_id_start", ()))
        arc_start = _ints(getattr(pydb, "cell_id_2_arc_id_start", ()))
        pin_cap = _floats(getattr(pydb, "flat_lib_pin_cap", ()))
        pin_rcap = _floats(getattr(pydb, "flat_lib_pin_rcap", ()))
        pin_fcap = _floats(getattr(pydb, "flat_lib_pin_fcap", ()))
        pin_cap_limit = _floats(getattr(pydb, "flat_lib_pin_cap_limit", ()))
        pin_slew_limit = _floats(getattr(pydb, "flat_lib_pin_slew_limit", ()))
        libarc_info = _list(getattr(pydb, "flat_libarc_info", None))
        f_delay_values = _list(getattr(pydb, "f_delay_flat_luts_values", None))
        r_delay_values = _list(getattr(pydb, "r_delay_flat_luts_values", None))
        f_trans_values = _list(getattr(pydb, "f_trans_flat_luts_values", None))
        r_trans_values = _list(getattr(pydb, "r_trans_flat_luts_values", None))
        f_delay_dim = _list(getattr(pydb, "f_delay_flat_luts_dim", None))
        f_delay_trans = _list(getattr(pydb, "f_delay_flat_luts_trans_table", None))
        r_delay_trans = _list(getattr(pydb, "r_delay_flat_luts_trans_table", None))
        f_trans_trans = _list(getattr(pydb, "f_trans_flat_luts_trans_table", None))
        r_trans_trans = _list(getattr(pydb, "r_trans_flat_luts_trans_table", None))
        f_delay_cap = _list(getattr(pydb, "f_delay_flat_luts_cap_table", None))
        r_delay_cap = _list(getattr(pydb, "r_delay_flat_luts_cap_table", None))
        f_trans_cap = _list(getattr(pydb, "f_trans_flat_luts_cap_table", None))
        r_trans_cap = _list(getattr(pydb, "r_trans_flat_luts_cap_table", None))
    except (TypeError, ValueError) as exc:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": [str(exc)],
        }

    num_cells = len(names)
    if (
        num_cells == 0
        or len(cell_info) != num_cells
        or len(pin_start) != num_cells + 1
        or len(arc_start) != num_cells + 1
    ):
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["invalid_cell_or_csr_layout"],
        }

    buffer_main_ids = set(
        _ints(getattr(pydb, "buffer_main_type_candidate_indices", None))
    )

    cell_main_id = np.empty(num_cells, dtype=np.int32)
    cell_size = np.empty(num_cells, dtype=np.int32)
    cell_vt = np.empty(num_cells, dtype=np.int32)
    cell_area = np.empty(num_cells, dtype=np.float64)
    cell_leakage = np.empty(num_cells, dtype=np.float64)
    cell_input_cap = np.empty(num_cells, dtype=np.float64)
    cell_output_cap = np.zeros(num_cells, dtype=np.float64)
    cell_max_cap = np.empty(num_cells, dtype=np.float64)
    cell_max_slew = np.empty(num_cells, dtype=np.float64)
    cell_is_buffer = np.empty(num_cells, dtype=bool)
    cell_arc_start = np.empty(num_cells + 1, dtype=np.int32)
    cell_arc_start[0] = 0

    arc_cell_id = []
    arc_local_id = []
    arc_source = []
    arc_sink = []
    arc_sense = []
    arc_trigger = []
    arc_dim = []
    arc_trans_start = [0]
    arc_trans_axis = []
    arc_cap_start = [0]
    arc_cap_axis = []
    arc_value_start = [0]
    arc_values = []

    for cell_id, row in enumerate(cell_info):
        row = list(row)
        if len(row) < 4:
            return {
                "artifact": ARTIFACT,
                "artifact_version": ARTIFACT_VERSION,
                "status": "unsupported",
                "unsupported_reasons": ["flat_libcell_info_requires_four_columns"],
            }
        main_id = int(row[1])
        size = int(row[2])
        vt = int(row[3])
        cell_main_id[cell_id] = main_id
        cell_size[cell_id] = size
        cell_vt[cell_id] = vt
        if cell_id < len(widths) and cell_id < len(heights):
            cell_area[cell_id] = (
                float(widths[cell_id]) * float(heights[cell_id]) / float(dbu * dbu)
            )
        else:
            cell_area[cell_id] = 0.0
        cell_leakage[cell_id] = (
            float(leakage[cell_id]) if cell_id < len(leakage) else 0.0
        )
        pin_begin = int(pin_start[cell_id])
        pin_end = int(pin_start[cell_id + 1])
        positive_caps = []
        cap_limits = []
        slew_limits = []
        for pin_id in range(pin_begin, min(pin_end, len(pin_cap))):
            if pin_id < len(pin_rcap):
                positive_caps.append(float(pin_rcap[pin_id]))
            if pin_id < len(pin_fcap):
                positive_caps.append(float(pin_fcap[pin_id]))
            if pin_id < len(pin_cap):
                positive_caps.append(float(pin_cap[pin_id]))
            if pin_id < len(pin_cap_limit):
                cap_limits.append(float(pin_cap_limit[pin_id]))
            if pin_id < len(pin_slew_limit):
                slew_limits.append(float(pin_slew_limit[pin_id]))
        cell_input_cap[cell_id] = max(
            [cap for cap in positive_caps if cap > 0.0],
            default=0.0,
        )
        cell_max_cap[cell_id] = max(cap_limits, default=0.0)
        cell_max_slew[cell_id] = max(slew_limits, default=0.0)
        cell_is_buffer[cell_id] = main_id in buffer_main_ids

        begin = int(arc_start[cell_id])
        end = int(arc_start[cell_id + 1])
        sink_names = {
            _text(row[1])
            for row in libarc_info[begin:end]
            if len(row) >= 2
        }
        if sink_names and len(pin_names) >= pin_end:
            output_caps = [
                float(pin_cap[pin_id])
                for pin_id in range(pin_begin, pin_end)
                if pin_names[pin_id] in sink_names
                and pin_id < len(pin_cap)
                and float(pin_cap[pin_id]) > 0.0
            ]
            cell_output_cap[cell_id] = max(output_caps, default=0.0)
        for arc_id in range(begin, min(end, len(libarc_info))):
            if arc_id >= len(f_delay_values) or arc_id >= len(r_delay_values):
                continue
            if arc_id >= len(f_trans_values) or arc_id >= len(r_trans_values):
                continue
            dim = _dimension(f_delay_dim[arc_id] if arc_id < len(f_delay_dim) else None)
            if dim is None:
                continue
            rows, cols = dim
            expected = rows * cols
            f_delay = _valid_table(
                _floats(f_delay_values[arc_id]),
                _floats(f_delay_trans[arc_id]) if arc_id < len(f_delay_trans) else [],
                _floats(f_delay_cap[arc_id]) if arc_id < len(f_delay_cap) else [],
                dim,
                expected,
            )
            r_delay = _valid_table(
                _floats(r_delay_values[arc_id]),
                _floats(r_delay_trans[arc_id]) if arc_id < len(r_delay_trans) else [],
                _floats(r_delay_cap[arc_id]) if arc_id < len(r_delay_cap) else [],
                dim,
                expected,
            )
            f_trans = _valid_table(
                _floats(f_trans_values[arc_id]),
                _floats(f_trans_trans[arc_id]) if arc_id < len(f_trans_trans) else [],
                _floats(f_trans_cap[arc_id]) if arc_id < len(f_trans_cap) else [],
                dim,
                expected,
            )
            r_trans = _valid_table(
                _floats(r_trans_values[arc_id]),
                _floats(r_trans_trans[arc_id]) if arc_id < len(r_trans_trans) else [],
                _floats(r_trans_cap[arc_id]) if arc_id < len(r_trans_cap) else [],
                dim,
                expected,
            )
            if f_delay is None or r_delay is None or f_trans is None or r_trans is None:
                continue
            trans_axis = _floats(f_delay_trans[arc_id])
            cap_axis = _floats(f_delay_cap[arc_id])
            arc_cell_id.append(cell_id)
            arc_local_id.append(int(libarc_info[arc_id][3]))
            arc_source.append(str(libarc_info[arc_id][0]))
            arc_sink.append(str(libarc_info[arc_id][1]))
            arc_sense.append(int(libarc_info[arc_id][4]))
            arc_trigger.append(int(libarc_info[arc_id][5]))
            arc_dim.append(dim)
            arc_trans_axis.extend(trans_axis)
            arc_trans_start.append(len(arc_trans_axis))
            arc_cap_axis.extend(cap_axis)
            arc_cap_start.append(len(arc_cap_axis))
            arc_values.extend(f_delay)
            arc_values.extend(r_delay)
            arc_values.extend(f_trans)
            arc_values.extend(r_trans)
            arc_value_start.append(len(arc_values))
        cell_arc_start[cell_id + 1] = len(arc_cell_id)

    if not arc_cell_id:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["no_two_dimensional_timing_arcs"],
        }

    metadata = getattr(pydb, "repair_library_metadata", None)
    return {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": "ok",
        **({"_native_cell_metadata": metadata} if metadata is not None else {}),
        "dbu": dbu,
        "num_cells": num_cells,
        "num_arcs": len(arc_cell_id),
        "names": names,
        "cell_main_id": np.ascontiguousarray(cell_main_id),
        "cell_size": np.ascontiguousarray(cell_size),
        "cell_vt": np.ascontiguousarray(cell_vt),
        "cell_area": np.ascontiguousarray(cell_area),
        "cell_leakage": np.ascontiguousarray(cell_leakage),
        "cell_input_cap": np.ascontiguousarray(cell_input_cap),
        "cell_output_cap": np.ascontiguousarray(cell_output_cap),
        "cell_max_cap": np.ascontiguousarray(cell_max_cap),
        "cell_max_slew": np.ascontiguousarray(cell_max_slew),
        "cell_is_buffer": np.ascontiguousarray(cell_is_buffer),
        "cell_arc_start": np.ascontiguousarray(cell_arc_start),
        "arc_cell_id": np.ascontiguousarray(arc_cell_id, dtype=np.int32),
        "arc_local_id": np.ascontiguousarray(arc_local_id, dtype=np.int32),
        "arc_source": arc_source,
        "arc_sink": arc_sink,
        "arc_sense": np.ascontiguousarray(arc_sense, dtype=np.int32),
        "arc_trigger": np.ascontiguousarray(arc_trigger, dtype=np.int32),
        "arc_dim": np.ascontiguousarray(arc_dim, dtype=np.int32).reshape(-1),
        "arc_trans_start": np.ascontiguousarray(arc_trans_start, dtype=np.int32),
        "arc_trans_axis": np.ascontiguousarray(arc_trans_axis, dtype=np.float64),
        "arc_cap_start": np.ascontiguousarray(arc_cap_start, dtype=np.int32),
        "arc_cap_axis": np.ascontiguousarray(arc_cap_axis, dtype=np.float64),
        "arc_value_start": np.ascontiguousarray(arc_value_start, dtype=np.int32),
        "arc_values": np.ascontiguousarray(arc_values, dtype=np.float64),
        "summary": {
            "status": "ok",
            "dbu": dbu,
            "num_cells": num_cells,
            "num_arcs": len(arc_cell_id),
            "num_buffer_main_types": len(buffer_main_ids),
            "num_buffer_cells": int(cell_is_buffer.sum()),
        },
    }
