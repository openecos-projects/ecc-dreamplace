"""Build the native rebuffer design view from a placement PyPlaceDB.

This module is intentionally a data boundary.  It copies the canonical
``placedb.pydb`` arrays into compact, contiguous arrays and validates their
shape.  ``flat_libcell_info`` is normalized to the numeric
``[main_id, size, vt]`` columns; the cell name remains in
``flat_libcell_names``.  Tree construction, candidate selection, and repair
decisions belong to ``rebuffer_cpp``.
"""

import numpy as np

ARTIFACT = "rebuffer_pydb_design_view"
ARTIFACT_VERSION = 1


def _as_numpy(value, dtype, *, field, allow_empty=False):
    if value is None:
        array = np.empty(0, dtype=dtype)
    else:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} cannot be converted to {dtype}") from exc
    array = np.ascontiguousarray(array)
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{field} is required")
    return array


def _as_numpy_2d(value, dtype, *, field, columns=None):
    if value is None:
        return np.empty((0, 0), dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} cannot be converted to {dtype}") from exc
    array = np.ascontiguousarray(array)
    if array.size == 0:
        return np.empty((0, 0 if columns is None else columns), dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{field} must be two-dimensional")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{field} must have {columns} columns")
    return array


def _as_optional_numpy(value, dtype, *, field):
    if value is None:
        return None
    array = _as_numpy(value, dtype, field=field, allow_empty=True)
    return array


def _buffer_metadata(pydb, node_count):
    """Return explicit or family-derived per-node buffer metadata.

    Some physical PyPlaceDB exports omit ``node_is_buffer`` while still
    exporting buffer main-type families and per-node main ids.  Deriving the
    boolean mask here keeps the native boundary complete without moving tree
    traversal or repair decisions into Python.
    """

    direct = _as_optional_numpy(
        getattr(pydb, "node_is_buffer", None),
        bool,
        field="node_is_buffer",
    )
    if direct is not None:
        return direct, "node_is_buffer"
    families = getattr(pydb, "buffer_main_type_candidate_indices", None)
    inst_main = getattr(pydb, "inst_main_id", None)
    if families is None or inst_main is None:
        return None, "none"
    main = _as_numpy(inst_main, np.int32, field="inst_main_id")
    if main.size != node_count:
        return None, "none"
    family_ids = np.asarray(list(families), dtype=np.int32).ravel()
    if family_ids.size == 0:
        return np.zeros(node_count, dtype=bool), "buffer_main_type_candidate_indices"
    return np.isin(main, family_ids), "buffer_main_type_candidate_indices"


_ORIENTATION_CODES = {
    "R0": 0,
    "R90": 1,
    "R180": 2,
    "R270": 3,
    "MX": 4,
    "MY": 5,
    "MXR90": 6,
    "MYR90": 7,
}


def _as_optional_pin_orientation(value, *, field):
    """Accept PyPlaceDB orientation strings or numeric direction codes.

    ``PyPlaceDB.pin_direct`` is the instance orientation string in the
    current ECC export (for example ``"N_R0"``), despite its historical
    name.  Preserve it as a stable numeric code so the native boundary does
    not reject an otherwise usable placement view.
    """

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    items = list(value)
    if not items:
        return np.empty(0, dtype=np.int8)
    numeric = []
    for item in items:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            normalized = item.strip().upper()
            if "_" in normalized:
                normalized = normalized.rsplit("_", 1)[-1]
            numeric.append(_ORIENTATION_CODES.get(normalized, -1))
        else:
            try:
                numeric.append(int(item))
            except (TypeError, ValueError):
                numeric.append(-1)
    return np.ascontiguousarray(numeric, dtype=np.int8)


def _as_names(value):
    if value is None:
        return ()
    names = []
    for item in list(value):
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        names.append(str(item))
    return tuple(names)


def _as_libcell_info(value):
    """Normalize PyPlaceDB's mixed name/numeric libcell rows."""

    if value is None:
        return np.empty((0, 3), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    rows = np.asarray(value, dtype=object)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("flat_libcell_info must be two-dimensional")
    normalized = []
    for row in rows:
        if len(row) < 4:
            raise ValueError("flat_libcell_info rows require four columns")
        normalized.append(
            [
                float(row[1]),
                float(row[2]),
                float(row[3]),
            ]
        )
    return np.ascontiguousarray(normalized, dtype=np.float64)


def _as_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_int_tuple(value):
    if value is None:
        return ()
    return tuple(int(item) for item in list(value))


def _append_length_error(reasons, field, actual, expected):
    if actual != expected:
        reasons.append(f"{field}_length_{actual}_expected_{expected}")


def _csv_start_mismatches(csr_start, element_count):
    return csr_start.size < 2 or int(csr_start[-1]) != int(element_count)


def _validate_csr_start(reasons, field, start, element_count):
    if start.size < 2:
        reasons.append(f"{field}_too_short")
        return
    if start[0] != 0:
        reasons.append(f"{field}_must_begin_at_zero")
    if start[-1] != element_count:
        reasons.append(f"{field}_does_not_cover_elements")
    if np.any(start < 0) or np.any(np.diff(start) < 0):
        reasons.append(f"{field}_must_be_monotonic_nonnegative")


def _validate_index_range(reasons, field, values, upper_bound, *, allow_negative_one=False):
    if values.size == 0:
        return
    lower = -1 if allow_negative_one else 0
    if np.any(values < lower) or np.any(values >= upper_bound):
        reasons.append(f"{field}_out_of_range")


def _empty_design(reasons):
    return {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": "unsupported",
        "source": "pydb",
        "unsupported_reasons": list(reasons),
        "dbu": 1,
        "num_nets": 0,
        "num_pins": 0,
        "num_nodes": 0,
        "num_libcells": 0,
        "num_lib_pins": 0,
        "flat_net2pin_map": np.empty(0, dtype=np.int32),
        "flat_net2pin_start_map": np.empty(0, dtype=np.int32),
        "net2driver_pin_map": np.empty(0, dtype=np.int32),
        "pin2node_map": np.empty(0, dtype=np.int32),
        "pin2net_map": np.empty(0, dtype=np.int32),
        "pin_offset_x": np.empty(0, dtype=np.float64),
        "pin_offset_y": np.empty(0, dtype=np.float64),
        "node_x": np.empty(0, dtype=np.float64),
        "node_y": np.empty(0, dtype=np.float64),
        "node_size_x": np.empty(0, dtype=np.float64),
        "node_size_y": np.empty(0, dtype=np.float64),
        "node_is_buffer": None,
        "node_is_fixed": None,
        "inst_main_id": np.empty(0, dtype=np.int32),
        "inst_libcell_offset": np.empty(0, dtype=np.int32),
        "main_id_2_cell_id_start": np.empty(0, dtype=np.int32),
        "cell_id_2_libpin_id_start": np.empty(0, dtype=np.int32),
        "pin_2_libpin_offset": np.empty(0, dtype=np.int32),
        "pin_direct": None,
        "flat_lib_pin_cap": np.empty(0, dtype=np.float64),
        "flat_lib_pin_rcap": np.empty(0, dtype=np.float64),
        "flat_lib_pin_cap_limit": np.empty(0, dtype=np.float64),
        "flat_lib_pin_slew_limit": np.empty(0, dtype=np.float64),
        "flat_lib_pin_drive_resistance": np.empty(0, dtype=np.float64),
        "flat_libcell_info": np.empty((0, 3), dtype=np.float64),
        "names": {"net": (), "pin": (), "node": (), "libcell": ()},
        "clock_pins": np.empty(0, dtype=np.int32),
        "clock_net_names": (),
        "rows": np.empty((0, 4), dtype=np.float64),
        "site_width": 0,
        "row_height": 0,
        "num_movable_nodes": 0,
        "buffer_main_type_index": -1,
        "buffer_main_type_candidate_indices": (),
        "summary": {
            "status": "unsupported",
            "source": "pydb",
            "dbu": 1,
            "num_nets": 0,
            "num_pins": 0,
            "num_nodes": 0,
            "num_libcells": 0,
            "num_lib_pins": 0,
            "has_buffer_metadata": False,
            "has_fixed_metadata": False,
            "has_pin_limits": False,
            "has_drive_resistance": False,
            "has_pin_direction": False,
            "has_libcell_info": False,
            "pin_cap_source": "none",
            "requires_legalization": True,
            "unsupported_reasons": list(reasons),
        },
    }


def build_rebuffer_design_from_pydb(
    pydb,
    *,
    require_buffer_metadata=False,
    require_names=False,
):
    """Return a compact, native-ready view of ``pydb``.

    The returned mapping is deliberately flat for direct pybind conversion.
    ``status == "unsupported"`` means the caller must not invoke the native
    planner.  Optional metadata is preserved as ``None`` rather than silently
    replaced with a guessed default.
    """

    if pydb is None:
        return _empty_design(["pydb_is_none"])

    reasons = []
    dbu = _as_int(getattr(pydb, "dbu", 0), 0)
    if dbu <= 0:
        reasons.append("invalid_dbu")

    try:
        flat_net2pin = _as_numpy(
            getattr(pydb, "flat_net2pin_map", None),
            np.int32,
            field="flat_net2pin_map",
            allow_empty=True,
        )
        flat_net2pin_start = _as_numpy(
            getattr(pydb, "flat_net2pin_start_map", None),
            np.int32,
            field="flat_net2pin_start_map",
            allow_empty=True,
        )
        net2driver = _as_numpy(
            getattr(pydb, "net2driver_pin_map", None),
            np.int32,
            field="net2driver_pin_map",
            allow_empty=True,
        )
        pin2node = _as_numpy(
            getattr(pydb, "pin2node_map", None),
            np.int32,
            field="pin2node_map",
        )
        pin2net = _as_numpy(
            getattr(pydb, "pin2net_map", None),
            np.int32,
            field="pin2net_map",
        )
        pin_offset_x = _as_numpy(
            getattr(pydb, "pin_offset_x", None),
            np.float64,
            field="pin_offset_x",
        )
        pin_offset_y = _as_numpy(
            getattr(pydb, "pin_offset_y", None),
            np.float64,
            field="pin_offset_y",
        )
        node_x = _as_numpy(
            getattr(pydb, "node_x", None),
            np.float64,
            field="node_x",
        )
        node_y = _as_numpy(
            getattr(pydb, "node_y", None),
            np.float64,
            field="node_y",
        )
        node_size_x = _as_numpy(
            getattr(pydb, "node_size_x", None),
            np.float64,
            field="node_size_x",
        )
        node_size_y = _as_numpy(
            getattr(pydb, "node_size_y", None),
            np.float64,
            field="node_size_y",
        )
        inst_main_id = _as_numpy(
            getattr(pydb, "inst_main_id", None),
            np.int32,
            field="inst_main_id",
        )
        inst_libcell_offset = _as_numpy(
            getattr(pydb, "inst_libcell_offset", None),
            np.int32,
            field="inst_libcell_offset",
        )
        main_id_2_cell_id_start = _as_numpy(
            getattr(pydb, "main_id_2_cell_id_start", None),
            np.int32,
            field="main_id_2_cell_id_start",
        )
        cell_id_2_libpin_id_start = _as_numpy(
            getattr(pydb, "cell_id_2_libpin_id_start", None),
            np.int32,
            field="cell_id_2_libpin_id_start",
        )
        pin_2_libpin_offset = _as_numpy(
            getattr(pydb, "pin_2_libpin_offset", None),
            np.int32,
            field="pin_2_libpin_offset",
        )
        flat_lib_pin_cap = _as_numpy(
            getattr(pydb, "flat_lib_pin_cap", None),
            np.float64,
            field="flat_lib_pin_cap",
            allow_empty=True,
        )
        flat_lib_pin_rcap = _as_numpy(
            getattr(pydb, "flat_lib_pin_rcap", None),
            np.float64,
            field="flat_lib_pin_rcap",
            allow_empty=True,
        )
        flat_lib_pin_cap_limit = _as_numpy(
            getattr(pydb, "flat_lib_pin_cap_limit", None),
            np.float64,
            field="flat_lib_pin_cap_limit",
            allow_empty=True,
        )
        flat_lib_pin_slew_limit = _as_numpy(
            getattr(pydb, "flat_lib_pin_slew_limit", None),
            np.float64,
            field="flat_lib_pin_slew_limit",
            allow_empty=True,
        )
        flat_lib_pin_drive_resistance = _as_numpy(
            getattr(pydb, "flat_lib_pin_drive_resistance", None),
            np.float64,
            field="flat_lib_pin_drive_resistance",
            allow_empty=True,
        )
        flat_libcell_info = _as_libcell_info(
            getattr(pydb, "flat_libcell_info", None),
        )
    except (TypeError, ValueError) as exc:
        return _empty_design([str(exc)])

    net_names = _as_names(getattr(pydb, "net_names", None))
    pin_names = _as_names(getattr(pydb, "pin_names", None))
    node_names = _as_names(getattr(pydb, "node_names", None))
    libcell_names = _as_names(getattr(pydb, "flat_libcell_names", None))

    num_nets = max(0, int(flat_net2pin_start.size) - 1)
    num_pins = int(pin2node.size)
    num_nodes = int(node_x.size)
    num_terminals = _as_int(getattr(pydb, "num_terminals", 0), 0)
    num_terminal_nis = _as_int(getattr(pydb, "num_terminal_NIs", 0), 0)
    num_libcells = len(libcell_names)

    if flat_net2pin_start.size < 2:
        reasons.append("missing_flat_net2pin_start_map")
    else:
        _validate_csr_start(
            reasons,
            "flat_net2pin_start_map",
            flat_net2pin_start,
            flat_net2pin.size,
        )
    _append_length_error(
        reasons,
        "net2driver_pin_map",
        int(net2driver.size),
        num_nets,
    )
    _append_length_error(reasons, "pin2node_map", int(pin2node.size), num_pins)
    _append_length_error(reasons, "pin2net_map", int(pin2net.size), num_pins)
    _append_length_error(reasons, "pin_offset_x", int(pin_offset_x.size), num_pins)
    _append_length_error(reasons, "pin_offset_y", int(pin_offset_y.size), num_pins)
    _append_length_error(reasons, "pin_2_libpin_offset", int(pin_2_libpin_offset.size), num_pins)
    for field, array in (
        ("node_x", node_x),
        ("node_y", node_y),
        ("node_size_x", node_size_x),
        ("node_size_y", node_size_y),
        ("inst_main_id", inst_main_id),
        ("inst_libcell_offset", inst_libcell_offset),
    ):
        _append_length_error(reasons, field, int(array.size), num_nodes)

    if num_libcells == 0:
        reasons.append("missing_flat_libcell_names")
    if not (np.all(np.isfinite(node_x)) and np.all(np.isfinite(node_y))):
        reasons.append("nonfinite_node_coordinates")
    _validate_index_range(reasons, "pin2node_map", pin2node, max(num_nodes, 1))
    _validate_index_range(reasons, "pin2net_map", pin2net, max(num_nets, 1))
    _validate_index_range(reasons, "flat_net2pin_map", flat_net2pin, max(num_pins, 1))
    _validate_index_range(
        reasons,
        "net2driver_pin_map",
        net2driver,
        max(num_pins, 1),
        allow_negative_one=True,
    )

    active_pin_cap = flat_lib_pin_rcap if flat_lib_pin_rcap.size else flat_lib_pin_cap
    pin_cap_source = (
        "rcap" if flat_lib_pin_rcap.size else ("cap" if flat_lib_pin_cap.size else "none")
    )
    num_lib_pins = int(active_pin_cap.size)
    if num_lib_pins == 0:
        reasons.append("missing_lib_pin_capacitance")
    for field, array in (
        ("flat_lib_pin_cap", flat_lib_pin_cap),
        ("flat_lib_pin_rcap", flat_lib_pin_rcap),
        ("flat_lib_pin_cap_limit", flat_lib_pin_cap_limit),
        ("flat_lib_pin_slew_limit", flat_lib_pin_slew_limit),
        ("flat_lib_pin_drive_resistance", flat_lib_pin_drive_resistance),
    ):
        if array.size and array.size != num_lib_pins:
            _append_length_error(reasons, field, int(array.size), num_lib_pins)

    if _csv_start_mismatches(main_id_2_cell_id_start, num_libcells):
        reasons.append("main_id_2_cell_id_start_does_not_cover_libcells")
    else:
        _validate_csr_start(
            reasons,
            "main_id_2_cell_id_start",
            main_id_2_cell_id_start,
            num_libcells,
        )
    if _csv_start_mismatches(cell_id_2_libpin_id_start, num_lib_pins):
        reasons.append("cell_id_2_libpin_id_start_does_not_cover_lib_pins")
    else:
        _validate_csr_start(
            reasons,
            "cell_id_2_libpin_id_start",
            cell_id_2_libpin_id_start,
            num_lib_pins,
        )

    if not flat_libcell_info.size and num_libcells:
        reasons.append("missing_flat_libcell_info")
    elif flat_libcell_info.size and flat_libcell_info.shape[0] != num_libcells:
        reasons.append("flat_libcell_info_shape_mismatch")

    node_is_buffer, buffer_metadata_source = _buffer_metadata(pydb, num_nodes)
    has_buffer_metadata = node_is_buffer is not None
    if node_is_buffer is not None:
        _append_length_error(reasons, "node_is_buffer", int(node_is_buffer.size), num_nodes)

    node_is_fixed = _as_optional_numpy(
        getattr(pydb, "node_is_fixed", None),
        bool,
        field="node_is_fixed",
    )
    has_fixed_metadata = node_is_fixed is not None
    if node_is_fixed is not None:
        _append_length_error(reasons, "node_is_fixed", int(node_is_fixed.size), num_nodes)

    pin_direct = _as_optional_pin_orientation(
        getattr(pydb, "pin_direct", None),
        field="pin_direct",
    )
    has_pin_direction = pin_direct is not None
    if pin_direct is not None:
        _append_length_error(reasons, "pin_direct", int(pin_direct.size), num_pins)

    clock_pins = _as_numpy(
        getattr(pydb, "clock_pins", None),
        np.int32,
        field="clock_pins",
        allow_empty=True,
    )
    _validate_index_range(reasons, "clock_pins", clock_pins, max(num_pins, 1))

    if net_names and len(net_names) != num_nets:
        reasons.append("net_names_length_mismatch")
    if pin_names and len(pin_names) != num_pins:
        reasons.append("pin_names_length_mismatch")
    if node_names and len(node_names) != num_nodes:
        reasons.append("node_names_length_mismatch")
    if require_names:
        for field, values, expected in (
            ("net_names", net_names, num_nets),
            ("pin_names", pin_names, num_pins),
            ("node_names", node_names, num_nodes),
        ):
            if len(values) != expected:
                reasons.append(f"missing_{field}")

    if require_buffer_metadata and not has_buffer_metadata:
        reasons.append("missing_node_is_buffer")

    buffer_main_type_index = _as_int(
        getattr(pydb, "buffer_main_type_index", -1),
        -1,
    )
    if buffer_main_type_index >= num_libcells:
        reasons.append("buffer_main_type_index_out_of_range")
    candidate_indices = _as_int_tuple(getattr(pydb, "buffer_main_type_candidate_indices", ()))
    if any(index < 0 or index >= num_libcells for index in candidate_indices):
        reasons.append("buffer_main_type_candidate_index_out_of_range")

    rows = _as_numpy_2d(
        getattr(pydb, "rows", None),
        np.float64,
        field="rows",
        columns=4,
    )

    status = "unsupported" if reasons else "ok"
    if reasons:
        design = _empty_design(reasons)
        design.update(
            {
                "dbu": max(1, dbu),
                "num_nets": num_nets,
                "num_pins": num_pins,
                "num_nodes": num_nodes,
                "num_base_nodes": num_nodes,
                "num_terminals": num_terminals,
                "num_terminal_NIs": num_terminal_nis,
                "num_libcells": num_libcells,
                "num_lib_pins": num_lib_pins,
            }
        )
        design["summary"].update(
            {
                "dbu": max(1, dbu),
                "num_nets": num_nets,
                "num_pins": num_pins,
                "num_nodes": num_nodes,
                "num_libcells": num_libcells,
                "num_lib_pins": num_lib_pins,
                "has_buffer_metadata": has_buffer_metadata,
                "buffer_metadata_source": buffer_metadata_source,
                "has_fixed_metadata": has_fixed_metadata,
                "has_pin_limits": bool(
                    flat_lib_pin_cap_limit.size and flat_lib_pin_slew_limit.size
                ),
                "has_drive_resistance": bool(flat_lib_pin_drive_resistance.size),
                "has_pin_direction": has_pin_direction,
                "has_libcell_info": bool(flat_libcell_info.size),
                "pin_cap_source": pin_cap_source,
                "unsupported_reasons": list(reasons),
            }
        )
        return design

    summary = {
        "status": status,
        "source": "pydb",
        "dbu": dbu,
        "num_nets": num_nets,
        "num_pins": num_pins,
        "num_nodes": num_nodes,
        "num_base_nodes": num_nodes,
        "num_terminals": num_terminals,
        "num_terminal_NIs": num_terminal_nis,
        "num_libcells": num_libcells,
        "num_lib_pins": num_lib_pins,
        "has_buffer_metadata": has_buffer_metadata,
        "buffer_metadata_source": buffer_metadata_source,
        "has_fixed_metadata": has_fixed_metadata,
        "has_pin_limits": bool(flat_lib_pin_cap_limit.size and flat_lib_pin_slew_limit.size),
        "has_drive_resistance": bool(flat_lib_pin_drive_resistance.size),
        "has_pin_direction": has_pin_direction,
        "has_libcell_info": bool(flat_libcell_info.size),
        "pin_cap_source": pin_cap_source,
        "requires_legalization": True,
        "unsupported_reasons": [],
    }
    return {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": status,
        "source": "pydb",
        "unsupported_reasons": [],
        "dbu": dbu,
        "num_nets": num_nets,
        "num_pins": num_pins,
        "num_nodes": num_nodes,
        "num_libcells": num_libcells,
        "num_lib_pins": num_lib_pins,
        "flat_net2pin_map": flat_net2pin,
        "flat_net2pin_start_map": flat_net2pin_start,
        "net2driver_pin_map": net2driver,
        "pin2node_map": pin2node,
        "pin2net_map": pin2net,
        "pin_offset_x": pin_offset_x,
        "pin_offset_y": pin_offset_y,
        "node_x": node_x,
        "node_y": node_y,
        "node_size_x": node_size_x,
        "node_size_y": node_size_y,
        "node_is_buffer": node_is_buffer,
        "node_is_fixed": node_is_fixed,
        "inst_main_id": inst_main_id,
        "inst_libcell_offset": inst_libcell_offset,
        "main_id_2_cell_id_start": main_id_2_cell_id_start,
        "cell_id_2_libpin_id_start": cell_id_2_libpin_id_start,
        "pin_2_libpin_offset": pin_2_libpin_offset,
        "pin_direct": pin_direct,
        "flat_lib_pin_cap": flat_lib_pin_cap,
        "flat_lib_pin_rcap": flat_lib_pin_rcap,
        "flat_lib_pin_cap_limit": flat_lib_pin_cap_limit,
        "flat_lib_pin_slew_limit": flat_lib_pin_slew_limit,
        "flat_lib_pin_drive_resistance": flat_lib_pin_drive_resistance,
        "flat_libcell_info": flat_libcell_info,
        "names": {
            "net": net_names,
            "pin": pin_names,
            "node": node_names,
            "libcell": libcell_names,
        },
        "clock_pins": clock_pins,
        "clock_net_names": _as_names(getattr(pydb, "clock_net_names", None)),
        "rows": rows,
        "site_width": _as_int(getattr(pydb, "site_width", 0), 0),
        "row_height": _as_int(getattr(pydb, "row_height", 0), 0),
        "num_movable_nodes": _as_int(getattr(pydb, "num_movable_nodes", 0), 0),
        "buffer_main_type_index": buffer_main_type_index,
        "buffer_main_type_candidate_indices": candidate_indices,
        "summary": summary,
    }
