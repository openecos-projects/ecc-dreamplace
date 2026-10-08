import json
from pathlib import Path

from dreamplace.ops.buffer_insertion.buffer_library import (
    lookup_lut_value_with_status_mode,
)


def _safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cell_id_for_master_name(metadata, master_name):
    target = str(master_name)
    names = [_decode_name(value) for value in _as_list(getattr(metadata, "flat_libcell_names", []))]
    for index, name in enumerate(names):
        if name == target:
            return int(index)
    raise ValueError(f"master_name={target!r} is not present in flat_libcell_names")


def _arc_ids_for_cell(metadata, cell_id):
    starts = [int(value) for value in _as_list(getattr(metadata, "cell_id_2_arc_id_start", []))]
    if int(cell_id) < 0 or int(cell_id) + 1 >= len(starts):
        raise ValueError(f"missing libarc range for cell_id={int(cell_id)}")
    return list(range(int(starts[int(cell_id)]), int(starts[int(cell_id) + 1])))


def _arc_lut_value(metadata, *, prefix, table_kind, arc_id, input_slew, output_load):
    values = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_values", []))
    trans = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_trans_table", []))
    cap = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_cap_table", []))
    dims = _as_list(getattr(metadata, f"{prefix}_{table_kind}_flat_luts_dim", []))
    arc_id = int(arc_id)
    if arc_id < 0 or arc_id >= len(values):
        raise ValueError(f"missing {prefix} {table_kind} LUT for arc_id={arc_id}")
    dim = dims[arc_id] if arc_id < len(dims) else []
    value, _status = lookup_lut_value_with_status_mode(
        values=values[arc_id],
        trans_axis=trans[arc_id] if arc_id < len(trans) else [],
        cap_axis=cap[arc_id] if arc_id < len(cap) else [],
        dim=dim,
        input_slew=float(input_slew),
        output_cap=float(output_load),
        boundary_mode="extrapolate",
    )
    return value


def summarize_metadata_liberty_arc_parity(
    *,
    metadata,
    master_name,
    input_slew,
    output_load,
    opensta_arc_samples,
):
    """Compare exported metadata LUT values against OpenSTA Liberty arc samples.

    This is a diagnostic helper. It deliberately bypasses segment transfer and
    checks the device LUT export for one committed buffer master at one input
    slew/output load point.
    """

    cell_id = _cell_id_for_master_name(metadata, master_name)
    arc_rows = []
    for arc_id in _arc_ids_for_cell(metadata, cell_id):
        for prefix in ("f", "r"):
            try:
                delay = _arc_lut_value(
                    metadata,
                    prefix=prefix,
                    table_kind="delay",
                    arc_id=arc_id,
                    input_slew=input_slew,
                    output_load=output_load,
                )
                output_slew = _arc_lut_value(
                    metadata,
                    prefix=prefix,
                    table_kind="trans",
                    arc_id=arc_id,
                    input_slew=input_slew,
                    output_load=output_load,
                )
                arc_rows.append(
                    {
                        "arc_id": int(arc_id),
                        "prefix": str(prefix),
                        "metadata_delay": float(delay),
                        "metadata_output_slew": float(output_slew),
                    }
                )
            except Exception as exc:
                arc_rows.append(
                    {
                        "arc_id": int(arc_id),
                        "prefix": str(prefix),
                        "error": str(exc),
                    }
                )

    samples = [dict(sample) for sample in opensta_arc_samples]
    delay_targets = [
        float(sample["liberty_gate_delay"])
        for sample in samples
        if sample.get("liberty_gate_delay") is not None
    ]
    slew_targets = [
        float(sample["liberty_output_slew"])
        for sample in samples
        if sample.get("liberty_output_slew") is not None
    ]
    finite_arc_rows = [
        row
        for row in arc_rows
        if row.get("metadata_delay") is not None
        and row.get("metadata_output_slew") is not None
    ]

    def closest(target, key):
        if target is None or not finite_arc_rows:
            return None
        row = min(
            finite_arc_rows,
            key=lambda item: abs(float(item[key]) - float(target)),
        )
        return {
            "arc_id": int(row["arc_id"]),
            "prefix": str(row["prefix"]),
            "metadata_value": float(row[key]),
            "opensta_value": float(target),
            "abs_error": abs(float(row[key]) - float(target)),
            "opensta_over_metadata": _safe_ratio(target, row[key]),
        }

    return {
        "status": "metadata_liberty_arc_parity",
        "master_name": str(master_name),
        "cell_id": int(cell_id),
        "input_slew": float(input_slew),
        "output_load": float(output_load),
        "arc_rows": arc_rows,
        "opensta_arc_samples": samples,
        "delay_targets": delay_targets,
        "output_slew_targets": slew_targets,
        "closest_delay_to_max_opensta": closest(max(delay_targets) if delay_targets else None, "metadata_delay"),
        "closest_slew_to_max_opensta": closest(max(slew_targets) if slew_targets else None, "metadata_output_slew"),
    }


def load_static_probe_row(path, *, segment_id):
    with Path(path).open("r", encoding="utf8") as stream:
        artifact = json.load(stream)
    rows = list(artifact.get("rows", []))
    for row in rows:
        if int(row.get("global_segment_id", row.get("segment_id", -1))) == int(segment_id):
            return row
    raise ValueError(f"segment_id={int(segment_id)} is not present in static probe")


def load_forced_z_delta(path, *, segment_id):
    with Path(path).open("r", encoding="utf8") as stream:
        artifact = json.load(stream)
    probe = dict(artifact.get("probe", {}))
    for row in probe.get("forced_results", []):
        for segment in row.get("selected_segments", []):
            if int(segment.get("segment_id", -1)) == int(segment_id):
                return float(row["delta_tns_vs_z0"])
    raise ValueError(f"segment_id={int(segment_id)} is not present in forced-z probe")


def summarize_top1_static_probe(
    *,
    static_probe_row,
    pysta_delta_tns,
    opensta_committed_arc,
    opensta_delta_tns,
):
    delay_values = [
        float(value)
        for value in opensta_committed_arc.get("gate_delay_ps", [])
    ]
    slew_values = [
        float(value)
        for value in opensta_committed_arc.get("output_slew_ps", [])
    ]
    static_delay = float(static_probe_row["static_buffer_delay"])
    static_slew = float(static_probe_row["static_buffer_output_slew"])
    delay_ratios = [
        ratio for ratio in (_safe_ratio(value, static_delay) for value in delay_values)
        if ratio is not None
    ]
    slew_ratios = [
        ratio for ratio in (_safe_ratio(value, static_slew) for value in slew_values)
        if ratio is not None
    ]
    return {
        "segment_id": int(static_probe_row["global_segment_id"]),
        "net_id": int(static_probe_row["net_id"]),
        "net_name": str(static_probe_row.get("net_name", "")),
        "z": float(static_probe_row["z_value"]),
        "bsu_index": float(static_probe_row["bsu_index"]),
        "pysta_static": {
            "driver_slew_ps": float(static_probe_row["driver_slew"]),
            "buffer_input_cap": float(static_probe_row["static_buffer_input_cap"]),
            "buffer_delay_ps": static_delay,
            "buffer_output_slew_ps": static_slew,
            "segment_delay_ps": float(static_probe_row["segment_delay"]),
            "segment_output_slew_ps": float(static_probe_row["segment_output_slew"]),
        },
        "opensta_committed_arc": {
            "buffer_master": opensta_committed_arc.get("buffer_master"),
            "input_slew_ps": list(opensta_committed_arc.get("input_slew_ps", [])),
            "output_load_cap": opensta_committed_arc.get("output_load_cap"),
            "gate_delay_ps": delay_values,
            "output_slew_ps": slew_values,
            "pin_arrival_delta_ps": list(opensta_committed_arc.get("pin_arrival_delta_ps", [])),
        },
        "ratios": {
            "opensta_gate_delay_over_pysta_static_min": (
                min(delay_ratios)
                if delay_ratios
                else None
            ),
            "opensta_gate_delay_over_pysta_static_max": (
                max(delay_ratios)
                if delay_ratios
                else None
            ),
            "opensta_output_slew_over_pysta_static_min": (
                min(slew_ratios)
                if slew_ratios
                else None
            ),
            "opensta_output_slew_over_pysta_static_max": (
                max(slew_ratios)
                if slew_ratios
                else None
            ),
            "pysta_delta_tns_over_opensta_delta_tns": _safe_ratio(
                pysta_delta_tns,
                opensta_delta_tns,
            ),
        },
        "delta_tns": {
            "pysta_ps": float(pysta_delta_tns),
            "opensta_ps": float(opensta_delta_tns),
        },
        "status": "static_probe_only",
        "next_gate": "wire real Liberty LUT tensors into analytic segment transfer and regenerate this report",
    }


def summarize_parent_visible_load_gap(
    *,
    probe_row,
    opensta_upstream_net_cap,
    opensta_driver_slew_ps=None,
    opensta_buffer_input_slew_ps=None,
):
    """Summarize the z=1 parent-visible-load mismatch for one probe row.

    The segment transfer primitive can be locally reasonable while the
    surrounding net-subgraph still gives the parent driver an unrealistically
    small load. This helper records the retained upstream capacitance that the
    relaxed model would need in order to match the committed OpenSTA upstream
    net capacitance for the same action.
    """

    opensta_upstream_net_cap = float(opensta_upstream_net_cap)
    static_parent_load = float(
        probe_row.get(
            "segment_upstream_visible_input_cap",
            probe_row.get("static_buffer_input_cap", 0.0),
        )
    )
    analytic_parent_load = float(
        probe_row.get(
            "analytic_transfer_upstream_visible_input_cap",
            static_parent_load,
        )
    )
    buffer_input_cap = float(
        probe_row.get(
            "static_buffer_input_cap",
            probe_row.get("analytic_transfer_upstream_visible_input_cap", 0.0),
        )
    )
    local_edge_cap = float((probe_row.get("edge_rc", {}) or {}).get("c", 0.0) or 0.0)
    upstream_fraction = 0.0
    fractions = (probe_row.get("fractions_by_repeater_count", {}) or {}).get("1")
    if fractions:
        upstream_fraction = float(fractions[0])
    local_upstream_wire_cap = local_edge_cap * upstream_fraction
    required_retained_cap_over_analytic = opensta_upstream_net_cap - analytic_parent_load
    required_retained_cap_over_buffer_input = opensta_upstream_net_cap - buffer_input_cap

    summary = {
        "segment_id": int(probe_row["global_segment_id"]),
        "net_id": int(probe_row["net_id"]),
        "net_name": str(probe_row.get("net_name", "")),
        "z": float(probe_row.get("z_value", 0.0)),
        "bsu_index": float(probe_row.get("bsu_index", 0.0)),
        "loads_pf": {
            "opensta_upstream_net_cap": opensta_upstream_net_cap,
            "static_parent_visible_load": static_parent_load,
            "analytic_parent_visible_load": analytic_parent_load,
            "buffer_input_cap": buffer_input_cap,
            "local_edge_cap": local_edge_cap,
            "local_upstream_wire_cap": local_upstream_wire_cap,
            "required_retained_cap_over_analytic": required_retained_cap_over_analytic,
            "required_retained_cap_over_buffer_input": required_retained_cap_over_buffer_input,
        },
        "ratios": {
            "opensta_over_analytic_parent_visible_load": _safe_ratio(
                opensta_upstream_net_cap,
                analytic_parent_load,
            ),
            "required_retained_over_local_upstream_wire_cap": _safe_ratio(
                required_retained_cap_over_analytic,
                local_upstream_wire_cap,
            ),
        },
        "status": "parent_visible_load_gap",
        "next_gate": (
            "derive an OpenSTA-equivalent retained upstream-cap model before "
            "wiring analytic segment_transfer into the segment-count hot path"
        ),
    }
    if opensta_driver_slew_ps is not None:
        summary["opensta_driver_slew_ps"] = list(opensta_driver_slew_ps)
    if opensta_buffer_input_slew_ps is not None:
        summary["opensta_buffer_input_slew_ps"] = list(opensta_buffer_input_slew_ps)
    return summary


def summarize_forced_frame_probe_sanity(
    *,
    probe_rows,
    segment_id,
    expected_net_name,
    expected_z_value,
    opensta_buffer_input_slew_ps,
    opensta_output_load_pf,
    opensta_gate_delay_ps,
    opensta_output_slew_ps,
):
    rows = [
        row
        for row in probe_rows
        if int(row.get("global_segment_id", row.get("segment_id", -1))) == int(segment_id)
    ]
    if not rows:
        raise ValueError(f"segment_id={int(segment_id)} is not present in forced-frame probe")

    by_sense = {str(row.get("sense", f"row{index}")): row for index, row in enumerate(rows)}
    observed_net_names = sorted({str(row.get("net_name", "")) for row in rows})
    observed_z_values = sorted({float(row.get("z_value", 0.0)) for row in rows})
    pysta_input_slews = [
        float(row.get("analytic_transfer_buffer_input_slew", row.get("segment_parent_slew", 0.0)))
        for row in rows
    ]
    pysta_output_loads = [
        float(row.get("analytic_transfer_buffer_output_load", row.get("segment_downstream_load", 0.0)))
        for row in rows
    ]
    pysta_delays = [
        float(row.get("analytic_transfer_buffer_delay", row.get("static_buffer_delay", 0.0)))
        for row in rows
    ]
    pysta_output_slews = [
        float(row.get("analytic_transfer_buffer_output_slew", row.get("static_buffer_output_slew", 0.0)))
        for row in rows
    ]
    opensta_buffer_input_slew_ps = [float(value) for value in opensta_buffer_input_slew_ps]
    opensta_gate_delay_ps = [float(value) for value in opensta_gate_delay_ps]
    opensta_output_slew_ps = [float(value) for value in opensta_output_slew_ps]
    opensta_output_load_pf = float(opensta_output_load_pf)

    max_pysta_input_slew = max(pysta_input_slews)
    min_opensta_input_slew = min(opensta_buffer_input_slew_ps)
    max_pysta_output_load = max(pysta_output_loads)

    return {
        "segment_id": int(segment_id),
        "observed_net_names": observed_net_names,
        "observed_z_values": observed_z_values,
        "expected_net_name": str(expected_net_name),
        "expected_z_value": float(expected_z_value),
        "matches_expected_frame": (
            observed_net_names == [str(expected_net_name)]
            and all(abs(value - float(expected_z_value)) <= 1e-9 for value in observed_z_values)
        ),
        "senses": sorted(by_sense.keys()),
        "pysta_forced_frame": {
            "input_slew_ps": pysta_input_slews,
            "output_load_pf": pysta_output_loads,
            "analytic_buffer_delay_ps": pysta_delays,
            "analytic_buffer_output_slew_ps": pysta_output_slews,
        },
        "opensta_committed_arc": {
            "buffer_input_slew_ps": opensta_buffer_input_slew_ps,
            "output_load_pf": opensta_output_load_pf,
            "gate_delay_ps": opensta_gate_delay_ps,
            "output_slew_ps": opensta_output_slew_ps,
        },
        "ratios": {
            "min_opensta_input_slew_over_max_pysta_input_slew": _safe_ratio(
                min_opensta_input_slew,
                max_pysta_input_slew,
            ),
            "opensta_output_load_over_max_pysta_output_load": _safe_ratio(
                opensta_output_load_pf,
                max_pysta_output_load,
            ),
            "min_opensta_gate_delay_over_max_pysta_delay": _safe_ratio(
                min(opensta_gate_delay_ps),
                max(pysta_delays),
            ),
            "min_opensta_output_slew_over_max_pysta_output_slew": _safe_ratio(
                min(opensta_output_slew_ps),
                max(pysta_output_slews),
            ),
        },
        "status": "forced_frame_sanity_only",
        "interpretation": (
            "This checks whether the diagnostic frame is the intended forced-z "
            "state. It is not a primitive-only replay because the PySTA input "
            "slew may differ from the committed OpenSTA buffer input slew."
        ),
    }
