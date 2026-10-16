import csv
import json
import math
from pathlib import Path

import torch

CRITICAL_PATH_FIELDNAMES = (
    "path_index",
    "path_valid",
    "invalid_reason",
    "endpoint_pin_id",
    "endpoint_pin_name",
    "endpoint_transition",
    "endpoint_slack_ps",
    "path_max_residual_ps",
    "step_index",
    "pin_id",
    "pin_name",
    "transition",
    "cumulative_arrival_ps",
    "pin_slew_ps",
    "pin_net_cap_pf",
    "incoming_arc_kind",
    "incoming_arc_id",
    "incoming_source_pin_id",
    "incoming_source_pin_name",
    "incoming_source_transition",
    "incoming_source_arrival_ps",
    "incoming_source_slew_ps",
    "incoming_output_load_pf",
    "incoming_delay_ps",
    "incoming_reconstructed_arrival_ps",
    "incoming_residual_ps",
    "lib_cell_idx",
    "lib_cell_name",
    "lib_arc_idx",
    "lib_arc_from_pin",
    "lib_arc_to_pin",
    "timing_sense",
    "timing_type",
)

_INVALID_REASON_NAMES = {
    0: "none",
    1: "no_predecessor",
    2: "cycle",
    3: "max_depth",
    4: "residual_tolerance",
}


def timing_index_tensor(values, reference):
    """Return integer indices on the tensor device used by timing queries."""
    if torch.is_tensor(values):
        return values.to(device=reference.device, dtype=torch.long)
    return torch.as_tensor(values, device=reference.device, dtype=torch.long)


def _cpu_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _tensor_values(timing_op, live_name, fallback_name):
    value = getattr(timing_op, live_name, None)
    if value is None:
        value = getattr(timing_op, fallback_name, None)
    return _cpu_list(value)


def _name(values, index):
    if not (0 <= index < len(values)):
        return ""
    value = values[index]
    return value.decode("utf-8") if hasattr(value, "decode") else str(value)


def _arc_port_name(flat_libarc_names, lib_arc_idx, column):
    if not (0 <= lib_arc_idx < len(flat_libarc_names)):
        return ""
    try:
        value = flat_libarc_names[lib_arc_idx][column]
    except (IndexError, TypeError):
        return ""
    return value.decode("utf-8") if hasattr(value, "decode") else str(value)


def _transition_name(transition):
    return "rise" if int(transition) == 0 else "fall"


def _path_delay(cell_delays, net_delays, arc_id, source_transition, sink_transition, sink_pin_id):
    if arc_id == -1:
        return net_delays[sink_transition][sink_pin_id]
    transition_key = (source_transition, sink_transition)
    delay_kind = {
        (0, 0): "rr",
        (1, 0): "fr",
        (0, 1): "rf",
        (1, 1): "ff",
    }[transition_key]
    return cell_delays[delay_kind][arc_id]


def build_critical_path_rows(
    *,
    batch,
    timing_op,
    pin_names,
    flat_inst_arcs_by_level,
    flat_libcell_names=(),
    flat_libarc_names=(),
):
    pin_names = list(pin_names)
    flat_libcell_names = list(flat_libcell_names)
    flat_libarc_names = list(flat_libarc_names)
    flat_arcs = _cpu_list(flat_inst_arcs_by_level)
    path_offsets = _cpu_list(batch.path_offsets)
    path_pins = _cpu_list(batch.path_pins)
    path_transitions = _cpu_list(batch.path_transitions)
    path_arc_ids = _cpu_list(batch.path_arc_ids)
    endpoint_pins = _cpu_list(batch.endpoint_pins)
    endpoint_transitions = _cpu_list(batch.endpoint_transitions)
    endpoint_slacks = _cpu_list(batch.endpoint_slacks)
    path_valid = _cpu_list(batch.path_valid)
    invalid_reasons = _cpu_list(batch.invalid_reason)
    max_residuals = _cpu_list(batch.max_residual_ps)

    arrivals = {
        0: _tensor_values(timing_op, "pin_rAAT_live_snapshot", "pin_rAAT"),
        1: _tensor_values(timing_op, "pin_fAAT_live_snapshot", "pin_fAAT"),
    }
    slews = {
        0: _tensor_values(timing_op, "pin_rtran_live", "pin_rtran"),
        1: _tensor_values(timing_op, "pin_ftran_live", "pin_ftran"),
    }
    loads = {
        0: _tensor_values(timing_op, "pin_net_cap_rise_live", "pin_net_cap_rise"),
        1: _tensor_values(timing_op, "pin_net_cap_fall_live", "pin_net_cap_fall"),
    }
    net_delays = {
        0: _tensor_values(timing_op, "pin_net_delay_rise_live", "pin_net_delay_rise"),
        1: _tensor_values(timing_op, "pin_net_delay_fall_live", "pin_net_delay_fall"),
    }
    cell_delays = {
        "rr": _cpu_list(getattr(timing_op, "cell_arc_rr_delays", None)),
        "fr": _cpu_list(getattr(timing_op, "cell_arc_fr_delays", None)),
        "rf": _cpu_list(getattr(timing_op, "cell_arc_rf_delays", None)),
        "ff": _cpu_list(getattr(timing_op, "cell_arc_ff_delays", None)),
    }
    required_values = [
        *arrivals.values(),
        *slews.values(),
        *loads.values(),
        *net_delays.values(),
        *cell_delays.values(),
    ]
    if any(not values for values in required_values):
        raise RuntimeError("critical path report requires live timing state")

    rows = []
    arc_cursor = 0
    for path_index in range(len(path_offsets) - 1):
        pin_begin = int(path_offsets[path_index])
        pin_end = int(path_offsets[path_index + 1])
        path_length = pin_end - pin_begin
        endpoint_pin = int(endpoint_pins[path_index])
        endpoint_transition = int(endpoint_transitions[path_index])
        endpoint_slack = float(endpoint_slacks[path_index])
        valid = bool(path_valid[path_index])
        invalid_reason = int(invalid_reasons[path_index])
        max_residual = float(max_residuals[path_index])

        for step_index in range(path_length):
            pin_offset = pin_begin + step_index
            pin_id = int(path_pins[pin_offset])
            transition = int(path_transitions[pin_offset])
            arrival = float(arrivals[transition][pin_id])
            row = {
                "path_index": path_index,
                "path_valid": int(valid),
                "invalid_reason": _INVALID_REASON_NAMES.get(invalid_reason, str(invalid_reason)),
                "endpoint_pin_id": endpoint_pin,
                "endpoint_pin_name": _name(pin_names, endpoint_pin),
                "endpoint_transition": _transition_name(endpoint_transition),
                "endpoint_slack_ps": endpoint_slack,
                "path_max_residual_ps": max_residual,
                "step_index": step_index,
                "pin_id": pin_id,
                "pin_name": _name(pin_names, pin_id),
                "transition": _transition_name(transition),
                "cumulative_arrival_ps": arrival,
                "pin_slew_ps": float(slews[transition][pin_id]),
                "pin_net_cap_pf": float(loads[transition][pin_id]),
                "incoming_arc_kind": "startpoint",
                "incoming_arc_id": "",
                "incoming_source_pin_id": "",
                "incoming_source_pin_name": "",
                "incoming_source_transition": "",
                "incoming_source_arrival_ps": "",
                "incoming_source_slew_ps": "",
                "incoming_output_load_pf": "",
                "incoming_delay_ps": "",
                "incoming_reconstructed_arrival_ps": "",
                "incoming_residual_ps": "",
                "lib_cell_idx": "",
                "lib_cell_name": "",
                "lib_arc_idx": "",
                "lib_arc_from_pin": "",
                "lib_arc_to_pin": "",
                "timing_sense": "",
                "timing_type": "",
            }
            if step_index:
                source_offset = pin_offset - 1
                source_pin_id = int(path_pins[source_offset])
                source_transition = int(path_transitions[source_offset])
                arc_id = int(path_arc_ids[arc_cursor])
                arc_cursor += 1
                source_arrival = float(arrivals[source_transition][source_pin_id])
                source_slew = float(slews[source_transition][source_pin_id])
                delay = float(
                    _path_delay(
                        cell_delays,
                        net_delays,
                        arc_id,
                        source_transition,
                        transition,
                        pin_id,
                    )
                )
                reconstructed_arrival = source_arrival + delay
                row.update(
                    {
                        "incoming_arc_kind": "net" if arc_id == -1 else "cell",
                        "incoming_arc_id": arc_id,
                        "incoming_source_pin_id": source_pin_id,
                        "incoming_source_pin_name": _name(pin_names, source_pin_id),
                        "incoming_source_transition": _transition_name(source_transition),
                        "incoming_source_arrival_ps": source_arrival,
                        "incoming_source_slew_ps": source_slew,
                        "incoming_delay_ps": delay,
                        "incoming_reconstructed_arrival_ps": reconstructed_arrival,
                        "incoming_residual_ps": arrival - reconstructed_arrival,
                    }
                )
                if arc_id >= 0:
                    arc = flat_arcs[arc_id]
                    lib_cell_idx = int(arc[2])
                    lib_arc_idx = int(arc[3])
                    row.update(
                        {
                            "incoming_output_load_pf": float(loads[transition][pin_id]),
                            "lib_cell_idx": lib_cell_idx,
                            "lib_cell_name": _name(flat_libcell_names, lib_cell_idx),
                            "lib_arc_idx": lib_arc_idx,
                            "lib_arc_from_pin": _arc_port_name(flat_libarc_names, lib_arc_idx, 0),
                            "lib_arc_to_pin": _arc_port_name(flat_libarc_names, lib_arc_idx, 1),
                            "timing_sense": int(arc[4]),
                            "timing_type": int(arc[5]),
                        }
                    )
            rows.append(row)

    if arc_cursor != len(path_arc_ids):
        raise RuntimeError(
            f"critical path arc count mismatch: consumed {arc_cursor}, have {len(path_arc_ids)}"
        )
    return rows


def critical_path_summary(batch, rows):
    residuals = [
        abs(float(row["incoming_residual_ps"]))
        for row in rows
        if row["incoming_residual_ps"] != "" and math.isfinite(float(row["incoming_residual_ps"]))
    ]
    return {
        "status": "ok",
        "failing_state_count": int(batch.failing_state_count),
        "selected_state_count": int(batch.selected_state_count),
        "valid_path_count": int(batch.valid_path_count),
        "invalid_path_count": int(batch.invalid_path_count),
        "row_count": len(rows),
        "max_recomputed_edge_residual_ps": max(residuals, default=0.0),
        "topology_epoch": int(batch.topology_epoch),
        "extraction_runtime_ms": float(batch.extraction_runtime_ms),
    }


def write_critical_path_report(csv_path, summary_path, *, batch, rows):
    csv_path = Path(csv_path)
    summary_path = Path(summary_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CRITICAL_PATH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary = critical_path_summary(batch, rows)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary
