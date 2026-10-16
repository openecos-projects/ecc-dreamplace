import torch

from .analytic import analytic_segment_transfer
from .schema import SegmentTransferInput


def _float_value(value):
    if torch.is_tensor(value):
        if int(value.numel()) != 1:
            return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        return float(value.detach().cpu().item())
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def _float_diagnostics(diagnostics):
    rows = {}
    for key, value in dict(diagnostics or {}).items():
        if torch.is_tensor(value):
            rows[key] = _float_value(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            rows[key] = value
        elif isinstance(value, (list, tuple)):
            rows[key] = list(value)
        elif isinstance(value, dict):
            rows[key] = dict(value)
        else:
            rows[key] = str(value)
    return rows


def transfer_result_to_row(result):
    return {
        "upstream_visible_load": _float_value(result.upstream_visible_load),
        "segment_delay": _float_value(result.segment_delay),
        "output_slew": _float_value(result.output_slew),
        "output_arrival": _float_value(result.output_arrival),
        "diagnostics": _float_diagnostics(result.diagnostics),
    }


def build_transfer_input_from_probe_row(
    probe_row,
    *,
    repeater_count,
    buffer_device=None,
    upstream_retained_capacitance=0.0,
    dtype=torch.float64,
):
    edge_rc = dict(probe_row.get("edge_rc", {}) or {})
    fractions = (probe_row.get("fractions_by_repeater_count", {}) or {}).get(
        str(int(repeater_count))
    )
    return SegmentTransferInput(
        input_arrival=torch.tensor(
            float(probe_row.get("segment_parent_arrival", probe_row.get("driver_arrival", 0.0))),
            dtype=dtype,
        ),
        input_slew=torch.tensor(
            float(probe_row.get("segment_parent_slew", probe_row.get("driver_slew", 0.0))),
            dtype=dtype,
        ),
        downstream_load=torch.tensor(
            float(probe_row.get("segment_downstream_load", 0.0)),
            dtype=dtype,
        ),
        edge_resistance=torch.tensor(float(edge_rc.get("r", 0.0) or 0.0), dtype=dtype),
        edge_capacitance=torch.tensor(float(edge_rc.get("c", 0.0) or 0.0), dtype=dtype),
        upstream_retained_capacitance=torch.tensor(
            float(upstream_retained_capacitance),
            dtype=dtype,
        ),
        repeater_count=int(repeater_count),
        bsu_index=torch.tensor(float(probe_row.get("bsu_index", 0.0)), dtype=dtype),
        split_fractions=tuple(float(value) for value in fractions) if fractions else None,
        buffer_device=buffer_device if int(repeater_count) else None,
    )


def replay_local_segment_transfer(
    *,
    base_input,
    buffer_device,
    bsu_index=None,
    split_fractions=(0.5, 0.5),
    upstream_retained_capacitance=0.0,
):
    n0_input = SegmentTransferInput(
        input_arrival=base_input.input_arrival,
        input_slew=base_input.input_slew,
        downstream_load=base_input.downstream_load,
        edge_resistance=base_input.edge_resistance,
        edge_capacitance=base_input.edge_capacitance,
        repeater_count=0,
        bsu_index=base_input.bsu_index,
        split_fractions=(1.0,),
    )
    n1_input = SegmentTransferInput(
        input_arrival=base_input.input_arrival,
        input_slew=base_input.input_slew,
        downstream_load=base_input.downstream_load,
        edge_resistance=base_input.edge_resistance,
        edge_capacitance=base_input.edge_capacitance,
        upstream_retained_capacitance=upstream_retained_capacitance,
        repeater_count=1,
        bsu_index=base_input.bsu_index if bsu_index is None else bsu_index,
        split_fractions=split_fractions,
        buffer_device=buffer_device,
    )
    n0 = analytic_segment_transfer(n0_input)
    n1 = analytic_segment_transfer(n1_input)
    return summarize_replay_results(n0, n1)


def replay_probe_row_segment_transfer(
    probe_row,
    *,
    buffer_device,
    upstream_retained_capacitance=0.0,
    dtype=torch.float64,
):
    n0 = analytic_segment_transfer(
        build_transfer_input_from_probe_row(
            probe_row,
            repeater_count=0,
            dtype=dtype,
        )
    )
    n1 = analytic_segment_transfer(
        build_transfer_input_from_probe_row(
            probe_row,
            repeater_count=1,
            buffer_device=buffer_device,
            upstream_retained_capacitance=upstream_retained_capacitance,
            dtype=dtype,
        )
    )
    summary = summarize_replay_results(n0, n1)
    summary["probe"] = {
        "segment_id": int(probe_row.get("global_segment_id", probe_row.get("segment_id", -1))),
        "net_id": int(probe_row.get("net_id", -1)),
        "net_name": str(probe_row.get("net_name", "")),
        "z_value": float(probe_row.get("z_value", 0.0)),
        "bsu_index": float(probe_row.get("bsu_index", 0.0)),
    }
    return summary


def summarize_replay_results(n0_result, n1_result):
    n0 = transfer_result_to_row(n0_result)
    n1 = transfer_result_to_row(n1_result)
    return {
        "n0": n0,
        "n1": n1,
        "delta_n1_minus_n0": {
            "upstream_visible_load": (
                n1["upstream_visible_load"] - n0["upstream_visible_load"]
            ),
            "segment_delay": n1["segment_delay"] - n0["segment_delay"],
            "output_slew": n1["output_slew"] - n0["output_slew"],
            "output_arrival": n1["output_arrival"] - n0["output_arrival"],
        },
        "status": "local_segment_transfer_replay",
    }
