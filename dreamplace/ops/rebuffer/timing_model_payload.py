"""Payload marshalling for the rebuffer timing model.

``PyPlaceDB`` exports tensors, lists and LUT tables; the timing model needs
plain Python lists and padded numpy tables.  These helpers keep that
conversion in one place so the model itself stays about timing.
"""

from dataclasses import dataclass

import numpy as np
import torch


def persistent_timing_state_payload(timing_model):
    """Copy the current Python TP state into the native physical-pin schema.

    RC inputs may include a trailing Steiner suffix. The graph owns timing
    state on physical pins; the native RC forest owns the Steiner domain.
    Field names and source attributes come from the native schema.
    """
    from . import rebuffer_cpp

    timing = timing_model.model
    if timing.num_pins != len(timing_model.pydb.pin_names):
        raise ValueError("persistent timing state requires the current matching carrier")
    schema = rebuffer_cpp.persistent_timing_state_schema(
        len(timing_model.pydb.pin_names), len(timing.flat_inst_arcs_by_level)
    )
    values = {}
    for name, spec in schema["fields"].items():
        source = getattr(timing, spec["source"], None)
        if source is None:
            raise ValueError(f"Python TP state is unavailable: {spec['source']}")
        if source.device.type != "cpu" or source.dtype != torch.float64:
            raise ValueError(f"{spec['source']} requires CPU float64 Python TP state")
        array = source.detach().numpy()
        if name.startswith("pin_net_"):
            array = array[: spec["length"]]
        values[name] = np.array(array, dtype=np.float64, order="C", copy=True)
    return values


def timing_pin_slack(timing):
    """Transport the operator's qualified endpoint projection to its pin view."""
    slack = timing.get_pin_slack().detach().clone()
    endpoint_slack = getattr(timing, "last_endpoint_slack_tensor", None)
    if endpoint_slack is not None:
        slack[timing.end_points.cpu().long()] = torch.inf
        slack[timing.last_endpoint_ids_tensor.cpu().long()] = endpoint_slack.detach().cpu()
    return slack


@dataclass(frozen=True)
class TimingSnapshot:
    wns_ps: float
    tns_ps: float
    worst_slack_ps: float
    endpoint_count: int
    violating_endpoint_count: int

    def as_dict(self):
        return {
            "wns_ps": self.wns_ps,
            "tns_ps": self.tns_ps,
            "worst_slack_ps": self.worst_slack_ps,
            "endpoint_count": self.endpoint_count,
            "violating_endpoint_count": self.violating_endpoint_count,
        }


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


def _pad_luts(values, trans, cap, dims):
    dims = np.asarray(list(dims), dtype=np.int64).reshape(-1, 2)
    count = len(dims)
    max_trans = int(dims[:, 0].max(initial=1))
    max_cap = int(dims[:, 1].max(initial=1))
    padded_values = np.zeros((count, max_trans * max_cap), dtype=np.float64)
    trans_axis = np.full((count, max_trans), np.inf, dtype=np.float64)
    cap_axis = np.full((count, max_cap), np.inf, dtype=np.float64)
    for index, (rows, cols) in enumerate(dims):
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0:
            continue
        table = np.asarray(values[index], dtype=np.float64).reshape(-1)
        if table.size != rows * cols:
            continue
        padded_values[index, : rows * cols] = table
        trans_row = np.asarray(trans[index], dtype=np.float64).reshape(-1)
        cap_row = np.asarray(cap[index], dtype=np.float64).reshape(-1)
        if trans_row.size == rows:
            trans_axis[index, :rows] = trans_row
        if cap_row.size == cols:
            cap_axis[index, :cols] = cap_row
    return (
        torch.from_numpy(padded_values),
        torch.from_numpy(trans_axis),
        torch.from_numpy(cap_axis),
        torch.from_numpy(dims),
    )


def _pin_pair_arc_mapping(pydb):
    """Build the pin-pair -> arc-id CSR consumed by the native path extractor.

    ``PyPlaceDB`` exposes predecessor pin/arc arrays but not the optional
    ``pin_pair_arc_keys`` view used by ``extract_critical_paths``.  Without
    this mapping the native extractor cannot distinguish reconvergent arcs and
    falls back to a slack-only predecessor walk, which can return a path that
    is not the endpoint's worst path.
    """

    from . import rebuffer_cpp

    return tuple(
        None if values is None else torch.from_numpy(values)
        for values in rebuffer_cpp.pin_pair_arc_mapping(pydb)
    )
