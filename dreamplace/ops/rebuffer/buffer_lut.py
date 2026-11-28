"""Liberty buffer delay/slew LUT helpers for the rebuffer overlay."""

from __future__ import annotations

import numpy as np
import torch


def _axis_index(axis, value):
    size = len(axis)
    if size <= 1 or value <= axis[0]:
        return 0
    if value >= axis[-1]:
        return size - 2
    lower, upper = 0, size - 1
    while upper - lower > 1:
        middle = (upper + lower) // 2
        if value >= axis[middle]:
            lower = middle
        else:
            upper = middle
    return lower


def _interpolate(values, trans_axis, cap_axis, trans, cap):
    if len(trans_axis) == 1 and len(cap_axis) == 1:
        return float(values[0])
    if len(trans_axis) == 1:
        index = _axis_index(cap_axis, cap)
        delta = (cap - cap_axis[index]) / (cap_axis[index + 1] - cap_axis[index])
        return float(values[index] * (1.0 - delta) + values[index + 1] * delta)
    if len(cap_axis) == 1:
        index = _axis_index(trans_axis, trans)
        delta = (trans - trans_axis[index]) / (trans_axis[index + 1] - trans_axis[index])
        return float(values[index] * (1.0 - delta) + values[index + 1] * delta)
    trans_index = _axis_index(trans_axis, trans)
    cap_index = _axis_index(cap_axis, cap)
    ds = (trans - trans_axis[trans_index]) / (
        trans_axis[trans_index + 1] - trans_axis[trans_index]
    )
    dc = (cap - cap_axis[cap_index]) / (cap_axis[cap_index + 1] - cap_axis[cap_index])
    stride = len(cap_axis)

    def at(row, column):
        return values[row * stride + column]

    return float(
        (1.0 - ds) * (1.0 - dc) * at(trans_index, cap_index)
        + ds * (1.0 - dc) * at(trans_index + 1, cap_index)
        + ds * dc * at(trans_index + 1, cap_index + 1)
        + (1.0 - ds) * dc * at(trans_index, cap_index + 1)
    )


class LibertyBufferLut:
    """Slew/load-dependent delay and output-slew table for one master."""

    def __init__(self, *, master_name, input_cap, slew_axis, load_axis, delay, output_slew):
        self.master_name = str(master_name)
        self.input_cap = float(input_cap)
        self.slew_axis = tuple(float(value) for value in slew_axis)
        self.load_axis = tuple(float(value) for value in load_axis)
        self.delay = np.asarray(delay, dtype=np.float64).reshape(
            1, len(self.slew_axis), len(self.load_axis)
        )
        self.output_slew = np.asarray(output_slew, dtype=np.float64).reshape(
            1, len(self.slew_axis), len(self.load_axis)
        )
        self.source = f"liberty:{self.master_name}"

    @classmethod
    def from_liberty_view(cls, view, master_name):
        names = [str(name) for name in view.get("names", ())]
        if master_name not in names:
            raise ValueError(f"unknown buffer master {master_name!r}")
        cell_id = names.index(master_name)
        start = int(view["cell_arc_start"][cell_id])
        end = int(view["cell_arc_start"][cell_id + 1])
        if start >= end:
            raise ValueError(f"buffer master {master_name!r} has no timing arcs")
        slew_axis = sorted(
            {
                float(value)
                for arc in range(start, end)
                for value in view["arc_trans_axis"][
                    view["arc_trans_start"][arc] : view["arc_trans_start"][arc + 1]
                ]
            }
        )
        load_axis = sorted(
            {
                float(value)
                for arc in range(start, end)
                for value in view["arc_cap_axis"][
                    view["arc_cap_start"][arc] : view["arc_cap_start"][arc + 1]
                ]
            }
        )
        delay = np.zeros((len(slew_axis), len(load_axis)), dtype=np.float64)
        output_slew = np.zeros_like(delay)
        for arc in range(start, end):
            rows = int(view["arc_dim"][2 * arc])
            cols = int(view["arc_dim"][2 * arc + 1])
            size = rows * cols
            trans_axis = view["arc_trans_axis"][
                view["arc_trans_start"][arc] : view["arc_trans_start"][arc + 1]
            ]
            cap_axis = view["arc_cap_axis"][
                view["arc_cap_start"][arc] : view["arc_cap_start"][arc + 1]
            ]
            values = view["arc_values"][
                view["arc_value_start"][arc] : view["arc_value_start"][arc + 1]
            ]
            for row, trans in enumerate(slew_axis):
                for column, cap in enumerate(load_axis):
                    delay[row, column] = max(
                        delay[row, column],
                        _interpolate(values[0 * size : 1 * size], trans_axis, cap_axis, trans, cap),
                        _interpolate(values[1 * size : 2 * size], trans_axis, cap_axis, trans, cap),
                    )
                    output_slew[row, column] = max(
                        output_slew[row, column],
                        _interpolate(values[2 * size : 3 * size], trans_axis, cap_axis, trans, cap),
                        _interpolate(values[3 * size : 4 * size], trans_axis, cap_axis, trans, cap),
                    )
        return cls(
            master_name=master_name,
            input_cap=float(view["cell_input_cap"][cell_id]),
            slew_axis=slew_axis,
            load_axis=load_axis,
            delay=delay,
            output_slew=output_slew,
        )

    def tensors(self, *, dtype=None, device=None):
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        return {
            "input_cap_by_size": torch.tensor([self.input_cap], **kwargs),
            "input_slew_axis": torch.tensor(self.slew_axis, **kwargs),
            "output_load_axis": torch.tensor(self.load_axis, **kwargs),
            "delay_lut": torch.tensor(self.delay, **kwargs),
            "output_slew_lut": torch.tensor(self.output_slew, **kwargs),
        }
