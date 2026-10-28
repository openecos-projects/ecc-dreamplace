"""Early cell math and shared net/lane adjoints on the selected native device."""

import torch

from . import lut_entry_2d_op
from .cell_aat_native import _CellAat, _tables
from .net_aat_native import _NetAat
from .timing_lanes_native import try_timing_lanes_native


def _merge_updates(timing, arcs, pins, updates):
    # Negation turns the shared max lanes into min lanes, including smooth-min.
    lanes = try_timing_lanes_native(
        timing,
        arcs[:, 1],
        tuple(-pin for pin in pins),
        tuple(-value for value in updates),
    )
    if lanes is not None:
        result = tuple(-lane for lane in lanes)
        return result[:2], result[2:]
    from .timing_propagation import scatter_timing_min

    result = tuple(
        scatter_timing_min(
            pin,
            arcs[:, 1],
            value,
            mode=timing.timing_aggregation_mode if i < 2 else "hard",
            tau_ps=timing.timing_aggregation_tau_ps,
        )
        for i, (pin, value) in enumerate(zip(pins, updates, strict=True))
    )
    return result[:2], result[2:]


def cell_early_native(timing, arcs, arrivals, slews, caps, *, surrogate,
                      latch_data_arcs=None):
    if surrogate or not timing._use_lut_2d_native_op_for_tensor(arrivals[0]):
        return None
    native = lut_entry_2d_op._select_module(arrivals[0])
    if getattr(native, "cell_early_schema_version", 0) != 1:
        raise RuntimeError("early native cell propagation requires a fresh ecc-dreamplace build")
    tables = _tables(timing)
    if tables is None:
        return None
    count = arrivals[0].numel()
    pins = (*arrivals, *slews, *(cap[:count] for cap in caps))
    updates = _CellAat.apply(
        arcs,
        *pins,
        tables,
        timing.timing_aggregation_mode == "smooth",
        1.0 / timing.timing_aggregation_tau_ps,
        True,
    )
    values = tuple(updates[:, edge] for edge in range(4))
    if latch_data_arcs is not None:
        ids = arcs[:, 3].long()
        blocked = (ids >= 0) & latch_data_arcs[ids.clamp(min=0)]
        # Only arrivals are cut. D->Q still participates in the independent
        # minimum-slew model used to query downstream delay and check tables.
        values = (*[torch.where(blocked, torch.inf, value) for value in values[:2]], *values[2:])
    return _merge_updates(timing, arcs, pins[:4], values)


def net_early_native(timing, arcs, arrivals, slews, delays, impulses):
    if not timing._use_lut_2d_native_op_for_tensor(arrivals[0]):
        return None
    from .timing_propagation import SLEW_SQRT_EPS

    safe_slews = tuple(
        torch.where(torch.isfinite(slew), slew, torch.zeros_like(slew)) for slew in slews
    )
    count = arrivals[0].numel()
    updates = list(
        _NetAat.apply(
            arcs,
            *arrivals,
            *safe_slews,
            *(value[:count] for value in (*delays, *impulses)),
            SLEW_SQRT_EPS,
        )
    )
    for edge in range(2):
        valid = torch.isfinite(slews[edge][arcs[:, 0]])
        updates[edge + 2] = torch.where(
            valid, updates[edge + 2], torch.full_like(updates[edge + 2], torch.inf)
        )
    return _merge_updates(timing, arcs, (*arrivals, *slews), tuple(updates))
