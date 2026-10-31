"""Transport active Python TP electrical arrays to the native ERC boundary."""

from ..rebuffer import rebuffer_cpp
from ..rebuffer.payloads import _numpy


def electrical_observation(timing_model, *, max_fanout=32, cap_margin=0.0, slew_margin=0.0):
    runtime = timing_model.current_graph()
    active = timing_model if runtime is None else runtime.model
    values = {
        "pin_slew_rise": _numpy(active.model.pin_rtran_live),
        "pin_slew_fall": _numpy(active.model.pin_ftran_live),
        "pin_load_rise": _numpy(active.loads["rise"]),
        "pin_load_fall": _numpy(active.loads["fall"]),
        "pin_max_slew": _numpy(active.pin_max_slew),
        "pin_max_cap": _numpy(active.pin_max_cap),
    }
    return dict(
        rebuffer_cpp.electrical_snapshot(
            active.pydb,
            values,
            int(max_fanout),
            float(cap_margin),
            float(slew_margin),
        )
    )
