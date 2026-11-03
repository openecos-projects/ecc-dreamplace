"""Transport native graph deltas into the existing Python TimingPropagation."""

import time

import numpy as np
import torch

_EVENT_SCHEDULE_KEYS = (
    "pin_rank",
    "cell_out_start",
    "cell_out_ids",
    "cell_in_start",
    "cell_in_ids",
    "net_out_start",
    "net_out_ids",
    "net_in_start",
    "net_in_ids",
)


def _rc_lanes(owner):
    count = len(owner.pydb.pin_names)
    return np.stack(
        [
            values[transition][:count].detach().cpu().numpy()
            for values in (owner.loads, owner.delays, owner.impulses)
            for transition in ("rise", "fall")
        ]
    )


class IncrementalTimingState:
    """One immutable previous graph/state used to evaluate the next net patch."""

    def __init__(self, previous):
        self.previous = previous
        self.operator = previous.model
        self.arcs_info = self.operator.arcs_info
        self.schedule = None

    def evaluate_affected(self, current):
        from dreamplace.ops.rebuffer import rebuffer_cpp

        started = time.perf_counter()
        native = getattr(current, "native_schedule", None)
        if native is None or any(key not in native for key in _EVENT_SCHEDULE_KEYS):
            changed_pins = getattr(current, "changed_pins", None)
            schedule_args = (
                self.previous.pydb,
                current.pydb,
                _rc_lanes(self.previous),
                _rc_lanes(current),
            )
            native = (
                rebuffer_cpp.affected_timing_schedule(*schedule_args, changed_pins)
                if changed_pins is not None
                else rebuffer_cpp.affected_timing_schedule(*schedule_args)
            )
        self.schedule = {
            key: torch.from_numpy(value)
            for key, value in native.items()
            if isinstance(value, np.ndarray)
        }
        schedule_cache_hit = bool(native.get("schedule_cache_hit", False))
        schedule_ms = (time.perf_counter() - started) * 1000
        values = current.model.forward_affected(
            self.operator, self.schedule, current.delays, current.impulses, current.loads
        )
        profile = getattr(current.model, "last_incremental_profile", None)
        if not isinstance(profile, dict):
            profile = {}
            current.model.last_incremental_profile = profile
        profile["schedule_cache_hit"] = schedule_cache_hit
        current.model.last_incremental_profile["schedule_ms"] = schedule_ms
        return current._snapshot_from_outputs(*values[:3])


__all__ = ["IncrementalTimingState"]
