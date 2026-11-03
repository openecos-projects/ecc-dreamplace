"""Preserve consumed window counters, while rebuilding all topology-owned caches."""

import copy
from dataclasses import dataclass, field

from dreamplace.flows.timing_opt_schedule import TimingOptWindowSchedule


@dataclass
class MidplaceSchedule:
    timing: TimingOptWindowSchedule | None = None
    timing_round: int = 0
    timing_summaries: list = field(default_factory=list)
    gp_steps: int = 0
    last_inflation_step: int | None = None
    inflation_rounds: dict[int, int] = field(default_factory=dict)

    def capture(self, placer, model, stage):
        owner = getattr(placer, "inflation_s5b1", None)
        if owner is not None:
            self.timing = copy.deepcopy(owner.schedule)
            self.timing_round = owner.round_index
            self.timing_summaries = copy.deepcopy(owner.summaries)
        controller = placer._routability_controller
        self.gp_steps = controller._gp_steps
        self.last_inflation_step = controller._last_inflation_step
        state = getattr(model, "inflation_state", None)
        if state is not None:
            self.inflation_rounds[stage] = int(state.num_area_adjust)

    def restore(self, owner, controller):
        if self.timing is not None:
            if owner is None:
                raise RuntimeError("midplace lost its S5B1 owner during runtime reconstruction")
            owner.schedule = copy.deepcopy(self.timing)
            owner.milestones = owner.schedule.milestones
            owner.pending_milestones = owner.schedule.pending_milestones
            owner.window_count = owner.schedule.window_count
            owner.round_index = self.timing_round
            owner.summaries = copy.deepcopy(self.timing_summaries)
            if owner.lane is not None:
                raise RuntimeError("fresh S5B1 owner unexpectedly retained virtual topology")
        controller._gp_steps = self.gp_steps
        controller._last_inflation_step = self.last_inflation_step
        return {
            "virtual_topology": "discarded_and_rebuilt",
            "routability_cache": "fresh",
            "timing_windows_completed": 0 if self.timing is None else self.timing.window_count,
            "pending_timing_milestones": []
            if self.timing is None
            else list(self.timing.pending_milestones),
            "inflation_rounds_completed": dict(self.inflation_rounds),
            "gp_steps_completed": self.gp_steps,
        }
