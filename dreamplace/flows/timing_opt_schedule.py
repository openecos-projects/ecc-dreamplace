"""One-shot overflow or inflation events for placement timing windows."""

import math


class TimingOptWindowSchedule:
    def __init__(self, milestones, max_windows):
        self.milestones = tuple(float(value) for value in milestones)
        self.max_windows = int(max_windows)
        if (
            len(self.milestones) > self.max_windows
            or any(not math.isfinite(value) or not 0 < value < 1 for value in self.milestones)
            or any(a <= b for a, b in zip(self.milestones, self.milestones[1:]))
        ):
            raise ValueError(
                "S5B1 milestones must be descending in (0, 1) and fit the window limit"
            )
        self.pending_milestones = list(self.milestones)
        self.attempts = set()
        self.window_count = 0

    @property
    def mode(self):
        return "overflow_milestones" if self.milestones else "inflation"

    def next_overflow_event(self, overflow, *, iteration, stage):
        if not self.pending_milestones or not overflow <= self.pending_milestones[0]:
            return None
        threshold = self.pending_milestones.pop(0)
        self.window_count += 1
        return {
            "mode": self.mode,
            "threshold": threshold,
            "overflow": float(overflow),
            "iteration": int(iteration),
            "stage": int(stage),
        }

    def inflation_event(self, *, iteration, stage):
        attempt = (int(stage), int(iteration))
        if self.milestones or attempt in self.attempts or self.window_count >= self.max_windows:
            return None
        self.attempts.add(attempt)
        self.window_count += 1
        return {"mode": self.mode, "iteration": int(iteration), "stage": int(stage)}
