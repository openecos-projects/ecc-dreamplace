"""Opt-in phase measurements across the repair commit and refresh boundary."""

import resource
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field


@dataclass
class RepairProfile:
    timings_ms: dict[str, float] = field(default_factory=dict)
    phase_rss_peak_kib: dict[str, int] = field(default_factory=dict)
    commits: dict[str, dict] = field(default_factory=dict)
    plans: dict[str, dict] = field(default_factory=dict)
    failure_stage: str | None = None

    @contextmanager
    def measure(self, phase):
        started = time.perf_counter()
        try:
            yield
        except Exception:
            if self.failure_stage is None:
                self.failure_stage = phase
            raise
        finally:
            self.phase_rss_peak_kib[phase] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.timings_ms[phase] = (
                self.timings_ms.get(phase, 0.0) + (time.perf_counter() - started) * 1000.0
            )


def measure_phase(profile, phase):
    return nullcontext() if profile is None else profile.measure(phase)
