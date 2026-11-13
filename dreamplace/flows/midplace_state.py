"""Session-scoped overflow requests and value-only placement continuation state."""

import math
from dataclasses import asdict, dataclass

from dreamplace.ops.placeio_ecc.export_options import resolve_flag


class MidplaceRestart(Exception):
    """The committed database requires a fresh placement call."""


class MidplaceBoundedStop(Exception):
    """The requested experimental continuation is complete."""


def create_session(engine):
    """Keep disabled placement flows independent of repair runtime imports."""
    if not resolve_flag(engine.params, "repair_buffering_midplace_enable"):
        if getattr(engine.params, "_midplace_objectives_pending", False):
            engine.params._midplace_objectives_pending = False
        buffering_active = bool(
            getattr(engine.params, "timing_opt_enabled", False)
            and getattr(engine.params, "timing_opt_buffering_enabled", False)
        ) or (
            str(getattr(engine.params, "joint_quality_profile", "") or "")
            == "discrete_alternating_sizing_buffering_v1"
        )
        if buffering_active:
            from dreamplace.flows.direct_buffer_commit import (
                DirectBufferCommitSession,
                supports_direct_buffer_commit,
            )

            if supports_direct_buffer_commit(engine.params, engine.placedb):
                return DirectBufferCommitSession(engine)
        return None
    from dreamplace.flows.midplace_repair import MidplaceRepairSession

    return MidplaceRepairSession(engine)


@dataclass(frozen=True)
class MidplaceConfig:
    overflow: float
    mode: str
    resume_policy: str
    continue_placement: bool
    max_passes: int
    bounded_steps: int
    artifact: str
    max_buffer_fraction: float = 0.001

    @classmethod
    def from_params(cls, params):
        config = cls(
            overflow=float(params.repair_buffering_midplace_overflow),
            mode=str(params.repair_buffering_midplace_mode),
            resume_policy=str(params.repair_buffering_midplace_resume_policy),
            continue_placement=resolve_flag(params, "repair_buffering_midplace_continue"),
            max_passes=int(params.repair_buffering_midplace_max_passes),
            bounded_steps=int(params.repair_buffering_midplace_bounded_steps),
            artifact=str(params.repair_buffering_midplace_artifact),
            max_buffer_fraction=float(params.repair_buffering_midplace_max_buffer_fraction),
        )
        if not math.isfinite(config.overflow) or config.overflow <= 0:
            raise ValueError("midplace overflow must be finite and positive")
        if (
            not math.isfinite(config.max_buffer_fraction)
            or not 0.0 <= config.max_buffer_fraction <= 1.0
        ):
            raise ValueError("midplace requires 0.0 <= max_buffer_fraction <= 1.0")
        if config.mode not in {"repair_design", "repair_design_then_timing"}:
            raise ValueError("midplace supports repair_design or repair_design_then_timing")
        if config.resume_policy not in {"cold", "warm_schedule"}:
            raise ValueError("midplace resume policy must be cold or warm_schedule")
        if config.bounded_steps < 0 or config.max_passes < 1:
            raise ValueError("midplace requires bounded_steps >= 0 and max_passes >= 1")
        if not resolve_flag(params, "repair_buffering_midplace_once"):
            raise ValueError("midplace currently requires once per placement session")
        if params.repair_buffering_midplace_initial_policy != "first_stable_step":
            raise ValueError("midplace initial policy must be first_stable_step")
        if params.repair_buffering_midplace_pending_policy != "defer":
            raise ValueError("midplace pending policy must be defer")
        if config.mode == "repair_design_then_timing" and not resolve_flag(
            params, "repair_buffering_midplace_timing_enable"
        ):
            raise ValueError("midplace combined mode requires explicit timing_enable")
        if resolve_flag(params, "repair_buffering_enable"):
            raise ValueError("midplace and terminal repair triggers are mutually exclusive")
        if not params.with_sta or str(params.place_io_engine) != "ecc":
            raise ValueError("midplace requires Python timing with the ECC placeio backend")
        if not params.global_place_flag or params.macro_place_flag:
            raise ValueError("midplace requires cell global placement")
        if str(getattr(params, "placement_sizing_mode", "place_only")) != "place_only":
            raise ValueError(
                "midplace currently requires the placement owner, without a joint ECO owner"
            )
        if resolve_flag(params, "openroad_handoff_enable"):
            raise ValueError("native midplace and external OpenROAD handoff are mutually exclusive")
        return config


@dataclass(frozen=True)
class PlacementCursor:
    """The next step's original loop indices; completed stages are never replayed."""

    stage: int = 0
    gamma: int = 0
    density: int = 0
    sub: int = 0
    raw_iteration: int = 0

    def start(self, stage, gamma=None, density=None):
        if stage != self.stage:
            return 0
        if gamma is None:
            return self.gamma
        if gamma != self.gamma:
            return 0
        if density is None:
            return self.density
        return self.sub if density == self.density else 0

    def after_step(self, *, gamma_count, density_count, sub_count):
        stage, gamma, density, sub = self.stage, self.gamma, self.density, self.sub + 1
        if sub >= sub_count:
            density, sub = density + 1, 0
        if density >= density_count:
            gamma, density = gamma + 1, 0
        if gamma >= gamma_count:
            stage, gamma = stage + 1, 0
        return PlacementCursor(stage, gamma, density, sub, self.raw_iteration + 1)


def stage_budgets(stages):
    return tuple(
        int(stage["iteration"])
        * int(stage.get("Llambda_density_weight_iteration", 1))
        * int(stage.get("Lsub_iteration", 1))
        for stage in stages
    )


def remaining_steps(stages, cursor, *, sub_iteration=None):
    if cursor.stage >= len(stages):
        return 0
    stage = stages[cursor.stage]
    sub = int(stage.get("Lsub_iteration", 1)) if sub_iteration is None else sub_iteration
    density = int(stage.get("Llambda_density_weight_iteration", 1))
    current = ((int(stage["iteration"]) - cursor.gamma) * density - cursor.density) * sub
    return current - cursor.sub + sum(stage_budgets(stages[cursor.stage + 1 :]))


@dataclass(frozen=True)
class OverflowObservation:
    overflow: float
    topology_generation: int
    cursor: PlacementCursor
    completed_effective_steps: int
    total_effective_steps: int


@dataclass(frozen=True)
class MidplaceRequest:
    request_id: str
    threshold: float
    previous_overflow: float | None
    crossing: OverflowObservation
    execution: OverflowObservation

    def as_dict(self):
        return asdict(self)


class OverflowTracker:
    """Latch the first crossing, including a first stable step below threshold.

    A pending placement transaction may change both overflow and topology. Only
    observations are retained while waiting: planning uses the execution input.
    Consumption belongs to this session, never to a PyDB or placer generation.
    """

    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.previous_overflow = None
        self.pending = None
        self.consumed = False

    def observe_initial(self, observation):
        """Latch an initially low metric, but execute only after a stable step."""
        if not math.isfinite(observation.overflow):
            raise ValueError("midplace initial overflow is not finite")
        self.previous_overflow = observation.overflow
        if observation.overflow <= self.threshold:
            self.pending = (None, observation)

    def observe(self, observation, *, pending_transaction=False):
        overflow = observation.overflow
        if not math.isfinite(overflow):
            raise ValueError("midplace overflow observation is not finite")
        if self.consumed:
            return None
        if self.pending is None and overflow <= self.threshold:
            self.pending = (self.previous_overflow, observation)
        self.previous_overflow = overflow
        if self.pending is None or pending_transaction:
            return None
        previous, crossing = self.pending
        self.consumed = True
        return MidplaceRequest("midplace-1", self.threshold, previous, crossing, observation)
