from dataclasses import FrozenInstanceError

import pytest
from dreamplace.flows.midplace_state import (
    OverflowObservation,
    OverflowTracker,
    PlacementCursor,
    remaining_steps,
    stage_budgets,
)


def observation(overflow, step=1, generation=0, stage=0):
    return OverflowObservation(
        overflow, generation, PlacementCursor(stage=stage, raw_iteration=step - 1), step, 100
    )


@pytest.mark.parametrize("values", [(0.8, 0.6), (0.9, 0.4), (0.5,), (0.6,)])
def test_first_stable_crossing_is_immutable_and_session_once(values):
    tracker = OverflowTracker(0.6)
    requests = [tracker.observe(observation(value, i + 1)) for i, value in enumerate(values)]
    request = requests[-1]
    assert all(value is None for value in requests[:-1])
    assert request.crossing == request.execution == observation(values[-1], len(values))
    assert request.previous_overflow == (values[-2] if len(values) > 1 else None)
    with pytest.raises(FrozenInstanceError):
        request.threshold = 0.8
    for generation, value in enumerate((0.7, 0.6, 0.5, 0.9, 0.3), start=1):
        assert tracker.observe(observation(value, generation=generation)) is None


def test_pending_crossing_survives_rebound_and_topology_change():
    tracker = OverflowTracker(0.6)
    assert tracker.observe(observation(0.8)) is None
    crossed = observation(0.55, 2)
    assert tracker.observe(crossed, pending_transaction=True) is None
    assert tracker.observe(observation(0.75, 3, generation=1), pending_transaction=True) is None
    execution = observation(0.9, 4, generation=2, stage=1)
    request = tracker.observe(execution)
    assert request.as_dict() == {
        "request_id": "midplace-1",
        "threshold": 0.6,
        "previous_overflow": 0.8,
        "crossing": {
            "overflow": 0.55,
            "topology_generation": 0,
            "cursor": {"stage": 0, "gamma": 0, "density": 0, "sub": 0, "raw_iteration": 1},
            "completed_effective_steps": 2,
            "total_effective_steps": 100,
        },
        "execution": {
            "overflow": 0.9,
            "topology_generation": 2,
            "cursor": {"stage": 1, "gamma": 0, "density": 0, "sub": 0, "raw_iteration": 3},
            "completed_effective_steps": 4,
            "total_effective_steps": 100,
        },
    }


def test_above_threshold_and_nonfinite_are_not_stable_triggers():
    tracker = OverflowTracker(0.6)
    assert tracker.observe(observation(0.600001)) is None
    with pytest.raises(ValueError, match="not finite"):
        tracker.observe(observation(float("nan")))
    assert not tracker.consumed


def test_initially_low_metric_latches_until_first_stable_step_even_after_rebound():
    tracker = OverflowTracker(0.6)
    initial = observation(0.5, step=0)
    tracker.observe_initial(initial)
    assert not tracker.consumed
    execution = observation(0.8, step=1)
    request = tracker.observe(execution)
    assert request.crossing == initial
    assert request.execution == execution


def test_nested_stage_cursor_resumes_exact_remaining_suffix():
    stages = [
        {"iteration": 2, "Llambda_density_weight_iteration": 2, "Lsub_iteration": 3},
        {"iteration": 4, "Lsub_iteration": 2},
    ]
    assert stage_budgets(stages) == (12, 8)
    original = [
        (stage, g, d, s)
        for stage, spec in enumerate(stages)
        for g in range(spec["iteration"])
        for d in range(spec.get("Llambda_density_weight_iteration", 1))
        for s in range(spec.get("Lsub_iteration", 1))
    ]
    for cut, indices in enumerate(original):
        current = PlacementCursor(*indices, raw_iteration=cut)
        spec = stages[current.stage]
        resume = current.after_step(
            gamma_count=spec["iteration"],
            density_count=spec.get("Llambda_density_weight_iteration", 1),
            sub_count=spec.get("Lsub_iteration", 1),
        )
        remaining = [
            (stage, g, d, s)
            for stage, spec in enumerate(stages)
            if stage >= resume.stage
            for g in range(resume.start(stage), spec["iteration"])
            for d in range(resume.start(stage, g), spec.get("Llambda_density_weight_iteration", 1))
            for s in range(resume.start(stage, g, d), spec.get("Lsub_iteration", 1))
        ]
        assert remaining == original[cut + 1 :]
        assert remaining_steps(stages, resume) == len(remaining)
        assert resume.raw_iteration == cut + 1


def test_remaining_budget_excludes_an_early_stopped_stage_and_tracks_live_sub_count():
    stages = [{"iteration": 10, "Lsub_iteration": 3}, {"iteration": 5, "Lsub_iteration": 2}]
    # Stage zero stopped after four actual steps; its unused 26 are not replayed.
    assert remaining_steps(stages, PlacementCursor(stage=1, raw_iteration=4)) == 10
    # Inflation changed the current stage's inner loop to four steps.
    assert remaining_steps(stages, PlacementCursor(gamma=2, sub=1), sub_iteration=4) == 41
