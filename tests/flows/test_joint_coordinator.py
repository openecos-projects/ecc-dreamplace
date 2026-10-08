from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamplace.flows.joint_coordinator import (
    JointCoordinator,
    Pin2PinPairGenerationState,
    SegmentJointMilestoneState,
)
from dreamplace.flows.joint_proximal import JointProximalObjective
from dreamplace.flows.segment_joint import (
    SegmentJointContext,
    apply_segment_overflow_milestone_transaction,
    apply_segment_rebootstrap_milestone_transaction,
    apply_segment_virtual_window_boundary,
    prepare_segment_direct_joint_iteration,
)
from dreamplace.NonLinearPlace import NonLinearPlace


class FakeBufferingLane:
    def __init__(self):
        self.z_param = torch.nn.Parameter(torch.tensor([-2.0, 4.5]))
        self.calls = []

    def prepare(self, model=None, params=None):
        self.calls.append(("prepare", model, params))
        return self

    def attach_state(self, data_collections):
        self.calls.append(("attach_state", data_collections))
        return self

    def summarize(self):
        return {"status": "ok", "state_kind": "segment_count"}

    def optimizer_parameters(self):
        return [self.z_param]

    def after_step(self, iteration, loss, metrics):
        with torch.no_grad():
            self.z_param.clamp_(0.0, 3.0)
        metrics["z_param_min"] = float(self.z_param.min())
        metrics["z_param_max"] = float(self.z_param.max())
        metrics["projected_after_step"] = True

    def project(self, **kwargs):
        self.calls.append(("project", kwargs))
        return SimpleNamespace(projected_buffer_count=2)

    def refresh_runtime_state(self, projection_result):
        self.calls.append(("refresh_runtime_state", projection_result))
        return {"status": "ok"}

    def build_commit_request(self, projection_result, **kwargs):
        self.calls.append(("build_commit_request", projection_result, kwargs))
        return SimpleNamespace(to_summary=lambda: {"request_version": 1})

    placedb = None
    config = SimpleNamespace(output_dir="out")


def _params():
    return SimpleNamespace(
        flow_kind="joint",
        placement_sizing_mode="joint",
        joint_buffer_overflow_gate=0.2,
        joint_buffer_commit_period=100,
    )


class FakeSegmentJointLane(FakeBufferingLane):
    def __init__(self):
        super().__init__()
        self.z_param = torch.nn.Parameter(torch.zeros(4))
        self.state = SimpleNamespace(
            z_param=self.z_param,
            bsu_index_param=torch.nn.Parameter(torch.full((4,), 7.0)),
            segment_net_index=torch.tensor([0, 0, 1, 2], dtype=torch.long),
            net_ids=torch.tensor([10, 20, 30], dtype=torch.long),
            segment_ids=torch.tensor([100, 101, 200, 300], dtype=torch.long),
            max_repeater_count=3,
        )

    def _current_state(self):
        return self.state


def _segment_joint_params(schedule_mode="overflow_milestones"):
    params = _params()
    params.joint_quality_profile = "segment_count_direct_joint_v1"
    params.joint_segment_action_schedule_mode = schedule_mode
    return params


def _pin2pin_segment_joint_params():
    params = _segment_joint_params()
    params.enable_net_weighting = 1
    params.net_weighting_scheme = "pin2pin"
    params.net_weighting_update_interval = 2
    params.timing_topology_enable_overflow_threshold = 0.35
    params.joint_segment_milestones = (0.30, 0.25, 0.20, 0.15, 0.10)
    return params


def test_pin2pin_rebootstrap_schedule_can_be_disabled_for_v2_replay():
    params = _pin2pin_segment_joint_params()
    params.joint_segment_pin2pin_rebootstrap_enabled = 0

    coordinator = JointCoordinator(params, buffering_lane=FakeSegmentJointLane())

    assert not coordinator.uses_pin2pin_rebootstrap_schedule
    assert coordinator.pin2pin_pair_generations is None


def test_no_sizing_segment_joint_summary_disables_sizing_lane():
    params = _segment_joint_params()
    params.joint_segment_sizing_enabled = 0
    params.placement_sizing_mode = "place_only"

    coordinator = JointCoordinator(params, buffering_lane=FakeSegmentJointLane())

    summary = coordinator.summarize()
    assert summary["sizing_enabled"] is False
    assert summary["activation"]["sizing_active_after_overflow_gate"] is False


def test_no_sizing_segment_joint_keeps_placement_optimizer():
    params = _segment_joint_params()
    params.joint_segment_sizing_enabled = 0
    params._joint_segment_sizing_enabled_explicit = True
    params.placement_sizing_mode = "place_only"
    params.global_place_stages = [{"optimizer": "nesterov"}]

    from dreamplace.flows.flow_config import apply_flow_defaults

    apply_flow_defaults(params)

    assert params.global_place_stages[0]["optimizer"] == "nesterov"


def test_joint_coordinator_collects_three_lane_param_groups():
    lane = FakeBufferingLane()
    coordinator = JointCoordinator(_params(), buffering_lane=lane)
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    size = torch.nn.Parameter(torch.tensor([2.0]))

    groups = coordinator.collect_param_groups(
        placement_params=[pos],
        sizing_params=[size],
    )

    assert [group["group_name"] for group in groups] == [
        "placement",
        "sizing",
        "buffering",
    ]
    assert groups[0]["joint_activation"] == "always"
    assert groups[1]["joint_activation"] == "overflow_gate"
    assert groups[2]["joint_activation"] == "overflow_gate"
    summary = coordinator.summarize()
    assert not summary["uses_joint_op"]
    assert summary["param_groups"][2]["param_count"] == 1


def test_pin2pin_generation_tracks_objective_and_step_consumption_cadence():
    state = Pin2PinPairGenerationState(update_interval=2)
    generation = state.install(
        iteration=7,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=12,
    )

    assert generation["generation_id"] == 1
    assert not state.periodic_update_due
    state.record_objective_evaluation(iteration=7, generation_id=1)
    state.record_placement_step(iteration=7, generation_id=1)
    assert not state.periodic_update_due
    state.record_objective_evaluation(iteration=8, generation_id=1)
    state.record_placement_step(iteration=8, generation_id=1)

    assert state.periodic_update_due
    summary = state.summarize()
    assert summary["current_generation"]["objective_eval_count"] == 2
    assert summary["current_generation"]["consumed_step_count"] == 2
    assert summary["next_periodic_due_after_consumed_steps"] == 2


def test_pin2pin_generation_rejects_two_installs_in_one_iteration():
    state = Pin2PinPairGenerationState(update_interval=15)
    state.install(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_clean",
        pair_count=0,
    )

    with pytest.raises(RuntimeError, match="already installed"):
        state.install(
            iteration=4,
            reason="milestone_rebootstrap",
            status="timing_violating",
            pair_count=1,
        )


def test_pin2pin_nesterov_consumption_requires_real_optimizer_evaluation():
    coordinator = JointCoordinator(
        _pin2pin_segment_joint_params(),
        buffering_lane=FakeSegmentJointLane(),
    )
    generation = coordinator.install_pin2pin_generation(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=12,
    )

    with pytest.raises(RuntimeError, match="did not evaluate"):
        coordinator.record_pin2pin_nesterov_step(
            iteration=4,
            generation_id=generation["generation_id"],
            optimizer_eval_count_before=10,
            optimizer_eval_count_after=10,
        )

    evidence = coordinator.record_pin2pin_nesterov_step(
        iteration=4,
        generation_id=generation["generation_id"],
        optimizer_eval_count_before=10,
        optimizer_eval_count_after=13,
    )

    assert evidence["optimizer_eval_count_delta"] == 3
    current = coordinator.pin2pin_pair_generations.current_generation
    assert current["objective_eval_count"] == 1
    assert current["consumed_step_count"] == 1


def test_pin2pin_generation_rejects_stale_or_unpaired_consumption():
    state = Pin2PinPairGenerationState(update_interval=15)
    state.install(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=1,
    )

    with pytest.raises(RuntimeError, match="objective evaluation"):
        state.record_placement_step(iteration=4, generation_id=1)
    state.record_objective_evaluation(iteration=4, generation_id=1)
    with pytest.raises(RuntimeError, match="current generation"):
        state.record_placement_step(iteration=4, generation_id=2)


@pytest.mark.parametrize(
    ("status", "pair_count", "message"),
    (
        ("timing_clean", 1, "zero pairs"),
        ("timing_violating", 0, "positive pair count"),
        ("unknown", 0, "invalid Pin2Pin generation status"),
    ),
)
def test_pin2pin_generation_validates_status_and_pair_count(
    status, pair_count, message
):
    state = Pin2PinPairGenerationState(update_interval=15)

    with pytest.raises(ValueError, match=message):
        state.install(
            iteration=1,
            reason="fixture",
            status=status,
            pair_count=pair_count,
        )


def test_coordinator_defers_crossed_milestone_until_activation_generation_is_consumed():
    coordinator = JointCoordinator(
        _pin2pin_segment_joint_params(),
        buffering_lane=FakeSegmentJointLane(),
    )

    before = coordinator.observe_pin2pin_segment_iteration(
        overflow=0.40,
        iteration=3,
    )
    assert not before["activation_due"]
    assert before["queue_after"] == []

    activated = coordinator.observe_pin2pin_segment_iteration(
        overflow=0.29,
        iteration=4,
    )
    assert activated["activation_due"]
    assert activated["queue_after"] == [0.30]
    generation = coordinator.install_pin2pin_generation(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=8,
    )
    coordinator.record_pin2pin_objective_evaluation(
        iteration=4,
        generation_id=generation["generation_id"],
    )
    coordinator.record_pin2pin_placement_step(
        iteration=4,
        generation_id=generation["generation_id"],
    )
    assert coordinator.begin_pin2pin_milestone_after_step(
        iteration=4,
        overflow=0.29,
    ) is None

    coordinator.record_pin2pin_objective_evaluation(iteration=5, generation_id=1)
    coordinator.record_pin2pin_placement_step(iteration=5, generation_id=1)
    event = coordinator.begin_pin2pin_milestone_after_step(
        iteration=5,
        overflow=0.29,
    )

    assert event["milestone"] == 0.30
    assert event["pair_generation_before_milestone"] == 1
    summary = coordinator.summarize()
    assert summary["pin2pin_pair_generations"]["current_generation"][
        "consumed_step_count"
    ] == 2


def test_timing_clean_confirmation_skips_one_queued_milestone_and_restarts_generation():
    coordinator = JointCoordinator(
        _pin2pin_segment_joint_params(),
        buffering_lane=FakeSegmentJointLane(),
    )
    coordinator.observe_pin2pin_segment_iteration(overflow=0.29, iteration=4)
    first = coordinator.install_pin2pin_generation(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_clean",
        pair_count=0,
    )
    coordinator.record_pin2pin_objective_evaluation(
        iteration=4,
        generation_id=first["generation_id"],
    )
    coordinator.record_pin2pin_placement_step(
        iteration=4,
        generation_id=first["generation_id"],
    )
    coordinator.record_pin2pin_objective_evaluation(iteration=5, generation_id=1)
    coordinator.record_pin2pin_placement_step(iteration=5, generation_id=1)
    confirmation = coordinator.install_pin2pin_generation(
        iteration=5,
        reason="milestone_clean_confirmation",
        status="timing_clean",
        pair_count=0,
    )

    record = coordinator.skip_pin2pin_milestone_after_timing_clean_confirmation(
        iteration=5,
        overflow=0.29,
        consumed_pair_generation=1,
        confirmation_pair_generation=confirmation["generation_id"],
        evidence={"wns": 0.0, "tns": 0.0, "pair_count": 0},
    )

    assert record["status"] == "skipped_timing_clean"
    assert record["milestone"] == 0.30
    assert record["pair_generation_before_milestone"] == 1
    assert record["pair_generation_after_milestone"] == 2
    assert coordinator.segment_milestones.queue == []


def test_incomplete_pending_milestone_preserves_queue_and_generation_evidence():
    coordinator = JointCoordinator(
        _pin2pin_segment_joint_params(),
        buffering_lane=FakeSegmentJointLane(),
    )
    coordinator.observe_pin2pin_segment_iteration(overflow=0.19, iteration=4)
    coordinator.install_pin2pin_generation(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=4,
    )

    result = coordinator.mark_incomplete_pending_milestone(
        iteration=3000,
        reason="placement_step_budget_exhausted",
    )

    assert result["status"] == "incomplete_pending_milestone"
    assert result["queue"] == [0.30, 0.25, 0.20]
    assert result["pair_generation"]["generation_id"] == 1
    assert coordinator.summarize()["status"] == "incomplete_pending_milestone"


def test_joint_coordinator_gates_sizing_and_buffering_learning_rates():
    coordinator = JointCoordinator(_params(), buffering_lane=FakeBufferingLane())
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    size = torch.nn.Parameter(torch.tensor([2.0]))
    buf = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = torch.optim.SGD(
        [
            {"params": [pos], "group_name": "placement", "joint_lane": "placement"},
            {"params": [size], "group_name": "sizing", "joint_lane": "sizing"},
            {"params": [buf], "group_name": "buffering", "joint_lane": "buffering"},
        ],
        lr=1.0,
    )

    assert not coordinator.apply_activation_to_optimizer(
        optimizer,
        overflow=0.5,
        sizing_lr=0.2,
        buffering_lr=0.03,
    )
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[2]["lr"] == 0.0

    assert coordinator.apply_activation_to_optimizer(
        optimizer,
        overflow=0.1,
        sizing_lr=0.2,
        buffering_lr=0.03,
    )
    assert optimizer.param_groups[1]["lr"] == 0.2
    assert optimizer.param_groups[2]["lr"] == 0.03


def test_joint_coordinator_activation_gate_is_inclusive():
    coordinator = JointCoordinator(_params(), buffering_lane=FakeBufferingLane())

    assert coordinator.should_enable_joint_lanes(0.2)
    assert not coordinator.should_enable_joint_lanes(0.2000001)


def test_joint_coordinator_clamps_segment_z_after_step():
    lane = FakeBufferingLane()
    coordinator = JointCoordinator(_params(), buffering_lane=lane)

    metrics = coordinator.post_step_clamp(iteration=7)

    assert torch.allclose(lane.z_param, torch.tensor([0.0, 3.0]))
    assert metrics["projected_after_step"]
    assert metrics["z_param_min"] == 0.0
    assert metrics["z_param_max"] == 3.0


def test_joint_coordinator_reports_gradient_metrics_per_lane():
    coordinator = JointCoordinator(_params(), buffering_lane=FakeBufferingLane())
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    size = torch.nn.Parameter(torch.tensor([2.0]))
    buf = torch.nn.Parameter(torch.tensor([3.0]))
    loss = pos.square().sum() + size.square().sum() + buf.square().sum()
    loss.backward()
    optimizer = torch.optim.SGD(
        [
            {"params": [pos], "group_name": "placement", "joint_lane": "placement"},
            {"params": [size], "group_name": "sizing", "joint_lane": "sizing"},
            {"params": [buf], "group_name": "buffering", "joint_lane": "buffering"},
        ],
        lr=1.0,
    )

    metrics = coordinator.gradient_metrics(optimizer)

    assert metrics["placement_grad_param_count"] == 1
    assert metrics["sizing_grad_param_count"] == 1
    assert metrics["buffering_grad_param_count"] == 1
    assert metrics["placement_grad_norm"] == 2.0
    assert metrics["sizing_grad_norm"] == 4.0
    assert metrics["buffering_grad_norm"] == 6.0


def test_joint_coordinator_commit_period_requires_open_overflow_gate():
    coordinator = JointCoordinator(_params(), buffering_lane=FakeBufferingLane())

    assert not coordinator.should_commit_buffers(0, 0.1)
    assert not coordinator.should_commit_buffers(100, 0.3)
    assert not coordinator.should_commit_buffers(99, 0.1)
    assert coordinator.should_commit_buffers(100, 0.1)


def test_joint_coordinator_uses_explicit_timing_topology_gate_for_forced_smoke():
    params = _params()
    params._timing_topology_enable_overflow_threshold_explicit = True
    params.timing_topology_enable_overflow_threshold = 1.1
    params.joint_buffer_commit_period = 1
    coordinator = JointCoordinator(params, buffering_lane=FakeBufferingLane())

    assert coordinator.overflow_gate == 1.1
    assert coordinator.should_commit_buffers(1, 0.27)


def test_joint_coordinator_prepares_commit_request_without_physical_execution():
    lane = FakeBufferingLane()
    coordinator = JointCoordinator(_params(), buffering_lane=lane)

    result = coordinator.prepare_buffer_commit_request(iteration=100)

    assert result["status"] == "request_ready"
    assert result["projected_buffer_count"] == 2
    assert result["runtime_refresh"]["status"] == "ok"
    assert [call[0] for call in lane.calls] == [
        "project",
        "refresh_runtime_state",
        "build_commit_request",
    ]
    assert result["commit_request"] == {"request_version": 1}
    assert coordinator.last_buffer_commit_request is not None
    assert not hasattr(JointCoordinator, "project_and_commit_buffers")


def test_joint_coordinator_passes_final_live_projection_context_to_lane():
    lane = FakeBufferingLane()
    coordinator = JointCoordinator(_params(), buffering_lane=lane)
    context = {"coordinate_source": "final_live_geometry"}

    coordinator.prepare_buffer_commit_request(
        iteration=100,
        reason="joint_buffer_final_commit",
        live_projection_context=context,
    )

    assert lane.calls[0] == (
        "project",
        {
            "iteration": 100,
            "final": True,
            "live_projection_context": context,
        },
    )


def _begin_queued_milestone(
    state,
    *,
    overflow=0.29,
    observed_iteration=7,
    step_iteration=8,
    pair_generation=1,
):
    state.activate(iteration=observed_iteration, overflow=0.35)
    state.queue_crossed(overflow=overflow, iteration=observed_iteration)
    return state.begin_next_after_step(
        iteration=step_iteration,
        overflow=overflow,
        pair_generation=pair_generation,
        pair_generation_consumed=True,
        pair_rebuild_serviced=False,
    )


def test_segment_joint_milestones_are_one_shot_across_overflow_rebound():
    state = SegmentJointMilestoneState()

    event_030 = _begin_queued_milestone(
        state,
        overflow=0.29,
        observed_iteration=10,
        step_iteration=11,
    )
    state.set_frozen_topology_generation(4)
    state.consume(event_030, transition={"accepted_action_count": 1})
    assert state.queue_crossed(overflow=0.34, iteration=12)["newly_queued"] == []
    assert state.queue_crossed(overflow=0.29, iteration=13)["newly_queued"] == []
    state.queue_crossed(overflow=0.24, iteration=14)
    event_025 = state.begin_next_after_step(
        iteration=14,
        overflow=0.24,
        pair_generation=2,
        pair_generation_consumed=True,
        pair_rebuild_serviced=False,
    )
    state.consume(event_025, transition={"accepted_action_count": 0})

    summary = state.summarize()
    assert [record["milestone"] for record in summary["trace"]] == [0.30, 0.25]
    assert all(record["frozen_topology_generation"] == 4 for record in summary["trace"])
    assert summary["status"][0]["status"] == "consumed"
    assert summary["status"][1]["status"] == "consumed"


def test_segment_joint_low_overflow_activation_queues_every_crossed_milestone():
    state = SegmentJointMilestoneState()

    activation = state.activate(iteration=1, overflow=0.18)
    observation = state.queue_crossed(overflow=0.18, iteration=1)

    assert activation["activated"]
    assert observation["newly_queued"] == [0.30, 0.25, 0.20]
    assert observation["queue_after"] == [0.30, 0.25, 0.20]
    status = {row["milestone"]: row["status"] for row in state.summarize()["status"]}
    assert status[0.30] == "queued"
    assert status[0.25] == "queued"
    assert status[0.20] == "queued"
    assert status[0.15] == "pending"


def test_segment_joint_multi_threshold_jump_preserves_descending_queue():
    state = SegmentJointMilestoneState()
    first = _begin_queued_milestone(
        state,
        overflow=0.29,
        observed_iteration=1,
        step_iteration=2,
    )
    state.consume(first, transition={"accepted_action_count": 0})

    observation = state.queue_crossed(overflow=0.18, iteration=3)
    event = state.begin_next_after_step(
        iteration=3,
        overflow=0.18,
        pair_generation=2,
        pair_generation_consumed=True,
        pair_rebuild_serviced=False,
    )

    assert observation["newly_queued"] == [0.25, 0.20]
    assert event["milestone"] == 0.25
    assert state.summarize()["queue"] == [0.20]
    status = {row["milestone"]: row["status"] for row in state.summarize()["status"]}
    assert status[0.25] == "ready"
    assert status[0.20] == "queued"


def test_segment_joint_activation_install_defers_milestone_until_later_iteration():
    state = SegmentJointMilestoneState()
    state.activate(iteration=4, overflow=0.19)
    state.queue_crossed(overflow=0.19, iteration=4)

    assert state.begin_next_after_step(
        iteration=4,
        overflow=0.19,
        pair_generation=1,
        pair_generation_consumed=True,
        pair_rebuild_serviced=True,
    ) is None
    event = state.begin_next_after_step(
        iteration=5,
        overflow=0.19,
        pair_generation=1,
        pair_generation_consumed=True,
        pair_rebuild_serviced=False,
    )
    assert event["milestone"] == 0.30
    assert event["pair_generation_before_milestone"] == 1


def test_segment_joint_transaction_requires_ordered_stages_before_consumption():
    state = SegmentJointMilestoneState()
    event = _begin_queued_milestone(state)
    state.set_frozen_topology_generation(3)

    for stage in SegmentJointMilestoneState.REBOOTSTRAP_TRANSACTION_STAGES[1:]:
        state.advance(event, stage=stage, evidence={"stage": stage})
    record = state.consume(
        event,
        transition={"accepted_action_count": 1},
        require_stage="tdp_rebootstrap_completed",
    )

    assert record["status"] == "consumed"
    assert [row["stage"] for row in record["stage_trace"]] == list(
        SegmentJointMilestoneState.REBOOTSTRAP_TRANSACTION_STAGES
    )


def test_segment_joint_transaction_rejects_out_of_order_stage():
    state = SegmentJointMilestoneState()
    event = _begin_queued_milestone(state)

    with pytest.raises(RuntimeError, match="must advance exactly once"):
        state.advance(event, stage="buffering_gradient_ready")


def test_segment_joint_transaction_failure_is_terminal_and_auditable():
    state = SegmentJointMilestoneState()
    event = _begin_queued_milestone(state)
    state.advance(event, stage="sizing_micro_loop_completed")

    record = state.fail(
        event,
        stage="sizing_micro_loop_completed",
        error="fixture failure",
    )

    assert record["status"] == "failed_terminal"
    assert state.summarize()["terminal_failure"]["error"] == "fixture failure"
    with pytest.raises(RuntimeError, match="failed terminally"):
        state.queue_crossed(overflow=0.24, iteration=8)


def test_segment_direct_joint_defers_lane_and_excludes_sizing_and_z_from_optimizer():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    size = torch.nn.Parameter(torch.tensor([2.0]))

    coordinator.prepare_lanes(model="model", data_collections="data")
    groups = coordinator.collect_param_groups(
        placement_params=[pos],
        sizing_params=[size],
    )

    assert lane.calls == []
    assert [group["group_name"] for group in groups] == ["placement"]
    assert all(
        all(parameter is not lane.z_param for parameter in group["params"])
        for group in groups
    )


def test_segment_direct_joint_records_resolved_action_schedule():
    coordinator = JointCoordinator(
        _segment_joint_params("overflow_milestones"),
        buffering_lane=FakeSegmentJointLane(),
    )

    assert coordinator.uses_overflow_milestone_actions
    assert not coordinator.uses_fixed_window_actions
    assert (
        coordinator.summarize()["segment_action_schedule_mode"]
        == "overflow_milestones"
    )


def test_segment_joint_prepare_context_owns_activation_callbacks():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            inst_cell_id=torch.tensor([7, 9], dtype=torch.long)
        )
    )
    callback_state = {"initial_ids": None, "freeze": [], "refresh": []}
    context = SegmentJointContext(
        coordinator=coordinator,
        freeze_live_timing_topology=lambda pos: callback_state["freeze"].append(pos)
        or 13,
        refresh_live_timing_topology=lambda pos: callback_state["refresh"].append(pos),
        record_initial_inst_cell_id=lambda current_model: callback_state.update(
            initial_ids=current_model.data_collections.inst_cell_id.detach().clone()
        ),
    )

    event = prepare_segment_direct_joint_iteration(
        context,
        model=model,
        pos=torch.tensor([1.0]),
        overflow=torch.tensor([0.29]),
        iteration=7,
    )

    assert event["activation"]
    assert callback_state["initial_ids"].tolist() == [7, 9]
    assert len(callback_state["freeze"]) == 1
    assert callback_state["refresh"] == []
    assert coordinator._segment_lane_prepared


def test_segment_joint_window_context_preserves_boundary_schedule():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(
        _segment_joint_params("fixed_window_compat"),
        buffering_lane=lane,
    )
    model = SimpleNamespace(
        data_collections=SimpleNamespace(buffer_segment_count_state=lane.state)
    )
    coordinator.activate_segment_lane(
        model=model,
        data_collections=model.data_collections,
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
    params = SimpleNamespace(
        joint_segment_virtual_window_steps=20,
        joint_segment_virtual_warmup_steps=0,
        joint_buffer_outer_iterations=2,
    )
    context = SegmentJointContext(
        coordinator=coordinator,
        freeze_live_timing_topology=lambda pos: 3,
        refresh_live_timing_topology=lambda pos: None,
        record_initial_inst_cell_id=lambda current_model: None,
    )

    assert (
        apply_segment_virtual_window_boundary(
            context,
            params=params,
            model=model,
            pos=torch.tensor([1.0]),
            optimizer=None,
            iteration=18,
        )
        is None
    )
    record = apply_segment_virtual_window_boundary(
        context,
        params=params,
        model=model,
        pos=torch.tensor([1.0]),
        optimizer=None,
        iteration=19,
    )

    assert record["window_index"] == 0
    assert record["accepted_action_count"] == 1
    assert record["following_placement_steps"] == 20


def _overflow_transaction_context(coordinator, *, area_delta=1.0):
    data = SimpleNamespace(
        size_logits=None,
        buffering_timing_topology={
            "topology_generation": 3,
            "frozen_topology_generation": 3,
        },
        runtime_cell_state_generation=1,
    )
    calls = []

    def apply_sizing(**kwargs):
        calls.append("sizing")
        return {
            "status": "applied",
            "num_changed_instances": 1,
            "area_delta_internal": area_delta,
            "density_visible_area_delta_internal": 1.0,
        }

    def refresh(pos):
        calls.append("refresh")
        return data.buffering_timing_topology

    def gradient(**kwargs):
        calls.append("gradient")
        coordinator.buffering_lane.z_param.grad = torch.tensor(
            [-4.0, -2.0, 1.0, 3.0]
        )
        return {"status": "ready", "runtime_ms": 1.0}

    context = SegmentJointContext(
        coordinator=coordinator,
        freeze_live_timing_topology=lambda pos: 3,
        refresh_live_timing_topology=refresh,
        record_initial_inst_cell_id=lambda model: None,
        data_collections=data,
        apply_discrete_sizing=apply_sizing,
        compact_sizing=lambda sizing: dict(sizing),
        compute_buffer_gradient=gradient,
    )
    return context, calls


def test_segment_joint_overflow_transaction_commits_after_refresh_and_gradient():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model", data_collections="data", topology_generation=3
    )
    coordinator.advance_segment_milestone(
        event=event, stage="sizing_gradient_frozen"
    )
    context, calls = _overflow_transaction_context(coordinator)

    record = apply_segment_overflow_milestone_transaction(
        context,
        params=_segment_joint_params(),
        model=SimpleNamespace(),
        pos=torch.tensor([1.0]),
        optimizer=None,
        event=event,
        sizing_frame={"summary": {"event_id": event["event_id"]}},
    )

    assert calls == ["sizing", "refresh", "gradient"]
    assert record["status"] == "consumed"
    assert record["transition"]["route_b"]["accepted_action_count"] == 1
    assert int(lane.z_param.detach().sum().item()) == 1
    assert lane.z_param.grad is None


def test_segment_joint_overflow_transaction_marks_refresh_failure_terminal():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model", data_collections="data", topology_generation=3
    )
    coordinator.advance_segment_milestone(
        event=event, stage="sizing_gradient_frozen"
    )
    context, calls = _overflow_transaction_context(coordinator, area_delta=2.0)

    with pytest.raises(RuntimeError, match="not visible to density"):
        apply_segment_overflow_milestone_transaction(
            context,
            params=_segment_joint_params(),
            model=SimpleNamespace(),
            pos=torch.tensor([1.0]),
            optimizer=None,
            event=event,
            sizing_frame={"summary": {"event_id": event["event_id"]}},
        )

    assert calls == ["sizing", "refresh"]
    failure = coordinator.segment_milestones.summarize()["terminal_failure"]
    assert failure["status"] == "failed_terminal"
    assert failure["transaction_stage"] == "sizing_applied"


def _rebootstrap_transaction_fixture(*, fail_rebuild=False):
    lane = FakeSegmentJointLane()
    params = _pin2pin_segment_joint_params()
    params.joint_segment_sizing_enabled = False
    coordinator = JointCoordinator(params, buffering_lane=lane)
    coordinator.observe_pin2pin_segment_iteration(overflow=0.29, iteration=4)
    data = SimpleNamespace(size_logits=None)
    coordinator.activate_segment_lane(
        model="model", data_collections=data, topology_generation=3
    )
    coordinator.install_pin2pin_generation(
        iteration=4,
        reason="activation_bootstrap",
        status="timing_violating",
        pair_count=8,
    )
    calls = []

    def refresh(pos):
        calls.append("refresh")

    def gradient(**kwargs):
        calls.append("gradient")
        lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
        return {"status": "ready", "runtime_ms": 1.0}

    def rebuild(*args, **kwargs):
        calls.append("rebuild")
        if fail_rebuild:
            raise RuntimeError("fixture rebuild failure")
        return {
            "generation_status": "timing_violating",
            "pin2pin_pair_count": 8,
            "update_count": 2,
            "wns": -1.0,
            "tns": -2.0,
            "pin2pin_endpoint_limit": 1,
            "pin2pin_path_pair_backend": "fixture",
            "pin2pin_backend": "fixture",
            "update_ms": 1.0,
        }, (0.0, 0.0, 0.0, 0.0)

    context = SegmentJointContext(
        coordinator=coordinator,
        freeze_live_timing_topology=lambda pos: 3,
        refresh_live_timing_topology=refresh,
        record_initial_inst_cell_id=lambda model: None,
        data_collections=data,
        compute_buffer_gradient=gradient,
        profile_enabled=lambda params: False,
        profile_clock=lambda enabled, pos: 1.0,
        timing_scalar=float,
        rebuild_pair_weights=rebuild,
        write_net_weight_artifact=lambda params, payload: calls.append("artifact"),
    )
    optimizer = SimpleNamespace(
        rebase_objective_state=lambda **kwargs: calls.append("rebase")
        or {"status": "ok"}
    )
    return params, coordinator, context, optimizer, calls


def _begin_rebootstrap_event(coordinator, *, iteration, overflow):
    generation = coordinator.pin2pin_pair_generations.current_generation_id
    coordinator.observe_pin2pin_segment_iteration(
        overflow=overflow, iteration=iteration
    )
    coordinator.record_pin2pin_objective_evaluation(
        iteration=iteration, generation_id=generation
    )
    coordinator.record_pin2pin_placement_step(
        iteration=iteration, generation_id=generation
    )
    return coordinator.begin_pin2pin_milestone_after_step(
        iteration=iteration, overflow=overflow
    )


def test_segment_joint_rebootstrap_continues_with_a_second_real_route_b_action():
    params, coordinator, context, optimizer, calls = _rebootstrap_transaction_fixture()
    pos = torch.tensor([1.0])
    model = SimpleNamespace()
    first = _begin_rebootstrap_event(coordinator, iteration=5, overflow=0.29)
    first_record = apply_segment_rebootstrap_milestone_transaction(
        context,
        params=params,
        placedb=None,
        model=model,
        pos=pos,
        optimizer=optimizer,
        event=first,
    )
    second = _begin_rebootstrap_event(coordinator, iteration=6, overflow=0.24)
    second_record = apply_segment_rebootstrap_milestone_transaction(
        context,
        params=params,
        placedb=None,
        model=model,
        pos=pos,
        optimizer=optimizer,
        event=second,
    )

    assert first_record["status"] == second_record["status"] == "consumed"
    assert first_record["transition"]["accepted_action_count"] == 1
    assert second_record["transition"]["accepted_action_count"] == 1
    assert first_record["transition"]["pair_generation_after"] == 2
    assert second_record["transition"]["pair_generation_after"] == 3
    assert int(coordinator.buffering_lane.z_param.detach().sum().item()) == 2
    assert calls == ["refresh", "gradient", "rebuild", "rebase", "artifact"] * 2


def test_segment_joint_rebootstrap_rebuild_failure_is_terminal():
    params, coordinator, context, optimizer, calls = _rebootstrap_transaction_fixture(
        fail_rebuild=True
    )
    event = _begin_rebootstrap_event(coordinator, iteration=5, overflow=0.29)
    with pytest.raises(RuntimeError, match="fixture rebuild failure"):
        apply_segment_rebootstrap_milestone_transaction(
            context,
            params=params,
            placedb=None,
            model=SimpleNamespace(),
            pos=torch.tensor([1.0]),
            optimizer=optimizer,
            event=event,
        )

    assert calls == ["refresh", "gradient", "rebuild"]
    failure = coordinator.segment_milestones.summarize()["terminal_failure"]
    assert failure["status"] == "failed_terminal"
    assert failure["transaction_stage"] == "tdp_state_invalidated"


def test_overflow_milestone_schedule_rejects_route_b_deferral():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )

    with pytest.raises(
        RuntimeError,
        match="deferral requires fixed_window_compat",
    ):
        coordinator.defer_segment_route_b(event=event)


def test_segment_direct_joint_consumes_shared_z_grad_without_second_backward():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    pos_before = pos.detach().clone()
    optimizer = torch.optim.SGD([pos], lr=0.1)

    record = coordinator.apply_segment_route_b(event=event, optimizer=optimizer)

    assert record["transition"]["accepted_action_count"] == 1
    assert record["transition"]["selected_segment_ids"] == [100]
    assert torch.equal(lane.z_param.detach(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert lane.z_param not in optimizer.state
    assert torch.equal(pos.detach(), pos_before)
    assert coordinator.post_step_clamp(iteration=7)["z_integer_invariant"]


def test_segment_direct_joint_completes_atomic_milestone_transaction():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    for stage in (
        "sizing_gradient_frozen",
        "placement_updated",
        "sizing_applied",
        "runtime_state_refreshed",
        "buffering_gradient_ready",
    ):
        coordinator.advance_segment_milestone(
            event=event,
            stage=stage,
            evidence={"fixture": stage},
        )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    record = coordinator.complete_segment_milestone(
        event=event,
        sizing={
            "status": "applied",
            "num_changed_instances": 1,
            "frame": {"runtime_ms": 1.0},
            "runtime_ms": 2.0,
        },
        refresh={"status": "ok", "runtime_ms": 3.0},
        buffering_gradient={"status": "ready", "runtime_ms": 4.0},
    )

    assert record["status"] == "consumed"
    assert record["transition"]["route_b"]["selected_segment_ids"] == [100]
    assert record["transition"]["buffer_count_after"] == 1
    route_b = record["transition"]["route_b"]
    assert route_b["input_state_digest"].startswith("sha256:")
    assert route_b["output_state_digest"].startswith("sha256:")
    assert route_b["input_state_digest"] != route_b["output_state_digest"]
    runtime_ms = record["transition"]["runtime_ms"]
    assert runtime_ms["sizing_gradient"] == 1.0
    assert runtime_ms["sizing_select"] == 2.0
    assert runtime_ms["refresh"] == 3.0
    assert runtime_ms["buffer_timing_forward_backward"] == 4.0
    assert runtime_ms["route_b"] >= 0.0
    assert runtime_ms["total"] >= 10.0
    assert record["transaction_stage"] == "route_b_applied"


def test_segment_direct_joint_rejects_nonimproving_integer_transition():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    for stage in (
        "sizing_gradient_frozen",
        "placement_updated",
        "sizing_applied",
        "runtime_state_refreshed",
        "buffering_gradient_ready",
    ):
        coordinator.advance_segment_milestone(
            event=event,
            stage=stage,
            evidence={"fixture": stage},
        )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    class NonImprovingModel:
        def __init__(self):
            self.timing_obj_call_count = 0

        def timing_obj(self, pos):
            self.timing_obj_call_count += 1
            loss = lane.z_param[0].square()
            zero = torch.zeros_like(loss)
            return -loss, -loss, zero, zero

        @staticmethod
        def _timing_loss(wns, tns, ws, ts):
            return -tns

    model = NonImprovingModel()
    record = coordinator.complete_segment_milestone(
        event=event,
        sizing={"status": "applied"},
        refresh={"status": "ok"},
        buffering_gradient={"status": "ready", "timing_loss": 0.0},
        model=model,
        pos=torch.nn.Parameter(torch.tensor([1.0])),
    )

    route_b = record["transition"]["route_b"]
    assert route_b["gradient_selected_action_count"] == 1
    assert route_b["accepted_action_count"] == 0
    assert route_b["batch_acceptance"]["policy"] == (
        "monotonic_loss_prefix_backtracking"
    )
    assert route_b["batch_acceptance"]["trial_prefixes"] == [
        {
            "selected_action_count": 1,
            "loss": 1.0,
            "actual_improvement": -1.0,
            "accepted": False,
        }
    ]
    assert route_b["batch_acceptance"]["baseline_source"] == (
        "refreshed_buffer_gradient"
    )
    assert route_b["batch_acceptance"]["baseline_evaluation_count"] == 0
    assert route_b["batch_acceptance"]["trial_evaluation_count"] == 1
    assert route_b["batch_acceptance"]["restore_evaluation_count"] == 0
    assert route_b["batch_acceptance"]["restored_loss"] == 0.0
    assert route_b["batch_acceptance"]["restored_loss_evaluated"] is False
    assert route_b["batch_acceptance"]["restoration_state_exact"] is True
    assert model.timing_obj_call_count == 1
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))


def test_segment_direct_joint_falls_back_to_fresh_baseline_without_gradient_loss():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    for stage in (
        "sizing_gradient_frozen",
        "placement_updated",
        "sizing_applied",
        "runtime_state_refreshed",
        "buffering_gradient_ready",
    ):
        coordinator.advance_segment_milestone(
            event=event,
            stage=stage,
            evidence={"fixture": stage},
        )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    class NonImprovingModel:
        def __init__(self):
            self.timing_obj_call_count = 0

        def timing_obj(self, pos):
            self.timing_obj_call_count += 1
            loss = lane.z_param[0].square()
            zero = torch.zeros_like(loss)
            return -loss, -loss, zero, zero

        @staticmethod
        def _timing_loss(wns, tns, ws, ts):
            return -tns

    model = NonImprovingModel()
    record = coordinator.complete_segment_milestone(
        event=event,
        sizing={"status": "applied"},
        refresh={"status": "ok"},
        buffering_gradient={"status": "ready"},
        model=model,
        pos=torch.nn.Parameter(torch.tensor([1.0])),
    )

    acceptance = record["transition"]["route_b"]["batch_acceptance"]
    assert acceptance["baseline_source"] == "timing_obj_evaluation"
    assert acceptance["baseline_evaluation_count"] == 1
    assert acceptance["trial_evaluation_count"] == 1
    assert acceptance["restore_evaluation_count"] == 0
    assert model.timing_obj_call_count == 2
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))


def test_segment_direct_joint_rejects_optimizer_owned_z():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
    optimizer = torch.optim.SGD([lane.z_param], lr=0.1)

    with pytest.raises(RuntimeError, match="must not belong to the optimizer"):
        coordinator.apply_segment_route_b(event=event, optimizer=optimizer)


def test_segment_direct_joint_rejects_optimizer_state_for_z():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([pos], lr=0.1)
    optimizer.state[lane.z_param] = {"step": 1}

    with pytest.raises(RuntimeError, match="must not have optimizer state"):
        coordinator.apply_segment_route_b(event=event, optimizer=optimizer)


def test_segment_direct_joint_rejects_nonfinite_shared_z_gradient():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([float("inf"), -2.0, 1.0, 3.0])

    with pytest.raises(RuntimeError, match="contains non-finite values"):
        coordinator.apply_segment_route_b(event=event)


@pytest.mark.parametrize(
    ("value", "message"),
    ((0.5, "became fractional"), (-1.0, "left its legal range"), (4.0, "left its legal range")),
)
def test_segment_direct_joint_rejects_illegal_terminal_z(value, message):
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    with torch.no_grad():
        lane.z_param[0] = value

    with pytest.raises(RuntimeError, match=message):
        coordinator.post_step_clamp(iteration=7)


def test_segment_direct_joint_defers_route_b_until_window_boundary():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(
        _segment_joint_params("fixed_window_compat"),
        buffering_lane=lane,
    )
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    activation_record = coordinator.defer_segment_route_b(event=event)
    assert activation_record["transition"]["deferred_to_window_boundary"]
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))

    record = coordinator.apply_segment_route_b_at_window_boundary(iteration=20)

    assert record["status"] == "window_boundary"
    assert record["transition"]["accepted_action_count"] == 1
    assert record["transition"]["selected_segment_ids"] == [100]
    assert torch.equal(lane.z_param.detach(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert coordinator.apply_segment_route_b_at_window_boundary(iteration=21) == record


def test_segment_direct_joint_can_disable_route_b_for_placement_control():
    lane = FakeSegmentJointLane()
    params = _segment_joint_params("fixed_window_compat")
    params.joint_segment_route_b_enabled = 0
    coordinator = JointCoordinator(params, buffering_lane=lane)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    record = coordinator.apply_segment_route_b_at_window_boundary(iteration=20)

    assert record["status"] == "disabled"
    assert record["transition"]["accepted_action_count"] == 0
    assert record["transition"]["buffer_count_before"] == 0
    assert record["transition"]["buffer_count_after"] == 0
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))
    assert not coordinator.summarize()["segment_route_b_enabled"]


def test_segment_direct_joint_applies_multiple_boundaries_without_optimizer_restart():
    lane = FakeSegmentJointLane()
    params = _segment_joint_params("fixed_window_compat")
    params.joint_proximal_enabled = 1
    params.joint_proximal_lambda_z = 2.0
    proximal = JointProximalObjective(params)
    model = SimpleNamespace(
        joint_proximal_objective=proximal,
        data_collections=SimpleNamespace(buffer_segment_count_state=lane.state),
    )
    pos = torch.nn.Parameter(torch.tensor([1.0]))
    proximal.capture_anchors(model, pos, force=True)
    coordinator = JointCoordinator(params, buffering_lane=lane)
    coordinator.activate_segment_lane(
        model=model,
        data_collections=model.data_collections,
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    first = coordinator.apply_segment_virtual_window_boundary(
        model=model,
        pos=pos,
        optimizer=None,
        iteration=20,
        window_index=0,
        placement_steps=20,
        following_placement_steps=20,
        final=False,
    )
    lane.z_param.grad = torch.tensor([-3.0, -2.0, 1.0, 3.0])
    second = coordinator.apply_segment_virtual_window_boundary(
        model=model,
        pos=pos,
        optimizer=None,
        iteration=40,
        window_index=1,
        placement_steps=20,
        following_placement_steps=0,
        final=True,
    )

    assert first["accepted_action_count"] == 1
    assert first["next_anchor"]["status"] == "created"
    assert second["accepted_action_count"] == 1
    assert int(lane.z_param.detach().sum().item()) == 2
    windows = coordinator.summarize()["segment_virtual_windows"]
    assert [window["window_index"] for window in windows] == [0, 1]
    assert [window["actual_placement_steps"] for window in windows] == [20, 20]


def test_window_schedule_does_not_apply_an_extra_terminal_route_b_transition():
    class RecordingCoordinator:
        def __init__(self):
            self.calls = []

        def apply_segment_route_b_at_window_boundary(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "window_boundary"}

    coordinator = RecordingCoordinator()
    placer = SimpleNamespace(joint_coordinator=coordinator)

    result = NonLinearPlace._apply_terminal_segment_route_b_if_needed(
        placer,
        SimpleNamespace(joint_segment_virtual_window_steps=100),
        iteration=359,
        optimizer="nesterov",
    )

    assert result is None
    assert coordinator.calls == []


def test_pin2pin_rebootstrap_schedule_preserves_milestone_buffers_at_core_commit():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(
        _pin2pin_segment_joint_params(),
        buffering_lane=lane,
    )
    coordinator._segment_lane_prepared = True
    with torch.no_grad():
        lane.z_param.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = coordinator

    result = NonLinearPlace._apply_terminal_segment_route_b_if_needed(
        placer,
        SimpleNamespace(),
        iteration=20,
        optimizer=None,
    )

    assert result is None
    assert torch.equal(
        lane.z_param.detach(),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    assert coordinator.summarize()["terminal_segment_reselection"]["reason"] == (
        "preserve_milestone_route_b_state_for_core_commit"
    )


def test_overflow_schedule_reselects_terminal_route_b_from_zero():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    coordinator.segment_milestones.activation_iteration = 7
    coordinator.segment_milestones.trace = [
        {"status": "consumed"},
        {"status": "consumed"},
    ]
    with torch.no_grad():
        lane.z_param.copy_(torch.tensor([0.0, 1.0, 0.0, 1.0]))

    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    class FinalFrameModel:
        def timing_obj(self, pos):
            loss = torch.sum((lane.z_param - target) ** 2)
            zero = loss * 0.0
            return loss, zero, zero, zero

        @staticmethod
        def _timing_loss(wns, tns, ws, ts):
            return wns

    model = FinalFrameModel()
    observed_counts = []

    def compute_gradient(**kwargs):
        observed_counts.append(lane.z_param.detach().clone())
        wns, tns, ws, ts = model.timing_obj(kwargs["pos"])
        loss = model._timing_loss(wns, tns, ws, ts)
        lane.z_param.grad = torch.autograd.grad(loss, lane.z_param)[0]
        return {
            "status": "ready",
            "event_id": int(kwargs["event"]["event_id"]),
            "timing_loss": float(loss.detach().item()),
        }

    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = coordinator
    placer.last_global_place_model = model
    placer.pos = [torch.nn.Parameter(torch.tensor([1.0]))]
    placer._compute_refreshed_segment_buffer_gradient = compute_gradient
    rebuild_calls = []

    def rebuild_terminal_state(**kwargs):
        rebuild_calls.append(kwargs)
        with torch.no_grad():
            lane.z_param.zero_()
        return lane.state, {
            "status": "rebuilt",
            "topology_generation_before": 3,
            "topology_generation_after": 4,
        }

    placer._rebuild_terminal_segment_route_b_state = rebuild_terminal_state

    result = NonLinearPlace._apply_terminal_segment_route_b_if_needed(
        placer,
        SimpleNamespace(joint_segment_virtual_window_steps=0),
        iteration=20,
        optimizer=None,
    )

    assert torch.equal(observed_counts[0], torch.zeros(4))
    assert len(rebuild_calls) == 1
    assert result["topology_rebuild"]["topology_generation_after"] == 4
    assert result["historical_buffer_count"] == 2
    assert result["final_buffer_count"] == 2
    assert result["terminal_reason"] == "max_rounds"
    assert [
        row["transition"]["selected_segment_ids"] for row in result["rounds"]
    ] == [[100], [200]]
    assert torch.equal(lane.z_param.detach(), target)


def test_overflow_terminal_reselection_drops_history_without_positive_action():
    lane = FakeSegmentJointLane()
    coordinator = JointCoordinator(_segment_joint_params(), buffering_lane=lane)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    coordinator.segment_milestones.activation_iteration = 7
    coordinator.segment_milestones.trace = [{"status": "consumed"}]
    with torch.no_grad():
        lane.z_param.copy_(torch.ones(4))

    class ZeroTargetModel:
        def timing_obj(self, pos):
            loss = torch.sum(lane.z_param.square())
            zero = loss * 0.0
            return loss, zero, zero, zero

        @staticmethod
        def _timing_loss(wns, tns, ws, ts):
            return wns

    model = ZeroTargetModel()

    def compute_gradient(**kwargs):
        wns, tns, ws, ts = model.timing_obj(kwargs["pos"])
        loss = model._timing_loss(wns, tns, ws, ts)
        lane.z_param.grad = torch.autograd.grad(loss, lane.z_param)[0]
        return {"status": "ready", "timing_loss": float(loss.detach().item())}

    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = coordinator
    placer.last_global_place_model = model
    placer.pos = [torch.nn.Parameter(torch.tensor([1.0]))]
    placer._compute_refreshed_segment_buffer_gradient = compute_gradient
    rebuild_calls = []

    def rebuild_terminal_state(**kwargs):
        rebuild_calls.append(kwargs)
        with torch.no_grad():
            lane.z_param.zero_()
        return lane.state, {
            "status": "rebuilt",
            "topology_generation_before": 3,
            "topology_generation_after": 4,
        }

    placer._rebuild_terminal_segment_route_b_state = rebuild_terminal_state

    result = NonLinearPlace._apply_terminal_segment_route_b_if_needed(
        placer,
        SimpleNamespace(joint_segment_virtual_window_steps=0),
        iteration=20,
        optimizer=None,
    )

    assert result["historical_buffer_count"] == 4
    assert len(rebuild_calls) == 1
    assert result["final_buffer_count"] == 0
    assert result["terminal_reason"] == "no_positive_action"
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))


def test_segment_direct_joint_route_b_uses_finite_count_proximal_increment():
    lane = FakeSegmentJointLane()
    params = _segment_joint_params()
    params.joint_proximal_enabled = 1
    params.joint_proximal_lambda_z = 2.0
    proximal = JointProximalObjective(params)
    proximal.anchors["z_param"] = torch.zeros(4)
    model = SimpleNamespace(joint_proximal_objective=proximal)
    coordinator = JointCoordinator(params, buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model=model,
        data_collections=SimpleNamespace(),
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])

    record = coordinator.apply_segment_route_b(event=event)

    transition = record["transition"]
    assert transition["proximal_adjustment"]["finite_increment_applied"]
    assert transition["proximal_adjustment"]["correction_min"] == 0.25
    assert transition["proximal_adjustment"]["correction_max"] == 0.25
    assert transition["action_grad_norm"] != transition["raw_grad_norm"]
    assert transition["selected_segment_ids"] == [100]
    assert torch.equal(lane.z_param.detach(), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_segment_direct_joint_calibrates_selected_action_without_second_backward(
    tmp_path,
):
    lane = FakeSegmentJointLane()
    params = _segment_joint_params()
    params.result_dir = str(tmp_path)
    params.design_name = lambda: "fixture"
    coordinator = JointCoordinator(params, buffering_lane=lane)
    event = coordinator.observe_segment_milestone(overflow=0.29, iteration=7)
    coordinator.activate_segment_lane(
        model="model",
        data_collections="data",
        topology_generation=3,
    )
    lane.z_param.grad = torch.tensor([-4.0, -2.0, 1.0, 3.0])
    grad_before = lane.z_param.grad.clone()

    class FakeModel:
        def __init__(self):
            self.grad_enabled = []

        def timing_obj(self, pos):
            self.grad_enabled.append(torch.is_grad_enabled())
            loss = 10.0 + torch.dot(
                lane.z_param,
                torch.tensor([-4.0, -2.0, 1.0, 3.0]),
            )
            zero = loss * 0.0
            return zero, loss, zero, zero

        @staticmethod
        def _timing_loss(wns, tns, ws, ts):
            return tns

    model = FakeModel()
    pos = torch.nn.Parameter(torch.tensor([1.0]))

    result = coordinator.calibrate_segment_actions(
        model=model,
        pos=pos,
        event=event,
    )

    assert result["sample_count"] == 1
    assert result["baseline_loss"] == 10.0
    assert result["restored_loss"] == 10.0
    assert result["actions"][0]["segment_id"] == 100
    assert result["actions"][0]["predicted_improvement"] == 4.0
    assert result["actions"][0]["explicit_improvement"] == 4.0
    assert result["sign_agreement_ratio"] == 1.0
    assert result["rank_correlation"] is None
    assert result["no_second_backward"]
    assert all(enabled is False for enabled in model.grad_enabled)
    assert torch.equal(lane.z_param.detach(), torch.zeros(4))
    assert torch.equal(lane.z_param.grad, grad_before)
    assert (tmp_path / "fixture_segment_joint_action_calibration.json").is_file()

    record = coordinator.apply_segment_route_b(event=event)
    assert record["transition"]["selected_segment_ids"] == [100]


def test_segment_direct_joint_best_restore_does_not_rewind_discrete_state():
    state = SimpleNamespace(z_param=torch.nn.Parameter(torch.ones(3)))
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = SimpleNamespace(is_segment_direct_joint=True)
    size_logits = torch.nn.Parameter(torch.tensor([2.0]))
    inst_cell_id = torch.tensor([7], dtype=torch.long)
    inst_libcell_offset = torch.tensor([3], dtype=torch.long)
    placer.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        size_logits=size_logits,
        vt_logits=None,
        inst_cell_id=inst_cell_id,
        inst_libcell_offset=inst_libcell_offset,
    )
    placer.placedb = SimpleNamespace(
        inst_cell_id=np.array([7], dtype=np.int32),
        inst_libcell_offset=np.array([3], dtype=np.int32),
    )
    early_stop_state = {
        "enabled": True,
        "restore_best": True,
        "best_iteration": 0,
        "best_loss": 1.0,
        "best_state": {
            "size_logits": torch.tensor([0.0]),
            "inst_cell_id": torch.tensor([1], dtype=torch.long),
            "inst_libcell_offset": torch.tensor([0], dtype=torch.long),
            "placedb_inst_cell_id": np.array([1], dtype=np.int32),
            "placedb_inst_libcell_offset": np.array([0], dtype=np.int32),
            "buffer_z_param": torch.zeros(3),
        },
    }

    summary = placer._restore_continuous_early_stop_best_state(early_stop_state)

    assert torch.equal(state.z_param.detach(), torch.ones(3))
    assert torch.equal(size_logits.detach(), torch.tensor([2.0]))
    assert torch.equal(inst_cell_id, torch.tensor([7]))
    assert torch.equal(inst_libcell_offset, torch.tensor([3]))
    assert np.array_equal(placer.placedb.inst_cell_id, np.array([7]))
    assert np.array_equal(placer.placedb.inst_libcell_offset, np.array([3]))
    assert not summary["sizing_restored"]
    assert summary["sizing_restore_reason"] == (
        "milestone_owned_discrete_sizing_state"
    )
    assert not summary["buffer_restored"]
    assert summary["buffer_restore_reason"] == "route_b_owned_discrete_state"


def test_nesterov_accepts_placement_only_joint_parameter_groups():
    NonLinearPlace._validate_optimizer_parameter_groups(
        "nesterov",
        [{"params": [torch.nn.Parameter(torch.ones(1))], "group_name": "placement"}],
    )

    with pytest.raises(ValueError, match="found groups=\\['sizing'\\]"):
        NonLinearPlace._validate_optimizer_parameter_groups(
            "nesterov",
            [
                {"params": [torch.nn.Parameter(torch.ones(1))], "group_name": "placement"},
                {"params": [torch.nn.Parameter(torch.ones(1))], "group_name": "sizing"},
            ],
        )
