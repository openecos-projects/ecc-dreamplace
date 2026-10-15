"""Metric publication at the joint activation seam."""
from types import SimpleNamespace

import torch

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.flows.joint_iteration import JointStepState, prepare_step, after_step
from dreamplace.flows.joint_coordinator import JointCoordinator
from dreamplace.NesterovAcceleratedGradientOptimizer import NesterovAcceleratedGradientOptimizer


def test_bootstrap_publishes_timing_and_violation_metrics():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.op_collections = SimpleNamespace(timing_propagation_op=SimpleNamespace(
        last_total_slew_violation=3.0, last_total_cap_violation=4.0,
        last_total_leakage=5.0,
    ))
    # These two calls are the external topology/STA resource boundary.
    placer._prepare_segment_direct_joint_iteration = lambda **kwargs: {
        "mode": "pin2pin_rebootstrap", "observation": {"activation_due": True},
    }
    placer._rebuild_and_install_segment_pin2pin = lambda **kwargs: {
        "timing": (-1.0, -2.0, 0.5, 0.0),
    }
    coordinator = SimpleNamespace(is_segment_direct_joint=True)
    metric = SimpleNamespace()
    step = JointStepState()
    prepare_step(
        placer._joint_iteration_context(), coordinator, SimpleNamespace(), None,
        SimpleNamespace(overflow=torch.tensor([0.2])), torch.ones(2), None,
        "adam", 4, metric, step, lambda: 0.0, lambda *args: None,
    )
    assert vars(metric) == {
        "wns": -1.0, "tns": -2.0, "timing_objective": 0.5,
        "slew_violation": 3.0, "cap_violation": 4.0, "leakage": 5.0,
    }
    assert step == JointStepState(
        prepared={"mode": "pin2pin_rebootstrap", "observation": {"activation_due": True}},
        topology_prepared=True, pin2pin_update={"timing": (-1.0, -2.0, 0.5, 0.0)},
    )


def test_real_nesterov_repeated_evaluations_consume_one_pair_generation():
    params = SimpleNamespace(
        joint_quality_profile="segment_count_direct_joint_v1",
        joint_segment_action_schedule_mode="overflow_milestones",
        enable_net_weighting=1, net_weighting_scheme="pin2pin",
        net_weighting_update_interval=2,
    )
    coordinator = JointCoordinator(params)
    coordinator.install_pin2pin_generation(
        iteration=4, reason="activation_bootstrap", status="timing_violating", pair_count=1,
    )
    position = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    calls = []

    def objective(pos):
        calls.append(pos.detach().clone())
        loss = pos.square().sum()
        gradient = torch.autograd.grad(loss, pos)[0]
        return loss, gradient

    optimizer = NesterovAcceleratedGradientOptimizer(
        [position], lr=0.1, obj_and_grad_fn=objective,
        constraint_fn=lambda pos: None, use_bb=False,
    )
    position.grad = torch.zeros_like(position)
    before = optimizer.param_groups[0]["obj_eval_count"]
    optimizer.step()
    after = optimizer.param_groups[0]["obj_eval_count"]
    assert len(calls) == 3 and after - before == 1
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.op_collections = SimpleNamespace(timing_propagation_op=None)
    placer._maybe_apply_segment_virtual_window_boundary = lambda **kwargs: None
    step = JointStepState(consumed_generation=1, eval_count_before=before)
    pending = after_step(
        placer._joint_iteration_context(), coordinator, params, None,
        SimpleNamespace(overflow=torch.tensor([0.4])), position, optimizer,
        "nesterov", 4, SimpleNamespace(), step,
    )
    assert pending is None and step.objective_recorded
    assert coordinator.summary["last_pin2pin_nesterov_consumption"] == {
        "iteration": 4, "pair_generation": 1,
        "optimizer_eval_count_before": before, "optimizer_eval_count_after": after,
        "optimizer_eval_count_delta": 1,
    }
    generation = coordinator.pin2pin_pair_generations.current_generation
    assert (generation["objective_eval_count"], generation["consumed_step_count"]) == (1, 1)
