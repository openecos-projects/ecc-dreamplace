"""Shared native FLUTE refresh precedes both objective consumers."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.joint_iteration import JointStepState
from dreamplace.flows.timing_iteration import TimingIterationContext
from dreamplace.flows.timing_objective_policy import diff_tdp_gate_status
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo

from dreamplace.flows import l_shape_iteration, placement_topology_iteration


@pytest.mark.parametrize("lane", ["timing", "route", "combined", "joint_frozen"])
def test_refresh_once_and_both_consumers_follow_live_geometry(monkeypatch, lane):
    topo = SteinerTopo(
        torch.arange(3, dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        deterministic_flag=True,
    )
    pos = torch.tensor([0.0, 1.0, 2.0, 0.0, 2.0, 0.0], requires_grad=True)
    topo.rebuild_tree(pos.detach())
    topo.edge_l_directions = torch.zeros_like(topo.flat_pin_from)
    events = []
    topo.l_direction_resolver = SimpleNamespace(
        resolve_l_directions=lambda tree: events.append("old_feedback")
        or torch.zeros_like(tree.flat_pin_from),
    )
    joint = JointStepState()
    if lane == "joint_frozen":
        topo.freeze_topology()
        joint.topology_prepared = True
    initial_generation = topo.topology_generation
    data = SimpleNamespace()
    ops = SimpleNamespace(steiner_topo_op=topo, pin_pos_op=lambda p: p)
    route_context = l_shape_iteration.LShapeIterationContext(data, ops)
    params = SimpleNamespace(
        routability_opt_flag=lane != "timing",
        l_shape_routability_flag=1,
        placement_sizing_mode="place_only",
        with_sta=True,
        differentiable_timing_obj=lane != "route",
        diff_timing_driven_placement=True,
        timing_topology_refresh_interval=15,
        timing_topology_enable_overflow_threshold=0.35,
        l_shape_update_interval=30,
        l_direction_use_gpugr=True,
        l_shape_use_ggr_topology=False,
        soft_l_assignment=False,
        l_shape_overflow_update_flag=False,
        l_shape_plot_flag=False,
    )
    model = SimpleNamespace(
        use_l_shape_routability=True,
        enable_l_shape_routability=True,
        l_shape_routability_op=None,
        use_timing_obj=lane != "route",
    )
    metric = SimpleNamespace(overflow=torch.tensor([0.1]))
    policy = SimpleNamespace(maybe_reenable=lambda *args: None)
    monkeypatch.setattr(
        l_shape_iteration,
        "_prepare_l_shape_inputs_from_gpugr",
        lambda *args, **kwargs: events.append("route_feedback")
        or {"route_entries": [], "metrics": {}, "route_xsize": 4, "route_ysize": 4},
    )

    def resolve(*args, **kwargs):
        events.append("new_feedback")
        topo.edge_l_directions = torch.ones_like(topo.flat_pin_from)
        return topo.edge_l_directions

    monkeypatch.setattr(l_shape_iteration, "_resolve_l_directions_for_l_shape", resolve)

    def refresh(position):
        events.append("timing_refresh")
        topo.rebuild_tree(position.detach())

    timing_context = TimingIterationContext(
        enabled=lambda p: p.differentiable_timing_obj,
        uses_net_weight=lambda p: False,
        project_weights=None,
        refresh_topology=refresh,
        carrier=lambda p: "direct_loss",
        reset_weights=lambda *args, **kwargs: None,
    )
    for iteration in (30, 31):
        gate = diff_tdp_gate_status(params, iteration, 0.1)
        placement_topology_iteration.update_before_objective(
            timing_context,
            route_context,
            params,
            None,
            model,
            pos,
            iteration,
            metric,
            policy,
            gate,
            joint if iteration == 30 else JointStepState(),
            lambda *args, **kwargs: events.append("publish"),
        )
        # Geometry and gradients remain live even on a step without a rebuild.
        x, y = topo(pos)
        torch.testing.assert_close(x, pos[:3][topo.pin_relate_x.long()])
        torch.testing.assert_close(y, pos[3:][topo.pin_relate_y.long()])
        gradient = torch.autograd.grad((x.square() + y.square()).sum(), pos)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        with torch.no_grad():
            pos.add_(0.1)
    assert topo.topology_generation - initial_generation == (lane != "joint_frozen")
    assert events == (
        ["timing_refresh", "old_feedback"]
        if lane == "timing"
        else ["route_feedback", "new_feedback", "publish"]
    )


def test_timing_gate_refreshes_before_objective_without_a_hardcoded_ten_step():
    events = []
    params = SimpleNamespace(
        placement_sizing_mode="place_only",
        with_sta=True,
        differentiable_timing_obj=True,
        diff_timing_driven_placement=True,
        timing_topology_refresh_interval=15,
        timing_topology_enable_overflow_threshold=0.35,
        routability_opt_flag=False,
    )
    context = TimingIterationContext(
        enabled=lambda p: True,
        uses_net_weight=lambda p: False,
        project_weights=None,
        refresh_topology=lambda pos: events.append("refresh"),
        carrier=lambda p: "direct_loss",
        reset_weights=lambda *args, **kwargs: None,
    )
    model = SimpleNamespace(use_timing_obj=True)
    for iteration, overflow in [(100, 0.1), (105, 0.1), (120, 0.5)]:
        gate = diff_tdp_gate_status(params, iteration, overflow)
        placement_topology_iteration.update_before_objective(
            context,
            None,
            params,
            None,
            model,
            None,
            iteration,
            None,
            None,
            gate,
            JointStepState(),
            None,
        )
        events.append("objective")
    assert events == ["objective", "refresh", "objective", "objective"]
