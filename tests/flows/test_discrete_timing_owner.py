"""Electrical action ownership and GP state transitions."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows import discrete_timing_owner as discrete_timing_owner_module
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.ops.buffer_insertion import discrete_candidate_buffer_runner
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane


def test_geometry_publish_creates_baseline_sizes_without_routability(monkeypatch):
    data = SimpleNamespace(
        node_size_x=torch.tensor([2.0, 3.0]),
        node_size_y=torch.tensor([4.0, 5.0]),
        inst_cell_id=torch.tensor([0, 1]),
        flat_libcell_info=(),
        flat_libcell_leakage=(),
    )
    placedb = SimpleNamespace()
    monkeypatch.setattr(
        discrete_timing_owner_module,
        "build_quad_candidate_table",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        discrete_timing_owner_module,
        "capture_area",
        lambda *_args, **_kwargs: SimpleNamespace(capacity=100.0),
    )

    DiscreteTimingOwner(
        SimpleNamespace(),
        placedb,
        data,
        SimpleNamespace(),
        lambda *_args, **_kwargs: None,
    )

    torch.testing.assert_close(data.original_node_size_x, data.node_size_x)
    torch.testing.assert_close(data.original_node_size_y, data.node_size_y)
    assert placedb.cooptimization_geometry is data


@pytest.mark.parametrize("requires_grad", [False, True])
def test_candidate_window_uses_the_models_combined_loss(monkeypatch, requires_grad):
    activation = torch.nn.Parameter(torch.zeros(1), requires_grad=requires_grad)
    state = SimpleNamespace(activation_param=activation)
    lane = BufferingOptimizationLane.__new__(BufferingOptimizationLane)
    lane.timing_loss = None
    lane.segment_capacity_controller = None
    lane.config = SimpleNamespace(continuous_steps=1)
    lane._current_state = lambda: state
    model = SimpleNamespace(data_collections=SimpleNamespace(pos=[torch.zeros(2)]))

    def timing_obj(_pos):
        wns = -2.0 + activation.sum()
        tns = -4.0 - 3.0 * activation.sum()
        return wns, tns, wns, tns

    def timing_loss(wns, tns, _ws, _ts):
        cap = 2.0 * activation.sum()
        slew = 3.0 * activation.sum()
        return -500.0 * wns - 5.0 * tns + cap + slew

    model.timing_obj, model._timing_loss = timing_obj, timing_loss
    observed = {}

    def scheduler(*, lane, model, rounds, allow_existing_activation):
        loss, metrics = lane.objective(model)
        observed["gradient"] = torch.autograd.grad(loss, activation)[0].tolist()
        observed["loss"] = float(loss.detach())
        snapshot = {"loss": float(loss.detach())}
        return SimpleNamespace(
            terminal_metrics=snapshot,
            best_restore={"initial_snapshot": snapshot},
            iterations=rounds,
            trace_path="candidate.json",
            trace_sha256="",
        )

    monkeypatch.setattr(
        discrete_candidate_buffer_runner,
        "run_discrete_candidate_buffer_scheduler",
        scheduler,
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.run_buffering(model, torch.zeros(2), capacity=100.0)

    assert observed == {"gradient": [-480.0], "loss": 1020.0}
    assert activation.requires_grad == requires_grad


def test_retained_candidate_freezes_actions_and_keeps_gp_position_gradients():
    position = torch.nn.Parameter(torch.tensor([2.0]))
    activation = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    activation.grad = torch.ones_like(activation)
    state = SimpleNamespace(activation_param=activation, candidate_net_id=torch.tensor([7]))
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: None,
        attach_state=lambda data: None,
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace(pos=[position])
    owner.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            freeze_nets=lambda net_ids: None,
            frozen_nets=SimpleNamespace(candidate_state=None),
        )
    )
    owner.refresh_topology = lambda pos: None
    model = SimpleNamespace(_virtual_cell_density_view=object(), _timing_geometry_cache=object())

    owner.retain_buffered_nets(model)
    (position.square().sum() * (1 + activation.sum())).backward()

    assert {
        "activation_requires_grad": activation.requires_grad,
        "activation_gradient": activation.grad,
        "position_gradient": position.grad.tolist(),
        "retained_state": owner.ops.steiner_topo_op.frozen_nets.candidate_state is state,
        "timing_enabled": owner.params.enable_relaxed_buffer_timing,
        "density_cache": model._virtual_cell_density_view,
        "geometry_cache": model._timing_geometry_cache,
    } == {
        "activation_requires_grad": False,
        "activation_gradient": None,
        "position_gradient": [8.0],
        "retained_state": True,
        "timing_enabled": True,
        "density_cache": None,
        "geometry_cache": None,
    }
