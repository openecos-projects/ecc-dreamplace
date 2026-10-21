"""Electrical action ownership and GP state transitions."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows import discrete_timing_owner as discrete_timing_owner_module
from dreamplace.flows.candidate_refresh import (
    collect_owner_candidate_dirty_nets,
)
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


def test_prepare_skips_topology_recompute_for_frozen_gr_snapshot():
    state = SimpleNamespace(activation_param=torch.zeros(1))
    refresh_calls = []
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **_kwargs: {
            "status": "ok",
            "reason": "frozen_gr_snapshot",
        },
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace()
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace(gr_sizing=object())
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            frozen_nets=SimpleNamespace(candidate_state=None)
        )
    )
    owner.refresh_topology = lambda _pos: refresh_calls.append(True)
    model = SimpleNamespace(
        _virtual_cell_density_view=object(),
    )

    owner.prepare(model, pos=torch.zeros(2), active_only=True)

    assert refresh_calls == []
    assert owner.data.buffer_retained_candidate_state["status"] == "ok"
    assert owner.ops.steiner_topo_op.frozen_nets.candidate_state is state
    assert model._virtual_cell_density_view is None


def test_prepare_reuses_topology_for_unchanged_geometry():
    state = SimpleNamespace(activation_param=torch.zeros(1))
    refresh_calls = []
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **_kwargs: {
            "status": "ok",
            "reason": "unchanged_geometry",
        },
    )
    topo = SimpleNamespace(
        topology_generation=1,
        frozen_topology_generation=3,
        frozen_nets=SimpleNamespace(candidate_state=None, net_ids=(7,)),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace(runtime_cell_state_generation=0)
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace(gr_sizing=None)
    owner.ops = SimpleNamespace(steiner_topo_op=topo)
    owner.refresh_topology = lambda _pos: refresh_calls.append(True)
    model = SimpleNamespace(_virtual_cell_density_view=object())
    pos = torch.tensor([1.0, 2.0])

    owner.prepare(model, pos=pos, active_only=True)
    owner.prepare(model, pos=pos, active_only=True)

    assert refresh_calls == [True]
    assert owner.topology_refresh_count == 1
    assert owner.topology_refresh_reuse_count == 1
    assert owner.last_topology_refresh_state["status"] == "reused"


def test_candidate_live_geometry_ignores_position_only_changes():
    state = SimpleNamespace(activation_param=torch.zeros(1))
    refresh_calls = []
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **_kwargs: {
            "status": "ok",
            "reason": "geometry_changed",
        },
    )
    topo = SimpleNamespace(
        topology_generation=1,
        frozen_topology_generation=3,
        frozen_nets=SimpleNamespace(candidate_state=None, net_ids=(7,)),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace(runtime_cell_state_generation=0)
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace(gr_sizing=None)
    owner.ops = SimpleNamespace(steiner_topo_op=topo)
    owner.refresh_topology = lambda _pos: refresh_calls.append(True)
    model = SimpleNamespace(_virtual_cell_density_view=object())
    pos = torch.tensor([1.0, 2.0])

    owner.prepare(model, pos=pos, active_only=True)
    owner.data.runtime_cell_state_generation += 1
    owner.prepare(model, pos=pos, active_only=True)

    assert refresh_calls == [True]
    assert owner.topology_refresh_count == 1
    assert owner.topology_refresh_reuse_count == 1
    assert owner.last_topology_refresh_state["status"] == "reused"


def test_prepare_refreshes_topology_after_structural_generation_change():
    state = SimpleNamespace(activation_param=torch.zeros(1))
    refresh_calls = []
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **_kwargs: {
            "status": "ok",
            "reason": "structural_geometry_change",
        },
    )
    topo = SimpleNamespace(
        topology_generation=1,
        frozen_topology_generation=3,
        frozen_nets=SimpleNamespace(candidate_state=None, net_ids=(7,)),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace(runtime_cell_state_generation=0)
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace(gr_sizing=None)
    owner.ops = SimpleNamespace(steiner_topo_op=topo)
    owner.refresh_topology = lambda _pos: refresh_calls.append(True)
    model = SimpleNamespace(_virtual_cell_density_view=object())
    pos = torch.tensor([1.0, 2.0])

    owner.prepare(model, pos=pos, active_only=True)
    topo.topology_generation += 1
    owner.prepare(model, pos=pos, active_only=True)

    assert refresh_calls == [True, True]
    assert owner.topology_refresh_count == 2
    assert owner.topology_refresh_reuse_count == 0
    assert owner.last_topology_refresh_state["status"] == "refreshed"


def test_sizing_actions_defer_candidate_refresh(monkeypatch):
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace()
    owner.data = SimpleNamespace()
    owner.ops = SimpleNamespace(timing_propagation_op=object())
    owner.lane = None
    owner.refresh_candidates_after_sizing = True
    owner._pending_dirty_instance_ids = set()
    owner._pending_dirty_net_ids = set()
    monkeypatch.setattr(
        discrete_timing_owner_module,
        "sync_runtime_cells",
        lambda *_args, **_kwargs: {"runtime_cell_state_synced": True},
    )

    result = owner._sync(
        SimpleNamespace(),
        {
            "applied_instance_ids": [3, 3, 8],
            "applied_cell_ids": [4, 4, 9],
        },
        SimpleNamespace(pos=torch.zeros(2)),
    )

    assert owner._pending_dirty_instance_ids == {3, 8}
    assert result["candidate_refresh_deferred"] is True
    assert result["pending_dirty_instance_count"] == 2


def test_pending_sizing_and_buffering_changes_collect_candidate_dirty_nets():
    state = SimpleNamespace(
        activation_param=torch.zeros(1),
        candidate_net_id=torch.tensor([1, 3, 4]),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = SimpleNamespace(
        _current_state=lambda: state,
        buffer_relaxed_timing_payload={
            "candidate_net_id": torch.tensor([1, 3, 4]),
        },
    )
    owner.placedb = SimpleNamespace(
        flat_node2pin_map=torch.tensor([0, 1, 2, 3]),
        flat_node2pin_start_map=torch.tensor([0, 0, 1, 2]),
        pin2net_map=torch.tensor([1, 3, 4, 5]),
    )
    owner.data = SimpleNamespace()
    owner._pending_dirty_instance_ids = {1}
    owner._pending_dirty_net_ids = {4}

    dirty = collect_owner_candidate_dirty_nets(owner)

    assert dirty == [1, 4]
    assert owner.last_candidate_dirty_net_summary["status"] == "ok"
    assert owner.last_candidate_dirty_net_summary["counts"][
        "dirty_net_count"
    ] == 2


def test_prepare_can_force_final_all_candidate_geometry_refresh():
    state = SimpleNamespace(activation_param=torch.zeros(1))
    observed = {}
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: (
            observed.update(kwargs)
            or {"status": "ok", "reason": "final_geometry"}
        ),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace()
    owner.params = SimpleNamespace()
    owner.placedb = SimpleNamespace(gr_sizing=object())
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            frozen_nets=SimpleNamespace(candidate_state=None),
        )
    )
    owner.refresh_topology = lambda _pos: None
    owner._pending_dirty_instance_ids = set()
    owner._pending_dirty_net_ids = set()
    model = SimpleNamespace(_virtual_cell_density_view=object())

    owner.prepare(
        model,
        pos=torch.zeros(2),
        refresh_records_all=True,
    )

    assert observed["refresh_records_all"] is True


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
        with torch.no_grad():
            activation.fill_(1.0)
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


def test_retained_candidate_keeps_a_frozen_diff_opt_topology_unchanged():
    activation = torch.nn.Parameter(
        torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    )
    state = SimpleNamespace(
        activation_param=activation,
        candidate_net_id=torch.tensor([7, 8]),
    )
    refresh_calls = []
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: refresh_calls.append(kwargs),
        attach_state=lambda _data: None,
    )
    frozen = SimpleNamespace(candidate_state=None)
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.lane = lane
    owner.data = SimpleNamespace(pos=[torch.tensor([2.0, 3.0])])
    owner.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            topology_frozen=True,
            freeze_nets=lambda _net_ids: pytest.fail(
                "frozen diff-opt topology must not be refrozen"
            ),
            frozen_nets=frozen,
        )
    )
    owner.refresh_topology = lambda _pos: pytest.fail(
        "frozen diff-opt topology must not be refreshed"
    )
    owner._pending_dirty_instance_ids = {2}
    owner._pending_dirty_net_ids = {8}
    owner._candidate_refresh_in_progress = False
    model = SimpleNamespace(
        _timing_geometry_cache=object(),
        _virtual_cell_density_view=object(),
    )

    owner.retain_buffered_nets(model)

    assert refresh_calls == []
    assert frozen.candidate_state is None
    assert state.timing_active_only is True
    assert activation.requires_grad is False
    assert activation.grad is None
    assert owner.params.enable_relaxed_buffer_timing is True
    assert owner._pending_dirty_instance_ids == set()
    assert owner._pending_dirty_net_ids == set()
