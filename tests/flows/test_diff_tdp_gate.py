"""Diff-TDP activation and carrier contracts at the timing flow boundary."""
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.flows import timing_gradient_projection
from dreamplace.flows import timing_topology


def _placer():
    return NonLinearPlace.__new__(NonLinearPlace)


def _params(**kwargs):
    defaults = {
        "placement_sizing_mode": "joint",
        "with_sta": 1,
        "differentiable_timing_obj": 1,
        "diff_timing_driven_placement": 1,
        "timing_topology_refresh_interval": 10,
        "timing_topology_enable_overflow_threshold": 0.2,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_joint_direct_loss_uses_only_the_frozen_step_gate_state():
    placer = _placer()
    params = _params()

    assert not placer._should_use_live_timing_loss(params)

    placer._diff_tdp_step_status = {
        "timing_objective_active": True,
        "topology_refresh_due": False,
    }
    assert placer._should_use_live_timing_loss(params)

    params.timing_placement_carrier = "gradient_net_weight"
    assert not placer._should_use_live_timing_loss(params)


def test_direct_tdp_refresh_is_owned_by_the_step_gate():
    placer = _placer()
    params = _params()
    placer._diff_tdp_step_status = {"timing_objective_active": True}

    assert placer._should_use_live_timing_loss(params)
    assert not placer._should_refresh_live_timing_before_objective(params)

    params.placement_sizing_mode = "size_only"
    params.diff_timing_driven_placement = 0
    assert placer._should_refresh_live_timing_before_objective(params)


def test_size_only_timing_objective_is_not_controlled_by_tdp_warmup_gate():
    placer = _placer()
    params = _params(placement_sizing_mode="size_only")

    assert not placer._diff_tdp_enabled(params)
    assert not placer._diff_tdp_controls_direct_timing_objective(params)

    placer._diff_tdp_step_status = {"timing_objective_active": True}

    class FakeModel:
        def initialize_timing_grad_balance(self, *args, **kwargs):
            raise AssertionError("size_only must not initialize placement balance")

    assert (
        placer._initialize_timing_grad_balance_if_needed(
            params,
            FakeModel(),
            torch.tensor([0.0], requires_grad=True),
            iteration=1,
            overflow=0.01,
        )
        is None
    )


def test_diff_tdp_gate_blocks_before_overflow_threshold():
    status = _placer()._diff_tdp_gate_status(
        _params(),
        iteration=10,
        overflow=torch.tensor(0.25),
    )

    assert not status["enabled"]
    assert status["reason"] == "overflow_gate"
    assert status["threshold"] == 0.2


def test_diff_tdp_gate_skips_initial_warmup_iteration():
    status = _placer()._diff_tdp_gate_status(
        _params(),
        iteration=0,
        overflow=torch.tensor(0.01),
    )

    assert not status["enabled"]
    assert status["reason"] == "warmup_iteration"


def test_diff_tdp_gate_enables_below_overflow_threshold_on_interval():
    status = _placer()._diff_tdp_gate_status(
        _params(),
        iteration=20,
        overflow=torch.tensor(0.19),
    )

    assert status["enabled"]
    assert status["reason"] == "ok"
    assert status["overflow"] == pytest.approx(0.19)


def test_diff_tdp_gate_respects_refresh_interval():
    status = _placer()._diff_tdp_gate_status(
        _params(),
        iteration=21,
        overflow=torch.tensor(0.19),
    )

    assert not status["enabled"]
    assert status["reason"] == "interval"
    assert status["timing_objective_active"]
    assert not status["topology_refresh_due"]


def test_diff_tdp_gate_separates_timing_activation_from_disabled_refresh():
    status = _placer()._diff_tdp_gate_status(
        _params(timing_topology_refresh_interval=0),
        iteration=21,
        overflow=torch.tensor(0.19),
    )

    assert status["timing_objective_active"]
    assert not status["topology_refresh_due"]
    assert status["reason"] == "disabled_interval"


def test_diff_tdp_gate_summary_records_active_steps_and_refreshes():
    placer = _placer()
    params = _params()
    placer._reset_diff_tdp_gate_summary(params)

    placer._record_diff_tdp_gate_status(
        params,
        placer._diff_tdp_gate_status(params, 19, torch.tensor(0.19)),
    )
    placer._record_diff_tdp_gate_status(
        params,
        placer._diff_tdp_gate_status(params, 20, torch.tensor(0.18)),
    )

    summary = placer._diff_tdp_gate_summary
    assert summary["timing_active_step_count"] == 2
    assert summary["timing_inactive_step_count"] == 0
    assert summary["topology_refresh_count"] == 1
    assert summary["first_timing_active_iteration"] == 19
    assert summary["first_timing_active_overflow"] == pytest.approx(0.19)


def test_diff_tdp_step_freeze_initializes_topology_on_first_active_step():
    placer = _placer()
    placer._live_timing_topology_initialized = True
    params = _params()
    placer._reset_diff_tdp_gate_summary(params)

    status = placer._freeze_diff_tdp_step_status(
        params,
        iteration=21,
        overflow=torch.tensor(0.19),
    )

    assert status["timing_objective_active"]
    assert status["topology_refresh_due"]
    assert status["reason"] == "activation_topology"
    assert placer._diff_tdp_step_status == status
    assert placer._diff_tdp_gate_summary["topology_refresh_count"] == 1


def test_timing_grad_balance_initializes_once_and_restores_new_model():
    class FakeModel:
        def __init__(self):
            self.initialize_count = 0
            self.applied = []

        def initialize_timing_grad_balance(self, pos, *, iteration, overflow):
            self.initialize_count += 1
            return {
                "status": "initialized",
                "initialized": True,
                "weight_applied": 0.25,
                "iteration": iteration,
                "overflow": overflow,
            }

        def apply_timing_grad_balance_state(self, state):
            self.applied.append(dict(state))

    placer = _placer()
    placer._timing_grad_balance_state = None
    placer._diff_tdp_step_status = {"timing_objective_active": True}
    params = _params(timing_grad_balance_target_ratio=0.1)
    first_model = FakeModel()

    first = placer._initialize_timing_grad_balance_if_needed(
        params,
        first_model,
        torch.tensor([0.0], requires_grad=True),
        iteration=20,
        overflow=0.19,
    )

    assert first_model.initialize_count == 1
    assert first["weight_applied"] == 0.25

    rebuilt_model = FakeModel()
    second = placer._initialize_timing_grad_balance_if_needed(
        params,
        rebuilt_model,
        torch.tensor([0.0], requires_grad=True),
        iteration=21,
        overflow=0.18,
    )

    assert rebuilt_model.initialize_count == 0
    assert rebuilt_model.applied == [first]
    assert second == first


@pytest.mark.parametrize("topology_prepared", [False, True])
def test_gradient_projection_context_restores_position_gradient_and_model_flag(topology_prepared):
    events = []

    class SteinerTopology:
        def rebuild_tree(self, _pin_pos):
            events.append("topology")
            return (None, None, None, None, None, None)

    class Model:
        use_timing_obj = True

        def __init__(self):
            self.op_collections = object()

        def timing_obj(self, position):
            value = position.sum()
            return value, value * 0.0, value, value

        def _timing_loss(self, wns, _tns, _ws, _ts):
            return wns.square()

    data = type(
        "Data",
        (),
        {
            "pin2node_map": torch.tensor([0, 1, 0, 1]),
            "pin2net_map": torch.tensor([0, 0, 1, 1]),
            "flat_net2pin_start_map": torch.tensor([0, 2, 4]),
            "net_weights": torch.ones(2),
        },
    )()
    placedb = type(
        "PlaceDB",
        (),
        {
            "num_nets": 2,
            "net_weights": np.ones(2, dtype=np.float32),
            "net_names": ["n0", "n1"],
        },
    )()
    model = Model()
    model.op_collections = type(
        "ModelOps",
        (),
        {
            "pin2pin_net_weight_op": staticmethod(
                lambda _position: torch.tensor(0.0)
            )
        },
    )()
    ops = type(
            "Ops",
            (),
            {
                "timing_propagation_op": object(),
                "pin_pos_op": staticmethod(lambda position: position),
                "steiner_topo_op": SteinerTopology(),
            },
    )()
    params = _params(
        timing_gradient_net_weight_scale=0.4,
        timing_gradient_net_weight_max=2.0,
        ignore_net_degree=100,
    )
    position = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    artifacts = []

    context = timing_gradient_projection.TimingGradientProjectionContext(
        topology_prepared=topology_prepared,
        params=params,
        placedb=placedb,
        model=model,
        pos=position,
        data_collections=data,
        op_collections=ops,
        iteration=20,
        gate_status={"enabled": True},
        timing_carrier="gradient_net_weight",
        record_timing_metrics=lambda *_timing: events.append("metrics"),
        write_artifact=lambda payload: artifacts.append(payload),
    )

    payload, projected = timing_gradient_projection.project_timing_gradient_to_net_weights(
        context
    )

    assert payload["status"] == "updated"
    assert projected is True
    assert events == (["metrics"] if topology_prepared else ["topology", "metrics"])
    assert artifacts[0]["carrier"] == "gradient_net_weight"
    assert torch.all(data.net_weights >= 1.0)
    assert np.all(placedb.net_weights >= 1.0)
    assert model.use_timing_obj is True
    assert position.grad is None


def test_timing_topology_context_publishes_generation_and_dbu_metadata():
    events = []

    class Topology:
        topology_frozen = False
        topology_generation = 3
        frozen_topology_generation = None
        net_flat_topo_sort = torch.tensor([0, 1])
        net_flat_topo_sort_start = torch.tensor([0, 2])
        pin_fa = torch.tensor([0, 1])
        flat_pin_to = torch.tensor([0, 1])
        flat_pin_to_start = torch.tensor([0, 1, 2])
        flat_pin_from = torch.tensor([1, 0])
        newx = torch.tensor([8.0, 12.0])
        newy = torch.tensor([4.0, 16.0])

        def rebuild_tree(self, _pin_pos):
            events.append("rebuild")
            return (
                self.net_flat_topo_sort,
                self.net_flat_topo_sort_start,
                self.pin_fa,
                self.flat_pin_to,
                self.flat_pin_to_start,
                self.flat_pin_from,
            )

        def __call__(self, _pin_pos):
            events.append("forward")
            return self.newx, self.newy

        def freeze_topology(self):
            self.topology_frozen = True
            self.frozen_topology_generation = self.topology_generation
            return self.frozen_topology_generation

    topo = Topology()
    data = SimpleNamespace()
    ops = SimpleNamespace(
        steiner_topo_op=topo,
        pin_pos_op=staticmethod(lambda _pos: torch.tensor([1.0, 2.0])),
    )
    placedb = SimpleNamespace(
        dbu=1000,
        r_unit=2.0,
        c_unit=0.25,
        params=SimpleNamespace(scale_factor=1.0, shift_factor=(0.0, 0.0)),
    )
    params = SimpleNamespace(scale_factor=2.0, shift_factor=(1.0, 3.0))
    context = timing_topology.TimingTopologyContext(
        data_collections=data,
        op_collections=ops,
        placedb=placedb,
        params=params,
        segment_direct_joint_enabled=False,
        record_stage=lambda name, _started: events.append(name),
        record_bytes=lambda name, _tensor: events.append(name),
        refresh_callback=lambda position: timing_topology.refresh_live_timing_topology(
            context, position
        ),
    )

    topology = timing_topology.refresh_live_timing_topology(
        context, torch.tensor([0.0])
    )
    assert "rebuild" in events
    assert topology["topology_generation"] == 3
    assert topology["node_x_dbu"].tolist() == [5, 7]
    assert topology["node_y_dbu"].tolist() == [5, 11]
    assert data.buffering_timing_topology_dbu == 1000
    assert data.buffering_timing_topology_r_unit == 2.0
    assert data.buffering_timing_topology_c_unit == 0.25

    generation = timing_topology.freeze_live_timing_topology(
        context, torch.tensor([0.0])
    )
    assert generation == 3
    assert data.buffering_timing_topology["topology_frozen"] is True
    assert events.count("rebuild") == 2


def test_gradient_net_weight_carrier_does_not_initialize_direct_loss_balance():
    class FakeModel:
        def __init__(self):
            self.initialize_count = 0

        def initialize_timing_grad_balance(self, pos, *, iteration, overflow):
            self.initialize_count += 1
            return {"initialized": True}

    placer = _placer()
    placer._timing_grad_balance_state = None
    placer._diff_tdp_step_status = {"timing_objective_active": True}
    params = _params(
        timing_placement_carrier="gradient_net_weight",
        timing_grad_balance_target_ratio=0.1,
    )
    model = FakeModel()

    result = placer._initialize_timing_grad_balance_if_needed(
        params,
        model,
        torch.tensor([0.0], requires_grad=True),
        iteration=20,
        overflow=0.19,
    )

    assert result is None
    assert model.initialize_count == 0
