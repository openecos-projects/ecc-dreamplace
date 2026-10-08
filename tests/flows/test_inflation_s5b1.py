"""Window-best/capacity boundaries; actual timing is qualified by the native probe."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.inflation_s5b1 import InflationS5B1
from dreamplace.ops.routability.cooptimization_area import SizingWindowGeometry


@pytest.fixture
def inflation_window_params():
    from dreamplace.Params import Params

    params = Params()
    template = json.loads(
        (Path(__file__).resolve().parents[2] / "dreamplace/params.json").read_text()
    )
    params.fromJson({key: metadata["default"] for key, metadata in template.items()})
    params._global_place_stages_explicit = True
    params.flow_kind = "placement"
    params.timing_opt_enabled = 1
    params.l_shape_use_ggr_topology = 0
    return params


@pytest.mark.parametrize("rounds", [1, 10])
def test_sizing_round_limit_keeps_window_limit_separate(rounds, inflation_window_params):
    from dreamplace.flows.flow_config import apply_flow_defaults

    params = inflation_window_params
    expected = (params.timing_opt_max_windows, rounds)
    params.timing_opt_sizing_rounds = rounds
    apply_flow_defaults(params)
    assert (params.timing_opt_max_windows, params.timing_opt_sizing_rounds) == expected


def test_pin2pin_placement_preserves_sizing_window_and_routability(inflation_window_params):
    from dreamplace.flows.flow_config import apply_flow_defaults
    from dreamplace.flows.timing_objective_policy import reset_legacy_net_weight_gate_summary

    params = inflation_window_params
    overrides = {
        "diff_timing_driven_placement": 0, "differentiable_timing_obj": 0,
        "enable_net_weighting": 1, "pin2pin_net_weighting": 1,
        "net_weighting_scheme": "pin2pin", "timing_opt_sizing_rounds": 10,
        "timing_opt_buffering_enabled": 0,
    }
    for name, value in overrides.items():
        setattr(params, name, value)
        setattr(params, f"_{name}_explicit", True)
    apply_flow_defaults(params)
    assert {name: getattr(params, name) for name in overrides} == overrides
    assert reset_legacy_net_weight_gate_summary(params)["enabled"]
    assert (params.routability_opt_flag, params.with_sta, params.placement_sizing_mode) == (
        1, 1, "place_only",
    )


@pytest.mark.parametrize("name,value", [
    ("timing_opt_max_windows", 6),
    ("timing_opt_sizing_rounds", 11),
    ("timing_opt_sizing_rounds", 0),
    ("timing_opt_sizing_rounds", 1.5),
    ("timing_opt_sizing_rounds", True),
])
def test_invalid_window_limits_are_rejected(name, value, inflation_window_params):
    from dreamplace.flows.flow_config import apply_flow_defaults

    setattr(inflation_window_params, name, value)
    with pytest.raises(ValueError, match=name + " must be an integer"):
        apply_flow_defaults(inflation_window_params)


def test_nesterov_gp_keeps_discrete_window_parameters_outside_optimizer(inflation_window_params):
    from dreamplace.flows.flow_config import apply_flow_defaults
    from dreamplace.NesterovAcceleratedGradientOptimizer import (
        NesterovAcceleratedGradientOptimizer,
    )
    from dreamplace.NonLinearPlace import NonLinearPlace

    params = inflation_window_params
    configured_stages = copy.deepcopy(params.global_place_stages)
    apply_flow_defaults(params)
    assert params.global_place_stages == configured_stages

    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    placer.pos = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([2., -1.]))])
    placer.size_params = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([1.]))])
    placer.vt_params = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1, 3))])
    placer.data_collections = SimpleNamespace(real_size=placer.size_params[0], size_logits=None)
    placer._validate_sizing_mode(params)
    groups = placer._optimizer_parameters(params)
    placer._validate_optimizer_parameter_groups(params.global_place_stages[0]["optimizer"], groups)
    assert [group["group_name"] for group in groups] == ["placement"]

    def objective_and_gradient(pos):
        objective = pos.square().sum()
        return objective, torch.autograd.grad(objective, pos)[0]

    optimizer = NesterovAcceleratedGradientOptimizer(
        groups, lr=0.1, obj_and_grad_fn=objective_and_gradient,
        constraint_fn=lambda pos: pos, use_bb=params.use_bb,
    )
    placer.pos[0].grad = torch.ones_like(placer.pos[0])
    before = placer.pos[0].detach().clone()
    optimizer.step()
    assert torch.isfinite(placer.pos[0]).all()
    assert not torch.equal(placer.pos[0], before)
    assert placer.size_params[0].item() == 1.
    assert torch.equal(placer.vt_params[0], torch.zeros(1, 3))


class PolynomialWindow(InflationS5B1):
    # A real Taylor move overshoots the minimum on round 2. Only the electrical
    # and master-sync boundaries are substituted; the production selector,
    # best-state owner, geometry and oscillation handling remain in use.
    def _loss(self, model, pos):
        return ((self.data.real_size - 2.4) ** 2).sum() + self.data.vt_logits.sum() * 0.

    def virtual_area(self, model, state=None):
        return 0.

    def _sync(self, model, summary, geometry):
        rows = torch.tensor(summary['applied_instance_ids'], dtype=torch.long)
        cells = torch.tensor(summary['applied_cell_ids'], dtype=torch.long)
        self.data.inst_cell_id[rows] = cells
        widths = self.data.flat_libcell_info[cells, 2]
        geometry.apply_sizes(rows, widths, torch.ones_like(widths))


@pytest.fixture
def polynomial_window_inputs():
    data = SimpleNamespace(
        flat_libcell_info=torch.tensor([[0., 0., 1., 0.], [0., 0., 2., 0.], [0., 0., 3., 0.]]),
        flat_libcell_leakage=torch.tensor([1., 2., 3.]),
        inst_size_lower=torch.tensor([1.]), inst_size_upper=torch.tensor([3.]),
        inst_vt_mask=torch.tensor([[True]]), inst_is_sizeable=torch.tensor([True]),
        inst_cell_id=torch.tensor([0]), real_size=torch.nn.Parameter(torch.tensor([1.])),
        vt_logits=torch.nn.Parameter(torch.zeros(1, 1)),
        node_size_x=torch.tensor([1.]), node_size_y=torch.tensor([1.]),
        original_node_size_x=torch.tensor([1.]), original_node_size_y=torch.tensor([1.]),
        node_areas=torch.tensor([1.]),
    )
    data.get_continuous_size_parameter = lambda: data.real_size
    data.sizing_parameterization = "real_size"
    params = SimpleNamespace(timing_opt_sizing_rounds=2,
                             discrete_gradient_topk_shared_budget_percent=1.,
                             placement_sizing_mode="place_only", timing_objective_lane="timing_only",
                             differentiable_timing_obj=0,
                             timing_opt_buffering_enabled=0)
    db = SimpleNamespace(num_movable_nodes=1, num_filler_nodes=0, area=4.,
                         total_fixed_node_area=0., total_space_area=4., regions=[])
    return params, db, data


@pytest.mark.parametrize('capacity,expected_size,best_round', [(4., 2., 1), (1.5, 1., 0)])
def test_restore_best_master_parameter_center_and_oscillation(
    capacity, expected_size, best_round, polynomial_window_inputs,
):
    params, db, data = polynomial_window_inputs
    owner = PolynomialWindow(params, db, data, None, None)
    pos = torch.tensor([10., 20.])
    geometry = SizingWindowGeometry(data, db, pos)
    model = SimpleNamespace(_timing_geometry_cache=None)
    summary = owner._sizing(model, pos, geometry, capacity)
    assert summary['best_round'] == best_round
    assert len(summary['rounds']) == 2
    assert summary['rounds'][-1]['loss'] > summary['rounds'][0]['loss']
    assert int(data.inst_cell_id[0]) == int(expected_size - 1)
    torch.testing.assert_close(data.real_size, torch.tensor([expected_size]))
    torch.testing.assert_close(pos + .5 * torch.cat((data.node_size_x, data.node_size_y)),
                               torch.tensor([10.5, 20.5]))
    assert owner.oscillation.history == ({0: [0, 1]} if best_round else {})


def test_sizing_only_windows_refresh_each_live_position(polynomial_window_inputs, monkeypatch):
    params, db, data = polynomial_window_inputs
    refreshed = []
    owner = PolynomialWindow(params, db, data, None,
                             lambda pos: refreshed.append(pos.detach().clone()))
    original_loss = owner._loss
    def electrical_loss(model, pos):
        assert params.differentiable_timing_obj == 1
        return original_loss(model, pos)
    monkeypatch.setattr(owner, "_loss", electrical_loss)
    for method in ("_prepare", "_buffering", "_retain_buffered_nets"):
        monkeypatch.setattr(owner, method, lambda *args: pytest.fail("S5-only entered buffering"))
    model = SimpleNamespace(_timing_geometry_cache=None)
    pos = torch.tensor([10., 20.])
    first_pos = pos.clone()
    first = owner.run(model, pos, iteration=10)
    pos.add_(2.)
    second_pos = pos.clone()
    second = owner.run(model, pos, iteration=20)
    assert len(refreshed) == 2
    torch.testing.assert_close(refreshed[0], first_pos)
    torch.testing.assert_close(refreshed[1], second_pos)
    assert first['sizing']['best_round'] == 1
    torch.testing.assert_close(data.real_size, torch.tensor([2.]))
    assert owner.lane is None
    for summary in (first, second):
        assert summary['buffering'] == {"status": "disabled", "scheduled": 0,
                                        "accepted": 0, "cumulative_count": 0}
        assert (summary['preparation_ms'], summary['buffering_ms']) == (0., 0.)
    assert (params.placement_sizing_mode, params.timing_objective_lane,
            params.differentiable_timing_obj) == (
        "place_only", "timing_only", 0,
    )


def milestone_owner(tmp_path, monkeypatch):
    """Keep real window publication; substitute the expensive S/B algorithms."""
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.params = SimpleNamespace(
        timing_opt_max_windows=5, result_dir=str(tmp_path), design_name=lambda: "case",
    )
    owner.milestones = (.30, .25, .20, .15, .10)
    owner.pending_milestones = list(owner.milestones)
    owner.window_count, owner.attempts, owner.summaries = 0, set(), []
    owner.capacity = 100.
    owner.placedb = SimpleNamespace(
        num_movable_nodes=1, num_filler_nodes=1, total_movable_node_area=1.,
        node_size_x=torch.tensor([1., 1.]).numpy(), node_size_y=torch.ones(2).numpy(),
    )
    owner.data = SimpleNamespace(
        node_size_x=torch.tensor([1., 1.]), node_size_y=torch.ones(2),
        node_areas=torch.ones(2), target_density=torch.tensor([.02]),
        sorted_node_map=torch.tensor([0], dtype=torch.int32),
    )
    monkeypatch.setattr(owner, "virtual_area", lambda model: 0.)
    monkeypatch.setattr(owner, "_sync_native_sizing", lambda: {"synced": True})

    def electrical_window(model, pos, *, iteration):
        # Change a real footprint to exercise filler/area/cache publication.
        owner.data.node_size_x[0] += .1
        model.objective_scale += 1.
        return {"iteration": iteration, "sizing": {"best_round": 0},
                "buffering": {"accepted": 0, "cumulative_count": 0}}

    monkeypatch.setattr(owner, "run", electrical_window)
    return owner


def test_milestones_survive_rebounds_stage_restart_and_skipped_thresholds(tmp_path, monkeypatch):
    from dreamplace.NesterovAcceleratedGradientOptimizer import (
        NesterovAcceleratedGradientOptimizer,
    )

    owner = milestone_owner(tmp_path, monkeypatch)
    refreshes = []
    model = SimpleNamespace(
        objective_scale=1., inflation_state=SimpleNamespace(target_area=2.),
        refresh_routability_operators=lambda: refreshes.append("route"),
        refresh_after_geometry_change=lambda: refreshes.append("geometry"),
    )
    pos = torch.nn.Parameter(torch.tensor([2., 3., 4., 5.]))

    def objective_and_gradient(value):
        objective = model.objective_scale * (value.square() + .1 * value.pow(4)).sum()
        return objective, torch.autograd.grad(objective, value)[0]

    optimizer = NesterovAcceleratedGradientOptimizer(
        [pos], lr=.01, obj_and_grad_fn=objective_and_gradient,
        constraint_fn=lambda value: value, use_bb=False,
    )
    restart_state = copy.deepcopy(optimizer.state_dict())
    previous_stage = 0
    evaluations = []
    metric = SimpleNamespace(overflow=torch.tensor([1.]))

    def evaluate(db, ops, live_pos, data):
        assert data.node_areas[0] == data.node_size_x[0] * data.node_size_y[0]
        assert db.total_movable_node_area == data.cooptimization_area.movable
        evaluations.append(data.cooptimization_area)
        metric.overflow = torch.tensor([.4])  # A window may raise overflow again.

    metric.evaluate = evaluate
    # Inflation attempts and stage numbers change, but the owner persists.
    for iteration, stage, overflow in [(1, 0, .31), (2, 0, .30), (3, 1, .4),
                                       (4, 1, .19), (5, 2, .4), (6, 2, .09)]:
        if stage != previous_stage:
            optimizer.load_state_dict(copy.deepcopy(restart_state))
            previous_stage = stage
        pos.grad = torch.ones_like(pos)
        optimizer.step()
        metric.overflow = torch.tensor([overflow], dtype=torch.float64)
        previous_count = owner.window_count
        owner.before_step(model, pos, optimizer, metric, {}, iteration=iteration, stage=stage)
        assert owner.before_inflation(model, pos, iteration=iteration, stage=stage) is None
        if owner.window_count != previous_count:
            group = optimizer.param_groups[0]
            assert all(group[field] == [] for field in ("g_k", "obj_k", "a_k", "alpha_k", "g_k_1"))
            torch.testing.assert_close(group["u_k"][0], pos)
            assert torch.equal(pos.grad, torch.zeros_like(pos))
            assert model.overflow.item() == pytest.approx(.4)
    assert [(item["trigger"]["threshold"], item["iteration"], item["stage"])
            for item in owner.summaries] == [(.30, 2, 0), (.25, 4, 1), (.20, 4, 1),
                                           (.15, 6, 2), (.10, 6, 2)]
    assert owner.pending_milestones == [] and owner.window_count == 5
    assert len(evaluations) == 3
    assert refreshes == ["route", "geometry"] * 5
    saved = json.loads((tmp_path / "case_inflation_s5b1.json").read_text())
    assert saved == {"windows": owner.summaries}
    # The following GP evaluation consumes a fresh gradient of the new objective.
    optimizer.step()
    _, fresh_gradient = objective_and_gradient(pos)
    torch.testing.assert_close(optimizer.param_groups[0]["g_k"][0], fresh_gradient)


def test_legacy_inflation_windows_keep_attempt_deduplication(tmp_path, monkeypatch):
    owner = milestone_owner(tmp_path, monkeypatch)
    owner.milestones, owner.pending_milestones = (), []
    model = SimpleNamespace(
        objective_scale=1., inflation_state=SimpleNamespace(target_area=None),
        refresh_routability_operators=lambda: None, refresh_after_geometry_change=lambda: None,
    )
    pos = torch.zeros(4)
    for iteration in range(7):
        owner.before_inflation(model, pos, iteration=iteration, stage=0)
        assert owner.before_inflation(model, pos, iteration=iteration, stage=0) is None
    assert owner.window_count == 5
    assert [summary["iteration"] for summary in owner.summaries] == list(range(5))
    assert all("trigger" not in summary for summary in owner.summaries)
