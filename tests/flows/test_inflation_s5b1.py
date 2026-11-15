"""Window-best/capacity boundaries; actual timing is qualified by the native probe."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.discrete.common import phase_metrics
from dreamplace.flows.s5b1.inflation_s5b1 import InflationS5B1
from dreamplace.flows.gp.continuation import TimingOptWindowSchedule
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
    from dreamplace.flows.config.resolver import apply_flow_defaults

    params = inflation_window_params
    expected = (params.timing_opt_max_windows, rounds)
    params.timing_opt_sizing_rounds = rounds
    apply_flow_defaults(params)
    assert (params.timing_opt_max_windows, params.timing_opt_sizing_rounds) == expected


def test_pin2pin_placement_preserves_sizing_window_and_routability(inflation_window_params):
    from dreamplace.flows.config.resolver import apply_flow_defaults
    from dreamplace.flows.gp.timing_carriers import reset_legacy_net_weight_gate_summary

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
    from dreamplace.flows.config.resolver import apply_flow_defaults

    setattr(inflation_window_params, name, value)
    with pytest.raises(ValueError, match=name + " must be an integer"):
        apply_flow_defaults(inflation_window_params)


def test_nesterov_gp_keeps_discrete_window_parameters_outside_optimizer(inflation_window_params):
    from dreamplace.flows.config.resolver import apply_flow_defaults
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


def test_typed_virtual_area_uses_each_selected_master_and_padding():
    state = SimpleNamespace(
        is_typed_candidate=True,
        activation_param=torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.]]),
        drive_type_cell_ids=torch.tensor([0, 1, 2, 3]),
    )
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.lane = SimpleNamespace(_current_state=lambda: state)
    owner.params = SimpleNamespace(cell_padding_x=0.2)
    owner.data = SimpleNamespace(
        flat_libcell_width=torch.tensor([1., 2., 3., 4.]),
        flat_libcell_height=torch.tensor([2., 2., 2., 3.]),
    )
    model = SimpleNamespace(_segment_virtual_buffer_size=lambda: (1.4, 2.))

    assert owner.virtual_area(model) == pytest.approx(16.0)


def test_phase_metrics_keep_unweighted_reporting_units():
    assert phase_metrics(
        {
            "wns": -1200.0,
            "tns": -450000.0,
            "total_slew_violation": 12.5,
            "total_cap_violation": 0.037,
        }
    ) == {
        "wns_ns": -1.2,
        "tns_ns": -450.0,
        "total_slew_violation_ns": 12.5,
        "total_cap_violation_pf": 0.037,
        "total_cap_violation_ff": 37.0,
    }
    assert phase_metrics({"cap_violation": 0.25})["total_cap_violation_pf"] == 0.25
    assert phase_metrics({"cap_violation": 0.25})["total_cap_violation_ff"] == 250.0


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
    params = SimpleNamespace(
        timing_opt_sizing_rounds=2,
        timing_opt_max_windows=2,
        discrete_gradient_topk_shared_budget_percent=1.0,
        placement_sizing_mode="place_only",
        timing_objective_lane="timing_only",
        differentiable_timing_obj=0,
        timing_opt_buffering_enabled=0,
    )
    db = SimpleNamespace(num_movable_nodes=1, num_filler_nodes=0, area=4.,
                         total_fixed_node_area=0., total_space_area=4., regions=[])
    return params, db, data


@pytest.fixture
def timing_window_model(polynomial_window_inputs):
    from dreamplace.PlaceObj import PlaceObj

    params, _, data = polynomial_window_inputs
    model = PlaceObj.__new__(PlaceObj)
    model.params = params
    model.timing_wns_coeff, model.timing_tns_coeff = 2., .03
    model.timing_slew_weight, model.timing_cap_weight = 2., 3.
    model.timing_grad_balance_weight = 7.
    model.timing_leakage_weight = 0.
    model._timing_geometry_cache = None
    timing = SimpleNamespace()
    model.op_collections = SimpleNamespace(timing_propagation_op=timing)

    def timing_obj(pos, **kwargs):
        error = (data.real_size - 2.4).square().sum() + data.vt_logits.sum() * 0.
        penalty = (data.real_size - 1.).square().sum()
        timing.last_total_slew_violation_tensor = penalty * 100.
        timing.last_total_cap_violation_tensor = penalty * 10.
        return -error, -100. * error, error * 0., error * 0.

    model.timing_obj = timing_obj
    return model


def window_weight_state(model):
    from dreamplace.flows.config.specs import TIMING_OPT_COEFFICIENT_FIELDS

    return {
        **{name: getattr(model, field) for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items()},
        "timing_grad_balance_weight": model.timing_grad_balance_weight,
    }


@pytest.mark.parametrize("policy,gp_weight,expected_loss,expected_gradient", [
    ({"mode": "inherit"}, 7., 235.6, 432.),
    ({"mode": "inherit", "wns": 500., "tns": 5., "slew": 1., "cap": 1.}, 7., 235.6, 432.),
    ({"mode": "fixed", "wns": 500., "tns": 5., "slew": 1., "cap": 1.}, 7., 270., -580.),
    ({"mode": "fixed", "wns": 500., "tns": 5., "slew": 1., "cap": 1.}, .04, 270., -580.),
])
def test_window_coefficients_control_real_loss_and_gradient_and_restore_gp(
    policy, gp_weight, expected_loss, expected_gradient, polynomial_window_inputs,
    timing_window_model,
):
    params, db, data = polynomial_window_inputs
    params.timing_opt_coefficients = policy
    model = timing_window_model
    model.timing_grad_balance_weight = gp_weight
    owner = PolynomialWindow(params, db, data, None, None)
    before = window_weight_state(model)
    with torch.no_grad():
        data.real_size.fill_(2.)
    with owner.electrical_window(), owner._sizing_coefficients(model) as effective:
        loss = InflationS5B1._loss(owner, model, torch.tensor([10., 20.]))
        gradient = torch.autograd.grad(loss, data.real_size)[0]
        assert (float(loss.detach()), float(gradient.detach())) == pytest.approx(
            (expected_loss, expected_gradient)
        )
        expected = before if policy["mode"] == "inherit" else {
            **{name: value for name, value in policy.items() if name != "mode"},
            "timing_grad_balance_weight": 1.,
        }
        assert effective == {"mode": policy["mode"], **expected}
    assert window_weight_state(model) == before
    assert (params.placement_sizing_mode, params.timing_objective_lane,
            params.differentiable_timing_obj) == ("place_only", "timing_only", 0)


@pytest.mark.parametrize("policy,best_round,expected_size", [
    ({"mode": "inherit"}, 0, 1.),
    ({"mode": "fixed", "wns": 500., "tns": 5., "slew": 1., "cap": 1.}, 1, 2.),
])
def test_coefficient_policy_reaches_discrete_selection_and_best_state(
    policy, best_round, expected_size, polynomial_window_inputs, timing_window_model,
    monkeypatch,
):
    params, db, data = polynomial_window_inputs
    params.timing_opt_coefficients = policy
    model = timing_window_model
    owner = PolynomialWindow(params, db, data, None, lambda pos: None)
    monkeypatch.setattr(owner, "_loss", InflationS5B1._loss.__get__(owner))
    before = window_weight_state(model)
    summary = owner.run(model, torch.tensor([10., 20.]), iteration=10)
    assert (summary["sizing"]["best_round"], float(data.real_size.detach())) == (
        best_round, expected_size,
    )
    assert summary["sizing"]["coefficients"]["mode"] == policy["mode"]
    assert window_weight_state(model) == before


def test_inherit_reads_current_coefficients_at_each_window(
    polynomial_window_inputs, timing_window_model,
):
    params, db, data = polynomial_window_inputs
    params.timing_opt_coefficients = {"mode": "inherit"}
    owner = PolynomialWindow(params, db, data, None, None)
    model = timing_window_model
    with owner._sizing_coefficients(model) as first:
        assert first == {"mode": "inherit", **window_weight_state(model)}
    model.timing_wns_coeff, model.timing_tns_coeff = 4., .06
    before = window_weight_state(model)
    with owner._sizing_coefficients(model) as second:
        assert second == {"mode": "inherit", **before}
    assert first != second
    assert window_weight_state(model) == before


def test_fixed_coefficients_restore_on_sizing_failure(
    polynomial_window_inputs, timing_window_model, monkeypatch,
):
    params, db, data = polynomial_window_inputs
    params.timing_opt_coefficients = {
        "mode": "fixed", "wns": 500., "tns": 5., "slew": 1., "cap": 1.,
    }
    model = timing_window_model
    before = window_weight_state(model)
    owner = PolynomialWindow(params, db, data, None, lambda pos: None)

    def fail(*args):
        assert window_weight_state(model) == {
            "wns": 500., "tns": 5., "slew": 1., "cap": 1.,
            "timing_grad_balance_weight": 1.,
        }
        raise RuntimeError("sizing probe failure")

    monkeypatch.setattr(owner, "run_sizing", fail)
    with pytest.raises(RuntimeError, match="sizing probe failure"):
        owner.run(model, torch.tensor([10., 20.]), iteration=10)
    assert window_weight_state(model) == before
    assert (params.placement_sizing_mode, params.timing_objective_lane,
            params.differentiable_timing_obj) == ("place_only", "timing_only", 0)


@pytest.mark.parametrize("policy", [
    "fixed", {"mode": "unknown"}, {"mode": "inherit", "wnz": 1.},
    {"mode": "fixed", "wns": 1., "tns": 1., "slew": 1.},
    *[{"mode": "fixed", "wns": invalid, "tns": 1., "slew": 1., "cap": 1.}
      for invalid in (-1., float("nan"), float("inf"), True, "1")],
])
def test_invalid_coefficient_policy_fails_before_flow_execution(policy, inflation_window_params):
    from dreamplace.flows.config.resolver import apply_flow_defaults

    inflation_window_params.timing_opt_coefficients = policy
    with pytest.raises(ValueError, match="timing_opt_coefficients"):
        apply_flow_defaults(inflation_window_params)


@pytest.mark.parametrize('capacity,expected_size,best_round', [(4., 2., 1), (1.5, 1., 0)])
def test_restore_best_master_parameter_center_and_oscillation(
    capacity, expected_size, best_round, polynomial_window_inputs,
):
    params, db, data = polynomial_window_inputs
    owner = PolynomialWindow(params, db, data, None, None)
    pos = torch.tensor([10., 20.])
    geometry = SizingWindowGeometry(data, db, pos)
    model = SimpleNamespace(_timing_geometry_cache=None)
    summary = owner.run_sizing(model, pos, geometry, capacity)
    assert summary['best_round'] == best_round
    assert len(summary['rounds']) == 2
    assert summary['rounds'][-1]['loss'] > summary['rounds'][0]['loss']
    assert int(data.inst_cell_id[0]) == int(expected_size - 1)
    torch.testing.assert_close(data.real_size, torch.tensor([expected_size]))
    torch.testing.assert_close(pos + .5 * torch.cat((data.node_size_x, data.node_size_y)),
                               torch.tensor([10.5, 20.5]))
    assert owner.oscillation.history == ({0: [0, 1]} if best_round else {})


def test_sizing_only_windows_refresh_each_live_position(
    polynomial_window_inputs, monkeypatch, timing_window_model,
):
    params, db, data = polynomial_window_inputs
    refreshed = []
    owner = PolynomialWindow(params, db, data, None,
                             lambda pos: refreshed.append(pos.detach().clone()))
    original_loss = owner._loss
    def electrical_loss(model, pos):
        assert params.differentiable_timing_obj == 1
        return original_loss(model, pos)
    monkeypatch.setattr(owner, "_loss", electrical_loss)
    for method in ("prepare", "run_buffering", "retain_buffered_nets"):
        monkeypatch.setattr(owner, method, lambda *args: pytest.fail("S5-only entered buffering"))
    model = timing_window_model
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


def test_placement_candidate_prepare_uses_dirty_native_refresh():
    activation = torch.nn.Parameter(torch.zeros(2, 2))
    state = SimpleNamespace(
        activation_param=activation,
        candidate_net_id=torch.tensor([4, 7]),
    )
    observed = {}
    lane = SimpleNamespace(
        _current_state=lambda: state,
        buffer_relaxed_timing_payload={
            "candidate_net_id": torch.tensor([4, 7]),
        },
        refresh_candidate_geometry=lambda **kwargs: (
            observed.update(kwargs)
            or {"status": "ok", "reason": "dirty_native_refresh"}
        ),
    )
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.lane = lane
    owner.params = SimpleNamespace(flow_kind="placement")
    owner.placedb = SimpleNamespace(
        flat_node2pin_map=torch.tensor([0, 1, 2]),
        flat_node2pin_start_map=torch.tensor([0, 0, 1, 2]),
        pin2net_map=torch.tensor([4, 7]),
    )
    owner.data = SimpleNamespace()
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            frozen_nets=SimpleNamespace(candidate_state=None),
        )
    )
    topology_refreshes = []
    owner.refresh_topology = topology_refreshes.append
    owner._pending_dirty_instance_ids = {1}
    owner._pending_dirty_net_ids = {7}
    owner._candidate_refresh_in_progress = False
    model = SimpleNamespace(
        _timing_geometry_cache=object(),
        _virtual_cell_density_view=object(),
    )

    owner._prepare(model, pos=torch.zeros(2), active_only=False)

    assert len(topology_refreshes) == 1
    torch.testing.assert_close(topology_refreshes[0], torch.zeros(2))
    assert observed["active_only"] is False
    assert observed["dirty_net_ids"] == [4, 7]
    assert observed["refresh_records_all"] is True
    assert observed["force_rebuild"] is False
    assert owner._pending_dirty_instance_ids == set()
    assert owner._pending_dirty_net_ids == set()
    assert model._timing_geometry_cache is None
    assert model._virtual_cell_density_view is None


def test_placement_candidate_topology_refresh_limits_geometry_to_active_nets():
    state = SimpleNamespace(
        activation_param=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        ),
        candidate_net_id=torch.tensor([4, 4, 9]),
    )
    observed = {}
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: (
            observed.update(kwargs) or {"status": "ok"}
        ),
        attach_state=lambda _data: None,
    )
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.lane = lane
    owner.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    owner.data = SimpleNamespace()
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            frozen_nets=SimpleNamespace(candidate_state=None),
        )
    )
    model = SimpleNamespace(
        _timing_geometry_cache=object(),
        _virtual_cell_density_view=object(),
    )

    summary = owner.refresh_candidate_geometry_after_topology(
        model,
        reason="placement_topology_changed",
    )

    assert summary == {"status": "ok"}
    assert observed["active_only"] is True
    assert observed["dirty_net_ids"] == [4]
    assert owner.ops.steiner_topo_op.frozen_nets.candidate_state is state
    assert owner.params.enable_relaxed_buffer_timing is True


def test_placement_candidate_retention_freezes_accepted_nets():
    position = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    state = SimpleNamespace(
        activation_param=torch.nn.Parameter(
            torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        ),
        candidate_net_id=torch.tensor([7, 8]),
        timing_active_only=False,
    )
    observed = {}
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: (
            observed.update(kwargs) or {"status": "ok"}
        ),
        attach_state=lambda _data: None,
    )
    frozen = SimpleNamespace(candidate_state=None)
    freeze_calls = []
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.lane = lane
    owner.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    owner.data = SimpleNamespace(pos=[position])
    owner.ops = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            freeze_nets=freeze_calls.append,
            frozen_nets=frozen,
        )
    )
    topology_refreshes = []
    owner.refresh_topology = topology_refreshes.append
    owner._pending_dirty_instance_ids = {2}
    owner._pending_dirty_net_ids = {8}
    owner._candidate_refresh_in_progress = False
    model = SimpleNamespace(
        _timing_geometry_cache=object(),
        _virtual_cell_density_view=object(),
    )

    owner._retain_buffered_nets(model)

    assert freeze_calls == [[7]]
    assert len(topology_refreshes) == 1
    assert topology_refreshes[0] is position
    assert observed["active_only"] is False
    assert observed["dirty_net_ids"] == [7]
    assert frozen.candidate_state is state
    assert state.timing_active_only is True
    assert state.activation_param.requires_grad is False
    assert state.activation_param.grad is None
    assert owner.params.enable_relaxed_buffer_timing is True
    assert owner._pending_dirty_instance_ids == set()
    assert owner._pending_dirty_net_ids == set()


def milestone_owner(tmp_path, monkeypatch):
    """Keep real window publication; substitute the expensive S/B algorithms."""
    owner = InflationS5B1.__new__(InflationS5B1)
    owner.params = SimpleNamespace(
        timing_opt_max_windows=5, result_dir=str(tmp_path), design_name=lambda: "case",
    )
    owner.schedule = TimingOptWindowSchedule((.30, .25, .20, .15, .10), 5)
    owner.summaries = []
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
    monkeypatch.setattr(owner, "sync_native_sizing", lambda: {"synced": True})

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
        previous_count = owner.schedule.window_count
        owner.before_step(model, pos, optimizer, metric, {}, iteration=iteration, stage=stage)
        assert owner.before_inflation(model, pos, iteration=iteration, stage=stage) is None
        if owner.schedule.window_count != previous_count:
            group = optimizer.param_groups[0]
            assert all(group[field] == [] for field in ("g_k", "obj_k", "a_k", "alpha_k", "g_k_1"))
            torch.testing.assert_close(group["u_k"][0], pos)
            assert torch.equal(pos.grad, torch.zeros_like(pos))
            assert model.overflow.item() == pytest.approx(.4)
    assert [(item["trigger"]["threshold"], item["iteration"], item["stage"])
            for item in owner.summaries] == [(.30, 2, 0), (.25, 4, 1), (.20, 4, 1),
                                           (.15, 6, 2), (.10, 6, 2)]
    assert owner.schedule.pending_milestones == [] and owner.schedule.window_count == 5
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
    owner.schedule = TimingOptWindowSchedule((), 5)
    model = SimpleNamespace(
        objective_scale=1., inflation_state=SimpleNamespace(target_area=None),
        refresh_routability_operators=lambda: None, refresh_after_geometry_change=lambda: None,
    )
    pos = torch.zeros(4)
    for iteration in range(7):
        owner.before_inflation(model, pos, iteration=iteration, stage=0)
        assert owner.before_inflation(model, pos, iteration=iteration, stage=0) is None
    assert owner.schedule.window_count == 5
    assert [summary["iteration"] for summary in owner.summaries] == list(range(5))
    assert [summary["trigger"] for summary in owner.summaries] == [
        {"mode": "inflation", "iteration": iteration, "stage": 0} for iteration in range(5)
    ]


def test_timing_only_milestone_window_accepts_absent_inflation_state(tmp_path, monkeypatch):
    from dreamplace.NesterovAcceleratedGradientOptimizer import (
        NesterovAcceleratedGradientOptimizer,
    )

    owner = milestone_owner(tmp_path, monkeypatch)
    model = SimpleNamespace(
        objective_scale=1.,
        refresh_routability_operators=lambda: None,
        refresh_after_geometry_change=lambda: None,
    )
    pos = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0, 5.0]))

    def objective_and_gradient(value):
        objective = value.square().sum()
        return objective, torch.autograd.grad(objective, value)[0]

    optimizer = NesterovAcceleratedGradientOptimizer(
        [pos],
        lr=0.01,
        obj_and_grad_fn=objective_and_gradient,
        constraint_fn=lambda value: value,
        use_bb=False,
    )
    metric = SimpleNamespace(overflow=torch.tensor([0.29]), evaluate=lambda *args: None)
    pos.grad = torch.ones_like(pos)
    optimizer.step()
    owner.before_step(model, pos, optimizer, metric, {}, iteration=1, stage=0)
    assert owner.schedule.window_count == 1
    assert owner.summaries[0]["trigger"]["threshold"] == 0.30


def test_placement_window_sizing_delegates_to_lut_action_owner():
    owner = InflationS5B1.__new__(InflationS5B1)
    calls = []
    delegate = SimpleNamespace(
        _pending_dirty_instance_ids={7, 9},
        _pending_dirty_net_ids={3},
    )

    def run_sizing(*args, **kwargs):
        calls.append((args, kwargs))
        return {"rounds": [{"round": 1}, {"round": 2}]}

    delegate.run_sizing = run_sizing
    owner._lut_action_owner = delegate
    owner._pending_dirty_instance_ids = {1}
    owner._pending_dirty_net_ids = set()
    owner.round_index = 0

    result = owner.run_sizing(
        "model",
        "pos",
        "geometry",
        "capacity",
        action_mask=[True, False, True, False],
    )

    assert result == {"rounds": [{"round": 1}, {"round": 2}]}
    assert calls == [
        (
            ("model", "pos", "geometry", "capacity"),
            {"action_mask": [True, False, True, False]},
        )
    ]
    assert owner._pending_dirty_instance_ids == {1, 7, 9}
    assert owner._pending_dirty_net_ids == {3}
    assert owner.round_index == 2
    assert delegate._pending_dirty_instance_ids == set()
    assert delegate._pending_dirty_net_ids == set()


def test_s5b1_run_defers_buffering_when_midplace_repair_enabled(monkeypatch):
    from contextlib import nullcontext

    owner = InflationS5B1.__new__(InflationS5B1)
    owner.params = SimpleNamespace(
        timing_opt_buffering_enabled=True,
        repair_buffering_midplace_enable=True,
    )
    owner.placedb = SimpleNamespace(
        regions=[],
        num_movable_nodes=1,
        num_filler_nodes=0,
        area=100.0,
        total_fixed_node_area=0.0,
        total_space_area=100.0,
    )
    owner.data = SimpleNamespace(
        sizing_parameterization="real_size",
        node_size_x=torch.tensor([2.0]),
        node_size_y=torch.tensor([2.0]),
        original_node_size_x=torch.tensor([2.0]),
        original_node_size_y=torch.tensor([2.0]),
    )
    owner.lane = object()
    owner.virtual_area = lambda _model: 0.0
    owner.electrical_window = nullcontext
    owner._sizing_coefficients = lambda _model: nullcontext({"wns": 1.0})
    owner.run_sizing = lambda *_args, **_kwargs: {"best_round": 1}
    prepare_calls = []
    owner.prepare = lambda *_args, **_kwargs: prepare_calls.append(1)

    model = SimpleNamespace(_timing_geometry_cache=None)
    pos = torch.tensor([0.0, 0.0])
    summary = owner.run(model, pos, iteration=12)

    assert prepare_calls == []
    assert summary["buffering"] == {
        "status": "deferred_to_midplace_repair_timing",
        "scheduled": 0,
        "accepted": 0,
        "cumulative_count": 0,
    }



def test_s5b1_direct_b1_session_skips_zero_accept_and_commits_and_restarts_on_accept(
    ecc_schema3_carrier_harness, tmp_path
):
    from contextlib import nullcontext

    from dreamplace.flows.discrete.buffer_commit import DirectBufferCommitSession
    from dreamplace.flows.gp.continuation import MidplaceRestart, PlacementCursor
    from dreamplace.Placer import create_session
    from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
    from dreamplace.ops.buffer_insertion.optimization_state import (
        build_typed_discrete_candidate_state,
    )

    placedb = ecc_schema3_carrier_harness.placedb
    rawdb = ecc_schema3_carrier_harness.rawdb
    params = SimpleNamespace(
        place_io_engine="ecc",
        timing_opt_enabled=1,
        timing_opt_buffering_enabled=1,
        repair_buffering_midplace_enable=0,
        timing_opt_overflow_milestones=(0.30, 0.20),
        timing_opt_max_windows=2,
        enable_relaxed_buffer_timing=False,
        enable_fillers=0,
        cell_padding_x=0.0,
        scale_factor=1.0,
        shift_factor=(0.0, 0.0),
        result_dir=str(tmp_path),
        design_name=lambda: "fixture",
    )
    engine = SimpleNamespace(params=params, placedb=placedb)
    session = create_session(engine)
    assert isinstance(session, DirectBufferCommitSession)

    table = ecc_schema3_carrier_harness.make_drive_type_table(tmp_path)

    def _build_owner_and_placer():
        n_mov = placedb.num_movable_nodes
        data = SimpleNamespace(
            sizing_parameterization="real_size",
            flat_libcell_info=torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 0.0]]),
            flat_libcell_leakage=torch.tensor([1.0, 2.0]),
            flat_libcell_width=torch.tensor([1000.0, 2000.0]),
            flat_libcell_height=torch.tensor([1000.0, 1000.0]),
            inst_cell_id=torch.zeros(n_mov, dtype=torch.long),
            node_size_x=torch.as_tensor(placedb.node_size_x, dtype=torch.float64),
            node_size_y=torch.as_tensor(placedb.node_size_y, dtype=torch.float64),
            original_node_size_x=torch.as_tensor(placedb.node_size_x, dtype=torch.float64),
            original_node_size_y=torch.as_tensor(placedb.node_size_y, dtype=torch.float64),
            node_areas=torch.as_tensor(
                placedb.node_size_x * placedb.node_size_y, dtype=torch.float64
            ),
            target_density=torch.tensor([0.5], dtype=torch.float64),
            sorted_node_map=torch.arange(n_mov, dtype=torch.int32),
            pin_offset_x=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            pin_offset_y=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            original_pin_offset_x=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            original_pin_offset_y=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            flat_node2pin_map=torch.as_tensor(placedb.flat_node2pin_map, dtype=torch.long),
            flat_node2pin_start_map=torch.as_tensor(
                placedb.flat_node2pin_start_map, dtype=torch.long
            ),
            node2fence_region_map=torch.as_tensor(
                placedb.node2fence_region_map, dtype=torch.int32
            ),
        )
        ops = SimpleNamespace(
            timing_propagation_op=SimpleNamespace(
                last_total_slew_violation=0.0, last_total_cap_violation=0.0
            ),
            steiner_topo_op=SimpleNamespace(topology_generation=1, frozen_nets=None),
            virtual_cell_density_op=None,
        )
        owner = InflationS5B1(params, placedb, data, ops, lambda _pos: None)
        owner.electrical_window = nullcontext
        owner._electrical_window = nullcontext
        owner._sizing_coefficients = lambda _m: nullcontext({"wns": 1.0})
        owner.run_sizing = lambda *_a, **_k: {"best_round": 0}

        pos = torch.tensor(
            list(placedb.node_x) + list(placedb.node_y), dtype=torch.float64
        )
        placer = SimpleNamespace(
            placedb=placedb,
            data_collections=data,
            op_collections=ops,
            pos=[pos],
            inflation_s5b1=owner,
            alternating_discrete=None,
            _routability_controller=SimpleNamespace(_gp_steps=4, _last_inflation_step=None),
        )
        return owner, placer

    owner, placer = _build_owner_and_placer()
    session.runtime_ready(placer, owner, placer._routability_controller)
    assert owner.direct_commit_enabled is True

    accept_plan = [0, 1]

    def _prepare(_model, _pos):
        records = [
            {
                "candidate_id": 0,
                "net_id": 0,
                "candidate_node_id": 1,
                "buffer_main_type_index": 0,
                "parent_node_id": 0,
                "child_node_ids": [1],
                "segment_split_ratio": 0.5,
                "x_dbu": 50000,
                "y_dbu": 0,
            }
        ]
        state = build_typed_discrete_candidate_state(
            records,
            buffer_main_type_index=0,
            legal_buffer_count=4,
            drive_type_table=table,
        )
        config = SimpleNamespace(
            mode="candidate",
            candidate_strategy="discrete_net_gradient",
            active_strategy="discrete_net_gradient",
            candidate_drive_type_mode="four_strength_table",
            route_b_selection_fraction=0.003,
            continuous_steps=1,
            output_dir=str(tmp_path / "buffering"),
            commit_enabled=True,
            segment_capacity_enabled=False,
            max_selected_actions=None,
        )
        owner.lane = BufferingOptimizationLane(
            config,
            buffer_optimization_state=state,
            buffer_relaxed_timing_payload={"metadata": {"state_kind": "candidate"}},
            placedb=placedb,
        )
        owner.lane.attach_state(owner.data)
        params.enable_relaxed_buffer_timing = True

    def _run_buffering(_model, _pos, _cap):
        accepted = accept_plan.pop(0)
        state = owner.lane._current_state()
        with torch.no_grad():
            state.activation_param.zero_()
            if accepted:
                state.activation_param[0, 0] = 1.0
        return {"status": "ok", "scheduled": 1, "accepted": accepted}

    owner.prepare = _prepare
    owner.run_buffering = _run_buffering

    model = SimpleNamespace(
        Lgamma_iteration=10,
        Llambda_density_weight_iteration=1,
        Lsub_iteration=2,
        gamma=torch.tensor(4.0),
        overflow=torch.tensor([0.28]),
        _timing_geometry_cache=None,
        _virtual_cell_density_view=None,
        _candidate_density_view=None,
        last_timing_objective_terms={"total_loss": -10.0},
        timing_obj=lambda _p: tuple(torch.tensor(v) for v in (-10.0, -20.0, 0.0, 0.0)),
        _timing_loss=lambda wns, tns, _ws, _ts: -(wns + tns),
        refresh_routability_operators=lambda: None,
        refresh_after_geometry_change=lambda: None,
    )
    optimizer = SimpleNamespace(rebase_objective_state=lambda **_k: {"rebased": True})

    # Window 1 crosses 0.30 with accepted=0 -> clears virtual state, does NOT restart
    metric1 = SimpleNamespace(
        overflow=torch.tensor([0.28]), evaluate=lambda *_a, **_k: None
    )
    owner.before_step(
        model, placer.pos[0], optimizer, metric1, None, iteration=5, stage=0
    )
    assert params.enable_relaxed_buffer_timing is False
    assert owner._pending_window_commit_request is None
    assert (
        session.after_timing_window_step(
            placer, model, owner, metric1, PlacementCursor(0, 1, 0, 0, 5)
        )
        is False
    )
    assert ecc_schema3_carrier_harness.commit_calls == []

    # Window 2 crosses 0.20 with accepted=1 -> commits directly to ECC DB & raises MidplaceRestart
    metric2 = SimpleNamespace(
        overflow=torch.tensor([0.18]), evaluate=lambda *_a, **_k: None
    )
    owner.before_step(
        model, placer.pos[0], optimizer, metric2, None, iteration=8, stage=0
    )
    assert params.enable_relaxed_buffer_timing is False
    assert owner.lane._current_state() is None
    assert owner._pending_window_commit_request is not None
    with pytest.raises(MidplaceRestart):
        session.after_timing_window_step(
            placer, model, owner, metric2, PlacementCursor(0, 2, 0, 0, 8)
        )

    assert ecc_schema3_carrier_harness.sta_calls == []
    assert len(ecc_schema3_carrier_harness.commit_calls) == 1
    assert ecc_schema3_carrier_harness.refresh_records[-1]["ecc_sta_run"] is False
    assert rawdb.sparse_timing_view is not None
    assert rawdb.sparse_timing_view.statistics()["num_states"] == 12
    assert placedb.pending_continuation_seed is not None
    assert session.resume_cursor == PlacementCursor(0, 2, 0, 0, 8)

    # Reconstruct fresh owner after restart -> restores committed count & 0-action terminal request
    fresh_owner, fresh_placer = _build_owner_and_placer()
    routability_ctrl = SimpleNamespace(_gp_steps=0, _last_inflation_step=None)
    session.runtime_ready(fresh_placer, fresh_owner, routability_ctrl)
    assert fresh_owner.committed_buffer_count == 1
    assert fresh_owner.window_count == 2
    assert list(fresh_owner.pending_milestones) == []
    terminal = fresh_owner.prepare_terminal_request(
        model, fresh_placer.pos[0], iteration=10
    )
    assert terminal["projected_buffer_count"] == 0
    assert fresh_owner.terminal_request.is_noop is True
    assert fresh_owner.terminal_summary["direct_committed_count"] == 1


@pytest.mark.parametrize(
    ("action_mode", "expected_surrogate_mode"),
    [("lut_action", "lut_only"), ("off", "surrogate_only")],
)
def test_s5b1_loss_respects_lut_action_without_triggering_surrogate(
    action_mode, expected_surrogate_mode
):
    from dreamplace.flows.s5b1.inflation_s5b1 import InflationS5B1

    seen_modes = []
    owner = object.__new__(InflationS5B1)
    owner.action_mode = action_mode
    model = SimpleNamespace(
        timing_obj=lambda pos, surrogate_mode=None: (
            seen_modes.append(surrogate_mode),
            (torch.tensor(0.0),),
        )[1],
        _timing_loss=lambda value: value,
    )
    assert float(owner._loss(model, torch.tensor([0.0]))) == pytest.approx(0.0)
    assert seen_modes == [expected_surrogate_mode]


def test_resolve_median_buffer_area_and_window_budget_sharing():
    from dreamplace.flows.discrete.common import (
        S5B1WindowAreaBudget,
        resolve_median_buffer_area,
    )
    from dreamplace.ops.routability.cooptimization_area import AreaState

    placedb = SimpleNamespace(
        flat_libcell_info=[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
        ],
        buffer_main_type_candidate_indices=[2],
        buffer_main_type_index=2,
        flat_libcell_width=[2.0, 3.0, 5.0, 11.0],
        flat_libcell_height=[2.0, 10.0, 10.0, 10.0],
    )
    data = SimpleNamespace(
        flat_libcell_width=torch.tensor([2.0, 3.0, 5.0, 11.0]),
        flat_libcell_height=torch.tensor([2.0, 10.0, 10.0, 10.0]),
    )
    params = SimpleNamespace(cell_padding_x=0.5, place_io_engine="unknown")

    median_area = resolve_median_buffer_area(data, placedb, params)
    assert median_area == pytest.approx(60.0)

    entry = AreaState(
        movable=600.0,
        filler=0.0,
        capacity=1000.0,
    )
    budget = S5B1WindowAreaBudget.for_window(entry, median_buffer_area=median_area)
    assert budget.safe_ceiling == pytest.approx(900.0)
    assert budget.remaining_area == pytest.approx(300.0)
    assert budget.max_window_area_delta == pytest.approx(75.0)
    assert budget.window_capacity == pytest.approx(675.0)
    assert budget.window_buffer_equivalent_count == 2
    assert budget.sizing_budget_percent(
        600.0, sizeable_count=20, fallback_percent=1.0
    ) == pytest.approx(10.0)
    assert budget.sizing_budget_percent(
        675.0, sizeable_count=20, fallback_percent=1.0
    ) == pytest.approx(1.0)
    assert budget.buffering_selection_fraction(
        615.0, affected_net_count=10, fallback_fraction=0.001
    ) == pytest.approx(0.1)
    assert (
        budget.buffering_selection_fraction(
            675.0, affected_net_count=10, fallback_fraction=0.001
        )
        is None
    )


def test_s5b1_window_shares_25_percent_remaining_area_between_s5_and_b1(monkeypatch):
    from contextlib import nullcontext

    from dreamplace.flows.discrete.common import (
        effective_buffering_selection_fraction,
        effective_sizing_budget_percent,
    )

    owner = InflationS5B1.__new__(InflationS5B1)
    owner.params = SimpleNamespace(
        timing_opt_buffering_enabled=True,
        repair_buffering_midplace_enable=False,
        discrete_gradient_topk_shared_budget_percent=1.0,
        buffering_route_b_selection_fraction=0.001,
        enable_relaxed_buffer_timing=True,
        cell_padding_x=0.0,
    )
    owner.placedb = SimpleNamespace(
        regions=[],
        num_movable_nodes=2,
        num_filler_nodes=0,
        area=1000.0,
        total_fixed_node_area=0.0,
        total_space_area=1000.0,
    )
    owner.data = SimpleNamespace(
        sizing_parameterization="real_size",
        inst_is_sizeable=torch.tensor([True] * 100),
        node_size_x=torch.tensor([30.0, 30.0]),
        node_size_y=torch.tensor([10.0, 10.0]),
        original_node_size_x=torch.tensor([30.0, 30.0]),
        original_node_size_y=torch.tensor([10.0, 10.0]),
    )
    owner.ops = SimpleNamespace(
        timing_propagation_op=None,
        steiner_topo_op=SimpleNamespace(topology_generation=1, frozen_nets=None),
        virtual_cell_density_op=None,
    )
    owner.median_buffer_area = 10.0
    owner._active_area_budget = None
    owner._lut_action_owner = SimpleNamespace(_active_area_budget=None)
    owner._pending_dirty_net_ids = set()
    owner.window_count = 1
    owner.direct_commit_enabled = False
    owner.virtual_area = lambda _model: 0.0
    owner.electrical_window = nullcontext
    owner._sizing_coefficients = lambda _model: nullcontext({"wns": 1.0})
    owner.retain_buffered_nets = lambda _model: None

    state = SimpleNamespace(
        activation_param=torch.zeros((4, 2), dtype=torch.float64),
        net_ids=torch.arange(50, dtype=torch.long),
        candidate_net_index=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    owner.lane = SimpleNamespace(
        config=SimpleNamespace(route_b_selection_fraction=0.001, continuous_steps=1),
        _current_state=lambda: state,
    )
    owner.prepare = lambda *_a, **_k: setattr(
        owner.params, "enable_relaxed_buffer_timing", True
    )

    observed = {}

    def _partial_sizing(model, _pos, _geom, capacity):
        observed["sizing_cap"] = capacity
        observed["sizing_pct_before"] = effective_sizing_budget_percent(owner, model)
        assert owner._lut_action_owner._active_area_budget is owner._active_area_budget
        # Consume 25 of the 75 window allowance (entry=600, safe=900, 25%*300=75).
        owner.data.node_size_x = torch.tensor([32.5, 30.0])
        observed["sizing_pct_after"] = effective_sizing_budget_percent(owner, model)
        return {"best_round": 1}

    def _fake_candidate_scheduler(**kwargs):
        observed["b1_selection_fraction"] = kwargs["selection_fraction"]
        observed["b1_gate_cap"] = kwargs["incremental_gate"].capacity
        return SimpleNamespace(
            strategy="discrete_net_gradient",
            iterations=1,
            terminal_reason="max_rounds",
            terminal_metrics={"loss": -5.0},
            metrics_trace=[],
            trace_path=None,
            trace_sha256=None,
            best_restore={"best_round": 1},
        )

    monkeypatch.setattr(
        "dreamplace.flows.discrete.buffering_filter.IncrementalBufferingGate.check_available",
        lambda _self: None,
    )
    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner.run_discrete_candidate_buffer_scheduler",
        _fake_candidate_scheduler,
    )
    owner.run_sizing = _partial_sizing

    model = SimpleNamespace(_timing_geometry_cache=None)
    pos = torch.tensor([0.0, 0.0, 0.0, 0.0])
    summary1 = owner.run(model, pos, iteration=5)

    assert observed == {
        "sizing_cap": pytest.approx(675.0),
        "sizing_pct_before": pytest.approx(8.0),
        "sizing_pct_after": pytest.approx(5.0),
        "b1_selection_fraction": pytest.approx(5.0 / 50.0),
        "b1_gate_cap": pytest.approx(675.0),
    }
    assert summary1["area_budget"] == {
        "enabled": True,
        "entry_movable": pytest.approx(600.0),
        "placement_capacity": pytest.approx(1000.0),
        "max_density_ratio": pytest.approx(0.90),
        "max_remaining_fraction": pytest.approx(0.25),
        "safe_ceiling": pytest.approx(900.0),
        "remaining_area": pytest.approx(300.0),
        "max_window_area_delta": pytest.approx(75.0),
        "window_capacity": pytest.approx(675.0),
        "median_buffer_area": pytest.approx(10.0),
        "window_buffer_equivalent_count": 8,
        "after_sizing_movable": pytest.approx(625.0),
        "sizing_consumed_area": pytest.approx(25.0),
        "buffering_remaining_allowance": pytest.approx(50.0),
        "after_window_movable": pytest.approx(625.0),
        "window_consumed_area": pytest.approx(25.0),
    }
    assert owner._active_area_budget is None
    assert owner._lut_action_owner._active_area_budget is None

    # Now verify that when S5 consumes the full 25% window allowance, B1 skips cleanly.
    def _exhaust_sizing(model, _pos, _geom, capacity):
        assert capacity == pytest.approx(693.75)
        owner.data.node_size_x = torch.tensor([39.375, 30.0])
        assert (
            effective_buffering_selection_fraction(
                owner, model, affected_net_count=50, fallback_fraction=0.001
            )
            is None
        )
        return {"best_round": 2}

    owner.run_sizing = _exhaust_sizing
    summary2 = owner.run(model, pos, iteration=6)
    assert summary2["buffering"] == {
        "status": "area_budget_exhausted",
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "scheduled": 0,
        "accepted": 0,
        "accepted_net_ids": [],
        "gradient_norm": None,
        "cumulative_count": 0,
    }
    assert owner.params.enable_relaxed_buffer_timing is False


def test_s5b1_window_area_budget_3round_rollover_and_proposal_quartile():
    from dreamplace.flows.discrete.common import (
        S5B1WindowAreaBudget,
        resolve_proposal_buffer_area,
    )
    from dreamplace.ops.routability.cooptimization_area import AreaState

    placedb = SimpleNamespace(
        flat_libcell_info=[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 2.0, 3.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
        ],
        buffer_main_type_candidate_indices=[2],
        buffer_main_type_index=2,
    )
    data = SimpleNamespace(
        flat_libcell_width=[10.0, 2.0, 1.0, 4.0, 8.0],
        flat_libcell_height=[1.0, 2.0, 2.0, 2.0, 2.0],
    )
    params = SimpleNamespace(cell_padding_x=0.0, place_io_engine="unknown")
    # Sorted legal areas = [2.0, 4.0, 8.0, 16.0]; index 4 // 4 = 1 -> 4.0.
    assert resolve_proposal_buffer_area(data, placedb, params) == pytest.approx(4.0)

    entry = AreaState(
        movable=600.0,
        filler=0.0,
        capacity=1000.0,
    )
    # remaining = 900 - 600 = 300; window max delta = 75 -> window_capacity = 675.
    budget = S5B1WindowAreaBudget.for_window(
        entry,
        median_buffer_area=10.0,
        proposal_buffer_area=4.0,
    )
    assert budget.round_allowance(600.0, round_idx=0, total_rounds=3) == pytest.approx(25.0)
    assert budget.round_capacity(600.0, round_idx=0, total_rounds=3) == pytest.approx(625.0)
    # Unused rollover: only +5.0 consumed in round 0 -> remaining 70.0 / 2 = 35.0 in round 1.
    assert budget.round_allowance(605.0, round_idx=1, total_rounds=3) == pytest.approx(35.0)
    assert budget.round_capacity(605.0, round_idx=1, total_rounds=3) == pytest.approx(640.0)
    # Round 2 takes all remaining capacity up to window_capacity=675.
    assert budget.round_capacity(620.0, round_idx=2, total_rounds=3) == pytest.approx(675.0)
    # Exhausted when current_movable >= window_capacity.
    assert budget.buffering_selection_fraction(
        current_movable=675.0,
        affected_net_count=50,
        fallback_fraction=0.1,
        round_idx=1,
        total_rounds=3,
    ) is None
