"""Master-dependent electrical limits and the complete violation derivative."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.ops.discrete_gradient_topk import ACTION_NAMES
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
from dreamplace.PlaceObj import PlaceObj


def _model():
    model = PlaceObj.__new__(PlaceObj)
    torch.nn.Module.__init__(model)
    model.data_collections = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        pin_2_libpin_offset=torch.tensor([0]),
        cell_id_2_libpin_id_start=torch.arange(6),
        flat_lib_pin_slew_limit=torch.tensor(
            [100.0, 120.0, 80.0, 90.0, 110.0], dtype=torch.float64
        ),
        flat_lib_pin_cap_limit=torch.tensor([0.04, 0.06, 0.03, 0.035, 0.05], dtype=torch.float64),
    )
    model._active_discrete_action_context = SimpleNamespace(
        action_names=ACTION_NAMES,
        target_cell_ids=torch.tensor([[1, 2, 3, 4]]),
        valid_instances=torch.ones((1, 4), dtype=torch.bool),
    )
    return model


def _limits(model, beta):
    model._active_discrete_action_context.beta = beta
    data = model.data_collections
    return model._apply_discrete_action_limits(
        data.flat_lib_pin_slew_limit[:1], data.flat_lib_pin_cap_limit[:1], data
    )


def test_action_limits_interpolate_from_one_current_master():
    model = _model()
    beta = torch.tensor([[0.1, 0.2, 0.15, 0.05]], dtype=torch.float64)
    # 100 + .1*20 - .2*20 - .15*10 + .05*10 = 97 ps.
    # .04 + .1*.02 - .2*.01 - .15*.005 + .05*.01 = .03975 pF.
    expected = (
        torch.tensor([97.0], dtype=torch.float64),
        torch.tensor([0.03975], dtype=torch.float64),
    )
    torch.testing.assert_close(_limits(model, beta), expected, atol=1e-12, rtol=0)


def test_negative_actions_include_limit_and_measured_value_derivatives():
    model = _model()
    timing = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(timing)
    timing.log_top_violations = False
    slew_slope = torch.tensor([0.0, 5.0, -2.0, 0.0], dtype=torch.float64)
    cap_slope = torch.tensor([0.0, -0.004, -0.003, 0.0], dtype=torch.float64)
    mask = torch.tensor([True])

    def evaluate(beta):
        slew_limit, cap_limit = _limits(model, beta)
        timing.pin_rtran_live = 130.0 + (beta * slew_slope).sum(dim=1)
        timing.pin_ftran_live = timing.pin_rtran_live - 5.0
        timing.pin_net_cap_rise_live = 0.05 + (beta * cap_slope).sum(dim=1)
        timing.pin_net_cap_fall_live = timing.pin_net_cap_rise_live - 0.001
        return torch.stack(
            (
                timing.get_total_slew_violation_tensor(mask, slew_limit),
                timing.get_total_cap_violation_tensor(mask, cap_limit),
            )
        )

    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    # size_down: 5 - (-20) ps and (-.004 - (-.01))*1000 fF.
    # VT slowdown: -2 - (-10) ps and (-.003 - (-.005))*1000 fF.
    jacobian = torch.autograd.functional.jacobian(evaluate, beta).squeeze(1)
    expected = torch.tensor(
        [
            [-20.0, 25.0, 8.0, -10.0],
            [-20.0, 6.0, 2.0, -10.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(jacobian, expected, atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(evaluate, (beta,), eps=1e-6, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("objective", ["tns", "loss"])
def test_limit_gradients_enter_weighted_objective_only(objective):
    model = _model()
    model.params = SimpleNamespace(timing_objective_lane="timing_slew_cap")
    model.timing_wns_coeff, model.timing_tns_coeff = 500.0, 5.0
    timing = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(timing)
    timing.log_top_violations = False
    model.op_collections = SimpleNamespace(timing_propagation_op=timing)
    timing.critical_endpoint_pruning_mode = "off"
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.params = SimpleNamespace(timing_action_score_objective=objective)
    owner.action_mode = "lut_action"
    context = model._active_discrete_action_context
    context.beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True])

    def timing_obj(pos, surrogate_mode, discrete_action_context):
        slew_limit, cap_limit = _limits(model, discrete_action_context.beta)
        timing.pin_rtran_live = context.beta.new_tensor([130.0])
        timing.pin_ftran_live = context.beta.new_tensor([125.0])
        timing.pin_net_cap_rise_live = context.beta.new_tensor([0.05])
        timing.pin_net_cap_fall_live = context.beta.new_tensor([0.049])
        timing.last_total_slew_violation_tensor = timing.get_total_slew_violation_tensor(
            mask, slew_limit
        )
        timing.last_total_cap_violation_tensor = timing.get_total_cap_violation_tensor(
            mask, cap_limit
        )
        zero = context.beta.sum() * 0
        return zero - 10, zero - 20, zero, zero

    model.timing_obj = timing_obj
    loss = owner._loss(model, torch.zeros(2, dtype=torch.float64), context)
    score = context.score_loss if context.score_loss is not None else loss
    gradient = torch.autograd.grad(score, context.beta)[0]
    expected = [[-40.0, 30.0, 15.0, -20.0]] if objective == "loss" else [[0.0] * 4]
    torch.testing.assert_close(gradient, context.beta.new_tensor(expected), atol=1e-12, rtol=0)


def test_timing_obj_impl_keeps_lut_only_active_until_limits_are_computed(monkeypatch):
    import dreamplace.PlaceObj as place_obj_module

    def fail_soft_interpolation(*args, **kwargs):
        raise AssertionError("lut_only call must not use soft size/VT limit interpolation")

    monkeypatch.setattr(
        place_obj_module,
        "compute_size_interpolated_pin_properties",
        fail_soft_interpolation,
    )
    import dreamplace.ops.timing_propagation.action_cell_aat as action_cell_aat

    monkeypatch.setattr(
        action_cell_aat,
        "refresh_action_valid_instances",
        lambda context: None,
    )
    model = _model()
    model.params = SimpleNamespace(
        timing_surrogate_mode="surrogate_only",
        timing_objective_lane="timing_slew_cap",
        critical_endpoint_pruning_mode="off",
        timing_action_pin_cap_gradient=0,
        export_pin_violation_csv=False,
    )
    model.placedb = SimpleNamespace(gr_sizing=None)
    model.invoke_timing_count = 0
    model.pin2libpin_flat_ids = torch.tensor([0])
    model.inst_pins_mask = torch.tensor([True])
    model.output_pin_mask = torch.tensor([True])
    model._should_write_timing_obj_profile = lambda iteration: False
    model._should_write_timing_artifacts = lambda iteration: False
    model._should_write_pin_violation_detail = lambda iteration: False
    model._production_fast_loop_enabled = lambda: True
    model._timing_obj_profile_interval = lambda: 0
    model._should_use_active_cone_aux_pin_masks = lambda: False
    model._should_reuse_zero_slew_aux = lambda timing_op: False
    model._should_copy_pin_slack_after_timing = lambda enabled: False
    model._should_refresh_debug_pin_caps_base = lambda base, enabled: False
    model._timing_rc_cap_input_tensor = lambda tensor: tensor
    model._maybe_write_rc_root_load_probe = lambda **kwargs: None
    model._relaxed_buffer_timing_enabled = lambda: False
    model._should_compute_timing_aux_tensor = lambda name: False
    model._timing_geometry_for_pos = lambda pos: (pos, pos[:1], pos[1:])
    model.pin_caps_op = lambda offset, data: (
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
    )
    model.data_collections.inst_libcell_offset = torch.tensor([0])
    model.data_collections.net_flat_topo_sort = torch.tensor([0])
    model.data_collections.net_flat_topo_sort_start = torch.tensor([0, 1])
    model.data_collections.pin_fa = torch.tensor([0])
    model.data_collections.flat_pin_to_start = torch.tensor([0, 1])
    model.data_collections.flat_pin_to = torch.tensor([0])
    model.data_collections.flat_pin_from = torch.tensor([0])

    timing = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(timing)
    timing.log_top_violations = False
    timing.pin_rtran_live = torch.tensor([130.0], dtype=torch.float64)
    timing.pin_ftran_live = torch.tensor([125.0], dtype=torch.float64)
    timing.pin_net_cap_rise_live = torch.tensor([0.05], dtype=torch.float64)
    timing.pin_net_cap_fall_live = torch.tensor([0.049], dtype=torch.float64)
    timing.forward = lambda delays, impulses, loads, **kwargs: (
        torch.tensor(-10.0, dtype=torch.float64),
        torch.tensor(-20.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    z = torch.tensor([0.0], dtype=torch.float64)
    model.op_collections = SimpleNamespace(
        elmore_delay_op=lambda *args: (z, z, z, z, z, z),
        timing_propagation_op=timing,
    )
    context = model._active_discrete_action_context
    context.beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    model.timing_obj(
        torch.zeros(2, dtype=torch.float64),
        surrogate_mode="lut_only",
        discrete_action_context=context,
    )
    assert model._active_timing_surrogate_mode is None
    grad = torch.autograd.grad(
        timing.last_total_slew_violation_tensor + timing.last_total_cap_violation_tensor,
        context.beta,
    )[0]
    torch.testing.assert_close(
        grad,
        torch.tensor([[-40.0, 30.0, 15.0, -20.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )

