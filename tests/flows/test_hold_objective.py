from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.diff_opt_spec import TimingCoefficients
from dreamplace.flows.discrete_timing_common import timing_sizing_coefficients
from dreamplace.flows.hold_objective import HoldObjective
from dreamplace.flows.metric_state import combined_timing_loss, enrich_timing_record
from dreamplace.flows.timing_opt_config import normalize_timing_opt_coefficients
from dreamplace.flows.timing_phase_coefficients import timing_phase_coefficients


def test_named_hold_weights_extend_the_four_item_flow_contract():
    legacy = TimingCoefficients.from_value([100, 1, 2, 3])
    assert legacy.to_dict() == {"wns": 100.0, "tns": 1.0, "slew": 2.0, "cap": 3.0}
    joint = TimingCoefficients.from_value(legacy.to_dict() | {"hold_wns": 4.0, "hold_tns": 0.1})
    assert asdict(joint) == legacy.to_dict() | {"hold_wns": 4.0, "hold_tns": 0.1}
    with pytest.raises(ValueError, match="wns, tns, slew, cap"):
        TimingCoefficients.from_value([100, 1, 2, 3, 4, 0.1])


def test_fixed_phase_uses_its_own_hold_weights_and_restores_after_error():
    model = SimpleNamespace(
        timing_wns_coeff=1.0,
        timing_tns_coeff=0.01,
        timing_slew_weight=2.0,
        timing_cap_weight=3.0,
        timing_grad_balance_weight=0.2,
        hold_objective=HoldObjective(4.0, 0.1, 20.0),
    )
    before = vars(model).copy()
    fixed = {"wns": 500.0, "tns": 5.0, "slew": 1.0, "cap": 1.0, "hold_wns": 10.0, "hold_tns": 2.0}
    with (
        pytest.raises(RuntimeError, match="phase failure"),
        timing_phase_coefficients(model, fixed),
    ):
        assert model.hold_objective == HoldObjective(10.0, 2.0, 20.0)
        assert model.timing_grad_balance_weight == 1.0
        raise RuntimeError("phase failure")
    assert vars(model) == before
    with timing_phase_coefficients(
        model, {k: v for k, v in fixed.items() if not k.startswith("hold_")}
    ):
        assert model.hold_objective == HoldObjective(0, 0, 20.0)
    with timing_phase_coefficients(model) as values:
        assert model.hold_objective == before["hold_objective"]
        assert (values["hold_wns"], values["hold_tns"]) == (4.0, 0.1)


def test_hold_margin_converts_once_and_loss_retains_its_gradient():
    params = SimpleNamespace(
        timing_hold_enabled=True,
        with_sta=1,
        timing_hold_margin_ns=0.02,
        timing_hold_coefficients={"wns": 2.0, "tns": 0.1},
    )
    objective = HoldObjective.from_params(params)
    assert objective == HoldObjective(2.0, 0.1, 20.0)
    slack = torch.tensor([-5.0, 10.0, 50.0], requires_grad=True)
    hold = SimpleNamespace(
        deficits=lambda margin: (
            -(slack - margin).clamp(max=0).min(),
            -(slack - margin).clamp(max=0).sum(),
        )
    )
    loss = objective.loss(SimpleNamespace(hold=hold))
    torch.testing.assert_close(loss, torch.tensor(53.5))
    (gradient,) = torch.autograd.grad(loss, slack)
    torch.testing.assert_close(gradient, torch.tensor([-2.1, -0.1, 0.0]))


@pytest.mark.parametrize("mode", ["fixed", "inherit"])
@pytest.mark.parametrize("fail", [False, True])
def test_gp_sizing_window_applies_hold_loss_and_restores_live_gp(mode, fail):
    model = SimpleNamespace(
        timing_wns_coeff=1.0, timing_tns_coeff=0.01,
        timing_slew_weight=2.0, timing_cap_weight=3.0,
        timing_grad_balance_weight=0.2,
        hold_objective=HoldObjective(4.0, 0.1, 20.0),
        _timing_coeff_growth_multiplier=3.5,
    )
    before = vars(model).copy()
    fixed = {
        "wns": 500., "tns": 5., "slew": 1., "cap": 1.,
        "hold_wns": 10., "hold_tns": 2.,
    }
    policy = {"mode": mode} | (fixed if mode == "fixed" else {})
    params = SimpleNamespace(timing_opt_coefficients=normalize_timing_opt_coefficients(policy))
    slack = torch.tensor(-5., requires_grad=True)
    evaluation = SimpleNamespace(hold=SimpleNamespace(
        hold_wns=slack, deficits=lambda margin: (margin - slack, margin - slack),
    ))

    def window():
        with timing_sizing_coefficients(params, model) as effective:
            loss = model.hold_objective.loss(evaluation) * model.timing_grad_balance_weight
            (gradient,) = torch.autograd.grad(loss, slack)
            expected = (300., -12.) if mode == "fixed" else (20.5, -.82)
            assert (float(loss.detach()), float(gradient)) == pytest.approx(expected)
            assert effective == {
                "mode": mode,
                **(fixed if mode == "fixed" else {
                    "wns": 1., "tns": .01, "slew": 2., "cap": 3.,
                    "hold_wns": 4., "hold_tns": .1,
                }),
                "timing_grad_balance_weight": 1. if mode == "fixed" else .2,
            }
            if fail:
                raise RuntimeError("window failure")

    if fail:
        with pytest.raises(RuntimeError, match="window failure"):
            window()
    else:
        window()
    assert vars(model) == before


def test_report_only_hold_keeps_setup_backward_out_of_early_graph():
    setup = torch.tensor(3.0, requires_grad=True)
    early = torch.tensor(-2.0, requires_grad=True)
    hold = SimpleNamespace(hold_wns=early, deficits=lambda margin: (-early, -early))
    loss = setup + HoldObjective(0.0, 0.0, 0.0).loss(SimpleNamespace(hold=hold))
    gradients = torch.autograd.grad(loss, (setup, early), allow_unused=True)
    assert gradients == (torch.tensor(1.0), None)


def test_restore_metric_scores_the_current_joint_objective_and_preserves_it():
    hold = torch.tensor(-2.0)
    model = SimpleNamespace(
        hold_objective=HoldObjective(3.0, 0.0, 0.0),
        data_collections=SimpleNamespace(pos=[torch.zeros(1)]),
        op_collections=SimpleNamespace(timing_propagation_op=SimpleNamespace(
            last_evaluation=SimpleNamespace(setup_wns=torch.tensor(-10.0)),
        )),
        _timing_loss=lambda wns, tns, ws, ts: -wns - tns * .01 - hold * 3 + 5,
    )
    assert combined_timing_loss(-10., -100., model) == 22.
    hold.fill_(-4.)
    score = combined_timing_loss(-10., -100., model)
    assert enrich_timing_record({"wns": -10., "tns": -100., "combined_timing_loss": score}) == {
        "wns": -10., "tns": -100., "combined_timing_loss": 28.,
    }
    assert combined_timing_loss(-10., -100.) == pytest.approx(.11)
    model.hold_objective = HoldObjective(0, 0, 0)
    assert combined_timing_loss(-10., -100., model) == pytest.approx(.11)
