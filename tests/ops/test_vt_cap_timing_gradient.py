from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
from dreamplace.PlaceObj import PlaceObj


class _CellDelay(torch.nn.Module):
    def forward(self, main_id, arc_offset, vt, size, slew, cap, arc_types=None):
        return 2.0 * vt + 8.0 * cap + 0.5 * slew


def test_fast_loop_vt_input_gradients_match_forward_only_when_chains_enabled(monkeypatch):
    obj = PlaceObj.__new__(PlaceObj)
    obj.params = SimpleNamespace(production_fast_loop=True)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        inst_main_id=torch.tensor([0]),
        inst_is_sizeable=torch.tensor([True]),
        pin_2_libpin_offset=torch.tensor([0]),
        main_id_2_cell_id_start=torch.tensor([0, 2]),
        flat_libcell_info=torch.tensor([[0., 0., 1., 0.], [1., 0., 1., 1.]]),
        cell_id_2_libpin_id_start=torch.tensor([0, 1]),
        flat_lib_pin_cap=torch.tensor([0.01, 0.03]),
        flat_lib_pin_rcap=torch.tensor([0.01, 0.03]),
        flat_lib_pin_fcap=torch.tensor([0.01, 0.03]),
    )
    data.get_size_var = lambda: torch.tensor([1.0])
    tp = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(tp)
    tp.production_fast_loop = True
    tp.device = torch.device("cpu")
    tp.dtype = torch.float32
    tp.cell_modeling_op = _CellDelay()
    tp.pin2node_map = torch.tensor([0])
    tp.inst_main_id = torch.tensor([0])
    tp.inst_is_sizeable = torch.tensor([True])
    tp.lib_arc_offsets = None
    tp.pin_net = torch.tensor([0])
    tp.size_var_getter = data.get_size_var
    data.get_vt_var = lambda: torch.softmax(data.vt_logits, dim=1)
    tp.vt_var_getter = data.get_vt_var
    tp.arcs_info = SimpleNamespace(r_delay_luts=object())
    tp.lut_entry_vectorized = lambda *args: torch.zeros_like(args[1])

    def evaluate(logits, *, variable_slew=False):
        data.vt_logits = logits
        cap, rcap, fcap = obj._apply_sizing_driven_pin_caps(
            *(torch.zeros(1) for _ in range(3)), data
        )
        torch.testing.assert_close(cap, rcap)
        torch.testing.assert_close(cap, fcap)
        slew = torch.tensor([0.1])
        if variable_slew:
            slew = slew + 0.04 * data.get_vt_var()[:, 1]
        delay = tp.r_delay_entry(
            torch.tensor([0]), slew, cap,
            torch.tensor([0]), torch.tensor([0]),
        )
        return (delay + 0.25 * obj._timing_rc_cap_input_tensor(cap)).sum()

    monkeypatch.delenv("AIMP_DETACH_TIMING_RC_CAP_INPUTS", raising=False)
    monkeypatch.delenv("AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS", raising=False)
    monkeypatch.delenv("AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS", raising=False)
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    default_grad = torch.autograd.grad(evaluate(logits), logits)[0][0, 1].item()
    assert default_grad > 0.0  # Direct VT delay and the RC-cap input proxy survive.

    monkeypatch.setenv("AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS", "1")
    full_grad = torch.autograd.grad(evaluate(logits), logits)[0][0, 1].item()
    epsilon = 1.0e-3
    plus = torch.tensor([[0.0, epsilon]])
    minus = torch.tensor([[0.0, -epsilon]])
    finite_diff = (evaluate(plus) - evaluate(minus)).item() / (2 * epsilon)

    assert full_grad - default_grad == pytest.approx(8.0 * 0.02 / 4, abs=1e-5)
    assert full_grad == pytest.approx(finite_diff, abs=1e-4)

    monkeypatch.delenv("AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS")
    default_slew_grad = torch.autograd.grad(
        evaluate(logits, variable_slew=True), logits
    )[0][0, 1].item()
    monkeypatch.setenv("AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS", "1")
    full_slew_grad = torch.autograd.grad(
        evaluate(logits, variable_slew=True), logits
    )[0][0, 1].item()
    monkeypatch.setenv("AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS", "1")
    complete_grad = torch.autograd.grad(
        evaluate(logits, variable_slew=True), logits
    )[0][0, 1].item()
    complete_finite_diff = (
        evaluate(plus, variable_slew=True) - evaluate(minus, variable_slew=True)
    ).item() / (2 * epsilon)

    assert full_slew_grad - default_slew_grad == pytest.approx(0.5 * 0.04 / 4, abs=1e-5)
    assert complete_grad == pytest.approx(complete_finite_diff, abs=1e-4)
