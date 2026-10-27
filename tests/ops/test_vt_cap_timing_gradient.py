from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
from dreamplace.PlaceObj import PlaceObj


class _CellDelay(torch.nn.Module):
    def forward(self, main_id, arc_offset, vt, size, slew, cap, arc_types=None):
        return 2.0 * vt + 8.0 * cap + 0.5 * slew


def test_sizing_pin_caps_keep_float64_values_and_vt_derivatives():
    obj = PlaceObj.__new__(PlaceObj)
    obj.params = SimpleNamespace(production_fast_loop=True)
    logits = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    caps = torch.tensor([0.01, 0.03], dtype=torch.float64)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        inst_main_id=torch.tensor([0]),
        inst_is_sizeable=torch.tensor([True]),
        pin_2_libpin_offset=torch.tensor([0]),
        main_id_2_cell_id_start=torch.tensor([0, 2]),
        flat_libcell_info=torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]),
        cell_id_2_libpin_id_start=torch.tensor([0, 1]),
        flat_lib_pin_cap=caps,
        flat_lib_pin_rcap=caps,
        flat_lib_pin_fcap=caps,
        get_size_var=lambda: torch.ones(1, dtype=torch.float64),
        get_vt_var=lambda: torch.softmax(logits, dim=1),
    )
    result = obj._apply_sizing_driven_pin_caps(
        *(torch.zeros(1, dtype=torch.float64) for _ in range(3)), data
    )
    torch.testing.assert_close(
        result,
        (torch.tensor([0.02], dtype=torch.float64),) * 3,
        atol=1e-12,
        rtol=0,
    )
    gradient = torch.autograd.grad(sum(value.sum() for value in result), logits)[0]
    torch.testing.assert_close(
        gradient,
        torch.tensor([[-0.015, 0.015]], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("variable_slew", [False, True])
def test_fast_loop_vt_gradient_includes_cell_load_slew_and_rc(variable_slew):
    obj = PlaceObj.__new__(PlaceObj)
    obj.params = SimpleNamespace(production_fast_loop=True)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        inst_main_id=torch.tensor([0]),
        inst_is_sizeable=torch.tensor([True]),
        pin_2_libpin_offset=torch.tensor([0]),
        main_id_2_cell_id_start=torch.tensor([0, 2]),
        flat_libcell_info=torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]),
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
            torch.tensor([0]),
            slew,
            cap,
            torch.tensor([0]),
            torch.tensor([0]),
        )
        return (delay + 0.25 * obj._timing_rc_cap_input_tensor(cap)).sum()

    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    gradient = torch.autograd.grad(evaluate(logits, variable_slew=variable_slew), logits)[0]
    # p=softmax(logits)[1], dp/dlogit_1=1/4 at zero. Cell delay,
    # input-cap-driven load, wire RC, and optional slew all depend on p.
    expected = (2.0 + 8.0 * 0.02 + 0.25 * 0.02 + 0.5 * 0.04 * variable_slew) / 4
    torch.testing.assert_close(gradient, torch.tensor([[-expected, expected]]))
    epsilon = 1.0e-3
    plus = torch.tensor([[0.0, epsilon]])
    minus = torch.tensor([[0.0, -epsilon]])
    finite_diff = (
        evaluate(plus, variable_slew=variable_slew) - evaluate(minus, variable_slew=variable_slew)
    ).item() / (2 * epsilon)
    assert gradient[0, 1].item() == pytest.approx(finite_diff, abs=1e-4)
