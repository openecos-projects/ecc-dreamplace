"""Hold checks use the real piecewise model and retain sparse global arc IDs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
from dreamplace.ops.timing_propagation.hold_check_model import HoldCheckModel
from dreamplace.ops.timing_propagation.timing_propagation import LUTS_INFO


@pytest.mark.parametrize("edge", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hold_check_interpolation_preserves_global_arc_ids_and_derivatives(edge, dtype):
    # The checks are global rows 1 and 3, with unrelated data arcs in between.
    # check(size=1)=10+2*clock_slew; check(size=2)=30+2*clock_slew.
    lut = LUTS_INFO(
        torch.tensor([[100.0] * 4, [10.0, 10.0, 12.0, 12.0],
                      [200.0] * 4, [30.0, 30.0, 32.0, 32.0]], dtype=dtype),
        torch.tensor([[0.0, 1.0]] * 4, dtype=dtype),
        torch.tensor([[0.0, 1.0]] * 4, dtype=dtype),
        torch.tensor([[2, 2]] * 4),
    )
    data = SimpleNamespace(
        device=torch.device("cpu"),
        flat_libcell_info=torch.tensor([[0, 0, 1, 0], [1, 0, 2, 0]]),
        flat_libarc_info=torch.tensor([
            [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 1, 0],
        ]),
        arcs_info=SimpleNamespace(**{name: lut for name in (
            "r_delay_luts", "f_delay_luts", "r_trans_luts", "f_trans_luts",
            "r_check_luts", "f_check_luts",
        )}),
    )
    model = CellModeling(data)
    size = torch.tensor([1.5], dtype=dtype, requires_grad=True)
    first = torch.tensor([0.25], dtype=dtype, requires_grad=True)
    second = torch.tensor([0.5], dtype=dtype)
    timing = SimpleNamespace(
        cell_modeling_op=model, device=data.device, arcs_info=data.arcs_info,
        pin2node_map=torch.tensor([0, 0]), inst_is_sizeable=torch.tensor([True]),
        inst_main_id=torch.tensor([0]),
        _get_surrogate_state=lambda: {"size_var": size, "vt_var": torch.tensor([0.0])},
        _vt_scalar=lambda value: value,
        lut_entry_vectorized=lambda _cells, clock, cap, arcs, _lut: model._lut_lookup(
            "r_min_check", arcs, clock, cap
        ),
    )
    check = HoldCheckModel(timing, np.array([0, 2, 0, 2]))
    # A second consumer reuses the registered dataset rather than registering twice.
    HoldCheckModel(timing, np.array([0, 2, 0, 2]))
    result = check.query(timing, torch.tensor([[0, 1, 0, 1]]), first, second, edge)
    gradient = torch.autograd.grad(result.sum(), (size, first))
    torch.testing.assert_close(
        (result, gradient),
        (torch.tensor([20.5], dtype=dtype),
         (torch.tensor([20.0], dtype=dtype), torch.tensor([2.0], dtype=dtype))),
        atol=1e-6, rtol=0,
    )
    ids = model.piecewise_state[check.names[edge]]["arc_index_table"]
    assert ids[ids >= 0].tolist() == [1, 3]
