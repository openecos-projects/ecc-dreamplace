"""The public piecewise model preserves timing precision through native LUTs."""

from types import SimpleNamespace

import pytest
import torch

from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
from dreamplace.ops.timing_propagation.timing_propagation import LUTS_INFO


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("selected_types", [None, [0, 0], [0, 3]])
def test_public_piecewise_model_preserves_values_and_coordinate_gradients(
    device, dtype, selected_types
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Four legal masters encode 10 + 20*(size-1) + 6*vt + 2*slew + 3*cap.
    lut = LUTS_INFO(
        torch.tensor([[b, b + 3, b + 2, b + 5] for b in (10, 30, 16, 36)],
                     dtype=dtype, device=device),
        torch.tensor([[0.0, 1.0]] * 4, dtype=dtype, device=device),
        torch.tensor([[0.0, 1.0]] * 4, dtype=dtype, device=device),
        torch.tensor([[2, 2]] * 4, device=device),
    )
    model = CellModeling(SimpleNamespace(
        device=torch.device(device),
        flat_libcell_info=torch.tensor(
            [[0, 0, 1, 0], [1, 0, 2, 0], [2, 0, 1, 1], [3, 0, 2, 1]],
            device=device,
        ),
        flat_libarc_info=torch.tensor(
            [[0, 1, i, 0, 1, 0] for i in range(4)], device=device
        ),
        arcs_info=SimpleNamespace(**{name: lut for name in (
            "r_delay_luts", "f_delay_luts", "r_trans_luts", "f_trans_luts",
        )}),
    ))
    # Sizing coordinates may be float32 even when timing is float64.
    size = torch.full((2,), 1.5, device=device, requires_grad=True)
    vt = torch.full((2,), 0.5, device=device, requires_grad=True)
    slew = torch.full((2,), 0.25, dtype=dtype, device=device, requires_grad=True)
    cap = torch.full((2,), 0.5, dtype=dtype, device=device, requires_grad=True)
    indices = torch.zeros(2, dtype=torch.long, device=device)
    arc_types = (None if selected_types is None else
                 torch.tensor(selected_types, device=device))
    result = model(indices, indices, vt, size, slew, cap, arc_types=arc_types)
    values = list(result.values()) if isinstance(result, dict) else [result]
    expected = torch.full((2,), 25.0, dtype=dtype, device=device)
    torch.testing.assert_close(values, [expected] * len(values), atol=1e-6, rtol=0)
    gradients = torch.autograd.grad(sum(value.sum() for value in values),
                                    (size, vt, slew, cap))
    torch.testing.assert_close(
        gradients,
        tuple(torch.full_like(coordinate, slope * len(values))
              for coordinate, slope in zip((size, vt, slew, cap), (20, 6, 2, 3))),
        atol=1e-6, rtol=0,
    )
