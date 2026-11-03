"""Double placement metadata must survive the real-size pin-cap transport."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.PlaceObj import PlaceObj


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_size_pin_caps_keep_metadata_dtype_and_gradients(dtype):
    obj = PlaceObj.__new__(PlaceObj)
    obj.params = SimpleNamespace(production_fast_loop=True, timing_surrogate_mode="auto")
    size = torch.tensor([1.5], dtype=dtype, requires_grad=True)
    vt = torch.tensor([[0.0, 0.0]], dtype=dtype, requires_grad=True)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        inst_main_id=torch.tensor([0]),
        inst_is_sizeable=torch.tensor([True]),
        pin_2_libpin_offset=torch.tensor([0]),
        main_id_2_cell_id_start=torch.tensor([0, 2]),
        flat_libcell_info=torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0]], dtype=dtype),
        cell_id_2_libpin_id_start=torch.tensor([0, 1]),
        flat_lib_pin_cap=torch.tensor([1.0, 4.0], dtype=dtype),
        flat_lib_pin_rcap=torch.tensor([1.0, 4.0], dtype=dtype),
        flat_lib_pin_fcap=torch.tensor([1.0, 4.0], dtype=dtype),
        get_size_var=lambda: size,
        get_vt_var=lambda: vt.softmax(dim=1),
    )
    caps = obj._apply_sizing_driven_pin_caps(*(torch.zeros(1, dtype=dtype) for _ in range(3)), data)
    assert all(value.dtype == dtype for value in caps)
    assert all(1.0 < float(value.detach()[0]) < 4.0 for value in caps)
    grads = torch.autograd.grad(sum(value.sum() for value in caps), (size, vt))
    assert all(bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0) for value in grads)
