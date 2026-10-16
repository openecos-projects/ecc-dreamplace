"""The indexed CUDA query preserves mixed-table and degenerate-axis derivatives."""

from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.timing_propagation import lut_entry_2d_op
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path
from dreamplace.ops.timing_propagation.timing_propagation import LUTS_INFO, TimingPropagation


@pytest.fixture(autouse=True, params=[False, True])
def fixed_buffer_timing_scope(request):
    with buffer_timing_fast_path() if request.param else nullcontext():
        yield


def _info(device, dtype):
    trans = torch.tensor([[0, 1, 2]] * 8, device=device, dtype=dtype)
    caps = trans.clone()
    trans[5, :2] = 0
    caps[6, :2] = 0
    trans[7, :2] = caps[7, :2] = 0
    values = torch.tensor(
        [
            [9, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 4, 0, 0, 0, 0, 0, 0],
            [1, 3, 6, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 7, 0, 0, 0, 0, 0],
            [1, 2, 3, 7, 0, 0, 0, 0, 0],
            [1, 2, 3, 7, 0, 0, 0, 0, 0],
            [1, 2, 3, 7, 0, 0, 0, 0, 0],
        ],
        device=device,
        dtype=dtype,
    )
    dims = torch.tensor(
        [[0, 0], [1, 1], [3, 0], [0, 3]] + [[2, 2]] * 4, device=device, dtype=torch.long
    )
    return LUTS_INFO(values, trans, caps, dims)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [-2.0, 0.0, 2.0])
def test_indexed_mixed_queries_match_python_values_and_input_gradients(
    native_device, native_reference, dtype, offset
):
    results = []
    for reference, device, mode in ((True, "cpu", "off"), (False, native_device, "on")):
        with native_reference() if reference else nullcontext():
            info = _info(device, dtype)
            model = TimingPropagation.__new__(TimingPropagation)
            model.timing_lut_2d_native_op = mode
            trans = torch.full((9,), 0.4 + offset, device=device, dtype=dtype, requires_grad=True)
            caps = torch.full((9,), 0.6 + offset, device=device, dtype=dtype, requires_grad=True)
            indices = torch.tensor([4, 2, 3, 1, 0, 5, 6, 7, -4], device=device)
            values = model.lut_entry_vectorized(indices, trans, caps, indices, info)
            grads = torch.autograd.grad(values.sum(), (trans, caps))
            results.append((values.detach().cpu(), tuple(grad.cpu() for grad in grads)))
    tolerance = 2e-5 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(results[1][0], results[0][0], atol=tolerance, rtol=tolerance)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_indexed_query_respects_the_current_cuda_stream(
    native_device,
):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        info = _info(native_device, torch.float64)
        model = TimingPropagation.__new__(TimingPropagation)
        model.timing_lut_2d_native_op = "on"
        trans = torch.full((2,), 0.4, device=native_device, dtype=torch.float64, requires_grad=True)
        caps = torch.full((2,), 0.6, device=native_device, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([5, 6], device=native_device)
        result = model.lut_entry_vectorized(indices, trans, caps, indices, info)
        grads = torch.autograd.grad(result.sum(), (trans, caps))
    if stream is not None:
        stream.synchronize()
    torch.testing.assert_close(result.cpu(), torch.tensor([1.6, 1.8], dtype=torch.float64))
    torch.testing.assert_close(grads[0].cpu(), torch.tensor([0.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(grads[1].cpu(), torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_native_degenerate_2d_gradients_match_linear_fallback(native_device):
    device = native_device
    info = _info(device, torch.float64)
    rows = torch.tensor([5, 6, 7], device=device)
    trans = torch.full((3,), 0.4, device=device, dtype=torch.float64, requires_grad=True)
    caps = torch.full((3,), 0.6, device=device, dtype=torch.float64, requires_grad=True)
    values = info.flat_luts_values[rows].detach().requires_grad_()
    result = lut_entry_2d_op.lut_entry_2d(
        trans,
        caps,
        info.flat_luts_trans_table[rows],
        info.flat_luts_cap_table[rows],
        values,
        info.trans_dims_actual[rows],
        info.cap_dims_actual[rows],
    )
    actual = torch.autograd.grad(result.sum(), (trans, caps, values))
    expected_values = torch.zeros_like(values)
    expected_values[0, :2] = values.new_tensor([0.4, 0.6])
    expected_values[1, [0, 2]] = values.new_tensor([0.6, 0.4])
    expected_values[2, 0] = 1
    expected = (
        torch.tensor([0, 2, 0], device=device, dtype=torch.float64),
        torch.tensor([1, 0, 0], device=device, dtype=torch.float64),
        expected_values,
    )
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target)
