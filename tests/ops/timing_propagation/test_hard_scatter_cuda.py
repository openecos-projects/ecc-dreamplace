"""Slew max values, ties and gradients against the unmodified Torch operator."""

from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path
from dreamplace.ops.timing_propagation.hard_scatter_native import scatter_slew_max


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_buffer_scope(request):
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with buffer_timing_fast_path() if request.param else nullcontext():
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        ([-100, 2, 42, 3], [10, 11, 2, 7, 4]),
        ([11, 7, 42, 4], [11, 11, 7, 7, 4]),
        ([-torch.inf, torch.inf, 42, 3], [-torch.inf, -torch.inf, torch.inf, torch.inf, 4]),
        ([torch.nan, 2, torch.nan, 3], [10, 11, torch.nan, 7, 4]),
    ],
)
def test_hard_scatter_preserves_torch_values_and_tied_gradients(
    native_device, dtype, include_self, values
):
    results = []
    for optimized in (False, True):
        dest = torch.tensor(values[0], dtype=dtype, device=native_device, requires_grad=True)
        src = torch.tensor(values[1], dtype=dtype, device=native_device, requires_grad=True)
        index = torch.tensor([0, 0, 1, 1, 3], device=native_device)
        out = (
            scatter_slew_max(dest, index, src, include_self=include_self)
            if optimized
            else torch.scatter_reduce(dest, 0, index, src, reduce="amax", include_self=include_self)
        )
        gradients = torch.autograd.grad(
            (out * out.new_tensor([1.3, -0.7, 2.1, -3])).sum(), (dest, src)
        )
        results.append((out, gradients))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("destinations", [0, 4])
def test_hard_scatter_empty_sources_keep_dest_values_and_gradients(
    native_device, include_self, destinations
):
    dest = torch.arange(destinations, device=native_device, dtype=torch.float64, requires_grad=True)
    src = torch.empty(0, device=native_device, dtype=torch.float64, requires_grad=True)
    index = torch.empty(0, device=native_device, dtype=torch.long)
    out = scatter_slew_max(dest, index, src, include_self=include_self)
    gradients = torch.autograd.grad(out.sum(), (dest, src))
    torch.testing.assert_close((out, gradients), (dest, (torch.ones_like(dest), src)))


@pytest.mark.parametrize("include_self", [False, True])
def test_hard_scatter_nondefault_stream_and_finite_difference(native_device, include_self):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        dest = torch.tensor(
            [0, 0.2, 3], device=native_device, dtype=torch.float64, requires_grad=True
        )
        src = torch.tensor(
            [0.3, 0.8, 0.7], device=native_device, dtype=torch.float64, requires_grad=True
        )
        index = torch.tensor([0, 0, 1], device=native_device)

        def fn(d, s):
            return scatter_slew_max(d, index, s, include_self=include_self)

        assert torch.autograd.gradcheck(fn, (dest, src), atol=1e-6, rtol=1e-3)
    if stream is not None:
        stream.synchronize()


def test_hard_scatter_keeps_inplace_version_checks_for_strided_sources(native_device):
    dest = torch.zeros(3, device=native_device, requires_grad=True)
    table = torch.ones((3, 2), device=native_device, requires_grad=True)
    source = table[:, 0]
    out = scatter_slew_max(dest, torch.arange(3, device=native_device), source)
    with torch.no_grad():
        table.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def test_hard_scatter_native_path_is_used_only_with_input_gradients(
    native_device, native_module, monkeypatch
):
    native = native_module
    forward = native.hard_scatter_forward
    calls = []

    def count_forward(*args):
        calls.append(1)
        return forward(*args)

    monkeypatch.setattr(native, "hard_scatter_forward", count_forward)
    dest = torch.zeros(3, device=native_device, requires_grad=True)
    src = torch.ones(2, device=native_device, requires_grad=True)
    index = torch.tensor([0, 1], device=native_device)
    scatter_slew_max(dest, index, src).sum().backward()
    with torch.no_grad():
        scatter_slew_max(dest, index, src)
    assert len(calls) == 1
