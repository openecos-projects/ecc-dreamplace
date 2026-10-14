from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path
from dreamplace.ops.timing_propagation.timing_propagation import smooth_scatter_max_tau


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_scatter_mode(request):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(request.param)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.fixture(autouse=True, params=[False, True])
def fixed_buffer_timing_scope(request):
    with buffer_timing_fast_path() if request.param else nullcontext():
        yield


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("tau", [0.5, 2.0])
def test_cuda_scatter_matches_cpu_values_and_input_gradients(
    native_device, native_reference, dtype, include_self, tau
):
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            dest = torch.tensor([-100, 2, 42, 3], dtype=dtype, device=device, requires_grad=True)
            src = torch.tensor([10, 11, 2, 7, 4], dtype=dtype, device=device, requires_grad=True)
            index = torch.tensor([0, 0, 1, 1, 3], device=device)
            out = smooth_scatter_max_tau(dest, index, src, include_self=include_self, tau=tau)
            gradient = torch.autograd.grad((out * out.new_tensor([1, 2, 3, 4])).sum(), (dest, src))
            results.append((out.detach().cpu(), tuple(g.cpu() for g in gradient)))
    torch.testing.assert_close(results[1][0], results[0][0])
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_cuda_scatter_preserves_infinite_groups_and_tied_gradients(
    native_device, native_reference, include_self, sign
):
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            dest = torch.tensor(
                [sign * torch.inf, 42], dtype=torch.float64, device=device, requires_grad=True
            )
            src = torch.tensor(
                [sign * torch.inf, sign * torch.inf],
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            )
            index = torch.zeros(2, dtype=torch.long, device=device)
            out = smooth_scatter_max_tau(dest, index, src, include_self=include_self, tau=2.0)
            gradient = torch.autograd.grad(out.sum(), (dest, src))
            results.append((out.detach().cpu(), tuple(g.cpu() for g in gradient)))
    torch.testing.assert_close(results[1][0], results[0][0])
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected)


def test_cuda_scatter_obeys_nondefault_stream_and_finite_differences(
    native_device,
):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        dest = torch.tensor([0, 2], dtype=torch.float64, device=native_device, requires_grad=True)
        src = torch.tensor([3, 4, 5], dtype=torch.float64, device=native_device, requires_grad=True)
        index = torch.tensor([0, 0, 1], device=native_device)

        def fn(d, s):
            return smooth_scatter_max_tau(d, index, s, include_self=True, tau=2.0)

        assert torch.autograd.gradcheck(fn, (dest, src), atol=1e-6, rtol=1e-3)
    if stream is not None:
        stream.synchronize()


@pytest.mark.parametrize("include_self", [False, True])
def test_cuda_scatter_preserves_nan_group_values_and_gradients(
    native_device, native_reference, include_self
):
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            dest = torch.tensor(
                [torch.nan, 2, 42], dtype=torch.float64, device=device, requires_grad=True
            )
            src = torch.tensor(
                [1, torch.nan, 3], dtype=torch.float64, device=device, requires_grad=True
            )
            index = torch.tensor([0, 1, 1], device=device)
            out = smooth_scatter_max_tau(dest, index, src, include_self=include_self, tau=2.0)
            gradient = torch.autograd.grad(out.sum(), (dest, src))
            results.append((out.detach().cpu(), tuple(g.cpu() for g in gradient)))
    torch.testing.assert_close(results[1], results[0], equal_nan=True)


@pytest.mark.parametrize("include_self", [False, True])
def test_cuda_scatter_shared_groups_repeat_exactly_in_deterministic_mode(
    native_device, include_self
):
    if not torch.are_deterministic_algorithms_enabled():
        pytest.skip("requires deterministic algorithms")
    generator = torch.Generator(device=native_device).manual_seed(3000)
    index = torch.randint(0, 64, (4096,), device=native_device, generator=generator)
    dest = torch.rand(64, device=native_device, generator=generator, requires_grad=True)
    src = torch.rand(4096, device=native_device, generator=generator, requires_grad=True)

    def evaluate():
        output = smooth_scatter_max_tau(dest, index, src, include_self=include_self, tau=2.0)
        gradients = torch.autograd.grad(output.sum(), (dest, src))
        return output, *gradients

    reference = evaluate()
    for _ in range(20):
        actual = evaluate()
        assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
