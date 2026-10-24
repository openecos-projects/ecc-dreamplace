"""Fused level aggregation preserves each lane's values and gradient ownership."""

from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_lanes_native import _TimingLanes
from dreamplace.ops.timing_propagation.timing_propagation import scatter_timing_max


@pytest.fixture(autouse=True)
def deterministic_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _reference(index, smooth, tau, *values):
    return tuple(
        scatter_timing_max(
            dest,
            index,
            src,
            mode="smooth" if smooth and lane < 2 else "hard",
            tau_ps=tau,
        )
        for lane, (dest, src) in enumerate(zip(values[:4], values[4:], strict=True))
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("selected", [(0, 1, 2, 3), (0,), (2,), (0, 1)])
@pytest.mark.parametrize("empty", [False, True])
def test_fused_lanes_match_reference_with_strided_vectors_and_unused_outputs(
    native_reference,
    native_device,
    native_module,
    dtype,
    smooth,
    selected,
    empty,
):
    assert hasattr(native_module, "timing_lanes_forward")
    index = torch.tensor(
        [] if empty else [0, 3, 1, 3, 0, 3], device=native_device, dtype=torch.int32
    )
    generator = torch.Generator(device=native_device).manual_seed(3000)
    rows = 0 if empty else 5
    dest = torch.rand((rows, 8), device=native_device, dtype=dtype, generator=generator)
    src = torch.rand((index.numel(), 8), device=native_device, dtype=dtype, generator=generator)
    results = []
    for fn in (_reference, _TimingLanes.apply):
        with native_reference() if fn is _reference else nullcontext():
            dest_copy, src_copy = dest.clone(), src.clone()
            values = tuple(
                matrix[:, lane].requires_grad_(True)
                for matrix in (dest_copy, src_copy)
                for lane in range(4)
            )
            outputs = fn(index, smooth, 2.0, *values)
            loss = sum((lane + 1) * outputs[lane].sum() for lane in selected)
            gradients = torch.autograd.grad(loss, values, allow_unused=True)
            results.append((tuple(value.detach() for value in outputs), gradients))
    torch.testing.assert_close(results[1], results[0], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("value", [0.0, -float("inf"), float("inf"), float("nan")])
@pytest.mark.parametrize("selected", [(0, 1, 2, 3), (0,), (2,)])
@pytest.mark.parametrize("upstream", [0.0, 3.0, float("inf")])
def test_fused_lanes_preserve_ties_nonfinite_values_and_adjoints(
    native_reference, native_device, smooth, value, selected, upstream
):
    index = torch.tensor([0, 0, 1], device=native_device)
    results = []
    for fn in (_reference, _TimingLanes.apply):
        with native_reference() if fn is _reference else nullcontext():
            # Unused lanes must remain outside the adjoint, including NaN inputs.
            values = tuple(
                torch.tensor(
                    [value, value, -float("inf")] if lane < 4 else [value] * 3,
                    device=native_device,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                for lane in range(8)
            )
            outputs = fn(index, smooth, 2.0, *values)
            gradients = torch.autograd.grad(
                tuple(outputs[lane] for lane in selected),
                values,
                grad_outputs=tuple(torch.full_like(outputs[lane], upstream) for lane in selected),
                allow_unused=True,
            )
            results.append((tuple(value.detach() for value in outputs), gradients))
    torch.testing.assert_close(results[1], results[0], equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("smooth", [False, True])
def test_fused_lanes_obey_stream_and_finite_differences(native_device, smooth):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        index = torch.tensor([0, 1, 1], device=native_device)
        values = tuple(
            torch.rand(3, device=native_device, dtype=torch.float64, requires_grad=True)
            for _ in range(8)
        )
        assert torch.autograd.gradcheck(
            lambda *v: _TimingLanes.apply(index, smooth, 2.0, *v),
            values,
            atol=1e-6,
            rtol=1e-3,
        )
    if stream is not None:
        stream.synchronize()


@pytest.mark.parametrize("mutation", ["index", "dest", "src", "output"])
def test_fused_lanes_reject_inplace_state_changes(native_device, mutation):
    index = torch.tensor([0, 1, 1], device=native_device, dtype=torch.int32)
    values = tuple(torch.rand(3, device=native_device, requires_grad=True) for _ in range(8))
    outputs = _TimingLanes.apply(index, True, 2.0, *values)
    with torch.no_grad():
        targets = {"index": index, "dest": values[0], "src": values[4], "output": outputs[0]}
        target = targets[mutation]
        target[0] += 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        outputs[0].sum().backward()
