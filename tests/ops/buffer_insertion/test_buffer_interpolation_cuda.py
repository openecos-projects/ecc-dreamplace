"""CUDA interpolation retains values, diagnostics and every input derivative."""

from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.buffer_insertion.buffer_interpolation import (
    interpolate_buffer_values_by_bsu_index,
)


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_interpolation_mode(request):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(request.param)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _interpolate(index, tables):
    return interpolate_buffer_values_by_bsu_index(
        index,
        per_size_input_cap=tables[0],
        per_size_delay=tables[1],
        per_size_output_slew=tables[2],
        coordinate_source="test_coordinates",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "index_grad,table_grads",
    [(True, (True, True, True)), (True, (False, False, False)), (False, (False, True, False))],
)
@pytest.mark.parametrize("empty", [False, True])
def test_cuda_interpolation_matches_cpu_objects_and_gradient_ownership(
    native_device, native_module, native_reference, dtype, index_grad, table_grads, empty
):
    assert hasattr(native_module, "buffer_interpolate_forward")
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            raw = [] if empty else [-float("inf"), -1, 0, 0.25, 1, 1.75, 3, 4, float("inf")]
            index = torch.tensor(raw, device=device, dtype=dtype, requires_grad=index_grad)
            tables = tuple(
                (
                    torch.arange(len(raw) * 8, device=device, dtype=dtype).reshape(len(raw), 8)
                    * 0.1
                    + field
                )[:, ::2].requires_grad_(requires)
                for field, requires in enumerate(table_grads)
            )
            output = _interpolate(index, tables)
            floating = [
                output[key]
                for key in (
                    "buffer_input_cap",
                    "buffer_delay",
                    "buffer_output_slew",
                    "bsu_index",
                    "bsu_index_alpha",
                )
            ]
            variables = [value for value in (index, *tables) if value.requires_grad]
            # The clipped/alpha outputs also remain differentiable exactly when the
            # index trains, without linking those diagnostics to trainable LUTs.
            loss = sum(
                (i + 1) * value.sum() for i, value in enumerate(floating) if value.requires_grad
            )
            gradients = torch.autograd.grad(loss, variables)
            results.append(
                (
                    {k: v.detach().cpu() for k, v in output.items() if torch.is_tensor(v)},
                    tuple(v.requires_grad for v in floating),
                    tuple(v.cpu() for v in gradients),
                    output["bsu_index_status"],
                )
            )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    assert (results[1][1], results[1][3]) == (results[0][1], results[0][3])
    torch.testing.assert_close(
        (results[1][0], results[1][2]),
        (results[0][0], results[0][2]),
        rtol=tolerance,
        atol=tolerance,
    )


def test_interpolation_uses_active_stream_and_passes_finite_differences(
    native_device,
):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        index = torch.tensor(
            [0.25, 1.6], device=native_device, dtype=torch.float64, requires_grad=True
        )
        tables = tuple(
            torch.rand((2, 4), device=native_device, dtype=torch.float64, requires_grad=True)
            for _ in range(3)
        )

        def fn(i, *values):
            result = _interpolate(i, values)
            return tuple(
                result[k] for k in ("buffer_input_cap", "buffer_delay", "buffer_output_slew")
            )

        assert torch.autograd.gradcheck(fn, (index, *tables), atol=1e-6, rtol=1e-3)
    if stream is not None:
        stream.synchronize()


@pytest.mark.parametrize("raw", [-1.0, 1.0, 1.25, 4.0])
def test_interpolation_nonfinite_upstream_gradients_match_reference(
    native_device, native_reference, raw
):
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            index = torch.tensor([raw], device=device, dtype=torch.float64, requires_grad=True)
            table = torch.tensor(
                [[1, 2, 4, 7]], device=device, dtype=torch.float64, requires_grad=True
            )
            output = _interpolate(index, (table, table, table))["buffer_delay"]
            grads = torch.autograd.grad(
                output, (index, table), grad_outputs=torch.full_like(output, torch.inf)
            )
            results.append(tuple(v.cpu() for v in grads))
    torch.testing.assert_close(results[1], results[0], equal_nan=True)


@pytest.mark.parametrize("field", [0, 1, 2])
def test_nonfinite_table_rejected_before_timing_consumes_it(native_device, field):
    index = torch.tensor([0.25], device=native_device)
    tables = [torch.ones((1, 3), device=native_device) for _ in range(3)]
    tables[field][0, 2] = torch.nan
    name = ("per_size_input_cap", "per_size_delay", "per_size_output_slew")[field]
    with pytest.raises(ValueError, match=f"{name} contains non-finite values"):
        _interpolate(index, tables)


def test_saved_inputs_reject_inplace_table_edits(native_device):
    index = torch.tensor([0.25], device=native_device, requires_grad=True)
    table = torch.ones((1, 3), device=native_device, requires_grad=True)
    result = _interpolate(index, (table, table, table))
    with torch.no_grad():
        table[0, 0] += 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(result["buffer_delay"].sum(), (index, table))
