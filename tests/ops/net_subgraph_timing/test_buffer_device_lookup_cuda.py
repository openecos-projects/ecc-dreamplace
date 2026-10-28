from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.net_subgraph_timing.segment_transfer.batched import (
    _lookup_device_per_size,
    lookup_buffer_device_per_size,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.schema import BufferDeviceLut
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path


def _torch_lookup(tables, slew, load):
    return _lookup_device_per_size(
        input_slew=slew,
        output_load=load,
        buffer_input_cap_by_size=tables["input_cap_by_size"],
        buffer_slew_axis=tables["input_slew_axis"],
        buffer_load_axis=tables["output_load_axis"],
        buffer_delay_lut=tables["delay_lut"],
        buffer_output_slew_lut=tables["output_slew_lut"],
    )


def _device(dtype, slew_count=3, load_count=2):
    generator = torch.Generator().manual_seed(32)
    shape = (4, slew_count, load_count)
    return BufferDeviceLut(
        input_cap_by_size=torch.tensor([0.01, 0.02, 0.04, 0.08], dtype=dtype),
        input_slew_axis=torch.linspace(0.2, 2, slew_count, dtype=dtype),
        output_load_axis=torch.linspace(0.1, 1, load_count, dtype=dtype),
        delay_lut=torch.rand(shape, generator=generator, dtype=dtype) + 0.5,
        output_slew_lut=torch.rand(shape, generator=generator, dtype=dtype) + 0.5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis_counts", [(1, 1), (1, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("count", [0, 1, 6])
def test_buffer_cuda_lookup_matches_all_strengths_and_coordinate_gradients(
    native_device, dtype, axis_counts, count
):
    device = _device(dtype, *axis_counts)
    results = []
    for fused in (False, True):
        tables = device.tensors(dtype=dtype, device=native_device)
        slew = torch.linspace(0, 3, count, device=native_device, dtype=dtype, requires_grad=True)
        load = torch.linspace(0, 2, count, device=native_device, dtype=dtype, requires_grad=True)
        values = (
            lookup_buffer_device_per_size(device, input_slew=slew, output_load=load, tables=tables)
            if fused
            else _torch_lookup(tables, slew, load)
        )
        if fused:
            assert "indexed_buffer_lookup" in tables
        weights = torch.arange(count * 4, device=native_device, dtype=dtype).reshape(count, 4) + 0.3
        loss = (
            values["per_size_delay"] * weights + values["per_size_output_slew"] * weights.flip(1)
        ).sum()
        loss = loss + (slew.sum() + load.sum()) * 0
        gradients = torch.autograd.grad(loss, (slew, load))
        results.append((values, gradients))
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(results[1], results[0], rtol=tolerance, atol=tolerance)


def test_buffer_cuda_lookup_refreshes_mutated_tables_and_preserves_live_table_gradients(
    native_device,
):
    device = _device(torch.float64)
    tables = device.tensors(dtype=torch.float64, device=native_device)
    slew = torch.tensor([0.6, 1.3], device=native_device, dtype=torch.float64, requires_grad=True)
    load = torch.tensor([0.3, 0.7], device=native_device, dtype=torch.float64, requires_grad=True)
    for live in (False, True):
        with buffer_timing_fast_path():
            lookup_buffer_device_per_size(device, input_slew=slew, output_load=load, tables=tables)
            tables["delay_lut"].add_(0.7)
            tables["delay_lut"].requires_grad_(live)
            actual = lookup_buffer_device_per_size(
                device, input_slew=slew, output_load=load, tables=tables
            )
        expected = _torch_lookup(tables, slew, load)
        variables = (slew, load, tables["delay_lut"]) if live else (slew, load)
        actual_grad = torch.autograd.grad(actual["per_size_delay"].sum(), variables)
        expected_grad = torch.autograd.grad(expected["per_size_delay"].sum(), variables)
        torch.testing.assert_close(
            (actual, actual_grad), (expected, expected_grad), rtol=1e-10, atol=1e-10
        )


def test_buffer_cuda_lookup_finite_differences_and_non_default_stream(
    native_device,
):
    device = _device(torch.float64)
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with (
        torch.cuda.stream(stream) if stream is not None else nullcontext(),
        buffer_timing_fast_path(),
    ):
        tables = device.tensors(dtype=torch.float64, device=native_device)

        def lookup(slew, load):
            values = lookup_buffer_device_per_size(
                device, input_slew=slew, output_load=load, tables=tables
            )
            return values["per_size_delay"], values["per_size_output_slew"]

        assert torch.autograd.gradcheck(
            lookup,
            (
                torch.tensor(
                    [0.6, 1.3], device=native_device, dtype=torch.float64, requires_grad=True
                ),
                torch.tensor(
                    [0.3, 0.7], device=native_device, dtype=torch.float64, requires_grad=True
                ),
            ),
            atol=1e-6,
            rtol=1e-3,
        )
    if stream is not None:
        stream.synchronize()
