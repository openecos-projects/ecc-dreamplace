import time

import pytest
import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    batched_segment_transfer,
    segment_transfer_forward_native_cuda,
    segment_transfer_forward_native_cpu,
    segment_transfer_native_recompute_autograd,
)
from test_segment_transfer_batched import (
    _batched_args,
    _fixture,
)


def _assert_outputs_close(left, right, *, atol=1e-7, rtol=1e-7):
    for key in (
        "upstream_visible_load",
        "segment_delay",
        "output_arrival",
        "output_slew",
        "first_buffer_input_slew",
        "first_buffer_output_load",
        "first_buffer_delay",
        "first_buffer_output_slew",
    ):
        torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol)


def test_native_cpu_forward_matches_python_batched_random_counts():
    fixture = _fixture(sample_count=128, nmax=4, dtype=torch.float64)
    args = _batched_args(fixture)

    reference = batched_segment_transfer(**args)
    native = segment_transfer_forward_native_cpu(**args)

    _assert_outputs_close(native, reference)


def test_native_cpu_forward_extrapolates_out_of_range_load():
    fixture = _fixture(sample_count=4, nmax=1, dtype=torch.float64)
    args = _batched_args(fixture)
    args["input_arrival"] = torch.zeros(4, dtype=torch.float64)
    args["input_slew"] = torch.zeros(4, dtype=torch.float64)
    args["downstream_load"] = torch.full((4,), 15.0, dtype=torch.float64)
    args["edge_resistance"] = torch.zeros(4, dtype=torch.float64)
    args["edge_capacitance"] = torch.zeros(4, dtype=torch.float64)
    args["repeater_count"] = torch.ones(4, dtype=torch.long)
    args["split_fractions"] = torch.full((4, 2), 0.5, dtype=torch.float64)
    args["bsu_index"] = torch.zeros(4, dtype=torch.float64)
    args["upstream_retained_cap"] = torch.zeros(4, dtype=torch.float64)

    native = segment_transfer_forward_native_cpu(**args)

    torch.testing.assert_close(
        native["first_buffer_delay"],
        torch.full((4,), 1.6, dtype=torch.float64),
    )
    torch.testing.assert_close(
        native["first_buffer_output_slew"],
        torch.full((4,), 0.8, dtype=torch.float64),
    )


def test_native_recompute_autograd_matches_python_gradients():
    reference_fixture = _fixture(sample_count=24, nmax=3, dtype=torch.float64, requires_grad=True)
    native_fixture = {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        if torch.is_tensor(value) and value.is_floating_point()
        else value.detach().clone()
        if torch.is_tensor(value)
        else value
        for key, value in reference_fixture.items()
        if key != "buffer_device"
    }
    reference_args = _batched_args(reference_fixture)
    native_args = _batched_args(native_fixture)

    reference = batched_segment_transfer(**reference_args)
    native = segment_transfer_native_recompute_autograd(**native_args)
    _assert_outputs_close(native, reference)

    reference_loss = (
        reference["segment_delay"].sum()
        + 0.7 * reference["output_slew"].sum()
        + 0.3 * reference["upstream_visible_load"].sum()
    )
    native_loss = (
        native["segment_delay"].sum()
        + 0.7 * native["output_slew"].sum()
        + 0.3 * native["upstream_visible_load"].sum()
    )
    reference_loss.backward()
    native_loss.backward()

    for key in (
        "input_slew",
        "downstream_load",
        "edge_resistance",
        "edge_capacitance",
        "bsu_index",
        "upstream_retained_cap",
    ):
        torch.testing.assert_close(
            native_fixture[key].grad,
            reference_fixture[key].grad,
            atol=1e-7,
            rtol=1e-7,
        )


def test_native_cpu_rejects_cuda_tensor_cleanly():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    fixture = _fixture(sample_count=4, nmax=1, dtype=torch.float64)
    args = _batched_args(fixture)
    args["input_arrival"] = args["input_arrival"].cuda()
    with pytest.raises(RuntimeError, match="CPU tensor"):
        segment_transfer_forward_native_cpu(**args)


def test_native_cpu_forward_speed_smoke_beats_python_batched():
    fixture = _fixture(sample_count=8192, nmax=4, dtype=torch.float32)
    args = _batched_args(fixture)
    for _ in range(3):
        batched_segment_transfer(**args)
        segment_transfer_forward_native_cpu(**args)

    start = time.perf_counter()
    for _ in range(10):
        batched_segment_transfer(**args)
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        segment_transfer_forward_native_cpu(**args)
    native_time = time.perf_counter() - start

    assert native_time < python_time


def _to_cuda_args(args):
    return {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in args.items()
    }


def test_cuda_native_forward_matches_python_and_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    fixture = _fixture(sample_count=512, nmax=4, dtype=torch.float32)
    args = _batched_args(fixture)
    cuda_args = _to_cuda_args(args)

    reference = batched_segment_transfer(**args)
    cpu = segment_transfer_forward_native_cpu(**args)
    cuda = {
        key: value.cpu()
        for key, value in segment_transfer_forward_native_cuda(**cuda_args).items()
    }

    _assert_outputs_close(cuda, reference, atol=3e-6, rtol=3e-6)
    _assert_outputs_close(cuda, cpu, atol=3e-6, rtol=3e-6)


def test_cuda_native_forward_speed_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    fixture = _fixture(sample_count=50000, nmax=4, dtype=torch.float32)
    args = _to_cuda_args(_batched_args(fixture))
    for _ in range(3):
        segment_transfer_forward_native_cuda(**args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        segment_transfer_forward_native_cuda(**args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert elapsed > 0.0
