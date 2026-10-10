import random
import time

import pytest
import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    analytic_segment_transfer,
    batched_segment_transfer,
    lookup_buffer_device_fixed_size,
    lookup_buffer_device_per_size,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


def _random_split_fractions(rng, count, nmax):
    values = [0.0 for _ in range(nmax + 1)]
    if count == 0:
        values[0] = 1.0
        return values
    cuts = sorted(rng.uniform(0.02, 0.98) for _ in range(count))
    previous = 0.0
    for index, cut in enumerate(cuts):
        values[index] = cut - previous
        previous = cut
    values[count] = 1.0 - previous
    return values


def _fixture(sample_count=64, nmax=4, *, dtype=torch.float64, requires_grad=False):
    rng = random.Random(3109)
    counts = []
    split_rows = []
    for _ in range(sample_count):
        count = rng.randint(0, nmax)
        counts.append(count)
        split_rows.append(_random_split_fractions(rng, count, nmax))
    device = default_fake_buffer_device(dtype=dtype)
    tables = device.tensors(dtype=dtype, device=torch.device("cpu"))

    def vector(low, high):
        values = torch.tensor(
            [rng.uniform(low, high) for _ in range(sample_count)],
            dtype=dtype,
        )
        return values.requires_grad_(requires_grad)

    return {
        "input_arrival": vector(0.0, 100.0),
        "input_slew": vector(0.1, 15.0),
        "downstream_load": vector(0.05, 8.0),
        "edge_resistance": vector(0.01, 6.0),
        "edge_capacitance": vector(0.0, 2.0),
        "repeater_count": torch.tensor(counts, dtype=torch.long),
        "split_fractions": torch.tensor(split_rows, dtype=dtype),
        "bsu_index": vector(0.0, 1.0),
        "upstream_retained_cap": vector(0.0, 0.5),
        "buffer_input_cap_by_size": tables["input_cap_by_size"],
        "buffer_slew_axis": tables["input_slew_axis"],
        "buffer_load_axis": tables["output_load_axis"],
        "buffer_delay_lut": tables["delay_lut"],
        "buffer_output_slew_lut": tables["output_slew_lut"],
        "buffer_device": device,
    }


def _batched_args(fixture):
    return {
        key: fixture[key]
        for key in (
            "input_arrival",
            "input_slew",
            "downstream_load",
            "edge_resistance",
            "edge_capacitance",
            "repeater_count",
            "split_fractions",
            "bsu_index",
            "upstream_retained_cap",
            "buffer_input_cap_by_size",
            "buffer_slew_axis",
            "buffer_load_axis",
            "buffer_delay_lut",
            "buffer_output_slew_lut",
        )
    }


def _reference_rows(fixture):
    rows = []
    sample_count = int(fixture["input_arrival"].numel())
    for index in range(sample_count):
        count = int(fixture["repeater_count"][index].item())
        transfer_input = SegmentTransferInput(
            input_arrival=fixture["input_arrival"][index],
            input_slew=fixture["input_slew"][index],
            downstream_load=fixture["downstream_load"][index],
            edge_resistance=fixture["edge_resistance"][index],
            edge_capacitance=fixture["edge_capacitance"][index],
            upstream_retained_capacitance=fixture["upstream_retained_cap"][index],
            repeater_count=count,
            bsu_index=fixture["bsu_index"][index],
            split_fractions=tuple(
                float(value)
                for value in fixture["split_fractions"][index, : count + 1].tolist()
            ),
            buffer_device=fixture["buffer_device"] if count > 0 else None,
        )
        rows.append(analytic_segment_transfer(transfer_input))
    return {
        "upstream_visible_load": torch.stack([row.upstream_visible_load for row in rows]),
        "segment_delay": torch.stack([row.segment_delay for row in rows]),
        "output_arrival": torch.stack([row.output_arrival for row in rows]),
        "output_slew": torch.stack([row.output_slew for row in rows]),
        "first_buffer_input_slew": torch.stack(
            [
                row.diagnostics.get(
                    "buffer_input_slew",
                    torch.zeros((), dtype=fixture["input_arrival"].dtype),
                )
                for row in rows
            ]
        ),
        "first_buffer_delay": torch.stack(
            [
                row.diagnostics.get(
                    "buffer_delay",
                    torch.zeros((), dtype=fixture["input_arrival"].dtype),
                )
                for row in rows
            ]
        ),
    }


def test_batched_transfer_matches_per_segment_reference():
    fixture = _fixture(sample_count=80, nmax=4)
    result = batched_segment_transfer(**_batched_args(fixture), return_debug=True)
    reference = _reference_rows(fixture)

    for key in (
        "upstream_visible_load",
        "segment_delay",
        "output_arrival",
        "output_slew",
        "first_buffer_input_slew",
        "first_buffer_delay",
    ):
        torch.testing.assert_close(result[key], reference[key])
    assert tuple(result["wire_delays"].shape) == (80, 5)
    assert tuple(result["buffer_delays"].shape) == (80, 4)


def test_per_size_lookup_matches_scalar_device_lookup_at_legal_sizes():
    fixture = _fixture(sample_count=3, nmax=1)
    input_slew = torch.tensor([1.0, 6.0, 18.0], dtype=torch.float64)
    output_load = torch.tensor([0.5, 4.0, 12.0], dtype=torch.float64)

    per_size = lookup_buffer_device_per_size(
        fixture["buffer_device"],
        input_slew=input_slew,
        output_load=output_load,
    )

    assert per_size["per_size_delay"].shape == (3, 2)
    for size_index in range(2):
        scalar = batched_segment_transfer(
            input_arrival=torch.zeros(3, dtype=torch.float64),
            input_slew=input_slew,
            downstream_load=output_load,
            edge_resistance=torch.zeros(3, dtype=torch.float64),
            edge_capacitance=torch.zeros(3, dtype=torch.float64),
            repeater_count=torch.ones(3, dtype=torch.long),
            split_fractions=torch.tensor(
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                dtype=torch.float64,
            ),
            bsu_index=torch.full((3,), float(size_index), dtype=torch.float64),
            upstream_retained_cap=torch.zeros(3, dtype=torch.float64),
            buffer_input_cap_by_size=fixture["buffer_input_cap_by_size"],
            buffer_slew_axis=fixture["buffer_slew_axis"],
            buffer_load_axis=fixture["buffer_load_axis"],
            buffer_delay_lut=fixture["buffer_delay_lut"],
            buffer_output_slew_lut=fixture["buffer_output_slew_lut"],
            return_debug=True,
        )
        torch.testing.assert_close(
            per_size["per_size_delay"][:, size_index],
            scalar["first_buffer_delay"],
        )
        torch.testing.assert_close(
            per_size["per_size_output_slew"][:, size_index],
            scalar["first_buffer_output_slew"],
        )


def test_fixed_size_lookup_matches_per_size_values_and_gradients():
    fixture = _fixture(sample_count=3, nmax=1)
    input_slew = torch.tensor(
        [1.0, 6.0, 18.0], dtype=torch.float64, requires_grad=True
    )
    output_load = torch.tensor(
        [0.5, 4.0, 12.0], dtype=torch.float64, requires_grad=True
    )
    fixed = lookup_buffer_device_fixed_size(
        fixture["buffer_device"],
        size_index=1,
        input_slew=input_slew,
        output_load=output_load,
    )
    fixed_loss = fixed["buffer_delay"].sum() + fixed["buffer_output_slew"].sum()
    fixed_grad = torch.autograd.grad(fixed_loss, (input_slew, output_load))

    reference = lookup_buffer_device_per_size(
        fixture["buffer_device"],
        input_slew=input_slew,
        output_load=output_load,
    )
    reference_loss = (
        reference["per_size_delay"][:, 1].sum()
        + reference["per_size_output_slew"][:, 1].sum()
    )
    reference_grad = torch.autograd.grad(reference_loss, (input_slew, output_load))

    torch.testing.assert_close(
        fixed["buffer_input_cap"], reference["per_size_input_cap"][:, 1]
    )
    torch.testing.assert_close(fixed["buffer_delay"], reference["per_size_delay"][:, 1])
    torch.testing.assert_close(
        fixed["buffer_output_slew"], reference["per_size_output_slew"][:, 1]
    )
    for actual, expected in zip(fixed_grad, reference_grad):
        torch.testing.assert_close(actual, expected)


def test_batched_transfer_preserves_gradients():
    fixture = _fixture(sample_count=16, nmax=3, requires_grad=True)
    result = batched_segment_transfer(**_batched_args(fixture))
    loss = (
        result["segment_delay"].sum()
        + 0.1 * result["output_slew"].sum()
        + result["upstream_visible_load"].sum()
    )
    loss.backward()

    for key in (
        "input_slew",
        "downstream_load",
        "edge_resistance",
        "edge_capacitance",
        "bsu_index",
        "upstream_retained_cap",
    ):
        grad = fixture[key].grad
        assert grad is not None, key
        assert torch.isfinite(grad).all(), key
        assert float(torch.max(torch.abs(grad)).item()) > 0.0, key


def test_batched_transfer_extrapolates_out_of_range_load_like_analytic():
    fixture = _fixture(sample_count=2, nmax=1, requires_grad=True)
    args = _batched_args(fixture)
    args["input_arrival"] = torch.zeros(2, dtype=torch.float64)
    args["input_slew"] = torch.zeros(2, dtype=torch.float64)
    args["downstream_load"] = torch.tensor(
        [15.0, 15.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    args["edge_resistance"] = torch.zeros(2, dtype=torch.float64)
    args["edge_capacitance"] = torch.zeros(2, dtype=torch.float64)
    args["repeater_count"] = torch.ones(2, dtype=torch.long)
    args["split_fractions"] = torch.tensor(
        [[0.5, 0.5], [0.25, 0.75]],
        dtype=torch.float64,
    )
    args["bsu_index"] = torch.zeros(2, dtype=torch.float64)
    args["upstream_retained_cap"] = torch.zeros(2, dtype=torch.float64)

    result = batched_segment_transfer(**args)

    torch.testing.assert_close(
        result["first_buffer_delay"],
        torch.full((2,), 1.6, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["first_buffer_output_slew"],
        torch.full((2,), 0.8, dtype=torch.float64),
    )
    result["first_buffer_delay"].sum().backward()
    torch.testing.assert_close(
        args["downstream_load"].grad,
        torch.full((2,), 0.04, dtype=torch.float64),
    )


def test_batched_transfer_rejects_invalid_shapes_and_counts():
    fixture = _fixture(sample_count=4, nmax=2)
    args = _batched_args(fixture)
    bad = dict(args)
    bad["split_fractions"] = bad["split_fractions"][:, :2]
    bad["repeater_count"] = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="repeater_count"):
        batched_segment_transfer(**bad)

    bad = dict(args)
    bad["repeater_count"] = torch.tensor([0, -1, 1, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="non-negative"):
        batched_segment_transfer(**bad)

    bad = dict(args)
    bad["split_fractions"] = bad["split_fractions"].clone()
    bad["split_fractions"][0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        batched_segment_transfer(**bad)


def test_batched_transfer_is_faster_than_per_segment_reference_smoke():
    fixture = _fixture(sample_count=512, nmax=4, dtype=torch.float32)
    args = _batched_args(fixture)
    for _ in range(3):
        batched_segment_transfer(**args)
    start = time.perf_counter()
    for _ in range(10):
        batched_segment_transfer(**args)
    batched_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(2):
        _reference_rows(fixture)
    reference_time = time.perf_counter() - start
    per_call_reference = reference_time / 2.0
    per_call_batched = batched_time / 10.0

    assert per_call_batched < per_call_reference * 0.5
