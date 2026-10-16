from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import (
    SLEW_SQRT_EPS,
    TimingPropagation,
)


class RecordingArcs:
    def __init__(self, arcs):
        self.arcs = arcs
        self.indices = None

    def __getitem__(self, indices):
        self.indices = indices.clone()
        return self.arcs[indices]


def _reference_propagation(arcs, inputs):
    arrivals = [[value] for value in inputs[0]]
    fall_arrivals = [[value] for value in inputs[1]]
    slews = [[value] for value in inputs[2]]
    fall_slews = [[value] for value in inputs[3]]
    for driver, sink in arcs:
        arrivals[sink].append(inputs[0][driver] + inputs[4][sink])
        fall_arrivals[sink].append(inputs[1][driver] + inputs[5][sink])
        slews[sink].append(
            (inputs[2][driver].square() + inputs[6][sink]).clamp_min(SLEW_SQRT_EPS).sqrt()
        )
        fall_slews[sink].append(
            (inputs[3][driver].square() + inputs[7][sink]).clamp_min(SLEW_SQRT_EPS).sqrt()
        )
    return tuple(
        torch.stack([torch.stack(values).amax() for values in pins])
        for pins in (arrivals, fall_arrivals, slews, fall_slews)
    )


def _gradients(outputs, inputs):
    loss = sum((i + 1) * output.square().sum() for i, output in enumerate(outputs))
    gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, gradients, strict=True)
    )


@pytest.mark.parametrize("net_ids", [[], [1], [0, 1, 2, 3], [3, 1, 0], [3, 2, 3]])
def test_net_aat_csr_preserves_arc_order_outputs_and_gradients(net_ids):
    arcs = torch.tensor([[0, 1], [0, 2], [3, 4], [5, 6], [5, 7], [5, 8]])
    starts = [0, 2, 2, 3, 6]
    selected_indices = [index for net in net_ids for index in range(starts[net], starts[net + 1])]
    recording = RecordingArcs(arcs)
    model = SimpleNamespace(
        net_flat_arcs=recording,
        net_flat_arcs_start=torch.tensor(starts),
        timing_aggregation_mode="hard",
        timing_aggregation_tau_ps=1.0,
    )
    generator = torch.Generator().manual_seed(42)
    values = tuple(torch.rand(9, dtype=torch.float64, generator=generator) for _ in range(8))
    actual_inputs = tuple(value.clone().requires_grad_() for value in values)
    expected_inputs = tuple(value.clone().requires_grad_() for value in values)
    actual = TimingPropagation.calculate_net_aat_level(
        model, torch.tensor(net_ids, dtype=torch.long), *actual_inputs
    )
    expected = _reference_propagation(arcs[selected_indices].tolist(), expected_inputs)

    if net_ids:
        torch.testing.assert_close(
            recording.indices,
            torch.tensor(selected_indices, dtype=torch.long),
            rtol=0,
            atol=0,
        )
    else:
        assert recording.indices is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        _gradients(actual, actual_inputs),
        _gradients(expected, expected_inputs),
        rtol=1e-12,
        atol=1e-12,
    )
