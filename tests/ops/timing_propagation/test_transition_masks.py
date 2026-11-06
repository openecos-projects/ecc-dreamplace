"""Legal arrival branches must not restrict the shared electrical slew."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation import lut_entry_2d_op
from dreamplace.ops.timing_propagation.action_cell_aat import apply_action_cell_aat
from dreamplace.ops.timing_propagation.cell_aat_native import _CellAat


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("mask", [1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 15])
def test_masked_values_and_six_adjoints_match_affine_branches(device, early, smooth, sense, mask):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device=device)

    coefficients = ((1, 2, 3), (2, 3, 2), (0.5, 0.3, 0.2), (0.8, 0.2, 0.4))
    tables = []
    for bias, ds, dc in coefficients:
        tables.extend(
            (
                tensor([[bias, bias + dc, bias + 4 * ds, bias + 4 * ds + dc]]),
                tensor([[0, 4]]),
                tensor([[0, 1]]),
                torch.tensor([2], device=device),
                torch.tensor([2], device=device),
            )
        )
    pins = [
        tensor(values).requires_grad_()
        for values in (
            [10, 0],
            [20, 0],
            [1, 0],
            [2, 0],
            [0, 0.4],
            [0, 0.6],
        )
    ]
    arcs = torch.tensor([[0, 1, 0, 0, sense, 0, 0, 0, mask]], device=device)
    actual = _CellAat.apply(arcs, *pins, tables, smooth, 0.5, early)[0, :4]
    zero = sum(pin.sum() for pin in pins) * 0
    arrivals, slews = [], []

    def reduce(values, *, electrical=False):
        values = torch.stack(values)
        if early:
            return (
                values.amin() if electrical or not smooth else -2 * torch.logsumexp(-values / 2, 0)
            )
        return values.amax() if not smooth else 2 * torch.logsumexp(values / 2, 0)

    for output in range(2):
        delay_candidates, slew_candidates = [], []
        for source in range(2):
            if sense and sense != (1 if source == output else -1):
                continue
            transition, load = pins[source + 2][0], pins[output + 4][1]
            bias, ds, dc = coefficients[output]
            if mask & (1 << (2 * source + output)):
                delay_candidates.append(pins[source][0] + bias + ds * transition + dc * load)
            bias, ds, dc = coefficients[output + 2]
            slew_candidates.append(bias + ds * transition + dc * load)
        arrivals.append(
            reduce(delay_candidates) if delay_candidates else zero + (torch.inf if early else -2e8)
        )
        slews.append(reduce(slew_candidates, electrical=True))
    expected = torch.stack((*arrivals, *slews))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weights = tensor([1, 2, 3, 4])
    losses = [
        (torch.where(torch.isfinite(values), values, 0) * weights).sum()
        for values in (actual, expected)
    ]
    gradients = [
        torch.autograd.grad(loss, pins, allow_unused=True, materialize_grads=True)
        for loss in losses
    ]
    torch.testing.assert_close(
        torch.stack(gradients[0]), torch.stack(gradients[1]), atol=1e-10, rtol=1e-10
    )
    if not early:
        values = lut_entry_2d_op._select_module(pins[0]).cell_aat_values(
            arcs, *pins, tables, smooth, 0.5
        )
        torch.testing.assert_close(values[0, :4], expected.detach(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("mask", [1, 2, 4, 8, 6, 9, 15])
def test_four_action_partials_obey_masked_affine_branches(device, sense, smooth, mask):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device=device)

    coefficients = tensor(
        [
            [[10, 2, 3], [8, -4, 1], [20, 1, 2], [5, 3, 2], [30, -1, 4]],
            [[12, 3, 2], [9, -2, 3], [22, 2, 1], [6, 4, 1], [32, -3, 2]],
            [[0.5, 0.3, 0.2], [0.4, 0.1, 0.1], [0.7, 0.5, 0.3], [0.3, 0.2, 0.1], [0.9, 0.4, 0.4]],
            [[0.8, 0.2, 0.4], [0.5, 0.1, 0.2], [0.9, 0.4, 0.5], [0.4, 0.3, 0.1], [1.1, 0.5, 0.3]],
        ]
    )
    tables = []
    for group in coefficients:
        bias, ds, dc = group.unbind(1)
        tables.extend(
            (
                torch.stack((bias, bias + dc, bias + 4 * ds, bias + 4 * ds + dc), dim=1),
                tensor([[0, 4]] * 5),
                tensor([[0, 1]] * 5),
                torch.full((5,), 2, device=device),
                torch.full((5,), 2, device=device),
            )
        )
    pins = tuple(
        tensor(value) for value in ([0, 0], [0, 0], [0.1, 0], [0.8, 0], [0, 0.4], [0, 0.6])
    )
    beta = tensor([[0, 0, 0, 0]]).requires_grad_()
    arcs = torch.tensor([[0, 1, 0, 0, sense, 0, 0, 0, mask]], device=device)
    valid = torch.ones((1, 4), dtype=torch.bool, device=device)
    context = SimpleNamespace(
        action_cell_aat_calls=0, evaluate_targets=False, valid_arcs=valid.clone()
    )
    actual = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        smooth,
        0.5,
        target_arc_ids=torch.tensor([[1, 2, 3, 4]], device=device),
        target_senses=torch.full((1, 4), sense, device=device),
        valid_arcs=valid,
        inst_ids=torch.tensor([0], device=device),
        beta=beta,
        context=context,
        global_rows=torch.tensor([0], device=device),
        defer_target_queries=True,
    )[0, :4]
    expected = []
    for lane in range(4):
        output = lane % 2
        candidates = []
        for source in range(2):
            if sense and sense != (1 if source == output else -1):
                continue
            if lane < 2 and not mask & (1 << (2 * source + output)):
                continue
            values = coefficients[lane] @ torch.stack(
                (tensor(1), pins[source + 2][0], pins[output + 4][1])
            )
            value = values[0] + (beta[0] * (values[1:] - values[0])).sum()
            candidates.append(value)
        if not candidates:
            expected.append(beta.sum() * 0 - 2e8)
        else:
            values = torch.stack(candidates)
            expected.append(2 * torch.logsumexp(values / 2, 0) if smooth else values.amax())
    expected = torch.stack(expected)
    weights = tensor([1, 2, 3, 4])
    actual_gradient = torch.autograd.grad((actual * weights).sum(), beta)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), beta)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10)
