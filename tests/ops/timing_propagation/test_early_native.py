"""Early CPU/CUDA LUT kernels against independent affine branch equations."""

import pytest
import torch
from dreamplace.ops.timing_propagation.cell_aat_native import _CellAat


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("reachable", ["both", "rise", "none"])
def test_early_cell_values_and_six_adjoints_match_branch_equations(
    device, dtype, sense, smooth, reachable
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    coefficients = ((1, 2, 3), (2, 3, 2), (0.5, 0.3, 0.2), (0.8, 0.2, 0.4))

    def tensor(values):
        return torch.tensor(values, device=device, dtype=dtype)

    tables = []
    for bias, slew, cap in coefficients:
        tables.extend(
            (
                tensor([[bias, bias + cap, bias + 4 * slew, bias + 4 * slew + cap]]),
                tensor([[0, 4]]),
                tensor([[0, 1]]),
                torch.tensor([2], device=device),
                torch.tensor([2], device=device),
            )
        )
    pins = [
        tensor(v).requires_grad_()
        for v in (
            [10, torch.inf],
            [20, torch.inf],
            [1, torch.inf],
            [2, torch.inf],
            [0, 0.4],
            [0, 0.6],
        )
    ]
    with torch.no_grad():
        if reachable != "both":
            pins[1][0] = pins[3][0] = torch.inf
        if reachable == "none":
            pins[0][0] = pins[2][0] = torch.inf
    arcs = torch.tensor([[0, 1, 0, 0, sense]], device=device)
    early = True
    actual = _CellAat.apply(arcs, *pins, tables, smooth, 0.5, early)[0, :4]
    expected = []
    zero = sum(torch.where(torch.isfinite(p), p, 0).sum() for p in pins) * 0
    for output in range(2):
        delays, slews = [], []
        for source in range(2):
            if sense and sense != (1 if source == output else -1):
                continue
            if not torch.isfinite(pins[source + 2][0]):
                continue
            transition, load = pins[source + 2][0], pins[output + 4][1]
            bias, ds, dc = coefficients[output]
            delays.append(pins[source][0] + bias + ds * transition + dc * load)
            bias, ds, dc = coefficients[output + 2]
            slews.append(bias + ds * transition + dc * load)
        if delays:
            delays, slews = torch.stack(delays), torch.stack(slews)
            arrival = -2 * torch.logsumexp(-delays / 2, 0) if smooth else delays.amin()
            expected.append((arrival, slews.amin()))
        else:
            expected.append((zero + torch.inf, zero + torch.inf))
    expected = torch.stack((expected[0][0], expected[1][0], expected[0][1], expected[1][1]))
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
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
        torch.stack(gradients[0]), torch.stack(gradients[1]), atol=tolerance, rtol=tolerance
    )
