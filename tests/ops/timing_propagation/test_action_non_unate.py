from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.action_cell_aat import apply_action_cell_aat


@pytest.mark.parametrize("smooth", [False, True])
def test_interpolate_before_max_when_target_switches_transition(smooth):
    dtype = torch.float64
    tables = []
    for _ in range(4):
        tables.extend(
            (
                torch.tensor([[20.0, 30.0], [35.0, 15.0]], dtype=dtype),
                torch.tensor([[0.0, 1.0]] * 2, dtype=dtype),
                torch.zeros((2, 1), dtype=dtype),
                torch.full((2,), 2),
                torch.ones(2, dtype=torch.long),
            )
        )
    pins = (
        torch.zeros(2, dtype=dtype),
        torch.zeros(2, dtype=dtype),
        torch.tensor([0.1, 0.0], dtype=dtype),
        torch.tensor([0.9, 0.0], dtype=dtype),
        torch.zeros(2, dtype=dtype),
        torch.zeros(2, dtype=dtype),
    )
    beta = torch.zeros((1, 4), dtype=dtype, requires_grad=True)
    context = SimpleNamespace(action_cell_aat_calls=0)
    result = apply_action_cell_aat(
        torch.tensor([[0, 1, 0, 0, 0]]),
        pins,
        tables,
        smooth,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.zeros((1, 4), dtype=torch.long),
        valid_arcs=torch.ones((1, 4), dtype=torch.bool),
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
        context=context,
        global_rows=torch.zeros(1, dtype=torch.long),
        defer_target_queries=True,
    )
    gradient = torch.autograd.grad(result[0, 0], beta)[0]

    def forward(step):
        branches = torch.tensor([21.0 + step * 12.0, 29.0 - step * 12.0], dtype=dtype)
        return torch.logsumexp(branches * 0.5, 0) / 0.5 if smooth else branches.max()

    numerical = (forward(1e-6) - forward(-1e-6)) / 2e-6
    torch.testing.assert_close(gradient, numerical.expand_as(gradient), atol=1e-8, rtol=0)
    if not smooth:
        # Secant of the maxima would be 33-29=+4, with the wrong sign.
        torch.testing.assert_close(gradient, torch.full_like(gradient, -12.0))
