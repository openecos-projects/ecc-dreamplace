"""The shared cap probe preserves both early and late base-point Jacobians."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import ACTION_NAMES
from dreamplace.ops.timing_propagation.action_pin_caps import apply_pin_caps


@pytest.mark.parametrize("channels", [2, 3])
def test_cap_channels_keep_current_values_and_four_direction_partials(channels):
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    upstream = torch.zeros((), dtype=torch.float64, requires_grad=True)
    context = SimpleNamespace(
        beta=beta,
        target_cell_ids=torch.tensor([[1, 2, 3, 4]]),
        valid_instances=torch.tensor([[True, True, False, True]]),
        action_names=ACTION_NAMES,
    )
    model = SimpleNamespace(
        _active_discrete_action_context=context,
        params=SimpleNamespace(timing_action_pin_cap_gradient=1),
    )
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0, 0, -1]),
        pin_2_libpin_offset=torch.tensor([0, 1, -1]),
        cell_id_2_libpin_id_start=torch.arange(0, 11, 2),
    )
    tables, caps = [], []
    for scale in range(1, channels + 1):
        tables.append(
            scale
            * torch.tensor([1, 0.2, 2, 0.4, 0.4, 0.1, 1.5, 0.3, 0.8, 0.16], dtype=torch.float64)
        )
        caps.append(
            scale * torch.tensor([1, 0.2, 7], dtype=torch.float64)
            + upstream * torch.tensor([2, 3, 4], dtype=torch.float64)
        )
    result = apply_pin_caps(model, tuple(caps), data, tuple(tables))
    torch.testing.assert_close(torch.stack(result), torch.stack(caps), rtol=0, atol=0)
    gradient, upstream_gradient = torch.autograd.grad(torch.stack(result).sum(), (beta, upstream))
    torch.testing.assert_close(
        gradient,
        sum(range(1, channels + 1)) * torch.tensor([[1.2, -0.7, 0, -0.24]], dtype=torch.float64),
    )
    torch.testing.assert_close(upstream_gradient, torch.tensor(9.0 * channels, dtype=torch.float64))
