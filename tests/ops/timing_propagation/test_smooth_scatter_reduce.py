import torch

from dreamplace.ops.timing_propagation.timing_propagation import (
    smooth_max,
    smooth_scatter_max,
    smooth_scatter_max_tau,
    smooth_scatter_min_tau,
)


def test_smooth_scatter_max_matches_grouped_logsumexp():
    dtype = torch.float64
    dest = torch.tensor([1.0, -2.0], dtype=dtype)
    index = torch.tensor([0, 0, 1], dtype=torch.long)
    src = torch.tensor([2.0, 4.0, 3.0], dtype=dtype)
    tau = 0.5

    actual = smooth_scatter_max_tau(dest, index, src, include_self=True, tau=tau)
    expected0 = tau * torch.logsumexp(torch.tensor([1.0, 2.0, 4.0], dtype=dtype) / tau, dim=0)
    expected1 = tau * torch.logsumexp(torch.tensor([-2.0, 3.0], dtype=dtype) / tau, dim=0)

    torch.testing.assert_close(actual, torch.stack([expected0, expected1]))


def test_legacy_smooth_scatter_max_uses_alpha_scale():
    dtype = torch.float64
    dest = torch.tensor([1.0], dtype=dtype)
    index = torch.tensor([0, 0], dtype=torch.long)
    src = torch.tensor([2.0, 4.0], dtype=dtype)
    alpha = 2.0

    actual = smooth_scatter_max(dest, index, src, include_self=True, alpha=alpha)
    expected = torch.logsumexp(torch.tensor([1.0, 2.0, 4.0], dtype=dtype) * alpha, dim=0) / alpha

    torch.testing.assert_close(actual, expected.reshape(1))


def test_smooth_scatter_min_matches_negative_smooth_max():
    dtype = torch.float64
    dest = torch.tensor([1.0, -2.0], dtype=dtype)
    index = torch.tensor([0, 0, 1], dtype=torch.long)
    src = torch.tensor([2.0, 4.0, 3.0], dtype=dtype)
    tau = 0.5

    actual = smooth_scatter_min_tau(dest, index, src, include_self=True, tau=tau)
    expected0 = -tau * torch.logsumexp(-torch.tensor([1.0, 2.0, 4.0], dtype=dtype) / tau, dim=0)
    expected1 = -tau * torch.logsumexp(-torch.tensor([-2.0, 3.0], dtype=dtype) / tau, dim=0)

    torch.testing.assert_close(actual, torch.stack([expected0, expected1]))


def test_smooth_scatter_max_spreads_gradient_to_competing_sources():
    dtype = torch.float64
    dest = torch.tensor([-100.0], dtype=dtype)
    index = torch.tensor([0, 0], dtype=torch.long)
    src = torch.tensor([10.0, 11.0], dtype=dtype, requires_grad=True)

    result = smooth_scatter_max_tau(dest, index, src, include_self=False, tau=2.0)
    result.sum().backward()

    assert src.grad is not None
    assert torch.all(src.grad > 0.0)
    assert torch.count_nonzero(src.grad).item() == 2


def test_smooth_scatter_max_preserves_unreached_negative_infinity_destinations():
    dtype = torch.float64
    dest = torch.tensor([-torch.inf, 0.0], dtype=dtype)
    index = torch.tensor([1], dtype=torch.long)
    src = torch.tensor([1.0], dtype=dtype)

    actual = smooth_scatter_max_tau(dest, index, src, include_self=True, tau=2.0)

    assert torch.isneginf(actual[0])
    assert torch.isfinite(actual[1])


def test_smooth_scatter_min_spreads_gradient_to_competing_sources():
    dtype = torch.float64
    dest = torch.tensor([100.0], dtype=dtype)
    index = torch.tensor([0, 0], dtype=torch.long)
    src = torch.tensor([10.0, 11.0], dtype=dtype, requires_grad=True)

    result = smooth_scatter_min_tau(dest, index, src, include_self=False, tau=2.0)
    result.sum().backward()

    assert src.grad is not None
    assert torch.all(src.grad > 0.0)
    assert torch.count_nonzero(src.grad).item() == 2


def test_smooth_scatter_min_preserves_unreached_positive_infinity_destinations():
    dtype = torch.float64
    dest = torch.tensor([torch.inf, 0.0], dtype=dtype)
    index = torch.tensor([1], dtype=torch.long)
    src = torch.tensor([-1.0], dtype=dtype)

    actual = smooth_scatter_min_tau(dest, index, src, include_self=True, tau=2.0)

    assert torch.isposinf(actual[0])
    assert torch.isfinite(actual[1])


def test_pairwise_smooth_max_can_be_forced_to_hard_max_by_env(monkeypatch):
    dtype = torch.float64
    a = torch.tensor([0.0], dtype=dtype)
    b = torch.tensor([1.0], dtype=dtype)

    monkeypatch.setenv("AIMP_FORCE_HARD_PAIRWISE_MAX", "1")
    actual = smooth_max(a, b, alpha=1.0)

    torch.testing.assert_close(actual, torch.maximum(a, b))
