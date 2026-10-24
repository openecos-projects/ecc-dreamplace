from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def _timing(mode, cached):
    timing = TimingPropagation.__new__(TimingPropagation)
    timing.dtype = torch.float64
    timing.use_cell_aat_static_cache = cached
    timing.pin_net = torch.tensor([0, 1, 2, 3, 4])
    timing.timing_aggregation_mode = mode
    timing.timing_aggregation_tau_ps = 2.0
    timing.arcs_info = SimpleNamespace(
        r_delay_luts=1.0, f_delay_luts=2.0, r_trans_luts=0.1, f_trans_luts=0.2
    )
    timing._entry_queries_with_optional_surrogate = lambda queries, _surrogate, **_: [
        q["pin_slew"] + q["luts"] * q["pin_net_caps"] + q["lib_arc_idxs"] + q["lib_cell_idxs"]
        for q in queries
    ]
    return timing


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_cached_cell_groups_preserve_values_and_gradients_after_arc_updates(mode):
    arcs = torch.tensor([[0, 2, 0, 0, 1], [1, 3, 0, 1, -1], [0, 4, 0, 2, 0], [1, 2, 0, 3, 0]])
    models = [_timing(mode, cached) for cached in (False, True)]
    for update in (None, "master", "sense"):
        if update == "master":
            arcs[0, 2:4] = torch.tensor([3, 4])
        elif update == "sense":
            arcs[0, 4] = -1
        results = []
        for model in models:
            inputs = [
                torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64, requires_grad=True)
                for _ in range(6)
            ]
            values = model.calculate_cell_aat_level(arcs, *inputs, use_surrogate=False)
            gradient = torch.autograd.grad(sum(v.sum() for v in values[1:5]), inputs)
            results.append((values, gradient))
        for actual, expected in zip(results[1][0], results[0][0], strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(results[1][1], results[0][1], strict=True):
            torch.testing.assert_close(actual, expected)
