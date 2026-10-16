import pytest
import torch
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def _timing(mode):
    timing = object.__new__(TimingPropagation)
    timing.dtype = torch.float64
    timing.FF_ids = torch.tensor([0, 1])
    timing.flat_inst_arcs_by_level_start = torch.tensor([0, 2])
    timing.flat_inst_arcs_by_level = torch.tensor([[0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 1]])
    timing.clk_pin_rtran = torch.tensor([1.0, 2.0], dtype=timing.dtype)
    timing.clk_pin_ftran = torch.tensor([3.0, 4.0], dtype=timing.dtype)
    timing.pin_net = torch.tensor([0, 1, 2])
    timing.timing_aggregation_mode = mode
    timing.timing_aggregation_tau_ps = 2.0
    timing.r_delay_entry = lambda cells, slew, load, arcs, pins, **kwargs: slew + 2.0 * load + arcs
    timing.f_delay_entry = (
        lambda cells, slew, load, arcs, pins, **kwargs: slew + 3.0 * load + arcs + 10.0
    )
    timing.r_tran_entry = lambda cells, slew, load, arcs, pins, **kwargs: 0.1 * slew + load
    timing.f_tran_entry = lambda cells, slew, load, arcs, pins, **kwargs: 0.1 * slew + 1.5 * load
    return timing


@pytest.mark.parametrize("mode", ["hard", "smooth"])
@pytest.mark.parametrize("seed", [99.0, -100.0])
def test_clk2q_replaces_old_q_state_preserving_pi_and_load_gradient(mode, seed):
    timing = _timing(mode)
    load = torch.tensor([0.0, 5.0, 0.0], dtype=timing.dtype, requires_grad=True)
    aat = torch.tensor([7.0, seed, 42.0], dtype=timing.dtype, requires_grad=True)
    slew = torch.tensor([8.0, seed, 43.0], dtype=timing.dtype)
    for current_load in [load, load + torch.tensor([0.0, 1.0, 0.0])]:
        nets, rise, fall, rtran, ftran = timing.calculate_clk2q_aat(
            aat, aat, slew, slew, current_load, current_load
        )
        rise_candidates = torch.tensor([1.0, 3.0], dtype=timing.dtype) + 2.0 * current_load[1]
        fall_candidates = torch.tensor([13.0, 15.0], dtype=timing.dtype) + 3.0 * current_load[1]
        aggregate = (
            (lambda x: x.max())
            if mode == "hard"
            else (lambda x: 2.0 * torch.logsumexp(x / 2.0, dim=0))
        )
        torch.testing.assert_close(rise, torch.stack([aat[0], aggregate(rise_candidates), aat[2]]))
        torch.testing.assert_close(fall, torch.stack([aat[0], aggregate(fall_candidates), aat[2]]))
        torch.testing.assert_close(rtran, torch.stack([slew[0], 0.2 + current_load[1], slew[2]]))
        torch.testing.assert_close(
            ftran, torch.stack([slew[0], 0.4 + 1.5 * current_load[1], slew[2]])
        )
        assert nets.tolist() == [1]
        grad_load, grad_seed = torch.autograd.grad(rise.sum() + fall.sum(), (load, aat))
        torch.testing.assert_close(grad_load, torch.tensor([0.0, 5.0, 0.0], dtype=timing.dtype))
        torch.testing.assert_close(grad_seed, torch.tensor([2.0, 0.0, 2.0], dtype=timing.dtype))


@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_clk2q_gradient_matches_finite_difference_with_stale_seed(mode):
    timing = _timing(mode)
    load = torch.tensor([0.0, 5.0, 0.0], dtype=timing.dtype, requires_grad=True)
    aat = torch.tensor([7.0, 99.0, 42.0], dtype=timing.dtype, requires_grad=True)
    slew = torch.tensor([8.0, 99.0, 43.0], dtype=timing.dtype)

    def propagate(current_load, old_seed):
        return timing.calculate_clk2q_aat(
            old_seed, old_seed, slew, slew, current_load, current_load
        )[1:]

    assert torch.autograd.gradcheck(propagate, (load, aat), rtol=1e-3, atol=1e-6)
