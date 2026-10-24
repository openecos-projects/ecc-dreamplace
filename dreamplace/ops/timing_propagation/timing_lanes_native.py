"""Aggregate a level's two arrival and two slew lanes in one native pass."""

import torch
from torch.autograd import Function

from . import lut_entry_2d_op
from .scatter_groups import scatter_groups


class _TimingLanes(Function):
    @staticmethod
    def forward(ctx, index, smooth, tau, *values):
        original_index = index
        destinations, sources = values[:4], values[4:]
        groups = scatter_groups(index, destinations[0].numel())
        index = index.long().contiguous()
        result = lut_entry_2d_op._select_module(destinations[0]).timing_lanes_forward(
            index,
            list(destinations),
            list(sources),
            smooth,
            tau,
            groups,
        )
        ctx.save_for_backward(original_index, *values, *result[:4], index, *result[4:])
        ctx.smooth = smooth
        ctx.set_materialize_grads(False)
        return tuple(result[:4])

    @staticmethod
    def backward(ctx, *gradients):
        *_, index, denominator, dest_factor, src_factor = ctx.saved_tensors
        empty = denominator.new_empty(0)
        grads = lut_entry_2d_op._select_module(denominator).timing_lanes_backward(
            index,
            denominator,
            dest_factor,
            src_factor,
            [empty if gradient is None else gradient for gradient in gradients],
            ctx.smooth,
        )
        return (
            None,
            None,
            None,
            *(None if gradients[lane % 4] is None else value for lane, value in enumerate(grads)),
        )


def try_timing_lanes_native(timing, index, destinations, sources):
    native = lut_entry_2d_op._select_module(destinations[0])
    if str(getattr(timing, "timing_aggregation_mode", "hard")) == "avesmooth":
        # The native lanes operator only knows hard/smooth (LSE).  AveSmooth
        # needs the near-max gradient mask, which is implemented by the
        # Python _AveScatterMax autograd function in scatter_timing_max.
        return None
    if (
        destinations[0].is_cuda
        and not torch.are_deterministic_algorithms_enabled()
        or not hasattr(native, "timing_lanes_forward")
    ):
        return None
    return _TimingLanes.apply(
        index,
        timing.timing_aggregation_mode == "smooth",
        timing.timing_aggregation_tau_ps,
        *destinations,
        *sources,
    )
