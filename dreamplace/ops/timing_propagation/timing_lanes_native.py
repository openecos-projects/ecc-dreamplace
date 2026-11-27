"""Aggregate a level's two arrival and two slew lanes in one native pass."""

import math

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


class _DriftTimingLanes(Function):
    """LSE AT merge at base ``tau`` in forward, with ``AtMergeTemperature`` drift VJP in backward."""

    @staticmethod
    def forward(ctx, index, tau, at_merge_temperature, merge_key, early, *values):
        destinations, sources = values[:4], values[4:]
        groups = scatter_groups(index, destinations[0].numel())
        index_c = index.long().contiguous()
        tau_f = float(tau)
        result = lut_entry_2d_op._select_module(destinations[0]).timing_lanes_forward(
            index_c,
            list(destinations),
            list(sources),
            True,
            tau_f,
            groups,
        )
        ctx.save_for_backward(*destinations, *sources, index_c, *result[4:])
        ctx.groups = groups
        ctx.tau = tau_f
        ctx.at_merge_temperature = at_merge_temperature
        ctx.merge_key = merge_key
        ctx.early = bool(early)
        ctx.set_materialize_grads(False)
        return tuple(result[:4])

    @staticmethod
    def backward(ctx, *gradients):
        saved = ctx.saved_tensors
        destinations = saved[:4]
        sources = saved[4:8]
        index_c = saved[8]
        denominator, dest_factor, src_factor = saved[9], saved[10], saved[11]
        temp = ctx.at_merge_temperature
        if temp is not None and not ctx.early:
            pins = int(destinations[0].numel())
            if pins > 0 and int(index_c.numel()) > 0:
                cand_groups = None
                for lane, lane_name in ((0, "rAAT"), (1, "fAAT")):
                    grad_lane = gradients[lane]
                    if grad_lane is None:
                        continue
                    if cand_groups is None:
                        cand_groups = torch.cat(
                            (
                                torch.arange(
                                    pins, device=index_c.device, dtype=torch.long
                                ),
                                index_c,
                            ),
                            dim=0,
                        )
                    raw_cand = torch.cat((destinations[lane], sources[lane]), dim=0)
                    cand_vals = torch.where(
                        raw_cand > -1e8,
                        raw_cand,
                        torch.full_like(raw_cand, -math.inf),
                    )
                    temp.observe(
                        (*ctx.merge_key, lane_name),
                        cand_groups,
                        cand_vals,
                        pins,
                        grad_lane.abs(),
                    )
        vjp_tau = float(temp.tau) if (temp is not None and not ctx.early) else ctx.tau
        module = lut_entry_2d_op._select_module(denominator)
        if abs(vjp_tau - ctx.tau) > 1e-9 and (
            gradients[0] is not None or gradients[1] is not None
        ):
            vjp_fwd = module.timing_lanes_forward(
                index_c,
                list(destinations),
                list(sources),
                True,
                vjp_tau,
                ctx.groups,
            )
            denominator, dest_factor, src_factor = vjp_fwd[4], vjp_fwd[5], vjp_fwd[6]
        empty = denominator.new_empty(0)
        grads = module.timing_lanes_backward(
            index_c,
            denominator,
            dest_factor,
            src_factor,
            [empty if gradient is None else gradient for gradient in gradients],
            True,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            *(
                None if gradients[lane % 4] is None else value
                for lane, value in enumerate(grads)
            ),
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
