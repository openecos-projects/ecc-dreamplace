"""First-order native derivatives for the timing scatter log-sum-exp."""

import torch
from torch.autograd import Function

from . import lut_entry_2d_op
from .scatter_groups import scatter_groups


class _SmoothScatterMax(Function):
    @staticmethod
    def forward(ctx, dest, index, src, tau, include_self):
        original = dest, index, src
        groups = (
            scatter_groups(index, dest.numel())
            if torch.are_deterministic_algorithms_enabled()
            else ()
        )
        index = index.long().contiguous()
        output, dest_weight, src_weight = lut_entry_2d_op._select_module(
            dest
        ).smooth_scatter_forward(
            dest.contiguous(), index, src.contiguous(), tau, include_self, groups
        )
        ctx.save_for_backward(*original, index, dest_weight, src_weight)
        return output

    @staticmethod
    def backward(ctx, gradient):
        _, _, _, index, dest_weight, src_weight = ctx.saved_tensors
        return gradient * dest_weight, None, gradient[index] * src_weight, None, None


def smooth_scatter_max_native(dest, index, src, *, tau, include_self):
    return _SmoothScatterMax.apply(dest, index, src, tau, include_self)
