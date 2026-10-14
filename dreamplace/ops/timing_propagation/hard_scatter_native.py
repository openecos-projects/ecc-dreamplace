"""First-order slew max derivatives with Torch's original tie convention."""

import torch
from torch.autograd import Function


class _HardScatterMax(Function):
    @staticmethod
    def forward(ctx, dest, index, src, include_self):
        from . import lut_entry_2d_op

        original_inputs = dest, index, src
        index = index.long().contiguous()
        dest, src = dest.contiguous(), src.contiguous()
        output, counts, touched = lut_entry_2d_op._select_module(dest).hard_scatter_forward(
            dest,
            index,
            src,
            include_self,
        )
        # Retain original version counters as well as contiguous kernel inputs.
        ctx.save_for_backward(*original_inputs, dest, index, src, output, counts, touched)
        ctx.include_self = include_self
        return output

    @staticmethod
    def backward(ctx, gradient):
        from . import lut_entry_2d_op

        _, _, _, dest, index, src, output, counts, touched = ctx.saved_tensors
        grad_dest, grad_src = lut_entry_2d_op._select_module(dest).hard_scatter_backward(
            dest,
            index,
            src,
            output,
            counts,
            touched,
            gradient.contiguous(),
            ctx.include_self,
        )
        return grad_dest, None, grad_src, None


def scatter_slew_max(dest, index, src, *, include_self=True):
    if torch.is_grad_enabled() and (dest.requires_grad or src.requires_grad):
        from . import lut_entry_2d_op

        native = lut_entry_2d_op._select_module(dest)
        if hasattr(native, "hard_scatter_forward") and hasattr(native, "hard_scatter_backward"):
            return _HardScatterMax.apply(dest, index, src, include_self)
    return torch.scatter_reduce(
        dest, 0, index.long(), src, reduce="amax", include_self=include_self
    )
