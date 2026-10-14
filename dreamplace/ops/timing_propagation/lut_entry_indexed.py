"""Query immutable scalar, 1D and 2D Liberty tables without per-type gathers."""

from torch.autograd import Function

from . import lut_entry_2d_op


class _IndexedLut(Function):
    @staticmethod
    def forward(ctx, trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims):
        output, dt, dc = lut_entry_2d_op._select_module(trans).indexed_forward(
            trans.contiguous(),
            caps.contiguous(),
            indices.long().contiguous(),
            trans_axes,
            cap_axes,
            values,
            trans_dims,
            cap_dims,
        )
        ctx.save_for_backward(
            trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims, dt, dc
        )
        return output

    @staticmethod
    def backward(ctx, gradient):
        *_, dt, dc = ctx.saved_tensors
        return gradient * dt, gradient * dc, None, None, None, None, None, None


def lut_entry_indexed(trans, caps, indices, info):
    return _IndexedLut.apply(
        trans,
        caps,
        indices,
        info.flat_luts_trans_table,
        info.flat_luts_cap_table,
        info.flat_luts_values,
        info.trans_dims_actual,
        info.cap_dims_actual,
    )
