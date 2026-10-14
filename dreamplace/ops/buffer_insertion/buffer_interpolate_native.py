"""One native interpolation pass and one host copy for buffer diagnostics."""

import torch
from torch.autograd import Function

from dreamplace.ops.timing_propagation import lut_entry_2d_op

from .buffer_interpolation import _clamp_status_from_bounds, _validate_per_size_tensor

TABLE_NAMES = ("per_size_input_cap", "per_size_delay", "per_size_output_slew")


class _BufferInterpolate(Function):
    @staticmethod
    def forward(ctx, index, cap, delay, slew):
        original = index, cap, delay, slew
        index, cap, delay, slew = (t.contiguous() for t in original)
        values, clipped, lo, hi, alpha, metadata = lut_entry_2d_op._select_module(
            index
        ).buffer_interpolate_forward(
            index,
            cap,
            delay,
            slew,
        )
        ctx.save_for_backward(*original, index, cap, delay, slew, lo, hi, alpha)
        nondifferentiable = [lo, hi, metadata]
        outputs = tuple(values[:, i] for i in range(3))
        for field, output in enumerate(outputs):
            if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[field + 1]:
                nondifferentiable.append(output)
        if not ctx.needs_input_grad[0]:
            nondifferentiable.extend((clipped, alpha))
        ctx.mark_non_differentiable(*nondifferentiable)
        ctx.set_materialize_grads(False)
        return (*outputs, clipped, lo, hi, alpha, metadata)

    @staticmethod
    def backward(ctx, grad_cap, grad_delay, grad_slew, grad_clipped, _, __, grad_alpha, ___):
        _, _, _, _, index, cap, delay, slew, lo, hi, alpha = ctx.saved_tensors
        grad_values = torch.stack(
            tuple(
                torch.zeros_like(index) if value is None else value
                for value in (grad_cap, grad_delay, grad_slew)
            ),
            dim=1,
        )
        if grad_clipped is None:
            grad_clipped = torch.zeros_like(index)
        if grad_alpha is None:
            grad_alpha = torch.zeros_like(index)
        return tuple(
            lut_entry_2d_op._select_module(index).buffer_interpolate_backward(
                index,
                cap,
                delay,
                slew,
                lo,
                hi,
                alpha,
                grad_values.contiguous(),
                grad_clipped.contiguous(),
                grad_alpha.contiguous(),
            )
        )


def try_buffer_interpolate_native(index, tables, coordinate_source):
    native = lut_entry_2d_op._select_module(index)
    if index.dtype not in (torch.float32, torch.float64) or not hasattr(
        native, "buffer_interpolate_forward"
    ):
        return None
    tables = tuple(
        _validate_per_size_tensor(name, table, bsu_index=index, check_finite=False)
        for name, table in zip(TABLE_NAMES, tables, strict=True)
    )
    sizes = tables[1].size(1)
    for name, table in zip(TABLE_NAMES, tables, strict=True):
        if table.size(1) != sizes:
            raise ValueError(f"{name} legal-size dimension does not match per_size_delay")
    cap, delay, slew, clipped, lo, hi, alpha, metadata = _BufferInterpolate.apply(index, *tables)
    flags, *raw = metadata.cpu().tolist()
    for bit, name in enumerate(TABLE_NAMES):
        if int(flags) & (1 << bit):
            raise ValueError(f"{name} contains non-finite values")
    if int(flags) & 8:
        raise ValueError("bsu_index contains NaN values")
    low, high = any(value < 0 for value in raw), any(value > sizes - 1 for value in raw)
    clipped_values = [max(0.0, min(float(sizes - 1), value)) for value in raw]
    status = _clamp_status_from_bounds(
        low=low,
        high=high,
        raw_min=min(raw) if raw else None,
        raw_max=max(raw) if raw else None,
        clipped_min=min(clipped_values) if raw else None,
        clipped_max=max(clipped_values) if raw else None,
        legal_buffer_count=sizes,
        coordinate_source=coordinate_source,
    )
    return {
        "buffer_input_cap": cap,
        "buffer_delay": delay,
        "buffer_output_slew": slew,
        "bsu_index": clipped,
        "bsu_index_lo": lo,
        "bsu_index_hi": hi,
        "bsu_index_alpha": alpha,
        "bsu_index_status": status,
    }
