"""Immutable buffer tables queried with indexed native LUT interpolation."""

import torch
from torch.autograd import Function

from dreamplace.ops.timing_propagation import lut_entry_2d_op


class _BufferLookup(Function):
    @staticmethod
    def forward(ctx, slew, load, prepared, original_tables):
        slew_axis, load_axis, delay, output_slew, slew_dims, load_dims = prepared
        sizes = delay.size(0)
        shape = (slew.numel(), sizes)
        rows = torch.arange(sizes, device=slew.device).expand(shape).reshape(-1).contiguous()
        x = slew[:, None].expand(shape).reshape(-1).contiguous()
        y = load[:, None].expand(shape).reshape(-1).contiguous()
        results = [
            lut_entry_2d_op._select_module(slew).buffer_indexed_forward(
                x, y, rows, slew_axis, load_axis, values, slew_dims, load_dims
            )
            for values in (delay, output_slew)
        ]
        ctx.save_for_backward(
            slew,
            load,
            *original_tables,
            *(torch.stack([result[i].reshape(shape) for result in results]) for i in (1, 2)),
        )
        return torch.stack([result[0].reshape(shape) for result in results])

    @staticmethod
    def backward(ctx, gradient):
        *_, ds, dc = ctx.saved_tensors
        return (
            (gradient * ds).sum(dim=0).sum(dim=1),
            (gradient * dc).sum(dim=0).sum(dim=1),
            None,
            None,
        )


def try_buffer_device_lookup_native(slew, load, tables):
    native = lut_entry_2d_op._select_module(slew)
    if not hasattr(native, "buffer_indexed_forward"):
        return None
    sources = tuple(
        tables[key]
        for key in ("input_slew_axis", "output_load_axis", "delay_lut", "output_slew_lut")
    )
    if any(value.requires_grad for value in sources):
        return None
    cached = tables.get("indexed_buffer_lookup")
    if cached is None or not all(
        old is value and version == value._version
        for old, version, value in zip(cached[0], cached[1], sources, strict=True)
    ):
        slew_axis, load_axis, delay, output_slew = sources
        sizes = delay.size(0)
        prepared = (
            slew_axis.expand(sizes, -1).contiguous(),
            load_axis.expand(sizes, -1).contiguous(),
            delay.reshape(sizes, -1).contiguous(),
            output_slew.reshape(sizes, -1).contiguous(),
            torch.full((sizes,), slew_axis.numel(), device=slew.device, dtype=torch.long),
            torch.full((sizes,), load_axis.numel(), device=slew.device, dtype=torch.long),
        )
        cached = (sources, tuple(value._version for value in sources), prepared)
        tables["indexed_buffer_lookup"] = cached
    result = _BufferLookup.apply(slew, load, cached[2], sources)
    return {"per_size_delay": result[0], "per_size_output_slew": result[1]}
