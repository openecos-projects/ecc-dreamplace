"""Net arrival/slew arithmetic and deterministic pin derivatives on CPU or CUDA."""

import torch
from torch.autograd import Function

from . import lut_entry_2d_op
from .hard_scatter_native import scatter_slew_max
from .scatter_groups import scatter_groups
from .timing_lanes_native import try_timing_lanes_native


class _NetAat(Function):
    @staticmethod
    def forward(ctx, arcs, ra, fa, rs, fs, rd, fd, ri, fi, eps):
        original_arcs = arcs
        groups = (*scatter_groups(arcs[:, 0], ra.numel()), *scatter_groups(arcs[:, 1], ra.numel()))
        arcs = arcs.long().contiguous()
        updates, jac = lut_entry_2d_op._select_module(ra).net_aat_forward(
            arcs, *(p.contiguous() for p in (ra, fa, rs, fs, rd, fd, ri, fi)), eps
        )
        ctx.save_for_backward(arcs, jac, *groups, original_arcs, rs, fs)
        ctx.pins = ra.numel()
        ctx.set_materialize_grads(False)
        return tuple(updates[:, i] for i in range(4))

    @staticmethod
    def backward(ctx, *gradients):
        arcs, jac, in_order, in_offsets, out_order, out_offsets, _, _, _ = ctx.saved_tensors
        groups = (in_order, in_offsets, out_order, out_offsets)
        gradient = torch.stack(
            tuple(jac.new_zeros(jac.size(0)) if value is None else value for value in gradients),
            dim=1,
        )
        grads = lut_entry_2d_op._select_module(jac).net_aat_backward(
            arcs,
            jac,
            gradient.contiguous(),
            ctx.pins,
            groups,
            gradients[2] is not None,
            gradients[3] is not None,
        )
        return None, *grads, None


def try_net_aat_native(timing, arcs, pins, eps):
    native = lut_entry_2d_op._select_module(pins[0])
    if pins[0].dtype not in (torch.float32, torch.float64) or not hasattr(
        native, "net_aat_forward"
    ):
        return None
    from .timing_propagation import scatter_timing_max

    # RC vectors may append Steiner nodes; arcs address design pins only.
    inputs = (*pins[:4], *(value[: pins[0].numel()] for value in pins[4:]))
    updates = _NetAat.apply(arcs, *inputs, eps)
    outputs = arcs[:, 1]
    lanes = try_timing_lanes_native(timing, outputs, pins[:4], updates)
    if lanes is not None:
        return lanes
    arrivals = tuple(
        scatter_timing_max(
            pins[i],
            outputs,
            updates[i],
            include_self=True,
            mode=timing.timing_aggregation_mode,
            tau_ps=timing.timing_aggregation_tau_ps,
        )
        for i in range(2)
    )
    slews = tuple(
        scatter_slew_max(
            pins[i],
            outputs,
            updates[i],
            include_self=True,
        )
        for i in range(2, 4)
    )
    return (*arrivals, *slews)
