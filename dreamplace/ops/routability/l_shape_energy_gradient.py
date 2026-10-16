"""Analytic adjoint of the fixed-epoch routing energy, including rectangle sizes."""

import importlib

import torch


def solve_poisson_maps(
    source,
    bin_area,
    dct2,
    idct2,
    idxst_idct,
    idct_idxst,
    inverse,
    field_x_coeff,
    field_y_coeff,
    fast_mode,
    need_fields,
):
    normalized = source * (1.0 / bin_area)
    spectrum = dct2.forward(normalized)
    field_x = field_y = None
    if need_fields:
        field_x = idxst_idct.forward(spectrum * field_x_coeff).clone()
        field_y = idct_idxst.forward(spectrum * field_y_coeff).clone()
    if fast_mode:
        potential = torch.zeros_like(source)
        energy = source.new_zeros(())
    else:
        potential = idct2.forward(spectrum * inverse).clone()
        potential.mul_(bin_area)
        energy = (potential * normalized).sum()
    return normalized, field_x, field_y, potential, energy


def demand_adjoint(potential, source, capacity, bin_area, beta, padding, padding_mask):
    adjoint = potential * (2.0 * beta / (bin_area * bin_area))
    adjoint.div_(capacity.clamp(min=1e-6))
    adjoint.mul_(source > 0)
    if padding:
        adjoint.masked_fill_(padding_mask, 0)
    return adjoint


def full_segment_backward(ctx, grad_output):
    extension = importlib.import_module(
        "dreamplace.ops.routability.segment_energy_gradient_"
        + ("cuda" if ctx.segment_pos.is_cuda else "cpp")
    )
    weights = (
        ctx.segment_weight
        if isinstance(ctx.segment_weight, torch.Tensor)
        else torch.ones_like(ctx.segment_size_x)
    )
    if ctx.hv_split_active:
        adjoint, adjoint_v = ctx.energy_gradient_h, ctx.energy_gradient_v
        directions = ctx.segment_is_horizontal
    else:
        adjoint = adjoint_v = ctx.energy_gradient
        directions = torch.empty(0, dtype=torch.bool, device=ctx.segment_pos.device)
    gradient_pos, gradient_x, gradient_y = extension.backward(
        ctx.segment_pos.contiguous(),
        ctx.segment_size_x.contiguous(),
        ctx.segment_size_y.contiguous(),
        ctx.ratio.contiguous(),
        weights.contiguous(),
        adjoint.contiguous(),
        adjoint_v.contiguous(),
        directions.contiguous(),
        ctx.xl,
        ctx.yl,
        ctx.bin_size_x,
        ctx.bin_size_y,
    )
    return (
        tuple(value * grad_output for value in (gradient_pos, gradient_x, gradient_y))
        + (None,) * 46
    )
