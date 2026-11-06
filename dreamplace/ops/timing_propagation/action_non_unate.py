"""Interpolate non-unate LUT branches before their rise/fall maximum."""

import torch

from .cell_aat_native import _CellAat


def interpolate_non_unate(
    arcs,
    pins,
    tables,
    smooth,
    alpha,
    target_rows,
    valid,
    inst,
    beta,
    *,
    base_point,
):
    # Split a non-unate arc into its positive/negative transition branches.
    # Interpolating their already-reduced maxima gives a finite secant when
    # the target master's winning branch differs from the current master's.
    branches = []
    for sense in (1, -1):
        signed = arcs.clone()
        signed[:, 4] = sense
        current = _CellAat.apply(signed, *pins, tables, smooth, alpha)
        result = current[:, :4]
        for action in range(beta.shape[1]):
            target = signed.clone()
            target[:, 3] = torch.where(valid[:, action], target_rows[:, action], arcs[:, 3])
            values = _CellAat.apply(target, *pins, tables, smooth, alpha)
            weight = beta[inst.clamp(min=0), action] * valid[:, action] * (inst >= 0)
            result = result + weight[:, None] * (values[:, :4] - current[:, :4])
        branches.append(result)
    positive, negative = branches
    merged = (
        torch.logaddexp(positive * alpha, negative * alpha) / alpha
        if smooth
        else torch.maximum(positive, negative)
    )
    fixed = _CellAat.apply(arcs, *pins, tables, smooth, alpha)
    if base_point:
        # Match native floating-point forward values exactly at beta=0.
        merged = fixed[:, :4].detach() + (merged - merged.detach())
    return torch.cat((merged, fixed[:, 4:]), dim=1)
