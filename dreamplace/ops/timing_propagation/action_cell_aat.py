"""Current plus four target-LUT queries for discrete action gradients."""

import torch

from . import lut_entry_2d_op
from .cell_aat_native import _CellAat


class _ActionCellAatContribution(torch.autograd.Function):
    """Zero-valued forward whose backward queries only active adjoint arcs."""

    @staticmethod
    def forward(
        ctx,
        arcs,
        beta,
        current,
        target_arc_ids,
        target_senses,
        valid_arcs,
        inst_ids,
        ra,
        fa,
        rs,
        fs,
        rc,
        fc,
        tables,
        smooth,
        alpha,
        num_instances,
        context,
        global_rows,
        cap_delta_r,
        cap_delta_f,
    ):
        ctx.save_for_backward(
            arcs,
            beta,
            current,
            target_arc_ids,
            target_senses,
            valid_arcs,
            inst_ids,
            ra,
            fa,
            rs,
            fs,
            rc,
            fc,
            global_rows,
            cap_delta_r,
            cap_delta_f,
            *tables,
        )
        ctx.smooth = bool(smooth)
        ctx.alpha = float(alpha)
        ctx.num_instances = int(num_instances)
        ctx.context = context
        return torch.zeros_like(current)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        (
            arcs,
            beta,
            current,
            target_arc_ids,
            target_senses,
            valid_arcs,
            inst_ids,
            ra,
            fa,
            rs,
            fs,
            rc,
            fc,
            global_rows,
            cap_delta_r,
            cap_delta_f,
        ) = saved[:16]
        tables = list(saved[16:36])
        native = lut_entry_2d_op._select_module(ra)
        grad_beta, finite = native.cell_aat_action_backward(
            arcs.long().contiguous(),
            target_arc_ids.long().contiguous(),
            target_senses.long().contiguous(),
            valid_arcs.bool().contiguous(),
            inst_ids.long().contiguous(),
            current.contiguous(),
            grad_output.contiguous(),
            *(pin.contiguous() for pin in (ra, fa, rs, fs, rc, fc)),
            cap_delta_r.contiguous(),
            cap_delta_f.contiguous(),
            tables,
            ctx.num_instances,
            ctx.smooth,
            ctx.alpha,
        )
        if ctx.context is not None:
            ctx.context.native_action_backward_used = True
            if global_rows is not None:
                rows = global_rows.to(device=finite.device).long()
                ctx.context.valid_arcs[rows] &= finite
            else:
                ctx.context._action_backward_all_finite = bool(finite.all().item())
        return (None, grad_beta) + (None,) * 19


def _deferred_action_values(
    arcs,
    pins,
    tables,
    smooth,
    alpha,
    *,
    target_arc_ids,
    target_senses,
    valid_arcs,
    inst_ids,
    beta,
    context,
    global_rows,
    cap_delta_r,
    cap_delta_f,
):
    # At beta=0 the current master supplies the state Jacobian. This includes
    # its own cap -> upstream RC -> input slew feedback as well as all other
    # actions. The direct master difference is a separate partial derivative.
    current = _CellAat.apply(arcs, *pins, tables, smooth, alpha)
    contribution = _ActionCellAatContribution.apply(
        arcs,
        beta,
        current,
        target_arc_ids,
        target_senses,
        valid_arcs,
        inst_ids,
        *pins,
        tuple(tables),
        smooth,
        alpha,
        int(beta.shape[0]),
        context,
        global_rows,
        cap_delta_r,
        cap_delta_f,
    )
    return current + contribution


def _target_rows(arcs, target_arc_ids, target_senses):
    action_count = int(target_arc_ids.shape[1])
    if int(arcs.shape[0]) == 0:
        return arcs[:, None, :].expand(0, action_count, arcs.shape[1]).reshape(0, arcs.shape[1])
    rows = arcs[:, None, :].expand(int(arcs.shape[0]), action_count, int(arcs.shape[1])).clone()
    valid_target = target_arc_ids >= 0
    rows[..., 3] = torch.where(
        valid_target,
        target_arc_ids.long(),
        arcs[:, 3].unsqueeze(1),
    )
    rows[..., 4] = torch.where(
        valid_target,
        target_senses.long(),
        arcs[:, 4].unsqueeze(1),
    )
    return rows.reshape(-1, int(arcs.shape[1]))




def apply_action_cell_aat(
    arcs,
    pins,
    tables,
    smooth,
    alpha,
    *,
    target_arc_ids,
    target_senses,
    valid_arcs,
    inst_ids,
    beta,
    context=None,
    global_rows=None,
    cap_delta_r=None,
    cap_delta_f=None,
    need_delays=True,
    use_native_action=True,
    defer_target_queries=False,
):
    """Return ``current + sum_a beta[a] * (target[a] - current)``.

    ``pins`` are the six current-cell electrical vectors.  Target rows reuse
    their current in/out lane indices and only replace the Liberty row and
    timing sense.  The discrete master change is the base-point difference
    ``lambda^T (X_target - X_current)``; those two LUT queries are constants.
    All state Jacobians stay connected, including arrival identity and slew
    propagation through multiple cells. The direct master partial derivative
    is queried at the current operating point; cap-induced operating-point
    changes reach the objective through the ordinary timing graph.
    """

    if int(arcs.shape[0]) == 0:
        return pins[0].new_zeros((0, 8))
    if context is not None:
        context.action_cell_aat_calls += 1
    non_unate = (arcs[:, 4] != 1) & (arcs[:, 4] != -1)
    non_unate &= ((target_senses == arcs[:, 4, None]) | ~valid_arcs).all(dim=1)
    if bool(non_unate.any()):
        from .action_non_unate import interpolate_non_unate

        rows = torch.nonzero(non_unate).flatten()
        result = pins[0].new_zeros((int(arcs.shape[0]), 8))
        values = interpolate_non_unate(
            arcs[rows],
            pins,
            tables,
            smooth,
            alpha,
            target_arc_ids[rows],
            valid_arcs[rows],
            inst_ids[rows],
            beta,
            base_point=defer_target_queries and not getattr(context, "evaluate_targets", False),
        )
        result = result.index_copy(0, rows, values)
        other = torch.nonzero(~non_unate).flatten()
        if other.numel():
            values = apply_action_cell_aat(
                arcs[other],
                pins,
                tables,
                smooth,
                alpha,
                target_arc_ids=target_arc_ids[other],
                target_senses=target_senses[other],
                valid_arcs=valid_arcs[other],
                inst_ids=inst_ids[other],
                beta=beta,
                context=context,
                global_rows=None if global_rows is None else global_rows[other],
                cap_delta_r=None if cap_delta_r is None else cap_delta_r[other],
                cap_delta_f=None if cap_delta_f is None else cap_delta_f[other],
                need_delays=need_delays,
                use_native_action=use_native_action,
                defer_target_queries=defer_target_queries,
            )
            result = result.index_copy(0, other, values)
        return result
    native = lut_entry_2d_op._select_module(pins[0])
    if cap_delta_r is None:
        cap_delta_r = torch.zeros(
            (int(arcs.shape[0]), int(target_arc_ids.shape[1])),
            device=arcs.device,
            dtype=pins[0].dtype,
        )
    if cap_delta_f is None:
        cap_delta_f = torch.zeros_like(cap_delta_r)
    if (
        context is not None
        and defer_target_queries
        and not getattr(context, "evaluate_targets", False)
        and global_rows is not None
        and hasattr(native, "cell_aat_action_backward")
    ):
        context.native_values_used = hasattr(native, "cell_aat_values")
        context.native_action_values_used = False
        return _deferred_action_values(
            arcs,
            pins,
            tables,
            smooth,
            alpha,
            target_arc_ids=target_arc_ids,
            target_senses=target_senses,
            valid_arcs=valid_arcs,
            inst_ids=inst_ids,
            beta=beta,
            context=context,
            global_rows=global_rows,
            cap_delta_r=cap_delta_r,
            cap_delta_f=cap_delta_f,
        )
    if context is not None:
        use_native_action = use_native_action and not getattr(context, "evaluate_targets", False)
        context.native_values_used = hasattr(native, "cell_aat_values")
        context.native_action_values_used = bool(use_native_action) and hasattr(
            native, "cell_aat_action_values"
        )
        action_values_fn = (
            getattr(native, "cell_aat_action_values", None) if use_native_action else None
        )
        if action_values_fn is not None:
            # One fused pass computes the current 8-lane value and the four
            # target arrival/slew deltas without materializing the 5x rows or
            # the unused target delay lanes.
            current, delta, finite = action_values_fn(
                arcs.long().contiguous(),
                target_arc_ids.long().contiguous(),
                target_senses.long().contiguous(),
                valid_arcs.bool().contiguous(),
                *(pin.contiguous() for pin in pins),
                tables,
                bool(need_delays),
                smooth,
                alpha,
            )
            current = _CellAat.apply(arcs, *pins, tables, smooth, alpha)
            delta = delta.detach()
            effective_valid = valid_arcs.bool() & finite.bool()
        else:
            delta = None
            effective_valid = None
    else:
        action_values_fn = None
        effective_valid = None
        delta = None

    if context is not None and action_values_fn is not None:
        # The fused delta is the local master partial. Current state
        # derivatives are supplied by _CellAat.
        pass
    elif context is not None:
        # Explicit interpolation for debug forwards, including nonzero beta.
        # Both master queries use the live operating point.
        target_arcs = _target_rows(arcs, target_arc_ids, target_senses)
        combined = torch.cat((arcs, target_arcs), dim=0)
        values = _CellAat.apply(combined, *pins, tables, smooth, alpha)
        current = values[: int(arcs.shape[0])]
        target = values[int(arcs.shape[0]) :].reshape(
            int(arcs.shape[0]),
            int(target_arc_ids.shape[1]),
            8,
        )
    else:
        current = _CellAat.apply(arcs, *pins, tables, smooth, alpha)
        target_arcs = _target_rows(arcs, target_arc_ids, target_senses)
        target = _CellAat.apply(target_arcs, *pins, tables, smooth, alpha)
        target = target.reshape(
            int(arcs.shape[0]),
            int(target_arc_ids.shape[1]),
            8,
        )
    if effective_valid is None:
        finite = (
            torch.isfinite(current[:, :4]).unsqueeze(1) & torch.isfinite(target[..., :4])
        ).all(dim=-1)
        effective_valid = valid_arcs.bool() & finite
        delta = torch.where(
            effective_valid.unsqueeze(-1),
            target - current.unsqueeze(1),
            torch.zeros_like(target),
        )
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        delta[..., 4:] = 0.0
    if context is not None and global_rows is not None:
        rows = global_rows.to(effective_valid.device).long()
        context.valid_arcs[rows] &= effective_valid
    inst_ids = inst_ids.long()
    safe_inst_ids = inst_ids.clamp(min=0)
    weights = (
        beta[safe_inst_ids]
        * effective_valid.to(beta.dtype)
        * (inst_ids >= 0).to(beta.dtype).unsqueeze(1)
    )
    contribution = (weights.unsqueeze(-1) * delta).sum(dim=1)
    return current + contribution


def refresh_action_valid_instances(context):
    """Recompute the all-arcs mask after target-value finiteness is known."""

    num_instances = int(context.beta.shape[0])
    num_actions = int(context.beta.shape[1])
    previous_valid = context.valid_instances.clone()
    valid = torch.zeros_like(context.valid_instances)
    rows = context.inst_ids >= 0
    if not bool(rows.any()):
        context.valid_instances.zero_()
        return context.valid_instances
    instance_ids = context.inst_ids[rows]
    counts = torch.zeros(
        (num_instances, num_actions),
        dtype=torch.long,
        device=context.valid_instances.device,
    )
    bad = torch.zeros_like(counts)
    for action in range(num_actions):
        counts[:, action].scatter_add_(
            0,
            instance_ids,
            torch.ones_like(instance_ids),
        )
        bad[:, action].scatter_add_(
            0,
            instance_ids,
            (~context.valid_arcs[rows, action]).long(),
        )
    valid = (counts > 0) & (bad == 0)
    context.valid_instances.copy_(valid & context.valid_instances)
    timing_invalid = previous_valid & ~context.valid_instances
    actions = context.mapping_summary.get("actions")
    if isinstance(actions, dict):
        for action_index, action in enumerate(context.action_names):
            summary = actions.get(action)
            if isinstance(summary, dict):
                summary["timing_invalid_instance_count"] = int(
                    timing_invalid[:, action_index].sum().item()
                )
                summary["timing_valid_instance_count"] = int(
                    context.valid_instances[:, action_index].sum().item()
                )
    return context.valid_instances


__all__ = [
    "apply_action_cell_aat",
    "refresh_action_valid_instances",
]
