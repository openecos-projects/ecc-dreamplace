"""Fixed-master cell LUT queries and six pin derivatives in one native operator."""

import torch
from torch.autograd import Function

from . import lut_entry_2d_op
from .hard_scatter_native import scatter_slew_max
from .scatter_groups import scatter_groups
from .timing_lanes_native import try_timing_lanes_native


class _CellAat(Function):
    @staticmethod
    def forward(ctx, arcs, ra, fa, rs, fs, rc, fc, tables, smooth, alpha):
        original = arcs, ra, fa, rs, fs, rc, fc
        native = lut_entry_2d_op._select_module(ra)
        groups = ()
        if not ra.is_cuda or torch.are_deterministic_algorithms_enabled():
            groups = (
                *scatter_groups(arcs[:, 0], ra.numel()),
                *scatter_groups(arcs[:, 1], ra.numel()),
            )
        arcs = arcs.long().contiguous()
        updates, jac = native.cell_aat_forward(
            arcs, *(p.contiguous() for p in (ra, fa, rs, fs, rc, fc)), tables, smooth, alpha
        )
        ctx.save_for_backward(arcs, jac, *groups, *original, *tables)
        ctx.group_count = len(groups)
        ctx.pin_count = ra.numel()
        return updates

    @staticmethod
    def backward(ctx, gradient):
        arcs, jac = ctx.saved_tensors[:2]
        groups = ctx.saved_tensors[2 : 2 + ctx.group_count]
        grads = lut_entry_2d_op._select_module(jac).cell_aat_backward(
            arcs, jac, gradient.contiguous(), ctx.pin_count, groups
        )
        return None, *grads, None, None, None


def _tables(timing):
    infos = (
        timing.arcs_info.r_delay_luts,
        timing.arcs_info.f_delay_luts,
        timing.arcs_info.r_trans_luts,
        timing.arcs_info.f_trans_luts,
    )
    tensors = []
    for info in infos:
        if info.use_2d_coeff_cache or info.use_2d_coeff_op:
            return None
        for value in (
            info.flat_luts_values,
            info.flat_luts_trans_table,
            info.flat_luts_cap_table,
            info.trans_dims_actual,
            info.cap_dims_actual,
        ):
            if value.requires_grad:
                return None
            tensors.append(value.contiguous())
    return tensors


def try_cell_aat_native(
    timing,
    arcs,
    pins,
    *,
    use_surrogate,
    need_delay_updates,
    global_arc_rows=None,
):
    native = lut_entry_2d_op._select_module(pins[0])
    action_context = getattr(timing, "_active_discrete_action_context", None)
    if (
        pins[0].dtype not in (torch.float32, torch.float64)
        or use_surrogate
        or getattr(timing, "timing_propagation_profile", False)
        or not timing._use_lut_2d_native_op_for_tensor(pins[0])
        or not hasattr(native, "cell_aat_forward")
    ):
        if action_context is not None:
            raise RuntimeError(
                "lut_action requires the fixed-master native cell LUT path"
            )
        return None
    tables = _tables(timing)
    if tables is None:
        if action_context is not None:
            raise RuntimeError(
                "lut_action requires raw 2D LUT tables; cached coefficient "
                "tables are not supported by the action path"
            )
        return None
    from .timing_propagation import (
        SMOOTH_MAX_ALPHA,
        SMOOTH_MAX_ENABLED,
        _force_hard_pairwise_max_enabled,
        scatter_timing_max,
    )

    # RC caps can include appended Steiner nodes; cell arcs index design pins.
    cell_pins = (*pins[:4], *(cap[: pins[0].numel()] for cap in pins[4:]))
    if action_context is not None:
        if global_arc_rows is None:
            raise RuntimeError(
                "lut_action requires global arc rows for the scalar LUT path"
            )
        from .action_cell_aat import apply_action_cell_aat

        global_arc_rows = global_arc_rows.to(device=arcs.device, dtype=torch.long)
        updates = apply_action_cell_aat(
            arcs,
            cell_pins,
            tables,
            SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled(),
            SMOOTH_MAX_ALPHA,
            target_arc_ids=action_context.target_arc_ids.index_select(
                0, global_arc_rows
            ),
            target_senses=action_context.target_senses.index_select(
                0, global_arc_rows
            ),
            valid_arcs=action_context.valid_arcs.index_select(0, global_arc_rows),
            inst_ids=action_context.inst_ids.index_select(0, global_arc_rows),
            beta=action_context.beta,
            context=action_context,
            global_rows=global_arc_rows,
            cap_delta_r=(
                action_context.output_rcap_delta.index_select(0, global_arc_rows)
                if action_context.output_rcap_delta is not None
                else None
            ),
            cap_delta_f=(
                action_context.output_fcap_delta.index_select(0, global_arc_rows)
                if action_context.output_fcap_delta is not None
                else None
            ),
            need_delays=bool(need_delay_updates),
            defer_target_queries=True,
        )
    else:
        smooth = SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled()
        if not torch.is_grad_enabled() and hasattr(native, "cell_aat_values"):
            # Value-only pass (exact action trials, diagnostics): skip the
            # 8x6 LUT Jacobian that _CellAat would materialize for backward.
            updates = native.cell_aat_values(
                arcs.long().contiguous(),
                *(pin.contiguous() for pin in cell_pins),
                tables,
                smooth,
                SMOOTH_MAX_ALPHA,
            )
        else:
            updates = _CellAat.apply(
                arcs,
                *cell_pins,
                tables,
                smooth,
                SMOOTH_MAX_ALPHA,
            )
    outputs = arcs[:, 1]
    lanes = try_timing_lanes_native(
        timing, outputs, pins[:4], tuple(updates[:, i] for i in range(4))
    )
    if lanes is not None:
        delays = tuple(updates[:, i] for i in range(4, 8)) if need_delay_updates else (None,) * 4
        return torch.unique(timing.pin_net[outputs]), *lanes, *delays
    arrivals = tuple(
        scatter_timing_max(
            pins[i],
            outputs,
            updates[:, i],
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
            updates[:, i],
            include_self=True,
        )
        for i in range(2, 4)
    )
    delays = tuple(updates[:, i] for i in range(4, 8)) if need_delay_updates else (None,) * 4
    return torch.unique(timing.pin_net[outputs]), *arrivals, *slews, *delays
