"""Shared state and electrical helpers for discrete timing owners."""

from contextlib import contextmanager

import torch

from dreamplace.flows.diff_sizing_spec import DIFF_SIZING_ACTION_MASKS
from dreamplace.flows.timing_opt_config import DEFAULT_TIMING_OPT_COEFFICIENTS
from dreamplace.flows.timing_phase_coefficients import timing_phase_coefficients
from dreamplace.ops.routability.cooptimization_area import capture_area


def timing_sizing_action_mask(params, action_mask=None):
    """Resolve window/diff-opt defaults while preserving explicit phase masks."""
    if action_mask is not None:
        return action_mask
    action = getattr(params, "timing_sizing_action", "up")
    if action not in DIFF_SIZING_ACTION_MASKS:
        raise ValueError("timing_sizing_action must be one of: up, down, quad")
    return DIFF_SIZING_ACTION_MASKS[action]


@contextmanager
def timing_electrical_window(params, data):
    """Temporarily expose the timing lane used by discrete sizing."""

    size, vt = data.get_continuous_size_parameter(), data.vt_logits
    mode, lane = params.placement_sizing_mode, params.timing_objective_lane
    timing_objective_enabled = params.differentiable_timing_obj
    requires_grad = (
        None if size is None else size.requires_grad,
        None if vt is None else vt.requires_grad,
    )
    params.placement_sizing_mode = "size_only"
    params.timing_objective_lane = "timing_slew_cap"
    params.differentiable_timing_obj = 1
    if size is not None:
        size.requires_grad_(requires_grad=True)
    if vt is not None:
        vt.requires_grad_(requires_grad=True)
    try:
        yield
    finally:
        params.placement_sizing_mode = mode
        params.timing_objective_lane = lane
        params.differentiable_timing_obj = timing_objective_enabled
        if size is not None:
            size.requires_grad_(requires_grad[0])
            size.grad = None
        if vt is not None:
            vt.requires_grad_(requires_grad[1])
            vt.grad = None


@contextmanager
def timing_sizing_coefficients(params, model):
    """Apply sizing-window setup/DRV/hold weights and restore the live GP policy."""

    policy = getattr(
        params,
        "timing_opt_coefficients",
        DEFAULT_TIMING_OPT_COEFFICIENTS,
    )
    fixed = None if policy["mode"] == "inherit" else {
        name: value for name, value in policy.items() if name != "mode"
    }
    with timing_phase_coefficients(model, fixed) as effective:
        yield {"mode": policy["mode"], **effective}


def virtual_timing_area(owner, model, state=None):
    """Return the area represented by candidate or segment virtual buffers."""

    if owner.lane is None and state is None:
        return 0.0
    state = owner.lane._current_state() if state is None else state
    if state is None:
        return 0.0
    if getattr(state, "is_typed_candidate", False):
        cell_ids = state.drive_type_cell_ids.to(
            device=owner.data.flat_libcell_width.device,
        )
        widths = owner.data.flat_libcell_width[cell_ids].double()
        heights = owner.data.flat_libcell_height[cell_ids].double()
        counts = state.activation_param.detach().double().sum(dim=0).to(widths.device)
        padding = 2.0 * float(getattr(owner.params, "cell_padding_x", 0.0))
        return float(((widths + padding) * heights * counts).sum())
    width, height = model._segment_virtual_buffer_size()
    if hasattr(state, "z_param"):
        count = state.z_param.detach().sum()
    elif hasattr(state, "activation_param"):
        count = state.activation_param.detach().sum()
    else:
        return 0.0
    return float(count) * width * height


def timing_area_fits(owner, model, capacity):
    """Evaluate current physical plus virtual timing area."""

    return capture_area(
        owner.data,
        owner.placedb,
        owner.virtual_area(model),
        capacity=capacity,
    ).fits()


def segment_counts(state):
    """Return positive virtual segment counts keyed by stable topology IDs."""

    if not hasattr(state, "z_param"):
        return {}
    rows = state.positive_segment_indices()
    return {
        (int(net), int(parent), int(child)): float(count)
        for net, parent, child, count in zip(
            state.segment_net_id[rows].detach().cpu().tolist(),
            state.parent_node_id[rows].detach().cpu().tolist(),
            state.child_node_id[rows].detach().cpu().tolist(),
            state.z_param[rows].detach().cpu().tolist(),
            strict=True,
        )
    }


def transfer_segment_counts(state, counts):
    """Restore retained segment counts by their topology identity."""

    if not counts or not hasattr(state, "z_param"):
        return
    candidate_keys = torch.stack(
        (
            state.segment_net_id,
            state.parent_node_id,
            state.child_node_id,
        ),
        dim=1,
    ).long()
    query_keys = torch.tensor(
        list(counts),
        dtype=torch.long,
        device=candidate_keys.device,
    )
    _, inverse = torch.unique(
        torch.cat((candidate_keys, query_keys)),
        dim=0,
        return_inverse=True,
    )
    candidate_ids = inverse[: state.num_segments]
    query_ids = inverse[state.num_segments :]
    row_by_key = torch.full(
        (int(inverse.max()) + 1,),
        -1,
        dtype=torch.long,
        device=inverse.device,
    )
    row_by_key[candidate_ids] = torch.arange(
        state.num_segments,
        device=inverse.device,
    )
    rows = row_by_key[query_ids]
    if bool((rows < 0).any()):
        raise RuntimeError("retained buffer segment lost its identity at window preparation")
    with torch.no_grad():
        state.z_param[rows] = torch.tensor(
            list(counts.values()),
            dtype=state.z_param.dtype,
            device=state.z_param.device,
        )


__all__ = [
    "segment_counts",
    "timing_area_fits",
    "timing_electrical_window",
    "timing_sizing_coefficients",
    "transfer_segment_counts",
    "virtual_timing_area",
]
