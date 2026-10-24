"""Project matching sparse arrival/required pairs for existing physical reports.

The optimization objective always uses legal state/check pairs. These detached
views serve pin-slack consumers and diagnostic reports, never timing loss.
"""

import torch

from .hard_scatter_native import scatter_slew_max


@torch.no_grad()
def propagate_required(timing, model, result, trace):
    """Reuse the scalar cell RAT operator on compact sparse state indices."""
    from .timing_propagation import scatter_timing_min

    required = result.required.new_full((len(model.state_pin), 2), 2e8)
    keys = model.check_state * 2 + model.check_transition
    required.view(-1).scatter_reduce_(
        0, keys, result.required[model.legal], reduce="amin"
    )
    for entry in reversed(trace):
        level = entry.level
        lanes = tuple(required[:, lane][level.states] for lane in range(2))
        if entry.net_delays is not None:
            lanes = tuple(
                scatter_timing_min(
                    value,
                    level.net[:, 0],
                    value[level.net[:, 1]] - entry.net_delays[:, lane],
                    include_self=True,
                    mode=timing.timing_aggregation_mode,
                    tau_ps=timing.timing_aggregation_tau_ps,
                )
                for lane, value in enumerate(lanes)
            )
        if entry.cell_delays is not None:
            _, *lanes = timing.calculate_cell_rat_level(
                level.cell[: level.arrival_cell_count],
                *lanes,
                *entry.cell_delays.unbind(1),
            )
        required[level.states] = torch.stack(lanes, dim=1)
    return required


@torch.no_grad()
def physical_projection(timing, model, result, state_required=None):
    count = timing.num_pins
    arrivals = torch.stack(
        tuple(
            scatter_slew_max(value.new_full((count,), -2e8), model.state_pin, value)
            for value in result.lanes[:2]
        ),
        dim=1,
    )
    required = arrivals.new_full((count, 2), 2e8)
    slack = required.clone()
    if state_required is None:
        states = model.check_state
        edges = model.check_transition
        values = result.required[model.legal]
        slacks = result.check_slack
    else:
        states = torch.arange(
            len(model.state_pin), device=model.device
        ).repeat_interleave(2)
        edges = torch.arange(2, device=model.device).repeat(len(model.state_pin))
        values = state_required.flatten()
        slacks = values - torch.stack(result.lanes[:2], dim=1).flatten()
    keys = model.state_pin[states] * 2 + edges
    minimum = -scatter_slew_max(slack.flatten().neg(), keys, -slacks)
    order = torch.arange(len(keys), device=keys.device)
    candidates = torch.where(slacks == minimum[keys], order, len(keys))
    first = torch.full((count * 2,), len(keys), device=keys.device, dtype=torch.long)
    first.scatter_reduce_(0, keys, candidates, reduce="amin", include_self=True)
    selected = first < len(keys)
    rows = first[selected]
    required.flatten()[selected] = values[rows]
    arrivals.flatten()[selected] = torch.stack(result.lanes[:2], dim=1)[
        states[rows], edges[rows]
    ]
    return (
        tuple(arrivals.unbind(1)),
        tuple(required.unbind(1)),
        minimum.reshape(count, 2),
    )


def physical_slew(model, lanes, count):
    """Preserve the live physical slew graph for the original DRV objective."""
    return tuple(
        scatter_slew_max(value.new_zeros(count), model.state_pin, value)
        for value in lanes[2:]
    )


@torch.no_grad()
def physical_cell_delays(timing, trace):
    values = timing.inrdelays.new_full(
        (len(timing.flat_inst_arcs_by_level), 4), -torch.inf
    )
    for entry in trace:
        if entry.cell_delays is not None:
            indices = entry.level.base_cells[: entry.level.arrival_cell_count]
            values[indices] = entry.cell_delays
    return tuple(values.unbind(1))
