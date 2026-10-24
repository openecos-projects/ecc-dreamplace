"""Sparse semantic states using the existing native electrical operators.

Each edge batch addresses state IDs, while physical load/RC and Liberty rows
retain their base identity. Only setup builds batches or transfers indices.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .cell_aat_native import _CellAat
from .net_aat_native import _NetAat
from .timing_lanes_native import _TimingLanes
from .hard_scatter_native import scatter_slew_max


@dataclass(frozen=True)
class _SparseTimingLevel:
    """Local native state IDs with maps to global arrival, RC and slew owners."""

    states: torch.Tensor
    pins: torch.Tensor
    slew_groups: torch.Tensor
    cell: torch.Tensor
    base_cells: torch.Tensor
    arrival_cell_count: int
    net: torch.Tensor
    arrival_destinations: torch.Tensor
    arrival_states: torch.Tensor
    slew_index: torch.Tensor
    slew_destinations: torch.Tensor
    adjustment: torch.Tensor


@dataclass(frozen=True)
class SparseTimingResult:
    lanes: tuple[torch.Tensor, ...]
    required: torch.Tensor
    check_slack: torch.Tensor
    endpoint_pins: torch.Tensor
    endpoint_slack: torch.Tensor
    wns: torch.Tensor
    tns: torch.Tensor


@dataclass(frozen=True)
class SparseTimingTrace:
    level: _SparseTimingLevel
    cell_delays: torch.Tensor | None
    net_delays: torch.Tensor | None


class SparseTimingSchedule:
    def __init__(
        self,
        view,
        base_cell_arcs,
        *,
        device,
        dtype,
        check_mask=None,
        aggregation_mode="hard",
        aggregation_tau_ps=2.0,
    ):
        view.require_supported()
        arrays = view.arrays
        self.dtype = dtype
        self.device = torch.device(device)
        self.aggregation_mode = str(aggregation_mode or "hard")
        self.aggregation_tau_ps = float(aggregation_tau_ps)
        if self.aggregation_tau_ps <= 0.0:
            raise ValueError("sparse endpoint aggregation tau must be positive")
        self.state_pin = torch.as_tensor(arrays["state_pin"], device=device)
        self.state_slew_group = torch.as_tensor(
            arrays["state_slew_group"], device=device
        )
        self.num_slew_groups = int(arrays["state_slew_group"].max(initial=-1)) + 1
        self.num_pins = len(arrays["pin_state_start"]) - 1
        self.valid = torch.as_tensor(arrays["state_valid"], device=device).bool()
        self.all_transitions_valid = bool(arrays["state_valid"].all())
        self.seed = torch.as_tensor(arrays["state_seed"], device=device).bool()
        self.reference = torch.as_tensor(
            arrays["state_reference"], device=device, dtype=dtype
        )
        self.check_index = torch.as_tensor(arrays["check_index"], device=device)
        self.check_values = torch.as_tensor(
            arrays["check_values"], device=device, dtype=dtype
        )
        legal = arrays["check_index"][:, 6].astype(bool)
        if check_mask is not None:
            legal = legal & check_mask
        self.legal = torch.as_tensor(legal, device=device)
        self.check_state = self.check_index[self.legal, 0]
        self.check_transition = self.check_index[self.legal, 3]
        # OpenSTA TNS counts the worst legal setup slack once per endpoint.
        # Multiple domains, capture edges and transitions are candidates.
        endpoint_pin = arrays["state_pin"][arrays["check_index"][legal, 0]]
        pins, check_to_endpoint = np.unique(endpoint_pin, return_inverse=True)
        self.endpoint_pins = torch.as_tensor(pins, device=device)
        self.check_to_endpoint = torch.as_tensor(check_to_endpoint, device=device)
        self.levels = self._levels(
            arrays, np.asarray(base_cell_arcs, dtype=np.int64).reshape(-1, 8)
        )

    def _levels(self, arrays, base_cell):
        def index(value):
            return torch.as_tensor(value, device=self.device, dtype=torch.int64)

        state_level = arrays["state_level"]
        groups = []
        for name in ("cell_edges", "net_edges", "slew_only_cell_edges"):
            edges = arrays[name]
            levels = state_level[edges[:, 2]]
            if np.any(state_level[edges[:, 1]] >= levels):
                raise RuntimeError(f"sparse {name} is not a forward DAG edge")
            order = np.argsort(levels, kind="stable")
            edges = edges[order]
            if name != "net_edges":
                rows = base_cell[edges[:, 0]].copy()
                rows[:, :2] = edges[:, 1:]
            else:
                rows = edges[:, 1:].copy()
                adjustments = arrays["net_threshold_adjustments"][order]
            offsets = np.r_[
                0,
                np.cumsum(
                    np.bincount(levels, minlength=int(state_level.max(initial=0)) + 1)
                ),
            ]
            groups.append((rows, offsets, edges[:, 0]))
        levels = []
        for level in range(len(groups[0][1]) - 1):
            cell, net, slew_only = tuple(
                rows[int(offsets[level]) : int(offsets[level + 1])]
                for rows, offsets, _ in groups
            )
            if len(cell) or len(net) or len(slew_only):
                arrival_cell_count = len(cell)
                base_cells = np.concatenate(
                    tuple(
                        ids[int(offsets[level]) : int(offsets[level + 1])]
                        for _, offsets, ids in (groups[0], groups[2])
                    )
                )
                cell = np.concatenate((cell, slew_only))
                states = np.unique(np.concatenate((cell[:, :2].ravel(), net.ravel())))
                slew_destinations, slew_index = np.unique(
                    arrays["state_slew_group"][np.concatenate((cell[:, 1], net[:, 1]))],
                    return_inverse=True,
                )
                cell = cell.copy()
                cell[:, :2] = np.searchsorted(states, cell[:, :2])
                net = np.searchsorted(states, net)
                arrival_destinations = np.unique(
                    np.concatenate((cell[:arrival_cell_count, 1], net[:, 1]))
                )
                adjustment = adjustments[
                    int(groups[1][1][level]) : int(groups[1][1][level + 1])
                ]
                levels.append(
                    _SparseTimingLevel(
                        states=index(states),
                        pins=index(arrays["state_pin"][states]),
                        slew_groups=index(arrays["state_slew_group"][states]),
                        cell=index(cell),
                        base_cells=index(base_cells),
                        arrival_cell_count=arrival_cell_count,
                        net=index(net),
                        arrival_destinations=index(arrival_destinations),
                        arrival_states=index(states[arrival_destinations]),
                        slew_index=index(slew_index),
                        slew_destinations=index(slew_destinations),
                        adjustment=torch.as_tensor(
                            adjustment, device=self.device, dtype=self.dtype
                        ),
                    )
                )
        return tuple(levels)

    def propagate(
        self,
        tables,
        caps,
        delays,
        impulses,
        *,
        smooth=False,
        tau_ps=2.0,
        cell_updates=None,
        net_updates=None,
        trace=None,
        arrival_plan=None,
        initial_arrivals=None,
    ):
        """Return state lanes; caps/delays/impulses stay in physical pin domain.

        tables are the same 20 tensors accepted by native cell_aat_forward.
        Autograd gathers sum all state contributions back to physical RC.
        """
        if tau_ps <= 0:
            raise ValueError("sparse timing tau must be positive")
        lanes = tuple(
            torch.where(
                self.seed[:, lane % 2],
                self.reference[:, lane],
                self.reference.new_full((), -2e8 if lane < 2 else 0.0),
            )
            for lane in range(4)
        )
        physical_slew = torch.stack(
            tuple(
                scatter_slew_max(
                    self.reference.new_zeros(self.num_slew_groups),
                    self.state_slew_group,
                    value,
                )
                for value in lanes[2:]
            )
        )
        arrivals = (
            torch.stack(lanes[:2]) if initial_arrivals is None else initial_arrivals
        )
        for level_index, level in enumerate(self.levels):
            cell, net = level.cell, level.net
            arrival_cell_count = level.arrival_cell_count
            # GBA electrical slew is shared across launch/exception states.
            # False paths still participate in electrical propagation, while
            # endpoint legality selects which arrival states can score slack.
            lanes = (
                *arrivals.index_select(1, level.states).unbind(),
                *physical_slew.index_select(1, level.slew_groups).unbind(),
            )
            slew_updates = []
            cell_delays = net_delays = None
            if cell.numel():
                local_caps = tuple(value[level.pins] for value in caps)
                updates = (
                    cell_updates(level, lanes, local_caps)
                    if cell_updates is not None
                    else _CellAat.apply(
                        cell, *lanes, *local_caps, tables, smooth, 1.0 / tau_ps
                    )
                )
                arrival_cell = cell[:arrival_cell_count]
                if trace is not None:
                    cell_delays = updates[:arrival_cell_count, 4:8].detach()
                arrival_updates = updates[:arrival_cell_count]
                if not self.all_transitions_valid:
                    mask = self.valid[level.states[arrival_cell[:, 1]]]
                    arrival_updates = torch.stack(
                        tuple(
                            torch.where(
                                mask[:, lane % 2],
                                arrival_updates[:, lane],
                                updates.new_full((), -2e8 if lane < 2 else 0.0),
                            )
                            for lane in range(4)
                        ),
                        dim=1,
                    )
                if arrival_cell_count:
                    rows = (
                        slice(None)
                        if arrival_plan is None
                        else arrival_plan[level_index][0]
                    )
                    lanes = _TimingLanes.apply(
                        arrival_cell[rows, 1],
                        smooth,
                        tau_ps,
                        *lanes,
                        *(arrival_updates[rows, lane] for lane in range(4)),
                    )
                # Query both arc classes in one native batch; only the first
                # rows create arrivals. The remaining rows influence slew.
                slew_updates.append(
                    tuple(
                        torch.cat(
                            (
                                arrival_updates[:, lane],
                                updates[arrival_cell_count:, lane],
                            )
                        )
                        for lane in (2, 3)
                    )
                )
            if net.numel():
                updates = (
                    net_updates(level, lanes, delays, impulses)
                    if net_updates is not None
                    else _NetAat.apply(
                        net,
                        *lanes,
                        *(value[level.pins] for value in (*delays, *impulses)),
                        1e-8,
                    )
                )
                updates = (
                    updates[0] + updates[2] * level.adjustment[:, 0],
                    updates[1] + updates[3] * level.adjustment[:, 1],
                    updates[2] * level.adjustment[:, 2],
                    updates[3] * level.adjustment[:, 3],
                )
                if not self.all_transitions_valid:
                    mask = self.valid[level.states[net[:, 1]]]
                    updates = tuple(
                        torch.where(
                            mask[:, lane % 2],
                            value,
                            value.new_full((), -2e8 if lane < 2 else 0.0),
                        )
                        for lane, value in enumerate(updates)
                    )
                if trace is not None:
                    net_delays = torch.stack(
                        tuple(
                            (updates[lane] - lanes[lane][net[:, 0]]).detach()
                            for lane in range(2)
                        ),
                        dim=1,
                    )
                rows = (
                    slice(None)
                    if arrival_plan is None
                    else arrival_plan[level_index][1]
                )
                lanes = _TimingLanes.apply(
                    net[rows, 1],
                    smooth,
                    tau_ps,
                    *lanes,
                    *(value[rows] for value in updates),
                )
                slew_updates.append(updates[2:])
            arrivals = arrivals.index_copy(
                1,
                level.arrival_states,
                torch.stack(lanes[:2]).index_select(1, level.arrival_destinations),
            )
            slew_destinations = physical_slew.index_select(1, level.slew_destinations)
            slew_values = torch.stack(
                tuple(
                    scatter_slew_max(
                        value,
                        level.slew_index,
                        torch.cat(tuple(update[lane] for update in slew_updates)),
                    )
                    for lane, value in enumerate(slew_destinations.unbind())
                )
            )
            physical_slew = physical_slew.index_copy(
                1,
                level.slew_destinations,
                slew_values,
            )
            if trace is not None:
                trace.append(SparseTimingTrace(level, cell_delays, net_delays))
        return (
            *arrivals.unbind(),
            *physical_slew.index_select(1, self.state_slew_group).unbind(),
        )

    def reduce_endpoints(self, lanes, check_lut_values):
        """Consume native affine rules and dynamic setup/recovery LUT values.

        check_lut_values is one value per exported candidate, including zeros
        for I/O and false-path rows. It must retain data-slew/master gradients.
        """
        if check_lut_values.shape != self.check_values[:, 0].shape:
            raise ValueError("sparse check LUT values do not align with native rules")
        required = self.check_values[:, 0] + self.check_values[:, 1] * check_lut_values
        arrivals = torch.stack(lanes[:2], dim=1)[
            self.check_state, self.check_transition
        ]
        slack = required[self.legal] - arrivals
        if self.aggregation_mode == "smooth":
            endpoint_ids = self.check_to_endpoint.long()
            maximum = slack.new_full((self.endpoint_pins.numel(),), -torch.inf)
            maximum.scatter_reduce_(
                0,
                endpoint_ids,
                -slack,
                reduce="amax",
                include_self=False,
            )
            shifted = torch.exp(
                (-slack - maximum[endpoint_ids]) / self.aggregation_tau_ps
            )
            total = slack.new_zeros(self.endpoint_pins.numel())
            total.index_add_(0, endpoint_ids, shifted)
            maximum_violation = maximum + self.aggregation_tau_ps * torch.log(total)
        else:
            maximum_violation = scatter_slew_max(
                slack.new_full((self.endpoint_pins.numel(),), -2e8),
                self.check_to_endpoint,
                -slack,
                include_self=True,
            )
        endpoint_slack = -maximum_violation
        zero = lanes[0].sum() * 0
        wns = (
            torch.minimum(endpoint_slack.min(), zero)
            if endpoint_slack.numel()
            else zero
        )
        tns = torch.clamp(endpoint_slack, max=0).sum()
        return SparseTimingResult(
            lanes, required, slack, self.endpoint_pins, endpoint_slack, wns, tns
        )
