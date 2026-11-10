"""Sparse semantic states using the existing native electrical operators.

Each edge batch addresses state IDs, while physical load/RC and Liberty rows
retain their base identity. Only setup builds batches or transfers indices.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .cell_aat_native import _CellAat
from .hard_scatter_native import scatter_slew_max
from .net_aat_native import _NetAat
from .timing_lanes_native import _TimingLanes


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
    net_transition_mask: torch.Tensor
    arrival_destinations: torch.Tensor
    arrival_states: torch.Tensor
    slew_index: torch.Tensor
    slew_destinations: torch.Tensor
    adjustment: torch.Tensor
    cell_arrival_mask: torch.Tensor | None = None
    net_arrival_mask: torch.Tensor | None = None


class _MutableStateWorkspace:
    __slots__ = ("arr_data", "slew_data", "arr_grad", "slew_grad", "token")

    def __init__(self, arrivals: torch.Tensor, physical_slew: torch.Tensor):
        if torch.is_grad_enabled():
            self.arr_data = arrivals.detach().clone()
            self.slew_data = physical_slew.detach().clone()
            self.arr_grad: torch.Tensor | None = None
            self.slew_grad: torch.Tensor | None = None
            self.token: torch.Tensor | None = _StateInit.apply(
                self,
                arrivals,
                physical_slew,
                arrivals.new_zeros((), requires_grad=True),
            )
        else:
            self.arr_data = arrivals.clone()
            self.slew_data = physical_slew.clone()
            self.arr_grad = None
            self.slew_grad = None
            self.token = None

    def read_level(self, states: torch.Tensor, slew_groups: torch.Tensor):
        if self.token is not None:
            ra, fa, rs, fs, self.token = _StateReadLevel.apply(
                self, states, slew_groups, self.token
            )
            return (ra, fa, rs, fs)
        return (
            *self.arr_data.index_select(1, states).unbind(),
            *self.slew_data.index_select(1, slew_groups).unbind(),
        )

    def write_arrivals_and_read_slew(
        self,
        arrival_states: torch.Tensor,
        arrival_destinations: torch.Tensor,
        ra_lane: torch.Tensor,
        fa_lane: torch.Tensor,
        slew_destinations: torch.Tensor,
    ):
        if self.token is not None:
            dest_r, dest_f, self.token = _StateWriteArrAndReadSlew.apply(
                self,
                arrival_states,
                arrival_destinations,
                ra_lane,
                fa_lane,
                slew_destinations,
                self.token,
            )
            return dest_r, dest_f
        self.arr_data[0].index_copy_(
            0, arrival_states, ra_lane.index_select(0, arrival_destinations)
        )
        self.arr_data[1].index_copy_(
            0, arrival_states, fa_lane.index_select(0, arrival_destinations)
        )
        slew_dest = self.slew_data.index_select(1, slew_destinations)
        return slew_dest[0], slew_dest[1]

    def write_slew(
        self,
        slew_destinations: torch.Tensor,
        slew_val_r: torch.Tensor,
        slew_val_f: torch.Tensor,
    ):
        if self.token is not None:
            self.token = _StateWriteSlew.apply(
                self, slew_destinations, slew_val_r, slew_val_f, self.token
            )
        else:
            self.slew_data[0].index_copy_(0, slew_destinations, slew_val_r)
            self.slew_data[1].index_copy_(0, slew_destinations, slew_val_f)

    def finalize(self):
        if self.token is not None:
            return _StateFinalizeAll.apply(self, self.token)
        return self.arr_data, self.slew_data


class _StateInit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, workspace: _MutableStateWorkspace, arrivals: torch.Tensor, physical_slew: torch.Tensor, token: torch.Tensor):
        ctx.workspace = workspace
        return token.clone()

    @staticmethod
    def backward(ctx, grad_token):
        workspace = ctx.workspace
        grad_arr = workspace.arr_grad if ctx.needs_input_grad[1] else None
        grad_slew = workspace.slew_grad if ctx.needs_input_grad[2] else None
        workspace.arr_grad = None
        workspace.slew_grad = None
        return None, grad_arr, grad_slew, None


class _StateReadLevel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, workspace: _MutableStateWorkspace, states: torch.Tensor, slew_groups: torch.Tensor, token: torch.Tensor):
        ctx.workspace = workspace
        ctx.save_for_backward(states, slew_groups)
        arr = workspace.arr_data.index_select(1, states)
        slew = workspace.slew_data.index_select(1, slew_groups)
        return arr[0], arr[1], slew[0], slew[1], token.clone()

    @staticmethod
    def backward(ctx, grad_ra, grad_fa, grad_rs, grad_fs, grad_token):
        workspace = ctx.workspace
        states, slew_groups = ctx.saved_tensors
        arr_grad = workspace.arr_grad
        slew_grad = workspace.slew_grad
        if arr_grad is not None:
            if grad_ra is not None:
                arr_grad[0].index_add_(0, states, grad_ra)
            if grad_fa is not None:
                arr_grad[1].index_add_(0, states, grad_fa)
        if slew_grad is not None:
            if grad_rs is not None:
                slew_grad[0].index_add_(0, slew_groups, grad_rs)
            if grad_fs is not None:
                slew_grad[1].index_add_(0, slew_groups, grad_fs)
        return None, None, None, grad_token


class _StateWriteArrAndReadSlew(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        workspace: _MutableStateWorkspace,
        arrival_states: torch.Tensor,
        arrival_destinations: torch.Tensor,
        ra_lane: torch.Tensor,
        fa_lane: torch.Tensor,
        slew_destinations: torch.Tensor,
        token: torch.Tensor,
    ):
        ctx.workspace = workspace
        ctx.lane_len = int(ra_lane.numel())
        ctx.save_for_backward(arrival_states, arrival_destinations, slew_destinations)
        workspace.arr_data[0].index_copy_(
            0, arrival_states, ra_lane.detach().index_select(0, arrival_destinations)
        )
        workspace.arr_data[1].index_copy_(
            0, arrival_states, fa_lane.detach().index_select(0, arrival_destinations)
        )
        slew_dest = workspace.slew_data.index_select(1, slew_destinations)
        return slew_dest[0], slew_dest[1], token.clone()

    @staticmethod
    def backward(ctx, grad_slew_dest_r, grad_slew_dest_f, grad_token):
        workspace = ctx.workspace
        arrival_states, arrival_destinations, slew_destinations = ctx.saved_tensors
        slew_grad = workspace.slew_grad
        if slew_grad is not None:
            if grad_slew_dest_r is not None:
                slew_grad[0].index_add_(0, slew_destinations, grad_slew_dest_r)
            if grad_slew_dest_f is not None:
                slew_grad[1].index_add_(0, slew_destinations, grad_slew_dest_f)
        arr_grad = workspace.arr_grad
        grad_ra = grad_fa = None
        if arr_grad is not None:
            grad_ra = arr_grad.new_zeros((ctx.lane_len,))
            grad_fa = arr_grad.new_zeros((ctx.lane_len,))
            grad_ra.index_copy_(0, arrival_destinations, arr_grad[0].index_select(0, arrival_states))
            grad_fa.index_copy_(0, arrival_destinations, arr_grad[1].index_select(0, arrival_states))
            arr_grad[0].index_fill_(0, arrival_states, 0.0)
            arr_grad[1].index_fill_(0, arrival_states, 0.0)
        return None, None, None, grad_ra, grad_fa, None, grad_token


class _StateWriteSlew(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        workspace: _MutableStateWorkspace,
        slew_destinations: torch.Tensor,
        slew_val_r: torch.Tensor,
        slew_val_f: torch.Tensor,
        token: torch.Tensor,
    ):
        ctx.workspace = workspace
        ctx.save_for_backward(slew_destinations)
        workspace.slew_data[0].index_copy_(0, slew_destinations, slew_val_r.detach())
        workspace.slew_data[1].index_copy_(0, slew_destinations, slew_val_f.detach())
        return token.clone()

    @staticmethod
    def backward(ctx, grad_token):
        workspace = ctx.workspace
        (slew_destinations,) = ctx.saved_tensors
        slew_grad = workspace.slew_grad
        grad_r = grad_f = None
        if slew_grad is not None:
            grad_r = slew_grad[0].index_select(0, slew_destinations)
            grad_f = slew_grad[1].index_select(0, slew_destinations)
            slew_grad[0].index_fill_(0, slew_destinations, 0.0)
            slew_grad[1].index_fill_(0, slew_destinations, 0.0)
        return None, None, grad_r, grad_f, grad_token


class _StateFinalizeAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, workspace: _MutableStateWorkspace, token: torch.Tensor):
        ctx.workspace = workspace
        return workspace.arr_data.clone(), workspace.slew_data.clone()

    @staticmethod
    def backward(ctx, grad_arr, grad_slew):
        workspace = ctx.workspace
        workspace.arr_grad = (
            grad_arr.clone() if grad_arr is not None else torch.zeros_like(workspace.arr_data)
        )
        workspace.slew_grad = (
            grad_slew.clone() if grad_slew is not None else torch.zeros_like(workspace.slew_data)
        )
        return None, workspace.arr_data.new_zeros(())


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
        self.early = view.early
        self.dtype = dtype
        self.device = torch.device(device)
        self.aggregation_mode = str(aggregation_mode or "hard")
        self.aggregation_tau_ps = float(aggregation_tau_ps)
        if self.aggregation_tau_ps <= 0.0:
            raise ValueError("sparse endpoint aggregation tau must be positive")
        self.state_pin = torch.as_tensor(arrays["state_pin"], device=device)
        self.state_slew_group = torch.as_tensor(arrays["state_slew_group"], device=device)
        self.num_slew_groups = int(arrays["state_slew_group"].max(initial=-1)) + 1
        self.num_pins = len(arrays["pin_state_start"]) - 1
        self.valid = torch.as_tensor(arrays["state_valid"], device=device).bool()
        self.all_transitions_valid = bool(arrays["state_valid"].all())
        self.seed = torch.as_tensor(arrays["state_seed"], device=device).bool()
        self.reference = torch.as_tensor(arrays["state_reference"], device=device, dtype=dtype)
        self.check_index = torch.as_tensor(arrays["check_index"], device=device)
        self.check_values = torch.as_tensor(arrays["check_values"], device=device, dtype=dtype)
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
        self._mask_cache_version = None
        self._cell_arrival_masks = ()
        self._net_arrival_masks = ()

    def _arrival_masks(self):
        version = (self.valid.data_ptr(), self.valid._version)
        if version != self._mask_cache_version:
            self._cell_arrival_masks = tuple(
                self.valid[level.states[level.cell[: level.arrival_cell_count, 1]]]
                & level.cell_arrival_mask
                if level.cell_arrival_mask is not None
                else None
                for level in self.levels
            )
            self._net_arrival_masks = tuple(
                self.valid[level.states[level.net[:, 1]]] & level.net_arrival_mask
                if level.net_arrival_mask is not None
                else None
                for level in self.levels
            )
            self._mask_cache_version = version
        return self._cell_arrival_masks, self._net_arrival_masks

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
            masks = arrays[name.replace("edges", "transition_mask")][order]
            if name != "net_edges":
                rows = base_cell[edges[:, 0]].copy()
                rows[:, :2] = edges[:, 1:]
                rows = np.column_stack((rows, masks))
            else:
                rows = edges[:, 1:].copy()
                adjustments = arrays["net_threshold_adjustments"][order]
            offsets = np.r_[
                0,
                np.cumsum(np.bincount(levels, minlength=int(state_level.max(initial=0)) + 1)),
            ]
            groups.append((rows, offsets, edges[:, 0], masks))
        levels = []
        for level in range(len(groups[0][1]) - 1):
            cell, net, slew_only = tuple(
                rows[int(offsets[level]) : int(offsets[level + 1])]
                for rows, offsets, _, _ in groups
            )
            if len(cell) or len(net) or len(slew_only):
                arrival_cell_count = len(cell)
                base_cells = np.concatenate(
                    tuple(
                        ids[int(offsets[level]) : int(offsets[level + 1])]
                        for _, offsets, ids, _ in (groups[0], groups[2])
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
                adjustment = adjustments[int(groups[1][1][level]) : int(groups[1][1][level + 1])]
                states_tensor = index(states)
                cell_tensor = index(cell)
                cell_tensor._pin_topology_static = True
                net_tensor = index(net)
                net_tensor._pin_topology_static = True
                net_mask_tensor = index(
                    groups[1][3][int(groups[1][1][level]) : int(groups[1][1][level + 1])]
                )
                arrival_cell = cell_tensor[:arrival_cell_count]
                cell_arrival_mask = torch.stack(
                    (arrival_cell[:, 8] & 5 != 0, arrival_cell[:, 8] & 10 != 0), dim=1
                )
                net_arrival_mask = torch.stack(
                    (net_mask_tensor & 1 != 0, net_mask_tensor & 8 != 0), dim=1
                )
                levels.append(
                    _SparseTimingLevel(
                        states=states_tensor,
                        pins=index(arrays["state_pin"][states]),
                        slew_groups=index(arrays["state_slew_group"][states]),
                        cell=cell_tensor,
                        base_cells=index(base_cells),
                        arrival_cell_count=arrival_cell_count,
                        net=net_tensor,
                        net_transition_mask=net_mask_tensor,
                        arrival_destinations=index(arrival_destinations),
                        arrival_states=index(states[arrival_destinations]),
                        slew_index=index(slew_index),
                        slew_destinations=index(slew_destinations),
                        adjustment=torch.as_tensor(
                            adjustment, device=self.device, dtype=self.dtype
                        ),
                        cell_arrival_mask=cell_arrival_mask,
                        net_arrival_mask=net_arrival_mask,
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
        initial_slew=None,
    ):
        """Return state lanes; caps/delays/impulses stay in physical pin domain.

        tables are the same 20 tensors accepted by native cell_aat_forward.
        Autograd gathers sum all state contributions back to physical RC.
        """
        if tau_ps <= 0:
            raise ValueError("sparse timing tau must be positive")
        unreachable = torch.inf if self.early else -2e8
        empty_slew = torch.inf if self.early else 0.0

        def scatter_slew(dest, index, src):
            sign = -1 if self.early else 1
            return sign * scatter_slew_max(sign * dest, index, sign * src)

        def merge(index, *values):
            sign = -1 if self.early else 1
            return tuple(
                sign * value
                for value in _TimingLanes.apply(
                    index, smooth, tau_ps, *(sign * value for value in values)
                )
            )

        lanes = tuple(
            torch.where(
                self.seed[:, lane % 2],
                self.reference[:, lane],
                self.reference.new_full((), unreachable if lane < 2 else empty_slew),
            )
            for lane in range(4)
        )
        physical_slew = torch.stack(
            tuple(
                scatter_slew(
                    self.reference.new_full((self.num_slew_groups,), empty_slew),
                    self.state_slew_group,
                    value,
                )
                for value in lanes[2:]
            )
        )
        if initial_slew is not None:
            physical_slew = (
                torch.minimum(physical_slew, initial_slew)
                if self.early
                else torch.maximum(physical_slew, initial_slew)
            )
        arrivals = torch.stack(lanes[:2]) if initial_arrivals is None else initial_arrivals
        workspace = _MutableStateWorkspace(arrivals, physical_slew)
        cell_masks, net_masks = self._arrival_masks()
        for level_index, level in enumerate(self.levels):
            cell, net = level.cell, level.net
            arrival_cell_count = level.arrival_cell_count
            # GBA electrical slew is shared across launch/exception states.
            # False paths still participate in electrical propagation, while
            # endpoint legality selects which arrival states can score slack.
            lanes = workspace.read_level(level.states, level.slew_groups)
            slew_updates = []
            cell_delays = net_delays = None
            if cell.numel():
                local_caps = tuple(value[level.pins] for value in caps)
                updates = (
                    cell_updates(level, lanes, local_caps)
                    if cell_updates is not None
                    else _CellAat.apply(
                        cell, *lanes, *local_caps, tables, smooth, 1.0 / tau_ps, self.early
                    )
                )
                arrival_cell = cell[:arrival_cell_count]
                if trace is not None:
                    bits = cell.new_tensor([1, 4, 2, 8])
                    cell_delays = torch.where(
                        (cell[:arrival_cell_count, 8, None] & bits) != 0,
                        updates[:arrival_cell_count, 4:8],
                        -torch.inf,
                    ).detach()
                arrival_updates = updates[:arrival_cell_count]
                mask = cell_masks[level_index]
                if mask is None:
                    mask = self.valid[level.states[arrival_cell[:, 1]]]
                    mask = mask & torch.stack(
                        (arrival_cell[:, 8] & 5 != 0, arrival_cell[:, 8] & 10 != 0), dim=1
                    )
                arrival_updates = torch.cat(
                    (
                        torch.where(mask, arrival_updates[:, :2], unreachable),
                        arrival_updates[:, 2:],
                    ),
                    dim=1,
                )
                if arrival_cell_count:
                    rows = slice(None) if arrival_plan is None else arrival_plan[level_index][0]
                    lanes = merge(
                        arrival_cell[rows, 1],
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
                net_lanes = lanes
                if self.early:
                    net_lanes = (
                        *lanes[:2],
                        *(torch.where(torch.isfinite(value), value, 0) for value in lanes[2:]),
                    )
                updates = (
                    net_updates(level, net_lanes, delays, impulses)
                    if net_updates is not None
                    else _NetAat.apply(
                        net,
                        *net_lanes,
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
                if trace is not None:
                    # Store the physical transfer before arrival legality is
                    # applied: multiple tagged edges can share this sink.
                    net_delays = torch.stack(
                        tuple(
                            (updates[lane] - net_lanes[lane][net[:, 0]]).detach()
                            for lane in range(2)
                        ),
                        dim=1,
                    )
                mask = net_masks[level_index]
                if mask is None:
                    mask = self.valid[level.states[net[:, 1]]]
                    mask = mask & torch.stack(
                        (level.net_transition_mask & 1 != 0, level.net_transition_mask & 8 != 0), dim=1
                    )
                updates = (
                    tuple(
                        torch.where(mask[:, lane], value, unreachable)
                        for lane, value in enumerate(updates[:2])
                    )
                    + updates[2:]
                )
                if self.early:
                    updates = updates[:2] + tuple(
                        torch.where(
                            torch.isfinite(lanes[lane + 2][net[:, 0]]), updates[lane + 2], torch.inf
                        )
                        for lane in range(2)
                    )
                rows = slice(None) if arrival_plan is None else arrival_plan[level_index][1]
                lanes = merge(
                    net[rows, 1],
                    *lanes,
                    *(value[rows] for value in updates),
                )
                slew_updates.append(updates[2:])
            slew_dest_r, slew_dest_f = workspace.write_arrivals_and_read_slew(
                level.arrival_states,
                level.arrival_destinations,
                lanes[0],
                lanes[1],
                level.slew_destinations,
            )
            slew_src_r = (
                slew_updates[0][0]
                if len(slew_updates) == 1
                else torch.cat(tuple(update[0] for update in slew_updates))
            )
            slew_src_f = (
                slew_updates[0][1]
                if len(slew_updates) == 1
                else torch.cat(tuple(update[1] for update in slew_updates))
            )
            workspace.write_slew(
                level.slew_destinations,
                scatter_slew(slew_dest_r, level.slew_index, slew_src_r),
                scatter_slew(slew_dest_f, level.slew_index, slew_src_f),
            )
            if trace is not None:
                trace.append(SparseTimingTrace(level, cell_delays, net_delays))
        arrivals, physical_slew = workspace.finalize()
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
        arrivals = torch.stack(lanes[:2], dim=1)[self.check_state, self.check_transition]
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
            shifted = torch.exp((-slack - maximum[endpoint_ids]) / self.aggregation_tau_ps)
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
        wns = torch.minimum(endpoint_slack.min(), zero) if endpoint_slack.numel() else zero
        tns = torch.clamp(endpoint_slack, max=0).sum()
        return SparseTimingResult(
            lanes, required, slack, self.endpoint_pins, endpoint_slack, wns, tns
        )
