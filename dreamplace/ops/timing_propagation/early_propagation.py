"""Context-paired early propagation, using the shared cell LUT and RC contracts."""

from dataclasses import dataclass

import torch

from .endpoint_qualification import endpoint_metrics


@dataclass
class EarlyTimingResult:
    context_slack: torch.Tensor
    endpoint_slack: torch.Tensor
    hold_wns: torch.Tensor
    hold_tns: torch.Tensor
    hold_worst: torch.Tensor
    hold_violations: torch.Tensor
    removal_wns: torch.Tensor
    removal_tns: torch.Tensor
    removal_worst: torch.Tensor
    removal_violations: torch.Tensor
    arrivals: dict
    slews: tuple[torch.Tensor, torch.Tensor]
    state_arrivals: torch.Tensor | None = None

    def deficits(self, margin_ps):
        negative = (self.endpoint_slack - margin_ps).clamp(max=0)
        zero = self.hold_wns * 0
        wns, tns, _ = endpoint_metrics(negative, zero)
        return -wns, -tns


@dataclass
class CellLane:
    source: torch.Tensor
    input_edge: int
    output_edge: int
    delay: torch.Tensor
    slew: torch.Tensor
    rows: torch.Tensor
    arc_sinks: torch.Tensor
    arrival_mask: torch.Tensor | None = None


def _merge(timing, base, pins, values, *, slew=False):
    from .timing_propagation import scatter_timing_min

    return scatter_timing_min(
        base,
        pins,
        values,
        mode="hard" if slew else timing.timing_aggregation_mode,
        tau_ps=timing.timing_aggregation_tau_ps,
    )


def _cell_lanes(timing, arcs, slews, caps, *, clock=False, surrogate=False,
                latch_data_arcs=None, physical_pins=None):
    """Query only legal input/output lanes; slew has independent min semantics."""
    lanes = []
    for output_edge in range(2):
        delay_entry = timing.r_delay_entry if output_edge == 0 else timing.f_delay_entry
        slew_entry = timing.r_tran_entry if output_edge == 0 else timing.f_tran_entry
        for input_edge in range(2):
            same = input_edge == output_edge
            allowed = (arcs[:, 4] == (1 if same else -1)) | (arcs[:, 4] == 0)
            if clock:
                allowed &= (arcs[:, 5] == (1 if input_edge == 0 else -1)) | (arcs[:, 5] == 0)
            rows = arcs[allowed]
            source, sink = rows[:, 0].long(), rows[:, 1].long()
            source_slew = slews[input_edge][source]
            reachable = torch.isfinite(source_slew)
            safe_slew = torch.where(reachable, source_slew, torch.zeros_like(source_slew))
            node_ids = sink if physical_pins is None else physical_pins[sink]
            args = (
                rows[:, 2].long(), safe_slew, caps[output_edge][sink],
                rows[:, 3].long(), node_ids,
            )
            delay = delay_entry(*args, use_surrogate=surrogate)
            slew = slew_entry(*args, use_surrogate=surrogate)
            if latch_data_arcs is not None:
                ids = rows[:, 3].long()
                blocked = (ids >= 0) & latch_data_arcs[ids.clamp(min=0)]
                delay = torch.where(blocked, torch.inf, delay)
            lanes.append(
                CellLane(
                    source,
                    input_edge,
                    output_edge,
                    torch.where(reachable, delay, torch.full_like(delay, torch.inf)),
                    torch.where(reachable, slew, torch.full_like(slew, torch.inf)),
                    torch.nonzero(allowed, as_tuple=False).flatten(),
                    arcs[:, 1].long(),
                    (rows[:, 8] & (1 << (2 * input_edge + output_edge)) != 0)
                    if rows.shape[1] > 8 else None,
                )
            )
    return lanes


def _cell_updates(timing, lanes, arrivals):
    next_arrivals, next_slews = [], []
    for edge in range(2):
        selected = [lane for lane in lanes if lane.output_edge == edge]
        pins = selected[0].arc_sinks
        rows = torch.cat([lane.rows for lane in selected])
        candidates = [arrivals[lane.input_edge][lane.source] + lane.delay for lane in selected]
        values = torch.cat([
            value if lane.arrival_mask is None else torch.where(lane.arrival_mask, value, torch.inf)
            for lane, value in zip(selected, candidates, strict=True)
        ])
        # Match native's hierarchy: reduce the legal lanes of each arc, then
        # reduce arcs at the sink. At hard ties this splits across arcs first.
        empty = arrivals[edge].new_full((pins.numel(),), torch.inf)
        values = _merge(timing, empty, rows, values)
        arc_slew = _merge(
            timing, empty, rows, torch.cat([lane.slew for lane in selected]), slew=True,
        )
        next_arrivals.append(values)
        next_slews.append(arc_slew)
    return tuple(next_arrivals), tuple(next_slews)


def _apply_cells(timing, lanes, arrivals, slews):
    updates, slew_updates = _cell_updates(timing, lanes, arrivals)
    pins = lanes[0].arc_sinks
    return (
        tuple(
            _merge(timing, arrival, pins, value)
            for arrival, value in zip(arrivals, updates, strict=True)
        ),
        tuple(
            _merge(timing, slew, pins, value, slew=True)
            for slew, value in zip(slews, slew_updates, strict=True)
        ),
    )


def _net_arcs(timing, nets):
    starts = timing.net_flat_arcs_start[nets]
    counts = timing.net_flat_arcs_start[nets + 1] - starts
    offsets = counts.cumsum(0) - counts
    indices = torch.repeat_interleave(starts - offsets, counts)
    indices += torch.arange(indices.numel(), device=nets.device)
    return timing.net_flat_arcs[indices]


def _apply_nets(timing, arcs, arrivals, slews, delays, impulses):
    from .timing_propagation import SLEW_SQRT_EPS

    source, sink = arcs[:, 0].long(), arcs[:, 1].long()
    next_arrivals, next_slews = [], []
    for edge in range(2):
        values = arrivals[edge][source] + delays[edge][sink]
        next_arrivals.append(_merge(timing, arrivals[edge], sink, values))
        source_slew = slews[edge][source]
        valid = torch.isfinite(source_slew)
        safe_slew = torch.where(valid, source_slew, torch.zeros_like(source_slew))
        slew = (safe_slew.square() + impulses[edge][sink]).clamp(min=SLEW_SQRT_EPS).sqrt()
        slew = torch.where(valid, slew, torch.full_like(slew, torch.inf))
        next_slews.append(_merge(timing, slews[edge], sink, slew, slew=True))
    return tuple(next_arrivals), tuple(next_slews)


class EarlyPropagation:
    """Keep launch domains separate until their real endpoint/check pairing."""

    def __init__(self, data, *, use_native=False):
        self.data = data
        self.use_native = use_native
        self.native_dispatch = {"cell": 0, "net": 0}
        self.latch_data_arcs = (
            torch.from_numpy(data.lib_arc_is_latch_data.copy())
            if data.lib_arc_is_latch_data.any() else None
        )
        self._check_signature = None
        self._check_model = None
        self._sparse_runtime = None
        self._sparse_signature = None
        self.launch_groups = {}
        for seed in data.launches:
            self.launch_groups.setdefault(seed.launch_key, []).append(seed)
    def _launch(self, timing, key, seeds, caps, surrogate):
        count = timing.num_pins

        def empty():
            return caps[0].new_full((count,), torch.inf)

        arrivals, slews = (empty(), empty()), (empty(), empty())
        source_type = key[3]
        launch_transition = key[2]
        for edge in range(2):
            if source_type == 1:
                selected = [
                    s for s in seeds
                    if s.data_transition == (1 if edge == 0 else -1)
                ]
                pins = torch.tensor([s.start_pin for s in selected], device=timing.device)
                arrivals[edge][pins.long()] = caps[0].new_tensor([s.arrival_ps for s in selected])
                slews[edge][pins.long()] = caps[0].new_tensor([s.slew_ps for s in selected])
            elif source_type == 2 and edge == (0 if launch_transition == 1 else 1):
                pins = torch.tensor([s.launch_pin for s in seeds], device=timing.device)
                arrivals[edge][pins.long()] = caps[0].new_tensor(
                    [s.launch_time_ps for s in seeds]
                )
                slews[edge][pins.long()] = caps[0].new_tensor([s.launch_slew_ps for s in seeds])
        if source_type == 2:
            offsets = timing.flat_inst_arcs_by_level_start
            arcs = timing.flat_inst_arcs_by_level[int(offsets[0]) : int(offsets[1])].clone()
            # Level-0 source ids address clock_pins; later levels use design pins.
            arcs[:, 0] = timing.clock_pins[arcs[:, 0].long()]
            outputs = torch.tensor([s.start_pin for s in seeds], device=timing.device)
            selected = torch.isin(arcs[:, 1], outputs)
            global_rows = torch.arange(int(offsets[0]), int(offsets[1]), device=timing.device)[selected]
            arcs = arcs[selected]
            lanes = _cell_lanes(timing, arcs, slews, caps, clock=True, surrogate=surrogate)
            from .early_actions import attach_cell_probes

            lanes = attach_cell_probes(timing, arcs, lanes, slews, caps, global_rows, clock=True)
            arrivals, slews = _apply_cells(timing, lanes, arrivals, slews)
        return arrivals, slews

    def forward(self, timing, delays, impulses, caps, *, surrogate=False,
                clock_surrogate=False, dynamic_provider=None):
        """Recompute from launch references; never reuse previous Q arrival bounds."""
        if self.data.sparse_view is not None:
            from .early_sparse import SparseEarlyPropagation

            signature = (
                id(timing), timing.device, timing.dtype, self.data.sparse_view.semantic_digest
            )
            if signature != self._sparse_signature:
                self._sparse_runtime = SparseEarlyPropagation(self, timing)
                self._sparse_signature = signature
            return self._sparse_runtime.forward(
                timing, delays, impulses, caps, surrogate=surrogate,
                clock_surrogate=clock_surrogate, dynamic_provider=dynamic_provider,
            )
        if self.latch_data_arcs is not None:
            self.latch_data_arcs = self.latch_data_arcs.to(timing.device)
        if dynamic_provider is not None:
            caps = dynamic_provider.apply_dynamic_net_cap_overlay(
                pin_net_cap_rise=caps[0], pin_net_cap_fall=caps[1],
            )
        domains, seed_slews = {}, []
        for key, seeds in self.launch_groups.items():
            arrivals, slews = self._launch(timing, key, seeds, caps, clock_surrogate)
            domains[key] = arrivals
            seed_slews.append(slews)
        count = timing.num_pins
        slews = tuple(caps[0].new_full((count,), torch.inf) for _ in range(2))
        if seed_slews:
            slews = tuple(torch.stack([s[edge] for s in seed_slews]).amin(0) for edge in range(2))
        # Unconstrained input paths still participate in the STA min-slew model.
        offsets = timing.flat_inst_arcs_by_level_start
        clock_arcs = timing.flat_inst_arcs_by_level[int(offsets[0]) : int(offsets[1])]
        starts = timing.start_points.long()
        input_mask = ~torch.isin(starts, clock_arcs[:, 1])
        for edge in range(2):
            slews = list(slews)
            slews[edge] = slews[edge].clone()
            slews[edge][starts[input_mask]] = caps[0].new_tensor(
                self.data.input_slew[:, edge].copy()
            )[input_mask]
            slews = tuple(slews)

        def nets_forward(nets, level):
            nonlocal slews
            arcs = _net_arcs(timing, nets)
            old_slews = slews
            for key, arrival in domains.items():
                native = None
                if self.use_native:
                    from .early_native import net_early_native

                    native = net_early_native(timing, arcs, arrival, old_slews, delays, impulses)
                if native is None:
                    result = _apply_nets(
                        timing, arcs, arrival, old_slews, delays, impulses
                    )
                else:
                    result = native
                    self.native_dispatch["net"] += 1
                if dynamic_provider is not None:
                    from dreamplace.ops.net_subgraph_timing.early_dynamic_provider import (
                        propagate_early_net_level,
                    )

                    result = propagate_early_net_level(
                        dynamic_provider, nets, arrival, old_slews, delays, impulses, result,
                        level_id=level, epoch=timing._dynamic_net_level_view_epoch,
                    )
                domains[key], slews = result

        nets_forward(torch.unique(timing.pin_net[starts]), 0)
        for level in range(1, len(offsets) - 1):
            arcs = timing.flat_inst_arcs_by_level[int(offsets[level]) : int(offsets[level + 1])]
            if not len(arcs):
                continue
            lanes = None
            old_slews = slews
            for key, arrival in domains.items():
                native = None
                if self.use_native and getattr(timing, "_active_discrete_action_context", None) is None:
                    from .early_native import cell_early_native

                    native = cell_early_native(
                        timing, arcs, arrival, old_slews, caps, surrogate=surrogate,
                        latch_data_arcs=self.latch_data_arcs,
                    )
                if native is None:
                    if lanes is None:
                        lanes = _cell_lanes(
                            timing, arcs, old_slews, caps, surrogate=surrogate,
                            latch_data_arcs=self.latch_data_arcs,
                        )
                    from .early_actions import attach_cell_probes

                    global_rows = torch.arange(int(offsets[level]), int(offsets[level + 1]), device=timing.device)
                    action_lanes = attach_cell_probes(
                        timing, arcs, lanes, old_slews, caps, global_rows,
                        latch_data_arcs=self.latch_data_arcs,
                    )
                    domains[key], slews = _apply_cells(timing, action_lanes, arrival, old_slews)
                else:
                    domains[key], slews = native
                    self.native_dispatch["cell"] += 1
            nets_forward(torch.unique(timing.pin_net[arcs[:, 1].long()]), level)
        return self._evaluate(timing, domains, slews, caps[0], surrogate=surrogate)

    def _prepare_checks(self, timing, reference, *, sparse=False):
        checks = timing.endpoints_timing_check_arcs
        signature = (checks.data_ptr(), tuple(checks.shape), timing.device, sparse)
        if signature == self._check_signature:
            return
        contexts = self.data.contexts
        groups = {key: i for i, key in enumerate(self.launch_groups)}

        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=timing.device)

        self._endpoint_pins = tensor([c.endpoint_pin for c in contexts])
        self._edges = tensor([0 if c.data_transition == 1 else 1 for c in contexts])
        self._domains = tensor([] if sparse else [groups[c.launch_key] for c in contexts])
        self._required_base = reference.new_tensor(
            [c.runtime_required_base_ps for c in contexts]
        )
        self._capture_slew = reference.new_tensor([c.capture_slew_ps for c in contexts])
        self._removal = tensor([c.check_type == 4 for c in contexts]).bool()
        self._has_check = tensor([c.check_type != 0 for c in contexts]).bool()
        rows_by_check = {}
        for row, arc in enumerate(checks.detach().cpu().tolist()):
            rows_by_check.setdefault((arc[0], arc[1], arc[6]), []).append((row, arc[5]))
        context_rows, check_rows = [], []
        for i, context in enumerate(contexts):
            if context.check_type == 0:
                continue
            edge = 0 if context.data_transition == 1 else 1
            identity = context.capture_pin, context.endpoint_pin, context.check_type
            matched = [
                row
                for row, trigger in rows_by_check.get(identity, ())
                if trigger in (0, context.capture_transition) and self.data.check_valid[row, edge]
            ]
            if not matched:
                raise RuntimeError("live hold context lost its qualified library check identity")
            context_rows.extend([i] * len(matched))
            check_rows.extend(matched)
        self._check_contexts = tensor(context_rows)
        self._check_rows = tensor(check_rows)
        self._check_signature = signature

    def _evaluate(self, timing, domains, slews, reference, *, surrogate=False, context_arrival=None):
        self._prepare_checks(timing, reference, sparse=context_arrival is not None)
        if surrogate and self._check_model is None:
            from .hold_check_model import HoldCheckModel

            self._check_model = HoldCheckModel(timing, self.data.lib_arc_check_types)
        count = len(self.data.contexts)
        arrival = reference.new_empty((0,))
        if count:
            if context_arrival is None:
                stacked = torch.stack([torch.stack(domains[key]) for key in self.launch_groups])
                arrival = stacked[self._domains, self._edges, self._endpoint_pins]
            else:
                arrival = context_arrival
            if not bool(torch.isfinite(arrival).all()):
                raise RuntimeError("qualified hold endpoint is unreachable in its launch domain")
        live_check = reference.new_full((count,), torch.inf)
        for edge, lut in enumerate((timing.arcs_info.r_check_luts, timing.arcs_info.f_check_luts)):
            selected = self._edges[self._check_contexts] == edge
            ids = self._check_contexts[selected]
            rows = timing.endpoints_timing_check_arcs[self._check_rows[selected]]
            if not bool(lut.valid_arc_mask[rows[:, 3].long()].all()):
                raise RuntimeError("live min check has no valid LUT for the active master/edge")
            data_slew = slews[edge][self._endpoint_pins[ids]]
            clock_slew = self._capture_slew[ids]
            # The normalized axes are related clock and constrained data for
            # both hold and removal, including negative-edge capture clocks.
            first, second = clock_slew, data_slew
            if surrogate:
                check = self._check_model.query(timing, rows, first, second, edge)
            else:
                check = timing.lut_entry_vectorized(
                    rows[:, 2].long(), first, second, rows[:, 3].long(), lut,
                )
                from .early_actions import attach_check_probes

                check = attach_check_probes(
                    timing, self._check_rows[selected], rows, check, first, second, lut,
                )
            live_check = live_check.scatter_reduce(0, ids, check, reduce="amin")
        # IO rows have a complete constant reference; register checks are live.
        if not bool(torch.isfinite(live_check[self._has_check]).all()):
            raise RuntimeError("live hold check produced unavailable timing values")
        live_check = torch.where(self._has_check, live_check, torch.zeros_like(live_check))
        slack = arrival - self._required_base - live_check
        metrics, endpoint_slacks = [], []
        zero = reference.sum() * 0
        for removal in (False, True):
            selected = self._removal == removal
            pins = self._endpoint_pins[selected]
            per_pin = _merge(
                timing, reference.new_full((timing.num_pins,), torch.inf), pins, slack[selected]
            )
            per_endpoint = per_pin[timing.end_points.long()]
            wns, tns, worst = endpoint_metrics(per_endpoint, zero)
            metrics.extend((wns, tns, worst, (per_endpoint < 0).sum()))
            endpoint_slacks.append(per_endpoint)
        return EarlyTimingResult(slack, endpoint_slacks[0], *metrics, domains, slews)
