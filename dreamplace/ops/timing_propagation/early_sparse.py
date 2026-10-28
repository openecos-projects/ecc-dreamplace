"""Min propagation over native launch/exception states and legal transitions."""

import numpy as np
import torch

from .cell_aat_native import _CellAat, _tables
from .early_actions import attach_cell_probes
from .early_propagation import _cell_lanes, _cell_updates
from .hard_scatter_native import scatter_slew_max
from .net_aat_native import _NetAat
from .sparse_timing import SparseTimingSchedule


class SparseEarlyPropagation:
    def __init__(self, owner, timing):
        self.owner = owner
        self.view = owner.data.sparse_view
        base_cells = timing.flat_inst_arcs_by_level.detach().cpu().numpy()
        self.schedule = SparseTimingSchedule(
            self.view,
            base_cells,
            device=timing.device,
            dtype=timing.dtype,
        )
        arrays = self.view.arrays
        self.context_states = torch.as_tensor(arrays["context_state"], device=timing.device)
        self.context_edges = torch.tensor(
            [0 if context.data_transition == 1 else 1 for context in owner.data.contexts],
            device=timing.device,
            dtype=torch.long,
        )
        self.receiver_states = torch.as_tensor(
            np.flatnonzero(arrays["state_role"] == 0), device=timing.device
        )
        clock_end = int(timing.flat_inst_arcs_by_level_start[1])
        # Placement pins and electrical slew groups have different numbering.
        # Seed only the matching boundary groups, including their lane masks.
        starts = timing.start_points.detach().cpu().numpy()
        input_rows = np.full(self.schedule.num_pins, -1, dtype=np.int64)
        inputs = ~np.isin(starts, base_cells[:clock_end, 1])
        input_rows[starts[inputs]] = np.flatnonzero(inputs)
        state_rows = input_rows[arrays["state_pin"]]
        self.input_slew_indices = []
        for edge in range(2):
            seeded = (state_rows >= 0) & arrays["state_seed"][:, edge].astype(bool)
            pairs = np.unique(
                np.column_stack((arrays["state_slew_group"][seeded], state_rows[seeded])),
                axis=0,
            )
            self.input_slew_indices.append(
                tuple(torch.as_tensor(value, device=timing.device) for value in pairs.T)
            )
        self.clock_rows = {
            id(level): torch.nonzero(level.base_cells < clock_end).flatten()
            for level in self.schedule.levels
        }
        self.level_ids = {id(level): index for index, level in enumerate(self.schedule.levels)}

    def forward(
        self, timing, delays, impulses, caps, *, surrogate, clock_surrogate, dynamic_provider
    ):
        owner, model = self.owner, self.schedule
        tables = _tables(timing)
        if dynamic_provider is not None:
            caps = dynamic_provider.apply_dynamic_net_cap_overlay(
                pin_net_cap_rise=caps[0],
                pin_net_cap_fall=caps[1],
            )
        if owner.latch_data_arcs is not None:
            owner.latch_data_arcs = owner.latch_data_arcs.to(timing.device)

        def cells(level, lanes, local_caps):
            arcs = level.cell.clone()
            arcs[:, 2:8] = timing.flat_inst_arcs_by_level[level.base_cells, 2:8]

            def query(rows, use_surrogate):
                selected = arcs if rows is None else arcs[rows]
                global_rows = level.base_cells if rows is None else level.base_cells[rows]
                action = getattr(timing, "_active_discrete_action_context", None)
                if owner.use_native and not use_surrogate and action is None and tables is not None:
                    updates = _CellAat.apply(
                        selected,
                        *lanes,
                        *local_caps,
                        tables,
                        timing.timing_aggregation_mode == "smooth",
                        1.0 / timing.timing_aggregation_tau_ps,
                        model.early,
                    )[:, :4]
                    if owner.latch_data_arcs is not None:
                        blocked = owner.latch_data_arcs[selected[:, 3]]
                        updates = torch.cat(
                            (
                                torch.where(blocked[:, None], torch.inf, updates[:, :2]),
                                updates[:, 2:],
                            ),
                            dim=1,
                        )
                    owner.native_dispatch["cell"] += 1
                    return updates
                branches = _cell_lanes(
                    timing,
                    selected,
                    lanes[2:],
                    local_caps,
                    clock=True,
                    surrogate=use_surrogate,
                    latch_data_arcs=owner.latch_data_arcs,
                    physical_pins=level.pins,
                )
                branches = attach_cell_probes(
                    timing,
                    selected,
                    branches,
                    lanes[2:],
                    local_caps,
                    global_rows,
                    clock=True,
                    latch_data_arcs=owner.latch_data_arcs,
                    physical_pins=level.pins,
                )
                arrivals, slews = _cell_updates(timing, branches, lanes[:2])
                return torch.stack((*arrivals, *slews), dim=1)

            updates = query(None, surrogate)
            if clock_surrogate != surrogate and self.clock_rows[id(level)].numel():
                rows = self.clock_rows[id(level)]
                updates = updates.index_copy(0, rows, query(rows, clock_surrogate))
            return updates

        def nets(level, lanes, net_delays, net_impulses):
            if dynamic_provider is not None:
                ids = torch.unique(timing.pin_net[level.pins[level.net[:, 1]]])
                if dynamic_provider.has_dynamic_nets(
                    ids,
                    level_id=self.level_ids[id(level)],
                    level_view_epoch=timing.critical_path_topology_epoch,
                ):
                    raise RuntimeError(
                        "min sparse propagation requires a min-capable dynamic net transfer"
                    )
            if owner.use_native:
                owner.native_dispatch["net"] += 1
                return _NetAat.apply(
                    level.net,
                    *lanes,
                    *(value[level.pins] for value in (*net_delays, *net_impulses)),
                    1e-8,
                )
            source = level.net[:, 0]
            sinks = level.pins[level.net[:, 1]]
            return (
                *(lanes[edge][source] + net_delays[edge][sinks] for edge in range(2)),
                *(
                    (lanes[edge + 2][source].square() + net_impulses[edge][sinks])
                    .clamp(min=1e-8)
                    .sqrt()
                    for edge in range(2)
                ),
            )

        initial_slew = caps[0].new_full((2, model.num_slew_groups), torch.inf)
        for edge, (groups, rows) in enumerate(self.input_slew_indices):
            initial_slew[edge, groups] = caps[0].new_tensor(
                owner.data.input_slew[:, edge].copy()
            )[rows]
        lanes = model.propagate(
            (),
            caps,
            delays,
            impulses,
            smooth=timing.timing_aggregation_mode == "smooth",
            tau_ps=timing.timing_aggregation_tau_ps,
            cell_updates=cells,
            net_updates=nets,
            initial_slew=initial_slew,
        )
        arrival = torch.stack(lanes[:2], dim=1)[self.context_states, self.context_edges]
        receivers = self.receiver_states
        slews = tuple(
            -scatter_slew_max(
                caps[0].new_full((timing.num_pins,), -torch.inf),
                model.state_pin[receivers],
                -lane[receivers],
            )
            for lane in lanes[2:]
        )
        result = owner._evaluate(
            timing, {}, slews, caps[0], surrogate=surrogate, context_arrival=arrival
        )
        result.state_arrivals = torch.stack(lanes[:2], dim=1)
        return result
