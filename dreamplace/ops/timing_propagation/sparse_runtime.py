"""Sparse state execution bound to the production TimingPropagation owner.

This adapter owns state layout and reporting projections. Electrical queries,
size/VT models, RC providers and optimization policy belong to their existing
owners; no optimizer or separate cell model is constructed here.
"""

from dataclasses import replace

import numpy as np
import torch

from .cell_aat_native import _CellAat, _tables
from .cell_aat_queries import query_cell_aat_updates
from .endpoint_qualification import endpoint_metrics
from .sparse_timing import SparseTimingSchedule
from .sparse_net_provider import SparseNetProvider
from .sparse_topology import SparseTimingSnapshot, SparseTopology
from .recovery_checks import TIMING_CHECK_CLASS_RECOVERY
from .sparse_reporting import (
    physical_projection,
    physical_slew,
    physical_cell_delays,
    propagate_required,
)


class SparseTimingRuntime:
    def __init__(self, timing):
        self.timing = timing
        self.topology_epoch = timing.critical_path_topology_epoch
        view = timing.sparse_timing_view
        self.view = view
        self.statistics = view.statistics()
        checks = view.arrays["check_index"]
        check_mask = np.ones(len(checks), dtype=bool)
        if timing.timing_metric_scope == "setup_only":
            arcs = timing.endpoints_timing_check_arcs.detach().cpu().numpy()
            dynamic = checks[:, 2] >= 0
            check_mask[dynamic] = (
                arcs[checks[dynamic, 2], 6] != TIMING_CHECK_CLASS_RECOVERY
            )
        base = timing.flat_inst_arcs_by_level.detach().cpu().numpy()
        self.schedule = SparseTimingSchedule(
            view, base, device=timing.device, dtype=timing.dtype, check_mask=check_mask
        )
        self._dynamic_checks = torch.nonzero(
            self.schedule.legal & (self.schedule.check_index[:, 2] >= 0)
        ).flatten()
        timing.last_traversal_pruning_result = None
        timing.last_traversal_pruning_iteration = None
        timing.critical_endpoint_selector = None
        self.endpoint_validity = torch.zeros(
            (timing.num_pins, 2), dtype=torch.bool, device=timing.device
        )
        legal = self.schedule.check_index[self.schedule.legal]
        self.endpoint_validity[self.schedule.state_pin[legal[:, 0]], legal[:, 3]] = True
        self._master_version = None
        self._topology = None
        self._pruning_result = None
        self._arrival_plan = None
        self._cell_arcs = {}
        clock_end = int(timing.flat_inst_arcs_by_level_start[1])
        self._clock_rows = {
            id(level): torch.nonzero(level.base_cells < clock_end).flatten()
            for level in self.schedule.levels
        }
        self.last_result = None
        self.net_provider = SparseNetProvider(timing, self.schedule)

    @property
    def topology(self):
        if (
            self._topology is None
            or self._topology_epoch != self.timing.critical_path_topology_epoch
        ):
            self._topology = SparseTopology(self.timing)
            self._topology_epoch = self.timing.critical_path_topology_epoch
        return self._topology

    def extract_paths(self, snapshot, **options):
        batch = self.topology.extract(snapshot, **options)
        self.timing.last_critical_path_extraction_stats = {
            "backend": "cpp_openmp_sparse_transition_aware",
            "topology_epoch": batch.topology_epoch,
            "failing_state_count": batch.failing_state_count,
            "selected_state_count": batch.selected_state_count,
            "valid_path_count": batch.valid_path_count,
            "invalid_path_count": batch.invalid_path_count,
            "max_residual_ps": float(batch.max_residual_ps.max())
            if batch.max_residual_ps.numel()
            else 0,
            "extraction_runtime_ms": batch.extraction_runtime_ms,
            "state_domain": "setup_plus_recovery"
            if options["include_recovery"]
            else "setup_only",
        }
        return batch

    def _arrival_selection(self):
        timing, model = self.timing, self.schedule
        if (
            not timing._traversal_pruning_enabled()
            or self.last_result is None
            or self._need_delays
        ):
            return None, None
        selector = timing._ensure_critical_endpoint_selector()
        full_interval = timing.critical_endpoint_full_refresh_interval
        iteration = timing.timing_forward_count
        if selector.should_refresh(iteration) or (
            full_interval > 0 and iteration % full_interval == 0
        ):
            return None, None
        result = timing.last_traversal_pruning_result
        if result is None:
            return None, None
        if result is not self._pruning_result:
            active = torch.zeros(
                len(model.state_pin), device=model.device, dtype=torch.bool
            )
            active[result.kept_state_indices.to(model.device)] = True
            self._arrival_plan = tuple(
                (
                    torch.nonzero(
                        active[level.states[level.cell[: level.arrival_cell_count, 1]]]
                    ).flatten(),
                    torch.nonzero(active[level.states[level.net[:, 1]]]).flatten(),
                )
                for level in model.levels
            )
            self._active_states = active
            self._pruning_result = result
        previous = torch.stack(
            tuple(value.detach() for value in self.last_result.lanes[:2])
        )
        initial = torch.where(self._active_states.unsqueeze(0), -2e8, previous)
        seeds = torch.where(model.seed.T, model.reference[:, :2].T, initial)
        return self._arrival_plan, seeds

    def _bind_masters(self):
        base = self.timing.flat_inst_arcs_by_level
        version = (base.data_ptr(), base._version)
        if version == self._master_version:
            return
        self._cell_arcs = {
            id(level): torch.cat((level.cell[:, :2], base[level.base_cells, 2:]), dim=1)
            for level in self.schedule.levels
        }
        self._master_version = version
        self._topology = None

    def _cell_updates(self, level, lanes, caps):
        from .timing_propagation import (
            SMOOTH_MAX_ALPHA,
            SMOOTH_MAX_ENABLED,
            _force_hard_pairwise_max_enabled,
        )

        timing = self.timing
        arcs = self._cell_arcs[id(level)]
        clk2q_surrogate, cell_surrogate = self._surrogate_flags

        def query(rows, use_surrogate):
            selected = arcs if rows is None else arcs[rows]
            if not use_surrogate and self._tables is not None:
                return _CellAat.apply(
                    selected,
                    *lanes,
                    *caps,
                    self._tables,
                    SMOOTH_MAX_ENABLED and not _force_hard_pairwise_max_enabled(),
                    SMOOTH_MAX_ALPHA,
                )
            result = query_cell_aat_updates(
                timing,
                selected,
                *lanes,
                *caps,
                use_surrogate=use_surrogate,
                need_delay_updates=self._need_delays,
                physical_pins=level.pins,
            )
            columns = result.lanes
            if self._need_delays:
                columns += result.delays
            return torch.stack(columns, dim=1)

        if clk2q_surrogate == cell_surrogate:
            return query(None, cell_surrogate)
        clocks = self._clock_rows[id(level)]
        if not clocks.numel():
            return query(None, cell_surrogate)
        if clocks.numel() == arcs.size(0):
            return query(None, clk2q_surrogate)
        result = query(None, cell_surrogate)
        return result.index_copy(0, clocks, query(clocks, clk2q_surrogate))

    def _check_values(self, lanes):
        timing, model = self.timing, self.schedule
        dynamic = self._dynamic_checks
        checks = model.check_values.new_zeros(model.check_values.size(0))
        if not dynamic.numel():
            return checks
        indices = model.check_index[dynamic]
        arcs = timing.endpoints_timing_check_arcs[indices[:, 2]]
        slew = torch.stack(lanes[2:], dim=1)[indices[:, 0], indices[:, 3]]
        clock_slew = model.check_values[dynamic, 2]
        values = tuple(
            timing.lut_entry_vectorized(arcs[:, 2], clock_slew, slew, arcs[:, 3], info)
            for info in (timing.arcs_info.r_check_luts, timing.arcs_info.f_check_luts)
        )
        checks[dynamic] = torch.where(indices[:, 3] == 0, *values)
        return checks

    def forward(
        self,
        delays,
        impulses,
        caps,
        *,
        surrogate_flags,
        dynamic_net_provider,
        surrogate_mode,
    ):
        timing, model = self.timing, self.schedule
        started = timing._sync_profile_clock()
        self._bind_masters()
        self._surrogate_flags = surrogate_flags
        self._tables = _tables(timing)
        snapshot_requested = bool(timing._critical_path_snapshot_requested)
        self._need_delays = (
            snapshot_requested or not timing._skip_full_rat_propagation_in_fast_loop()
        )
        if snapshot_requested and timing._critical_path_snapshot is not None:
            raise RuntimeError("critical path snapshot was captured more than once")
        trace = [] if self._need_delays else None
        arrival_plan, initial_arrivals = self._arrival_selection()
        self.net_provider.bind(dynamic_net_provider)
        if dynamic_net_provider is not None:
            dynamic_net_provider.begin_critical_path_net_delay_snapshot(
                base_pin_net_delay_rise=delays[0],
                base_pin_net_delay_fall=delays[1],
            )
        lanes = model.propagate(
            (),
            caps,
            delays,
            impulses,
            smooth=timing.timing_aggregation_mode == "smooth",
            tau_ps=timing.timing_aggregation_tau_ps,
            cell_updates=self._cell_updates,
            net_updates=self.net_provider,
            trace=trace,
            arrival_plan=arrival_plan,
            initial_arrivals=initial_arrivals,
        )
        timing.virtual_buffer_drv = (
            None
            if dynamic_net_provider is None
            else dynamic_net_provider.virtual_buffer_drv_tensors()
        )
        if dynamic_net_provider is not None:
            delays = dynamic_net_provider.consume_critical_path_net_delays(
                base_pin_net_delay_rise=delays[0],
                base_pin_net_delay_fall=delays[1],
            )
        result = model.reduce_endpoints(lanes, self._check_values(lanes))
        self.last_result = result
        if snapshot_requested:
            detached = replace(
                result,
                lanes=tuple(value.detach() for value in result.lanes),
                required=result.required.detach(),
                check_slack=result.check_slack.detach(),
                endpoint_slack=result.endpoint_slack.detach(),
                wns=result.wns.detach(),
                tns=result.tns.detach(),
            )
            timing._critical_path_snapshot = SparseTimingSnapshot(
                detached,
                tuple(trace),
                timing.critical_path_topology_epoch,
                self._master_version,
                self.view.semantic_digest,
            )
        full_rat = not timing._skip_full_rat_propagation_in_fast_loop()
        state_required = (
            propagate_required(timing, model, result, trace) if full_rat else None
        )
        arrivals, required, slack = physical_projection(
            timing, model, result, state_required
        )
        slews = physical_slew(model, result.lanes, timing.num_pins)
        cell_delays = physical_cell_delays(timing, trace) if full_rat else (None,) * 4
        timing.last_endpoint_slack_tensor = result.endpoint_slack
        timing.last_endpoint_ids_tensor = result.endpoint_pins.detach()
        timing.last_fast_loop_skipped_full_rat = not full_rat
        timing.update_critical_endpoint_pruning_state(
            iteration=timing.timing_forward_count
        )
        timing._refresh_traversal_pruning_state(iteration=timing.timing_forward_count)
        timing.timing_forward_count += 1
        timing._store_forward_state(
            pin_rAAT=arrivals[0],
            pin_fAAT=arrivals[1],
            pin_rRAT=required[0],
            pin_fRAT=required[1],
            pin_rtran=slews[0],
            pin_ftran=slews[1],
            pin_net_cap_rise=caps[0],
            pin_net_cap_fall=caps[1],
            pin_net_delay_rise=delays[0],
            pin_net_delay_fall=delays[1],
            pin_net_impulse_rise=impulses[0],
            pin_net_impulse_fall=impulses[1],
            rslack=slack[:, 0],
            fslack=slack[:, 1],
            slack=slack.min(dim=1).values,
            cell_arc_rr_delays=cell_delays[0],
            cell_arc_fr_delays=cell_delays[1],
            cell_arc_rf_delays=cell_delays[2],
            cell_arc_ff_delays=cell_delays[3],
            capture_critical_path=False,
        )
        contract = timing._build_device_contract(
            {
                "state_rise_aat": lanes[0],
                "state_fall_aat": lanes[1],
                "state_rise_slew": lanes[2],
                "state_fall_slew": lanes[3],
            },
            scope="sparse_forward",
        )
        timing._assert_device_contract(contract)
        timing.last_profile_payload = timing._build_profile_payload(
            total_runtime_ms=(timing._sync_profile_clock() - started) * 1000,
            stage_runtime_ms={},
            surrogate_mode=surrogate_mode,
            device_contract=contract,
            dynamic_provider_metadata=None
            if dynamic_net_provider is None
            else dynamic_net_provider.metadata,
        )
        timing.last_profile_payload.update(
            {
                "mode": "sparse",
                "full_arrival_refresh": arrival_plan is None,
                "electrical_slew_scope": "full_physical_graph",
                **self.statistics,
            }
        )
        wns, tns, worst = endpoint_metrics(result.endpoint_slack, caps[0].sum() * 0)
        timing._end_forward_runtime_state()
        return wns, tns, worst, 0
