"""Adapt sparse identities to the existing native path/pruning graph helpers."""

from dataclasses import dataclass
import time

import numpy as np
import torch

from .sparse_timing import SparseTimingResult, SparseTimingTrace
from .recovery_checks import TIMING_CHECK_CLASS_RECOVERY


@dataclass(frozen=True)
class SparseTimingSnapshot:
    result: SparseTimingResult
    trace: tuple[SparseTimingTrace, ...]
    topology_epoch: int
    master_version: tuple[int, int]
    semantic_digest: str


@dataclass(frozen=True)
class SparseTraversalResult:
    kept_flat_arc_indices: torch.Tensor
    kept_level_offsets: torch.Tensor
    kept_counts_by_level: torch.Tensor
    active_inst_count: int
    active_arc_count: int
    dropped_arc_count: int
    parallel_task_count: int
    preparation_runtime_ms: float
    kept_state_indices: torch.Tensor


@dataclass(frozen=True)
class SparseEndpointIncidence:
    pin_endpoint_incidence_count: torch.Tensor
    arc_endpoint_incidence_count: torch.Tensor
    active_endpoint_count: int
    pin_incidence_nonzero_count: int
    arc_incidence_nonzero_count: int
    incidence_runtime_ms: float


@dataclass(frozen=True)
class SparsePathBatch:
    path_offsets: torch.Tensor
    path_pins: torch.Tensor
    path_transitions: torch.Tensor
    path_arc_ids: torch.Tensor
    endpoint_pins: torch.Tensor
    endpoint_test_ids: torch.Tensor
    endpoint_transitions: torch.Tensor
    endpoint_slacks: torch.Tensor
    path_valid: torch.Tensor
    invalid_reason: torch.Tensor
    max_residual_ps: torch.Tensor
    failing_state_count: int
    selected_state_count: int
    valid_path_count: int
    invalid_path_count: int
    topology_epoch: int
    extraction_runtime_ms: float
    path_state_ids: torch.Tensor
    endpoint_state_ids: torch.Tensor
    endpoint_check_rows: torch.Tensor
    path_arrivals: torch.Tensor
    path_delays: torch.Tensor


class SparseTopology:
    def __init__(self, timing):
        from .timing_propagation import _tp_cpp

        if _tp_cpp is None:
            raise RuntimeError(
                "native timing graph helpers are required for sparse paths/pruning"
            )
        self.timing = timing
        arrays = timing.sparse_timing_view.arrays
        self.state_pin = torch.from_numpy(arrays["state_pin"])
        checks = arrays["check_index"]
        legal = timing._sparse_runtime.schedule.legal.cpu().numpy()
        self.legal_states = torch.from_numpy(np.unique(checks[legal, 0]))
        states = len(arrays["state_pin"])
        cells = arrays["cell_edges"]
        order = np.argsort(arrays["state_level"][cells[:, 2]], kind="stable")
        cells = cells[order]
        self.base_cells = torch.from_numpy(cells[:, 0].copy())
        base = timing.flat_inst_arcs_by_level.detach().cpu().long()
        arcs = base[self.base_cells].clone()
        arcs[:, :2] = torch.from_numpy(cells[:, 1:].copy())
        nets = arrays["net_edges"]
        source = np.concatenate((cells[:, 1], nets[:, 1]))
        sink = np.concatenate((cells[:, 2], nets[:, 2]))
        arc_id = np.r_[np.arange(len(cells)), np.full(len(nets), -1)]
        order = np.argsort(sink, kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(sink, minlength=states))]
        seeds = np.flatnonzero(arrays["state_seed"].any(axis=1))
        self.predecessors = tuple(
            torch.from_numpy(value.copy())
            for value in (
                offsets,
                source[order],
                arc_id[order],
                seeds,
            )
        )
        self.extractor = _tp_cpp.SetupCriticalPathExtractor(
            arcs,
            *self.predecessors,
            timing.critical_path_topology_epoch,
        )
        levels = arrays["state_level"][cells[:, 2]]
        level_offsets = np.r_[
            0, np.cumsum(np.bincount(levels, minlength=int(levels.max(initial=0)) + 1))
        ]
        node_ids = (
            timing.pin2node_map.detach().cpu()[self.state_pin]
            if timing.pin2node_map is not None
            else self.state_pin
        )
        self.pruner = _tp_cpp.CriticalEndpointTraversalPruner(
            arcs,
            torch.from_numpy(level_offsets),
            *self.predecessors,
            node_ids,
        )

    def _endpoint_states(self, endpoints):
        return self.legal_states[
            torch.isin(self.state_pin[self.legal_states], endpoints)
        ]

    def refresh(self, endpoints):
        native = self.pruner.refresh(self._endpoint_states(endpoints))
        kept = torch.unique(self.base_cells[native.kept_flat_arc_indices])
        timing = self.timing
        offsets = timing.flat_inst_arcs_by_level_start.detach().cpu().long()
        counts = torch.searchsorted(kept, offsets)
        return SparseTraversalResult(
            kept,
            counts,
            counts.diff(),
            native.active_inst_count,
            len(kept),
            len(timing.flat_inst_arcs_by_level) - len(kept),
            native.parallel_task_count,
            native.preparation_runtime_ms,
            native.kept_pin_indices,
        )

    def endpoint_incidence(self, endpoints):
        # Count a physical endpoint once even if several launch states reach it.
        # Reuse the native cone traversal; only identity reduction belongs here.
        started = time.perf_counter()
        pins = torch.zeros(self.timing.num_pins, dtype=torch.long)
        arcs = torch.zeros(len(self.timing.flat_inst_arcs_by_level), dtype=torch.long)
        endpoints = torch.unique(endpoints.long())
        for endpoint in endpoints:
            cone = self.pruner.refresh(self._endpoint_states(endpoint.reshape(1)))
            pins[torch.unique(self.state_pin[cone.kept_pin_indices])] += 1
            arcs[torch.unique(self.base_cells[cone.kept_flat_arc_indices])] += 1
        return SparseEndpointIncidence(
            pins,
            arcs,
            len(endpoints),
            int(torch.count_nonzero(pins)),
            int(torch.count_nonzero(arcs)),
            (time.perf_counter() - started) * 1000,
        )

    def extract(
        self, snapshot, *, global_k, max_depth, residual_tolerance_ps, include_recovery
    ):
        timing = self.timing
        if snapshot.topology_epoch != timing.critical_path_topology_epoch:
            raise RuntimeError("sparse path snapshot topology epoch is stale")
        if snapshot.semantic_digest != timing.sparse_timing_view.semantic_digest:
            raise RuntimeError("sparse path snapshot timing view is stale")
        base = timing.flat_inst_arcs_by_level
        if snapshot.master_version != (base.data_ptr(), base._version):
            raise RuntimeError("sparse path snapshot cell masters are stale")
        result = snapshot.result
        model = timing._sparse_runtime.schedule
        rows = torch.nonzero(model.legal).flatten()
        checks = model.check_index[rows]
        if not include_recovery:
            dynamic = checks[:, 2] >= 0
            recovery = torch.zeros(len(rows), dtype=torch.bool, device=rows.device)
            recovery[dynamic] = (
                timing.endpoints_timing_check_arcs[checks[dynamic, 2], 6]
                == TIMING_CHECK_CLASS_RECOVERY
            )
            rows, checks = rows[~recovery], checks[~recovery]
        slack = result.required.new_full((len(rows), 2), torch.inf)
        arrival = torch.stack(result.lanes[:2], dim=1)[checks[:, 0], checks[:, 3]]
        slack[torch.arange(len(rows), device=rows.device), checks[:, 3]] = (
            result.required[rows] - arrival
        )
        cell_delays = (
            torch.cat(
                tuple(
                    entry.cell_delays
                    for entry in snapshot.trace
                    if entry.cell_delays is not None
                ),
                dim=0,
            )
            if self.base_cells.numel()
            else result.required.new_empty((0, 4))
        )
        net_delays = result.required.new_zeros((len(self.state_pin), 2))
        for entry in snapshot.trace:
            if entry.net_delays is not None:
                net_delays[entry.level.states[entry.level.net[:, 1]]] = entry.net_delays

        def cpu(value):
            return value.detach().cpu().contiguous()

        native = self.extractor.extract(
            cpu(checks[:, 0]),
            cpu(rows),
            cpu(slack[:, 0]),
            cpu(slack[:, 1]),
            *(cpu(value) for value in result.lanes[:2]),
            cpu(net_delays[:, 0]),
            cpu(net_delays[:, 1]),
            *(cpu(cell_delays[:, lane]) for lane in range(4)),
            int(global_k),
            int(max_depth),
            float(residual_tolerance_ps),
        )
        arc_ids = native.path_arc_ids.clone().long()
        cell = arc_ids >= 0
        destination = torch.ones(len(native.path_pins), dtype=torch.bool)
        destination[native.path_offsets[:-1].long()] = False
        destination_positions = torch.nonzero(destination).flatten()
        source_edges = native.path_transitions[destination_positions - 1].long()
        sink_edges = native.path_transitions[destination_positions].long()
        sink_states = native.path_pins[destination_positions].long()
        path_delays = result.required.new_empty(len(arc_ids), device="cpu")
        path_delays[~cell] = cpu(net_delays)[sink_states[~cell], sink_edges[~cell]]
        path_delays[cell] = cpu(cell_delays)[
            arc_ids[cell], sink_edges[cell] * 2 + source_edges[cell]
        ]
        path_arrivals = cpu(torch.stack(result.lanes[:2], dim=1))[
            native.path_pins.long(), native.path_transitions.long()
        ]
        arc_ids[cell] = self.base_cells[arc_ids[cell]]
        return SparsePathBatch(
            path_offsets=native.path_offsets,
            path_pins=self.state_pin[native.path_pins.long()],
            path_transitions=native.path_transitions,
            path_arc_ids=arc_ids,
            endpoint_pins=self.state_pin[native.endpoint_pins.long()],
            endpoint_test_ids=model.check_index[
                native.endpoint_test_ids.to(model.device), 2
            ].cpu(),
            endpoint_transitions=native.endpoint_transitions,
            endpoint_slacks=native.endpoint_slacks,
            path_valid=native.path_valid,
            invalid_reason=native.invalid_reason,
            max_residual_ps=native.max_residual_ps,
            failing_state_count=native.failing_state_count,
            selected_state_count=native.selected_state_count,
            valid_path_count=native.valid_path_count,
            invalid_path_count=native.invalid_path_count,
            topology_epoch=native.topology_epoch,
            extraction_runtime_ms=native.extraction_runtime_ms,
            path_state_ids=native.path_pins,
            endpoint_state_ids=native.endpoint_pins,
            endpoint_check_rows=native.endpoint_test_ids,
            path_arrivals=path_arrivals,
            path_delays=path_delays,
        )
