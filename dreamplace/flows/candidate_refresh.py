"""Shared candidate-state refresh orchestration for placement and diff-opt."""

import time

import torch

from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
    collect_candidate_dirty_nets,
)


def candidate_topology_geometry_key(*, data, lane, ops, pos):
    """Identify the topology state whose candidate coordinates must stay live."""

    topology = getattr(ops, "steiner_topo_op", None)
    frozen_nets = getattr(topology, "frozen_nets", None)
    state = None if lane is None else lane._current_state()
    candidate_live_geometry = bool(state is not None and hasattr(state, "activation_param"))
    return _tensor_digest(
        (
            (
                "runtime_cell_state_generation",
                None
                if candidate_live_geometry
                else int(getattr(data, "runtime_cell_state_generation", 0) or 0),
            ),
            (
                "topology_generation",
                int(getattr(topology, "topology_generation", 0) or 0),
            ),
            (
                "frozen_topology_generation",
                getattr(topology, "frozen_topology_generation", None),
            ),
            (
                "frozen_net_ids",
                tuple(int(net_id) for net_id in getattr(frozen_nets, "net_ids", ()) or ()),
            ),
            ("pos", None if candidate_live_geometry else pos.detach()),
        )
    )


def refresh_topology_if_needed(owner, pos, *, force=False):
    """Refresh or reuse the live topology and publish a compact status."""

    if getattr(getattr(owner, "placedb", None), "gr_sizing", None) is not None:
        state = {"status": "skipped", "reason": "frozen_gr_snapshot"}
        owner.last_topology_refresh_state = state
        owner.data.buffer_topology_refresh_state = dict(state)
        return state

    geometry_key = (
        None
        if force
        else candidate_topology_geometry_key(
            data=owner.data,
            lane=owner.lane,
            ops=owner.ops,
            pos=pos,
        )
    )
    if geometry_key is not None and geometry_key == getattr(
        owner,
        "_topology_geometry_key",
        None,
    ):
        owner.topology_refresh_reuse_count = (
            int(getattr(owner, "topology_refresh_reuse_count", 0)) + 1
        )
        state = {
            "status": "reused",
            "geometry_key": geometry_key,
            "topology_generation": int(
                getattr(owner.ops.steiner_topo_op, "topology_generation", 0)
            ),
        }
    else:
        owner.refresh_topology(pos)
        owner.topology_refresh_reuse_count = int(getattr(owner, "topology_refresh_reuse_count", 0))
        owner._topology_geometry_key = candidate_topology_geometry_key(
            data=owner.data,
            lane=owner.lane,
            ops=owner.ops,
            pos=pos,
        )
        owner.topology_refresh_count = int(getattr(owner, "topology_refresh_count", 0)) + 1
        state = {
            "status": "refreshed",
            "geometry_key": owner._topology_geometry_key,
            "topology_generation": int(
                getattr(
                    getattr(owner.ops, "steiner_topo_op", None),
                    "topology_generation",
                    0,
                )
                or 0
            ),
            "forced": bool(force),
        }
    owner.last_topology_refresh_state = state
    owner.data.buffer_topology_refresh_state = dict(state)
    return state


def prepare_candidate_owner(
    owner,
    model,
    pos,
    *,
    active_only=False,
    refresh_records_all=False,
    force_topology=False,
    lane_flow_kind=None,
    force_rebuild_when_unbounded=False,
    clear_timing_geometry_cache=False,
    reason="live_topology_refresh",
):
    """Prepare or incrementally refresh an owner's buffering lane."""

    from dreamplace.ops.buffer_insertion.buffering_config import (
        build_buffering_config_from_params,
    )
    from dreamplace.ops.buffer_insertion.buffering_lane import (
        BufferingOptimizationLane,
    )

    prepare_started_at = time.perf_counter()
    runtime_profile = {"active_only": bool(active_only)}

    def record_stage(name, started_at):
        runtime_profile[name] = (time.perf_counter() - started_at) * 1000.0

    def finish(status):
        runtime_profile["status"] = str(status)
        runtime_profile["total_ms"] = (time.perf_counter() - prepare_started_at) * 1000.0
        owner.last_prepare_runtime_profile = runtime_profile
        owner.data.buffer_prepare_runtime_profile = dict(runtime_profile)

    stage_started_at = time.perf_counter()
    counts = {} if owner.lane is None else owner._counts(owner.lane._current_state())
    record_stage("counts_ms", stage_started_at)

    stage_started_at = time.perf_counter()
    owner._candidate_refresh_in_progress = True
    try:
        refresh_topology_if_needed(owner, pos, force=force_topology)
    finally:
        owner._candidate_refresh_in_progress = False
    record_stage("topology_ms", stage_started_at)
    runtime_profile["topology_status"] = str(
        (getattr(owner, "last_topology_refresh_state", {}) or {}).get(
            "status",
            "unknown",
        )
    )

    existing_state = None if owner.lane is None else owner.lane._current_state()
    if existing_state is not None and hasattr(
        existing_state,
        "activation_param",
    ):
        dirty_net_ids = collect_owner_candidate_dirty_nets(owner)
        runtime_profile["candidate_dirty_summary"] = compact_candidate_dirty_summary(
            owner.last_candidate_dirty_net_summary
        )
        owner.lane.retain_candidate_python_state = True
        stage_started_at = time.perf_counter()
        refresh_summary = owner.lane.refresh_candidate_geometry(
            model=model,
            params=owner.params,
            reason=reason,
            active_only=active_only,
            dirty_net_ids=dirty_net_ids,
            refresh_records_all=bool(refresh_records_all),
            force_rebuild=bool(force_rebuild_when_unbounded and dirty_net_ids is None),
        )
        record_stage("candidate_refresh_ms", stage_started_at)
        if refresh_summary.get("status") != "ok":
            raise RuntimeError(f"S5B1 candidate state refresh failed: {refresh_summary}")
        owner.data.buffer_retained_candidate_state = refresh_summary
        runtime_profile["dirty_net_count"] = int(len(dirty_net_ids or ()))
        runtime_profile["refresh_records_all"] = bool(refresh_records_all)
        owner._pending_dirty_instance_ids = set()
        owner._pending_dirty_net_ids = set()
        if active_only:
            owner.ops.steiner_topo_op.frozen_nets.candidate_state = owner.lane._current_state()
        model._virtual_cell_density_view = None
        if clear_timing_geometry_cache:
            model._timing_geometry_cache = None
        finish("candidate_state_refreshed")
        return

    # The topology refresh can remap retained state; rebuild the count
    # transfer payload only after it has completed.
    if owner.lane is not None:
        counts = owner._counts(owner.lane._current_state())
    owner.data.buffer_retained_segment_edges = tuple(counts)

    if lane_flow_kind is None:
        resolved_flow_kind = getattr(owner.params, "flow_kind", "placement")
        resolved_flow_kind = getattr(
            resolved_flow_kind,
            "value",
            resolved_flow_kind,
        )
        lane_flow_kind = (
            str(resolved_flow_kind)
            if str(resolved_flow_kind) in {"buffering", "joint"}
            else "placement"
        )

    stage_started_at = time.perf_counter()
    lane = BufferingOptimizationLane(
        build_buffering_config_from_params(
            owner.params,
            flow_kind=str(lane_flow_kind),
        ),
        placedb=owner.placedb,
    )
    lane.prepare(model=model, params=owner.params)
    record_stage("lane_prepare_ms", stage_started_at)
    if hasattr(lane._current_state(), "activation_param"):
        lane.retain_candidate_python_state = True
    state = lane._current_state()
    if state is None:
        raise RuntimeError(f"S5B1 state construction failed: {lane.summarize()}")
    owner._transfer_counts(state, counts)
    if hasattr(state, "z_param"):
        state.z_param.requires_grad_(requires_grad=False)
        state.bsu_index_param.requires_grad_(requires_grad=False)
    lane.attach_state(owner.data)
    owner.lane = lane
    model._virtual_cell_density_view = None
    owner._pending_dirty_instance_ids = set()
    owner._pending_dirty_net_ids = set()
    finish("lane_created")


def collect_owner_candidate_dirty_nets(owner):
    """Collect pending sizing and buffering changes into candidate-net IDs."""

    state = None if owner.lane is None else owner.lane._current_state()
    dirty_net_ids, summary = collect_candidate_dirty_nets(
        state=state,
        lane=owner.lane,
        placedb=owner.placedb,
        data=owner.data,
        changed_instance_ids=getattr(
            owner,
            "_pending_dirty_instance_ids",
            (),
        ),
        accepted_net_ids=getattr(
            owner,
            "_pending_dirty_net_ids",
            (),
        ),
    )
    owner.last_candidate_dirty_net_summary = summary
    owner.data.buffer_candidate_dirty_net_summary = dict(compact_candidate_dirty_summary(summary))
    return dirty_net_ids


def refresh_owner_candidate_geometry_after_topology(
    owner,
    model,
    *,
    reason,
):
    """Refresh active retained candidates after publishing live topology."""

    from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
        active_candidate_net_ids,
    )

    if owner.lane is None:
        return {"status": "skipped", "reason": "lane_uninitialized"}
    state = owner.lane._current_state()
    if state is None or not hasattr(state, "activation_param"):
        return {"status": "skipped", "reason": "non_candidate_state"}
    active_net_ids = active_candidate_net_ids(state)
    if not active_net_ids:
        return {
            "status": "ok",
            "reason": "no_active_candidates",
            "candidate_count": 0,
        }
    summary = owner.lane.refresh_candidate_geometry(
        model=model,
        params=owner.params,
        reason=reason,
        active_only=True,
        dirty_net_ids=active_net_ids,
    )
    owner.data.buffer_retained_candidate_state = summary
    owner.lane.attach_state(owner.data)
    owner.ops.steiner_topo_op.frozen_nets.candidate_state = owner.lane._current_state()
    owner.params.enable_relaxed_buffer_timing = True
    model._virtual_cell_density_view = None
    model._timing_geometry_cache = None
    return summary


def compact_candidate_dirty_summary(summary):
    """Keep profiling payloads small while retaining failure diagnostics."""

    summary = dict(summary or {})
    return {
        key: summary[key] for key in ("status", "reason", "counts", "metadata") if key in summary
    }


def accepted_candidate_net_ids(activation_before, state):
    """Return candidate nets whose position became newly active."""

    active_before = (
        activation_before.detach().reshape(activation_before.shape[0], -1).sum(dim=1) > 0
    )
    active_after = (
        state.activation_param.detach().reshape(state.activation_param.shape[0], -1).sum(dim=1) > 0
    )
    accepted_rows = torch.nonzero(
        active_before ^ active_after,
        as_tuple=False,
    ).flatten()
    candidate_net_id = getattr(state, "candidate_net_id", None)
    if candidate_net_id is None or not accepted_rows.numel():
        return []
    return (
        torch.unique(
            candidate_net_id.index_select(
                0,
                accepted_rows.to(candidate_net_id.device),
            ),
            sorted=True,
        )
        .detach()
        .cpu()
        .tolist()
    )


__all__ = [
    "accepted_candidate_net_ids",
    "candidate_topology_geometry_key",
    "collect_owner_candidate_dirty_nets",
    "compact_candidate_dirty_summary",
    "prepare_candidate_owner",
    "refresh_owner_candidate_geometry_after_topology",
    "refresh_topology_if_needed",
]
