"""Project retained virtual buffers once from the final timing frame."""

import torch

from dreamplace.ops.buffer_insertion.buffering_config import build_buffering_config_from_params
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
from dreamplace.ops.routability.cooptimization_area import capture_area


def prepare_terminal_request(owner, model, pos, *, iteration, refresh_geometry=True):
    """Materialize all retained counts from the final frame, without reselection."""
    from dreamplace.ops.buffer_insertion.buffering_lane import _tensor_digest
    from dreamplace.ops.buffer_insertion.projection import project_segment_count_buffer_state

    if owner.terminal_request is not None:
        raise RuntimeError("S5B1 final materialization was already prepared")
    if not bool(torch.isfinite(pos).all()):
        raise RuntimeError("S5B1 final placement coordinates are non-finite")
    state = None if owner.lane is None else owner.lane._current_state()
    candidate_mode = state is not None and hasattr(state, "activation_param")
    cumulative_count = 0
    if state is not None:
        count_tensor = state.activation_param if candidate_mode else state.z_param
        cumulative_count = int(count_tensor.detach().sum().cpu().item())
    if owner.lane is None:
        owner.lane = BufferingOptimizationLane(
            build_buffering_config_from_params(owner.params, flow_kind="placement"),
            placedb=owner.placedb,
        )
        projection = None
    elif candidate_mode:
        # Rebuild the live candidate geometry before final projection so
        # no coordinate from an earlier placement generation is committed.
        # The unified diff_opt controller has already refreshed the lane
        # after its last explicit phase; skipping a second rebuild there
        # avoids repacking a partially frozen final topology.
        if refresh_geometry:
            owner.prepare(model, pos, refresh_records_all=True)
        projection = owner.lane.project(iteration=iteration, final=True)
        if projection.projected_buffer_count != cumulative_count:
            raise RuntimeError("final projection did not preserve every cumulative buffer")
    else:
        # Rebuild the ordinary nets and repack current retained references.
        # Frozen buffered nets retain their identity; no new S or B occurs.
        owner.prepare(model, pos)
        topology = owner.ops.steiner_topo_op
        generation = topology.freeze_topology()
        with torch.no_grad():
            _, x, y = model._timing_geometry_for_pos(pos)
        scale, shift = float(owner.params.scale_factor), owner.params.shift_factor
        context = {
            "coordinate_source": "final_live_geometry",
            "placement_snapshot_identity": _tensor_digest((("pos", pos),)),
            "placement_snapshot_iteration": int(iteration),
            "topology_generation": generation,
            "frozen_topology_generation": generation,
            "expected_frozen_topology_generation": generation,
            "rectilinear_path_policy": "x_then_y",
            "node_x_dbu": x.double() / scale + shift[0],
            "node_y_dbu": y.double() / scale + shift[1],
        }
        payload = owner.lane.buffer_relaxed_timing_payload
        projection = project_segment_count_buffer_state(
            owner.lane._current_state(),
            nets=payload["nets"],
            legal_table=payload["buffer_legal_table"],
            live_projection_context=context,
        )
        if projection.projected_buffer_count != cumulative_count:
            raise RuntimeError("final projection did not preserve every cumulative buffer")
    owner.terminal_request = owner.lane.build_commit_request(
        projection,
        iteration=iteration,
        reason="inflation_s5b1_final_commit",
        commit_enabled=True,
        refresh_mode="full_rebuild",
        rebuild_mode="full_rebuild",
    )
    model._timing_geometry_cache = None
    candidate_provider_was_enabled = bool(
        getattr(owner.params, "enable_relaxed_buffer_timing", False)
    )
    if candidate_mode:
        # Terminal reporting must include any accepted typed actions while
        # preserving the provider setting used by the regular GP loop.
        owner.params.enable_relaxed_buffer_timing = True
    try:
        with owner.electrical_window(), torch.no_grad():
            outputs = model.timing_obj(pos)
            loss = model._timing_loss(*outputs)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("S5B1 final virtual timing loss is non-finite")
            gp_timing = dict(
                zip(("wns", "tns", "ws", "ts"), (float(value) for value in outputs), strict=True)
            )
            timing = owner.ops.timing_propagation_op
            gp_timing["slew_violation_ns"] = timing.last_total_slew_violation
            gp_timing["cap_violation_ff"] = timing.last_total_cap_violation
            gp_timing["objective_terms"] = dict(model.last_timing_objective_terms)
    finally:
        owner.params.enable_relaxed_buffer_timing = candidate_provider_was_enabled
    model._timing_geometry_cache = None
    owner.terminal_summary = {
        "virtual_count": cumulative_count,
        "scale_factor": float(owner.params.scale_factor),
        "dbu": float(owner.placedb.dbu),
        "gp_timing": gp_timing,
        "gp_area": capture_area(
            owner.data, owner.placedb, owner.virtual_area(model), capacity=owner.capacity
        ).__dict__,
        "request": owner.terminal_request.to_summary(),
    }
    if candidate_mode:
        owner.lane.retain_candidate_python_state = False
        from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
            _release_candidate_python_state,
        )

        _release_candidate_python_state(owner.lane)
    return {
        "status": "request_ready",
        "iteration": int(iteration),
        "projected_buffer_count": cumulative_count,
        "commit_request": owner.terminal_request.to_summary(),
    }
