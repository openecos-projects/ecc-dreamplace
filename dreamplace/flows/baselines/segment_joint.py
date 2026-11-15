"""ECC segment/joint scheduling transactions.

This module owns the scheduling decisions that surround a segment direct-joint
lane.  The placement engine remains responsible for the official optimizer
loop; this module only observes overflow, activates the lane, and commits a
fixed-window boundary through the coordinator.
"""


from dataclasses import dataclass
import copy
import math
import time
import torch
import hashlib
from typing import Callable
from dreamplace.flows.gp.metrics import timing_scalar


@dataclass
class SegmentJointContext:
    """Explicit owners used by one segment/joint scheduling transaction."""

    coordinator: object
    freeze_live_timing_topology: object
    refresh_live_timing_topology: object
    record_initial_inst_cell_id: object
    data_collections: object = None
    apply_discrete_sizing: object = None
    compact_sizing: object = None
    compute_buffer_gradient: object = None
    profile_enabled: object = None
    profile_clock: object = None
    capture_sizing_frame: object = None
    timing_scalar: object = None
    rebuild_pair_weights: object = None
    write_net_weight_artifact: object = None


def prepare_segment_direct_joint_iteration(
    context,
    *,
    model,
    pos,
    overflow,
    iteration,
):
    """Observe one placement step and activate the segment lane if needed."""
    coordinator = context.coordinator
    if coordinator is None or not coordinator.is_segment_direct_joint:
        return None

    overflow_value = float(
        torch.as_tensor(overflow).detach().reshape(-1)[-1].cpu().item()
    )
    if coordinator.uses_pin2pin_rebootstrap_schedule:
        observation = coordinator.observe_pin2pin_segment_iteration(
            overflow=overflow_value,
            iteration=iteration,
        )
        if observation["activation_due"]:
            context.record_initial_inst_cell_id(model)
            generation = context.freeze_live_timing_topology(pos)
            coordinator.activate_segment_lane(
                model=model,
                data_collections=model.data_collections,
                topology_generation=generation,
            )
        return {
            "mode": "pin2pin_rebootstrap",
            "observation": observation,
            "event": None,
        }

    event = coordinator.observe_segment_milestone(
        overflow=overflow_value,
        iteration=iteration,
    )
    if event is not None and event.get("activation"):
        context.record_initial_inst_cell_id(model)
        generation = context.freeze_live_timing_topology(pos)
        coordinator.activate_segment_lane(
            model=model,
            data_collections=model.data_collections,
            topology_generation=generation,
        )
    elif not coordinator.segment_milestones.active:
        context.refresh_live_timing_topology(pos)
    return event


def apply_segment_virtual_window_boundary(
    context,
    *,
    params,
    model,
    pos,
    optimizer,
    iteration,
):
    """Commit a fixed-window Route-B action at a placement boundary."""
    coordinator = context.coordinator
    if coordinator is None or not coordinator.is_segment_direct_joint:
        return None
    if not coordinator.uses_fixed_window_actions:
        return None

    window_steps = int(
        getattr(params, "joint_segment_virtual_window_steps", 0) or 0
    )
    if window_steps <= 0:
        return None
    outer_count = max(
        1,
        int(getattr(params, "joint_buffer_outer_iterations", 1) or 1),
    )
    completed_steps = int(iteration) + 1
    warmup_steps = int(
        getattr(params, "joint_segment_virtual_warmup_steps", 0) or 0
    )
    if warmup_steps > 0:
        if completed_steps < warmup_steps:
            return None
        offset = completed_steps - warmup_steps
        if offset % window_steps != 0:
            return None
        window_index = offset // window_steps
        placement_steps = warmup_steps if window_index == 0 else window_steps
        following_placement_steps = window_steps
    else:
        if completed_steps % window_steps != 0:
            return None
        window_index = completed_steps // window_steps - 1
        placement_steps = window_steps
        following_placement_steps = (
            0 if window_index + 1 == outer_count else window_steps
        )
    if window_index < 0 or window_index >= outer_count:
        return None
    return coordinator.apply_segment_virtual_window_boundary(
        model=model,
        pos=pos,
        optimizer=optimizer,
        iteration=completed_steps,
        window_index=window_index,
        placement_steps=placement_steps,
        following_placement_steps=following_placement_steps,
        final=window_index + 1 == outer_count,
    )


def apply_segment_overflow_milestone_transaction(
    context, *, params, model, pos, optimizer, event, sizing_frame
):
    """Apply sizing, refresh, gradient, and Route-B in milestone stage order."""
    coordinator = context.coordinator
    if event is None or not coordinator.uses_overflow_milestone_actions:
        return None
    data = context.data_collections
    try:
        coordinator.advance_segment_milestone(
            event=event,
            stage="placement_updated",
            evidence={"iteration": int(event["iteration"])},
        )
        sizing_started_at = time.perf_counter()
        sizing = context.apply_discrete_sizing(
            params=params,
            event=event,
            sizing_frame=sizing_frame,
        )
        sizing.setdefault(
            "runtime_ms",
            float((time.perf_counter() - sizing_started_at) * 1000.0),
        )
        compact_sizing = context.compact_sizing(sizing)
        coordinator.advance_segment_milestone(
            event=event,
            stage="sizing_applied",
            evidence={
                "status": compact_sizing.get("status"),
                "num_changed_instances": int(
                    compact_sizing.get("num_changed_instances", 0) or 0
                ),
                "action_digest": compact_sizing.get("action_digest"),
                "input_state_digest": compact_sizing.get("input_state_digest"),
                "output_state_digest": compact_sizing.get("output_state_digest"),
                "runtime_ms": compact_sizing.get("runtime_ms"),
            },
        )

        refresh_started_at = time.perf_counter()
        context.refresh_live_timing_topology(pos)
        topology = data.buffering_timing_topology
        refresh = {
            "status": "ok",
            "runtime_cell_state_generation": int(
                getattr(data, "runtime_cell_state_generation", 0) or 0
            ),
            "timing_cache_generation": int(
                sizing.get("timing_cache_generation", 0) or 0
            ),
            "runtime_pin_cap_view": sizing.get("runtime_pin_cap_view"),
            "topology_generation": int(topology.get("topology_generation", 0) or 0),
            "frozen_topology_generation": topology.get("frozen_topology_generation"),
            "area_delta_internal": float(
                sizing.get("area_delta_internal", 0.0) or 0.0
            ),
            "density_visible_area_delta_internal": float(
                sizing.get("density_visible_area_delta_internal", 0.0) or 0.0
            ),
            "runtime_ms": float((time.perf_counter() - refresh_started_at) * 1000.0),
        }
        if not math.isclose(
            refresh["area_delta_internal"],
            refresh["density_visible_area_delta_internal"],
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise RuntimeError("sizing area delta is not visible to density")
        coordinator.advance_segment_milestone(
            event=event,
            stage="runtime_state_refreshed",
            evidence=refresh,
        )

        buffering_gradient = context.compute_buffer_gradient(
            params=params,
            model=model,
            pos=pos,
            event=event,
            refresh_topology=False,
        )
        coordinator.advance_segment_milestone(
            event=event,
            stage="buffering_gradient_ready",
            evidence=buffering_gradient,
        )
        record = coordinator.complete_segment_milestone(
            event=event,
            sizing=compact_sizing,
            refresh=refresh,
            buffering_gradient=buffering_gradient,
            optimizer=optimizer,
            model=model,
            pos=pos,
        )
        state = coordinator.buffering_lane._current_state()
        state.z_param.grad = None
        if data.size_logits is not None:
            data.size_logits.grad = None
        return record
    except Exception as exc:
        pending = coordinator.segment_milestones.pending_event
        if pending is not None:
            coordinator.fail_segment_milestone(
                event=event,
                stage=str(pending.get("transaction_stage") or "unknown"),
                error=exc,
            )
        raise


def rebuild_and_install_segment_pin2pin(
    context,
    *,
    params,
    placedb,
    model,
    pos,
    optimizer,
    iteration,
    overflow,
    reason,
    force_clear_accumulation,
):
    coordinator = context.coordinator
    if not coordinator.uses_pin2pin_rebootstrap_schedule:
        raise RuntimeError("segment Pin2Pin rebuild requires rebootstrap scheduling")
    gate_status = {
        "enabled": True,
        "update_due": True,
        "iteration": int(iteration),
        "overflow": context.timing_scalar(overflow),
        "threshold": float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.35)
        ),
        "update_interval": int(
            getattr(params, "net_weighting_update_interval", 15) or 15
        ),
        "reason": str(reason),
    }
    payload, timing = context.rebuild_pair_weights(
        params,
        placedb,
        model,
        pos,
        iteration=iteration,
        gate_status=gate_status,
        update_reason=reason,
        force_clear_accumulation=force_clear_accumulation,
    )
    generation = coordinator.install_pin2pin_generation(
        iteration=iteration,
        reason=reason,
        status=payload["generation_status"],
        pair_count=payload["pin2pin_pair_count"],
        metadata={
            "update_count": payload["update_count"],
            "wns": payload["wns"],
            "tns": payload["tns"],
            "pin2pin_endpoint_limit": payload["pin2pin_endpoint_limit"],
            "pin2pin_path_pair_backend": payload["pin2pin_path_pair_backend"],
            "pin2pin_backend": payload["pin2pin_backend"],
        },
    )
    if not callable(getattr(optimizer, "rebase_objective_state", None)):
        raise RuntimeError(
            "Pin2Pin generation changes require Nesterov objective rebase"
        )
    rebase = optimizer.rebase_objective_state(reason=reason)
    payload.update({"pair_generation": generation, "nesterov_rebase": rebase})
    context.write_net_weight_artifact(params, payload)
    return {
        "payload": payload,
        "timing": timing,
        "generation": generation,
        "nesterov_rebase": rebase,
    }


def run_segment_milestone_sizing_micro_loop(
    context, *, params, model, pos, event
):
    requested_rounds = int(getattr(params, "joint_segment_sizing_rounds", 5) or 5)
    sizing_enabled = bool(getattr(params, "joint_segment_sizing_enabled", True))
    profile_enabled = context.profile_enabled(params)
    started_at = context.profile_clock(profile_enabled, pos)
    rounds = []
    terminal_reason = "disabled" if not sizing_enabled else "max_rounds"
    if sizing_enabled:
        for round_index in range(requested_rounds):
            frame = context.capture_sizing_frame(
                params=params,
                model=model,
                pos=pos,
                event=event,
                evaluate_objective=True,
                advance_stage=False,
            )
            frame["summary"]["frame"] = "post_tdp_step_sizing_round"
            frame["summary"]["round_index"] = int(round_index)
            sizing = context.apply_discrete_sizing(
                params=params,
                event=event,
                sizing_frame=frame,
            )
            compact = context.compact_sizing(sizing)
            compact["round_index"] = int(round_index)
            compact["gradient_digest"] = frame["summary"].get("gradient_digest")
            rounds.append(compact)
            changed = int(sizing.get("num_changed_instances", 0) or 0)
            if changed == 0:
                terminal_reason = "stationary"
                break
            context.refresh_live_timing_topology(pos)
            model._timing_geometry_cache = None
            if context.data_collections.size_logits is not None:
                context.data_collections.size_logits.grad = None
    summary = {
        "status": "disabled" if not sizing_enabled else "completed",
        "rounds_requested": requested_rounds,
        "rounds_attempted": len(rounds),
        "rounds_completed": len(rounds),
        "terminal_reason": terminal_reason,
        "rounds": rounds,
        "changed_instance_count": sum(
            int(record.get("num_changed_instances", 0) or 0) for record in rounds
        ),
        "area_delta_internal": sum(
            float(record.get("area_delta_internal", 0.0) or 0.0)
            for record in rounds
        ),
    }
    finished_at = context.profile_clock(profile_enabled, pos)
    summary["runtime_ms"] = float((finished_at - started_at) * 1000.0)
    if profile_enabled:
        summary["runtime_profile"] = {
            "enabled": True,
            "synchronized": bool(pos.is_cuda),
            "total_ms": float((finished_at - started_at) * 1000.0),
        }
    context.coordinator.advance_segment_milestone(
        event=event,
        stage="sizing_micro_loop_completed",
        evidence=copy.deepcopy(summary),
    )
    return summary


def apply_segment_rebootstrap_milestone_transaction(
    context, *, params, placedb, model, pos, optimizer, event
):
    coordinator = context.coordinator
    if event is None or not coordinator.uses_pin2pin_rebootstrap_schedule:
        return None
    try:
        sizing = run_segment_milestone_sizing_micro_loop(
            context, params=params, model=model, pos=pos, event=event
        )
        context.refresh_live_timing_topology(pos)
        buffering_gradient = context.compute_buffer_gradient(
            params=params,
            model=model,
            pos=pos,
            event=event,
            refresh_topology=False,
        )
        coordinator.advance_segment_milestone(
            event=event,
            stage="buffering_gradient_ready",
            evidence=buffering_gradient,
        )
        route_b = coordinator.apply_rebootstrap_milestone_route_b(
            event=event,
            buffering_gradient=buffering_gradient,
            optimizer=optimizer,
            model=model,
            pos=pos,
        )
        previous_generation = coordinator.pin2pin_pair_generations.current_generation_id
        coordinator.pin2pin_pair_generations.invalidate(
            iteration=event["iteration"],
            reason="milestone_state_change",
        )
        coordinator.advance_segment_milestone(
            event=event,
            stage="tdp_state_invalidated",
            evidence={
                "pair_generation": previous_generation,
                "reason": "milestone_state_change",
            },
        )
        rebuilt = rebuild_and_install_segment_pin2pin(
            context,
            params=params,
            placedb=placedb,
            model=model,
            pos=pos,
            optimizer=optimizer,
            iteration=event["iteration"],
            overflow=event["overflow"],
            reason="milestone_rebootstrap",
            force_clear_accumulation=True,
        )
        rebuild_payload = dict(rebuilt.get("payload", {}) or {})
        coordinator.advance_segment_milestone(
            event=event,
            stage="tdp_rebootstrap_completed",
            evidence={
                "pair_generation_before": previous_generation,
                "pair_generation_after": rebuilt["generation"]["generation_id"],
                "generation_status": rebuilt["generation"]["status"],
                "pair_count": rebuilt["generation"]["pair_count"],
                "runtime_ms": rebuild_payload.get("update_ms"),
                "nesterov_rebase": rebuilt["nesterov_rebase"],
            },
        )
        runtime_ms = {
            "sizing_micro_loop": float(sizing.get("runtime_ms", 0.0) or 0.0),
            "buffering_gradient": float(
                buffering_gradient.get("runtime_ms", 0.0) or 0.0
            ),
            "route_b": float(route_b.get("runtime_ms", 0.0) or 0.0),
            "pin2pin_rebuild": float(rebuild_payload.get("update_ms", 0.0) or 0.0),
        }
        runtime_ms["total"] = float(sum(runtime_ms.values()))
        transition = {
            "sizing_micro_loop": sizing,
            "buffering_gradient": buffering_gradient,
            "route_b": route_b,
            "pin2pin_rebuild": {
                "runtime_ms": runtime_ms["pin2pin_rebuild"],
                "runtime_profile": rebuild_payload.get("runtime_profile"),
            },
            "runtime_ms": runtime_ms,
            "pair_generation_before": previous_generation,
            "pair_generation_after": rebuilt["generation"]["generation_id"],
            "accepted_action_count": int(route_b.get("accepted_action_count", 0) or 0),
            "buffer_count_before": int(route_b.get("buffer_count_before", 0) or 0),
            "buffer_count_after": int(route_b.get("buffer_count_after", 0) or 0),
        }
        record = coordinator.complete_rebootstrap_segment_milestone(
            event=event,
            transition=transition,
        )
        state = coordinator.buffering_lane._current_state()
        state.z_param.grad = None
        if context.data_collections.size_logits is not None:
            context.data_collections.size_logits.grad = None
        return record
    except Exception as exc:
        pending = coordinator.segment_milestones.pending_event
        if pending is not None:
            coordinator.fail_segment_milestone(
                event=event,
                stage=str(pending.get("transaction_stage") or "unknown"),
                error=exc,
            )
        raise


@dataclass
class TerminalLaneContext:
    coordinator: object
    data_collections: object
    op_collections: object
    refresh_topology: Callable
    publish_lane: Callable


@dataclass
class TerminalRouteBContext:
    coordinator: object
    model: object
    pos: torch.Tensor
    digest_tensors: Callable
    static_snapshot: Callable
    arc_snapshot: Callable
    rebuild_state: Callable
    compute_gradient: Callable


def terminal_static_timing_snapshot(data, *, params, model, pos, frame):
    """Evaluate the current PySTA state with the segment provider disabled."""
    model_params = getattr(model, "params", params)
    missing = object()
    previous = getattr(model_params, "enable_relaxed_buffer_timing", missing)
    started_at = time.perf_counter()
    try:
        model_params.enable_relaxed_buffer_timing = False
        model._timing_geometry_cache = None
        with torch.no_grad():
            wns, tns, ws, ts = model.timing_obj(pos)
    finally:
        model._timing_geometry_cache = None
        if previous is missing:
            delattr(model_params, "enable_relaxed_buffer_timing")
        else:
            model_params.enable_relaxed_buffer_timing = previous
    topology = getattr(data, "buffering_timing_topology", {}) or {}
    return {
        "frame": str(frame),
        "provider_enabled": False,
        "wns": timing_scalar(wns),
        "tns": timing_scalar(tns),
        "ws": timing_scalar(ws),
        "ts": timing_scalar(ts),
        "topology_generation": int(topology.get("topology_generation", 0) or 0),
        "runtime_cell_state_generation": int(
            getattr(data, "runtime_cell_state_generation", 0) or 0
        ),
        "runtime_ms": float((time.perf_counter() - started_at) * 1000.0),
    }


def terminal_arc_cell_state_snapshot(data):
    """Report arc rows whose stored Liberty cell differs from runtime sizing."""
    if data is None:
        return {"status": "unavailable", "reason": "data_collections_missing"}
    pin2node = getattr(data, "pin2node_map", None)
    inst_cell_id = getattr(data, "inst_cell_id", None)
    summary = {}
    for name in ("flat_inst_arcs_by_level", "endpoints_constraint_arcs"):
        arcs = getattr(data, name, None)
        row = {"row_count": 0, "valid_row_count": 0, "mismatch_count": 0}
        if (
            torch.is_tensor(arcs)
            and arcs.ndim == 2
            and int(arcs.shape[1]) >= 3
            and torch.is_tensor(pin2node)
            and torch.is_tensor(inst_cell_id)
        ):
            row["row_count"] = int(arcs.shape[0])
            output_pin = arcs[:, 1].long().to(pin2node.device)
            valid_pin = (output_pin >= 0) & (output_pin < int(pin2node.numel()))
            safe_pin = output_pin.clamp(min=0, max=max(int(pin2node.numel()) - 1, 0))
            node_id = pin2node[safe_pin].long()
            valid_node = valid_pin & (node_id >= 0) & (node_id < int(inst_cell_id.numel()))
            row["valid_row_count"] = int(valid_node.sum().detach().cpu().item())
            if bool(valid_node.any().detach().cpu().item()):
                expected = inst_cell_id[
                    node_id[valid_node].to(inst_cell_id.device)
                ].long()
                stored = arcs[valid_node.to(arcs.device), 2].long().to(expected.device)
                mismatch = stored != expected
                row["mismatch_count"] = int(mismatch.sum().detach().cpu().item())
                if bool(mismatch.any().detach().cpu().item()):
                    row["mismatched_instance_count"] = int(
                        torch.unique(node_id[valid_node][mismatch.to(node_id.device)])
                        .numel()
                    )
                else:
                    row["mismatched_instance_count"] = 0
        summary[name] = row
    return summary


def rebuild_terminal_segment_route_b_state(context, *, params, model, pos):
    """Build a zero-initialized segment state on the final Steiner tree."""
    from dreamplace.ops.buffer_insertion.buffering_lane import (
        BufferingOptimizationLane,
    )

    started_at = time.perf_counter()
    coordinator = context.coordinator
    old_lane = coordinator.buffering_lane
    old_state = old_lane._current_state()
    topo_op = context.op_collections.steiner_topo_op
    generation_before = int(getattr(topo_op, "topology_generation", 0) or 0)
    frozen_generation_before = getattr(
        topo_op,
        "frozen_topology_generation",
        None,
    )
    if frozen_generation_before is None:
        raise RuntimeError(
            "terminal segment Route-B topology rebuild requires a frozen tree"
        )
    topo_op.unfreeze_topology()
    with torch.no_grad():
        context.refresh_topology(pos)
    generation_after = int(getattr(topo_op, "topology_generation", 0) or 0)
    if generation_after <= generation_before:
        raise RuntimeError("terminal segment Route-B did not rebuild the Steiner tree")
    frozen_generation_after = int(topo_op.freeze_topology())
    with torch.no_grad():
        topology = context.refresh_topology(pos)
    if int(topology.get("topology_generation", 0) or 0) != generation_after:
        raise RuntimeError("terminal segment Route-B published the wrong topology")
    if int(topology.get("frozen_topology_generation", 0) or 0) != generation_after:
        raise RuntimeError("terminal segment Route-B did not freeze the rebuilt tree")

    fresh_lane = BufferingOptimizationLane(
        old_lane.config,
        placedb=old_lane.placedb,
    )
    fresh_lane.prepare(model=model, params=params)
    fresh_lane.attach_state(context.data_collections)
    fresh_lane.register_model_parameters(model)
    fresh_state = fresh_lane._current_state()
    if fresh_state is None or not hasattr(fresh_state, "z_param"):
        raise RuntimeError("terminal Steiner rebuild did not create SegmentCountState")
    with torch.no_grad():
        fresh_state.z_param.zero_()
    if not bool((fresh_state.z_param.detach() == 0.0).all()):
        raise RuntimeError("terminal SegmentCountState did not initialize from zero")

    coordinator.buffering_lane = fresh_lane
    coordinator._segment_lane_prepared = True
    coordinator.summary["buffering_lane"] = fresh_lane.summarize()
    context.publish_lane(fresh_lane, fresh_state)
    model._timing_geometry_cache = None
    model._relaxed_buffer_dynamic_net_provider = None
    model._relaxed_buffer_dynamic_net_provider_key = None
    if hasattr(model, "_virtual_cell_density_view"):
        model._virtual_cell_density_view = None
    if context.op_collections is not None:
        context.op_collections.virtual_cell_density_op = None

    anchor_summary = None
    proximal = getattr(coordinator, "_proximal_objective", None)
    if proximal is not None:
        anchor_summary = proximal.capture_anchors(model, pos, force=True)
    summary = {
        "status": "rebuilt",
        "policy": "discard_uncommitted_z_before_topology_rebuild",
        "topology_generation_before": generation_before,
        "frozen_topology_generation_before": int(frozen_generation_before),
        "topology_generation_after": generation_after,
        "frozen_topology_generation_after": frozen_generation_after,
        "old_segment_count": int(old_state.z_param.numel()),
        "new_segment_count": int(fresh_state.z_param.numel()),
        "buffer_timing_topology_epoch": int(
            getattr(
                context.data_collections,
                "buffer_segment_timing_topology_epoch",
                0,
            )
            or 0
        ),
        "proximal_anchor": copy.deepcopy(anchor_summary),
        "runtime_ms": float((time.perf_counter() - started_at) * 1000.0),
    }
    coordinator.summary["terminal_segment_topology_rebuild"] = copy.deepcopy(
        summary
    )
    return fresh_state, summary


def terminal_route_b_round_budget(coordinator):
    """Select eligibility without requiring a model on skipped paths."""
    if (
        coordinator is None
        or not bool(getattr(coordinator, "is_segment_direct_joint", False))
        or not bool(
            getattr(coordinator, "uses_overflow_milestone_actions", False)
        )
        or not bool(getattr(coordinator, "segment_route_b_enabled", False))
        or not bool(getattr(coordinator, "_segment_lane_prepared", False))
    ):
        return 0
    if bool(
        getattr(coordinator, "uses_pin2pin_rebootstrap_schedule", False)
    ):
        coordinator.summary["terminal_segment_reselection"] = {
            "status": "skipped",
            "reason": "preserve_milestone_route_b_state_for_core_commit",
        }
        return 0

    milestone_records = [
        record
        for record in coordinator.segment_milestones.trace
        if str(record.get("status")) == "consumed"
    ]
    return len(milestone_records)


def reselect_terminal_segment_route_b(
    context, params, *, iteration, optimizer, round_budget
):
    """Recompute terminal Route-B actions on the rebuilt placement frame."""
    coordinator = context.coordinator
    model = context.model
    pos = context.pos
    if (
        model is None
        or not callable(getattr(model, "timing_obj", None))
        or not callable(getattr(model, "_timing_loss", None))
    ):
        raise RuntimeError(
            "terminal segment Route-B reselection requires the live timing model"
        )

    historical_state = coordinator.buffering_lane._current_state()
    historical_counts = historical_state.z_param.detach().clone()
    historical_digest = context.digest_tensors(
        (("z_param", historical_counts),)
    )
    timing_state_probe = {
        "static_before_rebuild": context.static_snapshot(
            params=params,
            model=model,
            pos=pos,
            frame="terminal_before_fresh_tree_rebuild",
        ),
        "arc_cell_state_before_rebuild": context.arc_snapshot(),
    }
    state, topology_rebuild = context.rebuild_state(
        params=params,
        model=model,
        pos=pos,
    )
    timing_state_probe["static_after_rebuild"] = (
        context.static_snapshot(
            params=params,
            model=model,
            pos=pos,
            frame="terminal_after_fresh_tree_rebuild",
        )
    )
    timing_state_probe["arc_cell_state_after_rebuild"] = (
        context.arc_snapshot()
    )
    coordinator.summary["terminal_segment_timing_state_probe"] = copy.deepcopy(
        timing_state_probe
    )
    expected_topology_generation = int(
        topology_rebuild["topology_generation_after"]
    )
    rounds = []
    terminal_reason = "max_rounds"

    try:
        for round_index in range(round_budget):
            event = {
                "event_id": -(round_index + 1),
                "iteration": int(iteration),
            }
            gradient = context.compute_gradient(
                params=params,
                model=model,
                pos=pos,
                event=event,
                refresh_topology=False,
                frame="terminal_final_placement_sizing",
                expected_topology_generation=expected_topology_generation,
            )
            if round_index == 0:
                segment_zero = {
                    "frame": "terminal_after_fresh_tree_rebuild",
                    "provider_enabled": True,
                    "forced_z": 0.0,
                    "timing_loss": float(gradient["timing_loss"]),
                }
                for key in ("wns", "tns"):
                    if gradient.get(key) is not None:
                        segment_zero[key] = float(gradient[key])
                for key in (
                    "topology_generation",
                    "runtime_cell_state_generation",
                ):
                    if gradient.get(key) is not None:
                        segment_zero[key] = int(gradient[key])
                timing_state_probe["segment_zero_after_rebuild"] = segment_zero
                static_before = timing_state_probe["static_before_rebuild"]
                static_after = timing_state_probe["static_after_rebuild"]
                if segment_zero.get("tns") is not None:
                    timing_state_probe["tns_deltas"] = {
                        "static_after_minus_before": float(
                            static_after["tns"] - static_before["tns"]
                        ),
                        "segment_zero_minus_static_after": float(
                            segment_zero["tns"] - static_after["tns"]
                        ),
                    }
                coordinator.summary["terminal_segment_timing_state_probe"] = (
                    copy.deepcopy(timing_state_probe)
                )
            transition = coordinator._apply_segment_route_b_transition(
                optimizer=optimizer,
                model=model,
                pos=pos,
            )
            state.z_param.grad = None
            rounds.append(
                {
                    "round_index": int(round_index),
                    "gradient": copy.deepcopy(gradient),
                    "transition": copy.deepcopy(transition),
                }
            )
            if int(transition.get("accepted_action_count", 0) or 0) == 0:
                terminal_reason = (
                    "no_improving_prefix"
                    if int(
                        transition.get("gradient_selected_action_count", 0) or 0
                    )
                    > 0
                    else "no_positive_action"
                )
                break
    except Exception:
        state.z_param.grad = None
        raise

    final_counts = state.z_param.detach().clone()
    result = {
        "status": "reselected",
        "policy": "rebuild_tree_reset_then_terminal_frame_route_b",
        "frame": "terminal_final_placement_sizing",
        "iteration": int(iteration),
        "round_budget": int(round_budget),
        "iterations": len(rounds),
        "terminal_reason": terminal_reason,
        "historical_buffer_count": int(historical_counts.sum().cpu().item()),
        "historical_nonzero_segment_count": int(
            (historical_counts != 0.0).sum().cpu().item()
        ),
        "historical_state_digest": historical_digest,
        "topology_rebuild": copy.deepcopy(topology_rebuild),
        "final_buffer_count": int(final_counts.sum().cpu().item()),
        "final_nonzero_segment_count": int(
            (final_counts != 0.0).sum().cpu().item()
        ),
        "final_state_digest": context.digest_tensors(
            (("z_param", final_counts),)
        ),
        "rounds": rounds,
    }
    coordinator.summary["terminal_segment_route_b_reselection"] = copy.deepcopy(
        result
    )
    return result

@dataclass
class FinalProjectionContext:
    joint_coordinator: object
    data_collections: object
    params: object
    refresh_topology: Callable
    summary: object = None

def final_segment_joint_projection_context(context, pos, *, iteration):
    coordinator = getattr(context, "joint_coordinator", None)
    if coordinator is None or not coordinator.is_segment_direct_joint:
        return None
    milestone_state = coordinator.segment_milestones
    if milestone_state is None or not milestone_state.active:
        return None
    terminal_topology_rebuild = dict(
        coordinator.summary.get("terminal_segment_topology_rebuild", {}) or {}
    )
    expected_generation = terminal_topology_rebuild.get(
        "topology_generation_after",
        milestone_state.frozen_topology_generation,
    )
    if expected_generation is None:
        raise RuntimeError("segment joint final projection has no frozen generation")

    with torch.no_grad():
        topology = context.refresh_topology(pos)
    topology_generation = int(topology.get("topology_generation", 0) or 0)
    frozen_generation = int(
        topology.get("frozen_topology_generation", 0) or 0
    )
    if topology_generation != int(expected_generation):
        raise RuntimeError(
            "segment joint final projection topology generation mismatch"
        )
    if frozen_generation != int(expected_generation):
        raise RuntimeError(
            "segment joint final projection frozen generation mismatch"
        )

    final_pos = pos.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(final_pos.shape)).encode("ascii"))
    digest.update(str(final_pos.dtype).encode("ascii"))
    digest.update(final_pos.numpy().tobytes())
    snapshot_identity = "sha256:" + digest.hexdigest()

    scale_factor = float(
        getattr(
            context.data_collections,
            "buffering_timing_topology_scale_factor",
            1.0,
        )
        or 1.0
    )
    shift_factor = getattr(context.params, "shift_factor", (0.0, 0.0))
    shift_x = float(shift_factor[0]) if len(shift_factor) > 0 else 0.0
    shift_y = float(shift_factor[1]) if len(shift_factor) > 1 else 0.0
    node_x_dbu = topology["node_x"].detach().to(torch.float64) / scale_factor
    node_y_dbu = topology["node_y"].detach().to(torch.float64) / scale_factor
    node_x_dbu = node_x_dbu + shift_x
    node_y_dbu = node_y_dbu + shift_y
    payload = {
        "coordinate_source": "final_live_geometry",
        "placement_snapshot_identity": snapshot_identity,
        "placement_snapshot_iteration": int(iteration),
        "topology_generation": topology_generation,
        "frozen_topology_generation": frozen_generation,
        "expected_frozen_topology_generation": int(expected_generation),
        "milestone_frozen_topology_generation": int(
            milestone_state.frozen_topology_generation
        ),
        "terminal_topology_rebuilt": bool(terminal_topology_rebuild),
        "rectilinear_path_policy": "x_then_y",
        "node_x_dbu": node_x_dbu,
        "node_y_dbu": node_y_dbu,
    }
    context.summary = {
        key: value
        for key, value in payload.items()
        if key not in {"node_x_dbu", "node_y_dbu"}
    }
    context.summary.update(
        {
            "coordinate_count": int(node_x_dbu.numel()),
            "openroad_placement_sync": "deferred_to_final_physical_executor",
        }
    )
    return payload
