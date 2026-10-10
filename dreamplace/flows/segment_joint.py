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
