"""Final Steiner rebuild and Route-B reselection for ECC segment joint flows.

The coordinator owns the terminal lane/counts and summaries. Topology refresh
uses the existing timing owner. Placement-owned buffer fields are published
through one callback, before model caches and proximal anchors are refreshed.
Failure keeps the existing propagation and gradient cleanup behavior.
"""

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Callable

import torch

from dreamplace.flows.metric_state import timing_scalar


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
