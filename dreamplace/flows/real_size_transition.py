"""State and accounting helpers for the ECC real-size transition."""

import copy
import logging
import time
from dataclasses import dataclass, field

from dreamplace.BasicPlace import size_var_to_logits
from .real_size_transition_profile import record_transition_stage, record_transition_bytes

import numpy as np
import torch


@dataclass
class RealSizeTransitionContext:
    """Explicit state boundary owned by one real-size transition."""

    data_collections: object
    placedb: object
    op_collections: object
    size_params: object
    last_global_place_model: object
    last_projection_frame: object
    discrete_gradient_topk_candidate_cache: object
    discrete_gradient_topk_candidate_cache_key: object
    active_profile: dict | None = None
    defer_placedb_sync: bool = False
    parameterization: str = "real_size"
    transition_done: bool = False
    projection_runtime_state: dict = field(default_factory=dict)
    restored_owner_state: dict | None = None


@dataclass(frozen=True)
class TransitionProjection:
    """Projection artifacts and the matching frame returned by the adapter."""

    paths: dict | None
    result: object
    frame: object


def clone_transition_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def reset_sizing_optimizer_state(optimizer):
    if optimizer is None:
        return
    sizing_parameters = {
        parameter
        for group in optimizer.param_groups
        if group.get("group_name") == "sizing"
        for parameter in group.get("params", ())
    }
    for parameter in sizing_parameters:
        optimizer.state.pop(parameter, None)


def transition_to_discrete(context, params, optimizer, iteration, *, project, refresh):
    """Project once, refresh geometry, replace the size owner, or roll back.

    The project adapter returns a matching TransitionProjection. The refresh
    adapter applies that projection to runtime state before parameter replacement.
    Caller-owned attributes are published by the NonLinearPlace compatibility
    entry in its finally block, including rollback state when refresh fails.
    """
    if context.parameterization != "real_size":
        return None
    if str(getattr(params, "real_size_execution_mode", "continuous_only")) != (
        "warmup_to_discrete"
    ) or context.transition_done:
        return None
    warmup_steps = int(getattr(params, "real_size_warmup_steps", 0) or 0)
    if int(iteration) + 1 < warmup_steps:
        return None

    projection_started = time.perf_counter()
    projection = project(
        params,
        stage_timing_summary={
            "artifact_scope": "real_size_warmup_transition",
            "iteration": int(iteration),
        },
    )
    result, frame = projection.result, projection.frame
    if projection.paths is None or result is None:
        raise RuntimeError("real_size warm-up transition requires a legal gate projection")
    if not result.validation.is_valid:
        raise RuntimeError(
            "real_size warm-up projection is invalid: "
            + "; ".join(result.validation.issues)
        )
    if frame is None or frame.projection_result is not result:
        raise RuntimeError("real_size warm-up transition requires a matching projection frame")
    data = context.data_collections
    old_real_size = getattr(data, "real_size", None)
    if old_real_size is None:
        raise RuntimeError("real_size warm-up transition has no real_size owner")
    profile = context.active_profile
    record_transition_stage(profile, "projection_pipeline_ms", projection_started)
    capture_started = time.perf_counter()
    transaction = capture_transaction(context, optimizer, frame)
    record_transition_stage(profile, "transaction_capture_ms", capture_started)
    transaction["projection_runtime_state"] = context.projection_runtime_state
    context.last_projection_frame = frame
    original_dynamics_mode = getattr(params, "continuous_size_dynamics_mode", "none")
    try:
        refresh_started = time.perf_counter()
        runtime_summary = refresh(params, context.placedb)
        record_transition_stage(profile, "runtime_refresh_total_ms", refresh_started)
        if profile is not None:
            profile.update({
                "candidate_instance_count": int(result.inst_ids.numel()),
                "changed_instance_count": int(frame.changed_inst_ids.numel()),
                "changed_pin_count": int(frame.changed_pin_ids.numel()),
                "state_digest": frame.state_digest,
                "topology_generation_before": int(runtime_summary.get(
                    "projection_frame_topology_generation_before", frame.topology_generation_before)),
                "topology_generation_after": int(runtime_summary.get(
                    "projection_frame_topology_generation_after", frame.topology_generation_before)),
                "timing_model_generation_before": int(runtime_summary.get(
                    "projection_frame_timing_model_generation_before", frame.timing_model_generation_before)),
                "timing_model_generation_after": int(runtime_summary.get(
                    "projection_frame_timing_model_generation_after", frame.timing_model_generation_before)),
                "topology_update_kind": runtime_summary.get("runtime_topology_refresh_kind"),
                "topology_rebuilt": bool(runtime_summary.get("runtime_topology_rebuilt", False)),
                "placedb_cpu_mirror_updated": not context.defer_placedb_sync,
            })

        conversion_started = time.perf_counter()
        projected_logits = size_var_to_logits(
            frame.projected_size_global.detach().clone(),
            data.inst_size_lower,
            data.inst_size_upper,
        ).to(device=old_real_size.device, dtype=old_real_size.dtype)
        new_size_logits = torch.nn.Parameter(projected_logits)
        record_transition_stage(profile, "size_to_logits_ms", conversion_started)
        parameter_indices = [
            index for index, parameter in enumerate(context.size_params)
            if parameter is old_real_size
        ]
        if len(parameter_indices) != 1:
            raise RuntimeError(
                "real_size warm-up transition requires exactly one registered real_size parameter"
            )
        reset_started = time.perf_counter()
        context.size_params[parameter_indices[0]] = new_size_logits
        for group in optimizer.param_groups:
            if group.get("group_name") == "sizing":
                group["params"] = [
                    new_size_logits if parameter is old_real_size else parameter
                    for parameter in group.get("params", ())
                ]
        optimizer.state.pop(old_real_size, None)
        data.size_logits = new_size_logits
        data.real_size = None
        data.sizing_parameterization = "logits"
        params.continuous_size_dynamics_mode = "discrete_gradient_topk"
        context.discrete_gradient_topk_candidate_cache = None
        context.discrete_gradient_topk_candidate_cache_key = None
        reset_sizing_optimizer_state(optimizer)
        record_transition_stage(profile, "optimizer_state_reset_ms", reset_started)
    except Exception:
        context.restored_owner_state = restore_transaction(context, transaction, optimizer)
        params.continuous_size_dynamics_mode = original_dynamics_mode
        context.transition_done = False
        raise

    finalize_started = time.perf_counter()
    context.transition_done = True
    summary = {
        "transitioned": True,
        "iteration": int(iteration),
        "warmup_steps": warmup_steps,
        "from_parameterization": "real_size",
        "to_parameterization": "logits",
        "projected_instance_count": int(result.inst_ids.numel()),
        "projection_paths": dict(projection.paths),
        "runtime_refresh": runtime_summary,
        "optimizer_state_reset": True,
        "discrete_dynamics_mode": "discrete_gradient_topk",
    }
    logging.info(
        "real_size warm-up transition at iteration=%d: projected %d instances "
        "and switched to discrete logits", int(iteration), int(result.inst_ids.numel()),
    )
    record_transition_stage(profile, "transition_finalize_ms", finalize_started)
    return summary


def sync_projected_real_size(data, parameterization, result):
    if parameterization != "real_size" or data is None:
        return {"applied": False, "reason": "logit_parameterization",
                "updated_instances": 0, "max_gap_before": 0.0}
    real_size = getattr(data, "real_size", None)
    if real_size is None:
        raise RuntimeError("real_size mode has no real_size parameter")
    legal_mask = getattr(result, "has_legal_candidate", None)
    inst_ids = getattr(result, "inst_ids", None)
    projected_size = getattr(result, "projected_size", None)
    if legal_mask is None or inst_ids is None or projected_size is None:
        raise RuntimeError("projection result lacks continuous size fields")
    legal_mask = legal_mask.bool()
    if legal_mask.numel() == 0 or not bool(torch.any(legal_mask).item()):
        return {"applied": False, "reason": "no_legal_projection",
                "updated_instances": 0, "max_gap_before": 0.0}
    ids = inst_ids[legal_mask].long().to(real_size.device)
    sizes = projected_size[legal_mask].to(device=real_size.device, dtype=real_size.dtype)
    before = real_size[ids].detach()
    max_gap = float((before - sizes).abs().max().item()) if before.numel() else 0.0
    with torch.no_grad():
        real_size[ids] = sizes
    return {"applied": True, "reason": "legal_projection",
            "updated_instances": int(ids.numel()), "max_gap_before": max_gap}


def capture_transaction(context, optimizer, frame):
    """Capture only state that the transition is allowed to mutate."""
    data_collections = context.data_collections
    changed_inst_ids = frame.changed_inst_ids
    changed_pin_ids = frame.changed_pin_ids
    tensor_slices = {}
    for name in (
        "inst_cell_id",
        "inst_libcell_offset",
        "node_size_x",
        "node_size_y",
        "node_areas",
        "original_node_size_x",
        "original_node_size_y",
        "inst_size_init",
        "vt_logits",
    ):
        value = getattr(data_collections, name, None)
        if value is not None and torch.is_tensor(value):
            ids = changed_inst_ids.to(device=value.device)
            tensor_slices[name] = (ids.detach().clone(), value[ids].detach().clone())
    for name in (
        "pin_offset_x",
        "pin_offset_y",
        "original_pin_offset_x",
        "original_pin_offset_y",
    ):
        value = getattr(data_collections, name, None)
        if value is not None and torch.is_tensor(value):
            ids = changed_pin_ids.to(device=value.device)
            tensor_slices[name] = (ids.detach().clone(), value[ids].detach().clone())

    placedb_slices = {}
    placedb = context.placedb
    if placedb is not None and not context.defer_placedb_sync:
        if changed_inst_ids.device.type != "cpu":
            record_transition_bytes(
                context.active_profile,
                "device_to_host_bytes",
                changed_inst_ids,
            )
        if changed_pin_ids.device.type != "cpu":
            record_transition_bytes(
                context.active_profile,
                "device_to_host_bytes",
                changed_pin_ids,
            )
        inst_ids_cpu = changed_inst_ids.detach().cpu().numpy()
        pin_ids_cpu = changed_pin_ids.detach().cpu().numpy()
        for name in (
            "inst_cell_id",
            "inst_libcell_offset",
            "node_size_x",
            "node_size_y",
            "inst_size_init",
        ):
            value = getattr(placedb, name, None)
            if value is not None:
                copied_values = np.asarray(value[inst_ids_cpu]).copy()
                placedb_slices[name] = (inst_ids_cpu.copy(), copied_values)
                if context.active_profile is not None:
                    context.active_profile["placedb_cpu_bytes_read"] += int(
                        copied_values.nbytes
                    )
        for name in ("pin_offset_x", "pin_offset_y"):
            value = getattr(placedb, name, None)
            if value is not None:
                copied_values = np.asarray(value[pin_ids_cpu]).copy()
                placedb_slices[name] = (pin_ids_cpu.copy(), copied_values)
                if context.active_profile is not None:
                    context.active_profile["placedb_cpu_bytes_read"] += int(
                        copied_values.nbytes
                    )

    topo_op = getattr(context.op_collections, "steiner_topo_op", None)
    topo_state = None
    if topo_op is not None:
        topo_state = {
            name: clone_transition_value(getattr(topo_op, name, None))
            for name in (
                "newx",
                "newy",
                "pin_relate_x",
                "pin_relate_y",
                "net_vertex_start",
                "net_steiner_start",
                "pin_fa",
                "flat_pin_to",
                "flat_pin_from",
                "flat_pin_to_start",
                "net_flat_topo_sort",
                "net_flat_topo_sort_start",
                "topology_generation",
                "rebuild_count",
                "_frozen_topology_generation",
            )
        }

    optimizer_groups = []
    optimizer_state = {}
    if optimizer is not None:
        optimizer_groups = [
            {
                key: (list(value) if key == "params" else value)
                for key, value in group.items()
            }
            for group in optimizer.param_groups
        ]
        optimizer_state = {
            parameter: {
                key: clone_transition_value(value) for key, value in state.items()
            }
            for parameter, state in optimizer.state.items()
        }

    model_state = None
    model = context.last_global_place_model
    if model is not None:
        density_view = getattr(model, "_virtual_cell_density_view", None)
        density_ops = getattr(density_view, "_density_ops", None)
        model_state = {
            "timing_geometry_cache": getattr(model, "_timing_geometry_cache", None),
            "density_ops": None if not isinstance(density_ops, dict) else dict(density_ops),
        }

    data_topology_names = (
        "net_flat_topo_sort",
        "net_flat_topo_sort_start",
        "pin_fa",
        "flat_pin_to",
        "flat_pin_to_start",
        "flat_pin_from",
        "buffering_timing_topology",
    )
    data_topology = {
        name: clone_transition_value(getattr(data_collections, name, None))
        for name in data_topology_names
        if hasattr(data_collections, name)
    }
    return {
        "tensor_slices": tensor_slices,
        "placedb_slices": placedb_slices,
        "topology_state": topo_state,
        "data_topology": data_topology,
        "runtime_cell_state_generation": getattr(
            data_collections, "runtime_cell_state_generation", 0
        ),
        "timing_model_generation": getattr(data_collections, "timing_model_generation", 0),
        "sizing_parameterization": getattr(
            data_collections, "sizing_parameterization", None
        ),
        "real_size": getattr(data_collections, "real_size", None),
        "size_logits": getattr(data_collections, "size_logits", None),
        "size_params": list(context.size_params),
        "optimizer_groups": optimizer_groups,
        "optimizer_state": optimizer_state,
        "model_state": model_state,
        "projection_frame": context.last_projection_frame,
        "discrete_gradient_topk_candidate_cache": context.discrete_gradient_topk_candidate_cache,
        "discrete_gradient_topk_candidate_cache_key": context.discrete_gradient_topk_candidate_cache_key,
    }


def restore_transaction(context, snapshot, optimizer):
    """Restore the captured state and return owner fields for the caller."""
    data_collections = context.data_collections
    for name, (ids, values) in snapshot["tensor_slices"].items():
        target = getattr(data_collections, name, None)
        if target is not None:
            with torch.no_grad():
                target[ids.to(device=target.device)] = values.to(
                    device=target.device,
                    dtype=target.dtype,
                )
    placedb = context.placedb
    if placedb is not None:
        for name, (ids, values) in snapshot["placedb_slices"].items():
            target = getattr(placedb, name, None)
            if target is not None:
                target[ids] = values

    for name, value in snapshot["data_topology"].items():
        setattr(data_collections, name, value)
    topo_op = getattr(context.op_collections, "steiner_topo_op", None)
    if topo_op is not None and snapshot["topology_state"] is not None:
        for name, value in snapshot["topology_state"].items():
            setattr(topo_op, name, value)

    data_collections.runtime_cell_state_generation = snapshot[
        "runtime_cell_state_generation"
    ]
    data_collections.timing_model_generation = snapshot["timing_model_generation"]
    data_collections.sizing_parameterization = snapshot["sizing_parameterization"]
    data_collections.real_size = snapshot["real_size"]
    data_collections.size_logits = snapshot["size_logits"]
    if optimizer is not None:
        optimizer.param_groups[:] = snapshot["optimizer_groups"]
        optimizer.state.clear()
        optimizer.state.update(snapshot["optimizer_state"])

    model = context.last_global_place_model
    model_state = snapshot["model_state"]
    if model is not None and model_state is not None:
        model._timing_geometry_cache = model_state["timing_geometry_cache"]
        density_view = getattr(model, "_virtual_cell_density_view", None)
        density_ops = getattr(density_view, "_density_ops", None)
        if isinstance(density_ops, dict) and model_state["density_ops"] is not None:
            density_ops.clear()
            density_ops.update(model_state["density_ops"])
    return {
        "size_params": torch.nn.ParameterList(snapshot["size_params"]),
        "projection_frame": snapshot["projection_frame"],
        "projection_runtime_state": snapshot.get("projection_runtime_state", {}),
        "discrete_gradient_topk_candidate_cache": snapshot[
            "discrete_gradient_topk_candidate_cache"
        ],
        "discrete_gradient_topk_candidate_cache_key": snapshot[
            "discrete_gradient_topk_candidate_cache_key"
        ],
    }
