"""Runtime accounting for the ECC real-size transition."""


import json
import os
import resource
import time
from contextlib import contextmanager
import torch
import copy
import logging
from dataclasses import dataclass, field
from dreamplace.BasicPlace import size_var_to_logits
import numpy as np
from typing import Callable
from dreamplace.ops.discrete_gradient_topk import apply_discrete_gradient_topk_update, apply_quad_gradient_from_data_collections, build_quad_candidate_table
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState


def transition_profile_device(data_collections):
    if data_collections is None:
        return None
    for name in ("real_size", "size_logits", "inst_cell_id"):
        value = getattr(data_collections, name, None)
        if torch.is_tensor(value):
            return value.device
    return None


def synchronize_transition_device(data_collections):
    device = transition_profile_device(data_collections)
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def record_transition_stage(profile, name, started_at):
    if profile is not None:
        profile[name] = profile.get(name, 0.0) + (
            time.perf_counter() - started_at
        ) * 1000.0


def record_transition_bytes(profile, name, *tensors):
    if profile is None:
        return
    byte_count = 0
    for tensor in tensors:
        if torch.is_tensor(tensor):
            byte_count += int(tensor.numel()) * int(tensor.element_size())
    profile[name] = int(profile.get(name, 0)) + byte_count


def record_transition_artifacts(profile, paths):
    if profile is None:
        return
    artifact_bytes = 0
    artifact_count = 0
    for path in dict(paths or {}).values():
        if path and os.path.isfile(path):
            artifact_bytes += int(os.path.getsize(path))
            artifact_count += 1
    profile["artifact_bytes_written"] = int(
        profile.get("artifact_bytes_written", 0)
    ) + artifact_bytes
    profile["artifact_file_count"] = int(
        profile.get("artifact_file_count", 0)
    ) + artifact_count


def process_peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def finalize_transition_accounting(profile):
    coarse_stages = (
        "projection_pipeline_ms",
        "transaction_capture_ms",
        "runtime_refresh_total_ms",
        "size_to_logits_ms",
        "optimizer_state_reset_ms",
        "transition_finalize_ms",
        "cuda_synchronize_ms",
    )
    accounted_ms = sum(float(profile.get(name, 0.0)) for name in coarse_stages)
    total_ms = float(profile.get("transition_total_ms", 0.0))
    profile["accounted_stage_names"] = list(coarse_stages)
    profile["accounted_transition_ms"] = accounted_ms
    profile["unaccounted_transition_ms"] = max(total_ms - accounted_ms, 0.0)
    profile["accounted_transition_ratio"] = (
        min(accounted_ms / total_ms, 1.0) if total_ms > 0.0 else 0.0
    )


def write_transition_profile(profile, latest_path, trace_path):
    if not profile:
        return None
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as fp:
        json.dump(profile, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(profile, ensure_ascii=False) + "\n")
    return {"latest_path": latest_path, "trace_path": trace_path}


@contextmanager
def transition_profile(data, params, iteration, backend, latest_path, trace_path):
    """Measure one transition without owning placement or parameter state."""
    synchronize_transition_device(data)
    warmup_steps = int(getattr(params, "real_size_warmup_steps", 0) or 0)
    profile = {
        "artifact_version": 2,
        "status": "started",
        "clock": "perf_counter_wall",
        "backend": backend,
        "artifact_policy": str(
            getattr(
                params,
                "real_size_transition_artifact_policy",
                "summary_only",
            )
            or "summary_only"
        ),
        "iteration": int(iteration),
        "warmup_steps": warmup_steps,
        "projection_pipeline_ms": 0.0,
        "transaction_capture_ms": 0.0,
        "runtime_refresh_total_ms": 0.0,
        "transition_finalize_ms": 0.0,
        "projection_compute_ms": 0.0,
        "artifact_record_build_ms": 0.0,
        "artifact_write_ms": 0.0,
        "candidate_enumerate_ms": 0.0,
        "projection_context_ms": 0.0,
        "projection_request_ms": 0.0,
        "projection_score_ms": 0.0,
        "projection_resolve_ms": 0.0,
        "projection_validate_ms": 0.0,
        "projection_finalize_ms": 0.0,
        "projection_call_count": 0,
        "projection_frame_build_ms": 0.0,
        "pin_offset_compute_ms": 0.0,
        "runtime_cell_apply_ms": 0.0,
        "runtime_pin_offset_apply_ms": 0.0,
        "topology_refresh_ms": 0.0,
        "pin_pos_build_ms": 0.0,
        "pin_pos_device_sync_ms": 0.0,
        "topology_op_ms": 0.0,
        "timing_model_refresh_ms": 0.0,
        "timing_propagation_ms": 0.0,
        "size_to_logits_ms": 0.0,
        "optimizer_state_reset_ms": 0.0,
        "cuda_synchronize_ms": 0.0,
        "device_to_host_bytes": 0,
        "placedb_cpu_bytes_read": 0,
        "placedb_cpu_bytes_written": 0,
        "placedb_cpu_mirror_updated": False,
        "artifact_bytes_written": 0,
        "artifact_file_count": 0,
        "cpu_rss_start_bytes": process_peak_rss_bytes(),
    }
    device = transition_profile_device(data)
    profile["device"] = None if device is None else str(device)
    cuda = device is not None and device.type == "cuda" and torch.cuda.is_available()
    if cuda:
        profile["gpu_allocated_start_bytes"] = int(torch.cuda.memory_allocated(device))
        profile["gpu_reserved_start_bytes"] = int(torch.cuda.memory_reserved(device))
    started = time.perf_counter()
    try:
        yield profile
        profile["status"] = "completed"
    except Exception as exc:
        profile["status"] = "failed"
        profile["error"] = str(exc)
        raise
    finally:
        sync_started = time.perf_counter()
        synchronize_transition_device(data)
        profile["cuda_synchronize_ms"] += (time.perf_counter() - sync_started) * 1000.0
        profile["transition_total_ms"] = (time.perf_counter() - started) * 1000.0
        profile["cpu_peak_rss_bytes"] = process_peak_rss_bytes()
        if cuda:
            profile["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            profile["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        finalize_transition_accounting(profile)
        profile["profile_path"] = write_transition_profile(profile, latest_path, trace_path)


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


UNSET = object()

@dataclass
class DiscreteSizingStepContext:
    cache: Callable
    cache_key: Callable
    candidate_table: Callable
    config: Callable
    data_collections: object
    oscillation: Callable
    publish_candidate_cache: Callable
    publish_oscillation: Callable
    select: Callable
    sizing_mode: Callable
    skip_artifact: Callable
    sync_cells: Callable
    write_artifact: Callable
    summary: object = UNSET
    topk_summary: object = UNSET

def apply_discrete_sizing_dynamics(context, params, before_snapshot, iteration, total_iterations, config, summary, size_logits, before_logits, phase, progress, raw_delta_max_abs):
    data_collections = getattr(context, "data_collections", None)
    size_grad = None if size_logits.grad is None else size_logits.grad.detach().clone()
    vt_logits = getattr(data_collections, "vt_logits", None)
    before_vt_logits = (
        None if before_snapshot is None else before_snapshot.get("vt_logits")
    )
    vt_grad = (
        None
        if vt_logits is None or vt_logits.grad is None
        else vt_logits.grad.detach().clone()
    )
    vt_percent = float(config["discrete_gradient_topk_vt_percent"])
    shared_budget = config["discrete_gradient_topk_shared_budget"]
    if shared_budget and context.sizing_mode(params) != "size_only":
        raise ValueError("shared cell budget is currently supported only in size_only")
    quad_gradient_enabled = (
        (vt_percent > 0.0 or shared_budget)
        and vt_logits is not None
        and vt_logits.ndim == 2
        and (int(vt_logits.shape[1]) > 1 or shared_budget)
    )
    if shared_budget and not quad_gradient_enabled:
        raise RuntimeError("shared cell budget requires VT metadata")
    oscillation_state = None
    blocked_instances = None
    if shared_budget:
        oscillation_state = context.oscillation()
        if oscillation_state is None:
            oscillation_state = CellOscillationState()
            context.publish_oscillation(oscillation_state)
        if config["discrete_gradient_topk_oscillation_veto"]:
            blocked_instances = oscillation_state.blocked(
                phase="timing",
                iteration=iteration,
                count=int(data_collections.inst_cell_id.numel()),
                device=size_logits.device,
            )
    with torch.no_grad():
        size_logits.copy_(before_logits.detach())
        if quad_gradient_enabled:
            if vt_logits is None or before_vt_logits is None or (
                vt_grad is None and int(vt_logits.shape[1]) > 1
            ):
                raise RuntimeError(
                    "discrete quad-gradient requires vt_logits "
                    "and a populated vt gradient"
                )
            if vt_grad is None:
                vt_grad = torch.zeros_like(vt_logits)
            vt_logits.copy_(before_vt_logits.detach())
    if quad_gradient_enabled:
        cache_key = (
            "quad_gradient",
            str(size_logits.device),
            str(size_logits.dtype),
            tuple(int(dim) for dim in data_collections.flat_libcell_info.shape),
        )
        if context.cache_key() != cache_key:
            context.publish_candidate_cache(
                build_quad_candidate_table(
                    data_collections.flat_libcell_info,
                    data_collections.flat_libcell_leakage,
                ),
                cache_key,
            )
        target_logits, target_vt_logits, topk_summary = (
            apply_quad_gradient_from_data_collections(
                data_collections=data_collections,
                size_logits=before_logits.detach().to(
                    device=size_logits.device,
                    dtype=size_logits.dtype,
                ),
                size_grad=size_grad,
                vt_logits=before_vt_logits.detach().to(
                    device=vt_logits.device,
                    dtype=vt_logits.dtype,
                ),
                vt_grad=vt_grad,
                size_up_percent=config[
                    "discrete_gradient_topk_up_percent"
                ],
                size_down_percent=config[
                    "discrete_gradient_topk_down_percent"
                ],
                vt_percent=vt_percent,
                shared_budget_percent=(
                    config["discrete_gradient_topk_shared_budget_percent"]
                    if shared_budget
                    else None
                ),
                blocked_instances=blocked_instances,
                candidate_table=context.cache(),
            )
        )
    else:
        target_vt_logits = None
        target_logits, _target_sizes, topk_summary, _instance_candidate_table = (
            context.select(
                params=params,
                size_logits=before_logits.detach().to(
                    device=size_logits.device,
                    dtype=size_logits.dtype,
                ),
                size_grad=size_grad,
                policy={
                    "up_percent": config[
                        "discrete_gradient_topk_up_percent"
                    ],
                    "down_percent": config[
                        "discrete_gradient_topk_down_percent"
                    ],
                    "preserve_vt": config[
                        "discrete_gradient_topk_preserve_vt"
                    ],
                    "ranking_mode": config[
                        "discrete_gradient_topk_ranking_mode"
                    ],
                    "step_mode": config[
                        "discrete_gradient_topk_step_mode"
                    ],
                },
            )
        )
    with torch.no_grad():
        size_logits.copy_(target_logits.to(device=size_logits.device, dtype=size_logits.dtype))
        if quad_gradient_enabled:
            vt_logits.copy_(
                target_vt_logits.to(
                    device=vt_logits.device,
                    dtype=vt_logits.dtype,
                )
            )
        elif (
            config["discrete_gradient_topk_preserve_vt"]
            and vt_logits is not None
            and before_vt_logits is not None
        ):
            vt_logits.copy_(before_vt_logits.detach())
    applied_delta = size_logits.detach() - before_logits.detach()
    applied_delta_max_abs = (
        0.0
        if applied_delta.numel() == 0
        else float(applied_delta.abs().max().item())
    )
    topk_summary.update(
        {
            "iteration": int(iteration),
            "total_iterations": int(total_iterations),
            "phase": phase,
            "phase_progress": progress,
            "phase_lr_scale": 0.0,
            "raw_continuous_delta_max_abs": raw_delta_max_abs,
            "applied_delta_max_abs": applied_delta_max_abs,
            "config": {
                "up_percent": config["discrete_gradient_topk_up_percent"],
                "down_percent": config["discrete_gradient_topk_down_percent"],
                "vt_percent": vt_percent,
                "shared_budget": shared_budget,
                "oscillation_veto": config["discrete_gradient_topk_oscillation_veto"],
                "shared_budget_percent": (
                    config["discrete_gradient_topk_shared_budget_percent"]
                    if shared_budget
                    else None
                ),
                "preserve_vt": (
                    False
                    if quad_gradient_enabled
                    else config["discrete_gradient_topk_preserve_vt"]
                ),
                "ranking_mode": (
                    "quad_gradient_taylor"
                    if quad_gradient_enabled
                    else config["discrete_gradient_topk_ranking_mode"]
                ),
                "step_mode": config["discrete_gradient_topk_step_mode"],
                "lambda_update_mode": config["discrete_gradient_topk_lambda_update_mode"],
                "lambda_update_scale": config["discrete_gradient_topk_lambda_update_scale"],
                "lambda_min": config["discrete_gradient_topk_lambda_min"],
                "lambda_max": config["discrete_gradient_topk_lambda_max"],
            },
        }
    )
    logging.info(
        "discrete sizing update iteration=%d upsize_cells=%d "
        "downsize_cells=%d vt_cells=%d changed_cells=%d "
        "selected_up=%d selected_down=%d",
        int(iteration),
        int(topk_summary.get("num_up_applied", 0) or 0),
        int(topk_summary.get("num_down_applied", 0) or 0),
        int(topk_summary.get("num_vt_applied", 0) or 0),
        int(topk_summary.get("num_changed_instances", 0) or 0),
        int(topk_summary.get("num_up_selected", 0) or 0),
        int(topk_summary.get("num_down_selected", 0) or 0),
    )
    previous_cell_ids = []
    if oscillation_state is not None and topk_summary["applied_instance_ids"]:
        previous_cell_ids = data_collections.inst_cell_id[
            topk_summary["applied_instance_ids"]
        ].detach().cpu().tolist()
    topk_summary.update(context.sync_cells(params, topk_summary))
    if oscillation_state is not None:
        oscillation_state.record(
            phase="timing",
            iteration=iteration,
            instance_ids=topk_summary["applied_instance_ids"],
            previous_ids=previous_cell_ids,
            target_ids=topk_summary["applied_cell_ids"],
        )
        topk_summary["reversals"] = oscillation_state.reversals
        topk_summary["repeated_reversals"] = oscillation_state.repeated_reversals
        logging.info(
            "discrete oscillation iteration=%d frozen_cells=%d "
            "prevented_moves=%d replacement_actions=%d reversals=%d repeated_reversals=%d",
            int(iteration),
            topk_summary["num_blocked_instances"],
            topk_summary["num_prevented_moves"],
            topk_summary["num_replacement_actions"],
            oscillation_state.reversals,
            oscillation_state.repeated_reversals,
        )
    if context.skip_artifact(params):
        topk_summary["artifact_write_skipped_by_fast_loop"] = True
        artifact_info = None
        context.topk_summary = dict(topk_summary)
    else:
        topk_summary["artifact_write_skipped_by_fast_loop"] = False
        artifact_info = context.write_artifact(params, topk_summary)
    if artifact_info is not None:
        topk_summary["latest_path"] = artifact_info["latest_path"]
        topk_summary["trace_path"] = artifact_info["trace_path"]
    summary.update(
        {
            "enabled": True,
            "phase": phase,
            "phase_progress": progress,
            "phase_lr_scale": 0.0,
            "raw_delta_max_abs": raw_delta_max_abs,
            "trust_ratio": 1.0,
            "applied_delta_max_abs": applied_delta_max_abs,
            "discrete_gradient_topk": topk_summary,
        }
    )
    for key, value in topk_summary.items():
        if key not in ("ranking", "step", "size", "config"):
            summary[f"discrete_gradient_topk_{key}"] = value
    ranking_summary = topk_summary.get("ranking") or {}
    for key, value in ranking_summary.items():
        summary[f"discrete_gradient_topk_{key}"] = value
    step_summary = topk_summary.get("step") or {}
    for key, value in step_summary.items():
        summary[f"discrete_gradient_topk_{key}"] = value
    size_summary = topk_summary.get("size") or {}
    for key, value in size_summary.items():
        summary[f"discrete_gradient_topk_size_{key}"] = value
    context.summary = summary
    return summary


def select_discrete_gradient_topk(
    context,
    *,
    params,
    size_logits,
    size_grad,
    policy,
):
    data_collections = getattr(context, "data_collections", None)
    ranking_mode = str(policy["ranking_mode"])
    required_tensors = {
        "size_grad": size_grad,
        "inst_size_lower": getattr(data_collections, "inst_size_lower", None),
        "inst_size_upper": getattr(data_collections, "inst_size_upper", None),
        "inst_is_sizeable": getattr(data_collections, "inst_is_sizeable", None),
        "inst_cell_id": getattr(data_collections, "inst_cell_id", None),
        "flat_libcell_info": getattr(data_collections, "flat_libcell_info", None),
        "flat_libcell_leakage": (
            getattr(data_collections, "flat_libcell_leakage", None)
            if ranking_mode == "leakage_taylor"
            else torch.tensor(0)
        ),
    }
    missing_tensors = [
        name for name, value in required_tensors.items() if value is None
    ]
    if missing_tensors:
        raise RuntimeError(
            "discrete gradient top-k requires missing tensors: "
            + ", ".join(missing_tensors)
        )

    config = context.config(params)
    cache_config = dict(config)
    cache_config["discrete_gradient_topk_preserve_vt"] = bool(
        policy["preserve_vt"]
    )
    candidate_table = context.candidate_table(
        cache_config,
        data_collections,
        size_logits.device,
        size_logits.dtype,
    )
    target_logits, target_sizes, summary = apply_discrete_gradient_topk_update(
        size_logits=size_logits.detach(),
        size_grad=size_grad.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_size_lower=data_collections.inst_size_lower.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_size_upper=data_collections.inst_size_upper.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_is_sizeable=data_collections.inst_is_sizeable.to(
            device=size_logits.device
        ).bool(),
        inst_cell_id=data_collections.inst_cell_id.to(
            device=size_logits.device
        ).long(),
        flat_libcell_info=data_collections.flat_libcell_info.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        flat_libcell_leakage=(
            data_collections.flat_libcell_leakage.to(
                device=size_logits.device,
                dtype=size_logits.dtype,
            )
            if ranking_mode == "leakage_taylor"
            else None
        ),
        up_percent=float(policy["up_percent"]),
        down_percent=float(policy["down_percent"]),
        preserve_vt=bool(policy["preserve_vt"]),
        ranking_mode=ranking_mode,
        step_mode=str(policy["step_mode"]),
        lambda_update_mode=config["discrete_gradient_topk_lambda_update_mode"],
        lambda_update_scale=config["discrete_gradient_topk_lambda_update_scale"],
        lambda_min=config["discrete_gradient_topk_lambda_min"],
        lambda_max=config["discrete_gradient_topk_lambda_max"],
        instance_candidate_table=candidate_table,
    )
    return target_logits, target_sizes, summary, candidate_table
