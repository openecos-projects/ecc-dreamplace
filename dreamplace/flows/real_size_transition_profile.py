"""Runtime accounting for the ECC real-size transition."""

import json
import os
import resource
import time
from contextlib import contextmanager

import torch


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
