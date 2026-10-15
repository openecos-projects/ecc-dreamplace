"""Existing placement step profiling: timestamps and files only."""
import os
import json
import time

def full_step_profile_enabled(params):
    return bool(getattr(params, "full_step_profile", False))


def full_step_profile_interval(params):
    try:
        return max(1, int(getattr(params, "full_step_profile_interval", 1) or 1))
    except (TypeError, ValueError):
        return 1


def full_step_profile_latest_path(params):
    return os.path.join(
        params.result_dir,
        f"{params.design_name()}_full_step_profile_latest.json",
    )


def full_step_profile_trace_path(params):
    return os.path.join(
        params.result_dir,
        f"{params.design_name()}_full_step_profile.jsonl",
    )


def reset_full_step_profile(params):
    if not full_step_profile_enabled(params):
        return
    for path in (
        full_step_profile_latest_path(params),
        full_step_profile_trace_path(params),
    ):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def should_write_full_step_profile(params, iteration):
    if not full_step_profile_enabled(params):
        return False
    return int(iteration) % full_step_profile_interval(params) == 0


def write_full_step_profile_record(params, record):
    if not full_step_profile_enabled(params):
        return None
    payload = dict(record)
    payload["artifact"] = "full_step_profile_latest"
    payload["artifact_version"] = 1
    payload["design_name"] = params.design_name()
    payload["full_step_profile_interval"] = full_step_profile_interval(params)
    payload["timestamp"] = time.time()
    latest_path = full_step_profile_latest_path(params)
    trace_path = full_step_profile_trace_path(params)
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    trace_payload = dict(payload)
    trace_payload["artifact"] = "full_step_profile_record"
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
    return {"latest_path": latest_path, "trace_path": trace_path, "payload": payload}


def record_iteration(write_record, params, iteration, Lgamma_step,
                     Llambda_density_weight_step, Lsub_step, profile_now,
                     profile_step_start, profile_stage_ms,
                     timing_backward_component_profile, piecewise_size_backend_profile,
                     cell_model_forward_profile):
    full_step_ms = (profile_now() - profile_step_start) * 1000.0
    accounted_stage_names = (
        "move_boundary_ms",
        "initialize_density_weight_ms",
        "zero_grad_ms",
        "eval_metrics_evaluate_ms",
        "refresh_live_timing_topology_ms",
        "obj_and_grad_fn_ms",
        "record_timing_metrics_ms",
        "size_debug_grad_stats_ms",
        "shared_topology_update_ms",
        "plot_ms",
        "optimizer_step_ms",
        "real_size_projection_ms",
        "net_weighting_update_ms",
        "metric_logging_ms",
        "best_pos_update_ms",
    )
    accounted_ms = sum(
        float(profile_stage_ms.get(name) or 0.0)
        for name in accounted_stage_names
    )
    record = {
        "iter": int(iteration),
        "detailed_step": [
            int(Lgamma_step),
            int(Llambda_density_weight_step),
            int(Lsub_step),
        ],
        "full_step_ms": full_step_ms,
        "profile_accounted_ms": accounted_ms,
        "unaccounted_ms": full_step_ms - accounted_ms,
        "profile_note": (
            "optimizer_step_ms is a top-level stage; "
            "optimizer_parameter_update_ms, discrete_gradient_topk_update_ms, "
            "and size_debug_trace_ms are nested diagnostics."
        ),
    }
    record.update(profile_stage_ms)
    if timing_backward_component_profile:
        record["timing_backward_component_profile"] = (
            timing_backward_component_profile
        )
    if piecewise_size_backend_profile:
        record["piecewise_size_backend_profile"] = (
            piecewise_size_backend_profile
        )
    if cell_model_forward_profile:
        record["cell_model_forward_profile"] = (
            cell_model_forward_profile
        )
    write_record(params, record)
