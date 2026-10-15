import logging
import types
import json
import os
import time
from dataclasses import dataclass

from dreamplace.flows.flow_config import FlowKind, resolved_flow_kind


@dataclass
class OptimizationFlowResult:
    flow_kind: object
    engine: object


def _profile_enabled(params):
    return bool(getattr(params, "initialization_profile", False))


def _design_name(params):
    design_name_attr = getattr(params, "design_name", None)
    if callable(design_name_attr):
        return design_name_attr()
    return getattr(params, "base_design_name", "design")


def _write_initialization_profile(params, record):
    if not _profile_enabled(params):
        return None
    result_dir = getattr(params, "result_dir", None)
    if not result_dir:
        return None
    os.makedirs(result_dir, exist_ok=True)
    design_name = _design_name(params)
    latest_path = os.path.join(
        result_dir,
        f"{design_name}_initialization_profile_latest.json",
    )
    trace_path = os.path.join(
        result_dir,
        f"{design_name}_initialization_profile.jsonl",
    )
    payload = dict(record)
    payload["artifact"] = "initialization_profile_latest"
    payload["artifact_version"] = 1
    payload["design_name"] = design_name
    payload["timestamp"] = time.time()
    with open(latest_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
    trace_payload = dict(payload)
    trace_payload["artifact"] = "initialization_profile_record"
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(trace_payload, sort_keys=True) + "\n")
    return latest_path


def _write_effective_params_manifest(launch):
    manifest = launch.get("effective_params_manifest")
    if not isinstance(manifest, dict):
        return None
    targets = []
    workspace = launch.get("workspace")
    result_dir = launch.get("result_dir")
    if workspace:
        targets.append(("workspace", str(workspace)))
    if result_dir and str(result_dir) != str(workspace):
        targets.append(("result", str(result_dir)))
    if not targets:
        return None
    payload = dict(manifest)
    paths = {}
    for label, directory in targets:
        os.makedirs(directory, exist_ok=True)
        payload["run_context"] = {
            "workspace": str(launch.get("workspace") or ""),
            "result_dir": None
            if launch.get("result_dir") is None
            else str(launch["result_dir"]),
            "output_def": None
            if launch.get("output_def") is None
            else str(launch["output_def"]),
            "output_verilog": None
            if launch.get("output_verilog") is None
            else str(launch["output_verilog"]),
            "setup_margin": launch.get("setup_margin"),
            "manifest_scope": label,
        }
        path = os.path.join(directory, "effective_params.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")
        paths[label] = path
    return paths


def _is_failed_run_result(run_result, flow_kind=None):
    if not isinstance(run_result, dict):
        return False
    status = str(run_result.get("status") or "").lower()
    if status and status not in {"ok", "success"}:
        return True
    if flow_kind == FlowKind.STA and status == "ok":
        return False
    for key in ("hpwl", "density"):
        value = run_result.get(key)
        if value in (None, ""):
            continue
        try:
            if value == float("inf") or value != value:
                return True
        except TypeError:
            continue
    return False


def run_optimization_flow(params, launch, engine_cls):
    flow_started_at = time.perf_counter()
    flow_kind = resolved_flow_kind(params)
    logging.info("selected optimization flow=%s", flow_kind.value)
    profile_record = {
        "flow_kind": flow_kind.value,
        "workspace": launch.get("workspace"),
        "output_def": None if launch.get("output_def") is None else str(launch.get("output_def")),
        "output_verilog": None
        if launch.get("output_verilog") is None
        else str(launch.get("output_verilog")),
    }
    manifest_paths = _write_effective_params_manifest(launch)
    if manifest_paths:
        profile_record["effective_params_manifest"] = manifest_paths

    stage_started_at = time.perf_counter()
    engine = engine_cls(params)
    profile_record["engine_init_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
    # Keep launch workspace as the compatibility design handle. The flow layer
    # does not parse workspace internals; backend adapters decide whether the
    # workspace or params.design_inputs are authoritative for their database.
    stage_started_at = time.perf_counter()
    engine.setup_rawdb(
        data_manager=types.SimpleNamespace(dir_workspace=launch["workspace"])
    )
    profile_record["setup_rawdb_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
    stage_started_at = time.perf_counter()
    run_result = engine.run()
    profile_record["run_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
    profile_record["run_failed"] = _is_failed_run_result(run_result, flow_kind)
    if isinstance(run_result, dict):
        profile_record["run_result_status"] = run_result.get("status")
        profile_record["run_result_stop_reason"] = run_result.get("stop_reason")
    if launch.get("output_def") and flow_kind != FlowKind.STA:
        if profile_record["run_failed"]:
            logging.warning(
                "skip write_back to %s because optimization run failed",
                launch["output_def"],
            )
            profile_record["write_back_skipped_reason"] = "run_failed"
            profile_record["write_back_ms"] = 0.0
        else:
            stage_started_at = time.perf_counter()
            engine.write_back(str(launch["output_def"]))
            profile_record["write_back_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
    else:
        profile_record["write_back_ms"] = 0.0
    if launch.get("output_verilog") and flow_kind != FlowKind.STA:
        if profile_record["run_failed"]:
            logging.warning(
                "skip Verilog writeback to %s because optimization run failed",
                launch["output_verilog"],
            )
            profile_record["write_verilog_skipped_reason"] = "run_failed"
            profile_record["write_verilog_ms"] = 0.0
        else:
            stage_started_at = time.perf_counter()
            engine.write_verilog(str(launch["output_verilog"]))
            profile_record["write_verilog_ms"] = (
                time.perf_counter() - stage_started_at
            ) * 1000.0
    else:
        profile_record["write_verilog_ms"] = 0.0
    placer_profile = getattr(getattr(engine, "placer", None), "initialization_profile", None)
    if isinstance(placer_profile, dict):
        profile_record.update(placer_profile)
    rawdb_profile = getattr(getattr(engine, "placedb", None), "rawdb_initialization_profile", None)
    if isinstance(rawdb_profile, dict):
        profile_record.update(rawdb_profile)
    profile_record["total_flow_ms"] = (time.perf_counter() - flow_started_at) * 1000.0
    _write_initialization_profile(params, profile_record)

    return OptimizationFlowResult(flow_kind=flow_kind, engine=engine)
