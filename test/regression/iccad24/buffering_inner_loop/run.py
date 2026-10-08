#!/usr/bin/env python3
"""Run the canonical ICCAD24 buffering inner-loop smoke.

This regression validates the new ``Placer.py --flow-kind buffering`` path:
``NonLinearPlace`` builds relaxed buffer state, enters the buffering inner loop,
uses ``PlaceObj.timing_obj()``, runs virtual projection/runtime refresh, and
does not perform a physical OpenROAD commit by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ICCAD24_CASES = (
    "NV_NVDLA_partition_m",
    "NV_NVDLA_partition_p",
    "aes_256",
    "ariane136",
    "hidden1",
    "hidden2",
    "hidden3",
    "hidden4",
    "hidden5",
    "mempool_tile_wrap",
)

SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
PROFILE_PATH = SCRIPT_DIR / "params" / "candidate_discrete_net_gradient_nv_m.json"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OPENROAD = Path(
    os.environ.get("AUTODMP_OPENROAD_BIN", "/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "buffering_inner_loop"

CSV_COLUMNS = (
    "case",
    "status",
    "failure",
    "flow_kind",
    "buffering_mode",
    "candidate_strategy",
    "fixed_bsu_index",
    "continuous_steps",
    "continuous_lr",
    "max_selected_actions",
    "commit_enabled",
    "segment_integer_projection_interval",
    "segment_integer_projection_start_step",
    "segment_integer_projection_project_bsu",
    "segment_integer_projection_reset_optimizer_state",
    "run_id",
    "run_root",
    "result_dir",
    "log_path",
    "summary_path",
    "summary_status",
    "state_build_status",
    "state_kind",
    "candidate_count",
    "segment_count",
    "affected_net_count",
    "net_count",
    "projection_count",
    "periodic_integer_projection_count",
    "projected_buffer_count",
    "runtime_refresh_status",
    "runtime_refresh_projected_buffer_count",
    "runtime_refresh_affected_net_count",
    "loss_initial",
    "loss_final",
    "tns_initial",
    "tns_final",
    "wns_initial",
    "wns_final",
    "step_wall_ms_initial",
    "step_wall_ms_final",
    "step_wall_ms_mean",
    "step_wall_ms_max",
    "objective_forward_ms_mean",
    "objective_backward_ms_mean",
    "objective_step_wall_ms_mean",
    "projection_refresh_wall_ms_mean",
    "projection_refresh_wall_ms_max",
    "periodic_integer_projection_z_nonzero_count_final",
    "periodic_integer_projection_z_sum_final",
    "periodic_integer_projection_z_max_final",
    "grad_bu_initial_norm",
    "grad_bu_final_norm",
    "grad_z_initial_norm",
    "grad_z_final_norm",
    "grad_bsu_initial_norm",
    "grad_bsu_final_norm",
    "segment_count_timing_backend_requested",
    "segment_count_timing_backend_used",
    "segment_count_timing_backend_fallback_reason",
    "segment_transfer_backend_requested",
    "segment_transfer_backend_used",
    "commit_status",
    "commit_reason",
    "commit_action_count",
    "commit_attempted_action_count",
    "commit_accepted_action_count",
    "commit_rejected_action_count",
    "commit_failed_action_count",
    "commit_before_wns",
    "commit_before_tns",
    "commit_after_wns",
    "commit_after_tns",
    "commit_delta_wns",
    "commit_delta_tns",
    "commit_artifact_path",
    "profile_path",
    "profile_sha256",
    "trace_path",
    "trace_sha256",
    "trace_audit_status",
    "replay_status",
    "replay_before_wns_ns",
    "replay_before_tns_ns",
    "replay_after_wns_ns",
    "replay_after_tns_ns",
    "replay_delta_wns_ns",
    "replay_delta_tns_ns",
    "replay_buffer_count",
    "replay_runtime_sec",
    "qor_release_status",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_external_provenance_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("artifact") != "autodmp_tools_paper_campaign_manifest":
        raise ValueError("external provenance is not a tools-paper campaign manifest")
    execution = dict(payload.get("execution_manifest") or {})
    if execution.get("autodmp_revision") != _tool_output(
        ["git", "rev-parse", "HEAD"]
    ).splitlines()[0]:
        raise ValueError("external provenance AutoDMP revision mismatch")
    source_hashes = dict(execution.get("source_file_sha256") or {})
    relative_runner = str(Path(__file__).resolve().relative_to(AUTODMP_ROOT))
    if source_hashes.get(relative_runner) != _sha256(Path(__file__).resolve()):
        raise ValueError("external provenance candidate runner hash mismatch")
    if len(str(execution.get("autodmp_tracked_diff_sha256") or "")) != 64:
        raise ValueError("external provenance lacks tracked diff identity")
    return {
        "path": str(path),
        "campaign_id": payload.get("campaign_id"),
        "execution_manifest_digest": payload.get("execution_manifest_digest"),
        "tracked_diff_sha256": execution.get("autodmp_tracked_diff_sha256"),
    }


def _tool_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=str(AUTODMP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _load_candidate_profile(path: Path) -> dict[str, Any]:
    profile = _read_json(path)
    expected = {
        "profile_kind": "candidate_discrete_net_gradient_full_design",
        "flow_kind": "buffering",
        "buffering_mode": "candidate",
        "candidate_strategy": "discrete_net_gradient",
        "selection_fraction": 0.001,
        "affected_nets_per_action": 1000,
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"candidate Route-B profile requires {key}={value!r}")
    suite = list(profile.get("suite") or ())
    if not suite or len(suite) != len(set(suite)):
        raise ValueError("candidate Route-B profile requires a nonempty unique suite")
    invalid_cases = sorted(set(suite) - set(ICCAD24_CASES))
    if invalid_cases:
        raise ValueError(
            "candidate Route-B profile contains unsupported cases: "
            + ", ".join(invalid_cases)
        )
    if int(profile.get("rounds", 0)) != 5:
        raise ValueError("candidate Route-B profile requires five rounds")
    if int(profile.get("gpu_id", -1)) < 0:
        raise ValueError("candidate Route-B profile requires a nonnegative gpu_id")
    if int(profile.get("fixed_bsu_index", -1)) != 7:
        raise ValueError("candidate Route-B profile requires fixed bsu=7")
    if profile.get("buffer_master") != "BUFx4f_ASAP7_75t_R":
        raise ValueError("candidate Route-B profile requires BUFx4f_ASAP7_75t_R")
    generation = dict(profile.get("candidate_generation") or {})
    if generation != {
        "policy": "segment_only",
        "mode": "equal_count",
        "max_candidates_per_segment": 3,
        "include_tree_node_candidates": False,
        "full_design": True,
    }:
        raise ValueError("candidate Route-B profile has an invalid generation policy")
    if profile.get("candidate_net_subgraph_backend") not in {
        "native_explicit_autograd",
        "cuda_explicit_autograd",
    }:
        raise ValueError("candidate Route-B profile requires an explicit-autograd backend")
    if profile.get("timing_surrogate_mode") != "lut_only":
        raise ValueError("candidate Route-B profile requires lut_only timing")
    if not profile.get("one_final_coordinate_commit"):
        raise ValueError("candidate Route-B profile requires one final coordinate commit")
    return profile


def _write_profile_params(
    *,
    canonical_params_path: Path,
    output_path: Path,
    profile: dict[str, Any],
) -> Path:
    params = _read_json(canonical_params_path)
    generation = dict(profile["candidate_generation"])
    params.update(
        {
            "buffering_candidate_policy": generation["policy"],
            "buffering_max_repeaters_per_segment": generation[
                "max_candidates_per_segment"
            ],
            "buffering_include_tree_node_candidates": int(
                generation["include_tree_node_candidates"]
            ),
            "buffering_runtime_profile": 1,
            "timing_surrogate_mode": profile["timing_surrogate_mode"],
            "place_io_engine": "openroad",
        }
    )
    if profile["candidate_net_subgraph_backend"] == "cuda_explicit_autograd":
        params["timing_propagation_device"] = "cuda"
    _write_json(output_path, params)
    return output_path


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _metric_trace_endpoints(metrics_trace: Any) -> dict[str, float | None]:
    if not isinstance(metrics_trace, list) or not metrics_trace:
        return {
            "loss_initial": None,
            "loss_final": None,
            "tns_initial": None,
            "tns_final": None,
            "wns_initial": None,
            "wns_final": None,
            "step_wall_ms_initial": None,
            "step_wall_ms_final": None,
            "step_wall_ms_mean": None,
            "step_wall_ms_max": None,
            "objective_step_wall_ms_mean": None,
            "projection_refresh_wall_ms_mean": None,
            "projection_refresh_wall_ms_max": None,
            "periodic_integer_projection_z_nonzero_count_final": None,
            "periodic_integer_projection_z_sum_final": None,
            "periodic_integer_projection_z_max_final": None,
            "grad_bu_initial_norm": None,
            "grad_bu_final_norm": None,
        }
    first = metrics_trace[0] if isinstance(metrics_trace[0], dict) else {}
    last = metrics_trace[-1] if isinstance(metrics_trace[-1], dict) else {}
    step_wall_values = [
        float(item["step_wall_ms"])
        for item in metrics_trace
        if isinstance(item, dict)
        and item.get("step_wall_ms") is not None
    ]
    objective_step_values = [
        float(item["objective_step_wall_ms"])
        for item in metrics_trace
        if isinstance(item, dict)
        and item.get("objective_step_wall_ms") is not None
    ]
    objective_forward_values = [
        float(item["objective_forward_ms"])
        for item in metrics_trace
        if isinstance(item, dict)
        and item.get("objective_forward_ms") is not None
    ]
    objective_backward_values = [
        float(item["objective_backward_ms"])
        for item in metrics_trace
        if isinstance(item, dict)
        and item.get("objective_backward_ms") is not None
    ]
    projection_refresh_values = [
        float(item["projection_refresh_wall_ms"])
        for item in metrics_trace
        if isinstance(item, dict)
        and item.get("projection_refresh_wall_ms") is not None
    ]
    return {
        "loss_initial": _as_float(first.get("loss")),
        "loss_final": _as_float(last.get("loss")),
        "tns_initial": _as_float(first.get("tns")),
        "tns_final": _as_float(last.get("tns")),
        "wns_initial": _as_float(first.get("wns")),
        "wns_final": _as_float(last.get("wns")),
        "step_wall_ms_initial": _as_float(first.get("step_wall_ms")),
        "step_wall_ms_final": _as_float(last.get("step_wall_ms")),
        "step_wall_ms_mean": (
            sum(step_wall_values) / len(step_wall_values)
            if step_wall_values
            else None
        ),
        "step_wall_ms_max": max(step_wall_values) if step_wall_values else None,
        "objective_step_wall_ms_mean": (
            sum(objective_step_values) / len(objective_step_values)
            if objective_step_values
            else None
        ),
        "objective_forward_ms_mean": (
            sum(objective_forward_values) / len(objective_forward_values)
            if objective_forward_values
            else None
        ),
        "objective_backward_ms_mean": (
            sum(objective_backward_values) / len(objective_backward_values)
            if objective_backward_values
            else None
        ),
        "projection_refresh_wall_ms_mean": (
            sum(projection_refresh_values) / len(projection_refresh_values)
            if projection_refresh_values
            else None
        ),
        "projection_refresh_wall_ms_max": (
            max(projection_refresh_values)
            if projection_refresh_values
            else None
        ),
        "periodic_integer_projection_z_nonzero_count_final": _as_int(
            last.get("periodic_integer_projection_z_nonzero_count_after")
        ),
        "periodic_integer_projection_z_sum_final": _as_float(
            last.get("periodic_integer_projection_z_sum_after")
        ),
        "periodic_integer_projection_z_max_final": _as_float(
            last.get("periodic_integer_projection_z_max_after")
        ),
        "grad_bu_initial_norm": _as_float(first.get("grad_bu_norm")),
        "grad_bu_final_norm": _as_float(last.get("grad_bu_norm")),
        "grad_z_initial_norm": _as_float(first.get("grad_z_norm")),
        "grad_z_final_norm": _as_float(last.get("grad_z_norm")),
        "grad_bsu_initial_norm": _as_float(first.get("grad_bsu_norm")),
        "grad_bsu_final_norm": _as_float(last.get("grad_bsu_norm")),
    }


def resolve_cases(raw_cases: list[str] | None) -> list[str]:
    if not raw_cases:
        return ["NV_NVDLA_partition_m"]
    cases: list[str] = []
    for raw in raw_cases:
        for part in str(raw).split(","):
            case = part.strip()
            if case:
                cases.append(case)
    invalid = [case for case in cases if case not in ICCAD24_CASES]
    if invalid:
        raise SystemExit(
            "unsupported ICCAD24 case(s): "
            + ", ".join(invalid)
            + "; expected one of "
            + ", ".join(ICCAD24_CASES)
        )
    return cases


def asap7_inputs(benchmark_root: Path) -> dict[str, list[Path] | Path]:
    asap7 = benchmark_root / "ASAP7"
    return {
        "tech_lef": asap7 / "lef" / "asap7_tech_1x_201209.lef",
        "lef": [
            asap7 / "lef" / "asap7sc7p5t_27_R_1x_201211.lef",
            asap7 / "lef" / "sram_asap7_16x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_32x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_64x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_64x64_1rw.lef",
        ],
        "lib": [
            asap7 / "lib" / "asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
            asap7 / "lib" / "asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
            asap7 / "lib" / "asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
            asap7 / "lib" / "asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
            asap7 / "lib" / "asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
            asap7 / "lib" / "sram_asap7_16x256_1rw.lib",
            asap7 / "lib" / "sram_asap7_32x256_1rw.lib",
            asap7 / "lib" / "sram_asap7_64x256_1rw.lib",
            asap7 / "lib" / "sram_asap7_64x64_1rw.lib",
        ],
        "rc_tcl": asap7 / "setRC.tcl",
    }


def validate_case_inputs(benchmark_root: Path, case: str) -> dict[str, Path]:
    case_dir = benchmark_root / "design" / case
    workspace = case_dir / "workspace"
    paths = {
        "case_dir": case_dir,
        "workspace": workspace,
        "params_json": workspace / "config" / "dreamplace_config" / "param.json",
        "def": case_dir / f"{case}.def",
        "verilog": case_dir / f"{case}.v",
        "sdc": case_dir / f"{case}.sdc",
    }
    tech = asap7_inputs(benchmark_root)
    required = list(paths.values()) + [tech["tech_lef"], tech["rc_tcl"]]
    required.extend(tech["lef"])
    required.extend(tech["lib"])
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise SystemExit(f"{case}: missing required input(s): " + ", ".join(missing))
    return paths


def build_placer_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    buffering_mode: str,
    continuous_steps: int,
    continuous_lr: float | None,
    params_json: Path | None = None,
    candidate_strategy: str = "continuous",
    fixed_bsu_index: int | None = None,
    gpu_id: int | None = None,
    committed_def_path: Path | None = None,
    segment_count_timing_backend: str | None = None,
    segment_transfer_backend: str | None = None,
    segment_integer_projection_interval: int | None = None,
    segment_integer_projection_start_step: int | None = None,
    segment_integer_projection_project_bsu: int | None = None,
    segment_integer_projection_reset_optimizer_state: int | None = None,
    max_selected_actions: int | None = None,
    commit_enabled: bool = False,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    params_json = Path(params_json) if params_json is not None else paths["params_json"]
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(params_json),
        "--flow-kind",
        "buffering",
        "--place-io-engine",
        "openroad",
        "--buffering-mode",
        buffering_mode,
        "--workspace",
        str(paths["workspace"]),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--def-input",
        str(paths["def"]),
        "--verilog-input",
        str(paths["verilog"]),
        "--sdc",
        str(paths["sdc"]),
        "--rc-tcl",
        str(tech["rc_tcl"]),
        "--tech-lef",
        str(tech["tech_lef"]),
        "--buffering-output-dir",
        str(result_dir),
    ]
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])
    command.extend(
        [
            "--with-sta",
            "--iterations",
            "1",
            "--buffering-continuous-steps",
            str(int(continuous_steps)),
        ]
    )
    if buffering_mode == "candidate":
        command.extend(["--buffering-candidate-strategy", str(candidate_strategy)])
    if candidate_strategy == "continuous" and continuous_lr is not None:
        command.extend(["--buffering-continuous-lr", str(float(continuous_lr))])
    if fixed_bsu_index is not None:
        command.extend(["--buffering-fixed-bsu-index", str(int(fixed_bsu_index))])
    if gpu_id is not None:
        command.extend(["--gpu", "1", "--gpu-id", str(int(gpu_id))])
    if committed_def_path is not None:
        command.extend(
            ["--buffering-committed-def-path", str(committed_def_path)]
        )
    if segment_integer_projection_interval is not None:
        command.extend(
            [
                "--buffering-segment-integer-projection-interval",
                str(int(segment_integer_projection_interval)),
            ]
        )
    if segment_integer_projection_start_step is not None:
        command.extend(
            [
                "--buffering-segment-integer-projection-start-step",
                str(int(segment_integer_projection_start_step)),
            ]
        )
    if segment_integer_projection_project_bsu is not None:
        command.extend(
            [
                "--buffering-segment-integer-projection-project-bsu",
                str(int(segment_integer_projection_project_bsu)),
            ]
        )
    if segment_integer_projection_reset_optimizer_state is not None:
        command.extend(
            [
                "--buffering-segment-integer-projection-reset-optimizer-state",
                str(int(segment_integer_projection_reset_optimizer_state)),
            ]
        )
    if max_selected_actions is not None:
        command.extend(["--buffering-max-selected-actions", str(int(max_selected_actions))])
    if segment_count_timing_backend:
        command.extend(
            [
                "--buffering-segment-count-timing-backend",
                str(segment_count_timing_backend),
            ]
        )
    if segment_transfer_backend:
        command.extend(
            [
                "--buffering-segment-transfer-backend",
                str(segment_transfer_backend),
            ]
        )
    command.extend(["--buffering-commit-enabled", "1" if commit_enabled else "0"])
    return command


def find_inner_loop_summary(result_dir: Path, case: str) -> Path | None:
    preferred = result_dir / f"{case}_buffering_inner_loop_summary.json"
    if preferred.exists():
        return preferred
    matches = sorted(result_dir.rglob("*_buffering_inner_loop_summary.json"))
    return matches[-1] if matches else None


def summarize_inner_loop_result(
    *,
    case: str,
    run_id: str,
    run_root: Path,
    result_dir: Path,
    log_path: Path,
    command_returncode: int,
    buffering_mode: str,
    continuous_steps: int,
    continuous_lr: float | None,
    max_selected_actions: int | None,
    commit_enabled: bool,
    segment_integer_projection_interval: int | None = None,
    segment_integer_projection_start_step: int | None = None,
    segment_integer_projection_project_bsu: int | None = None,
    segment_integer_projection_reset_optimizer_state: int | None = None,
    summary_path: Path | None = None,
    candidate_strategy: str = "continuous",
    fixed_bsu_index: int | None = None,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": case,
        "flow_kind": "buffering",
        "buffering_mode": buffering_mode,
        "candidate_strategy": candidate_strategy,
        "fixed_bsu_index": fixed_bsu_index,
        "continuous_steps": int(continuous_steps),
        "continuous_lr": (
            None if continuous_lr is None else float(continuous_lr)
        ),
        "max_selected_actions": max_selected_actions,
        "commit_enabled": int(bool(commit_enabled)),
        "segment_integer_projection_interval": _optional_int(
            segment_integer_projection_interval
        ),
        "segment_integer_projection_start_step": _optional_int(
            segment_integer_projection_start_step
        ),
        "segment_integer_projection_project_bsu": _optional_int(
            segment_integer_projection_project_bsu
        ),
        "segment_integer_projection_reset_optimizer_state": _optional_int(
            segment_integer_projection_reset_optimizer_state
        ),
        "run_id": run_id,
        "run_root": str(run_root),
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "summary_path": str(summary_path or ""),
        "profile_path": str(profile_path or ""),
        "profile_sha256": (
            _sha256(profile_path) if profile_path is not None and profile_path.is_file() else ""
        ),
    }
    if command_returncode != 0:
        row.update(
            {
                "status": "failed",
                "failure": f"Placer.py exited with code {command_returncode}",
            }
        )
        return row
    if summary_path is None:
        row.update(
            {
                "status": "failed",
                "failure": "missing buffering inner-loop summary artifact",
            }
        )
        return row

    payload = _read_json(summary_path)
    summary = dict(payload.get("summary", {}) or {})
    runtime_refresh = dict(summary.get("runtime_refresh", {}) or {})
    inner_loop = dict(summary.get("inner_loop", {}) or {})
    commit = dict(summary.get("commit", {}) or {})
    metric_endpoints = _metric_trace_endpoints(inner_loop.get("metrics_trace"))
    row.update(
        {
            "summary_status": summary.get("status"),
            "state_build_status": summary.get("state_build_status"),
            "state_kind": summary.get("state_kind"),
            "candidate_count": _as_int(summary.get("candidate_count")),
            "segment_count": _as_int(summary.get("segment_count")),
            "affected_net_count": _as_int(summary.get("affected_net_count")),
            "net_count": _as_int(summary.get("net_count")),
            "projection_count": _as_int(inner_loop.get("projection_count")),
            "periodic_integer_projection_count": _as_int(
                inner_loop.get("periodic_integer_projection_count")
            ),
            "projected_buffer_count": _as_int(
                summary.get("last_projected_buffer_count")
            ),
            "runtime_refresh_status": runtime_refresh.get("status"),
            "runtime_refresh_projected_buffer_count": _as_int(
                runtime_refresh.get("projected_buffer_count")
            ),
            "runtime_refresh_affected_net_count": _as_int(
                runtime_refresh.get("affected_net_count")
            ),
            **metric_endpoints,
            "segment_count_timing_backend_requested": summary.get(
                "segment_count_timing_backend_requested"
            ),
            "segment_count_timing_backend_used": summary.get(
                "segment_count_timing_backend_used"
            ),
            "segment_count_timing_backend_fallback_reason": summary.get(
                "segment_count_timing_backend_fallback_reason"
            ),
            "segment_transfer_backend_requested": summary.get(
                "segment_transfer_backend_requested"
            ),
            "segment_transfer_backend_used": summary.get(
                "segment_transfer_backend_used",
                summary.get("segment_transfer_backend"),
            ),
            "commit_status": commit.get("status"),
            "commit_reason": commit.get("reason"),
            "commit_action_count": _as_int(commit.get("action_count")),
            "commit_attempted_action_count": _as_int(
                commit.get("attempted_action_count")
            ),
            "commit_accepted_action_count": _as_int(
                commit.get("accepted_action_count")
            ),
            "commit_rejected_action_count": _as_int(
                commit.get("rejected_action_count")
            ),
            "commit_failed_action_count": _as_int(commit.get("failed_action_count")),
            "commit_before_wns": _as_float(commit.get("before_wns")),
            "commit_before_tns": _as_float(commit.get("before_tns")),
            "commit_after_wns": _as_float(commit.get("after_wns")),
            "commit_after_tns": _as_float(commit.get("after_tns")),
            "commit_delta_wns": _as_float(commit.get("delta_wns")),
            "commit_delta_tns": _as_float(commit.get("delta_tns")),
            "commit_artifact_path": commit.get("artifact_path"),
            "trace_path": summary.get("trace_path"),
            "trace_sha256": summary.get("trace_sha256"),
        }
    )
    failures = []
    if summary.get("status") != "completed":
        failures.append(f"summary status is {summary.get('status')!r}")
    if summary.get("state_build_status") != "ok":
        failures.append(f"state_build_status is {summary.get('state_build_status')!r}")
    if buffering_mode == "segment":
        if summary.get("state_kind") != "segment_count":
            failures.append(f"state_kind is {summary.get('state_kind')!r}")
        if not row["segment_count"] or row["segment_count"] <= 0:
            failures.append("segment_count is not positive")
    elif not row["candidate_count"] or row["candidate_count"] <= 0:
        failures.append("candidate_count is not positive")
    if runtime_refresh.get("status") != "ok":
        failures.append(f"runtime_refresh status is {runtime_refresh.get('status')!r}")
    if commit_enabled:
        if commit.get("status") not in {
            "accepted",
            "rejected",
            "failed",
            "blocked",
            "skipped",
        }:
            failures.append(f"commit status is {commit.get('status')!r}; expected final commit result")
        has_metrics = (
            row.get("commit_before_wns") is not None
            and row.get("commit_after_wns") is not None
            and row.get("commit_before_tns") is not None
            and row.get("commit_after_tns") is not None
        )
        has_reason = bool(
            commit.get("reason")
            or commit.get("reject_reason")
            or commit.get("error")
            or commit.get("backend_status")
        )
        if not has_metrics and not has_reason:
            failures.append("commit-enabled summary has neither OpenSTA metrics nor structured reason")
    elif commit.get("status") != "disabled":
        failures.append(f"commit status is {commit.get('status')!r}; expected disabled")
    row["status"] = "pass" if not failures else "failed"
    row["failure"] = "; ".join(failures)
    return row


def audit_candidate_route_b_trace(
    *,
    summary_path: Path,
    profile: dict[str, Any],
    commit_enabled: bool,
    logical_gpu_id: int | None = None,
) -> dict[str, Any]:
    summary = dict(_read_json(summary_path).get("summary", {}) or {})
    trace_path = Path(str(summary.get("trace_path") or ""))
    failures: list[str] = []
    if summary.get("mode") != "candidate":
        failures.append("summary_mode_mismatch")
    if summary.get("strategy") != "discrete_net_gradient":
        failures.append("summary_strategy_mismatch")
    if summary.get("state_kind") != "candidate":
        failures.append("summary_state_kind_mismatch")
    if not trace_path.is_file():
        failures.append("missing_candidate_scheduler_trace")
        return {
            "status": "failed",
            "failures": failures,
            "trace_path": str(trace_path),
        }

    trace_bytes = trace_path.read_bytes()
    trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    if trace_sha256 != str(summary.get("trace_sha256") or ""):
        failures.append("summary_trace_sha256_mismatch")
    trace = json.loads(trace_bytes)
    if trace.get("artifact") != "candidate_discrete_scheduler_trace":
        failures.append("trace_artifact_mismatch")
    candidate_count = int(trace.get("candidate_count", 0) or 0)
    affected_net_count = int(trace.get("affected_net_count", 0) or 0)
    if candidate_count <= 0 or affected_net_count <= 0:
        failures.append("empty_full_design_candidate_scope")
    if int(trace.get("fixed_bsu_index", -1)) != int(profile["fixed_bsu_index"]):
        failures.append("fixed_bsu_mismatch")
    if not math.isclose(
        float(trace.get("selection_fraction", -1.0)),
        float(profile["selection_fraction"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failures.append("selection_fraction_mismatch")
    if int(trace.get("affected_nets_per_action", 0) or 0) != int(
        profile["affected_nets_per_action"]
    ):
        failures.append("affected_nets_per_action_mismatch")
    if int(trace.get("rounds_requested", 0) or 0) != int(profile["rounds"]):
        failures.append("round_count_mismatch")

    rounds = list(trace.get("rounds", ()) or ())
    if not rounds:
        failures.append("missing_scheduler_round")
    expected_prefix = (
        int(math.ceil(affected_net_count / float(profile["affected_nets_per_action"])))
        if affected_net_count > 0
        else 0
    )
    previous_after = 0
    for index, record in enumerate(rounds):
        if int(record.get("prefix_size", -1)) != expected_prefix:
            failures.append(f"round_{index}_prefix_mismatch")
        if not bool(record.get("activation_binary_before")):
            failures.append(f"round_{index}_nonbinary_input")
        if not bool(record.get("activation_binary_after")):
            failures.append(f"round_{index}_nonbinary_output")
        before = int(record.get("active_candidate_count_before", -1))
        after = int(record.get("active_candidate_count_after", -1))
        if index == 0 and before != 0:
            failures.append("initial_activation_not_exact_zero")
        if before != previous_after or after < before:
            failures.append(f"round_{index}_active_count_not_monotonic")
        previous_after = after
        if int(record.get("raw_grad_nonfinite_count", -1)) != 0:
            failures.append(f"round_{index}_nonfinite_gradient")
        if int(record.get("raw_grad_finite_count", -1)) != candidate_count:
            failures.append(f"round_{index}_finite_gradient_count_mismatch")
        accepted = int(record.get("accepted_action_count", 0) or 0)
        if accepted > expected_prefix:
            failures.append(f"round_{index}_accepted_count_exceeds_prefix")
        if len(record.get("selected_rows", ()) or ()) != accepted:
            failures.append(f"round_{index}_selected_identity_count_mismatch")
    if rounds and int(rounds[0].get("raw_grad_nonzero_count", 0) or 0) <= 0:
        failures.append("first_round_gradient_all_zero")

    projection = dict(trace.get("projection", {}) or {})
    if projection.get("status") != "ok" or not projection.get(
        "exact_active_candidate_match"
    ):
        failures.append("exact_projection_failed")
    if int(projection.get("active_candidate_count", -1)) != previous_after:
        failures.append("projection_active_count_mismatch")
    runtime = dict(trace.get("runtime", {}) or {})
    for key in (
        "cold_forward_ms",
        "gradient_ms",
        "scheduler_ms",
        "terminal_forward_ms",
        "projection_materialization_ms",
        "runtime_refresh_ms",
    ):
        if runtime.get(key) is None:
            failures.append(f"missing_runtime_{key}")

    provenance = dict(trace.get("backend_provenance", {}) or {})
    expected_generation = dict(profile["candidate_generation"])
    expected_provenance = {
        "net_subgraph_forward_backend": profile["candidate_net_subgraph_backend"],
        "timing_surrogate_mode": profile["timing_surrogate_mode"],
        "fixed_bsu_index": int(profile["fixed_bsu_index"]),
        "fixed_buffer_master_name": profile["buffer_master"],
        "candidate_policy": expected_generation["policy"],
        "candidate_generation_mode": expected_generation["mode"],
        "max_candidates_per_segment": expected_generation[
            "max_candidates_per_segment"
        ],
        "include_tree_node_candidates": expected_generation[
            "include_tree_node_candidates"
        ],
        "full_design_scope": True,
        "physical_commit_owner": "PlacementEngine",
        "physical_commit_backend": "openroad",
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            failures.append(f"backend_provenance_{key}_mismatch")
    generation_count_fields = (
        "candidate_bearing_net_count",
        "supported_tree_net_count",
        "total_tree_edge_count",
        "coordinate_supported_edge_count",
        "nonzero_tree_edge_count",
        "edge_with_candidate_count",
        "edge_without_candidate_count",
        "candidate_attempt_count",
        "rounded_endpoint_skip_count",
        "coordinate_dedup_count",
    )
    if any(provenance.get(key) is None for key in generation_count_fields):
        failures.append("candidate_generation_counts_missing")
    else:
        nonzero_edge_count = int(provenance["nonzero_tree_edge_count"])
        edge_with_candidate_count = int(provenance["edge_with_candidate_count"])
        edge_without_candidate_count = int(provenance["edge_without_candidate_count"])
        if edge_with_candidate_count + edge_without_candidate_count != nonzero_edge_count:
            failures.append("candidate_edge_accounting_mismatch")
        generated_candidate_count = (
            int(provenance["candidate_attempt_count"])
            - int(provenance["rounded_endpoint_skip_count"])
            - int(provenance["coordinate_dedup_count"])
        )
        if generated_candidate_count != int(provenance.get("candidate_count", -1)):
            failures.append("candidate_coordinate_accounting_mismatch")
        if int(provenance["candidate_bearing_net_count"]) != int(
            provenance["supported_tree_net_count"]
        ):
            failures.append("candidate_net_coverage_incomplete")
    expected_device = f"cuda:{int(profile['gpu_id'] if logical_gpu_id is None else logical_gpu_id)}"
    if not str(provenance.get("state_device") or "").startswith(expected_device):
        failures.append("state_device_mismatch")

    runtime_profile = dict(summary.get("runtime_profile", {}) or {})
    if not runtime_profile.get("enabled"):
        failures.append("missing_state_preparation_runtime_profile")
    if "lane_prepare_ms" not in runtime_profile:
        failures.append("missing_lane_prepare_runtime")

    commit = dict(summary.get("commit", {}) or {})
    trace_commit = dict(trace.get("commit", {}) or {})
    if commit_enabled:
        attempted = int(commit.get("attempted_action_count", 0) or 0)
        accepted = int(commit.get("accepted_action_count", 0) or 0)
        rejected = int(commit.get("rejected_action_count", 0) or 0)
        failed = int(commit.get("failed_action_count", 0) or 0)
        skipped = int(commit.get("skipped_action_count", 0) or 0)
        if commit.get("status") != "accepted" or accepted <= 0:
            failures.append("final_commit_has_no_accepted_action")
        if failed != 0 or skipped != 0 or attempted != accepted + rejected:
            failures.append("commit_action_accounting_mismatch")
        if trace_commit.get("status") != commit.get("status"):
            failures.append("post_commit_trace_not_updated")
    else:
        if commit.get("status") != "disabled":
            failures.append("no_commit_run_performed_physical_mutation")
        if trace_commit.get("commit_enabled"):
            failures.append("no_commit_trace_marks_commit_enabled")

    return {
        "artifact": "candidate_route_b_trace_audit",
        "artifact_version": 1,
        "status": "pass" if not failures else "failed",
        "failures": failures,
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha256,
        "candidate_count": candidate_count,
        "affected_net_count": affected_net_count,
        "expected_prefix_size": expected_prefix,
        "rounds_executed": len(rounds),
        "final_active_candidate_count": previous_after,
        "commit_enabled": bool(commit_enabled),
        "commit": commit,
        "backend_provenance": provenance,
    }


def _load_equal_spaced_replay_runner():
    path = SCRIPT_DIR.parent / "equal_spaced_buffering" / "run.py"
    spec = importlib.util.spec_from_file_location(
        "candidate_route_b_equal_spaced_replay",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_candidate_def_only_replay(
    *,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    committed_def: Path,
    output_dir: Path,
    profile: dict[str, Any],
    expected_buffer_count: int,
    openroad_num_threads: int | None = None,
) -> dict[str, Any]:
    replay = _load_equal_spaced_replay_runner()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    failures = []
    for name, def_input, method in (
        ("before", input_def, "b0"),
        ("after", committed_def, "ours_replay"),
    ):
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_def = run_dir / f"{case}_{name}.def"
        tcl_path = run_dir / "run.tcl"
        log_path = run_dir / "run.log"
        power_path = run_dir / "power.rpt"
        tcl_path.write_text(
            replay.generate_openroad_tcl(
                benchmark_root=benchmark_root,
                case=case,
                def_input=def_input,
                output_def=output_def,
                power_report=power_path,
                method=method,
                buffer_master=profile["buffer_master"],
                b1_max_buffer_percent=3.0,
            ),
            encoding="utf-8",
        )
        openroad_command = [str(openroad_bin)]
        if openroad_num_threads is not None:
            openroad_command.extend(
                ("-threads", str(int(openroad_num_threads)))
            )
        openroad_command.extend(("-exit", str(tcl_path)))
        execution = replay._run_command(
            openroad_command,
            cwd=run_dir,
            log_path=log_path,
        )
        metrics = replay._parse_metrics(log_path)
        if execution.get("status") != "ok":
            failures.append(f"{name}_openroad_exit_{execution.get('returncode')}")
        if not output_def.is_file():
            failures.append(f"{name}_missing_output_def")
        runs[name] = {
            "input_def": str(def_input),
            "input_def_sha256": _sha256(def_input),
            "output_def": str(output_def),
            "output_def_sha256": _sha256(output_def) if output_def.is_file() else "",
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
            "execution": execution,
            "metrics": metrics,
        }

    action_audit = None
    after_def = Path(runs["after"]["output_def"])
    if after_def.is_file():
        action_audit = replay.audit_def_mutation(
            input_def=input_def,
            final_def=after_def,
            expected_buffer_master=profile["buffer_master"],
            require_new_buffers=True,
        )
        _write_json(output_dir / "action_audit.json", action_audit)
        if action_audit.get("status") != "pass":
            failures.extend(action_audit.get("failures", ()) or ())
        if int(action_audit.get("added_buffer_count", -1)) != int(
            expected_buffer_count
        ):
            failures.append("replay_buffer_count_mismatch")

    before_metrics = dict(runs["before"].get("metrics", {}) or {})
    after_metrics = dict(runs["after"].get("metrics", {}) or {})
    before_wns = _as_float(before_metrics.get("wns_ns"))
    before_tns = _as_float(before_metrics.get("tns_ns"))
    after_wns = _as_float(after_metrics.get("wns_ns"))
    after_tns = _as_float(after_metrics.get("tns_ns"))
    if None in (before_wns, before_tns, after_wns, after_tns):
        failures.append("missing_def_only_timing_metric")
    delta_wns = None if before_wns is None or after_wns is None else after_wns - before_wns
    delta_tns = None if before_tns is None or after_tns is None else after_tns - before_tns
    implementation_status = "pass" if not failures else "failed"
    qor_status = (
        "pass"
        if implementation_status == "pass" and delta_tns is not None and delta_tns > 0.0
        else "blocked_for_qor"
        if implementation_status == "pass"
        else "not_evaluated"
    )
    runtime_sec = sum(
        float(dict(runs[name].get("execution", {}) or {}).get("runtime_sec", 0.0) or 0.0)
        for name in ("before", "after")
    )
    result = {
        "artifact": "candidate_route_b_def_only_replay",
        "artifact_version": 1,
        "status": implementation_status,
        "qor_release_status": qor_status,
        "failures": failures,
        "openroad_bin": str(openroad_bin),
        "openroad_version": _tool_output([str(openroad_bin), "-version"]),
        "protocol": list(profile["post_commit_protocol"]),
        "before_wns_ns": before_wns,
        "before_tns_ns": before_tns,
        "after_wns_ns": after_wns,
        "after_tns_ns": after_tns,
        "delta_wns_ns": delta_wns,
        "delta_tns_ns": delta_tns,
        "buffer_count": (
            None if action_audit is None else action_audit.get("added_buffer_count")
        ),
        "runtime_sec": runtime_sec,
        "action_audit": action_audit,
        "runs": runs,
    }
    _write_json(output_dir / "def_only_replay.json", result)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    with log_path.open("w", encoding="utf-8", errors="ignore") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        help=f"Frozen Candidate Route-B profile; canonical profile: {PROFILE_PATH}",
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="ICCAD24 case name. Can be passed multiple times or comma-separated.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)),
    )
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--openroad-num-threads", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--logical-gpu-id", type=int)
    parser.add_argument("--buffering-mode", choices=("candidate", "segment"), default="segment")
    parser.add_argument(
        "--buffering-candidate-strategy",
        choices=("continuous", "discrete_net_gradient"),
        default="continuous",
    )
    parser.add_argument("--buffering-fixed-bsu-index", type=int)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--buffering-continuous-steps", type=int, default=120)
    parser.add_argument("--buffering-continuous-lr", type=float, default=0.01)
    parser.add_argument("--buffering-max-selected-actions", type=int)
    parser.add_argument("--buffering-segment-count-timing-backend")
    parser.add_argument(
        "--buffering-segment-transfer-backend",
        choices=("static_size_table", "segment_transfer_python", "segment_transfer_native"),
    )
    parser.add_argument("--buffering-segment-integer-projection-interval", type=int)
    parser.add_argument("--buffering-segment-integer-projection-start-step", type=int)
    parser.add_argument("--buffering-segment-integer-projection-project-bsu", type=int, choices=(0, 1))
    parser.add_argument("--buffering-segment-integer-projection-reset-optimizer-state", type=int, choices=(0, 1))
    parser.add_argument("--buffering-commit-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write command files without running Placer.py or enforcing the smoke gate.",
    )
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--external-provenance-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(raw_argv)
    if args.openroad_num_threads is not None and args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    benchmark_root = args.benchmark_root.resolve()
    profile_path = args.profile.resolve() if args.profile is not None else None
    profile = _load_candidate_profile(profile_path) if profile_path is not None else None
    if profile is not None:
        conflicting_flags = {
            "--buffering-mode",
            "--buffering-candidate-strategy",
            "--buffering-fixed-bsu-index",
            "--gpu-id",
            "--buffering-continuous-steps",
            "--buffering-continuous-lr",
            "--buffering-max-selected-actions",
            "--buffering-segment-count-timing-backend",
            "--buffering-segment-transfer-backend",
            "--buffering-segment-integer-projection-interval",
            "--buffering-segment-integer-projection-start-step",
            "--buffering-segment-integer-projection-project-bsu",
            "--buffering-segment-integer-projection-reset-optimizer-state",
        }
        requested_conflicts = sorted(set(raw_argv) & conflicting_flags)
        if requested_conflicts:
            raise ValueError(
                "--profile owns algorithm controls; remove: "
                + ", ".join(requested_conflicts)
            )
        cases = resolve_cases(args.cases or list(profile["suite"]))
        invalid_cases = sorted(set(cases) - set(profile["suite"]))
        if invalid_cases:
            raise ValueError("case is outside the frozen profile: " + ", ".join(invalid_cases))
        buffering_mode = str(profile["buffering_mode"])
        candidate_strategy = str(profile["candidate_strategy"])
        continuous_steps = int(profile["rounds"])
        continuous_lr = None
        fixed_bsu_index = int(profile["fixed_bsu_index"])
        gpu_id = (
            int(args.logical_gpu_id)
            if args.logical_gpu_id is not None
            else int(profile["gpu_id"])
        )
        max_selected_actions = None
    else:
        cases = resolve_cases(args.cases)
        buffering_mode = args.buffering_mode
        candidate_strategy = args.buffering_candidate_strategy
        continuous_steps = args.buffering_continuous_steps
        continuous_lr = args.buffering_continuous_lr
        fixed_bsu_index = args.buffering_fixed_bsu_index
        gpu_id = args.gpu_id
        max_selected_actions = args.buffering_max_selected_actions
    if gpu_id is None or int(gpu_id) < 0:
        raise ValueError("logical GPU ID must be nonnegative")
    execution_environment = (
        None
        if args.cuda_visible_devices is None
        else {"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)}
    )
    run_root = args.output_root.resolve() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    tracked_worktree_status = (
        ""
        if args.plan_only
        else _tool_output(
            ["git", "status", "--porcelain", "--untracked-files=no"]
        )
    )
    external_provenance = (
        _validate_external_provenance_manifest(
            args.external_provenance_manifest.resolve()
        )
        if args.external_provenance_manifest is not None
        else None
    )
    if not args.plan_only:
        if not args.python_bin.is_file():
            raise FileNotFoundError(f"missing Python binary: {args.python_bin}")
        if profile is not None and not args.openroad_bin.is_file():
            raise FileNotFoundError(f"missing OpenROAD binary: {args.openroad_bin}")
        if (
            profile is not None
            and tracked_worktree_status
            and external_provenance is None
        ):
            raise RuntimeError(
                "frozen Candidate Route-B evidence requires a tracked-clean AutoDMP "
                "worktree; commit or restore tracked changes first"
            )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "artifact": "buffering_inner_loop_regression_manifest",
        "artifact_version": 2,
        "benchmark_root": str(benchmark_root),
        "run_root": str(run_root),
        "run_id": args.run_id,
        "cases": cases,
        "profile_path": str(profile_path or ""),
        "profile_sha256": (
            _sha256(profile_path) if profile_path is not None else ""
        ),
        "profile": profile,
        "autodmp_root": str(AUTODMP_ROOT),
        "autodmp_commit": (
            "" if args.plan_only else _tool_output(["git", "rev-parse", "HEAD"])
        ),
        "autodmp_tracked_worktree_status": tracked_worktree_status,
        "autodmp_tracked_worktree_clean": not bool(tracked_worktree_status),
        "python_bin": str(args.python_bin),
        "openroad_bin": str(args.openroad_bin),
        "openroad_num_threads": args.openroad_num_threads,
        "tool_versions": (
            {}
            if args.plan_only
            else {
                "python": _tool_output([str(args.python_bin), "--version"]),
                "python_cuda": _tool_output(
                    [
                        str(args.python_bin),
                        "-c",
                        (
                            "import json, torch; print(json.dumps({"
                            "'python_torch': torch.__version__, "
                            "'cuda_runtime': torch.version.cuda, "
                            "'cuda_available': torch.cuda.is_available(), "
                            "'device_count': torch.cuda.device_count(), "
                            f"'device_name': torch.cuda.get_device_name({int(gpu_id)}) "
                            f"if torch.cuda.device_count() > {int(gpu_id)} else None}}))"
                        ),
                    ]
                ),
                "openroad": _tool_output([str(args.openroad_bin), "-version"]),
            }
        ),
        "commit_enabled": bool(args.buffering_commit_enabled),
        "def_only_replay": bool(
            profile is not None
            and args.buffering_commit_enabled
            and not args.source_only
        ),
        "cuda_visible_devices": args.cuda_visible_devices,
        "logical_gpu_id": int(gpu_id),
        "source_only": bool(args.source_only),
        "external_provenance": external_provenance,
        "case_results": {},
    }
    for case in cases:
        case_root = run_root / case
        result_dir = case_root / "result"
        log_path = case_root / "run.log"
        inputs = validate_case_inputs(benchmark_root, case)
        params_json = inputs["params_json"]
        committed_def_path = None
        if profile is not None:
            params_json = _write_profile_params(
                canonical_params_path=inputs["params_json"],
                output_path=case_root / "candidate_route_b_params.json",
                profile=profile,
            )
            if args.buffering_commit_enabled:
                committed_def_path = case_root / f"{case}_candidate_committed.def"
            details.setdefault("input_provenance", {})[case] = {
                "input_def": str(inputs["def"]),
                "input_def_sha256": _sha256(inputs["def"]),
                "sdc": str(inputs["sdc"]),
                "sdc_sha256": _sha256(inputs["sdc"]),
                "canonical_params": str(inputs["params_json"]),
                "canonical_params_sha256": _sha256(inputs["params_json"]),
                "run_params": str(params_json),
                "run_params_sha256": _sha256(params_json),
            }
        command = build_placer_command(
            python_bin=args.python_bin,
            benchmark_root=benchmark_root,
            case=case,
            result_dir=result_dir,
            buffering_mode=buffering_mode,
            continuous_steps=continuous_steps,
            continuous_lr=continuous_lr,
            params_json=params_json,
            candidate_strategy=candidate_strategy,
            fixed_bsu_index=fixed_bsu_index,
            gpu_id=gpu_id,
            committed_def_path=committed_def_path,
            segment_count_timing_backend=args.buffering_segment_count_timing_backend,
            segment_transfer_backend=args.buffering_segment_transfer_backend,
            segment_integer_projection_interval=(
                args.buffering_segment_integer_projection_interval
            ),
            segment_integer_projection_start_step=(
                args.buffering_segment_integer_projection_start_step
            ),
            segment_integer_projection_project_bsu=(
                args.buffering_segment_integer_projection_project_bsu
            ),
            segment_integer_projection_reset_optimizer_state=(
                args.buffering_segment_integer_projection_reset_optimizer_state
            ),
            max_selected_actions=max_selected_actions,
            commit_enabled=bool(args.buffering_commit_enabled),
        )
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "command.txt").write_text(
            shlex.join(command) + "\n",
            encoding="utf-8",
        )
        if args.plan_only:
            row = {
                "case": case,
                "status": "plan_only",
                "failure": "",
                "flow_kind": "buffering",
                "buffering_mode": buffering_mode,
                "candidate_strategy": candidate_strategy,
                "fixed_bsu_index": fixed_bsu_index,
                "continuous_steps": continuous_steps,
                "continuous_lr": continuous_lr,
                "max_selected_actions": max_selected_actions,
                "commit_enabled": int(bool(args.buffering_commit_enabled)),
                "segment_integer_projection_interval": _optional_int(
                    args.buffering_segment_integer_projection_interval
                ),
                "segment_integer_projection_start_step": _optional_int(
                    args.buffering_segment_integer_projection_start_step
                ),
                "segment_integer_projection_project_bsu": _optional_int(
                    args.buffering_segment_integer_projection_project_bsu
                ),
                "segment_integer_projection_reset_optimizer_state": _optional_int(
                    args.buffering_segment_integer_projection_reset_optimizer_state
                ),
                "run_id": args.run_id,
                "run_root": str(run_root),
                "result_dir": str(result_dir),
                "log_path": str(log_path),
                "summary_path": "",
            }
        else:
            execution_started_at = time.perf_counter()
            returncode = run_command(
                command,
                cwd=REPO_ROOT,
                log_path=log_path,
                environment=execution_environment,
            )
            execution_wall_sec = time.perf_counter() - execution_started_at
            summary_path = find_inner_loop_summary(result_dir, case)
            row = summarize_inner_loop_result(
                case=case,
                run_id=args.run_id,
                run_root=run_root,
                result_dir=result_dir,
                log_path=log_path,
                command_returncode=returncode,
                buffering_mode=buffering_mode,
                candidate_strategy=candidate_strategy,
                fixed_bsu_index=fixed_bsu_index,
                continuous_steps=continuous_steps,
                continuous_lr=continuous_lr,
                max_selected_actions=max_selected_actions,
                commit_enabled=bool(args.buffering_commit_enabled),
                segment_integer_projection_interval=(
                    args.buffering_segment_integer_projection_interval
                ),
                segment_integer_projection_start_step=(
                    args.buffering_segment_integer_projection_start_step
                ),
                segment_integer_projection_project_bsu=(
                    args.buffering_segment_integer_projection_project_bsu
                ),
                segment_integer_projection_reset_optimizer_state=(
                    args.buffering_segment_integer_projection_reset_optimizer_state
                ),
                summary_path=summary_path,
                profile_path=profile_path,
            )
            row["autodmp_runtime_sec"] = execution_wall_sec
            trace_audit = None
            replay_result = None
            if profile is not None and summary_path is not None:
                trace_audit = audit_candidate_route_b_trace(
                    summary_path=summary_path,
                    profile=profile,
                    commit_enabled=bool(args.buffering_commit_enabled),
                    logical_gpu_id=int(gpu_id),
                )
                _write_json(case_root / "candidate_route_b_trace_audit.json", trace_audit)
                row["trace_audit_status"] = trace_audit["status"]
                row["trace_path"] = trace_audit.get("trace_path")
                row["trace_sha256"] = trace_audit.get("trace_sha256")
                if trace_audit["status"] != "pass":
                    row["status"] = "failed"
                    row["failure"] = "; ".join(
                        value
                        for value in (
                            row.get("failure"),
                            *trace_audit.get("failures", ()),
                        )
                        if value
                    )
            if (
                profile is not None
                and args.buffering_commit_enabled
                and row["status"] == "pass"
                and committed_def_path is not None
                and not args.source_only
            ):
                replay_result = run_candidate_def_only_replay(
                    openroad_bin=args.openroad_bin.resolve(),
                    benchmark_root=benchmark_root,
                    case=case,
                    input_def=inputs["def"],
                    committed_def=committed_def_path,
                    output_dir=case_root / "def_only_replay",
                    profile=profile,
                    expected_buffer_count=int(
                        row.get("commit_accepted_action_count", 0) or 0
                    ),
                    openroad_num_threads=args.openroad_num_threads,
                )
                row.update(
                    {
                        "replay_status": replay_result["status"],
                        "replay_before_wns_ns": replay_result["before_wns_ns"],
                        "replay_before_tns_ns": replay_result["before_tns_ns"],
                        "replay_after_wns_ns": replay_result["after_wns_ns"],
                        "replay_after_tns_ns": replay_result["after_tns_ns"],
                        "replay_delta_wns_ns": replay_result["delta_wns_ns"],
                        "replay_delta_tns_ns": replay_result["delta_tns_ns"],
                        "replay_buffer_count": replay_result["buffer_count"],
                        "replay_runtime_sec": replay_result["runtime_sec"],
                        "qor_release_status": replay_result["qor_release_status"],
                    }
                )
                if replay_result["status"] != "pass":
                    row["status"] = "failed"
                    row["failure"] = "; ".join(
                        value
                        for value in (
                            row.get("failure"),
                            *replay_result.get("failures", ()),
                        )
                        if value
                    )
            elif profile is not None:
                row.setdefault("replay_status", "not_run")
                row.setdefault("qor_release_status", "not_evaluated")
        rows.append(row)
        details["case_results"][case] = {
            "row": row,
            "command": command,
            "trace_audit": trace_audit if not args.plan_only else None,
            "def_only_replay": replay_result if not args.plan_only else None,
        }
        print(
            f"{case}: {row['status']} "
            f"summary_status={row.get('summary_status')} "
            f"candidate_count={row.get('candidate_count')} "
            f"runtime_refresh={row.get('runtime_refresh_status')}"
        )

    csv_path = run_root / "buffering_inner_loop_summary.csv"
    json_path = run_root / "buffering_inner_loop_summary.json"
    write_csv(csv_path, rows)
    _write_json(json_path, details)
    _write_json(run_root / "manifest.json", details)
    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    return 0 if all(row["status"] in ("pass", "plan_only") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
