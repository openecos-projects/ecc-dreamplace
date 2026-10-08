#!/usr/bin/env python3
"""Run ICCAD24 joint placement+sizing+buffering smokes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "joint_place_sizing_buffering"

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

CSV_COLUMNS = (
    "date",
    "run_id",
    "case",
    "gate",
    "status",
    "failure",
    "iterations",
    "commit_period",
    "outer_iterations",
    "max_buffers_per_segment",
    "commit_count",
    "inserted_buffers",
    "initial_wns",
    "initial_tns",
    "final_wns",
    "final_tns",
    "runtime_per_step",
    "result_dir",
    "log_path",
    "summary_path",
    "refresh_summary_path",
    "refresh_status",
    "refresh_generation",
    "post_rebuild_summary_path",
    "post_rebuild_status",
    "post_rebuild_timing_status",
    "fresh_nonlinear_place_constructed",
    "openroad_eco_mode",
    "openroad_eco_status",
    "openroad_eco_summary_path",
    "commit_transaction_status",
    "commit_qor_status",
    "commit_qor_pass",
    "committed_delta_tns",
    "terminal_summary_path",
    "terminal_status",
    "terminal_qor_status",
    "def_only_opensta_status",
    "def_only_opensta_wns",
    "def_only_opensta_tns",
    "def_only_vs_in_process_wns",
    "def_only_vs_in_process_tns",
    "def_only_final_def_hash_match",
    "def_only_opensta_replay_path",
    "terminal_added_area_um2",
    "terminal_final_overflow",
    "joint_quality_profile",
    "proximal_enabled",
    "proximal_total_weighted",
    "segment_projection_min_z_to_insert",
    "segment_projection_candidates_before_topk",
    "segment_projection_candidates_after_criticality_filter",
    "segment_projection_criticality_filtered_count",
    "segment_projection_candidates",
    "segment_criticality_overlay_status",
    "segment_criticality_overlay_updated_sink_count",
    "segment_criticality_overlay_negative_sink_count",
    "segment_z_max",
    "segment_z_ge_min_insert_count",
    "segment_z_ge_0p5_count",
    "selected_segment_id",
    "selected_net_name",
    "selected_z_value",
    "selected_bsu_value",
    "selected_projection_score",
    "selected_downstream_criticality_sum",
    "command_path",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _last_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _metric_records(result_dir: Path, case: str) -> list[dict[str, Any]]:
    path = _last_existing(
        [
            result_dir / f"{case}_metrics.jsonl",
            result_dir / f"{case}_place_metrics.jsonl",
        ]
    )
    return _read_jsonl(path)


def _average_step_runtime(records: list[dict[str, Any]], log_path: Path | None = None) -> float | None:
    values = [
        _as_float(record.get("full_step_ms"))
        for record in records
        if _as_float(record.get("full_step_ms")) is not None
    ]
    if values:
        return sum(values) / len(values) / 1000.0
    values = _full_step_runtime_from_log(log_path)
    if values:
        return sum(values) / len(values)
    return None


def _full_step_runtime_from_log(log_path: Path | None) -> list[float]:
    if log_path is None or not log_path.exists():
        return []
    pattern = re.compile(r"full step\s+([0-9.]+)\s+ms")
    values: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)) / 1000.0)
    return values


def _last_commit_summary(summary: dict[str, Any]) -> dict[str, Any]:
    trace = list(summary.get("buffer_commit_trace", ()) or ())
    if not trace:
        return {}
    return dict((trace[-1] or {}).get("commit", {}) or {})


def _committed_metric(summary: dict[str, Any], key: str) -> float | None:
    commit = _last_commit_summary(summary)
    value = commit.get(key)
    if value is None:
        value = commit.get(f"post_{key.split('_', 1)[-1]}")
    return _as_float(value)


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
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"{case}: missing required input(s): " + ", ".join(missing))
    return paths


def effective_commit_period(commit_period: int, joint_quality_profile: str) -> int:
    return (
        0
        if joint_quality_profile == "segment_count_direct_joint_v1"
        else int(commit_period)
    )


def build_placer_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    iterations: int,
    commit_period: int,
    outer_iterations: int,
    max_buffers_per_segment: int,
    gate: str,
    joint_quality_profile: str = "",
    dry_run_config: bool = False,
    extra_placer_args: list[str] | None = None,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    resolved_commit_period = effective_commit_period(
        commit_period,
        joint_quality_profile,
    )
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(paths["params_json"]),
        "--flow-kind",
        "joint",
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
        "--iterations",
        str(iterations),
        "--joint-buffer-commit-period",
        str(resolved_commit_period),
        "--joint-buffer-outer-iterations",
        str(outer_iterations),
        "--joint-buffer-max-per-segment",
        str(max_buffers_per_segment),
    ]
    if joint_quality_profile:
        command.extend(["--joint-quality-profile", joint_quality_profile])
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])
    if gate == "gate_b":
        if joint_quality_profile != "segment_count_direct_joint_v1":
            command.extend(
                [
                    "--timing-topology-enable-overflow-threshold",
                    "1.1",
                ]
            )
        command.extend(["--buffering-commit-enabled", "0"])
    if gate == "gate_c":
        command.extend(
            [
                "--buffering-commit-enabled",
                "1",
                "--buffering-committed-def-path",
                str(result_dir / f"{case}_joint_committed.def"),
            ]
        )
    if dry_run_config:
        command.append("--dry-run-config")
    if extra_placer_args:
        command.extend(extra_placer_args)
    return command


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="ignore") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def find_joint_summary(result_dir: Path, case: str) -> Path | None:
    path = result_dir / f"{case}_joint_flow_summary.json"
    return path if path.exists() else None


def find_joint_post_rebuild_summary(result_dir: Path, case: str) -> Path | None:
    path = result_dir / f"{case}_joint_post_rebuild_summary.json"
    return path if path.exists() else None


def find_buffer_commit_refresh_summary(result_dir: Path, case: str) -> Path | None:
    path = result_dir / f"{case}_buffer_commit_pydb_refresh_summary.json"
    return path if path.exists() else None


def find_segment_joint_terminal_summary(result_dir: Path, case: str) -> Path | None:
    path = result_dir / f"{case}_segment_joint_committed_verification.json"
    return path if path.exists() else None


def find_segment_joint_def_only_replay(result_dir: Path, case: str) -> Path | None:
    path = result_dir / f"{case}_segment_joint_def_only_opensta_replay.json"
    return path if path.exists() else None


def summarize_joint_result(
    *,
    case: str,
    gate: str,
    run_id: str,
    result_dir: Path,
    log_path: Path,
    command_path: Path,
    returncode: int,
    iterations: int,
    commit_period: int,
    max_buffers_per_segment: int,
    summary_path: Path | None,
    joint_quality_profile: str = "",
    outer_iterations: int = 1,
) -> dict[str, Any]:
    payload = _read_json(summary_path)
    summary = dict(payload.get("summary", {}) or {})
    post_rebuild_path = find_joint_post_rebuild_summary(result_dir, case)
    post_rebuild_payload = _read_json(post_rebuild_path)
    post_rebuild_result = dict(
        post_rebuild_payload.get("post_rebuild_result", {}) or {}
    )
    refresh_summary_path = find_buffer_commit_refresh_summary(result_dir, case)
    refresh_summary_payload = _read_json(refresh_summary_path)
    terminal_summary_path = find_segment_joint_terminal_summary(result_dir, case)
    terminal_summary = _read_json(terminal_summary_path)
    def_only_replay_path = find_segment_joint_def_only_replay(result_dir, case)
    def_only_replay = _read_json(def_only_replay_path)
    def_only_load_contract = dict(def_only_replay.get("load_contract", {}) or {})
    def_only_metrics = dict(def_only_replay.get("opensta_after", {}) or {})
    def_only_gap = dict(def_only_replay.get("def_only_vs_in_process_gap", {}) or {})
    terminal_def_sha256 = terminal_summary.get("final_def_sha256")
    replay_def_sha256 = def_only_load_contract.get("final_def_sha256")
    def_only_final_def_hash_match = bool(
        terminal_def_sha256
        and replay_def_sha256
        and terminal_def_sha256 == replay_def_sha256
    )
    trace = list(summary.get("buffer_commit_trace", ()) or ())
    proximal = dict(summary.get("proximal", {}) or {})
    buffering_lane = dict(summary.get("buffering_lane", {}) or {})
    buffering_projection = dict(buffering_lane.get("projection", {}) or {})
    z_distribution = dict(
        buffering_projection.get("z_value_distribution")
        or buffering_lane.get("z_value_distribution")
        or {}
    )
    selected_diagnostics = list(
        buffering_projection.get("selected_candidate_diagnostics")
        or buffering_lane.get("projection_selected_candidate_diagnostics")
        or []
    )
    selected_first = dict(selected_diagnostics[0] if selected_diagnostics else {})
    last_commit = trace[-1] if trace else {}
    commit_payload = dict(last_commit.get("commit", {}) or {})
    records = _metric_records(result_dir, case)
    first_metric = records[0] if records else {}
    last_metric = records[-1] if records else {}
    initial_wns = _as_float(first_metric.get("wns"))
    initial_tns = _as_float(first_metric.get("tns"))
    final_wns = _committed_metric(summary, "after_wns")
    final_tns = _committed_metric(summary, "after_tns")
    if terminal_summary:
        final_wns = _as_float(
            dict(terminal_summary.get("opensta_after", {}) or {}).get("wns")
        )
        final_tns = _as_float(
            dict(terminal_summary.get("opensta_after", {}) or {}).get("tns")
        )
    if final_wns is None:
        final_wns = _as_float(last_metric.get("wns"))
    if final_tns is None:
        final_tns = _as_float(last_metric.get("tns"))
    status = "pass" if returncode == 0 else "fail"
    failure = "" if returncode == 0 else f"command_returncode_{returncode}"
    if returncode == 0 and gate == "gate_b" and not trace:
        status = "fail"
        failure = "missing_gate_b_commit_trace"
    if returncode == 0 and gate == "gate_c":
        if not trace:
            status = "fail"
            failure = "missing_gate_c_commit_trace"
        elif commit_payload.get("status") == "disabled":
            status = "fail"
            failure = "gate_c_commit_disabled"
        elif commit_payload.get("status") not in {"accepted", "rejected", "failed", "skipped"}:
            status = "fail"
            failure = f"unexpected_gate_c_commit_status_{commit_payload.get('status')}"
        elif (
            _as_int(commit_payload.get("accepted_action_count")) or 0
        ) > 0 and not post_rebuild_result and not terminal_summary:
            status = "fail"
            failure = "missing_gate_c_post_rebuild_summary"
        elif (
            _as_int(commit_payload.get("accepted_action_count")) or 0
        ) > 0 and not refresh_summary_payload and not terminal_summary:
            status = "fail"
            failure = "missing_gate_c_refresh_summary"
        elif post_rebuild_result and post_rebuild_result.get("status") != "ok":
            status = "fail"
            failure = "gate_c_post_rebuild_failed"
        elif (
            refresh_summary_payload
            and refresh_summary_payload.get("refresh_status") != "ok"
        ):
            status = "fail"
            failure = "gate_c_refresh_failed"
        elif terminal_summary and terminal_summary.get("status") != "ok":
            status = "fail"
            failure = "gate_c_terminal_verification_failed"
        elif joint_quality_profile == "segment_count_direct_joint_v1":
            if not def_only_replay:
                status = "fail"
                failure = "missing_gate_c_def_only_opensta_replay"
            elif def_only_replay.get("status") != "ok":
                status = "fail"
                failure = "gate_c_def_only_opensta_replay_failed"
            elif (
                def_only_load_contract.get("design_carrier") != "final_def"
                or def_only_load_contract.get("fresh_openroad_bridge") is not True
                or def_only_load_contract.get("verilog_read") is not False
            ):
                status = "fail"
                failure = "gate_c_def_only_load_contract_failed"
            elif not def_only_final_def_hash_match:
                status = "fail"
                failure = "gate_c_def_only_final_def_hash_mismatch"
    return {
        "date": time.strftime("%Y-%m-%d"),
        "run_id": run_id,
        "case": case,
        "gate": gate,
        "status": status,
        "failure": failure,
        "iterations": int(iterations),
        "commit_period": int(commit_period),
        "outer_iterations": int(outer_iterations),
        "max_buffers_per_segment": int(max_buffers_per_segment),
        "commit_count": len(trace),
        "inserted_buffers": _as_int(commit_payload.get("accepted_action_count")),
        "initial_wns": initial_wns,
        "initial_tns": initial_tns,
        "final_wns": final_wns,
        "final_tns": final_tns,
        "runtime_per_step": _average_step_runtime(records, log_path),
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "summary_path": "" if summary_path is None else str(summary_path),
        "refresh_summary_path": (
            "" if refresh_summary_path is None else str(refresh_summary_path)
        ),
        "refresh_status": refresh_summary_payload.get("refresh_status")
        or dict(terminal_summary.get("refresh_summary", {}) or {}).get("status"),
        "refresh_generation": refresh_summary_payload.get("refresh_generation")
        or dict(
            dict(terminal_summary.get("refresh_summary", {}) or {}).get(
                "rebuild", {}
            )
            or {}
        ).get("topology_generation"),
        "post_rebuild_summary_path": (
            "" if post_rebuild_path is None else str(post_rebuild_path)
        ),
        "post_rebuild_status": post_rebuild_result.get("status")
        or terminal_summary.get("status"),
        "post_rebuild_timing_status": post_rebuild_result.get(
            "post_rebuild_timing_status"
        ),
        "fresh_nonlinear_place_constructed": post_rebuild_result.get(
            "fresh_nonlinear_place_constructed"
        ),
        "openroad_eco_mode": post_rebuild_result.get("openroad_eco_mode"),
        "openroad_eco_status": post_rebuild_result.get("openroad_eco_status"),
        "openroad_eco_summary_path": post_rebuild_result.get(
            "openroad_eco_summary_path",
            "",
        ),
        "commit_transaction_status": commit_payload.get("transaction_status")
        or commit_payload.get("status"),
        "commit_qor_status": commit_payload.get("qor_status"),
        "commit_qor_pass": commit_payload.get("qor_pass"),
        "committed_delta_tns": _as_float(
            terminal_summary.get("delta_tns")
            if terminal_summary
            else commit_payload.get("delta_tns")
        ),
        "terminal_summary_path": (
            "" if terminal_summary_path is None else str(terminal_summary_path)
        ),
        "terminal_status": terminal_summary.get("status"),
        "terminal_qor_status": terminal_summary.get("qor_status"),
        "def_only_opensta_status": def_only_replay.get("status"),
        "def_only_opensta_wns": _as_float(def_only_metrics.get("wns")),
        "def_only_opensta_tns": _as_float(def_only_metrics.get("tns")),
        "def_only_vs_in_process_wns": _as_float(def_only_gap.get("wns")),
        "def_only_vs_in_process_tns": _as_float(def_only_gap.get("tns")),
        "def_only_final_def_hash_match": def_only_final_def_hash_match,
        "def_only_opensta_replay_path": (
            "" if def_only_replay_path is None else str(def_only_replay_path)
        ),
        "terminal_added_area_um2": _as_float(
            terminal_summary.get("added_area_um2")
        ),
        "terminal_final_overflow": _as_float(
            dict(terminal_summary.get("final_placement_metrics", {}) or {}).get(
                "overflow"
            )
        ),
        "joint_quality_profile": joint_quality_profile or proximal.get("profile"),
        "proximal_enabled": proximal.get("enabled"),
        "proximal_total_weighted": _as_float(proximal.get("total_weighted")),
        "segment_projection_min_z_to_insert": _as_float(
            buffering_projection.get("min_z_to_insert")
            or buffering_lane.get("segment_projection_min_z_to_insert")
        ),
        "segment_projection_candidates_before_topk": _as_int(
            buffering_projection.get("projected_candidate_count_before_topk")
        ),
        "segment_projection_candidates_after_criticality_filter": _as_int(
            buffering_projection.get("projected_candidate_count_after_criticality_filter")
        ),
        "segment_projection_criticality_filtered_count": _as_int(
            buffering_projection.get("criticality_filtered_candidate_count")
            or buffering_lane.get("criticality_filtered_candidate_count")
        ),
        "segment_projection_candidates": _as_int(
            buffering_projection.get("projected_candidate_count")
        ),
        "segment_criticality_overlay_status": buffering_projection.get(
            "criticality_overlay_status"
        ),
        "segment_criticality_overlay_updated_sink_count": _as_int(
            buffering_projection.get("criticality_overlay_updated_sink_count")
        ),
        "segment_criticality_overlay_negative_sink_count": _as_int(
            buffering_projection.get("criticality_overlay_negative_sink_count")
        ),
        "segment_z_max": _as_float(z_distribution.get("max")),
        "segment_z_ge_min_insert_count": _as_int(
            z_distribution.get("ge_min_insert_count")
        ),
        "segment_z_ge_0p5_count": _as_int(z_distribution.get("ge_0p5_count")),
        "selected_segment_id": _as_int(selected_first.get("segment_id")),
        "selected_net_name": selected_first.get("net_name"),
        "selected_z_value": _as_float(selected_first.get("z_value")),
        "selected_bsu_value": _as_float(selected_first.get("bsu_value")),
        "selected_projection_score": _as_float(
            selected_first.get("projection_selection_score")
        ),
        "selected_downstream_criticality_sum": _as_float(
            selected_first.get("downstream_sink_criticality_sum")
        ),
        "command_path": str(command_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)),
    )
    parser.add_argument("--gate", choices=("gate_a", "gate_b", "gate_c"), default="gate_a")
    parser.add_argument(
        "--joint-quality-profile",
        choices=("proximal_alternating_v1", "segment_count_direct_joint_v1"),
        default="",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--commit-period", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument("--max-buffers-per-segment", type=int, default=3)
    parser.add_argument(
        "--extra-placer-args",
        action="append",
        default=[],
        help=(
            "Additional arguments appended to the Placer.py command. "
            "Use this only for focused smoke/debug runs."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    cases = resolve_cases(args.cases)
    run_root = args.output_root.resolve() / args.run_id
    extra_placer_args = []
    for raw in args.extra_placer_args:
        extra_placer_args.extend(shlex.split(raw))
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "benchmark_root": str(benchmark_root),
        "run_root": str(run_root),
        "run_id": args.run_id,
        "gate": args.gate,
        "outer_iterations": int(args.outer_iterations),
        "cases": cases,
        "case_results": {},
    }
    resolved_commit_period = effective_commit_period(
        args.commit_period,
        args.joint_quality_profile,
    )
    for case in cases:
        case_root = run_root / case
        result_dir = case_root / "result"
        log_path = case_root / "run.log"
        command_path = case_root / "command.txt"
        command = build_placer_command(
            python_bin=args.python_bin,
            benchmark_root=benchmark_root,
            case=case,
            result_dir=result_dir,
            iterations=args.iterations,
            commit_period=args.commit_period,
            outer_iterations=args.outer_iterations,
            max_buffers_per_segment=args.max_buffers_per_segment,
            gate=args.gate,
            joint_quality_profile=args.joint_quality_profile,
            dry_run_config=args.dry_run_config,
            extra_placer_args=extra_placer_args,
        )
        case_root.mkdir(parents=True, exist_ok=True)
        command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
        if args.plan_only:
            row = {
                "date": time.strftime("%Y-%m-%d"),
                "run_id": args.run_id,
                "case": case,
                "gate": args.gate,
                "status": "plan_only",
                "failure": "",
                "iterations": int(args.iterations),
                "commit_period": resolved_commit_period,
                "outer_iterations": int(args.outer_iterations),
                "max_buffers_per_segment": int(args.max_buffers_per_segment),
                "joint_quality_profile": args.joint_quality_profile,
                "commit_count": 0,
                "inserted_buffers": None,
                "result_dir": str(result_dir),
                "log_path": str(log_path),
                "summary_path": "",
                "command_path": str(command_path),
            }
        else:
            returncode = run_command(command, cwd=REPO_ROOT, log_path=log_path)
            summary_path = find_joint_summary(result_dir, case)
            row = summarize_joint_result(
                case=case,
                gate=args.gate,
                run_id=args.run_id,
                result_dir=result_dir,
                log_path=log_path,
                command_path=command_path,
                returncode=returncode,
                iterations=args.iterations,
                commit_period=resolved_commit_period,
                max_buffers_per_segment=args.max_buffers_per_segment,
                summary_path=summary_path,
                joint_quality_profile=args.joint_quality_profile,
                outer_iterations=args.outer_iterations,
            )
        rows.append(row)
        details["case_results"][case] = {"row": row, "command": command}
        print(
            f"{case}: {row['status']} gate={args.gate} "
            f"commits={row.get('commit_count')} failure={row.get('failure')}"
        )

    csv_path = run_root / "joint_place_sizing_buffering_summary.csv"
    json_path = run_root / "joint_place_sizing_buffering_summary.json"
    write_csv(csv_path, rows)
    _write_json(json_path, details)
    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    return 0 if all(row["status"] in ("pass", "plan_only") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
