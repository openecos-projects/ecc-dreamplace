#!/usr/bin/env python3
"""Run the fixed-recipe B0/B1/Ours equal-spaced buffering comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
PROFILE_PATH = SCRIPT_DIR / "params" / "paper_profile.json"


def _path_from_env(name: str, fallback: Path | str) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else Path(fallback)


DEFAULT_BENCHMARK_ROOT = _path_from_env(
    "AUTODMP_ICCAD24_BENCHMARK_ROOT",
    AUTODMP_ROOT / "benchmarks" / "iccad24_benchmark",
)
DEFAULT_PYTHON = _path_from_env("AUTODMP_PYTHON_BIN", sys.executable)
DEFAULT_OPENROAD = _path_from_env(
    "AUTODMP_OPENROAD_BIN",
    AUTODMP_ROOT / "build" / "thirdparty" / "OpenROAD" / "src" / "openroad",
)
DEFAULT_OUTPUT_ROOT = _path_from_env(
    "AUTODMP_BUFFERING_OUTPUT_ROOT",
    AUTODMP_ROOT / "logs" / "regression" / "iccad24" / "equal_spaced_buffering",
)
METHODS = ("b0", "b1", "ours")
EXPERIMENT_METHODS = METHODS + ("b1_rd_rt",)
RELAXED_CAPACITY_FEASIBILITY_TOLERANCE = 1.0e-4
METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.+)$")
COMPONENT_RE = re.compile(r"^\s*-\s+(\S+)\s+(\S+)")
POWER_TOTAL_RE = re.compile(
    r"^\s*Total\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)
VIOLATION_RE = r"^.* +([-\.0-9]+) +\(VIOLATED\)"

CSV_COLUMNS = (
    "date",
    "run_id",
    "case",
    "method",
    "status",
    "failure",
    "input_def_sha256",
    "final_def",
    "final_def_sha256",
    "wns_ns",
    "tns_ns",
    "delta_tns_ns",
    "buffer_count",
    "power_w",
    "area",
    "slew_violation",
    "cap_violation",
    "runtime_sec",
    "action_audit_status",
    "action_audit_path",
    "run_log",
    "artifact_path",
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


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _tcl_quote(value: Path | str) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def _load_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    profile = _read_json(path)
    if tuple(profile.get("methods", ())) != METHODS:
        raise ValueError("paper profile must define exactly b0, b1, ours")
    if not profile.get("suite"):
        raise ValueError("paper profile has an empty suite")
    segment = dict(profile.get("segment_mode") or {})
    if int(segment.get("max_repeaters_per_segment", 0)) != 3:
        raise ValueError("paper profile requires Nmax=3")
    if int(segment.get("integer_projection_interval", -1)) != 0:
        raise ValueError("paper profile must disable periodic integer projection")
    if not segment.get("one_final_hard_commit"):
        raise ValueError("paper profile must require exactly one final hard commit")
    if not segment.get("buffer_master"):
        raise ValueError("paper profile requires one fixed buffer master")
    if "gpu_id" in segment and int(segment["gpu_id"]) < 0:
        raise ValueError("paper profile requires a nonnegative gpu_id")
    return profile


def _resolve_selected(raw_values: list[str] | None, allowed: tuple[str, ...]) -> list[str]:
    if not raw_values:
        return list(allowed)
    selected: list[str] = []
    for raw in raw_values:
        for item in str(raw).split(","):
            value = item.strip()
            if value:
                selected.append(value)
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError("unsupported selection: " + ", ".join(invalid))
    return selected


def _asap7_inputs(benchmark_root: Path) -> dict[str, Path | list[Path]]:
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


def _case_inputs(benchmark_root: Path, case: str) -> dict[str, Path]:
    root = benchmark_root / "design" / case
    paths = {
        "root": root,
        "workspace": root / "workspace",
        "params_json": root / "workspace" / "config" / "dreamplace_config" / "param.json",
        "def": root / f"{case}.def",
        "verilog": root / f"{case}.v",
        "sdc": root / f"{case}.sdc",
    }
    tech = _asap7_inputs(benchmark_root)
    required = list(paths.values()) + [tech["tech_lef"], tech["rc_tcl"]]
    required.extend(tech["lef"])
    required.extend(tech["lib"])
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{case}: missing input(s): " + ", ".join(missing))
    return paths


def _parse_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    metrics: dict[str, Any] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = METRIC_RE.match(line.strip())
        if match is None:
            continue
        key, raw = match.groups()
        metrics[key] = _as_float(raw) if _as_float(raw) is not None else raw
    return metrics


def _parse_power_total(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = POWER_TOTAL_RE.match(line)
        if match is not None:
            return _as_float(match.group(4))
    return None


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = None
    if environment:
        env = dict(os.environ)
        env.update({str(key): str(value) for key, value in environment.items()})
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
    }


def _openroad_command(
    openroad_bin: Path,
    tcl_path: Path,
    openroad_num_threads: int | None,
) -> list[str]:
    command = [str(openroad_bin)]
    if openroad_num_threads is not None:
        command.extend(("-threads", str(int(openroad_num_threads))))
    command.extend(("-exit", str(tcl_path)))
    return command


def _tool_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""


def _components_from_def(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"^COMPONENTS\s+\d+\s*;", text, flags=re.MULTILINE)
    end = re.search(r"^END COMPONENTS\s*$", text, flags=re.MULTILINE)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError(f"{path}: missing DEF COMPONENTS section")
    components: dict[str, str] = {}
    for line in text[start.end() : end.start()].splitlines():
        match = COMPONENT_RE.match(line)
        if match is not None:
            instance, master = match.groups()
            components[instance] = master
    return components


def audit_def_mutation(
    *,
    input_def: Path,
    final_def: Path,
    expected_buffer_master: str,
    require_new_buffers: bool,
) -> dict[str, Any]:
    before = _components_from_def(input_def)
    after = _components_from_def(final_def)
    before_names = set(before)
    after_names = set(after)
    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    changed = sorted(
        name for name in before_names & after_names if before[name] != after[name]
    )
    unexpected_added = sorted(
        name for name in added if after[name] != expected_buffer_master
    )
    failures = []
    if removed:
        failures.append("instances_removed")
    if changed:
        failures.append("existing_instance_master_changed")
    if unexpected_added:
        failures.append("new_instance_not_expected_buffer_master")
    if require_new_buffers and not added:
        failures.append("no_buffer_instances_added")
    return {
        "status": "pass" if not failures else "failed",
        "input_def": str(input_def),
        "final_def": str(final_def),
        "expected_buffer_master": expected_buffer_master,
        "before_component_count": len(before),
        "after_component_count": len(after),
        "added_buffer_count": len(added),
        "added_instances": added,
        "removed_instances": removed,
        "master_changed_instances": [
            {"instance": name, "before": before[name], "after": after[name]}
            for name in changed
        ],
        "unexpected_added_instances": [
            {"instance": name, "master": after[name]} for name in unexpected_added
        ],
        "failures": failures,
    }


def _tcl_preamble(*, benchmark_root: Path, case: str, def_input: Path) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    lines = [
        "proc emit_metric {name value} { puts \"METRIC|$name|$value\" }",
        "proc safe_metric {name script default_value} {",
        "  if {[catch {uplevel 1 $script} value]} { emit_metric $name $default_value } else { emit_metric $name $value }",
        "}",
        "proc refresh_timing {} {",
        "  if {[llength [info commands update_timing]]} { update_timing }",
        "}",
        "proc sum_check_type_violation {check_flag} {",
        "  set tmp_report [file join [pwd] \"openroad_[pid]_${check_flag}.rpt\"]",
        "  if {[catch {eval report_check_types $check_flag -no_line_splits -digits 6 > $tmp_report}]} { return 0.0 }",
        "  set total 0.0",
        "  set fp [open $tmp_report r]",
        "  set file_data [split [read $fp] \\n]",
        "  close $fp",
        "  file delete -force $tmp_report",
        "  foreach line $file_data {",
        f"    if {{[regexp {_tcl_quote(VIOLATION_RE)} $line -> slack_val]}} {{ set total [expr {{$total - $slack_val}}] }}",
        "  }",
        "  return $total",
        "}",
        "set start_us [clock microseconds]",
        f"read_lef {_tcl_quote(tech['tech_lef'])}",
    ]
    for lef in tech["lef"]:
        lines.append(f"read_lef {_tcl_quote(lef)}")
    for liberty in tech["lib"]:
        lines.append(f"read_liberty {_tcl_quote(liberty)}")
    lines.extend(
        [
            f"read_def -continue_on_errors {_tcl_quote(def_input)}",
            f"read_sdc {_tcl_quote(inputs['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
        ]
    )
    return lines


def generate_openroad_tcl(
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
    output_def: Path,
    power_report: Path,
    method: str,
    buffer_master: str,
    b1_max_buffer_percent: float,
    source_only: bool = False,
) -> str:
    if method not in ("b0", "b1", "b1_rd_rt", "ours_replay"):
        raise ValueError(f"unsupported OpenROAD Tcl method: {method}")
    lines = _tcl_preamble(
        benchmark_root=benchmark_root,
        case=case,
        def_input=def_input,
    )
    lines.extend(
        [
            "refresh_timing",
            'safe_metric "pre_wns_ns" {worst_slack -max} ""',
            'safe_metric "pre_tns_ns" {total_negative_slack -max} ""',
        ]
    )
    if method in ("b1", "b1_rd_rt"):
        lines.extend(
            [
                f"set allowed_buffer_cells [get_lib_cells {_tcl_quote(buffer_master)}]",
                "if {[llength $allowed_buffer_cells] != 1} { error \"missing fixed buffer master\" }",
                "set_dont_use [get_lib_cells *]",
                "unset_dont_use $allowed_buffer_cells",
                f'emit_metric "buffer_only_master" "{buffer_master}"',
            ]
        )
        if method == "b1_rd_rt":
            lines.extend(
                [
                    'emit_metric "buffer_only_sequence" "repair_design,buffer"',
                    "set repair_start_us [clock microseconds]",
                    "set repair_design_start_us [clock microseconds]",
                    "repair_design",
                    'emit_metric "repair_design_runtime_sec" [expr {([clock microseconds] - $repair_design_start_us) / 1000000.0}]',
                    "detailed_placement",
                    "estimate_parasitics -placement",
                    "refresh_timing",
                    'safe_metric "post_repair_design_wns_ns" {worst_slack -max} ""',
                    'safe_metric "post_repair_design_tns_ns" {total_negative_slack -max} ""',
                    'emit_metric "post_repair_design_slew_violation" [sum_check_type_violation "-max_slew"]',
                    'emit_metric "post_repair_design_cap_violation" [sum_check_type_violation "-max_capacitance"]',
                ]
            )
        else:
            lines.extend(
                [
                    'emit_metric "buffer_only_sequence" "buffer"',
                    "set repair_start_us [clock microseconds]",
                ]
            )
        lines.extend(
            [
                "set repair_timing_start_us [clock microseconds]",
                "repair_timing -setup -sequence \"buffer\" -skip_last_gasp "
                "-skip_pin_swap -skip_gate_cloning -skip_size_down -skip_vt_swap "
                f"-skip_crit_vt_swap -skip_buffer_removal -max_buffer_percent {float(b1_max_buffer_percent):.12g}",
                'emit_metric "repair_timing_runtime_sec" [expr {([clock microseconds] - $repair_timing_start_us) / 1000000.0}]',
                'emit_metric "repair_runtime_sec" [expr {([clock microseconds] - $repair_start_us) / 1000000.0}]',
            ]
        )
    else:
        lines.append('emit_metric "repair_runtime_sec" 0.0')
    if source_only:
        lines.extend(
            [
                f"write_def {_tcl_quote(output_def)}",
                'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
                "exit",
            ]
        )
    else:
        lines.extend(
            [
            "detailed_placement",
            "estimate_parasitics -placement",
            "refresh_timing",
            'safe_metric "wns_ns" {worst_slack -max} ""',
            'safe_metric "tns_ns" {total_negative_slack -max} ""',
            'emit_metric "slew_violation" [sum_check_type_violation "-max_slew"]',
            'emit_metric "cap_violation" [sum_check_type_violation "-max_capacitance"]',
            'safe_metric "instance_count" {llength [get_cells *]} ""',
            'safe_metric "area" {rsz::design_area} ""',
            f"report_power > {_tcl_quote(power_report)}",
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
            ]
        )
    return "\n".join(lines) + "\n"


def build_ours_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    committed_def: Path,
    profile: dict[str, Any],
    capacity_grid: str | None = None,
    segment_strategy: str = "continuous",
    logical_gpu_id: int | None = None,
    def_input: Path | None = None,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    def_input = inputs["def"] if def_input is None else Path(def_input)
    tech = _asap7_inputs(benchmark_root)
    segment = dict(profile["segment_mode"])
    if segment_strategy not in ("continuous", "discrete_net_gradient"):
        raise ValueError(f"unsupported segment strategy: {segment_strategy}")
    if segment_strategy == "discrete_net_gradient" and capacity_grid is not None:
        raise ValueError("discrete_net_gradient cannot be combined with capacity control")
    z_init = 0.0 if segment_strategy == "discrete_net_gradient" else float(segment["z_init"])
    projection_start_step = 0
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(inputs["params_json"]),
        "--flow-kind",
        "buffering",
        "--place-io-engine",
        "openroad",
        "--buffering-mode",
        "segment",
        "--buffering-segment-strategy",
        segment_strategy,
        "--workspace",
        str(inputs["workspace"]),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--def-input",
        str(def_input),
        "--verilog-input",
        str(inputs["verilog"]),
        "--sdc",
        str(inputs["sdc"]),
        "--rc-tcl",
        str(tech["rc_tcl"]),
        "--tech-lef",
        str(tech["tech_lef"]),
        "--with-sta",
        "--gpu",
        "1",
        "--gpu-id",
        str(int(segment.get("gpu_id", 0) if logical_gpu_id is None else logical_gpu_id)),
        "--iterations",
        "1",
        "--buffering-committed-def-path",
        str(committed_def),
        "--buffering-continuous-steps",
        str(int(segment["continuous_steps"])),
        "--buffering-continuous-lr",
        str(float(segment["continuous_lr"])),
        "--buffering-max-repeaters-per-segment",
        str(int(segment["max_repeaters_per_segment"])),
        "--buffering-segment-count-z-init",
        str(z_init),
        "--buffering-fixed-bsu-index",
        str(int(segment["fixed_bsu_index"])),
        "--buffering-route-b-selection-fraction",
        str(float(segment.get("selection_fraction", 0.001))),
        "--buffering-segment-count-timing-backend",
        str(segment["timing_backend"]),
        "--buffering-segment-transfer-backend",
        str(segment["transfer_backend"]),
        "--buffering-segment-integer-projection-interval",
        str(int(segment["integer_projection_interval"])),
        "--buffering-segment-integer-projection-start-step",
        str(projection_start_step),
        "--buffering-commit-enabled",
        "1",
    ]
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for liberty in tech["lib"]:
        command.extend(["--lib", str(liberty)])
    if capacity_grid is not None:
        command.extend(
            [
                "--buffering-segment-capacity-enabled",
                "1",
                "--buffering-segment-capacity-grid",
                str(capacity_grid),
            ]
        )
    return command


def _find_ours_summary(result_dir: Path, case: str) -> Path | None:
    direct = result_dir / f"{case}_buffering_inner_loop_summary.json"
    if direct.exists():
        return direct
    matches = sorted(result_dir.rglob(f"{case}_buffering_inner_loop_summary.json"))
    return matches[-1] if matches else None


def _is_no_action_commit(summary: dict[str, Any]) -> bool:
    """Recognize a completed Route-B run that had nothing to commit."""

    inner = dict(summary.get("inner_loop") or {})
    commit = dict(summary.get("commit") or {})
    return (
        summary.get("status") == "completed"
        and inner.get("terminal_reason")
        in {"no_positive_action", "no_improving_prefix"}
        and commit.get("status") in {"skipped", "disabled_or_no_actions"}
        and commit.get("reason") == "no_projected_buffer_actions"
        and int(commit.get("action_count", 0) or 0) == 0
        and int(commit.get("attempted_action_count", 0) or 0) == 0
        and int(commit.get("accepted_action_count", 0) or 0) == 0
        and int(commit.get("failed_action_count", 0) or 0) == 0
        and int(commit.get("rejected_action_count", 0) or 0) == 0
    )


def _no_action_audit(*, input_def: Path, expected_buffer_master: str) -> dict[str, Any]:
    """Create an explicit audit for a valid zero-mutation buffering result."""

    return {
        "status": "skipped",
        "reason": "no_projected_buffer_actions",
        "input_def": str(input_def),
        "final_def": str(input_def),
        "expected_buffer_master": expected_buffer_master,
        "added_buffer_count": 0,
        "added_instances": [],
        "removed_instances": [],
        "master_changed_instances": [],
        "unexpected_added_instances": [],
        "failures": [],
    }


def _ours_contract_failures(
    summary: dict[str, Any],
    profile: dict[str, Any],
    *,
    segment_strategy: str,
) -> list[str]:
    segment = dict(profile["segment_mode"])
    inner = dict(summary.get("inner_loop") or {})
    commit = dict(summary.get("commit") or {})
    failures = []
    if summary.get("status") != "completed":
        failures.append("inner_loop_not_completed")
    if summary.get("state_kind") != "segment_count":
        failures.append("not_segment_count_state")
    if not summary.get("full_design_scope"):
        failures.append("not_full_design_scope")
    if int(summary.get("max_repeater_count", -1)) != int(segment["max_repeaters_per_segment"]):
        failures.append("nmax_mismatch")
    if summary.get("fixed_bsu_index") != int(segment["fixed_bsu_index"]):
        failures.append("fixed_bsu_mismatch")
    if summary.get("buffer_size_optimization") is not False:
        failures.append("buffer_size_was_optimized")
    if int(inner.get("projection_count", -1)) != 1:
        failures.append("not_final_projection_only")
    if int(inner.get("periodic_integer_projection_count", -1)) != 0:
        failures.append("periodic_integer_projection_used")
    if segment_strategy == "discrete_net_gradient":
        if summary.get("strategy") != "discrete_net_gradient":
            failures.append("discrete_strategy_not_recorded")
        if not inner.get("terminal_metrics"):
            failures.append("missing_discrete_terminal_metrics")
        trace_path = inner.get("trace_path")
        if not trace_path or not Path(trace_path).exists():
            failures.append("missing_discrete_scheduler_trace")
    no_action_commit = _is_no_action_commit(summary)
    if not no_action_commit and commit.get("status") != "accepted":
        failures.append("final_commit_not_accepted")
    if not no_action_commit and int(commit.get("action_count", 0) or 0) <= 0:
        failures.append("no_projected_buffer_actions")
    if int(commit.get("failed_action_count", 0) or 0) != 0:
        failures.append("failed_buffer_actions")
    if int(commit.get("rejected_action_count", 0) or 0) != 0:
        failures.append("rejected_buffer_actions")
    if (
        not no_action_commit
        and int(commit.get("accepted_action_count", 0) or 0)
        != int(commit.get("action_count", 0) or 0)
    ):
        failures.append("not_all_projected_actions_committed")
    return failures


def _capacity_contract_failures(summary: dict[str, Any], capacity_grid: str) -> list[str]:
    """Validate the frozen continuous resource state before accepting QoR."""

    failures = []
    if not summary.get("segment_capacity_enabled"):
        failures.append("segment_capacity_not_enabled")
    if int(summary.get("segment_capacity_selected_grid_factor", -1)) != int(capacity_grid):
        failures.append("segment_capacity_grid_mismatch")
    trace = list((summary.get("inner_loop") or {}).get("metrics_trace") or ())
    if not trace:
        return failures + ["missing_segment_capacity_metrics_trace"]
    final_metrics = dict(trace[-1])
    violation = _as_float(final_metrics.get("segment_capacity_max_violation"))
    if violation is None:
        failures.append("missing_final_segment_capacity_violation")
    elif violation > RELAXED_CAPACITY_FEASIBILITY_TOLERANCE:
        failures.append("final_segment_capacity_infeasible")
    return failures


def _close_segment_capacity_committed_fidelity(
    *,
    summary: dict[str, Any],
    committed_def: Path,
    final_audit: dict[str, Any],
) -> dict[str, Any]:
    """Add the DEF-only replay map to the capacity artifact after final commit."""

    artifacts = dict(summary.get("segment_capacity_artifacts") or {})
    projection_path = artifacts.get("segment_capacity_projection_fidelity.json")
    config_path = artifacts.get("segment_capacity_config.json")
    if not projection_path or not config_path:
        raise ValueError("capacity-enabled summary is missing capacity artifact paths")
    from dreamplace.ops.buffer_insertion.segment_capacity import (
        close_committed_def_fidelity,
    )

    return close_committed_def_fidelity(
        projection_path=Path(projection_path),
        config_path=Path(config_path),
        committed_def_path=committed_def,
        committed_instance_names=final_audit.get("added_instances", ()),
    )


def _method_result(
    *,
    case: str,
    method: str,
    input_def: Path,
    final_def: Path,
    metrics: dict[str, Any],
    runtime_sec: float | None,
    audit: dict[str, Any] | None,
    artifact_path: Path,
    run_log: Path,
    failure: str = "",
) -> dict[str, Any]:
    audit_status = audit.get("status") if audit is not None else "not_run"
    buffer_count = audit.get("added_buffer_count") if audit is not None else None
    if failure:
        status = "failed"
    elif audit_status == "pass":
        status = "pass"
    elif audit_status == "skipped":
        status = "skipped"
    else:
        status = "failed"
    return {
        "case": case,
        "method": method,
        "status": status,
        "failure": failure,
        "input_def_sha256": _sha256(input_def),
        "final_def": str(final_def),
        "final_def_sha256": _sha256(final_def) if final_def.exists() else "",
        "wns_ns": _as_float(metrics.get("wns_ns")),
        "tns_ns": _as_float(metrics.get("tns_ns")),
        "buffer_count": buffer_count,
        "power_w": _parse_power_total(artifact_path.parent / "power.rpt"),
        "area": _as_float(metrics.get("area")),
        "slew_violation": _as_float(metrics.get("slew_violation")),
        "cap_violation": _as_float(metrics.get("cap_violation")),
        "runtime_sec": runtime_sec,
        "action_audit_status": audit_status,
        "action_audit_path": str(artifact_path.parent / "action_audit.json") if audit is not None else "",
        "run_log": str(run_log),
        "artifact_path": str(artifact_path),
    }


def _run_openroad_method(
    *,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    method: str,
    input_def: Path,
    case_root: Path,
    profile: dict[str, Any],
    plan_only: bool,
    source_only: bool = False,
    openroad_num_threads: int | None = None,
) -> dict[str, Any]:
    method_root = case_root / method
    method_root.mkdir(parents=True, exist_ok=True)
    final_def = method_root / f"{case}_{method}.def"
    power_report = method_root / "power.rpt"
    tcl_path = method_root / "run.tcl"
    log_path = method_root / "run.log"
    tcl_path.write_text(
        generate_openroad_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=input_def,
            output_def=final_def,
            power_report=power_report,
            method=method,
            buffer_master=str(profile["segment_mode"]["buffer_master"]),
            b1_max_buffer_percent=float(profile["openroad_buffer_only"]["max_buffer_percent"]),
            source_only=source_only,
        ),
        encoding="utf-8",
    )
    if plan_only:
        return {
            "case": case,
            "method": method,
            "status": "plan_only",
            "failure": "",
            "final_def": str(final_def),
            "run_log": str(log_path),
            "artifact_path": str(method_root / "result.json"),
        }
    execution = _run_command(
        _openroad_command(openroad_bin, tcl_path, openroad_num_threads),
        cwd=method_root,
        log_path=log_path,
    )
    metrics = _parse_metrics(log_path)
    failure = "" if execution["status"] == "ok" else f"OpenROAD exited with code {execution['returncode']}"
    audit = None
    if not failure and final_def.exists():
        audit = audit_def_mutation(
            input_def=input_def,
            final_def=final_def,
            expected_buffer_master=str(profile["segment_mode"]["buffer_master"]),
            require_new_buffers=(method in ("b1", "b1_rd_rt")),
        )
        if method == "b0" and audit["added_buffer_count"]:
            audit["status"] = "failed"
            audit["failures"].append("b0_inserted_buffers")
        _write_json(method_root / "action_audit.json", audit)
        if audit["status"] != "pass":
            failure = "; ".join(audit["failures"])
    elif not failure:
        failure = "missing_final_def"
    artifact_path = method_root / "result.json"
    result = _method_result(
        case=case,
        method=method,
        input_def=input_def,
        final_def=final_def,
        metrics=metrics,
        runtime_sec=execution["runtime_sec"],
        audit=audit,
        artifact_path=artifact_path,
        run_log=log_path,
        failure=failure,
    )
    _write_json(
        artifact_path,
        {
            "artifact": "equal_spaced_buffering_method_result",
            "artifact_version": 1,
            "result": result,
            "metrics": metrics,
            "execution": execution,
            "audit": audit,
            "source_only": bool(source_only),
        },
    )
    return result


def _run_ours(
    *,
    python_bin: Path,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    case_root: Path,
    profile: dict[str, Any],
    plan_only: bool,
    method_name: str = "ours",
    method_root_name: str = "ours",
    environment: dict[str, str] | None = None,
    expected_action_count: int | None = None,
    capacity_grid: str | None = None,
    segment_strategy: str = "continuous",
    logical_gpu_id: int | None = None,
    source_only: bool = False,
    def_input: Path | None = None,
    openroad_num_threads: int | None = None,
) -> dict[str, Any]:
    inputs = _case_inputs(benchmark_root, case)
    def_input = inputs["def"] if def_input is None else Path(def_input)
    method_root = case_root / method_root_name
    result_dir = method_root / "autodmp_result"
    committed_def = method_root / f"{case}_{method_root_name}_committed.def"
    replay_def = method_root / f"{case}_{method_root_name}.def"
    method_root.mkdir(parents=True, exist_ok=True)
    command = build_ours_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=result_dir,
        committed_def=committed_def,
        profile=profile,
        capacity_grid=capacity_grid,
        segment_strategy=segment_strategy,
        logical_gpu_id=logical_gpu_id,
        def_input=def_input,
    )
    (method_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if plan_only:
        return {
            "case": case,
            "method": method_name,
            "status": "plan_only",
            "failure": "",
            "final_def": str(replay_def),
            "run_log": str(method_root / "autodmp.log"),
            "artifact_path": str(method_root / "result.json"),
        }
    execution = _run_command(
        command,
        cwd=AUTODMP_ROOT,
        log_path=method_root / "autodmp.log",
        environment=environment,
    )
    summary_path = _find_ours_summary(result_dir, case)
    summary = _read_json(summary_path).get("summary", {}) if summary_path is not None else {}
    failures = []
    if execution["status"] != "ok":
        failures.append(f"Placer.py exited with code {execution['returncode']}")
    if summary_path is None:
        failures.append("missing_buffering_inner_loop_summary")
    else:
        failures.extend(
            _ours_contract_failures(
                summary,
                profile,
                segment_strategy=segment_strategy,
            )
        )
        if capacity_grid is not None:
            failures.extend(_capacity_contract_failures(summary, capacity_grid))
        if expected_action_count is not None:
            commit = dict(summary.get("commit") or {})
            for key in ("action_count", "attempted_action_count", "accepted_action_count"):
                if int(commit.get(key, -1) or -1) != int(expected_action_count):
                    failures.append(f"{key}_mismatch")
    no_action_commit = execution["status"] == "ok" and _is_no_action_commit(summary)
    commit_audit = None
    if no_action_commit:
        commit_audit = _no_action_audit(
            input_def=def_input,
            expected_buffer_master=str(profile["segment_mode"]["buffer_master"]),
        )
        _write_json(method_root / "commit_action_audit.json", commit_audit)
        _write_json(method_root / "action_audit.json", commit_audit)
    elif committed_def.exists():
        commit_audit = audit_def_mutation(
            input_def=def_input,
            final_def=committed_def,
            expected_buffer_master=str(profile["segment_mode"]["buffer_master"]),
            require_new_buffers=True,
        )
        _write_json(method_root / "commit_action_audit.json", commit_audit)
        if commit_audit["status"] != "pass":
            failures.extend(commit_audit["failures"])
    elif not no_action_commit:
        failures.append("missing_committed_def")

    replay_metrics: dict[str, Any] = {}
    replay_execution: dict[str, Any] = {}
    final_audit = None
    capacity_committed_fidelity = None
    replay_log = method_root / "replay.log"
    if no_action_commit:
        final_audit = commit_audit
    elif not failures and not source_only:
        replay_tcl = method_root / "replay.tcl"
        replay_power = method_root / "power.rpt"
        replay_tcl.write_text(
            generate_openroad_tcl(
                benchmark_root=benchmark_root,
                case=case,
                def_input=committed_def,
                output_def=replay_def,
                power_report=replay_power,
                method="ours_replay",
                buffer_master=str(profile["segment_mode"]["buffer_master"]),
                b1_max_buffer_percent=float(profile["openroad_buffer_only"]["max_buffer_percent"]),
            ),
            encoding="utf-8",
        )
        replay_execution = _run_command(
            _openroad_command(openroad_bin, replay_tcl, openroad_num_threads),
            cwd=method_root,
            log_path=replay_log,
        )
        replay_metrics = _parse_metrics(replay_log)
        if replay_execution["status"] != "ok":
            failures.append(f"OpenROAD replay exited with code {replay_execution['returncode']}")
        elif not replay_def.exists():
            failures.append("missing_replayed_def")
        else:
            final_audit = audit_def_mutation(
                input_def=def_input,
                final_def=replay_def,
                expected_buffer_master=str(profile["segment_mode"]["buffer_master"]),
                require_new_buffers=True,
            )
            _write_json(method_root / "action_audit.json", final_audit)
            if final_audit["status"] != "pass":
                failures.extend(final_audit["failures"])
            elif capacity_grid is not None:
                try:
                    capacity_committed_fidelity = _close_segment_capacity_committed_fidelity(
                        summary=summary,
                        committed_def=replay_def,
                        final_audit=final_audit,
                    )
                except (OSError, ValueError, TypeError) as exc:
                    failures.append(f"committed_capacity_fidelity_failed: {exc}")
                else:
                    if capacity_committed_fidelity.get("status") != "pass":
                        failures.append("committed_capacity_fidelity_failed")
    artifact_path = method_root / "result.json"
    result = _method_result(
        case=case,
        method=method_name,
        input_def=def_input,
        final_def=(
            def_input
            if no_action_commit
            else (committed_def if source_only else replay_def)
        ),
        metrics=replay_metrics,
        runtime_sec=(
            execution.get("runtime_sec", 0.0)
            + (0.0 if source_only else replay_execution.get("runtime_sec", 0.0))
        ),
        audit=commit_audit if source_only or no_action_commit else final_audit,
        artifact_path=artifact_path,
        run_log=(
            method_root / "autodmp.log"
            if source_only
            else (replay_log if replay_log.exists() else method_root / "autodmp.log")
        ),
        failure="; ".join(failures),
    )
    result["commit_action_audit_path"] = str(method_root / "commit_action_audit.json")
    result["autodmp_summary_path"] = str(summary_path or "")
    if no_action_commit:
        result["skip_reason"] = "no_projected_buffer_actions"
        result["topology_mutated"] = False
    _write_json(
        artifact_path,
        {
            "artifact": "equal_spaced_buffering_method_result",
            "artifact_version": 1,
            "result": result,
            "execution": execution,
            "summary_path": str(summary_path or ""),
            "summary": summary,
            "commit_audit": commit_audit,
            "replay_execution": replay_execution,
            "replay_metrics": replay_metrics,
            "final_audit": final_audit,
            "segment_capacity_committed_fidelity": capacity_committed_fidelity,
            "source_only": bool(source_only),
        },
    )
    return result


def _parse_fidelity_counts(raw: str, *, nmax: int) -> tuple[int, ...]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            count = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid fidelity repeater count: {item}") from exc
        if count < 0 or count > int(nmax):
            raise ValueError(f"fidelity repeater count must be in [0, {int(nmax)}]: {count}")
        values.append(count)
    if not values:
        raise ValueError("fidelity count list is empty")
    return tuple(sorted(set(values)))


def _fidelity_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(profile))
    segment = dict(result["segment_mode"])
    segment.update(
        {
            "continuous_steps": 1,
            "continuous_lr": 0.0,
            "z_init": 0.0,
            "integer_projection_interval": 0,
            "one_final_hard_commit": True,
        }
    )
    result["segment_mode"] = segment
    return result


def _selected_device_probe_rows(path: Path, *, segment_id: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    rows = []
    for row in payload.get("rows", []):
        observed = row.get("global_segment_id", row.get("segment_id", -1))
        if int(observed) != int(segment_id):
            continue
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "sense",
                    "segment_id",
                    "global_segment_id",
                    "net_id",
                    "net_name",
                    "z_value",
                    "bsu_index",
                    "segment_parent_slew",
                    "segment_downstream_load",
                    "analytic_transfer_source",
                    "analytic_transfer_repeater_count",
                    "analytic_transfer_segment_delay",
                    "analytic_transfer_output_slew",
                    "analytic_transfer_upstream_visible_input_cap",
                    "analytic_transfer_buffer_input_slew",
                    "analytic_transfer_buffer_output_load",
                    "analytic_transfer_buffer_delay",
                    "analytic_transfer_buffer_output_slew",
                    "analytic_transfer_buffer_input_slews",
                    "analytic_transfer_buffer_output_loads",
                    "analytic_transfer_buffer_delays",
                    "analytic_transfer_buffer_output_slews",
                )
            }
        )
    return rows


def _committed_segment_arc_samples(summary: dict[str, Any], *, segment_id: int) -> list[dict[str, Any]]:
    commit = dict(summary.get("commit") or {})
    raw_artifact_path = str(commit.get("artifact_path", ""))
    artifact_path = Path(raw_artifact_path) if raw_artifact_path else None
    if artifact_path is None or not artifact_path.is_file():
        return []
    payload = _read_json(artifact_path)
    rows = []
    for result in payload.get("results", []):
        if int(result.get("segment_id", -1)) != int(segment_id):
            continue
        diagnostic = dict(result.get("hard_commit_diagnostic") or {})
        samples = dict(diagnostic.get("cell_arc_delay_samples") or {})
        selected = dict(samples.get("selected_max_delay_sample") or {})
        arrival = dict(samples.get("pin_arrival_delta") or {})
        rows.append(
            {
                "action_id": result.get("action_id"),
                "inserted_buffer_name": result.get("inserted_buffer_name"),
                "candidate_location_x_dbu": result.get("candidate_location_x_dbu"),
                "candidate_location_y_dbu": result.get("candidate_location_y_dbu"),
                "segment_split_index": result.get("segment_split_index"),
                "segment_split_count_on_edge": result.get("segment_split_count_on_edge"),
                "segment_split_ratio": result.get("segment_split_ratio"),
                "cell_arc_sample_status": samples.get("status"),
                "buffer_master_name": samples.get("buffer_master_name"),
                "buffer_input_pin_name": samples.get("buffer_input_pin_name"),
                "buffer_output_pin_name": samples.get("buffer_output_pin_name"),
                "input_slew_ps": selected.get("input_slew"),
                "output_load_cap_pf": selected.get("output_load_cap_pf"),
                "liberty_gate_delay_ps": selected.get("liberty_gate_delay"),
                "liberty_output_slew_ps": selected.get("liberty_output_slew"),
                "pin_arrival_delta_ps": {
                    "rise": arrival.get("r_arrival_delta"),
                    "fall": arrival.get("f_arrival_delta"),
                },
            }
        )
    return sorted(rows, key=lambda row: (row.get("segment_split_index", -1), row.get("action_id", -1)))


def _relative_error(predicted: Any, observed: Any) -> float | None:
    predicted_value = _as_float(predicted)
    observed_value = _as_float(observed)
    if predicted_value is None or observed_value is None:
        return None
    return abs(predicted_value - observed_value) / max(abs(observed_value), 1.0e-12)


def _evaluate_fidelity_observations(observations: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    criteria = {
        "max_stage_delay_relative_error": 0.20,
        "max_stage_input_slew_relative_error": 0.10,
        "max_stage_output_slew_relative_error": 0.10,
        "output_load_context": "recorded_only_not_thresholded",
    }
    comparisons = []
    failures = []
    for observation in observations:
        repeater_count = int(observation["repeater_count"])
        if repeater_count == 0:
            comparisons.append({"repeater_count": 0, "status": "pass", "stages": []})
            continue
        rise_rows = [
            row for row in observation.get("pysta_device_probe_rows", []) if row.get("sense") == "rise"
        ]
        committed = list(observation.get("committed_arc_samples", []))
        if len(rise_rows) != 1 or len(committed) != repeater_count:
            failures.append(f"n{repeater_count}_missing_stage_correspondence")
            comparisons.append(
                {
                    "repeater_count": repeater_count,
                    "status": "failed",
                    "failure": failures[-1],
                }
            )
            continue
        probe = rise_rows[0]
        if int(probe.get("analytic_transfer_repeater_count", -1)) != repeater_count:
            failures.append(f"n{repeater_count}_repeater_count_mismatch")
        delays = list(probe.get("analytic_transfer_buffer_delays") or [])
        input_slews = list(probe.get("analytic_transfer_buffer_input_slews") or [])
        output_loads = list(probe.get("analytic_transfer_buffer_output_loads") or [])
        output_slews = list(probe.get("analytic_transfer_buffer_output_slews") or [])
        stage_lists = (delays, input_slews, output_loads, output_slews)
        if any(len(values) != repeater_count for values in stage_lists):
            failures.append(f"n{repeater_count}_pysta_stage_array_length_mismatch")
        stages = []
        for stage_index in range(min(repeater_count, *(len(values) for values in stage_lists))):
            committed_stage = committed[stage_index]
            if stage_index + 1 < repeater_count:
                output_slew_reference = committed[stage_index + 1].get("input_slew_ps")
                output_slew_reference_kind = "next_stage_input_slew_ps"
            else:
                output_slew_reference = committed_stage.get("liberty_output_slew_ps")
                output_slew_reference_kind = "final_stage_liberty_output_slew_ps"
            stage = {
                "stage_index": stage_index,
                "predicted_buffer_delay_ps": delays[stage_index],
                "observed_liberty_gate_delay_ps": committed_stage.get("liberty_gate_delay_ps"),
                "delay_relative_error": _relative_error(
                    delays[stage_index], committed_stage.get("liberty_gate_delay_ps")
                ),
                "predicted_input_slew_ps": input_slews[stage_index],
                "observed_input_slew_ps": committed_stage.get("input_slew_ps"),
                "input_slew_relative_error": _relative_error(
                    input_slews[stage_index], committed_stage.get("input_slew_ps")
                ),
                "predicted_output_slew_ps": output_slews[stage_index],
                "observed_output_slew_ps": output_slew_reference,
                "output_slew_reference": output_slew_reference_kind,
                "output_slew_relative_error": _relative_error(
                    output_slews[stage_index], output_slew_reference
                ),
                "predicted_output_load_pf": output_loads[stage_index],
                "observed_output_net_cap_pf": committed_stage.get("output_load_cap_pf"),
                "output_load_relative_error": _relative_error(
                    output_loads[stage_index], committed_stage.get("output_load_cap_pf")
                ),
            }
            stages.append(stage)
            for metric, limit in (
                ("delay_relative_error", criteria["max_stage_delay_relative_error"]),
                ("input_slew_relative_error", criteria["max_stage_input_slew_relative_error"]),
                ("output_slew_relative_error", criteria["max_stage_output_slew_relative_error"]),
            ):
                value = stage[metric]
                if value is None or value > limit:
                    failures.append(f"n{repeater_count}_stage{stage_index}_{metric}")
        comparisons.append(
            {
                "repeater_count": repeater_count,
                "status": "pass" if not any(item.startswith(f"n{repeater_count}_") for item in failures) else "failed",
                "stages": stages,
            }
        )
    decision = (
        "suitable_for_relaxed_stage_fidelity"
        if not failures
        else "not_suitable_for_relaxed_stage_fidelity"
    )
    return decision, {"criteria": criteria, "failures": failures, "comparisons": comparisons}


def _run_segment_count_fidelity(
    *,
    python_bin: Path,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    run_root: Path,
    profile: dict[str, Any],
    fidelity_segment_id: int,
    counts: tuple[int, ...],
    plan_only: bool,
) -> dict[str, Any]:
    inputs = _case_inputs(benchmark_root, case)
    profile_for_probe = _fidelity_profile(profile)
    fidelity_root = run_root / case / "segment_transfer_fidelity"
    rows = []
    observations = []
    for count in counts:
        count_root = fidelity_root / f"n{count}"
        if count == 0:
            row = _run_openroad_method(
                openroad_bin=openroad_bin,
                benchmark_root=benchmark_root,
                case=case,
                method="b0",
                input_def=inputs["def"],
                case_root=count_root,
                profile=profile_for_probe,
                plan_only=plan_only,
            )
            observations.append(
                {
                    "repeater_count": 0,
                    "row": row,
                    "committed_arc_samples": [],
                    "pysta_device_probe_rows": [],
                }
            )
            rows.append(row)
            continue

        method_root_name = "ours"
        device_probe_path = count_root / method_root_name / "segment_device_probe_forced.json"
        environment = {
            "AIMP_BUFFERING_SEGMENT_FORCE_Z_COMMIT_TOPK": "1",
            "AIMP_BUFFERING_SEGMENT_FORCE_Z_SEGMENT_IDS": str(int(fidelity_segment_id)),
            "AIMP_BUFFERING_SEGMENT_FORCE_Z_VALUE": str(int(count)),
            "AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON": str(device_probe_path),
            "AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_FORCED_ONLY": "1",
            "AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS": str(int(fidelity_segment_id)),
        }
        row = _run_ours(
            python_bin=python_bin,
            openroad_bin=openroad_bin,
            benchmark_root=benchmark_root,
            case=case,
            case_root=count_root,
            profile=profile_for_probe,
            plan_only=plan_only,
            method_name=f"fidelity_n{count}",
            method_root_name=method_root_name,
            environment=environment,
            expected_action_count=int(count),
        )
        raw_summary_path = str(row.get("autodmp_summary_path", ""))
        summary_path = Path(raw_summary_path) if raw_summary_path else None
        summary = (
            _read_json(summary_path).get("summary", {})
            if summary_path is not None and summary_path.is_file()
            else {}
        )
        committed_samples = _committed_segment_arc_samples(
            summary,
            segment_id=int(fidelity_segment_id),
        )
        device_rows = _selected_device_probe_rows(
            device_probe_path,
            segment_id=int(fidelity_segment_id),
        )
        if row["status"] == "pass" and len(committed_samples) != int(count):
            row["status"] = "failed"
            row["failure"] = "; ".join(
                value for value in (row.get("failure"), "committed_arc_sample_count_mismatch") if value
            )
        observations.append(
            {
                "repeater_count": int(count),
                "row": row,
                "committed_arc_samples": committed_samples,
                "pysta_device_probe_rows": device_rows,
            }
        )
        rows.append(row)

    model_fidelity_decision, stage_fidelity = _evaluate_fidelity_observations(observations)
    report = {
        "artifact": "equal_spaced_segment_multicount_fidelity",
        "artifact_version": 1,
        "status": "pass" if all(row["status"] in ("pass", "plan_only") for row in rows) else "failed",
        "scope": "diagnostic_only_not_a_main_table_method",
        "case": case,
        "segment_id": int(fidelity_segment_id),
        "counts": list(counts),
        "input_def": str(inputs["def"]),
        "input_def_sha256": _sha256(inputs["def"]),
        "frozen_profile": profile,
        "probe_overrides": profile_for_probe["segment_mode"],
        "model_fidelity_decision": model_fidelity_decision,
        "stage_fidelity": stage_fidelity,
        "observations": observations,
    }
    _write_json(fidelity_root / "fidelity_report.json", report)
    return report


def _write_table(path: Path, rows: list[dict[str, Any]], run_id: str) -> None:
    lines = [
        "# Equal-Spaced Buffering Run",
        "",
        f"Run ID: `{run_id}`",
        "",
        "| Case | Method | Status | WNS (ns) | TNS (ns) | Delta TNS vs B0 (ns) | Buffers | Power (W) | Area | Slew Violation | Cap Violation | Runtime (s) | Action Audit | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    b0_tns = {row["case"]: row.get("tns_ns") for row in rows if row["method"] == "b0"}
    for row in rows:
        tns = _as_float(row.get("tns_ns"))
        base = _as_float(b0_tns.get(row["case"]))
        row["delta_tns_ns"] = None if tns is None or base is None else tns - base
        values = (
            row["case"],
            row["method"],
            row["status"],
            "" if row.get("wns_ns") is None else f"{row['wns_ns']:.6f}",
            "" if tns is None else f"{tns:.6f}",
            "" if row["delta_tns_ns"] is None else f"{row['delta_tns_ns']:.6f}",
            "" if row.get("buffer_count") is None else str(row["buffer_count"]),
            "" if row.get("power_w") is None else f"{row['power_w']:.6g}",
            "" if row.get("area") is None else f"{row['area']:.6g}",
            "" if row.get("slew_violation") is None else f"{row['slew_violation']:.6g}",
            "" if row.get("cap_violation") is None else f"{row['cap_violation']:.6g}",
            "" if row.get("runtime_sec") is None else f"{row['runtime_sec']:.3f}",
            row.get("action_audit_status", "") or "",
            row.get("failure", "") or "",
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], run_id: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {"date": date.today().isoformat(), "run_id": run_id, **row}
            writer.writerow(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--openroad-num-threads", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--logical-gpu-id", type=int)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--method", dest="methods", action="append")
    parser.add_argument(
        "--capacity-grid",
        choices=("4", "8"),
        help=(
            "Run ours with fixed-placement segment capacity control on a "
            "4x4 or 8x8 base-bin coarsening grid."
        ),
    )
    parser.add_argument(
        "--segment-strategy",
        choices=("continuous", "discrete_net_gradient"),
        default="continuous",
        help="Choose continuous relaxed buffering or Route B discrete net-gradient scheduling.",
    )
    parser.add_argument(
        "--fidelity-segment-id",
        type=int,
        help="Run the diagnostic committed n=0..Nmax probe for one segment; not a main-table method.",
    )
    parser.add_argument(
        "--fidelity-counts",
        default="0,1,2,3",
        help="Comma-separated integer repeater counts for --fidelity-segment-id.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--def-input",
        type=Path,
        help="override the fixed-position input DEF for one selected case",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.openroad_num_threads is not None and args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    profile = _load_profile(args.profile.resolve())
    logical_gpu_id = (
        int(args.logical_gpu_id)
        if args.logical_gpu_id is not None
        else int(profile["segment_mode"].get("gpu_id", 0))
    )
    if logical_gpu_id < 0:
        raise ValueError("--logical-gpu-id must be nonnegative")
    execution_environment = (
        None
        if args.cuda_visible_devices is None
        else {"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)}
    )
    if args.segment_strategy == "discrete_net_gradient" and args.capacity_grid is not None:
        raise ValueError("--segment-strategy discrete_net_gradient does not allow --capacity-grid")
    cases = _resolve_selected(args.cases, tuple(profile["suite"]))
    methods = _resolve_selected(args.methods, EXPERIMENT_METHODS)
    if args.def_input is not None:
        if len(cases) != 1:
            raise ValueError("--def-input requires exactly one selected case")
        if methods != ["ours"]:
            raise ValueError("--def-input is supported only with --method ours")
        args.def_input = args.def_input.resolve()
        if not args.plan_only and not args.def_input.is_file():
            raise FileNotFoundError(args.def_input)
    if args.fidelity_segment_id is not None:
        if args.methods:
            raise ValueError("--method cannot be used with --fidelity-segment-id")
        if len(cases) != 1:
            raise ValueError("--fidelity-segment-id requires exactly one --case")
        counts = _parse_fidelity_counts(
            args.fidelity_counts,
            nmax=int(profile["segment_mode"]["max_repeaters_per_segment"]),
        )
    else:
        counts = ()
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    run_root = output_root / args.run_id
    if not args.plan_only:
        if not args.openroad_bin.exists():
            raise FileNotFoundError(f"missing OpenROAD binary: {args.openroad_bin}")
        if not args.python_bin.exists():
            raise FileNotFoundError(f"missing Python binary: {args.python_bin}")
    manifest = {
        "artifact_version": 1,
        "profile_path": str(args.profile.resolve()),
        "profile_sha256": _sha256(args.profile.resolve()),
        "profile": profile,
        "autodmp_root": str(AUTODMP_ROOT),
        "benchmark_root": str(benchmark_root),
        "openroad_bin": str(args.openroad_bin),
        "python_bin": str(args.python_bin),
        "tool_versions": (
            {}
            if args.plan_only
            else {
                "openroad": _tool_version([str(args.openroad_bin), "-version"]),
                "python": _tool_version([str(args.python_bin), "--version"]),
            }
        ),
        "cases": {},
        "methods": methods,
        "def_only_replay": True,
        "capacity_grid": args.capacity_grid,
        "segment_strategy": args.segment_strategy,
        "cuda_visible_devices": args.cuda_visible_devices,
        "logical_gpu_id": logical_gpu_id,
        "source_only": bool(args.source_only),
        "openroad_num_threads": args.openroad_num_threads,
    }
    for case in cases:
        inputs = _case_inputs(benchmark_root, case)
        case_input_def = args.def_input if args.def_input is not None else inputs["def"]
        manifest["cases"][case] = {
            "input_def": str(case_input_def),
            "input_def_sha256": (
                "plan_only"
                if args.plan_only and not case_input_def.is_file()
                else _sha256(case_input_def)
            ),
            "sdc": str(inputs["sdc"]),
            "sdc_sha256": _sha256(inputs["sdc"]),
        }
    _write_json(run_root / "manifest.json", manifest)

    if args.fidelity_segment_id is not None:
        report = _run_segment_count_fidelity(
            python_bin=args.python_bin,
            openroad_bin=args.openroad_bin,
            benchmark_root=benchmark_root,
            case=cases[0],
            run_root=run_root,
            profile=profile,
            fidelity_segment_id=int(args.fidelity_segment_id),
            counts=counts,
            plan_only=bool(args.plan_only),
        )
        print(f"fidelity_report: {run_root / cases[0] / 'segment_transfer_fidelity' / 'fidelity_report.json'}")
        return 0 if report["status"] in ("pass", "plan_only") else 1

    rows: list[dict[str, Any]] = []
    for case in cases:
        inputs = _case_inputs(benchmark_root, case)
        case_root = run_root / case
        for method in methods:
            if method in ("b0", "b1", "b1_rd_rt"):
                row = _run_openroad_method(
                    openroad_bin=args.openroad_bin,
                    benchmark_root=benchmark_root,
                    case=case,
                    method=method,
                    input_def=inputs["def"],
                    case_root=case_root,
                    profile=profile,
                    plan_only=bool(args.plan_only),
                    source_only=bool(args.source_only),
                    openroad_num_threads=args.openroad_num_threads,
                )
            else:
                row = _run_ours(
                    python_bin=args.python_bin,
                    openroad_bin=args.openroad_bin,
                    benchmark_root=benchmark_root,
                    case=case,
                    case_root=case_root,
                    profile=profile,
                    plan_only=bool(args.plan_only),
                    capacity_grid=args.capacity_grid,
                    segment_strategy=args.segment_strategy,
                    logical_gpu_id=logical_gpu_id,
                    environment=execution_environment,
                    source_only=bool(args.source_only),
                    def_input=args.def_input,
                    openroad_num_threads=args.openroad_num_threads,
                )
            rows.append(row)
            print(f"{case} {method}: {row['status']} {row.get('failure', '')}")
    _write_csv(run_root / "comparison.csv", rows, args.run_id)
    _write_table(run_root / "experiment_table.md", rows, args.run_id)
    _write_json(run_root / "comparison.json", {"manifest": manifest, "rows": rows})
    print(f"run_root: {run_root}")
    return 0 if all(row["status"] in ("pass", "skipped", "plan_only") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
