#!/usr/bin/env python3
"""Run ICCAD24 staged diff-TDP + diff-sizing + OpenROAD repair experiments."""

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

REPAIR_MODES = ("none", "repair_design", "repair_timing_setup")
OPENROAD_METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.+)$")
VIOLATION_RE = re.compile(r"^.* +([-\.0-9]+) +\(VIOLATED\)")
SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
COMMON_REGRESSION_DIR = AUTODMP_ROOT / "test/regression/common"
if str(COMMON_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_REGRESSION_DIR))

from placement_validation import (  # noqa: E402
    detailed_placement_tcl_lines,
    inventory_tcl_proc_lines,
    macro_halo_blockage_tcl_proc_lines,
    qualify_placement,
    read_inventory,
)

DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "staged_joint"
TOOLS_RELEASE_PROFILE = (
    AUTODMP_ROOT
    / "test/regression/iccad24/tools_paper_four_table/profiles/tools_release_v1.json"
)
PLACEMENT_RUNNER = (
    AUTODMP_ROOT
    / "test/regression/iccad24/joint_place_sizing_buffering/run_random_init_validation.py"
)
EQUAL_BUFFERING_RUNNER = (
    AUTODMP_ROOT / "test/regression/iccad24/equal_spaced_buffering/run.py"
)
EQUAL_BUFFERING_PROFILE = (
    AUTODMP_ROOT
    / "test/regression/iccad24/equal_spaced_buffering/params/route_b_allcases_0p1pct.json"
)
TOOLS_RELEASE_METHODS = ("autodmp_staged_psb",)

CSV_COLUMNS = (
    "date",
    "run_id",
    "case",
    "flow_name",
    "status",
    "failed_stage",
    "failure",
    "repair_mode",
    "stage1_status",
    "stage1_runtime_sec",
    "stage1_output_def",
    "stage1_log_path",
    "stage1_pysta_wns",
    "stage1_pysta_tns",
    "stage1_final_overflow",
    "stage1_actual_iterations",
    "stage1_entered_legalization",
    "tdp_gate_open_count",
    "tdp_gate_closed_count",
    "tdp_net_weight_reset_count",
    "stage2_status",
    "stage2_runtime_sec",
    "stage2_output_def",
    "stage2_log_path",
    "pre_repair_wns",
    "pre_repair_tns",
    "post_repair_wns",
    "post_repair_tns",
    "delta_tns_from_pre_repair",
    "slew_violation",
    "cap_violation",
    "instance_count",
    "area",
    "runtime_total_sec",
    "repair_summary_path",
    "command_plan_path",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tools_release_profile(path: Path) -> dict[str, Any]:
    profile = _read_json(path)
    if profile.get("artifact") != "autodmp_tools_paper_release_profile":
        raise ValueError("invalid tools-release profile artifact")
    if int(profile.get("artifact_version", -1)) != 1:
        raise ValueError("unsupported tools-release profile version")
    if tuple(profile.get("methods", {}).get("full_flow", ())) != (
        "or_native_full_flow",
        "autodmp_staged_psb",
        "autodmp_coordinated_psb",
    ):
        raise ValueError("tools-release full-flow method matrix mismatch")
    if (
        dict(profile.get("track_a") or {}).get(
            "detailed_placement_search_window"
        )
        != "full_core"
    ):
        raise ValueError("tools-release detailed-placement policy mismatch")
    return profile


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _delta(lhs: Any, rhs: Any) -> float | None:
    lhs_float = _as_float(lhs)
    rhs_float = _as_float(rhs)
    if lhs_float is None or rhs_float is None:
        return None
    return lhs_float - rhs_float


def _tcl_quote(value: str | Path) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


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


def _append_design_inputs(command: list[str], *, benchmark_root: Path, case: str, def_input: Path) -> None:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    command.extend(
        [
            "--workspace",
            str(paths["workspace"]),
            "--base-design-name",
            case,
            "--def-input",
            str(def_input),
            "--verilog-input",
            str(paths["verilog"]),
            "--sdc",
            str(paths["sdc"]),
            "--rc-tcl",
            str(tech["rc_tcl"]),
            "--tech-lef",
            str(tech["tech_lef"]),
        ]
    )
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])


def build_autodmp_stage_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    flow_kind: str,
    def_input: Path,
    output_def: Path,
    iterations: int,
    place_io_engine: str | None = None,
    diff_tdp_enabled: bool = True,
    tdp_enable_threshold: float | None = 0.2,
    enable_net_weighting: int | None = None,
    extra_args: list[str] | None = None,
    dry_run_config: bool = False,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(paths["params_json"]),
        "--flow-kind",
        flow_kind,
        "--result-dir",
        str(result_dir),
        "--output-def",
        str(output_def),
        "--iterations",
        str(int(iterations)),
    ]
    if place_io_engine:
        command.extend(["--place-io-engine", str(place_io_engine)])
    _append_design_inputs(command, benchmark_root=benchmark_root, case=case, def_input=def_input)
    if flow_kind == "placement":
        command.extend(
            [
                "--with-sta",
                "--diff-timing-driven-placement",
                "1" if diff_tdp_enabled else "0",
                "--timing-placement-carrier",
                "gradient_net_weight",
                "--timing-gradient-net-weight-scale",
                "0.4",
                "--timing-gradient-net-weight-max",
                "2.0",
                "--timing-topology-refresh-interval",
                "10",
            ]
        )
        if tdp_enable_threshold is not None:
            command.extend(
                [
                    "--timing-topology-enable-overflow-threshold",
                    str(float(tdp_enable_threshold)),
                ]
            )
        if enable_net_weighting is not None:
            command.extend(["--enable-net-weighting", str(int(enable_net_weighting))])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    if dry_run_config:
        command.append("--dry-run-config")
    return command


def build_tools_release_placement_command(
    *,
    python_bin: Path,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    output_root: Path,
    profile: dict[str, Any],
    physical_gpu: str,
    logical_gpu_id: int,
    openroad_num_threads: int,
) -> list[str]:
    placement = dict(profile["placement"])
    pin2pin = dict(placement["pin2pin"])
    r0 = dict(profile["input_domains"]["R0"])
    return [
        str(python_bin),
        str(PLACEMENT_RUNNER),
        "--benchmark-root",
        str(benchmark_root),
        "--output-root",
        str(output_root),
        "--run-id",
        "run",
        "--python-bin",
        str(python_bin),
        "--openroad-bin",
        str(openroad_bin),
        "--openroad-num-threads",
        str(int(openroad_num_threads)),
        "--case",
        case,
        "--method",
        "TDP-pin2pin",
        "--outer-iterations",
        "1",
        "--iterations",
        str(int(placement["max_steps"])),
        "--seed",
        str(int(r0["seed"])),
        "--gpu-id",
        str(int(logical_gpu_id)),
        "--cuda-visible-devices",
        str(physical_gpu),
        "--placement-optimizer",
        str(placement["optimizer"]),
        "--enable-fillers",
        "--timing-topology-enable-overflow-threshold",
        str(pin2pin["activation_overflow"]),
        "--pin2pin-weight",
        str(pin2pin["weight"]),
        "--pin2pin-min-weight",
        str(pin2pin["min_weight"]),
        "--pin2pin-max-weight",
        str(pin2pin["max_weight"]),
        "--pin2pin-accumulate-weight",
        str(pin2pin["accumulate_weight"]),
        "--net-weighting-npaths",
        str(pin2pin["path_limit"]),
        "--plot",
        "0",
        "--no-track-b",
        "--source-only",
    ]


def build_tools_release_sizing_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    def_input: Path,
    output_def: Path,
    profile: dict[str, Any],
    logical_gpu_id: int,
) -> list[str]:
    sizing = dict(profile["sizing"])
    return build_autodmp_stage_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=result_dir,
        flow_kind="sizing",
        def_input=def_input,
        output_def=output_def,
        iterations=int(sizing["iterations"]),
        place_io_engine="openroad",
        extra_args=[
            "--with-sta",
            "--gpu",
            "1",
            "--gpu-id",
            str(int(logical_gpu_id)),
            "--legalize",
            "0",
            "--enable-fillers",
            "0",
            "--discrete-gradient-topk-up-percent",
            str(float(sizing["up_percent"])),
            "--discrete-gradient-topk-down-percent",
            str(float(sizing["down_percent"])),
            "--plot",
            "0",
        ],
    )


def build_tools_release_buffering_command(
    *,
    python_bin: Path,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    output_root: Path,
    def_input: Path,
    physical_gpu: str,
    logical_gpu_id: int,
) -> list[str]:
    return [
        str(python_bin),
        str(EQUAL_BUFFERING_RUNNER),
        "--profile",
        str(EQUAL_BUFFERING_PROFILE),
        "--benchmark-root",
        str(benchmark_root),
        "--output-root",
        str(output_root),
        "--run-id",
        "run",
        "--python-bin",
        str(python_bin),
        "--openroad-bin",
        str(openroad_bin),
        "--case",
        case,
        "--method",
        "ours",
        "--segment-strategy",
        "discrete_net_gradient",
        "--cuda-visible-devices",
        str(physical_gpu),
        "--logical-gpu-id",
        str(int(logical_gpu_id)),
        "--def-input",
        str(def_input),
        "--source-only",
    ]


def generate_tools_checkpoint_tcl(
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
    output_def: Path,
    input_inventory: Path,
    evaluated_inventory: Path,
    dpl_report: Path,
    detailed_placement_search_window: str,
) -> str:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    lines = [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        *inventory_tcl_proc_lines(),
        *macro_halo_blockage_tcl_proc_lines(),
        "set start_us [clock microseconds]",
        f"read_lef {_tcl_quote(tech['tech_lef'])}",
    ]
    for lef in tech["lef"]:
        lines.append(f"read_lef {_tcl_quote(lef)}")
    for liberty in tech["lib"]:
        lines.append(f"read_liberty {_tcl_quote(liberty)}")
    lines.extend(
        (
            f"read_def -continue_on_errors {_tcl_quote(def_input)}",
            f"read_sdc {_tcl_quote(paths['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            f"dump_inventory {_tcl_quote(input_inventory)}",
            f"file delete -force {_tcl_quote(dpl_report)}",
            "set checkpoint_macro_halo_blockages [create_macro_halo_blockages]",
            'emit_metric "macro_halo_blockage_count" [llength $checkpoint_macro_halo_blockages]',
            *detailed_placement_tcl_lines(
                search_window=detailed_placement_search_window
            ),
            f'if {{[catch {{check_placement -verbose -report_file_name {_tcl_quote(dpl_report)}}} message]}} {{ emit_metric "placement_valid" 0 }} else {{ emit_metric "placement_valid" 1 }}',
            f"dump_inventory {_tcl_quote(evaluated_inventory)}",
            "destroy_macro_halo_blockages $checkpoint_macro_halo_blockages",
            "estimate_parasitics -placement",
            'if {[llength [info commands update_timing]]} { update_timing } else { catch {worst_slack -max} }',
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def repair_command_for_mode(repair_mode: str) -> str | None:
    if repair_mode == "none":
        return None
    if repair_mode == "repair_design":
        return "repair_design"
    if repair_mode == "repair_timing_setup":
        return "repair_timing -setup"
    raise ValueError(f"unsupported repair_mode: {repair_mode}")


def repair_mode_choices() -> tuple[str, ...]:
    return REPAIR_MODES


def generate_openroad_repair_tcl(
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
    output_dir: Path,
    repair_mode: str,
    output_def: Path | None = None,
    run_timing_driven_placement: bool = False,
) -> str:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    repair_command = repair_command_for_mode(repair_mode)
    lines = [
        "proc emit_metric {name value} { puts \"METRIC|$name|$value\" }",
        "proc safe_metric {name script default_value} {",
        "  if {[catch {uplevel 1 $script} value]} { emit_metric $name $default_value } else { emit_metric $name $value }",
        "}",
        "proc refresh_timing {label} {",
        "  if {[llength [info commands update_timing]]} {",
        "    update_timing",
        "    emit_metric \"${label}_timing_refresh\" \"update_timing\"",
        "  } else {",
        "    safe_metric \"${label}_timing_refresh_wns_probe\" {worst_slack -max} \"\"",
        "    emit_metric \"${label}_timing_refresh\" \"slack_probe\"",
        "  }",
        "}",
        "proc sum_check_type_violation {check_flag} {",
        "  set tmp_report [file join [pwd] \"openroad_[pid]_${check_flag}.rpt\"]",
        "  if {[catch {eval report_check_types $check_flag -no_line_splits -digits 6 > $tmp_report} err]} {",
        "    puts stderr \"WARN: failed to report $check_flag violations: $err\"",
        "    return 0.0",
        "  }",
        "  set total 0.0",
        "  set fp [open $tmp_report r]",
        "  set file_data [split [read $fp] \\n]",
        "  close $fp",
        "  file delete -force $tmp_report",
        "  foreach line $file_data {",
        f"    if {{[regexp {_tcl_quote(VIOLATION_RE.pattern)} $line -> slack_val]}} {{",
        "      set total [expr {$total - $slack_val}]",
        "    }",
        "  }",
        "  return $total",
        "}",
        "set start_us [clock microseconds]",
        f"read_lef {_tcl_quote(tech['tech_lef'])}",
    ]
    for lef in tech["lef"]:
        lines.append(f"read_lef {_tcl_quote(lef)}")
    for lib in tech["lib"]:
        lines.append(f"read_liberty {_tcl_quote(lib)}")
    lines.extend(
        [
            f"read_def -continue_on_errors {_tcl_quote(def_input)}",
            f"read_verilog {_tcl_quote(paths['verilog'])}",
            f"read_sdc {_tcl_quote(paths['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
        ]
    )
    if run_timing_driven_placement:
        lines.extend(
            [
                "emit_metric \"baseline_timing_driven_placement_requested\" 1",
                "global_placement -timing_driven -density 0.8 -init_density_penalty 0.01",
                "detailed_placement",
                "estimate_parasitics -placement",
            ]
        )
    else:
        lines.append("emit_metric \"baseline_timing_driven_placement_requested\" 0")
    lines.extend(
        [
            "refresh_timing \"pre_repair\"",
            'safe_metric "pre_repair_wns" {worst_slack -max} ""',
            'safe_metric "pre_repair_tns" {total_negative_slack -max} ""',
        ]
    )
    if repair_command is not None:
        lines.extend(
            [
                f"set repair_start_us [clock microseconds]",
                repair_command,
                "emit_metric \"repair_runtime_sec\" [expr {([clock microseconds] - $repair_start_us) / 1000000.0}]",
                "estimate_parasitics -placement",
            ]
        )
    else:
        lines.extend(
            [
                "emit_metric \"repair_runtime_sec\" 0.0",
            ]
        )
    lines.extend(
        [
            "refresh_timing \"post_repair\"",
            'safe_metric "post_repair_wns" {worst_slack -max} ""',
            'safe_metric "post_repair_tns" {total_negative_slack -max} ""',
            'emit_metric "slew_violation" [sum_check_type_violation "-max_slew"]',
            'emit_metric "cap_violation" [sum_check_type_violation "-max_capacitance"]',
            'safe_metric "instance_count" {llength [get_cells *]} ""',
            'safe_metric "area" {rsz::design_area} ""',
            f"report_power > {_tcl_quote(output_dir / f'{case}_power.rpt')}",
        ]
    )
    if output_def is not None:
        lines.append(f"write_def {_tcl_quote(output_def)}")
    lines.extend(
        [
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_openroad_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in log_text.splitlines():
        match = OPENROAD_METRIC_RE.match(line.strip())
        if not match:
            continue
        key, raw_value = match.groups()
        parsed = _as_float(raw_value)
        metrics[key] = raw_value if parsed is None else parsed
    return metrics


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


def run_stage_command(command: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> dict[str, Any]:
    started = time.perf_counter()
    if dry_run:
        return {
            "status": "dry_run",
            "returncode": 0,
            "runtime_sec": 0.0,
            "log_path": str(log_path),
        }
    returncode = run_command(command, cwd=cwd, log_path=log_path)
    return {
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "runtime_sec": time.perf_counter() - started,
        "failure": "" if returncode == 0 else f"command exited with code {returncode}",
        "log_path": str(log_path),
    }


def read_stage1_placement_debug_summary(result_dir: Path, case: str) -> dict[str, Any]:
    summary_path = result_dir / f"{case}_placement_debug_summary.json"
    summary = _read_json(summary_path)
    if summary:
        summary["summary_path"] = str(summary_path)
    return summary


def run_openroad_tcl(
    *,
    openroad_bin: Path,
    tcl_path: Path,
    log_path: Path,
    dry_run: bool,
    openroad_num_threads: int | None = None,
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "dry_run",
            "returncode": 0,
            "runtime_sec": 0.0,
            "metrics": {},
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
        }
    if not openroad_bin.exists():
        return {
            "status": "unsupported",
            "failure": f"missing OpenROAD binary: {openroad_bin}",
            "returncode": None,
            "runtime_sec": 0.0,
            "metrics": {},
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
        }
    started = time.perf_counter()
    command = [str(openroad_bin)]
    if openroad_num_threads is not None:
        command.extend(("-threads", str(int(openroad_num_threads))))
    command.extend(("-exit", str(tcl_path)))
    returncode = run_command(command, cwd=tcl_path.parent, log_path=log_path)
    runtime_sec = time.perf_counter() - started
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    status = "ok" if returncode == 0 else "failed"
    return {
        "status": status,
        "failure": "" if status == "ok" else f"OpenROAD exited with code {returncode}",
        "returncode": returncode,
        "runtime_sec": runtime_sec,
        "metrics": parse_openroad_metrics(log_text),
        "tcl_path": str(tcl_path),
        "log_path": str(log_path),
    }


def qualify_tools_checkpoint(
    stage: dict[str, Any],
    *,
    input_inventory: Path,
    evaluated_inventory: Path,
    dpl_report: Path,
) -> dict[str, Any]:
    result = dict(stage)
    metrics = dict(result.get("metrics") or {})
    required = (input_inventory, evaluated_inventory)
    if result.get("status") != "ok" or any(not path.is_file() for path in required):
        result["placement_validation"] = {
            "effective_placement_valid": 0,
            "mode": "missing_checkpoint_artifacts",
        }
        return result
    try:
        raw_placement_valid = int(metrics.get("placement_valid", 0) or 0)
        validation = qualify_placement(
            raw_placement_valid=raw_placement_valid,
            reference_inventory=read_inventory(input_inventory),
            evaluated_inventory=read_inventory(evaluated_inventory),
            dpl_report_path=dpl_report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["placement_validation"] = {
            "effective_placement_valid": 0,
            "mode": "checkpoint_validation_error",
            "error": str(error),
        }
        return result
    metrics["placement_valid_raw"] = raw_placement_valid
    metrics["placement_valid"] = int(validation["effective_placement_valid"])
    result["metrics"] = metrics
    result["placement_validation"] = validation
    return result


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    return _as_float(metrics.get(name))


def build_row(
    *,
    date: str,
    run_id: str,
    case: str,
    flow_name: str,
    repair_mode: str,
    stage1: dict[str, Any] | None,
    stage2: dict[str, Any] | None,
    repair: dict[str, Any],
    runtime_total_sec: float,
    command_plan_path: Path,
    stage1_output_def: Path | None = None,
    stage2_output_def: Path | None = None,
) -> dict[str, Any]:
    metrics = repair.get("metrics") or {}
    stage1_debug = {}
    if stage1 is not None:
        stage1_debug = dict(stage1.get("placement_debug_summary") or {})
    gate = dict(stage1_debug.get("diff_tdp_gate_summary") or {})
    status = repair.get("status")
    failed_stage = ""
    failure = repair.get("failure", "")
    for label, stage in (("stage1", stage1), ("stage2", stage2)):
        if stage is not None and stage.get("status") not in ("ok", "dry_run"):
            status = "failed"
            failed_stage = label
            failure = stage.get("failure", "")
            break
    if not failed_stage and repair.get("status") not in ("ok", "dry_run"):
        failed_stage = "repair"
    pre_tns = _metric(metrics, "pre_repair_tns")
    post_tns = _metric(metrics, "post_repair_tns")
    return {
        "date": date,
        "run_id": run_id,
        "case": case,
        "flow_name": flow_name,
        "status": status,
        "failed_stage": failed_stage,
        "failure": failure,
        "repair_mode": repair_mode,
        "stage1_status": "" if stage1 is None else stage1.get("status"),
        "stage1_runtime_sec": "" if stage1 is None else stage1.get("runtime_sec"),
        "stage1_output_def": "" if stage1_output_def is None else str(stage1_output_def),
        "stage1_log_path": "" if stage1 is None else stage1.get("log_path", ""),
        "stage1_pysta_wns": stage1_debug.get("stage1_final_pysta_wns_ns"),
        "stage1_pysta_tns": stage1_debug.get("stage1_final_pysta_tns_ns"),
        "stage1_final_overflow": stage1_debug.get("final_overflow"),
        "stage1_actual_iterations": stage1_debug.get("actual_iterations"),
        "stage1_entered_legalization": stage1_debug.get("entered_legalization"),
        "tdp_gate_open_count": gate.get("open_count"),
        "tdp_gate_closed_count": gate.get("closed_count"),
        "tdp_net_weight_reset_count": gate.get("net_weight_reset_count"),
        "stage2_status": "" if stage2 is None else stage2.get("status"),
        "stage2_runtime_sec": "" if stage2 is None else stage2.get("runtime_sec"),
        "stage2_output_def": "" if stage2_output_def is None else str(stage2_output_def),
        "stage2_log_path": "" if stage2 is None else stage2.get("log_path", ""),
        "pre_repair_wns": _metric(metrics, "pre_repair_wns"),
        "pre_repair_tns": pre_tns,
        "post_repair_wns": _metric(metrics, "post_repair_wns"),
        "post_repair_tns": post_tns,
        "delta_tns_from_pre_repair": _delta(post_tns, pre_tns),
        "slew_violation": _metric(metrics, "slew_violation"),
        "cap_violation": _metric(metrics, "cap_violation"),
        "instance_count": _metric(metrics, "instance_count"),
        "area": _metric(metrics, "area"),
        "runtime_total_sec": runtime_total_sec,
        "repair_summary_path": repair.get("summary_path", ""),
        "command_plan_path": str(command_plan_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_command(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shlex.join(command) + "\n", encoding="utf-8")


def run_tools_release_staged(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    run_root: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    method_id = "autodmp_staged_psb"
    method_root = run_root / case / method_id
    method_root.mkdir(parents=True, exist_ok=True)
    plan_only = bool(args.dry_run_config)
    failures: list[str] = []
    stages: dict[str, Any] = {}

    stage1_root = method_root / "stage1_placement"
    stage1_source_root = stage1_root / "source"
    stage1_command = build_tools_release_placement_command(
        python_bin=args.python_bin,
        openroad_bin=args.openroad_bin,
        benchmark_root=benchmark_root,
        case=case,
        output_root=stage1_source_root,
        profile=profile,
        physical_gpu=str(args.cuda_visible_devices),
        logical_gpu_id=int(args.logical_gpu_id),
        openroad_num_threads=int(args.openroad_num_threads or 32),
    )
    _write_command(stage1_root / "command.txt", stage1_command)
    stage1_case_root = stage1_source_root / "run" / case
    r0_def = stage1_case_root / "r0" / "R0.def"
    stage1_raw_def = stage1_case_root / "TDP-pin2pin" / f"{case}_raw.def"
    stage1_summary_path = stage1_case_root / "case_summary.json"

    stage1_legal_root = method_root / "stage1_legalize"
    stage1_legal_def = stage1_legal_root / f"{case}_stage1_legalized.def"
    stage1_legal_tcl = stage1_legal_root / "legalize.tcl"
    stage1_input_inventory = stage1_legal_root / "input_inventory.tsv"
    stage1_evaluated_inventory = stage1_legal_root / "evaluated_inventory.tsv"
    stage1_dpl_report = stage1_legal_root / "dpl_placement_report.json"
    stage1_legal_tcl.parent.mkdir(parents=True, exist_ok=True)
    stage1_legal_tcl.write_text(
        generate_tools_checkpoint_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=stage1_raw_def,
            output_def=stage1_legal_def,
            input_inventory=stage1_input_inventory,
            evaluated_inventory=stage1_evaluated_inventory,
            dpl_report=stage1_dpl_report,
            detailed_placement_search_window=profile["track_a"][
                "detailed_placement_search_window"
            ],
        ),
        encoding="utf-8",
    )

    stage2_root = method_root / "stage2_sizing"
    stage2_raw_def = stage2_root / f"{case}_stage2_sizing_raw.def"
    stage2_command = build_tools_release_sizing_command(
        python_bin=args.python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=stage2_root / "autodmp_result",
        def_input=stage1_legal_def,
        output_def=stage2_raw_def,
        profile=profile,
        logical_gpu_id=int(args.logical_gpu_id),
    )
    _write_command(stage2_root / "command.txt", stage2_command)

    stage2_legal_root = method_root / "stage2_legalize"
    stage2_legal_def = stage2_legal_root / f"{case}_stage2_legalized.def"
    stage2_legal_tcl = stage2_legal_root / "legalize.tcl"
    stage2_input_inventory = stage2_legal_root / "input_inventory.tsv"
    stage2_evaluated_inventory = stage2_legal_root / "evaluated_inventory.tsv"
    stage2_dpl_report = stage2_legal_root / "dpl_placement_report.json"
    stage2_legal_tcl.parent.mkdir(parents=True, exist_ok=True)
    stage2_legal_tcl.write_text(
        generate_tools_checkpoint_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=stage2_raw_def,
            output_def=stage2_legal_def,
            input_inventory=stage2_input_inventory,
            evaluated_inventory=stage2_evaluated_inventory,
            dpl_report=stage2_dpl_report,
            detailed_placement_search_window=profile["track_a"][
                "detailed_placement_search_window"
            ],
        ),
        encoding="utf-8",
    )

    stage3_root = method_root / "stage3_buffering"
    stage3_source_root = stage3_root / "source"
    stage3_command = build_tools_release_buffering_command(
        python_bin=args.python_bin,
        openroad_bin=args.openroad_bin,
        benchmark_root=benchmark_root,
        case=case,
        output_root=stage3_source_root,
        def_input=stage2_legal_def,
        physical_gpu=str(args.cuda_visible_devices),
        logical_gpu_id=int(args.logical_gpu_id),
    )
    _write_command(stage3_root / "command.txt", stage3_command)
    stage3_method_root = stage3_source_root / "run" / case / "ours"
    stage3_result_path = stage3_method_root / "result.json"
    final_raw_def = stage3_method_root / f"{case}_ours_committed.def"

    command_plan = {
        "artifact": "tools_release_staged_command_plan",
        "artifact_version": 1,
        "case": case,
        "method_id": method_id,
        "order": [
            "pin2pin_placement",
            "placement_checkpoint_legalization",
            "fixed_position_sizing",
            "sizing_checkpoint_legalization",
            "equal_spaced_route_b",
        ],
        "commands": {
            "pin2pin_placement": stage1_command,
            "placement_checkpoint_legalization": [
                str(args.openroad_bin),
                "-exit",
                str(stage1_legal_tcl),
            ],
            "fixed_position_sizing": stage2_command,
            "sizing_checkpoint_legalization": [
                str(args.openroad_bin),
                "-exit",
                str(stage2_legal_tcl),
            ],
            "equal_spaced_route_b": stage3_command,
        },
        "handoffs": [
            [str(stage1_raw_def), str(stage1_legal_def)],
            [str(stage1_legal_def), str(stage2_raw_def)],
            [str(stage2_raw_def), str(stage2_legal_def)],
            [str(stage2_legal_def), str(final_raw_def)],
        ],
    }
    _write_json(method_root / "command_plan.json", command_plan)

    if not plan_only:
        stages["pin2pin_placement"] = run_stage_command(
            stage1_command,
            cwd=AUTODMP_ROOT,
            log_path=stage1_root / "launcher.log",
            dry_run=False,
        )
        stage1_summary = _read_json(stage1_summary_path)
        stage1_method = dict(
            dict(stage1_summary.get("methods") or {}).get("TDP-pin2pin") or {}
        )
        if (
            stages["pin2pin_placement"].get("status") != "ok"
            or stage1_summary.get("status") != "pass"
            or stage1_method.get("status") != "pass"
            or not r0_def.is_file()
            or not stage1_raw_def.is_file()
        ):
            failures.append("pin2pin_placement_failed")

        if not failures:
            stages["placement_checkpoint_legalization"] = run_openroad_tcl(
                openroad_bin=args.openroad_bin,
                tcl_path=stage1_legal_tcl,
                log_path=stage1_legal_root / "legalize.log",
                dry_run=False,
                openroad_num_threads=args.openroad_num_threads,
            )
            stages["placement_checkpoint_legalization"] = qualify_tools_checkpoint(
                stages["placement_checkpoint_legalization"],
                input_inventory=stage1_input_inventory,
                evaluated_inventory=stage1_evaluated_inventory,
                dpl_report=stage1_dpl_report,
            )
            if (
                stages["placement_checkpoint_legalization"].get("status") != "ok"
                or not stage1_legal_def.is_file()
                or int(
                    dict(
                        stages["placement_checkpoint_legalization"].get("metrics")
                        or {}
                    ).get("placement_valid", 0)
                    or 0
                )
                != 1
            ):
                failures.append("placement_checkpoint_legalization_failed")

        if not failures:
            stages["fixed_position_sizing"] = run_stage_command(
                stage2_command,
                cwd=AUTODMP_ROOT,
                log_path=stage2_root / "run.log",
                dry_run=False,
            )
            if (
                stages["fixed_position_sizing"].get("status") != "ok"
                or not stage2_raw_def.is_file()
            ):
                failures.append("fixed_position_sizing_failed")

        if not failures:
            stages["sizing_checkpoint_legalization"] = run_openroad_tcl(
                openroad_bin=args.openroad_bin,
                tcl_path=stage2_legal_tcl,
                log_path=stage2_legal_root / "legalize.log",
                dry_run=False,
                openroad_num_threads=args.openroad_num_threads,
            )
            stages["sizing_checkpoint_legalization"] = qualify_tools_checkpoint(
                stages["sizing_checkpoint_legalization"],
                input_inventory=stage2_input_inventory,
                evaluated_inventory=stage2_evaluated_inventory,
                dpl_report=stage2_dpl_report,
            )
            if (
                stages["sizing_checkpoint_legalization"].get("status") != "ok"
                or not stage2_legal_def.is_file()
                or int(
                    dict(
                        stages["sizing_checkpoint_legalization"].get("metrics")
                        or {}
                    ).get("placement_valid", 0)
                    or 0
                )
                != 1
            ):
                failures.append("sizing_checkpoint_legalization_failed")

        if not failures:
            stages["equal_spaced_route_b"] = run_stage_command(
                stage3_command,
                cwd=AUTODMP_ROOT,
                log_path=stage3_root / "launcher.log",
                dry_run=False,
            )
            stage3_payload = _read_json(stage3_result_path)
            stage3_result = dict(stage3_payload.get("result") or {})
            if (
                stages["equal_spaced_route_b"].get("status") != "ok"
                or stage3_payload.get("artifact")
                != "equal_spaced_buffering_method_result"
                or stage3_result.get("status") != "pass"
                or not final_raw_def.is_file()
            ):
                failures.append("equal_spaced_route_b_failed")
    else:
        stages = {
            name: {"status": "plan_only", "runtime_sec": 0.0}
            for name in command_plan["order"]
        }

    result = {
        "artifact": "tools_release_full_flow_method_result",
        "artifact_version": 1,
        "case": case,
        "method_id": method_id,
        "status": "plan_only" if plan_only else ("pass" if not failures else "failed"),
        "failures": failures,
        "profile_path": str(args.tools_release_profile),
        "profile_sha256": _sha256(args.tools_release_profile),
        "command_plan_path": str(method_root / "command_plan.json"),
        "stage_order": list(command_plan["order"]),
        "stages": stages,
        "r0_def": str(r0_def),
        "r0_def_sha256": _sha256(r0_def) if r0_def.is_file() else None,
        "raw_def": str(final_raw_def),
        "raw_def_sha256": _sha256(final_raw_def) if final_raw_def.is_file() else None,
        "source_artifacts": {
            "placement_case_summary": str(stage1_summary_path),
            "placement_raw_def": str(stage1_raw_def),
            "placement_legalized_def": str(stage1_legal_def),
            "sizing_raw_def": str(stage2_raw_def),
            "sizing_legalized_def": str(stage2_legal_def),
            "buffering_result": str(stage3_result_path),
        },
    }
    _write_json(method_root / "result.json", result)
    return result


def run_autodmp_staged(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    case_root: Path,
    command_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = validate_case_inputs(benchmark_root, case)
    flow_root = case_root / "autodmp_staged"
    stage1_output_def = flow_root / "stage1_diff_tdp.def"
    stage2_output_def = flow_root / "stage2_diff_sizing.def"
    stage1_result = flow_root / "stage1_result"
    stage2_result = flow_root / "stage2_result"
    stage1_command = build_autodmp_stage_command(
        python_bin=args.python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=stage1_result,
        flow_kind="placement",
        def_input=paths["def"],
        output_def=stage1_output_def,
        iterations=args.tdp_iterations,
        place_io_engine=args.autodmp_place_io_engine,
        diff_tdp_enabled=bool(args.stage1_diff_tdp),
        tdp_enable_threshold=args.tdp_enable_threshold,
        enable_net_weighting=args.stage1_enable_net_weighting,
        extra_args=args.stage1_placer_arg,
        dry_run_config=args.dry_run_config,
    )
    stage2_command = build_autodmp_stage_command(
        python_bin=args.python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=stage2_result,
        flow_kind="sizing",
        def_input=stage1_output_def,
        output_def=stage2_output_def,
        iterations=args.sizing_iterations,
        place_io_engine=args.autodmp_place_io_engine,
        extra_args=args.stage2_placer_arg,
        dry_run_config=args.dry_run_config,
    )
    stage1_log = flow_root / "stage1_diff_tdp.log"
    stage2_log = flow_root / "stage2_diff_sizing.log"
    _write_command(flow_root / "stage1_command.txt", stage1_command)
    _write_command(flow_root / "stage2_command.txt", stage2_command)
    stage1 = run_stage_command(stage1_command, cwd=REPO_ROOT, log_path=stage1_log, dry_run=args.dry_run_config)
    stage1["placement_debug_summary"] = read_stage1_placement_debug_summary(
        stage1_result,
        case,
    )
    stage2 = {
        "status": "skipped",
        "failure": "stage1 did not complete",
        "log_path": str(stage2_log),
    }
    if args.stage1_only:
        stage2["failure"] = "stage1-only mode"
    elif stage1["status"] in ("ok", "dry_run"):
        stage2 = run_stage_command(stage2_command, cwd=REPO_ROOT, log_path=stage2_log, dry_run=args.dry_run_config)
    repair_dir = flow_root / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_tcl = repair_dir / "repair.tcl"
    repair_log = repair_dir / "repair.log"
    repaired_def = repair_dir / f"{case}_autodmp_staged_repaired.def"
    repair_tcl.write_text(
        generate_openroad_repair_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=stage1_output_def if args.stage1_only else stage2_output_def,
            output_dir=repair_dir,
            repair_mode=args.repair_mode,
            output_def=repaired_def,
            run_timing_driven_placement=False,
        ),
        encoding="utf-8",
    )
    repair = {
        "status": "skipped",
        "failure": "stage1 did not complete" if args.stage1_only else "stage2 did not complete",
        "metrics": {},
        "tcl_path": str(repair_tcl),
        "log_path": str(repair_log),
    }
    if (
        (args.stage1_only and stage1["status"] in ("ok", "dry_run"))
        or (not args.stage1_only and stage2["status"] in ("ok", "dry_run"))
    ):
        repair = run_openroad_tcl(
            openroad_bin=args.openroad_bin,
            tcl_path=repair_tcl,
            log_path=repair_log,
            dry_run=args.dry_run_config,
            openroad_num_threads=args.openroad_num_threads,
        )
    repair_summary_path = repair_dir / "repair_summary.json"
    repair["summary_path"] = str(repair_summary_path)
    _write_json(
        repair_summary_path,
        {
            "case": case,
            "flow_name": "autodmp_staged",
            "stage1": stage1,
            "stage2": stage2,
            "repair": repair,
            "stage1_placement_debug_summary": stage1.get("placement_debug_summary", {}),
            "stage1_command": stage1_command,
            "stage2_command": stage2_command,
            "repair_tcl_path": str(repair_tcl),
        },
    )
    row = build_row(
        date=args.date,
        run_id=args.run_id,
        case=case,
        flow_name="autodmp_staged",
        repair_mode=args.repair_mode,
        stage1=stage1,
        stage2=None if args.stage1_only else stage2,
        repair=repair,
        runtime_total_sec=float(stage1.get("runtime_sec") or 0.0)
        + (0.0 if args.stage1_only else float(stage2.get("runtime_sec") or 0.0))
        + float(repair.get("runtime_sec") or 0.0),
        command_plan_path=command_plan_path,
        stage1_output_def=stage1_output_def,
        stage2_output_def=None if args.stage1_only else stage2_output_def,
    )
    detail = {
        "row": row,
        "stage1": stage1,
        "stage2": stage2,
        "repair": repair,
        "stage1_command": stage1_command,
        "stage2_command": stage2_command,
    }
    return row, detail


def run_openroad_baseline(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    case_root: Path,
    command_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = validate_case_inputs(benchmark_root, case)
    flow_root = case_root / "openroad_baseline"
    flow_root.mkdir(parents=True, exist_ok=True)
    tcl_path = flow_root / "openroad_tdp_repair.tcl"
    log_path = flow_root / "openroad_tdp_repair.log"
    output_def = flow_root / f"{case}_openroad_tdp_repaired.def"
    tcl_path.write_text(
        generate_openroad_repair_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=paths["def"],
            output_dir=flow_root,
            repair_mode=args.repair_mode,
            output_def=output_def,
            run_timing_driven_placement=True,
        ),
        encoding="utf-8",
    )
    repair = run_openroad_tcl(
        openroad_bin=args.openroad_bin,
        tcl_path=tcl_path,
        log_path=log_path,
        dry_run=args.dry_run_config,
        openroad_num_threads=args.openroad_num_threads,
    )
    repair_summary_path = flow_root / "repair_summary.json"
    repair["summary_path"] = str(repair_summary_path)
    _write_json(
        repair_summary_path,
        {
            "case": case,
            "flow_name": "openroad_tdp_baseline",
            "repair": repair,
            "tcl_path": str(tcl_path),
        },
    )
    row = build_row(
        date=args.date,
        run_id=args.run_id,
        case=case,
        flow_name="openroad_tdp_baseline",
        repair_mode=args.repair_mode,
        stage1=None,
        stage2=None,
        repair=repair,
        runtime_total_sec=float(repair.get("runtime_sec") or 0.0),
        command_plan_path=command_plan_path,
    )
    detail = {"row": row, "repair": repair, "tcl_path": str(tcl_path)}
    return row, detail


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--python-bin", type=Path, default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)))
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_BENCHMARK_ROOT / "openroad")
    parser.add_argument("--openroad-num-threads", type=int)
    parser.add_argument("--repair-mode", choices=repair_mode_choices(), default="repair_design")
    parser.add_argument("--tdp-iterations", type=int, default=400)
    parser.add_argument("--sizing-iterations", type=int, default=100)
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage1-diff-tdp", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tdp-enable-threshold", type=float, default=0.2)
    parser.add_argument("--stage1-enable-net-weighting", type=int, choices=(0, 1))
    parser.add_argument(
        "--autodmp-place-io-engine",
        choices=("openroad", "ieda"),
        default="openroad",
        help=(
            "Backend used by AutoDMP staged phases. OpenROAD is the staged "
            "default because it avoids iEDA timing-adapter sizing write-back "
            "crashes seen in multi-case sweeps."
        ),
    )
    parser.add_argument("--stage1-placer-arg", action="append", default=[])
    parser.add_argument("--stage2-placer-arg", action="append", default=[])
    parser.add_argument(
        "--tools-release-method",
        choices=TOOLS_RELEASE_METHODS,
        help="run one fixed-profile tools-paper source flow",
    )
    parser.add_argument(
        "--tools-release-profile",
        type=Path,
        default=TOOLS_RELEASE_PROFILE,
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--logical-gpu-id", type=int, default=0)
    parser.add_argument("--skip-autodmp", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Write command/Tcl plan and summaries without launching AutoDMP/OpenROAD.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    args.python_bin = args.python_bin.resolve()
    args.openroad_bin = args.openroad_bin.resolve()
    if args.openroad_num_threads is not None and args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    cases = resolve_cases(args.cases)
    run_root = args.output_root.resolve() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if args.tools_release_method is not None:
        if args.logical_gpu_id < 0:
            raise ValueError("--logical-gpu-id must be nonnegative")
        args.tools_release_profile = args.tools_release_profile.resolve()
        profile = _load_tools_release_profile(args.tools_release_profile)
        if not args.dry_run_config:
            for path in (
                args.python_bin,
                args.openroad_bin,
                args.tools_release_profile,
                EQUAL_BUFFERING_PROFILE,
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        rows = []
        for case in cases:
            row = run_tools_release_staged(
                args=args,
                benchmark_root=benchmark_root,
                case=case,
                run_root=run_root,
                profile=profile,
            )
            rows.append(row)
            print(
                f"{case}/{args.tools_release_method}: "
                f"{row['status']} {','.join(row['failures'])}"
            )
        _write_json(
            run_root / "tools_release_summary.json",
            {
                "artifact": "tools_release_full_flow_campaign",
                "artifact_version": 1,
                "run_id": args.run_id,
                "method_id": args.tools_release_method,
                "profile_path": str(args.tools_release_profile),
                "profile_sha256": _sha256(args.tools_release_profile),
                "rows": rows,
            },
        )
        return 0 if all(row["status"] in {"pass", "plan_only"} for row in rows) else 1
    command_plan_path = run_root / "command_plan.json"
    details: dict[str, Any] = {
        "artifact": "staged_joint_summary",
        "artifact_version": 1,
        "run_id": args.run_id,
        "date": args.date,
        "benchmark_root": str(benchmark_root),
        "run_root": str(run_root),
        "repair_mode": args.repair_mode,
        "autodmp_place_io_engine": args.autodmp_place_io_engine,
        "tdp_iterations": int(args.tdp_iterations),
        "sizing_iterations": int(args.sizing_iterations),
        "stage1_only": bool(args.stage1_only),
        "stage1_diff_tdp": int(args.stage1_diff_tdp),
        "tdp_enable_threshold": args.tdp_enable_threshold,
        "stage1_enable_net_weighting": args.stage1_enable_net_weighting,
        "dry_run_config": bool(args.dry_run_config),
        "cases": cases,
        "case_results": {},
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_root = run_root / case
        case_details: dict[str, Any] = {}
        if not args.skip_autodmp:
            row, detail = run_autodmp_staged(
                args=args,
                benchmark_root=benchmark_root,
                case=case,
                case_root=case_root,
                command_plan_path=command_plan_path,
            )
            rows.append(row)
            case_details["autodmp_staged"] = detail
            print(f"{case}/autodmp_staged: {row['status']} post_tns={row.get('post_repair_tns')}")
        if not args.skip_baseline:
            row, detail = run_openroad_baseline(
                args=args,
                benchmark_root=benchmark_root,
                case=case,
                case_root=case_root,
                command_plan_path=command_plan_path,
            )
            rows.append(row)
            case_details["openroad_tdp_baseline"] = detail
            print(f"{case}/openroad_tdp_baseline: {row['status']} post_tns={row.get('post_repair_tns')}")
        details["case_results"][case] = case_details
    csv_path = run_root / "staged_joint_summary.csv"
    json_path = run_root / "staged_joint_summary.json"
    details["summary_csv"] = str(csv_path)
    details["summary_json"] = str(json_path)
    write_csv(csv_path, rows)
    _write_json(json_path, {"case_results": details["case_results"], "rows": rows, **{k: v for k, v in details.items() if k != "case_results"}})
    _write_json(command_plan_path, details)
    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    print(f"command_plan: {command_plan_path}")
    ok_statuses = {"ok", "dry_run"}
    return 0 if rows and all(str(row.get("status")) in ok_statuses for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
