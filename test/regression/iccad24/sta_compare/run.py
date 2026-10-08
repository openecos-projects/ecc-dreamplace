#!/usr/bin/env python3
"""Run ICCAD24 STA comparison between AutoDMP timing artifacts and OpenROAD.

This is a regression entrypoint, not an optimization flow. It keeps all run
artifacts under logs/regression/iccad24/sta_compare and writes a single CSV/JSON
pair for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
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

OPENROAD_METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.+)$")
VIOLATION_RE = re.compile(r"^.* +([-\.0-9]+) +\(VIOLATED\)")

SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "sta_compare"

CSV_COLUMNS = (
    "case",
    "status",
    "failure",
    "openroad_status",
    "openroad_wns_ns",
    "openroad_tns_ns",
    "openroad_slew_violation_ns",
    "openroad_cap_violation_pf",
    "openroad_fanout_violation",
    "openroad_instance_count",
    "openroad_area",
    "openroad_runtime_sec",
    "openroad_sta_query_repeat_count",
    "openroad_sta_query_first_sec",
    "openroad_sta_query_avg_sec",
    "openroad_sta_query_min_sec",
    "openroad_sta_query_max_sec",
    "openroad_sta_query_total_sec",
    "openroad_log",
    "autodmp_status",
    "autodmp_stage",
    "autodmp_python_wns_ns",
    "autodmp_python_tns_ns",
    "autodmp_backend_wns_ns",
    "autodmp_backend_tns_ns",
    "autodmp_post_wns_ns",
    "autodmp_post_tns_ns",
    "autodmp_slew_violation",
    "autodmp_cap_violation",
    "autodmp_log",
    "delta_post_wns_ns_vs_openroad",
    "delta_post_tns_ns_vs_openroad",
    "delta_python_wns_ns_vs_openroad",
    "delta_python_tns_ns_vs_openroad",
    "delta_backend_wns_ns_vs_openroad",
    "delta_backend_tns_ns_vs_openroad",
)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _ps_to_ns(value: Any) -> float | None:
    parsed = _as_float(value)
    return None if parsed is None else parsed / 1000.0


def _delta(lhs: Any, rhs: Any) -> float | None:
    lhs_float = _as_float(lhs)
    rhs_float = _as_float(rhs)
    if lhs_float is None or rhs_float is None:
        return None
    return lhs_float - rhs_float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tcl_quote(value: str | Path) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def resolve_cases(raw_cases: list[str] | None) -> list[str]:
    if not raw_cases or raw_cases == ["all"]:
        return list(ICCAD24_CASES)
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


def build_placer_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    flow_kind: str,
    iterations: int,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(paths["params_json"]),
        "--flow-kind",
        flow_kind,
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
    ]
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])
    if flow_kind != "sta":
        command.extend(["--iterations", str(iterations)])
    return command


def generate_openroad_sta_tcl(
    *,
    benchmark_root: Path,
    case: str,
    parasitics: str,
    output_dir: Path,
    sta_query_repeats: int = 0,
    def_input: Path | None = None,
) -> str:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    def_path = def_input or paths["def"]
    lines = [
        "proc emit_metric {name value} { puts \"METRIC|$name|$value\" }",
        "proc safe_metric {name script default_value} {",
        "  if {[catch {uplevel 1 $script} value]} { emit_metric $name $default_value } else { emit_metric $name $value }",
        "}",
        "proc measure_setup_sta_query {repeat_count} {",
        "  if {$repeat_count <= 0} { return }",
        "  set first_start [clock microseconds]",
        "  update",
        "  set first_wns [worst_slack -max]",
        "  set first_tns [total_negative_slack -max]",
        "  set first_us [expr {[clock microseconds] - $first_start}]",
        "  emit_metric \"sta_query_first_sec\" [expr {$first_us / 1000000.0}]",
        "  emit_metric \"sta_query_first_wns\" $first_wns",
        "  emit_metric \"sta_query_first_tns\" $first_tns",
        "  set total_us 0",
        "  set min_us -1",
        "  set max_us 0",
        "  set last_wns \"\"",
        "  set last_tns \"\"",
        "  for {set i 0} {$i < $repeat_count} {incr i} {",
        "    set iter_start [clock microseconds]",
        "    update",
        "    set last_wns [worst_slack -max]",
        "    set last_tns [total_negative_slack -max]",
        "    set elapsed_us [expr {[clock microseconds] - $iter_start}]",
        "    set total_us [expr {$total_us + $elapsed_us}]",
        "    if {$min_us < 0 || $elapsed_us < $min_us} { set min_us $elapsed_us }",
        "    if {$elapsed_us > $max_us} { set max_us $elapsed_us }",
        "  }",
        "  emit_metric \"sta_query_repeat_count\" $repeat_count",
        "  emit_metric \"sta_query_total_sec\" [expr {$total_us / 1000000.0}]",
        "  emit_metric \"sta_query_avg_sec\" [expr {$total_us / (1000000.0 * $repeat_count)}]",
        "  emit_metric \"sta_query_min_sec\" [expr {$min_us / 1000000.0}]",
        "  emit_metric \"sta_query_max_sec\" [expr {$max_us / 1000000.0}]",
        "  emit_metric \"sta_query_last_wns\" $last_wns",
        "  emit_metric \"sta_query_last_tns\" $last_tns",
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
            f"read_def -continue_on_errors {_tcl_quote(def_path)}",
            f"read_verilog {_tcl_quote(paths['verilog'])}",
            f"read_sdc {_tcl_quote(paths['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
            f"measure_setup_sta_query {max(0, int(sta_query_repeats))}",
        ]
    )
    if parasitics == "global_routing":
        lines.extend(
            [
                "set_routing_layers -signal M2-M9 -clock M2-M9",
                f"global_route -allow_congestion -congestion_iterations 50 -congestion_report_file {_tcl_quote(output_dir / f'{case}_congestion.rpt')}",
                "estimate_parasitics -global_routing",
            ]
        )
    lines.extend(
        [
            'safe_metric "setup_wns" {worst_slack -max} ""',
            'safe_metric "setup_tns" {total_negative_slack -max} ""',
            'emit_metric "slew_violation" [sum_check_type_violation "-max_slew"]',
            'emit_metric "cap_violation" [sum_check_type_violation "-max_capacitance"]',
            'emit_metric "fanout_violation" [sum_check_type_violation "-max_fanout"]',
            'safe_metric "instance_count" {llength [get_cells *]} ""',
            'safe_metric "area" {rsz::design_area} ""',
            'emit_metric "parasitics_source" "' + parasitics + '"',
            "report_power > " + _tcl_quote(output_dir / f"{case}_power.rpt"),
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


def run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def run_openroad_case(
    *,
    benchmark_root: Path,
    openroad_bin: Path,
    case: str,
    output_dir: Path,
    parasitics: str,
    sta_query_repeats: int,
    def_input: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tcl_path = output_dir / "openroad_sta.tcl"
    log_path = output_dir / "openroad_sta.log"
    tcl_path.write_text(
        generate_openroad_sta_tcl(
            benchmark_root=benchmark_root,
            case=case,
            parasitics=parasitics,
            output_dir=output_dir,
            sta_query_repeats=sta_query_repeats,
            def_input=def_input,
        ),
        encoding="utf-8",
    )
    if dry_run:
        return {
            "status": "dry_run",
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
        }
    if not openroad_bin.exists():
        return {
            "status": "failed",
            "failure": f"missing OpenROAD binary: {openroad_bin}",
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
        }
    returncode = run_command([str(openroad_bin), "-exit", str(tcl_path)], cwd=output_dir, log_path=log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    metrics = parse_openroad_metrics(log_text)
    status = "ok" if returncode == 0 else "failed"
    return {
        "status": status,
        "failure": "" if status == "ok" else f"OpenROAD exited with code {returncode}",
        "returncode": returncode,
        "metrics": metrics,
        "tcl_path": str(tcl_path),
        "log_path": str(log_path),
    }


def run_autodmp_case(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    output_dir: Path,
    iterations: int,
    flow_kind: str,
    def_input: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "autodmp_sta.log"
    command = build_placer_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        result_dir=output_dir,
        flow_kind=flow_kind,
        iterations=iterations,
    )
    if def_input is not None:
        try:
            def_index = command.index("--def-input")
        except ValueError:
            command.extend(["--def-input", str(def_input)])
        else:
            command[def_index + 1] = str(def_input)
    (output_dir / "autodmp_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "log_path": str(log_path),
        }
    returncode = run_command(command, cwd=REPO_ROOT, log_path=log_path)
    timing_summary_path = output_dir / f"{case}_timing_stage_summary.json"
    endpoint_summary_path = output_dir / f"{case}_endpoint_timing_compare_summary.json"
    timing_summary = _read_json(timing_summary_path)
    endpoint_summary = _read_json(endpoint_summary_path)
    if returncode != 0:
        status = "failed"
        failure = f"AutoDMP exited with code {returncode}"
    elif not timing_summary and not endpoint_summary:
        status = "missing_summary"
        failure = "missing timing summary artifacts"
    else:
        status = "ok"
        failure = ""
    return {
        "status": status,
        "failure": failure,
        "returncode": returncode,
        "timing_summary": timing_summary,
        "endpoint_summary": endpoint_summary,
        "timing_summary_path": str(timing_summary_path),
        "endpoint_summary_path": str(endpoint_summary_path),
        "log_path": str(log_path),
    }


def summarize_autodmp(result: dict[str, Any]) -> dict[str, Any]:
    timing_summary = result.get("timing_summary") or {}
    endpoint_summary = result.get("endpoint_summary") or {}
    post = timing_summary.get("post_legalization") or timing_summary.get("pre_projection") or {}
    stage_name = "post_legalization" if timing_summary.get("post_legalization") else "pre_projection"
    return {
        "autodmp_status": result.get("status"),
        "autodmp_stage": stage_name if timing_summary else "",
        "autodmp_python_wns_ns": _ps_to_ns(endpoint_summary.get("python_wns")),
        "autodmp_python_tns_ns": _ps_to_ns(endpoint_summary.get("python_tns")),
        "autodmp_backend_wns_ns": _ps_to_ns(endpoint_summary.get("backend_wns", endpoint_summary.get("ieda_wns"))),
        "autodmp_backend_tns_ns": _ps_to_ns(endpoint_summary.get("backend_tns", endpoint_summary.get("ieda_tns"))),
        "autodmp_post_wns_ns": _ps_to_ns(post.get("wns")),
        "autodmp_post_tns_ns": _ps_to_ns(post.get("tns")),
        "autodmp_slew_violation": post.get("slew_violation"),
        "autodmp_cap_violation": post.get("cap_violation"),
        "autodmp_log": result.get("log_path", ""),
    }


def build_summary_row(case: str, openroad: dict[str, Any], autodmp: dict[str, Any]) -> dict[str, Any]:
    openroad_metrics = openroad.get("metrics") or {}
    row = {
        "case": case,
        "openroad_status": openroad.get("status"),
        "openroad_wns_ns": openroad_metrics.get("setup_wns"),
        "openroad_tns_ns": openroad_metrics.get("setup_tns"),
        "openroad_slew_violation_ns": openroad_metrics.get("slew_violation"),
        "openroad_cap_violation_pf": openroad_metrics.get("cap_violation"),
        "openroad_fanout_violation": openroad_metrics.get("fanout_violation"),
        "openroad_instance_count": openroad_metrics.get("instance_count"),
        "openroad_area": openroad_metrics.get("area"),
        "openroad_runtime_sec": openroad_metrics.get("runtime_sec"),
        "openroad_sta_query_repeat_count": openroad_metrics.get("sta_query_repeat_count"),
        "openroad_sta_query_first_sec": openroad_metrics.get("sta_query_first_sec"),
        "openroad_sta_query_avg_sec": openroad_metrics.get("sta_query_avg_sec"),
        "openroad_sta_query_min_sec": openroad_metrics.get("sta_query_min_sec"),
        "openroad_sta_query_max_sec": openroad_metrics.get("sta_query_max_sec"),
        "openroad_sta_query_total_sec": openroad_metrics.get("sta_query_total_sec"),
        "openroad_log": openroad.get("log_path", ""),
    }
    row.update(summarize_autodmp(autodmp))
    row["delta_post_wns_ns_vs_openroad"] = _delta(row.get("autodmp_post_wns_ns"), row.get("openroad_wns_ns"))
    row["delta_post_tns_ns_vs_openroad"] = _delta(row.get("autodmp_post_tns_ns"), row.get("openroad_tns_ns"))
    row["delta_python_wns_ns_vs_openroad"] = _delta(row.get("autodmp_python_wns_ns"), row.get("openroad_wns_ns"))
    row["delta_python_tns_ns_vs_openroad"] = _delta(row.get("autodmp_python_tns_ns"), row.get("openroad_tns_ns"))
    row["delta_backend_wns_ns_vs_openroad"] = _delta(row.get("autodmp_backend_wns_ns"), row.get("openroad_wns_ns"))
    row["delta_backend_tns_ns_vs_openroad"] = _delta(row.get("autodmp_backend_tns_ns"), row.get("openroad_tns_ns"))
    failures = [item.get("failure", "") for item in (openroad, autodmp) if item.get("failure")]
    row["failure"] = "; ".join(failures)
    statuses = [str(openroad.get("status") or ""), str(autodmp.get("status") or "")]
    active_statuses = [status for status in statuses if status != "skipped"]
    if any(status not in ("ok", "dry_run", "skipped") for status in statuses):
        row["status"] = "failed"
    elif not active_statuses:
        row["status"] = "skipped"
    elif all(status == "ok" for status in active_statuses):
        row["status"] = "ok" if len(active_statuses) == len(statuses) else "partial"
    elif all(status == "dry_run" for status in active_statuses):
        row["status"] = "dry_run"
    else:
        row["status"] = "partial"
    return row


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
    parser.add_argument("--cases", nargs="*", default=["all"], help="ICCAD24 case names or all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)))
    parser.add_argument("--openroad-bin", type=Path, default=None)
    parser.add_argument("--autodmp-flow-kind", choices=("sta", "sizing"), default="sta")
    parser.add_argument("--autodmp-iterations", type=int, default=1)
    parser.add_argument("--openroad-parasitics", choices=("placement", "global_routing"), default="placement")
    parser.add_argument(
        "--def-input",
        type=Path,
        default=None,
        help="Override the benchmark DEF for both OpenROAD and AutoDMP STA checks.",
    )
    parser.add_argument(
        "--openroad-sta-query-repeats",
        type=int,
        default=0,
        help=(
            "After OpenROAD has read the design and estimated parasitics, measure repeated "
            "update + worst_slack -max + total_negative_slack -max queries."
        ),
    )
    parser.add_argument("--skip-openroad", action="store_true")
    parser.add_argument("--skip-autodmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    def_input = args.def_input.resolve() if args.def_input is not None else None
    if def_input is not None and not def_input.exists():
        raise SystemExit(f"missing --def-input: {def_input}")
    openroad_bin = args.openroad_bin or benchmark_root / "openroad"
    cases = resolve_cases(args.cases)
    run_root = args.output_root.resolve() / args.run_id
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "benchmark_root": str(benchmark_root),
        "run_root": str(run_root),
        "cases": cases,
        "openroad_parasitics": args.openroad_parasitics,
        "def_input": "" if def_input is None else str(def_input),
        "dry_run": bool(args.dry_run),
        "case_results": {},
    }
    for case in cases:
        case_root = run_root / case
        openroad = {"status": "skipped", "failure": "", "metrics": {}}
        autodmp = {"status": "skipped", "failure": ""}
        if not args.skip_openroad:
            openroad = run_openroad_case(
                benchmark_root=benchmark_root,
                openroad_bin=openroad_bin,
                case=case,
                output_dir=case_root / "openroad",
                parasitics=args.openroad_parasitics,
                sta_query_repeats=args.openroad_sta_query_repeats,
                def_input=def_input,
                dry_run=args.dry_run,
            )
        if not args.skip_autodmp:
            autodmp = run_autodmp_case(
                python_bin=args.python_bin,
                benchmark_root=benchmark_root,
                case=case,
                output_dir=case_root / "autodmp",
                iterations=args.autodmp_iterations,
                flow_kind=args.autodmp_flow_kind,
                def_input=def_input,
                dry_run=args.dry_run,
            )
        row = build_summary_row(case, openroad, autodmp)
        rows.append(row)
        details["case_results"][case] = {"openroad": openroad, "autodmp": autodmp, "row": row}
        print(f"{case}: {row['status']} openroad={row['openroad_status']} autodmp={row['autodmp_status']}")
    csv_path = run_root / "sta_compare_summary.csv"
    json_path = run_root / "sta_compare_summary.json"
    write_csv(csv_path, rows)
    _write_json(json_path, details)
    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    return 0 if all(row["status"] in ("ok", "dry_run", "partial") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
