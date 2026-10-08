#!/usr/bin/env python3
"""Run DEF-only OpenROAD global routing and OpenSTA evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
COMMON_DIR = AUTODMP_ROOT / "test" / "regression" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from def_only_track_a import (  # noqa: E402
    _common_tcl_procs,
    _read_design_lines,
    _tcl_quote,
    parse_metrics,
    parse_power_total,
    sha256_file,
    validate_complete_iccad24_liberty_manifest,
)


REQUIRED_METRICS = (
    "raw_hpwl_dbu",
    "wns_ns",
    "tns_ns",
    "endpoint_count",
    "violating_endpoint_count",
    "slew_violation_count",
    "slew_violation_total",
    "cap_violation_count",
    "cap_violation_total",
    "instance_count",
    "net_count",
    "iterm_count",
    "total_cell_area_um2",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_tcl(
    *,
    def_input: Path,
    tech_lef: Path,
    lefs: list[Path],
    libs: list[Path],
    sdc: Path,
    rc_tcl: Path,
    output_def: Path,
    congestion_report: Path,
    endpoint_csv: Path,
    power_report: Path,
) -> str:
    libs = tuple(libs)
    validate_complete_iccad24_liberty_manifest(libs)
    lines = _common_tcl_procs()
    lines.extend(
        _read_design_lines(
            def_input=def_input,
            tech_lef=tech_lef,
            lefs=lefs,
            libs=libs,
            sdc=sdc,
            rc_tcl=rc_tcl,
        )
    )
    lines.extend(
        (
            "set route_start_us [clock microseconds]",
            'emit_metric "raw_hpwl_dbu" [canonical_hpwl]',
            "set_routing_layers -signal M2-M9 -clock M2-M9",
            f"if {{[catch {{global_route -allow_congestion -congestion_iterations 50 -congestion_report_file {_tcl_quote(congestion_report)}}} message]}} {{",
            '  puts stderr "global_route failed: $message"',
            "  exit 2",
            "}",
            "estimate_parasitics -global_routing",
            "refresh_timing",
            'emit_metric "legalized_hpwl_dbu" [canonical_hpwl]',
            'safe_metric "wns_ns" {worst_slack -max} ""',
            'safe_metric "tns_ns" {total_negative_slack -max} ""',
            "lassign [check_type_summary -max_slew] slew_count slew_total",
            "lassign [check_type_summary -max_capacitance] cap_count cap_total",
            'emit_metric "slew_violation_count" $slew_count',
            'emit_metric "slew_violation_total" $slew_total',
            'emit_metric "cap_violation_count" $cap_count',
            'emit_metric "cap_violation_total" $cap_total',
            'emit_metric "instance_count" [llength [[ord::get_db_block] getInsts]]',
            'emit_metric "net_count" [llength [[ord::get_db_block] getNets]]',
            'emit_metric "iterm_count" [llength [[ord::get_db_block] getITerms]]',
            "lassign [area_summary] total_cell_area macro_area",
            'emit_metric "total_cell_area_um2" $total_cell_area',
            'emit_metric "macro_area_um2" $macro_area',
            f"dump_endpoint_slacks {_tcl_quote(endpoint_csv)}",
            "set_power_activity -input -activity 0.1 -duty 0.5",
            f"report_power > {_tcl_quote(power_report)}",
            f'emit_metric "global_route_runtime_sec" [expr {{([clock microseconds] - $route_start_us) / 1000000.0}}]',
            f"write_def {_tcl_quote(output_def)}",
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--openroad-bin", type=Path, required=True)
    parser.add_argument("--openroad-num-threads", type=int, default=8)
    parser.add_argument("--def-input", type=Path, required=True)
    parser.add_argument("--tech-lef", type=Path, required=True)
    parser.add_argument("--lef", dest="lefs", action="append", type=Path, required=True)
    parser.add_argument("--lib", dest="libs", action="append", type=Path, required=True)
    parser.add_argument("--sdc", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=21600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.openroad_num_threads <= 0 or args.timeout_sec <= 0:
        raise ValueError("thread and timeout settings must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "openroad_bin": args.openroad_bin.resolve(),
        "def_input": args.def_input.resolve(),
        "tech_lef": args.tech_lef.resolve(),
        "lefs": [path.resolve() for path in args.lefs],
        "libs": [path.resolve() for path in args.libs],
        "sdc": args.sdc.resolve(),
        "rc_tcl": args.rc_tcl.resolve(),
    }
    validate_complete_iccad24_liberty_manifest(inputs["libs"])
    required = [inputs["openroad_bin"], inputs["def_input"], inputs["tech_lef"], inputs["sdc"], inputs["rc_tcl"]]
    required.extend(inputs["lefs"])
    required.extend(inputs["libs"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input(s): " + ", ".join(missing))

    tcl_path = output_dir / "evaluate_global_route.tcl"
    log_path = output_dir / "evaluate_global_route.log"
    output_def = output_dir / "evaluated_global_route.def"
    congestion_report = output_dir / "congestion.rpt"
    endpoint_csv = output_dir / "endpoint_slacks.csv"
    power_report = output_dir / "power.rpt"
    summary_path = output_dir / "global_route_summary.json"
    tcl_path.write_text(
        build_tcl(
            def_input=inputs["def_input"],
            tech_lef=inputs["tech_lef"],
            lefs=inputs["lefs"],
            libs=inputs["libs"],
            sdc=inputs["sdc"],
            rc_tcl=inputs["rc_tcl"],
            output_def=output_def,
            congestion_report=congestion_report,
            endpoint_csv=endpoint_csv,
            power_report=power_report,
        ),
        encoding="utf-8",
    )
    command = [
        str(inputs["openroad_bin"]),
        "-exit",
        "-no_init",
        "-threads",
        str(args.openroad_num_threads),
        str(tcl_path),
    ]
    started = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=output_dir,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                timeout=args.timeout_sec,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
    metrics = parse_metrics(log_path)
    failures = []
    if returncode != 0 or timed_out:
        failures.append("openroad_execution_failed")
    missing_metrics = [name for name in REQUIRED_METRICS if metrics.get(name) is None]
    if missing_metrics:
        failures.append("missing_metrics:" + ",".join(missing_metrics))
    if not output_def.is_file():
        failures.append("missing_evaluated_def")
    power_total_w = parse_power_total(power_report)
    if power_total_w is None:
        failures.append("missing_power_total")
    summary = {
        "artifact": "def_only_global_route_opensta_evaluation",
        "artifact_version": 1,
        "status": "pass" if not failures else "failed",
        "case": args.case,
        "method_id": args.method_id,
        "failures": failures,
        "inputs": {
            "paths": {
                key: [str(item) for item in value] if isinstance(value, list) else str(value)
                for key, value in inputs.items()
            },
            "sha256": {
                key: [sha256_file(item) for item in value]
                if isinstance(value, list)
                else sha256_file(value)
                for key, value in inputs.items()
            },
        },
        "execution": {
            "command": command,
            "command_line": shlex.join(command),
            "returncode": returncode,
            "timed_out": timed_out,
            "runtime_sec": time.perf_counter() - started,
            "log_path": str(log_path),
        },
        "metrics": {**metrics, "power_total_w": power_total_w},
        "artifacts": {
            "tcl": str(tcl_path),
            "log": str(log_path),
            "congestion_report": str(congestion_report),
            "endpoint_csv": str(endpoint_csv),
            "power_report": str(power_report),
            "evaluated_def": str(output_def),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    print(f"summary: {summary_path}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
