#!/usr/bin/env python3
"""Run ICCAD24 diff timing-driven placement smokes.

The runner fixes the Phase-1 experiment matrix from
``docs/diff_timing_driven_placement/2026-06-25_diff_timing_driven_placement_hardening_plan.md``:

* placement-only
* diff TDP gated by overflow < 0.2
* diff TDP ungated

It writes command/log artifacts under ``logs/regression/...`` and keeps a
compact CSV/JSON summary for quick inspection.
"""

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
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "diff_timing_driven_placement"

ITERATION_RE = re.compile(r"iteration\s+(\d+).*")
FIELD_RE = re.compile(r"(Obj|WL|Overflow|WNS|TNS|TimingObj)\s+([-+0-9.Ee]+)")
DIFF_TDP_ENABLE_RE = re.compile(
    r"Diff TDP enabled timing objective at iteration=(\d+) "
    r"overflow=([-+0-9.Ee]+) threshold=([-+0-9.Ee]+) interval=(\d+)"
)
DIFF_TDP_GRADIENT_NET_WEIGHT_RE = re.compile(
    r"Diff TDP gradient net-weight update at iteration=(\d+) "
    r"overflow=([-+0-9.Ee]+)"
)

CSV_COLUMNS = (
    "case",
    "mode",
    "status",
    "failure",
    "iterations",
    "overflow_gate",
    "diff_tdp_enabled",
    "timing_enable_count",
    "first_timing_enable_iteration",
    "first_timing_enable_overflow",
    "final_iteration",
    "final_objective",
    "final_wl",
    "final_overflow",
    "final_wns",
    "final_tns",
    "final_timing_objective",
    "run_id",
    "run_root",
    "result_dir",
    "log_path",
    "command_path",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def resolve_modes(raw_modes: list[str] | None) -> list[str]:
    allowed = ("placement_only", "diff_tdp_gated", "diff_tdp_ungated")
    if not raw_modes:
        return list(allowed)
    modes: list[str] = []
    for raw in raw_modes:
        for part in str(raw).split(","):
            mode = part.strip()
            if mode:
                modes.append(mode)
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise SystemExit(
            "unsupported mode(s): "
            + ", ".join(invalid)
            + "; expected one of "
            + ", ".join(allowed)
        )
    return modes


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
    output_def: Path | None,
    mode: str,
    iterations: int,
    gated_overflow: float,
    refresh_interval: int,
    placer_args: list[str] | None = None,
    dry_run_config: bool = False,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(paths["params_json"]),
        "--flow-kind",
        "placement",
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
        str(int(iterations)),
    ]
    if output_def is not None:
        command.extend(["--output-def", str(output_def)])
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])
    if mode != "placement_only":
        threshold = gated_overflow if mode == "diff_tdp_gated" else 1.1
        command.extend(
            [
                "--with-sta",
                "--diff-timing-driven-placement",
                "1",
                "--timing-topology-refresh-interval",
                str(int(refresh_interval)),
                "--timing-topology-enable-overflow-threshold",
                str(float(threshold)),
            ]
        )
    if placer_args:
        command.extend(str(arg) for arg in placer_args)
    if dry_run_config:
        command.append("--dry-run-config")
    return command


def parse_placement_log(log_text: str) -> dict[str, Any]:
    final_metrics: dict[str, Any] = {}
    for line in log_text.splitlines():
        if "iteration" not in line:
            continue
        iteration_match = ITERATION_RE.search(line)
        if not iteration_match:
            continue
        parsed = {"iteration": int(iteration_match.group(1))}
        for key, raw_value in FIELD_RE.findall(line):
            parsed[key.lower()] = _as_float(raw_value)
        final_metrics = parsed

    timing_events = []
    for match in DIFF_TDP_ENABLE_RE.finditer(log_text):
        timing_events.append(
            {
                "iteration": int(match.group(1)),
                "overflow": float(match.group(2)),
                "threshold": float(match.group(3)),
                "interval": int(match.group(4)),
                "carrier": "direct_loss",
            }
        )
    for match in DIFF_TDP_GRADIENT_NET_WEIGHT_RE.finditer(log_text):
        timing_events.append(
            {
                "iteration": int(match.group(1)),
                "overflow": float(match.group(2)),
                "threshold": None,
                "interval": None,
                "carrier": "gradient_net_weight",
            }
        )
    timing_events.sort(key=lambda event: event["iteration"])
    return {"final_metrics": final_metrics, "timing_events": timing_events}


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


def summarize_result(
    *,
    case: str,
    mode: str,
    run_id: str,
    run_root: Path,
    result_dir: Path,
    log_path: Path,
    command_path: Path,
    iterations: int,
    overflow_gate: float | None,
    returncode: int,
    plan_only: bool,
) -> dict[str, Any]:
    row = {
        "case": case,
        "mode": mode,
        "iterations": int(iterations),
        "overflow_gate": overflow_gate,
        "diff_tdp_enabled": int(mode != "placement_only"),
        "run_id": run_id,
        "run_root": str(run_root),
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "command_path": str(command_path),
    }
    if plan_only:
        row.update({"status": "plan_only", "failure": ""})
        return row
    if returncode != 0:
        row.update(
            {
                "status": "failed",
                "failure": f"Placer.py exited with code {returncode}",
            }
        )
        return row
    parsed = parse_placement_log(log_path.read_text(encoding="utf-8", errors="ignore"))
    final_metrics = parsed["final_metrics"]
    timing_events = parsed["timing_events"]
    first_event = timing_events[0] if timing_events else {}
    row.update(
        {
            "status": "pass",
            "failure": "",
            "timing_enable_count": len(timing_events),
            "first_timing_enable_iteration": first_event.get("iteration"),
            "first_timing_enable_overflow": first_event.get("overflow"),
            "final_iteration": final_metrics.get("iteration"),
            "final_objective": final_metrics.get("obj"),
            "final_wl": final_metrics.get("wl"),
            "final_overflow": final_metrics.get("overflow"),
            "final_wns": final_metrics.get("wns"),
            "final_tns": final_metrics.get("tns"),
            "final_timing_objective": final_metrics.get("timingobj"),
        }
    )
    if mode == "diff_tdp_gated" and timing_events:
        bad_events = [
            event
            for event in timing_events
            if event["overflow"] >= float(overflow_gate)
        ]
        if bad_events:
            row["status"] = "failed"
            row["failure"] = "diff TDP enabled before overflow gate"
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
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--mode", dest="modes", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)),
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--overflow-gate", type=float, default=0.2)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument(
        "--placer-arg",
        dest="placer_args",
        action="append",
        default=[],
        help="Append one raw argument to Placer.py; repeat for flags and values.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    cases = resolve_cases(args.cases)
    modes = resolve_modes(args.modes)
    run_root = args.output_root.resolve() / args.run_id
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "benchmark_root": str(benchmark_root),
        "run_root": str(run_root),
        "run_id": args.run_id,
        "cases": cases,
        "modes": modes,
        "case_results": {},
    }
    for case in cases:
        details["case_results"].setdefault(case, {})
        for mode in modes:
            case_root = run_root / case / mode
            result_dir = case_root / "result"
            output_def = case_root / "final.def"
            log_path = case_root / "run.log"
            command_path = case_root / "command.txt"
            overflow_gate = None if mode == "placement_only" else float(args.overflow_gate)
            command = build_placer_command(
                python_bin=args.python_bin,
                benchmark_root=benchmark_root,
                case=case,
                result_dir=result_dir,
                output_def=output_def,
                mode=mode,
                iterations=args.iterations,
                gated_overflow=args.overflow_gate,
                refresh_interval=args.refresh_interval,
                placer_args=args.placer_args,
                dry_run_config=args.dry_run_config,
            )
            case_root.mkdir(parents=True, exist_ok=True)
            command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
            returncode = 0
            if not args.plan_only:
                returncode = run_command(command, cwd=REPO_ROOT, log_path=log_path)
            row = summarize_result(
                case=case,
                mode=mode,
                run_id=args.run_id,
                run_root=run_root,
                result_dir=result_dir,
                log_path=log_path,
                command_path=command_path,
                iterations=args.iterations,
                overflow_gate=overflow_gate,
                returncode=returncode,
                plan_only=args.plan_only,
            )
            rows.append(row)
            details["case_results"][case][mode] = {"row": row, "command": command}
            print(
                f"{case}/{mode}: {row['status']} "
                f"timing_enable_count={row.get('timing_enable_count')} "
                f"final_tns={row.get('final_tns')}"
            )

    csv_path = run_root / "diff_timing_driven_placement_summary.csv"
    json_path = run_root / "diff_timing_driven_placement_summary.json"
    write_csv(csv_path, rows)
    _write_json(json_path, details)
    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    return 0 if all(row["status"] in ("pass", "plan_only") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
