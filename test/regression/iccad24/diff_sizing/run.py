#!/usr/bin/env python3
"""Run the fixed-profile ICCAD24 AutoDMP/OpenROAD sizing campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
TOOLS_DIR = SCRIPT_DIR.parent / "tools_paper_four_table"
COMMON_TRACK_A = SCRIPT_DIR.parents[1] / "common" / "def_only_track_a.py"
PROFILE_PATH = TOOLS_DIR / "profiles" / "tools_release_v1.json"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_OUTPUT_ROOT = AUTODMP_ROOT / "logs" / "regression" / "iccad24" / "diff_sizing_tools"
DEFAULT_OPENROAD = Path("/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
METHODS = ("or_pure_sizeup", "autodmp_diff_sizing")
METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.*)$")

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import campaign_contract as contract


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tcl_quote(value: Path | str) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ)
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    error_lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith("Error:")
    ]
    return {
        "status": (
            "ok"
            if completed.returncode == 0 and not error_lines
            else "failed"
        ),
        "returncode": int(completed.returncode),
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
        "error_lines": error_lines,
    }


def _asap7_inputs(benchmark_root: Path) -> dict[str, Path | list[Path]]:
    root = benchmark_root / "ASAP7"
    return {
        "tech_lef": root / "lef" / "asap7_tech_1x_201209.lef",
        "lef": [
            root / "lef" / "asap7sc7p5t_27_R_1x_201211.lef",
            root / "lef" / "sram_asap7_16x256_1rw.lef",
            root / "lef" / "sram_asap7_32x256_1rw.lef",
            root / "lef" / "sram_asap7_64x256_1rw.lef",
            root / "lef" / "sram_asap7_64x64_1rw.lef",
        ],
        "lib": [
            root / "lib" / "asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
            root / "lib" / "asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
            root / "lib" / "asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
            root / "lib" / "asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
            root / "lib" / "asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
            root / "lib" / "sram_asap7_16x256_1rw.lib",
            root / "lib" / "sram_asap7_32x256_1rw.lib",
            root / "lib" / "sram_asap7_64x256_1rw.lib",
            root / "lib" / "sram_asap7_64x64_1rw.lib",
        ],
        "rc_tcl": root / "setRC.tcl",
    }


def _case_inputs(benchmark_root: Path, case: str) -> dict[str, Path]:
    root = benchmark_root / "design" / case
    return {
        "root": root,
        "workspace": root / "workspace",
        "params_json": root / "workspace" / "config" / "dreamplace_config" / "param.json",
        "def": root / f"{case}.def",
        "verilog": root / f"{case}.v",
        "sdc": root / f"{case}.sdc",
    }


def _require_inputs(benchmark_root: Path, case: str) -> None:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    paths = list(inputs.values()) + [tech["tech_lef"], tech["rc_tcl"]]
    paths.extend(tech["lef"])
    paths.extend(tech["lib"])
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{case}: missing inputs: " + ", ".join(missing))


def _append_design_inputs(
    command: list[str],
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
) -> None:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    command.extend(
        (
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
        )
    )
    for lef in tech["lef"]:
        command.extend(("--lef", str(lef)))
    for liberty in tech["lib"]:
        command.extend(("--lib", str(liberty)))


def build_autodmp_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    output_def: Path,
    iterations: int,
    logical_gpu_id: int,
    plan_only: bool,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(inputs["params_json"]),
        "--flow-kind",
        "sizing",
        "--place-io-engine",
        "openroad",
        "--workspace",
        str(inputs["workspace"]),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--output-def",
        str(output_def),
        "--iterations",
        str(int(iterations)),
        "--gpu",
        "1",
        "--gpu-id",
        str(int(logical_gpu_id)),
        "--legalize",
        "0",
        "--enable-fillers",
        "0",
        "--discrete-gradient-topk-up-percent",
        "30.0",
        "--discrete-gradient-topk-down-percent",
        "0.0",
    ]
    _append_design_inputs(
        command,
        benchmark_root=benchmark_root,
        case=case,
        def_input=inputs["def"],
    )
    if plan_only:
        command.append("--dry-run-config")
    return command


def _openroad_preamble(
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    lines = [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        "proc refresh_timing {} {",
        "  if {[llength [info commands update_timing]]} {",
        "    update_timing",
        "  } else {",
        "    catch {worst_slack -max}",
        "  }",
        "}",
        f"read_lef {_tcl_quote(tech['tech_lef'])}",
    ]
    lines.extend(f"read_lef {_tcl_quote(lef)}" for lef in tech["lef"])
    lines.extend(f"read_liberty {_tcl_quote(liberty)}" for liberty in tech["lib"])
    lines.extend(
        (
            f"read_def {_tcl_quote(def_input)}",
            f"read_sdc {_tcl_quote(inputs['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
        )
    )
    return lines


def build_openroad_sizeup_tcl(
    *,
    benchmark_root: Path,
    case: str,
    def_input: Path,
    output_def: Path,
) -> str:
    lines = _openroad_preamble(
        benchmark_root=benchmark_root,
        case=case,
        def_input=def_input,
    )
    lines.extend(
        (
            "estimate_parasitics -placement",
            "refresh_timing",
            "set method_start_us [clock microseconds]",
            "repair_timing -setup -sequence \"sizeup\" -skip_last_gasp "
            "-skip_pin_swap -skip_gate_cloning -skip_size_down -skip_vt_swap "
            "-skip_crit_vt_swap -skip_buffer_removal",
            'emit_metric "core_optimizer_runtime_sec" [expr {([clock microseconds] - $method_start_us) / 1000000.0}]',
            f"write_def {_tcl_quote(output_def)}",
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def build_track_a_command(
    *,
    python_bin: Path,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    method_id: str,
    reference_def: Path,
    committed_def: Path,
    output_dir: Path,
    buffer_master: str,
    plan_only: bool,
    openroad_num_threads: int | None = None,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(COMMON_TRACK_A),
        "--case",
        case,
        "--method-id",
        method_id,
        "--openroad-bin",
        str(openroad_bin),
        "--reference-def",
        str(reference_def),
        "--def-input",
        str(committed_def),
        "--tech-lef",
        str(tech["tech_lef"]),
    ]
    if openroad_num_threads is not None:
        command.extend(
            ("--openroad-num-threads", str(int(openroad_num_threads)))
        )
    for lef in tech["lef"]:
        command.extend(("--lef", str(lef)))
    for liberty in tech["lib"]:
        command.extend(("--lib", str(liberty)))
    command.extend(
        (
            "--sdc",
            str(inputs["sdc"]),
            "--rc-tcl",
            str(tech["rc_tcl"]),
            "--buffer-master",
            buffer_master,
            "--output-dir",
            str(output_dir),
        )
    )
    if plan_only:
        command.append("--plan-only")
    return command


def _load_track_a_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != "canonical_def_only_track_a":
        return {}
    if int(payload.get("artifact_version", -1)) != 1:
        return {}
    return payload


def _method_row(
    *,
    method_id: str,
    case: str,
    method_root: Path,
    source_command: list[str],
    track_a_command: list[str],
    source_execution: dict[str, Any] | None,
    track_a_execution: dict[str, Any] | None,
    track_a_summary: dict[str, Any],
    plan_only: bool,
) -> dict[str, Any]:
    failures = []
    if not plan_only:
        source_ok = bool(
            source_execution
            and source_execution.get("status") == "ok"
            and source_execution.get("committed_def_exists")
        )
        if not source_ok:
            failures.append("source_execution_failed")
        track_a_ok = bool(
            track_a_execution
            and track_a_execution.get("status") == "ok"
            and track_a_summary.get("status") == "pass"
        )
        if not track_a_execution or track_a_execution.get("status") != "ok":
            failures.append("track_a_execution_failed")
        if track_a_summary.get("status") != "pass":
            failures.append("track_a_failed")
        if track_a_ok:
            mutation = dict(track_a_summary.get("mutation") or {})
            if int(mutation.get("coordinate_changed_count", -1)) != 0:
                failures.append("method_changed_coordinates_before_track_a")
            if int(mutation.get("inserted_buffer_count", -1)) != 0:
                failures.append("unexpected_inserted_buffers")
            if int(mutation.get("inserted_other_count", -1)) != 0:
                failures.append("unexpected_inserted_instances")
            if int(mutation.get("deleted_instance_count", -1)) != 0:
                failures.append("unexpected_deleted_instances")
            if (
                method_id == "autodmp_diff_sizing"
                and case == "NV_NVDLA_partition_m"
                and int(mutation.get("resize_count", 0)) <= 0
            ):
                failures.append("known_positive_sizing_state_has_zero_resizes")
    return {
        "artifact": "diff_sizing_tools_method_result",
        "artifact_version": 1,
        "case": case,
        "method_id": method_id,
        "status": "plan_only" if plan_only else ("pass" if not failures else "failed"),
        "failures": failures,
        "method_root": str(method_root),
        "source_command": source_command,
        "track_a_command": track_a_command,
        "source_execution": source_execution,
        "track_a_execution": track_a_execution,
        "track_a_summary": track_a_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--openroad-num-threads", type=int)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--method", dest="methods", action="append", choices=METHODS)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--logical-gpu-id", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.openroad_num_threads is not None and args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    profile = contract.load_release_profile(args.profile.resolve())
    sizing = dict(profile["sizing"])
    buffer_master = str(profile["buffering"]["buffer_master"])
    cases = contract.selected_cases(args.cases)
    methods = tuple(args.methods or METHODS)
    benchmark_root = args.benchmark_root.resolve()
    run_root = args.output_root.resolve() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.plan_only:
        for case in cases:
            _require_inputs(benchmark_root, case)
        for path in (args.python_bin, args.openroad_bin, COMMON_TRACK_A):
            if not path.is_file():
                raise FileNotFoundError(path)
    rows = []
    for case in cases:
        inputs = _case_inputs(benchmark_root, case)
        for method_id in methods:
            method_root = run_root / case / method_id
            method_root.mkdir(parents=True, exist_ok=True)
            source_log = method_root / "source.log"
            committed_def = method_root / f"{case}_{method_id}_committed_raw.def"
            if method_id == "autodmp_diff_sizing":
                result_dir = method_root / "autodmp_result"
                source_command = build_autodmp_command(
                    python_bin=args.python_bin,
                    benchmark_root=benchmark_root,
                    case=case,
                    result_dir=result_dir,
                    output_def=committed_def,
                    iterations=int(sizing["iterations"]),
                    logical_gpu_id=int(args.logical_gpu_id),
                    plan_only=bool(args.plan_only),
                )
                source_tcl = None
                source_environment = {
                    "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)
                }
            else:
                source_tcl = method_root / "source.tcl"
                source_tcl.write_text(
                    build_openroad_sizeup_tcl(
                        benchmark_root=benchmark_root,
                        case=case,
                        def_input=inputs["def"],
                        output_def=committed_def,
                    ),
                    encoding="utf-8",
                )
                source_command = [str(args.openroad_bin)]
                if args.openroad_num_threads is not None:
                    source_command.extend(
                        ("-threads", str(int(args.openroad_num_threads)))
                    )
                source_command.append(str(source_tcl))
                source_environment = None
            track_a_root = method_root / "track_a"
            track_a_command = build_track_a_command(
                python_bin=args.python_bin,
                openroad_bin=args.openroad_bin,
                benchmark_root=benchmark_root,
                case=case,
                method_id=method_id,
                reference_def=inputs["def"],
                committed_def=committed_def,
                output_dir=track_a_root,
                buffer_master=buffer_master,
                plan_only=bool(args.plan_only),
                openroad_num_threads=args.openroad_num_threads,
            )
            (method_root / "source_command.txt").write_text(
                shlex.join(source_command) + "\n", encoding="utf-8"
            )
            (method_root / "track_a_command.txt").write_text(
                shlex.join(track_a_command) + "\n", encoding="utf-8"
            )
            source_execution = None
            track_a_execution = None
            track_a_summary = {}
            if not args.plan_only:
                source_execution = _run_command(
                    source_command,
                    cwd=AUTODMP_ROOT,
                    log_path=source_log,
                    environment=source_environment,
                )
                source_execution["committed_def_exists"] = committed_def.is_file()
                if source_execution["status"] == "ok" and committed_def.is_file():
                    track_a_execution = _run_command(
                        track_a_command,
                        cwd=AUTODMP_ROOT,
                        log_path=method_root / "track_a_launcher.log",
                    )
                    track_a_summary = _load_track_a_summary(
                        track_a_root / "track_a_summary.json"
                    )
            row = _method_row(
                method_id=method_id,
                case=case,
                method_root=method_root,
                source_command=source_command,
                track_a_command=track_a_command,
                source_execution=source_execution,
                track_a_execution=track_a_execution,
                track_a_summary=track_a_summary,
                plan_only=bool(args.plan_only),
            )
            _write_json(method_root / "result.json", row)
            rows.append(row)
            print(f"{case}/{method_id}: {row['status']} {','.join(row['failures'])}")
    summary = {
        "artifact": "diff_sizing_tools_campaign",
        "artifact_version": 1,
        "run_id": args.run_id,
        "profile_path": str(args.profile.resolve()),
        "profile_digest": contract.digest_payload(profile),
        "benchmark_root": str(benchmark_root),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "logical_gpu_id": int(args.logical_gpu_id),
        "rows": rows,
    }
    _write_json(run_root / "campaign_summary.json", summary)
    print(f"run_root: {run_root}")
    return 0 if all(row["status"] in {"pass", "plan_only"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
