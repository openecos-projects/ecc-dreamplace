#!/usr/bin/env python3
"""Run the physical staged TDP/sizing/buffering alternating experiment.

Each outer iteration is an explicit physical-state transaction:

    TDP -> OpenROAD legalization -> fixed-position sizing
        -> OpenROAD legalization -> equal-spaced buffering

The buffered state is evaluated before it is accepted as the next iteration's
input.  All timing decisions use the common DEF-only, nine-Liberty OpenROAD
evaluator.  This runner is intentionally an experiment entry point; the
individual optimization stages remain owned by Placer.py.
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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = AUTODMP_ROOT.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import campaign_protocol as protocol  # noqa: E402


STAGED_RUNNER_PATH = AUTODMP_ROOT / "test/regression/iccad24/staged_joint/run.py"
_staged_spec = importlib.util.spec_from_file_location(
    "autodmp_staged_joint_helpers", STAGED_RUNNER_PATH
)
if _staged_spec is None or _staged_spec.loader is None:
    raise RuntimeError(f"cannot load staged runner helpers: {STAGED_RUNNER_PATH}")
_staged_helpers = importlib.util.module_from_spec(_staged_spec)
_staged_spec.loader.exec_module(_staged_helpers)


CANONICAL_CASES = protocol.CANONICAL_CASES
DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get(
        "AUTODMP_ICCAD24_BENCHMARK_ROOT",
        str(REPO_ROOT / "workspace_case/iccad24-benchmark"),
    )
)
DEFAULT_OUTPUT_ROOT = (
    AUTODMP_ROOT / "logs/regression/iccad24/timing_physical_synthesis"
)
DEFAULT_PYTHON = Path(
    os.environ.get(
        "AUTODMP_PYTHON_BIN", "/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python"
    )
)
DEFAULT_OPENROAD = Path(
    os.environ.get("AUTODMP_OPENROAD_BIN", "/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
)
DEFAULT_BUFFER_PROFILE = (
    AUTODMP_ROOT
    / "test/regression/iccad24/equal_spaced_buffering/params/route_b_allcases_0p1pct.json"
)

CSV_FIELDS = (
    "case",
    "status",
    "outer_iterations",
    "selected_outer_iteration",
    "selected_def",
    "selected_wns_ns",
    "selected_tns_ns",
    "sequential_outer0_tns_ns",
    "openroad_baseline_tns_ns",
    "runtime_sec",
    "summary_path",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
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


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_case_list(values: list[str] | None) -> list[str]:
    if not values:
        return [CANONICAL_CASES[0]]
    selected: list[str] = []
    for raw in values:
        selected.extend(part.strip() for part in str(raw).split(",") if part.strip())
    invalid = sorted(set(selected) - set(CANONICAL_CASES))
    if invalid:
        raise ValueError("unsupported case(s): " + ", ".join(invalid))
    return selected


def _environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
        }
    )
    return env


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
    timeout_sec: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
    return {
        "status": "pass" if returncode == 0 and not timed_out else "failed",
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
        "command": command,
    }


def _tech_args(benchmark_root: Path) -> list[str]:
    technology = protocol.technology_paths(benchmark_root)
    args = ["--tech-lef", str(technology["tech_lef"])]
    for path in technology["lefs"]:
        args.extend(("--lef", str(path)))
    for path in technology["libs"]:
        args.extend(("--lib", str(path)))
    args.extend(("--rc-tcl", str(technology["rc_tcl"])))
    return args


def _case_base_args(
    *,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    result_dir: Path,
) -> list[str]:
    paths = protocol.case_paths(benchmark_root, case)
    params_json = (
        benchmark_root
        / "design"
        / case
        / "workspace/config/dreamplace_config/param.json"
    )
    workspace = benchmark_root / "design" / case / "workspace"
    return [
        str(params_json),
        "--place-io-engine",
        "openroad",
        "--workspace",
        str(workspace),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--def-input",
        str(input_def),
        "--verilog-input",
        str(paths["verilog"]),
        "--sdc",
        str(paths["sdc"]),
    ] + _tech_args(benchmark_root)


def _profile_values(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    placement = dict(profile.get("placement") or {})
    pin2pin = dict(placement.get("pin2pin") or {})
    buffering = dict(profile.get("buffering") or {})
    return pin2pin, buffering


def _build_tdp_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    input_def: Path,
    result_dir: Path,
    output_def: Path,
    restart: bool,
) -> list[str]:
    pin2pin, _ = _profile_values(profile)
    continuation = bool(restart)
    tdp_iterations = (
        int(args.continuation_tdp_iterations)
        if continuation
        else int(args.tdp_iterations)
    )
    handoff_tolerance = (
        float(args.continuation_placement_overflow_tolerance)
        if continuation
        else float(args.placement_overflow_tolerance)
    )
    command = [
        str(args.python_bin),
        str(AUTODMP_ROOT / "dreamplace/Placer.py"),
        *_case_base_args(
            benchmark_root=args.benchmark_root,
            case=case,
            input_def=input_def,
            result_dir=result_dir,
        ),
        "--flow-kind",
        "placement",
        "--output-def",
        str(output_def),
        "--iterations",
        str(tdp_iterations),
        "--with-sta",
        "--gpu",
        "1",
        "--gpu-id",
        str(int(args.logical_gpu_id)),
        "--placement-optimizer",
        "nesterov",
        "--random-center-init",
        "0",
        "--enable-fillers",
        "0" if continuation else "1",
        "--legalize",
        "0",
        "--stop-overflow",
        "0.1",
        "--placement-overflow-tolerance",
        str(handoff_tolerance),
        "--diff-timing-driven-placement",
        "0",
        "--enable-net-weighting",
        "1",
        "--net-weighting-scheme",
        "pin2pin",
        "--pin2pin-weight",
        str(float(pin2pin.get("weight", 0.0005))),
        "--pin2pin-min-weight",
        str(float(pin2pin.get("min_weight", 10.0))),
        "--pin2pin-max-weight",
        str(float(pin2pin.get("max_weight", 50.0))),
        "--pin2pin-accumulate-weight",
        str(float(pin2pin.get("accumulate_weight", 0.2))),
        "--net-weighting-npaths",
        str(float(pin2pin.get("path_limit", 0.0))),
        "--net-weighting-update-interval",
        str(int(pin2pin.get("update_interval", 15))),
        "--timing-topology-enable-overflow-threshold",
        str(float(pin2pin.get("activation_overflow", 0.3))),
        "--plot",
        "0",
    ]
    if continuation:
        # A continuation starts close to the density handoff. Reuse the
        # stable carrier used by the validated continuation probe and avoid
        # adding a second filler population to an already legalized state.
        command.extend(
            [
                "--timing-placement-carrier",
                "gradient_net_weight",
                "--timing-gradient-net-weight-scale",
                "0.4",
                "--timing-gradient-net-weight-max",
                "2.0",
            ]
        )
    if continuation and args.nesterov_step_cap is not None:
        command.extend(
            [
                "--placement-initial-learning-rate-max",
                str(float(args.nesterov_step_cap)),
            ]
        )
    return command


def _build_sizing_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    input_def: Path,
    result_dir: Path,
    output_def: Path,
) -> list[str]:
    sizing = dict(profile.get("sizing") or {})
    return [
        str(args.python_bin),
        str(AUTODMP_ROOT / "dreamplace/Placer.py"),
        *_case_base_args(
            benchmark_root=args.benchmark_root,
            case=case,
            input_def=input_def,
            result_dir=result_dir,
        ),
        "--flow-kind",
        "sizing",
        "--output-def",
        str(output_def),
        "--iterations",
        str(int(args.sizing_iterations)),
        "--with-sta",
        "--gpu",
        "1",
        "--gpu-id",
        str(int(args.logical_gpu_id)),
        "--legalize",
        "0",
        "--enable-fillers",
        "0",
        "--discrete-gradient-topk-up-percent",
        str(float(sizing.get("up_percent", 30.0))),
        "--discrete-gradient-topk-down-percent",
        str(float(sizing.get("down_percent", 0.0))),
        "--plot",
        "0",
    ]


def _build_buffering_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    input_def: Path,
    result_dir: Path,
    output_def: Path,
) -> list[str]:
    _, buffering = _profile_values(profile)
    return [
        str(args.python_bin),
        str(AUTODMP_ROOT / "dreamplace/Placer.py"),
        *_case_base_args(
            benchmark_root=args.benchmark_root,
            case=case,
            input_def=input_def,
            result_dir=result_dir,
        ),
        "--flow-kind",
        "buffering",
        "--output-def",
        str(output_def),
        "--iterations",
        "1",
        "--buffering-mode",
        "segment",
        "--buffering-segment-strategy",
        "discrete_net_gradient",
        "--buffering-continuous-steps",
        str(int(buffering.get("rounds", 5))),
        "--buffering-max-repeaters-per-segment",
        str(int(buffering.get("max_repeaters_per_segment", 3))),
        "--buffering-fixed-bsu-index",
        str(int(buffering.get("fixed_bsu_index", 7))),
        "--buffering-segment-count-z-init",
        "0",
        "--buffering-segment-count-timing-backend",
        str(buffering.get("timing_backend", "cpp_cuda_segment_transfer_explicit_autograd")),
        "--buffering-segment-transfer-backend",
        str(buffering.get("transfer_backend", "segment_transfer_native")),
        "--buffering-segment-integer-projection-interval",
        "0",
        "--buffering-segment-integer-projection-start-step",
        "0",
        "--buffering-commit-enabled",
        "1",
        "--buffering-committed-def-path",
        str(output_def),
        "--with-sta",
        "--gpu",
        "1",
        "--gpu-id",
        str(int(args.logical_gpu_id)),
        "--enable-fillers",
        "0",
        "--plot",
        "0",
    ]


def _write_legalization_tcl(
    *,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_def: Path,
    root: Path,
) -> Path:
    input_inventory = root / "input_inventory.tsv"
    evaluated_inventory = root / "evaluated_inventory.tsv"
    dpl_report = root / "dpl_placement_report.json"
    tcl_path = root / "legalize.tcl"
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text(
        _staged_helpers.generate_tools_checkpoint_tcl(
            benchmark_root=benchmark_root,
            case=case,
            def_input=input_def,
            output_def=output_def,
            input_inventory=input_inventory,
            evaluated_inventory=evaluated_inventory,
            dpl_report=dpl_report,
            detailed_placement_search_window="full_core",
        ),
        encoding="utf-8",
    )
    return tcl_path


def _parse_metric_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.is_file():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("METRIC|"):
            continue
        try:
            _, key, raw = line.split("|", 2)
        except ValueError:
            continue
        value = _finite_float(raw)
        metrics[key] = raw if value is None else value
    return metrics


def _write_fixed_eval_tcl(
    *,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    tcl_path: Path,
) -> None:
    paths = protocol.case_paths(benchmark_root, case)
    tech = protocol.technology_paths(benchmark_root)
    lines = [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        'proc safe_metric {name script default_value} {',
        '  if {[catch {uplevel 1 $script} value]} { emit_metric $name $default_value } else { emit_metric $name $value }',
        '}',
        f'read_lef {{{tech["tech_lef"]}}}',
    ]
    for lef in tech["lefs"]:
        lines.append(f"read_lef {{{lef}}}")
    for liberty in tech["libs"]:
        lines.append(f"read_liberty {{{liberty}}}")
    lines.extend(
        [
            f"read_def -continue_on_errors {{{input_def}}}",
            f"read_sdc {{{paths['sdc']}}}",
            "set_ideal_network [all_clocks]",
            f"source {{{tech['rc_tcl']}}}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
            "if {[llength [info commands update_timing]]} { update_timing }",
            'safe_metric "wns_ns" {worst_slack -max} ""',
            'safe_metric "tns_ns" {total_negative_slack -max} ""',
            'safe_metric "instance_count" {llength [get_cells *]} ""',
            "exit",
        ]
    )
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_def(
    *,
    args: argparse.Namespace,
    case: str,
    input_def: Path,
    root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    tcl_path = root / "evaluate.tcl"
    log_path = root / "evaluate.log"
    _write_fixed_eval_tcl(
        benchmark_root=args.benchmark_root,
        case=case,
        input_def=input_def,
        tcl_path=tcl_path,
    )
    execution = _run(
        [str(args.openroad_bin), "-threads", str(args.openroad_num_threads), "-exit", str(tcl_path)],
        cwd=root,
        log_path=log_path,
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    metrics = _parse_metric_log(log_path)
    return {
        "status": "pass" if execution["status"] == "pass" else "failed",
        "execution": execution,
        "metrics": metrics,
        "wns_ns": _finite_float(metrics.get("wns_ns")),
        "tns_ns": _finite_float(metrics.get("tns_ns")),
        "tcl_path": str(tcl_path),
        "log_path": str(log_path),
    }


def _legalize(
    *,
    args: argparse.Namespace,
    case: str,
    input_def: Path,
    output_def: Path,
    root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    tcl_path = _write_legalization_tcl(
        benchmark_root=args.benchmark_root,
        case=case,
        input_def=input_def,
        output_def=output_def,
        root=root,
    )
    log_path = root / "legalize.log"
    execution = _run(
        [str(args.openroad_bin), "-threads", str(args.openroad_num_threads), "-exit", str(tcl_path)],
        cwd=root,
        log_path=log_path,
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    metrics = _parse_metric_log(log_path)
    valid = output_def.is_file() and int(metrics.get("placement_valid", 0) or 0) == 1
    return {
        "status": "pass" if execution["status"] == "pass" and valid else "failed",
        "execution": execution,
        "metrics": metrics,
        "placement_valid": valid,
        "output_def": str(output_def),
        "tcl_path": str(tcl_path),
        "log_path": str(log_path),
    }


def _prepare_r0(
    *,
    args: argparse.Namespace,
    case: str,
    case_root: Path,
    environment: dict[str, str],
) -> Path:
    r0_root = case_root / "r0"
    r0_def = r0_root / "R0.def"
    if args.input_r0_def:
        source = Path(str(args.input_r0_def).format(case=case)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not r0_def.is_file():
            r0_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, r0_def)
        if _sha256(r0_def) != _sha256(source):
            raise ValueError(f"{case}: existing R0 differs from --input-r0-def")
        _write_json(
            r0_root / "manifest.json",
            {
                "status": "pass",
                "producer": "external_prepared_r0",
                "source": str(source),
                "source_sha256": _sha256(source),
            },
        )
        return r0_def
    manifest = r0_root / "R0_manifest.json"
    if r0_def.is_file() and _read_json(manifest).get("status") == "pass":
        return r0_def
    command = protocol.build_r0_command(
        python_bin=args.python_bin,
        benchmark_root=args.benchmark_root,
        case=case,
        # build_r0_command appends the case name to the campaign root.
        output_root=case_root.parent,
        seed=args.seed,
    )
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=r0_root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if execution["status"] != "pass" or not r0_def.is_file():
        raise RuntimeError(f"{case}: R0 preparation failed: {execution}")
    return r0_def


def _accept_buffer(
    sizing_eval: dict[str, Any],
    buffer_eval: dict[str, Any],
) -> tuple[bool, str]:
    sizing_tns = sizing_eval.get("tns_ns")
    buffer_tns = buffer_eval.get("tns_ns")
    if sizing_tns is None or buffer_tns is None:
        return False, "missing_fixed_state_tns"
    if buffer_tns > sizing_tns + 1.0e-6:
        return True, "buffered_tns_improves_sizing_state"
    if abs(buffer_tns - sizing_tns) <= 1.0e-6:
        sizing_wns = sizing_eval.get("wns_ns")
        buffer_wns = buffer_eval.get("wns_ns")
        if sizing_wns is not None and buffer_wns is not None and buffer_wns >= sizing_wns:
            return True, "tns_tie_and_wns_not_worse"
    return False, "buffered_state_not_better_than_sizing_state"


def _run_outer_iteration(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    outer_index: int,
    input_def: Path,
    case_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    root = case_root / f"outer_{outer_index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}

    tdp_root = root / "tdp"
    tdp_raw = tdp_root / f"{case}_tdp_raw.def"
    tdp_command = _build_tdp_command(
        args=args,
        profile=profile,
        case=case,
        input_def=input_def,
        result_dir=tdp_root / "autodmp_result",
        output_def=tdp_raw,
        restart=outer_index > 0,
    )
    (tdp_root / "command.txt").parent.mkdir(parents=True, exist_ok=True)
    (tdp_root / "command.txt").write_text(shlex.join(tdp_command) + "\n", encoding="utf-8")
    stages["tdp"] = _run(
        tdp_command,
        cwd=AUTODMP_ROOT,
        log_path=tdp_root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    stages["tdp"]["output_def"] = str(tdp_raw)
    if stages["tdp"]["status"] != "pass" or not tdp_raw.is_file():
        return {"status": "failed", "input_def": str(input_def), "stages": stages, "failed_stage": "tdp"}

    tdp_legal_root = root / "tdp_legalize"
    tdp_legal = tdp_legal_root / f"{case}_tdp_legalized.def"
    stages["tdp_legalize"] = _legalize(
        args=args,
        case=case,
        input_def=tdp_raw,
        output_def=tdp_legal,
        root=tdp_legal_root,
        environment=environment,
    )
    if stages["tdp_legalize"]["status"] != "pass":
        return {"status": "failed", "input_def": str(input_def), "stages": stages, "failed_stage": "tdp_legalize"}
    stages["tdp_legalize"]["evaluation"] = _evaluate_def(
        args=args, case=case, input_def=tdp_legal, root=tdp_legal_root / "fixed_eval", environment=environment
    )

    sizing_root = root / "sizing"
    sizing_raw = sizing_root / f"{case}_sizing_raw.def"
    sizing_command = _build_sizing_command(
        args=args,
        profile=profile,
        case=case,
        input_def=tdp_legal,
        result_dir=sizing_root / "autodmp_result",
        output_def=sizing_raw,
    )
    sizing_root.mkdir(parents=True, exist_ok=True)
    (sizing_root / "command.txt").write_text(shlex.join(sizing_command) + "\n", encoding="utf-8")
    stages["sizing"] = _run(
        sizing_command,
        cwd=AUTODMP_ROOT,
        log_path=sizing_root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    stages["sizing"]["output_def"] = str(sizing_raw)
    if stages["sizing"]["status"] != "pass" or not sizing_raw.is_file():
        return {"status": "failed", "input_def": str(input_def), "stages": stages, "failed_stage": "sizing"}

    sizing_legal_root = root / "sizing_legalize"
    sizing_legal = sizing_legal_root / f"{case}_sizing_legalized.def"
    stages["sizing_legalize"] = _legalize(
        args=args,
        case=case,
        input_def=sizing_raw,
        output_def=sizing_legal,
        root=sizing_legal_root,
        environment=environment,
    )
    if stages["sizing_legalize"]["status"] != "pass":
        return {"status": "failed", "input_def": str(input_def), "stages": stages, "failed_stage": "sizing_legalize"}
    sizing_eval = _evaluate_def(
        args=args, case=case, input_def=sizing_legal, root=sizing_legal_root / "fixed_eval", environment=environment
    )
    stages["sizing_legalize"]["evaluation"] = sizing_eval

    buffering_root = root / "buffering"
    buffering_candidate = buffering_root / f"{case}_buffering_candidate.def"
    buffering_command = _build_buffering_command(
        args=args,
        profile=profile,
        case=case,
        input_def=sizing_legal,
        result_dir=buffering_root / "autodmp_result",
        output_def=buffering_candidate,
    )
    buffering_root.mkdir(parents=True, exist_ok=True)
    (buffering_root / "command.txt").write_text(shlex.join(buffering_command) + "\n", encoding="utf-8")
    stages["buffering"] = _run(
        buffering_command,
        cwd=AUTODMP_ROOT,
        log_path=buffering_root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    stages["buffering"]["output_def"] = str(buffering_candidate)
    if stages["buffering"]["status"] != "pass" or not buffering_candidate.is_file():
        return {"status": "failed", "input_def": str(input_def), "stages": stages, "failed_stage": "buffering"}
    buffer_eval = _evaluate_def(
        args=args, case=case, input_def=buffering_candidate, root=buffering_root / "fixed_eval", environment=environment
    )
    stages["buffering"]["evaluation"] = buffer_eval
    accepted, reason = _accept_buffer(sizing_eval, buffer_eval)
    accepted_def = buffering_candidate if accepted else sizing_legal
    stages["buffering"]["acceptance"] = {
        "accepted": accepted,
        "reason": reason,
        "next_input_def": str(accepted_def),
    }
    return {
        "status": "pass",
        "outer_index": outer_index,
        "input_def": str(input_def),
        "input_def_sha256": _sha256(input_def),
        "stages": stages,
        "accepted_def": str(accepted_def),
        "accepted_def_sha256": _sha256(accepted_def),
        "accepted_state": "buffered" if accepted else "sizing_legalized",
        "accepted_wns_ns": (buffer_eval if accepted else sizing_eval).get("wns_ns"),
        "accepted_tns_ns": (buffer_eval if accepted else sizing_eval).get("tns_ns"),
    }


def _write_openroad_baseline_tcl(
    *,
    args: argparse.Namespace,
    case: str,
    input_def: Path,
    output_def: Path,
    tcl_path: Path,
) -> None:
    paths = protocol.case_paths(args.benchmark_root, case)
    tech = protocol.technology_paths(args.benchmark_root)
    lines = [f'read_lef {{{tech["tech_lef"]}}}']
    for lef in tech["lefs"]:
        lines.append(f"read_lef {{{lef}}}")
    for liberty in tech["libs"]:
        lines.append(f"read_liberty {{{liberty}}}")
    lines.extend(
        [
            f"read_def -continue_on_errors {{{input_def}}}",
            f"read_sdc {{{paths['sdc']}}}",
            "set_ideal_network [all_clocks]",
            f"source {{{tech['rc_tcl']}}}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
            "global_placement -timing_driven -density 0.8 -init_density_penalty 0.01",
            "detailed_placement",
            "repair_design",
            "repair_timing -setup",
            "estimate_parasitics -placement",
            "write_def " + "{" + str(output_def) + "}",
            "exit",
        ]
    )
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_openroad_baseline(
    *,
    args: argparse.Namespace,
    case: str,
    r0_def: Path,
    case_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    root = case_root / "openroad_baseline"
    output_def = root / f"{case}_openroad_native.def"
    tcl_path = root / "run.tcl"
    _write_openroad_baseline_tcl(
        args=args, case=case, input_def=r0_def, output_def=output_def, tcl_path=tcl_path
    )
    execution = _run(
        [str(args.openroad_bin), "-threads", str(args.openroad_num_threads), "-exit", str(tcl_path)],
        cwd=root,
        log_path=root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    evaluation = (
        _evaluate_def(
            args=args, case=case, input_def=output_def, root=root / "fixed_eval", environment=environment
        )
        if output_def.is_file()
        else {"status": "failed"}
    )
    return {
        "status": "pass" if execution["status"] == "pass" and evaluation.get("status") == "pass" else "failed",
        "execution": execution,
        "output_def": str(output_def),
        "evaluation": evaluation,
    }


def _run_case(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    run_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    case_root = run_root / case
    case_root.mkdir(parents=True, exist_ok=True)
    r0_def = _prepare_r0(args=args, case=case, case_root=case_root, environment=environment)
    current_def = r0_def
    iterations: list[dict[str, Any]] = []
    failures: list[str] = []
    for outer_index in range(int(args.outer_iterations)):
        result = _run_outer_iteration(
            args=args,
            profile=profile,
            case=case,
            outer_index=outer_index,
            input_def=current_def,
            case_root=case_root,
            environment=environment,
        )
        iterations.append(result)
        if result.get("status") != "pass":
            failures.append(f"outer_{outer_index:03d}_{result.get('failed_stage', 'unknown')}")
            break
        current_def = Path(str(result["accepted_def"]))

    valid_iterations = [item for item in iterations if item.get("status") == "pass"]
    selected = max(
        valid_iterations,
        key=lambda item: (
            float(item.get("accepted_tns_ns"))
            if item.get("accepted_tns_ns") is not None
            else -float("inf"),
            float(item.get("accepted_wns_ns"))
            if item.get("accepted_wns_ns") is not None
            else -float("inf"),
        ),
        default=None,
    )
    baseline = (
        _run_openroad_baseline(
            args=args, case=case, r0_def=r0_def, case_root=case_root, environment=environment
        )
        if args.run_openroad_baseline
        else {"status": "disabled"}
    )
    summary = {
        "artifact": "autodmp_physical_staged_alternating_case_result",
        "artifact_version": 1,
        "case": case,
        "status": "pass" if valid_iterations and not failures else "failed",
        "failures": failures,
        "outer_iterations_requested": int(args.outer_iterations),
        "outer_iterations_completed": len(valid_iterations),
        "r0_def": str(r0_def),
        "r0_def_sha256": _sha256(r0_def),
        "iterations": iterations,
        "selected": selected,
        "openroad_baseline": baseline,
        "runtime_sec": time.perf_counter() - started,
    }
    _write_json(case_root / "case_summary.json", summary)
    return summary


def _write_campaign_summary(run_root: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        selected = dict(result.get("selected") or {})
        baseline_eval = dict(dict(result.get("openroad_baseline") or {}).get("evaluation") or {})
        rows.append(
            {
                "case": result.get("case"),
                "status": result.get("status"),
                "outer_iterations": result.get("outer_iterations_requested"),
                "selected_outer_iteration": selected.get("outer_index"),
                "selected_def": selected.get("accepted_def"),
                "selected_wns_ns": selected.get("accepted_wns_ns"),
                "selected_tns_ns": selected.get("accepted_tns_ns"),
                "sequential_outer0_tns_ns": next(
                    (
                        item.get("accepted_tns_ns")
                        for item in result.get("iterations", [])
                        if int(item.get("outer_index", -1)) == 0
                    ),
                    None,
                ),
                "openroad_baseline_tns_ns": baseline_eval.get("tns_ns"),
                "runtime_sec": result.get("runtime_sec"),
                "summary_path": str(run_root / str(result.get("case")) / "case_summary.json"),
            }
        )
    with (run_root / "campaign_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        run_root / "campaign_summary.json",
        {"artifact": "autodmp_physical_staged_alternating_campaign", "results": results},
    )
    lines = [
        "# Physical Staged Alternating Campaign",
        "",
        "The table uses fixed-state DEF-only OpenROAD/OpenSTA evaluation.",
        "",
        "| Case | Status | Selected outer iteration | Selected WNS (ns) | Selected TNS (ns) | Sequential outer-0 TNS (ns) | OpenROAD baseline TNS (ns) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            parsed = _finite_float(value)
            return "n/a" if parsed is None else f"{parsed:.6f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case']}`",
                    str(row["status"]),
                    (
                        "n/a"
                        if row["selected_outer_iteration"] is None
                        else str(row["selected_outer_iteration"])
                    ),
                    fmt(row["selected_wns_ns"]),
                    fmt(row["selected_tns_ns"]),
                    fmt(row["sequential_outer0_tns_ns"]),
                    fmt(row["openroad_baseline_tns_ns"]),
                ]
            )
            + " |"
        )
    (run_root / "campaign_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("alternating_%Y%m%d_%H%M%S"))
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--buffer-profile", type=Path, default=DEFAULT_BUFFER_PROFILE)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--tdp-iterations", type=int, default=3000)
    parser.add_argument(
        "--continuation-tdp-iterations",
        type=int,
        default=100,
        help="Short TDP window used after the first accepted outer state.",
    )
    parser.add_argument("--sizing-iterations", type=int, default=50)
    parser.add_argument("--logical-gpu-id", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default="7")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--openroad-num-threads", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--nesterov-step-cap", type=float, default=1.0)
    parser.add_argument("--placement-overflow-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--continuation-placement-overflow-tolerance",
        type=float,
        default=0.001,
        help="Explicit handoff tolerance for continuation TDP windows.",
    )
    parser.add_argument("--input-r0-def")
    parser.add_argument("--run-openroad-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_root = args.output_root.resolve()
    args.python_bin = args.python_bin.resolve()
    args.openroad_bin = args.openroad_bin.resolve()
    args.buffer_profile = args.buffer_profile.resolve()
    if (
        args.outer_iterations <= 0
        or args.tdp_iterations <= 0
        or args.continuation_tdp_iterations <= 0
        or args.sizing_iterations <= 0
    ):
        raise ValueError("iteration counts must be positive")
    if args.nesterov_step_cap is not None and args.nesterov_step_cap <= 0:
        raise ValueError("nesterov step cap must be positive")
    if args.placement_overflow_tolerance < 0:
        raise ValueError("placement overflow tolerance must be nonnegative")
    if args.continuation_placement_overflow_tolerance < 0:
        raise ValueError(
            "continuation placement overflow tolerance must be nonnegative"
        )
    cases = _as_case_list(args.cases)
    protocol.validate_benchmark_inputs(args.benchmark_root, cases)
    profile = _read_json(args.buffer_profile)
    if not profile:
        raise FileNotFoundError(args.buffer_profile)
    run_root = args.output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    environment = _environment(args)
    manifest = {
        "artifact": "autodmp_physical_staged_alternating_campaign_manifest",
        "artifact_version": 1,
        "run_id": args.run_id,
        "benchmark_root": str(args.benchmark_root),
        "benchmark_root_resolved": str(args.benchmark_root.resolve()),
        "python_bin": str(args.python_bin),
        "openroad_bin": str(args.openroad_bin),
        "openroad_bin_sha256": _sha256(args.openroad_bin),
        "profile": str(args.buffer_profile),
        "profile_sha256": _sha256(args.buffer_profile),
        "cases": cases,
        "outer_iterations": args.outer_iterations,
        "tdp_iterations": args.tdp_iterations,
        "continuation_tdp_iterations": args.continuation_tdp_iterations,
        "sizing_iterations": args.sizing_iterations,
        "seed": args.seed,
        "nesterov_step_cap": args.nesterov_step_cap,
        "placement_overflow_tolerance": args.placement_overflow_tolerance,
        "continuation_placement_overflow_tolerance": (
            args.continuation_placement_overflow_tolerance
        ),
        "stage_order": [
            "tdp",
            "tdp_legalize",
            "sizing",
            "sizing_legalize",
            "buffering",
            "buffer_acceptance",
        ],
    }
    _write_json(run_root / "campaign_manifest.json", manifest)
    results = []
    for case in cases:
        try:
            result = _run_case(
                args=args,
                profile=profile,
                case=case,
                run_root=run_root,
                environment=environment,
            )
        except Exception as error:  # preserve a durable failure artifact
            result = {"case": case, "status": "failed", "failures": [repr(error)]}
            _write_json(run_root / case / "case_summary.json", result)
        results.append(result)
        selected = dict(result.get("selected") or {})
        print(
            f"{case}: status={result.get('status')} "
            f"selected_tns={selected.get('accepted_tns_ns')} "
            f"failures={','.join(result.get('failures', [])) or '-'}"
        )
    _write_campaign_summary(run_root, results)
    print(f"run_root: {run_root}")
    print(f"summary: {run_root / 'campaign_summary.md'}")
    return 0 if all(result.get("status") == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
