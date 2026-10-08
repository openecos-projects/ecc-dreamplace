#!/usr/bin/env python3
"""Run the controlled random-init placement + buffering validation matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from random_init_snapshot import (
    parse_def_component_placements,
    sha256_file,
    xy_coordinate_sha256,
)


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
METHODS = ("m1", "m2", "m3", "m4")
OPENROAD_REWEIGHT_ONLY_METHOD = "OR-TDP-reweight-only"
OPENROAD_GPL_PLAIN_METHOD = "OR-GPL-plain"
OPENROAD_NATIVE_FULL_FLOW_METHOD = "OR-native-full-flow"
OPENROAD_PLACEMENT_METHODS = (
    "m1",
    OPENROAD_REWEIGHT_ONLY_METHOD,
    OPENROAD_GPL_PLAIN_METHOD,
    OPENROAD_NATIVE_FULL_FLOW_METHOD,
)
OVERFLOW_MILESTONE_ARMS = {
    "P": {
        "placement_timing_enabled": False,
        "pin2pin_enabled": False,
        "sizing_enabled": False,
        "route_b_enabled": False,
    },
    "TDP": {
        "placement_timing_enabled": True,
        "pin2pin_enabled": False,
        "sizing_enabled": False,
        "route_b_enabled": False,
    },
    "TDP-pin2pin": {
        "placement_timing_enabled": False,
        "pin2pin_enabled": True,
        "sizing_enabled": False,
        "route_b_enabled": False,
    },
    "TDP-pin2pin-SB": {
        "placement_timing_enabled": False,
        "pin2pin_enabled": True,
        "sizing_enabled": True,
        "route_b_enabled": True,
    },
    "S": {
        "placement_timing_enabled": True,
        "pin2pin_enabled": False,
        "sizing_enabled": True,
        "route_b_enabled": False,
    },
    "B": {
        "placement_timing_enabled": True,
        "pin2pin_enabled": False,
        "sizing_enabled": False,
        "route_b_enabled": True,
    },
    "SB": {
        "placement_timing_enabled": True,
        "pin2pin_enabled": False,
        "sizing_enabled": True,
        "route_b_enabled": True,
    },
}
OVERFLOW_MILESTONE_METHODS = tuple(OVERFLOW_MILESTONE_ARMS)
EXPERIMENT_METHODS = (
    METHODS
    + (
        "m4_no_density",
        OPENROAD_REWEIGHT_ONLY_METHOD,
        OPENROAD_GPL_PLAIN_METHOD,
        OPENROAD_NATIVE_FULL_FLOW_METHOD,
    )
    + OVERFLOW_MILESTONE_METHODS
)
FILLER_COMPATIBLE_METHODS = frozenset(
    {"P", "TDP", "TDP-pin2pin", "TDP-pin2pin-SB"}
)
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "joint_place_sizing_buffering"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OPENROAD = Path("/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
DEFAULT_PLACEMENT_OPTIMIZER = "nesterov"
DEFAULT_OVERFLOW_MILESTONE_ITERATIONS = 500
DEF_ONLY_GAP_RELATIVE_TOLERANCE = 1.0e-5
# The OpenROAD bridge returns Liberty user time units (1 ps for ICCAD24).
DEF_ONLY_GAP_ABSOLUTE_TOLERANCE = 5.0
METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.*)$")
POWER_TOTAL_RE = re.compile(
    r"^\s*Total\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)
M1_RESIZED_RE = re.compile(r"\[INFO RSZ-0039\] Resized (\d+) instances\.")
M1_INSERTED_RE = re.compile(r"\[INFO RSZ-0038\] Inserted (\d+) buffers")
M1_REBUFFER_RE = re.compile(
    r"\[INFO GPL-0109\].*gcells created: (\d+), deleted: (\d+)"
)
M1_AREA_RE = re.compile(
    r"\[INFO GPL-0107\].*delta area: ([-+.0-9eE]+) um\^2"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: Any) -> int | None:
    value = _as_float(value)
    return None if value is None else int(value)


def _def_only_gap_exceeds_tolerance(physical: dict[str, Any]) -> bool:
    """Compare replay and in-process OpenSTA values in their shared raw unit.

    The OpenROAD bridge reports timing in the Liberty user time unit (1 ps for
    the ICCAD24 libraries), while the publication-facing evaluator converts
    those values to ns.  The two fresh OpenSTA queries should therefore be
    compared with a scale-aware tolerance, rather than a fixed ns tolerance
    applied to raw bridge values.
    """
    gaps = dict(physical.get("def_only_vs_in_process_gap", {}) or {})
    replay = dict(physical.get("def_only_opensta_after", {}) or {})
    in_process = dict(physical.get("opensta_after", {}) or {})
    for name in ("wns", "tns"):
        gap = _as_float(gaps.get(name))
        replay_value = _as_float(replay.get(name))
        in_process_value = _as_float(in_process.get(name))
        if gap is None:
            return True
        if replay_value is None and in_process_value is None:
            if abs(gap) > DEF_ONLY_GAP_ABSOLUTE_TOLERANCE:
                return True
            continue
        if replay_value is None or in_process_value is None:
            return True
        reference = max(abs(replay_value), abs(in_process_value), 1.0)
        tolerance = max(
            DEF_ONLY_GAP_ABSOLUTE_TOLERANCE,
            DEF_ONLY_GAP_RELATIVE_TOLERANCE * reference,
        )
        if abs(gap) > tolerance:
            return True
    return False


def _buffer_action_count_mismatch(
    physical: dict[str, Any], expected_count: int
) -> bool:
    """Allow a rejected all-or-nothing preflight to finish as a no-op."""
    accepted = int(physical.get("accepted_buffer_count", 0) or 0)
    actual = int(physical.get("buffer_count", 0) or 0)
    if expected_count > 0 and accepted == 0 and actual == 0:
        preflight_rejected = any(
            str(row.get("stage") or "") == "insert_buffers"
            and str(row.get("status") or "") == "preflight_rejected"
            and int(row.get("accepted_action_count", 0) or 0) == 0
            for row in physical.get("physical_action_trace", ()) or ()
        )
        if (
            preflight_rejected
            and physical.get("qor_status")
            == "buffer_preflight_rejected_preserved_prebuffer_state"
        ):
            return False
    return accepted != expected_count or actual != expected_count


def _resolve_enable_fillers(
    requested: bool | None,
    methods: list[str],
) -> bool:
    selected = set(methods)
    if requested is None:
        return bool(selected) and selected <= FILLER_COMPATIBLE_METHODS
    if requested and not selected <= FILLER_COMPATIBLE_METHODS:
        raise ValueError(
            "--enable-fillers is currently qualified only for placement arms: "
            "P, TDP, TDP-pin2pin, and TDP-pin2pin-SB"
        )
    return bool(requested)


def _tcl_quote(value: Path | str) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def _tool_output(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _git_head(path: Path) -> str:
    return _tool_output(["git", "rev-parse", "HEAD"], path).splitlines()[0]


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    started = time.perf_counter()
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


def _selected(values: list[str] | None, allowed: tuple[str, ...]) -> list[str]:
    if not values:
        return list(allowed)
    result = []
    for raw in values:
        result.extend(item.strip() for item in raw.split(",") if item.strip())
    invalid = sorted(set(result) - set(allowed))
    if invalid:
        raise ValueError("unsupported selection: " + ", ".join(invalid))
    return result


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


def _placement_budget(
    params_json: Path,
    optimizer_override: str | None = None,
) -> dict[str, Any]:
    params = _read_json(params_json)
    stages = list(params.get("global_place_stages") or ())
    if not stages:
        raise ValueError(f"{params_json}: missing global_place_stages")
    resolved = []
    total = 0
    for index, stage in enumerate(stages):
        iteration = int(stage.get("iteration", 0))
        llambda = int(stage.get("Llambda_density_weight_iteration", 1))
        lsub = int(stage.get("Lsub_iteration", 1))
        steps = iteration * llambda * lsub
        total += steps
        source_optimizer = stage.get("optimizer")
        resolved.append(
            {
                "stage": index,
                "iteration": iteration,
                "Llambda_density_weight_iteration": llambda,
                "Lsub_iteration": lsub,
                "optimizer_steps": steps,
                "optimizer": (
                    str(optimizer_override)
                    if optimizer_override is not None
                    else source_optimizer
                ),
                "source_optimizer": source_optimizer,
            }
        )
    if total <= 0:
        raise ValueError(f"{params_json}: nonpositive placement budget")
    return {"P": total, "stages": resolved}


def _base_placer_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    result_dir: Path,
    enable_fillers: bool = False,
    legalize: bool = False,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(inputs["params_json"]),
        "--place-io-engine",
        "openroad",
        "--workspace",
        str(inputs["workspace"]),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--def-input",
        str(input_def),
        "--verilog-input",
        str(inputs["verilog"]),
        "--sdc",
        str(inputs["sdc"]),
        "--rc-tcl",
        str(tech["rc_tcl"]),
        "--tech-lef",
        str(tech["tech_lef"]),
        "--random-center-init",
        "0",
        "--enable-fillers",
        "1" if enable_fillers else "0",
        "--legalize",
        "1" if legalize else "0",
    ]
    for lef in tech["lef"]:
        command.extend(("--lef", str(lef)))
    for liberty in tech["lib"]:
        command.extend(("--lib", str(liberty)))
    return command


def build_r0_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    case_root: Path,
    seed: int,
) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(SCRIPT_DIR / "random_init_snapshot.py"),
        "--case",
        case,
        "--params-json",
        str(inputs["params_json"]),
        "--workspace",
        str(inputs["workspace"]),
        "--source-def",
        str(inputs["def"]),
        "--verilog",
        str(inputs["verilog"]),
        "--sdc",
        str(inputs["sdc"]),
        "--rc-tcl",
        str(tech["rc_tcl"]),
        "--tech-lef",
        str(tech["tech_lef"]),
        "--seed",
        str(seed),
        "--result-dir",
        str(case_root / "r0" / "result"),
        "--output-def",
        str(case_root / "r0" / "R0.def"),
        "--manifest",
        str(case_root / "r0" / "R0_manifest.json"),
    ]
    for lef in tech["lef"]:
        command.extend(("--lef", str(lef)))
    for liberty in tech["lib"]:
        command.extend(("--lib", str(liberty)))
    return command


def build_m2_m4_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    r0_def: Path,
    method_root: Path,
    iterations: int,
    warmup_iterations: int,
    outer_iterations: int,
    gpu_id: int,
    placement_optimizer: str,
    route_b_enabled: bool,
    virtual_density_enabled: bool,
) -> list[str]:
    result_dir = method_root / "autodmp_result"
    output_def = method_root / f"{case}_raw.def"
    command = _base_placer_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        input_def=r0_def,
        result_dir=result_dir,
    )
    command.extend(
        (
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--gpu",
            "1",
            "--gpu-id",
            str(gpu_id),
            "--iterations",
            str(iterations * outer_iterations + warmup_iterations),
            "--placement-optimizer",
            str(placement_optimizer),
            "--joint-buffer-outer-iterations",
            str(outer_iterations),
            "--joint-segment-virtual-window-steps",
            str(iterations),
            "--joint-segment-virtual-warmup-steps",
            str(warmup_iterations),
            "--joint-buffer-max-per-segment",
            "3",
            "--joint-segment-route-b-enabled",
            "1" if route_b_enabled else "0",
            "--joint-segment-virtual-density-enabled",
            "1" if virtual_density_enabled else "0",
            "--buffering-commit-enabled",
            "1" if route_b_enabled else "0",
            "--buffering-committed-def-path",
            str(method_root / f"{case}_committed.def"),
            "--output-def",
            str(output_def),
        )
    )
    return command


def build_m3_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    m2_def: Path,
    method_root: Path,
    rounds: int,
    gpu_id: int,
) -> list[str]:
    command = _base_placer_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        input_def=m2_def,
        result_dir=method_root / "autodmp_result",
    )
    command.extend(
        (
            "--flow-kind",
            "buffering",
            "--buffering-mode",
            "segment",
            "--buffering-segment-strategy",
            "discrete_net_gradient",
            "--with-sta",
            "--gpu",
            "1",
            "--gpu-id",
            str(gpu_id),
            "--iterations",
            "1",
            "--buffering-continuous-steps",
            str(rounds),
            "--buffering-max-repeaters-per-segment",
            "3",
            "--buffering-segment-count-z-init",
            "0",
            "--buffering-fixed-bsu-index",
            "7",
            "--buffering-discrete-count-proximal-lambda",
            "0.0001",
            "--buffering-segment-count-timing-backend",
            "cpp_cuda_segment_transfer_explicit_autograd",
            "--buffering-segment-transfer-backend",
            "segment_transfer_native",
            "--buffering-segment-integer-projection-interval",
            "0",
            "--buffering-segment-integer-projection-start-step",
            "0",
            "--buffering-commit-enabled",
            "1",
            "--buffering-committed-def-path",
            str(method_root / f"{case}_committed.def"),
        )
    )
    return command


def build_overflow_milestone_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    r0_def: Path,
    method_root: Path,
    iterations: int,
    gpu_id: int,
    placement_optimizer: str,
    placement_timing_enabled: bool,
    sizing_enabled: bool,
    route_b_enabled: bool,
    enable_fillers: bool = False,
    legalize: bool = False,
    stop_overflow: float = 0.1,
    timing_grad_balance_target_ratio: float = 0.1,
    timing_topology_enable_overflow_threshold: float = 0.3,
    pin2pin_enabled: bool = False,
    pin2pin_weight: float = 2.5e-5,
    pin2pin_min_weight: float = 10.0,
    pin2pin_max_weight: float = 50.0,
    pin2pin_accumulate_weight: float = 0.2,
    net_weighting_npaths: float = 0.0,
    pin2pin_update_interval: int = 15,
    joint_segment_milestones=(0.30, 0.25, 0.20, 0.15, 0.10),
    joint_segment_sizing_rounds: int = 5,
    joint_segment_sizing_up_percent: float = 10.0,
    joint_segment_buffering_rounds: int = 1,
    joint_segment_buffering_selection_fraction: float = 0.001,
    joint_segment_pin2pin_rebootstrap_enabled: bool = True,
    num_bins_x: int | None = None,
    num_bins_y: int | None = None,
    plot_flag: int | None = None,
) -> list[str]:
    """Build one placement control or overflow-milestone action arm."""
    command = _base_placer_command(
        python_bin=python_bin,
        benchmark_root=benchmark_root,
        case=case,
        input_def=r0_def,
        result_dir=method_root / "autodmp_result",
        enable_fillers=enable_fillers,
        legalize=legalize,
    )
    commit_enabled = bool(sizing_enabled or route_b_enabled)
    command.extend(
        (
            "--flow-kind",
            "joint" if commit_enabled else "placement",
            "--gpu",
            "1",
            "--gpu-id",
            str(gpu_id),
            "--iterations",
            str(iterations),
            "--placement-optimizer",
            str(placement_optimizer),
            "--diff-timing-driven-placement",
            "1" if placement_timing_enabled else "0",
            "--enable-net-weighting",
            "1" if pin2pin_enabled else "0",
            "--stop-overflow",
            str(float(stop_overflow)),
        )
    )
    if placement_timing_enabled or pin2pin_enabled:
        command.extend(
            (
                "--with-sta",
                "--timing-topology-enable-overflow-threshold",
                str(timing_topology_enable_overflow_threshold),
            )
        )
    if num_bins_x is not None:
        command.extend(("--num-bins-x", str(num_bins_x)))
    if num_bins_y is not None:
        command.extend(("--num-bins-y", str(num_bins_y)))
    if placement_timing_enabled:
        command.extend(
            (
                "--timing-topology-refresh-interval",
                "10",
                "--timing-grad-balance-target-ratio",
                str(timing_grad_balance_target_ratio),
            )
        )
    if pin2pin_enabled:
        command.extend(
            (
                "--net-weighting-scheme",
                "pin2pin",
                "--pin2pin-weight",
                str(pin2pin_weight),
                "--pin2pin-min-weight",
                str(pin2pin_min_weight),
                "--pin2pin-max-weight",
                str(pin2pin_max_weight),
                "--pin2pin-accumulate-weight",
                str(pin2pin_accumulate_weight),
                "--net-weighting-npaths",
                str(net_weighting_npaths),
                "--net-weighting-update-interval",
                str(int(pin2pin_update_interval)),
            )
        )
    if commit_enabled:
        command.extend(
            (
                "--joint-quality-profile",
                "segment_count_direct_joint_v1",
                "--joint-buffer-outer-iterations",
                "1",
                "--joint-buffer-max-per-segment",
                "3",
                "--joint-segment-sizing-enabled",
                "1" if sizing_enabled else "0",
                "--joint-segment-route-b-enabled",
                "1" if route_b_enabled else "0",
                "--joint-segment-virtual-density-enabled",
                "1" if route_b_enabled else "0",
                "--buffering-commit-enabled",
                "1",
                "--buffering-committed-def-path",
                str(method_root / f"{case}_committed.def"),
            )
        )
        for milestone in joint_segment_milestones:
            command.extend(("--joint-segment-milestone", str(float(milestone))))
        command.extend(
            (
                "--joint-segment-sizing-rounds",
                str(int(joint_segment_sizing_rounds)),
                "--joint-segment-sizing-up-percent",
                str(float(joint_segment_sizing_up_percent)),
                "--joint-segment-buffering-rounds",
                str(int(joint_segment_buffering_rounds)),
                "--joint-segment-buffering-selection-fraction",
                str(float(joint_segment_buffering_selection_fraction)),
                "--joint-segment-pin2pin-rebootstrap-enabled",
                "1" if joint_segment_pin2pin_rebootstrap_enabled else "0",
            )
        )
    else:
        command.extend(("--buffering-commit-enabled", "0"))
    command.extend(("--output-def", str(method_root / f"{case}_raw.def")))
    if plot_flag is not None:
        command.extend(("--plot", str(int(plot_flag))))
    return command


def _tcl_preamble(benchmark_root: Path, case: str, def_input: Path) -> list[str]:
    inputs = _case_inputs(benchmark_root, case)
    tech = _asap7_inputs(benchmark_root)
    lines = [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        "proc safe_metric {name script default_value} {",
        "  if {[catch {uplevel 1 $script} value]} { emit_metric $name $default_value } else { emit_metric $name $value }",
        "}",
        "proc refresh_timing {} {",
        "  if {[llength [info commands update_timing]]} { update_timing } else { catch {worst_slack -max} }",
        "}",
        "proc dump_coordinates {path} {",
        "  set fp [open $path w]",
        "  foreach inst [[ord::get_db_block] getInsts] {",
        "    set status [$inst getPlacementStatus]",
        "    if {$status eq \"FIRM\" || $status eq \"LOCKED\" || $status eq \"COVER\"} { continue }",
        "    lassign [$inst getLocation] x y",
        '    puts $fp "[$inst getName]\t$x\t$y"',
        "  }",
        "  close $fp",
        "}",
        "proc dump_endpoint_slacks {path} {",
        "  set fp [open $path w]",
        '  puts $fp "endpoint,slack_ns"',
        "  set endpoint_count 0",
        "  set violating_count 0",
        "  set groups [sta::path_group_names]",
        "  foreach endpoint [sta::endpoints] {",
        "    set best_slack {}",
        "    foreach group $groups {",
        "      if {![catch {sta::endpoint_slack $endpoint $group max} slack]} {",
        "        if {$best_slack eq {} || $slack < $best_slack} { set best_slack $slack }",
        "      }",
        "    }",
        "    if {$best_slack ne {}} {",
        "      incr endpoint_count",
        "      if {$best_slack < 0.0} { incr violating_count }",
        '      puts $fp "[get_full_name $endpoint],$best_slack"',
        "    }",
        "  }",
        "  close $fp",
        "  emit_metric endpoint_count $endpoint_count",
        "  emit_metric violating_endpoint_count $violating_count",
        "}",
        "proc check_type_summary {flag} {",
        '  set path [file join [pwd] "check_type_[pid]_[string map {- _} $flag].rpt"]',
        "  if {[catch {eval report_check_types $flag -no_line_splits -digits 6 > $path}]} { return {0 0.0} }",
        "  set fp [open $path r]",
        "  set lines [split [read $fp] \n]",
        "  close $fp",
        "  file delete -force $path",
        "  set count 0",
        "  set total 0.0",
        "  foreach line $lines {",
        "    if {[regexp {([-+.0-9]+)[[:space:]]+\\(VIOLATED\\)} $line -> slack]} {",
        "      incr count",
        "      set total [expr {$total - $slack}]",
        "    }",
        "  }",
        "  return [list $count $total]",
        "}",
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
            f"read_sdc {_tcl_quote(inputs['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
        )
    )
    return lines


def generate_m1_tcl(
    *,
    benchmark_root: Path,
    case: str,
    r0_def: Path,
    output_def: Path,
    pre_gpl_coordinates: Path,
    target_density: float,
    reweight_only: bool = False,
    timing_driven: bool = True,
) -> str:
    if reweight_only and not timing_driven:
        raise ValueError("reweight_only requires timing_driven")
    timing_flags = []
    if timing_driven:
        timing_flags.append("-timing_driven")
    if reweight_only:
        timing_flags.append("-timing_driven_reweight_only")
    timing_suffix = "" if not timing_flags else " " + " ".join(timing_flags)
    lines = _tcl_preamble(benchmark_root, case, r0_def)
    lines.extend(
        (
            f"dump_coordinates {_tcl_quote(pre_gpl_coordinates)}",
            "estimate_parasitics -placement",
            "refresh_timing",
            "set gpl_start_us [clock microseconds]",
            f"global_placement -skip_initial_place{timing_suffix} "
            f"-density {target_density:.12g} -init_density_penalty 0.01",
            'emit_metric "core_runtime_sec" [expr {([clock microseconds] - $gpl_start_us) / 1000000.0}]',
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def generate_openroad_native_full_flow_tcl(
    *,
    benchmark_root: Path,
    case: str,
    r0_def: Path,
    output_def: Path,
    pre_gpl_coordinates: Path,
    target_density: float,
) -> str:
    lines = _tcl_preamble(benchmark_root, case, r0_def)
    lines.extend(
        (
            f"dump_coordinates {_tcl_quote(pre_gpl_coordinates)}",
            "estimate_parasitics -placement",
            "refresh_timing",
            "set method_start_us [clock microseconds]",
            f"global_placement -skip_initial_place -timing_driven "
            f"-density {target_density:.12g} -init_density_penalty 0.01",
            "detailed_placement",
            "estimate_parasitics -placement",
            "refresh_timing",
            "repair_design",
            "detailed_placement",
            "estimate_parasitics -placement",
            "refresh_timing",
            "repair_timing -setup",
            "detailed_placement",
            'if {[catch {check_placement -verbose} message]} { emit_metric "placement_valid" 0 } else { emit_metric "placement_valid" 1 }',
            "estimate_parasitics -placement",
            "refresh_timing",
            'emit_metric "core_runtime_sec" [expr {([clock microseconds] - $method_start_us) / 1000000.0}]',
            'safe_metric "source_wns_ns" {worst_slack -max} ""',
            'safe_metric "source_tns_ns" {total_negative_slack -max} ""',
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def generate_evaluation_tcl(
    *,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_def: Path,
    endpoint_csv: Path,
    power_report: Path,
    closure: bool,
) -> str:
    lines = _tcl_preamble(benchmark_root, case, input_def)
    lines.extend(("estimate_parasitics -placement", "refresh_timing"))
    if closure:
        lines.extend(
            (
                "repair_design",
                "detailed_placement",
                "estimate_parasitics -placement",
                "refresh_timing",
                "repair_timing -setup",
            )
        )
    lines.extend(
        (
            "detailed_placement",
            'if {[catch {check_placement -verbose} message]} { emit_metric "placement_valid" 0 } else { emit_metric "placement_valid" 1 }',
            "estimate_parasitics -placement",
            "refresh_timing",
            'safe_metric "wns_ns" {worst_slack -max} ""',
            'safe_metric "tns_ns" {total_negative_slack -max} ""',
            "lassign [check_type_summary -max_slew] slew_count slew_total",
            "lassign [check_type_summary -max_capacitance] cap_count cap_total",
            'emit_metric "slew_violation_count" $slew_count',
            'emit_metric "slew_violation_total" $slew_total',
            'emit_metric "cap_violation_count" $cap_count',
            'emit_metric "cap_violation_total" $cap_total',
            'safe_metric "instance_count" {llength [[ord::get_db_block] getInsts]} ""',
            'safe_metric "net_count" {llength [[ord::get_db_block] getNets]} ""',
            'safe_metric "area_um2" {expr {[rsz::design_area] * 1.0e12}} ""',
            f"dump_endpoint_slacks {_tcl_quote(endpoint_csv)}",
            f"report_power > {_tcl_quote(power_report)}",
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def _parse_metrics(log_path: Path) -> dict[str, Any]:
    result = {}
    if not log_path.is_file():
        return result
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = METRIC_RE.match(line.strip())
        if match is None:
            continue
        key, raw = match.groups()
        parsed = _as_float(raw)
        result[key] = raw if parsed is None else parsed
    return result


def _parse_power_total(path: Path) -> float | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = POWER_TOTAL_RE.match(line)
        if match is not None:
            return _as_float(match.group(4))
    return None


def _parse_m1_native_actions(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"status": "missing", "log_path": str(log_path)}
    resized = 0
    inserted = 0
    rebuffer_rounds = []
    repair_area_delta_um2 = 0.0
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = M1_RESIZED_RE.search(line)
        if match is not None:
            resized += int(match.group(1))
        match = M1_INSERTED_RE.search(line)
        if match is not None:
            inserted += int(match.group(1))
        match = M1_REBUFFER_RE.search(line)
        if match is not None:
            created = int(match.group(1))
            deleted = int(match.group(2))
            rebuffer_rounds.append(
                {
                    "created": created,
                    "deleted": deleted,
                    "net_instance_delta": created - deleted,
                }
            )
        match = M1_AREA_RE.search(line)
        if match is not None:
            repair_area_delta_um2 += float(match.group(1))
    return {
        "status": "ok",
        "log_path": str(log_path),
        "explicit_resized_instances": resized,
        "explicit_inserted_buffers": inserted,
        "repair_design_rounds": rebuffer_rounds,
        "repair_design_net_instance_delta": sum(
            row["net_instance_delta"] for row in rebuffer_rounds
        ),
        "repair_design_area_delta_um2": repair_area_delta_um2,
    }


def _reweight_only_baseline_failures(method_result: dict[str, Any]) -> list[str]:
    failures = []
    if not method_result.get("r0_coordinate_match"):
        failures.append("r0_coordinate_mismatch")
    actions = dict(method_result.get("native_timing_driven_actions", {}) or {})
    if any(
        (
            int(actions.get("explicit_resized_instances", 0) or 0),
            int(actions.get("explicit_inserted_buffers", 0) or 0),
            int(actions.get("repair_design_net_instance_delta", 0) or 0),
        )
    ) or abs(float(actions.get("repair_design_area_delta_um2", 0.0) or 0.0)) > 0.0:
        failures.append("native_timing_driven_action_detected")
    mutation = dict(method_result.get("track_a", {}).get("mutation", {}) or {})
    if int(mutation.get("added_count", 0) or 0) != 0:
        failures.append("instance_added")
    if int(mutation.get("removed_count", 0) or 0) != 0:
        failures.append("instance_removed")
    return failures


def _coordinate_tsv_hash(
    path: Path,
    names: set[str] | None = None,
) -> tuple[str, int]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and (names is None or fields[0] in names):
            rows.append((fields[0], int(fields[1]), int(fields[2])))
    rows.sort()
    serialized = "".join(f"{name}\t{x}\t{y}\n" for name, x, y in rows)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), len(rows)


def _m1_coordinate_identity(
    coordinate_path: Path,
    r0_def: Path,
    r0_manifest: dict[str, Any],
) -> dict[str, Any]:
    coordinate_hash = None
    coordinate_count = 0
    if coordinate_path.is_file():
        r0_records = parse_def_component_placements(r0_def)
        movable_names = {
            name
            for name, record in r0_records.items()
            if (record.get("placement") or {}).get("status")
            not in {"FIXED", "COVER"}
        }
        coordinate_hash, coordinate_count = _coordinate_tsv_hash(
            coordinate_path,
            movable_names,
        )
    expected_hash = r0_manifest.get("coordinate_identity", {}).get(
        "movable_nodes_xy_sha256"
    )
    return {
        "pre_gpl_coordinate_sha256": coordinate_hash,
        "pre_gpl_coordinate_count": coordinate_count,
        "r0_coordinate_sha256": expected_hash,
        "r0_coordinate_match": coordinate_hash == expected_hash,
    }


def _external_r0_manifest(r0_def: Path, case: str, seed: int) -> dict[str, Any]:
    records = parse_def_component_placements(r0_def)
    movable_names = {
        name
        for name, record in records.items()
        if (record.get("placement") or {}).get("status") not in {"FIXED", "COVER"}
    }
    return {
        "artifact": "random_center_r0_manifest",
        "artifact_version": 1,
        "status": "pass",
        "case": case,
        "seed": int(seed),
        "randomization": "external_prepared_snapshot",
        "source_def": str(r0_def),
        "source_def_sha256": sha256_file(r0_def),
        "r0_def": str(r0_def),
        "r0_def_sha256": sha256_file(r0_def),
        "counts": {
            "movable_nodes": len(movable_names),
        },
        "coordinate_identity": {
            "movable_nodes_xy_sha256": xy_coordinate_sha256(
                records,
                movable_names,
            ),
        },
        "failures": [],
    }


def _external_r0_path(args: argparse.Namespace, case: str) -> Path | None:
    if args.input_r0_def is None:
        return None
    value = str(args.input_r0_def)
    if "{case}" in value:
        value = value.format(case=case)
    return Path(value).resolve()


def _def_added_instances(before: Path, after: Path) -> dict[str, Any]:
    before_records = parse_def_component_placements(before)
    after_records = parse_def_component_placements(after)
    added = sorted(set(after_records) - set(before_records))
    removed = sorted(set(before_records) - set(after_records))
    added_master_counts = Counter(
        str(after_records[name].get("master", "")) for name in added
    )
    return {
        "before_count": len(before_records),
        "after_count": len(after_records),
        "added_count": len(added),
        "removed_count": len(removed),
        "added_name_sha256": hashlib.sha256(
            ("\n".join(added) + "\n").encode("utf-8")
        ).hexdigest(),
        "removed_name_sha256": hashlib.sha256(
            ("\n".join(removed) + "\n").encode("utf-8")
        ).hexdigest(),
        "added_instance_sample": added[:32],
        "removed_instance_sample": removed[:32],
        "added_master_counts": dict(sorted(added_master_counts.items())),
    }


def _run_openroad_tcl(
    *,
    openroad_bin: Path,
    tcl_path: Path,
    log_path: Path,
    openroad_num_threads: int,
) -> dict[str, Any]:
    return _run_command(
        [
            str(openroad_bin),
            "-exit",
            "-no_init",
            "-threads",
            str(int(openroad_num_threads)),
            str(tcl_path),
        ],
        cwd=tcl_path.parent,
        log_path=log_path,
    )


def _method_raw_def(case_root: Path, case: str, method: str) -> Path:
    if (
        method in OVERFLOW_MILESTONE_ARMS
        and (
            OVERFLOW_MILESTONE_ARMS[method]["sizing_enabled"]
            or OVERFLOW_MILESTONE_ARMS[method]["route_b_enabled"]
        )
    ):
        return (
            case_root
            / method
            / "autodmp_result"
            / f"{case}_segment_joint_final.def"
        )
    if method in {"m3", "m4", "m4_no_density"}:
        return case_root / method / f"{case}_committed.def"
    return case_root / method / f"{case}_raw.def"


def _milestone_evidence(method_root: Path, case: str) -> dict[str, Any]:
    path = method_root / "autodmp_result" / f"{case}_joint_flow_summary.json"
    placement_debug_path = (
        method_root
        / "autodmp_result"
        / f"{case}_placement_debug_summary.json"
    )
    summary = dict(_read_json(path).get("summary", {}) or {})
    placement_debug = _read_json(placement_debug_path)
    timing_gate = dict(placement_debug.get("diff_tdp_gate_summary", {}) or {})
    timing_grad_balance = dict(
        placement_debug.get("timing_grad_balance", {}) or {}
    )
    legacy_net_weight_gate = dict(
        placement_debug.get("legacy_net_weight_gate_summary", {}) or {}
    )
    legacy_net_weight = dict(
        placement_debug.get("legacy_net_weight_summary", {}) or {}
    )
    state = dict(summary.get("segment_direct_joint", {}) or {})
    terminal_buffer_reselection = dict(
        summary.get("terminal_segment_route_b_reselection", {}) or {}
    )
    pair_generations = dict(summary.get("pin2pin_pair_generations", {}) or {})
    trace = [dict(row) for row in state.get("trace", ())]
    consumed = [row for row in trace if row.get("status") == "consumed"]
    completed = [
        row
        for row in trace
        if row.get("status") in {"consumed", "skipped_timing_clean"}
    ]

    sizing_action_count = 0
    buffer_action_count = 0
    stage_runtime_ms: dict[str, float] = {}
    per_milestone = []
    for row in consumed:
        transition = dict(row.get("transition", {}) or {})
        sizing = dict(transition.get("sizing", {}) or {})
        sizing_micro_loop = dict(
            transition.get("sizing_micro_loop", {}) or {}
        )
        if sizing_micro_loop:
            sizing_delta_count = int(
                sizing_micro_loop.get("changed_instance_count", 0) or 0
            )
            sizing_area_delta = float(
                sizing_micro_loop.get("area_delta_internal", 0.0) or 0.0
            )
            sizing_rounds = [
                dict(record)
                for record in sizing_micro_loop.get("rounds", ())
            ]
        else:
            sizing_delta_count = int(sizing.get("num_changed_instances", 0) or 0)
            sizing_area_delta = float(sizing.get("area_delta_internal", 0.0) or 0.0)
            sizing_rounds = []
        sizing_action_count += sizing_delta_count
        route_b = dict(transition.get("route_b", {}) or {})
        buffer_action_count += int(
            route_b.get(
                "accepted_action_count",
                transition.get("accepted_action_count", 0),
            )
            or 0
        )
        runtime_ms = {
            str(name): float(value or 0.0)
            for name, value in dict(transition.get("runtime_ms", {}) or {}).items()
        }
        if sizing_micro_loop:
            runtime_ms["sizing_micro_loop"] = sum(
                float(record.get("runtime_ms", 0.0) or 0.0)
                for record in sizing_rounds
            )
            runtime_ms["buffer_timing_forward_backward"] = float(
                dict(transition.get("buffering_gradient", {}) or {}).get(
                    "runtime_ms", 0.0
                )
                or 0.0
            )
            runtime_ms["route_b"] = float(route_b.get("runtime_ms", 0.0) or 0.0)
            runtime_ms["total"] = sum(
                value for name, value in runtime_ms.items() if name != "total"
            )
        for name, value in runtime_ms.items():
            stage_runtime_ms[name] = stage_runtime_ms.get(name, 0.0) + value
        per_milestone.append(
            {
                "event_id": int(row.get("event_id", -1)),
                "milestone": float(row["milestone"]),
                "iteration": int(row["iteration"]),
                "queued_iteration": _as_int(row.get("queued_iteration")),
                "queue_before": list(row.get("queue_before", ()) or ()),
                "queue_after": list(
                    row.get("queue_after", row.get("queue_after_begin", ())) or ()
                ),
                "pair_generation_before": _as_int(
                    row.get(
                        "pair_generation_before_milestone",
                        transition.get("pair_generation_before"),
                    )
                ),
                "pair_generation_after": _as_int(
                    transition.get("pair_generation_after")
                ),
                "transaction_stages": [
                    record.get("stage")
                    for record in row.get("stage_trace", ())
                ],
                "sizing_action_count": sizing_delta_count,
                "sizing_area_delta_internal": sizing_area_delta,
                "sizing_micro_loop": sizing_micro_loop,
                "sizing_rounds": sizing_rounds,
                "buffer_action_count": int(
                    route_b.get("accepted_action_count", 0) or 0
                ),
                "runtime_ms": runtime_ms,
            }
        )
    return {
        "artifact_path": str(path),
        "schedule_mode": summary.get("segment_action_schedule_mode"),
        "reached_milestones": [float(row["milestone"]) for row in completed],
        "consumed_milestones": [float(row["milestone"]) for row in consumed],
        "timing_clean_skips": [
            copy
            for copy in trace
            if copy.get("status") == "skipped_timing_clean"
        ],
        "milestone_records": trace,
        "sizing_action_count": sizing_action_count,
        "buffer_action_count": buffer_action_count,
        "terminal_buffer_reselection": terminal_buffer_reselection,
        "stage_runtime_ms": stage_runtime_ms,
        "per_milestone": per_milestone,
        "terminal_failure": state.get("terminal_failure"),
        "pending_queue": list(state.get("queue", ()) or ()),
        "pending_event": state.get("pending_event"),
        "activation_iteration": state.get("activation_iteration"),
        "queue_trace": list(state.get("queue_trace", ()) or ()),
        "pair_generations": pair_generations,
        "incomplete_pending_milestone": summary.get(
            "incomplete_pending_milestone"
        ),
        "final_overflow": _as_float(placement_debug.get("final_overflow")),
        "timing_gate": timing_gate,
        "timing_grad_balance": timing_grad_balance,
        "legacy_net_weight_gate": legacy_net_weight_gate,
        "legacy_net_weight": legacy_net_weight,
        "placement_debug_artifact_path": str(placement_debug_path),
    }


def _segment_joint_physical_evidence(method_root: Path, case: str) -> dict[str, Any]:
    path = (
        method_root
        / "autodmp_result"
        / f"{case}_segment_joint_committed_verification.json"
    )
    if not path.is_file():
        return {"status": "missing", "artifact_path": str(path)}
    payload = _read_json(path)
    return {
        "status": payload.get("status"),
        "artifact_path": str(path),
        "physical_action_trace": list(payload.get("physical_action_trace", ()) or ()),
        "full_action_order": list(payload.get("full_action_order", ()) or ()),
        "legalization_status": payload.get("legalization_status"),
        "parasitic_refresh_status": payload.get("parasitic_refresh_status"),
        "def_only_opensta_verification_status": payload.get(
            "def_only_opensta_verification_status"
        ),
        "def_only_opensta_after": dict(
            payload.get("def_only_opensta_after", {}) or {}
        ),
        "def_only_opensta_replay": dict(
            payload.get("def_only_opensta_replay", {}) or {}
        ),
        "def_only_vs_in_process_gap": dict(
            payload.get("def_only_vs_in_process_gap", {}) or {}
        ),
        "opensta_before": dict(payload.get("opensta_before", {}) or {}),
        "opensta_after": dict(payload.get("opensta_after", {}) or {}),
        "delta_wns": _as_float(payload.get("delta_wns")),
        "delta_tns": _as_float(payload.get("delta_tns")),
        "qor_status": payload.get("qor_status"),
        "qor_pass": payload.get("qor_status") == "pass",
        "sizing_substitution_count": _as_int(payload.get("sizing_action_count")),
        "buffer_count": _as_int(payload.get("buffer_count")),
        "accepted_buffer_count": _as_int(payload.get("accepted_action_count")),
        "sizing_master_verification": dict(
            payload.get("sizing_master_verification", {}) or {}
        ),
        "sizing_added_area_um2": _as_float(payload.get("sizing_added_area_um2")),
        "buffer_added_area_um2": _as_float(payload.get("buffer_added_area_um2")),
        "total_added_area_um2": _as_float(payload.get("added_area_um2")),
        "final_overflow": _as_float(
            dict(payload.get("final_placement_metrics", {}) or {}).get("overflow")
        ),
        "runtime_ms": _as_float(payload.get("runtime_ms")),
        "refresh_summary": dict(payload.get("refresh_summary", {}) or {}),
        "final_def_path": payload.get("final_def_path"),
        "final_def_sha256": payload.get("final_def_sha256"),
    }


def _add_post_action_step_evidence(
    milestones: dict[str, Any],
    total_optimizer_steps: int | None,
) -> None:
    records = list(milestones.get("per_milestone", ()) or ())
    if total_optimizer_steps is None:
        milestones["post_action_placement_steps"] = []
        milestones["post_final_action_placement_steps"] = None
        return
    post_steps = [
        {
            "milestone": float(row["milestone"]),
            "iteration": int(row["iteration"]),
            "steps": max(0, int(total_optimizer_steps) - int(row["iteration"]) - 1),
        }
        for row in records
    ]
    milestones["post_action_placement_steps"] = post_steps
    milestones["post_final_action_placement_steps"] = (
        None if not post_steps else int(post_steps[-1]["steps"])
    )


def _overflow_milestone_method_failures(
    method: str,
    method_result: dict[str, Any],
) -> list[str]:
    failures = []
    milestones = dict(method_result.get("milestones", {}) or {})
    if milestones.get("final_overflow") is None:
        failures.append("final_overflow_missing")

    sizing_count = int(milestones.get("sizing_action_count", 0) or 0)
    buffer_count = int(milestones.get("buffer_action_count", 0) or 0)
    arm = OVERFLOW_MILESTONE_ARMS[method]
    expects_direct_timing = bool(arm["placement_timing_enabled"])
    expects_pin2pin = bool(arm["pin2pin_enabled"])
    expects_sizing = bool(arm["sizing_enabled"])
    expects_buffering = bool(arm["route_b_enabled"])
    if not expects_sizing and sizing_count > 0:
        failures.append("sizing_action_ownership")
    if not expects_buffering and buffer_count > 0:
        failures.append("buffer_action_ownership")

    timing_gate = dict(milestones.get("timing_gate", {}) or {})
    timing_active_steps = int(timing_gate.get("timing_active_step_count", 0) or 0)
    topology_refreshes = int(timing_gate.get("topology_refresh_count", 0) or 0)
    if (timing_active_steps > 0) != expects_direct_timing:
        failures.append("timing_action_ownership")
    if expects_direct_timing:
        first_overflow = _as_float(timing_gate.get("first_timing_active_overflow"))
        threshold = _as_float(timing_gate.get("threshold"))
        if (
            first_overflow is None
            or threshold is None
            or not first_overflow < threshold
        ):
            failures.append("timing_overflow_gate")
        if topology_refreshes <= 0:
            failures.append("timing_topology_refresh")
        timing_grad_balance = dict(
            milestones.get("timing_grad_balance", {}) or {}
        )
        target_ratio = _as_float(timing_grad_balance.get("target_ratio"))
        if target_ratio is None:
            failures.append("timing_grad_balance_missing")
        elif target_ratio > 0.0:
            if (
                timing_grad_balance.get("status") != "initialized"
                or not timing_grad_balance.get("initialized")
            ):
                failures.append("timing_grad_balance_not_initialized")
            first_iteration = _as_int(
                timing_gate.get("first_timing_active_iteration")
            )
            balance_iteration = _as_int(timing_grad_balance.get("iteration"))
            if balance_iteration != first_iteration:
                failures.append("timing_grad_balance_activation_frame")

    legacy_gate = dict(milestones.get("legacy_net_weight_gate", {}) or {})
    legacy_update = dict(milestones.get("legacy_net_weight", {}) or {})
    legacy_update_count = int(legacy_gate.get("update_count", 0) or 0)
    pair_generations = dict(milestones.get("pair_generations", {}) or {})
    managed_pin2pin = bool(pair_generations.get("generation_history"))
    if managed_pin2pin and expects_pin2pin:
        history = [
            dict(record)
            for record in pair_generations.get("generation_history", ())
        ]
        install_counts = {
            str(iteration): int(count)
            for iteration, count in dict(
                pair_generations.get("install_count_by_iteration", {}) or {}
            ).items()
        }
        if any(count != 1 for count in install_counts.values()):
            failures.append("pin2pin_generation_install_multiplicity")
        for generation in history:
            status = generation.get("status")
            pair_count = int(generation.get("pair_count", 0) or 0)
            if status == "timing_clean" and pair_count != 0:
                failures.append("pin2pin_clean_generation_nonempty")
            elif status == "timing_violating" and pair_count <= 0:
                failures.append("pin2pin_violating_generation_empty")
            elif status not in {"timing_clean", "timing_violating"}:
                failures.append("pin2pin_generation_status")
            if int(generation.get("objective_eval_count", 0) or 0) < int(
                generation.get("consumed_step_count", 0) or 0
            ):
                failures.append("pin2pin_generation_consumption_order")
        if history[0].get("install_reason") != "activation_bootstrap":
            failures.append("pin2pin_activation_generation")
    else:
        if (legacy_update_count > 0) != expects_pin2pin:
            failures.append("pin2pin_action_ownership")
    if expects_pin2pin and not managed_pin2pin:
        first_overflow = _as_float(legacy_gate.get("first_update_overflow"))
        threshold = _as_float(legacy_gate.get("threshold"))
        if (
            first_overflow is None
            or threshold is None
            or not first_overflow < threshold
        ):
            failures.append("pin2pin_overflow_gate")
        if legacy_gate.get("scheme") != "pin2pin":
            failures.append("pin2pin_scheme")
        if int(legacy_update.get("pin2pin_pair_count", 0) or 0) <= 0:
            failures.append("pin2pin_empty_pairs")
        if legacy_update.get("pin2pin_backend") != "cuda":
            failures.append("pin2pin_cuda_backend")
        if (
            legacy_update.get("pin2pin_path_pair_backend")
            != "cpp_openmp_transition_aware"
        ):
            failures.append("pin2pin_path_pair_backend")
        native_summary = dict(
            legacy_update.get("pin2pin_native_summary", {}) or {}
        )
        if native_summary.get("state_domain") != "setup_plus_recovery":
            failures.append("pin2pin_state_domain")
        if native_summary.get("pair_normalization_policy") != (
            "selected_max_endpoint_gba_wns"
        ):
            failures.append("pin2pin_normalization_policy")
        if _as_int(native_summary.get("recovery_endpoint_count")) is None:
            failures.append("pin2pin_recovery_endpoint_count_missing")
        objective = _as_float(legacy_update.get("pin2pin_objective"))
        if objective is None or objective <= 0.0:
            failures.append("pin2pin_objective")

    if not expects_sizing and not expects_buffering:
        return failures
    if milestones.get("schedule_mode") != "overflow_milestones":
        failures.append("schedule_mode")
    if milestones.get("terminal_failure") is not None:
        failures.append("terminal_failure")
    if milestones.get("incomplete_pending_milestone") is not None:
        failures.append("incomplete_pending_milestone")
    if milestones.get("pending_queue"):
        failures.append("milestone_queue_not_empty")
    if milestones.get("pending_event") is not None:
        failures.append("milestone_transaction_pending")
    if milestones.get("reached_milestones") != [0.30, 0.25, 0.20, 0.15, 0.10]:
        failures.append("milestone_coverage")
    physical = dict(method_result.get("physical_commit", {}) or {})
    if physical.get("status") != "ok":
        failures.append("physical_commit_status")
        return failures
    expected_order = ["coordinate_sync"]
    if expects_sizing:
        expected_order.append("apply_sizing")
    terminal_reselection = dict(
        milestones.get("terminal_buffer_reselection", {}) or {}
    )
    final_buffer_count = 0
    if expects_buffering and managed_pin2pin:
        final_buffer_count = buffer_count
        expected_schedule = dict(
            method_result.get("expected_milestone_schedule", {}) or {}
        )
        expected_rounds = _as_int(expected_schedule.get("sizing_rounds"))
        expected_stages = [
            "tdp_objective_and_step_consumed",
            "sizing_micro_loop_completed",
            "buffering_gradient_ready",
            "route_b_applied",
            "tdp_state_invalidated",
            "tdp_rebootstrap_completed",
        ]
        for record in milestones.get("per_milestone", ()) or ():
            if list(record.get("transaction_stages", ()) or ()) != expected_stages:
                failures.append("milestone_transaction_stage_order")
            sizing_micro_loop = dict(record.get("sizing_micro_loop", {}) or {})
            if not sizing_micro_loop:
                failures.append("milestone_sizing_micro_loop_missing")
                continue
            if expected_rounds is not None and int(
                sizing_micro_loop.get("rounds_requested", 0) or 0
            ) != expected_rounds:
                failures.append("milestone_sizing_round_budget")
            rounds = list(sizing_micro_loop.get("rounds", ()) or ())
            if int(sizing_micro_loop.get("rounds_attempted", -1)) != len(rounds):
                failures.append("milestone_sizing_round_count")
            digests = [row.get("gradient_digest") for row in rounds]
            if any(not digest for digest in digests) or len(set(digests)) != len(digests):
                failures.append("milestone_sizing_gradient_identity")
            up_percent = _as_float(expected_schedule.get("sizing_up_percent"))
            if up_percent is not None:
                for sizing_round in rounds:
                    denominator = _as_int(
                        sizing_round.get("num_up_improving_instances")
                    )
                    numerator = _as_int(sizing_round.get("num_up_selected"))
                    if denominator is None or numerator is None:
                        failures.append("milestone_sizing_selection_denominator")
                        continue
                    expected_numerator = (
                        0
                        if denominator <= 0 or up_percent <= 0.0
                        else min(
                            denominator,
                            max(1, int(math.ceil(denominator * up_percent / 100.0))),
                        )
                    )
                    if numerator != expected_numerator:
                        failures.append("milestone_sizing_selection_denominator")
            before_generation = _as_int(record.get("pair_generation_before"))
            after_generation = _as_int(record.get("pair_generation_after"))
            if (
                before_generation is None
                or after_generation is None
                or after_generation <= before_generation
            ):
                failures.append("milestone_pair_generation_transition")
    elif expects_buffering:
        final_buffer_count_value = _as_int(
            terminal_reselection.get("final_buffer_count")
        )
        valid_terminal_reselection = (
            terminal_reselection.get("status") == "reselected"
            and terminal_reselection.get("policy")
            == "rebuild_tree_reset_then_terminal_frame_route_b"
            and terminal_reselection.get("frame")
            == "terminal_final_placement_sizing"
            and _as_int(terminal_reselection.get("historical_buffer_count"))
            == buffer_count
            and _as_int(terminal_reselection.get("round_budget"))
            == len(milestones.get("reached_milestones", ()) or ())
            and (_as_int(terminal_reselection.get("iterations")) or 0) > 0
            and (_as_int(terminal_reselection.get("iterations")) or 0)
            <= (_as_int(terminal_reselection.get("round_budget")) or 0)
            and terminal_reselection.get("terminal_reason")
            in {"max_rounds", "no_positive_action", "no_improving_prefix"}
            and final_buffer_count_value is not None
            and final_buffer_count_value >= 0
        )
        if not valid_terminal_reselection:
            failures.append("terminal_buffer_reselection")
        else:
            final_buffer_count = int(final_buffer_count_value)
    if expects_buffering and final_buffer_count > 0:
        expected_order.append("insert_buffers")
    actual_order = [
        str(row.get("stage"))
        for row in physical.get("physical_action_trace", ())
    ]
    if actual_order != expected_order:
        failures.append("physical_action_order")
    if physical.get("legalization_status") != "ok":
        failures.append("legalization")
    if physical.get("parasitic_refresh_status") != "ok":
        failures.append("parasitic_refresh")
    if physical.get("def_only_opensta_verification_status") != "ok":
        failures.append("def_only_replay")
    if _def_only_gap_exceeds_tolerance(physical):
        failures.append("def_only_gap")
    if expects_sizing:
        sizing_verify = dict(physical.get("sizing_master_verification", {}) or {})
        physical_sizing_count = int(
            physical.get("sizing_substitution_count", 0) or 0
        )
        if sizing_count > 0:
            if sizing_verify.get("status") != "ok":
                failures.append("sizing_master_verification")
            if physical_sizing_count <= 0:
                failures.append("physical_sizing_count")
        elif physical_sizing_count != 0:
            failures.append("unexpected_physical_sizing")
    elif int(physical.get("sizing_substitution_count", 0) or 0) != 0:
        failures.append("unexpected_physical_sizing")
    if expects_buffering:
        if _buffer_action_count_mismatch(physical, final_buffer_count):
            failures.append("buffer_action_count_parity")
    return failures


def _lifecycle_windows(method_root: Path, case: str) -> list[dict[str, Any]]:
    path = method_root / "autodmp_result" / f"{case}_joint_outer_lifecycle_summary.json"
    payload = _read_json(path)
    return [dict(row) for row in payload.get("windows", ())]


def _lifecycle_steps(method_root: Path, case: str) -> list[int]:
    return [
        int(row["actual_placement_steps"])
        for row in _lifecycle_windows(method_root, case)
        if row.get("actual_placement_steps") is not None
    ]


def _m3_rounds(method_root: Path) -> int | None:
    traces = sorted(method_root.rglob("segment_discrete_scheduler_trace.json"))
    if not traces:
        return None
    return _as_int(_read_json(traces[-1]).get("iterations"))


def _optimizer_steps(method_root: Path, case: str) -> int | None:
    path = method_root / "autodmp_result" / f"{case}_placement_debug_summary.json"
    return _as_int(_read_json(path).get("optimizer_steps"))


def _committed_buffer_count(method_result: dict[str, Any]) -> int | None:
    return _as_int(
        dict(method_result.get("track_a", {}) or {})
        .get("mutation", {})
        .get("added_count")
    )


def _accepted_window_action_count(method_result: dict[str, Any]) -> int:
    return sum(
        int(window.get("accepted_action_count", 0) or 0)
        for window in method_result.get("virtual_windows", ())
    )


def _evaluate_method(
    *,
    openroad_bin: Path,
    benchmark_root: Path,
    case: str,
    method: str,
    method_root: Path,
    input_def: Path,
    r0_def: Path,
    track_b: bool,
    openroad_num_threads: int,
) -> dict[str, Any]:
    track_a_root = method_root / "track_a"
    track_a_root.mkdir(parents=True, exist_ok=True)
    track_a_def = track_a_root / f"{case}_{method}_track_a.def"
    track_a_tcl = track_a_root / "evaluate.tcl"
    endpoint_csv = track_a_root / "endpoint_slacks.csv"
    power_report = track_a_root / "power.rpt"
    track_a_tcl.write_text(
        generate_evaluation_tcl(
            benchmark_root=benchmark_root,
            case=case,
            input_def=input_def,
            output_def=track_a_def,
            endpoint_csv=endpoint_csv,
            power_report=power_report,
            closure=False,
        ),
        encoding="utf-8",
    )
    execution = _run_openroad_tcl(
        openroad_bin=openroad_bin,
        tcl_path=track_a_tcl,
        log_path=track_a_root / "evaluate.log",
        openroad_num_threads=openroad_num_threads,
    )
    metrics = _parse_metrics(track_a_root / "evaluate.log")
    track_a = {
        "status": "pass" if execution["status"] == "ok" and track_a_def.is_file() else "failed",
        "execution": execution,
        "input_def": str(input_def),
        "input_def_sha256": sha256_file(input_def) if input_def.is_file() else None,
        "final_def": str(track_a_def),
        "final_def_sha256": sha256_file(track_a_def) if track_a_def.is_file() else None,
        "metrics": metrics,
        "power_total_w": _parse_power_total(power_report),
        "endpoint_csv": str(endpoint_csv),
        "mutation": _def_added_instances(r0_def, track_a_def) if track_a_def.is_file() else {},
    }
    result = {"track_a": track_a}
    if not track_b or track_a["status"] != "pass":
        return result

    track_b_root = method_root / "track_b"
    track_b_root.mkdir(parents=True, exist_ok=True)
    track_b_def = track_b_root / f"{case}_{method}_track_b.def"
    track_b_tcl = track_b_root / "evaluate.tcl"
    track_b_endpoint_csv = track_b_root / "endpoint_slacks.csv"
    track_b_power = track_b_root / "power.rpt"
    track_b_tcl.write_text(
        generate_evaluation_tcl(
            benchmark_root=benchmark_root,
            case=case,
            input_def=track_a_def,
            output_def=track_b_def,
            endpoint_csv=track_b_endpoint_csv,
            power_report=track_b_power,
            closure=True,
        ),
        encoding="utf-8",
    )
    track_b_execution = _run_openroad_tcl(
        openroad_bin=openroad_bin,
        tcl_path=track_b_tcl,
        log_path=track_b_root / "evaluate.log",
        openroad_num_threads=openroad_num_threads,
    )
    track_b_metrics = _parse_metrics(track_b_root / "evaluate.log")
    result["track_b"] = {
        "status": "pass" if track_b_execution["status"] == "ok" and track_b_def.is_file() else "failed",
        "execution": track_b_execution,
        "input_def": str(track_a_def),
        "input_def_sha256": sha256_file(track_a_def),
        "final_def": str(track_b_def),
        "final_def_sha256": sha256_file(track_b_def) if track_b_def.is_file() else None,
        "metrics": track_b_metrics,
        "power_total_w": _parse_power_total(track_b_power),
        "endpoint_csv": str(track_b_endpoint_csv),
        "repair_mutation": _def_added_instances(track_a_def, track_b_def) if track_b_def.is_file() else {},
    }
    return result


def _run_case(args: argparse.Namespace, case: str, run_root: Path) -> dict[str, Any]:
    case_root = run_root / case
    case_root.mkdir(parents=True, exist_ok=True)
    prior_result = _read_json(case_root / "case_summary.json") if args.resume else {}
    inputs = _case_inputs(args.benchmark_root, case)
    params = _read_json(inputs["params_json"])
    seed = int(args.seed if args.seed is not None else params.get("random_seed", 3000))
    budget = _placement_budget(
        inputs["params_json"],
        optimizer_override=args.placement_optimizer,
    )
    iterations = int(args.iterations if args.iterations is not None else budget["P"])
    warmup_iterations = int(args.warmup_iterations)
    target_density = float(params.get("target_density", 0.8))
    if prior_result and (
        _as_int(prior_result.get("P")) != iterations
        or _as_int(prior_result.get("K")) != int(args.outer_iterations)
        or _as_int(prior_result.get("warmup_iterations") or 0)
        != warmup_iterations
    ):
        raise ValueError(
            f"{case}: resume P/K mismatch; existing="
            f"{prior_result.get('P')}/{prior_result.get('K')} "
            f"warmup={prior_result.get('warmup_iterations', 0)} "
            f"requested={iterations}/{args.outer_iterations} "
            f"warmup={warmup_iterations}"
        )
    methods = list(args.methods)
    result: dict[str, Any] = {
        "artifact": "random_init_placement_buffering_case_result",
        "artifact_version": 1,
        "case": case,
        "seed": seed,
        "K": int(args.outer_iterations),
        "P": iterations,
        "warmup_iterations": warmup_iterations,
        "placement_budget": budget,
        "target_density": target_density,
        "openroad_num_threads": int(args.openroad_num_threads),
        "methods": {},
        "failures": [],
    }

    external_r0_def = _external_r0_path(args, case)
    if external_r0_def is None:
        r0_command = build_r0_command(
            python_bin=args.python_bin,
            benchmark_root=args.benchmark_root,
            case=case,
            case_root=case_root,
            seed=seed,
        )
    else:
        r0_command = ["external_prepared_r0", str(external_r0_def)]
    (case_root / "r0" / "command.txt").parent.mkdir(parents=True, exist_ok=True)
    (case_root / "r0" / "command.txt").write_text(shlex.join(r0_command) + "\n", encoding="utf-8")
    if args.plan_only:
        result["status"] = "plan_only"
        return result
    r0_def = case_root / "r0" / "R0.def"
    r0_manifest_path = case_root / "r0" / "R0_manifest.json"
    if external_r0_def is not None:
        if not external_r0_def.is_file():
            result["status"] = "failed"
            result["failures"].append("external_r0_missing")
            return result
        r0_def.parent.mkdir(parents=True, exist_ok=True)
        if r0_def.is_file() and sha256_file(r0_def) != sha256_file(external_r0_def):
            raise ValueError(f"{case}: existing external R0 copy differs from source")
        if not r0_def.is_file():
            shutil.copyfile(external_r0_def, r0_def)
        r0_manifest = _external_r0_manifest(r0_def, case, seed)
        _write_json(r0_manifest_path, r0_manifest)
        r0_execution = {
            "status": "ok",
            "returncode": 0,
            "runtime_sec": 0.0,
            "log_path": str(case_root / "r0" / "run.log"),
            "external_prepared": True,
        }
    elif (
        args.resume
        and r0_def.is_file()
        and _read_json(r0_manifest_path).get("status") == "pass"
    ):
        r0_execution = {
            "status": "ok",
            "returncode": 0,
            "runtime_sec": 0.0,
            "log_path": str(case_root / "r0" / "run.log"),
            "resumed": True,
        }
    else:
        r0_execution = _run_command(
            r0_command,
            cwd=AUTODMP_ROOT,
            log_path=case_root / "r0" / "run.log",
        )
    r0_manifest = _read_json(r0_manifest_path)
    result["r0"] = {"execution": r0_execution, "manifest": r0_manifest}
    if r0_execution["status"] != "ok" or r0_manifest.get("status") != "pass":
        result["status"] = "failed"
        result["failures"].append("r0_gate_failed")
        return result
    if args.r0_only:
        result["status"] = "pass"
        return result

    environment = {
        "CUDA_VISIBLE_DEVICES": str(
            args.cuda_visible_devices
            if args.cuda_visible_devices is not None
            else args.gpu_id
        )
    }
    for method in methods:
        method_root = case_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        prior_method = dict(
            dict(prior_result.get("methods", {}) or {}).get(method, {}) or {}
        )
        prior_raw_def = Path(str(prior_method.get("raw_def") or ""))
        prior_track_b_ok = (
            not args.track_b
            or dict(prior_method.get("track_b", {}) or {}).get("status") == "pass"
        )
        if (
            args.resume
            and prior_method.get("status") == "pass"
            and prior_raw_def.is_file()
            and prior_track_b_ok
            and not args.reevaluate
        ):
            if method in OPENROAD_PLACEMENT_METHODS:
                prior_method.update(
                    _m1_coordinate_identity(
                        method_root / "pre_gpl_coordinates.tsv",
                        r0_def,
                        r0_manifest,
                    )
                )
            prior_method["resumed"] = True
            result["methods"][method] = prior_method
            continue
        if (
            args.resume
            and args.reevaluate
            and prior_method.get("execution", {}).get("status") == "ok"
        ):
            raw_def = _method_raw_def(case_root, case, method)
            if not raw_def.is_file():
                result["failures"].append(f"{method}_missing_def_for_reevaluation")
                continue
            method_result = prior_method
            method_result["raw_def"] = str(raw_def)
            method_result["raw_def_sha256"] = sha256_file(raw_def)
            if method in OPENROAD_PLACEMENT_METHODS:
                method_result.update(
                    _m1_coordinate_identity(
                        method_root / "pre_gpl_coordinates.tsv",
                        r0_def,
                        r0_manifest,
                    )
                )
                method_result["native_timing_driven_actions"] = (
                    _parse_m1_native_actions(method_root / "run.log")
                )
            if method in OVERFLOW_MILESTONE_METHODS:
                total_steps = _optimizer_steps(method_root, case)
                milestones = _milestone_evidence(method_root, case)
                _add_post_action_step_evidence(milestones, total_steps)
                method_result["total_optimizer_steps"] = total_steps
                method_result["milestones"] = milestones
                arm = OVERFLOW_MILESTONE_ARMS[method]
                if arm["sizing_enabled"] or arm["route_b_enabled"]:
                    method_result["physical_commit"] = (
                        _segment_joint_physical_evidence(method_root, case)
                    )
            method_result.update(
                _evaluate_method(
                    openroad_bin=args.openroad_bin,
                    benchmark_root=args.benchmark_root,
                    case=case,
                    method=method,
                    method_root=method_root,
                    input_def=raw_def,
                    r0_def=r0_def,
                    track_b=args.track_b,
                    openroad_num_threads=args.openroad_num_threads,
                )
            )
            method_result["status"] = (
                "pass" if method_result["track_a"]["status"] == "pass" else "failed"
            )
            method_result["end_to_end_runtime_sec"] = (
                _as_float(method_result.get("execution", {}).get("runtime_sec")) or 0.0
            ) + (
                _as_float(
                    method_result.get("track_a", {})
                    .get("execution", {})
                    .get("runtime_sec")
                )
                or 0.0
            )
            result["methods"][method] = method_result
            continue
        existing_raw_def = _method_raw_def(case_root, case, method)
        if (
            args.resume
            and args.reevaluate
            and not prior_method
            and existing_raw_def.is_file()
            and (method_root / "run.log").is_file()
        ):
            method_result = {
                "execution": {
                    "status": "ok",
                    "returncode": 0,
                    "runtime_sec": None,
                    "log_path": str(method_root / "run.log"),
                    "resumed_from_existing_artifact": True,
                },
                "raw_def": str(existing_raw_def),
                "raw_def_sha256": sha256_file(existing_raw_def),
            }
            if method in {"m2", "m4", "m4_no_density"}:
                method_result["actual_placement_steps"] = _lifecycle_steps(
                    method_root,
                    case,
                )
                method_result["total_optimizer_steps"] = _optimizer_steps(
                    method_root,
                    case,
                )
            if method in OVERFLOW_MILESTONE_METHODS:
                method_result["total_optimizer_steps"] = _optimizer_steps(
                    method_root,
                    case,
                )
                method_result["milestones"] = _milestone_evidence(
                    method_root,
                    case,
                )
                method_result["virtual_windows"] = _lifecycle_windows(
                    method_root,
                    case,
                )
            if method == "m3":
                method_result["route_b_rounds"] = _m3_rounds(method_root)
            method_result.update(
                _evaluate_method(
                    openroad_bin=args.openroad_bin,
                    benchmark_root=args.benchmark_root,
                    case=case,
                    method=method,
                    method_root=method_root,
                    input_def=existing_raw_def,
                    r0_def=r0_def,
                    track_b=args.track_b,
                    openroad_num_threads=args.openroad_num_threads,
                )
            )
            method_result["status"] = (
                "pass" if method_result["track_a"]["status"] == "pass" else "failed"
            )
            result["methods"][method] = method_result
            continue
        if method in OPENROAD_PLACEMENT_METHODS:
            raw_def = _method_raw_def(case_root, case, method)
            coordinate_path = method_root / "pre_gpl_coordinates.tsv"
            tcl_path = method_root / "run.tcl"
            if method == OPENROAD_NATIVE_FULL_FLOW_METHOD:
                tcl = generate_openroad_native_full_flow_tcl(
                    benchmark_root=args.benchmark_root,
                    case=case,
                    r0_def=r0_def,
                    output_def=raw_def,
                    pre_gpl_coordinates=coordinate_path,
                    target_density=target_density,
                )
            else:
                tcl = generate_m1_tcl(
                    benchmark_root=args.benchmark_root,
                    case=case,
                    r0_def=r0_def,
                    output_def=raw_def,
                    pre_gpl_coordinates=coordinate_path,
                    target_density=target_density,
                    reweight_only=method == OPENROAD_REWEIGHT_ONLY_METHOD,
                    timing_driven=method != OPENROAD_GPL_PLAIN_METHOD,
                )
            tcl_path.write_text(tcl, encoding="utf-8")
            execution = _run_openroad_tcl(
                openroad_bin=args.openroad_bin,
                tcl_path=tcl_path,
                log_path=method_root / "run.log",
                openroad_num_threads=args.openroad_num_threads,
            )
            method_result = {
                "execution": execution,
                "raw_def": str(raw_def),
                "metrics": _parse_metrics(method_root / "run.log"),
                "native_timing_driven_actions": _parse_m1_native_actions(
                    method_root / "run.log"
                ),
            }
            method_result.update(
                _m1_coordinate_identity(
                    coordinate_path,
                    r0_def,
                    r0_manifest,
                )
            )
        elif method == "m2":
            command = build_m2_m4_command(
                python_bin=args.python_bin,
                benchmark_root=args.benchmark_root,
                case=case,
                r0_def=r0_def,
                method_root=method_root,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                outer_iterations=args.outer_iterations,
                gpu_id=0,
                placement_optimizer=args.placement_optimizer,
                route_b_enabled=False,
                virtual_density_enabled=False,
            )
            (method_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            execution = _run_command(
                command,
                cwd=AUTODMP_ROOT,
                log_path=method_root / "run.log",
                environment=environment,
            )
            method_result = {
                "execution": execution,
                "raw_def": str(_method_raw_def(case_root, case, method)),
                "actual_placement_steps": _lifecycle_steps(method_root, case),
                "total_optimizer_steps": _optimizer_steps(method_root, case),
                "virtual_windows": _lifecycle_windows(method_root, case),
            }
        elif method == "m3":
            m2_def = _method_raw_def(case_root, case, "m2")
            if not m2_def.is_file():
                result["failures"].append("m3_missing_m2_checkpoint")
                continue
            matched_rounds = len(
                _lifecycle_windows(case_root / "m2", case)
            )
            if matched_rounds <= 0:
                result["failures"].append("m3_no_reached_m2_window")
                continue
            command = build_m3_command(
                python_bin=args.python_bin,
                benchmark_root=args.benchmark_root,
                case=case,
                m2_def=m2_def,
                method_root=method_root,
                rounds=matched_rounds,
                gpu_id=0,
            )
            (method_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            execution = _run_command(
                command,
                cwd=AUTODMP_ROOT,
                log_path=method_root / "run.log",
                environment=environment,
            )
            method_result = {
                "execution": execution,
                "raw_def": str(_method_raw_def(case_root, case, method)),
                "route_b_rounds": _m3_rounds(method_root),
                "input_m2_def_sha256": sha256_file(m2_def),
            }
        elif method in {"m4", "m4_no_density"}:
            command = build_m2_m4_command(
                python_bin=args.python_bin,
                benchmark_root=args.benchmark_root,
                case=case,
                r0_def=r0_def,
                method_root=method_root,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                outer_iterations=args.outer_iterations,
                gpu_id=0,
                placement_optimizer=args.placement_optimizer,
                route_b_enabled=True,
                virtual_density_enabled=method == "m4",
            )
            (method_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            execution = _run_command(
                command,
                cwd=AUTODMP_ROOT,
                log_path=method_root / "run.log",
                environment=environment,
            )
            method_result = {
                "execution": execution,
                "raw_def": str(_method_raw_def(case_root, case, method)),
                "actual_placement_steps": _lifecycle_steps(method_root, case),
                "total_optimizer_steps": _optimizer_steps(method_root, case),
                "virtual_windows": _lifecycle_windows(method_root, case),
            }
        elif method in OVERFLOW_MILESTONE_METHODS:
            arm = OVERFLOW_MILESTONE_ARMS[method]
            command = build_overflow_milestone_command(
                python_bin=args.python_bin,
                benchmark_root=args.benchmark_root,
                case=case,
                r0_def=r0_def,
                method_root=method_root,
                iterations=iterations,
                gpu_id=0,
                placement_optimizer=args.placement_optimizer,
                placement_timing_enabled=arm["placement_timing_enabled"],
                sizing_enabled=arm["sizing_enabled"],
                route_b_enabled=arm["route_b_enabled"],
                enable_fillers=args.enable_fillers,
                legalize=args.legalize,
                stop_overflow=args.stop_overflow,
                pin2pin_enabled=arm["pin2pin_enabled"],
                pin2pin_weight=args.pin2pin_weight,
                pin2pin_min_weight=args.pin2pin_min_weight,
                pin2pin_max_weight=args.pin2pin_max_weight,
                pin2pin_accumulate_weight=args.pin2pin_accumulate_weight,
                net_weighting_npaths=args.net_weighting_npaths,
                pin2pin_update_interval=args.pin2pin_update_interval,
                joint_segment_milestones=args.joint_segment_milestones,
                joint_segment_sizing_rounds=args.joint_segment_sizing_rounds,
                joint_segment_sizing_up_percent=(
                    args.joint_segment_sizing_up_percent
                ),
                joint_segment_buffering_rounds=(
                    args.joint_segment_buffering_rounds
                ),
                joint_segment_buffering_selection_fraction=(
                    args.joint_segment_buffering_selection_fraction
                ),
                joint_segment_pin2pin_rebootstrap_enabled=bool(
                    args.joint_segment_pin2pin_rebootstrap_enabled
                ),
                num_bins_x=args.num_bins_x,
                num_bins_y=args.num_bins_y,
                timing_grad_balance_target_ratio=(
                    args.timing_grad_balance_target_ratio
                ),
                timing_topology_enable_overflow_threshold=(
                    args.timing_topology_enable_overflow_threshold
                ),
                plot_flag=args.plot,
            )
            (method_root / "command.txt").write_text(
                shlex.join(command) + "\n",
                encoding="utf-8",
            )
            execution = _run_command(
                command,
                cwd=AUTODMP_ROOT,
                log_path=method_root / "run.log",
                environment=environment,
            )
            total_steps = _optimizer_steps(method_root, case)
            milestones = _milestone_evidence(method_root, case)
            _add_post_action_step_evidence(milestones, total_steps)
            method_result = {
                "execution": execution,
                "raw_def": str(_method_raw_def(case_root, case, method)),
                "total_optimizer_steps": total_steps,
                "milestones": milestones,
                "expected_milestone_schedule": {
                    "thresholds": list(args.joint_segment_milestones),
                    "sizing_rounds": int(args.joint_segment_sizing_rounds),
                    "sizing_up_percent": float(
                        args.joint_segment_sizing_up_percent
                    ),
                    "buffering_rounds": int(
                        args.joint_segment_buffering_rounds
                    ),
                    "buffering_selection_fraction": float(
                        args.joint_segment_buffering_selection_fraction
                    ),
                    "pin2pin_rebootstrap_enabled": bool(
                        args.joint_segment_pin2pin_rebootstrap_enabled
                    ),
                },
            }
            if arm["sizing_enabled"] or arm["route_b_enabled"]:
                method_result["physical_commit"] = (
                    _segment_joint_physical_evidence(method_root, case)
                )
        else:
            raise AssertionError(method)
        raw_def = Path(method_result["raw_def"])
        if execution["status"] != "ok" or not raw_def.is_file():
            method_result["status"] = "failed"
            result["failures"].append(f"{method}_execution_failed")
        else:
            method_result["status"] = "pass"
            method_result["raw_def_sha256"] = sha256_file(raw_def)
            if args.source_only:
                method_result["evaluation"] = {
                    "status": "skipped",
                    "reason": "source_only",
                }
            else:
                method_result.update(
                    _evaluate_method(
                        openroad_bin=args.openroad_bin,
                        benchmark_root=args.benchmark_root,
                        case=case,
                        method=method,
                        method_root=method_root,
                        input_def=raw_def,
                    r0_def=r0_def,
                    track_b=args.track_b,
                    openroad_num_threads=args.openroad_num_threads,
                    )
                )
            if not args.source_only and method_result["track_a"]["status"] != "pass":
                method_result["status"] = "failed"
                result["failures"].append(f"{method}_track_a_failed")
            method_result["end_to_end_runtime_sec"] = (
                _as_float(method_result.get("execution", {}).get("runtime_sec")) or 0.0
            ) + (
                _as_float(
                    method_result.get("track_a", {})
                    .get("execution", {})
                    .get("runtime_sec")
                )
                or 0.0
            )
        result["methods"][method] = method_result

    for method in OVERFLOW_MILESTONE_METHODS:
        if method not in result["methods"]:
            continue
        method_result = result["methods"][method]
        mechanism_failures = _overflow_milestone_method_failures(
            method,
            method_result,
        )
        method_result["mechanism_audit"] = {
            "status": "pass" if not mechanism_failures else "failed",
            "failures": mechanism_failures,
        }
        if mechanism_failures:
            method_result["status"] = "failed"
            result["failures"].append(f"{method}_mechanism_failed")

    maximum_total_steps = warmup_iterations + iterations * int(args.outer_iterations)
    for method in ("m2", "m4"):
        if method in result["methods"]:
            total_steps = _as_int(
                result["methods"][method].get("total_optimizer_steps")
            )
            if (
                total_steps is None
                or total_steps <= 0
                or total_steps > maximum_total_steps
            ):
                result["failures"].append(f"{method}_total_optimizer_steps_mismatch")
    m3_rounds = None
    if "m3" in result["methods"]:
        m3_rounds = _as_int(result["methods"]["m3"].get("route_b_rounds"))
        if m3_rounds is None or m3_rounds <= 0:
            result["failures"].append("m3_route_b_round_mismatch")
    if "m4" in result["methods"]:
        m4_windows = list(result["methods"]["m4"].get("virtual_windows", ()))
        active_rounds = sum(
            str(dict(window.get("route_b", {}) or {}).get("status"))
            in {"window_boundary", "disabled"}
            for window in m4_windows
        )
        m4_total_steps = _as_int(
            result["methods"]["m4"].get("total_optimizer_steps")
        )
        actions_reaching_placement = 0
        for window in m4_windows:
            action_iteration = _as_int(
                dict(window.get("route_b", {}) or {}).get("iteration")
            )
            if (
                m4_total_steps is not None
                and action_iteration is not None
                and m4_total_steps > action_iteration
            ):
                actions_reaching_placement += int(
                    window.get("accepted_action_count", 0) or 0
                )
        result["methods"]["m4"]["active_route_b_rounds"] = active_rounds
        result["methods"]["m4"]["actions_reaching_placement"] = actions_reaching_placement
        if active_rounds <= 0:
            result["failures"].append("m4_active_route_b_round_mismatch")
        if m3_rounds is not None and active_rounds != m3_rounds:
            result["failures"].append("m3_m4_route_b_round_mismatch")
        if actions_reaching_placement <= 0:
            result["failures"].append("m4_no_buffer_action_reached_later_placement")
    if "m1" in result["methods"] and not result["methods"]["m1"].get("r0_coordinate_match"):
        result["failures"].append("m1_r0_coordinate_mismatch")
    for baseline_method in (
        OPENROAD_REWEIGHT_ONLY_METHOD,
        OPENROAD_GPL_PLAIN_METHOD,
    ):
        if baseline_method not in result["methods"]:
            continue
        baseline_result = result["methods"][baseline_method]
        baseline_failures = _reweight_only_baseline_failures(baseline_result)
        baseline_result["baseline_audit"] = {
            "status": "pass" if not baseline_failures else "failed",
            "failures": baseline_failures,
        }
        if baseline_failures:
            baseline_result["status"] = "failed"
            result["failures"].append(
                f"{baseline_method.lower().replace('-', '_')}_baseline_failed"
            )

    if all(method in result["methods"] for method in ("m3", "m4")):
        m3_result = result["methods"]["m3"]
        m4_result = result["methods"]["m4"]
        m3_metrics = m3_result.get("track_a", {}).get("metrics", {})
        m4_metrics = m4_result.get("track_a", {}).get("metrics", {})
        m3_tns = _as_float(m3_metrics.get("tns_ns"))
        m4_tns = _as_float(m4_metrics.get("tns_ns"))
        m3_wns = _as_float(m3_metrics.get("wns_ns"))
        m4_wns = _as_float(m4_metrics.get("wns_ns"))
        m3_buffer_count = _committed_buffer_count(m3_result)
        m4_buffer_count = _committed_buffer_count(m4_result)
        m4_accepted_actions = _accepted_window_action_count(m4_result)
        resource_parity_pass = (
            m3_buffer_count is not None
            and m4_buffer_count is not None
            and m3_buffer_count == m4_buffer_count == m4_accepted_actions
        )
        coupling = {
            "tns_gain_ns": None if m3_tns is None or m4_tns is None else m4_tns - m3_tns,
            "wns_gain_ns": None if m3_wns is None or m4_wns is None else m4_wns - m3_wns,
            "resource_parity": {
                "m3_committed_buffer_count": m3_buffer_count,
                "m4_committed_buffer_count": m4_buffer_count,
                "m4_accepted_window_action_count": m4_accepted_actions,
                "pass": resource_parity_pass,
            },
        }
        if not resource_parity_pass:
            result["failures"].append("m3_m4_committed_buffer_budget_mismatch")
        if m3_wns is not None and m4_wns is not None:
            guard = max(0.020, 0.02 * abs(m3_wns))
            coupling["wns_guard_ns"] = guard
            coupling["wns_guard_pass"] = m4_wns >= m3_wns - guard
        coupling["tns_improved"] = coupling["tns_gain_ns"] is not None and coupling["tns_gain_ns"] > 0.0
        result["coupling"] = coupling
        if resource_parity_pass and not coupling["tns_improved"]:
            result["failures"].append("m4_did_not_improve_tns_over_m3")
        if resource_parity_pass and coupling.get("wns_guard_pass") is False:
            result["failures"].append("m4_failed_wns_guard")
        for metric_name in ("slew_violation_count", "cap_violation_count"):
            m3_value = _as_float(m3_metrics.get(metric_name))
            m4_value = _as_float(m4_metrics.get(metric_name))
            if m3_value is None or m4_value is None:
                result["failures"].append(f"missing_{metric_name}_guard")
                continue
            guard_pass = m4_value == 0.0 if m3_value == 0.0 else m4_value <= 1.05 * m3_value
            coupling[f"{metric_name}_guard_pass"] = guard_pass
            if resource_parity_pass and not guard_pass:
                result["failures"].append(f"m4_failed_{metric_name}_guard")
    result["status"] = "pass" if not result["failures"] else "failed"
    try:
        result["slack_histogram"] = _generate_slack_histogram(
            case_root,
            case,
            result["methods"],
        )
    except Exception as exc:
        result["slack_histogram"] = {
            "status": "failed",
            "reason": str(exc),
        }
    _write_json(case_root / "case_summary.json", result)
    return result


def _generate_slack_histogram(
    case_root: Path,
    case: str,
    methods: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    # Matplotlib 3.5 still reads this alias while rendering axes titles.
    if "Inf" not in np.__dict__:
        np.Inf = np.inf
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    series = {}
    for method, result in methods.items():
        endpoint_path = Path(
            str(result.get("track_a", {}).get("endpoint_csv") or "")
        )
        if not endpoint_path.is_file():
            continue
        values = []
        with endpoint_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                value = _as_float(row.get("slack_ns"))
                if value is not None:
                    values.append(value)
        if values:
            series[method] = np.asarray(values, dtype=np.float64)
    if not series:
        return {"status": "skipped", "reason": "no_endpoint_slacks"}
    all_values = np.concatenate(list(series.values()))
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    if upper <= lower:
        upper = lower + 1.0e-6
    bins = np.linspace(lower, upper, 81)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for method, values in series.items():
        axis.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=method,
        )
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_title(f"{case}: Track A endpoint slack")
    axis.set_xlabel("Setup slack (ns)")
    axis.set_ylabel("Endpoint count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90)
    png_path = case_root / "track_a_slack_histogram.png"
    pdf_path = case_root / "track_a_slack_histogram.pdf"
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    payload = {
        "status": "ok",
        "png": str(png_path),
        "pdf": str(pdf_path),
        "bin_count": 80,
        "slack_min_ns": lower,
        "slack_max_ns": upper,
        "methods": {
            method: {
                "endpoint_count": int(values.size),
                "violating_endpoint_count": int(np.sum(values < 0.0)),
            }
            for method, values in series.items()
        },
    }
    _write_json(case_root / "track_a_slack_histogram.json", payload)
    return payload


def _write_summary_csv(path: Path, case_results: list[dict[str, Any]]) -> None:
    columns = (
        "case",
        "method",
        "status",
        "wns_ns",
        "tns_ns",
        "violating_endpoints",
        "slew_violation_count",
        "cap_violation_count",
        "final_overflow",
        "buffer_count",
        "area_um2",
        "power_total_w",
        "timing_active_step_count",
        "timing_inactive_step_count",
        "topology_refresh_count",
        "first_timing_active_iteration",
        "first_timing_active_overflow",
        "timing_grad_balance_target_ratio",
        "timing_grad_balance_wl_grad_l1",
        "timing_grad_balance_timing_grad_l1",
        "timing_grad_balance_weight",
        "pin2pin_update_count",
        "pin2pin_first_update_iteration",
        "pin2pin_first_update_overflow",
        "pin2pin_pair_count",
        "pin2pin_weight",
        "pin2pin_path_limit",
        "pin2pin_path_limit_policy",
        "pin2pin_observed_weight_min",
        "pin2pin_observed_weight_max",
        "pin2pin_observed_weight_mean",
        "pin2pin_objective",
        "pin2pin_weighted_objective",
        "pin2pin_backend",
        "pin2pin_path_pair_backend",
        "pin2pin_state_domain",
        "pin2pin_recovery_endpoint_count",
        "pin2pin_pair_normalization_wns",
        "pin2pin_pair_normalization_policy",
        "method_runtime_sec",
        "track_a_runtime_sec",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case_result in case_results:
            for method, result in case_result.get("methods", {}).items():
                track_a = result.get("track_a", {})
                metrics = track_a.get("metrics", {})
                timing_gate = result.get("milestones", {}).get("timing_gate", {})
                timing_grad_balance = result.get("milestones", {}).get(
                    "timing_grad_balance",
                    {},
                )
                legacy_gate = result.get("milestones", {}).get(
                    "legacy_net_weight_gate",
                    {},
                )
                legacy_update = result.get("milestones", {}).get(
                    "legacy_net_weight",
                    {},
                )
                native_summary = dict(
                    legacy_update.get("pin2pin_native_summary", {}) or {}
                )
                writer.writerow(
                    {
                        "case": case_result["case"],
                        "method": method,
                        "status": result.get("status"),
                        "wns_ns": metrics.get("wns_ns"),
                        "tns_ns": metrics.get("tns_ns"),
                        "violating_endpoints": metrics.get("violating_endpoint_count"),
                        "slew_violation_count": metrics.get("slew_violation_count"),
                        "cap_violation_count": metrics.get("cap_violation_count"),
                        "final_overflow": result.get("milestones", {}).get(
                            "final_overflow"
                        ),
                        "buffer_count": track_a.get("mutation", {}).get("added_count"),
                        "area_um2": metrics.get("area_um2"),
                        "power_total_w": track_a.get("power_total_w"),
                        "timing_active_step_count": timing_gate.get(
                            "timing_active_step_count"
                        ),
                        "timing_inactive_step_count": timing_gate.get(
                            "timing_inactive_step_count"
                        ),
                        "topology_refresh_count": timing_gate.get(
                            "topology_refresh_count"
                        ),
                        "first_timing_active_iteration": timing_gate.get(
                            "first_timing_active_iteration"
                        ),
                        "first_timing_active_overflow": timing_gate.get(
                            "first_timing_active_overflow"
                        ),
                        "timing_grad_balance_target_ratio": timing_grad_balance.get(
                            "target_ratio"
                        ),
                        "timing_grad_balance_wl_grad_l1": timing_grad_balance.get(
                            "wirelength_grad_norm"
                        ),
                        "timing_grad_balance_timing_grad_l1": timing_grad_balance.get(
                            "timing_grad_norm"
                        ),
                        "timing_grad_balance_weight": timing_grad_balance.get(
                            "weight_applied"
                        ),
                        "pin2pin_update_count": legacy_gate.get("update_count"),
                        "pin2pin_first_update_iteration": legacy_gate.get(
                            "first_update_iteration"
                        ),
                        "pin2pin_first_update_overflow": legacy_gate.get(
                            "first_update_overflow"
                        ),
                        "pin2pin_pair_count": legacy_update.get(
                            "pin2pin_pair_count"
                        ),
                        "pin2pin_weight": legacy_update.get("pin2pin_weight"),
                        "pin2pin_path_limit": legacy_update.get(
                            "pin2pin_path_limit"
                        ),
                        "pin2pin_path_limit_policy": legacy_update.get(
                            "pin2pin_path_limit_policy"
                        ),
                        "pin2pin_observed_weight_min": legacy_update.get(
                            "pin2pin_observed_weight_min"
                        ),
                        "pin2pin_observed_weight_max": legacy_update.get(
                            "pin2pin_observed_weight_max"
                        ),
                        "pin2pin_observed_weight_mean": legacy_update.get(
                            "pin2pin_observed_weight_mean"
                        ),
                        "pin2pin_objective": legacy_update.get(
                            "pin2pin_objective"
                        ),
                        "pin2pin_weighted_objective": legacy_update.get(
                            "pin2pin_weighted_objective"
                        ),
                        "pin2pin_backend": legacy_update.get("pin2pin_backend"),
                        "pin2pin_path_pair_backend": legacy_update.get(
                            "pin2pin_path_pair_backend"
                        ),
                        "pin2pin_state_domain": native_summary.get(
                            "state_domain"
                        ),
                        "pin2pin_recovery_endpoint_count": native_summary.get(
                            "recovery_endpoint_count"
                        ),
                        "pin2pin_pair_normalization_wns": native_summary.get(
                            "pair_normalization_wns"
                        ),
                        "pin2pin_pair_normalization_policy": native_summary.get(
                            "pair_normalization_policy"
                        ),
                        "method_runtime_sec": result.get("execution", {}).get("runtime_sec"),
                        "track_a_runtime_sec": track_a.get("execution", {}).get("runtime_sec"),
                    }
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--openroad-num-threads", type=int, default=32)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--method", dest="method_values", action="append")
    parser.add_argument("--outer-iterations", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--input-r0-def",
        type=Path,
        help="Use a prepared R0 DEF; accepts a literal path or a {case} template.",
    )
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Physical CUDA visibility binding. When set, --gpu-id is the "
            "logical device index inside that visibility set."
        ),
    )
    parser.add_argument(
        "--placement-optimizer",
        default=DEFAULT_PLACEMENT_OPTIMIZER,
    )
    parser.add_argument(
        "--enable-fillers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable runtime placement fillers; defaults to enabled for "
            "filler-compatible placement arms"
        ),
    )
    parser.add_argument(
        "--legalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run AutoDMP standard-cell legalization after global placement",
    )
    parser.add_argument(
        "--timing-grad-balance-target-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--timing-topology-enable-overflow-threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument("--stop-overflow", type=float, default=0.1)
    parser.add_argument("--pin2pin-weight", type=float, default=2.5e-5)
    parser.add_argument("--pin2pin-min-weight", type=float, default=10.0)
    parser.add_argument("--pin2pin-max-weight", type=float, default=50.0)
    parser.add_argument("--pin2pin-accumulate-weight", type=float, default=0.2)
    parser.add_argument(
        "--net-weighting-npaths",
        type=float,
        default=0.0,
        help="legacy net-selection percentage; 3.0 matches Efficient-TDP and 0 keeps all",
    )
    parser.add_argument("--pin2pin-update-interval", type=int, default=15)
    parser.add_argument(
        "--joint-segment-milestone",
        dest="joint_segment_milestones",
        action="append",
        type=float,
    )
    parser.add_argument("--joint-segment-sizing-rounds", type=int, default=5)
    parser.add_argument(
        "--joint-segment-sizing-up-percent",
        type=float,
        default=10.0,
    )
    parser.add_argument("--joint-segment-buffering-rounds", type=int, default=1)
    parser.add_argument(
        "--joint-segment-buffering-selection-fraction",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--joint-segment-pin2pin-rebootstrap-enabled",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--num-bins-x", type=int)
    parser.add_argument("--num-bins-y", type=int)
    parser.add_argument("--plot", type=int, choices=(0, 1))
    parser.add_argument("--track-b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="write source raw DEF/artifacts without running the legacy evaluator",
    )
    parser.add_argument(
        "--r0-only",
        action="store_true",
        help="generate and qualify deterministic R0 without running a method",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_root = args.output_root.resolve()
    args.python_bin = args.python_bin.resolve()
    args.openroad_bin = args.openroad_bin.resolve()
    if args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    if not math.isfinite(args.net_weighting_npaths) or not 0.0 <= args.net_weighting_npaths <= 100.0:
        raise ValueError("net_weighting_npaths must be a percentage in [0, 100]")
    if args.num_bins_x is not None and args.num_bins_x <= 0:
        raise ValueError("num_bins_x must be positive")
    if args.num_bins_y is not None and args.num_bins_y <= 0:
        raise ValueError("num_bins_y must be positive")
    if not (
        math.isfinite(args.timing_topology_enable_overflow_threshold)
        and 0.0 < args.timing_topology_enable_overflow_threshold <= 1.0
    ):
        raise ValueError(
            "timing_topology_enable_overflow_threshold must be in (0, 1]"
        )
    if not math.isfinite(args.stop_overflow) or not 0.0 < args.stop_overflow <= 1.0:
        raise ValueError("stop_overflow must be in (0, 1]")
    if args.pin2pin_update_interval <= 0:
        raise ValueError("pin2pin_update_interval must be positive")
    if args.joint_segment_milestones is None:
        args.joint_segment_milestones = [0.30, 0.25, 0.20, 0.15, 0.10]
    if any(
        not math.isfinite(value)
        for value in args.joint_segment_milestones
    ) or any(
        args.joint_segment_milestones[index]
        <= args.joint_segment_milestones[index + 1]
        for index in range(len(args.joint_segment_milestones) - 1)
    ):
        raise ValueError("joint_segment_milestones must be strictly descending")
    if args.joint_segment_sizing_rounds <= 0:
        raise ValueError("joint_segment_sizing_rounds must be positive")
    if not 0.0 < args.joint_segment_sizing_up_percent <= 100.0:
        raise ValueError("joint_segment_sizing_up_percent must be in (0, 100]")
    if args.joint_segment_buffering_rounds != 1:
        raise ValueError("joint_segment_buffering_rounds must equal one")
    if not 0.0 < args.joint_segment_buffering_selection_fraction <= 1.0:
        raise ValueError(
            "joint_segment_buffering_selection_fraction must be in (0, 1]"
        )
    args.cases = _selected(args.cases, ICCAD24_CASES) if args.cases else [ICCAD24_CASES[0]]
    if args.r0_only and args.method_values:
        raise ValueError("--r0-only cannot be combined with --method")
    args.methods = (
        []
        if args.r0_only
        else (
            _selected(args.method_values, EXPERIMENT_METHODS)
            if args.method_values
            else list(METHODS)
        )
    )
    baseline_methods = {
        OPENROAD_REWEIGHT_ONLY_METHOD,
        OPENROAD_GPL_PLAIN_METHOD,
        OPENROAD_NATIVE_FULL_FLOW_METHOD,
    }
    legacy_methods = (
        set(args.methods) - set(OVERFLOW_MILESTONE_METHODS) - baseline_methods
    )
    milestone_methods = set(args.methods) & set(OVERFLOW_MILESTONE_METHODS)
    args.enable_fillers = _resolve_enable_fillers(
        args.enable_fillers,
        args.methods,
    )
    if legacy_methods and milestone_methods:
        raise ValueError(
            "legacy M1-M4 and overflow-milestone P/TDP/S/B/SB protocols must run "
            "in separate campaigns"
        )
    if args.iterations is None and milestone_methods:
        args.iterations = DEFAULT_OVERFLOW_MILESTONE_ITERATIONS
    if args.outer_iterations is None:
        args.outer_iterations = 1 if milestone_methods or not legacy_methods else 3
    required_outer_iterations = 1 if milestone_methods or not legacy_methods else 3
    if args.outer_iterations != required_outer_iterations:
        protocol = "overflow-milestone" if milestone_methods else "controlled primary"
        raise ValueError(
            f"the {protocol} protocol requires K={required_outer_iterations}"
        )
    run_root = args.output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "artifact": "random_init_placement_buffering_campaign_manifest",
        "artifact_version": 1,
        "run_id": args.run_id,
        "cases": args.cases,
        "methods": args.methods,
        "K": args.outer_iterations,
        "max_iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "benchmark_root": str(args.benchmark_root),
        "python_bin": str(args.python_bin),
        "openroad_bin": str(args.openroad_bin),
        "openroad_bin_sha256": sha256_file(args.openroad_bin),
        "openroad_version": _tool_output([str(args.openroad_bin), "-version"], AUTODMP_ROOT),
        "autodmp_commit": _git_head(AUTODMP_ROOT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "gpu_id": args.gpu_id,
        "cuda_visible_devices": (
            args.cuda_visible_devices
            if args.cuda_visible_devices is not None
            else str(args.gpu_id)
        ),
        "placement_optimizer": args.placement_optimizer,
        "enable_fillers": args.enable_fillers,
        "legalize": args.legalize,
        "timing_grad_balance_target_ratio": (
            args.timing_grad_balance_target_ratio
        ),
        "timing_topology_enable_overflow_threshold": (
            args.timing_topology_enable_overflow_threshold
        ),
        "stop_overflow": args.stop_overflow,
        "pin2pin_weight": args.pin2pin_weight,
        "pin2pin_min_weight": args.pin2pin_min_weight,
        "pin2pin_max_weight": args.pin2pin_max_weight,
        "pin2pin_accumulate_weight": args.pin2pin_accumulate_weight,
        "net_weighting_npaths": args.net_weighting_npaths,
        "pin2pin_update_interval": args.pin2pin_update_interval,
        "joint_segment_milestones": list(args.joint_segment_milestones),
        "joint_segment_sizing_rounds": args.joint_segment_sizing_rounds,
        "joint_segment_sizing_up_percent": args.joint_segment_sizing_up_percent,
        "joint_segment_buffering_rounds": args.joint_segment_buffering_rounds,
        "joint_segment_buffering_selection_fraction": (
            args.joint_segment_buffering_selection_fraction
        ),
        "joint_segment_pin2pin_rebootstrap_enabled": bool(
            args.joint_segment_pin2pin_rebootstrap_enabled
        ),
        "num_bins_x": args.num_bins_x,
        "num_bins_y": args.num_bins_y,
        "track_b": args.track_b,
        "source_only": args.source_only,
        "r0_only": args.r0_only,
        "input_r0_def": (
            str(args.input_r0_def.resolve()) if args.input_r0_def is not None else None
        ),
        "plot": args.plot,
    }
    _write_json(run_root / "campaign_manifest.json", provenance)
    case_results = []
    for case in args.cases:
        result = _run_case(args, case, run_root)
        case_results.append(result)
        print(
            f"{case}: status={result.get('status')} "
            f"failures={','.join(result.get('failures', ())) or '-'}"
        )
    summary = {"provenance": provenance, "case_results": case_results}
    _write_json(run_root / "campaign_summary.json", summary)
    _write_summary_csv(run_root / "campaign_summary.csv", case_results)
    return 0 if all(row.get("status") in {"pass", "plan_only"} for row in case_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
