#!/usr/bin/env python3
"""Implementation behind the tools-paper campaign CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = SCRIPT_DIR.parents[6]
PROFILE_PATH = SCRIPT_DIR / "profiles" / "tools_release_v1.json"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
DEFAULT_OUTPUT_ROOT = AUTODMP_ROOT / "logs" / "regression" / "iccad24" / "tools_paper_four_table"
DEFAULT_OPENROAD = Path("/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
GNU_TIME = Path("/usr/bin/time")
COMMON_TRACK_A = AUTODMP_ROOT / "test/regression/common/def_only_track_a.py"
DIFF_SIZING_RUNNER = AUTODMP_ROOT / "test/regression/iccad24/diff_sizing/run.py"
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
CANDIDATE_BUFFERING_RUNNER = (
    AUTODMP_ROOT / "test/regression/iccad24/buffering_inner_loop/run.py"
)
CANDIDATE_BUFFERING_PROFILE = (
    AUTODMP_ROOT
    / "test/regression/iccad24/buffering_inner_loop/params/candidate_discrete_net_gradient_allcases_cuda.json"
)
STAGED_FULL_FLOW_RUNNER = (
    AUTODMP_ROOT / "test/regression/iccad24/staged_joint/run.py"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate
import artifact_adapters as adapters
import campaign_contract as contract
import qualification as artifact_qualification


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_output(command: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (completed.stdout or completed.stderr).strip() or "unavailable"


def _git_head(path: Path) -> str:
    return _tool_output(["git", "rev-parse", "HEAD"], path)


def _git_tracked_diff_sha256(path: Path) -> str:
    completed = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to capture tracked worktree diff: "
            + completed.stderr.decode("utf-8", errors="ignore")
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_file_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        SCRIPT_DIR / "run.py",
        SCRIPT_DIR / "campaign_contract.py",
        SCRIPT_DIR / "artifact_adapters.py",
        SCRIPT_DIR / "aggregate.py",
        SCRIPT_DIR / "qualification.py",
        PROFILE_PATH,
        AUTODMP_ROOT / "test/regression/common/def_only_track_a.py",
        AUTODMP_ROOT
        / "test/regression/iccad24/joint_place_sizing_buffering/run_random_init_validation.py",
        AUTODMP_ROOT / "test/regression/iccad24/diff_sizing/run.py",
        AUTODMP_ROOT / "test/regression/iccad24/equal_spaced_buffering/run.py",
        EQUAL_BUFFERING_PROFILE,
        AUTODMP_ROOT / "test/regression/iccad24/buffering_inner_loop/run.py",
        CANDIDATE_BUFFERING_PROFILE,
        STAGED_FULL_FLOW_RUNNER,
        AUTODMP_ROOT / "dreamplace/Placer.py",
        AUTODMP_ROOT / "dreamplace/placer_cli.py",
        AUTODMP_ROOT / "dreamplace/NonLinearPlace.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing campaign source file(s): " + ", ".join(missing))
    return {
        str(path.relative_to(AUTODMP_ROOT)): _sha256(path)
        for path in paths
    }


def _native_library_hashes() -> dict[str, str]:
    relative_paths = (
        "build/dreamplace/ops/pin2pin_attraction/pin2pin_attraction_cuda.cpython-311-x86_64-linux-gnu.so",
        "build/dreamplace/ops/steiner_topo/steiner_topo_cpp.cpython-311-x86_64-linux-gnu.so",
        "build/dreamplace/ops/timing_propagation/timing_propagation_cpp.cpython-311-x86_64-linux-gnu.so",
    )
    paths = [AUTODMP_ROOT / relative for relative in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing campaign native library(s): " + ", ".join(missing))
    return {relative: _sha256(path) for relative, path in zip(relative_paths, paths)}


def _gpu_model(physical_gpu_id: int) -> str:
    return _tool_output(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            str(physical_gpu_id),
        ],
        AUTODMP_ROOT,
    )


def _gpu_uuid(physical_gpu_id: int) -> str:
    return _tool_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            "-i",
            str(physical_gpu_id),
        ],
        AUTODMP_ROOT,
    )


def _gpu_compute_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    output = _tool_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        AUTODMP_ROOT,
    )
    if output in {"", "unavailable", "No running processes found"}:
        return []
    processes = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        try:
            pid = int(fields[1])
            used_memory_mb = int(fields[3])
        except ValueError as error:
            raise ValueError(f"invalid nvidia-smi compute-process row: {line}") from error
        processes.append(
            {
                "pid": pid,
                "process_name": fields[2],
                "used_memory_mb": used_memory_mb,
            }
        )
    return sorted(processes, key=lambda process: process["pid"])


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor() or "unknown"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


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
        "def": root / f"{case}.def",
        "sdc": root / f"{case}.sdc",
    }


def _track_a_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    method_id: str,
    reference_def: Path,
    def_input: Path,
    output_dir: Path,
    evaluation_mode: str = "track_a",
) -> list[str]:
    inputs = _case_inputs(args.benchmark_root.resolve(), case)
    tech = _asap7_inputs(args.benchmark_root.resolve())
    command = [
        str(args.python_bin.resolve()),
        str(COMMON_TRACK_A),
        "--case",
        case,
        "--method-id",
        method_id,
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(int(args.cpu_threads)),
        "--reference-def",
        str(reference_def),
        "--def-input",
        str(def_input),
        "--tech-lef",
        str(tech["tech_lef"]),
    ]
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
            str(profile["buffering"]["buffer_master"]),
            "--output-dir",
            str(output_dir),
            "--evaluation-mode",
            evaluation_mode,
            "--detailed-placement-search-window",
            str(profile["track_a"]["detailed_placement_search_window"]),
        )
    )
    return command


def _sizing_source_command(
    *,
    args: argparse.Namespace,
    case: str,
    method_id: str,
    attempt_root: Path,
) -> tuple[list[str], Path]:
    source_method = {
        "or_pure_sizeup": "or_pure_sizeup",
        "autodmp_diff_sizing": "autodmp_diff_sizing",
    }[method_id]
    source_output_root = attempt_root / "source"
    source_run_id = "run"
    command = [
        str(args.python_bin.resolve()),
        str(DIFF_SIZING_RUNNER),
        "--profile",
        str(args.profile.resolve()),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--output-root",
        str(source_output_root),
        "--run-id",
        source_run_id,
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(int(args.cpu_threads)),
        "--case",
        case,
        "--method",
        source_method,
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--logical-gpu-id",
        "0",
    ]
    result_path = (
        source_output_root
        / source_run_id
        / case
        / source_method
        / "result.json"
    )
    return command, result_path


def _placement_source_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    method_id: str,
    attempt_root: Path,
) -> tuple[list[str], Path, Path, Path]:
    source_method = {
        "or_gpl_plain": "OR-GPL-plain",
        "or_tdp_reweight_only": "OR-TDP-reweight-only",
        "autodmp_p_ctrl": "P",
        "autodmp_pin2pin": "TDP-pin2pin",
    }[method_id]
    source_output_root = attempt_root / "source"
    source_run_id = "run"
    placement = dict(profile["placement"])
    r0 = dict(profile["input_domains"]["R0"])
    command = [
        str(args.python_bin.resolve()),
        str(PLACEMENT_RUNNER),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--output-root",
        str(source_output_root),
        "--run-id",
        source_run_id,
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(int(args.cpu_threads)),
        "--case",
        case,
        "--method",
        source_method,
        "--outer-iterations",
        "1",
        "--iterations",
        str(int(placement["max_steps"])),
        "--seed",
        str(int(r0["seed"])),
        "--gpu-id",
        "0",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--placement-optimizer",
        str(placement["optimizer"]),
        "--plot",
        "0",
        "--no-track-b",
        "--source-only",
    ]
    if method_id in {"autodmp_p_ctrl", "autodmp_pin2pin"}:
        command.append("--enable-fillers")
    if method_id == "autodmp_pin2pin":
        pin2pin = dict(placement["pin2pin"])
        command.extend(
            (
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
            )
        )
    case_root = source_output_root / source_run_id / case
    result_path = case_root / "case_summary.json"
    raw_def = case_root / source_method / f"{case}_raw.def"
    r0_def = case_root / "r0" / "R0.def"
    return command, result_path, raw_def, r0_def


def _buffering_source_command(
    *,
    args: argparse.Namespace,
    case: str,
    method_id: str,
    attempt_root: Path,
    campaign_manifest_path: Path,
) -> tuple[list[str], Path, Path]:
    source_output_root = attempt_root / "source"
    source_run_id = "run"
    if method_id in {
        "or_repair_design_buffer_only",
        "autodmp_equal_spaced_route_b",
    }:
        source_method = (
            "b1_rd_rt"
            if method_id == "or_repair_design_buffer_only"
            else "ours"
        )
        command = [
            str(args.python_bin.resolve()),
            str(EQUAL_BUFFERING_RUNNER),
            "--profile",
            str(EQUAL_BUFFERING_PROFILE),
            "--benchmark-root",
            str(args.benchmark_root.resolve()),
            "--output-root",
            str(source_output_root),
            "--run-id",
            source_run_id,
            "--python-bin",
            str(args.python_bin.resolve()),
            "--openroad-bin",
            str(args.openroad_bin.resolve()),
            "--openroad-num-threads",
            str(int(args.cpu_threads)),
            "--case",
            case,
            "--method",
            source_method,
            "--segment-strategy",
            "discrete_net_gradient",
            "--cuda-visible-devices",
            str(args.gpu_id),
            "--logical-gpu-id",
            "0",
            "--source-only",
        ]
        result_path = source_output_root / source_run_id / case / source_method / "result.json"
        raw_def = (
            source_output_root
            / source_run_id
            / case
            / source_method
            / f"{case}_{source_method}"
        )
        raw_def = raw_def.with_name(
            raw_def.name + ("_committed.def" if source_method == "ours" else ".def")
        )
        return command, result_path, raw_def

    if method_id != "autodmp_candidate_route_b":
        raise ValueError(f"unsupported buffering method: {method_id}")
    command = [
        str(args.python_bin.resolve()),
        str(CANDIDATE_BUFFERING_RUNNER),
        "--profile",
        str(CANDIDATE_BUFFERING_PROFILE),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--output-root",
        str(source_output_root),
        "--run-id",
        source_run_id,
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(int(args.cpu_threads)),
        "--case",
        case,
        "--buffering-commit-enabled",
        "1",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--logical-gpu-id",
        "0",
        "--source-only",
        "--external-provenance-manifest",
        str(campaign_manifest_path),
    ]
    result_path = source_output_root / source_run_id / "manifest.json"
    raw_def = (
        source_output_root
        / source_run_id
        / case
        / f"{case}_candidate_committed.def"
    )
    return command, result_path, raw_def


def _full_flow_source_command(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    case: str,
    method_id: str,
    attempt_root: Path,
) -> tuple[list[str], Path, Path, Path]:
    source_output_root = attempt_root / "source"
    source_run_id = "run"
    if method_id == "autodmp_staged_psb":
        command = [
            str(args.python_bin.resolve()),
            str(STAGED_FULL_FLOW_RUNNER),
            "--benchmark-root",
            str(args.benchmark_root.resolve()),
            "--output-root",
            str(source_output_root),
            "--run-id",
            source_run_id,
            "--python-bin",
            str(args.python_bin.resolve()),
            "--openroad-bin",
            str(args.openroad_bin.resolve()),
            "--openroad-num-threads",
            str(int(args.cpu_threads)),
            "--case",
            case,
            "--tools-release-method",
            method_id,
            "--tools-release-profile",
            str(args.profile.resolve()),
            "--cuda-visible-devices",
            str(args.gpu_id),
            "--logical-gpu-id",
            "0",
        ]
        method_root = source_output_root / source_run_id / case / method_id
        result_path = method_root / "result.json"
        raw_def = (
            method_root
            / "stage3_buffering"
            / "source"
            / "run"
            / case
            / "ours"
            / f"{case}_ours_committed.def"
        )
        r0_def = (
            method_root
            / "stage1_placement"
            / "source"
            / "run"
            / case
            / "r0"
            / "R0.def"
        )
        return command, result_path, raw_def, r0_def

    source_method = {
        "or_native_full_flow": "OR-native-full-flow",
        "autodmp_coordinated_psb": "TDP-pin2pin-SB",
    }.get(method_id)
    if source_method is None:
        raise ValueError(f"unsupported full-flow method: {method_id}")
    placement = dict(profile["placement"])
    pin2pin = dict(placement["pin2pin"])
    r0 = dict(profile["input_domains"]["R0"])
    command = [
        str(args.python_bin.resolve()),
        str(PLACEMENT_RUNNER),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--output-root",
        str(source_output_root),
        "--run-id",
        source_run_id,
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(int(args.cpu_threads)),
        "--case",
        case,
        "--method",
        source_method,
        "--outer-iterations",
        "1",
        "--iterations",
        str(int(placement["max_steps"])),
        "--seed",
        str(int(r0["seed"])),
        "--gpu-id",
        "0",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--placement-optimizer",
        str(placement["optimizer"]),
        "--plot",
        "0",
        "--no-track-b",
        "--source-only",
    ]
    if method_id == "autodmp_coordinated_psb":
        command.extend(
            (
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
            )
        )
    case_root = source_output_root / source_run_id / case
    result_path = case_root / "case_summary.json"
    raw_def = (
        case_root
        / source_method
        / (
            f"{case}_raw.def"
            if method_id == "or_native_full_flow"
            else f"autodmp_result/{case}_segment_joint_final.def"
        )
    )
    r0_def = case_root / "r0" / "R0.def"
    return command, result_path, raw_def, r0_def


def _run_attempt_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if not GNU_TIME.is_file():
        raise FileNotFoundError(GNU_TIME)
    peak_rss_path = stdout_path.parent / f"{stdout_path.stem}_peak_rss_kb.txt"
    wrapped_command = [
        str(GNU_TIME),
        "-f",
        "%M",
        "-o",
        str(peak_rss_path),
        "--",
        *command,
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            wrapped_command,
            cwd=AUTODMP_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    try:
        peak_lines = peak_rss_path.read_text(encoding="utf-8").splitlines()
        peak_values = [
            int(line.strip())
            for line in peak_lines
            if line.strip().isdigit()
        ]
        if len(peak_values) != 1:
            raise ValueError(f"expected one RSS value, found {peak_values}")
        peak_rss_kb = peak_values[0]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to collect peak RSS for {command[0]}") from error
    return {
        "returncode": int(completed.returncode),
        "runtime_sec": time.perf_counter() - started,
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_path": str(peak_rss_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _attempt_id(
    run_root: Path,
    *,
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
) -> str:
    parent = run_root / "attempts" / table_id / case / method_id
    existing = sorted(path.name for path in parent.glob("attempt-*") if path.is_dir())
    sequence = len(existing) + 1
    suffix = contract.digest_payload(
        {
            "campaign_id": campaign_id,
            "table_id": table_id,
            "case": case,
            "method_id": method_id,
            "sequence": sequence,
        }
    )[:12]
    return f"attempt-{sequence:04d}-{suffix}"


def _load_or_run_dpost_input_metrics(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    run_root: Path,
    case: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    input_root = run_root / "inputs" / "D_post" / case
    summary_path = input_root / "track_a_summary.json"
    if summary_path.is_file():
        summary = adapters.read_json_artifact(
            summary_path,
            artifact=adapters.TRACK_A_ARTIFACT,
            versions={adapters.TRACK_A_VERSION},
        )
        if summary.get("status") != "pass":
            raise ValueError(f"{summary_path}: cached D_post Track A did not pass")
        return dict(summary.get("metrics") or {})
    def_input = _case_inputs(args.benchmark_root.resolve(), case)["def"]
    command = _track_a_command(
        args=args,
        profile=profile,
        case=case,
        method_id="input_dpost",
        reference_def=def_input,
        def_input=def_input,
        output_dir=input_root,
    )
    (input_root / "command.txt").parent.mkdir(parents=True, exist_ok=True)
    (input_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    execution = _run_attempt_command(
        command,
        stdout_path=input_root / "stdout.log",
        stderr_path=input_root / "stderr.log",
        environment=environment,
    )
    _write_json(input_root / "execution.json", execution)
    if execution["returncode"] != 0 or not summary_path.is_file():
        raise RuntimeError(f"{case}: D_post canonical input evaluation failed")
    summary = adapters.read_json_artifact(
        summary_path,
        artifact=adapters.TRACK_A_ARTIFACT,
        versions={adapters.TRACK_A_VERSION},
    )
    if summary.get("status") != "pass":
        raise RuntimeError(f"{case}: D_post canonical input evaluation did not pass")
    return dict(summary.get("metrics") or {})


def _load_or_run_r0_input_metrics(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    run_root: Path,
    case: str,
    r0_def: Path,
    r0_def_sha256: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    input_root = run_root / "inputs" / "R0" / case / "input_state"
    summary_path = input_root / "input_state_summary.json"
    input_identity_path = input_root / "input_identity.json"
    expected_identity = {
        "artifact": "autodmp_tools_paper_r0_timing_input_identity",
        "artifact_version": 1,
        "case": case,
        "r0_def_sha256": r0_def_sha256,
    }
    if summary_path.is_file():
        if not input_identity_path.is_file():
            raise ValueError("cached R0 timing input has no identity artifact")
        cached_identity = json.loads(
            input_identity_path.read_text(encoding="utf-8")
        )
        if cached_identity != expected_identity:
            raise ValueError("cached R0 timing input identity mismatch")
        summary = adapters.read_json_artifact(
            summary_path,
            artifact=adapters.INPUT_STATE_ARTIFACT,
            versions={adapters.TRACK_A_VERSION},
        )
        if summary.get("status") != "pass":
            raise ValueError(f"{summary_path}: cached R0 Track A did not pass")
        return dict(summary.get("metrics") or {})
    if not r0_def.is_file() or _sha256(r0_def) != r0_def_sha256:
        raise ValueError("R0 timing input DEF hash mismatch")
    command = _track_a_command(
        args=args,
        profile=profile,
        case=case,
        method_id="input_r0",
        reference_def=r0_def,
        def_input=r0_def,
        output_dir=input_root,
        evaluation_mode="input_state",
    )
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    _write_json(input_identity_path, expected_identity)
    execution = _run_attempt_command(
        command,
        stdout_path=input_root / "stdout.log",
        stderr_path=input_root / "stderr.log",
        environment=environment,
    )
    _write_json(input_root / "execution.json", execution)
    if execution["returncode"] != 0 or not summary_path.is_file():
        raise RuntimeError(f"{case}: R0 canonical input evaluation failed")
    summary = adapters.read_json_artifact(
        summary_path,
        artifact=adapters.INPUT_STATE_ARTIFACT,
        versions={adapters.TRACK_A_VERSION},
    )
    if summary.get("status") != "pass":
        raise RuntimeError(f"{case}: R0 canonical input evaluation did not pass")
    return dict(summary.get("metrics") or {})


def _execution_environment(execution: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    thread_count = str(int(execution["omp_num_threads"]))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(execution["cuda_visible_devices"]),
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )
    return environment


def _write_normalized_rows(run_root: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            contract.PRIMARY_TABLE_ORDER.index(row["table_id"]),
            contract.ICCAD24_CASES.index(row["case"]),
            contract.PRIMARY_METHODS[row["table_id"]].index(row["method_id"]),
        ),
    )
    _write_json(run_root / "normalized_rows.json", ordered)
    columns = (
        "campaign_id",
        "table_id",
        "case",
        "method_id",
        "attempt_id",
        "input_domain",
        "required_cuda",
        "status",
        "terminal_reason",
        "input_metrics",
        "final_metrics",
        "mutation",
        "runtime",
        "source_artifact",
        "track_a_artifact",
    )
    with (run_root / "normalized_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), sort_keys=True, allow_nan=False)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in columns
                }
            )


def _load_normalized_rows(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "normalized_rows.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: normalized rows must be a list")
    return [dict(row) for row in payload]


def _seal_or_fail_row(
    row: dict[str, Any],
    *,
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
    attempt_id: str,
    input_domain: str,
    required_cuda: bool,
    source_execution: dict[str, Any] | None,
    track_a_execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if row.get("status") != "pass":
        return row
    try:
        if source_execution is None:
            raise ValueError("passing row is missing parent execution evidence")
        source_peak = int(source_execution["peak_rss_kb"])
        track_a_peak = int(
            (track_a_execution or source_execution)["peak_rss_kb"]
        )
        if source_peak < 0 or track_a_peak < 0:
            raise ValueError("peak RSS must be nonnegative")
        runtime = dict(row.get("runtime") or {})
        runtime.update(
            {
                "source_peak_rss_kb": source_peak,
                "track_a_peak_rss_kb": track_a_peak,
                "peak_rss_kb": max(source_peak, track_a_peak),
            }
        )
        row = {**row, "runtime": runtime}
        return artifact_qualification.seal_passing_row(row)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return adapters.terminal_failure_row(
            campaign_id=campaign_id,
            table_id=table_id,
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain=input_domain,
            required_cuda=required_cuda,
            status="artifact_contract_failed",
            terminal_reason=str(error),
            source_artifact=dict(row.get("source_artifact") or {}),
        )


def _record_r0_identity(
    row: dict[str, Any],
    *,
    run_root: Path,
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
    attempt_id: str,
    required_cuda: bool,
) -> dict[str, Any]:
    if row.get("status") != "pass":
        return row
    r0_hash = dict(row.get("source_artifact") or {}).get(
        "r0_movable_xy_sha256"
    )
    source = dict(row.get("source_artifact") or {})
    r0_def = Path(str(source.get("r0_def") or ""))
    r0_def_sha256 = str(source.get("r0_def_sha256") or "")
    if (
        len(str(r0_hash or "")) != 64
        or len(r0_def_sha256) != 64
        or not r0_def.is_file()
        or _sha256(r0_def) != r0_def_sha256
    ):
        return adapters.terminal_failure_row(
            campaign_id=campaign_id,
            table_id=table_id,
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="R0",
            required_cuda=required_cuda,
            status="provenance_mismatch",
            terminal_reason="R0 DEF or coordinate identity is missing or invalid",
            source_artifact=source,
        )
    r0_identity = {
        "artifact": "autodmp_tools_paper_r0_identity",
        "artifact_version": 1,
        "case": case,
        "movable_nodes_xy_sha256": r0_hash,
        "r0_def_sha256": r0_def_sha256,
    }
    r0_identity_path = run_root / "inputs" / "R0" / case / "identity.json"
    if r0_identity_path.is_file():
        existing_identity = json.loads(r0_identity_path.read_text(encoding="utf-8"))
        if existing_identity != r0_identity:
            return adapters.terminal_failure_row(
                campaign_id=campaign_id,
                table_id=table_id,
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="R0",
                required_cuda=required_cuda,
                status="provenance_mismatch",
                terminal_reason="R0 coordinate identity differs across R0-domain arms",
                source_artifact=dict(row.get("source_artifact") or {}),
            )
    else:
        _write_json(r0_identity_path, r0_identity)
    return row


def _attach_r0_input_metrics(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    run_root: Path,
    environment: dict[str, str],
    campaign_id: str,
    table_id: str,
    case: str,
    method_id: str,
    attempt_id: str,
    required_cuda: bool,
) -> dict[str, Any]:
    if row.get("status") != "pass":
        return row
    source = dict(row.get("source_artifact") or {})
    try:
        input_metrics = _load_or_run_r0_input_metrics(
            args=args,
            profile=profile,
            run_root=run_root,
            case=case,
            r0_def=Path(str(source["r0_def"])),
            r0_def_sha256=str(source["r0_def_sha256"]),
            environment=environment,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return adapters.terminal_failure_row(
            campaign_id=campaign_id,
            table_id=table_id,
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="R0",
            required_cuda=required_cuda,
            status="artifact_contract_failed",
            terminal_reason=str(error),
            source_artifact=source,
        )
    return {**row, "input_metrics": input_metrics}


def _validate_selected_rows(
    rows: list[dict[str, Any]],
    *,
    campaign_id: str,
    selected_cases: tuple[str, ...],
    selected_tables: tuple[str, ...],
    require_complete: bool,
) -> None:
    expected_keys = {
        (row["table_id"], row["case"], row["method_id"])
        for row in contract.primary_matrix(selected_cases)
        if row["table_id"] in selected_tables
    }
    actual_keys = []
    for row in rows:
        contract.validate_normalized_row(row, campaign_id=campaign_id)
        if row.get("status") == "pass":
            artifact_qualification.validate_sealed_passing_row(
                row,
                campaign_id=campaign_id,
            )
        key = (row["table_id"], row["case"], row["method_id"])
        if key not in expected_keys:
            raise ValueError(f"normalized row is outside selected matrix: {key}")
        actual_keys.append(key)
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("normalized rows contain duplicate keys")
    if require_complete:
        contract.validate_completed_matrix(
            rows,
            cases=selected_cases,
            tables=selected_tables,
            campaign_id=campaign_id,
        )


def _write_final_tables(
    *,
    run_root: Path,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    campaign_id: str,
    selected_cases: tuple[str, ...],
    selected_tables: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    _validate_selected_rows(
        rows,
        campaign_id=campaign_id,
        selected_cases=selected_cases,
        selected_tables=selected_tables,
        require_complete=True,
    )
    artifacts = aggregate.write_table_files(
        run_root / "tables",
        rows,
        table_order=selected_tables,
        case_order=selected_cases,
        method_order=profile["methods"],
    )
    failures = [
        {
            "table_id": row["table_id"],
            "case": row["case"],
            "method_id": row["method_id"],
            "attempt_id": row["attempt_id"],
            "status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
        }
        for row in rows
        if row.get("status") != "pass"
    ]
    failure_report_path = run_root / "failure_and_limitations.json"
    _write_json(
        failure_report_path,
        {
            "artifact": "autodmp_tools_paper_failure_and_limitations",
            "artifact_version": 1,
            "campaign_id": campaign_id,
            "failure_count": len(failures),
            "failures": failures,
            "limitations": [
                "QoR wins are TNS-first Pareto results, not unconditional dominance.",
                "Runtime speedup is valid only inside one runtime_comparable_group.",
                "Efficient-TDP is external sidecar evidence and is not in the primary denominator.",
            ],
        },
    )
    pilot_cases = ("NV_NVDLA_partition_m", "ariane136", "hidden5")
    scope_kind = (
        "formal_ten_case"
        if selected_cases == contract.ICCAD24_CASES
        and selected_tables == contract.PRIMARY_TABLE_ORDER
        else (
            "three_case_pilot"
            if selected_cases == pilot_cases
            and selected_tables == contract.PRIMARY_TABLE_ORDER
            else "selected_scope"
        )
    )
    expected_count = sum(
        len(contract.PRIMARY_METHODS[table_id]) for table_id in selected_tables
    ) * len(selected_cases)
    all_pass = all(row.get("status") == "pass" for row in rows)
    completion_audit_path = run_root / "completion_audit.json"
    completion_audit = {
        "artifact": "autodmp_tools_paper_completion_audit",
        "artifact_version": 1,
        "campaign_id": campaign_id,
        "scope_kind": scope_kind,
        "status": "complete_pass" if all_pass else "complete_with_terminal_failures",
        "checks": {
            "selected_matrix_row_count": {
                "status": "pass" if len(rows) == expected_count else "failed",
                "actual": len(rows),
                "expected": expected_count,
            },
            "all_rows_terminal": {
                "status": (
                    "pass"
                    if all(row.get("status") in contract.TERMINAL_STATUSES for row in rows)
                    else "failed"
                )
            },
            "all_rows_pass": {"status": "pass" if all_pass else "failed"},
            "passing_rows_sealed_and_physically_valid": {"status": "pass"},
            "canonical_tables_emitted": {"status": "pass"},
            "formal_120_row_denominator": {
                "status": (
                    "pass"
                    if scope_kind == "formal_ten_case" and len(rows) == 120
                    else "not_applicable"
                )
            },
            "pilot_release_gate": {
                "status": (
                    "pass"
                    if scope_kind == "three_case_pilot" and all_pass
                    else (
                        "failed"
                        if scope_kind == "three_case_pilot"
                        else "not_applicable"
                    )
                )
            },
        },
    }
    _write_json(completion_audit_path, completion_audit)

    table_files = {}
    for artifact_id, paths in artifacts.items():
        table_files[artifact_id] = {
            name: {"path": path, "sha256": _sha256(Path(path))}
            for name, path in paths.items()
        }
    indexed_rows = []
    for row in rows:
        attempt_row_path = (
            run_root
            / "attempts"
            / row["table_id"]
            / row["case"]
            / row["method_id"]
            / row["attempt_id"]
            / "normalized_row.json"
        )
        indexed_rows.append(
            {
                "table_id": row["table_id"],
                "case": row["case"],
                "method_id": row["method_id"],
                "attempt_id": row["attempt_id"],
                "status": row["status"],
                "normalized_row": str(attempt_row_path),
                "normalized_row_sha256": _sha256(attempt_row_path),
                "source_artifact": row.get("source_artifact"),
                "track_a_artifact": row.get("track_a_artifact"),
            }
        )
    artifact_index_path = run_root / "artifact_index.json"
    _write_json(
        artifact_index_path,
        {
            "artifact": "autodmp_tools_paper_artifact_index",
            "artifact_version": 1,
            "campaign_id": campaign_id,
            "campaign_manifest": str(run_root / "campaign_manifest.json"),
            "campaign_manifest_sha256": _sha256(
                run_root / "campaign_manifest.json"
            ),
            "normalized_rows": str(run_root / "normalized_rows.json"),
            "normalized_rows_sha256": _sha256(run_root / "normalized_rows.json"),
            "table_files": table_files,
            "completion_audit": str(completion_audit_path),
            "completion_audit_sha256": _sha256(completion_audit_path),
            "failure_and_limitations": str(failure_report_path),
            "failure_and_limitations_sha256": _sha256(failure_report_path),
            "rows": indexed_rows,
        },
    )
    artifacts["campaign_reports"] = {
        "artifact_index": str(artifact_index_path),
        "completion_audit": str(completion_audit_path),
        "failure_and_limitations": str(failure_report_path),
    }
    _write_json(run_root / "table_artifacts.json", artifacts)
    return artifacts


def _execute_sizing_attempt(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    case: str,
    method_id: str,
    input_metrics: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    attempt_id = _attempt_id(
        run_root,
        campaign_id=identity["campaign_id"],
        table_id="sizing",
        case=case,
        method_id=method_id,
    )
    attempt_root = run_root / "attempts" / "sizing" / case / method_id / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    command, result_path = _sizing_source_command(
        args=args,
        case=case,
        method_id=method_id,
        attempt_root=attempt_root,
    )
    command_digest = contract.digest_payload(command)
    attempt_manifest = {
        "artifact": "autodmp_tools_paper_attempt_manifest",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "release_profile_digest": identity["release_profile_digest"],
        "execution_manifest_digest": identity["execution_manifest_digest"],
        "attempt_id": attempt_id,
        "table_id": "sizing",
        "case": case,
        "method_id": method_id,
        "input_domain": "D_post",
        "required_cuda": bool(profile["method_specs"][method_id]["required_cuda"]),
        "command_sha256": command_digest,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "status": "running",
    }
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    (attempt_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    if attempt_manifest["required_cuda"] and execution["gpu_model"] == "unavailable":
        row = adapters.terminal_failure_row(
            campaign_id=identity["campaign_id"],
            table_id="sizing",
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="D_post",
            required_cuda=True,
            status="cuda_unavailable",
            terminal_reason="physical GPU binding is unavailable",
        )
        process_result = None
    else:
        process_result = _run_attempt_command(
            command,
            stdout_path=attempt_root / "stdout.log",
            stderr_path=attempt_root / "stderr.log",
            environment=environment,
        )
        _write_json(attempt_root / "process_result.json", process_result)
        if process_result["returncode"] != 0:
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="sizing",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="D_post",
                required_cuda=attempt_manifest["required_cuda"],
                status="execution_failed",
                terminal_reason=(
                    f"source runner exited with code {process_result['returncode']}"
                ),
                source_artifact={"expected_result_path": str(result_path)},
            )
        else:
            try:
                row = adapters.normalize_sizing_result(
                    result_path=result_path,
                    campaign_id=identity["campaign_id"],
                    attempt_id=attempt_id,
                    input_metrics=input_metrics,
                    required_cuda=attempt_manifest["required_cuda"],
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                row = adapters.terminal_failure_row(
                    campaign_id=identity["campaign_id"],
                    table_id="sizing",
                    case=case,
                    method_id=method_id,
                    attempt_id=attempt_id,
                    input_domain="D_post",
                    required_cuda=attempt_manifest["required_cuda"],
                    status="artifact_contract_failed",
                    terminal_reason=str(error),
                    source_artifact={"expected_result_path": str(result_path)},
                )

    row = _seal_or_fail_row(
        row,
        campaign_id=identity["campaign_id"],
        table_id="sizing",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="D_post",
        required_cuda=attempt_manifest["required_cuda"],
        source_execution=process_result,
    )
    _write_json(attempt_root / "normalized_row.json", row)
    qualification = {
        "artifact": "autodmp_tools_paper_attempt_qualification",
        "artifact_version": 1,
        "attempt_id": attempt_id,
        "status": "pass" if row["status"] == "pass" else "failed",
        "terminal_status": row["status"],
        "terminal_reason": row.get("terminal_reason"),
    }
    _write_json(attempt_root / "qualification.json", qualification)
    attempt_manifest["status"] = row["status"]
    attempt_manifest["terminal_reason"] = row.get("terminal_reason")
    attempt_manifest["process_result"] = process_result
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    return row


def _execute_placement_attempt(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    case: str,
    method_id: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    attempt_id = _attempt_id(
        run_root,
        campaign_id=identity["campaign_id"],
        table_id="placement",
        case=case,
        method_id=method_id,
    )
    attempt_root = (
        run_root / "attempts" / "placement" / case / method_id / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    command, result_path, raw_def, r0_def = _placement_source_command(
        args=args,
        profile=profile,
        case=case,
        method_id=method_id,
        attempt_root=attempt_root,
    )
    required_cuda = bool(profile["method_specs"][method_id]["required_cuda"])
    attempt_manifest = {
        "artifact": "autodmp_tools_paper_attempt_manifest",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "release_profile_digest": identity["release_profile_digest"],
        "execution_manifest_digest": identity["execution_manifest_digest"],
        "attempt_id": attempt_id,
        "table_id": "placement",
        "case": case,
        "method_id": method_id,
        "input_domain": "R0",
        "required_cuda": required_cuda,
        "command_sha256": contract.digest_payload(command),
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "status": "running",
    }
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    (attempt_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    source_execution = None
    track_a_execution = None
    if required_cuda and execution["gpu_model"] == "unavailable":
        row = adapters.terminal_failure_row(
            campaign_id=identity["campaign_id"],
            table_id="placement",
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="R0",
            required_cuda=True,
            status="cuda_unavailable",
            terminal_reason="physical GPU binding is unavailable",
        )
    else:
        source_execution = _run_attempt_command(
            command,
            stdout_path=attempt_root / "stdout.log",
            stderr_path=attempt_root / "stderr.log",
            environment=environment,
        )
        _write_json(attempt_root / "process_result.json", source_execution)
        if source_execution["returncode"] != 0:
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="placement",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="R0",
                required_cuda=required_cuda,
                status="execution_failed",
                terminal_reason=(
                    f"source runner exited with code {source_execution['returncode']}"
                ),
                source_artifact={"expected_result_path": str(result_path)},
            )
        elif not raw_def.is_file() or not r0_def.is_file():
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="placement",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="R0",
                required_cuda=required_cuda,
                status="artifact_contract_failed",
                terminal_reason="placement source runner did not produce raw DEF and R0",
                source_artifact={"expected_result_path": str(result_path)},
            )
        else:
            track_a_root = attempt_root / "track_a"
            track_a_command = _track_a_command(
                args=args,
                profile=profile,
                case=case,
                method_id=method_id,
                reference_def=r0_def,
                def_input=raw_def,
                output_dir=track_a_root,
            )
            (attempt_root / "track_a_command.txt").write_text(
                shlex.join(track_a_command) + "\n", encoding="utf-8"
            )
            track_a_execution = _run_attempt_command(
                track_a_command,
                stdout_path=attempt_root / "track_a_stdout.log",
                stderr_path=attempt_root / "track_a_stderr.log",
                environment=environment,
            )
            _write_json(attempt_root / "track_a_process_result.json", track_a_execution)
            if track_a_execution["returncode"] != 0:
                row = adapters.terminal_failure_row(
                    campaign_id=identity["campaign_id"],
                    table_id="placement",
                    case=case,
                    method_id=method_id,
                    attempt_id=attempt_id,
                    input_domain="R0",
                    required_cuda=required_cuda,
                    status="evaluation_failed",
                    terminal_reason=(
                        "canonical Track A exited with code "
                        f"{track_a_execution['returncode']}"
                    ),
                    source_artifact={"path": str(result_path)},
                )
            else:
                try:
                    row = adapters.normalize_placement_result(
                        result_path=result_path,
                        track_a_path=track_a_root / "track_a_summary.json",
                        track_a_launcher_runtime_sec=track_a_execution["runtime_sec"],
                        campaign_id=identity["campaign_id"],
                        attempt_id=attempt_id,
                        method_id=method_id,
                        required_cuda=required_cuda,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row = adapters.terminal_failure_row(
                        campaign_id=identity["campaign_id"],
                        table_id="placement",
                        case=case,
                        method_id=method_id,
                        attempt_id=attempt_id,
                        input_domain="R0",
                        required_cuda=required_cuda,
                        status="artifact_contract_failed",
                        terminal_reason=str(error),
                        source_artifact={"path": str(result_path)},
                    )
    row = _seal_or_fail_row(
        row,
        campaign_id=identity["campaign_id"],
        table_id="placement",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="R0",
        required_cuda=required_cuda,
        source_execution=source_execution,
        track_a_execution=track_a_execution,
    )
    row = _record_r0_identity(
        row,
        run_root=run_root,
        campaign_id=identity["campaign_id"],
        table_id="placement",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        required_cuda=required_cuda,
    )
    row = _attach_r0_input_metrics(
        row,
        args=args,
        profile=profile,
        run_root=run_root,
        environment=environment,
        campaign_id=identity["campaign_id"],
        table_id="placement",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        required_cuda=required_cuda,
    )
    _write_json(attempt_root / "normalized_row.json", row)
    _write_json(
        attempt_root / "qualification.json",
        {
            "artifact": "autodmp_tools_paper_attempt_qualification",
            "artifact_version": 1,
            "attempt_id": attempt_id,
            "status": "pass" if row["status"] == "pass" else "failed",
            "terminal_status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
        },
    )
    attempt_manifest["status"] = row["status"]
    attempt_manifest["terminal_reason"] = row.get("terminal_reason")
    attempt_manifest["process_result"] = source_execution
    attempt_manifest["track_a_process_result"] = track_a_execution
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    return row


def _execute_buffering_attempt(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    case: str,
    method_id: str,
    input_metrics: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    attempt_id = _attempt_id(
        run_root,
        campaign_id=identity["campaign_id"],
        table_id="buffering",
        case=case,
        method_id=method_id,
    )
    attempt_root = (
        run_root / "attempts" / "buffering" / case / method_id / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    command, result_path, raw_def = _buffering_source_command(
        args=args,
        case=case,
        method_id=method_id,
        attempt_root=attempt_root,
        campaign_manifest_path=run_root / "campaign_manifest.json",
    )
    required_cuda = bool(profile["method_specs"][method_id]["required_cuda"])
    attempt_manifest = {
        "artifact": "autodmp_tools_paper_attempt_manifest",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "release_profile_digest": identity["release_profile_digest"],
        "execution_manifest_digest": identity["execution_manifest_digest"],
        "attempt_id": attempt_id,
        "table_id": "buffering",
        "case": case,
        "method_id": method_id,
        "input_domain": "D_post",
        "required_cuda": required_cuda,
        "command_sha256": contract.digest_payload(command),
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "status": "running",
    }
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    (attempt_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    source_execution = None
    track_a_execution = None
    if required_cuda and execution["gpu_model"] == "unavailable":
        row = adapters.terminal_failure_row(
            campaign_id=identity["campaign_id"],
            table_id="buffering",
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="D_post",
            required_cuda=True,
            status="cuda_unavailable",
            terminal_reason="physical GPU binding is unavailable",
        )
    else:
        source_execution = _run_attempt_command(
            command,
            stdout_path=attempt_root / "stdout.log",
            stderr_path=attempt_root / "stderr.log",
            environment=environment,
        )
        _write_json(attempt_root / "process_result.json", source_execution)
        if source_execution["returncode"] != 0:
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="buffering",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="D_post",
                required_cuda=required_cuda,
                status="execution_failed",
                terminal_reason=(
                    f"source runner exited with code {source_execution['returncode']}"
                ),
                source_artifact={"expected_result_path": str(result_path)},
            )
        elif not raw_def.is_file():
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="buffering",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="D_post",
                required_cuda=required_cuda,
                status="artifact_contract_failed",
                terminal_reason="buffering source runner did not produce committed raw DEF",
                source_artifact={"expected_result_path": str(result_path)},
            )
        else:
            reference_def = _case_inputs(args.benchmark_root.resolve(), case)["def"]
            track_a_root = attempt_root / "track_a"
            track_a_command = _track_a_command(
                args=args,
                profile=profile,
                case=case,
                method_id=method_id,
                reference_def=reference_def,
                def_input=raw_def,
                output_dir=track_a_root,
            )
            (attempt_root / "track_a_command.txt").write_text(
                shlex.join(track_a_command) + "\n", encoding="utf-8"
            )
            track_a_execution = _run_attempt_command(
                track_a_command,
                stdout_path=attempt_root / "track_a_stdout.log",
                stderr_path=attempt_root / "track_a_stderr.log",
                environment=environment,
            )
            _write_json(attempt_root / "track_a_process_result.json", track_a_execution)
            if track_a_execution["returncode"] != 0:
                row = adapters.terminal_failure_row(
                    campaign_id=identity["campaign_id"],
                    table_id="buffering",
                    case=case,
                    method_id=method_id,
                    attempt_id=attempt_id,
                    input_domain="D_post",
                    required_cuda=required_cuda,
                    status="evaluation_failed",
                    terminal_reason=(
                        "canonical Track A exited with code "
                        f"{track_a_execution['returncode']}"
                    ),
                    source_artifact={"path": str(result_path)},
                )
            else:
                try:
                    if method_id == "autodmp_candidate_route_b":
                        row = adapters.normalize_candidate_buffering_result(
                            manifest_path=result_path,
                            track_a_path=track_a_root / "track_a_summary.json",
                            track_a_launcher_runtime_sec=track_a_execution["runtime_sec"],
                            campaign_id=identity["campaign_id"],
                            attempt_id=attempt_id,
                            case=case,
                            input_metrics=input_metrics,
                        )
                    else:
                        row = adapters.normalize_equal_spaced_buffering_result(
                            result_path=result_path,
                            track_a_path=track_a_root / "track_a_summary.json",
                            track_a_launcher_runtime_sec=track_a_execution["runtime_sec"],
                            campaign_id=identity["campaign_id"],
                            attempt_id=attempt_id,
                            method_id=method_id,
                            input_metrics=input_metrics,
                            required_cuda=required_cuda,
                        )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row = adapters.terminal_failure_row(
                        campaign_id=identity["campaign_id"],
                        table_id="buffering",
                        case=case,
                        method_id=method_id,
                        attempt_id=attempt_id,
                        input_domain="D_post",
                        required_cuda=required_cuda,
                        status="artifact_contract_failed",
                        terminal_reason=str(error),
                        source_artifact={"path": str(result_path)},
                    )
    row = _seal_or_fail_row(
        row,
        campaign_id=identity["campaign_id"],
        table_id="buffering",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="D_post",
        required_cuda=required_cuda,
        source_execution=source_execution,
        track_a_execution=track_a_execution,
    )
    _write_json(attempt_root / "normalized_row.json", row)
    _write_json(
        attempt_root / "qualification.json",
        {
            "artifact": "autodmp_tools_paper_attempt_qualification",
            "artifact_version": 1,
            "attempt_id": attempt_id,
            "status": "pass" if row["status"] == "pass" else "failed",
            "terminal_status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
        },
    )
    attempt_manifest["status"] = row["status"]
    attempt_manifest["terminal_reason"] = row.get("terminal_reason")
    attempt_manifest["process_result"] = source_execution
    attempt_manifest["track_a_process_result"] = track_a_execution
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    return row


def _execute_full_flow_attempt(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    case: str,
    method_id: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    attempt_id = _attempt_id(
        run_root,
        campaign_id=identity["campaign_id"],
        table_id="full_flow",
        case=case,
        method_id=method_id,
    )
    attempt_root = (
        run_root / "attempts" / "full_flow" / case / method_id / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    command, result_path, raw_def, r0_def = _full_flow_source_command(
        args=args,
        profile=profile,
        case=case,
        method_id=method_id,
        attempt_root=attempt_root,
    )
    required_cuda = bool(profile["method_specs"][method_id]["required_cuda"])
    attempt_manifest = {
        "artifact": "autodmp_tools_paper_attempt_manifest",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "release_profile_digest": identity["release_profile_digest"],
        "execution_manifest_digest": identity["execution_manifest_digest"],
        "attempt_id": attempt_id,
        "table_id": "full_flow",
        "case": case,
        "method_id": method_id,
        "input_domain": "R0",
        "required_cuda": required_cuda,
        "command_sha256": contract.digest_payload(command),
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "status": "running",
    }
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    (attempt_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    source_execution = None
    track_a_execution = None
    if required_cuda and execution["gpu_model"] == "unavailable":
        row = adapters.terminal_failure_row(
            campaign_id=identity["campaign_id"],
            table_id="full_flow",
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain="R0",
            required_cuda=True,
            status="cuda_unavailable",
            terminal_reason="physical GPU binding is unavailable",
        )
    else:
        source_execution = _run_attempt_command(
            command,
            stdout_path=attempt_root / "stdout.log",
            stderr_path=attempt_root / "stderr.log",
            environment=environment,
        )
        _write_json(attempt_root / "process_result.json", source_execution)
        if source_execution["returncode"] != 0:
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="full_flow",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="R0",
                required_cuda=required_cuda,
                status="execution_failed",
                terminal_reason=(
                    f"source runner exited with code {source_execution['returncode']}"
                ),
                source_artifact={"expected_result_path": str(result_path)},
            )
        elif not result_path.is_file() or not raw_def.is_file() or not r0_def.is_file():
            row = adapters.terminal_failure_row(
                campaign_id=identity["campaign_id"],
                table_id="full_flow",
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                input_domain="R0",
                required_cuda=required_cuda,
                status="artifact_contract_failed",
                terminal_reason="full-flow source runner did not produce result, raw DEF, and R0",
                source_artifact={"expected_result_path": str(result_path)},
            )
        else:
            track_a_root = attempt_root / "track_a"
            track_a_command = _track_a_command(
                args=args,
                profile=profile,
                case=case,
                method_id=method_id,
                reference_def=r0_def,
                def_input=raw_def,
                output_dir=track_a_root,
            )
            (attempt_root / "track_a_command.txt").write_text(
                shlex.join(track_a_command) + "\n", encoding="utf-8"
            )
            track_a_execution = _run_attempt_command(
                track_a_command,
                stdout_path=attempt_root / "track_a_stdout.log",
                stderr_path=attempt_root / "track_a_stderr.log",
                environment=environment,
            )
            _write_json(
                attempt_root / "track_a_process_result.json",
                track_a_execution,
            )
            if track_a_execution["returncode"] != 0:
                row = adapters.terminal_failure_row(
                    campaign_id=identity["campaign_id"],
                    table_id="full_flow",
                    case=case,
                    method_id=method_id,
                    attempt_id=attempt_id,
                    input_domain="R0",
                    required_cuda=required_cuda,
                    status="evaluation_failed",
                    terminal_reason=(
                        "canonical Track A exited with code "
                        f"{track_a_execution['returncode']}"
                    ),
                    source_artifact={"path": str(result_path)},
                )
            else:
                try:
                    row = adapters.normalize_full_flow_result(
                        result_path=result_path,
                        track_a_path=track_a_root / "track_a_summary.json",
                        track_a_launcher_runtime_sec=track_a_execution["runtime_sec"],
                        campaign_id=identity["campaign_id"],
                        attempt_id=attempt_id,
                        method_id=method_id,
                        required_cuda=required_cuda,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row = adapters.terminal_failure_row(
                        campaign_id=identity["campaign_id"],
                        table_id="full_flow",
                        case=case,
                        method_id=method_id,
                        attempt_id=attempt_id,
                        input_domain="R0",
                        required_cuda=required_cuda,
                        status="artifact_contract_failed",
                        terminal_reason=str(error),
                        source_artifact={"path": str(result_path)},
                    )
    row = _seal_or_fail_row(
        row,
        campaign_id=identity["campaign_id"],
        table_id="full_flow",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        input_domain="R0",
        required_cuda=required_cuda,
        source_execution=source_execution,
        track_a_execution=track_a_execution,
    )
    row = _record_r0_identity(
        row,
        run_root=run_root,
        campaign_id=identity["campaign_id"],
        table_id="full_flow",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        required_cuda=required_cuda,
    )
    row = _attach_r0_input_metrics(
        row,
        args=args,
        profile=profile,
        run_root=run_root,
        environment=environment,
        campaign_id=identity["campaign_id"],
        table_id="full_flow",
        case=case,
        method_id=method_id,
        attempt_id=attempt_id,
        required_cuda=required_cuda,
    )
    _write_json(attempt_root / "normalized_row.json", row)
    _write_json(
        attempt_root / "qualification.json",
        {
            "artifact": "autodmp_tools_paper_attempt_qualification",
            "artifact_version": 1,
            "attempt_id": attempt_id,
            "status": "pass" if row["status"] == "pass" else "failed",
            "terminal_status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
        },
    )
    attempt_manifest["status"] = row["status"]
    attempt_manifest["terminal_reason"] = row.get("terminal_reason")
    attempt_manifest["process_result"] = source_execution
    attempt_manifest["track_a_process_result"] = track_a_execution
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    return row


def _execute_reuse_attempt(
    *,
    entry: dict[str, Any],
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    key: tuple[str, str, str],
    current_dpost_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    table_id, case, method_id = key
    attempt_id = _attempt_id(
        run_root,
        campaign_id=identity["campaign_id"],
        table_id=table_id,
        case=case,
        method_id=method_id,
    )
    attempt_root = run_root / "attempts" / table_id / case / method_id / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    spec = dict(profile["method_specs"][method_id])
    attempt_manifest = {
        "artifact": "autodmp_tools_paper_attempt_manifest",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "release_profile_digest": identity["release_profile_digest"],
        "execution_manifest_digest": identity["execution_manifest_digest"],
        "attempt_id": attempt_id,
        "table_id": table_id,
        "case": case,
        "method_id": method_id,
        "input_domain": spec["input_domain"],
        "required_cuda": bool(spec["required_cuda"]),
        "reuse_entry": entry,
        "status": "running",
    }
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    report_path = attempt_root / "reuse_qualification.json"
    try:
        prior_row, report = artifact_qualification.qualify_reuse_entry(
            entry,
            current_profile=profile,
            current_execution=execution,
            expected_key=key,
            current_dpost_metrics=current_dpost_metrics,
        )
        report.update(
            {
                "campaign_id": identity["campaign_id"],
                "attempt_id": attempt_id,
            }
        )
        _write_json(report_path, report)
        row = json.loads(json.dumps(prior_row))
        original_campaign = row["campaign_id"]
        original_attempt = row["attempt_id"]
        row["campaign_id"] = identity["campaign_id"]
        row["attempt_id"] = attempt_id
        source = dict(row.get("source_artifact") or {})
        source.update(
            {
                "reused_from_campaign": original_campaign,
                "original_attempt_id": original_attempt,
                "reuse_qualification_path": str(report_path),
                "reuse_qualification_sha256": artifact_qualification.sha256(
                    report_path
                ),
            }
        )
        row["source_artifact"] = source
        if row["input_domain"] == "R0":
            row = _record_r0_identity(
                row,
                run_root=run_root,
                campaign_id=identity["campaign_id"],
                table_id=table_id,
                case=case,
                method_id=method_id,
                attempt_id=attempt_id,
                required_cuda=bool(spec["required_cuda"]),
            )
        reused = row.get("status") == "pass"
        if not reused:
            report["status"] = "failed"
            report["terminal_reason"] = row.get("terminal_reason")
            _write_json(report_path, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {
            "artifact": "autodmp_tools_paper_reuse_qualification",
            "artifact_version": 1,
            "campaign_id": identity["campaign_id"],
            "attempt_id": attempt_id,
            "table_id": table_id,
            "case": case,
            "method_id": method_id,
            "status": "failed",
            "terminal_reason": str(error),
        }
        _write_json(report_path, report)
        row = adapters.terminal_failure_row(
            campaign_id=identity["campaign_id"],
            table_id=table_id,
            case=case,
            method_id=method_id,
            attempt_id=attempt_id,
            input_domain=str(spec["input_domain"]),
            required_cuda=bool(spec["required_cuda"]),
            status="provenance_mismatch",
            terminal_reason=str(error),
            source_artifact={
                "reuse_index_path": execution.get("reuse_index_path"),
                "indexed_normalized_row": entry.get("normalized_row"),
            },
        )
        reused = False

    _write_json(attempt_root / "normalized_row.json", row)
    _write_json(
        attempt_root / "qualification.json",
        {
            "artifact": "autodmp_tools_paper_attempt_qualification",
            "artifact_version": 1,
            "attempt_id": attempt_id,
            "status": "pass" if reused else "failed",
            "terminal_status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
            "reuse_qualification_path": str(report_path),
            "reuse_qualification_sha256": artifact_qualification.sha256(
                report_path
            ),
        },
    )
    attempt_manifest["status"] = row["status"]
    attempt_manifest["terminal_reason"] = row.get("terminal_reason")
    attempt_manifest["reuse_qualification_path"] = str(report_path)
    _write_json(attempt_root / "attempt_manifest.json", attempt_manifest)
    return row, reused


def _apply_reuse_index(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    identity: dict[str, str],
    execution: dict[str, Any],
    run_root: Path,
    selected: dict[tuple[str, str, str], dict[str, Any]],
    selected_cases: tuple[str, ...],
    selected_tables: tuple[str, ...],
    environment: dict[str, str],
) -> set[tuple[str, str, str]]:
    if args.reuse_index is None:
        return set()
    entries = artifact_qualification.load_reuse_index(args.reuse_index.resolve())
    selected_keys = {
        (row["table_id"], row["case"], row["method_id"])
        for row in contract.primary_matrix(selected_cases)
        if row["table_id"] in selected_tables
    }
    outside = sorted(set(entries) - selected_keys)
    if outside:
        raise ValueError(f"reuse-index contains rows outside selected matrix: {outside}")
    reused_keys: set[tuple[str, str, str]] = set()
    dpost_metrics: dict[str, dict[str, Any]] = {}
    order = {key: index for index, key in enumerate(sorted(selected_keys))}
    for key, entry in sorted(entries.items(), key=lambda item: order[item[0]]):
        if key in selected:
            if args.resume and selected[key].get("status") == "pass":
                continue
            raise ValueError(f"reuse-index duplicates an existing campaign row: {key}")
        table_id, case, _ = key
        current_metrics = None
        if table_id in {"buffering", "sizing"}:
            if case not in dpost_metrics:
                dpost_metrics[case] = _load_or_run_dpost_input_metrics(
                    args=args,
                    profile=profile,
                    run_root=run_root,
                    case=case,
                    environment=environment,
                )
            current_metrics = dpost_metrics[case]
        row, reused = _execute_reuse_attempt(
            entry=entry,
            profile=profile,
            identity=identity,
            execution=execution,
            run_root=run_root,
            key=key,
            current_dpost_metrics=current_metrics,
        )
        if reused:
            selected[key] = row
            reused_keys.add(key)
            _write_normalized_rows(run_root, list(selected.values()))
        print(
            f"reuse/{table_id}/{case}/{key[2]}: "
            + ("pass" if reused else f"{row['status']} (fresh run scheduled)")
        )
    return reused_keys


def build_execution_manifest(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    openroad = args.openroad_bin.resolve()
    python_bin = args.python_bin.resolve()
    if not args.plan_only:
        for path in (openroad, python_bin, args.benchmark_root.resolve()):
            if not path.exists():
                raise FileNotFoundError(path)
    gpu_uuid = _gpu_uuid(args.gpu_id)
    gpu_processes = _gpu_compute_processes(gpu_uuid)
    if not args.plan_only and not args.aggregate_only and gpu_processes:
        raise RuntimeError(
            f"physical GPU {args.gpu_id} is not exclusive: {gpu_processes}"
        )
    return {
        "artifact": contract.EXECUTION_ARTIFACT,
        "artifact_version": contract.EXECUTION_VERSION,
        "release_profile_digest": contract.digest_payload(profile),
        "selected_cases": list(contract.selected_cases(args.cases)),
        "selected_tables": list(args.tables or contract.PRIMARY_TABLE_ORDER),
        "autodmp_revision": _git_head(AUTODMP_ROOT),
        "autodmp_tracked_diff_sha256": _git_tracked_diff_sha256(AUTODMP_ROOT),
        "source_file_sha256": _source_file_hashes(),
        "native_library_sha256": _native_library_hashes(),
        "openroad_path": str(openroad),
        "openroad_sha256": _sha256(openroad) if openroad.is_file() else "plan_only",
        "openroad_version": _tool_output([str(openroad), "-version"], AUTODMP_ROOT),
        "python_path": str(python_bin),
        "python_version": _tool_output([str(python_bin), "--version"], AUTODMP_ROOT),
        "torch_version": _tool_output(
            [str(python_bin), "-c", "import torch; print(torch.__version__)"],
            AUTODMP_ROOT,
        ),
        "cuda_runtime_version": _tool_output(
            [str(python_bin), "-c", "import torch; print(torch.version.cuda)"],
            AUTODMP_ROOT,
        ),
        "cuda_visible_devices": str(args.gpu_id),
        "logical_gpu_index": 0,
        "gpu_model": _gpu_model(args.gpu_id),
        "gpu_uuid": gpu_uuid,
        "gpu_preflight_compute_processes": gpu_processes,
        "cpu_model": _cpu_model(),
        "cpu_count": int(os.cpu_count() or 1),
        "omp_num_threads": int(args.cpu_threads),
        "torch_num_threads": int(args.cpu_threads),
        "torch_num_interop_threads": 1,
        "openroad_num_threads": int(args.cpu_threads),
        "case_concurrency": 1,
        "measurement_policy": "cold_process_per_attempt",
        "benchmark_root": str(args.benchmark_root.resolve()),
        "reuse_index_path": (
            str(args.reuse_index.resolve()) if args.reuse_index is not None else None
        ),
        "reuse_index_sha256": (
            _sha256(args.reuse_index.resolve())
            if args.reuse_index is not None
            else None
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument(
        "--table",
        dest="tables",
        action="append",
        choices=contract.PRIMARY_TABLE_ORDER,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-index", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    profile = contract.load_release_profile(args.profile.resolve())
    execution = build_execution_manifest(args, profile)
    identity = contract.campaign_identity(profile, execution)
    run_root = args.output_root.resolve() / identity["campaign_id"]
    manifest_path = run_root / "campaign_manifest.json"
    manifest = {
        "artifact": "autodmp_tools_paper_campaign_manifest",
        "artifact_version": 1,
        **identity,
        "release_profile_path": str(args.profile.resolve()),
        "release_profile": profile,
        "execution_manifest": execution,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("campaign ID collision or manifest drift")
    else:
        _write_json(manifest_path, manifest)

    selected_tables = tuple(args.tables or contract.PRIMARY_TABLE_ORDER)
    selected_case_ids = tuple(execution["selected_cases"])
    matrix = [
        row
        for row in contract.primary_matrix(selected_case_ids)
        if row["table_id"] in selected_tables
    ]
    plan_rows = []
    for row in matrix:
        planned = dict(row)
        if row["table_id"] == "sizing":
            planned_root = (
                run_root
                / "attempts"
                / "sizing"
                / row["case"]
                / row["method_id"]
                / "<attempt_id>"
            )
            command, result_path = _sizing_source_command(
                args=args,
                case=row["case"],
                method_id=row["method_id"],
                attempt_root=planned_root,
            )
            planned["command"] = command
            planned["expected_result_path"] = str(result_path)
        elif row["table_id"] == "placement":
            planned_root = (
                run_root
                / "attempts"
                / "placement"
                / row["case"]
                / row["method_id"]
                / "<attempt_id>"
            )
            command, result_path, raw_def, r0_def = _placement_source_command(
                args=args,
                profile=profile,
                case=row["case"],
                method_id=row["method_id"],
                attempt_root=planned_root,
            )
            planned["command"] = command
            planned["expected_result_path"] = str(result_path)
            planned["expected_raw_def"] = str(raw_def)
            planned["expected_r0_def"] = str(r0_def)
        elif row["table_id"] == "buffering":
            planned_root = (
                run_root
                / "attempts"
                / "buffering"
                / row["case"]
                / row["method_id"]
                / "<attempt_id>"
            )
            command, result_path, raw_def = _buffering_source_command(
                args=args,
                case=row["case"],
                method_id=row["method_id"],
                attempt_root=planned_root,
                campaign_manifest_path=manifest_path,
            )
            planned["command"] = command
            planned["expected_result_path"] = str(result_path)
            planned["expected_raw_def"] = str(raw_def)
        elif row["table_id"] == "full_flow":
            planned_root = (
                run_root
                / "attempts"
                / "full_flow"
                / row["case"]
                / row["method_id"]
                / "<attempt_id>"
            )
            command, result_path, raw_def, r0_def = _full_flow_source_command(
                args=args,
                profile=profile,
                case=row["case"],
                method_id=row["method_id"],
                attempt_root=planned_root,
            )
            planned["command"] = command
            planned["expected_result_path"] = str(result_path)
            planned["expected_raw_def"] = str(raw_def)
            planned["expected_r0_def"] = str(r0_def)
        else:
            raise AssertionError(row["table_id"])
        plan_rows.append(planned)
    command_plan = {
        "artifact": "autodmp_tools_paper_command_plan",
        "artifact_version": 1,
        "campaign_id": identity["campaign_id"],
        "status": (
            "plan_only"
            if args.plan_only
            else (
                "executable"
                if set(selected_tables) <= set(contract.PRIMARY_TABLE_ORDER)
                else "adapters_pending"
            )
        ),
        "rows": plan_rows,
    }
    _write_json(run_root / "command_plan.json", command_plan)

    if args.aggregate_only:
        normalized_path = run_root / "normalized_rows.json"
        if not normalized_path.is_file():
            raise FileNotFoundError(normalized_path)
        rows = _load_normalized_rows(run_root)
        _write_final_tables(
            run_root=run_root,
            rows=rows,
            profile=profile,
            campaign_id=identity["campaign_id"],
            selected_cases=selected_case_ids,
            selected_tables=selected_tables,
        )
        print(f"run_root: {run_root}")
        return 0
    if args.plan_only:
        print(f"campaign_id: {identity['campaign_id']}")
        print(f"run_root: {run_root}")
        print(f"planned_rows: {len(matrix)}")
        return 0
    unsupported = sorted(set(selected_tables) - set(contract.PRIMARY_TABLE_ORDER))
    if unsupported:
        raise RuntimeError(
            "execution adapters are not complete for table(s): "
            + ", ".join(unsupported)
        )

    environment = _execution_environment(execution)
    existing_rows = _load_normalized_rows(run_root)
    _validate_selected_rows(
        existing_rows,
        campaign_id=identity["campaign_id"],
        selected_cases=selected_case_ids,
        selected_tables=selected_tables,
        require_complete=False,
    )
    if existing_rows and not args.resume:
        raise RuntimeError(
            "campaign already has terminal attempts; use --resume to append retries"
        )
    selected: dict[tuple[str, str, str], dict[str, Any]] = {
        (row["table_id"], row["case"], row["method_id"]): row
        for row in existing_rows
    }
    reused_keys = _apply_reuse_index(
        args=args,
        profile=profile,
        identity=identity,
        execution=execution,
        run_root=run_root,
        selected=selected,
        selected_cases=selected_case_ids,
        selected_tables=selected_tables,
        environment=environment,
    )
    for case in selected_case_ids:
        if "placement" in selected_tables:
            for method_id in contract.PRIMARY_METHODS["placement"]:
                key = ("placement", case, method_id)
                previous = selected.get(key)
                if (
                    previous
                    and previous.get("status") == "pass"
                    and (args.resume or key in reused_keys)
                ):
                    continue
                row = _execute_placement_attempt(
                    args=args,
                    profile=profile,
                    identity=identity,
                    execution=execution,
                    run_root=run_root,
                    case=case,
                    method_id=method_id,
                    environment=environment,
                )
                selected[key] = row
                _write_normalized_rows(run_root, list(selected.values()))
                print(
                    f"placement/{case}/{method_id}: {row['status']}"
                    + (
                        ""
                        if not row.get("terminal_reason")
                        else f" ({row['terminal_reason']})"
                    )
                )
        if "sizing" in selected_tables:
            input_metrics = _load_or_run_dpost_input_metrics(
                args=args,
                profile=profile,
                run_root=run_root,
                case=case,
                environment=environment,
            )
            for method_id in contract.PRIMARY_METHODS["sizing"]:
                key = ("sizing", case, method_id)
                previous = selected.get(key)
                if (
                    previous
                    and previous.get("status") == "pass"
                    and (args.resume or key in reused_keys)
                ):
                    continue
                row = _execute_sizing_attempt(
                    args=args,
                    profile=profile,
                    identity=identity,
                    execution=execution,
                    run_root=run_root,
                    case=case,
                    method_id=method_id,
                    input_metrics=input_metrics,
                    environment=environment,
                )
                selected[key] = row
                _write_normalized_rows(run_root, list(selected.values()))
                print(
                    f"sizing/{case}/{method_id}: {row['status']}"
                    + (
                        ""
                        if not row.get("terminal_reason")
                        else f" ({row['terminal_reason']})"
                    )
                )
        if "buffering" in selected_tables:
            input_metrics = _load_or_run_dpost_input_metrics(
                args=args,
                profile=profile,
                run_root=run_root,
                case=case,
                environment=environment,
            )
            for method_id in contract.PRIMARY_METHODS["buffering"]:
                key = ("buffering", case, method_id)
                previous = selected.get(key)
                if (
                    previous
                    and previous.get("status") == "pass"
                    and (args.resume or key in reused_keys)
                ):
                    continue
                row = _execute_buffering_attempt(
                    args=args,
                    profile=profile,
                    identity=identity,
                    execution=execution,
                    run_root=run_root,
                    case=case,
                    method_id=method_id,
                    input_metrics=input_metrics,
                    environment=environment,
                )
                selected[key] = row
                _write_normalized_rows(run_root, list(selected.values()))
                print(
                    f"buffering/{case}/{method_id}: {row['status']}"
                    + (
                        ""
                        if not row.get("terminal_reason")
                        else f" ({row['terminal_reason']})"
                    )
                )
        if "full_flow" in selected_tables:
            for method_id in contract.PRIMARY_METHODS["full_flow"]:
                key = ("full_flow", case, method_id)
                previous = selected.get(key)
                if (
                    previous
                    and previous.get("status") == "pass"
                    and (args.resume or key in reused_keys)
                ):
                    continue
                row = _execute_full_flow_attempt(
                    args=args,
                    profile=profile,
                    identity=identity,
                    execution=execution,
                    run_root=run_root,
                    case=case,
                    method_id=method_id,
                    environment=environment,
                )
                selected[key] = row
                _write_normalized_rows(run_root, list(selected.values()))
                print(
                    f"full_flow/{case}/{method_id}: {row['status']}"
                    + (
                        ""
                        if not row.get("terminal_reason")
                        else f" ({row['terminal_reason']})"
                    )
                )

    normalized_rows = list(selected.values())
    _write_normalized_rows(run_root, normalized_rows)
    _write_final_tables(
        run_root=run_root,
        rows=normalized_rows,
        profile=profile,
        campaign_id=identity["campaign_id"],
        selected_cases=selected_case_ids,
        selected_tables=selected_tables,
    )
    print(f"campaign_id: {identity['campaign_id']}")
    print(f"run_root: {run_root}")
    return 0 if all(row.get("status") == "pass" for row in normalized_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
