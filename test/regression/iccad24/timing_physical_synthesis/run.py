#!/usr/bin/env python3
"""Prepare and validate the v2 timing physical synthesis campaign inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_PROFILE = SCRIPT_DIR / "profiles" / "timing_physical_synthesis_v3.json"
DEFAULT_OUTPUT_ROOT = (
    AUTODMP_ROOT / "logs" / "regression" / "iccad24" / "timing_physical_synthesis"
)
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")
DEFAULT_OPENROAD = Path("/home/zhaoxueyan/code/OpenROAD/build/bin/openroad")
COMMON_EVALUATOR = AUTODMP_ROOT / "test" / "regression" / "common" / "def_only_track_a.py"
INPUT_DOMAINS = ("R0", "D_post")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import campaign_protocol as protocol


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_input_domains(values: list[str] | None) -> tuple[str, ...]:
    requested = set(values or INPUT_DOMAINS)
    return tuple(name for name in INPUT_DOMAINS if name in requested)


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
    status = "execution_failed"
    returncode = None
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                timeout=timeout_sec,
            )
            returncode = int(completed.returncode)
            status = "pass" if returncode == 0 else "execution_failed"
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "execution_failed"
    return {
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
        "command": command,
    }


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
        }
    )
    return environment


def _python_version(python_bin: Path) -> str:
    try:
        completed = subprocess.run(
            [str(python_bin), "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (completed.stdout or completed.stderr).strip() or "unavailable"


def _technology_args(benchmark_root: Path) -> list[str]:
    technology = protocol.technology_paths(benchmark_root)
    command = ["--tech-lef", str(technology["tech_lef"])]
    for path in technology["lefs"]:
        command.extend(("--lef", str(path)))
    for path in technology["libs"]:
        command.extend(("--lib", str(path)))
    command.extend(("--rc-tcl", str(technology["rc_tcl"])))
    return command


def _evaluator_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    method_id: str,
    reference_def: Path,
    def_input: Path,
    output_dir: Path,
    evaluation_mode: str,
) -> list[str]:
    paths = protocol.case_paths(benchmark_root, case)
    command = [
        str(args.python_bin.resolve()),
        str(COMMON_EVALUATOR),
        "--case",
        case,
        "--method-id",
        method_id,
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(args.cpu_threads),
        "--reference-def",
        str(reference_def),
        "--def-input",
        str(def_input),
        *_technology_args(benchmark_root),
        "--sdc",
        str(paths["sdc"]),
        "--buffer-master",
        str(args.buffer_master),
        "--output-dir",
        str(output_dir),
        "--evaluation-mode",
        evaluation_mode,
    ]
    return command


def _prepare_dpost(
    *,
    benchmark_root: Path,
    case: str,
    run_root: Path,
) -> dict[str, Any]:
    source = protocol.case_paths(benchmark_root, case)["def"]
    output = run_root / "initial_state" / "D_post" / case / f"{case}.def"
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = protocol.sha256_file(source)
    if output.is_file() and protocol.sha256_file(output) != source_hash:
        raise ValueError(f"D_post output differs from packaged source: {output}")
    if not output.is_file():
        shutil.copyfile(source, output)
    return {
        "producer": "packaged_iccad24_def_passthrough",
        "source": str(source),
        "source_sha256": source_hash,
        "normalized_path": str(output),
        "normalized_sha256": protocol.sha256_file(output),
        "normalization": "none",
        "coordinates_fixed": True,
    }


def _prepare_r0(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    run_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    paths = protocol.case_paths(benchmark_root, case)
    r0_root = run_root / "initial_state" / "R0" / case
    r0_def = r0_root / "R0.def"
    r0_manifest_path = r0_root / "R0_manifest.json"
    if r0_def.is_file() and r0_manifest_path.is_file():
        manifest = _read_json(r0_manifest_path)
        if manifest.get("status") == "pass":
            if manifest.get("r0_def_sha256") == protocol.sha256_file(r0_def):
                return {
                    "status": "pass",
                    "reused": True,
                    "r0_def": str(r0_def),
                    "r0_manifest": str(r0_manifest_path),
                    "r0_def_sha256": protocol.sha256_file(r0_def),
                }
        raise ValueError(f"existing R0 artifact is not reusable: {r0_root}")

    params_json = benchmark_root / "design" / case / "workspace" / "config" / "dreamplace_config" / "param.json"
    workspace = benchmark_root / "design" / case / "workspace"
    command = [
        str(args.python_bin.resolve()),
        str(SCRIPT_DIR.parent / "joint_place_sizing_buffering" / "random_init_snapshot.py"),
        "--case",
        case,
        "--params-json",
        str(params_json),
        "--workspace",
        str(workspace),
        "--source-def",
        str(paths["def"]),
        "--verilog",
        str(paths["verilog"]),
        "--sdc",
        str(paths["sdc"]),
        *_technology_args(benchmark_root),
        "--seed",
        str(int(args.seed)),
        "--result-dir",
        str(r0_root / "result"),
        "--output-def",
        str(r0_def),
        "--manifest",
        str(r0_manifest_path),
    ]
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=r0_root / "run.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if execution["status"] != "pass" or not r0_manifest_path.is_file() or not r0_def.is_file():
        return {
            "status": "execution_failed",
            "reused": False,
            "execution": execution,
            "r0_def": str(r0_def),
            "r0_manifest": str(r0_manifest_path),
        }
    manifest = _read_json(r0_manifest_path)
    if manifest.get("status") != "pass":
        return {
            "status": "artifact_contract_failed",
            "reused": False,
            "execution": execution,
            "r0_def": str(r0_def),
            "r0_manifest": str(r0_manifest_path),
            "reason": "R0 producer returned a failed manifest",
        }
    return {
        "status": "pass",
        "reused": False,
        "execution": execution,
        "r0_def": str(r0_def),
        "r0_manifest": str(r0_manifest_path),
        "r0_def_sha256": protocol.sha256_file(r0_def),
    }


def _evaluate_initial_state(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    state_name: str,
    state_def: Path,
    evaluation_mode: str,
    run_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    output_dir = run_root / "initial_state" / state_name / case / "evaluation"
    summary_name = {
        "R0": "input_state_summary.json",
        "D_post": "fixed_state_summary.json",
    }[state_name]
    summary_path = output_dir / summary_name
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("status") == "pass":
            return {
                "status": "pass",
                "reused": True,
                "summary": str(summary_path),
                "metrics": summary.get("metrics", {}),
            }
        raise ValueError(f"existing {state_name} evaluation is not reusable: {output_dir}")
    command = _evaluator_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        method_id=f"input_{state_name}",
        reference_def=state_def,
        def_input=state_def,
        output_dir=output_dir,
        evaluation_mode=evaluation_mode,
    )
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=output_dir / "launcher.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if execution["status"] != "pass" or not summary_path.is_file():
        return {
            "status": "evaluation_failed",
            "reused": False,
            "execution": execution,
            "summary": str(summary_path),
        }
    summary = _read_json(summary_path)
    return {
        "status": summary.get("status"),
        "reused": False,
        "execution": execution,
        "summary": str(summary_path),
        "metrics": summary.get("metrics", {}),
        "fixed_state_audit": summary.get("fixed_state_audit"),
    }


def _write_protocol_artifacts(
    *,
    run_root: Path,
    campaign_manifest: dict[str, Any],
    dpost_manifest: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    _write_json(run_root / "manifest.json", campaign_manifest)
    _write_json(run_root / "dpost_manifest.json", dpost_manifest)
    _write_json(
        run_root / "budget_manifest.json",
        {
            "artifact": "autodmp_timing_physical_synthesis_budget_manifest",
            "artifact_version": 1,
            "campaign_id": campaign_manifest["campaign_id"],
            "budget": profile["budget"],
        },
    )
    _write_json(
        run_root / "openroad_protocol.json",
        {
            "artifact": "autodmp_timing_physical_synthesis_openroad_protocol",
            "artifact_version": 1,
            "campaign_id": campaign_manifest["campaign_id"],
            "evaluators": profile["evaluators"],
            "openroad": profile["openroad"],
            "track_a": profile.get("track_a", {}),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--benchmark-root", type=Path, default=protocol.DEFAULT_BENCHMARK_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--buffer-master", default="BUFx4f_ASAP7_75t_R")
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument(
        "--input-domain",
        dest="input_domains",
        action="append",
        choices=INPUT_DOMAINS,
        help="Prepare only the selected input domain; repeat to select both.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cpu_threads <= 0 or args.timeout_sec <= 0:
        raise ValueError("cpu_threads and timeout_sec must be positive")
    profile = protocol.load_profile(args.profile.resolve())
    selected = tuple(args.cases or protocol.CANONICAL_CASES)
    input_domains = _selected_input_domains(args.input_domains)
    benchmark_root = Path(args.benchmark_root)
    benchmark_manifest = protocol.validate_benchmark_inputs(benchmark_root, selected)
    source_identity = protocol.build_source_identity(AUTODMP_ROOT)
    dpost_manifest = protocol.build_dpost_manifest(
        benchmark_manifest,
        Path(args.output_root) / "pending" / "initial_state" / "D_post",
    )
    campaign_manifest = protocol.build_campaign_manifest(
        profile=profile,
        benchmark_manifest=benchmark_manifest,
        profile_path=args.profile,
        source_identity=source_identity,
    )
    run_root = Path(args.output_root).resolve() / campaign_manifest["campaign_id"]
    dpost_manifest = protocol.build_dpost_manifest(
        benchmark_manifest,
        run_root / "initial_state" / "D_post",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    _write_protocol_artifacts(
        run_root=run_root,
        campaign_manifest=campaign_manifest,
        dpost_manifest=dpost_manifest,
        profile=profile,
    )
    _write_json(
        run_root / "effective_config.json",
        {
            "profile": str(args.profile.resolve()),
            "benchmark_root": str(Path(args.benchmark_root).absolute()),
            "benchmark_root_resolved": str(Path(args.benchmark_root).resolve()),
            "python_bin": str(args.python_bin.resolve()),
            "openroad_bin": str(args.openroad_bin.resolve()),
            "gpu_id": args.gpu_id,
            "cpu_threads": args.cpu_threads,
            "seed": args.seed,
            "buffer_master": args.buffer_master,
            "timeout_sec": args.timeout_sec,
            "source_identity_digest": source_identity["identity_digest"],
            "prepare_inputs": args.prepare_inputs,
            "input_domains": list(input_domains),
        },
    )
    if args.plan_only or not args.prepare_inputs:
        print(f"campaign_id: {campaign_manifest['campaign_id']}")
        print(f"run_root: {run_root}")
        print(f"cases: {len(selected)}")
        print("status: plan_only" if args.plan_only else "status: manifest_ready")
        return 0

    environment = _environment(args)
    results = {}
    failures = []
    for case in selected:
        if "D_post" in input_domains:
            dpost = _prepare_dpost(
                benchmark_root=benchmark_root,
                case=case,
                run_root=run_root,
            )
            dpost_eval = _evaluate_initial_state(
                args=args,
                benchmark_root=benchmark_root,
                case=case,
                state_name="D_post",
                state_def=Path(dpost["normalized_path"]),
                evaluation_mode="fixed_state",
                run_root=run_root,
                environment=environment,
            )
        else:
            dpost = {"status": "not_requested"}
            dpost_eval = {"status": "not_requested"}

        if "R0" in input_domains:
            r0 = _prepare_r0(
                args=args,
                benchmark_root=benchmark_root,
                case=case,
                run_root=run_root,
                environment=environment,
            )
            if r0["status"] == "pass":
                r0_eval = _evaluate_initial_state(
                    args=args,
                    benchmark_root=benchmark_root,
                    case=case,
                    state_name="R0",
                    state_def=Path(r0["r0_def"]),
                    evaluation_mode="input_state",
                    run_root=run_root,
                    environment=environment,
                )
            else:
                r0_eval = {
                    "status": "not_run",
                    "reason": "R0 preparation failed",
                }
        else:
            r0 = {"status": "not_requested"}
            r0_eval = {"status": "not_requested"}

        result = {
            "case": case,
            "D_post": {**dpost, "evaluation": dpost_eval},
            "R0": {**r0, "evaluation": r0_eval},
        }
        results[case] = result
        case_failed = (
            "R0" in input_domains
            and (r0["status"] != "pass" or r0_eval.get("status") != "pass")
        ) or (
            "D_post" in input_domains and dpost_eval.get("status") != "pass"
        )
        if case_failed:
            failures.append(case)
        _write_json(run_root / "initial_state" / case / "preparation.json", result)
        print(
            f"{case}: R0={r0.get('status')}/{r0_eval.get('status')} "
            f"D_post={dpost_eval.get('status')}"
        )
    _write_json(
        run_root / "initial_state" / "preparation_summary.json",
        {
            "artifact": "autodmp_timing_physical_synthesis_initial_state_summary",
            "artifact_version": 1,
            "campaign_id": campaign_manifest["campaign_id"],
            "status": "pass" if not failures else "failed",
            "requested_input_domains": list(input_domains),
            "cases": results,
            "failed_cases": failures,
        },
    )
    print(f"campaign_id: {campaign_manifest['campaign_id']}")
    print(f"run_root: {run_root}")
    print(f"status: {'pass' if not failures else 'failed'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
