#!/usr/bin/env python3
"""Run the frozen five-round Route-B runtime profile without physical commit."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from run import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PYTHON,
    _case_inputs,
    _load_profile,
    _write_json,
    build_ours_command,
)


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
ROUTE_B_PROFILE_PATH = SCRIPT_DIR / "params" / "route_b_allcases_0p1pct.json"
if str(AUTODMP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTODMP_ROOT))


def _find_summary(result_dir, case):
    direct = result_dir / f"{case}_buffering_inner_loop_summary.json"
    if direct.is_file():
        return direct
    matches = sorted(result_dir.rglob(f"{case}_buffering_inner_loop_summary.json"))
    return matches[-1] if matches else None


def _run(command, *, cwd, log_path, environment):
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "returncode": int(completed.returncode),
        "wall_sec": float(time.perf_counter() - started_at),
    }


def _write_runtime_params(source_path, destination_path, *, detailed_timing_profile=False):
    params = json.loads(source_path.read_text(encoding="utf-8"))
    overrides = {
        "buffering_runtime_profile": True,
        "timing_obj_profile": bool(detailed_timing_profile),
        "timing_propagation_profile": bool(detailed_timing_profile),
        "buffering_dynamic_provider_profile": bool(detailed_timing_profile),
        "full_step_profile": False,
        "timing_artifact_write_interval": 0,
        "pin_violation_detail_interval": 0,
    }
    params.update(overrides)
    _write_json(destination_path, params)
    return overrides


def _valid_round_execution(inner_loop, requested_rounds):
    iterations = int(inner_loop.get("iterations", -1))
    requested_rounds = int(requested_rounds)
    if iterations == requested_rounds:
        return True
    return (
        0 < iterations < requested_rounds
        and str(inner_loop.get("terminal_reason"))
        in {"no_positive_action", "no_improving_prefix"}
    )


def _run_injected_python_reference(placer_args):
    from dreamplace.ops.buffer_insertion import buffering_state_builder

    original = buffering_state_builder._native_segment_builder_fallback_reason
    buffering_state_builder._native_segment_builder_fallback_reason = (
        lambda _config, _params: "runtime_gate_forced_python_reference"
    )
    try:
        from dreamplace.Placer import main as placer_main

        return int(placer_main(placer_args) or 0)
    finally:
        buffering_state_builder._native_segment_builder_fallback_reason = original


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--profile", type=Path, default=ROUTE_B_PROFILE_PATH)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("route_b_runtime_%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--state-builder-backend",
        choices=("auto", "python"),
        default="auto",
        help="Regression-only Python-reference injection; not a production flow option.",
    )
    parser.add_argument(
        "--detailed-timing-profile",
        action="store_true",
        help=(
            "Enable synchronized timing-propagation and dynamic-provider profiling; "
            "intended for attribution, not benchmark timing."
        ),
    )
    parser.add_argument("--injected-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("placer_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.injected_run:
        placer_args = list(args.placer_args)
        if placer_args and placer_args[0] == "--":
            placer_args = placer_args[1:]
        return _run_injected_python_reference(placer_args)
    profile = _load_profile(args.profile.resolve())
    if args.case not in profile["suite"]:
        raise ValueError(f"unsupported profile case: {args.case}")
    inputs = _case_inputs(args.benchmark_root.resolve(), args.case)
    run_root = args.output_root.resolve() / args.run_id / args.case
    result_dir = run_root / "autodmp_result"
    run_root.mkdir(parents=True, exist_ok=True)
    params_path = run_root / "runtime_param.json"
    param_overrides = _write_runtime_params(
        inputs["params_json"],
        params_path,
        detailed_timing_profile=args.detailed_timing_profile,
    )
    command = build_ours_command(
        python_bin=args.python_bin.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        case=args.case,
        result_dir=result_dir,
        committed_def=run_root / "must_not_exist.def",
        profile=profile,
        segment_strategy="discrete_net_gradient",
    )
    command[2] = str(params_path)
    command.extend(["--buffering-commit-enabled", "0"])
    if args.state_builder_backend == "python":
        command = [
            str(args.python_bin.resolve()),
            str(Path(__file__).resolve()),
            "--case",
            args.case,
            "--injected-run",
            "--",
            *command[2:],
        ]
    (run_root / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=run_root / "placer.log",
        environment=dict(os.environ),
    )
    summary_path = _find_summary(result_dir, args.case)
    summary = {}
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "summary", {}
        )
    inner_loop = dict(summary.get("inner_loop") or {})
    runtime_profile = dict(summary.get("runtime_profile") or {})
    state_build_metadata = dict(summary.get("state_build_metadata") or {})
    failures = []
    if execution["returncode"] != 0:
        failures.append("placer_failed")
    if summary.get("strategy") != "discrete_net_gradient":
        failures.append("strategy_mismatch")
    if not _valid_round_execution(
        inner_loop,
        profile["segment_mode"]["continuous_steps"],
    ):
        failures.append("round_count_mismatch")
    if not runtime_profile:
        failures.append("missing_runtime_profile")
    if dict(summary.get("commit") or {}).get("status") != "disabled":
        failures.append("physical_commit_not_disabled")
    backend_used = state_build_metadata.get("segment_state_builder_backend_used")
    expected_backend = (
        "native_cpp" if args.state_builder_backend == "auto" else "python"
    )
    if backend_used != expected_backend:
        failures.append("state_builder_backend_mismatch")
    result = {
        "status": "pass" if not failures else "failed",
        "failures": failures,
        "execution": execution,
        "summary_path": "" if summary_path is None else str(summary_path),
        "runtime_profile": runtime_profile,
        "state_builder_backend_requested": args.state_builder_backend,
        "state_builder_backend_used": backend_used,
        "state_builder_fallback_reason": state_build_metadata.get(
            "segment_state_builder_fallback_reason"
        ),
        "inner_loop": {
            "iterations": inner_loop.get("iterations"),
            "terminal_reason": inner_loop.get("terminal_reason"),
            "terminal_metrics": inner_loop.get("terminal_metrics"),
            "trace_path": inner_loop.get("trace_path"),
        },
        "param_overrides": param_overrides,
    }
    _write_json(run_root / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
