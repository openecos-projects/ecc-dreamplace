#!/usr/bin/env python3
"""Measure the Route-B native topology packer before production integration."""

from __future__ import annotations

import argparse
import hashlib
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
    _load_profile,
    _write_json,
    build_ours_command,
)


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
ROUTE_B_PROFILE_PATH = SCRIPT_DIR / "params" / "route_b_allcases_0p1pct.json"
if str(AUTODMP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTODMP_ROOT))


class _ProbeComplete(Exception):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_contract(tensor):
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def _run_injected_probe(artifact_path, placer_args):
    from dreamplace.ops.buffer_insertion import buffering_state_builder
    from dreamplace.ops.net_subgraph_timing import net_subgraph_timing_cpp

    original_builder = buffering_state_builder.build_buffering_state_for_model

    def probe_builder(model, params, config):
        result = buffering_state_builder.build_native_segment_count_packing_for_model(
            model,
            params,
            max_repeater_count=int(config.max_repeaters_per_segment),
        )
        artifact = {
            "artifact": "route_b_native_packer_direct_probe",
            "artifact_version": 1,
            "status": result.get("status"),
            "reason": result.get("reason"),
            "metadata": dict(result.get("metadata") or {}),
            "native_module": str(net_subgraph_timing_cpp.__file__),
            "native_module_sha256": _sha256(net_subgraph_timing_cpp.__file__),
        }
        packed = result.get("packed")
        if packed is not None:
            artifact["prepared_timing_inputs"] = {
                name: _tensor_contract(tensor)
                for name, tensor in packed["prepared_timing_inputs"].items()
            }
            artifact["packed_segment_geometry"] = {
                name: _tensor_contract(tensor)
                for name, tensor in packed["packed_segment_geometry"].items()
            }
        _write_json(Path(artifact_path), artifact)
        raise _ProbeComplete

    buffering_state_builder.build_buffering_state_for_model = probe_builder
    try:
        from dreamplace.Placer import main as placer_main

        placer_main(placer_args)
    except _ProbeComplete:
        return 0
    finally:
        buffering_state_builder.build_buffering_state_for_model = original_builder
    raise RuntimeError("native packer probe injection point was not reached")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--profile", type=Path, default=ROUTE_B_PROFILE_PATH)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        default=time.strftime("route_b_native_packer_direct_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--early-gate-sec", type=float, default=10.0)
    parser.add_argument("--injected-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("placer_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.injected_run:
        placer_args = list(args.placer_args)
        if placer_args and placer_args[0] == "--":
            placer_args = placer_args[1:]
        if args.artifact is None:
            raise ValueError("injected native packer probe requires --artifact")
        return _run_injected_probe(args.artifact.resolve(), placer_args)

    profile = _load_profile(args.profile.resolve())
    if args.case not in profile["suite"]:
        raise ValueError(f"unsupported profile case: {args.case}")
    run_root = args.output_root.resolve() / args.run_id / args.case
    result_dir = run_root / "autodmp_result"
    artifact_path = run_root / "native_packer_probe.json"
    run_root.mkdir(parents=True, exist_ok=True)
    placer_command = build_ours_command(
        python_bin=args.python_bin.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        case=args.case,
        result_dir=result_dir,
        committed_def=run_root / "must_not_exist.def",
        profile=profile,
        segment_strategy="discrete_net_gradient",
    )
    placer_args = placer_command[2:] + ["--buffering-commit-enabled", "0"]
    command = [
        str(args.python_bin.resolve()),
        str(Path(__file__).resolve()),
        "--case",
        args.case,
        "--injected-run",
        "--artifact",
        str(artifact_path),
        "--",
        *placer_args,
    ]
    (run_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started_at = time.perf_counter()
    with (run_root / "placer.log").open("w", encoding="utf-8", errors="ignore") as stream:
        completed = subprocess.run(
            command,
            cwd=str(AUTODMP_ROOT),
            env=dict(os.environ),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    execution_wall_sec = time.perf_counter() - started_at
    artifact = (
        json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact_path.is_file()
        else {}
    )
    native_total_ms = float((artifact.get("metadata") or {}).get("native_total_ms", 0.0))
    failures = []
    if completed.returncode != 0:
        failures.append("placer_or_probe_failed")
    if artifact.get("status") != "ok":
        failures.append("native_packer_not_ok")
    if native_total_ms <= 0.0:
        failures.append("missing_native_total_ms")
    if native_total_ms > float(args.early_gate_sec) * 1000.0:
        failures.append("native_packer_early_gate_missed")
    result = {
        "status": "pass" if not failures else "failed",
        "failures": failures,
        "case": args.case,
        "execution_returncode": int(completed.returncode),
        "execution_wall_sec": float(execution_wall_sec),
        "early_gate_sec": float(args.early_gate_sec),
        "native_total_ms": native_total_ms,
        "artifact_path": str(artifact_path),
        "log_path": str(run_root / "placer.log"),
    }
    _write_json(run_root / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
