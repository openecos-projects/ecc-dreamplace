#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git_provenance(path):
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--short", "--untracked-files=no"], text=True
    ).strip()
    return {"commit": commit, "tracked_worktree_dirty": bool(status)}


def cmake_provenance(cache_path):
    requested = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS_RELEASE",
        "XPLACE_ENABLE_CUDA",
    }
    values = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            key_and_type, separator, value = line.partition("=")
            key = key_and_type.partition(":")[0]
            if separator and key in requested:
                values[key] = value
    return values


def environment_provenance(xplace_root):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cmake_cache = os.environ.get("ECC_GPUGR_BENCHMARK_CMAKE_CACHE")
    cmake_cache_path = (
        Path(cmake_cache).expanduser()
        if cmake_cache
        else xplace_root / "build_cpu_pr" / "CMakeCache.txt"
    )
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cmake": cmake_provenance(cmake_cache_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark an explicit Xplace GPUGR backend")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-def", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-name", required=True)
    parser.add_argument("--backend", required=True, choices=("cpu_pr", "cpu_pr_mt", "cuda"))
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--route-xsize", type=int, default=512)
    parser.add_argument("--route-ysize", type=int, default=512)
    parser.add_argument("--bottom-routing-layer", default="")
    parser.add_argument("--top-routing-layer", default="")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser.parse_args()


def write_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    config_path = args.config.expanduser().resolve()
    input_def = args.input_def.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dreamplace_root = Path(__file__).resolve().parents[1]
    xplace_root = Path(
        os.environ.get(
            "ECC_XPLACE_ROOT", dreamplace_root / "thirdparty" / "xplace"
        )
    ).expanduser().resolve()
    gpugr_extensions = sorted((xplace_root / "cpp_to_py" / "cpybin").glob("gpugr*.so"))
    if not gpugr_extensions:
        raise RuntimeError(f"no installed GPUGR extension under {xplace_root}")
    lef_paths = [str(Path(path).expanduser().resolve()) for path in config["lef_input"]]
    params = SimpleNamespace(
        lef_input=lef_paths,
        result_dir=str(output_dir),
        gpugr_bottom_routing_layer=args.bottom_routing_layer,
        gpugr_top_routing_layer=args.top_routing_layer,
    )
    request = {
        "input_def": str(input_def),
        "out_dir": str(output_dir),
        "design_name": args.design_name,
        "benchmark": args.benchmark or f"{args.design_name}_{args.backend}_t{args.threads}",
        "gpu": args.gpu,
        "threads": args.threads,
        "route_xsize": args.route_xsize,
        "route_ysize": args.route_ysize,
        "rrr_iters": 0,
        "skip_m1_route": True,
        "save_artifacts": args.save_artifacts,
        "profile_enabled": True,
        "profile_prefix": f"benchmark.{args.design_name}.{args.backend}.t{args.threads}",
        "backend": args.backend,
        "bottom_routing_layer": args.bottom_routing_layer,
        "top_routing_layer": args.top_routing_layer,
    }
    summary = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "request": request,
        "config_path": str(config_path),
        "source": {
            "ecc_dreamplace": git_provenance(dreamplace_root),
            "xplace": git_provenance(dreamplace_root / "thirdparty" / "xplace"),
            "xplace_runtime_root": str(xplace_root),
            "gpugr_extension": str(gpugr_extensions[0]),
            "gpugr_extension_sha256": sha256_file(gpugr_extensions[0]),
        },
        "environment": environment_provenance(xplace_root),
        "input_sha256_status": "pending",
    }
    write_summary(summary_path, summary)
    print(json.dumps({"event": "start", **summary}, sort_keys=True), flush=True)

    summary["input_sha256"] = {
        "def": sha256_file(input_def),
        "lefs": {path: sha256_file(path) for path in lef_paths},
    }
    summary["input_sha256_status"] = "complete"
    write_summary(summary_path, summary)

    started = time.perf_counter()
    try:
        result = XplaceGPUGR(params, SimpleNamespace()).run_gpugr(**request)
        native_stats = dict(result.get("native_stats", {}))
        if (
            args.backend in ("cpu_pr", "cpu_pr_mt")
            and native_stats.get("terminal_status") != "completed"
        ):
            raise RuntimeError(f"native route did not complete: {native_stats}")
        maps = result["maps"]
        map_hashes = {
            key: tensor_sha256(maps[key])
            for key in ("capacity_map", "raw_wire_demand_map", "wire_demand_map", "via_demand_map")
        }
        summary.update(
            {
                "status": "success",
                "exit_code": 0,
                "elapsed_wall_sec": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "native_stats": native_stats,
                "map_sha256": map_hashes,
                "metrics": dict(result["metrics"]),
                "artifact_paths": dict(result.get("artifact_paths", {})),
            }
        )
        print(json.dumps({"event": "complete", **summary}, sort_keys=True), flush=True)
    except KeyboardInterrupt as error:
        summary.update(
            {
                "status": "interrupted",
                "exit_code": 130,
                "elapsed_wall_sec": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "exception_type": type(error).__name__,
                "exception": "benchmark interrupted",
            }
        )
        print(json.dumps({"event": "interrupted", **summary}, sort_keys=True), flush=True)
        raise
    except BaseException as error:
        summary.update(
            {
                "status": "failed",
                "exit_code": 1,
                "elapsed_wall_sec": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "exception_type": type(error).__name__,
                "exception": str(error),
            }
        )
        print(json.dumps({"event": "failed", **summary}, sort_keys=True), flush=True)
        raise
    finally:
        write_summary(summary_path, summary)


if __name__ == "__main__":
    main()
