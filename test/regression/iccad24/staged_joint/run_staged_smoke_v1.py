#!/usr/bin/env python3
"""Run ICCAD24 canonical staged_smoke_v1 joint flow and replay final DEFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "regression" / "iccad24" / "staged_joint"
DEFAULT_PYTHON = Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python")

CSV_COLUMNS = (
    "case",
    "status",
    "run_dir",
    "runtime_sec",
    "summary_path",
    "final_def",
    "final_def_sha256",
    "finalization_wns_ns",
    "finalization_tns_ns",
    "replay_wns_ns",
    "replay_tns_ns",
    "replay_num_instances",
    "replay_runtime_sec",
    "delta_wns_ns",
    "delta_tns_ns",
    "error",
)


def _tcl_quote(value: str | Path) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cases(raw_cases: list[str] | None) -> list[str]:
    if not raw_cases:
        return list(ICCAD24_CASES)
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


def build_joint_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    result_dir: Path,
    output_def: Path,
    outer_iterations: int,
    gpu: int,
    gpu_id: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    command = [
        str(python_bin),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        str(paths["params_json"]),
        "--flow-kind",
        "joint",
        "--joint-quality-profile",
        "staged_smoke_v1",
        "--joint-staged-smoke-outer-iterations",
        str(int(outer_iterations)),
        "--place-io-engine",
        "openroad",
        "--gpu",
        str(int(gpu)),
        "--gpu-id",
        str(int(gpu_id)),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--output-def",
        str(output_def),
        "--workspace",
        str(paths["workspace"]),
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
    ]
    for lef in tech["lef"]:
        command.extend(["--lef", str(lef)])
    for lib in tech["lib"]:
        command.extend(["--lib", str(lib)])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


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


def _parse_metric_line(line: str) -> tuple[str, Any] | None:
    if not line.startswith("METRIC|"):
        return None
    _, key, raw = line.rstrip("\n").split("|", 2)
    parsed = _as_float(raw)
    return key, raw if parsed is None else parsed


def parse_replay_metrics(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_metric_line(line)
        if parsed:
            key, value = parsed
            metrics[key] = value
    return metrics


def generate_def_only_replay_tcl(
    *,
    benchmark_root: Path,
    case: str,
    final_def: Path,
) -> str:
    paths = validate_case_inputs(benchmark_root, case)
    tech = asap7_inputs(benchmark_root)
    lines = [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        "set start_us [clock microseconds]",
        f"emit_metric \"case\" {_tcl_quote(case)}",
        f"read_lef {_tcl_quote(tech['tech_lef'])}",
    ]
    for lef in tech["lef"]:
        lines.append(f"read_lef {_tcl_quote(lef)}")
    for lib in tech["lib"]:
        lines.append(f"read_liberty {_tcl_quote(lib)}")
    lines.extend(
        [
            f"read_def -continue_on_errors {_tcl_quote(final_def)}",
            f"read_sdc {_tcl_quote(paths['sdc'])}",
            "set_ideal_network [all_clocks]",
            f"source {_tcl_quote(tech['rc_tcl'])}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
            'emit_metric "wns_ns" [worst_slack -max]',
            'emit_metric "tns_ns" [total_negative_slack -max]',
            'emit_metric "num_instances" [llength [get_cells *]]',
            'emit_metric "runtime_sec" [expr {([clock microseconds] - $start_us) / 1000000.0}]',
            "exit",
        ]
    )
    return "\n".join(lines) + "\n"


def final_metrics_from_summary(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    finalization = summary.get("finalization") or {}
    direct = finalization.get("direct_metrics") or {}
    wns_ps = _as_float((direct.get("post_repair_timing_wns") or {}).get("value"))
    tns_ps = _as_float((direct.get("post_repair_timing_tns") or {}).get("value"))
    if wns_ps is None or tns_ps is None:
        timing = finalization.get("timing_summary") or {}
        post = timing.get("post_repair_timing") or {}
        wns_ps = _as_float(post.get("wns"))
        tns_ps = _as_float(post.get("tns"))
    return (
        None if wns_ps is None else wns_ps / 1000.0,
        None if tns_ps is None else tns_ps / 1000.0,
    )


def run_case(args: argparse.Namespace, run_root: Path, case: str) -> dict[str, Any]:
    case_root = run_root / case
    result_dir = case_root / "result"
    requested_def = result_dir / f"{case}_requested_output.def"
    summary_path = result_dir / f"{case}_joint_staged_smoke_summary.json"
    run_log = case_root / "run.log"
    command_path = case_root / "command.txt"
    command = build_joint_command(
        python_bin=args.python_bin,
        benchmark_root=args.benchmark_root,
        case=case,
        result_dir=result_dir,
        output_def=requested_def,
        outer_iterations=args.outer_iterations,
        gpu=args.gpu,
        gpu_id=args.gpu_id,
        extra_args=args.placer_arg,
    )
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.perf_counter()
    if args.skip_completed and summary_path.exists():
        returncode = 0
    else:
        returncode = run_command(command, cwd=REPO_ROOT, log_path=run_log)
    runtime_sec = time.perf_counter() - started
    if returncode != 0:
        return {
            "case": case,
            "status": "failed",
            "run_dir": str(case_root),
            "runtime_sec": runtime_sec,
            "error": f"Placer exited with code {returncode}",
        }
    if not summary_path.exists():
        return {
            "case": case,
            "status": "failed",
            "run_dir": str(case_root),
            "runtime_sec": runtime_sec,
            "error": f"missing summary: {summary_path}",
        }
    summary = _read_json(summary_path)
    finalization = summary.get("finalization") or {}
    final_def_info = finalization.get("final_def") or {}
    final_def = Path(str(final_def_info.get("path") or ""))
    if not final_def.exists():
        return {
            "case": case,
            "status": "failed",
            "run_dir": str(case_root),
            "runtime_sec": runtime_sec,
            "summary_path": str(summary_path),
            "error": f"missing final DEF: {final_def}",
        }
    final_wns_ns, final_tns_ns = final_metrics_from_summary(summary)
    replay_dir = case_root / "replay_fast"
    replay_tcl = replay_dir / "eval_fast.tcl"
    replay_log = replay_dir / "openroad.log"
    replay_tcl.parent.mkdir(parents=True, exist_ok=True)
    replay_tcl.write_text(
        generate_def_only_replay_tcl(
            benchmark_root=args.benchmark_root,
            case=case,
            final_def=final_def,
        ),
        encoding="utf-8",
    )
    replay_status = "ok"
    replay_error = ""
    if args.skip_completed and replay_log.exists():
        replay_returncode = 0
    else:
        replay_returncode = run_command(
            [str(args.openroad_bin), "-exit", str(replay_tcl)],
            cwd=replay_tcl.parent,
            log_path=replay_log,
        )
    if replay_returncode != 0:
        replay_status = "failed"
        replay_error = f"OpenROAD replay exited with code {replay_returncode}"
    replay_metrics = parse_replay_metrics(replay_log)
    replay_wns_ns = _as_float(replay_metrics.get("wns_ns"))
    replay_tns_ns = _as_float(replay_metrics.get("tns_ns"))
    status = "ok" if replay_status == "ok" else "failed"
    return {
        "case": case,
        "status": status,
        "run_dir": str(case_root),
        "runtime_sec": runtime_sec,
        "summary_path": str(summary_path),
        "final_def": str(final_def),
        "final_def_sha256": _sha256(final_def),
        "finalization_wns_ns": final_wns_ns,
        "finalization_tns_ns": final_tns_ns,
        "replay_wns_ns": replay_wns_ns,
        "replay_tns_ns": replay_tns_ns,
        "replay_num_instances": _as_float(replay_metrics.get("num_instances")),
        "replay_runtime_sec": _as_float(replay_metrics.get("runtime_sec")),
        "delta_wns_ns": None if final_wns_ns is None or replay_wns_ns is None else replay_wns_ns - final_wns_ns,
        "delta_tns_ns": None if final_tns_ns is None or replay_tns_ns is None else replay_tns_ns - final_tns_ns,
        "error": replay_error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_baselines(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = _read_json(path)
    baselines: dict[str, dict[str, Any]] = {}
    for item in data.get("results", []):
        case = item.get("case")
        baseline = item.get("baseline") or {}
        if case:
            baselines[str(case)] = baseline
    return baselines


def _fmt_ns(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.6f}ns"


def _delta(value: Any, baseline: Any) -> float | None:
    parsed = _as_float(value)
    base = _as_float(baseline)
    if parsed is None or base is None:
        return None
    return parsed - base


def write_bundle(
    *,
    bundle_root: Path,
    rows: list[dict[str, Any]],
    run_root: Path,
    baseline_manifest: Path | None,
) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    defs_dir = bundle_root / "defs"
    defs_dir.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines(baseline_manifest)
    manifest_results = []
    table_lines = [
        "# Replayed staged_smoke_v1 Joint-Flow DEF Bundle",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Selection rule: every row comes from the current canonical ",
        "`--flow-kind joint --joint-quality-profile staged_smoke_v1` flow, then ",
        "is independently replayed by OpenROAD in DEF-only mode.",
        "",
        f"Source run root: `{run_root}`",
        "",
        "| Case | Replay WNS/TNS | Finalization WNS/TNS | OpenROAD baseline WNS/TNS | Delta TNS vs baseline | DEF artifact | Notes |",
        "|------|----------------|----------------------|---------------------------|-----------------------|--------------|-------|",
    ]
    checksum_paths: list[Path] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        case = str(row["case"])
        src_def = Path(str(row["final_def"]))
        dst_def = defs_dir / f"{case}_staged_smoke_v1_replayed.def"
        shutil.copy2(src_def, dst_def)
        checksum_paths.append(dst_def)
        baseline = baselines.get(case, {})
        delta_tns = _delta(row.get("replay_tns_ns"), baseline.get("tns_ns"))
        def_rel = dst_def.relative_to(bundle_root)
        manifest_results.append(
            {
                "case": case,
                "status": row.get("status"),
                "run_kind": "canonical_joint_staged_smoke_v1_replayed",
                "source_run_dir": row.get("run_dir"),
                "source_summary_path": row.get("summary_path"),
                "source_final_def_path": row.get("final_def"),
                "bundle_relative_def_path": str(def_rel),
                "copied_def_path": str(dst_def),
                "copied_def_sha256": _sha256(dst_def),
                "finalization_wns_ns": row.get("finalization_wns_ns"),
                "finalization_tns_ns": row.get("finalization_tns_ns"),
                "replay_wns_ns": row.get("replay_wns_ns"),
                "replay_tns_ns": row.get("replay_tns_ns"),
                "delta_replay_vs_finalization_wns_ns": row.get("delta_wns_ns"),
                "delta_replay_vs_finalization_tns_ns": row.get("delta_tns_ns"),
                "baseline": baseline,
                "delta_vs_openroad_tns_ns": delta_tns,
                "delta_vs_openroad_wns_ns": _delta(row.get("replay_wns_ns"), baseline.get("wns_ns")),
            }
        )
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case}`",
                    f"`{_fmt_ns(row.get('replay_wns_ns'))}` / `{_fmt_ns(row.get('replay_tns_ns'))}`",
                    f"`{_fmt_ns(row.get('finalization_wns_ns'))}` / `{_fmt_ns(row.get('finalization_tns_ns'))}`",
                    f"`{_fmt_ns(baseline.get('wns_ns'))}` / `{_fmt_ns(baseline.get('tns_ns'))}`",
                    "`n/a`" if delta_tns is None else f"`{delta_tns:+.6f}ns`",
                    f"`{def_rel}`",
                    "DEF-only replay metric",
                ]
            )
            + " |"
        )
    manifest = {
        "artifact": "staged_smoke_v1_replayed_def_bundle",
        "artifact_version": 1,
        "bundle_created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_root": str(bundle_root),
        "source_run_root": str(run_root),
        "baseline_manifest": None if baseline_manifest is None else str(baseline_manifest),
        "selection_rule": "current canonical staged_smoke_v1 joint flow, DEF-only replay metrics",
        "results": manifest_results,
    }
    _write_json(bundle_root / "manifest.json", manifest)
    table_path = bundle_root / "staged_smoke_v1_replayed_vs_openroad.md"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    readme = [
        "# Replayed staged_smoke_v1 Joint-Flow DEF Bundle",
        "",
        "This bundle is generated from the current canonical staged joint flow.",
        "Each DEF is copied from the OpenROAD-direct final DEF emitted by",
        "`openroad_bridge.write_def`, then checked with independent DEF-only",
        "OpenROAD replay.",
        "",
        "Files:",
        "",
        "- `manifest.json`: machine-readable metrics and source paths.",
        "- `staged_smoke_v1_replayed_vs_openroad.md`: human-readable table.",
        "- `defs/`: replayed final DEF artifacts.",
        "- `SHA256SUMS`: checksums for tracked bundle artifacts.",
        "",
    ]
    (bundle_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    checksum_paths.extend([bundle_root / "manifest.json", table_path, bundle_root / "README.md"])
    lines = []
    for path in sorted(checksum_paths):
        lines.append(f"{_sha256(path)}  {path.relative_to(bundle_root)}")
    (bundle_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("staged_smoke_v1_replayed_%Y%m%d_%H%M%S"))
    parser.add_argument("--python-bin", type=Path, default=Path(os.environ.get("PYTHON_BIN", DEFAULT_PYTHON)))
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_BENCHMARK_ROOT / "openroad")
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--placer-arg", action="append", default=[])
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--update-bundle", action="store_true")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "best_known_joint_flow_table_defs_20260627" / "manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.benchmark_root = args.benchmark_root.resolve()
    args.openroad_bin = args.openroad_bin.resolve()
    run_root = args.output_root.resolve() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in resolve_cases(args.cases):
        row = run_case(args, run_root, case)
        rows.append(row)
        print(
            f"{case}: {row.get('status')} "
            f"final_tns={row.get('finalization_tns_ns')} "
            f"replay_tns={row.get('replay_tns_ns')} "
            f"err={row.get('error', '')}",
            flush=True,
        )
    write_csv(run_root / "staged_smoke_v1_replayed_summary.csv", rows)
    _write_json(
        run_root / "staged_smoke_v1_replayed_summary.json",
        {
            "artifact": "staged_smoke_v1_replayed_summary",
            "artifact_version": 1,
            "run_root": str(run_root),
            "run_id": args.run_id,
            "rows": rows,
        },
    )
    if args.update_bundle:
        bundle_root = args.bundle_root
        if bundle_root is None:
            bundle_root = args.output_root.resolve() / f"{args.run_id}_replayed_defs"
        write_bundle(
            bundle_root=bundle_root.resolve(),
            rows=rows,
            run_root=run_root,
            baseline_manifest=args.baseline_manifest.resolve() if args.baseline_manifest else None,
        )
        print(f"bundle_root: {bundle_root.resolve()}")
    print(f"summary_json: {run_root / 'staged_smoke_v1_replayed_summary.json'}")
    return 0 if rows and all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
