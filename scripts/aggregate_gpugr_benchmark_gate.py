#!/usr/bin/env python3
"""Aggregate repeated benchmark summaries into median/p95 gate evidence."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups = {}
    for summary_path in sorted(args.root.glob("*/results/benchmark_summary.json")):
        tag = summary_path.parents[1].name
        if tag == "warmup":
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        request = summary.get("request", {})
        key = (request.get("backend"), request.get("threads"))
        native = summary.get("native_stats", {})
        groups.setdefault(key, []).append(
            {
                "tag": tag,
                "status": summary.get("status"),
                "exit_code": summary.get("exit_code"),
                "elapsed_wall_sec": summary.get("elapsed_wall_sec"),
                "native_total_seconds": native.get("total_seconds"),
                "peak_rss_kib": summary.get("peak_rss_kib"),
                "effective_workers": native.get("effective_workers"),
                "route_hash64": native.get("route_hash64"),
                "wire_map_hash64": native.get("wire_map_hash64"),
                "via_map_hash64": native.get("via_map_hash64"),
                "metrics": summary.get("metrics", {}),
            }
        )

    report = {"fixture_root": str(args.root), "groups": []}
    ordered = sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    for (backend, threads), runs in ordered:
        native_times = sorted(
            run["native_total_seconds"] for run in runs if run["native_total_seconds"] is not None
        )
        wall_times = sorted(
            run["elapsed_wall_sec"] for run in runs if run["elapsed_wall_sec"] is not None
        )
        route_hashes = {run["route_hash64"] for run in runs}
        wire_hashes = {run["wire_map_hash64"] for run in runs}
        via_hashes = {run["via_map_hash64"] for run in runs}
        statuses = sorted({str(run["status"]) for run in runs})
        report["groups"].append(
            {
                "backend": backend,
                "threads": threads,
                "runs": len(runs),
                "statuses": statuses,
                "native_total_seconds": native_times,
                "native_median_sec": statistics.median(native_times) if native_times else None,
                "native_p95_sec": percentile(native_times, 0.95),
                "wall_median_sec": statistics.median(wall_times) if wall_times else None,
                "wall_p95_sec": percentile(wall_times, 0.95),
                "peak_rss_kib_max": max(
                    run["peak_rss_kib"] for run in runs if run["peak_rss_kib"] is not None
                ),
                "effective_workers": sorted({run["effective_workers"] for run in runs}),
                "route_hash64_unique": sorted(route_hashes),
                "wire_map_hash64_unique": sorted(wire_hashes),
                "via_map_hash64_unique": sorted(via_hashes),
                "deterministic": (
                    len(route_hashes) == 1 and len(wire_hashes) == 1 and len(via_hashes) == 1
                ),
                "metrics_last": runs[-1]["metrics"],
            }
        )

    serial = next((group for group in report["groups"] if group["backend"] == "cpu_pr"), None)
    if serial and serial["native_median_sec"]:
        for group in report["groups"]:
            if group["native_median_sec"]:
                speedup = serial["native_median_sec"] / group["native_median_sec"]
                group["speedup_vs_serial_cpu_pr"] = round(speedup, 4)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
