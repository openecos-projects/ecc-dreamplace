"""Inspect native max/min structure without running numeric timing or sizing.

Limits must be frozen before launch. External supervision enforces wall/RSS
while native preparation is running; this process checks state/edge counts
and saves their diagnostics before rejecting an oversized completed model.
"""

import argparse
import json
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import preflight
from dreamplace.ops.placeio_ecc.hold_schema import read_hold_timing_schema
from dreamplace.ops.placeio_ecc.sparse_timing_schema import read_sparse_timing_view

NUMERIC_PHASES = (
    "clock_propagation",
    "data_slew",
    "data_arrival",
    "required",
    "path_analysis",
    "delay_graph_initialization",
)


def inspect(pydb, output, limits):
    maximum = read_sparse_timing_view(pydb)
    maximum.require_supported()
    reasons = np.asarray(pydb.endpoints_min_reason).reshape(-1, 2)
    codes, counts = np.unique(reasons, return_counts=True)
    endpoints = np.asarray(pydb.end_points)
    unqualified = np.flatnonzero(np.any(reasons != 0, axis=1))
    qualification = {
        "lane_reason_counts": dict(zip(map(str, codes.tolist()), counts.tolist(), strict=True)),
        "unqualified_endpoint_count": len(unqualified),
        "unqualified_endpoint_samples": [
            {
                "pin": pydb.pin_names[int(endpoints[row])],
                "reason": reasons[row].tolist(),
            }
            for row in unqualified[:100]
        ],
    }
    covered = np.zeros((len(pydb.pin_names), 2), dtype=bool)
    for row in pydb.hold_contexts:
        covered[row["endpoint_pin"], 0 if row["data_transition"] == 1 else 1] = True
    valid = np.asarray(pydb.endpoints_min_valid).reshape(-1, 2)
    missing = np.flatnonzero((covered[endpoints] != valid).any(axis=1))
    qualification.update(
        context_count=len(pydb.hold_contexts),
        context_coverage_mismatch_count=len(missing),
        context_coverage_mismatch_samples=[
            {
                "pin": pydb.pin_names[int(endpoints[row])],
                "valid": valid[row].tolist(),
                "covered": covered[endpoints[row]].tolist(),
                "reason": reasons[row].tolist(),
            }
            for row in missing[:100]
        ],
        producer_diagnostics=dict(Counter(pydb.timing_diagnostics)),
    )
    # Keep producer diagnostics even when the min schema correctly rejects it.
    (output / "min_qualification.json").write_text(json.dumps(qualification, indent=2))
    hold = read_hold_timing_schema(pydb)
    minimum = hold.sparse_view
    if minimum is None:
        raise RuntimeError("structure inspection requires a native min graph")
    native = dict(pydb.timing_preparation)
    numeric_calls = {name: native["phase_calls"][name] for name in NUMERIC_PHASES}
    views = {}
    for name, view in (("max", maximum), ("min", minimum)):
        arrays = view.arrays
        counts = np.diff(arrays["pin_state_start"])
        peak_pins = np.argsort(counts, stable=True)[-20:][::-1]
        samples = []
        for pin in peak_pins:
            begin, end = arrays["pin_state_start"][pin : pin + 2]
            samples.append(
                {
                    "pin": pydb.pin_names[int(pin)],
                    "states": int(counts[pin]),
                    "domains": [
                        view.domains[i] for i in np.unique(arrays["state_domain"][begin:end])
                    ],
                    "tags": [view.tags[i] for i in np.unique(arrays["state_tag"][begin:end])],
                }
            )
        views[name] = {
            **view.statistics(),
            "states_per_pin_p50": float(np.percentile(counts, 50)) if len(counts) else 0,
            "states_per_pin_p99": float(np.percentile(counts, 99)) if len(counts) else 0,
            "high_state_pins": samples,
        }
    states = sum(view["num_states"] for view in views.values())
    edges = sum(
        view[key]
        for view in views.values()
        for key in ("num_tagged_cell_arcs", "num_tagged_net_arcs", "num_slew_only_cell_arcs")
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    failures = []
    if any(numeric_calls.values()):
        failures.append("model-only preparation executed a full numeric phase")
    if states > limits["max_states"]:
        failures.append("max_states exceeded")
    if edges > limits["max_edges"]:
        failures.append("max_edges exceeded")
    if rss > limits["max_rss_bytes"]:
        failures.append("max_rss_bytes exceeded")
    report = {
        "status": "passed" if not failures else "failed",
        "scope": (
            "native model-only preparation and schema/qualification validation; "
            "no timing forward, sizing or placement"
        ),
        "device": "cpu",
        "max_min_states": states,
        "max_min_edges": edges,
        "peak_rss_bytes": rss,
        "whole_pipeline_rss_bytes_per_state": rss / states if states else None,
        "array_bytes_per_state": sum(v["array_bytes"] for v in views.values()) / states
        if states
        else None,
        "native_preparation": native,
        "numeric_phase_calls": numeric_calls,
        "min_qualification": qualification,
        "views": views,
        "limits": limits,
        "failures": failures,
    }
    (output / "structure_scale.json").write_text(json.dumps(report, indent=2))
    print(
        "STRUCTURE_SCALE",
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "max_min_states",
                    "max_min_edges",
                    "peak_rss_bytes",
                    "numeric_phase_calls",
                )
            }
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "ecc-root", "native-module", "limits"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.timing_mode = "model_only"
    limits = json.loads(args.limits.read_text())
    if any(
        limits[name] <= 0 for name in ("max_wall_s", "max_rss_bytes", "max_states", "max_edges")
    ):
        raise ValueError("resource limits must be frozen positive numbers")
    preflight.run(args, export=lambda pydb, output: inspect(pydb, output, limits))


if __name__ == "__main__":
    main()
