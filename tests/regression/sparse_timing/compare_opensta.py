"""Compare unique endpoint slack against OpenSTA's worst legal candidates.

All arithmetic is in ps. Golden rows may repeat an endpoint across path groups;
only the worst candidate contributes to WNS/TNS, independent of file ordering.
This endpoint gate does not certify per-rule identity or path-class membership.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_golden(path):
    endpoints = {}
    with path.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["endpoint"]
            if "/" in name:
                instance, port = name.rsplit("/", 1)
                name = f"{instance}:{port}"
            slack = float(row["slack_ns"]) * 1000
            if not np.isfinite(slack):
                raise ValueError(f"nonfinite golden slack: {name}")
            if name not in endpoints or slack < endpoints[name]["slack_ps"]:
                endpoints[name] = {
                    "slack_ps": slack,
                    "launch_clock": row["launch_clock"],
                    "capture_clock": row["capture_clock"],
                }
                for field in ("aat_ps", "required_ps"):
                    if field in row:
                        endpoints[name][field] = float(row[field])
                for field in ("transition", "is_path_delay", "is_check", "is_output_delay"):
                    if field in row:
                        endpoints[name][field] = row[field]
    return endpoints


def tolerance(golden_ps):
    return max(2.0, 0.03 * abs(golden_ps))


def compare(actual, golden):
    rows = []
    for name in sorted(actual.keys() & golden.keys()):
        reference = golden[name]["slack_ps"]
        error = actual[name] - reference
        rows.append({
            "endpoint": name,
            **golden[name],
            "sparse_slack_ps": actual[name],
            "error_ps": error,
            "tolerance_ps": tolerance(reference),
            "passed": bool(abs(error) <= tolerance(reference)),
        })
    rows.sort(key=lambda row: abs(row["error_ps"]), reverse=True)
    summary = {
        "golden_endpoint_count": len(golden),
        "sparse_endpoint_count": len(actual),
        "missing_endpoints": sorted(golden.keys() - actual.keys()),
        "extra_endpoints": sorted(actual.keys() - golden.keys()),
        "compared_endpoints": len(rows),
        "slack_failures": sum(not row["passed"] for row in rows),
        "max_absolute_error_ps": max((abs(row["error_ps"]) for row in rows), default=0),
        "tolerance": {"relative": 0.03, "absolute_floor_ps": 2.0},
    }
    for label, values in (("sparse", actual.values()), ("golden", (r["slack_ps"] for r in golden.values()))):
        slacks = np.array(list(values))
        summary[label] = {
            "wns_ps": float(slacks.min(initial=0)),
            "tns_ps": float(np.minimum(slacks, 0).sum()),
            "violating_endpoints": int((slacks < 0).sum()),
        }
    summary["aggregate_passed"] = all(
        abs(summary["sparse"][metric] - summary["golden"][metric]) <= tolerance(summary["golden"][metric])
        for metric in ("wns_ps", "tns_ps")
    )
    summary["endpoint_gate_passed"] = not (
        summary["missing_endpoints"] or summary["extra_endpoints"] or summary["slack_failures"]
    ) and summary["aggregate_passed"]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.export / "base_arrays.npz") as base:
        pin_names = base["pin_names"]
    timing = dict(np.load(args.timing / "timing.npz"))
    names = pin_names[timing["endpoint_pins"]]
    slacks = timing["endpoint_slack"]
    if len(set(names)) != len(names) or not np.isfinite(slacks).all():
        raise ValueError("sparse endpoint slacks must be finite and unique")
    actual = dict(zip(names.tolist(), slacks.tolist()))
    summary, rows = compare(actual, read_golden(args.golden))
    arrays = dict(np.load(args.export / "sparse_arrays.npz"))
    metadata = json.loads((args.export / "sparse_metadata.json").read_text())["metadata"]
    checks = arrays["check_index"]
    legal_rows = np.flatnonzero(checks[:, 6])
    order = np.argsort(timing["check_slack"], kind="stable")
    pins = arrays["state_pin"][checks[legal_rows[order], 0]]
    unique_pins, first = np.unique(pins, return_index=True)
    worst_checks = dict(zip(pin_names[unique_pins], legal_rows[order[first]]))
    required_failures = 0
    for row in rows:
        index = worst_checks[row["endpoint"]]
        state, capture, _, edge, _, kind, _ = checks[index]
        row.update(
            sparse_aat_ps=float(timing["state_aat"][state, edge]),
            sparse_required_ps=float(timing["required"][index]),
            sparse_transition=("^", "v")[edge],
            sparse_launch_clock=metadata["domains"][arrays["state_domain"][state]],
            sparse_capture_clock=metadata["domains"][capture],
            sparse_rule_kind=("normal", "max_delay", "multicycle", "false_path")[kind],
        )
        if "required_ps" in row:
            row["required_error_ps"] = row["sparse_required_ps"] - row["required_ps"]
            row["required_passed"] = abs(row["required_error_ps"]) <= tolerance(row["required_ps"])
            required_failures += not row["required_passed"]
    if rows and "required_ps" in rows[0]:
        summary["required_failures_on_selected_endpoint_candidate"] = required_failures
        summary["endpoint_gate_passed"] &= required_failures == 0
    summary.update(export=str(args.export.resolve()), timing=str(args.timing.resolve()), golden_file=str(args.golden.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "endpoints.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["endpoint"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
