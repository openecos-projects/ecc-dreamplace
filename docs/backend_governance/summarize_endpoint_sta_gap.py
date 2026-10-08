#!/usr/bin/env python3
"""Summarize Python endpoint slack by endpoint check class.

This script intentionally works from CSV artifacts only, so backend parity
debugging can be repeated without rebuilding either backend.
"""

import argparse
import csv
import math
from collections import defaultdict


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _normalize_pin_name(value):
    return str(value or "").replace("/", ":").replace("\\[", "[").replace("\\]", "]")


def _pin_role(pin_name):
    leaf = _normalize_pin_name(pin_name).rsplit(":", 1)[-1].upper()
    if leaf in {"D", "DI", "DATA", "DIN"}:
        return "data"
    if leaf in {"CLK", "CK", "CP", "CLOCK", "GCLK"}:
        return "clock"
    if any(token in leaf for token in ("RESET", "RST", "CLR", "CLEAR", "SET", "PRESET")):
        return "async_control"
    return "unknown"


def _check_class(debug_row):
    explicit = str(debug_row.get("endpoint_check_class") or "").strip()
    if explicit:
        return explicit
    count = int(float(debug_row.get("constraint_arc_count") or 0))
    if count > 0:
        return "setup_exported"
    role = _pin_role(debug_row.get("pin_name") or debug_row.get("normalized_pin_name") or "")
    if role == "async_control":
        return "async_control_unmodeled"
    return "unsupported_unmodeled"


def _read_debug(debug_csv):
    rows = {}
    with open(debug_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            key = row.get("normalized_pin_name") or _normalize_pin_name(row.get("pin_name"))
            rows[key] = row
    return rows


def _metric_bucket():
    return {
        "count": 0,
        "negative": 0,
        "wns": None,
        "tns": 0.0,
    }


def _delta_bucket():
    return {
        "count": 0,
        "comparable": 0,
        "skipped_transition_mismatch": 0,
        "skipped_sentinel": 0,
        "slack_delta_sum": 0.0,
        "abs_slack_delta_sum": 0.0,
        "aat_delta_sum": 0.0,
        "rat_delta_sum": 0.0,
        "decomposition_residual_abs_sum": 0.0,
        "max_abs_slack_delta": 0.0,
        "max_abs_slack_delta_pin": "",
    }


def _update_metric(bucket, slack):
    if slack is None:
        return
    bucket["count"] += 1
    if bucket["wns"] is None or slack < bucket["wns"]:
        bucket["wns"] = slack
    if slack < 0.0:
        bucket["negative"] += 1
        bucket["tns"] += slack


def _is_finite_number(value):
    return value is not None and math.isfinite(value)


def _is_sentinel_value(value):
    return value <= -5.0e7 or value >= 5.0e7


def _transition_values(row, prefix):
    return {
        "rise": (
            _float_or_none(row.get(f"{prefix}_r_aat")),
            _float_or_none(row.get(f"{prefix}_r_rat")),
        ),
        "fall": (
            _float_or_none(row.get(f"{prefix}_f_aat")),
            _float_or_none(row.get(f"{prefix}_f_rat")),
        ),
    }


def _critical_transition(values):
    best_transition = None
    best_slack = None
    for transition, (aat, rat) in values.items():
        if not (_is_finite_number(aat) and _is_finite_number(rat)):
            continue
        slack = rat - aat
        if best_slack is None or slack < best_slack:
            best_transition = transition
            best_slack = slack
    return best_transition, best_slack


def _update_delta(bucket, row):
    py_values = _transition_values(row, "py")
    backend_values = _transition_values(row, "cpp")
    py_transition, py_slack = _critical_transition(py_values)
    backend_transition, backend_slack = _critical_transition(backend_values)
    if py_transition is None or backend_transition is None:
        return
    bucket["count"] += 1
    if py_transition != backend_transition:
        bucket["skipped_transition_mismatch"] += 1
        return
    py_aat, py_rat = py_values[py_transition]
    backend_aat, backend_rat = backend_values[backend_transition]
    if any(_is_sentinel_value(value) for value in (py_aat, py_rat, backend_aat, backend_rat)):
        bucket["skipped_sentinel"] += 1
        return
    slack_delta = py_slack - backend_slack
    bucket["comparable"] += 1
    bucket["slack_delta_sum"] += slack_delta
    bucket["abs_slack_delta_sum"] += abs(slack_delta)
    if abs(slack_delta) > bucket["max_abs_slack_delta"]:
        bucket["max_abs_slack_delta"] = abs(slack_delta)
        bucket["max_abs_slack_delta_pin"] = row.get("pin_name") or ""
    # Positive aat_delta means backend arrival is later than Python arrival.
    # Positive rat_delta means backend required time is later than Python RAT.
    aat_delta = backend_aat - py_aat
    rat_delta = backend_rat - py_rat
    # Since slack = RAT - AAT, py_slack - backend_slack should equal
    # (py_rat - backend_rat) - (py_aat - backend_aat), or aat_delta - rat_delta.
    residual = slack_delta - (aat_delta - rat_delta)
    bucket["aat_delta_sum"] += aat_delta
    bucket["rat_delta_sum"] += rat_delta
    bucket["decomposition_residual_abs_sum"] += abs(residual)


def _print_delta_summary(compare_csv, debug_csv):
    debug_rows = _read_debug(debug_csv)
    buckets = defaultdict(_delta_bucket)
    with open(compare_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pin_name = row.get("pin_name") or ""
            normalized = row.get("normalized_pin_name") or _normalize_pin_name(pin_name)
            debug_row = debug_rows.get(normalized, {})
            cls = _check_class(debug_row)
            role = debug_row.get("endpoint_pin_role") or _pin_role(pin_name)
            for key in ("delta:all", f"delta:{cls}", f"delta:role:{role}"):
                _update_delta(buckets[key], row)
            if cls == "setup_exported":
                _update_delta(buckets["delta:objective:setup_exported"], row)
            if "RESET" in normalized.upper() or normalized.upper().endswith(":RST") or ":RST_" in normalized.upper():
                _update_delta(buckets["delta:name:reset"], row)

    print("delta_bucket,count,comparable,skipped_transition_mismatch,skipped_sentinel,mean_slack_delta_ps,mean_abs_slack_delta_ps,mean_backend_minus_python_aat_ps,mean_backend_minus_python_rat_ps,mean_abs_residual_ps,max_abs_slack_delta_ps,max_abs_slack_delta_pin")
    for key in sorted(buckets):
        bucket = buckets[key]
        count = bucket["count"]
        comparable = bucket["comparable"]
        mean_slack_delta = bucket["slack_delta_sum"] / comparable if comparable else 0.0
        mean_abs_slack_delta = bucket["abs_slack_delta_sum"] / comparable if comparable else 0.0
        mean_aat_delta = bucket["aat_delta_sum"] / comparable if comparable else 0.0
        mean_rat_delta = bucket["rat_delta_sum"] / comparable if comparable else 0.0
        mean_residual = bucket["decomposition_residual_abs_sum"] / comparable if comparable else 0.0
        print(
            f"{key},{count},{comparable},{bucket['skipped_transition_mismatch']},"
            f"{bucket['skipped_sentinel']},{mean_slack_delta:.6f},{mean_abs_slack_delta:.6f},"
            f"{mean_aat_delta:.6f},{mean_rat_delta:.6f},{mean_residual:.6f},"
            f"{bucket['max_abs_slack_delta']:.6f},{bucket['max_abs_slack_delta_pin']}"
        )


def summarize(slack_csv, debug_csv):
    debug_rows = _read_debug(debug_csv)
    buckets = defaultdict(_metric_bucket)
    with open(slack_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pin_name = row.get("pin_name") or ""
            normalized = row.get("normalized_pin_name") or _normalize_pin_name(pin_name)
            debug_row = debug_rows.get(normalized, {})
            cls = _check_class(debug_row)
            role = debug_row.get("endpoint_pin_role") or _pin_role(pin_name)
            slack = _float_or_none(row.get("py_min_slack_ps"))
            for key in ("all", cls, f"role:{role}"):
                _update_metric(buckets[key], slack)
            if cls == "setup_exported":
                _update_metric(buckets["objective:setup_exported"], slack)
            if "RESET" in normalized.upper() or normalized.upper().endswith(":RST") or ":RST_" in normalized.upper():
                _update_metric(buckets["name:reset"], slack)
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slack-csv", required=True)
    parser.add_argument("--debug-csv", required=True)
    parser.add_argument("--compare-csv")
    parser.add_argument("--include-deltas", action="store_true")
    args = parser.parse_args()

    buckets = summarize(args.slack_csv, args.debug_csv)
    print("bucket,count,wns_ps,tns_ps,negative")
    for key in sorted(buckets):
        bucket = buckets[key]
        wns = "" if bucket["wns"] is None else f"{bucket['wns']:.6f}"
        print(f"{key},{bucket['count']},{wns},{bucket['tns']:.6f},{bucket['negative']}")

    if args.compare_csv:
        debug_rows = _read_debug(args.debug_csv)
        compare_buckets = defaultdict(_metric_bucket)
        with open(args.compare_csv, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                pin_name = row.get("pin_name") or ""
                normalized = row.get("normalized_pin_name") or _normalize_pin_name(pin_name)
                debug_row = debug_rows.get(normalized, {})
                cls = _check_class(debug_row)
                role = debug_row.get("endpoint_pin_role") or _pin_role(pin_name)
                slack = _float_or_none(row.get("cpp_slack"))
                for key in ("backend:all", f"backend:{cls}", f"backend:role:{role}"):
                    _update_metric(compare_buckets[key], slack)
                if cls == "setup_exported":
                    _update_metric(compare_buckets["backend:objective:setup_exported"], slack)
                if "RESET" in normalized.upper() or normalized.upper().endswith(":RST") or ":RST_" in normalized.upper():
                    _update_metric(compare_buckets["backend:name:reset"], slack)

        for key in sorted(compare_buckets):
            bucket = compare_buckets[key]
            wns = "" if bucket["wns"] is None else f"{bucket['wns']:.6f}"
            print(f"{key},{bucket['count']},{wns},{bucket['tns']:.6f},{bucket['negative']}")

        if args.include_deltas:
            _print_delta_summary(args.compare_csv, args.debug_csv)


if __name__ == "__main__":
    main()
