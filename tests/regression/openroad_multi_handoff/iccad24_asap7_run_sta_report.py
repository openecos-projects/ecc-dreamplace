import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import def_validation
import iccad24_asap7_buffer_only_runner as placement_runner


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_BASE = THIS_DIR / "artifacts" / "iccad24_asap7_run_sta_report"
STA_METRIC_PREFIX = "AUTODMP_STA_METRIC"
REQUIRED_STA_METRICS = (
    "gate_count",
    "wns_ns",
    "tns_ns",
    "total_leakage_uw",
)

MARKDOWN_COLUMNS = [
    ("casename", "casename"),
    ("design/case", "design"),
    ("baseline family", "baseline_family_display"),
    ("mode/config", "mode_config"),
    ("gate count", "gate_count"),
    ("WNS(ns)", "WNS(ns)"),
    ("TNS(ns)", "TNS(ns)"),
    ("total slew violation difference(ns)", "total_slew_violation_difference(ns)"),
    (
        "Total Load Capacitance Violation Difference(fF)",
        "Total Load Capacitance Violation Difference(fF)",
    ),
    ("Total Leakage(uw)", "Total Leakage(uw)"),
    ("run dir / artifact dir", "artifact_dir"),
    ("status/failure", "status"),
    ("failure", "failure"),
]

COMPARISON_MARKDOWN_COLUMNS = [
    ("casename", "casename"),
    ("baseline family", "baseline_family_display"),
    ("source", "source"),
    ("handoff mode", "handoff_mode"),
    ("target density", "target_density"),
    ("stop overflow", "stop_overflow"),
    ("max handoff overflow", "max_handoff_overflow"),
    ("with STA", "with_sta"),
    ("restart policy", "handoff_restart_policy"),
    ("final DEF validation", "final_def_validation_status"),
    ("handoff count", "handoff_count"),
    ("added buffers", "added_buffer_count"),
    ("removed buffers", "removed_buffer_count"),
    ("surviving buffers", "surviving_buffer_count"),
    ("status", "status"),
    ("WNS(ns)", "WNS(ns)"),
    ("TNS(ns)", "TNS(ns)"),
    ("total slew viol(ns)", "total_slew_violation_difference(ns)"),
    ("load cap viol(fF)", "Total Load Capacitance Violation Difference(fF)"),
    ("leakage(uw)", "Total Leakage(uw)"),
    ("gate count", "gate_count"),
]

BEST_HIGHER_IS_BETTER = {"WNS(ns)", "TNS(ns)"}
BEST_LOWER_IS_BETTER = {
    "total_slew_violation_difference(ns)",
    "Total Load Capacitance Violation Difference(fF)",
    "Total Leakage(uw)",
    "gate_count",
}

DELTA_METRICS = (
    "WNS(ns)",
    "TNS(ns)",
    "total_slew_violation_difference(ns)",
    "Total Load Capacitance Violation Difference(fF)",
    "Total Leakage(uw)",
    "gate_count",
)

VIOLATION_ROW_RE = re.compile(
    r"\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\(VIOLATED\)\s*$"
)
LIBERTY_TIME_UNIT_RE = re.compile(r'time_unit\s*:\s*"([^"]+)"')
MODE_CONFIG_VALUE_RE_TEMPLATE = r"(?:^|_)%s([^_]+)"
TIMING_METRIC_KEYS = {
    "wns_ns",
    "tns_ns",
}
TIME_UNIT_TO_NS = {
    "s": 1.0e9,
    "1s": 1.0e9,
    "ms": 1.0e6,
    "1ms": 1.0e6,
    "us": 1.0e3,
    "1us": 1.0e3,
    "ns": 1.0,
    "1ns": 1.0,
    "ps": 1.0e-3,
    "1ps": 1.0e-3,
    "fs": 1.0e-6,
    "1fs": 1.0e-6,
}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def read_text_if_exists(path):
    path = Path(path)
    return path.read_text(errors="replace") if path.exists() else ""


def format_label_value(value):
    if value is None:
        return "Native"
    text = "%.12g" % float(value)
    return text.replace("-", "m").replace(".", "p")


def expand_variants(
    handoff_modes,
    target_density_overrides,
    stop_overflow_overrides,
    with_sta_values,
    max_handoff_overflows,
    handoff_restart_policies=None,
):
    handoff_restart_policies = handoff_restart_policies or ["cold"]
    variants = []
    for (
        handoff_mode,
        target_density,
        stop_overflow,
        with_sta,
        max_handoff_overflow,
        handoff_restart_policy,
    ) in itertools.product(
        handoff_modes,
        target_density_overrides,
        stop_overflow_overrides,
        with_sta_values,
        max_handoff_overflows,
        handoff_restart_policies,
    ):
        label = "%s_td%s_so%s_sta%d_mho%s" % (
            handoff_mode.replace("-", "_"),
            format_label_value(target_density),
            format_label_value(stop_overflow),
            1 if with_sta else 0,
            format_label_value(max_handoff_overflow),
        )
        if handoff_restart_policy != "cold":
            label += "_rst%s" % handoff_restart_policy
        variants.append(
            {
                "label": label,
                "handoff_mode": handoff_mode,
                "handoff_restart_policy": handoff_restart_policy,
                "target_density_override": target_density,
                "stop_overflow_override": stop_overflow,
                "with_sta": bool(with_sta),
                "max_handoff_overflow": max_handoff_overflow,
            }
        )
    return variants


def placement_failure_text(summary):
    failure = summary.get("failure")
    if not failure:
        return ""
    if isinstance(failure, str):
        return failure
    return "%s: %s" % (failure.get("type"), failure.get("message"))


def build_report_row(placement_summary, variant_label, sta_result=None):
    sta_result = sta_result or {}
    config = placement_summary.get("config", {}) or {}
    overrides = config.get("diagnostic_param_overrides", {}) or {}
    mode_config = variant_label
    if config.get("preserve_routability_opt") and "_roptnative" not in mode_config:
        mode_config += "_roptnative"
    failure = placement_failure_text(placement_summary)
    final_def_validation = placement_summary.get("final_def_validation", {}) or {}
    handoff_summary = placement_summary.get("handoff_summary", {}) or {}
    session_summary = placement_summary.get("session", {}) or {}
    session_churn = session_summary.get("buffer_churn_totals", {}) or {}
    source = placement_summary.get("source")
    if source is None:
        source = (
            "warm"
            if config.get("handoff_restart_policy") == "warm_schedule"
            else "cold"
        )
    if failure:
        status = "placement_failed"
    elif final_def_validation.get("status") == "failed":
        status = "skipped_invalid_final_def"
        failure = final_def_validation.get("failure", "")
    elif not placement_summary.get("final_def_exists"):
        status = "skipped_no_final_def"
        failure = "final DEF was not produced"
    elif final_def_validation.get("status") != "passed":
        status = "skipped_unvalidated_final_def"
        failure = "final DEF validation status is %s" % (
            final_def_validation.get("status") or "missing"
        )
    else:
        status = sta_result.get("status", "sta_pending")
        failure = sta_result.get("failure", "")
    metrics = sta_result.get("metrics", {}) or {}
    return {
        "design": placement_summary.get("design"),
        "mode_config": mode_config,
        "source": source,
        "baseline_family": None,
        "handoff_mode": placement_summary.get("handoff_mode"),
        "handoff_restart_policy": config.get("handoff_restart_policy", "cold"),
        "with_sta": bool(placement_summary.get("with_sta")),
        "target_density": overrides.get("target_density"),
        "stop_overflow": overrides.get("stop_overflow"),
        "max_handoff_overflow": placement_summary.get("max_handoff_overflow"),
        "final_def_validation_status": final_def_validation.get("status"),
        "final_def": placement_summary.get("final_def"),
        "sta_log_path": sta_result.get("log_path"),
        "handoff_count": handoff_summary.get(
            "handoff_count",
            session_summary.get("event_count"),
        ),
        "added_buffer_count": handoff_summary.get(
            "added_buffer_count",
            session_churn.get("added_buffer_count"),
        ),
        "removed_buffer_count": handoff_summary.get(
            "removed_buffer_count",
            session_churn.get("removed_buffer_count"),
        ),
        "surviving_buffer_count": handoff_summary.get(
            "surviving_buffer_count",
            session_churn.get("surviving_buffer_count"),
        ),
        "gate_count": metrics.get("gate_count"),
        "WNS(ns)": metrics.get("wns_ns"),
        "TNS(ns)": metrics.get("tns_ns"),
        "total_slew_violation_difference(ns)": metrics.get(
            "total_slew_violation_difference_ns"
        ),
        "Total Load Capacitance Violation Difference(fF)": metrics.get(
            "total_load_capacitance_violation_difference_ff"
        ),
        "Total Leakage(uw)": metrics.get("total_leakage_uw"),
        "run_dir": placement_summary.get("run_dir"),
        "artifact_dir": sta_result.get("artifact_dir")
        or placement_summary.get("run_dir"),
        "status": status,
        "failure": failure,
    }


def baseline_family_from_row(row):
    source = row.get("source")
    mode_config = row.get("mode_config") or ""
    restart_policy = row.get("handoff_restart_policy")
    handoff_mode = row.get("handoff_mode")
    if source == "original":
        return "original_def_opensta"
    if source == "original_repaired":
        return "original_def_openroad_repair_opensta"
    if "no_handoff_repair" in mode_config:
        return "autodmp_no_handoff_openroad_repair_opensta"
    if handoff_mode == "no-handoff" or mode_config.startswith("no_handoff"):
        return "autodmp_no_handoff_opensta"
    if source == "warm" or restart_policy == "warm_schedule" or "_rstwarm_schedule" in mode_config:
        return "autodmp_handoff_warm_opensta"
    return "autodmp_handoff_cold_opensta"


def baseline_family_display_from_key(key):
    mapping = {
        "original_def_opensta": "Original_DEF",
        "original_def_openroad_repair_opensta": "Original_DEF + OpenROAD Repair",
        "autodmp_no_handoff_opensta": "AutoDMP + No_Handoff",
        "autodmp_no_handoff_openroad_repair_opensta": "AutoDMP + No_Handoff + OpenROAD Repair",
        "autodmp_handoff_cold_opensta": "AutoDMP + Handoff + Cold",
        "autodmp_handoff_warm_opensta": "AutoDMP + Handoff + Warm",
    }
    return mapping.get(key, key)


def parse_label_float(value):
    if value in (None, ""):
        return None
    text = str(value)
    if text == "Native":
        return None
    sign = -1 if text.startswith("m") else 1
    if sign < 0:
        text = text[1:]
    return sign * float(text.replace("p", "."))


def mode_config_float_value(mode_config, key):
    match = re.search(MODE_CONFIG_VALUE_RE_TEMPLATE % re.escape(key), mode_config or "")
    if not match:
        return None
    return parse_label_float(match.group(1))


def fill_missing_config_columns(row):
    mode_config = row.get("mode_config") or ""
    if row.get("max_handoff_overflow") in (None, ""):
        row["max_handoff_overflow"] = mode_config_float_value(mode_config, "mho")
    return row


def _row_metric(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _density_key(row):
    value = row.get("target_density")
    if value in (None, ""):
        return None
    return "%.12g" % float(value)


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return current - baseline


def add_metric_deltas(rows):
    enriched = [dict(row) for row in rows]
    original_by_design = {}
    original_repaired_by_design = {}
    no_handoff_by_design_density = {}
    for row in enriched:
        fill_missing_config_columns(row)
        row["casename"] = row.get("design")
        row["baseline_family"] = baseline_family_from_row(row)
        row["baseline_family_display"] = baseline_family_display_from_key(
            row["baseline_family"]
        )
        if row.get("status") != "sta_completed":
            continue
        if row["baseline_family"] == "original_def_opensta":
            original_by_design[row.get("design")] = row
        if row["baseline_family"] == "original_def_openroad_repair_opensta":
            original_repaired_by_design[row.get("design")] = row
        if row["baseline_family"] == "autodmp_no_handoff_opensta":
            no_handoff_by_design_density[(row.get("design"), _density_key(row))] = row

    for row in enriched:
        original = original_by_design.get(row.get("design"))
        original_repaired = original_repaired_by_design.get(row.get("design"))
        same_density_no_handoff = no_handoff_by_design_density.get(
            (row.get("design"), _density_key(row))
        )
        for metric in DELTA_METRICS:
            current_value = _row_metric(row, metric)
            original_value = _row_metric(original or {}, metric)
            no_handoff_value = _row_metric(same_density_no_handoff or {}, metric)
            row["delta_vs_original_%s" % metric] = _delta(
                current_value,
                original_value,
            )
            row["delta_vs_same_density_no_handoff_%s" % metric] = _delta(
                current_value,
                no_handoff_value,
            )
            row["delta_vs_original_repaired_%s" % metric] = _delta(
                current_value,
                _row_metric(original_repaired or {}, metric),
            )
    return enriched


def tcl_quote(path):
    return "{%s}" % str(path).replace("}", "\\}")


def absolute_path(path):
    return str(Path(path).expanduser().resolve())


def generate_sta_tcl(design, tech_bundle, final_def, sdc_path, sta_dir):
    del design
    sta_dir = Path(absolute_path(sta_dir))
    cell_usage = sta_dir / "cell_usage.rpt"
    power = sta_dir / "power.rpt"
    slew = sta_dir / "max_slew_violators.rpt"
    cap = sta_dir / "max_capacitance_violators.rpt"
    lines = [
        "read_lef %s" % tcl_quote(tech_bundle["tech_lef"]),
    ]
    for lef_path in tech_bundle.get("lef_paths", []):
        lines.append("read_lef %s" % tcl_quote(lef_path))
    for lib_path in tech_bundle.get("lib_paths", []):
        lines.append("read_liberty %s" % tcl_quote(lib_path))
    lines.extend(
        [
            "read_def %s" % tcl_quote(absolute_path(final_def)),
            "read_sdc %s" % tcl_quote(absolute_path(sdc_path)),
            "set_cmd_units -time ns -capacitance fF",
            "source %s" % tcl_quote(tech_bundle["rc_tcl"]),
            "estimate_parasitics -placement",
            "report_cell_usage -file %s" % tcl_quote(cell_usage),
            "set gate_count [sta::network_instance_count]",
            'puts "AUTODMP_STA_METRIC gate_count $gate_count count"',
            "set wns [sta::worst_slack -max]",
            'puts [format "AUTODMP_STA_METRIC wns_ns %.12g ns" $wns]',
            "set tns [sta::total_negative_slack -max]",
            'puts [format "AUTODMP_STA_METRIC tns_ns %.12g ns" $tns]',
            "report_check_types -max_slew -violators -digits 6 > %s"
            % tcl_quote(slew),
            "report_check_types -max_capacitance -violators -digits 6 > %s"
            % tcl_quote(cap),
            "report_power > %s" % tcl_quote(power),
        ]
    )
    return "\n".join(lines) + "\n"


def original_def_repair_command(margin):
    return (
        "repair_timing -setup "
        "-setup_margin %.12g "
        '-sequence "unbuffer,buffer,split" '
        "-skip_last_gasp -skip_pin_swap -skip_gate_cloning -skip_size_down"
    ) % margin


def generate_original_def_repair_tcl(
    design,
    tech_bundle,
    input_def,
    sdc_path,
    output_def,
    repair_dir,
    margin,
):
    del design, repair_dir
    lines = [
        "read_lef %s" % tcl_quote(tech_bundle["tech_lef"]),
    ]
    for lef_path in tech_bundle.get("lef_paths", []):
        lines.append("read_lef %s" % tcl_quote(lef_path))
    for lib_path in tech_bundle.get("lib_paths", []):
        lines.append("read_liberty %s" % tcl_quote(lib_path))
    lines.extend(
        [
            "read_def %s" % tcl_quote(absolute_path(input_def)),
            "read_sdc %s" % tcl_quote(absolute_path(sdc_path)),
            "set_cmd_units -time ns -capacitance fF",
            "source %s" % tcl_quote(tech_bundle["rc_tcl"]),
            "estimate_parasitics -placement",
            original_def_repair_command(margin),
            "write_def %s" % tcl_quote(absolute_path(output_def)),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_numeric(value):
    if value in (None, ""):
        return None
    number = float(value)
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def normalize_time_unit(unit):
    if not unit:
        return "ns"
    return str(unit).strip().strip('"').lower().replace(" ", "")


def time_unit_to_ns_scale(unit):
    normalized = normalize_time_unit(unit)
    if normalized in TIME_UNIT_TO_NS:
        return TIME_UNIT_TO_NS[normalized]
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(fs|ps|ns|us|ms|s)", normalized)
    if not match:
        return 1.0
    return float(match.group(1)) * TIME_UNIT_TO_NS["1%s" % match.group(2)]


def detect_liberty_time_unit(lib_paths):
    for lib_path in lib_paths or []:
        text = read_text_if_exists(lib_path)
        match = LIBERTY_TIME_UNIT_RE.search(text)
        if match:
            return normalize_time_unit(match.group(1))
    return "ns"


def convert_time_value_to_ns(value, unit):
    if value is None:
        return None
    return value * time_unit_to_ns_scale(unit)


def parse_sta_metric_lines(text):
    metrics = {}
    for line in text.splitlines():
        if not line.startswith(STA_METRIC_PREFIX + " "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        key = parts[1]
        value = parse_numeric(parts[2])
        if key in TIMING_METRIC_KEYS:
            unit = parts[3]
            if key.endswith("_ns") and normalize_time_unit(unit) in ("ps", "1ps"):
                unit = "ns"
            value = convert_time_value_to_ns(value, unit)
        metrics[key] = value
    return metrics


def sum_violation_report_slacks(text, time_unit="ns"):
    total = 0.0
    for line in text.splitlines():
        match = VIOLATION_ROW_RE.search(line)
        if match:
            slack = float(match.group(1))
            if slack < 0.0:
                total += abs(slack)
    return convert_time_value_to_ns(total, time_unit)


def parse_total_leakage_uw(text):
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "Total":
            return float(parts[3]) * 1e6
    return None


def run_sta_for_summary(placement_summary, sta_base, openroad_bin, variant_label):
    design = placement_summary.get("design", "unknown")
    sta_dir = Path(sta_base) / variant_label / design
    sta_dir.mkdir(parents=True, exist_ok=True)
    tcl_path = sta_dir / "run_openroad_sta.tcl"
    log_path = sta_dir / "openroad_sta.log"
    metrics_path = sta_dir / "sta_metrics.json"
    tcl_text = generate_sta_tcl(
        design=design,
        tech_bundle=placement_summary["tech_bundle"],
        final_def=placement_summary["final_def"],
        sdc_path=placement_summary["case"]["sdc_path"],
        sta_dir=sta_dir,
    )
    tcl_path.write_text(tcl_text)
    try:
        completed = subprocess.run(
            [openroad_bin, str(tcl_path.resolve())],
            cwd=str(sta_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        result = {
            "status": "sta_failed",
            "failure": "openroad binary not found: %s" % openroad_bin,
            "metrics": {},
            "artifact_dir": str(sta_dir),
            "tcl_path": str(tcl_path),
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "returncode": None,
        }
        log_path.write_text(str(exc))
        write_json(metrics_path, result)
        return result
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""))
    metrics = parse_sta_metric_lines(completed.stdout or "")
    metrics["total_slew_violation_difference_ns"] = sum_violation_report_slacks(
        read_text_if_exists(sta_dir / "max_slew_violators.rpt"),
        time_unit="ns",
    )
    metrics["total_load_capacitance_violation_difference_ff"] = (
        sum_violation_report_slacks(
            read_text_if_exists(sta_dir / "max_capacitance_violators.rpt")
        )
    )
    total_leakage_uw = parse_total_leakage_uw(read_text_if_exists(sta_dir / "power.rpt"))
    if total_leakage_uw is not None:
        metrics["total_leakage_uw"] = total_leakage_uw
    parse_warnings = []
    missing_metrics = [
        metric for metric in REQUIRED_STA_METRICS if metrics.get(metric) is None
    ]
    if completed.returncode == 0 and missing_metrics:
        parse_warnings.append(
            "missing STA metrics: %s" % ", ".join(missing_metrics)
        )
    if completed.returncode != 0:
        status = "sta_failed"
        failure = "openroad exit %d" % completed.returncode
    elif parse_warnings:
        status = "sta_parse_failed"
        failure = "; ".join(parse_warnings)
    else:
        status = "sta_completed"
        failure = ""
    result = {
        "status": status,
        "failure": failure,
        "parse_warnings": parse_warnings,
        "metrics": metrics,
        "artifact_dir": str(sta_dir),
        "tcl_path": str(tcl_path),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "returncode": completed.returncode,
    }
    write_json(metrics_path, result)
    return result


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.6g" % value
    return str(value).replace("\n", " ")


def render_markdown_table(rows):
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        if not normalized.get("casename"):
            normalized["casename"] = normalized.get("design")
        if not normalized.get("baseline_family"):
            normalized["baseline_family"] = baseline_family_from_row(normalized)
        if not normalized.get("baseline_family_display"):
            normalized["baseline_family_display"] = baseline_family_display_from_key(
                normalized.get("baseline_family")
            )
        normalized_rows.append(normalized)
    headers = [header for header, _ in MARKDOWN_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in normalized_rows:
        lines.append(
            "| "
            + " | ".join(format_cell(row.get(key)) for _, key in MARKDOWN_COLUMNS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _best_values(rows):
    best = {}
    for metric in BEST_HIGHER_IS_BETTER | BEST_LOWER_IS_BETTER:
        values = [
            _row_metric(row, metric)
            for row in rows
            if row.get("status") == "sta_completed" and _row_metric(row, metric) is not None
        ]
        if not values:
            continue
        best[metric] = max(values) if metric in BEST_HIGHER_IS_BETTER else min(values)
    return best


def _format_comparison_cell(row, key, best):
    value = row.get(key)
    text = format_cell(value)
    if key in best:
        numeric_value = _row_metric(row, key)
        if numeric_value is not None and abs(numeric_value - best[key]) <= 1e-12:
            return "**%s**" % text
    return text


def render_grouped_comparison_markdown(rows):
    rows = add_metric_deltas(rows)
    by_design = {}
    for row in rows:
        by_design.setdefault(row.get("design") or "unknown", []).append(row)
    lines = ["# ICCAD24 OpenSTA Comparison", ""]
    for design in sorted(by_design):
        design_rows = by_design[design]
        best = _best_values(design_rows)
        lines.append("## %s" % design)
        lines.append("")
        headers = [header for header, _ in COMPARISON_MARKDOWN_COLUMNS]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in design_rows:
            lines.append(
                "| "
                + " | ".join(
                    _format_comparison_cell(row, key, best)
                    for _, key in COMPARISON_MARKDOWN_COLUMNS
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_rows_csv(rows):
    rows = add_metric_deltas(rows)
    priority = ["casename", "design", "baseline_family", "baseline_family_display"]
    fieldnames = [key for key in priority if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = []

    class _Writer:
        def write(self, text):
            output.append(text)

    writer = csv.DictWriter(_Writer(), fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return "".join(output)


def parse_optional_float_values(values):
    if not values:
        return [None]
    result = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if not text:
                continue
            if text.lower() in ("none", "native", "default"):
                result.append(None)
            else:
                result.append(float(text))
    return result


def parse_bool_values(values):
    if not values:
        return [False]
    result = []
    for value in values:
        text = str(value).strip().lower()
        result.append(text in ("1", "true", "yes", "on", "sta1"))
    return result


def make_run_dir(artifact_base, no_timestamp=False):
    artifact_base = Path(artifact_base)
    if no_timestamp:
        artifact_base.mkdir(parents=True, exist_ok=True)
        return artifact_base
    run_dir = artifact_base / time.strftime("%Y%m%d_%H%M%S")
    if run_dir.exists():
        run_dir = Path("%s_%d" % (run_dir, os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dry_run_placement_summary(design, variant, variant_artifact_base):
    overrides = {
        key: value
        for key, value in {
            "target_density": variant["target_density_override"],
            "stop_overflow": variant["stop_overflow_override"],
        }.items()
        if value is not None
    }
    return {
        "design": design,
        "handoff_mode": variant["handoff_mode"],
        "handoff_restart_policy": variant.get("handoff_restart_policy", "cold"),
        "with_sta": variant["with_sta"],
        "max_handoff_overflow": variant["max_handoff_overflow"],
        "failure": None,
        "final_def_exists": False,
        "run_dir": str(variant_artifact_base / design),
        "config": {
            "diagnostic_param_overrides": overrides,
            "handoff_restart_policy": variant.get("handoff_restart_policy", "cold"),
        },
    }


def build_original_def_summary(case, tech_bundle, artifact_dir):
    final_def = case["def_input"]
    final_def_validation = {"status": "missing", "failure": "final DEF is missing"}
    if Path(final_def).exists():
        try:
            final_def_validation = def_validation.validate_final_def_coordinates(final_def)
            final_def_validation = dict(final_def_validation)
            final_def_validation["status"] = "passed"
        except Exception as exc:
            final_def_validation = {"status": "failed", "failure": str(exc)}
    return {
        "design": case["design"],
        "case": case,
        "tech_bundle": tech_bundle,
        "handoff_mode": "original",
        "source": "original",
        "handoff_restart_policy": "cold",
        "with_sta": False,
        "max_handoff_overflow": None,
        "failure": None,
        "final_def_exists": Path(final_def).exists(),
        "final_def": str(final_def),
        "run_dir": str(artifact_dir),
        "final_def_validation": final_def_validation,
        "config": {
            "diagnostic_param_overrides": {},
            "handoff_restart_policy": "cold",
        },
    }


def _validate_existing_final_def(final_def):
    if not Path(final_def).exists():
        return {"status": "missing", "failure": "final DEF is missing"}
    try:
        summary = def_validation.validate_final_def_coordinates(final_def)
        summary = dict(summary)
        summary["status"] = "passed"
        return summary
    except Exception as exc:
        return {"status": "failed", "failure": str(exc)}


def _openroad_error_from_log(text):
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[ERROR ") or stripped.startswith("Error:"):
            return stripped
    return None


def run_original_def_repair_summary(
    case,
    tech_bundle,
    artifact_dir,
    openroad_bin,
    margin=0.0,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    design = case["design"]
    repaired_def = artifact_dir / ("%s.original_repaired.def" % design)
    tcl_path = artifact_dir / "repair_original_def.tcl"
    log_path = artifact_dir / "openroad_repair.log"
    summary_path = artifact_dir / "repair_summary.json"
    repair_command = original_def_repair_command(margin)
    tcl_path.write_text(
        generate_original_def_repair_tcl(
            design=design,
            tech_bundle=tech_bundle,
            input_def=case["def_input"],
            sdc_path=case["sdc_path"],
            output_def=repaired_def,
            repair_dir=artifact_dir,
            margin=margin,
        )
    )
    failure = None
    returncode = None
    try:
        completed = subprocess.run(
            [openroad_bin, str(tcl_path.resolve())],
            cwd=str(artifact_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        returncode = completed.returncode
        log_text = (completed.stdout or "") + (completed.stderr or "")
        log_path.write_text(log_text)
        if completed.returncode != 0:
            failure = {
                "type": "OpenROADRepairFailed",
                "message": "openroad exit %d" % completed.returncode,
            }
        else:
            openroad_error = _openroad_error_from_log(log_text)
            if openroad_error:
                failure = {
                    "type": "OpenROADRepairFailed",
                    "message": openroad_error,
                }
    except FileNotFoundError as exc:
        log_path.write_text(str(exc))
        failure = {
            "type": "FileNotFoundError",
            "message": "openroad binary not found: %s" % openroad_bin,
        }
    summary = {
        "design": design,
        "case": case,
        "tech_bundle": tech_bundle,
        "handoff_mode": "original_repair",
        "source": "original_repaired",
        "handoff_restart_policy": "cold",
        "with_sta": False,
        "max_handoff_overflow": None,
        "failure": failure,
        "final_def_exists": repaired_def.exists(),
        "final_def": str(repaired_def),
        "run_dir": str(artifact_dir),
        "repair_command": repair_command,
        "repair_tcl_path": str(tcl_path),
        "repair_log_path": str(log_path),
        "repair_summary_path": str(summary_path),
        "repair_returncode": returncode,
        "final_def_validation": _validate_existing_final_def(repaired_def),
        "config": {
            "diagnostic_param_overrides": {},
            "handoff_restart_policy": "cold",
        },
    }
    write_json(summary_path, summary)
    return summary


def run_variant(variant, designs, benchmark_root, run_dir, args):
    placement_summaries = []
    sta_results = []
    rows = []
    variant_artifact_base = run_dir / "placement" / variant["label"]
    for design in designs:
        if args.dry_run:
            placement_summary = dry_run_placement_summary(
                design,
                variant,
                variant_artifact_base,
            )
            sta_result = None
        else:
            placement_summary = placement_runner.run_one(
                design,
                benchmark_root=benchmark_root,
                artifact_base=variant_artifact_base,
                target_iter=args.target_iter,
                trigger_period=args.trigger_period,
                margin=args.margin,
                plot_interval=args.plot_interval,
                with_sta=variant["with_sta"],
                handoff_mode=variant["handoff_mode"],
                handoff_restart_policy=variant.get("handoff_restart_policy", "cold"),
                target_density_override=variant["target_density_override"],
                stop_overflow_override=variant["stop_overflow_override"],
                max_handoff_overflow=variant["max_handoff_overflow"],
                preserve_routability_opt=args.preserve_routability_opt,
            )
            if (
                placement_summary.get("failure")
                or not placement_summary.get("final_def_exists")
                or (
                    (placement_summary.get("final_def_validation", {}) or {}).get(
                        "status"
                    )
                    != "passed"
                )
            ):
                sta_result = None
            else:
                sta_result = run_sta_for_summary(
                    placement_summary,
                    run_dir / "sta",
                    openroad_bin=args.openroad_bin,
                    variant_label=variant["label"],
                )
        placement_summaries.append(placement_summary)
        if sta_result:
            sta_results.append(sta_result)
        rows.append(build_report_row(placement_summary, variant["label"], sta_result))
    return {
        "variant": variant,
        "placement_summaries": placement_summaries,
        "sta_results": sta_results,
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=placement_runner.DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument("--artifact-base", type=Path, default=DEFAULT_ARTIFACT_BASE)
    parser.add_argument("--design", dest="designs", action="append")
    parser.add_argument(
        "--handoff-mode",
        choices=placement_runner.HANDOFF_MODES,
        action="append",
    )
    parser.add_argument(
        "--handoff-restart-policy",
        dest="handoff_restart_policies",
        action="append",
        choices=placement_runner.HANDOFF_RESTART_POLICIES,
        default=None,
    )
    parser.add_argument("--target-density-override", action="append")
    parser.add_argument("--stop-overflow-override", action="append")
    parser.add_argument("--max-handoff-overflow", action="append")
    parser.add_argument("--with-sta-value", action="append")
    parser.add_argument("--with-sta", dest="with_sta_value", action="append_const", const="true")
    parser.add_argument("--target-iter", type=int)
    parser.add_argument("--trigger-period", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--plot-interval", type=int)
    parser.add_argument("--openroad-bin", default="openroad")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--preserve-routability-opt", action="store_true")
    parser.add_argument("--include-original-def-baseline", action="store_true")
    parser.add_argument("--include-original-def-repair-baseline", action="store_true")
    args = parser.parse_args(argv)

    requested_cases = placement_runner.parse_csv(args.designs or [])
    if args.dry_run:
        designs = requested_cases or list(placement_runner.CASE_ORDER)
        manifest = {"cases": [{"design": design} for design in designs]}
    else:
        manifest = placement_runner.build_manifest(
            benchmark_root=args.benchmark_root,
            requested_cases=requested_cases or None,
        )
        designs = [case["design"] for case in manifest["cases"]]

    variants = expand_variants(
        handoff_modes=args.handoff_mode or ["buffer-only"],
        target_density_overrides=parse_optional_float_values(
            args.target_density_override
        ),
        stop_overflow_overrides=parse_optional_float_values(
            args.stop_overflow_override
        ),
        with_sta_values=parse_bool_values(args.with_sta_value),
        max_handoff_overflows=parse_optional_float_values(args.max_handoff_overflow)
        if args.max_handoff_overflow
        else [0.2],
        handoff_restart_policies=args.handoff_restart_policies,
    )
    run_dir = make_run_dir(args.artifact_base, no_timestamp=args.no_timestamp)
    variant_results = []
    rows = []
    if args.include_original_def_baseline:
        if args.dry_run:
            original_rows = [
                {
                    "design": design,
                    "source": "original",
                    "mode_config": "original_def",
                    "handoff_mode": "original",
                    "handoff_restart_policy": "cold",
                    "with_sta": False,
                    "target_density": None,
                    "stop_overflow": None,
                    "max_handoff_overflow": None,
                    "status": "sta_pending",
                    "failure": "",
                }
                for design in designs
            ]
            rows.extend(original_rows)
        else:
            tech_bundle = placement_runner.asap7_tech_bundle(args.benchmark_root / "ASAP7")
            original_base = run_dir / "original_def"
            for case in manifest["cases"]:
                placement_summary = build_original_def_summary(
                    case,
                    tech_bundle,
                    original_base / case["design"],
                )
                if (
                    placement_summary.get("failure")
                    or not placement_summary.get("final_def_exists")
                    or (
                        (placement_summary.get("final_def_validation", {}) or {}).get(
                            "status"
                        )
                        != "passed"
                    )
                ):
                    sta_result = None
                else:
                    sta_result = run_sta_for_summary(
                        placement_summary,
                        run_dir / "sta",
                        openroad_bin=args.openroad_bin,
                        variant_label="original_def",
                    )
                rows.append(
                    build_report_row(
                        placement_summary,
                        "original_def",
                        sta_result,
                    )
                )
    if args.include_original_def_repair_baseline:
        if args.dry_run:
            original_repair_rows = [
                {
                    "design": design,
                    "source": "original_repaired",
                    "mode_config": "original_def_repair_timing",
                    "handoff_mode": "original_repair",
                    "handoff_restart_policy": "cold",
                    "with_sta": False,
                    "target_density": None,
                    "stop_overflow": None,
                    "max_handoff_overflow": None,
                    "status": "sta_pending",
                    "failure": "",
                }
                for design in designs
            ]
            rows.extend(original_repair_rows)
        else:
            tech_bundle = placement_runner.asap7_tech_bundle(args.benchmark_root / "ASAP7")
            original_repair_base = run_dir / "original_def_repair"
            for case in manifest["cases"]:
                placement_summary = run_original_def_repair_summary(
                    case,
                    tech_bundle,
                    artifact_dir=original_repair_base / case["design"],
                    openroad_bin=args.openroad_bin,
                    margin=args.margin,
                )
                if (
                    placement_summary.get("failure")
                    or not placement_summary.get("final_def_exists")
                    or (
                        (placement_summary.get("final_def_validation", {}) or {}).get(
                            "status"
                        )
                        != "passed"
                    )
                ):
                    sta_result = None
                else:
                    sta_result = run_sta_for_summary(
                        placement_summary,
                        run_dir / "sta",
                        openroad_bin=args.openroad_bin,
                        variant_label="original_def_repair_timing",
                    )
                rows.append(
                    build_report_row(
                        placement_summary,
                        "original_def_repair_timing",
                        sta_result,
                    )
                )
    for variant in variants:
        result = run_variant(variant, designs, args.benchmark_root, run_dir, args)
        variant_results.append(result)
        rows.extend(result["rows"])
        enriched_rows = add_metric_deltas(rows)
        write_json(
            run_dir / "run_sta_summary.json",
            {
                "manifest": manifest,
                "variants": variant_results,
                "rows": rows,
                "enriched_rows": enriched_rows,
            },
        )
        (run_dir / "run_sta_summary.md").write_text(render_markdown_table(rows))
        (run_dir / "run_sta_summary.csv").write_text(render_rows_csv(enriched_rows))
        (run_dir / "comparison_summary.md").write_text(
            render_grouped_comparison_markdown(enriched_rows)
        )
    write_json(run_dir / "run_manifest.json", {"manifest": manifest, "variants": variants})
    print("run_sta_summary_md=%s" % (run_dir / "run_sta_summary.md"))
    print("run_sta_summary_json=%s" % (run_dir / "run_sta_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
