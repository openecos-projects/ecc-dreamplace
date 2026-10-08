#!/usr/bin/env python3
"""Deterministic four-table aggregation for normalized campaign rows."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


TNS_TIE_TOLERANCE_NS = 0.001
PARETO_RELATIVE_TOLERANCE = 0.001
MATCHED_BASELINES = {
    "placement": "or_tdp_reweight_only",
    "buffering": "or_repair_design_buffer_only",
    "sizing": "or_pure_sizeup",
    "full_flow": "or_native_full_flow",
}
TABLE_FILENAMES = {
    "placement": "table1_placement",
    "buffering": "table2_buffering",
    "sizing": "table3_sizing",
    "full_flow": "table4_full_flow",
}
CSV_COLUMNS = (
    "case",
    "method_id",
    "status",
    "terminal_reason",
    "input_wns_ns",
    "input_tns_ns",
    "final_wns_ns",
    "final_tns_ns",
    "state_delta_tns_ns",
    "normalized_state_gain",
    "matched_baseline_gap_tns_ns",
    "normalized_baseline_advantage",
    "tns_result_vs_baseline",
    "raw_hpwl_dbu",
    "legalized_hpwl_dbu",
    "violating_endpoint_count",
    "resize_count",
    "inserted_buffer_count",
    "added_cell_area_um2",
    "total_cell_area_um2",
    "power_total_w",
    "slew_violation_count",
    "slew_violation_total",
    "cap_violation_count",
    "cap_violation_total",
    "core_optimizer_runtime_sec",
    "initialization_runtime_sec",
    "physical_transaction_runtime_sec",
    "legalization_runtime_sec",
    "track_a_runtime_sec",
    "end_to_end_runtime_sec",
    "peak_rss_kb",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "runtime_comparable_group",
    "source_artifact_path",
    "track_a_artifact_path",
    "attempt_id",
)


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_tns_gap(gap_ns: float | None) -> str | None:
    if gap_ns is None:
        return None
    if abs(gap_ns) <= TNS_TIE_TOLERANCE_NS:
        return "tie"
    return "win" if gap_ns > 0.0 else "loss"


def relative_pareto_change(value: Any, baseline: Any) -> float | None:
    value_float = _finite_float(value)
    baseline_float = _finite_float(baseline)
    if value_float is None or baseline_float is None or baseline_float == 0.0:
        return None
    change = (value_float - baseline_float) / abs(baseline_float)
    return 0.0 if abs(change) <= PARETO_RELATIVE_TOLERANCE else change


def enrich_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_list = [dict(row) for row in rows]
    index = {
        (row.get("table_id"), row.get("case"), row.get("method_id")): row
        for row in rows_list
    }
    if len(index) != len(rows_list):
        raise ValueError("normalized rows contain duplicate keys")
    enriched = []
    for row in rows_list:
        table_id = str(row["table_id"])
        case = str(row["case"])
        method_id = str(row["method_id"])
        final_metrics = dict(row.get("final_metrics") or {})
        input_metrics = dict(row.get("input_metrics") or {})
        mutation = dict(row.get("mutation") or {})
        runtime = dict(row.get("runtime") or {})
        input_tns = _finite_float(input_metrics.get("tns_ns"))
        final_tns = _finite_float(final_metrics.get("tns_ns"))
        state_delta = (
            None if input_tns is None or final_tns is None else final_tns - input_tns
        )
        normalized_state = (
            None
            if state_delta is None or input_tns == 0.0
            else state_delta / abs(input_tns)
        )
        baseline_id = MATCHED_BASELINES[table_id]
        baseline = index.get((table_id, case, baseline_id))
        baseline_tns = None
        if baseline is not None and baseline.get("status") == "pass":
            baseline_tns = _finite_float(
                dict(baseline.get("final_metrics") or {}).get("tns_ns")
            )
        baseline_gap = (
            None
            if final_tns is None or baseline_tns is None
            else final_tns - baseline_tns
        )
        normalized_baseline = (
            None
            if baseline_gap is None or input_tns in (None, 0.0)
            else baseline_gap / abs(input_tns)
        )
        enriched.append(
            {
                **row,
                "input_wns_ns": _finite_float(input_metrics.get("wns_ns")),
                "input_tns_ns": input_tns,
                "final_wns_ns": _finite_float(final_metrics.get("wns_ns")),
                "final_tns_ns": final_tns,
                "state_delta_tns_ns": state_delta,
                "normalized_state_gain": normalized_state,
                "matched_baseline_gap_tns_ns": baseline_gap,
                "normalized_baseline_advantage": normalized_baseline,
                "tns_result_vs_baseline": (
                    "baseline"
                    if method_id == baseline_id and row.get("status") == "pass"
                    else classify_tns_gap(baseline_gap)
                ),
                "raw_hpwl_dbu": final_metrics.get("raw_hpwl_dbu"),
                "legalized_hpwl_dbu": final_metrics.get("legalized_hpwl_dbu"),
                "violating_endpoint_count": final_metrics.get(
                    "violating_endpoint_count"
                ),
                "resize_count": mutation.get("resize_count"),
                "inserted_buffer_count": mutation.get("inserted_buffer_count"),
                "added_cell_area_um2": mutation.get("added_cell_area_um2"),
                "total_cell_area_um2": final_metrics.get("total_cell_area_um2"),
                "power_total_w": final_metrics.get("power_total_w"),
                "slew_violation_count": final_metrics.get("slew_violation_count"),
                "slew_violation_total": final_metrics.get("slew_violation_total"),
                "cap_violation_count": final_metrics.get("cap_violation_count"),
                "cap_violation_total": final_metrics.get("cap_violation_total"),
                "core_optimizer_runtime_sec": runtime.get(
                    "core_optimizer_runtime_sec"
                ),
                "initialization_runtime_sec": runtime.get(
                    "initialization_runtime_sec"
                ),
                "physical_transaction_runtime_sec": runtime.get(
                    "physical_transaction_runtime_sec"
                ),
                "legalization_runtime_sec": runtime.get(
                    "legalization_runtime_sec"
                ),
                "track_a_runtime_sec": runtime.get("track_a_runtime_sec"),
                "end_to_end_runtime_sec": runtime.get("end_to_end_runtime_sec"),
                "peak_rss_kb": runtime.get("peak_rss_kb"),
                "gpu_peak_allocated_bytes": runtime.get(
                    "gpu_peak_allocated_bytes"
                ),
                "gpu_peak_reserved_bytes": runtime.get(
                    "gpu_peak_reserved_bytes"
                ),
                "runtime_comparable_group": runtime.get(
                    "runtime_comparable_group"
                ),
                "source_artifact_path": dict(
                    row.get("source_artifact") or {}
                ).get("path"),
                "track_a_artifact_path": dict(
                    row.get("track_a_artifact") or {}
                ).get("path"),
            }
        )
    return sorted(
        enriched,
        key=lambda row: (row["table_id"], row["case"], row["method_id"]),
    )


def _flat_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in CSV_COLUMNS}


def _median(values: Iterable[Any]) -> float | None:
    finite = [
        value
        for value in (_finite_float(raw) for raw in values)
        if value is not None
    ]
    return None if not finite else float(statistics.median(finite))


def summarize_tables(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    enriched = enrich_rows(rows)
    index = {
        (row["table_id"], row["case"], row["method_id"]): row
        for row in enriched
    }
    tables: dict[str, Any] = {}
    for table_id in sorted({row["table_id"] for row in enriched}):
        table_rows = [row for row in enriched if row["table_id"] == table_id]
        methods = sorted({row["method_id"] for row in table_rows})
        baseline_id = MATCHED_BASELINES[table_id]
        method_summaries = {}
        for method_id in methods:
            method_rows = [
                row
                for row in enriched
                if row["table_id"] == table_id and row["method_id"] == method_id
            ]
            pass_rows = [row for row in method_rows if row.get("status") == "pass"]
            outcomes = [
                row.get("tns_result_vs_baseline")
                for row in pass_rows
                if row.get("tns_result_vs_baseline") in {"win", "tie", "loss"}
            ]
            runtime_speedups = []
            hpwl_changes = []
            area_changes = []
            power_changes = []
            for row in pass_rows:
                baseline = index.get((table_id, row["case"], baseline_id))
                if baseline is None or baseline.get("status") != "pass":
                    continue
                if (
                    row.get("runtime_comparable_group")
                    == baseline.get("runtime_comparable_group")
                ):
                    method_runtime = _finite_float(row.get("end_to_end_runtime_sec"))
                    baseline_runtime = _finite_float(
                        baseline.get("end_to_end_runtime_sec")
                    )
                    if (
                        method_runtime is not None
                        and baseline_runtime is not None
                        and method_runtime > 0.0
                        and baseline_runtime > 0.0
                    ):
                        runtime_speedups.append(baseline_runtime / method_runtime)
                hpwl_changes.append(
                    relative_pareto_change(
                        row.get("legalized_hpwl_dbu"),
                        baseline.get("legalized_hpwl_dbu"),
                    )
                )
                area_changes.append(
                    relative_pareto_change(
                        row.get("total_cell_area_um2"),
                        baseline.get("total_cell_area_um2"),
                    )
                )
                power_changes.append(
                    relative_pareto_change(
                        row.get("power_total_w"),
                        baseline.get("power_total_w"),
                    )
                )
            method_summaries[method_id] = {
                "row_count": len(method_rows),
                "pass_count": len(pass_rows),
                "failure_count": len(method_rows) - len(pass_rows),
                "tns_wins": outcomes.count("win"),
                "tns_ties": outcomes.count("tie"),
                "tns_losses": outcomes.count("loss"),
                "median_normalized_state_gain": _median(
                    row.get("normalized_state_gain") for row in pass_rows
                ),
                "median_normalized_baseline_advantage": _median(
                    row.get("normalized_baseline_advantage") for row in pass_rows
                ),
                "median_matched_end_to_end_speedup_baseline_over_method": _median(
                    runtime_speedups
                ),
                "median_relative_hpwl_change_vs_baseline": _median(hpwl_changes),
                "median_relative_area_change_vs_baseline": _median(area_changes),
                "median_relative_power_change_vs_baseline": _median(power_changes),
            }
        tables[table_id] = {
            "matched_baseline": baseline_id,
            "row_count": len(table_rows),
            "methods": method_summaries,
        }
    return {
        "artifact": "autodmp_tools_paper_aggregate_summary",
        "artifact_version": 1,
        "row_count": len(enriched),
        "tables": tables,
    }


def write_table_files(
    output_dir: Path,
    rows: Iterable[dict[str, Any]],
    *,
    table_order: Iterable[str],
    case_order: Iterable[str],
    method_order: dict[str, Iterable[str]],
) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = enrich_rows(rows)
    artifacts: dict[str, dict[str, str]] = {}
    for table_id in table_order:
        order = {method: index for index, method in enumerate(method_order[table_id])}
        case_rank = {case: index for index, case in enumerate(case_order)}
        table_rows = sorted(
            (row for row in enriched if row["table_id"] == table_id),
            key=lambda row: (case_rank[row["case"]], order[row["method_id"]]),
        )
        base = output_dir / TABLE_FILENAMES[table_id]
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        md_path = base.with_suffix(".md")
        json_path.write_text(
            json.dumps(table_rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(_flat_csv_row(row) for row in table_rows)
        markdown_columns = (
            "case",
            "method_id",
            "status",
            "final_wns_ns",
            "final_tns_ns",
            "state_delta_tns_ns",
            "tns_result_vs_baseline",
            "legalized_hpwl_dbu",
            "resize_count",
            "inserted_buffer_count",
            "added_cell_area_um2",
            "power_total_w",
            "end_to_end_runtime_sec",
            "source_artifact_path",
            "track_a_artifact_path",
            "terminal_reason",
        )
        lines = [
            f"# {TABLE_FILENAMES[table_id]}",
            "",
            "| " + " | ".join(markdown_columns) + " |",
            "| " + " | ".join("---" for _ in markdown_columns) + " |",
        ]
        for row in table_rows:
            lines.append(
                "| "
                + " | ".join(
                    "" if row.get(column) is None else str(row.get(column))
                    for column in markdown_columns
                )
                + " |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        artifacts[table_id] = {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }
    summary = summarize_tables(enriched)
    summary_json = output_dir / "aggregate_summary.json"
    summary_markdown = output_dir / "aggregate_summary.md"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = ["# Aggregate summary", ""]
    for table_id, table in summary["tables"].items():
        lines.extend(
            [
                f"## {table_id}",
                "",
                "| method_id | pass / rows | TNS W/T/L | median state gain | median baseline advantage | median matched E2E speedup | median HPWL change | median area change | median power change |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method_id, method in table["methods"].items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        method_id,
                        f"{method['pass_count']} / {method['row_count']}",
                        f"{method['tns_wins']}/{method['tns_ties']}/{method['tns_losses']}",
                        str(method["median_normalized_state_gain"]),
                        str(method["median_normalized_baseline_advantage"]),
                        str(method["median_matched_end_to_end_speedup_baseline_over_method"]),
                        str(method["median_relative_hpwl_change_vs_baseline"]),
                        str(method["median_relative_area_change_vs_baseline"]),
                        str(method["median_relative_power_change_vs_baseline"]),
                    )
                )
                + " |"
            )
        lines.append("")
    summary_markdown.write_text("\n".join(lines), encoding="utf-8")
    artifacts["aggregate_summary"] = {
        "json": str(summary_json),
        "markdown": str(summary_markdown),
    }
    return artifacts
