#!/usr/bin/env python3
"""Canonical DEF-only OpenROAD/OpenSTA evaluation for regression campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from placement_validation import (
    DETAILED_PLACEMENT_SEARCH_WINDOWS,
    detailed_placement_tcl_lines,
    inventory_tcl_proc_lines,
    macro_halo_blockage_tcl_proc_lines,
    qualify_placement,
    read_inventory,
)


ARTIFACT_KIND = "canonical_def_only_track_a"
INPUT_STATE_ARTIFACT_KIND = "canonical_def_only_input_state"
FIXED_STATE_ARTIFACT_KIND = "canonical_def_only_fixed_state"
PHYSICAL_FINALIZE_ARTIFACT_KIND = "canonical_def_only_physical_finalize"
ARTIFACT_VERSION = 1
EXPECTED_ICCAD24_LIBERTY_BASENAMES = (
    "asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
    "asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
    "asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
    "asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
    "asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
    "sram_asap7_16x256_1rw.lib",
    "sram_asap7_32x256_1rw.lib",
    "sram_asap7_64x256_1rw.lib",
    "sram_asap7_64x64_1rw.lib",
)
METRIC_RE = re.compile(r"^METRIC\|([^|]+)\|(.*)$")
POWER_TOTAL_RE = re.compile(
    r"^\s*Total\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)
REQUIRED_METRICS = (
    "raw_hpwl_dbu",
    "legalized_hpwl_dbu",
    "placement_valid",
    "wns_ns",
    "tns_ns",
    "endpoint_count",
    "violating_endpoint_count",
    "slew_violation_count",
    "slew_violation_total",
    "cap_violation_count",
    "cap_violation_total",
    "instance_count",
    "net_count",
    "iterm_count",
    "total_cell_area_um2",
    "macro_area_um2",
    "track_a_runtime_sec",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tcl_quote(value: Path | str) -> str:
    return "{" + str(value).replace("}", "\\}") + "}"


def validate_complete_iccad24_liberty_manifest(libs: Iterable[Path]) -> None:
    """Reject any ICCAD24 evaluation that does not load the complete 9-lib set."""

    expected = set(EXPECTED_ICCAD24_LIBERTY_BASENAMES)
    actual_names = [Path(path).name for path in libs]
    actual = set(actual_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicate_names = sorted(
        name for name in actual if actual_names.count(name) > 1
    )
    if missing or unexpected or duplicate_names or len(actual_names) != len(expected):
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if duplicate_names:
            details.append("duplicates=" + ",".join(duplicate_names))
        raise ValueError(
            "ICCAD24 OpenSTA evaluation requires the complete 9-Liberty "
            "manifest (including SIMPLE_RVT): " + "; ".join(details)
        )


def _as_number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer():
        return int(parsed)
    return parsed


def parse_metrics(path: Path) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    if not path.is_file():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = METRIC_RE.match(line.strip())
        if match is None:
            continue
        key, raw = match.groups()
        number = _as_number(raw)
        metrics[key] = raw if number is None else number
    return metrics


def parse_power_total(path: Path) -> float | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = POWER_TOTAL_RE.match(line)
        if match is not None:
            return float(match.group(4))
    return None


def fixed_state_coordinate_audit(
    input_inventory: dict[str, dict[str, Any]],
    evaluated_inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove that fixed-state evaluation did not move or reclassify instances."""

    invariant_fields = ("x_dbu", "y_dbu", "orient", "placement_status")
    input_names = set(input_inventory)
    evaluated_names = set(evaluated_inventory)
    missing = sorted(input_names - evaluated_names)
    unexpected = sorted(evaluated_names - input_names)
    changed = []
    for name in sorted(input_names & evaluated_names):
        before = input_inventory[name]
        after = evaluated_inventory[name]
        differences = {
            field: {"before": before[field], "after": after[field]}
            for field in invariant_fields
            if before[field] != after[field]
        }
        if differences:
            changed.append({"instance": name, "differences": differences})
    failures = []
    if missing:
        failures.append("instance_missing")
    if unexpected:
        failures.append("unexpected_instance")
    if changed:
        failures.append("coordinate_or_status_changed")
    return {
        "status": "pass" if not failures else "failed",
        "input_instance_count": len(input_inventory),
        "evaluated_instance_count": len(evaluated_inventory),
        "missing_instance_count": len(missing),
        "missing_instances": missing[:100],
        "unexpected_instance_count": len(unexpected),
        "unexpected_instances": unexpected[:100],
        "changed_instance_count": len(changed),
        "changed_instances": changed[:100],
        "failures": failures,
    }


def summarize_mutations(
    reference: dict[str, dict[str, Any]],
    committed: dict[str, dict[str, Any]],
    *,
    buffer_masters: Iterable[str],
) -> dict[str, Any]:
    buffer_master_set = set(buffer_masters)
    reference_names = set(reference)
    committed_names = set(committed)
    inserted_names = sorted(committed_names - reference_names)
    deleted_names = sorted(reference_names - committed_names)
    resized_names = sorted(
        name
        for name in reference_names & committed_names
        if not int(reference[name]["is_block"])
        and reference[name]["master"] != committed[name]["master"]
    )
    coordinate_changed_names = sorted(
        name
        for name in reference_names & committed_names
        if (
            reference[name]["x_dbu"],
            reference[name]["y_dbu"],
            reference[name]["orient"],
            reference[name]["placement_status"],
        )
        != (
            committed[name]["x_dbu"],
            committed[name]["y_dbu"],
            committed[name]["orient"],
            committed[name]["placement_status"],
        )
    )
    inserted_buffer_names = sorted(
        name
        for name in inserted_names
        if committed[name]["master"] in buffer_master_set
    )
    inserted_other_names = sorted(set(inserted_names) - set(inserted_buffer_names))
    reference_cell_area = sum(
        float(row["area_um2"])
        for row in reference.values()
        if not int(row["is_block"])
    )
    committed_cell_area = sum(
        float(row["area_um2"])
        for row in committed.values()
        if not int(row["is_block"])
    )
    return {
        "resize_count": len(resized_names),
        "resized_instances": resized_names,
        "coordinate_changed_count": len(coordinate_changed_names),
        "coordinate_changed_instances": coordinate_changed_names,
        "inserted_buffer_count": len(inserted_buffer_names),
        "inserted_buffer_instances": inserted_buffer_names,
        "inserted_other_count": len(inserted_other_names),
        "inserted_other_instances": inserted_other_names,
        "deleted_instance_count": len(deleted_names),
        "deleted_instances": deleted_names,
        "reference_total_cell_area_um2": reference_cell_area,
        "committed_total_cell_area_um2": committed_cell_area,
        "added_cell_area_um2": committed_cell_area - reference_cell_area,
    }


def _common_tcl_procs() -> list[str]:
    return [
        'proc emit_metric {name value} { puts "METRIC|$name|$value" }',
        "proc safe_metric {name script default_value} {",
        "  if {[catch {uplevel 1 $script} value]} {",
        "    emit_metric $name $default_value",
        "  } else {",
        "    emit_metric $name $value",
        "  }",
        "}",
        "proc refresh_timing {} {",
        "  if {[llength [info commands update_timing]]} {",
        "    update_timing",
        "  } else {",
        "    catch {worst_slack -max}",
        "  }",
        "}",
        "proc canonical_hpwl {} {",
        "  set total 0",
        "  foreach net [[ord::get_db_block] getNets] {",
        "    set sig_type [$net getSigType]",
        '    if {$sig_type eq "POWER" || $sig_type eq "GROUND" || [$net isSpecial]} { continue }',
        "    set x_min 1e30",
        "    set y_min 1e30",
        "    set x_max -1e30",
        "    set y_max -1e30",
        "    set terminal_count 0",
        "    foreach iterm [$net getITerms] {",
        "      lassign [$iterm getAvgXY] valid x y",
        "      if {!$valid} { continue }",
        "      set x_min [expr {min($x_min, $x)}]",
        "      set y_min [expr {min($y_min, $y)}]",
        "      set x_max [expr {max($x_max, $x)}]",
        "      set y_max [expr {max($y_max, $y)}]",
        "      incr terminal_count",
        "    }",
        "    foreach bterm [$net getBTerms] {",
        "      set bterm_seen 0",
        "      foreach bpin [$bterm getBPins] {",
        "        set box [$bpin getBBox]",
        "        set x [expr {([$box xMin] + [$box xMax]) / 2.0}]",
        "        set y [expr {([$box yMin] + [$box yMax]) / 2.0}]",
        "        set x_min [expr {min($x_min, $x)}]",
        "        set y_min [expr {min($y_min, $y)}]",
        "        set x_max [expr {max($x_max, $x)}]",
        "        set y_max [expr {max($y_max, $y)}]",
        "        set bterm_seen 1",
        "      }",
        "      if {$bterm_seen} { incr terminal_count }",
        "    }",
        "    if {$terminal_count >= 2} {",
        "      set total [expr {$total + $x_max - $x_min + $y_max - $y_min}]",
        "    }",
        "  }",
        "  return $total",
        "}",
        "proc area_summary {} {",
        "  set block [ord::get_db_block]",
        "  set dbu [$block getDbUnitsPerMicron]",
        "  set cell_area_dbu2 0",
        "  set macro_area_dbu2 0",
        "  foreach inst [$block getInsts] {",
        "    set area [[$inst getMaster] getArea]",
        "    if {[$inst isBlock]} {",
        "      set macro_area_dbu2 [expr {$macro_area_dbu2 + $area}]",
        "    } else {",
        "      set cell_area_dbu2 [expr {$cell_area_dbu2 + $area}]",
        "    }",
        "  }",
        "  set scale [expr {double($dbu) * double($dbu)}]",
        "  return [list [expr {$cell_area_dbu2 / $scale}] [expr {$macro_area_dbu2 / $scale}]]",
        "}",
        *inventory_tcl_proc_lines(),
        *macro_halo_blockage_tcl_proc_lines(),
        "proc dump_endpoint_slacks {path} {",
        "  set fp [open $path w]",
        '  puts $fp "endpoint,slack_ns"',
        "  set endpoint_count 0",
        "  set violating_count 0",
        "  set groups [sta::path_group_names]",
        "  foreach endpoint [sta::endpoints] {",
        "    set best_slack {}",
        "    foreach group $groups {",
        "      if {![catch {sta::endpoint_slack $endpoint $group max} slack]} {",
        "        if {$best_slack eq {} || $slack < $best_slack} { set best_slack $slack }",
        "      }",
        "    }",
        "    if {$best_slack ne {}} {",
        "      incr endpoint_count",
        "      if {$best_slack < 0.0} { incr violating_count }",
        '      puts $fp "[get_full_name $endpoint],$best_slack"',
        "    }",
        "  }",
        "  close $fp",
        "  emit_metric endpoint_count $endpoint_count",
        "  emit_metric violating_endpoint_count $violating_count",
        "}",
        "proc check_type_summary {flag} {",
        '  set path [file join [pwd] "check_type_[pid]_[string map {- _} $flag].rpt"]',
        "  if {[catch {eval report_check_types $flag -no_line_splits -digits 6 > $path}]} {",
        "    return {0 0.0}",
        "  }",
        "  set fp [open $path r]",
        "  set lines [split [read $fp] \n]",
        "  close $fp",
        "  file delete -force $path",
        "  set count 0",
        "  set total 0.0",
        "  foreach line $lines {",
        "    if {[regexp {([-+.0-9]+)[[:space:]]+\\(VIOLATED\\)} $line -> slack]} {",
        "      incr count",
        "      set total [expr {$total - $slack}]",
        "    }",
        "  }",
        "  return [list $count $total]",
        "}",
    ]


def _read_design_lines(
    *,
    def_input: Path,
    tech_lef: Path,
    lefs: Iterable[Path],
    libs: Iterable[Path] = (),
    sdc: Path | None = None,
    rc_tcl: Path | None = None,
) -> list[str]:
    lines = [f"read_lef {_tcl_quote(tech_lef)}"]
    lines.extend(f"read_lef {_tcl_quote(path)}" for path in lefs)
    lines.extend(f"read_liberty {_tcl_quote(path)}" for path in libs)
    lines.append(f"read_def {_tcl_quote(def_input)}")
    if sdc is not None:
        lines.extend(
            (
                f"read_sdc {_tcl_quote(sdc)}",
                "set_ideal_network [all_clocks]",
            )
        )
    if rc_tcl is not None:
        lines.append(f"source {_tcl_quote(rc_tcl)}")
    lines.extend(
        (
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
        )
    )
    return lines


def build_inventory_tcl(
    *,
    def_input: Path,
    tech_lef: Path,
    lefs: Iterable[Path],
    inventory_path: Path,
) -> str:
    lines = _common_tcl_procs()
    lines.extend(
        _read_design_lines(
            def_input=def_input,
            tech_lef=tech_lef,
            lefs=lefs,
        )
    )
    lines.extend(
        (
            f"dump_inventory {_tcl_quote(inventory_path)}",
            'emit_metric "instance_count" [llength [[ord::get_db_block] getInsts]]',
            'emit_metric "net_count" [llength [[ord::get_db_block] getNets]]',
            'emit_metric "iterm_count" [llength [[ord::get_db_block] getITerms]]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def build_evaluation_tcl(
    *,
    def_input: Path,
    tech_lef: Path,
    lefs: Iterable[Path],
    libs: Iterable[Path],
    sdc: Path,
    rc_tcl: Path,
    output_def: Path,
    inventory_path: Path,
    evaluated_inventory_path: Path,
    dpl_report_path: Path,
    endpoint_csv: Path,
    power_report: Path,
    evaluation_mode: str = "track_a",
    detailed_placement_search_window: str = "full_core",
) -> str:
    libs = tuple(libs)
    validate_complete_iccad24_liberty_manifest(libs)
    if evaluation_mode not in {
        "track_a",
        "input_state",
        "fixed_state",
        "physical_finalize",
    }:
        raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
    lines = _common_tcl_procs()
    lines.extend(
        _read_design_lines(
            def_input=def_input,
            tech_lef=tech_lef,
            lefs=lefs,
            libs=libs,
            sdc=sdc,
            rc_tcl=rc_tcl,
        )
    )
    lines.extend(
        (
            "set track_a_start_us [clock microseconds]",
            'emit_metric "raw_hpwl_dbu" [canonical_hpwl]',
            f"dump_inventory {_tcl_quote(inventory_path)}",
        )
    )
    if evaluation_mode in {"track_a", "physical_finalize"}:
        lines.extend(
            [
                f"file delete -force {_tcl_quote(dpl_report_path)}",
                "set track_a_macro_halo_blockages [create_macro_halo_blockages]",
                'emit_metric "macro_halo_blockage_count" [llength $track_a_macro_halo_blockages]',
                *detailed_placement_tcl_lines(
                    search_window=detailed_placement_search_window
                ),
                f'if {{[catch {{check_placement -verbose -report_file_name {_tcl_quote(dpl_report_path)}}} message]}} {{',
                '  emit_metric "placement_valid" 0',
                "} else {",
                '  emit_metric "placement_valid" 1',
                "}",
            ]
        )
    elif evaluation_mode == "fixed_state":
        lines.extend(
            [
                f"file delete -force {_tcl_quote(dpl_report_path)}",
                f'if {{[catch {{check_placement -verbose -report_file_name {_tcl_quote(dpl_report_path)}}} message]}} {{',
                '  emit_metric "placement_valid" 0',
                "} else {",
                '  emit_metric "placement_valid" 1',
                "}",
            ]
        )
    else:
        lines.append('emit_metric "placement_valid" -1')
    lines.append(f"dump_inventory {_tcl_quote(evaluated_inventory_path)}")
    if evaluation_mode in {"track_a", "physical_finalize"}:
        lines.append("destroy_macro_halo_blockages $track_a_macro_halo_blockages")
    lines.extend(
        (
            "estimate_parasitics -placement",
            "refresh_timing",
            'emit_metric "legalized_hpwl_dbu" [canonical_hpwl]',
            'safe_metric "wns_ns" {worst_slack -max} ""',
            'safe_metric "tns_ns" {total_negative_slack -max} ""',
            "lassign [check_type_summary -max_slew] slew_count slew_total",
            "lassign [check_type_summary -max_capacitance] cap_count cap_total",
            'emit_metric "slew_violation_count" $slew_count',
            'emit_metric "slew_violation_total" $slew_total',
            'emit_metric "cap_violation_count" $cap_count',
            'emit_metric "cap_violation_total" $cap_total',
            'emit_metric "instance_count" [llength [[ord::get_db_block] getInsts]]',
            'emit_metric "net_count" [llength [[ord::get_db_block] getNets]]',
            'emit_metric "iterm_count" [llength [[ord::get_db_block] getITerms]]',
            "lassign [area_summary] total_cell_area macro_area",
            'emit_metric "total_cell_area_um2" $total_cell_area',
            'emit_metric "macro_area_um2" $macro_area',
            f"dump_endpoint_slacks {_tcl_quote(endpoint_csv)}",
            "set_power_activity -input -activity 0.1 -duty 0.5",
            f"report_power > {_tcl_quote(power_report)}",
            f"write_def {_tcl_quote(output_def)}",
            'emit_metric "track_a_runtime_sec" [expr {([clock microseconds] - $track_a_start_us) / 1000000.0}]',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "returncode": int(completed.returncode),
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
    }


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + ", ".join(missing))


def blocked_layer_only_waiver(report_path: Path) -> dict[str, Any]:
    """Accept only a nonempty report containing blocked-layer failures alone."""
    evidence = {"applied": False, "ignored_category": "Blocked_layers_failures"}
    try:
        categories = json.loads(report_path.read_text())["DPL"]["category"]
        counts = {name: len(category["violations"]) for name, category in categories.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return evidence
    evidence["violation_counts"] = counts
    evidence["applied"] = bool(counts.get("Blocked_layers_failures", 0)) and all(
        count == 0 for name, count in counts.items() if name != "Blocked_layers_failures"
    )
    return evidence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--openroad-bin", type=Path, required=True)
    parser.add_argument("--openroad-num-threads", type=int)
    parser.add_argument("--reference-def", type=Path, required=True)
    parser.add_argument("--def-input", type=Path, required=True)
    parser.add_argument("--tech-lef", type=Path, required=True)
    parser.add_argument("--lef", dest="lefs", action="append", type=Path, required=True)
    parser.add_argument("--lib", dest="libs", action="append", type=Path, required=True)
    parser.add_argument("--sdc", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--buffer-master", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ignore-blocked-layer-violations", action="store_true",
        help="Exclude only reported blocked-layer failures from fixed-state acceptance.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("track_a", "input_state", "fixed_state", "physical_finalize"),
        default="track_a",
    )
    parser.add_argument(
        "--detailed-placement-search-window",
        choices=DETAILED_PLACEMENT_SEARCH_WINDOWS,
        default="full_core",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.openroad_num_threads is not None and args.openroad_num_threads <= 0:
        raise ValueError("openroad_num_threads must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "openroad_bin": args.openroad_bin.resolve(),
        "reference_def": args.reference_def.resolve(),
        "def_input": args.def_input.resolve(),
        "tech_lef": args.tech_lef.resolve(),
        "lefs": [path.resolve() for path in args.lefs],
        "libs": [path.resolve() for path in args.libs],
        "sdc": args.sdc.resolve(),
        "rc_tcl": args.rc_tcl.resolve(),
    }
    validate_complete_iccad24_liberty_manifest(inputs["libs"])
    if not args.plan_only:
        _require_files(
            (
                inputs["openroad_bin"],
                inputs["reference_def"],
                inputs["def_input"],
                inputs["tech_lef"],
                *inputs["lefs"],
                *inputs["libs"],
                inputs["sdc"],
                inputs["rc_tcl"],
            )
        )

    reference_tcl = output_dir / "reference_inventory.tcl"
    reference_log = output_dir / "reference_inventory.log"
    reference_inventory = output_dir / "reference_inventory.tsv"
    evaluation_tcl = output_dir / "evaluate.tcl"
    evaluation_log = output_dir / "evaluate.log"
    committed_inventory = output_dir / "committed_inventory.tsv"
    evaluated_inventory = output_dir / "evaluated_inventory.tsv"
    dpl_report = output_dir / "dpl_placement_report.json"
    endpoint_csv = output_dir / "endpoint_slacks.csv"
    power_report = output_dir / "power.rpt"
    output_def = output_dir / "evaluated.def"
    if args.evaluation_mode == "track_a":
        artifact_kind = ARTIFACT_KIND
        summary_path = output_dir / "track_a_summary.json"
    elif args.evaluation_mode == "fixed_state":
        artifact_kind = FIXED_STATE_ARTIFACT_KIND
        summary_path = output_dir / "fixed_state_summary.json"
    elif args.evaluation_mode == "physical_finalize":
        artifact_kind = PHYSICAL_FINALIZE_ARTIFACT_KIND
        summary_path = output_dir / "physical_finalize_summary.json"
    else:
        artifact_kind = INPUT_STATE_ARTIFACT_KIND
        summary_path = output_dir / "input_state_summary.json"

    reference_tcl.write_text(
        build_inventory_tcl(
            def_input=inputs["reference_def"],
            tech_lef=inputs["tech_lef"],
            lefs=inputs["lefs"],
            inventory_path=reference_inventory,
        ),
        encoding="utf-8",
    )
    evaluation_tcl.write_text(
        build_evaluation_tcl(
            def_input=inputs["def_input"],
            tech_lef=inputs["tech_lef"],
            lefs=inputs["lefs"],
            libs=inputs["libs"],
            sdc=inputs["sdc"],
            rc_tcl=inputs["rc_tcl"],
            output_def=output_def,
            inventory_path=committed_inventory,
            evaluated_inventory_path=evaluated_inventory,
            dpl_report_path=dpl_report,
            endpoint_csv=endpoint_csv,
            power_report=power_report,
            evaluation_mode=args.evaluation_mode,
            detailed_placement_search_window=args.detailed_placement_search_window,
        ),
        encoding="utf-8",
    )
    openroad_prefix = [str(inputs["openroad_bin"])]
    if args.openroad_num_threads is not None:
        openroad_prefix.extend(
            ("-threads", str(int(args.openroad_num_threads)))
        )
    commands = {
        "reference_inventory": [*openroad_prefix, str(reference_tcl)],
        "evaluation": [*openroad_prefix, str(evaluation_tcl)],
    }
    (output_dir / "commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        shlex.join(commands["evaluation"]) + "\n",
        encoding="utf-8",
    )
    input_manifest = {
        "case": args.case,
        "method_id": args.method_id,
        "buffer_masters": sorted(set(args.buffer_master)),
        "openroad_num_threads": args.openroad_num_threads,
        "detailed_placement_search_window": args.detailed_placement_search_window,
        "paths": {
            key: [str(item) for item in value] if isinstance(value, list) else str(value)
            for key, value in inputs.items()
        },
    }
    if not args.plan_only:
        input_manifest["sha256"] = {
            key: [sha256_file(item) for item in value]
            if isinstance(value, list)
            else sha256_file(value)
            for key, value in inputs.items()
        }

    if args.plan_only:
        summary = {
            "artifact": artifact_kind,
            "artifact_version": ARTIFACT_VERSION,
            "status": "plan_only",
            "inputs": input_manifest,
            "commands": commands,
            "artifacts": {
                "reference_tcl": str(reference_tcl),
                "evaluation_tcl": str(evaluation_tcl),
                "summary": str(summary_path),
            },
        }
        _write_json(summary_path, summary)
        print(f"summary: {summary_path}")
        return 0

    reference_execution = _run_command(
        commands["reference_inventory"],
        cwd=output_dir,
        log_path=reference_log,
    )
    evaluation_execution = _run_command(
        commands["evaluation"],
        cwd=output_dir,
        log_path=evaluation_log,
    )
    metrics = parse_metrics(evaluation_log)
    failures: list[str] = []
    if reference_execution["returncode"] != 0:
        failures.append("reference_inventory_execution_failed")
    if evaluation_execution["returncode"] != 0:
        failures.append("evaluation_execution_failed")
    missing_metrics = [name for name in REQUIRED_METRICS if metrics.get(name) is None]
    if missing_metrics:
        failures.append("missing_metrics:" + ",".join(missing_metrics))
    placement_validation = None
    if (
        args.evaluation_mode in {"track_a", "physical_finalize"}
        and reference_inventory.is_file()
        and evaluated_inventory.is_file()
        and metrics.get("placement_valid") is not None
    ):
        raw_placement_valid = int(metrics["placement_valid"])
        placement_validation = qualify_placement(
            raw_placement_valid=raw_placement_valid,
            reference_inventory=read_inventory(reference_inventory),
            evaluated_inventory=read_inventory(evaluated_inventory),
            dpl_report_path=dpl_report,
        )
        metrics["placement_valid_raw"] = raw_placement_valid
        metrics["placement_valid"] = int(
            placement_validation["effective_placement_valid"]
        )
    fixed_state_audit = None
    if (
        args.evaluation_mode == "fixed_state"
        and committed_inventory.is_file()
        and evaluated_inventory.is_file()
    ):
        raw_placement_valid = int(metrics.get("placement_valid", 0))
        effective_placement_valid = raw_placement_valid == 1
        if args.ignore_blocked_layer_violations:
            placement_validation = blocked_layer_only_waiver(dpl_report)
            placement_validation["policy"] = "ignore_blocked_layers_only"
            placement_validation["raw_placement_valid"] = raw_placement_valid
            if raw_placement_valid == 0 and placement_validation["applied"]:
                effective_placement_valid = True
            placement_validation["effective_placement_valid"] = effective_placement_valid
        fixed_state_audit = fixed_state_coordinate_audit(
            read_inventory(committed_inventory),
            read_inventory(evaluated_inventory),
        )
        fixed_state_audit["raw_placement_valid"] = raw_placement_valid
        fixed_state_audit["physical_placement_valid"] = bool(
            raw_placement_valid == 1
        )
        fixed_state_audit["accepted_under_placement_policy"] = effective_placement_valid
        metrics["placement_valid_raw"] = raw_placement_valid
        metrics["placement_valid"] = int(
            effective_placement_valid
            and fixed_state_audit["status"] == "pass"
        )
        if not effective_placement_valid:
            failures.append("fixed_state_input_placement_invalid")
        if fixed_state_audit["status"] != "pass":
            failures.append("fixed_state_coordinate_audit_failed")
    if args.evaluation_mode in {
        "track_a",
        "fixed_state",
        "physical_finalize",
    } and metrics.get("placement_valid") != 1:
        failures.append("placement_invalid")
    if args.evaluation_mode == "input_state" and metrics.get("placement_valid") != -1:
        failures.append("input_state_placement_marker_invalid")
    required_artifacts = (
        reference_inventory,
        committed_inventory,
        evaluated_inventory,
        endpoint_csv,
        power_report,
        output_def,
    )
    missing_artifacts = [str(path) for path in required_artifacts if not path.is_file()]
    if missing_artifacts:
        failures.append("missing_artifacts:" + ",".join(missing_artifacts))

    mutation = None
    if reference_inventory.is_file() and committed_inventory.is_file():
        mutation = summarize_mutations(
            read_inventory(reference_inventory),
            read_inventory(committed_inventory),
            buffer_masters=args.buffer_master,
        )
    power_total_w = parse_power_total(power_report)
    if power_total_w is None:
        failures.append("missing_power_total")
    status = "pass" if not failures else "failed"
    summary = {
        "artifact": artifact_kind,
        "artifact_version": ARTIFACT_VERSION,
        "status": status,
        "failures": failures,
        "inputs": input_manifest,
        "executions": {
            "reference_inventory": reference_execution,
            "evaluation": evaluation_execution,
        },
        "metrics": {**metrics, "power_total_w": power_total_w},
        "placement_validation": placement_validation,
        "fixed_state_audit": fixed_state_audit,
        "mutation": mutation,
        "artifacts": {
            "reference_tcl": str(reference_tcl),
            "reference_inventory": str(reference_inventory),
            "evaluation_tcl": str(evaluation_tcl),
            "evaluation_log": str(evaluation_log),
            "committed_inventory": str(committed_inventory),
            "evaluated_inventory": str(evaluated_inventory),
            "dpl_placement_report": str(dpl_report),
            "dpl_placement_report_sha256": (
                sha256_file(dpl_report) if dpl_report.is_file() else None
            ),
            "endpoint_csv": str(endpoint_csv),
            "power_report": str(power_report),
            "evaluated_def": str(output_def),
            "evaluated_def_sha256": sha256_file(output_def) if output_def.is_file() else None,
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    print(f"summary: {summary_path}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
