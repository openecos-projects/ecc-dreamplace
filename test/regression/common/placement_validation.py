#!/usr/bin/env python3
"""Shared physical-placement qualification for regression evaluators."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


INVENTORY_FIELDS = (
    "instance",
    "master",
    "is_block",
    "area_um2",
    "x_dbu",
    "y_dbu",
    "orient",
    "placement_status",
    "bbox_x_min_dbu",
    "bbox_y_min_dbu",
    "bbox_x_max_dbu",
    "bbox_y_max_dbu",
    "keepout_x_min_dbu",
    "keepout_y_min_dbu",
    "keepout_x_max_dbu",
    "keepout_y_max_dbu",
)

_INTEGER_FIELDS = {
    "is_block",
    "x_dbu",
    "y_dbu",
    "bbox_x_min_dbu",
    "bbox_y_min_dbu",
    "bbox_x_max_dbu",
    "bbox_y_max_dbu",
    "keepout_x_min_dbu",
    "keepout_y_min_dbu",
    "keepout_x_max_dbu",
    "keepout_y_max_dbu",
}

_MACRO_INVARIANT_FIELDS = (
    "master",
    "is_block",
    "x_dbu",
    "y_dbu",
    "orient",
    "placement_status",
    "bbox_x_min_dbu",
    "bbox_y_min_dbu",
    "bbox_x_max_dbu",
    "bbox_y_max_dbu",
    "keepout_x_min_dbu",
    "keepout_y_min_dbu",
    "keepout_x_max_dbu",
    "keepout_y_max_dbu",
)

DETAILED_PLACEMENT_SEARCH_WINDOWS = ("full_core",)


def detailed_placement_tcl_lines(*, search_window: str) -> list[str]:
    if search_window != "full_core":
        raise ValueError(
            f"unsupported detailed-placement search window: {search_window}"
        )
    return [
        "set dpl_block [ord::get_db_block]",
        "set dpl_core [$dpl_block getCoreArea]",
        "set dpl_dbu_per_micron [$dpl_block getDbUnitsPerMicron]",
        "set dpl_max_displacement_x_um [expr {int(ceil(double([$dpl_core xMax] - [$dpl_core xMin]) / double($dpl_dbu_per_micron)))}]",
        "set dpl_max_displacement_y_um [expr {int(ceil(double([$dpl_core yMax] - [$dpl_core yMin]) / double($dpl_dbu_per_micron)))}]",
        "if {$dpl_max_displacement_x_um < 1} { set dpl_max_displacement_x_um 1 }",
        "if {$dpl_max_displacement_y_um < 1} { set dpl_max_displacement_y_um 1 }",
        'emit_metric "detailed_placement_search_window" "full_core"',
        'emit_metric "detailed_placement_max_displacement_x_um" $dpl_max_displacement_x_um',
        'emit_metric "detailed_placement_max_displacement_y_um" $dpl_max_displacement_y_um',
        "detailed_placement -max_displacement [list $dpl_max_displacement_x_um $dpl_max_displacement_y_um]",
    ]


def inventory_tcl_proc_lines() -> list[str]:
    header = "\\t".join(INVENTORY_FIELDS)
    return [
        "proc dump_inventory {path} {",
        "  set block [ord::get_db_block]",
        "  set dbu [$block getDbUnitsPerMicron]",
        "  set scale [expr {double($dbu) * double($dbu)}]",
        "  set fp [open $path w]",
        f'  puts $fp "{header}"',
        "  foreach inst [$block getInsts] {",
        "    set master [$inst getMaster]",
        "    set area [expr {[$master getArea] / $scale}]",
        "    lassign [$inst getLocation] x y",
        "    set bbox [$inst getBBox]",
        "    set bbox_x_min [$bbox xMin]",
        "    set bbox_y_min [$bbox yMin]",
        "    set bbox_x_max [$bbox xMax]",
        "    set bbox_y_max [$bbox yMax]",
        "    set halo_left 0",
        "    set halo_bottom 0",
        "    set halo_right 0",
        "    set halo_top 0",
        "    set halo_box [$inst getHalo]",
        '    if {$halo_box ne "NULL" && ![$halo_box isSoft]} {',
        "      set halo [$inst getTransformedHalo]",
        "      set halo_left [$halo xMin]",
        "      set halo_bottom [$halo yMin]",
        "      set halo_right [$halo xMax]",
        "      set halo_top [$halo yMax]",
        "    }",
        "    set keepout_x_min [expr {$bbox_x_min - $halo_left}]",
        "    set keepout_y_min [expr {$bbox_y_min - $halo_bottom}]",
        "    set keepout_x_max [expr {$bbox_x_max + $halo_right}]",
        "    set keepout_y_max [expr {$bbox_y_max + $halo_top}]",
        '    puts $fp "[$inst getName]\\t[$master getName]\\t[$inst isBlock]\\t$area\\t$x\\t$y\\t[$inst getOrient]\\t[$inst getPlacementStatus]\\t$bbox_x_min\\t$bbox_y_min\\t$bbox_x_max\\t$bbox_y_max\\t$keepout_x_min\\t$keepout_y_min\\t$keepout_x_max\\t$keepout_y_max"',
        "  }",
        "  close $fp",
        "}",
    ]


def macro_halo_blockage_tcl_proc_lines() -> list[str]:
    return [
        "proc create_macro_halo_blockages {} {",
        "  set block [ord::get_db_block]",
        "  set created {}",
        "  foreach inst [$block getInsts] {",
        "    if {![$inst isBlock]} { continue }",
        "    set halo_box [$inst getHalo]",
        '    if {$halo_box eq "NULL" || [$halo_box isSoft]} { continue }',
        "    set bbox [$inst getBBox]",
        "    set halo [$inst getTransformedHalo]",
        "    set x_min [expr {[$bbox xMin] - [$halo xMin]}]",
        "    set y_min [expr {[$bbox yMin] - [$halo yMin]}]",
        "    set x_max [expr {[$bbox xMax] + [$halo xMax]}]",
        "    set y_max [expr {[$bbox yMax] + [$halo yMax]}]",
        "    lappend created [odb::dbBlockage_create $block $x_min $y_min $x_max $y_max]",
        "  }",
        "  return $created",
        "}",
        "proc destroy_macro_halo_blockages {blockages} {",
        "  foreach blockage $blockages { odb::dbBlockage_destroy $blockage }",
        "}",
    ]


def read_inventory(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise ValueError(f"invalid inventory header: {reader.fieldnames}")
        for row in reader:
            name = str(row["instance"])
            if name in result:
                raise ValueError(f"duplicate inventory instance: {name}")
            parsed = {
                field: int(row[field]) if field in _INTEGER_FIELDS else str(row[field])
                for field in INVENTORY_FIELDS
                if field not in {"instance", "area_um2"}
            }
            parsed["area_um2"] = float(row["area_um2"])
            result[name] = parsed
    return result


def _rect(row: dict[str, Any], prefix: str) -> tuple[int, int, int, int]:
    return (
        int(row[f"{prefix}_x_min_dbu"]),
        int(row[f"{prefix}_y_min_dbu"]),
        int(row[f"{prefix}_x_max_dbu"]),
        int(row[f"{prefix}_y_max_dbu"]),
    )


def _overlaps(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> bool:
    lx0, ly0, lx1, ly1 = lhs
    rx0, ry0, rx1, ry1 = rhs
    return lx0 < rx1 and rx0 < lx1 and ly0 < ry1 and ry0 < ly1


def _macro_geometry_summary(
    reference: dict[str, dict[str, Any]],
    evaluated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_macros = {
        name: row for name, row in reference.items() if int(row["is_block"])
    }
    evaluated_macros = {
        name: row for name, row in evaluated.items() if int(row["is_block"])
    }
    missing_macros = sorted(set(reference_macros) - set(evaluated_macros))
    unexpected_macros = sorted(set(evaluated_macros) - set(reference_macros))
    changed_macros = sorted(
        name
        for name in set(reference_macros) & set(evaluated_macros)
        if any(
            reference_macros[name][field] != evaluated_macros[name][field]
            for field in _MACRO_INVARIANT_FIELDS
        )
    )

    cells = sorted(
        (
            (name, row, _rect(row, "bbox"))
            for name, row in evaluated.items()
            if not int(row["is_block"])
        ),
        key=lambda item: item[2][0],
    )
    macros_by_x = sorted(
        (
            (name, row, _rect(row, "bbox"), _rect(row, "keepout"))
            for name, row in evaluated_macros.items()
        ),
        key=lambda item: item[3][0],
    )
    bbox_overlap_pairs: list[str] = []
    keepout_overlap_pairs: list[str] = []
    active_macros: list[
        tuple[str, dict[str, Any], tuple[int, int, int, int], tuple[int, int, int, int]]
    ] = []
    next_macro = 0
    max_cell_x_max = -(2**63)
    for cell_name, _cell, cell_bbox in cells:
        max_cell_x_max = max(max_cell_x_max, cell_bbox[2])
        while (
            next_macro < len(macros_by_x)
            and macros_by_x[next_macro][3][0] < max_cell_x_max
        ):
            active_macros.append(macros_by_x[next_macro])
            next_macro += 1
        active_macros = [
            macro for macro in active_macros if macro[3][2] > cell_bbox[0]
        ]
        for macro_name, _macro, macro_bbox, macro_keepout in active_macros:
            if _overlaps(macro_bbox, cell_bbox):
                bbox_overlap_pairs.append(f"{macro_name}|{cell_name}")
            if _overlaps(macro_keepout, cell_bbox):
                keepout_overlap_pairs.append(f"{macro_name}|{cell_name}")

    return {
        "reference_macro_count": len(reference_macros),
        "evaluated_macro_count": len(evaluated_macros),
        "cell_count": len(cells),
        "missing_macro_count": len(missing_macros),
        "missing_macros": missing_macros,
        "unexpected_macro_count": len(unexpected_macros),
        "unexpected_macros": unexpected_macros,
        "changed_macro_count": len(changed_macros),
        "changed_macros": changed_macros,
        "macro_cell_bbox_overlap_count": len(bbox_overlap_pairs),
        "macro_cell_bbox_overlap_pairs": bbox_overlap_pairs[:100],
        "macro_cell_keepout_overlap_count": len(keepout_overlap_pairs),
        "macro_cell_keepout_overlap_pairs": keepout_overlap_pairs[:100],
    }


def read_dpl_report(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = dict(dict(payload.get("DPL") or {}).get("category") or {})
    result: dict[str, list[str]] = {}
    for category_name, category in categories.items():
        sources: list[str] = []
        for violation in list(dict(category).get("violations") or []):
            for source in list(dict(violation).get("sources") or []):
                source_payload = dict(source)
                if source_payload.get("type") == "inst" and source_payload.get("name"):
                    sources.append(str(source_payload["name"]))
        result[str(category_name)] = sorted(set(sources))
    return result


def qualify_placement(
    *,
    raw_placement_valid: int,
    reference_inventory: dict[str, dict[str, Any]],
    evaluated_inventory: dict[str, dict[str, Any]],
    dpl_report_path: Path,
) -> dict[str, Any]:
    geometry = _macro_geometry_summary(reference_inventory, evaluated_inventory)
    failures = read_dpl_report(dpl_report_path)
    macro_names = {
        name for name, row in evaluated_inventory.items() if int(row["is_block"])
    }
    padding_sources = failures.get("Padding_failures", [])
    macro_invariant_ok = not any(
        int(geometry[field])
        for field in (
            "missing_macro_count",
            "unexpected_macro_count",
            "changed_macro_count",
        )
    )
    macro_geometry_ok = not any(
        int(geometry[field])
        for field in (
            "macro_cell_bbox_overlap_count",
            "macro_cell_keepout_overlap_count",
        )
    )
    padding_only = set(failures) == {"Padding_failures"}
    padding_sources_are_macros = bool(padding_sources) and set(padding_sources) <= macro_names

    qualified_false_positive = (
        raw_placement_valid == 0
        and padding_only
        and padding_sources_are_macros
        and macro_invariant_ok
        and macro_geometry_ok
    )
    native_pass = raw_placement_valid == 1 and macro_invariant_ok and macro_geometry_ok
    effective_valid = native_pass or qualified_false_positive
    if native_pass:
        mode = "native_opendp_pass"
    elif qualified_false_positive:
        mode = "qualified_fixed_macro_grid_padding_false_positive"
    else:
        mode = "failed"
    return {
        "validation_contract": "opendp_plus_exact_macro_keepout_v1",
        "raw_placement_valid": int(raw_placement_valid),
        "effective_placement_valid": int(effective_valid),
        "mode": mode,
        "dpl_failure_categories": sorted(failures),
        "dpl_failure_sources": failures,
        "qualified_padding_failure_count": (
            len(padding_sources) if qualified_false_positive else 0
        ),
        "macro_invariant_ok": macro_invariant_ok,
        "macro_geometry_ok": macro_geometry_ok,
        "geometry": geometry,
    }
