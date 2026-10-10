#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


DIEAREA_RE = re.compile(
    r"\bDIEAREA\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
    re.IGNORECASE,
)
COMPONENTS_RE = re.compile(r"^\s*COMPONENTS\s+\d+\s*;", re.IGNORECASE)
END_COMPONENTS_RE = re.compile(r"^\s*END\s+COMPONENTS\b", re.IGNORECASE)
PLACEMENT_RE = re.compile(
    r"\+\s*(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*([A-Z0-9_]*)",
    re.IGNORECASE,
)
UNITS_RE = re.compile(
    r"\bUNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;",
    re.IGNORECASE,
)
COMPONENT_RE = re.compile(r"^\s*-\s+(\S+)\s+(\S+)\b", re.IGNORECASE)
LEF_MACRO_RE = re.compile(r"^\s*MACRO\s+(\S+)\s*$", re.IGNORECASE)
LEF_SIZE_RE = re.compile(
    r"^\s*SIZE\s+([0-9]*\.?[0-9]+)\s+BY\s+([0-9]*\.?[0-9]+)\s*;",
    re.IGNORECASE,
)
LEF_END_RE = re.compile(r"^\s*END\s+(\S+)\s*$", re.IGNORECASE)
PLOT_DPI = 240
HEATMAP_BINS = 160
COLORBAR_LABEL = "placed cell count / bin"
FINAL_DEF_KEYS = (
    "final_def",
    "final_def_path",
    "output_def",
    "def_path",
    "placed_def",
)
ORIGINAL_DEF_KEYS = (
    "original_def",
    "original_def_path",
    "source_def",
    "source_def_path",
    "input_def",
)


def default_plot_style():
    return {
        "dpi": PLOT_DPI,
        "bins": [HEATMAP_BINS, HEATMAP_BINS],
        "colorbar_label": COLORBAR_LABEL,
        "shared_colorbar": True,
        "shared_color_scale": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def bbox(points):
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def rect_bbox(rects):
    if not rects:
        return None
    return [
        min(rect["x"] for rect in rects),
        min(rect["y"] for rect in rects),
        max(rect["x"] + rect["width"] for rect in rects),
        max(rect["y"] + rect["height"] for rect in rects),
    ]


def parse_lef_macro_sizes(lef_paths, dbu_per_micron):
    sizes = {}
    if not lef_paths or not dbu_per_micron:
        return sizes
    for lef_path in lef_paths:
        path = Path(lef_path)
        if not path.exists():
            continue
        current_macro = None
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            macro_match = LEF_MACRO_RE.match(line)
            if macro_match:
                current_macro = macro_match.group(1)
                continue
            if current_macro:
                size_match = LEF_SIZE_RE.match(line)
                if size_match:
                    sizes[current_macro] = (
                        int(round(float(size_match.group(1)) * dbu_per_micron)),
                        int(round(float(size_match.group(2)) * dbu_per_micron)),
                    )
                    continue
                end_match = LEF_END_RE.match(line)
                if end_match and end_match.group(1) == current_macro:
                    current_macro = None
    return sizes


def rotate_size_for_orient(width, height, orient):
    if orient and orient.upper() in {"E", "W", "FE", "FW"}:
        return height, width
    return width, height


def macro_rectangles(fixed_components, macro_sizes):
    rects = []
    for component in fixed_components:
        size = macro_sizes.get(component["master"])
        if not size:
            continue
        width, height = rotate_size_for_orient(
            size[0],
            size[1],
            component.get("orient"),
        )
        if width <= 0 or height <= 0:
            continue
        rects.append(
            {
                "name": component["name"],
                "master": component["master"],
                "x": component["x"],
                "y": component["y"],
                "width": width,
                "height": height,
            }
        )
    return rects


def parse_def(def_path, macro_sizes=None):
    path = Path(def_path)
    text = path.read_text(errors="replace")
    diearea_match = DIEAREA_RE.search(text)
    diearea = None
    if diearea_match:
        diearea = [int(value) for value in diearea_match.groups()]
    units_match = UNITS_RE.search(text)
    dbu_per_micron = int(units_match.group(1)) if units_match else None

    placed = []
    fixed_points = []
    fixed_components = []
    in_components = False
    for line in text.splitlines():
        if COMPONENTS_RE.match(line):
            in_components = True
            continue
        if in_components and END_COMPONENTS_RE.match(line):
            in_components = False
            continue
        if not in_components:
            continue
        component_match = COMPONENT_RE.match(line)
        placement_match = PLACEMENT_RE.search(line)
        if not component_match or not placement_match:
            continue
        point = (int(placement_match.group(2)), int(placement_match.group(3)))
        if placement_match.group(1).upper() == "PLACED":
            placed.append(point)
        else:
            fixed_points.append(point)
            fixed_components.append(
                {
                    "name": component_match.group(1),
                    "master": component_match.group(2),
                    "x": point[0],
                    "y": point[1],
                    "orient": placement_match.group(4) or "",
                }
            )

    macros = macro_rectangles(fixed_components, macro_sizes or {})

    return {
        "path": str(path),
        "diearea": diearea,
        "dbu_per_micron": dbu_per_micron,
        "movable_count": len(placed),
        "fixed_count": len(fixed_points),
        "macro_count": len(macros),
        "movable_bbox": bbox(placed),
        "fixed_bbox": bbox(fixed_points),
        "macro_bbox": rect_bbox(macros),
        "_movable_points": placed,
        "_fixed_points": fixed_points,
        "_fixed_components": fixed_components,
        "_macros": macros,
    }


def is_plottable(def_path):
    if not def_path or not Path(def_path).exists():
        return False
    try:
        summary = parse_def(def_path)
    except OSError:
        return False
    return summary["diearea"] is not None and (
        summary["movable_count"] > 0 or summary["fixed_count"] > 0
    )


def collect_placement_summaries(summary):
    placements = []
    placements.extend(summary.get("placement_summaries", []) or [])
    for variant in summary.get("variants", []) or []:
        placements.extend(variant.get("placement_summaries", []) or [])
    return placements


def mode_matches(row, placement):
    row_mode = row.get("mode_config")
    placement_mode = (
        placement.get("mode_config")
        or placement.get("variant_label")
        or placement.get("label")
    )
    return not row_mode or not placement_mode or row_mode == placement_mode


def matching_placement(row, placements):
    design = row.get("design")
    for placement in placements:
        if placement.get("design") == design and mode_matches(row, placement):
            return placement
    return None


def path_from_keys(*sources, keys):
    for source in sources:
        if not source:
            continue
        for key in keys:
            value = source.get(key)
            if value:
                return Path(value)
    return None


def resolve_original_def(row, placement, benchmark_root):
    path = path_from_keys(row, placement, keys=ORIGINAL_DEF_KEYS)
    if path:
        return path
    design = row.get("design") or (placement or {}).get("design")
    if not design:
        return None
    return benchmark_root / "design" / design / ("%s.def" % design)


def resolve_final_def(row, placement):
    path = path_from_keys(row, placement, keys=FINAL_DEF_KEYS)
    if path:
        return path
    for source in (row, placement or {}):
        run_dir = source.get("run_dir") or source.get("artifact_dir")
        design = source.get("design")
        if run_dir and design:
            candidates = sorted(Path(run_dir).glob("*.def"))
            exact = Path(run_dir) / ("%s.def" % design)
            if exact.exists():
                return exact
            if candidates:
                return candidates[0]
    return None


def design_input_paths(row, placement, key):
    values = []
    for source in (placement or {}, row or {}):
        design_inputs = source.get("design_inputs") or {}
        value = design_inputs.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            values.extend(value)
        else:
            values.append(value)
    return values


def infer_lef_paths(row, placement, benchmark_root):
    paths = []
    for key in ("tech_lef", "lef"):
        paths.extend(design_input_paths(row, placement, key))
    if paths:
        return [Path(path) for path in paths]

    asap7_root = Path(benchmark_root) / "ASAP7" / "lef"
    if not asap7_root.exists():
        return []
    preferred = [
        "asap7_tech_1x_201209.lef",
        "asap7sc7p5t_27_R_1x_201211.lef",
        "sram_asap7_16x256_1rw.lef",
        "sram_asap7_32x256_1rw.lef",
        "sram_asap7_64x256_1rw.lef",
        "sram_asap7_64x64_1rw.lef",
    ]
    found = [asap7_root / name for name in preferred if (asap7_root / name).exists()]
    return found or sorted(asap7_root.glob("*.lef"))


def heatmap_counts(points, diearea):
    xl, yl, xh, yh = diearea
    if not points:
        return np.zeros((HEATMAP_BINS, HEATMAP_BINS))
    xs, ys = zip(*points)
    image, _, _ = np.histogram2d(
        ys,
        xs,
        bins=HEATMAP_BINS,
        range=[[yl, yh], [xl, xh]],
    )
    return image


def draw_heatmap(ax, image, diearea, vmax):
    xl, yl, xh, yh = diearea
    return ax.imshow(
        image,
        extent=[xl, xh, yl, yh],
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=max(1.0, vmax),
        interpolation="nearest",
        aspect="equal",
    )


def draw_macros(ax, macros, scale):
    for macro in macros:
        ax.add_patch(
            Rectangle(
                (macro["x"], macro["y"]),
                macro["width"],
                macro["height"],
                fill=False,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.9,
            )
        )


def format_um_ticks(ax, diearea, scale):
    xl, yl, xh, yh = diearea
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(["%g" % (tick / scale) for tick in xticks])
    ax.set_yticklabels(["%g" % (tick / scale) for tick in yticks])
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def plot_points(ax, parsed, title, diearea, image, vmax):
    xl, yl, xh, yh = diearea
    ax.set_title(title)
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_aspect("equal", adjustable="box")
    mesh = draw_heatmap(ax, image, diearea, vmax)
    draw_macros(ax, parsed["_macros"], parsed.get("dbu_per_micron") or 1)
    scale = parsed.get("dbu_per_micron") or 1
    format_um_ticks(ax, diearea, scale)
    return mesh


def write_plot(original, final, design, mode_config, plot_path):
    dieareas = [area for area in (original["diearea"], final["diearea"]) if area]
    xl = min(area[0] for area in dieareas)
    yl = min(area[1] for area in dieareas)
    xh = max(area[2] for area in dieareas)
    yh = max(area[3] for area in dieareas)
    diearea = [xl, yl, xh, yh]

    width = 14 if (xh - xl) >= (yh - yl) else 11
    height = 6.2 if (xh - xl) >= (yh - yl) else 8
    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
    fig.suptitle("%s / %s" % (design, mode_config))
    original_heatmap = heatmap_counts(original["_movable_points"], diearea)
    final_heatmap = heatmap_counts(final["_movable_points"], diearea)
    shared_vmax = max(float(original_heatmap.max()), float(final_heatmap.max()), 1.0)
    mesh = plot_points(
        axes[0],
        original,
        "Original DEF",
        diearea,
        original_heatmap,
        shared_vmax,
    )
    plot_points(
        axes[1],
        final,
        "AutoDMP/OpenROAD final DEF",
        diearea,
        final_heatmap,
        shared_vmax,
    )
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    colorbar.set_label(COLORBAR_LABEL)
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    return diearea, shared_vmax


def public_summary(parsed):
    return {
        key: value
        for key, value in parsed.items()
        if not key.startswith("_")
    }


def generate(summary_path, benchmark_root, output_dir):
    summary = json.loads(Path(summary_path).read_text())
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    placements = collect_placement_summaries(summary)
    results = {"plots": [], "skipped": []}

    for row in summary.get("rows", []) or []:
        placement = matching_placement(row, placements)
        design = row.get("design") or (placement or {}).get("design")
        mode_config = row.get("mode_config") or "unknown"
        final_def = resolve_final_def(row, placement)
        status = row.get("status")
        if status != "sta_completed" and not is_plottable(final_def):
            results["skipped"].append(
                {
                    "design": design,
                    "mode_config": mode_config,
                    "reason": "missing_or_unplottable_final_def",
                }
            )
            continue
        if not is_plottable(final_def):
            results["skipped"].append(
                {
                    "design": design,
                    "mode_config": mode_config,
                    "reason": "missing_or_unplottable_final_def",
                }
            )
            continue
        original_def = resolve_original_def(row, placement, benchmark_root)
        if not is_plottable(original_def):
            results["skipped"].append(
                {
                    "design": design,
                    "mode_config": mode_config,
                    "reason": "missing_or_unplottable_original_def",
                }
            )
            continue

        dbu_per_micron = parse_def(original_def).get("dbu_per_micron") or parse_def(
            final_def
        ).get("dbu_per_micron")
        macro_sizes = parse_lef_macro_sizes(
            infer_lef_paths(row, placement, benchmark_root),
            dbu_per_micron,
        )
        original = parse_def(original_def, macro_sizes=macro_sizes)
        final = parse_def(final_def, macro_sizes=macro_sizes)
        plot_path = output_dir / (
            "%s__%s__original_vs_ours.png" % (design, mode_config)
        )
        shared_diearea, shared_color_vmax = write_plot(
            original,
            final,
            design,
            mode_config,
            plot_path,
        )
        results["plots"].append(
            {
                "design": design,
                "mode_config": mode_config,
                "original_def": str(original_def),
                "final_def": str(final_def),
                "plot_path": str(plot_path),
                "compare_plot_path": str(plot_path),
                "plot_style": default_plot_style(),
                "shared_diearea": shared_diearea,
                "shared_color_vmax": shared_color_vmax,
                "original": public_summary(original),
                "final": public_summary(final),
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def main(argv=None):
    args = parse_args(argv)
    generate(args.summary, args.benchmark_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
