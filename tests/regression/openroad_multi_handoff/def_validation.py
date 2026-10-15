import re
from pathlib import Path


class DefCoordinateValidationError(ValueError):
    pass


_DIEAREA_RE = re.compile(
    r"\bDIEAREA\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
    re.IGNORECASE,
)
_COMPONENTS_RE = re.compile(r"^\s*COMPONENTS\s+(\d+)\s*;", re.IGNORECASE)
_END_COMPONENTS_RE = re.compile(r"^\s*END\s+COMPONENTS\b", re.IGNORECASE)
_PLACEMENT_RE = re.compile(
    r"\+\s*(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)",
    re.IGNORECASE,
)


def _bbox_for(points):
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _is_outside_die(point, diearea, tolerance_dbu):
    x, y = point
    xl, yl, xh, yh = diearea
    return (
        x < xl - tolerance_dbu
        or x > xh + tolerance_dbu
        or y < yl - tolerance_dbu
        or y > yh + tolerance_dbu
    )


def parse_def_placement_summary(def_path, tolerance_dbu=0):
    path = Path(def_path)
    text = path.read_text(errors="replace")
    diearea_match = _DIEAREA_RE.search(text)
    diearea = None
    if diearea_match:
        diearea = tuple(int(value) for value in diearea_match.groups())

    component_count = 0
    movable_points = []
    fixed_points = []
    in_components = False

    for line in text.splitlines():
        components_match = _COMPONENTS_RE.match(line)
        if components_match:
            component_count = int(components_match.group(1))
            in_components = True
            continue
        if in_components and _END_COMPONENTS_RE.match(line):
            in_components = False
            continue
        if not in_components:
            continue

        placement_match = _PLACEMENT_RE.search(line)
        if not placement_match:
            continue

        status = placement_match.group(1).upper()
        point = (int(placement_match.group(2)), int(placement_match.group(3)))
        if status == "PLACED":
            movable_points.append(point)
        elif status == "FIXED":
            fixed_points.append(point)

    movable_outside_die_count = 0
    fixed_outside_die_count = 0
    if diearea is not None:
        movable_outside_die_count = sum(
            1 for point in movable_points if _is_outside_die(point, diearea, tolerance_dbu)
        )
        fixed_outside_die_count = sum(
            1 for point in fixed_points if _is_outside_die(point, diearea, tolerance_dbu)
        )

    return {
        "def_path": str(path),
        "diearea": diearea,
        "component_count": component_count,
        "movable_count": len(movable_points),
        "fixed_count": len(fixed_points),
        "movable_bbox": _bbox_for(movable_points),
        "fixed_bbox": _bbox_for(fixed_points),
        "movable_outside_die_count": movable_outside_die_count,
        "fixed_outside_die_count": fixed_outside_die_count,
        "tolerance_dbu": tolerance_dbu,
    }


def validate_final_def_coordinates(def_path, tolerance_dbu=0):
    summary = parse_def_placement_summary(def_path, tolerance_dbu=tolerance_dbu)
    if summary["diearea"] is None:
        raise DefCoordinateValidationError("DEF is missing DIEAREA")
    if summary["movable_count"] == 0:
        raise DefCoordinateValidationError("DEF has zero movable PLACED components")
    if summary["movable_outside_die_count"]:
        raise DefCoordinateValidationError(
            "%d movable component coordinates are outside DIEAREA"
            % summary["movable_outside_die_count"]
        )
    return summary
