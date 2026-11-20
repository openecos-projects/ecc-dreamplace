"""Explicit process-independent drive-strength input shared by CLI and ECC."""

import math

DRIVE_TYPE_TABLE_SCHEMA = "ecc.discrete_buffer_drive_strength.v1"


def build_drive_strength_table(pdk_name, buffer_masters, source_pdk, *, drive_strengths):
    rows = []
    seen = set()
    for entry in drive_strengths:
        name = str(entry["master"])
        strength = float(entry["strength"])
        if name not in buffer_masters:
            raise ValueError(f"configured buffer is not in the legal buffer list: {name}")
        if name in seen or not math.isfinite(strength) or strength <= 0:
            raise ValueError(f"invalid or duplicate buffer strength: {name}")
        seen.add(name)
        rows.append({"drive_type_id": len(rows), "master_name": name, "drive_strength": strength})
    if not rows:
        raise ValueError("explicit drive_strengths are required for candidate buffering")
    return {
        "schema": DRIVE_TYPE_TABLE_SCHEMA,
        "source": "case-pdk-buffer-list",
        "library": str(pdk_name),
        "units": {"drive_strength": "relative"},
        "source_pdk": str(source_pdk),
        "buffers": rows,
    }
