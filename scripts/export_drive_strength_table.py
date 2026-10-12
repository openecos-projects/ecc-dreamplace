#!/usr/bin/env python3
"""Export the fixed ICS55 four-buffer strength table from a case PDK JSON."""

import argparse
import json
from pathlib import Path

from dreamplace.ops.buffer_insertion.drive_type_table import (
    DRIVE_TYPE_TABLE_SCHEMA,
    EXPECTED_DRIVE_TYPE_MASTERS,
)


def export_table(pdk_path, output_path):
    pdk_path = Path(pdk_path)
    payload = json.loads(pdk_path.read_text())
    configured = tuple(str(value) for value in payload.get("buffers", ()))
    missing = [name for name in EXPECTED_DRIVE_TYPE_MASTERS if name not in configured]
    if missing:
        raise ValueError(f"case PDK is missing approved buffers: {missing}")
    rows = [
        {
            "drive_type_id": index,
            "master_name": name,
            "drive_strength": strength,
        }
        for index, (name, strength) in enumerate(
            zip(EXPECTED_DRIVE_TYPE_MASTERS, (8.0, 12.0, 16.0, 20.0))
        )
    ]
    result = {
        "schema": DRIVE_TYPE_TABLE_SCHEMA,
        "source": "case-pdk-buffer-list",
        "library": str(payload.get("name", "ics55")),
        "units": {"drive_strength": "relative"},
        "source_pdk": str(pdk_path),
        "buffers": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdk_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    export_table(args.pdk_json, args.output_json)
    print(args.output_json)


if __name__ == "__main__":
    main()
