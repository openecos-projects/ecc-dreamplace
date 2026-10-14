#!/usr/bin/env python3
"""Export the fixed ICS55 four-buffer strength table from a case PDK JSON."""

import argparse
import json
from pathlib import Path

from dreamplace.flows.buffer_drive_table import build_drive_strength_table


def export_table(pdk_path, output_path):
    pdk_path = Path(pdk_path)
    payload = json.loads(pdk_path.read_text())
    result = build_drive_strength_table(
        payload.get("name", "ics55"), payload.get("buffers", ()), pdk_path
    )
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
