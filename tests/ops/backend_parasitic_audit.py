#!/usr/bin/env python3
"""Audit backend parasitic-state inputs for OpenROAD/iEDA STA comparisons."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from dreamplace.ops.placeio_common.backend_parasitic_state import (  # noqa: E402
    audit_backend_parasitic_state,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Audit OpenROAD/iEDA parasitic-state inputs for backend STA parity."
    )
    parser.add_argument("--workspace", required=True, help="AiEDA workspace directory.")
    parser.add_argument("--rc-tcl", default="", help="OpenROAD setRC Tcl path, if used.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    report = audit_backend_parasitic_state(args.workspace, args.rc_tcl)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
