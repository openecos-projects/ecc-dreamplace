#!/usr/bin/env python3

import argparse
import json
import os
from types import SimpleNamespace

from .backend_parasitic_state import (
    audit_backend_parasitic_state,
    parasitic_state_passed,
)
from .parity_audit import audit_pydb_parity


def _namespace_from_mapping(mapping):
    return SimpleNamespace(**dict(mapping or {}))


def _load_params_json(params_path):
    with open(params_path, "r") as f:
        return _namespace_from_mapping(json.load(f))


def _read_ieda_pydb(params, workspace):
    from dreamplace.ops.placeio_ieda import place_io as placeio_ieda

    raw_db = placeio_ieda.PlaceIOFunction.read(params, workspace)
    return placeio_ieda.PlaceIOFunction.pydb(raw_db)


def _read_openroad_pydb(params):
    from dreamplace.ops.placeio_openroad import place_io as placeio_openroad

    raw_db = placeio_openroad.PlaceIOFunction.read(params)
    return placeio_openroad.PlaceIOFunction.pydb(raw_db)


def _write_report(report, output):
    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def run_backend_parity_smoke(
    params,
    workspace,
    output,
    ieda_reader=None,
    openroad_reader=None,
    profile="core_topology",
):
    if profile not in ("full", "core_topology"):
        raise ValueError(f"Unsupported backend parity profile: {profile}")
    ieda_reader = ieda_reader or _read_ieda_pydb
    openroad_reader = openroad_reader or _read_openroad_pydb

    ieda_pydb = ieda_reader(params, workspace)
    openroad_pydb = openroad_reader(params)
    report = audit_pydb_parity(
        ieda_pydb,
        openroad_pydb,
        lhs_label="ieda",
        rhs_label="openroad",
    )
    design_inputs = getattr(params, "design_inputs", {}) or {}
    parasitic_state = audit_backend_parasitic_state(
        workspace,
        design_inputs.get("rc_tcl", ""),
    )
    report["parasitic_state"] = parasitic_state
    report["parasitic_state_passed"] = parasitic_state_passed(parasitic_state)
    report["full_sta_parity_passed"] = bool(
        report.get("passed") and report["parasitic_state_passed"]
    )
    _write_report(report, output)
    pass_key = "full_sta_parity_passed" if profile == "full" else "core_topology_passed"
    return 0 if report[pass_key] else 1


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a pydb parity smoke audit between iEDA and OpenROAD backends."
    )
    parser.add_argument("--params", required=True, help="JSON params file with design_inputs.")
    parser.add_argument("--workspace", required=True, help="iEDA workspace directory.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--profile",
        choices=("full", "core_topology"),
        default="core_topology",
        help=(
            "Exit with success on core topology parity by default; use full "
            "to require timing and optional metadata parity as well."
        ),
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    params = _load_params_json(args.params)
    return run_backend_parity_smoke(
        params,
        args.workspace,
        args.output,
        profile=args.profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
