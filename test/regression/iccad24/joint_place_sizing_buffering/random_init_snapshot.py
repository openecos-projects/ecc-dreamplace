#!/usr/bin/env python3
"""Materialize one reusable standard-cell-only random-center R0 DEF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
if str(AUTODMP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTODMP_ROOT))

from dreamplace import NonLinearPlace, Placer, placer_cli


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_def_component_placements(path: Path) -> dict[str, dict[str, Any]]:
    """Parse COMPONENT placement records without assuming one record per line."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    start_marker = "COMPONENTS"
    end_marker = "END COMPONENTS"
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError(f"{path}: missing DEF COMPONENTS section")
    section = text[start:end]
    records: dict[str, dict[str, Any]] = {}
    for raw_record in section.split(";"):
        fields = raw_record.strip().split()
        if len(fields) < 3 or fields[0] != "-":
            continue
        name, master = fields[1], fields[2]
        placement = None
        for status in ("FIXED", "PLACED", "COVER"):
            marker = f"+ {status}"
            marker_index = raw_record.find(marker)
            if marker_index < 0:
                continue
            tail = raw_record[marker_index + len(marker) :]
            left = tail.find("(")
            right = tail.find(")", left + 1)
            if left < 0 or right < 0:
                continue
            coordinate_fields = tail[left + 1 : right].split()
            if len(coordinate_fields) < 2:
                continue
            orientation_fields = tail[right + 1 :].split()
            placement = {
                "status": status,
                "x": int(coordinate_fields[0]),
                "y": int(coordinate_fields[1]),
                "orient": orientation_fields[0] if orientation_fields else "N",
            }
            break
        records[name] = {
            "name": name,
            "master": master,
            "placement": placement,
        }
    return records


def coordinate_sha256(
    records: dict[str, dict[str, Any]],
    names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    selected = sorted(records if names is None else set(names))
    rows = []
    for name in selected:
        record = records.get(name)
        if record is None or record.get("placement") is None:
            rows.append(f"{name}\tMISSING")
            continue
        placement = record["placement"]
        rows.append(
            "\t".join(
                (
                    name,
                    str(record.get("master", "")),
                    str(placement["status"]),
                    str(placement["x"]),
                    str(placement["y"]),
                    str(placement["orient"]),
                )
            )
        )
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def xy_coordinate_sha256(
    records: dict[str, dict[str, Any]],
    names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    selected = sorted(records if names is None else set(names))
    rows = []
    for name in selected:
        placement = (records.get(name) or {}).get("placement")
        if placement is None:
            rows.append(f"{name}\tMISSING")
        else:
            rows.append(f"{name}\t{placement['x']}\t{placement['y']}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _preservation_failures(
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
    names: list[str],
    label: str,
) -> list[str]:
    failures = []
    for name in names:
        before = source.get(name)
        after = output.get(name)
        if before is None or after is None:
            failures.append(f"{label}_component_missing:{name}")
            continue
        if before.get("master") != after.get("master"):
            failures.append(f"{label}_master_changed:{name}")
        if before.get("placement") != after.get("placement"):
            failures.append(f"{label}_placement_changed:{name}")
    return failures


def _placer_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        str(args.params_json),
        "--flow-kind",
        "placement",
        "--place-io-engine",
        "openroad",
        "--workspace",
        str(args.workspace),
        "--result-dir",
        str(args.result_dir),
        "--base-design-name",
        args.case,
        "--def-input",
        str(args.source_def),
        "--verilog-input",
        str(args.verilog),
        "--sdc",
        str(args.sdc),
        "--rc-tcl",
        str(args.rc_tcl),
        "--tech-lef",
        str(args.tech_lef),
        "--gpu",
        "0",
        "--enable-fillers",
        "0",
        "--random-center-init",
        "1",
    ]
    for lef in args.lef:
        argv.extend(("--lef", str(lef)))
    for liberty in args.lib:
        argv.extend(("--lib", str(liberty)))
    return argv


def materialize_random_init_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    parser = placer_cli.build_arg_parser()
    placer_args = parser.parse_args(_placer_argv(args))
    params, launch = placer_cli.build_effective_params_from_args(placer_args)
    params.random_seed = int(args.seed)
    params.random_center_init_flag = 1
    params.enable_fillers = 0
    params.legalize_flag = 0
    params.detailed_place_flag = 0
    params.routability_opt_flag = 0
    params.get_congestion_map = 0
    params.plot_flag = 0
    params.with_sta = 0

    engine = Placer.PlacementEngine(params)
    engine.setup_rawdb(
        data_manager=types.SimpleNamespace(dir_workspace=launch["workspace"])
    )
    engine.setup_placedb()

    num_movable = int(engine.placedb.num_movable_nodes)
    num_nodes = int(engine.placedb.num_nodes)
    source_x = np.asarray(engine.placedb.node_x[:num_movable]).copy()
    source_y = np.asarray(engine.placedb.node_y[:num_movable]).copy()
    node_names = [
        _decode_name(value)
        for value in list(engine.placedb.node_names)[:num_movable]
    ]

    engine.placer = NonLinearPlace.NonLinearPlace(params, engine.placedb, None)
    macro_mask_tensor = getattr(
        engine.placer.data_collections,
        "movable_macro_mask",
        None,
    )
    if macro_mask_tensor is None:
        macro_mask = np.zeros(num_movable, dtype=np.bool_)
    else:
        macro_mask = np.asarray(
            macro_mask_tensor.detach().cpu().numpy(),
            dtype=np.bool_,
        ).reshape(-1)[:num_movable]
    if macro_mask.size != num_movable:
        raise ValueError("movable macro mask length does not match movable nodes")

    pos = engine.placer.pos[0]
    standard_mask = ~macro_mask
    with torch.no_grad():
        runtime_x = pos[:num_movable]
        runtime_y = pos[num_nodes : num_nodes + num_movable]
        if macro_mask.any():
            macro_indices = torch.as_tensor(
                np.flatnonzero(macro_mask),
                device=pos.device,
                dtype=torch.long,
            )
            runtime_x.index_copy_(
                0,
                macro_indices,
                torch.as_tensor(
                    source_x[macro_mask],
                    device=pos.device,
                    dtype=pos.dtype,
                ),
            )
            runtime_y.index_copy_(
                0,
                macro_indices,
                torch.as_tensor(
                    source_y[macro_mask],
                    device=pos.device,
                    dtype=pos.dtype,
                ),
            )
        runtime_x_np = runtime_x.detach().cpu().numpy().copy()
        runtime_y_np = runtime_y.detach().cpu().numpy().copy()

    moved_standard = np.logical_or(
        np.abs(runtime_x_np[standard_mask] - source_x[standard_mask]) > 1.0e-6,
        np.abs(runtime_y_np[standard_mask] - source_y[standard_mask]) > 1.0e-6,
    )
    macro_names = [name for name, selected in zip(node_names, macro_mask) if selected]
    standard_names = [
        name for name, selected in zip(node_names, standard_mask) if selected
    ]

    args.output_def.parent.mkdir(parents=True, exist_ok=True)
    engine.write_back(str(args.output_def))

    source_records = parse_def_component_placements(args.source_def)
    output_records = parse_def_component_placements(args.output_def)
    fixed_names = sorted(
        name
        for name, record in source_records.items()
        if (record.get("placement") or {}).get("status") in {"FIXED", "COVER"}
    )
    failures = []
    failures.extend(
        _preservation_failures(
            source_records,
            output_records,
            macro_names,
            "movable_macro",
        )
    )
    failures.extend(
        _preservation_failures(
            source_records,
            output_records,
            fixed_names,
            "fixed_object",
        )
    )
    missing_standard = sorted(set(standard_names) - set(output_records))
    if missing_standard:
        failures.append(f"missing_randomized_standard_cells:{len(missing_standard)}")
    if standard_names and int(moved_standard.sum()) == 0:
        failures.append("no_standard_cell_coordinate_changed")

    input_paths = [
        args.source_def,
        args.verilog,
        args.sdc,
        args.rc_tcl,
        args.tech_lef,
        *args.lef,
        *args.lib,
    ]
    manifest = {
        "artifact": "random_center_r0_manifest",
        "artifact_version": 1,
        "status": "pass" if not failures else "failed",
        "case": args.case,
        "seed": int(args.seed),
        "randomization": "center_local_gaussian_std_cells_only",
        "gaussian_scale_fraction": 0.001,
        "source_def": str(args.source_def),
        "source_def_sha256": sha256_file(args.source_def),
        "r0_def": str(args.output_def),
        "r0_def_sha256": sha256_file(args.output_def),
        "counts": {
            "movable_nodes": num_movable,
            "randomized_standard_cells": len(standard_names),
            "moved_standard_cells": int(moved_standard.sum()),
            "preserved_movable_macros": len(macro_names),
            "preserved_fixed_objects": len(fixed_names),
        },
        "coordinate_identity": {
            "all_components_sha256": coordinate_sha256(output_records),
            "movable_nodes_sha256": coordinate_sha256(output_records, node_names),
            "movable_nodes_xy_sha256": xy_coordinate_sha256(
                output_records,
                node_names,
            ),
            "randomized_standard_cells_sha256": coordinate_sha256(
                output_records,
                standard_names,
            ),
            "movable_macros_sha256": coordinate_sha256(
                output_records,
                macro_names,
            ),
            "fixed_objects_sha256": coordinate_sha256(output_records, fixed_names),
        },
        "input_sha256": {
            str(path): sha256_file(path) for path in input_paths
        },
        "failures": failures,
    }
    _write_json(args.manifest, manifest)
    if failures:
        raise RuntimeError("R0 validation failed: " + "; ".join(failures[:8]))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-def", type=Path, required=True)
    parser.add_argument("--verilog", type=Path, required=True)
    parser.add_argument("--sdc", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--tech-lef", type=Path, required=True)
    parser.add_argument("--lef", type=Path, action="append", default=[])
    parser.add_argument("--lib", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-def", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    materialize_random_init_snapshot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
