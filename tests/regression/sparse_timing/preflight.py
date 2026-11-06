"""Read a frozen ECC input and persist sparse metadata without optimizing it.

The manifest supplies tech_lef, lefs, libs, def, sdc and Tcl sources (a mapping
from relative source path to its original file). No input is rewritten.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dreamplace.ops.placeio_ecc.sparse_timing_schema import read_sparse_timing_view

BASE_FIELDS = (
    "pin_names",
    "pin2net_map",
    "net_flat_arcs",
    "net2driver_pin_map",
    "flat_inst_arcs_by_level",
    "flat_inst_arcs_by_level_start",
    "clock_pins",
    "end_points",
    "endpoints_timing_check_arcs",
    "clk_pin_rtran",
    "clk_pin_ftran",
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def padded_rows(rows, fill):
    width = max((len(row) for row in rows), default=0)
    values = np.full((len(rows), width), fill, dtype=np.float64)
    for index, row in enumerate(rows):
        values[index, : len(row)] = row
    return values


def load_native(native_module, ecc_root):
    sys.path.insert(0, str(ecc_root.resolve()))
    import ecc_tools_bin

    spec = importlib.util.spec_from_file_location("ecc_tools_bin.ecc_py", native_module)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sys.modules[spec.name] = native
    ecc_tools_bin.ecc_py = native
    return native


def save_export(pydb, output):
    native = pydb.timing_sparse
    np.savez_compressed(output / "sparse_arrays.npz", **native["arrays"])
    metadata = {key: value for key, value in native.items() if key != "arrays"}
    (output / "sparse_metadata.json").write_text(json.dumps(metadata, indent=2))
    base = {name: np.asarray(getattr(pydb, name)) for name in BASE_FIELDS}
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        for suffix in ("values", "trans_table", "cap_table", "dim"):
            name = f"{prefix}_flat_luts_{suffix}"
            rows = getattr(pydb, name)
            base[name] = (
                np.asarray(rows, dtype=np.int64)
                if suffix == "dim"
                else padded_rows(rows, 0 if suffix == "values" else np.inf)
            )
    np.savez_compressed(output / "base_arrays.npz", **base)
    return read_sparse_timing_view(pydb)


def run(args, *, export=None):
    inputs = json.loads(args.manifest.read_text())
    paths = [inputs[name] for name in ("tech_lef", "def", "sdc")]
    paths += inputs["lefs"] + inputs["libs"] + list(inputs["sources"].values())
    paths += [inputs[name] for name in ("sta_config", "hierarchy_verilog") if inputs.get(name)]
    provenance = {
        "inputs": {str(Path(path).resolve()): file_digest(path) for path in paths},
        "native_module": str(args.native_module.resolve()),
        "native_sha256": file_digest(args.native_module),
        "dreamplace_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "thread_number": args.threads,
        "requested_timing_mode": args.timing_mode,
        "timing_relative_tolerance": 0.03,
        "timing_absolute_floor_ps": 2.0,
        "status": "running",
        "phase_seconds": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / "manifest.json"
    report.write_text(json.dumps(provenance, indent=2))

    def measured(name, operation, *values, **options):
        provenance["stage"] = name
        report.write_text(json.dumps(provenance, indent=2))
        started = time.perf_counter()
        try:
            return operation(*values, **options)
        finally:
            provenance["phase_seconds"][name] = time.perf_counter() - started

    for relative, source in inputs["sources"].items():
        target = args.output / relative
        target.resolve().relative_to(args.output.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(source).resolve())
    native = load_native(args.native_module, args.ecc_root)
    from chipcompiler.tools.ecc.module import ECCToolsModule

    print("NATIVE_MODULE", native.__file__, flush=True)
    module = ECCToolsModule()
    start = time.perf_counter()
    os.chdir(args.output)
    try:
        for name in ("output", "feature"):
            (args.output / name).mkdir()
        measured(
            "init_config", module.init_config, "", args.output / "output", args.output / "feature"
        )
        measured("read_techlef", module.init_techlef, inputs["tech_lef"])
        measured("read_lefs", module.init_lefs, inputs["lefs"])
        if not measured("read_def", module.read_def, inputs["def"]):
            raise RuntimeError("native DEF read failed")
        summary = measured(
            "prepare_timing",
            module.prepare_place_timing,
            {
                "lib_paths": inputs["libs"],
                "sdc": inputs["sdc"],
                "spef": inputs.get("spef", ""),
                "work_dir": args.output / "sta",
                "thread_number": args.threads,
                "max_paths": 20,
                "timing_mode": args.timing_mode,
                "config": inputs.get("sta_config", ""),
            },
        )
        pydb = measured(
            "materialize",
            module.pydb,
            module.get_dmInst_ptr(),
            512,
            512,
            routability_opt_flag=False,
            with_sta=True,
            timing_model_only=args.timing_mode == "model_only",
        )
        provenance["timing_summary"] = summary
        provenance["native_preparation"] = pydb.timing_preparation
        provenance["schema_version"] = pydb.timing_schema_version
        provenance["pin_count"] = len(pydb.pin_names)
        provenance["endpoint_count"] = len(pydb.end_points)
        provenance["clock_pin_count"] = len(pydb.clock_pins)
        view = measured(
            "inspect_export", save_export if export is None else export, pydb, args.output
        )
        provenance["sparse"] = view.statistics()
        provenance["status"] = "unsupported" if view.unsupported else "export_validated"
        print(json.dumps(provenance, indent=2), flush=True)
    except Exception as error:
        provenance["status"] = "failed"
        provenance["error"] = str(error)
        raise
    finally:
        provenance["elapsed_seconds"] = time.perf_counter() - start
        report.write_text(json.dumps(provenance, indent=2))
        module.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ecc-root", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timing-mode", choices=("complete", "model_only"), default="complete")
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
