"""Two real master commits and a failed native refresh through PlaceIO.

The fixture is the real-library three-cell through case. Every checkpoint
builds the production timing factory and evaluates a zero-wire forward. This
tests native mutation and model lifetimes, not placement or topology mutation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import preflight
import torch
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
from dreamplace.Params import Params
from fixed_sta import production_timing
from structure_scale import NUMERIC_PHASES


def run(args):
    inputs = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    native = preflight.load_native(args.native_module, args.ecc_root)
    from chipcompiler.tools.ecc.module import ECCToolsModule

    torch.set_num_threads(args.threads)
    module = ECCToolsModule()
    report = {
        "status": "running",
        "scope": __doc__,
        "native_path": str(Path(native.__file__).resolve()),
        "native_sha256": preflight.file_digest(args.native_module),
        "inputs": {str(args.manifest): preflight.file_digest(args.manifest)},
        "checkpoints": [],
    }
    target = args.output / "refresh_results.json"
    try:
        module.init_config("", args.output / "native", args.output / "feature")
        module.init_techlef(inputs["tech_lef"])
        module.init_lefs(inputs["lefs"])
        if not module.read_def(inputs["def"]):
            raise RuntimeError("native DEF read failed")
        params = Params()
        params.with_sta = True
        params.timing_hold_enabled = True
        params.design_inputs = {
            "lib_paths": inputs["libs"],
            "sdc": inputs["sdc"],
            "work_dir": str(args.output / "sta"),
            "timing_mode": "model_only",
            "thread_number": args.threads,
        }
        backend = PlaceIOFunction.read(params, module)
        zeros = None
        previous = []

        def checkpoint(expected_master):
            nonlocal zeros
            generation = backend.refresh_generation
            output = args.output / f"generation-{generation}"
            output.mkdir()
            view = preflight.save_export(backend.pydb, output)
            timing = production_timing(
                view, dict(np.load(output / "base_arrays.npz")), "cpu", backend=backend
            )
            if timing.timing_model_lease is None:
                raise RuntimeError("production factory lost the native generation lease")
            zeros = torch.zeros(timing.num_pins, dtype=torch.float64)
            caps = torch.as_tensor(view.arrays["pin_reference_cap"], dtype=torch.float64)
            values = timing(
                {"rise": zeros, "fall": zeros},
                {"rise": zeros, "fall": zeros},
                {"rise": caps[:, 0], "fall": caps[:, 1]},
                surrogate_mode="lut_only",
            )
            if timing._sparse_runtime is None or not all(
                torch.isfinite(torch.as_tensor(v)).all() for v in values
            ):
                raise RuntimeError("production factory did not evaluate finite sparse timing")
            for old in previous:
                try:
                    old({}, {}, {})
                except RuntimeError as error:
                    if "invalid or replaced native generation" not in str(error):
                        raise
                else:
                    raise RuntimeError("a previous tensor generation remained executable")
            pydb = backend.pydb
            phase_calls = dict(pydb.timing_preparation)["phase_calls"]
            if any(phase_calls[name] for name in NUMERIC_PHASES):
                raise RuntimeError("model-only refresh executed a full numeric timing phase")
            arrays = view.arrays
            state_keys = sorted(
                (
                    str(pydb.pin_names[pin]),
                    view.domains[domain],
                    view.tags[tag],
                    view.launch_tags[launch],
                    int(source),
                    int(role),
                    valid.tolist(),
                )
                for pin, domain, tag, launch, source, role, valid in zip(
                    arrays["state_pin"],
                    arrays["state_domain"],
                    arrays["state_tag"],
                    arrays["state_launch_tag"],
                    arrays["state_source_type"],
                    arrays["state_role"],
                    arrays["state_valid"],
                    strict=True,
                )
            )
            node = list(pydb.node_names).index("a")
            cell = int(pydb.main_id_2_cell_id_start[int(pydb.inst_main_id[node])])
            cell += int(pydb.inst_libcell_offset[node])
            actual_master = str(pydb.flat_libcell_info[cell][0])
            if actual_master != expected_master:
                raise RuntimeError("committed master was not reflected in the new native model")
            report["checkpoints"].append(
                {
                    "generation": generation,
                    "master": actual_master,
                    "state_key_digest": view.state_key_digest,
                    "named_state_key_digest": hashlib.sha256(
                        json.dumps(state_keys).encode()
                    ).hexdigest(),
                    "model_digest": view.model_digest,
                    "phase_calls": phase_calls,
                    "wns_ps": float(timing._sparse_runtime.last_result.wns),
                    "tns_ps": float(timing._sparse_runtime.last_result.tns),
                    "rejected_old_generations": len(previous),
                }
            )
            previous.append(timing)
            target.write_text(json.dumps(report, indent=2))

        checkpoint("BUFX2H7R")
        for master in ("BUFX4H7R", "BUFX1H7R"):
            pydb = backend.pydb
            names = [str(row[0]) for row in pydb.flat_libcell_info]
            cells = [
                int(pydb.main_id_2_cell_id_start[int(main)]) + int(offset) if int(main) >= 0 else -1
                for main, offset in zip(pydb.inst_main_id, pydb.inst_libcell_offset, strict=True)
            ]
            cells[list(pydb.node_names).index("a")] = names.index(master)
            mutation = PlaceIOFunction.apply_sizing(backend, cells, names)
            if not mutation["ok"] or mutation["accepted_count"] != 1:
                raise RuntimeError("native master transaction was not accepted")
            PlaceIOFunction.refresh(backend)
            checkpoint(master)

        if len({row["named_state_key_digest"] for row in report["checkpoints"]}) != 1:
            raise RuntimeError("equivalent buffer masters changed named state identities")
        if len({row["model_digest"] for row in report["checkpoints"]}) != 3:
            raise RuntimeError("master/cap changes did not rebuild the model identity")

        bad_sdc = args.output / "invalid-refresh.sdc"
        bad_sdc.write_text(Path(inputs["sdc"]).read_text() + "\ninvalid_sdc_for_refresh_test\n")
        backend.timing_inputs["sdc"] = str(bad_sdc)
        try:
            PlaceIOFunction.refresh(backend)
        except RuntimeError as error:
            report["injected_native_failure"] = str(error)
        else:
            raise RuntimeError("invalid SDC unexpectedly published a generation")
        if backend.refresh_generation != 2 or backend.pydb is not None or backend.timing_enabled:
            raise RuntimeError("failed refresh published partial or stale state")
        for old in previous:
            try:
                old({}, {}, {})
            except RuntimeError as error:
                if "invalid or replaced native generation" not in str(error):
                    raise
            else:
                raise RuntimeError("failed native refresh left an executable timing operator")
        report["rejected_after_failed_refresh"] = len(previous)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2))
        module.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "ecc-root", "native-module"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
