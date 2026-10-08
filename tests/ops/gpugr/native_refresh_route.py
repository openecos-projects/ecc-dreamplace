"""Bounded native route/master/refresh/buffer qualification on the GCD fixture.

Run explicitly with --manifest (ECC-Tools fixture manifest), --backend and
--output. Three routes exercise the same production PlaceDB and mutation owners.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chipcompiler.tools.ecc.module import ECCToolsModule
from dreamplace.Placer import PlacementEngine
from dreamplace.placer_cli import _load_params_json
from dreamplace.ops.buffer_insertion.buffering_lane import BufferCommitRequest, _action_digest
from dreamplace.ops.placeio_common.physical_mutation import (
    physical_mutation_backend_for,
    write_native_def,
)
from dreamplace.ops.routability.gpugr_context import (
    build_parser_cache_inputs,
    get_cached_gpugr_operator,
    write_back_movable_lpos,
)
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo


def name(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def master(db, node):
    main = int(db.inst_main_id[node])
    cell = int(db.main_id_2_cell_id_start[main]) + int(db.inst_libcell_offset[node])
    return name(db.flat_libcell_names[cell])


def snapshot(db):
    return {
        "nodes": {
            name(n): master(db, i)
            for i, n in enumerate(db.node_names)
            if int(db.inst_main_id[i]) >= 0
        },
        "pins": {
            name(n): [
                name(db.net_names[int(db.pin2net_map[i])]),
                float(db.pin_offset_x[i]),
                float(db.pin_offset_y[i]),
            ]
            for i, n in enumerate(db.pin_names)
            if int(db.pin2net_map[i]) >= 0
        },
        "counts": [int(db.num_nodes), len(db.pin_names), len(db.net_names)],
    }


def check_steiner():
    pos = torch.tensor([1.0, 11.0, 7.0, 2.0, 13.0, 8.0], requires_grad=True)
    results = []
    for deterministic in (False, True):
        topo = SteinerTopo(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            deterministic_flag=deterministic,
        )
        topo.rebuild_tree(pos)
        topo.freeze_topology()
        x, y = topo(pos)
        gradient = torch.autograd.grad(x.sum() + y.sum(), pos)[0]
        assert gradient.shape == pos.shape and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0 and topo.forward_count == 1
        results.append(
            {
                "deterministic": deterministic,
                "vertices": x.numel(),
                "gradient": gradient.tolist(),
                "generation": topo.topology_generation,
            }
        )
    return results


def make_engine(args, overrides=None):
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    pdk, inputs = manifest["pdk"], manifest["inputs"]
    parent = Path(__file__).resolve().parents[6]
    params = _load_params_json(
        parent / "chipcompiler/tools/ecc_dreamplace/configs/dreamplace_ecc.json"
    )
    params.fromJson(
        {
            "place_io_engine": args.backend,
            "base_design_name": "gcd",
            "gpu": 0,
            "num_threads": 2,
            "random_seed": 3000,
            "with_sta": 1,
            "routability_opt_flag": 1,
            "gpugr_backend": "cpu_pr_mt",
            "route_num_bins_x": 32,
            "route_num_bins_y": 32,
            "auto_adjust_bins": 0,
            "num_bins_x": 32,
            "num_bins_y": 32,
            "enable_fillers": 0,
            "cell_padding_x": 0,
            "macro_halo_x": 0,
            "macro_halo_y": 0,
            "target_density": 0.8,
            "plot_flag": 0,
            "random_center_init_flag": 0,
            "result_dir": str(args.output),
            "def_input": inputs["def"],
            "lef_input": [pdk["tech_lef"], *pdk["lefs"]],
            "verilog_input": inputs["verilog"],
            "design_inputs": {
                "tech_lef": pdk["tech_lef"],
                "lef": pdk["lefs"],
                "lib": pdk["typical_libs"],
                "sdc": pdk["sdc"],
                "def": inputs["def"],
                "verilog": inputs["verilog"],
                "rc_tcl": str(args.rc_tcl),
                "work_dir": str(args.output / "sta"),
            },
        }
    )
    params.fromJson(overrides or {})
    module = None
    if args.backend == "ecc":
        module = ECCToolsModule()
        module.init_config(
            manifest["config"]["db_ecc"], args.output / "native", args.output / "feature"
        )
        module.init_techlef(pdk["tech_lef"])
        module.init_lefs(pdk["lefs"])
        assert module.read_def(inputs["def"])
    engine = PlacementEngine(params)
    engine.setup_rawdb(module)
    engine.setup_placedb()
    return engine, module


def run(args):
    engine, module = make_engine(args)
    db = engine.placedb
    records = []

    def route(label):
        pos = torch.from_numpy(np.concatenate([db.node_x, db.node_y])).clone()
        lpos, names = build_parser_cache_inputs(pos, engine.params, db)
        write_back_movable_lpos(pos, engine.params, db)
        operator = get_cached_gpugr_operator(engine.params, db)
        assert not operator.parser_cache_would_hit("custom", "gcd", names, len(names))
        result = operator.run_gpugr(
            out_dir=str(args.output / label),
            design_name="gcd",
            threads=2,
            route_xsize=32,
            route_ysize=32,
            rrr_iters=0,
            backend="cpu_pr_mt",
            keep_temp_def=True,
            save_artifacts=True,
            parser_cache_enable=True,
            parser_cache_node_lpos=lpos,
            parser_cache_node_names=names,
        )
        assert result["maps"] and all(torch.isfinite(v).all() for v in result["maps"].values())
        assert operator.parser_cache_would_hit("custom", "gcd", names, len(names))
        records.append(
            {
                "stage": label,
                "generation": db.topology_generation,
                "snapshot": snapshot(db.pydb),
                "metrics": result["metrics"],
                "artifacts": result.get("artifact_paths", {}),
            }
        )
        write_native_def(db, args.output / f"{label}.def")
        return operator

    old_operator = route("initial")
    cell_names = [name(c) for c in db.flat_libcell_names]
    chosen = next(
        (i for i in range(db.num_movable_nodes) if master(db.pydb, i) == "BUFX1P4H7L"), None
    )
    assert chosen is not None and "BUFX3H7L" in cell_names
    selected_name = name(db.node_names[chosen])
    db.inst_cell_id[chosen] = cell_names.index("BUFX3H7L")
    sizing = db.write_sizing_back()
    assert sizing["accepted_count" if args.backend == "ecc" else "applied"] == 1, sizing
    mutation_backend = physical_mutation_backend_for(db)
    mode = "full_rebuild" if args.backend == "ecc" else "all"
    sizing_refresh = mutation_backend.refresh(refresh_mode=mode, rebuild_mode=mode)
    assert old_operator._parser_db_cache is None
    after_sizing_operator = route("after_sizing")
    assert after_sizing_operator is not old_operator
    assert records[0]["snapshot"]["counts"] == records[1]["snapshot"]["counts"]
    assert records[1]["snapshot"]["nodes"][selected_name] == "BUFX3H7L"
    native = db.pydb
    clock_nets = {
        name(native.net_names[int(native.pin2net_map[int(pin)])])
        for pin in native.clock_pins
        if int(native.pin2net_map[int(pin)]) >= 0
    }
    signal_net = next(
        i
        for i, n in enumerate(native.net_names)
        if name(n) not in clock_nets and len(native.net2pin_map[i]) >= 3
    )
    pins = [int(i) for i in native.net2pin_map[signal_net]]
    driver = int(native.flat_net2pin_map[int(native.flat_net2pin_start_map[signal_net])])
    loads = [i for i in pins if i != driver][:2]
    action = {
        "action_id": 0,
        "action_kind": "buffer_insert",
        "net_name": name(native.net_names[signal_net]),
        "buffer_main_type_index": 0,
        "bsu": 0,
        "buffer_master_id": cell_names.index("BUFX3H7L"),
        "buffer_master_name": "BUFX3H7L",
        "candidate_location_x_dbu": 10000,
        "candidate_location_y_dbu": 10000,
        "driver_pin_name": name(native.pin_names[driver]).replace(":", "/"),
        "load_pin_name": name(native.pin_names[loads[0]]).replace(":", "/"),
        "downstream_pin_names": [name(native.pin_names[i]).replace(":", "/") for i in loads],
    }
    request = BufferCommitRequest(
        version=1,
        mode="candidate",
        actions=(action,),
        action_digest=_action_digest((action,)),
        iteration=0,
        reason="native-route-refresh",
        mutation_kind="topology-changing",
        commit_enabled=True,
        committed_def_path=str(args.output / "buffer_committed.def"),
        refresh_mode=mode,
        rebuild_mode=mode,
        projection_metadata={},
        runtime_refresh_summary={},
    )
    buffer_commit = mutation_backend.commit_buffers(request, output_dir=str(args.output / "buffer"))
    assert buffer_commit["accepted_action_count"] == 1, buffer_commit
    buffer_refresh = mutation_backend.refresh(refresh_mode=mode, rebuild_mode=mode)
    assert after_sizing_operator._parser_db_cache is None
    route("after_buffer")
    assert records[2]["snapshot"]["counts"][0] == records[1]["snapshot"]["counts"][0] + 1
    for pin in action["downstream_pin_names"]:
        canonical = pin.replace("/", ":")
        assert records[2]["snapshot"]["pins"][canonical][0] != action["net_name"]
    report = {
        "backend": args.backend,
        "routes": records,
        "sizing": sizing,
        "sizing_refresh": sizing_refresh,
        "buffer_commit": buffer_commit,
        "buffer_refresh": buffer_refresh,
        "capabilities": db.backend_caps,
        "steiner": check_steiner(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if module:
        module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("ecc", "openroad"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    run(parser.parse_args())
