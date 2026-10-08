"""Gate-A probe of real GGR routing, preserving missing pin access evidence.

Run with the same Python/Torch and loader as the isolated Xplace build.
This is qualification tooling, not the production RC implementation.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from route_fixture import make_route_fixture


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_graph(pack):
    """Independent diagnostic checks; never repair disconnected/cyclic routes."""
    pin_net = pack["pin_net"]
    vertex_net = pack["vertex_net"]
    pin_vertex = pack["pin_vertex"]
    edge_from, edge_to = pack["edge_from"], pack["edge_to"]
    np.testing.assert_equal(vertex_net[edge_from], vertex_net[edge_to])
    mapped = pin_vertex >= 0
    np.testing.assert_equal(vertex_net[pin_vertex[mapped]], pin_net[mapped])
    adjacency = [[] for _ in vertex_net]
    for u, v in zip(edge_from, edge_to, strict=True):
        adjacency[u].append(v)
        adjacency[v].append(u)
    net_pins = [[] for _ in pack["net_names"]]
    for pin, net in enumerate(pin_net):
        if net >= 0:
            net_pins[net].append(pin)
    counts = {
        "nets": len(net_pins),
        "pins": len(pin_net),
        "vertices": len(vertex_net),
        "edges": len(edge_from),
        "wire_edges": int(np.count_nonzero(pack["edge_kind"] == 0)),
        "via_edges": int(np.count_nonzero(pack["edge_kind"] == 1)),
        "missing_local_paths": int(pack["missing_local_path_count"]),
        "clamped_pin_layers": int(pack["clamped_pin_layer_count"]),
        "duplicate_edges_removed": int(pack["duplicate_edge_count"]),
    }
    issues = {}
    examples = {}
    for net, members in enumerate(net_pins):
        drivers = [p for p in members if pack["pin_is_driver"][p]]
        reasons = []
        if len(drivers) != 1:
            reasons.append("driver_count")
        if any(pin_vertex[p] < 0 for p in members):
            reasons.append("missing_attachment")
        if any(pack["pin_local_path_missing"][p] for p in members):
            reasons.append("missing_local_access_path")
        if pack["net_status"][net] >= 2:
            reasons.append("failed_or_unrouted")
        begin, end = pack["net_vertex_start"][net : net + 2]
        start, finish = pack["net_edge_start"][net : net + 2]
        seen = set()
        if drivers and pin_vertex[drivers[0]] >= 0:
            pending = [int(pin_vertex[drivers[0]])]
            while pending:
                v = pending.pop()
                if v not in seen:
                    seen.add(v)
                    pending.extend(adjacency[v])
        if end > begin and len(seen) != end - begin:
            reasons.append("disconnected")
        if finish - start > end - begin - 1:
            reasons.append("cycle")
        for reason in reasons:
            issues[reason] = issues.get(reason, 0) + 1
            examples.setdefault(reason, [])
            if len(examples[reason]) < 3:
                examples[reason].append(pack["net_names"][net])
    return {"counts": counts, "issues": issues, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xplace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fixture", choices=("branched", "small-pins"), default="branched")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.xplace / "build_cpu_pr"
    io_path = next((build / "cpp_to_py/io_parser").glob("io_parser*.so"))
    gr_path = next((build / "cpp_to_py/gpugr").glob("gpugr*.so"))
    io = load_extension("io_parser", io_path)
    gr = load_extension("gpugr", gr_path)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gr_native_access_") as temporary:
        work = Path(temporary)
        if args.config:
            config = json.loads(args.config.read_text())
            inputs = config["INPUT"]
            lefs = [inputs["tech_lef_path"], *inputs["lef_paths"]]
            source_def = Path(inputs["def_path"])
            def_path = work / "input.def"
            if source_def.suffix == ".gz":
                with gzip.open(source_def, "rt") as stream:
                    def_path.write_text(stream.read())
            else:
                def_path.write_bytes(source_def.read_bytes())
            bottom, top, grid, threads = "MET2", "MET5", 512, 4
        else:
            lef, design = make_route_fixture(args.xplace, small_pins=args.fixture == "small-pins")
            lef_path, def_path = work / "toy.lef", work / "toy.def"
            lef_path.write_text(lef)
            def_path.write_text(design)
            lefs = [str(lef_path)]
            bottom, top, grid, threads = "M2", "M3", 8, 2
        assert io.load_params(
            {
                "lefs": lefs,
                "def": str(def_path),
                "lite_mode": True,
                "random_place": False,
                "num_threads": threads,
            }
        )
        rawdb = io.create_database()
        rawdb.load()
        rawdb.setup()
        gpdb = io.create_gpdatabase(rawdb)
        gpdb.setup()
        gr.read_flute(
            str(args.xplace / "thirdparty/flute/POWV9.dat"),
            str(args.xplace / "thirdparty/flute/POST9.dat"),
        )
        assert gr.load_gr_params(
            {
                "backend": "cpu_pr_mt",
                "rrrIters": 0,
                "threads": threads,
                "route_xSize": grid,
                "route_ySize": grid,
                "bottom_routing_layer": bottom,
                "top_routing_layer": top,
            }
        )
        grdb = gr.create_grdatabase(rawdb, gpdb)
        router = gr.create_routeforce(grdb)
        router.run_ggr()
        pack = router.timing_route_pack()
        report = audit_graph(pack)
        report.update(
            {
                "backend": "cpu_pr_mt",
                "router_calls": 1,
                "run_stats": dict(router.run_stats()),
                "dbu_per_micron": pack["dbu_per_micron"],
                "pin_access_model": pack["pin_access_model"],
                "extension_path": str(gr_path),
                "extension_sha256": hashlib.sha256(gr_path.read_bytes()).hexdigest(),
                "input_def_sha256": hashlib.sha256(def_path.read_bytes()).hexdigest(),
                "routing_layers": [bottom, top],
                "grid": [grid, grid],
                "threads": threads,
                "elapsed_seconds": time.perf_counter() - started,
                "gr_rc_qualified": False,
            }
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        arrays = {k: v for k, v in pack.items() if isinstance(v, np.ndarray)}
        np.savez(args.output / "native-route-pack.npz", **arrays)
        (args.output / "route-identities.json").write_text(
            json.dumps({k: pack[k] for k in ("pin_names", "net_names", "layer_names")}, indent=2)
            + "\n"
        )
        print("GR_NATIVE_ROUTE_REPORT=" + json.dumps(report), flush=True)
        if not args.config:
            assert report["counts"]["via_edges"] > 0
            graph_errors = ("driver_count", "missing_attachment", "disconnected", "cycle")
            assert not any(k in report["issues"] for k in graph_errors), report
            if args.fixture == "branched":
                assert report["issues"] == {}, report
            else:
                assert report["issues"] == {"missing_local_access_path": 1}, report


if __name__ == "__main__":
    main()
