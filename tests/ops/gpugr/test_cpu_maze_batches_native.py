"""Stress logical RRR batches and native RC export on a congested toy grid.

Cells deliberately overlap to produce routing pressure. This fixture exercises
routing state and connectivity, rather than placement legality or design QoR.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity


@pytest.fixture(scope="module")
def multi_batch_route(tmp_path_factory):
    path = Path(__file__).resolve().parents[2] / "flows/gr_sizing_fixture.py"
    spec = importlib.util.spec_from_file_location("multi_batch_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    directory = fixture.make_fixture(tmp_path_factory.mktemp("maze_batches") / "inputs")
    components, nets = [], []
    for index in range(65):
        components.extend(
            [
                f"- s{index} BUFX1H7R + PLACED ( 1000 1000 ) N ;",
                f"- t{index} BUFX1H7R + PLACED ( 8000 8000 ) N ;",
            ]
        )
        nets.append(f"- n{index} ( s{index} Y ) ( t{index} A ) ;")
    design = (
        'VERSION 5.8 ; DIVIDERCHAR "/" ; BUSBITCHARS "[]" ; DESIGN many ;\n'
        "UNITS DISTANCE MICRONS 1000 ; DIEAREA ( 0 0 ) ( 10000 10000 ) ;\n"
        "ROW R core 0 0 N DO 10 BY 1 STEP 1000 0 ;\n"
        "TRACKS Y 500 DO 10 STEP 1000 LAYER M1 M3 ;\n"
        "TRACKS X 500 DO 10 STEP 1000 LAYER M2 ;\n"
        f"COMPONENTS {len(components)} ;\n"
        + "\n".join(components)
        + "\nEND COMPONENTS\nPINS 0 ; END PINS\n"
        + f"NETS {len(nets)} ;\n"
        + "\n".join(nets)
        + "\nEND NETS\nEND DESIGN\n"
    )
    (directory / "many.def").write_text(design)
    backend = XplaceGPUGR(None, None)
    _, gr, *_ = backend._import_xplace_modules(resolved_backend="cpu_pr_maze")
    from cpp_to_py import io_parser as io

    assert io.load_params(
        {
            "lef": str(directory / "fixture.lef"),
            "def": str(directory / "many.def"),
            "lite_mode": True,
            "random_place": False,
            "num_threads": 1,
        }
    )
    raw = io.create_database()
    raw.load()
    raw.setup()
    gp = io.create_gpdatabase(raw)
    gp.setup()
    xplace = backend._ensure_xplace_python_path()
    gr.read_flute(
        str(xplace / "thirdparty/flute/POWV9.dat"), str(xplace / "thirdparty/flute/POST9.dat")
    )
    pins = {name: index for index, name in enumerate(gp.pin_names())}
    net_ids = {name: index for index, name in enumerate(gp.net_names())}
    mapping = PinMapping(
        pins,
        net_ids,
        gp.pin_id2net_id_tensor().to(torch.int32),
        torch.tensor([pins[f"s{index}:Y"] for index in range(65)], dtype=torch.int32),
        torch.ones(len(net_ids), dtype=torch.bool),
    )

    def route(workers):
        assert gr.load_gr_params(
            {
                "backend": "cpu_pr_maze",
                "rrrIters": 3,
                "threads": workers,
                "route_xSize": 8,
                "route_ySize": 8,
                "bottom_routing_layer": "M1",
                "top_routing_layer": "M3",
            }
        )
        router = gr.create_routeforce(gr.create_grdatabase(raw, gp))
        router.run_ggr()
        return router

    return route, directory / "fixture.lef", mapping


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_multiple_batches_keep_routes_and_rc_connected(multi_batch_route, workers):
    route, lef, mapping = multi_batch_route
    oracle, actual = route(1), route(workers)
    reference, stats = oracle.run_stats(), actual.run_stats()
    assert stats["terminal_status"] == "completed"
    assert stats["maze_route_failure_count"] == 0
    assert stats["maze_max_batch_size"] <= stats["maze_batch_size_limit"]
    assert stats["peak_memory_bytes"] <= stats["memory_budget_bytes"]
    assert stats["rrr_iterations"][0]["selected_net_count"] > stats["maze_batch_size_limit"]
    assert actual.raw_routes() == oracle.raw_routes()
    keys = ("route_hash64", "wire_map_hash64", "via_map_hash64")
    assert tuple(stats[key] for key in keys) == tuple(reference[key] for key in keys)
    assert [tuple(row[key] for key in keys) for row in stats["rrr_iterations"]] == [
        tuple(row[key] for key in keys) for row in reference["rrr_iterations"]
    ]
    pack = actual.timing_route_pack()
    rc = RCParameters.from_lefs([lef], pack["layer_names"])
    op = GRParasiticsOp.prepare(
        pack,
        rc,
        mapping,
        RouteIdentity("multi-batch-input", "multi-batch-geometry", 1),
        dtype=torch.float64,
    )
    caps = torch.full((len(mapping.pin_names),), 0.01, dtype=torch.float64, requires_grad=True)
    delay = op(caps, caps, caps)[2]["generic"]
    gradient = torch.autograd.grad(delay.sum(), caps)[0]
    assert torch.isfinite(delay).all() and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
