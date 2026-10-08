"""Real native routing fixtures using the installed production extensions."""

import pytest
import torch
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR
from route_fixture import make_route_fixture


def route_fixture(directory, *, small_pins=False, pin_layer="M2"):
    backend = XplaceGPUGR(None, None)
    _, gr, *_ = backend._import_xplace_modules()
    xplace = backend._ensure_xplace_python_path()
    from cpp_to_py import io_parser as io
    lef, design = make_route_fixture(xplace, small_pins=small_pins, pin_layer=pin_layer)
    if small_pins:
        # Separate physical cells, two loads sharing one coarse gcell.
        design = design.replace("( 8000 5000 )", "( 8000 3000 )")
    lef_path, def_path = directory / "toy.lef", directory / "toy.def"
    lef_path.write_text(lef)
    def_path.write_text(design)
    torch.set_num_threads(2)
    assert io.load_params(
        {
            "lef": str(lef_path),
            "def": str(def_path),
            "lite_mode": True,
            "random_place": False,
            "num_threads": 2,
        }
    )
    rawdb = io.create_database()
    rawdb.load()
    rawdb.setup()
    gpdb = io.create_gpdatabase(rawdb)
    gpdb.setup()
    gr.read_flute(
        str(xplace / "thirdparty/flute/POWV9.dat"), str(xplace / "thirdparty/flute/POST9.dat")
    )
    assert gr.load_gr_params(
        {
            "backend": "cpu_pr_mt",
            "rrrIters": 0,
            "threads": 2,
            "route_xSize": 2 if small_pins else 8,
            "route_ySize": 2 if small_pins else 8,
            "bottom_routing_layer": "M2",
            "top_routing_layer": "M3",
        }
    )
    grdb = gr.create_grdatabase(rawdb, gpdb)
    router = gr.create_routeforce(grdb)
    router.run_ggr()
    return router.timing_route_pack(), lef_path


@pytest.fixture(scope="module")
def native_route(tmp_path_factory):
    return route_fixture(tmp_path_factory.mktemp("gr_native_tree"))


@pytest.fixture(scope="module")
def native_small_route(tmp_path_factory):
    return route_fixture(
        tmp_path_factory.mktemp("gr_native_small_pins"), small_pins=True, pin_layer="M1"
    )


@pytest.fixture
def snapshot_inputs(native_route):
    from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
    from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity

    pack, lef = native_route
    # Permute PyDB pin IDs to prove that routing vertices and physical pin
    # indices are independent. The actual IO driver is PyDB pin 2.
    mapping = PinMapping(
        {"IN": 2, "U1:A": 0, "U3:A": 1},
        {"N1": 0},
        torch.tensor([0, 0, 0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([True]),
    )
    rc = RCParameters.from_lefs([lef], pack["layer_names"])
    return pack, rc, mapping, RouteIdentity("fixture-input", "fixture-geometry", 1)
