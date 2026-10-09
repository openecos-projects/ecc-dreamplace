"""Exercise CPU RRR through RouteForce, route export and the native RC tree."""

import importlib.util
from pathlib import Path

import pytest
import torch
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity


@pytest.fixture(scope="module")
def maze_fixture(tmp_path_factory):
    path = Path(__file__).resolve().parents[2] / "flows/gr_sizing_fixture.py"
    spec = importlib.util.spec_from_file_location("maze_gr_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    directory = fixture.make_fixture(tmp_path_factory.mktemp("cpu_maze_native") / "inputs")
    backend = XplaceGPUGR(None, None)
    _, gr, *_ = backend._import_xplace_modules(resolved_backend="cpu_pr_maze")
    from cpp_to_py import io_parser as io

    assert io.load_params(
        {
            "lef": str(directory / "fixture.lef"),
            "def": str(directory / "fixture.def"),
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
    pin_ids = {name: i for i, name in enumerate(gp.pin_names())}
    net_ids = {name: i for i, name in enumerate(gp.net_names())}
    eligible = torch.ones(len(net_ids), dtype=torch.bool)
    eligible[net_ids["clk"]] = False
    mapping = PinMapping(
        pin_ids,
        net_ids,
        gp.pin_id2net_id_tensor().to(torch.int32),
        torch.tensor(
            [pin_ids[name] for name in ("clk", "in", "launch:Q", "b1:Y", "b2:Y", "capture:Q")],
            dtype=torch.int32,
        ),
        eligible,
    )

    def route(backend, iters, workers):
        assert gr.load_gr_params(
            {
                "backend": backend,
                "rrrIters": iters,
                "threads": workers,
                "route_xSize": 8,
                "route_ySize": 8,
                "bottom_routing_layer": "M1",
                "top_routing_layer": "M3",
            }
        )
        db = gr.create_grdatabase(raw, gp)
        router = gr.create_routeforce(db)
        router.run_ggr()
        return router

    return route, directory / "fixture.lef", mapping


def hashes(stats):
    return tuple(stats[key] for key in ("route_hash64", "wire_map_hash64", "via_map_hash64"))


def test_zero_rrr_preserves_parallel_pattern_route(maze_fixture):
    route, _, _ = maze_fixture
    baseline = route("cpu_pr_mt", 0, 4)
    maze = route("cpu_pr_maze", 0, 4)
    assert maze.raw_routes() == baseline.raw_routes()
    assert hashes(maze.run_stats()) == hashes(baseline.run_stats())
    for actual, expected in zip(maze.dmd_map(), baseline.dmd_map(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_non_improving_pass_does_not_skip_remaining_cost_schedule(maze_fixture):
    route, _, _ = maze_fixture
    router = route("cpu_pr_maze", 3, 2)
    stats = router.run_stats()
    iterations = stats["rrr_iterations"]
    # This congested fixture rejects the first candidate state. The next
    # passes still need to run because their via and congestion costs change.
    assert not iterations[0]["best_state"]
    assert iterations[0]["overflow_net_count"] > 0
    assert [row["iteration"] for row in iterations] == [1, 2, 3]
    assert stats["completed_rrr_iters"] == stats["requested_rrr_iters"] == 3
    assert hashes(stats) == hashes(iterations[-1]) == hashes(stats["accepted_route_qor"])


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("iters", [1, 3])
def test_conflicting_batch_researches_and_preserves_rc_connectivity(maze_fixture, workers, iters):
    route, lef, mapping = maze_fixture
    oracle = route("cpu_pr_maze", iters, 1)
    actual = route("cpu_pr_maze", iters, workers)
    stats, reference = actual.run_stats(), oracle.run_stats()
    assert stats["terminal_status"] == "completed"
    assert stats["maze_route_failure_count"] == 0
    assert stats["maze_failure_reasons"] == {}
    assert stats["conflict_fallback_count"] > 0
    assert hashes(stats) == hashes(reference)
    assert actual.raw_routes() == oracle.raw_routes()
    assert [hashes(row) for row in stats["rrr_iterations"]] == [
        hashes(row) for row in reference["rrr_iterations"]
    ]
    for row in stats["rrr_iterations"]:
        assert row["rerouted_net_count"] == row["selected_net_count"] > 0
        assert row["failed_net_count"] == 0
    # The exported RC uses the final completed cost pass. An intermediate
    # overflow minimum is diagnostic, rather than the returned route state.
    assert hashes(stats) == hashes(stats["rrr_iterations"][-1])
    assert hashes(stats) == hashes(stats["accepted_route_qor"])
    pack = actual.timing_route_pack()
    rc = RCParameters.from_lefs([lef], pack["layer_names"])
    op = GRParasiticsOp.prepare(
        pack,
        rc,
        mapping,
        RouteIdentity("fixture-input", "fixture-geometry", 1),
        dtype=torch.float64,
    )
    caps = torch.full((len(mapping.pin_names),), 0.01, dtype=torch.float64, requires_grad=True)
    delay = op(caps, caps, caps)[2]["generic"]
    gradient = torch.autograd.grad(delay.sum(), caps)[0]
    assert torch.isfinite(delay).all() and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
