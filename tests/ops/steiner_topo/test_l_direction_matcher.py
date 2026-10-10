"""Native route matching against the original Python geometric rules."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.steiner_topo.egr_l_direction import EGRLDirectionResolver
from l_direction_reference import PythonLDirectionReference


def wire(a, b, order=0):
    return dict(real1=a, real2=b, is_horizontal=a[1] == b[1], is_vertical=a[0] == b[0], order=order)


def resolver(routes, *, axes=True):
    names = list(routes)
    db = SimpleNamespace(
        num_nets=len(names),
        net_names=names,
        num_pins=2 * len(names),
        pin2net_map=np.repeat(np.arange(len(names)), 2),
        dbu=1,
    )
    result = EGRLDirectionResolver(db, SimpleNamespace(scale_factor=1.0, shift_factor=(0.0, 0.0)))
    result.egr_net_data = routes
    if axes:
        result._route_grid_x_um = np.arange(-5.0, 16.0)
        result._route_grid_y_um = np.arange(-5.0, 16.0)
        result._route_pitch_x_um = result._route_pitch_y_um = 1.0
    return result


def topology(pairs):
    points = np.asarray(pairs).reshape(-1, 2)
    return SimpleNamespace(
        flat_pin_from=torch.arange(0, len(points), 2, dtype=torch.int32),
        flat_pin_to=torch.arange(1, len(points), 2, dtype=torch.int32),
        newx=torch.tensor(points[:, 0]),
        newy=torch.tensor(points[:, 1]),
        net_steiner_start=torch.zeros(1, dtype=torch.int32),
    )


def python_reference(r, topo):
    """Original full-edge pass and simultaneous known-path fallback."""
    reference = PythonLDirectionReference(r.placedb, r.params)
    for field in (
        "egr_net_data",
        "_route_grid_x_um",
        "_route_grid_y_um",
        "_route_pitch_x_um",
        "_route_pitch_y_um",
    ):
        setattr(reference, field, getattr(r, field))
    r = reference
    x = (topo.newx.numpy() / r.params.scale_factor + r.params.shift_factor[0]) / r.placedb.dbu
    y = (topo.newy.numpy() / r.params.scale_factor + r.params.shift_factor[1]) / r.placedb.dbu
    nets = r._precompute_vertex_to_net(r.placedb.num_pins, topo.net_steiner_start.numpy(), len(x))
    directions = np.full(topo.flat_pin_from.numel(), r.UNKNOWN, dtype=np.int32)
    records = []
    for edge, (a, b) in enumerate(
        zip(topo.flat_pin_from.tolist(), topo.flat_pin_to.tolist(), strict=True)
    ):
        if a < 0 or b < 0 or nets[a] < 0 or nets[a] >= r.placedb.num_nets:
            continue
        name = r.placedb.net_names[nets[a]]
        data = r.egr_net_data.get(name)
        if data is None:
            continue
        p, q = (x[a], y[a]), (x[b], y[b])
        directions[edge] = r._determine_l_direction_for_edge(data, p, q)
        records.append(
            dict(
                edge_idx=edge,
                path_axis_indices=r._path_axis_indices(
                    r._snap_point_to_route_grid(data, p), r._snap_point_to_route_grid(data, q)
                ),
            )
        )
    if records and np.isin(directions, [r.UNKNOWN, r.FAKE_STRAIGHT]).any():
        r._fallback_unresolved_with_path_maps(records, directions)
    return torch.from_numpy(directions)


@pytest.mark.parametrize("source", ["gpugr", "egr"])
@pytest.mark.parametrize("axes", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matching_rules_and_fallback(source, axes, dtype):
    routes = {
        "horizontal": dict(
            source=source, wires=[wire((0.0, 0.0), (5.0, 0.0)), wire((5.0, 0.0), (5.0, 5.0))]
        ),
        "vertical": dict(
            source=source, wires=[wire((0.0, 0.0), (0.0, 5.0)), wire((0.0, 5.0), (5.0, 5.0))]
        ),
        "straight": dict(source=source, wires=[]),
        "coincident": dict(source=source, wires=[]),
        "single": dict(source=source, wires=[wire((0.0, 0.0), (5.0, 0.0))]),
        "unknown": dict(source=source, wires=[]),
        "reverse": dict(
            source=source, wires=[wire((5.0, 5.0), (3.0, 5.0)), wire((3.0, 5.0), (3.0, 8.0))]
        ),
        "shared": dict(
            source=source,
            wires=[wire((12.0, 12.0), (10.0, 12.0)), wire((10.0, 12.0), (10.0, 14.0))],
        ),
        "order": dict(
            source=source, wires=[wire((0.0, 0.0), (0.0, 5.0), 1), wire((0.0, 0.0), (5.0, 0.0), 0)]
        ),
        "snap_tie": dict(
            source=source, wires=[wire((1.0, 1.0), (5.0, 1.0)), wire((5.0, 1.0), (5.0, 5.0))]
        ),
    }
    r = resolver(routes, axes=axes)
    topo = topology(
        [
            ((0.0, 0.0), (5.0, 5.0)),
            ((0.0, 0.0), (5.0, 5.0)),
            ((0.0, 0.0), (0.0, 5.0)),
            ((2.0, 2.0), (2.0, 2.0)),
            ((0.0, 0.0), (5.0, 5.0)),
            ((0.0, 0.0), (5.0, 5.0)),
            ((-3.0, -3.0), (5.0, 5.0)),
            ((0.0, 0.0), (5.0, 5.0)),
            ((0.0, 0.0), (5.0, 5.0)),
            ((0.5, 0.5), (5.0, 5.0)),
        ]
    )
    topo.newx, topo.newy = topo.newx.to(dtype), topo.newy.to(dtype)
    expected = python_reference(r, topo)
    actual = r.resolve_l_directions(topo)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[:4].tolist() == [r.H_FIRST, r.V_FIRST, r.STRAIGHT, r.UNKNOWN]
    # Live coordinates are consumed without rebuilding the route-wire matcher.
    prepared = r._native_matcher
    topo.newx += 0.2
    torch.testing.assert_close(
        r.resolve_l_directions(topo), python_reference(r, topo), rtol=0, atol=0
    )
    assert r._native_matcher is prepared


def test_randomized_wire_ranking_and_partial_domains():
    rng = np.random.default_rng(3000)
    routes, pairs = {}, []
    for n in range(1200):
        points = rng.uniform(-4, 14, (5, 2)).round(6)
        wires = [
            wire(tuple(a), (b[0], a[1]) if i % 2 else (a[0], b[1]), i % 3)
            for i, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True))
        ]
        routes[f"n{n}"] = dict(source="gpugr" if n % 2 else "egr", wires=wires)
        pairs.append(tuple(rng.uniform(-4, 14, (2, 2))))
    r, topo = resolver(routes), topology(pairs)
    r.egr_net_data.pop("n10")
    topo.flat_pin_from[11] = -1
    topo.newx[24:26], topo.newy[24:26] = 1e-5, 0.0
    torch.testing.assert_close(
        r.resolve_l_directions(topo), python_reference(r, topo), rtol=0, atol=0
    )


def test_feedback_replacement_invalidates_native_routes(tmp_path):
    r = resolver({"net": {}})
    topo = topology([((0.0, 0.0), (5.0, 5.0))])
    for first, corner, expected in [("H", (5, 0), r.H_FIRST), ("V", (0, 5), r.V_FIRST)]:
        points = [(0, 0), corner, (5, 5)]
        entries = [
            dict(
                type="wire",
                grid_x1=a[0],
                grid_y1=a[1],
                grid_x2=b[0],
                grid_y2=b[1],
                dbu_center_x1=a[0],
                dbu_center_y1=a[1],
                dbu_center_x2=b[0],
                dbu_center_y2=b[1],
                orientation=first if i == 0 else ("V" if first == "H" else "H"),
            )
            for i, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True))
        ]
        r.parse_gpugr_route_entries([dict(net_name="net", entries=entries)])
        assert r._native_matcher is None
        assert r.resolve_l_directions(topo).tolist() == [expected]
    guide = tmp_path / "route.guide"
    guide.write_text("guide net\nwire 0 0 5 0 0 0 5 0 MET2\nwire 5 0 5 5 5 0 5 5 MET3\n")
    r.parse_egr_guide(guide)
    assert r._native_matcher is None
    assert r.resolve_l_directions(topo).tolist() == [r.H_FIRST]


def test_native_geometry_domain_checks():
    r = resolver({"net": dict(wires=[])})
    topo = topology([((0.0, 0.0), (5.0, 5.0))])
    topo.flat_pin_to[0] = 2
    with pytest.raises(ValueError, match="outside geometry"):
        r.resolve_l_directions(topo)
    topo.flat_pin_to[0] = 1
    topo.newx[0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        r.resolve_l_directions(topo)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_straight_epsilon_preserves_geometry_dtype(dtype):
    r = resolver({"net": dict(wires=[])}, axes=False)
    topo = topology([((0.0, 0.0), (1e-5, 5.0))])
    topo.newx, topo.newy = topo.newx.to(dtype), topo.newy.to(dtype)
    torch.testing.assert_close(
        r.resolve_l_directions(topo), python_reference(r, topo), rtol=0, atol=0
    )
