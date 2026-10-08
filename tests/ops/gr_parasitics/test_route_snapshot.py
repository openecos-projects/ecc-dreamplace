"""Native GR with local pin RC and explicitly recorded electrical tree reduction."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from dreamplace.ops.gr_parasitics.route_snapshot import prepare_snapshot
from rc_oracle import scalar_oracle


def test_real_branch_wire_via_rc_units_and_driver_root(snapshot_inputs):
    route, _, _, _ = snapshot_inputs
    tree = prepare_snapshot(*snapshot_inputs, dtype=torch.float64)
    assert (len(tree.parent), len(tree.edge_child), len(tree.root_vertex)) == (14, 13, 1)
    assert tree.pin_to_vertex.dtype == torch.int64
    assert tree.parent.dtype == torch.int32
    assert tree.root_vertex.tolist() == [tree.pin_to_vertex[2].item()]
    parent_layer, child_layer = (
        tree.vertex_layer[tree.edge_parent],
        tree.vertex_layer[tree.edge_child],
    )
    wires = parent_layer == child_layer
    lengths = (
        (tree.vertex_xy_um[tree.edge_parent] - tree.vertex_xy_um[tree.edge_child]).abs().sum(1)
    )
    # Independent values from the synthetic LEF, not from the parsed RC arrays.
    resistance = torch.where(parent_layer == 1, 2.0, 3.0) * lengths
    capacitance = torch.zeros_like(lengths)
    capacitance[parent_layer == 1] = 0.004 * lengths[parent_layer == 1]
    capacitance[parent_layer == 2] = 0.007 * lengths[parent_layer == 2]
    resistance[~wires] = 2.5
    capacitance[~wires] = 0
    torch.testing.assert_close(tree.edge_resistance, resistance, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(tree.edge_capacitance, capacitance, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(tree.wire_cap.sum(), capacitance.sum(), rtol=1e-9, atol=1e-12)
    assert tree.rc_parameters.via_cap_model == "omitted"
    assert np.count_nonzero(route["edge_kind"] == 1) == 2
    order = {v: i for i, v in enumerate(tree.topo_order.tolist())}
    for parent, child in zip(tree.edge_parent.tolist(), tree.edge_child.tolist(), strict=True):
        assert order[parent] < order[child]
        assert (
            child
            in tree.child_vertex[tree.child_start[parent] : tree.child_start[parent + 1]].tolist()
        )
    assert tree.incoming_resistance[tree.root_vertex].tolist() == [0]
    assert tree.parent[tree.root_vertex].tolist() == [-1]
    # The IO center is at the 1.875 um gcell center. Cell centers differ
    # from their grid points by 0.75 and 0.5 um Manhattan lengths.
    assert tree.model_report["attachment_length_um"] == 1.25
    assert torch.count_nonzero(tree.edge_resistance == 0) == 1


def test_prepare_rejects_missing_attachment_and_missing_via_rc(snapshot_inputs):
    route, rc, mapping, identity = snapshot_inputs
    incomplete = dict(route)
    incomplete["pin_vertex"] = route["pin_vertex"].copy()
    incomplete["pin_vertex"][1] = -1
    with pytest.raises(RuntimeError, match="missing pin attachment"):
        prepare_snapshot(incomplete, rc, mapping, identity)
    invalid = replace(rc, via_resistance=np.array([2.5, np.nan]))
    with pytest.raises(RuntimeError, match="missing/ambiguous via resistance"):
        prepare_snapshot(route, invalid, mapping, identity)


def test_physical_driver_without_timing_arcs_and_mismatched_timed_driver(snapshot_inputs):
    route, rc, mapping, identity = snapshot_inputs
    physical = prepare_snapshot(
        route, rc, replace(mapping, driver_pin=torch.tensor([-1])), identity
    )
    assert physical.root_vertex.tolist() == [physical.pin_to_vertex[2].item()]
    with pytest.raises(RuntimeError, match="native/PyDB driver mismatch"):
        prepare_snapshot(route, rc, replace(mapping, driver_pin=torch.tensor([0])), identity)


def test_connected_cycle_reduction_retains_all_wire_cap_and_is_deterministic(snapshot_inputs):
    route, rc, mapping, identity = snapshot_inputs
    loop = dict(route)
    # Add a distinct path through the other metal layer alongside one wire.
    # All three new edges have nonzero R; this is not an exact duplicate.
    occupied = set(
        zip(route["vertex_layer"], route["vertex_x_dbu"], route["vertex_y_dbu"], strict=True)
    )
    a, b = next(
        (int(a), int(b))
        for a, b, k in zip(route["edge_from"], route["edge_to"], route["edge_kind"], strict=True)
        if k == 0
        and route["vertex_layer"][a] == 1
        and (2, route["vertex_x_dbu"][a], route["vertex_y_dbu"][a]) not in occupied
        and (2, route["vertex_x_dbu"][b], route["vertex_y_dbu"][b]) not in occupied
    )
    first = len(route["vertex_net"])
    for field, values in {
        "vertex_net": [0, 0],
        "vertex_layer": [2, 2],
        "vertex_x_dbu": [route["vertex_x_dbu"][a], route["vertex_x_dbu"][b]],
        "vertex_y_dbu": [route["vertex_y_dbu"][a], route["vertex_y_dbu"][b]],
        "edge_from": [a, first, first + 1],
        "edge_to": [first, first + 1, b],
        "edge_kind": [1, 0, 1],
    }.items():
        loop[field] = np.append(route[field], values).astype(np.int32)
    loop["net_vertex_start"] = np.array([0, first + 2], dtype=np.int32)
    loop["net_edge_start"] = np.array([0, len(loop["edge_from"])], dtype=np.int32)
    tree = prepare_snapshot(loop, rc, mapping, identity, dtype=torch.float64)
    replay = prepare_snapshot(loop, rc, mapping, identity, dtype=torch.float64)
    torch.testing.assert_close(tree.parent, replay.parent, rtol=0, atol=0)
    assert tree.model_report == replay.model_report
    assert tree.model_report["loop_count"] == 1
    dropped = tree.model_report["loop_edges"][0]
    e = dropped["raw_edge_id"]
    u, v = int(loop["edge_from"][e]), int(loop["edge_to"][e])
    assert dropped["net_id"] == 0
    assert [dropped["from_layer"], dropped["to_layer"]] == [
        int(loop["vertex_layer"][u]),
        int(loop["vertex_layer"][v]),
    ]
    assert dropped["resistance_ohm"] > 0
    original = prepare_snapshot(route, rc, mapping, identity, dtype=torch.float64)
    extra_length = (
        abs(int(route["vertex_x_dbu"][a]) - int(route["vertex_x_dbu"][b]))
        + abs(int(route["vertex_y_dbu"][a]) - int(route["vertex_y_dbu"][b]))
    ) / route["dbu_per_micron"]
    # The new M3 chord has 0.007 pF/um, including when its R is omitted.
    torch.testing.assert_close(
        tree.wire_cap.sum(),
        original.wire_cap.sum() + extra_length * 0.007,
        rtol=1e-9,
        atol=1e-12,
    )
    from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp

    caps = torch.tensor([0.01, 0.02, 0.0], dtype=torch.float64)
    values = GRParasiticsOp(tree)(caps, caps, caps)
    for group, golden in zip(values, scalar_oracle(tree, caps), strict=True):
        torch.testing.assert_close(group["generic"], golden, rtol=1e-9, atol=1e-12)


def test_native_small_pin_attachment_rc_and_gradient(native_small_route):
    from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
    from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
    from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity

    route, lef = native_small_route
    mapping = PinMapping(
        {name: index for index, name in enumerate(route["pin_names"])},
        {"N1": 0},
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([True]),
    )
    assert route["pin_names"][0] == "U0:Y" and route["pin_is_driver"].tolist() == [1, 0, 0]
    rc = RCParameters.from_lefs([lef], route["layer_names"])
    op = GRParasiticsOp.prepare(
        route, rc, mapping, RouteIdentity("small", "shape", 1), dtype=torch.float64
    )
    tree = op.snapshot
    assert route["pin_local_path_missing"].tolist() == [1, 1, 1]
    assert route["pin_physical_layer"].tolist() == [0, 0, 0]
    assert route["pin_access_layer"].tolist() == [1, 1, 1]
    assert route["pin_vertex"][1] == route["pin_vertex"][2]
    assert tree.pin_to_vertex[1] != tree.pin_to_vertex[2]
    assert tree.pin_grid_vertex[1] == tree.pin_grid_vertex[2]
    expected_wire = 0.0
    for p, grid in enumerate(route["pin_vertex"]):
        length = (
            abs(int(route["pin_x_dbu"][p]) - int(route["vertex_x_dbu"][grid]))
            + abs(int(route["pin_y_dbu"][p]) - int(route["vertex_y_dbu"][grid]))
        ) / route["dbu_per_micron"]
        assert length > 0
        expected_wire += length * 0.004
        pin_id, grid_id = int(tree.pin_to_vertex[p]), int(tree.pin_grid_vertex[p])
        assert pin_id != grid_id
        child = grid_id if p == 0 else pin_id
        assert int(tree.parent[child]) == (pin_id if p == 0 else grid_id)
        # M2 r=2 ohm/um; the missing M1->M2 access via adds 2.5 ohm.
        assert float(tree.incoming_resistance[child]) == pytest.approx(length * 2 + 2.5)
        assert float(tree.wire_cap[pin_id]) == pytest.approx(length * 0.004 / 2)
    assert tree.model_report["attachment_via_count"] == 3
    assert tree.model_report["attachment_capacitance_pf"] == pytest.approx(expected_wire)
    caps = torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64, requires_grad=True)
    values = op(caps, caps, caps)
    for group, golden in zip(values, scalar_oracle(tree, caps), strict=True):
        torch.testing.assert_close(group["generic"], golden, rtol=1e-9, atol=1e-12)
    grad = torch.autograd.grad(values[2]["generic"][1], caps)[0]
    for p in (1, 2):
        step = 1e-4 * max(float(caps[p].detach()), 1e-3)
        plus, minus = caps.detach().clone(), caps.detach().clone()
        plus[p] += step
        minus[p] -= step
        fd = (op(plus, plus, plus)[2]["generic"][1] - op(minus, minus, minus)[2]["generic"][1]) / (
            2 * step
        )
        assert float(grad[p]) == pytest.approx(float(fd), rel=1e-3, abs=1e-6)
