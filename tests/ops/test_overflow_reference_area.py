"""Overflow uses its GP entry scale or the published combined movable area."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.EvalMetrics import EvalMetrics
from dreamplace.flows.handoff_state import build_continuation_seed
from dreamplace.ops.adjust_node_area.adjust_node_area import AdjustNodeArea
from dreamplace.ops.electric_potential.electric_overflow import ElectricOverflow
from dreamplace.ops.routability.cooptimization_area import (
    initialize_overflow_reference,
    publish_area,
)


@pytest.mark.parametrize("mode", ["initial", "ordinary"])
def test_sizing_inflation_uses_selected_overflow_area(mode):
    db = SimpleNamespace(
        total_movable_node_area=4., num_movable_nodes=2, num_filler_nodes=1,
        node_size_x=np.array([2., 2., 10.]), node_size_y=np.ones(3),
    )
    data = SimpleNamespace(
        node_size_x=torch.from_numpy(db.node_size_x),
        node_size_y=torch.from_numpy(db.node_size_y),
        node_areas=torch.tensor([2., 2., 10.], dtype=torch.float64),
        target_density=torch.tensor([.7], dtype=torch.float64),
        sorted_node_map=torch.tensor([0, 1], dtype=torch.int32),
    )
    initialize_overflow_reference(db, mode=mode)
    raw_overflow = torch.tensor([.5], dtype=torch.float64)
    ops = {name: lambda pos: (raw_overflow.clone(), torch.tensor([1.]))
           for name in ("overflow", "goverflow")}
    for growth, virtual_area, current_area in [(1., 0., 4.), (1.2, 0., 4.8), (1.5, 2., 9.2)]:
        data.node_size_x[:2].mul_(growth)
        publish_area(data, db, virtual_area, capacity=30.)
        metric = EvalMetrics()
        metric.evaluate(db, ops, torch.zeros(6), data)
        denominator = 4. if mode == "initial" else current_area
        torch.testing.assert_close(metric.overflow, raw_overflow / denominator)
        torch.testing.assert_close(metric.goverflow, metric.overflow)
        if mode == "initial":
            assert metric.overflow.item() > .1
    assert db.total_movable_node_area > db.overflow_reference_area


def test_ordinary_preserves_legacy_area_without_published_geometry():
    db = SimpleNamespace(total_movable_node_area=4.)
    data = SimpleNamespace()
    initialize_overflow_reference(db, mode="ordinary")
    raw_overflow = torch.tensor([.5], dtype=torch.float64)
    ops = {name: lambda pos: (raw_overflow.clone(), torch.tensor([1.]))
           for name in ("overflow", "goverflow")}
    for area in (4., 8.):
        db.total_movable_node_area = area
        metric = EvalMetrics()
        metric.evaluate(db, ops, torch.zeros(6), data)
        torch.testing.assert_close(metric.overflow, raw_overflow / area)
        torch.testing.assert_close(metric.goverflow, metric.overflow)


def test_native_inflation_uses_initial_area_with_updated_density_map():
    width = torch.ones(2, dtype=torch.float64)
    height = torch.ones(2, dtype=torch.float64)
    density = torch.tensor([.1], dtype=torch.float64)
    pos = torch.tensor([1., 4., 0., 0.], dtype=torch.float64)
    data = SimpleNamespace(
        node_size_x=width, node_size_y=height, node_areas=width * height,
        target_density=density, sorted_node_map=torch.tensor([0, 1], dtype=torch.int32),
    )
    db = SimpleNamespace(
        total_movable_node_area=2., num_movable_nodes=2, num_filler_nodes=0,
        node_size_x=width.numpy(), node_size_y=height.numpy(),
    )
    initialize_overflow_reference(db)
    overflow = ElectricOverflow(
        width, height, torch.tensor([2.5, 7.5], dtype=torch.float64),
        torch.tensor([.5, 1.5], dtype=torch.float64), density,
        0., 0., 10., 2., 5., 1., 2, 0, 0, 0, 1, data.sorted_node_map,
    )
    adjust = AdjustNodeArea(
        flat_node2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_node2pin_start_map=torch.tensor([0, 1, 2], dtype=torch.int32),
        pin_weights=None, xl=0., yl=0., xh=10., yh=2.,
        num_movable_nodes=2, num_filler_nodes=0,
        route_num_bins_x=2, route_num_bins_y=2, pin_num_bins_x=2, pin_num_bins_y=2,
        total_place_area=20., total_whitespace_area=18., max_route_opt_adjust_rate=2.,
        max_pin_opt_adjust_rate=2., unit_pin_capacity=1.,
    )
    pin_x = torch.full((2,), .5, dtype=torch.float64)
    pin_y = pin_x.clone()
    overflow(pos)  # Populate the pre-inflation density cache.
    assert adjust(pos, width, height, pin_x, pin_y, density,
                  torch.full((2, 2), 2., dtype=torch.float64), None) == (True, True, False)
    publish_area(data, db, 0., capacity=20.)
    overflow.reset()
    metric = EvalMetrics()
    metric.evaluate(db, {"overflow": overflow, "goverflow": overflow}, pos, data)
    raw_overflow, _ = overflow(pos)
    torch.testing.assert_close(metric.overflow, raw_overflow / 2.)
    torch.testing.assert_close(metric.goverflow, metric.overflow)
    assert db.total_movable_node_area > 2.
    assert density.item() > .1
    assert metric.overflow.item() > (raw_overflow / db.total_movable_node_area).item()


def test_topology_continuation_carries_reference_but_fresh_gp_recaptures_it():
    db = SimpleNamespace(
        total_movable_node_area=4., num_nodes=3, num_movable_nodes=2,
        num_physical_nodes=2, num_filler_nodes=1,
        node_names=["u0", "u1"],
    )
    initialize_overflow_reference(db)
    db.total_movable_node_area = 11.
    seed = build_continuation_seed(
        db, {}, np.arange(6.), 1,
        pos_to_numpy=np.asarray, decode_node_name=str,
    )
    restarted_db = SimpleNamespace(total_movable_node_area=11.)
    initialize_overflow_reference(restarted_db, seed)
    assert (seed["overflow_reference_area"], restarted_db.overflow_reference_area) == (4., 4.)
    initialize_overflow_reference(restarted_db)
    assert restarted_db.overflow_reference_area == 11.
