"""Ordinary inflation updates real geometry while retaining density headroom."""

import pytest
import torch

from dreamplace.ops.adjust_node_area.adjust_node_area import AdjustNodeArea


@pytest.mark.parametrize('entry_density', [.8, .1])
def test_inflation_density_budget_and_live_pin_geometry(entry_density):
    op = AdjustNodeArea(
        flat_node2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_node2pin_start_map=torch.tensor([0, 1, 2], dtype=torch.int32),
        pin_weights=None, xl=0., yl=0., xh=10., yh=2.,
        num_movable_nodes=2, num_filler_nodes=0,
        route_num_bins_x=2, route_num_bins_y=2, pin_num_bins_x=2, pin_num_bins_y=2,
        total_place_area=20., total_whitespace_area=18.,
        max_route_opt_adjust_rate=2., max_pin_opt_adjust_rate=2.,
        area_adjust_stop_ratio=.01, route_area_adjust_stop_ratio=.01,
        pin_area_adjust_stop_ratio=.05, unit_pin_capacity=1.,
    )
    pos = torch.tensor([1., 4., 0., 0.], dtype=torch.float64)
    width, height = torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    pin_x, pin_y = torch.full((2,), .5, dtype=torch.float64), torch.full((2,), .5, dtype=torch.float64)
    pin_positions = pos + torch.cat((pin_x, pin_y))
    density = torch.tensor([entry_density], dtype=torch.float64)
    shared_density = density
    route_map = torch.full((2, 2), 2., dtype=torch.float64)
    flags = op(pos, width, height, pin_x, pin_y, density, route_map, None)
    assert flags == (True, True, False)
    assert (width * height).sum() > 2.
    torch.testing.assert_close(pos + torch.cat((pin_x, pin_y)), pin_positions)
    assert density is shared_density
    occupancy = float((width * height).sum()) / 20.
    if entry_density == .8:
        assert occupancy < entry_density
        torch.testing.assert_close(density, torch.tensor([entry_density], dtype=torch.float64))
    else:
        assert occupancy > entry_density
        torch.testing.assert_close(density, torch.tensor([occupancy], dtype=torch.float64))
