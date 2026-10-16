"""S-window physical/working geometry and combined area ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.discrete_gradient_topk.runtime_cells import sync_runtime_cells
from dreamplace.ops.routability.cooptimization_area import (
    AreaState,
    SizingWindowGeometry,
    capture_area,
    coordinate_area,
    publish_area,
)


def make_geometry(*, filler=4.0):
    width = torch.tensor([2.0, 2.0, filler], dtype=torch.float64)
    height = torch.ones(3, dtype=torch.float64)
    data = SimpleNamespace(
        node_size_x=width,
        node_size_y=height,
        node_areas=width * height,
        sorted_node_map=torch.argsort(width[:2]).to(torch.int32),
        original_node_size_x=torch.tensor([1.0, 2.0, filler], dtype=torch.float64),
        original_node_size_y=height.clone(),
        target_density=torch.tensor([0.8]),
        pin2node_map=torch.tensor([0, 1]),
        pin_2_libpin_offset=torch.tensor([0, 0]),
        pin_offset_x=torch.tensor([0.75, 0.25], dtype=torch.float64),
        pin_offset_y=torch.tensor([0.5, 0.5], dtype=torch.float64),
        original_pin_offset_x=torch.tensor([0.25, 0.25], dtype=torch.float64),
        original_pin_offset_y=torch.tensor([0.5, 0.5], dtype=torch.float64),
        inst_cell_id=torch.tensor([0, 0]),
        flat_libcell_width=torch.tensor([1.0, 1.5, 3.0, 10.0], dtype=torch.float64),
        flat_libcell_height=torch.ones(4, dtype=torch.float64),
        cell_id_2_libpin_id_start=torch.tensor([0, 1, 2, 3]),
        flat_lib_pin_offset_x=torch.tensor([0.25, 0.4, 0.6, 0.8], dtype=torch.float64),
        flat_lib_pin_offset_y=torch.full((4,), 0.5, dtype=torch.float64),
    )
    db = SimpleNamespace(
        num_movable_nodes=2,
        num_filler_nodes=1,
        area=12.0,
        total_fixed_node_area=2.0,
        total_space_area=10.0,
        total_movable_node_area=4.0,
        total_filler_node_area=filler,
        node_size_x=width.numpy().copy(),
        node_size_y=height.numpy().copy(),
        pin_offset_x=data.pin_offset_x.numpy().copy(),
        pin_offset_y=data.pin_offset_y.numpy().copy(),
    )
    pos = torch.tensor([1.0, 5.0, 7.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    return data, db, pos


def test_runtime_sizing_preserves_centers_and_entry_footprint_across_rounds():
    data, db, pos = make_geometry()
    window = SizingWindowGeometry(data, db, pos)
    before_area = capture_area(data, db)
    model = SimpleNamespace(_timing_geometry_cache=None, _virtual_cell_density_view=None)

    def resize(cell_id):
        return sync_runtime_cells(
            SimpleNamespace(cell_padding_x=0),
            {"applied_instance_ids": [0], "applied_cell_ids": [cell_id]},
            data_collections=data,
            placedb=db,
            timing_op=None,
            model=model,
            direct_joint=False,
            geometry_window=window,
        )

    resized = resize(1)
    assert resized["movable_area_after_internal"] is None
    assert db.total_movable_node_area == 4.0  # Deferred until the final window state.
    torch.testing.assert_close(data.node_size_x, torch.tensor([2.0, 2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(
        data.original_node_size_x, torch.tensor([1.5, 2.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        data.original_pin_offset_x, torch.tensor([0.4, 0.25], dtype=torch.float64)
    )
    torch.testing.assert_close(
        pos[0] + data.pin_offset_x[0], torch.tensor(1.65, dtype=torch.float64)
    )
    resize(3)  # A temporary large master must not ratchet the preserved footprint.
    resize(2)
    torch.testing.assert_close(data.node_size_x[:2], torch.tensor([3.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(pos[:2] + data.node_size_x[:2] * 0.5, window.center_x)
    torch.testing.assert_close(pos[3:5] + data.node_size_y[:2] * 0.5, window.center_y)
    assert coordinate_area(before_area, data, db, pos) == AreaState(5.0, 3.0, 10.0)


@pytest.mark.parametrize(
    "filler,added_area,expected",
    [
        (4.0, 1.0, AreaState(5.0, 3.0, 10.0)),
        (1.0, 3.0, AreaState(7.0, 0.0, 10.0)),
        (0.0, -1.0, AreaState(3.0, 0.0, 10.0)),
        (4.0, -1.0, AreaState(3.0, 5.0, 10.0)),
    ],
)
def test_area_coordination_scales_only_existing_filler_once(filler, added_area, expected):
    data, db, pos = make_geometry(filler=filler)
    before = capture_area(data, db)
    target_density = float(data.target_density)
    filler_center = pos[2] + data.node_size_x[2] * 0.5
    data.node_size_x[0] += added_area
    result = coordinate_area(before, data, db, pos)
    assert result == expected
    torch.testing.assert_close(pos[2] + data.node_size_x[2] * 0.5, filler_center)
    assert float(data.target_density.item()) == pytest.approx(max(target_density, expected.density))
    np.testing.assert_array_equal(db.node_size_x, data.node_size_x.numpy())


def test_cumulative_virtual_area_and_materialization_do_not_double_charge():
    data, db, pos = make_geometry()
    first = coordinate_area(capture_area(data, db), data, db, pos, virtual_area=1.0)
    assert first == AreaState(5.0, 3.0, 10.0)
    second = coordinate_area(first, data, db, pos, virtual_area=2.0)
    assert second == AreaState(6.0, 2.0, 10.0)
    # The native replacement changes the base view, not its combined footprint.
    data.node_size_x = torch.tensor([2.0, 2.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    data.node_size_y = torch.ones(5, dtype=torch.float64)
    db.num_movable_nodes = 4
    assert capture_area(data, db) == second


def test_downsize_with_no_filler_nodes_preserves_density_headroom_without_creating_nodes():
    data, db, pos = make_geometry()
    db.num_filler_nodes = 0
    before = capture_area(data, db)
    density_target = data.target_density.clone()
    data.node_size_x[0] = 1.0
    assert coordinate_area(before, data, db, pos) == AreaState(3.0, 0.0, 10.0)
    torch.testing.assert_close(data.target_density, density_target)
    assert db.num_filler_nodes == 0 and pos.numel() == 6


def test_virtual_buffers_charge_occupancy_without_tightening_density_target():
    data, db, pos = make_geometry(filler=0.)
    db.num_filler_nodes = 0
    density_target = data.target_density.clone()
    before = capture_area(data, db)
    data.node_size_x[0] += .1
    state = coordinate_area(before, data, db, pos, virtual_area=.2)
    assert state == AreaState(4.3, 0., 10.)
    assert db.total_movable_node_area == pytest.approx(4.1)
    torch.testing.assert_close(data.target_density, density_target)
    assert publish_area(data, db, virtual_area=.2, capacity=10.) == state
    torch.testing.assert_close(data.target_density, density_target)


def test_growth_exhausting_density_headroom_raises_target_and_refreshes_in_place():
    data, db, pos = make_geometry(filler=0.)
    db.num_filler_nodes = 0
    data.target_density.fill_(.5)
    shared_target = data.target_density
    state = coordinate_area(capture_area(data, db), data, db, pos, virtual_area=2.)
    assert state == AreaState(6., 0., 10.)
    assert data.target_density is shared_target
    assert float(data.target_density) == pytest.approx(.6)
    # An inflation controller may deliberately choose a new target; publishing
    # physical accounting must retain that choice when it has enough headroom.
    data.target_density.fill_(.9)
    assert publish_area(data, db, virtual_area=2., capacity=10.) == state
    assert float(data.target_density) == pytest.approx(.9)


def test_combined_capacity_failure_leaves_filler_and_density_untouched():
    data, db, pos = make_geometry()
    before = capture_area(data, db)
    sizes, position, density = data.node_size_x.clone(), pos.clone(), data.target_density.clone()
    with pytest.raises(ValueError, match="exceeds placement capacity"):
        coordinate_area(before, data, db, pos, virtual_area=7.0)
    torch.testing.assert_close(data.node_size_x, sizes)
    torch.testing.assert_close(pos, position)
    torch.testing.assert_close(data.target_density, density)
