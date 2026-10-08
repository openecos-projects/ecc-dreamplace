from types import SimpleNamespace

import pytest
import torch

from dreamplace.ops.buffer_insertion.virtual_cell_density import (
    VirtualCellDensityOp,
    build_combined_virtual_position,
    build_equal_spaced_virtual_positions,
    build_segment_endpoint_mapping,
)


@pytest.mark.parametrize("padding", [0.0, 1.0])
def test_virtual_buffer_footprint_matches_padded_native_cell(padding):
    from dreamplace.PlaceObj import PlaceObj
    from dreamplace.ops.routability.cooptimization_area import capture_area

    model = SimpleNamespace(
        params=SimpleNamespace(cell_padding_x=padding),
        data_collections=SimpleNamespace(
            buffer_segment_count_payload={"buffer_legal_table": {"legal_cell_ids": [0]}},
            flat_libcell_width=torch.tensor([3.0]),
            flat_libcell_height=torch.tensor([2.0]),
        ),
        _segment_count_state_for_virtual_density=lambda: SimpleNamespace(fixed_bsu_index=0),
    )
    width, height = PlaceObj._segment_virtual_buffer_size(model)
    placedb = SimpleNamespace(num_movable_nodes=1, num_filler_nodes=0)
    before = SimpleNamespace(node_size_x=torch.tensor([4.0]), node_size_y=torch.tensor([2.0]))
    virtual = capture_area(before, placedb, width * height, capacity=100.0)
    placedb.num_movable_nodes = 2
    materialized = SimpleNamespace(
        node_size_x=torch.tensor([4.0, 3.0 + 2.0 * padding]),
        node_size_y=torch.tensor([2.0, 2.0]),
    )
    assert virtual == capture_area(materialized, placedb, capacity=100.0)


def test_equal_spaced_positions_support_zero_through_three_buffers():
    parent_x = torch.tensor([0.0, 10.0, 20.0])
    parent_y = torch.zeros(3)
    child_x = torch.tensor([8.0, 16.0, 28.0])
    child_y = torch.tensor([4.0, 8.0, 12.0])
    counts = torch.tensor([0.0, 1.0, 3.0], requires_grad=True)

    result = build_equal_spaced_virtual_positions(
        parent_x,
        parent_y,
        child_x,
        child_y,
        counts,
    )

    assert result["segment_indices"].tolist() == [1, 2, 2, 2]
    assert result["repeater_indices"].tolist() == [1, 1, 2, 3]
    torch.testing.assert_close(
        result["fractions"],
        torch.tensor([0.5, 0.25, 0.5, 0.75]),
    )
    torch.testing.assert_close(
        result["x"],
        torch.tensor([13.0, 22.0, 24.0, 26.0]),
    )
    torch.testing.assert_close(
        result["y"],
        torch.tensor([4.0, 3.0, 6.0, 9.0]),
    )
    assert counts.grad is None


def test_equal_spaced_positions_have_endpoint_jacobian():
    parent_x = torch.tensor([2.0], requires_grad=True)
    parent_y = torch.tensor([3.0], requires_grad=True)
    child_x = torch.tensor([14.0], requires_grad=True)
    child_y = torch.tensor([15.0], requires_grad=True)
    counts = torch.tensor([2.0])

    result = build_equal_spaced_virtual_positions(
        parent_x,
        parent_y,
        child_x,
        child_y,
        counts,
    )
    objective = result["x"][0] + 2.0 * result["x"][1]
    objective.backward()

    torch.testing.assert_close(parent_x.grad, torch.tensor([4.0 / 3.0]))
    torch.testing.assert_close(child_x.grad, torch.tensor([5.0 / 3.0]))
    assert parent_y.grad is None
    assert child_y.grad is None


def test_segment_endpoint_mapping_resolves_compact_to_live_topology_ids():
    prepared = {
        "edge_to_segment_id": torch.tensor([-1, 17, 23]),
        "edge_parent_compact_id": torch.tensor([0, 2, 4]),
        "edge_child_compact_id": torch.tensor([1, 3, 5]),
        "flat_topo_node_id": torch.tensor([31, 48, 9, 77, 20, 3]),
    }

    parent, child = build_segment_endpoint_mapping(
        prepared,
        torch.tensor([23, 17]),
    )

    assert parent.tolist() == [20, 9]
    assert child.tolist() == [3, 77]


def test_combined_position_preserves_dreamplace_node_order():
    pos = torch.tensor([1.0, 10.0, 20.0, 2.0, 11.0, 21.0])
    combined = build_combined_virtual_position(
        pos,
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        2.0,
        3.0,
        num_movable_nodes=1,
        num_terminals=1,
        num_filler_nodes=1,
    )

    torch.testing.assert_close(
        combined,
        torch.tensor([1.0, 5.0, 10.0, 20.0, 2.0, 6.0, 11.0, 21.0]),
    )


def test_combined_position_keeps_terminal_nis_in_fixed_range():
    pos = torch.tensor(
        [1.0, 10.0, 20.0, 30.0, 2.0, 11.0, 21.0, 31.0]
    )
    combined = build_combined_virtual_position(
        pos,
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        2.0,
        3.0,
        num_movable_nodes=1,
        num_terminals=0,
        num_fixed_nodes=2,
        num_filler_nodes=1,
    )

    torch.testing.assert_close(
        combined,
        torch.tensor(
            [1.0, 5.0, 10.0, 20.0, 30.0, 2.0, 6.0, 11.0, 21.0, 31.0]
        ),
    )


def test_virtual_density_has_endpoint_gradients_but_no_density_to_z_gradient():
    endpoint_x = torch.tensor([0.0, 10.0], requires_grad=True)
    endpoint_y = torch.tensor([0.0, 0.0], requires_grad=True)
    state = SimpleNamespace(
        segment_ids=torch.tensor([11, 12], dtype=torch.long),
        z_param=torch.nn.Parameter(torch.tensor([1.0, 0.0])),
    )
    prepared = {
        "edge_to_segment_id": torch.tensor([11, 12]),
        "edge_parent_compact_id": torch.tensor([0, 1]),
        "edge_child_compact_id": torch.tensor([1, 2]),
        "flat_topo_node_id": torch.tensor([0, 1, 2]),
    }
    calls = []

    class FakeDensity:
        def __call__(self, combined_pos):
            calls.append(combined_pos.detach().clone())
            return combined_pos.square().sum()

    def factory(**kwargs):
        return FakeDensity()

    op = VirtualCellDensityOp(
        segment_state=state,
        prepared_timing_inputs=prepared,
        num_movable_nodes=1,
        num_terminals=1,
        num_filler_nodes=1,
        node_size_x=torch.tensor([2.0, 3.0, 4.0]),
        node_size_y=torch.tensor([2.0, 3.0, 4.0]),
        buffer_size_x=2.0,
        buffer_size_y=2.0,
        movable_macro_mask=torch.tensor([False]),
        density_op_factory=factory,
    )
    pos = torch.tensor([1.0, 10.0, 20.0, 2.0, 11.0, 21.0], requires_grad=True)

    loss = op(pos, endpoint_x, endpoint_y)
    loss.backward()

    assert len(calls) == 1
    assert op.last_metadata["active_virtual_cell_count"] == 1
    assert op.last_metadata["density_to_z_gradient"] is False
    assert state.z_param.grad is None
    assert endpoint_x.grad is not None
    assert endpoint_x.grad.abs().sum() > 0
    assert endpoint_y.grad is not None
    assert pos.grad is not None


def test_virtual_density_sizes_follow_total_virtual_cell_count():
    endpoint_x = torch.tensor([0.0, 10.0], requires_grad=True)
    endpoint_y = torch.tensor([0.0, 0.0], requires_grad=True)
    state = SimpleNamespace(
        segment_ids=torch.tensor([11], dtype=torch.long),
        z_param=torch.nn.Parameter(torch.tensor([2.0])),
    )
    prepared = {
        "edge_to_segment_id": torch.tensor([11]),
        "edge_parent_compact_id": torch.tensor([0]),
        "edge_child_compact_id": torch.tensor([1]),
        "flat_topo_node_id": torch.tensor([0, 1]),
    }
    calls = []
    factory_kwargs = []

    class FakeDensity:
        def __call__(self, combined_pos):
            calls.append(combined_pos)
            return combined_pos.square().sum()

    def factory(**kwargs):
        factory_kwargs.append(kwargs)
        return FakeDensity()

    op = VirtualCellDensityOp(
        segment_state=state,
        prepared_timing_inputs=prepared,
        num_movable_nodes=1,
        num_terminals=0,
        num_fixed_nodes=1,
        num_filler_nodes=0,
        node_size_x=torch.tensor([2.0, 3.0]),
        node_size_y=torch.tensor([2.0, 3.0]),
        buffer_size_x=2.0,
        buffer_size_y=2.0,
        movable_macro_mask=torch.tensor([False]),
        density_op_factory=factory,
    )
    pos = torch.tensor([1.0, 10.0, 2.0, 11.0], requires_grad=True)

    loss = op(pos, endpoint_x, endpoint_y)
    loss.backward()

    assert len(calls) == 1
    assert calls[0].numel() == 8
    assert factory_kwargs[0]["node_size_x"].numel() == 4
    assert op.last_metadata["active_virtual_cell_count"] == 2
    assert state.z_param.grad is None
    # Later GP coordinates need fresh positions but the same prepared layout.
    shifted = op(pos, endpoint_x + 3.0, endpoint_y)
    torch.testing.assert_close(calls[1][1:3], calls[0][1:3] + 3.0)
    assert len(factory_kwargs) == 1
    assert shifted != loss
    # A later integer B update replaces the layout and its cell-size operator.
    with torch.no_grad():
        state.z_param.fill_(3.0)
    op(pos, endpoint_x, endpoint_y)
    assert op.last_metadata["active_virtual_cell_count"] == 3
    assert factory_kwargs[-1]["node_size_x"].numel() == 5


def test_virtual_density_rejects_fractional_state():
    state = SimpleNamespace(
        segment_ids=torch.tensor([1], dtype=torch.long),
        z_param=torch.nn.Parameter(torch.tensor([0.5])),
    )
    with pytest.raises(ValueError, match="integer-valued"):
        VirtualCellDensityOp(
            segment_state=state,
            prepared_timing_inputs={
                "edge_to_segment_id": torch.tensor([1]),
                "edge_parent_compact_id": torch.tensor([0]),
                "edge_child_compact_id": torch.tensor([1]),
                "flat_topo_node_id": torch.tensor([0, 1]),
            },
            num_movable_nodes=0,
            num_terminals=1,
            num_filler_nodes=0,
            node_size_x=torch.tensor([1.0]),
            node_size_y=torch.tensor([1.0]),
            buffer_size_x=1.0,
            buffer_size_y=1.0,
            movable_macro_mask=torch.empty(0, dtype=torch.bool),
            density_op_factory=lambda **kwargs: None,
        )._active_rows()
