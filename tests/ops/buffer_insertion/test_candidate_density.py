from types import SimpleNamespace

import torch
from dreamplace.ops.buffer_insertion.candidate_density import (
    CandidateDensityView,
    install_candidate_density,
)


def test_typed_candidate_density_uses_live_fraction_and_master_footprint():
    activation = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
    state = SimpleNamespace(
        activation_param=activation,
        drive_type_cell_ids=torch.arange(4),
        selected_drive_type=lambda: activation.detach().argmax(dim=1),
        candidate_records=(
            {"parent_node_id": 0, "child_node_ids": [1], "segment_split_ratio": 0.25},
            {"parent_node_id": 0, "child_node_ids": [1], "segment_split_ratio": 0.75},
        ),
    )
    x = torch.tensor([0.0, 100.0], requires_grad=True)
    y = torch.tensor([0.0, 40.0], requires_grad=True)
    captured = {}

    def density_factory(*args, **kwargs):
        captured.update(kwargs)

        def op(pos, **options):
            captured["pos"] = pos.detach().clone()
            return pos.sum()

        return op

    model = SimpleNamespace(
        params=SimpleNamespace(cell_padding_x=0),
        num_bins_x=2,
        num_bins_y=2,
        placedb=SimpleNamespace(
            num_movable_nodes=1, num_terminals=0, num_physical_nodes=1, num_filler_nodes=0
        ),
        data_collections=SimpleNamespace(
            flat_libcell_width=torch.tensor([2.0, 4.0, 6.0, 8.0]),
            flat_libcell_height=torch.tensor([1.0, 1.0, 2.0, 2.0]),
            node_size_x=torch.tensor([2.0]),
            node_size_y=torch.tensor([1.0]),
            movable_macro_mask=torch.tensor([False]),
        ),
        _timing_geometry_for_pos=lambda pos: (None, x, y),
        build_electric_potential=density_factory,
    )
    view = CandidateDensityView(model, state)
    loss = view(torch.tensor([10.0, 20.0], requires_grad=True))
    x_grad, y_grad, activation_grad = torch.autograd.grad(
        loss,
        (x, y, activation),
        allow_unused=True,
    )
    assert torch.equal(captured["pos"], torch.tensor([10.0, 23.0, 71.0, 20.0, 9.5, 29.0]))
    assert torch.equal(captured["node_size_x"], torch.tensor([2.0, 4.0, 8.0]))
    assert torch.equal(captured["node_size_y"], torch.tensor([1.0, 1.0, 2.0]))
    assert (x_grad.tolist(), y_grad.tolist(), activation_grad) == ([1.0, 1.0], [1.0, 1.0], None)


def test_candidate_density_reuses_operator_across_live_geometry_refreshes():
    positions = torch.tensor([0.0, 100.0, 200.0])
    activation = torch.tensor([[1.0, 0.0]])
    state = SimpleNamespace(
        activation_param=activation,
        drive_type_cell_ids=torch.arange(2),
        selected_drive_type=lambda: activation.argmax(dim=1),
        candidate_records=(
            {"parent_node_id": 0, "child_node_ids": [1], "segment_split_ratio": 0.5},
        ),
    )
    builds = []

    def factory(*args, **kwargs):
        builds.append(kwargs)
        return lambda position, **options: position.sum()

    model = SimpleNamespace(
        params=SimpleNamespace(cell_padding_x=0),
        num_bins_x=2,
        num_bins_y=2,
        placedb=SimpleNamespace(
            num_movable_nodes=1, num_terminals=0, num_physical_nodes=1, num_filler_nodes=0
        ),
        data_collections=SimpleNamespace(
            flat_libcell_width=torch.tensor([2.0, 4.0]),
            flat_libcell_height=torch.ones(2),
            node_size_x=torch.tensor([2.0]),
            node_size_y=torch.ones(1),
            movable_macro_mask=torch.tensor([False]),
        ),
        _timing_geometry_for_pos=lambda position: (None, positions, positions),
        _candidate_density_view=None,
        build_electric_potential=factory,
    )
    position = torch.tensor([10.0, 20.0])
    install_candidate_density(model, state)
    first = model._candidate_density_view(position)
    state.candidate_records = (
        {"parent_node_id": 1, "child_node_ids": [2], "segment_split_ratio": 0.5},
    )
    install_candidate_density(model, state)
    second = model._candidate_density_view(position)
    assert len(builds) == 1
    torch.testing.assert_close(second - first, torch.tensor(200.0))

    activation[:] = torch.tensor([[0.0, 1.0]])
    install_candidate_density(model, state)
    model._candidate_density_view(position)
    assert len(builds) == 2

    from dreamplace.PlaceObj import PlaceObj

    model.op_collections = SimpleNamespace()
    model._virtual_cell_density_view = None
    model.data_collections.node_size_x.mul_(2)
    PlaceObj.refresh_after_geometry_change(model)
    model._candidate_density_view(position)
    assert len(builds) == 3
    torch.testing.assert_close(builds[-1]["node_size_x"], torch.tensor([4.0, 4.0]))
