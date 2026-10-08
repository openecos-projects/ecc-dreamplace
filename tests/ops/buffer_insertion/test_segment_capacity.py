import pytest
import torch

from dreamplace.ops.buffer_insertion.segment_capacity import (
    build_segment_capacity_controller,
    capacity_grid_from_base_map,
    close_committed_def_fidelity,
)
from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)


def _line_state(*, initial_z=1.0, y=2, coordinates_dbu=None):
    net = {
        "net_id": 7,
        "driver_pin_id": 0,
        "coordinates": {0: (1, y), 1: (7, y)},
        "rc_tree": {
            "root_node_id": 0,
            "children_by_node": {0: [1], 1: []},
            "edge_rc": {(0, 1): {"r": 1.0, "c": 1.0}},
        },
    }
    if coordinates_dbu is not None:
        net["coordinates_dbu"] = coordinates_dbu
    return build_segment_count_state(
        [net],
        buffer_main_type_index=1,
        legal_buffer_count=2,
        max_repeater_count=3,
        initial_z=initial_z,
        initial_bsu_index=1.0,
        dtype=torch.float64,
    )


def _grid(*, capacity=16.0):
    return capacity_grid_from_base_map(
        torch.full((8, 8), float(capacity), dtype=torch.float64),
        xl=0.0,
        yl=0.0,
        bin_size_x=1.0,
        bin_size_y=1.0,
        factor=4,
    )


def _controller(*, capacity=16.0, initial_z=1.0, coordinates_dbu=None):
    state = _line_state(initial_z=initial_z, coordinates_dbu=coordinates_dbu)
    return build_segment_capacity_controller(
        state,
        grid=_grid(capacity=capacity),
        row_height=1.0,
        buffer_width=1.0,
        buffer_height=1.0,
        fixed_cell_id=21,
        fixed_master_name="BUF_X1",
    )


def test_swept_geometry_distributes_horizontal_segment_by_exact_overlap():
    controller = _controller()

    assert controller.grid.num_bins_x == 2
    assert controller.grid.num_bins_y == 2
    assert controller.geometry_summary["segment_bin_nnz"] == 2
    assert controller.normalized_weight.detach().cpu().tolist() == pytest.approx(
        [0.5, 0.5]
    )

    area = controller.added_area(controller.state.z_value())
    assert float(area.sum().item()) == pytest.approx(1.0)
    assert area.detach().cpu().tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0])


def test_core_edge_clipping_preserves_full_buffer_area_in_legal_bins():
    state = _line_state(initial_z=1.0, y=0)
    controller = build_segment_capacity_controller(
        state,
        grid=_grid(capacity=16.0),
        row_height=1.0,
        buffer_width=1.0,
        buffer_height=1.0,
        fixed_cell_id=21,
        fixed_master_name="BUF_X1",
    )

    area = controller.added_area(controller.state.z_value())

    assert float(area.sum().item()) == pytest.approx(1.0)
    assert controller.geometry_summary["core_clipped_segment_count"] == 1
    assert controller.geometry_summary["swept_footprint_outside_core_area_total"] == pytest.approx(3.0)
    assert controller.geometry_summary["weight_sum_min"] == pytest.approx(1.0)
    assert controller.geometry_summary["weight_sum_max"] == pytest.approx(1.0)


def test_wholly_outside_core_footprint_is_rejected():
    state = _line_state(initial_z=1.0, y=-2)

    with pytest.raises(ValueError, match="no positive in-core area"):
        build_segment_capacity_controller(
            state,
            grid=_grid(capacity=16.0),
            row_height=1.0,
            buffer_width=1.0,
            buffer_height=1.0,
            fixed_cell_id=21,
            fixed_master_name="BUF_X1",
        )


def test_area_operator_has_unit_gradient_and_no_integer_switch():
    controller = _controller(initial_z=0.37)
    z = controller.state.z_param
    area_sum = controller.added_area(controller.state.z_value()).sum()
    area_sum.backward()

    assert float(z.grad.item()) == pytest.approx(1.0)

    with torch.no_grad():
        z.fill_(1.37)
    controller.state.z_param.grad = None
    controller.added_area(controller.state.z_value()).sum().backward()
    assert float(z.grad.item()) == pytest.approx(1.0)


def test_phr_constraint_penalizes_over_capacity_and_updates_detached_dual():
    controller = _controller(capacity=1.0, initial_z=1.0)
    timing_loss = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    loss, metrics = controller.compose_objective(timing_loss)
    loss.backward()

    assert metrics["segment_capacity_max_violation"] > 0.0
    assert controller.state.z_param.grad is not None
    assert torch.isfinite(controller.state.z_param.grad).all()

    before = controller.dual_state.clone()
    after_metrics = controller.update_dual_after_step(
        0,
        objective_metrics={
            "timing_loss": torch.tensor(10.0, dtype=torch.float64),
            "tns": torch.tensor(-10.0, dtype=torch.float64),
            "wns": torch.tensor(-1.0, dtype=torch.float64),
        },
    )
    assert bool((controller.dual_state >= before).all())
    assert after_metrics["segment_capacity_dual_max"] > 0.0
    assert controller.dual_state.requires_grad is False
    assert after_metrics["timing_loss_pre_step"] == pytest.approx(10.0)
    assert after_metrics["tns_pre_step"] == pytest.approx(-10.0)
    assert after_metrics["wns_pre_step"] == pytest.approx(-1.0)
    assert after_metrics["z_value_distribution"]["count"] == 1


def test_projection_fidelity_uses_openroad_origin_coordinate_semantics():
    controller = _controller(initial_z=1.0)
    fidelity = controller.close_projection_fidelity(
        [
            {
                "candidate_location_x_dbu": 2.0,
                "candidate_location_y_dbu": 2.0,
            }
        ]
    )

    assert fidelity["projected_action_count"] == 1
    assert sum(fidelity["projected_area_by_bin"]) == pytest.approx(1.0)
    assert fidelity["projected_outside_core_area"] == pytest.approx(0.0)


def test_committed_def_fidelity_uses_final_def_component_origins(tmp_path):
    controller = _controller(capacity=100.0, initial_z=1.0)
    controller.close_projection_fidelity(
        [{"candidate_location_x_dbu": 2.0, "candidate_location_y_dbu": 2.0}]
    )
    paths = controller.write_artifacts(tmp_path)
    committed_def = tmp_path / "committed.def"
    committed_def.write_text(
        "COMPONENTS 1 ;\n"
        "- inserted_buffer BUF_X1 + PLACED ( 2 2 ) N ;\n"
        "END COMPONENTS\n",
        encoding="utf-8",
    )

    fidelity = close_committed_def_fidelity(
        projection_path=paths["segment_capacity_projection_fidelity.json"],
        config_path=paths["segment_capacity_config.json"],
        committed_def_path=committed_def,
        committed_instance_names=["inserted_buffer"],
    )

    assert fidelity["status"] == "pass"
    assert fidelity["committed_status"] == "pass"
    assert fidelity["committed_instance_count"] == 1
    assert fidelity["committed_exact_action_count_match"]
    assert sum(fidelity["committed_area_by_bin"]) == pytest.approx(1.0)
    assert fidelity["committed_outside_core_area"] == pytest.approx(0.0)
    assert fidelity["committed_max_overshoot_in_buffer_quanta"] == pytest.approx(0.0)


def test_committed_def_fidelity_rejects_missing_added_buffer(tmp_path):
    controller = _controller(capacity=100.0, initial_z=1.0)
    controller.close_projection_fidelity(
        [{"candidate_location_x_dbu": 2.0, "candidate_location_y_dbu": 2.0}]
    )
    paths = controller.write_artifacts(tmp_path)
    committed_def = tmp_path / "committed.def"
    committed_def.write_text("COMPONENTS 0 ;\nEND COMPONENTS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing added buffer placement"):
        close_committed_def_fidelity(
            projection_path=paths["segment_capacity_projection_fidelity.json"],
            config_path=paths["segment_capacity_config.json"],
            committed_def_path=committed_def,
            committed_instance_names=["inserted_buffer"],
        )


def test_zero_capacity_crossing_fails_instead_of_dropping_segment():
    state = _line_state(initial_z=1.0)
    capacity = torch.full((8, 8), 16.0, dtype=torch.float64)
    capacity[:4, :4] = 0.0
    grid = capacity_grid_from_base_map(
        capacity,
        xl=0.0,
        yl=0.0,
        bin_size_x=1.0,
        bin_size_y=1.0,
        factor=4,
    )

    with pytest.raises(ValueError, match="zero-capacity crossings"):
        build_segment_capacity_controller(
            state,
            grid=grid,
            row_height=1.0,
            buffer_width=1.0,
            buffer_height=1.0,
            fixed_cell_id=21,
            fixed_master_name="BUF_X1",
        )


def test_segment_state_rows_use_explicit_dbu_coordinates_for_resource_geometry():
    state = _line_state(
        coordinates_dbu={0: (1000, 2000), 1: (7000, 2000)},
    )

    assert state.segment_rows[0]["parent_x_dbu"] == 1000
    assert state.segment_rows[0]["parent_y_dbu"] == 2000
    assert state.segment_rows[0]["child_x_dbu"] == 7000
    assert state.segment_rows[0]["child_y_dbu"] == 2000
