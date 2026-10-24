import pytest
import torch

from sparse_fixtures import (
    inputs,
    sparse_view,
    production_timing,
    run_production,
)


def test_sparse_critical_paths_keep_state_and_physical_identity(native_device):
    timing = production_timing(native_device)
    timing.request_critical_path_snapshot()
    run_production(timing, inputs(native_device))
    paths = timing.extract_setup_critical_paths()
    assert paths.valid_path_count > 0
    assert paths.invalid_path_count == 0
    assert paths.path_pins.tolist() == [1, 2, 3] * paths.valid_path_count
    assert paths.path_state_ids.tolist() == [1, 3, 5] * paths.valid_path_count
    assert paths.path_arc_ids.tolist() == [-1, 0] * paths.valid_path_count
    assert paths.endpoint_pins.tolist() == [3] * paths.valid_path_count
    assert paths.endpoint_state_ids.tolist() == [5] * paths.valid_path_count
    assert torch.all(paths.max_residual_ps <= 1e-3)


def test_sparse_paths_reject_master_change_after_snapshot():
    timing = production_timing("cpu")
    timing.request_critical_path_snapshot()
    run_production(timing, inputs("cpu"))
    timing.flat_inst_arcs_by_level.add_(0)
    with pytest.raises(RuntimeError, match="masters are stale"):
        timing.extract_setup_critical_paths()


def test_sparse_pruner_maps_legal_states_to_existing_physical_consumers(native_device):
    timing = production_timing(native_device)
    timing.critical_endpoint_pruning_mode = "dynamic"
    run_production(timing, inputs(native_device))
    assert timing.last_active_endpoint_ids == [3]
    assert timing.last_traversal_pruning_result.kept_flat_arc_indices.tolist() == [0]
    assert timing.last_traversal_pruning_result.kept_state_indices.tolist() == list(
        range(6)
    )
    incidence = timing.compute_active_endpoint_incidence([3])
    assert incidence.pin_endpoint_incidence_count.tolist() == [1, 1, 1, 1]
    assert incidence.arc_endpoint_incidence_count.tolist() == [1]
    assert incidence.active_endpoint_count == 1


def test_sparse_pruned_arrivals_keep_selected_gradients_and_full_slew(native_device):
    timing = production_timing(native_device)
    timing.sparse_timing_view.arrays["check_index"][2:, 6] = 0
    timing.critical_endpoint_pruning_mode = "dynamic"
    run_production(timing, inputs(native_device))
    values = inputs(native_device)
    _, pruned, _, _ = run_production(timing, values)
    pruned_lanes = timing._sparse_runtime.last_result.lanes
    pruned_gradient = torch.autograd.grad(pruned, values, retain_graph=True)
    assert not timing.last_profile_payload["full_arrival_refresh"]
    timing.critical_endpoint_pruning_mode = "off"
    _, full, _, _ = run_production(timing, values)
    full_lanes = timing._sparse_runtime.last_result.lanes
    full_gradient = torch.autograd.grad(full, values)
    torch.testing.assert_close(pruned, full)
    torch.testing.assert_close(pruned_lanes[2:], full_lanes[2:])
    torch.testing.assert_close(pruned_gradient, full_gradient)
    timing.critical_endpoint_pruning_mode = "dynamic"
    timing.request_critical_path_snapshot()
    run_production(timing, inputs(native_device))
    assert timing.last_profile_payload["full_arrival_refresh"]


def test_sparse_view_replacement_rebuilds_schedule():
    timing = production_timing("cpu")
    run_production(timing, inputs("cpu"))
    previous = timing._sparse_runtime
    timing.sparse_timing_view, _ = sparse_view(cell=True, false_path=True)
    run_production(timing, inputs("cpu"))
    assert timing._sparse_runtime is not previous
    assert timing._sparse_runtime.schedule.legal.tolist() == [True, True, False, False]


@pytest.mark.parametrize("scope", ["setup_only", "setup_plus_recovery"])
def test_sparse_metric_scope_and_path_scope_use_native_check_classes(
    native_device, scope
):
    timing = production_timing(native_device)
    timing.timing_metric_scope = scope
    timing.endpoints_timing_check_arcs = torch.tensor(
        [[0, 3, 0, 0, 1, 0, 1, 0], [0, 3, 0, 0, 1, 0, 3, 0]],
        device=native_device,
        dtype=torch.long,
    )
    timing.sparse_timing_view.arrays["check_index"][:, 2] = [0, 0, 1, 1]
    timing.request_critical_path_snapshot()
    run_production(timing, inputs(native_device))
    assert timing._sparse_runtime.schedule.legal.tolist() == [
        True,
        True,
        scope == "setup_plus_recovery",
        scope == "setup_plus_recovery",
    ]
    paths = timing.extract_setup_critical_paths()
    assert paths.valid_path_count == 0
    timing.request_critical_path_snapshot()
    run_production(timing, inputs(native_device))
    paths = timing.extract_pin2pin_critical_paths()
    assert paths.valid_path_count == (2 if scope == "setup_plus_recovery" else 0)


def test_sparse_report_uses_legal_state_qualification():
    from dreamplace.ops.timing_propagation.endpoint_qualification import (
        reporting_endpoint_validity,
    )

    timing = production_timing("cpu")
    run_production(timing, inputs("cpu"))
    # Legacy path-specific qualification cannot represent the endpoint pair.
    fallback = torch.zeros((1, 2), dtype=torch.bool)
    assert reporting_endpoint_validity(timing, fallback).tolist() == [[True, True]]


def test_sparse_full_rat_and_path_report_keep_matching_launch_states(native_device):
    from dreamplace.ops.timing_propagation.timing_path_report import (
        build_critical_path_rows,
    )

    timing = production_timing(native_device)
    timing.production_fast_loop = False
    timing.sparse_timing_view.arrays["check_values"][:2, 0] = 1.0
    timing.request_critical_path_snapshot()
    run_production(timing, inputs(native_device))
    assert not timing.last_fast_loop_skipped_full_rat
    # The low-arrival domain is the critical one after its tighter requirement.
    paths = timing.extract_setup_critical_paths(global_k=1)
    assert paths.path_state_ids.tolist() == [0, 2, 4]
    slack = timing.get_pin_slack()
    assert torch.all(slack < 0)
    rows = build_critical_path_rows(
        batch=paths,
        timing_op=timing,
        pin_names=["a", "b", "in", "out"],
        flat_inst_arcs_by_level=timing.flat_inst_arcs_by_level,
    )
    assert all(
        abs(row["incoming_residual_ps"]) < 1e-3 for row in rows if row["step_index"]
    )
