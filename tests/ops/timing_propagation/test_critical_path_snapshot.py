import torch

from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def _shell():
    timing = object.__new__(TimingPropagation)
    timing._critical_path_snapshot_requested = False
    timing._critical_path_snapshot = None
    timing._critical_path_constraint_state = None
    timing._setup_critical_path_extractor = None
    timing._setup_critical_path_extractor_epoch = None
    timing.last_critical_path_extraction_stats = None
    timing.timing_aggregation_mode = "hard"
    timing.critical_path_topology_epoch = 3
    timing.num_pins = 3
    timing.end_points = torch.tensor([2], dtype=torch.int32)
    return timing


def _capture_setup_and_recovery_snapshot():
    timing = _shell()
    timing.num_pins = 4
    timing.end_points = torch.tensor([2, 3], dtype=torch.int32)
    timing.endpoints_timing_check_arcs = torch.tensor(
        [
            [0, 2, 0, 0, 1, 0, 3, 0],
            [0, 3, 0, 0, 1, 0, 1, 0],
        ],
        dtype=torch.int32,
    )
    timing.flat_inst_arcs_by_level = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0],
            [2, 3, 0, 1, -1, 1, 1],
        ],
        dtype=torch.int32,
    )
    timing.pin_pred_start = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32)
    timing.pin_pred_pin = torch.tensor([0, 1, 2], dtype=torch.int32)
    timing.pin_pred_arc_id = torch.tensor([0, -1, 1], dtype=torch.int32)
    timing.start_points = torch.tensor([0], dtype=torch.int32)
    timing.request_critical_path_snapshot()
    timing._critical_path_constraint_state = {
        "endpoint_pins": torch.tensor([3], dtype=torch.int32),
        "test_ids": torch.tensor([7], dtype=torch.int64),
        "rise_rat": torch.tensor([37.0]),
        "fall_rat": torch.tensor([33.0]),
    }
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.tensor([10.0, 15.0, 18.0, 42.0]),
        pin_fAAT=torch.tensor([20.0, 27.0, 31.0, 31.0]),
        pin_rRAT=torch.tensor([0.0, 0.0, 12.0, 0.0]),
        pin_fRAT=torch.tensor([0.0, 0.0, 24.0, 0.0]),
        pin_net_delay_rise=torch.tensor([0.0, 0.0, 3.0, 0.0]),
        pin_net_delay_fall=torch.tensor([0.0, 0.0, 4.0, 0.0]),
        cell_arc_rr_delays=torch.tensor([5.0, 0.0]),
        cell_arc_fr_delays=torch.tensor([0.0, 11.0]),
        cell_arc_rf_delays=torch.tensor([0.0, 13.0]),
        cell_arc_ff_delays=torch.tensor([7.0, 0.0]),
    )
    return timing


def test_snapshot_request_captures_one_packed_cpu_transfer():
    timing = _shell()
    timing.request_critical_path_snapshot()
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.tensor([0.0, 1.0, 2.0]),
        pin_fAAT=torch.tensor([0.0, 2.0, 3.0]),
        pin_rRAT=torch.tensor([10.0, 10.0, 1.0]),
        pin_fRAT=torch.tensor([10.0, 10.0, 2.0]),
        pin_net_delay_rise=torch.tensor([0.0, 0.5, 0.5]),
        pin_net_delay_fall=torch.tensor([0.0, 0.6, 0.6]),
        cell_arc_rr_delays=torch.tensor([1.0, 2.0]),
        cell_arc_fr_delays=torch.tensor([3.0, 4.0]),
        cell_arc_rf_delays=torch.tensor([5.0, 6.0]),
        cell_arc_ff_delays=torch.tensor([7.0, 8.0]),
    )
    snapshot = timing.consume_critical_path_snapshot()
    assert snapshot["packed"].device.type == "cpu"
    assert snapshot["endpoint_pins"].tolist() == [2]
    assert snapshot["endpoint_rise_slack"].tolist() == [-1.0]
    assert snapshot["endpoint_fall_slack"].tolist() == [-1.0]
    assert snapshot["recovery_endpoint_pins"].numel() == 0
    assert snapshot["cell_delay_ff"].tolist() == [7.0, 8.0]
    assert snapshot["topology_epoch"] == 3
    assert timing._critical_path_snapshot is None
    assert not timing._critical_path_snapshot_requested


def test_snapshot_preserves_setup_constraint_arc_multiplicity():
    timing = _shell()
    timing.end_points = torch.tensor([1, 2], dtype=torch.int32)
    timing.request_critical_path_snapshot()
    timing._critical_path_constraint_state = {
        "endpoint_pins": torch.tensor([2, 2], dtype=torch.int32),
        "test_ids": torch.tensor([7, 8], dtype=torch.int64),
        "rise_rat": torch.tensor([1.0, 0.5]),
        "fall_rat": torch.tensor([2.0, 1.5]),
    }
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.tensor([0.0, 1.0, 2.0]),
        pin_fAAT=torch.tensor([0.0, 2.0, 3.0]),
        pin_rRAT=torch.tensor([10.0, 5.0, 0.5]),
        pin_fRAT=torch.tensor([10.0, 6.0, 1.5]),
        pin_net_delay_rise=torch.zeros(3),
        pin_net_delay_fall=torch.zeros(3),
        cell_arc_rr_delays=torch.ones(1),
        cell_arc_fr_delays=torch.ones(1),
        cell_arc_rf_delays=torch.ones(1),
        cell_arc_ff_delays=torch.ones(1),
    )

    snapshot = timing.consume_critical_path_snapshot()

    assert snapshot["endpoint_pins"].tolist() == [2, 2, 1]
    assert snapshot["endpoint_test_ids"].tolist() == [7, 8, -1]
    assert snapshot["endpoint_rise_slack"].tolist() == [-1.0, -1.5, 4.0]
    assert snapshot["endpoint_fall_slack"].tolist() == [-1.0, -1.5, 4.0]


def test_setup_snapshot_excludes_async_check_endpoints_from_fallback():
    timing = _shell()
    timing.end_points = torch.tensor([1, 2], dtype=torch.int32)
    timing.endpoints_timing_check_arcs = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 2, 0],
            [0, 2, 0, 0, 1, 0, 1, 0],
        ],
        dtype=torch.int32,
    )
    timing.request_critical_path_snapshot()
    timing._critical_path_constraint_state = {
        "endpoint_pins": torch.tensor([2], dtype=torch.int32),
        "test_ids": torch.tensor([0], dtype=torch.int64),
        "rise_rat": torch.tensor([1.0]),
        "fall_rat": torch.tensor([2.0]),
    }
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.tensor([0.0, 1.0, 2.0]),
        pin_fAAT=torch.tensor([0.0, 2.0, 3.0]),
        pin_rRAT=torch.tensor([10.0, 5.0, 1.0]),
        pin_fRAT=torch.tensor([10.0, 6.0, 2.0]),
        pin_net_delay_rise=torch.zeros(3),
        pin_net_delay_fall=torch.zeros(3),
        cell_arc_rr_delays=torch.ones(1),
        cell_arc_fr_delays=torch.ones(1),
        cell_arc_rf_delays=torch.ones(1),
        cell_arc_ff_delays=torch.ones(1),
    )

    snapshot = timing.consume_critical_path_snapshot()

    assert snapshot["endpoint_pins"].tolist() == [2]
    assert snapshot["endpoint_test_ids"].tolist() == [0]
    assert snapshot["recovery_endpoint_pins"].numel() == 0


def test_recovery_endpoint_is_kept_in_separate_max_check_domain():
    timing = _shell()
    timing.end_points = torch.tensor([0, 1, 2], dtype=torch.int32)
    timing.endpoints_timing_check_arcs = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0, 4, 0],
            [0, 1, 0, 0, 1, 0, 3, 0],
            [0, 2, 0, 0, 1, 0, 1, 0],
        ],
        dtype=torch.int32,
    )
    timing.request_critical_path_snapshot()
    timing._critical_path_constraint_state = {
        "endpoint_pins": torch.tensor([2], dtype=torch.int32),
        "test_ids": torch.tensor([0], dtype=torch.int64),
        "rise_rat": torch.tensor([1.0]),
        "fall_rat": torch.tensor([2.0]),
    }
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.tensor([0.0, 1.0, 2.0]),
        pin_fAAT=torch.tensor([0.0, 2.0, 3.0]),
        pin_rRAT=torch.tensor([-4.0, -2.0, 1.0]),
        pin_fRAT=torch.tensor([-3.0, 1.0, 2.0]),
        pin_net_delay_rise=torch.zeros(3),
        pin_net_delay_fall=torch.zeros(3),
        cell_arc_rr_delays=torch.ones(1),
        cell_arc_fr_delays=torch.ones(1),
        cell_arc_rf_delays=torch.ones(1),
        cell_arc_ff_delays=torch.ones(1),
    )

    snapshot = timing.consume_critical_path_snapshot()

    assert snapshot["endpoint_pins"].tolist() == [2]
    assert snapshot["recovery_endpoint_pins"].tolist() == [1]
    assert snapshot["recovery_endpoint_rise_slack"].tolist() == [-3.0]
    assert torch.isinf(snapshot["recovery_endpoint_fall_slack"]).all()


def test_timing_metric_scope_defaults_to_setup_plus_recovery_and_supports_setup_only():
    timing = _shell()
    timing.end_points = torch.tensor([1, 2, 3], dtype=torch.int32)
    timing.endpoints_timing_check_arcs = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 3, 0],  # recovery-only endpoint
            [0, 2, 0, 0, 1, 0, 1, 0],  # setup endpoint
        ],
        dtype=torch.int32,
    )
    timing.endpoints_constraint_arcs = torch.tensor(
        [[0, 2, 0, 0, 1, 0, 1, 0]], dtype=torch.int32
    )

    timing.timing_metric_scope = "setup_plus_recovery"
    assert timing._timing_metric_endpoint_mask(timing.end_points.device).tolist() == [
        True,
        True,
        True,
    ]

    timing.timing_metric_scope = "setup_only"
    assert timing._timing_metric_endpoint_mask(timing.end_points.device).tolist() == [
        False,
        True,
        True,
    ]


def test_pin2pin_extraction_combines_setup_and_worst_recovery_transition():
    setup_timing = _capture_setup_and_recovery_snapshot()
    setup_batch = setup_timing.extract_setup_critical_paths(global_k=0)

    assert setup_batch.failing_state_count == 1
    assert setup_batch.endpoint_pins.tolist() == [3]
    assert setup_batch.endpoint_test_ids.tolist() == [7]
    assert setup_batch.endpoint_transitions.tolist() == [0]
    assert setup_timing.last_critical_path_extraction_stats["state_domain"] == (
        "setup_only"
    )
    assert setup_timing.last_critical_path_extraction_stats[
        "recovery_endpoint_count"
    ] == 0

    pin2pin_timing = _capture_setup_and_recovery_snapshot()
    pin2pin_batch = pin2pin_timing.extract_pin2pin_critical_paths(global_k=2)

    assert pin2pin_batch.failing_state_count == 2
    assert pin2pin_batch.selected_state_count == 2
    assert pin2pin_batch.valid_path_count == 2
    assert pin2pin_batch.endpoint_pins.tolist() == [2, 3]
    assert pin2pin_batch.endpoint_test_ids.tolist() == [-1, 7]
    assert pin2pin_batch.endpoint_transitions.tolist() == [1, 0]
    assert pin2pin_batch.endpoint_slacks.tolist() == [-7.0, -5.0]
    assert pin2pin_timing.last_critical_path_extraction_stats["state_domain"] == (
        "setup_plus_recovery"
    )
    assert pin2pin_timing.last_critical_path_extraction_stats[
        "recovery_endpoint_count"
    ] == 1


def test_clear_releases_pending_snapshot():
    timing = _shell()
    timing.request_critical_path_snapshot()
    timing.clear_critical_path_snapshot_request()
    assert timing._critical_path_snapshot is None
    assert not timing._critical_path_snapshot_requested


def test_snapshot_is_not_built_without_request():
    timing = _shell()
    timing._capture_critical_path_snapshot(
        pin_rAAT=torch.zeros(3),
        pin_fAAT=torch.zeros(3),
        pin_rRAT=torch.zeros(3),
        pin_fRAT=torch.zeros(3),
        pin_net_delay_rise=torch.zeros(3),
        pin_net_delay_fall=torch.zeros(3),
        cell_arc_rr_delays=None,
        cell_arc_fr_delays=None,
        cell_arc_rf_delays=None,
        cell_arc_ff_delays=None,
    )
    assert timing._critical_path_snapshot is None
