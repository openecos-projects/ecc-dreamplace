import pytest
import torch

from dreamplace.ops.timing_propagation import timing_propagation


def _native_module():
    module = timing_propagation._tp_cpp
    assert module is not None
    assert hasattr(module, "SetupCriticalPathExtractor")
    return module


def _fixture():
    module = _native_module()
    flat_arcs = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0, 0],
            [2, 3, 0, 1, -1, 1, 1],
        ],
        dtype=torch.int32,
    )
    extractor = module.SetupCriticalPathExtractor(
        flat_arcs,
        torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, -1, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        7,
    )
    inputs = {
        "endpoint_pins": torch.tensor([3], dtype=torch.int32),
        "endpoint_test_ids": torch.tensor([0], dtype=torch.int32),
        "endpoint_rise_slack": torch.tensor([-5.0]),
        "endpoint_fall_slack": torch.tensor([-2.0]),
        "pin_rise_aat": torch.tensor([10.0, 15.0, 18.0, 42.0]),
        "pin_fall_aat": torch.tensor([20.0, 27.0, 31.0, 31.0]),
        "pin_net_delay_rise": torch.tensor([0.0, 0.0, 3.0, 0.0]),
        "pin_net_delay_fall": torch.tensor([0.0, 0.0, 4.0, 0.0]),
        "cell_delay_rr": torch.tensor([5.0, 0.0]),
        "cell_delay_fr": torch.tensor([0.0, 11.0]),
        "cell_delay_rf": torch.tensor([0.0, 13.0]),
        "cell_delay_ff": torch.tensor([7.0, 0.0]),
    }
    return extractor, inputs


def _extract(extractor, inputs, global_k):
    return extractor.extract(
        inputs["endpoint_pins"],
        inputs["endpoint_test_ids"],
        inputs["endpoint_rise_slack"],
        inputs["endpoint_fall_slack"],
        inputs["pin_rise_aat"],
        inputs["pin_fall_aat"],
        inputs["pin_net_delay_rise"],
        inputs["pin_net_delay_fall"],
        inputs["cell_delay_rr"],
        inputs["cell_delay_fr"],
        inputs["cell_delay_rf"],
        inputs["cell_delay_ff"],
        global_k,
        32,
        1.0e-4,
    )


def test_global_k_selects_endpoint_transition_states():
    extractor, inputs = _fixture()
    worst = _extract(extractor, inputs, 1)
    assert worst.failing_state_count == 2
    assert worst.selected_state_count == 1
    assert worst.endpoint_pins.tolist() == [3]
    assert worst.endpoint_transitions.tolist() == [0]
    assert worst.endpoint_slacks.tolist() == pytest.approx([-5.0])

    all_failing = _extract(extractor, inputs, 0)
    assert all_failing.selected_state_count == 2
    assert all_failing.endpoint_transitions.tolist() == [0, 1]

    oversized = _extract(extractor, inputs, 100)
    assert oversized.selected_state_count == 2


def test_negative_unate_backtrace_recovers_transition_sequence():
    extractor, inputs = _fixture()
    batch = _extract(extractor, inputs, 1)
    assert batch.valid_path_count == 1
    assert batch.path_offsets.tolist() == [0, 4]
    assert batch.path_pins.tolist() == [0, 1, 2, 3]
    assert batch.path_transitions.tolist() == [1, 1, 1, 0]
    assert batch.path_arc_ids.tolist() == [0, -1, 1]
    assert batch.path_valid.tolist() == [True]
    assert batch.max_residual_ps.tolist() == pytest.approx([0.0])
    assert batch.topology_epoch == 7


def test_extractor_is_deterministic_for_repeated_calls():
    extractor, inputs = _fixture()
    first = _extract(extractor, inputs, 0)
    second = _extract(extractor, inputs, 0)
    for name in (
        "path_offsets",
        "path_pins",
        "path_transitions",
        "path_arc_ids",
        "endpoint_pins",
        "endpoint_transitions",
        "endpoint_slacks",
        "path_valid",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_declared_start_continues_through_matching_async_predecessor():
    module = _native_module()
    extractor = module.SetupCriticalPathExtractor(
        torch.tensor(
            [
                [0, 1, 0, 0, 1, 0, 0],
                [1, 2, 0, 1, 1, 0, 1],
            ],
            dtype=torch.int32,
        ),
        torch.tensor([0, 0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        11,
    )
    batch = extractor.extract(
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0, 3.0, 6.0]),
        torch.tensor([1.0, 3.0, 6.0]),
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([2.0, 3.0]),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([2.0, 3.0]),
        0,
        8,
        1.0e-4,
    )

    assert batch.path_valid.tolist() == [True]
    assert batch.path_pins.tolist() == [0, 1, 2]
    assert batch.path_transitions.tolist() == [0, 0, 0]


def test_declared_start_stops_when_predecessor_does_not_explain_launch_aat():
    module = _native_module()
    extractor = module.SetupCriticalPathExtractor(
        torch.tensor([[0, 1, 0, 0, 1, 0, 0]], dtype=torch.int32),
        torch.tensor([0, 0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        12,
    )
    batch = extractor.extract(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0, 20.0]),
        torch.tensor([1.0, 20.0]),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([2.0]),
        torch.zeros(1),
        torch.zeros(1),
        torch.tensor([2.0]),
        0,
        8,
        1.0e-4,
    )

    assert batch.path_valid.tolist() == [True]
    assert batch.path_pins.tolist() == [1]


def test_invalid_endpoint_shape_fails_closed():
    extractor, inputs = _fixture()
    inputs["endpoint_fall_slack"] = torch.tensor([-2.0, -1.0])
    with pytest.raises(RuntimeError, match="endpoint"):
        _extract(extractor, inputs, 0)


def test_duplicate_endpoint_pin_preserves_setup_test_multiplicity():
    extractor, inputs = _fixture()
    inputs["endpoint_pins"] = torch.tensor([3, 3], dtype=torch.int32)
    inputs["endpoint_test_ids"] = torch.tensor([10, 11], dtype=torch.int32)
    inputs["endpoint_rise_slack"] = torch.tensor([-5.0, -4.0])
    inputs["endpoint_fall_slack"] = torch.tensor([1.0, 1.0])

    batch = _extract(extractor, inputs, 0)

    assert batch.failing_state_count == 2
    assert batch.endpoint_pins.tolist() == [3, 3]
    assert batch.endpoint_test_ids.tolist() == [10, 11]
    assert batch.endpoint_transitions.tolist() == [0, 0]
    assert batch.path_offsets.tolist() == [0, 4, 8]
