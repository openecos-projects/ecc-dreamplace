import pytest
import torch

from dreamplace.ops.timing_net_weighting import timing_net_weighting_cpp


def _native_module():
    module = timing_net_weighting_cpp
    assert hasattr(module, "accumulate_pin2pin_pairs")
    return module


def test_packed_pair_accumulation_matches_efficient_tdp_formula():
    module = _native_module()
    result = module.accumulate_pin2pin_pairs(
        torch.tensor([0, 3, 6], dtype=torch.int64),
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        torch.tensor([-5.0, -2.5]),
        torch.tensor([True, True]),
        torch.tensor([10, 10, 20], dtype=torch.int64),
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty((0,), dtype=torch.float32),
        -5.0,
        10.0,
        50.0,
        0.2,
    )
    assert result.raw_pair_count == 2
    assert result.unique_pair_count == 1
    assert result.updated_pair_count == 1
    assert result.pair_keys.tolist() == [[1, 2]]
    assert result.pair_weights.tolist() == pytest.approx([10.1])


def test_existing_pairs_are_merged_in_canonical_order():
    module = _native_module()
    result = module.accumulate_pin2pin_pairs(
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([4, 5], dtype=torch.int64),
        torch.tensor([-4.0]),
        torch.tensor([True]),
        torch.tensor([40, 50, 60, 70, 80, 90], dtype=torch.int64),
        torch.tensor([[1, 2], [4, 5]], dtype=torch.int64),
        torch.tensor([12.0, 49.9]),
        -4.0,
        10.0,
        50.0,
        0.2,
    )
    assert result.pair_keys.tolist() == [[1, 2], [4, 5]]
    assert result.pair_weights.tolist() == pytest.approx([12.0, 50.0])
    assert result.clamped_pair_count == 1


def test_invalid_offsets_fail_closed():
    module = _native_module()
    with pytest.raises(RuntimeError, match="offset"):
        module.accumulate_pin2pin_pairs(
            torch.tensor([0, 3], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([-1.0]),
            torch.tensor([True]),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.empty((0, 2), dtype=torch.int64),
            torch.empty((0,), dtype=torch.float32),
            -1.0,
            10.0,
            50.0,
            0.2,
        )
