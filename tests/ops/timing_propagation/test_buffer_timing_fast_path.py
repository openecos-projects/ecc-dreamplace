import pytest
import torch
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import (
    buffer_timing_fast_path,
)
from torch.utils import deterministic


@pytest.mark.parametrize("initial_fill", [False, True])
@pytest.mark.parametrize("exception", [False, True])
def test_buffer_scope_preserves_determinism_and_restores_nested_fill_state(initial_fill, exception):
    old_fill = deterministic.fill_uninitialized_memory
    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        deterministic.fill_uninitialized_memory = initial_fill
        torch.use_deterministic_algorithms(mode=True)

        def exercise_scope():
            with buffer_timing_fast_path():
                assert not deterministic.fill_uninitialized_memory
                assert torch.are_deterministic_algorithms_enabled()
                with buffer_timing_fast_path():
                    assert not deterministic.fill_uninitialized_memory
                assert not deterministic.fill_uninitialized_memory
                if exception:
                    raise RuntimeError("scope exit")

        if exception:
            with pytest.raises(RuntimeError, match="scope exit"):
                exercise_scope()
        else:
            exercise_scope()
        assert deterministic.fill_uninitialized_memory == initial_fill
    finally:
        deterministic.fill_uninitialized_memory = old_fill
        torch.use_deterministic_algorithms(mode=old_algorithms, warn_only=old_warn_only)
